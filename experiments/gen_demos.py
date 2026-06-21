"""Sprint S3 (BC) — generate expert demonstrations from the search agent.

Distilling the S1/S2 ``LookaheadAgent`` into the ``MaskablePPO`` net (the next
step, ``train_bc.py``) needs a supervised dataset of
``(observation, chosen_flat_action, action_mask)`` tuples logged at every
*real* learner decision, using the EXACT frozen codec
(:func:`heat.ml.features.encode_observation`,
:func:`heat.ml.action_codec.encode_action_index`,
:func:`~heat.ml.action_codec.legal_action_mask`) so the data is byte-compatible
with the policy the trainer builds.

Why drive the engine directly (not ``HeatEnv``)
------------------------------------------------
The expert is a :class:`~heat.agents.search_agent.LookaheadAgent` -- a
``BaseAgent`` whose ``choose_gear``/``choose_cards`` *plan-once* over the live
``GameState`` and need the real ``Decision.legal`` set the engine hands in. We
therefore run :func:`heat.engine.driver.run_round_driver` ourselves (exactly the
loop :class:`heat.ml.env.HeatEnv` uses), route the learner seat through the
expert and every other seat through the rollout/opponent policy, and snapshot a
training tuple at each learner decision BEFORE applying the expert's action.

Matching the env's information set (critical correctness points)
----------------------------------------------------------------
1. **Only log decisions the BC net will actually face.** ``HeatEnv`` auto-
   resolves any learner decision whose legal-action mask has ``<= 1`` True entry
   (a forced gear, the empty-hand ``()`` CARDS play, etc.) and never surfaces it
   as an RL step (see ``HeatEnv._forced_action``). We replicate that: a tuple is
   logged only when ``mask.sum() > 1``. Degenerate decisions still get the
   expert's action *applied* (so the game advances identically) -- they are just
   not recorded, so the BC target distribution matches what the trained net sees.
2. **Action-target snapping matches ``env._decode_legal``'s value-multiset
   rule.** The design flags this as a BC risk. ``encode_action_index`` already
   collapses a CARDS play to its value-multiset index (the same collapse the env
   does), so the logged flat target is exactly the index ``decode_action`` /
   ``env._decode_legal`` would round-trip back to a legal play. We additionally
   *assert* (in :func:`_encode_target`) that the target index is set in the mask,
   i.e. the expert's chosen action is codec-legal -- the dataset can never carry
   an off-mask target.
3. **4p determinization belief (S2).** When generating 4p demos with a
   ``determinize_hidden`` expert, the expert queries search on the same
   pool-``hand+draw_pile`` / keep-discard info set the env exposes, so the
   targets are consistent with the env's real (hidden-info) information set. The
   default expert here is the S2 determinized agent in 4p and the plain (solo
   no-op) agent in solo.

Track-disjoint train/val split
-------------------------------
The design demands a *track-disjoint* split (not a row-shuffle): generalization
to the limit-1 corner is the whole point, and a row-shuffle would leak corners
from the same track into both splits. We partition the GENERATED track *seeds*
into train / val bands and tag every tuple with its source track seed, so a
downstream consumer can verify disjointness (``train_bc.py`` does).

Output
------
A compressed ``.npz`` with parallel arrays ``obs`` ``(N, OBS_DIM) float32``,
``action`` ``(N,) int64``, ``mask`` ``(N, ACTION_DIM) bool``, ``kind`` ``(N,)
int8`` (the ``DecisionKind`` value, for per-kind metrics), ``track_seed``
``(N,) int64``, and a ``split`` ``(N,)`` byte array (``b"train"``/``b"val"``).
Plus a small JSON sidecar with provenance (codec version, expert config, seeds).

Usage:
    python experiments/gen_demos.py --train-tracks 60 --val-tracks 20 --out data/bc_demos.npz
    python experiments/gen_demos.py --players 1 --train-tracks 80 --val-tracks 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.ml.action_codec import (
    _play_to_multiset,
    encode_action_index,
    legal_action_mask,
)
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.tracks.generator import TrackGenParams, generate_track


# Mirror the eval harness's held-out, limit-1-weighted generated distribution
# (experiments/eval_search.py) so the demos are drawn from the same corner mix
# the BC net is gated on. Train/val seed bands are disjoint sub-ranges of this.
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),  # weight toward tight (limit-1) corners
    laps=2,
)

#: Seed band for TRAIN tracks. Distinct from both the eval harness's held-out
#: band (900_000+, experiments/eval_search.py) and the VAL band below, so the
#: BC net is never trained on a track it is later validated or gated on.
_TRAIN_SEED_BASE = 100_000
#: Seed band for VAL tracks (track-disjoint from train; also disjoint from the
#: eval harness's 900_000+ gate band).
_VAL_SEED_BASE = 500_000


_KIND_TO_INT: dict[DecisionKind, int] = {
    DecisionKind.GEAR: 0,
    DecisionKind.CARDS: 1,
    DecisionKind.REACT: 2,
    DecisionKind.SLIPSTREAM: 3,
    DecisionKind.DISCARD: 4,
}


@dataclass
class _DemoBuffer:
    """Growable column store for the demonstration tuples."""

    obs: list[np.ndarray] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    kind: list[int] = field(default_factory=list)
    track_seed: list[int] = field(default_factory=list)
    split: list[str] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        mask: np.ndarray,
        kind: int,
        track_seed: int,
        split: str,
    ) -> None:
        self.obs.append(obs)
        self.action.append(action)
        self.mask.append(mask)
        self.kind.append(kind)
        self.track_seed.append(track_seed)
        self.split.append(split)

    def __len__(self) -> int:
        return len(self.action)


def _encode_target(decision: Decision, engine_action: object) -> int | None:
    """Encode the expert's chosen engine action to its frozen flat index.

    Uses :func:`heat.ml.action_codec.encode_action_index`, which collapses a
    CARDS play to its value-multiset index -- exactly the collapse the env's
    ``_decode_legal`` round-trips through. The returned index is the BC target.

    Returns ``None`` when the action has **no codec encoding**. The flat action
    space is a deliberate, behavior-covering SUBSET of the engine's true action
    space (most relevant for REACT: the codec's fixed 8-slot
    ``_REACT_TABLE`` does not enumerate every legal
    ``(cooldown, boost, adrenaline_*)`` combination, e.g. ``cooldown_count=1`` +
    ``adrenaline_speed``). A ``MaskablePPO`` policy can only ever *emit* an
    in-table action, so an expert choice outside the table is not a learnable
    target -- it is dropped (and counted) rather than forcing a wrong target. The
    env never hits this because the policy only emits codec actions; the
    scripted/search expert can pick any legal engine action.
    """
    try:
        return encode_action_index(decision, engine_action)
    except ValueError:
        return None


def _generate_one_track(
    track_seed: int,
    split: str,
    *,
    num_players: int,
    game_seed: int,
    expert: BaseAgent,
    opponent: BaseAgent,
    buf: _DemoBuffer,
) -> tuple[int, int]:
    """Run one race on a generated track, logging the expert's real decisions.

    Drives ``run_round_driver`` like ``HeatEnv``: the learner seat (0) is the
    ``expert``; every other seat is ``opponent``. At each learner decision with
    ``> 1`` legal action (matching the env's auto-resolve of degenerate
    decisions), a ``(obs, target, mask, kind)`` tuple is logged BEFORE the
    expert's action is applied. Returns ``(logged, unencodable)`` -- the number
    of tuples logged and the number of real-choice decisions whose expert action
    had no codec encoding (dropped, see :func:`_encode_target`).

    The expert's action is always *applied* (even for degenerate decisions, so
    the game advances exactly as it would in play); only the logging is gated on
    a real choice.
    """
    track = generate_track(track_seed, _TIGHT_PARAMS)
    state = GameState.create(track, num_players, seed=game_seed)
    for player in state.players:
        player.lap = 1  # mirror Game.__init__ / HeatEnv.reset

    learner_id = 0
    logged = 0
    unencodable = 0
    gen = run_round_driver(state)
    send_value: object = None

    while True:
        if state.is_game_over or state.round_num > MAX_ROUNDS:
            break
        try:
            decision = gen.send(send_value)
        except StopIteration:
            if state.is_game_over or state.round_num > MAX_ROUNDS:
                break
            gen = run_round_driver(state)
            send_value = None
            continue

        if decision.player_id != learner_id:
            send_value = opponent_action(opponent, decision, state)
            continue

        # Learner decision. Compute the mask FIRST: the env only surfaces real
        # choices (mask.sum() > 1) as RL steps and auto-resolves the rest, so the
        # BC net only ever needs to imitate the real-choice decisions.
        mask = legal_action_mask(decision, state)
        n_legal = int(mask.sum())

        engine_action = opponent_action(expert, decision, state)

        if n_legal > 1:
            target = _encode_target(decision, engine_action)
            if target is None:
                # Expert chose an action outside the codec's representable set
                # (e.g. an off-table REACT). The BC policy could never emit it,
                # so it is not a learnable target: skip logging, still apply it.
                unencodable += 1
            else:
                # Contract guard: an encodable action MUST be set in the mask.
                # encode_action_index collapses CARDS to the same value-multiset
                # the env decodes to, so this holds by construction; the assert
                # makes any codec drift loud rather than silent.
                if not (0 <= target < ACTION_DIM and bool(mask[target])):
                    raise AssertionError(
                        f"expert action {engine_action!r} for {decision.kind} "
                        f"encoded to off-mask index {target} (codec drift?)"
                    )
                obs = encode_observation(state, learner_id, decision)
                buf.add(
                    obs=obs,
                    action=target,
                    mask=mask,
                    kind=_KIND_TO_INT[decision.kind],
                    track_seed=track_seed,
                    split=split,
                )
                logged += 1

        send_value = engine_action

    return logged, unencodable


def _make_expert(num_players: int, args: argparse.Namespace) -> BaseAgent:
    """Build the demonstration expert (the S1/S2 LookaheadAgent).

    In 4p the expert uses the S2 hidden-info determinization belief (pool each
    opponent's ``hand+draw_pile``, keep discard fixed) so its targets are
    consistent with the env's real information set; in solo it is a no-op and the
    plain agent is used. Branching control (``top_k``/``sim_budget``) is plumbed
    through so demo generation stays affordable on a large track count.
    """
    determinize = num_players > 1 and not args.open_hand
    return LookaheadAgent(
        name="DemoExpert",
        horizon=args.horizon,
        n_determinizations=args.dets,
        determinize_hidden=determinize,
        top_k=args.top_k,
        sim_budget=args.sim_budget,
        seed=args.expert_seed,
    )


def generate_dataset(args: argparse.Namespace) -> dict:
    """Generate the train+val demonstration dataset; return a stats summary."""
    buf = _DemoBuffer()

    bands = [
        ("train", _TRAIN_SEED_BASE, args.train_tracks),
        ("val", _VAL_SEED_BASE, args.val_tracks),
    ]

    t0 = time.perf_counter()
    total_unencodable = 0
    for split, base, n_tracks in bands:
        for i in range(n_tracks):
            track_seed = base + i
            expert = _make_expert(args.players, args)
            opponent = HeuristicAgent(name="DemoOpp")
            _, unenc = _generate_one_track(
                track_seed,
                split,
                num_players=args.players,
                game_seed=args.game_seed + i,
                expert=expert,
                opponent=opponent,
                buf=buf,
            )
            total_unencodable += unenc
    elapsed = time.perf_counter() - t0

    obs = np.asarray(buf.obs, dtype=np.float32)
    action = np.asarray(buf.action, dtype=np.int64)
    mask = np.asarray(buf.mask, dtype=bool)
    kind = np.asarray(buf.kind, dtype=np.int8)
    track_seed = np.asarray(buf.track_seed, dtype=np.int64)
    split = np.asarray(buf.split, dtype="S5")

    if obs.shape[0] == 0:
        raise RuntimeError("no demonstrations generated (check track counts)")
    if obs.shape[1] != OBS_DIM or mask.shape[1] != ACTION_DIM:
        raise RuntimeError(
            f"shape mismatch: obs {obs.shape} mask {mask.shape} vs contract "
            f"OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=obs,
        action=action,
        mask=mask,
        kind=kind,
        track_seed=track_seed,
        split=split,
    )

    # Sidecar provenance (next to the .npz). Contract version is recorded so the
    # trainer can fail fast if the dataset was built against a drifted codec.
    meta = {
        "codec_version": CODEC_VERSION,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "num_players": args.players,
        "horizon": args.horizon,
        "dets": args.dets,
        "top_k": args.top_k,
        "sim_budget": args.sim_budget,
        "open_hand": bool(args.open_hand),
        "expert_seed": args.expert_seed,
        "game_seed": args.game_seed,
        "train_seed_base": _TRAIN_SEED_BASE,
        "val_seed_base": _VAL_SEED_BASE,
        "train_tracks": args.train_tracks,
        "val_tracks": args.val_tracks,
        "n_tuples": int(obs.shape[0]),
    }
    meta_path = os.path.splitext(args.out)[0] + ".demos.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # --- Report dataset size + class balance of decision types ---
    int_to_kind = {v: k.name for k, v in _KIND_TO_INT.items()}
    train_mask = split == b"train"
    val_mask = split == b"val"
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())

    def kind_balance(sel: np.ndarray) -> dict[str, int]:
        ks, counts = np.unique(kind[sel], return_counts=True)
        return {int_to_kind[int(k)]: int(c) for k, c in zip(ks, counts)}

    # Limit-1 CARDS decisions are the headline corner skill; count CARDS tuples
    # logged while the car sits on/before a limit-1 corner is not directly
    # available here (no track context stored), so we report per-kind balance
    # and leave the limit-1 accuracy slice to the trainer (which has the obs).
    summary = {
        "n_tuples": int(obs.shape[0]),
        "n_train": n_train,
        "n_val": n_val,
        "unencodable_dropped": total_unencodable,
        "train_tracks": args.train_tracks,
        "val_tracks": args.val_tracks,
        "elapsed_s": round(elapsed, 1),
        "out": args.out,
        "meta_path": meta_path,
        "train_kind_balance": kind_balance(train_mask),
        "val_kind_balance": kind_balance(val_mask),
    }
    return summary


def _print_summary(s: dict) -> None:
    print("\n=== gen_demos summary ===")
    print(f"  wrote {s['n_tuples']} tuples to {s['out']}")
    print(f"  sidecar: {s['meta_path']}")
    print(
        f"  train: {s['n_train']} tuples over {s['train_tracks']} tracks; "
        f"val: {s['n_val']} tuples over {s['val_tracks']} tracks "
        f"(track-disjoint)"
    )
    print(
        f"  dropped (no codec encoding, e.g. off-table REACT): "
        f"{s['unencodable_dropped']}"
    )
    print(f"  generation time: {s['elapsed_s']}s")
    print("  train decision-kind balance:")
    for k, c in sorted(s["train_kind_balance"].items()):
        print(f"    {k:<10} {c:>7}  ({c / max(1, s['n_train']) * 100:.1f}%)")
    print("  val decision-kind balance:")
    for k, c in sorted(s["val_kind_balance"].items()):
        print(f"    {k:<10} {c:>7}  ({c / max(1, s['n_val']) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/bc_demos.npz",
                        help="output .npz path (sidecar .demos.json alongside)")
    parser.add_argument("--players", type=int, default=4,
                        help="seats per game (1 = solo; 4 = the determinized 4p)")
    parser.add_argument("--train-tracks", type=int, default=60,
                        help="number of generated TRAIN tracks (disjoint band)")
    parser.add_argument("--val-tracks", type=int, default=20,
                        help="number of generated VAL tracks (disjoint band)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="expert LookaheadAgent rollout depth")
    parser.add_argument("--dets", type=int, default=2,
                        help="expert determinizations per candidate")
    parser.add_argument("--top-k", type=int, default=6,
                        help="expert top-k branching cap (None=full); 6 = the "
                             "S2 'free accuracy' setting, ~2.4x cheaper")
    parser.add_argument("--sim-budget", type=int, default=None,
                        help="expert per-move clone budget (None=unbounded)")
    parser.add_argument("--open-hand", action="store_true",
                        help="force open-hand 4p search (no determinization); "
                             "default uses the S2 hidden-info belief in 4p")
    parser.add_argument("--expert-seed", type=int, default=0,
                        help="seed mixed into the expert's per-turn rollout RNG")
    parser.add_argument("--game-seed", type=int, default=7000,
                        help="base seed for the per-track game RNG")
    args = parser.parse_args()

    print(
        f"gen_demos: players={args.players} train_tracks={args.train_tracks} "
        f"val_tracks={args.val_tracks} horizon={args.horizon} dets={args.dets} "
        f"top_k={args.top_k} open_hand={args.open_hand} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    summary = generate_dataset(args)
    _print_summary(summary)


if __name__ == "__main__":
    main()
