"""Sprint A1 (Option A) -- Monte-Carlo value dataset for the learned leaf.

Builds the supervised dataset the value-net trainer (``train_value.py``) regresses
on: every solo learner state labeled with its realized **rounds-remaining**, the
Monte-Carlo cost-to-go target the leaf consumes as ``V_leaf = -rounds_remaining``
(README A `2.1`). This is a *structural copy* of ``experiments/gen_demos.py`` -- the
same proven solo driver loop, the same track-disjoint TRAIN/VAL seed bands, the
same ``_TIGHT_PARAMS`` corner mix -- but it labels rounds-remaining instead of an
action target.

Why drive the engine directly (not ``HeatEnv``)
------------------------------------------------
Exactly as ``gen_demos.py``: the rollout policy is a :class:`BaseAgent`, so we run
:func:`heat.engine.driver.run_round_driver` ourselves (the loop
:class:`heat.ml.env.HeatEnv` uses) and route the lone learner seat (0) through the
``HeuristicAgent`` -- the S1-validated rollout policy (README A `2.5`: the strong
heuristic is the *wrong* policy on this distribution). **Solo only: no opponent
seats.**

What is logged, and the MC labeling rule
-----------------------------------------
1. **Log every learner decision, not just real choices.** ``gen_demos`` only logs
   ``mask.sum() > 1`` decisions because BC imitates the *action* the env surfaces.
   V scores a *state*, not a pending decision, so we record a row at **every**
   learner decision the driver hands us (a state is a state regardless of how many
   moves are legal from it).
2. **Encode with ``decision=None``.** V values a rolled-out *leaf state*, so we
   pass ``decision=None`` to :func:`encode_observation` (the zero-filled phase
   block) -- the SAME convention A2 will use when it encodes a search leaf
   (README A `2.3`). A round-trip parity test asserts byte-identity with the leaf
   encoding.
3. **Backfill rounds-remaining on completion.** Each row stores the
   ``state.round_num`` it was logged at. On race completion we set
   ``finish_round = state.round_num`` and backfill
   ``rounds_remaining = finish_round - round_num`` for every row of that race
   (a Monte-Carlo return -- the realized cost-to-go, README A `2.1`).
4. **Drop + count ``MAX_ROUNDS`` races.** A race that hits ``MAX_ROUNDS`` without
   the solo car finishing has no valid ``finish_round`` (the target would be
   poisoned), so its rows are discarded and the truncated-race count is reported
   as a sanity check (the heuristic finishes ~100% solo on the gate band, so this
   should be near-zero -- README A `7`).

Per-state corner context
-------------------------
For honest per-bucket error reporting (README A success criteria), each row also
stores ``corner_limit_at_state`` -- the speed-limit of the next corner ahead at
the moment the state was logged, via :func:`heat.engine.rules.distance_to_next_corner`.
The trainer buckets val MAE by this, with the **limit-1 bucket called out** (the
high-variance, high-stakes slice).

Track-disjoint train/val split
-------------------------------
TRAIN ``100_000+`` / VAL ``500_000+`` -- the exact bands ``gen_demos.py`` and the
eval harness use, both disjoint from the held-out GATE band (``900_000+``), so V
is never trained or validated on a gate track.

Output
------
A compressed ``.npz`` with parallel arrays ``obs (N, OBS_DIM) float32``,
``rounds_remaining (N,) float32``, ``corner_limit_at_state (N,) int8``,
``track_seed (N,) int64``, and ``split (N,) S5`` (``b"train"``/``b"val"``). Plus a
``.value.json`` provenance sidecar (codec version, bands, generator policy,
rollouts-per-track).

Usage:
    python experiments/gen_value_data.py --train-tracks 60 --val-tracks 20 \
        --rollouts 1 --out data/value_data.npz
    python experiments/gen_value_data.py --train-tracks 80 --val-tracks 20 --rollouts 4
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
from heat.engine.driver import run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.engine.rules import distance_to_next_corner
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import CODEC_VERSION, OBS_DIM
from heat.models.game_state import GameState
from heat.tracks.generator import TrackGenParams, generate_track


# Mirror the eval harness's held-out, limit-1-weighted generated distribution
# (experiments/eval_search.py) -- the SAME corner mix the value net is gated on.
# Train/val seed bands are disjoint sub-ranges of this. Copied verbatim from
# gen_demos.py so V trains on exactly the demo distribution.
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),  # weight toward tight (limit-1) corners
    laps=2,
)

#: Seed band for TRAIN tracks. Distinct from both the eval harness's held-out
#: band (900_000+, experiments/eval_search.py) and the VAL band below, so the
#: value net is never trained on a track it is later validated or gated on.
_TRAIN_SEED_BASE = 100_000
#: Seed band for VAL tracks (track-disjoint from train; also disjoint from the
#: eval harness's 900_000+ gate band).
_VAL_SEED_BASE = 500_000

#: Sentinel for "no corner ahead" (a corner-less track). int8-storable; the
#: trainer treats it as its own bucket. Distinct from any real speed limit (>= 1).
_NO_CORNER_LIMIT = 0


@dataclass
class _ValueBuffer:
    """Growable column store for the value-regression rows.

    ``round_num`` and ``finish_round`` are kept per-row so the Monte-Carlo target
    ``rounds_remaining = finish_round - round_num`` can be backfilled once a race
    completes; a sentinel ``finish_round == -1`` marks a row whose race was dropped
    (``MAX_ROUNDS``-truncated) so it can be filtered out before saving.
    """

    obs: list[np.ndarray] = field(default_factory=list)
    round_num: list[int] = field(default_factory=list)
    finish_round: list[int] = field(default_factory=list)
    corner_limit: list[int] = field(default_factory=list)
    track_seed: list[int] = field(default_factory=list)
    split: list[str] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        round_num: int,
        corner_limit: int,
        track_seed: int,
        split: str,
    ) -> int:
        """Append a row (finish_round unset). Return its index for backfilling."""
        self.obs.append(obs)
        self.round_num.append(round_num)
        self.finish_round.append(-1)  # backfilled on race completion
        self.corner_limit.append(corner_limit)
        self.track_seed.append(track_seed)
        self.split.append(split)
        return len(self.obs) - 1

    def __len__(self) -> int:
        return len(self.obs)


def _corner_limit_at_state(state: GameState, player_id: int) -> int:
    """Speed-limit of the next corner ahead of ``player_id`` (for bucketed error).

    Uses :func:`heat.engine.rules.distance_to_next_corner` -- the same convention
    the codec's track block uses. Returns :data:`_NO_CORNER_LIMIT` when the track
    has no corners ahead (``distance_to_next_corner`` returns ``(None, ...)``).
    """
    player = state.get_player(player_id)
    corner, _dist = distance_to_next_corner(state.track, player.position)
    if corner is None:
        return _NO_CORNER_LIMIT
    return int(corner.speed_limit)


def _generate_one_track(
    track_seed: int,
    split: str,
    *,
    game_seed: int,
    policy: BaseAgent,
    buf: _ValueBuffer,
) -> bool:
    """Drive one full solo race, logging a row at every learner decision.

    Mirrors ``gen_demos._generate_one_track`` (the proven solo driver loop:
    ``player.lap = 1`` init, the ``MAX_ROUNDS`` guard, the StopIteration
    re-arming of ``run_round_driver``), but for the single learner seat (0) and
    logging a value row -- ``obs = encode_observation(state, 0, decision=None)``
    plus the current ``round_num`` and corner context -- at EVERY learner
    decision rather than only real choices.

    On completion, backfills ``rounds_remaining = finish_round - round_num`` for
    this race's rows. Returns ``True`` if the race finished cleanly (rows kept),
    ``False`` if it hit ``MAX_ROUNDS`` without finishing (rows dropped + counted).
    """
    track = generate_track(track_seed, _TIGHT_PARAMS)
    state = GameState.create(track, 1, seed=game_seed)
    for player in state.players:
        player.lap = 1  # mirror Game.__init__ / HeatEnv.reset

    learner_id = 0
    row_ids: list[int] = []  # indices of rows logged for THIS race (to backfill)
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
            # Solo: there are no other seats, so this branch is never taken. Kept
            # to mirror gen_demos's loop exactly (and be robust if a future driver
            # surfaces a non-learner decision).
            send_value = opponent_action(policy, decision, state)
            continue

        # V scores a STATE, not a pending decision: log EVERY learner decision
        # (not just real choices), encoded with decision=None to match A2's leaf
        # encoding. Record the round_num so the MC target can be backfilled.
        obs = encode_observation(state, learner_id, decision=None)
        row_id = buf.add(
            obs=obs,
            round_num=state.round_num,
            corner_limit=_corner_limit_at_state(state, learner_id),
            track_seed=track_seed,
            split=split,
        )
        row_ids.append(row_id)

        send_value = opponent_action(policy, decision, state)

    finished = state.is_game_over and not (state.round_num > MAX_ROUNDS)
    if finished:
        finish_round = state.round_num
        for row_id in row_ids:
            buf.finish_round[row_id] = finish_round
    # If not finished, the rows keep finish_round == -1 and are dropped at save.
    return finished


def _policy_factory(args: argparse.Namespace):
    """Return a zero-arg factory that builds seat 0's trajectory policy.

    The MC labeling (rounds_remaining, decision=None, MAX_ROUNDS drop) is
    policy-independent -- only *who drives seat 0* changes between the A1 default
    and the A3 light-policy-iteration step:

      * ``--policy heuristic`` (default): :class:`HeuristicAgent` -- byte-identical
        to A1 (``HeuristicAgent(name="ValueGen")``), so A1's ``test_value_net.py``
        is unaffected. The S1-validated rollout policy (README A `2.5`).
      * ``--policy learned``: the A2 agent
        ``LookaheadAgent(leaf_value="learned", value_model_path=<current V>)`` --
        the SAME agent A2 ships -- so V can be refit on trajectories the heuristic
        never demonstrates (Sprint A3 §2.6). A2's ``top_k``/``sim_budget``
        branching control is threaded so search-policy generation stays affordable.

    A factory (not a shared instance) is returned so each (track, rollout) gets an
    independent policy with clean per-game RNG state, exactly as A1 rebuilt a fresh
    ``HeuristicAgent`` per rollout.
    """
    policy = getattr(args, "policy", "heuristic")
    if policy == "heuristic":
        return lambda: HeuristicAgent(name="ValueGen")
    if policy == "learned":
        if getattr(args, "value_model", None) is None:
            raise ValueError("--value-model is required when --policy learned")
        value_model = args.value_model
        horizon = getattr(args, "horizon", 2)
        dets = getattr(args, "dets", 2)
        top_k = getattr(args, "top_k", None)
        sim_budget = getattr(args, "sim_budget", None)
        return lambda: LookaheadAgent(
            name="ValueGenLearned",
            horizon=horizon,
            n_determinizations=dets,
            leaf_value="learned",
            value_model_path=value_model,
            top_k=top_k,
            sim_budget=sim_budget,
        )
    raise ValueError(f"unknown --policy {policy!r} (expected heuristic|learned)")


def generate_dataset(args: argparse.Namespace) -> dict:
    """Generate the train+val value dataset; return a stats summary.

    ``args.rollouts`` independent races are driven per track (the
    rollouts-per-track CLI knob -- raise it to lower MC variance at the limit-1
    corners, README A `7`). Each rollout reseeds the game RNG so it is a distinct
    sample of the own-deck draw process.

    Seat 0's trajectory policy is selected by :func:`_policy_factory` (``heuristic``
    default / ``learned`` for the A3 iteration step); the labeling is identical
    either way.
    """
    buf = _ValueBuffer()
    make_policy = _policy_factory(args)
    policy_name = getattr(args, "policy", "heuristic")

    bands = [
        ("train", _TRAIN_SEED_BASE, args.train_tracks),
        ("val", _VAL_SEED_BASE, args.val_tracks),
    ]

    t0 = time.perf_counter()
    races_total = 0
    races_dropped = 0
    for split, base, n_tracks in bands:
        for i in range(n_tracks):
            track_seed = base + i
            for r in range(args.rollouts):
                # Distinct game seed per (track, rollout) so each rollout is an
                # independent sample of the own-deck draw (MC over the draw
                # chance node -- README A 2.2).
                game_seed = args.game_seed + i * args.rollouts + r
                policy = make_policy()
                finished = _generate_one_track(
                    track_seed,
                    split,
                    game_seed=game_seed,
                    policy=policy,
                    buf=buf,
                )
                races_total += 1
                if not finished:
                    races_dropped += 1
    elapsed = time.perf_counter() - t0

    if len(buf) == 0:
        raise RuntimeError("no states logged (check track/rollout counts)")

    # Assemble columns, then DROP rows whose race was MAX_ROUNDS-truncated
    # (finish_round == -1): their MC target is undefined and would poison V.
    obs = np.asarray(buf.obs, dtype=np.float32)
    round_num = np.asarray(buf.round_num, dtype=np.int64)
    finish_round = np.asarray(buf.finish_round, dtype=np.int64)
    corner_limit = np.asarray(buf.corner_limit, dtype=np.int8)
    track_seed = np.asarray(buf.track_seed, dtype=np.int64)
    split = np.asarray(buf.split, dtype="S5")

    keep = finish_round >= 0
    n_dropped_rows = int((~keep).sum())
    obs = obs[keep]
    round_num = round_num[keep]
    finish_round = finish_round[keep]
    corner_limit = corner_limit[keep]
    track_seed = track_seed[keep]
    split = split[keep]

    rounds_remaining = (finish_round - round_num).astype(np.float32)

    if obs.shape[0] == 0:
        raise RuntimeError(
            "all races hit MAX_ROUNDS without finishing -- no valid rows "
            "(the heuristic should finish ~100% solo; check the track params)"
        )
    if obs.shape[1] != OBS_DIM:
        raise RuntimeError(
            f"shape mismatch: obs {obs.shape} vs contract OBS_DIM={OBS_DIM}"
        )
    # MC return is non-negative by construction (finish_round >= round_num at
    # every logged decision); assert so a labeling bug is loud, not silent.
    if rounds_remaining.min() < 0:
        raise AssertionError(
            f"negative rounds_remaining ({rounds_remaining.min()}): a row was "
            "logged after its race's finish_round (labeling bug)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=obs,
        rounds_remaining=rounds_remaining,
        corner_limit_at_state=corner_limit,
        track_seed=track_seed,
        split=split,
    )

    # Sidecar provenance (next to the .npz). codec_version is recorded so the
    # trainer can fail fast if the dataset was built against a drifted codec.
    # generator_policy stays "HeuristicAgent" for the A1 default (test_value_net
    # asserts this exact string); the A3 learned step records the search agent and
    # the value model it was driven by, for provenance.
    if policy_name == "learned":
        generator_policy = "LookaheadAgent(leaf_value=learned)"
    else:
        generator_policy = "HeuristicAgent"
    meta = {
        "codec_version": CODEC_VERSION,
        "obs_dim": OBS_DIM,
        "generator_policy": generator_policy,
        "policy": policy_name,
        "value_model": (
            getattr(args, "value_model", None) if policy_name == "learned" else None
        ),
        "num_players": 1,
        "rollouts_per_track": args.rollouts,
        "train_seed_base": _TRAIN_SEED_BASE,
        "val_seed_base": _VAL_SEED_BASE,
        "train_tracks": args.train_tracks,
        "val_tracks": args.val_tracks,
        "game_seed": args.game_seed,
        "n_rows": int(obs.shape[0]),
        "races_total": races_total,
        "races_dropped_max_rounds": races_dropped,
        "rows_dropped_max_rounds": n_dropped_rows,
    }
    meta_path = os.path.splitext(args.out)[0] + ".value.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    train_sel = split == b"train"
    val_sel = split == b"val"
    summary = {
        "n_rows": int(obs.shape[0]),
        "n_train": int(train_sel.sum()),
        "n_val": int(val_sel.sum()),
        "races_total": races_total,
        "races_dropped_max_rounds": races_dropped,
        "rows_dropped_max_rounds": n_dropped_rows,
        "rounds_remaining_mean": float(rounds_remaining.mean()),
        "rounds_remaining_max": float(rounds_remaining.max()),
        "train_tracks": args.train_tracks,
        "val_tracks": args.val_tracks,
        "rollouts": args.rollouts,
        "elapsed_s": round(elapsed, 1),
        "out": args.out,
        "meta_path": meta_path,
    }
    return summary


def _print_summary(s: dict) -> None:
    print("\n=== gen_value_data summary ===")
    print(f"  wrote {s['n_rows']} rows to {s['out']}")
    print(f"  sidecar: {s['meta_path']}")
    print(
        f"  train: {s['n_train']} rows over {s['train_tracks']} tracks; "
        f"val: {s['n_val']} rows over {s['val_tracks']} tracks "
        f"(track-disjoint, {s['rollouts']} rollout(s)/track)"
    )
    print(
        f"  races: {s['races_total']} total, "
        f"{s['races_dropped_max_rounds']} dropped (MAX_ROUNDS, no finish) "
        f"-> {s['rows_dropped_max_rounds']} rows dropped"
    )
    print(
        f"  rounds_remaining: mean={s['rounds_remaining_mean']:.2f} "
        f"max={s['rounds_remaining_max']:.0f}"
    )
    print(f"  generation time: {s['elapsed_s']}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/value_data.npz",
                        help="output .npz path (sidecar .value.json alongside)")
    parser.add_argument("--train-tracks", type=int, default=60,
                        help="number of generated TRAIN tracks (disjoint band)")
    parser.add_argument("--val-tracks", type=int, default=20,
                        help="number of generated VAL tracks (disjoint band)")
    parser.add_argument("--rollouts", type=int, default=1,
                        help="MC rollouts (full races) per track; raise to lower "
                             "value variance at the limit-1 corners (README A 7)")
    parser.add_argument("--game-seed", type=int, default=7000,
                        help="base seed for the per-(track,rollout) game RNG")
    parser.add_argument("--policy", type=str, default="heuristic",
                        choices=["heuristic", "learned"],
                        help="seat-0 trajectory policy: 'heuristic' (A1 default) "
                             "or 'learned' (A2 LookaheadAgent for the A3 "
                             "policy-iteration step; requires --value-model)")
    parser.add_argument("--value-model", type=str, default=None,
                        help="V checkpoint path driving the learned policy "
                             "(required when --policy learned)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="LookaheadAgent rollout depth (--policy learned only)")
    parser.add_argument("--dets", type=int, default=2,
                        help="LookaheadAgent determinizations per candidate "
                             "(--policy learned only)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="LookaheadAgent top-k branching cap, A2's affordability "
                             "knob (--policy learned only)")
    parser.add_argument("--sim-budget", type=int, default=None,
                        help="LookaheadAgent per-move clone budget, A2's "
                             "affordability knob (--policy learned only)")
    args = parser.parse_args()
    if args.policy == "learned" and args.value_model is None:
        parser.error("--value-model is required when --policy learned")

    policy_desc = (
        "HeuristicAgent"
        if args.policy == "heuristic"
        else f"LookaheadAgent(learned, V={args.value_model})"
    )
    print(
        f"gen_value_data: train_tracks={args.train_tracks} "
        f"val_tracks={args.val_tracks} rollouts={args.rollouts} "
        f"policy={policy_desc} (solo) "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM})"
    )
    summary = generate_dataset(args)
    _print_summary(summary)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("gen_value_data", main)
