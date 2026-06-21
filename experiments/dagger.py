"""Sprint S4 (DAgger) -- close the BC distribution gap on the learner's states.

S3's behavioral gate failed *by design*: behavioral cloning learns the expert's
labels well on the states the **expert** visits, but the expert almost never
spins, so the post-spin 0-heat **recovery** states the cloned net actually lands
in are essentially absent from the demos. The net's ~22% CARDS mistakes drop it
into exactly those unseen states, where it has no learned recovery and re-spins
-- the limit-1 death-spiral. (See "Sprint S3 -- Outcome" in
``docs/sprint-search-imitation-design.md``.)

DAgger (Dataset Aggregation, Ross et al. 2011) is the matched fix: roll out the
**current learner**, collect the states *it* actually visits, query the expert
``LookaheadAgent`` for the correct action **on those exact states**, aggregate
the new ``(obs, expert_target, mask)`` tuples into the dataset, and retrain.
Iterating drives the training distribution toward the learner's own state
distribution -- including the post-spin recovery states BC never saw.

What this module does (and how it mirrors gen_demos exactly)
------------------------------------------------------------
One DAgger *iteration* = roll out + aggregate + retrain:

1. **Roll out the current learner** through ``run_round_driver`` -- the exact
   loop ``HeatEnv`` / ``gen_demos`` use. The learner seat is driven by an
   :class:`MLAgent` loaded from the current checkpoint (so the trajectory follows
   the LEARNER's distribution -- this is the whole point of DAgger, vs gen_demos
   which follows the EXPERT's distribution).
2. **Label with the expert.** At each learner real-choice decision
   (``mask.sum() > 1`` -- the same auto-resolve rule the env applies), query a
   fresh-per-track ``LookaheadAgent`` for the action it would take *in that exact
   state*, encode it with the frozen codec, and log ``(obs, expert_target,
   mask)``. Then **apply the learner's own action** to advance the game (NOT the
   expert's) -- so the next state is one the learner reaches.
3. **Aggregate** the new tuples into the running dataset (DAgger's "D <- D u D_i")
   and **retrain** BC on the union via :func:`train_bc.train_bc`.

Hard constraints inherited from S3 (non-negotiable -- see the doc's S3 outcome):

* **Label generation mirrors gen_demos exactly.** Same frozen obs/action codec
  (:func:`heat.ml.features.encode_observation`,
  :func:`heat.ml.action_codec.encode_action_index`, ``spaces.CODEC_VERSION``);
  CARDS targets snap via the value-multiset rule (``encode_action_index`` ==
  ``env._decode_legal``); only real-choice (``mask.sum() > 1``) learner states
  are logged; the SAME ~3-4% of unencodable expert actions (off-table REACT) are
  **dropped and counted**, never crashed or fabricated. The shared helpers
  (:func:`gen_demos._encode_target`, ``_DemoBuffer``, ``_KIND_TO_INT``,
  ``_TIGHT_PARAMS``, the seed bands, ``_make_expert``) are imported from
  ``gen_demos`` so there is one source of truth.
* **4p uses the S2 determinization belief.** In 4p the expert is built with
  ``determinize_hidden=True`` (pool opponent ``hand+draw_pile``, keep discard
  fixed, id-sort then shuffle) -- the same info set ``env`` exposes -- so the
  DAgger labels are consistent with the env's information set.

Why querying the expert on the learner's state is correct
---------------------------------------------------------
The ``LookaheadAgent`` caches its joint ``(gear, cards)`` plan keyed on a per-turn
signature that includes the player's *current* gear. At the GEAR decision the
expert plans from the learner's live pre-shift state and we log its gear. The
learner then applies its OWN (possibly different) gear; by the CARDS decision the
player's committed gear has changed, so the expert's turn signature differs and
``choose_cards`` recomputes the plan *restricted to the learner's committed
gear*. The CARDS label is therefore the expert's best play **conditioned on the
gear the learner actually chose** -- precisely the DAgger label ("what the expert
would do in the state the learner reached"), not a stale plan. A fresh expert is
built per track so no plan cache leaks across games (same discipline as
gen_demos).

Track-disjoint split (kept identical to gen_demos)
--------------------------------------------------
Aggregated tuples carry the same ``train``/``val`` split (track-disjoint seed
bands) so the trainer's val metrics stay honest across iterations. The DAgger
rollouts draw from the SAME train/val track bands as the seed dataset, so a
DAgger track is never one the gate (900_000+) is run on.

Usage:
    # iterate from a BC seed checkpoint; aggregate onto a seed dataset
    python experiments/dagger.py --seed-ckpt checkpoints/bc.zip \\
        --seed-data data/bc_demos.npz --iterations 3 \\
        --rollout-train-tracks 20 --rollout-val-tracks 8 \\
        --out-prefix checkpoints/dagger

    # cold start (no seed checkpoint): iter 0 trains BC on the seed dataset only
    python experiments/dagger.py --seed-data data/bc_demos.npz --iterations 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.ml_agent import MLAgent
from heat.engine.driver import run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.ml.action_codec import legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.tracks.generator import generate_track

# Single source of truth for the codec-faithful demo machinery (the S3 contract).
import gen_demos  # noqa: E402  (sibling experiments module)
import train_bc  # noqa: E402


# ---------------------------------------------------------------------------
# One learner rollout: collect expert labels on the LEARNER's own states
# ---------------------------------------------------------------------------


def _rollout_one_track(
    track_seed: int,
    split: str,
    *,
    num_players: int,
    game_seed: int,
    learner: BaseAgent,
    expert: BaseAgent,
    opponent: BaseAgent,
    buf: "gen_demos._DemoBuffer",
) -> tuple[int, int]:
    """Roll out the LEARNER on one track, labeling its states with the EXPERT.

    This is the DAgger counterpart of :func:`gen_demos._generate_one_track`. The
    crucial difference: the **learner**'s action is applied to advance the game
    (so the trajectory follows the learner's state distribution), while the
    **expert** is queried only to produce the training *label* at each logged
    state. ``gen_demos`` instead applies the expert's action (expert
    distribution) -- that asymmetry is exactly why BC misses recovery states and
    DAgger fixes it.

    Returns ``(logged, unencodable)`` -- tuples logged and real-choice learner
    decisions whose EXPERT action had no codec encoding (dropped, same rule as
    gen_demos). Degenerate decisions (``mask.sum() <= 1``) are auto-resolved (the
    learner's action applied) and never logged, matching ``HeatEnv``.
    """
    track = generate_track(track_seed, gen_demos._TIGHT_PARAMS)
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

        # Learner decision. Compute the mask FIRST (the env only surfaces real
        # choices, mask.sum() > 1, as RL steps and auto-resolves the rest), then
        # ask the LEARNER for the action that will actually be played.
        mask = legal_action_mask(decision, state)
        n_legal = int(mask.sum())
        learner_action = opponent_action(learner, decision, state)

        if n_legal > 1:
            # DAgger label: what would the EXPERT do *in this exact state*? Query
            # the expert WITHOUT mutating the live game (the expert plans on
            # internal clones and never advances the live RNG). The observation
            # snapshot is taken before any action is applied, so obs/mask/label
            # all describe the same learner-visited state.
            obs = encode_observation(state, learner_id, decision)
            expert_action = opponent_action(expert, decision, state)
            target = gen_demos._encode_target(decision, expert_action)
            if target is None:
                # Expert chose an action outside the codec's representable set
                # (e.g. an off-table REACT). The BC policy could never emit it, so
                # it is not a learnable target: drop it (same as gen_demos), and
                # still advance the game with the learner's action below.
                unencodable += 1
            else:
                if not (0 <= target < ACTION_DIM and bool(mask[target])):
                    raise AssertionError(
                        f"expert action {expert_action!r} for {decision.kind} "
                        f"encoded to off-mask index {target} (codec drift?)"
                    )
                buf.add(
                    obs=obs,
                    action=target,
                    mask=mask,
                    kind=gen_demos._KIND_TO_INT[decision.kind],
                    track_seed=track_seed,
                    split=split,
                )
                logged += 1

        # Advance the game with the LEARNER's action (DAgger: learner's
        # distribution). Even degenerate decisions apply the learner's action so
        # the trajectory is exactly the one the learner would drive.
        send_value = learner_action

    return logged, unencodable


def collect_dagger_dataset(
    learner_ckpt: str,
    *,
    num_players: int,
    train_tracks: int,
    val_tracks: int,
    expert_args: argparse.Namespace,
    game_seed_base: int,
) -> tuple[dict, dict]:
    """Roll out the learner over train+val track bands; return arrays + stats.

    The learner is an :class:`MLAgent` reloaded from ``learner_ckpt`` once and
    reused across tracks (its model is read-only at inference). A FRESH expert
    instance is built per track (its plan cache is keyed per turn and must not
    leak across games), exactly as ``gen_demos`` does. Returns ``(arrays, stats)``
    where ``arrays`` has the same keys ``gen_demos`` writes
    (``obs/action/mask/kind/track_seed/split``).
    """
    buf = gen_demos._DemoBuffer()
    learner = MLAgent(learner_ckpt, deterministic=True, name="DAggerLearner")

    bands = [
        ("train", gen_demos._TRAIN_SEED_BASE, train_tracks),
        ("val", gen_demos._VAL_SEED_BASE, val_tracks),
    ]

    t0 = time.perf_counter()
    total_unencodable = 0
    for split, base, n_tracks in bands:
        for i in range(n_tracks):
            track_seed = base + i
            expert = gen_demos._make_expert(num_players, expert_args)
            opponent = HeuristicAgent(name="DAggerOpp")
            _, unenc = _rollout_one_track(
                track_seed,
                split,
                num_players=num_players,
                game_seed=game_seed_base + i,
                learner=learner,
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

    if obs.shape[0] > 0 and (
        obs.shape[1] != OBS_DIM or mask.shape[1] != ACTION_DIM
    ):
        raise RuntimeError(
            f"shape mismatch: obs {obs.shape} mask {mask.shape} vs contract "
            f"OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM}"
        )

    arrays = {
        "obs": obs,
        "action": action,
        "mask": mask,
        "kind": kind,
        "track_seed": track_seed,
        "split": split,
    }
    stats = {
        "n_collected": int(obs.shape[0]),
        "n_train": int((split == b"train").sum()),
        "n_val": int((split == b"val").sum()),
        "unencodable_dropped": total_unencodable,
        "elapsed_s": round(elapsed, 1),
    }
    return arrays, stats


# ---------------------------------------------------------------------------
# Dataset aggregation (DAgger's D <- D u D_i)
# ---------------------------------------------------------------------------


def _load_npz(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def aggregate_datasets(parts: list[dict]) -> dict:
    """Concatenate demonstration arrays into one aggregated dataset.

    All parts must share the frozen contract shapes; an empty part (no tuples)
    is skipped. Row order is preserved (seed dataset first, then each DAgger
    iteration in order) so the split column stays aligned with its rows.
    """
    parts = [p for p in parts if p["obs"].shape[0] > 0]
    if not parts:
        raise RuntimeError("no data to aggregate (all parts empty)")
    for p in parts:
        if p["obs"].shape[1] != OBS_DIM or p["mask"].shape[1] != ACTION_DIM:
            raise RuntimeError("a dataset part has off-contract shapes")
    return {
        "obs": np.concatenate([p["obs"] for p in parts]).astype(np.float32),
        "action": np.concatenate([p["action"] for p in parts]).astype(np.int64),
        "mask": np.concatenate([p["mask"] for p in parts]).astype(bool),
        "kind": np.concatenate([p["kind"] for p in parts]).astype(np.int8),
        "track_seed": np.concatenate(
            [p["track_seed"] for p in parts]
        ).astype(np.int64),
        "split": np.concatenate([p["split"] for p in parts]).astype("S5"),
    }


def _save_dataset(arrays: dict, out: str) -> str:
    """Write an aggregated dataset .npz + .demos.json sidecar (codec-stamped)."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "codec_version": CODEC_VERSION,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "n_tuples": int(arrays["obs"].shape[0]),
        "source": "dagger.py (aggregated)",
    }
    meta_path = os.path.splitext(out)[0] + ".demos.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta_path


# ---------------------------------------------------------------------------
# Retrain BC on the aggregated dataset
# ---------------------------------------------------------------------------


def _retrain_bc(
    data_path: str, out_ckpt: str, args: argparse.Namespace
) -> dict:
    """Retrain the BC policy on the aggregated dataset via train_bc.train_bc.

    Each DAgger iteration retrains the policy from scratch on the *whole*
    aggregated dataset -- the standard DAgger formulation (train on D, not a warm
    continuation), which avoids compounding optimizer state across iterations and
    keeps each policy a clean function of the current aggregate.
    """
    bc_args = argparse.Namespace(
        data=data_path,
        out=out_ckpt,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        eval_every=max(1, args.epochs),  # report once at the end (quiet iters)
        # Per-iteration early stopping: each DAgger retrain restores its best-val
        # checkpoint instead of shipping the overfit last-epoch one (val
        # CARDS-acc peaks ~epoch 2-3 then regresses on the growing aggregate).
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        net=args.net,
        device=args.device,
        seed=args.seed,
        players_meta=args.players,
    )
    return train_bc.train_bc(bc_args)


# ---------------------------------------------------------------------------
# The DAgger loop
# ---------------------------------------------------------------------------


def run_dagger(args: argparse.Namespace) -> dict:
    """Run the full DAgger loop and return a per-iteration summary.

    Iteration 0 establishes the seed policy: if ``--seed-ckpt`` is given it is
    used directly as the iter-0 learner; otherwise BC is trained on the seed
    dataset alone (a cold start). Each subsequent iteration rolls out the current
    learner, labels with the expert, aggregates, and retrains.
    """
    expert_args = _expert_namespace(args)

    # The running aggregate starts from the seed dataset (always present -- it
    # anchors the val split's track-disjoint bands and gives the cold start
    # something to learn from).
    seed_arrays = _load_npz(args.seed_data)
    aggregate_parts = [seed_arrays]

    history: list[dict] = []
    os.makedirs(
        os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True
    )

    # --- Establish the iter-0 learner checkpoint. ---
    if args.seed_ckpt:
        current_ckpt = args.seed_ckpt
        print(f"[dagger] iter 0: using seed checkpoint {current_ckpt}")
        history.append({"iter": 0, "ckpt": current_ckpt, "seeded": True})
    else:
        # Cold start: train BC on the seed dataset alone.
        agg = aggregate_datasets(aggregate_parts)
        data0 = f"{args.out_prefix}_agg0.npz"
        _save_dataset(agg, data0)
        current_ckpt = f"{args.out_prefix}_iter0.zip"
        print(f"[dagger] iter 0 (cold start): BC on seed dataset "
              f"({agg['obs'].shape[0]} tuples) -> {current_ckpt}")
        bc = _retrain_bc(data0, current_ckpt, args)
        history.append({"iter": 0, "ckpt": current_ckpt, "seeded": False,
                        "n_train": bc["n_train"], "final": bc.get("final", {})})

    # --- DAgger iterations. ---
    for it in range(1, args.iterations + 1):
        print(f"\n[dagger] iter {it}: rolling out learner {current_ckpt}")
        new_arrays, roll_stats = collect_dagger_dataset(
            current_ckpt,
            num_players=args.players,
            train_tracks=args.rollout_train_tracks,
            val_tracks=args.rollout_val_tracks,
            expert_args=expert_args,
            # Vary the game seed per iteration so repeated tracks see fresh draws
            # (more state coverage), while staying within the train/val bands.
            game_seed_base=args.game_seed + it * 1000,
        )
        print(
            f"[dagger] iter {it}: collected {roll_stats['n_collected']} tuples "
            f"(train={roll_stats['n_train']} val={roll_stats['n_val']}, "
            f"dropped unencodable={roll_stats['unencodable_dropped']}, "
            f"{roll_stats['elapsed_s']}s)"
        )

        aggregate_parts.append(new_arrays)
        agg = aggregate_datasets(aggregate_parts)
        data_it = f"{args.out_prefix}_agg{it}.npz"
        _save_dataset(agg, data_it)

        out_ckpt = f"{args.out_prefix}_iter{it}.zip"
        print(f"[dagger] iter {it}: retrain BC on aggregate "
              f"({agg['obs'].shape[0]} tuples) -> {out_ckpt}")
        bc = _retrain_bc(data_it, out_ckpt, args)
        current_ckpt = out_ckpt
        history.append({
            "iter": it,
            "ckpt": out_ckpt,
            "collected": roll_stats,
            "n_agg": int(agg["obs"].shape[0]),
            "final": bc.get("final", {}),
        })

    return {
        "final_ckpt": current_ckpt,
        "iterations": args.iterations,
        "history": history,
    }


def _expert_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Build the gen_demos-compatible expert config namespace.

    Reuses ``gen_demos._make_expert`` so the DAgger expert is byte-identical to
    the demo expert: solo = plain agent, 4p = S2 determinized agent, with the
    same horizon / dets / top_k branching control.
    """
    return argparse.Namespace(
        players=args.players,
        horizon=args.horizon,
        dets=args.dets,
        top_k=args.top_k,
        sim_budget=args.sim_budget,
        open_hand=args.open_hand,
        expert_seed=args.expert_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Seed inputs.
    parser.add_argument("--seed-data", type=str, default="data/bc_demos.npz",
                        help="seed demonstration .npz (from gen_demos.py)")
    parser.add_argument("--seed-ckpt", type=str, default=None,
                        help="seed BC checkpoint to start iter 0 from "
                             "(omit for a cold start that trains BC on seed-data)")
    parser.add_argument("--out-prefix", type=str, default="checkpoints/dagger",
                        help="prefix for per-iteration checkpoints + aggregates")
    # DAgger loop.
    parser.add_argument("--iterations", type=int, default=3,
                        help="number of DAgger roll-out+retrain iterations")
    parser.add_argument("--rollout-train-tracks", type=int, default=20,
                        help="train-band tracks rolled out per iteration")
    parser.add_argument("--rollout-val-tracks", type=int, default=8,
                        help="val-band tracks rolled out per iteration")
    # Expert config (mirrors gen_demos defaults).
    parser.add_argument("--players", type=int, default=4,
                        help="seats per game (1 = solo; 4 = determinized 4p)")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--dets", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--sim-budget", type=int, default=None)
    parser.add_argument("--open-hand", action="store_true",
                        help="force open-hand 4p expert (no determinization)")
    parser.add_argument("--expert-seed", type=int, default=0)
    parser.add_argument("--game-seed", type=int, default=8000)
    # BC retrain hyperparameters (passed straight to train_bc).
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8,
                        help="per-iteration BC early-stop patience (epochs "
                             "without selection-metric improvement); 0 or "
                             ">= epochs disables early stopping")
    parser.add_argument("--early-stop-metric", type=str, default="val_ce",
                        choices=["val_ce", "val_acc", "val_cards_acc"],
                        help="metric selecting each iteration's best-val "
                             "checkpoint (passed to train_bc)")
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"dagger: seed_data={args.seed_data} seed_ckpt={args.seed_ckpt} "
        f"iterations={args.iterations} players={args.players} "
        f"rollout_tracks={args.rollout_train_tracks}+{args.rollout_val_tracks} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    summary = run_dagger(args)
    print("\n=== dagger summary ===")
    print(f"  final checkpoint: {summary['final_ckpt']}")
    for h in summary["history"]:
        line = f"  iter {h['iter']}: {h['ckpt']}"
        if "n_agg" in h:
            line += f"  (aggregate={h['n_agg']} tuples)"
        f = h.get("final") or {}
        if f:
            va = f.get("val", {})
            line += (f"  val acc={va.get('acc', float('nan')):.3f} "
                     f"cards_acc={va.get('cards_acc', float('nan')):.3f}")
        print(line)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("dagger", main)
