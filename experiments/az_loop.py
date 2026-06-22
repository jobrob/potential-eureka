"""Sprint C3 -- the closed AlphaZero loop at small scale (the collapse guard).

Turns C2's single generate->train->eval cycle into the iterated loop the
Option-C README S5 rung 4 needs: for ``gen`` in ``1..N``, generate self-play
targets with the **current-best** net in ``MCTSAgent`` (exploration on), train a
new net on the **aggregated** target buffer, evaluate it in-search AND net-only on
the held-out solo field, and **promote it only if it clears the Wilson-LB /
best-checkpoint guard** -- never promote a regression (the 8C bug-2 discipline).
Stop after ``K`` consecutive non-improving generations and report the plateau.

Why this is its own sprint (the single most important discipline)
-----------------------------------------------------------------
This repo has a long history of healthy-looking training collapses (8C, S3,
Sprint-5): the training curve looks fine while the *behavior* is worse than
random. The gate here is therefore the **behavioral** held-out eval (worst-case L1
spins/pass + rounds on generated tracks), never the training loss; the promotion
guard uses a **bootstrap lower-confidence bound** (``eval_az`` CIs) so a small
lucky sample cannot fake a pass; and both in-search and net-only are measured so a
divergence is visible.

The pieces, all reused (no reinvention)
---------------------------------------
* generation -- ``gen_selfplay.generate_dataset`` (C2), exploration ON, on a
  per-generation seed slice disjoint from the eval band AND C2's ``300_000`` band
  AND the BC/value precursor bands (asserted at startup);
* training -- ``train_az.train_az`` (C2) on the aggregated buffer (DAgger-style
  across generations, with a recency window so stale early targets age out);
* eval + the rung-4 CI gate -- ``eval_az`` (the held-out solo field, in-search +
  net-only, bootstrap-CI worst-case spins/rounds);
* the promotion guard -- the A3 ``value_iterate`` pattern (strict-improvement stop
  rule + best-checkpoint preservation), adapted to the solo CI metric here.

``--smoke`` shrinks generations/tracks/sims to prove the pipeline end-to-end in
minutes (the S4 posture: pipeline real, headline numbers may need a real campaign).

Usage:
    python experiments/az_loop.py --smoke --warm checkpoints/c3_warm_gen0.zip
    python experiments/az_loop.py --warm checkpoints/c3_warm_gen0.zip \
        --generations 6 --tracks 80 --val-tracks 20 --sims 16 --gate-games 24
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

import gen_selfplay
import train_az
import eval_az
from eval_search import _HELDOUT_BASE, _run_field
from eval_az import build_solo_labels, agg_spin_ci, agg_rounds_ci, CI

from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


# ---------------------------------------------------------------------------
# Seed bands for the C3 loop's self-play (per generation, all disjoint)
# ---------------------------------------------------------------------------

#: C3 self-play base, distinct from C2's _SELFPLAY_SEED_BASE=300_000, the BC/value
#: precursor bands (100_000 / 500_000), and the held-out eval band (900_000). Each
#: generation gets its own 10_000-wide slice within [_C3_SELFPLAY_BASE, ...).
_C3_SELFPLAY_BASE = 600_000
#: Per-generation slice width (train + val tracks must fit; 10_000 is ample).
_C3_GEN_SLICE = 10_000

_C2_SELFPLAY_BASE = 300_000
_EVAL_BASE = 900_000
_BC_TRAIN_BASE = 100_000
_BC_VAL_BASE = 500_000


def _gen_selfplay_base(generation: int) -> int:
    """The disjoint self-play seed base for ``generation`` (1-indexed)."""
    return _C3_SELFPLAY_BASE + (generation - 1) * _C3_GEN_SLICE


def _assert_c3_band_disjoint(generation: int, n_tracks: int) -> None:
    """Fail fast if generation ``generation``'s self-play slice overlaps any band.

    The slice is ``[base, base+n_tracks)``. We require a clear gap (a 100_000-wide
    window) to the eval band, C2's self-play band, and the BC/value precursor
    bands -- the same disjointness discipline ``gen_selfplay`` enforces, extended
    to the per-generation C3 slices (which must also not collide with each other,
    guaranteed by the ``_C3_GEN_SLICE`` stride being wider than any plausible
    track count).
    """
    base = _gen_selfplay_base(generation)
    hi = base + n_tracks
    if n_tracks >= _C3_GEN_SLICE:
        raise ValueError(
            f"generation {generation} needs {n_tracks} tracks but the per-gen "
            f"slice is only {_C3_GEN_SLICE} wide; raise _C3_GEN_SLICE"
        )
    for name, other_base in (
        ("eval-heldout", _EVAL_BASE),
        ("c2-selfplay", _C2_SELFPLAY_BASE),
        ("bc-train", _BC_TRAIN_BASE),
        ("bc-val", _BC_VAL_BASE),
    ):
        o_lo, o_hi = other_base, other_base + 100_000
        if not (hi <= o_lo or base >= o_hi):
            raise ValueError(
                f"C3 gen-{generation} self-play band [{base},{hi}) overlaps the "
                f"{name} band [{o_lo},{o_hi}); a gated/precursor/C2 track must "
                "never be a C3 self-play track"
            )


# ---------------------------------------------------------------------------
# The CI gate metric + the strict-improvement rule (the collapse guard)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopGate:
    """The C3 promotion metric: worst-case L1 spins + rounds, as bootstrap CIs.

    Promotion compares the **lower-confidence bound** of the candidate against the
    incumbent's so a small lucky sample cannot fake a pass (the anti-fake-pass
    discipline). The point estimates are kept for the report. Lower is better on
    both axes.
    """

    spins: CI  # worst-case (max) L1 spins/pass, bootstrap CI
    rounds: CI  # mean rounds-to-finish, bootstrap CI
    finish_rate: float

    def as_dict(self) -> dict:
        return {
            "spins": self.spins.__dict__,
            "rounds": self.rounds.__dict__,
            "finish_rate": self.finish_rate,
        }


def _improves_ci(cand: CI, best: CI, eps: float = 1e-9) -> bool:
    """Candidate strictly better than incumbent on this axis (CI-aware, lower=better).

    A trustworthy improvement requires the candidate's *point* to be below the
    incumbent's AND the candidate's CI upper bound to be below the incumbent's
    point (the gain is not an artifact of a wide interval). NaN on either side is
    inert (never an improvement). This is stricter than a raw point compare,
    matching the README "Wilson-LB / bootstrap CI, not a point estimate" bar.
    """
    if cand.point != cand.point or best.point != best.point:
        return False
    return cand.point < best.point - eps and cand.hi < best.point - eps


def _regresses_ci(cand: CI, best: CI, eps: float = 1e-9) -> bool:
    """Candidate strictly worse than incumbent on this axis (point compare, lower=better).

    Regression is judged on the point estimate (conservative: any real worsening
    blocks promotion, even within CI noise -- we never promote something that
    looks worse). NaN on either side is inert.
    """
    if cand.point != cand.point or best.point != best.point:
        return False
    return cand.point > best.point + eps


def strictly_improves(cand: LoopGate, best: LoopGate) -> bool:
    """True iff ``cand`` strictly improves on ``best`` per the C3 promotion rule.

    Rule (the A3 pattern, CI-hardened): a candidate is promoted iff it strictly
    improves on at least one axis (CI-separated) AND regresses on neither
    (point-wise) AND finishes 100%. A collapse (finish < 100%, or a regressed
    axis) is never promoted -- the 8C guard.
    """
    if cand.finish_rate < 0.999:
        return False
    if _regresses_ci(cand.spins, best.spins) or _regresses_ci(cand.rounds, best.rounds):
        return False
    return _improves_ci(cand.spins, best.spins) or _improves_ci(cand.rounds, best.rounds)


# ---------------------------------------------------------------------------
# The three reused steps (generate / train / gate)
# ---------------------------------------------------------------------------


def _generate(
    *,
    generation: int,
    model_path: str,
    out: str,
    tracks: int,
    val_tracks: int,
    sims: int,
    seed: int,
    game_seed: int,
    root_selector: str = "puct",
) -> dict:
    """Generate one generation of self-play targets with the current-best net.

    Thin wrapper over C2's ``gen_selfplay.generate_dataset`` with exploration ON.
    The per-generation disjoint seed slice is installed by overriding
    ``gen_selfplay._SELFPLAY_SEED_BASE`` (the module reads it at call time) AFTER
    asserting the slice is disjoint from every other band. Restores the C2 default
    afterward so the module is left untouched for other callers.
    """
    _assert_c3_band_disjoint(generation, tracks + val_tracks)
    base = _gen_selfplay_base(generation)
    saved = gen_selfplay._SELFPLAY_SEED_BASE
    gen_selfplay._SELFPLAY_SEED_BASE = base
    try:
        return gen_selfplay.generate_dataset(argparse.Namespace(
            out=out, model=model_path, tracks=tracks, val_tracks=val_tracks,
            sims=sims, dirichlet_eps=0.25, dirichlet_alpha=0.5,
            temperature_moves=10, seed=seed + generation, game_seed=game_seed + generation,
            root_selector=root_selector,
        ))
    finally:
        gen_selfplay._SELFPLAY_SEED_BASE = saved


def _aggregate(npz_paths: list[str], out: str) -> dict:
    """Materialize the recency-windowed union of self-play ``.npz`` buffers.

    Concatenates the parallel target arrays (``obs``/``pi``/``mask``/``z``/
    ``kind``/``track_seed``/``split``) across ``npz_paths`` (the window) and writes
    a ``.selfplay.json`` sidecar so the unchanged ``train_az._load_dataset`` (which
    validates the sidecar's codec_version) consumes the aggregate directly. The
    per-row track-disjoint ``split`` is carried verbatim so train/val disjointness
    is preserved across the union. Mirrors ``value_iterate._merge_datasets``.
    """
    import numpy as np

    cols = {k: [] for k in ("obs", "pi", "mask", "z", "kind", "track_seed", "split")}
    per_source = []
    for p in npz_paths:
        data = np.load(p)
        n = int(data["obs"].shape[0])
        if data["obs"].shape[1] != OBS_DIM:
            raise ValueError(f"{p}: obs {data['obs'].shape} != OBS_DIM={OBS_DIM}")
        for k in cols:
            cols[k].append(data[k])
        per_source.append({"path": p, "n_rows": n})

    merged = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(out, **merged)

    n_rows = int(merged["obs"].shape[0])
    meta = {
        "codec_version": CODEC_VERSION,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "num_players": 1,
        "generator": "az_loop aggregate (recency window)",
        "n_rows": n_rows,
        "sources": per_source,
    }
    with open(os.path.splitext(out)[0] + ".selfplay.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return {"out": out, "n_rows": n_rows, "n_sources": len(npz_paths),
            "per_source": per_source}


def _train(*, data: str, out: str, epochs: int, c_v: float, device: str,
           net: str, seed: int) -> dict:
    """Train a new AZ net on the aggregated buffer via C2's ``train_az``."""
    return train_az.train_az(argparse.Namespace(
        data=data, out=out, epochs=epochs, batch=256, lr=3e-4, c_v=c_v,
        weight_decay=1e-4, eval_every=max(1, epochs // 2), patience=8,
        net=net, device=device, seed=seed,
    ))


def _gate(*, model_path: str, gate_games: int, sims: int, horizon: int, dets: int,
          seed: int) -> tuple[LoopGate, LoopGate]:
    """Gate a checkpoint on the held-out solo band; return (in-search, net-only).

    Runs the ``eval_az`` solo field (Heuristic / Lookahead / AZ-MCTS / AZ-net) once
    and reads the two AZ contenders' bootstrap CIs off the returned ``_AgentAgg``s.
    Both gates share the single field run (one expensive eval, two readouts).
    """
    labels = build_solo_labels(
        model_path=model_path, sims=sims, horizon=horizon, dets=dets, seed=seed,
        include_net_only=True,
    )
    track_seeds = [_HELDOUT_BASE + i for i in range(gate_games)]
    solo, _ = _run_field(labels, track_seeds=track_seeds, num_players=1,
                         game_seed_base=seed)

    def _read(label: str) -> LoopGate:
        agg = solo[label]
        return LoopGate(
            spins=agg_spin_ci(agg, 1, seed=seed),
            rounds=agg_rounds_ci(agg, seed=seed + 1),
            finish_rate=agg.finish_rate(),
        )

    return _read("AZ-MCTS"), _read("AZ-net")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class GenRecord:
    """One generation's outcome (for the report)."""

    generation: int
    data_path: str
    model_path: str
    train_summary: dict
    in_search: LoopGate
    net_only: LoopGate
    promoted: bool
    reason: str


def run_loop(args: argparse.Namespace) -> dict:
    """Run the closed loop; return a report dict.

    Starts from the warm gen-0 prior (``--warm``), gated first to establish the
    incumbent the first generation must strictly beat (best-checkpoint
    preservation: the shipped net is never worse than the warm prior). Then up to
    ``--generations`` times: generate self-play with the current best, train on the
    recency-windowed aggregate, gate in-search + net-only, apply the CI promotion
    rule. Stop after ``--stop-patience`` consecutive non-improving generations.
    The best-gating net is copied to ``--out``.
    """
    os.makedirs(args.workdir, exist_ok=True)

    # Gate the warm prior so generation 1 has a real incumbent to beat.
    print("\n--- gating warm gen-0 prior (the incumbent to beat) ---")
    best_model = args.warm
    best_in_search, best_net_only = _gate(
        model_path=best_model, gate_games=args.gate_games, sims=args.sims,
        horizon=args.horizon, dets=args.dets, seed=args.seed,
    )
    print(f"  warm gen-0 in-search gate: {best_in_search.as_dict()}")
    print(f"  warm gen-0 net-only  gate: {best_net_only.as_dict()}")

    buffer_paths: list[str] = []
    records: list[GenRecord] = []
    non_improving = 0
    t0 = time.perf_counter()

    for gen in range(1, args.generations + 1):
        print(f"\n=== generation {gen}/{args.generations} ===")
        data_path = os.path.join(args.workdir, f"gen{gen}.npz")
        agg_path = os.path.join(args.workdir, f"agg{gen}.npz")
        model_path = os.path.join(args.workdir, f"net_gen{gen}.zip")

        # (1) self-play with the CURRENT best net (exploration on).
        print(f"  [1/3] self-play with best net ({best_model})")
        gen_summary = _generate(
            generation=gen, model_path=best_model, out=data_path,
            tracks=args.tracks, val_tracks=args.val_tracks, sims=args.sims,
            seed=args.seed, game_seed=args.game_seed,
            root_selector=getattr(args, "root_selector", "puct"),
        )
        print(f"        logged {gen_summary['n_rows']} targets "
              f"(pi_entropy={gen_summary['pi_entropy_mean']:.3f}, "
              f"{gen_summary['races_dropped_max_rounds']} races dropped)")

        # (2) train on the recency-windowed aggregate (DAgger-style).
        buffer_paths.append(data_path)
        window = buffer_paths[-args.buffer_window:]
        agg_summary = _aggregate(window, agg_path)
        print(f"  [2/3] training on aggregate of {agg_summary['n_sources']} gens "
              f"({agg_summary['n_rows']} rows; window={args.buffer_window})")
        train_summary = _train(
            data=agg_path, out=model_path, epochs=args.epochs, c_v=args.c_v,
            device=args.device, net=args.net, seed=args.seed,
        )

        # (3) gate in-search + net-only, apply the CI promotion rule.
        print("  [3/3] gating new net on the held-out solo band (in-search + net-only)")
        in_search, net_only = _gate(
            model_path=model_path, gate_games=args.gate_games, sims=args.sims,
            horizon=args.horizon, dets=args.dets, seed=args.seed,
        )
        print(f"        in-search: spins={in_search.spins.fmt()} "
              f"rounds={in_search.rounds.fmt()} finish={in_search.finish_rate * 100:.0f}%")
        print(f"        net-only:  spins={net_only.spins.fmt()} "
              f"rounds={net_only.rounds.fmt()} finish={net_only.finish_rate * 100:.0f}%")

        # Promotion is judged on the IN-SEARCH gate (the Option-C agent); the
        # net-only gate is reported for the fast-bot divergence signal.
        promoted = strictly_improves(in_search, best_in_search)
        if promoted:
            reason = "in-search strictly improves (CI-separated) -> promoted"
            best_in_search = in_search
            best_net_only = net_only
            best_model = model_path
            non_improving = 0
            print(f"        -> {reason}")
        else:
            non_improving += 1
            reason = (f"no CI-separated improvement -> incumbent kept "
                      f"({non_improving}/{args.stop_patience} non-improving)")
            print(f"        -> {reason}")

        records.append(GenRecord(
            generation=gen, data_path=data_path, model_path=model_path,
            train_summary=train_summary, in_search=in_search, net_only=net_only,
            promoted=promoted, reason=reason,
        ))

        if non_improving >= args.stop_patience:
            print(f"\n  STOP: {non_improving} consecutive non-improving generations "
                  f"(plateau). Reporting the gap to LookaheadAgent.")
            break

    elapsed = time.perf_counter() - t0

    # Ship the best-gating net (best-checkpoint preservation: never worse than the
    # warm prior; if no generation improved, the warm prior ships verbatim).
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    shutil.copyfile(best_model, args.out)
    best_meta = os.path.splitext(best_model)[0] + ".meta.json"
    if os.path.exists(best_meta):
        shutil.copyfile(best_meta, os.path.splitext(args.out)[0] + ".meta.json")

    report = {
        "out": args.out,
        "best_model_source": best_model,
        "best_is_warm": best_model == args.warm,
        "best_in_search": best_in_search.as_dict(),
        "best_net_only": best_net_only.as_dict(),
        "generations_run": len(records),
        "generations_requested": args.generations,
        "buffer_window": args.buffer_window,
        "stop_patience": args.stop_patience,
        "elapsed_s": round(elapsed, 1),
        "generations": [
            {
                "generation": r.generation,
                "promoted": r.promoted,
                "reason": r.reason,
                "in_search": r.in_search.as_dict(),
                "net_only": r.net_only.as_dict(),
                "model_path": r.model_path,
                "val_loss_best": r.train_summary.get("best_metric"),
                "val": r.train_summary.get("best_val", {}),
            }
            for r in records
        ],
    }
    report_path = os.path.splitext(args.out)[0] + ".loop.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    report["report_path"] = report_path
    return report


def _print_report(report: dict) -> None:
    print("\n=== az_loop summary ===")
    print(f"  generations run: {report['generations_run']} "
          f"(requested {report['generations_requested']}, "
          f"window={report['buffer_window']}, stop_patience={report['stop_patience']})")
    for g in report["generations"]:
        tag = "PROMOTED" if g["promoted"] else "kept incumbent"
        isg = g["in_search"]
        print(f"    gen {g['generation']}: in-search spins="
              f"{isg['spins']['point']:.3f} rounds={isg['rounds']['point']:.3f} "
              f"finish={isg['finish_rate'] * 100:.0f}%  [{tag}]")
    src = "WARM gen-0 (no generation improved)" if report["best_is_warm"] \
        else report["best_model_source"]
    print(f"  best-gating net: {src}")
    print(f"    in-search gate: {report['best_in_search']}")
    print(f"  shipped -> {report['out']}  (report {report['report_path']})")
    print(f"  total time: {report['elapsed_s']}s")
    print("\n  NOTE: the rung-4 verdict (BEAT LookaheadAgent) is the held-out "
          "behavioral gate -- run experiments/eval_az.py on the shipped net for the "
          "CI-gated rung-4 table. This loop's promotion guard prevents a collapse "
          "from being shipped; it does not by itself assert rung 4.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=str, required=True,
                        help="warm gen-0 prior (from mint_warm_prior.py; codec v3 "
                             "with a .meta.json sidecar). The loop CANNOT bootstrap "
                             "from cold (C2 proved that futile).")
    parser.add_argument("--out", type=str, default="checkpoints/c3_best.zip",
                        help="output path for the best-gating net (sidecar + "
                             ".loop.json report alongside)")
    parser.add_argument("--workdir", type=str, default="runs/az_loop",
                        help="scratch dir for per-generation data/aggregates/nets")
    parser.add_argument("--generations", type=int, default=6,
                        help="max self-play generations")
    parser.add_argument("--tracks", type=int, default=80,
                        help="self-play TRAIN tracks per generation")
    parser.add_argument("--val-tracks", type=int, default=20,
                        help="self-play VAL tracks per generation (track-disjoint)")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTS sims for self-play AND the in-search gate")
    parser.add_argument("--buffer-window", type=int, default=3,
                        help="aggregate the last N generations' targets (recency "
                             "window; stale early targets age out)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="train_az epochs per generation")
    parser.add_argument("--c-v", type=float, default=1.0,
                        help="value-loss weight in L=CE+c_v*MSE")
    parser.add_argument("--gate-games", type=int, default=24,
                        help="held-out tracks for the promotion gate (900_000+ band)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="LookaheadAgent rollout depth (the rung-1 reference)")
    parser.add_argument("--dets", type=int, default=2,
                        help="LookaheadAgent determinizations")
    parser.add_argument("--stop-patience", type=int, default=2,
                        help="stop after K consecutive non-improving generations")
    parser.add_argument("--root-selector", type=str, default="puct",
                        choices=["puct", "gumbel"],
                        help="self-play root action selector + policy target (C4): "
                             "'puct' (default, visit-count target) or 'gumbel' "
                             "(Gumbel top-m + Sequential Halving, completed-Q target)")
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"])
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cuda|cpu for the trainer")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed (self-play, train, gate)")
    parser.add_argument("--game-seed", type=int, default=8000,
                        help="base per-track game RNG seed for self-play")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny smoke (2 gens, 4/2 tracks, sims=8, 6 gate games, "
                             "4 epochs, window 2)")
    args = parser.parse_args()

    if args.smoke:
        args.generations = 2
        args.tracks = 4
        args.val_tracks = 2
        args.sims = 8
        args.gate_games = 6
        args.epochs = 4
        args.buffer_window = 2
        args.stop_patience = 99  # run both gens in smoke (don't early-stop)

    print(
        f"az_loop: warm={args.warm} generations={args.generations} "
        f"tracks={args.tracks} val={args.val_tracks} sims={args.sims} "
        f"window={args.buffer_window} gate_games={args.gate_games} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    report = run_loop(args)
    _print_report(report)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("az_loop", main)
