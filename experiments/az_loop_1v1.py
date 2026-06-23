"""Sprint C6 -- the perfect-info 1v1 win/loss AlphaZero loop (the pivot loop).

The competitive analogue of ``az_loop`` (C3): for ``gen`` in ``1..N``, generate
**1v1 win/loss** self-play targets with the current-best net against a
**frozen-snapshot league** (the current net + N frozen past best-of-generation
snapshots + the strong/weak heuristic anchors), train a new net on the
recency-windowed aggregate with ``value_mode=winloss`` (tanh head + MSE-on-±1),
and **promote it only if it clears the Wilson-LB seat-neutral 1v1 win-rate gate**
vs the frozen strong-heuristic reference. The frozen snapshots are the anti-cycle
insurance (the project's self-play-collapse nemesis, Sprints 5/8C/Sprint-B): a
net that beats only its immediate predecessor but regresses vs an older self is
caught because the older self is still in the pool.

What is reused (no reinvention)
-------------------------------
* generation -- ``gen_selfplay.generate_dataset`` with ``two_player=True`` (the
  1v1 win/loss generator), on the C3 per-generation disjoint seed slices;
* aggregation -- ``az_loop._aggregate`` (the recency window), unchanged;
* training -- ``train_az.train_az`` with ``value_mode="winloss"``;
* the gate -- ``eval_1v1.league_gate_1v1`` (seat-neutral, num_seats=2, Wilson-LB)
  + the ``az_loop`` best-checkpoint preservation discipline (re-pointed at the 1v1
  win-rate instead of the solo spins/rounds CI).

``--smoke`` shrinks generations/tracks/sims to prove the pipeline end-to-end in
minutes (the C0-C5 posture: pipeline real, headline numbers need the Tier-0 run).

Usage:
    python experiments/az_loop_1v1.py --smoke --warm checkpoints/c6_warm_gen0.zip
    python experiments/az_loop_1v1.py --warm checkpoints/c6_warm_gen0.zip \\
        --generations 6 --tracks 64 --val-tracks 16 --sims 16 \\
        --league-snapshots 2 --gate-games 24 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

import gen_selfplay
import train_az
from az_loop import _aggregate, _assert_c3_band_disjoint, _gen_selfplay_base
from eval_1v1 import league_gate_1v1, make_mcts_focal

from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


# ---------------------------------------------------------------------------
# Generation: 1v1 win/loss self-play vs a league member
# ---------------------------------------------------------------------------


def _generate_1v1(
    *,
    generation: int,
    model_path: str,
    opponent_snapshot: str | None,
    out: str,
    tracks: int,
    val_tracks: int,
    sims: int,
    seed: int,
    game_seed: int,
    traj_greedy: bool = False,
    workers: int = 1,
) -> dict:
    """Generate one generation of 1v1 win/loss self-play with the current net.

    Thin wrapper over the C6 ``gen_selfplay.generate_dataset(two_player=True)`` on
    the per-generation disjoint seed slice (reusing the C3 band bookkeeping).
    ``opponent_snapshot=None`` is net-vs-net (both seats log); a path / "strong" /
    "weak" sentinel is net-vs-frozen (only the current-net seat logs).
    """
    _assert_c3_band_disjoint(generation, tracks + val_tracks)
    base = _gen_selfplay_base(generation)
    saved = gen_selfplay._SELFPLAY_SEED_BASE
    gen_selfplay._SELFPLAY_SEED_BASE = base
    try:
        return gen_selfplay.generate_dataset(argparse.Namespace(
            out=out, model=model_path, tracks=tracks, val_tracks=val_tracks,
            sims=sims, dirichlet_eps=0.25, dirichlet_alpha=0.5,
            temperature_moves=10, seed=seed + generation,
            game_seed=game_seed + generation, root_selector="puct",
            traj_greedy=traj_greedy, two_player=True,
            opponent_snapshot=opponent_snapshot, workers=workers,
        ))
    finally:
        gen_selfplay._SELFPLAY_SEED_BASE = saved


def _train_winloss(*, data: str, out: str, epochs: int, c_v: float, device: str,
                   net: str, seed: int) -> dict:
    """Train a winloss AZ net on the aggregated 1v1 buffer (tanh head + MSE-on-±1)."""
    return train_az.train_az(argparse.Namespace(
        data=data, out=out, epochs=epochs, batch=256, lr=3e-4, c_v=c_v,
        weight_decay=1e-4, eval_every=max(1, epochs // 2), patience=8,
        net=net, device=device, seed=seed, value_mode="winloss",
        warm_value=None, critic_anchor="none", c_anchor=1.0,
        lr_critic=3e-5, freeze_critic_epochs=0,
    ))


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


@dataclass
class GenRecord1v1:
    generation: int
    model_path: str
    train_summary: dict
    gate: dict
    promoted: bool          # ship-promoted (cleared the vs-strong parity ship-gate)
    improving: bool         # read (b): beat its own predecessor (the stop signal)
    reason: str


_PARITY = 0.5


def _ship_promote(cand_lb: float, best_lb: float, parity: float = _PARITY) -> bool:
    """Ship-gate (best-checkpoint preservation): which net is copied to ``--out``.

    A candidate is ship-worthy only if its seat-neutral vs-STRONG Wilson-LB clears
    parity (beats the strong heuristic outright) AND exceeds the incumbent's. This
    is the FINAL bar, NOT a per-generation keep-going signal -- decoupling the two
    is the C6-Tier-0 fix: the learner advances every generation regardless, so the
    curriculum can escalate even before any net is good enough to ship.
    """
    return cand_lb > parity + 1e-9 and cand_lb > best_lb + 1e-9


def _is_improving(gate: dict, parity: float = _PARITY) -> bool:
    """Read (b), the curriculum-escalator stop signal: did gen k beat gen k-1?

    Uses the vs-prev (self-play) win-rate POINT estimate > parity, not the
    Wilson-LB and not the vs-strong gate. The point estimate (rather than the LB)
    avoids false-stopping on the wide intervals of a 48-game Tier-0 gate, while
    still catching a genuine plateau/collapse (the learner failing to beat its own
    predecessor). Absent a vs-prev read (should not happen post-fix), treat as
    non-improving.
    """
    prev = gate.get("vs_prev")
    return bool(prev) and prev["win_rate"] > parity + 1e-9


def _league_opponents(
    *, snapshots: list[str], n_snapshots: int
) -> list[str | None]:
    """The per-generation opponent pool: current-net + N frozen snapshots + anchors.

    ``None`` is the net-vs-net (current-net) member; each frozen-snapshot PATH is a
    past best-of-generation checkpoint; "strong"/"weak" are the heuristic anchors.
    The N most-recent snapshots are kept (AlphaStar-style retention -- bounded
    storage, the loop's existing per-generation checkpoints).
    """
    pool: list[str | None] = [None]  # net-vs-net (the current net both seats)
    pool += snapshots[-n_snapshots:]
    pool += ["strong", "weak"]
    return pool


def run_loop(args: argparse.Namespace) -> dict:
    """Run the C6 1v1 win/loss loop; return a report dict.

    Starts from the warm gen-0 prior (``--warm``); the first generation must beat
    it. Each generation: generate 1v1 self-play vs each league member (round-robin
    over the pool), train a winloss net on the recency window, gate on the
    seat-neutral 1v1 win-rate vs the frozen strong heuristic (Wilson-LB > 50%
    promotes; an older self in the league catches a regression). The best-gating
    net is copied to ``--out``; its checkpoint joins the frozen-snapshot pool.
    """
    os.makedirs(args.workdir, exist_ok=True)

    # Three SEPARATE roles (the C6-Tier-0 decoupling fix):
    #  * current_model -- the learner; advances EVERY generation (gen k self-plays
    #    and trains from gen k-1). This is what drives the curriculum escalator.
    #  * best_model    -- best-checkpoint-for-shipping; only advances when a net
    #    clears the vs-strong parity ship-gate. Copied to --out at the end.
    #  * snapshots     -- frozen past nets for the anti-cycle league pool.
    current_model = args.warm
    best_model = args.warm
    snapshots: list[str] = []  # frozen per-generation checkpoint paths (league pool)

    # Gate the warm prior so generation 1 has a real reference.
    print("\n--- gating warm gen-0 prior (the 1v1 reference to beat) ---")
    warm_focal = make_mcts_focal(best_model, sims=args.sims, seed=args.seed)
    track_seeds = [900_000 + i for i in range(args.gate_games)]
    best_gate = league_gate_1v1(
        warm_focal, track_seeds=track_seeds, seed_base=args.seed + 9000,
    )
    print(f"  warm gen-0 vs strong: {best_gate['vs_strong']['win_rate'] * 100:.1f}% "
          f"(LB {best_gate['vs_strong']['wilson_lb'] * 100:.1f}%)")

    records: list[GenRecord1v1] = []
    buffer_paths: list[str] = []
    non_improving = 0
    t0 = time.perf_counter()

    for gen in range(1, args.generations + 1):
        print(f"\n=== generation {gen}/{args.generations} ===")
        model_path = os.path.join(args.workdir, f"net_gen{gen}.zip")
        agg_path = os.path.join(args.workdir, f"agg{gen}.npz")

        # (1) 1v1 self-play vs each league member (round-robin), pooled into one
        #     per-generation dataset (the frozen snapshots break the cycle).
        pool = _league_opponents(snapshots=snapshots, n_snapshots=args.league_snapshots)
        print(f"  [1/3] 1v1 self-play with current net ({current_model}) vs league "
              f"pool of {len(pool)} members")
        per_member_paths: list[str] = []
        per_member = max(1, args.tracks // len(pool))
        per_member_val = max(1, args.val_tracks // len(pool))
        for mi, opp in enumerate(pool):
            member_out = os.path.join(args.workdir, f"gen{gen}_m{mi}.npz")
            _generate_1v1(
                generation=gen * 100 + mi,  # disjoint per-member band slice
                model_path=current_model, opponent_snapshot=opp, out=member_out,
                tracks=per_member, val_tracks=per_member_val, sims=args.sims,
                seed=args.seed, game_seed=args.game_seed,
                traj_greedy=args.traj_greedy, workers=args.workers,
            )
            per_member_paths.append(member_out)

        # Aggregate this generation's per-member datasets into one gen buffer.
        gen_data = os.path.join(args.workdir, f"gen{gen}.npz")
        _aggregate(per_member_paths, gen_data)
        buffer_paths.append(gen_data)

        # (2) train a winloss net on the recency-windowed aggregate.
        window = buffer_paths[-args.buffer_window:]
        agg_summary = _aggregate(window, agg_path)
        print(f"  [2/3] training (winloss) on aggregate of {agg_summary['n_sources']} "
              f"gens ({agg_summary['n_rows']} rows; window={args.buffer_window})")
        train_summary = _train_winloss(
            data=agg_path, out=model_path, epochs=args.epochs, c_v=args.c_v,
            device=args.device, net=args.net, seed=args.seed,
        )
        ece = train_summary.get("v_calibration_ece", float("nan"))
        print(f"        value-head calibration ECE (the C6 number to watch): {ece:.3f}")

        # (3) gate: vs-strong (the ship-gate, read a) AND vs the PREVIOUS
        #     generation's net (read b, the curriculum escalator). Read (b) is
        #     computed EVERY generation -- against gen k-1 (the warm net for gen 1),
        #     regardless of whether anything has shipped yet. This is the single
        #     most important read and was silently skipped pre-fix.
        print("  [3/3] gating new net (seat-neutral 1v1 win-rate, Wilson-LB)")
        cand_focal = make_mcts_focal(model_path, sims=args.sims, seed=args.seed)
        prev_focal = make_mcts_focal(current_model, sims=args.sims, seed=args.seed + 1)
        gate = league_gate_1v1(
            cand_focal, track_seeds=track_seeds, seed_base=args.seed + 9000,
            make_prev=prev_focal,
        )
        s = gate["vs_strong"]
        pv = gate.get("vs_prev") or {}
        print(f"        vs strong: {s['win_rate'] * 100:.1f}% "
              f"(LB {s['wilson_lb'] * 100:.1f}%, {s['wins']}/{s['games']})")
        if pv:
            print(f"        vs gen-{gen - 1} (read b): {pv['win_rate'] * 100:.1f}% "
                  f"(LB {pv['wilson_lb'] * 100:.1f}%)")

        # Ship-gate (read a): does this net beat the STRONG heuristic outright? This
        # only decides which checkpoint to copy to --out; it does NOT gate the loop.
        cand_lb = gate["vs_strong"]["wilson_lb"]
        best_lb = best_gate["vs_strong"]["wilson_lb"]
        promoted = _ship_promote(cand_lb, best_lb)
        if promoted:
            best_model = model_path
            best_gate = gate

        # Stop signal (read b): the learner ALWAYS advances and its checkpoint ALWAYS
        # joins the league; we only STOP when it stops beating its own predecessor
        # (a genuine plateau/collapse), not when it fails to out-race the heuristic.
        improving = _is_improving(gate)
        current_model = model_path
        snapshots.append(model_path)
        non_improving = 0 if improving else non_improving + 1

        reason = (
            f"ship={'YES' if promoted else 'no'} "
            f"(vs-strong LB {cand_lb * 100:.1f}% / parity 50%); "
            f"improving={'YES' if improving else 'no'} "
            f"(vs gen-{gen - 1} {pv.get('win_rate', float('nan')) * 100:.1f}%); "
            f"{non_improving}/{args.stop_patience} non-improving"
        )
        print(f"        -> {reason}")

        records.append(GenRecord1v1(
            generation=gen, model_path=model_path, train_summary=train_summary,
            gate=gate, promoted=promoted, improving=improving, reason=reason,
        ))

        if non_improving >= args.stop_patience:
            print(f"\n  STOP: {non_improving} consecutive non-improving generations "
                  f"(learner no longer beating its own predecessor).")
            break

    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    shutil.copyfile(best_model, args.out)
    best_meta = os.path.splitext(best_model)[0] + ".meta.json"
    if os.path.exists(best_meta):
        shutil.copyfile(best_meta, os.path.splitext(args.out)[0] + ".meta.json")

    report = {
        "out": args.out,
        "best_model_source": best_model,
        "best_is_warm": best_model == args.warm,
        "best_gate": best_gate,
        "generations_run": len(records),
        "generations_requested": args.generations,
        "elapsed_s": round(elapsed, 1),
        "generations": [
            {
                "generation": r.generation,
                "promoted": r.promoted,
                "improving": r.improving,
                "reason": r.reason,
                "gate": r.gate,
                "value_calibration_ece": r.train_summary.get("v_calibration_ece"),
                "model_path": r.model_path,
            }
            for r in records
        ],
    }
    report_path = os.path.splitext(args.out)[0] + ".loop1v1.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    report["report_path"] = report_path
    return report


def _print_report(report: dict) -> None:
    print("\n=== az_loop_1v1 summary (the C6 moving-loop reads) ===")
    print(f"  generations run: {report['generations_run']} "
          f"(requested {report['generations_requested']})")
    for g in report["generations"]:
        tag = "SHIP" if g["promoted"] else ("improving" if g.get("improving") else "flat")
        s = g["gate"]["vs_strong"]
        ece = g.get("value_calibration_ece", float("nan"))
        prev = g["gate"].get("vs_prev")
        prev_s = (f" vs_self={prev['win_rate'] * 100:.0f}% (LB {prev['wilson_lb'] * 100:.0f}%)"
                  if prev else "")
        print(f"    gen {g['generation']}: vs_strong={s['win_rate'] * 100:.0f}% "
              f"(LB {s['wilson_lb'] * 100:.0f}%){prev_s}  ECE={ece:.3f}  [{tag}]")
    # The moving-loop reads: (b) does it beat earlier selves (the curriculum
    # escalator -- THE most important read)? (a) does the strong-heuristic win-rate
    # climb? (c) does ECE tighten?
    prev_curve = [(g["gate"].get("vs_prev") or {}).get("win_rate")
                  for g in report["generations"]]
    strong_curve = [g["gate"]["vs_strong"]["win_rate"] for g in report["generations"]]
    ece_curve = [g.get("value_calibration_ece", float("nan"))
                 for g in report["generations"]]
    print(f"  (b) vs-prev (self) win-rate curve: "
          f"{[round(x, 2) if x is not None else None for x in prev_curve]}")
    print(f"  (a) vs-strong win-rate curve: {[round(x, 2) for x in strong_curve]}")
    print(f"  (c) value calibration ECE curve: {[round(x, 3) for x in ece_curve]}")
    src = "WARM gen-0 (no generation improved)" if report["best_is_warm"] \
        else report["best_model_source"]
    print(f"  best-gating net: {src}")
    print(f"  shipped -> {report['out']}  (report {report['report_path']})")
    print(f"  total time: {report['elapsed_s']}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=str, required=True,
                        help="warm gen-0 prior (codec v3 + .meta.json; ideally a "
                             "value_mode=winloss net, else the search uses the raw "
                             "critic until generation 1 trains a winloss head)")
    parser.add_argument("--out", type=str, default="checkpoints/c6_best.zip")
    parser.add_argument("--workdir", type=str, default="runs/az_loop_1v1")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--tracks", type=int, default=64,
                        help="1v1 self-play TRAIN tracks per generation (split "
                             "across the league pool)")
    parser.add_argument("--val-tracks", type=int, default=16)
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTS sims for self-play AND the 1v1 gate")
    parser.add_argument("--league-snapshots", type=int, default=2,
                        help="number of frozen past snapshots in the league pool "
                             "(the anti-cycle ladder; + strong/weak anchors)")
    parser.add_argument("--buffer-window", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--c-v", type=float, default=1.0)
    parser.add_argument("--gate-games", type=int, default=24,
                        help="held-out tracks (x2 seats = seat-neutral 1v1 games)")
    parser.add_argument("--stop-patience", type=int, default=2)
    parser.add_argument("--traj-greedy", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="C7: parallel self-play workers for generation "
                             "(threaded through to gen_selfplay; default 1 = "
                             "serial, byte-for-byte unchanged)")
    parser.add_argument("--net", type=str, default="small",
                        choices=["default", "small", "large"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game-seed", type=int, default=8000)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny smoke (2 gens, 4/2 tracks, sims=6, 4 gate games, "
                             "4 epochs, 1 snapshot)")
    args = parser.parse_args()

    if args.smoke:
        args.generations = 2
        args.tracks = 4
        args.val_tracks = 2
        args.sims = 6
        args.gate_games = 4
        args.epochs = 4
        args.league_snapshots = 1
        args.buffer_window = 2
        args.stop_patience = 99

    print(
        f"az_loop_1v1: warm={args.warm} generations={args.generations} "
        f"tracks={args.tracks} val={args.val_tracks} sims={args.sims} "
        f"league_snapshots={args.league_snapshots} gate_games={args.gate_games} "
        f"net={args.net} (codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, "
        f"ACTION_DIM={ACTION_DIM})"
    )
    report = run_loop(args)
    _print_report(report)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("az_loop_1v1", main)
