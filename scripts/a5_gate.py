#!/usr/bin/env python
"""A5 anti-collapse gate experiment (Sprint A5, §4.3 / §5).

Runs the A5 recipe for each requested ``--arms`` x ``--seeds`` at ``--timesteps``
on the Tiny-Heat 2-player bed and tabulates, per run:

    arm | seed | final vs-weak | peak vs-weak | regression (peak-final) |
    final vs-oldest-self | min entropy | controller engaged? | stage1 | wall(s)

plus per-arm mean +- std summaries, as markdown. There is NO tuning logic: the
script only measures and tabulates; the G1-G5 gate call is made by reading the
table (A3 discipline -- a FAIL is reported, never tuned away).

The gate is ~6 runs x ~3-5 min. It accepts subset invocation (a single arm and/or
a single seed) so the runs can be chunked to stay inside command timeouts; the
combined table is assembled by hand from the per-run rows, which print
immediately.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/a5_gate.py                       # all 6 runs
    PYTHONPATH=src python scripts/a5_gate.py --arms pool --seeds 0 # one run
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from heat.ml.selfplay.recipe import A5Config, Stage1ValidationError, train_selfplay_a5
from heat.ml.selfplay.tiny_heat import tiny_heat_track


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A5 anti-collapse gate experiment.")
    parser.add_argument("--arms", choices=["pool", "pure"], nargs="+",
                        default=["pool", "pure"],
                        help="Arms to run (default: pool pure).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds per arm (default: 0 1 2).")
    parser.add_argument("--timesteps", type=int, default=150_000,
                        help="Env steps per run (default: 150000).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout / PPO update (default: 2048).")
    parser.add_argument("--players", type=int, default=2,
                        help="Seats per game (default: 2).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Trunk widths (default: 256 256).")
    parser.add_argument("--head", choices=["masked", "dotprod"], default="masked",
                        help="Action head (default: masked).")
    return parser.parse_args(argv)


def _run_once(
    arm: str, seed: int, args: argparse.Namespace
) -> dict[str, float | str]:
    """Train one arm+seed and return a row of gate metrics."""
    config = A5Config(
        n_steps=args.n_steps,
        total_timesteps=args.timesteps,
        hidden_sizes=tuple(args.hidden),
        num_players=args.players,
        device="cpu",
        seed=seed,
        head=args.head,
        pool_prob=0.0 if arm == "pure" else 0.5,
    )

    t0 = time.perf_counter()
    try:
        _policy, records = train_selfplay_a5(config, track=tiny_heat_track())
        stage1 = "pass"
    except Stage1ValidationError as exc:
        wall = time.perf_counter() - t0
        return {
            "arm": arm, "seed": float(seed), "final_weak": float("nan"),
            "peak_weak": float("nan"), "regression": float("nan"),
            "final_oldest": float("nan"),
            "min_entropy": float(exc.diagnostics.get("entropy", float("nan"))),
            "engaged": float("nan"), "stage1": "FAIL", "wall": wall,
        }
    wall = time.perf_counter() - t0

    if not records:  # pragma: no cover - eval_every < n_iterations always here
        return {
            "arm": arm, "seed": float(seed), "final_weak": float("nan"),
            "peak_weak": float("nan"), "regression": float("nan"),
            "final_oldest": float("nan"), "min_entropy": float("nan"),
            "engaged": float("nan"), "stage1": stage1, "wall": wall,
        }

    last = records[-1]
    final_weak = last["winrate_vs_weak"]
    peak_weak = max(r["winrate_vs_weak"] for r in records)
    return {
        "arm": arm,
        "seed": float(seed),
        "final_weak": final_weak,
        "peak_weak": peak_weak,
        "regression": peak_weak - final_weak,
        "final_oldest": last["winrate_vs_oldest"],
        "min_entropy": last["min_entropy"],
        "engaged": last["ever_engaged"],
        "stage1": stage1,
        "wall": wall,
    }


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("=" * 78)
    print("HEAT A5 anti-collapse gate (Tiny-Heat, 2p)")
    print("=" * 78)
    print(
        f"arms={args.arms} seeds={args.seeds} timesteps={args.timesteps} "
        f"n_steps={args.n_steps} players={args.players} head={args.head}"
    )
    print()
    print(
        "| arm | seed | final vs-weak | peak vs-weak | regression | "
        "final vs-oldest | min entropy | engaged? | stage1 | wall(s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")

    rows: list[dict[str, float | str]] = []
    for arm in args.arms:
        for seed in args.seeds:
            row = _run_once(arm, seed, args)
            rows.append(row)
            engaged = row["engaged"]
            engaged_str = (
                "nan" if isinstance(engaged, float) and not np.isfinite(engaged)
                else ("yes" if engaged else "no")
            )
            print(
                f"| {row['arm']} | {int(row['seed'])} | "
                f"{_fmt(float(row['final_weak']))} | "
                f"{_fmt(float(row['peak_weak']))} | "
                f"{_fmt(float(row['regression']))} | "
                f"{_fmt(float(row['final_oldest']))} | "
                f"{_fmt(float(row['min_entropy']))} | {engaged_str} | "
                f"{row['stage1']} | {float(row['wall']):.0f} |",
                flush=True,
            )

    # --- per-arm summaries ---
    print()
    print("### Per-arm summary (mean +- std over seeds)")
    print()
    print(
        "| arm | final vs-weak | regression | final vs-oldest | min entropy | "
        "any engaged | any stage1 fail |"
    )
    print("|---|---|---|---|---|---|---|")
    for arm in args.arms:
        arm_rows = [r for r in rows if r["arm"] == arm]

        def _summ(key: str) -> str:
            vals = [
                float(r[key]) for r in arm_rows  # noqa: B023
                if np.isfinite(float(r[key]))
            ]
            if not vals:
                return "nan"
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return f"{mean:.3f} +- {std:.3f}"

        any_engaged = any(
            np.isfinite(float(r["engaged"])) and float(r["engaged"]) > 0
            for r in arm_rows
        )
        any_stage1_fail = any(r["stage1"] == "FAIL" for r in arm_rows)
        print(
            f"| {arm} | {_summ('final_weak')} | {_summ('regression')} | "
            f"{_summ('final_oldest')} | {_summ('min_entropy')} | "
            f"{'yes' if any_engaged else 'no'} | "
            f"{'yes' if any_stage1_fail else 'no'} |"
        )

    print()
    print("(Gate call G1-G5 is made by reading this table; no tuning here.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
