#!/usr/bin/env python
"""A6 dense terminal-margin gate experiment (Sprint A6, §4.4 / §5).

An A/B sample-efficiency experiment over the committed A5 *pool* recipe,
identical between arms except the dense terminal-margin coefficient:

    baseline (margin_coef=0.0)  vs  margin (margin_coef=0.5)

x seeds {0,1,2} x 150k steps, Tiny-Heat 2p, masked head. The baseline arm is a
re-run of the A5 gate's pool arm, so its cells should be consistent with
``docs/direction-A/A5-anticollapse-recipe.md`` §8.

Per run (read off the periodic eval records) it reports the vs-weak winrate at
the ~20k / ~40k / ~80k checkpoints and final, the steps-to-first-eval->=70%, the
min rollout entropy, and wall time; then per-arm mean +- std. There is NO tuning
logic -- the script only measures and tabulates; the G1-G4 gate call is made by
reading the table (A3/A5 discipline -- a FAIL is reported, never tuned away).

The gate is 6 runs x ~2.5-4 min. It accepts subset invocation (a single arm
and/or seed) so runs can be chunked to stay inside command timeouts; the combined
table is assembled by hand from the per-run rows, which print immediately.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/a6_gate.py                          # all 6 runs
    PYTHONPATH=src python scripts/a6_gate.py --arms margin --seeds 0  # one run
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from heat.ml.selfplay.recipe import A5Config, Stage1ValidationError, train_selfplay_a5
from heat.ml.selfplay.tiny_heat import tiny_heat_track

#: Checkpoint step targets (~20k / ~40k / ~80k) at which the vs-weak winrate is
#: read off the nearest eval record. Eval lands every 10 iterations (~20k steps
#: at n_steps=2048), so these fall on real eval checkpoints.
_CHECKPOINTS: tuple[int, ...] = (20_000, 40_000, 80_000)
#: The winrate-vs-weak bar used for the steps-to-threshold sample-efficiency
#: metric (design §5 G2).
_WR_THRESHOLD: float = 0.70

#: margin_coef per arm.
_ARM_COEF: dict[str, float] = {"baseline": 0.0, "margin": 0.5}


def _nearest_record(
    records: list[dict[str, float]], target_steps: int
) -> dict[str, float]:
    """Return the eval record whose cumulative ``steps`` is nearest ``target``."""
    return min(records, key=lambda r: abs(r["steps"] - target_steps))


def _steps_to_threshold(records: list[dict[str, float]]) -> float:
    """First cumulative ``steps`` at which vs-weak winrate >= threshold, else nan."""
    for rec in records:
        if rec["winrate_vs_weak"] >= _WR_THRESHOLD:
            return rec["steps"]
    return float("nan")


def _run_once(
    arm: str, seed: int, args: argparse.Namespace
) -> dict[str, float | str]:
    """Train one arm+seed (pool recipe) and return a row of gate metrics."""
    config = A5Config(
        n_steps=args.n_steps,
        total_timesteps=args.timesteps,
        hidden_sizes=tuple(args.hidden),
        num_players=args.players,
        device="cpu",
        seed=seed,
        head=args.head,
        pool_prob=0.5,  # the committed pool arm for both A/B cells
        margin_coef=_ARM_COEF[arm],
    )

    t0 = time.perf_counter()
    try:
        _policy, records = train_selfplay_a5(config, track=tiny_heat_track())
        stage1 = "pass"
    except Stage1ValidationError as exc:
        wall = time.perf_counter() - t0
        return {
            "arm": arm, "seed": float(seed), "wr20k": float("nan"),
            "wr40k": float("nan"), "wr80k": float("nan"), "final": float("nan"),
            "steps_to_70": float("nan"),
            "min_entropy": float(exc.diagnostics.get("entropy", float("nan"))),
            "stage1": "FAIL", "wall": wall,
        }
    wall = time.perf_counter() - t0

    if not records:  # pragma: no cover - eval_every < n_iterations always here
        return {
            "arm": arm, "seed": float(seed), "wr20k": float("nan"),
            "wr40k": float("nan"), "wr80k": float("nan"), "final": float("nan"),
            "steps_to_70": float("nan"), "min_entropy": float("nan"),
            "stage1": stage1, "wall": wall,
        }

    wr = [_nearest_record(records, cp)["winrate_vs_weak"] for cp in _CHECKPOINTS]
    return {
        "arm": arm,
        "seed": float(seed),
        "wr20k": wr[0],
        "wr40k": wr[1],
        "wr80k": wr[2],
        "final": records[-1]["winrate_vs_weak"],
        "steps_to_70": _steps_to_threshold(records),
        "min_entropy": records[-1]["min_entropy"],
        "stage1": stage1,
        "wall": wall,
    }


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def _fmt_steps(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{int(value)}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("=" * 84)
    print("HEAT A6 dense terminal-margin gate (Tiny-Heat, 2p, pool recipe)")
    print("=" * 84)
    print(
        f"arms={args.arms} seeds={args.seeds} timesteps={args.timesteps} "
        f"n_steps={args.n_steps} players={args.players} head={args.head}"
    )
    print()
    print(
        "| arm | seed | wr@20k | wr@40k | wr@80k | final | steps->=70% | "
        "min entropy | stage1 | wall(s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")

    rows: list[dict[str, float | str]] = []
    for arm in args.arms:
        for seed in args.seeds:
            row = _run_once(arm, seed, args)
            rows.append(row)
            print(
                f"| {row['arm']} | {int(row['seed'])} | "
                f"{_fmt(float(row['wr20k']))} | {_fmt(float(row['wr40k']))} | "
                f"{_fmt(float(row['wr80k']))} | {_fmt(float(row['final']))} | "
                f"{_fmt_steps(float(row['steps_to_70']))} | "
                f"{_fmt(float(row['min_entropy']))} | {row['stage1']} | "
                f"{float(row['wall']):.0f} |",
                flush=True,
            )

    # --- per-arm summaries ---
    print()
    print("### Per-arm summary (mean +- std over seeds)")
    print()
    print(
        "| arm | wr@20k | wr@40k | wr@80k | final | mean steps->=70% |"
    )
    print("|---|---|---|---|---|---|")
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

        print(
            f"| {arm} | {_summ('wr20k')} | {_summ('wr40k')} | {_summ('wr80k')} | "
            f"{_summ('final')} | {_summ('steps_to_70')} |"
        )

    print()
    print("(Gate call G1-G4 is made by reading this table; no tuning here.)")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A6 dense-margin gate experiment.")
    parser.add_argument("--arms", choices=["baseline", "margin"], nargs="+",
                        default=["baseline", "margin"],
                        help="Arms to run (default: baseline margin).")
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


if __name__ == "__main__":
    sys.exit(main())
