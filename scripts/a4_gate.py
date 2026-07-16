#!/usr/bin/env python
"""A4 flat-vs-structured observation-encoder gate experiment.

Runs the committed A5 pool recipe on Tiny-Heat with identical settings except
for ``A0Config.encoder``.  It reports the fixed ~40k sample-efficiency point,
area under the periodic vs-weak evaluation curve, final/peak/regression and the
A5 collapse guards.  The script measures only; it never tunes or changes the
default based on its own output.

Usage from the repository root::

    PYTHONPATH=src python scripts/a4_gate.py
    PYTHONPATH=src python scripts/a4_gate.py --encoders structured --seeds 0
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from heat.ml.selfplay.recipe import A5Config, Stage1ValidationError, train_selfplay_a5
from heat.ml.selfplay.tiny_heat import tiny_heat_track

_SAMPLE_EFFICIENCY_STEPS = 40_000


def _nearest_record(
    records: list[dict[str, float]], target_steps: int
) -> dict[str, float]:
    return min(records, key=lambda record: abs(record["steps"] - target_steps))


def _normalized_auc(records: list[dict[str, float]]) -> float:
    """Trapezoidal vs-weak AUC, normalized to a win-rate in ``[0, 1]``."""
    if not records:
        return float("nan")
    # Before the first measurement use the 2-player chance line.  This avoids
    # silently granting either encoder its first observed win rate at step zero.
    steps = np.asarray([0.0, *(r["steps"] for r in records)], dtype=np.float64)
    rates = np.asarray(
        [0.5, *(r["winrate_vs_weak"] for r in records)], dtype=np.float64
    )
    return float(np.trapezoid(rates, steps) / steps[-1])


def _failed_row(
    encoder: str,
    seed: int,
    wall: float,
    entropy: float,
) -> dict[str, float | str]:
    return {
        "encoder": encoder,
        "seed": float(seed),
        "wr40k": float("nan"),
        "auc": float("nan"),
        "final": float("nan"),
        "peak": float("nan"),
        "regression": float("nan"),
        "min_entropy": entropy,
        "engaged": float("nan"),
        "stage1": "FAIL",
        "wall": wall,
    }


def _run_once(
    encoder: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    config = A5Config(
        n_steps=args.n_steps,
        total_timesteps=args.timesteps,
        hidden_sizes=tuple(args.hidden),
        num_players=2,
        device="cpu",
        seed=seed,
        head="masked",
        encoder=encoder,
        pool_prob=0.5,
    )
    started = time.perf_counter()
    try:
        _policy, records = train_selfplay_a5(config, track=tiny_heat_track())
    except Stage1ValidationError as exc:
        return _failed_row(
            encoder,
            seed,
            time.perf_counter() - started,
            float(exc.diagnostics.get("entropy", float("nan"))),
        )
    wall = time.perf_counter() - started
    if not records:
        return _failed_row(encoder, seed, wall, float("nan"))

    rates = [record["winrate_vs_weak"] for record in records]
    final = rates[-1]
    peak = max(rates)
    return {
        "encoder": encoder,
        "seed": float(seed),
        "wr40k": _nearest_record(records, _SAMPLE_EFFICIENCY_STEPS)[
            "winrate_vs_weak"
        ],
        "auc": _normalized_auc(records),
        "final": final,
        "peak": peak,
        "regression": peak - final,
        "min_entropy": records[-1]["min_entropy"],
        "engaged": records[-1]["ever_engaged"],
        "stage1": "pass",
        "wall": wall,
    }


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4 encoder ablation gate.")
    parser.add_argument(
        "--encoders",
        choices=["flat", "structured"],
        nargs="+",
        default=["flat", "structured"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("HEAT A4 encoder gate (Tiny-Heat, 2p, masked head, pool recipe)")
    print(
        f"encoders={args.encoders} seeds={args.seeds} "
        f"timesteps={args.timesteps} n_steps={args.n_steps} hidden={args.hidden}"
    )
    print()
    print(
        "| encoder | seed | wr@40k | AUC | final | peak | regression | "
        "min entropy | engaged? | stage1 | wall(s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    rows: list[dict[str, float | str]] = []
    for encoder in args.encoders:
        for seed in args.seeds:
            row = _run_once(encoder, seed, args)
            rows.append(row)
            engaged = float(row["engaged"])
            engaged_text = (
                "nan" if not np.isfinite(engaged) else ("yes" if engaged else "no")
            )
            print(
                f"| {encoder} | {seed} | {_fmt(float(row['wr40k']))} | "
                f"{_fmt(float(row['auc']))} | {_fmt(float(row['final']))} | "
                f"{_fmt(float(row['peak']))} | {_fmt(float(row['regression']))} | "
                f"{_fmt(float(row['min_entropy']))} | {engaged_text} | "
                f"{row['stage1']} | {float(row['wall']):.0f} |",
                flush=True,
            )

    print()
    print("### Per-encoder summary (mean +- population std)")
    print()
    print("| encoder | wr@40k | AUC | final | regression | min entropy |")
    print("|---|---|---|---|---|---|")
    for encoder in args.encoders:
        encoder_rows = [row for row in rows if row["encoder"] == encoder]

        def summary(key: str) -> str:
            values = [
                float(row[key])
                for row in encoder_rows
                if np.isfinite(float(row[key]))
            ]
            if not values:
                return "nan"
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            return f"{mean:.3f} +- {std:.3f}"

        print(
            f"| {encoder} | {summary('wr40k')} | {summary('auc')} | "
            f"{summary('final')} | {summary('regression')} | "
            f"{summary('min_entropy')} |"
        )

    print()
    print("Apply A4 G1-G5 manually from this fixed table; the script does not tune.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
