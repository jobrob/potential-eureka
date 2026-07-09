#!/usr/bin/env python
"""A3 head-comparison gate experiment (Sprint A3, §4.4 / G3).

Runs the A0 G2-style learning probe for BOTH action heads at equal trunk
compute, so a human can read the resulting markdown table and make the G3 gate
call: does the feature-derived ``dotprod`` head **match or beat** the free
``masked`` head across seeds?

For each ``head`` in {masked, dotprod} and each ``seed`` in {0, 1, 2} it trains a
single-seat A0 policy on Tiny-Heat, 2 players, versus a weak ``HeuristicAgent``,
for ``--timesteps`` steps on CPU, and reports the mean episode return over the
final ``--final-window`` iterations. Per head it also reports the mean +- std of
those per-seed values and the wall time.

There is deliberately NO pass/fail logic here: the script only measures and
tabulates. The gate is a judgment call made by reading the table (overlapping
+-1 std counts as "matches"; a consistent shortfall is a FAIL -- do not tune
around it).

Usage (from the repo root)::

    PYTHONPATH=src python scripts/compare_heads_a3.py
    PYTHONPATH=src python scripts/compare_heads_a3.py --timesteps 60000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from heat.ml.selfplay import A0Config, train


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 head-comparison gate probe.")
    parser.add_argument("--timesteps", type=int, default=60_000,
                        help="Env steps per run (default: 60000).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout / PPO update (default: 2048).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds to run per head (default: 0 1 2).")
    parser.add_argument("--final-window", type=int, default=10,
                        help="Iterations averaged for the final metric (default: 10).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Shared trunk widths (equal compute; default: 256 256).")
    return parser.parse_args(argv)


def _run_once(
    head: str, seed: int, args: argparse.Namespace
) -> tuple[float, float]:
    """Train one head+seed; return ``(final_window_mean_return, wall_seconds)``."""
    from heat.agents.heuristic_agent import HeuristicAgent
    from heat.ml.selfplay.tiny_heat import tiny_heat_track

    config = A0Config(
        n_steps=args.n_steps,
        total_timesteps=args.timesteps,
        hidden_sizes=tuple(args.hidden),
        num_players=2,
        device="cpu",
        seed=seed,
        head=head,
    )

    returns: list[float] = []

    def _on_iter(_iteration: int, info: dict[str, float]) -> None:
        r = info["mean_episode_return"]
        if np.isfinite(r):
            returns.append(float(r))

    t0 = time.perf_counter()
    train(
        config,
        opponents=lambda: HeuristicAgent(),
        track=tiny_heat_track(),
        on_iteration=_on_iter,
    )
    elapsed = time.perf_counter() - t0

    window = returns[-args.final_window :] if returns else []
    final_mean = float(np.mean(window)) if window else float("nan")
    return final_mean, elapsed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    heads = ["masked", "dotprod"]

    print("=" * 68)
    print("HEAT A3 head-comparison gate probe (Tiny-Heat 2p vs weak heuristic)")
    print("=" * 68)
    print(
        f"timesteps={args.timesteps} n_steps={args.n_steps} "
        f"seeds={args.seeds} final_window={args.final_window} "
        f"hidden={args.hidden}"
    )

    # per_run[head] = list of (seed, final_mean, wall)
    per_run: dict[str, list[tuple[int, float, float]]] = {h: [] for h in heads}
    for head in heads:
        for seed in args.seeds:
            print(f"  running head={head} seed={seed} ...", flush=True)
            final_mean, wall = _run_once(head, seed, args)
            per_run[head].append((seed, final_mean, wall))
            print(
                f"    -> final-{args.final_window} mean return "
                f"{final_mean:+.4f}  ({wall:.1f}s)",
                flush=True,
            )

    # --- Per-run markdown table. ---
    print()
    print(f"### Per-run (mean episode return over final {args.final_window} iters)")
    print()
    print("| head | seed | final-window mean return | wall (s) |")
    print("|---|---|---|---|")
    for head in heads:
        for seed, final_mean, wall in per_run[head]:
            print(f"| {head} | {seed} | {final_mean:+.4f} | {wall:.1f} |")

    # --- Per-head summary table. ---
    print()
    print("### Per-head summary")
    print()
    print("| head | mean +- std | wall total (s) |")
    print("|---|---|---|")
    summary: dict[str, tuple[float, float]] = {}
    for head in heads:
        finals = [f for _s, f, _w in per_run[head] if np.isfinite(f)]
        walls = sum(w for _s, _f, w in per_run[head])
        mean = statistics.mean(finals) if finals else float("nan")
        std = statistics.pstdev(finals) if len(finals) > 1 else 0.0
        summary[head] = (mean, std)
        print(f"| {head} | {mean:+.4f} +- {std:.4f} | {walls:.1f} |")

    # --- Advisory readout of the G3 comparison (NOT authoritative). ---
    print()
    m_mean, m_std = summary["masked"]
    d_mean, d_std = summary["dotprod"]
    overlaps = (d_mean + d_std) >= (m_mean - m_std) and (m_mean + m_std) >= (
        d_mean - d_std
    )
    if d_mean >= m_mean or overlaps:
        verdict = "dotprod matches-or-beats masked (advisory PASS)"
    else:
        verdict = "dotprod below masked without overlap (advisory FAIL)"
    print(f"Advisory G3 read: {verdict}")
    print("(Human makes the final call from the table above.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
