#!/usr/bin/env python
"""Bounded T2 collector throughput benchmark (disposable dev diagnostics).

This script never saves policies or allocates a learned-policy generation.  It
measures the existing A8 recipe with profiling enabled and emits one JSON object
per run plus an aggregate verdict-friendly summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Literal

from heat.ml.selfplay.phase1 import A8Config, train_selfplay_a8

CollectorMode = Literal["scalar", "phase"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bounded T2a/T2b benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="dev-t2-collector")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--modes", choices=["scalar", "phase"], nargs="+", default=["scalar", "phase"]
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--run-cap-seconds", type=float, default=900.0)
    parser.add_argument("--campaign-cap-seconds", type=float, default=1800.0)
    return parser.parse_args(argv)


def _run_once(
    *,
    label: str,
    mode: CollectorMode,
    seed: int,
    timesteps: int,
    n_steps: int,
    heartbeat_seconds: float,
    run_cap_seconds: float,
    campaign_started: float,
    campaign_cap_seconds: float,
) -> dict[str, object]:
    """Run one profiled A8 repeat with heartbeat and hard-cap checks."""
    config = A8Config(
        total_timesteps=timesteps,
        n_steps=n_steps,
        seed=seed,
        device="cpu",
        collector_mode=mode,
    )
    run_started = time.perf_counter()
    last_heartbeat = run_started
    print(
        f"heartbeat label={label} mode={mode} seed={seed} status=started",
        flush=True,
    )

    def progress(iteration: int, record: dict[str, float]) -> None:
        """Emit bounded progress and stop if either wall-time cap is crossed."""
        nonlocal last_heartbeat
        now = time.perf_counter()
        run_elapsed = now - run_started
        campaign_elapsed = now - campaign_started
        if run_elapsed > run_cap_seconds:
            raise TimeoutError(
                f"run cap exceeded: mode={mode} seed={seed} {run_elapsed:.1f}s"
            )
        if campaign_elapsed > campaign_cap_seconds:
            raise TimeoutError(
                f"campaign cap exceeded after {campaign_elapsed:.1f}s"
            )
        if now - last_heartbeat >= heartbeat_seconds:
            print(
                f"heartbeat label={label} mode={mode} seed={seed} "
                f"iteration={iteration} steps={int(record['steps'])} "
                f"elapsed={run_elapsed:.1f}s",
                flush=True,
            )
            last_heartbeat = now

    _policy, records = train_selfplay_a8(
        config, on_iteration=progress, profile=True
    )
    wall_seconds = time.perf_counter() - run_started
    n_recorded = sum(record["n_recorded"] for record in records)
    games = sum(record["games"] for record in records)
    rollout_seconds = sum(record["rollout_seconds"] for record in records)
    inference_seconds = sum(record["inference_seconds"] for record in records)
    live_decisions = sum(record["live_decisions"] for record in records)
    simultaneous = sum(
        record["simultaneous_live_decisions"] for record in records
    )
    available_rows = sum(record["available_phase_rows"] for record in records)
    available_phases = sum(
        record["available_phase_count"] for record in records
    )
    action_calls = sum(record["action_inference_calls"] for record in records)
    histogram = {
        str(size): int(
            sum(record[f"action_batch_{size}_calls"] for record in records)
        )
        for size in range(1, 7)
    }
    result: dict[str, object] = {
        "label": label,
        "mode": mode,
        "seed": seed,
        "iterations": len(records),
        "n_recorded": int(n_recorded),
        "games": int(games),
        "wall_seconds": wall_seconds,
        "rollout_seconds": rollout_seconds,
        "rollout_transitions_per_second": (
            n_recorded / rollout_seconds if rollout_seconds else 0.0
        ),
        "rollout_games_per_second": games / rollout_seconds if rollout_seconds else 0.0,
        "inference_seconds": inference_seconds,
        "live_decisions": int(live_decisions),
        "simultaneous_live_decisions": int(simultaneous),
        "simultaneous_share": simultaneous / live_decisions if live_decisions else 0.0,
        "mean_available_batch": (
            available_rows / available_phases if available_phases else 0.0
        ),
        "action_inference_calls": int(action_calls),
        "action_inference_rows": int(
            sum(record["action_inference_rows"] for record in records)
        ),
        "bootstrap_inference_calls": int(
            sum(record["bootstrap_inference_calls"] for record in records)
        ),
        "action_batch_histogram": histogram,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _matched_summary(results: list[dict[str, object]]) -> dict[str, object]:
    """Return per-seed and median phase/scalar speed and inference ratios."""
    by_key = {(str(row["mode"]), int(row["seed"])): row for row in results}
    seeds = sorted(
        seed
        for mode, seed in by_key
        if mode == "scalar" and ("phase", seed) in by_key
    )
    pairs: list[dict[str, float | int]] = []
    for seed in seeds:
        scalar = by_key[("scalar", seed)]
        phase = by_key[("phase", seed)]
        scalar_tps = float(scalar["rollout_transitions_per_second"])
        phase_tps = float(phase["rollout_transitions_per_second"])
        scalar_inference = float(scalar["inference_seconds"])
        phase_inference = float(phase["inference_seconds"])
        pairs.append(
            {
                "seed": seed,
                "rollout_speedup": phase_tps / scalar_tps,
                "inference_wall_reduction": (
                    1.0 - phase_inference / scalar_inference
                    if scalar_inference
                    else 0.0
                ),
            }
        )
    return {
        "matched_pairs": pairs,
        "median_rollout_speedup": (
            statistics.median(float(pair["rollout_speedup"]) for pair in pairs)
            if pairs
            else None
        ),
        "median_inference_wall_reduction": (
            statistics.median(
                float(pair["inference_wall_reduction"]) for pair in pairs
            )
            if pairs
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run requested disposable repeats inside one campaign hard cap."""
    args = _parse_args(argv)
    campaign_started = time.perf_counter()
    results: list[dict[str, object]] = []
    for seed in args.seeds:
        for raw_mode in args.modes:
            mode: CollectorMode = raw_mode
            if time.perf_counter() - campaign_started > args.campaign_cap_seconds:
                raise TimeoutError("campaign cap reached before next repeat")
            results.append(
                _run_once(
                    label=args.label,
                    mode=mode,
                    seed=seed,
                    timesteps=args.timesteps,
                    n_steps=args.n_steps,
                    heartbeat_seconds=args.heartbeat_seconds,
                    run_cap_seconds=args.run_cap_seconds,
                    campaign_started=campaign_started,
                    campaign_cap_seconds=args.campaign_cap_seconds,
                )
            )
    summary = {
        "label": args.label,
        "campaign_seconds": time.perf_counter() - campaign_started,
        "results": results,
        **_matched_summary(results),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
