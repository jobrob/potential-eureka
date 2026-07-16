"""Bounded correctness and throughput screens for A8 T1 process parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.ml.policy_registry import find_registered_run
from heat.ml.selfplay.campaign import CampaignResult, CampaignRun, run_campaign
from heat.ml.selfplay.checkpoint import load_policy
from heat.ml.selfplay.eval_harness import evaluate_policy, held_out_tracks
from heat.ml.selfplay.phase1 import A8Config
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.selfplay.tiny_heat import tiny_heat_track


class _Heartbeat:
    """Print progress while one blocking evaluation mode is in flight."""

    def __init__(self, label: str, interval: float = 15.0) -> None:
        self.label = label
        self.interval = interval
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        """Emit a timestamp until the blocking mode completes."""
        while not self.stop.wait(self.interval):
            elapsed = time.perf_counter() - self.started
            print(f"heartbeat mode={self.label} wall={elapsed:.1f}s", flush=True)

    def __enter__(self) -> _Heartbeat:
        """Start heartbeat logging."""
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop heartbeat logging promptly."""
        self.stop.set()
        self.thread.join(timeout=1.0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one independently bounded T1 stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("correctness", "training", "evaluation"))
    parser.add_argument(
        "--out", type=Path, default=Path("runs/a8_t1_parallel.json")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Accepted T1b worker count used by the evaluation screen.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout <= 0.0:
        parser.error("--workers and --timeout must be positive")
    return args


def _source_provenance() -> dict[str, Any]:
    """Capture the commit and dirty flag without creating a policy generation."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_commit": commit, "dirty": bool(status.strip())}


def _write_stage(path: Path, stage: str, result: dict[str, Any]) -> None:
    """Atomically merge one stage into the disposable raw T1 artifact."""
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "experiment": "T1 parallel campaign diagnostic",
            "policy_ids": ["dev-t1-R00", "dev-t1-R01", "dev-t1-R02"],
            "comparison_worthy": False,
            "saves_checkpoints": False,
            "source": _source_provenance(),
        }
    )
    payload[stage] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _tiny_config(seed: int) -> A8Config:
    """Return T1a's tiny real-training recipe."""
    return A8Config(
        total_timesteps=16,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2,),
        snapshot_every=100,
        pool_capacity=1,
        pool_prob=0.0,
        device="cpu",
        seed=seed,
    )


def _training_config(seed: int) -> A8Config:
    """Return the frozen T1b 50k five-epoch A8 diagnostic recipe."""
    return A8Config(
        total_timesteps=50_000,
        n_steps=1024,
        n_epochs=5,
        head="masked",
        encoder="flat",
        anchor_share=0.0,
        track_base_seed=80_008,
        device="cpu",
        seed=seed,
    )


def _signature(result: CampaignResult) -> tuple[str, int, int, int, str]:
    """Project a campaign result onto fields concurrency must preserve exactly."""
    return (
        result.run_id,
        result.seed,
        result.recorded_steps,
        result.games,
        result.policy_sha256,
    )


def _progress(event: str, run_id: str | None, elapsed: float) -> None:
    """Print machine-readable campaign starts, heartbeats, and completions."""
    print(f"progress event={event} run={run_id or '-'} wall={elapsed:.1f}s", flush=True)


def _run_mode(
    runs: list[CampaignRun], workers: int, timeout: float
) -> tuple[float, list[CampaignResult]]:
    """Time one fixed-work campaign concurrency mode."""
    print(f"mode start workers={workers} jobs={len(runs)}", flush=True)
    started = time.perf_counter()
    results = run_campaign(
        runs,
        max_workers=workers,
        timeout_seconds=timeout,
        progress=_progress,
    )
    wall = time.perf_counter() - started
    print(f"mode complete workers={workers} wall={wall:.3f}s", flush=True)
    return wall, results


def _correctness(timeout: float) -> dict[str, Any]:
    """Run T1a's tiny exact-identity campaign and two-cell evaluation."""
    runs = [
        CampaignRun("dev-t1a-R01", _tiny_config(1)),
        CampaignRun("dev-t1a-R00", _tiny_config(0)),
    ]
    one_wall, one = _run_mode(runs, 1, min(timeout, 300.0))
    two_wall, two = _run_mode(runs, 2, min(timeout, 300.0))
    campaign_equal = [_signature(result) for result in one] == [
        _signature(result) for result in two
    ]
    torch.manual_seed(81)
    policy = build_policy(A0Config(hidden_sizes=(16,)))
    kwargs = {
        "opponents": {"weak": HeuristicAgent},
        "seat_counts": (2, 4),
        "splits": {"tiny": tiny_heat_track()},
        "games_per_cell": 2,
        "seed": 17,
        "device": torch.device("cpu"),
    }
    sequential = evaluate_policy(policy, **kwargs)
    rng_before = torch.random.get_rng_state().clone()
    parallel = evaluate_policy(policy, **kwargs, parallel=True, max_workers=2)
    rng_equal = torch.equal(rng_before, torch.random.get_rng_state())
    evaluation_equal = sequential == parallel
    passed = campaign_equal and evaluation_equal and rng_equal
    return {
        "passed": passed,
        "campaign_equal": campaign_equal,
        "evaluation_equal": evaluation_equal,
        "parent_rng_equal": rng_equal,
        "one_worker_wall_seconds": one_wall,
        "two_worker_wall_seconds": two_wall,
        "campaign_results": [asdict(result) for result in one],
    }


def _training(timeout: float) -> dict[str, Any]:
    """Run T1b at one, two, and three workers over identical fixed work."""
    runs = [
        CampaignRun(f"dev-t1-R{seed:02d}", _training_config(seed))
        for seed in range(3)
    ]
    modes: dict[int, tuple[float, list[CampaignResult]]] = {}
    for workers in (1, 2, 3):
        modes[workers] = _run_mode(runs, workers, timeout)
    baseline_wall, baseline_results = modes[1]
    baseline_signature = [_signature(result) for result in baseline_results]
    baseline_unit_rate = sum(r.recorded_steps for r in baseline_results) / sum(
        r.wall_seconds for r in baseline_results
    )
    rows: list[dict[str, Any]] = []
    for workers, (wall, results) in modes.items():
        exact = [_signature(result) for result in results] == baseline_signature
        unit_rate = sum(r.recorded_steps for r in results) / sum(
            r.wall_seconds for r in results
        )
        rows.append(
            {
                "workers": workers,
                "wall_seconds": wall,
                "speedup": baseline_wall / wall,
                "transitions_per_second": sum(r.recorded_steps for r in results) / wall,
                "games_per_second": sum(r.games for r in results) / wall,
                "per_worker_unit_rate_ratio": unit_rate / baseline_unit_rate,
                "exact": exact,
                "results": [asdict(result) for result in results],
            }
        )
    accepted = [
        row
        for row in rows
        if row["exact"] and row["per_worker_unit_rate_ratio"] >= 0.9
    ]
    best = max(accepted, key=lambda row: row["speedup"])
    passed = bool(best["speedup"] >= 1.5)
    return {
        "passed": passed,
        "accepted_workers": int(best["workers"]),
        "best_speedup": best["speedup"],
        "exact_all_modes": all(row["exact"] for row in rows),
        "modes": rows,
    }


def _evaluation(workers: int) -> dict[str, Any]:
    """Run T1c's frozen six-cell G0002-R00@1000K timing screen."""
    registered = find_registered_run("G0002-R00")
    policy = load_policy(str(registered["selected_checkpoint"]))
    kwargs = {
        "opponents": {
            "weak": HeuristicAgent,
            "heuristic_repaired": StrongHeuristicAgent,
        },
        "seat_counts": (2, 4, 6),
        "splits": {"validation_900000": held_out_tracks(10)},
        "games_per_cell": 20,
        "seed": 0,
        "device": torch.device("cpu"),
    }
    with _Heartbeat("evaluation-sequential"):
        started = time.perf_counter()
        sequential = evaluate_policy(policy, **kwargs)
        sequential_wall = time.perf_counter() - started
    with _Heartbeat(f"evaluation-parallel-{workers}"):
        started = time.perf_counter()
        parallel = evaluate_policy(
            policy, **kwargs, parallel=True, max_workers=workers
        )
        parallel_wall = time.perf_counter() - started
    exact = sequential == parallel
    speedup = sequential_wall / parallel_wall
    return {
        "passed": exact and speedup >= 1.5,
        "exact": exact,
        "workers": workers,
        "sequential_wall_seconds": sequential_wall,
        "parallel_wall_seconds": parallel_wall,
        "speedup": speedup,
        "policy_id": "G0002-R00@1000K",
        "cells": [asdict(cell) for cell in sequential.cells],
    }


def main(argv: list[str] | None = None) -> int:
    """Run one T1 stage, persist raw evidence, and print its gate verdict."""
    args = _parse_args(argv)
    if args.stage == "correctness":
        result = _correctness(args.timeout)
    elif args.stage == "training":
        result = _training(args.timeout)
    else:
        result = _evaluation(args.workers)
    _write_stage(args.out, args.stage, result)
    print(
        f"T1 stage={args.stage} passed={result['passed']} out={args.out}",
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
