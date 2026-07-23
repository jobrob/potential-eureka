#!/usr/bin/env python
"""A10 fresh-process exact-resume gate for the native collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

from heat.ml.selfplay.phase1 import A8Config, A8ResumeConfig, train_selfplay_a8
from heat.ml.selfplay.training_state import (
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    training_state_digest,
)
from heat.tracks.generator import TrackGenParams

if __package__:
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_native import native_source_tree_identity


_ROOT = Path(__file__).resolve().parents[1]


def _config(total_iterations: int) -> A8Config:
    """Return the small deterministic native recipe shared by all processes."""
    n_steps = 64
    return A8Config(
        total_timesteps=total_iterations * n_steps,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2, 3, 4),
        snapshot_every=1,
        pool_capacity=3,
        pool_prob=0.5,
        collector_mode="native",
        native_workers=4,
        native_ready_capacity=24,
        stage1_enabled=False,
        device="cpu",
        seed=9_010,
    )


def _worker(args: argparse.Namespace) -> int:
    """Run one isolated segment and save only after its PPO boundary drains."""
    resume = A8ResumeConfig(
        campaign_id=args.campaign_id,
        source_identity=args.source_identity,
        load_path=args.load,
        save_path=args.output,
        max_iterations=args.iterations,
    )
    _policy, records = train_selfplay_a8(
        _config(args.total_iterations), resume=resume
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "completed_iteration": int(records[-1]["iteration"]),
                "native_next_game_id": int(records[-1].get("native_next_game_id", -1)),
            }
        ),
        flush=True,
    )
    return 0


def _run_process(
    *,
    output: Path,
    load: Path | None,
    iterations: int,
    total_iterations: int,
    campaign_id: str,
    source_identity: str,
    timeout: float,
) -> None:
    """Launch one genuinely fresh interpreter with a hard timeout."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--iterations",
        str(iterations),
        "--total-iterations",
        str(total_iterations),
        "--campaign-id",
        campaign_id,
        "--source-identity",
        source_identity,
    ]
    if load is not None:
        command.extend(["--load", str(load)])
    subprocess.run(command, cwd=_ROOT, check=True, timeout=timeout)


def _load(
    path: Path,
    *,
    total_iterations: int,
    campaign_id: str,
    source_identity: str,
) -> dict[str, Any]:
    """Load a result through the production hash and recipe gates."""
    recipe = resolved_recipe(_config(total_iterations), TrackGenParams())
    return load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id=campaign_id,
        source_identity=source_identity,
        anchor_identity=None,
    )


def _verify(args: argparse.Namespace) -> int:
    """Compare uninterrupted, one-stop, and two-stop native executions."""
    started = perf_counter()
    source_identity = native_source_tree_identity()
    campaign_id = "dev-d3-a10-resume"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="d3-a10-resume-", dir=args.output.parent
    ) as raw_directory:
        directory = Path(raw_directory)
        paths = {
            "uninterrupted": directory / "uninterrupted.pt",
            "one_stop": directory / "one_stop.pt",
            "two_stop": directory / "two_stop.pt",
        }
        _run_process(
            output=paths["uninterrupted"],
            load=None,
            iterations=args.total_iterations,
            total_iterations=args.total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            timeout=args.process_timeout_seconds,
        )
        first = args.total_iterations // 2
        for load, count in (
            (None, first),
            (paths["one_stop"], args.total_iterations - first),
        ):
            _run_process(
                output=paths["one_stop"],
                load=load,
                iterations=count,
                total_iterations=args.total_iterations,
                campaign_id=campaign_id,
                source_identity=source_identity,
                timeout=args.process_timeout_seconds,
            )
        split = (1, 1, args.total_iterations - 2)
        for index, count in enumerate(split):
            _run_process(
                output=paths["two_stop"],
                load=None if index == 0 else paths["two_stop"],
                iterations=count,
                total_iterations=args.total_iterations,
                campaign_id=campaign_id,
                source_identity=source_identity,
                timeout=args.process_timeout_seconds,
            )
        states = {
            name: _load(
                path,
                total_iterations=args.total_iterations,
                campaign_id=campaign_id,
                source_identity=source_identity,
            )
            for name, path in paths.items()
        }

    digests = {name: training_state_digest(state) for name, state in states.items()}
    components = {
        component: {
            name: training_state_digest({component: state[component]})
            for name, state in states.items()
        }
        for component in (
            "policy",
            "optimizer",
            "controller",
            "snapshot_pool",
            "rng",
            "schedule",
            "progress",
            "records",
            "native_collector",
            "native_runtime_receipt",
        )
    }
    passed = len(set(digests.values())) == 1 and all(
        len(set(values.values())) == 1 for values in components.values()
    )
    result = {
        "schema_version": 1,
        "stage": "D3_A10_native_fresh_process_resume",
        "hypothesis": "Native safe-boundary resume is exact across fresh processes.",
        "confidence_before_run": 0.90,
        "status": "pass" if passed else "fail",
        "pass": passed,
        "campaign_id": campaign_id,
        "source_identity": source_identity,
        "total_iterations": args.total_iterations,
        "digests": digests,
        "component_digests": components,
        "elapsed_seconds": perf_counter() - started,
        "process_timeout_seconds": args.process_timeout_seconds,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


def _parse_args() -> argparse.Namespace:
    """Parse controller and private worker modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a10_native_resume.json"),
    )
    parser.add_argument("--load", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--total-iterations", type=int, default=4)
    parser.add_argument("--campaign-id")
    parser.add_argument("--source-identity")
    parser.add_argument("--process-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    """Dispatch the requested controller or private subprocess action."""
    args = _parse_args()
    if args.total_iterations < 3:
        raise ValueError("fresh-process resume needs at least three iterations")
    if args.worker:
        if args.iterations is None or args.campaign_id is None or args.source_identity is None:
            raise ValueError("worker identity and iteration fields are required")
        return _worker(args)
    return _verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
