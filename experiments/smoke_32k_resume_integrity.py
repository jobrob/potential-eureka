#!/usr/bin/env python
"""Disposable two-update fresh-process resume/integrity smoke for Question B.

Design Q03/Q07: before registering G0005, prove that snapshot use and
complete-state resume remain valid across a forced fresh-process load.

This is a ``dev-*`` smoke. It does not allocate G0005 and is not a learning
result. Defaults are intentionally tiny (CPU-friendly) so the gate can run
without a full 32K rollout; pass ``--n-steps 32768`` on the GPU/native box when
reconfirming the production recipe shape.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
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


def _config(
    *,
    total_iterations: int,
    n_steps: int,
    workers: int,
    ready_capacity: int,
    device: str,
) -> A8Config:
    """Return the small native recipe shared by all smoke processes."""
    return A8Config(
        total_timesteps=total_iterations * n_steps,
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2, 3, 4),
        snapshot_every=1,
        pool_capacity=3,
        pool_prob=0.5,
        collector_mode="native",
        native_workers=workers,
        native_ready_capacity=ready_capacity,
        stage1_enabled=False,
        device=device,
        seed=32_001,
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
        _config(
            total_iterations=args.total_iterations,
            n_steps=args.n_steps,
            workers=args.workers,
            ready_capacity=args.ready_capacity,
            device=args.device,
        ),
        resume=resume,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "completed_iteration": int(records[-1]["iteration"]),
                "steps": int(records[-1]["steps"]),
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
    n_steps: int,
    workers: int,
    ready_capacity: int,
    device: str,
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
        "--n-steps",
        str(n_steps),
        "--workers",
        str(workers),
        "--ready-capacity",
        str(ready_capacity),
        "--device",
        device,
    ]
    if load is not None:
        command.extend(["--load", str(load)])
    subprocess.run(command, cwd=_ROOT, check=True, timeout=timeout)


def _load(
    path: Path,
    *,
    total_iterations: int,
    n_steps: int,
    workers: int,
    ready_capacity: int,
    device: str,
    campaign_id: str,
    source_identity: str,
) -> dict[str, Any]:
    """Load a result through the production hash and recipe gates."""
    recipe = resolved_recipe(
        _config(
            total_iterations=total_iterations,
            n_steps=n_steps,
            workers=workers,
            ready_capacity=ready_capacity,
            device=device,
        ),
        TrackGenParams(),
    )
    return load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id=campaign_id,
        source_identity=source_identity,
        anchor_identity=None,
    )


def _verify(args: argparse.Namespace) -> int:
    """Compare uninterrupted vs two-update fresh-process native executions."""
    started = perf_counter()
    source_identity = native_source_tree_identity()
    campaign_id = "dev-32k-qb-resume-smoke"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="32k-resume-", dir=args.output.parent
    ) as raw_directory:
        directory = Path(raw_directory)
        uninterrupted = directory / "uninterrupted.pt"
        resumed = directory / "resumed.pt"
        _run_process(
            output=uninterrupted,
            load=None,
            iterations=args.total_iterations,
            total_iterations=args.total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            n_steps=args.n_steps,
            workers=args.workers,
            ready_capacity=args.ready_capacity,
            device=args.device,
            timeout=args.process_timeout_seconds,
        )
        # Two updates then fresh-process load for the remainder.
        first = 2
        second = args.total_iterations - first
        _run_process(
            output=resumed,
            load=None,
            iterations=first,
            total_iterations=args.total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            n_steps=args.n_steps,
            workers=args.workers,
            ready_capacity=args.ready_capacity,
            device=args.device,
            timeout=args.process_timeout_seconds,
        )
        _run_process(
            output=resumed,
            load=resumed,
            iterations=second,
            total_iterations=args.total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            n_steps=args.n_steps,
            workers=args.workers,
            ready_capacity=args.ready_capacity,
            device=args.device,
            timeout=args.process_timeout_seconds,
        )
        states = {
            name: _load(
                path,
                total_iterations=args.total_iterations,
                n_steps=args.n_steps,
                workers=args.workers,
                ready_capacity=args.ready_capacity,
                device=args.device,
                campaign_id=campaign_id,
                source_identity=source_identity,
            )
            for name, path in (
                ("uninterrupted", uninterrupted),
                ("two_update_resume", resumed),
            )
        }

    digests = {name: training_state_digest(state) for name, state in states.items()}
    passed = len(set(digests.values())) == 1
    result = {
        "schema_version": 1,
        "stage": "32k_question_b_two_update_fresh_process_resume",
        "question_id": "B",
        "g0005_allocated": False,
        "status": "pass" if passed else "fail",
        "pass": passed,
        "campaign_id": campaign_id,
        "source_identity": source_identity,
        "total_iterations": args.total_iterations,
        "n_steps": args.n_steps,
        "digests": digests,
        "elapsed_seconds": perf_counter() - started,
        "note": (
            "Disposable smoke only. Re-run with production n_steps/workers on the "
            "native machine before registering G0005."
        ),
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
        default=Path("runs/32k_learning_quality/resume_smoke.json"),
    )
    parser.add_argument("--load", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--total-iterations", type=int, default=4)
    parser.add_argument("--campaign-id")
    parser.add_argument("--source-identity")
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ready-capacity", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--process-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned smoke config and exit without launching processes.",
    )
    return parser.parse_args()


def main() -> int:
    """Dispatch dry-run, worker, or controller verification."""
    args = _parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "32k_question_b_two_update_fresh_process_resume",
                    "g0005_allocated": False,
                    "total_iterations": args.total_iterations,
                    "n_steps": args.n_steps,
                    "workers": args.workers,
                    "ready_capacity": args.ready_capacity,
                    "device": args.device,
                    "note": "Dry-run only; no training started.",
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.total_iterations < 3:
        raise ValueError("fresh-process resume needs at least three iterations")
    if args.worker:
        if (
            args.iterations is None
            or args.campaign_id is None
            or args.source_identity is None
        ):
            raise ValueError("worker identity and iteration fields are required")
        return _worker(args)
    return _verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
