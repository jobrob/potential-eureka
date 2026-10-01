#!/usr/bin/env python
"""Question B scaffolding: native 32K + codec v4 learning-quality campaign shell.

Decision locked: Question B trains on the live codec v4 observation with the
native 32,768-row collector. This script resolves the frozen recipe, plans the
three sequential seed runs, and can optionally drive ``train_selfplay_a8``.

Authorization boundary (do not weaken):
* G0005 is NOT allocated here. ``experiments/policy_registry/`` is never written.
* Default mode is ``--dry-run``: print the resolved plan and exit.
* ``--execute`` still uses disposable ``dev-32k-qb-*`` campaign IDs only. A real
  registered pilot requires a separate, explicit G0005 registry edit by James
  after the recipe and source receipt are captured.
* Do not launch the full three-seed 1M pilot without that registration and an
  explicit go-ahead; the hard caps below bound a runaway job only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from heat.ml.selfplay.phase1 import A8Config, A8ResumeConfig, train_selfplay_a8
from heat.ml.selfplay.training_state import (
    recipe_sha256,
    resolved_recipe,
)
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.tracks.generator import TrackGenParams

if __package__:
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_native import native_source_tree_identity


# ---------------------------------------------------------------------------
# Frozen Question B contract (design Q02 / Q03). Keep literals in one place.
# ---------------------------------------------------------------------------

QUESTION_ID = "B"
QUESTION_LABEL = "native_32k_codec_v4"
CODEC_PIN = 4
PARENT_GENERATION_FOR_LINEAGE = "G0002"
# Next unused registry ID — NOT allocated by this script.
PROVISIONAL_GENERATION_ID = "G0005"

RUN_SEEDS = (0, 1, 2)
N_STEPS = 32_768
MAX_UPDATES = 40
STOP_ACTUAL_ROWS = 1_000_000
MILESTONE_THRESHOLDS = (300_000, 500_000, 750_000, 1_000_000)
SAVE_EVERY_UPDATES = 5
PER_RUN_TIME_CAP_SECONDS = 15 * 60

NATIVE_WORKERS = 48
NATIVE_READY_CAPACITY = 288
NATIVE_REFILL_RESERVE = 1.30
TORCH_THREADS = 8
TORCH_INTEROP_THREADS = 2

DEFAULT_OUT_DIR = Path("runs/32k_learning_quality")


def question_b_recipe_fields() -> dict[str, Any]:
    """Return the human-readable Question B recipe fields for receipts."""
    if CODEC_VERSION != CODEC_PIN:
        raise RuntimeError(
            f"Question B pins codec v{CODEC_PIN}, but live CODEC_VERSION="
            f"{CODEC_VERSION}; refuse to plan a mismatched campaign"
        )
    return {
        "question_id": QUESTION_ID,
        "question_label": QUESTION_LABEL,
        "provisional_generation_id": PROVISIONAL_GENERATION_ID,
        "parent_generation_id": PARENT_GENERATION_FOR_LINEAGE,
        "g0005_allocated": False,
        "policy_contract": {
            "codec_version": CODEC_PIN,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
        },
        "collector": "native",
        "native_workers": NATIVE_WORKERS,
        "native_ready_capacity": NATIVE_READY_CAPACITY,
        "native_refill_reserve_factor": NATIVE_REFILL_RESERVE,
        "n_steps": N_STEPS,
        "max_updates": MAX_UPDATES,
        "stop_actual_rows": STOP_ACTUAL_ROWS,
        "milestone_thresholds": list(MILESTONE_THRESHOLDS),
        "ppo": {
            "n_epochs": 5,
            "batch_size": 256,
            "learning_rate": 0.0003,
            "gamma": 0.999,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "entropy": {
            "ent_coef": 0.01,
            "entropy_floor": 0.40,
            "ent_scale_up": 1.5,
            "ent_scale_down": 0.98,
            "ent_coef_max": 0.10,
        },
        "self_play": {
            "pool_prob": 0.5,
            "pool_capacity": 5,
            "snapshot_every": 1,
            "anchor_share": 0.0,
        },
        "training_domain": {
            "seat_counts": [2, 3, 4, 5, 6],
            "track_base_seed": 80_008,
            "margin_coef": 0.0,
        },
        "run_seeds": list(RUN_SEEDS),
        "note": (
            "Thirty 32K updates are not the same learning schedule as G0002 "
            "(~1K updates of 1,024). Snapshot spacing and entropy reaction change."
        ),
    }


def build_question_b_config(
    *,
    seed: int,
    device: str = "cpu",
    n_steps: int = N_STEPS,
    max_updates: int = MAX_UPDATES,
    native_workers: int = NATIVE_WORKERS,
    native_ready_capacity: int = NATIVE_READY_CAPACITY,
) -> A8Config:
    """Build the frozen native Question B ``A8Config`` for one seed."""
    fields = question_b_recipe_fields()
    ppo = fields["ppo"]
    entropy = fields["entropy"]
    self_play = fields["self_play"]
    domain = fields["training_domain"]
    return A8Config(
        total_timesteps=max_updates * n_steps,
        n_steps=n_steps,
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(entropy["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        learning_rate=float(ppo["learning_rate"]),
        hidden_sizes=(256, 256),
        head="masked",
        encoder="flat",
        seed=seed,
        device=device,
        entropy_floor=float(entropy["entropy_floor"]),
        ent_scale_up=float(entropy["ent_scale_up"]),
        ent_scale_down=float(entropy["ent_scale_down"]),
        ent_coef_max=float(entropy["ent_coef_max"]),
        snapshot_every=int(self_play["snapshot_every"]),
        pool_capacity=int(self_play["pool_capacity"]),
        pool_prob=float(self_play["pool_prob"]),
        margin_coef=float(domain["margin_coef"]),
        stage1_enabled=False,
        seat_counts=tuple(domain["seat_counts"]),
        track_base_seed=int(domain["track_base_seed"]),
        anchor_share=float(self_play["anchor_share"]),
        collector_mode="native",
        native_workers=native_workers,
        native_ready_capacity=native_ready_capacity,
        native_refill_reserve_factor=NATIVE_REFILL_RESERVE,
    )


def disposable_campaign_id(seed: int) -> str:
    """Return a non-registry ``dev-*`` run ID for scaffolding / smoke work."""
    return f"dev-32k-qb-r{seed:02d}"


def planned_run_paths(out_dir: Path, seed: int) -> dict[str, Path]:
    """Return the per-seed artifact layout used by execute and dry-run plans."""
    run_dir = out_dir / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "state": run_dir / "training_state.pt",
        "receipt": run_dir / "run_receipt.json",
        "milestones": run_dir / "milestones",
        "heartbeat": run_dir / "heartbeat.json",
    }


def resolve_plan(
    *,
    out_dir: Path,
    device: str = "cpu",
    seeds: tuple[int, ...] = RUN_SEEDS,
) -> dict[str, Any]:
    """Resolve recipe + identities without allocating G0005 or touching disk."""
    recipe = question_b_recipe_fields()
    config = build_question_b_config(seed=seeds[0], device=device)
    resolved = resolved_recipe(config, TrackGenParams())
    source_identity = native_source_tree_identity()
    runs = []
    for seed in seeds:
        paths = planned_run_paths(out_dir, seed)
        runs.append(
            {
                "seed": seed,
                "campaign_id": disposable_campaign_id(seed),
                "provisional_agent_id": f"{PROVISIONAL_GENERATION_ID}-R{seed:02d}",
                "paths": {key: str(path) for key, path in paths.items()},
            }
        )
    return {
        "schema_version": 1,
        "stage": "32k_learning_quality_question_b",
        "question_id": QUESTION_ID,
        "g0005_allocated": False,
        "registry_write_allowed": False,
        "recipe": recipe,
        "resolved_a8_recipe": resolved,
        "resolved_a8_recipe_sha256": recipe_sha256(resolved),
        "source_identity": source_identity,
        "runs": runs,
        "stop_rule": {
            "first_crossing_actual_rows": STOP_ACTUAL_ROWS,
            "max_updates": MAX_UPDATES,
            "per_run_time_cap_seconds": PER_RUN_TIME_CAP_SECONDS,
            "recovery_every_updates": SAVE_EVERY_UPDATES,
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary sibling so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _configure_torch_threads() -> None:
    """Apply the frozen 8/2 CPU thread shape used by the A10 soak arm."""
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)


def _run_one_seed(
    *,
    seed: int,
    out_dir: Path,
    device: str,
    source_identity: str,
    n_steps: int,
    max_updates: int,
    native_workers: int,
    native_ready_capacity: int,
    time_cap_seconds: float,
) -> dict[str, Any]:
    """Train one disposable Question B seed until the first-crossing stop."""
    paths = planned_run_paths(out_dir, seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["milestones"].mkdir(parents=True, exist_ok=True)
    config = build_question_b_config(
        seed=seed,
        device=device,
        n_steps=n_steps,
        max_updates=max_updates,
        native_workers=native_workers,
        native_ready_capacity=native_ready_capacity,
    )
    campaign_id = disposable_campaign_id(seed)
    started = perf_counter()
    emitted: dict[int, dict[str, Any]] = {}

    def on_checkpoint(threshold: int, policy: object, record: dict[str, float]) -> None:
        """Persist the first-crossing milestone policy under a stable name."""
        del policy  # weights live in the full training-state save below
        path = paths["milestones"] / f"cross_{threshold}.json"
        payload = {
            "threshold": threshold,
            "actual_steps": int(record["steps"]),
            "iteration": int(record["iteration"]),
            "rollout_entropy": float(record.get("rollout_entropy", float("nan"))),
            "campaign_id": campaign_id,
            "seed": seed,
        }
        _atomic_json(path, payload)
        emitted[threshold] = payload

    def on_iteration(iteration: int, record: dict[str, float]) -> None:
        """Heartbeat after every completed update; stall detection is external."""
        _atomic_json(
            paths["heartbeat"],
            {
                "campaign_id": campaign_id,
                "seed": seed,
                "iteration": iteration,
                "actual_steps": int(record["steps"]),
                "rollout_entropy": float(record.get("rollout_entropy", float("nan"))),
                "elapsed_seconds": perf_counter() - started,
            },
        )

    def should_stop(iteration: int, record: dict[str, float]) -> bool:
        """Stop on first 1M crossing, update cap, or per-run wall-clock cap."""
        if int(record["steps"]) >= STOP_ACTUAL_ROWS:
            return True
        if iteration >= max_updates:
            return True
        return (perf_counter() - started) >= time_cap_seconds

    resume = A8ResumeConfig(
        campaign_id=campaign_id,
        source_identity=source_identity,
        save_path=paths["state"],
        save_every_iterations=SAVE_EVERY_UPDATES,
    )
    _policy, records = train_selfplay_a8(
        config,
        checkpoint_steps=MILESTONE_THRESHOLDS,
        on_checkpoint=on_checkpoint,
        on_iteration=on_iteration,
        should_stop=should_stop,
        resume=resume,
        profile=True,
    )
    final = records[-1] if records else {}
    receipt = {
        "schema_version": 1,
        "question_id": QUESTION_ID,
        "campaign_id": campaign_id,
        "seed": seed,
        "g0005_allocated": False,
        "codec_version": CODEC_PIN,
        "source_identity": source_identity,
        "completed_updates": len(records),
        "actual_steps": int(final.get("steps", 0)),
        "selected_1m": STOP_ACTUAL_ROWS in emitted,
        "milestones": emitted,
        "elapsed_seconds": perf_counter() - started,
        "state_path": str(paths["state"]),
        "note": (
            "Disposable dev-* run. Register G0005 and re-run under registered "
            "IDs before treating weights as a learning-quality pilot."
        ),
    }
    _atomic_json(paths["receipt"], receipt)
    return receipt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dry-run / execute controls for the Question B campaign shell."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Artifact root (default: runs/32k_learning_quality).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Resolve and print the plan only (default).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run disposable dev-* training. Does NOT allocate G0005. "
            "Still refuses registry writes."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(RUN_SEEDS),
        help="Training seeds (default: 0 1 2).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--workers", type=int, default=NATIVE_WORKERS)
    parser.add_argument("--ready-capacity", type=int, default=NATIVE_READY_CAPACITY)
    parser.add_argument(
        "--time-cap-seconds",
        type=float,
        default=float(PER_RUN_TIME_CAP_SECONDS),
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="Optional path to write the dry-run plan JSON.",
    )
    args = parser.parse_args(argv)
    if args.execute:
        args.dry_run = False
    if args.n_steps < 1 or args.max_updates < 1:
        parser.error("--n-steps and --max-updates must be positive")
    if args.workers < 1 or args.ready_capacity < args.workers * 6:
        parser.error("ready capacity must hold a six-seat group per worker")
    return args


def main(argv: list[str] | None = None) -> int:
    """Dry-run the Question B plan, or execute disposable native training."""
    args = _parse_args(argv)
    seeds = tuple(args.seeds)
    plan = resolve_plan(out_dir=args.out_dir, device=args.device, seeds=seeds)
    plan_path = args.plan_out or (args.out_dir / "question_b_plan.json")
    if args.dry_run:
        _atomic_json(plan_path, plan)
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        print(
            "\nDRY-RUN only. G0005 was NOT allocated. No training started.\n"
            "To run disposable native training later:\n"
            "  PYTHONPATH=src python experiments/run_32k_learning_quality.py "
            "--execute --out-dir runs/32k_learning_quality\n"
            "Register G0005 in experiments/policy_registry/ only after an "
            "explicit decision and a captured source receipt.",
            flush=True,
        )
        return 0

    # Execute path: still no registry mutation.
    _configure_torch_threads()
    source_identity = plan["source_identity"]
    _atomic_json(plan_path, plan)
    receipts: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"Question B execute seed={seed} campaign={disposable_campaign_id(seed)}", flush=True)
        receipt = _run_one_seed(
            seed=seed,
            out_dir=args.out_dir,
            device=args.device,
            source_identity=source_identity,
            n_steps=args.n_steps,
            max_updates=args.max_updates,
            native_workers=args.workers,
            native_ready_capacity=args.ready_capacity,
            time_cap_seconds=args.time_cap_seconds,
        )
        receipts.append(receipt)
    summary = {
        "schema_version": 1,
        "question_id": QUESTION_ID,
        "g0005_allocated": False,
        "receipts": receipts,
    }
    _atomic_json(args.out_dir / "execute_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    from _runlog import run_main

    run_main("run_32k_learning_quality", main)
