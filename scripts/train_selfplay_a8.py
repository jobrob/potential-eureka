#!/usr/bin/env python
"""Train and evaluate the A8 full-rules domain-randomized baseline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
import time

import torch

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.ml.selfplay.checkpoint import load_policy, save_policy
from heat.ml.selfplay.eval_harness import evaluate_policy, held_out_tracks
from heat.ml.selfplay.phase1 import A8Config, train_selfplay_a8
from heat.ml.selfplay.policy import PPOPolicy
from heat.ml.selfplay.snapshots import SnapshotAgent


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for resource-count options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _legacy_strong() -> StrongHeuristicAgent:
    """Build the pre-H1 opponent retained for matched S1/S2 measurement."""
    return StrongHeuristicAgent(avoid_certain_spins=False)


def _evaluation_opponents(
    anchor: BaseAgent | None,
    *,
    include_search_candidate: bool = False,
) -> dict[str, Callable[[], BaseAgent]]:
    """Return A8 rulers, with the failed B1 search candidate opt-in only."""
    opponents: dict[str, Callable[[], BaseAgent]] = {
        "weak": HeuristicAgent,
        "strong": _legacy_strong,
        "heuristic_repaired": StrongHeuristicAgent,
    }
    if include_search_candidate:
        opponents["static_search_v1_candidate"] = StaticSearchAgent
    if anchor is not None:
        opponents["fixed_anchor"] = lambda: anchor
    return opponents


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A8 full-rules self-play baseline.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--seats", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track-base-seed", type=int, default=80_008)
    parser.add_argument(
        "--anchor-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Frozen policy checkpoint used by the opt-in S2 anchor arm.",
    )
    parser.add_argument(
        "--anchor-share",
        type=float,
        default=0.0,
        help="Share of snapshot-opponent iterations replaced by the anchor.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--collector",
        choices=["scalar", "phase", "lanes", "native"],
        default="scalar",
        help="Rollout mode (scalar reference, diagnostics, or D3 native).",
    )
    parser.add_argument(
        "--lane-count",
        type=int,
        default=8,
        help="Independent games used when --collector lanes (default: 8).",
    )
    parser.add_argument(
        "--native-workers",
        type=int,
        default=48,
        help="Native active-game slots/admission helpers (A10 default: 48).",
    )
    parser.add_argument(
        "--native-ready-capacity",
        type=int,
        default=288,
        help="Maximum complete ready rows in one native lease (A10 default: 288).",
    )
    parser.add_argument(
        "--native-refill-reserve-factor",
        type=float,
        default=1.3,
        help="Manifest-v2 unfinished-row reserve multiplier (default: 1.3).",
    )
    parser.add_argument(
        "--torch-threads",
        type=_positive_int,
        default=None,
        help="Override Torch intra-op CPU threads (native default: 8).",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=_positive_int,
        default=None,
        help="Override Torch inter-op CPU threads (native default: 2).",
    )
    parser.add_argument("--save", type=str, default=None, metavar="PATH")
    parser.add_argument("--checkpoint-dir", type=str, default=None, metavar="DIR")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 300_000],
    )
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-tracks", type=int, default=5)
    parser.add_argument("--eval-seats", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument(
        "--eval-search-candidate",
        action="store_true",
        help="Include the B1-rejected StaticSearchV1 candidate as an extra ruler.",
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Print one training row every N iterations (default: 1).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Measure rollout, encoding, inference, preparation, and PPO time.",
    )
    return parser.parse_args(argv)


def _resolve_torch_threads(args: argparse.Namespace) -> tuple[int | None, int | None]:
    """Apply the adopted A10 CPU runtime only to native collection."""
    resolved_torch_threads = args.torch_threads
    resolved_torch_interop_threads = args.torch_interop_threads
    if args.collector == "native":
        resolved_torch_threads = resolved_torch_threads or 8
        resolved_torch_interop_threads = resolved_torch_interop_threads or 2
    return resolved_torch_threads, resolved_torch_interop_threads


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved_torch_threads, resolved_torch_interop_threads = (
        _resolve_torch_threads(args)
    )
    if resolved_torch_threads is not None:
        torch.set_num_threads(resolved_torch_threads)
    if resolved_torch_interop_threads is not None:
        torch.set_num_interop_threads(resolved_torch_interop_threads)
    config = A8Config(
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        hidden_sizes=tuple(args.hidden),
        seat_counts=tuple(args.seats),
        seed=args.seed,
        track_base_seed=args.track_base_seed,
        anchor_share=args.anchor_share,
        collector_mode=args.collector,
        lane_count=args.lane_count,
        native_workers=args.native_workers,
        native_ready_capacity=args.native_ready_capacity,
        native_refill_reserve_factor=args.native_refill_reserve_factor,
        device=args.device,
        head="masked",
        encoder="flat",
        margin_coef=0.0,
        stage1_enabled=False,
    )
    if args.anchor_share > 0.0 and args.anchor_checkpoint is None:
        raise ValueError("--anchor-checkpoint is required with --anchor-share")
    anchor = (
        SnapshotAgent(load_policy(args.anchor_checkpoint), name="S2-FixedAnchor")
        if args.anchor_checkpoint is not None
        else None
    )

    def log(_iteration: int, record: dict[str, float]) -> None:
        if _iteration % args.log_every != 0:
            return
        profile_text = ""
        if "rollout_seconds" in record:
            simulation = max(
                0.0,
                record["rollout_seconds"]
                - record["encoding_seconds"]
                - record["inference_seconds"],
            )
            profile_text = (
                f" rollout={record['rollout_seconds']:.3f}s"
                f"(sim={simulation:.3f}s encode={record['encoding_seconds']:.3f}s"
                f" infer={record['inference_seconds']:.3f}s)"
                f" calls={int(record['action_inference_calls'])}"
                f" rows={int(record['action_inference_rows'])}"
                f" prepare={record['prepare_seconds']:.3f}s"
                f" update={record['update_seconds']:.3f}s"
            )
        print(
            f"iter={int(record['iteration'])} seats={int(record['seat_count'])} "
            f"steps={int(record['steps'])} games={int(record['total_games'])} "
            f"opponents={int(record['opponent_current_iterations'])}/"
            f"{int(record['opponent_snapshot_iterations'])}/"
            f"{int(record['opponent_anchor_iterations'])} "
            f"entropy={record['entropy']:.3f} "
            f"policy_loss={record['policy_loss']:+.4f} "
            f"value_loss={record['value_loss']:.4f}{profile_text}",
            flush=True,
        )

    print(
        "A8 full-rules pilot: "
        f"timesteps={args.timesteps} n_steps={args.n_steps} seats={args.seats} "
        f"training_namespace={args.track_base_seed} seed={args.seed} "
        f"collector={args.collector} lane_count={args.lane_count} "
        f"torch_threads={torch.get_num_threads()} "
        f"torch_interop_threads={torch.get_num_interop_threads()}"
    )
    started = time.perf_counter()
    checkpoint_paths: list[tuple[int, Path]] = []
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(
        step: int, checkpoint_policy: PPOPolicy, _record: dict[str, float]
    ) -> None:
        if checkpoint_dir is None:
            return
        path = checkpoint_dir / f"step_{step}.pt"
        save_policy(checkpoint_policy, config, path)
        checkpoint_paths.append((step, path))
        print(f"checkpoint {step} -> {path}", flush=True)

    policy, records = train_selfplay_a8(
        config,
        anchor=anchor,
        on_iteration=log,
        checkpoint_steps=tuple(args.checkpoint_steps),
        on_checkpoint=checkpoint,
        profile=args.profile,
    )
    train_wall = time.perf_counter() - started
    print(f"training wall={train_wall:.1f}s iterations={len(records)}")
    if args.profile:
        totals = {
            key: sum(record[key] for record in records)
            for key in (
                "rollout_seconds",
                "encoding_seconds",
                "inference_seconds",
                "prepare_seconds",
                "update_seconds",
                "iteration_seconds",
            )
        }
        measured = totals["iteration_seconds"]
        simulation = max(
            0.0,
            totals["rollout_seconds"]
            - totals["encoding_seconds"]
            - totals["inference_seconds"],
        )
        overhead = max(
            0.0,
            measured
            - totals["rollout_seconds"]
            - totals["prepare_seconds"]
            - totals["update_seconds"],
        )

        def share(seconds: float) -> float:
            """Return a phase's percentage of measured training time."""
            return 100.0 * seconds / measured if measured else 0.0

        print(
            "profile summary: "
            f"measured={measured:.3f}s "
            f"rollout={totals['rollout_seconds']:.3f}s/{share(totals['rollout_seconds']):.1f}% "
            f"simulation={simulation:.3f}s/{share(simulation):.1f}% "
            f"encoding={totals['encoding_seconds']:.3f}s/{share(totals['encoding_seconds']):.1f}% "
            f"inference={totals['inference_seconds']:.3f}s/{share(totals['inference_seconds']):.1f}% "
            f"prepare={totals['prepare_seconds']:.3f}s/{share(totals['prepare_seconds']):.1f}% "
            f"update={totals['update_seconds']:.3f}s/{share(totals['update_seconds']):.1f}% "
            f"overhead={overhead:.3f}s/{share(overhead):.1f}%",
            flush=True,
        )
        live_decisions = sum(record["live_decisions"] for record in records)
        simultaneous = sum(
            record["simultaneous_live_decisions"] for record in records
        )
        phase_rows = sum(record["available_phase_rows"] for record in records)
        phase_count = sum(record["available_phase_count"] for record in records)
        calls = sum(record["action_inference_calls"] for record in records)
        histogram_limit = max(6, args.lane_count if args.collector == "lanes" else 1)
        batch_histogram = {
            size: int(sum(record[f"action_batch_{size}_calls"] for record in records))
            for size in range(1, histogram_limit + 1)
        }
        drain_rows = sum(record["drain_action_rows"] for record in records)
        print(
            "collector summary: "
            f"simultaneous_share="
            f"{(100.0 * simultaneous / live_decisions if live_decisions else 0.0):.1f}% "
            f"mean_available_batch="
            f"{(phase_rows / phase_count if phase_count else 0.0):.2f} "
            f"action_calls={int(calls)} live_rows={int(live_decisions)} "
            f"mean_action_batch={(live_decisions / calls if calls else 0.0):.2f} "
            f"drain_rows={int(drain_rows)} "
            f"batch_histogram={batch_histogram}",
            flush=True,
        )

    if args.save:
        save_policy(policy, config, args.save)
        print(f"saved -> {args.save}")

    heldout = held_out_tracks(args.eval_tracks)
    eval_targets: list[tuple[str, PPOPolicy]] = [
        (f"step {step}", load_policy(path)) for step, path in checkpoint_paths
    ]
    if not checkpoint_paths or checkpoint_paths[-1][0] < int(records[-1]["steps"]):
        eval_targets.append(("final", policy))
    for label, target_policy in eval_targets:
        report = evaluate_policy(
            target_policy,
            opponents=_evaluation_opponents(
                anchor,
                include_search_candidate=args.eval_search_candidate,
            ),
            seat_counts=tuple(args.eval_seats),
            splits={"heldout": heldout},
            games_per_cell=args.eval_games,
            seed=args.seed,
            device=torch.device("cpu"),
        )
        markdown = f"## {label}\n\n{report.to_markdown()}"
        print(markdown)
        if checkpoint_dir is not None:
            safe_label = label.replace(" ", "_")
            (checkpoint_dir / f"{safe_label}.eval.md").write_text(
                markdown + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
