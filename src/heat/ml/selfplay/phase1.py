"""A8 full-rules, domain-randomized small-scale self-play training.

This module lifts the proven A5 anti-collapse recipe from the fixed Tiny-Heat
bed to full generated tracks and variable 2--6-player fields.  It deliberately
keeps the successful flat observation encoder and masked action head as defaults;
failed A3/A4/A6 arms remain opt-in experiments rather than being bundled into
the Phase-1 baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from heat.agents.base import BaseAgent
from heat.ml.model import resolve_device
from heat.ml.native_env.collector import (
    NativeCollector,
    NativeCollectorState,
    native_runtime_receipt,
)
from heat.ml.spaces import (
    CARDS_OFFSET,
    CARDS_SIZE,
    DISCARD_OFFSET,
    DISCARD_SIZE,
    GEAR_OFFSET,
    GEAR_SIZE,
    REACT_OFFSET,
    REACT_SIZE,
    SLIPSTREAM_OFFSET,
    SLIPSTREAM_SIZE,
)
from heat.ml.selfplay.multiseat import CollectorTiming, MultiSeatCollector
from heat.ml.selfplay.policy import PPOPolicy, build_policy
from heat.ml.selfplay.ppo import ppo_update
from heat.ml.selfplay.recipe import A5Config, EntropyController
from heat.ml.selfplay.snapshots import SnapshotAgent, SnapshotPool
from heat.ml.selfplay.training_state import (
    TRAINING_STATE_SCHEMA,
    TrainingStateError,
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    save_training_state,
)
from heat.ml.selfplay.vector_collector import ExactLaneCollector
from heat.tracks.generator import TrackGenParams, TrackSampler, track_sampler


@dataclass
class A8Config(A5Config):
    """Configuration for A8's full-rules domain-randomized baseline.

    ``num_players`` is inherited for checkpoint/config compatibility but is not
    used for collection; :attr:`seat_counts` defines the training distribution.
    """

    #: S1's three-seed gate preferred gentler updates over A5's shared default.
    n_epochs: int = 5
    seat_counts: tuple[int, ...] = (2, 3, 4, 5, 6)
    #: Non-zero namespace: training tracks can never collide with A7 held-out.
    track_base_seed: int = 80_008
    #: Share of snapshot-opponent iterations replaced by one fixed policy.
    anchor_share: float = 0.0
    #: Scalar remains the reference until a bounded collector gate is passed.
    collector_mode: Literal["scalar", "phase", "lanes", "native"] = "scalar"
    #: Independent games used by the opt-in Direction D1 lane collector.
    lane_count: int = 8
    #: Active native slots/admission helpers resolved by the A10 refill grid.
    native_workers: int = 48
    #: Complete ready groups must fit without splitting across a policy lease.
    native_ready_capacity: int = 288
    #: Identity-keyed action-sampling namespace (a learning recipe field).
    native_sampling_seed: int = 0xD3A90001
    #: Fixed-form rows-per-game estimator coefficient for deterministic manifests.
    native_rows_ema_alpha: float = 0.25
    #: Frozen manifest formula identity stored in native checkpoints.
    native_manifest_version: int = 2
    #: Conservative unfinished-row credit used by manifest-v2 refill admission.
    native_refill_reserve_factor: float = 1.3

    def __post_init__(self) -> None:
        if not self.seat_counts:
            raise ValueError("seat_counts must not be empty")
        if len(set(self.seat_counts)) != len(self.seat_counts):
            raise ValueError("seat_counts must not contain duplicates")
        if any(seats < 2 or seats > 6 for seats in self.seat_counts):
            raise ValueError(
                f"seat_counts must all be in 2..6, got {self.seat_counts}"
            )
        if self.track_base_seed == 0:
            raise ValueError(
                "track_base_seed=0 is reserved for held-out evaluation"
            )
        if not 0.0 <= self.anchor_share <= 1.0:
            raise ValueError(
                f"anchor_share must be in [0, 1], got {self.anchor_share}"
            )
        if self.collector_mode not in {"scalar", "phase", "lanes", "native"}:
            raise ValueError(
                "collector_mode must be 'scalar', 'phase', 'lanes', or 'native', "
                f"got {self.collector_mode!r}"
            )
        if self.lane_count < 1:
            raise ValueError("lane_count must be positive")
        if self.native_workers < 1:
            raise ValueError("native_workers must be positive")
        if self.native_ready_capacity < self.native_workers * max(self.seat_counts):
            raise ValueError(
                "native_ready_capacity must hold one full group per native worker"
            )
        if not 0.0 < self.native_rows_ema_alpha <= 1.0:
            raise ValueError("native_rows_ema_alpha must be in (0, 1]")
        if self.native_manifest_version != 2:
            raise ValueError("only native_manifest_version=2 is supported")
        if not 1.0 <= self.native_refill_reserve_factor <= 2.0:
            raise ValueError("native_refill_reserve_factor must be in [1, 2]")


@dataclass(frozen=True)
class A8ResumeConfig:
    """Opt-in safe-boundary checkpoint settings for one A8 process segment."""

    campaign_id: str
    source_identity: str
    load_path: str | Path | None = None
    save_path: str | Path | None = None
    save_every_iterations: int = 0
    max_iterations: int | None = None
    anchor_identity: str | None = None

    def __post_init__(self) -> None:
        registered = re.fullmatch(r"G\d{4}-R\d{2}", self.campaign_id)
        if not self.campaign_id.startswith("dev-") and registered is None:
            raise ValueError("campaign_id must be dev-* or a registered G####-R## run")
        if not self.source_identity:
            raise ValueError("source_identity must not be empty")
        if self.save_every_iterations < 0:
            raise ValueError("save_every_iterations must be non-negative")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive when supplied")
        if self.save_every_iterations and self.save_path is None:
            raise ValueError("save_path is required when periodic state saving is enabled")


class SeatCountSchedule:
    """Balanced random seat-count schedule.

    Each block contains every configured seat count exactly once in a shuffled
    order.  This gives domain randomization from the first update without the
    small-run imbalance of independent uniform draws.
    """

    def __init__(
        self, seat_counts: tuple[int, ...], rng: np.random.Generator
    ) -> None:
        self._seat_counts = seat_counts
        self._rng = rng
        self._pending: list[int] = []

    def next(self) -> int:
        """Return the next seat count, refilling with a shuffled block."""
        if not self._pending:
            block = list(self._seat_counts)
            self._rng.shuffle(block)
            self._pending = block
        return self._pending.pop()

    def export_state(self) -> dict[str, Any]:
        """Return the pending balanced block and generator state."""
        return {
            "seat_counts": self._seat_counts,
            "pending": list(self._pending),
            "rng": self._rng.bit_generator.state,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a compatible pending block without drawing another value."""
        if tuple(state.get("seat_counts", ())) != self._seat_counts:
            raise ValueError("seat-count schedule does not match the resolved recipe")
        pending = state.get("pending")
        rng_state = state.get("rng")
        if not isinstance(pending, list) or not isinstance(rng_state, dict):
            raise ValueError("invalid seat-count schedule state")
        if any(value not in self._seat_counts for value in pending):
            raise ValueError("seat-count schedule contains an invalid pending value")
        self._pending = [int(value) for value in pending]
        self._rng.bit_generator.state = rng_state


# Taken-action ranges. Offsets are the frozen action-space contract.
_ACTION_PHASES: tuple[tuple[str, int, int], ...] = (
    ("gear", GEAR_OFFSET, GEAR_SIZE),
    ("cards", CARDS_OFFSET, CARDS_SIZE),
    ("react", REACT_OFFSET, REACT_SIZE),
    ("slipstream", SLIPSTREAM_OFFSET, SLIPSTREAM_SIZE),
    ("discard", DISCARD_OFFSET, DISCARD_SIZE),
)


def _phase_rollout_diagnostics(
    entropy: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, float]:
    """Per-phase means of the pre-update rollout entropy and legal-action count.

    A row is grouped by the phase range that contains its taken action. Entropy
    is that row's full masked-policy entropy, not a phase-conditional score.
    Legal count is the mean number of True mask entries on those rows. A phase
    with no rows reports zeros so every record has the same keys.
    """
    legal_count = masks.to(dtype=torch.float32).sum(dim=-1)
    diagnostics: dict[str, float] = {}
    for name, offset, size in _ACTION_PHASES:
        selected = (actions >= offset) & (actions < offset + size)
        rows = int(selected.sum().item())
        if rows == 0:
            diagnostics[f"rollout_entropy_{name}"] = 0.0
            diagnostics[f"rollout_legal_count_{name}"] = 0.0
        else:
            diagnostics[f"rollout_entropy_{name}"] = float(
                entropy[selected].mean().item()
            )
            diagnostics[f"rollout_legal_count_{name}"] = float(
                legal_count[selected].mean().item()
            )
        diagnostics[f"rollout_rows_{name}"] = float(rows)
    return diagnostics


def training_track_source(
    config: A8Config, params: TrackGenParams | None = None
) -> TrackSampler:
    """Build A8's full-rules, held-out-disjoint training track sampler."""
    source = track_sampler(params or TrackGenParams(), base_seed=config.track_base_seed)
    assert isinstance(source, TrackSampler)
    return source


def train_selfplay_a8(
    config: A8Config,
    *,
    track_params: TrackGenParams | None = None,
    anchor: BaseAgent | None = None,
    on_iteration: object = None,
    checkpoint_steps: tuple[int, ...] = (),
    on_checkpoint: object = None,
    profile: bool = False,
    resume: A8ResumeConfig | None = None,
    should_stop: object = None,
) -> tuple[PPOPolicy, list[dict[str, float]]]:
    """Train on generated full-rules tracks with balanced 2--6-seat sampling.

    Returns the policy plus one diagnostic record per PPO iteration.  Periodic
    skill evaluation is intentionally outside the hot loop and is run through
    A7 after training; full held-out grids are too expensive to mix into every
    small-scale update. When ``profile`` is true, each record also contains
    rollout, encoding, inference, preparation, update, and total iteration times.
    ``resume`` opts into D0's full-state checkpoint at completed PPO boundaries;
    omitting it preserves the original in-memory training path. ``should_stop``
    may request a graceful stop after an iteration; the completed boundary is
    checkpointed before the function returns.
    """
    if config.anchor_share > 0.0 and anchor is None:
        raise ValueError("anchor is required when anchor_share is positive")
    if (
        resume is not None
        and config.anchor_share > 0.0
        and resume.anchor_identity is None
    ):
        raise ValueError("anchor_identity is required to resume an anchored recipe")
    device = torch.device(resolve_device(config.device))
    # Seed BEFORE constructing the policy.  A5 inherited the older order that
    # seeded after initialization; A8 scale curves need the configured seed to
    # cover initial weights as well as rollout/update sampling.
    if config.seed is not None:
        torch.manual_seed(config.seed)
    policy = build_policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    rng = np.random.default_rng(config.seed)
    domain_seed = (config.seed or 0) ^ 0xA8A8A8
    schedule = SeatCountSchedule(
        config.seat_counts, np.random.default_rng(domain_seed)
    )
    resolved_track_params = track_params or TrackGenParams()
    tracks = training_track_source(config, resolved_track_params)

    controller = EntropyController(config)
    pool = SnapshotPool(capacity=config.pool_capacity)
    records: list[dict[str, float]] = []
    total_steps = 0
    total_games = 0
    opponent_iterations = {"current": 0, "snapshot": 0, "anchor": 0}
    min_entropy = float("inf")
    ever_engaged = False
    emitted_checkpoints: set[int] = set()
    native_state = NativeCollectorState()

    recipe = resolved_recipe(config, resolved_track_params)
    recipe_digest = recipe_sha256(recipe)
    start_iteration = 0
    if resume is not None and resume.load_path is not None:
        payload = load_training_state(
            resume.load_path,
            expected_recipe_sha256=recipe_digest,
            campaign_id=resume.campaign_id,
            source_identity=resume.source_identity,
            anchor_identity=resume.anchor_identity,
        )
        if payload.get("resolved_device") != str(device):
            raise TrainingStateError(
                "resolved device changed across resume: "
                f"checkpoint={payload.get('resolved_device')!r} runtime={str(device)!r}"
            )
        try:
            policy.load_state_dict(payload["policy"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            controller_state = payload["controller"]
            if not isinstance(controller_state, dict):
                raise ValueError("invalid entropy controller state")
            controller.current_ent_coef = float(controller_state["current_ent_coef"])
            controller.engaged = bool(controller_state["engaged"])
            pool_state = payload["snapshot_pool"]
            if not isinstance(pool_state, dict):
                raise ValueError("invalid snapshot pool state")
            pool.restore_state(pool_state, lambda: build_policy(config))
            schedule_state = payload["schedule"]
            if not isinstance(schedule_state, dict):
                raise ValueError("invalid schedule state")
            schedule.restore_state(schedule_state)
            progress = payload["progress"]
            if not isinstance(progress, dict):
                raise ValueError("invalid progress state")
            start_iteration = int(progress["iteration"])
            if int(progress["policy_version"]) != start_iteration:
                raise ValueError("policy version does not match completed iteration")
            if int(progress["next_iteration"]) != start_iteration + 1:
                raise ValueError("next iteration does not follow completed iteration")
            total_steps = int(progress["total_steps"])
            total_games = int(progress["total_games"])
            opponent_iterations = {
                key: int(progress["opponent_iterations"][key])
                for key in ("current", "snapshot", "anchor")
            }
            min_entropy = float(progress["min_entropy"])
            ever_engaged = bool(progress["ever_engaged"])
            emitted_checkpoints = {
                int(value) for value in progress["emitted_checkpoints"]
            }
            loaded_records = payload["records"]
            if not isinstance(loaded_records, list) or not all(
                isinstance(record, dict) for record in loaded_records
            ):
                raise ValueError("invalid diagnostic records")
            records = [
                {str(key): float(value) for key, value in record.items()}
                for record in loaded_records
            ]
            rng_state = payload["rng"]
            if not isinstance(rng_state, dict):
                raise ValueError("invalid RNG state")
            rng.bit_generator.state = rng_state["training_numpy"]
            torch.set_rng_state(rng_state["torch_cpu"])
            cuda_state = rng_state["torch_cuda"]
            if cuda_state is not None:
                if not torch.cuda.is_available():
                    raise ValueError("checkpoint requires CUDA RNG state")
                torch.cuda.set_rng_state_all(cuda_state)
            if config.collector_mode == "native":
                native_payload = payload["native_collector"]
                if not isinstance(native_payload, dict):
                    raise ValueError("invalid native collector checkpoint state")
                native_state.restore_state(native_payload)
                expected_native_receipt = native_runtime_receipt(
                    config.native_workers, config.native_ready_capacity
                )
                if payload.get("native_runtime_receipt") != expected_native_receipt:
                    raise ValueError("native runtime receipt changed across resume")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise TrainingStateError(
                f"invalid nested training state in {resume.load_path}"
            ) from exc

    n_iterations = max(1, config.total_timesteps // config.n_steps)
    end_iteration = n_iterations
    if resume is not None and resume.max_iterations is not None:
        end_iteration = min(n_iterations, start_iteration + resume.max_iterations)
    for iteration in range(start_iteration + 1, end_iteration + 1):
        iteration_started = perf_counter() if profile else 0.0
        num_players = schedule.next()
        scripted_seats: dict[int, BaseAgent] | None = None
        opponent_source = "current"
        if len(pool) > 0 and rng.random() < config.pool_prob:
            seat = int(rng.integers(num_players))
            if anchor is not None and rng.random() < config.anchor_share:
                scripted_seats = {seat: anchor}
                opponent_source = "anchor"
            else:
                scripted_seats = {seat: pool.sample(rng)}
                opponent_source = "snapshot"
        opponent_iterations[opponent_source] += 1

        collector_timing = CollectorTiming() if profile else None
        rollout_started = perf_counter() if profile else 0.0
        native_batch: dict[str, torch.Tensor] | None = None
        native_collector: NativeCollector | None = None
        if config.collector_mode == "native":
            native_scripted: dict[int, SnapshotAgent] = {}
            for seat, agent in (scripted_seats or {}).items():
                if not isinstance(agent, SnapshotAgent):
                    raise ValueError(
                        "native collection requires policy-backed SnapshotAgent opponents"
                    )
                native_scripted[seat] = agent
            native_collector = NativeCollector(
                tracks,
                num_players,
                state=native_state,
                worker_count=config.native_workers,
                ready_capacity=config.native_ready_capacity,
                sampling_seed=config.native_sampling_seed,
                rows_ema_alpha=config.native_rows_ema_alpha,
                manifest_version=config.native_manifest_version,
                refill_reserve_factor=config.native_refill_reserve_factor,
                scripted_seats=native_scripted,
                margin_coef=config.margin_coef,
            )
            native_batch, _episode_returns = native_collector.collect(
                policy,
                config.n_steps,
                device,
                rng,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                policy_version=iteration - 1,
                timing=collector_timing,
            )
            games_collected = native_collector.last_games_collected
        elif config.collector_mode == "lanes":
            lane_collector = ExactLaneCollector(
                tracks,
                num_players,
                lane_count=config.lane_count,
                scripted_seats=scripted_seats,
                margin_coef=config.margin_coef,
            )
            buffers, _episode_returns = lane_collector.collect(
                policy,
                config.n_steps,
                device,
                rng,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                policy_version=iteration - 1,
                timing=collector_timing,
            )
            games_collected = lane_collector.last_games_collected
        else:
            scalar_collector = MultiSeatCollector(
                tracks,
                num_players,
                scripted_seats=scripted_seats,
                margin_coef=config.margin_coef,
                collector_mode=config.collector_mode,
            )
            buffers, _episode_returns = scalar_collector.collect(
                policy,
                config.n_steps,
                device,
                rng,
                gamma=config.gamma,
                timing=collector_timing,
            )
            games_collected = scalar_collector.last_games_collected
        rollout_seconds = perf_counter() - rollout_started if profile else 0.0
        total_games += games_collected
        prepare_started = perf_counter() if profile else 0.0
        if native_batch is not None:
            batch = native_batch
            n_recorded = int(batch["actions"].shape[0])
        else:
            batches: list[dict[str, torch.Tensor]] = []
            n_recorded = 0
            for buffer in buffers:
                if len(buffer) == 0:
                    continue
                if config.collector_mode != "lanes":
                    buffer.compute_gae(0.0, config.gamma, config.gae_lambda)
                batches.append(buffer.get())
                n_recorded += len(buffer)
            if not batches:  # pragma: no cover - a completed game always records
                continue
            batch = {
                key: torch.cat([part[key] for part in batches], dim=0)
                for key in batches[0]
            }

        with torch.no_grad():
            _logp, _value, entropy = policy.evaluate(
                batch["obs"], batch["actions"], batch["masks"]
            )
            mean_entropy = float(entropy.mean().item())
            phase_diagnostics = _phase_rollout_diagnostics(
                entropy, batch["actions"], batch["masks"]
            )
        min_entropy = min(min_entropy, mean_entropy)
        controller.update(mean_entropy)
        ever_engaged = ever_engaged or controller.engaged
        iter_config = replace(config, ent_coef=controller.current_ent_coef)
        prepare_seconds = perf_counter() - prepare_started if profile else 0.0
        update_started = perf_counter() if profile else 0.0
        losses = ppo_update(policy, optimizer, batch, iter_config)
        update_seconds = perf_counter() - update_started if profile else 0.0
        total_steps += n_recorded

        if iteration == 1 or iteration % config.snapshot_every == 0:
            pool.push(policy, f"iter{iteration}")

        record: dict[str, float] = {
            "iteration": float(iteration),
            "steps": float(total_steps),
            "seat_count": float(num_players),
            "n_recorded": float(n_recorded),
            "games": float(games_collected),
            "total_games": float(total_games),
            "opponent_current_iterations": float(opponent_iterations["current"]),
            "opponent_snapshot_iterations": float(opponent_iterations["snapshot"]),
            "opponent_anchor_iterations": float(opponent_iterations["anchor"]),
            "ent_coef": controller.current_ent_coef,
            "engaged": float(controller.engaged),
            "ever_engaged": float(ever_engaged),
            **losses,
            # After losses so an optimizer diagnostic cannot replace these.
            "rollout_entropy": mean_entropy,
            "min_entropy": min_entropy,
            **phase_diagnostics,
        }
        if profile:
            assert collector_timing is not None
            record.update(
                {
                    "rollout_seconds": rollout_seconds,
                    "encoding_seconds": collector_timing.encoding_seconds,
                    "inference_seconds": collector_timing.inference_seconds,
                    "live_decisions": float(collector_timing.live_decisions),
                    "simultaneous_live_decisions": float(
                        collector_timing.simultaneous_live_decisions
                    ),
                    "sequential_live_decisions": float(
                        collector_timing.sequential_live_decisions
                    ),
                    "action_inference_calls": float(
                        collector_timing.action_inference_calls
                    ),
                    "action_inference_rows": float(
                        collector_timing.action_inference_rows
                    ),
                    "bootstrap_inference_calls": float(
                        collector_timing.bootstrap_inference_calls
                    ),
                    "available_phase_count": float(
                        collector_timing.available_phase_count
                    ),
                    "available_phase_rows": float(
                        collector_timing.available_phase_rows
                    ),
                    "recorded_action_rows": float(
                        collector_timing.recorded_action_rows or n_recorded
                    ),
                    "drain_action_rows": float(collector_timing.drain_action_rows),
                    "completed_transition_rows": float(
                        collector_timing.completed_transition_rows or n_recorded
                    ),
                    "lane_count": float(
                        collector_timing.lane_count
                        if config.collector_mode in {"lanes", "native"}
                        else 1
                    ),
                    "mean_action_batch": (
                        collector_timing.action_inference_rows
                        / collector_timing.action_inference_calls
                        if collector_timing.action_inference_calls
                        else 0.0
                    ),
                    "max_action_batch": float(
                        max(collector_timing.action_batch_histogram, default=0)
                    ),
                    "rollout_overshoot": float(n_recorded - config.n_steps),
                    "native_next_game_id": float(native_state.next_game_id),
                    "native_rows_per_game_ema": native_state.rows_per_game_ema,
                    "native_admission_batches": float(
                        native_collector.last_admission_batches
                        if native_collector is not None
                        else 0
                    ),
                    "native_refill_games": float(
                        native_collector.last_refill_games
                        if native_collector is not None
                        else 0
                    ),
                    "native_peak_active_slots": float(
                        native_collector.last_peak_active_slots
                        if native_collector is not None
                        else 0
                    ),
                    "prepare_seconds": prepare_seconds,
                    "update_seconds": update_seconds,
                    "iteration_seconds": perf_counter() - iteration_started,
                }
            )
            histogram_limit = (
                config.native_ready_capacity
                if config.collector_mode == "native"
                else max(6, config.lane_count)
            )
            for batch_size in range(1, histogram_limit + 1):
                record[f"action_batch_{batch_size}_calls"] = float(
                    collector_timing.action_batch_histogram.get(batch_size, 0)
                )
        records.append(record)
        for threshold in checkpoint_steps:
            if total_steps >= threshold and threshold not in emitted_checkpoints:
                emitted_checkpoints.add(threshold)
                record["checkpoint_step"] = float(threshold)
                if callable(on_checkpoint):
                    on_checkpoint(threshold, policy, record)
        if callable(on_iteration):
            on_iteration(iteration, record)

        stop_requested = bool(
            should_stop(iteration, record) if callable(should_stop) else False
        )

        if resume is not None and resume.save_path is not None:
            periodic = (
                resume.save_every_iterations > 0
                and iteration % resume.save_every_iterations == 0
            )
            if periodic or iteration == end_iteration or stop_requested:
                checkpoint_payload: dict[str, Any] = {
                    "schema_version": TRAINING_STATE_SCHEMA,
                    "recipe": recipe,
                    "recipe_sha256": recipe_digest,
                    "campaign_id": resume.campaign_id,
                    "source_identity": resume.source_identity,
                    "anchor_identity": resume.anchor_identity,
                    "resolved_device": str(device),
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "controller": {
                        "current_ent_coef": controller.current_ent_coef,
                        "engaged": controller.engaged,
                    },
                    "snapshot_pool": pool.export_state(),
                    "schedule": schedule.export_state(),
                    "rng": {
                        "training_numpy": rng.bit_generator.state,
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    },
                    "progress": {
                        "iteration": iteration,
                        "next_iteration": iteration + 1,
                        "policy_version": iteration,
                        "total_steps": total_steps,
                        "total_games": total_games,
                        "opponent_iterations": dict(opponent_iterations),
                        "min_entropy": min_entropy,
                        "ever_engaged": ever_engaged,
                        "emitted_checkpoints": sorted(emitted_checkpoints),
                    },
                    "records": records,
                }
                if config.collector_mode == "native":
                    checkpoint_payload["native_collector"] = (
                        native_state.export_state()
                    )
                    checkpoint_payload["native_runtime_receipt"] = (
                        native_runtime_receipt(
                            config.native_workers, config.native_ready_capacity
                        )
                    )
                save_training_state(resume.save_path, checkpoint_payload)
        if stop_requested:
            break

    return policy, records
