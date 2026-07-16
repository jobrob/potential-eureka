"""A8 full-rules, domain-randomized small-scale self-play training.

This module lifts the proven A5 anti-collapse recipe from the fixed Tiny-Heat
bed to full generated tracks and variable 2--6-player fields.  It deliberately
keeps the successful flat observation encoder and masked action head as defaults;
failed A3/A4/A6 arms remain opt-in experiments rather than being bundled into
the Phase-1 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

import numpy as np
import torch

from heat.agents.base import BaseAgent
from heat.ml.model import resolve_device
from heat.ml.selfplay.multiseat import CollectorTiming, MultiSeatCollector
from heat.ml.selfplay.policy import PPOPolicy, build_policy
from heat.ml.selfplay.ppo import ppo_update
from heat.ml.selfplay.recipe import A5Config, EntropyController
from heat.ml.selfplay.snapshots import SnapshotPool
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
    #: Scalar remains the reference until the T2 throughput gate is passed.
    collector_mode: Literal["scalar", "phase"] = "scalar"

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
        if self.collector_mode not in {"scalar", "phase"}:
            raise ValueError(
                "collector_mode must be 'scalar' or 'phase', "
                f"got {self.collector_mode!r}"
            )


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
) -> tuple[PPOPolicy, list[dict[str, float]]]:
    """Train on generated full-rules tracks with balanced 2--6-seat sampling.

    Returns the policy plus one diagnostic record per PPO iteration.  Periodic
    skill evaluation is intentionally outside the hot loop and is run through
    A7 after training; full held-out grids are too expensive to mix into every
    small-scale update. When ``profile`` is true, each record also contains
    rollout, encoding, inference, preparation, update, and total iteration times.
    """
    if config.anchor_share > 0.0 and anchor is None:
        raise ValueError("anchor is required when anchor_share is positive")
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
    tracks = training_track_source(config, track_params)

    controller = EntropyController(config)
    pool = SnapshotPool(capacity=config.pool_capacity)
    records: list[dict[str, float]] = []
    total_steps = 0
    total_games = 0
    opponent_iterations = {"current": 0, "snapshot": 0, "anchor": 0}
    min_entropy = float("inf")
    ever_engaged = False
    emitted_checkpoints: set[int] = set()

    n_iterations = max(1, config.total_timesteps // config.n_steps)
    for iteration in range(1, n_iterations + 1):
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

        collector = MultiSeatCollector(
            tracks,
            num_players,
            scripted_seats=scripted_seats,
            margin_coef=config.margin_coef,
            collector_mode=config.collector_mode,
        )
        collector_timing = CollectorTiming() if profile else None
        rollout_started = perf_counter() if profile else 0.0
        buffers, _episode_returns = collector.collect(
            policy,
            config.n_steps,
            device,
            rng,
            gamma=config.gamma,
            timing=collector_timing,
        )
        rollout_seconds = perf_counter() - rollout_started if profile else 0.0
        games_collected = collector.last_games_collected
        total_games += games_collected
        prepare_started = perf_counter() if profile else 0.0
        batches: list[dict[str, torch.Tensor]] = []
        n_recorded = 0
        for buffer in buffers:
            if len(buffer) == 0:
                continue
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
            "entropy": mean_entropy,
            "ent_coef": controller.current_ent_coef,
            "engaged": float(controller.engaged),
            "min_entropy": min_entropy,
            "ever_engaged": float(ever_engaged),
            **losses,
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
                    "prepare_seconds": prepare_seconds,
                    "update_seconds": update_seconds,
                    "iteration_seconds": perf_counter() - iteration_started,
                }
            )
            for batch_size in range(1, 7):
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

    return policy, records
