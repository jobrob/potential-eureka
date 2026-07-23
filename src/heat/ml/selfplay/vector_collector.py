"""Bounded cross-game policy batching over Direction D environments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray

from heat.agents.base import BaseAgent
from heat.engine.game import MAX_ROUNDS
from heat.ml.env import TrackSource
from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.multiseat import CollectorTiming
from heat.ml.selfplay.policy import PPOPolicy
from heat.ml.spaces import ACTION_DIM, OBS_DIM
from heat.ml.vector_env import CompletedTransition, LegacyLaneEnvironment, ReadyDecision


@dataclass(frozen=True)
class _PolicyRecord:
    """Policy outputs retained for one admitted PPO transition."""

    observation: NDArray[np.float32]
    action: int
    logp: float
    value: float
    legal_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class _CompletedSample:
    """Join one retained policy output to its exact rewarded span."""

    policy: _PolicyRecord
    transition: CompletedTransition


class ExactLaneCollector:
    """Collect exact per-seat fragments from independent legacy-engine lanes.

    The collector admits exactly ``n_steps`` policy actions in stable ready-row
    order. It then keeps the policy frozen and uses unrecorded bridge actions only
    until every admitted action has a reward and successor observation. GAE is
    computed independently for each ``(game_id, seat_id)`` stream.
    """

    def __init__(
        self,
        track: TrackSource,
        num_players: int,
        *,
        lane_count: int,
        scripted_seats: dict[int, BaseAgent] | None = None,
        margin_coef: float = 0.0,
    ) -> None:
        if num_players < 2 or num_players > 6:
            raise ValueError(f"num_players must be in 2..6, got {num_players}")
        if lane_count < 1:
            raise ValueError("lane_count must be positive")
        self._track = track
        self.num_players = num_players
        self.lane_count = lane_count
        self.scripted_seats = dict(scripted_seats or {})
        self.margin_coef = margin_coef
        self.last_games_collected = 0

    @staticmethod
    def _game_seeds(rng: np.random.Generator, count: int) -> tuple[int, ...]:
        """Draw deterministic positive engine seeds from the training RNG."""
        values = rng.integers(0, 2**31 - 1, size=count)
        return tuple(int(value) for value in values)

    @staticmethod
    def _record_completions(
        transitions: tuple[CompletedTransition, ...],
        records: dict[ReadyDecision, _PolicyRecord],
        open_recorded: set[ReadyDecision],
        samples: dict[tuple[int, int], list[_CompletedSample]],
    ) -> None:
        """Join newly completed spans to retained policy rows exactly once."""
        for transition in transitions:
            record = records.get(transition.decision)
            if record is None:
                continue
            if transition.decision not in open_recorded:
                raise RuntimeError(
                    f"recorded transition completed twice: {transition.decision}"
                )
            key = (transition.decision.game_id, transition.decision.seat_id)
            samples.setdefault(key, []).append(
                _CompletedSample(policy=record, transition=transition)
            )
            open_recorded.remove(transition.decision)

    @staticmethod
    def _endpoint_values(
        streams: list[list[_CompletedSample]],
        policy: PPOPolicy,
        device: torch.device,
        timing: CollectorTiming | None,
    ) -> list[float]:
        """Value every nonterminal or truncated stream endpoint in one batch."""
        endpoint_rows: list[NDArray[np.float32]] = []
        endpoint_masks: list[NDArray[np.bool_]] = []
        endpoint_streams: list[int] = []
        values = [0.0] * len(streams)
        for stream_index, stream in enumerate(streams):
            transition = stream[-1].transition
            successor = transition.successor_observation
            if successor is None:
                continue
            mask = transition.successor_legal_mask
            if mask is None:
                mask = np.ones(ACTION_DIM, dtype=np.bool_)
            endpoint_rows.append(successor)
            endpoint_masks.append(mask)
            endpoint_streams.append(stream_index)
        if not endpoint_rows:
            return values

        started = perf_counter() if timing is not None else 0.0
        observations = torch.as_tensor(
            np.stack(endpoint_rows), dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(
            np.stack(endpoint_masks), dtype=torch.bool, device=device
        )
        actions = torch.as_tensor(
            [int(np.flatnonzero(mask)[0]) for mask in endpoint_masks],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            _logp, endpoint_values, _entropy = policy.evaluate(
                observations, actions, masks
            )
        if timing is not None:
            timing.inference_seconds += perf_counter() - started
            timing.bootstrap_inference_calls += 1
        for stream_index, value in zip(
            endpoint_streams, endpoint_values.detach().cpu().tolist(), strict=True
        ):
            values[stream_index] = float(value)
        return values

    @classmethod
    def _build_buffers(
        cls,
        samples: dict[tuple[int, int], list[_CompletedSample]],
        policy: PPOPolicy,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
        timing: CollectorTiming | None,
    ) -> list[RolloutBuffer]:
        """Build ordered per-game/per-seat buffers and compute endpoint-safe GAE."""
        streams = [
            sorted(stream, key=lambda item: item.transition.decision.decision_index)
            for _key, stream in sorted(samples.items())
        ]
        endpoint_values = cls._endpoint_values(streams, policy, device, timing)
        buffers: list[RolloutBuffer] = []
        for stream, endpoint_value in zip(streams, endpoint_values, strict=True):
            buffer = RolloutBuffer(
                capacity=len(stream),
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                device=device,
            )
            for index, sample in enumerate(stream):
                transition = sample.transition
                reward = transition.reward
                # Time-limit truncation is represented by done=True plus a
                # successor observation. Fold its bootstrap into the final reward,
                # matching the established scalar collector convention.
                if (
                    index == len(stream) - 1
                    and transition.done
                    and transition.successor_observation is not None
                ):
                    reward += gamma * endpoint_value
                buffer.add(
                    obs=sample.policy.observation,
                    action=sample.policy.action,
                    logp=sample.policy.logp,
                    value=sample.policy.value,
                    reward=reward,
                    done=transition.done,
                    mask=sample.policy.legal_mask,
                )
            last_value = 0.0 if stream[-1].transition.done else endpoint_value
            buffer.compute_gae(last_value, gamma, gae_lambda)
            buffers.append(buffer)
        return buffers

    @staticmethod
    def _act(
        environment: LegacyLaneEnvironment,
        ready: tuple[ReadyDecision, ...],
        policy: PPOPolicy,
        device: torch.device,
        timing: CollectorTiming | None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Observe ready lanes and sample one exact cross-game policy batch."""
        encode_started = perf_counter() if timing is not None else 0.0
        batch = environment.observe(ready)
        if timing is not None:
            timing.encoding_seconds += perf_counter() - encode_started
        observations = torch.as_tensor(
            batch.observations, dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(batch.legal_masks, dtype=torch.bool, device=device)
        inference_started = perf_counter() if timing is not None else 0.0
        actions, logps, values, _entropy = policy.act(observations, masks)
        if timing is not None:
            timing.inference_seconds += perf_counter() - inference_started
            timing.record_action_call(len(ready))
            timing.live_decisions += len(ready)
        return (
            batch.observations,
            batch.legal_masks,
            actions.detach().cpu().numpy().astype(np.int64, copy=False),
            logps.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def collect(
        self,
        policy: PPOPolicy,
        n_steps: int,
        device: torch.device,
        rng: np.random.Generator,
        *,
        gamma: float,
        gae_lambda: float,
        policy_version: int,
        timing: CollectorTiming | None = None,
    ) -> tuple[list[RolloutBuffer], list[float]]:
        """Collect exactly ``n_steps`` retained actions and return GAE-ready buffers."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if timing is not None:
            timing.reset()
            timing.lane_count = self.lane_count
        environment = LegacyLaneEnvironment(
            self._track,
            scripted_seats=self.scripted_seats,
            margin_coef=self.margin_coef,
        )
        next_game_id = self.lane_count
        environment.reset(
            tuple(range(self.lane_count)),
            self._game_seeds(rng, self.lane_count),
            (self.num_players,) * self.lane_count,
        )

        records: dict[ReadyDecision, _PolicyRecord] = {}
        open_recorded: set[ReadyDecision] = set()
        samples: dict[tuple[int, int], list[_CompletedSample]] = {}
        counted_finished: set[int] = set()
        self.last_games_collected = 0

        while len(records) < n_steps:
            ready = environment.ready()
            if not ready:
                raise RuntimeError("lane pool has no ready decision before target")
            observations, masks, actions, logps, values = self._act(
                environment, ready, policy, device, timing
            )
            admitted = min(len(ready), n_steps - len(records))
            for row, identity in enumerate(ready[:admitted]):
                records[identity] = _PolicyRecord(
                    observation=observations[row].copy(),
                    action=int(actions[row]),
                    logp=float(logps[row]),
                    value=float(values[row]),
                    legal_mask=masks[row].copy(),
                )
                open_recorded.add(identity)
            if timing is not None:
                timing.recorded_action_rows += admitted
                timing.drain_action_rows += len(ready) - admitted
            transitions = environment.step(
                ready, actions, policy_version=policy_version
            )
            if timing is not None:
                timing.completed_transition_rows += len(transitions)
            self._record_completions(
                transitions, records, open_recorded, samples
            )
            finished = environment.finished_game_ids()
            newly_finished = set(finished) - counted_finished
            self.last_games_collected += len(newly_finished)
            counted_finished.update(newly_finished)
            if len(records) < n_steps and finished:
                count = len(finished)
                game_ids = tuple(range(next_game_id, next_game_id + count))
                next_game_id += count
                environment.replace_finished(
                    game_ids,
                    self._game_seeds(rng, count),
                    (self.num_players,) * count,
                )

        max_drain_rows = self.lane_count * self.num_players * 5 * MAX_ROUNDS
        while open_recorded:
            ready = environment.ready()
            if not ready:
                raise RuntimeError(
                    "lane drain reached no ready decisions with recorded spans open"
                )
            _observations, _masks, actions, _logps, _values = self._act(
                environment, ready, policy, device, timing
            )
            if timing is not None:
                timing.drain_action_rows += len(ready)
                if timing.drain_action_rows > max_drain_rows:
                    raise RuntimeError("lane drain exceeded its bounded action limit")
            transitions = environment.step(
                ready, actions, policy_version=policy_version
            )
            if timing is not None:
                timing.completed_transition_rows += len(transitions)
            self._record_completions(
                transitions, records, open_recorded, samples
            )
            finished = environment.finished_game_ids()
            newly_finished = set(finished) - counted_finished
            self.last_games_collected += len(newly_finished)
            counted_finished.update(newly_finished)

        environment.drain(policy_version=policy_version)
        completed_count = sum(len(stream) for stream in samples.values())
        if completed_count != n_steps:
            raise RuntimeError(
                f"expected {n_steps} retained transitions, completed {completed_count}"
            )
        buffers = self._build_buffers(
            samples,
            policy,
            device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            timing=timing,
        )
        return buffers, []
