"""End-to-end native CPU rollout collector for Direction D3 A9."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch

from heat.engine.game import MAX_ROUNDS
from heat.ml.env import TrackSource
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native
from heat.ml.native_env.coordinator import (
    BatchedPolicy,
    CpuPolicyCoordinator,
    ReadyBatch,
    RegisteredPolicy,
)
from heat.ml.selfplay.multiseat import CollectorTiming
from heat.ml.selfplay.policy import PPOPolicy
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.spaces import ACTION_DIM, OBS_DIM
from heat.models.game_state import GameState
from heat.models.track import Track


_KIND_CODE = {"gear": 1, "cards": 2, "react": 3, "slipstream": 4, "discard": 5}


def _policy_sha256(policy: PPOPolicy) -> str:
    """Hash the frozen live policy at native admission and collection drain."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def native_runtime_receipt(worker_count: int, ready_capacity: int) -> dict[str, object]:
    """Query the exact native build/buffer contract without admitting games."""
    from heat_native import NativePool  # type: ignore[import-untyped]

    pool = NativePool(
        {
            "row_capacity": ready_capacity,
            "worker_count": worker_count,
            "slot_capacity": worker_count,
            "queue_capacity": worker_count,
        },
        np.zeros((ready_capacity, OBS_DIM), dtype=np.float32),
    )
    try:
        return dict(pool.receipt())
    finally:
        pool.close(timeout_ms=5_000)


@dataclass
class NativeCollectorState:
    """Checkpointed deterministic manifest and ready-buffer identity state."""

    next_game_id: int = 0
    next_lease_token: int = 1
    rows_per_game_ema: float = 800.0
    completed_games: int = 0

    def export_state(self) -> dict[str, int | float]:
        """Return the safe-boundary fields required for exact continuation."""
        return {
            "next_game_id": self.next_game_id,
            "next_lease_token": self.next_lease_token,
            "rows_per_game_ema": self.rows_per_game_ema,
            "completed_games": self.completed_games,
        }

    def restore_state(self, payload: dict[str, object]) -> None:
        """Restore validated state without admitting any native game."""
        next_game_id = int(cast(Any, payload["next_game_id"]))
        next_lease_token = int(cast(Any, payload["next_lease_token"]))
        rows_per_game_ema = float(cast(Any, payload["rows_per_game_ema"]))
        completed_games = int(cast(Any, payload["completed_games"]))
        if next_game_id < 0 or next_lease_token < 1 or completed_games < 0:
            raise ValueError("native collector counters are invalid")
        if not math.isfinite(rows_per_game_ema) or rows_per_game_ema <= 0.0:
            raise ValueError("native rows-per-game estimate is invalid")
        self.next_game_id = next_game_id
        self.next_lease_token = next_lease_token
        self.rows_per_game_ema = rows_per_game_ema
        self.completed_games = completed_games


@dataclass(frozen=True)
class _Pending:
    observation: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    action: int
    logp: float
    value: float
    previous: tuple[int, int, bool]
    decision_sequence: int


@dataclass(frozen=True)
class _CompletedRow:
    game_id: int
    seat_id: int
    decision_sequence: int
    observation: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    action: int
    logp: float
    value: float
    reward: float
    done: bool


@dataclass
class _Slot:
    game_id: int
    bridge: NativeStateBridge
    boundary: dict[str, object]
    sequences: dict[int, int]
    pending: dict[int, _Pending] = field(default_factory=dict)
    returns: dict[int, float] = field(default_factory=dict)
    recorded: int = 0


@dataclass(frozen=True)
class _ReadyMeta:
    slot: _Slot
    seat_id: int
    kind: str
    decision_sequence: int
    policy_id: int
    record_for_ppo: bool
    observation: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    previous: tuple[int, int, bool]


@dataclass
class _ReadyGroup:
    slot: _Slot
    kind: str
    forced_actions: dict[int, int]
    rows: list[_ReadyMeta]


class NativeCollector:
    """Run full native games, batch policy rows, and finalize one PPO batch."""

    def __init__(
        self,
        track: Track | TrackSource,
        num_players: int,
        *,
        state: NativeCollectorState,
        worker_count: int,
        ready_capacity: int,
        sampling_seed: int,
        rows_ema_alpha: float,
        manifest_version: int = 2,
        refill_reserve_factor: float = 1.3,
        margin_coef: float = 0.0,
        scripted_seats: dict[int, SnapshotAgent] | None = None,
    ) -> None:
        if num_players < 2 or num_players > 6:
            raise ValueError("native collector supports 2..6 seats")
        if worker_count < 1:
            raise ValueError("native worker_count must be positive")
        if ready_capacity < worker_count * num_players:
            raise ValueError("native ready_capacity must hold one group per slot")
        if not 0.0 < rows_ema_alpha <= 1.0:
            raise ValueError("native rows_ema_alpha must be in (0, 1]")
        if manifest_version != 2:
            raise ValueError("continuous refill requires native manifest version 2")
        if not 1.0 <= refill_reserve_factor <= 2.0:
            raise ValueError("native refill_reserve_factor must be in [1, 2]")
        self._track_source = track
        self.num_players = num_players
        self.state = state
        self.worker_count = worker_count
        self.ready_capacity = ready_capacity
        self.sampling_seed = sampling_seed
        self.rows_ema_alpha = rows_ema_alpha
        self.manifest_version = manifest_version
        self.refill_reserve_factor = refill_reserve_factor
        self.margin_coef = margin_coef
        self.scripted_seats = dict(scripted_seats or {})
        if any(not 0 <= seat < num_players for seat in self.scripted_seats):
            raise ValueError("native scripted seat is out of range")
        self.policy_seats = [
            seat for seat in range(num_players) if seat not in self.scripted_seats
        ]
        if not self.policy_seats:
            raise ValueError("native collection requires at least one live-policy seat")
        self.last_games_collected = 0
        self.last_runtime_receipt: dict[str, object] = {}
        self.last_pool_stats: dict[str, object] = {}
        self.last_coordinator_stats: dict[str, object] = {}
        self.last_admission_batches = 0
        self.last_refill_games = 0
        self.last_peak_active_slots = 0

    def _resolve_track(self, seed: int) -> Track:
        """Resolve one deterministic Python-generated track manifest."""
        source = self._track_source
        if callable(source) and not isinstance(source, Track):
            return source(seed)
        return source

    def _new_slot(self, rng: np.random.Generator) -> _Slot:
        """Create one compact native game with a monotonic logical identity."""
        game_id = self.state.next_game_id
        self.state.next_game_id += 1
        seed = int(rng.integers(0, 2**31 - 1))
        scalar = GameState.create(self._resolve_track(seed), self.num_players, seed=seed)
        for player in scalar.players:
            player.lap = 1
        bridge = legacy_state_to_native(scalar, game_id=game_id)
        return _Slot(
            game_id=game_id,
            bridge=bridge,
            boundary=bridge.start_round(),
            sequences={seat: 0 for seat in range(self.num_players)},
            returns={seat: 0.0 for seat in self.policy_seats},
        )

    def _admit_slots(
        self,
        pool: Any,
        rng: np.random.Generator,
        count: int,
    ) -> list[_Slot]:
        """Admit and verify one deterministic batch of compact native games."""
        if count < 1:
            return []
        slots = [self._new_slot(rng) for _ in range(count)]
        expected_digests = {
            slot.game_id: slot.bridge.canonical_digest() for slot in slots
        }
        pool.admit([{"state": slot.bridge.native_handle} for slot in slots])
        admitted: dict[int, int] = {}
        while len(admitted) < len(slots):
            for result in pool.wait_completed(len(slots), 5_000):
                admitted[int(cast(Any, result["game_id"]))] = int(
                    cast(Any, result["digest"])
                )
        if admitted != expected_digests:
            raise RuntimeError("native manifest admission changed a game digest")
        self.last_admission_batches += 1
        return slots

    def _close_pending(
        self,
        slot: _Slot,
        seat: int,
        rows: list[_CompletedRow],
        *,
        done: bool,
        terminated: bool,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Close one live-policy span at its next real choice or game end."""
        pending = slot.pending.pop(seat)
        reward = slot.bridge.reward_from_fields(
            seat, pending.previous, done, terminated=terminated
        )
        if done and self.margin_coef != 0.0:
            reward += self.margin_coef * slot.bridge.terminal_margin(seat)
        if done and not terminated:
            reward += bootstrap_value
        rows.append(
            _CompletedRow(
                game_id=slot.game_id,
                seat_id=seat,
                decision_sequence=pending.decision_sequence,
                observation=pending.observation,
                legal_mask=pending.legal_mask,
                action=pending.action,
                logp=pending.logp,
                value=pending.value,
                reward=reward,
                done=done,
            )
        )
        slot.returns[seat] += reward
        slot.recorded += 1

    def _apply_group(self, group: _ReadyGroup, actions: dict[int, int]) -> None:
        """Submit one complete simultaneous or sequential native group."""
        combined = {**group.forced_actions, **actions}
        if group.kind == "gear":
            group.slot.boundary = group.slot.bridge.apply_gear_actions(combined)
        elif group.kind == "cards":
            group.slot.boundary = group.slot.bridge.apply_card_actions(combined)
        else:
            if len(combined) != 1:
                raise RuntimeError("sequential native group must contain one action")
            seat, action = next(iter(combined.items()))
            group.slot.boundary = group.slot.bridge.apply_flat_action(seat, action)

    def _prepare_group(
        self,
        slot: _Slot,
        rows: list[_CompletedRow],
        timing: CollectorTiming | None,
    ) -> _ReadyGroup:
        """Resolve forced actions and materialize only real policy decisions."""
        kind = str(slot.boundary["kind"])
        kind_code = _KIND_CODE[kind]
        if kind in {"gear", "cards"}:
            seats = [
                int(value)
                for value in cast(list[Any], slot.boundary["player_ids"])
            ]
        else:
            seats = [int(cast(Any, slot.boundary["player_id"]))]
        forced_actions: dict[int, int] = {}
        ready_rows: list[_ReadyMeta] = []
        encode_started = perf_counter() if timing is not None else 0.0
        for seat in seats:
            observation = slot.bridge.observation_for_kind(seat, kind_code)
            legal_mask = slot.bridge.legal_mask_for_kind(seat, kind_code)
            legal = np.flatnonzero(legal_mask)
            if len(legal) == 0:
                raise RuntimeError("native decision has no legal action")
            if len(legal) == 1:
                forced_actions[seat] = int(legal[0])
                continue
            record = seat in self.policy_seats
            if record and seat in slot.pending:
                self._close_pending(
                    slot, seat, rows, done=False, terminated=False
                )
            policy_id = 0 if record else 1
            ready_rows.append(
                _ReadyMeta(
                    slot=slot,
                    seat_id=seat,
                    kind=kind,
                    decision_sequence=slot.sequences[seat],
                    policy_id=policy_id,
                    record_for_ppo=record,
                    observation=observation,
                    legal_mask=legal_mask,
                    previous=slot.bridge.reward_state(seat),
                )
            )
        if timing is not None:
            timing.encoding_seconds += perf_counter() - encode_started
            if kind in {"gear", "cards"}:
                live = sum(row.record_for_ppo for row in ready_rows)
                if live:
                    timing.available_phase_count += 1
                    timing.available_phase_rows += live
        return _ReadyGroup(slot, kind, forced_actions, ready_rows)

    def _dispatch(
        self,
        groups: list[_ReadyGroup],
        coordinator: CpuPolicyCoordinator,
        policy_version: int,
        timing: CollectorTiming | None,
    ) -> None:
        """Dispatch complete groups once and scatter aligned submissions."""
        metas = [row for group in groups for row in group.rows]
        if not metas:
            for group in groups:
                self._apply_group(group, {})
            return
        if len(metas) > self.ready_capacity:
            raise RuntimeError("native ready lease exceeds configured capacity")
        observations = np.ascontiguousarray(
            np.stack([row.observation for row in metas]), dtype=np.float32
        )
        masks = np.ascontiguousarray(
            np.stack([row.legal_mask for row in metas]), dtype=np.bool_
        )
        group_tokens: list[int] = []
        group_sizes: list[int] = []
        for group in groups:
            size = len(group.rows)
            if size == 0:
                continue
            epoch = int(cast(Any, group.slot.boundary["decision_epoch"]))
            token = ((group.slot.game_id << 20) ^ epoch) & ((1 << 64) - 1)
            group_tokens.extend([token] * size)
            group_sizes.extend([size] * size)
        lease = self.state.next_lease_token
        self.state.next_lease_token += 1
        ready = ReadyBatch(
            observations=observations,
            legal_masks=masks,
            group_tokens=np.asarray(group_tokens, dtype=np.uint64),
            group_sizes=np.asarray(group_sizes, dtype=np.uint16),
            game_ids=np.asarray([row.slot.game_id for row in metas], dtype=np.uint64),
            seat_ids=np.asarray([row.seat_id for row in metas], dtype=np.int16),
            decision_sequences=np.asarray(
                [row.decision_sequence for row in metas], dtype=np.uint32
            ),
            policy_ids=np.asarray([row.policy_id for row in metas], dtype=np.uint16),
            policy_versions=np.asarray(
                [policy_version if row.policy_id == 0 else 0 for row in metas],
                dtype=np.uint32,
            ),
            value_only=np.zeros(len(metas), dtype=np.bool_),
            lease_token=lease,
        )
        inference_started = perf_counter() if timing is not None else 0.0
        submission = coordinator.dispatch(ready)
        if submission.lease_token != lease:
            raise RuntimeError("native policy submission returned a stale lease")
        if timing is not None:
            timing.inference_seconds += perf_counter() - inference_started
            timing.live_decisions += sum(row.record_for_ppo for row in metas)
            simultaneous = sum(
                row.record_for_ppo and row.kind in {"gear", "cards"}
                for row in metas
            )
            timing.simultaneous_live_decisions += simultaneous
            timing.sequential_live_decisions += (
                sum(row.record_for_ppo for row in metas) - simultaneous
            )

        offset = 0
        for group in groups:
            inferred: dict[int, int] = {}
            for meta in group.rows:
                action = int(submission.actions[offset])
                inferred[meta.seat_id] = action
                meta.slot.sequences[meta.seat_id] += 1
                if meta.record_for_ppo:
                    meta.slot.pending[meta.seat_id] = _Pending(
                        observation=meta.observation,
                        legal_mask=meta.legal_mask,
                        action=action,
                        logp=float(submission.logps[offset]),
                        value=float(submission.values[offset]),
                        previous=meta.previous,
                        decision_sequence=meta.decision_sequence,
                    )
                offset += 1
            self._apply_group(group, inferred)

    def _bootstrap_and_finish(
        self,
        slots: list[tuple[_Slot, bool]],
        rows: list[_CompletedRow],
        coordinator: CpuPolicyCoordinator,
        policy_version: int,
        gamma: float,
        timing: CollectorTiming | None,
    ) -> list[float]:
        """Close terminal/truncated spans, batching all value-only rows."""
        bootstrap_meta: list[tuple[_Slot, int, NDArray[np.float32]]] = []
        for slot, terminated in slots:
            if not terminated:
                for seat in sorted(slot.pending):
                    row = slot.bridge.bootstrap_row(seat)
                    bootstrap_meta.append(
                        (slot, seat, cast(NDArray[np.float32], row["observation"]))
                    )
        bootstrap_values: dict[tuple[int, int], float] = {}
        if bootstrap_meta:
            count = len(bootstrap_meta)
            lease = self.state.next_lease_token
            self.state.next_lease_token += 1
            ready = ReadyBatch(
                observations=np.ascontiguousarray(
                    np.stack([item[2] for item in bootstrap_meta]), dtype=np.float32
                ),
                legal_masks=np.zeros((count, ACTION_DIM), dtype=np.bool_),
                group_tokens=np.arange(count, dtype=np.uint64),
                group_sizes=np.ones(count, dtype=np.uint16),
                game_ids=np.asarray(
                    [item[0].game_id for item in bootstrap_meta], dtype=np.uint64
                ),
                seat_ids=np.asarray([item[1] for item in bootstrap_meta], dtype=np.int16),
                decision_sequences=np.asarray(
                    [item[0].sequences[item[1]] for item in bootstrap_meta],
                    dtype=np.uint32,
                ),
                policy_ids=np.zeros(count, dtype=np.uint16),
                policy_versions=np.full(count, policy_version, dtype=np.uint32),
                value_only=np.ones(count, dtype=np.bool_),
                lease_token=lease,
            )
            inference_started = perf_counter() if timing is not None else 0.0
            submission = coordinator.dispatch(ready)
            if timing is not None:
                timing.inference_seconds += perf_counter() - inference_started
                timing.bootstrap_inference_calls += 1
            for index, (slot, seat, _observation) in enumerate(bootstrap_meta):
                bootstrap_values[(slot.game_id, seat)] = gamma * float(
                    submission.values[index]
                )

        episode_returns: list[float] = []
        for slot, terminated in slots:
            for seat in sorted(slot.pending):
                self._close_pending(
                    slot,
                    seat,
                    rows,
                    done=True,
                    terminated=terminated,
                    bootstrap_value=bootstrap_values.get((slot.game_id, seat), 0.0),
                )
            episode_returns.extend(slot.returns[seat] for seat in self.policy_seats)
            alpha = self.rows_ema_alpha
            self.state.rows_per_game_ema = (
                (1.0 - alpha) * self.state.rows_per_game_ema
                + alpha * max(1, slot.recorded)
            )
            self.state.completed_games += 1
        return episode_returns

    def _finalize_batch(
        self,
        rows: list[_CompletedRow],
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Bulk-copy compact rows through the native arena and return PPO tensors."""
        from heat_native import NativeTrajectoryArena

        block_rows = 1024
        arena = NativeTrajectoryArena(block_rows, max(1, math.ceil(len(rows) / block_rows)))
        observations = np.ascontiguousarray(
            np.stack([row.observation for row in rows]), dtype=np.float32
        )
        masks = np.ascontiguousarray(
            np.stack([row.legal_mask for row in rows]), dtype=np.bool_
        )
        arena.append(
            observations,
            masks,
            np.asarray([row.game_id for row in rows], dtype=np.uint64),
            np.asarray([row.seat_id for row in rows], dtype=np.int16),
            np.asarray([row.decision_sequence for row in rows], dtype=np.uint32),
            np.zeros(len(rows), dtype=np.uint16),
            np.asarray([row.action for row in rows], dtype=np.int64),
            np.asarray([row.logp for row in rows], dtype=np.float32),
            np.asarray([row.value for row in rows], dtype=np.float32),
            np.asarray([row.reward for row in rows], dtype=np.float32),
            np.asarray([row.done for row in rows], dtype=np.bool_),
            np.ones(len(rows), dtype=np.bool_),
        )
        finalized = arena.finalize(gamma, gae_lambda)
        return {
            "obs": torch.as_tensor(finalized["obs"], device=device),
            "actions": torch.as_tensor(finalized["actions"], device=device),
            "logps": torch.as_tensor(finalized["logps"], device=device),
            "values": torch.as_tensor(finalized["values"], device=device),
            "advantages": torch.as_tensor(finalized["advantages"], device=device),
            "returns": torch.as_tensor(finalized["returns"], device=device),
            "masks": torch.as_tensor(finalized["masks"], device=device),
        }

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
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Collect deterministic complete manifests behind one policy barrier."""
        if device.type != "cpu":
            raise ValueError("A9 native collector is CPU-only")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        from heat_native import NativePool

        if timing is not None:
            timing.reset()
        before_policy = _policy_sha256(policy)
        registered: list[RegisteredPolicy] = [
            RegisteredPolicy(0, policy_version, cast(BatchedPolicy, policy))
        ]
        if self.scripted_seats:
            opponents = {id(agent): agent for agent in self.scripted_seats.values()}
            if len(opponents) != 1:
                raise ValueError("A9 native collector supports one frozen opponent")
            opponent = next(iter(opponents.values()))
            registered.append(
                RegisteredPolicy(1, 0, cast(BatchedPolicy, opponent.policy))
            )
        coordinator = CpuPolicyCoordinator(
            tuple(registered), sampling_seed=self.sampling_seed
        )
        pool = NativePool(
            {
                "row_capacity": self.ready_capacity,
                "worker_count": self.worker_count,
                "slot_capacity": self.worker_count,
                "queue_capacity": self.worker_count,
            },
            np.zeros((self.ready_capacity, OBS_DIM), dtype=np.float32),
        )
        self.last_runtime_receipt = dict(pool.receipt())
        completed_rows: list[_CompletedRow] = []
        episode_returns: list[float] = []
        games_collected = 0
        self.last_admission_batches = 0
        self.last_refill_games = 0
        self.last_peak_active_slots = 0
        try:
            active: list[_Slot] = []
            while active or len(completed_rows) < n_steps:
                if len(completed_rows) < n_steps and len(active) < self.worker_count:
                    remaining = n_steps - len(completed_rows)
                    vacancies = self.worker_count - len(active)
                    projected_active_rows = sum(
                        max(
                            0.0,
                            self.state.rows_per_game_ema
                            * self.refill_reserve_factor
                            - slot.recorded,
                        )
                        for slot in active
                    )
                    uncovered_rows = max(0.0, remaining - projected_active_rows)
                    manifested = min(
                        vacancies,
                        math.ceil(uncovered_rows / self.state.rows_per_game_ema),
                    )
                    if manifested:
                        refilling = self.last_admission_batches > 0
                        additions = self._admit_slots(pool, rng, manifested)
                        active.extend(additions)
                        if refilling:
                            self.last_refill_games += len(additions)
                        self.last_peak_active_slots = max(
                            self.last_peak_active_slots, len(active)
                        )

                finished: list[tuple[_Slot, bool]] = []
                groups: list[_ReadyGroup] = []
                for slot in sorted(active, key=lambda item: item.game_id):
                    while True:
                        kind = str(slot.boundary["kind"])
                        if kind == "game_complete":
                            finished.append((slot, True))
                            break
                        if kind == "round_complete":
                            if slot.bridge.round_num > MAX_ROUNDS:
                                finished.append((slot, False))
                                break
                            slot.boundary = slot.bridge.start_round()
                            continue
                        group = self._prepare_group(slot, completed_rows, timing)
                        if group.rows:
                            groups.append(group)
                            break
                        self._apply_group(group, {})
                if finished:
                    episode_returns.extend(
                        self._bootstrap_and_finish(
                            finished,
                            completed_rows,
                            coordinator,
                            policy_version,
                            gamma,
                            timing,
                        )
                    )
                    finished_ids = {slot.game_id for slot, _terminated in finished}
                    active = [slot for slot in active if slot.game_id not in finished_ids]
                    games_collected += len(finished)
                if groups:
                    self._dispatch(groups, coordinator, policy_version, timing)
            if _policy_sha256(policy) != before_policy:
                raise RuntimeError("live policy mutated while native games were admitted")
            self.last_games_collected = games_collected
            self.last_pool_stats = dict(pool.stats())
            self.last_coordinator_stats = coordinator.stats()
            if timing is not None:
                histogram = cast(
                    dict[int, int],
                    self.last_coordinator_stats["batch_size_histogram"],
                )
                timing.action_batch_histogram = dict(histogram)
                timing.action_inference_calls = int(
                    cast(Any, self.last_coordinator_stats["inference_calls"])
                )
                timing.action_inference_rows = int(
                    cast(Any, self.last_coordinator_stats["rows"])
                )
                timing.recorded_action_rows = len(completed_rows)
                timing.completed_transition_rows = len(completed_rows)
                timing.lane_count = self.worker_count
            batch = self._finalize_batch(
                completed_rows,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
            )
            return batch, episode_returns
        finally:
            pool.close(timeout_ms=5_000)
