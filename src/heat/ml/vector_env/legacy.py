"""Exact multi-game adapter around the legacy Python engine for Direction D1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from heat.agents.base import BaseAgent
from heat.engine.driver import Decision, RoundDriver, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.env import TrackSource
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import ACTION_DIM, OBS_DIM, step_reward, terminal_margin
from heat.ml.vector_env.protocol import (
    CompletedTransition,
    PolicyBatch,
    ReadyDecision,
)


@dataclass(frozen=True)
class _PendingSpan:
    """Action state retained until the same seat next needs policy input."""

    decision: ReadyDecision
    action: int
    previous_state: GameState
    policy_version: int


@dataclass
class _LegacyLane:
    """One independently seeded scalar game paused at a policy boundary."""

    game_id: int
    state: GameState
    driver: RoundDriver
    decision_index: int = 0
    ready_decision: Decision | None = None
    ready_identity: ReadyDecision | None = None
    pending: dict[int, _PendingSpan] = field(default_factory=dict)
    finished: bool = False


class LegacyLaneEnvironment:
    """Expose independent legacy-engine games through the D0 batched protocol.

    Each lane advances automatic, forced, and scripted work until it reaches one
    live policy decision. Calling :meth:`step` applies at most one action per lane
    and advances that lane to its next live decision or terminal state. Rule logic
    remains entirely inside :func:`run_round_driver`.
    """

    def __init__(
        self,
        track: TrackSource,
        *,
        scripted_seats: dict[int, BaseAgent] | None = None,
        margin_coef: float = 0.0,
    ) -> None:
        self._track_source = track
        self._scripted_seats = dict(scripted_seats or {})
        self._margin_coef = margin_coef
        self._lanes: dict[int, _LegacyLane] = {}

    @staticmethod
    def _episode_flags(state: GameState) -> tuple[bool, bool]:
        """Return the scalar collector's termination and truncation flags."""
        terminated = state.is_game_over
        truncated = not terminated and state.round_num > MAX_ROUNDS
        return terminated, truncated

    def _resolve_track(self, seed: int) -> Track:
        """Resolve a fixed track or invoke the deterministic track sampler."""
        source = self._track_source
        if callable(source) and not isinstance(source, Track):
            return source(seed)
        return source

    def reset(
        self,
        game_ids: tuple[int, ...],
        seeds: tuple[int, ...],
        seat_counts: tuple[int, ...],
    ) -> tuple[ReadyDecision, ...]:
        """Replace all lanes and advance each new game to its first live choice."""
        lanes = self._build_lanes(game_ids, seeds, seat_counts)
        self._lanes = lanes
        return self.ready()

    def ready(self) -> tuple[ReadyDecision, ...]:
        """Return current live choices in stable game-identity order."""
        identities = (
            lane.ready_identity
            for lane in self._lanes.values()
            if lane.ready_identity is not None
        )
        return tuple(sorted(identity for identity in identities if identity is not None))

    def finished_game_ids(self) -> tuple[int, ...]:
        """Return terminal lanes in stable identity order."""
        return tuple(sorted(
            game_id for game_id, lane in self._lanes.items() if lane.finished
        ))

    def replace_finished(
        self,
        game_ids: tuple[int, ...],
        seeds: tuple[int, ...],
        seat_counts: tuple[int, ...],
    ) -> tuple[ReadyDecision, ...]:
        """Atomically replace every terminal lane with one fresh game."""
        finished = self.finished_game_ids()
        if len(game_ids) != len(finished):
            raise ValueError(
                "replacement count must equal finished lane count: "
                f"finished={len(finished)}, replacements={len(game_ids)}"
            )
        active_ids = set(self._lanes) - set(finished)
        if active_ids.intersection(game_ids):
            raise ValueError("replacement game_ids must not collide with active lanes")
        replacements = self._build_lanes(game_ids, seeds, seat_counts)
        for game_id in finished:
            del self._lanes[game_id]
        self._lanes.update(replacements)
        return self.ready()

    def observe(self, decisions: tuple[ReadyDecision, ...]) -> PolicyBatch:
        """Encode validated ready identities without advancing any lane."""
        rows: list[NDArray[np.float32]] = []
        masks: list[NDArray[np.bool_]] = []
        for identity in decisions:
            lane, decision = self._validated_ready(identity)
            rows.append(encode_observation(lane.state, decision.player_id, decision))
            masks.append(legal_action_mask(decision, lane.state))
        observations = (
            np.stack(rows).astype(np.float32, copy=False)
            if rows
            else np.empty((0, OBS_DIM), dtype=np.float32)
        )
        legal_masks = (
            np.stack(masks).astype(np.bool_, copy=False)
            if masks
            else np.empty((0, ACTION_DIM), dtype=np.bool_)
        )
        return PolicyBatch(
            decisions=decisions,
            observations=observations,
            legal_masks=legal_masks,
        )

    def step(
        self,
        decisions: tuple[ReadyDecision, ...],
        actions: NDArray[np.int64],
        *,
        policy_version: int,
    ) -> tuple[CompletedTransition, ...]:
        """Apply validated rows once and advance their lanes to stable boundaries."""
        if actions.shape != (len(decisions),):
            raise ValueError(
                f"actions must have shape ({len(decisions)},), got {actions.shape}"
            )
        if len({identity.game_id for identity in decisions}) != len(decisions):
            raise ValueError("step accepts at most one ready decision per game")
        self._validate_policy_version(policy_version)

        prepared: list[tuple[_LegacyLane, Decision, ReadyDecision, int, object]] = []
        for row, identity in enumerate(decisions):
            lane, decision = self._validated_ready(identity)
            action = int(actions[row])
            mask = legal_action_mask(decision, lane.state)
            if action < 0 or action >= ACTION_DIM or not bool(mask[action]):
                raise ValueError(f"illegal flat action {action} for {identity}")
            decoded = decode_legal_action(decision, lane.state, action)
            prepared.append((lane, decision, identity, action, decoded))

        completed: list[CompletedTransition] = []
        for lane, decision, identity, action, decoded in prepared:
            seat = decision.player_id
            if seat in lane.pending:  # pragma: no cover - guarded by advance logic
                raise RuntimeError(f"seat {seat} already has a pending transition")
            lane.pending[seat] = _PendingSpan(
                decision=identity,
                action=action,
                previous_state=lane.state.clone(reseed=0),
                policy_version=policy_version,
            )
            lane.ready_decision = None
            lane.ready_identity = None
            lane.decision_index += 1
            completed.extend(self._advance_lane(lane, decoded))
        return tuple(completed)

    def drain(self, *, policy_version: int) -> tuple[CompletedTransition, ...]:
        """Confirm the lanes are at stable policy boundaries.

        ``step`` already advances all automatic work before returning. The D1
        collector owns bridge-action draining because selecting those actions
        requires the frozen policy; therefore no additional transition can be
        completed here without admitting policy input.
        """
        self._validate_policy_version(policy_version)
        return ()

    def semantic_snapshot(self, game_id: int) -> dict[str, object]:
        """Return the D0 canonical state for one lane."""
        # Avoid importing the eager selfplay package while vector_env itself is
        # still initializing; the oracle is only needed by this diagnostic.
        from heat.ml.selfplay.semantic_contract import canonical_game_state

        try:
            lane = self._lanes[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown game_id {game_id}") from exc
        return canonical_game_state(lane.state)

    def _build_lanes(
        self,
        game_ids: tuple[int, ...],
        seeds: tuple[int, ...],
        seat_counts: tuple[int, ...],
    ) -> dict[int, _LegacyLane]:
        """Construct and advance a validated set of new lanes."""
        if not (len(game_ids) == len(seeds) == len(seat_counts)):
            raise ValueError("game_ids, seeds, and seat_counts must have equal length")
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("game_ids must be unique")
        if any(seats < 2 or seats > 6 for seats in seat_counts):
            raise ValueError("seat_counts must all be in 2..6")
        if any(seat < 0 for seat in self._scripted_seats):
            raise ValueError("scripted seat ids must be non-negative")
        for seats in seat_counts:
            if any(seat >= seats for seat in self._scripted_seats):
                raise ValueError(f"scripted seat is out of range for {seats}-seat lane")
            if len(self._scripted_seats) >= seats:
                raise ValueError("each lane must contain at least one policy seat")

        lanes: dict[int, _LegacyLane] = {}
        for game_id, seed, seats in zip(game_ids, seeds, seat_counts, strict=True):
            state = GameState.create(self._resolve_track(seed), seats, seed=seed)
            # Mirror Game.__init__ and the existing scalar self-play collector.
            for player in state.players:
                player.lap = 1
            lane = _LegacyLane(
                game_id=game_id,
                state=state,
                driver=run_round_driver(state),
            )
            self._advance_lane(lane)
            lanes[game_id] = lane
        return lanes

    def _validated_ready(
        self, identity: ReadyDecision
    ) -> tuple[_LegacyLane, Decision]:
        """Resolve a current identity or reject it before state mutation."""
        try:
            lane = self._lanes[identity.game_id]
        except KeyError as exc:
            raise ValueError(f"unknown game_id {identity.game_id}") from exc
        if lane.ready_identity != identity or lane.ready_decision is None:
            raise ValueError(f"stale or non-ready decision identity: {identity}")
        return lane, lane.ready_decision

    def _validate_policy_version(self, policy_version: int) -> None:
        """Reject a PPO update while any legacy lane still holds old-policy work."""
        mismatches = sorted(
            {
                pending.policy_version
                for lane in self._lanes.values()
                for pending in lane.pending.values()
                if pending.policy_version != policy_version
            }
        )
        if mismatches:
            raise ValueError(
                "cannot cross policy versions with pending transitions: "
                f"held={mismatches}, requested={policy_version}"
            )

    def _complete_at_decision(
        self, lane: _LegacyLane, decision: Decision
    ) -> CompletedTransition | None:
        """Close this seat's prior span at its next live policy observation."""
        pending = lane.pending.pop(decision.player_id, None)
        if pending is None:
            return None
        reward = step_reward(
            pending.previous_state,
            lane.state,
            decision.player_id,
            done=False,
            terminated=False,
            reward_mode="race",
            shaping_weight=0.0,
            spinout_weight=0.0,
        )
        successor = encode_observation(lane.state, decision.player_id, decision)
        successor_mask = legal_action_mask(decision, lane.state)
        return CompletedTransition(
            decision=pending.decision,
            action=pending.action,
            reward=reward,
            done=False,
            successor_observation=successor,
            successor_legal_mask=successor_mask,
            policy_version=pending.policy_version,
        )

    def _finish_lane(
        self, lane: _LegacyLane, *, terminated: bool, truncated: bool
    ) -> list[CompletedTransition]:
        """Close all remaining seat spans at a terminal or time-limit boundary."""
        completed: list[CompletedTransition] = []
        for seat in sorted(lane.pending):
            pending = lane.pending[seat]
            reward = step_reward(
                pending.previous_state,
                lane.state,
                seat,
                done=True,
                terminated=terminated,
                reward_mode="race",
                shaping_weight=0.0,
                spinout_weight=0.0,
            )
            if self._margin_coef != 0.0:
                reward += self._margin_coef * terminal_margin(lane.state, seat)
            successor = (
                encode_observation(lane.state, seat, None) if truncated else None
            )
            completed.append(
                CompletedTransition(
                    decision=pending.decision,
                    action=pending.action,
                    reward=reward,
                    done=True,
                    successor_observation=successor,
                    successor_legal_mask=None,
                    policy_version=pending.policy_version,
                )
            )
        lane.pending.clear()
        lane.ready_decision = None
        lane.ready_identity = None
        lane.finished = True
        return completed

    def _advance_lane(
        self, lane: _LegacyLane, send_value: object = None
    ) -> list[CompletedTransition]:
        """Advance one lane until its next live choice or episode boundary."""
        completed: list[CompletedTransition] = []
        while True:
            terminated, truncated = self._episode_flags(lane.state)
            if terminated or truncated:
                completed.extend(
                    self._finish_lane(
                        lane, terminated=terminated, truncated=truncated
                    )
                )
                return completed
            try:
                decision = lane.driver.send(send_value)
            except StopIteration:
                terminated, truncated = self._episode_flags(lane.state)
                if terminated or truncated:
                    completed.extend(
                        self._finish_lane(
                            lane, terminated=terminated, truncated=truncated
                        )
                    )
                    return completed
                lane.driver = run_round_driver(lane.state)
                send_value = None
                continue

            seat = decision.player_id
            if seat in self._scripted_seats:
                send_value = opponent_action(
                    self._scripted_seats[seat], decision, lane.state
                )
                continue
            forced = forced_action(decision, lane.state)
            if forced is not NO_FORCED:
                send_value = forced
                continue

            transition = self._complete_at_decision(lane, decision)
            if transition is not None:
                completed.append(transition)
            lane.ready_decision = decision
            lane.ready_identity = ReadyDecision(
                game_id=lane.game_id,
                seat_id=seat,
                decision_index=lane.decision_index,
            )
            return completed
