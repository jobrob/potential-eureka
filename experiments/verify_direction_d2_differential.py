#!/usr/bin/env python
"""Bounded randomized differential gate for Direction D2 Chunk 5."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, simultaneous_decisions
from heat.engine.phases import (
    PhaseTraceCallback,
    phase_play_cards,
    step_adrenaline,
    step_replenish,
    step_reveal_and_move,
)
from heat.models.cards import Card, CardType
from heat.models.game_state import GameEvent, GameState, Phase
from heat.ml.action_codec import decode_legal_action, encode_action_index, legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.selfplay.semantic_contract import canonical_event, canonical_game_state
from heat.ml.vector_env.bridge import legacy_states_to_tensor, tensor_semantic_snapshot
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.kernels.common import TensorKernelEvent, TensorKernelResult
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    tensor_legal_action_masks,
    tensor_observations,
)
from heat.ml.vector_env.random_inputs import RecordedDrawInputs
from heat.ml.vector_env.state import TensorGameState
from heat.tracks.generator import generate_track


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODES = (
    "normal",
    "traffic",
    "stress_reshuffle",
    "clutter_replenish",
    "finish_replenish",
)


@dataclass(frozen=True)
class PreparedBatch:
    """One replayable scalar/tensor comparison batch."""

    source_states: list[GameState]
    scalar_states: list[GameState]
    scalar_choices: list[dict[int, tuple[Card, ...]]]
    game_ids: list[int]
    modes: list[str]
    tensor_state: TensorGameState
    decisions: TensorDecisionBatch
    actions: torch.Tensor
    draw_inputs: RecordedDrawInputs
    expected_events: dict[int, list[dict[str, object]]]
    react_decisions: list[Decision]
    stress_draw_count: int
    replenish_draw_count: int

    @property
    def transition_count(self) -> int:
        """Count simultaneous card choices as randomized state transitions."""
        return int(self.actions.shape[0])


class DifferentialMismatch(AssertionError):
    """One exact mismatch with the identity needed for a replay artifact."""

    def __init__(
        self,
        message: str,
        *,
        game_id: int | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(message)
        self.game_id = game_id
        self.phase = phase


def _write_json(path: Path, value: object) -> None:
    """Write one stable human-readable evidence artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_tree_identity() -> str:
    """Hash the current source inputs, including untracked D2 files."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "experiments",
            "pyproject.toml",
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    digest = hashlib.sha256()
    for relative in sorted(line for line in result.stdout.splitlines() if line):
        path = _REPO_ROOT / relative
        if path.is_file():
            digest.update(relative.replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _ensure_stress_in_hand(state: GameState, player_id: int) -> None:
    """Swap one stress card into a hand without changing zone sizes."""
    player = state.get_player(player_id)
    if any(card.card_type is CardType.STRESS for card in player.hand):
        return
    for pile in (player.deck._draw_pile, player.deck._discard_pile):
        for index, card in enumerate(pile):
            if card.card_type is CardType.STRESS:
                pile[index], player.hand[0] = player.hand[0], card
                return
    raise AssertionError("starting deck has no stress card")


def build_case(case_index: int, seed: int) -> GameState:
    """Build one deterministic varied state at the Chunk-5 cards boundary."""
    seats = 2 + case_index % 5
    mode = _MODES[case_index % len(_MODES)]
    state_seed = seed + case_index * 17
    state = GameState.create(
        generate_track(state_seed),
        seats,
        seed=state_seed + 100_000,
    )
    for player in state.players:
        player.lap = 1
        player.gear = 1

    target = state.players[0]
    target.position = 8 + case_index % max(1, state.track.length - 16)
    for opponent in state.players[1:]:
        opponent.position = (
            target.position - 4 - opponent.player_id % 3
        ) % state.track.length
        opponent.lap = 0

    if mode == "traffic" and seats >= 3:
        state.players[1].position = (target.position + 4) % state.track.length
        state.players[2].position = state.players[1].position
    elif mode == "stress_reshuffle":
        _ensure_stress_in_hand(state, target.player_id)
        target.deck._discard_pile.extend(target.deck._draw_pile)
        target.deck._draw_pile.clear()
    elif mode == "clutter_replenish":
        target.gear = 4
        playable = next(
            card for card in target.hand if card.card_type is not CardType.HEAT
        )
        target.deck._discard_pile.extend(
            card for card in target.hand if card is not playable
        )
        heat = target.heat_pool[:3]
        target.heat_pool = target.heat_pool[3:]
        target.hand = [playable, *heat]
    elif mode == "finish_replenish":
        target.position = state.track.length - 1
        target.lap = state.track.laps

    state.compute_turn_order()
    state.event_log.clear()
    return state


def _turn_start_event(
    state: GameState, player_id: int, pre_play_hand: list[str]
) -> GameEvent:
    """Build the exact driver turn-start receipt for the scalar oracle."""
    player = state.get_player(player_id)
    next_corner, distance = rules.distance_to_next_corner(state.track, player.position)
    return GameEvent(
        state.round_num,
        Phase.REVEAL_AND_MOVE,
        player_id,
        "turn_start",
        {
            "hand": pre_play_hand,
            "hand_size": len(pre_play_hand),
            "gear": player.gear,
            "heat_available": player.heat_available,
            "position": player.position,
            "next_corner_dist": distance if next_corner is not None else None,
            "next_corner_speed_limit": (
                next_corner.speed_limit if next_corner is not None else None
            ),
        },
    )


def run_scalar_to_next_react(
    state: GameState,
    choices: dict[int, tuple[Card, ...]],
    *,
    record_draws: bool,
    trace: PhaseTraceCallback | None = None,
) -> tuple[list[GameEvent], Decision, dict[int, list[list[str]]], dict[int, list[str]]]:
    """Run the scalar oracle through skipped cars and optionally record draws."""
    pre_play_hands = {
        player.player_id: [card.display_name for card in player.hand]
        for player in state.active_players
    }
    stress_records: dict[int, list[list[str]]] = defaultdict(list)
    replenish_records: dict[int, list[str]] = defaultdict(list)
    stress_closed: dict[int, bool] = defaultdict(lambda: True)

    if record_draws:
        for player in state.players:
            original_draw = player.deck.draw
            player_id = player.player_id

            def recording_draw(
                count: int = 1,
                *,
                original: Callable[[int], list[Card]] = original_draw,
                recorded_player: int = player_id,
            ) -> list[Card]:
                if state.current_phase is Phase.REVEAL_AND_MOVE:
                    if stress_closed[recorded_player]:
                        stress_records[recorded_player].append([])
                        stress_closed[recorded_player] = False
                drawn = original(count)
                if state.current_phase is Phase.REVEAL_AND_MOVE:
                    stress_records[recorded_player][-1].extend(
                        card.id for card in drawn
                    )
                    if drawn and drawn[-1].card_type is CardType.SPEED:
                        stress_closed[recorded_player] = True
                elif state.current_phase is Phase.REPLENISH:
                    replenish_records[recorded_player].extend(
                        card.id for card in drawn
                    )
                return drawn

            player.deck.draw = recording_draw  # type: ignore[method-assign]

    events = phase_play_cards(state, choices)
    if trace is not None:
        trace("cards_chosen", state)
    for player_id in list(state.turn_order):
        player = state.get_player(player_id)
        if player.finished:
            continue
        events.append(_turn_start_event(state, player_id, pre_play_hands[player_id]))
        if player.cluttered:
            player.gear = 1
            events.extend(step_replenish(state, player))
            continue
        before_reveal = len(state.event_log)
        reveal_events = step_reveal_and_move(state, player, trace=trace)
        events.extend(
            event
            for event in state.event_log[before_reveal:]
            if event.event_type == "stress_resolved"
        )
        events.extend(reveal_events)
        if player.finished:
            events.extend(step_replenish(state, player))
            continue
        events.extend(step_adrenaline(state, player))
        if trace is not None:
            trace("adrenaline", state)
        options = rules.legal_react_options(
            player, list(state.active_players), state.starting_player_count
        )
        return (
            events,
            Decision(DecisionKind.REACT, player_id, options),
            dict(stress_records),
            dict(replenish_records),
        )
    raise RuntimeError("case did not reach a real REACT decision")


def _choose_action(
    state: GameState,
    decision: Decision,
    mode: str,
    rng: np.random.Generator,
) -> int:
    """Choose a legal codec action while forcing the focal branch when needed."""
    legal = decision.legal
    assert isinstance(legal, list)
    target_id = state.turn_order[0]
    if decision.player_id == target_id and mode == "stress_reshuffle":
        chosen = next(
            play
            for play in legal
            if any(card.card_type is CardType.STRESS for card in play)
        )
    elif decision.player_id == target_id and mode == "finish_replenish":
        chosen = max(legal, key=lambda play: sum(card.value for card in play))
    else:
        chosen = legal[int(rng.integers(0, len(legal)))]
    return encode_action_index(decision, chosen)


def prepare_batch(
    start: int,
    count: int,
    seed: int,
    *,
    case_indices: list[int] | None = None,
) -> PreparedBatch:
    """Create a complete recorded scalar/tensor batch without timing setup."""
    cases = (
        list(range(start, start + count))
        if case_indices is None
        else case_indices
    )
    if len(cases) != count or len(set(cases)) != count:
        raise ValueError("case_indices must contain one unique entry per game")
    source_states = [build_case(case_index, seed) for case_index in cases]
    scalar_states = [build_case(case_index, seed) for case_index in cases]
    game_ids = [1_000_000 + case_index for case_index in cases]
    modes = [_MODES[case_index % len(_MODES)] for case_index in cases]
    tensor_state = legacy_states_to_tensor(source_states, game_ids=game_ids)

    decision_games: list[int] = []
    decision_players: list[int] = []
    actions: list[int] = []
    scalar_choices: list[dict[int, tuple[Card, ...]]] = []
    for case_index, (game_id, mode, source, scalar) in enumerate(
        zip(game_ids, modes, source_states, scalar_states, strict=True)
    ):
        resolved_case = cases[case_index]
        action_rng = np.random.default_rng(seed + resolved_case * 1_009)
        choices: dict[int, tuple[Card, ...]] = {}
        scalar_decisions = {
            decision.player_id: decision
            for decision in simultaneous_decisions(scalar, DecisionKind.CARDS)
        }
        for decision in simultaneous_decisions(source, DecisionKind.CARDS):
            action = _choose_action(source, decision, mode, action_rng)
            decoded = decode_legal_action(
                scalar_decisions[decision.player_id], scalar, action
            )
            assert isinstance(decoded, tuple)
            choices[decision.player_id] = decoded
            decision_games.append(game_id)
            decision_players.append(decision.player_id)
            actions.append(action)
        scalar_choices.append(choices)

    stress_records: dict[tuple[int, int], list[list[str]]] = {}
    replenish_records: dict[tuple[int, int], list[str]] = {}
    expected_events: dict[int, list[dict[str, object]]] = {}
    react_decisions: list[Decision] = []
    for game_id, scalar, choices in zip(
        game_ids, scalar_states, scalar_choices, strict=True
    ):
        events, react, stress, replenishments = run_scalar_to_next_react(
            scalar, choices, record_draws=True
        )
        expected_events[game_id] = [canonical_event(event) for event in events]
        react_decisions.append(react)
        stress_records.update(
            ((game_id, player_id), records)
            for player_id, records in stress.items()
        )
        replenish_records.update(
            ((game_id, player_id), records)
            for player_id, records in replenishments.items()
        )

    decisions = TensorDecisionBatch.create(
        decision_games,
        decision_players,
        [TensorDecisionKind.CARDS] * len(actions),
    )
    return PreparedBatch(
        source_states=source_states,
        scalar_states=scalar_states,
        scalar_choices=scalar_choices,
        game_ids=game_ids,
        modes=modes,
        tensor_state=tensor_state,
        decisions=decisions,
        actions=torch.tensor(actions, dtype=torch.int64),
        draw_inputs=RecordedDrawInputs.create(
            tensor_state, stress_records, replenish_records
        ),
        expected_events=expected_events,
        react_decisions=react_decisions,
        stress_draw_count=sum(
            len(sequence)
            for records in stress_records.values()
            for sequence in records
        ),
        replenish_draw_count=sum(len(records) for records in replenish_records.values()),
    )


def rebuild_scalar_replays(
    batch: PreparedBatch, seed: int
) -> tuple[list[GameState], list[dict[int, tuple[Card, ...]]]]:
    """Rebuild pristine scalar inputs carrying the batch's exact actions."""
    action_by_identity = dict(
        zip(
            zip(
                batch.decisions.game_ids.tolist(),
                batch.decisions.player_ids.tolist(),
                strict=True,
            ),
            batch.actions.tolist(),
            strict=True,
        )
    )
    states: list[GameState] = []
    choices_by_game: list[dict[int, tuple[Card, ...]]] = []
    for game_id in batch.game_ids:
        case_index = game_id - 1_000_000
        state = build_case(case_index, seed)
        choices: dict[int, tuple[Card, ...]] = {}
        for decision in simultaneous_decisions(state, DecisionKind.CARDS):
            decoded = decode_legal_action(
                decision,
                state,
                action_by_identity[(game_id, decision.player_id)],
            )
            assert isinstance(decoded, tuple)
            choices[decision.player_id] = decoded
        states.append(state)
        choices_by_game.append(choices)
    return states, choices_by_game


def _events_by_game(
    events: tuple[TensorKernelEvent, ...],
) -> dict[int, list[dict[str, object]]]:
    """Group tensor receipts by stable game identity."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        grouped[event.game_id].append(event.semantic())
    return dict(grouped)


def verify_result(batch: PreparedBatch, result: TensorKernelResult) -> None:
    """Demand exact scalar parity for state, receipts, observations, and masks."""
    for game_id, scalar in zip(batch.game_ids, batch.scalar_states, strict=True):
        actual = tensor_semantic_snapshot(result.state, game_id)
        expected = canonical_game_state(scalar)
        if actual != expected:
            raise DifferentialMismatch(
                f"state mismatch for game_id {game_id}",
                game_id=game_id,
                phase=str(actual["current_phase"]),
            )
    actual_events = _events_by_game(result.events)
    for game_id in batch.game_ids:
        if actual_events.get(game_id, []) != batch.expected_events[game_id]:
            compared = zip(
                actual_events.get(game_id, []),
                batch.expected_events[game_id],
            )
            first = next(
                (actual for actual, expected in compared if actual != expected),
                None,
            )
            raise DifferentialMismatch(
                f"ordered semantic event mismatch for game_id {game_id}",
                game_id=game_id,
                phase=str(first["phase"]) if first is not None else None,
            )
    expected_observations = np.stack(
        [
            encode_observation(state, decision.player_id, decision)
            for state, decision in zip(
                batch.scalar_states, batch.react_decisions, strict=True
            )
        ]
    )
    expected_masks = np.stack(
        [
            legal_action_mask(decision, state)
            for state, decision in zip(
                batch.scalar_states, batch.react_decisions, strict=True
            )
        ]
    )
    actual_observations = tensor_observations(
        result.state, result.next_decisions
    ).numpy()
    actual_masks = tensor_legal_action_masks(
        result.state, result.next_decisions
    ).numpy()
    for row, game_id in enumerate(batch.game_ids):
        if not np.array_equal(actual_observations[row], expected_observations[row]):
            raise DifferentialMismatch(
                f"REACT observation mismatch for game_id {game_id}",
                game_id=game_id,
                phase=Phase.ADRENALINE.value,
            )
        if not np.array_equal(actual_masks[row], expected_masks[row]):
            raise DifferentialMismatch(
                f"REACT legal-mask mismatch for game_id {game_id}",
                game_id=game_id,
                phase=Phase.ADRENALINE.value,
            )


def _assert_tensor_equal(left: TensorGameState, right: TensorGameState) -> None:
    """Compare every mutable tensor field exactly."""
    for (name, left_value), (right_name, right_value) in zip(
        left.tensor_fields(), right.tensor_fields(), strict=True
    ):
        if name != right_name or not torch.equal(left_value, right_value):
            raise AssertionError(f"tensor repeat mismatch in {name}")


def _determinism_checks(batch: PreparedBatch, result: TensorKernelResult) -> None:
    """Check repeat, resume, lane order, and focal-lane isolation once."""
    repeated = apply_cards_to_react(
        batch.tensor_state, batch.decisions, batch.actions, batch.draw_inputs
    )
    _assert_tensor_equal(result.state, repeated.state)
    if result.events != repeated.events:
        raise AssertionError("repeat event mismatch")

    buffer = __import__("io").BytesIO()
    torch.save(batch.tensor_state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    resumed = apply_cards_to_react(
        restored,
        batch.decisions,
        batch.actions,
        RecordedDrawInputs(
            batch.draw_inputs.card_ids.clone(),
            batch.draw_inputs.lengths.clone(),
            batch.draw_inputs.replenish_card_ids.clone(),
            batch.draw_inputs.replenish_lengths.clone(),
        ),
    )
    _assert_tensor_equal(result.state, resumed.state)

    action_by_identity = dict(
        zip(
            zip(
                batch.decisions.game_ids.tolist(),
                batch.decisions.player_ids.tolist(),
                strict=True,
            ),
            batch.actions.tolist(),
            strict=True,
        )
    )
    reversed_states = list(reversed(batch.source_states))
    reversed_ids = list(reversed(batch.game_ids))
    reversed_tensor = legacy_states_to_tensor(reversed_states, game_ids=reversed_ids)
    reversed_games: list[int] = []
    reversed_players: list[int] = []
    reversed_actions: list[int] = []
    for game_id, state in zip(reversed_ids, reversed_states, strict=True):
        for decision in simultaneous_decisions(state, DecisionKind.CARDS):
            reversed_games.append(game_id)
            reversed_players.append(decision.player_id)
            reversed_actions.append(action_by_identity[(game_id, decision.player_id)])
    reversed_decisions = TensorDecisionBatch.create(
        reversed_games,
        reversed_players,
        [TensorDecisionKind.CARDS] * len(reversed_actions),
    )
    stress: dict[tuple[int, int], list[list[str]]] = {}
    replenishments: dict[tuple[int, int], list[str]] = {}
    vocabulary = batch.tensor_state.card_id_vocabulary
    for lane, game_id in enumerate(batch.game_ids):
        player_count = int(batch.tensor_state.player_present[lane].sum().item())
        for player in range(player_count):
            sequences = [
                [
                    vocabulary[int(card_id) - 1]
                    for card_id in batch.draw_inputs.card_ids[
                        lane, player, stress_index, : int(length.item())
                    ].tolist()
                ]
                for stress_index, length in enumerate(
                    batch.draw_inputs.lengths[lane, player]
                )
                if int(length.item()) > 0
            ]
            if sequences:
                stress[(game_id, player)] = sequences
            replenish_length = int(
                batch.draw_inputs.replenish_lengths[lane, player].item()
            )
            if replenish_length:
                replenishments[(game_id, player)] = [
                    vocabulary[int(card_id) - 1]
                    for card_id in batch.draw_inputs.replenish_card_ids[
                        lane, player, :replenish_length
                    ].tolist()
                ]
    reversed_result = apply_cards_to_react(
        reversed_tensor,
        reversed_decisions,
        torch.tensor(reversed_actions, dtype=torch.int64),
        RecordedDrawInputs.create(reversed_tensor, stress, replenishments),
    )
    for game_id in batch.game_ids:
        if tensor_semantic_snapshot(reversed_result.state, game_id) != (
            tensor_semantic_snapshot(result.state, game_id)
        ):
            raise AssertionError(f"lane-order mismatch for game_id {game_id}")
    if _events_by_game(reversed_result.events) != batch.expected_events:
        raise AssertionError("lane-order event mismatch")

    focal_game = batch.game_ids[0]
    focal_state = legacy_states_to_tensor(
        [batch.source_states[0]], game_ids=[focal_game]
    )
    focal_rows = batch.decisions.game_ids == focal_game
    focal_decisions = TensorDecisionBatch.create(
        batch.decisions.game_ids[focal_rows].tolist(),
        batch.decisions.player_ids[focal_rows].tolist(),
        [TensorDecisionKind.CARDS] * int(focal_rows.sum().item()),
    )
    focal_stress = {
        key: value for key, value in stress.items() if key[0] == focal_game
    }
    focal_replenishments = {
        key: value for key, value in replenishments.items() if key[0] == focal_game
    }
    focal_result = apply_cards_to_react(
        focal_state,
        focal_decisions,
        batch.actions[focal_rows],
        RecordedDrawInputs.create(
            focal_state, focal_stress, focal_replenishments
        ),
    )
    if tensor_semantic_snapshot(focal_result.state, focal_game) != (
        tensor_semantic_snapshot(result.state, focal_game)
    ):
        raise AssertionError("focal lane changed when neighboring lanes were removed")


def _recorded_inputs_for_game(
    batch: PreparedBatch, lane: int
) -> dict[str, object]:
    """Decode one lane's stress and replenish records for exact replay."""
    vocabulary = batch.tensor_state.card_id_vocabulary
    stress: dict[str, list[list[str]]] = {}
    replenish: dict[str, list[str]] = {}
    player_count = int(batch.tensor_state.player_present[lane].sum().item())
    for player in range(player_count):
        player_id = int(batch.tensor_state.player_ids[lane, player].item())
        sequences: list[list[str]] = []
        for stress_index, length_value in enumerate(
            batch.draw_inputs.lengths[lane, player]
        ):
            length = int(length_value.item())
            if length:
                sequences.append(
                    [
                        vocabulary[int(card_id) - 1]
                        for card_id in batch.draw_inputs.card_ids[
                            lane, player, stress_index, :length
                        ].tolist()
                    ]
                )
        if sequences:
            stress[str(player_id)] = sequences
        replenish_length = int(
            batch.draw_inputs.replenish_lengths[lane, player].item()
        )
        if replenish_length:
            replenish[str(player_id)] = [
                vocabulary[int(card_id) - 1]
                for card_id in batch.draw_inputs.replenish_card_ids[
                    lane, player, :replenish_length
                ].tolist()
            ]
    return {"stress": stress, "replenish": replenish}


def _failure_payload(
    exc: Exception,
    batch: PreparedBatch | None,
    result: TensorKernelResult | None,
    *,
    batch_start: int,
    seed: int,
) -> dict[str, object]:
    """Build the complete smallest replay receipt available for a mismatch."""
    payload: dict[str, object] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "batch_start_game": batch_start,
        "campaign_seed": seed,
    }
    if batch is None:
        return payload
    requested_game = (
        exc.game_id if isinstance(exc, DifferentialMismatch) else None
    )
    game_id = (
        requested_game
        if requested_game in batch.game_ids
        else batch.game_ids[0]
    )
    lane = batch.game_ids.index(game_id)
    case_index = game_id - 1_000_000
    action_rows = batch.decisions.game_ids == game_id
    payload.update(
        {
            "game_id": game_id,
            "case_index": case_index,
            "mode": batch.modes[lane],
            "phase": (
                exc.phase if isinstance(exc, DifferentialMismatch) else None
            ),
            "track_seed": seed + case_index * 17,
            "game_rng_seed": seed + case_index * 17 + 100_000,
            "action_seed": seed + case_index * 1_009,
            "actions": [
                {
                    "player_id": int(player_id),
                    "flat_action": int(action),
                }
                for player_id, action in zip(
                    batch.decisions.player_ids[action_rows].tolist(),
                    batch.actions[action_rows].tolist(),
                    strict=True,
                )
            ],
            "random_inputs": _recorded_inputs_for_game(batch, lane),
            "input_snapshot": canonical_game_state(batch.source_states[lane]),
            "expected_snapshot": canonical_game_state(batch.scalar_states[lane]),
            "expected_events": batch.expected_events[game_id],
        }
    )
    if result is not None:
        payload["actual_snapshot"] = tensor_semantic_snapshot(
            result.state, game_id
        )
        payload["actual_events"] = _events_by_game(result.events).get(game_id, [])
    return payload


def run_campaign(args: argparse.Namespace) -> int:
    """Run the bounded campaign and persist pass/fail evidence."""
    started = perf_counter()
    transitions = 0
    games = 0
    stress_draws = 0
    replenish_draws = 0
    reshuffled_lanes = 0
    batches = 0
    failure: dict[str, object] | None = None
    batch: PreparedBatch | None = None
    result: TensorKernelResult | None = None
    try:
        while transitions < args.transitions:
            if perf_counter() - started > args.timeout_seconds:
                raise TimeoutError("Chunk-5 differential campaign exceeded its cap")
            count = min(args.batch_size, max(1, args.transitions - transitions))
            batch = prepare_batch(games, count, args.seed)
            result = apply_cards_to_react(
                batch.tensor_state,
                batch.decisions,
                batch.actions,
                batch.draw_inputs,
            )
            verify_result(batch, result)
            if batches == 0:
                _determinism_checks(batch, result)
            for lane in range(batch.tensor_state.batch_size):
                if not torch.equal(
                    batch.tensor_state.rng_words[lane], result.state.rng_words[lane]
                ):
                    reshuffled_lanes += 1
            transitions += batch.transition_count
            games += count
            stress_draws += batch.stress_draw_count
            replenish_draws += batch.replenish_draw_count
            batches += 1
            print(
                f"D2 Chunk 5: {transitions}/{args.transitions} transitions, "
                f"{games} games, {perf_counter() - started:.1f}s",
                flush=True,
            )
    except Exception as exc:
        failure = _failure_payload(
            exc,
            batch,
            result,
            batch_start=games,
            seed=args.seed,
        )

    elapsed = perf_counter() - started
    result_artifact = {
        "schema_version": 1,
        "stage": "D2_chunk_5_random_differential",
        "status": "pass" if failure is None else "fail",
        "hypothesis": (
            "The cards-to-first-REACT tensor slice preserves exact legacy semantics, "
            "draw counts, and per-lane RNG state across randomized states."
        ),
        "confidence_before_run": 0.8,
        "requested_transitions": args.transitions,
        "completed_transitions": transitions,
        "games": games,
        "batches": batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "stress_draws": stress_draws,
        "replenish_draws": replenish_draws,
        "lanes_with_rng_advancement": reshuffled_lanes,
        "semantic_mismatches": 0 if failure is None else 1,
        "determinism_checks": (
            ["repeat", "save_resume", "lane_reorder", "focal_lane_isolation"]
            if batches > 0
            else []
        ),
        "modes": list(_MODES),
        "elapsed_seconds": elapsed,
        "timeout_seconds": args.timeout_seconds,
        "source_identity": _source_tree_identity(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_device": "cpu",
        },
        "failure": failure,
    }
    _write_json(args.output, result_artifact)
    print(json.dumps(result_artifact, indent=2, sort_keys=True))
    return 0 if failure is None and transitions >= args.transitions else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-size campaign and explicit wall-clock cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=5_005)
    parser.add_argument(
        "--timeout-seconds", type=float, default=1_800.0
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d2_chunk5_differential.json"),
    )
    args = parser.parse_args(argv)
    if args.transitions < 1 or args.batch_size < 1 or args.batch_size > 256:
        parser.error("transitions must be positive and batch-size must be in 1..256")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run_campaign(_parse_args()))
