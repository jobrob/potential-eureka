"""Focused phase-by-phase differential gates for Direction D2 kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from io import BytesIO

import numpy as np
import torch

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, simultaneous_decisions
from heat.engine.phases import (
    phase_play_cards,
    phase_shift_gears,
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
from heat.ml.vector_env.kernels import (
    TensorKernelEvent,
    apply_cards_to_react,
    apply_gear_actions,
)
from heat.ml.vector_env.kernels.common import TensorKernelResult
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    tensor_legal_action_masks,
    tensor_observations,
)
from heat.tracks.generator import generate_track
from heat.ml.vector_env.state import TensorGameState
from heat.ml.vector_env.random_inputs import RecordedDrawInputs


GearChoice = tuple[int, int]


def _state(seed: int, seats: int) -> GameState:
    """Build one deterministic started generated-track state."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 100_000)
    for player in state.players:
        player.lap = 1
    return state


def _gear_batch(
    states: list[GameState],
    game_ids: list[int],
    choices: dict[tuple[int, int], GearChoice],
) -> tuple[TensorDecisionBatch, torch.Tensor]:
    """Encode the complete non-forced gear phase in stable lane/player order."""
    decision_game_ids: list[int] = []
    player_ids: list[int] = []
    actions: list[int] = []
    for game_id, state in zip(game_ids, states, strict=True):
        for decision in simultaneous_decisions(state, DecisionKind.GEAR):
            choice = choices[(game_id, decision.player_id)]
            decision_game_ids.append(game_id)
            player_ids.append(decision.player_id)
            actions.append(choice[0] - rules.MIN_GEAR)
    return (
        TensorDecisionBatch.create(
            decision_game_ids,
            player_ids,
            [TensorDecisionKind.GEAR] * len(actions),
        ),
        torch.tensor(actions, dtype=torch.int64),
    )


def _scalar_gear_phase(
    state: GameState,
    game_id: int,
    choices: dict[tuple[int, int], GearChoice],
) -> list[GameEvent]:
    """Apply the oracle gear phase with the driver's forced-seat insertion order."""
    payload: dict[int, GearChoice] = {
        player.player_id: (1, 0)
        for player in state.active_players
        if player.spun_out
    }
    for decision in simultaneous_decisions(state, DecisionKind.GEAR):
        payload[decision.player_id] = choices[(game_id, decision.player_id)]
    return phase_shift_gears(state, payload)


def _events_by_game(
    events: tuple[TensorKernelEvent, ...],
) -> dict[int, list[dict[str, object]]]:
    """Group tensor semantic events by stable game identity."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        grouped[event.game_id].append(event.semantic())
    return grouped


def _assert_next_card_rows(
    states: list[GameState],
    game_ids: list[int],
    result_state: TensorGameState,
    result: TensorKernelResult,
) -> None:
    """Compare post-gear card observations and masks for every active seat."""
    expected_observations = np.stack(
        [
            encode_observation(state, decision.player_id, decision)
            for state in states
            for decision in simultaneous_decisions(state, DecisionKind.CARDS)
        ]
    )
    expected_masks = np.stack(
        [
            legal_action_mask(decision, state)
            for state in states
            for decision in simultaneous_decisions(state, DecisionKind.CARDS)
        ]
    )
    assert result.next_decisions.game_ids.tolist() == [
        game_id
        for game_id, state in zip(game_ids, states, strict=True)
        for _decision in simultaneous_decisions(state, DecisionKind.CARDS)
    ]
    np.testing.assert_array_equal(
        tensor_observations(result_state, result.next_decisions).numpy(),
        expected_observations,
    )
    np.testing.assert_array_equal(
        tensor_legal_action_masks(result_state, result.next_decisions).numpy(),
        expected_masks,
    )


def test_gear_kernel_matches_state_events_and_next_card_rows() -> None:
    """Readable 2/4/6-seat fixtures cover paid heat, spin, and finished seats."""
    states = [_state(41_002, 2), _state(41_004, 4), _state(41_006, 6)]
    game_ids = [2, 4, 6]
    states[0].players[0].spun_out = True
    states[0].players[0].gear = 3
    states[1].players[-1].finished = True
    states[1].players[-1].finish_order = 1

    choices: dict[tuple[int, int], GearChoice] = {}
    for game_id, state in zip(game_ids, states, strict=True):
        for decision in simultaneous_decisions(state, DecisionKind.GEAR):
            legal = decision.legal
            assert isinstance(legal, list)
            choices[(game_id, decision.player_id)] = legal[-1]

    tensor_state = legacy_states_to_tensor(states, game_ids=game_ids)
    decisions, actions = _gear_batch(states, game_ids, choices)
    result = apply_gear_actions(tensor_state, decisions, actions)
    expected_events = {
        game_id: [
            canonical_event(event)
            for event in _scalar_gear_phase(state, game_id, choices)
        ]
        for game_id, state in zip(game_ids, states, strict=True)
    }

    for game_id, state in zip(game_ids, states, strict=True):
        assert tensor_semantic_snapshot(result.state, game_id) == canonical_game_state(
            state
        )
    assert _events_by_game(result.events) == expected_events
    _assert_next_card_rows(states, game_ids, result.state, result)


def test_gear_kernel_has_1000_transition_batch_order_differential() -> None:
    """A batch-256 randomized gate proves exact state/events independent of lanes."""
    rng = np.random.default_rng(3_003)
    states = [_state(42_000 + index, 2 + index % 5) for index in range(256)]
    game_ids = [50_000 + index for index in range(len(states))]
    choices: dict[tuple[int, int], GearChoice] = {}
    transition_count = 0
    for game_id, state in zip(game_ids, states, strict=True):
        for player in state.players:
            player.gear = int(rng.integers(1, 5))
            if player.heat_available and bool(rng.integers(0, 4) == 0):
                player.pay_heat(int(rng.integers(1, player.heat_available + 1)))
            player.spun_out = bool(rng.integers(0, 10) == 0)
            transition_count += 1
        for decision in simultaneous_decisions(state, DecisionKind.GEAR):
            legal = decision.legal
            assert isinstance(legal, list)
            choices[(game_id, decision.player_id)] = legal[
                int(rng.integers(0, len(legal)))
            ]
    assert transition_count >= 1_000

    tensor_state = legacy_states_to_tensor(states, game_ids=game_ids)
    decisions, actions = _gear_batch(states, game_ids, choices)
    result = apply_gear_actions(tensor_state, decisions, actions)

    reversed_states = list(reversed(states))
    reversed_ids = list(reversed(game_ids))
    reversed_tensor = legacy_states_to_tensor(reversed_states, game_ids=reversed_ids)
    reversed_decisions, reversed_actions = _gear_batch(
        reversed_states, reversed_ids, choices
    )
    reordered_result = apply_gear_actions(
        reversed_tensor, reversed_decisions, reversed_actions
    )

    expected_events: dict[int, list[dict[str, object]]] = {}
    for game_id, state in zip(game_ids, states, strict=True):
        expected_events[game_id] = [
            canonical_event(event)
            for event in _scalar_gear_phase(state, game_id, choices)
        ]
        expected_snapshot = canonical_game_state(state)
        assert tensor_semantic_snapshot(result.state, game_id) == expected_snapshot
        assert (
            tensor_semantic_snapshot(reordered_result.state, game_id)
            == expected_snapshot
        )
    assert _events_by_game(result.events) == expected_events
    assert _events_by_game(reordered_result.events) == expected_events


def _card_choice(
    state: GameState,
    decision: Decision,
    *,
    prefer_stress: bool,
    prefer_speed: bool,
) -> tuple[tuple[Card, ...], int]:
    """Choose one readable legal card branch and return its flat codec index."""
    legal = decision.legal
    assert isinstance(legal, list)
    chosen = legal[0]
    if prefer_stress:
        chosen = next(
            play
            for play in legal
            if any(card.card_type is CardType.STRESS for card in play)
        )
    elif prefer_speed:
        speed_plays = [
            play
            for play in legal
            if all(card.card_type is CardType.SPEED for card in play)
        ]
        chosen = max(speed_plays, key=lambda play: sum(card.value for card in play))
    index = encode_action_index(decision, chosen)
    canonical = decode_legal_action(decision, state, index)
    assert isinstance(canonical, tuple)
    return canonical, index


def _record_stress_draws(
    state: GameState, player_id: int, cards: tuple[Card, ...]
) -> list[list[str]]:
    """Record top-of-deck sequences through the first basic card per stress."""
    pile = list(state.get_player(player_id).deck.draw_pile)
    records: list[list[str]] = []
    for card in cards:
        if card.card_type is not CardType.STRESS:
            continue
        sequence: list[str] = []
        while pile:
            drawn = pile.pop()
            sequence.append(drawn.id)
            if drawn.card_type is CardType.SPEED:
                break
        if sequence and drawn.card_type is not CardType.SPEED:
            raise AssertionError("fixture would require the deferred reshuffle path")
        records.append(sequence)
    return records


def _turn_start_event(
    state: GameState, player_id: int, pre_play_hand: list[str]
) -> GameEvent:
    """Build the exact driver receipt that sits between cards and reveal."""
    player = state.get_player(player_id)
    next_corner, distance = rules.distance_to_next_corner(
        state.track, player.position
    )
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


def _scalar_cards_to_react(
    state: GameState,
    choices: dict[int, tuple[Card, ...]],
) -> tuple[list[GameEvent], object]:
    """Run the scalar phase functions to the same first-REACT boundary."""
    target_id = state.turn_order[0]
    pre_play_hand = [card.display_name for card in state.get_player(target_id).hand]
    events = phase_play_cards(state, choices)
    events.append(_turn_start_event(state, target_id, pre_play_hand))
    before_reveal = len(state.event_log)
    reveal_events = step_reveal_and_move(state, state.get_player(target_id))
    stress_events = [
        event
        for event in state.event_log[before_reveal:]
        if event.event_type == "stress_resolved"
    ]
    events.extend(stress_events)
    events.extend(reveal_events)
    events.extend(step_adrenaline(state, state.get_player(target_id)))
    react = rules.legal_react_options(
        state.get_player(target_id), list(state.active_players), state.starting_player_count
    )
    return events, react


def _scalar_cards_to_next_react(
    state: GameState,
    choices: dict[int, tuple[Card, ...]],
) -> tuple[list[GameEvent], Decision, dict[int, list[list[str]]], dict[int, list[str]]]:
    """Run the scalar oracle through skipped cars while recording every draw."""
    pre_play_hands = {
        player.player_id: [card.display_name for card in player.hand]
        for player in state.active_players
    }
    stress_records: dict[int, list[list[str]]] = defaultdict(list)
    replenish_records: dict[int, list[str]] = defaultdict(list)
    stress_closed: dict[int, bool] = defaultdict(lambda: True)

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
                groups = stress_records[recorded_player]
                if stress_closed[recorded_player]:
                    groups.append([])
                    stress_closed[recorded_player] = False
            drawn = original(count)
            if state.current_phase is Phase.REVEAL_AND_MOVE:
                stress_records[recorded_player][-1].extend(card.id for card in drawn)
                if drawn and drawn[-1].card_type is CardType.SPEED:
                    stress_closed[recorded_player] = True
            elif state.current_phase is Phase.REPLENISH:
                replenish_records[recorded_player].extend(card.id for card in drawn)
            return drawn

        player.deck.draw = recording_draw  # type: ignore[method-assign]

    events = phase_play_cards(state, choices)
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
        reveal_events = step_reveal_and_move(state, player)
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
        options = rules.legal_react_options(
            player, list(state.active_players), state.starting_player_count
        )
        return (
            events,
            Decision(DecisionKind.REACT, player_id, options),
            dict(stress_records),
            dict(replenish_records),
        )
    raise AssertionError("fixture did not reach a REACT decision")


def test_cards_to_react_matches_stress_traffic_lap_and_react_rows() -> None:
    """The vertical slice matches exact state/events through its first REACT row."""
    modes = ("stress", "traffic", "lap")
    tensor_states: list[GameState] = []
    scalar_states: list[GameState] = []
    game_ids = [70, 71, 72]
    for offset, mode in enumerate(modes):
        seed = 43_000 + offset
        tensor_source = _state(seed, 2 + offset * 2)
        scalar = _state(seed, 2 + offset * 2)
        for candidate in (tensor_source, scalar):
            leader = candidate.players[0]
            for opponent in candidate.players[1:]:
                opponent.lap = 0
                opponent.position = 0
            if mode == "lap":
                leader.position = candidate.track.length - 2
            else:
                leader.position = 10
            if mode == "traffic":
                candidate.players[1].position = 14
                candidate.players[2].position = 14
            candidate.compute_turn_order()
            phase_shift_gears(
                candidate,
                {
                    player.player_id: (player.gear, 0)
                    for player in candidate.active_players
                },
            )
            candidate.event_log.clear()
        tensor_states.append(tensor_source)
        scalar_states.append(scalar)

    tensor_state = legacy_states_to_tensor(tensor_states, game_ids=game_ids)
    decisions_game: list[int] = []
    decisions_player: list[int] = []
    action_indices: list[int] = []
    scalar_choices: list[dict[int, tuple[Card, ...]]] = []
    draw_records: dict[tuple[int, int], list[list[str]]] = {}
    for game_id, mode, state in zip(game_ids, modes, tensor_states, strict=True):
        choices: dict[int, tuple[Card, ...]] = {}
        target_id = state.turn_order[0]
        for decision in simultaneous_decisions(state, DecisionKind.CARDS):
            chosen, action_index = _card_choice(
                state,
                decision,
                prefer_stress=mode == "stress" and decision.player_id == target_id,
                prefer_speed=mode in ("traffic", "lap") and decision.player_id == target_id,
            )
            choices[decision.player_id] = chosen
            decisions_game.append(game_id)
            decisions_player.append(decision.player_id)
            action_indices.append(action_index)
            if decision.player_id == target_id:
                draw_records[(game_id, target_id)] = _record_stress_draws(
                    state, target_id, chosen
                )
        scalar_choices.append(choices)

    tensor_decisions = TensorDecisionBatch.create(
        decisions_game,
        decisions_player,
        [TensorDecisionKind.CARDS] * len(action_indices),
    )
    recorded = RecordedDrawInputs.create(tensor_state, draw_records)
    result = apply_cards_to_react(
        tensor_state,
        tensor_decisions,
        torch.tensor(action_indices, dtype=torch.int64),
        recorded,
    )

    action_by_identity = dict(
        zip(zip(decisions_game, decisions_player, strict=True), action_indices, strict=True)
    )
    reversed_states = list(reversed(tensor_states))
    reversed_ids = list(reversed(game_ids))
    reversed_tensor = legacy_states_to_tensor(reversed_states, game_ids=reversed_ids)
    reversed_game_ids: list[int] = []
    reversed_players: list[int] = []
    reversed_actions: list[int] = []
    for game_id, state in zip(reversed_ids, reversed_states, strict=True):
        for decision in simultaneous_decisions(state, DecisionKind.CARDS):
            reversed_game_ids.append(game_id)
            reversed_players.append(decision.player_id)
            reversed_actions.append(action_by_identity[(game_id, decision.player_id)])
    reversed_decisions = TensorDecisionBatch.create(
        reversed_game_ids,
        reversed_players,
        [TensorDecisionKind.CARDS] * len(reversed_actions),
    )
    reordered_result = apply_cards_to_react(
        reversed_tensor,
        reversed_decisions,
        torch.tensor(reversed_actions, dtype=torch.int64),
        RecordedDrawInputs.create(reversed_tensor, draw_records),
    )

    expected_events: dict[int, list[dict[str, object]]] = {}
    scalar_react_decisions = []
    for game_id, state, choices in zip(
        game_ids, scalar_states, scalar_choices, strict=True
    ):
        events, react_options = _scalar_cards_to_react(state, choices)
        expected_events[game_id] = [canonical_event(event) for event in events]
        scalar_react_decisions.append(
            Decision(DecisionKind.REACT, state.turn_order[0], react_options)
        )
        assert tensor_semantic_snapshot(result.state, game_id) == canonical_game_state(
            state
        )
        assert (
            tensor_semantic_snapshot(reordered_result.state, game_id)
            == canonical_game_state(state)
        )
        target = state.get_player(state.turn_order[0])
        if game_id == 70:
            assert any(event.event_type == "stress_resolved" for event in events)
        elif game_id == 71:
            assert target.position != 14
        else:
            assert target.lap == 2
    assert _events_by_game(result.events) == expected_events
    assert _events_by_game(reordered_result.events) == expected_events

    np.testing.assert_array_equal(
        tensor_observations(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                encode_observation(state, decision.player_id, decision)
                for state, decision in zip(
                    scalar_states, scalar_react_decisions, strict=True
                )
            ]
        ),
    )
    np.testing.assert_array_equal(
        tensor_legal_action_masks(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                legal_action_mask(decision, state)
                for state, decision in zip(
                    scalar_states, scalar_react_decisions, strict=True
                )
            ]
        ),
    )


def _campaign_cards_state(seed: int, seats: int, index: int) -> GameState:
    """Build a reproducible non-finishing state with varied traffic geometry."""
    state = _state(seed, seats)
    leader = state.players[0]
    leader.position = 5 + index % max(1, state.track.length - 12)
    for opponent in state.players[1:]:
        opponent.lap = 0
        opponent.position = (
            leader.position + 3 + (opponent.player_id % 3)
        ) % state.track.length
    state.compute_turn_order()
    phase_shift_gears(
        state,
        {
            player.player_id: (player.gear, 0)
            for player in state.active_players
        },
    )
    state.event_log.clear()
    return state


def test_cards_to_react_has_1000_transition_random_differential() -> None:
    """Batch-256 randomized cards, draws, traffic, events, and REACT rows match."""
    rng = np.random.default_rng(4_004)
    tensor_states = [
        _campaign_cards_state(44_000 + index, 2 + index % 5, index)
        for index in range(256)
    ]
    scalar_states = [
        _campaign_cards_state(44_000 + index, 2 + index % 5, index)
        for index in range(256)
    ]
    game_ids = [80_000 + index for index in range(256)]
    tensor_state = legacy_states_to_tensor(tensor_states, game_ids=game_ids)

    decision_game_ids: list[int] = []
    decision_players: list[int] = []
    actions: list[int] = []
    choices_by_game: list[dict[int, tuple[Card, ...]]] = []
    draw_records: dict[tuple[int, int], list[list[str]]] = {}
    for game_id, state in zip(game_ids, tensor_states, strict=True):
        target_id = state.turn_order[0]
        choices: dict[int, tuple[Card, ...]] = {}
        for decision in simultaneous_decisions(state, DecisionKind.CARDS):
            legal = decision.legal
            assert isinstance(legal, list)
            chosen = legal[int(rng.integers(0, len(legal)))]
            action = encode_action_index(decision, chosen)
            canonical = decode_legal_action(decision, state, action)
            assert isinstance(canonical, tuple)
            chosen = canonical
            choices[decision.player_id] = chosen
            decision_game_ids.append(game_id)
            decision_players.append(decision.player_id)
            actions.append(action)
            if decision.player_id == target_id:
                draw_records[(game_id, target_id)] = _record_stress_draws(
                    state, target_id, chosen
                )
        choices_by_game.append(choices)
    assert len(actions) >= 1_000

    decisions = TensorDecisionBatch.create(
        decision_game_ids,
        decision_players,
        [TensorDecisionKind.CARDS] * len(actions),
    )
    result = apply_cards_to_react(
        tensor_state,
        decisions,
        torch.tensor(actions, dtype=torch.int64),
        RecordedDrawInputs.create(tensor_state, draw_records),
    )
    repeated = apply_cards_to_react(
        tensor_state,
        decisions,
        torch.tensor(actions, dtype=torch.int64),
        RecordedDrawInputs.create(tensor_state, draw_records),
    )
    for (name, first), (repeated_name, second) in zip(
        result.state.tensor_fields(), repeated.state.tensor_fields(), strict=True
    ):
        assert name == repeated_name
        assert torch.equal(first, second), name
    assert result.events == repeated.events
    assert torch.equal(
        result.next_decisions.game_ids, repeated.next_decisions.game_ids
    )
    assert torch.equal(
        result.next_decisions.player_ids, repeated.next_decisions.player_ids
    )
    assert torch.equal(result.next_decisions.kinds, repeated.next_decisions.kinds)

    expected_events: dict[int, list[dict[str, object]]] = {}
    scalar_react_decisions: list[Decision] = []
    for game_id, state, choices in zip(
        game_ids, scalar_states, choices_by_game, strict=True
    ):
        events, react_options = _scalar_cards_to_react(state, choices)
        expected_events[game_id] = [canonical_event(event) for event in events]
        react_decision = Decision(
            DecisionKind.REACT, state.turn_order[0], react_options
        )
        scalar_react_decisions.append(react_decision)
        assert tensor_semantic_snapshot(result.state, game_id) == canonical_game_state(
            state
        )
    assert _events_by_game(result.events) == expected_events

    np.testing.assert_array_equal(
        tensor_observations(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                encode_observation(state, decision.player_id, decision)
                for state, decision in zip(
                    scalar_states, scalar_react_decisions, strict=True
                )
            ]
        ),
    )
    np.testing.assert_array_equal(
        tensor_legal_action_masks(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                legal_action_mask(decision, state)
                for state, decision in zip(
                    scalar_states, scalar_react_decisions, strict=True
                )
            ]
        ),
    )

    focal_state = legacy_states_to_tensor([tensor_states[0]], game_ids=[game_ids[0]])
    focal_rows = [
        row for row, game_id in enumerate(decision_game_ids) if game_id == game_ids[0]
    ]
    focal_decisions = TensorDecisionBatch.create(
        [decision_game_ids[row] for row in focal_rows],
        [decision_players[row] for row in focal_rows],
        [TensorDecisionKind.CARDS] * len(focal_rows),
    )
    focal_result = apply_cards_to_react(
        focal_state,
        focal_decisions,
        torch.tensor([actions[row] for row in focal_rows], dtype=torch.int64),
        RecordedDrawInputs.create(
            focal_state,
            {
                key: value
                for key, value in draw_records.items()
                if key[0] == game_ids[0]
            },
        ),
    )
    assert tensor_semantic_snapshot(
        focal_result.state, game_ids[0]
    ) == tensor_semantic_snapshot(result.state, game_ids[0])
    assert _events_by_game(focal_result.events)[game_ids[0]] == expected_events[
        game_ids[0]
    ]


def _chunk5_state(seed: int, mode: str) -> GameState:
    """Build one readable reshuffle or replenish-continuation fixture."""
    state = _state(seed, 3)
    target = state.players[0]
    target.position = 20
    for opponent in state.players[1:]:
        opponent.position = 0
        opponent.lap = 0

    if mode == "stress_reshuffle":
        target.gear = 1
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
        target.gear = 1
        target.position = state.track.length - 1
        target.lap = state.track.laps
    else:  # pragma: no cover - test helper guard
        raise ValueError(mode)
    state.compute_turn_order()
    state.event_log.clear()
    return state


def test_chunk5_reshuffles_and_replenishes_before_the_next_react() -> None:
    """Stress, clutter, and finish skips match state, draws, RNG, and events."""
    modes = ("stress_reshuffle", "clutter_replenish", "finish_replenish")
    tensor_states = [
        _chunk5_state(45_000 + index, mode)
        for index, mode in enumerate(modes)
    ]
    scalar_states = [
        _chunk5_state(45_000 + index, mode)
        for index, mode in enumerate(modes)
    ]
    game_ids = [90_000 + index for index in range(len(modes))]
    tensor_state = legacy_states_to_tensor(tensor_states, game_ids=game_ids)

    decision_games: list[int] = []
    decision_players: list[int] = []
    actions: list[int] = []
    scalar_choices: list[dict[int, tuple[Card, ...]]] = []
    for game_id, mode, tensor_source, scalar in zip(
        game_ids, modes, tensor_states, scalar_states, strict=True
    ):
        choices: dict[int, tuple[Card, ...]] = {}
        for decision in simultaneous_decisions(tensor_source, DecisionKind.CARDS):
            legal = decision.legal
            assert isinstance(legal, list)
            if decision.player_id == tensor_source.turn_order[0] and mode == "stress_reshuffle":
                chosen = next(
                    play
                    for play in legal
                    if any(card.card_type is CardType.STRESS for card in play)
                )
            elif decision.player_id == tensor_source.turn_order[0] and mode == "finish_replenish":
                chosen = max(legal, key=lambda play: sum(card.value for card in play))
            else:
                chosen = legal[0]
            action = encode_action_index(decision, chosen)
            scalar_decision = next(
                item
                for item in simultaneous_decisions(scalar, DecisionKind.CARDS)
                if item.player_id == decision.player_id
            )
            decoded = decode_legal_action(scalar_decision, scalar, action)
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
        events, react, stress, replenishments = _scalar_cards_to_next_react(
            scalar, choices
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
    result = apply_cards_to_react(
        tensor_state,
        decisions,
        torch.tensor(actions, dtype=torch.int64),
        RecordedDrawInputs.create(
            tensor_state,
            stress_records,
            replenish_records,
        ),
    )
    payload = BytesIO()
    torch.save(tensor_state, payload)
    payload.seek(0)
    restored = torch.load(payload, weights_only=False)
    resumed = apply_cards_to_react(
        restored,
        decisions,
        torch.tensor(actions, dtype=torch.int64),
        RecordedDrawInputs.create(restored, stress_records, replenish_records),
    )
    for (name, actual), (resumed_name, resumed_value) in zip(
        result.state.tensor_fields(), resumed.state.tensor_fields(), strict=True
    ):
        assert name == resumed_name
        assert torch.equal(actual, resumed_value), name
    assert result.events == resumed.events
    assert torch.equal(
        result.next_decisions.player_ids, resumed.next_decisions.player_ids
    )

    for game_id, scalar in zip(game_ids, scalar_states, strict=True):
        assert tensor_semantic_snapshot(result.state, game_id) == canonical_game_state(
            scalar
        )
    assert _events_by_game(result.events) == expected_events
    assert result.next_decisions.player_ids.tolist() == [
        decision.player_id for decision in react_decisions
    ]
    assert any(event.event_type == "stress_resolved" for event in result.events)
    assert sum(event.event_type == "replenish" for event in result.events) == 2
    np.testing.assert_array_equal(
        tensor_observations(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                encode_observation(state, decision.player_id, decision)
                for state, decision in zip(
                    scalar_states, react_decisions, strict=True
                )
            ]
        ),
    )
    np.testing.assert_array_equal(
        tensor_legal_action_masks(result.state, result.next_decisions).numpy(),
        np.stack(
            [
                legal_action_mask(decision, state)
                for state, decision in zip(
                    scalar_states, react_decisions, strict=True
                )
            ]
        ),
    )
