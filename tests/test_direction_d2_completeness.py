"""Remaining currently-applicable correctness checks from the D2 design."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.verify_direction_d2_differential import (
    DifferentialMismatch,
    PreparedBatch,
    _failure_payload,
    prepare_batch,
    run_scalar_to_next_react,
    verify_result,
)
from heat.engine import rules
from heat.engine.driver import DecisionKind, simultaneous_decisions
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.ml.action_codec import decode_legal_action, encode_action_index
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.ml.vector_env.bridge import legacy_states_to_tensor, tensor_semantic_snapshot
from heat.ml.vector_env.kernels import apply_cards_to_react
from heat.ml.vector_env.kernels.common import TensorKernelResult
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    tensor_legal_action_masks,
    tensor_observations,
)
from heat.ml.vector_env.random_inputs import RecordedDrawInputs
from heat.ml.vector_env.state import TensorGameState
from heat.tracks.generator import generate_track


def _run_prepared(
    start: int, count: int, seed: int = 7_007
) -> tuple[PreparedBatch, TensorKernelResult]:
    """Run and verify one prepared campaign batch for focused assertions."""
    batch = prepare_batch(start, count, seed)
    result = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    verify_result(batch, result)
    return batch, result


def _first_difference(left: object, right: object, path: str = "root") -> str:
    """Locate the first nested semantic mismatch for readable lockstep failures."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                return f"{path}.{key}: missing"
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def _focal_receipt(
    result: TensorKernelResult, game_id: int
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    """Return one stable game's complete state/event/REACT receipt."""
    rows = (result.next_decisions.game_ids == game_id).nonzero(
        as_tuple=False
    ).flatten()
    assert len(rows) == 1
    row = int(rows[0].item())
    return (
        tensor_semantic_snapshot(result.state, game_id),
        [event.semantic() for event in result.events if event.game_id == game_id],
        tensor_observations(result.state, result.next_decisions)[row].numpy(),
        tensor_legal_action_masks(result.state, result.next_decisions)[row].numpy(),
    )


def test_cards_kernel_is_exact_at_batches_1_8_32_and_256() -> None:
    """An RNG-advancing focal game is identical at every designed batch width."""
    start = 330_002  # stress_reshuffle mode
    game_id = 1_000_000 + start
    expected: tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray] | None = None
    for batch_size in (1, 8, 32, 256):
        _batch, result = _run_prepared(start, batch_size)
        actual = _focal_receipt(result, game_id)
        if expected is None:
            expected = actual
            continue
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        np.testing.assert_array_equal(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])


def test_early_finished_neighbor_cannot_change_a_focal_kernel_trace() -> None:
    """A lane that finishes its first car cannot consume a neighbor's work."""
    start = 340_000  # normal focal; offset 4 is finish_replenish
    focal_game = 1_000_000 + start
    full_batch, full_result = _run_prepared(start, 5)
    _alone_batch, alone_result = _run_prepared(start, 1)

    finished_game = focal_game + 4
    finished_events = [
        event for event in full_result.events if event.game_id == finished_game
    ]
    assert any(
        event.event_type == "reveal_and_move"
        and bool(event.data["finished"])
        for event in finished_events
    )
    assert any(event.event_type == "replenish" for event in finished_events)
    full_focal = _focal_receipt(full_result, focal_game)
    alone_focal = _focal_receipt(alone_result, focal_game)
    assert full_focal[0] == alone_focal[0]
    assert full_focal[1] == alone_focal[1]
    np.testing.assert_array_equal(full_focal[2], alone_focal[2])
    np.testing.assert_array_equal(full_focal[3], alone_focal[3])
    assert len(full_batch.game_ids) == 5


def _swap_card_into_hand(state: GameState, card_id: str, hand_index: int) -> None:
    """Place one existing card at a stable hand position without duplication."""
    player = state.players[0]
    for index, card in enumerate(player.hand):
        if card.id == card_id:
            player.hand[index], player.hand[hand_index] = (
                player.hand[hand_index],
                player.hand[index],
            )
            return
    for pile in (player.deck._draw_pile, player.deck._discard_pile):
        for index, card in enumerate(pile):
            if card.id == card_id:
                pile[index], player.hand[hand_index] = (
                    player.hand[hand_index],
                    pile[index],
                )
                return
    raise AssertionError(f"missing card {card_id}")


def _multi_stress_corner_state(seed: int) -> GameState:
    """Build a readable upgrade plus two-stress corner-crossing state."""
    state = GameState.create(generate_track(seed), 3, seed=seed + 100_000)
    for player in state.players:
        player.lap = 1
        player.gear = 1
    target = state.players[0]
    _swap_card_into_hand(state, "p0_upg_5", 0)
    _swap_card_into_hand(state, "p0_stress_0", 1)
    _swap_card_into_hand(state, "p0_stress_1", 2)
    target.gear = 3
    target.position = (state.track.corners[0].start - 2) % state.track.length
    for opponent in state.players[1:]:
        opponent.position = max(0, target.position - 5 - opponent.player_id)
        opponent.lap = 0
    state.compute_turn_order()
    state.event_log.clear()
    assert state.turn_order[0] == target.player_id
    return state


def test_internal_lockstep_covers_upgrade_two_stress_and_corner_crossing() -> None:
    """Every internal D2 comparison point matches on the missing branches."""
    source = _multi_stress_corner_state(350_000)
    scalar = _multi_stress_corner_state(350_000)
    game_id = 350_000
    desired_ids = {"p0_upg_5", "p0_stress_0", "p0_stress_1"}
    scalar_decisions = {
        decision.player_id: decision
        for decision in simultaneous_decisions(scalar, DecisionKind.CARDS)
    }
    choices: dict[int, tuple[Card, ...]] = {}
    decision_games: list[int] = []
    decision_players: list[int] = []
    actions: list[int] = []
    for decision in simultaneous_decisions(source, DecisionKind.CARDS):
        legal = decision.legal
        assert isinstance(legal, list)
        chosen = (
            next(play for play in legal if {card.id for card in play} == desired_ids)
            if decision.player_id == 0
            else legal[0]
        )
        action = encode_action_index(decision, chosen)
        decoded = decode_legal_action(
            scalar_decisions[decision.player_id], scalar, action
        )
        assert isinstance(decoded, tuple)
        choices[decision.player_id] = decoded
        decision_games.append(game_id)
        decision_players.append(decision.player_id)
        actions.append(action)

    scalar_trace: list[tuple[str, dict[str, object]]] = []
    _events, react, stress, replenishments = run_scalar_to_next_react(
        scalar,
        choices,
        record_draws=True,
        trace=lambda label, state: scalar_trace.append(
            (label, canonical_game_state(state))
        ),
    )
    tensor_state = legacy_states_to_tensor([source], game_ids=[game_id])
    tensor_trace: list[tuple[str, dict[str, object]]] = []
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
            {(game_id, player): records for player, records in stress.items()},
            {
                (game_id, player): records
                for player, records in replenishments.items()
            },
        ),
        trace=lambda label, state, _lanes, _players: tensor_trace.append(
            (label, tensor_semantic_snapshot(state, game_id))
        ),
    )

    assert [label for label, _snapshot in scalar_trace] == [
        "cards_chosen",
        "stress_resolved:0",
        "stress_resolved:1",
        "speed_revealed",
        "movement_resolved",
        "adrenaline",
    ]
    for tensor_point, scalar_point in zip(
        tensor_trace, scalar_trace, strict=True
    ):
        assert tensor_point[0] == scalar_point[0]
        assert tensor_point[1] == scalar_point[1], _first_difference(
            tensor_point[1], scalar_point[1]
        )
    target = scalar.players[0]
    assert len(
        rules.corners_crossed(
            target.turn_start_position,
            target.position,
            scalar.track,
            target.speed_from_cards,
        )
    ) >= 1
    assert sum(event.event_type == "stress_resolved" for event in result.events) == 2
    assert react.player_id == result.next_decisions.player_ids[0].item()


def _zone_card_ids(state: TensorGameState, lane: int, player: int) -> list[int]:
    """Collect every real card identity across all tensor card zones."""
    result: list[int] = []
    for zone in (
        state.hand,
        state.draw_pile,
        state.discard_pile,
        state.heat_pool,
        state.cooldown_pool,
        state.cards_played,
    ):
        length = int(zone.lengths[lane, player].item())
        result.extend(int(item) for item in zone.card_ids[lane, player, :length])
    return result


def test_explicit_card_position_finish_and_mask_invariants() -> None:
    """Conservation and simple validity checks hold across every focal branch."""
    batch, result = _run_prepared(360_000, 5)
    for lane in range(result.state.batch_size):
        player_count = int(result.state.player_present[lane].sum().item())
        track_length = int(result.state.track_lengths[lane].item())
        for player in range(player_count):
            before = sorted(_zone_card_ids(batch.tensor_state, lane, player))
            after = sorted(_zone_card_ids(result.state, lane, player))
            assert before == after
            assert len(after) == len(set(after))
            position = int(result.state.position[lane, player].item())
            assert 0 <= position < track_length
        finish_orders = result.state.finish_order[lane][
            result.state.finished[lane]
        ].tolist()
        assert sorted(finish_orders) == list(range(1, len(finish_orders) + 1))
    masks = tensor_legal_action_masks(result.state, result.next_decisions)
    assert bool(torch.all(torch.any(masks, dim=1)))


def test_mismatch_payload_contains_a_complete_single_game_replay() -> None:
    """A failing differential preserves actions, draws, seeds, and both states."""
    batch = prepare_batch(370_002, 1, 7_007)
    result = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )
    corrupted = result.state.clone()
    corrupted.position[0, 0] = -1
    mismatched = TensorKernelResult(corrupted, result.events, result.next_decisions)
    with pytest.raises(DifferentialMismatch) as caught:
        verify_result(batch, mismatched)
    payload = _failure_payload(
        caught.value,
        batch,
        mismatched,
        batch_start=370_002,
        seed=7_007,
    )

    assert payload["game_id"] == batch.game_ids[0]
    assert payload["phase"] is not None
    assert payload["actions"]
    assert payload["random_inputs"]
    assert payload["input_snapshot"]
    assert payload["expected_snapshot"]
    assert payload["actual_snapshot"]
    assert payload["expected_events"]
    assert payload["actual_events"]
    assert payload["track_seed"] == 7_007 + 370_002 * 17
