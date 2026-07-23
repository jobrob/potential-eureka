"""Focused gates for the Direction D2-R compiled numeric redesign."""

from __future__ import annotations

import pytest
import torch

from experiments.verify_direction_d2_differential import prepare_batch
from heat.ml.vector_env.compiled import (
    MAX_EVENT_RECEIPTS,
    apply_compiled_common_core,
    apply_compiled_r2,
    materialize_receipts,
    receipts_from_events,
    prepare_action_matrix,
    pure_cards_to_react_common,
)
from heat.ml.vector_env.kernels import TensorKernelEvent, apply_cards_to_react
from heat.ml.vector_env.observations import (
    tensor_legal_action_masks,
    tensor_observations,
)
from heat.ml.vector_env.bridge import tensor_semantic_snapshot


def test_r0_receipts_round_trip_every_cards_to_react_event_schema() -> None:
    """Numeric receipts preserve fields, cards, and cross-lane event order."""
    batch = prepare_batch(340_000, 10, 7_007)
    result = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )

    receipts = receipts_from_events(result.events, result.state)

    assert materialize_receipts(receipts, result.state) == result.events
    assert {
        event.event_type for event in result.events
    } == {
        "play_cards",
        "turn_start",
        "stress_resolved",
        "reveal_and_move",
        "adrenaline_granted",
        "replenish",
    }


def test_r0_receipts_fail_clearly_on_per_game_overflow() -> None:
    """The fixed event capacity never truncates an oversized lane."""
    batch = prepare_batch(500_000, 1, 7_007)
    template = TensorKernelEvent(
        game_id=batch.game_ids[0],
        round_num=1,
        phase="play_cards",
        player_id=0,
        event_type="play_cards",
        data={"cards": [], "cluttered": False},
    )

    with pytest.raises(OverflowError, match="exceeds"):
        receipts_from_events(
            tuple(template for _ in range(MAX_EVENT_RECEIPTS + 1)),
            batch.tensor_state,
        )


@pytest.mark.parametrize("start", [500_000, 500_001])
def test_r1_eager_pure_core_matches_normal_and_traffic_oracle(start: int) -> None:
    """The pure common core matches eager D2 before compiler involvement."""
    batch = prepare_batch(start, 1, 7_007)
    eager = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )

    result = apply_compiled_common_core(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )

    assert tensor_semantic_snapshot(result.state, batch.game_ids[0]) == tensor_semantic_snapshot(
        eager.state, batch.game_ids[0]
    )
    assert materialize_receipts(result.receipts, result.state) == eager.events
    assert torch.equal(result.next_decisions.game_ids, eager.next_decisions.game_ids)
    assert torch.equal(result.next_decisions.player_ids, eager.next_decisions.player_ids)
    assert torch.equal(result.next_decisions.kinds, eager.next_decisions.kinds)
    assert torch.equal(result.observations, tensor_observations(eager.state, eager.next_decisions))
    assert torch.equal(
        result.legal_masks,
        tensor_legal_action_masks(eager.state, eager.next_decisions),
    )


def test_r1_common_core_is_one_full_graph() -> None:
    """Dynamo captures the tensor body with fullgraph=True and no fallback."""
    batch = prepare_batch(500_000, 1, 7_007)
    actions = prepare_action_matrix(batch.tensor_state, batch.decisions, batch.actions)
    compiled = torch.compile(
        pure_cards_to_react_common,
        backend="eager",
        fullgraph=True,
    )

    output = compiled(
        batch.tensor_state,
        actions,
        batch.draw_inputs.card_ids,
        batch.draw_inputs.lengths,
    )

    assert output.observations.shape == (1, 104)
    assert output.legal_masks.shape == (1, 516)


@pytest.mark.parametrize("start", [500_000, 500_001, 500_003, 500_004])
def test_r2_matches_every_non_reshuffle_fixture_mode(start: int) -> None:
    """R2 adds clutter and finish continuation without changing common paths."""
    batch = prepare_batch(start, 1, 7_007)
    eager = apply_cards_to_react(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )

    result = apply_compiled_r2(
        batch.tensor_state,
        batch.decisions,
        batch.actions,
        batch.draw_inputs,
    )

    for (expected_name, expected), (actual_name, actual) in zip(
        eager.state.tensor_fields(), result.state.tensor_fields(), strict=True
    ):
        assert actual_name == expected_name
        assert torch.equal(actual, expected), expected_name
    assert materialize_receipts(result.receipts, result.state) == eager.events
    assert torch.equal(result.next_decisions.game_ids, eager.next_decisions.game_ids)
    assert torch.equal(result.next_decisions.player_ids, eager.next_decisions.player_ids)
    assert torch.equal(result.next_decisions.kinds, eager.next_decisions.kinds)
    assert torch.equal(result.observations, tensor_observations(eager.state, eager.next_decisions))
    assert torch.equal(
        result.legal_masks,
        tensor_legal_action_masks(eager.state, eager.next_decisions),
    )


def test_r2_lane_order_and_focal_isolation_are_exact() -> None:
    """Changing neighbours or lane order cannot alter one game's R2 result."""
    cases = [500_000, 500_001, 500_003, 500_004]
    wide = prepare_batch(0, len(cases), 7_007, case_indices=cases)
    reversed_batch = prepare_batch(
        0, len(cases), 7_007, case_indices=list(reversed(cases))
    )
    focal = prepare_batch(0, 1, 7_007, case_indices=[cases[0]])

    wide_result = apply_compiled_r2(
        wide.tensor_state, wide.decisions, wide.actions, wide.draw_inputs
    )
    reversed_result = apply_compiled_r2(
        reversed_batch.tensor_state,
        reversed_batch.decisions,
        reversed_batch.actions,
        reversed_batch.draw_inputs,
    )
    focal_result = apply_compiled_r2(
        focal.tensor_state, focal.decisions, focal.actions, focal.draw_inputs
    )
    game_id = focal.game_ids[0]

    assert tensor_semantic_snapshot(wide_result.state, game_id) == tensor_semantic_snapshot(
        reversed_result.state, game_id
    )
    assert tensor_semantic_snapshot(wide_result.state, game_id) == tensor_semantic_snapshot(
        focal_result.state, game_id
    )
    wide_events = [
        event.semantic()
        for event in materialize_receipts(wide_result.receipts, wide_result.state)
        if event.game_id == game_id
    ]
    reversed_events = [
        event.semantic()
        for event in materialize_receipts(
            reversed_result.receipts, reversed_result.state
        )
        if event.game_id == game_id
    ]
    focal_events = [
        event.semantic()
        for event in materialize_receipts(focal_result.receipts, focal_result.state)
    ]
    assert wide_events == reversed_events == focal_events
