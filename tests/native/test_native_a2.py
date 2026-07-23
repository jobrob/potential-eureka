"""Representative hard-branch gates for Direction D3 chunk A2."""

from __future__ import annotations

import pytest

from experiments.verify_direction_d2_differential import prepare_batch
from heat.engine import rules
from heat.engine.phases import (
    ReactDecision,
    step_adrenaline,
    step_check_corner,
    step_discard,
    step_react,
    step_replenish,
    step_reveal_and_move,
    step_slipstream,
)
from heat.ml.native_env.bridge import legacy_state_to_native
from heat.ml.selfplay.semantic_contract import canonical_game_state


@pytest.mark.parametrize("case_start", [0, 25, 100])
def test_cards_to_first_react_matches_weighted_scalar_cases(case_start: int) -> None:
    """Normal, traffic, stress/reshuffle, clutter, and finish branches are exact."""
    batch = prepare_batch(case_start, 25, 5_005)
    for source, expected, choices, game_id, react in zip(
        batch.source_states,
        batch.scalar_states,
        batch.scalar_choices,
        batch.game_ids,
        batch.react_decisions,
        strict=True,
    ):
        native = legacy_state_to_native(source, game_id=game_id)
        boundary = native.apply_cards_to_react(choices)
        assert boundary["kind"] == "react"
        assert boundary["player_id"] == react.player_id
        assert native.semantic_snapshot() == canonical_game_state(expected), (
            game_id,
            batch.modes[batch.game_ids.index(game_id)],
        )


def _advance_scalar_to_next_react(state: object, completed_player: int) -> int | None:
    """Mirror the driver tail after one player replenishes, for A2 comparison."""
    from heat.models.game_state import GameState

    assert isinstance(state, GameState)
    completed_index = state.turn_order.index(completed_player)
    for player_id in state.turn_order[completed_index + 1 :]:
        player = state.get_player(player_id)
        if player.finished:
            continue
        if player.cluttered:
            player.gear = 1
            step_replenish(state, player)
            continue
        step_reveal_and_move(state, player)
        if player.finished:
            step_replenish(state, player)
            continue
        step_adrenaline(state, player)
        return player_id
    for player in state.active_players:
        player.spun_out = False
    state.round_num += 1
    return None


@pytest.mark.parametrize("case_start", [200, 225, 250])
def test_react_movement_corner_discard_tail_matches_scalar(case_start: int) -> None:
    """Boost, traffic, slipstream, corner/spin, discard, and replenish stay exact."""
    batch = prepare_batch(case_start, 25, 7_007)
    for offset, (state, game_id, react) in enumerate(
        zip(batch.scalar_states, batch.game_ids, batch.react_decisions, strict=True)
    ):
        player = state.get_player(react.player_id)
        options = react.legal
        decision = ReactDecision(
            cooldown_count=min(1, options.max_cooldown),
            use_boost=bool(options.can_boost and offset % 2 == 0),
            use_adrenaline_speed=bool(options.has_adrenaline and offset % 3 == 0),
            use_adrenaline_cooldown=bool(options.has_adrenaline and offset % 5 == 0),
        )
        native = legacy_state_to_native(state, game_id=game_id)
        step_react(state, player, decision)
        boundary = native.apply_react(player.player_id, decision)

        if player.finished:
            step_replenish(state, player)
            next_player = _advance_scalar_to_next_react(state, player.player_id)
        elif rules.legal_slipstream(player, list(state.active_players), state.track):
            assert boundary["kind"] == "slipstream"
            assert native.semantic_snapshot() == canonical_game_state(state)
            take = offset % 2 == 1
            step_slipstream(state, player, take)
            step_check_corner(state, player)
            boundary = native.apply_slipstream(player.player_id, take)
            next_player = None
        else:
            step_check_corner(state, player)
            next_player = None

        if boundary["kind"] == "discard":
            assert native.semantic_snapshot() == canonical_game_state(state)
            step_discard(state, player, [])
            step_replenish(state, player)
            next_player = _advance_scalar_to_next_react(state, player.player_id)
            boundary = native.apply_discard(player.player_id, [])
        elif not player.finished and next_player is None:
            step_replenish(state, player)
            next_player = _advance_scalar_to_next_react(state, player.player_id)

        expected_kind = "react" if next_player is not None else "round_complete"
        assert boundary["kind"] == expected_kind
        if next_player is not None:
            assert boundary["player_id"] == next_player
        assert native.semantic_snapshot() == canonical_game_state(state), (
            game_id,
            batch.modes[offset],
            boundary,
        )
