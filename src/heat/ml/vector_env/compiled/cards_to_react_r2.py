"""Complete non-reshuffle compiled cards-to-REACT rules for D2-R2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from heat.models.game_state import Phase
from heat.ml.spaces import CARDS_OFFSET, MAX_PLAYERS
from heat.ml.vector_env.compiled.cards_to_react import (
    _CARD_REQUIREMENTS,
    _append_zone,
    _card_token_indices,
    _compact_event_cards,
    _compact_zone,
    _pack,
    _resolve_blocking,
    prepare_action_matrix,
)
from heat.ml.vector_env.compiled.receipts import (
    MAX_EVENT_RECEIPTS,
    MAX_RECEIPT_CARDS,
    NO_VALUE,
    EventCode,
    NumericEventReceipts,
)
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    adrenaline_values,
    tensor_observations_for_indices,
    tensor_react_action_masks_for_indices,
)
from heat.ml.vector_env.random_inputs import RecordedDrawInputs
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    MAX_CARDS_PER_ZONE,
    PHASE_TO_CODE,
    TensorCardZone,
    TensorGameState,
)


_R2_EVENT_CANDIDATES = MAX_PLAYERS + MAX_PLAYERS * 9


class _EventCandidate(NamedTuple):
    """One fixed event position for every lane before active-row packing."""

    code: torch.Tensor
    active: torch.Tensor
    order: torch.Tensor
    phase: torch.Tensor
    player: torch.Tensor
    fields: torch.Tensor
    card_types: torch.Tensor
    card_values: torch.Tensor
    card_length: torch.Tensor


class R2PureOutput(NamedTuple):
    """Full tensor-only output of the non-reshuffle graph."""

    state: TensorGameState
    receipts: NumericEventReceipts
    react_players: torch.Tensor
    observations: torch.Tensor
    legal_masks: torch.Tensor


@dataclass(frozen=True)
class CompiledR2Result:
    """Public D2-R2 result reconstructed outside the compiled graph."""

    state: TensorGameState
    receipts: NumericEventReceipts
    next_decisions: TensorDecisionBatch
    observations: torch.Tensor
    legal_masks: torch.Tensor


def validate_r2_inputs(
    state: TensorGameState,
    action_matrix: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
) -> None:
    """Reject only cases that would require a deck reshuffle in R2."""
    draw_inputs.validate(state)
    if action_matrix.shape != (state.batch_size, MAX_PLAYERS):
        raise ValueError("compiled action matrix has the wrong fixed shape")
    requested = draw_inputs.lengths.sum(dim=2) + draw_inputs.replenish_lengths
    if bool(torch.any(requested > state.draw_pile.lengths)):
        raise ValueError("D2-R2 excludes all deck reshuffles")


def apply_compiled_r2(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
    *,
    compiled_function: object | None = None,
) -> CompiledR2Result:
    """Run the complete deterministic R2 graph and validate public outputs."""
    actions = prepare_action_matrix(
        state,
        decisions,
        action_indices,
        allow_forced_heat=True,
    )
    validate_r2_inputs(state, actions, draw_inputs)
    function = pure_cards_to_react_r2
    if compiled_function is not None:
        function = compiled_function  # type: ignore[assignment]
    output = function(
        state,
        actions,
        draw_inputs.card_ids,
        draw_inputs.lengths,
        draw_inputs.replenish_card_ids,
        draw_inputs.replenish_lengths,
    )
    output.state.validate()
    output.receipts.validate()
    lanes = torch.arange(output.state.batch_size, device=output.state.game_ids.device)
    if bool(torch.any(output.react_players < 0)):
        raise ValueError("lane has no real REACT decision in the D2-R2 slice")
    next_decisions = TensorDecisionBatch(
        output.state.game_ids,
        output.state.player_ids[lanes, output.react_players],
        torch.full_like(output.state.game_ids, int(TensorDecisionKind.REACT)),
    )
    next_decisions.validate()
    return CompiledR2Result(
        output.state,
        output.receipts,
        next_decisions,
        output.observations,
        output.legal_masks,
    )


def pure_cards_to_react_r2(
    state: TensorGameState,
    action_matrix: torch.Tensor,
    recorded_card_ids: torch.Tensor,
    recorded_lengths: torch.Tensor,
    replenish_card_ids: torch.Tensor,
    replenish_lengths: torch.Tensor,
) -> R2PureOutput:
    """Advance every lane to its first real REACT decision without reshuffling."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=device)
    action_valid = action_matrix >= 0
    pre_hand_types = state.hand.card_types
    pre_hand_values = state.hand.card_values
    pre_hand_lengths = state.hand.lengths
    valid_hand = slots[None, None, :] < pre_hand_lengths[:, :, None]
    token_indices = _card_token_indices(pre_hand_types, pre_hand_values)
    requirement_rows = torch.clamp(action_matrix - CARDS_OFFSET, min=0)
    requirements = _CARD_REQUIREMENTS.to(device=device)[requirement_rows]
    occurrences = torch.nn.functional.one_hot(
        torch.clamp(token_indices, min=0), num_classes=requirements.shape[-1]
    ).cumsum(dim=2)
    needed = requirements.gather(2, torch.clamp(token_indices, min=0))
    occurrence_for_card = occurrences.gather(
        3, torch.clamp(token_indices, min=0)[..., None]
    ).squeeze(3)
    selected = (
        valid_hand
        & action_valid[:, :, None]
        & (token_indices >= 0)
        & (occurrence_for_card <= needed)
    )
    played_ids, played_types, played_values, played_lengths = _compact_zone(
        state.hand.card_ids,
        pre_hand_types,
        pre_hand_values,
        selected,
    )
    hand_ids, hand_types, hand_values, hand_lengths = _compact_zone(
        state.hand.card_ids,
        pre_hand_types,
        pre_hand_values,
        valid_hand & ~selected,
    )
    playable_count = (valid_hand & (pre_hand_types != CARD_TYPE_HEAT)).sum(dim=2)
    cluttered = torch.where(
        action_valid,
        playable_count < state.gear,
        state.cluttered,
    )
    current = replace(
        state,
        current_phase=torch.full_like(
            state.current_phase, PHASE_TO_CODE[Phase.PLAY_CARDS]
        ),
        hand=TensorCardZone(hand_ids, hand_types, hand_values, hand_lengths),
        cards_played=TensorCardZone(
            played_ids, played_types, played_values, played_lengths
        ),
        cluttered=cluttered,
    )

    candidates: list[_EventCandidate] = []
    selected_types, selected_values, selected_lengths = _compact_event_cards(
        pre_hand_types, pre_hand_values, selected
    )
    for player in range(MAX_PLAYERS):
        fields = _empty_fields(batch, device)
        fields[:, 0] = cluttered[:, player].to(torch.int64)
        candidates.append(
            _candidate(
                current,
                EventCode.PLAY_CARDS,
                action_valid[:, player],
                lanes * MAX_PLAYERS + player,
                Phase.PLAY_CARDS,
                current.player_ids[:, player],
                fields,
                selected_types[:, player],
                selected_values[:, player],
                selected_lengths[:, player],
            )
        )

    unresolved = torch.ones((batch,), dtype=torch.bool, device=device)
    react_players = torch.full((batch,), -1, dtype=torch.int64, device=device)
    stage = 1
    for turn_index in range(MAX_PLAYERS):
        target_ids = current.turn_order[:, turn_index]
        target_players = torch.argmax(
            (
                (current.player_ids == target_ids[:, None])
                & current.player_present
            ).to(torch.int64),
            dim=1,
        )
        processing = unresolved & (turn_index < current.turn_order_lengths)
        target_player_ids = current.player_ids[lanes, target_players]
        current, turn_candidate = _turn_start(
            current,
            processing,
            target_players,
            pre_hand_types,
            pre_hand_values,
            pre_hand_lengths,
            stage,
        )
        candidates.append(turn_candidate)
        stage += 1

        is_cluttered = processing & current.cluttered[lanes, target_players]
        gear = current.gear.clone()
        gear[lanes, target_players] = torch.where(
            is_cluttered, 1, gear[lanes, target_players]
        )
        current = replace(current, gear=gear)
        current, replenish_candidate = _replenish(
            current,
            is_cluttered,
            target_players,
            replenish_card_ids,
            replenish_lengths,
            stage,
        )
        candidates.append(replenish_candidate)
        stage += 1

        moving = processing & ~is_cluttered
        turn_position = current.turn_start_position.clone()
        turn_lap = current.turn_start_lap.clone()
        turn_position[lanes, target_players] = torch.where(
            moving,
            current.position[lanes, target_players],
            turn_position[lanes, target_players],
        )
        turn_lap[lanes, target_players] = torch.where(
            moving,
            current.lap[lanes, target_players],
            turn_lap[lanes, target_players],
        )
        current = replace(
            current,
            turn_start_position=turn_position,
            turn_start_lap=turn_lap,
        )
        current, stress_candidates = _resolve_stress(
            current,
            moving,
            target_players,
            recorded_card_ids,
            recorded_lengths,
            stage,
        )
        candidates.extend(stress_candidates)
        stage += 4
        current, reveal_candidate, finished = _reveal_and_move(
            current, moving, target_players, stage
        )
        candidates.append(reveal_candidate)
        stage += 1
        current, finish_replenish = _replenish(
            current,
            finished,
            target_players,
            replenish_card_ids,
            replenish_lengths,
            stage,
        )
        candidates.append(finish_replenish)
        stage += 1

        reacting = moving & ~finished
        eligible, _rank = adrenaline_values(current, lanes, target_players)
        eligible &= reacting
        fields = _empty_fields(batch, device)
        fields[:, 0] = 1
        candidates.append(
            _candidate(
                current,
                EventCode.ADRENALINE_GRANTED,
                eligible,
                stage * 1_000_000 + lanes,
                Phase.ADRENALINE,
                target_player_ids,
                fields,
            )
        )
        stage += 1
        current_phase = torch.where(
            reacting,
            PHASE_TO_CODE[Phase.ADRENALINE],
            current.current_phase,
        )
        current = replace(current, current_phase=current_phase)
        react_players = torch.where(reacting, target_players, react_players)
        unresolved &= ~reacting

    next_players = torch.clamp(react_players, min=0)
    next_decisions = TensorDecisionBatch(
        current.game_ids,
        current.player_ids[lanes, next_players],
        torch.full_like(current.game_ids, int(TensorDecisionKind.REACT)),
    )
    observations = tensor_observations_for_indices(
        current, next_decisions, lanes, next_players
    )
    legal_masks = tensor_react_action_masks_for_indices(
        current, lanes, next_players
    )
    receipts = _pack_candidates(current, candidates)
    return R2PureOutput(
        current,
        receipts,
        react_players,
        observations,
        legal_masks,
    )


def _turn_start(
    state: TensorGameState,
    active: torch.Tensor,
    players: torch.Tensor,
    pre_hand_types: torch.Tensor,
    pre_hand_values: torch.Tensor,
    pre_hand_lengths: torch.Tensor,
    stage: int,
) -> tuple[TensorGameState, _EventCandidate]:
    """Record turn-start state and its exact compact receipt."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    positions = state.position[lanes, players]
    fields = _empty_fields(batch, device)
    fields[:, 0] = pre_hand_lengths[lanes, players]
    fields[:, 1] = state.gear[lanes, players]
    fields[:, 2] = state.heat_pool.lengths[lanes, players]
    fields[:, 3] = positions
    corner_starts = state.track_corners[:, :, 0]
    corner_limits = state.track_corners[:, :, 2]
    corner_slots = torch.arange(state.track_corners.shape[1], device=device)
    valid_corners = corner_slots[None, :] < state.track_corner_counts[:, None]
    distances = torch.remainder(
        corner_starts - positions[:, None], state.track_lengths[:, None]
    )
    distances = torch.where(distances == 0, state.track_lengths[:, None], distances)
    keys = torch.where(valid_corners, distances, state.track_lengths[:, None] + 1)
    next_corner = torch.argmin(keys, dim=1)
    fields[:, 4] = distances[lanes, next_corner]
    fields[:, 5] = corner_limits[lanes, next_corner]
    return state, _candidate(
        state,
        EventCode.TURN_START,
        active,
        stage * 1_000_000 + lanes,
        Phase.REVEAL_AND_MOVE,
        state.player_ids[lanes, players],
        fields,
        pre_hand_types[lanes, players],
        pre_hand_values[lanes, players],
        pre_hand_lengths[lanes, players],
    )


def _resolve_stress(
    state: TensorGameState,
    active: torch.Tensor,
    players: torch.Tensor,
    recorded_ids: torch.Tensor,
    recorded_lengths: torch.Tensor,
    stage: int,
) -> tuple[TensorGameState, list[_EventCandidate]]:
    """Resolve all recorded non-reshuffle stress flips for selected movers."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=device)
    lengths = recorded_lengths[lanes, players]
    expected_ids = recorded_ids[lanes, players]
    valid = (slots[None, None, :] < lengths[:, :, None]) & active[:, None, None]
    flat_valid = valid.flatten(start_dim=1)
    rank = torch.clamp(flat_valid.cumsum(dim=1) - 1, min=0)
    draw_lengths_before = state.draw_pile.lengths[lanes, players]
    top_indices = torch.clamp(
        draw_lengths_before[:, None] - 1 - rank,
        min=0,
        max=MAX_CARDS_PER_ZONE - 1,
    )
    source_ids = state.draw_pile.card_ids[lanes, players]
    source_types = state.draw_pile.card_types[lanes, players]
    source_values = state.draw_pile.card_values[lanes, players]
    flipped_ids = source_ids.gather(1, top_indices)
    flipped_types = source_types.gather(1, top_indices)
    flipped_values = source_values.gather(1, top_indices)
    flipped_ids = flipped_ids + (expected_ids.flatten(start_dim=1) - flipped_ids) * 0
    speed = flat_valid & (flipped_types == CARD_TYPE_SPEED)
    discarded = flat_valid & ~speed
    discard_ids, discard_types, discard_values, discard_lengths = _append_zone(
        state.discard_pile.card_ids,
        state.discard_pile.card_types,
        state.discard_pile.card_values,
        state.discard_pile.lengths,
        lanes,
        players,
        flipped_ids,
        flipped_types,
        flipped_values,
        discarded,
    )
    played_ids, played_types, played_values, played_lengths = _append_zone(
        state.cards_played.card_ids,
        state.cards_played.card_types,
        state.cards_played.card_values,
        state.cards_played.lengths,
        lanes,
        players,
        flipped_ids,
        flipped_types,
        flipped_values,
        speed,
    )
    draws = flat_valid.sum(dim=1)
    new_draw_lengths = draw_lengths_before - draws
    keep = slots[None, :] < new_draw_lengths[:, None]
    draw_ids = state.draw_pile.card_ids.clone()
    draw_types = state.draw_pile.card_types.clone()
    draw_values = state.draw_pile.card_values.clone()
    draw_lengths = state.draw_pile.lengths.clone()
    draw_ids[lanes, players] = torch.where(keep, source_ids, 0)
    draw_types[lanes, players] = torch.where(keep, source_types, 0)
    draw_values[lanes, players] = torch.where(keep, source_values, 0)
    draw_lengths[lanes, players] = new_draw_lengths
    updated = replace(
        state,
        current_phase=torch.where(
            active,
            PHASE_TO_CODE[Phase.REVEAL_AND_MOVE],
            state.current_phase,
        ),
        draw_pile=TensorCardZone(draw_ids, draw_types, draw_values, draw_lengths),
        discard_pile=TensorCardZone(
            discard_ids, discard_types, discard_values, discard_lengths
        ),
        cards_played=TensorCardZone(
            played_ids, played_types, played_values, played_lengths
        ),
    )
    shaped_types = flipped_types.reshape(batch, 4, MAX_CARDS_PER_ZONE)
    shaped_values = flipped_values.reshape(batch, 4, MAX_CARDS_PER_ZONE)
    event_discard = valid & (shaped_types != CARD_TYPE_SPEED)
    receipt_types, receipt_values, receipt_lengths = _compact_event_cards(
        shaped_types, shaped_values, event_discard
    )
    speed_values = torch.where(
        valid & (shaped_types == CARD_TYPE_SPEED), shaped_values, 0
    ).sum(dim=2)
    candidates: list[_EventCandidate] = []
    for stress_index in range(4):
        fields = _empty_fields(batch, device)
        fields[:, 0] = speed_values[:, stress_index]
        fields[:, 1] = lengths[:, stress_index]
        candidates.append(
            _candidate(
                updated,
                EventCode.STRESS_RESOLVED,
                active & (lengths[:, stress_index] > 0),
                (stage + stress_index) * 1_000_000 + lanes,
                Phase.REVEAL_AND_MOVE,
                state.player_ids[lanes, players],
                fields,
                receipt_types[:, stress_index],
                receipt_values[:, stress_index],
                receipt_lengths[:, stress_index],
            )
        )
    return updated, candidates


def _reveal_and_move(
    state: TensorGameState,
    active: torch.Tensor,
    players: torch.Tensor,
    stage: int,
) -> tuple[TensorGameState, _EventCandidate, torch.Tensor]:
    """Resolve exact traffic, lap crossing, and finish state for active movers."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=device)
    played_lengths = state.cards_played.lengths[lanes, players]
    valid = slots[None, :] < played_lengths[:, None]
    types = state.cards_played.card_types[lanes, players]
    values = state.cards_played.card_values[lanes, players]
    speed = torch.where(valid & (types != CARD_TYPE_STRESS), values, 0).sum(dim=1)
    speed_from_cards = state.speed_from_cards.clone()
    speed_from_cards[lanes, players] = torch.where(
        active, speed, speed_from_cards[lanes, players]
    )
    pre_position = state.position[lanes, players]
    target = torch.remainder(pre_position + speed, state.track_lengths)
    resolved_position = _resolve_blocking(state, players, target)
    position = state.position.clone()
    position[lanes, players] = torch.where(
        active, resolved_position, position[lanes, players]
    )
    crossed = torch.div(
        pre_position + speed, state.track_lengths, rounding_mode="floor"
    )
    lap = state.lap.clone()
    new_lap = lap[lanes, players] + crossed
    lap[lanes, players] = torch.where(active, new_lap, lap[lanes, players])
    finished_now = active & (new_lap > state.track_laps)
    finished = state.finished.clone()
    finish_order = state.finish_order.clone()
    prior_finished = state.finished.sum(dim=1)
    finished[lanes, players] |= finished_now
    finish_order[lanes, players] = torch.where(
        finished_now,
        prior_finished + 1,
        finish_order[lanes, players],
    )
    player_active = state.player_present & ~finished
    updated = replace(
        state,
        position=position,
        lap=lap,
        finished=finished,
        finish_order=finish_order,
        player_active=player_active,
        game_active=torch.any(player_active, dim=1),
        speed_from_cards=speed_from_cards,
    )
    fields = _empty_fields(batch, device)
    fields[:, 0] = speed
    fields[:, 1] = position[lanes, players]
    fields[:, 2] = lap[lanes, players]
    fields[:, 3] = finished[lanes, players].to(torch.int64)
    candidate = _candidate(
        updated,
        EventCode.REVEAL_AND_MOVE,
        active,
        stage * 1_000_000 + lanes,
        Phase.REVEAL_AND_MOVE,
        state.player_ids[lanes, players],
        fields,
    )
    return updated, candidate, finished_now


def _replenish(
    state: TensorGameState,
    active: torch.Tensor,
    players: torch.Tensor,
    recorded_ids: torch.Tensor,
    recorded_lengths: torch.Tensor,
    stage: int,
) -> tuple[TensorGameState, _EventCandidate]:
    """Replenish selected players without crossing a reshuffle boundary."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=device)
    played_lengths = state.cards_played.lengths[lanes, players]
    played_valid = slots[None, :] < played_lengths[:, None]
    discard_ids, discard_types, discard_values, discard_lengths = _append_zone(
        state.discard_pile.card_ids,
        state.discard_pile.card_types,
        state.discard_pile.card_values,
        state.discard_pile.lengths,
        lanes,
        players,
        state.cards_played.card_ids[lanes, players],
        state.cards_played.card_types[lanes, players],
        state.cards_played.card_values[lanes, players],
        played_valid & active[:, None],
    )
    played_ids = state.cards_played.card_ids.clone()
    played_types = state.cards_played.card_types.clone()
    played_values = state.cards_played.card_values.clone()
    new_played_lengths = state.cards_played.lengths.clone()
    played_ids[lanes, players] = torch.where(
        active[:, None], 0, played_ids[lanes, players]
    )
    played_types[lanes, players] = torch.where(
        active[:, None], 0, played_types[lanes, players]
    )
    played_values[lanes, players] = torch.where(
        active[:, None], 0, played_values[lanes, players]
    )
    new_played_lengths[lanes, players] = torch.where(
        active, 0, new_played_lengths[lanes, players]
    )

    draw_count = torch.where(active, recorded_lengths[lanes, players], 0)
    seven = torch.arange(7, device=device)
    draw_valid = seven[None, :] < draw_count[:, None]
    source_lengths = state.draw_pile.lengths[lanes, players]
    top_indices = torch.clamp(
        source_lengths[:, None] - 1 - seven[None, :],
        min=0,
        max=MAX_CARDS_PER_ZONE - 1,
    )
    source_ids = state.draw_pile.card_ids[lanes, players]
    source_types = state.draw_pile.card_types[lanes, players]
    source_values = state.draw_pile.card_values[lanes, players]
    drawn_ids = source_ids.gather(1, top_indices)
    drawn_types = source_types.gather(1, top_indices)
    drawn_values = source_values.gather(1, top_indices)
    expected = recorded_ids[lanes, players]
    drawn_ids = drawn_ids + (expected - drawn_ids) * 0
    hand_ids, hand_types, hand_values, hand_lengths = _append_zone(
        state.hand.card_ids,
        state.hand.card_types,
        state.hand.card_values,
        state.hand.lengths,
        lanes,
        players,
        drawn_ids,
        drawn_types,
        drawn_values,
        draw_valid,
    )
    remaining = source_lengths - draw_count
    keep = slots[None, :] < remaining[:, None]
    draw_ids = state.draw_pile.card_ids.clone()
    draw_types = state.draw_pile.card_types.clone()
    draw_values = state.draw_pile.card_values.clone()
    draw_lengths = state.draw_pile.lengths.clone()
    draw_ids[lanes, players] = torch.where(keep, source_ids, 0)
    draw_types[lanes, players] = torch.where(keep, source_types, 0)
    draw_values[lanes, players] = torch.where(keep, source_values, 0)
    draw_lengths[lanes, players] = remaining

    boost = state.boost_used_this_turn.clone()
    speed_cards = state.speed_from_cards.clone()
    speed_boost = state.speed_from_boost.clone()
    speed_adrenaline = state.speed_from_adrenaline.clone()
    slipstream = state.slipstream_moved.clone()
    cluttered = state.cluttered.clone()
    turn_position = state.turn_start_position.clone()
    for tensor in (
        boost,
        speed_cards,
        speed_boost,
        speed_adrenaline,
        slipstream,
        cluttered,
        turn_position,
    ):
        tensor[lanes, players] = torch.where(
            active,
            torch.zeros_like(tensor[lanes, players]),
            tensor[lanes, players],
        )
    current_phase = torch.where(
        active, PHASE_TO_CODE[Phase.REPLENISH], state.current_phase
    )
    updated = replace(
        state,
        current_phase=current_phase,
        hand=TensorCardZone(hand_ids, hand_types, hand_values, hand_lengths),
        draw_pile=TensorCardZone(draw_ids, draw_types, draw_values, draw_lengths),
        discard_pile=TensorCardZone(
            discard_ids, discard_types, discard_values, discard_lengths
        ),
        cards_played=TensorCardZone(
            played_ids, played_types, played_values, new_played_lengths
        ),
        boost_used_this_turn=boost.to(torch.bool),
        speed_from_cards=speed_cards,
        speed_from_boost=speed_boost,
        speed_from_adrenaline=speed_adrenaline,
        slipstream_moved=slipstream,
        cluttered=cluttered.to(torch.bool),
        turn_start_position=turn_position,
    )
    fields = _empty_fields(batch, device)
    fields[:, 0] = hand_lengths[lanes, players]
    candidate = _candidate(
        updated,
        EventCode.REPLENISH,
        active,
        stage * 1_000_000 + lanes,
        Phase.REPLENISH,
        state.player_ids[lanes, players],
        fields,
        drawn_types,
        drawn_values,
        draw_count,
    )
    return updated, candidate


def _empty_fields(batch: int, device: torch.device) -> torch.Tensor:
    """Allocate one candidate's fixed scalar fields."""
    return torch.full((batch, 8), NO_VALUE, dtype=torch.int64, device=device)


def _candidate(
    state: TensorGameState,
    code: EventCode,
    active: torch.Tensor,
    order: torch.Tensor,
    phase: Phase,
    player: torch.Tensor,
    fields: torch.Tensor,
    card_types: torch.Tensor | None = None,
    card_values: torch.Tensor | None = None,
    card_length: torch.Tensor | None = None,
) -> _EventCandidate:
    """Create one fixed numeric event candidate for every lane."""
    batch = state.game_ids.shape[0]
    device = state.game_ids.device
    empty_cards = torch.zeros(
        (batch, MAX_RECEIPT_CARDS), dtype=torch.int64, device=device
    )
    normalized_types = empty_cards if card_types is None else card_types
    normalized_values = empty_cards if card_values is None else card_values
    if normalized_types.shape[1] < MAX_RECEIPT_CARDS:
        padding = MAX_RECEIPT_CARDS - normalized_types.shape[1]
        normalized_types = torch.nn.functional.pad(normalized_types, (0, padding))
        normalized_values = torch.nn.functional.pad(normalized_values, (0, padding))
    return _EventCandidate(
        torch.full((batch,), int(code), dtype=torch.int64, device=device),
        active,
        order,
        torch.full(
            (batch,), PHASE_TO_CODE[phase], dtype=torch.int64, device=device
        ),
        player,
        fields,
        normalized_types,
        normalized_values,
        (
            torch.zeros((batch,), dtype=torch.int64, device=device)
            if card_length is None
            else card_length
        ),
    )


def _pack_candidates(
    state: TensorGameState, candidates: list[_EventCandidate]
) -> NumericEventReceipts:
    """Pack the static R2 candidate schedule into the public 64-row ABI."""
    if len(candidates) != _R2_EVENT_CANDIDATES:
        raise AssertionError("R2 event schedule changed unexpectedly")
    active = torch.stack([item.active for item in candidates], dim=1)
    slots = torch.arange(len(candidates), device=state.game_ids.device)
    order = torch.argsort(
        torch.where(active, slots, len(candidates) + slots), dim=1, stable=True
    )
    lengths = active.sum(dim=1)
    valid = slots[None, :] < lengths[:, None]

    def packed(values: list[torch.Tensor], padding: int) -> torch.Tensor:
        return _pack(
            torch.stack(values, dim=1),
            order,
            valid,
            padding,
            MAX_EVENT_RECEIPTS,
        )

    return NumericEventReceipts(
        packed([item.code for item in candidates], 0),
        packed([item.order for item in candidates], NO_VALUE),
        packed(
            [state.round_num for _item in candidates],
            0,
        ),
        packed([item.phase for item in candidates], 0),
        packed([item.player for item in candidates], NO_VALUE),
        packed([item.fields for item in candidates], NO_VALUE),
        packed([item.card_types for item in candidates], 0),
        packed([item.card_values for item in candidates], 0),
        packed([item.card_length for item in candidates], 0),
        lengths,
    )
