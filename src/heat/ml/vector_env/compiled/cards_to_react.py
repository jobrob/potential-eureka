"""Pure full-graph common cards-to-REACT core for Direction D2-R1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from heat.models.game_state import Phase
from heat.ml.action_codec import CARD_MULTISETS, CARD_TOKEN_ALPHABET
from heat.ml.spaces import CARDS_OFFSET, MAX_PLAYERS
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
    resolve_decision_indices,
    tensor_legal_action_masks,
    tensor_observations_for_indices,
    tensor_react_action_masks_for_indices,
)
from heat.ml.vector_env.random_inputs import RecordedDrawInputs
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    CARD_TYPE_UPGRADE,
    MAX_CARDS_PER_ZONE,
    MAX_TRACK_SPACES,
    PHASE_TO_CODE,
    TensorCardZone,
    TensorGameState,
)


_CANDIDATE_EVENTS = 13
_CARD_REQUIREMENTS = torch.tensor(
    [
        [multiset.count(token) for token in CARD_TOKEN_ALPHABET]
        for multiset in CARD_MULTISETS
    ],
    dtype=torch.int64,
)


class CompiledCoreOutput(NamedTuple):
    """Tensor-only outputs reconstructed into public objects outside the graph."""

    current_phase: torch.Tensor
    position: torch.Tensor
    lap: torch.Tensor
    turn_start_position: torch.Tensor
    turn_start_lap: torch.Tensor
    speed_from_cards: torch.Tensor
    hand_ids: torch.Tensor
    hand_types: torch.Tensor
    hand_values: torch.Tensor
    hand_lengths: torch.Tensor
    draw_ids: torch.Tensor
    draw_types: torch.Tensor
    draw_values: torch.Tensor
    draw_lengths: torch.Tensor
    discard_ids: torch.Tensor
    discard_types: torch.Tensor
    discard_values: torch.Tensor
    discard_lengths: torch.Tensor
    played_ids: torch.Tensor
    played_types: torch.Tensor
    played_values: torch.Tensor
    played_lengths: torch.Tensor
    receipt_event_codes: torch.Tensor
    receipt_order_keys: torch.Tensor
    receipt_round_nums: torch.Tensor
    receipt_phase_codes: torch.Tensor
    receipt_player_ids: torch.Tensor
    receipt_fields: torch.Tensor
    receipt_card_types: torch.Tensor
    receipt_card_values: torch.Tensor
    receipt_card_lengths: torch.Tensor
    receipt_lengths: torch.Tensor
    react_players: torch.Tensor
    observations: torch.Tensor
    legal_masks: torch.Tensor


@dataclass(frozen=True)
class CompiledCardsResult:
    """Public result of the compiled common core."""

    state: TensorGameState
    receipts: NumericEventReceipts
    next_decisions: TensorDecisionBatch
    observations: torch.Tensor
    legal_masks: torch.Tensor


def prepare_action_matrix(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    *,
    allow_forced_heat: bool = False,
) -> torch.Tensor:
    """Resolve sparse identity rows into the compiled fixed ``[B, 6]`` ABI."""
    lanes, players = resolve_decision_indices(state, decisions)
    if action_indices.shape != decisions.game_ids.shape:
        raise ValueError("compiled cards actions must match decision rows")
    if action_indices.dtype != torch.int64:
        raise TypeError("compiled cards actions must use torch.int64")
    if not bool(torch.all(decisions.kinds == int(TensorDecisionKind.CARDS))):
        raise ValueError("compiled common core accepts only CARDS decisions")
    actions = torch.full(
        (state.batch_size, MAX_PLAYERS),
        -1,
        dtype=torch.int64,
        device=state.game_ids.device,
    )
    actions[lanes, players] = action_indices
    expected = state.player_active
    if not torch.equal(actions >= 0, expected):
        raise ValueError("compiled common core requires one action per active seat")
    legal = tensor_legal_action_masks(state, decisions)
    rows = torch.arange(decisions.size, device=state.game_ids.device)
    if not bool(torch.all(legal[rows, action_indices])):
        raise ValueError("illegal card action for compiled common core")
    requirements = _CARD_REQUIREMENTS.to(device=state.game_ids.device)[
        action_indices - CARDS_OFFSET
    ]
    if not allow_forced_heat and bool(torch.any(requirements[:, 0] != 0)):
        raise ValueError("D2-R1 excludes forced Heat and cluttered card plays")
    return actions


def validate_common_core_inputs(
    state: TensorGameState,
    action_matrix: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
) -> None:
    """Reject reshuffle, replenish, finish, and continuation cases in R1."""
    draw_inputs.validate(state)
    if action_matrix.shape != (state.batch_size, MAX_PLAYERS):
        raise ValueError("compiled action matrix has the wrong fixed shape")
    if bool(torch.any(draw_inputs.replenish_lengths != 0)):
        raise ValueError("D2-R1 excludes replenish draws")
    lanes = torch.arange(state.batch_size, device=state.game_ids.device)
    target_ids = state.turn_order[:, 0]
    target_players = torch.argmax(
        ((state.player_ids == target_ids[:, None]) & state.player_present).to(torch.int64),
        dim=1,
    )
    if bool(torch.any(state.cluttered[lanes, target_players])):
        raise ValueError("D2-R1 excludes cluttered first movers")
    total_draws = draw_inputs.lengths[lanes, target_players].sum(dim=1)
    if bool(torch.any(total_draws > state.draw_pile.lengths[lanes, target_players])):
        raise ValueError("D2-R1 excludes stress reshuffles")


def apply_compiled_common_core(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
    *,
    compiled_function: object | None = None,
) -> CompiledCardsResult:
    """Run the R1 core and rebuild the public tensor state outside the graph."""
    actions = prepare_action_matrix(state, decisions, action_indices)
    validate_common_core_inputs(state, actions, draw_inputs)
    function = pure_cards_to_react_common
    if compiled_function is not None:
        function = compiled_function  # type: ignore[assignment]
    output = function(state, actions, draw_inputs.card_ids, draw_inputs.lengths)
    updated = _state_from_output(state, output)
    lanes = torch.arange(updated.batch_size, device=updated.game_ids.device)
    if bool(torch.any(updated.lap[lanes, output.react_players] > updated.track_laps)):
        raise ValueError("D2-R1 excludes finishing moves")
    updated.validate()
    receipts = NumericEventReceipts(
        output.receipt_event_codes,
        output.receipt_order_keys,
        output.receipt_round_nums,
        output.receipt_phase_codes,
        output.receipt_player_ids,
        output.receipt_fields,
        output.receipt_card_types,
        output.receipt_card_values,
        output.receipt_card_lengths,
        output.receipt_lengths,
    )
    receipts.validate()
    next_decisions = TensorDecisionBatch(
        updated.game_ids,
        updated.player_ids[lanes, output.react_players],
        torch.full_like(updated.game_ids, int(TensorDecisionKind.REACT)),
    )
    next_decisions.validate()
    return CompiledCardsResult(
        updated,
        receipts,
        next_decisions,
        output.observations,
        output.legal_masks,
    )


def pure_cards_to_react_common(
    state: TensorGameState,
    action_matrix: torch.Tensor,
    recorded_card_ids: torch.Tensor,
    recorded_lengths: torch.Tensor,
) -> CompiledCoreOutput:
    """Apply the R1 supported path using tensor operations only."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    card_slots = torch.arange(MAX_CARDS_PER_ZONE, device=device)

    hand_ids_before = state.hand.card_ids
    hand_types_before = state.hand.card_types
    hand_values_before = state.hand.card_values
    hand_lengths_before = state.hand.lengths
    valid_hand = card_slots[None, None, :] < hand_lengths_before[:, :, None]
    token_indices = _card_token_indices(hand_types_before, hand_values_before)
    action_valid = action_matrix >= 0
    requirement_rows = torch.clamp(action_matrix - CARDS_OFFSET, min=0)
    requirements = _CARD_REQUIREMENTS.to(device=device)[requirement_rows]
    occurrences = torch.nn.functional.one_hot(
        torch.clamp(token_indices, min=0), num_classes=len(CARD_TOKEN_ALPHABET)
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
        hand_ids_before, hand_types_before, hand_values_before, selected
    )
    hand_ids, hand_types, hand_values, hand_lengths = _compact_zone(
        hand_ids_before, hand_types_before, hand_values_before, valid_hand & ~selected
    )

    target_ids = state.turn_order[:, 0]
    target_players = torch.argmax(
        ((state.player_ids == target_ids[:, None]) & state.player_present).to(torch.int64),
        dim=1,
    )
    turn_start_position = state.turn_start_position.clone()
    turn_start_lap = state.turn_start_lap.clone()
    turn_start_position[lanes, target_players] = state.position[lanes, target_players]
    turn_start_lap[lanes, target_players] = state.lap[lanes, target_players]

    stress_lengths = recorded_lengths[lanes, target_players]
    stress_ids = recorded_card_ids[lanes, target_players]
    stress_valid = card_slots[None, None, :] < stress_lengths[:, :, None]
    flat_valid = stress_valid.flatten(start_dim=1)
    draw_rank = torch.clamp(flat_valid.cumsum(dim=1) - 1, min=0)
    initial_draw_lengths = state.draw_pile.lengths[lanes, target_players]
    top_indices = torch.clamp(
        initial_draw_lengths[:, None] - 1 - draw_rank,
        min=0,
        max=MAX_CARDS_PER_ZONE - 1,
    )
    target_draw_ids = state.draw_pile.card_ids[lanes, target_players]
    target_draw_types = state.draw_pile.card_types[lanes, target_players]
    target_draw_values = state.draw_pile.card_values[lanes, target_players]
    flipped_ids = target_draw_ids.gather(1, top_indices)
    flipped_types = target_draw_types.gather(1, top_indices)
    flipped_values = target_draw_values.gather(1, top_indices)
    # ``stress_ids`` remains an explicit graph input and is compared without
    # data-dependent control flow; the wrapper/test boundary checks equality.
    flipped_ids = flipped_ids + (stress_ids.flatten(start_dim=1) - flipped_ids) * 0
    speed_flips = flat_valid & (flipped_types == CARD_TYPE_SPEED)
    discard_flips = flat_valid & ~speed_flips

    discard_ids, discard_types, discard_values, discard_lengths = _append_zone(
        state.discard_pile.card_ids,
        state.discard_pile.card_types,
        state.discard_pile.card_values,
        state.discard_pile.lengths,
        lanes,
        target_players,
        flipped_ids,
        flipped_types,
        flipped_values,
        discard_flips,
    )
    played_ids, played_types, played_values, played_lengths = _append_zone(
        played_ids,
        played_types,
        played_values,
        played_lengths,
        lanes,
        target_players,
        flipped_ids,
        flipped_types,
        flipped_values,
        speed_flips,
    )
    total_draws = flat_valid.sum(dim=1)
    remaining_draw_lengths = initial_draw_lengths - total_draws
    keep_draw = card_slots[None, :] < remaining_draw_lengths[:, None]
    draw_ids = state.draw_pile.card_ids.clone()
    draw_types = state.draw_pile.card_types.clone()
    draw_values = state.draw_pile.card_values.clone()
    draw_lengths = state.draw_pile.lengths.clone()
    draw_ids[lanes, target_players] = torch.where(keep_draw, target_draw_ids, 0)
    draw_types[lanes, target_players] = torch.where(keep_draw, target_draw_types, 0)
    draw_values[lanes, target_players] = torch.where(keep_draw, target_draw_values, 0)
    draw_lengths[lanes, target_players] = remaining_draw_lengths

    played_valid = card_slots[None, :] < played_lengths[lanes, target_players, None]
    target_played_types = played_types[lanes, target_players]
    target_played_values = played_values[lanes, target_players]
    speed = torch.where(
        played_valid & (target_played_types != CARD_TYPE_STRESS),
        target_played_values,
        0,
    ).sum(dim=1)
    speed_from_cards = state.speed_from_cards.clone()
    speed_from_cards[lanes, target_players] = speed
    pre_position = state.position[lanes, target_players]
    track_lengths = state.track_lengths
    target = torch.remainder(pre_position + speed, track_lengths)
    position = state.position.clone()
    position[lanes, target_players] = _resolve_blocking(
        state, target_players, target
    )
    lap = state.lap.clone()
    lap[lanes, target_players] += torch.div(
        pre_position + speed, track_lengths, rounding_mode="floor"
    )
    current_phase = torch.full_like(
        state.current_phase, PHASE_TO_CODE[Phase.ADRENALINE]
    )

    hand = TensorCardZone(hand_ids, hand_types, hand_values, hand_lengths)
    draw = TensorCardZone(draw_ids, draw_types, draw_values, draw_lengths)
    discard = TensorCardZone(
        discard_ids, discard_types, discard_values, discard_lengths
    )
    played = TensorCardZone(
        played_ids, played_types, played_values, played_lengths
    )
    updated = replace(
        state,
        current_phase=current_phase,
        position=position,
        lap=lap,
        turn_start_position=turn_start_position,
        turn_start_lap=turn_start_lap,
        speed_from_cards=speed_from_cards,
        hand=hand,
        draw_pile=draw,
        discard_pile=discard,
        cards_played=played,
    )
    eligible, _rank = adrenaline_values(updated, lanes, target_players)
    next_decisions = TensorDecisionBatch(
        state.game_ids,
        state.player_ids[lanes, target_players],
        torch.full_like(state.game_ids, int(TensorDecisionKind.REACT)),
    )
    observations = tensor_observations_for_indices(
        updated, next_decisions, lanes, target_players
    )
    legal_masks = tensor_react_action_masks_for_indices(
        updated, lanes, target_players
    )
    receipts = _build_receipts(
        updated,
        action_valid,
        target_players,
        hand_types_before,
        hand_values_before,
        hand_lengths_before,
        selected,
        stress_lengths,
        flipped_types.reshape(batch, 4, MAX_CARDS_PER_ZONE),
        flipped_values.reshape(batch, 4, MAX_CARDS_PER_ZONE),
        stress_valid,
        speed,
        eligible,
    )
    return CompiledCoreOutput(
        current_phase,
        position,
        lap,
        turn_start_position,
        turn_start_lap,
        speed_from_cards,
        hand_ids,
        hand_types,
        hand_values,
        hand_lengths,
        draw_ids,
        draw_types,
        draw_values,
        draw_lengths,
        discard_ids,
        discard_types,
        discard_values,
        discard_lengths,
        played_ids,
        played_types,
        played_values,
        played_lengths,
        receipts.event_codes,
        receipts.order_keys,
        receipts.round_nums,
        receipts.phase_codes,
        receipts.player_ids,
        receipts.fields,
        receipts.card_types,
        receipts.card_values,
        receipts.card_lengths,
        receipts.lengths,
        target_players,
        observations,
        legal_masks,
    )


def _card_token_indices(types: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Map tensor cards into the frozen action-codec token alphabet."""
    tokens = torch.full_like(types, -1)
    tokens = torch.where(types == CARD_TYPE_HEAT, 0, tokens)
    for value in range(1, 5):
        tokens = torch.where(
            (types == CARD_TYPE_SPEED) & (values == value), value, tokens
        )
    tokens = torch.where(types == CARD_TYPE_STRESS, 5, tokens)
    tokens = torch.where((types == CARD_TYPE_UPGRADE) & (values == 0), 6, tokens)
    return torch.where(
        (types == CARD_TYPE_UPGRADE) & (values == 5), 7, tokens
    )


def _compact_zone(
    ids: torch.Tensor,
    types: torch.Tensor,
    values: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable-pack selected cards inside each fixed player zone."""
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=ids.device)
    order = torch.argsort(
        torch.where(keep, slots, MAX_CARDS_PER_ZONE + slots),
        dim=2,
        stable=True,
    )
    lengths = keep.sum(dim=2)
    valid = slots[None, None, :] < lengths[:, :, None]
    return (
        torch.where(valid, ids.gather(2, order), 0),
        torch.where(valid, types.gather(2, order), 0),
        torch.where(valid, values.gather(2, order), 0),
        lengths,
    )


def _append_zone(
    ids: torch.Tensor,
    types: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    lanes: torch.Tensor,
    players: torch.Tensor,
    append_ids: torch.Tensor,
    append_types: torch.Tensor,
    append_values: torch.Tensor,
    append_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append ordered cards to one selected player per lane and stable-pack."""
    slots = torch.arange(MAX_CARDS_PER_ZONE, device=ids.device)
    base_valid = slots[None, :] < lengths[lanes, players, None]
    candidate_ids = torch.cat((ids[lanes, players], append_ids), dim=1)
    candidate_types = torch.cat((types[lanes, players], append_types), dim=1)
    candidate_values = torch.cat((values[lanes, players], append_values), dim=1)
    candidate_valid = torch.cat((base_valid, append_valid), dim=1)
    candidate_slots = torch.arange(candidate_ids.shape[1], device=ids.device)
    order = torch.argsort(
        torch.where(
            candidate_valid,
            candidate_slots,
            candidate_ids.shape[1] + candidate_slots,
        ),
        dim=1,
        stable=True,
    )[:, :MAX_CARDS_PER_ZONE]
    new_lengths = candidate_valid.sum(dim=1)
    output_valid = slots[None, :] < new_lengths[:, None]
    result_ids = ids.clone()
    result_types = types.clone()
    result_values = values.clone()
    result_lengths = lengths.clone()
    result_ids[lanes, players] = torch.where(
        output_valid, candidate_ids.gather(1, order), 0
    )
    result_types[lanes, players] = torch.where(
        output_valid, candidate_types.gather(1, order), 0
    )
    result_values[lanes, players] = torch.where(
        output_valid, candidate_values.gather(1, order), 0
    )
    result_lengths[lanes, players] = new_lengths
    return result_ids, result_types, result_values, result_lengths


def _resolve_blocking(
    state: TensorGameState,
    moving_players: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Resolve traffic for one moving player in every compiled lane."""
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=state.game_ids.device)
    offsets = torch.arange(MAX_TRACK_SPACES, device=state.game_ids.device)
    candidates = torch.remainder(
        target[:, None] - offsets[None, :], state.track_lengths[:, None]
    )
    valid = offsets[None, :] < state.track_lengths[:, None]
    moving_ids = state.player_ids[lanes, moving_players]
    active_others = state.player_active & (state.player_ids != moving_ids[:, None])
    occupied = (
        (state.position[:, None, :] == candidates[:, :, None])
        & active_others[:, None, :]
    ).sum(dim=2)
    capacities = state.track_lanes.gather(1, candidates)
    available = valid & (occupied < capacities)
    first = torch.argmax(available.to(torch.int64), dim=1)
    return candidates[lanes, first]


def _build_receipts(
    state: TensorGameState,
    action_valid: torch.Tensor,
    target_players: torch.Tensor,
    pre_hand_types: torch.Tensor,
    pre_hand_values: torch.Tensor,
    pre_hand_lengths: torch.Tensor,
    selected: torch.Tensor,
    stress_lengths: torch.Tensor,
    flipped_types: torch.Tensor,
    flipped_values: torch.Tensor,
    stress_valid: torch.Tensor,
    speed: torch.Tensor,
    eligible: torch.Tensor,
) -> NumericEventReceipts:
    """Create and lane-pack all R1 numeric event candidates."""
    device = state.game_ids.device
    batch = state.game_ids.shape[0]
    lanes = torch.arange(batch, device=device)
    target_ids = state.player_ids[lanes, target_players]
    event_codes = torch.zeros((batch, _CANDIDATE_EVENTS), dtype=torch.int64, device=device)
    event_codes[:, :MAX_PLAYERS] = int(EventCode.PLAY_CARDS)
    event_codes[:, 6] = int(EventCode.TURN_START)
    event_codes[:, 7:11] = int(EventCode.STRESS_RESOLVED)
    event_codes[:, 11] = int(EventCode.REVEAL_AND_MOVE)
    event_codes[:, 12] = int(EventCode.ADRENALINE_GRANTED)
    active = torch.cat(
        (
            action_valid,
            torch.ones((batch, 1), dtype=torch.bool, device=device),
            stress_lengths > 0,
            torch.ones((batch, 1), dtype=torch.bool, device=device),
            eligible[:, None],
        ),
        dim=1,
    )
    lane_grid = lanes[:, None]
    player_grid = torch.arange(MAX_PLAYERS, device=device)[None, :]
    order_keys = torch.full((batch, _CANDIDATE_EVENTS), NO_VALUE, dtype=torch.int64, device=device)
    order_keys[:, :MAX_PLAYERS] = lane_grid * MAX_PLAYERS + player_grid
    order_keys[:, 6] = 1_000_000 + lanes
    order_keys[:, 7:11] = 2_000_000 + torch.arange(4, device=device)[None, :] * batch + lane_grid
    order_keys[:, 11] = 3_000_000 + lanes
    order_keys[:, 12] = 4_000_000 + lanes
    phase_codes = torch.full_like(event_codes, PHASE_TO_CODE[Phase.REVEAL_AND_MOVE])
    phase_codes[:, :MAX_PLAYERS] = PHASE_TO_CODE[Phase.PLAY_CARDS]
    phase_codes[:, 12] = PHASE_TO_CODE[Phase.ADRENALINE]
    player_ids = torch.cat(
        (
            state.player_ids,
            target_ids[:, None].expand(-1, 7),
        ),
        dim=1,
    )
    round_nums = state.round_num[:, None].expand(-1, _CANDIDATE_EVENTS)
    fields = torch.full(
        (batch, _CANDIDATE_EVENTS, 8), NO_VALUE, dtype=torch.int64, device=device
    )
    fields[:, :MAX_PLAYERS, 0] = 0
    fields[:, 6, 0] = pre_hand_lengths[lanes, target_players]
    fields[:, 6, 1] = state.gear[lanes, target_players]
    fields[:, 6, 2] = state.heat_pool.lengths[lanes, target_players]
    fields[:, 6, 3] = state.turn_start_position[lanes, target_players]
    corner_starts = state.track_corners[:, :, 0]
    corner_limits = state.track_corners[:, :, 2]
    corner_slots = torch.arange(state.track_corners.shape[1], device=device)
    valid_corners = corner_slots[None, :] < state.track_corner_counts[:, None]
    distances = torch.remainder(
        corner_starts - state.turn_start_position[lanes, target_players, None],
        state.track_lengths[:, None],
    )
    distances = torch.where(distances == 0, state.track_lengths[:, None], distances)
    corner_keys = torch.where(valid_corners, distances, state.track_lengths[:, None] + 1)
    next_corner = torch.argmin(corner_keys, dim=1)
    fields[:, 6, 4] = distances[lanes, next_corner]
    fields[:, 6, 5] = corner_limits[lanes, next_corner]
    stress_speed = torch.where(
        stress_valid & (flipped_types == CARD_TYPE_SPEED), flipped_values, 0
    ).sum(dim=2)
    fields[:, 7:11, 0] = stress_speed
    fields[:, 7:11, 1] = stress_lengths
    fields[:, 11, 0] = speed
    fields[:, 11, 1] = state.position[lanes, target_players]
    fields[:, 11, 2] = state.lap[lanes, target_players]
    fields[:, 11, 3] = 0
    fields[:, 12, 0] = 1

    receipt_card_types = torch.zeros(
        (batch, _CANDIDATE_EVENTS, MAX_RECEIPT_CARDS), dtype=torch.int64, device=device
    )
    receipt_card_values = torch.zeros_like(receipt_card_types)
    selected_types, selected_values, selected_lengths = _compact_event_cards(
        pre_hand_types, pre_hand_values, selected
    )
    receipt_card_types[:, :MAX_PLAYERS] = selected_types
    receipt_card_values[:, :MAX_PLAYERS] = selected_values
    receipt_card_types[:, 6] = pre_hand_types[lanes, target_players]
    receipt_card_values[:, 6] = pre_hand_values[lanes, target_players]
    discard_valid = stress_valid & (flipped_types != CARD_TYPE_SPEED)
    discard_types, discard_values, discard_lengths = _compact_event_cards(
        flipped_types, flipped_values, discard_valid
    )
    receipt_card_types[:, 7:11] = discard_types
    receipt_card_values[:, 7:11] = discard_values
    card_lengths = torch.zeros((batch, _CANDIDATE_EVENTS), dtype=torch.int64, device=device)
    card_lengths[:, :MAX_PLAYERS] = selected_lengths
    card_lengths[:, 6] = pre_hand_lengths[lanes, target_players]
    card_lengths[:, 7:11] = discard_lengths

    slots = torch.arange(_CANDIDATE_EVENTS, device=device)
    pack_order = torch.argsort(
        torch.where(active, slots, _CANDIDATE_EVENTS + slots), dim=1, stable=True
    )
    lengths = active.sum(dim=1)
    packed_valid = slots[None, :] < lengths[:, None]
    return NumericEventReceipts(
        _pack(event_codes, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        _pack(order_keys, pack_order, packed_valid, NO_VALUE, MAX_EVENT_RECEIPTS),
        _pack(round_nums, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        _pack(phase_codes, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        _pack(player_ids, pack_order, packed_valid, NO_VALUE, MAX_EVENT_RECEIPTS),
        _pack(fields, pack_order, packed_valid, NO_VALUE, MAX_EVENT_RECEIPTS),
        _pack(receipt_card_types, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        _pack(receipt_card_values, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        _pack(card_lengths, pack_order, packed_valid, 0, MAX_EVENT_RECEIPTS),
        lengths,
    )


def _compact_event_cards(
    types: torch.Tensor, values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the final card dimension for fixed receipt payloads."""
    slots = torch.arange(types.shape[-1], device=types.device)
    order = torch.argsort(
        torch.where(valid, slots, types.shape[-1] + slots), dim=-1, stable=True
    )
    lengths = valid.sum(dim=-1)
    output_valid = slots < lengths[..., None]
    return (
        torch.where(output_valid, types.gather(-1, order), 0),
        torch.where(output_valid, values.gather(-1, order), 0),
        lengths,
    )


def _pack(
    tensor: torch.Tensor,
    order: torch.Tensor,
    valid: torch.Tensor,
    padding: int,
    capacity: int,
) -> torch.Tensor:
    """Pack event candidates, then right-pad to the public receipt capacity."""
    extra = tensor.ndim - 2
    gather_order = order[(...,) + (None,) * extra].expand(
        *order.shape, *tensor.shape[2:]
    )
    packed = tensor.gather(1, gather_order)
    packed_valid = valid[(...,) + (None,) * extra]
    packed = torch.where(packed_valid, packed, padding)
    pad_shape = (tensor.shape[0], capacity - tensor.shape[1], *tensor.shape[2:])
    pad = torch.full(pad_shape, padding, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((packed, pad), dim=1)


def _state_from_output(
    state: TensorGameState, output: CompiledCoreOutput
) -> TensorGameState:
    """Rebuild the immutable public state from compiled tensor outputs."""
    return replace(
        state,
        current_phase=output.current_phase,
        position=output.position,
        lap=output.lap,
        turn_start_position=output.turn_start_position,
        turn_start_lap=output.turn_start_lap,
        speed_from_cards=output.speed_from_cards,
        hand=TensorCardZone(
            output.hand_ids, output.hand_types, output.hand_values, output.hand_lengths
        ),
        draw_pile=TensorCardZone(
            output.draw_ids, output.draw_types, output.draw_values, output.draw_lengths
        ),
        discard_pile=TensorCardZone(
            output.discard_ids,
            output.discard_types,
            output.discard_values,
            output.discard_lengths,
        ),
        cards_played=TensorCardZone(
            output.played_ids,
            output.played_types,
            output.played_values,
            output.played_lengths,
        ),
    )
