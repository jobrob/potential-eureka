"""Exact cards-to-first-REACT tensor slice for Direction D2 Chunk 4."""

from __future__ import annotations

from collections.abc import Callable

import torch

from heat.models.game_state import Phase
from heat.ml.action_codec import CARD_MULTISETS, CARD_TOKEN_ALPHABET
from heat.ml.spaces import CARDS_OFFSET, MAX_PLAYERS
from heat.ml.vector_env.kernels.common import TensorKernelEvent, TensorKernelResult
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    adrenaline_values,
    resolve_decision_indices,
    tensor_legal_action_masks,
)
from heat.ml.vector_env.random_inputs import (
    MAX_STRESS_DRAWS,
    RecordedDrawInputs,
    reshuffle_discard_into_draw,
)
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
    TensorStateCapacityError,
)


_CARD_REQUIREMENTS = torch.tensor(
    [
        [multiset.count(token) for token in CARD_TOKEN_ALPHABET]
        for multiset in CARD_MULTISETS
    ],
    dtype=torch.int64,
)

TensorTraceCallback = Callable[
    [str, TensorGameState, torch.Tensor, torch.Tensor], None
]


def apply_cards_to_react(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
    *,
    trace: TensorTraceCallback | None = None,
) -> TensorKernelResult:
    """Apply cards and advance one seat per lane to its first REACT decision.

    Draw order is explicit and checked against the stored deck. Lanes restore
    and advance their own Python RNG state only when a deck reshuffle occurs.
    Cluttered and newly finished cars replenish, then their lanes continue to
    the next real REACT boundary.
    """
    lanes, players = resolve_decision_indices(state, decisions)
    _validate_inputs(state, decisions, action_indices, draw_inputs, lanes, players)
    updated = state.clone()
    pre_play_hands = [
        [
            _zone_displays(updated, updated.hand, lane, player)
            for player in range(MAX_PLAYERS)
        ]
        for lane in range(updated.batch_size)
    ]

    updated.current_phase.fill_(PHASE_TO_CODE[Phase.PLAY_CARDS])
    selected = _selected_cards(updated, lanes, players, action_indices)
    _apply_card_choices(updated, lanes, players, selected)
    play_events = _play_events(updated, decisions, lanes, players)
    _trace_snapshot(trace, "cards_chosen", updated, lanes, players)

    events = list(play_events)
    unresolved = torch.ones(
        updated.batch_size, dtype=torch.bool, device=updated.game_ids.device
    )
    react_players = torch.full(
        (updated.batch_size,), -1, dtype=torch.int64, device=updated.game_ids.device
    )
    for turn_index in range(MAX_PLAYERS):
        target_lanes = unresolved.nonzero(as_tuple=False).flatten()
        if len(target_lanes) == 0:
            break
        if bool(torch.any(updated.turn_order_lengths[target_lanes] <= turn_index)):
            raise ValueError("lane has no real REACT decision in the D2 vertical slice")
        target_players = _turn_players_at(updated, target_lanes, turn_index)
        events.extend(
            _turn_start_events(
                updated,
                target_lanes,
                target_players,
                pre_play_hands,
            )
        )

        cluttered = updated.cluttered[target_lanes, target_players]
        if bool(torch.any(cluttered)):
            cluttered_lanes = target_lanes[cluttered]
            cluttered_players = target_players[cluttered]
            updated.gear[cluttered_lanes, cluttered_players] = 1
            events.extend(
                _replenish(
                    updated,
                    cluttered_lanes,
                    cluttered_players,
                    draw_inputs,
                )
            )

        moving_lanes = target_lanes[~cluttered]
        moving_players = target_players[~cluttered]
        if len(moving_lanes) == 0:
            continue
        updated.turn_start_position[moving_lanes, moving_players] = updated.position[
            moving_lanes, moving_players
        ]
        updated.turn_start_lap[moving_lanes, moving_players] = updated.lap[
            moving_lanes, moving_players
        ]
        events.extend(
            _resolve_recorded_stress(
                updated,
                moving_lanes,
                moving_players,
                draw_inputs,
                trace,
            )
        )
        events.extend(
            _reveal_and_move(
                updated,
                moving_lanes,
                moving_players,
                trace,
            )
        )

        finished = updated.finished[moving_lanes, moving_players]
        if bool(torch.any(finished)):
            events.extend(
                _replenish(
                    updated,
                    moving_lanes[finished],
                    moving_players[finished],
                    draw_inputs,
                )
            )
        reacting_lanes = moving_lanes[~finished]
        reacting_players = moving_players[~finished]
        if len(reacting_lanes) == 0:
            continue
        updated.current_phase[reacting_lanes] = PHASE_TO_CODE[Phase.ADRENALINE]
        eligible, _rank = adrenaline_values(
            updated, reacting_lanes, reacting_players
        )
        events.extend(
            _adrenaline_events(
                updated,
                reacting_lanes,
                reacting_players,
                eligible,
            )
        )
        _trace_snapshot(
            trace,
            "adrenaline",
            updated,
            reacting_lanes,
            reacting_players,
        )
        react_players[reacting_lanes] = reacting_players
        unresolved[reacting_lanes] = False

    if bool(torch.any(unresolved)):
        raise ValueError("lane has no real REACT decision in the D2 vertical slice")
    next_lanes = (react_players >= 0).nonzero(as_tuple=False).flatten()
    next_decisions = TensorDecisionBatch.create(
        updated.game_ids[next_lanes].tolist(),
        updated.player_ids[next_lanes, react_players[next_lanes]].tolist(),
        [TensorDecisionKind.REACT] * len(next_lanes),
        device=updated.game_ids.device,
    )
    updated.validate()
    return TensorKernelResult(
        updated,
        tuple(events),
        next_decisions,
    )


def _trace_snapshot(
    trace: TensorTraceCallback | None,
    label: str,
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> None:
    """Send an immutable diagnostic state copy without affecting fast paths."""
    if trace is not None:
        trace(label, state.clone(), lanes.clone(), players.clone())


def _validate_inputs(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> None:
    """Validate complete card coverage, action legality, and explicit draws."""
    draw_inputs.validate(state)
    if action_indices.shape != decisions.game_ids.shape:
        raise ValueError("action_indices shape must match card decisions")
    if action_indices.dtype != torch.int64 or action_indices.device != state.game_ids.device:
        raise TypeError("card action_indices must be int64 on the state device")
    if not bool(torch.all(decisions.kinds == int(TensorDecisionKind.CARDS))):
        raise ValueError("cards-to-react kernel accepts only CARDS decisions")
    if len(set(zip(lanes.tolist(), players.tolist(), strict=True))) != decisions.size:
        raise ValueError("duplicate card decision for one game/player")
    actual = torch.zeros_like(state.player_present)
    actual[lanes, players] = True
    if not torch.equal(actual, state.player_active):
        raise ValueError("card decisions must cover every active player")

    masks = tensor_legal_action_masks(state, decisions)
    rows = torch.arange(decisions.size, device=state.game_ids.device)
    encoded = action_indices >= 0
    if bool(torch.any(encoded & ~masks[rows, torch.clamp(action_indices, min=0)])):
        raise ValueError("card action_indices contain an illegal choice")
    forced_empty = (~encoded) & ~torch.any(masks, dim=1)
    if bool(
        torch.any(
            forced_empty
            & (state.hand.lengths[lanes, players] != 0)
        )
    ) or not bool(torch.all(encoded | forced_empty)):
        raise ValueError("-1 is reserved for an unencoded empty-hand forced play")


def _turn_players_at(
    state: TensorGameState,
    lanes: torch.Tensor,
    turn_index: int,
) -> torch.Tensor:
    """Resolve one turn-order position to a player slot in selected lanes."""
    turn_ids = state.turn_order[lanes, turn_index]
    matches = (
        (state.player_ids[lanes] == turn_ids[:, None])
        & state.player_present[lanes]
    )
    if not bool(torch.all(matches.sum(dim=1) == 1)):
        raise ValueError("turn order contains an unknown player identity")
    return torch.argmax(matches.to(torch.int64), dim=1)


def _selected_cards(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
    action_indices: torch.Tensor,
) -> torch.Tensor:
    """Select the first legal concrete hand cards for each value multiset."""
    types = state.hand.card_types[lanes, players]
    values = state.hand.card_values[lanes, players]
    indices = torch.arange(MAX_CARDS_PER_ZONE, device=types.device)
    valid = indices[None, :] < state.hand.lengths[lanes, players, None]
    token_codes = torch.full_like(types, -1)
    token_codes = torch.where(types == CARD_TYPE_HEAT, 0, token_codes)
    for value in range(1, 5):
        token_codes = torch.where(
            (types == CARD_TYPE_SPEED) & (values == value), value, token_codes
        )
    token_codes = torch.where(types == CARD_TYPE_STRESS, 5, token_codes)
    token_codes = torch.where(
        (types == CARD_TYPE_UPGRADE) & (values == 0), 6, token_codes
    )
    token_codes = torch.where(
        (types == CARD_TYPE_UPGRADE) & (values == 5), 7, token_codes
    )

    requirements = torch.zeros(
        (len(action_indices), len(CARD_TOKEN_ALPHABET)),
        dtype=torch.int64,
        device=types.device,
    )
    encoded = action_indices >= 0
    if bool(torch.any(encoded)):
        requirements[encoded] = _CARD_REQUIREMENTS.to(types.device)[
            action_indices[encoded] - CARDS_OFFSET
        ]
    selected = torch.zeros_like(valid)
    for token in range(len(CARD_TOKEN_ALPHABET)):
        matches = (token_codes == token) & valid
        occurrence = torch.cumsum(matches.to(torch.int64), dim=1)
        selected |= matches & (occurrence <= requirements[:, token, None])
    if not torch.equal(selected.sum(dim=1), requirements.sum(dim=1)):
        raise ValueError("card action cannot be realized from tensor hand")
    return selected


def _apply_card_choices(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
    selected: torch.Tensor,
) -> None:
    """Move selected cards from ordered hands into ordered played zones."""
    valid = torch.arange(MAX_CARDS_PER_ZONE, device=selected.device)[None, :] < (
        state.hand.lengths[lanes, players, None]
    )
    playable = valid & (state.hand.card_types[lanes, players] != CARD_TYPE_HEAT)
    state.cluttered[lanes, players] = playable.sum(dim=1) < state.gear[lanes, players]
    _write_compacted(
        state.cards_played,
        lanes,
        players,
        state.hand,
        selected,
    )
    _write_compacted(
        state.hand,
        lanes,
        players,
        state.hand,
        valid & ~selected,
    )


def _write_compacted(
    destination: TensorCardZone,
    lanes: torch.Tensor,
    players: torch.Tensor,
    source: TensorCardZone,
    keep: torch.Tensor,
) -> None:
    """Stable-compact selected source cards into a destination zone."""
    indices = torch.arange(MAX_CARDS_PER_ZONE, device=keep.device)[None, :]
    order = torch.argsort(
        torch.where(keep, indices, MAX_CARDS_PER_ZONE + indices),
        dim=1,
        stable=True,
    )
    lengths = keep.sum(dim=1)
    valid_output = indices < lengths[:, None]
    for destination_tensor, source_tensor in (
        (destination.card_ids, source.card_ids),
        (destination.card_types, source.card_types),
        (destination.card_values, source.card_values),
    ):
        gathered = source_tensor[lanes, players].gather(1, order)
        destination_tensor[lanes, players] = torch.where(valid_output, gathered, 0)
    destination.lengths[lanes, players] = lengths


def _play_events(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> tuple[TensorKernelEvent, ...]:
    """Emit one exact play-cards event per active decision row."""
    events: list[TensorKernelEvent] = []
    for row in range(decisions.size):
        lane = int(lanes[row].item())
        player = int(players[row].item())
        events.append(
            TensorKernelEvent(
                int(decisions.game_ids[row].item()),
                int(state.round_num[lane].item()),
                Phase.PLAY_CARDS.value,
                int(decisions.player_ids[row].item()),
                "play_cards",
                {
                    "cards": _zone_displays(state, state.cards_played, lane, player),
                    "cluttered": bool(state.cluttered[lane, player].item()),
                },
            )
        )
    return tuple(events)


def _turn_start_events(
    state: TensorGameState,
    lanes: torch.Tensor,
    target_players: torch.Tensor,
    pre_play_hands: list[list[list[str]]],
) -> tuple[TensorKernelEvent, ...]:
    """Emit the driver's turn-start receipt for selected game lanes."""
    events: list[TensorKernelEvent] = []
    for row in range(len(lanes)):
        lane = int(lanes[row].item())
        player = int(target_players[row].item())
        position = int(state.position[lane, player].item())
        corner_count = int(state.track_corner_counts[lane].item())
        length = int(state.track_lengths[lane].item())
        best_distance: int | None = None
        best_limit: int | None = None
        for corner in range(corner_count):
            start = int(state.track_corners[lane, corner, 0].item())
            distance = (start - position) % length
            if distance == 0:
                distance = length
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_limit = int(state.track_corners[lane, corner, 2].item())
        hand = pre_play_hands[lane][player]
        events.append(
            TensorKernelEvent(
                int(state.game_ids[lane].item()),
                int(state.round_num[lane].item()),
                Phase.REVEAL_AND_MOVE.value,
                int(state.player_ids[lane, player].item()),
                "turn_start",
                {
                    "hand": hand,
                    "hand_size": len(hand),
                    "gear": int(state.gear[lane, player].item()),
                    "heat_available": int(state.heat_pool.lengths[lane, player].item()),
                    "position": position,
                    "next_corner_dist": best_distance,
                    "next_corner_speed_limit": best_limit,
                },
            )
        )
    return tuple(events)


def _resolve_recorded_stress(
    state: TensorGameState,
    lanes: torch.Tensor,
    target_players: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
    trace: TensorTraceCallback | None,
) -> tuple[TensorKernelEvent, ...]:
    """Resolve explicit stress flips, including exact per-lane reshuffles."""
    state.current_phase[lanes] = PHASE_TO_CODE[Phase.REVEAL_AND_MOVE]
    played_types = state.cards_played.card_types[lanes, target_players].clone()
    played_lengths = state.cards_played.lengths[lanes, target_players].clone()
    stress_count = (
        (played_types == CARD_TYPE_STRESS)
        & (
            torch.arange(MAX_CARDS_PER_ZONE, device=lanes.device)[None, :]
            < played_lengths[:, None]
        )
    ).sum(dim=1)
    events: list[TensorKernelEvent] = []

    for stress_index in range(MAX_STRESS_DRAWS):
        has_stress = stress_index < stress_count
        lengths = draw_inputs.lengths[lanes, target_players, stress_index]
        if bool(torch.any(~has_stress & (lengths != 0))):
            raise ValueError("recorded draws supplied for a missing stress card")
        discarded_names: list[list[str]] = [[] for _ in range(len(lanes))]
        resolved_values = torch.zeros(
            len(lanes), dtype=torch.int64, device=state.game_ids.device
        )
        for draw_index in range(draw_inputs.card_ids.shape[-1]):
            drawing = has_stress & (draw_index < lengths)
            if not bool(torch.any(drawing)):
                continue
            draw_rows = drawing.nonzero(as_tuple=False).flatten()
            draw_lanes = lanes[drawing]
            draw_players = target_players[drawing]
            draw_lengths = state.draw_pile.lengths[draw_lanes, draw_players]
            for local_row in (draw_lengths <= 0).nonzero(
                as_tuple=False
            ).flatten().tolist():
                reshuffle_discard_into_draw(
                    state,
                    int(draw_lanes[local_row].item()),
                    int(draw_players[local_row].item()),
                )
            draw_lengths = state.draw_pile.lengths[draw_lanes, draw_players]
            if bool(torch.any(draw_lengths <= 0)):
                raise ValueError("recorded stress draw exceeds the available deck")
            local_rows = torch.arange(len(draw_lanes), device=lanes.device)
            top_indices = draw_lengths - 1
            actual_ids = state.draw_pile.card_ids[
                draw_lanes, draw_players, top_indices
            ]
            expected_ids = draw_inputs.card_ids[
                draw_lanes, draw_players, stress_index, draw_index
            ]
            if not torch.equal(actual_ids, expected_ids):
                raise ValueError("recorded draw does not match the tensor deck top")
            card_types = state.draw_pile.card_types[
                draw_lanes, draw_players, top_indices
            ].clone()
            card_values = state.draw_pile.card_values[
                draw_lanes, draw_players, top_indices
            ].clone()
            _clear_zone_positions(
                state.draw_pile, draw_lanes, draw_players, top_indices
            )
            state.draw_pile.lengths[draw_lanes, draw_players] -= 1

            is_speed = card_types == CARD_TYPE_SPEED
            local_last = draw_index == (lengths[drawing] - 1)
            if bool(torch.any(is_speed != local_last)):
                raise ValueError("each recorded stress sequence must end at its first speed")
            if bool(torch.any(is_speed)):
                _append_cards(
                    state.cards_played,
                    draw_lanes[is_speed],
                    draw_players[is_speed],
                    actual_ids[is_speed],
                    card_types[is_speed],
                    card_values[is_speed],
                )
                resolved_values[draw_rows[is_speed]] = card_values[is_speed]
            non_speed = ~is_speed
            if bool(torch.any(non_speed)):
                _append_cards(
                    state.discard_pile,
                    draw_lanes[non_speed],
                    draw_players[non_speed],
                    actual_ids[non_speed],
                    card_types[non_speed],
                    card_values[non_speed],
                )
                for local_row in local_rows[non_speed].tolist():
                    subset_row = int(draw_rows[local_row].item())
                    discarded_names[subset_row].append(
                        _card_display(
                            int(card_types[local_row].item()),
                            int(card_values[local_row].item()),
                        )
                    )
        for row in has_stress.nonzero(as_tuple=False).flatten().tolist():
            lane = int(lanes[row].item())
            player = int(target_players[row].item())
            events.append(
                TensorKernelEvent(
                    int(state.game_ids[lane].item()),
                    int(state.round_num[lane].item()),
                    Phase.REVEAL_AND_MOVE.value,
                    int(state.player_ids[lane, player].item()),
                    "stress_resolved",
                    {
                        "value": int(resolved_values[row].item()),
                        "flipped_count": int(lengths[row].item()),
                        "discarded": discarded_names[row],
                    },
                )
            )
        if bool(torch.any(has_stress)):
            _trace_snapshot(
                trace,
                f"stress_resolved:{stress_index}",
                state,
                lanes[has_stress],
                target_players[has_stress],
            )
    return tuple(events)


def _reveal_and_move(
    state: TensorGameState,
    lanes: torch.Tensor,
    target_players: torch.Tensor,
    trace: TensorTraceCallback | None,
) -> tuple[TensorKernelEvent, ...]:
    """Reveal speed, resolve traffic, credit laps, and emit movement receipts."""
    types = state.cards_played.card_types[lanes, target_players]
    values = state.cards_played.card_values[lanes, target_players]
    valid = torch.arange(MAX_CARDS_PER_ZONE, device=lanes.device)[None, :] < (
        state.cards_played.lengths[lanes, target_players, None]
    )
    speed = torch.where(valid & (types != CARD_TYPE_STRESS), values, 0).sum(dim=1)
    state.speed_from_cards[lanes, target_players] = speed
    pre_position = state.position[lanes, target_players].clone()
    _trace_snapshot(trace, "speed_revealed", state, lanes, target_players)
    length = state.track_lengths[lanes]
    target = torch.remainder(pre_position + speed, length)
    state.position[lanes, target_players] = _resolve_blocking(
        state, lanes, target_players, target
    )
    state.lap[lanes, target_players] += torch.div(
        pre_position + speed, length, rounding_mode="floor"
    )
    would_finish = state.lap[lanes, target_players] > state.track_laps[lanes]
    if bool(torch.any(would_finish)):
        finish_lanes = lanes[would_finish]
        finish_players = target_players[would_finish]
        prior_finished = state.finished[finish_lanes].sum(dim=1)
        state.finished[finish_lanes, finish_players] = True
        state.finish_order[finish_lanes, finish_players] = prior_finished + 1
        state.player_active.copy_(state.player_present & ~state.finished)
        state.game_active.copy_(torch.any(state.player_active, dim=1))
    _trace_snapshot(trace, "movement_resolved", state, lanes, target_players)

    events: list[TensorKernelEvent] = []
    for row in range(len(lanes)):
        lane = int(lanes[row].item())
        player = int(target_players[row].item())
        events.append(
            TensorKernelEvent(
                int(state.game_ids[lane].item()),
                int(state.round_num[lane].item()),
                Phase.REVEAL_AND_MOVE.value,
                int(state.player_ids[lane, player].item()),
                "reveal_and_move",
                {
                    "speed": int(speed[row].item()),
                    "new_position": int(state.position[lane, player].item()),
                    "lap": int(state.lap[lane, player].item()),
                    "finished": bool(state.finished[lane, player].item()),
                },
            )
        )
    return tuple(events)


def _resolve_blocking(
    state: TensorGameState,
    lanes: torch.Tensor,
    moving_players: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Resolve occupied target spaces for selected moving players."""
    offsets = torch.arange(MAX_TRACK_SPACES, device=lanes.device)
    candidates = torch.remainder(
        target[:, None] - offsets[None, :], state.track_lengths[lanes, None]
    )
    valid_offsets = offsets[None, :] < state.track_lengths[lanes, None]
    positions = state.position[lanes]
    moving_ids = state.player_ids[lanes, moving_players]
    active_others = state.player_active[lanes] & (
        state.player_ids[lanes] != moving_ids[:, None]
    )
    occupied = (
        (positions[:, None, :] == candidates[:, :, None])
        & active_others[:, None, :]
    ).sum(dim=2)
    capacities = state.track_lanes[lanes].gather(1, candidates)
    available = valid_offsets & (occupied < capacities)
    first = torch.argmax(available.to(torch.int64), dim=1)
    rows = torch.arange(len(lanes), device=lanes.device)
    return candidates[rows, first]


def _adrenaline_events(
    state: TensorGameState,
    lanes: torch.Tensor,
    target_players: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[TensorKernelEvent, ...]:
    """Emit grants for selected lanes whose reacting seat is trailing."""
    events: list[TensorKernelEvent] = []
    for row in eligible.nonzero(as_tuple=False).flatten().tolist():
        lane = int(lanes[row].item())
        player = int(target_players[row].item())
        events.append(
            TensorKernelEvent(
                int(state.game_ids[lane].item()),
                int(state.round_num[lane].item()),
                Phase.ADRENALINE.value,
                int(state.player_ids[lane, player].item()),
                "adrenaline_granted",
                {"eligible": True},
            )
        )
    return tuple(events)


def _replenish(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
    draw_inputs: RecordedDrawInputs,
) -> tuple[TensorKernelEvent, ...]:
    """Replenish selected players with exact draw validation and RNG updates."""
    events: list[TensorKernelEvent] = []
    state.current_phase[lanes] = PHASE_TO_CODE[Phase.REPLENISH]
    for row in range(len(lanes)):
        lane = int(lanes[row].item())
        player = int(players[row].item())
        played_length = int(state.cards_played.lengths[lane, player].item())
        discard_length = int(state.discard_pile.lengths[lane, player].item())
        if discard_length + played_length > MAX_CARDS_PER_ZONE:
            raise TensorStateCapacityError("discard pile has no replenish capacity")
        for destination, source in (
            (state.discard_pile.card_ids, state.cards_played.card_ids),
            (state.discard_pile.card_types, state.cards_played.card_types),
            (state.discard_pile.card_values, state.cards_played.card_values),
        ):
            destination[
                lane,
                player,
                discard_length : discard_length + played_length,
            ] = source[lane, player, :played_length]
            source[lane, player].zero_()
        state.discard_pile.lengths[lane, player] += played_length
        state.cards_played.lengths[lane, player] = 0

        expected_length = int(
            draw_inputs.replenish_lengths[lane, player].item()
        )
        cards_needed = max(0, 7 - int(state.hand.lengths[lane, player].item()))
        drawn_names: list[str] = []
        for draw_index in range(cards_needed):
            if int(state.draw_pile.lengths[lane, player].item()) == 0:
                reshuffle_discard_into_draw(state, lane, player)
            draw_length = int(state.draw_pile.lengths[lane, player].item())
            if draw_length == 0:
                break
            top_index = draw_length - 1
            actual_id = state.draw_pile.card_ids[lane, player, top_index].clone()
            if draw_index >= expected_length or not torch.equal(
                actual_id,
                draw_inputs.replenish_card_ids[lane, player, draw_index],
            ):
                raise ValueError("recorded replenish draw does not match the deck top")
            card_type = state.draw_pile.card_types[lane, player, top_index].clone()
            card_value = state.draw_pile.card_values[lane, player, top_index].clone()
            _clear_zone_positions(
                state.draw_pile,
                torch.tensor([lane], dtype=torch.int64, device=lanes.device),
                torch.tensor([player], dtype=torch.int64, device=lanes.device),
                torch.tensor([top_index], dtype=torch.int64, device=lanes.device),
            )
            state.draw_pile.lengths[lane, player] -= 1
            _append_cards(
                state.hand,
                torch.tensor([lane], dtype=torch.int64, device=lanes.device),
                torch.tensor([player], dtype=torch.int64, device=lanes.device),
                actual_id[None],
                card_type[None],
                card_value[None],
            )
            drawn_names.append(
                _card_display(int(card_type.item()), int(card_value.item()))
            )
        if len(drawn_names) != expected_length:
            raise ValueError("recorded replenish draw count does not match the deck")

        state.boost_used_this_turn[lane, player] = False
        state.speed_from_cards[lane, player] = 0
        state.speed_from_boost[lane, player] = 0
        state.speed_from_adrenaline[lane, player] = 0
        state.slipstream_moved[lane, player] = 0
        state.cluttered[lane, player] = False
        state.turn_start_position[lane, player] = 0
        events.append(
            TensorKernelEvent(
                int(state.game_ids[lane].item()),
                int(state.round_num[lane].item()),
                Phase.REPLENISH.value,
                int(state.player_ids[lane, player].item()),
                "replenish",
                {
                    "hand_size": int(state.hand.lengths[lane, player].item()),
                    "drawn": drawn_names,
                },
            )
        )
    return tuple(events)


def _append_cards(
    zone: TensorCardZone,
    lanes: torch.Tensor,
    players: torch.Tensor,
    card_ids: torch.Tensor,
    card_types: torch.Tensor,
    card_values: torch.Tensor,
) -> None:
    """Append one card to each selected ordered card zone."""
    indices = zone.lengths[lanes, players]
    if bool(torch.any(indices >= MAX_CARDS_PER_ZONE)):
        raise TensorStateCapacityError("card zone has no append capacity")
    zone.card_ids[lanes, players, indices] = card_ids
    zone.card_types[lanes, players, indices] = card_types
    zone.card_values[lanes, players, indices] = card_values
    zone.lengths[lanes, players] += 1


def _clear_zone_positions(
    zone: TensorCardZone,
    lanes: torch.Tensor,
    players: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Zero selected padded card positions after a top-of-deck pop."""
    zone.card_ids[lanes, players, indices] = 0
    zone.card_types[lanes, players, indices] = 0
    zone.card_values[lanes, players, indices] = 0


def _zone_displays(
    state: TensorGameState,
    zone: TensorCardZone,
    lane: int,
    player: int,
) -> list[str]:
    """Return exact legacy display names for one ordered card zone."""
    del state  # State is kept in the signature for symmetry with later receipts.
    length = int(zone.lengths[lane, player].item())
    return [
        _card_display(
            int(zone.card_types[lane, player, index].item()),
            int(zone.card_values[lane, player, index].item()),
        )
        for index in range(length)
    ]


def _card_display(card_type: int, value: int) -> str:
    """Match ``Card.display_name`` for semantic event payloads."""
    if card_type == CARD_TYPE_SPEED:
        return str(value)
    if card_type == CARD_TYPE_HEAT:
        return "Heat"
    if card_type == CARD_TYPE_STRESS:
        return "Stress"
    if card_type == CARD_TYPE_UPGRADE:
        return f"Upgrade({value})"
    raise ValueError(f"unknown tensor card type {card_type}")
