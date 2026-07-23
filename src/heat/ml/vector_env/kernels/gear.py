"""Exact batched gear-phase update for Direction D2 Chunk 3."""

from __future__ import annotations

import torch

from heat.models.game_state import Phase
from heat.ml.spaces import GEAR_OFFSET
from heat.ml.vector_env.kernels.common import TensorKernelEvent, TensorKernelResult
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    resolve_decision_indices,
    tensor_legal_action_masks,
)
from heat.ml.vector_env.state import (
    MAX_CARDS_PER_ZONE,
    PHASE_TO_CODE,
    TensorCardZone,
    TensorGameState,
    TensorStateCapacityError,
)


def apply_gear_actions(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
) -> TensorKernelResult:
    """Apply one complete simultaneous gear phase across all active lanes."""
    lanes, players = resolve_decision_indices(state, decisions)
    _validate_inputs(state, decisions, action_indices, lanes, players)
    legal_masks = tensor_legal_action_masks(state, decisions)
    rows = torch.arange(decisions.size, device=action_indices.device)
    if not bool(torch.all(legal_masks[rows, action_indices])):
        raise ValueError("gear action_indices contain an illegal choice")

    updated = state.clone()
    updated.current_phase.fill_(PHASE_TO_CODE[Phase.SHIFT_GEARS])
    old_gears = updated.gear[lanes, players].clone()
    new_gears = action_indices - GEAR_OFFSET + 1
    heat_costs = (torch.abs(new_gears - old_gears) == 2).to(torch.int64)
    updated.gear[lanes, players] = new_gears
    _pay_heat(updated, lanes, players, heat_costs)

    forced = updated.player_active & updated.spun_out
    updated.gear[forced] = 1
    events = _gear_events(
        updated,
        decisions,
        lanes,
        players,
        old_gears,
        new_gears,
        heat_costs,
    )
    next_decisions = _card_decisions(updated)
    updated.validate()
    return TensorKernelResult(updated, events, next_decisions)


def _validate_inputs(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    action_indices: torch.Tensor,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> None:
    """Require exactly one chosen action for every non-forced active seat."""
    if action_indices.shape != decisions.game_ids.shape:
        raise ValueError("action_indices shape must match decisions")
    if action_indices.dtype != torch.int64:
        raise TypeError("action_indices must use torch.int64")
    if action_indices.device != state.game_ids.device:
        raise ValueError("state, decisions, and actions must share one device")
    if not bool(torch.all(decisions.kinds == int(TensorDecisionKind.GEAR))):
        raise ValueError("gear kernel accepts only GEAR decisions")
    actual = torch.zeros_like(state.player_present)
    if len(set(zip(lanes.tolist(), players.tolist(), strict=True))) != decisions.size:
        raise ValueError("duplicate gear decision for one game/player")
    actual[lanes, players] = True
    expected = state.player_active & ~state.spun_out
    if not torch.equal(actual, expected):
        raise ValueError("gear decisions must cover every non-spun active player")


def _pay_heat(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
    heat_costs: torch.Tensor,
) -> None:
    """Move paid heat from the front of each heat pool to its discard pile."""
    paying = heat_costs == 1
    if not bool(torch.any(paying)):
        return
    pay_lanes = lanes[paying]
    pay_players = players[paying]
    if bool(torch.any(state.heat_pool.lengths[pay_lanes, pay_players] < 1)):
        raise ValueError("gear shift cannot pay required heat")
    if bool(
        torch.any(
            state.discard_pile.lengths[pay_lanes, pay_players]
            >= MAX_CARDS_PER_ZONE
        )
    ):
        raise TensorStateCapacityError("discard pile has no capacity for paid heat")

    heat_cards = _gather_zone(state.heat_pool, pay_lanes, pay_players)
    paid = tuple(component[:, 0].clone() for component in heat_cards[:3])
    for component in heat_cards[:3]:
        component[:, :-1] = component[:, 1:].clone()
        component[:, -1] = 0
    heat_cards[3] -= 1
    _write_zone(state.heat_pool, pay_lanes, pay_players, heat_cards)

    discard = _gather_zone(state.discard_pile, pay_lanes, pay_players)
    discard_rows = torch.arange(len(pay_lanes), device=pay_lanes.device)
    discard_index = discard[3]
    for component, value in zip(discard[:3], paid, strict=True):
        component[discard_rows, discard_index] = value
    discard[3] += 1
    _write_zone(state.discard_pile, pay_lanes, pay_players, discard)


def _gather_zone(
    zone: TensorCardZone, lanes: torch.Tensor, players: torch.Tensor
) -> list[torch.Tensor]:
    """Gather mutable copies of a card zone for selected seats."""
    return [
        zone.card_ids[lanes, players].clone(),
        zone.card_types[lanes, players].clone(),
        zone.card_values[lanes, players].clone(),
        zone.lengths[lanes, players].clone(),
    ]


def _write_zone(
    zone: TensorCardZone,
    lanes: torch.Tensor,
    players: torch.Tensor,
    values: list[torch.Tensor],
) -> None:
    """Write selected card-zone rows back after an exact ordered update."""
    zone.card_ids[lanes, players] = values[0]
    zone.card_types[lanes, players] = values[1]
    zone.card_values[lanes, players] = values[2]
    zone.lengths[lanes, players] = values[3]


def _gear_events(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    lanes: torch.Tensor,
    players: torch.Tensor,
    old_gears: torch.Tensor,
    new_gears: torch.Tensor,
    heat_costs: torch.Tensor,
) -> tuple[TensorKernelEvent, ...]:
    """Emit scalar-order gear events, with forced spun seats first per lane."""
    events: list[TensorKernelEvent] = []
    for lane in range(state.batch_size):
        game_id = int(state.game_ids[lane].item())
        round_num = int(state.round_num[lane].item())
        for player in range(state.player_present.shape[1]):
            if bool((state.player_active & state.spun_out)[lane, player].item()):
                events.append(
                    TensorKernelEvent(
                        game_id,
                        round_num,
                        Phase.SHIFT_GEARS.value,
                        int(state.player_ids[lane, player].item()),
                        "gear_shift",
                        {"new_gear": 1, "heat_cost": 0, "spun_out": True},
                    )
                )
        lane_rows = (lanes == lane).nonzero(as_tuple=False).flatten()
        for row_tensor in lane_rows:
            row = int(row_tensor.item())
            heat_cost = int(heat_costs[row].item())
            data: dict[str, object] = {
                "old_gear": int(old_gears[row].item()),
                "new_gear": int(new_gears[row].item()),
                "heat_cost": heat_cost,
            }
            if heat_cost:
                data["heat_available"] = int(
                    state.heat_pool.lengths[lanes[row], players[row]].item()
                )
            events.append(
                TensorKernelEvent(
                    int(decisions.game_ids[row].item()),
                    round_num,
                    Phase.SHIFT_GEARS.value,
                    int(decisions.player_ids[row].item()),
                    "gear_shift",
                    data,
                )
            )
    return tuple(events)


def _card_decisions(state: TensorGameState) -> TensorDecisionBatch:
    """Create the next simultaneous CARDS decision for every active player."""
    lanes, players = state.player_active.nonzero(as_tuple=True)
    return TensorDecisionBatch.create(
        state.game_ids[lanes].tolist(),
        state.player_ids[lanes, players].tolist(),
        [TensorDecisionKind.CARDS] * len(lanes),
        device=state.game_ids.device,
    )
