"""Tensor-native observations and GEAR/CARDS masks for Direction D2 Chunk 2.

Observations match the live codec (v4): fixed corner scales, corrected lap and
finish distance, and the public race fields in phase indices 10-16.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import torch

from heat.ml.action_codec import CARD_MULTISETS, CARD_TOKEN_ALPHABET, REACT_TABLE
from heat.ml.spaces import (
    ACTION_DIM,
    BLOCK_PHASE_CONTEXT,
    CARDS_OFFSET,
    CORNER_LENGTH_SCALE,
    CORNER_SPEED_SCALE,
    GEAR_OFFSET,
    HEAT_COST_SCALE,
    LANE_SCALE,
    MAX_CORNERS,
    MAX_PLAYERS,
    OBS_DIM,
    REACT_OFFSET,
    SPEED_LIMIT_SCALE,
    TRACK_LENGTH_SCALE,
)
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    CARD_TYPE_UPGRADE,
    MAX_CARDS_PER_ZONE,
    MAX_TRACK_SPACES,
    TensorCardZone,
    TensorGameState,
)


def _build_card_requirements() -> torch.Tensor:
    """Build the frozen codec multiset table once as per-token count rows."""
    token_index = {token: index for index, token in enumerate(CARD_TOKEN_ALPHABET)}
    rows: list[list[int]] = []
    for multiset in CARD_MULTISETS:
        counts = [0] * len(CARD_TOKEN_ALPHABET)
        for token in multiset:
            counts[token_index[token]] += 1
        rows.append(counts)
    return torch.tensor(rows, dtype=torch.int64)


_CARD_REQUIREMENTS = _build_card_requirements()


class TensorDecisionKind(IntEnum):
    """Decision kinds implemented by the Chunk-2 tensor path."""

    GEAR = 0
    CARDS = 1
    REACT = 2


@dataclass(frozen=True)
class TensorDecisionBatch:
    """Stable game/seat identities and kinds for pending tensor decisions."""

    game_ids: torch.Tensor
    player_ids: torch.Tensor
    kinds: torch.Tensor

    @classmethod
    def create(
        cls,
        game_ids: Sequence[int],
        player_ids: Sequence[int],
        kinds: Sequence[TensorDecisionKind],
        *,
        device: torch.device | str | None = None,
    ) -> TensorDecisionBatch:
        """Build a validated decision batch from small identity sequences."""
        result = cls(
            game_ids=torch.tensor(game_ids, dtype=torch.int64, device=device),
            player_ids=torch.tensor(player_ids, dtype=torch.int64, device=device),
            kinds=torch.tensor(kinds, dtype=torch.int64, device=device),
        )
        result.validate()
        return result

    @property
    def size(self) -> int:
        """Return the number of pending decision rows."""
        return int(self.game_ids.shape[0])

    def validate(self) -> None:
        """Check the fixed one-dimensional decision contract."""
        if self.game_ids.ndim != 1 or self.player_ids.ndim != 1 or self.kinds.ndim != 1:
            raise ValueError("decision fields must be one-dimensional")
        if self.size < 1:
            raise ValueError("at least one tensor decision is required")
        if self.player_ids.shape != self.game_ids.shape or self.kinds.shape != self.game_ids.shape:
            raise ValueError("decision fields must have equal lengths")
        for name, tensor in (
            ("game_ids", self.game_ids),
            ("player_ids", self.player_ids),
            ("kinds", self.kinds),
        ):
            if tensor.dtype != torch.int64:
                raise TypeError(f"{name} dtype {tensor.dtype} != torch.int64")
            if tensor.device != self.game_ids.device:
                raise ValueError("all decision fields must share one device")
        valid = (self.kinds == int(TensorDecisionKind.GEAR)) | (
            self.kinds == int(TensorDecisionKind.CARDS)
        ) | (self.kinds == int(TensorDecisionKind.REACT))
        if not bool(torch.all(valid)):
            raise ValueError("tensor path supports only GEAR, CARDS, and REACT")


def tensor_observations(
    state: TensorGameState, decisions: TensorDecisionBatch
) -> torch.Tensor:
    """Encode pending seats directly from tensor state as ``float32`` rows."""
    lanes, players = resolve_decision_indices(state, decisions)
    return tensor_observations_for_indices(state, decisions, lanes, players)


def tensor_observations_for_indices(
    state: TensorGameState,
    decisions: TensorDecisionBatch,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> torch.Tensor:
    """Encode already-resolved decision rows without Python-side validation."""
    hand_counts = _observation_hand_counts(state.hand, lanes, players)
    hand_histogram = torch.stack(
        (
            hand_counts[:, 1],
            hand_counts[:, 2],
            hand_counts[:, 3],
            hand_counts[:, 4],
            hand_counts[:, 0],
            hand_counts[:, 5],
            hand_counts[:, 6],
            hand_counts[:, 7],
        ),
        dim=1,
    ).to(torch.float64)
    hand_histogram = torch.clamp(hand_histogram / 7.0, 0.0, 1.0)

    gear = state.gear[lanes, players]
    gear_one_hot = gear[:, None] == torch.arange(
        1, 5, dtype=torch.int64, device=gear.device
    )[None, :]
    gear_one_hot = gear_one_hot.to(torch.float64)

    track_lengths = state.track_lengths[lanes]
    track_laps = state.track_laps[lanes]
    position = state.position[lanes, players]
    lap = state.lap[lanes, players]
    heat_count = state.heat_pool.lengths[lanes, players]
    finished = state.finished[lanes, players]
    kinematics = torch.stack(
        (
            _ratio(position, track_lengths),
            _ratio(lap, track_laps),
            torch.clamp(heat_count.to(torch.float64) / 6.0, 0.0, 1.0),
            finished.to(torch.float64),
        ),
        dim=1,
    )

    deck_composition = _deck_composition(state, lanes, players)
    track_block = _track_block(state, lanes, players)
    adrenaline = _adrenaline_context(state, lanes, players)
    opponents = _opponent_slots(state, lanes, players)

    phase = torch.zeros(
        (decisions.size, BLOCK_PHASE_CONTEXT),
        dtype=torch.float64,
        device=state.game_ids.device,
    )
    phase[:, 0] = (decisions.kinds == int(TensorDecisionKind.GEAR)).to(torch.float64)
    phase[:, 1] = (decisions.kinds == int(TensorDecisionKind.CARDS)).to(torch.float64)
    is_react = decisions.kinds == int(TensorDecisionKind.REACT)
    phase[:, 2] = is_react.to(torch.float64)
    react_cooldown = torch.where(gear == 1, 3, torch.where(gear == 2, 1, 0))
    react_eligible, _rank = adrenaline_values(state, lanes, players)
    phase[:, 5] = (
        (heat_count > 0) & ~state.boost_used_this_turn[lanes, players] & is_react
    ).to(torch.float64)
    phase[:, 6] = (react_eligible & is_react).to(torch.float64)
    phase[:, 7] = torch.where(
        is_react,
        torch.clamp(react_cooldown.to(torch.float64) / 3.0, 0.0, 1.0),
        0.0,
    )
    phase[:, 9] = torch.clamp(
        state.round_num[lanes].to(torch.float64) / 50.0, 0.0, 1.0
    )
    # Indices 10-16 are state, not the decision. 17-18 stay 0. Same values as
    # features._write_v4_race_context, including a decision-free +2-space probe.
    speed = (
        state.speed_from_cards[lanes, players]
        + state.speed_from_boost[lanes, players]
        + state.speed_from_adrenaline[lanes, players]
    )
    turn_start_position = state.turn_start_position[lanes, players]
    turn_start_lap = state.turn_start_lap[lanes, players]
    safe_length = torch.clamp(track_lengths, min=1)
    safe_laps = torch.clamp(track_laps, min=1)
    moved = (lap - turn_start_lap) * safe_length + (position - turn_start_position)
    phase[:, 10] = torch.clamp(
        speed.to(torch.float64) / float(CORNER_SPEED_SCALE), 0.0, 1.0
    )
    phase[:, 11] = _ratio(turn_start_position, safe_length)
    phase[:, 12] = torch.clamp(
        (lap - turn_start_lap).to(torch.float64) / safe_laps.to(torch.float64),
        -1.0,
        1.0,
    )
    heat_scale = float(HEAT_COST_SCALE)
    phase[:, 13] = torch.clamp(
        _public_heat_cost(state, lanes, players, moved, speed).to(torch.float64)
        / heat_scale,
        0.0,
        1.0,
    )
    phase[:, 14] = torch.clamp(
        _public_heat_cost(state, lanes, players, moved, speed + 1).to(torch.float64)
        / heat_scale,
        0.0,
        1.0,
    )
    phase[:, 15] = torch.clamp(
        _public_heat_cost(state, lanes, players, moved + 2, speed).to(torch.float64)
        / heat_scale,
        0.0,
        1.0,
    )
    phase[:, 16] = torch.clamp(
        track_lengths.to(torch.float64) / float(TRACK_LENGTH_SCALE), 0.0, 1.0
    )

    observation = torch.cat(
        (
            hand_histogram,
            gear_one_hot,
            kinematics,
            deck_composition,
            track_block,
            adrenaline,
            opponents,
            phase,
        ),
        dim=1,
    )
    if observation.shape != (decisions.size, OBS_DIM):
        raise AssertionError(
            f"tensor observation shape {tuple(observation.shape)} != "
            f"({decisions.size}, {OBS_DIM})"
        )
    return torch.clamp(observation, -1.0, 1.0).to(torch.float32)


def tensor_react_action_masks_for_indices(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
) -> torch.Tensor:
    """Return REACT masks for resolved seats using only tensor operations."""
    masks = torch.zeros(
        (lanes.shape[0], ACTION_DIM),
        dtype=torch.bool,
        device=state.game_ids.device,
    )
    gear = state.gear[lanes, players]
    max_cooldown = torch.where(gear == 1, 3, torch.where(gear == 2, 1, 0))
    can_boost = (state.heat_pool.lengths[lanes, players] > 0) & ~state.boost_used_this_turn[
        lanes, players
    ]
    has_adrenaline, _rank = adrenaline_values(state, lanes, players)
    table = torch.tensor(REACT_TABLE, dtype=torch.int64, device=gear.device)
    cooldown = table[:, 0]
    use_boost = table[:, 1].to(torch.bool)
    use_adrenaline_speed = table[:, 2].to(torch.bool)
    use_adrenaline_cooldown = table[:, 3].to(torch.bool)
    react_legal = cooldown[None, :] <= (
        max_cooldown[:, None] + use_adrenaline_cooldown[None, :].to(torch.int64)
    )
    react_legal &= ~use_boost[None, :] | can_boost[:, None]
    uses_adrenaline = use_adrenaline_speed | use_adrenaline_cooldown
    react_legal &= ~uses_adrenaline[None, :] | has_adrenaline[:, None]
    masks[:, REACT_OFFSET : REACT_OFFSET + len(REACT_TABLE)] = react_legal
    return masks


def tensor_legal_action_masks(
    state: TensorGameState, decisions: TensorDecisionBatch
) -> torch.Tensor:
    """Return exact flattened GEAR/CARDS legal masks without scalar rule calls."""
    lanes, players = resolve_decision_indices(state, decisions)
    masks = torch.zeros(
        (decisions.size, ACTION_DIM),
        dtype=torch.bool,
        device=state.game_ids.device,
    )
    is_gear = decisions.kinds == int(TensorDecisionKind.GEAR)
    is_cards = decisions.kinds == int(TensorDecisionKind.CARDS)
    is_react = decisions.kinds == int(TensorDecisionKind.REACT)

    if bool(torch.any(is_gear)):
        current = state.gear[lanes, players]
        heat = state.heat_pool.lengths[lanes, players]
        targets = torch.arange(1, 5, dtype=torch.int64, device=current.device)
        distance = torch.abs(targets[None, :] - current[:, None])
        gear_legal = (distance <= 1) | ((distance == 2) & (heat[:, None] >= 1))
        masks[:, GEAR_OFFSET : GEAR_OFFSET + 4] = gear_legal & is_gear[:, None]

    if bool(torch.any(is_cards)):
        token_counts, playable_count, unknown_playable = _action_hand_counts(
            state.hand, lanes, players
        )
        requirements = _card_requirements(state.game_ids.device)
        sizes = requirements.sum(dim=1)
        chosen_gear = state.gear[lanes, players]
        can_supply = torch.all(
            requirements[None, :, :] <= token_counts[:, None, :], dim=2
        )

        enough_playable = playable_count >= chosen_gear
        normal = (
            can_supply
            & (sizes[None, :] == chosen_gear[:, None])
            & (requirements[None, :, 0] == 0)
        )

        heat_needed = torch.clamp(chosen_gear - playable_count, min=0)
        forced_target = token_counts.clone()
        forced_target[:, 0] = torch.minimum(token_counts[:, 0], heat_needed)
        forced = torch.all(
            requirements[None, :, :] == forced_target[:, None, :], dim=2
        ) & (unknown_playable[:, None] == 0)
        cards_legal = torch.where(enough_playable[:, None], normal, forced)
        masks[:, CARDS_OFFSET : CARDS_OFFSET + len(CARD_MULTISETS)] = (
            cards_legal & is_cards[:, None]
        )

    if bool(torch.any(is_react)):
        gear = state.gear[lanes, players]
        max_cooldown = torch.where(gear == 1, 3, torch.where(gear == 2, 1, 0))
        can_boost = (state.heat_pool.lengths[lanes, players] > 0) & ~state.boost_used_this_turn[
            lanes, players
        ]
        has_adrenaline, _rank = adrenaline_values(state, lanes, players)
        table = torch.tensor(REACT_TABLE, dtype=torch.int64, device=gear.device)
        cooldown = table[:, 0]
        use_boost = table[:, 1].to(torch.bool)
        use_adrenaline_speed = table[:, 2].to(torch.bool)
        use_adrenaline_cooldown = table[:, 3].to(torch.bool)
        react_legal = cooldown[None, :] <= (
            max_cooldown[:, None] + use_adrenaline_cooldown[None, :].to(torch.int64)
        )
        react_legal &= ~use_boost[None, :] | can_boost[:, None]
        uses_adrenaline = use_adrenaline_speed | use_adrenaline_cooldown
        react_legal &= ~uses_adrenaline[None, :] | has_adrenaline[:, None]
        masks[:, REACT_OFFSET : REACT_OFFSET + len(REACT_TABLE)] = (
            react_legal & is_react[:, None]
        )

    if bool(torch.any(is_gear & state.spun_out[lanes, players])):
        raise ValueError("spun-out players do not yield GEAR decisions")
    return masks


def resolve_decision_indices(
    state: TensorGameState, decisions: TensorDecisionBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve stable game/player identities to tensor lane and slot indices."""
    decisions.validate()
    if decisions.game_ids.device != state.game_ids.device:
        raise ValueError("state and decisions must be on the same device")
    game_matches = decisions.game_ids[:, None] == state.game_ids[None, :]
    if not bool(torch.all(game_matches.sum(dim=1) == 1)):
        raise KeyError("each decision game_id must identify exactly one tensor lane")
    lanes = torch.argmax(game_matches.to(torch.int64), dim=1)

    lane_player_ids = state.player_ids[lanes]
    player_matches = (
        lane_player_ids == decisions.player_ids[:, None]
    ) & state.player_present[lanes]
    if not bool(torch.all(player_matches.sum(dim=1) == 1)):
        raise KeyError("each decision player_id must identify one present player")
    players = torch.argmax(player_matches.to(torch.int64), dim=1)
    if not bool(torch.all(state.player_active[lanes, players])):
        raise ValueError("finished players cannot have pending decisions")
    return lanes, players


def _valid_cards(
    zone: TensorCardZone, lanes: torch.Tensor, players: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather card types/values and a padding mask for selected seats."""
    types = zone.card_types[lanes, players]
    values = zone.card_values[lanes, players]
    indices = torch.arange(
        MAX_CARDS_PER_ZONE, dtype=torch.int64, device=types.device
    )
    valid = indices[None, :] < zone.lengths[lanes, players, None]
    return types, values, valid


def _observation_hand_counts(
    zone: TensorCardZone, lanes: torch.Tensor, players: torch.Tensor
) -> torch.Tensor:
    """Count hand categories in the observation codec's eight-slot layout."""
    types, values, valid = _valid_cards(zone, lanes, players)
    categories = (
        (types == CARD_TYPE_HEAT),
        (types == CARD_TYPE_SPEED) & (values == 1),
        (types == CARD_TYPE_SPEED) & (values == 2),
        (types == CARD_TYPE_SPEED) & (values == 3),
        (types == CARD_TYPE_SPEED) & (values == 4),
        (types == CARD_TYPE_STRESS),
        (types == CARD_TYPE_UPGRADE) & (values == 0),
        (types == CARD_TYPE_UPGRADE) & (values != 0),
    )
    return torch.stack([(category & valid).sum(dim=1) for category in categories], dim=1)


def _action_hand_counts(
    zone: TensorCardZone, lanes: torch.Tensor, players: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count canonical action tokens plus unrepresentable playable cards."""
    types, values, valid = _valid_cards(zone, lanes, players)
    categories = (
        types == CARD_TYPE_HEAT,
        (types == CARD_TYPE_SPEED) & (values == 1),
        (types == CARD_TYPE_SPEED) & (values == 2),
        (types == CARD_TYPE_SPEED) & (values == 3),
        (types == CARD_TYPE_SPEED) & (values == 4),
        types == CARD_TYPE_STRESS,
        (types == CARD_TYPE_UPGRADE) & (values == 0),
        (types == CARD_TYPE_UPGRADE) & (values == 5),
    )
    counts = torch.stack(
        [(category & valid).sum(dim=1) for category in categories], dim=1
    )
    playable = ((types != CARD_TYPE_HEAT) & valid).sum(dim=1)
    known_playable = counts[:, 1:].sum(dim=1)
    return counts, playable, playable - known_playable


def _deck_composition(
    state: TensorGameState, lanes: torch.Tensor, players: torch.Tensor
) -> torch.Tensor:
    """Encode own draw/discard composition exactly as the scalar encoder."""
    draw_types, _draw_values, draw_valid = _valid_cards(
        state.draw_pile, lanes, players
    )
    discard_types, _discard_values, discard_valid = _valid_cards(
        state.discard_pile, lanes, players
    )
    counts = []
    for card_type in (
        CARD_TYPE_SPEED,
        CARD_TYPE_HEAT,
        CARD_TYPE_STRESS,
        CARD_TYPE_UPGRADE,
    ):
        counts.append(
            ((draw_types == card_type) & draw_valid).sum(dim=1)
            + ((discard_types == card_type) & discard_valid).sum(dim=1)
        )
    draw_size = state.draw_pile.lengths[lanes, players]
    discard_size = state.discard_pile.lengths[lanes, players]
    total = draw_size + discard_size
    return torch.stack(
        tuple(_ratio(count, total) for count in counts)
        + (_ratio(draw_size, total), _ratio(discard_size, total)),
        dim=1,
    )


def _public_heat_cost(
    state: TensorGameState,
    lanes: torch.Tensor,
    players: torch.Tensor,
    spaces_moved: torch.Tensor,
    speed: torch.Tensor,
) -> torch.Tensor:
    """Sum corner heat on a geometric path. Does not mutate ``state``.

    Matches ``rules.corners_crossed``: no move pays nothing, a move of at least
    one lap pays every on-track corner once, and a shorter move pays a corner
    whose span meets the forward arc. The turn-start space itself is excluded.
    ``speed`` is cards plus boost plus adrenaline, without slipstream.
    """
    length = torch.clamp(state.track_lengths[lanes], min=1)
    start = state.turn_start_position[lanes, players]
    corners = state.track_corners[lanes]
    slot = torch.arange(MAX_CORNERS, device=length.device)
    valid = slot[None, :] < state.track_corner_counts[lanes, None]
    corner_start = corners[:, :, 0]
    corner_end = corners[:, :, 1]
    limits = corners[:, :, 2]
    low = torch.clamp(corner_start, min=0)
    high = torch.minimum(corner_end, length[:, None] - 1)
    on_track = valid & (corner_end >= corner_start) & (low <= high)

    relative_start = torch.remainder(low - start[:, None], length[:, None])
    relative_end = torch.remainder(high - start[:, None], length[:, None])
    spaces = spaces_moved[:, None]
    crosses_origin = relative_start > relative_end
    overlaps_arc = torch.where(
        crosses_origin,
        (relative_start <= spaces) | (relative_end >= 1),
        (relative_start <= spaces) & (relative_end >= 1),
    )
    moved = spaces_moved[:, None]
    hit = on_track & (moved > 0) & ((moved >= length[:, None]) | overlaps_arc)
    per_corner = torch.clamp(speed[:, None] - limits, min=0)
    return torch.where(hit, per_corner, torch.zeros_like(per_corner)).sum(dim=1)


def _track_block(
    state: TensorGameState, lanes: torch.Tensor, players: torch.Tensor
) -> torch.Tensor:
    """Encode corner slots and track globals with the live v4 scales.

    Corner limit, length, and lanes use the fixed codec divisors. Laps remaining
    count the current one-based lap, and distance to the finish uses completed
    distance. A finished car, or one already past the last lap, reads 0 for both.
    """
    length = state.track_lengths[lanes]
    position = state.position[lanes, players]
    corners = state.track_corners[lanes]
    corner_indices = torch.arange(MAX_CORNERS, device=length.device)
    valid = corner_indices[None, :] < state.track_corner_counts[lanes, None]
    starts = corners[:, :, 0]
    ends = corners[:, :, 1]
    limits = corners[:, :, 2]
    corner_lengths = ends - starts + 1

    distances = torch.remainder(starts - position[:, None], length[:, None])
    distances = torch.where(distances == 0, length[:, None], distances)
    sort_keys = torch.where(valid, distances, length[:, None] + 1)
    order = torch.argsort(sort_keys, dim=1, stable=True)

    entry_lanes = state.track_lanes[lanes].gather(
        1, torch.clamp(starts, min=0, max=MAX_TRACK_SPACES - 1)
    )
    intrinsic = torch.stack(
        (
            _ratio(distances, length[:, None]),
            torch.clamp(limits.to(torch.float64) / float(SPEED_LIMIT_SCALE), 0.0, 1.0),
            torch.clamp(
                corner_lengths.to(torch.float64) / float(CORNER_LENGTH_SCALE), 0.0, 1.0
            ),
            torch.clamp(entry_lanes.to(torch.float64) / float(LANE_SCALE), 0.0, 1.0),
        ),
        dim=2,
    )
    intrinsic = torch.where(valid[:, :, None], intrinsic, 0.0)
    ordered = intrinsic.gather(1, order[:, :, None].expand(-1, -1, 4))

    safe_length = torch.clamp(length, min=1)
    safe_laps = torch.clamp(state.track_laps[lanes], min=1)
    player_lap = state.lap[lanes, players]
    current_lap = torch.clamp(player_lap, min=1)
    done = state.finished[lanes, players] | (player_lap > safe_laps)
    total_length = safe_length * safe_laps
    completed = (current_lap - 1) * safe_length + position
    zeros = torch.zeros(done.shape, dtype=torch.float64, device=done.device)
    heat = state.heat_pool.lengths[lanes, players]
    globals_ = torch.stack(
        (
            torch.where(done, zeros, _ratio(safe_laps - current_lap + 1, safe_laps)),
            torch.where(done, zeros, _ratio(total_length - completed, total_length)),
            torch.clamp(heat.to(torch.float64) / 6.0, 0.0, 1.0),
            _ratio(position, safe_length),
        ),
        dim=1,
    )
    return torch.cat((ordered.flatten(start_dim=1), globals_), dim=1)


def _adrenaline_context(
    state: TensorGameState, lanes: torch.Tensor, players: torch.Tensor
) -> torch.Tensor:
    """Encode exact adrenaline eligibility and public rank for selected seats."""
    eligible, rank = adrenaline_values(state, lanes, players)
    return torch.stack((eligible.to(torch.float64), rank), dim=1)


def adrenaline_values(
    state: TensorGameState, lanes: torch.Tensor, players: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact eligibility and rank values shared by observations/masks."""
    present = state.player_present[lanes]
    active = state.player_active[lanes]
    all_laps = state.lap[lanes]
    all_positions = state.position[lanes]
    all_ids = state.player_ids[lanes]
    own_lap = state.lap[lanes, players]
    own_position = state.position[lanes, players]
    own_id = state.player_ids[lanes, players]

    worse = (all_laps < own_lap[:, None]) | (
        (all_laps == own_lap[:, None])
        & (
            (all_positions < own_position[:, None])
            | (
                (all_positions == own_position[:, None])
                & (all_ids < own_id[:, None])
            )
        )
    )
    worse_count = (worse & active).sum(dim=1)
    recipients = torch.where(state.starting_player_count[lanes] >= 5, 2, 1)
    eligible = (
        (active.sum(dim=1) > 1)
        & (worse_count < recipients)
        & state.player_active[lanes, players]
    )

    ahead = (all_laps > own_lap[:, None]) | (
        (all_laps == own_lap[:, None])
        & (all_positions > own_position[:, None])
    )
    ahead_count = (ahead & present & (all_ids != own_id[:, None])).sum(dim=1)
    player_count = present.sum(dim=1)
    rank = _ratio(ahead_count, player_count - 1)
    return eligible, rank


def _opponent_slots(
    state: TensorGameState, lanes: torch.Tensor, players: torch.Tensor
) -> torch.Tensor:
    """Encode opponents sorted by stable player identity with padded slots."""
    present = state.player_present[lanes]
    all_ids = state.player_ids[lanes]
    own_id = state.player_ids[lanes, players]
    opponents = present & (all_ids != own_id[:, None])
    padding_key = torch.iinfo(torch.int64).max
    sort_keys = torch.where(opponents, all_ids, padding_key)
    order = torch.argsort(sort_keys, dim=1, stable=True)[:, : MAX_PLAYERS - 1]
    opponent_present = opponents.gather(1, order)

    opponent_position = state.position[lanes].gather(1, order)
    opponent_gear = state.gear[lanes].gather(1, order)
    opponent_lap = state.lap[lanes].gather(1, order)
    opponent_finished = state.finished[lanes].gather(1, order)
    own_position = state.position[lanes, players]
    own_lap = state.lap[lanes, players]
    length = state.track_lengths[lanes]
    laps = state.track_laps[lanes]

    raw = torch.remainder(opponent_position - own_position[:, None], length[:, None])
    half = length.to(torch.float64) / 2.0
    signed = torch.where(
        raw.to(torch.float64) > half[:, None],
        raw.to(torch.float64) - length[:, None].to(torch.float64),
        raw.to(torch.float64),
    )
    relative = torch.clamp(signed / half[:, None], -1.0, 1.0)
    gear = torch.clamp(opponent_gear.to(torch.float64) / 4.0, 0.0, 1.0)
    lap_delta = torch.clamp(
        (opponent_lap - own_lap[:, None]).to(torch.float64)
        / laps[:, None].to(torch.float64),
        -1.0,
        1.0,
    )
    slots = torch.stack(
        (
            opponent_present.to(torch.float64),
            relative,
            gear,
            lap_delta,
            opponent_finished.to(torch.float64),
        ),
        dim=2,
    )
    return torch.where(opponent_present[:, :, None], slots, 0.0).flatten(start_dim=1)


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide as Python float arithmetic and clamp to the observation range."""
    safe_denominator = torch.clamp(denominator, min=1)
    return torch.clamp(
        numerator.to(torch.float64) / safe_denominator.to(torch.float64), 0.0, 1.0
    )


def _card_requirements(device: torch.device) -> torch.Tensor:
    """Return the frozen codec multiset table as per-token count rows."""
    return _CARD_REQUIREMENTS.to(device=device)
