"""Test-only bridge from legacy ``GameState`` objects to D2 tensor state."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from heat.ml.selfplay.semantic_contract import SEMANTIC_CONTRACT_VERSION
from heat.ml.spaces import MAX_CORNERS, MAX_PLAYERS
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    CARD_TYPE_UPGRADE,
    MAX_CARDS_PER_ZONE,
    MAX_SPIN_RECORDS,
    MAX_TRACK_SPACES,
    MAX_TRACK_STARTS,
    PHASE_ORDER,
    PHASE_TO_CODE,
    PYTHON_RNG_WORDS,
    TensorCardZone,
    TensorGameState,
    TensorStateCapacityError,
)
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState


_CARD_TYPE_TO_CODE = {
    CardType.SPEED: CARD_TYPE_SPEED,
    CardType.HEAT: CARD_TYPE_HEAT,
    CardType.STRESS: CARD_TYPE_STRESS,
    CardType.UPGRADE: CARD_TYPE_UPGRADE,
}
_CARD_CODE_TO_VALUE = {code: kind.value for kind, code in _CARD_TYPE_TO_CODE.items()}


def legacy_states_to_tensor(
    states: Sequence[GameState], *, game_ids: Sequence[int]
) -> TensorGameState:
    """Pack legacy states without clipping any semantic-oracle field."""
    if not states:
        raise ValueError("at least one legacy state is required")
    if len(states) != len(game_ids):
        raise ValueError("game_ids length must match states")
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("game_ids must be unique stable identities")

    for state in states:
        _check_capacities(state)

    batch = len(states)
    player_shape = (batch, MAX_PLAYERS)

    game_id_tensor = torch.tensor(game_ids, dtype=torch.int64)
    player_present = _false(*player_shape)
    player_active = _false(*player_shape)
    game_active = _false(batch)
    round_num = _zeros(batch)
    current_phase = _zeros(batch)
    turn_order = _zeros(*player_shape)
    turn_order_lengths = _zeros(batch)
    starting_player_count = _zeros(batch)
    stress_counter = _zeros(batch)

    rng_version = _zeros(batch)
    rng_words = _zeros(batch, PYTHON_RNG_WORDS)
    rng_gauss_present = _false(batch)
    rng_gauss = torch.zeros((batch,), dtype=torch.float64)

    track_lengths = _zeros(batch)
    track_space_indices = _zeros(batch, MAX_TRACK_SPACES)
    track_lanes = _zeros(batch, MAX_TRACK_SPACES)
    track_corner_counts = _zeros(batch)
    track_corners = _zeros(batch, MAX_CORNERS, 3)
    track_start_counts = _zeros(batch)
    track_start_positions = _zeros(batch, MAX_TRACK_STARTS)
    track_laps = _zeros(batch)

    player_ids = _zeros(*player_shape)
    gear = _zeros(*player_shape)
    position = _zeros(*player_shape)
    lap = _zeros(*player_shape)
    spun_out = _false(*player_shape)
    finished = _false(*player_shape)
    finish_order = _zeros(*player_shape)
    spin_log = _zeros(batch, MAX_PLAYERS, MAX_SPIN_RECORDS, 2)
    spin_log_lengths = _zeros(*player_shape)
    boost_used_this_turn = _false(*player_shape)
    speed_from_cards = _zeros(*player_shape)
    speed_from_boost = _zeros(*player_shape)
    speed_from_adrenaline = _zeros(*player_shape)
    slipstream_moved = _zeros(*player_shape)
    cluttered = _false(*player_shape)
    turn_start_position = _zeros(*player_shape)
    turn_start_lap = _zeros(*player_shape)

    all_card_ids = sorted(
        {
            card.id
            for state in states
            for player in state.players
            for zone in _player_card_zones(player)
            for card in zone
        }
    )
    vocabulary = tuple(all_card_ids)
    card_index = {card_id: index + 1 for index, card_id in enumerate(vocabulary)}
    card_zones = {
        name: _empty_card_zone(batch)
        for name in (
            "hand",
            "draw_pile",
            "discard_pile",
            "heat_pool",
            "cooldown_pool",
            "cards_played",
        )
    }

    track_names: list[str] = []
    player_names: list[tuple[str, ...]] = []
    for lane, state in enumerate(states):
        player_count = len(state.players)
        player_present[lane, :player_count] = True
        game_active[lane] = not state.is_game_over
        round_num[lane] = state.round_num
        current_phase[lane] = PHASE_TO_CODE[state.current_phase]
        turn_order_lengths[lane] = len(state.turn_order)
        turn_order[lane, : len(state.turn_order)] = torch.tensor(state.turn_order)
        starting_player_count[lane] = state.starting_player_count
        stress_counter[lane] = state._stress_counter

        rng_state = state.rng.getstate()
        rng_version[lane] = int(rng_state[0])
        words = rng_state[1]
        if len(words) != PYTHON_RNG_WORDS:
            raise TensorStateCapacityError(
                f"Python RNG state has {len(words)} words; expected {PYTHON_RNG_WORDS}"
            )
        rng_words[lane] = torch.tensor(words, dtype=torch.int64)
        if rng_state[2] is not None:
            rng_gauss_present[lane] = True
            rng_gauss[lane] = float(rng_state[2])

        track = state.track
        track_names.append(track.name)
        track_lengths[lane] = track.length
        track_space_indices[lane, : track.length] = torch.tensor(
            [space.index for space in track.spaces]
        )
        track_lanes[lane, : track.length] = torch.tensor(
            [space.lanes for space in track.spaces]
        )
        track_corner_counts[lane] = len(track.corners)
        for corner_index, corner in enumerate(track.corners):
            track_corners[lane, corner_index] = torch.tensor(
                [corner.start, corner.end, corner.speed_limit]
            )
        track_start_counts[lane] = len(track.start_positions)
        track_start_positions[lane, : len(track.start_positions)] = torch.tensor(
            track.start_positions
        )
        track_laps[lane] = track.laps

        names = ["" for _ in range(MAX_PLAYERS)]
        for slot, player in enumerate(state.players):
            names[slot] = player.name
            player_active[lane, slot] = not player.finished
            player_ids[lane, slot] = player.player_id
            gear[lane, slot] = player.gear
            position[lane, slot] = player.position
            lap[lane, slot] = player.lap
            spun_out[lane, slot] = player.spun_out
            finished[lane, slot] = player.finished
            finish_order[lane, slot] = player.finish_order
            spin_log_lengths[lane, slot] = len(player.spin_log)
            if player.spin_log:
                spin_log[lane, slot, : len(player.spin_log)] = torch.tensor(
                    player.spin_log
                )
            boost_used_this_turn[lane, slot] = player.boost_used_this_turn
            speed_from_cards[lane, slot] = player.speed_from_cards
            speed_from_boost[lane, slot] = player.speed_from_boost
            speed_from_adrenaline[lane, slot] = player.speed_from_adrenaline
            slipstream_moved[lane, slot] = player.slipstream_moved
            cluttered[lane, slot] = player.cluttered
            turn_start_position[lane, slot] = player.turn_start_position
            turn_start_lap[lane, slot] = player.turn_start_lap

            for name, cards in zip(
                card_zones,
                _player_card_zones(player),
                strict=True,
            ):
                _write_cards(
                    card_zones[name], lane, slot, cards, card_index=card_index
                )
        player_names.append(tuple(names))

    tensor_state = TensorGameState(
        game_ids=game_id_tensor,
        game_active=game_active,
        player_present=player_present,
        player_active=player_active,
        round_num=round_num,
        current_phase=current_phase,
        turn_order=turn_order,
        turn_order_lengths=turn_order_lengths,
        starting_player_count=starting_player_count,
        stress_counter=stress_counter,
        rng_version=rng_version,
        rng_words=rng_words,
        rng_gauss_present=rng_gauss_present,
        rng_gauss=rng_gauss,
        track_lengths=track_lengths,
        track_space_indices=track_space_indices,
        track_lanes=track_lanes,
        track_corner_counts=track_corner_counts,
        track_corners=track_corners,
        track_start_counts=track_start_counts,
        track_start_positions=track_start_positions,
        track_laps=track_laps,
        player_ids=player_ids,
        gear=gear,
        position=position,
        lap=lap,
        spun_out=spun_out,
        finished=finished,
        finish_order=finish_order,
        spin_log=spin_log,
        spin_log_lengths=spin_log_lengths,
        boost_used_this_turn=boost_used_this_turn,
        speed_from_cards=speed_from_cards,
        speed_from_boost=speed_from_boost,
        speed_from_adrenaline=speed_from_adrenaline,
        slipstream_moved=slipstream_moved,
        cluttered=cluttered,
        turn_start_position=turn_start_position,
        turn_start_lap=turn_start_lap,
        hand=card_zones["hand"],
        draw_pile=card_zones["draw_pile"],
        discard_pile=card_zones["discard_pile"],
        heat_pool=card_zones["heat_pool"],
        cooldown_pool=card_zones["cooldown_pool"],
        cards_played=card_zones["cards_played"],
        track_names=tuple(track_names),
        player_names=tuple(player_names),
        card_id_vocabulary=vocabulary,
    )
    tensor_state.validate()
    return tensor_state


def tensor_semantic_snapshot(
    tensor_state: TensorGameState, game_id: int
) -> dict[str, object]:
    """Rebuild a D0-comparable semantic snapshot for one stable game identity."""
    matches = (tensor_state.game_ids == game_id).nonzero(as_tuple=False).flatten()
    if len(matches) != 1:
        raise KeyError(f"unknown game_id {game_id}")
    lane = int(matches[0].item())
    player_count = int(tensor_state.player_present[lane].sum().item())
    track_length = int(tensor_state.track_lengths[lane].item())
    corner_count = int(tensor_state.track_corner_counts[lane].item())
    start_count = int(tensor_state.track_start_counts[lane].item())
    turn_count = int(tensor_state.turn_order_lengths[lane].item())

    players: list[dict[str, object]] = []
    for slot in range(player_count):
        spin_count = int(tensor_state.spin_log_lengths[lane, slot].item())
        players.append(
            {
                "player_id": _int(tensor_state.player_ids, lane, slot),
                "name": tensor_state.player_names[lane][slot],
                "gear": _int(tensor_state.gear, lane, slot),
                "position": _int(tensor_state.position, lane, slot),
                "lap": _int(tensor_state.lap, lane, slot),
                "hand": _cards(tensor_state, tensor_state.hand, lane, slot),
                "draw_pile": _cards(
                    tensor_state, tensor_state.draw_pile, lane, slot
                ),
                "discard_pile": _cards(
                    tensor_state, tensor_state.discard_pile, lane, slot
                ),
                "heat_pool": _cards(
                    tensor_state, tensor_state.heat_pool, lane, slot
                ),
                "cooldown_pool": _cards(
                    tensor_state, tensor_state.cooldown_pool, lane, slot
                ),
                "spun_out": _bool(tensor_state.spun_out, lane, slot),
                "finished": _bool(tensor_state.finished, lane, slot),
                "finish_order": _int(tensor_state.finish_order, lane, slot),
                "spin_log": tensor_state.spin_log[
                    lane, slot, :spin_count
                ].tolist(),
                "cards_played": _cards(
                    tensor_state, tensor_state.cards_played, lane, slot
                ),
                "boost_used_this_turn": _bool(
                    tensor_state.boost_used_this_turn, lane, slot
                ),
                "speed_from_cards": _int(
                    tensor_state.speed_from_cards, lane, slot
                ),
                "speed_from_boost": _int(
                    tensor_state.speed_from_boost, lane, slot
                ),
                "speed_from_adrenaline": _int(
                    tensor_state.speed_from_adrenaline, lane, slot
                ),
                "slipstream_moved": _int(
                    tensor_state.slipstream_moved, lane, slot
                ),
                "cluttered": _bool(tensor_state.cluttered, lane, slot),
                "turn_start_position": _int(
                    tensor_state.turn_start_position, lane, slot
                ),
                "turn_start_lap": _int(
                    tensor_state.turn_start_lap, lane, slot
                ),
            }
        )

    rng_gauss: float | None = None
    if bool(tensor_state.rng_gauss_present[lane].item()):
        rng_gauss = float(tensor_state.rng_gauss[lane].item())
    return {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "track": {
            "name": tensor_state.track_names[lane],
            "spaces": [
                {
                    "index": _int(tensor_state.track_space_indices, lane, index),
                    "lanes": _int(tensor_state.track_lanes, lane, index),
                }
                for index in range(track_length)
            ],
            "corners": [
                {
                    "start": _int(tensor_state.track_corners, lane, index, 0),
                    "end": _int(tensor_state.track_corners, lane, index, 1),
                    "speed_limit": _int(
                        tensor_state.track_corners, lane, index, 2
                    ),
                }
                for index in range(corner_count)
            ],
            "start_positions": tensor_state.track_start_positions[
                lane, :start_count
            ].tolist(),
            "laps": int(tensor_state.track_laps[lane].item()),
        },
        "round_num": int(tensor_state.round_num[lane].item()),
        "current_phase": PHASE_ORDER[
            int(tensor_state.current_phase[lane].item())
        ].value,
        "turn_order": tensor_state.turn_order[lane, :turn_count].tolist(),
        "starting_player_count": int(
            tensor_state.starting_player_count[lane].item()
        ),
        "stress_counter": int(tensor_state.stress_counter[lane].item()),
        "rng_state": [
            int(tensor_state.rng_version[lane].item()),
            tensor_state.rng_words[lane].tolist(),
            rng_gauss,
        ],
        "players": players,
    }


def _check_capacities(state: GameState) -> None:
    """Reject any field that cannot fit before allocating or copying tensors."""
    if not 2 <= len(state.players) <= MAX_PLAYERS:
        raise TensorStateCapacityError(
            f"player count {len(state.players)} exceeds Direction D2 range 2-{MAX_PLAYERS}"
        )
    checks = (
        ("track spaces", state.track.length, MAX_TRACK_SPACES),
        ("track corners", len(state.track.corners), MAX_CORNERS),
        ("track starts", len(state.track.start_positions), MAX_TRACK_STARTS),
        ("turn order", len(state.turn_order), MAX_PLAYERS),
    )
    for label, actual, capacity in checks:
        if actual > capacity:
            raise TensorStateCapacityError(
                f"{label} count {actual} exceeds fixed capacity {capacity}"
            )
    for player in state.players:
        if len(player.spin_log) > MAX_SPIN_RECORDS:
            raise TensorStateCapacityError(
                f"player {player.player_id} spin log count {len(player.spin_log)} "
                f"exceeds fixed capacity {MAX_SPIN_RECORDS}"
            )
        for name, cards in zip(
            (
                "hand",
                "draw pile",
                "discard pile",
                "heat pool",
                "cooldown pool",
                "cards played",
            ),
            _player_card_zones(player),
            strict=True,
        ):
            if len(cards) > MAX_CARDS_PER_ZONE:
                raise TensorStateCapacityError(
                    f"player {player.player_id} {name} count {len(cards)} "
                    f"exceeds fixed capacity {MAX_CARDS_PER_ZONE}"
                )


def _player_card_zones(player: object) -> tuple[Sequence[Card], ...]:
    """Return card zones in the TensorGameState field order."""
    from heat.models.player_state import PlayerState

    if not isinstance(player, PlayerState):
        raise TypeError("expected PlayerState")
    return (
        player.hand,
        player.deck.draw_pile,
        player.deck.discard_pile,
        player.heat_pool,
        player.cooldown_pool,
        player.cards_played,
    )


def _empty_card_zone(batch: int) -> TensorCardZone:
    """Allocate one zero-padded card zone."""
    shape = (batch, MAX_PLAYERS, MAX_CARDS_PER_ZONE)
    return TensorCardZone(
        card_ids=torch.zeros(shape, dtype=torch.int64),
        card_types=torch.zeros(shape, dtype=torch.int64),
        card_values=torch.zeros(shape, dtype=torch.int64),
        lengths=torch.zeros((batch, MAX_PLAYERS), dtype=torch.int64),
    )


def _zeros(*shape: int) -> torch.Tensor:
    """Allocate an integer tensor for exact discrete rule state."""
    return torch.zeros(shape, dtype=torch.int64)


def _false(*shape: int) -> torch.Tensor:
    """Allocate a boolean tensor for masks and rule flags."""
    return torch.zeros(shape, dtype=torch.bool)


def _write_cards(
    zone: TensorCardZone,
    lane: int,
    player: int,
    cards: Sequence[Card],
    *,
    card_index: dict[str, int],
) -> None:
    """Write one ordered card sequence into a padded zone."""
    zone.lengths[lane, player] = len(cards)
    for index, card in enumerate(cards):
        zone.card_ids[lane, player, index] = card_index[card.id]
        zone.card_types[lane, player, index] = _CARD_TYPE_TO_CODE[card.card_type]
        zone.card_values[lane, player, index] = card.value


def _cards(
    state: TensorGameState, zone: TensorCardZone, lane: int, player: int
) -> list[dict[str, object]]:
    """Decode one ordered card zone into canonical semantic dictionaries."""
    count = int(zone.lengths[lane, player].item())
    cards: list[dict[str, object]] = []
    for index in range(count):
        identifier = int(zone.card_ids[lane, player, index].item())
        type_code = int(zone.card_types[lane, player, index].item())
        cards.append(
            {
                "type": _CARD_CODE_TO_VALUE[type_code],
                "value": int(zone.card_values[lane, player, index].item()),
                "id": state.card_id_vocabulary[identifier - 1],
            }
        )
    return cards


def _int(tensor: torch.Tensor, *indices: int) -> int:
    """Read an integer scalar without leaking NumPy or torch scalar types."""
    return int(tensor[indices].item())


def _bool(tensor: torch.Tensor, *indices: int) -> bool:
    """Read a boolean scalar without leaking a torch scalar type."""
    return bool(tensor[indices].item())
