"""Test-only bridge between legacy/D2 state and the compact native A1 state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct
from typing import Any

import numpy as np
import torch

from heat.ml.vector_env.bridge import (
    legacy_states_to_tensor,
)
from heat.ml.vector_env.state import PHASE_ORDER, TensorGameState
from heat.models.game_state import GameState
from heat.models.cards import Card, CardType
from heat.engine.phases import ReactDecision
from heat.engine.driver import Decision, DecisionKind


_ZONE_NAMES = (
    "hand",
    "draw_pile",
    "discard_pile",
    "heat_pool",
    "cooldown_pool",
    "cards_played",
)


@dataclass(frozen=True)
class NativeStateBridge:
    """Own one opaque native state plus immutable Python identity metadata."""

    _native: Any
    _template: TensorGameState

    @property
    def native_handle(self) -> Any:
        """Return the opaque handle accepted by bulk native pool admission."""
        return self._native

    @property
    def round_num(self) -> int:
        """Return the native round counter without exporting the full state."""
        return int(self._native.round_num)

    def reward_state(self, player_id: int) -> tuple[int, int, bool]:
        """Capture the three previous-state fields used by the A8 reward."""
        position, lap, spun_out = self._native.reward_state(player_id)
        return int(position), int(lap), bool(spun_out)

    def reward_from_fields(
        self,
        player_id: int,
        previous: tuple[int, int, bool],
        done: bool,
        *,
        terminated: bool = False,
        shaping_weight: float = 0.0,
        spinout_weight: float = 0.0,
    ) -> float:
        """Compute reward from a compact pending transition snapshot."""
        position, lap, spun_out = previous
        return float(
            self._native.reward(
                player_id,
                position,
                lap,
                spun_out,
                done,
                terminated,
                False,
                shaping_weight,
                spinout_weight,
            )
        )

    def to_tensor_state(self) -> TensorGameState:
        """Decode the compact test payload into the established D2 state oracle."""
        payload = self._native.export_payload()
        card_ids, vocabulary = _decode_reserved_cards(
            payload["card_ids"], self._template.card_id_vocabulary
        )
        state = replace(
            self._template.clone(), card_id_vocabulary=vocabulary
        )
        control = payload["control"]
        state.game_ids[0] = int(control[0])
        state.game_active[0] = bool(control[1])
        state.round_num[0] = int(control[2])
        state.current_phase[0] = int(control[3])
        state.turn_order_lengths[0] = int(control[4])
        state.starting_player_count[0] = int(control[5])
        state.stress_counter[0] = int(control[6])
        state.rng_version[0] = int(control[7])
        state.rng_gauss_present[0] = bool(control[8])
        state.rng_gauss[0] = float(payload["rng_gauss"][0])
        state.rng_words[0].copy_(
            torch.from_numpy(payload["rng_words"].astype(np.int64))
        )
        state.turn_order[0].copy_(_tensor_i64(payload["turn_order"]))

        track_control = payload["track_control"]
        state.track_lengths[0] = int(track_control[0])
        state.track_corner_counts[0] = int(track_control[1])
        state.track_start_counts[0] = int(track_control[2])
        state.track_laps[0] = int(track_control[3])
        state.track_space_indices[0].copy_(_tensor_i64(payload["track_indices"]))
        state.track_lanes[0].copy_(_tensor_i64(payload["track_lanes"]))
        state.track_corners[0].copy_(_tensor_i64(payload["track_corners"]))
        state.track_start_positions[0].copy_(_tensor_i64(payload["track_starts"]))

        players = payload["players"]
        player_fields = (
            (state.player_present, 0, torch.bool),
            (state.player_active, 1, torch.bool),
            (state.player_ids, 2, torch.int64),
            (state.gear, 3, torch.int64),
            (state.position, 4, torch.int64),
            (state.lap, 5, torch.int64),
            (state.spun_out, 6, torch.bool),
            (state.finished, 7, torch.bool),
            (state.finish_order, 8, torch.int64),
            (state.spin_log_lengths, 9, torch.int64),
            (state.boost_used_this_turn, 10, torch.bool),
            (state.speed_from_cards, 11, torch.int64),
            (state.speed_from_boost, 12, torch.int64),
            (state.speed_from_adrenaline, 13, torch.int64),
            (state.slipstream_moved, 14, torch.int64),
            (state.cluttered, 15, torch.bool),
            (state.turn_start_position, 16, torch.int64),
            (state.turn_start_lap, 17, torch.int64),
        )
        for target, column, dtype in player_fields:
            target[0].copy_(torch.from_numpy(players[:, column]).to(dtype=dtype))
        state.spin_log[0].copy_(_tensor_i64(payload["spin_log"]))

        for zone_index, zone_name in enumerate(_ZONE_NAMES):
            zone = getattr(state, zone_name)
            if np.any(payload["zone_lengths"][zone_index] > zone.card_ids.shape[2]):
                raise ValueError("native card zone exceeds the legacy D2 bridge capacity")
            width = zone.card_ids.shape[2]
            zone.card_ids[0].copy_(
                _tensor_i64(card_ids[zone_index, :, :width])
            )
            zone.card_types[0].copy_(
                _tensor_i64(payload["card_types"][zone_index, :, :width])
            )
            zone.card_values[0].copy_(
                _tensor_i64(payload["card_values"][zone_index, :, :width])
            )
            zone.lengths[0].copy_(
                _tensor_i64(payload["zone_lengths"][zone_index])
            )
        state.validate()
        return state

    def semantic_snapshot(self) -> dict[str, object]:
        """Return the exact canonical D0 snapshot after a native round trip."""
        return _semantic_snapshot_from_payload(
            self._native.export_payload(), self._template
        )

    def receipt(self) -> dict[str, object]:
        """Expose compact layout capacities without exposing native fields."""
        return dict(self._native.receipt())

    def observation(self, player_id: int, decision: Decision | None) -> np.ndarray:
        """Encode one policy or value observation entirely in native code."""
        kind = 0 if decision is None else _DECISION_KIND_CODE[decision.kind]
        return np.asarray(self._native.observation(player_id, kind), dtype=np.float32)

    def observation_for_kind(self, player_id: int, kind: int) -> np.ndarray:
        """Encode one row from the native program-counter decision code."""
        return np.asarray(self._native.observation(player_id, kind), dtype=np.float32)

    def legal_mask(self, decision: Decision) -> np.ndarray:
        """Return the native frozen-codec legal mask for one decision."""
        kind = _DECISION_KIND_CODE[decision.kind]
        return np.asarray(
            self._native.legal_mask(decision.player_id, kind), dtype=np.bool_
        )

    def legal_mask_for_kind(self, player_id: int, kind: int) -> np.ndarray:
        """Return the frozen legal mask for a native decision code."""
        return np.asarray(self._native.legal_mask(player_id, kind), dtype=np.bool_)

    def bootstrap_row(self, player_id: int) -> dict[str, object]:
        """Return a value-only observation row for a truncated game."""
        return dict(self._native.bootstrap_row(player_id))

    def reward(
        self,
        previous: GameState,
        player_id: int,
        done: bool,
        *,
        terminated: bool = False,
        reward_mode: str = "race",
        shaping_weight: float = 0.0,
        spinout_weight: float = 0.0,
    ) -> float:
        """Compute the frozen step reward from its minimal previous-state fields."""
        player = previous.get_player(player_id)
        return float(
            self._native.reward(
                player_id,
                player.position,
                player.lap,
                player.spun_out,
                done,
                terminated,
                reward_mode == "solo",
                shaping_weight,
                spinout_weight,
            )
        )

    def terminal_margin(self, player_id: int) -> float:
        """Return the native normalized lead over the best opponent."""
        return float(self._native.terminal_margin(player_id))

    def canonical_digest(self) -> int:
        """Return the compact state and RNG digest used by full-race gates."""
        return int(self._native.canonical_digest())

    def boundary_receipt(self) -> np.ndarray:
        """Return the fixed numeric verification receipt for the current pause."""
        return np.asarray(self._native.boundary_receipt(), dtype=np.uint64)

    def apply_cards_to_react(
        self, choices: dict[int, tuple[Card, ...]]
    ) -> dict[str, object]:
        """Apply simultaneous card identities and advance to the first REACT."""
        ids = np.zeros((6, 4), dtype=np.int64)
        lengths = np.zeros((6,), dtype=np.int64)
        player_ids = self._template.player_ids[0].tolist()
        vocabulary = {
            card_id: index + 1
            for index, card_id in enumerate(self._template.card_id_vocabulary)
        }
        for player_id, cards in choices.items():
            slot = player_ids.index(player_id)
            lengths[slot] = len(cards)
            ids[slot, : len(cards)] = [vocabulary[card.id] for card in cards]
        return dict(self._native.apply_cards_to_react(ids, lengths))

    def start_round(self) -> dict[str, object]:
        """Begin one checked native round and expose its simultaneous gear rows."""
        return dict(self._native.start_round())

    def apply_gears(
        self, choices: dict[int, tuple[int, int]]
    ) -> dict[str, object]:
        """Apply a complete simultaneous gear group in stable seat order."""
        gears = np.full((6,), -1, dtype=np.int64)
        costs = np.full((6,), -1, dtype=np.int64)
        player_ids = self._template.player_ids[0].tolist()
        for player_id, (gear, heat_cost) in choices.items():
            slot = player_ids.index(player_id)
            gears[slot] = gear
            costs[slot] = heat_cost
        return dict(self._native.apply_gears(gears, costs))

    def apply_gear_actions(self, actions: dict[int, int]) -> dict[str, object]:
        """Apply a simultaneous group of frozen-codec gear indices natively."""
        rows = np.full((6,), -1, dtype=np.int64)
        player_ids = self._template.player_ids[0].tolist()
        for player_id, action in actions.items():
            rows[player_ids.index(player_id)] = action
        return dict(self._native.apply_gear_actions(rows))

    def apply_card_actions(self, actions: dict[int, int]) -> dict[str, object]:
        """Decode and apply simultaneous frozen-codec card indices natively."""
        rows = np.full((6,), -1, dtype=np.int64)
        player_ids = self._template.player_ids[0].tolist()
        for player_id, action in actions.items():
            rows[player_ids.index(player_id)] = action
        return dict(self._native.apply_card_actions(rows))

    def apply_flat_action(self, player_id: int, action: int) -> dict[str, object]:
        """Decode and apply one sequential frozen-codec action natively."""
        return dict(self._native.apply_flat_action(player_id, action))

    def apply_react(
        self, player_id: int, decision: ReactDecision
    ) -> dict[str, object]:
        """Apply one REACT decision and stop at the next real decision."""
        return dict(
            self._native.apply_react(
                player_id,
                decision.cooldown_count,
                decision.use_boost,
                decision.use_adrenaline_speed,
                decision.use_adrenaline_cooldown,
            )
        )

    def apply_slipstream(self, player_id: int, take: bool) -> dict[str, object]:
        """Apply a slipstream decision and stop at the next real decision."""
        return dict(self._native.apply_slipstream(player_id, take))

    def apply_discard(
        self, player_id: int, cards: list[Card]
    ) -> dict[str, object]:
        """Apply a discard decision and advance to the next real decision."""
        # Discardable cards are existing Speed/Upgrade tokens, so their IDs are
        # always in the immutable input vocabulary. Avoid reconstructing the
        # complete tensor state on this hot decision submission path.
        vocabulary = {
            card_id: index + 1
            for index, card_id in enumerate(self._template.card_id_vocabulary)
        }
        ids = np.zeros((7,), dtype=np.int64)
        ids[: len(cards)] = [vocabulary[card.id] for card in cards]
        return dict(self._native.apply_discard(player_id, ids, len(cards)))


def _semantic_snapshot_from_payload(
    payload: dict[str, np.ndarray[Any, Any]], template: TensorGameState
) -> dict[str, object]:
    """Decode the expanded native card arena into the canonical D0 receipt."""
    control = payload["control"]
    track_control = payload["track_control"]
    fields = payload["players"]
    lengths = payload["zone_lengths"]
    card_ids = payload["card_ids"]
    card_types = payload["card_types"]
    card_values = payload["card_values"]
    type_names = {1: "speed", 2: "heat", 3: "stress", 4: "upgrade"}

    def cards(zone: int, player: int) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for index in range(int(lengths[zone, player])):
            identifier = int(card_ids[zone, player, index])
            identity = (
                f"stress_penalty_{identifier & 0x7FFF}"
                if identifier >= 0x8000
                else template.card_id_vocabulary[identifier - 1]
            )
            result.append(
                {
                    "type": type_names[int(card_types[zone, player, index])],
                    "value": int(card_values[zone, player, index]),
                    "id": identity,
                }
            )
        return result

    player_count = int(sum(bool(value) for value in fields[:, 0]))
    players: list[dict[str, object]] = []
    for slot in range(player_count):
        spin_count = int(fields[slot, 9])
        players.append(
            {
                "player_id": int(fields[slot, 2]),
                "name": template.player_names[0][slot],
                "gear": int(fields[slot, 3]),
                "position": int(fields[slot, 4]),
                "lap": int(fields[slot, 5]),
                "hand": cards(0, slot),
                "draw_pile": cards(1, slot),
                "discard_pile": cards(2, slot),
                "heat_pool": cards(3, slot),
                "cooldown_pool": cards(4, slot),
                "spun_out": bool(fields[slot, 6]),
                "finished": bool(fields[slot, 7]),
                "finish_order": int(fields[slot, 8]),
                "spin_log": payload["spin_log"][slot, :spin_count].tolist(),
                "cards_played": cards(5, slot),
                "boost_used_this_turn": bool(fields[slot, 10]),
                "speed_from_cards": int(fields[slot, 11]),
                "speed_from_boost": int(fields[slot, 12]),
                "speed_from_adrenaline": int(fields[slot, 13]),
                "slipstream_moved": int(fields[slot, 14]),
                "cluttered": bool(fields[slot, 15]),
                "turn_start_position": int(fields[slot, 16]),
                "turn_start_lap": int(fields[slot, 17]),
            }
        )
    gauss: float | None = None
    if bool(control[8]):
        gauss = float(payload["rng_gauss"][0])
    track_length = int(track_control[0])
    corner_count = int(track_control[1])
    start_count = int(track_control[2])
    return {
        "contract_version": 1,
        "track": {
            "name": template.track_names[0],
            "spaces": [
                {
                    "index": int(payload["track_indices"][index]),
                    "lanes": int(payload["track_lanes"][index]),
                }
                for index in range(track_length)
            ],
            "corners": [
                {
                    "start": int(payload["track_corners"][index, 0]),
                    "end": int(payload["track_corners"][index, 1]),
                    "speed_limit": int(payload["track_corners"][index, 2]),
                }
                for index in range(corner_count)
            ],
            "start_positions": payload["track_starts"][:start_count].tolist(),
            "laps": int(track_control[3]),
        },
        "round_num": int(control[2]),
        "current_phase": PHASE_ORDER[int(control[3])].value,
        "turn_order": payload["turn_order"][: int(control[4])].tolist(),
        "starting_player_count": int(control[5]),
        "stress_counter": int(control[6]),
        "rng_state": [int(control[7]), payload["rng_words"].tolist(), gauss],
        "players": players,
    }


def scalar_canonical_digest(
    state: GameState,
    *,
    game_id: int,
    card_id_vocabulary: tuple[str, ...],
) -> int:
    """Hash scalar logical state with the same compact A5 byte contract."""
    try:
        from heat_native._core import _fnv1a64  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional-package error path
        raise RuntimeError("install the optional native/ package first") from exc

    result = bytearray()
    add = result.extend
    add(struct.pack("<Q?HBBB H", game_id, not state.is_game_over, state.round_num,
                    PHASE_ORDER.index(state.current_phase), len(state.turn_order),
                    state.starting_player_count, state._stress_counter))
    for player_id in state.turn_order:
        add(struct.pack("<h", player_id))
    track = state.track
    add(struct.pack("<BBBB", track.length, len(track.corners),
                    len(track.start_positions), track.laps))
    for space in track.spaces:
        add(struct.pack("<hB", space.index, space.lanes))
    for corner in track.corners:
        add(struct.pack("<hhh", corner.start, corner.end, corner.speed_limit))
    for start in track.start_positions:
        add(struct.pack("<h", start))

    card_index = {
        identity: index + 1
        for index, identity in enumerate(card_id_vocabulary)
    }
    type_code = {
        CardType.SPEED: 1,
        CardType.HEAT: 2,
        CardType.STRESS: 3,
        CardType.UPGRADE: 4,
    }
    for player in state.players:
        add(struct.pack(
            "<h?bhh??h?hhhh?hhH",
            player.player_id,
            not player.finished,
            player.gear,
            player.position,
            player.lap,
            player.spun_out,
            player.finished,
            player.finish_order,
            player.boost_used_this_turn,
            player.speed_from_cards,
            player.speed_from_boost,
            player.speed_from_adrenaline,
            player.slipstream_moved,
            player.cluttered,
            player.turn_start_position,
            player.turn_start_lap,
            len(player.spin_log),
        ))
        for round_num, corner_start in player.spin_log:
            add(struct.pack("<hh", round_num, corner_start))
        zones = (
            player.hand,
            player.deck.draw_pile,
            player.deck.discard_pile,
            player.heat_pool,
            player.cooldown_pool,
            player.cards_played,
        )
        for zone in zones:
            add(struct.pack("<H", len(zone)))
            for card in zone:
                identity = (
                    0x8000 | int(card.id.removeprefix("stress_penalty_"))
                    if card.id.startswith("stress_penalty_")
                    else card_index[card.id]
                )
                add(struct.pack("<HBb", identity, type_code[card.card_type], card.value))

    _rng_version, rng_words, gauss = state.rng.getstate()
    add(struct.pack("<625I", *rng_words))
    add(struct.pack("<?d", gauss is not None, 0.0 if gauss is None else gauss))
    return int(_fnv1a64(bytes(result)))


def legacy_state_to_native(state: GameState, *, game_id: int) -> NativeStateBridge:
    """Pack one representable legacy state into the opaque compact A1 state."""
    try:
        from heat_native._core import _NativeState
    except ImportError as exc:  # pragma: no cover - optional-package error path
        raise RuntimeError("install the optional native/ package first") from exc

    tensor_state = legacy_states_to_tensor([state], game_ids=[game_id])
    return NativeStateBridge(_NativeState(_pack_payload(tensor_state)), tensor_state)


_DECISION_KIND_CODE = {
    DecisionKind.GEAR: 1,
    DecisionKind.CARDS: 2,
    DecisionKind.REACT: 3,
    DecisionKind.SLIPSTREAM: 4,
    DecisionKind.DISCARD: 5,
}


def _pack_payload(state: TensorGameState) -> dict[str, np.ndarray[Any, Any]]:
    """Create strict fixed-shape arrays for the test-only native import boundary."""
    state.validate()
    if state.batch_size != 1:
        raise ValueError("native A1 bridge accepts exactly one state")
    control = np.array(
        [
            int(state.game_ids[0]),
            int(state.game_active[0]),
            int(state.round_num[0]),
            int(state.current_phase[0]),
            int(state.turn_order_lengths[0]),
            int(state.starting_player_count[0]),
            int(state.stress_counter[0]),
            int(state.rng_version[0]),
            int(state.rng_gauss_present[0]),
        ],
        dtype=np.int64,
    )
    track_control = np.array(
        [
            int(state.track_lengths[0]),
            int(state.track_corner_counts[0]),
            int(state.track_start_counts[0]),
            int(state.track_laps[0]),
        ],
        dtype=np.int64,
    )
    player_columns = (
        state.player_present,
        state.player_active,
        state.player_ids,
        state.gear,
        state.position,
        state.lap,
        state.spun_out,
        state.finished,
        state.finish_order,
        state.spin_log_lengths,
        state.boost_used_this_turn,
        state.speed_from_cards,
        state.speed_from_boost,
        state.speed_from_adrenaline,
        state.slipstream_moved,
        state.cluttered,
        state.turn_start_position,
        state.turn_start_lap,
    )
    players = np.stack(
        [_numpy_i64(column[0]) for column in player_columns], axis=1
    )
    zones = [getattr(state, name) for name in _ZONE_NAMES]
    return {
        "control": control,
        "turn_order": _numpy_i64(state.turn_order[0]),
        "track_control": track_control,
        "track_indices": _numpy_i64(state.track_space_indices[0]),
        "track_lanes": _numpy_i64(state.track_lanes[0]),
        "track_corners": _numpy_i64(state.track_corners[0]),
        "track_starts": _numpy_i64(state.track_start_positions[0]),
        "players": np.ascontiguousarray(players, dtype=np.int64),
        "spin_log": _numpy_i64(state.spin_log[0]),
        "card_ids": np.stack([_numpy_i64(zone.card_ids[0]) for zone in zones]),
        "card_types": np.stack(
            [_numpy_i64(zone.card_types[0]) for zone in zones]
        ),
        "card_values": np.stack(
            [_numpy_i64(zone.card_values[0]) for zone in zones]
        ),
        "zone_lengths": np.stack(
            [_numpy_i64(zone.lengths[0]) for zone in zones]
        ),
        "rng_words": np.ascontiguousarray(
            state.rng_words[0].cpu().numpy(), dtype=np.uint32
        ),
        "rng_gauss": np.array([float(state.rng_gauss[0])], dtype=np.float64),
    }


def _numpy_i64(tensor: torch.Tensor) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Copy one CPU tensor into a strict contiguous int64 bridge array."""
    return np.ascontiguousarray(tensor.cpu().numpy(), dtype=np.int64)


def _tensor_i64(array: np.ndarray[Any, Any]) -> torch.Tensor:
    """View a native-export array as the D2 oracle's int64 dtype."""
    return torch.from_numpy(array).to(dtype=torch.int64)


def _decode_reserved_cards(
    card_ids: np.ndarray[Any, Any],
    vocabulary: tuple[str, ...],
) -> tuple[np.ndarray[Any, Any], tuple[str, ...]]:
    """Materialize native reserved stress tokens into D0 debug identities."""
    decoded = np.array(card_ids, dtype=np.int64, copy=True, order="C")
    names = list(vocabulary)
    mapping: dict[int, int] = {}
    for token in sorted(int(value) for value in np.unique(decoded) if value >= 0x8000):
        stress_id = token & 0x7FFF
        mapping[token] = len(names) + 1
        names.append(f"stress_penalty_{stress_id}")
    for token, identifier in mapping.items():
        decoded[decoded == token] = identifier
    return decoded, tuple(names)
