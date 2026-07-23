"""Canonical scalar-engine snapshots used as Direction D's semantic oracle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

from heat.engine.driver import Decision
from heat.models.cards import Card
from heat.models.game_state import GameEvent, GameState
from heat.ml.action_codec import legal_action_mask
from heat.ml.features import encode_observation


SEMANTIC_CONTRACT_VERSION = 1


def _json_value(value: object) -> object:
    """Convert engine and RNG values into lossless, stable JSON primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Card):
        return {
            "type": value.card_type.value,
            "value": value.value,
            "id": value.id,
        }
    if is_dataclass(value):
        return _json_value(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported semantic snapshot type: {type(value).__name__}")


def _cards(cards: object) -> object:
    """Canonicalize a card or nested legal-card payload."""
    return _json_value(cards)


def canonical_game_state(state: GameState) -> dict[str, object]:
    """Return every mutable rule field, including private card and RNG state."""
    players: list[dict[str, object]] = []
    for player in state.players:
        players.append(
            {
                "player_id": player.player_id,
                "name": player.name,
                "gear": player.gear,
                "position": player.position,
                "lap": player.lap,
                "hand": _cards(player.hand),
                "draw_pile": _cards(player.deck.draw_pile),
                "discard_pile": _cards(player.deck.discard_pile),
                "heat_pool": _cards(player.heat_pool),
                "cooldown_pool": _cards(player.cooldown_pool),
                "spun_out": player.spun_out,
                "finished": player.finished,
                "finish_order": player.finish_order,
                "spin_log": _json_value(player.spin_log),
                "cards_played": _cards(player.cards_played),
                "boost_used_this_turn": player.boost_used_this_turn,
                "speed_from_cards": player.speed_from_cards,
                "speed_from_boost": player.speed_from_boost,
                "speed_from_adrenaline": player.speed_from_adrenaline,
                "slipstream_moved": player.slipstream_moved,
                "cluttered": player.cluttered,
                "turn_start_position": player.turn_start_position,
                "turn_start_lap": player.turn_start_lap,
            }
        )
    return {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "track": {
            "name": state.track.name,
            "spaces": [
                {"index": space.index, "lanes": space.lanes}
                for space in state.track.spaces
            ],
            "corners": [
                {
                    "start": corner.start,
                    "end": corner.end,
                    "speed_limit": corner.speed_limit,
                }
                for corner in state.track.corners
            ],
            "start_positions": list(state.track.start_positions),
            "laps": state.track.laps,
        },
        "round_num": state.round_num,
        "current_phase": state.current_phase.value,
        "turn_order": list(state.turn_order),
        "starting_player_count": state.starting_player_count,
        "stress_counter": state._stress_counter,
        "rng_state": _json_value(state.rng.getstate()),
        "players": players,
    }


def canonical_decision(decision: Decision) -> dict[str, object]:
    """Return a stable representation of a yielded decision and legal payload."""
    return {
        "kind": decision.kind.value,
        "player_id": decision.player_id,
        "legal": _json_value(decision.legal),
    }


def canonical_event(event: GameEvent) -> dict[str, object]:
    """Return a stable event receipt for automatic phase and terminal coverage."""
    return {
        "round_num": event.round_num,
        "phase": event.phase.value,
        "player_id": event.player_id,
        "event_type": event.event_type,
        "data": _json_value(event.data),
    }


def canonical_json(value: object) -> str:
    """Serialize semantic data with byte-stable ordering and no NaN values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def semantic_sha256(value: object) -> str:
    """Hash one canonical semantic object."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_decision_row(
    state: GameState,
    decision: Decision,
    *,
    game_id: int,
    decision_index: int,
) -> dict[str, Any]:
    """Capture state, decision, acting-seat observation and legal mask together."""
    observation = encode_observation(state, decision.player_id, decision)
    mask = legal_action_mask(decision, state)
    row: dict[str, Any] = {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "game_id": game_id,
        "decision_index": decision_index,
        "decision": canonical_decision(decision),
        "observation": observation.astype(np.float32, copy=False).tolist(),
        "legal_mask": mask.astype(np.bool_, copy=False).tolist(),
        "state": canonical_game_state(state),
    }
    row["sha256"] = semantic_sha256(row)
    return row
