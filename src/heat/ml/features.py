"""Deterministic observation encoder for the HEAT RL layer (Sprint 5a).

:func:`encode_observation` maps a ``GameState`` + the learning seat's id (+ the
current ``Decision``) to a fixed-length, normalized ``float32`` vector of shape
``(OBS_DIM,)``. It is a *pure* function: it never mutates ``state``, reads only
public APIs, and every output value lies in ``[-1, 1]``.

The feature blocks are packed in the exact order/count frozen in
:mod:`heat.ml.spaces` (§3.1). Partial observability is honest: opponent *hands*
and deck order are NOT exposed -- only public opponent state (position, gear,
lap, finished). Deck-composition features use the learning seat's OWN deck.
"""

from __future__ import annotations

import numpy as np

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.models.cards import CardType
from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.ml import spaces
from heat.ml.spaces import OBS_DIM


def _clip01(x: float) -> float:
    """Clamp into [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _clip_signed(x: float) -> float:
    """Clamp into [-1, 1]."""
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def _hand_histogram(player: PlayerState) -> list[float]:
    """8 floats: Speed 1-4 counts, Heat, Stress, Upgrade-0, Upgrade-5, /7."""
    counts = {
        "S1": 0, "S2": 0, "S3": 0, "S4": 0,
        "H": 0, "ST": 0, "U0": 0, "U5": 0,
    }
    for card in player.hand:
        if card.card_type == CardType.SPEED:
            key = f"S{card.value}"
            if key in counts:
                counts[key] += 1
        elif card.card_type == CardType.HEAT:
            counts["H"] += 1
        elif card.card_type == CardType.STRESS:
            counts["ST"] += 1
        elif card.card_type == CardType.UPGRADE:
            counts["U0" if card.value == 0 else "U5"] += 1
    order = ["S1", "S2", "S3", "S4", "H", "ST", "U0", "U5"]
    return [_clip01(counts[k] / 7.0) for k in order]


def _own_gear(player: PlayerState) -> list[float]:
    """4 floats: one-hot gear 1-4."""
    vec = [0.0, 0.0, 0.0, 0.0]
    if rules.MIN_GEAR <= player.gear <= rules.MAX_GEAR:
        vec[player.gear - 1] = 1.0
    return vec


def _own_kinematics(player: PlayerState, track) -> list[float]:
    """4 floats: position/length, lap/laps, heat/6, finished."""
    length = track.length or 1
    laps = track.laps or 1
    return [
        _clip01(player.position / length),
        _clip01(player.lap / laps),
        _clip01(player.heat_available / rules.HEAT_POOL_SIZE),
        1.0 if player.finished else 0.0,
    ]


def _deck_composition(player: PlayerState) -> list[float]:
    """6 floats: fraction of {Speed, Heat, Stress, Upgrade} across draw+discard,
    plus draw-pile-size/total and discard-pile-size/total. Own deck only."""
    deck = player.deck
    all_cards = list(deck.draw_pile) + list(deck.discard_pile)
    total = len(all_cards)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n_speed = sum(1 for c in all_cards if c.card_type == CardType.SPEED)
    n_heat = sum(1 for c in all_cards if c.card_type == CardType.HEAT)
    n_stress = sum(1 for c in all_cards if c.card_type == CardType.STRESS)
    n_upgrade = sum(1 for c in all_cards if c.card_type == CardType.UPGRADE)
    return [
        _clip01(n_speed / total),
        _clip01(n_heat / total),
        _clip01(n_stress / total),
        _clip01(n_upgrade / total),
        _clip01(deck.draw_pile_size / total),
        _clip01(deck.discard_pile_size / total),
    ]


def _track_lookahead(player: PlayerState, track) -> list[float]:
    """4 floats: dist-to-next-corner/length, next speed_limit/maxlimit,
    current-space lanes (normalized), in-corner flag."""
    length = track.length or 1
    corner, dist = rules.distance_to_next_corner(track, player.position)
    if corner is None:
        dist_norm = 1.0
        limit_norm = 1.0
    else:
        dist_norm = _clip01(dist / length)
        max_limit = max((c.speed_limit for c in track.corners), default=1) or 1
        limit_norm = _clip01(corner.speed_limit / max_limit)
    # Current-space lanes, normalized by the track's max lanes.
    if 0 <= player.position < len(track.spaces):
        lanes = track.spaces[player.position].lanes
    else:
        lanes = 1
    max_lanes = max((s.lanes for s in track.spaces), default=1) or 1
    lanes_norm = _clip01(lanes / max_lanes)
    in_corner = 1.0 if track.get_corner_at(player.position) is not None else 0.0
    return [dist_norm, limit_norm, lanes_norm, in_corner]


def _own_rank(state: GameState, player: PlayerState) -> float:
    """Learner rank in [0, 1]: 0 == leader, 1 == last, by (lap, position)."""
    n = state.num_players
    if n <= 1:
        return 0.0
    ahead = 0
    for other in state.players:
        if other.player_id == player.player_id:
            continue
        # "Ahead" = further along (more laps, or same lap further position).
        if (other.lap, other.position) > (player.lap, player.position):
            ahead += 1
    return _clip01(ahead / (n - 1))


def _adrenaline_context(state: GameState, player: PlayerState) -> list[float]:
    """2 floats: adrenaline-eligible flag, own rank/(n-1)."""
    eligible = rules.adrenaline_eligible(
        player, list(state.active_players), state.starting_player_count
    )
    return [1.0 if eligible else 0.0, _own_rank(state, player)]


def _opponent_slots(state: GameState, player: PlayerState, track) -> list[float]:
    """5 * (MAX_PLAYERS - 1) floats. Per opponent slot:
    [presence, relative_position (signed), gear/4, lap_delta (signed), finished].
    Absent slots are zero-filled (presence 0). Only PUBLIC opponent state."""
    length = track.length or 1
    laps = track.laps or 1
    opponents = [p for p in state.players if p.player_id != player.player_id]
    # Stable order by player_id so the encoding is deterministic.
    opponents.sort(key=lambda p: p.player_id)

    slots: list[float] = []
    for i in range(spaces.MAX_PLAYERS - 1):
        if i < len(opponents):
            opp = opponents[i]
            # Relative position: signed, wrap-aware, normalized to [-1, 1].
            raw = (opp.position - player.position) % length
            # Map [0, length) to (-1, 1]: ahead positive, behind negative.
            half = length / 2.0
            if raw > half:
                raw -= length
            rel_pos = _clip_signed(raw / half) if half > 0 else 0.0
            gear_norm = _clip01(opp.gear / rules.MAX_GEAR)
            lap_delta = _clip_signed((opp.lap - player.lap) / laps)
            finished = 1.0 if opp.finished else 0.0
            slots.extend([1.0, rel_pos, gear_norm, lap_delta, finished])
        else:
            slots.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return slots


def _phase_context(state: GameState, decision: "Decision | None") -> list[float]:
    """BLOCK_PHASE_CONTEXT floats: one-hot decision kind (5), react-context bits
    (can_boost, has_adrenaline, max_cooldown/3), slipstream-available flag,
    round_num/cap, then explicit zero-padding to fill the block. ``decision is
    None`` (reset) yields an all-zero block."""
    block = [0.0] * spaces.BLOCK_PHASE_CONTEXT
    if decision is None:
        return block

    kinds = [
        DecisionKind.GEAR,
        DecisionKind.CARDS,
        DecisionKind.REACT,
        DecisionKind.SLIPSTREAM,
        DecisionKind.DISCARD,
    ]
    # one-hot decision kind (indices 0..4)
    for i, k in enumerate(kinds):
        if decision.kind == k:
            block[i] = 1.0

    # React-context bits (indices 5..7) when this is a REACT decision.
    if decision.kind == DecisionKind.REACT:
        opts: rules.ReactOptions = decision.legal
        block[5] = 1.0 if opts.can_boost else 0.0
        block[6] = 1.0 if opts.has_adrenaline else 0.0
        block[7] = _clip01(opts.max_cooldown / 3.0)

    # Slipstream-available flag (index 8).
    if decision.kind == DecisionKind.SLIPSTREAM:
        block[8] = 1.0

    # round_num / cap (index 9). Use a soft cap; clipped to [0, 1].
    round_cap = 50.0
    block[9] = _clip01(state.round_num / round_cap)

    # indices 10..end remain explicit zero-padding.
    return block


def encode_observation(
    state: GameState,
    player_id: int,
    decision: "Decision | None",
) -> np.ndarray:
    """Return a ``float32`` vector of shape ``(OBS_DIM,)`` for the learning seat.

    Pure: no mutation of ``state``. All values are in ``[-1, 1]``. ``decision``
    supplies the phase/decision-context block; ``None`` at episode reset
    boundaries yields a zero-filled context block.
    """
    player = state.get_player(player_id)
    track = state.track

    values: list[float] = []
    values += _hand_histogram(player)              # 8
    values += _own_gear(player)                    # 4
    values += _own_kinematics(player, track)       # 4
    values += _deck_composition(player)            # 6
    values += _track_lookahead(player, track)      # 4
    values += _adrenaline_context(state, player)   # 2
    values += _opponent_slots(state, player, track)  # 25
    values += _phase_context(state, decision)      # BLOCK_PHASE_CONTEXT

    vec = np.asarray(values, dtype=np.float32)
    assert vec.shape == (OBS_DIM,), (
        f"encode_observation produced {vec.shape}, expected ({OBS_DIM},)"
    )
    # Defensive: guarantee the contract bound even if a scaler drifted.
    np.clip(vec, -1.0, 1.0, out=vec)
    return vec
