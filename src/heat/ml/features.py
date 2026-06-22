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


def _track_block(player: PlayerState, track) -> list[float]:
    """BLOCK_TRACK floats: MAX_CORNERS ego-centric corner slots + a globals
    sub-block (Sprint B Option A whole-track obs; replaces the old 4-dim
    next-corner lookahead).

    Ego-centric: the corner slots are ordered by forward (wrap-aware) distance
    from the learner's current position, matching
    ``rules.distance_to_next_corner`` (a corner standing under the player counts
    as a full lap away). With ``MAX_CORNERS=8`` and the generator capping tracks
    at 7 corners, every corner of every generated track lands in exactly one
    slot. Absent slots are zero-filled; because every real corner has
    ``dist_ahead > 0``, a slot whose ``dist_ahead`` is 0 is unambiguously
    padding. All fields are pre-clipped to ``[0, 1]``.

    Per-corner slot (4 floats):
      dist_ahead   = fwd_dist / track.length
      speed_limit  = corner.speed_limit / max(speed_limit over corners)
      corner_len   = (end - start + 1) / max(corner_len over corners)
      lanes        = lanes at the corner entry / max(lanes over spaces)

    Globals sub-block (4 floats):
      laps_remaining  = (laps - player.lap) / laps
      dist_to_finish  = (laps*length - (player.lap*length + position)) / (laps*length)
      heat_pool       = player.heat_available / HEAT_POOL_SIZE
      pos_in_lap      = player.position / length
    """
    length = track.length or 1
    pos = player.position
    corners = list(track.corners)
    max_limit = max((c.speed_limit for c in corners), default=1) or 1
    max_clen = max(((c.end - c.start + 1) for c in corners), default=1) or 1
    max_lanes = max((s.lanes for s in track.spaces), default=1) or 1

    # Forward distance to each corner start (wrap-aware), matching
    # rules.distance_to_next_corner's convention (a corner at pos == one lap).
    def fwd_dist(c) -> int:
        d = (c.start - pos) % length
        return length if d == 0 else d

    ordered = sorted(corners, key=fwd_dist)

    slots: list[float] = []
    for i in range(spaces.MAX_CORNERS):
        if i < len(ordered):
            c = ordered[i]
            d = fwd_dist(c)
            if 0 <= c.start < len(track.spaces):
                entry_lanes = track.spaces[c.start].lanes
            else:
                entry_lanes = 1
            slots += [
                _clip01(d / length),
                _clip01(c.speed_limit / max_limit),
                _clip01((c.end - c.start + 1) / max_clen),
                _clip01(entry_lanes / max_lanes),
            ]
        else:
            slots += [0.0, 0.0, 0.0, 0.0]  # padding (dist_ahead == 0 marker)

    laps = track.laps or 1
    laps_remaining = _clip01((laps - player.lap) / laps)
    total_len = length * laps
    abs_pos = player.lap * length + pos
    dist_to_finish = _clip01((total_len - abs_pos) / total_len)
    heat = _clip01(player.heat_available / rules.HEAT_POOL_SIZE)
    pos_in_lap = _clip01(pos / length)
    globals_ = [laps_remaining, dist_to_finish, heat, pos_in_lap]

    return slots + globals_


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


#: Soft cap for normalizing ``round_num`` into [0, 1] (index 9 of the phase
#: block). Module-level so the always-encode path in :func:`encode_observation`
#: and any future reader share the one constant.
_ROUND_CAP: float = 50.0

#: Index of the ``round_num`` feature within the phase-context block. It is the
#: ONLY non-decision-context field in that block, so it is written by
#: :func:`encode_observation` (always), not by :func:`_phase_context` (which is
#: gated on a real ``decision``). Indices 0..8 are the genuine decision context.
_PHASE_ROUND_NUM_INDEX: int = 9


def _phase_context(state: GameState, decision: "Decision | None") -> list[float]:
    """BLOCK_PHASE_CONTEXT floats of *decision-specific* context only: one-hot
    decision kind (indices 0..4), react-context bits (5..7: can_boost,
    has_adrenaline, max_cooldown/3), the slipstream-available flag (8), then
    explicit zero-padding to fill the block. ``decision is None`` (reset / the
    value-path) yields an all-zero block.

    NOTE (obs purity, code-review 2026-06-22 #1): ``round_num`` is pure game
    state, NOT decision context, so it is deliberately NOT written here -- it is
    written unconditionally by :func:`encode_observation` (at
    ``_PHASE_ROUND_NUM_INDEX``) so the encoding is identical whether ``decision``
    is a real decision or ``None``. Were it set here, the value path
    (``decision=None``) and the policy/prior path would encode the SAME state
    differently, breaking the "observation is a pure function of game state"
    contract that Option C's value head depends on. Indices 0..8 remain the only
    decision-gated fields.
    """
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

    # index 9 (round_num) is set by encode_observation, NOT here (see docstring).
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
    values += _track_block(player, track)          # BLOCK_TRACK (36)
    values += _adrenaline_context(state, player)   # 2
    values += _opponent_slots(state, player, track)  # 25
    phase_block_start = len(values)
    phase = _phase_context(state, decision)        # BLOCK_PHASE_CONTEXT
    # round_num is PURE GAME STATE, not decision context, so it is written here
    # (unconditionally) rather than inside the decision-gated _phase_context.
    # This guarantees the observation is a pure function of game state: encoding
    # the same `state` with decision=None (Option C's value path) and with a real
    # decision (the policy/prior path) is identical on every index EXCEPT the
    # genuine decision-context bits 0..8 (code-review 2026-06-22 #1). Soft-capped
    # and clipped to [0, 1].
    #
    # Leak tradeoff (considered, accepted): because the value target is
    # `-rounds_remaining`, exposing the round counter lets the value net partly
    # memorize the counter instead of learning a genuine cost-to-go. We keep
    # round_num IN the observation -- removing it is a separate decision -- but
    # make it state-pure here so train-time and search-time encodings match.
    phase[_PHASE_ROUND_NUM_INDEX] = _clip01(state.round_num / _ROUND_CAP)
    values += phase
    assert phase_block_start + spaces.BLOCK_PHASE_CONTEXT == OBS_DIM, (
        "phase block must be the final block for the round_num index to be "
        "absolute-stable"
    )

    vec = np.asarray(values, dtype=np.float32)
    assert vec.shape == (OBS_DIM,), (
        f"encode_observation produced {vec.shape}, expected ({OBS_DIM},)"
    )
    # Defensive: guarantee the contract bound even if a scaler drifted.
    np.clip(vec, -1.0, 1.0, out=vec)
    return vec
