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

from itertools import chain

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
    """8 floats: Speed 1-4 counts, Heat, Stress, Upgrade-0, Upgrade-5, /7.

    Sprint C9: fixed-index integer counters instead of the string-keyed dict +
    ``f"S{value}"`` f-strings. The slot order is the frozen output order
    ``[S1, S2, S3, S4, H, ST, U0, U5]`` (indices 0..7). Bit-identical output.
    Speed cards with a value outside 1..4 fall outside the slot range and are
    dropped -- exactly as the original ``if key in counts`` guard did.
    """
    counts = [0, 0, 0, 0, 0, 0, 0, 0]  # S1 S2 S3 S4 H ST U0 U5
    for card in player.hand:
        ct = card.card_type
        if ct == CardType.SPEED:
            v = card.value
            if 1 <= v <= 4:
                counts[v - 1] += 1
        elif ct == CardType.HEAT:
            counts[4] += 1
        elif ct == CardType.STRESS:
            counts[5] += 1
        elif ct == CardType.UPGRADE:
            counts[6 if card.value == 0 else 7] += 1
    return [_clip01(c / 7.0) for c in counts]


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
    plus draw-pile-size/total and discard-pile-size/total. Own deck only.

    Sprint C9: a SINGLE pass over draw+discard accumulating the four card-type
    counts, replacing the prior ``list(draw)+list(discard)`` allocation and four
    independent genexpr passes. Bit-identical output (the same six floats); the
    profile (C9 motivation) showed this block at ~47% of encode, dominated by the
    redundant 4x iteration. Uses ``itertools.chain`` so no intermediate list is
    built, and the per-type counts are read with one ``c.card_type`` lookup each.
    """
    deck = player.deck
    n_speed = n_heat = n_stress = n_upgrade = 0
    total = 0
    for c in chain(deck.draw_pile, deck.discard_pile):
        total += 1
        ct = c.card_type
        if ct == CardType.SPEED:
            n_speed += 1
        elif ct == CardType.HEAT:
            n_heat += 1
        elif ct == CardType.STRESS:
            n_stress += 1
        elif ct == CardType.UPGRADE:
            n_upgrade += 1
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        _clip01(n_speed / total),
        _clip01(n_heat / total),
        _clip01(n_stress / total),
        _clip01(n_upgrade / total),
        _clip01(deck.draw_pile_size / total),
        _clip01(deck.discard_pile_size / total),
    ]


# ---------------------------------------------------------------------------
# Sprint C7: per-track precompute of the STATE-INVARIANT corner arrays.
# ---------------------------------------------------------------------------
#
# The C6 profile (C6-findings §Point-1, item 2) measured ``_track_block`` rebuilt
# on EVERY leaf eval (~968x/game) though the track is constant within a game.
# Most of the per-corner work is a pure function of the TRACK, not the player:
# the normalization maxes (``max_limit``/``max_clen``/``max_lanes``) and each
# corner's intrinsic ``(speed_limit, corner_len, entry_lanes)`` never change as
# the car moves. Only the forward-distance ordering (``fwd_dist``), the per-slot
# ``dist_ahead``, and the globals depend on the player's position/lap.
#
# We DO NOT memoize the whole ``_track_block`` output (that depends on the player
# position, so it would risk a stale obs). We precompute ONLY the track-invariant
# corner table once per track object, keyed by ``id(track)`` (a track is a mutable
# dataclass, so it is unhashable; the object identity is stable within a game --
# the generator builds one track per game). The cached values are byte-identical
# to what the un-cached path computes, so the assembled observation is unchanged
# (pinned by ``tests/test_c7_throughput.py``). Bounded to the most-recent few
# tracks so a long parallel run cannot leak memory.
_TRACK_PRECOMPUTE_CACHE_CAP: int = 8
#: id(track) -> (fingerprint, (max_limit, max_clen, max_lanes, corner_table)) where
#: corner_table is a list of (start, norm_limit, norm_clen, norm_lanes) per corner
#: in the track's native corner order. ``norm_*`` are the already-divided [0,1]
#: values. The ``fingerprint`` guards against ``id()`` recycling (see below).
_track_precompute: "dict[int, tuple]" = {}


def _track_fingerprint(track) -> tuple:
    """A cheap structural identity tag for ``track`` (recycled-id guard).

    ``id(track)`` is only unique while the object is ALIVE -- once a track is
    garbage-collected CPython readily reuses its address for a DIFFERENT track, so
    a stale cache entry keyed by the recycled id would silently serve the wrong
    corner table (a poisoned observation). This fingerprint -- built from the
    track's invariant structure (corner count + each corner's geometry, space
    count, laps) -- is compared on every cache hit; a mismatch means the id was
    recycled and we recompute. It is O(corners) (tiny: tracks cap at 7 corners),
    so the guard is far cheaper than the corner-table build it protects.
    """
    return (
        len(track.spaces),
        track.laps,
        tuple((c.start, c.end, c.speed_limit) for c in track.corners),
    )


def _track_precompute_for(track) -> tuple:
    """Return (and cache) the state-invariant corner table for ``track``.

    Pure function of the TRACK: the normalization maxes and each corner's
    normalized intrinsic features (speed-limit, length, entry lanes). Keyed by
    ``id(track)`` (the object is stable within a game; a track is unhashable),
    with a structural :func:`_track_fingerprint` stored alongside so a RECYCLED
    ``id()`` (a freed track's address reused for a different track) is detected and
    recomputed rather than served stale. The returned tuple is consumed by
    :func:`_track_block`, which still computes the player-dependent ordering /
    distances / globals per call -- so the assembled observation stays
    bit-identical to the un-cached path.
    """
    key = id(track)
    fingerprint = _track_fingerprint(track)
    cached = _track_precompute.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    corners = list(track.corners)
    spaces_list = track.spaces
    n_spaces = len(spaces_list)
    max_limit = max((c.speed_limit for c in corners), default=1) or 1
    max_clen = max(((c.end - c.start + 1) for c in corners), default=1) or 1
    max_lanes = max((s.lanes for s in spaces_list), default=1) or 1

    corner_table = []
    for c in corners:
        if 0 <= c.start < n_spaces:
            entry_lanes = spaces_list[c.start].lanes
        else:
            entry_lanes = 1
        corner_table.append((
            c.start,
            _clip01(c.speed_limit / max_limit),
            _clip01((c.end - c.start + 1) / max_clen),
            _clip01(entry_lanes / max_lanes),
        ))

    result = (max_limit, max_clen, max_lanes, corner_table)
    if key not in _track_precompute and len(_track_precompute) >= _TRACK_PRECOMPUTE_CACHE_CAP:
        # Bounded: drop the oldest entry (insertion-ordered dict) so a long
        # multi-track run does not accumulate stale per-track tables. (A recycled-id
        # overwrite reuses the existing slot, so it is not an eviction trigger.)
        _track_precompute.pop(next(iter(_track_precompute)))
    # Store the fingerprint alongside the result so a later recycled-id hit on this
    # slot is detected and recomputed (never served stale).
    _track_precompute[key] = (fingerprint, result)
    return result


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

    # Sprint C7: the per-corner normalization maxes and intrinsic normalized
    # features are a pure function of the TRACK (constant within a game), so they
    # are precomputed once and reused. Only the player-dependent forward-distance
    # ordering / dist_ahead and the globals are recomputed here, so the assembled
    # block is byte-identical to the un-cached path (the corner_table is in native
    # corner order, and the stable sort by fwd_dist reproduces the original
    # ``sorted(corners, key=fwd_dist)`` ordering exactly).
    _max_limit, _max_clen, _max_lanes, corner_table = _track_precompute_for(track)

    # Forward distance to a corner start (wrap-aware), matching
    # rules.distance_to_next_corner's convention (a corner at pos == one lap).
    def fwd_dist(start: int) -> int:
        d = (start - pos) % length
        return length if d == 0 else d

    # Stable sort by forward distance -- identical to the original
    # ``sorted(corners, key=fwd_dist)`` because corner_table is in native order.
    ordered = sorted(corner_table, key=lambda entry: fwd_dist(entry[0]))

    slots: list[float] = []
    for i in range(spaces.MAX_CORNERS):
        if i < len(ordered):
            start, norm_limit, norm_clen, norm_lanes = ordered[i]
            d = fwd_dist(start)
            slots += [
                _clip01(d / length),
                norm_limit,
                norm_clen,
                norm_lanes,
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
    """2 floats: adrenaline-eligible flag, own rank/(n-1).

    Sprint C9 micro-opt: pass ``state.players`` directly instead of
    ``list(state.active_players)``. ``adrenaline_eligible`` re-filters out
    finished players internally (``active = [p for p in all_players if not
    p.finished]``), so the eligibility result is identical -- but we avoid
    building (and then copying, via ``list(...)``) the active sub-list every leaf.
    """
    eligible = rules.adrenaline_eligible(
        player, state.players, state.starting_player_count
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

    # Sprint C9 micro-opt: hoist the per-call invariants out of the slot loop
    # (own position, the half-length used for the signed wrap, and the player's
    # lap). Bit-identical -- the arithmetic per opponent is unchanged, just no
    # longer re-read/recomputed each iteration.
    own_pos = player.position
    own_lap = player.lap
    half = length / 2.0

    slots: list[float] = []
    for i in range(spaces.MAX_PLAYERS - 1):
        if i < len(opponents):
            opp = opponents[i]
            # Relative position: signed, wrap-aware, normalized to [-1, 1].
            raw = (opp.position - own_pos) % length
            # Map [0, length) to (-1, 1]: ahead positive, behind negative.
            if raw > half:
                raw -= length
            rel_pos = _clip_signed(raw / half) if half > 0 else 0.0
            gear_norm = _clip01(opp.gear / rules.MAX_GEAR)
            lap_delta = _clip_signed((opp.lap - own_lap) / laps)
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


def _state_prefix(state: GameState, player: PlayerState, track) -> list[float]:
    """The seat-state prefix: every observation block BEFORE the phase tail.

    This is the portion of the observation that depends ONLY on
    ``(state, player, track)`` and NOT on the pending ``decision``. It is
    identical whether the obs is the prior obs (a real decision) or the value obs
    (``decision=None``) for the SAME seat -- which is exactly what the Sprint C9
    Tier-2 shared-prefix optimization exploits to encode it once per leaf and
    reuse it for both the prior and value vectors.

    The block order/counts are the frozen codec layout (``spaces`` §3.1). The
    returned list has length ``OBS_DIM - BLOCK_PHASE_CONTEXT``.
    """
    values: list[float] = []
    values += _hand_histogram(player)              # 8
    values += _own_gear(player)                    # 4
    values += _own_kinematics(player, track)       # 4
    values += _deck_composition(player)            # 6
    values += _track_block(player, track)          # BLOCK_TRACK (36)
    values += _adrenaline_context(state, player)   # 2
    values += _opponent_slots(state, player, track)  # 25
    return values


def _phase_tail(state: GameState, decision: "Decision | None") -> list[float]:
    """The final phase block (length ``BLOCK_PHASE_CONTEXT``) with ``round_num``.

    Combines the decision-gated context (:func:`_phase_context`) with the
    unconditional, state-pure ``round_num`` field written at
    ``_PHASE_ROUND_NUM_INDEX``. Factored out of :func:`encode_observation` so the
    shared-prefix pair path (Sprint C9) writes the phase tail through the EXACT
    same code, guaranteeing byte-identity.

    round_num is PURE GAME STATE, not decision context, so it is written here
    (unconditionally) rather than inside the decision-gated ``_phase_context``.
    This guarantees the observation is a pure function of game state: encoding the
    same ``state`` with ``decision=None`` (Option C's value path) and with a real
    decision (the policy/prior path) is identical on every index EXCEPT the
    genuine decision-context bits 0..8 (code-review 2026-06-22 #1). Soft-capped
    and clipped to [0, 1].

    Leak tradeoff (considered, accepted): because the value target is
    ``-rounds_remaining``, exposing the round counter lets the value net partly
    memorize the counter instead of learning a genuine cost-to-go. We keep
    round_num IN the observation -- removing it is a separate decision -- but make
    it state-pure here so train-time and search-time encodings match.
    """
    phase = _phase_context(state, decision)        # BLOCK_PHASE_CONTEXT
    phase[_PHASE_ROUND_NUM_INDEX] = _clip01(state.round_num / _ROUND_CAP)
    return phase


def _assemble(prefix: list[float], phase: list[float]) -> np.ndarray:
    """Concatenate prefix + phase into the final clipped ``float32`` vector.

    The single assembly path shared by :func:`encode_observation` and the C9
    shared-prefix pair path, so both produce byte-identical vectors. ``prefix`` is
    NOT mutated (``prefix + phase`` builds a new list).
    """
    assert len(prefix) + spaces.BLOCK_PHASE_CONTEXT == OBS_DIM, (
        "phase block must be the final block for the round_num index to be "
        "absolute-stable"
    )
    vec = np.asarray(prefix + phase, dtype=np.float32)
    assert vec.shape == (OBS_DIM,), (
        f"encode_observation produced {vec.shape}, expected ({OBS_DIM},)"
    )
    # Defensive: guarantee the contract bound even if a scaler drifted.
    np.clip(vec, -1.0, 1.0, out=vec)
    return vec


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
    prefix = _state_prefix(state, player, track)
    phase = _phase_tail(state, decision)
    return _assemble(prefix, phase)


def encode_observation_pair(
    state: GameState,
    player_id: int,
    decision: "Decision | None",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the PRIOR obs and the VALUE obs for ONE seat, sharing the prefix.

    Sprint C9 Tier-2: a clean MCTS decision leaf encodes the SAME state twice for
    the same seat -- once with the pending ``decision`` (the prior obs read by
    ``_edges_for``) and once with ``decision=None`` (the value obs read by
    ``_evaluate_leaf``). When the moving seat equals the evaluated seat
    (``mover == to_move_pid`` -- all solo leaves and learner-seat leaves in 2p),
    every block before the phase tail is byte-identical, so we compute the
    seat-state prefix ONCE and assemble both vectors, recomputing only the small
    phase block.

    Returns ``(prior_obs, value_obs)`` where:
      * ``prior_obs == encode_observation(state, player_id, decision)``
      * ``value_obs == encode_observation(state, player_id, None)``
    byte-for-byte (``np.array_equal``). The caller MUST guarantee the seat
    equality precondition; this helper does not check it (it is the prior+value
    of the SAME ``player_id``).
    """
    player = state.get_player(player_id)
    track = state.track
    prefix = _state_prefix(state, player, track)
    prior_obs = _assemble(prefix, _phase_tail(state, decision))
    value_obs = _assemble(prefix, _phase_tail(state, None))
    return prior_obs, value_obs
