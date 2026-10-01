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
from typing import TypeAlias, cast

import numpy as np

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.models.cards import CardType
from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.models.track import Track
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


def _resolve_codec(codec_version: int | None) -> int:
    """Return a playable codec, defaulting to the live version."""
    version = spaces.CODEC_VERSION if codec_version is None else codec_version
    if version not in spaces.LEGACY_PLAYABLE_CODECS:
        raise ValueError(
            f"unsupported codec_version {version}; "
            f"expected one of {spaces.LEGACY_PLAYABLE_CODECS}"
        )
    return version


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


def _own_kinematics(player: PlayerState, track: Track) -> list[float]:
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
# Most of the per-corner work is a pure function of the TRACK, not the player.
# Only the forward-distance ordering and the globals depend on the car.
#
# The cache stores RAW geometry (start, speed limit, corner length, entry lanes),
# not already-normalized floats. v3 divides by the track's own maxima; v4 divides
# by the fixed scales in :mod:`heat.ml.spaces`. Normalizing at read time means a
# v4 cache entry cannot be served to a v3 encode. Keyed by ``id(track)`` (a track
# is unhashable; the object is stable within a game) plus a fingerprint so a
# recycled id is not served stale. Bounded so a long run cannot leak memory.
_TRACK_PRECOMPUTE_CACHE_CAP: int = 8
#: (start, end, speed_limit, entry_lanes) per corner, plus length and laps.
TrackFingerprint: TypeAlias = tuple[int, int, tuple[tuple[int, int, int, int], ...]]
#: (start, speed_limit, corner_len, entry_lanes) in native corner order.
RawCorner: TypeAlias = tuple[int, int, int, int]
TrackPrecompute: TypeAlias = list[RawCorner]
_track_precompute: dict[int, tuple[TrackFingerprint, TrackPrecompute]] = {}


def _track_fingerprint(track: Track) -> TrackFingerprint:
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
    spaces_list = track.spaces
    n_spaces = len(spaces_list)
    corners: list[tuple[int, int, int, int]] = []
    for corner in track.corners:
        if 0 <= corner.start < n_spaces:
            entry_lanes = spaces_list[corner.start].lanes
        else:
            entry_lanes = 1
        corners.append((corner.start, corner.end, corner.speed_limit, entry_lanes))
    return (n_spaces, track.laps, tuple(corners))


def _track_precompute_for(track: Track) -> TrackPrecompute:
    """Return (and cache) the raw corner geometry for ``track``.

    Keyed by ``id(track)`` with a structural :func:`_track_fingerprint` so a
    recycled address is recomputed rather than served stale. Normalization is
    NOT cached: :func:`_track_block` scales the raw rows for the requested codec.
    """
    key = id(track)
    fingerprint = _track_fingerprint(track)
    cached = _track_precompute.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    spaces_list = track.spaces
    n_spaces = len(spaces_list)
    corner_table: list[RawCorner] = []
    for corner in track.corners:
        if 0 <= corner.start < n_spaces:
            entry_lanes = spaces_list[corner.start].lanes
        else:
            entry_lanes = 1
        corner_table.append((
            corner.start,
            corner.speed_limit,
            corner.end - corner.start + 1,
            entry_lanes,
        ))

    if key not in _track_precompute and len(_track_precompute) >= _TRACK_PRECOMPUTE_CACHE_CAP:
        # Bounded: drop the oldest entry (insertion-ordered dict). A recycled-id
        # overwrite reuses the existing slot, so it is not an eviction trigger.
        _track_precompute.pop(next(iter(_track_precompute)))
    _track_precompute[key] = (fingerprint, corner_table)
    return corner_table


def _scaled_corner(
    speed_limit: int,
    corner_len: int,
    entry_lanes: int,
    codec_version: int,
    max_limit: int,
    max_clen: int,
    max_lanes: int,
) -> tuple[float, float, float]:
    """Scale one corner's limit, length, and entry lanes for ``codec_version``."""
    if codec_version == 3:
        return (
            _clip01(speed_limit / max_limit),
            _clip01(corner_len / max_clen),
            _clip01(entry_lanes / max_lanes),
        )
    return (
        _clip01(speed_limit / spaces.SPEED_LIMIT_SCALE),
        _clip01(corner_len / spaces.CORNER_LENGTH_SCALE),
        _clip01(entry_lanes / spaces.LANE_SCALE),
    )


def _track_block(
    player: PlayerState,
    track: Track,
    codec_version: int | None = None,
) -> list[float]:
    """BLOCK_TRACK floats: ego-centric corner slots plus the four globals.

    ``dist_ahead`` is forward distance / track length in every codec. v3 divides
    speed limit, corner length, and lanes by that track's own maxima, and uses
    ``(laps - lap) / laps`` plus ``lap * length + position`` for the finish.
    v4 uses the fixed scales and one-based laps still to drive. A finished car's
    v4 laps-remaining and distance-to-finish are 0.
    """
    version = _resolve_codec(codec_version)
    length = track.length or 1
    pos = player.position
    corner_table = _track_precompute_for(track)
    if version == 3:
        max_limit = max((row[1] for row in corner_table), default=1) or 1
        max_clen = max((row[2] for row in corner_table), default=1) or 1
        max_lanes = max((space.lanes for space in track.spaces), default=1) or 1
    else:
        max_limit = max_clen = max_lanes = 1

    # Forward distance to a corner start (wrap-aware). A corner at ``pos`` is one
    # lap away, matching ``rules.distance_to_next_corner``.
    def fwd_dist(start: int) -> int:
        distance = (start - pos) % length
        return length if distance == 0 else distance

    ordered = sorted(corner_table, key=lambda entry: fwd_dist(entry[0]))

    slots: list[float] = []
    for i in range(spaces.MAX_CORNERS):
        if i < len(ordered):
            start, speed_limit, corner_len, entry_lanes = ordered[i]
            norm_limit, norm_clen, norm_lanes = _scaled_corner(
                speed_limit,
                corner_len,
                entry_lanes,
                version,
                max_limit,
                max_clen,
                max_lanes,
            )
            slots += [
                _clip01(fwd_dist(start) / length),
                norm_limit,
                norm_clen,
                norm_lanes,
            ]
        else:
            slots += [0.0, 0.0, 0.0, 0.0]  # padding (dist_ahead == 0 marker)

    laps = track.laps or 1
    total_len = length * laps
    if version == 3:
        laps_remaining = _clip01((laps - player.lap) / laps)
        abs_pos = player.lap * length + pos
        dist_to_finish = _clip01((total_len - abs_pos) / total_len)
    elif player.finished or player.lap > laps:
        laps_remaining = 0.0
        dist_to_finish = 0.0
    else:
        current_lap = max(player.lap, 1)
        laps_remaining = _clip01((laps - current_lap + 1) / laps)
        completed = (current_lap - 1) * length + pos
        dist_to_finish = _clip01((total_len - completed) / total_len)
    heat = _clip01(player.heat_available / rules.HEAT_POOL_SIZE)
    pos_in_lap = _clip01(pos / length)
    return slots + [laps_remaining, dist_to_finish, heat, pos_in_lap]


def _lap_spaces_moved(player: PlayerState, length: int) -> int:
    """Lap-aware step count from the turn start, matching the corner check."""
    return (
        (player.lap - player.turn_start_lap) * length
        + (player.position - player.turn_start_position)
    )


def _public_heat_cost(
    player: PlayerState,
    track: Track,
    spaces_moved: int,
    end_pos: int,
    speed: int,
) -> int:
    """Sum corner heat on a geometric path. Does not mutate ``player``."""
    if spaces_moved <= 0:
        return 0
    crossed = rules.corners_crossed(
        player.turn_start_position,
        end_pos,
        track,
        spaces_moved=spaces_moved,
    )
    return sum(rules.corner_heat_cost(speed, corner) for corner in crossed)


def _write_v4_race_context(
    phase: list[float], state: GameState, player: PlayerState
) -> None:
    """Fill phase indices 10-16 from public player/track state. 17-18 stay 0."""
    track = state.track
    length = track.length or 1
    laps = track.laps or 1
    speed = rules.corner_speed_for_check(player)
    moved = _lap_spaces_moved(player, length)
    extended = moved + 2
    phase[10] = _clip01(speed / spaces.CORNER_SPEED_SCALE)
    phase[11] = _clip01(player.turn_start_position / length)
    phase[12] = _clip_signed((player.lap - player.turn_start_lap) / laps)
    phase[13] = _clip01(
        _public_heat_cost(player, track, moved, player.position, speed)
        / spaces.HEAT_COST_SCALE
    )
    phase[14] = _clip01(
        _public_heat_cost(player, track, moved, player.position, speed + 1)
        / spaces.HEAT_COST_SCALE
    )
    phase[15] = _clip01(
        _public_heat_cost(
            player,
            track,
            extended,
            (player.turn_start_position + extended) % length,
            speed,
        )
        / spaces.HEAT_COST_SCALE
    )
    phase[16] = _clip01(track.length / spaces.TRACK_LENGTH_SCALE)


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


def _opponent_slots(
    state: GameState, player: PlayerState, track: Track
) -> list[float]:
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
        opts = cast(rules.ReactOptions, decision.legal)
        block[5] = 1.0 if opts.can_boost else 0.0
        block[6] = 1.0 if opts.has_adrenaline else 0.0
        block[7] = _clip01(opts.max_cooldown / 3.0)

    # Slipstream-available flag (index 8).
    if decision.kind == DecisionKind.SLIPSTREAM:
        block[8] = 1.0

    # index 9 (round_num) and, for codec v4, indices 10-16 are written by
    # _phase_tail, NOT here. Indices 17-18 stay zero.
    return block


def _state_prefix(
    state: GameState,
    player: PlayerState,
    track: Track,
    codec_version: int,
) -> list[float]:
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
    values += _track_block(player, track, codec_version)  # BLOCK_TRACK (36)
    values += _adrenaline_context(state, player)   # 2
    values += _opponent_slots(state, player, track)  # 25
    return values


def _phase_tail(
    state: GameState,
    player: PlayerState,
    decision: "Decision | None",
    codec_version: int,
) -> list[float]:
    """The final phase block (length ``BLOCK_PHASE_CONTEXT``).

    Combines the decision-gated context (:func:`_phase_context`) with the
    unconditional, state-pure ``round_num`` field written at
    ``_PHASE_ROUND_NUM_INDEX``. Codec v4 also fills indices 10-16 from the
    player and track. Factored out of :func:`encode_observation` so the
    shared-prefix pair path (Sprint C9) writes the phase tail through the EXACT
    same code, guaranteeing byte-identity.

    round_num and the v4 race fields are PURE GAME STATE, not decision context,
    so they are written here (unconditionally) rather than inside the
    decision-gated ``_phase_context``. Encoding the same ``state`` with
    ``decision=None`` and with a real decision is identical on every index
    EXCEPT the genuine decision-context bits 0..8 (code-review 2026-06-22 #1).
    Codec v3 leaves indices 10-18 at 0.

    Leak tradeoff (considered, accepted): because the value target is
    ``-rounds_remaining``, exposing the round counter lets the value net partly
    memorize the counter instead of learning a genuine cost-to-go. We keep
    round_num IN the observation -- removing it is a separate decision -- but make
    it state-pure here so train-time and search-time encodings match.
    """
    phase = _phase_context(state, decision)        # BLOCK_PHASE_CONTEXT
    phase[_PHASE_ROUND_NUM_INDEX] = _clip01(state.round_num / _ROUND_CAP)
    if codec_version == 4:
        _write_v4_race_context(phase, state, player)
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
    codec_version: int | None = None,
) -> np.ndarray:
    """Return a ``float32`` vector of shape ``(OBS_DIM,)`` for the learning seat.

    Pure: no mutation of ``state``. All values are in ``[-1, 1]``. ``decision``
    supplies the phase/decision-context block; ``None`` at episode reset
    boundaries yields a zero-filled decision block. ``codec_version`` defaults to
    the live codec (v4). Pass ``3`` to reproduce a historical checkpoint's inputs.
    """
    version = _resolve_codec(codec_version)
    player = state.get_player(player_id)
    track = state.track
    prefix = _state_prefix(state, player, track, version)
    phase = _phase_tail(state, player, decision, version)
    return _assemble(prefix, phase)


def encode_observation_pair(
    state: GameState,
    player_id: int,
    decision: "Decision | None",
    codec_version: int | None = None,
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

    Returns ``(prior_obs, value_obs)`` where, for the same ``codec_version``:
      * ``prior_obs == encode_observation(state, player_id, decision, codec_version)``
      * ``value_obs == encode_observation(state, player_id, None, codec_version)``
    byte-for-byte (``np.array_equal``). The caller MUST guarantee the seat
    equality precondition; this helper does not check it (it is the prior+value
    of the SAME ``player_id``).
    """
    version = _resolve_codec(codec_version)
    player = state.get_player(player_id)
    track = state.track
    prefix = _state_prefix(state, player, track, version)
    prior_obs = _assemble(prefix, _phase_tail(state, player, decision, version))
    value_obs = _assemble(prefix, _phase_tail(state, player, None, version))
    return prior_obs, value_obs
