"""Frozen ML contract constants and spaces for the HEAT RL layer (Sprint 5a).

This module is the single source of truth for the observation/action/reward
contract that the feature extractor (5a), the Gym environment (5b), the model
(5c), and the agent (5d) must all agree on. Per the roadmap (§3, §4.2) it is
landed and frozen FIRST; everything else codes against the constants here.

Contract summary
----------------
* Observation: a fixed-length ``float32`` vector of shape ``(OBS_DIM,)`` whose
  values all lie in ``[-1, 1]`` (see :mod:`heat.ml.features`).
* Action: a single flattened, masked ``Discrete(ACTION_DIM)`` space. The
  per-decision-kind sub-ranges (offsets) are defined here and consumed by
  :mod:`heat.ml.action_codec`.
* Reward: :func:`step_reward` — sparse terminal placement reward plus optional
  (default-off) dense shaping.

Changing any constant here is a *contract change*, not a local edit: it ripples
to 5b/5c/5d and to the checkpoint sidecar (``codec_version``).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from heat.models.game_state import GameState
from heat.models.player_state import PlayerState

# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------

#: Bumped whenever OBS_DIM / ACTION_DIM / the codec layout changes. The 5d
#: checkpoint sidecar records this so a stale model fails fast on load.
#: v2 (Sprint B): replaced the 4-dim next-corner lookahead with an all-corners
#: ego-centric track block (Option A whole-track obs). OBS_DIM 72 -> 104. v1
#: checkpoints are intentionally rejected by the MLAgent version tripwire.
#: v3 (Option C prep, code-review 2026-06-22 #1): made the observation a pure
#: function of game state -- ``round_num`` (phase-block index 9) is now encoded
#: unconditionally by ``features.encode_observation`` instead of only when a real
#: ``decision`` is present. OBS_DIM/ACTION_DIM unchanged; only the value written
#: at index 9 changes (it is now identical for ``decision=None`` and a real
#: decision). v1/v2 checkpoints are rejected by the MLAgent version tripwire.
CODEC_VERSION: int = 3

# ---------------------------------------------------------------------------
# Observation space (§3.1)
# ---------------------------------------------------------------------------

#: Fixed observation vector length. v2: 104 (was 72), after swapping the 4-dim
#: track-lookahead block for the 36-dim all-corners track block (see below).
OBS_DIM: int = 104

#: Maximum number of seats. Variable player counts are encoded with a fixed
#: opponent-slot layout; absent/finished opponents are zero-filled with a
#: presence flag of 0.
MAX_PLAYERS: int = 6

# Feature-block sizes (§3.1 table). The ORDER and COUNT are the contract; the
# 5a producer packs exactly these, in this order, zero-padding the tail to
# reach OBS_DIM.
BLOCK_HAND_HISTOGRAM: int = 8
BLOCK_OWN_GEAR: int = 4
BLOCK_OWN_KINEMATICS: int = 4
BLOCK_DECK_COMPOSITION: int = 6

# --- Track block (Sprint B Option A: all-corners ego-centric whole-track obs) ---
#: Maximum number of corner slots. The generator caps tracks at 7 corners
#: (``num_corners_range=(3, 7)``), so 8 slots encode EVERY corner of EVERY
#: generated track with one slot of slack.
MAX_CORNERS: int = 8
#: Floats per corner slot: (dist_ahead, speed_limit, corner_len, lanes). No
#: explicit presence bit -- ``dist_ahead == 0`` is the padding marker, since
#: every real corner reads ``dist_ahead > 0`` (a corner at the current position
#: counts as a full lap away, ``rules.distance_to_next_corner``).
CORNER_SLOT_FLOATS: int = 4
#: Track-global floats appended after the corner slots: (laps_remaining,
#: dist_to_finish, heat_pool, pos_in_lap).
TRACK_GLOBALS: int = 4
#: All-corners ego-centric track block: MAX_CORNERS slots + the globals
#: sub-block. Replaces the old 4-dim BLOCK_TRACK_LOOKAHEAD.
BLOCK_TRACK: int = MAX_CORNERS * CORNER_SLOT_FLOATS + TRACK_GLOBALS  # 36

BLOCK_ADRENALINE_CONTEXT: int = 2
#: 5 floats per opponent slot, for the MAX_PLAYERS - 1 opponent seats.
OPP_SLOT_FLOATS: int = 5
BLOCK_OPPONENT_SLOTS: int = OPP_SLOT_FLOATS * (MAX_PLAYERS - 1)  # 25
#: Phase / decision context (one-hot kind + react/slipstream/round bits). The
#: remainder up to OBS_DIM is explicit zero padding owned by this block. NOTE
#: (§3.1 footgun): this is DERIVED, so the new BLOCK_TRACK must be subtracted
#: here -- otherwise the phase-context block silently absorbs the size change and
#: the obs layout corrupts. The literal assert below guards against a miscount.
BLOCK_PHASE_CONTEXT: int = (
    OBS_DIM
    - BLOCK_HAND_HISTOGRAM
    - BLOCK_OWN_GEAR
    - BLOCK_OWN_KINEMATICS
    - BLOCK_DECK_COMPOSITION
    - BLOCK_TRACK
    - BLOCK_ADRENALINE_CONTEXT
    - BLOCK_OPPONENT_SLOTS
)  # == 19

assert BLOCK_PHASE_CONTEXT >= 0, "feature blocks exceed OBS_DIM"
assert BLOCK_PHASE_CONTEXT == 19, BLOCK_PHASE_CONTEXT  # freeze the intended size


def observation_space() -> gym.spaces.Box:
    """Return the bounded, normalized observation space.

    All features are scaled into ``[-1, 1]`` by :mod:`heat.ml.features`.
    """
    return gym.spaces.Box(
        low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Action space layout (§3.2)
# ---------------------------------------------------------------------------
#
# A single flattened Discrete(ACTION_DIM). Each decision kind owns a contiguous
# sub-range; the offsets below are the contract. The env masks the union down
# to the current decision kind's legal actions.

#: GEAR: index -> target gear 1..4.
GEAR_SIZE: int = 4

#: CARDS: index over distinct card value-multisets reachable across gears 1-4.
#: A play is a size-``gear`` multiset over the 8-token alphabet
#: {Speed1, Speed2, Speed3, Speed4, Upgrade0, Upgrade5, Stress0, Heat}. The
#: count of all such multisets for gear in 1..4 is the (bounded, enumerate-once)
#: constant ``C``. See :mod:`heat.ml.action_codec` for the enumeration that
#: produces exactly this many entries.
CARDS_SIZE: int = 494  # C

#: REACT: fixed 8-slot enumeration of legal
#: (cooldown_count, use_boost, use_adrenaline_speed, use_adrenaline_cooldown).
REACT_SIZE: int = 8

#: SLIPSTREAM: {take, decline}.
SLIPSTREAM_SIZE: int = 2

#: DISCARD: index 0 == discard none; index k == discard the k lowest discardable
#: cards (by value then id). Capped to MAX_PLAYERS-independent bound.
DISCARD_SIZE: int = 8

# Contiguous offsets (start index of each sub-range).
GEAR_OFFSET: int = 0
CARDS_OFFSET: int = GEAR_OFFSET + GEAR_SIZE          # 4
REACT_OFFSET: int = CARDS_OFFSET + CARDS_SIZE        # 498
SLIPSTREAM_OFFSET: int = REACT_OFFSET + REACT_SIZE   # 506
DISCARD_OFFSET: int = SLIPSTREAM_OFFSET + SLIPSTREAM_SIZE  # 508

#: Total flattened action-space size (frozen).
ACTION_DIM: int = DISCARD_OFFSET + DISCARD_SIZE  # 516


# ---------------------------------------------------------------------------
# Reward (§3.3)
# ---------------------------------------------------------------------------

#: Reward mode (Sprint 8C). ``"race"`` (default) == the existing terminal
#: placement reward; ``"solo"`` == dense progress (already on via
#: ``SHAPING_WEIGHT``) + a terminal finish bonus paid ONLY when the lone car
#: FINISHES (``terminated``), never when the episode truncates. Set from
#: ``PPOConfig.reward_mode`` by :func:`heat.ml.model.apply_shaping_config`,
#: mirroring the ``SHAPING_WEIGHT`` global pattern. Default ``"race"`` keeps every
#: existing run byte-for-byte unchanged.
REWARD_MODE: str = "race"

#: Terminal finish bonus for solo mode (Sprint 8C), paid once when the lone car
#: completes its laps (``terminated``). Positive-only -- no negative rewards
#: anywhere in solo, so there is no "crash to end the episode early" exploit.
#: Default matches the prototype's 5.0 (``experiments/proto_solo.py``).
SOLO_FINISH_BONUS: float = 5.0

#: Dense shaping weight. Defaulted to 0 so the first training run is pure-sparse
#: (see §6.3). 5c may tune this via PPOConfig without editing 5b.
SHAPING_WEIGHT: float = 0.0

#: Per-space progress coefficient inside the (default-off) dense shaping term.
SHAPING_PROGRESS_COEF: float = 1.0

#: Optional bounded anti-spinout penalty weight (Idea 3). Default 0 == off, so
#: existing behavior is unchanged. Penalizes the learner the step it spins out
#: (overshoots a corner speed limit / cannot pay the heat), nudging the policy
#: away from degenerate over-pushing.
SHAPING_SPINOUT_WEIGHT: float = 0.0

#: Hard cap on the per-step spinout penalty magnitude, keeping the term from
#: dominating the sparse win signal even if ``SHAPING_SPINOUT_WEIGHT`` is large.
SHAPING_SPINOUT_CAP: float = 0.05


def _placement_reward(state: GameState, learner_id: int) -> float:
    """Sparse terminal placement reward in ``[-1, +1]`` from finish order.

    ``+1`` for the winner, graded linearly down for mid-ranks:
    ``1 - 2*(rank-1)/(n-1)`` where ``rank`` is 1-based finish position among the
    starting field. Returns 0.0 if the learner has not finished.
    """
    n = state.starting_player_count or state.num_players
    if n <= 1:
        return 0.0
    player = state.get_player(learner_id)
    if not player.finished:
        return 0.0
    rank = player.finish_order  # 1-based
    return 1.0 - 2.0 * (rank - 1) / (n - 1)


def terminal_margin(state: GameState, player_id: int) -> float:
    """Normalized progress lead over the best opponent at game end (Sprint A6).

    A *dense* terminal target: how far ahead of the whole field the learner is,
    as a fraction of one full lap. Positive == leading; a one-full-lap lead
    saturates at ``+1`` (and a one-lap deficit at ``-1``). Unlike the sparse
    :func:`_placement_reward` (which says only *whether* you won), this grades a
    dominant position apart from a lucky squeaker, so GAE returns -- and therefore
    the PPO value loss AND advantages -- carry graded information.

    Using the same absolute-progress arithmetic as
    :func:`heat.ml.features._track_block`
    (``abs_pos = player.lap * length + player.position``), each player's
    *remaining* distance to the finish is::

        remaining(p) = 0.0                                  if p.finished
                     = max(0, laps*length - abs_pos(p))     otherwise

    A finished player has zero remaining, so it counts as strictly ahead of any
    opponent still on track. The margin is the gap between the best (smallest)
    opponent remaining and the learner's own, normalized by ``track.length`` and
    clipped to ``[-1, +1]``::

        margin = clip((min_opp remaining(opp) - remaining(me)) / length, -1, +1)

    In an unclipped 2-seat game ``margin(s, 0) == -margin(s, 1)`` (antisymmetric).
    Returns ``0.0`` for a solo field (``n <= 1``), which has no opponent to lead.
    """
    n = state.starting_player_count or state.num_players
    if n <= 1:
        return 0.0
    length = state.track.length or 1
    laps = state.track.laps or 1
    total_len = laps * length

    def _remaining(p: PlayerState) -> float:
        if p.finished:
            return 0.0
        abs_pos = p.lap * length + p.position
        return max(0.0, float(total_len - abs_pos))

    my_remaining = _remaining(state.get_player(player_id))
    opp_remaining = [
        _remaining(p) for p in state.players if p.player_id != player_id
    ]
    if not opp_remaining:  # pragma: no cover - guarded by n <= 1 above
        return 0.0
    margin = (min(opp_remaining) - my_remaining) / length
    return float(np.clip(margin, -1.0, 1.0))


def _spinout_penalty(
    prev: GameState, curr: GameState, learner_id: int
) -> float:
    """Return ``1.0`` the step the learner *newly* spins out, else ``0.0``.

    Reads the engine's :attr:`heat.models.player_state.PlayerState.spun_out`
    flag, which the corner-check sets when the car overshoots a corner speed
    limit it cannot pay for. Firing only on the rising edge (``not prev`` ->
    ``curr``) charges the penalty once per spin rather than every step the flag
    stays set. The caller scales this in ``[0, 1]`` value by a weight and a hard
    cap, so the term is bounded by construction.
    """
    prev_p = prev.get_player(learner_id)
    curr_p = curr.get_player(learner_id)
    return 1.0 if (curr_p.spun_out and not prev_p.spun_out) else 0.0


def step_reward(
    prev: GameState,
    curr: GameState,
    learner_id: int,
    done: bool,
    *,
    terminated: bool = False,
    reward_mode: str | None = None,
    shaping_weight: float | None = None,
    spinout_weight: float | None = None,
) -> float:
    """Reward for the learning seat between two driver steps.

    Default policy (§3.3, ``REWARD_MODE == "race"``):
      * Terminal (sparse): graded placement reward computed from
        ``curr.finished_players`` when ``done`` is True.
      * Dense shaping (optional, ``SHAPING_WEIGHT`` defaults to 0): lap-aware
        per-step progress of the learner. Kept tiny so it never dominates the
        win signal; off by default for the first run.

    Solo policy (Sprint 8C, ``REWARD_MODE == "solo"``):
      * Terminal: a fixed :data:`SOLO_FINISH_BONUS` paid ONLY when the episode
        genuinely *terminates* (the lone car finished its laps), never on
        truncation. ``_placement_reward`` returns 0 for the solo (``n <= 1``)
        field, so the bonus is the whole terminal signal. No negative branch ->
        no crash-to-end-the-episode exploit; speed is induced purely by
        ``gamma < 1`` discounting the same bonus + progress earlier (§4.3).
      * Dense progress: the same shaping term below, run with ``SHAPING_WEIGHT > 0``.

    ``terminated`` defaults to ``False`` so any existing caller (race mode) is
    byte-for-byte unchanged; only the solo branch reads it. ``reward_mode``,
    ``shaping_weight``, and ``spinout_weight`` optionally override their mutable
    module defaults for isolated consumers such as Direction-A self-play, whose
    sparse reward contract must not inherit configuration from an SB3 run in the
    same process.
    """
    reward = 0.0
    active_mode = REWARD_MODE if reward_mode is None else reward_mode
    active_shaping = SHAPING_WEIGHT if shaping_weight is None else shaping_weight
    active_spinout = (
        SHAPING_SPINOUT_WEIGHT if spinout_weight is None else spinout_weight
    )

    if done:
        if active_mode == "solo":
            # Solo terminal: finish bonus ONLY on a real finish (terminated),
            # never on truncation. _placement_reward returns 0 for n<=1, so the
            # placement term contributes nothing -- the bonus is the whole
            # terminal signal. No negative branch => no crash-to-end exploit.
            if terminated:
                reward += SOLO_FINISH_BONUS
        else:
            reward += _placement_reward(curr, learner_id)

    if active_shaping != 0.0:
        prev_p = prev.get_player(learner_id)
        curr_p = curr.get_player(learner_id)
        length = curr.track.length or 1
        # Lap-aware spaces advanced this step.
        prev_abs = prev_p.lap * length + prev_p.position
        curr_abs = curr_p.lap * length + curr_p.position
        progress = (curr_abs - prev_abs) / length
        # Solo mode is positive-only (DoD a, §4.3): the engine can push a car
        # BACKWARD (slipstream resolution / spinout), which would make the raw
        # progress delta negative. Clamp it at 0 in solo so there is never a
        # negative reward -- removing any incentive to crash/spin to stop
        # accruing. Race mode keeps the signed progress unchanged.
        if active_mode == "solo":
            progress = max(0.0, progress)
        reward += active_shaping * SHAPING_PROGRESS_COEF * progress

    # Optional bounded anti-spinout penalty (Idea 3); default-off (weight 0). The
    # raw penalty is in [0, 1]; the subtracted magnitude is hard-capped at
    # SHAPING_SPINOUT_CAP so it can never dominate the placement reward.
    if active_spinout != 0.0:
        penalty = _spinout_penalty(prev, curr, learner_id)  # in [0, 1]
        reward -= min(SHAPING_SPINOUT_CAP, active_spinout * penalty)

    return reward
