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

# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------

#: Bumped whenever OBS_DIM / ACTION_DIM / the codec layout changes. The 5d
#: checkpoint sidecar records this so a stale model fails fast on load.
CODEC_VERSION: int = 1

# ---------------------------------------------------------------------------
# Observation space (§3.1)
# ---------------------------------------------------------------------------

#: Fixed observation vector length (PLAN.md "~60-80").
OBS_DIM: int = 72

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
BLOCK_TRACK_LOOKAHEAD: int = 4
BLOCK_ADRENALINE_CONTEXT: int = 2
#: 5 floats per opponent slot, for the MAX_PLAYERS - 1 opponent seats.
OPP_SLOT_FLOATS: int = 5
BLOCK_OPPONENT_SLOTS: int = OPP_SLOT_FLOATS * (MAX_PLAYERS - 1)  # 25
#: Phase / decision context (one-hot kind + react/slipstream/round bits). The
#: remainder up to OBS_DIM is explicit zero padding owned by this block.
BLOCK_PHASE_CONTEXT: int = (
    OBS_DIM
    - BLOCK_HAND_HISTOGRAM
    - BLOCK_OWN_GEAR
    - BLOCK_OWN_KINEMATICS
    - BLOCK_DECK_COMPOSITION
    - BLOCK_TRACK_LOOKAHEAD
    - BLOCK_ADRENALINE_CONTEXT
    - BLOCK_OPPONENT_SLOTS
)  # == 19

assert BLOCK_PHASE_CONTEXT >= 0, "feature blocks exceed OBS_DIM"


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

#: Dense shaping weight. Defaulted to 0 so the first training run is pure-sparse
#: (see §6.3). 5c may tune this via PPOConfig without editing 5b.
SHAPING_WEIGHT: float = 0.0

#: Per-space progress coefficient inside the (default-off) dense shaping term.
SHAPING_PROGRESS_COEF: float = 1.0


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


def step_reward(
    prev: GameState,
    curr: GameState,
    learner_id: int,
    done: bool,
) -> float:
    """Reward for the learning seat between two driver steps.

    Default policy (§3.3):
      * Terminal (sparse): graded placement reward computed from
        ``curr.finished_players`` when ``done`` is True.
      * Dense shaping (optional, ``SHAPING_WEIGHT`` defaults to 0): lap-aware
        per-step progress of the learner. Kept tiny so it never dominates the
        win signal; off by default for the first run.

    The shaping coefficients live as module-level constants so 5c can tune them
    (via PPOConfig) without editing the 5b environment.
    """
    reward = 0.0

    if done:
        reward += _placement_reward(curr, learner_id)

    if SHAPING_WEIGHT != 0.0:
        prev_p = prev.get_player(learner_id)
        curr_p = curr.get_player(learner_id)
        length = curr.track.length or 1
        # Lap-aware spaces advanced this step.
        prev_abs = prev_p.lap * length + prev_p.position
        curr_abs = curr_p.lap * length + curr_p.position
        progress = (curr_abs - prev_abs) / length
        reward += SHAPING_WEIGHT * SHAPING_PROGRESS_COEF * progress

    return reward
