"""Static per-action feature table for the dot-product head (Sprint A3).

The A0/A2 policy scores actions with a free ``Linear(hidden, ACTION_DIM)`` head:
every flat action index owns an untied weight vector, so nothing learned about
one card play transfers to a similar one and rarely-legal indices barely train.
A3 replaces that with **feature-derived** action scoring -- each flat action is
embedded from a fixed feature vector describing *what the action is* (its kind,
target gear, card-token histogram, react fields, ...), and the policy scores it
by a dot-product against a state embedding.

This module owns the *static* half of that: a ``(ACTION_DIM, ACTION_FEAT_DIM)``
float32 table, computed once at import from the codec's frozen enumerations
(:data:`heat.ml.action_codec._CARD_MULTISETS`, ``_TOKEN_ALPHABET``,
``_REACT_TABLE``) and the :mod:`heat.ml.spaces` sub-range offsets. Nothing here
depends on game state: the flat action space has fully index-intrinsic semantics
(the card sub-range enumerates frozen value-multisets, REACT is a fixed 8-slot
table, ...). State-dependent context (e.g. the heat cost of shifting *now*) is
the observation encoder's job, not the action head's.

Block layout of a feature row (``ACTION_FEAT_DIM == 25``, all values in [0, 1]):

===================  ====  ============================================
block                dims  content
===================  ====  ============================================
kind one-hot            5  GEAR / CARDS / REACT / SLIPSTREAM / DISCARD
gear                    4  one-hot target gear 1-4
cards histogram         8  per-token count (``_TOKEN_ALPHABET`` order) / 4
cards size              1  multiset size / 4
cards value sum         1  sum of printed values / 20
react                   4  cooldown_count/4, boost, adr_speed, adr_cooldown
slipstream              1  1.0 = take, 0.0 = decline
discard                 1  k / 7
===================  ====  ============================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from heat.engine import rules
from heat.ml.action_codec import _CARD_MULTISETS, _REACT_TABLE, _TOKEN_ALPHABET
from heat.ml.spaces import (
    ACTION_DIM,
    CARDS_OFFSET,
    DISCARD_OFFSET,
    DISCARD_SIZE,
    GEAR_OFFSET,
    GEAR_SIZE,
    REACT_OFFSET,
    SLIPSTREAM_OFFSET,
)

#: Width of one action feature row (see the block table in the module docstring).
ACTION_FEAT_DIM: int = 25

# --- Column offsets within a feature row (the frozen block layout). ---
_KIND_OFF: int = 0          # 5 dims: GEAR/CARDS/REACT/SLIPSTREAM/DISCARD
_GEAR_OFF: int = 5          # 4 dims: one-hot gear 1-4
_HIST_OFF: int = 9          # 8 dims: per-token histogram
_SIZE_COL: int = 17         # 1 dim: multiset size / 4
_VALSUM_COL: int = 18       # 1 dim: printed value sum / 20
_REACT_OFF: int = 19        # 4 dims: cooldown/4, boost, adr_speed, adr_cooldown
_SLIP_COL: int = 23         # 1 dim: 1.0 take / 0.0 decline
_DISC_COL: int = 24         # 1 dim: k / 7

# --- Kind one-hot column indices (order matches the spaces sub-ranges). ---
_KIND_GEAR: int = 0
_KIND_CARDS: int = 1
_KIND_REACT: int = 2
_KIND_SLIPSTREAM: int = 3
_KIND_DISCARD: int = 4

#: Deterministic printed value of each card token (Speed by value, Upgrade by
#: value, Stress / Heat contribute 0). Used for the "cards value sum" feature.
_TOKEN_VALUE: dict[str, int] = {
    "H": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
    "ST": 0,
    "U0": 0,
    "U5": 5,
}

# Normalization denominators (kept explicit so the [0, 1] range is auditable).
_HIST_DENOM: float = 4.0     # max token count in a size-4 multiset
_SIZE_DENOM: float = 4.0     # max multiset size (gear 4)
_VALSUM_DENOM: float = 20.0  # max printed value sum (4x U5)
_COOLDOWN_DENOM: float = 4.0  # max cooldown_count in the react table (3 + adren)
_DISCARD_DENOM: float = 7.0  # max discard-k (DISCARD_SIZE - 1)


def _build_table() -> NDArray[np.float32]:
    """Construct the static ``(ACTION_DIM, ACTION_FEAT_DIM)`` feature table."""
    table = np.zeros((ACTION_DIM, ACTION_FEAT_DIM), dtype=np.float32)

    # --- GEAR rows: one-hot target gear 1..4. ---
    for i in range(GEAR_SIZE):
        row = GEAR_OFFSET + i
        gear = rules.MIN_GEAR + i
        table[row, _KIND_OFF + _KIND_GEAR] = 1.0
        table[row, _GEAR_OFF + (gear - rules.MIN_GEAR)] = 1.0

    # --- CARDS rows: histogram + size + printed-value sum of each multiset. ---
    token_index = {tok: i for i, tok in enumerate(_TOKEN_ALPHABET)}
    for j, multiset in enumerate(_CARD_MULTISETS):
        row = CARDS_OFFSET + j
        table[row, _KIND_OFF + _KIND_CARDS] = 1.0
        for token in multiset:
            table[row, _HIST_OFF + token_index[token]] += 1.0 / _HIST_DENOM
        table[row, _SIZE_COL] = len(multiset) / _SIZE_DENOM
        table[row, _VALSUM_COL] = (
            sum(_TOKEN_VALUE[t] for t in multiset) / _VALSUM_DENOM
        )

    # --- REACT rows: the fixed 8-slot table's (cooldown, flags) fields. ---
    for i, slot in enumerate(_REACT_TABLE):
        row = REACT_OFFSET + i
        cooldown_count, use_boost, use_adr_speed, use_adr_cooldown = slot
        table[row, _KIND_OFF + _KIND_REACT] = 1.0
        table[row, _REACT_OFF + 0] = cooldown_count / _COOLDOWN_DENOM
        table[row, _REACT_OFF + 1] = float(use_boost)
        table[row, _REACT_OFF + 2] = float(use_adr_speed)
        table[row, _REACT_OFF + 3] = float(use_adr_cooldown)

    # --- SLIPSTREAM rows: offset 0 == take (1.0), offset 1 == decline (0.0). ---
    table[SLIPSTREAM_OFFSET + 0, _KIND_OFF + _KIND_SLIPSTREAM] = 1.0
    table[SLIPSTREAM_OFFSET + 0, _SLIP_COL] = 1.0
    table[SLIPSTREAM_OFFSET + 1, _KIND_OFF + _KIND_SLIPSTREAM] = 1.0
    # decline: kind one-hot only (slip feature stays 0.0).

    # --- DISCARD rows: index k == discard the k lowest cards. ---
    for k in range(DISCARD_SIZE):
        row = DISCARD_OFFSET + k
        table[row, _KIND_OFF + _KIND_DISCARD] = 1.0
        table[row, _DISC_COL] = k / _DISCARD_DENOM

    return table


#: The precomputed static feature table (built once at import).
_TABLE: NDArray[np.float32] = _build_table()

# --- Module-level contract asserts (fail fast on a codec/layout drift). ---
assert _TABLE.shape == (ACTION_DIM, ACTION_FEAT_DIM), _TABLE.shape
assert not np.isnan(_TABLE).any(), "action feature table has NaN"
assert _TABLE.min() >= 0.0 and _TABLE.max() <= 1.0, "features escape [0, 1]"
# Exactly one kind one-hot per row (the kind blocks are mutually exclusive).
_kind_sums = _TABLE[:, _KIND_OFF : _KIND_OFF + 5].sum(axis=1)
assert np.all(_kind_sums == 1.0), "kind one-hot not mutually exclusive"
# Spot-check the ("S3", "S3") multiset row's histogram (S3 count 2 -> 0.5).
_S3_S3_ROW = CARDS_OFFSET + _CARD_MULTISETS.index(("S3", "S3"))
assert _TABLE[_S3_S3_ROW, _HIST_OFF + _TOKEN_ALPHABET.index("S3")] == 0.5


def action_feature_table() -> NDArray[np.float32]:
    """Return a copy of the static ``(ACTION_DIM, ACTION_FEAT_DIM)`` table.

    A fresh copy is returned each call so callers (e.g. a policy registering it
    as a buffer) can never alias/mutate the shared module singleton.
    """
    return _TABLE.copy()
