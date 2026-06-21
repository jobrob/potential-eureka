"""Deterministic unit tests for the stochastic resource model.

Known deck -> known ``speed_distribution`` / ``expected_speed`` / ``speed_bands``,
checked by hand. The model is a pure combinatorial function of the owned-Basic
value multiset, so every assertion below is exact arithmetic (no RNG).
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from heat.models.cards import Card, CardType, Deck
from heat.models.player_state import PlayerState
from heat.planning.resource_model import (
    INTENTS,
    SpeedResourceModel,
    basic_value_multiset,
)


def _approx_sum_to_one(dist: dict[int, float]) -> None:
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Distribution shape from a known multiset
# ---------------------------------------------------------------------------


def test_gear1_mean_distribution_is_uniform_over_distinct_values() -> None:
    """A 1-card draw at gear 1 (mean) is just the value distribution itself."""
    # Pool {1,2,3,4}: gear-1 mean draws one card uniformly.
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    dist = model.speed_distribution(gear=1, intent="mean")
    _approx_sum_to_one(dist)
    assert dist == pytest.approx({1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})


def test_gear2_mean_distribution_matches_handcomputed_pair_sums() -> None:
    """Gear-2 mean draws 2 of {1,2,3,4} without replacement; sums hand-checked."""
    pool = (1, 2, 3, 4)
    model = SpeedResourceModel(basic_values=pool)
    dist = model.speed_distribution(gear=2, intent="mean")
    _approx_sum_to_one(dist)

    # Enumerate the 6 unordered pairs by hand: sums 3,4,5,5,6,7.
    expected_counts: dict[int, int] = {}
    for combo in combinations(pool, 2):
        expected_counts[sum(combo)] = expected_counts.get(sum(combo), 0) + 1
    total = sum(expected_counts.values())
    expected = {s: c / total for s, c in expected_counts.items()}
    assert dist == pytest.approx(expected)
    # Concretely: {3:1/6, 4:1/6, 5:2/6, 6:1/6, 7:1/6}
    assert dist[5] == pytest.approx(2 / 6)


def test_expected_speed_equals_distribution_mean() -> None:
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    dist = model.speed_distribution(gear=2, intent="mean")
    mean = sum(s * p for s, p in dist.items())
    assert model.expected_speed(gear=2, intent="mean") == pytest.approx(mean)
    # Mean pair-sum of {1,2,3,4} = 2 * mean(2.5) = 5.0.
    assert model.expected_speed(gear=2, intent="mean") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Intent ordering: conserve <= mean <= push (the key fidelity knob)
# ---------------------------------------------------------------------------


def test_intent_orders_expected_speed_conserve_le_mean_le_push() -> None:
    """conserve should be slowest, push fastest, mean in between (per gear)."""
    model = SpeedResourceModel(basic_values=(1, 1, 2, 2, 3, 3, 4, 4))
    for gear in (1, 2, 3):
        cons = model.expected_speed(gear, "conserve")
        mean = model.expected_speed(gear, "mean")
        push = model.expected_speed(gear, "push")
        assert cons <= mean + 1e-9, (gear, cons, mean)
        assert mean <= push + 1e-9, (gear, mean, push)
        # On a non-degenerate pool the conditioning should actually separate.
        assert cons < push


def test_conserve_biases_toward_low_cards() -> None:
    """conserve at gear 1 should put more mass on the lowest value than push."""
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    cons = model.speed_distribution(1, "conserve")
    push = model.speed_distribution(1, "push")
    assert cons.get(1, 0.0) > push.get(1, 0.0)
    assert push.get(4, 0.0) > cons.get(4, 0.0)


# ---------------------------------------------------------------------------
# Bands + extra offset
# ---------------------------------------------------------------------------


def test_speed_bands_full_support_sums_to_one_and_is_sorted() -> None:
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    bands = model.speed_bands(gear=2, intent="mean")
    speeds = [s for s, _ in bands]
    assert speeds == sorted(speeds)
    assert math.isclose(sum(p for _, p in bands), 1.0, abs_tol=1e-9)


def test_speed_bands_collapse_respects_max_bands() -> None:
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    bands = model.speed_bands(gear=2, intent="mean", max_bands=2)
    assert len(bands) <= 2
    assert math.isclose(sum(p for _, p in bands), 1.0, abs_tol=1e-9)


def test_expected_extra_shifts_every_speed() -> None:
    """A boost/adrenaline EV offset shifts the whole distribution by that amount."""
    base = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    shifted = SpeedResourceModel(basic_values=(1, 2, 3, 4), expected_extra=2.0)
    bd = base.speed_distribution(1, "mean")
    sd = shifted.speed_distribution(1, "mean")
    assert sd == pytest.approx({s + 2: p for s, p in bd.items()})


def test_gear_zero_returns_degenerate_extra_only() -> None:
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4), expected_extra=0.0)
    assert model.speed_distribution(0, "mean") == {0: 1.0}


def test_invalid_intent_raises() -> None:
    model = SpeedResourceModel(basic_values=(1, 2, 3))
    with pytest.raises(ValueError):
        model.speed_distribution(1, "sideways")


# ---------------------------------------------------------------------------
# from_player + basic_value_multiset (composition wiring)
# ---------------------------------------------------------------------------


def _player_with_cards(cards: list[Card]) -> PlayerState:
    """Build a minimal PlayerState whose deck holds exactly ``cards`` (empty hand)."""
    deck = Deck(list(cards))
    return PlayerState(player_id=0, name="t", deck=deck, hand=[])


def test_basic_value_multiset_collects_only_speed_cards() -> None:
    cards = [
        Card(CardType.SPEED, 1, "s1"),
        Card(CardType.SPEED, 3, "s3"),
        Card(CardType.HEAT, 0, "h"),
        Card(CardType.STRESS, 0, "st"),
        Card(CardType.UPGRADE, 5, "u"),
    ]
    p = _player_with_cards(cards)
    assert sorted(basic_value_multiset(p)) == [1, 3]


def test_from_player_folds_stress_as_mean_basic() -> None:
    """A stress card adds one population member at the mean owned-Basic value."""
    # Two Basics {2,4} -> mean 3; one stress folds in as a '3'.
    cards = [
        Card(CardType.SPEED, 2, "s2"),
        Card(CardType.SPEED, 4, "s4"),
        Card(CardType.STRESS, 0, "st"),
    ]
    p = _player_with_cards(cards)
    model = SpeedResourceModel.from_player(p)
    # Population (rounded ints): [2, 3, 4] -- the stress became a 3.
    assert sorted(model._int_pool) == [2, 3, 4]


def test_from_player_standard_deck_distribution_is_well_formed() -> None:
    """A standard starting player yields a valid, normalized distribution."""
    import random

    p = PlayerState.create(0, rng=random.Random(0))
    model = SpeedResourceModel.from_player(p)
    for gear in (1, 2, 3, 4):
        for intent in INTENTS:
            dist = model.speed_distribution(gear, intent)
            _approx_sum_to_one(dist)
            assert all(s >= 0 for s in dist)
