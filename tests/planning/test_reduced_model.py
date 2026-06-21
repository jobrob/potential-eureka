"""Deterministic unit tests for the reduced one-turn simulator.

Each test hand-builds a ``(ReducedState, ReducedAction)`` whose true-engine
outcome is computable by hand from the rules (``corners_crossed`` /
``corner_heat_cost`` / spin-at-``cost>heat`` / cooldown-in-gears-1-2), and
asserts ``step_reduced`` matches. Speed is made deterministic by using a
single-value resource pool so ``expected_speed`` is exact and every mode agrees.
"""

from __future__ import annotations

import pytest

from heat.engine import rules
from heat.models.track import Corner, Space, Track
from heat.planning.reduced_model import (
    ReducedAction,
    ReducedState,
    StepOutcome,
    step_reduced,
)
from heat.planning.resource_model import SpeedResourceModel


def _track(length: int, corners: list[Corner], laps: int = 1) -> Track:
    """A bare straight track with explicit corners (no blocking, 1 lane)."""
    spaces = [Space(i, lanes=1) for i in range(length)]
    return Track(
        name="t", spaces=spaces, corners=corners,
        start_positions=[0], laps=laps,
    )


def _fixed_speed_model(value: int) -> SpeedResourceModel:
    """A model whose every draw is exactly ``value`` per card.

    A single-value owned-Basic pool means a gear-``g`` play sums to ``g*value``
    deterministically (the distribution is a point mass), so the speed is exact
    and ``expected``/``banded``/``empirical`` all predict the same outcome -- the
    arithmetic is fully hand-checkable.
    """
    return SpeedResourceModel(basic_values=(value,) * 6)


# ---------------------------------------------------------------------------
# Speed wiring: gear * per-card value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gear,value", [(1, 2), (2, 2), (3, 1), (2, 3)])
def test_point_speed_is_gear_times_value(gear: int, value: int) -> None:
    model = _fixed_speed_model(value)
    # Long open track, no corners: car advances exactly gear*value spaces.
    track = _track(length=50, corners=[])
    state = ReducedState(pos=0, heat=6, gear=gear)
    out = step_reduced(
        state, ReducedAction(dgear=0, intent="mean"), track, model, mode="expected"
    )
    assert out.next_pos == gear * value
    assert not out.spun
    assert out.next_gear == gear


# ---------------------------------------------------------------------------
# Corner heat arithmetic (no spin): cost = max(0, speed - limit)
# ---------------------------------------------------------------------------


def test_corner_overspeed_pays_exact_heat() -> None:
    """Crossing one corner above its limit costs (speed-limit) heat, no spin."""
    # gear 2 * value 3 = speed 6; corner at [5,5] limit 4 -> cost = 6-4 = 2.
    model = _fixed_speed_model(3)
    track = _track(length=50, corners=[Corner(start=5, end=5, speed_limit=4)])
    state = ReducedState(pos=0, heat=6, gear=2)
    out = step_reduced(
        state, ReducedAction(dgear=0, intent="mean"), track, model, mode="expected"
    )
    assert not out.spun
    assert out.next_pos == 6
    # heat 6 - cost 2 = 4; gear 2 cools min(1, expected_in_hand=1, room) -> but
    # room = HEAT_POOL_SIZE(6) - 4 = 2, cooldown_amount(2)=1, so +1 -> 5.
    assert out.next_heat == 5
    assert out.next_gear == 2


def test_under_limit_corner_costs_no_heat() -> None:
    model = _fixed_speed_model(2)  # gear 2 -> speed 4
    track = _track(length=50, corners=[Corner(start=3, end=3, speed_limit=5)])
    state = ReducedState(pos=0, heat=3, gear=2)
    out = step_reduced(
        state, ReducedAction(dgear=0, intent="mean"), track, model, mode="expected"
    )
    assert not out.spun
    assert out.next_pos == 4
    # No corner cost; gear-2 cooldown adds 1 (room available): 3 -> 4.
    assert out.next_heat == 4


# ---------------------------------------------------------------------------
# Spin trigger: spins exactly when corner cost > heat available
# ---------------------------------------------------------------------------


def test_spin_when_cost_exceeds_heat() -> None:
    """cost(=4) > heat(=3) -> spin: reset to corner.start-1, heat 0, gear 1."""
    # gear 3 * value 3 = speed 9; corner [6,6] limit 5 -> cost 4. heat=3 < 4.
    model = _fixed_speed_model(3)
    corner = Corner(start=6, end=6, speed_limit=5)
    track = _track(length=50, corners=[corner])
    state = ReducedState(pos=0, heat=3, gear=3)
    out = step_reduced(
        state, ReducedAction(dgear=0, intent="mean"), track, model, mode="expected"
    )
    assert out.spun
    assert out.next_pos == corner.start - 1  # 5
    assert out.next_heat == 0
    assert out.next_gear == 1


def test_no_spin_when_cost_equals_heat() -> None:
    """cost == heat is payable (spin requires cost STRICTLY greater)."""
    # gear 2 * value 4 = speed 8; corner [4,4] limit 5 -> cost 3 == heat 3.
    model = _fixed_speed_model(4)
    track = _track(length=50, corners=[Corner(start=4, end=4, speed_limit=5)])
    state = ReducedState(pos=0, heat=3, gear=2)
    out = step_reduced(
        state, ReducedAction(dgear=0, intent="mean"), track, model, mode="expected"
    )
    assert not out.spun
    assert out.next_pos == 8
    # heat 3 - 3 = 0, then gear-2 cooldown +1 (room=6) -> 1.
    assert out.next_heat == 1


# ---------------------------------------------------------------------------
# Cooldown refill: gears 1-2 only
# ---------------------------------------------------------------------------


def test_gear1_refills_heat_more_than_gear2() -> None:
    """Gear 1 cools up to 3 (capped by expected-in-hand=1); gear 3 cools nothing."""
    model = _fixed_speed_model(1)  # tiny speed so no corner interaction
    track = _track(length=50, corners=[])

    # Gear 1: cooldown_amount=3 but capped at expected-in-hand (1) -> +1.
    g1 = step_reduced(
        ReducedState(pos=0, heat=2, gear=1),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="expected",
    )
    assert g1.next_heat == 3  # 2 + 1

    # Gear 3: no cooldown -> heat unchanged at 2.
    g3 = step_reduced(
        ReducedState(pos=0, heat=2, gear=3),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="expected",
    )
    assert g3.next_heat == 2


def test_cooldown_never_exceeds_pool_size() -> None:
    """Cooldown is clamped to HEAT_POOL_SIZE -- a full pool gains nothing."""
    model = _fixed_speed_model(1)
    track = _track(length=50, corners=[])
    out = step_reduced(
        ReducedState(pos=0, heat=rules.HEAT_POOL_SIZE, gear=1),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="expected",
    )
    assert out.next_heat == rules.HEAT_POOL_SIZE


# ---------------------------------------------------------------------------
# Gear-shift heat payment (the +-2 shift costs 1 heat)
# ---------------------------------------------------------------------------


def test_plus_two_gear_shift_pays_one_heat() -> None:
    """A +2 shift (gear 1->3) costs 1 heat up front, paid before the corner."""
    model = _fixed_speed_model(1)  # gear 3 -> speed 3, no corners
    track = _track(length=50, corners=[])
    out = step_reduced(
        ReducedState(pos=0, heat=4, gear=1),
        ReducedAction(dgear=2, intent="mean"), track, model, mode="expected",
    )
    assert out.next_gear == 3
    # heat 4 - 1 (shift) = 3; gear 3 -> no cooldown.
    assert out.next_heat == 3


def test_plus_one_gear_shift_is_free() -> None:
    model = _fixed_speed_model(1)
    track = _track(length=50, corners=[])
    out = step_reduced(
        ReducedState(pos=0, heat=4, gear=1),
        ReducedAction(dgear=1, intent="mean"), track, model, mode="expected",
    )
    assert out.next_gear == 2
    # No shift cost; gear-2 cooldown +1 -> 5.
    assert out.next_heat == 5


def test_illegal_plus_two_shift_with_no_heat_falls_back_to_nearest_legal() -> None:
    """With 0 heat a +2 shift is illegal; we fall back to the nearest legal gear."""
    model = _fixed_speed_model(1)
    track = _track(length=50, corners=[])
    out = step_reduced(
        ReducedState(pos=0, heat=0, gear=1),
        ReducedAction(dgear=2, intent="mean"), track, model, mode="expected",
    )
    # legal_gear_shifts(1, 0) = [(1,0),(2,0)]; nearest to target 3 is gear 2.
    assert out.next_gear == 2


# ---------------------------------------------------------------------------
# p_spin marginalization + modes
# ---------------------------------------------------------------------------


def test_p_spin_is_zero_when_no_corner() -> None:
    model = _fixed_speed_model(2)
    track = _track(length=50, corners=[])
    out = step_reduced(
        ReducedState(pos=0, heat=6, gear=2),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="banded",
    )
    assert out.p_spin == 0.0
    assert out.exp_rounds_cost == pytest.approx(1.0)


def test_p_spin_one_when_every_draw_spins() -> None:
    """Deterministic over-limit draw with no heat: certain spin -> p_spin == 1."""
    model = _fixed_speed_model(3)  # gear 2 -> speed 6
    track = _track(length=50, corners=[Corner(start=4, end=4, speed_limit=1)])
    out = step_reduced(
        ReducedState(pos=0, heat=0, gear=2),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="banded",
    )
    assert out.spun
    assert out.p_spin == pytest.approx(1.0)
    assert out.exp_rounds_cost == pytest.approx(2.0)


def test_partial_spin_probability_with_spread_distribution() -> None:
    """A mixed distribution yields 0 < p_spin < 1 even if the point pick is safe."""
    # Pool {1,2,3,4} at gear 1: speeds 1..4 each 1/4. Corner limit 2 with heat 1:
    # cost = max(0, speed-2); spin iff cost>1 -> speed>=4 -> only speed 4 spins.
    model = SpeedResourceModel(basic_values=(1, 2, 3, 4))
    track = _track(length=50, corners=[Corner(start=2, end=2, speed_limit=2)])
    out = step_reduced(
        ReducedState(pos=0, heat=1, gear=1),
        ReducedAction(dgear=0, intent="mean"), track, model, mode="banded",
    )
    assert out.p_spin == pytest.approx(0.25)


def test_invalid_mode_raises() -> None:
    model = _fixed_speed_model(2)
    track = _track(length=50, corners=[])
    with pytest.raises(ValueError):
        step_reduced(
            ReducedState(pos=0, heat=6, gear=2),
            ReducedAction(dgear=0, intent="mean"), track, model, mode="bogus",
        )


def test_step_outcome_is_frozen_dataclass() -> None:
    out = StepOutcome(
        next_pos=1, next_heat=2, next_gear=3, spun=False,
        p_spin=0.0, exp_rounds_cost=1.0,
    )
    with pytest.raises(Exception):
        out.next_pos = 9  # type: ignore[misc]
