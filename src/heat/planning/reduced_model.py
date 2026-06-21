"""Reduced one-turn simulator -- ``step_reduced(s, a) -> s'`` over ``(pos, heat, gear)``.

The DP's transition kernel (README B-design 3.2), and the object the B0 spike
measures against the true engine. The reduced state drops the hand, draw order,
and discard; speed is supplied by the :class:`SpeedResourceModel` and everything
else (gear-shift heat, corner crossing/cost, spin trigger, cooldown refill) is
computed with the **real** ``rules.*`` primitives so the abstraction error lives
*only* in the resource model, not in re-implemented arithmetic.

Transition (mirrors ``engine.phases``):

1. **Pay gear-shift heat.** A ``|dgear| == 2`` shift costs 1 heat
   (``rules.legal_gear_shifts``); the new gear is clamped to ``[1, 4]``.
2. **Resolve speed ``S``** at ``(gear', intent)`` per the fidelity ``mode``:
   ``expected`` uses ``E[S]`` (rounded), ``banded``/``empirical`` marginalize
   over the resource model's distribution.
3. **Corners + cost.** ``rules.corners_crossed(pos, pos+S)`` then
   ``sum(rules.corner_heat_cost(S, c))`` -- the exact engine formula.
4. **Spin vs advance.** If ``cost > heat`` (heat *after* the gear payment): the
   engine spins -- reset ``pos`` to ``max(0, first_corner.start - 1)``, heat to
   ``0`` (the engine pays all remaining heat), gear to ``1``. Else advance to
   ``pos + S``, ``heat -= cost``, then add the **expected cooldown refill** for
   ``gear'`` (gears 1-2 only), bounded by the pool and the modeled heat clog.

The ``StepOutcome`` carries both the point prediction (``next_*``, ``spun``) and
the distributional summaries the DP wants (``p_spin``, ``exp_rounds_cost``) so a
single call serves both the spike's apples-to-apples check and a future solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from heat.engine import rules
from heat.models.track import Track
from heat.planning.resource_model import SpeedResourceModel


#: Fidelity modes for resolving the per-turn speed (README 3.2 ladder).
MODES: tuple[str, ...] = ("expected", "banded", "empirical")

#: Expected number of heat cards sitting in hand available to cool, used to bound
#: the modeled cooldown refill. The hand is dropped by the abstraction, so we
#: model cooldown yield as the gear's ``cooldown_amount`` capped at this estimate
#: of clogged heat in hand. Sized at 1: across a solo game a player typically has
#: ~1 heat card clogging the 7-card hand at any time (3 starting heat + paid heat
#: cycle through a ~17-card deck). R3 (clog timing) is exactly what the spike's
#: heat-error metric exposes; this is the in-expectation treatment the design asks
#: for, not a claim of exactness.
_EXPECTED_HEAT_IN_HAND: float = 1.0


@dataclass(frozen=True)
class ReducedState:
    """A point in the reduced MDP: absolute progress, heat available, gear."""

    pos: int
    heat: int
    gear: int


@dataclass(frozen=True)
class ReducedAction:
    """A reduced action: gear delta (legal shift) and an intended speed band."""

    dgear: int
    intent: str


@dataclass(frozen=True)
class StepOutcome:
    """Result of one reduced turn.

    Attributes:
        next_pos: Predicted absolute position after the turn (spin-reset or
            advanced). Position only -- lap is implicit in absolute ``pos`` for
            the within-track checks the spike makes; the engine keeps the lap on
            a spin and credits laps on advance, which the spike reads separately.
        next_heat: Predicted heat available after paying gear shift + corner cost
            (and adding expected cooldown), or ``0`` on a spin.
        next_gear: Predicted gear (``1`` on a spin, else the shifted gear).
        spun: Point prediction of whether the modeled (mode-resolved) turn spins.
        p_spin: Probability of a spin marginalized over the resource model's
            speed distribution (always computed, all modes) -- the DP's risk term.
        exp_rounds_cost: Expected rounds this transition costs (``1`` for a clean
            turn; more when ``p_spin`` is high, since a spin burns the turn and
            forces a slow recovery). A simple ``1 + p_spin`` shaping the DP uses
            as the per-step cost-to-go increment.
    """

    next_pos: int
    next_heat: int
    next_gear: int
    spun: bool
    p_spin: float
    exp_rounds_cost: float


def _shift_gear(gear: int, dgear: int, heat: int) -> tuple[int, int]:
    """Apply a gear delta via the real ``legal_gear_shifts``; return (gear', cost).

    The requested ``dgear`` is matched against the legal shifts for the current
    gear and heat. If the exact target is illegal (e.g. a +2 with no heat, or a
    shift past the [1,4] clamp), we fall back to the nearest legal gear in the
    direction of intent -- the controller would re-validate the same way
    (README 5.3). Returns the resulting gear and the heat cost paid (0 or 1).
    """
    legal = rules.legal_gear_shifts(gear, heat)
    target = gear + dgear
    # Exact match if available.
    for new_gear, cost in legal:
        if new_gear == target:
            return new_gear, cost
    # Nearest legal gear toward the target (the controller's re-validation).
    best = min(legal, key=lambda gc: (abs(gc[0] - target), gc[1]))
    return best[0], best[1]


def _corner_cost_at_speed(track: Track, pos: int, speed: int) -> tuple[list, int]:
    """Corners crossed advancing ``speed`` from ``pos`` and their total heat cost.

    Pure reuse of ``rules.corners_crossed`` + ``rules.corner_heat_cost`` -- the
    exact engine arithmetic, lap-aware via ``spaces_moved=speed`` so a full-lap
    move still charges every corner. Returns ``(crossed_corners, total_cost)``.
    """
    if speed <= 0:
        return [], 0
    end_pos = (pos + speed) % track.length if track.length else pos
    crossed = rules.corners_crossed(pos, end_pos, track, spaces_moved=speed)
    cost = sum(rules.corner_heat_cost(speed, c) for c in crossed)
    return crossed, cost


def _expected_cooldown(gear: int, heat_after: int) -> int:
    """Expected heat returned by cooldown at ``gear`` (gears 1-2 only).

    The engine cools heat cards *in hand* (``cooldown_amount`` 3/1/0 by gear),
    bounded by how many heat cards actually clog the hand. The abstraction drops
    the hand, so we credit the gear's cooldown amount capped by
    :data:`_EXPECTED_HEAT_IN_HAND` and by the room left in the pool
    (``HEAT_POOL_SIZE - heat_after``). Integer-rounded so ``next_heat`` stays an
    integer the DP indexes on.
    """
    amount = rules.cooldown_amount(gear)
    if amount <= 0:
        return 0
    room = max(0, rules.HEAT_POOL_SIZE - heat_after)
    return int(min(amount, _EXPECTED_HEAT_IN_HAND, room))


def _resolve_speed(model: SpeedResourceModel, gear: int, intent: str, mode: str):
    """Return ``(point_speed, distribution)`` for ``(gear, intent)`` under ``mode``.

    * ``expected`` -- point speed is ``round(E[S])``; the distribution is still
      returned (for ``p_spin``), but the point prediction uses the mean.
    * ``banded`` / ``empirical`` -- point speed is the distribution's *mode*
      (most-likely band), and the full distribution drives ``p_spin``. (Banded
      and empirical share the same combinatorial distribution here -- the spike
      reports both so we can see whether the cheaper point statistic suffices.)
    """
    dist = model.speed_distribution(gear, intent)
    if mode == "expected":
        point = int(round(model.expected_speed(gear, intent)))
    else:
        # Most-likely speed (ties -> lower speed, the conservative pick).
        point = min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return point, dist


def step_reduced(
    state: ReducedState,
    action: ReducedAction,
    track: Track,
    model: SpeedResourceModel,
    *,
    mode: str = "banded",
) -> StepOutcome:
    """Predict one reduced turn ``(pos, heat, gear) -> StepOutcome``.

    See the module docstring for the transition. ``mode`` selects the speed
    fidelity (:data:`MODES`). ``p_spin`` is always marginalized over the resource
    model's full distribution regardless of ``mode``, so the DP gets a calibrated
    risk even when the point prediction uses the mean.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    # 1. Gear shift (real legal-shift logic) + its heat payment.
    new_gear, gear_cost = _shift_gear(state.gear, action.dgear, state.heat)
    heat_after_shift = state.heat - gear_cost

    # 2. Resolve the point speed + the full distribution for p_spin.
    point_speed, dist = _resolve_speed(model, new_gear, action.intent, mode)

    # 3. p_spin: marginalize the spin indicator over the speed distribution. A
    #    draw spins iff its corner cost exceeds the post-shift heat.
    p_spin = 0.0
    for speed, prob in dist.items():
        _, cost = _corner_cost_at_speed(track, state.pos, speed)
        if cost > heat_after_shift:
            p_spin += prob

    # 4. Point transition at the resolved speed.
    crossed, cost = _corner_cost_at_speed(track, state.pos, point_speed)
    spun = cost > heat_after_shift

    if spun:
        # Engine: reset to corner.start-1, pay ALL remaining heat (heat -> 0),
        # gear -> 1. The first crossed corner is where the engine places the car.
        spin_corner = crossed[0]
        next_pos = max(0, spin_corner.start - 1)
        return StepOutcome(
            next_pos=next_pos,
            next_heat=0,
            next_gear=1,
            spun=True,
            p_spin=p_spin,
            exp_rounds_cost=1.0 + p_spin,
        )

    # Advance: pay corner cost, then add expected cooldown refill (gears 1-2).
    next_pos = (state.pos + point_speed) % track.length if track.length else state.pos
    heat_paid = heat_after_shift - cost
    refill = _expected_cooldown(new_gear, heat_paid)
    next_heat = min(rules.HEAT_POOL_SIZE, heat_paid + refill)
    return StepOutcome(
        next_pos=next_pos,
        next_heat=next_heat,
        next_gear=new_gear,
        spun=False,
        p_spin=p_spin,
        exp_rounds_cost=1.0 + p_spin,
    )
