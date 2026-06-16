"""Pure move-evaluator for the strong heuristic agent (Sprint 6E redesign).

Everything here is a *pure* function over ``(GameState, player_id, candidate
move)``: no mutation of game state, no global RNG, no I/O. The agent
(:mod:`heat.agents.strong_heuristic`) and a future ``LookaheadAgent`` both
import these so the value model lives in exactly one place.

The single currency is **expected net spaces of progress over the rest of the
race**, but the objective is **relative**, not absolute. HEAT is a zero-sum
finishing-order race; an agent that only maximises its own progress has no
lever once its own move is locally optimal. So every term in
:func:`evaluate_move` is expressed in spaces, but the dominant *positional*
levers are decoupled from raw own-progress:

    value(plan) =  alpha * own_progress           # spaces_gained (down-weightable)
                 + RELATIVE_WEIGHT * relative_term  # saturating gap change (P0)
                 - heat_price_eff * heat_spent      # rank-adjusted heat price (P0)
                 - P(spinout) * spinout_loss_eff    # rank-adjusted spin loss (P0)
                 + solvency_term                    # real forward-heat projection (P2)
                 + block_term                       # real block projection (P3)

The two structural breaks from a pure own-progress maximiser are:

1. ``relative_term`` -- a *saturating* (``tanh``) function of the change in the
   signed gap to the nearest K rivals. Pulling away from a rival you already
   crush saturates to ~0; closing/holding a *contested* (near-zero) gap is
   valuable. This is NOT collinear with own-progress (a linear gap-difference
   would be); it changes the argmax in wheel-to-wheel situations.
2. A **lead-dependent risk posture**: ``heat_price_eff`` and
   ``spinout_loss_eff`` shift with race rank (a leader conserves and refuses
   risk; a trailer pushes and accepts variance). This changes *which* plan is
   optimal as a function of standing, not distance.

All tunables are interpretable quantities in *spaces* (or as bounded swings /
fractions), documented at their definitions, so 6B can sweep them empirically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Track
from heat.engine import rules


# ---------------------------------------------------------------------------
# Tunables (all in "spaces" currency or derived from the rules)
# ---------------------------------------------------------------------------

#: Default shadow price of one heat, in spaces. Derivation: one heat buys, on
#: average, the ability to exceed a corner limit by ~1 (≈ keeps 1 space of
#: speed you'd otherwise shed) or fund a ±2 gear shift (tempo). A gear-1 turn
#: recovers up to 3 heat at the cost of a slow turn. ~1.0-1.5 spaces/heat is
#: defensible; we take the midpoint 1.25 as the reproducible default. It is an
#: interpretable scalar, unlike the old ``*10``/``*15`` penalties.
DEFAULT_HEAT_PRICE: float = 1.25

#: Spaces lost when a car spins out: it is stopped just before the corner,
#: loses its speed for the turn, picks up stress (future clutter), and burns
#: the turn. Modelled as a large fixed loss in the same currency. Retuned up
#: from 8.0 to 11.0 (P2) so the *expected* spin penalty
#: ``p_spinout * spinout_loss_eff`` reliably exceeds the marginal spaces a
#: risky play buys -- driving strong spins strictly below the weak heuristic.
SPINOUT_LOSS_SPACES: float = 11.0

#: Penalty (spaces) for projecting heat-insolvent before clearing upcoming
#: corners -- i.e. the plan risks a *future* forced spin. Scaled by how far
#: negative the projected heat trajectory goes. Sized at ~ SPINOUT_LOSS_SPACES/2
#: per heat of shortfall: a real deterrent that does not dominate relative_term.
SOLVENCY_PENALTY_PER_HEAT: float = 5.0

#: Mean value of a freshly drawn Basic (SPEED) card from the standard 1..4
#: starting deck, used as a fallback when the agent's own deck is unknown.
_FALLBACK_BASIC_MEAN: float = 2.5

# --- P0: relative finishing-order objective ----------------------------------

#: Weight on the relative (gap-change) term. Default 1.0; surfaced so 6B can
#: sweep it. Raise ``alpha`` (own-progress weight) rather than shrinking this if
#: the agent proves too aggressive -- that keeps the relative *shape* intact.
RELATIVE_WEIGHT: float = 1.0

#: Own-progress weight. Kept at 1.0 -- raw progress is genuinely most of what
#: wins a race; the relative term only *tilts* contested decisions.
OWN_PROGRESS_WEIGHT: float = 1.0

#: Gap scale (spaces) over which the saturating gap-value ``f`` flattens.
#: The doc's first guess was 3.0; empirically 5.0 separates the ladder best
#: (a wider steep region keeps contests within ~2 moves' reach valuable without
#: over-rewarding wheel-to-wheel jostling that costs own progress).
GAP_SOFTNESS: float = 5.0

#: Max value (spaces) of fully winning one contest. The doc's first guess was
#: 2.0; empirically 1.0 separates best -- at 2.0 the term distorted card choices
#: enough to regress own progress against a strong own-progress maximiser. 1.0
#: is still large enough to flip a close card choice (unlike the old capped-0.6
#: bonus) but keeps own-progress dominant.
GAP_SCALE: float = 1.0

#: Number of nearest rivals the relative term considers (the car directly ahead
#: and directly behind by race progress). Keeps the term cheap and undominated
#: by far-off cars.
NEAREST_RIVALS_K: int = 2

#: Lead-dependent risk posture swings (P0). ``lead_frac`` is 1.0 for the leader,
#: 0.0 for last. A leader pays MORE per heat and fears a spin MORE (conserve); a
#: trailer pays LESS and discounts spins (push, accept variance). Deliberately
#: larger than the WIP's inert +/-10% so they actually change behaviour.
LEAD_PRICE_SWING: float = 0.4   # +/-40% around base heat price
LEAD_RISK_SWING: float = 0.3    # +/-30% around base spin-out loss

# --- P2: spin-out aversion schedule ------------------------------------------

#: Heat buffer (spaces of slack against the worst-case stress/boost flip) over
#: which spin-out probability decays smoothly to ~0. Two heat of buffer => ~0
#: risk. Replaces the WIP's hard step function.
SPINOUT_SLACK_SCALE: float = 2.0

#: Expected per-corner *overage* a sensible driver pays for tempo when a corner
#: is genuinely reachable in the solvency horizon (used by the real projection).
_PROJECTED_CORNER_OVERAGE: float = 1.0

# --- P3: real block projection ------------------------------------------------

#: Weight on blocked-rival spaces, in the relative currency (denied spaces are
#: real gap widening). Default 1.0.
BLOCK_WEIGHT: float = 1.0


# ---------------------------------------------------------------------------
# Deck-quality model (the agent knows its own deck *contents*, not order)
# ---------------------------------------------------------------------------


def expected_basic_value(player: PlayerState) -> float:
    """Mean value of the Basic (SPEED) cards in the player's whole deck+hand.

    Used as the expected value of a stress flip / boost flip / future draw.
    Stress/boost resolution flips until a Basic card appears, so the relevant
    statistic is the mean of the *Basic* cards the player owns -- derivable
    from deck contents without consuming draw order. Falls back to the 1..4
    deck mean when the player owns no Basic cards (degenerate).
    """
    speed_values: list[int] = []
    for card in player.deck:
        if card.card_type == CardType.SPEED:
            speed_values.append(card.value)
    for card in player.hand:
        if card.card_type == CardType.SPEED:
            speed_values.append(card.value)
    if not speed_values:
        return _FALLBACK_BASIC_MEAN
    return sum(speed_values) / len(speed_values)


def max_basic_value(player: PlayerState) -> float:
    """Highest Basic (SPEED) card value the player owns (deck + hand).

    A stress / boost flip resolves to *some* Basic card the player owns, so the
    realised speed contributed by a stress card can be as high as this value.
    The strong agent uses it to size a safety buffer against the *bad* flip
    (see :func:`evaluate_move`'s ``speed_variance``) so it does not plan corner
    crossings on a razor-thin heat margin. Falls back to 4 (the top of the
    standard 1..4 deck) when no Basic cards are owned.
    """
    values: list[int] = [
        c.value for c in player.deck if c.card_type == CardType.SPEED
    ]
    values += [c.value for c in player.hand if c.card_type == CardType.SPEED]
    return float(max(values)) if values else 4.0


def play_speed(
    cards: tuple[Card, ...],
    expected_stress: float,
) -> float:
    """Expected speed contributed by a card play.

    Non-stress cards contribute their value (heat cards contribute 0 via
    ``calculate_speed``). Each stress card contributes ``expected_stress``
    (the mean Basic value it will flip to at reveal).
    """
    base = float(rules.calculate_speed(cards))
    stress_count = sum(1 for c in cards if c.card_type == CardType.STRESS)
    return base + stress_count * expected_stress


def play_speed_variance(
    cards: tuple[Card, ...],
    expected_stress: float,
    max_basic: float,
) -> float:
    """Upside speed margin of a play beyond its expected speed.

    Each Stress card flips to a random owned Basic at reveal; the realised
    value can exceed the mean (``expected_stress``) by up to
    ``max_basic - expected_stress``. The sum across the play's stress cards is
    the most the corner-check speed can overshoot the plan -- the buffer a
    risk-averse driver keeps so an above-mean flip does not force a spin.
    """
    stress_count = sum(1 for c in cards if c.card_type == CardType.STRESS)
    return stress_count * max(0.0, max_basic - expected_stress)


# ---------------------------------------------------------------------------
# Corner / projection helpers
# ---------------------------------------------------------------------------


def landed_lap_and_pos(
    track: Track,
    from_position: int,
    from_lap: int,
    spaces: int,
) -> tuple[int, int]:
    """Lap-aware landing: ``(end_lap, end_pos)`` after moving ``spaces``.

    Fixes the P1 lap-accounting bug: :func:`rules.calculate_move_position`
    returns the position modulo ``track.length`` and only a *boolean*
    ``crossed_finish`` -- so a caller that keeps the original ``from_lap`` sees a
    *lower* lap-aware progress on any finish-crossing move, silently zeroing the
    positional terms exactly on the high-value laps. We credit laps via
    :func:`rules.laps_completed_by_move`, which also handles a single move that
    spans two or more laps.
    """
    if track.length == 0:
        return from_lap, from_position
    if spaces <= 0:
        return from_lap, from_position
    laps = rules.laps_completed_by_move(from_position, spaces, track)
    new_pos, _ = rules.calculate_move_position(
        from_position, spaces, track, from_lap
    )
    return from_lap + laps, new_pos


def corners_crossed_by_move(
    track: Track,
    from_position: int,
    from_lap: int,
    spaces: int,
) -> list[Corner]:
    """Corners crossed by advancing ``spaces`` from ``(from_position)``.

    Lap-aware: a full-lap move that lands on the start space still crosses
    every corner (mirrors :func:`rules.corners_crossed` with ``spaces_moved``).
    """
    if spaces <= 0:
        return []
    new_pos, _ = rules.calculate_move_position(
        from_position, spaces, track, from_lap
    )
    return rules.corners_crossed(from_position, new_pos, track, spaces_moved=spaces)


def corner_cost_for_speed(
    crossed: list[Corner],
    corner_speed: float,
) -> int:
    """Total heat cost across ``crossed`` corners at integer ``corner_speed``.

    ``corner_speed`` is the speed used for the corner check (cards + boost +
    adrenaline, EXCLUDING slipstream). Rounded down to an int because corner
    cost is integer-valued; expected stress contributions are conservative.
    """
    speed_int = int(corner_speed)
    return sum(rules.corner_heat_cost(speed_int, c) for c in crossed)


def project_solvency(
    state: GameState,
    player: PlayerState,
    end_position: int,
    end_lap: int,
    heat_after_turn: int,
    planned_gear: int,
    expected_speed: float,
    horizon_corners: int,
    heat_price: float,
) -> float:
    """Penalty (spaces) when the plan risks running the heat pool dry (P2).

    A faithful forward-heat projection. Walking corner-by-corner from the
    landing space, we model the heat trajectory honestly:

    - **Cooldown is credited conditionally**, NOT once per corner regardless.
      A low gear only recovers heat if there is room to take a slow turn before
      the corner; we credit at most one low-gear turn's cooldown when the
      distance to the corner is large enough (``dist >= expected_speed``),
      capped at the pool size. The WIP credited full cooldown every corner,
      which made the projected pool monotonically *rise* so the penalty never
      fired -- the term was dead weight.
    - **Cost is the real overage** the player must carry through the corner,
      using a deck-quality expected arrival speed rather than a flat 0.5. A
      sensible driver sheds most speed for a corner, but cannot always arrive
      under the limit; we charge the larger of the realistic per-corner overage
      and the overage at the projected arrival speed.

    The term fires a penalty (scaled by ``SOLVENCY_PENALTY_PER_HEAT``) only when
    the projected pool would go negative before the corners are cleared -- i.e.
    the plan genuinely cannot stay solvent. Otherwise it is zero (no speculative
    drag on aggression).
    """
    track = state.track
    if horizon_corners <= 0 or not track.corners:
        return 0.0

    heat = float(heat_after_turn)
    cooldown_per_turn = rules.cooldown_amount(planned_gear)
    # Realistic per-corner arrival overage: a sensible driver sheds speed but
    # carries a little over the limit for tempo. Use the larger of a small
    # baseline and the deck-quality arrival overage above the *tightest* limit.
    arrival_speed = max(1, int(round(expected_speed)))
    worst_shortfall = 0.0

    pos = end_position
    for _ in range(horizon_corners):
        corner, dist = rules.distance_to_next_corner(track, pos)
        if corner is None:
            break
        # Credit cooldown only if there is room for a slow turn before the
        # corner (the car can decelerate / take a low gear). One turn's worth,
        # capped at the pool.
        if dist >= max(1, int(round(expected_speed))) and cooldown_per_turn > 0:
            heat = min(float(rules.HEAT_POOL_SIZE), heat + cooldown_per_turn)
        # Pay the real overage at the corner. A controlled approach pays the
        # baseline tempo overage; a fast deck pays more if it cannot shed below
        # the limit. corner_heat_cost already clamps at 0 for slow corners.
        overage = max(
            _PROJECTED_CORNER_OVERAGE,
            float(rules.corner_heat_cost(arrival_speed, corner)),
        )
        heat -= overage
        if heat < worst_shortfall:
            worst_shortfall = heat
        # Advance past this corner for the next iteration.
        pos = (corner.end + 1) % track.length

    if worst_shortfall < 0:
        # Risk of a future forced spin: penalise the shortfall only.
        return worst_shortfall * SOLVENCY_PENALTY_PER_HEAT
    return 0.0


# ---------------------------------------------------------------------------
# Opponent / positional helpers
# ---------------------------------------------------------------------------


def race_progress(player: PlayerState, track: Track) -> int:
    """Total spaces travelled (lap-aware), for ranking players head-to-head."""
    return player.lap * track.length + player.position


def race_rank(state: GameState, player_id: int) -> tuple[int, int]:
    """Return ``(rank, num_active)`` for ``player_id`` (rank 0 = leader).

    Ranked by total race progress among non-finished players (and the focal
    player). Finished players count ahead of everyone still racing.
    """
    track = state.track
    me = state.get_player(player_id)
    ahead = 0
    active = 0
    for other in state.players:
        if other.player_id == player_id:
            continue
        if other.finished:
            ahead += 1
            continue
        active += 1
        if race_progress(other, track) > race_progress(me, track):
            ahead += 1
    return ahead, active + 1


def lead_frac(state: GameState, player_id: int) -> float:
    """Return the focal player's standing as a fraction in ``[0, 1]``.

    1.0 = leader, 0.0 = last among active cars. ``0.5`` (neutral) when the
    player is the sole survivor / no rivals (``total <= 1``).
    """
    rank, total = race_rank(state, player_id)
    if total <= 1:
        return 0.5
    return 1.0 - rank / (total - 1)


def effective_risk(
    state: GameState,
    player_id: int,
    base_heat_price: float,
) -> tuple[float, float]:
    """Lead-dependent risk posture (P0): ``(heat_price_eff, spinout_loss_eff)``.

    A leader (``lead_frac -> 1``) pays MORE per heat and fears a spin MORE
    (conserve, bank heat, refuse marginal risk). A trailer (``lead_frac -> 0``)
    pays LESS and discounts spins (push, spend the pool, accept variance to make
    up places -- finishing "safely last" gains nothing). The end-game heat-price
    decay is applied by the agent *after* this rank adjustment, multiplicatively.
    """
    lf = lead_frac(state, player_id)
    # Map lead_frac in [0,1] to a centred swing in [-1, +1].
    centred = (lf - 0.5) * 2.0
    heat_price_eff = base_heat_price * (1.0 + LEAD_PRICE_SWING * centred)
    spinout_loss_eff = SPINOUT_LOSS_SPACES * (1.0 + LEAD_RISK_SWING * centred)
    return max(0.0, heat_price_eff), max(0.0, spinout_loss_eff)


def _gap_value(gap: float) -> float:
    """Saturating gap-value ``f(g) = GAP_SCALE * tanh(g / GAP_SOFTNESS)``.

    Monotone, concave for ``g>0`` and convex for ``g<0``, so
    ``f(gap_after) - f(gap_before)`` is largest in magnitude when the gap is
    near zero -- i.e. wheel-to-wheel, where own-progress is a poor proxy for
    finishing prospects.
    """
    return GAP_SCALE * math.tanh(gap / GAP_SOFTNESS)


def relative_term(
    state: GameState,
    player_id: int,
    prog_self_before: int,
    prog_self_after: int,
) -> float:
    """Expected change in the closest competitive gaps (P0 -- the new lever).

    For the ``K = NEAREST_RIVALS_K`` rivals whose signed gap to the focal player
    is smallest in magnitude (the cars directly ahead and behind by race
    progress), sum ``w(r) * (f(gap_after) - f(gap_before))`` where
    ``gap = prog_self - prog(rival)`` and ``f`` is the saturating
    :func:`_gap_value`. Rivals have not moved yet this turn, so ``prog(rival)``
    is fixed within the decision. ``w = 1.0`` for the nearest ``K``, ``0``
    beyond.

    This is deliberately *not* "did I pass someone" (collinear with own
    progress) nor a linear gap-difference (also collinear). The ``tanh``
    saturation down-weights uncontested gaps and up-weights contested ones, so
    two plans with identical own-progress but different opponent geometry score
    differently -- the structural break from the absolute objective.
    """
    track = state.track
    gaps_before: list[float] = []
    for other in state.players:
        if other.player_id == player_id or other.finished:
            continue
        op = race_progress(other, track)
        gaps_before.append(float(prog_self_before - op))
    if not gaps_before:
        return 0.0  # sole survivor / no rivals
    # Keep the K rivals with the smallest |gap_before| (nearest contestants).
    # Pair each before-gap with its after-gap: rivals are stationary, so
    # gap_after = gap_before + (prog_self_after - prog_self_before).
    delta_self = float(prog_self_after - prog_self_before)
    indexed = sorted(range(len(gaps_before)), key=lambda i: abs(gaps_before[i]))
    total = 0.0
    for i in indexed[:NEAREST_RIVALS_K]:
        gb = gaps_before[i]
        ga = gb + delta_self
        total += _gap_value(ga) - _gap_value(gb)
    return total


@dataclass(frozen=True)
class PositionEffect:
    """Outcome of landing on ``target_pos`` given blocking."""

    landed_pos: int
    spaces_lost_to_block: int


def resolve_landing(
    state: GameState,
    player_id: int,
    target_pos: int,
) -> PositionEffect:
    """Where the player actually lands after blocking, and spaces lost.

    Pure read of :func:`rules.resolve_blocked_position`. ``spaces_lost`` is the
    forward distance between the intended and actual space (0 if not blocked).
    """
    track = state.track
    landed = rules.resolve_blocked_position(
        target_pos, track, state.players, player_id
    )
    lost = (target_pos - landed) % track.length
    return PositionEffect(landed_pos=landed, spaces_lost_to_block=lost)


def block_value(
    state: GameState,
    player_id: int,
    landed_pos: int,
    landed_lap: int,
) -> float:
    """Real block projection (P3), valued in the relative currency.

    For the focal agent's landing ``(landed_lap, landed_pos)``, check each
    trailing rival that could reach the contested space next turn and ask: *if I
    occupy this space, does the rival's natural next move land on a space whose
    lanes are now full (counting me), forcing
    :func:`rules.resolve_blocked_position` to bounce it backward?* The denied
    spaces widen the (negative, trailing) gap to that rival, and the value is
    priced through the **same saturating gap function** the relative term uses,
    so a block is worth exactly the relative-currency gap change it buys -- it
    composes consistently with :func:`relative_term` rather than double-counting
    raw spaces. It does not construct an illegal move (pure read of the rules).

    The rival's "natural next move" is approximated by its current ``gear`` as a
    mean card play (gear spaces of expected movement) -- a deterministic, cheap
    proxy that is enough to detect whether our car plugs the lane it needs.
    """
    track = state.track
    length = track.length
    if length == 0:
        return 0.0

    my_prog = landed_lap * length + landed_pos
    landed_idx = landed_pos % length

    # Build the field as it stands AFTER the agent lands: same players, but the
    # focal car sits at landed_idx. We read positions only (resolve_blocked_position
    # ignores the focal car via moving_player_id, so we model the rival moving
    # *into* a space the agent now occupies by placing the agent there).
    best_value = 0.0
    for other in state.players:
        if other.player_id == player_id or other.finished:
            continue
        rival_prog = race_progress(other, track)
        gap = my_prog - rival_prog
        # Only a *trailing* rival close enough to reach our landing space (or
        # just past it) next turn can be bounced. Its gear bounds its reach.
        reach = max(1, other.gear)
        if not (1 <= gap <= reach + 2):
            continue
        # The rival's natural target: move `gear` spaces (expected play) forward.
        target_pos, _ = rules.calculate_move_position(
            other.position, reach, track, other.lap
        )
        # Resolve where the rival actually lands, given the agent now occupies
        # landed_idx. We snapshot positions: the focal car is at landed_idx.
        denied = _denied_spaces_for_rival(
            state, player_id, landed_idx, other, target_pos
        )
        if denied <= 0:
            continue
        # Price the block in the relative currency: pushing the rival back by
        # ``denied`` spaces widens our (positive) lead gap to it from
        # ``gap`` to ``gap + denied``. Value that gap change through the same
        # saturating ``f`` the relative term uses, so blocking is never worth
        # more than the contest it actually decides.
        gap_change = _gap_value(gap + denied) - _gap_value(gap)
        if gap_change > best_value:
            best_value = gap_change
    return BLOCK_WEIGHT * best_value


def _denied_spaces_for_rival(
    state: GameState,
    player_id: int,
    agent_landed_idx: int,
    rival: PlayerState,
    rival_target_pos: int,
) -> float:
    """Forward spaces the rival loses because the agent plugs its landing lane.

    Pure read of :func:`rules.resolve_blocked_position` with the field including
    the agent at ``agent_landed_idx``. Returns ``intended - actual`` forward
    distance for the rival (0 if it is not bounced).
    """
    track = state.track
    length = track.length

    # Construct the occupancy the rival faces: every other active car at its
    # current position, plus the agent at its post-move landing space. We pass
    # this as ``all_players`` to resolve_blocked_position, with the *rival* as
    # the moving player so it counts the agent (and others) as blockers.
    others: list[PlayerState] = []
    for p in state.players:
        if p.player_id == rival.player_id:
            continue
        if p.finished:
            continue
        if p.player_id == player_id:
            # Place the agent at its landed space (post-move occupancy).
            others.append(_PosProxy(p.player_id, agent_landed_idx, p.finished))
        else:
            others.append(_PosProxy(p.player_id, p.position % length, p.finished))

    landed = rules.resolve_blocked_position(
        rival_target_pos, track, others, rival.player_id
    )
    intended = rival_target_pos % length
    denied = (intended - landed) % length
    # Guard against a spurious full-lap "denial".
    if denied >= length:
        return 0.0
    return float(denied)


@dataclass(frozen=True)
class _PosProxy:
    """Minimal position-only stand-in for resolve_blocked_position reads."""

    player_id: int
    position: int
    finished: bool


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveEval:
    """Scored breakdown of a candidate move (for debugging/tests)."""

    value: float
    expected_speed: float
    heat_spent: int
    corner_cost: int
    p_spinout: float
    relative_value: float = 0.0


def evaluate_move(
    state: GameState,
    player_id: int,
    *,
    expected_speed: float,
    heat_spent: int,
    from_position: int,
    from_lap: int,
    planned_gear: int,
    heat_price: float,
    horizon_corners: int,
    opponent_aware: bool,
    blocking: bool,
    enable_solvency: bool,
    speed_variance: float = 0.0,
    spinout_loss: float | None = None,
) -> MoveEval:
    """Score a candidate move in expected net spaces of progress (relative).

    This is the single objective. ``expected_speed`` is the corner-check speed
    (cards + expected stress + planned boost/adrenaline, EXCLUDING slipstream);
    ``heat_spent`` is heat paid *this turn* for the plan (gear shift + boost);
    corner overage is priced inside. ``speed_variance`` is the upside margin by
    which the realised speed can exceed ``expected_speed`` (from above-mean
    stress flips, see :func:`play_speed_variance`); it is used ONLY to size the
    spin-out risk against the *bad* case, not to inflate progress.

    ``heat_price`` should be the rank-adjusted ``heat_price_eff`` (the agent
    derives it via :func:`effective_risk` + end-game decay). ``spinout_loss``,
    if given, is the rank-adjusted ``spinout_loss_eff``; it defaults to the
    base :data:`SPINOUT_LOSS_SPACES` so existing callers (and the strength-0
    floor, which is rank-neutral) are unaffected.

    The caller supplies booleans gating the lookahead / opponent / blocking
    terms so the difficulty ladder is a pure function of which terms are
    switched on. ``opponent_aware`` now gates the **relative_term** (the new P0
    lever), not the deleted overtake bonus.
    """
    player = state.get_player(player_id)
    track = state.track
    heat_available = player.heat_available
    spin_loss = SPINOUT_LOSS_SPACES if spinout_loss is None else spinout_loss

    crossed = corners_crossed_by_move(
        track, from_position, from_lap, int(round(expected_speed))
    )
    corner_cost = corner_cost_for_speed(crossed, expected_speed)

    # Total heat the plan needs at the *expected* speed (this-turn spend +
    # corner overage). This is the central estimate.
    total_heat_needed = heat_spent + corner_cost

    # Conservative ("bad flip") heat need: the realised corner-check speed can
    # exceed the plan by ``speed_variance`` when stress cards flip high. A
    # strong driver keeps a buffer for that case rather than planning on the
    # mean, which is what kept the naive evaluator's heat pool perpetually near
    # zero (forced spins on any above-mean flip).
    risk_speed = expected_speed + speed_variance
    crossed_risk = corners_crossed_by_move(
        track, from_position, from_lap, int(round(risk_speed))
    )
    corner_cost_risk = corner_cost_for_speed(crossed_risk, risk_speed)
    total_heat_risk = heat_spent + corner_cost_risk

    # Probability the plan cannot pay -> forced spin (P2: smooth schedule).
    if total_heat_needed > heat_available:
        # Even the expected case cannot pay: a certain spin.
        p_spinout = 1.0
    elif crossed:
        # Affordable in expectation; price the residual risk by how thin the
        # buffer is against the worst-case (high-flip) heat need. A smooth ramp
        # over ``SPINOUT_SLACK_SCALE`` heat of slack replaces the old hard step:
        # ~0 risk once two heat of buffer remain, rising to ~1.0 as the buffer
        # vanishes. With ``speed_variance == 0`` this reduces to the central
        # slack; a positive variance shrinks the effective buffer so a play that
        # only just covers the mean is penalised for the bad flip.
        risk_slack = heat_available - total_heat_risk
        if risk_slack >= SPINOUT_SLACK_SCALE:
            p_spinout = 0.0
        else:
            p_spinout = max(
                0.0, min(1.0, (SPINOUT_SLACK_SCALE - risk_slack) / SPINOUT_SLACK_SCALE)
            )
    else:
        p_spinout = 0.0

    # Lap-aware landing (P1): credit laps so finish-crossing moves see the
    # correct end_lap / end progress, instead of the stale from_lap that
    # silently zeroed the positional terms on the high-value laps.
    end_lap, new_pos = landed_lap_and_pos(
        track, from_position, from_lap, int(round(expected_speed))
    )
    effect = resolve_landing(state, player_id, new_pos)
    spaces_gained = expected_speed - effect.spaces_lost_to_block

    # The landed lap after any block-induced backward bounce: a block only ever
    # pushes the car *backward* within the same crossing, so the landed lap is
    # end_lap unless the bounce crossed back over the start line. Recompute the
    # landed progress from the realised landing space for the positional terms.
    landed_lap = end_lap
    if effect.spaces_lost_to_block > 0 and effect.landed_pos > new_pos:
        # Bounced back across the start line: drop a lap.
        landed_lap = end_lap - 1

    value = OWN_PROGRESS_WEIGHT * spaces_gained
    value -= heat_price * heat_spent
    value -= p_spinout * spin_loss

    if enable_solvency:
        heat_after = heat_available - total_heat_needed
        value += project_solvency(
            state,
            player,
            effect.landed_pos,
            landed_lap,
            heat_after,
            planned_gear,
            expected_basic_value(player) * planned_gear,
            horizon_corners,
            heat_price,
        )

    # Relative finishing-order objective (strength >= 2, P0): the saturating
    # gap-change against the nearest rivals. Decoupled from own-progress so it
    # changes the argmax in contested situations (not just a tiebreak).
    rel_value = 0.0
    if opponent_aware:
        prog_self_before = race_progress(player, track)
        prog_self_after = landed_lap * track.length + effect.landed_pos
        rel_value = relative_term(
            state, player_id, prog_self_before, prog_self_after
        )
        value += RELATIVE_WEIGHT * rel_value

    if blocking:
        # Real block projection (P3), already in the relative currency.
        value += block_value(state, player_id, effect.landed_pos, landed_lap)

    return MoveEval(
        value=value,
        expected_speed=expected_speed,
        heat_spent=heat_spent,
        corner_cost=corner_cost,
        p_spinout=p_spinout,
        relative_value=rel_value,
    )
