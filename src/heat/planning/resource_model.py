"""Stochastic resource model -- ``P(S | gear, intent)`` from a deck composition.

The heart of Option B's abstraction (README B-design 3.2). The reduced DP does
*not* enumerate the 7-card hand / draw-pile order; instead it treats the realized
per-turn speed ``S`` as a random variable whose distribution depends only on the
current ``gear`` (how many cards are committed), an *intent* (whether the driver
plays its low cards to conserve heat or its high cards to push speed), and the
player's owned **Basic (SPEED) card value multiset** -- which is public-to-self
and order-free (``_move_eval.expected_basic_value`` already reasons this way).

Three fidelities are offered, matching the spike's go/no-go ladder:

* :meth:`expected_speed` -- collapse ``S`` to a single mean (cheapest).
* :meth:`speed_bands` -- a small ``[(speed, prob), ...]`` discretization the DP
  can reason about an above-mean draw with (the variance that causes limit-1
  spins).
* :meth:`speed_distribution` -- the full ``{speed: prob}`` over the ``gear``-card
  draw, conditioned on intent via order statistics.

**What is folded in.** A *stress* card flips to a random owned Basic, so in the
deck's value multiset we represent each stress card by the mean owned-Basic value
(``expected_basic_value``); a *heat* card played (forced, cluttered) contributes
0. Boost / adrenaline are added at their expected value as a flat per-turn
offset (``expected_extra``) -- B0 does not model the boost *decision*, only its
mean contribution, exactly as ``_move_eval.play_speed`` does.

**Intent via order statistics.** Because the driver *chooses* which ``gear`` of
its hand to play, the realized speed is not a blind ``gear``-card sum: a careful
driver plays its lowest cards into a tight corner (``conserve``) and its highest
when it wants speed (``push``). We model this conditional by drawing ``gear``
cards from the owned multiset and selecting, per intent:

* ``conserve`` -> the driver holds high cards back: take the *gear lowest* of a
  larger candidate draw (lower order statistics).
* ``push``     -> take the *gear highest* (upper order statistics).
* ``mean``     -> a plain ``gear``-card draw (the centre).

This is the single most important fidelity knob (README 3.2) and the spike
measures whether it is faithful at limit-1.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

from heat.agents import _move_eval as ME
from heat.models.cards import CardType
from heat.models.player_state import PlayerState
from heat.engine import rules


#: Intents the model conditions on. ``conserve`` biases the play toward low
#: cards (heat-safe into tight corners), ``push`` toward high cards (speed),
#: ``mean`` is the unconditioned centre.
INTENTS: tuple[str, ...] = ("conserve", "mean", "push")

#: How many *extra* candidate cards beyond ``gear`` the conserve/push order
#: statistics are taken from. The driver does not see the whole deck each turn --
#: it holds a 7-card hand -- so it can shade its play toward low/high but cannot
#: always realize the global min/max. We model the realized hand as a window of
#: ``gear + _INTENT_SPREAD`` candidate draws and take the gear lowest (conserve)
#: or gear highest (push). Larger => stronger conditioning (closer to the deck
#: extremes); ``0`` collapses every intent to ``mean``.
_INTENT_SPREAD: int = 2


@dataclass
class SpeedResourceModel:
    """``P(S | gear, intent)`` derived from a player's owned Basic-card multiset.

    Build with :meth:`from_player`. The model is a pure function of the value
    multiset + an expected per-turn extra (boost/adrenaline EV); it holds no
    live-hand or draw-order state, so it is stable across a turn (the whole point
    of the abstraction).

    Attributes:
        basic_values: The owned Basic (SPEED) card values, with each *stress*
            card folded in as the mean owned-Basic value (its expected flip).
            This is the population the per-turn draw samples from.
        expected_extra: Mean extra speed added per turn from boost / adrenaline,
            applied as a flat offset to every speed in the distribution. ``0.0``
            by default (B0's transition does not exercise boost), surfaced so the
            spike can fold a controller's expected boost in if needed.
    """

    basic_values: tuple[float, ...]
    expected_extra: float = 0.0
    # Cache of the integer multiset used for combinatorial draws (built once).
    _int_pool: tuple[int, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        # Round the (possibly fractional, due to stress-as-mean) values to the
        # nearest integer for the combinatorial draw: realized card values are
        # integers, and the stress mean is a small bias we apply by rounding so
        # the band speeds stay integer-valued (the engine moves integer spaces).
        self._int_pool = tuple(sorted(int(round(v)) for v in self.basic_values))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_player(
        cls, player: PlayerState, *, expected_extra: float = 0.0
    ) -> "SpeedResourceModel":
        """Build the model from a live ``PlayerState`` (deck + hand contents).

        Collects every owned Basic (SPEED) card value across deck and hand (the
        order-free composition ``_move_eval.expected_basic_value`` uses), then
        folds each owned *stress* card in as one extra population member at the
        mean owned-Basic value (its expected flip). Upgrade/heat cards do not
        enter the speed population (heat plays as 0; upgrades are not Basic and
        are flipped past during stress resolution).
        """
        speed_values = basic_value_multiset(player)
        if not speed_values:
            # Degenerate: no owned Basics. Fall back to the standard 1..4 mean so
            # the model is still defined (mirrors expected_basic_value's fallback).
            speed_values = [int(round(ME._FALLBACK_BASIC_MEAN))]

        # Fold stress cards in at the expected owned-Basic value (their flip EV).
        stress_count = sum(
            1
            for c in list(player.deck) + list(player.hand)
            if c.card_type == CardType.STRESS
        )
        stress_ev = ME.expected_basic_value(player)
        population = list(speed_values) + [stress_ev] * stress_count
        return cls(basic_values=tuple(population), expected_extra=expected_extra)

    # ------------------------------------------------------------------
    # Core distribution
    # ------------------------------------------------------------------

    def speed_distribution(self, gear: int, intent: str) -> dict[int, float]:
        """``{realized_speed: probability}`` for a turn at ``gear`` under ``intent``.

        The driver commits ``gear`` cards. We enumerate every ``gear``-card draw
        (without replacement) from the owned Basic multiset, then condition on
        intent via order statistics over a slightly larger candidate window
        (:data:`_INTENT_SPREAD`):

        * ``mean``     -- the plain ``gear``-card draw sum.
        * ``conserve`` -- the *gear lowest* cards of a ``gear + spread`` draw
          (the driver shades low), summed.
        * ``push``     -- the *gear highest* cards of a ``gear + spread`` draw.

        The flat :attr:`expected_extra` (boost/adrenaline EV) is added to every
        outcome. Probabilities are uniform over the equally-likely draws and are
        normalized to sum to 1.

        Returns a dict so the DP can marginalize or sample. ``gear <= 0`` (a
        cluttered no-move turn is modeled elsewhere) returns ``{extra: 1.0}``.
        """
        if intent not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}, got {intent!r}")
        extra = int(round(self.expected_extra))
        if gear <= 0:
            return {extra: 1.0}

        dist = _speed_distribution_cached(
            self._int_pool, gear, intent, _INTENT_SPREAD
        )
        if extra == 0:
            return dict(dist)
        return {speed + extra: prob for speed, prob in dist}

    def expected_speed(self, gear: int, intent: str) -> float:
        """Mean realized speed ``E[S | gear, intent]`` (the cheapest fidelity)."""
        dist = self.speed_distribution(gear, intent)
        return sum(speed * prob for speed, prob in dist.items())

    def speed_bands(
        self, gear: int, intent: str, *, max_bands: int | None = None
    ) -> list[tuple[int, float]]:
        """A few-moment ``[(speed, prob), ...]`` discretization (the DP's input).

        Returns the full support sorted by speed when ``max_bands`` is None (the
        distribution is already small for the 1..4 deck). When ``max_bands`` is
        given, collapses the distribution to that many representative bands by
        merging the tails into the nearest kept quantile -- a "few-moment" view
        the DP can iterate over cheaply. Probabilities always sum to 1.
        """
        dist = self.speed_distribution(gear, intent)
        items = sorted(dist.items())
        if max_bands is None or len(items) <= max_bands:
            return items
        return _collapse_to_bands(items, max_bands)


# ---------------------------------------------------------------------------
# Composition helper (public per the B0 deliverable list)
# ---------------------------------------------------------------------------


def basic_value_multiset(player: PlayerState) -> list[int]:
    """Owned Basic (SPEED) card values across the player's deck and hand.

    The order-free speed population the resource model samples from (the same
    set ``_move_eval.expected_basic_value`` averages). Heat/Stress/Upgrade cards
    are excluded -- stress is folded in separately (as its flip EV) by
    :meth:`SpeedResourceModel.from_player`, and heat/upgrade contribute 0 / are
    flipped past.
    """
    values: list[int] = []
    for card in list(player.deck) + list(player.hand):
        if card.card_type == CardType.SPEED:
            values.append(card.value)
    return values


# ---------------------------------------------------------------------------
# Cached combinatorial core (pure function of the integer pool + params)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _speed_distribution_cached(
    pool: tuple[int, ...], gear: int, intent: str, spread: int
) -> tuple[tuple[int, float], ...]:
    """Cached ``((speed, prob), ...)`` for an integer card pool.

    Pure (hashable args, no state) so it caches across the many snapshots the
    spike evaluates (decks are largely identical). Returns a sorted tuple of
    ``(speed, prob)`` pairs; the public method copies it into a dict.
    """
    n = len(pool)
    if n == 0:
        return ((0, 1.0),)

    # The candidate window: conserve/push draw `gear + spread` then keep the
    # gear lowest/highest; mean draws exactly `gear`. Clamp the window to the
    # pool size so we never request more cards than exist.
    if intent == "mean":
        draw = min(gear, n)
        counts: Counter[int] = Counter()
        for combo in combinations(pool, draw):
            counts[sum(combo)] += 1
        total = sum(counts.values())
        return tuple(sorted((s, c / total) for s, c in counts.items()))

    window = min(gear + spread, n)
    take = min(gear, window)
    counts = Counter()
    for combo in combinations(pool, window):
        ordered = sorted(combo)
        if intent == "conserve":
            chosen = ordered[:take]
        else:  # push
            chosen = ordered[window - take:]
        counts[sum(chosen)] += 1
    total = sum(counts.values())
    return tuple(sorted((s, c / total) for s, c in counts.items()))


def _collapse_to_bands(
    items: list[tuple[int, float]], max_bands: int
) -> list[tuple[int, float]]:
    """Collapse a sorted ``(speed, prob)`` list to at most ``max_bands`` bands.

    Greedy equal-mass bucketing: walk the sorted support accumulating mass until
    a bucket holds ~``1/max_bands`` of the total, emit the probability-weighted
    rounded speed for that bucket, and continue. Preserves the mean closely while
    giving the DP a compact support. Probabilities sum to 1.
    """
    target = 1.0 / max_bands
    bands: list[tuple[int, float]] = []
    acc_mass = 0.0
    acc_weighted = 0.0
    for speed, prob in items:
        acc_mass += prob
        acc_weighted += speed * prob
        if acc_mass >= target and len(bands) < max_bands - 1:
            bands.append((int(round(acc_weighted / acc_mass)), acc_mass))
            acc_mass = 0.0
            acc_weighted = 0.0
    if acc_mass > 0:
        bands.append((int(round(acc_weighted / acc_mass)), acc_mass))
    # Renormalize (rounding of the representative speed does not touch mass, but
    # guard against float drift).
    total = sum(p for _, p in bands)
    return [(s, p / total) for s, p in bands]
