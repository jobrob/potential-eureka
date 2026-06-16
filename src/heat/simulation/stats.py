"""Aggregate statistics over a batch of HEAT game outcomes.

Pure aggregation: this module takes a ``list[GameOutcome]`` (as produced by
:func:`heat.simulation.runner.run_batch`) and computes summary statistics. It
performs no engine work and no I/O, so it is trivially testable on hand-built
outcomes.

Grouping is controlled by the ``by`` parameter:
  * ``by="agent_type"`` (default) groups player-games by their agent class
    name (e.g. "HeuristicAgent" vs "RandomAgent"), aggregating across seats.
    This is the grouping used for headline "Heuristic vs Random" comparisons.
  * ``by="player_id"`` groups by seat index.

Heat-efficiency note: the engine exposes no cumulative "heat spent" counter, so
heat efficiency is reported here as ``avg_heat_remaining`` -- the mean of each
group's ``heat_remaining`` (heat left in the pool at game end). This is a proxy,
not a spend figure. If a future sprint adds a real counter, this is the single
place to extend.

Usage:
    >>> from heat.simulation.stats import aggregate_stats, format_summary
    >>> stats = aggregate_stats(outcomes, by="agent_type")
    >>> print(format_summary(stats))
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from heat.simulation.runner import GameOutcome, PlayerOutcome

_VALID_GROUPINGS = ("agent_type", "player_id")

# Standard ELO constants (surfaced as parameters; these are the fixed defaults
# so that results are reproducible).
_DEFAULT_ELO_START = 1500.0
_DEFAULT_ELO_K = 24.0

# Minimal-TrueSkill constants (single-team free-for-all). These mirror the
# canonical TrueSkill defaults closely enough for ranking purposes while keeping
# the implementation dependency-free and deterministic.
_TS_MU = 25.0
_TS_SIGMA = _TS_MU / 3.0  # 8.333...
_TS_BETA = _TS_SIGMA / 2.0  # skill-to-performance noise
_TS_TAU = _TS_SIGMA / 100.0  # dynamics / additive variance per game


# ----------------------------------------------------------------------
# Result data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AgentStats:
    """Aggregated statistics for one grouping key (agent type or seat).

    Attributes:
        agent_type: The group label (agent class name, or ``"player_<id>"``).
        games_played: Number of player-games in this group.
        wins: Number of those games won.
        win_rate: ``wins / games_played``.
        finish_position_counts: Map of 1-based rank -> count.
        avg_finish_position: Mean finishing rank.
        avg_heat_remaining: Mean heat left in pool at game end (efficiency proxy).
    """

    agent_type: str
    games_played: int
    wins: int
    win_rate: float
    finish_position_counts: dict[int, int]
    avg_finish_position: float
    avg_heat_remaining: float


@dataclass(frozen=True)
class SimulationStats:
    """Top-level summary over a batch of games.

    Attributes:
        num_games: Number of games in the batch.
        num_players: Players per game (from the first outcome).
        avg_rounds: Mean rounds across games.
        min_rounds: Fewest rounds in any game.
        max_rounds: Most rounds in any game.
        per_agent: Map of group label -> :class:`AgentStats`.
        win_counts_by_player_id: Wins keyed by seat index.
        finish_distribution_by_player_id: Per-seat map of rank -> count.
    """

    num_games: int
    num_players: int
    avg_rounds: float
    min_rounds: int
    max_rounds: int
    per_agent: dict[str, AgentStats]
    win_counts_by_player_id: dict[int, int]
    finish_distribution_by_player_id: dict[int, dict[int, int]]


# ----------------------------------------------------------------------
# Sprint 6B result data structures (evaluation overhaul)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HeadToHeadStats:
    """Two-player, head-to-head result of agent A vs agent B.

    Attributes:
        agent_a: Label for the first agent.
        agent_b: Label for the second agent.
        games: Total decisive games counted.
        a_wins: Games won by A.
        b_wins: Games won by B.
        a_win_rate: ``a_wins / games`` (B's rate is ``1 - a_win_rate``).
        a_win_rate_ci: Wilson 95% interval on ``a_win_rate``.
    """

    agent_a: str
    agent_b: str
    games: int
    a_wins: int
    b_wins: int
    a_win_rate: float
    a_win_rate_ci: tuple[float, float]


@dataclass(frozen=True)
class EloRating:
    """An ELO rating for one agent, with an uncertainty interval.

    Attributes:
        agent_type: The agent label.
        rating: Final ELO rating.
        games: Number of pairwise comparisons that fed this rating.
        rating_ci: Bootstrap 95% interval on the ELO rating.
    """

    agent_type: str
    rating: float
    games: int
    rating_ci: tuple[float, float]


@dataclass(frozen=True)
class TrueSkillRating:
    """A TrueSkill skill posterior for one agent.

    The conservative skill estimate is ``mu - 3 * sigma``; ``sigma`` itself is
    the (native) uncertainty of the estimate.

    Attributes:
        agent_type: The agent label.
        mu: Mean of the skill posterior.
        sigma: Standard deviation (uncertainty) of the skill posterior.
        games: Number of games that fed this rating.
    """

    agent_type: str
    mu: float
    sigma: float
    games: int

    @property
    def conservative(self) -> float:
        """Conservative skill estimate ``mu - 3*sigma``."""
        return self.mu - 3.0 * self.sigma


@dataclass(frozen=True)
class RoundRobinStats:
    """Aggregated round-robin scoreboard across a pool of agents.

    Attributes:
        ratings: ``agent_type -> EloRating`` (always populated).
        trueskill: ``agent_type -> TrueSkillRating`` (only when TrueSkill ran).
        pairwise: ``(agent_a, agent_b) -> HeadToHeadStats`` for each ordered
            pair with ``agent_a < agent_b`` (a single canonical entry per pair).
        per_track: ``track_name -> {agent_type: win_rate}`` when a cross-track
            sweep was run, else ``None``.
    """

    ratings: dict[str, EloRating]
    trueskill: dict[str, TrueSkillRating] | None
    pairwise: dict[tuple[str, str], HeadToHeadStats]
    per_track: dict[str, dict[str, float]] | None


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _group_key(po: PlayerOutcome, by: str) -> str:
    """Return the grouping label for a player-outcome under ``by``."""
    if by == "agent_type":
        return po.agent_type
    return f"player_{po.player_id}"


def _tally_finishes(positions: Sequence[int]) -> dict[int, int]:
    """Count occurrences of each finishing rank."""
    counts: dict[int, int] = defaultdict(int)
    for pos in positions:
        counts[pos] += 1
    return dict(counts)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def aggregate_stats(
    outcomes: Sequence[GameOutcome],
    *,
    by: str = "agent_type",
) -> SimulationStats:
    """Aggregate a batch of game outcomes into a :class:`SimulationStats`.

    Args:
        outcomes: Non-empty sequence of :class:`GameOutcome`.
        by: Grouping key, ``"agent_type"`` (default) or ``"player_id"``.

    Returns:
        A :class:`SimulationStats` summarising rounds, per-group win rates,
        finish distributions, and average heat remaining.

    Raises:
        ValueError: if ``outcomes`` is empty or ``by`` is invalid.
    """
    if not outcomes:
        raise ValueError("Cannot aggregate stats over an empty outcome list")
    if by not in _VALID_GROUPINGS:
        raise ValueError(
            f"Invalid grouping {by!r}; expected one of {_VALID_GROUPINGS}"
        )

    num_games = len(outcomes)
    num_players = outcomes[0].num_players

    rounds = [o.total_rounds for o in outcomes]
    avg_rounds = _mean(rounds)
    min_rounds = min(rounds)
    max_rounds = max(rounds)

    # Accumulators keyed by group label.
    group_positions: dict[str, list[int]] = defaultdict(list)
    group_wins: dict[str, int] = defaultdict(int)
    group_heat: dict[str, list[int]] = defaultdict(list)

    # Per-seat accumulators (always tracked, independent of `by`).
    win_counts_by_player_id: dict[int, int] = defaultdict(int)
    finish_by_player_id: dict[int, list[int]] = defaultdict(list)

    for outcome in outcomes:
        for po in outcome.players:
            key = _group_key(po, by)
            group_positions[key].append(po.finish_position)
            group_heat[key].append(po.heat_remaining)
            if po.player_id == outcome.winner_id:
                group_wins[key] += 1

            finish_by_player_id[po.player_id].append(po.finish_position)
            if po.player_id == outcome.winner_id:
                win_counts_by_player_id[po.player_id] += 1

    per_agent: dict[str, AgentStats] = {}
    for key, positions in group_positions.items():
        games_played = len(positions)
        wins = group_wins.get(key, 0)
        per_agent[key] = AgentStats(
            agent_type=key,
            games_played=games_played,
            wins=wins,
            win_rate=wins / games_played if games_played else 0.0,
            finish_position_counts=_tally_finishes(positions),
            avg_finish_position=_mean(positions),
            avg_heat_remaining=_mean(group_heat[key]),
        )

    finish_distribution_by_player_id = {
        pid: _tally_finishes(positions)
        for pid, positions in finish_by_player_id.items()
    }

    return SimulationStats(
        num_games=num_games,
        num_players=num_players,
        avg_rounds=avg_rounds,
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        per_agent=per_agent,
        win_counts_by_player_id=dict(win_counts_by_player_id),
        finish_distribution_by_player_id=finish_distribution_by_player_id,
    )


# ----------------------------------------------------------------------
# Sprint 6B: confidence intervals (pure functions of counts)
# ----------------------------------------------------------------------

# z for a two-sided 95% interval (standard normal quantile at 0.975).
_Z_95 = 1.959963984540054


def wilson_interval(
    wins: int, games: int, *, z: float = _Z_95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation near 0/1 and for small ``games``.
    Returns ``(lo, hi)`` clamped to ``[0, 1]``. For ``games == 0`` returns the
    whole interval ``(0.0, 1.0)`` (maximal uncertainty, never NaN).

    Pure function of ``(wins, games)`` -- keeps the stats module's
    pure-aggregation contract.
    """
    if games <= 0:
        return (0.0, 1.0)
    if wins < 0 or wins > games:
        raise ValueError(f"wins ({wins}) must be in [0, games={games}]")

    n = float(games)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    lo = center - margin
    hi = center + margin
    return (max(0.0, lo), min(1.0, hi))


def intervals_overlap(
    ci_a: tuple[float, float], ci_b: tuple[float, float]
) -> bool:
    """True if two closed intervals overlap (touching counts as overlap).

    Used for the significance flag: two agents whose rating/win-rate CIs overlap
    are reported as *not significantly different*.
    """
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return lo_a <= hi_b and lo_b <= hi_a


# ----------------------------------------------------------------------
# Sprint 6B: ELO over GameOutcomes (pure aggregation, deterministic)
# ----------------------------------------------------------------------


def _seat_label(po: PlayerOutcome, by: str) -> str:
    """Group label for a player-outcome (mirrors ``_group_key``)."""
    if by == "agent_type":
        return po.agent_type
    return f"player_{po.player_id}"


def _pairwise_results_from_outcomes(
    outcomes: Sequence[GameOutcome], by: str
) -> list[tuple[str, str]]:
    """Decompose each game's ``finish_order`` into ``(winner, loser)`` pairs.

    For each game, the player at finish rank ``i`` beats every player at a worse
    rank ``j > i``. Games are processed in ``game_index`` order so the resulting
    list -- and therefore any ELO/TrueSkill pass over it -- is deterministic and
    independent of the input ordering.

    Player ids are mapped to their group label via ``by`` (so an ELO pass over
    ``agent_type`` aggregates seats of the same agent class).
    """
    ordered = sorted(outcomes, key=lambda o: o.game_index)
    pairs: list[tuple[str, str]] = []
    for outcome in ordered:
        label_by_pid = {
            po.player_id: _seat_label(po, by) for po in outcome.players
        }
        finish = outcome.finish_order
        for i in range(len(finish)):
            winner = label_by_pid[finish[i]]
            for j in range(i + 1, len(finish)):
                loser = label_by_pid[finish[j]]
                pairs.append((winner, loser))
    return pairs


def _elo_from_pairs(
    pairs: Sequence[tuple[str, str]],
    labels: Sequence[str],
    *,
    start: float,
    k: float,
) -> dict[str, float]:
    """Run sequential ELO updates over an ordered list of ``(winner, loser)``.

    Pure: a fixed ``pairs`` order yields identical ratings.
    """
    ratings = {label: start for label in labels}
    for winner, loser in pairs:
        rw = ratings[winner]
        rl = ratings[loser]
        expected_w = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        ratings[winner] = rw + k * (1.0 - expected_w)
        ratings[loser] = rl + k * (expected_w - 1.0)
    return ratings


def compute_elo(
    outcomes: Sequence[GameOutcome],
    *,
    by: str = "agent_type",
    start: float = _DEFAULT_ELO_START,
    k: float = _DEFAULT_ELO_K,
    bootstrap: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, EloRating]:
    """Compute ELO ratings (with bootstrap CIs) from a batch of outcomes.

    ELO is derived by replaying each game's ``finish_order`` as a sequence of
    pairwise wins (winner beats each lower finisher) and applying the standard
    ELO update with a fixed ``k``. Because the outcomes are sorted by
    ``game_index`` first, the result is deterministic and independent of the
    input order.

    The CI is a seeded percentile bootstrap: resample the ``game_index``-sorted
    outcome list with replacement ``bootstrap`` times, recompute ELO each time,
    and take the 2.5/97.5 percentiles. With ``bootstrap == 0`` the CI degenerates
    to ``(rating, rating)``.

    Pure function of the inputs (no I/O); deterministic given ``bootstrap_seed``.
    """
    if by not in _VALID_GROUPINGS:
        raise ValueError(
            f"Invalid grouping {by!r}; expected one of {_VALID_GROUPINGS}"
        )
    if not outcomes:
        return {}

    ordered = sorted(outcomes, key=lambda o: o.game_index)
    labels = sorted(
        {_seat_label(po, by) for o in ordered for po in o.players}
    )
    games_per_label = _count_games_per_label(ordered, by)

    point_pairs = _pairwise_results_from_outcomes(ordered, by)
    point = _elo_from_pairs(point_pairs, labels, start=start, k=k)

    # Bootstrap: resample whole games (preserving the per-game pairwise block
    # structure), recompute ELO, collect rating samples per label.
    samples: dict[str, list[float]] = {label: [] for label in labels}
    if bootstrap > 0:
        rng = random.Random(bootstrap_seed)
        n = len(ordered)
        for _ in range(bootstrap):
            resampled = [ordered[rng.randrange(n)] for _ in range(n)]
            pairs = _pairwise_results_from_outcomes(resampled, by)
            r = _elo_from_pairs(pairs, labels, start=start, k=k)
            for label in labels:
                samples[label].append(r[label])

    result: dict[str, EloRating] = {}
    for label in labels:
        if bootstrap > 0:
            lo, hi = _percentile_ci(samples[label])
        else:
            lo = hi = point[label]
        result[label] = EloRating(
            agent_type=label,
            rating=point[label],
            games=games_per_label.get(label, 0),
            rating_ci=(lo, hi),
        )
    return result


def _count_games_per_label(
    outcomes: Sequence[GameOutcome], by: str
) -> dict[str, int]:
    """Count player-games per group label."""
    counts: dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        for po in outcome.players:
            counts[_seat_label(po, by)] += 1
    return dict(counts)


def _percentile_ci(
    samples: Sequence[float], *, lo_pct: float = 2.5, hi_pct: float = 97.5
) -> tuple[float, float]:
    """Percentile interval over a sample list (linear interpolation)."""
    if not samples:
        return (0.0, 0.0)
    ordered = sorted(samples)
    return (_percentile(ordered, lo_pct), _percentile(ordered, hi_pct))


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence."""
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo_idx = int(math.floor(rank))
    hi_idx = int(math.ceil(rank))
    if lo_idx == hi_idx:
        return ordered[lo_idx]
    frac = rank - lo_idx
    return ordered[lo_idx] * (1.0 - frac) + ordered[hi_idx] * frac


# ----------------------------------------------------------------------
# Sprint 6B: minimal in-repo TrueSkill (dependency-free, deterministic)
# ----------------------------------------------------------------------


def compute_trueskill(
    outcomes: Sequence[GameOutcome],
    *,
    by: str = "agent_type",
    mu: float = _TS_MU,
    sigma: float = _TS_SIGMA,
    beta: float = _TS_BETA,
    tau: float = _TS_TAU,
) -> dict[str, TrueSkillRating]:
    """Compute TrueSkill-style ratings from a batch of outcomes.

    A minimal, dependency-free single-player-per-team TrueSkill: each game's
    full ``finish_order`` is consumed as a ranked outcome and decomposed into
    adjacent ranked pairs, each updated with the standard Gaussian
    win-probability update. ``sigma`` is the native uncertainty and shrinks with
    more games.

    Deterministic: outcomes are sorted by ``game_index`` first, so the result is
    independent of the input order. Pure (no I/O).
    """
    if by not in _VALID_GROUPINGS:
        raise ValueError(
            f"Invalid grouping {by!r}; expected one of {_VALID_GROUPINGS}"
        )
    if not outcomes:
        return {}

    ordered = sorted(outcomes, key=lambda o: o.game_index)
    labels = sorted({_seat_label(po, by) for o in ordered for po in o.players})
    mus = {label: mu for label in labels}
    sigmas = {label: sigma for label in labels}
    games_per_label = _count_games_per_label(ordered, by)

    for outcome in ordered:
        label_by_pid = {
            po.player_id: _seat_label(po, by) for po in outcome.players
        }
        finish = outcome.finish_order
        # Decompose the N-way ranking into adjacent winner/loser pairs.
        for i in range(len(finish) - 1):
            winner = label_by_pid[finish[i]]
            loser = label_by_pid[finish[i + 1]]
            _ts_update_pair(winner, loser, mus, sigmas, beta=beta, tau=tau)

    return {
        label: TrueSkillRating(
            agent_type=label,
            mu=mus[label],
            sigma=sigmas[label],
            games=games_per_label.get(label, 0),
        )
        for label in labels
    }


def _ts_update_pair(
    winner: str,
    loser: str,
    mus: dict[str, float],
    sigmas: dict[str, float],
    *,
    beta: float,
    tau: float,
) -> None:
    """Apply a single two-player TrueSkill update (winner ranked above loser)."""
    if winner == loser:
        return

    # Add dynamics noise so sigma cannot collapse to zero (keeps it responsive).
    sw = math.sqrt(sigmas[winner] ** 2 + tau ** 2)
    sl = math.sqrt(sigmas[loser] ** 2 + tau ** 2)

    c = math.sqrt(2.0 * beta ** 2 + sw ** 2 + sl ** 2)
    t = (mus[winner] - mus[loser]) / c
    v = _ts_v(t)
    w = v * (v + t)

    mus[winner] = mus[winner] + (sw ** 2 / c) * v
    mus[loser] = mus[loser] - (sl ** 2 / c) * v
    sigmas[winner] = math.sqrt(sw ** 2 * (1.0 - (sw ** 2 / c ** 2) * w))
    sigmas[loser] = math.sqrt(sl ** 2 * (1.0 - (sl ** 2 / c ** 2) * w))


def _ts_v(t: float) -> float:
    """v(t) = pdf(t) / cdf(t): the mean of a truncated standard Gaussian.

    Guarded against the cdf underflowing to 0 for very negative ``t`` (a large
    upset), returning ``-t`` in the limit so the update stays finite.
    """
    cdf = _std_normal_cdf(t)
    if cdf < 1e-12:
        return -t
    pdf = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    return pdf / cdf


def _std_normal_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def format_summary(stats: SimulationStats) -> str:
    """Render a :class:`SimulationStats` as a human-readable table.

    Returns a multi-line string; does not print (the caller prints).
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("HEAT Simulation Summary")
    lines.append("=" * 60)
    lines.append(f"Games:   {stats.num_games}")
    lines.append(f"Players: {stats.num_players}")
    lines.append(
        f"Rounds:  avg={stats.avg_rounds:.2f}  "
        f"min={stats.min_rounds}  max={stats.max_rounds}"
    )
    lines.append("-" * 60)
    lines.append(
        f"{'Group':<18}{'Games':>7}{'Wins':>7}{'Win%':>8}"
        f"{'AvgPos':>8}{'AvgHeat':>9}"
    )
    lines.append("-" * 60)

    # Sort groups by win rate descending for a readable headline ordering.
    for key in sorted(
        stats.per_agent,
        key=lambda k: stats.per_agent[k].win_rate,
        reverse=True,
    ):
        s = stats.per_agent[key]
        lines.append(
            f"{s.agent_type:<18}{s.games_played:>7}{s.wins:>7}"
            f"{s.win_rate * 100:>7.1f}%"
            f"{s.avg_finish_position:>8.2f}{s.avg_heat_remaining:>9.2f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def format_round_robin(rr: RoundRobinStats) -> str:
    """Render a :class:`RoundRobinStats` scoreboard, flagging non-significance.

    Agents are listed best-to-worst by ELO. Each rating shows its bootstrap CI.
    A trailing note flags any adjacent pair whose ELO CIs overlap as *not
    significantly different* (the ordering there is within sampling noise).
    """
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("HEAT Round-Robin Scoreboard")
    lines.append("=" * 64)

    ranked = sorted(
        rr.ratings.values(), key=lambda r: r.rating, reverse=True
    )
    lines.append(f"{'Agent':<22}{'ELO':>8}{'95% CI':>20}{'Games':>8}")
    lines.append("-" * 64)
    for r in ranked:
        ci = f"[{r.rating_ci[0]:.0f}, {r.rating_ci[1]:.0f}]"
        lines.append(
            f"{r.agent_type:<22}{r.rating:>8.0f}{ci:>20}{r.games:>8}"
        )

    if rr.trueskill:
        lines.append("-" * 64)
        lines.append(f"{'Agent':<22}{'mu':>8}{'sigma':>8}{'mu-3sig':>10}")
        for label in [r.agent_type for r in ranked]:
            ts = rr.trueskill.get(label)
            if ts is None:
                continue
            lines.append(
                f"{ts.agent_type:<22}{ts.mu:>8.2f}{ts.sigma:>8.2f}"
                f"{ts.conservative:>10.2f}"
            )

    # Significance notes on adjacent ELO pairs.
    notes: list[str] = []
    for higher, lower in zip(ranked, ranked[1:]):
        if intervals_overlap(higher.rating_ci, lower.rating_ci):
            notes.append(
                f"  {higher.agent_type} vs {lower.agent_type}: "
                f"NOT significant (CIs overlap)"
            )
    if notes:
        lines.append("-" * 64)
        lines.append("Significance (adjacent pairs):")
        lines.extend(notes)

    lines.append("=" * 64)
    return "\n".join(lines)
