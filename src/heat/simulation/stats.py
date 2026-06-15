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

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from heat.simulation.runner import GameOutcome, PlayerOutcome

_VALID_GROUPINGS = ("agent_type", "player_id")


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
