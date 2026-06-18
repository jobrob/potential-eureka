"""Win-rate evaluation for a trained HEAT RL agent (Sprint 5d).

Reuses the existing simulation/stats layer rather than rebuilding it:

* :func:`ml_agent_factory` mirrors ``random_agent_factory`` /
  ``heuristic_agent_factory`` (``runner.py``) so an :class:`~heat.agents.ml_agent.MLAgent`
  slots into :func:`heat.simulation.runner.run_batch` unchanged.
* :func:`evaluate_ml` runs a batch of the trained agent against an opponent
  factory and aggregates with :func:`heat.simulation.stats.aggregate_stats`.

Picklability / parallelism (§6.5)
---------------------------------
``run_batch(parallel=True)`` pickles the agent *factories* to worker processes
(Windows ``spawn``). An SB3 model is heavy and awkward to pickle, so the factory
carries only the **model path** (a string); :class:`MLAgent` lazy-loads the
model inside the worker and nulls it in ``__getstate__``. The factory itself is
a :func:`functools.partial` of a top-level constructor (NOT a lambda/closure),
so it *is* picklable.

Even so, :func:`evaluate_ml` defaults to ``parallel=False``: each worker would
reload the full SB3 model (and torch) per process, and HEAT eval batches are
modest, so the process-pool overhead and per-worker model-load cost typically
outweigh the speedup. Parallel remains available for large batches via
``parallel=True``.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from typing import Mapping

from heat.agents.ml_agent import MLAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine.game import Agent
from heat.models.track import Track
from heat.simulation.runner import (
    AgentFactory,
    GameOutcome,
    heuristic_agent_factory,
    run_batch,
)
from heat.simulation.stats import (
    AgentStats,
    EloRating,
    HeadToHeadStats,
    RoundRobinStats,
    TrueSkillRating,
    aggregate_stats,
    compute_elo,
    compute_trueskill,
    wilson_interval,
)
from heat.tracks.loader import load_track_by_name


def _make_ml_agent(
    player_id: int,
    seed: int | None,
    model_path: str,
    name: str | None,
    deterministic: bool,
) -> Agent:
    """Top-level (picklable) constructor for an :class:`MLAgent`.

    Ignores ``seed`` -- the policy is deterministic by default and the model is
    loaded from ``model_path`` inside the (possibly worker) process, never
    passed as a live object (§6.5).
    """
    agent_name = name if name is not None else f"MLAgent-{player_id}"
    return MLAgent(model_path, deterministic=deterministic, name=agent_name)


def ml_agent_factory(
    model_path: str,
    name: str | None = None,
    *,
    deterministic: bool = True,
) -> AgentFactory:
    """Return a picklable factory producing :class:`MLAgent`s from ``model_path``.

    Mirrors ``random_agent_factory`` / ``heuristic_agent_factory``. The returned
    callable is a :func:`functools.partial` of a top-level constructor, so it can
    pickle into ``ProcessPoolExecutor`` workers; it carries only the checkpoint
    *path*, never a live SB3 model (§6.5).
    """
    return functools.partial(
        _make_ml_agent,
        model_path=model_path,
        name=name,
        deterministic=deterministic,
    )


def _make_strong_heuristic(
    player_id: int,
    seed: int | None,
    name: str | None,
    strength: int,
) -> Agent:
    """Top-level (picklable) constructor for a :class:`StrongHeuristicAgent`.

    Mirrors :func:`_make_ml_agent` / ``runner._make_strong_heuristic``: a plain
    top-level function (no lambda/closure) so the factory pickles into
    ``ProcessPoolExecutor`` workers. The agent is deterministic given the state,
    so ``seed`` is ignored here (the gate / eval is order-independent).
    """
    agent_name = name if name is not None else f"StrongHeuristic-{player_id}"
    return StrongHeuristicAgent(name=agent_name, strength=strength)


def strong_heuristic_agent_factory(
    strength: int = 2,
    name: str | None = None,
) -> AgentFactory:
    """Return a picklable factory producing :class:`StrongHeuristicAgent`s.

    Mirrors :func:`heuristic_agent_factory`: the returned callable is a
    :func:`functools.partial` of a top-level constructor (never a lambda), so it
    pickles into ``run_batch(parallel=True)`` workers. ``strength`` defaults to
    ``2`` -- the same default rung :func:`heat.ml.training._scripted_opponents`
    instantiates the strong scripted pool at, so a gate built from this factory
    scores against the opponent the learner actually trains against (Idea 8).
    """
    return functools.partial(
        _make_strong_heuristic,
        strength=strength,
        name=name,
    )


def evaluate_ml(
    model_path: str,
    *,
    opponent_factory: AgentFactory | None = None,
    num_games: int = 100,
    num_players: int = 2,
    track: Track | str | None = None,
    seed: int | None = 0,
    parallel: bool = False,
    progress: bool = False,
    learner_seat: int = 0,
) -> dict[str, AgentStats]:
    """Evaluate a trained checkpoint vs an opponent over a batch of games.

    The :class:`MLAgent` sits at ``learner_seat`` (default seat 0); the remaining
    ``num_players - 1`` seats are built from ``opponent_factory`` (default:
    :func:`heuristic_agent_factory`).

    Args:
        model_path: Path to the SB3 checkpoint (``.zip``, with sidecar).
        opponent_factory: Picklable factory for the opponent seats. Defaults to
            ``heuristic_agent_factory()``.
        num_games: Number of games to run (>= 1).
        num_players: Seats per game (2..6).
        track: A :class:`Track`, a track name (e.g. ``"usa"``), or ``None`` to
            default to ``"usa"``.
        seed: Base seed for deterministic, order-independent runs.
        parallel: Run games across worker processes. Defaults to ``False`` (§6.5
            -- per-worker SB3 model reload usually outweighs the speedup).
        progress: Show a ``tqdm`` progress bar if installed.
        learner_seat: The seat the :class:`MLAgent` occupies (default 0). The
            other seats are filled by ``opponent_factory``. Sweeping this over
            ``range(num_players)`` measures whether the policy is seat-robust
            (Idea 10) rather than overfit to the front seat.

    Returns:
        The ``per_agent`` mapping from :func:`aggregate_stats` (grouped by
        ``agent_type``): ``{agent_type -> AgentStats}``. The headline
        ``"MLAgent"`` vs ``"HeuristicAgent"`` win rates live here.
    """
    if not (2 <= num_players <= 6):
        raise ValueError(f"num_players must be 2..6, got {num_players}")
    if not (0 <= learner_seat < num_players):
        raise ValueError(
            f"learner_seat {learner_seat} out of range for {num_players} players"
        )

    if opponent_factory is None:
        opponent_factory = heuristic_agent_factory()

    if track is None:
        track = load_track_by_name("usa")
    elif isinstance(track, str):
        track = load_track_by_name(track)

    factories: list[AgentFactory] = [opponent_factory] * num_players
    factories[learner_seat] = ml_agent_factory(model_path)

    outcomes: list[GameOutcome] = run_batch(
        track,
        factories,
        num_games=num_games,
        parallel=parallel,
        seed=seed,
        progress=progress,
    )

    stats = aggregate_stats(outcomes, by="agent_type")
    return stats.per_agent


# ----------------------------------------------------------------------
# Sprint 6B: richer evaluation harness (head-to-head, round-robin, x-track)
# ----------------------------------------------------------------------


def _resolve_track(track: Track | str | None) -> Track:
    """Resolve a Track / track-name / None (default ``"usa"``) to a Track."""
    if track is None:
        return load_track_by_name("usa")
    if isinstance(track, str):
        return load_track_by_name(track)
    return track


def _relabel_outcomes(
    outcomes: list[GameOutcome], seat_labels: Mapping[int, str]
) -> list[GameOutcome]:
    """Return outcomes with each seat's ``agent_type`` replaced by a pool label.

    ``seat_labels`` maps ``player_id -> pool label``. This lets ELO/TrueSkill
    (which group ``by="agent_type"``) aggregate by the *pool* identity rather
    than the agent's class name -- so two distinct pool entries that happen to
    share a class (e.g. two MLAgents from different paths) stay separate, and a
    single entry's games aggregate across pairs.
    """
    relabelled: list[GameOutcome] = []
    for o in outcomes:
        new_players = tuple(
            dataclasses.replace(po, agent_type=seat_labels[po.player_id])
            for po in o.players
        )
        relabelled.append(dataclasses.replace(o, players=new_players))
    return relabelled


def head_to_head(
    factory_a: AgentFactory,
    factory_b: AgentFactory,
    *,
    label_a: str = "A",
    label_b: str = "B",
    num_games: int = 200,
    track: Track | str | None = None,
    seed: int = 0,
    parallel: bool = False,
) -> HeadToHeadStats:
    """Two-player A-vs-B over a seeded batch, with seat-order cancellation.

    Runs half the games with A at seat 0 (B at seat 1) and half with the seats
    swapped, so any first-mover seat advantage cancels. Wins are attributed to A
    or B by seat, not by agent class -- so A and B may even be the same agent
    type.

    Args:
        factory_a, factory_b: Picklable seat factories for A and B.
        label_a, label_b: Labels recorded in the returned stats.
        num_games: Total games (split as evenly as possible across the two seat
            orders).
        track: Track / track-name / ``None`` (defaults to ``"usa"``).
        seed: Base seed (each seat order uses a derived seed for determinism).
        parallel: Passed through to ``run_batch``; default ``False`` because an
            MLAgent factory reloads its SB3 model per worker.

    Returns:
        A :class:`HeadToHeadStats` (A's win rate + Wilson 95% CI).
    """
    if num_games < 1:
        raise ValueError(f"num_games must be >= 1, got {num_games}")

    resolved = _resolve_track(track)
    half = num_games // 2
    remainder = num_games - half

    a_wins = 0
    b_wins = 0
    games = 0

    # Order 1: A at seat 0, B at seat 1.
    if remainder > 0:
        out1 = run_batch(
            resolved,
            [factory_a, factory_b],
            num_games=remainder,
            parallel=parallel,
            seed=seed,
        )
        for o in out1:
            games += 1
            if o.winner_id == 0:
                a_wins += 1
            else:
                b_wins += 1

    # Order 2: B at seat 0, A at seat 1 (different derived seed so the two
    # halves are independent draws, not mirror images of the same games).
    if half > 0:
        out2 = run_batch(
            resolved,
            [factory_b, factory_a],
            num_games=half,
            parallel=parallel,
            seed=seed + 1,
        )
        for o in out2:
            games += 1
            if o.winner_id == 1:
                a_wins += 1
            else:
                b_wins += 1

    a_win_rate = a_wins / games if games else 0.0
    return HeadToHeadStats(
        agent_a=label_a,
        agent_b=label_b,
        games=games,
        a_wins=a_wins,
        b_wins=b_wins,
        a_win_rate=a_win_rate,
        a_win_rate_ci=wilson_interval(a_wins, games),
    )


def _round_robin_outcomes(
    factories: Mapping[str, AgentFactory],
    *,
    num_games_per_pair: int,
    num_players: int,
    track: Track,
    seed: int,
    parallel: bool,
) -> tuple[list[GameOutcome], dict[tuple[str, str], HeadToHeadStats]]:
    """Run every unordered pair and return (relabelled outcomes, pairwise table).

    Each pair's games are relabelled by pool key and stamped with globally unique
    ``game_index`` values (a deterministic per-pair offset) so the combined list
    has a stable, total ``game_index`` ordering for ELO/TrueSkill.
    """
    labels = sorted(factories)
    combined: list[GameOutcome] = []
    pairwise: dict[tuple[str, str], HeadToHeadStats] = {}

    pair_idx = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            fa, fb = factories[la], factories[lb]

            # Distinct seed per pair keeps draws independent; deterministic.
            pair_seed = seed + pair_idx * 100_003

            seat_factories: list[AgentFactory] = [fa, fb]
            seat_labels = {0: la, 1: lb}
            # Fill any extra seats by cycling the two contenders, so >2-player
            # games still pit the pair against each other (plus stochastic
            # duplicates), per the design's multi-opponent guidance.
            for seat in range(2, num_players):
                if seat % 2 == 0:
                    seat_factories.append(fa)
                    seat_labels[seat] = la
                else:
                    seat_factories.append(fb)
                    seat_labels[seat] = lb

            raw = run_batch(
                track,
                seat_factories,
                num_games=num_games_per_pair,
                parallel=parallel,
                seed=pair_seed,
            )

            # Head-to-head table (seat-0 vs seat-1 only, decisive by finish).
            a_wins = sum(1 for o in raw if o.winner_id == 0)
            b_wins = sum(1 for o in raw if o.winner_id == 1)
            decisive = a_wins + b_wins
            pairwise[(la, lb)] = HeadToHeadStats(
                agent_a=la,
                agent_b=lb,
                games=decisive,
                a_wins=a_wins,
                b_wins=b_wins,
                a_win_rate=a_wins / decisive if decisive else 0.0,
                a_win_rate_ci=wilson_interval(a_wins, decisive),
            )

            relabelled = _relabel_outcomes(raw, seat_labels)
            # Re-stamp game_index into a global, collision-free, ordered range.
            base = pair_idx * num_games_per_pair
            for k, o in enumerate(relabelled):
                combined.append(dataclasses.replace(o, game_index=base + k))
            pair_idx += 1

    return combined, pairwise


def round_robin_elo(
    factories: Mapping[str, AgentFactory],
    *,
    num_games_per_pair: int = 200,
    num_players: int = 2,
    tracks: list[Track] | None = None,
    seed: int = 0,
    rating: str = "elo",
    bootstrap: int = 1000,
    elo_k: float = 24.0,
    elo_start: float = 1500.0,
    parallel: bool = False,
) -> RoundRobinStats:
    """Round-robin every pair in ``factories``; compute ratings + pairwise table.

    Args:
        factories: ``label -> picklable factory`` for the agent pool (>= 2).
        num_games_per_pair: Games per unordered pair (per track).
        num_players: Seats per game (2..6). >2 seats fill by cycling the pair.
        tracks: Optional list of tracks to sweep (cross-track). When given, the
            round-robin runs on every track and the ratings aggregate across all
            of them; ``per_track`` win-rates are populated. ``None`` -> a single
            ``"usa"`` track and ``per_track=None``.
        seed: Base seed (deterministic, order-independent).
        rating: ``"elo"`` (default; pairwise decomposition + bootstrap CI) or
            ``"trueskill"`` (native N-way mu/sigma, additionally populated).
        bootstrap: ELO CI resample count (seeded -> deterministic). 0 disables.
        elo_k, elo_start: ELO constants (surfaced for reproducibility).
        parallel: Passed to ``run_batch``. **Defaults to ``False``**: an MLAgent
            in the pool reloads its SB3 model per worker, so sequential is the
            safe default; only top-level/``functools.partial`` factories (never
            lambdas) are safe with ``parallel=True``.

    Returns:
        A :class:`RoundRobinStats` with ELO ratings (always), an optional
        TrueSkill table, the pairwise head-to-head table, and optional per-track
        win-rates.
    """
    if rating not in ("elo", "trueskill"):
        raise ValueError(f"rating must be 'elo' or 'trueskill', got {rating!r}")
    if len(factories) < 2:
        raise ValueError("round_robin_elo needs at least 2 factories")
    if not (2 <= num_players <= 6):
        raise ValueError(f"num_players must be 2..6, got {num_players}")

    sweep = tracks if tracks else [load_track_by_name("usa")]

    all_outcomes: list[GameOutcome] = []
    pairwise: dict[tuple[str, str], HeadToHeadStats] = {}
    per_track: dict[str, dict[str, float]] | None = (
        {} if tracks else None
    )

    base = 0
    for t_idx, track in enumerate(sweep):
        outcomes, pw = _round_robin_outcomes(
            factories,
            num_games_per_pair=num_games_per_pair,
            num_players=num_players,
            track=track,
            seed=seed + t_idx * 1_000_003,
            parallel=parallel,
        )
        # Re-stamp global game_index across tracks so the combined list keeps a
        # total, deterministic order.
        for o in outcomes:
            all_outcomes.append(
                dataclasses.replace(o, game_index=base + o.game_index)
            )
        base += len(outcomes)

        # On the first track, take the pairwise table (single-track h2h view).
        if t_idx == 0:
            pairwise = pw

        if per_track is not None:
            stats = aggregate_stats(outcomes, by="agent_type")
            per_track[track.name] = {
                label: s.win_rate for label, s in stats.per_agent.items()
            }

    ratings: dict[str, EloRating] = compute_elo(
        all_outcomes,
        by="agent_type",
        start=elo_start,
        k=elo_k,
        bootstrap=bootstrap,
        bootstrap_seed=seed,
    )

    trueskill: dict[str, TrueSkillRating] | None = None
    if rating == "trueskill":
        trueskill = compute_trueskill(all_outcomes, by="agent_type")

    return RoundRobinStats(
        ratings=ratings,
        trueskill=trueskill,
        pairwise=pairwise,
        per_track=per_track,
    )


def evaluate_cross_track(
    factories: Mapping[str, AgentFactory] | list[AgentFactory],
    *,
    tracks: list[Track],
    num_games: int = 100,
    seed: int = 0,
    parallel: bool = False,
) -> dict[str, dict[str, AgentStats]]:
    """Run a fixed multi-agent eval across multiple tracks; per-track stats.

    The same heterogeneous pool of agents plays on each track in ``tracks``
    (one factory per seat). With 6A landed ``tracks`` may come from a track
    generator/sampler; without it, pass the static tracks
    (e.g. ``[load_track_by_name("usa"), load_track_by_name("silverstone")]``).
    The harness is identical either way.

    Args:
        factories: Either a ``label -> factory`` mapping (seated in sorted-label
            order) or an ordered ``list`` of factories (one per seat, 2..6).
        tracks: Non-empty list of tracks to sweep.
        num_games: Games per track.
        seed: Base seed (each track uses a derived seed).
        parallel: Passed to ``run_batch`` (default ``False`` for MLAgent pools).

    Returns:
        ``track_name -> {agent_type: AgentStats}``.
    """
    if not tracks:
        raise ValueError("evaluate_cross_track needs at least one track")

    if isinstance(factories, Mapping):
        seat_factories = [factories[label] for label in sorted(factories)]
    else:
        seat_factories = list(factories)
    if not (2 <= len(seat_factories) <= 6):
        raise ValueError(
            f"need 2..6 seat factories, got {len(seat_factories)}"
        )

    result: dict[str, dict[str, AgentStats]] = {}
    for t_idx, track in enumerate(tracks):
        outcomes = run_batch(
            track,
            seat_factories,
            num_games=num_games,
            parallel=parallel,
            seed=seed + t_idx * 1_000_003,
        )
        stats = aggregate_stats(outcomes, by="agent_type")
        result[track.name] = dict(stats.per_agent)
    return result


# ----------------------------------------------------------------------
# Sprint 8C: round-robin LEAGUE evaluator (free-for-all + rank-based ratings)
# ----------------------------------------------------------------------


@dataclasses.dataclass
class LeagueLadder:
    """Ratings over a pool of contenders from the league evaluator (§7.5).

    Attributes:
        ratings: ``label -> EloRating`` (always populated; rank-based ELO).
        trueskill: ``label -> TrueSkillRating`` when TrueSkill ran, else ``None``.
        pairwise: optional 2-player head-to-head table (``mode="pairwise"`` only).
        outcomes: the relabelled per-game :class:`GameOutcome` list (8D forward-
            compat seam, §7.5): ELO/TrueSkill are pure functions of this list, so
            8D can persist it and recompute ratings without replaying races.
            ``None`` by default to keep 8C small-N output light.
    """

    ratings: dict[str, EloRating]
    trueskill: dict[str, TrueSkillRating] | None = None
    pairwise: dict[tuple[str, str], HeadToHeadStats] | None = None
    outcomes: list[GameOutcome] | None = None


def _seat_rotations(labels: list[str], num_players: int) -> list[list[str]]:
    """Cyclic seat rotations of ``labels`` for seat-order cancellation (§7.4).

    Returns ``num_players`` rotations of the field so each contender occupies
    every grid slot equally over the set (a Latin-square-style cancellation of
    the front-seat advantage). Deterministic given the input order.
    """
    n = len(labels)
    return [[labels[(i + r) % n] for i in range(n)] for r in range(num_players)]


def _free_for_all_outcomes(
    contenders: Mapping[str, AgentFactory],
    *,
    tracks: list[Track],
    num_players: int,
    games_per_matchup: int,
    seed: int,
    parallel: bool,
) -> list[GameOutcome]:
    """Play every size-``num_players`` field of distinct contenders (§7.3/7.4).

    Enumerates the size-``num_players`` combinations of contender labels; each
    field is played on every track, on each cyclic seat rotation (so seat order
    cancels), relabelled by pool key, and stamped with a globally unique,
    deterministic ``game_index`` so the combined list has a stable total order
    for ELO / TrueSkill.
    """
    labels = sorted(contenders)
    if len(labels) < num_players:
        raise ValueError(
            f"need >= {num_players} contenders for a {num_players}-player "
            f"free-for-all, got {len(labels)}"
        )

    combined: list[GameOutcome] = []
    games_per_rotation = max(1, games_per_matchup // num_players)
    gindex = 0
    field_idx = 0

    for field in itertools.combinations(labels, num_players):
        for t_idx, track in enumerate(tracks):
            for r_idx, seating in enumerate(
                _seat_rotations(list(field), num_players)
            ):
                seat_factories = [contenders[lbl] for lbl in seating]
                seat_labels = {seat: lbl for seat, lbl in enumerate(seating)}
                # Deterministic, collision-free per-(field, track, rotation) seed.
                batch_seed = (
                    seed
                    + field_idx * 1_000_003
                    + t_idx * 10_007
                    + r_idx * 101
                )
                raw = run_batch(
                    track,
                    seat_factories,
                    num_games=games_per_rotation,
                    parallel=parallel,
                    seed=batch_seed,
                )
                relabelled = _relabel_outcomes(raw, seat_labels)
                for o in relabelled:
                    combined.append(
                        dataclasses.replace(o, game_index=gindex)
                    )
                    gindex += 1
        field_idx += 1

    return combined


def evaluate_league(
    contenders: Mapping[str, AgentFactory],
    *,
    tracks: list[Track],
    num_players: int = 4,
    games_per_matchup: int = 50,
    mode: str = "free_for_all",
    rating: str = "trueskill",
    seed: int = 0,
    parallel: bool = False,
    # --- 8D additions (additive; defaults preserve 8C small-N behavior) ---
    matchup_sampling: str = "exhaustive",
    fields_per_contender: int | None = None,
    games_per_field: int | None = None,
) -> LeagueLadder:
    """Round-robin a pool of checkpoints + reference heuristics; rank-based ratings.

    ``free_for_all`` (default): enumerate size-``num_players`` combinations of the
    contender labels, play each on rotated seatings (§7.4 -- seat order cancels),
    relabel by pool key (:func:`_relabel_outcomes`), and feed the combined
    outcomes to :func:`heat.simulation.stats.compute_elo` /
    :func:`~heat.simulation.stats.compute_trueskill` (both already rank-based and
    N-way). ``pairwise``: delegate to :func:`round_robin_elo`.

    The rating math is a pure function of the seeded outcome list, so a fixed
    ``(contenders, tracks, seed)`` yields byte-identical ratings (§7.5 / the
    determinism test). ``contenders`` may mix :func:`ml_agent_factory` checkpoints
    and :func:`heuristic_agent_factory` / :func:`strong_heuristic_agent_factory`
    reference yardsticks -- all picklable.

    Args:
        contenders: ``label -> picklable factory`` (>= ``num_players`` in
            free-for-all; >= 2 in pairwise).
        tracks: fixed held-out track set (non-empty).
        num_players: seats per free-for-all race (2..6).
        games_per_matchup: games per field (split across the seat rotations).
        mode: ``"free_for_all"`` (default) or ``"pairwise"``.
        rating: ``"trueskill"`` (default headline) or ``"elo"`` (always computed).
        seed: base seed (deterministic).
        parallel: passed to ``run_batch`` (default ``False`` for MLAgent pools).
        matchup_sampling: ``"exhaustive"`` (8C) or ``"sampled"`` (8D -- not yet
            implemented; raises ``NotImplementedError``).
        fields_per_contender / games_per_field: 8D sampled-mode knobs (unused in
            8C; accepted so the API is stable for 8D).

    Returns:
        A :class:`LeagueLadder` with ELO ratings (always), optional TrueSkill,
        an optional pairwise table (pairwise mode), and the relabelled outcomes.
    """
    if not tracks:
        raise ValueError("evaluate_league needs at least one track")
    if rating not in ("elo", "trueskill"):
        raise ValueError(f"rating must be 'elo' or 'trueskill', got {rating!r}")
    if matchup_sampling not in ("exhaustive", "sampled"):
        raise ValueError(
            f"matchup_sampling must be 'exhaustive' or 'sampled', "
            f"got {matchup_sampling!r}"
        )
    if matchup_sampling == "sampled":
        # 8D forward-compat seam (§7.5): the param + LeagueLadder.outcomes field
        # exist now; the balanced field sampler is filled in by Sprint 8D.
        raise NotImplementedError(
            "matchup_sampling='sampled' is a Sprint 8D feature; use "
            "'exhaustive' for the 8C small-N evaluator"
        )

    if mode == "pairwise":
        rr = round_robin_elo(
            contenders,
            num_games_per_pair=games_per_matchup,
            num_players=num_players,
            tracks=tracks,
            seed=seed,
            rating=rating,
            parallel=parallel,
        )
        return LeagueLadder(
            ratings=rr.ratings,
            trueskill=rr.trueskill,
            pairwise=rr.pairwise,
            outcomes=None,
        )
    if mode != "free_for_all":
        raise ValueError(
            f"mode must be 'free_for_all' or 'pairwise', got {mode!r}"
        )
    if not (2 <= num_players <= 6):
        raise ValueError(f"num_players must be 2..6, got {num_players}")

    outcomes = _free_for_all_outcomes(
        contenders,
        tracks=tracks,
        num_players=num_players,
        games_per_matchup=games_per_matchup,
        seed=seed,
        parallel=parallel,
    )

    ratings = compute_elo(outcomes, by="agent_type", bootstrap_seed=seed)
    trueskill = (
        compute_trueskill(outcomes, by="agent_type")
        if rating == "trueskill"
        else None
    )
    return LeagueLadder(
        ratings=ratings,
        trueskill=trueskill,
        pairwise=None,
        outcomes=outcomes,
    )
