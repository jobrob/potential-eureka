"""Batch simulation runner for the HEAT board game.

Runs many games headlessly (with ``logging_enabled=False`` for speed) and
returns a flat, fully-picklable list of :class:`GameOutcome` records. Supports
both a sequential path and a parallel path backed by
:class:`concurrent.futures.ProcessPoolExecutor`.

The runner takes agent *factories* (callables returning fresh agents), never
reused agent instances: seeded agents carry per-game RNG state, so reusing an
instance across games would continue its stream rather than reset it.

Determinism is execution-order-independent. With a fixed ``seed``, each game's
outcome is keyed only on its ``game_index`` (not on which worker ran it), so
sequential and parallel runs produce identical per-game results.

Usage:
    >>> from heat.tracks.loader import load_track_by_name
    >>> from heat.simulation.runner import run_batch, random_agent_factory
    >>> track = load_track_by_name("usa")
    >>> factories = [random_agent_factory(), random_agent_factory()]
    >>> outcomes = run_batch(track, factories, num_games=10, seed=42)
    >>> len(outcomes)
    10

Picklability note: under ``parallel=True`` the worker and the agent factories
are pickled to child processes (Windows uses ``spawn``). Factories must be
top-level callables or :func:`functools.partial` of top-level constructors.
Lambdas / local closures are *not* picklable; passing one triggers a
transparent fallback to the sequential path (with a warning). Callers that
intentionally use lambdas should pass ``parallel=False``.
"""

from __future__ import annotations

import functools
import logging
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents.search_agent import LookaheadAgent, DEFAULT_SPIN_PENALTY
from heat.engine.game import Agent, Game
from heat.models.track import Track

logger = logging.getLogger(__name__)

# A factory takes (player_id, seed) and returns a fresh agent instance.
AgentFactory = Callable[[int, "int | None"], Agent]

# Stride between consecutive games' base seeds; large enough that per-player
# seed offsets (player_id) never overlap between adjacent games.
_GAME_SEED_STRIDE = 1000


# ----------------------------------------------------------------------
# Result data structures (primitives/tuples only -> fully picklable)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerOutcome:
    """Per-player result of a single game.

    Attributes:
        player_id: Seat index (0-based).
        name: Player display name.
        agent_type: ``agent.__class__.__name__`` of the seat's agent.
        finish_position: 1-based rank (1 = winner).
        final_lap: Lap number at game end.
        final_position: Track space index at game end.
        heat_remaining: Heat available in the pool at game end (0..6).
    """

    player_id: int
    name: str
    agent_type: str
    finish_position: int
    final_lap: int
    final_position: int
    heat_remaining: int


@dataclass(frozen=True)
class GameOutcome:
    """Result of a single completed game.

    Attributes:
        game_index: 0-based index of this game within the batch.
        seed: The derived per-game base seed used, or ``None`` if unseeded.
        num_players: Number of seats in the game.
        winner_id: Player id of the winner (``finish_order[0]``).
        winner_name: Winner's display name.
        finish_order: Player ids best-to-worst.
        total_rounds: Rounds played.
        players: Per-player outcomes, ordered by ``player_id``.
    """

    game_index: int
    seed: int | None
    num_players: int
    winner_id: int
    winner_name: str
    finish_order: tuple[int, ...]
    total_rounds: int
    players: tuple[PlayerOutcome, ...]


# ----------------------------------------------------------------------
# Picklable agent factories
# ----------------------------------------------------------------------


def _make_random(player_id: int, seed: int | None, name: str | None) -> Agent:
    """Top-level constructor for a seeded :class:`RandomAgent`."""
    agent_name = name if name is not None else f"Random-{player_id}"
    return RandomAgent(seed=seed, name=agent_name)


def _make_heuristic(player_id: int, seed: int | None, name: str | None) -> Agent:
    """Top-level constructor for a :class:`HeuristicAgent` (ignores ``seed``)."""
    agent_name = name if name is not None else f"Heuristic-{player_id}"
    return HeuristicAgent(name=agent_name)


def _make_strong_heuristic(
    player_id: int,
    seed: int | None,
    name: str | None,
    strength: int,
    heat_price: float | None,
    use_seed: bool,
) -> Agent:
    """Top-level constructor for a :class:`StrongHeuristicAgent`.

    The agent is deterministic given the state; ``seed`` is only used for
    equal-value tie-breaks, and only when ``use_seed`` is True (so the default
    factory stays fully deterministic and order-independent under
    ``run_batch``). When enabled, the per-player derived seed is threaded
    through so a seeded batch is reproducible.
    """
    agent_name = name if name is not None else f"StrongHeuristic-{player_id}"
    return StrongHeuristicAgent(
        name=agent_name,
        strength=strength,
        heat_price=heat_price,
        seed=seed if use_seed else None,
    )


def random_agent_factory(name: str | None = None) -> AgentFactory:
    """Return a picklable factory producing seeded :class:`RandomAgent`s."""
    return functools.partial(_make_random, name=name)


def heuristic_agent_factory(name: str | None = None) -> AgentFactory:
    """Return a picklable factory producing :class:`HeuristicAgent`s."""
    return functools.partial(_make_heuristic, name=name)


def strong_heuristic_agent_factory(
    name: str | None = None,
    *,
    strength: int = 2,
    heat_price: float | None = None,
    use_seed: bool = False,
) -> AgentFactory:
    """Return a picklable factory producing :class:`StrongHeuristicAgent`s.

    Mirrors :func:`heuristic_agent_factory`: a top-level ``functools.partial``
    (no lambdas/closures) so it pickles cleanly for ``run_batch(parallel=True)``.

    Args:
        name: Optional fixed display name (else ``StrongHeuristic-{id}``).
        strength: Difficulty rung 0..3 (see :class:`StrongHeuristicAgent`).
        heat_price: Optional override of the shadow price of heat in spaces.
        use_seed: If True, thread the per-player derived seed into the agent
            for equal-value tie-breaking. Off by default to keep the agent
            fully deterministic regardless of seat order.
    """
    return functools.partial(
        _make_strong_heuristic,
        name=name,
        strength=strength,
        heat_price=heat_price,
        use_seed=use_seed,
    )


def _make_lookahead(
    player_id: int,
    seed: int | None,
    name: str | None,
    horizon: int,
    n_determinizations: int,
    spin_penalty: float,
    leaf_value: str,
    value_model_path: str | None,
    determinize_hidden: bool,
    top_k: int | None,
    sim_budget: int | None,
    use_seed: bool,
) -> Agent:
    """Top-level (picklable) constructor for a :class:`LookaheadAgent`.

    Carries only config primitives (never a live rollout-policy instance) so the
    enclosing :func:`functools.partial` pickles into ``ProcessPoolExecutor``
    workers; the agent builds its own default :class:`HeuristicAgent` rollout
    policy. ``seed`` is threaded into the per-turn rollout RNG only when
    ``use_seed`` is set, so the default factory stays order-independent. The
    S2 levers (``determinize_hidden``, ``top_k``, ``sim_budget``) are all plain
    primitives so the league/tuned-agent factory pickles unchanged. The A2
    learned-leaf lever (``value_model_path``) is likewise a plain string so the
    learned-leaf agent pickles by path -- each worker reloads V on first use.
    """
    agent_name = name if name is not None else f"Lookahead-{player_id}"
    return LookaheadAgent(
        name=agent_name,
        horizon=horizon,
        n_determinizations=n_determinizations,
        spin_penalty=spin_penalty,
        leaf_value=leaf_value,
        value_model_path=value_model_path,
        determinize_hidden=determinize_hidden,
        top_k=top_k,
        sim_budget=sim_budget,
        seed=seed if use_seed else None,
    )


def lookahead_agent_factory(
    name: str | None = None,
    *,
    horizon: int = 2,
    n_determinizations: int = 2,
    spin_penalty: float = DEFAULT_SPIN_PENALTY,
    leaf_value: str = "progress",
    value_model_path: str | None = None,
    determinize_hidden: bool = False,
    top_k: int | None = None,
    sim_budget: int | None = None,
    use_seed: bool = False,
) -> AgentFactory:
    """Return a picklable factory producing :class:`LookaheadAgent`s.

    Mirrors :func:`strong_heuristic_agent_factory`: a top-level
    ``functools.partial`` of a top-level constructor (no lambdas/closures) so it
    pickles cleanly for ``run_batch(parallel=True)``. The agent owns its default
    rollout policy, so only config primitives cross the process boundary.

    Args:
        name: Optional fixed display name (else ``Lookahead-{id}``).
        horizon: Rollout depth in rounds (``0`` == greedy one-ply).
        n_determinizations: Clones averaged per candidate (own-draw variance, and
            -- with ``determinize_hidden`` -- opponent-hand variance).
        spin_penalty: ``lambda`` in ``progress - lambda * spins`` (spaces/spin).
        leaf_value: ``"progress"``, ``"move_eval"``, or ``"learned"`` (A2).
        value_model_path: A1 value-net checkpoint path; required when
            ``leaf_value == "learned"``. A plain string so the learned-leaf agent
            pickles by path (each worker reloads V on first use).
        determinize_hidden: Re-sample opponents' hidden hands/decks per rollout
            (S2 multiplayer determinization). No effect in solo.
        top_k: Keep only the best ``top_k`` candidates by the fast prior.
        sim_budget: Per-move cap on rollout clones.
        use_seed: Thread the per-player derived seed into the rollout RNG (off by
            default so the agent is fully deterministic regardless of seat order).
    """
    return functools.partial(
        _make_lookahead,
        name=name,
        horizon=horizon,
        n_determinizations=n_determinizations,
        spin_penalty=spin_penalty,
        leaf_value=leaf_value,
        value_model_path=value_model_path,
        determinize_hidden=determinize_hidden,
        top_k=top_k,
        sim_budget=sim_budget,
        use_seed=use_seed,
    )


# ----------------------------------------------------------------------
# Seeding helpers
# ----------------------------------------------------------------------


def _derive_game_seed(base_seed: int | None, game_index: int) -> int | None:
    """Derive a per-game base seed from the batch seed and game index."""
    if base_seed is None:
        return None
    return base_seed + game_index * _GAME_SEED_STRIDE


def _first_unpicklable(
    track: Track, agent_factories: Sequence[AgentFactory]
) -> str | None:
    """Return a label for the first unpicklable worker arg, or ``None``.

    Used as a pre-flight check before spawning a process pool, since spawn
    surfaces pickling failures only asynchronously (and after partial work).
    """
    candidates = [("track", track)]
    candidates += [
        (f"agent_factories[{i}]", f) for i, f in enumerate(agent_factories)
    ]
    for label, obj in candidates:
        try:
            pickle.dumps(obj)
        except (TypeError, AttributeError, pickle.PicklingError):
            return label
    return None


# ----------------------------------------------------------------------
# Worker (top-level / importable for ProcessPoolExecutor)
# ----------------------------------------------------------------------


def run_single_game(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    game_index: int,
    base_seed: int | None,
    laps: int | None = None,
) -> GameOutcome:
    """Run one game and return its :class:`GameOutcome`.

    Top-level and picklable so it can be dispatched to a worker process.

    Determinism: when ``base_seed`` is not ``None``, the global ``random``
    module is seeded with the derived per-game seed (the engine and some
    agents use the global RNG), and each factory receives a per-player
    derived seed. Both depend only on ``game_index``, so the result is
    independent of which process/order ran it.

    The ``laps`` override mutates the passed ``track`` in place. Under
    ``ProcessPoolExecutor`` each worker receives its own pickled copy, so the
    mutation is process-local; in the sequential path the caller's track is
    mutated (consistent with ``scripts/run_race.py`` semantics).
    """
    game_seed = _derive_game_seed(base_seed, game_index)

    # Migration (Sprint 5, Step A3): deck shuffling now uses the per-game RNG
    # owned by GameState, threaded via ``Game(seed=...)`` below, instead of the
    # module-global ``random``. The global seed is retained transitionally so
    # any remaining global-RNG consumers (e.g. agents that fall back to it) stay
    # deterministic; it can be removed once the engine no longer touches the
    # global module at all.
    if game_seed is not None:
        random.seed(game_seed)

    if laps is not None:
        track.laps = laps

    agents: list[Agent] = []
    for player_id, factory in enumerate(agent_factories):
        per_player_seed = (
            game_seed + player_id if game_seed is not None else None
        )
        agents.append(factory(player_id, per_player_seed))

    game = Game(track, agents, logging_enabled=False, seed=game_seed)
    result = game.run()

    player_outcomes: list[PlayerOutcome] = []
    for player in game.state.players:
        pid = player.player_id
        player_outcomes.append(
            PlayerOutcome(
                player_id=pid,
                name=player.name,
                agent_type=type(agents[pid]).__name__,
                finish_position=player.finish_order,
                final_lap=player.lap,
                final_position=player.position,
                heat_remaining=player.heat_available,
            )
        )
    # Order by player_id for stable, deterministic output.
    player_outcomes.sort(key=lambda po: po.player_id)

    finish_order = tuple(result.finish_order)
    winner_id = finish_order[0]
    winner_name = game.state.get_player(winner_id).name

    return GameOutcome(
        game_index=game_index,
        seed=game_seed,
        num_players=len(agents),
        winner_id=winner_id,
        winner_name=winner_name,
        finish_order=finish_order,
        total_rounds=result.total_rounds,
        players=tuple(player_outcomes),
    )


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------


def _run_sequential(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    num_games: int,
    base_seed: int | None,
    laps: int | None,
    progress: bool,
) -> list[GameOutcome]:
    """Run all games sequentially in this process."""
    indices: Sequence[int] = range(num_games)
    indices = _maybe_progress(indices, total=num_games, enabled=progress)
    return [
        run_single_game(track, agent_factories, i, base_seed, laps)
        for i in indices
    ]


def _run_parallel(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    num_games: int,
    base_seed: int | None,
    laps: int | None,
    max_workers: int | None,
    progress: bool,
) -> list[GameOutcome]:
    """Run games across worker processes, re-sorting results by game_index."""
    outcomes: list[GameOutcome] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_game, track, agent_factories, i, base_seed, laps
            )
            for i in range(num_games)
        ]
        completed = _maybe_progress(
            as_completed(futures), total=num_games, enabled=progress
        )
        for future in completed:
            outcomes.append(future.result())
    # Futures complete out of order; restore game_index ordering.
    outcomes.sort(key=lambda o: o.game_index)
    return outcomes


def _maybe_progress(iterable, total: int, enabled: bool):
    """Wrap an iterable in a tqdm progress bar if requested and available."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover - tqdm is optional
        logger.warning("tqdm not installed; progress bar disabled.")
        return iterable
    return tqdm(iterable, total=total)


def run_batch(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    num_games: int,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    seed: int | None = None,
    laps: int | None = None,
    progress: bool = False,
) -> list[GameOutcome]:
    """Run ``num_games`` games and return outcomes ordered by ``game_index``.

    Args:
        track: The track to race on (``laps`` may be overridden via ``laps``).
        agent_factories: One picklable factory per seat (1..6).
        num_games: Number of games to run (>= 1).
        parallel: Use a process pool. Forced off when ``num_games == 1``.
        max_workers: Worker count for the process pool (``None`` = default).
        seed: Base seed for deterministic, execution-order-independent runs.
        laps: Optional per-game laps override (applied process-locally).
        progress: Show a ``tqdm`` progress bar if ``tqdm`` is installed.

    Returns:
        A list of :class:`GameOutcome`, one per game, ordered by ``game_index``.

    Raises:
        ValueError: if the factory count is not in 1..6, or ``num_games < 1``.

    Notes:
        If parallel execution fails to pickle the worker/factories (e.g. a
        lambda factory), the runner logs a warning and falls back to the
        sequential path so the batch still completes.
    """
    n = len(agent_factories)
    if not (1 <= n <= 6):
        raise ValueError(f"Need 1..6 agent factories, got {n}")
    if num_games < 1:
        raise ValueError(f"num_games must be >= 1, got {num_games}")

    # Single game always runs sequentially (no executor overhead/benefit).
    if not parallel or num_games == 1:
        return _run_sequential(
            track, agent_factories, num_games, seed, laps, progress
        )

    # Pre-flight picklability check. Under spawn the executor only raises a
    # pickling error asynchronously (from future.result()), by which point
    # some games may already have run. Detecting it up front lets us fall
    # back cleanly to a from-scratch sequential run with no partial work.
    unpicklable = _first_unpicklable(track, agent_factories)
    if unpicklable is not None:
        logger.warning(
            "Parallel execution disabled: %s is not picklable; falling back "
            "to sequential. Use top-level/partial factories for parallelism.",
            unpicklable,
        )
        return _run_sequential(
            track, agent_factories, num_games, seed, laps, progress
        )

    try:
        return _run_parallel(
            track, agent_factories, num_games, seed, laps, max_workers, progress
        )
    except (TypeError, AttributeError, pickle.PicklingError) as exc:
        logger.warning(
            "Parallel execution failed to pickle (%s); falling back to "
            "sequential. Use top-level/partial factories for parallelism.",
            exc,
        )
        return _run_sequential(
            track, agent_factories, num_games, seed, laps, progress
        )
