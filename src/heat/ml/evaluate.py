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

import functools

from heat.agents.ml_agent import MLAgent
from heat.engine.game import Agent
from heat.models.track import Track
from heat.simulation.runner import (
    AgentFactory,
    GameOutcome,
    heuristic_agent_factory,
    run_batch,
)
from heat.simulation.stats import AgentStats, aggregate_stats
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
) -> dict[str, AgentStats]:
    """Evaluate a trained checkpoint vs an opponent over a batch of games.

    Seat 0 is the :class:`MLAgent`; the remaining ``num_players - 1`` seats are
    built from ``opponent_factory`` (default: :func:`heuristic_agent_factory`).

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

    Returns:
        The ``per_agent`` mapping from :func:`aggregate_stats` (grouped by
        ``agent_type``): ``{agent_type -> AgentStats}``. The headline
        ``"MLAgent"`` vs ``"HeuristicAgent"`` win rates live here.
    """
    if not (2 <= num_players <= 6):
        raise ValueError(f"num_players must be 2..6, got {num_players}")

    if opponent_factory is None:
        opponent_factory = heuristic_agent_factory()

    if track is None:
        track = load_track_by_name("usa")
    elif isinstance(track, str):
        track = load_track_by_name(track)

    factories: list[AgentFactory] = [ml_agent_factory(model_path)]
    factories += [opponent_factory] * (num_players - 1)

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
