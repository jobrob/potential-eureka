"""Opponent-policy adapter for the HEAT Gym environment (Sprint 5b).

:class:`HeatEnv` drives a single learning seat against a pool of opponent
policies. The driver (``run_round_driver``) yields a :class:`Decision` for
*every* seat, including opponents; the env must turn an opponent's ``Decision``
into the concrete engine action the driver's ``send(...)`` expects.

This module isolates that "Decision -> BaseAgent pull-method" dispatch so
``env.py`` stays focused on the gym mechanics. The opponents are ordinary
:class:`heat.agents.base.BaseAgent` instances (``HeuristicAgent``,
``RandomAgent``, or — in 5c self-play — frozen ``MLAgent`` snapshots), so they
always return legal moves.
"""

from __future__ import annotations

from typing import cast

from heat.agents.base import BaseAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card
from heat.models.game_state import GameState


def opponent_action(agent: BaseAgent, decision: Decision, state: GameState) -> object:
    """Return the concrete engine action an opponent ``agent`` makes for
    ``decision``.

    Dispatches on ``decision.kind`` to the matching ``BaseAgent`` pull-method,
    passing the already-enumerated legal set carried by the ``Decision`` so the
    opponent only ever chooses a legal move. The return type matches what the
    driver expects for that kind (see :class:`heat.engine.driver.Decision`).

    Note the REACT path: ``BaseAgent.choose_react`` takes the *unpacked*
    ``ReactOptions`` fields (``max_cooldown``, ``can_boost``, ``has_adrenaline``),
    not the ``ReactOptions`` object itself.
    """
    kind = decision.kind
    pid = decision.player_id

    if kind == DecisionKind.GEAR:
        # legal: list[(new_gear, heat_cost)]
        legal_gears = cast(list[tuple[int, int]], decision.legal)
        return agent.choose_gear(state, pid, legal_gears)

    if kind == DecisionKind.CARDS:
        # legal: list[tuple[Card, ...]]
        legal_plays = cast(list[tuple[Card, ...]], decision.legal)
        return agent.choose_cards(state, pid, legal_plays)

    if kind == DecisionKind.REACT:
        # legal: rules.ReactOptions -> unpack into choose_react's fields.
        opts = cast(rules.ReactOptions, decision.legal)
        result: ReactDecision = agent.choose_react(
            state,
            pid,
            opts.max_cooldown,
            opts.can_boost,
            opts.has_adrenaline,
        )
        return result

    if kind == DecisionKind.SLIPSTREAM:
        return agent.choose_slipstream(state, pid)

    if kind == DecisionKind.DISCARD:
        # legal: list[Card]
        discardable = cast(list[Card], decision.legal)
        return agent.choose_discard(state, pid, discardable)

    raise ValueError(f"Unknown decision kind {kind!r}")  # pragma: no cover
