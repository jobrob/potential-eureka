"""Focused contracts for the versioned T2 static-bot repairs."""

from __future__ import annotations

import pickle

from heat.agents import _move_eval as ME
from heat.agents.static_v2 import (
    HeuristicV2Agent,
    RepairedHeuristicV2Agent,
    StaticSearchV2Agent,
)
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.engine.game import Game
from heat.tracks.generator import generate_track
from heat.simulation.runner import (
    heuristic_v2_agent_factory,
    repaired_heuristic_v2_agent_factory,
    static_search_v2_agent_factory,
)


def _card(card_id: str, value: int = 1, *, stress: bool = False) -> Card:
    """Build a small distinct card for admission and risk tests."""
    card_type = CardType.STRESS if stress else CardType.SPEED
    return Card(card_type, 0 if stress else value, card_id)


def _eval(value: float, risk: float) -> ME.MoveEval:
    """Build only the evaluator fields the V2 admission rules consume."""
    return ME.MoveEval(value, 1.0, 0, 0, risk)


def test_v2_cache_signatures_change_across_rounds_and_hands() -> None:
    """A same-position later turn cannot reuse either V2 bot's cached plan."""
    state = GameState.create(generate_track(715_000), 2, seed=17)
    for agent in (StaticSearchV2Agent(), RepairedHeuristicV2Agent()):
        first = agent._turn_signature(state, 0)
        state.round_num += 1
        assert agent._turn_signature(state, 0) != first
        state.round_num -= 1

        player = state.get_player(0)
        original = player.hand[0]
        player.hand[0] = _card("replacement", original.value)
        assert agent._turn_signature(state, 0) != first
        player.hand[0] = original


def test_search_v2_vetoes_certain_spin_and_reserves_safe_slot(monkeypatch) -> None:
    """Top-k always exposes a safe line when the analytical prior has one."""
    state = GameState.create(generate_track(715_001), 2, seed=18)
    certain = ((1, 0), (_card("certain"),))
    risky_a = ((1, 0), (_card("risky-a"),))
    risky_b = ((1, 0), (_card("risky-b"),))
    safe = ((1, 0), (_card("safe"),))
    values = {
        "certain": _eval(40.0, 1.0),
        "risky-a": _eval(30.0, 0.8),
        "risky-b": _eval(20.0, 0.6),
        "safe": _eval(1.0, 0.0),
    }
    monkeypatch.setattr(
        "heat.agents.static_v2._move_risk",
        lambda _state, _pid, _gear, cards: values[cards[0].id],
    )
    monkeypatch.setattr(
        StrongHeuristicAgent,
        "_play_guarantees_spin",
        staticmethod(lambda _state, _pid, cards, _heat: cards[0].id == "certain"),
    )
    agent = StaticSearchV2Agent()
    agent.top_k = 2

    admitted = agent._prune_top_k(
        state, 0, [certain, risky_a, risky_b, safe]
    )

    assert certain not in admitted
    assert admitted == [risky_a, safe]


def test_weak_v2_avoids_high_risk_stress_when_safe_play_exists(monkeypatch) -> None:
    """The weak V2 guard overrides its old Stress-card preservation bonus."""
    state = GameState.create(generate_track(715_002), 2, seed=19)
    state.get_player(0).gear = 1
    stress = (_card("stress", stress=True),)
    safe = (_card("safe", 1),)
    monkeypatch.setattr(
        "heat.agents.static_v2._move_risk",
        lambda _state, _pid, _gear, cards: (
            _eval(5.0, 0.75) if cards == stress else _eval(1.0, 0.0)
        ),
    )

    assert HeuristicV2Agent().choose_cards(state, 0, [stress, safe]) == safe


def test_repaired_v2_filters_high_risk_stress_candidate() -> None:
    """The repaired V2 guard applies to both joint and card-only planning."""
    stress = (_card("stress", stress=True),)
    safe = (_card("safe", 1),)
    scored = [
        (_eval(5.0, 0.75), False, ((2, 0), stress)),
        (_eval(1.0, 0.0), False, ((2, 0), safe)),
    ]

    eligible = RepairedHeuristicV2Agent()._reject_avoidable_certain_spins(scored)

    assert eligible == [scored[1]]


def test_v2_population_replays_a_full_game_deterministically() -> None:
    """The repaired policies remain legal and deterministic as a group."""
    track = generate_track(715_003)

    def run() -> tuple[list[int], list[object]]:
        game = Game(
            track,
            [StaticSearchV2Agent(), RepairedHeuristicV2Agent(), HeuristicV2Agent()],
            logging_enabled=True,
            seed=20,
        )
        result = game.run()
        return result.finish_order, list(result.event_log)

    first = run()
    assert first == run()
    assert len(first[0]) == 3


def test_v2_factories_are_picklable_and_keep_versioned_names() -> None:
    """The adopted bots can cross worker boundaries without identity drift."""
    factories = (
        (heuristic_v2_agent_factory(), HeuristicV2Agent, "HeuristicWeakV2-2"),
        (
            repaired_heuristic_v2_agent_factory(),
            RepairedHeuristicV2Agent,
            "HeuristicRepairedV2-2",
        ),
        (static_search_v2_agent_factory(), StaticSearchV2Agent, "StaticSearchV2-2"),
    )
    for factory, expected_type, expected_name in factories:
        agent = pickle.loads(pickle.dumps(factory))(2, 99)
        assert isinstance(agent, expected_type)
        assert agent.name == expected_name
