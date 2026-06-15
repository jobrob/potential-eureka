"""Tests for the step-wise round driver (Step D1)."""

from __future__ import annotations

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track


def _track(laps: int = 2) -> Track:
    spaces = [Space(i, lanes=2) for i in range(16)]
    corners = [Corner(start=5, end=6, speed_limit=3),
               Corner(start=12, end=13, speed_limit=2)]
    return Track("DriverTrack", spaces, corners, [0, 1, 2, 3], laps=laps)


def _start_game_state(num_players: int = 4, seed: int = 1) -> GameState:
    state = GameState.create(_track(), num_players, seed=seed)
    for p in state.players:
        p.lap = 1
    return state


def _default_action(decision: Decision):
    """Answer any decision with a simple, always-legal action."""
    if decision.kind == DecisionKind.GEAR:
        return decision.legal[0]
    if decision.kind == DecisionKind.CARDS:
        return decision.legal[0]
    if decision.kind == DecisionKind.REACT:
        return ReactDecision()  # do nothing
    if decision.kind == DecisionKind.SLIPSTREAM:
        return False
    if decision.kind == DecisionKind.DISCARD:
        return []  # discard nothing
    raise AssertionError(f"unexpected decision {decision.kind}")


def _drive(state: GameState, answer=_default_action) -> list[Decision]:
    """Drive one round, recording the decisions yielded."""
    gen = run_round_driver(state)
    seen: list[Decision] = []
    try:
        decision = next(gen)
        while True:
            seen.append(decision)
            action = answer(decision)
            decision = gen.send(action)
    except StopIteration:
        pass
    return seen


class TestDecisionSequence:
    def test_gear_and_card_phases_collected_first(self) -> None:
        state = _start_game_state()
        seen = _drive(state)
        kinds = [d.kind for d in seen]
        # The first 4 decisions are GEAR (one per active player), then 4 CARDS,
        # before any per-player sequential decision appears.
        assert kinds[:4] == [DecisionKind.GEAR] * 4
        assert kinds[4:8] == [DecisionKind.CARDS] * 4

    def test_gear_legal_matches_rules(self) -> None:
        state = _start_game_state()
        gen = run_round_driver(state)
        first = next(gen)
        assert first.kind == DecisionKind.GEAR
        player = state.get_player(first.player_id)
        expected = rules.legal_gear_shifts(player.gear, player.heat_available)
        assert first.legal == expected

    def test_react_legal_is_react_options(self) -> None:
        state = _start_game_state()
        seen = _drive(state)
        react = [d for d in seen if d.kind == DecisionKind.REACT]
        assert react
        for d in react:
            assert isinstance(d.legal, rules.ReactOptions)

    def test_round_advances_after_drive(self) -> None:
        state = _start_game_state()
        assert state.round_num == 1
        _drive(state)
        assert state.round_num == 2


class TestClutteredSkip:
    def test_cluttered_player_skips_react_slip_discard(self) -> None:
        """A player forced to fill with heat must not get React/slip/discard
        decisions, only GEAR + CARDS."""
        state = _start_game_state(num_players=1, seed=3)
        player = state.players[0]
        player.gear = 4  # must play 4 cards

        # Build a hand with too few playable (non-Heat) cards for gear 4.
        from heat.models.cards import Card, CardType
        heats = [c for c in player.hand if c.card_type == CardType.HEAT]
        # Replace hand with 1 speed + 4 heat -> only 1 playable < 4 needed.
        speed = next(
            c for c in player.hand if c.card_type == CardType.SPEED
        )
        extra_heat = [Card(CardType.HEAT, 0, f"p0_xheat_{i}") for i in range(4)]
        player.hand = [speed] + extra_heat

        assert rules.is_cluttered_hand(player.hand, player.gear)

        seen = _drive(state)
        kinds = [d.kind for d in seen]
        assert DecisionKind.REACT not in kinds
        assert DecisionKind.SLIPSTREAM not in kinds
        assert DecisionKind.DISCARD not in kinds
        # Cluttered car forced to gear 1.
        assert player.gear == 1


class TestSimultaneity:
    def test_all_gear_collected_before_apply(self) -> None:
        """No gear is applied until every active player's gear is collected
        (the driver yields all GEAR decisions before the CARDS phase, and
        gears are unchanged during gear collection)."""
        state = _start_game_state()
        original_gears = {p.player_id: p.gear for p in state.players}
        gen = run_round_driver(state)
        decision = next(gen)
        # Answer the gear decisions; during collection no gear should change.
        while decision.kind == DecisionKind.GEAR:
            assert all(
                state.get_player(pid).gear == original_gears[pid]
                for pid in original_gears
            )
            decision = gen.send(decision.legal[0])
        # Now we are at CARDS, meaning gears have been applied.
        assert decision.kind == DecisionKind.CARDS
