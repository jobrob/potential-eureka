"""Tests for heat.agents module."""

from __future__ import annotations

import pytest

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.models.track import Corner, Space, Track
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent
from heat.agents.random_agent import RandomAgent
from heat.agents.heuristic_agent import HeuristicAgent


# ---------------------------------------------------------------------------
# Test track helpers
# ---------------------------------------------------------------------------

def _small_track(laps: int = 1) -> Track:
    """A minimal track for testing: 10 spaces, 1 corner at positions 4-6."""
    spaces = [Space(i, lanes=2) for i in range(10)]
    corners = [Corner(start=4, end=6, speed_limit=3)]
    return Track("TestSmall", spaces, corners, [0, 1, 2, 3], laps=laps)


def _make_game_state(
    track: Track | None = None,
    num_players: int = 2,
) -> GameState:
    """Create a GameState for testing agents."""
    if track is None:
        track = _small_track()
    state = GameState.create(track, num_players, logging_enabled=False)
    for p in state.players:
        p.lap = 1
    return state


# ===========================================================================
# Step 1: BaseAgent tests
# ===========================================================================


class TestBaseAgent:
    def test_cannot_instantiate_directly(self) -> None:
        """BaseAgent is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_subclass_implementing_all_methods_works(self) -> None:
        """A subclass implementing all abstract methods can be instantiated."""

        class ConcreteAgent(BaseAgent):
            def choose_gear(self, state, player_id, legal_gears):
                return legal_gears[0]

            def choose_cards(self, state, player_id, legal_plays):
                return legal_plays[0]

            def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
                return ReactDecision()

            def choose_slipstream(self, state, player_id):
                return False

            def choose_discard(self, state, player_id, discardable):
                return []

        agent = ConcreteAgent(name="TestBot")
        assert agent.name == "TestBot"

    def test_repr(self) -> None:
        """__repr__ returns ClassName('name')."""

        class ConcreteAgent(BaseAgent):
            def choose_gear(self, state, player_id, legal_gears):
                return legal_gears[0]

            def choose_cards(self, state, player_id, legal_plays):
                return legal_plays[0]

            def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
                return ReactDecision()

            def choose_slipstream(self, state, player_id):
                return False

            def choose_discard(self, state, player_id, discardable):
                return []

        agent = ConcreteAgent(name="MyBot")
        assert repr(agent) == "ConcreteAgent('MyBot')"

    def test_subclass_missing_method_raises(self) -> None:
        """A subclass missing an abstract method cannot be instantiated."""

        class PartialAgent(BaseAgent):
            def choose_gear(self, state, player_id, legal_gears):
                return legal_gears[0]
            # Missing other methods

        with pytest.raises(TypeError):
            PartialAgent()  # type: ignore[abstract]


# ===========================================================================
# Step 2: RandomAgent tests
# ===========================================================================


class TestRandomAgent:
    def test_choose_gear_always_legal(self) -> None:
        """choose_gear always returns a member of legal_gears."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        legal = [(1, 0), (2, 0), (3, 1)]
        for _ in range(50):
            choice = agent.choose_gear(state, 0, legal)
            assert choice in legal

    def test_choose_cards_always_legal(self) -> None:
        """choose_cards always returns a member of legal_plays."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        c1 = Card(CardType.SPEED, 3, "s3")
        c2 = Card(CardType.SPEED, 4, "s4")
        c3 = Card(CardType.SPEED, 5, "s5")
        legal = [(c1,), (c2,), (c3,)]
        for _ in range(50):
            choice = agent.choose_cards(state, 0, legal)
            assert choice in legal

    def test_choose_react_cooldown_in_range(self) -> None:
        """choose_react cooldown_count is between 0 and max_cooldown."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        for _ in range(50):
            decision = agent.choose_react(state, 0, max_cooldown=3, can_boost=True, has_adrenaline=True)
            assert 0 <= decision.cooldown_count <= 3

    def test_choose_react_no_boost_when_cannot(self) -> None:
        """choose_react never boosts when can_boost is False."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        for _ in range(50):
            decision = agent.choose_react(state, 0, max_cooldown=1, can_boost=False, has_adrenaline=False)
            assert decision.use_boost is False
            assert decision.use_adrenaline_speed is False
            assert decision.use_adrenaline_cooldown is False

    def test_choose_slipstream_returns_bool(self) -> None:
        """choose_slipstream returns True or False."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        results = set()
        for _ in range(50):
            results.add(agent.choose_slipstream(state, 0))
        assert results == {True, False}

    def test_choose_discard_subset(self) -> None:
        """choose_discard returns a subset of discardable."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        discardable = [
            Card(CardType.SPEED, 1, "d1"),
            Card(CardType.SPEED, 2, "d2"),
            Card(CardType.SPEED, 3, "d3"),
        ]
        for _ in range(50):
            result = agent.choose_discard(state, 0, discardable)
            assert all(c in discardable for c in result)

    def test_choose_discard_can_be_empty_or_nonempty(self) -> None:
        """choose_discard can return empty or non-empty subsets."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        discardable = [
            Card(CardType.SPEED, 1, "d1"),
            Card(CardType.SPEED, 2, "d2"),
        ]
        sizes = set()
        for _ in range(50):
            result = agent.choose_discard(state, 0, discardable)
            sizes.add(len(result))
        # Should see at least empty and non-empty
        assert 0 in sizes
        assert any(s > 0 for s in sizes)

    def test_choose_discard_empty_input(self) -> None:
        """choose_discard returns empty for empty input."""
        agent = RandomAgent(seed=42)
        state = _make_game_state()
        assert agent.choose_discard(state, 0, []) == []

    def test_seed_reproducibility(self) -> None:
        """Two agents with the same seed produce the same results."""
        state = _make_game_state()
        legal_gears = [(1, 0), (2, 0), (3, 1)]
        c1 = Card(CardType.SPEED, 3, "s3")
        c2 = Card(CardType.SPEED, 4, "s4")
        legal_plays = [(c1,), (c2,)]

        agent1 = RandomAgent(seed=999)
        agent2 = RandomAgent(seed=999)

        for _ in range(20):
            assert agent1.choose_gear(state, 0, legal_gears) == agent2.choose_gear(state, 0, legal_gears)
            assert agent1.choose_cards(state, 0, legal_plays) == agent2.choose_cards(state, 0, legal_plays)

    def test_is_base_agent_subclass(self) -> None:
        """RandomAgent is a subclass of BaseAgent."""
        agent = RandomAgent(seed=0)
        assert isinstance(agent, BaseAgent)

    def test_repr(self) -> None:
        """RandomAgent repr works."""
        agent = RandomAgent(seed=0, name="RNG")
        assert repr(agent) == "RandomAgent('RNG')"


# ===========================================================================
# Step 3: HeuristicAgent tests
# ===========================================================================


def _straight_track(laps: int = 1) -> Track:
    """A track with no corners -- pure straights."""
    spaces = [Space(i, lanes=2) for i in range(20)]
    return Track("Straight", spaces, [], [0, 1, 2, 3], laps=laps)


def _corner_track() -> Track:
    """Track with a corner at positions 3-5, speed limit 3."""
    spaces = [Space(i, lanes=2) for i in range(15)]
    corners = [Corner(start=3, end=5, speed_limit=3)]
    return Track("CornerTrack", spaces, corners, [0, 1, 2, 3], laps=1)


class TestHeuristicChooseGear:
    def test_prefers_low_gear_near_corner(self) -> None:
        """Near a corner, heuristic should prefer lower gears."""
        agent = HeuristicAgent()
        track = _corner_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 2  # 1 space before corner start
        player.gear = 2

        legal = [(1, 0), (2, 0), (3, 0)]
        gear, _ = agent.choose_gear(state, 0, legal)
        # Should prefer gear 1 or 2 near a corner, not 3
        assert gear <= 2

    def test_prefers_high_gear_on_straight(self) -> None:
        """On a straight, heuristic should prefer higher gears."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.gear = 2

        legal = [(1, 0), (2, 0), (3, 0), (4, 1)]
        gear, _ = agent.choose_gear(state, 0, legal)
        assert gear >= 3

    def test_prefers_gear1_when_heat_in_hand(self) -> None:
        """With heat in hand, should prefer gear 1 for cooldown."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.gear = 1
        # Replace hand with 3 heat cards + 4 speed cards
        player.hand = [
            Card(CardType.HEAT, 0, f"h{i}") for i in range(3)
        ] + [
            Card(CardType.SPEED, v, f"s{v}") for v in [2, 3, 4, 5]
        ]

        legal = [(1, 0), (2, 0)]
        gear, _ = agent.choose_gear(state, 0, legal)
        assert gear == 1  # cooldown bonus should win

    def test_avoids_gear_when_not_enough_playable(self) -> None:
        """Should skip gears that would clutter."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.gear = 1
        # Only 2 playable cards
        player.hand = [
            Card(CardType.SPEED, 3, "s3"),
            Card(CardType.SPEED, 4, "s4"),
        ] + [Card(CardType.HEAT, 0, f"h{i}") for i in range(5)]

        legal = [(1, 0), (2, 0), (3, 0)]
        gear, _ = agent.choose_gear(state, 0, legal)
        assert gear <= 2  # can't play 3 cards

    def test_always_returns_legal(self) -> None:
        """Result must always be in legal_gears."""
        agent = HeuristicAgent()
        state = _make_game_state()
        legal = [(2, 0), (3, 0)]
        choice = agent.choose_gear(state, 0, legal)
        assert choice in legal


class TestHeuristicChooseCards:
    def test_prefers_higher_speed_on_straight(self) -> None:
        """On a straight, should prefer higher total speed."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.gear = 1

        c_low = Card(CardType.SPEED, 1, "s1")
        c_high = Card(CardType.SPEED, 6, "s6")
        legal = [(c_low,), (c_high,)]
        choice = agent.choose_cards(state, 0, legal)
        assert choice == (c_high,)

    def test_prefers_lower_speed_near_corner(self) -> None:
        """Near a corner with low speed limit, should prefer low speed."""
        agent = HeuristicAgent()
        track = _corner_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 2  # near corner (starts at 3, limit=3)
        player.gear = 1
        player.heat_pool = [Card(CardType.HEAT, 0, f"hp{i}") for i in range(2)]

        c_low = Card(CardType.SPEED, 2, "s2")
        c_high = Card(CardType.SPEED, 6, "s6")
        legal = [(c_low,), (c_high,)]
        choice = agent.choose_cards(state, 0, legal)
        assert choice == (c_low,)  # 6 would overshoot corner limit

    def test_single_option(self) -> None:
        """With only one option, should return it."""
        agent = HeuristicAgent()
        state = _make_game_state()
        c1 = Card(CardType.SPEED, 3, "s3")
        legal = [(c1,)]
        assert agent.choose_cards(state, 0, legal) == (c1,)


class TestHeuristicChooseReact:
    def test_cooldown_max_heat_in_hand(self) -> None:
        """Should cool as many heat cards as possible from hand."""
        agent = HeuristicAgent()
        state = _make_game_state()
        player = state.get_player(0)
        # Put 2 heat cards in hand
        player.hand = [
            Card(CardType.HEAT, 0, "h0"),
            Card(CardType.HEAT, 0, "h1"),
            Card(CardType.SPEED, 3, "s3"),
        ]

        decision = agent.choose_react(state, 0, max_cooldown=3, can_boost=False, has_adrenaline=False)
        assert decision.cooldown_count == 2  # 2 heat in hand, max=3, so cool 2

    def test_no_boost_when_low_heat(self) -> None:
        """Should not boost when heat is low and far from finish."""
        agent = HeuristicAgent()
        track = _straight_track(laps=2)  # long track, far from finish
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.heat_pool = [Card(CardType.HEAT, 0, "hp0")]  # only 1 heat

        decision = agent.choose_react(state, 0, max_cooldown=0, can_boost=True, has_adrenaline=False)
        assert decision.use_boost is False

    def test_boost_when_plenty_of_heat(self) -> None:
        """Should boost when heat is ample."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5
        player.heat_pool = [Card(CardType.HEAT, 0, f"hp{i}") for i in range(5)]
        player.speed_from_cards = 3

        decision = agent.choose_react(state, 0, max_cooldown=0, can_boost=True, has_adrenaline=False)
        assert decision.use_boost is True


class TestHeuristicChooseSlipstream:
    def test_takes_free_slipstream(self) -> None:
        """Should take slipstream when no corners crossed."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 5

        assert agent.choose_slipstream(state, 0) is True

    def test_declines_when_would_spinout(self) -> None:
        """Should decline slipstream that would cause expensive corner."""
        agent = HeuristicAgent()
        track = _corner_track()  # corner at 3-5, limit 3
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 2  # +2 = 4, crosses into corner 3-5
        player.speed_from_cards = 6  # way over speed limit
        player.heat_pool = []  # no heat to pay

        assert agent.choose_slipstream(state, 0) is False


class TestHeuristicCornerAvoidance:
    def test_no_boost_when_over_corner_limit(self):
        """Should not boost when speed already exceeds corner limit."""
        agent = HeuristicAgent()
        track = _corner_track()  # corner at 3-5, limit 3
        state = _make_game_state(track=track)
        player = state.get_player(0)
        # Player started before the corner and is now past it
        player.turn_start_position = 1
        player.position = 5  # inside the corner
        player.gear = 2
        player.speed_from_cards = 4  # exceeds corner limit of 3
        player.heat_pool = [Card(CardType.HEAT, 0, f"hp{i}") for i in range(5)]

        decision = agent.choose_react(
            state, 0, max_cooldown=0, can_boost=True, has_adrenaline=False
        )
        assert decision.use_boost is False

    def test_no_boost_when_heat_critically_low(self):
        """Should not boost when heat <= 2."""
        agent = HeuristicAgent()
        track = _straight_track()
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.turn_start_position = 3
        player.position = 5
        player.speed_from_cards = 2
        player.heat_pool = [Card(CardType.HEAT, 0, f"hp{i}") for i in range(2)]

        decision = agent.choose_react(
            state, 0, max_cooldown=0, can_boost=True, has_adrenaline=False
        )
        assert decision.use_boost is False

    def test_choose_cards_avoids_spinout_combo(self):
        """Should prefer low-speed combo when high-speed would cause spinout."""
        agent = HeuristicAgent()
        track = _corner_track()  # corner at 3-5, limit 3
        state = _make_game_state(track=track)
        player = state.get_player(0)
        player.position = 2  # will cross corner
        player.gear = 2
        player.heat_pool = [Card(CardType.HEAT, 0, "hp0")]  # only 1 heat

        c1 = Card(CardType.SPEED, 1, "s1")
        c2 = Card(CardType.SPEED, 2, "s2")
        c3 = Card(CardType.SPEED, 3, "s3")
        c4 = Card(CardType.SPEED, 4, "s4")

        # Low combo: speed=3 (within limit), high combo: speed=7 (way over)
        legal = [(c1, c2), (c3, c4)]
        choice = agent.choose_cards(state, 0, legal)
        assert choice == (c1, c2)


class TestHeuristicChooseDiscard:
    def test_returns_empty_when_few_playable(self) -> None:
        """Should keep cards when few playable in hand."""
        agent = HeuristicAgent()
        state = _make_game_state()
        player = state.get_player(0)
        player.gear = 2
        player.hand = [
            Card(CardType.SPEED, 1, "s1"),
            Card(CardType.SPEED, 2, "s2"),
            Card(CardType.SPEED, 3, "s3"),
            Card(CardType.HEAT, 0, "h0"),
        ]
        discardable = [
            Card(CardType.SPEED, 1, "s1"),
            Card(CardType.SPEED, 2, "s2"),
            Card(CardType.SPEED, 3, "s3"),
        ]
        result = agent.choose_discard(state, 0, discardable)
        assert result == []  # 3 playable <= gear(2) + 2 = 4

    def test_discards_value1_when_plenty(self) -> None:
        """Should discard value-1 speed cards when plenty of playable cards."""
        agent = HeuristicAgent()
        state = _make_game_state()
        player = state.get_player(0)
        player.gear = 1
        # 7 playable cards, gear=1, 7 > 1+2=3 so plenty
        player.hand = [
            Card(CardType.SPEED, 1, "s1a"),
            Card(CardType.SPEED, 1, "s1b"),
            Card(CardType.SPEED, 3, "s3"),
            Card(CardType.SPEED, 4, "s4"),
            Card(CardType.SPEED, 5, "s5"),
            Card(CardType.SPEED, 6, "s6a"),
            Card(CardType.SPEED, 6, "s6b"),
        ]
        discardable = list(player.hand)
        result = agent.choose_discard(state, 0, discardable)
        # Should discard the value-1 cards
        assert len(result) == 2
        assert all(c.value == 1 for c in result)

    def test_never_discards_upgrade(self) -> None:
        """Should never discard upgrade cards."""
        agent = HeuristicAgent()
        state = _make_game_state()
        player = state.get_player(0)
        player.gear = 1
        player.hand = [
            Card(CardType.UPGRADE, 5, "u5"),
            Card(CardType.SPEED, 1, "s1"),
            Card(CardType.SPEED, 2, "s2"),
            Card(CardType.SPEED, 3, "s3"),
            Card(CardType.SPEED, 4, "s4"),
            Card(CardType.SPEED, 5, "s5"),
            Card(CardType.SPEED, 6, "s6"),
        ]
        discardable = list(player.hand)
        result = agent.choose_discard(state, 0, discardable)
        assert all(c.card_type != CardType.UPGRADE for c in result)
