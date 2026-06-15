"""Tests for GameState-owned RNG and seed reproducibility (Step A2)."""

from __future__ import annotations

import random

from heat.models.game_state import GameState
from heat.models.track import Space, Track


def _track(laps: int = 1) -> Track:
    spaces = [Space(index=i, lanes=2) for i in range(12)]
    return Track(
        name="rng-test",
        spaces=spaces,
        corners=[],
        laps=laps,
        start_positions=[0, 1, 2, 3],
    )


def _deck_orders(state: GameState) -> list[list[str]]:
    return [[c.id for c in p.deck] for p in state.players]


def _hands(state: GameState) -> list[list[str]]:
    return [[c.id for c in p.hand] for p in state.players]


class TestSeedReproducibility:
    def test_same_seed_same_deck_order(self) -> None:
        s1 = GameState.create(_track(), 4, seed=42)
        s2 = GameState.create(_track(), 4, seed=42)
        assert _deck_orders(s1) == _deck_orders(s2)

    def test_same_seed_same_hands(self) -> None:
        s1 = GameState.create(_track(), 4, seed=42)
        s2 = GameState.create(_track(), 4, seed=42)
        assert _hands(s1) == _hands(s2)

    def test_different_seed_different_order(self) -> None:
        s1 = GameState.create(_track(), 4, seed=1)
        s2 = GameState.create(_track(), 4, seed=2)
        assert _deck_orders(s1) != _deck_orders(s2)

    def test_deck_order_independent_of_global_random(self) -> None:
        """With an explicit seed, the global random state must not matter."""
        random.seed(11111)
        s1 = GameState.create(_track(), 4, seed=777)
        random.seed(22222)
        s2 = GameState.create(_track(), 4, seed=777)
        assert _deck_orders(s1) == _deck_orders(s2)
        assert _hands(s1) == _hands(s2)


class TestBackCompatGlobalSeed:
    def test_seed_none_forks_from_global(self) -> None:
        """random.seed(X); create(seed=None) is reproducible across runs."""
        random.seed(999)
        s1 = GameState.create(_track(), 4)
        random.seed(999)
        s2 = GameState.create(_track(), 4)
        assert _deck_orders(s1) == _deck_orders(s2)
        assert _hands(s1) == _hands(s2)

    def test_seed_none_different_global_differs(self) -> None:
        random.seed(1)
        s1 = GameState.create(_track(), 4)
        random.seed(2)
        s2 = GameState.create(_track(), 4)
        assert _deck_orders(s1) != _deck_orders(s2)


class TestRngOwnership:
    def test_player_decks_share_game_rng(self) -> None:
        state = GameState.create(_track(), 4, seed=5)
        for p in state.players:
            assert p.deck._rng is state.rng
