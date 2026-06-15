"""Parity tests for the unified legal-action API (Step C1).

Each new rules.legal_* function is asserted to return exactly what the
original inline expression in game.py produced, across many random states.
"""

from __future__ import annotations

import random

from heat.engine import rules
from heat.models.cards import CardType
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track


def _track(laps: int = 2) -> Track:
    spaces = [Space(index=i, lanes=2) for i in range(20)]
    corners = [Corner(start=5, end=6, speed_limit=3),
               Corner(start=14, end=15, speed_limit=2)]
    return Track(
        name="legal-test",
        spaces=spaces,
        corners=corners,
        start_positions=[0, 1, 2, 3],
        laps=laps,
    )


def _randomize(state: GameState, rng: random.Random) -> None:
    """Perturb players into a varied mid-game configuration."""
    for p in state.players:
        p.gear = rng.randint(1, 4)
        p.position = rng.randint(0, state.track.length - 1)
        p.lap = rng.randint(1, state.track.laps)
        p.boost_used_this_turn = rng.random() < 0.5
        # Vary heat available by paying some heat.
        pay = rng.randint(0, p.heat_available)
        if pay:
            p.pay_heat(pay)
        if rng.random() < 0.2:
            p.finished = True
            p.finish_order = p.player_id + 1


class TestReactParity:
    def test_react_options_match_inline(self) -> None:
        rng = random.Random(1)
        for trial in range(200):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            active = list(state.active_players)
            for player in state.players:
                opts = rules.legal_react_options(
                    player, active, state.starting_player_count
                )
                # Literal recomputation of game.py:362-374.
                exp_max_cooldown = rules.cooldown_amount(player.gear)
                exp_can_boost = (
                    player.heat_available > 0
                    and not player.boost_used_this_turn
                )
                exp_adrenaline = rules.adrenaline_eligible(
                    player, active, state.starting_player_count
                )
                assert opts.max_cooldown == exp_max_cooldown
                assert opts.can_boost == exp_can_boost
                assert opts.has_adrenaline == exp_adrenaline


class TestSlipstreamParity:
    def test_slipstream_matches_inline(self) -> None:
        rng = random.Random(2)
        for trial in range(200):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            active = list(state.active_players)
            for player in state.players:
                got = rules.legal_slipstream(player, active, state.track)
                exp = rules.slipstream_eligible(player, active, state.track)
                assert got == exp


class TestDiscardParity:
    def test_discards_match_inline(self) -> None:
        rng = random.Random(3)
        for trial in range(200):
            state = GameState.create(_track(), 4, seed=trial)
            _randomize(state, rng)
            for player in state.players:
                got = rules.legal_discards(player)
                exp = [
                    c
                    for c in player.hand
                    if c.card_type in (CardType.SPEED, CardType.UPGRADE)
                ]
                assert got == exp
