"""Native observation, mask, reward, and receipt gates for D3 chunk A4."""

from __future__ import annotations

import random

import numpy as np

from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision
from heat.ml.action_codec import legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.native_env.bridge import NativeStateBridge, legacy_state_to_native
from heat.ml.spaces import step_reward, terminal_margin
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _state(seed: int, seats: int) -> GameState:
    """Create a deterministic started game with meaningful lap features."""
    state = GameState.create(generate_track(seed), seats, seed=seed + 60_000)
    for player in state.players:
        player.lap = 1
    return state


def _choose(decision: Decision, rng: random.Random) -> object:
    """Choose a reproducible legal decoded action for each decision kind."""
    if decision.kind in (DecisionKind.GEAR, DecisionKind.CARDS):
        legal = list(decision.legal)  # type: ignore[arg-type]
        return legal[rng.randrange(len(legal))]
    if decision.kind is DecisionKind.REACT:
        options = decision.legal
        adrenaline_cooldown = bool(options.has_adrenaline and rng.randrange(2))
        return ReactDecision(
            cooldown_count=rng.randrange(
                options.max_cooldown + int(adrenaline_cooldown) + 1
            ),
            use_boost=bool(options.can_boost and rng.randrange(2)),
            use_adrenaline_speed=bool(
                options.has_adrenaline and rng.randrange(2)
            ),
            use_adrenaline_cooldown=adrenaline_cooldown,
        )
    if decision.kind is DecisionKind.SLIPSTREAM:
        return bool(rng.randrange(2))
    if decision.kind is DecisionKind.DISCARD:
        cards = list(decision.legal)  # type: ignore[arg-type]
        return cards[: rng.randrange(len(cards) + 1)]
    raise AssertionError(decision.kind)


def _assert_row(
    native: NativeStateBridge, state: GameState, decision: Decision
) -> None:
    """Require native policy/value arrays to be byte-identical to Python."""
    np.testing.assert_array_equal(
        native.observation(decision.player_id, decision),
        encode_observation(state, decision.player_id, decision),
    )
    np.testing.assert_array_equal(
        native.observation(decision.player_id, None),
        encode_observation(state, decision.player_id, None),
    )
    np.testing.assert_array_equal(
        native.legal_mask(decision), legal_action_mask(decision, state)
    )


def _submit(
    native: NativeStateBridge, decision: Decision, action: object
) -> dict[str, object]:
    """Submit a sequential action to the native runner."""
    if decision.kind is DecisionKind.REACT:
        assert isinstance(action, ReactDecision)
        return native.apply_react(decision.player_id, action)
    if decision.kind is DecisionKind.SLIPSTREAM:
        assert isinstance(action, bool)
        return native.apply_slipstream(decision.player_id, action)
    if decision.kind is DecisionKind.DISCARD:
        assert isinstance(action, list)
        assert all(isinstance(card, Card) for card in action)
        return native.apply_discard(decision.player_id, action)
    raise AssertionError(decision.kind)


def _run_checked_round(seed: int, seats: int) -> int:
    """Compare every policy row and reward boundary through one complete round."""
    state = _state(seed, seats)
    native = legacy_state_to_native(state, game_id=80_000 + seed)
    rng = random.Random(seed + 90_000)
    native.start_round()
    driver = run_round_driver(state)
    decision = next(driver)
    checked = 0

    gear_choices: dict[int, tuple[int, int]] = {}
    while decision.kind is DecisionKind.GEAR:
        _assert_row(native, state, decision)
        checked += 1
        action = _choose(decision, rng)
        assert isinstance(action, tuple)
        gear_choices[decision.player_id] = action
        decision = driver.send(action)
    native.apply_gears(gear_choices)

    card_choices: dict[int, tuple[Card, ...]] = {}
    while decision.kind is DecisionKind.CARDS:
        _assert_row(native, state, decision)
        checked += 1
        action = _choose(decision, rng)
        assert isinstance(action, tuple)
        card_choices[decision.player_id] = action
        decision = driver.send(action)
    native.apply_cards_to_react(card_choices)

    while True:
        _assert_row(native, state, decision)
        checked += 1
        previous = state.clone(reseed=0)
        action = _choose(decision, rng)
        try:
            next_decision = driver.send(action)
        except StopIteration:
            _submit(native, decision, action)
            next_decision = None
        else:
            _submit(native, decision, action)
        expected = step_reward(
            previous,
            state,
            decision.player_id,
            False,
            shaping_weight=0.013,
            spinout_weight=0.021,
        )
        assert native.reward(
            previous,
            decision.player_id,
            False,
            shaping_weight=0.013,
            spinout_weight=0.021,
        ) == expected
        receipt = native.boundary_receipt()
        assert receipt.shape == (8,)
        assert int(receipt[-1]) == native.canonical_digest()
        if next_decision is None:
            return checked
        decision = next_decision


def test_randomized_boundary_arrays_and_rewards_are_exact() -> None:
    """Generated 2-6 seat rounds compare every native A4 output byte."""
    checked = sum(
        _run_checked_round(5_000 + case, 2 + case % 5) for case in range(30)
    )
    assert checked >= 250


def test_bootstrap_rows_are_value_only_and_exact() -> None:
    """Truncation rows carry a state-pure observation and no legal action."""
    state = _state(5_100, 4)
    native = legacy_state_to_native(state, game_id=81_000)
    for player in state.players:
        row = native.bootstrap_row(player.player_id)
        assert row["value_only"] is True
        np.testing.assert_array_equal(
            row["observation"], encode_observation(state, player.player_id, None)
        )
        assert not np.asarray(row["legal_mask"]).any()


def test_terminal_rewards_and_margins_match_all_seats() -> None:
    """Placement, solo finish, and dense terminal margins match the oracle."""
    state = _state(5_200, 4)
    for rank, player in enumerate(state.players, start=1):
        player.position = (4 - rank) * 3
        if rank <= 2:
            player.finished = True
            player.finish_order = rank
            player.lap = state.track.laps + 1
    native = legacy_state_to_native(state, game_id=82_000)
    previous = state.clone(reseed=0)
    for player in state.players:
        assert native.reward(previous, player.player_id, True) == step_reward(
            previous, state, player.player_id, True
        )
        assert native.terminal_margin(player.player_id) == terminal_margin(
            state, player.player_id
        )

    # The frozen native state capacity is 2-6 seats. Reward mode is iteration
    # configuration, so its finish-bonus branch can still be checked directly.
    solo = _state(5_201, 2)
    solo.players[0].finished = True
    solo.players[0].finish_order = 1
    solo_native = legacy_state_to_native(solo, game_id=82_001)
    assert solo_native.reward(
        solo.clone(reseed=0),
        0,
        True,
        terminated=True,
        reward_mode="solo",
    ) == 5.0
