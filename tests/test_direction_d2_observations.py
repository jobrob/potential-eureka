"""Chunk-2 observation and GEAR/CARDS mask parity for Direction D2."""

from __future__ import annotations

import numpy as np

from heat.engine import rules
from heat.engine.driver import (
    Decision,
    DecisionKind,
    run_round_driver,
    simultaneous_decisions,
)
from heat.engine.phases import phase_shift_gears
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.features import encode_observation
from heat.ml.vector_env.bridge import legacy_states_to_tensor
from heat.ml.vector_env.observations import (
    TensorDecisionBatch,
    TensorDecisionKind,
    tensor_legal_action_masks,
    tensor_observations,
)
from heat.tracks.generator import TrackGenParams, generate_track, track_sampler


def _tensor_kind(kind: DecisionKind) -> TensorDecisionKind:
    """Map the two legacy decision kinds implemented by Chunk 2."""
    if kind is DecisionKind.GEAR:
        return TensorDecisionKind.GEAR
    if kind is DecisionKind.CARDS:
        return TensorDecisionKind.CARDS
    raise ValueError(f"unsupported Chunk-2 decision {kind.value}")


def _assert_parity(
    state: GameState, decisions: list[Decision], *, game_id: int
) -> None:
    """Compare tensor outputs with the scalar encoder and action codec exactly."""
    tensor_state = legacy_states_to_tensor([state], game_ids=[game_id])
    tensor_decisions = TensorDecisionBatch.create(
        [game_id] * len(decisions),
        [decision.player_id for decision in decisions],
        [_tensor_kind(decision.kind) for decision in decisions],
    )
    actual_observations = tensor_observations(tensor_state, tensor_decisions).numpy()
    actual_masks = tensor_legal_action_masks(tensor_state, tensor_decisions).numpy()
    expected_observations = np.stack(
        [
            encode_observation(state, decision.player_id, decision)
            for decision in decisions
        ]
    )
    expected_masks = np.stack(
        [legal_action_mask(decision, state) for decision in decisions]
    )
    np.testing.assert_array_equal(actual_observations, expected_observations)
    np.testing.assert_array_equal(actual_masks, expected_masks)


def _random_legal_action(
    state: GameState, decision: Decision, rng: np.random.Generator
) -> object:
    """Choose a seeded legal action while retaining empty-mask forced actions."""
    forced = forced_action(decision, state)
    if forced is not NO_FORCED:
        return forced
    legal_indices = np.flatnonzero(legal_action_mask(decision, state))
    return decode_legal_action(decision, state, int(rng.choice(legal_indices)))


def test_d0_coordinates_match_for_gear_and_cards_across_two_to_six_seats() -> None:
    """The frozen D0 seeds produce exactly equal batched rows at both phases."""
    sampler = track_sampler(TrackGenParams(), base_seed=80_008)
    for seats in range(2, 7):
        seed = 19_000 + seats * 100
        state = GameState.create(sampler(seed), seats, seed=seed)
        for player in state.players:
            player.lap = 1

        gear_decisions = simultaneous_decisions(state, DecisionKind.GEAR)
        _assert_parity(state, gear_decisions, game_id=seats)
        phase_shift_gears(
            state,
            {
                decision.player_id: decision.legal[0]  # type: ignore[index]
                for decision in gear_decisions
            },
        )
        card_decisions = simultaneous_decisions(state, DecisionKind.CARDS)
        _assert_parity(state, card_decisions, game_id=seats)


def test_randomized_generated_states_match_through_three_rounds() -> None:
    """Seeded varied actions exercise changing hands, heat, positions, and ranks."""
    rng = np.random.default_rng(2_026_0720)
    for game_index, seed in enumerate((31_101, 31_102, 31_103, 31_104, 31_105)):
        seats = 2 + game_index
        state = GameState.create(generate_track(seed), seats, seed=seed + 90_000)
        for player in state.players:
            player.lap = 1

        for _round in range(3):
            driver = run_round_driver(state)
            send_value: object = None
            while True:
                try:
                    decision = driver.send(send_value)
                except StopIteration:
                    break
                if decision.kind in (DecisionKind.GEAR, DecisionKind.CARDS):
                    _assert_parity(state, [decision], game_id=seed)
                send_value = _random_legal_action(state, decision, rng)


def test_card_masks_match_for_forced_heat_and_empty_hands() -> None:
    """Cluttered and degenerate hands retain the codec's exact forced semantics."""
    state = GameState.create(generate_track(31_106), 2, seed=121_106)
    for player in state.players:
        player.lap = 1
    player = state.players[0]
    player.gear = 4

    player.hand = [
        Card(CardType.HEAT, 0, f"forced-heat-{index}") for index in range(7)
    ]
    forced_heat = Decision(
        DecisionKind.CARDS,
        player.player_id,
        rules.legal_card_plays(player.hand, player.gear),
    )
    _assert_parity(state, [forced_heat], game_id=31_106)

    player.hand = []
    empty_hand = Decision(
        DecisionKind.CARDS,
        player.player_id,
        rules.legal_card_plays(player.hand, player.gear),
    )
    _assert_parity(state, [empty_hand], game_id=31_106)
    assert not legal_action_mask(empty_hand, state).any()


def test_tensor_outputs_are_independent_of_lane_order_and_batch_width() -> None:
    """Stable identities isolate a decision from padding and neighboring lanes."""
    focal = GameState.create(generate_track(31_107), 4, seed=121_107)
    for player in focal.players:
        player.lap = 1
    decision = simultaneous_decisions(focal, DecisionKind.GEAR)[2]

    baseline_state = legacy_states_to_tensor([focal], game_ids=[7])
    baseline_decision = TensorDecisionBatch.create(
        [7], [decision.player_id], [TensorDecisionKind.GEAR]
    )
    expected_observation = tensor_observations(baseline_state, baseline_decision)
    expected_mask = tensor_legal_action_masks(baseline_state, baseline_decision)

    for width in (1, 8, 32, 256):
        game_ids = list(range(1_000, 1_000 + width - 1)) + [7]
        states = [focal] * width
        wide_state = legacy_states_to_tensor(states, game_ids=game_ids)
        wide_decision = TensorDecisionBatch.create(
            [7], [decision.player_id], [TensorDecisionKind.GEAR]
        )
        assert np.array_equal(
            tensor_observations(wide_state, wide_decision).numpy(),
            expected_observation.numpy(),
        )
        assert np.array_equal(
            tensor_legal_action_masks(wide_state, wide_decision).numpy(),
            expected_mask.numpy(),
        )

    neighbors = [
        GameState.create(generate_track(31_108), 2, seed=121_108),
        GameState.create(generate_track(31_109), 6, seed=121_109),
    ]
    for state in neighbors:
        for player in state.players:
            player.lap = 1
    for states, game_ids in (
        ([neighbors[0], focal, neighbors[1]], [8, 7, 9]),
        ([neighbors[1], neighbors[0], focal], [9, 8, 7]),
    ):
        reordered = legacy_states_to_tensor(states, game_ids=game_ids)
        reordered_decision = TensorDecisionBatch.create(
            [7], [decision.player_id], [TensorDecisionKind.GEAR]
        )
        assert np.array_equal(
            tensor_observations(reordered, reordered_decision).numpy(),
            expected_observation.numpy(),
        )
        assert np.array_equal(
            tensor_legal_action_masks(reordered, reordered_decision).numpy(),
            expected_mask.numpy(),
        )
