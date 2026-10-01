"""Outcome-only evaluation preserves race and random state without rollout storage."""
import copy

import numpy as np
import pytest
import torch

from heat.agents.heuristic_agent import HeuristicAgent
from heat.ml.selfplay import multiseat
from heat.ml.selfplay.eval_harness import _EvalCollector
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.selfplay.semantic_contract import canonical_game_state
from heat.tracks.generator import generate_track


class AlternatingOpponent(HeuristicAgent):
    """Use per-game state so parity covers stateful opponents too."""

    def __init__(self):
        super().__init__()
        self.gear_calls = 0

    def choose_gear(self, state, player_id, legal):
        self.gear_calls += 1
        return legal[self.gear_calls % len(legal)]


@pytest.mark.parametrize("seats", [2, 6])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("device_name", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_outcome_only_matches_collector(seats, truncated, device_name, monkeypatch):
    """Match the old path at normal and time-limit endings, including sampled bootstraps."""
    if truncated:
        monkeypatch.setattr(multiseat, "MAX_ROUNDS", 2)
    device = torch.device(device_name)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        policy = build_policy(A0Config(hidden_sizes=(16,))).to(device)
    track = generate_track(1415000)
    opponents = {seat: AlternatingOpponent() for seat in range(1, seats)}

    def play(outcome_only):
        agents = copy.deepcopy(opponents)
        collector = _EvalCollector(track, seats, scripted_seats=agents)
        rng = np.random.default_rng(2000)
        with torch.random.fork_rng():
            torch.manual_seed(9000)
            if outcome_only:
                state = collector.play_game(policy, device, rng)
            else:
                collector.collect(policy, 1, device, rng, gamma=1.0)
                state = collector.last_state
            random_state = torch.get_rng_state().clone()
            cuda_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
        assert state is not None
        assert collector._episode_flags(state) == (not truncated, truncated)
        return (canonical_game_state(state), random_state, cuda_state,
                rng.bit_generator.state, [agent.gear_calls for agent in agents.values()])

    expected = play(False)

    def storage_forbidden(*args, **kwargs):
        pytest.fail("outcome evaluation must not allocate or fill a rollout buffer")

    monkeypatch.setattr(multiseat, "RolloutBuffer", storage_forbidden)
    monkeypatch.setattr(_EvalCollector, "_store", storage_forbidden)
    actual = play(True)
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])
    if expected[2] is not None:
        assert torch.equal(actual[2], expected[2])
    assert actual[3:] == expected[3:]
