"""Tests for the Sprint C2 AlphaZero policy+value trainer (``train_az.py``).

Covers exactly the contract-critical guarantees the C2 design's Deliverables flag:

  7. the AZ checkpoint **loads as a ``MaskablePPO``**, carries the ``.meta.json``
     contract sidecar, and round-trips through ``MLAgent`` returning only legal
     moves (the full MLAgent / NetAdapter load path the search prior takes).
  8. the **value head is actually trained** -- unlike S3 BC (which left the critic
     uninitialized), the critic is a first-class target, so its output moves from
     init on a held batch; AND the actor trunk/head also moves (AZ trains both).
  9. the soft-policy CE (``_policy_ce``) and masked-logprob path
     (``_masked_log_probs``) behave correctly: CE is finite with zero mass on
     illegal actions, and a perfect prediction gives strictly lower CE than a wrong
     one.

Mirrors ``tests/test_value_net.py`` (the experiment-script import + tiny-fixture +
``_same`` state-dict comparison style). A tiny self-play dataset + a few-epoch CPU
train keeps the whole file fast (seconds), with no committed data.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest
import torch

# The experiments scripts are not a package; add the dir to the path so the
# Sprint-C2 deliverables can be imported by their module name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_selfplay as G  # noqa: E402
import train_az  # noqa: E402

from heat.ml.env import HeatEnv  # noqa: E402
from heat.ml.model import PPOConfig, build_model  # noqa: E402
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny dataset + checkpoint fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """A fresh random-weight codec-v3 ``MaskablePPO`` checkpoint (the generator's
    pre-training prior)."""
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c2prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


@pytest.fixture(scope="module")
def tiny_data(cold_prior, tmp_path_factory) -> str:
    """Generate a tiny self-play dataset once (a few tracks, small sims)."""
    out = str(tmp_path_factory.mktemp("selfplay") / "sp.npz")
    args = argparse.Namespace(
        out=out, model=cold_prior, tracks=4, val_tracks=2, sims=8,
        dirichlet_eps=0.25, dirichlet_alpha=0.5, temperature_moves=10,
        seed=0, game_seed=8000,
    )
    G.generate_dataset(args)
    return out


def _train_args(data, out, **over):
    base = dict(
        data=data, out=out, epochs=4, batch=64, lr=3e-4, c_v=1.0,
        weight_decay=1e-4, eval_every=2, patience=8, net="default",
        device="cpu", seed=0,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def az_checkpoint(tiny_data, tmp_path_factory) -> str:
    """Train a tiny few-epoch AZ checkpoint on the tiny dataset."""
    out = str(tmp_path_factory.mktemp("az_ckpt") / "az.zip")
    train_az.train_az(_train_args(tiny_data, out))
    return out


# ---------------------------------------------------------------------------
# 7. checkpoint loads as MaskablePPO + contract sidecar + MLAgent round-trip
# ---------------------------------------------------------------------------


class TestCheckpointContract:
    def test_loads_as_maskable_ppo(self, az_checkpoint) -> None:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(az_checkpoint, device="cpu")
        assert model.observation_space.shape == (OBS_DIM,)
        assert model.action_space.n == ACTION_DIM

    def test_has_contract_sidecar(self, az_checkpoint) -> None:
        from heat.ml.training import load_meta

        meta = load_meta(az_checkpoint)
        assert meta["obs_dim"] == OBS_DIM
        assert meta["action_dim"] == ACTION_DIM
        assert meta["codec_version"] == CODEC_VERSION
        assert meta["num_players"] == 1
        assert meta["track_name"] == "generated-az"

    def test_loads_under_ml_agent_tripwire(self, az_checkpoint) -> None:
        """The §3.4 sidecar tripwire (obs_dim/action_dim/codec_version) passes --
        the exact validation the MCTS prior's NetAdapter load performs."""
        from heat.agents.ml_agent import MLAgent

        agent = MLAgent(az_checkpoint, name="AZ")
        agent._validate_meta()  # raises CheckpointMismatchError on any drift

    def test_roundtrips_through_ml_agent_returning_legal_moves(self, az_checkpoint) -> None:
        """The AZ checkpoint drives an MLAgent that returns only legal moves: a
        full solo game completes (the engine raises on any illegal move)."""
        from heat.agents.ml_agent import MLAgent
        from heat.engine.game import Game
        from heat.tracks.generator import generate_track, TrackGenParams

        params = TrackGenParams(
            num_corners_range=(4, 7), speed_limit_choices=(1, 1, 2, 3), laps=2
        )
        track = generate_track(900_222, params)
        agent = MLAgent(az_checkpoint, deterministic=True, name="AZ")
        result = Game(track, [agent], logging_enabled=False, seed=0).run()
        assert result.finish_order == [0]


# ---------------------------------------------------------------------------
# 8. the value head is actually trained (critic moved AND actor moved)
# ---------------------------------------------------------------------------


class TestValueHeadTrained:
    def test_critic_and_actor_both_move_from_init(self, tiny_data, tmp_path) -> None:
        """AZ trains BOTH the critic and the actor (unlike S3 BC which left the
        critic uninitialized): both the value head and the actor head must differ
        from the same-seed fresh init the trainer started from.
        """
        out = str(tmp_path / "az.zip")
        # A few high-lr epochs so the move is unmistakable.
        train_az.train_az(_train_args(tiny_data, out, epochs=4, lr=1e-2))

        from sb3_contrib import MaskablePPO

        trained = MaskablePPO.load(out, device="cpu").policy
        # The init the trainer started from: same seed/config (the build_model call
        # in train_az uses share_features_extractor=False).
        fresh = build_model(
            HeatEnv(num_players=1),
            PPOConfig(device="cpu", seed=0, share_features_extractor=False),
        ).policy

        def _same(a, b) -> bool:
            sa, sb = a.state_dict(), b.state_dict()
            return all(torch.equal(sa[k], sb[k]) for k in sa)

        # The critic head must have moved (V is a first-class AZ target).
        assert not _same(trained.value_net, fresh.value_net), (
            "the critic head did not train -- the value target is inert (the S3 "
            "BC footgun)"
        )
        # The actor head must also have moved (the soft-policy CE term).
        assert not _same(trained.action_net, fresh.action_net), (
            "the actor head did not train -- the policy CE term is inert"
        )

    def test_value_output_moves_on_a_held_batch(self, tiny_data, cold_prior) -> None:
        """The trained critic's output on a held batch differs from the pre-training
        (cold) net's output -- the value head learned a signal, not a constant."""
        out_dir = os.path.dirname(cold_prior)
        out = os.path.join(out_dir, "az_held.zip")
        train_az.train_az(_train_args(tiny_data, out, epochs=4, lr=1e-2))

        from sb3_contrib import MaskablePPO

        data = np.load(tiny_data)
        obs = torch.as_tensor(data["obs"][:64].astype(np.float32))

        cold = MaskablePPO.load(cold_prior, device="cpu").policy
        trained = MaskablePPO.load(out, device="cpu").policy
        with torch.no_grad():
            v_cold = cold.predict_values(obs).reshape(-1)
            v_trained = trained.predict_values(obs).reshape(-1)
        assert torch.isfinite(v_trained).all()
        # The trained value head must have moved away from the random init.
        assert not torch.allclose(v_cold, v_trained, atol=1e-4), (
            "the value head output is unchanged from init -- the critic did not "
            "learn the MC return"
        )


# ---------------------------------------------------------------------------
# 9. soft-policy CE + masked-logprob path behave correctly
# ---------------------------------------------------------------------------


class TestPolicyLossPath:
    @pytest.fixture(scope="class")
    def policy(self):
        """A freshly built MaskablePPO policy on CPU (random weights are fine: we
        check the loss arithmetic, not a trained number)."""
        model = build_model(
            HeatEnv(num_players=1),
            PPOConfig(device="cpu", seed=0, share_features_extractor=False),
        )
        return model.policy

    def _one_sample(self, n_legal: int = 4):
        """A single (obs, mask) sample with the first ``n_legal`` actions legal."""
        rng = np.random.default_rng(0)
        obs = torch.as_tensor(
            rng.uniform(-1, 1, size=(1, OBS_DIM)).astype(np.float32)
        )
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, :n_legal] = True
        return obs, mask

    def test_masked_log_probs_zero_mass_on_illegal(self, policy) -> None:
        """``_masked_log_probs`` puts (effectively) zero probability mass on illegal
        actions and a proper distribution over the legal set.

        The masked categorical (the exact ``MaskablePPO.predict`` path) drives
        illegal logits to a large negative sentinel before the log-softmax, so the
        illegal log-probs are far below every legal one and exponentiate to ~0 (the
        masked softmax) -- NOT necessarily a literal ``-inf``. The contract
        ``_policy_ce`` relies on is exactly this: zero mass on illegal actions, and
        the legal probabilities summing to 1.
        """
        obs, mask = self._one_sample(n_legal=4)
        with torch.no_grad():
            log_probs = train_az._masked_log_probs(policy, obs, mask)
        lp = log_probs[0]
        illegal_lp = lp[~mask[0]]
        legal_lp = lp[mask[0]]
        # Illegal actions carry ~zero probability mass (exp(log p) ~ 0) and sit far
        # below every legal log-prob.
        assert float(illegal_lp.exp().sum()) < 1e-6
        assert float(illegal_lp.max()) < float(legal_lp.min()) - 10.0
        # Legal actions: finite, and the legal probabilities sum to 1.
        assert torch.isfinite(legal_lp).all()
        assert float(legal_lp.exp().sum()) == pytest.approx(1.0, abs=1e-5)

    def test_policy_ce_is_finite_with_zero_mass_on_illegal(self, policy) -> None:
        """CE is finite when ``pi`` has zero mass on illegal actions (the 0*-inf
        product is handled as 0), and is non-negative (a proper cross-entropy)."""
        obs, mask = self._one_sample(n_legal=4)
        with torch.no_grad():
            log_probs = train_az._masked_log_probs(policy, obs, mask)
        # A visit-distribution-like pi over the legal set, zero on illegal.
        pi = torch.zeros((1, ACTION_DIM))
        pi[0, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1])
        ce = train_az._policy_ce(log_probs, pi)
        assert torch.isfinite(ce).all(), "CE is non-finite (0*-inf not handled)"
        assert float(ce[0]) >= 0.0

    def test_perfect_prediction_has_lower_ce_than_wrong(self, policy) -> None:
        """A pi that matches the policy's own softmax has strictly lower CE than a
        pi concentrated on the policy's least-likely legal action (CE is minimized
        at the prediction)."""
        obs, mask = self._one_sample(n_legal=4)
        with torch.no_grad():
            log_probs = train_az._masked_log_probs(policy, obs, mask)
        legal = mask[0]
        probs = log_probs[0].exp()

        # pi == the policy's own distribution over the legal set (the "perfect"
        # target for THIS net: CE then equals the policy entropy, its minimum).
        pi_perfect = torch.zeros((1, ACTION_DIM))
        pi_perfect[0, legal] = probs[legal]
        pi_perfect = pi_perfect / pi_perfect.sum()

        # pi == one-hot on the policy's LEAST-likely legal action (a wrong target).
        legal_idx = torch.nonzero(legal, as_tuple=False).reshape(-1)
        worst = legal_idx[torch.argmin(probs[legal_idx])]
        pi_wrong = torch.zeros((1, ACTION_DIM))
        pi_wrong[0, worst] = 1.0

        ce_perfect = float(train_az._policy_ce(log_probs, pi_perfect)[0])
        ce_wrong = float(train_az._policy_ce(log_probs, pi_wrong)[0])
        assert ce_perfect < ce_wrong, (
            "matching the prediction must give lower CE than a wrong one-hot target"
        )
