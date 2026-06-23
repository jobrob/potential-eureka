"""Tests for Sprint C11 -- the lean network forward (equivalence contract).

C11 is a PURE PERFORMANCE sprint over the per-leaf neural path, in two wins:

  * **Win 1 -- actor-only prior.** ``policy_prior`` was a C8 shim over the combined
    ``evaluate`` that computed (and discarded) the entire separate critic head.
    C11 makes it run ONLY the actor pipeline. The probs must be byte-identical to
    the old combined path (``np.array_equal`` against ``evaluate``'s probs, which
    run the identical actor ops on the identical obs).
  * **Win 2 -- functional forward.** The per-``nn.Module`` dispatch is replaced by a
    plain ``F.linear`` + ``relu``/``tanh`` pipeline over weight tensors extracted
    once at load. ``F.linear`` is exactly what ``nn.Linear.forward`` calls in the
    same order, so this is bit-equivalent. We pin it against the explicit
    Module-dispatch path (the C8 implementation, reconstructed here) within a tight
    ``atol`` (expect EXACT -- ``atol=0``; the 1e-6 bound only guards genuine ULP
    drift) across both solo and two-player modes and both seats.

The reproducibility property (same ``(state, seed)`` -> identical plan) and the
``__getstate__`` pickle-by-path contract (now also nulling the extracted-tensor
cache ``_fwd``) are checked too.

Runnable WITHOUT a trained checkpoint (a fresh random-weight codec-v3 prior is
enough -- C11 is net-agnostic), mirroring ``tests/test_c8_lean_inference.py``.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from heat.agents.mcts_agent import MCTSAgent, MCTSConfig, NetAdapter  # noqa: E402
from heat.engine.driver import DecisionKind, run_round_driver  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.ml.action_codec import decode_action, legal_action_mask  # noqa: E402
from heat.ml.features import encode_observation  # noqa: E402
from heat.ml.spaces import ACTION_DIM, OBS_DIM  # noqa: E402
from heat.tracks.generator import generate_track  # noqa: E402


# ---------------------------------------------------------------------------
# Cold priors (random-weight codec-v3 checkpoints), one per #players mode
# ---------------------------------------------------------------------------


def _mint_prior(tmp_path_factory, num_players: int) -> str:
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp(f"c11p{num_players}") / "cold.zip")
    model = build_model(
        HeatEnv(num_players=num_players), PPOConfig(seed=0, device="cpu")
    )
    save_checkpoint(
        model, out, track_name="generated", num_players=num_players, seed=0
    )
    return out


@pytest.fixture(scope="module")
def solo_prior(tmp_path_factory) -> str:
    return _mint_prior(tmp_path_factory, 1)


@pytest.fixture(scope="module")
def twop_prior(tmp_path_factory) -> str:
    return _mint_prior(tmp_path_factory, 2)


@pytest.fixture(scope="module")
def adapter(solo_prior) -> NetAdapter:
    a = NetAdapter(solo_prior)
    a._get_model()
    return a


# ---------------------------------------------------------------------------
# The reference: the C8 Module-dispatch forward (reconstructed exactly)
# ---------------------------------------------------------------------------


def _module_dispatch_forward(policy, obs, mask, value_mode="rounds"):
    """The pre-C11 path: ``extract_features`` + ``mlp_extractor.forward_*`` + the
    ``action_net`` / ``value_net`` modules (full ``nn.Module`` dispatch). Returns
    ``(probs, value)`` the same way C8's ``evaluate`` did."""
    import torch

    ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
    with torch.inference_mode():
        feats = policy.extract_features(ob)
        pf, vf = feats if isinstance(feats, tuple) else (feats, feats)
        latent_pi = policy.mlp_extractor.forward_actor(pf)
        logits = policy.action_net(latent_pi)
        mt = torch.as_tensor(
            np.asarray(mask, dtype=bool), dtype=torch.bool
        ).reshape(logits.shape)
        ml = torch.where(mt, logits, torch.tensor(-1e8, dtype=logits.dtype))
        probs = torch.softmax(ml, dim=-1).reshape(-1).numpy().copy()
        latent_vf = policy.mlp_extractor.forward_critic(vf)
        v = policy.value_net(latent_vf)
        if value_mode == "winloss":
            v = torch.tanh(v)
        return probs, float(v.reshape(-1)[0])


# ---------------------------------------------------------------------------
# A battery of REAL decision states (obs + legal mask) for a given mode/seat
# ---------------------------------------------------------------------------


def _decision_battery(
    num_players: int = 1, seat: int = 0, n: int = 25
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect ``n`` (obs, mask) pairs at real decisions of ``seat`` across a game.

    The obs is encoded from ``seat``'s perspective (perfect-info, the C6 mover
    convention), so two-player seats 0/1 each exercise their own observation.
    """
    track = generate_track(seed=4242)
    state = GameState.create(track, num_players, logging_enabled=True, seed=7)
    for p in state.players:
        p.lap = 1

    out: list[tuple[np.ndarray, np.ndarray]] = []
    gen = run_round_driver(state)
    send: object = None
    guard = 0
    while len(out) < n and guard < 8000:
        guard += 1
        try:
            decision = gen.send(send)
        except StopIteration:
            if state.is_game_over:
                break
            gen = run_round_driver(state)
            send = None
            continue
        mask = legal_action_mask(decision, state)
        if int(mask.sum()) >= 1 and decision.player_id == seat:
            obs = encode_observation(state, decision.player_id, decision)
            out.append((obs.copy(), mask.copy()))
        flat = int(np.argmax(mask)) if mask.any() else -1
        send = decode_action(decision, flat, state) if flat >= 0 else None
    assert out, f"battery collected no decisions for seat {seat} ({num_players}p)"
    return out


# ---------------------------------------------------------------------------
# Win 1 -- the actor-only prior is byte-identical to the combined evaluate probs
# ---------------------------------------------------------------------------


class TestActorOnlyPriorByteIdentical:
    def test_policy_prior_array_equal_evaluate(self, adapter) -> None:
        """The actor-only ``policy_prior`` returns EXACTLY ``evaluate``'s probs
        (same actor ops, same obs, same masked softmax) -- np.array_equal."""
        for obs, mask in _decision_battery(1, 0, 12):
            eval_probs, _ = adapter.evaluate(obs, mask)
            prior = adapter.policy_prior(obs, mask)
            assert np.array_equal(prior, eval_probs)

    def test_policy_prior_array_equal_module_dispatch(self, adapter) -> None:
        """And byte-identical to the pre-C11 Module-dispatch probs."""
        policy = adapter._get_model().policy
        rng = np.random.default_rng(7)
        for _ in range(30):
            obs = rng.standard_normal(OBS_DIM).astype(np.float32)
            mask = rng.random(ACTION_DIM) > rng.uniform(0.1, 0.9)
            mask[int(rng.integers(ACTION_DIM))] = True
            ref_probs, _ = _module_dispatch_forward(policy, obs, mask)
            assert np.array_equal(adapter.policy_prior(obs, mask), ref_probs)


# ---------------------------------------------------------------------------
# Win 2 -- functional forward is bit-equivalent to Module dispatch (both modes/seats)
# ---------------------------------------------------------------------------


class TestFunctionalEquivalence:
    @pytest.mark.parametrize(
        "mode,seat",
        [("solo", 0), ("twop", 0), ("twop", 1)],
    )
    def test_evaluate_matches_module_dispatch(
        self, mode, seat, solo_prior, twop_prior
    ) -> None:
        path = solo_prior if mode == "solo" else twop_prior
        num_players = 1 if mode == "solo" else 2
        a = NetAdapter(path)
        a._get_model()
        policy = a._get_model().policy

        max_dp = 0.0
        max_dv = 0.0
        for obs, mask in _decision_battery(num_players, seat, 15):
            ref_probs, ref_v = _module_dispatch_forward(policy, obs, mask)
            probs, v = a.evaluate(obs, mask)
            max_dp = max(max_dp, float(np.abs(probs - ref_probs).max()))
            max_dv = max(max_dv, abs(v - ref_v))
        # F.linear IS nn.Linear's op in the same order -> expect EXACT.
        assert max_dp == 0.0, f"{mode} seat{seat}: prob max-diff {max_dp} (expected 0)"
        assert max_dv == 0.0, f"{mode} seat{seat}: value max-diff {max_dv} (expected 0)"

    def test_leaf_value_matches_module_dispatch(self, adapter) -> None:
        policy = adapter._get_model().policy
        for obs, mask in _decision_battery(1, 0, 12):
            _, ref_v = _module_dispatch_forward(policy, obs, mask)
            # leaf_value drops the policy head; the critic op order is unchanged.
            assert adapter.leaf_value(obs) == ref_v

    def test_winloss_value_matches_module_dispatch(self, solo_prior) -> None:
        a = NetAdapter(solo_prior)
        a._get_model()
        a._value_mode = "winloss"
        policy = a._get_model().policy
        for obs, mask in _decision_battery(1, 0, 8):
            _, ref_v = _module_dispatch_forward(policy, obs, mask, "winloss")
            _, v = a.evaluate(obs, mask)
            assert v == ref_v
            assert a.leaf_value(obs) == ref_v
            assert abs(v) <= 1.0 + 1e-6  # tanh-bounded


# ---------------------------------------------------------------------------
# The extracted-tensor cache: rebuilds on model change, nulled in the pickle
# ---------------------------------------------------------------------------


class TestForwardCache:
    def test_fwd_cache_built_lazily_and_keyed_to_model(self, solo_prior) -> None:
        a = NetAdapter(solo_prior)
        assert a._fwd is None
        a._get_model()
        a.leaf_value(np.zeros(OBS_DIM, dtype=np.float32))  # first forward builds it
        assert a._fwd is not None
        assert a._fwd["_model_id"] == id(a._get_model())

    def test_fwd_not_pickled(self, solo_prior) -> None:
        import pickle

        a = NetAdapter(solo_prior)
        a._get_model()
        a.leaf_value(np.zeros(OBS_DIM, dtype=np.float32))  # build _fwd + _obs_buf
        assert a._fwd is not None
        restored = pickle.loads(pickle.dumps(a))
        assert restored._fwd is None  # extracted tensors are process-local scratch
        assert restored._obs_buf is None
        assert restored._model is None  # pickle-by-path contract intact


# ---------------------------------------------------------------------------
# Determinism: a fixed (state, seed) plan is byte-stable under the C11 path
# ---------------------------------------------------------------------------


class TestDeterminismPreserved:
    def test_same_seed_same_plan(self, solo_prior) -> None:
        track = generate_track(seed=11)

        def plan() -> tuple:
            state = GameState.create(track, 1, logging_enabled=True, seed=3)
            for p in state.players:
                p.lap = 1
            agent = MCTSAgent(
                model_path=solo_prior, config=MCTSConfig(n_simulations=16), seed=99
            )
            gen = run_round_driver(state)
            decision = next(gen)
            while decision.kind != DecisionKind.GEAR:
                mask = legal_action_mask(decision, state)
                flat = int(np.argmax(mask))
                decision = gen.send(decode_action(decision, flat, state))
            return agent.choose_gear(state, 0, list(decision.legal))

        assert plan() == plan(), "C11 path broke the (state, seed) reproducibility"
