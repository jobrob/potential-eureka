"""Tests for Sprint C8 -- the lean inference path (behaviour-equivalence contract).

C8 is a PURE PERFORMANCE sprint over the per-leaf neural/plumbing path: a single
combined forward per leaf, a raw masked log-softmax that bypasses the SB3
``Distribution`` machinery, ``inference_mode`` + a reused input buffer, and a
per-worker ``torch.set_num_threads(1)``. None of it may change a single value the
search reads, so the whole sprint's risk is a numerical / determinism regression.

These tests pin the first-class equivalence contract the design makes load-bearing:

  1. **Lean ``evaluate`` == the SB3 path to tight tolerance** -- over a battery of
     real decision states + random masks, the lean ``NetAdapter.evaluate`` probs
     match ``policy.get_distribution(...).distribution.probs`` and the lean value
     matches ``policy.predict_values`` within ``atol=1e-5`` (the masked softmax is
     the identical operation; SB3's logit pre-normalization is a softmax no-op and
     it likewise substitutes ``-1e8`` for masked-out logits).
  2. **The compatibility shims do not diverge** -- ``policy_prior`` / ``leaf_value``
     return exactly what ``evaluate`` would (single source of truth), so any caller
     / older test that uses them directly sees identical numbers.
  3. **value_mode tanh is applied in the combined call** -- a ``winloss`` adapter's
     ``evaluate`` value equals ``tanh(predict_values)`` and is bounded ``[-1, 1]``.
  4. **The reused input buffer never aliases** -- two evaluations with different obs
     return different, correct results (the in-place ``copy_`` overwrites cleanly).
  5. **Determinism preserved** -- a fixed ``(state, seed)`` MCTS plan is byte-stable
     under the lean path (the real reproducibility contract).

Runnable WITHOUT a trained checkpoint (a fresh random-weight codec-v3 prior is
enough -- C8 is net-agnostic). Mirrors the cold-prior fixture + experiments-on-path
style of ``tests/test_c7_throughput.py``.
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
from heat.ml.action_codec import legal_action_mask  # noqa: E402
from heat.ml.features import encode_observation  # noqa: E402
from heat.ml.spaces import ACTION_DIM, OBS_DIM  # noqa: E402
from heat.tracks.generator import generate_track  # noqa: E402


# ---------------------------------------------------------------------------
# Shared cold prior (a random-weight codec-v3 checkpoint; C8 is net-agnostic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """Mint a fresh random-weight codec-v3 ``MaskablePPO`` checkpoint once.

    C8 verifies numerical equivalence of the lean inference path, which is
    net-agnostic; random weights are enough and CPU matches the NetAdapter path.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c8prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


@pytest.fixture(scope="module")
def adapter(cold_prior) -> NetAdapter:
    """A loaded NetAdapter (value_mode='rounds', the default solo path)."""
    a = NetAdapter(cold_prior)
    a._get_model()  # force the load + meta validation up front
    return a


def _sb3_reference(model, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """The historical (pre-C8) SB3 path: get_distribution probs + predict_values."""
    import torch

    ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        dist = model.policy.get_distribution(
            ob, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1)
        )
        probs = np.asarray(dist.distribution.probs.detach()).reshape(-1)
        v = float(np.asarray(model.policy.predict_values(ob).detach()).reshape(-1)[0])
    return probs, v


# ---------------------------------------------------------------------------
# A battery of REAL decision states (obs + legal mask) from a driven solo game
# ---------------------------------------------------------------------------


def _decision_battery(n: int = 25) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect ``n`` (obs, mask) pairs at real learner decisions across a game.

    Drives a solo game with a legal-default policy and snapshots the codec obs +
    legal mask at every decision with >= 1 legal action, so the equivalence test
    exercises GEAR / CARDS / REACT / SLIPSTREAM / DISCARD masks of varied width.
    """
    track = generate_track(seed=4242)
    state = GameState.create(track, 1, logging_enabled=True, seed=7)
    for p in state.players:
        p.lap = 1

    out: list[tuple[np.ndarray, np.ndarray]] = []
    gen = run_round_driver(state)
    send: object = None
    guard = 0
    while len(out) < n and guard < 4000:
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
        if int(mask.sum()) >= 1:
            obs = encode_observation(state, decision.player_id, decision)
            out.append((obs.copy(), mask.copy()))
        # advance with a legal default (decode the first set mask bit)
        from heat.ml.action_codec import decode_action

        flat = int(np.argmax(mask)) if mask.any() else -1
        send = decode_action(decision, flat, state) if flat >= 0 else None
    assert out, "battery collected no decision states"
    return out


# ---------------------------------------------------------------------------
# 1. Lean evaluate == SB3 path to tight tolerance
# ---------------------------------------------------------------------------


class TestEvaluateEquivalence:
    def test_probs_and_value_match_sb3(self, adapter) -> None:
        """``evaluate`` probs/value equal the SB3 get_distribution/predict_values
        outputs within atol=1e-5 over a battery of real decision states."""
        model = adapter._get_model()
        battery = _decision_battery()
        max_dp = 0.0
        max_dv = 0.0
        for obs, mask in battery:
            ref_probs, ref_v = _sb3_reference(model, obs, mask)
            lean_probs, lean_v = adapter.evaluate(obs, mask)
            assert lean_probs.shape == (ACTION_DIM,)
            max_dp = max(max_dp, float(np.abs(lean_probs - ref_probs).max()))
            max_dv = max(max_dv, abs(lean_v - ref_v))
        assert max_dp < 1e-5, f"prob max-diff {max_dp} exceeds 1e-5"
        assert max_dv < 1e-5, f"value max-diff {max_dv} exceeds 1e-5"

    def test_probs_match_sb3_on_random_masks(self, adapter) -> None:
        """The masked softmax matches SB3 over adversarial random masks (varied
        sparsity), not just the engine's legal masks."""
        model = adapter._get_model()
        rng = np.random.default_rng(0)
        max_dp = 0.0
        for _ in range(40):
            obs = rng.standard_normal(OBS_DIM).astype(np.float32)
            mask = rng.random(ACTION_DIM) > rng.uniform(0.1, 0.9)
            mask[int(rng.integers(ACTION_DIM))] = True  # >= 1 legal
            ref_probs, _ = _sb3_reference(model, obs, mask)
            lean_probs, _ = adapter.evaluate(obs, mask)
            max_dp = max(max_dp, float(np.abs(lean_probs - ref_probs).max()))
        assert max_dp < 1e-5, f"random-mask prob max-diff {max_dp} exceeds 1e-5"

    def test_probs_are_a_masked_distribution(self, adapter) -> None:
        """Lean probs sum to 1, are nonnegative, and put ~0 mass off the mask."""
        battery = _decision_battery(10)
        for obs, mask in battery:
            probs, _ = adapter.evaluate(obs, mask)
            assert probs.min() >= 0.0
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            off = probs[~np.asarray(mask, dtype=bool)]
            assert float(off.sum()) < 1e-5  # masked-out actions carry ~no mass


# ---------------------------------------------------------------------------
# 2. The compatibility shims are the SAME math as evaluate (no divergence)
# ---------------------------------------------------------------------------


class TestShimsMatchEvaluate:
    def test_policy_prior_equals_evaluate_probs(self, adapter) -> None:
        for obs, mask in _decision_battery(10):
            probs_eval, _ = adapter.evaluate(obs, mask)
            probs_shim = adapter.policy_prior(obs, mask)
            assert np.array_equal(probs_shim, probs_eval)

    def test_leaf_value_equals_evaluate_value(self, adapter) -> None:
        for obs, mask in _decision_battery(10):
            _, v_eval = adapter.evaluate(obs, mask)
            v_shim = adapter.leaf_value(obs)
            # leaf_value and evaluate run the SAME critic head on the SAME obs.
            assert v_shim == pytest.approx(v_eval, abs=1e-7)

    def test_leaf_value_matches_sb3_predict_values(self, adapter) -> None:
        model = adapter._get_model()
        for obs, _mask in _decision_battery(10):
            _, ref_v = _sb3_reference(model, obs, np.ones(ACTION_DIM, dtype=bool))
            assert adapter.leaf_value(obs) == pytest.approx(ref_v, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. value_mode tanh is applied in the combined call
# ---------------------------------------------------------------------------


class TestValueModeTanh:
    def test_winloss_value_is_tanh_of_rounds(self, cold_prior) -> None:
        """A winloss adapter's evaluate/leaf value == tanh(raw critic), bounded."""
        rounds = NetAdapter(cold_prior)
        rounds._get_model()
        winloss = NetAdapter(cold_prior)
        winloss._get_model()
        winloss._value_mode = "winloss"  # simulate a winloss sidecar

        for obs, mask in _decision_battery(8):
            _, v_rounds = rounds.evaluate(obs, mask)
            _, v_winloss = winloss.evaluate(obs, mask)
            assert abs(v_winloss) <= 1.0 + 1e-6  # tanh-bounded
            assert v_winloss == pytest.approx(float(np.tanh(v_rounds)), abs=1e-6)
            # the shim agrees with the combined call under winloss too
            assert winloss.leaf_value(obs) == pytest.approx(v_winloss, abs=1e-7)


# ---------------------------------------------------------------------------
# 4. The reused input buffer never aliases across calls
# ---------------------------------------------------------------------------


class TestReusedBufferNoAlias:
    def test_distinct_obs_give_distinct_results(self, adapter) -> None:
        battery = _decision_battery(6)
        # Evaluate each obs twice (interleaved) and confirm the buffer reuse does
        # not leak a previous obs's result into the next.
        first = [adapter.leaf_value(obs) for obs, _ in battery]
        # interleave a different obs between repeats
        for i, (obs, _mask) in enumerate(battery):
            other = battery[(i + 1) % len(battery)][0]
            adapter.leaf_value(other)  # overwrite the buffer with a different obs
            again = adapter.leaf_value(obs)  # must reproduce the original value
            assert again == pytest.approx(first[i], abs=1e-7)

    def test_obs_buffer_not_pickled(self, cold_prior) -> None:
        """The reused torch buffer is process-local scratch -- never travels in a
        pickle (the adapter must stay light and rebuild it lazily per process)."""
        import pickle

        a = NetAdapter(cold_prior)
        a._get_model()
        a.leaf_value(np.zeros(OBS_DIM, dtype=np.float32))  # build the buffer
        assert a._obs_buf is not None
        restored = pickle.loads(pickle.dumps(a))
        assert restored._obs_buf is None
        assert restored._model is None  # the pickle-by-path contract is intact


# ---------------------------------------------------------------------------
# 5. Determinism: a fixed (state, seed) plan is byte-stable under the lean path
# ---------------------------------------------------------------------------


class TestDeterminismPreserved:
    def test_same_seed_same_plan(self, cold_prior) -> None:
        track = generate_track(seed=11)

        def plan() -> tuple:
            state = GameState.create(track, 1, logging_enabled=True, seed=3)
            for p in state.players:
                p.lap = 1
            agent = MCTSAgent(
                model_path=cold_prior, config=MCTSConfig(n_simulations=16), seed=99
            )
            gen = run_round_driver(state)
            decision = next(gen)
            while decision.kind != DecisionKind.GEAR:
                from heat.ml.action_codec import decode_action

                mask = legal_action_mask(decision, state)
                flat = int(np.argmax(mask))
                decision = gen.send(decode_action(decision, flat, state))
            gear = agent.choose_gear(state, 0, list(decision.legal))
            return gear

        assert plan() == plan(), "lean path broke the (state, seed) reproducibility"
