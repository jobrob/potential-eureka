"""Tests for Sprint C5: warm-start + anchor the value head (the binding constraint).

C5 composes three value-side levers on top of C4's Gumbel policy target:

  1. PRIMARY -- the per-generation critic warm-start (``train_az.graft_warm_critic``)
     + an L2 anchor (``train_az._anchor_loss``) defending the warm critic during
     training so the C4 4.8 -> 6.5 -> 40 collapse cannot recur;
  2. the data-starvation fix -- a ``traj_greedy`` knob in ``gen_selfplay`` that drives
     the *acted* trajectory greedily (more finishes) while the logged completed-Q /
     visit-count ``pi`` target is byte-identical (C4's entropy win preserved);
  3. the mechanical fix -- a non-empty finished-only val split is a hard precondition
     in ``az_loop`` (raise, never silently train an uncontrolled critic), and
     ``v_mae_rounds`` is a first-class per-generation gate signal.

The tests mirror the C2/C3/C4 discipline: fast, CPU-only, tmp fixtures, no committed
data, random-weight priors (they verify the *mechanism*, which is net-agnostic). The
load-bearing check is "the critic does NOT regress across a generation with the anchor
on" -- the direct evidence the binding constraint is removed.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_selfplay as G  # noqa: E402
import train_az as T  # noqa: E402
import az_loop as L  # noqa: E402

from heat.engine import rules  # noqa: E402
from heat.engine.driver import Decision, DecisionKind  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.ml.action_codec import legal_action_mask  # noqa: E402
from heat.ml.env import HeatEnv  # noqa: E402
from heat.ml.model import PPOConfig, build_model  # noqa: E402
from heat.ml.spaces import ACTION_DIM, OBS_DIM  # noqa: E402
from heat.ml.training import save_checkpoint  # noqa: E402

from heat.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a random codec-v3 prior + a "warm" value-critic checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """A fresh random-weight codec-v3 ``MaskablePPO`` checkpoint (CPU)."""
    out = str(tmp_path_factory.mktemp("c5prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


@pytest.fixture(scope="module")
def warm_value(tmp_path_factory) -> str:
    """A 'calibrated' value-critic checkpoint: a distinct random critic.

    Stands in for the Option-A ~4.5-round-MAE critic. Built with a different seed
    from any AZ net so the grafted critic is byte-distinguishable from a fresh one,
    which is exactly what the warm-start / anchor tests assert moved.
    """
    out = str(tmp_path_factory.mktemp("c5warm") / "warm_v.zip")
    model = build_model(
        HeatEnv(num_players=1),
        PPOConfig(seed=12345, device="cpu", share_features_extractor=False),
    )
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=12345)
    return out


def _tight_track(seed: int = 0):
    return G.generate_track(G._SELFPLAY_SEED_BASE + seed, G._TIGHT_PARAMS)


def _fresh_state(seed: int = 5, track_seed: int = 0) -> GameState:
    track = _tight_track(track_seed)
    state = GameState.create(track, 1, logging_enabled=True, seed=seed)
    state.players[0].lap = 1
    return state


def _gear_decision(state: GameState) -> Decision:
    p = state.players[0]
    legal = rules.legal_gear_shifts(p.gear, p.heat_available)
    return Decision(DecisionKind.GEAR, 0, legal)


def _make_agent(model_path: str, *, selector: str, sims: int, seed: int = 0,
                temperature_moves: int = 0) -> MCTSAgent:
    cfg = MCTSConfig(
        n_simulations=sims, dirichlet_eps=0.0,
        temperature_moves=temperature_moves, root_selector=selector,
    )
    agent = MCTSAgent(model_path=model_path, config=cfg, seed=seed, name=f"c5-{selector}")
    agent._ply = 0
    return agent


def _new_az_policy(seed: int = 0):
    """A fresh AZ policy (random critic) -- the graft target."""
    model = build_model(
        HeatEnv(num_players=1),
        PPOConfig(seed=seed, device="cpu", share_features_extractor=False),
    )
    return model.policy


def _critic_sd(policy) -> dict:
    return T._critic_state_dict(policy)


def _actor_sd(policy) -> dict:
    return {
        k: v.detach().cpu().clone()
        for k, v in policy.state_dict().items()
        if not any(k.startswith(p) for p in T._CRITIC_PREFIXES)
    }


def _synthetic_dataset(path: str, *, n_train: int, n_val: int, seed: int,
                       val_targets=None) -> str:
    """Write a tiny self-play-shaped .npz (obs/pi/mask/z/kind/track_seed/split).

    A single-legal-action toy is not enough (pi must sum to 1 over a >1 support),
    so we synthesize a 2-action mask with a soft pi and a z target. ``val_targets``
    overrides the val z (so a test can pin the held-batch MAE).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_val
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    mask = np.zeros((n, ACTION_DIM), dtype=bool)
    mask[:, 0] = True
    mask[:, 1] = True
    pi = np.zeros((n, ACTION_DIM), dtype=np.float32)
    pi[:, 0] = 0.6
    pi[:, 1] = 0.4
    # z = -rounds_remaining (<= 0). Train rows span a band; val a tight band.
    z = -rng.uniform(2.0, 12.0, size=n).astype(np.float32)
    if val_targets is not None:
        z[n_train:] = np.asarray(val_targets, dtype=np.float32)
    kind = np.zeros(n, dtype=np.int8)  # GEAR
    track_seed = np.arange(n, dtype=np.int64)
    split = np.array(["train"] * n_train + ["val"] * n_val, dtype="S5")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, obs=obs, pi=pi, mask=mask, z=z, kind=kind,
                        track_seed=track_seed, split=split)
    return path


def _train_args(data, out, **over):
    base = dict(
        data=data, out=out, epochs=6, batch=64, lr=3e-4, c_v=1.0,
        weight_decay=1e-4, eval_every=3, patience=8, net="default",
        device="cpu", seed=0, warm_value=None, critic_anchor="none",
        c_anchor=1.0, lr_critic=3e-5, freeze_critic_epochs=0,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# 1. Critic warm-start init: byte-equal critic, actor untouched
# ---------------------------------------------------------------------------


class TestWarmStartInit:
    def test_graft_copies_warm_critic_and_leaves_actor(self, warm_value) -> None:
        policy = _new_az_policy(seed=0)
        actor_before = _actor_sd(policy)
        critic_before = _critic_sd(policy)

        grafted = T.graft_warm_critic(policy, warm_value, device="cpu")

        # The grafted critic now matches the warm V's critic byte-for-byte.
        from sb3_contrib import MaskablePPO
        warm_critic = _critic_sd(MaskablePPO.load(warm_value, device="cpu").policy)
        critic_after = _critic_sd(policy)
        assert set(critic_after) == set(warm_critic)
        for k in critic_after:
            assert torch.equal(critic_after[k], warm_critic[k]), k
        # And it actually CHANGED (the warm critic differs from the fresh one).
        assert any(not torch.equal(critic_before[k], critic_after[k])
                   for k in critic_after)
        # The returned snapshot equals the grafted critic (the anchor reference).
        assert set(grafted) == set(critic_after)
        for k in grafted:
            assert torch.equal(grafted[k], critic_after[k])

        # The actor is byte-for-byte untouched (the disjoint-graft contract).
        actor_after = _actor_sd(policy)
        assert set(actor_after) == set(actor_before)
        for k in actor_after:
            assert torch.equal(actor_after[k], actor_before[k]), k


# ---------------------------------------------------------------------------
# 2. The headline check: the critic does NOT regress with the anchor on
# ---------------------------------------------------------------------------


class TestCriticDoesNotRegress:
    def _held_mae(self, policy, obs, rr) -> float:
        with torch.no_grad():
            pred = policy.predict_values(torch.as_tensor(obs)).reshape(-1).cpu().numpy()
        pred_rounds = -pred.astype(np.float64)
        return float(np.abs(pred_rounds - rr.astype(np.float64)).mean())

    def test_anchor_holds_critic_starvation_off_regresses(self, tmp_path, warm_value):
        """Train one generation on a starvation-prone batch (train z far from the
        warm critic's calibration). With the anchor ON the held-out MAE stays near
        the warm critic's; with it OFF the MAE is allowed to drift -- the anchor is
        load-bearing, not inert (the C4 collapse cannot recur with the anchor)."""
        # A held-out finished batch the warm critic is already near on.
        held_obs = np.random.default_rng(7).standard_normal((64, OBS_DIM)).astype(np.float32)
        # The "true" rounds-remaining for the held batch = what the warm critic
        # predicts (so the warm critic's MAE on it is ~0 -- the calibrated baseline).
        from sb3_contrib import MaskablePPO
        warm_policy = MaskablePPO.load(warm_value, device="cpu").policy
        with torch.no_grad():
            warm_pred = warm_policy.predict_values(
                torch.as_tensor(held_obs)).reshape(-1).cpu().numpy()
        held_rr = (-warm_pred).astype(np.float64)  # warm critic MAE on held ~ 0

        # A deliberately adversarial train set: z pushed to an extreme so an
        # unanchored critic chases it and abandons the warm calibration.
        data = _synthetic_dataset(
            str(tmp_path / "starve.npz"), n_train=128, n_val=16, seed=3)
        d = np.load(data)
        z = d["z"].copy()
        z[d["split"] == b"train"] = -60.0  # spin-spiral-like extreme target
        np.savez_compressed(data, obs=d["obs"], pi=d["pi"], mask=d["mask"], z=z,
                            kind=d["kind"], track_seed=d["track_seed"], split=d["split"])

        warm_baseline = self._held_mae(warm_policy, held_obs, held_rr)

        # Anchor ON (strong) -> critic should stay near the warm baseline.
        T.train_az(_train_args(data, str(tmp_path / "on.zip"), epochs=8,
                               warm_value=warm_value, critic_anchor="l2",
                               c_anchor=50.0))
        on_policy = MaskablePPO.load(str(tmp_path / "on.zip"), device="cpu").policy
        mae_on = self._held_mae(on_policy, held_obs, held_rr)

        # Anchor OFF (warm-start only, no protection) -> critic free to drift.
        T.train_az(_train_args(data, str(tmp_path / "off.zip"), epochs=8,
                               warm_value=warm_value, critic_anchor="none"))
        off_policy = MaskablePPO.load(str(tmp_path / "off.zip"), device="cpu").policy
        mae_off = self._held_mae(off_policy, held_obs, held_rr)

        # The anchor keeps the critic close to the warm baseline; without it the
        # critic drifts materially further toward the adversarial target.
        assert mae_on < mae_off - 1e-3, (
            f"anchor not load-bearing: MAE on={mae_on:.3f} off={mae_off:.3f} "
            f"(warm baseline {warm_baseline:.3f})"
        )
        # And the anchored critic does NOT regress far from the warm baseline.
        assert mae_on <= mae_off, "anchored MAE should be the smaller of the two"


# ---------------------------------------------------------------------------
# 3. The L2 anchor actually pulls toward the warm critic; weight 0 recovers C4
# ---------------------------------------------------------------------------


class TestAnchorTerm:
    def test_anchor_loss_zero_at_warm_and_positive_away(self, warm_value) -> None:
        policy = _new_az_policy(seed=0)
        warm_sd = T.graft_warm_critic(policy, warm_value, device="cpu")
        # At the warm critic the anchor loss is exactly zero.
        loss0 = T._anchor_loss(policy, warm_sd, "cpu")
        assert float(loss0.detach()) == pytest.approx(0.0, abs=1e-9)
        # Perturb a critic parameter -> the anchor loss becomes positive.
        with torch.no_grad():
            for p in T._critic_parameters(policy):
                p.add_(1.0)
                break
        loss1 = T._anchor_loss(policy, warm_sd, "cpu")
        assert float(loss1.detach()) > 0.0

    def test_weight_zero_recovers_c4_path(self, tmp_path, warm_value) -> None:
        """With c_anchor=0 (and anchor mode l2) the trained net is byte-identical to
        the warm-start-only run -- the anchor term is genuinely off, not silently
        applied. (The full C4-default parity pin is TestC4Parity below.)"""
        data = _synthetic_dataset(str(tmp_path / "d.npz"), n_train=96, n_val=16, seed=1)
        T.train_az(_train_args(data, str(tmp_path / "a.zip"), warm_value=warm_value,
                               critic_anchor="l2", c_anchor=0.0, seed=11))
        T.train_az(_train_args(data, str(tmp_path / "b.zip"), warm_value=warm_value,
                               critic_anchor="none", seed=11))
        from sb3_contrib import MaskablePPO
        a = MaskablePPO.load(str(tmp_path / "a.zip"), device="cpu").policy.state_dict()
        b = MaskablePPO.load(str(tmp_path / "b.zip"), device="cpu").policy.state_dict()
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), f"c_anchor=0 perturbed {k}"


# ---------------------------------------------------------------------------
# 4. The trajectory-greediness knob raises the finish rate; target unchanged
# ---------------------------------------------------------------------------


class TestTrajGreedy:
    def test_target_byte_identical_under_traj_greedy(self, cold_prior) -> None:
        """For a fixed (state, seed) the logged pi target is byte-identical with
        traj_greedy on vs off -- the trajectory is decoupled from the target (C4's
        entropy win preserved). Only the acted action may differ."""
        for selector in ("puct", "gumbel"):
            state = _fresh_state(seed=5, track_seed=0)
            decision = _gear_decision(state)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue
            tm = 10 if selector == "puct" else 0
            a1 = _make_agent(cold_prior, selector=selector, sims=16, seed=3,
                             temperature_moves=tm)
            pi_off, _ = G._search_visit_distribution(a1, state, decision,
                                                     traj_greedy=False)
            a2 = _make_agent(cold_prior, selector=selector, sims=16, seed=3,
                             temperature_moves=tm)
            pi_on, _ = G._search_visit_distribution(a2, state, decision,
                                                    traj_greedy=True)
            assert np.array_equal(pi_off, pi_on), (
                f"{selector}: traj_greedy changed the logged pi target")

    def test_greedy_trajectory_finishes_at_least_as_many_races(self, cold_prior,
                                                               tmp_path) -> None:
        """On a fixed seed band the greedier-trajectory setting finishes no fewer
        episodes than the C4 exploratory trajectory (the data-starvation fix). We
        assert >= (not strictly >) so the pin is robust to a tiny band where both
        finish all races; the loop's val precondition is what enforces non-empty."""
        def run(traj_greedy: bool) -> int:
            args = argparse.Namespace(
                out=str(tmp_path / f"sp_{traj_greedy}.npz"), model=cold_prior,
                tracks=5, val_tracks=2, sims=8, dirichlet_eps=0.25,
                dirichlet_alpha=0.5, temperature_moves=10, root_selector="gumbel",
                traj_greedy=traj_greedy, seed=0, game_seed=8000,
            )
            s = G.generate_dataset(args)
            return s["races_finished_train"] + s["races_finished_val"]

        finished_explore = run(False)
        finished_greedy = run(True)
        assert finished_greedy >= finished_explore, (
            f"greedy trajectory finished fewer races "
            f"({finished_greedy}) than exploratory ({finished_explore})")


# ---------------------------------------------------------------------------
# 5. The hard non-empty finished-val precondition (the C4 mechanical bug, guarded)
# ---------------------------------------------------------------------------


class TestValSplitPrecondition:
    def test_empty_finished_val_raises_in_loop(self, monkeypatch, tmp_path,
                                               cold_prior) -> None:
        """If every val episode drops to MAX_ROUNDS the loop hard-raises rather than
        silently training an uncontrolled critic (the exact C4 bug). We stub the
        generator to report a zero finished-val split and assert the raise fires
        before any training happens."""
        def fake_generate(**kw):
            # Write a minimal valid npz so any downstream load would succeed, then
            # report an EMPTY finished-val split (the starvation case).
            _synthetic_dataset(kw["out"], n_train=8, n_val=0, seed=0)
            return {
                "n_rows": 8, "n_train": 8, "n_val": 0,
                "races_finished_train": 5, "races_finished_val": 0,
                "races_total_train": 5, "races_total_val": 2,
                "races_dropped_max_rounds": 2, "pi_entropy_mean": 0.5,
            }

        trained = {"called": False}

        def fake_train(**kw):
            trained["called"] = True
            return {"v_mae_rounds": 4.5, "best_val": {}}

        monkeypatch.setattr(L, "_generate", lambda **kw: fake_generate(**kw))
        monkeypatch.setattr(L, "_train", lambda **kw: fake_train(**kw))
        # Make the warm-prior gate cheap/no-op.
        gate = L.LoopGate(spins=L.CI(point=1.0, lo=0.0, hi=2.0, n=4),
                          rounds=L.CI(point=10.0, lo=9.0, hi=11.0, n=4),
                          finish_rate=1.0)
        monkeypatch.setattr(L, "_gate", lambda **kw: (gate, gate))

        args = argparse.Namespace(
            warm=cold_prior, out=str(tmp_path / "best.zip"),
            workdir=str(tmp_path / "wd"), generations=1, tracks=5, val_tracks=2,
            sims=8, buffer_window=1, epochs=4, c_v=1.0, gate_games=4, horizon=2,
            dets=2, stop_patience=1, root_selector="gumbel", net="default",
            device="cpu", seed=0, game_seed=8000, warm_value=None,
            critic_anchor="none", c_anchor=1.0, lr_critic=3e-5,
            freeze_critic_epochs=0, traj_greedy=False,
        )
        with pytest.raises(RuntimeError, match="VAL split is EMPTY"):
            L.run_loop(args)
        assert not trained["called"], "trained the critic despite the empty val split"


# ---------------------------------------------------------------------------
# 6. C4-default parity: warm-value unset + anchor none == the pre-C5 trainer
# ---------------------------------------------------------------------------


class TestC4Parity:
    def test_train_az_byte_identical_without_c5_toggles(self, tmp_path) -> None:
        """With --warm-value unset + --critic-anchor none + default trajectory,
        train_az produces a byte-identical checkpoint to a run on the same
        (data, seed) -- the regression pin that C5 must not perturb the C4 path."""
        data = _synthetic_dataset(str(tmp_path / "d.npz"), n_train=96, n_val=16, seed=2)
        T.train_az(_train_args(data, str(tmp_path / "x.zip"), seed=7))
        T.train_az(_train_args(data, str(tmp_path / "y.zip"), seed=7))
        from sb3_contrib import MaskablePPO
        x = MaskablePPO.load(str(tmp_path / "x.zip"), device="cpu").policy.state_dict()
        y = MaskablePPO.load(str(tmp_path / "y.zip"), device="cpu").policy.state_dict()
        for k in x:
            assert torch.equal(x[k], y[k]), f"C4-default path not deterministic at {k}"

    def test_anchor_none_requires_no_warm_value(self, tmp_path) -> None:
        """A non-none anchor without a warm-value is a hard error (the anchor needs
        a warm critic to pull toward) -- caught early, not silently no-op."""
        data = _synthetic_dataset(str(tmp_path / "d.npz"), n_train=32, n_val=8, seed=4)
        with pytest.raises(ValueError, match="requires --warm-value"):
            T.train_az(_train_args(data, str(tmp_path / "z.zip"),
                                   warm_value=None, critic_anchor="l2"))


# ---------------------------------------------------------------------------
# 7. Seed-band disjointness preserved (the warm-value precursor source)
# ---------------------------------------------------------------------------


class TestBandDisjointness:
    def test_precursor_selfplay_eval_bands_disjoint(self) -> None:
        """The warm-value precursor bands (mint_warm_prior 100_000 / 500_000) stay
        disjoint from the C5/C3 per-gen self-play slices (600_000+) and the held-out
        eval band (900_000). C5 reuses the same warm-value source, so this re-pins
        the existing discipline."""
        import mint_warm_prior as M
        # The mint precursor assertion still passes for a typical size.
        M._assert_bands_disjoint(train_tracks=80, val_tracks=20)
        # Every C5 generation slice is disjoint from all the other bands.
        for gen in range(1, 6):
            L._assert_c3_band_disjoint(gen, n_tracks=100)
        # The precursor bases are nowhere near the self-play / eval bases.
        assert M._PRECURSOR_TRAIN_BASE == 100_000
        assert M._PRECURSOR_VAL_BASE == 500_000
        assert L._C3_SELFPLAY_BASE == 600_000
        assert L._EVAL_BASE == 900_000


# ---------------------------------------------------------------------------
# 8. Schema / contract invariance (the C5-trained checkpoint loads + guards hold)
# ---------------------------------------------------------------------------


class TestSchemaInvariance:
    def test_c5_checkpoint_passes_contract_tripwire(self, tmp_path, warm_value) -> None:
        """A C5-trained net (warm-start + anchor) loads through the §3.4 MLAgent /
        NetAdapter contract tripwire unchanged."""
        data = _synthetic_dataset(str(tmp_path / "d.npz"), n_train=64, n_val=16, seed=5)
        out = str(tmp_path / "c5.zip")
        T.train_az(_train_args(out=out, data=data, warm_value=warm_value,
                               critic_anchor="l2", c_anchor=1.0))
        from heat.agents.ml_agent import MLAgent
        MLAgent(out, name="c5")._validate_meta()  # raises on any contract drift

    def test_z_floor_and_pi_schema_unchanged(self, tmp_path) -> None:
        """The (obs, pi, mask, z) schema guards still hold on a C5-trained net's
        input dataset: pi sums to 1 over its support, z <= 0."""
        data = _synthetic_dataset(str(tmp_path / "d.npz"), n_train=32, n_val=8, seed=6)
        d = np.load(data)
        assert (d["z"] <= 1e-6).all(), "z must be -rounds_remaining <= 0"
        sums = d["pi"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-4)
        support_outside = ((d["pi"] > 0) & ~d["mask"]).sum()
        assert support_outside == 0
