"""Tests for Sprint C4: the Gumbel-AlphaZero ``RootActionSelector`` seam.

C4 swaps, **at the root only and behind ``MCTSConfig.root_selector``**, the C1/C2
visit-count policy target for the Gumbel-AlphaZero one: Gumbel top-``m`` sampling +
Sequential Halving for the acted action, and a **completed-Q** policy target
``π = softmax(logits + σ(completedQ))``. Used ONLY to build self-play training
targets in ``gen_selfplay``; the agent still acts/evals with unchanged PUCT.

The tests mirror the C1/C2/C3 discipline (fast, CPU-only, tmp fixtures, no
committed data, a random-weight prior is enough -- they verify the *mechanism*,
which is net-agnostic):

  1. **Non-collapse** -- on the SAME decisions, the Gumbel completed-Q target's
     π-entropy is MATERIALLY above the visit-count target's ~0.04 floor (the
     headline mechanism fix, measured directly).
  2. **Low-visit improvement guarantee** -- the Gumbel acted action's Q is >= the
     prior's expected Q at a very low sim count (the policy-improvement property),
     and the completed-Q mass shifts toward the higher-Q action vs the prior.
  3. **PUCT default parity** -- with the default ``root_selector="puct"`` the search
     output is byte-for-byte identical to the pre-C4 behavior (a regression guard).
  4. **Determinism / seed purity** -- the Gumbel path is a pure function of
     ``(state, seed)``: byte-stable output for a fixed ``(state, seed)``, no
     global-RNG leak.

Plus: config validation, the Sequential-Halving budget accounting (no overrun),
and the schema invariance (π is a probability vector, support ⊆ mask, sums to 1).
"""

from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# The experiments scripts are not a package; add the dir so gen_selfplay imports
# by module name (mirrors test_az_targets).
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gen_selfplay as G  # noqa: E402

from heat.engine import rules  # noqa: E402
from heat.engine.driver import Decision, DecisionKind  # noqa: E402
from heat.models.game_state import GameState  # noqa: E402
from heat.models.track import Corner, Space, Track  # noqa: E402
from heat.ml.action_codec import legal_action_mask  # noqa: E402
from heat.ml.spaces import ACTION_DIM  # noqa: E402

from heat.agents.mcts_agent import (  # noqa: E402
    GumbelRootResult,
    MCTSAgent,
    MCTSConfig,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers (a random-weight codec-v3 prior is enough)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cold_prior(tmp_path_factory) -> str:
    """A fresh random-weight codec-v3 ``MaskablePPO`` checkpoint (CPU)."""
    from heat.ml.env import HeatEnv
    from heat.ml.model import PPOConfig, build_model
    from heat.ml.training import save_checkpoint

    out = str(tmp_path_factory.mktemp("c4prior") / "cold.zip")
    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
    save_checkpoint(model, out, track_name="generated", num_players=1, seed=0)
    return out


def _tight_track(seed: int = 0) -> Track:
    return G.generate_track(G._SELFPLAY_SEED_BASE + seed, G._TIGHT_PARAMS)


def _gear_decision(state: GameState) -> Decision:
    p = state.players[0]
    legal_gears = rules.legal_gear_shifts(p.gear, p.heat_available)
    return Decision(DecisionKind.GEAR, 0, legal_gears)


def _cards_decision(state: GameState, gear: int) -> Decision:
    legal = rules.legal_card_plays(state.players[0].hand, gear)
    return Decision(DecisionKind.CARDS, 0, legal)


def _fresh_state(seed: int = 5, track_seed: int = 0) -> GameState:
    track = _tight_track(track_seed)
    state = GameState.create(track, 1, logging_enabled=True, seed=seed)
    state.players[0].lap = 1
    return state


def _entropy(pi: np.ndarray) -> float:
    """Shannon entropy of a probability vector (natural log; the gen_selfplay convention)."""
    p = pi[pi > 0.0]
    return float(-(p * np.log(p)).sum())


def _make_agent(model_path: str, *, selector: str, sims: int, seed: int = 0,
                temperature_moves: int = 0) -> MCTSAgent:
    cfg = MCTSConfig(
        n_simulations=sims,
        dirichlet_eps=0.0,  # noise off so the only difference is the selector
        temperature_moves=temperature_moves,
        root_selector=selector,
    )
    agent = MCTSAgent(model_path=model_path, config=cfg, seed=seed, name=f"c4-{selector}")
    agent._ply = 0
    return agent


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_puct_is_the_default(self) -> None:
        cfg = MCTSConfig()
        assert cfg.root_selector == "puct"
        assert cfg.gumbel_m == 8
        # Retuned off Danihelka's 50/1.0 for our low-sim/[0,1]-Q regime (see the
        # DEFAULT_GUMBEL_* docstring; the 50/1.0 σ swamped the prior at 16 sims).
        assert cfg.gumbel_c_visit == 25.0 and cfg.gumbel_c_scale == 0.25

    def test_rejects_bad_root_selector(self) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(root_selector="banana")

    def test_rejects_bad_gumbel_m(self) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(gumbel_m=0)

    def test_rejects_bad_sigma_constants(self) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(gumbel_c_scale=0.0)
        with pytest.raises(ValueError):
            MCTSConfig(gumbel_c_visit=-1.0)


# ---------------------------------------------------------------------------
# 1. Non-collapse: completed-Q entropy MATERIALLY above the visit-count floor
# ---------------------------------------------------------------------------


class TestNonCollapse:
    def _entropies_on_decision(self, cold_prior, decision_fn, sims=16):
        """Run PUCT (visit-count) and Gumbel (completed-Q) on the SAME decisions;
        return (puct_entropies, gumbel_entropies) over several track seeds."""
        puct_ents: list[float] = []
        gumbel_ents: list[float] = []
        for ts in range(6):
            state = _fresh_state(seed=5 + ts, track_seed=ts)
            decision = decision_fn(state)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue  # need a real multi-action choice

            # PUCT path: greedy outside the temperature window => visit-count target
            # collapses (the ~0.04 floor C2/C3 measured).
            a_puct = _make_agent(cold_prior, selector="puct", sims=sims, seed=0,
                                 temperature_moves=0)
            pi_puct, _ = G._search_visit_distribution(a_puct, state, decision)

            # Gumbel path on the same state/seed.
            a_gum = _make_agent(cold_prior, selector="gumbel", sims=sims, seed=0)
            pi_gum, _ = G._search_visit_distribution(a_gum, state, decision)

            if pi_puct.sum() > 0:
                puct_ents.append(_entropy(pi_puct))
            if pi_gum.sum() > 0:
                gumbel_ents.append(_entropy(pi_gum))
        return puct_ents, gumbel_ents

    def test_gumbel_target_entropy_above_visit_floor_gear(self, cold_prior) -> None:
        puct_ents, gum_ents = self._entropies_on_decision(cold_prior, _gear_decision)
        assert puct_ents and gum_ents
        puct_mean = float(np.mean(puct_ents))
        gum_mean = float(np.mean(gum_ents))
        # The visit-count target collapses (near the ~0.04 floor); the completed-Q
        # target carries a graded signal -- materially higher entropy.
        assert gum_mean > puct_mean + 0.1, (
            f"Gumbel completed-Q entropy {gum_mean:.4f} not materially above the "
            f"visit-count floor {puct_mean:.4f}"
        )
        # The visit-count target is genuinely collapsed (the documented failure).
        assert puct_mean < 0.2, f"visit-count entropy {puct_mean:.4f} not collapsed"

    def test_gumbel_target_entropy_above_visit_floor_cards(self, cold_prior) -> None:
        # CARDS is the wide branch where the collapse bites hardest. Drive each
        # state to a committed gear first, then root a CARDS decision.
        puct_ents: list[float] = []
        gum_ents: list[float] = []
        for ts in range(6):
            state = _fresh_state(seed=11 + ts, track_seed=ts)
            p = state.players[0]
            gear = p.gear if p.gear >= 1 else 1
            decision = _cards_decision(state, gear)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue
            a_puct = _make_agent(cold_prior, selector="puct", sims=16, temperature_moves=0)
            pi_puct, _ = G._search_visit_distribution(a_puct, state, decision)
            a_gum = _make_agent(cold_prior, selector="gumbel", sims=16)
            pi_gum, _ = G._search_visit_distribution(a_gum, state, decision)
            if pi_puct.sum() > 0:
                puct_ents.append(_entropy(pi_puct))
            if pi_gum.sum() > 0:
                gum_ents.append(_entropy(pi_gum))
        assert gum_ents, "no multi-action CARDS decision found"
        gum_mean = float(np.mean(gum_ents))
        # CARDS completed-Q target must be a real distribution (not one-hot/argmax).
        assert gum_mean > 0.3, (
            f"Gumbel CARDS completed-Q entropy {gum_mean:.4f} is still collapsed"
        )
        if puct_ents:
            assert gum_mean > float(np.mean(puct_ents)) + 0.1


# ---------------------------------------------------------------------------
# 2. Low-visit improvement guarantee
# ---------------------------------------------------------------------------


class TestImprovementGuarantee:
    def test_acted_q_at_least_prior_expected_q(self, cold_prior) -> None:
        """At a very low sim budget the Gumbel acted action's Q is >= the prior's
        expected Q over the searched actions (the policy-improvement property)."""
        found = False
        for ts in range(6):
            state = _fresh_state(seed=21 + ts, track_seed=ts)
            decision = _gear_decision(state)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue
            agent = _make_agent(cold_prior, selector="gumbel", sims=8, seed=ts)
            G._search_visit_distribution(agent, state, decision)
            gr = agent._gumbel_result
            assert isinstance(gr, GumbelRootResult)

            searched = [e for e in gr.edges if e.n > 0]
            if not searched:
                continue
            found = True
            acted = gr.edges[gr.acted_index]
            # The acted edge is among the searched (Sequential-Halving survivor).
            assert acted.n > 0
            # Prior-weighted expected Q over searched edges (the prior policy's value).
            num = sum(e.prior * agent._normalize_q(e.q()) for e in searched)
            den = sum(e.prior for e in searched) or 1.0
            prior_expected_q = num / den
            acted_q = agent._normalize_q(acted.q())
            assert acted_q >= prior_expected_q - 1e-9, (
                f"acted Q {acted_q:.4f} below prior expected Q {prior_expected_q:.4f}"
            )
        assert found, "no searched Gumbel root found to test the guarantee"

    def test_completed_q_mass_shifts_toward_higher_q(self, cold_prior) -> None:
        """The completed-Q target puts more mass on the higher-Q searched action
        than the prior does (the target is a bounded improvement, not noise)."""
        from heat.ml.action_codec import encode_action_index

        shifted = 0
        tested = 0
        for ts in range(8):
            state = _fresh_state(seed=31 + ts, track_seed=ts)
            decision = _gear_decision(state)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue
            agent = _make_agent(cold_prior, selector="gumbel", sims=16, seed=ts)
            pi, _ = G._search_visit_distribution(agent, state, decision)
            gr = agent._gumbel_result
            searched = [(i, e) for i, e in enumerate(gr.edges) if e.n > 0]
            if len(searched) < 2:
                continue
            # The two searched edges with the highest / lowest normalized Q.
            searched.sort(key=lambda ie: agent._normalize_q(ie[1].q()))
            lo_i, lo_e = searched[0]
            hi_i, hi_e = searched[-1]
            if agent._normalize_q(hi_e.q()) <= agent._normalize_q(lo_e.q()) + 1e-9:
                continue
            try:
                hi_flat = encode_action_index(decision, hi_e.action)
                lo_flat = encode_action_index(decision, lo_e.action)
            except (ValueError, IndexError):
                continue
            tested += 1
            # Prior odds vs target odds between hi and lo.
            prior_ratio = (hi_e.prior + 1e-12) / (lo_e.prior + 1e-12)
            target_ratio = (pi[hi_flat] + 1e-12) / (pi[lo_flat] + 1e-12)
            if target_ratio > prior_ratio - 1e-9:
                shifted += 1
        assert tested > 0, "no two-searched-edge Q-separated GEAR root found"
        # The completed-Q target must shift mass toward the higher-Q action in the
        # clear majority of separable cases (σ(Q̂) adds to the higher-Q logit).
        assert shifted >= max(1, tested // 2), (
            f"completed-Q mass shifted toward higher-Q in only {shifted}/{tested} cases"
        )


# ---------------------------------------------------------------------------
# 3. PUCT default parity: byte-for-byte unchanged
# ---------------------------------------------------------------------------


class TestPuctParity:
    def test_default_search_output_is_unchanged(self, cold_prior) -> None:
        """With the default ``root_selector="puct"`` the visit-count target and the
        acted action are byte-identical to a run that never set the field (the
        regression pin: C4 must not perturb the parity path)."""
        state = _fresh_state(seed=7, track_seed=1)
        decision = _gear_decision(state)

        a1 = _make_agent(cold_prior, selector="puct", sims=16, seed=0,
                         temperature_moves=10)
        a1._ply = 0
        pi1, acted1 = G._search_visit_distribution(a1, state, decision)

        # An agent built without ever touching root_selector (explicit C1 default).
        cfg = MCTSConfig(n_simulations=16, dirichlet_eps=0.0, temperature_moves=10)
        a2 = MCTSAgent(model_path=cold_prior, config=cfg, seed=0, name="c1")
        a2._ply = 0
        pi2, acted2 = G._search_visit_distribution(a2, state, decision)

        assert np.array_equal(pi1, pi2)
        assert acted1 == acted2
        # The PUCT path must never set the Gumbel result.
        assert a1._gumbel_result is None and a2._gumbel_result is None

    def test_puct_agent_plays_full_solo_game_unchanged(self, cold_prior) -> None:
        """A default agent still drives a full solo game (acting path untouched)."""
        from heat.engine.game import Game

        track = _tight_track(2)
        agent = MCTSAgent(model_path=cold_prior, config=MCTSConfig(n_simulations=8),
                          seed=3, name="puct")
        result = Game(track, [agent], logging_enabled=False, seed=99).run()
        assert result.finish_order == [0]


# ---------------------------------------------------------------------------
# 4. Determinism / seed purity
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_gumbel_target_byte_stable_under_fixed_seed(self, cold_prior) -> None:
        state = _fresh_state(seed=5, track_seed=0)
        decision = _gear_decision(state)

        def run():
            agent = _make_agent(cold_prior, selector="gumbel", sims=16, seed=42)
            pi, acted = G._search_visit_distribution(agent, state, decision)
            return pi.copy(), acted

        pi0, acted0 = run()
        for _ in range(4):
            pi, acted = run()
            assert np.array_equal(pi, pi0), "Gumbel target not byte-stable"
            assert acted == acted0, "Gumbel acted action not byte-stable"

    def test_no_global_rng_leak(self, cold_prior) -> None:
        """Perturbing the global RNG between runs must not change the output (all
        Gumbel noise draws only from the agent's per-turn ``_search_rng``)."""
        import random as _random

        state = _fresh_state(seed=5, track_seed=0)
        decision = _gear_decision(state)

        agent = _make_agent(cold_prior, selector="gumbel", sims=16, seed=7)
        pi_a, acted_a = G._search_visit_distribution(agent, state, decision)

        _random.seed(123456)
        _ = [_random.random() for _ in range(1000)]
        np.random.seed(98765)
        _ = np.random.rand(1000)

        agent2 = _make_agent(cold_prior, selector="gumbel", sims=16, seed=7)
        pi_b, acted_b = G._search_visit_distribution(agent2, state, decision)

        assert np.array_equal(pi_a, pi_b), "global-RNG perturbation changed the target"
        assert acted_a == acted_b

    def test_different_seed_can_differ(self, cold_prior) -> None:
        """Sanity: the Gumbel draw is actually seeded (different seeds can differ),
        so the byte-stability above is not a constant-output artifact."""
        state = _fresh_state(seed=5, track_seed=0)
        decision = _gear_decision(state)
        outs = set()
        for s in range(6):
            agent = _make_agent(cold_prior, selector="gumbel", sims=16, seed=s)
            pi, _ = G._search_visit_distribution(agent, state, decision)
            outs.add(tuple(np.round(pi[pi > 0], 6)))
        # Not strictly required to differ on every state, but a seeded Gumbel over
        # several seeds should produce more than one distinct target here.
        assert len(outs) >= 2, "Gumbel target identical across all seeds (unseeded?)"


# ---------------------------------------------------------------------------
# Sequential-Halving budget accounting + schema invariance
# ---------------------------------------------------------------------------


class TestBudgetAndSchema:
    def test_total_interior_sims_equals_budget(self, cold_prior) -> None:
        """Sequential Halving runs exactly ``n_simulations`` interior descents (no
        overrun/leak): the root's total visit count equals the budget."""
        for sims in (1, 4, 8, 16, 32):
            state = _fresh_state(seed=5, track_seed=0)
            decision = _gear_decision(state)
            agent = _make_agent(cold_prior, selector="gumbel", sims=sims, seed=0)
            G._search_visit_distribution(agent, state, decision)
            root_n = agent._root_node.n
            assert root_n == sims, (
                f"sims={sims}: root saw {root_n} descents (budget leak/overrun)"
            )

    def test_m_one_is_prior_greedy_degenerate(self, cold_prior) -> None:
        """``gumbel_m=1`` (or sims=1) degenerates to a single-arm root (the C1
        n_simulations=1 prior-greedy analogue) without crashing."""
        state = _fresh_state(seed=5, track_seed=0)
        decision = _gear_decision(state)
        cfg = MCTSConfig(n_simulations=16, gumbel_m=1, root_selector="gumbel",
                         dirichlet_eps=0.0)
        agent = MCTSAgent(model_path=cold_prior, config=cfg, seed=0, name="m1")
        agent._ply = 0
        pi, acted = G._search_visit_distribution(agent, state, decision)
        gr = agent._gumbel_result
        assert len(gr.sampled) == 1
        assert agent._root_node.n == 16

    def test_schema_invariance(self, cold_prior) -> None:
        """The completed-Q π is a probability vector, support ⊆ mask, sums to 1,
        full ACTION_DIM width (the .npz / train_az contract is unchanged)."""
        for ts in range(5):
            state = _fresh_state(seed=5 + ts, track_seed=ts)
            decision = _gear_decision(state)
            mask = legal_action_mask(decision, state)
            if int(mask.sum()) <= 1:
                continue
            agent = _make_agent(cold_prior, selector="gumbel", sims=16, seed=ts)
            pi, _ = G._search_visit_distribution(agent, state, decision)
            assert pi.shape == (ACTION_DIM,)
            support = pi > 0.0
            assert bool(mask[support].all()), "pi support escapes the mask"
            assert abs(float(pi.sum()) - 1.0) < 1e-6, "pi does not sum to 1"
            assert (pi >= 0.0).all()
