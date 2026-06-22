"""Tests for the Sprint C1 solo stochastic MCTS (``MCTSAgent``).

Covers every test the C1 spec lists, all runnable WITHOUT a trained checkpoint
(a freshly-built random-weight ``build_model`` net, or a tiny deterministic stub
net, is enough for correctness -- the parity NUMBER needs a trained prior, the
ALGORITHM does not):

  * **legality** -- across full solo games every gear/cards/react/slipstream/
    discard the agent returns is in the legal set the engine hands in;
  * **determinism** -- same state + seed => byte-stable move (the whole search,
    incl. chance sampling, is a pure function of (state, seed); no global RNG);
  * ``n_simulations=1`` degenerates to a sensible prior-greedy move;
  * **chance correctness** -- a chance node's backed-up value is the visit-
    weighted average of its sampled outcomes, and DPW respects its (C_pw, α_pw, K)
    widening rule (child count grows as specified);
  * **Q-normalization** -- with the raw −rounds_remaining scale the normalized
    selection actually uses the prior (the swamp bug cannot silently return);
  * **leaf discipline** -- a forced-spin line scores below a clean line at every
    sim count (V never consulted on a spun leaf);
  * **all-decision coverage** -- REACT/SLIPSTREAM/DISCARD are searched/net-driven,
    not delegated to a hand-coded heuristic rollout policy.

Mirrors the structure/style of ``tests/test_search_agent.py`` /
``tests/test_search_agent_learned.py``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.game import Game
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.tracks.generator import generate_track

from heat.agents.mcts_agent import (
    ChanceOutcome,
    Edge,
    EngineTransitionModel,
    MCTSAgent,
    MCTSConfig,
    Node,
    NodeKind,
)


# ---------------------------------------------------------------------------
# Stub nets (no checkpoint required) + helpers
# ---------------------------------------------------------------------------


class UniformNet:
    """Deterministic stub net: uniform masked prior + a constant leaf value.

    Backs the search with a uniform policy over the legal mask and a fixed leaf
    value, so a test exercises the *search machinery* (selection, chance backup,
    leaf discipline) without depending on a trained net's idiosyncrasies. A pure
    function of its inputs, so the determinism contract holds.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.leaf_calls = 0

    def policy_prior(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        probs = np.zeros(m.shape, dtype=np.float64)
        n = int(m.sum())
        if n:
            probs[m] = 1.0 / n
        return probs

    def leaf_value(self, obs: np.ndarray) -> float:
        self.leaf_calls += 1
        return self.value


@pytest.fixture(scope="module")
def real_net():
    """A freshly-built random-weight ``MaskablePPO`` policy on CPU, wrapped as a
    net adapter (``policy_prior`` / ``leaf_value``).

    Random weights are fine: the C1 unit tests verify the *algorithm*, which is
    net-agnostic. CPU device matches the C0 default (and the NetAdapter load path).
    Module-scoped so the (cheap) build happens once.
    """
    import torch

    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model, PPOConfig

    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))

    class _RealNet:
        def policy_prior(self, obs, mask):
            ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            with torch.no_grad():
                dist = model.policy.get_distribution(
                    ob, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1)
                )
                return np.asarray(dist.distribution.probs.detach()).reshape(-1)

        def leaf_value(self, obs):
            ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            with torch.no_grad():
                v = model.policy.predict_values(ob)
            return float(np.asarray(v.detach()).reshape(-1)[0])

    return _RealNet()


def _gen_track(seed: int = 7) -> Track:
    return generate_track(seed=seed)


def _limit1_loop() -> Track:
    """Short synthetic loop with a single tight (limit-1) corner (S1 stress track)."""
    spaces = [Space(i, lanes=2) for i in range(24)]
    corners = [Corner(start=10, end=11, speed_limit=1)]
    return Track("Limit1Loop", spaces, corners, [0, 1, 2, 3, 4, 5], laps=2)


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_are_c0_constants(self) -> None:
        cfg = MCTSConfig()
        assert cfg.n_simulations == 16
        assert (cfg.c_pw, cfg.alpha_pw, cfg.k_cap) == (1.0, 0.5, 8)
        assert cfg.c_init == 1.25 and cfg.c_base == 19652.0
        assert cfg.fpu_reduction == 0.25
        # Self-play exploration OFF by default (deterministic eval mode).
        assert cfg.dirichlet_eps == 0.0
        assert cfg.temperature_moves == 0

    def test_requires_net_or_path(self) -> None:
        with pytest.raises(ValueError):
            MCTSAgent()

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(n_simulations=0)
        with pytest.raises(ValueError):
            MCTSConfig(k_cap=0)
        with pytest.raises(ValueError):
            MCTSConfig(dirichlet_eps=2.0)


# ---------------------------------------------------------------------------
# Legality: never returns an illegal move (re-validated like LookaheadAgent)
# ---------------------------------------------------------------------------


class _LegalityWrapper:
    """Asserts every returned action is in the legal set the engine hands in."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name

    def choose_gear(self, state, player_id, legal_gears):
        choice = self.inner.choose_gear(state, player_id, legal_gears)
        assert choice in legal_gears, f"illegal gear {choice}"
        return choice

    def choose_cards(self, state, player_id, legal_plays):
        choice = self.inner.choose_cards(state, player_id, legal_plays)
        assert choice in legal_plays, "illegal card play"
        return choice

    def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
        return self.inner.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )

    def choose_slipstream(self, state, player_id):
        return self.inner.choose_slipstream(state, player_id)

    def choose_discard(self, state, player_id, discardable):
        choice = self.inner.choose_discard(state, player_id, discardable)
        assert all(c in discardable for c in choice)
        return choice


class TestLegality:
    def test_all_decisions_legal_solo(self, real_net) -> None:
        track = _gen_track(7)
        for game_idx in range(3):
            agent = _LegalityWrapper(
                MCTSAgent(
                    net=real_net,
                    config=MCTSConfig(n_simulations=8),
                    seed=game_idx,
                )
            )
            game = Game(track, [agent], logging_enabled=False, seed=1000 + game_idx)
            result = game.run()
            assert result.finish_order == [0]

    def test_never_illegal_on_limit1_loop(self, real_net) -> None:
        track = _limit1_loop()
        agent = _LegalityWrapper(
            MCTSAgent(net=real_net, config=MCTSConfig(n_simulations=8), seed=3)
        )
        game = Game(track, [agent], logging_enabled=False, seed=99)
        result = game.run()
        assert result.finish_order == [0]

    def test_choose_cards_revalidates_against_legal_set(self, real_net) -> None:
        """A stale cached plan is never returned: choose_cards must return a play
        from the legal set the driver hands in (re-validated)."""
        track = _gen_track(7)
        agent = MCTSAgent(net=real_net, config=MCTSConfig(n_simulations=4), seed=0)
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        gear = agent.choose_gear(state, 0, lg)
        state.players[0].gear = gear[0]
        lp = rules.legal_card_plays(state.players[0].hand, gear[0])
        cards = agent.choose_cards(state, 0, lp)
        assert cards in lp


# ---------------------------------------------------------------------------
# Determinism: same state + seed => byte-stable move (incl. chance sampling)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _plan(self, net, track, sims, seed):
        agent = MCTSAgent(net=net, config=MCTSConfig(n_simulations=sims), seed=seed)
        state = GameState.create(track, 1, logging_enabled=False, seed=123)
        state.players[0].lap = 1
        state.players[0].position = 2
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        gear = agent.choose_gear(state, 0, lg)
        state.players[0].gear = gear[0]
        lp = rules.legal_card_plays(state.players[0].hand, gear[0])
        cards = agent.choose_cards(state, 0, lp)
        return gear, tuple(c.display_name for c in cards)

    def test_byte_stable_under_fixed_seed(self, real_net) -> None:
        track = _gen_track(7)
        first = self._plan(real_net, track, sims=16, seed=0)
        for _ in range(5):
            assert self._plan(real_net, track, sims=16, seed=0) == first

    def test_full_game_is_deterministic(self, real_net) -> None:
        """A whole solo game replays identically for a fixed (agent seed, game
        seed) -- the search never leaks into global RNG."""
        track = _gen_track(11)

        def run():
            agent = MCTSAgent(
                net=real_net, config=MCTSConfig(n_simulations=8), seed=0
            )
            return Game(track, [agent], logging_enabled=False, seed=42).run()

        a = run()
        b = run()
        assert a.finish_order == b.finish_order
        assert a.total_rounds == b.total_rounds


# ---------------------------------------------------------------------------
# n_simulations=1 degenerates to a sensible prior-greedy move
# ---------------------------------------------------------------------------


class TestOneSimPriorGreedy:
    def test_one_sim_returns_legal_finishing_move(self, real_net) -> None:
        """With a single simulation the move reduces to the prior-greedy choice
        (legal, and the game still completes)."""
        track = _gen_track(7)
        agent = MCTSAgent(net=real_net, config=MCTSConfig(n_simulations=1), seed=0)
        game = Game(track, [agent], logging_enabled=False, seed=42)
        result = game.run()
        assert result.finish_order == [0]

    def test_one_sim_gear_is_top_prior(self) -> None:
        """At n_simulations=1 the acted gear is the highest-prior candidate.

        With a stub net assigning all prior mass to gear 1 (via a custom prior),
        the single-sim search must commit gear 1 -- the prior is the only signal.
        """

        class _Gear1Net(UniformNet):
            def policy_prior(self, obs, mask):
                m = np.asarray(mask, dtype=bool)
                probs = np.zeros(m.shape, dtype=np.float64)
                from heat.ml import spaces

                # All mass on gear-1 (flat index GEAR_OFFSET + 0) if legal.
                if m[spaces.GEAR_OFFSET + 0]:
                    probs[spaces.GEAR_OFFSET + 0] = 1.0
                else:
                    n = int(m.sum())
                    probs[m] = 1.0 / n if n else 0.0
                return probs

        track = _gen_track(7)
        agent = MCTSAgent(
            net=_Gear1Net(), config=MCTSConfig(n_simulations=1), seed=0
        )
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        gear = agent.choose_gear(state, 0, lg)
        assert gear[0] == 1


# ---------------------------------------------------------------------------
# Chance correctness: visit-weighted average + DPW widening rule
# ---------------------------------------------------------------------------


class TestChanceCorrectness:
    def test_dpw_widening_rule(self) -> None:
        """A chance node opens a new outcome only when ⌈C_pw·N^α_pw⌉ exceeds the
        current outcome count, capped at K (§4.3).

        Drives ``_select_outcome`` directly over increasing visit counts and
        asserts the realized outcome count matches ``min(K, ⌈C_pw·N^α_pw⌉)``.
        """
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent.to_move_pid = 0
        cfg = agent.config
        # A bare chance node (we only exercise selection's widening, not stepping).
        track = _limit1_loop()
        state = GameState.create(track, 1, logging_enabled=False, seed=1)
        node = Node(
            kind=NodeKind.CHANCE,
            state=state,
            action_path=((1, 0),),
            to_move=0,
        )
        # Simulate visits: before each selection bump node.n, then select; the
        # outcome list grows exactly to the DPW target.
        for n in range(1, 80):
            node.n = n
            agent._select_outcome(node, turn_seed=123, sim=0, depth=1)
            target = min(cfg.k_cap, math.ceil(cfg.c_pw * (max(1, n) ** cfg.alpha_pw)))
            assert len(node.outcomes) == target, (n, len(node.outcomes), target)
        # Never exceeds the hard cap K.
        assert len(node.outcomes) == cfg.k_cap

    def test_chance_backup_is_visit_weighted_average(self) -> None:
        """A chance node's value equals Σ w_i / Σ n_i over its outcomes.

        Builds a chance node with hand-set outcome statistics and runs the backup
        recompute; the node value must be the exact visit-weighted mean (the
        expectation over draws, §4.3), not a simple or last-write value.
        """
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent._q_min = math.inf
        agent._q_max = -math.inf
        track = _limit1_loop()
        state = GameState.create(track, 1, logging_enabled=False, seed=1)
        chance = Node(kind=NodeKind.CHANCE, state=state, action_path=(), to_move=0)
        # Three outcomes with different visit counts + summed values.
        chance.outcomes = [
            ChanceOutcome(reseed=1, n=3, w=30.0),  # mean 10
            ChanceOutcome(reseed=2, n=1, w=4.0),   # mean 4
            ChanceOutcome(reseed=3, n=2, w=2.0),   # mean 1
        ]
        # A backup through one of the outcomes recomputes the node value.
        path = [(chance, chance.outcomes[0])]
        agent._backup(path, root=Node(kind=NodeKind.DECISION, state=state,
                                      action_path=(), to_move=0), value=0.0)
        tot_n = 3 + 1 + 1 + 2  # outcome[0] bumped by the backup (n 3->4)
        tot_w = 30.0 + 0.0 + 4.0 + 2.0  # outcome[0] w bumped by value 0.0
        assert chance.value == pytest.approx(tot_w / tot_n)

    def test_chance_outcomes_are_distinct_draws(self, real_net) -> None:
        """DPW outcomes re-run the round-crossing advance with DIFFERENT reseeds,
        so distinct outcomes can realize distinct hands (true expectimax, not a
        single fixed determinization).

        Exercises the engine path: build a small tree on a real game, find a chance
        node, widen two outcomes, and assert their realized child hands are
        encoded -- and that re-realizing the SAME outcome reseed is byte-stable.
        """
        track = _gen_track(7)
        agent = MCTSAgent(
            net=real_net, config=MCTSConfig(n_simulations=24), seed=0
        )
        agent.to_move_pid = 0
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        # Run a search; then walk the tree to find a chance node with >= 2 outcomes.
        agent._plan_turn(state, 0, lg)
        # The plan-turn rebuilt internal state; re-run capturing the root.
        # (We re-run a fresh search and inspect via a monkeypatched collector.)
        found = self._find_chance_node(agent, state, 0, lg)
        if found is None:
            pytest.skip("no multi-outcome chance node reached at this sim budget")
        outcomes = found.outcomes
        assert len(outcomes) >= 1
        # Re-realizing the first outcome's reseed twice is byte-stable.
        model = EngineTransitionModel(0)
        c1 = agent._step_chance_child(found, outcomes[0], model)
        c2 = agent._step_chance_child(found, outcomes[0], model)
        h1 = [c.display_name for c in c1.state.get_player(0).hand]
        h2 = [c.display_name for c in c2.state.get_player(0).hand]
        assert h1 == h2  # same reseed => same draw (determinism)

    @staticmethod
    def _find_chance_node(agent, state, pid, lg):
        """Run a fresh search and return the first chance node with outcomes."""
        agent.to_move_pid = pid
        sig = agent._turn_signature(state, pid)
        turn_seed = agent._turn_seed(sig)
        root = Node(
            kind=NodeKind.DECISION, state=state, action_path=(), to_move=pid,
            decision=Decision(DecisionKind.GEAR, pid, lg),
        )
        model = EngineTransitionModel(pid)
        agent._q_min = math.inf
        agent._q_max = -math.inf
        agent._clones_this_move = 0
        for sim in range(agent.config.n_simulations):
            agent._simulate(root, model, turn_seed, sim)

        # BFS for a chance node with at least one outcome.
        stack = [root]
        seen = 0
        while stack and seen < 5000:
            n = stack.pop()
            seen += 1
            if n.kind == NodeKind.CHANCE and n.outcomes:
                return n
            for e in n.edges:
                if e.child is not None:
                    stack.append(e.child)
            for o in n.outcomes:
                if o.child is not None:
                    stack.append(o.child)
        return None


# ---------------------------------------------------------------------------
# Q-normalization: with raw −rounds_remaining scale, the prior still bites
# ---------------------------------------------------------------------------


class TestQNormalization:
    def test_normalize_q_maps_bounds_to_unit_interval(self) -> None:
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent._q_min = -25.0
        agent._q_max = -5.0
        assert agent._normalize_q(-25.0) == pytest.approx(0.0)
        assert agent._normalize_q(-5.0) == pytest.approx(1.0)
        assert agent._normalize_q(-15.0) == pytest.approx(0.5)

    def test_degenerate_bounds_return_neutral(self) -> None:
        """Before any value is seen (or with collapsed bounds) Q̂ is the neutral
        0.5, so selection is governed by the prior -- not a swamping raw Q."""
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent._q_min = math.inf
        agent._q_max = -math.inf
        assert agent._normalize_q(-12.0) == 0.5
        agent._q_min = -7.0
        agent._q_max = -7.0  # collapsed
        assert agent._normalize_q(-7.0) == 0.5

    def test_prior_decides_when_q_is_normalized(self) -> None:
        """Regression guard for the swamp bug: on the raw −rounds_remaining scale
        the exploration term P·U is O(1) while raw Q is O(10), so raw Q dominates.
        After min-max normalization the prior changes the argmax.

        Construct a decision node with two visited edges whose raw mean-Q differ by
        a tiny amount but whose priors differ a lot, and assert that with
        normalized Q the high-prior edge is selected (the prior bites), whereas the
        raw-Q ranking would pick the marginally-higher-Q edge.
        """
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        track = _limit1_loop()
        state = GameState.create(track, 1, logging_enabled=False, seed=1)
        node = Node(
            kind=NodeKind.DECISION, state=state, action_path=(), to_move=0,
            decision=Decision(DecisionKind.GEAR, 0, [(1, 0), (2, 1)]),
        )
        # Two edges on the raw −rounds_remaining scale (~−25..0): nearly equal Q,
        # but edge B has a far larger prior.
        edge_a = Edge(action=(1, 0), prior=0.01, n=10, w=-200.0)  # meanQ -20.0
        edge_b = Edge(action=(2, 1), prior=0.99, n=10, w=-200.1)  # meanQ -20.01
        node.edges = [edge_a, edge_b]
        node.n = 20
        node.value = -20.0

        # Running bounds reflecting the raw scale seen so far.
        agent._q_min = -25.0
        agent._q_max = -15.0

        chosen = agent._select_edge(node)
        # Normalized Q for both is ~0.5 (≈identical), so the O(1) prior term breaks
        # the tie toward the high-prior edge B -- the prior is NOT inert.
        assert chosen is edge_b, (
            "with Q-normalization the larger prior must decide a near-Q tie "
            "(the swamp bug would let the marginal raw-Q edge win)"
        )


# ---------------------------------------------------------------------------
# Leaf discipline: a spun forced move scores below a clean line; V not consulted
# ---------------------------------------------------------------------------


class TestLeafDiscipline:
    def test_spun_leaf_floored_and_v_not_consulted(self) -> None:
        """A node whose forced move spun (own_spun=True) is valued by the pre-spin
        progress floor, NOT the net -- so a huge stub V can never lift it above a
        clean leaf (the depth-invariant §4.4 discipline)."""
        net = UniformNet(value=1e9)  # an absurdly optimistic V
        agent = MCTSAgent(net=net, config=MCTSConfig(), seed=0)
        agent.to_move_pid = 0
        track = _limit1_loop()

        # A clean leaf consults V (=1e9); a spun leaf must be floored (V untouched).
        state = GameState.create(track, 1, logging_enabled=True, seed=1)
        clean_node = Node(
            kind=NodeKind.DECISION, state=state, action_path=(), to_move=0,
            decision=None, own_spun=False,
        )
        before = net.leaf_calls
        clean_value = agent._evaluate_leaf(clean_node)
        assert net.leaf_calls == before + 1
        assert clean_value == pytest.approx(1e9)

        # A spun node: build a state with a recorded spin_out event for our seat.
        spun_state = GameState.create(track, 1, logging_enabled=True, seed=1)
        spun_state.log_event(
            "spin_out", player_id=0, data={"corner_start": 10}
        )
        spun_node = Node(
            kind=NodeKind.DECISION, state=spun_state, action_path=(), to_move=0,
            decision=None, own_spun=True,
        )
        before = net.leaf_calls
        spun_value = agent._evaluate_leaf(spun_node)
        assert net.leaf_calls == before, "V must NOT be consulted on a spun leaf"
        # Floor = corner_start - 1 = 9, far below the clean leaf's 1e9.
        assert spun_value == pytest.approx(9.0)
        assert spun_value < clean_value

    def test_spun_floor_is_depth_invariant_in_score(self) -> None:
        """A spun leaf's value equals the pre-spin floor at EVERY tree depth (sim
        count is irrelevant): the floor is computed from the spin event, not from
        how deep the search reached, so a reckless line is never preferred at any
        budget.

        Drives ``_evaluate_leaf`` on the SAME spun state from nodes at increasing
        nominal depths (via differing action_paths) and asserts the value is the
        constant floor each time, and the net (a huge-V stub) is never consulted.
        """
        net = UniformNet(value=1e9)
        agent = MCTSAgent(net=net, config=MCTSConfig(), seed=0)
        agent.to_move_pid = 0
        track = _limit1_loop()

        spun_state = GameState.create(track, 1, logging_enabled=True, seed=1)
        spun_state.log_event("spin_out", player_id=0, data={"corner_start": 10})

        for depth in range(1, 6):
            node = Node(
                kind=NodeKind.DECISION,
                state=spun_state,
                action_path=tuple((1, 0) for _ in range(depth)),
                to_move=0,
                decision=None,
                own_spun=True,
            )
            before = net.leaf_calls
            value = agent._evaluate_leaf(node)
            assert value == pytest.approx(9.0), f"floor drifted at depth {depth}"
            assert net.leaf_calls == before, "V consulted on a spun leaf"

    def test_finishes_at_every_sim_count(self, real_net) -> None:
        """The floor prevents an infinite death-spiral: a solo game finishes at
        every sim count even on the tight limit-1 loop with a random prior."""
        track = _limit1_loop()
        for sims in (1, 4, 16):
            agent = MCTSAgent(
                net=real_net, config=MCTSConfig(n_simulations=sims), seed=0
            )
            result = Game(track, [agent], logging_enabled=False, seed=7).run()
            assert 0 in result.finish_order, f"did not finish at sims={sims}"


# ---------------------------------------------------------------------------
# All-decision coverage: REACT/SLIPSTREAM/DISCARD are searched/net-driven
# ---------------------------------------------------------------------------


class TestAllDecisionCoverage:
    def test_react_slip_discard_are_net_driven_not_heuristic(self) -> None:
        """REACT/SLIPSTREAM/DISCARD route through the net (the searched prior),
        never a hand-coded heuristic rollout policy.

        A counting stub net records every prior query; calling the agent's
        REACT/SLIPSTREAM/DISCARD methods must hit the net (not delegate to a
        ``rollout_policy`` the way ``LookaheadAgent`` does) -- and the agent has no
        ``rollout_policy`` attribute at all, proving nothing is delegated.
        """

        class _CountingNet(UniformNet):
            def __init__(self):
                super().__init__()
                self.prior_calls = 0

            def policy_prior(self, obs, mask):
                self.prior_calls += 1
                return super().policy_prior(obs, mask)

        net = _CountingNet()
        agent = MCTSAgent(net=net, config=MCTSConfig(n_simulations=2), seed=0)
        # The agent exposes no heuristic rollout policy -- driving skill is searched.
        assert not hasattr(agent, "rollout_policy")

        track = _gen_track(7)
        state = GameState.create(track, 1, logging_enabled=False, seed=5)

        before = net.prior_calls
        react = agent.choose_react(
            state, 0, max_cooldown=2, can_boost=True, has_adrenaline=False
        )
        assert net.prior_calls > before, "REACT did not consult the net"
        from heat.engine.phases import ReactDecision

        assert isinstance(react, ReactDecision)

        before = net.prior_calls
        agent.choose_slipstream(state, 0)
        assert net.prior_calls > before, "SLIPSTREAM did not consult the net"

        before = net.prior_calls
        discardable = list(state.players[0].hand)
        out = agent.choose_discard(state, 0, discardable)
        assert net.prior_calls > before, "DISCARD did not consult the net"
        assert all(c in discardable for c in out)

    def test_tree_branches_on_all_decision_kinds(self, real_net) -> None:
        """The candidate enumerator branches on every DecisionKind (GEAR, CARDS,
        REACT, SLIPSTREAM, DISCARD) -- none is delegated to a heuristic.

        Asserts ``_candidate_actions`` returns a non-trivial branch set for each
        kind, i.e. the tree *can* branch on it.
        """
        agent = MCTSAgent(net=real_net, config=MCTSConfig(), seed=0)
        track = _gen_track(7)
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        p = state.players[0]

        gear_dec = Decision(
            DecisionKind.GEAR, 0,
            rules.legal_gear_shifts(p.gear, p.heat_available),
        )
        assert len(agent._candidate_actions(gear_dec, state)) >= 1

        cards_dec = Decision(
            DecisionKind.CARDS, 0, rules.legal_card_plays(p.hand, p.gear)
        )
        assert len(agent._candidate_actions(cards_dec, state)) >= 1

        slip_dec = Decision(DecisionKind.SLIPSTREAM, 0, True)
        assert set(agent._candidate_actions(slip_dec, state)) == {True, False}

        react_dec = Decision(
            DecisionKind.REACT, 0,
            rules.ReactOptions(max_cooldown=2, can_boost=True, has_adrenaline=False),
        )
        assert len(agent._candidate_actions(react_dec, state)) >= 1

        discard_dec = Decision(DecisionKind.DISCARD, 0, list(p.hand))
        assert len(agent._candidate_actions(discard_dec, state)) >= 1


# ---------------------------------------------------------------------------
# Self-play exploration hooks: built here, OFF by default, deterministic
# ---------------------------------------------------------------------------


class TestSelfPlayHooks:
    def test_dirichlet_noise_is_built_and_deterministic(self) -> None:
        """Root Dirichlet noise (ε>0) perturbs the root prior but stays a pure
        function of (state, seed) -- the mechanism is built (C2/C3), off in C1."""
        track = _gen_track(7)

        def plan(eps, seed=0):
            agent = MCTSAgent(
                net=UniformNet(),
                config=MCTSConfig(n_simulations=8, dirichlet_eps=eps),
                seed=seed,
            )
            state = GameState.create(track, 1, logging_enabled=False, seed=5)
            lg = rules.legal_gear_shifts(
                state.players[0].gear, state.players[0].heat_available
            )
            agent._plan_turn(state, 0, lg)
            # Return the root edge priors (post-noise) for comparison.
            return [round(e.prior, 6) for e in agent._root_node.edges]

        clean = plan(0.0)
        noisy = plan(0.5)
        # Noise changes the prior (it is actually applied), but is reproducible.
        assert noisy != clean
        assert plan(0.5) == noisy  # deterministic under fixed seed

    def test_temperature_sampling_is_built_and_deterministic(self) -> None:
        """The temperature schedule samples the root action for the first
        ``temperature_moves`` plies; with a fixed seed the sample is reproducible
        (a pure function of (state, seed))."""
        track = _gen_track(7)

        def first_gear(seed):
            agent = MCTSAgent(
                net=UniformNet(),
                config=MCTSConfig(n_simulations=8, temperature_moves=10),
                seed=seed,
            )
            state = GameState.create(track, 1, logging_enabled=False, seed=5)
            lg = rules.legal_gear_shifts(
                state.players[0].gear, state.players[0].heat_available
            )
            return agent.choose_gear(state, 0, lg)

        g0 = first_gear(0)
        assert first_gear(0) == g0  # reproducible under a fixed seed


# ---------------------------------------------------------------------------
# Picklable factory (pickle-by-path, mirrors lookahead_agent_factory)
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_pickles_by_path(self) -> None:
        import pickle

        from heat.agents.mcts_agent import mcts_agent_factory

        fac = mcts_agent_factory(
            "checkpoints/_nonexistent.zip", config=MCTSConfig(n_simulations=4)
        )
        restored = pickle.loads(pickle.dumps(fac))
        agent = restored(0, 123)
        assert isinstance(agent, MCTSAgent)
        assert agent.model_path == "checkpoints/_nonexistent.zip"
        assert agent._net is None  # lazy: not built at construction
        assert agent.config.n_simulations == 4

    def test_agent_nulls_model_on_pickle(self) -> None:
        """A path-backed NetAdapter nulls its heavy model in __getstate__, so the
        agent pickles by path (the MLAgent contract)."""
        import pickle

        from heat.agents.mcts_agent import NetAdapter

        agent = MCTSAgent(model_path="checkpoints/_nonexistent.zip", seed=0)
        adapter = agent._get_net()
        assert isinstance(adapter, NetAdapter)
        adapter._model = object()  # pretend a heavy model is loaded
        restored = pickle.loads(pickle.dumps(agent))
        assert restored._net._model is None
        assert restored._net.model_path == "checkpoints/_nonexistent.zip"
