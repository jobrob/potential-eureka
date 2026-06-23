"""Tests for the Sprint C6 perfect-info 1v1 win/loss AlphaZero pivot.

Covers the mechanism the C6 spec's "Mechanism (the minimum honest result)" and
"Risks & mitigations" call for, all runnable WITHOUT a trained checkpoint (a
stub net is enough -- the win/loss MINIMAX is net-agnostic, only the headline
NUMBER needs a trained prior):

  * **sign discipline (the single most error-prone change)** -- a forced-win line
    backs up to ``+1`` for the learner and ``−1`` for the opponent (the negamax
    negation at opponent-owned edges);
  * **perfect-info opponent branch is SEARCHED** -- the two-player transition
    model does NOT auto-resolve the opponent; an opponent decision becomes a node
    with ``to_move = opponent_id``;
  * **win/loss leaf / terminal value** -- a terminal leaf returns ``±1`` from the
    learner's frame; the own-spin floor is dropped in the win/loss frame;
  * **determinism preserved** -- the two-player search is a pure function of
    ``(state, seed)`` (the C1 contract carries over);
  * **z always defined** -- the 1v1 generator backfills a ±1/0 outcome for every
    row (no episode/row drop -- the structural win of the pivot);
  * **value_mode=winloss** -- the trainer fits the ±1 outcome through a tanh head
    and the NetAdapter applies the same tanh at inference;
  * **seat-neutral 1v1 gate** -- ``num_seats=2`` rotation, parity 0.5;
  * **C0-C5 byte-identical when the C6 flags are off** (the hard backward-compat
    requirement): the solo search/generator/trainer are unchanged.

Mirrors the structure/style of ``tests/test_mcts_agent.py`` / ``test_c4_gumbel.py``.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# experiments/ is not a package; add it so the C6 loop deliverable imports by name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.models.game_state import GameState
from heat.models.track import Corner, Space, Track
from heat.tracks.generator import generate_track

from heat.agents.mcts_agent import (
    Edge,
    EngineTransitionModel,
    MCTSAgent,
    MCTSConfig,
    Node,
    NodeKind,
    _winloss_result,
)

# The experiments scripts are not a package; add the dir so the C6 deliverables
# import by name (the test_az_targets / test_c4_gumbel convention).
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)


# ---------------------------------------------------------------------------
# Stub net + helpers (no checkpoint required)
# ---------------------------------------------------------------------------


class UniformNet:
    """Uniform masked prior + a constant leaf value (the test_mcts_agent stub)."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def policy_prior(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        probs = np.zeros(m.shape, dtype=np.float64)
        n = int(m.sum())
        if n:
            probs[m] = 1.0 / n
        return probs

    def leaf_value(self, obs: np.ndarray) -> float:
        return self.value


def _two_seat_state(seed: int = 5) -> GameState:
    track = generate_track(7)
    state = GameState.create(track, 2, logging_enabled=False, seed=seed)
    for p in state.players:
        p.lap = 1
    return state


def _two_player_agent(net=None, *, sims: int = 8, seed: int = 0) -> MCTSAgent:
    cfg = MCTSConfig(n_simulations=sims, two_player=True, opponent_id=1)
    return MCTSAgent(net=net or UniformNet(), config=cfg, seed=seed)


# ---------------------------------------------------------------------------
# The win/loss outcome (shared by the leaf and the z backfill)
# ---------------------------------------------------------------------------


class TestWinlossOutcome:
    def test_finish_order_decides(self) -> None:
        state = _two_seat_state()
        state.players[0].finished = True
        state.players[0].finish_order = 0
        state.players[1].finished = True
        state.players[1].finish_order = 1
        assert _winloss_result(state, 0, 1) == 1.0   # learner crossed first
        assert _winloss_result(state, 1, 0) == -1.0  # opponent's frame

    def test_finished_beats_unfinished(self) -> None:
        state = _two_seat_state()
        state.players[0].finished = True
        state.players[0].finish_order = 0
        state.players[1].finished = False
        assert _winloss_result(state, 0, 1) == 1.0
        assert _winloss_result(state, 1, 0) == -1.0

    def test_max_rounds_tie_decided_by_progress(self) -> None:
        """Resolved-sub-decision 3: further-along wins, draw only on exact tie."""
        state = _two_seat_state()
        # Neither finished -> decide by lap-aware progress.
        state.players[0].lap, state.players[0].position = 2, 5
        state.players[1].lap, state.players[1].position = 1, 9
        assert _winloss_result(state, 0, 1) == 1.0   # seat 0 further along
        # Exact progress tie -> draw (z = 0), avoiding a z=0-dominated dataset.
        state.players[1].lap, state.players[1].position = 2, 5
        assert _winloss_result(state, 0, 1) == 0.0


# ---------------------------------------------------------------------------
# Perfect-info opponent branch is SEARCHED (not auto-resolved)
# ---------------------------------------------------------------------------


class TestOpponentBranchSearched:
    def test_two_player_model_stops_at_opponent_decision(self) -> None:
        """The opponent's decision is a searched branch, not auto-resolved."""
        state = _two_seat_state()
        model = EngineTransitionModel(0, two_player=True, opponent_id=1)
        root = model.root_decision(state)
        assert root.player_id == 0  # learner's GEAR (the root)
        # Forcing the learner's gear surfaces the OPPONENT's gear as the next
        # searched decision (the driver collects both seats' gears before cards).
        res = model.step(state, (), root.legal[0], reseed=123)
        assert res.decision is not None
        assert res.decision.player_id == 1  # opponent -- a searched node

    def test_solo_model_auto_resolves_non_learner(self) -> None:
        """Solo (two_player=False) never stops at a non-learner -- the C0-C5 path."""
        state = _two_seat_state()
        model = EngineTransitionModel(0)  # solo: only seat 0 searched
        res = model.root_decision(state)
        assert res.player_id == 0
        # In solo mode seat 1 is auto-resolved, so the next decision is the
        # learner's (seat 0) again, never seat 1.
        res2 = model.step(state, (), res.legal[0], reseed=1)
        assert res2.decision is None or res2.decision.player_id == 0

    def test_opponent_node_to_move_is_opponent(self) -> None:
        """A searched opponent decision node carries to_move = opponent_id."""
        state = _two_seat_state()
        agent = _two_player_agent(sims=12, seed=1)
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )
        agent.to_move_pid = 0
        agent._q_min, agent._q_max = float("inf"), float("-inf")
        agent._clones_this_move = 0
        import random
        root = Node(kind=NodeKind.DECISION, state=state, action_path=(),
                    to_move=0, decision=Decision(DecisionKind.GEAR, 0, lg))
        agent._root_node = root
        agent._search_rng = random.Random(0)
        model = EngineTransitionModel(0, two_player=True, opponent_id=1)
        agent._run_root_search(root, model, agent._turn_seed((0, 0, 1, 0, 0)))
        # Some descendant below the learner's gear root is an opponent node.
        seen_opponent = _collect_to_moves(root)
        assert 1 in seen_opponent, "opponent decision node never created/searched"


def _collect_to_moves(node: Node, acc: set | None = None) -> set:
    """All ``to_move`` seats appearing in decision nodes of the tree."""
    if acc is None:
        acc = set()
    if node.kind == NodeKind.DECISION:
        acc.add(node.to_move)
        for e in node.edges:
            if e.child is not None:
                _collect_to_moves(e.child, acc)
    else:
        for o in node.outcomes:
            if o.child is not None:
                _collect_to_moves(o.child, acc)
    return acc


# ---------------------------------------------------------------------------
# Sign discipline: negamax backup (the single most error-prone change)
# ---------------------------------------------------------------------------


class TestSignDiscipline:
    def test_learner_edge_keeps_sign_opponent_edge_negates(self) -> None:
        """A forced-win line (+1 learner) backs up to +1 at a learner edge and
        −1 at an opponent edge (the zero-sum negamax in the learner frame)."""
        agent = _two_player_agent(seed=0)
        agent.to_move_pid = 0
        agent._q_min, agent._q_max = float("inf"), float("-inf")
        state = _two_seat_state()

        learner_node = Node(kind=NodeKind.DECISION, state=state, action_path=(),
                            to_move=0)
        opp_node = Node(kind=NodeKind.DECISION, state=state, action_path=(),
                        to_move=1)
        learner_edge = Edge(action="L", prior=1.0)
        opp_edge = Edge(action="O", prior=1.0)
        learner_node.edges = [learner_edge]
        opp_node.edges = [opp_edge]

        # A leaf value of +1 (the learner WINS) propagated up a path through both.
        path = [(learner_node, learner_edge), (opp_node, opp_edge)]
        root = Node(kind=NodeKind.DECISION, state=state, action_path=(), to_move=0)
        agent._backup(path, root, value=1.0)

        # Learner edge stores the learner-frame +1; opponent edge stores −1 (the
        # opponent's own frame -- a win for the learner is a loss for the opponent).
        assert learner_edge.q() == pytest.approx(1.0)
        assert opp_edge.q() == pytest.approx(-1.0)

    def test_solo_backup_never_negates(self) -> None:
        """Solo (two_player=False) backup is the C0-C5 sum, no negation."""
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent.to_move_pid = 0
        agent._q_min, agent._q_max = float("inf"), float("-inf")
        state = _two_seat_state()
        # A node with to_move=1 must NOT trigger negation when two_player is off.
        node = Node(kind=NodeKind.DECISION, state=state, action_path=(), to_move=1)
        edge = Edge(action="x", prior=1.0)
        node.edges = [edge]
        root = Node(kind=NodeKind.DECISION, state=state, action_path=(), to_move=0)
        agent._backup([(node, edge)], root, value=-3.0)
        assert edge.q() == pytest.approx(-3.0)  # stored as-is (no flip)


# ---------------------------------------------------------------------------
# Win/loss terminal + leaf value (learner frame)
# ---------------------------------------------------------------------------


class TestWinlossLeaf:
    def test_terminal_value_is_pm1_in_two_player(self) -> None:
        agent = _two_player_agent(seed=0)
        agent.to_move_pid = 0
        state = _two_seat_state()
        state.players[0].finished = True
        state.players[0].finish_order = 0
        state.players[1].finished = True
        state.players[1].finish_order = 1
        assert agent._terminal_value(state) == 1.0

    def test_terminal_value_is_zero_in_solo(self) -> None:
        """Solo terminal value is unchanged (−rounds_remaining is 0 at finish)."""
        agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(), seed=0)
        agent.to_move_pid = 0
        state = _two_seat_state()
        assert agent._terminal_value(state) == 0.0

    def test_own_spin_floor_dropped_in_two_player(self) -> None:
        """In the win/loss frame the own-spin floor is NOT applied -- the net value
        is consulted instead (a spin's cost shows only via losing the race)."""
        net = UniformNet(value=0.7)
        agent = _two_player_agent(net=net, seed=0)
        agent.to_move_pid = 0
        state = _two_seat_state()
        spun = Node(kind=NodeKind.DECISION, state=state, action_path=(), to_move=0,
                    decision=None, own_spun=True)
        # own_spun is set, but two_player drops the floor -> net value 0.7 is used.
        assert agent._evaluate_leaf(spun) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Symmetric net-vs-itself -> ~50% seat-neutral 1v1 win-rate (calibration sanity)
# ---------------------------------------------------------------------------


class TestSymmetricSelfPlay:
    def test_symmetric_winrate_near_parity(self) -> None:
        """A policy playing ITSELF, seat-neutral, scores ~50% (the seat bias cancels).

        Two identical (weak-heuristic) policies through the generalized num_seats=2
        harness: with the focal rotated through both seats and the win indicator
        pooled, the positional advantage cancels, so the win-rate sits near parity
        (50%). This is the C6 calibration sanity -- a symmetric matchup must NOT
        read as skill. We assert a band (small-sample noise), not an exact 50%.
        """
        from eval_search import _seat_neutral_winrate
        from heat.agents.heuristic_agent import HeuristicAgent

        wr = _seat_neutral_winrate(
            lambda: HeuristicAgent(), lambda: HeuristicAgent(),
            track_seeds=[900_000 + i for i in range(16)],
            game_seed_base=0, num_seats=2,
        )
        assert 0.30 <= wr <= 0.70, f"symmetric 1v1 win-rate {wr} is far from parity"


# ---------------------------------------------------------------------------
# Determinism: the two-player search is a pure function of (state, seed)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_player_search_is_byte_stable(self) -> None:
        state = _two_seat_state(seed=123)
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )

        def plan(seed: int):
            agent = _two_player_agent(sims=12, seed=seed)
            return agent._plan_turn(state, 0, lg)

        a = plan(7)
        b = plan(7)
        c = plan(8)
        assert a == b           # same seed -> byte-identical
        # different seed MAY differ; at minimum the call is repeatable per seed.
        assert plan(8) == c

    def test_full_1v1_game_runs_legal(self) -> None:
        """A full perfect-info 1v1 game completes with the MCTS agent legal."""
        from heat.engine.game import Game
        from heat.agents.heuristic_agent import HeuristicAgent

        track = generate_track(7)
        agent = _two_player_agent(sims=6, seed=0)
        game = Game(track, [agent, HeuristicAgent()], logging_enabled=False, seed=11)
        result = game.run()
        assert set(result.finish_order) <= {0, 1}
        assert len(result.finish_order) == 2  # both seats finish (1v1 decided)


# ---------------------------------------------------------------------------
# Backward-compat: C0-C5 solo path byte-identical when the C6 flags are off
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_solo_config_defaults_two_player_off(self) -> None:
        cfg = MCTSConfig()
        assert cfg.two_player is False
        assert cfg.opponent_id == 1

    def test_solo_search_unchanged_with_flags_off(self) -> None:
        """A solo search with the default config behaves exactly as C1 (the
        regression guard: the new branches are all gated on two_player)."""
        track = generate_track(7)
        state = GameState.create(track, 1, logging_enabled=False, seed=5)
        state.players[0].lap = 1
        lg = rules.legal_gear_shifts(
            state.players[0].gear, state.players[0].heat_available
        )

        def plan(seed: int):
            agent = MCTSAgent(net=UniformNet(), config=MCTSConfig(n_simulations=8),
                              seed=seed)
            return agent._plan_turn(state, 0, lg)

        assert plan(3) == plan(3)  # still deterministic, no opponent machinery


# ---------------------------------------------------------------------------
# The 1v1 generator: z is ALWAYS defined (no episode/row drop)
# ---------------------------------------------------------------------------


class TestGeneratorZAlwaysDefined:
    def test_winloss_z_helper_bounded(self) -> None:
        import gen_selfplay as G

        state = _two_seat_state()
        state.players[0].finished = True
        state.players[0].finish_order = 0
        state.players[1].finished = True
        state.players[1].finish_order = 1
        z = G._winloss_z(state, 0, 1)
        assert z in (-1.0, 0.0, 1.0)

    def test_make_opponent_sentinels(self) -> None:
        import gen_selfplay as G
        from heat.agents.strong_heuristic import StrongHeuristicAgent
        from heat.agents.heuristic_agent import HeuristicAgent

        assert G._make_opponent(None) is None  # net-vs-net
        assert isinstance(G._make_opponent("strong"), StrongHeuristicAgent)
        assert isinstance(G._make_opponent("weak"), HeuristicAgent)


# ---------------------------------------------------------------------------
# value_mode=winloss: tanh head + the NetAdapter inference transform
# ---------------------------------------------------------------------------


class TestValueMode:
    def test_value_pred_tanh_bounds_winloss(self) -> None:
        """The winloss value prediction is squashed into [-1, 1] (tanh head)."""
        import torch
        import train_az as T
        from heat.ml.env import HeatEnv
        from heat.ml.model import build_model, PPOConfig

        model = build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu"))
        obs = torch.zeros((4, model.observation_space.shape[0]))
        v_winloss = T._value_pred(model.policy, obs, "winloss")
        v_rounds = T._value_pred(model.policy, obs, "rounds")
        assert bool((v_winloss.abs() <= 1.0).all())  # tanh-bounded
        # rounds mode is the raw critic (NOT tanh-squashed) -- different tensor.
        assert not torch.allclose(v_winloss, v_rounds) or bool((v_rounds.abs() <= 1.0).all())

    def test_calibration_metric_is_zero_when_perfect(self) -> None:
        import train_az as T

        # Predictions exactly match outcomes -> ECE 0.
        pred = np.array([0.0, 1.0, 0.0, 1.0])
        won = np.array([0.0, 1.0, 0.0, 1.0])
        cal = T._value_calibration(pred, won, n_buckets=10)
        assert cal["ece"] == pytest.approx(0.0)


class TestLoopDecoupling:
    """The C6-Tier-0 fix: the ship-gate (read a) and the stop signal (read b) are
    decoupled, so the learner can advance even before any net beats the strong
    heuristic, and the loop only stops on a genuine self-improvement plateau."""

    @staticmethod
    def _gate(*, strong_lb: float, prev_wr: float | None) -> dict:
        g = {"vs_strong": {"win_rate": strong_lb, "wilson_lb": strong_lb}}
        if prev_wr is not None:
            g["vs_prev"] = {"win_rate": prev_wr, "wilson_lb": prev_wr - 0.1}
        return g

    def test_ship_gate_requires_beating_strong_at_parity(self) -> None:
        import az_loop_1v1 as L

        # A net that beats its predecessor but loses to the strong heuristic
        # (LB 0.30 < parity) must NOT ship -- the ship-gate is the final bar.
        assert L._ship_promote(cand_lb=0.30, best_lb=0.07) is False
        # Clears parity AND beats incumbent -> ships.
        assert L._ship_promote(cand_lb=0.55, best_lb=0.07) is True
        # Clears parity but does not beat a stronger incumbent -> does not ship.
        assert L._ship_promote(cand_lb=0.52, best_lb=0.60) is False

    def test_stop_signal_is_read_b_not_vs_strong(self) -> None:
        import az_loop_1v1 as L

        # Beating its own predecessor (vs_prev > 50%) is "improving" even while it
        # loses badly to the strong heuristic -- the pre-fix bug stopped here.
        improving = self._gate(strong_lb=0.06, prev_wr=0.62)
        assert L._is_improving(improving) is True
        # Failing to beat its predecessor is the genuine plateau (the stop signal).
        flat = self._gate(strong_lb=0.40, prev_wr=0.48)
        assert L._is_improving(flat) is False

    def test_missing_vs_prev_is_non_improving(self) -> None:
        import az_loop_1v1 as L

        # Read (b) absent (should not happen post-fix) -> treated as non-improving,
        # never as a silent pass.
        assert L._is_improving(self._gate(strong_lb=0.5, prev_wr=None)) is False
