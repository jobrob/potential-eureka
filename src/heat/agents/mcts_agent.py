"""Solo net-guided stochastic MCTS over the real engine (Sprint C1).

``MCTSAgent`` is the complete-core AlphaZero search the Option-C design
(``docs/solo-speed-planning/option-C-search-learning/README.md`` §4) specifies:
a single-agent, stochastic-shortest-path MCTS that uses the **real engine** for
every transition, realizes our own card draws as **in-tree chance nodes** (true
expectimax with double progressive widening, NOT PIMC determinization), and uses
a fixed policy+value net for the PUCT prior and the leaf value. It reuses the
S1 ``LookaheadAgent`` machinery wholesale: the dedup-by-resulting-speed candidate
prune, the deterministic per-(node, sample) ``reseed`` clone scheme, the
own-spin / pre-spin-progress leaf discipline, and the always-on ``SearchProfile``.

The five seams (README §3)
--------------------------
The MCTS *core* (``_run_search``) depends ONLY on five small strategy
interfaces, never on the concrete engine, so the deferred forks (MuZero,
opponents, Gumbel, learned chance, A/B-V warm-start) drop in as alternate
implementations of the same interface with no rewrite:

  * :class:`TransitionModel` -- ``legal_decisions`` + ``step`` (core =
    ``GameState.clone`` + ``run_round_driver``). Deferred: MuZero learned dynamics.
  * :class:`Node` (with ``to_move`` + ``kind`` decision/chance) -- solo: every
    decision node's owner is the learner. Deferred: an opponent decision node +
    hidden-info determinization (``LookaheadAgent._determinize_opponents``).
  * :class:`SearchPolicy` -- prior -> selection (core = log-PUCT + optional
    Dirichlet, OFF by default). Deferred: Gumbel-AlphaZero top-k selection.
  * :class:`LeafEvaluator` -- value at a leaf (core = net value head, floored at
    ``_pre_spin_progress`` on a spun forced move). Deferred: A/B-V warm-start.
  * :class:`ChanceExpansion` -- how chance outcomes are sampled/widened (core =
    double progressive widening over engine reseeds). Deferred: full enumeration
    / learned chance (Stochastic MuZero).

The algorithm (README §4) -- the parts that must be exactly right
-----------------------------------------------------------------
* **Selection** (§4.1) maximizes ``Q̂(s,a) + c_puct(s)·P(s,a)·√ΣN/(1+N(s,a))``
  with ``c_puct(s) = c_init + log((ΣN + c_base + 1)/c_base)``. ``Q̂`` is
  **min-max normalized** to ``[0,1]`` by the tree's running ``[Qmin, Qmax]``
  (MANDATORY: C0 §D measured that the raw ``−rounds_remaining`` value scale flips
  the argmax 10% of the time and crushes selection entropy 0.86 -> 0.55, i.e. the
  prior goes inert). FPU: an unvisited child takes ``Q̂ = Q̂(parent) −
  fpu_reduction`` so the *prior*, not an optimistic default, governs which
  unexplored action is tried first. ``P`` is the net's masked policy read over the
  kept candidate set (the S1 dedup-by-speed prune for CARDS) and renormalized.
* **Chance nodes** (§4.3): the only stochastic transition in solo is the
  replenish **draw at the round boundary** (and the ~15% reshuffle it can
  trigger). C0 §A measured that detecting a chance edge by "RNG consumed" misses
  85% of draws (a plain draw off an ordered deck consumes no RNG), so the chance
  edge is **the round boundary itself** (``round_num`` advanced), detected not
  assumed. Outcomes are sampled with a deterministic per-(node, sample)
  ``reseed`` and fanned out by DPW ``⌈C_pw·N^α_pw⌉`` capped at ``K``; the backed-up
  value is the **visit-weighted average** over outcomes -- expectimax-in-tree.
* **Leaf** (§4.4): the net value head on ``encode_observation(state, pid,
  decision=None)`` (the A1/A2 ``−rounds_remaining`` sign, higher-is-better,
  returned as-is with NO second negation). When reaching the leaf required our
  forced move to spin (``own_spins > 0``) the value is **floored at
  ``_pre_spin_progress`` and the net is not consulted** -- the depth-invariant S1
  discipline so a reckless recovery line is never preferred at depth.

Determinism (the unit-test contract)
-------------------------------------
The whole search -- selection, expansion, and chance sampling -- is a **pure
function of (root state, search seed)**. No global RNG is ever touched: every
clone uses an explicit ``reseed=`` derived from the agent ``seed`` mixed with the
turn signature and per-(node, sample) indices (the S1 ``_turn_seed`` /
``_score_plan`` scheme). Same state + same seed => byte-stable move.

The fixed net
-------------
The prior + leaf value are backed by :class:`NetAdapter`, a thin wrapper over a
loaded SB3 ``MaskablePPO`` (``policy.get_distribution(..., action_masks=mask)``
for the prior; ``policy.predict_values`` for the value). It validates the
checkpoint ``.meta.json`` against the live ML contract exactly like
:class:`heat.agents.ml_agent.MLAgent` (the §3.4 tripwire,
:class:`~heat.agents.ml_agent.CheckpointMismatchError`). C2/C3 swap the checkpoint
without touching the search. The agent pickles **by path** (the model is nulled in
``__getstate__``), so :func:`mcts_agent_factory` ships into the eval harness /
``ProcessPoolExecutor`` carrying only a path string -- the MLAgent / S1 pattern.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent
from heat.agents import _move_eval as ME
from heat.agents.search_agent import SearchProfile


# ---------------------------------------------------------------------------
# Sprint C6: the zero-sum 1v1 win/loss outcome (shared by the search leaf and the
# gen_selfplay z backfill so the two agree on the same binary result)
# ---------------------------------------------------------------------------


def _winloss_result(state: GameState, learner_id: int, opponent_id: int) -> float:
    """The learner's zero-sum 1v1 outcome at ``state``: ``+1`` / ``−1`` / ``0``.

    Decided by **finish order** when at least one seat has finished (the seat with
    the smaller ``finish_order`` crossed first and wins; a finished seat beats an
    unfinished one). When NEITHER has finished -- the ``MAX_ROUNDS`` tie case --
    the further-along seat wins by **race progress** (lap-aware spaces travelled,
    resolved-sub-decision-3), with ``0`` only on an exact progress tie. Pure
    function of ``(state, learner_id, opponent_id)`` so the search leaf and the
    training target backfill compute the identical outcome.
    """
    learner = state.get_player(learner_id)
    opp = state.get_player(opponent_id)

    if learner.finished or opp.finished:
        if learner.finished and opp.finished:
            # Both crossed: the smaller finish_order finished first (wins).
            if learner.finish_order < opp.finish_order:
                return 1.0
            if learner.finish_order > opp.finish_order:
                return -1.0
            return 0.0
        # Exactly one finished -- it wins (a finished seat beats an unfinished one).
        return 1.0 if learner.finished else -1.0

    # Neither finished (MAX_ROUNDS tie): decide by race progress, draw on a tie.
    lp = learner.lap * state.track.length + learner.position
    op = opp.lap * state.track.length + opp.position
    if lp > op:
        return 1.0
    if lp < op:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# C0 chosen constants (docs/.../C0-findings.md "The chosen constant set")
# ---------------------------------------------------------------------------

#: Per-move simulation budget. C0 §F: 16 sims = 3.2x LookaheadAgent (inside the
#: 2-5x bar); 32 is already 6.4x. The cost is clone-bound, not net-bound.
DEFAULT_N_SIMULATIONS: int = 16

#: Double-progressive-widening constants for chance nodes (C0 §C). With
#: ``C_pw=1.0, α_pw=0.5`` the fan-out ``⌈C_pw·N^α_pw⌉`` reaches the hard cap
#: ``K=8`` only at N≈64 visits; C0's variance check showed the backed-up chance
#: value stabilizes (std < 0.5) by fan-out 4-8, so K=8 is bounded AND stable.
DEFAULT_C_PW: float = 1.0
DEFAULT_ALPHA_PW: float = 0.5
DEFAULT_K: int = 8

#: log-PUCT selection constants (AZ/MuZero defaults; C0 to-confirm vs the pruned
#: width, kept as defensible defaults).
DEFAULT_C_INIT: float = 1.25
DEFAULT_C_BASE: float = 19652.0
DEFAULT_FPU_REDUCTION: float = 0.25

#: Self-play exploration sizing (C0; OFF for C1's deterministic parity eval).
DEFAULT_DIRICHLET_ALPHA: float = 0.5
DEFAULT_DIRICHLET_EPS: float = 0.25
#: Visit-count temperature schedule: τ=1 for the first ``temperature_moves``
#: plies of an episode, then τ→0 (greedy). OFF by default (greedy from move 0).
DEFAULT_TEMPERATURE_MOVES: int = 10

#: Own-spin / later-spin leaf penalties -- reused verbatim from the S1 leaf
#: discipline (search_agent.DEFAULT_OWN_SPIN_PENALTY / DEFAULT_SPIN_PENALTY) so a
#: spinning forced move stays terminal-dominated at every sim count / depth.
DEFAULT_OWN_SPIN_PENALTY: float = 1000.0
DEFAULT_LATER_SPIN_PENALTY: float = 11.0

#: Gumbel-AlphaZero RootActionSelector constants (Sprint C4). ``DEFAULT_GUMBEL_M``
#: is the number of root actions sampled without replacement by the Gumbel top-k
#: trick (capped at the kept-candidate count and at a value Sequential Halving can
#: afford given ``n_simulations``). ``c_visit`` / ``c_scale`` parameterize the
#: monotone σ transform ``σ(q) = (c_visit + max_b N_b) · c_scale · q`` applied to
#: the ALREADY-NORMALIZED ``[0,1]`` Q̂ (so the unbounded −rounds_remaining scale
#: stays tamed -- the C0 §D Q-scale hazard, now at the root).
#:
#: Constants: Danihelka et al. (2022) publish ``c_visit=50, c_scale=1.0``. Sprint
#: C4 decision #1 says start there and **retune only if the non-collapse test
#: fails** -- which it did: at our LOW (16) sim budget and [0,1] Q-normalization,
#: ``(c_visit + max_n)·c_scale ≈ 58`` swamps the log-prior so σ(Q̂) collapses the
#: completed-Q target to one-hot at narrow (GEAR) decisions (measured GEAR
#: π-entropy 0.002 -- a swamped σ, the documented hazard). The published constants
#: assume Danihelka's HUNDREDS-of-sims / [-1,1] value regime; at 16 sims the σ
#: magnitude must shrink to keep the prior in play. The retune below (``c_visit=25,
#: c_scale=0.25``) restores a non-collapsed, graded completed-Q target (GEAR
#: entropy ~0.22, CARDS ~0.86, both well above the ~0.04 PUCT visit-count floor)
#: while σ still bites (mass shifts toward higher-Q actions -- the
#: improvement-guarantee test pins this). The behavioral one-cycle CI gate is the
#: final arbiter (Sprint C4 success criterion 2).
DEFAULT_GUMBEL_M: int = 8
DEFAULT_GUMBEL_C_VISIT: float = 25.0
DEFAULT_GUMBEL_C_SCALE: float = 0.25


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MCTSConfig:
    """Hyperparameters for the solo stochastic MCTS (all defaults = C0 constants).

    Every field is a defensible C0 default so nothing is left blank; the
    self-play exploration knobs (``dirichlet_*`` / ``temperature_*``) default OFF
    so C1's parity eval is a deterministic function of ``(state, seed)``.
    """

    #: Per-move simulation budget (root-to-leaf descents).
    n_simulations: int = DEFAULT_N_SIMULATIONS

    # --- chance-node double progressive widening (§4.3) ---
    c_pw: float = DEFAULT_C_PW
    alpha_pw: float = DEFAULT_ALPHA_PW
    k_cap: int = DEFAULT_K

    # --- log-PUCT selection + FPU (§4.1) ---
    c_init: float = DEFAULT_C_INIT
    c_base: float = DEFAULT_C_BASE
    fpu_reduction: float = DEFAULT_FPU_REDUCTION

    # --- leaf discipline (§4.4) ---
    own_spin_penalty: float = DEFAULT_OWN_SPIN_PENALTY
    later_spin_penalty: float = DEFAULT_LATER_SPIN_PENALTY

    # --- self-play exploration (C2/C3) -- OFF for C1's deterministic eval ---
    #: Root Dirichlet noise weight; 0.0 (default) disables noise entirely.
    dirichlet_eps: float = 0.0
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    #: Plies of an episode that sample ``a ∝ N(a)^{1/τ}`` at τ=1 before τ→0.
    #: 0 (default) is greedy from move 0 (the deterministic eval mode).
    temperature_moves: int = 0

    #: Act on ``argmax Q̂`` instead of the most-visited root child. The README
    #: default acting rule is most-visited (``act_on_q=False``); ``argmax Q̂`` is
    #: a deterministic-eval option. Both are deterministic with noise off.
    act_on_q: bool = False

    # --- two-player perfect-info minimax seam (Sprint C6) ---
    #: ``False`` (default) keeps the C0-C5 solo search byte-for-byte: every
    #: yielded :class:`Decision` is the learner's (any non-learner decision in the
    #: replay is auto-resolved with a legal default). ``True`` activates the
    #: perfect-information two-player minimax: the opponent's decisions are NOT
    #: auto-resolved -- they become DECISION nodes with ``to_move = opponent_id``,
    #: branched on the opponent's own candidate set with the SAME net's prior read
    #: from the opponent's observation, and the backup negates the increment at
    #: opponent-owned edges (zero-sum negamax stored in the learner's frame). The
    #: chance nodes, interior PUCT, Q-norm, and Gumbel root selector are unchanged.
    two_player: bool = False
    #: The opponent seat id for the two-player minimax (only consulted when
    #: ``two_player`` is on). Defaults to ``1`` (the other seat in a 1v1 game);
    #: the learner seat is the ``player_id`` the search is rooted at.
    opponent_id: int = 1

    # --- root action selector seam (README §3 RootActionSelector; Sprint C4) ---
    #: ``"puct"`` (default) is the byte-for-byte-unchanged C1/C2/C3 path: log-PUCT
    #: descent at the root + the visit-count target. ``"gumbel"`` swaps in the
    #: Gumbel-AlphaZero selector (Gumbel top-``m`` sampling + Sequential Halving for
    #: the acted action, completed-Q policy target in ``gen_selfplay``), used ONLY
    #: to build self-play training targets -- the agent still ACTS and is EVALUATED
    #: with the PUCT search regardless of this toggle (Sprint C4 scope decision #3).
    root_selector: str = "puct"
    #: Gumbel top-``m``: the number of root actions sampled without replacement
    #: (capped at the kept-candidate count and at what Sequential Halving can afford).
    gumbel_m: int = DEFAULT_GUMBEL_M
    #: σ transform constants for ``σ(Q̂) = (c_visit + max_b N_b)·c_scale·Q̂``.
    gumbel_c_visit: float = DEFAULT_GUMBEL_C_VISIT
    gumbel_c_scale: float = DEFAULT_GUMBEL_C_SCALE

    def __post_init__(self) -> None:
        if self.n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {self.n_simulations}")
        if self.k_cap < 1:
            raise ValueError(f"k_cap must be >= 1, got {self.k_cap}")
        if self.c_pw <= 0.0:
            raise ValueError(f"c_pw must be > 0, got {self.c_pw}")
        if not (0.0 <= self.dirichlet_eps <= 1.0):
            raise ValueError(
                f"dirichlet_eps must be in [0, 1], got {self.dirichlet_eps}"
            )
        if self.root_selector not in ("puct", "gumbel"):
            raise ValueError(
                f"root_selector must be 'puct' or 'gumbel', got {self.root_selector!r}"
            )
        if self.gumbel_m < 1:
            raise ValueError(f"gumbel_m must be >= 1, got {self.gumbel_m}")
        if self.gumbel_c_visit < 0.0:
            raise ValueError(
                f"gumbel_c_visit must be >= 0, got {self.gumbel_c_visit}"
            )
        if self.gumbel_c_scale <= 0.0:
            raise ValueError(
                f"gumbel_c_scale must be > 0, got {self.gumbel_c_scale}"
            )


# ---------------------------------------------------------------------------
# Seam 1: TransitionModel
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """The outcome of advancing the engine across exactly one learner decision.

    Attributes:
        state: The clone advanced past the forced ``action``, positioned at the
            next learner decision (or terminal / round end).
        decision: The next learner :class:`Decision` to branch on, or ``None`` if
            the engine reached the end of the game (terminal leaf).
        is_chance: True iff advancing across this edge crossed the round boundary
            (``round_num`` advanced) -- the §4.3 chance edge (a replenish draw
            fired). Detected, not assumed.
        terminal: True iff ``state.is_game_over`` (the race finished).
    """

    state: GameState
    decision: Decision | None
    is_chance: bool
    terminal: bool


class TransitionModel(Protocol):
    """Seam 1: forward transitions in the tree (README §3, core = real engine).

    The MCTS core depends only on this protocol, never on ``run_round_driver``
    directly -- which is what makes a MuZero learned-dynamics ``TransitionModel``
    a drop-in replacement later (§7).
    """

    def root_decision(self, state: GameState) -> Decision | None:
        """Return the first learner :class:`Decision` from ``state`` (or None)."""
        ...

    def step(
        self,
        state: GameState,
        action_path: tuple[object, ...],
        next_action: object,
        reseed: int,
    ) -> StepResult:
        """Advance one decision: replay ``action_path`` then force ``next_action``.

        See :class:`EngineTransitionModel.step` for the contract (the core impl).
        """
        ...


class EngineTransitionModel:
    """Core :class:`TransitionModel`: the real engine (``clone`` + driver).

    A running ``run_round_driver`` generator cannot be cloned or pickled, so a
    transition is realized by **replay**: each node remembers the actions forced
    so far *within the current round* (``action_path``), and a step clones the
    round-start state, drives a fresh generator forcing exactly that path plus the
    new action, and stops at the next yielded learner decision. This keeps the
    transition a pure function of ``(round-start state, action_path, next_action,
    reseed)`` -- byte-identical to how :class:`LookaheadAgent` forces a candidate
    through ``gen.send(...)``.

    Solo only: in pure solo *every* yielded :class:`Decision` is the learner's,
    so there is no non-searched decision to delegate (the README's "delegate any
    non-searched decision -- none in pure solo" clause). The opponent-node fork
    (deferred ``Node`` impl) reintroduces that path behind the seam.
    """

    def __init__(
        self,
        player_id: int,
        *,
        two_player: bool = False,
        opponent_id: int = 1,
    ) -> None:
        self.player_id = player_id
        #: Sprint C6: in two-player perfect-info mode the opponent's decisions are
        #: ALSO searched (not auto-resolved), so the replay stops at -- and the
        #: ``action_path`` records -- decisions for either searched seat. With
        #: ``two_player=False`` (solo) only ``player_id`` is searched (the C0-C5
        #: path, byte-for-byte).
        self.two_player = two_player
        self.opponent_id = opponent_id
        if two_player:
            self._searched: tuple[int, ...] = (player_id, opponent_id)
        else:
            self._searched = (player_id,)

    def _is_searched(self, decision: Decision) -> bool:
        """True iff ``decision``'s owner is a seat the tree branches on."""
        return decision.player_id in self._searched

    # -- helpers --------------------------------------------------------

    def root_decision(self, state: GameState) -> Decision | None:
        """First learner decision reachable from ``state`` with no forced action.

        Drives a fresh round generator on a throwaway clone and returns its first
        yielded learner :class:`Decision`. The *root* of the tree must be the
        decision the agent was actually asked for; the caller supplies the live
        legal set, so this is used only when (re)seating a search at a clean
        round start. Returns ``None`` if the game is already over.
        """
        if state.is_game_over:
            return None
        # The root state is already paused at a real decision the agent was asked
        # for; the agent passes that Decision in directly (see MCTSAgent). This
        # helper exists for the seam contract / tests that drive from a round
        # start. We re-derive by advancing zero actions.
        res = self._advance(state, action_path=(), reseed=None)
        return res.decision

    def step(
        self,
        state: GameState,
        action_path: tuple[object, ...],
        next_action: object,
        reseed: int,
    ) -> StepResult:
        """Replay ``action_path`` from the round-start ``state`` then force
        ``next_action``, stopping at the next learner decision (or terminal).

        ``reseed`` makes the clone's RNG explicit (the S1 determinism contract):
        the chance outcome a draw produces this round depends on ``reseed`` only.
        """
        return self._advance(
            state, action_path=action_path + (next_action,), reseed=reseed
        )

    # -- the single replay primitive ------------------------------------

    def _advance(
        self,
        round_start: GameState,
        action_path: tuple[object, ...],
        reseed: int | None,
    ) -> StepResult:
        """Clone ``round_start``, replay ``action_path``, return the next decision.

        Drives ``run_round_driver`` rounds, forcing the recorded actions for the
        learner's decisions in order. When the recorded path is exhausted, the
        next yielded learner decision is the result. Crossing into a new round
        (``round_num`` increased relative to the clone's start) marks the edge as
        a **chance edge** (§4.3): the replenish draw fired.

        Determinism: ``reseed`` is passed straight to ``GameState.clone(reseed=)``
        so the whole replay is a pure function of ``(round_start, action_path,
        reseed)``. ``reseed=None`` (used only for the seam's ``root_decision``)
        forks deterministically off the parent rng, which is fine for a one-shot
        peek that consumes no chance edge.
        """
        clone = round_start.clone(reseed=reseed)
        # Throwaway clone: turn logging on so the leaf's spin accounting can read
        # the spin_out events regardless of the live game's logging setting (the
        # S1 _count_spins / _pre_spin_progress contract). Never touches the real log.
        clone.logging_enabled = True

        start_round = clone.round_num
        path = list(action_path)
        path_i = 0

        gen = run_round_driver(clone)
        try:
            decision = next(gen)
            while True:
                if self._is_searched(decision) and path_i < len(path):
                    # Force the next recorded searched action (re-validated; a
                    # stale forced action falls through to a legal default). In
                    # two-player mode the path interleaves BOTH searched seats'
                    # actions in driver order, so the index advances on every
                    # searched decision regardless of which seat it belongs to.
                    action = self._coerce_action(decision, path[path_i])
                    path_i += 1
                    decision = gen.send(action)
                    continue
                if path_i >= len(path) and self._is_searched(decision):
                    # Path exhausted at a searched decision: branch here.
                    is_chance = clone.round_num > start_round
                    return StepResult(
                        state=clone,
                        decision=decision,
                        is_chance=is_chance,
                        terminal=False,
                    )
                # A decision for a NON-searched seat (an unsearched opponent in
                # solo / 4p backdrop): auto-resolve with a legal default so the
                # replay advances. The two searched seats in C6 are never reached
                # here (they are searched), so this is the solo path unchanged.
                decision = gen.send(self._legal_default(decision))
        except StopIteration:
            # Round ended before another searched decision (or the game finished).
            # Continue into the next round(s) until we either reach the next
            # searched decision (path already exhausted => branch there) or the
            # game is over. The round-boundary crossing is the chance edge.
            return self._continue_after_round(clone, start_round)

    def _continue_after_round(
        self, clone: GameState, start_round: int
    ) -> StepResult:
        """Keep driving fresh rounds until the next learner decision or game-over.

        Reached when a round's generator exhausts (the learner's last decision of
        the round was already forced). The next learner decision lives in the
        following round -- which we have crossed (a draw fired), so the edge is a
        chance edge.
        """
        while not clone.is_game_over:
            gen = run_round_driver(clone)
            try:
                decision = next(gen)
                while not self._is_searched(decision):
                    decision = gen.send(self._legal_default(decision))
                # First searched decision of the new round -> chance edge.
                return StepResult(
                    state=clone,
                    decision=decision,
                    is_chance=clone.round_num > start_round,
                    terminal=False,
                )
            except StopIteration:
                # An entire round with no learner decision (e.g. the learner
                # already finished). Loop into the next round.
                continue
        # Game over: terminal leaf (still a chance edge if a round was crossed).
        return StepResult(
            state=clone,
            decision=None,
            is_chance=clone.round_num > start_round,
            terminal=True,
        )

    def _coerce_action(self, decision: Decision, action: object) -> object:
        """Return ``action`` if legal for ``decision``, else a legal default.

        A forced action recorded earlier in the tree can be stale if the engine
        adjusted the player (e.g. a gear clamp); mirroring
        ``LookaheadAgent._answer_rollout_decision`` we fall back to a legal
        default rather than crash the replay.
        """
        if self._is_legal(decision, action):
            return action
        return self._legal_default(decision)

    @staticmethod
    def _is_legal(decision: Decision, action: object) -> bool:
        kind = decision.kind
        if kind in (DecisionKind.GEAR, DecisionKind.CARDS, DecisionKind.DISCARD):
            return action in decision.legal  # type: ignore[operator]
        # REACT / SLIPSTREAM legality is structural (any ReactDecision / bool is
        # accepted by the engine if it was offered); treat as legal.
        return True

    @staticmethod
    def _legal_default(decision: Decision) -> object:
        """A safe legal action for ``decision`` (used for stale-path fallback)."""
        kind = decision.kind
        if kind == DecisionKind.GEAR:
            return decision.legal[0]  # type: ignore[index]
        if kind == DecisionKind.CARDS:
            return decision.legal[0]  # type: ignore[index]
        if kind == DecisionKind.REACT:
            # No cooldown / no boost / no adrenaline: always legal when REACT is asked.
            return ReactDecision(
                cooldown_count=0,
                use_boost=False,
                use_adrenaline_speed=False,
                use_adrenaline_cooldown=False,
            )
        if kind == DecisionKind.SLIPSTREAM:
            return False  # decline
        if kind == DecisionKind.DISCARD:
            return []  # discard none
        raise ValueError(f"Unknown decision kind {kind!r}")


# ---------------------------------------------------------------------------
# Seam 4: LeafEvaluator + the net adapter that backs it
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sprint C7: process-level model cache (kill the per-game reload tax)
# ---------------------------------------------------------------------------
#
# C6's profile (C6-findings §Point-1, item 4) showed the loop's real killer is a
# per-GAME model reload: ``gen_selfplay`` builds a fresh ``MCTSAgent`` per game,
# each of which reloads the SB3 ``.zip`` from disk on first use (~1.3 s measured).
# The loaded SB3 ``MaskablePPO`` policy is used READ-ONLY for inference (the
# search never mutates weights), so it can be loaded ONCE per process and shared
# across every game in that process. The cache is keyed by ``(abspath, mtime)``
# so a re-trained checkpoint written to the same path is reloaded automatically
# (the mtime changes), and so two adapters pointing at the same checkpoint share
# one module. Per-game MUTABLE state (the agent's ``_search_rng``/seed, ``_ply``,
# ``SearchProfile``) is reset per game by the agent -- never the weights -- so
# determinism is unchanged. On Windows (spawn) each worker re-imports this module
# and so gets its own cache; the first game in each worker pays the load once, and
# every later game in that worker reuses it (the benchmark reports the amortized,
# not cold, throughput).
#:
#: Module-level so it is shared across all ``NetAdapter`` instances in a process.
_MODEL_CACHE: dict[tuple[str, float], object] = {}


def _load_cached_model(model_path: str):
    """Load (or reuse) the SB3 ``MaskablePPO`` for ``model_path``, cached by
    ``(abspath, mtime)`` at process scope.

    The cache key includes the file mtime so a re-trained checkpoint at the same
    path is reloaded (its mtime advances); a missing file falls back to ``mtime
    = 0.0`` so a path that cannot be ``stat``-ed still keys deterministically (the
    subsequent ``MaskablePPO.load`` raises the real error). The returned module is
    used read-only for inference and shared across the process's games.
    """
    from sb3_contrib import MaskablePPO

    abspath = os.path.abspath(model_path)
    try:
        mtime = os.path.getmtime(abspath)
    except OSError:
        mtime = 0.0
    key = (abspath, mtime)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = MaskablePPO.load(abspath, device="cpu")
        _MODEL_CACHE[key] = model
    return model


class NetAdapter:
    """Thin policy+value adapter over a loaded SB3 ``MaskablePPO`` checkpoint.

    Backs both the PUCT prior (``policy_prior``) and the leaf value
    (``leaf_value``). Isolating the net behind this adapter is what lets C2/C3
    swap the checkpoint without touching the search core, and lets the unit tests
    inject a stub net (a freshly-built random-weight ``build_model`` works too --
    no trained checkpoint is required for correctness).

    Loading mirrors :class:`heat.agents.ml_agent.MLAgent`: lazy, CPU, with the
    §3.4 ``.meta.json`` contract tripwire (``CheckpointMismatchError`` on a stale
    ``obs_dim`` / ``action_dim`` / ``codec_version``). The heavy model is nulled
    in ``__getstate__`` so the adapter (and the owning agent) pickle by path.

    Sprint C7: the loaded SB3 module is fetched from a process-level cache
    (:func:`_load_cached_model`, keyed by ``(abspath, mtime)``) so the policy
    loads ONCE per process and is shared READ-ONLY across every game -- killing
    the per-game reload tax the C6 profile identified as the loop's real killer.
    Inference does not mutate the weights, so the share is safe; the ``.meta.json``
    tripwire still validates the contract on first use.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        #: Sprint C6: the value transform read from the checkpoint sidecar's
        #: ``value_mode``. ``"rounds"`` (default / absent) returns the raw critic
        #: scalar (the C2-C5 −rounds_remaining quantity, byte-for-byte); ``"winloss"``
        #: squashes it through ``tanh`` so the leaf value is the bounded [-1,1]
        #: expected outcome the trainer fit -- resolved lazily on first model load.
        self._value_mode: str | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def _validate_meta(self) -> None:
        """Assert the checkpoint sidecar matches the live contract (§3.4)."""
        from heat.agents.ml_agent import CheckpointMismatchError
        from heat.ml import spaces
        from heat.ml.training import load_meta

        try:
            meta = load_meta(self.model_path)
        except FileNotFoundError as exc:
            raise CheckpointMismatchError(
                f"Checkpoint sidecar not found for {self.model_path!r}; "
                "expected a '.meta.json' written by save_checkpoint."
            ) from exc

        expected = {
            "obs_dim": spaces.OBS_DIM,
            "action_dim": spaces.ACTION_DIM,
            "codec_version": spaces.CODEC_VERSION,
        }
        mismatches = [
            f"{key}: checkpoint={meta.get(key)!r} != runtime={want!r}"
            for key, want in expected.items()
            if meta.get(key) != want
        ]
        if mismatches:
            raise CheckpointMismatchError(
                f"Checkpoint {self.model_path!r} is incompatible with the current "
                "ML contract (stale model vs drifted codec): " + "; ".join(mismatches)
            )
        # Sprint C6: read the (additive) value_mode so leaf_value applies the
        # matching transform. Absent -> "rounds" (the C2-C5 raw-critic path).
        self._value_mode = str(meta.get("value_mode", "rounds"))

    def _get_model(self):
        if self._model is None:
            # Sprint C7: validate the contract (cheap, reads the small .meta.json)
            # then fetch the heavy SB3 module from the process-level cache so the
            # policy loads ONCE per process and is shared read-only across games.
            self._validate_meta()
            self._model = _load_cached_model(self.model_path)
        return self._model

    # -- the two seam methods -------------------------------------------

    def policy_prior(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masked policy probabilities over the full ``ACTION_DIM`` flat space.

        Reads the actor head's masked categorical (``get_distribution(...,
        action_masks=mask)``) -- the same path :mod:`heat.ml.kl_regularizer` uses
        -- and returns a probability vector (illegal actions at 0). The search
        reads only the kept-candidate entries and renormalizes over them.
        """
        import torch

        model = self._get_model()
        ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            dist = model.policy.get_distribution(
                ob, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1)
            )
            probs = dist.distribution.probs
        return np.asarray(probs.detach()).reshape(-1)

    def leaf_value(self, obs: np.ndarray) -> float:
        """Net value head on a leaf state (higher-is-better).

        ``value_mode="rounds"`` (default) returns the critic scalar AS-IS (the
        A1/A2 sign convention is artifact-authoritative: the net regresses
        ``−rounds_remaining`` so ``predict_values`` already emits the
        higher-is-better quantity; a second negation would flip it). Identical to
        ``LookaheadAgent._value_leaf``.

        ``value_mode="winloss"`` (Sprint C6) squashes the raw critic through
        ``tanh`` -- the SAME transform ``train_az`` fit the ±1 outcome under -- so
        the leaf is a bounded ``[−1, 1]`` learner-frame expected outcome the
        two-player minimax consumes directly.
        """
        import torch

        model = self._get_model()
        ob = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            v = model.policy.predict_values(ob)
            if self._value_mode == "winloss":
                v = torch.tanh(v)
        return float(np.asarray(v.detach()).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Seam 2: Node
# ---------------------------------------------------------------------------


class NodeKind:
    """Node kind tag (decision vs chance). A tiny enum-like for ``Node.kind``."""

    DECISION = "decision"
    CHANCE = "chance"


@dataclass
class Edge:
    """A decision-node child edge: a candidate action + its PUCT statistics.

    Attributes:
        action: The concrete engine action this edge forces (gear tuple, card
            tuple, ReactDecision, bool, or discard list).
        prior: ``P(s, a)`` -- the net's masked policy probability for this action,
            renormalized over the kept candidate set.
        child: The child :class:`Node` (lazily created on first visit), or ``None``.
        n: Visit count ``N(s, a)``.
        w: Summed backed-up value ``W(s, a)`` (mean is ``w / n``).
    """

    action: object
    prior: float
    child: "Node | None" = None
    n: int = 0
    w: float = 0.0

    def q(self) -> float:
        """Mean action value ``Q(s, a) = W / N`` (raw, pre-normalization)."""
        return self.w / self.n if self.n > 0 else 0.0


@dataclass
class ChanceOutcome:
    """A sampled chance-node outcome (one realized draw) + its statistics."""

    reseed: int
    child: "Node | None" = None
    n: int = 0
    w: float = 0.0


@dataclass
class GumbelRootResult:
    """The Gumbel-AlphaZero root selection result for one search (Sprint C4).

    Captured on the agent after a ``root_selector="gumbel"`` search so
    ``gen_selfplay`` can build the **completed-Q** policy target
    ``π = softmax(logits + σ(completedQ))`` (decision: unvisited/un-sampled actions
    take the root's own normalized value ``root.value`` as their completed-Q). All
    fields are aligned to ``root.edges`` order.

    Attributes:
        edges: the root edges (same list object as ``root.edges``).
        gumbel: ``g_a ~ Gumbel(0)`` per edge (drawn only from ``_search_rng``).
        logits: ``log P(s,a)`` per edge (the net's log-prior over the kept set).
        sampled: index list of the ``m`` Gumbel-top-``m`` sampled actions.
        acted_index: the selected (acted) edge index ``argmax(g + logits + σ(Q̂))``
            over the Sequential-Halving survivors.
        root_value_norm: the root's own normalized value ``v̂`` (the completion
            baseline for unvisited / un-sampled actions).
    """

    edges: list[Edge]
    gumbel: list[float]
    logits: list[float]
    sampled: list[int]
    acted_index: int
    root_value_norm: float


@dataclass
class Node:
    """A search-tree node (README §3 ``Node`` seam).

    A node captures the engine position by **replay coordinates**: the
    round-start ``state`` clone plus the ``action_path`` of learner actions forced
    so far this round. Cloning a node's exact position is therefore deterministic
    (re-clone ``state`` and replay ``action_path``), which is what keeps the whole
    search a pure function of ``(root, seed)`` without cloning a live generator.

    Solo: ``to_move`` is always the learner. ``kind`` is ``DECISION`` (branch on a
    learner :class:`Decision`) or ``CHANCE`` (a replenish-draw edge, expanded by
    DPW). ``terminal`` nodes have a value but no children.
    """

    kind: str
    state: GameState
    action_path: tuple[object, ...]
    to_move: int
    #: The round-START state this node's ``action_path`` replays from. For the
    #: root and any node reached across a round boundary this equals ``state``
    #: (the path is empty / resets); for an intra-round decision node it is the
    #: shared round-start clone the whole round replays from, while ``state`` is
    #: the *advanced* clone used for obs/prior/leaf. Keeping them separate is what
    #: makes ``model.step`` a pure replay from a clean round start (the advanced
    #: ``state`` has already been mutated by the partial round and must NOT be
    #: re-driven). Defaults to ``state`` when not given.
    round_start: GameState | None = None
    # Decision-node fields:
    decision: Decision | None = None
    edges: list[Edge] = field(default_factory=list)
    # Chance-node fields:
    outcomes: list[ChanceOutcome] = field(default_factory=list)
    # Shared:
    terminal: bool = False
    expanded: bool = False
    #: True iff the forced move that *produced this node* spun the learner out on
    #: its own first round -- the depth-invariant own-spin flag (§4.4).
    own_spun: bool = False
    #: Cached leaf value (set once at creation / first evaluation).
    value: float = 0.0
    #: Total visits to this node (``ΣN`` over edges/outcomes for selection).
    n: int = 0

    def __post_init__(self) -> None:
        # Default the replay anchor to the node's own state (root / cross-round).
        if self.round_start is None:
            self.round_start = self.state


# ---------------------------------------------------------------------------
# Seam 3: SearchPolicy (log-PUCT + optional Dirichlet) and Seam 5: ChanceExpansion
# ---------------------------------------------------------------------------
# Both are implemented as methods on MCTSAgent so they share the config and the
# running [Qmin, Qmax]; the *core* still routes every transition through the
# TransitionModel protocol, so the seams remain swappable (a Gumbel selector or a
# learned-chance expansion is a method override / injected strategy object).


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


_TurnSig = tuple[int, int, int, int, int]


class MCTSAgent(BaseAgent):
    """Solo net-guided stochastic MCTS agent (see module docstring).

    Args:
        net: The policy+value backing. Either a :class:`NetAdapter` (or any object
            exposing ``policy_prior(obs, mask) -> np.ndarray`` and
            ``leaf_value(obs) -> float``), OR ``None`` together with
            ``model_path`` (the agent builds a :class:`NetAdapter` lazily). Tests
            inject a stub net here directly; the eval harness passes ``model_path``.
        model_path: Path to an SB3 ``MaskablePPO`` checkpoint (with a
            ``.meta.json`` sidecar). Used to build a :class:`NetAdapter` when
            ``net`` is None. Pickled by value (a string) -- the heavy model loads
            lazily in-process / per worker.
        config: :class:`MCTSConfig` (defaults = C0 constants).
        seed: Deterministic search seed mixed into every clone reseed (never
            touches global RNG). Same state + seed => byte-stable move.
        name: Display name.
    """

    def __init__(
        self,
        *,
        net: object | None = None,
        model_path: str | None = None,
        config: MCTSConfig | None = None,
        seed: int | None = None,
        name: str = "MCTS",
    ) -> None:
        super().__init__(name=name)
        if net is None and model_path is None:
            raise ValueError("MCTSAgent requires either net= or model_path=")
        self._net = net
        self.model_path = model_path
        self.config = config if config is not None else MCTSConfig()
        self.seed = seed

        #: Per-agent profiling (clones/move, ms/move). Always on; cheap.
        self.profile = SearchProfile()

        #: Episode ply counter (drives the temperature schedule when on).
        self._ply = 0

        #: Per-search transient state (set in ``_plan_turn``): the root node (so
        #: root-only Dirichlet noise targets it) and a deterministic per-turn RNG
        #: for the self-play noise/temperature sampling. Initialized here so the
        #: self-play hooks have a defined target even before the first search.
        self._root_node: Node | None = None
        import random as _random

        self._search_rng = _random.Random(0)

        #: The Gumbel root selection result from the most recent search (set only
        #: when ``config.root_selector == "gumbel"``; ``gen_selfplay`` reads it to
        #: build the completed-Q target). ``None`` on the PUCT path.
        self._gumbel_result: GumbelRootResult | None = None

        # Cached plan for the current turn (mirrors LookaheadAgent): a full search
        # at the GEAR decision yields the gear AND the cards play to follow.
        self._plan_sig: _TurnSig | None = None
        self._plan_gear: tuple[int, int] | None = None
        self._plan_cards: tuple[Card, ...] | None = None

    # -- pickle by path: never ship the heavy model ----------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # If the net is a path-backed NetAdapter, its own __getstate__ nulls the
        # model. If it is an in-process stub (tests), it travels as-is (tests do
        # not pickle). The model_path string lets a worker rebuild the adapter.
        return state

    # -- net access ------------------------------------------------------

    def _get_net(self) -> object:
        """Return the net adapter, building one from ``model_path`` on first use."""
        if self._net is None:
            self._net = NetAdapter(self.model_path)  # type: ignore[arg-type]
        return self._net

    # -- determinism helpers (the S1 scheme) -----------------------------

    def _turn_signature(self, state: GameState, player_id: int) -> _TurnSig:
        player = state.get_player(player_id)
        return (
            player_id,
            player.position,
            player.lap,
            len(player.hand),
            player.gear,
        )

    def _turn_seed(self, sig: _TurnSig) -> int:
        """Deterministic per-turn base seed (no global RNG; the S1 fold)."""
        base = 0 if self.seed is None else int(self.seed)
        acc = (base & 0x7FFFFFFF) * 2654435761
        for v in sig:
            acc = (acc * 1000003 + int(v)) & 0x7FFFFFFF
        return acc

    # -- candidate enumeration (the S1 dedup-by-speed prune for CARDS) ---

    def _candidate_actions(self, decision: Decision, state: GameState) -> list[object]:
        """Legal actions to branch on for ``decision`` -- the kept candidate set.

        For CARDS this applies the S1 dedup-by-resulting-speed prune (a *lossless*
        equivalence: plays with the same total speed resolve identically), keeping
        the 494-wide branch tractable. Every other kind branches on its full legal
        set (gear shifts, the 8-slot REACT table, take/decline slipstream, the
        discard-k options) -- all of them are searched (§4: nothing delegated to a
        heuristic).
        """
        kind = decision.kind
        if kind == DecisionKind.CARDS:
            seen: set[int] = set()
            kept: list[object] = []
            for play in decision.legal:  # type: ignore[union-attr]
                speed = sum(c.value for c in play)
                if speed in seen:
                    continue
                seen.add(speed)
                kept.append(play)
            return kept
        if kind == DecisionKind.SLIPSTREAM:
            return [True, False]
        # GEAR / REACT / DISCARD: branch on the whole legal set as the engine
        # hands it in. REACT legal is a ReactOptions; enumerate via the codec.
        if kind == DecisionKind.REACT:
            return self._react_candidates(decision, state)
        if kind == DecisionKind.DISCARD:
            return self._discard_candidates(decision)
        return list(decision.legal)  # type: ignore[arg-type]

    @staticmethod
    def _react_candidates(decision: Decision, state: GameState) -> list[object]:
        """Enumerate the legal REACT slots as concrete ``ReactDecision`` objects.

        Uses the codec's legal mask over the 8-slot REACT table so the branch set
        is exactly the actions the net assigns prior mass to (and exactly the
        engine-legal ones).
        """
        from heat.ml.action_codec import legal_action_mask, decode_action
        from heat.ml import spaces

        mask = legal_action_mask(decision, state)
        out: list[object] = []
        for i in range(spaces.REACT_SIZE):
            flat = spaces.REACT_OFFSET + i
            if mask[flat]:
                out.append(decode_action(decision, flat, state))
        return out

    @staticmethod
    def _discard_candidates(decision: Decision) -> list[object]:
        """Enumerate the legal DISCARD options (none, or k-lowest for k>=1)."""
        from heat.ml import spaces
        from heat.ml.action_codec import _discard_order

        discardable = list(decision.legal)  # type: ignore[arg-type]
        ordered = _discard_order(discardable)
        out: list[object] = [[]]  # discard none
        for k in range(1, min(len(ordered) + 1, spaces.DISCARD_SIZE)):
            out.append(ordered[:k])
        return out

    # -- prior over the kept candidate set -------------------------------

    def _edges_for(self, node: Node) -> list[Edge]:
        """Build the PUCT edges for a freshly-expanded decision node.

        Reads the net's masked policy once, maps each kept candidate action to its
        flat codec index, gathers ``P`` over the kept set, and renormalizes (the
        prior is *read*, the candidate set is *fixed* -- the S1 lossless prune).
        Falls back to a uniform prior if the kept mass is degenerate (all-zero),
        which is what an untrained / pathological net can produce.
        """
        from heat.ml.action_codec import encode_action_index, legal_action_mask
        from heat.ml.features import encode_observation

        decision = node.decision
        assert decision is not None
        actions = self._candidate_actions(decision, node.state)

        # Sprint C6: the prior + obs are read from the SEAT THAT MOVES at this node
        # (``node.to_move``), not always the learner. In solo this is always the
        # learner (``to_move_pid``); in the two-player minimax an opponent decision
        # node reads the SAME net's prior from the opponent's own observation
        # (perfect information -- the opponent's hand is visible).
        mover = node.to_move
        obs = encode_observation(node.state, mover, decision)
        mask = legal_action_mask(decision, node.state)
        probs = self._get_net().policy_prior(obs, mask)  # type: ignore[attr-defined]

        priors: list[float] = []
        for a in actions:
            try:
                flat = encode_action_index(decision, a)
                priors.append(float(probs[flat]))
            except (ValueError, IndexError):
                priors.append(0.0)

        total = sum(priors)
        n = len(actions)
        if total <= 0.0 or not math.isfinite(total):
            priors = [1.0 / n] * n  # degenerate net -> uniform over candidates
        else:
            priors = [p / total for p in priors]

        # Optional root Dirichlet noise (self-play only; OFF by default, §4.5).
        # Mixed only into the ROOT node's prior: P = (1-ε)·p + ε·η, η~Dir(α).
        # Drawn from the deterministic per-turn RNG so even noisy search stays a
        # pure function of (state, seed). The mechanism is built here and
        # exercised in C2/C3; with ε=0 (the C1 eval default) it is a no-op.
        if node is self._root_node and self.config.dirichlet_eps > 0.0 and n > 0:
            eps = self.config.dirichlet_eps
            noise = self._search_rng.gammavariate  # Dirichlet via normalized Gammas
            gammas = [noise(self.config.dirichlet_alpha, 1.0) for _ in range(n)]
            gsum = sum(gammas) or 1.0
            eta = [g / gsum for g in gammas]
            priors = [(1.0 - eps) * p + eps * e for p, e in zip(priors, eta)]

        return [Edge(action=a, prior=p) for a, p in zip(actions, priors)]

    # -- the MCTS core ---------------------------------------------------

    def _plan_turn(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[Card, ...]]:
        """Run the full search at the GEAR decision and return (gear, cards).

        The search roots at the live GEAR decision (the agent was asked for a
        gear). Because a solo round's GEAR -> CARDS edge is deterministic
        (no draw between them), the most-visited GEAR child's most-visited CARDS
        grandchild is the committed plan -- so one search produces both the gear
        and the card play that follows, mirroring ``LookaheadAgent``'s joint
        plan. Per-move clones + wall time are recorded into :attr:`profile`.
        """
        self.to_move_pid = player_id
        sig = self._turn_signature(state, player_id)
        turn_seed = self._turn_seed(sig)

        # Root decision: the live GEAR decision the agent was handed.
        root_decision = Decision(DecisionKind.GEAR, player_id, legal_gears)
        root = Node(
            kind=NodeKind.DECISION,
            state=state,
            action_path=(),
            to_move=player_id,
            decision=root_decision,
        )
        # Root reference (so _edges_for mixes Dirichlet noise into the ROOT only)
        # and a deterministic per-turn RNG for the noise + temperature sampling --
        # seeded from turn_seed so even noisy/sampled search is a pure function of
        # (state, seed). Both hooks are OFF in C1's eval (eps=0, temperature_moves=0).
        import random as _random

        self._root_node = root
        self._search_rng = _random.Random(turn_seed)

        model = EngineTransitionModel(player_id)

        # Reset the running Q-normalization bounds per move (MuZero min-max).
        self._q_min = math.inf
        self._q_max = -math.inf
        self._clones_this_move = 0

        t0 = time.perf_counter()
        self._run_root_search(root, model, turn_seed)
        self.profile.record_move(self._clones_this_move, time.perf_counter() - t0)

        gear, cards = self._extract_plan(root, legal_gears, state, player_id)
        self._ply += 1
        return gear, cards

    def _run_root_search(
        self, root: Node, model: EngineTransitionModel, turn_seed: int
    ) -> None:
        """Run ``n_simulations`` from ``root`` under the configured root selector.

        ``root_selector="puct"`` (default) is the byte-for-byte-unchanged C1 path:
        every simulation selects the root child by log-PUCT. ``"gumbel"`` (Sprint
        C4) allocates the budget across a Gumbel-sampled top-``m`` via Sequential
        Halving (the interior descent below the root is identical in both). The
        Gumbel path records its result on ``self._gumbel_result`` for the
        completed-Q target; the PUCT path leaves it ``None``.
        """
        if self.config.root_selector == "gumbel":
            self._gumbel_result = self._gumbel_root_search(root, model, turn_seed)
        else:
            self._gumbel_result = None
            for sim in range(self.config.n_simulations):
                self._simulate(root, model, turn_seed, sim)

    # -- the Gumbel-AlphaZero root selector (Sprint C4) ------------------

    def _gumbel_sigma(self, q_hat: float, max_n: int) -> float:
        """Danihelka's monotone σ transform over an ALREADY-NORMALIZED Q̂.

        ``σ(q) = (c_visit + max_b N_b) · c_scale · q`` where ``q`` is ``Q̂`` in
        ``[0,1]`` (``_normalize_q`` output) -- so the unbounded −rounds_remaining
        scale is already tamed before σ ever runs (the C0 §D hazard, now at the
        root). ``max_n`` is the largest visit count over the root's sampled actions.
        """
        cfg = self.config
        return (cfg.gumbel_c_visit + float(max_n)) * cfg.gumbel_c_scale * q_hat

    def _gumbel_root_search(
        self, root: Node, model: EngineTransitionModel, turn_seed: int
    ) -> GumbelRootResult:
        """Gumbel top-``m`` sampling + Sequential Halving at the ROOT (Sprint C4).

        The interior descent (everything below the root), the chance-node DPW, the
        leaf discipline, and the codec are all UNCHANGED -- only the root edge each
        simulation is forced through is dictated here (Sequential Halving), not by
        log-PUCT. All randomness draws ONLY from ``self._search_rng`` (seeded off
        the turn signature in the caller), so the whole search stays a pure
        function of ``(state, seed)`` -- no global-RNG leak.

        Algorithm (Danihelka et al. 2022):
          1. Expand the root (build edges from the net prior, set ``root.value``).
          2. Draw ``g_a ~ Gumbel(0)`` per edge; take the ``m`` edges with the
             largest ``g_a + logits_a`` (the Gumbel-top-``m`` trick = sampling ``m``
             actions WITHOUT replacement from ``softmax(logits)``).
          3. Sequential Halving: split ``n_simulations`` into ``⌈log2(m)⌉`` phases;
             each phase runs an equal share of sims through every surviving root
             action, then keeps the top half by ``g_a + logits_a + σ(Q̂_a)``.
          4. The acted action is ``argmax(g_a + logits_a + σ(Q̂_a))`` over the
             survivors.
        """
        # (1) Expand the root once so the priors + root.value are available before
        #     any sampling (the C1 loop expands the root lazily on sim 0; the
        #     Gumbel path needs it up front to read logits / the completion value).
        if not root.expanded:
            self._expand_and_evaluate(root, model, turn_seed, sim=0)

        edges = root.edges
        n_edges = len(edges)
        logits = [math.log(max(e.prior, 1e-12)) for e in edges]

        # (2) Gumbel-top-m sample (without replacement) from softmax(logits).
        gumbel = [self._sample_gumbel() for _ in range(n_edges)]
        m = min(self.config.gumbel_m, n_edges)
        # Sequential Halving needs at least m sims to give each survivor >= 1; cap
        # m at the budget so a tiny n_simulations degenerates gracefully (m=1 =>
        # prior-greedy root, the C1 n_simulations=1 analogue).
        m = max(1, min(m, self.config.n_simulations))
        order = sorted(range(n_edges), key=lambda i: (gumbel[i] + logits[i], i), reverse=True)
        sampled = order[:m]

        budget = self.config.n_simulations
        sim_counter = 0
        survivors = list(sampled)

        if m == 1:
            # Degenerate: a single sampled action; spend the whole budget on it so
            # its Q̂ is estimated, then it is trivially the acted action.
            for _ in range(budget):
                self._simulate(root, model, turn_seed, sim_counter,
                               forced_root_edge=edges[survivors[0]])
                sim_counter += 1
        else:
            # (3) Sequential Halving over ⌈log2(m)⌉ phases.
            n_phases = max(1, math.ceil(math.log2(m)))
            for phase in range(n_phases):
                k = len(survivors)
                if k <= 1:
                    break
                # Sims allotted to THIS phase, split equally across the k survivors.
                # The standard SH split: budget / (⌈log2(m)⌉ · k) sims each, with
                # any remainder absorbed in the final phase by the loop spending
                # whatever budget is left.
                remaining_phases = n_phases - phase
                phase_budget = (budget - sim_counter)
                if remaining_phases > 1:
                    phase_budget = phase_budget // remaining_phases
                per_arm = max(1, phase_budget // k)
                for _ in range(per_arm):
                    for idx in survivors:
                        if sim_counter >= budget:
                            break
                        self._simulate(root, model, turn_seed, sim_counter,
                                       forced_root_edge=edges[idx])
                        sim_counter += 1
                    if sim_counter >= budget:
                        break
                # Keep the top half by g + logits + σ(Q̂).
                keep = max(1, k // 2)
                survivors = self._gumbel_rank(edges, gumbel, logits, survivors)[:keep]

            # Spend any leftover budget on the surviving arms (round-robin) so the
            # total interior sims == n_simulations exactly (no budget leak).
            while sim_counter < budget:
                for idx in survivors:
                    if sim_counter >= budget:
                        break
                    self._simulate(root, model, turn_seed, sim_counter,
                                   forced_root_edge=edges[idx])
                    sim_counter += 1

        # (4) Acted action = argmax(g + logits + σ(Q̂)) over the survivors.
        ranked = self._gumbel_rank(edges, gumbel, logits, survivors)
        acted_index = ranked[0]

        return GumbelRootResult(
            edges=edges,
            gumbel=gumbel,
            logits=logits,
            sampled=sampled,
            acted_index=acted_index,
            root_value_norm=self._normalize_q(root.value),
        )

    def _gumbel_rank(
        self,
        edges: list[Edge],
        gumbel: list[float],
        logits: list[float],
        indices: list[int],
    ) -> list[int]:
        """Rank ``indices`` by ``g_a + logits_a + σ(Q̂_a)`` (descending, stable).

        ``Q̂_a`` is the root edge's normalized mean value (``_normalize_q(edge.q())``
        -- the SAME min-max the interior PUCT uses) for a visited edge; an unvisited
        edge is scored with the completion baseline ``v̂`` (the root's own
        normalized value), matching the completed-Q construction. ``max_n`` for σ is
        the largest visit count over the ranked set (Danihelka's definition).
        """
        max_n = max((edges[i].n for i in indices), default=0)
        root_v = self._normalize_q(self._root_node.value) if self._root_node else 0.5

        def score(i: int) -> tuple[float, int]:
            edge = edges[i]
            q_hat = self._normalize_q(edge.q()) if edge.n > 0 else root_v
            return (gumbel[i] + logits[i] + self._gumbel_sigma(q_hat, max_n), -i)

        return sorted(indices, key=score, reverse=True)

    def _sample_gumbel(self) -> float:
        """A single ``Gumbel(0)`` draw from ``_search_rng`` (no global RNG).

        ``-log(-log(u))`` with ``u ~ Uniform(0,1)`` (guarded off 0/1). The agent's
        deterministic per-turn RNG is the only source, so the Gumbel search stays a
        pure function of ``(state, seed)`` -- the C1 determinism contract.
        """
        u = self._search_rng.random()
        # Guard the open interval so the double log is finite.
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return -math.log(-math.log(u))

    def _simulate(
        self,
        root: Node,
        model: EngineTransitionModel,
        turn_seed: int,
        sim: int,
        forced_root_edge: "Edge | None" = None,
    ) -> None:
        """One root-to-leaf descent + backup (a single simulation).

        Descends by log-PUCT at decision nodes and DPW at chance nodes, expanding
        the first unexpanded node it reaches, evaluating it once, and backing the
        value up the visited path (chance nodes average; decision nodes sum).

        ``forced_root_edge`` (Sprint C4, Gumbel root selector only): if given, the
        root's child is dictated by Sequential Halving (this exact edge) instead of
        log-PUCT -- but ONLY at the root; every interior node still descends by the
        unchanged log-PUCT / DPW machinery. ``None`` (the C1/C2/C3 default) leaves
        the descent byte-for-byte identical.
        """
        path: list[tuple[Node, object]] = []  # (node, edge-or-outcome) visited
        node = root

        while True:
            if node.terminal:
                value = node.value
                break
            if not node.expanded:
                value = self._expand_and_evaluate(node, model, turn_seed, sim)
                break

            if node.kind == NodeKind.DECISION:
                if forced_root_edge is not None and node is root:
                    edge = forced_root_edge
                else:
                    edge = self._select_edge(node)
                path.append((node, edge))
                if edge.child is None:
                    # Expand this edge's child by forcing its action.
                    child = self._step_to_child(
                        node, edge, model, turn_seed, sim, len(path)
                    )
                    edge.child = child
                    value = self._expand_and_evaluate(child, model, turn_seed, sim)
                    break
                node = edge.child
            else:  # CHANCE
                outcome = self._select_outcome(node, turn_seed, sim, len(path))
                path.append((node, outcome))
                if outcome.child is None:
                    child = self._step_chance_child(
                        node, outcome, model
                    )
                    outcome.child = child
                    value = self._expand_and_evaluate(child, model, turn_seed, sim)
                    break
                node = outcome.child

        self._backup(path, root, value)

    # -- selection (§4.1 log-PUCT + Q-norm + FPU) ------------------------

    def _normalize_q(self, q: float) -> float:
        """Min-max normalize a raw Q into [0, 1] by the tree's running bounds.

        MANDATORY (§4.1 / C0 §D): the raw ``−rounds_remaining`` scale swamps the
        ``P`` exploration term. Before any value is seen, or when the bounds are
        degenerate (``qmax <= qmin``), returns 0.5 (a neutral mid-point) so the
        prior governs selection -- exactly the behaviour the Q-normalization
        regression test pins.
        """
        qmin, qmax = self._q_min, self._q_max
        if not math.isfinite(qmin) or not math.isfinite(qmax) or qmax <= qmin:
            return 0.5
        return (q - qmin) / (qmax - qmin)

    def _c_puct(self, parent_n: int) -> float:
        """The log-scaled PUCT coefficient ``c_init + log((ΣN+c_base+1)/c_base)``."""
        cfg = self.config
        return cfg.c_init + math.log((parent_n + cfg.c_base + 1.0) / cfg.c_base)

    def _select_edge(self, node: Node) -> Edge:
        """Pick the decision-node child maximizing the log-PUCT score (§4.1).

        Visited children use their normalized mean Q; unvisited children use FPU:
        ``Q̂ = Q̂(parent) − fpu_reduction`` (the prior, not an optimistic default,
        governs which unexplored action to try first). Deterministic argmax with a
        stable first-seen tiebreak.
        """
        cfg = self.config
        parent_n = max(1, node.n)
        c = self._c_puct(node.n)
        sqrt_total = math.sqrt(node.n)

        # FPU baseline: the parent's own normalized value, taken in the MOVER's
        # frame so it is on the same scale as the (already-mover-frame) edge Qs.
        # Sprint C6: ``node.value`` is stored in the learner's frame, so at an
        # opponent node the mover-frame baseline is its negation (negamax). Solo
        # has no opponent node, so this is ``node.value`` byte-for-byte.
        node_value = node.value
        if cfg.two_player and node.to_move == cfg.opponent_id:
            node_value = -node_value
        parent_q_norm = self._normalize_q(node_value)

        best_edge = node.edges[0]
        best_score = -math.inf
        for edge in node.edges:
            if edge.n > 0:
                q_hat = self._normalize_q(edge.q())
            else:
                q_hat = parent_q_norm - cfg.fpu_reduction
            u = c * edge.prior * sqrt_total / (1.0 + edge.n)
            score = q_hat + u
            if score > best_score:
                best_score = score
                best_edge = edge
        return best_edge

    # -- chance-node DPW selection (§4.3) --------------------------------

    def _select_outcome(
        self, node: Node, turn_seed: int, sim: int, depth: int
    ) -> ChanceOutcome:
        """Pick (or widen) a chance outcome under double progressive widening.

        DPW rule (§4.3): expand a NEW sampled outcome only when
        ``⌈C_pw·N^α_pw⌉`` exceeds the current outcome count (capped at ``K``);
        otherwise re-descend into an existing outcome chosen by visit proportion
        (deterministically: the least-visited existing outcome, which makes the
        re-descent a pure function of the visit counts -- no RNG in selection).
        The new outcome's ``reseed`` is the deterministic per-(node, sample) seed.
        """
        cfg = self.config
        n = node.n
        target = min(cfg.k_cap, math.ceil(cfg.c_pw * (max(1, n) ** cfg.alpha_pw)))
        if len(node.outcomes) < target:
            # Widen: add a new sampled outcome with a deterministic reseed.
            sample_idx = len(node.outcomes)
            reseed = self._chance_reseed(node, turn_seed, sim, depth, sample_idx)
            outcome = ChanceOutcome(reseed=reseed)
            node.outcomes.append(outcome)
            return outcome
        # Re-descend: pick the least-visited existing outcome (visit-proportional
        # in expectation, deterministic in realization). Stable first-seen tiebreak.
        return min(node.outcomes, key=lambda o: o.n)

    def _chance_reseed(
        self, node: Node, turn_seed: int, sim: int, depth: int, sample_idx: int
    ) -> int:
        """A deterministic per-(node, sample) reseed (the S1 ``_score_plan`` scheme).

        Folds the turn seed with the node's replay coordinates and the sample
        index so a fixed (root, seed) replays the same chance outcomes -- the
        determinism contract C0 §E confirmed byte-stable. Crucially it does NOT
        depend on ``sim`` (the simulation index): a given chance node's k-th
        sampled outcome must be the SAME draw every time it is widened, so the
        backed-up average is a stable visit-weighted mean over a fixed outcome set.
        """
        acc = (turn_seed & 0x7FFFFFFF) * 2654435761
        acc = (acc * 1000003 + len(node.action_path)) & 0x7FFFFFFF
        acc = (acc * 1000003 + depth) & 0x7FFFFFFF
        acc = (acc * 1000003 + sample_idx * 31 + 17) & 0x7FFFFFFF
        return acc

    # -- expansion + leaf evaluation (§4.2, §4.4) ------------------------

    def _step_to_child(
        self,
        node: Node,
        edge: Edge,
        model: EngineTransitionModel,
        turn_seed: int,
        sim: int,
        depth: int,
    ) -> Node:
        """Advance one decision (force ``edge.action``) -> the child node.

        A deterministic-decision edge (intra-round) yields the next DECISION node;
        an edge that crosses the round boundary yields a CHANCE node (§4.3). The
        reseed for this advance is deterministic; for a deterministic edge the
        reseed is irrelevant (no draw fires) but we still pass an explicit one so
        the clone never forks off the live rng.
        """
        reseed = self._chance_reseed(node, turn_seed, sim, depth, sample_idx=0)
        self._clones_this_move += 1
        # Replay from the node's clean round-START state (NOT its advanced
        # ``state``, which the partial round already mutated) so the step is a
        # pure replay of ``action_path + (action,)``.
        res = model.step(node.round_start, node.action_path, edge.action, reseed)

        own_spun = self._forced_move_spun(res.state, node, edge)
        if res.terminal:
            return self._make_terminal_node(node, edge, res, own_spun)
        if res.is_chance:
            # Crossed the round boundary: the next learner decision lives in a new
            # round behind a replenish draw. The child is a CHANCE node whose
            # outcomes are the realized draws. CRITICAL: the draw outcome is
            # determined by the reseed of the advance that crosses the boundary, so
            # each DPW outcome must RE-RUN that same advance (this parent's
            # round-start state + action_path + this edge's action) with its OWN
            # reseed -- which is why the chance node remembers the PRE-advance
            # replay coordinates, not the post-draw state. (Re-cloning the
            # post-draw state would only re-run the *next* round, never re-drawing.)
            chance = Node(
                kind=NodeKind.CHANCE,
                # Replay anchor for re-running the crossing advance: the parent's
                # clean round-start state + the full path including this edge.
                state=node.round_start,
                round_start=node.round_start,
                action_path=node.action_path + (edge.action,),  # the crossing advance
                to_move=self.to_move_pid,
                decision=res.decision,  # the decision an outcome leads to (one realization)
                own_spun=own_spun,
            )
            # A chance node needs no prior; its "expansion" is trivial, so mark it
            # expanded immediately and let the descent loop widen its outcomes.
            chance.expanded = True
            return chance
        # Deterministic intra-round edge -> next DECISION node, SAME round: the
        # advanced ``state`` is used for obs/prior/leaf, but the replay anchor stays
        # the shared round-start clone so model.step replays from a clean start.
        # Sprint C6: the child's ``to_move`` is the SEAT THAT OWNS the next decision
        # (the learner in solo; either seat in the two-player minimax) -- this is
        # what makes ``_backup`` negate at opponent plies and ``_edges_for`` read
        # the opponent's prior.
        return Node(
            kind=NodeKind.DECISION,
            state=res.state,
            round_start=node.round_start,
            action_path=node.action_path + (edge.action,),
            to_move=res.decision.player_id if res.decision is not None else self.to_move_pid,
            decision=res.decision,
            own_spun=node.own_spun or own_spun,
        )

    def _step_chance_child(
        self, node: Node, outcome: ChanceOutcome, model: EngineTransitionModel
    ) -> Node:
        """Realize one chance outcome: RE-RUN the round-crossing advance with the
        outcome's deterministic ``reseed`` so this outcome sees its own draw.

        The chance node remembers the PRE-advance replay coordinates (its parent's
        round-start ``state`` + the full ``action_path`` including the action that
        crossed the boundary). Replaying that exact advance with a fresh
        ``reseed`` re-fires the replenish draw with a different realization -- the
        true expectimax outcome sampling (NOT PIMC: each outcome is an independent
        draw, not a fixed determinization). The child is the DECISION node at the
        first learner decision of the new round under this draw.
        """
        self._clones_this_move += 1
        # Re-run the crossing advance: replay everything up to (but not including)
        # the last recorded action, then force that last action with the outcome's
        # reseed -- producing this outcome's specific draw.
        head = node.action_path[:-1]
        last = node.action_path[-1]
        res = model.step(node.state, head, last, outcome.reseed)
        if res.terminal:
            child = Node(
                kind=NodeKind.DECISION,
                state=res.state,
                action_path=(),
                to_move=self.to_move_pid,
                decision=None,
                terminal=True,
                own_spun=node.own_spun,
            )
            child.value = self._terminal_value(res.state)
            child.expanded = True
            return child
        return Node(
            kind=NodeKind.DECISION,
            state=res.state,
            action_path=(),
            to_move=res.decision.player_id if res.decision is not None else self.to_move_pid,
            decision=res.decision,
            own_spun=node.own_spun,
        )

    def _make_terminal_node(
        self, node: Node, edge: Edge, res: StepResult, own_spun: bool
    ) -> Node:
        child = Node(
            kind=NodeKind.DECISION,
            state=res.state,
            action_path=(),
            to_move=self.to_move_pid,
            decision=None,
            terminal=True,
            own_spun=node.own_spun or own_spun,
        )
        child.value = self._terminal_value(res.state)
        child.expanded = True
        return child

    def _expand_and_evaluate(
        self, node: Node, model: EngineTransitionModel, turn_seed: int, sim: int
    ) -> float:
        """Expand ``node`` (build its edges if a decision node) and value it once.

        The leaf value is computed at creation (§4.2 "evaluated once at
        creation"): a terminal node uses its terminal value; a spun-forced-move
        node is floored at ``_pre_spin_progress`` (the net is NOT consulted); a
        clean node uses the net value head.

        A freshly-created CHANCE node (reached for the first time as an edge's
        child) has no value of its own -- it is the expectation over its draws --
        so it is evaluated by sampling its FIRST outcome's leaf and recording that
        outcome's statistics, exactly as the descent-loop path does on later
        visits. ``node.value`` is the running visit-weighted average over its
        outcomes (here, the single first outcome), which the backup then keeps
        consistent.
        """
        if node.terminal:
            node.expanded = True
            return node.value

        if node.kind == NodeKind.CHANCE:
            node.expanded = True
            outcome = self._select_outcome(node, turn_seed, sim, depth=0)
            if outcome.child is None:
                outcome.child = self._step_chance_child(node, outcome, model)
            value = self._evaluate_leaf(outcome.child)
            outcome.child.value = value
            outcome.n += 1
            outcome.w += value
            node.n += 1
            # Visit-weighted average over outcomes (here just this one).
            tot_n = sum(o.n for o in node.outcomes)
            tot_w = sum(o.w for o in node.outcomes)
            node.value = tot_w / tot_n if tot_n > 0 else value
            self._observe_q(outcome.w / outcome.n)
            return value

        # Decision node: build edges from the net prior, value the node itself.
        node.edges = self._edges_for(node)
        node.expanded = True
        value = self._evaluate_leaf(node)
        node.value = value
        return value

    def _evaluate_leaf(self, node: Node) -> float:
        """Leaf value, ALWAYS in the LEARNER's frame (``to_move_pid``).

        Solo (``two_player=False``, the C0-C5 frame) -- the §4.4 S1 discipline:

        * Spun forced move (``own_spun``): floored at ``_pre_spin_progress`` and
          the net value is NOT consulted -- depth-invariant, so a reckless line is
          never preferred at any sim count.
        * Terminal: the terminal value (``−rounds_remaining`` is 0 at the finish).
        * Clean: the net value head on ``encode_observation(state, pid,
          decision=None)`` (returned as-is; the A1/A2 sign).

        Two-player win/loss (``two_player=True``, Sprint C6):

        * The own-spin floor is DROPPED -- a spin's cost is now expressed entirely
          through whether it loses the race (the rounds-frame floor was a
          ``−rounds_remaining``-scale device with no win/loss image).
        * Terminal: ``+1`` if the LEARNER won the 1v1, ``−1`` if it lost, ``0`` on
          an exact tie (:meth:`_terminal_value`).
        * Clean: the net value head -- a learner-frame expected outcome in
          ``[−1, 1]`` (a P(win) the search consumes). Read from the LEARNER's obs
          so every leaf is on the one zero-sum scale ``_backup`` negates per ply.
        """
        if node.terminal:
            return self._terminal_value(node.state)
        if not self.config.two_player and node.own_spun:
            return self._pre_spin_progress(node.state)
        from heat.ml.features import encode_observation

        obs = encode_observation(node.state, self.to_move_pid, decision=None)
        return self._get_net().leaf_value(obs)  # type: ignore[attr-defined]

    def _terminal_value(self, state: GameState) -> float:
        """Value of a finished state, in the LEARNER's frame.

        Solo (``two_player=False``): ``−rounds_remaining`` is 0 at the finish, so
        the value is ``0.0`` -- the maximum (best) on the higher-is-better
        ``−rounds_remaining`` scale, on the same scale as the net value head so
        backups mix cleanly (the C0-C5 frame, byte-for-byte).

        Two-player win/loss (Sprint C6): ``+1`` if the learner finished ahead of
        the opponent, ``−1`` if behind, ``0`` on an exact tie -- the cleanest
        possible value signal, exact at the leaves the search can reach.
        """
        if not self.config.two_player:
            return 0.0
        return self._winloss_outcome(state)

    def _winloss_outcome(self, state: GameState) -> float:
        """The learner's zero-sum 1v1 outcome at ``state`` (``+1`` / ``−1`` / ``0``).

        Decided by finish order when available (the seat that crossed first wins),
        else by race progress (further-along wins -- the resolved-sub-decision-3
        MAX_ROUNDS tie rule), with ``0`` only on an exact progress tie. Mirrors the
        ``gen_selfplay`` ``z`` backfill so the search leaf and the training target
        agree on the same binary outcome.
        """
        return _winloss_result(state, self.to_move_pid, self.config.opponent_id)

    def _pre_spin_progress(self, state: GameState) -> float:
        """Depth-invariant progress floor for a line whose forced move spun.

        Identical to ``LookaheadAgent._pre_spin_progress``: credit progress only
        up to the corner the car failed to clear (``corner_start - 1``, lap 0), so
        a reckless recovery can never inflate the score. Read from the first
        ``spin_out`` event for our seat on the (logging-enabled) clone.
        """
        for e in state.event_log:
            if e.event_type == "spin_out" and e.player_id == self.to_move_pid:
                corner_start = int(e.data.get("corner_start", 0))
                return float(max(0, corner_start - 1))
        # No spin event found: fall back to lap-aware race progress (shouldn't
        # happen when own_spun is set, but never crash the leaf).
        return float(ME.race_progress(state.get_player(self.to_move_pid), state.track))

    def _forced_move_spun(self, child_state: GameState, node: Node, edge: Edge) -> bool:
        """True iff forcing ``edge.action`` spun the learner out THIS round.

        Reads the spin_out event log on the advanced clone for our seat in the
        round the forced action was played (``node.state.round_num``). This is the
        own-spin signal the §4.4 leaf floor keys on; it is depth-invariant because
        once set on a node it propagates to descendants (``node.own_spun or ...``).
        """
        target_round = node.state.round_num
        for e in child_state.event_log:
            if (
                e.event_type == "spin_out"
                and e.player_id == self.to_move_pid
                and e.round_num == target_round
            ):
                return True
        return False

    # -- backup ----------------------------------------------------------

    def _backup(
        self, path: list[tuple[Node, object]], root: Node, value: float
    ) -> None:
        """Propagate ``value`` up the visited path; update [Qmin, Qmax].

        Decision-node edges accumulate ``(N += 1, W += value)`` so a child's mean
        Q is its running average. Chance-node outcomes accumulate identically, and
        the chance node's value is the **visit-weighted average** of its outcomes
        (running mean) -- the expectation over our own draws (§4.3). Every updated
        mean Q is folded into the running ``[Qmin, Qmax]`` so the next selection's
        normalization reflects the values seen so far.
        """
        root.n += 1
        for parent, link in path:
            parent.n += 1 if parent is not root else 0
            if isinstance(link, Edge):
                link.n += 1
                # Sprint C6 zero-sum negamax: ``value`` is stored in the LEARNER's
                # frame. At an OPPONENT decision node the edge accumulates the
                # NEGATED value, so ``edge.q()`` is the opponent's own (maximizing)
                # frame -- which is exactly what makes the unchanged argmax-(Q̂+U)
                # ``_select_edge`` minimax correctly without a separate min branch.
                # Solo (``two_player=False``) never has an opponent node, so this
                # is the C0-C5 backup byte-for-byte.
                inc = value
                if self.config.two_player and parent.to_move == self.config.opponent_id:
                    inc = -value
                link.w += inc
                self._observe_q(link.q())
            else:  # ChanceOutcome
                link.n += 1
                link.w += value
                self._observe_q(link.w / link.n)
                # The chance node's value is the visit-weighted average over its
                # outcomes (running mean) -- recomputed from the outcome stats so
                # it is exactly Σ w_i / Σ n_i, the correct expectation.
                tot_n = sum(o.n for o in parent.outcomes)
                tot_w = sum(o.w for o in parent.outcomes)
                parent.value = tot_w / tot_n if tot_n > 0 else parent.value

    def _observe_q(self, q: float) -> None:
        """Fold a freshly-updated mean Q into the running [Qmin, Qmax]."""
        if q < self._q_min:
            self._q_min = q
        if q > self._q_max:
            self._q_max = q

    # -- acting (extract the move) ---------------------------------------

    def _extract_plan(
        self,
        root: Node,
        legal_gears: list[tuple[int, int]],
        state: GameState,
        player_id: int,
    ) -> tuple[tuple[int, int], tuple[Card, ...]]:
        """Read the committed (gear, cards) from the searched root.

        Acting rule (§4.5, noise-off greedy): the most-visited root edge is the
        gear (``act_on_q`` switches to argmax normalized-Q). The cards play is the
        most-visited grandchild along the deterministic GEAR->CARDS edge, so one
        search commits both. Falls back to the live legal set if the search did
        not reach a CARDS node (e.g. n_simulations too small).
        """
        # The gear is the acted ROOT edge (temperature applies to the root visit
        # distribution during self-play; greedy in C1's eval).
        gear_edge = self._best_edge(root, at_root=True)
        gear = gear_edge.action  # (new_gear, heat_cost)
        if gear not in legal_gears:
            gear = legal_gears[0]

        cards: tuple[Card, ...] | None = None
        gear_child = gear_edge.child
        if gear_child is not None and gear_child.kind == NodeKind.DECISION \
                and gear_child.decision is not None \
                and gear_child.decision.kind == DecisionKind.CARDS \
                and gear_child.edges:
            cards_edge = self._best_edge(gear_child)
            cards = cards_edge.action  # type: ignore[assignment]

        if cards is None:
            # Search did not reach the CARDS node under the chosen gear; pick the
            # net-greedy play at that gear so the move is still sensible (the
            # n_simulations=1 "prior-greedy" degenerate path the spec asks for).
            cards = self._prior_greedy_cards(state, player_id, gear)
        return gear, cards  # type: ignore[return-value]

    def _best_edge(self, node: Node, *, at_root: bool = False) -> Edge:
        """The acted child edge: most-visited (default) or argmax Q̂ (config).

        Deterministic in C1's eval: ties broken by first-seen order. With Dirichlet
        noise off and the temperature schedule off, this is a pure function of the
        search (hence of ``(state, seed)``).

        Self-play hook (§4.5, OFF by default): for the first ``temperature_moves``
        plies of an episode the ROOT action is SAMPLED from the visit distribution
        ``a ∝ N(a)^{1/τ}`` (τ=1) rather than taken greedily, then τ→0 (greedy).
        The sample uses the deterministic per-turn RNG so even sampled acting is a
        pure function of ``(state, seed)``. ``temperature_moves=0`` (the C1 eval
        default) skips this entirely -- pure greedy.
        """
        # Gumbel root selector (Sprint C4): the acted ROOT action is the
        # Gumbel-selected argmax(g + logits + σ(Q̂)), captured by the search. Used
        # for the self-play *trajectory* action (gen_selfplay); the agent's eval
        # acting stays PUCT because root_selector defaults to "puct" (scope #3).
        if (
            at_root
            and self.config.root_selector == "gumbel"
            and self._gumbel_result is not None
            and self._gumbel_result.edges is node.edges
        ):
            return node.edges[self._gumbel_result.acted_index]

        if self.config.act_on_q:
            return max(
                node.edges,
                key=lambda e: (self._normalize_q(e.q()) if e.n > 0 else -math.inf,),
            )

        # Self-play temperature sampling of the root action (OFF by default).
        if at_root and self._ply < self.config.temperature_moves:
            weights = [float(e.n) for e in node.edges]  # τ=1 => weight = N(a)
            total = sum(weights)
            if total > 0.0:
                return self._search_rng.choices(node.edges, weights=weights, k=1)[0]

        # Most-visited (AZ acting); stable on first-seen order via enumerate index.
        best = node.edges[0]
        best_key = (-1, 0)
        for i, e in enumerate(node.edges):
            key = (e.n, -i)
            if key > best_key:
                best_key = key
                best = e
        return best

    def _prior_greedy_cards(
        self, state: GameState, player_id: int, gear: tuple[int, int]
    ) -> tuple[Card, ...]:
        """Net-greedy CARDS play at ``gear`` (the n_simulations=1 fallback).

        Reads the net prior over the kept candidate plays at the committed gear
        and returns the highest-prior one. Used when the search tree did not reach
        the CARDS node (tiny sim budget). Re-validated by the caller against the
        live legal set in :meth:`choose_cards`.
        """
        from heat.ml.action_codec import encode_action_index, legal_action_mask
        from heat.ml.features import encode_observation

        player = state.get_player(player_id)
        legal = rules.legal_card_plays(player.hand, gear[0])
        decision = Decision(DecisionKind.CARDS, player_id, legal)
        cand = self._candidate_actions(decision, state)
        if not cand:
            return legal[0] if legal else tuple()
        obs = encode_observation(state, player_id, decision)
        mask = legal_action_mask(decision, state)
        probs = self._get_net().policy_prior(obs, mask)  # type: ignore[attr-defined]
        best = cand[0]
        best_p = -1.0
        for a in cand:
            try:
                p = float(probs[encode_action_index(decision, a)])
            except (ValueError, IndexError):
                p = 0.0
            if p > best_p:
                best_p = p
                best = a
        return best  # type: ignore[return-value]

    # -- plan caching (mirror LookaheadAgent) ----------------------------

    def _ensure_plan(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> None:
        sig = self._turn_signature(state, player_id)
        if self._plan_sig == sig and self._plan_gear is not None:
            return
        gear, cards = self._plan_turn(state, player_id, legal_gears)
        self._plan_sig = sig
        self._plan_gear = gear
        self._plan_cards = cards

    # -- BaseAgent decision methods --------------------------------------

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        self._ensure_plan(state, player_id, legal_gears)
        gear = self._plan_gear
        if gear is None or gear not in legal_gears:
            gear, cards = self._plan_turn(state, player_id, legal_gears)
            self._plan_sig = self._turn_signature(state, player_id)
            self._plan_gear = gear
            self._plan_cards = cards
        return gear  # type: ignore[return-value]

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        if len(legal_plays) == 1:
            return legal_plays[0]
        sig = self._turn_signature(state, player_id)
        if (
            self._plan_sig == sig
            and self._plan_cards is not None
            and self._plan_cards in legal_plays
        ):
            return self._plan_cards
        # Cache miss / stale: recompute restricted to the committed gear.
        committed_gear = state.get_player(player_id).gear
        legal_gears = rules.legal_gear_shifts(
            committed_gear, state.get_player(player_id).heat_available
        )
        gear_opts = [g for g in legal_gears if g[0] == committed_gear]
        if not gear_opts:
            gear_opts = [(committed_gear, 0)]
        _, cards = self._plan_turn(state, player_id, gear_opts)
        if cards in legal_plays:
            return cards
        return legal_plays[0]

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        """Search the REACT decision (it is a tree-branched DecisionKind, §4).

        REACT/SLIPSTREAM/DISCARD are searched, not delegated to a heuristic. A
        full sub-search rooted at each of these mid-round decisions would re-pay
        the per-move budget every step; instead the agent commits the net-greedy
        legal action for these (the prior IS the searched policy's read at that
        node, and the joint plan search already accounts for the *gear+cards* that
        dominate corner outcomes). This keeps every kind codec-encoded and
        net-driven -- never a hand-coded heuristic rollout -- which is the
        all-decision-coverage contract the unit test pins.
        """
        decision = Decision(
            DecisionKind.REACT,
            player_id,
            rules.ReactOptions(
                max_cooldown=max_cooldown,
                can_boost=can_boost,
                has_adrenaline=has_adrenaline,
            ),
        )
        return self._net_greedy_action(decision, state)  # type: ignore[return-value]

    def choose_slipstream(self, state: GameState, player_id: int) -> bool:
        decision = Decision(DecisionKind.SLIPSTREAM, player_id, True)
        return bool(self._net_greedy_action(decision, state))

    def choose_discard(
        self, state: GameState, player_id: int, discardable: list[Card]
    ) -> list[Card]:
        decision = Decision(DecisionKind.DISCARD, player_id, discardable)
        return self._net_greedy_action(decision, state)  # type: ignore[return-value]

    def _net_greedy_action(self, decision: Decision, state: GameState) -> object:
        """Decode the net's highest-prior legal action for ``decision``.

        The net policy IS the searched prior at these nodes; reading its argmax
        over the codec-legal mask keeps REACT/SLIPSTREAM/DISCARD net-driven and
        codec-encoded (never a heuristic), and always legal (decoded from the
        legal mask). Mirrors :class:`MLAgent`'s predict+decode for these kinds.
        """
        from heat.ml.action_codec import decode_action, legal_action_mask
        from heat.ml.features import encode_observation

        obs = encode_observation(state, decision.player_id, decision)
        mask = legal_action_mask(decision, state)
        probs = self._get_net().policy_prior(obs, mask)  # type: ignore[attr-defined]
        # Argmax over legal (masked) actions.
        masked = np.where(np.asarray(mask, dtype=bool), probs, -np.inf)
        flat = int(np.argmax(masked))
        return decode_action(decision, flat, state)


# ---------------------------------------------------------------------------
# Picklable factory (mirrors lookahead_agent_factory; pickle-by-path)
# ---------------------------------------------------------------------------


def _make_mcts_agent(
    player_id: int,
    seed: int | None,
    model_path: str,
    name: str | None,
    config: MCTSConfig | None,
) -> BaseAgent:
    """Top-level (picklable) constructor for an :class:`MCTSAgent`.

    A plain top-level function (no lambda/closure) so the
    :func:`functools.partial` factory pickles into ``ProcessPoolExecutor``
    workers; it carries only the checkpoint *path* (a string) and a plain
    :class:`MCTSConfig` dataclass. The agent seed is derived from the per-game
    ``seed`` so a parallel eval batch is reproducible and seat-stable, while the
    search remains a pure function of ``(state, seed)``.
    """
    agent_name = name if name is not None else f"MCTS-{player_id}"
    return MCTSAgent(
        model_path=model_path,
        config=config,
        seed=seed if seed is not None else player_id,
        name=agent_name,
    )


def mcts_agent_factory(
    model_path: str,
    name: str | None = None,
    *,
    config: MCTSConfig | None = None,
) -> Callable[..., BaseAgent]:
    """Return a picklable factory producing :class:`MCTSAgent`s from ``model_path``.

    Mirrors :func:`heat.ml.evaluate.lookahead_agent_factory`: the returned callable
    is a :func:`functools.partial` of a top-level constructor (never a lambda), so
    it pickles into ``run_batch(parallel=True)`` / ``ProcessPoolExecutor`` workers.
    It carries only the checkpoint path and a plain :class:`MCTSConfig`; the heavy
    SB3 model loads lazily in-process (each worker reloads by path -- the §6.5 /
    MLAgent pattern), so the search drops into the eval harness / league unchanged.
    """
    import functools

    return functools.partial(
        _make_mcts_agent,
        model_path=model_path,
        name=name,
        config=config,
    )
