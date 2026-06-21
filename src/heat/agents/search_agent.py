"""Decision-time forward-rollout search agent (Sprint S1 / "Sprint 9a").

``LookaheadAgent`` exploits the one structural advantage this project has over a
model-free learner: **we own a perfect forward simulator**
(:func:`heat.engine.driver.run_round_driver` + :meth:`GameState.clone`). Where
:class:`~heat.agents.strong_heuristic.StrongHeuristicAgent` scores a candidate
``(gear, cards)`` pair with a *static* one-ply value
(:mod:`heat.agents._move_eval`), this agent **actually plays the move forward**
``horizon`` rounds through the real engine -- using a configurable default
rollout policy for every other decision (opponents AND its own
REACT/SLIPSTREAM/DISCARD, AND every subsequent round) -- and scores the
resulting line by ``progress - own_spin_penalty*own_spins -
spin_penalty*later_spins``, where ``own_spins`` are spins the forced candidate
move causes on the FIRST round (the lever the search controls) and a line whose
own move spins is credited only *pre-spin* progress so deeper horizons cannot
reward reckless recovery (see ``_leaf_score`` / ``_pre_spin_progress``). Because the simulator is
exact, the agent can *see a spin coming* on the next corner before it commits to
the gear that causes it. That is precisely the lever the model-free investigation
(``project-solo-driving-decision-trace``) found PPO could never sample: the clean
limit-1 recovery.

Design (mirrors the strong heuristic's plan-once/execute-consistently contract):

  * At :meth:`choose_gear` the agent enumerates the joint
    ``legal_gear_shifts x legal_card_plays`` space (pruned by a
    dedup-by-resulting-speed filter to tame the 494-wide CARDS branch), forks the
    live state, and rolls each candidate ``(gear, cards)`` plan forward to the
    horizon, scoring ``progress - spin_penalty * spins`` averaged over
    ``n_determinizations`` clones (own-draw randomness). It caches the best
    ``(gear, cards)`` keyed by a turn signature.
  * :meth:`choose_cards` returns the cached play (re-validated against the legal
    set the engine hands in), or recomputes if stale.
  * :meth:`choose_react` / :meth:`choose_slipstream` / :meth:`choose_discard`
    delegate to the configurable ``rollout_policy`` instance.

``horizon=0`` short-circuits the rollout and reduces to **greedy one-ply**: the
candidate plans are scored by the leaf value (``progress`` or ``move_eval``)
applied to a single forced first round, which is exactly what a 1-ply search
does. This keeps a clean, testable equivalence (see ``tests/test_search_agent``).

**Determinism.** A unit test requires byte-stable plan selection for a fixed
state + seed. ``GameState.clone()`` with the default (``reseed=None``) forks a
child RNG from the parent AND advances the parent by one draw -- so cloning N
candidates in a loop from the *live* state would couple their seeds to loop
order and to how many draws the live game already made. We therefore derive a
**deterministic per-(candidate, determinization) reseed** from a single turn
seed and clone with an explicit ``reseed=`` each time. Same state + same turn
seed => identical clones => identical scores => identical argmax. The turn seed
is taken from the agent's own optional ``seed`` mixed with the turn signature,
so it never reaches into global RNG.

**Hidden information (S1 scope).** Solo is nearly deterministic (only the
learner's own draws are random) and is the primary validation target. In
multiplayer the rollout clones the *actual* opponent hands/decks ("open-hand"
search) rather than sampling determinizations from the public belief -- proper
hidden-info determinization is explicitly DEFERRED to Sprint S2. The
``n_determinizations`` average therefore captures the learner's own-draw variance
in solo, and is a (mild, optimistic) open-hand average in 4p. This is documented
here and called out in the eval harness; treat solo/limit-1 as the gating metric.
"""

from __future__ import annotations

import random
import time
from typing import Callable

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents import _move_eval as ME
from heat.ml.opponents import opponent_action


#: Turn signature: identifies the planning context so a cached plan is only
#: reused within the same turn for the same player. Mirrors the strong
#: heuristic's ``_TurnSig``.
_TurnSig = tuple[int, int, int, int, int]

#: Default penalty (in spaces) charged per spin-out observed in the rollout's
#: *later* rounds (rollout-policy-driven). Sized comparably to
#: :data:`heat.agents._move_eval.SPINOUT_LOSS_SPACES`. These spins are largely
#: identical across candidates (they happen in steps the search does not force),
#: so this term mostly nudges; the lever that actually steers is the
#: *own-move* penalty below.
DEFAULT_SPIN_PENALTY: float = 11.0

#: Penalty (in spaces) charged per spin-out the learner suffers on the FIRST
#: simulated round -- i.e. directly caused by the forced candidate ``(gear,
#: cards)`` move (plus the learner's own REACT/slipstream that round). Unlike
#: later-round spins, this one *differs across candidates*: it is exactly the
#: spin the search controls. It is sized to dominate any plausible progress a
#: reckless line can bank, so a candidate whose own move spins is effectively
#: terminal-dominated regardless of horizon depth. This is the fix for the
#: "inert spin_penalty" pathology (Problem A) and, together with pre-spin
#: progress crediting, the "horizon blows up" pathology (Problem B).
DEFAULT_OWN_SPIN_PENALTY: float = 1000.0

#: Default branching cap. ``None`` keeps every distinct-by-speed candidate (the
#: S1 behaviour). An int keeps only the ``top_k`` candidates ranked by the fast
#: ``_move_eval`` prior before the expensive rollout -- the design's "top-k by a
#: fast prior" branching control. Sized to comfortably contain the best play on
#: the tight distribution while taming the 494-wide CARDS branch at depth.
DEFAULT_TOP_K: int | None = None

#: Default per-move simulation budget (clones). ``None`` = unbounded (S1). When
#: set, the planner stops scoring further candidates once this many rollout
#: clones have been spent this move, so deep/wide search stays affordable. The
#: top-k prior ordering ensures the highest-prior candidates are scored first,
#: so a budget cut never drops a likely-best play before a likely-worse one.
DEFAULT_SIM_BUDGET: int | None = None


class SearchProfile:
    """Lightweight per-agent profiling accumulator (clones, time, moves).

    The design asks S2 to "record profiling numbers (clones/move, ms/move)".
    A :class:`LookaheadAgent` owns one of these and updates it on every planned
    move; :meth:`summary` returns the per-move averages the eval harness prints.
    Cheap (a few counter increments) so it is always on -- no flag to forget.
    """

    __slots__ = ("clones", "moves", "seconds")

    def __init__(self) -> None:
        self.clones = 0
        self.moves = 0
        self.seconds = 0.0

    def record_move(self, clones: int, seconds: float) -> None:
        self.clones += clones
        self.seconds += seconds
        self.moves += 1

    def clones_per_move(self) -> float:
        return self.clones / self.moves if self.moves else 0.0

    def ms_per_move(self) -> float:
        return (self.seconds / self.moves * 1000.0) if self.moves else 0.0

    def summary(self) -> str:
        return (
            f"moves={self.moves} clones/move={self.clones_per_move():.1f} "
            f"ms/move={self.ms_per_move():.2f}"
        )


def _default_rollout_policy() -> BaseAgent:
    """Build the default rollout/leaf policy.

    A factory (not a module-level singleton) so each agent owns an independent
    policy instance and there is no shared mutable state across agents/threads.
    The design names :class:`HeuristicAgent` as the default because, in the
    prototype, wrapping a heuristic rollout took spins-per-limit-1-pass to 0.
    """
    return HeuristicAgent(name="RolloutHeuristic")


class LookaheadAgent(BaseAgent):
    """Forward-rollout search agent (see module docstring).

    Args:
        name: Display name (default ``"Lookahead"``).
        rollout_policy: The default policy used for every decision the search
            does not itself force -- opponents, the learner's own
            REACT/SLIPSTREAM/DISCARD, and every round past the first. Defaults
            to a fresh :class:`HeuristicAgent`. Accepts either a ready
            :class:`BaseAgent` instance or a zero-arg factory returning one.
        horizon: Number of rounds to simulate forward per candidate. ``0``
            short-circuits to greedy one-ply (no rollout); ``2`` (the default)
            is the prototype's sweet spot -- deep enough to anticipate the
            next-turn corner.
        n_determinizations: Number of independent clones to average each
            candidate's score over, capturing the learner's own-draw randomness.
            Default ``2`` (the prototype value).
        spin_penalty: ``lambda`` charged per spin in the rollout's *later*
            rounds (spaces per spin). Default :data:`DEFAULT_SPIN_PENALTY`.
        own_spin_penalty: penalty charged per spin the learner suffers on the
            *first* simulated round -- the spin directly caused by the forced
            candidate move, and the only spin the search can actually steer.
            Sized large to dominate banked progress. Default
            :data:`DEFAULT_OWN_SPIN_PENALTY`.
        leaf_value: ``"progress"`` (lap-aware race progress -- the prototype's
            objective), ``"move_eval"`` (the strong heuristic's spaces-currency
            :func:`_move_eval.evaluate_move`), or ``"learned"`` (Sprint A2: a
            trained rest-of-lap value net ``V`` scores a *clean* rolled-out leaf
            in place of ``progress``; spinning leaves keep the ``_pre_spin_progress``
            floor unchanged). Default ``"progress"``.
        value_model_path: Path to the A1 value-net checkpoint (SB3 ``.zip`` with a
            ``.meta.json`` sidecar). Required when ``leaf_value == "learned"``;
            ignored otherwise. The model is lazy-loaded once on first use (CPU
            device) and validated against the live ML contract
            (``obs_dim``/``action_dim``/``codec_version``), mirroring
            :class:`heat.agents.ml_agent.MLAgent`. Nulled in ``__getstate__`` so
            the agent pickles by path to ``ProcessPoolExecutor`` workers.
        determinize_hidden: When True (S2), each rollout clone re-samples the
            *opponents'* hidden hands/decks from the public belief (uniform over
            the cards known to be in their hand+draw pile) before rolling forward
            -- "determinized search". When False (S1 default), the rollout sees
            the opponents' actual hands ("open-hand" search). Has no effect in
            solo play (no opponents). See :meth:`_determinize_opponents`.
        top_k: Branching control. When set, only the ``top_k`` candidate plays
            ranked highest by the fast ``_move_eval`` prior are rolled out (the
            others are pruned before the expensive search). ``None`` (default)
            keeps every distinct-by-speed candidate (S1).
        sim_budget: Per-move cap on rollout clones. When set, the planner stops
            after this many clones have been spent on the current move (candidates
            are scored highest-prior-first so a cut never drops a likely-best
            play first). ``None`` (default) is unbounded (S1).
        seed: Optional seed mixed into the per-turn rollout RNG so a seeded batch
            is reproducible. The agent is otherwise a deterministic function of
            the state (no global RNG is ever touched).
    """

    def __init__(
        self,
        name: str = "Lookahead",
        *,
        rollout_policy: BaseAgent | Callable[[], BaseAgent] | None = None,
        horizon: int = 2,
        n_determinizations: int = 2,
        spin_penalty: float = DEFAULT_SPIN_PENALTY,
        own_spin_penalty: float = DEFAULT_OWN_SPIN_PENALTY,
        leaf_value: str = "progress",
        value_model_path: str | None = None,
        determinize_hidden: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        sim_budget: int | None = DEFAULT_SIM_BUDGET,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        if n_determinizations < 1:
            raise ValueError(
                f"n_determinizations must be >= 1, got {n_determinizations}"
            )
        if leaf_value not in ("progress", "move_eval", "learned"):
            raise ValueError(
                "leaf_value must be 'progress', 'move_eval', or 'learned', got "
                f"{leaf_value!r}"
            )
        if leaf_value == "learned" and value_model_path is None:
            raise ValueError(
                "value_model_path is required when leaf_value == 'learned' "
                "(the A1 value-net checkpoint the leaf scores clean leaves with)"
            )
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
        if sim_budget is not None and sim_budget < 1:
            raise ValueError(
                f"sim_budget must be >= 1 or None, got {sim_budget}"
            )

        # Resolve the rollout policy (instance or factory) into an instance.
        if rollout_policy is None:
            self.rollout_policy: BaseAgent = _default_rollout_policy()
        elif isinstance(rollout_policy, BaseAgent):
            self.rollout_policy = rollout_policy
        else:
            self.rollout_policy = rollout_policy()

        self.horizon = horizon
        self.n_determinizations = n_determinizations
        self.spin_penalty = float(spin_penalty)
        self.own_spin_penalty = float(own_spin_penalty)
        self.leaf_value = leaf_value
        self.value_model_path = value_model_path
        self.determinize_hidden = determinize_hidden
        self.top_k = top_k
        self.sim_budget = sim_budget
        self.seed = seed

        #: Lazily-loaded value net (SB3 ``MaskablePPO``) for ``leaf_value ==
        #: "learned"``; ``None`` until the first clean leaf is scored. Nulled in
        #: ``__getstate__`` so the agent pickles by path (the MLAgent pattern).
        self._value_model = None

        #: Per-agent profiling (clones/move, ms/move). Always on; cheap.
        self.profile = SearchProfile()

        # Cached plan for the current turn (mirrors StrongHeuristicAgent).
        self._plan_sig: _TurnSig | None = None
        self._plan_gear: tuple[int, int] | None = None
        self._plan_cards: tuple[Card, ...] | None = None

    # ------------------------------------------------------------------
    # Learned-leaf value net (Sprint A2) -- lazy load, MLAgent patterns
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Pickle by path: never ship the heavy SB3 value model to workers.

        Mirrors :meth:`heat.agents.ml_agent.MLAgent.__getstate__`. The model is
        reloaded from :attr:`value_model_path` in the worker on first use, so the
        learned-leaf agent's ``functools.partial`` factory pickles into a
        ``ProcessPoolExecutor`` carrying only the path string.
        """
        state = self.__dict__.copy()
        state["_value_model"] = None
        return state

    def _validate_value_meta(self) -> None:
        """Assert the value checkpoint's sidecar matches the live ML contract.

        Identical tripwire to :meth:`MLAgent._validate_meta` (§3.4): a stale V --
        one trained against a now-drifted ``obs_dim`` / ``action_dim`` /
        ``codec_version`` -- fails fast with a
        :class:`~heat.agents.ml_agent.CheckpointMismatchError` instead of silently
        scoring leaves with a garbage value head.
        """
        from heat.agents.ml_agent import CheckpointMismatchError
        from heat.ml import spaces
        from heat.ml.training import load_meta

        try:
            meta = load_meta(self.value_model_path)
        except FileNotFoundError as exc:
            raise CheckpointMismatchError(
                f"Value-net sidecar not found for {self.value_model_path!r}; "
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
                f"Value net {self.value_model_path!r} is incompatible with the "
                "current ML contract (stale model vs drifted codec): "
                + "; ".join(mismatches)
            )

    def _get_value_model(self):
        """Lazy-load + validate the value net (CPU), caching it on the instance.

        Mirrors :meth:`MLAgent._get_model`: validate the sidecar tripwire first,
        then ``MaskablePPO.load(..., device="cpu")`` once. CPU inference is fine
        at this scale (one critic forward per clean leaf -- README A `Risks`).
        """
        if self._value_model is None:
            from sb3_contrib import MaskablePPO

            self._validate_value_meta()
            self._value_model = MaskablePPO.load(
                self.value_model_path, device="cpu"
            )
        return self._value_model

    def _value_leaf(self, clone: GameState, player_id: int) -> float:
        """Learned rest-of-lap value of a clean rolled-out leaf (Sprint A2).

        Encodes the leaf *state* with ``encode_observation(clone, player_id,
        decision=None)`` -- byte-identical to the convention A1 trained V on (the
        zero-filled phase block; V scores a state, not a pending decision) -- runs
        the critic forward (``policy.predict_values``), and returns the **negated**
        scalar.

        The net regresses ``target = -rounds_remaining`` (README A `2.1`), so
        ``predict_values`` already emits the higher-is-better quantity and the
        leaf returns it directly (negating ``-(-rounds_remaining)`` would flip the
        sign back to *more rounds is better*). We pass it through as-is: a state
        nearer the finish (fewer rounds remaining) predicts closer to ``0`` (a
        larger, less-negative value) and is preferred, slotting into the existing
        ``_leaf_score`` sign convention exactly like ``progress``.
        """
        import numpy as np
        import torch

        from heat.ml.features import encode_observation

        obs = encode_observation(clone, player_id, decision=None)
        model = self._get_value_model()
        ob = torch.as_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            v = model.policy.predict_values(ob)
        return float(np.asarray(v.detach()).reshape(-1)[0])

    # ------------------------------------------------------------------
    # Plan caching helpers (mirror StrongHeuristicAgent)
    # ------------------------------------------------------------------

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
        """A deterministic per-turn seed (no global RNG).

        Mixes the agent's optional ``seed`` with the turn signature so two turns
        with different contexts use different rollout streams, while the SAME
        turn always reseeds the same clones -- the determinism the unit test
        requires. ``hash`` is salted by ``PYTHONHASHSEED`` for some types, so we
        fold the signature ints by hand into a stable, platform-independent value.
        """
        base = 0 if self.seed is None else int(self.seed)
        acc = (base & 0x7FFFFFFF) * 2654435761
        for v in sig:
            acc = (acc * 1000003 + int(v)) & 0x7FFFFFFF
        return acc

    # ------------------------------------------------------------------
    # Candidate enumeration + dedup-by-speed prune (Risk 2)
    # ------------------------------------------------------------------

    def _candidate_plans(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], tuple[Card, ...]]]:
        """Enumerate ``(gear, cards)`` candidates, deduped by resulting speed.

        The 494-wide CARDS space explodes the branching factor; many distinct
        card plays move the car the *same* number of spaces (e.g. two different
        value-2 cards). Since the engine resolves a turn by total speed, plays
        that yield the same ``(gear, total card value)`` are interchangeable for
        the rollout, so we keep only the first play seen per ``(gear, speed)``
        key. This is the prototype's prune (design Risk 2): it preserves every
        *distinct outcome* while collapsing redundant branches.
        """
        player = state.get_player(player_id)
        plans: list[tuple[tuple[int, int], tuple[Card, ...]]] = []
        seen: set[tuple[int, int]] = set()
        for gear, gear_heat in legal_gears:
            for play in rules.legal_card_plays(player.hand, gear):
                speed = sum(c.value for c in play)
                key = (gear, speed)
                if key in seen:
                    continue
                seen.add(key)
                plans.append(((gear, gear_heat), play))
        return plans

    def _candidate_prior(
        self,
        state: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
    ) -> float:
        """Fast static score of a ``(gear, cards)`` candidate (no rollout).

        Used only to RANK candidates for top-k branching control and budget
        ordering -- never to choose the final move (the rollout does that). It
        reuses the strong heuristic's spaces-currency :func:`_move_eval` so the
        ranking matches the project's own notion of a good move, which is far
        cheaper than a clone+rollout. Computed against the *live* state's player
        (no clone), so the whole prior pass costs nothing in clones.
        """
        player = state.get_player(player_id)
        exp_stress = ME.expected_basic_value(player)
        exp_speed = ME.play_speed(cards, exp_stress)
        result = ME.evaluate_move(
            state,
            player_id,
            expected_speed=exp_speed,
            heat_spent=gear[1],
            from_position=player.position,
            from_lap=player.lap,
            planned_gear=gear[0],
            heat_price=ME.DEFAULT_HEAT_PRICE,
            horizon_corners=1,
            opponent_aware=True,
            blocking=False,
            enable_solvency=True,
        )
        return result.value

    def _prune_top_k(
        self,
        state: GameState,
        player_id: int,
        candidates: list[tuple[tuple[int, int], tuple[Card, ...]]],
    ) -> list[tuple[tuple[int, int], tuple[Card, ...]]]:
        """Order candidates by the fast prior and keep the best (top-k).

        Returns candidates sorted by descending prior (so the planner scores the
        most promising lines first -- which makes a ``sim_budget`` cut drop only
        the least-promising candidates). When ``top_k`` is set, truncates to that
        many. The sort is *stable* on the original enumeration order via the
        index tiebreak, so determinism is preserved (equal-prior candidates keep
        their S1 order). When ``top_k`` is None we still sort -- this only matters
        for budget ordering and never changes the unbudgeted argmax.
        """
        scored = [
            (self._candidate_prior(state, player_id, gear, cards), idx, gear, cards)
            for idx, (gear, cards) in enumerate(candidates)
        ]
        # Descending prior, ties broken by original index (stable, deterministic).
        scored.sort(key=lambda t: (-t[0], t[1]))
        ordered = [(gear, cards) for _, _, gear, cards in scored]
        if self.top_k is not None:
            ordered = ordered[: self.top_k]
        return ordered

    # ------------------------------------------------------------------
    # Hidden-information determinization (S2)
    # ------------------------------------------------------------------

    def _determinize_opponents(
        self, clone: GameState, player_id: int, det_rng: random.Random
    ) -> None:
        """Re-sample opponents' hidden hands/decks from the public belief.

        From the learner's seat the *contents* of an opponent's hand+draw-pile
        are public (the standard deck plus any stress/heat they have taken, all
        observable from play), but the *partition* into "in hand" vs "in draw
        pile" and the draw order are hidden. The maximum-entropy belief
        consistent with that knowledge is: pool each opponent's current
        ``hand + draw_pile`` cards, shuffle, deal back a hand of the same size,
        and leave the remainder as the (shuffled) draw pile. The discard pile is
        left untouched (cards already played/discarded are public and their
        position is fixed). This is the "determinized search" the design calls
        for; averaging over several such determinizations approximates the
        expectation over the opponents' hidden state.

        Mutates ``clone`` in place. Only opponents are resampled -- the learner's
        own hand is not hidden from itself. No-op in solo play. The learner's own
        future *draws* remain a chance node handled by ``n_determinizations``
        re-seeding (see :meth:`_score_plan`).
        """
        for opp in clone.players:
            if opp.player_id == player_id or opp.finished:
                continue
            deck = opp.deck
            hand_size = len(opp.hand)
            # Pool the hidden cards (hand + draw pile); discard stays put.
            pool = list(opp.hand) + list(deck.draw_pile)
            if not pool:
                continue
            # Canonicalize before shuffling so the result depends only on the
            # det RNG + the (multiset) contents, never on prior list order
            # (mirrors Deck.attach_rng's id-sort-then-shuffle discipline).
            pool.sort(key=lambda c: c.id)
            det_rng.shuffle(pool)
            opp.hand = pool[:hand_size]
            remaining = pool[hand_size:]
            # Rebuild the draw pile in-place, preserving the existing discard.
            deck._draw_pile = list(remaining)

    # ------------------------------------------------------------------
    # Rollout core
    # ------------------------------------------------------------------

    def _score_plan(
        self,
        state: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
        turn_seed: int,
        plan_index: int,
    ) -> float:
        """Average score of a single ``(gear, cards)`` plan over determinizations.

        Each determinization clones the live ``state`` with an explicit,
        deterministic ``reseed`` (so the score is a pure function of the state +
        turn seed + indices, never of clone-loop order or the live game's RNG
        position) and rolls it forward ``horizon`` rounds via
        :meth:`_rollout_once`. ``horizon == 0`` scores the forced first round
        only (greedy one-ply).
        """
        total = 0.0
        for det in range(self.n_determinizations):
            reseed = (turn_seed + plan_index * 100003 + det * 31) & 0x7FFFFFFF
            clone = state.clone(reseed=reseed)
            # We count spins by reading the spin_out event log on the clone, so
            # logging must be on regardless of the live game's setting. The
            # clone is throwaway, so this never touches the real event log.
            clone.logging_enabled = True
            if self.determinize_hidden:
                # A chance node over the opponents' hidden state: re-deal their
                # hands/decks from the public belief, with a det-specific RNG
                # derived from the same deterministic reseed so the whole score
                # stays a pure function of (state, turn_seed, plan, det).
                self._determinize_opponents(
                    clone, player_id, random.Random(reseed ^ 0x5DEECE66)
                )
            total += self._rollout_once(clone, player_id, gear, cards)
        return total / self.n_determinizations

    def _rollout_once(
        self,
        clone: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
    ) -> float:
        """Roll one clone forward and return the leaf score for this line.

        The first round forces the learner's ``gear`` then ``cards``; every other
        decision (opponents, the learner's own REACT/SLIPSTREAM/DISCARD, and all
        decisions in subsequent rounds) is answered by ``rollout_policy`` via
        :func:`opponent_action`. ``horizon`` rounds are simulated (or 1 round
        when ``horizon == 0``, scored at the leaf without crediting depth).

        We record the round number the clone starts at so that, after the
        rollout, spins can be split into **own-move spins** (those that occurred
        in the FIRST simulated round -- caused by the forced candidate move,
        which is what differs across candidates and what the search can steer)
        and later **rollout spins** (rollout-policy-driven, largely identical
        across candidates). The split is the key to making the spin penalty
        actually bite (Problem A); scoring is delegated to :meth:`_leaf_score`.
        """
        rounds = max(1, self.horizon)
        forced_gear = gear
        forced_cards = cards
        first_round = True
        first_round_num = clone.round_num

        for _ in range(rounds):
            if clone.is_game_over:
                break
            gen = run_round_driver(clone)
            try:
                decision = next(gen)
                while True:
                    action = self._answer_rollout_decision(
                        clone,
                        decision,
                        player_id,
                        forced_gear if first_round else None,
                        forced_cards if first_round else None,
                    )
                    decision = gen.send(action)
            except StopIteration:
                pass
            first_round = False

        # Count spins ONCE over the whole rollout (reading the cumulative log per
        # round would multiply-count an early spin -- the validator's fix). Split
        # by the round they occurred in: first-round spins are the forced move's
        # own spins.
        own_spins, later_spins = self._count_spins(
            clone, player_id, first_round_num
        )
        return self._leaf_score(
            clone, player_id, gear, cards, own_spins, later_spins
        )

    def _answer_rollout_decision(
        self,
        clone: GameState,
        decision: Decision,
        player_id: int,
        forced_gear: tuple[int, int] | None,
        forced_cards: tuple[Card, ...] | None,
    ) -> object:
        """Answer one rollout decision: force the candidate, else delegate.

        On the first round, the learner's GEAR/CARDS are forced to the candidate
        plan (re-validated against the legal set the driver hands in -- a forced
        gear/cards may be stale if the engine adjusted the player, in which case
        we fall back to the rollout policy). Everything else routes through
        ``opponent_action`` so the rollout policy answers it.
        """
        if decision.player_id == player_id:
            if decision.kind == DecisionKind.GEAR and forced_gear is not None:
                if forced_gear in decision.legal:  # type: ignore[operator]
                    return forced_gear
            elif decision.kind == DecisionKind.CARDS and forced_cards is not None:
                if forced_cards in decision.legal:  # type: ignore[operator]
                    return forced_cards
        # Opponents, the learner's other decisions, and all later rounds.
        return opponent_action(self.rollout_policy, decision, clone)

    def _count_spins(
        self, clone: GameState, player_id: int, first_round_num: int
    ) -> tuple[int, int]:
        """Count ``spin_out`` events for ``player_id``, split by round.

        Returns ``(own_spins, later_spins)`` where ``own_spins`` are spins that
        occurred in the FIRST simulated round (``round_num == first_round_num``)
        -- the spins directly caused by the forced candidate move -- and
        ``later_spins`` are all spins in subsequent rollout rounds (rollout-policy
        driven). ``GameEvent.round_num`` is recorded on each event, so the split
        is exact. (The clone is discarded after scoring, so the growing log is
        harmless and cheap at ``horizon <= 2``.)
        """
        own = 0
        later = 0
        for e in clone.event_log:
            if e.event_type == "spin_out" and e.player_id == player_id:
                if e.round_num == first_round_num:
                    own += 1
                else:
                    later += 1
        return own, later

    def _leaf_score(
        self,
        clone: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
        own_spins: int,
        later_spins: int,
    ) -> float:
        """Score a rolled-out leaf.

        ``leaf_value == "progress"`` reads the lap-aware race progress reached
        on the clone (the prototype objective). ``leaf_value == "move_eval"``
        instead values the *forced first move* with the strong heuristic's
        spaces-currency evaluator (a static one-ply value), which is the cheap
        high-quality leaf the design lists as an option. ``leaf_value ==
        "learned"`` (Sprint A2) scores a **clean** leaf with the trained value net
        V (``-rounds_remaining``, higher is better) in place of ``progress``; a
        spinning leaf (``own_spins > 0``) keeps the ``_pre_spin_progress`` floor
        unchanged (V is never called there), so the corner-skill scoring is
        preserved by construction.

        Two penalties are then subtracted:

        * ``own_spin_penalty * own_spins`` -- spins the forced candidate move
          caused on the first round. This is the term that *differs across
          candidates* (Problem A): a candidate whose own move spins is dominated
          by one whose does not. Sized large so it cannot be outrun by progress.
        * ``spin_penalty * later_spins`` -- spins in later rollout rounds; mostly
          a constant offset across candidates, kept as a mild tie-breaker.

        **Horizon stability (Problem B).** When the learner spins on the forced
        first move, the engine resets it to ``corner.start - 1`` and the rollout
        policy then drives ``horizon - 1`` more rounds, banking recovery
        progress. End-of-horizon ``progress`` therefore *grows with depth* for a
        reckless line, which let h>=3 select spinning lines. To make the leaf
        depth-stable, when ``own_spins > 0`` we evaluate progress at the
        *pre-spin* position (where the car was when its forced move failed),
        never crediting the post-spin recovery. A clean line is then strictly
        preferred at every horizon.
        """
        player = clone.get_player(player_id)
        if self.leaf_value == "progress":
            if own_spins > 0:
                value = self._pre_spin_progress(clone, player_id)
            else:
                value = float(ME.race_progress(player, clone.track))
        elif self.leaf_value == "learned":
            # Sprint A2: V replaces ONLY the `value` term and ONLY on clean
            # leaves. A line whose forced move spun keeps the depth-invariant
            # `_pre_spin_progress` floor (V is never consulted), so the dominant
            # own-spin penalty still bites at every horizon and a spinning
            # candidate stays terminal-dominated relative to any clean one.
            if own_spins > 0:
                value = self._pre_spin_progress(clone, player_id)
            else:
                value = self._value_leaf(clone, player_id)
        else:  # "move_eval"
            value = self._move_eval_value(clone, player_id, gear, cards)
        return (
            value
            - self.own_spin_penalty * own_spins
            - self.spin_penalty * later_spins
        )

    def _pre_spin_progress(self, clone: GameState, player_id: int) -> float:
        """Depth-invariant progress floor for a line whose forced move spun.

        Reads the first ``spin_out`` event for our seat and credits progress only
        up to the corner it failed to clear (``corner_start - 1``, lap 0). This
        is the **primary** Problem-B lever: it caps the score so the rollout
        policy's post-spin recovery over the remaining ``horizon - 1`` rounds
        cannot inflate a reckless line, no matter how deep the horizon. An
        ablation confirms it: at horizon 3 without this floor the agent banks
        recovery progress and selects spinning lines (~370 game spins across the
        held-out set); with it, ~8. (The large ``own_spin_penalty`` is a second,
        independent guard that also cures the blow-up on its own -- we keep both
        so neither is load-bearing alone.) Using lap 0 keeps this strictly below
        any clean line's progress, which is all the argmax needs.
        """
        track = clone.track
        for e in clone.event_log:
            if e.event_type == "spin_out" and e.player_id == player_id:
                corner_start = int(e.data.get("corner_start", 0))
                return float(max(0, corner_start - 1))
        # No spin event found (shouldn't happen when own_spins>0): fall back.
        return float(ME.race_progress(clone.get_player(player_id), track))

    def _move_eval_value(
        self,
        clone: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
    ) -> float:
        """Static one-ply value of the forced move via ``_move_eval``.

        Used only when ``leaf_value == "move_eval"``. Mirrors the strong
        heuristic's per-play scoring at strength-2 settings, evaluated against the
        *pre-rollout* player position so the value reflects the candidate move
        itself rather than the rolled-out state.
        """
        player = clone.get_player(player_id)
        exp_stress = ME.expected_basic_value(player)
        exp_speed = ME.play_speed(cards, exp_stress)
        result = ME.evaluate_move(
            clone,
            player_id,
            expected_speed=exp_speed,
            heat_spent=gear[1],
            from_position=player.turn_start_position,
            from_lap=player.turn_start_lap,
            planned_gear=gear[0],
            heat_price=ME.DEFAULT_HEAT_PRICE,
            horizon_corners=1,
            opponent_aware=True,
            blocking=False,
            enable_solvency=True,
        )
        return result.value

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def _plan_turn(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[Card, ...]]:
        """Search the candidate plans and return the best ``(gear, cards)``.

        Deterministic: candidates are scored in a fixed (prior-ranked) order and
        ties are broken by first-seen order, so the same state + seed always
        yields the same plan (no RNG in the selection).

        S2 adds two tractability levers, both of which preserve the unbudgeted /
        unpruned argmax on the S1 default config (``top_k=None``,
        ``sim_budget=None``):

          * **top-k branching control** -- candidates are ranked by the fast
            ``_move_eval`` prior; only the best ``top_k`` are rolled out.
          * **sim budget** -- once ``sim_budget`` rollout clones have been spent
            this move, scoring stops. Because candidates are scored in
            prior-descending order, a budget cut only ever drops the
            least-promising (by prior) lines.

        Per-move clones and wall time are recorded into :attr:`profile`.
        """
        candidates = self._candidate_plans(state, player_id, legal_gears)
        if not candidates:
            # Degenerate fallback: cheapest legal gear with any legal play.
            gear = legal_gears[0]
            plays = rules.legal_card_plays(state.get_player(player_id).hand, gear[0])
            self.profile.record_move(0, 0.0)
            return gear, plays[0]

        sig = self._turn_signature(state, player_id)
        turn_seed = self._turn_seed(sig)

        # Rank by the fast prior (and truncate to top_k); ordering also makes the
        # budget cut prior-aware. The original enumeration index is the score's
        # plan_index so the per-candidate reseed stream is stable regardless of
        # the prior re-ordering.
        ordered = self._prune_top_k(state, player_id, candidates)
        index_of = {(gear, cards): i for i, (gear, cards) in enumerate(candidates)}

        clones_per_score = self.n_determinizations
        t0 = time.perf_counter()
        clones_spent = 0
        best_plan = ordered[0]
        best_value = float("-inf")
        for gear, cards in ordered:
            if (
                self.sim_budget is not None
                and clones_spent + clones_per_score > self.sim_budget
                and clones_spent > 0
            ):
                # Out of budget; keep the best found so far. (Always score at
                # least one candidate so a tiny budget still returns a real plan.)
                break
            plan_index = index_of[(gear, cards)]
            value = self._score_plan(
                state, player_id, gear, cards, turn_seed, plan_index
            )
            clones_spent += clones_per_score
            if value > best_value:
                best_value = value
                best_plan = (gear, cards)
        self.profile.record_move(clones_spent, time.perf_counter() - t0)
        return best_plan

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

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        self._ensure_plan(state, player_id, legal_gears)
        gear = self._plan_gear
        if gear is None or gear not in legal_gears:
            # Recompute against the handed legal set (state may have shifted).
            gear, cards = self._plan_turn(state, player_id, legal_gears)
            self._plan_sig = self._turn_signature(state, player_id)
            self._plan_gear = gear
            self._plan_cards = cards
        return gear

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

        # Cache miss / stale: recompute the plan at the current gear and return
        # its play (re-validated). Recomputing the full joint plan keeps the
        # gear/cards consistent with what the search actually preferred.
        legal_gears = rules.legal_gear_shifts(
            state.get_player(player_id).gear,
            state.get_player(player_id).heat_available,
        )
        # The gear is already committed at this point; restrict the search to it.
        committed_gear = state.get_player(player_id).gear
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
        return self.rollout_policy.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )

    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        return self.rollout_policy.choose_slipstream(state, player_id)

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        return self.rollout_policy.choose_discard(state, player_id, discardable)
