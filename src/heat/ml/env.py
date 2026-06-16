"""Gymnasium environment for the HEAT engine (Sprint 5b).

:class:`HeatEnv` exposes one HEAT race as a single-agent RL problem: a single
*learning seat* plays against a pool of supplied opponent policies. Each RL
``step`` corresponds to exactly one :class:`~heat.engine.driver.Decision` for
the learning seat; the env auto-advances every opponent decision (and every
forced/degenerate learner decision) internally.

Backbone
--------
The env drives :func:`heat.engine.driver.run_round_driver` — a generator that
yields a ``Decision`` at each agent decision point and resumes via
``gen.send(action)``. ``run_round_driver`` runs exactly ONE round and advances
``state.round_num`` itself; the env starts a fresh generator each time the
current one raises ``StopIteration``.

Contract (frozen in :mod:`heat.ml.spaces`, consumed via :mod:`heat.ml.features`
and :mod:`heat.ml.action_codec`):

* Observation: ``Box(-1, 1, (OBS_DIM,), float32)`` from ``encode_observation``.
* Action: a single flattened ``Discrete(ACTION_DIM)``; the legal subset for the
  current decision is exposed via :meth:`action_masks` (the sb3-contrib
  MaskablePPO hook) and stashed in ``info["action_mask"]``.
* Reward: ``step_reward(prev, curr, learner_id, done)``.

Forced / all-False decisions
----------------------------
Some learner decisions are degenerate: e.g. a literally empty hand yields a
CARDS decision whose only legal play is ``()`` (``legal_action_mask`` all-False),
or a single forced gear. Exposing an all-False mask would crash MaskablePPO and
exposing a one-hot mask wastes an RL step on a non-choice. The env therefore
**auto-resolves** any learner decision with <= 1 legal action by sending the
single forced engine action (``decision.legal[0]``) and continuing to advance,
never surfacing it as an RL step.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.tracks.loader import load_track_by_name
from heat.ml import spaces
from heat.ml.action_codec import (
    _play_to_multiset,
    decode_action,
    legal_action_mask,
)
from heat.ml.features import encode_observation
from heat.ml.opponents import opponent_action
from heat.ml.spaces import ACTION_DIM, OBS_DIM, step_reward

#: Opponents may be supplied as ready BaseAgent instances or as zero-arg
#: factories that build one (factories let env copies hold independent RNG
#: state — useful for vectorized envs).
OpponentSpec = BaseAgent | Callable[[], BaseAgent]

#: Sentinel distinguishing "no forced action" from a legitimate forced action
#: whose value is falsy (e.g. SLIPSTREAM ``False`` or the empty CARDS play
#: ``()``), so ``_forced_action`` can return those without ambiguity.
_NO_FORCED = object()


def _default_track() -> Track:
    """Load the default training track (USA), falling back to any track found."""
    return load_track_by_name("usa")


class HeatEnv(gym.Env):
    """Single-learning-seat Gymnasium environment over the HEAT engine.

    Args:
        track: the race track. Defaults to the USA track if omitted.
        num_players: total seats (learner + opponents). Must be >= 2 and
            <= ``spaces.MAX_PLAYERS``.
        opponents: policies for the non-learner seats. Either a single
            :class:`BaseAgent`/factory (reused for all opponents) or a sequence
            of length ``num_players - 1``. Each entry may be a ``BaseAgent`` or
            a zero-arg factory returning one. Defaults to ``HeuristicAgent``.
        learner_id: the seat the RL policy controls (default 0).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: Track | None = None,
        num_players: int = 4,
        opponents: OpponentSpec | Sequence[OpponentSpec] | None = None,
        learner_id: int = 0,
    ) -> None:
        super().__init__()

        if not (2 <= num_players <= spaces.MAX_PLAYERS):
            raise ValueError(
                f"num_players must be in [2, {spaces.MAX_PLAYERS}], got {num_players}"
            )
        if not (0 <= learner_id < num_players):
            raise ValueError(
                f"learner_id {learner_id} out of range for {num_players} players"
            )

        self.track: Track = track if track is not None else _default_track()
        self.num_players = num_players
        self.learner_id = learner_id
        self._opponent_specs = self._normalize_opponents(opponents, num_players)

        self.observation_space = spaces.observation_space()
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

        # Live episode state, populated by reset().
        self.state: GameState | None = None
        self._gen = None
        self._decision: Decision | None = None  # current learner decision
        self._opponents: dict[int, BaseAgent] = {}
        self._done = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_opponents(
        opponents: OpponentSpec | Sequence[OpponentSpec] | None,
        num_players: int,
    ) -> list[OpponentSpec]:
        """Resolve the ``opponents`` argument to one spec per opponent seat."""
        n_opp = num_players - 1
        if opponents is None:
            return [HeuristicAgent for _ in range(n_opp)]
        # A single agent/factory broadcasts to all opponent seats.
        if isinstance(opponents, BaseAgent) or callable(opponents):
            return [opponents for _ in range(n_opp)]
        specs = list(opponents)
        if len(specs) != n_opp:
            raise ValueError(
                f"expected {n_opp} opponents for {num_players} players, "
                f"got {len(specs)}"
            )
        return specs

    def _build_opponents(self) -> dict[int, BaseAgent]:
        """Instantiate one opponent ``BaseAgent`` per non-learner seat.

        Factories are called fresh each reset so per-episode RNG (e.g.
        ``RandomAgent``) does not leak across episodes.
        """
        agents: dict[int, BaseAgent] = {}
        opp_iter = iter(self._opponent_specs)
        for pid in range(self.num_players):
            if pid == self.learner_id:
                continue
            spec = next(opp_iter)
            agents[pid] = spec if isinstance(spec, BaseAgent) else spec()
        return agents

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode and advance to the first learner decision.

        Returns ``(obs, info)`` where ``obs`` encodes the learner's view at the
        first decision it must act on (or the terminal state if the race ends
        with no learner decision, which the bounded engine makes effectively
        impossible). ``info`` carries ``action_mask``.
        """
        super().reset(seed=seed)

        self.state = GameState.create(
            self.track, self.num_players, seed=seed
        )
        # Mirror Game.__init__: the race has started, everyone is on lap 1.
        for player in self.state.players:
            player.lap = 1

        self._opponents = self._build_opponents()
        self._done = False
        self._gen = run_round_driver(self.state)
        self._decision = None

        decision = self._advance_to_learner(first_action=None)
        self._decision = decision

        obs = self._encode(decision)
        info = self._info(decision)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the learner's ``action`` and advance to the next decision.

        ``action`` is a flat index into ``Discrete(ACTION_DIM)``; it MUST be
        legal under the current ``action_masks()`` (MaskablePPO guarantees this).
        Returns the gym 5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        if self.state is None or self._gen is None:
            raise RuntimeError("step() called before reset()")
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        if self._decision is None:
            raise RuntimeError("no pending learner decision to act on")

        # Snapshot for the reward delta. clone() is cheap (no event log copy).
        # Pass an explicit reseed so cloning does NOT advance the live game RNG
        # (the default fork draws from self.state.rng); the clone's own RNG is
        # never used here -- we only read scalar player/track state from prev.
        prev = self.state.clone(reseed=0)

        engine_action = self._decode_legal(self._decision, int(action))
        decision = self._advance_to_learner(first_action=engine_action)
        self._decision = decision

        terminated, truncated = self._episode_flags()
        self._done = terminated or truncated

        reward = step_reward(prev, self.state, self.learner_id, self._done)

        obs = self._encode(decision)
        info = self._info(decision)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the legal-action mask for the current decision (MaskablePPO
        hook). Shape ``(ACTION_DIM,)`` bool.

        When the episode is over or there is no pending learner decision, every
        action is reported legal so a sampler never sees an all-False mask.
        """
        if self.state is None or self._decision is None or self._done:
            return np.ones(ACTION_DIM, dtype=bool)
        return legal_action_mask(self._decision, self.state)

    # ------------------------------------------------------------------
    # Internal advancing logic
    # ------------------------------------------------------------------

    def _advance_to_learner(self, first_action: object) -> Decision | None:
        """Drive the round generator until a *real* learner decision is pending
        or the episode ends.

        ``first_action`` is sent into the live generator first (the learner's
        decoded action, or ``None`` to prime a freshly-created generator). All
        opponent decisions, and any forced/degenerate learner decision (<= 1
        legal action), are resolved internally and never returned.

        Returns the pending learner :class:`Decision`, or ``None`` if the
        episode ended (game over / truncation) with no decision to make.
        """
        assert self.state is not None and self._gen is not None
        send_value = first_action

        while True:
            if self._episode_over():
                return None
            try:
                decision = self._gen.send(send_value)
            except StopIteration:
                # One round finished; start the next round's generator. The
                # round counter was advanced inside the generator.
                if self._episode_over():
                    return None
                self._gen = run_round_driver(self.state)
                send_value = None
                continue

            if decision.player_id != self.learner_id:
                # Opponent decision: resolve and keep advancing.
                send_value = opponent_action(
                    self._opponents[decision.player_id], decision, self.state
                )
                continue

            # Learner decision. Auto-resolve degenerate ones (<= 1 legal action
            # or an all-False mask, e.g. empty hand -> CARDS legal == [()]).
            forced = self._forced_action(decision)
            if forced is not _NO_FORCED:
                send_value = forced
                continue

            return decision

    def _decode_legal(self, decision: Decision, flat_index: int) -> object:
        """Decode a flat action to the concrete object the driver expects,
        guaranteeing it is one the driver will accept as legal.

        For CARDS the driver checks ``chosen not in legal_plays`` with *order-
        and identity-sensitive* tuple equality (``rules.legal_card_plays``
        enumerates ``itertools.combinations`` of the actual hand cards in hand
        order). ``decode_action`` realizes a value-multiset to lowest-id
        representatives in token-sorted order, which need not match a legal
        tuple's element ordering or representative choice. We therefore snap the
        decoded play to the legal tuple with the same value-multiset.

        All other kinds round-trip cleanly: GEAR returns the actual
        ``(new_gear, heat_cost)`` tuple from ``decision.legal``; REACT /
        SLIPSTREAM / DISCARD are not subject to a tuple-equality legality guard.
        """
        decoded = decode_action(decision, flat_index, self.state)

        if decision.kind == DecisionKind.CARDS:
            target = _play_to_multiset(decoded)
            for legal_play in decision.legal:
                if _play_to_multiset(legal_play) == target:
                    return legal_play
            # Should be unreachable: the mask only exposes realizable multisets.
            raise ValueError(  # pragma: no cover - defensive
                f"decoded card play {decoded} has no matching legal tuple"
            )

        return decoded

    def _forced_action(self, decision: Decision) -> object:
        """Return the single forced engine action for a degenerate learner
        ``decision``, or :data:`_NO_FORCED` if the decision is a real choice.

        A decision is degenerate when its mask has <= 1 legal action. This
        subsumes the all-False case (empty-hand CARDS, ``legal == [()]``):
        the engine still expects a single forced action, which we send directly
        rather than handing an unusable / one-hot mask to the policy.

        For an all-False mask (a true forced play with no codec index, e.g. the
        empty-hand ``()`` play), send the engine's single legal option directly
        (``decision.legal[0]``). For a one-hot mask, decode the single legal
        flat index so the action is built through the same codec path a real
        step would use.
        """
        mask = legal_action_mask(decision, self.state)
        legal_idx = np.flatnonzero(mask)
        if len(legal_idx) > 1:
            return _NO_FORCED
        if len(legal_idx) == 0:
            # All-False mask: the engine still has exactly one legal option.
            # This only arises for kinds whose ``legal`` is an indexable
            # sequence (CARDS ``[()]`` / GEAR / DISCARD); REACT always has the
            # "do nothing" slot legal, so it never reaches here.
            return decision.legal[0]
        return self._decode_legal(decision, int(legal_idx[0]))

    def _episode_over(self) -> bool:
        """Whether the episode should stop advancing (game over or truncation)."""
        terminated, truncated = self._episode_flags()
        return terminated or truncated

    def _episode_flags(self) -> tuple[bool, bool]:
        """Return ``(terminated, truncated)`` for the current state."""
        assert self.state is not None
        terminated = self.state.is_game_over
        truncated = (not terminated) and (self.state.round_num > MAX_ROUNDS)
        return terminated, truncated

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------

    def _encode(self, decision: Decision | None) -> np.ndarray:
        """Encode the learner's observation at ``decision`` (or terminal)."""
        assert self.state is not None
        return encode_observation(self.state, self.learner_id, decision)

    def _info(self, decision: Decision | None) -> dict[str, Any]:
        """Build the per-step ``info`` dict, including the action mask."""
        if decision is None or self._done:
            mask = np.ones(ACTION_DIM, dtype=bool)
        else:
            mask = legal_action_mask(decision, self.state)
        return {"action_mask": mask}
