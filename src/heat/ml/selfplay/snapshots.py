"""Frozen-policy snapshots for anti-collapse self-play (Sprint A5).

Historical self-play collapse (Sprint 5 Phase 2, 6C/6D leagues, 8C BC->PPO) is
the failure A5 hardens against. One of its two guards (with the entropy floor in
:mod:`heat.ml.selfplay.recipe`) is a *recent-snapshot opponent pool*: rather than
only ever chasing its current self, the live policy occasionally trains against a
frozen copy of a recent version of itself, damping the 84-93% self-chasing
oscillation the A2 probe observed.

This module provides the two pieces that pool needs:

* :class:`SnapshotAgent` -- a frozen, deep-copied policy behind the ordinary
  :class:`heat.agents.base.BaseAgent` pull-method interface, so it drops straight
  into :class:`~heat.ml.selfplay.multiseat.MultiSeatCollector`'s existing
  ``scripted_seats`` hook (the collector, buffer, and trainer are untouched).
* :class:`SnapshotPool` -- a bounded, in-memory deque of recent snapshots with
  uniform sampling and an ``oldest()`` "previous-self" yardstick.

Deliberate deviation (design §2.2): the sprint plan's on-disk, PFSP-weighted
``league.py`` is *not* reused here -- an in-process deque of <=5 frozen policies
sampled uniformly buys everything A5 needs without coupling to checkpoint files.
PFSP + persisted leagues return at A7/A8; :class:`SnapshotAgent` is the adapter
both will reuse.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import cast

import numpy as np
import torch

from heat.agents.base import BaseAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.features import encode_observation
from heat.ml.selfplay.policy import PPOPolicy


class SnapshotAgent(BaseAgent):
    """A frozen policy snapshot behind the :class:`BaseAgent` pull interface.

    Construction **deep-copies** the policy, ``.eval()``s it, and moves it to
    CPU, so a pool entry is immutable as the live network keeps training (a copy
    shares no parameters with the original -- see the A5 immutability test).

    Every ``choose_*`` method reconstructs the :class:`~heat.engine.driver.Decision`
    its arguments came from (matching :func:`heat.ml.opponents.opponent_action`'s
    per-kind ``legal`` payloads exactly -- REACT rebuilds
    :class:`heat.engine.rules.ReactOptions` from its unpacked fields, SLIPSTREAM's
    ``legal`` is ``True``) and resolves it through the *same* legality plumbing the
    collector uses: a forced/degenerate decision auto-resolves via
    :func:`~heat.ml.action_codec.forced_action`; otherwise the state is encoded,
    masked, sampled by the policy, and decoded back to an engine action. No
    meta/tripwire machinery is needed -- an in-process snapshot cannot drift from
    the codec within a single run.

    Args:
        policy: the live policy to freeze. Deep-copied at construction; the
            original is never referenced again.
        name: display name (defaults to ``"Snapshot"``).
    """

    def __init__(self, policy: PPOPolicy, *, name: str = "Snapshot") -> None:
        super().__init__(name=name)
        self._device = torch.device("cpu")
        frozen = copy.deepcopy(policy).to(self._device)
        frozen.eval()
        self._policy: PPOPolicy = frozen

    # ------------------------------------------------------------------
    # Shared act path (mirrors MultiSeatCollector's per-decision handling).
    # ------------------------------------------------------------------

    def _decide(self, decision: Decision, state: GameState) -> object:
        """Resolve ``decision`` to a concrete engine action via the codec path.

        Forced/degenerate decisions are auto-resolved exactly as the collector
        does; a real choice is encoded, masked, sampled (batch size 1, CPU) and
        decoded back to the action the driver's ``send`` expects.
        """
        forced = forced_action(decision, state)
        if forced is not NO_FORCED:
            return forced

        obs = encode_observation(state, decision.player_id, decision)
        mask = legal_action_mask(decision, state)
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        mask_t = torch.as_tensor(
            mask, dtype=torch.bool, device=self._device
        ).unsqueeze(0)
        action_t, _logp, _value, _entropy = self._policy.act(obs_t, mask_t)
        return decode_legal_action(decision, state, int(action_t.item()))

    # ------------------------------------------------------------------
    # BaseAgent interface: rebuild the Decision, delegate to _decide.
    # ------------------------------------------------------------------

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        decision = Decision(DecisionKind.GEAR, player_id, legal_gears)
        return cast("tuple[int, int]", self._decide(decision, state))

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        decision = Decision(DecisionKind.CARDS, player_id, legal_plays)
        return cast("tuple[Card, ...]", self._decide(decision, state))

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        # Rebuild the ReactOptions object opponent_action unpacks into fields.
        opts = rules.ReactOptions(
            max_cooldown=max_cooldown,
            can_boost=can_boost,
            has_adrenaline=has_adrenaline,
        )
        decision = Decision(DecisionKind.REACT, player_id, opts)
        return cast("ReactDecision", self._decide(decision, state))

    def choose_slipstream(self, state: GameState, player_id: int) -> bool:
        # SLIPSTREAM's legal payload is simply True (a decision is only asked
        # when slipstream is eligible).
        decision = Decision(DecisionKind.SLIPSTREAM, player_id, True)
        return cast("bool", self._decide(decision, state))

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        decision = Decision(DecisionKind.DISCARD, player_id, discardable)
        return cast("list[Card]", self._decide(decision, state))


class SnapshotPool:
    """A bounded, in-memory pool of recent frozen-policy snapshots (Sprint A5).

    Holds at most ``capacity`` :class:`SnapshotAgent`s in insertion order; when
    full, :meth:`push` evicts the oldest. Opponents are drawn uniformly by
    :meth:`sample`, and :meth:`oldest` returns the earliest surviving snapshot --
    the "previous self" the A5 gate measures climb against (G2).

    Args:
        capacity: maximum snapshots retained (default 5, per design §4.2).
    """

    def __init__(self, capacity: int = 5) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._entries: deque[SnapshotAgent] = deque(maxlen=capacity)

    def push(self, policy: PPOPolicy, label: str) -> None:
        """Freeze ``policy`` (deep-copied inside :class:`SnapshotAgent`) and add
        it, evicting the oldest snapshot if the pool is at capacity."""
        self._entries.append(SnapshotAgent(policy, name=label))

    def sample(self, rng: np.random.Generator) -> SnapshotAgent:
        """Return a uniformly-sampled snapshot. Raises if the pool is empty."""
        if not self._entries:
            raise IndexError("cannot sample from an empty SnapshotPool")
        idx = int(rng.integers(len(self._entries)))
        return self._entries[idx]

    def oldest(self) -> SnapshotAgent:
        """Return the earliest surviving snapshot. Raises if the pool is empty."""
        if not self._entries:
            raise IndexError("SnapshotPool is empty")
        return self._entries[0]

    def __len__(self) -> int:
        return len(self._entries)
