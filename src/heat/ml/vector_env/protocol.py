"""Implementation-neutral batched environment boundary for Direction D1+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, order=True)
class ReadyDecision:
    """Stable identity for one seat decision awaiting a policy action."""

    game_id: int
    seat_id: int
    decision_index: int


@dataclass(frozen=True)
class PolicyBatch:
    """Ordered policy inputs for a set of ready decision identities."""

    decisions: tuple[ReadyDecision, ...]
    observations: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]


@dataclass(frozen=True)
class CompletedTransition:
    """One rewarded sample completed without crossing a policy version."""

    decision: ReadyDecision
    action: int
    reward: float
    done: bool
    successor_observation: NDArray[np.float32] | None
    successor_legal_mask: NDArray[np.bool_] | None
    policy_version: int


class BatchedEnvironment(Protocol):
    """Minimum exact lane substrate consumed by the future D1 collector."""

    def reset(
        self,
        game_ids: tuple[int, ...],
        seeds: tuple[int, ...],
        seat_counts: tuple[int, ...],
    ) -> tuple[ReadyDecision, ...]:
        """Create bounded lanes and return their first ready decisions."""
        ...

    def ready(self) -> tuple[ReadyDecision, ...]:
        """Return every currently ready identity in deterministic order."""
        ...

    def finished_game_ids(self) -> tuple[int, ...]:
        """Return terminal lane identities awaiting replacement."""
        ...

    def replace_finished(
        self,
        game_ids: tuple[int, ...],
        seeds: tuple[int, ...],
        seat_counts: tuple[int, ...],
    ) -> tuple[ReadyDecision, ...]:
        """Replace every finished lane with one newly identified game."""
        ...

    def observe(self, decisions: tuple[ReadyDecision, ...]) -> PolicyBatch:
        """Encode partial observations and exact legal masks for ready rows."""
        ...

    def step(
        self,
        decisions: tuple[ReadyDecision, ...],
        actions: NDArray[np.int64],
        *,
        policy_version: int,
    ) -> tuple[CompletedTransition, ...]:
        """Apply rows once and return any transitions whose rewards are known."""
        ...

    def drain(self, *, policy_version: int) -> tuple[CompletedTransition, ...]:
        """Reach a bounded safe boundary without admitting new policy work."""
        ...

    def semantic_snapshot(self, game_id: int) -> dict[str, object]:
        """Return a canonical state comparable with the scalar engine oracle."""
        ...
