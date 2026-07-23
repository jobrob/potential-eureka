"""Shared results and semantic event receipts for Direction D2 kernels."""

from __future__ import annotations

from dataclasses import dataclass

from heat.ml.vector_env.observations import TensorDecisionBatch
from heat.ml.vector_env.state import TensorGameState


@dataclass(frozen=True)
class TensorKernelEvent:
    """One legacy-comparable event tagged with its stable tensor game identity."""

    game_id: int
    round_num: int
    phase: str
    player_id: int | None
    event_type: str
    data: dict[str, object]

    def semantic(self) -> dict[str, object]:
        """Return the exact shape produced by D0's ``canonical_event``."""
        return {
            "round_num": self.round_num,
            "phase": self.phase,
            "player_id": self.player_id,
            "event_type": self.event_type,
            "data": self.data,
        }


@dataclass(frozen=True)
class TensorKernelResult:
    """Updated tensor state, emitted events, and next policy decisions."""

    state: TensorGameState
    events: tuple[TensorKernelEvent, ...]
    next_decisions: TensorDecisionBatch
