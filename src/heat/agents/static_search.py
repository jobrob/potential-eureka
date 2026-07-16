"""Reproducible simulator-backed candidate investigated for A8 evaluation."""

from __future__ import annotations

from heat.agents.search_agent import LookaheadAgent


class StaticSearchAgent(LookaheadAgent):
    """Versioned static-search candidate selected by the A8 H2 investigation.

    B1 did not promote this candidate as the strong default because it failed
    the homogeneous weak-field strength gate. The configuration remains frozen
    and reproducible for explicit research comparisons.
    """

    VERSION = "StaticSearchV1"

    def __init__(self, name: str = VERSION) -> None:
        """Build the evaluated one-round, determinized, top-six search policy."""
        super().__init__(
            name=name,
            horizon=1,
            n_determinizations=2,
            determinize_hidden=True,
            top_k=6,
            leaf_value="progress",
        )
