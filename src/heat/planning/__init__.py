"""Option B planning package -- offline whole-track DP over a reduced HEAT MDP.

Sprint B0 (this package's first contents) is a *feasibility spike*: it builds the
stochastic resource model (``resource_model``) and a reduced one-turn simulator
(``reduced_model``) and -- in ``experiments/spike_reduced_model.py`` -- measures
how faithfully they predict the true engine's per-turn outcomes. Nothing here
solves the DP or drives a game yet; B1/B2 depend on B0's go/no-go verdict.

See ``docs/solo-speed-planning/option-B-track-dp/`` for the design.
"""

from __future__ import annotations

from heat.planning.resource_model import SpeedResourceModel
from heat.planning.reduced_model import (
    ReducedAction,
    ReducedState,
    StepOutcome,
    step_reduced,
)

__all__ = [
    "SpeedResourceModel",
    "ReducedState",
    "ReducedAction",
    "StepOutcome",
    "step_reduced",
]
