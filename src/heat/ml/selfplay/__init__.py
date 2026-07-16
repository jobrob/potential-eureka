"""Custom PPO self-play training spine for the HEAT RL layer (Sprint A0).

This package is the Direction-A substrate: a minimal, transparent PPO training
loop we *own* (rather than extending SB3 ``MaskablePPO``), so later sprints can
add an N-seat shared-policy rollout (A2), a variable-length legal-action
dot-product head (A3), and a structured hidden-info encoder (A4) by swapping
*internals* behind a stable policy interface -- without fighting SB3's fixed
``Discrete`` head or its ``.learn()`` rollout contract.

A0 is plumbing + correctness, NOT architecture. It deliberately reuses the
existing flat observation (:mod:`heat.ml.features`), the existing masking
(:func:`heat.ml.action_codec.legal_action_mask`), and the existing single-seat
:class:`heat.ml.env.HeatEnv`, so the only new thing under test is the training
loop itself. The SB3 path (``heat.ml.model`` / ``heat.ml.training``) is left
untouched -- A0 is purely additive, retaining a known-good baseline.

Modules
-------
* :mod:`~heat.ml.selfplay.policy` -- :class:`HeatPolicy`, the swappable
  ``act`` / ``evaluate`` interface (A0 = masked-categorical MLP).
* :mod:`~heat.ml.selfplay.buffer` -- :class:`RolloutBuffer` + GAE.
* :mod:`~heat.ml.selfplay.ppo` -- the clipped-surrogate PPO update + ``train``
  loop, plus :class:`A0Config`.
"""

from __future__ import annotations

from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.checkpoint import (
    CheckpointMismatchError,
    load_policy,
    save_policy,
)
from heat.ml.selfplay.eval_harness import (
    EvalCell,
    EvalReport,
    evaluate_policy,
    evaluate_vs_anchor,
    held_out_tracks,
)
from heat.ml.selfplay.multiseat import MultiSeatCollector, train_multiseat
from heat.ml.selfplay.phase1 import (
    A8Config,
    SeatCountSchedule,
    train_selfplay_a8,
    training_track_source,
)
from heat.ml.selfplay.policy import (
    DotProductPolicy,
    HeatPolicy,
    StructuredObservationEncoder,
    build_policy,
)
from heat.ml.selfplay.ppo import A0Config, train
from heat.ml.selfplay.recipe import (
    A5Config,
    EntropyController,
    Stage1ValidationError,
    train_selfplay_a5,
)
from heat.ml.selfplay.snapshots import SnapshotAgent, SnapshotPool

__all__ = [
    "A0Config",
    "A5Config",
    "A8Config",
    "CheckpointMismatchError",
    "DotProductPolicy",
    "EntropyController",
    "EvalCell",
    "EvalReport",
    "HeatPolicy",
    "MultiSeatCollector",
    "RolloutBuffer",
    "SnapshotAgent",
    "SnapshotPool",
    "SeatCountSchedule",
    "Stage1ValidationError",
    "StructuredObservationEncoder",
    "build_policy",
    "evaluate_policy",
    "evaluate_vs_anchor",
    "held_out_tracks",
    "load_policy",
    "save_policy",
    "train",
    "train_multiseat",
    "train_selfplay_a5",
    "train_selfplay_a8",
    "training_track_source",
]
