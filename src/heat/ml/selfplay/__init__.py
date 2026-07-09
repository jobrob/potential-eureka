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
from heat.ml.selfplay.multiseat import MultiSeatCollector, train_multiseat
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.ppo import A0Config, train

__all__ = [
    "A0Config",
    "HeatPolicy",
    "MultiSeatCollector",
    "RolloutBuffer",
    "train",
    "train_multiseat",
]
