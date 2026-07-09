"""Swappable policy interface for the custom PPO loop (Sprint A0).

:class:`HeatPolicy` is the contract the trainer (:mod:`heat.ml.selfplay.ppo`)
codes against; A3/A4 re-implement only its *internals* (the trunk, the action
head, the encoder) while keeping the ``act`` / ``evaluate`` signatures and return
tuples below frozen. That stability is the whole point of A0: the loop never
needs to change when the architecture does.

A0 implementation
-----------------
A small MLP trunk over the flat ``OBS_DIM`` observation produces (a) policy
logits over the fixed ``ACTION_DIM`` union action space and (b) a value scalar.
Legality is enforced by ``logits.masked_fill(~mask, -inf)`` before building a
:class:`torch.distributions.Categorical`, so **no illegal action can ever
receive probability mass** (the smoke test asserts this against
:func:`heat.ml.action_codec.legal_action_mask`).

When A3 lands, ``mask`` generalizes from "the boolean legal-action mask over a
fixed head" to "the set of currently-legal action feature vectors", and the head
becomes a dot-product over that variable set -- but ``act`` / ``evaluate`` keep
exactly these shapes and return tuples, so the trainer is untouched.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from heat.ml.spaces import ACTION_DIM, OBS_DIM


class HeatPolicy(nn.Module):
    """Masked-categorical actor-critic over the flat HEAT observation (A0).

    A shared MLP trunk feeds two linear heads: a policy head emitting one logit
    per action in the fixed ``ACTION_DIM`` union space, and a value head emitting
    a single state-value scalar. The trainer calls :meth:`act` during rollout
    collection and :meth:`evaluate` during the PPO update; both apply the legal-
    action ``mask`` so illegal actions never receive probability mass.

    Args:
        obs_dim: observation width (defaults to the frozen ``OBS_DIM``).
        action_dim: action-space width (defaults to the frozen ``ACTION_DIM``).
        hidden_sizes: trunk hidden-layer widths. A small default keeps the smoke
            test fast; real runs pass a larger trunk.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    # ------------------------------------------------------------------
    # Internal: build the masked action distribution + value for a batch.
    # ------------------------------------------------------------------

    def _distribution_and_value(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Return ``(masked Categorical, value (B,))`` for a batch of obs.

        Illegal logits are driven to ``-inf`` before the softmax so the
        resulting distribution places exactly zero mass on them. ``mask`` is a
        boolean tensor where ``True`` marks a *legal* action.
        """
        latent = self.trunk(obs)
        logits = self.policy_head(latent)
        # masked_fill(~mask, -inf): illegal actions get -inf logits -> 0 prob.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, neg_inf)
        dist = Categorical(logits=masked_logits)
        value = self.value_head(latent).squeeze(-1)
        return dist, value

    # ------------------------------------------------------------------
    # Frozen interface (A3/A4 swap internals only).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for each obs in the batch (rollout collection).

        Args:
            obs: ``(B, OBS_DIM)`` float32 observations.
            mask: ``(B, ACTION_DIM)`` bool legal-action mask (``True`` == legal).

        Returns:
            ``(action, logp, value, entropy)`` where ``action`` is ``(B,)`` long,
            and ``logp``, ``value``, ``entropy`` are each ``(B,)`` float.
        """
        dist, value = self._distribution_and_value(obs, mask)
        action = dist.sample()  # type: ignore[no-untyped-call]
        logp = dist.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return action, logp, value, entropy

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score given ``actions`` under the current policy (PPO update).

        Unlike :meth:`act`, this keeps the graph so gradients flow through the
        returned tensors.

        Args:
            obs: ``(B, OBS_DIM)`` float32 observations.
            actions: ``(B,)`` long actions to score.
            mask: ``(B, ACTION_DIM)`` bool legal-action mask (``True`` == legal).

        Returns:
            ``(logp, value, entropy)``, each ``(B,)`` float.
        """
        dist, value = self._distribution_and_value(obs, mask)
        logp = dist.log_prob(actions)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return logp, value, entropy
