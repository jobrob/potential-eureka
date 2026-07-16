"""KL-to-BC anti-forgetting regularizer for the Sprint S4 BC -> PPO fine-tune.

The S4 fine-tune warm-starts PPO from the behavior-cloned (or DAgger) checkpoint
and runs the opponent ramp. The risk the design flags (the "8C bug-2" pattern) is
**catastrophic forgetting of corner discipline** during the strong phase: PPO can
walk the actor away from the clean limit-1 recovery BC/DAgger painstakingly
distilled, chasing short-term win-rate against hard opponents.

The prescribed lever (doc §4 S4 risk note) is a **KL-to-BC regularizer on the
actor** -- NOT a frozen critic, because the BC checkpoint's value head is
uninitialized (BC supervised only the actor). We add, to PPO's per-minibatch loss,

    L_kl = coef * mean_b KL( pi_theta(. | s_b)  ||  pi_BC(. | s_b) )

where ``pi_BC`` is the frozen reference policy loaded from the warm-start
checkpoint and ``pi_theta`` is the policy being fine-tuned. Both distributions are
the *masked* categorical (illegal actions get -inf logit), evaluated on the
rollout observations + the rollout's stored action masks, so the penalty is
defined over exactly the legal action set the agent actually faces. As ``coef``
rises the fine-tuned actor is pulled back toward the BC behavior; ``coef = 0``
(the default) is a no-op and the run is byte-for-byte standard MaskablePPO.

Why a class-level patch (and why that is safe)
----------------------------------------------
The warm-start path loads the model via ``MaskablePPO.load`` (in
``training._load_model_with_gamma``), which reconstructs a plain ``MaskablePPO`` --
not a subclass we control. Rather than fork the load path, :func:`set_kl_to_bc`
installs a wrapped ``train`` on the ``MaskablePPO`` class that augments the loss
with the KL term, reading its config from a module global. The wrap is idempotent
(installed once) and delegates to the original ``train`` whenever the regularizer
is disabled, so importing this module changes nothing until :func:`set_kl_to_bc`
is called with ``coef > 0``. :func:`clear_kl_to_bc` restores the no-op state
(used by tests).

This module is OPT-IN: nothing imports it unless ``finetune_ppo.py`` is invoked
with ``--kl-to-bc-coef > 0``. It is the doc's fallback-adjacent anti-forgetting
lever, kept separate from the proven warm-start code so the default path has zero
dependency on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.distributions import Categorical

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from heat.ml.model import resolve_device


@dataclass
class _KLState:
    """Process-global KL-to-BC configuration (None reference => disabled)."""

    reference: MaskablePPO | None = None
    coef: float = 0.0


_STATE = _KLState()
#: Set once the class-level ``train`` wrapper has been installed.
_PATCHED = False
#: The original (unwrapped) ``MaskablePPO.train``, captured at patch time.
_ORIG_TRAIN: Callable[[MaskablePPO], None] | None = None


def set_kl_to_bc(
    *, reference_path: str, coef: float, device: str = "auto"
) -> None:
    """Enable the KL-to-BC regularizer with a frozen reference at ``reference_path``.

    Loads the reference (BC/DAgger) checkpoint, freezes it (eval mode, no grad),
    and installs the class-level ``train`` wrapper if not already installed. A
    ``coef <= 0`` disables the penalty without unloading the reference.
    """
    if coef <= 0.0:
        _STATE.reference = None
        _STATE.coef = 0.0
        return
    ref = MaskablePPO.load(reference_path, device=resolve_device(device))
    ref.policy.set_training_mode(False)
    for p in ref.policy.parameters():
        p.requires_grad_(False)
    _STATE.reference = ref
    _STATE.coef = float(coef)
    _install_patch()


def clear_kl_to_bc() -> None:
    """Disable the regularizer (the reference is dropped; the wrapper stays a no-op)."""
    _STATE.reference = None
    _STATE.coef = 0.0


def _reference_masked_logprobs(
    obs: th.Tensor, action_masks: np.ndarray
) -> th.Tensor:
    """Log-probabilities of the FROZEN reference policy over the masked actions.

    Returns a ``(B, ACTION_DIM)`` tensor of ``log pi_BC(a | s)`` with illegal
    actions at ``-inf`` (the masked categorical), on the same device as ``obs``.
    """
    ref = _STATE.reference
    assert ref is not None  # guarded by the caller
    with th.no_grad():
        dist = ref.policy.get_distribution(obs, action_masks=action_masks)
        # MaskableCategorical exposes the (masked) per-action logits; log-softmax
        # over them is the masked log-prob vector.
        logits = cast(Categorical, dist.distribution).logits
        return F.log_softmax(logits, dim=1)


def _kl_to_reference(
    policy: MaskableActorCriticPolicy,
    obs: th.Tensor,
    action_masks: np.ndarray,
) -> th.Tensor:
    """``mean_b KL( pi_theta(.|s_b) || pi_BC(.|s_b) )`` over the masked actions.

    Both distributions are the masked categorical. KL is computed in the standard
    direction (current || reference) so the penalty grows as the fine-tuned policy
    moves probability mass onto actions the BC reference deemed unlikely. Masked
    (``-inf``) actions contribute ``0`` (``p=0``), so the sum is over the legal set
    only; we guard the ``0 * -inf`` with ``nan_to_num``.
    """
    cur = policy.get_distribution(obs, action_masks=action_masks)
    cur_logp = F.log_softmax(
        cast(Categorical, cur.distribution).logits,
        dim=1,
    )
    ref_logp = _reference_masked_logprobs(obs, action_masks)
    p = cur_logp.exp()
    kl = p * (cur_logp - ref_logp)
    kl = th.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
    return kl.sum(dim=1).mean()


def _install_patch() -> None:
    """Install the class-level ``train`` wrapper on ``MaskablePPO`` (idempotent)."""
    global _PATCHED, _ORIG_TRAIN
    if _PATCHED:
        return
    _ORIG_TRAIN = MaskablePPO.train

    original_train = _ORIG_TRAIN

    def _train_with_kl(self: MaskablePPO) -> None:
        if _STATE.reference is None or _STATE.coef <= 0.0:
            # Disabled: behave exactly like stock MaskablePPO.
            return original_train(self)
        _train_loop_with_kl(self, _STATE.reference, _STATE.coef)

    MaskablePPO.train = _train_with_kl  # type: ignore[method-assign]
    _PATCHED = True


def _train_loop_with_kl(
    self: MaskablePPO, reference: MaskablePPO, coef: float
) -> None:
    """MaskablePPO.train with a KL-to-reference term added to the minibatch loss.

    A faithful copy of sb3-contrib 2.9's ``MaskablePPO.train`` (the standard
    clipped-surrogate + value + entropy loss) with one addition: ``coef * KL(pi ||
    pi_BC)`` over the rollout's masked action set, so the fine-tuned actor is
    regularized toward the BC reference. The reference is on the same device as the
    policy; observations/masks come straight from the rollout buffer.
    """
    from gymnasium import spaces

    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_schedule = cast(Callable[[float], float], self.clip_range)
    clip_range = clip_schedule(self._current_progress_remaining)
    clip_range_vf = None
    if self.clip_range_vf is not None:
        clip_vf_schedule = cast(Callable[[float], float], self.clip_range_vf)
        clip_range_vf = clip_vf_schedule(self._current_progress_remaining)

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []
    kl_to_bc_losses = []
    continue_training = True

    for epoch in range(self.n_epochs):
        approx_kl_divs = []
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations,
                actions,
                action_masks=rollout_data.action_masks,
            )
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(
                ratio, 1 - clip_range, 1 + clip_range
            )
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            pg_losses.append(policy_loss.item())
            clip_fraction = th.mean(
                (th.abs(ratio - 1) > clip_range).float()
            ).item()
            clip_fractions.append(clip_fraction)

            if self.clip_range_vf is None:
                values_pred = values
            else:
                assert clip_range_vf is not None
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values,
                    -clip_range_vf,
                    clip_range_vf,
                )
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            entropy_losses.append(entropy_loss.item())

            # --- The S4 addition: KL-to-BC anti-forgetting penalty. ---
            kl_bc = _kl_to_reference(
                self.policy,
                rollout_data.observations,
                cast(np.ndarray, rollout_data.action_masks),
            )
            kl_to_bc_losses.append(kl_bc.item())

            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
                + coef * kl_bc
            )

            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_div = (
                    th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                )
                approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                break

            self.policy.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += 1
        if not continue_training:
            break

    from stable_baselines3.common.utils import explained_variance

    explained_var = explained_variance(
        self.rollout_buffer.values.flatten(),
        self.rollout_buffer.returns.flatten(),
    )
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_to_bc_loss", np.mean(kl_to_bc_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
        self.logger.record("train/clip_range_vf", clip_range_vf)
