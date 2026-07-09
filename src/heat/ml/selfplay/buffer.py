"""Rollout buffer + GAE for the custom PPO loop (Sprint A0).

:class:`RolloutBuffer` accumulates one rollout of single-seat transitions
(``obs, action, logp, value, reward, done, mask``) collected by driving
:class:`heat.ml.env.HeatEnv`, then computes Generalized Advantage Estimation
(GAE-lambda) advantages and the corresponding value targets (returns).

GAE is the single most bug-prone part of a from-scratch PPO (sign of the
TD-residual, where the ``done`` mask cuts the bootstrap, and returns =
advantages + values). It is isolated here, kept tiny and explicit, so the A0
sanity gate (G2) and the smoke test can exercise it in isolation. The buffer is
deliberately flat/single-stream; the A2 multi-seat collector will produce one
per-seat stream each and reuse this same advantage math.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


class RolloutBuffer:
    """Fixed-capacity store of one rollout's transitions, with GAE.

    Each :meth:`add` appends one environment step. After ``capacity`` steps,
    call :meth:`compute_gae` with the bootstrap value of the state *following*
    the last stored step, then read the batched tensors via :meth:`get`.

    Args:
        capacity: number of steps the buffer holds (the rollout length).
        obs_dim: observation width per step.
        action_dim: action-space width (mask width) per step.
        device: torch device the batched tensors in :meth:`get` are placed on.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.logps = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=bool)

        # Filled by compute_gae().
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._gae_done = False

    def __len__(self) -> int:
        return self._pos

    @property
    def full(self) -> bool:
        """Whether the buffer has stored ``capacity`` steps."""
        return self._pos >= self.capacity

    def reset(self) -> None:
        """Drop all stored transitions so the buffer can collect a new rollout."""
        self._pos = 0
        self._gae_done = False

    def add(
        self,
        obs: NDArray[np.float32],
        action: int,
        logp: float,
        value: float,
        reward: float,
        done: bool,
        mask: NDArray[np.bool_],
    ) -> None:
        """Append one transition. Raises if the buffer is already full."""
        if self._pos >= self.capacity:
            raise RuntimeError("RolloutBuffer is full; call compute_gae()/reset()")
        i = self._pos
        self.obs[i] = obs
        self.actions[i] = action
        self.logps[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.masks[i] = mask
        self._pos += 1

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        """Fill ``advantages`` and ``returns`` via GAE-lambda.

        Walks the stored steps in reverse, accumulating the TD-residual
        ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)`` into
        ``A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}``. The
        ``(1 - done_t)`` factor cuts both the value bootstrap and the advantage
        recursion at episode boundaries, so credit never leaks across episodes.
        Value targets (returns) are ``A_t + V(s_t)`` -- the standard PPO target
        for the value-function regression.

        Args:
            last_value: bootstrap value ``V(s_T)`` of the state following the
                last stored step (0.0 if that step ended an episode).
            gamma: discount factor.
            gae_lambda: GAE smoothing coefficient in ``[0, 1]``.
        """
        n = self._pos
        last_gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_value * nonterminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]
        self._gae_done = True

    def get(self) -> dict[str, torch.Tensor]:
        """Return the stored rollout as a dict of batched tensors on ``device``.

        Keys: ``obs`` ``(N, OBS_DIM)``, ``actions`` ``(N,)`` long, ``logps``
        ``(N,)``, ``values`` ``(N,)``, ``advantages`` ``(N,)``, ``returns``
        ``(N,)``, ``masks`` ``(N, ACTION_DIM)`` bool. Requires
        :meth:`compute_gae` to have been called first.
        """
        if not self._gae_done:
            raise RuntimeError("call compute_gae() before get()")
        n = self._pos
        dev = self.device
        return {
            "obs": torch.as_tensor(self.obs[:n], device=dev),
            "actions": torch.as_tensor(self.actions[:n], device=dev),
            "logps": torch.as_tensor(self.logps[:n], device=dev),
            "values": torch.as_tensor(self.values[:n], device=dev),
            "advantages": torch.as_tensor(self.advantages[:n], device=dev),
            "returns": torch.as_tensor(self.returns[:n], device=dev),
            "masks": torch.as_tensor(self.masks[:n], device=dev),
        }
