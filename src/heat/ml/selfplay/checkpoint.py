"""Minimal on-disk checkpoint for Direction-A self-play policies (Sprint A7).

Direction-A policies (:class:`heat.ml.selfplay.policy.HeatPolicy` /
:class:`~heat.ml.selfplay.policy.DotProductPolicy`) currently live only in
memory, so a trained run cannot be scored later. A7 adds the smallest possible
checkpoint that makes a policy *evaluable*:

* :func:`save_policy` writes a single ``torch.save`` blob carrying the policy
  ``state_dict`` plus the four numbers needed to rebuild it (``head``,
  ``hidden_sizes``, ``obs_dim``, ``action_dim``) and a ``codec_version``
  tripwire.
* :func:`load_policy` rebuilds the policy via the same
  :func:`~heat.ml.selfplay.policy.build_policy` factory the trainer uses and
  **fails fast** (:class:`CheckpointMismatchError`) if the checkpoint's
  obs/action/codec contract does not match the live :mod:`heat.ml.spaces`
  contract -- mirroring :class:`heat.agents.ml_agent.MLAgent`'s §3.4 meta
  tripwire in a single guard, with no ``.meta.json`` sidecar.

This is deliberately tiny -- it is NOT the SB3 sidecar system
(:func:`heat.ml.training.save_checkpoint`); that path stays for MLAgent ``.zip``
checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch

from heat.ml.selfplay.policy import PPOPolicy, build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint does not match the live ML contract.

    Signals that the policy on disk was built against a different
    ``OBS_DIM`` / ``ACTION_DIM`` / ``CODEC_VERSION`` than the code now running,
    so its outputs cannot be trusted. Fail fast rather than load a stale policy
    that would silently emit garbage actions (the :class:`heat.agents.ml_agent.MLAgent`
    §3.4 tripwire lesson, applied to the tiny Direction-A checkpoint).
    """


def save_policy(policy: PPOPolicy, config: A0Config, path: str | Path) -> None:
    """Write ``policy`` to ``path`` with the contract fields needed to reload it.

    The blob is a plain dict (``torch.save``) with the policy ``state_dict`` plus
    the head/trunk/obs/action dims and the live ``CODEC_VERSION``. The head and
    trunk widths come from ``config`` (they define how :func:`build_policy`
    reconstructs the network); the obs/action dims are read off the policy so a
    round-trip is self-describing.

    Args:
        policy: the trained :class:`HeatPolicy` / :class:`DotProductPolicy`.
        config: the run config (only ``head`` and ``hidden_sizes`` are stored).
        path: destination file path.
    """
    blob = {
        "state_dict": policy.state_dict(),
        "head": config.head,
        "hidden_sizes": list(config.hidden_sizes),
        "obs_dim": int(policy.obs_dim),
        "action_dim": int(policy.action_dim),
        "codec_version": CODEC_VERSION,
    }
    torch.save(blob, str(path))


def load_policy(path: str | Path) -> PPOPolicy:
    """Rebuild the policy saved at ``path`` and return it in ``eval()`` mode.

    Reconstructs the network with :func:`build_policy` (the same factory the
    trainer uses) from the stored ``head`` / ``hidden_sizes``, then loads the
    ``state_dict``. Before touching any weights it asserts the checkpoint's
    obs/action/codec contract matches the live :mod:`heat.ml.spaces` contract,
    raising :class:`CheckpointMismatchError` on any drift.

    Raises:
        CheckpointMismatchError: on any ``obs_dim`` / ``action_dim`` /
            ``codec_version`` mismatch with the running contract.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)

    expected = {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "codec_version": CODEC_VERSION,
    }
    mismatches = [
        f"{key}: checkpoint={blob.get(key)!r} != runtime={want!r}"
        for key, want in expected.items()
        if blob.get(key) != want
    ]
    if mismatches:
        raise CheckpointMismatchError(
            f"Checkpoint {str(path)!r} is incompatible with the current ML "
            "contract (stale policy vs drifted codec): " + "; ".join(mismatches)
        )

    config = A0Config(
        head=str(blob["head"]),
        hidden_sizes=tuple(int(h) for h in blob["hidden_sizes"]),
    )
    policy = build_policy(config)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy
