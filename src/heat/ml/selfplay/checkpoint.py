"""Minimal on-disk checkpoint for Direction-A self-play policies (Sprint A7).

Direction-A policies (:class:`heat.ml.selfplay.policy.HeatPolicy` /
:class:`~heat.ml.selfplay.policy.DotProductPolicy`) currently live only in
memory, so a trained run cannot be scored later. A7 adds the smallest possible
checkpoint that makes a policy *evaluable*:

* :func:`save_policy` writes a single ``torch.save`` blob carrying the policy
  ``state_dict`` plus the fields needed to rebuild it (``head``, ``encoder``,
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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from heat.ml.selfplay.policy import PPOPolicy, build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, LEGACY_PLAYABLE_CODECS, OBS_DIM


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint does not match the live ML contract.

    Signals that the policy on disk was built against a different
    ``OBS_DIM`` / ``ACTION_DIM`` / ``CODEC_VERSION`` than the code now running,
    so its outputs cannot be trusted. Fail fast rather than load a stale policy
    that would silently emit garbage actions (the :class:`heat.agents.ml_agent.MLAgent`
    §3.4 tripwire lesson, applied to the tiny Direction-A checkpoint).
    """


_AVERAGE_METADATA_KEYS = (
    "head",
    "encoder",
    "hidden_sizes",
    "obs_dim",
    "action_dim",
    "codec_version",
)


def average_policy_checkpoints(
    paths: Sequence[str | Path], output_path: str | Path
) -> None:
    """Uniformly average compatible policy checkpoints into one derived policy.

    Floating tensors are accumulated in float64 and cast back to their original
    dtype. Non-floating tensors must be identical. Architecture and codec
    metadata must also match exactly, so averaging can never bridge incompatible
    policies silently.

    Args:
        paths: two or more compatible source checkpoint paths.
        output_path: destination for the derived checkpoint blob.

    Raises:
        ValueError: when fewer than two sources are supplied or any metadata,
            state key, shape, dtype, or non-floating tensor differs.
    """
    sources = [Path(path) for path in paths]
    if len(sources) < 2:
        raise ValueError("checkpoint averaging requires at least two sources")

    blobs: list[dict[str, Any]] = [
        torch.load(str(path), map_location="cpu", weights_only=False)
        for path in sources
    ]
    first = blobs[0]
    for path, blob in zip(sources[1:], blobs[1:], strict=True):
        for key in _AVERAGE_METADATA_KEYS:
            if blob.get(key) != first.get(key):
                raise ValueError(
                    f"checkpoint {path} has incompatible {key}: "
                    f"{blob.get(key)!r} != {first.get(key)!r}"
                )

    states = [blob.get("state_dict") for blob in blobs]
    if not all(isinstance(state, dict) for state in states):
        raise ValueError("every checkpoint must contain a state_dict")
    typed_states = [state for state in states if isinstance(state, dict)]
    keys = list(typed_states[0])
    if any(list(state) != keys for state in typed_states[1:]):
        raise ValueError("checkpoint state_dict keys or ordering differ")

    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in typed_states]
        reference = tensors[0]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError(f"state_dict entry {key!r} is not a tensor")
        if any(
            tensor.shape != reference.shape or tensor.dtype != reference.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"checkpoint tensor {key!r} shape or dtype differs")
        if reference.is_floating_point() or reference.is_complex():
            accumulation_dtype = (
                torch.complex128 if reference.is_complex() else torch.float64
            )
            averaged[key] = (
                torch.stack([tensor.to(accumulation_dtype) for tensor in tensors])
                .mean(dim=0)
                .to(reference.dtype)
            )
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating checkpoint tensor {key!r} differs")
            averaged[key] = reference.clone()

    output = {key: value for key, value in first.items() if key != "state_dict"}
    output["state_dict"] = averaged
    output["derived_from"] = [str(path).replace("\\", "/") for path in sources]
    output["derivation"] = {
        "method": "uniform_parameter_mean",
        "count": len(sources),
        "accumulation_dtype": "float64_or_complex128",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(destination))


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
        "encoder": config.encoder,
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
    ``state_dict``. Dims must match the live contract. Codec 3 and 4 both load;
    the returned policy keeps the blob's codec so historical inputs stay on v3.
    Anything else raises :class:`CheckpointMismatchError`.

    Raises:
        CheckpointMismatchError: on an ``obs_dim`` / ``action_dim`` mismatch, or
            a ``codec_version`` outside ``(3, 4)``.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)

    mismatches: list[str] = []
    if blob.get("obs_dim") != OBS_DIM:
        mismatches.append(
            f"obs_dim: checkpoint={blob.get('obs_dim')!r} != runtime={OBS_DIM!r}"
        )
    if blob.get("action_dim") != ACTION_DIM:
        mismatches.append(
            f"action_dim: checkpoint={blob.get('action_dim')!r} != runtime={ACTION_DIM!r}"
        )
    if blob.get("codec_version") not in LEGACY_PLAYABLE_CODECS:
        mismatches.append(
            "codec_version: checkpoint="
            f"{blob.get('codec_version')!r} != runtime={LEGACY_PLAYABLE_CODECS!r}"
        )
    if mismatches:
        raise CheckpointMismatchError(
            f"Checkpoint {str(path)!r} is incompatible with the current ML "
            "contract (stale policy vs drifted codec): " + "; ".join(mismatches)
        )

    config = A0Config(
        head=str(blob["head"]),
        encoder=str(blob.get("encoder", "flat")),
        hidden_sizes=tuple(int(h) for h in blob["hidden_sizes"]),
    )
    policy = build_policy(config)
    policy.load_state_dict(blob["state_dict"])
    policy.codec_version = int(blob["codec_version"])
    policy.eval()
    return policy
