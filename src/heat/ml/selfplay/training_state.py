"""Versioned, atomic full-state checkpoints for segmented self-play training."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
import pickle
import secrets
import struct
from typing import Any, cast

import numpy as np
import torch

from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


TRAINING_STATE_SCHEMA = 1


class TrainingStateError(RuntimeError):
    """Raised when a full-state checkpoint is corrupt or incompatible."""


def resolved_recipe(config: object, track_params: object | None) -> dict[str, Any]:
    """Return the JSON-safe recipe and live codec contract used for validation."""
    if not is_dataclass(config):
        raise TypeError("training config must be a dataclass instance")
    config_values = asdict(cast(Any, config))
    params = asdict(cast(Any, track_params)) if track_params is not None else None
    return {
        "config": config_values,
        "track_params": params,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "codec_version": CODEC_VERSION,
    }


def recipe_sha256(recipe: dict[str, Any]) -> str:
    """Hash a resolved recipe with stable JSON ordering and separators."""
    encoded = json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_receipt_path(path: str | Path) -> Path:
    """Return the SHA-256 sidecar path for a training-state checkpoint."""
    checkpoint = Path(path)
    return checkpoint.with_suffix(checkpoint.suffix + ".sha256")


def _sha256_file(path: Path) -> str:
    """Hash a file without loading the whole checkpoint into memory again."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Flush bytes to a same-directory temporary file and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_state(path: str | Path, payload: dict[str, Any]) -> str:
    """Atomically write one checkpoint and its SHA-256 receipt.

    The checkpoint replacement happens before the receipt replacement. A crash
    between them leaves a safe hash mismatch rather than silently accepting an
    unreceipted state.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        os.replace(temporary, destination)
        receipt = f"{digest}  {destination.name}\n".encode("ascii")
        _atomic_bytes(checkpoint_receipt_path(destination), receipt)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _read_verified_digest(path: Path) -> str:
    """Verify the receipt format and checkpoint bytes before deserialization."""
    receipt_path = checkpoint_receipt_path(path)
    try:
        line = receipt_path.read_text(encoding="ascii").strip()
        expected, separator, filename = line.partition("  ")
    except OSError as exc:
        raise TrainingStateError(f"missing checkpoint receipt: {receipt_path}") from exc
    if separator != "  " or filename != path.name or len(expected) != 64:
        raise TrainingStateError(f"invalid checkpoint receipt: {receipt_path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise TrainingStateError(
            f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_training_state(
    path: str | Path,
    *,
    expected_recipe_sha256: str,
    campaign_id: str,
    source_identity: str,
    anchor_identity: str | None,
) -> dict[str, Any]:
    """Verify and deserialize a compatible training checkpoint.

    All byte-, schema-, recipe-, campaign-, source-, anchor-, and structural
    checks happen before the caller receives the payload and mutates a trainer.
    """
    source = Path(path)
    _read_verified_digest(source)
    try:
        loaded = torch.load(str(source), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise TrainingStateError(f"cannot deserialize checkpoint: {source}") from exc
    if not isinstance(loaded, dict):
        raise TrainingStateError("training checkpoint root must be a dictionary")
    payload: dict[str, Any] = loaded
    stored_recipe = payload.get("recipe")
    if not isinstance(stored_recipe, dict):
        raise TrainingStateError("training checkpoint recipe must be a dictionary")
    if recipe_sha256(stored_recipe) != payload.get("recipe_sha256"):
        raise TrainingStateError("training checkpoint recipe fingerprint is invalid")
    expected = {
        "schema_version": TRAINING_STATE_SCHEMA,
        "recipe_sha256": expected_recipe_sha256,
        "campaign_id": campaign_id,
        "source_identity": source_identity,
        "anchor_identity": anchor_identity,
    }
    mismatches = [
        f"{key}: checkpoint={payload.get(key)!r} runtime={value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise TrainingStateError(
            "incompatible training checkpoint: " + "; ".join(mismatches)
        )
    required = {
        "policy",
        "optimizer",
        "controller",
        "snapshot_pool",
        "rng",
        "schedule",
        "progress",
        "records",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise TrainingStateError(f"training checkpoint is missing fields: {missing}")
    if not isinstance(payload["policy"], dict):
        raise TrainingStateError("checkpoint policy state must be a dictionary")
    if not isinstance(payload["optimizer"], dict):
        raise TrainingStateError("checkpoint optimizer state must be a dictionary")
    return payload


def _digest_value(digest: Any, value: object) -> None:
    """Feed one nested checkpoint value into a type-delimited canonical digest."""
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
    elif isinstance(value, int):
        digest.update(f"int:{value};".encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"float:")
        digest.update(struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode("ascii"))
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        meta = f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
        digest.update(meta)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"ndarray:{array.dtype}:{array.shape}:".encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _digest_value(digest, value.item())
    elif isinstance(value, dict):
        digest.update(f"dict:{len(value)}:".encode("ascii"))
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}:".encode("ascii"))
        for item in value:
            _digest_value(digest, item)
    else:
        raise TypeError(f"unsupported checkpoint digest type: {type(value).__name__}")


def training_state_digest(payload: dict[str, Any]) -> str:
    """Return a serialization-independent digest for resume equivalence gates."""
    digest = hashlib.sha256()
    _digest_value(digest, payload)
    return digest.hexdigest()
