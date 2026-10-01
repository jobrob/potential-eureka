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
    """Return the SHA-256 sidecar path for one checkpoint file."""
    checkpoint = Path(path)
    return checkpoint.with_suffix(checkpoint.suffix + ".sha256")


def _generations_dir(logical: Path) -> Path:
    """Return the sibling directory that holds immutable generations."""
    return logical.with_name(logical.name + ".generations")


def _latest_pointer_path(logical: Path) -> Path:
    """Return the pointer replaced only after a generation pair is complete."""
    return logical.with_name(logical.name + ".latest")


def _is_single_filename(name: str) -> bool:
    """Reject pointer values that are empty or leave the generations directory."""
    return bool(name) and name == Path(name).name and name not in {".", ".."}


def _parse_generation_filename(name: str, suffix: str) -> int | None:
    """Return the generation number in ``name``, or None when it is not one."""
    if not _is_single_filename(name):
        return None
    if suffix:
        if not name.endswith(suffix):
            return None
        stem = name[: -len(suffix)]
    else:
        stem = name
    if not stem.isascii() or not stem.isdigit():
        return None
    return int(stem)


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


def _pointer_text(logical: Path) -> str | None:
    """Return the stripped latest-pointer text, or None when it is absent."""
    pointer = _latest_pointer_path(logical)
    if not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise TrainingStateError(f"cannot read checkpoint pointer: {pointer}") from exc
    stripped = text.strip()
    return stripped or None


def _allocate_generation(logical: Path) -> Path:
    """Pick the next generation path without reusing or replacing an older one."""
    directory = _generations_dir(logical)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = logical.suffix
    highest = 0
    for child in directory.iterdir():
        number = _parse_generation_filename(child.name, suffix)
        if number is not None:
            highest = max(highest, number)
    number = highest + 1
    while True:
        candidate = directory / f"{number:06d}{suffix}"
        if not candidate.exists() and not checkpoint_receipt_path(candidate).exists():
            return candidate
        number += 1


def _write_new_checkpoint(destination: Path, payload: dict[str, Any]) -> str:
    """Flush one new generation file and return the digest of those bytes."""
    if destination.exists() or checkpoint_receipt_path(destination).exists():
        raise TrainingStateError(
            f"refusing to replace checkpoint generation: {destination}"
        )
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        if destination.exists():
            raise TrainingStateError(
                f"refusing to replace checkpoint generation: {destination}"
            )
        os.replace(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _publish_checkpoint_receipt(checkpoint: Path, digest: str) -> None:
    """Atomically write the SHA-256 receipt beside one new generation file."""
    receipt_path = checkpoint_receipt_path(checkpoint)
    if receipt_path.exists():
        raise TrainingStateError(f"refusing to replace checkpoint receipt: {receipt_path}")
    receipt = f"{digest}  {checkpoint.name}\n".encode("ascii")
    _atomic_bytes(receipt_path, receipt)


def _publish_latest_pointer(logical: Path, generation_name: str) -> None:
    """Replace the latest pointer after both generation files are complete."""
    _atomic_bytes(
        _latest_pointer_path(logical),
        f"{generation_name}\n".encode("ascii"),
    )


def _checkpoint_name_for_entry(name: str, suffix: str) -> str | None:
    """Map a generation file or its receipt back to the checkpoint filename."""
    if _parse_generation_filename(name, suffix) is not None:
        return name
    if name.endswith(".sha256"):
        checkpoint_name = name[: -len(".sha256")]
        if _parse_generation_filename(checkpoint_name, suffix) is not None:
            return checkpoint_name
    return None


def _retain_published_generations(logical: Path, previous_name: str | None) -> None:
    """Keep the newly published generation and the previous pointer target.

    Older generation files are removed only after the pointer has moved. The
    current and previous pointer names are never deleted.
    """
    current_name = _pointer_text(logical)
    keep = {
        name
        for name in (current_name, previous_name)
        if name is not None and _is_single_filename(name)
    }
    directory = _generations_dir(logical)
    if not directory.is_dir() or not keep:
        return
    for child in list(directory.iterdir()):
        if not child.is_file():
            continue
        checkpoint_name = _checkpoint_name_for_entry(child.name, logical.suffix)
        if checkpoint_name is None or checkpoint_name in keep:
            continue
        child.unlink(missing_ok=True)


def save_training_state(path: str | Path, payload: dict[str, Any]) -> str:
    """Publish the next immutable generation, then move the latest pointer.

    The checkpoint and its receipt are written under ``<name>.generations``
    and never replace an older generation. ``<name>.latest`` is replaced only
    after the receipt matches the checkpoint bytes. A crash before that pointer
    move leaves the previous published pair loadable; a partial new generation
    may remain on disk but is not selected. After the pointer moves, the new
    pair and the previously published pair are kept and older generations are
    deleted. Legacy bytes already stored at ``path`` are not overwritten.
    """
    logical = Path(path)
    previous_name = _pointer_text(logical)
    generation = _allocate_generation(logical)
    digest = _write_new_checkpoint(generation, payload)
    _publish_checkpoint_receipt(generation, digest)
    if _read_verified_digest(generation) != digest:
        raise TrainingStateError(
            f"checkpoint receipt did not match published bytes: {generation}"
        )
    _publish_latest_pointer(logical, generation.name)
    _retain_published_generations(logical, previous_name)
    return digest


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
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise TrainingStateError(f"cannot read checkpoint: {path}") from exc
    if actual != expected:
        raise TrainingStateError(
            f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def published_checkpoint_path(path: str | Path) -> Path:
    """Return the checkpoint file a later load of ``path`` would verify.

    New saves publish a generation and a latest pointer. Older saves are the
    logical file itself. Callers that need the bytes or the receipt should use
    this path, not assume the bytes were written at the logical name.
    """
    return _resolve_load_path(Path(path))


def _resolve_load_path(logical: Path) -> Path:
    """Return the pointer's generation, or a legacy checkpoint when no pointer exists."""
    pointer = _latest_pointer_path(logical)
    if pointer.is_file():
        name = _pointer_text(logical)
        if name is None or _parse_generation_filename(name, logical.suffix) is None:
            raise TrainingStateError(f"invalid checkpoint pointer: {pointer}")
        return _generations_dir(logical) / name
    if logical.is_file():
        return logical
    raise TrainingStateError(f"missing training checkpoint: {logical}")


def load_training_state(
    path: str | Path,
    *,
    expected_recipe_sha256: str,
    campaign_id: str,
    source_identity: str,
    anchor_identity: str | None,
) -> dict[str, Any]:
    """Verify and deserialize the newest published training checkpoint.

    A ``<name>.latest`` pointer selects the generation. Without one, the legacy
    ``path`` plus ``path.sha256`` pair is used. Receipt, schema, recipe,
    campaign, source, anchor, and structural checks then run unchanged, before
    the caller receives the payload and mutates a trainer.
    """
    source = _resolve_load_path(Path(path))
    if not source.is_file():
        raise TrainingStateError(f"missing training checkpoint: {source}")
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
