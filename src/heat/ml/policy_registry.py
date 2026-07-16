"""Validation and lookup helpers for immutable learned-policy generations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

GENERATION_ID_RE = re.compile(r"^G\d{4}$")
AGENT_ID_RE = re.compile(r"^(G\d{4})-R(\d{2})$")
CHECKPOINT_ID_RE = re.compile(r"^G\d{4}-R\d{2}@(final|\d+K)$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_REGISTRY_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "policy_registry"


class PolicyRegistryError(ValueError):
    """Raised when registry chronology, identity, or artifact evidence drifts."""


def recipe_sha256(recipe: dict[str, Any]) -> str:
    """Hash the canonical recipe payload, excluding its stored hash field."""
    payload = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object or raise a registry-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyRegistryError(f"Cannot read registry JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyRegistryError(f"Registry file must contain an object: {path}")
    return value


def load_generation(
    generation_id: str,
    registry_root: Path = DEFAULT_REGISTRY_ROOT,
) -> dict[str, Any]:
    """Load one generation manifest by its immutable identifier."""
    if not GENERATION_ID_RE.fullmatch(generation_id):
        raise PolicyRegistryError(f"Invalid generation ID: {generation_id!r}")
    return _read_json(registry_root / f"{generation_id.lower()}.json")


def find_registered_run(
    agent_id: str,
    registry_root: Path = DEFAULT_REGISTRY_ROOT,
) -> dict[str, Any]:
    """Return the registered selected-run record for ``G####-R##``."""
    match = AGENT_ID_RE.fullmatch(agent_id)
    if match is None:
        raise PolicyRegistryError(f"Invalid agent ID: {agent_id!r}")
    manifest = load_generation(match.group(1), registry_root)
    for run in manifest.get("runs", []):
        if run.get("agent_id") == agent_id:
            return dict(run)
    raise PolicyRegistryError(f"Unknown registered agent: {agent_id}")


def validate_registry(
    registry_root: Path = DEFAULT_REGISTRY_ROOT,
    *,
    workspace_root: Path | None = None,
    verify_artifacts: bool = False,
) -> None:
    """Validate chronology, recipe hashes, IDs, and optional checkpoint hashes."""
    index = _read_json(registry_root / "index.json")
    entries = index.get("generations")
    if not isinstance(entries, list) or not entries:
        raise PolicyRegistryError("index.json must contain non-empty generations")

    expected_numbers = list(range(1, len(entries) + 1))
    actual_numbers: list[int] = []
    seen_agents: set[str] = set()
    known_generations: set[str] = set()
    root = workspace_root or registry_root.parents[1]

    for entry in entries:
        generation_id = entry.get("generation_id")
        if not isinstance(generation_id, str) or not GENERATION_ID_RE.fullmatch(
            generation_id
        ):
            raise PolicyRegistryError(f"Invalid generation entry: {generation_id!r}")
        actual_numbers.append(int(generation_id[1:]))
        parent = entry.get("parent_generation_id")
        if parent is not None and parent not in known_generations:
            raise PolicyRegistryError(
                f"{generation_id} parent must be an earlier registered generation"
            )

        manifest_path = root / str(entry.get("manifest"))
        manifest = _read_json(manifest_path)
        if manifest.get("generation_id") != generation_id:
            raise PolicyRegistryError(f"Manifest ID mismatch for {generation_id}")
        if manifest.get("parent_generation_id") != parent:
            raise PolicyRegistryError(f"Parent mismatch for {generation_id}")
        recipe = manifest.get("recipe")
        if not isinstance(recipe, dict):
            raise PolicyRegistryError(f"Missing recipe object for {generation_id}")
        if recipe.get("recipe_sha256") != recipe_sha256(recipe):
            raise PolicyRegistryError(f"Recipe hash mismatch for {generation_id}")

        provenance = manifest.get("source_provenance")
        if not isinstance(provenance, dict):
            raise PolicyRegistryError(f"Missing source provenance for {generation_id}")
        quality = provenance.get("quality")
        if quality == "retroactively_reconstructed":
            if int(generation_id[1:]) > 3:
                raise PolicyRegistryError(
                    "Only the G0001-G0003 backfill may use reconstructed provenance"
                )
        elif quality in {"captured_clean", "captured_dirty"}:
            commit = provenance.get("git_commit")
            if not isinstance(commit, str) or not GIT_COMMIT_RE.fullmatch(commit):
                raise PolicyRegistryError(f"Invalid source commit for {generation_id}")
            if quality == "captured_dirty":
                patch_hash = provenance.get("dirty_patch_sha256")
                patch_path = provenance.get("dirty_patch")
                if not isinstance(patch_hash, str) or not SHA256_RE.fullmatch(
                    patch_hash
                ):
                    raise PolicyRegistryError(
                        f"Invalid dirty patch hash for {generation_id}"
                    )
                if not isinstance(patch_path, str) or not patch_path:
                    raise PolicyRegistryError(
                        f"Missing dirty patch path for {generation_id}"
                    )
        else:
            raise PolicyRegistryError(
                f"Unknown source provenance quality for {generation_id}: {quality!r}"
            )

        runs = manifest.get("runs")
        if not isinstance(runs, list) or not runs:
            raise PolicyRegistryError(f"{generation_id} has no registered runs")
        for run in runs:
            agent_id = run.get("agent_id")
            checkpoint_id = run.get("selected_checkpoint_id")
            if not isinstance(agent_id, str) or not AGENT_ID_RE.fullmatch(agent_id):
                raise PolicyRegistryError(f"Invalid agent ID in {generation_id}")
            if not agent_id.startswith(generation_id + "-"):
                raise PolicyRegistryError(f"Agent {agent_id} is in the wrong generation")
            if agent_id in seen_agents:
                raise PolicyRegistryError(f"Duplicate agent ID: {agent_id}")
            if not isinstance(checkpoint_id, str) or not CHECKPOINT_ID_RE.fullmatch(
                checkpoint_id
            ):
                raise PolicyRegistryError(f"Invalid checkpoint ID for {agent_id}")
            if not checkpoint_id.startswith(agent_id + "@"):
                raise PolicyRegistryError(f"Checkpoint ID mismatch for {agent_id}")
            seen_agents.add(agent_id)

            if verify_artifacts:
                artifact = root / str(run.get("selected_checkpoint"))
                if not artifact.is_file():
                    raise PolicyRegistryError(f"Missing checkpoint: {artifact}")
                if artifact.stat().st_size != run.get("bytes"):
                    raise PolicyRegistryError(f"Checkpoint size mismatch: {artifact}")
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                if digest != run.get("sha256"):
                    raise PolicyRegistryError(f"Checkpoint hash mismatch: {artifact}")
        known_generations.add(generation_id)

    if actual_numbers != expected_numbers:
        raise PolicyRegistryError("Generation IDs must be contiguous and chronological")
    if index.get("next_generation_number") != len(entries) + 1:
        raise PolicyRegistryError("next_generation_number is not the next unused ID")
