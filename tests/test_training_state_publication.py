"""Failure and retention gates for immutable training-checkpoint publication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from heat.ml.selfplay.training_state import (
    TRAINING_STATE_SCHEMA,
    TrainingStateError,
    checkpoint_receipt_path,
    load_training_state,
    recipe_sha256,
    save_training_state,
    training_state_digest,
)

_CAMPAIGN = "dev-publication"
_SOURCE = "publication-test"


def _payload(marker: str) -> dict[str, object]:
    """Build one valid checkpoint whose marker changes the digest."""
    recipe = {"kind": "publication-test"}
    return {
        "schema_version": TRAINING_STATE_SCHEMA,
        "recipe": recipe,
        "recipe_sha256": recipe_sha256(recipe),
        "campaign_id": _CAMPAIGN,
        "source_identity": _SOURCE,
        "anchor_identity": None,
        "policy": {"marker": marker},
        "optimizer": {},
        "controller": {},
        "snapshot_pool": {},
        "rng": {},
        "schedule": {},
        "progress": {},
        "records": [],
    }


def _load(path: Path) -> dict[str, object]:
    """Load a publication-test checkpoint through the production contract."""
    recipe = {"kind": "publication-test"}
    loaded = load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id=_CAMPAIGN,
        source_identity=_SOURCE,
        anchor_identity=None,
    )
    return loaded


def _pointer_name(path: Path) -> str:
    """Return the generation filename published for a logical checkpoint."""
    pointer = path.with_name(path.name + ".latest")
    return pointer.read_text(encoding="ascii").strip()


def test_failed_receipt_keeps_the_published_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt failure must not replace the last loadable pair."""
    path = tmp_path / "resume.pt"
    first = _payload("first")
    save_training_state(path, first)
    digest = training_state_digest(_load(path))
    assert digest == training_state_digest(first)
    assert _pointer_name(path) == "000001.pt"

    def fail_receipt(_checkpoint: Path, _digest: str) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(
        "heat.ml.selfplay.training_state._publish_checkpoint_receipt",
        fail_receipt,
    )
    with pytest.raises(OSError, match="injected receipt failure"):
        save_training_state(path, _payload("second"))

    assert training_state_digest(_load(path)) == digest
    assert _pointer_name(path) == "000001.pt"
    assert not path.exists()


def test_first_save_receipt_failure_is_not_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first failed publish must not leave an unverified logical checkpoint."""
    path = tmp_path / "resume.pt"

    def fail_receipt(_checkpoint: Path, _digest: str) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(
        "heat.ml.selfplay.training_state._publish_checkpoint_receipt",
        fail_receipt,
    )
    with pytest.raises(OSError, match="injected receipt failure"):
        save_training_state(path, _payload("only"))

    with pytest.raises(TrainingStateError):
        _load(path)
    assert not path.exists()
    assert not path.with_name(path.name + ".latest").exists()


def test_third_save_keeps_only_the_newest_two_generations(tmp_path: Path) -> None:
    """The latest pair and its predecessor remain; the oldest generation is removed."""
    path = tmp_path / "resume.pt"
    payloads = [_payload("one"), _payload("two"), _payload("three")]
    for payload in payloads:
        save_training_state(path, payload)

    loaded = _load(path)
    assert training_state_digest(loaded) == training_state_digest(payloads[2])
    assert loaded["policy"] == payloads[2]["policy"]
    assert _pointer_name(path) == "000003.pt"

    directory = path.with_name(path.name + ".generations")
    checkpoints = sorted(child.name for child in directory.glob("*.pt"))
    assert checkpoints == ["000002.pt", "000003.pt"]
    assert (directory / "000002.pt.sha256").is_file()
    assert (directory / "000003.pt.sha256").is_file()
    assert not (directory / "000001.pt").exists()
    assert not (directory / "000001.pt.sha256").exists()


def test_legacy_sidecar_loads_until_a_pointer_is_published(tmp_path: Path) -> None:
    """An old path-plus-sidecar pair still loads and is not overwritten in place."""
    path = tmp_path / "resume.pt"
    legacy = _payload("legacy")
    torch.save(legacy, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_receipt_path(path).write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    original = path.read_bytes()
    assert training_state_digest(_load(path)) == training_state_digest(legacy)

    updated = _payload("updated")
    save_training_state(path, updated)

    assert path.read_bytes() == original
    loaded = _load(path)
    assert training_state_digest(loaded) == training_state_digest(updated)
    assert loaded["policy"] == updated["policy"]
    assert _pointer_name(path) == "000001.pt"
