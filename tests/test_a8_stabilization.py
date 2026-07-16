"""Focused checks for A8 S3 checkpoint averaging and frozen identities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.stabilize_a8_checkpoints import _specs
from heat.ml.selfplay.checkpoint import (
    average_policy_checkpoints,
    load_policy,
    save_policy,
)
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config


def _constant_checkpoint(path: Path, value: float, hidden: int = 8) -> None:
    """Write a tiny valid policy whose floating tensors all share one value."""
    config = A0Config(hidden_sizes=(hidden,))
    policy = build_policy(config)
    with torch.no_grad():
        for tensor in policy.state_dict().values():
            tensor.fill_(value)
    save_policy(policy, config, path)


def test_checkpoint_average_is_uniform_loadable_and_auditable(tmp_path: Path) -> None:
    """Averaging preserves the policy contract and records every source path."""
    sources = [tmp_path / f"source_{index}.pt" for index in range(3)]
    for index, source in enumerate(sources, start=1):
        _constant_checkpoint(source, float(index))
    output = tmp_path / "average.pt"

    average_policy_checkpoints(sources, output)

    loaded = load_policy(output)
    assert all(
        torch.allclose(tensor, torch.full_like(tensor, 2.0))
        for tensor in loaded.state_dict().values()
    )
    blob = torch.load(output, map_location="cpu", weights_only=False)
    assert blob["derived_from"] == [path.as_posix() for path in sources]
    assert blob["derivation"]["method"] == "uniform_parameter_mean"


def test_checkpoint_average_rejects_incompatible_architectures(tmp_path: Path) -> None:
    """S3 cannot silently average tensors from different policy contracts."""
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _constant_checkpoint(first, 1.0, hidden=8)
    _constant_checkpoint(second, 2.0, hidden=16)
    with pytest.raises(ValueError, match="hidden_sizes"):
        average_policy_checkpoints([first, second], tmp_path / "bad.pt")


def test_g0004_recipe_uses_one_frozen_window_for_every_seed() -> None:
    """Every run averages 500k, 750k, and 1M without per-seed selection."""
    specs = _specs(Path("runs/a8_s3_average"))
    assert [spec.agent_id for spec in specs] == [
        "G0004-R00",
        "G0004-R01",
        "G0004-R02",
    ]
    assert all(
        tuple(path.rsplit("_", 1)[-1] for path in spec.sources)
        == ("500000.pt", "750000.pt", "1000000.pt")
        for spec in specs
    )
