"""Focused checks for learned-policy generation identity and provenance."""

from __future__ import annotations

from pathlib import Path

from heat.ml.policy_registry import (
    DEFAULT_REGISTRY_ROOT,
    find_registered_run,
    load_generation,
    validate_registry,
)


def test_retroactive_policy_registry_is_complete_and_hashed() -> None:
    """The three A8 generations resolve to the exact selected checkpoints."""
    workspace = Path(__file__).resolve().parents[1]
    validate_registry(
        DEFAULT_REGISTRY_ROOT,
        workspace_root=workspace,
        verify_artifacts=True,
    )
    assert find_registered_run("G0002-R00")["selected_checkpoint"] == (
        "runs/a8_s1_5ep_seed0/step_1000000.pt"
    )
    assert find_registered_run("G0003-R02")["selected_checkpoint_id"] == (
        "G0003-R02@final"
    )


def test_generation_changes_only_capture_recipe_changes() -> None:
    """Runs vary seeds, while generation manifests preserve recipe differences."""
    g1 = load_generation("G0001")
    g2 = load_generation("G0002")
    g3 = load_generation("G0003")
    assert [run["seed"] for run in g2["runs"]] == [0, 1, 2]
    assert g1["recipe"]["training_config"]["n_epochs"] == 10
    assert g2["recipe"]["training_config"]["n_epochs"] == 5
    assert g2["recipe"]["training_config"]["anchor_share"] == 0.0
    assert g3["recipe"]["training_config"]["anchor_share"] == 0.4
    assert g3["recipe"]["opponent_anchor"]["agent_checkpoint_id"] == (
        "G0002-R00@1000K"
    )
