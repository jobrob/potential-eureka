"""Focused preparation checks for the deferred Direction D2 Chunk-6 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmark_direction_d2 import (
    _load_passed_correctness,
    _parse_args,
    project_d1_end_to_end,
)


def test_d1_projection_keeps_inference_update_and_preparation_fixed() -> None:
    """The estimate accelerates only the explicitly replaceable rollout work."""
    artifact = {
        "runs": [
            {
                "collector_mode": "scalar",
                "repeat": 0,
                "rollout_seconds": 70.0,
                "inference_seconds": 40.0,
                "update_seconds": 10.0,
                "prepare_seconds": 1.0,
                "iteration_seconds": 81.0,
            },
            {"collector_mode": "lanes"},
        ]
    }

    projection = project_d1_end_to_end(artifact, 5.0)

    run = projection["runs"][0]
    assert run["replaceable_rollout_seconds"] == 30.0
    assert run["projected_iteration_seconds"] == 57.0
    assert projection["median_projected_end_to_end_speedup"] == pytest.approx(
        81.0 / 57.0
    )


def test_benchmark_requires_the_completed_chunk5_gate(tmp_path: Path) -> None:
    """Timing cannot start from a smoke, failed, or mismatched correctness run."""
    path = tmp_path / "chunk5.json"
    passed = {
        "stage": "D2_chunk_5_random_differential",
        "status": "pass",
        "completed_transitions": 100_349,
        "semantic_mismatches": 0,
    }
    path.write_text(json.dumps(passed), encoding="utf-8")
    assert _load_passed_correctness(path) == passed

    passed["completed_transitions"] = 99_999
    path.write_text(json.dumps(passed), encoding="utf-8")
    with pytest.raises(ValueError, match="passed 100k"):
        _load_passed_correctness(path)


def test_benchmark_cli_retains_the_frozen_batch_curve() -> None:
    """Default launch parameters contain every designed CPU batch size."""
    args = _parse_args([])
    assert args.batch_sizes == [1, 8, 32, 64, 128, 256]
    assert args.timeout_seconds == 3_600.0
