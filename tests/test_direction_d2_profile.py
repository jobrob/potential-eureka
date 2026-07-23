"""Focused checks for the bounded Direction D2 profiling harness."""

from __future__ import annotations

import json
from time import perf_counter

import pytest

from experiments.profile_direction_d2 import (
    _measure,
    _python_profile,
    _torch_profile,
)
from experiments.verify_direction_d2_differential import prepare_batch


def test_profile_measurement_reports_median_and_spread() -> None:
    """Component timing retains every sample and a finite summary."""
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = _measure(
        operation,
        warmups=2,
        repeats=3,
        deadline=perf_counter() + 5.0,
    )

    assert calls == 5
    assert len(result["samples_seconds"]) == 3
    assert float(result["median_seconds"]) >= 0.0
    assert float(result["relative_spread"]) >= 0.0


def test_deep_profiles_are_json_serializable_for_one_lane() -> None:
    """Both profilers emit bounded machine-readable operator receipts."""
    batch = prepare_batch(500_000, 1, 7_007)
    deadline = perf_counter() + 30.0

    python_result = _python_profile(
        batch,
        repeats=1,
        top=5,
        deadline=deadline,
    )
    torch_result = _torch_profile(
        batch,
        repeats=1,
        top=5,
        deadline=deadline,
    )

    assert int(python_result["total_calls"]) > 0
    assert torch_result["top_by_self_cpu_time"]
    assert "direction_d2_full_slice" in {
        row["operator"] for row in torch_result["top_by_self_cpu_time"]
    }
    assert json.loads(json.dumps({"python": python_result, "torch": torch_result}))


def test_profile_deadline_is_enforced() -> None:
    """A stale deadline stops profiling before an unbounded measurement."""
    with pytest.raises(TimeoutError, match="wall-clock cap"):
        _measure(
            lambda: None,
            warmups=1,
            repeats=1,
            deadline=perf_counter() - 1.0,
        )
