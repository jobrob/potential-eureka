"""Focused regression checks for the A10 operational-soak reporter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.soak_direction_d3_a10 import _atomic_json, _parse_args


def test_soak_defaults_name_the_bounded_confirmation_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next soak defaults reproduce the 48-slot, eight-thread candidate."""
    monkeypatch.setattr(sys, "argv", ["soak_direction_d3_a10.py"])

    args = _parse_args()

    assert args.workers == 48
    assert args.ready_capacity == 288
    assert args.torch_threads == 8
    assert args.torch_interop_threads == 2


def test_soak_receipt_serializes_numpy_gate_scalars(tmp_path: Path) -> None:
    """The final reporter accepts NumPy comparison results without data loss."""
    output = tmp_path / "receipt.json"
    _atomic_json(
        output,
        {"passed": np.bool_(False), "ratio": np.float64(0.8283094811617953)},
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passed": False,
        "ratio": 0.8283094811617953,
    }
