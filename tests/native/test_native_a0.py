"""Focused build/API gate for Direction D3 chunk A0."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from heat_native import NativePool


def _pool(rows: int = 4) -> NativePool:
    return NativePool(
        {"row_capacity": rows},
        np.zeros((rows, 104), dtype=np.float32),
    )


def test_receipt_and_strict_fixed_buffer_roundtrip() -> None:
    """The opaque object exposes a frozen receipt and never force-casts buffers."""
    pool = _pool()
    receipt = pool.receipt()
    assert receipt["protocol_version"] == 1
    assert receipt["state_schema_hash"] == "d3-a5-v3-r104-a516-p6-r200-c424-t90"
    assert receipt["codec_version"] == 3
    assert receipt["build_type"] in {"debug", "release"}

    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    destination = np.empty_like(source)
    token = pool.roundtrip(source, destination)
    assert token == 1
    np.testing.assert_array_equal(destination, source)

    with pytest.raises(TypeError, match="float32"):
        pool.roundtrip(source.astype(np.float64), destination)
    with pytest.raises(ValueError, match="C-contiguous"):
        pool.roundtrip(source[:, ::2], destination[:, :3])
    destination.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        pool.roundtrip(source, destination)


def test_native_wait_releases_gil_and_close_is_checked() -> None:
    """A waiting native call lets another Python thread run and close is final."""
    pool = _pool()
    progressed = threading.Event()

    def mark_progress() -> None:
        time.sleep(0.01)
        progressed.set()

    thread = threading.Thread(target=mark_progress)
    thread.start()
    pool.gil_release_smoke(50)
    thread.join(timeout=1)
    assert progressed.is_set()

    pool.close()
    assert pool.closed
    with pytest.raises(RuntimeError, match="closed"):
        pool.gil_release_smoke(0)
