"""Tests for Sprint A3 (Option A): the bounded value-iteration orchestrator.

The expensive ML steps (generate / refit / gate) are stubbed where the test is
about the *orchestration logic* -- the stop rule, the best-checkpoint
preservation, the dataset union -- not the ML. The one step exercised for real is
``_merge_datasets`` (the union invariant), driven with tiny synthetic ``.npz``
files because that is the load-bearing data-correctness guarantee.

Covered:
  (a) **Union-of-data invariant.** Iteration N's refit trains on >= all rows from
      every prior round (the seed dataset + every generated round so far). The
      orchestrator merges with ``_merge_datasets`` and the merged row count is
      monotone non-decreasing and equals the sum of source rows.
  (b) **Stop rule fires on no-improvement.** A monkeypatched gate that returns a
      non-improving metric makes the orchestrator stop after the first
      non-improving round and discard it (best stays the seed).
  (c) **Best-checkpoint preservation.** A worse later iteration never overwrites
      the shipped V; the best-gating checkpoint is the one copied to ``--out``.

The orchestration tests stub generate/refit so they are fast and deterministic.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# experiments/ is not a package; add it so the A3 deliverables import by name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import value_iterate  # noqa: E402
from heat.ml.spaces import OBS_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_npz(path: str, n_rows: int, *, train_frac: float = 0.6) -> None:
    """Write a synthetic value .npz with ``n_rows`` rows (and its sidecar)."""
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    obs = rng.uniform(-1, 1, size=(n_rows, OBS_DIM)).astype(np.float32)
    rr = rng.integers(0, 20, size=n_rows).astype(np.float32)
    cl = rng.integers(0, 4, size=n_rows).astype(np.int8)
    ts = rng.integers(0, 10, size=n_rows).astype(np.int64)
    n_train = int(n_rows * train_frac)
    split = np.array(
        [b"train"] * n_train + [b"val"] * (n_rows - n_train), dtype="S5"
    )
    np.savez_compressed(
        path,
        obs=obs,
        rounds_remaining=rr,
        corner_limit_at_state=cl,
        track_seed=ts,
        split=split,
    )
    # Minimal sidecar so train_value._load_dataset's codec cross-check passes (not
    # exercised here, but keeps the file shaped like a real one).
    import json

    from heat.ml.spaces import CODEC_VERSION

    with open(os.path.splitext(path)[0] + ".value.json", "w", encoding="utf-8") as fh:
        json.dump({"codec_version": CODEC_VERSION, "obs_dim": OBS_DIM}, fh)


def _gate(spins: float, rounds: float) -> value_iterate.GateMetric:
    return value_iterate.GateMetric(
        worst_case_limit1_spins=spins, rounds_to_finish=rounds
    )


def _orch_args(tmp_path, **overrides) -> argparse.Namespace:
    base = dict(
        seed_model=str(tmp_path / "seed.zip"),
        seed_data=str(tmp_path / "seed.npz"),
        out=str(tmp_path / "out.zip"),
        workdir=str(tmp_path / "wd"),
        iterations=2,
        train_tracks=2,
        val_tracks=1,
        rollouts=1,
        gate_games=2,
        epochs=1,
        horizon=1,
        dets=1,
        top_k=6,
        sim_budget=None,
        loss="mse",
        net="default",
        device="cpu",
        game_seed=7000,
        seed=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# GateMetric stop-rule unit tests
# ---------------------------------------------------------------------------


def test_gate_strict_improvement_on_either_axis():
    base = _gate(0.5, 20.0)
    assert _gate(0.5, 19.0).strictly_improves_on(base)  # rounds down only
    assert _gate(0.4, 20.0).strictly_improves_on(base)  # spins down only
    assert _gate(0.4, 19.0).strictly_improves_on(base)  # both down


def test_gate_no_improvement_is_not_strict():
    base = _gate(0.5, 20.0)
    assert not _gate(0.5, 20.0).strictly_improves_on(base)  # identical
    # Better rounds but regressed spins -> blocked by the "neither regresses" rule.
    assert not _gate(0.7, 19.0).strictly_improves_on(base)
    # Better spins but regressed rounds -> blocked.
    assert not _gate(0.4, 21.0).strictly_improves_on(base)


def test_gate_nan_axis_is_inert():
    base = _gate(float("nan"), 20.0)
    # A real lower rounds with a NaN spins axis on both sides still improves.
    assert _gate(float("nan"), 19.0).strictly_improves_on(base)
    # NaN cannot itself be an improvement.
    assert not _gate(float("nan"), 20.0).strictly_improves_on(base)


# ---------------------------------------------------------------------------
# (a) union-of-data invariant (real _merge_datasets)
# ---------------------------------------------------------------------------


def test_merge_datasets_is_row_union(tmp_path):
    p1 = str(tmp_path / "a.npz")
    p2 = str(tmp_path / "b.npz")
    p3 = str(tmp_path / "c.npz")
    _write_tiny_npz(p1, 10)
    _write_tiny_npz(p2, 7)
    _write_tiny_npz(p3, 5)

    out2 = str(tmp_path / "u2.npz")
    out3 = str(tmp_path / "u3.npz")
    s2 = value_iterate._merge_datasets([p1, p2], out2)
    s3 = value_iterate._merge_datasets([p1, p2, p3], out3)

    assert s2["n_rows"] == 17
    assert s3["n_rows"] == 22
    # Iteration N (3 sources) trains on >= all prior rows (2 sources).
    assert s3["n_rows"] >= s2["n_rows"]

    merged = np.load(out3)
    assert merged["obs"].shape == (22, OBS_DIM)
    # Split labels are carried verbatim (track-disjoint semantics preserved).
    assert set(np.unique(merged["split"]).tolist()) == {b"train", b"val"}


def test_orchestrator_trains_on_growing_union(tmp_path, monkeypatch):
    """Each refit sees a union whose row count is monotone non-decreasing and
    includes the seed dataset plus every round generated so far."""
    _write_tiny_npz(str(tmp_path / "seed.npz"), 8)
    (tmp_path / "seed.zip").write_bytes(b"seed-model")

    union_sizes: list[int] = []

    def fake_generate(*, out, **kw):
        _write_tiny_npz(out, 4)
        return {"n_rows": 4, "races_dropped_max_rounds": 0}

    def fake_refit(*, data, out, **kw):
        rows = int(np.load(data)["obs"].shape[0])
        union_sizes.append(rows)
        # Produce a "checkpoint" file so best-checkpoint copy has something to copy.
        with open(out, "wb") as fh:
            fh.write(b"ckpt")
        return {"out": out}

    # Always improve so both iterations run (drives the union to grow twice).
    improving = iter([_gate(0.4, 19.0), _gate(0.3, 18.0)])

    def fake_gate(**kw):
        # First call is the seed baseline; then each round.
        return next(fake_gate._it)

    fake_gate._it = iter([_gate(0.5, 20.0), _gate(0.4, 19.0), _gate(0.3, 18.0)])

    monkeypatch.setattr(value_iterate, "_generate", fake_generate)
    monkeypatch.setattr(value_iterate, "_refit", fake_refit)
    monkeypatch.setattr(value_iterate, "_gate", fake_gate)

    report = value_iterate.run_iteration(_orch_args(tmp_path))

    assert len(union_sizes) == 2
    # Round 1 union = seed(8) + round1(4) = 12; round 2 = +round2(4) = 16.
    assert union_sizes == [12, 16]
    assert union_sizes[1] >= union_sizes[0]  # monotone non-decreasing
    assert report["iterations_run"] == 2


# ---------------------------------------------------------------------------
# (b) stop rule fires on no-improvement
# ---------------------------------------------------------------------------


def test_stop_rule_discards_non_improving_round(tmp_path, monkeypatch):
    """A first round that does not strictly improve is discarded and iteration
    stops; the best stays the seed V."""
    _write_tiny_npz(str(tmp_path / "seed.npz"), 8)
    (tmp_path / "seed.zip").write_bytes(b"seed-model")

    def fake_generate(*, out, **kw):
        _write_tiny_npz(out, 4)
        return {"n_rows": 4, "races_dropped_max_rounds": 0}

    def fake_refit(*, data, out, **kw):
        with open(out, "wb") as fh:
            fh.write(b"ckpt")
        return {"out": out}

    # seed gate, then a round that does NOT improve (identical metric).
    gate_seq = iter([_gate(0.5, 20.0), _gate(0.5, 20.0)])
    monkeypatch.setattr(value_iterate, "_generate", fake_generate)
    monkeypatch.setattr(value_iterate, "_refit", fake_refit)
    monkeypatch.setattr(value_iterate, "_gate", lambda **kw: next(gate_seq))

    report = value_iterate.run_iteration(_orch_args(tmp_path, iterations=2))

    assert report["iterations_run"] == 1  # stopped after the first non-improver
    assert report["rounds"][0]["retained"] is False
    assert report["best_is_seed"] is True
    # Shipped V is the seed model (preserved verbatim).
    assert open(report["out"], "rb").read() == b"seed-model"


def test_hard_cap_is_two_iterations(tmp_path, monkeypatch):
    """Even if every round improves and the user asks for more, never exceed 2."""
    _write_tiny_npz(str(tmp_path / "seed.npz"), 8)
    (tmp_path / "seed.zip").write_bytes(b"seed-model")

    def fake_generate(*, out, **kw):
        _write_tiny_npz(out, 4)
        return {"n_rows": 4, "races_dropped_max_rounds": 0}

    def fake_refit(*, data, out, **kw):
        with open(out, "wb") as fh:
            fh.write(b"ckpt")
        return {"out": out}

    # Always strictly improving so the cap (not the stop rule) is what halts it.
    rounds = [_gate(0.5, 20.0)] + [_gate(0.5 - 0.05 * i, 20.0 - i) for i in range(1, 6)]
    gate_seq = iter(rounds)
    monkeypatch.setattr(value_iterate, "_generate", fake_generate)
    monkeypatch.setattr(value_iterate, "_refit", fake_refit)
    monkeypatch.setattr(value_iterate, "_gate", lambda **kw: next(gate_seq))

    report = value_iterate.run_iteration(_orch_args(tmp_path, iterations=5))
    assert report["iterations_run"] == 2  # hard cap, not 5


# ---------------------------------------------------------------------------
# (c) best-checkpoint preservation
# ---------------------------------------------------------------------------


def test_best_checkpoint_preserved_when_later_round_regresses(tmp_path, monkeypatch):
    """Round 1 improves (retained); round 2 regresses (discarded). The shipped V
    is round 1's checkpoint, NOT round 2's worse one."""
    _write_tiny_npz(str(tmp_path / "seed.npz"), 8)
    (tmp_path / "seed.zip").write_bytes(b"seed-model")

    def fake_generate(*, out, **kw):
        _write_tiny_npz(out, 4)
        return {"n_rows": 4, "races_dropped_max_rounds": 0}

    def fake_refit(*, data, out, **kw):
        # Tag each round's checkpoint with its filename so we can identify which
        # one was shipped.
        with open(out, "wb") as fh:
            fh.write(os.path.basename(out).encode())
        return {"out": out}

    # seed(0.5,20) -> round1 improves (0.4,19) -> round2 regresses (0.6,21).
    gate_seq = iter([_gate(0.5, 20.0), _gate(0.4, 19.0), _gate(0.6, 21.0)])
    monkeypatch.setattr(value_iterate, "_generate", fake_generate)
    monkeypatch.setattr(value_iterate, "_refit", fake_refit)
    monkeypatch.setattr(value_iterate, "_gate", lambda **kw: next(gate_seq))

    report = value_iterate.run_iteration(_orch_args(tmp_path, iterations=2))

    assert report["iterations_run"] == 2
    assert report["rounds"][0]["retained"] is True
    assert report["rounds"][1]["retained"] is False
    assert report["best_is_seed"] is False
    # The shipped V is round 1's checkpoint (value_round1.zip), not round 2's.
    shipped = open(report["out"], "rb").read()
    assert shipped == b"value_round1.zip"
    assert report["best_gate"] == _gate(0.4, 19.0).as_dict()
