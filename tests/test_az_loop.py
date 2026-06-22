"""Tests for the Sprint C3 closed AlphaZero loop (``az_loop.py``).

Mirrors the C2 / A3 test style: the expensive ML steps (self-play generate /
train / gate) are stubbed where the test is about the *orchestration logic* -- the
CI promotion guard, best-checkpoint preservation, the recency-windowed aggregate,
and the seed-band disjointness -- and the load-bearing data-correctness step
(``_aggregate``) is exercised for real with tiny synthetic ``.npz`` files. One
end-to-end ``--smoke``-style run validates the pipeline produces a contract-checked
checkpoint.

Covered (the C3 Deliverables test list):
  * the loop runs >= 2 generations end-to-end (stubbed ML) and ships a checkpoint;
  * a deliberately-worse injected net is NOT promoted (incumbent kept -- the 8C
    collapse guard);
  * the aggregation buffer respects its cap / recency window;
  * the self-play vs eval (and C2 / precursor) seed bands are asserted disjoint.

The CI gate math (``_improves_ci`` / ``strictly_improves``) is unit-tested
directly. Everything runs on tiny synthetic fixtures in seconds -- no committed
data, no real net training in the orchestration tests.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# experiments/ is not a package; add it so the C3 deliverables import by name.
_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import az_loop  # noqa: E402
from eval_az import CI  # noqa: E402
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ci(point: float, lo: float | None = None, hi: float | None = None, n: int = 24) -> CI:
    """A CI with sensible default bounds around ``point`` (a +-0.001 tight band)."""
    if point != point:  # NaN
        return CI(float("nan"), float("nan"), float("nan"), 0)
    if lo is None:
        lo = point - 0.001
    if hi is None:
        hi = point + 0.001
    return CI(point, lo, hi, n)


def _gate(spins: float, rounds: float, finish: float = 1.0,
          *, spins_hi: float | None = None, rounds_hi: float | None = None) -> az_loop.LoopGate:
    return az_loop.LoopGate(
        spins=_ci(spins, hi=spins_hi),
        rounds=_ci(rounds, hi=rounds_hi),
        finish_rate=finish,
    )


def _write_tiny_selfplay_npz(path: str, n_rows: int, *, train_frac: float = 0.6) -> None:
    """Write a synthetic self-play .npz (target arrays + sidecar) for _aggregate."""
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    obs = rng.uniform(-1, 1, size=(n_rows, OBS_DIM)).astype(np.float32)
    # pi: a proper distribution over the first few legal actions per row.
    pi = np.zeros((n_rows, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((n_rows, ACTION_DIM), dtype=bool)
    for i in range(n_rows):
        k = int(rng.integers(2, 5))
        pi[i, :k] = 1.0 / k
        mask[i, :k] = True
    z = (-rng.integers(0, 20, size=n_rows)).astype(np.float32)
    kind = rng.integers(0, 5, size=n_rows).astype(np.int8)
    ts = rng.integers(0, 10, size=n_rows).astype(np.int64)
    n_train = int(n_rows * train_frac)
    split = np.array([b"train"] * n_train + [b"val"] * (n_rows - n_train), dtype="S5")
    np.savez_compressed(path, obs=obs, pi=pi, mask=mask, z=z, kind=kind,
                        track_seed=ts, split=split)
    import json
    with open(os.path.splitext(path)[0] + ".selfplay.json", "w", encoding="utf-8") as fh:
        json.dump({"codec_version": CODEC_VERSION, "obs_dim": OBS_DIM,
                   "action_dim": ACTION_DIM}, fh)


def _loop_args(tmp_path, **overrides) -> argparse.Namespace:
    base = dict(
        warm=str(tmp_path / "warm.zip"),
        out=str(tmp_path / "out.zip"),
        workdir=str(tmp_path / "wd"),
        generations=2,
        tracks=4,
        val_tracks=2,
        sims=8,
        buffer_window=2,
        epochs=2,
        c_v=1.0,
        gate_games=4,
        horizon=1,
        dets=1,
        stop_patience=99,
        net="default",
        device="cpu",
        seed=0,
        game_seed=8000,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# CI promotion-guard unit tests (the collapse guard)
# ---------------------------------------------------------------------------


class TestPromotionGuard:
    def test_strict_improvement_on_either_axis_promotes(self):
        base = _gate(0.50, 20.0)
        # Spins improve with CI separated below the incumbent point.
        assert az_loop.strictly_improves(_gate(0.40, 20.0, spins_hi=0.41), base)
        # Rounds improve, CI separated.
        assert az_loop.strictly_improves(_gate(0.50, 19.0, rounds_hi=19.1), base)

    def test_no_improvement_is_not_promoted(self):
        base = _gate(0.50, 20.0)
        assert not az_loop.strictly_improves(_gate(0.50, 20.0), base)  # identical

    def test_regression_blocks_promotion_even_with_other_axis_better(self):
        base = _gate(0.50, 20.0)
        # Rounds better but spins regressed -> blocked ("neither regresses").
        assert not az_loop.strictly_improves(
            _gate(0.70, 19.0, rounds_hi=19.1), base
        )

    def test_overlapping_ci_is_not_a_promotion(self):
        """A point edge whose CI overlaps the incumbent point is NOT promoted
        (the anti-fake-pass discipline: a lucky sample cannot fake a pass)."""
        base = _gate(0.50, 20.0)
        # Spins point lower (0.49) but the CI upper bound (0.51) overlaps 0.50.
        cand = az_loop.LoopGate(spins=CI(0.49, 0.47, 0.51, 24),
                                rounds=_ci(20.0), finish_rate=1.0)
        assert not az_loop.strictly_improves(cand, base)

    def test_collapse_with_subhundred_finish_is_never_promoted(self):
        """A candidate that gates better but does NOT finish 100% is a collapse
        and is never promoted (the 8C guard)."""
        base = _gate(0.50, 20.0)
        cand = _gate(0.10, 10.0, finish=0.8, spins_hi=0.11, rounds_hi=10.1)
        assert not az_loop.strictly_improves(cand, base)


# ---------------------------------------------------------------------------
# Seed-band disjointness
# ---------------------------------------------------------------------------


class TestSeedBands:
    def test_each_generation_slice_is_disjoint_from_all_bands(self):
        # A range of generations, each with a realistic track count, must not
        # raise (all disjoint from eval / C2 self-play / precursor bands).
        for gen in range(1, 11):
            az_loop._assert_c3_band_disjoint(gen, n_tracks=120)

    def test_c3_base_is_disjoint_from_c2_selfplay_base(self):
        assert az_loop._C3_SELFPLAY_BASE != az_loop._C2_SELFPLAY_BASE
        # Generation-1 slice does not collide with C2's 300_000 band.
        base1 = az_loop._gen_selfplay_base(1)
        assert base1 >= az_loop._C2_SELFPLAY_BASE + 100_000 or \
            base1 + 100_000 <= az_loop._C2_SELFPLAY_BASE

    def test_generation_slices_do_not_overlap_each_other(self):
        bases = [az_loop._gen_selfplay_base(g) for g in range(1, 6)]
        # Strictly increasing by the slice stride; no two slices share a track.
        assert bases == sorted(bases)
        assert all(b2 - b1 >= az_loop._C3_GEN_SLICE for b1, b2 in zip(bases, bases[1:]))

    def test_oversized_track_count_is_rejected(self):
        with pytest.raises(ValueError):
            az_loop._assert_c3_band_disjoint(1, n_tracks=az_loop._C3_GEN_SLICE + 1)


# ---------------------------------------------------------------------------
# Aggregation buffer (real _aggregate)
# ---------------------------------------------------------------------------


class TestAggregateBuffer:
    def test_aggregate_is_row_union_with_sidecar(self, tmp_path):
        p1 = str(tmp_path / "g1.npz")
        p2 = str(tmp_path / "g2.npz")
        _write_tiny_selfplay_npz(p1, 10)
        _write_tiny_selfplay_npz(p2, 7)
        out = str(tmp_path / "agg.npz")
        s = az_loop._aggregate([p1, p2], out)
        assert s["n_rows"] == 17
        merged = np.load(out)
        assert merged["obs"].shape == (17, OBS_DIM)
        assert merged["pi"].shape == (17, ACTION_DIM)
        assert set(np.unique(merged["split"]).tolist()) == {b"train", b"val"}
        # The sidecar exists and carries the live codec version.
        import json
        with open(os.path.splitext(out)[0] + ".selfplay.json") as fh:
            meta = json.load(fh)
        assert meta["codec_version"] == CODEC_VERSION

    def test_recency_window_caps_the_aggregate(self, tmp_path):
        """The loop aggregates only the last ``buffer_window`` generations: an
        older buffer outside the window is NOT included in the union."""
        paths = []
        for g in range(1, 4):  # 3 generations, 5 rows each
            p = str(tmp_path / f"g{g}.npz")
            _write_tiny_selfplay_npz(p, 5)
            paths.append(p)
        window = 2
        # The loop slices buffer_paths[-window:]; emulate that here.
        windowed = paths[-window:]
        out = str(tmp_path / "agg.npz")
        s = az_loop._aggregate(windowed, out)
        # Only 2 of 3 generations (10 rows), not all 3 (15) -- the stale gen aged out.
        assert s["n_sources"] == 2
        assert s["n_rows"] == 10


# ---------------------------------------------------------------------------
# Orchestration: deliberately-worse net is not promoted (incumbent kept)
# ---------------------------------------------------------------------------


class TestOrchestrationPromotion:
    def _stub_steps(self, monkeypatch, tmp_path, gate_seq):
        """Stub generate / aggregate-train / gate so the orchestration is fast.

        ``gate_seq`` is the sequence of (in_search, net_only) LoopGate pairs the
        gate returns: first the warm-prior baseline, then one per generation.
        """
        (tmp_path / "warm.zip").write_bytes(b"warm-model")
        (tmp_path / "warm.meta.json").write_text("{}")

        def fake_generate(*, generation, model_path, out, **kw):
            _write_tiny_selfplay_npz(out, 6)
            return {"n_rows": 6, "pi_entropy_mean": 0.5,
                    "races_dropped_max_rounds": 0}

        def fake_train(*, data, out, **kw):
            # Tag each generation's checkpoint with its filename so we can tell
            # which net was shipped.
            with open(out, "wb") as fh:
                fh.write(os.path.basename(out).encode())
            with open(os.path.splitext(out)[0] + ".meta.json", "w") as fh:
                fh.write("{}")
            return {"best_metric": 1.0, "best_val": {}}

        seq = iter(gate_seq)

        def fake_gate(**kw):
            return next(seq)

        monkeypatch.setattr(az_loop, "_generate", fake_generate)
        monkeypatch.setattr(az_loop, "_train", fake_train)
        monkeypatch.setattr(az_loop, "_gate", fake_gate)

    def test_worse_net_is_not_promoted_incumbent_kept(self, tmp_path, monkeypatch):
        """Both generations produce a net that gates WORSE than the warm prior;
        neither is promoted and the shipped net is the warm prior verbatim."""
        warm = (_gate(0.30, 18.0), _gate(0.40, 19.0))
        worse1 = (_gate(0.90, 30.0), _gate(0.95, 31.0))  # clearly worse
        worse2 = (_gate(0.80, 28.0), _gate(0.85, 29.0))  # still worse
        self._stub_steps(monkeypatch, tmp_path, [warm, worse1, worse2])

        report = az_loop.run_loop(_loop_args(tmp_path, generations=2))

        assert report["generations_run"] == 2
        assert all(not g["promoted"] for g in report["generations"])
        assert report["best_is_warm"] is True
        # The shipped net is the warm prior (best-checkpoint preservation).
        assert open(report["out"], "rb").read() == b"warm-model"

    def test_improving_net_is_promoted_then_best_preserved(self, tmp_path, monkeypatch):
        """Gen 1 strictly improves (promoted); gen 2 regresses (kept). The shipped
        net is gen 1's, not gen 2's worse one."""
        warm = (_gate(0.50, 20.0), _gate(0.60, 21.0))
        better1 = (_gate(0.30, 18.0, spins_hi=0.31, rounds_hi=18.1),
                   _gate(0.40, 19.0))
        worse2 = (_gate(0.70, 25.0), _gate(0.80, 26.0))
        self._stub_steps(monkeypatch, tmp_path, [warm, better1, worse2])

        report = az_loop.run_loop(_loop_args(tmp_path, generations=2))

        assert report["generations"][0]["promoted"] is True
        assert report["generations"][1]["promoted"] is False
        assert report["best_is_warm"] is False
        shipped = open(report["out"], "rb").read()
        assert shipped == b"net_gen1.zip"  # gen 1, not gen 2

    def test_stop_rule_fires_after_consecutive_non_improving(self, tmp_path, monkeypatch):
        """With stop_patience=1, a single non-improving generation halts the loop."""
        warm = (_gate(0.30, 18.0), _gate(0.40, 19.0))
        worse1 = (_gate(0.90, 30.0), _gate(0.95, 31.0))
        # Provide a second generation's gate too in case it were reached.
        worse2 = (_gate(0.80, 28.0), _gate(0.85, 29.0))
        self._stub_steps(monkeypatch, tmp_path, [warm, worse1, worse2])

        report = az_loop.run_loop(_loop_args(tmp_path, generations=5, stop_patience=1))
        assert report["generations_run"] == 1  # stopped after the first non-improver


# ---------------------------------------------------------------------------
# End-to-end smoke: the loop runs >= 2 generations and ships a real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_loop_smoke_end_to_end_produces_contract_checkpoint(tmp_path):
    """A real (un-stubbed) >=2-generation smoke run ships a checkpoint that loads
    as a MaskablePPO and passes the MLAgent contract tripwire.

    Marked slow: it runs real self-play + training on tiny inputs (seconds-minutes
    on CPU). It is the C3 Deliverables "the loop runs >=2 generations end-to-end at
    --smoke and emits a contract-checked checkpoint" guarantee.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model, PPOConfig
    from heat.ml.training import save_checkpoint

    warm = str(tmp_path / "warm.zip")
    save_checkpoint(
        build_model(HeatEnv(num_players=1), PPOConfig(seed=0, device="cpu")),
        warm, track_name="generated", num_players=1, seed=0,
    )

    out = str(tmp_path / "c3_best.zip")
    args = _loop_args(
        tmp_path, warm=warm, out=out, generations=2, tracks=2, val_tracks=2,
        sims=6, gate_games=3, epochs=2, buffer_window=2,
    )
    report = az_loop.run_loop(args)
    assert report["generations_run"] >= 2

    from sb3_contrib import MaskablePPO
    from heat.agents.ml_agent import MLAgent

    model = MaskablePPO.load(out, device="cpu")
    assert model.observation_space.shape == (OBS_DIM,)
    assert model.action_space.n == ACTION_DIM
    MLAgent(out, name="c3")._validate_meta()  # contract tripwire passes
