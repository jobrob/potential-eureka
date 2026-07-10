"""Tests for the A5 anti-collapse self-play recipe (§6 of the A5 design).

Covers:
  1. SnapshotAgent legality + degenerate handling (a collector game vs a
     snapshot completes with zero illegal actions; forced decisions resolve).
  2. Snapshot immutability (pool push deep-copies; live training does not mutate
     a pool entry's state_dict).
  3. SnapshotPool mechanics (capacity eviction, oldest() identity, uniform
     sample).
  4. EntropyController (below floor rises + caps; above floor*1.2 decays toward
     but never below base; between leaves unchanged).
  5. Stage-1 (absurd threshold raises Stage1ValidationError; disabled does not).
  6. train_selfplay_a5 smoke (tiny run completes, pool populated, eval records
     with finite fields, finite losses).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.multiseat import MultiSeatCollector
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.recipe import (
    A5Config,
    EntropyController,
    Stage1ValidationError,
    train_selfplay_a5,
)
from heat.ml.selfplay.snapshots import SnapshotAgent, SnapshotPool
from heat.ml.selfplay.tiny_heat import tiny_heat_track

DEVICE = torch.device("cpu")


def _policy(seed: int = 0) -> HeatPolicy:
    """A tiny fresh policy (small trunk keeps the sweeps fast)."""
    torch.manual_seed(seed)
    return HeatPolicy(hidden_sizes=(16, 16))


def _config(**overrides: object) -> A5Config:
    """A tiny A5Config for fast tests."""
    defaults: dict[str, object] = {
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 2,
        "total_timesteps": 384,
        "hidden_sizes": (16, 16),
        "num_players": 2,
        "seed": 0,
        "device": "cpu",
    }
    defaults.update(overrides)
    return A5Config(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. SnapshotAgent legality + degenerate handling
# ---------------------------------------------------------------------------


class _LegalityCollector(MultiSeatCollector):
    """Collector that asserts every stored action was legal under its mask."""

    def _store(  # type: ignore[override]
        self,
        buffers: list[RolloutBuffer],
        seat: int,
        pending: object,
        reward: float,
        done: bool,
    ) -> None:
        p = pending
        assert bool(p.mask[p.action]), (  # type: ignore[attr-defined]
            f"seat {seat} sampled illegal action {p.action}"  # type: ignore[attr-defined]
        )
        super()._store(buffers, seat, pending, reward, done)  # type: ignore[arg-type]


def test_snapshot_agent_legality_and_completion() -> None:
    """A collector game with a SnapshotAgent opponent in seat 1 completes: the
    recorded seat's every action is masked-legal, and its stream ends done=True
    (forced/degenerate decisions -- e.g. empty-hand CARDS -- resolve internally
    via forced_action, so the snapshot never emits an illegal move)."""
    snapshot = SnapshotAgent(_policy(1), name="snap")
    collector = _LegalityCollector(
        tiny_heat_track(), 2, scripted_seats={1: snapshot}
    )
    buffers, returns = collector.collect(
        _policy(0), 200, DEVICE, np.random.default_rng(0), gamma=0.99
    )
    assert len(buffers[1]) == 0, "scripted snapshot seat must not be recorded"
    assert len(buffers[0]) > 0, "policy seat must be recorded"
    assert bool(buffers[0].dones[len(buffers[0]) - 1]), "stream must end done=True"
    assert returns  # at least one completed game


# ---------------------------------------------------------------------------
# 2. Snapshot immutability
# ---------------------------------------------------------------------------


def test_snapshot_immutability_under_training() -> None:
    """A pushed snapshot deep-copies the policy: mutating the live policy's
    parameters afterwards does not change the pool entry's parameters."""
    policy = _policy(0)
    pool = SnapshotPool(capacity=5)
    pool.push(policy, "iter1")
    snap = pool.oldest()
    before = copy.deepcopy(snap._policy.state_dict())

    # Simulate a training step: perturb every live parameter.
    with torch.no_grad():
        for param in policy.parameters():
            param.add_(1.0)

    after = snap._policy.state_dict()
    for key, tensor in before.items():
        assert torch.equal(tensor, after[key]), (
            f"pool entry parameter {key} changed after live-policy mutation"
        )
    # And the live policy really did change (guards against a no-op test).
    live = policy.state_dict()
    assert any(
        not torch.equal(before[k], live[k]) for k in before
    ), "live policy was not actually mutated"


# ---------------------------------------------------------------------------
# 3. SnapshotPool mechanics
# ---------------------------------------------------------------------------


def test_snapshot_pool_capacity_and_oldest() -> None:
    """Capacity evicts the oldest; oldest() returns the earliest survivor."""
    pool = SnapshotPool(capacity=3)
    for i in range(5):
        pool.push(_policy(i), f"iter{i}")
    assert len(pool) == 3, "pool must not exceed capacity"
    # After pushing iter0..iter4 with capacity 3, survivors are iter2,3,4.
    assert pool.oldest().name == "iter2"


def test_snapshot_pool_uniform_sample() -> None:
    """sample() draws uniformly across all pool entries (statistical smoke)."""
    pool = SnapshotPool(capacity=4)
    for i in range(4):
        pool.push(_policy(i), f"iter{i}")
    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(400):
        name = pool.sample(rng).name
        counts[name] = counts.get(name, 0) + 1
    assert len(counts) == 4, "every entry should be sampled at least once"
    # Uniform over 4 entries -> ~100 each in 400 draws; loose bounds.
    for name, c in counts.items():
        assert 50 <= c <= 160, f"entry {name} sampled {c} times (non-uniform)"


# ---------------------------------------------------------------------------
# 4. EntropyController
# ---------------------------------------------------------------------------


def test_entropy_controller_parachute_semantics() -> None:
    """Below floor the coef rises and caps at ent_coef_max; above floor*1.2 it
    decays toward but never below base; in the band it stays unchanged."""
    config = _config(
        ent_coef=0.01, entropy_floor=0.40, ent_scale_up=1.5,
        ent_scale_down=0.98, ent_coef_max=0.10,
    )
    ctrl = EntropyController(config)
    assert ctrl.current_ent_coef == 0.01

    # Below floor: rises by 1.5x, engaged True.
    ctrl.update(0.30)
    assert ctrl.engaged
    assert abs(ctrl.current_ent_coef - 0.015) < 1e-9

    # Repeatedly below floor: caps at ent_coef_max, never exceeds it.
    for _ in range(50):
        ctrl.update(0.10)
    assert ctrl.current_ent_coef <= 0.10 + 1e-12
    assert abs(ctrl.current_ent_coef - 0.10) < 1e-9

    # Well above floor*1.2 (0.48): decays by 0.98x, not engaged.
    ctrl.update(1.0)
    assert not ctrl.engaged
    assert abs(ctrl.current_ent_coef - 0.098) < 1e-9

    # Many recovered steps: decays toward but never below the base coef.
    for _ in range(1000):
        ctrl.update(1.0)
    assert ctrl.current_ent_coef >= 0.01 - 1e-12
    assert abs(ctrl.current_ent_coef - 0.01) < 1e-9

    # In the band [floor, floor*1.2] = [0.40, 0.48]: unchanged.
    ctrl.update(0.44)
    assert not ctrl.engaged
    assert abs(ctrl.current_ent_coef - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# 5. Stage-1 validation
# ---------------------------------------------------------------------------


def test_stage1_raises_on_absurd_threshold() -> None:
    """An impossible stage1_min_winrate forces a Stage1ValidationError at the
    Stage-1 iteration, carrying diagnostics."""
    config = _config(
        n_steps=64, total_timesteps=64 * 3, eval_games=6,
        stage1_enabled=True, stage1_iter=2, stage1_min_winrate=1.01,
        snapshot_every=1,
    )
    try:
        train_selfplay_a5(config, track=tiny_heat_track())
    except Stage1ValidationError as exc:
        assert exc.diagnostics["iteration"] == 2.0
        assert "winrate_vs_weak" in exc.diagnostics
    else:  # pragma: no cover - the check must fire
        raise AssertionError("Stage1ValidationError was not raised")


def test_stage1_disabled_does_not_raise() -> None:
    """With stage1_enabled=False the loop completes even with an impossible
    threshold."""
    config = _config(
        n_steps=64, total_timesteps=64 * 3, eval_games=6,
        stage1_enabled=False, stage1_iter=2, stage1_min_winrate=1.01,
        snapshot_every=1, eval_every=2,
    )
    policy, records = train_selfplay_a5(config, track=tiny_heat_track())
    assert isinstance(policy, HeatPolicy)
    assert records  # at least one eval landed


# ---------------------------------------------------------------------------
# 6. train_selfplay_a5 smoke
# ---------------------------------------------------------------------------


def test_train_selfplay_a5_smoke() -> None:
    """3 tiny iterations: completes, pool populated, >=1 eval record with finite
    fields, finite losses."""
    config = _config(
        n_steps=128, total_timesteps=128 * 3, eval_games=6,
        snapshot_every=1, eval_every=2, stage1_enabled=False,
    )
    infos: list[dict[str, float]] = []
    policy, records = train_selfplay_a5(
        config, track=tiny_heat_track(),
        on_iteration=lambda it, info: infos.append(info),
    )
    assert isinstance(policy, HeatPolicy)
    assert len(infos) == 3, "one callback per iteration"
    for info in infos:
        for key in ("policy_loss", "value_loss", "entropy", "ent_coef"):
            assert np.isfinite(info[key]), f"{key} not finite: {info[key]}"

    assert records, "at least one eval record expected"
    for rec in records:
        for key in (
            "winrate_vs_weak", "return_vs_weak", "winrate_vs_oldest",
            "return_vs_oldest", "entropy", "ent_coef", "min_entropy",
        ):
            assert np.isfinite(rec[key]), f"eval record {key} not finite"
        assert 0.0 <= rec["winrate_vs_weak"] <= 1.0
        assert 0.0 <= rec["winrate_vs_oldest"] <= 1.0
