"""Tests for the A8 full-rules domain-randomized Phase-1 baseline."""

from __future__ import annotations

import numpy as np
import pytest

from heat.ml.selfplay.phase1 import (
    A8Config,
    SeatCountSchedule,
    train_selfplay_a8,
    training_track_source,
)
from heat.ml.selfplay.eval_harness import held_out_tracks
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.snapshots import SnapshotAgent
from scripts.train_selfplay_a8 import _evaluation_opponents, _parse_args


def _fingerprint(track: object) -> tuple:
    return (
        len(track.spaces),  # type: ignore[attr-defined]
        track.laps,  # type: ignore[attr-defined]
        tuple(s.lanes for s in track.spaces),  # type: ignore[attr-defined]
        tuple((c.start, c.end, c.speed_limit) for c in track.corners),  # type: ignore[attr-defined]
        tuple(track.start_positions),  # type: ignore[attr-defined]
    )


def test_a8_adopts_s1_five_epoch_default() -> None:
    """A8 uses the S1 winner by default while retaining an explicit override."""
    assert A8Config().n_epochs == 5
    assert _parse_args([]).n_epochs == 5
    assert _parse_args(["--n-epochs", "3"]).n_epochs == 3


def test_a8_anchor_cli_is_opt_in() -> None:
    """The S2 treatment stays off unless both anchor options are supplied."""
    assert A8Config().anchor_share == 0.0
    args = _parse_args(
        ["--anchor-checkpoint", "anchor.pt", "--anchor-share", "0.4"]
    )
    assert args.anchor_checkpoint == "anchor.pt"
    assert args.anchor_share == pytest.approx(0.4)


def test_a8_collector_cli_keeps_scalar_reference_and_allows_phase() -> None:
    """T2 is opt-in until its throughput gate adopts a new default."""
    assert A8Config().collector_mode == "scalar"
    assert _parse_args([]).collector == "scalar"
    assert _parse_args(["--collector", "phase"]).collector == "phase"


def test_a8_evaluation_suite_keeps_failed_search_candidate_opt_in() -> None:
    """B1's failed candidate cannot silently become the default strong rung."""
    assert _parse_args([]).eval_search_candidate is False
    assert _parse_args(["--eval-search-candidate"]).eval_search_candidate is True
    opponents = _evaluation_opponents(None)
    assert tuple(opponents) == (
        "weak",
        "strong",
        "heuristic_repaired",
    )
    with_search = _evaluation_opponents(None, include_search_candidate=True)
    assert with_search["static_search_v1_candidate"]().name == "StaticSearchV1"


def test_a8_config_rejects_invalid_domains() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        A8Config(seat_counts=())
    with pytest.raises(ValueError, match="2..6"):
        A8Config(seat_counts=(1, 2))
    with pytest.raises(ValueError, match="duplicates"):
        A8Config(seat_counts=(2, 2))
    with pytest.raises(ValueError, match="reserved"):
        A8Config(track_base_seed=0)
    with pytest.raises(ValueError, match="anchor_share"):
        A8Config(anchor_share=1.1)


def test_balanced_schedule_covers_every_seat_once_per_block() -> None:
    schedule = SeatCountSchedule((2, 3, 4, 5, 6), np.random.default_rng(7))
    first = [schedule.next() for _ in range(5)]
    second = [schedule.next() for _ in range(5)]
    assert sorted(first) == [2, 3, 4, 5, 6]
    assert sorted(second) == [2, 3, 4, 5, 6]


def test_training_namespace_is_disjoint_from_heldout_sample() -> None:
    source = training_track_source(A8Config(track_base_seed=80_008))
    training = {_fingerprint(source(seed)) for seed in range(30)}
    heldout = {_fingerprint(track) for track in held_out_tracks(10)}
    assert not training & heldout


def test_a8_training_smoke_covers_variable_seats() -> None:
    seen: list[dict[str, float]] = []
    config = A8Config(
        seat_counts=(2, 3),
        total_timesteps=64,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        snapshot_every=1,
        pool_prob=0.0,
        collector_mode="phase",
        device="cpu",
        seed=0,
    )
    policy, records = train_selfplay_a8(
        config, on_iteration=lambda _iteration, record: seen.append(record)
    )
    assert policy.encoder == "flat"
    assert len(records) == 2
    assert {int(record["seat_count"]) for record in records} == {2, 3}
    assert len(seen) == len(records)
    assert records[-1]["total_games"] >= 2
    for record in records:
        assert record["games"] >= 1
        assert record["n_recorded"] > 0
        for key in ("policy_loss", "value_loss", "entropy"):
            assert np.isfinite(record[key])


def test_a8_fixed_anchor_replaces_snapshot_iterations() -> None:
    """A 100% anchor share leaves the initial pool warm-up as current-self."""
    config = A8Config(
        seat_counts=(2,),
        total_timesteps=64,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        snapshot_every=1,
        pool_prob=1.0,
        anchor_share=1.0,
        device="cpu",
        seed=8,
    )
    anchor = SnapshotAgent(build_policy(config), name="test-anchor")
    _policy, records = train_selfplay_a8(config, anchor=anchor)
    final = records[-1]
    assert final["opponent_current_iterations"] == 1.0
    assert final["opponent_snapshot_iterations"] == 0.0
    assert final["opponent_anchor_iterations"] == len(records) - 1


def test_a8_milestone_checkpoints_emit_once_after_crossing() -> None:
    emitted: list[tuple[int, float]] = []
    config = A8Config(
        seat_counts=(2,),
        total_timesteps=64,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        pool_prob=0.0,
        device="cpu",
        seed=4,
    )
    _policy, records = train_selfplay_a8(
        config,
        checkpoint_steps=(1, 10),
        on_checkpoint=lambda step, _policy, record: emitted.append(
            (step, record["steps"])
        ),
    )
    assert [step for step, _actual in emitted] == [1, 10]
    assert all(actual >= step for step, actual in emitted)
    assert sum("checkpoint_step" in record for record in records) == 1


def test_a8_profile_reports_consistent_phase_times() -> None:
    """Opt-in P0 timing is finite, non-negative, and nested within rollout."""
    config = A8Config(
        seat_counts=(2,),
        total_timesteps=32,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        hidden_sizes=(16,),
        pool_prob=0.0,
        device="cpu",
        seed=5,
    )
    _policy, records = train_selfplay_a8(config, profile=True)
    record = records[0]
    keys = (
        "rollout_seconds",
        "encoding_seconds",
        "inference_seconds",
        "prepare_seconds",
        "update_seconds",
        "iteration_seconds",
    )
    assert all(np.isfinite(record[key]) and record[key] >= 0.0 for key in keys)
    assert record["encoding_seconds"] + record["inference_seconds"] <= record[
        "rollout_seconds"
    ]
    assert record["live_decisions"] == record["action_inference_rows"]
    assert record["simultaneous_live_decisions"] > 0
    assert record["available_phase_rows"] >= record["available_phase_count"]
    assert (
        record["rollout_seconds"]
        + record["prepare_seconds"]
        + record["update_seconds"]
        <= record["iteration_seconds"]
    )
