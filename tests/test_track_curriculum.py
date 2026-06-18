"""Tests for the Sprint B step-aware track-difficulty curriculum (Idea 2).

Covers ``CurriculumSchedule`` / ``StepAwareTrackSampler`` per the design §5.5:
endpoint correctness, monotone interpolation, validity at every fraction,
picklability (the SubprocVecEnv requirement), and step plumbing.
"""

from __future__ import annotations

import pickle

import pytest

from heat.tracks.validate import MIN_TRACK_LENGTH, validate_track
from heat.tracks.generator import (
    CurriculumSchedule,
    StepAwareTrackSampler,
    TrackGenParams,
    default_curriculum_schedule,
    easy_track_params,
    generate_track,
)

HORIZON = 1_000_000


def _schedule(horizon: int = HORIZON) -> CurriculumSchedule:
    return default_curriculum_schedule(horizon)


class TestEndpoints:
    def test_step_zero_is_easy_equivalent(self) -> None:
        sched = _schedule()
        easy = easy_track_params()
        p0 = sched.params_at(0)
        # The interpolated bounds at step 0 equal the easy endpoint's bounds.
        assert p0.length_range == easy.length_range
        assert p0.num_corners_range == easy.num_corners_range
        assert p0.corner_len_range == easy.corner_len_range
        assert p0.laps == easy.laps

    def test_step_horizon_is_full_default(self) -> None:
        sched = _schedule()
        full = TrackGenParams()
        ph = sched.params_at(HORIZON)
        assert ph.length_range == full.length_range
        assert ph.num_corners_range == full.num_corners_range
        assert ph.corner_len_range == full.corner_len_range
        assert ph.laps == full.laps

    def test_beyond_horizon_clamps_to_full(self) -> None:
        sched = _schedule()
        full = TrackGenParams()
        ph = sched.params_at(2 * HORIZON)
        assert ph.length_range == full.length_range
        assert ph.num_corners_range == full.num_corners_range
        assert ph.laps == full.laps

    def test_zero_horizon_is_full(self) -> None:
        # A non-positive horizon means "no ramp" -> full difficulty immediately.
        sched = CurriculumSchedule(
            easy=easy_track_params(), full=TrackGenParams(), horizon_steps=0
        )
        assert sched.params_at(0).length_range == TrackGenParams().length_range


class TestMonotone:
    def test_bounds_non_decreasing_in_step(self) -> None:
        sched = _schedule()
        steps = [int(f * HORIZON) for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
        params = [sched.params_at(s) for s in steps]
        for a, b in zip(params, params[1:]):
            assert b.length_range[1] >= a.length_range[1]
            assert b.num_corners_range[1] >= a.num_corners_range[1]
            assert b.laps >= a.laps


class TestValidityAtEveryFraction:
    def test_constructs_and_generates_at_each_fraction(self) -> None:
        sched = _schedule()
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            step = int(frac * HORIZON)
            params = sched.params_at(step)  # __post_init__ must not raise
            # Easy endpoint floor: never below MIN_TRACK_LENGTH.
            assert params.length_range[0] >= MIN_TRACK_LENGTH
            # Ranges are non-inverted.
            assert params.length_range[0] <= params.length_range[1]
            assert params.num_corners_range[0] <= params.num_corners_range[1]
            assert params.corner_len_range[0] <= params.corner_len_range[1]
            # And a real track generates + validates from these bounds.
            track = generate_track(12345, params)
            validate_track(track)

    def test_easy_endpoint_floor(self) -> None:
        assert easy_track_params().length_range[0] >= MIN_TRACK_LENGTH


class TestPicklability:
    def test_sampler_round_trips_and_samples(self) -> None:
        sampler = StepAwareTrackSampler(_schedule(), base_seed=3, step=123)
        blob = pickle.dumps(sampler)
        restored = pickle.loads(blob)
        assert restored._step == 123
        # Still samples a valid track after the round-trip (same seed -> same).
        t1 = sampler(seed=7)
        t2 = restored(seed=7)
        validate_track(t1)
        validate_track(t2)
        assert t1.length == t2.length
        assert len(t1.corners) == len(t2.corners)


class TestStepPlumbing:
    def test_set_step_changes_difficulty_distribution(self) -> None:
        sampler = StepAwareTrackSampler(_schedule(), base_seed=0)

        def mean_corner_count(s: StepAwareTrackSampler, n: int = 40) -> float:
            total = 0
            for seed in range(n):
                total += len(s(seed=seed).corners)
            return total / n

        sampler.set_step(0)
        easy_mean = mean_corner_count(sampler)
        sampler.set_step(HORIZON)
        full_mean = mean_corner_count(sampler)
        # The full distribution allows more corners (num_corners_range widens to
        # (3, 7) from the easy (2, 3)), so the mean strictly rises.
        assert full_mean > easy_mean

    def test_call_uses_params_at_current_step(self) -> None:
        sampler = StepAwareTrackSampler(_schedule(), base_seed=0, step=0)
        assert sampler.params_at_current() == sampler.schedule.params_at(0)
        sampler.set_step(HORIZON)
        assert sampler.params_at_current() == sampler.schedule.params_at(HORIZON)
