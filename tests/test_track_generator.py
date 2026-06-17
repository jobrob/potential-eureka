"""Tests for procedural track generation (Sprint 6A, tracks/generator.py).

Gates:
  * Validity: validate_track(generate_track(s)) never raises over many seeds.
  * Determinism: same seed -> identical track; uses only its private RNG
    (global random state does not affect the result -> no global-RNG leak).
  * Distribution sanity: lengths / corner counts span their configured ranges;
    no generated track violates min_corner_gap.
  * Schema conformance: space indices 0..length-1; corners in range; enough
    start positions.
  * track_sampler determinism and seed-derivation.
"""

from __future__ import annotations

import random

import pytest

from heat.ml.spaces import MAX_PLAYERS
from heat.tracks.generator import (
    TrackGenParams,
    TrackSampler,
    generate_track,
    track_sampler,
)
from heat.tracks.validate import TrackValidationError, validate_track

SEEDS = list(range(200))


class TestValidity:
    def test_generated_tracks_are_valid(self) -> None:
        for s in SEEDS:
            track = generate_track(s)
            validate_track(track)  # raises on failure

    def test_default_params_have_high_accept_rate(self) -> None:
        # The default-param generator should produce a valid track for every
        # seed in a wide range (no over-constraint).
        for s in range(50):
            generate_track(s)  # raises RuntimeError if budget exhausted

    def test_over_constrained_params_raise(self) -> None:
        # Retry-exhaustion: a tiny track where a single mandatory corner
        # consumes enough spaces that fewer than MAX_PLAYERS straight start
        # spaces remain. length 8, gap 1 + corner len 3 -> corner at [1..3],
        # straights = {0,4,5,6,7} = 5 < 6 -> the start grid never fits -> the
        # generate-then-validate loop exhausts its budget and raises.
        over = TrackGenParams(
            length_range=(8, 8),
            num_corners_range=(1, 1),
            corner_len_range=(3, 3),
            min_corner_gap=1,
            min_start_positions=MAX_PLAYERS,
        )
        with pytest.raises(RuntimeError):
            generate_track(0, over, max_retries=20)

    def test_invalid_params_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            TrackGenParams(min_start_positions=MAX_PLAYERS - 1)
        with pytest.raises(ValueError):
            TrackGenParams(length_range=(4, 4))  # below MIN_TRACK_LENGTH
        with pytest.raises(ValueError):
            TrackGenParams(speed_limit_choices=())


class TestDeterminism:
    def test_same_seed_identical_track(self) -> None:
        a = generate_track(123)
        b = generate_track(123)
        assert a.name == b.name
        assert a.laps == b.laps
        assert a.spaces == b.spaces
        assert a.corners == b.corners
        assert a.start_positions == b.start_positions

    def test_no_global_rng_leak(self) -> None:
        # Seeding the global random module differently between the two calls
        # must not change the generated track -> proves the generator uses only
        # its private random.Random(seed).
        random.seed(1)
        a = generate_track(42)
        random.seed(987654321)
        b = generate_track(42)
        assert a.spaces == b.spaces
        assert a.corners == b.corners
        assert a.start_positions == b.start_positions

    def test_different_seeds_can_differ(self) -> None:
        layouts = {
            (
                generate_track(s).length,
                tuple((c.start, c.end, c.speed_limit) for c in generate_track(s).corners),
            )
            for s in range(20)
        }
        assert len(layouts) > 1


class TestDistributionSanity:
    def test_lengths_and_corner_counts_span_ranges(self) -> None:
        params = TrackGenParams()
        lengths = set()
        corner_counts = set()
        for s in SEEDS:
            t = generate_track(s, params)
            lengths.add(t.length)
            corner_counts.add(len(t.corners))
        # Not all identical (a constant generator would collapse these to 1).
        assert len(lengths) > 1
        assert len(corner_counts) > 1
        # Lengths stay within the configured range.
        assert min(lengths) >= params.length_range[0]
        assert max(lengths) <= params.length_range[1]

    def test_min_corner_gap_respected(self) -> None:
        params = TrackGenParams()
        for s in SEEDS:
            t = generate_track(s, params)
            corners = sorted(t.corners, key=lambda c: c.start)
            for prev, nxt in zip(corners, corners[1:]):
                gap = nxt.start - prev.end - 1
                assert gap >= params.min_corner_gap, (
                    f"seed {s}: gap {gap} < min_corner_gap "
                    f"{params.min_corner_gap} between {prev} and {nxt}"
                )


class TestSchemaConformance:
    def test_space_indices_consecutive(self) -> None:
        t = generate_track(7)
        assert [s.index for s in t.spaces] == list(range(t.length))

    def test_corners_in_range_and_ordered(self) -> None:
        t = generate_track(7)
        for c in t.corners:
            assert 0 <= c.start <= c.end < t.length
            assert c.speed_limit > 0

    def test_enough_start_positions(self) -> None:
        for s in range(20):
            t = generate_track(s)
            assert len(t.start_positions) >= MAX_PLAYERS

    def test_custom_name_and_laps(self) -> None:
        t = generate_track(3, TrackGenParams(laps=5), name="my-track")
        assert t.name == "my-track"
        assert t.laps == 5

    def test_default_name_includes_seed(self) -> None:
        assert generate_track(99).name == "gen-99"


class TestTrackSampler:
    def test_sampler_seed_determinism(self) -> None:
        sampler = track_sampler()
        a = sampler(7)
        b = sampler(7)
        assert a.spaces == b.spaces
        assert a.corners == b.corners
        assert a.start_positions == b.start_positions

    def test_sampler_seed_derives_from_episode_seed(self) -> None:
        # sampler(seed) should equal generate_track(seed) when base_seed == 0.
        sampler = track_sampler(base_seed=0)
        from_sampler = sampler(55)
        direct = generate_track(55)
        assert from_sampler.corners == direct.corners
        assert from_sampler.start_positions == direct.start_positions

    def test_sampler_none_seed_varies(self) -> None:
        sampler = track_sampler()
        layouts = {
            tuple((c.start, c.end) for c in sampler(None).corners) for _ in range(10)
        }
        # Unseeded resets draw fresh track seeds -> should not all be identical.
        assert len(layouts) > 1

    def test_base_seed_shifts_stream(self) -> None:
        a = track_sampler(base_seed=0)(10)
        b = track_sampler(base_seed=1000)(10)
        # Different base_seed -> generally a different track for the same seed.
        differs = (
            a.corners != b.corners
            or a.start_positions != b.start_positions
            or a.length != b.length
        )
        assert differs

    def test_track_sampler_returns_picklable_class(self) -> None:
        assert isinstance(track_sampler(), TrackSampler)

    def test_sampler_picklable_and_consistent_after_pickle(self) -> None:
        # The sampler crosses the SubprocVecEnv (spawn) process boundary, so it
        # must pickle and reproduce the same seeded tracks afterwards.
        import pickle

        sampler = TrackSampler(base_seed=3)
        restored = pickle.loads(pickle.dumps(sampler))
        a = sampler(42)
        b = restored(42)
        assert a.corners == b.corners
        assert a.start_positions == b.start_positions
        assert a.length == b.length
