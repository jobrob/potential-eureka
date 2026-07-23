"""Focused contracts for the T0 frozen-static matchup map."""

from __future__ import annotations

import pytest

from experiments.eval_static_matchup_map import (
    AGENTS_BY_ID,
    FROZEN_AGENTS,
    V2_AGENTS,
    _cell_key,
    _agent_style,
    _game_seed,
    _parse_args,
    race_specs,
    summarize_races,
)
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents.static_v2 import (
    HeuristicV2Agent,
    RepairedHeuristicV2Agent,
    StaticSearchV2Agent,
)


def test_t0_frozen_population_has_explicit_configuration_identities() -> None:
    """T0 cannot silently add a tunable or unversioned strategic bot."""
    assert tuple(spec.agent_id for spec in FROZEN_AGENTS) == (
        "heuristic_weak_v1",
        "heuristic_repaired_v1",
        "static_search_v1",
    )
    assert isinstance(AGENTS_BY_ID["heuristic_weak_v1"].factory(), HeuristicAgent)
    repaired = AGENTS_BY_ID["heuristic_repaired_v1"].factory()
    assert isinstance(repaired, StrongHeuristicAgent)
    assert repaired.strength == 2
    assert repaired._avoid_certain_spins is True
    assert isinstance(AGENTS_BY_ID["static_search_v1"].factory(), StaticSearchAgent)


def test_t0_defaults_are_bounded_and_cover_every_directed_cell() -> None:
    """The default campaign is one full track-by-start-seat screen."""
    args = _parse_args([])
    assert args.tracks == 12
    assert args.repeats == 1
    assert args.max_seconds == 1_800.0
    assert len(FROZEN_AGENTS) * (len(FROZEN_AGENTS) - 1) * 2 * 3 == 36


def test_v2_population_is_versioned_without_changing_v1_defaults() -> None:
    """The repaired population is explicit and writes to a separate artifact."""
    assert tuple(spec.agent_id for spec in V2_AGENTS) == (
        "heuristic_weak_v2",
        "heuristic_repaired_v2",
        "static_search_v2",
    )
    assert isinstance(V2_AGENTS[0].factory(), HeuristicV2Agent)
    assert isinstance(V2_AGENTS[1].factory(), RepairedHeuristicV2Agent)
    assert isinstance(V2_AGENTS[2].factory(), StaticSearchV2Agent)
    assert _parse_args([]).out.name == "t0_static_matchup_map.json"
    assert _parse_args(["--population", "v2"]).out.name == (
        "t0_static_matchup_map_v2.json"
    )
    assert _parse_args(
        ["--population", "v2", "--opponent-population", "v1"]
    ).out.name == "t0_static_matchup_map_v2_vs_v1.json"
    assert _agent_style("static_search_v2") == _agent_style("static_search_v1")
    assert _parse_args(["--focal-style", "static_search"]).focal_style == (
        "static_search"
    )


def test_t0_rejects_sizes_that_would_break_seed_coordinate_fields() -> None:
    """Track/repeat bounds preserve unique decimal fields in matched seeds."""
    with pytest.raises(SystemExit):
        _parse_args(["--tracks", "101"])
    with pytest.raises(SystemExit):
        _parse_args(["--repeats", "11"])


def test_race_specs_cross_tracks_and_starting_seats_with_matched_seeds() -> None:
    """Each track/seat coordinate appears once and its seed ignores matchup."""
    specs = race_specs(
        family="default_generated",
        family_index=0,
        track_base=715_000,
        tracks=3,
        seat_count=4,
        repeats=2,
        seed=151,
    )
    assert len(specs) == 3 * 4 * 2
    assert len(
        {(row.track_seed, row.focal_seat, row.repeat) for row in specs}
    ) == len(specs)
    assert specs[0].game_seed == _game_seed(151, 0, 4, 0, 0, 0)


def test_cell_summary_preserves_starting_seat_slices_and_search_latency() -> None:
    """Raw races retain the regime and timing detail T0 exists to expose."""
    races = [
        {
            "won": True,
            "focal_seat": 0,
            "seat_count": 2,
            "placement_reward": 1.0,
            "rounds": 10,
            "elapsed_seconds": 0.2,
            "focal_search": {"moves": 10, "seconds": 0.1, "clones": 100},
            "field_search": {"moves": 0, "seconds": 0.0, "clones": 0},
        },
        {
            "won": False,
            "focal_seat": 1,
            "seat_count": 2,
            "placement_reward": -1.0,
            "rounds": 12,
            "elapsed_seconds": 0.4,
            "focal_search": {"moves": 20, "seconds": 0.2, "clones": 200},
            "field_search": {"moves": 0, "seconds": 0.0, "clones": 0},
        },
    ]
    summary = summarize_races(races)
    assert summary["games"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["mean_placement_reward"] == 0.0
    assert summary["mean_rounds"] == 11
    assert summary["focal_search_ms_per_move"] == pytest.approx(10.0)
    assert set(summary["starting_seats"]) == {"0", "1"}
    assert _cell_key("a", "b", "tight", 6) == "a|b|tight|6"


def test_empty_cell_cannot_be_misreported() -> None:
    """A stopped campaign cannot fabricate a zero-game result."""
    with pytest.raises(ValueError, match="empty"):
        summarize_races([])
