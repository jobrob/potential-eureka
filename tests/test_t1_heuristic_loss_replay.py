"""Focused contracts for T1 repaired-heuristic loss replay."""

from __future__ import annotations

from typing import Any

import pytest

from experiments.diag_t1_heuristic_losses import (
    FAMILIES,
    OPPONENT_IDS,
    SEAT_COUNTS,
    TracingRepairedAgent,
    LossSpec,
    _classify_turns,
    _manual_review_ids,
    render_narrative,
    run_replay,
    select_losses,
)
from heat.engine import rules
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _t0_payload() -> dict[str, Any]:
    """Build a small complete T0-shaped fixture with three losses per stratum."""
    cells = []
    ordinal = 0
    for family in FAMILIES:
        for opponent in OPPONENT_IDS:
            for seats in SEAT_COUNTS:
                races = []
                for index in range(4):
                    races.append(
                        {
                            "won": index == 3,
                            "place": 2 if index < 3 else 1,
                            "rounds": 20 + index,
                            "track_seed": 700_000 + ordinal * 10 + index,
                            "game_seed": 800_000 + ordinal * 10 + index,
                            "focal_seat": index % seats,
                        }
                    )
                key = f"heuristic_repaired_v1|{opponent}|{family}|{seats}"
                cells.append(
                    {
                        "key": key,
                        "focal": "heuristic_repaired_v1",
                        "opponent": opponent,
                        "family": family,
                        "seat_count": seats,
                        "races": races,
                    }
                )
                ordinal += 1
    return {"complete": True, "cells": cells}


def test_t1_selection_is_stable_balanced_and_loss_only() -> None:
    """Every opponent/family/seat stratum contributes the same fixed losses."""
    payload = _t0_payload()
    first = select_losses(payload, per_stratum=2, sample_seed=9)
    assert first == select_losses(payload, per_stratum=2, sample_seed=9)
    assert len(first) == 2 * 12
    counts: dict[tuple[str, str, int], int] = {}
    for spec in first:
        key = (spec.family, spec.opponent, spec.seats)
        counts[key] = counts.get(key, 0) + 1
        assert spec.expected_place > 1
    assert set(counts.values()) == {2}


def test_t1_rejects_incomplete_or_underfilled_t0_data() -> None:
    """T1 cannot silently interpret partial or inadequately sampled strata."""
    with pytest.raises(ValueError, match="complete"):
        select_losses({"complete": False, "cells": []})
    with pytest.raises(ValueError, match="needs 4"):
        select_losses(_t0_payload(), per_stratum=4)


def test_tracer_planner_audit_ranks_the_actual_joint_plan_best() -> None:
    """Diagnostic planner math must reproduce the production argmax."""
    state = GameState.create(generate_track(715_000), 2, seed=1)
    for player in state.players:
        player.lap = 1
    tracer = TracingRepairedAgent()
    legal = rules.legal_gear_shifts(
        state.get_player(0).gear, state.get_player(0).heat_available
    )
    tracer.choose_gear(state, 0, legal)
    planner = tracer.records[-1]["planner"]
    assert planner["chosen"] == planner["top"][0]
    assert planner["chosen"]["eligible"] is True


def test_final_straight_audit_agrees_with_no_future_corner_control() -> None:
    """Finish-aware production math matches the no-future-corner control."""
    state = GameState.create(generate_track(715_001), 2, seed=1)
    player = state.get_player(0)
    player.lap = state.track.laps
    player.position = max(corner.end for corner in state.track.corners) + 1
    player.heat_pool.clear()
    tracer = TracingRepairedAgent()
    legal = rules.legal_gear_shifts(player.gear, player.heat_available)
    tracer.choose_gear(state, 0, legal)
    planner = tracer.records[-1]["planner"]
    assert planner["chosen"]["gear"] == 2
    assert planner["no_future_solvency_best"]["gear"] == 2
    assert planner["no_future_solvency_regret"] == 0.0


def test_tracer_detects_same_size_hand_cache_staleness() -> None:
    """The production cache signature omits hand contents, which T1 records."""
    state = GameState.create(generate_track(715_001), 2, seed=1)
    player = state.get_player(0)
    tracer = TracingRepairedAgent()
    legal = rules.legal_gear_shifts(player.gear, player.heat_available)
    tracer.choose_gear(state, 0, legal)
    player.hand = [
        Card(CardType.STRESS, 0, f"replacement-{index}")
        for index in range(len(player.hand))
    ]
    tracer.choose_gear(state, 0, legal)
    assert tracer.records[-1]["planner"]["stale_cached_plan"] is True


def test_replay_records_ineligible_stale_cached_play() -> None:
    """A cached plan may bypass the current guaranteed-spin eligibility set."""
    replay = run_replay(
        LossSpec(
            source_key="fixture",
            family="tight_generated",
            opponent="heuristic_weak_v1",
            seats=6,
            track_seed=915_007,
            game_seed=1_511_607_050,
            focal_seat=5,
            expected_place=6,
            expected_rounds=40,
            sample_rank=0,
        )
    )
    assert "stale_cached_gear_plan" in replay["signals"]
    assert "planner_cache_regret" in replay["signals"]
    assert any(
        turn["decisions"]["gear"]["planner"]["chosen"]["eligible"] is False
        for turn in replay["turns"]
        if "gear" in turn["decisions"]
    )


def test_turn_classifier_separates_agent_anomaly_from_traffic_context() -> None:
    """Traffic is counted but does not inflate the manual suspicion score."""
    turns = [
        {"flags": ["finish_underpush", "traffic_blocking"]},
        {"flags": ["traffic_blocking"]},
    ]
    signals, score = _classify_turns(turns, final_heat=4)
    assert signals == ["finish_underpush", "traffic_blocking", "stranded_heat"]
    assert score == 5


def test_manual_set_keeps_baselines_and_adds_highest_unread_scores() -> None:
    """Manual reading is not limited to classifier-selected suspicious logs."""
    replays = []
    for index in range(20):
        replays.append(
            {
                "replay_id": f"r{index}",
                "spec": {"sample_rank": 0 if index < 12 else 1},
                "anomaly_score": index,
            }
        )
    selected = _manual_review_ids(replays)
    assert selected[:12] == [f"r{index}" for index in range(12)]
    assert selected[12:] == ["r19", "r18", "r17", "r16", "r15", "r14"]


def test_manual_narrative_names_coordinate_outcome_and_flag() -> None:
    """A reviewer can identify the exact odd turn without opening raw JSON."""
    replay = {
        "replay_id": "demo",
        "spec": {
            "family": "tight_generated",
            "opponent": "heuristic_weak_v1",
            "seats": 2,
            "track_seed": 1,
            "game_seed": 2,
            "focal_seat": 0,
        },
        "place": 2,
        "rounds": 10,
        "final_heat": 4,
        "signals": ["finish_underpush"],
        "anomaly_score": 4,
        "turns": [],
    }
    text = render_narrative(replay)
    assert "track=1 game=2 focal_seat=0" in text
    assert "outcome=place 2/2" in text
    assert "finish_underpush" in text
