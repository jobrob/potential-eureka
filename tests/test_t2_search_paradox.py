"""Focused contracts for the T2 paired search-paradox diagnostic."""

from __future__ import annotations

from typing import Any

import pytest

from experiments.diag_t2_search_paradox import (
    FAMILIES,
    SEAT_COUNTS,
    TracingStaticSearchAgent,
    _manual_review_ids,
    select_pairs,
)
from heat.agents.static_search import StaticSearchAgent
from heat.engine import rules
from heat.models.game_state import GameState
from heat.tracks.generator import generate_track


def _t0_payload() -> dict[str, Any]:
    """Build a complete T0-shaped fixture with four paradox pairs per stratum."""
    cells = []
    ordinal = 0
    for family in FAMILIES:
        for seats in SEAT_COUNTS:
            for opponent in ("heuristic_weak_v1", "heuristic_repaired_v1"):
                races = []
                for index in range(5):
                    weak = opponent == "heuristic_weak_v1"
                    races.append(
                        {
                            "won": (not weak) if index < 4 else weak,
                            "place": 2 if weak and index < 4 else 1,
                            "rounds": 20 + index,
                            "track_seed": 700_000 + ordinal * 10 + index,
                            "game_seed": 800_000 + ordinal * 10 + index,
                            "focal_seat": index % seats,
                        }
                    )
                # Coordinates must match across opponent cells.
                if opponent == "heuristic_repaired_v1":
                    previous = cells[-1]["races"]
                    for row, source in zip(races, previous, strict=True):
                        row["track_seed"] = source["track_seed"]
                        row["game_seed"] = source["game_seed"]
                        row["focal_seat"] = source["focal_seat"]
                cells.append(
                    {
                        "focal": "static_search_v1",
                        "opponent": opponent,
                        "family": family,
                        "seat_count": seats,
                        "races": races,
                    }
                )
            ordinal += 1
    return {"complete": True, "cells": cells}


def test_pair_selection_is_stable_balanced_and_exactly_matched() -> None:
    """Every family/seat stratum contributes matched opposite outcomes."""
    payload = _t0_payload()
    first = select_pairs(payload, per_stratum=3, sample_seed=9)
    assert first == select_pairs(payload, per_stratum=3, sample_seed=9)
    assert len(first) == 3 * 6
    counts: dict[tuple[str, int], int] = {}
    for spec in first:
        key = (spec.family, spec.seats)
        counts[key] = counts.get(key, 0) + 1
        assert spec.repaired_place == 1
        assert spec.weak_place > 1
    assert set(counts.values()) == {3}


def test_pair_selection_rejects_partial_or_underfilled_input() -> None:
    """T2 must not weaken its paired-coordinate contract silently."""
    with pytest.raises(ValueError, match="complete"):
        select_pairs({"complete": False, "cells": []})
    with pytest.raises(ValueError, match="needs 5"):
        select_pairs(_t0_payload(), per_stratum=5)


def test_trace_score_and_plan_match_uninstrumented_search() -> None:
    """Instrumentation reproduces the production score and top-six choice."""
    state = GameState.create(generate_track(715_000), 4, seed=321)
    for player in state.players:
        player.lap = 1
    plain = StaticSearchAgent()
    traced = TracingStaticSearchAgent()
    legal = rules.legal_gear_shifts(
        state.get_player(0).gear, state.get_player(0).heat_available
    )
    candidates = traced._candidate_plans(state, 0, legal)
    gear, cards = candidates[0]
    turn_seed = traced._turn_seed(traced._turn_signature(state, 0))
    score, _rows = traced._trace_plan(state, 0, gear, cards, turn_seed, 0)
    assert score == plain._score_plan(state, 0, gear, cards, turn_seed, 0)
    assert traced.choose_gear(state, 0, legal) == plain.choose_gear(state, 0, legal)
    assert traced._plan_cards == plain._plan_cards
    assert len(traced.records[0]["chosen"]["matched_determinizations"]) == 2


def test_manual_set_keeps_six_baselines_then_highest_unread() -> None:
    """Manual review covers every stratum before anomaly-ranked additions."""
    pairs = []
    for index in range(18):
        pairs.append(
            {
                "pair_id": f"p{index}",
                "spec": {"sample_rank": index % 3},
                "manual_score": index,
                "replays": [],
            }
        )
    selected = _manual_review_ids(pairs)
    assert selected[:6] == [f"p{index}" for index in range(0, 18, 3)]
    assert len(selected) == 12
