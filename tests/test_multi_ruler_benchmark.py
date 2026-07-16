"""Focused matrix-contract tests for the B2 multi-ruler benchmark."""

from __future__ import annotations

import pytest

from experiments.eval_multi_ruler_benchmark import (
    CONTENDERS,
    RULER_LABELS,
    SEAT_COUNTS,
    _cell_key,
    _migrate_payload_identities,
    _parse_args,
    summarize_payload,
)


def test_b2_default_matrix_is_fixed_and_complete() -> None:
    """The reliable run cannot silently drop a recipe, ruler, or seat count."""
    args = _parse_args([])
    assert args.games == 200
    assert args.tracks == 40
    assert len(CONTENDERS) == 6
    assert {spec.generation_id for spec in CONTENDERS} == {"G0002", "G0003"}
    assert CONTENDERS[0].agent_id == "G0002-R00"
    assert CONTENDERS[-1].selected_checkpoint_id == "G0003-R02@final"
    assert RULER_LABELS == (
        "weak",
        "heuristic_repaired",
        "search_candidate",
        "historical_g0002_r00_1000k",
    )
    assert SEAT_COUNTS == (2, 4, 6)
    assert len(CONTENDERS) * len(RULER_LABELS) * len(SEAT_COUNTS) == 72


def test_b2_cell_keys_are_stable() -> None:
    """Partial results use an unambiguous resume key."""
    assert _cell_key("G0003-R01", "search_candidate", 6) == (
        "G0003-R01|search_candidate|6"
    )


def test_explicit_registered_contenders_are_parsed_for_completion_runs() -> None:
    """The B2 harness can evaluate a frozen baseline/candidate set explicitly."""
    args = _parse_args(["--agents", "G0002-R00", "G0004-R00"])
    assert args.agents == ["G0002-R00", "G0004-R00"]


def test_legacy_b2_result_ids_migrate_without_changing_measurements() -> None:
    """Old aliases become registry IDs while raw race evidence stays intact."""
    payload = {
        "config": {
            "contenders": [{"label": "s2_seed1", "recipe": "S2_anchor"}],
            "rulers": ["historical_s1_seed0"],
        },
        "cells": [
            {
                "key": "s2_seed1|historical_s1_seed0|6",
                "contender": "s2_seed1",
                "recipe": "S2_anchor",
                "ruler": "historical_s1_seed0",
                "seats": 6,
                "cell": {"games": 200, "wins": 33},
            }
        ],
    }
    assert _migrate_payload_identities(payload)
    assert payload["cells"][0]["key"] == (
        "G0003-R01|historical_g0002_r00_1000k|6"
    )
    assert payload["cells"][0]["selected_checkpoint_id"] == "G0003-R01@final"
    assert payload["cells"][0]["cell"] == {"games": 200, "wins": 33}


def test_completion_summary_applies_predeclared_stability_and_final_gates() -> None:
    """Lower candidate variance with equal breadth passes the frozen S3 rule."""
    cells = []
    aggregates = {
        "G0002": (-0.1, 0.1, 0.3),
        "G0004": (0.05, 0.1, 0.15),
    }
    for generation, run_values in aggregates.items():
        for seed, reward in enumerate(run_values):
            agent_id = f"{generation}-R{seed:02d}"
            for ruler in RULER_LABELS:
                for seats in SEAT_COUNTS:
                    cells.append(
                        {
                            "contender": agent_id,
                            "recipe": generation,
                            "ruler": ruler,
                            "seats": seats,
                            "cell": {"mean_placement_reward": reward},
                        }
                    )

    summary = summarize_payload({"cells": cells})

    stabilization = summary["gates"]["stabilization"]
    assert stabilization["pass"] is True
    assert stabilization["candidate_to_baseline_sd_ratio"] == pytest.approx(0.25)
    assert summary["gates"]["final_g0002"]["pass"] is False
    assert summary["gates"]["final_g0004"]["pass"] is True
