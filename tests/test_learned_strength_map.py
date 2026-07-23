"""Focused contracts for the L0 learned-policy evidence consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyze_learned_strength_map import (
    _parse_milestone_markdown,
    build_l0_report,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_milestone_parser_preserves_ruler_seat_rewards(tmp_path: Path) -> None:
    """The legacy Markdown table is parsed without treating percentages as reward."""

    table = """\
## step 300000

| opponent | split | seats | games | wins | win% | Wilson LB | Wilson UB | chance | mean reward | LB>chance |
|---|---|---|---|---|---|---|---|---|---|---|
| strong | heldout | 2 | 50 | 28 | 56.0% | 0.423 | 0.688 | 0.500 | +0.120 | no |
| strong | heldout | 4 | 50 | 16 | 32.0% | 0.208 | 0.458 | 0.250 | +0.067 | no |
| strong | heldout | 6 | 50 | 9 | 18.0% | 0.098 | 0.308 | 0.167 | -0.096 | no |
| weak | heldout | 2 | 50 | 26 | 52.0% | 0.385 | 0.652 | 0.500 | +0.080 | no |
| weak | heldout | 4 | 50 | 12 | 24.0% | 0.143 | 0.374 | 0.250 | -0.187 | no |
| weak | heldout | 6 | 50 | 7 | 14.0% | 0.070 | 0.262 | 0.167 | -0.160 | no |
"""
    path = tmp_path / "step_300000.eval.md"
    path.write_text(table, encoding="utf-8")

    rows = _parse_milestone_markdown(path)

    assert len(rows) == 6
    assert rows[0]["mean_placement_reward"] == pytest.approx(0.120)
    assert rows[-1] == {
        "opponent": "weak",
        "split": "heldout",
        "seats": 6,
        "games": 50,
        "wins": 7,
        "mean_placement_reward": -0.160,
    }


def test_l0_consolidates_frozen_artifacts_without_mixing_final_data() -> None:
    """L0 keeps evidence roles, paired seeds, and historical-final limits explicit."""

    report = build_l0_report(REPO_ROOT)

    assert report["status"] == "complete_existing_evidence_only"
    assert report["policy_registry"]["generation_allocated"] is False
    assert report["findings"]["total_existing_races"] == 28_800
    assert report["findings"]["g0003_b2_positive_matched_seeds"] == 2
    assert report["findings"]["g0004_s3_positive_matched_seeds"] == 3
    assert "track_family_outcomes" in report["coverage"][
        "not_recoverable_from_existing_artifacts"
    ]

    final = next(
        matrix
        for matrix in report["matrices"]
        if matrix["artifact_id"] == "a8_untouched_final"
    )
    assert final["evidence_role"] == "historical_untouched_final_do_not_tune"
    assert final["generations"]["G0002"]["median_aggregate_reward"] == pytest.approx(
        0.13902777777777778
    )


def test_l0_cli_writes_a_deterministic_machine_readable_receipt(
    tmp_path: Path,
) -> None:
    """The command writes one complete JSON artifact at the requested path."""

    output = tmp_path / "l0.json"
    assert main(["--root", str(REPO_ROOT), "--out", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["investigation_id"] == "L0"
    assert payload["next_decision"]["freeze_runtime_recipe"]["n_steps"] == 32_768
    assert payload["minimum_live_benchmark"]["promotion"]["games_per_cell"] == 200
