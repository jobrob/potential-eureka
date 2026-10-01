"""Focused CPU tests for Question B 32K learning-quality scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyze_32k_learning_quality import (
    analyze_promotion,
    collapse_guard_from_records,
    evaluate_gates,
    paired_bootstrap_interval,
    summarize_runs,
    validate_completeness,
)
from experiments.eval_32k_learning_quality import (
    MILESTONE_GAMES,
    PROMOTION_GAMES,
    PROMOTION_RULERS,
    SEAT_COUNTS,
    TRACK_BASE,
    TRACK_COUNT,
    build_race_row,
    candidate_contender,
    cell_key,
    config_hash,
    load_resumable_payload,
    matrix_manifest,
    plan_payload,
    track_sha256,
)
from experiments.run_32k_learning_quality import (
    CODEC_PIN,
    MILESTONE_THRESHOLDS,
    N_STEPS,
    PROVISIONAL_GENERATION_ID,
    QUESTION_ID,
    RUN_SEEDS,
    build_question_b_config,
    disposable_campaign_id,
    question_b_recipe_fields,
    resolve_plan,
)
from heat.ml.selfplay.eval_harness import race_coordinates
from heat.ml.spaces import CODEC_VERSION
from heat.tracks.generator import generate_track


def test_question_b_recipe_pins_codec_v4_and_native_32k() -> None:
    """Question B is native 32K + codec v4; G0005 stays unallocated."""
    recipe = question_b_recipe_fields()
    assert recipe["question_id"] == "B"
    assert recipe["g0005_allocated"] is False
    assert recipe["provisional_generation_id"] == "G0005"
    assert recipe["policy_contract"]["codec_version"] == 4
    assert CODEC_VERSION == CODEC_PIN == 4
    assert recipe["n_steps"] == 32_768 == N_STEPS
    assert recipe["self_play"]["snapshot_every"] == 1
    assert list(RUN_SEEDS) == [0, 1, 2]
    assert list(MILESTONE_THRESHOLDS) == [300_000, 500_000, 750_000, 1_000_000]

    config = build_question_b_config(seed=0, device="cpu")
    assert config.collector_mode == "native"
    assert config.n_steps == 32_768
    assert config.native_workers == 48
    assert config.native_ready_capacity == 288
    assert config.snapshot_every == 1
    assert disposable_campaign_id(1).startswith("dev-32k-qb-")


def test_dry_run_plan_does_not_claim_registry_write(tmp_path: Path) -> None:
    """resolve_plan is scaffolding only: no G0005 allocation flag."""
    plan = resolve_plan(out_dir=tmp_path)
    assert plan["g0005_allocated"] is False
    assert plan["registry_write_allowed"] is False
    assert plan["question_id"] == QUESTION_ID
    assert len(plan["runs"]) == 3
    assert all(run["campaign_id"].startswith("dev-") for run in plan["runs"])
    assert PROVISIONAL_GENERATION_ID not in {
        run["campaign_id"] for run in plan["runs"]
    }


def test_run_cli_dry_run_writes_plan_without_training(tmp_path: Path) -> None:
    """``--dry-run`` exits 0 and never creates training_state checkpoints."""
    from experiments.run_32k_learning_quality import main

    plan_out = tmp_path / "plan.json"
    code = main(["--dry-run", "--out-dir", str(tmp_path / "out"), "--plan-out", str(plan_out)])
    assert code == 0
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    assert plan["g0005_allocated"] is False
    assert not list((tmp_path / "out").glob("**/training_state.pt"))


def test_coordinate_manifest_balances_and_records_six_seat_omission() -> None:
    """Promotion coordinates are identical across contenders; 6-seat omission noted."""
    manifest = matrix_manifest(matrix_id="promotion", games=PROMOTION_GAMES)
    coords = race_coordinates(PROMOTION_GAMES, TRACK_COUNT, 6)
    assert len(coords) == PROMOTION_GAMES
    assert len(manifest["cells"]["6"]["coordinates"]) == PROMOTION_GAMES
    assert manifest["cells"]["6"]["omitted_note"] is not None
    # Tracks cycle evenly.
    track_hits = [0] * TRACK_COUNT
    for track_index, _focal, _repeat in coords:
        track_hits[track_index] += 1
    assert set(track_hits) == {PROMOTION_GAMES // TRACK_COUNT}


def test_raw_row_schema_and_resume_rejection(tmp_path: Path) -> None:
    """Raw rows carry Q05 identities; resume rejects config-hash drift."""
    track = generate_track(TRACK_BASE)
    contender = candidate_contender(
        agent_id="dev-32k-qb-r00",
        training_seed=0,
        checkpoint="runs/fake.pt",
    )
    row = build_race_row(
        matrix_id="promotion",
        contender=contender,
        ruler_id="heuristic_weak_v2",
        seat_count=4,
        race_index=0,
        track_index=0,
        focal_seat=0,
        repeat_index=0,
        track=track,
        game_seed=3_200_000,
        outcome_win=True,
        placement_reward=0.5,
        terminal_round=12,
        completion_state="scored",
        source_identity="test",
        config_hash="abc",
        elapsed_seconds=0.1,
    )
    assert row["track_sha256"] == track_sha256(track)
    assert row["contender_agent_id"] == "dev-32k-qb-r00"
    assert "placement_reward" in row
    assert cell_key("promotion", "G0002-R00", "heuristic_weak_v2", 2) == (
        "promotion|G0002-R00|heuristic_weak_v2|2"
    )

    path = tmp_path / "promotion.json"
    payload = {
        "question_id": QUESTION_ID,
        "codec_pin": CODEC_PIN,
        "config": {"config_hash": "deadbeef"},
        "cells": [{"key": "a", "mean_placement_reward": 0.1}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_resumable_payload(path, expected_config_hash="deadbeef", resume=True)
    assert "a" in loaded
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_resumable_payload(path, expected_config_hash="other", resume=True)


def test_eval_plan_names_v2_rulers_and_fresh_track_band() -> None:
    """Dry-run eval plan freezes the Q04 band and V2 promotion rulers."""
    plan = plan_payload(candidate_specs=(), include_baselines=False)
    assert plan["track_band"]["base"] == 1_415_000
    assert plan["track_band"]["count"] == 40
    assert plan["matrices"]["promotion"]["games_per_cell"] == 200
    assert plan["matrices"]["milestone"]["games_per_cell"] == MILESTONE_GAMES
    assert plan["matrices"]["promotion"]["rulers"] == list(PROMOTION_RULERS)
    assert plan["g0005_allocated"] is False
    assert SEAT_COUNTS == (2, 4, 6)


def _toy_matrix(candidate_reward: float, baseline_reward: float) -> dict:
    """Build a minimal complete promotion-like payload for gate tests."""
    rulers = list(PROMOTION_RULERS)
    seats = list(SEAT_COUNTS)
    contenders = []
    cells = []
    for generation, reward in (("G0005", candidate_reward), ("G0002", baseline_reward)):
        for seed in (0, 1, 2):
            agent_id = f"{generation}-R{seed:02d}"
            contenders.append(
                {
                    "agent_id": agent_id,
                    "generation_id": generation,
                    "training_seed": seed,
                    "checkpoint_id": f"{agent_id}@1000K",
                    "checkpoint": f"runs/{agent_id}.pt",
                    "role": "candidate" if generation == "G0005" else "baseline",
                }
            )
            for ruler in rulers:
                for seat in seats:
                    rows = [
                        {
                            "race_coordinate_id": f"{agent_id}|{ruler}|{seat}|{g}",
                            "contender_agent_id": agent_id,
                            "contender_training_seed": seed,
                            "ruler_id": ruler,
                            "seat_count": seat,
                            "track_seed": TRACK_BASE + (g % 40),
                            "game_seed": 3_200_000 + g,
                            "focal_starting_seat": g % seat,
                            "placement_reward": reward,
                            "track_sha256": "x" * 64,
                        }
                        for g in range(2)  # tiny raw rows for schema; games_per_cell below
                    ]
                    # Use games_per_cell matching raw_rows for completeness.
                    cells.append(
                        {
                            "key": f"promotion|{agent_id}|{ruler}|{seat}",
                            "matrix_id": "promotion",
                            "contender": agent_id,
                            "generation_id": generation,
                            "ruler": ruler,
                            "seats": seat,
                            "games": 2,
                            "mean_placement_reward": reward,
                            "raw_rows": rows,
                        }
                    )
    return {
        "complete": True,
        "question_id": QUESTION_ID,
        "config": {
            "contenders": contenders,
            "rulers": rulers,
            "seat_counts": seats,
            "games_per_cell": 2,
        },
        "cells": cells,
    }


def test_gate_boundaries_pass_and_fail() -> None:
    """Aggregate guard flips exactly at the predeclared -0.030 tolerance."""
    summary_pass = summarize_runs(_toy_matrix(0.10, 0.12))
    gates_pass = evaluate_gates(
        candidate=summary_pass["generations"]["G0005"],
        baseline=summary_pass["generations"]["G0002"],
        collapse={"pass": True, "applied": True},
    )
    assert gates_pass["aggregate"]["pass"] is True
    assert gates_pass["all_pass"] is True

    summary_fail = summarize_runs(_toy_matrix(0.00, 0.12))
    gates_fail = evaluate_gates(
        candidate=summary_fail["generations"]["G0005"],
        baseline=summary_fail["generations"]["G0002"],
        collapse={"pass": True, "applied": True},
    )
    assert gates_fail["aggregate"]["pass"] is False
    assert gates_fail["all_pass"] is False


def test_collapse_guard_uses_rollout_entropy_not_update_entropy() -> None:
    """Three saturated low-rollout-entropy updates fail; update_entropy ignored."""
    ok = collapse_guard_from_records(
        {
            "r0": [
                {"iteration": 1, "rollout_entropy": 0.9, "ent_coef": 0.10},
                {"iteration": 2, "rollout_entropy": 0.9, "ent_coef": 0.10},
                {"iteration": 3, "rollout_entropy": 0.9, "ent_coef": 0.10},
            ]
        }
    )
    assert ok["pass"] is True

    bad = collapse_guard_from_records(
        {
            "r0": [
                {
                    "iteration": 1,
                    "rollout_entropy": 0.2,
                    "update_entropy": 0.9,
                    "ent_coef": 0.10,
                },
                {
                    "iteration": 2,
                    "rollout_entropy": 0.2,
                    "update_entropy": 0.9,
                    "ent_coef": 0.10,
                },
                {
                    "iteration": 3,
                    "rollout_entropy": 0.2,
                    "update_entropy": 0.9,
                    "ent_coef": 0.10,
                },
            ]
        }
    )
    assert bad["pass"] is False


def test_deterministic_summary_and_bootstrap() -> None:
    """Analysis is deterministic for the same toy matrix."""
    payload = _toy_matrix(0.10, 0.05)
    assert validate_completeness(payload)["ok"] is True
    first = analyze_promotion(payload, candidate_generation="G0005")
    second = analyze_promotion(payload, candidate_generation="G0005")
    assert first["disposition"] == "PASS"
    assert first == second
    interval = paired_bootstrap_interval([0.1, 0.05, 0.0, -0.02])
    assert interval["low"] <= interval["mean"] <= interval["high"]


def test_analyze_cli_dry_run() -> None:
    """Analyze dry-run prints gate constants without requiring artifacts."""
    from experiments.analyze_32k_learning_quality import main

    assert main(["--dry-run"]) == 0


def test_config_hash_is_stable() -> None:
    """Evaluation config hashing is order-insensitive via sort_keys."""
    left = config_hash({"b": 1, "a": [3, 2]})
    right = config_hash({"a": [3, 2], "b": 1})
    assert left == right
