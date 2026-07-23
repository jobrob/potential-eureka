#!/usr/bin/env python
"""Bounded Direction D semantic, resume, scalar, and lane experiments."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from heat.engine.driver import Decision, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.selfplay.phase1 import A8Config, A8ResumeConfig, train_selfplay_a8
from heat.ml.selfplay.semantic_contract import (
    SEMANTIC_CONTRACT_VERSION,
    canonical_decision_row,
    canonical_event,
    canonical_game_state,
    semantic_sha256,
)
from heat.ml.selfplay.training_state import (
    TrainingStateError,
    checkpoint_receipt_path,
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    training_state_digest,
)
from heat.tracks.generator import TrackGenParams, track_sampler


_REPO_ROOT = Path(__file__).resolve().parents[1]


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS layout used for passive RSS telemetry."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current and peak working set without instrumenting allocations."""
    if sys.platform != "win32":
        return None, None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not success:
        return None, None
    return int(counters.working_set_size), int(counters.peak_working_set_size)


def _write_json(path: Path, value: object) -> None:
    """Write one human-inspectable result artifact with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_tree_identity() -> str:
    """Hash the current code inputs, including untracked source files."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        "experiments",
        "pyproject.toml",
    ]
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    digest = hashlib.sha256()
    for relative in sorted(line for line in result.stdout.splitlines() if line):
        path = _REPO_ROOT / relative
        if not path.is_file():
            continue
        encoded = relative.replace("\\", "/").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _d0_resume_config(total_iterations: int) -> A8Config:
    """Return the short deterministic recipe shared by all D0-R processes."""
    n_steps = 64
    return A8Config(
        total_timesteps=total_iterations * n_steps,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=2,
        hidden_sizes=(32,),
        seat_counts=(2, 3, 4, 5, 6),
        snapshot_every=1,
        pool_capacity=3,
        pool_prob=0.5,
        device="cpu",
        seed=19,
        collector_mode="scalar",
        stage1_enabled=False,
    )


def _resume_worker(args: argparse.Namespace) -> int:
    """Run one fresh-process D0-R segment and atomically save its boundary."""
    config = _d0_resume_config(args.total_iterations)
    resume = A8ResumeConfig(
        campaign_id=args.campaign_id,
        source_identity=args.source_identity,
        load_path=args.load,
        save_path=args.output,
        max_iterations=args.iterations,
    )
    _policy, records = train_selfplay_a8(config, resume=resume)
    print(
        json.dumps(
            {
                "completed_iteration": int(records[-1]["iteration"]),
                "steps": int(records[-1]["steps"]),
                "checkpoint": str(args.output),
            }
        ),
        flush=True,
    )
    return 0


def _run_worker(
    *,
    output: Path,
    load: Path | None,
    iterations: int,
    total_iterations: int,
    campaign_id: str,
    source_identity: str,
    timeout_seconds: float,
) -> None:
    """Launch one isolated process segment with a hard timeout."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_resume-worker",
        "--output",
        str(output),
        "--iterations",
        str(iterations),
        "--total-iterations",
        str(total_iterations),
        "--campaign-id",
        campaign_id,
        "--source-identity",
        source_identity,
    ]
    if load is not None:
        command.extend(["--load", str(load)])
    subprocess.run(
        command,
        cwd=_REPO_ROOT,
        check=True,
        timeout=timeout_seconds,
    )


def _load_d0_state(
    path: Path, *, campaign_id: str, source_identity: str, total_iterations: int
) -> dict[str, Any]:
    """Load one D0-R state through the same compatibility gates as training."""
    config = _d0_resume_config(total_iterations)
    recipe = resolved_recipe(config, TrackGenParams())
    return load_training_state(
        path,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id=campaign_id,
        source_identity=source_identity,
        anchor_identity=None,
    )


def _invalid_checkpoint_probes(
    control: Path,
    directory: Path,
    *,
    campaign_id: str,
    source_identity: str,
    total_iterations: int,
) -> dict[str, bool]:
    """Prove damaged bytes, receipts, and source identities fail closed."""
    results: dict[str, bool] = {}
    truncated = directory / "truncated.pt"
    shutil.copy2(control, truncated)
    shutil.copy2(checkpoint_receipt_path(control), checkpoint_receipt_path(truncated))
    data = truncated.read_bytes()
    truncated.write_bytes(data[: max(1, len(data) // 3)])
    try:
        _load_d0_state(
            truncated,
            campaign_id=campaign_id,
            source_identity=source_identity,
            total_iterations=total_iterations,
        )
    except TrainingStateError:
        results["truncated_rejected"] = True
    else:
        results["truncated_rejected"] = False

    bad_receipt = directory / "bad_receipt.pt"
    shutil.copy2(control, bad_receipt)
    checkpoint_receipt_path(bad_receipt).write_text(
        f"{'0' * 64}  {bad_receipt.name}\n", encoding="ascii"
    )
    try:
        _load_d0_state(
            bad_receipt,
            campaign_id=campaign_id,
            source_identity=source_identity,
            total_iterations=total_iterations,
        )
    except TrainingStateError:
        results["bad_receipt_rejected"] = True
    else:
        results["bad_receipt_rejected"] = False

    try:
        _load_d0_state(
            control,
            campaign_id=campaign_id,
            source_identity="incompatible-source",
            total_iterations=total_iterations,
        )
    except TrainingStateError:
        results["source_mismatch_rejected"] = True
    else:
        results["source_mismatch_rejected"] = False
    return results


def run_resume_experiment(args: argparse.Namespace) -> int:
    """Compare uninterrupted, one-stop, and two-stop fresh-process controls."""
    started = perf_counter()
    source_identity = _source_tree_identity()
    campaign_id = "dev-d0-resume"
    total_iterations = args.total_iterations
    per_process_timeout = min(180.0, args.timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="heat-d0-r-") as raw_directory:
        directory = Path(raw_directory)
        control = directory / "control.pt"
        one_stop = directory / "one_stop.pt"
        two_stop = directory / "two_stop.pt"

        _run_worker(
            output=control,
            load=None,
            iterations=total_iterations,
            total_iterations=total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            timeout_seconds=per_process_timeout,
        )
        first = total_iterations // 2
        _run_worker(
            output=one_stop,
            load=None,
            iterations=first,
            total_iterations=total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            timeout_seconds=per_process_timeout,
        )
        _run_worker(
            output=one_stop,
            load=one_stop,
            iterations=total_iterations - first,
            total_iterations=total_iterations,
            campaign_id=campaign_id,
            source_identity=source_identity,
            timeout_seconds=per_process_timeout,
        )
        split_a = max(1, total_iterations // 3)
        split_b = max(1, (total_iterations - split_a) // 2)
        split_c = total_iterations - split_a - split_b
        for load, count in ((None, split_a), (two_stop, split_b), (two_stop, split_c)):
            _run_worker(
                output=two_stop,
                load=load,
                iterations=count,
                total_iterations=total_iterations,
                campaign_id=campaign_id,
                source_identity=source_identity,
                timeout_seconds=per_process_timeout,
            )

        states = {
            "uninterrupted": _load_d0_state(
                control,
                campaign_id=campaign_id,
                source_identity=source_identity,
                total_iterations=total_iterations,
            ),
            "one_stop": _load_d0_state(
                one_stop,
                campaign_id=campaign_id,
                source_identity=source_identity,
                total_iterations=total_iterations,
            ),
            "two_stop": _load_d0_state(
                two_stop,
                campaign_id=campaign_id,
                source_identity=source_identity,
                total_iterations=total_iterations,
            ),
        }
        digests = {
            name: training_state_digest(state) for name, state in states.items()
        }
        component_digests = {
            component: {
                name: training_state_digest({component: state[component]})
                for name, state in states.items()
            }
            for component in (
                "policy",
                "optimizer",
                "controller",
                "snapshot_pool",
                "rng",
                "schedule",
                "progress",
                "records",
            )
        }
        invalid = _invalid_checkpoint_probes(
            control,
            directory,
            campaign_id=campaign_id,
            source_identity=source_identity,
            total_iterations=total_iterations,
        )

    equality_pass = len(set(digests.values())) == 1
    result = {
        "experiment": "D0-R",
        "hypothesis": "safe-boundary fresh-process resume is complete-state exact",
        "confidence": 0.85,
        "campaign_id": campaign_id,
        "source_identity": source_identity,
        "total_iterations": total_iterations,
        "digests": digests,
        "component_digests": component_digests,
        "invalid_checkpoint_probes": invalid,
        "equality_pass": equality_pass,
        "invalid_rejection_pass": all(invalid.values()),
        "pass": equality_pass and all(invalid.values()),
        "wall_seconds": perf_counter() - started,
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


def _deterministic_action(state: GameState, decision: Decision) -> object:
    """Choose the lowest canonical legal action without policy randomness."""
    forced = forced_action(decision, state)
    if forced is not NO_FORCED:
        return forced
    mask = legal_action_mask(decision, state)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("semantic corpus decision has no legal action")
    return decode_legal_action(decision, state, int(indices[0]))


def run_semantic_corpus(args: argparse.Namespace) -> int:
    """Generate deterministic scalar traces spanning every supported seat count."""
    sampler = track_sampler(TrackGenParams(), base_seed=80_008)
    decision_fixtures: dict[str, dict[str, Any]] = {}
    decision_digests: list[str] = []
    event_fixtures: dict[str, dict[str, object]] = {}
    event_digests: list[str] = []
    terminals: list[dict[str, object]] = []
    game_id = 0
    for seats in range(2, 7):
        for repeat in range(args.games_per_seat):
            seed = 19_000 + seats * 100 + repeat
            state = GameState.create(sampler(seed), seats, seed=seed)
            for player in state.players:
                player.lap = 1
            decision_index = 0
            while not state.is_game_over and state.round_num <= MAX_ROUNDS:
                driver = run_round_driver(state)
                send_value: object = None
                while True:
                    try:
                        decision = driver.send(send_value)
                    except StopIteration as stop:
                        for event in stop.value:
                            canonical = canonical_event(event)
                            event_digests.append(semantic_sha256(canonical))
                            event_key = f"{seats}:{canonical['phase']}:{canonical['event_type']}"
                            event_fixtures.setdefault(event_key, canonical)
                        break
                    row = canonical_decision_row(
                        state,
                        decision,
                        game_id=game_id,
                        decision_index=decision_index,
                    )
                    decision_digests.append(str(row["sha256"]))
                    fixture_key = f"{seats}:{decision.kind.value}"
                    decision_fixtures.setdefault(fixture_key, row)
                    decision_index += 1
                    send_value = _deterministic_action(state, decision)
            terminals.append(
                {
                    "game_id": game_id,
                    "seat_count": seats,
                    "seed": seed,
                    "terminated": state.is_game_over,
                    "finish_order": [
                        player.player_id for player in state.finished_players
                    ],
                    "state": canonical_game_state(state),
                }
            )
            game_id += 1
    decision_kinds = sorted(
        {str(row["decision"]["kind"]) for row in decision_fixtures.values()}
    )
    event_phases = sorted(
        {str(event["phase"]) for event in event_fixtures.values()}
    )
    decision_phases = {
        "gear": "shift_gears",
        "cards": "play_cards",
        "react": "react",
        "slipstream": "slipstream",
        "discard": "discard",
    }
    engine_phases = sorted(
        set(event_phases) | {decision_phases[kind] for kind in decision_kinds}
    )
    corpus: dict[str, object] = {
        "contract_version": SEMANTIC_CONTRACT_VERSION,
        "source_identity": _source_tree_identity(),
        "games": game_id,
        "seat_counts": list(range(2, 7)),
        "decision_kinds": decision_kinds,
        "event_phases": event_phases,
        "engine_phases": engine_phases,
        "decision_trace_count": len(decision_digests),
        "decision_trace_sha256": semantic_sha256(decision_digests),
        "decision_trace_digests": decision_digests,
        "decision_fixtures": list(decision_fixtures.values()),
        "event_trace_count": len(event_digests),
        "event_trace_sha256": semantic_sha256(event_digests),
        "event_trace_digests": event_digests,
        "event_fixtures": list(event_fixtures.values()),
        "terminals": terminals,
    }
    corpus["sha256"] = semantic_sha256(corpus)
    _write_json(Path(args.output), corpus)
    summary = {
        "output": str(args.output),
        "sha256": corpus["sha256"],
        "games": game_id,
        "decision_trace_count": len(decision_digests),
        "decision_fixtures": len(decision_fixtures),
        "event_trace_count": len(event_digests),
        "event_fixtures": len(event_fixtures),
        "decision_kinds": decision_kinds,
        "event_phases": event_phases,
        "engine_phases": engine_phases,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _runtime_receipt() -> dict[str, object]:
    """Return stable software/hardware facts needed to interpret D0-B."""
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "source_identity": _source_tree_identity(),
    }


def run_baseline(args: argparse.Namespace) -> int:
    """Run three fixed scalar A8 workloads and report repeatability/projection."""
    experiment_started = perf_counter()
    repeats: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        if perf_counter() - experiment_started >= args.timeout_seconds:
            raise TimeoutError("D0-B exceeded its predeclared time limit")
        config = A8Config(
            total_timesteps=args.transitions,
            n_steps=1024,
            batch_size=256,
            n_epochs=5,
            hidden_sizes=(256, 256),
            seat_counts=(2, 3, 4, 5, 6),
            seed=0,
            device="cpu",
            collector_mode="scalar",
            head="masked",
            encoder="flat",
            margin_coef=0.0,
            anchor_share=0.0,
            stage1_enabled=False,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        _policy, records = train_selfplay_a8(config, profile=True)
        wall_seconds = perf_counter() - started
        working_set, peak_working_set = _process_memory_bytes()
        rollout_seconds = sum(record["rollout_seconds"] for record in records)
        iteration_seconds = sum(record["iteration_seconds"] for record in records)
        recorded = int(records[-1]["steps"])
        repeats.append(
            {
                "repeat": repeat,
                "seed": 0,
                "requested_transitions": args.transitions,
                "recorded_transitions": recorded,
                "iterations": len(records),
                "games": int(records[-1]["total_games"]),
                "wall_seconds": wall_seconds,
                "iteration_seconds": iteration_seconds,
                "rollout_seconds": rollout_seconds,
                "encoding_seconds": sum(
                    record["encoding_seconds"] for record in records
                ),
                "inference_seconds": sum(
                    record["inference_seconds"] for record in records
                ),
                "prepare_seconds": sum(
                    record["prepare_seconds"] for record in records
                ),
                "update_seconds": sum(
                    record["update_seconds"] for record in records
                ),
                "rollout_transitions_per_second": recorded / rollout_seconds,
                "trained_transitions_per_second": recorded / wall_seconds,
                "process_working_set_bytes": working_set,
                "process_peak_working_set_bytes": peak_working_set,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
            }
        )
        print(
            f"D0-B repeat {repeat + 1}/{args.repeats}: "
            f"rollout={recorded / rollout_seconds:.1f} transitions/s "
            f"end_to_end={recorded / wall_seconds:.1f} transitions/s",
            flush=True,
        )

    rollout_rates = [
        float(repeat["rollout_transitions_per_second"]) for repeat in repeats
    ]
    trained_rates = [
        float(repeat["trained_transitions_per_second"]) for repeat in repeats
    ]
    median_rollout = statistics.median(rollout_rates)
    median_trained = statistics.median(trained_rates)
    relative_spread = (
        (max(rollout_rates) - min(rollout_rates)) / median_rollout
        if median_rollout
        else float("inf")
    )
    prior_low = 1750.0
    prior_high = 1850.0
    drift = (
        (prior_low - median_rollout) / prior_low
        if median_rollout < prior_low
        else (median_rollout - prior_high) / prior_high
        if median_rollout > prior_high
        else 0.0
    )
    result = {
        "experiment": "D0-B",
        "hypothesis": "scalar A8 reproduces the prior rollout band with <=5% spread",
        "confidence": 0.90,
        "runtime": _runtime_receipt(),
        "repeats": repeats,
        "median_rollout_transitions_per_second": median_rollout,
        "median_trained_transitions_per_second": median_trained,
        "relative_rollout_spread": relative_spread,
        "repeatability_pass": relative_spread <= 0.05,
        "prior_rollout_band": [prior_low, prior_high],
        "relative_drift_outside_prior_band": drift,
        "drift_investigation_required": drift > 0.10,
        "practical_48h_transitions_at_90pct_uptime": int(
            median_trained * 48.0 * 3600.0 * 0.90
        ),
        "wall_seconds": perf_counter() - experiment_started,
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["repeatability_pass"] else 1


def _action_histogram(records: list[dict[str, float]]) -> dict[int, int]:
    """Aggregate dynamic action-batch histogram fields from training records."""
    histogram: dict[int, int] = {}
    prefix = "action_batch_"
    suffix = "_calls"
    for record in records:
        for key, value in record.items():
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            size = int(key[len(prefix) : -len(suffix)])
            histogram[size] = histogram.get(size, 0) + int(value)
    return {size: count for size, count in sorted(histogram.items()) if count}


def _histogram_percentile(histogram: dict[int, int], quantile: float) -> float:
    """Return the nearest-rank batch size for one weighted histogram."""
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    target = max(1, int(np.ceil(quantile * total)))
    cumulative = 0
    for size, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return float(size)
    return float(max(histogram))


def _d1_training_run(
    *,
    transitions: int,
    n_steps: int,
    collector_mode: Literal["scalar", "lanes"],
    lane_count: int,
    repeat: int,
) -> dict[str, Any]:
    """Run one matched D1-L arm and return end-to-end telemetry."""
    config = A8Config(
        total_timesteps=transitions,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=5,
        hidden_sizes=(256, 256),
        seat_counts=(2, 3, 4, 5, 6),
        seed=0,
        device="cpu",
        collector_mode=collector_mode,
        lane_count=lane_count,
        head="masked",
        encoder="flat",
        margin_coef=0.0,
        anchor_share=0.0,
        stage1_enabled=False,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    _policy, records = train_selfplay_a8(config, profile=True)
    wall_seconds = perf_counter() - started
    working_set, peak_working_set = _process_memory_bytes()
    recorded = int(records[-1]["steps"])
    rollout_seconds = sum(record["rollout_seconds"] for record in records)
    action_rows = sum(record["action_inference_rows"] for record in records)
    action_calls = sum(record["action_inference_calls"] for record in records)
    histogram = _action_histogram(records)
    return {
        "repeat": repeat,
        "seed": 0,
        "collector_mode": collector_mode,
        "lane_count": lane_count if collector_mode == "lanes" else 1,
        "requested_transitions": transitions,
        "recorded_transitions": recorded,
        "iterations": len(records),
        "games": int(records[-1]["total_games"]),
        "wall_seconds": wall_seconds,
        "iteration_seconds": sum(record["iteration_seconds"] for record in records),
        "rollout_seconds": rollout_seconds,
        "encoding_seconds": sum(record["encoding_seconds"] for record in records),
        "inference_seconds": sum(record["inference_seconds"] for record in records),
        "prepare_seconds": sum(record["prepare_seconds"] for record in records),
        "update_seconds": sum(record["update_seconds"] for record in records),
        "rollout_transitions_per_second": recorded / rollout_seconds,
        "trained_transitions_per_second": recorded / wall_seconds,
        "action_inference_calls": int(action_calls),
        "action_inference_rows": int(action_rows),
        "mean_action_batch": action_rows / action_calls if action_calls else 0.0,
        "p50_action_batch": _histogram_percentile(histogram, 0.50),
        "p95_action_batch": _histogram_percentile(histogram, 0.95),
        "max_action_batch": max(histogram, default=0),
        "action_batch_histogram": histogram,
        "recorded_action_rows": int(
            sum(record["recorded_action_rows"] for record in records)
        ),
        "drain_action_rows": int(
            sum(record["drain_action_rows"] for record in records)
        ),
        "rollout_overshoot": int(
            sum(record["rollout_overshoot"] for record in records)
        ),
        "process_working_set_bytes": working_set,
        "process_peak_working_set_bytes": peak_working_set,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }


def run_lane_benchmark(args: argparse.Namespace) -> int:
    """Run matched scalar and 8/16/32-lane D1-L end-to-end repeats."""
    experiment_started = perf_counter()
    runs: list[dict[str, Any]] = []
    arms: list[tuple[Literal["scalar", "lanes"], int]] = [
        ("scalar", 1),
        *(("lanes", count) for count in args.lane_counts),
    ]
    total_runs = args.repeats * len(arms)
    for repeat in range(args.repeats):
        for collector_mode, lane_count in arms:
            if perf_counter() - experiment_started >= args.timeout_seconds:
                raise TimeoutError("D1-L exceeded its predeclared time limit")
            run = _d1_training_run(
                transitions=args.transitions,
                n_steps=args.n_steps,
                collector_mode=collector_mode,
                lane_count=lane_count,
                repeat=repeat,
            )
            runs.append(run)
            label = "scalar" if collector_mode == "scalar" else f"lanes-{lane_count}"
            print(
                f"D1-L {len(runs)}/{total_runs} {label} repeat={repeat + 1}: "
                f"rollout={float(run['rollout_transitions_per_second']):.1f}/s "
                f"end_to_end={float(run['trained_transitions_per_second']):.1f}/s "
                f"mean_batch={float(run['mean_action_batch']):.2f}",
                flush=True,
            )

    summaries: dict[str, dict[str, Any]] = {}
    for collector_mode, lane_count in arms:
        label = "scalar" if collector_mode == "scalar" else f"lanes-{lane_count}"
        selected = [
            run
            for run in runs
            if run["collector_mode"] == collector_mode
            and int(run["lane_count"]) == lane_count
        ]
        trained_rates = [
            float(run["trained_transitions_per_second"]) for run in selected
        ]
        rollout_rates = [
            float(run["rollout_transitions_per_second"]) for run in selected
        ]
        median_trained = statistics.median(trained_rates)
        relative_spread = (
            (max(trained_rates) - min(trained_rates)) / median_trained
            if median_trained
            else float("inf")
        )
        total_calls = sum(int(run["action_inference_calls"]) for run in selected)
        total_rows = sum(int(run["action_inference_rows"]) for run in selected)
        summaries[label] = {
            "collector_mode": collector_mode,
            "lane_count": lane_count,
            "median_rollout_transitions_per_second": statistics.median(rollout_rates),
            "median_trained_transitions_per_second": median_trained,
            "relative_trained_rate_spread": relative_spread,
            "stable_repeatability": relative_spread <= 0.05,
            "mean_action_batch": total_rows / total_calls if total_calls else 0.0,
            "median_p50_action_batch": statistics.median(
                float(run["p50_action_batch"]) for run in selected
            ),
            "median_p95_action_batch": statistics.median(
                float(run["p95_action_batch"]) for run in selected
            ),
            "total_drain_action_rows": sum(
                int(run["drain_action_rows"]) for run in selected
            ),
            "total_rollout_overshoot": sum(
                int(run["rollout_overshoot"]) for run in selected
            ),
        }

    scalar_rate = float(summaries["scalar"]["median_trained_transitions_per_second"])
    lane_summaries: list[tuple[str, dict[str, Any]]] = []
    for lane_count in args.lane_counts:
        label = f"lanes-{lane_count}"
        summary = summaries[label]
        summary["speedup_vs_scalar"] = (
            float(summary["median_trained_transitions_per_second"]) / scalar_rate
        )
        lane_summaries.append((label, summary))
    stable_lanes = [
        item for item in lane_summaries if bool(item[1]["stable_repeatability"])
    ]
    candidates = stable_lanes or lane_summaries
    best_label, best = max(
        candidates, key=lambda item: float(item[1]["speedup_vs_scalar"])
    )
    best_speedup = float(best["speedup_vs_scalar"])
    batch_gate = float(best["mean_action_batch"]) >= 8.0
    stable_gate = bool(best["stable_repeatability"])
    if best_speedup >= 1.5 and batch_gate and stable_gate:
        decision = "go"
    elif best_speedup < 1.3:
        decision = "stop"
    else:
        decision = "redesign"
    result = {
        "experiment": "D1-L",
        "hypothesis": (
            "independent lanes average at least eight live rows and deliver "
            ">=1.5x end-to-end trained transitions per second"
        ),
        "confidence": 0.70,
        "runtime": _runtime_receipt(),
        "workload": {
            "transitions": args.transitions,
            "n_steps": args.n_steps,
            "repeats": args.repeats,
            "lane_counts": args.lane_counts,
            "timeout_seconds": args.timeout_seconds,
        },
        "runs": runs,
        "summaries": summaries,
        "best_lane_arm": best_label,
        "best_stable_speedup_vs_scalar": best_speedup,
        "batch_gate_pass": batch_gate,
        "repeatability_gate_pass": stable_gate,
        "adoption_gate_pass": decision == "go",
        "decision": decision,
        "practical_48h_transitions_at_90pct_uptime": int(
            float(best["median_trained_transitions_per_second"])
            * 48.0
            * 3600.0
            * 0.90
        ),
        "wall_seconds": perf_counter() - experiment_started,
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bounded Direction D commands and explicit output/time limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resume = subparsers.add_parser("resume", help="run D0-R fresh-process controls")
    resume.add_argument(
        "--output", default="runs/direction_d/d0_resume.json", type=Path
    )
    resume.add_argument("--total-iterations", type=int, default=6)
    resume.add_argument("--timeout-seconds", type=float, default=600.0)
    resume.set_defaults(handler=run_resume_experiment)

    worker = subparsers.add_parser("_resume-worker")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--load", type=Path)
    worker.add_argument("--iterations", type=int, required=True)
    worker.add_argument("--total-iterations", type=int, required=True)
    worker.add_argument("--campaign-id", required=True)
    worker.add_argument("--source-identity", required=True)
    worker.set_defaults(handler=_resume_worker)

    corpus = subparsers.add_parser("corpus", help="write the scalar semantic corpus")
    corpus.add_argument(
        "--output", default="runs/direction_d/d0_semantic_corpus.json", type=Path
    )
    corpus.add_argument("--games-per-seat", type=int, default=1)
    corpus.set_defaults(handler=run_semantic_corpus)

    baseline = subparsers.add_parser("baseline", help="run D0-B scalar repeats")
    baseline.add_argument(
        "--output", default="runs/direction_d/d0_baseline.json", type=Path
    )
    baseline.add_argument("--transitions", type=int, default=100_000)
    baseline.add_argument("--repeats", type=int, default=3)
    baseline.add_argument("--timeout-seconds", type=float, default=900.0)
    baseline.set_defaults(handler=run_baseline)

    lanes = subparsers.add_parser("lanes", help="run D1-L scalar/lane repeats")
    lanes.add_argument(
        "--output", default="runs/direction_d/d1_lane_benchmark.json", type=Path
    )
    lanes.add_argument("--transitions", type=int, default=100_000)
    lanes.add_argument("--n-steps", type=int, default=1024)
    lanes.add_argument("--repeats", type=int, default=3)
    lanes.add_argument("--lane-counts", type=int, nargs="+", default=[8, 16, 32])
    lanes.add_argument("--timeout-seconds", type=float, default=2700.0)
    lanes.set_defaults(handler=run_lane_benchmark)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one explicitly selected D0 experiment."""
    args = _parse_args(argv)
    if args.command == "resume" and args.total_iterations < 3:
        raise ValueError("D0-R needs at least three iterations for two stops")
    if getattr(args, "timeout_seconds", 1.0) <= 0.0:
        raise ValueError("timeout must be positive")
    if getattr(args, "games_per_seat", 1) < 1:
        raise ValueError("games_per_seat must be positive")
    if getattr(args, "repeats", 1) < 1:
        raise ValueError("repeats must be positive")
    if getattr(args, "transitions", 1) < 1:
        raise ValueError("transitions must be positive")
    if getattr(args, "n_steps", 1) < 1:
        raise ValueError("n_steps must be positive")
    if any(count < 1 for count in getattr(args, "lane_counts", [1])):
        raise ValueError("lane counts must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
