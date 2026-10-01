#!/usr/bin/env python
"""Run the selected A10 native soak candidate with bounded telemetry.

Hypothesis (0.75 confidence before measurement): the selected native CPU arm
sustains at least 5,000 trained transitions/s for one hour without a progress
stall, material post-warmup memory growth, CPU clock collapse, or an invalid
safe-boundary checkpoint. This is a disposable ``dev-*`` run; its learned
weights are not a registered policy campaign or learning-quality result.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
from time import monotonic
from typing import Any

import numpy as np
import torch

from heat.ml.selfplay.phase1 import A8Config, A8ResumeConfig, train_selfplay_a8
from heat.ml.selfplay.training_state import (
    checkpoint_receipt_path,
    published_checkpoint_path,
    load_training_state,
    recipe_sha256,
    resolved_recipe,
    training_state_digest,
)
from heat.tracks.generator import TrackGenParams

if __package__:
    from experiments.verify_direction_d3_native import native_source_tree_identity
else:
    from verify_direction_d3_native import native_source_tree_identity


_MIB = 1024 * 1024


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS layout for passive RSS telemetry."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


class _ProcessPowerThrottlingState(ctypes.Structure):
    """Windows PROCESS_POWER_THROTTLING_STATE layout."""

    _fields_ = [
        ("version", wintypes.DWORD),
        ("control_mask", wintypes.DWORD),
        ("state_mask", wintypes.DWORD),
    ]


def _parse_args() -> argparse.Namespace:
    """Parse the frozen A10 arm and bounded operational controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=3_600.0)
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--system-sample-seconds", type=float, default=120.0)
    parser.add_argument("--stall-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-every-iterations", type=int, default=25)
    parser.add_argument("--n-steps", type=int, default=32_768)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--ready-capacity", type=int, default=288)
    parser.add_argument("--refill-reserve-factor", type=float, default=1.30)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--torch-interop-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-power-throttling", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/direction_d/d3_a10_native_soak.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/direction_d/d3_a10_native_soak_state.pt"),
    )
    args = parser.parse_args()
    if (
        args.duration_seconds <= 0.0
        or args.sample_seconds <= 0.0
        or args.system_sample_seconds <= 0.0
    ):
        parser.error("duration and sample interval must be positive")
    if args.stall_seconds <= args.sample_seconds:
        parser.error("stall threshold must exceed the sample interval")
    if args.checkpoint_every_iterations < 1:
        parser.error("checkpoint interval must be positive")
    if args.workers < 1 or args.ready_capacity < args.workers * 6:
        parser.error("ready capacity must hold a six-seat group per worker")
    if not 1.0 <= args.refill_reserve_factor <= 2.0:
        parser.error("refill reserve factor must be in [1, 2]")
    return args


def _process_power_throttling(*, disable: bool = False) -> dict[str, Any]:
    """Read or disable this process's Windows execution-speed throttling."""
    if os.name != "nt":
        return {"available": False, "reason": "not Windows"}
    execution_speed = 0x1
    process_power_throttling = 4
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_information = kernel32.GetProcessInformation
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    get_information.restype = wintypes.BOOL
    set_information = kernel32.SetProcessInformation
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if disable:
        requested = _ProcessPowerThrottlingState(
            version=1,
            control_mask=execution_speed,
            state_mask=0,
        )
        if not set_information(
            process,
            process_power_throttling,
            ctypes.byref(requested),
            ctypes.sizeof(requested),
        ):
            return {
                "available": True,
                "disable_requested": True,
                "disable_succeeded": False,
                "windows_error": ctypes.get_last_error(),
            }
    state = _ProcessPowerThrottlingState(version=1)
    if not get_information(
        process,
        process_power_throttling,
        ctypes.byref(state),
        ctypes.sizeof(state),
    ):
        return {
            "available": False,
            "windows_error": ctypes.get_last_error(),
        }
    return {
        "available": True,
        "disable_requested": disable,
        "disable_succeeded": (not disable) or not bool(state.state_mask & execution_speed),
        "control_mask": int(state.control_mask),
        "state_mask": int(state.state_mask),
        "execution_speed_controlled": bool(state.control_mask & execution_speed),
        "execution_speed_throttled": bool(state.state_mask & execution_speed),
    }


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current and peak process working set on Windows."""
    if os.name != "nt":
        return None, None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_info.restype = ctypes.c_int
    success = get_info(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        return None, None
    return int(counters.working_set_size), int(counters.peak_working_set_size)


def _powershell_system_counters() -> dict[str, Any]:
    """Read CPU frequency/load and the available ACPI thermal-zone counters."""
    script = r"""
$result = [ordered]@{
  cpu_frequency_mhz = $null
  cpu_utilization_percent = $null
  cpu_performance_percent = $null
  cpu_maximum_frequency_percent = $null
  acpi_thermal_zones_celsius = @()
  busiest_logical_processors = @()
  error = $null
}
try {
  $corePerformance = @{}
  $coreUtility = @{}
  $samples = (Get-Counter -Counter @(
    '\Processor Information(_Total)\Processor Frequency',
    '\Processor Information(*)\% Processor Performance',
    '\Processor Information(*)\% Processor Utility',
    '\Processor Information(_Total)\% of Maximum Frequency',
    '\Processor(_Total)\% Processor Time',
    '\Thermal Zone Information(*)\Temperature'
  ) -MaxSamples 1 -ErrorAction Stop).CounterSamples
  foreach ($sample in $samples) {
    if ($sample.Path -like '*Processor Frequency' -and $sample.InstanceName -eq '_total') {
      $result.cpu_frequency_mhz = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*% Processor Performance' -and $sample.InstanceName -eq '_total') {
      $result.cpu_performance_percent = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*% Processor Performance') {
      $corePerformance[$sample.InstanceName] = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*% Processor Utility' -and $sample.InstanceName -ne '_total') {
      $coreUtility[$sample.InstanceName] = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*% of Maximum Frequency') {
      $result.cpu_maximum_frequency_percent = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*% Processor Time') {
      $result.cpu_utilization_percent = [math]::Round($sample.CookedValue, 3)
    } elseif ($sample.Path -like '*Thermal Zone Information*Temperature') {
      $result.acpi_thermal_zones_celsius += [math]::Round($sample.CookedValue - 273.15, 3)
    }
  }
  $result.busiest_logical_processors = @(
    $coreUtility.GetEnumerator() |
      Sort-Object Value -Descending |
      Select-Object -First 6 |
      ForEach-Object {
        [ordered]@{
          instance = $_.Key
          utility_percent = $_.Value
          performance_percent = $corePerformance[$_.Key]
        }
      }
  )
} catch {
  $result.error = 'performance counters: ' + $_.Exception.Message
}
$result | ConvertTo-Json -Compress -Depth 3
"""
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        value = json.loads(completed.stdout)
        return value if isinstance(value, dict) else {"error": "invalid JSON root"}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"error": f"system counters unavailable: {exc}"}


def _gpu_telemetry() -> dict[str, Any]:
    """Read passive NVIDIA telemetry; the selected training path remains CPU-only."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,clocks.current.sm,clocks.current.memory,"
        "power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        row = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        if len(row) != 7:
            raise ValueError("unexpected nvidia-smi field count")
        return {
            "name": row[0],
            "temperature_celsius": float(row[1]),
            "sm_clock_mhz": float(row[2]),
            "memory_clock_mhz": float(row[3]),
            "power_watts": float(row[4]),
            "utilization_percent": float(row[5]),
            "memory_used_mib": float(row[6]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"error": f"nvidia-smi unavailable: {exc}"}


def _json_default(value: object) -> object:
    """Convert NumPy scalars while rejecting unknown receipt value types."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write a recoverable progress receipt without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _Monitor:
    """Collect passive samples and expose one durable heartbeat stream."""

    def __init__(
        self,
        *,
        started: float,
        duration_seconds: float,
        sample_seconds: float,
        system_sample_seconds: float,
        stall_seconds: float,
        progress_path: Path,
    ) -> None:
        self.started = started
        self.duration_seconds = duration_seconds
        self.sample_seconds = sample_seconds
        self.system_sample_seconds = system_sample_seconds
        self.stall_seconds = stall_seconds
        self.progress_path = progress_path
        self.iterations: list[dict[str, Any]] = []
        self.system_samples: list[dict[str, Any]] = []
        self.stall_samples: list[dict[str, Any]] = []
        self.last_progress_at = started
        self.last_steps = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        """Start passive sampling alongside the training loop."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait briefly for any in-flight counter query."""
        self._stop.set()
        self._thread.join(timeout=20)

    def note_iteration(self, iteration: int, record: dict[str, float]) -> None:
        """Record one completed PPO boundary for stability and stall analysis."""
        now = monotonic()
        current_memory, peak_memory = _process_memory_bytes()
        sample = {
            "iteration": iteration,
            "elapsed_seconds": now - self.started,
            "steps": int(record["steps"]),
            "recorded_rows": int(record["n_recorded"]),
            "iteration_seconds": float(record["iteration_seconds"]),
            "rollout_seconds": float(record["rollout_seconds"]),
            "inference_seconds": float(record["inference_seconds"]),
            "update_seconds": float(record["update_seconds"]),
            "process_working_set_bytes": current_memory,
            "process_peak_working_set_bytes": peak_memory,
        }
        with self._lock:
            self.iterations.append(sample)
            self.last_progress_at = now
            self.last_steps = int(record["steps"])

    def should_stop(self, _iteration: int, _record: dict[str, float]) -> bool:
        """Stop only after the requested wall time and a complete PPO boundary."""
        return monotonic() - self.started >= self.duration_seconds

    def _sample_loop(self) -> None:
        """Sample system counters and write a progress receipt once per interval."""
        next_system_sample = self.started + self.system_sample_seconds
        latest_clock: float | None = None
        while not self._stop.is_set():
            now = monotonic()
            with self._lock:
                last_progress = self.last_progress_at
                steps = self.last_steps
            elapsed = now - self.started
            since_progress = now - last_progress
            current_memory, peak_memory = _process_memory_bytes()
            system_sample: dict[str, Any] = {
                "cpu_frequency_mhz": None,
                "cpu_utilization_percent": None,
                "cpu_performance_percent": None,
                "cpu_maximum_frequency_percent": None,
                "acpi_thermal_zones_celsius": [],
                "busiest_logical_processors": [],
                "gpu": None,
            }
            if now >= next_system_sample:
                system_sample.update(_powershell_system_counters())
                system_sample["gpu"] = _gpu_telemetry()
                clock = system_sample.get("cpu_frequency_mhz")
                if clock is not None:
                    latest_clock = float(clock)
                next_system_sample = now + self.system_sample_seconds
            sample = {
                "elapsed_seconds": elapsed,
                "trained_steps": steps,
                "aggregate_trained_transitions_per_second": (
                    steps / elapsed if elapsed > 0.0 else 0.0
                ),
                "seconds_since_completed_iteration": since_progress,
                "process_working_set_bytes": current_memory,
                "process_peak_working_set_bytes": peak_memory,
                **system_sample,
            }
            with self._lock:
                self.system_samples.append(sample)
                if steps > 0 and since_progress > self.stall_seconds:
                    self.stall_samples.append(sample)
                progress = {
                    "schema_version": 1,
                    "stage": "D3_A10_native_operational_soak",
                    "status": "running",
                    "requested_duration_seconds": self.duration_seconds,
                    "latest": sample,
                    "completed_iterations": len(self.iterations),
                    "stall_samples": len(self.stall_samples),
                    "iteration_samples": list(self.iterations),
                    "system_samples": list(self.system_samples),
                    "stall_sample_details": list(self.stall_samples),
                }
            _atomic_json(self.progress_path, progress)
            rss = current_memory / _MIB if current_memory is not None else float("nan")
            print(
                f"A10 soak heartbeat elapsed={elapsed:.1f}s trained={steps} "
                f"rate={sample['aggregate_trained_transitions_per_second']:.1f}/s "
                f"rss={rss:.1f}MiB cpu_clock={latest_clock}MHz "
                f"since_iteration={since_progress:.1f}s",
                flush=True,
            )
            if self._stop.wait(self.sample_seconds):
                break


def _window_rates(iterations: list[dict[str, Any]], duration: float) -> list[float]:
    """Return trained throughput for six equal wall-time windows."""
    if not iterations:
        return []
    boundaries = np.linspace(0.0, duration, 7)
    rates: list[float] = []
    previous_steps = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        rows = [row for row in iterations if float(row["elapsed_seconds"]) <= end]
        end_steps = int(rows[-1]["steps"]) if rows else previous_steps
        rates.append(float((end_steps - previous_steps) / (end - start)))
        previous_steps = end_steps
    return rates


def _median_memory_growth(samples: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    """Compare early and late post-warmup working-set medians."""
    values = [
        (float(row["elapsed_seconds"]), int(row["process_working_set_bytes"]))
        for row in samples
        if row.get("process_working_set_bytes") is not None
    ]
    if len(values) < 6:
        return None, None
    usable = values[max(1, len(values) // 6) :]
    width = max(2, len(usable) // 5)
    early = statistics.median(value for _elapsed, value in usable[:width])
    late = statistics.median(value for _elapsed, value in usable[-width:])
    growth = int(late - early)
    return growth, growth / early if early else None


def _clock_ratio(samples: list[dict[str, Any]]) -> float | None:
    """Compare late and early sustained CPU-frequency medians."""
    clocks = [
        float(row["cpu_performance_percent"])
        for row in samples
        if row.get("cpu_performance_percent") is not None
    ]
    if len(clocks) < 6:
        clocks = [
            float(row["cpu_frequency_mhz"])
            for row in samples
            if row.get("cpu_frequency_mhz") is not None
        ]
    if len(clocks) < 6:
        return None
    width = max(2, len(clocks) // 3)
    early = statistics.median(clocks[:width])
    late = statistics.median(clocks[-width:])
    return late / early if early else None


def _config(args: argparse.Namespace) -> A8Config:
    """Return the frozen selected A10 arm with ample iterations for timed stop."""
    estimated_steps = math.ceil(args.duration_seconds * 8_000.0)
    total_timesteps = max(args.n_steps * 2, estimated_steps)
    return A8Config(
        total_timesteps=total_timesteps,
        n_steps=args.n_steps,
        batch_size=256,
        n_epochs=5,
        hidden_sizes=(256, 256),
        seat_counts=(2, 3, 4, 5, 6),
        seed=args.seed,
        device="cpu",
        collector_mode="native",
        native_workers=args.workers,
        native_ready_capacity=args.ready_capacity,
        native_refill_reserve_factor=args.refill_reserve_factor,
        head="masked",
        encoder="flat",
        margin_coef=0.0,
        anchor_share=0.0,
        stage1_enabled=False,
    )


def main() -> int:
    """Run the timed soak, verify its checkpoint, and apply operational gates."""
    args = _parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    power_throttling = _process_power_throttling(
        disable=args.disable_power_throttling
    )
    source_identity = native_source_tree_identity()
    config = _config(args)
    campaign_id = "dev-d3-a10-soak"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    started = monotonic()
    monitor = _Monitor(
        started=started,
        duration_seconds=args.duration_seconds,
        sample_seconds=args.sample_seconds,
        system_sample_seconds=args.system_sample_seconds,
        stall_seconds=args.stall_seconds,
        progress_path=progress_path,
    )
    print(
        "A10 soak started: "
        f"duration={args.duration_seconds:.0f}s workers={args.workers} "
        f"ready_capacity={args.ready_capacity} n_steps={args.n_steps} "
        f"reserve={args.refill_reserve_factor:.2f}",
        flush=True,
    )
    monitor.start()
    training_finished = started
    try:
        _policy, records = train_selfplay_a8(
            config,
            profile=True,
            on_iteration=monitor.note_iteration,
            should_stop=monitor.should_stop,
            resume=A8ResumeConfig(
                campaign_id=campaign_id,
                source_identity=source_identity,
                save_path=args.checkpoint,
                save_every_iterations=args.checkpoint_every_iterations,
            ),
        )
    finally:
        training_finished = monotonic()
        monitor.stop()
    elapsed = training_finished - started

    recipe = resolved_recipe(config, TrackGenParams())
    checkpoint = load_training_state(
        args.checkpoint,
        expected_recipe_sha256=recipe_sha256(recipe),
        campaign_id=campaign_id,
        source_identity=source_identity,
        anchor_identity=None,
    )
    checkpoint_iteration = int(checkpoint["progress"]["iteration"])
    final_iteration = int(records[-1]["iteration"])
    checkpoint_pass = checkpoint_iteration == final_iteration
    trained_steps = int(records[-1]["steps"])
    aggregate_rate = trained_steps / elapsed
    window_rates = _window_rates(monitor.iterations, elapsed)
    median_window_rate = float(
        statistics.median(window_rates) if window_rates else 0.0
    )
    minimum_window_ratio = float(
        min(window_rates) / median_window_rate if median_window_rate else 0.0
    )
    memory_growth, memory_growth_ratio = _median_memory_growth(monitor.system_samples)
    memory_limit = None
    memory_pass = memory_growth is not None and memory_growth_ratio is not None
    if memory_pass:
        early_values = [
            int(row["process_working_set_bytes"])
            for row in monitor.system_samples
            if row.get("process_working_set_bytes") is not None
        ]
        memory_limit = max(128 * _MIB, int(statistics.median(early_values) * 0.10))
        memory_pass = memory_growth <= memory_limit
    clock_ratio = _clock_ratio(monitor.system_samples)
    clock_pass = clock_ratio is not None and clock_ratio >= 0.90
    duration_pass = elapsed >= args.duration_seconds
    throughput_pass = aggregate_rate >= 5_000.0
    stability_pass = bool(minimum_window_ratio >= 0.90)
    stall_pass = bool(not monitor.stall_samples)
    passed = all(
        (
            duration_pass,
            throughput_pass,
            stability_pass,
            memory_pass,
            stall_pass,
            clock_pass,
            checkpoint_pass,
        )
    )
    receipt_text = checkpoint_receipt_path(
        published_checkpoint_path(args.checkpoint)
    ).read_text(encoding="ascii").strip()
    artifact = {
        "schema_version": 1,
        "stage": "D3_A10_native_operational_soak",
        "status": "pass" if passed else "fail",
        "decision": "operational_gate_pass" if passed else "investigate",
        "hypothesis": (
            "The selected native CPU arm sustains at least 5,000 trained "
            "transitions/s for one hour without stalls, material post-warmup "
            "memory growth, CPU clock collapse, or checkpoint corruption."
        ),
        "confidence_before_run": 0.75,
        "scope": {
            "requested_duration_seconds": args.duration_seconds,
            "sample_seconds": args.sample_seconds,
            "system_sample_seconds": args.system_sample_seconds,
            "stall_seconds": args.stall_seconds,
            "workers": args.workers,
            "ready_capacity": args.ready_capacity,
            "n_steps": args.n_steps,
            "refill_reserve_factor": args.refill_reserve_factor,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "checkpoint_every_iterations": args.checkpoint_every_iterations,
            "disable_power_throttling": args.disable_power_throttling,
            "recipe": "A8 flat/masked, hidden 256x256, PPO 5 epochs, batch 256",
            "campaign_id": campaign_id,
            "policy_disposition": "disposable dev soak; not a registered generation",
        },
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_identity": source_identity,
            "process_power_throttling": power_throttling,
        },
        "result": {
            "elapsed_seconds": elapsed,
            "trained_transitions": trained_steps,
            "completed_iterations": len(records),
            "trained_transitions_per_second": aggregate_rate,
            "six_window_rates": window_rates,
            "median_window_rate": median_window_rate,
            "minimum_window_to_median_ratio": minimum_window_ratio,
            "post_warmup_memory_growth_bytes": memory_growth,
            "post_warmup_memory_growth_ratio": memory_growth_ratio,
            "memory_growth_limit_bytes": memory_limit,
            "late_to_early_cpu_clock_ratio": clock_ratio,
            "stall_sample_count": len(monitor.stall_samples),
        },
        "gates": {
            "duration_pass": duration_pass,
            "throughput_pass": throughput_pass,
            "throughput_stability_pass": stability_pass,
            "memory_pass": memory_pass,
            "stall_pass": stall_pass,
            "cpu_clock_pass": clock_pass,
            "checkpoint_pass": checkpoint_pass,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "completed_iteration": checkpoint_iteration,
            "sha256_receipt": receipt_text,
            "complete_state_digest": training_state_digest(checkpoint),
        },
        "iterations": monitor.iterations,
        "system_samples": monitor.system_samples,
        "stall_samples": monitor.stall_samples,
        "telemetry_note": (
            "ACPI thermal-zone counters are recorded when Windows exposes them; "
            "they are not treated as CPU-die temperature. CPU clock stability is "
            "the thermal-throttling acceptance signal. GPU telemetry is passive "
            "because the selected path is CPU-only."
        ),
    }
    _atomic_json(args.output, artifact)
    progress_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "elapsed_seconds": elapsed,
                "trained_transitions": trained_steps,
                "trained_transitions_per_second": aggregate_rate,
                "gates": artifact["gates"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
