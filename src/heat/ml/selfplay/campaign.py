"""Deterministic process supervisor for independent A8 training runs."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from time import perf_counter, process_time
from typing import Callable, Sequence

import torch

from heat.ml.selfplay.phase1 import A8Config, train_selfplay_a8
from heat.ml.selfplay.policy import PPOPolicy


@dataclass(frozen=True)
class CampaignRun:
    """One fully resolved independent A8 training job."""

    run_id: str
    config: A8Config


@dataclass(frozen=True)
class CampaignResult:
    """Stable identity and measurements returned by one campaign worker."""

    run_id: str
    seed: int
    recorded_steps: int
    games: int
    wall_seconds: float
    cpu_seconds: float
    policy_sha256: str


ProgressCallback = Callable[[str, str | None, float], None]


def canonical_policy_sha256(policy: PPOPolicy) -> str:
    """Hash tensor names, dtypes, shapes, and raw CPU bytes canonically."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _configure_worker() -> None:
    """Give every spawned worker the same non-oversubscribed Torch shape."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _run_campaign_job(run: CampaignRun) -> CampaignResult:
    """Train one resolved job and return only disposable measurements."""
    started = perf_counter()
    cpu_started = process_time()
    policy, records = train_selfplay_a8(run.config)
    if not records:
        raise RuntimeError("training returned no iteration records")
    final = records[-1]
    return CampaignResult(
        run_id=run.run_id,
        seed=int(run.config.seed or 0),
        recorded_steps=int(final["steps"]),
        games=int(final["total_games"]),
        wall_seconds=perf_counter() - started,
        cpu_seconds=process_time() - cpu_started,
        policy_sha256=canonical_policy_sha256(policy),
    )


def _stop_executor(executor: ProcessPoolExecutor) -> None:
    """Stop running children after a deadline instead of leaking benchmark work."""
    processes = list(getattr(executor, "_processes", {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def run_campaign(
    runs: Sequence[CampaignRun],
    *,
    max_workers: int,
    timeout_seconds: float,
    progress: ProgressCallback | None = None,
    heartbeat_seconds: float = 15.0,
) -> list[CampaignResult]:
    """Run resolved independent A8 jobs and return run-id-sorted results."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("campaign run IDs must be unique")
    if not runs:
        return []

    started = perf_counter()
    last_heartbeat = started
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_configure_worker,
    )
    futures: dict[Future[CampaignResult], str] = {
        executor.submit(_run_campaign_job, run): run.run_id for run in runs
    }
    if progress is not None:
        for run_id in run_ids:
            progress("started", run_id, 0.0)
    results: list[CampaignResult] = []
    try:
        pending = set(futures)
        while pending:
            elapsed = perf_counter() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0.0:
                outstanding = sorted(futures[future] for future in pending)
                raise TimeoutError(
                    f"campaign exceeded {timeout_seconds:.1f}s; outstanding={outstanding}"
                )
            done, pending = wait(
                pending,
                timeout=min(1.0, remaining),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                run_id = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    raise RuntimeError(f"campaign run {run_id!r} failed") from exc
                results.append(result)
                if progress is not None:
                    progress("completed", run_id, perf_counter() - started)
            now = perf_counter()
            if progress is not None and pending and now - last_heartbeat >= heartbeat_seconds:
                progress("heartbeat", None, now - started)
                last_heartbeat = now
    except BaseException:
        _stop_executor(executor)
        raise
    else:
        executor.shutdown(wait=True)
    return sorted(results, key=lambda result: result.run_id)
