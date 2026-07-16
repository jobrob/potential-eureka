"""Focused correctness tests for T1 process-level parallelism."""

from __future__ import annotations

import copy

import pytest
import torch

from heat.agents.heuristic_agent import HeuristicAgent
from heat.ml.selfplay.campaign import (
    CampaignRun,
    canonical_policy_sha256,
    run_campaign,
)
from heat.ml.selfplay.eval_harness import evaluate_policy
from heat.ml.selfplay.phase1 import A8Config
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.selfplay.tiny_heat import tiny_heat_track


def _tiny_config(seed: int, *, anchor_share: float = 0.0) -> A8Config:
    """Return a fast full-game config that still exercises real A8 training."""
    return A8Config(
        total_timesteps=16,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        hidden_sizes=(16,),
        seat_counts=(2,),
        snapshot_every=100,
        pool_capacity=1,
        pool_prob=0.0,
        anchor_share=anchor_share,
        device="cpu",
        seed=seed,
    )


def test_canonical_hash_detects_tensor_change() -> None:
    """Canonical hashes ignore object identity but detect changed weights."""
    policy = build_policy(A0Config(hidden_sizes=(16,)))
    clone = copy.deepcopy(policy)
    assert canonical_policy_sha256(policy) == canonical_policy_sha256(clone)
    with torch.no_grad():
        next(clone.parameters()).view(-1)[0] += 1.0
    assert canonical_policy_sha256(policy) != canonical_policy_sha256(clone)


def test_campaign_is_exact_and_sorted_across_worker_counts() -> None:
    """One- and two-worker runs preserve hashes, counters, and run ordering."""
    runs = [
        CampaignRun("dev-t1-R01", _tiny_config(1)),
        CampaignRun("dev-t1-R00", _tiny_config(0)),
    ]
    sequential = run_campaign(runs, max_workers=1, timeout_seconds=300.0)
    concurrent = run_campaign(runs, max_workers=2, timeout_seconds=300.0)
    assert [result.run_id for result in sequential] == [
        "dev-t1-R00",
        "dev-t1-R01",
    ]
    signature = lambda result: (  # noqa: E731 - compact comparison projection
        result.run_id,
        result.seed,
        result.recorded_steps,
        result.games,
        result.policy_sha256,
    )
    assert [signature(result) for result in sequential] == [
        signature(result) for result in concurrent
    ]


def test_parallel_evaluation_equals_sequential_and_preserves_rng() -> None:
    """Parallel cells reproduce the ordered report without consuming parent RNG."""
    torch.manual_seed(81)
    policy = build_policy(A0Config(hidden_sizes=(16,)))
    kwargs = {
        "opponents": {"weak": HeuristicAgent},
        "seat_counts": (2, 4),
        "splits": {"tiny": tiny_heat_track()},
        "games_per_cell": 2,
        "seed": 17,
        "device": torch.device("cpu"),
    }
    sequential = evaluate_policy(policy, **kwargs)
    before = torch.random.get_rng_state().clone()
    parallel = evaluate_policy(policy, **kwargs, parallel=True, max_workers=2)
    after = torch.random.get_rng_state()
    assert parallel == sequential
    assert torch.equal(after, before)


def test_parallel_evaluation_rejects_invalid_inputs_before_spawn() -> None:
    """CUDA and the first unpicklable factory fail with clear input names."""
    policy = build_policy(A0Config(hidden_sizes=(16,)))
    with pytest.raises(ValueError, match="CPU policies only"):
        evaluate_policy(policy, parallel=True, device=torch.device("cuda"))
    with pytest.raises(TypeError, match=r"opponents\['bad'\]"):
        evaluate_policy(
            policy,
            opponents={"bad": lambda: HeuristicAgent()},
            splits={"tiny": tiny_heat_track()},
            games_per_cell=1,
            parallel=True,
        )


def test_campaign_worker_failure_names_the_run() -> None:
    """A worker-side trainer error propagates with its stable run identity."""
    run = CampaignRun("dev-t1-broken", _tiny_config(0, anchor_share=1.0))
    with pytest.raises(RuntimeError, match="dev-t1-broken"):
        run_campaign([run], max_workers=1, timeout_seconds=300.0)
