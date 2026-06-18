"""Sprint 8C: OpponentSchedule unit tests (§9)."""

from __future__ import annotations

import pickle

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.ml.training import (
    OpponentSchedule,
    OpponentStage,
    _StrongHeuristicFactory,
    _mixed_strength_pool,
)


def _three_stage() -> OpponentSchedule:
    return OpponentSchedule(
        stages=(
            OpponentStage("weak", 100),
            OpponentStage("mixed", 100),
            OpponentStage("strong", 100),
        )
    )


def _kinds(pool):
    """Resolve a pool of zero-arg factories/classes to instance class names."""
    names = []
    for spec in pool:
        agent = spec() if callable(spec) else spec
        names.append(type(agent).__name__)
    return names


def test_stage_boundaries():
    """Stages 0/1/2 give weak/mixed/strong; out-of-range clamps to the last."""
    sch = _three_stage()
    n = 5  # 4 opponent seats -> the strong template cycles through Random too
    assert "StrongHeuristicAgent" not in _kinds(sch.pool_for(n, 0))  # weak
    # mixed has a strong rung but also weak + random
    mixed = _kinds(sch.pool_for(n, 1))
    assert "StrongHeuristicAgent" in mixed
    assert "HeuristicAgent" in mixed
    # strong: the broadened pool
    assert "StrongHeuristicAgent" in _kinds(sch.pool_for(n, 2))
    # out-of-range clamps to the final (strong) stage (same pool composition).
    assert _kinds(sch.pool_for(n, 99)) == _kinds(sch.pool_for(n, 2))


def test_pool_composition():
    """Weak has no strong agent; strong has Strong + Random; sizes are n-1."""
    sch = _three_stage()
    for n in (2, 4, 6):
        weak = sch.pool_for(n, 0)
        strong = sch.pool_for(n, 2)
        mixed = sch.pool_for(n, 1)
        assert len(weak) == n - 1
        assert len(strong) == n - 1
        assert len(mixed) == n - 1
        assert "StrongHeuristicAgent" not in _kinds(weak)

    # Strong pool spans strengths 2 & 3 (broadened). With 4 opponent seats
    # (5 players) the [Strong3, Strong2, Heuristic, Random] template also yields
    # a Random; with fewer seats it may not cycle that far.
    strong5 = sch.pool_for(5, 2)
    strengths = {
        getattr(spec, "strength", None)
        for spec in strong5
        if isinstance(spec, _StrongHeuristicFactory)
    }
    assert strengths == {2, 3}
    assert "RandomAgent" in _kinds(strong5)


def test_schedule_picklable():
    """The schedule and every factory it produces survive pickling (SubprocVecEnv)."""
    sch = _three_stage()
    round_tripped = pickle.loads(pickle.dumps(sch))
    assert round_tripped == sch

    for stage in range(3):
        for spec in sch.pool_for(4, stage):
            # Each opponent spec must itself pickle (it crosses the spawn boundary).
            pickle.loads(pickle.dumps(spec))


def test_mixed_pool_has_expected_rungs():
    """_mixed_strength_pool cycles Strong(2)/Heuristic/Random across the seats."""
    pool = _mixed_strength_pool(4)
    assert len(pool) == 3
    kinds = _kinds(pool)
    assert "StrongHeuristicAgent" in kinds
    assert "HeuristicAgent" in kinds
    assert "RandomAgent" in kinds
