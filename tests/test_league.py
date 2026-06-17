"""Unit tests for the Sprint 6D opponent league + PFSP sampler.

Covers the four ``test_league.py`` gate bullets from
``docs/sprint6d-league-selfplay-design.md``:

1. Retention is deterministic & seeded, keeps highest-value entries, and is
   provably NOT FIFO (a constructed case where FIFO order and value order
   disagree).
2. The current-best anchor is never evicted.
3. PFSP weights are >= 0, sum to 1, respect ``p_min``/``p_max`` clamping, and
   rank correctly (a 20%-win-rate opponent outweighs an 80% one under "hard";
   the "even"/"variance" variant peaks near 0.5).
4. Win-rate bookkeeping: ``record_result`` updates ``games``/``wins_vs_learner``
   and the derived win-rate; a never-played entry uses the neutral prior.

These are fast (no training, no SB3) and run in the default gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from heat.ml.league import NEUTRAL_PRIOR, League, LeagueEntry


def _entry(idx: int, *, gate_score=None, games=0, wins=0, anchor=False) -> LeagueEntry:
    return LeagueEntry(
        path=f"snap{idx}",
        snapshot_index=idx,
        gate_score=gate_score,
        games=games,
        wins_vs_learner=wins,
        is_anchor=anchor,
    )


# ---------------------------------------------------------------------------
# (1) Retention: deterministic, value-based (not FIFO)
# ---------------------------------------------------------------------------


def test_retention_is_not_fifo_keeps_highest_value() -> None:
    """A constructed FIFO-vs-value disagreement proves retention is value-based.

    The OLDEST entry (snapshot_index 0) is the STRONGEST (highest gate_score).
    FIFO would evict it first; the keep-strong policy must keep it and evict a
    weak recent one instead.
    """
    league = League(
        capacity=2, strength_weight=1.0, diversity_weight=0.0  # strength only
    )
    league.add(_entry(0, gate_score=0.9))  # oldest, strongest
    league.add(_entry(1, gate_score=0.2))  # weak
    league.add(_entry(2, gate_score=0.3))  # weak-ish, newest

    league.retain()

    kept = {e.path for e in league.entries}
    assert len(league.entries) == 2
    # Strongest (oldest) survives -> not FIFO. The weakest (snap1) is evicted.
    assert "snap0" in kept
    assert "snap1" not in kept


def test_retention_is_deterministic_across_repeated_calls() -> None:
    """Same adds + same seed -> same retained set; retain is idempotent."""

    def build() -> League:
        lg = League(capacity=3)
        for i, gs in enumerate([0.5, 0.1, 0.9, 0.4, 0.7]):
            lg.add(_entry(i, gate_score=gs))
        return lg

    lg1 = build()
    lg1.retain()
    first = [e.path for e in lg1.entries]

    # Calling retain again must not change anything (idempotent under capacity).
    lg1.retain()
    assert [e.path for e in lg1.entries] == first

    # Rebuilding from scratch yields the identical retained set (deterministic).
    lg2 = build()
    lg2.retain()
    assert [e.path for e in lg2.entries] == first


def test_retention_tiebreak_uses_snapshot_index() -> None:
    """Equal value -> keep the lower snapshot_index deterministically."""
    league = League(capacity=2, strength_weight=1.0, diversity_weight=0.0)
    # All identical gate scores -> pure tie; tie-break is snapshot_index asc.
    for i in range(4):
        league.add(_entry(i, gate_score=0.5))
    league.retain()
    kept = sorted(e.snapshot_index for e in league.entries)
    assert kept == [0, 1]  # lowest indices retained on a tie


# ---------------------------------------------------------------------------
# (2) Best anchor never evicted
# ---------------------------------------------------------------------------


def test_anchor_never_evicted_even_when_weakest() -> None:
    """The anchor survives retention even if it is the lowest-value entry."""
    league = League(capacity=2, strength_weight=1.0, diversity_weight=0.0)
    league.add(_entry(0, gate_score=0.01, anchor=True))  # weak BUT anchor
    league.add(_entry(1, gate_score=0.9))
    league.add(_entry(2, gate_score=0.8))

    league.retain()

    kept = {e.path for e in league.entries}
    assert "snap0" in kept  # anchor retained despite being weakest
    assert len(league.entries) == 2


def test_set_anchor_is_exclusive() -> None:
    league = League(capacity=8)
    league.add(_entry(0))
    league.add(_entry(1))
    league.set_anchor("snap0")
    league.set_anchor("snap1")  # moves anchor
    anchors = [e.path for e in league.entries if e.is_anchor]
    assert anchors == ["snap1"]


# ---------------------------------------------------------------------------
# (3) PFSP weights: >= 0, sum to 1, clamping, ranking
# ---------------------------------------------------------------------------


def test_pfsp_weights_nonnegative_and_sum_to_one() -> None:
    league = League(pfsp_mode="even", min_games=1)
    league.add(_entry(0, games=10, wins=5))  # learner wr 0.5
    league.add(_entry(1, games=10, wins=2))  # learner wr 0.8
    league.add(_entry(2, games=10, wins=9))  # learner wr 0.1

    w = league.weights()
    assert all(v >= 0.0 for v in w.values())
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)


def test_pfsp_hard_ranks_low_win_rate_higher() -> None:
    """Under "hard", a 20%-learner-win-rate opponent outweighs an 80% one."""
    league = League(pfsp_mode="hard", pfsp_exponent=2.0, min_games=1)
    # learner wins 2/10 vs hard_opp -> learner wr 0.2 (the opp the learner loses to)
    league.add(_entry(0, games=10, wins=8))  # learner wr 0.2  -> "hard"
    # learner wins 8/10 vs easy_opp -> learner wr 0.8
    league.add(_entry(1, games=10, wins=2))  # learner wr 0.8  -> "easy"

    w = league.weights()
    assert w["snap0"] > w["snap1"]


def test_pfsp_even_peaks_near_half() -> None:
    """The "even"/"variance" weight is maximal for a ~0.5 win-rate opponent."""
    league = League(pfsp_mode="even", min_games=1)
    league.add(_entry(0, games=10, wins=5))  # learner wr 0.5  -> peak
    league.add(_entry(1, games=10, wins=2))  # learner wr 0.8
    league.add(_entry(2, games=10, wins=9))  # learner wr 0.1

    w = league.weights()
    assert w["snap0"] > w["snap1"]
    assert w["snap0"] > w["snap2"]


def test_pfsp_variance_alias_matches_even() -> None:
    even = League(pfsp_mode="even", min_games=1)
    var = League(pfsp_mode="variance", min_games=1)
    for lg in (even, var):
        lg.add(_entry(0, games=10, wins=3))
        lg.add(_entry(1, games=10, wins=7))
    assert even.weights() == var.weights()


def test_pfsp_clamping_bounds_every_weight() -> None:
    """p_min/p_max clamp every per-entry probability (feasible band).

    Without clamping the wr=0 opponent would dominate (~all the mass under
    "hard"). With ``p_max=0.5`` it is capped, the residual flows to the others,
    and ``p_min=0.1`` keeps the easy opponent from being starved -- while the
    distribution still sums to 1.
    """
    league = League(pfsp_mode="hard", pfsp_exponent=3.0, min_games=1,
                    p_min=0.1, p_max=0.5)
    league.add(_entry(0, games=10, wins=10))  # learner wr 0.0 -> huge raw weight
    league.add(_entry(1, games=10, wins=0))   # learner wr 1.0 -> ~0 raw weight
    league.add(_entry(2, games=10, wins=5))   # learner wr 0.5

    w = league.weights()
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    # Every entry stays within the clamp band: no single opponent dominates
    # (p_max), none is fully starved (p_min).
    for v in w.values():
        assert v >= 0.1 - 1e-9
        assert v <= 0.5 + 1e-9
    # The dominant (wr=0) opponent is pinned at the p_max ceiling.
    assert math.isclose(w["snap0"], 0.5, rel_tol=1e-6)


def test_pfsp_all_zero_priority_falls_back_to_uniform() -> None:
    """When every "even" priority is 0 (all wr in {0,1}), weights are uniform."""
    league = League(pfsp_mode="even", min_games=1)
    league.add(_entry(0, games=10, wins=0))   # learner wr 1.0 -> wr*(1-wr)=0
    league.add(_entry(1, games=10, wins=10))  # learner wr 0.0 -> 0
    w = league.weights()
    assert math.isclose(w["snap0"], 0.5, rel_tol=1e-9)
    assert math.isclose(w["snap1"], 0.5, rel_tol=1e-9)


def test_sample_is_seeded_and_reproducible() -> None:
    def build() -> League:
        lg = League(pfsp_mode="even", min_games=1)
        lg.add(_entry(0, games=10, wins=5))
        lg.add(_entry(1, games=10, wins=2))
        lg.add(_entry(2, games=10, wins=8))
        return lg

    a = build().sample(20, np.random.default_rng(123))
    b = build().sample(20, np.random.default_rng(123))
    assert a == b  # same seed -> identical draw
    assert all(p in {"snap0", "snap1", "snap2"} for p in a)


def test_sample_empty_or_zero_k() -> None:
    assert League().sample(3, np.random.default_rng(0)) == []
    lg = League()
    lg.add(_entry(0))
    assert lg.sample(0, np.random.default_rng(0)) == []


# ---------------------------------------------------------------------------
# (4) Win-rate bookkeeping
# ---------------------------------------------------------------------------


def test_record_result_updates_games_and_wins() -> None:
    league = League(min_games=1)
    league.add(_entry(0))
    league.record_result("snap0", learner_won=True)   # entry loses
    league.record_result("snap0", learner_won=False)  # entry wins
    league.record_result("snap0", learner_won=False)  # entry wins

    e = league.entries[0]
    assert e.games == 3
    assert e.wins_vs_learner == 2
    # learner won 1 of 3 -> learner win-rate 1/3.
    assert math.isclose(e.learner_win_rate(min_games=1), 1.0 / 3.0, rel_tol=1e-9)


def test_unplayed_entry_uses_neutral_prior() -> None:
    league = League(min_games=2)
    league.add(_entry(0))
    e = league.entries[0]
    assert e.learner_win_rate(min_games=2) == NEUTRAL_PRIOR
    # One game recorded but below min_games -> still neutral prior.
    league.record_result("snap0", learner_won=True)
    assert e.learner_win_rate(min_games=2) == NEUTRAL_PRIOR
    # Reaching min_games switches to the observed rate.
    league.record_result("snap0", learner_won=True)
    assert math.isclose(e.learner_win_rate(min_games=2), 1.0, rel_tol=1e-9)


def test_record_result_unknown_path_is_noop() -> None:
    league = League()
    league.add(_entry(0))
    league.record_result("does-not-exist", learner_won=True)  # dropped silently
    assert league.entries[0].games == 0


def test_record_win_rate_folds_aggregate_estimate() -> None:
    league = League(min_games=1)
    league.add(_entry(0))
    league.record_win_rate("snap0", 0.75, games=8)  # learner wins 6 of 8
    e = league.entries[0]
    assert e.games == 8
    assert e.wins_vs_learner == 2  # entry won the other 2
    assert math.isclose(e.learner_win_rate(min_games=1), 0.75, rel_tol=1e-9)


def test_add_same_path_preserves_bookkeeping() -> None:
    """Re-adding a path updates metadata but keeps accumulated win-rate stats."""
    league = League()
    league.add(_entry(0, gate_score=0.5))
    league.record_result("snap0", learner_won=False)
    league.record_result("snap0", learner_won=True)
    # Re-add same path with new metadata.
    league.add(LeagueEntry(path="snap0", snapshot_index=5, gate_score=0.9))
    e = league.entries[0]
    assert len(league.entries) == 1
    assert e.snapshot_index == 5
    assert e.gate_score == 0.9
    assert e.games == 2  # bookkeeping preserved
    assert e.wins_vs_learner == 1
