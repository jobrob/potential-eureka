"""Tests for the HEAT RL evaluation harness (Sprint 5d).

Covers :mod:`heat.ml.evaluate`:
* ``ml_agent_factory`` is picklable (a top-level ``functools.partial``, not a
  lambda/closure) and carries only the model *path* (§6.5).
* ``evaluate_ml`` runs ``run_batch`` on a tiny throwaway model and produces
  ``AgentStats`` with a sane ``win_rate`` in [0, 1] and ``games_played`` matching
  the batch size.

The "MLAgent beats heuristic" success metric is a MANUAL eval (§6.2/§7), NOT
asserted here.
"""

from __future__ import annotations

import functools
import pickle

import pytest

from heat.agents.ml_agent import MLAgent
from heat.ml.evaluate import (
    evaluate_cross_track,
    evaluate_ml,
    head_to_head,
    ml_agent_factory,
    round_robin_elo,
)
from heat.simulation.runner import (
    GameOutcome,
    PlayerOutcome,
    heuristic_agent_factory,
    random_agent_factory,
)
from heat.simulation.stats import (
    compute_elo,
    compute_trueskill,
    intervals_overlap,
    wilson_interval,
)
from heat.tracks.loader import load_track_by_name


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> str:
    """Train and save a tiny throwaway model once for this module."""
    from heat.ml.model import PPOConfig
    from heat.ml.training import smoke_train

    ckpt = str(tmp_path_factory.mktemp("ckpt") / "tiny_model")
    config = PPOConfig(
        net_arch=[16, 16],
        features_extractor_hidden=[16],
        features_dim=16,
        n_steps=64,
        batch_size=32,
        seed=0,
        verbose=0,
    )
    _model, saved = smoke_train(
        total_timesteps=64,
        num_players=2,
        checkpoint_path=ckpt,
        config=config,
    )
    assert saved == ckpt
    return ckpt


def test_ml_agent_factory_is_picklable_and_path_only() -> None:
    """The factory must pickle (top-level partial) and carry only the path."""
    factory = ml_agent_factory("some/path", name="ML")
    assert isinstance(factory, functools.partial)

    restored = pickle.loads(pickle.dumps(factory))
    agent = restored(0, None)
    assert isinstance(agent, MLAgent)
    assert agent.model_path == "some/path"
    assert agent._model is None  # lazy: nothing loaded just by constructing


@pytest.mark.slow
def test_evaluate_ml_produces_sane_stats(tiny_checkpoint) -> None:
    """The eval harness runs run_batch and returns AgentStats with a sane
    win_rate in [0, 1] and games_played matching the batch size."""
    num_games = 6
    per_agent = evaluate_ml(
        tiny_checkpoint,
        num_games=num_games,
        num_players=2,
        track="usa",
        seed=0,
        parallel=False,
    )

    assert "MLAgent" in per_agent
    assert "HeuristicAgent" in per_agent

    ml_stats = per_agent["MLAgent"]
    assert ml_stats.games_played == num_games
    assert 0.0 <= ml_stats.win_rate <= 1.0
    assert ml_stats.wins == sum(
        c for pos, c in ml_stats.finish_position_counts.items() if pos == 1
    )

    # Two seats, one MLAgent + one HeuristicAgent: wins partition across them.
    total_wins = sum(s.wins for s in per_agent.values())
    assert total_wins == num_games


# ======================================================================
# Sprint 6B: evaluation overhaul gates
# ======================================================================


def _make_outcome(
    game_index: int, finish_order: tuple[int, ...], agent_types: dict[int, str]
) -> GameOutcome:
    """Build a minimal GameOutcome for pure-aggregation tests.

    ``finish_order`` is best-to-worst player ids; ``agent_types`` maps each
    player id to its agent label.
    """
    n = len(finish_order)
    rank_of = {pid: i + 1 for i, pid in enumerate(finish_order)}
    players = tuple(
        PlayerOutcome(
            player_id=pid,
            name=f"p{pid}",
            agent_type=agent_types[pid],
            finish_position=rank_of[pid],
            final_lap=1,
            final_position=0,
            heat_remaining=0,
        )
        for pid in sorted(agent_types)
    )
    return GameOutcome(
        game_index=game_index,
        seed=game_index,
        num_players=n,
        winner_id=finish_order[0],
        winner_name=f"p{finish_order[0]}",
        finish_order=finish_order,
        total_rounds=1,
        players=players,
    )


def _strong_beats_weak_outcomes(
    n: int, *, strong_wins_frac: float = 0.8
) -> list[GameOutcome]:
    """A synthetic batch where 'Strong' beats 'Weak' in ``strong_wins_frac``.

    Wins are *interleaved* (not blocked) so that online ratings like TrueSkill
    -- which are inherently sequence-sensitive -- see a realistic, stationary
    win stream rather than a run of wins followed by a run of upsets.
    """
    types = {0: "Strong", 1: "Weak"}
    outcomes: list[GameOutcome] = []
    # Deterministic interleave: distribute Weak's wins evenly across the batch.
    weak_wins = n - int(round(n * strong_wins_frac))
    weak_win_indices = (
        {round(i * n / weak_wins) for i in range(weak_wins)}
        if weak_wins
        else set()
    )
    for gi in range(n):
        order = (1, 0) if gi in weak_win_indices else (0, 1)
        outcomes.append(_make_outcome(gi, order, types))
    return outcomes


# ---- ELO determinism --------------------------------------------------


def test_elo_deterministic_and_order_independent() -> None:
    """Same outcomes -> identical ELO; permuting the input order is irrelevant."""
    outcomes = _strong_beats_weak_outcomes(40)

    r1 = compute_elo(outcomes, bootstrap=50, bootstrap_seed=7)
    r2 = compute_elo(list(reversed(outcomes)), bootstrap=50, bootstrap_seed=7)

    assert set(r1) == {"Strong", "Weak"}
    for label in r1:
        assert r1[label].rating == pytest.approx(r2[label].rating)
        assert r1[label].rating_ci == pytest.approx(r2[label].rating_ci)

    # Stronger agent rated higher.
    assert r1["Strong"].rating > r1["Weak"].rating
    # Bootstrap CI is deterministic given the seed.
    r3 = compute_elo(outcomes, bootstrap=50, bootstrap_seed=7)
    assert r1["Strong"].rating_ci == r3["Strong"].rating_ci


def test_elo_bootstrap_seed_determinism() -> None:
    """Bootstrap ELO CIs depend only on the (fixed) bootstrap seed."""
    outcomes = _strong_beats_weak_outcomes(30)
    a = compute_elo(outcomes, bootstrap=100, bootstrap_seed=123)
    b = compute_elo(outcomes, bootstrap=100, bootstrap_seed=123)
    assert a["Strong"].rating_ci == b["Strong"].rating_ci
    lo, hi = a["Strong"].rating_ci
    assert lo <= a["Strong"].rating <= hi


# ---- Confidence intervals: Wilson sanity ------------------------------


def test_wilson_interval_bounds_and_degenerate() -> None:
    """0 <= lo <= rate <= hi <= 1; degenerate batches stay valid (no NaN)."""
    eps = 1e-9
    for wins, games in [(7, 10), (0, 10), (10, 10), (1, 3), (50, 100)]:
        lo, hi = wilson_interval(wins, games)
        rate = wins / games
        assert -eps <= lo <= rate + eps
        assert rate - eps <= hi <= 1.0 + eps
        assert lo == lo and hi == hi  # not NaN

    # All-wins / all-losses degenerate batches: still a valid (non-NaN) interval.
    # Unlike the normal approximation, Wilson does NOT collapse to a zero-width
    # spike at 0/1 -- it keeps a finite interval acknowledging the uncertainty.
    lo_w, hi_w = wilson_interval(5, 5)
    assert 0.0 < lo_w < 1.0 and hi_w == pytest.approx(1.0, abs=1e-9)
    lo_l, hi_l = wilson_interval(0, 5)
    assert lo_l == pytest.approx(0.0, abs=1e-9) and 0.0 < hi_l < 1.0

    # Zero games -> maximal-uncertainty interval, never NaN.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_n() -> None:
    """Larger N at the same rate yields a strictly narrower interval."""
    small = wilson_interval(6, 10)
    large = wilson_interval(600, 1000)
    width_small = small[1] - small[0]
    width_large = large[1] - large[0]
    assert width_large < width_small


def test_intervals_overlap_flag() -> None:
    assert intervals_overlap((0.4, 0.6), (0.55, 0.7)) is True
    assert intervals_overlap((0.4, 0.6), (0.61, 0.7)) is False
    assert intervals_overlap((0.0, 1.0), (0.3, 0.5)) is True


# ---- Head-to-head symmetry -------------------------------------------


def test_head_to_head_symmetry_and_seat_cancellation() -> None:
    """A-vs-B and B-vs-A on the same seeds give consistent, mirrored rates.

    Heuristic vs Random over a dual-seat batch: the win rates of the two agents
    must sum to 1 (HEAT has no draws), and dual-seat running must not leave a
    large residual seat-0 bias (the two seat orders are symmetric).
    """
    hf = heuristic_agent_factory()
    rf = random_agent_factory()

    ab = head_to_head(
        hf, rf, label_a="Heuristic", label_b="Random",
        num_games=40, track="usa", seed=0,
    )
    ba = head_to_head(
        rf, hf, label_a="Random", label_b="Heuristic",
        num_games=40, track="usa", seed=0,
    )

    # No draws: A and B win rates partition all decisive games.
    assert ab.a_wins + ab.b_wins == ab.games
    assert ab.a_win_rate + (ab.b_wins / ab.games) == pytest.approx(1.0)

    # Heuristic should beat Random comfortably and consistently from either
    # framing: Heuristic-as-A (ab) and Heuristic-as-B (ba) both > 0.5.
    assert ab.a_win_rate > 0.5
    heuristic_rate_as_b = ba.b_wins / ba.games
    assert heuristic_rate_as_b > 0.5


def test_head_to_head_no_large_seat_bias() -> None:
    """Two identical agents head-to-head should be near 50/50 after dual-seat."""
    rf0 = random_agent_factory()
    rf1 = random_agent_factory()
    h = head_to_head(rf0, rf1, num_games=80, track="usa", seed=3)
    # Dual-seat cancels first-mover advantage: with identical random agents the
    # split should be roughly even (generous band for a modest batch).
    assert 0.30 <= h.a_win_rate <= 0.70
    lo, hi = h.a_win_rate_ci
    assert 0.0 <= lo <= h.a_win_rate <= hi <= 1.0


# ---- Cross-track aggregation -----------------------------------------


def test_evaluate_cross_track_keys_and_bounds() -> None:
    """Per-track stats dict has the right track keys and win-rates in [0,1]."""
    usa = load_track_by_name("usa")
    silverstone = load_track_by_name("silverstone")
    factories = {
        "Heuristic": heuristic_agent_factory(),
        "Random": random_agent_factory(),
    }
    result = evaluate_cross_track(
        factories, tracks=[usa, silverstone], num_games=10, seed=0
    )
    assert set(result) == {usa.name, silverstone.name}
    for track_name, per_agent in result.items():
        assert {"HeuristicAgent", "RandomAgent"} <= set(per_agent)
        for s in per_agent.values():
            assert 0.0 <= s.win_rate <= 1.0
            assert s.games_played == 10


# ---- Sanity ranking with significance --------------------------------


def test_round_robin_sanity_ranking_significant() -> None:
    """Heuristic > Random over a large seeded batch, with non-overlapping CIs."""
    factories = {
        "Heuristic": heuristic_agent_factory(),
        "Random": random_agent_factory(),
    }
    rr = round_robin_elo(
        factories, num_games_per_pair=60, num_players=2, seed=0,
        rating="elo", bootstrap=200,
    )
    assert set(rr.ratings) == {"Heuristic", "Random"}
    assert rr.ratings["Heuristic"].rating > rr.ratings["Random"].rating

    # At this batch size the ranking is real: ELO CIs do not overlap.
    assert not intervals_overlap(
        rr.ratings["Heuristic"].rating_ci, rr.ratings["Random"].rating_ci
    )

    # Pairwise table is present and consistent.
    assert ("Heuristic", "Random") in rr.pairwise
    h2h = rr.pairwise[("Heuristic", "Random")]
    assert h2h.a_wins + h2h.b_wins == h2h.games
    assert h2h.a_win_rate > 0.5


def test_round_robin_tiny_batch_flags_not_significant() -> None:
    """On a tiny batch the CI machinery refuses to over-claim a ranking.

    Two identical random agents over very few games: their ELO CIs must overlap
    (correctly flagged not-significant), not assert a spurious ordering.
    """
    factories = {
        "RandomA": random_agent_factory(),
        "RandomB": random_agent_factory(),
    }
    rr = round_robin_elo(
        factories, num_games_per_pair=4, num_players=2, seed=1,
        rating="elo", bootstrap=200,
    )
    assert intervals_overlap(
        rr.ratings["RandomA"].rating_ci, rr.ratings["RandomB"].rating_ci
    )


# ---- TrueSkill path ---------------------------------------------------


def test_trueskill_finite_and_sigma_shrinks() -> None:
    """TrueSkill mu/sigma are finite; sigma shrinks with more games; mu orders
    match ELO on the same synthetic batch."""
    import math

    few = _strong_beats_weak_outcomes(8)
    many = _strong_beats_weak_outcomes(80)

    ts_few = compute_trueskill(few)
    ts_many = compute_trueskill(many)

    for ts in (ts_few, ts_many):
        for r in ts.values():
            assert math.isfinite(r.mu) and math.isfinite(r.sigma)
            assert r.sigma > 0.0

    # More games -> smaller uncertainty.
    assert ts_many["Strong"].sigma < ts_few["Strong"].sigma

    # mu ordering matches ELO ordering.
    elo = compute_elo(many, bootstrap=0)
    assert ts_many["Strong"].mu > ts_many["Weak"].mu
    assert elo["Strong"].rating > elo["Weak"].rating


def test_round_robin_trueskill_populated() -> None:
    """rating='trueskill' populates RoundRobinStats.trueskill with finite mu."""
    import math

    factories = {
        "Heuristic": heuristic_agent_factory(),
        "Random": random_agent_factory(),
    }
    rr = round_robin_elo(
        factories, num_games_per_pair=40, num_players=2, seed=0,
        rating="trueskill", bootstrap=0,
    )
    assert rr.trueskill is not None
    for r in rr.trueskill.values():
        assert math.isfinite(r.mu) and r.sigma > 0.0
    assert rr.trueskill["Heuristic"].mu > rr.trueskill["Random"].mu
