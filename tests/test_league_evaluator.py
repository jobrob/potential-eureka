"""Sprint 8C: round-robin league evaluator tests (§9).

Uses cheap scripted agents (no SB3 load) so the suite stays fast. Covers seat-
order cancellation, deterministic ratings on a fixed seed, the
stronger-outranks-weaker sanity, and the pairwise-mode delegation. Plus the 8D
forward-compat seams (`outcomes` field, `matchup_sampling='sampled'`
NotImplementedError).
"""

from __future__ import annotations

import pytest

from heat.ml.evaluate import LeagueLadder, evaluate_league
from heat.simulation.runner import (
    heuristic_agent_factory,
    random_agent_factory,
    strong_heuristic_agent_factory,
)
from heat.tracks.generator import generate_track


def _tracks(n: int = 1):
    return [generate_track(90_100_000 + i, name=f"lt-{i}") for i in range(n)]


def test_seat_order_cancelled():
    """4 identical agents get ~equal ratings -- no systematic seat edge (§7.4)."""
    contenders = {
        "h0": heuristic_agent_factory(),
        "h1": heuristic_agent_factory(),
        "h2": heuristic_agent_factory(),
        "h3": heuristic_agent_factory(),
    }
    ladder = evaluate_league(
        contenders, tracks=_tracks(2), num_players=4,
        games_per_matchup=24, rating="trueskill", seed=0,
    )
    # Identical agents differ only by RNG/tie-break noise: with seat rotation no
    # contender should get a systematic skill edge, so the TrueSkill means stay
    # clustered well within a couple of beta (the ~8.3 default skill scale).
    mus = [r.mu for r in ladder.trueskill.values()]
    spread = max(mus) - min(mus)
    assert spread < 8.0, f"seat-order not cancelled: mu spread={spread}"


def test_deterministic_ratings_fixed_seed():
    """Same (contenders, tracks, seed) twice -> identical ratings + trueskill."""
    contenders = {
        "strong": strong_heuristic_agent_factory(strength=2),
        "heur": heuristic_agent_factory(),
        "rand": random_agent_factory(),
        "rand2": random_agent_factory(),
    }
    kw = dict(tracks=_tracks(1), num_players=4, games_per_matchup=8,
              rating="trueskill", seed=7)
    a = evaluate_league(contenders, **kw)
    b = evaluate_league(contenders, **kw)

    assert {k: v.rating for k, v in a.ratings.items()} == {
        k: v.rating for k, v in b.ratings.items()
    }
    assert {k: (v.mu, v.sigma) for k, v in a.trueskill.items()} == {
        k: (v.mu, v.sigma) for k, v in b.trueskill.items()
    }


def test_stronger_agent_outranks_weaker():
    """StrongHeuristic outranks Random in both ELO and TrueSkill (rank wiring)."""
    contenders = {
        "strong": strong_heuristic_agent_factory(strength=3),
        "heur": heuristic_agent_factory(),
        "rand": random_agent_factory(),
        "rand2": random_agent_factory(),
    }
    ladder = evaluate_league(
        contenders, tracks=_tracks(1), num_players=4,
        games_per_matchup=12, rating="trueskill", seed=1,
    )
    assert ladder.ratings["strong"].rating > ladder.ratings["rand"].rating
    assert (
        ladder.trueskill["strong"].conservative
        > ladder.trueskill["rand"].conservative
    )


def test_outcomes_seam_populated():
    """Free-for-all populates LeagueLadder.outcomes (8D recompute seam, §7.5)."""
    contenders = {
        "a": heuristic_agent_factory(),
        "b": heuristic_agent_factory(),
        "c": random_agent_factory(),
        "d": random_agent_factory(),
    }
    ladder = evaluate_league(
        contenders, tracks=_tracks(1), num_players=4,
        games_per_matchup=4, seed=0,
    )
    assert ladder.outcomes is not None and len(ladder.outcomes) > 0
    # Every outcome's seat labels are pool keys (relabelled), not class names.
    labels = {po.agent_type for o in ladder.outcomes for po in o.players}
    assert labels <= {"a", "b", "c", "d"}


def test_pairwise_mode_delegates():
    """mode='pairwise' produces a pairwise table consistent with round_robin."""
    contenders = {
        "strong": strong_heuristic_agent_factory(strength=2),
        "rand": random_agent_factory(),
    }
    ladder = evaluate_league(
        contenders, tracks=_tracks(1), num_players=2,
        games_per_matchup=10, mode="pairwise", rating="elo", seed=0,
    )
    assert isinstance(ladder, LeagueLadder)
    assert ladder.pairwise is not None
    assert ("rand", "strong") in ladder.pairwise or (
        "strong", "rand"
    ) in ladder.pairwise


def test_sampled_matchup_not_implemented():
    """The 8D 'sampled' seam exists but raises NotImplementedError in 8C (§7.5)."""
    contenders = {
        "a": heuristic_agent_factory(),
        "b": random_agent_factory(),
        "c": random_agent_factory(),
        "d": random_agent_factory(),
    }
    with pytest.raises(NotImplementedError):
        evaluate_league(
            contenders, tracks=_tracks(1), num_players=4,
            matchup_sampling="sampled", seed=0,
        )
