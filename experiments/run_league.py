"""Sprint 8D: the large round-robin league + attribution (thin, heavy shell).

Reads the campaign ``manifest.jsonl`` (written by ``experiments/run_campaign.py``),
builds the contender map (every swept checkpoint + the weak/strong heuristic
references + optional baseline anchors), pins the held-out track set (§8), and
runs :func:`heat.ml.evaluate.evaluate_league` in ``matchup_sampling="sampled"``
mode over a balanced field list built by :func:`heat.ml.sweep.balanced_fields`.

Every per-game :class:`~heat.simulation.runner.GameOutcome` is persisted to
``league_games.jsonl`` so ratings recompute without replaying races (§7.4); the
attribution emitters then write ``attribution.md`` / ``ladder.csv`` /
``effects.json`` (§9.4).

**Heavy compute** (~0.5-1.5 GPU-hours, §6) -- NOT run by the test suite. Launch:

    PYTHONPATH=src python experiments/run_league.py --out-dir campaign_out
"""

from __future__ import annotations

import argparse
import os

from heat.ml.evaluate import (
    evaluate_league,
    ml_agent_factory,
    strong_heuristic_agent_factory,
)
from heat.ml.sweep import (
    ManifestReader,
    balanced_fields,
    build_attribution,
    write_attribution_artifacts,
    write_outcomes_jsonl,
)
from heat.simulation.runner import heuristic_agent_factory
from heat.tracks.generator import generate_track

#: The pinned held-out league track seeds (§8) -- distinct from the training
#: distribution and from the gate's _HOLDOUT_TRACK_SEEDS, so the league measures
#: cross-track generalization on a frozen, reproducible set. Recorded here so any
#: rerun (or a future campaign) ranks on the identical tracks.
_LEAGUE_TRACK_SEEDS: tuple[int, ...] = (
    95_000_001,
    95_000_002,
    95_000_003,
    95_000_004,
)

#: The factors the attribution analyses (the default campaign's axes, §4.1).
_CAMPAIGN_FACTORS: tuple[str, ...] = (
    "phases.0.steps",
    "phases.steps_ratio",
    "schedule.pool_kinds",
    "ppo.shaping_weight",
    "ppo.net_profile",
    "seed",
)


def league_tracks():
    """The pinned held-out track set (§8)."""
    return [
        generate_track(s, name=f"league-{s}") for s in _LEAGUE_TRACK_SEEDS
    ]


def build_contenders(reader: ManifestReader, *, anchors: dict | None = None):
    """``label -> picklable factory`` for the league (§7.5).

    Swept checkpoints are the headline; the weak/strong heuristics are anchors
    that keep ratings comparable across campaigns. ``anchors`` adds extra named
    baseline checkpoints (e.g. a Sprint-B baseline) when available.
    """
    contenders = {}
    for row in reader.done_rows():
        if row.checkpoint:
            contenders[row.run_id] = ml_agent_factory(row.checkpoint)
    contenders["weak_heuristic"] = heuristic_agent_factory()
    contenders["strong_heuristic"] = strong_heuristic_agent_factory(strength=2)
    contenders["strong_heuristic3"] = strong_heuristic_agent_factory(strength=3)
    for label, path in (anchors or {}).items():
        contenders[label] = ml_agent_factory(path)
    return contenders


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 8D league + attribution")
    parser.add_argument("--out-dir", default="campaign_out")
    parser.add_argument("--fields-per-contender", type=int, default=30)
    parser.add_argument("--games-per-field", type=int, default=2)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rating", default="trueskill", choices=["trueskill", "elo"]
    )
    args = parser.parse_args()

    reader = ManifestReader(os.path.join(args.out_dir, "manifest.jsonl"))
    contenders = build_contenders(reader)
    print(f"League: {len(contenders)} contenders", flush=True)

    fields = balanced_fields(
        list(contenders),
        num_players=args.num_players,
        fields_per_contender=args.fields_per_contender,
        seed=args.seed,
    )
    print(f"Balanced fields: {len(fields)} (<< C(n,k))", flush=True)

    ladder = evaluate_league(
        contenders,
        tracks=league_tracks(),
        num_players=args.num_players,
        matchup_sampling="sampled",
        sampled_fields=fields,
        games_per_field=args.games_per_field,
        rating=args.rating,
        seed=args.seed,
    )

    # Persist every per-game outcome (the recompute substrate, §7.4).
    games_path = os.path.join(args.out_dir, "league_games.jsonl")
    write_outcomes_jsonl(ladder.outcomes or [], games_path)
    print(f"Persisted {len(ladder.outcomes or [])} games -> {games_path}", flush=True)

    # Attribution over the manifest x ladder (§9).
    ratings = ladder.trueskill if args.rating == "trueskill" else ladder.ratings
    attribution = build_attribution(
        reader.rows(),
        ratings,
        factors=_CAMPAIGN_FACTORS,
        rating_attr="mu" if args.rating == "trueskill" else "rating",
    )
    paths = write_attribution_artifacts(
        attribution,
        args.out_dir,
        rating_label="TrueSkill mu" if args.rating == "trueskill" else "ELO",
    )
    print(f"Attribution artifacts: {paths}", flush=True)


if __name__ == "__main__":
    main()
