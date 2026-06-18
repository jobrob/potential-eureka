"""Sprint 8C launch script: solo pretrain + opponent curriculum + league eval.

Collapses the two-script prototype handoff (``experiments/proto_solo.py`` ->
``experiments/proto_finetune.py``) into ONE gated, league-scored run:

  solo pretrain (gamma 0.99) -> fine-tune vs weak -> mixed -> strong (gamma 0.999)

driven by :func:`heat.ml.training.train_self_play` with an explicit
:class:`~heat.ml.training.TrainingPhase` list. Sprint A's Wilson-LB gate +
best-checkpoint preservation flow across all opponent phases (the solo phase is
ungated against the race yardstick -- it is out-of-distribution there). The best
checkpoint is then scored against the reference heuristics with
:func:`heat.ml.evaluate.evaluate_league` and the ladder is printed.

Run:  PYTHONPATH=src python train_8c.py
"""

from __future__ import annotations

import time

from heat.ml.model import PPOConfig
from heat.ml.training import (
    default_8c_phases,
    sprint_8c_curriculum,
    train_self_play,
)
from heat.ml.evaluate import (
    evaluate_league,
    heuristic_agent_factory,
    ml_agent_factory,
    strong_heuristic_agent_factory,
)
from heat.tracks.generator import generate_track

SEED = 0
NUM_PLAYERS = 4
#: Fixed held-out tracks for the post-train league ladder (distinct band from the
#: training/gate seeds so the ladder is a generalization signal).
HELDOUT_SEEDS = (80_000_001, 80_000_002, 80_000_003)


def main() -> None:
    # The proto hyperparams (proto_solo.py): 8 envs, 1024-step rollouts. gamma is
    # set per-phase by the phase list, so the PPOConfig.gamma here is only the
    # build-time default for the first phase (overridden by phase 0's gamma).
    config = PPOConfig(
        verbose=1,
        seed=SEED,
        n_envs=8,
        n_steps=1024,
        batch_size=256,
        shaping_progress_coef=1.0,
    )
    curriculum = sprint_8c_curriculum()
    phases = default_8c_phases(NUM_PLAYERS)

    print(f"=== SPRINT 8C TRAIN START {time.strftime('%H:%M:%S')} ===", flush=True)
    for p in phases:
        print(
            f"  phase {p.name:6s}: players={p.num_players} reward={p.reward_mode} "
            f"gamma={p.gamma} shaping={p.shaping_weight} steps={p.steps:,} "
            f"pool={p.pool_kind}",
            flush=True,
        )
    t0 = time.time()
    model, best_path = train_self_play(
        config,
        curriculum,
        num_players=NUM_PLAYERS,
        phases=phases,
    )
    print(
        f"=== TRAIN DONE in {(time.time() - t0) / 60:.1f} min; "
        f"best checkpoint -> {best_path} ===",
        flush=True,
    )

    # League ladder: the trained policy vs the reference heuristics, free-for-all
    # on the held-out track set, seat-order-cancelled.
    tracks = [generate_track(s, name=f"heldout-{s}") for s in HELDOUT_SEEDS]
    contenders = {
        "8c": ml_agent_factory(best_path),
        "strong2": strong_heuristic_agent_factory(strength=2),
        "strong3": strong_heuristic_agent_factory(strength=3),
        "heuristic": heuristic_agent_factory(),
    }
    print("\n=== LEAGUE LADDER (free-for-all, seat-cancelled) ===", flush=True)
    ladder = evaluate_league(
        contenders,
        tracks=tracks,
        num_players=NUM_PLAYERS,
        games_per_matchup=40,
        rating="trueskill",
        seed=SEED + 555,
    )
    if ladder.trueskill is not None:
        ranked = sorted(
            ladder.trueskill.values(), key=lambda r: r.conservative, reverse=True
        )
        for r in ranked:
            print(
                f"  {r.agent_type:10s}  mu={r.mu:6.2f}  sigma={r.sigma:5.2f}  "
                f"cons={r.conservative:6.2f}  (n={r.games})",
                flush=True,
            )
    print("\n  ELO:", flush=True)
    for label, er in sorted(
        ladder.ratings.items(), key=lambda kv: kv[1].rating, reverse=True
    ):
        print(f"  {label:10s}  elo={er.rating:7.1f}  (n={er.games})", flush=True)


if __name__ == "__main__":
    main()
