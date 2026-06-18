"""Decisive diagnostic: can the CURRENT code train a good policy AT ALL?

Simplest learnable config -- fixed 'usa' track, weak heuristic opponents, NO
curriculum, NO seat randomization, NO strong opponents -- closest to the
Sprint-5 setup that reportedly reached ~94% vs the weak heuristic. If this
climbs well above 0%, the v2 obs + core training path is healthy and the poor
Sprint-B baseline is the harder generated+strong+curriculum problem. If it stays
near 0%, there is a real regression in the core obs/training path.
"""

from __future__ import annotations

import dataclasses
import time

from heat.tracks import load_track_by_name
from heat.ml.model import PPOConfig
from heat.ml.training import CurriculumConfig, train_self_play
from heat.ml.evaluate import evaluate_ml
from heat.simulation.runner import heuristic_agent_factory

NUM_PLAYERS = 4
SEED = 0
TOTAL = 400_000


def _wr(pa) -> float:
    ml = pa.get("MLAgent")
    return float(ml.win_rate) if ml is not None else 0.0


def main() -> None:
    track = load_track_by_name("usa")
    cfg = PPOConfig(verbose=1, seed=SEED, n_envs=8, n_steps=1024, batch_size=256)
    curr = CurriculumConfig(
        total_timesteps=TOTAL,
        phase1_steps=TOTAL,            # Phase-1 only
        phase1_eval_every=100_000,
        gate_games=40,
        run_name="heat_ppo_canlearn",
        # all Sprint A/B features OFF -> legacy simple setup
        use_strong_heuristic_opponents=False,
        randomize_seat=False,
        broaden_phase1_mix=False,
        use_track_curriculum=False,
    )
    print(f"=== CAN-LEARN TRAIN START {time.strftime('%H:%M:%S')} (fixed usa, weak opp) ===",
          flush=True)
    t0 = time.time()
    model, best_path = train_self_play(
        config=cfg, curriculum=curr, num_players=NUM_PLAYERS, track=track
    )
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min -> {best_path} ===", flush=True)

    pa = evaluate_ml(best_path, opponent_factory=heuristic_agent_factory(),
                     num_games=120, num_players=NUM_PLAYERS, seed=SEED,
                     parallel=False, track=track)
    wr = _wr(pa)
    print(f"\nBEST vs weak HeuristicAgent on usa: {wr*100:5.1f}%", flush=True)
    print("  >>> >>~25% means it LEARNED (core path healthy);"
          " ~0% means a real core regression.", flush=True)


if __name__ == "__main__":
    main()
