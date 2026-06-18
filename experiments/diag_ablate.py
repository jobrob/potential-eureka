"""Ablation: isolate which Sprint-A/B feature tanks the generated-track run.

Trains short runs on GENERATED tracks (track=None) with weak opponents and
toggles one feature at a time, then evaluates the BEST checkpoint vs the weak
heuristic on held-out generated tracks. Run name selects the variant.
"""

from __future__ import annotations

import sys
import time

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


VARIANTS = {
    # name: dict of CurriculumConfig overrides (on top of the simple base)
    "gen_base":   dict(),                                  # generated, weak, no extras
    "gen_seat":   dict(randomize_seat=True),               # + seat randomization
    "gen_curr":   dict(use_track_curriculum=True, curriculum_horizon_steps=TOTAL),
    # Retuned: reach full difficulty at 40% of training, full for the last 60%.
    "gen_curr40": dict(use_track_curriculum=True,
                       curriculum_horizon_steps=int(0.4 * TOTAL)),
    # seat + STRONG opponents (matches the full-run opponent setup) -- isolates
    # whether strong opponents are what collapses from-scratch training.
    "gen_seat_strong": dict(randomize_seat=True,
                            use_strong_heuristic_opponents=True,
                            broaden_phase1_mix=True),
}


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "gen_base"
    over = VARIANTS[variant]
    cfg = PPOConfig(verbose=0, seed=SEED, n_envs=8, n_steps=1024, batch_size=256)
    base = dict(
        total_timesteps=TOTAL,
        phase1_steps=TOTAL,
        phase1_eval_every=100_000,
        gate_games=40,
        run_name=f"heat_ppo_{variant}",
        use_strong_heuristic_opponents=False,
        randomize_seat=False,
        broaden_phase1_mix=False,
        use_track_curriculum=False,
    )
    base.update(over)
    curr = CurriculumConfig(**base)
    print(f"=== {variant} START {time.strftime('%H:%M:%S')} overrides={over} ===", flush=True)
    t0 = time.time()
    model, best_path = train_self_play(
        config=cfg, curriculum=curr, num_players=NUM_PLAYERS, track=None
    )
    # Held-out generated eval: evaluate_ml with track=None samples generated tracks.
    pa = evaluate_ml(best_path, opponent_factory=heuristic_agent_factory(),
                     num_games=120, num_players=NUM_PLAYERS, seed=SEED + 777,
                     parallel=False, track=None)
    print(f"=== {variant} DONE in {(time.time()-t0)/60:.1f} min: "
          f"BEST vs weak (held-out generated) = {_wr(pa)*100:5.1f}% ===", flush=True)


if __name__ == "__main__":
    main()
