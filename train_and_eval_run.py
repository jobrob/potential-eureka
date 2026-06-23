"""Sprint A launch script: trustworthy & measurable self-play run + eval.

Not committed; the manual launch config for the "trustworthy & measurable run"
(docs/sprint-A-trustworthy-run.md). Wires the Sprint-A preset so the held-out
gate is honest before any long run / Sprint B:

* Phase-1-heavy budget (Idea 4: ``phase1_steps == total_timesteps``) -- the
  periodic Phase-1 gate (Idea 7) preserves the BEST held-out checkpoint.
* The gate scores against the opponent we train against (Idea 8: strong
  heuristic) on the Wilson lower bound of pooled held-out games (Idea 9), with
  ``gate_games`` raised so the bound is usable.
* Randomized learner seats in training (Idea 10), measured back via a
  seat-swept ``evaluate_ml``.
* Non-zero reward shaping (Idea 3) on the PPOConfig; ``normalize_obs`` stays OFF
  (Idea 15, recorded constraint).

After training, the BEST checkpoint is evaluated vs BOTH the weak and the strong
heuristic, and seat-swept to confirm no seat-0 overfit.
"""

from __future__ import annotations

import time

from heat.ml.model import PPOConfig
from heat.ml.training import sprint_a_curriculum, train_self_play
from heat.ml.evaluate import (
    evaluate_ml,
    strong_heuristic_agent_factory,
)
from heat.simulation.runner import heuristic_agent_factory

NUM_PLAYERS = 4
EVAL_GAMES = 300
SEED = 0
TOTAL_TIMESTEPS = 2_000_000


def _win_rate(per_agent) -> float:
    ml = per_agent.get("MLAgent")
    return float(ml.win_rate) if ml is not None else 0.0


def main() -> None:
    # Idea 3: a small non-zero shaping weight gives a thicker early gradient on
    # the sparse generated-track problem; the bounded anti-spinout term stays off
    # by default (raise shaping_spinout_weight to enable it).
    cfg = PPOConfig(verbose=1, seed=SEED, shaping_weight=0.05)
    curr = sprint_a_curriculum(TOTAL_TIMESTEPS, run_name="heat_ppo_sprintA")

    print(f"=== TRAIN START {time.strftime('%H:%M:%S')} ===", flush=True)
    t0 = time.time()
    # train on procedurally generated tracks by default (track=None); the gate
    # uses the fixed held-out generated set.
    model, best_path = train_self_play(
        config=cfg, curriculum=curr, num_players=NUM_PLAYERS
    )
    train_dt = time.time() - t0
    print(
        f"=== TRAIN DONE in {train_dt/60:.1f} min -> {best_path} (BEST) ===",
        flush=True,
    )

    print(f"=== EVAL START {time.strftime('%H:%M:%S')} ===", flush=True)
    t1 = time.time()

    # (b) report BOTH weak- and strong-heuristic win-rates of the BEST checkpoint.
    weak = evaluate_ml(
        best_path,
        opponent_factory=heuristic_agent_factory(),
        num_games=EVAL_GAMES,
        num_players=NUM_PLAYERS,
        seed=SEED,
        parallel=False,
    )
    strong = evaluate_ml(
        best_path,
        opponent_factory=strong_heuristic_agent_factory(),
        num_games=EVAL_GAMES,
        num_players=NUM_PLAYERS,
        seed=SEED,
        parallel=False,
    )
    eval_dt = time.time() - t1
    print(f"=== EVAL DONE in {eval_dt/60:.1f} min ===", flush=True)

    print("\n=== RESULTS (BEST checkpoint win-rate) ===", flush=True)
    print(f"  vs weak   HeuristicAgent:       {_win_rate(weak)*100:5.1f}%", flush=True)
    print(f"  vs strong StrongHeuristicAgent: {_win_rate(strong)*100:5.1f}%", flush=True)

    # (d) seat sweep: place the MLAgent at every seat vs the strong heuristic and
    # confirm the win-rate spread is small (no seat-0 overfit).
    print("\n=== SEAT SWEEP (vs strong heuristic) ===", flush=True)
    seat_rates: list[float] = []
    for seat in range(NUM_PLAYERS):
        per_agent = evaluate_ml(
            best_path,
            opponent_factory=strong_heuristic_agent_factory(),
            num_games=EVAL_GAMES // NUM_PLAYERS,
            num_players=NUM_PLAYERS,
            seed=SEED,
            parallel=False,
            learner_seat=seat,
        )
        wr = _win_rate(per_agent)
        seat_rates.append(wr)
        print(f"  seat {seat}: {wr*100:5.1f}%", flush=True)
    spread = max(seat_rates) - min(seat_rates) if seat_rates else 0.0
    print(f"  seat spread: {spread*100:5.1f}% (smaller == more seat-robust)", flush=True)


if __name__ == "__main__":
    main()
