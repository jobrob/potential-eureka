"""Prototype stage 2-3: warm-start fine-tune the solo-pretrained policy.

Pipeline under test (the OPPONENT-axis curriculum that should replace Sprint A's
train-vs-strong and Sprint B's track curriculum):

  solo pretrain (done, proto_solo.py) -> fine-tune vs WEAK -> ramp to STRONG

Hypothesis: a competent solo base makes vs-strong fine-tuning SUCCEED where
from-scratch collapsed to ~4%. Reference points:
  * from-scratch vs strong         ~4% vs weak  (collapse)
  * direct from-scratch vs weak     ~40% vs weak (gen_seat, 400k)
  * solo cold-transfer, no fine-tune 27.5% vs weak
"""

from __future__ import annotations

import time

from sb3_contrib import MaskablePPO

from heat.ml.vec import make_vec_env
from heat.ml.training import _scripted_opponents, save_checkpoint
from heat.ml.model import PPOConfig
from heat.ml.evaluate import evaluate_ml, strong_heuristic_agent_factory
from heat.simulation.runner import heuristic_agent_factory

SEED = 0
STAGE_STEPS = 400_000
NUM_PLAYERS = 4
EVAL_GAMES = 160
SOLO_CKPT = "checkpoints/heat_ppo_solo_proto"


def _wr(pa) -> float:
    ml = pa.get("MLAgent")
    return float(ml.win_rate) if ml is not None else 0.0


def _env(use_strong: bool):
    opponents = _scripted_opponents(NUM_PLAYERS, use_strong=use_strong,
                                    broaden_mix=use_strong)
    return make_vec_env(track=None, num_players=NUM_PLAYERS, opponents=opponents,
                        learner_id=0, n_envs=8, vec_cls="subproc", seed=SEED,
                        shaping_weight=0.05, randomize_seat=True)


def _report(tag: str, path: str) -> None:
    weak = evaluate_ml(path, opponent_factory=heuristic_agent_factory(),
                       num_games=EVAL_GAMES, num_players=NUM_PLAYERS,
                       seed=SEED + 555, parallel=False)
    strong = evaluate_ml(path, opponent_factory=strong_heuristic_agent_factory(),
                         num_games=EVAL_GAMES, num_players=NUM_PLAYERS,
                         seed=SEED + 555, parallel=False)
    print(f"  [{tag}] vs weak {_wr(weak)*100:5.1f}%   vs strong {_wr(strong)*100:5.1f}%",
          flush=True)


def main() -> None:
    cfg = PPOConfig(seed=SEED, n_envs=8)  # for sidecar metadata only

    # ---- Stage 1: warm-start from solo, fine-tune vs WEAK -------------------
    venv_weak = _env(use_strong=False)
    print(f"=== load solo checkpoint -> fine-tune vs WEAK {time.strftime('%H:%M:%S')} ===",
          flush=True)
    model = MaskablePPO.load(SOLO_CKPT, env=venv_weak)
    t0 = time.time()
    model.learn(total_timesteps=STAGE_STEPS, progress_bar=False,
                reset_num_timesteps=True)
    print(f"=== stage1 (vs weak) done in {(time.time()-t0)/60:.1f} min ===", flush=True)
    p1 = "checkpoints/heat_ppo_ft_weak"
    save_checkpoint(model, p1, track_name="generated", num_players=NUM_PLAYERS,
                    seed=SEED, ppo_config=cfg, vec_env=venv_weak)
    _report("after vs-weak fine-tune", p1)

    # ---- Stage 2: ramp to STRONG -------------------------------------------
    venv_strong = _env(use_strong=True)
    model.set_env(venv_strong)
    print(f"=== ramp to STRONG {time.strftime('%H:%M:%S')} ===", flush=True)
    t1 = time.time()
    model.learn(total_timesteps=STAGE_STEPS, progress_bar=False,
                reset_num_timesteps=True)
    print(f"=== stage2 (vs strong) done in {(time.time()-t1)/60:.1f} min ===", flush=True)
    p2 = "checkpoints/heat_ppo_ft_strong"
    save_checkpoint(model, p2, track_name="generated", num_players=NUM_PLAYERS,
                    seed=SEED, ppo_config=cfg, vec_env=venv_strong)
    _report("after vs-strong fine-tune", p2)

    print("\n=== REFERENCE ===", flush=True)
    print("  from-scratch vs strong   ~ 4% weak / 3% strong (collapse)", flush=True)
    print("  direct from-scratch weak ~40% weak", flush=True)
    print("  solo cold transfer        27.5% weak", flush=True)


if __name__ == "__main__":
    main()
