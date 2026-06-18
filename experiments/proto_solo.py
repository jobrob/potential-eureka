"""Prototype: solo time-trial pretraining.

Hypothesis: the from-scratch collapse is a learning-SIGNAL problem, not an
opponent-strength problem. A solo race with a dense progress reward (+ a finish
bonus, discounted by gamma so finishing SOONER is worth more) gives a clean,
stationary signal that teaches the core single-agent skills (gears, heat, corner
speed, not spinning out). We then check whether that solo skill TRANSFERS: drop
the solo-trained policy into a 4-player race vs the weak heuristic.

Reward design (no negative rewards -> no crash-to-end-episode exploit):
  reward_t = SHAPING progress this step           (dense; from the env)
           + FINISH_BONUS if the lap(s) completed  (terminal)
Speed is incentivised purely by gamma<1: the same progress/bonus is worth more
the earlier it arrives, so the policy minimises steps-to-finish.

Metrics:
  1. solo steps-to-finish, fresh vs trained (lower = faster = it learned to drive)
  2. transfer: trained-solo policy win-rate in a 4p race vs the weak heuristic
"""

from __future__ import annotations

import time

import numpy as np
import gymnasium as gym

from heat.ml.vec import make_vec_env
from heat.ml.env import HeatEnv
from heat.ml.model import PPOConfig, build_model
from heat.ml.training import save_checkpoint
from heat.ml.evaluate import evaluate_ml
from heat.simulation.runner import heuristic_agent_factory
from heat.tracks import track_sampler

SEED = 0
TOTAL = 400_000
FINISH_BONUS = 5.0
GAMMA = 0.99
SHAPING_WEIGHT = 1.0


class SoloFinishBonus(gym.Wrapper):
    """Add a terminal finish bonus on top of the env's dense progress reward.

    The env already returns progress shaping (placement reward is 0 for a solo
    race). We add ``FINISH_BONUS`` when the episode TERMINATES (laps completed),
    not when it truncates (ran out of steps) -- so the only way to collect the
    bonus is to actually finish, and gamma<1 makes finishing fast worth more.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not truncated:
            reward += FINISH_BONUS
        return obs, reward, terminated, truncated, info


def _solo_env_fns(n_envs: int):
    # Build solo (num_players=1) envs with dense progress shaping ON, wrapped to
    # add the finish bonus. Reuse make_vec_env's shaping plumbing via one env at
    # a time is awkward; build raw HeatEnvs here for clarity.
    sampler = track_sampler(base_seed=SEED)

    def make(rank: int):
        def _fn():
            import heat.ml.spaces as spaces
            spaces.SHAPING_WEIGHT = SHAPING_WEIGHT      # dense progress reward ON
            spaces.SHAPING_PROGRESS_COEF = 1.0
            env = HeatEnv(track=sampler, num_players=1, opponents=None,
                          learner_id=0)
            return SoloFinishBonus(env)
        return _fn

    return [make(i) for i in range(n_envs)]


def _solo_steps_to_finish(model, n_tracks: int = 40) -> tuple[float, float]:
    """Mean steps-to-finish over held-out solo tracks (model=None -> random)."""
    import heat.ml.spaces as spaces
    spaces.SHAPING_WEIGHT = SHAPING_WEIGHT
    sampler = track_sampler(base_seed=99_000)  # held-out band
    steps_list, finished = [], 0
    for k in range(n_tracks):
        env = HeatEnv(track=sampler, num_players=1, opponents=None, learner_id=0)
        obs, info = env.reset(seed=99_000 + k)
        steps = 0
        for _ in range(5000):
            mask = env.action_masks()
            if model is None:
                a = int(np.random.choice(np.flatnonzero(mask)))
            else:
                a, _ = model.predict(obs, action_masks=mask, deterministic=True)
                a = int(a)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                if term:
                    finished += 1
                    steps_list.append(steps)
                break
    mean_steps = float(np.mean(steps_list)) if steps_list else float("nan")
    return mean_steps, finished / n_tracks


def main() -> None:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    cfg = PPOConfig(verbose=1, seed=SEED, n_envs=8, n_steps=1024, batch_size=256,
                    gamma=GAMMA)

    print("=== fresh (untrained) solo speed ===", flush=True)
    # Build a throwaway model for the random baseline measurement.
    venv0 = SubprocVecEnv(_solo_env_fns(8))
    fresh = build_model(venv0, cfg)
    fresh_steps, fresh_fin = _solo_steps_to_finish(None)
    print(f"  RANDOM policy: mean steps-to-finish={fresh_steps:.1f}  finish-rate={fresh_fin*100:.0f}%",
          flush=True)
    fresh_model_steps, fresh_model_fin = _solo_steps_to_finish(fresh)
    print(f"  FRESH net    : mean steps-to-finish={fresh_model_steps:.1f}  finish-rate={fresh_model_fin*100:.0f}%",
          flush=True)

    print(f"\n=== SOLO TRAIN START {time.strftime('%H:%M:%S')} ({TOTAL} steps) ===", flush=True)
    t0 = time.time()
    venv = SubprocVecEnv(_solo_env_fns(8))
    model = build_model(venv, cfg)
    model.learn(total_timesteps=TOTAL, progress_bar=False)
    print(f"=== SOLO TRAIN DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)

    trained_steps, trained_fin = _solo_steps_to_finish(model)
    print(f"\n  TRAINED solo : mean steps-to-finish={trained_steps:.1f}  finish-rate={trained_fin*100:.0f}%",
          flush=True)
    print(f"  (lower steps than {fresh_steps:.1f} random => it learned to drive FAST)",
          flush=True)

    # Transfer test: save and drop into a 4-player race vs the weak heuristic.
    path = "checkpoints/heat_ppo_solo_proto"
    save_checkpoint(model, path, track_name="generated", num_players=1,
                    seed=SEED, ppo_config=cfg, vec_env=venv)
    print("\n=== TRANSFER: solo policy in 4p race vs weak heuristic ===", flush=True)
    try:
        pa = evaluate_ml(path, opponent_factory=heuristic_agent_factory(),
                         num_games=120, num_players=4, seed=SEED + 555,
                         parallel=False)
        ml = pa.get("MLAgent")
        wr = float(ml.win_rate) if ml else 0.0
        print(f"  solo-trained vs weak heuristic (4p): {wr*100:.1f}%  "
              f"(>25% random-field baseline => solo skill transfers)", flush=True)
    except Exception as e:  # noqa: BLE001
        print("  transfer eval raised:", repr(e), flush=True)


if __name__ == "__main__":
    main()
