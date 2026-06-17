#!/usr/bin/env python
"""CLI to train a HEAT RL agent via the self-play curriculum (Sprint 6C).

Front-end over :func:`heat.ml.training.train_self_play`, wiring command-line
arguments to a :class:`heat.ml.model.PPOConfig` + :class:`~heat.ml.training.CurriculumConfig`.
Replaces the uncommitted throwaway ``train_and_eval_run.py`` with a committed,
reproducible, observable entry point.

Key 6C features exposed:

* **Vectorized envs** (``--n-envs`` / ``--vec``) to beat the Python engine
  bottleneck, with optional **CUDA** device selection (``--device``, CPU
  fallback is automatic — see :func:`heat.ml.model.resolve_device`).
* A **larger net** profile (``--net-profile``) worth the GPU once throughput is up.
* **Self-play stability:** the run writes the *best-evaluated* checkpoint to the
  canonical ``--run-name`` path and the (possibly collapsed) final model to
  ``<run-name>_final`` — so a Phase-2 collapse can never lose the good model.
* **Shaping schedule** (``--shaping-weight-start/-end``) and
  **return/reward normalization** (``--normalize-reward`` / ``--normalize-obs``).
* **TensorBoard** (``--tensorboard-log``): PPO's built-in scalars plus the
  ``eval/gate_score`` each gate run, so a collapse is visible live.

By default training runs on **procedurally generated tracks** (a fresh random
track each episode, §6A), gated on a held-out generated set, so the policy
learns to race in general rather than memorizing one layout. Pass ``--track
<name>`` to pin a single fixed track instead.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/train_ml.py \\
        --timesteps 2000000 --phase1-steps 1000000 --snapshot-every 200000 \\
        --n-envs 8 --vec subproc --device auto --net-profile large \\
        --normalize-reward --shaping-weight-start 0.02 \\
        --run-name heat_ppo --seed 0 --tensorboard-log runs/heat
        # add --track usa to pin one track instead of generated tracks
"""

from __future__ import annotations

import argparse
import sys

from heat.ml.model import PPOConfig, net_profile_config
from heat.ml.training import CurriculumConfig, train_self_play


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a HEAT RL agent (vectorized self-play, gated promotion).",
    )
    # --- curriculum / schedule ---
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps (default: 200000).")
    parser.add_argument("--phase1-steps", type=int, default=100_000,
                        help="Phase-1 (scripted-pool) timesteps (default: 100000).")
    parser.add_argument("--snapshot-every", type=int, default=50_000,
                        help="Phase-2 snapshot/eval cadence in steps (default: 50000).")
    parser.add_argument("--max-snapshots", type=int, default=3,
                        help="Max frozen snapshots retained, FIFO (default: 3).")
    parser.add_argument("--snapshot-mix", type=float, default=0.5,
                        help="Max snapshot opponent fraction (ramp target; default: 0.5).")
    parser.add_argument("--gate-games", type=int, default=20,
                        help="Games per inline eval-gate evaluation (default: 20).")

    # --- §6D opponent league + PFSP ---
    parser.add_argument("--use-league", action="store_true",
                        help="Use the 6D PFSP opponent league for Phase-2 snapshot "
                             "seats instead of the FIFO pool.")
    parser.add_argument("--league-capacity", type=int, default=8,
                        help="League pool capacity (entries retained; default: 8).")
    parser.add_argument("--pfsp-mode", choices=["even", "variance", "hard"],
                        default="even",
                        help="PFSP weighting: even/variance (close games, default) "
                             "or hard (current losses).")
    parser.add_argument("--league-eval-games", type=int, default=4,
                        help="Games per sampled opponent for PFSP win-rate "
                             "bookkeeping each chunk (default: 4).")

    # --- §6E strong-heuristic curriculum ---
    parser.add_argument("--use-strong-opponents", action="store_true",
                        help="Upgrade scripted opponents from HeuristicAgent to the "
                             "6E StrongHeuristicAgent strength bar.")

    # --- throughput / hardware ---
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Parallel envs; >1 uses SubprocVecEnv (default: 1).")
    parser.add_argument("--vec", choices=["subproc", "dummy"], default="subproc",
                        help="Vec env class when n-envs>1 (default: subproc).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Torch device; cuda falls back to CPU (default: auto).")
    parser.add_argument("--net-profile", choices=["small", "large"], default="small",
                        help="Network size profile (default: small).")

    # --- shaping schedule (§2.3) ---
    parser.add_argument("--shaping-weight-start", type=float, default=0.0,
                        help="Dense shaping weight at the start of Phase 2 (default: 0).")
    parser.add_argument("--shaping-weight-end", type=float, default=0.0,
                        help="Dense shaping weight at the end of Phase 2 (default: 0).")

    # --- normalization (§2.6) ---
    parser.add_argument("--normalize-reward", action="store_true",
                        help="Wrap the vec env in VecNormalize (norm_reward).")
    parser.add_argument("--normalize-obs", action="store_true",
                        help="Also normalize observations (VecNormalize norm_obs).")

    # --- §2.5 warm-up schedule ---
    parser.add_argument("--phase2-lr-scale", type=float, default=1.0,
                        help="LR multiplier for the Phase-2 warm-up chunk(s).")
    parser.add_argument("--phase2-ent-scale", type=float, default=1.0,
                        help="Entropy multiplier for the Phase-2 warm-up chunk(s).")
    parser.add_argument("--warmup-chunks", type=int, default=1,
                        help="Number of leading Phase-2 chunks the warm-up covers.")

    # --- run bookkeeping ---
    parser.add_argument("--players", type=int, default=4,
                        help="Seats per game, 2..6 (default: 4).")
    parser.add_argument("--track", default=None,
                        help="Pin a specific track by name (e.g. 'usa'). Default: "
                             "None = train on procedurally generated random tracks "
                             "(a fresh track per episode; gated on a held-out set).")
    parser.add_argument("--run-name", default="heat_ppo",
                        help="Base checkpoint name (default: heat_ppo).")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoints (default: checkpoints).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for reproducibility (default: 0).")
    parser.add_argument("--tensorboard-log", default=None,
                        help="TensorBoard log dir (PPO scalars + eval gate score).")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[PPOConfig, CurriculumConfig]:
    """Translate parsed CLI args into a ``(PPOConfig, CurriculumConfig)`` pair.

    Separated from :func:`main` so it is unit-testable without running training
    (see ``tests/test_train_cli.py``). The net-size profile is applied first,
    then the remaining PPO knobs are overlaid.
    """
    base = net_profile_config(args.net_profile)
    ppo = PPOConfig(
        net_arch=base.net_arch,
        features_extractor_hidden=base.features_extractor_hidden,
        features_dim=base.features_dim,
        seed=args.seed,
        device=args.device,
        n_envs=args.n_envs,
        tensorboard_log=args.tensorboard_log,
        shaping_weight=args.shaping_weight_start,
    )

    curriculum = CurriculumConfig(
        total_timesteps=args.timesteps,
        phase1_steps=args.phase1_steps,
        snapshot_every=args.snapshot_every,
        max_snapshots=args.max_snapshots,
        snapshot_mix=args.snapshot_mix,
        gate_games=args.gate_games,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        shaping_weight_start=args.shaping_weight_start,
        shaping_weight_end=args.shaping_weight_end,
        phase2_lr_scale=args.phase2_lr_scale,
        phase2_ent_scale=args.phase2_ent_scale,
        warmup_chunks=args.warmup_chunks,
        normalize_reward=args.normalize_reward,
        normalize_obs=args.normalize_obs,
        use_league=args.use_league,
        league_capacity=args.league_capacity,
        league_pfsp_mode=args.pfsp_mode,
        league_eval_games=args.league_eval_games,
        use_strong_heuristic_opponents=args.use_strong_opponents,
    )
    return ppo, curriculum


def main(argv: list[str] | None = None) -> int:
    from heat.tracks.loader import load_track_by_name

    args = _parse_args(argv)
    ppo, curriculum = build_configs(args)

    track = load_track_by_name(args.track) if args.track else None

    model, best_path = train_self_play(
        ppo,
        curriculum,
        num_players=args.players,
        track=track,
    )
    del model  # not needed past saving; the best checkpoint is on disk

    final_path = f"{best_path}_final"
    print("=" * 60)
    print("HEAT ML Training complete")
    print("=" * 60)
    print(f"Best checkpoint:  {best_path}")
    print(f"Final checkpoint: {final_path}")
    if args.tensorboard_log:
        print(f"TensorBoard logs: {args.tensorboard_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
