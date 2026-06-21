"""Sprint S4 (BC -> PPO fine-tune) -- warm-start RL from the cloned/DAgger net.

The AlphaStar / OpenAI-Five recipe: take the behavior-cloned (or DAgger-improved)
policy as the *initialization* for self-play RL, then run the existing opponent
curriculum (solo -> weak -> mixed -> strong). BC/DAgger give the policy clean
limit-1 corner behavior in-distribution; PPO then pushes win-rate and -- crucially
-- learns the **value head** from reward (the BC checkpoint's critic is
uninitialized: BC supervised only the actor), so the policy can be improved by
advantage rather than only imitated.

This is a thin, runnable wrapper over the already-merged machinery:

* ``train_self_play(..., warm_start_path=<bc_or_dagger.zip>, phases=...)`` loads
  the BC/DAgger checkpoint's weights instead of ``build_model`` (the 8C
  ``warm_start_path`` hook), then drives the ``TrainingPhase`` ramp with the
  Wilson-LB gate + best-checkpoint preservation (so a collapsing strong phase can
  never overwrite a better earlier checkpoint -- the 8C bug-2 guard the S4 risk
  note calls for).
* The opponent ramp comes from :func:`default_8c_phases`; ``--smoke`` shrinks the
  per-phase step budgets so the *pipeline* can be proven end-to-end in seconds
  without a multi-hour campaign.

Anti-forgetting lever (KL-to-BC)
--------------------------------
The S4 risk note prescribes a KL-to-BC regularizer on the **actor** (not a frozen
critic, which the BC checkpoint doesn't have) to resist catastrophic forgetting of
corner discipline during the strong phase. ``--kl-to-bc-coef C`` enables it via
:class:`heat.ml.kl_regularizer.MaskablePPOWithKLToBC`, which penalizes the policy
for drifting from the (frozen) BC reference distribution each update. It defaults
to 0.0 (off) so the smoke path uses the byte-for-byte proven warm-start code; turn
it on for the real strong-phase fine-tune.

Honesty
-------
This lands the *code path* and proves it runs end-to-end. The headline
"fine-tuned net >= search agent, > weak win-rate" result needs a real multi-hour
GPU campaign (RTX 4080, torch cu126); the smoke run here is NOT that. After a real
run, gate the produced ``best`` checkpoint with ``experiments/eval_bc.py`` (solo
limit-1 spins/pass + rounds, held-out generated) and ``experiments/eval_dagger.py``
/ the league, never finish-rate alone.

Usage:
    # smoke: prove the warm-start pipeline runs end-to-end (seconds, CPU)
    python experiments/finetune_ppo.py --warm-start checkpoints/bc.zip \\
        --smoke --run-name heat_s4_smoke

    # real fine-tune (GPU): full ramp + KL-to-BC during the strong phase
    python experiments/finetune_ppo.py --warm-start checkpoints/dagger_iter3.zip \\
        --device cuda --net large --kl-to-bc-coef 0.5 --run-name heat_s4
"""

from __future__ import annotations

import argparse
import dataclasses

from heat.ml.model import PPOConfig, net_profile_config
from heat.ml.training import (
    TrainingPhase,
    default_8c_phases,
    sprint_8c_curriculum,
    train_self_play,
)


def _smoke_phases(num_players: int) -> list[TrainingPhase]:
    """A tiny solo -> weak -> strong ramp to prove the pipeline end-to-end.

    Mirrors the structure of :func:`default_8c_phases` (a solo phase then opponent
    rungs, with the solo->race gamma handoff) but with step budgets shrunk to the
    smallest values that still exercise every code path (>= one learn chunk per
    phase). Not a training recipe -- a pipeline smoke test.
    """
    return [
        TrainingPhase("solo", 1, "solo", 0.99, 1.0, 512, None),
        TrainingPhase("weak", num_players, "race", 0.999, 0.05, 512, "weak"),
        TrainingPhase("strong", num_players, "race", 0.999, 0.05, 512, "strong"),
    ]


def finetune(args: argparse.Namespace) -> tuple[str, str]:
    """Warm-start PPO from a BC/DAgger checkpoint and run the opponent ramp.

    Returns ``(best_checkpoint_path, final_checkpoint_path)`` -- the best-evaluated
    checkpoint (the one to gate/ship) and the possibly-collapsed final weights.
    """
    # Base PPO config; size the net to match the warm-start checkpoint's net (a
    # warm start MUST use the same architecture the checkpoint was saved with --
    # MaskablePPO.load reconstructs the saved policy, but the env/config still
    # drive the surrounding loop and any KL-reference build).
    cfg = PPOConfig(
        device=args.device,
        seed=args.seed,
        n_envs=args.n_envs,
        ent_coef=args.ent_coef,
        # verbose=1 so stable-baselines3 prints periodic rollout/train tables --
        # without it a ~40 min fine-tune logs NOTHING and looks dead even while
        # training. The run must show signs of life in its log.
        verbose=args.verbose,
    )
    if args.net != "default":
        cfg = net_profile_config(args.net, cfg)

    if args.smoke:
        # Keep the smoke run single-env + tiny n_steps so a phase's one learn
        # chunk completes near-instantly.
        cfg = dataclasses.replace(cfg, n_envs=1, n_steps=256, batch_size=64)
        phases = _smoke_phases(args.players)
        chunk = 256
        gate_games = 2
    else:
        phases = default_8c_phases(args.players)
        chunk = args.eval_every
        gate_games = args.gate_games

    curriculum = sprint_8c_curriculum(
        total_timesteps=sum(p.steps for p in phases),
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
    )
    curriculum = dataclasses.replace(
        curriculum,
        phase1_eval_every=chunk,
        gate_games=gate_games,
    )

    print(
        f"finetune_ppo: warm_start={args.warm_start} players={args.players} "
        f"net={args.net} device={args.device} smoke={args.smoke} "
        f"kl_to_bc_coef={args.kl_to_bc_coef} run_name={args.run_name}"
    )
    print("  phases: " + ", ".join(
        f"{p.name}({p.num_players}p,{p.reward_mode},{p.steps})" for p in phases
    ))

    if args.kl_to_bc_coef > 0.0:
        # Anti-forgetting: penalize actor drift from the frozen BC reference.
        # Imported lazily so the proven (no-KL) smoke path has zero dependency on
        # the regularizer module.
        from heat.ml.kl_regularizer import set_kl_to_bc

        set_kl_to_bc(reference_path=args.warm_start, coef=args.kl_to_bc_coef,
                     device=args.device)
        print(f"  KL-to-BC anti-forgetting ON (coef={args.kl_to_bc_coef})")

    model, best_path = train_self_play(
        config=cfg,
        curriculum=curriculum,
        num_players=args.players,
        warm_start_path=args.warm_start,
        phases=phases,
    )
    final_path = f"{best_path}_final"
    print("\n=== finetune_ppo summary ===")
    print(f"  best checkpoint:  {best_path}")
    print(f"  final checkpoint: {final_path}")
    print("  NOTE: gate the BEST checkpoint with experiments/eval_bc.py "
          "(solo limit-1) + the league, never finish-rate alone.")
    return best_path, final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=str, required=True,
                        help="BC or DAgger checkpoint to warm-start from (.zip)")
    parser.add_argument("--run-name", type=str, default="heat_s4_finetune",
                        help="checkpoint run name (best = this, final = _final)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"],
                        help="MUST match the warm-start checkpoint's net size")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--verbose", type=int, default=1,
                        help="SB3 verbosity (1 = print periodic rollout/train "
                             "tables so a long run shows signs of life; 0 = quiet)")
    parser.add_argument("--eval-every", type=int, default=50_000,
                        help="learn steps per gate chunk (non-smoke)")
    parser.add_argument("--gate-games", type=int, default=20)
    parser.add_argument("--kl-to-bc-coef", type=float, default=0.0,
                        help="KL-to-BC actor regularizer weight (0 = off). The "
                             "S4 anti-forgetting lever; recommended during the "
                             "strong phase of a real run.")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny step budgets to prove the pipeline end-to-end")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("finetune_ppo", main)
