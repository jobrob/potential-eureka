#!/usr/bin/env python
"""Minimal CLI to run the A0 custom-PPO self-play skeleton (Sprint A0).

Thin front-end over :func:`heat.ml.selfplay.train`. A0 is the substrate spike --
this script just collects rollouts by driving the existing single-seat
:class:`heat.ml.env.HeatEnv` (scripted opponents) and takes PPO updates, printing
the per-iteration loss terms and mean episode return so the G1 (runs end-to-end)
and G2 (return improves beyond noise) gates can be eyeballed.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/train_selfplay_a0.py \\
        --timesteps 50000 --n-steps 2048 --players 4 \\
        --opponent heuristic --device auto --seed 0
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from heat.agents.base import BaseAgent
from heat.ml.selfplay import A0Config, train


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the A0 custom-PPO self-play training skeleton.",
    )
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total environment steps to train for (default: 50000).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps collected per rollout / PPO update (default: 2048).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size for the PPO update (default: 256).")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Optimization passes per rollout (default: 10).")
    parser.add_argument("--players", type=int, default=4,
                        help="Seats per game, 2..6 (default: 4).")
    parser.add_argument("--opponent", choices=["heuristic", "random"],
                        default="heuristic",
                        help="Scripted opponent policy (default: heuristic).")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Torch device; cuda falls back to CPU (default: auto).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for reproducibility (default: 0).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Policy/value MLP trunk widths (default: 256 256).")
    parser.add_argument("--head", choices=["masked", "dotprod"], default="masked",
                        help="Action head: masked free Linear (default) or the A3 "
                             "feature-derived dot-product head.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # RandomAgent is imported lazily so the default heuristic path has no cost.
    from heat.agents.heuristic_agent import HeuristicAgent
    from heat.agents.random_agent import RandomAgent

    args = _parse_args(argv)

    config = A0Config(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        hidden_sizes=tuple(args.hidden),
        num_players=args.players,
        device=args.device,
        seed=args.seed,
        head=args.head,
    )

    # A factory so each opponent seat gets an independent agent (with its own
    # RNG stream for RandomAgent).
    opponents: Callable[[], BaseAgent]
    if args.opponent == "random":
        opponents = lambda: RandomAgent(seed=args.seed)  # noqa: E731
    else:
        opponents = lambda: HeuristicAgent()  # noqa: E731

    def _log(iteration: int, info: dict[str, float]) -> None:
        print(
            f"iter {iteration:>4d} | "
            f"policy_loss {info['policy_loss']:+.4f} | "
            f"value_loss {info['value_loss']:.4f} | "
            f"update_entropy {info['update_entropy']:.4f} | "
            f"mean_return {info['mean_episode_return']:+.4f} "
            f"(n_ep={int(info['n_episodes'])})"
        )

    print("=" * 60)
    print("HEAT A0 custom-PPO self-play skeleton")
    print("=" * 60)
    print(
        f"timesteps={args.timesteps} n_steps={args.n_steps} "
        f"players={args.players} opponent={args.opponent} head={args.head} "
        f"device={args.device} seed={args.seed}"
    )

    train(config, opponents=opponents, on_iteration=_log)

    print("=" * 60)
    print("A0 run complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
