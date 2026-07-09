#!/usr/bin/env python
"""Minimal CLI for the A2 N-agent shared-policy self-play harness (Sprint A2).

Thin front-end over :func:`heat.ml.selfplay.multiseat.train_multiseat`. Unlike the
A0 script (single learner vs scripted opponents), every non-scripted seat here is
driven by the *same live policy* and recorded into its own per-seat trajectory
stream. The per-iteration print matches A0's format; a final line reports the G3
throughput measurement (recorded transitions/sec) so it can be eyeballed against
the A0 single-seat rate.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/train_selfplay_a2.py \\
        --timesteps 10000 --players 2 --track tiny --device cpu --seed 0

Add ``--scripted-opponents N`` to fill the last N seats with ``HeuristicAgent``s
(default 0 == pure self-play).
"""

from __future__ import annotations

import argparse
import sys
import time

from heat.agents.base import BaseAgent
from heat.models.track import Track
from heat.ml.selfplay import A0Config, train_multiseat


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the A2 N-agent shared-policy self-play harness.",
    )
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Total transitions to train for (default: 10000).")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Transitions collected per rollout / update (default: 512).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size for the PPO update (default: 256).")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Optimization passes per rollout (default: 10).")
    parser.add_argument("--players", type=int, default=2,
                        help="Seats per game, 2..6 (default: 2).")
    parser.add_argument("--track", choices=["tiny", "usa"], default="tiny",
                        help="Track bed: tiny-heat or the full USA track (default: tiny).")
    parser.add_argument("--scripted-opponents", type=int, default=0,
                        help="Fill the last N seats with HeuristicAgents (default: 0).")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Torch device; cuda falls back to CPU (default: auto).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for reproducibility (default: 0).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Policy/value MLP trunk widths (default: 256 256).")
    return parser.parse_args(argv)


def _resolve_track(name: str) -> Track:
    """Return the requested track bed."""
    if name == "tiny":
        from heat.ml.selfplay.tiny_heat import tiny_heat_track

        return tiny_heat_track()
    from heat.tracks.loader import load_track_by_name

    return load_track_by_name("usa")


def main(argv: list[str] | None = None) -> int:
    from heat.agents.heuristic_agent import HeuristicAgent

    args = _parse_args(argv)

    if not (2 <= args.players <= 6):
        raise SystemExit(f"--players must be in 2..6, got {args.players}")
    if not (0 <= args.scripted_opponents < args.players):
        raise SystemExit(
            f"--scripted-opponents must be in 0..{args.players - 1}, "
            f"got {args.scripted_opponents}"
        )

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
    )

    # Fill the LAST N seats with scripted HeuristicAgents (pure self-play if 0).
    scripted: dict[int, BaseAgent] = {
        args.players - 1 - i: HeuristicAgent()
        for i in range(args.scripted_opponents)
    }
    track = _resolve_track(args.track)
    n_policy_seats = args.players - args.scripted_opponents

    # G3 throughput bookkeeping: total recorded transitions and wall time.
    stats = {"recorded": 0}
    t0 = time.perf_counter()

    def _log(iteration: int, info: dict[str, float]) -> None:
        stats["recorded"] += int(info["n_recorded"])
        print(
            f"iter {iteration:>4d} | "
            f"policy_loss {info['policy_loss']:+.4f} | "
            f"value_loss {info['value_loss']:.4f} | "
            f"entropy {info['entropy']:.4f} | "
            f"mean_return {info['mean_episode_return']:+.4f} "
            f"(n_ep={int(info['n_episodes'])}, n_rec={int(info['n_recorded'])})"
        )

    print("=" * 64)
    print("HEAT A2 N-agent shared-policy self-play harness")
    print("=" * 64)
    print(
        f"timesteps={args.timesteps} n_steps={args.n_steps} "
        f"players={args.players} (policy={n_policy_seats}, "
        f"scripted={args.scripted_opponents}) track={args.track} "
        f"device={args.device} seed={args.seed}"
    )

    train_multiseat(config, track=track, scripted_seats=scripted or None,
                    on_iteration=_log)

    elapsed = time.perf_counter() - t0
    rate = stats["recorded"] / elapsed if elapsed > 0 else float("nan")
    print("=" * 64)
    print(
        f"A2 run complete: {stats['recorded']} transitions in {elapsed:.2f}s "
        f"-> {rate:,.0f} recorded transitions/sec (G3)"
    )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
