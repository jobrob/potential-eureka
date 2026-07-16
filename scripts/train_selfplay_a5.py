#!/usr/bin/env python
"""CLI for the A5 anti-collapse self-play recipe (Sprint A5).

Thin front-end over :func:`heat.ml.selfplay.recipe.train_selfplay_a5`. Trains a
shared self-play policy with the entropy-floor parachute + recent-snapshot
opponent pool + Stage-1 validation, printing per-iteration diagnostics and each
eval record as it lands.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/train_selfplay_a5.py \\
        --timesteps 150000 --players 2 --track tiny --seed 0 --arm pool

``--arm pure`` is the ablation arm (``pool_prob 0`` -- no pool opponents ever).
"""

from __future__ import annotations

import argparse
import sys
import time

from heat.models.track import Track
from heat.ml.selfplay.recipe import A5Config, Stage1ValidationError, train_selfplay_a5


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A5 self-play recipe.")
    parser.add_argument("--timesteps", type=int, default=150_000,
                        help="Total transitions to train for (default: 150000).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Transitions per rollout / update (default: 2048).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size for the PPO update (default: 256).")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Optimization passes per rollout (default: 10).")
    parser.add_argument("--players", type=int, default=2,
                        help="Seats per game, 2..6 (default: 2).")
    parser.add_argument("--track", choices=["tiny", "usa"], default="tiny",
                        help="Track bed: tiny-heat or the full USA track.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Torch device; cuda falls back to CPU (default: auto).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for reproducibility (default: 0).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Policy/value MLP trunk widths (default: 256 256).")
    parser.add_argument("--head", choices=["masked", "dotprod"], default="masked",
                        help="Action head (default: masked, per the A3 result).")
    parser.add_argument("--encoder", choices=["flat", "structured"], default="flat",
                        help="Observation encoder (default: flat until A4 gate passes).")
    parser.add_argument("--arm", choices=["pool", "pure"], default="pool",
                        help="pool (default) uses the snapshot pool; pure sets "
                             "pool_prob=0 (the ablation arm).")
    parser.add_argument("--entropy-floor", type=float, default=0.40,
                        help="Controller engages below this entropy (default: 0.40).")
    parser.add_argument("--pool-prob", type=float, default=0.5,
                        help="Per-iteration pool-opponent probability (default: 0.5).")
    parser.add_argument("--margin-coef", type=float, default=0.0,
                        help="A6 dense terminal-margin coefficient added to the "
                             "training reward (default: 0.0 == off).")
    parser.add_argument("--pool-capacity", type=int, default=5,
                        help="Recent snapshots retained (default: 5).")
    parser.add_argument("--snapshot-every", type=int, default=10,
                        help="Iterations between pool pushes (default: 10).")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Iterations between eval checkpoints (default: 10).")
    parser.add_argument("--eval-games", type=int, default=40,
                        help="Eval games per checkpoint (default: 40).")
    parser.add_argument("--no-stage1", action="store_true",
                        help="Disable the Stage-1 validation check.")
    parser.add_argument("--save", type=str, default=None, metavar="PATH",
                        help="Save the FINAL trained policy to PATH (Sprint A7 "
                             "checkpoint: save_policy). Anchor tip: an early "
                             "checkpoint is just a second run with the SAME "
                             "--seed and a smaller --timesteps -- iterations "
                             "1..k are byte-identical, so e.g. --timesteps 30720 "
                             "(15 iters at n_steps=2048) --seed 2 --save "
                             "anchor.pt reproduces exactly the policy the full "
                             "150k --seed 2 run holds at iteration 15.")
    return parser.parse_args(argv)


def _resolve_track(name: str) -> Track:
    if name == "tiny":
        from heat.ml.selfplay.tiny_heat import tiny_heat_track

        return tiny_heat_track()
    from heat.tracks.loader import load_track_by_name

    return load_track_by_name("usa")


def build_config(args: argparse.Namespace) -> A5Config:
    """Build an :class:`A5Config` from parsed CLI args (shared with the gate)."""
    return A5Config(
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
        encoder=args.encoder,
        entropy_floor=args.entropy_floor,
        pool_prob=0.0 if args.arm == "pure" else args.pool_prob,
        margin_coef=args.margin_coef,
        pool_capacity=args.pool_capacity,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        stage1_enabled=not args.no_stage1,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not (2 <= args.players <= 6):
        raise SystemExit(f"--players must be in 2..6, got {args.players}")

    config = build_config(args)
    track = _resolve_track(args.track)

    def _log(iteration: int, info: dict[str, float]) -> None:
        engaged = "ENGAGED" if info["engaged"] else "-"
        print(
            f"iter {iteration:>4d} | "
            f"policy_loss {info['policy_loss']:+.4f} | "
            f"value_loss {info['value_loss']:.4f} | "
            f"entropy {info['entropy']:.4f} | "
            f"ent_coef {info['ent_coef']:.4f} {engaged} | "
            f"steps {int(info['steps'])}",
            flush=True,
        )

    print("=" * 72)
    print("HEAT A5 anti-collapse self-play recipe")
    print("=" * 72)
    print(
        f"arm={args.arm} timesteps={args.timesteps} n_steps={args.n_steps} "
        f"players={args.players} track={args.track} head={args.head} "
        f"encoder={args.encoder} "
        f"seed={args.seed} entropy_floor={args.entropy_floor} "
        f"pool_prob={config.pool_prob} stage1={config.stage1_enabled}"
    )

    t0 = time.perf_counter()
    try:
        policy, records = train_selfplay_a5(config, track=track, on_iteration=_log)
    except Stage1ValidationError as exc:
        print("=" * 72)
        print(f"STAGE-1 ABORT: {exc}")
        print(f"diagnostics: {exc.diagnostics}")
        return 2
    elapsed = time.perf_counter() - t0

    if args.save is not None:
        from heat.ml.selfplay.checkpoint import save_policy

        save_policy(policy, config, args.save)
        print(f"Saved final policy -> {args.save}")

    print("=" * 72)
    print("Eval records:")
    print("| iter | steps | vs-weak WR | vs-oldest WR | entropy | ent_coef | engaged |")
    print("|---|---|---|---|---|---|---|")
    for rec in records:
        print(
            f"| {int(rec['iteration'])} | {int(rec['steps'])} | "
            f"{rec['winrate_vs_weak']:.3f} | {rec['winrate_vs_oldest']:.3f} | "
            f"{rec['entropy']:.3f} | {rec['ent_coef']:.4f} | "
            f"{'yes' if rec['ever_engaged'] else 'no'} |"
        )
    print("=" * 72)
    print(f"A5 run complete in {elapsed:.1f}s ({len(records)} eval checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
