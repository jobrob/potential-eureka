"""Sprint S4 gate -- behavioral + league eval of a fine-tuned / DAgger net.

The S4 success criterion is: the learned net (NO search at inference) drives
limit-1 corners **>= the search expert** on worst-case spins/pass AND beats the
weak heuristic on generated 4p win-rate. This harness reports both, honestly:

1. **Behavioral gate (held-out generated, solo).** Reuses the *exact*
   spins-by-corner-limit / finish / rounds reconstruction from
   ``experiments/eval_search.py`` (and ``eval_bc.py``'s gate verdict), comparing
   the candidate net against the Heuristic (bar to clear) and the Lookahead
   expert (the imitation target). Gates on **worst-case (max) L1 spins/pass and
   rounds-to-finish**, NEVER finish-rate alone -- the S3 lesson (the engine's lap
   counter "finishes" even a crawl-and-spin policy).
2. **League gate (4p win-rate, seat-neutral + Wilson-LB).** Rotates the candidate
   through all four seats (so the front-seat positional advantage cancels, per
   the seat-bias finding) and reports the pooled win-rate + its Wilson 95% lower
   bound vs the weak heuristic (the success bar: LB > parity = 25%) and vs the
   strong heuristic (reported, flagged below parity).

Both load the candidate as an ordinary :class:`MLAgent` through the frozen
contract sidecar + ``CODEC_VERSION`` tripwire, so a green result here is a green
result in ``evaluate_ml`` / the production league.

Usage:
    python experiments/eval_dagger.py --net-ckpt checkpoints/dagger_iter3.zip \\
        --games 24 --winrate-tracks 8
    python experiments/eval_dagger.py --net-ckpt checkpoints/heat_s4 --games 24
"""

from __future__ import annotations

import argparse

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.ml_agent import MLAgent
from heat.agents.search_agent import LookaheadAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.simulation.stats import wilson_interval

from eval_search import (  # noqa: E402  (sibling experiments module)
    _HELDOUT_BASE,
    _run_field,
    _print_field,
    _seat_neutral_winrate,
    _seat_neutral_win_counts,
)
from eval_bc import _gate  # noqa: E402  (the S3/S4 behavioral gate verdict)


def _league_gate(
    make_net: "callable",
    *,
    track_seeds: list[int],
    seed_base: int,
) -> None:
    """Seat-neutral 4p win-rate + Wilson-LB of the candidate net (the S4 league bar).

    Success bar: the candidate's pooled win-rate vs three weak heuristics, with a
    Wilson 95% **lower bound > parity** (25% in a 4p field). The strong-heuristic
    number is reported and flagged (competitive if LB >= parity, soft otherwise) --
    secondary, as in S2. All seat-neutral (focal rotated through all four seats).
    """
    print("\n=== S4 league gate (4p win-rate, seat-neutral + Wilson-LB) ===")
    parity = 1.0 / 4.0

    weak_wins, weak_games = _seat_neutral_win_counts(
        make_net, lambda: HeuristicAgent(),
        track_seeds=track_seeds, game_seed_base=seed_base,
    )
    strong_wins, strong_games = _seat_neutral_win_counts(
        make_net, lambda: StrongHeuristicAgent(strength=2),
        track_seeds=track_seeds, game_seed_base=seed_base + 4000,
    )
    weak_wr = weak_wins / weak_games if weak_games else 0.0
    strong_wr = strong_wins / strong_games if strong_games else 0.0
    weak_lb, _ = wilson_interval(weak_wins, weak_games)
    strong_lb, _ = wilson_interval(strong_wins, strong_games)

    print(f"  parity (4p) = {parity * 100:.0f}%")
    print(f"  vs 3x weak  heuristic: {weak_wr * 100:.1f}%  "
          f"(Wilson-LB {weak_lb * 100:.1f}%, {weak_wins}/{weak_games})")
    print(f"  vs 3x strong heuristic: {strong_wr * 100:.1f}%  "
          f"(Wilson-LB {strong_lb * 100:.1f}%, {strong_wins}/{strong_games})")
    print(
        f"  -> vs weak: Wilson-LB > parity:  "
        f"{'PASS' if weak_lb > parity + 1e-9 else 'FAIL'}\n"
        f"  -> vs strong competitive:        "
        f"{'PASS (LB >= parity)' if strong_lb >= parity - 1e-9 else 'SOFT (< parity)'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-ckpt", type=str, required=True,
                        help="candidate net checkpoint (.zip + .meta.json sidecar)")
    parser.add_argument("--games", type=int, default=24,
                        help="held-out generated tracks for the behavioral gate")
    parser.add_argument("--winrate-tracks", type=int, default=8,
                        help="held-out tracks for the seat-neutral league gate "
                             "(x4 seats each)")
    parser.add_argument("--players", type=int, default=1,
                        help="seats for the behavioral field (1 = solo gate)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="expert LookaheadAgent rollout depth")
    parser.add_argument("--dets", type=int, default=2,
                        help="expert determinizations per candidate")
    parser.add_argument("--top-k", type=int, default=6,
                        help="expert top-k branching cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-league", action="store_true",
                        help="run only the (faster) behavioral gate")
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]
    net_path = args.net_ckpt

    def make_heuristic() -> BaseAgent:
        return HeuristicAgent()

    def make_lookahead() -> BaseAgent:
        return LookaheadAgent(
            horizon=args.horizon,
            n_determinizations=args.dets,
            determinize_hidden=args.players > 1,
            top_k=args.top_k,
        )

    def make_net() -> BaseAgent:
        return MLAgent(net_path, deterministic=True, name="Net")

    labels = {
        "Heuristic": make_heuristic,
        "Lookahead": make_lookahead,
        "BC": make_net,  # keyed "BC" so eval_bc._gate finds the candidate
    }

    print(
        f"eval_dagger: net={net_path} {args.games} held-out generated tracks "
        f"(tight/limit-1 weighted), players={args.players}, "
        f"expert horizon={args.horizon} dets={args.dets} top_k={args.top_k}"
    )

    field, prof = _run_field(
        labels,
        track_seeds=track_seeds,
        num_players=args.players,
        game_seed_base=args.seed,
    )
    title = "SOLO (behavioral gate)" if args.players == 1 else f"{args.players}P"
    _print_field(title, field, prof)
    _gate(field, args.players)

    if not args.skip_league:
        league_seeds = [_HELDOUT_BASE + i for i in range(args.winrate_tracks)]
        _league_gate(
            make_net, track_seeds=league_seeds, seed_base=args.seed + 9000
        )


if __name__ == "__main__":
    from _runlog import run_main

    run_main("eval_dagger", main)
