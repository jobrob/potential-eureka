"""Sprint S3 (BC) — behavioral eval of the cloned net on held-out generated tracks.

The gate for S3 is *behavioral*, not label accuracy: does the BC net (NO search
at inference) drive limit-1 corners within a small margin of the search expert
and **finish** generated tracks? This harness answers that by reusing the exact
spins-by-corner-limit / finish / rounds reconstruction from
``experiments/eval_search.py`` (the design's cross-cutting reporting rule), and
adding the BC checkpoint -- loaded as an ordinary :class:`MLAgent` through the
frozen contract sidecar + ``CODEC_VERSION`` -- as a contender alongside:

    * ``Heuristic``   -- the hand-coded reference (the bar to clear)
    * ``Lookahead``   -- the S1/S2 search EXPERT the BC net is cloned from
    * ``BC``          -- the cloned MaskablePPO net (this sprint's deliverable)

It runs the SAME held-out generated track band as ``eval_search.py``
(``_HELDOUT_BASE`` = 900_000+, tight/limit-1-weighted params) -- which is
DISJOINT from the train (100_000+) and val (500_000+) bands ``gen_demos.py``
samples, so the BC net is gated on tracks it has never seen.

The BC net is evaluated through :class:`MLAgent` exactly as ``evaluate_ml``
would load it (the sidecar tripwire asserts the codec matches), so a green
result here is a green result in the league.

Usage:
    python experiments/eval_bc.py --bc checkpoints/bc_solo.zip --games 24
    python experiments/eval_bc.py --bc checkpoints/bc_4p.zip --games 30 --players 4
"""

from __future__ import annotations

import argparse

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.ml_agent import MLAgent
from heat.agents.search_agent import LookaheadAgent

# Reuse the eval_search reconstruction + reporting verbatim (single source of
# truth for the spins-by-limit / pass accounting the design mandates).
from eval_search import (  # noqa: E402  (sibling experiments module)
    _HELDOUT_BASE,
    _run_field,
    _print_field,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=str, required=True,
                        help="BC checkpoint path (.zip, with .meta.json sidecar)")
    parser.add_argument("--games", type=int, default=24,
                        help="held-out generated tracks to evaluate on")
    parser.add_argument("--players", type=int, default=1,
                        help="seats per game (1 = solo primary gate; 4 = 4p)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="expert LookaheadAgent rollout depth")
    parser.add_argument("--dets", type=int, default=2,
                        help="expert determinizations per candidate")
    parser.add_argument("--top-k", type=int, default=6,
                        help="expert top-k branching cap")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed for game RNG (track seeds are held-out)")
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]
    bc_path = args.bc

    def make_heuristic() -> BaseAgent:
        return HeuristicAgent()

    def make_lookahead() -> BaseAgent:
        determinize = args.players > 1
        return LookaheadAgent(
            horizon=args.horizon,
            n_determinizations=args.dets,
            determinize_hidden=determinize,
            top_k=args.top_k,
        )

    def make_bc() -> BaseAgent:
        # Loaded exactly as evaluate_ml would: the MLAgent lazy-loads the model
        # and validates the contract sidecar (OBS_DIM/ACTION_DIM/CODEC_VERSION).
        return MLAgent(bc_path, deterministic=True, name="BC")

    labels = {
        "Heuristic": make_heuristic,
        "Lookahead": make_lookahead,
        "BC": make_bc,
    }

    print(
        f"eval_bc: bc={bc_path} {args.games} held-out generated tracks "
        f"(tight/limit-1 weighted), players={args.players}, "
        f"expert horizon={args.horizon} dets={args.dets} top_k={args.top_k}"
    )

    field, prof = _run_field(
        labels,
        track_seeds=track_seeds,
        num_players=args.players,
        game_seed_base=args.seed,
    )
    title = "SOLO (primary gate)" if args.players == 1 else f"{args.players}P"
    _print_field(title, field, prof)

    _gate(field, args.players)


def _gate(field, players: int) -> None:
    """Print the S3 gate verdict: BC vs expert/heuristic on limit-1, honestly.

    The S3 success criterion is: the BC net (no search) achieves
    spins/limit-1-pass *within a small margin of the search agent* and
    *finishes* generated tracks. We report BC's worst-case (max) and p90
    limit-1 spins/pass against both the Lookahead expert and the Heuristic, plus
    finish rate -- and flag the headline "a learned policy clears the limit-1
    corner" result (BC finishes AND its worst-case limit-1 spins <= heuristic).
    """
    bc = field.get("BC")
    look = field.get("Lookahead")
    heur = field.get("Heuristic")
    if bc is None or look is None or heur is None:
        print("\n(missing aggregates for the gate)")
        return

    def fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.3f}"

    bc_mean, bc_p90, bc_max = bc.spin_stats(1)
    lk_mean, lk_p90, lk_max = look.spin_stats(1)
    hr_mean, hr_p90, hr_max = heur.spin_stats(1)

    print("\n=== S3 gate (limit-1 spins/pass; BC = no search at inference) ===")
    print(f"  BC        finish={bc.finish_rate() * 100:.0f}%  rounds={fmt(bc.mean_rounds())}  "
          f"L1 spins/pass mean/p90/max = {fmt(bc_mean)}/{fmt(bc_p90)}/{fmt(bc_max)}")
    print(f"  Lookahead finish={look.finish_rate() * 100:.0f}%  rounds={fmt(look.mean_rounds())}  "
          f"L1 spins/pass mean/p90/max = {fmt(lk_mean)}/{fmt(lk_p90)}/{fmt(lk_max)}")
    print(f"  Heuristic finish={heur.finish_rate() * 100:.0f}%  rounds={fmt(heur.mean_rounds())}  "
          f"L1 spins/pass mean/p90/max = {fmt(hr_mean)}/{fmt(hr_p90)}/{fmt(hr_max)}")

    # Gate components (NaN-safe: no limit-1 passes -> treat that metric as vacuous).
    bc_finishes = bc.finish_rate() >= (0.999 if players == 1 else 0.0)
    le_heur = (
        bc_max != bc_max or hr_max != hr_max or bc_max <= hr_max + 1e-9
    )
    # "within a small margin of the expert": BC's worst-case L1 within +0.5 of
    # the expert's worst-case (a generous but explicit margin the doc asks us to
    # report rather than hide).
    margin = 0.5
    near_expert = (
        bc_max != bc_max or lk_max != lk_max or bc_max <= lk_max + margin + 1e-9
    )
    print(
        f"  -> BC finishes solo:                 "
        f"{'PASS' if bc_finishes else 'FAIL'}\n"
        f"  -> BC worst-case L1 <= heuristic:    "
        f"{'PASS' if le_heur else 'FAIL'}\n"
        f"  -> BC worst-case L1 within {margin:+.1f} of expert: "
        f"{'PASS' if near_expert else 'FAIL'}"
    )


if __name__ == "__main__":
    from _runlog import run_main

    run_main("eval_bc", main)
