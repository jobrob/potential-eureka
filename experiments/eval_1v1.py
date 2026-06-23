"""Sprint C6 gate -- seat-neutral 1v1 win-rate of the perfect-info MCTS agent.

The C6 "moving loop" prize is read three ways, all seat-neutral and Wilson-LB
gated on the held-out generated ``_TIGHT_PARAMS`` distribution:

  (a) **vs a frozen reference over time** -- seat-neutral 1v1 win-rate vs
      ``StrongHeuristicAgent(strength=2)``; the real bar is **Wilson-LB > 50%**
      (parity). Win-rate vs the *weak* heuristic is a sanity floor.
  (b) **vs earlier generations of itself** -- seat-neutral 1v1 win-rate of the
      candidate vs a frozen earlier-generation MCTS (the direct curriculum-
      escalator signal); a Wilson-LB-separated win-rate **> 50%**, climbing, is
      the single most important read.
  (c) **value-head calibration** -- reported by ``train_az`` (the ECE), not here.

Everything is **seat-neutral**: 1v1 has a turn-order seat bias (the front seat
wins more, all else equal -- the documented finding behind
``_seat_neutral_win_counts``), so the focal agent is rotated through BOTH seats
and the win indicator pooled (``num_seats=2``, parity = 50%). This is the 2-seat
analogue of the S4 ``eval_dagger._league_gate`` (which used ``num_seats=4``,
parity 25%); we reuse the SAME harness with ``num_seats=2``.

The focal agent is the two-player perfect-info ``MCTSAgent`` (in-search, the C6
artifact). It plays a 1v1 game by branching the opponent's decisions as a
searched minimax node (``MCTSConfig.two_player`` on); the opponent seat plays
heuristically (the anchor) or as a frozen MCTS (vs earlier selves).

Usage:
    python experiments/eval_1v1.py --net checkpoints/c6_best.zip --games 24
    python experiments/eval_1v1.py --net checkpoints/c6_gen3.zip \\
        --prev-net checkpoints/c6_gen2.zip --games 24 --sims 16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.mcts_agent import MCTSConfig, _make_mcts_agent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.simulation.stats import wilson_interval

from eval_search import _HELDOUT_BASE, _seat_neutral_win_counts


#: 1v1 parity: with two seats and the seat bias cancelled, even play is 50%.
_PARITY_1V1 = 0.5


def make_mcts_focal(model_path: str, *, sims: int, seed: int) -> "callable":
    """A picklable factory making a two-player perfect-info MCTS focal agent.

    Seat-rotated by ``_seat_neutral_win_counts``; ``opponent_id`` is set to the
    OTHER seat per construction (the focal is always created knowing it faces the
    single other seat in a 1v1). ``two_player=True`` makes the opponent a searched
    minimax node (perfect information -- both hands visible).
    """
    def make(seat: int = 0) -> BaseAgent:
        cfg = MCTSConfig(
            n_simulations=sims, two_player=True, opponent_id=1 - seat,
        )
        return _make_mcts_agent(
            player_id=seat, seed=seed, model_path=model_path,
            name="C6-MCTS", config=cfg,
        )
    # _seat_neutral_win_counts calls make() with no args (seat 0 framing); the
    # opponent_id flips automatically only matters for the focal's own search,
    # which is symmetric in a 1v1, so the no-arg default is sufficient.
    return lambda: make(0)


def _winrate_with_lb(
    make_focal: "callable",
    make_opponent: "callable",
    *,
    track_seeds: list[int],
    seed_base: int,
) -> tuple[float, float, int, int]:
    """``(win_rate, wilson_lb, wins, games)`` seat-neutral 1v1 (num_seats=2)."""
    wins, games = _seat_neutral_win_counts(
        make_focal, make_opponent,
        track_seeds=track_seeds, game_seed_base=seed_base, num_seats=2,
    )
    wr = wins / games if games else 0.0
    lb, _ = wilson_interval(wins, games)
    return wr, lb, wins, games


def league_gate_1v1(
    make_focal: "callable",
    *,
    track_seeds: list[int],
    seed_base: int,
    make_prev: "callable | None" = None,
) -> dict:
    """The C6 seat-neutral 1v1 win-rate gate (reads (a) and (b)).

    Returns a dict with the strong/weak/prev win-rates + Wilson-LBs. The promotion
    bar (read (a)) is the **strong-heuristic Wilson-LB > 50%**; read (b) is the
    win-rate vs the previous generation (``make_prev``), the curriculum-escalator
    signal. All seat-neutral (focal rotated through both seats).
    """
    strong_wr, strong_lb, strong_w, strong_g = _winrate_with_lb(
        make_focal, lambda: StrongHeuristicAgent(strength=2),
        track_seeds=track_seeds, seed_base=seed_base,
    )
    weak_wr, weak_lb, weak_w, weak_g = _winrate_with_lb(
        make_focal, lambda: HeuristicAgent(),
        track_seeds=track_seeds, seed_base=seed_base + 4000,
    )
    out = {
        "parity": _PARITY_1V1,
        "vs_strong": {"win_rate": strong_wr, "wilson_lb": strong_lb,
                      "wins": strong_w, "games": strong_g},
        "vs_weak": {"win_rate": weak_wr, "wilson_lb": weak_lb,
                    "wins": weak_w, "games": weak_g},
        # The promotion bar: Wilson-LB vs the strong heuristic clears parity.
        "promote": strong_lb > _PARITY_1V1 + 1e-9,
    }
    if make_prev is not None:
        prev_wr, prev_lb, prev_w, prev_g = _winrate_with_lb(
            make_focal, make_prev,
            track_seeds=track_seeds, seed_base=seed_base + 8000,
        )
        out["vs_prev"] = {"win_rate": prev_wr, "wilson_lb": prev_lb,
                          "wins": prev_w, "games": prev_g}
    return out


def _print_gate(gate: dict) -> None:
    p = gate["parity"]
    print("\n=== C6 seat-neutral 1v1 win-rate gate (Wilson-LB) ===")
    print(f"  parity (1v1) = {p * 100:.0f}%")
    s = gate["vs_strong"]
    w = gate["vs_weak"]
    print(f"  vs strong heuristic: {s['win_rate'] * 100:.1f}%  "
          f"(Wilson-LB {s['wilson_lb'] * 100:.1f}%, {s['wins']}/{s['games']})")
    print(f"  vs weak   heuristic: {w['win_rate'] * 100:.1f}%  "
          f"(Wilson-LB {w['wilson_lb'] * 100:.1f}%, {w['wins']}/{w['games']})")
    if "vs_prev" in gate:
        pv = gate["vs_prev"]
        print(f"  vs gen k-1 (self):   {pv['win_rate'] * 100:.1f}%  "
              f"(Wilson-LB {pv['wilson_lb'] * 100:.1f}%, {pv['wins']}/{pv['games']})")
    print(
        f"  -> (a) vs strong: Wilson-LB > 50%: "
        f"{'PASS' if gate['promote'] else 'not yet'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=str, required=True,
                        help="candidate C6 net checkpoint (.zip + .meta.json, "
                             "value_mode=winloss)")
    parser.add_argument("--prev-net", type=str, default=None,
                        help="an earlier-generation net for read (b) (vs self); "
                             "omit to skip the self-vs-self read")
    parser.add_argument("--games", type=int, default=24,
                        help="held-out tracks (x2 seats = seat-neutral 1v1 games)")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTS sims/move for the in-search 1v1 agent")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]
    make_focal = make_mcts_focal(args.net, sims=args.sims, seed=args.seed)
    make_prev = None
    if args.prev_net is not None:
        make_prev = make_mcts_focal(args.prev_net, sims=args.sims, seed=args.seed + 1)

    print(f"eval_1v1: net={args.net} prev={args.prev_net} games={args.games} "
          f"sims={args.sims} (seat-neutral, num_seats=2)")
    gate = league_gate_1v1(
        make_focal, track_seeds=track_seeds, seed_base=args.seed + 9000,
        make_prev=make_prev,
    )
    _print_gate(gate)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("eval_1v1", main)
