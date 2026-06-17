#!/usr/bin/env python3
"""Watch the trained ML agent race vs a mix of heuristics on a random track.

Drops a saved checkpoint's :class:`MLAgent` into a 4-player race against a mix
of heuristic opponents on a *procedurally generated* track, and prints a live
play-by-play (via :func:`heat.viewer.watch_game`) plus the ML agent's own
gear / cards / react / slipstream / discard decisions each turn, so you can see
exactly what it is choosing.

Usage (from repo root):
    PYTHONPATH=src python scripts/watch_ml_race.py
    PYTHONPATH=src python scripts/watch_ml_race.py --races 3 --laps 2 --seed 42
    PYTHONPATH=src python scripts/watch_ml_race.py --stochastic   # sample, don't argmax
    PYTHONPATH=src python scripts/watch_ml_race.py --checkpoint checkpoints/heat_ppo_strong
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The viewer renders the track with Unicode box-drawing art; force UTF-8 so the
# Windows console (cp1252 by default) does not choke on it.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents.ml_agent import MLAgent
from heat.tracks.generator import generate_track
from heat.viewer import watch_game


def _fmt_cards(cards) -> str:
    if not cards:
        return "(none)"
    return "[" + ", ".join(f"{c.card_type.name}:{c.value}" for c in cards) + "]"


class DecisionLogger:
    """Agent wrapper that prints every decision the wrapped agent makes."""

    def __init__(self, inner, tag: str = ">>") -> None:
        self.inner = inner
        self.name = inner.name
        self.tag = tag

    def _ctx(self, state, player_id: str | int) -> str:
        p = state.get_player(player_id)
        return (f"lap {p.lap} pos {p.position} gear {p.gear} "
                f"heat {p.heat_available} hand {len(p.hand)}")

    def choose_gear(self, state, player_id, legal_gears):
        choice = self.inner.choose_gear(state, player_id, legal_gears)
        print(f"   {self.tag} [{self._ctx(state, player_id)}]")
        print(f"   {self.tag} GEAR      -> gear {choice[0]} (heat cost {choice[1]}) "
              f"| legal: {[g for g, _ in legal_gears]}")
        return choice

    def choose_cards(self, state, player_id, legal_plays):
        choice = self.inner.choose_cards(state, player_id, legal_plays)
        speed = sum(getattr(c, "value", 0) for c in choice)
        print(f"   {self.tag} CARDS     -> {_fmt_cards(choice)} (sum {speed}) "
              f"| {len(legal_plays)} legal plays")
        return choice

    def choose_react(self, state, player_id, max_cooldown, can_boost, has_adrenaline):
        d = self.inner.choose_react(state, player_id, max_cooldown, can_boost, has_adrenaline)
        print(f"   {self.tag} REACT     -> cooldown {d.cooldown_count}/{max_cooldown} "
              f"boost {d.use_boost} adrenaline_spd {d.use_adrenaline_speed} "
              f"adrenaline_cd {d.use_adrenaline_cooldown}")
        return d

    def choose_slipstream(self, state, player_id):
        choice = self.inner.choose_slipstream(state, player_id)
        print(f"   {self.tag} SLIPSTREAM-> {'TAKE +2' if choice else 'decline'}")
        return choice

    def choose_discard(self, state, player_id, discardable):
        choice = self.inner.choose_discard(state, player_id, discardable)
        if choice:
            print(f"   {self.tag} DISCARD   -> {_fmt_cards(choice)}")
        return choice


def _build_opponents() -> list:
    """A mix of the heuristic agents for the 3 non-ML seats."""
    return [
        StrongHeuristicAgent(name="Strong-2", strength=2),
        HeuristicAgent(name="Heuristic-A"),
        HeuristicAgent(name="Heuristic-B"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/heat_ppo_strong",
                        help="Path to the saved checkpoint (default: checkpoints/heat_ppo_strong).")
    parser.add_argument("--races", type=int, default=2, help="Number of races (default: 2).")
    parser.add_argument("--laps", type=int, default=None,
                        help="Laps override (default: generated track's default).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; each race uses seed+i. Default: random.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of argmax (default: deterministic).")
    parser.add_argument("--ml-seat", type=int, default=0,
                        help="Seat (0..3) the ML agent starts in (default: 0).")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1_000_000)

    for i in range(args.races):
        race_seed = base_seed + i
        # Seed the global RNG too (the engine's deck shuffle uses it), so a given
        # --seed reproduces the whole race, not just the track layout.
        random.seed(race_seed)
        track = generate_track(race_seed, name=f"random-{race_seed}")
        if args.laps is not None:
            track.laps = args.laps

        ml = DecisionLogger(
            MLAgent(args.checkpoint, deterministic=not args.stochastic, name="ML-Agent"),
            tag="ML>>",
        )
        opponents = _build_opponents()
        agents = opponents[:]
        agents.insert(min(args.ml_seat, 3), ml)
        names = [a.name for a in agents]

        print("\n" + "=" * 70)
        print(f"RACE {i + 1}/{args.races}  |  track '{track.name}'  "
              f"len {track.length} corners {len(track.corners)} laps {track.laps}")
        print(f"  checkpoint: {args.checkpoint}  ({'stochastic' if args.stochastic else 'deterministic'})")
        print(f"  lineup: {', '.join(names)}   (ML agent = 'ML-Agent', decisions prefixed 'ML>>')")
        print("=" * 70)

        result = watch_game(track, agents, player_names=names, show_standings=True)

        finish = getattr(result, "finish_order", None)
        if finish:
            placings = " > ".join(names[pid] for pid in finish)
            ml_place = finish.index(args.ml_seat) + 1 if args.ml_seat in finish else "?"
            print(f"\nRACE {i + 1} RESULT: {placings}")
            print(f"  ML-Agent finished {ml_place}/{len(finish)}")


if __name__ == "__main__":
    main()
