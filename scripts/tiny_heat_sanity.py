#!/usr/bin/env python
"""Tiny-Heat sanity + throughput / calibration harness (Sprint A1, design §4.4).

Runs ``--games N`` self-play games on **both** Tiny-Heat and USA by driving the
existing single-seat :class:`~heat.ml.env.HeatEnv`, and reports the numbers that
are the A1 deliverable -- not just "it runs":

* **Correctness (asserts, design §5 G1):** every chosen action is in the current
  decision's ``legal_action_mask``; every game **terminates** (``is_game_over``)
  and **none truncates** at ``MAX_ROUNDS``.
* **Cost (design §5 G2):** wall-clock **games/sec** and **env-steps/sec** (single
  process, CPU), plus the Tiny/USA games-per-sec **ratio**.
* **Non-degeneracy diagnostics (design §5 G3):** mean rounds/game, mean learner
  decisions/episode, mean corner encounters/game, finish rate, and the fraction
  of episodes with a real CARDS / heat decision -- the calibration signal that
  tells us whether the tiny game still exercises the hard axes or collapsed into
  a trivial sprint (the design §6 deck-fork trigger).

The learner seat picks a *legal* action each step (uniform over the legal mask by
default, or first-legal in ``--random`` deterministic mode is NOT used -- random
here selects the *opponent* policy). Opponents default to
:class:`~heat.agents.heuristic_agent.HeuristicAgent`; ``--random`` uses
:class:`~heat.agents.random_agent.RandomAgent`.

Exit code is non-zero if any correctness assert fails (so CI / the gate can rely
on it).

Usage::

    PYTHONPATH=src python scripts/tiny_heat_sanity.py --games 200
    PYTHONPATH=src python scripts/tiny_heat_sanity.py --games 200 --random
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.engine.driver import DecisionKind
from heat.engine.game import MAX_ROUNDS
from heat.ml.env import HeatEnv
from heat.ml.selfplay.tiny_heat import make_tiny_env


@dataclass
class GameStats:
    """Accumulated per-track metrics over all sanity games."""

    games: int = 0
    illegal_actions: int = 0
    truncations: int = 0
    unfinished_games: int = 0
    total_env_steps: int = 0
    # Per-game samples (averaged at report time).
    rounds_per_game: list[int] = field(default_factory=list)
    decisions_per_game: list[int] = field(default_factory=list)
    corner_encounters_per_game: list[int] = field(default_factory=list)
    finish_rate_per_game: list[float] = field(default_factory=list)
    episodes_with_cards: int = 0
    episodes_with_heat: int = 0
    wall_seconds: float = 0.0


def _learner_corner_spaces(env: HeatEnv) -> set[int]:
    """All space indices covered by a corner on the env's current track."""
    covered: set[int] = set()
    for corner in env.track.corners:
        covered.update(range(corner.start, corner.end + 1))
    return covered


def _play_one_game(
    env: HeatEnv,
    rng: np.random.Generator,
    stats: GameStats,
) -> None:
    """Drive ``env`` through one full episode, accumulating into ``stats``.

    The learner picks a uniformly-random *legal* action each step (cross-checked
    against the mask so an illegal pick is recorded as a correctness failure).
    """
    seed = int(rng.integers(0, 2**31 - 1))
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]

    corner_spaces = _learner_corner_spaces(env)
    n_decisions = 0
    saw_cards = False
    saw_heat = False
    corner_encounters = 0
    seen_corner_positions: set[int] = set()

    assert env.state is not None
    learner = env.learner_id

    # Heat-payment detection: track the learner's heat-pool size across steps; a
    # decrease means heat was paid (a real heat decision bit this episode).
    prev_heat = env.state.players[learner].heat_available

    done = False
    while not done:
        # Classify the *current* learner decision before acting on it.
        decision = env._decision  # noqa: SLF001 - harness needs the decision kind
        if decision is not None:
            n_decisions += 1
            if decision.kind == DecisionKind.CARDS:
                saw_cards = True

        # Pick a uniformly-random legal action and cross-check the mask (G1).
        legal_idx = np.flatnonzero(mask)
        if legal_idx.size == 0:
            # Should never happen: the env auto-resolves all-False masks.
            stats.illegal_actions += 1
            break
        action = int(rng.choice(legal_idx))
        if not bool(mask[action]):  # pragma: no cover - defensive
            stats.illegal_actions += 1

        obs, _reward, terminated, truncated, info = env.step(action)
        stats.total_env_steps += 1
        mask = info["action_mask"]

        assert env.state is not None
        # Corner encounter: learner's position falls inside a corner span.
        pos = env.state.players[learner].position
        if pos in corner_spaces and pos not in seen_corner_positions:
            corner_encounters += 1
            seen_corner_positions.add(pos)
        cur_heat = env.state.players[learner].heat_available
        if cur_heat < prev_heat:
            saw_heat = True
        prev_heat = cur_heat

        if truncated:
            stats.truncations += 1
        done = bool(terminated or truncated)

    assert env.state is not None
    final = env.state
    if not final.is_game_over:
        stats.unfinished_games += 1
    if final.round_num > MAX_ROUNDS:
        # Defensive: a game that ran past MAX_ROUNDS is a truncation even if the
        # env did not flag it on the terminal step.
        stats.truncations += 1

    n_players = env.num_players
    n_finished = sum(1 for p in final.players if p.finished)

    stats.games += 1
    stats.rounds_per_game.append(final.round_num)
    stats.decisions_per_game.append(n_decisions)
    stats.corner_encounters_per_game.append(corner_encounters)
    stats.finish_rate_per_game.append(n_finished / n_players)
    if saw_cards:
        stats.episodes_with_cards += 1
    if saw_heat:
        stats.episodes_with_heat += 1


def _run_track(
    label: str,
    env_factory: object,
    games: int,
    rng: np.random.Generator,
) -> GameStats:
    """Run ``games`` self-play games on the env built by ``env_factory``."""
    stats = GameStats()
    # Build the env once and reuse it across games (reset() starts each episode).
    env = env_factory()  # type: ignore[operator]
    start = time.perf_counter()
    for _ in range(games):
        _play_one_game(env, rng, stats)
    stats.wall_seconds = time.perf_counter() - start
    print(f"  [{label}] {games} games in {stats.wall_seconds:.2f}s")
    return stats


def _mean(xs: list[float] | list[int]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _report(
    tiny: GameStats,
    usa: GameStats,
) -> bool:
    """Print the Tiny-vs-USA table; return True if all correctness asserts pass."""
    tiny_gps = tiny.games / tiny.wall_seconds if tiny.wall_seconds else float("nan")
    usa_gps = usa.games / usa.wall_seconds if usa.wall_seconds else float("nan")
    tiny_sps = (
        tiny.total_env_steps / tiny.wall_seconds if tiny.wall_seconds else float("nan")
    )
    usa_sps = (
        usa.total_env_steps / usa.wall_seconds if usa.wall_seconds else float("nan")
    )
    ratio = tiny_gps / usa_gps if usa_gps else float("nan")

    def frac_cards(s: GameStats) -> float:
        return s.episodes_with_cards / s.games if s.games else float("nan")

    def frac_heat(s: GameStats) -> float:
        return s.episodes_with_heat / s.games if s.games else float("nan")

    rows = [
        ("games", f"{tiny.games}", f"{usa.games}"),
        ("games/sec", f"{tiny_gps:.1f}", f"{usa_gps:.1f}"),
        ("env-steps/sec", f"{tiny_sps:.0f}", f"{usa_sps:.0f}"),
        ("mean rounds/game", f"{_mean(tiny.rounds_per_game):.2f}",
         f"{_mean(usa.rounds_per_game):.2f}"),
        ("mean decisions/ep", f"{_mean(tiny.decisions_per_game):.2f}",
         f"{_mean(usa.decisions_per_game):.2f}"),
        ("mean corners/game", f"{_mean(tiny.corner_encounters_per_game):.2f}",
         f"{_mean(usa.corner_encounters_per_game):.2f}"),
        ("finish rate", f"{_mean(tiny.finish_rate_per_game):.2f}",
         f"{_mean(usa.finish_rate_per_game):.2f}"),
        ("frac ep w/ CARDS", f"{frac_cards(tiny):.2f}", f"{frac_cards(usa):.2f}"),
        ("frac ep w/ heat", f"{frac_heat(tiny):.2f}", f"{frac_heat(usa):.2f}"),
        ("illegal actions", f"{tiny.illegal_actions}", f"{usa.illegal_actions}"),
        ("truncations", f"{tiny.truncations}", f"{usa.truncations}"),
        ("unfinished games", f"{tiny.unfinished_games}", f"{usa.unfinished_games}"),
    ]

    print()
    print("=" * 56)
    print(f"{'metric':<22}{'Tiny-Heat':>16}{'USA':>16}")
    print("-" * 56)
    for name, tv, uv in rows:
        print(f"{name:<22}{tv:>16}{uv:>16}")
    print("-" * 56)
    print(f"{'Tiny/USA games-per-sec ratio':<38}{ratio:>16.2f}")
    print("=" * 56)

    ok = True
    for s, lbl in ((tiny, "Tiny-Heat"), (usa, "USA")):
        if s.illegal_actions:
            print(f"FAIL [{lbl}]: {s.illegal_actions} illegal action(s)")
            ok = False
        if s.truncations:
            print(f"FAIL [{lbl}]: {s.truncations} truncation(s) at MAX_ROUNDS")
            ok = False
        if s.unfinished_games:
            print(f"FAIL [{lbl}]: {s.unfinished_games} game(s) did not finish")
            ok = False
    return ok


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny-Heat sanity + throughput harness.")
    p.add_argument("--games", type=int, default=200,
                   help="Self-play games per track (default: 200).")
    p.add_argument("--players", type=int, default=2,
                   help="Seats per game (default: 2).")
    p.add_argument("--random", action="store_true",
                   help="Use RandomAgent opponents instead of HeuristicAgent.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for reproducibility (default: 0).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)

    def opp_factory() -> object:
        if args.random:
            return lambda: RandomAgent(seed=args.seed)
        return lambda: HeuristicAgent()

    def tiny_factory() -> HeatEnv:
        return make_tiny_env(num_players=args.players, opponents=opp_factory())

    def usa_factory() -> HeatEnv:
        return HeatEnv(
            track=None,  # default USA
            num_players=args.players,
            opponents=opp_factory(),  # type: ignore[arg-type]
            randomize_seat=True,
        )

    print("=" * 56)
    print("Tiny-Heat sanity + throughput harness (A1)")
    print("=" * 56)
    print(
        f"games={args.games} players={args.players} "
        f"opponent={'random' if args.random else 'heuristic'} seed={args.seed}"
    )

    tiny = _run_track("Tiny-Heat", tiny_factory, args.games, rng)
    usa = _run_track("USA", usa_factory, args.games, rng)

    ok = _report(tiny, usa)
    if not ok:
        print("\nSANITY FAILED: correctness asserts violated.")
        return 1
    print("\nSANITY PASSED: all games legal, terminating, no truncation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
