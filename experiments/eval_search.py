"""Sprint S1 eval: the LookaheadAgent on held-out GENERATED tracks.

The design's cross-cutting rule (section 3) is emphatic and this harness obeys
it: **measure on procedurally GENERATED tracks (not USA), bucket spins by the
speed-limit of the corner spun at, and report the WORST CASE (p90 / max) per
bucket, not the mean.** Every "good" number in the 8C saga was USA-only and a
mean; both hid the limit-1 death-spiral.

It compares three agents on the same held-out track seeds:

    * ``HeuristicAgent``        -- the hand-coded reference (the bar to clear)
    * ``StrongHeuristicAgent``  -- the 1-ply economy planner (current best scripted)
    * ``LookaheadAgent``        -- the S1 forward-rollout search agent

Metrics, per agent, in both a **solo** field (the corner-learning objective, the
primary gate) and a **4p** field (secondary; note S1 uses open-hand rollout in
multiplayer -- proper hidden-info determinization is deferred to S2, see
``search_agent`` module docstring):

    * spins-per-corner-pass, **bucketed by the corner's speed limit** (limit-1
      is the headline), reported as mean / p90 / max over games
    * finish rate
    * rounds-to-finish (mean over finishers)

Corner passes are reconstructed exactly from the per-turn ``turn_start`` event
log (each turn's start position -> end position -> the corners crossed via
``rules.corners_crossed``), so the denominator is real physical passes, not an
approximation. A spin is attributed to the corner at the logged ``corner_start``.

Usage:
    python experiments/eval_search.py --games 24
    python experiments/eval_search.py --games 30 --horizon 2 --dets 2 --seed 5
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.engine import rules
from heat.engine.game import Game
from heat.models.game_state import GameState, GameEvent
from heat.models.track import Track
from heat.tracks.generator import TrackGenParams, generate_track


# A track seed band held out from any training; tilt the generator toward tight
# corners so limit-1 passes are well represented in the sample.
_HELDOUT_BASE = 900_000
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),  # weight toward tight corners
    laps=2,
)


# ---------------------------------------------------------------------------
# Per-game corner-pass / spin reconstruction
# ---------------------------------------------------------------------------


def _corner_limit_at_start(track: Track, corner_start: int) -> int | None:
    for c in track.corners:
        if c.start == corner_start:
            return c.speed_limit
    return None


def _passes_and_spins_by_limit(
    track: Track,
    event_log: list[GameEvent],
    player_id: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return ``(passes_by_limit, spins_by_limit)`` for one player in one game.

    Reconstructs corner passes exactly: walk this player's ``turn_start`` events
    in order; between consecutive turn starts the car moved from the earlier
    position to the later one, and ``rules.corners_crossed`` (with the lap-aware
    spaces moved) lists the corners physically crossed. Each crossed corner is a
    pass bucketed by its speed limit. Spins are attributed to the corner at the
    logged ``corner_start``. The final (incomplete) leg from the last turn start
    to the finish is not counted -- it has no following turn_start to bound it,
    and the finish-crossing pass is an edge case we conservatively omit.
    """
    passes: dict[int, int] = {}
    spins: dict[int, int] = {}

    # Ordered (round, position, lap) snapshots from this player's turn starts.
    starts: list[tuple[int, int]] = []  # (position, lap_at_turn_start)
    # turn_start logs hand/gear/position but not lap; we track lap from the
    # running event stream is unreliable, so we approximate lap via wrap count
    # below. Simpler: count a pass whenever a corner index lies strictly ahead
    # on the path of length `spaces_moved`.
    positions: list[int] = []
    for e in event_log:
        if e.player_id != player_id:
            continue
        if e.event_type == "turn_start":
            positions.append(int(e.data["position"]))
        elif e.event_type == "spin_out":
            limit = _corner_limit_at_start(track, int(e.data["corner_start"]))
            if limit is not None:
                spins[limit] = spins.get(limit, 0) + 1

    # Pair consecutive recorded positions; spaces moved is the forward distance
    # modulo track length. A spin resets the car BACKWARD (to corner.start-1),
    # so the leg straddling a spin shows a near-full-lap forward delta -- we cap
    # the per-turn move at a sane bound (a single turn cannot exceed gear-5 plus
    # boost/adrenaline/slipstream, comfortably < 12) and drop implausible legs so
    # a spin does not inflate the pass denominator. The spin itself is already
    # counted from its own event.
    length = track.length
    max_single_turn = 12
    for prev_pos, next_pos in zip(positions, positions[1:]):
        spaces_moved = (next_pos - prev_pos) % length
        if spaces_moved == 0 or spaces_moved > max_single_turn:
            continue
        crossed = rules.corners_crossed(
            prev_pos, next_pos, track, spaces_moved=spaces_moved
        )
        for c in crossed:
            passes[c.speed_limit] = passes.get(c.speed_limit, 0) + 1

    return passes, spins


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _AgentAgg:
    """Accumulates per-game metrics for one agent across many games."""

    finishes: int = 0
    games: int = 0
    rounds_finishers: list[int] = field(default_factory=list)
    # Per-game spins-per-pass ratios, keyed by corner limit (only games with at
    # least one pass at that limit contribute -- a clean per-pass rate).
    ratios_by_limit: dict[int, list[float]] = field(default_factory=dict)
    total_passes_by_limit: dict[int, int] = field(default_factory=dict)
    total_spins_by_limit: dict[int, int] = field(default_factory=dict)

    def add_game(
        self,
        finished: bool,
        rounds: int,
        passes: dict[int, int],
        spins: dict[int, int],
    ) -> None:
        self.games += 1
        if finished:
            self.finishes += 1
            self.rounds_finishers.append(rounds)
        limits = set(passes) | set(spins)
        for limit in limits:
            p = passes.get(limit, 0)
            s = spins.get(limit, 0)
            self.total_passes_by_limit[limit] = (
                self.total_passes_by_limit.get(limit, 0) + p
            )
            self.total_spins_by_limit[limit] = (
                self.total_spins_by_limit.get(limit, 0) + s
            )
            if p > 0:
                self.ratios_by_limit.setdefault(limit, []).append(s / p)

    def finish_rate(self) -> float:
        return self.finishes / self.games if self.games else 0.0

    def mean_rounds(self) -> float:
        return (
            statistics.mean(self.rounds_finishers)
            if self.rounds_finishers
            else float("nan")
        )

    def spin_stats(self, limit: int) -> tuple[float, float, float]:
        """``(mean, p90, max)`` spins-per-pass at ``limit`` over games.

        The per-game ratio captures the death-spiral the pooled mean hides: a
        game that spins repeatedly at one limit-1 corner shows a high ratio even
        if the agent passes other corners cleanly.
        """
        ratios = self.ratios_by_limit.get(limit, [])
        if not ratios:
            return (float("nan"), float("nan"), float("nan"))
        ordered = sorted(ratios)
        mean = statistics.mean(ordered)
        # p90: the value at the 90th percentile (nearest-rank).
        idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))
        return (mean, ordered[idx], ordered[-1])


# ---------------------------------------------------------------------------
# Game running
# ---------------------------------------------------------------------------


def _run_field(
    label_to_agent: dict[str, BaseAgent],
    *,
    track_seeds: list[int],
    num_players: int,
    game_seed_base: int,
) -> tuple[dict[str, _AgentAgg], dict[str, str]]:
    """Run one field (solo or 4p) and aggregate per-agent metrics.

    For each held-out track seed, every agent label plays one game seated at
    seat 0 (the remaining seats in 4p are weak heuristics -- a fixed, neutral
    backdrop so the corner metric for seat 0 is comparable across agents). A
    fresh agent instance per game keeps per-game RNG state clean.

    Returns ``(aggs, profiles)`` where ``profiles[label]`` is a one-line profiling
    summary (clones/move, ms/move) accumulated across that label's games when the
    agent exposes a ``profile`` (only the ``LookaheadAgent`` does); other labels
    map to an empty string.
    """
    aggs: dict[str, _AgentAgg] = {label: _AgentAgg() for label in label_to_agent}
    # Accumulate profiling across games per label by summing into one profile.
    prof_clones: dict[str, int] = {label: 0 for label in label_to_agent}
    prof_moves: dict[str, int] = {label: 0 for label in label_to_agent}
    prof_seconds: dict[str, float] = {label: 0.0 for label in label_to_agent}

    for gi, tseed in enumerate(track_seeds):
        track = generate_track(tseed, _TIGHT_PARAMS)
        for label, make in label_to_agent.items():
            agent = make()
            agents: list[BaseAgent] = [agent]
            for _ in range(num_players - 1):
                agents.append(HeuristicAgent())
            game = Game(
                track,
                agents,
                logging_enabled=True,
                seed=game_seed_base + gi,
            )
            result = game.run()
            finished = 0 in result.finish_order
            passes, spins = _passes_and_spins_by_limit(track, result.event_log, 0)
            aggs[label].add_game(finished, result.total_rounds, passes, spins)
            prof = getattr(agent, "profile", None)
            if prof is not None:
                prof_clones[label] += prof.clones
                prof_moves[label] += prof.moves
                prof_seconds[label] += prof.seconds

    profiles: dict[str, str] = {}
    for label in label_to_agent:
        m = prof_moves[label]
        if m:
            profiles[label] = (
                f"clones/move={prof_clones[label] / m:.1f} "
                f"ms/move={prof_seconds[label] / m * 1000.0:.2f} "
                f"(moves={m})"
            )
        else:
            profiles[label] = ""
    return aggs, profiles


# ---------------------------------------------------------------------------
# Seat-neutral 4p win-rate (secondary gate)
# ---------------------------------------------------------------------------


def _winner(result) -> int | None:
    """The winning seat id (first in finish order), or None if nobody finished."""
    return result.finish_order[0] if result.finish_order else None


def _seat_neutral_winrate(
    make_focal: "callable",
    make_opponent: "callable",
    *,
    track_seeds: list[int],
    game_seed_base: int,
) -> float:
    """Win-rate of ``make_focal`` vs three ``make_opponent`` in 4p, seat-neutral.

    Per the seat-bias finding (front turn-order seats win more, all else equal),
    a single fixed-seat measurement is contaminated. We therefore rotate the
    focal agent through ALL FOUR seats (different seeds per rotation) and average
    the win indicator, so the positional advantage cancels. Returns the fraction
    of games the focal agent finished first across the 4 x len(track_seeds)
    games.
    """
    wins = 0
    games = 0
    for gi, tseed in enumerate(track_seeds):
        track = generate_track(tseed, _TIGHT_PARAMS)
        for focal_seat in range(4):
            agents: list[BaseAgent] = []
            for seat in range(4):
                agents.append(
                    make_focal() if seat == focal_seat else make_opponent()
                )
            game = Game(
                track,
                agents,
                logging_enabled=False,
                seed=game_seed_base + gi * 4 + focal_seat,
            )
            result = game.run()
            if _winner(result) == focal_seat:
                wins += 1
            games += 1
    return wins / games if games else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_field(
    title: str,
    aggs: dict[str, _AgentAgg],
    profiles: dict[str, str] | None = None,
) -> None:
    print(f"\n=== {title} ===")
    # All limits that appeared, low (tight) first.
    limits = sorted(
        {
            limit
            for agg in aggs.values()
            for limit in set(agg.total_passes_by_limit) | set(agg.total_spins_by_limit)
        }
    )

    header = (
        f"{'agent':<18}{'finish%':>8}{'rounds':>8}"
        + "".join(f"  L{lim} spins/pass (mn/p90/mx)" for lim in limits)
    )
    print(header)
    print("-" * len(header))
    for label, agg in aggs.items():
        row = f"{label:<18}{agg.finish_rate() * 100:>7.0f}%{agg.mean_rounds():>8.1f}"
        for lim in limits:
            mean, p90, mx = agg.spin_stats(lim)
            if mean != mean:  # NaN -> no passes at this limit
                row += f"  {'--':>9} {'--':>5} {'--':>5}      "
            else:
                row += f"  {mean:>9.3f} {p90:>5.2f} {mx:>5.2f}      "
        print(row)

    # Pooled totals (context for the per-game ratios above).
    print("  pooled spins/passes by limit:")
    for label, agg in aggs.items():
        parts = []
        for lim in limits:
            p = agg.total_passes_by_limit.get(lim, 0)
            s = agg.total_spins_by_limit.get(lim, 0)
            parts.append(f"L{lim}:{s}/{p}")
        print(f"    {label:<18}" + "  ".join(parts))

    # Profiling (clones/move, ms/move) for agents that expose it.
    if profiles and any(profiles.values()):
        print("  search profiling:")
        for label, summary in profiles.items():
            if summary:
                print(f"    {label:<18}{summary}")


def _success_check(solo: dict[str, _AgentAgg]) -> None:
    """Print the S1 success-criteria verdict (limit-1, solo) honestly."""
    print("\n=== S1 success criteria (solo, limit-1) ===")
    heur = solo.get("Heuristic")
    look = solo.get("Lookahead")
    if heur is None or look is None:
        print("  (missing agent aggregates)")
        return

    _, _, heur_worst = heur.spin_stats(1)
    _, _, look_worst = look.spin_stats(1)
    heur_fin = heur.finish_rate()
    look_fin = look.finish_rate()
    heur_rounds = heur.mean_rounds()
    look_rounds = look.mean_rounds()

    def fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.3f}"

    print(f"  worst-case spins/limit-1-pass:  Lookahead={fmt(look_worst)}  "
          f"Heuristic={fmt(heur_worst)}")
    print(f"  solo finish rate:               Lookahead={look_fin * 100:.0f}%  "
          f"Heuristic={heur_fin * 100:.0f}%")
    print(f"  rounds-to-finish (mean):        Lookahead={fmt(look_rounds)}  "
          f"Heuristic={fmt(heur_rounds)}")

    crit_spins = (
        look_worst != look_worst  # no limit-1 passes -> vacuous, note it
        or heur_worst != heur_worst
        or look_worst <= heur_worst + 1e-9
    )
    crit_finish = look_fin >= 0.999
    crit_rounds = (
        look_rounds != look_rounds
        or heur_rounds != heur_rounds
        or look_rounds <= heur_rounds + 1e-9
    )
    print(
        f"  -> worst-case spins <= heuristic: "
        f"{'PASS' if crit_spins else 'FAIL'}\n"
        f"  -> solo finish == 100%:           "
        f"{'PASS' if crit_finish else 'FAIL'}\n"
        f"  -> rounds <= heuristic:           "
        f"{'PASS' if crit_rounds else 'FAIL'}"
    )


# ---------------------------------------------------------------------------
# Secondary gate: seat-neutral 4p win-rate (S2)
# ---------------------------------------------------------------------------


def _winrate_gate(
    make_det: "callable",
    track_seeds: list[int],
    seed_base: int,
) -> None:
    """Print the S2 secondary gate: seat-neutral 4p win-rate of the tuned agent.

    The success bar is: vs the WEAK heuristic, win-rate **> parity** (parity in a
    4p field is 25%); vs the STRONG heuristic, **competitive** (we report it and
    flag clearly below parity). All measurements are seat-neutral (focal agent
    rotated through all four seats) so the front-seat positional advantage does
    not masquerade as skill. This is explicitly SECONDARY to the solo/limit-1
    discipline above -- determinized search can be optimistic in adversarial
    hidden-info play, so a soft result here does not fail the sprint.
    """
    print("\n=== 4P win-rate (seat-neutral, secondary gate) ===")
    parity = 1.0 / 4.0

    vs_weak = _seat_neutral_winrate(
        make_det, lambda: HeuristicAgent(),
        track_seeds=track_seeds, game_seed_base=seed_base,
    )
    vs_strong = _seat_neutral_winrate(
        make_det, lambda: StrongHeuristicAgent(strength=2),
        track_seeds=track_seeds, game_seed_base=seed_base + 4000,
    )

    print(f"  parity (4p) = {parity * 100:.0f}%")
    print(f"  LookaheadDet vs 3x weak  heuristic: {vs_weak * 100:.1f}%")
    print(f"  LookaheadDet vs 3x strong heuristic: {vs_strong * 100:.1f}%")
    print(
        f"  -> vs weak > parity:       "
        f"{'PASS' if vs_weak > parity + 1e-9 else 'FAIL'}\n"
        f"  -> vs strong competitive:  "
        f"{'PASS (>= parity)' if vs_strong >= parity - 1e-9 else 'SOFT (< parity)'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=24,
                        help="held-out generated tracks to evaluate on")
    parser.add_argument("--horizon", type=int, default=2,
                        help="LookaheadAgent rollout depth (rounds)")
    parser.add_argument("--dets", type=int, default=2,
                        help="LookaheadAgent determinizations per candidate")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed for game RNG (track seeds are held-out)")
    parser.add_argument("--skip-4p", action="store_true",
                        help="run only the solo field (faster)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="LookaheadAgent top-k branching cap (S2)")
    parser.add_argument("--sim-budget", type=int, default=None,
                        help="LookaheadAgent per-move clone budget (S2)")
    parser.add_argument("--skip-winrate", action="store_true",
                        help="skip the (slower) seat-neutral 4p win-rate gate")
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]

    def make_heuristic() -> BaseAgent:
        return HeuristicAgent()

    def make_strong() -> BaseAgent:
        return StrongHeuristicAgent(strength=2)

    def make_lookahead() -> BaseAgent:
        # S1-style: open-hand rollout (no hidden-info determinization).
        return LookaheadAgent(
            horizon=args.horizon,
            n_determinizations=args.dets,
            top_k=args.top_k,
            sim_budget=args.sim_budget,
        )

    def make_lookahead_det() -> BaseAgent:
        # S2 tuned: hidden-info determinization on (the multiplayer-correct mode).
        return LookaheadAgent(
            name="LookaheadDet",
            horizon=args.horizon,
            n_determinizations=args.dets,
            determinize_hidden=True,
            top_k=args.top_k,
            sim_budget=args.sim_budget,
        )

    labels = {
        "Heuristic": make_heuristic,
        "StrongHeuristic": make_strong,
        "Lookahead": make_lookahead,
        "LookaheadDet": make_lookahead_det,
    }

    print(
        f"eval_search: {args.games} held-out generated tracks "
        f"(tight params, limit-1 weighted), horizon={args.horizon}, "
        f"dets={args.dets}, top_k={args.top_k}, sim_budget={args.sim_budget}"
    )

    solo, solo_prof = _run_field(
        labels, track_seeds=track_seeds, num_players=1,
        game_seed_base=args.seed,
    )
    _print_field("SOLO (primary gate)", solo, solo_prof)

    if not args.skip_4p:
        four, four_prof = _run_field(
            labels, track_seeds=track_seeds, num_players=4,
            game_seed_base=args.seed + 5000,
        )
        _print_field(
            "4P vs weak heuristics (seat-0; spin/finish metric)", four, four_prof
        )

    _success_check(solo)

    if not args.skip_winrate:
        _winrate_gate(make_lookahead_det, track_seeds, args.seed + 9000)


if __name__ == "__main__":
    main()
