"""Sprint B0 fidelity harness -- does the reduced model predict the true engine?

THE go/no-go gate for Option B. We draw ``N`` ``(GameState, player)`` snapshots
from real games (HeuristicAgent *and* the S1 LookaheadAgent, on the held-out
generated tight-corner band so limit-1 is well represented), and for each
snapshot and each candidate ``(gear, intent)`` we compare:

  * the **reduced prediction** -- ``step_reduced`` over ``(pos, heat, gear)`` with
    a ``SpeedResourceModel`` built from the live deck composition, and
  * the **true outcome** -- clone the live ``GameState``, force *that exact gear*
    and the *intent's card-selection rule* (the same rule the model assumes), run
    exactly ONE turn via ``run_round_driver``, and read the true
    ``next_pos / next_heat / spun`` off the resulting state.

The forced action in the true rollout MUST match the action the model predicted,
or the comparison is meaningless -- so the card play is chosen by the SAME
intent->cards rule (:func:`_intent_card_play`) the resource model's order
statistics encode (conserve = play your lowest cards, push = your highest).

Metrics are bucketed by the speed limit of the corner the turn crosses (limit-1
is the headline), reporting median + p90 absolute error in ``next_heat`` and
``next_pos`` and the predicted-vs-actual **spin confusion** (precision/recall).
A one-line VERDICT per fidelity mode is printed against the Rung-0 bar:

  * median abs end-of-turn heat error <= 1 (overall AND limit-1),
  * spin-agreement >= ~90% (limit-1),
  * next_pos median error <= 1.

This harness does NOT tune the model. A FAIL is a valid, valuable outcome.

Usage:
    python experiments/spike_reduced_model.py --n 50
    python experiments/spike_reduced_model.py --n 200 --mode banded --seed 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow `from _runlog import run_main` and src-layout imports when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.search_agent import LookaheadAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind, run_round_driver
from heat.engine.game import Game
from heat.engine.phases import ReactDecision
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.opponents import opponent_action
from heat.planning.reduced_model import (
    MODES,
    ReducedAction,
    ReducedState,
    step_reduced,
)
from heat.planning.resource_model import INTENTS, SpeedResourceModel
from heat.tracks.generator import TrackGenParams, generate_track


# Reuse the shared held-out eval band (tight corners -> limit-1 well represented).
_HELDOUT_BASE = 900_000
_TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),
    laps=2,
)


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """One captured (state, player) decision point at a turn start.

    ``state`` is a clone of the live game taken at the GEAR decision for
    ``player_id`` (before any action is applied), so it is a faithful resume
    point for forcing a candidate turn. ``track`` is shared (immutable).
    """

    state: GameState
    player_id: int


def _capture_snapshots(
    make_agent,
    *,
    n_target: int,
    track_seeds: list[int],
    game_seed_base: int,
) -> list[Snapshot]:
    """Run games with ``make_agent`` in seat 0 and clone the GEAR-decision state.

    Drives ``run_round_driver`` directly so we can intercept the seat-0 GEAR
    decision and snapshot the live state *before* the action is applied (the
    exact resume point the fidelity check forces candidates from). The seat-0
    agent's actual choices are then applied to continue the game forward, so the
    snapshots trace a realistic trajectory (corners, recovery states). Returns up
    to ``n_target`` snapshots.
    """
    snaps: list[Snapshot] = []
    for gi, tseed in enumerate(track_seeds):
        if len(snaps) >= n_target:
            break
        track = generate_track(tseed, _TIGHT_PARAMS)
        agent = make_agent()
        state = GameState.create(
            track, num_players=1, logging_enabled=True,
            seed=game_seed_base + gi,
        )
        _drive_game_capturing(state, agent, snaps, n_target)
    return snaps[:n_target]


def _drive_game_capturing(
    state: GameState,
    agent: BaseAgent,
    snaps: list[Snapshot],
    n_target: int,
) -> None:
    """Drive one solo game forward, snapshotting each seat-0 GEAR decision."""
    guard = 0
    while not state.is_game_over and guard < 400:
        guard += 1
        gen = run_round_driver(state)
        try:
            decision = next(gen)
            while True:
                if (
                    decision.kind == DecisionKind.GEAR
                    and decision.player_id == 0
                    and len(snaps) < n_target
                ):
                    # Snapshot BEFORE applying the action: a faithful resume
                    # point. Clone with copy_event_log=False (we re-run one turn).
                    snaps.append(Snapshot(state.clone(reseed=12345 + len(snaps)), 0))
                action = opponent_action(agent, decision, state)
                decision = gen.send(action)
        except StopIteration:
            pass
        if len(snaps) >= n_target:
            return


# ---------------------------------------------------------------------------
# Intent -> card-play rule (MUST match the resource model's order statistics)
# ---------------------------------------------------------------------------


def _intent_card_play(
    hand: list[Card], gear: int, intent: str
) -> tuple[Card, ...]:
    """Pick the ``gear``-card play matching ``intent`` (the model's assumption).

    The resource model conditions speed on intent by order statistics over the
    owned Basics: ``conserve`` plays the driver's *lowest* cards, ``push`` the
    *highest*, ``mean`` the middle. The true-engine rollout MUST force the same
    rule, or the prediction and the truth describe different actions. We rank the
    legal plays by total card value and pick the lowest (conserve), highest
    (push), or median (mean) play.

    Returns a legal card tuple from ``rules.legal_card_plays`` (so the engine
    accepts it). When the hand is cluttered (fewer playable than gear) there is
    one forced play and intent is moot.
    """
    legal = rules.legal_card_plays(hand, gear)
    if len(legal) == 1:
        return legal[0]
    # Rank by total played-card value (stress counts as 0 here, matching
    # calculate_speed; the engine resolves stress to a basic at reveal, which the
    # model already folds in as expected value -- consistent treatment).
    ranked = sorted(legal, key=lambda play: (sum(c.value for c in play),))
    if intent == "conserve":
        return ranked[0]
    if intent == "push":
        return ranked[-1]
    return ranked[len(ranked) // 2]  # mean -> median-value play


# ---------------------------------------------------------------------------
# True-engine one-turn rollout for a forced (gear, intent)
# ---------------------------------------------------------------------------


@dataclass
class TrueOutcome:
    """Ground-truth one-turn outcome read off the engine after a forced action."""

    next_pos: int
    next_heat: int
    spun: bool
    crossed_limits: tuple[int, ...]  # speed limits of corners crossed this turn


def _run_one_true_turn(
    snap_state: GameState,
    player_id: int,
    target_gear: int,
    intent: str,
    rollout_policy: BaseAgent,
) -> TrueOutcome | None:
    """Clone the snapshot, force ``(target_gear, intent-cards)``, run ONE turn.

    Returns the true ``next_pos / next_heat / spun`` and the limits of corners
    crossed (for bucketing), or ``None`` if the forced gear is not legal for this
    player (the candidate is simply not evaluated -- the reduced model is asked
    the same legal-gear question via ``_shift_gear``, but we only compare on
    candidates the engine can actually execute, so the buckets stay honest).

    Every decision other than the forced seat-0 GEAR/CARDS is answered by
    ``rollout_policy`` (the same heuristic the snapshot agent and the reduced
    model's REACT/etc. assume), so the only thing that varies is the forced move.
    """
    clone = snap_state.clone(reseed=777)
    clone.logging_enabled = True
    player = clone.get_player(player_id)

    legal_gears = rules.legal_gear_shifts(player.gear, player.heat_available)
    forced_gear = next((g for g in legal_gears if g[0] == target_gear), None)
    if forced_gear is None:
        return None

    forced_cards: tuple[Card, ...] | None = None
    start_pos = player.position
    start_lap = player.lap

    gen = run_round_driver(clone)
    try:
        decision = next(gen)
        while True:
            action = _answer(
                clone, decision, player_id, forced_gear,
                target_gear, intent, rollout_policy,
            )
            decision = gen.send(action)
    except StopIteration:
        pass

    player = clone.get_player(player_id)
    spun = any(
        e.event_type == "spin_out" and e.player_id == player_id
        and e.round_num == snap_state.round_num
        for e in clone.event_log
    )
    # Limits of corners physically crossed this turn (from start to end pos).
    crossed = _crossed_corner_limits(
        clone.track, start_pos, start_lap, player, spun, clone.event_log,
        player_id, snap_state.round_num,
    )
    return TrueOutcome(
        next_pos=player.position,
        next_heat=player.heat_available,
        spun=spun,
        crossed_limits=crossed,
    )


def _answer(
    clone: GameState,
    decision: Decision,
    player_id: int,
    forced_gear: tuple[int, int],
    target_gear: int,
    intent: str,
    rollout_policy: BaseAgent,
) -> object:
    """Answer a driver decision: force seat-0 GEAR/CARDS, else delegate."""
    if decision.player_id == player_id:
        if decision.kind == DecisionKind.GEAR:
            if forced_gear in decision.legal:  # type: ignore[operator]
                return forced_gear
        elif decision.kind == DecisionKind.CARDS:
            player = clone.get_player(player_id)
            play = _intent_card_play(player.hand, player.gear, intent)
            if play in decision.legal:  # type: ignore[operator]
                return play
    return opponent_action(rollout_policy, decision, clone)


def _crossed_corner_limits(
    track: Track,
    start_pos: int,
    start_lap: int,
    player,
    spun: bool,
    event_log: list,
    player_id: int,
    round_num: int,
) -> tuple[int, ...]:
    """Speed limits of corners this player crossed this turn (for bucketing).

    On a clean turn, reconstruct from start->end position (lap-aware). On a spin,
    the car was reset backward, so the crossed corner is the one logged in the
    ``spin_out`` event (``corner_start``) -- read it directly.
    """
    if spun:
        for e in event_log:
            if (
                e.event_type == "spin_out"
                and e.player_id == player_id
                and e.round_num == round_num
            ):
                cstart = int(e.data.get("corner_start", -1))
                for c in track.corners:
                    if c.start == cstart:
                        return (c.speed_limit,)
        return ()
    spaces = (player.lap - start_lap) * track.length + (player.position - start_pos)
    if spaces <= 0:
        return ()
    crossed = rules.corners_crossed(start_pos, player.position, track, spaces_moved=spaces)
    return tuple(c.speed_limit for c in crossed)


# ---------------------------------------------------------------------------
# Reduced prediction for a forced (gear, intent)
# ---------------------------------------------------------------------------


def _predict_reduced(
    snap_state: GameState,
    player_id: int,
    target_gear: int,
    intent: str,
    mode: str,
):
    """Reduced ``step_reduced`` prediction for the same forced ``(gear, intent)``.

    Builds the resource model from the live deck composition and steps the
    reduced state. ``dgear`` is the delta from the player's current gear to the
    forced ``target_gear`` (``step_reduced`` re-validates it through the real
    ``legal_gear_shifts``, matching the engine path).
    """
    player = snap_state.get_player(player_id)
    model = SpeedResourceModel.from_player(player)
    rstate = ReducedState(
        pos=player.position, heat=player.heat_available, gear=player.gear
    )
    action = ReducedAction(dgear=target_gear - player.gear, intent=intent)
    return step_reduced(rstate, action, snap_state.track, model, mode=mode)


# ---------------------------------------------------------------------------
# Error accumulation + bucketing
# ---------------------------------------------------------------------------

#: Spin-decision criteria evaluated side by side. "point" = the model's single
#: point-speed ``spun`` flag (the original, mode-insensitive criterion); the
#: ``p>tau`` entries are the tail-aware criteria that use the marginal ``p_spin``
#: the banded/empirical models compute -- the fix that gives those modes a fair
#: spin-agreement test. Thresholds are fixed and principled (0.5 = "more likely
#: than not"; 0.3 = a risk-averse planner), NOT tuned to clear the bar.
_SPIN_CRITERIA: tuple[str, ...] = ("point", "p>0.5", "p>0.3")


@dataclass
class _Bucket:
    """Per-limit error accumulator for one fidelity mode.

    Stores the raw per-sample spin data (the model's point ``spun`` flag, its
    marginal ``p_spin``, and the true engine ``spun``) rather than pre-tallied
    confusion counts, so spin-agreement can be recomputed under any spin-decision
    *criterion* -- the point flag OR a tail-aware ``p_spin > tau`` threshold. This
    is the B0 fix: the point ``spun`` is derived from a single point-speed and is
    near-identical across modes, so it never exercises the banded/empirical
    models' actual risk signal (``p_spin``); evaluating ``p_spin > tau`` gives
    those modes the fair test the Rung-0 spin-agreement bar intends.
    """

    heat_errs: list[int] = field(default_factory=list)
    pos_errs: list[int] = field(default_factory=list)
    point_preds: list[bool] = field(default_factory=list)  # model point spun flag
    p_spins: list[float] = field(default_factory=list)      # model marginal P(spin)
    act_spins: list[bool] = field(default_factory=list)     # true engine spun

    def add(
        self,
        heat_err: int,
        pos_err: int,
        point_pred: bool,
        p_spin: float,
        act_spin: bool,
    ) -> None:
        self.heat_errs.append(heat_err)
        self.pos_errs.append(pos_err)
        self.point_preds.append(point_pred)
        self.p_spins.append(p_spin)
        self.act_spins.append(act_spin)

    @property
    def n(self) -> int:
        return len(self.heat_errs)

    def _preds_for(self, criterion: str) -> list[bool]:
        """Binary spin predictions under ``criterion`` ('point' | 'p>0.5' | 'p>0.3')."""
        if criterion == "point":
            return self.point_preds
        tau = float(criterion.split(">")[1])
        return [p > tau for p in self.p_spins]

    def _confusion(self, criterion: str) -> tuple[int, int, int, int]:
        tp = fp = fn = tn = 0
        for pred, act in zip(self._preds_for(criterion), self.act_spins):
            if pred and act:
                tp += 1
            elif pred and not act:
                fp += 1
            elif not pred and act:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    def agreement(self, criterion: str = "point") -> float:
        tp, fp, fn, tn = self._confusion(criterion)
        total = tp + fp + fn + tn
        return (tp + tn) / total if total else float("nan")

    def spin_precision(self, criterion: str = "point") -> float:
        tp, fp, _, _ = self._confusion(criterion)
        denom = tp + fp
        return tp / denom if denom else float("nan")

    def spin_recall(self, criterion: str = "point") -> float:
        tp, _, fn, _ = self._confusion(criterion)
        denom = tp + fn
        return tp / denom if denom else float("nan")

    def best_criterion(self) -> tuple[str, float]:
        """Return the (criterion, agreement) with the highest spin-agreement.

        Used for the per-mode verdict: a mode's spin-agreement PASSES if ANY
        principled criterion (point or a fixed p_spin threshold) clears the bar,
        per the B0 doc's 'report all three so we pick the cheapest that passes'.
        """
        cands = [(c, self.agreement(c)) for c in _SPIN_CRITERIA]
        cands = [(c, a) for c, a in cands if a == a]  # drop NaN
        if not cands:
            return ("point", float("nan"))
        return max(cands, key=lambda ca: ca[1])

    def calibration(self) -> list[tuple[str, int, float]]:
        """Actual spin rate bucketed by predicted ``p_spin`` band (is p_spin informative?).

        Returns ``[(band_label, n, actual_spin_rate), ...]`` -- if ``p_spin`` is
        discriminative, the actual spin rate rises monotonically across bands.
        """
        bands = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        out: list[tuple[str, int, float]] = []
        for lo, hi in bands:
            idx = [i for i, p in enumerate(self.p_spins) if lo <= p < hi]
            if not idx:
                continue
            rate = sum(self.act_spins[i] for i in idx) / len(idx)
            out.append((f"[{lo:.2f},{hi:.2f})", len(idx), rate))
        return out


def _median(xs: list[int]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _p90(xs: list[int]) -> float:
    if not xs:
        return float("nan")
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))
    return float(ordered[idx])


# ---------------------------------------------------------------------------
# The comparison sweep
# ---------------------------------------------------------------------------


def _evaluate_mode(
    snaps: list[Snapshot],
    mode: str,
) -> tuple[dict[int, _Bucket], _Bucket]:
    """Run the apples-to-apples comparison for one fidelity ``mode``.

    For every snapshot, every legal candidate gear (1..4) and every intent, pair
    the reduced prediction against the true one-turn engine outcome and record
    the heat/pos error and spin confusion, bucketed by the limit of the corner
    crossed. Candidates that cross no corner contribute to heat/pos error but to
    an "open" (limit 99) spin bucket only when relevant -- we bucket every paired
    sample by each crossed limit (a turn crossing two corners contributes to both
    limits' buckets), and also accumulate an ``overall`` bucket.

    Returns ``(by_limit, overall)``.
    """
    by_limit: dict[int, _Bucket] = {}
    overall = _Bucket()
    rollout = HeuristicAgent(name="RolloutHeuristic")

    for snap in snaps:
        player = snap.state.get_player(snap.player_id)
        cur_gear = player.gear
        # Candidate gears the engine could legally shift to from here.
        legal_targets = sorted({g for g, _ in rules.legal_gear_shifts(
            cur_gear, player.heat_available
        )})
        for target_gear in legal_targets:
            for intent in INTENTS:
                true = _run_one_true_turn(
                    snap.state, snap.player_id, target_gear, intent, rollout
                )
                if true is None:
                    continue
                pred = _predict_reduced(
                    snap.state, snap.player_id, target_gear, intent, mode
                )
                heat_err = abs(pred.next_heat - true.next_heat)
                # Position error: compare modulo track length so a lap-wrap in
                # one but not the other does not register a spurious full-lap gap.
                length = snap.state.track.length
                pos_err = abs(pred.next_pos - true.next_pos) % length
                pos_err = min(pos_err, length - pos_err)
                act_spin = true.spun

                overall.add(heat_err, pos_err, pred.spun, pred.p_spin, act_spin)
                limits = true.crossed_limits or ()
                for lim in limits:
                    by_limit.setdefault(lim, _Bucket()).add(
                        heat_err, pos_err, pred.spun, pred.p_spin, act_spin
                    )
    return by_limit, overall


# ---------------------------------------------------------------------------
# Reporting + verdict
# ---------------------------------------------------------------------------


def _print_mode_report(
    mode: str, by_limit: dict[int, _Bucket], overall: _Bucket
) -> bool:
    """Print the per-bucket error table for one mode and return its PASS/FAIL."""
    print(f"\n=== fidelity mode: {mode} ===")
    header = (
        f"{'bucket':<10}{'n':>6}{'heat med':>10}{'heat p90':>10}"
        f"{'pos med':>9}{'pos p90':>9}{'spin agree':>12}"
        f"{'spin P':>8}{'spin R':>8}"
    )
    print(header)
    print("-" * len(header))

    def row(label: str, b: _Bucket) -> None:
        print(
            f"{label:<10}{b.n:>6}"
            f"{_median(b.heat_errs):>10.2f}{_p90(b.heat_errs):>10.2f}"
            f"{_median(b.pos_errs):>9.2f}{_p90(b.pos_errs):>9.2f}"
            f"{b.agreement():>12.3f}"
            f"{b.spin_precision():>8.3f}{b.spin_recall():>8.3f}"
        )

    for lim in sorted(by_limit):
        row(f"L{lim}", by_limit[lim])
    row("overall", overall)

    l1 = by_limit.get(1)
    l1_missing = l1 is None or l1.n == 0

    # --- Tail-aware spin-agreement: the B0 fix. The per-bucket table above uses
    # the point criterion; here we re-score the headline L1 bucket under each
    # principled spin criterion so the banded/empirical models' p_spin signal is
    # actually exercised. (point/p>0.5/p>0.3 are identical for the heat/pos
    # columns -- only the spin call changes.)
    if not l1_missing:
        print("  limit-1 spin-agreement by spin-criterion (the Rung-0 headline):")
        for crit in _SPIN_CRITERIA:
            print(
                f"    {crit:<7} agree={l1.agreement(crit):.3f}  "
                f"P={l1.spin_precision(crit):.3f}  R={l1.spin_recall(crit):.3f}"
            )
        cal = l1.calibration()
        if cal:
            print("  limit-1 p_spin calibration (actual spin rate by predicted band):")
            for label, n, rate in cal:
                print(f"    p_spin {label}  n={n:>4}  actual spin rate={rate:.3f}")

    # --- Rung-0 verdict (spin uses the BEST principled criterion for this mode) ---
    best_crit, best_agree = ("point", float("nan")) if l1_missing else l1.best_criterion()
    heat_overall_ok = (
        overall.n > 0 and _median(overall.heat_errs) <= 1.0 + 1e-9
    )
    heat_l1_ok = l1 is not None and l1.n > 0 and _median(l1.heat_errs) <= 1.0 + 1e-9
    pos_overall_ok = overall.n > 0 and _median(overall.pos_errs) <= 1.0 + 1e-9
    spin_l1_ok = not l1_missing and best_agree >= 0.90 - 1e-9

    def mark(ok: bool, missing: bool = False) -> str:
        if missing:
            return "N/A (no limit-1 samples)"
        return "PASS" if ok else "FAIL"

    print("  Rung-0 checks:")
    print(f"    heat median <= 1 (overall):  {mark(heat_overall_ok)}  "
          f"(median={_median(overall.heat_errs):.2f})")
    print(f"    heat median <= 1 (limit-1):  {mark(heat_l1_ok, l1_missing)}"
          + ("" if l1_missing else f"  (median={_median(l1.heat_errs):.2f})"))
    print(f"    pos  median <= 1 (overall):  {mark(pos_overall_ok)}  "
          f"(median={_median(overall.pos_errs):.2f})")
    print(f"    spin-agreement >= 90% (L1):  {mark(spin_l1_ok, l1_missing)}"
          + ("" if l1_missing
             else f"  (best agree={best_agree:.3f} via '{best_crit}'; "
                  f"point={l1.agreement('point'):.3f})"))

    passed = (
        heat_overall_ok and heat_l1_ok and pos_overall_ok and spin_l1_ok
        and not l1_missing
    )
    print(f"  VERDICT [{mode}]: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50,
                        help="number of (state, player) snapshots to evaluate")
    parser.add_argument("--mode", type=str, default=None, choices=list(MODES),
                        help="single fidelity mode to run; default runs all three")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed for game RNG (track seeds are held-out)")
    parser.add_argument("--tracks", type=int, default=40,
                        help="held-out track seeds to draw snapshots across")
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.tracks)]
    modes = [args.mode] if args.mode else list(MODES)

    # Half the snapshots from the heuristic, half from the S1 search agent (R4:
    # test on the recovery states the search agent visits, not just heuristic
    # ones). Interleave so a short track set still mixes both sources.
    half = max(1, args.n // 2)

    def make_heuristic() -> BaseAgent:
        return HeuristicAgent()

    def make_lookahead() -> BaseAgent:
        return LookaheadAgent(horizon=2, n_determinizations=2, seed=7)

    print(f"spike_reduced_model: capturing snapshots "
          f"(target n={args.n}, tracks={len(track_seeds)}, seed={args.seed})")
    snaps_h = _capture_snapshots(
        make_heuristic, n_target=half,
        track_seeds=track_seeds, game_seed_base=args.seed,
    )
    snaps_l = _capture_snapshots(
        make_lookahead, n_target=args.n - len(snaps_h),
        track_seeds=track_seeds, game_seed_base=args.seed + 100,
    )
    snaps = snaps_h + snaps_l
    print(f"  captured {len(snaps)} snapshots "
          f"({len(snaps_h)} heuristic + {len(snaps_l)} lookahead)")
    if not snaps:
        print("  no snapshots captured -- aborting")
        return

    results: dict[str, bool] = {}
    for mode in modes:
        by_limit, overall = _evaluate_mode(snaps, mode)
        results[mode] = _print_mode_report(mode, by_limit, overall)

    print("\n=== B0 go/no-go summary ===")
    any_pass = False
    for mode in modes:
        verdict = "PASS" if results[mode] else "FAIL"
        print(f"  {mode:<10} -> {verdict}")
        any_pass = any_pass or results[mode]
    cheapest = next((m for m in MODES if m in results and results[m]), None)
    if any_pass:
        print(f"  GO: cheapest passing mode = {cheapest}")
    else:
        print("  NO-GO at this N: no fidelity mode meets the Rung-0 bar. "
              "Inspect the per-bucket table above for the dominant divergence "
              "(heat error vs limit-1 spin-agreement) before deciding to "
              "re-spike with a mitigation or fold into Option A.")


if __name__ == "__main__":
    from _runlog import run_main

    run_main("spike_reduced_model", main)
