"""Sprint C3 eval: the rung-4 gate -- the AZ net BEATS ``LookaheadAgent``.

This is the C3 behavioral gate (README S5 rung 4). It evaluates a trained AZ
checkpoint on the held-out solo field **two ways**:

  * **in-search** -- the net as the prior+value of the C1 ``MCTSAgent`` (the
    full Option-C agent);
  * **net-only** -- the net as a plain ``MLAgent`` (masked arg-max, no search);
    the "fast bot" payoff (how much skill distilled into the net itself).

and reports, against ``LookaheadAgent`` (rung 1) and ``HeuristicAgent`` (rung 0):

  * worst-case (max) L1 spins/pass, with a **bootstrap CI over the held-out
    tracks** (NOT a point estimate -- the S3/8C anti-fake-pass discipline);
  * rounds-to-finish (mean over finishers), with a bootstrap CI;
  * 100% solo finish;
  * heat-efficiency (dist/heat, cooldowns/game) -- the *budgeting* check.

The rung-4 verdict is BEAT iff, on the bootstrap lower-confidence comparison, the
AZ net (in-search) has strictly lower worst-case L1 spins/pass AND strictly lower
rounds-to-finish than ``LookaheadAgent`` AND 100% finish. A plateau (match but not
beat) is reported HONESTLY as the C3 process success, never papered over with
finish-rate.

This module imports the ``eval_search`` machinery VERBATIM (the
``_passes_and_spins_by_limit`` / ``_AgentAgg`` / ``_heat_efficiency`` /
``_run_field`` / ``_HELDOUT_BASE`` / ``_TIGHT_PARAMS`` symbols -- one source of
truth, never a fork), exactly as ``eval_mcts.py`` does, and adds only the AZ
contenders + the bootstrap-CI gate + the rung-4 verdict.

Usage:
    python experiments/eval_az.py --games 24 --sims 16 --model checkpoints/c3_best.zip
    python experiments/eval_az.py --games 24 --model checkpoints/c3_best.zip --net-only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

# Import the eval_search machinery VERBATIM (single source of truth).
sys.path.insert(0, os.path.dirname(__file__))
from eval_search import (  # noqa: E402
    _HELDOUT_BASE,
    _AgentAgg,
    _print_field,
    _run_field,
)

from heat.agents.base import BaseAgent  # noqa: E402
from heat.agents.heuristic_agent import HeuristicAgent  # noqa: E402
from heat.agents.search_agent import LookaheadAgent  # noqa: E402
from heat.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: E402
from heat.agents.ml_agent import MLAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Bootstrap CI over the held-out tracks (the anti-fake-pass discipline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    """A bootstrap confidence interval over a per-game statistic.

    ``point`` is the statistic on the full sample; ``lo``/``hi`` are the
    ``alpha/2`` and ``1-alpha/2`` percentiles of the bootstrap-resampled
    statistic. ``n`` is the sample size. A NaN point (no eligible games) yields
    a NaN interval -- inert, never a pass.
    """

    point: float
    lo: float
    hi: float
    n: int

    def fmt(self) -> str:
        if self.point != self.point:
            return "n/a"
        return f"{self.point:.3f} [{self.lo:.3f},{self.hi:.3f}]"


def _bootstrap_ci(
    values: list[float],
    stat: str,
    *,
    n_boot: int = 2000,
    alpha: float = 0.10,
    seed: int = 0,
) -> CI:
    """Bootstrap CI of ``stat`` ('max' or 'mean') over per-game ``values``.

    Resamples ``values`` with replacement ``n_boot`` times, recomputes ``stat``
    on each resample, and returns the ``[alpha/2, 1-alpha/2]`` percentile interval
    (default 90% CI). 'max' is the worst-case-spins axis -- its bootstrap captures
    how lucky the observed max was; 'mean' is the rounds-to-finish axis.
    """
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    arr = np.asarray(vals, dtype=np.float64)
    fn = np.max if stat == "max" else np.mean
    point = float(fn(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = fn(arr[idx])
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return CI(point, lo, hi, n)


def agg_spin_ci(agg: _AgentAgg, limit: int, *, seed: int = 0) -> CI:
    """Bootstrap CI of the worst-case (max) per-game spins/pass at ``limit``.

    The per-game ratios are exactly the ``_AgentAgg.ratios_by_limit`` the harness
    already records (only games with a pass at that limit contribute). Resampling
    them bounds how sample-dependent the headline max is -- a single unlucky game
    cannot fake a worse opponent, and a single lucky game cannot fake a pass.
    """
    ratios = agg.ratios_by_limit.get(limit, [])
    return _bootstrap_ci(list(ratios), "max", seed=seed)


def agg_rounds_ci(agg: _AgentAgg, *, seed: int = 1) -> CI:
    """Bootstrap CI of the mean rounds-to-finish over finisher games."""
    return _bootstrap_ci([float(r) for r in agg.rounds_finishers], "mean", seed=seed)


# ---------------------------------------------------------------------------
# Rung-4 verdict
# ---------------------------------------------------------------------------


def _beats_on_ci(cand: CI, base: CI) -> bool:
    """True iff ``cand`` strictly beats ``base`` with separated CIs (lower=better).

    A trustworthy "beat" is not a point-estimate edge: we require the candidate's
    upper CI bound to sit below the baseline's lower CI bound (the intervals do
    not overlap, candidate better). A NaN on either side is never a beat.
    """
    if cand.point != cand.point or base.point != base.point:
        return False
    return cand.hi < base.lo - 1e-9


def rung4_verdict(
    solo: dict[str, _AgentAgg],
    *,
    az_label: str,
    seed: int = 0,
) -> dict:
    """Print + return the C3 rung-4 verdict: ``az_label`` BEATS ``LookaheadAgent``.

    The bar (README S5 rung 4): strictly lower worst-case L1 spins/pass AND
    strictly lower rounds-to-finish than ``LookaheadAgent``, with 100% finish,
    gated by the bootstrap CI (separated intervals), with heat-efficiency
    confirming better *budgeting*. A non-beat is reported as an HONEST plateau.
    """
    print(f"\n=== C3 rung-4 success criteria (solo, L1): {az_label} BEATS LookaheadAgent ===")
    look = solo.get("Lookahead")
    az = solo.get(az_label)
    if look is None or az is None:
        print("  (missing Lookahead / AZ aggregates)")
        return {"beat": False, "reason": "missing aggregates"}

    az_spins = agg_spin_ci(az, 1, seed=seed)
    look_spins = agg_spin_ci(look, 1, seed=seed)
    az_rounds = agg_rounds_ci(az, seed=seed + 1)
    look_rounds = agg_rounds_ci(look, seed=seed + 1)
    az_dph, az_cd = az.heat_efficiency()
    look_dph, look_cd = look.heat_efficiency()

    print(f"  worst-case spins/L1-pass (point [90% CI]):  {az_label}={az_spins.fmt()}  "
          f"Lookahead={look_spins.fmt()}")
    print(f"  rounds-to-finish (point [90% CI]):          {az_label}={az_rounds.fmt()}  "
          f"Lookahead={look_rounds.fmt()}")
    print(f"  solo finish rate:                           {az_label}="
          f"{az.finish_rate() * 100:.0f}%  Lookahead={look.finish_rate() * 100:.0f}%")

    def _fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.2f}"

    print(f"  heat efficiency (dist/heat):                {az_label}={_fmt(az_dph)} "
          f"(cooldowns/game={_fmt(az_cd)})  Lookahead={_fmt(look_dph)} "
          f"(cooldowns/game={_fmt(look_cd)})")

    beat_spins = _beats_on_ci(az_spins, look_spins)
    beat_rounds = _beats_on_ci(az_rounds, look_rounds)
    finish = az.finish_rate() >= 0.999
    beat = beat_spins and beat_rounds and finish

    print(f"  -> worst-case L1 spins < Lookahead (CI-separated): "
          f"{'PASS' if beat_spins else 'FAIL'}")
    print(f"  -> rounds-to-finish < Lookahead (CI-separated):    "
          f"{'PASS' if beat_rounds else 'FAIL'}")
    print(f"  -> solo finish == 100%:                            "
          f"{'PASS' if finish else 'FAIL'}")

    if beat:
        print("  VERDICT: rung-4 PASS -- the AZ net BEATS LookaheadAgent (CI-gated). "
              "Option C's reason to exist is met.")
        reason = "rung-4 BEAT (CI-separated on both axes, 100% finish)"
    else:
        print(
            "  VERDICT: rung-4 NOT MET -- the AZ net matches/trails LookaheadAgent.\n"
            "  Per README S5 this is a legitimate HONEST plateau (report the gap +\n"
            "  recommend a deferred-seam escalation or the A/B spine), NOT a fake\n"
            "  pass. Do not paper over with finish-rate."
        )
        reason = "plateau -- did not beat Lookahead on both CI-separated axes"
    return {
        "beat": beat,
        "reason": reason,
        "az_spins": az_spins.__dict__,
        "look_spins": look_spins.__dict__,
        "az_rounds": az_rounds.__dict__,
        "look_rounds": look_rounds.__dict__,
        "az_finish": az.finish_rate(),
        "az_dist_per_heat": float(az_dph) if az_dph == az_dph else None,
        "look_dist_per_heat": float(look_dph) if look_dph == look_dph else None,
    }


# ---------------------------------------------------------------------------
# Field construction (in-search + net-only AZ contenders)
# ---------------------------------------------------------------------------


def build_solo_labels(
    *,
    model_path: str,
    sims: int,
    horizon: int,
    dets: int,
    seed: int,
    include_net_only: bool,
    in_search_label: str = "AZ-MCTS",
    net_only_label: str = "AZ-net",
) -> dict[str, "callable"]:
    """The solo field: Heuristic, Lookahead, AZ in-search, (optionally) AZ net-only.

    Each agent is built by a fresh-per-game factory (clean per-game RNG/profile);
    the AZ net loads lazily by path through the NetAdapter / MLAgent tripwire
    (codec v3), so the field ships into ``_run_field`` carrying only a path string.
    """
    cfg = MCTSConfig(n_simulations=sims)

    def make_mcts() -> BaseAgent:
        return MCTSAgent(model_path=model_path, config=cfg, seed=seed,
                         name=in_search_label)

    def make_net_only() -> BaseAgent:
        return MLAgent(model_path, deterministic=True, name=net_only_label)

    labels: dict[str, "callable"] = {
        "Heuristic": lambda: HeuristicAgent(),
        "Lookahead": lambda: LookaheadAgent(horizon=horizon, n_determinizations=dets),
        in_search_label: make_mcts,
    }
    if include_net_only:
        labels[net_only_label] = make_net_only
    return labels


def evaluate_az(args: argparse.Namespace) -> dict:
    """Run the rung-4 eval and return a report dict (also prints the field)."""
    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]
    labels = build_solo_labels(
        model_path=args.model,
        sims=args.sims,
        horizon=args.horizon,
        dets=args.dets,
        seed=args.seed,
        include_net_only=not args.skip_net_only,
    )
    solo, solo_prof = _run_field(
        labels, track_seeds=track_seeds, num_players=1, game_seed_base=args.seed
    )
    _print_field("SOLO (rung-4 gate)", solo, solo_prof)

    report = {"in_search": rung4_verdict(solo, az_label="AZ-MCTS", seed=args.seed)}
    if not args.skip_net_only:
        report["net_only"] = rung4_verdict(solo, az_label="AZ-net", seed=args.seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        help="SB3 MaskablePPO checkpoint (the promoted C3 net, "
                             "codec v3, with a .meta.json sidecar)")
    parser.add_argument("--games", type=int, default=24,
                        help="held-out generated tracks to evaluate on")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTSAgent n_simulations (C0 default 16)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="LookaheadAgent rollout depth (the rung-1 baseline)")
    parser.add_argument("--dets", type=int, default=2,
                        help="LookaheadAgent determinizations")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed for game RNG (track seeds are held-out)")
    parser.add_argument("--skip-net-only", action="store_true",
                        help="evaluate only the in-search agent (faster)")
    parser.add_argument("--net-only", action="store_true",
                        help="(no-op alias kept for symmetry; net-only is on by "
                             "default unless --skip-net-only)")
    args = parser.parse_args()

    from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
    print(
        f"eval_az: {args.games} held-out generated tracks (tight, L1-weighted), "
        f"MCTS sims={args.sims}, model={args.model} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    evaluate_az(args)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from _runlog import run_main

    run_main("eval_az", main)
