"""Sprint C1 eval: the solo stochastic ``MCTSAgent`` on the held-out solo field.

Rung 2 of the Option-C ladder (README §5): with a **fixed** net, the C-MCTS must
**MATCH** ``LookaheadAgent`` -- worst-case (max) spins/limit-1-pass <= Lookahead
(and <= Heuristic), rounds-to-finish <= Lookahead, 100% solo finish -- at a
*reported* (not necessarily smaller) ms/move + clones/move within the C0 budget.

This is a thin sibling of ``experiments/eval_search.py``: it imports that
harness's spin/finish/heat-efficiency machinery **verbatim** (the
``_passes_and_spins_by_limit`` / ``_AgentAgg`` / ``_heat_efficiency`` /
``_run_field`` / ``_HELDOUT_BASE`` / ``_TIGHT_PARAMS`` symbols -- one source of
truth, not a fork) and only adds the ``MCTSAgent`` contender + the rung-2 verdict
print (mirroring ``_success_check``).

The fixed net
-------------
``--model PATH`` supplies the SB3 ``MaskablePPO`` checkpoint (with a ``.meta.json``
sidecar) that drives the search's prior + leaf value. Because ``CODEC_VERSION``
was bumped 2->3, the S3 BC checkpoint the C1 doc suggested as the warm prior is
REJECTED by the contract tripwire -- and no v3 checkpoint exists yet. So:

  * with no ``--model``, the harness MINTS a fresh **random-weight** v3 checkpoint
    (``build_model`` + a proper v3 ``.meta.json``) and reports the result as the
    honest **cold-start FLOOR** -- NOT a parity claim. A random prior has no
    driving skill, so it will not match ``LookaheadAgent``; that gap is exactly
    what C2's training lifts. Demonstrating full rung-2 parity needs a *trained*
    v3 fixed prior (a fast follow-up: regenerate a small BC/value net under codec
    v3, then re-run this harness with ``--model``).

Either way the harness RUNS end-to-end through the real checkpoint-load +
tripwire + net-adapter path and prints the rung-2 verdict, so a v3 prior drops in
with no code change.

Usage:
    python experiments/eval_mcts.py --games 12 --sims 16
    python experiments/eval_mcts.py --games 24 --sims 16 --model checkpoints/c_prior.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

# Import the eval_search machinery VERBATIM (single source of truth).
sys.path.insert(0, os.path.dirname(__file__))
from eval_search import (  # noqa: E402
    _HELDOUT_BASE,
    _TIGHT_PARAMS,
    _AgentAgg,
    _print_field,
    _run_field,
)

from heat.agents.base import BaseAgent  # noqa: E402
from heat.agents.heuristic_agent import HeuristicAgent  # noqa: E402
from heat.agents.strong_heuristic import StrongHeuristicAgent  # noqa: E402
from heat.agents.search_agent import LookaheadAgent  # noqa: E402
from heat.agents.mcts_agent import MCTSAgent, MCTSConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Cold-start fixed net (random-weight v3 checkpoint)
# ---------------------------------------------------------------------------


def _mint_cold_start_checkpoint(out_dir: str, seed: int = 0) -> str:
    """Build a fresh random-weight ``MaskablePPO`` + a v3 ``.meta.json`` sidecar.

    The honest cold-start FLOOR prior (README C1 Risk 1): a randomly-initialized
    ``build_model`` net saved with the proper contract sidecar so it loads through
    the exact ``MLAgent``/``NetAdapter`` tripwire path the trained prior will. CPU
    device (the C0 default). Returns the checkpoint path.
    """
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model, PPOConfig
    from heat.ml.training import save_checkpoint

    model = build_model(HeatEnv(num_players=1), PPOConfig(seed=seed, device="cpu"))
    path = os.path.join(out_dir, "c1_cold_start.zip")
    save_checkpoint(
        model,
        path,
        track_name="generated",
        num_players=1,
        seed=seed,
    )
    return path


# ---------------------------------------------------------------------------
# Rung-2 verdict (mirrors eval_search._success_check, MCTS vs Lookahead)
# ---------------------------------------------------------------------------


def _rung2_verdict(solo: dict[str, _AgentAgg], cold_start: bool) -> None:
    """Print the C1 rung-2 verdict honestly: MCTS MATCH LookaheadAgent (solo, L1).

    The bar (README §5 rung 2): worst-case spins/limit-1-pass <= Lookahead (and
    <= Heuristic), rounds-to-finish <= Lookahead, 100% solo finish. ms/move +
    clones/move are reported by ``_print_field``'s profiling block. When the prior
    is the cold-start random net, the verdict is framed as a FLOOR, not a parity
    claim (a random prior cannot be expected to match the search baseline).
    """
    print("\n=== C1 rung-2 success criteria (solo, limit-1): MATCH LookaheadAgent ===")
    if cold_start:
        print(
            "  NOTE: prior = random-weight COLD-START net (no v3 trained "
            "checkpoint exists yet). Numbers below are the honest cold-start "
            "FLOOR, NOT a parity claim. Full rung-2 parity needs a trained v3\n"
            "  fixed prior (fast follow-up: re-run with --model <v3 checkpoint>)."
        )

    heur = solo.get("Heuristic")
    look = solo.get("Lookahead")
    mcts = solo.get("MCTS")
    if heur is None or look is None or mcts is None:
        print("  (missing agent aggregates)")
        return

    _, _, heur_worst = heur.spin_stats(1)
    _, _, look_worst = look.spin_stats(1)
    _, _, mcts_worst = mcts.spin_stats(1)

    def fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.3f}"

    print(
        f"  worst-case spins/limit-1-pass:  MCTS={fmt(mcts_worst)}  "
        f"Lookahead={fmt(look_worst)}  Heuristic={fmt(heur_worst)}"
    )
    print(
        f"  solo finish rate:               MCTS={mcts.finish_rate() * 100:.0f}%  "
        f"Lookahead={look.finish_rate() * 100:.0f}%  "
        f"Heuristic={heur.finish_rate() * 100:.0f}%"
    )
    print(
        f"  rounds-to-finish (mean):        MCTS={fmt(mcts.mean_rounds())}  "
        f"Lookahead={fmt(look.mean_rounds())}  Heuristic={fmt(heur.mean_rounds())}"
    )

    # Vacuous (no L1 passes) counts as "not failed" but is flagged by fmt == n/a.
    crit_spins_look = (
        mcts_worst != mcts_worst
        or look_worst != look_worst
        or mcts_worst <= look_worst + 1e-9
    )
    crit_spins_heur = (
        mcts_worst != mcts_worst
        or heur_worst != heur_worst
        or mcts_worst <= heur_worst + 1e-9
    )
    crit_finish = mcts.finish_rate() >= 0.999
    mcts_rounds = mcts.mean_rounds()
    look_rounds = look.mean_rounds()
    crit_rounds = (
        mcts_rounds != mcts_rounds
        or look_rounds != look_rounds
        or mcts_rounds <= look_rounds + 1e-9
    )

    print(
        f"  -> worst-case spins <= Lookahead: "
        f"{'PASS' if crit_spins_look else 'FAIL'}\n"
        f"  -> worst-case spins <= Heuristic: "
        f"{'PASS' if crit_spins_heur else 'FAIL'}\n"
        f"  -> solo finish == 100%:           "
        f"{'PASS' if crit_finish else 'FAIL'}\n"
        f"  -> rounds <= Lookahead:           "
        f"{'PASS' if crit_rounds else 'FAIL'}"
    )
    if cold_start:
        print(
            "  (a FAIL here on the cold-start prior is EXPECTED and is the C2 "
            "training target -- it is the floor, not the verdict.)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=12,
                        help="held-out generated tracks to evaluate on")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTSAgent n_simulations (C0 default 16)")
    parser.add_argument("--horizon", type=int, default=2,
                        help="LookaheadAgent rollout depth (the rung-1 baseline)")
    parser.add_argument("--dets", type=int, default=2,
                        help="LookaheadAgent determinizations")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed for game RNG (track seeds are held-out)")
    parser.add_argument("--model", type=str, default=None,
                        help="SB3 MaskablePPO checkpoint for the MCTS prior+value "
                             "(with a .meta.json sidecar). When omitted, a random "
                             "cold-start v3 checkpoint is minted and the result is "
                             "reported as the cold-start FLOOR, not a parity claim.")
    parser.add_argument("--skip-4p", action="store_true",
                        help="run only the solo field (the primary gate)")
    args = parser.parse_args()

    track_seeds = [_HELDOUT_BASE + i for i in range(args.games)]

    cold_start = args.model is None
    tmpdir = None
    model_path = args.model
    if cold_start:
        tmpdir = tempfile.mkdtemp(prefix="c1_cold_")
        model_path = _mint_cold_start_checkpoint(tmpdir, seed=args.seed)

    cfg = MCTSConfig(n_simulations=args.sims)

    def make_heuristic() -> BaseAgent:
        return HeuristicAgent()

    def make_strong() -> BaseAgent:
        return StrongHeuristicAgent(strength=2)

    def make_lookahead() -> BaseAgent:
        return LookaheadAgent(horizon=args.horizon, n_determinizations=args.dets)

    def make_mcts() -> BaseAgent:
        # Each game gets a fresh agent (clean per-game profile/RNG); the net loads
        # lazily from model_path through the NetAdapter tripwire (codec v3).
        return MCTSAgent(model_path=model_path, config=cfg, seed=args.seed, name="MCTS")

    labels = {
        "Heuristic": make_heuristic,
        "StrongHeuristic": make_strong,
        "Lookahead": make_lookahead,
        "MCTS": make_mcts,
    }

    print(
        f"eval_mcts: {args.games} held-out generated tracks (tight, L1-weighted), "
        f"MCTS sims={args.sims}, Lookahead horizon={args.horizon}/dets={args.dets}, "
        f"prior={'COLD-START random v3' if cold_start else model_path}"
    )

    solo, solo_prof = _run_field(
        labels, track_seeds=track_seeds, num_players=1, game_seed_base=args.seed
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

    _rung2_verdict(solo, cold_start=cold_start)


if __name__ == "__main__":
    from _runlog import run_main

    run_main("eval_mcts", main)
