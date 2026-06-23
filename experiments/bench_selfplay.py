"""Sprint C7 -- self-play throughput benchmark (a first-class deliverable).

The C7 prize is a MEASURED >=10x self-play generation throughput improvement with
a bit-identical dataset (C6-findings established the loop ran at ~0.015% of the
games an AlphaZero result needs because generation is too slow). This harness is
how that claim is substantiated: it runs ``gen_selfplay`` on a fixed config + seed
and reports

  * games/sec, sims/sec, ms/move (the headline throughput numbers);
  * the per-phase breakdown (encode / net-fwd / engine / clone) via cProfile, so a
    regression in any one phase is visible;
  * the model-load amortization (how many times the SB3 ``.zip`` was loaded from
    disk, and the per-game share) -- the proof the per-game reload tax is gone.

It does NOT change the search, the targets, or the loop's learning logic; it only
measures. Run it BEFORE and AFTER a change on the same config to read the speedup.

Usage:
    # baseline / after, serial:
    python experiments/bench_selfplay.py --model checkpoints/c5_main_best.zip \\
        --tracks 6 --sims 16 --two-player
    # parallel scaling:
    python experiments/bench_selfplay.py --model checkpoints/c5_main_best.zip \\
        --tracks 24 --sims 16 --two-player --workers 4
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import gen_selfplay
from heat.agents import mcts_agent
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM


# Per-phase attribution. We sum cProfile ``tottime`` (time IN the frame itself,
# excluding sub-calls) and charge each frame to the FIRST phase whose marker it
# matches -- so no frame is double-counted and the buckets sum to wall (minus the
# unattributed "other"). The markers reproduce the C6 profile's decomposition
# (C6-findings §Point-1): net-fwd is the torch forward path (linear / logsumexp /
# Categorical / relu / tensor marshalling); encode is the pure-Python obs build;
# engine is run_round_driver; clone is GameState/Player/deck cloning.
def _match(fname: str, func: str, key: str) -> bool:
    return key in func or key in fname


def _classify(fname: str, func: str) -> str:
    base = os.path.basename(fname)
    # net-fwd: torch forward + the distribution/softmax path + tensor marshalling.
    if "torch" in fname or func in (
        "linear", "logsumexp", "relu", "softmax", "log_softmax",
    ) or "as_tensor" in func or func == "reshape" or "Categorical" in func:
        return "net-fwd"
    # encode: the observation builder (features.py) -- the per-leaf obs cost.
    if base == "features.py":
        return "encode"
    # engine: the rules driver.
    if base == "driver.py" or func == "run_round_driver" or base in ("game.py", "phases.py", "rules.py"):
        return "engine"
    # clone: GameState / PlayerState / deck cloning.
    if "clone" in func or base in ("game_state.py", "player_state.py", "deck.py"):
        return "clone"
    return "other"


def _bucket_seconds(stats: pstats.Stats) -> dict[str, float]:
    """Sum tottime seconds per phase bucket from a cProfile Stats (no double count)."""
    buckets = {"encode": 0.0, "net-fwd": 0.0, "engine": 0.0, "clone": 0.0, "other": 0.0}
    # stats.stats: {(file, lineno, name): (cc, nc, tottime, cumtime, callers)}
    for (fname, _lineno, func), (_cc, _nc, tottime, _cumtime, _callers) in stats.stats.items():
        buckets[_classify(fname, func)] += tottime
    return buckets


def _build_args(a: argparse.Namespace, out: str) -> argparse.Namespace:
    """A gen_selfplay Namespace for the fixed benchmark config."""
    return argparse.Namespace(
        out=out,
        model=a.model,
        tracks=a.tracks,
        val_tracks=a.val_tracks,
        sims=a.sims,
        dirichlet_eps=a.dirichlet_eps,
        dirichlet_alpha=a.dirichlet_alpha,
        temperature_moves=a.temperature_moves,
        root_selector=a.root_selector,
        traj_greedy=a.traj_greedy,
        two_player=a.two_player,
        opponent_snapshot=a.opponent_snapshot,
        workers=a.workers,
        seed=a.seed,
        game_seed=a.game_seed,
    )


def run_benchmark(a: argparse.Namespace) -> dict:
    """Run gen_selfplay under the fixed config; return the throughput report."""
    out = os.path.join(a.tmpdir, "bench_selfplay.npz")
    os.makedirs(a.tmpdir, exist_ok=True)
    gen_args = _build_args(a, out)

    # Reset the process-level model cache so the load-amortization line reflects a
    # clean cold start for THIS benchmark run (the worker cache is per-process; for
    # workers==1 this is the main process's cache).
    mcts_agent._MODEL_CACHE.clear()
    gen_selfplay._NET_ADAPTER_CACHE.clear()

    # Count + time every SB3 load that actually hits disk (a cache miss). In the
    # C7 path this fires ONCE per process; with --baseline-no-cache it fires once
    # per GAME (the pre-C7 reload tax), so the report's load count is the honest
    # before/after proof. We patch MaskablePPO.load itself so both paths are timed.
    load_calls = {"n": 0, "cold_s": 0.0}
    from sb3_contrib import MaskablePPO

    _orig_sb3_load = MaskablePPO.load.__func__

    def _timed_sb3_load(cls, path, *pargs, **pkwargs):
        t = time.perf_counter()
        m = _orig_sb3_load(cls, path, *pargs, **pkwargs)
        dt = time.perf_counter() - t
        load_calls["n"] += 1
        load_calls["cold_s"] += dt
        return m

    MaskablePPO.load = classmethod(_timed_sb3_load)

    # --baseline-no-cache reproduces the PRE-C7 per-game reload tax for an honest
    # before/after: it disables BOTH the process model cache and the shared adapter
    # so each game reloads the SB3 .zip from disk on its first leaf eval, exactly as
    # the C6 ``MCTSAgent``-per-game generator did. The search output is unchanged
    # (the weights are identical); only the load count + wall time differ.
    _saved_cached_adapter = gen_selfplay._cached_net_adapter
    _saved_get_model = mcts_agent.NetAdapter._get_model
    if getattr(a, "baseline_no_cache", False):
        gen_selfplay._cached_net_adapter = lambda mp: mcts_agent.NetAdapter(mp)

        def _uncached_get_model(self):
            if self._model is None:
                self._validate_meta()
                self._model = MaskablePPO.load(self.model_path, device="cpu")
            return self._model

        mcts_agent.NetAdapter._get_model = _uncached_get_model

    try:
        if a.profile and a.workers == 1:
            prof = cProfile.Profile()
            t0 = time.perf_counter()
            prof.enable()
            summary = gen_selfplay.generate_dataset(gen_args)
            prof.disable()
            wall = time.perf_counter() - t0
            stats = pstats.Stats(prof)
            buckets = _bucket_seconds(stats)
        else:
            # Parallel runs cannot be cProfiled across processes; report wall-clock
            # throughput only (the per-phase breakdown is a serial diagnostic).
            t0 = time.perf_counter()
            summary = gen_selfplay.generate_dataset(gen_args)
            wall = time.perf_counter() - t0
            buckets = None
    finally:
        MaskablePPO.load = classmethod(_orig_sb3_load)
        gen_selfplay._cached_net_adapter = _saved_cached_adapter
        mcts_agent.NetAdapter._get_model = _saved_get_model

    n_games = summary["races_total"]
    # ms/move and clones/move are the per-move search averages summed across games
    # (gen_selfplay's profile accounting, preserved across workers). games/sec is the
    # wall-clock throughput -- the headline number the >=10x claim is read from.
    ms_per_move = summary["ms_per_move"]
    clones_per_move = summary["clones_per_move"]
    games_per_s = n_games / wall if wall > 0 else float("nan")

    report = {
        "config": {
            "model": a.model,
            "tracks": a.tracks,
            "val_tracks": a.val_tracks,
            "sims": a.sims,
            "two_player": a.two_player,
            "opponent_snapshot": a.opponent_snapshot,
            "workers": a.workers,
            "codec_version": CODEC_VERSION,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
        },
        "wall_s": round(wall, 3),
        "n_games": n_games,
        "n_rows": summary["n_rows"],
        "games_per_s": games_per_s,
        "ms_per_move": ms_per_move,
        "clones_per_move": clones_per_move,
        "model_loads": load_calls["n"],
        "model_load_cold_s": round(load_calls["cold_s"], 3),
        "model_load_per_game_s": round(load_calls["cold_s"] / n_games, 4) if n_games else float("nan"),
        "phase_buckets_s": buckets,
    }
    # sims/sec: sims-per-move is fixed (a.sims); ms_per_move gives moves/sec.
    if ms_per_move and ms_per_move == ms_per_move:  # not NaN
        moves_per_s = 1000.0 / ms_per_move
        report["search_moves_per_s"] = round(moves_per_s, 1)
        report["sims_per_s"] = round(moves_per_s * a.sims, 1)
    return report


def _print_report(r: dict) -> None:
    c = r["config"]
    print("\n=== bench_selfplay (Sprint C7 throughput) ===")
    print(f"  config: model={os.path.basename(c['model'])} tracks={c['tracks']} "
          f"sims={c['sims']} two_player={c['two_player']} "
          f"opp={c['opponent_snapshot']} workers={c['workers']}")
    print(f"  wall: {r['wall_s']}s  games={r['n_games']} rows={r['n_rows']}")
    print(f"  THROUGHPUT: games/sec={r['games_per_s']:.3f}  "
          f"ms/move={r['ms_per_move']:.2f}  clones/move={r['clones_per_move']:.1f}")
    if "sims_per_s" in r:
        print(f"              search-moves/sec={r['search_moves_per_s']:.1f}  "
              f"sims/sec={r['sims_per_s']:.1f} (per worker)")
    print(f"  MODEL-LOAD AMORTIZATION: loads={r['model_loads']} "
          f"(cold {r['model_load_cold_s']}s total, {r['model_load_per_game_s']}s/game)")
    if r["model_loads"] <= c["workers"]:
        print(f"    -> reload tax GONE: model loaded once per process "
              f"(<= workers={c['workers']}), not once per game")
    else:
        print(f"    -> WARNING: model loaded {r['model_loads']}x for {r['n_games']} "
              f"games (reload tax still present!)")
    if r["phase_buckets_s"]:
        b = r["phase_buckets_s"]
        tot = sum(b.values()) or 1.0
        print("  PER-PHASE (tottime, cProfile, serial):")
        for k in ("encode", "net-fwd", "engine", "clone", "other"):
            print(f"    {k:<8} {b[k]:.3f}s  ({b[k] / tot * 100:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True,
                   help="SB3 MaskablePPO checkpoint (codec v3 + .meta.json)")
    p.add_argument("--tracks", type=int, default=6,
                   help="TRAIN tracks (games) to run for the benchmark")
    p.add_argument("--val-tracks", type=int, default=0)
    p.add_argument("--sims", type=int, default=16, help="MCTS sims/move (C0 default 16)")
    p.add_argument("--two-player", action="store_true",
                   help="benchmark the C6 1v1 win/loss generator (the C7 Tier-0 config)")
    p.add_argument("--opponent-snapshot", type=str, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel self-play workers (C7); >1 disables cProfile")
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--temperature-moves", type=int, default=10)
    p.add_argument("--root-selector", type=str, default="puct", choices=["puct", "gumbel"])
    p.add_argument("--traj-greedy", action="store_true")
    p.add_argument("--no-profile", dest="profile", action="store_false",
                   help="skip the cProfile per-phase breakdown (faster wall timing)")
    p.add_argument("--baseline-no-cache", action="store_true",
                   help="reproduce the PRE-C7 per-game model-reload tax (disable the "
                        "process model cache + shared adapter) so each game reloads "
                        "the SB3 .zip -- the honest before/after baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game-seed", type=int, default=8000)
    p.add_argument("--tmpdir", type=str, default="runs/bench_selfplay")
    p.set_defaults(profile=True)
    a = p.parse_args()

    # The self-play band must stay disjoint from the eval/precursor bands; the
    # benchmark uses gen_selfplay's own band, so no extra seeding is needed.
    print(f"bench_selfplay: model={a.model} tracks={a.tracks} sims={a.sims} "
          f"two_player={a.two_player} workers={a.workers} "
          f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})")
    report = run_benchmark(a)
    _print_report(report)


if __name__ == "__main__":
    main()
