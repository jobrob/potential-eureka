"""Sprint C3 -- mint the trained-WARM gen-0 prior the AZ loop bootstraps from.

C2's verdict (C2-findings.md) was explicit: one generate->train->eval cycle off a
**cold-start random** codec-v3 net produced no improvement signal (collapsed visit
targets, a ~14-round-MAE value head). Its #1 recommendation for C3: **start the
loop from a trained-warm v3 prior, not a cold random one.** The machinery already
exists; this script is pure glue over it -- no new model, codec, or training loop.

The warm gen-0 net = warm actor (BC) + warm critic (value), in ONE checkpoint
---------------------------------------------------------------------------
A warm gen-0 net must carry BOTH a non-degenerate policy prior AND a calibrated
leaf value under the live codec v3. The two halves are produced by the existing
precursor pipelines and **combined**, exploiting ``share_features_extractor=False``
(the model.py default): the actor and critic own byte-disjoint module sets
(``pi_features_extractor`` / ``mlp_extractor.policy_net`` / ``action_net`` for the
actor; ``vf_features_extractor`` / ``mlp_extractor.value_net`` / ``value_net`` for
the critic). So:

  1. **Warm actor** -- ``gen_demos --players 1`` (solo search-teacher demos on the
     100_000/500_000 bands) + ``train_bc`` -> a checkpoint whose ACTOR is BC-warm
     (critic at init).
  2. **Warm critic** -- ``gen_value_data`` (solo MC value data, same bands) +
     ``train_value`` -> a checkpoint whose CRITIC is value-warm (actor at init).
  3. **Combine** -- load both policies, copy the critic-side modules from the value
     net into the BC net (a disjoint state-dict graft), and re-save as one
     ``MaskablePPO`` checkpoint. Because the modules are disjoint, the graft leaves
     the BC actor untouched and replaces only the init critic with the trained one.

The result loads through the EXACT ``MLAgent`` / ``MCTSAgent.NetAdapter`` tripwire
the loop consumes, and is validated to (a) pass the §3.4 contract sidecar and (b)
beat a random net in ``eval_az`` before it is fed to the loop (the C3 spec's
"confirm it beats a random net" gate).

Seed-band disjointness
----------------------
The BC/value precursor data live in the 100_000 (train) / 500_000 (val) bands --
DISJOINT from the self-play band (300_000, ``gen_selfplay``) and the held-out eval
band (900_000, ``eval_search``). So the warm prior is never trained on a self-play
or gate track. Asserted at startup.

Usage:
    python experiments/mint_warm_prior.py --smoke --out checkpoints/c3_warm_gen0.zip
    python experiments/mint_warm_prior.py --train-tracks 80 --val-tracks 20 \
        --bc-epochs 30 --value-epochs 30 --out checkpoints/c3_warm_gen0.zip
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

import gen_demos
import gen_value_data
import train_bc
import train_value

from heat.ml.spaces import CODEC_VERSION, OBS_DIM, ACTION_DIM


# Precursor seed bands (gen_demos / gen_value_data). Disjoint from self-play
# (300_000) and the held-out eval band (900_000).
_PRECURSOR_TRAIN_BASE = 100_000
_PRECURSOR_VAL_BASE = 500_000
_SELFPLAY_BASE = 300_000
_EVAL_BASE = 900_000

# The critic-side module prefixes (share_features_extractor=False makes these
# byte-disjoint from the actor's). Matches train_value._critic_parameters.
_CRITIC_PREFIXES = (
    "vf_features_extractor.",
    "mlp_extractor.value_net.",
    "value_net.",
)


def _assert_bands_disjoint(train_tracks: int, val_tracks: int) -> None:
    """Fail fast if the precursor bands could overlap self-play / eval bands."""
    for name, base, n in (
        ("precursor-train", _PRECURSOR_TRAIN_BASE, train_tracks),
        ("precursor-val", _PRECURSOR_VAL_BASE, val_tracks),
    ):
        hi = base + max(n, 100_000)  # require a clear window, as gen_selfplay does
        for other_name, other_base in (("self-play", _SELFPLAY_BASE),
                                       ("eval-heldout", _EVAL_BASE)):
            o_lo, o_hi = other_base, other_base + 100_000
            if not (hi <= o_lo or base >= o_hi):
                raise ValueError(
                    f"{name} band [{base},{hi}) overlaps the {other_name} band "
                    f"[{o_lo},{o_hi}); the warm prior would train on a "
                    f"{other_name} track"
                )


def _graft_critic(bc_path: str, value_path: str, out: str, *, seed: int) -> str:
    """Combine the BC actor + value critic into one MaskablePPO checkpoint.

    Loads the BC net (warm actor / init critic) and the value net (init actor /
    warm critic), copies every critic-prefixed parameter+buffer from the value
    net's policy state dict into the BC net's, and re-saves. The critic prefixes
    are byte-disjoint from the actor's (share_features_extractor=False), so the
    BC actor is preserved exactly and only the init critic is replaced.
    """
    from sb3_contrib import MaskablePPO
    from heat.ml.model import PPOConfig
    from heat.ml.training import save_checkpoint

    bc = MaskablePPO.load(bc_path, device="cpu")
    value = MaskablePPO.load(value_path, device="cpu")

    bc_sd = bc.policy.state_dict()
    val_sd = value.policy.state_dict()

    grafted = 0
    for k in bc_sd:
        if any(k.startswith(p) for p in _CRITIC_PREFIXES):
            if k not in val_sd:
                raise KeyError(
                    f"critic key {k!r} missing from the value net state dict "
                    "(architecture mismatch between BC and value checkpoints)"
                )
            if bc_sd[k].shape != val_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on critic key {k!r}: BC {tuple(bc_sd[k].shape)} "
                    f"vs value {tuple(val_sd[k].shape)}"
                )
            bc_sd[k] = val_sd[k].clone()
            grafted += 1
    if grafted == 0:
        raise RuntimeError(
            "no critic parameters grafted -- the critic prefixes did not match any "
            "state-dict key (model architecture changed?)"
        )
    bc.policy.load_state_dict(bc_sd)

    # Re-save through the standard contract-sidecar path so MLAgent / NetAdapter
    # load it unchanged. The combined net IS the warm gen-0 prior.
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cfg = PPOConfig(seed=seed, device="cpu", share_features_extractor=False)
    sidecar = save_checkpoint(
        bc, out, track_name="generated", num_players=1, seed=seed, ppo_config=cfg
    )
    return sidecar


def _verify_warm_vs_random(warm_path: str, *, games: int, sims: int, seed: int) -> dict:
    """Confirm the warm net beats a random net in-search (eval_az), per the C3 spec.

    Mints a random cold net and runs a small head-to-head: the warm prior's
    in-search worst-case L1 spins/pass and rounds should be strictly better than
    the random net's. A non-beat here means the warm prior carries no skill and the
    loop must NOT be started (C2's cold-start futility). Returns a small dict.
    """
    import tempfile
    import eval_az
    from eval_search import _HELDOUT_BASE, _run_field
    from heat.ml.env import HeatEnv
    from heat.ml.model import build_model, PPOConfig
    from heat.ml.training import save_checkpoint
    from heat.agents.mcts_agent import MCTSAgent, MCTSConfig

    tmpdir = tempfile.mkdtemp(prefix="c3_randcmp_")
    rand_path = os.path.join(tmpdir, "random.zip")
    save_checkpoint(
        build_model(HeatEnv(num_players=1), PPOConfig(seed=seed + 7, device="cpu")),
        rand_path, track_name="generated", num_players=1, seed=seed + 7,
    )

    cfg = MCTSConfig(n_simulations=sims)
    track_seeds = [_HELDOUT_BASE + i for i in range(games)]

    def _mk(path, label):
        return lambda: MCTSAgent(model_path=path, config=cfg, seed=seed, name=label)

    labels = {"warm": _mk(warm_path, "warm"), "random": _mk(rand_path, "random")}
    solo, _ = _run_field(labels, track_seeds=track_seeds, num_players=1,
                         game_seed_base=seed)
    warm = solo["warm"]
    rand = solo["random"]
    _, _, warm_worst = warm.spin_stats(1)
    _, _, rand_worst = rand.spin_stats(1)
    warm_rounds = warm.mean_rounds()
    rand_rounds = rand.mean_rounds()

    def _lt(a, b):  # a strictly better (lower), NaN-safe
        return a == a and b == b and a < b - 1e-9

    beats = _lt(warm_worst, rand_worst) or _lt(warm_rounds, rand_rounds)
    print("\n=== warm-vs-random sanity (in-search, held-out solo) ===")
    print(f"  worst-case L1 spins/pass: warm={warm_worst:.3f}  random={rand_worst:.3f}")
    print(f"  rounds-to-finish:         warm={warm_rounds:.3f}  random={rand_rounds:.3f}")
    print(f"  -> warm beats random (either axis): {'PASS' if beats else 'FAIL'}")
    if not beats:
        print("  WARNING: the warm prior does NOT beat a random net. Starting the "
              "AZ loop from it is the cold-start futility C2 proved -- do not "
              "launch the loop until this passes (raise tracks/epochs).")
    return {
        "warm_worst_l1": float(warm_worst) if warm_worst == warm_worst else None,
        "random_worst_l1": float(rand_worst) if rand_worst == rand_worst else None,
        "warm_rounds": float(warm_rounds) if warm_rounds == warm_rounds else None,
        "random_rounds": float(rand_rounds) if rand_rounds == rand_rounds else None,
        "warm_beats_random": bool(beats),
    }


def mint_warm_prior(args: argparse.Namespace) -> dict:
    """Generate BC+value precursor data, train both, graft, validate. Returns a dict."""
    _assert_bands_disjoint(args.train_tracks, args.val_tracks)
    os.makedirs(args.workdir, exist_ok=True)

    bc_data = os.path.join(args.workdir, "bc_demos.npz")
    value_data = os.path.join(args.workdir, "value_data.npz")
    bc_ckpt = os.path.join(args.workdir, "bc_actor.zip")
    value_ckpt = os.path.join(args.workdir, "value_critic.zip")

    # (1) warm actor: solo BC demos from the search teacher + BC train.
    print("\n--- [1/4] generating solo BC demos (search teacher) ---")
    gen_demos.generate_dataset(argparse.Namespace(
        out=bc_data, players=1, train_tracks=args.train_tracks,
        val_tracks=args.val_tracks, horizon=args.horizon, dets=args.dets,
        top_k=args.top_k, sim_budget=args.sim_budget, open_hand=False,
        expert_seed=args.seed, game_seed=args.game_seed,
    ))
    print("\n--- [1/4] training BC actor ---")
    train_bc.train_bc(argparse.Namespace(
        data=bc_data, out=bc_ckpt, epochs=args.bc_epochs, batch=256, lr=3e-4,
        eval_every=max(1, args.bc_epochs // 2), patience=args.patience,
        early_stop_metric="val_ce", net=args.net, device=args.device,
        seed=args.seed, players_meta=1,
    ))

    # (2) warm critic: solo MC value data + value train.
    print("\n--- [2/4] generating solo MC value data ---")
    gen_value_data.generate_dataset(argparse.Namespace(
        out=value_data, train_tracks=args.train_tracks, val_tracks=args.val_tracks,
        rollouts=args.rollouts, game_seed=args.game_seed + 1, policy="heuristic",
        value_model=None, horizon=args.horizon, dets=args.dets, top_k=args.top_k,
        sim_budget=args.sim_budget,
    ))
    print("\n--- [2/4] training value critic ---")
    train_value.train_value(argparse.Namespace(
        data=value_data, out=value_ckpt, epochs=args.value_epochs, batch=256,
        lr=3e-4, loss="mse", eval_every=max(1, args.value_epochs // 2),
        patience=args.patience, net=args.net, device=args.device, seed=args.seed,
    ))

    # (3) graft the warm critic into the warm-actor net -> the warm gen-0 prior.
    print("\n--- [3/4] grafting warm critic into warm actor (warm gen-0 net) ---")
    sidecar = _graft_critic(bc_ckpt, value_ckpt, args.out, seed=args.seed)
    print(f"  wrote warm gen-0 prior -> {args.out} (sidecar {sidecar})")

    # Validate the §3.4 contract tripwire (the exact MLAgent / NetAdapter check).
    from heat.agents.ml_agent import MLAgent
    MLAgent(args.out, name="warm")._validate_meta()
    print("  contract tripwire (obs_dim/action_dim/codec_version): PASS")

    # (4) confirm it beats a random net in-search before the loop consumes it.
    print("\n--- [4/4] warm-vs-random sanity (in-search) ---")
    sanity = _verify_warm_vs_random(
        args.out, games=args.verify_games, sims=args.sims, seed=args.seed
    )

    return {"out": args.out, "sidecar": sidecar, "sanity": sanity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="checkpoints/c3_warm_gen0.zip",
                        help="output warm gen-0 checkpoint (codec v3, sidecar alongside)")
    parser.add_argument("--workdir", type=str, default="data/c3_warm",
                        help="scratch dir for precursor data + half-checkpoints")
    parser.add_argument("--train-tracks", type=int, default=80)
    parser.add_argument("--val-tracks", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=1,
                        help="MC rollouts per track for the value data")
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--value-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2,
                        help="search-teacher rollout depth (BC + value gen)")
    parser.add_argument("--dets", type=int, default=2,
                        help="search-teacher determinizations")
    parser.add_argument("--top-k", type=int, default=6,
                        help="search-teacher top-k branching cap (S2 'free accuracy')")
    parser.add_argument("--sim-budget", type=int, default=None)
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"])
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cuda|cpu for the two trainers")
    parser.add_argument("--sims", type=int, default=16,
                        help="MCTS sims for the warm-vs-random sanity check")
    parser.add_argument("--verify-games", type=int, default=12,
                        help="held-out tracks for the warm-vs-random sanity check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game-seed", type=int, default=7000)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny smoke (6/2 tracks, 4 epochs, sims=8, 6 verify games)")
    args = parser.parse_args()

    if args.smoke:
        args.train_tracks = 6
        args.val_tracks = 2
        args.bc_epochs = 4
        args.value_epochs = 4
        args.sims = 8
        args.verify_games = 4
        args.patience = 0

    print(
        f"mint_warm_prior: train_tracks={args.train_tracks} val={args.val_tracks} "
        f"bc_epochs={args.bc_epochs} value_epochs={args.value_epochs} net={args.net} "
        f"(codec v{CODEC_VERSION}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})"
    )
    result = mint_warm_prior(args)
    print("\n=== mint_warm_prior summary ===")
    print(f"  warm gen-0 prior: {result['out']}")
    s = result["sanity"]
    print(f"  warm beats random (in-search): {s['warm_beats_random']}")


if __name__ == "__main__":
    from _runlog import run_main

    run_main("mint_warm_prior", main)
