"""Sprint A1 (Option A) -- value-net regression trainer for the learned leaf.

Trains the model's **critic head only** to regress the Monte-Carlo value target
``target = -rounds_remaining`` produced by ``experiments/gen_value_data.py``, so
the leaf consumes ``V_leaf = predict_values(state)`` with the same sign convention
as ``progress`` (larger == better -- README A `2.1`). The output is an ordinary
SB3 ``MaskablePPO`` checkpoint (zip + ``.meta.json`` sidecar via
:func:`heat.ml.training.save_checkpoint`) carrying the contract tripwire fields,
so A2's ``MLAgent``-style load (``obs_dim``/``action_dim``/``codec_version``) finds
exactly the architecture it expects.

Why this is NOT PPO (and the symmetry with ``train_bc.py``)
-----------------------------------------------------------
``train_bc.py`` supervises the **actor** (masked cross-entropy on demonstrated
actions) and leaves the critic at init. This trainer is the mirror image: it
supervises the **critic** (MSE/Huber on the MC return) and leaves the actor at
init. We still build a real :class:`sb3_contrib.MaskablePPO` via
:func:`heat.ml.model.build_model` so the trained weights live in the exact
``HeatMLPExtractor`` + critic architecture A2 / C expect, but the optimization is a
hand-rolled supervised Adam loop over ``model.policy`` -- never ``model.learn``.

Only the critic is trained
---------------------------
With ``PPOConfig(share_features_extractor=False)`` the critic owns its own trunk
(``vf_features_extractor`` + ``mlp_extractor.value_net`` + ``value_net``), fully
disjoint from the actor's (README A `2.3`). We build the optimizer over **only the
critic parameters** so the actor trunk/head stay at initialization -- the actor is
never consulted by the leaf (it calls ``predict_values``), so an uncalibrated actor
does not matter.

Track-disjoint split + reporting
---------------------------------
The dataset carries a per-row ``split`` (``train``/``val``) from track-disjoint
seed bands in ``gen_value_data``; we honor it directly (no row shuffle across the
boundary). We report per-epoch train/val **MSE**, and -- the A1 gate -- val **MAE
bucketed by ``corner_limit_at_state``** (the limit-1 bucket called out as the
high-variance slice) and by **distance-to-finish bins**, plus a calibration sanity
check (mean predicted rounds-remaining must decrease as the car nears the finish).

Usage:
    python experiments/train_value.py --data data/value_data.npz \
        --out checkpoints/value.zip
    python experiments/train_value.py --data data/value_data.npz --epochs 40 \
        --loss huber --net large
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from torch import nn

from heat.ml.env import HeatEnv
from heat.ml.model import PPOConfig, build_model, net_profile_config, resolve_device
from heat.ml.spaces import CODEC_VERSION, OBS_DIM
from heat.ml.training import save_checkpoint


#: Distance-to-finish bins (in rounds-remaining) for the bucketed MAE print.
#: Near-finish (small rounds-remaining) is where calibration matters most for the
#: leaf, so the bins are finer there.
_DIST_BINS = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 9999)]


def _load_dataset(path: str) -> dict:
    """Load the value ``.npz`` and validate it against the live codec.

    Fails fast (like ``train_bc._load_dataset`` / ``MLAgent._validate_meta``) if
    the dataset was generated against a different OBS_DIM / codec_version, so a
    stale dataset cannot silently train a garbage value net.
    """
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    rounds_remaining = data["rounds_remaining"].astype(np.float32)
    corner_limit = data["corner_limit_at_state"].astype(np.int64)
    split = data["split"]
    track_seed = data["track_seed"].astype(np.int64)

    if obs.shape[1] != OBS_DIM:
        raise ValueError(
            f"dataset obs {obs.shape} does not match the live contract "
            f"OBS_DIM={OBS_DIM}"
        )

    # Cross-check the sidecar's codec version (fail fast on drift).
    meta_path = os.path.splitext(path)[0] + ".value.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("codec_version") != CODEC_VERSION:
            raise ValueError(
                f"dataset codec_version {meta.get('codec_version')} != live "
                f"CODEC_VERSION {CODEC_VERSION}; regenerate the dataset"
            )

    if rounds_remaining.min() < 0:
        raise ValueError(
            f"{int((rounds_remaining < 0).sum())} rows have negative "
            "rounds_remaining (corrupt dataset)"
        )

    return {
        "obs": obs,
        "rounds_remaining": rounds_remaining,
        "corner_limit": corner_limit,
        "split": split,
        "track_seed": track_seed,
    }


def _split_indices(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for the train / val splits (track-disjoint by construction)."""
    train = np.flatnonzero(split == b"train")
    val = np.flatnonzero(split == b"val")
    return train, val


def _critic_parameters(policy) -> list[nn.Parameter]:
    """Return ONLY the critic-side parameters (the actor is frozen).

    With ``share_features_extractor=False`` the critic owns a disjoint set of
    modules: its own features extractor (``vf_features_extractor``), the value
    half of the shared-arch ``mlp_extractor`` (``mlp_extractor.value_net``), and
    the value head (``value_net``). These are byte-disjoint from the actor's
    (``pi_features_extractor`` / ``mlp_extractor.policy_net`` / ``action_net``),
    so optimizing only this list leaves the actor at initialization.
    """
    modules = [
        policy.vf_features_extractor,
        policy.mlp_extractor.value_net,
        policy.value_net,
    ]
    params: list[nn.Parameter] = []
    for m in modules:
        params += list(m.parameters())
    return params


def _predict_values(policy, obs: torch.Tensor) -> torch.Tensor:
    """Run the critic forward, returning a flat ``(B,)`` value tensor.

    ``policy.predict_values`` is the EXACT hook A2's leaf calls; it extracts
    vf-features through the critic extractor, runs the critic MLP, and applies the
    value head, returning ``(B, 1)``. We flatten to ``(B,)`` for the regression
    loss.
    """
    return policy.predict_values(obs).reshape(-1)


def _evaluate(
    policy,
    obs: torch.Tensor,
    target: torch.Tensor,
    rounds_remaining: np.ndarray,
    corner_limit: np.ndarray,
    device: str,
    loss_fn,
    batch: int = 4096,
) -> dict:
    """Compute MSE + MAE (overall, per-corner-limit, per-distance-bin), no grad.

    Predictions are in *negated-rounds* units (the training target); we negate back
    to predicted-rounds-remaining for the MAE/calibration report so the numbers are
    human-readable in the quantity V actually estimates.
    """
    policy.set_training_mode(False)
    n = obs.shape[0]
    preds = np.empty(n, dtype=np.float64)
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            ob = obs[sl].to(device)
            tg = target[sl].to(device)
            pv = _predict_values(policy, ob)
            total_loss += float(loss_fn(pv, tg).sum())
            preds[sl] = pv.cpu().numpy().astype(np.float64)

    # Predicted rounds-remaining = -prediction (target was -rounds_remaining).
    pred_rounds = -preds
    abs_err = np.abs(pred_rounds - rounds_remaining.astype(np.float64))

    # Per-corner-limit MAE (limit-1 is the high-variance slice called out).
    by_limit: dict[int, dict] = {}
    for lim in sorted(set(int(x) for x in corner_limit)):
        sel = corner_limit == lim
        by_limit[lim] = {"mae": float(abs_err[sel].mean()), "n": int(sel.sum())}

    # Per-distance-to-finish-bin MAE + mean predicted rounds (the calibration
    # signal: mean predicted rounds-remaining should fall as the bin nears 0).
    by_dist: list[dict] = []
    for lo, hi in _DIST_BINS:
        sel = (rounds_remaining >= lo) & (rounds_remaining < hi)
        if not sel.any():
            by_dist.append({"lo": lo, "hi": hi, "n": 0,
                            "mae": float("nan"), "pred_mean": float("nan")})
            continue
        by_dist.append({
            "lo": lo, "hi": hi, "n": int(sel.sum()),
            "mae": float(abs_err[sel].mean()),
            "pred_mean": float(pred_rounds[sel].mean()),
        })

    return {
        "mse": total_loss / n,
        "mae": float(abs_err.mean()),
        "by_limit": by_limit,
        "by_dist": by_dist,
    }


def _calibration_monotone(by_dist: list[dict]) -> bool:
    """True if mean predicted rounds-remaining decreases as dist-to-finish does.

    Walks the distance bins from far-from-finish to near-finish (``_DIST_BINS`` is
    ordered near->far, so we reverse) and checks the mean prediction is
    non-increasing -- the README A `4` Rung-0 calibration sanity check. Empty bins
    are skipped.
    """
    means = [b["pred_mean"] for b in reversed(by_dist)
             if b["n"] > 0 and b["pred_mean"] == b["pred_mean"]]
    return all(a >= b - 1e-6 for a, b in zip(means, means[1:]))


def train_value(args: argparse.Namespace) -> dict:
    """Train the critic-only value net and save it as a MaskablePPO checkpoint."""
    ds = _load_dataset(args.data)
    train_idx, val_idx = _split_indices(ds["split"])
    if len(train_idx) == 0:
        raise RuntimeError("no training rows in dataset")

    device = resolve_device(args.device)

    # Build the architecture A2 / C expect. share_features_extractor=False so the
    # critic owns its trunk. The env only supplies SB3 the obs/action spaces; we
    # never step it (value regression is offline).
    cfg = PPOConfig(
        device=args.device, seed=args.seed, share_features_extractor=False
    )
    if args.net != "default":
        # net_profile_config replaces only the net-size fields, preserving
        # share_features_extractor=False from the base cfg above.
        cfg = net_profile_config(args.net, cfg)
    env = HeatEnv(num_players=2)
    model = build_model(env, cfg)
    policy = model.policy
    policy.to(device)

    # Target = -rounds_remaining (negated so larger == better, matching the leaf
    # sign -- README A 2.1).
    obs = torch.as_tensor(ds["obs"])
    target = torch.as_tensor(-ds["rounds_remaining"])  # (N,)
    rounds_remaining = ds["rounds_remaining"]
    corner_limit = ds["corner_limit"]

    tr_obs, tr_tgt = obs[train_idx], target[train_idx]
    va_obs, va_tgt = obs[val_idx], target[val_idx]
    tr_rr, tr_cl = rounds_remaining[train_idx], corner_limit[train_idx]
    va_rr, va_cl = rounds_remaining[val_idx], corner_limit[val_idx]

    loss_fn = (nn.SmoothL1Loss(reduction="none") if args.loss == "huber"
               else nn.MSELoss(reduction="none"))

    # Optimize ONLY the critic params -> the actor stays at init.
    critic_params = _critic_parameters(policy)
    optimizer = torch.optim.Adam(critic_params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    n_train = len(train_idx)
    have_val = len(val_idx) > 0

    # Track the best-val (lowest val MSE) checkpoint and restore it before saving,
    # mirroring train_bc's best-val selection. --patience 0/>= epochs disables
    # early stopping but still restores the best-val weights.
    best_state = None
    best_mse = float("inf")
    best_epoch = -1
    best_val: dict = {}
    epochs_since_improve = 0
    early_stop_enabled = have_val and 0 < args.patience < args.epochs
    if not have_val:
        print("WARNING: empty val set -- early stopping disabled; saving "
              "final-epoch weights.")

    history = []
    stopped_early = False
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n_train, args.batch):
            sel = order[start : start + args.batch]
            ob = tr_obs[sel].to(device)
            tg = tr_tgt[sel].to(device)
            pv = _predict_values(policy, ob)
            loss = loss_fn(pv, tg).mean()
            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(sel)
            seen += len(sel)
        train_loss = epoch_loss / max(1, seen)

        va = None
        if have_val:
            va = _evaluate(policy, va_obs, va_tgt, va_rr, va_cl, device, loss_fn)
            if va["mse"] < best_mse:
                best_mse = va["mse"]
                best_epoch = epoch + 1
                best_val = va
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                )
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        is_print_epoch = (
            (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        )
        if is_print_epoch:
            tr = _evaluate(policy, tr_obs, tr_tgt, tr_rr, tr_cl, device, loss_fn)
            va_print = va if va is not None else {
                "mse": float("nan"), "mae": float("nan"),
                "by_limit": {}, "by_dist": [],
            }
            history.append({"epoch": epoch + 1, "train": tr, "val": va_print})
            print(
                f"epoch {epoch + 1:3d}  "
                f"train mse={tr['mse']:.4f} mae={tr['mae']:.3f}  |  "
                f"val mse={va_print['mse']:.4f} mae={va_print['mae']:.3f}"
            )
            _print_buckets(va_print)

        if early_stop_enabled and epochs_since_improve >= args.patience:
            stopped_early = True
            print(
                f"early stop at epoch {epoch + 1}: val MSE has not improved for "
                f"{args.patience} epochs (best epoch {best_epoch}, "
                f"best val MSE={best_mse:.4f})"
            )
            break

    elapsed = time.perf_counter() - t0

    # Restore best-val weights so the SAVED checkpoint is the best-val one.
    if best_state is not None:
        policy.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"restored best-val weights from epoch {best_epoch} "
              f"(val MSE={best_mse:.4f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sidecar = save_checkpoint(
        model,
        args.out,
        track_name="generated",
        num_players=1,
        seed=args.seed,
        ppo_config=cfg,
    )

    final = history[-1] if history else {}
    summary = {
        "out": args.out,
        "sidecar": sidecar,
        "device": device,
        "n_train": n_train,
        "n_val": len(val_idx),
        "epochs": args.epochs,
        "loss": args.loss,
        "elapsed_s": round(elapsed, 1),
        "final": final,
        "best_epoch": best_epoch,
        "best_mse": best_mse if best_mse != float("inf") else float("nan"),
        "best_val": best_val,
        "stopped_early": stopped_early,
        "calibration_monotone": (
            _calibration_monotone(best_val["by_dist"]) if best_val else None
        ),
    }
    return summary


def _print_buckets(va: dict) -> None:
    """Print val MAE bucketed by corner limit and by distance-to-finish."""
    by_limit = va.get("by_limit") or {}
    if by_limit:
        print("    val MAE by corner speed-limit:")
        for lim in sorted(by_limit):
            tag = "  <-- limit-1 (high-variance slice)" if lim == 1 else ""
            label = "no-corner" if lim == 0 else f"limit-{lim}"
            b = by_limit[lim]
            print(f"      {label:<10} mae={b['mae']:.3f}  (n={b['n']}){tag}")
    by_dist = va.get("by_dist") or []
    if by_dist:
        print("    val MAE by rounds-to-finish bin (calibration: pred_mean falls"
              " toward finish):")
        for b in by_dist:
            if b["n"] == 0:
                continue
            hi = "inf" if b["hi"] >= 9999 else str(b["hi"])
            print(
                f"      [{b['lo']:>2},{hi:>3})  mae={b['mae']:.3f}  "
                f"pred_mean={b['pred_mean']:.2f}  (n={b['n']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/value_data.npz",
                        help="value .npz from gen_value_data.py")
    parser.add_argument("--out", type=str, default="checkpoints/value.zip",
                        help="output SB3 checkpoint path (sidecar alongside)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "huber"],
                        help="regression loss on the value head")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="print train/val metrics every N epochs (the val "
                             "MSE for selection is computed EVERY epoch)")
    parser.add_argument("--patience", type=int, default=8,
                        help="early-stop if val MSE has not improved for N "
                             "epochs; 0 or >= epochs disables (still restores "
                             "best-val)")
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"],
                        help="net size profile (default = PPOConfig defaults)")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cuda|cpu (resolve_device handles fallback)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"train_value: data={args.data} epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} loss={args.loss} net={args.net} (codec v{CODEC_VERSION})"
    )
    summary = train_value(args)
    print("\n=== train_value summary ===")
    print(f"  saved {summary['out']} (sidecar {summary['sidecar']})")
    print(f"  device={summary['device']} epochs={summary['epochs']} "
          f"loss={summary['loss']} time={summary['elapsed_s']}s")
    print(f"  train rows={summary['n_train']} val rows={summary['n_val']}")
    if summary.get("best_epoch", -1) >= 0:
        print(
            f"  SELECTED best epoch {summary['best_epoch']} by val MSE="
            f"{summary['best_mse']:.4f}"
            f"{' (stopped early)' if summary.get('stopped_early') else ''}"
        )
        bv = summary.get("best_val", {})
        if bv:
            print(f"  BEST-VAL  mse={bv['mse']:.4f} mae={bv['mae']:.3f}")
            _print_buckets(bv)
        mono = summary.get("calibration_monotone")
        if mono is not None:
            print(f"  CALIBRATION monotone-by-distance: {mono}")
    else:
        print("  SELECTED final-epoch weights (no val set)")


if __name__ == "__main__":
    from _runlog import run_main

    run_main("train_value", main)
