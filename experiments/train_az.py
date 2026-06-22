"""Sprint C2 -- AlphaZero policy+value supervised trainer (one generation).

Trains a ``MaskablePPO`` toward the self-play targets ``experiments/gen_selfplay.py``
produced: the MCTS **visit distribution** ``π`` (soft policy target) and the
Monte-Carlo **value target** ``z = −rounds_remaining`` (pre-spin-floored). The
loss is the AlphaZero objective (README §4.6):

    L = CE(π, masked policy logits) + c_v · MSE(z, value head) + wd · ‖θ‖²

trained over **both** the actor trunk/head AND the critic (unlike S3 BC, which
left the critic uninitialized -- here V is a first-class target, exactly what
makes the trained net a useful search *leaf*, not just a prior).

Why this is NOT PPO (and the symmetry with train_bc / train_value)
------------------------------------------------------------------
``train_bc`` supervises only the actor (hard-label masked CE); ``train_value``
supervises only the critic (MSE on the MC return). This trainer is their union: a
hand-rolled supervised Adam loop over ``model.policy`` minimizing the joint AZ
loss. We still build a real :class:`sb3_contrib.MaskablePPO` via
:func:`heat.ml.model.build_model` so the weights live in the exact
``HeatMLPExtractor`` + actor/critic architecture ``MLAgent`` / the
``MCTSAgent.NetAdapter`` expect -- never ``model.learn``.

The soft-policy cross-entropy (contract-faithful)
-------------------------------------------------
The policy target is a *distribution* ``π`` (visit counts), not a one-hot label,
so CE is ``−Σ_a π_a · log p_a`` where ``p = softmax(masked logits)`` from
``policy.get_distribution(obs, action_masks=mask)`` -- the EXACT masked path
``MaskablePPO.predict`` (hence ``MLAgent`` / the NetAdapter prior) takes at
inference. Illegal actions get a ``−inf`` logit; ``π`` has zero mass there
(asserted in gen_selfplay), so the cross-entropy is well-defined over the legal
set only.

``share_features_extractor=False`` (the actor/critic decoupling)
----------------------------------------------------------------
The README C2 risk note: training actor and critic jointly can destabilize. The
separate trunks (``share_features_extractor=False``, the ``model.py`` Idea-13
default) decouple their gradients; ``c_v`` is tuned on the track-disjoint val
split to balance the two losses. We build with that default explicitly.

Track-disjoint split + reporting
---------------------------------
The dataset carries a per-row ``split`` (``train``/``val``) from track-disjoint
seed bands in ``gen_selfplay``; honored directly (no row shuffle). We report
train/val policy CE, top-1 policy agreement with ``argmax π`` (overall +
CARDS-only), and value-head MSE / MAE in rounds-remaining units -- the
calibration the rung-3 re-eval consumes.

Usage:
    python experiments/train_az.py --data data/selfplay.npz --out checkpoints/c_az.zip
    python experiments/train_az.py --data data/selfplay.npz --epochs 40 --c-v 1.0
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
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM
from heat.ml.training import save_checkpoint


_KIND_NAMES = {0: "GEAR", 1: "CARDS", 2: "REACT", 3: "SLIPSTREAM", 4: "DISCARD"}
_CARDS_KIND = 1


def _load_dataset(path: str) -> dict:
    """Load the self-play ``.npz`` and validate it against the live codec.

    Fails fast (like ``train_bc`` / ``train_value`` / ``MLAgent._validate_meta``)
    if the dataset was generated against a different OBS_DIM / ACTION_DIM /
    codec_version, so a stale dataset cannot silently train a garbage net.
    """
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    pi = data["pi"].astype(np.float32)
    mask = data["mask"].astype(bool)
    z = data["z"].astype(np.float32)
    kind = data["kind"].astype(np.int64)
    split = data["split"]
    track_seed = data["track_seed"].astype(np.int64)

    if obs.shape[1] != OBS_DIM or mask.shape[1] != ACTION_DIM or pi.shape[1] != ACTION_DIM:
        raise ValueError(
            f"dataset shapes obs{obs.shape}/pi{pi.shape}/mask{mask.shape} do not "
            f"match the live contract OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM}"
        )

    meta_path = os.path.splitext(path)[0] + ".selfplay.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("codec_version") != CODEC_VERSION:
            raise ValueError(
                f"dataset codec_version {meta.get('codec_version')} != live "
                f"CODEC_VERSION {CODEC_VERSION}; regenerate self-play data"
            )

    # Every π must be a proper distribution whose support is within the mask.
    sums = pi.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        bad = int((np.abs(sums - 1.0) > 1e-4).sum())
        raise ValueError(f"{bad} rows have a pi that does not sum to 1")
    support_in_mask = ((pi > 0) & ~mask).sum()
    if support_in_mask:
        raise ValueError(f"{int(support_in_mask)} pi entries lie outside the mask")
    if z.max() > 1e-6:
        raise ValueError(
            f"{int((z > 1e-6).sum())} rows have positive z (z must be "
            "-rounds_remaining <= 0)"
        )

    return {
        "obs": obs, "pi": pi, "mask": mask, "z": z,
        "kind": kind, "split": split, "track_seed": track_seed,
    }


def _split_indices(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for the train / val splits (track-disjoint by construction)."""
    train = np.flatnonzero(split == b"train")
    val = np.flatnonzero(split == b"val")
    return train, val


def _masked_log_probs(policy, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return ``log p`` over the full action space under the masked policy.

    Uses ``policy.get_distribution(obs, action_masks=mask)`` -- the EXACT masked
    path ``MaskablePPO.predict`` / ``MLAgent`` / the NetAdapter prior take. The
    masked categorical's ``logits`` already have illegal actions at ``−inf``;
    ``log_softmax`` over them yields ``−inf`` log-probs there (π is zero there, so
    the CE term ``π·log p`` is ``0·−inf`` -> handled as 0 below).
    """
    dist = policy.get_distribution(obs, action_masks=mask.cpu().numpy())
    logits = dist.distribution.logits  # (B, ACTION_DIM), illegal -> -inf
    return torch.log_softmax(logits, dim=1)


def _policy_ce(log_probs: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy ``−Σ_a π_a log p_a`` per sample (B,).

    ``π`` is zero on illegal actions (where ``log p = −inf``); ``0 · −inf`` is NaN
    in IEEE, so we mask the product to the support of ``π`` (π>0) before summing --
    the mathematically-correct ``Σ_{a: π_a>0} π_a log p_a``.
    """
    term = pi * log_probs
    term = torch.where(pi > 0, term, torch.zeros_like(term))
    return -term.sum(dim=1)


def _evaluate(
    policy,
    obs: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor,
    z: torch.Tensor,
    kind: np.ndarray,
    device: str,
    batch: int = 4096,
) -> dict:
    """Policy CE + top-1 agreement (overall & CARDS) + value MSE/MAE, no grad.

    Top-1 agreement compares the policy argmax (the action ``MLAgent`` would emit)
    against ``argmax π`` (the search's most-visited / acted target) -- the analog
    of BC accuracy, but against the visit target rather than a hard label.
    """
    policy.set_training_mode(False)
    n = obs.shape[0]
    total_ce = 0.0
    total_correct = 0
    cards_correct = 0
    cards_total = 0
    total_v_se = 0.0
    pred_rounds = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            ob = obs[sl].to(device)
            pt = pi[sl].to(device)
            mk = mask[sl].to(device)
            zt = z[sl].to(device)
            log_probs = _masked_log_probs(policy, ob, mk)
            ce = _policy_ce(log_probs, pt)
            total_ce += float(ce.sum())
            pred = log_probs.argmax(dim=1)
            tgt = pt.argmax(dim=1)
            correct = pred == tgt
            total_correct += int(correct.sum())
            kslice = kind[sl]
            is_cards = torch.as_tensor(kslice == _CARDS_KIND, device=device)
            cards_correct += int((correct & is_cards).sum())
            cards_total += int(is_cards.sum())
            v = policy.predict_values(ob).reshape(-1)
            total_v_se += float(((v - zt) ** 2).sum())
            pred_rounds[sl] = (-v).cpu().numpy().astype(np.float64)

    rounds_remaining = (-z.cpu().numpy()).astype(np.float64)
    v_mae_rounds = float(np.abs(pred_rounds - rounds_remaining).mean())
    return {
        "ce": total_ce / n,
        "acc": total_correct / n,
        "cards_acc": (cards_correct / cards_total) if cards_total else float("nan"),
        "cards_n": cards_total,
        "v_mse": total_v_se / n,
        "v_mae_rounds": v_mae_rounds,
    }


def train_az(args: argparse.Namespace) -> dict:
    """Train the AZ policy+value net and save it as a MaskablePPO checkpoint."""
    ds = _load_dataset(args.data)
    train_idx, val_idx = _split_indices(ds["split"])
    if len(train_idx) == 0:
        raise RuntimeError("no training rows in dataset")

    device = resolve_device(args.device)

    # Build the architecture MLAgent / the NetAdapter expect. The separate
    # actor/critic trunks (share_features_extractor=False) decouple the two losses'
    # gradients (the README C2 risk mitigation). The env only supplies the
    # obs/action spaces; we never step it (AZ training is offline).
    cfg = PPOConfig(
        device=args.device, seed=args.seed, share_features_extractor=False
    )
    if args.net != "default":
        cfg = net_profile_config(args.net, cfg)
    env = HeatEnv(num_players=1)
    model = build_model(env, cfg)
    policy = model.policy
    policy.to(device)

    obs = torch.as_tensor(ds["obs"])
    pi = torch.as_tensor(ds["pi"])
    mask = torch.as_tensor(ds["mask"])
    z = torch.as_tensor(ds["z"])
    kind = ds["kind"]

    tr = train_idx
    va = val_idx
    tr_obs, tr_pi, tr_mask, tr_z, tr_kind = obs[tr], pi[tr], mask[tr], z[tr], kind[tr]
    va_obs, va_pi, va_mask, va_z, va_kind = obs[va], pi[va], mask[va], z[va], kind[va]

    # weight decay (the wd·‖θ‖² term) folded into Adam.
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    n_train = len(tr)
    have_val = len(va) > 0

    best_state = None
    best_metric = float("inf")  # selecting on val total loss (CE + c_v·MSE)
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
            pt = tr_pi[sel].to(device)
            mk = tr_mask[sel].to(device)
            zt = tr_z[sel].to(device)

            log_probs = _masked_log_probs(policy, ob, mk)
            ce = _policy_ce(log_probs, pt).mean()
            v = policy.predict_values(ob).reshape(-1)
            mse = ((v - zt) ** 2).mean()
            loss = ce + args.c_v * mse

            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(sel)
            seen += len(sel)

        va_metrics = None
        if have_val:
            va_metrics = _evaluate(
                policy, va_obs, va_pi, va_mask, va_z, va_kind, device
            )
            candidate = va_metrics["ce"] + args.c_v * va_metrics["v_mse"]
            if candidate < best_metric:
                best_metric = candidate
                best_epoch = epoch + 1
                best_val = va_metrics
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                )
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        is_print_epoch = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if is_print_epoch:
            tr_m = _evaluate(policy, tr_obs, tr_pi, tr_mask, tr_z, tr_kind, device)
            vp = va_metrics if va_metrics is not None else {
                "ce": float("nan"), "acc": float("nan"), "cards_acc": float("nan"),
                "cards_n": 0, "v_mse": float("nan"), "v_mae_rounds": float("nan"),
            }
            history.append({"epoch": epoch + 1, "train": tr_m, "val": vp})
            print(
                f"epoch {epoch + 1:3d}  "
                f"train ce={tr_m['ce']:.4f} acc={tr_m['acc']:.3f} "
                f"vmse={tr_m['v_mse']:.3f}  |  "
                f"val ce={vp['ce']:.4f} acc={vp['acc']:.3f} "
                f"cards_acc={vp['cards_acc']:.3f} vmse={vp['v_mse']:.3f} "
                f"vmae_rounds={vp['v_mae_rounds']:.2f}"
            )

        if early_stop_enabled and epochs_since_improve >= args.patience:
            stopped_early = True
            print(
                f"early stop at epoch {epoch + 1}: val loss has not improved for "
                f"{args.patience} epochs (best epoch {best_epoch}, "
                f"best val loss={best_metric:.4f})"
            )
            break

    elapsed = time.perf_counter() - t0

    if best_state is not None:
        policy.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"restored best-val weights from epoch {best_epoch} "
              f"(val loss={best_metric:.4f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sidecar = save_checkpoint(
        model,
        args.out,
        track_name="generated-az",
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
        "n_val": len(va),
        "epochs": args.epochs,
        "c_v": args.c_v,
        "weight_decay": args.weight_decay,
        "elapsed_s": round(elapsed, 1),
        "final": final,
        "best_epoch": best_epoch,
        "best_metric": best_metric if best_metric != float("inf") else float("nan"),
        "best_val": best_val,
        "stopped_early": stopped_early,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/selfplay.npz",
                        help="self-play .npz from gen_selfplay.py")
    parser.add_argument("--out", type=str, default="checkpoints/c_az.zip",
                        help="output SB3 checkpoint path (sidecar alongside)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--c-v", type=float, default=1.0,
                        help="value-loss weight c_v in L=CE+c_v*MSE (tune on val)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight decay (the wd*||theta||^2 term), folded into Adam")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8,
                        help="early-stop if val loss has not improved for N epochs; "
                             "0 or >= epochs disables (still restores best-val)")
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"])
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cuda|cpu (resolve_device handles fallback)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"train_az: data={args.data} epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} c_v={args.c_v} wd={args.weight_decay} net={args.net} "
        f"(codec v{CODEC_VERSION})"
    )
    summary = train_az(args)
    print("\n=== train_az summary ===")
    print(f"  saved {summary['out']} (sidecar {summary['sidecar']})")
    print(f"  device={summary['device']} epochs={summary['epochs']} "
          f"c_v={summary['c_v']} time={summary['elapsed_s']}s")
    print(f"  train rows={summary['n_train']} val rows={summary['n_val']}")
    if summary.get("best_epoch", -1) >= 0:
        bv = summary.get("best_val", {})
        print(
            f"  SELECTED best epoch {summary['best_epoch']} by val loss="
            f"{summary['best_metric']:.4f}"
            f"{' (stopped early)' if summary.get('stopped_early') else ''}"
        )
        if bv:
            print(
                f"  BEST-VAL  ce={bv['ce']:.4f} acc={bv['acc']:.3f} "
                f"cards_acc={bv['cards_acc']:.3f} (cards_n={bv['cards_n']}) "
                f"v_mse={bv['v_mse']:.4f} v_mae_rounds={bv['v_mae_rounds']:.2f}"
            )
    else:
        print("  SELECTED final-epoch weights (no val set)")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from _runlog import run_main

    run_main("train_az", main)
