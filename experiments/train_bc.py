"""Sprint S3 (BC) — supervised behavioral cloning of the search agent.

Trains a ``MaskablePPO`` policy to reproduce the S1/S2 ``LookaheadAgent``'s
decisions by minimizing *masked* cross-entropy on the demonstration dataset
produced by ``experiments/gen_demos.py``. The output is an ordinary SB3
checkpoint (zip + ``.meta.json`` sidecar) that loads as an :class:`MLAgent` and
drops into ``evaluate_ml`` / the league unchanged -- no search at inference.

Why this is NOT PPO
-------------------
PPO optimizes a clipped policy-gradient objective against *environment reward*.
Here we have *expert labels*, so the right objective is plain supervised
cross-entropy of the policy's action distribution against the demonstrated flat
action. We still build a real :class:`sb3_contrib.MaskablePPO` (via
:func:`heat.ml.model.build_model`) so the trained weights live in exactly the
architecture ``MLAgent`` expects, but the optimization is a hand-rolled
supervised loop over ``model.policy`` -- never ``model.learn``.

Masked cross-entropy (the contract-faithful loss)
--------------------------------------------------
At inference ``MLAgent`` calls ``model.predict(obs, action_masks=mask)``, which
builds a :class:`MaskableDistribution` from the actor latent and applies the
mask (illegal actions get ``-inf`` logit) before the arg-max. We train against
the *same* masked distribution: ``loss = -mean(log_prob(target))`` where
``log_prob`` comes from ``policy.get_distribution(obs, action_masks=mask)``. The
mask is the SAME bool array logged in the dataset (and the same one
``legal_action_mask`` produces at inference), so the train-time and test-time
action spaces are identical -- the expert's chosen index is always among the
legal set (``gen_demos`` asserts this), so the target is always a valid label.

Only the **actor** trunk + head are supervised. The critic (value net) is left
at initialization: ``MLAgent`` does a masked arg-max over the actor and never
touches the value head, so an uncalibrated critic does not affect the cloned
policy's play. (S4's BC->PPO fine-tune will train the critic from reward.)

Track-disjoint split + reporting
---------------------------------
The dataset carries a per-row ``split`` (``train``/``val``) computed from
track-disjoint seed bands in ``gen_demos``; we honor it directly (no row
shuffle across the boundary). We report train/val cross-entropy AND top-1 action
accuracy, both overall and **on CARDS decisions specifically** (the corner
skill: which speed to play through a tight corner lives in the CARDS choice).

Usage:
    python experiments/train_bc.py --data data/bc_demos.npz --out checkpoints/bc.zip
    python experiments/train_bc.py --data data/bc_demos.npz --epochs 40 --net large
"""

from __future__ import annotations

import argparse
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


# DecisionKind int codes as written by gen_demos (kept in sync there).
_KIND_NAMES = {0: "GEAR", 1: "CARDS", 2: "REACT", 3: "SLIPSTREAM", 4: "DISCARD"}
_CARDS_KIND = 1


def _load_dataset(path: str) -> dict:
    """Load the demonstration ``.npz`` and validate it against the live codec.

    Fails fast (like ``MLAgent._validate_meta``) if the dataset was generated
    against a different OBS_DIM / ACTION_DIM / codec_version, so a stale dataset
    cannot silently train a garbage policy.
    """
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    action = data["action"].astype(np.int64)
    mask = data["mask"].astype(bool)
    kind = data["kind"].astype(np.int64)
    split = data["split"]
    track_seed = data["track_seed"].astype(np.int64)

    if obs.shape[1] != OBS_DIM or mask.shape[1] != ACTION_DIM:
        raise ValueError(
            f"dataset shapes obs{obs.shape}/mask{mask.shape} do not match the "
            f"live contract OBS_DIM={OBS_DIM} ACTION_DIM={ACTION_DIM}"
        )

    # Cross-check the sidecar's codec version if present.
    meta_path = os.path.splitext(path)[0] + ".demos.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("codec_version") != CODEC_VERSION:
            raise ValueError(
                f"dataset codec_version {meta.get('codec_version')} != live "
                f"CODEC_VERSION {CODEC_VERSION}; regenerate demos"
            )

    # Every logged target must be legal under its own mask (gen_demos guarantees
    # this; re-assert so a corrupted dataset is caught before training).
    legal = mask[np.arange(len(action)), action]
    if not legal.all():
        bad = int((~legal).sum())
        raise ValueError(f"{bad} dataset rows have an off-mask target action")

    return {
        "obs": obs,
        "action": action,
        "mask": mask,
        "kind": kind,
        "split": split,
        "track_seed": track_seed,
    }


def _split_indices(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for the train / val splits (track-disjoint by construction)."""
    train = np.flatnonzero(split == b"train")
    val = np.flatnonzero(split == b"val")
    return train, val


def _nll(policy, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor):
    """Return ``(per_sample_nll, pred_actions)`` under the masked policy.

    Uses ``policy.get_distribution(obs, action_masks=mask)`` -- the EXACT path
    ``MaskablePPO.predict`` (and thus ``MLAgent``) takes at inference: it
    extracts pi-features through the (possibly unshared) actor extractor, runs
    the actor MLP, builds the categorical, and applies the mask (illegal actions
    get ``-inf`` logit). Supervised cross-entropy against the demonstrated action
    is then ``-log_prob(action)``; the arg-max of the (masked) distribution is
    the deterministic action ``MLAgent`` would emit, so accuracy here == the
    cloned policy's agreement with the expert at inference.
    """
    dist = policy.get_distribution(obs, action_masks=mask.cpu().numpy())
    log_prob = dist.log_prob(action)  # (B,)
    # The masked categorical's per-action logits, for the arg-max prediction.
    pred = dist.distribution.logits.argmax(dim=1)
    return -log_prob, pred


def _evaluate(
    policy,
    obs: torch.Tensor,
    action: torch.Tensor,
    mask: torch.Tensor,
    kind: np.ndarray,
    device: str,
    batch: int = 4096,
) -> dict:
    """Compute cross-entropy + top-1 accuracy (overall and CARDS-only), no grad."""
    policy.set_training_mode(False)
    n = obs.shape[0]
    total_ce = 0.0
    total_correct = 0
    cards_correct = 0
    cards_total = 0
    with torch.no_grad():
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            ob = obs[sl].to(device)
            ac = action[sl].to(device)
            mk = mask[sl].to(device)
            nll, pred = _nll(policy, ob, mk, ac)
            total_ce += float(nll.sum())
            correct = pred == ac
            total_correct += int(correct.sum())
            kslice = kind[sl]
            is_cards = torch.as_tensor(kslice == _CARDS_KIND, device=device)
            cards_correct += int((correct & is_cards).sum())
            cards_total += int(is_cards.sum())
    return {
        "ce": total_ce / n,
        "acc": total_correct / n,
        "cards_acc": (cards_correct / cards_total) if cards_total else float("nan"),
        "cards_n": cards_total,
    }


def train_bc(args: argparse.Namespace) -> dict:
    """Train the BC policy and save it as a MaskablePPO checkpoint."""
    ds = _load_dataset(args.data)
    train_idx, val_idx = _split_indices(ds["split"])
    if len(train_idx) == 0:
        raise RuntimeError("no training rows in dataset")

    device = resolve_device(args.device)

    # Build the policy architecture MLAgent expects. The env is only used to give
    # SB3 the observation/action spaces; we never step it (BC is offline).
    cfg = PPOConfig(device=args.device, seed=args.seed)
    if args.net != "default":
        cfg = net_profile_config(args.net, cfg)
    env = HeatEnv(num_players=2)
    model = build_model(env, cfg)
    policy = model.policy
    policy.to(device)

    # Tensors. Keep the full dataset on CPU and move minibatches to device.
    obs = torch.as_tensor(ds["obs"])
    action = torch.as_tensor(ds["action"])
    mask = torch.as_tensor(ds["mask"])
    kind = ds["kind"]

    tr_obs, tr_act, tr_mask = obs[train_idx], action[train_idx], mask[train_idx]
    tr_kind = kind[train_idx]
    va_obs, va_act, va_mask = obs[val_idx], action[val_idx], mask[val_idx]
    va_kind = kind[val_idx]

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n_train = len(train_idx)

    history = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        order = rng.permutation(n_train)
        epoch_ce = 0.0
        seen = 0
        for start in range(0, n_train, args.batch):
            sel = order[start : start + args.batch]
            ob = tr_obs[sel].to(device)
            ac = tr_act[sel].to(device)
            mk = tr_mask[sel].to(device)
            nll, _ = _nll(policy, ob, mk, ac)
            loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            epoch_ce += float(loss.detach()) * len(sel)
            seen += len(sel)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            tr = _evaluate(policy, tr_obs, tr_act, tr_mask, tr_kind, device)
            va = _evaluate(policy, va_obs, va_act, va_mask, va_kind, device)
            history.append({"epoch": epoch + 1, "train": tr, "val": va})
            print(
                f"epoch {epoch + 1:3d}  "
                f"train ce={tr['ce']:.4f} acc={tr['acc']:.3f} "
                f"cards_acc={tr['cards_acc']:.3f}  |  "
                f"val ce={va['ce']:.4f} acc={va['acc']:.3f} "
                f"cards_acc={va['cards_acc']:.3f}"
            )

    elapsed = time.perf_counter() - t0

    # Save as an ordinary MaskablePPO checkpoint (+ contract sidecar) so MLAgent
    # / evaluate_ml load it unchanged. No VecNormalize was used in BC, so no
    # vecnorm sidecar is written and MLAgent treats it as un-normalized.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sidecar = save_checkpoint(
        model,
        args.out,
        track_name="generated-bc",
        num_players=args.players_meta,
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
        "elapsed_s": round(elapsed, 1),
        "final": final,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/bc_demos.npz",
                        help="demonstration .npz from gen_demos.py")
    parser.add_argument("--out", type=str, default="checkpoints/bc.zip",
                        help="output SB3 checkpoint path (sidecar alongside)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="report train/val metrics every N epochs")
    parser.add_argument("--net", type=str, default="default",
                        choices=["default", "small", "large"],
                        help="net size profile (default = PPOConfig defaults)")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cuda|cpu (resolve_device handles fallback)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players-meta", type=int, default=4,
                        help="num_players recorded in the checkpoint sidecar "
                             "(metadata only; does not affect training)")
    args = parser.parse_args()

    print(
        f"train_bc: data={args.data} epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} net={args.net} (codec v{CODEC_VERSION})"
    )
    summary = train_bc(args)
    print("\n=== train_bc summary ===")
    print(f"  saved {summary['out']} (sidecar {summary['sidecar']})")
    print(f"  device={summary['device']} epochs={summary['epochs']} "
          f"time={summary['elapsed_s']}s")
    print(f"  train rows={summary['n_train']} val rows={summary['n_val']}")
    if summary["final"]:
        f = summary["final"]
        print(
            f"  FINAL  train ce={f['train']['ce']:.4f} acc={f['train']['acc']:.3f} "
            f"cards_acc={f['train']['cards_acc']:.3f}"
        )
        print(
            f"  FINAL  val   ce={f['val']['ce']:.4f} acc={f['val']['acc']:.3f} "
            f"cards_acc={f['val']['cards_acc']:.3f} (cards_n={f['val']['cards_n']})"
        )


if __name__ == "__main__":
    main()
