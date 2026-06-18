# HEAT — Training Guide

This project trains a reinforcement-learning agent to play **HEAT: Pedal to the Metal**.
This guide covers how to run training with the current (Sprint 8C / 8D) workflow.

The recipe that works: **pretrain solo, then ramp up the opponents** —
solo time-trial → fine-tune vs a weak heuristic → mixed → strong heuristic.
Training against strong opponents from scratch collapses; this curriculum does not.

---

## 1. Setup

You need Python 3.10+.

**CPU-only (works everywhere):**

```bash
pip install -e ".[ml]"
```

**GPU (recommended for real runs):** install a CUDA torch wheel *first*, then the ML extra.

```bash
# pick the wheel that matches your GPU's CUDA version (e.g. cu124, cu126)
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install -e ".[ml]"
```

> An RTX 4080 needs the `cu126` wheel. The default PyPI torch is CPU-only — if you
> install that, training silently falls back to CPU.

All commands below set `PYTHONPATH=src` so Python can find the `heat` package.

- **bash / Linux / macOS:** `PYTHONPATH=src python <script>`
- **Windows PowerShell:** `$env:PYTHONPATH="src"; python <script>`

---

## 2. Train one agent (the main workflow)

This is the script you want most of the time. It runs the full solo → weak → mixed →
strong curriculum, keeps the **best** held-out checkpoint, and prints a league ladder
of the trained policy against the reference heuristics at the end.

```bash
PYTHONPATH=src python train_8c.py
```

What happens:

1. **Solo pretrain** — the agent learns to drive a clean lap on its own (`gamma 0.99`).
2. **Opponent ramp** — fine-tune vs weak → mixed → strong heuristics (`gamma 0.999`).
   Each opponent phase is gated on the Wilson lower-bound of held-out games, so only
   genuine improvements are kept.
3. **League ladder** — the best checkpoint is scored free-for-all (seat-order cancelled)
   on a fixed held-out track set vs `strong2`, `strong3`, and the weak heuristic,
   printed as TrueSkill + ELO.

The best checkpoint path is printed as `best checkpoint -> ...` when training finishes.

---

## 3. Sweep many configs + rank them (Sprint 8D)

If you want to compare hyperparameters rather than train a single agent, use the
mass-training campaign. **This is heavy compute** (~8–12 GPU-hours for the full grid)
and is not run by the tests.

**Step 1 — run the campaign.** Trains every config in the sweep sequentially, isolates
per-run failures, and writes a resumable `manifest.jsonl`.

```bash
PYTHONPATH=src python experiments/run_campaign.py --out-dir campaign_out
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--method random` / `--method lhs` | sample the grid instead of running all of it |
| `--max-configs N` | cap the number of runs (cheaper first pass) |
| `--no-resume` | start fresh instead of skipping already-`done` runs |

Re-launching with the same spec **resumes** — runs already marked `done` are skipped.

**Step 2 — rank the results.** Builds a large round-robin league over every checkpoint
plus heuristic anchors, then writes the analysis. ~0.5–1.5 GPU-hours.

```bash
PYTHONPATH=src python experiments/run_league.py --out-dir campaign_out
```

Outputs in `campaign_out/`:

- `league_games.jsonl` — every per-game result (ratings recompute without re-racing)
- `attribution.md` — which factors actually moved the rating, in prose
- `ladder.csv` — the ranked leaderboard
- `effects.json` — per-factor effect sizes

---

## 4. Earlier single-run script

`train_and_eval_run.py` is the older Sprint A launcher (Phase-1-heavy budget, then
eval vs weak/strong heuristics with a seat sweep). Kept for reference — prefer
`train_8c.py` for new runs.

---

## 5. Tests

```bash
PYTHONPATH=src python -m pytest
```

The suite is fast and does **not** launch any GPU training; the heavy campaign /
league scripts are excluded.
