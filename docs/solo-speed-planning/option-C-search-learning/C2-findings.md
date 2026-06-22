# Sprint C2 — Findings: one generation of self-play targets + AZ train, rung-3 re-eval

> **Type:** design-gate outcome (mirrors the C0-findings style).
> **Deliverables exercised:** `experiments/gen_selfplay.py` (target generator),
> `experiments/train_az.py` (policy+value trainer), `experiments/eval_mcts.py
> --compare-model` (the new rung-3 head-to-head mode).
> **Reproduce:**
> ```
> # 1. self-play targets from the cold-start (pre-training) prior
> PYTHONPATH=src python experiments/gen_selfplay.py \
>     --tracks 60 --val-tracks 15 --sims 16 \
>     --model checkpoints/c2_cold_prior.zip --out data/selfplay.npz
> # 2. one AZ train generation (CE policy + MSE value + wd)
> PYTHONPATH=src python experiments/train_az.py \
>     --data data/selfplay.npz --out checkpoints/c2_az.zip --epochs 40 --c-v 1.0
> # 3. rung-3 re-eval: trained-net-in-search vs pre-training-net-in-search
> PYTHONPATH=src python experiments/eval_mcts.py --games 24 --sims 16 --skip-4p \
>     --model checkpoints/c2_az.zip --compare-model checkpoints/c2_cold_prior.zip
> ```
> Numbers below are from one full run on this box (Windows 11, RTX 4080, torch
> cu126; the trainer ran on **CPU** — the dataset is tiny and the bottleneck is
> generation/eval clone cost, not the forward, exactly as C0 found).
>
> **Caveat (read first):** the pre-training prior is the **cold-start random-weight
> codec-v3 net** (`checkpoints/c2_cold_prior.zip`). No *trained* v3 checkpoint
> existed before C2 (the codec bumped 2→3 and rejected the S3 BC net), so the
> cold-start net is the only legitimate pre-training baseline available. As the C2
> design anticipated, a random prior produces near-collapsed visit targets and
> noisy value targets, so "beating" it is a low and somewhat degenerate bar — the
> result is reported honestly as a **go/no-go signal for C3**, not masked.

---

## TL;DR — rung-3 NOT MET (and that is the honest, expected signal)

**The trained net did NOT beat the pre-training (cold-start) net on either rung-2
metric.** Worst-case L1 spins/pass and rounds-to-finish are both *worse or
equal* for the trained net than for the cold prior. **Both** MCTS contenders are
catastrophically far from the rung-1 `LookaheadAgent` bar (≈150/126 rounds vs
**19.3**; L1 spins/pass ≈15–27 vs **0.045**). One generate→train→evaluate cycle
off a cold-start prior produced **no improvement signal** — exactly the outcome
README §5 / the C2 design flags as the trigger for C3's "the prior must be a
trained-warm net and/or the C0 visit budget is too low" branch (and the Gumbel
seam-pull). The pipeline is correct end-to-end; the *signal* is absent because the
data the cold prior generates is too poor to learn a better prior+value from in one
shot.

---

## Config used

| Stage | Setting | Value |
|---|---|---|
| **generation** | tracks (train/val) | **60 / 15** (track-disjoint, self-play band base `300_000`) |
| | sims | **16** (C0 budget) |
| | Dirichlet eps / alpha | **0.25 / 0.5** (C0) |
| | temperature_moves | **10** (tau=1 window, then tau→0) |
| | prior (pre-training net) | `checkpoints/c2_cold_prior.zip` (cold-start, random-weight, codec v3) |
| | output | `data/selfplay.npz` (+ `.selfplay.json` sidecar, codec v3) |
| **dataset** | rows logged | **11,607** (train 8,902 / val 2,705) |
| | races / dropped (MAX_ROUNDS) | 75 / **15** → **7,181 rows dropped** (cold prior spin-spirals to MAX_ROUNDS) |
| | off-table acted moves dropped | 0 |
| | kind balance (train) | GEAR 31.6%, REACT 30.7%, CARDS 20.5%, DISCARD 17.1% (SLIPSTREAM ~0 in solo) |
| | pi entropy (overall / cards) | **0.039 / 0.060** — near-collapsed (the C0 visit-budget go/no-go surfacing) |
| | z (−rounds_remaining, floored) | mean −27.7, min −145 |
| **train** | epochs / batch / lr / wd | 40 (early-stopped epoch 20) / 256 / 3e-4 / 1e-4 |
| | c_v (value-loss weight) | **1.0** (tuned on val — see below) |
| | device | cpu (6.9 s) |

### c_v tuning on the val split

| c_v | best epoch | val CE | val acc | val cards_acc | val v_mse | **val v_MAE (rounds)** |
|---|---|---|---|---|---|---|
| 0.25 | 12 | 1.0006 | 0.555 | 0.407 | 535.8 | **14.22** |
| **1.0** | 12 | 1.0167 | 0.554 | 0.400 | 536.4 | **14.24** |

`c_v` barely moves the outcome: the value head plateaus at the **same ~14-rounds
MAE** regardless of the loss balance, because the *targets themselves* are noisy
(cold-prior, spin-floored returns spanning −1…−145). A value head off by ~14 rounds
is not a useful search leaf. `c_v=1.0` (the design default, `checkpoints/c2_az.zip`)
was used for the rung-3 eval; `c_v=0.25` (`checkpoints/c2_az_cv025.zip`) is kept for
reference and is statistically indistinguishable.

---

## Rung-3 table (held-out solo, 24 tracks, `_HELDOUT_BASE = 900_000` band, sims=16)

| agent | finish% | rounds | L1 spins/pass (mn/p90/**mx**) | dist/heat | cooldowns/game | ms/move |
|---|---|---|---|---|---|---|
| Heuristic (rung 0) | 100% | 26.2 | 0.465 / 1.50 / **1.50** | 14.19 | 9.54 | — |
| **Lookahead (rung 1)** | 100% | **19.3** | 0.045 / 0.25 / **0.50** | 15.07 | 10.83 | 4.07 |
| MCTS-trained (`c2_az`) | 100% | 150.6 | 14.93 / 9.71 / **192.0** | 86.83 | **0.00** | 8.94 |
| MCTS-prior (`c2_cold_prior`) | 100% | 126.5 | 26.86 / 92.5 / **173.0** | 18.06 | 19.58 | 8.07 |

**Rung-3 verdict (the only one that matters for C2):**

| criterion | trained | prior | result |
|---|---|---|---|
| worst-case L1 spins/pass (lower = better) | 192.0 | 173.0 | **FAIL** (trained higher) |
| rounds-to-finish (lower = better) | 150.6 | 126.5 | **FAIL** (trained higher) |

**VERDICT: rung-3 NOT MET.** The trained net did not beat the pre-training net on
**either** metric, let alone both.

---

## Honest reading (do NOT paper over with finish-rate — the S3/8C footgun)

- **Finish-rate is 100% for every agent and is meaningless here.** A policy that
  crawls and spins still "finishes" — at **150 rounds** vs Lookahead's 19. Per the
  README this is exactly the number we must NOT lead with. The load-bearing numbers
  (worst-case L1 spins/pass, rounds) say both search agents are far below rung 1.
- **The trained net's `dist/heat=86.8` with `cooldowns/game=0.0` is a tell, not a
  win.** That is not "better heat budgeting"; it is a **degenerate never-spend-heat
  policy** the miscalibrated value head (≈14-rounds MAE) drives — which is precisely
  why it racks up **192 spins at a single L1 corner**. High dist/heat here is the
  symptom of the failure, not evidence against it.
- **Why no signal:** the cold-start prior produces (a) **collapsed visit targets**
  (pi entropy 0.039 ≈ argmax — barely better than BC, with BC's gap) and (b) **noisy
  MC value targets** dominated by spin-spirals (7,181 of ~18.8k candidate rows
  dropped to MAX_ROUNDS truncation; the floored returns still span −1…−145). One CE
  pass over a near-one-hot target plus one MSE pass over a high-variance target
  cannot manufacture a better-than-net prior+value. This is the C0/C2-flagged
  failure mode, observed.

## Go / no-go for C3 (this is the real deliverable)

The pipeline is **built, correct, and contract-checked** (see the unit suite:
`tests/test_az_targets.py`, `tests/test_train_az.py` — 26 tests green, including the
single-most-important pre-spin-floor check on a forced-spin track). The missing
piece is the **improvement signal**, and C2's value is telling C3 *why* it is
missing and what to change:

1. **The prior must be a trained-warm net, not a cold-start random net.** The
   biggest lever: re-mint the pre-training prior as a *trained* v3 net (a small
   BC/value net regenerated under codec v3, or an A/B value warm-start — the targets
   are definitionally identical, per README §4.6) so the self-play data carries
   actual driving skill and a non-degenerate visit distribution. C2 deliberately
   could not do this (no trained v3 checkpoint existed); it is C3's first move.
2. **The C0 visit budget (16 sims) is likely too low** to produce
   better-than-net targets off a weak prior — the pi entropy of 0.039 says the
   search is barely exploring. This is the **Gumbel-seam-pull trigger** the C0
   findings reserved: Gumbel-AlphaZero's low-visit improvement guarantee is the
   lever to make 16 sims go further before paying for more clones.
3. **Loop it (C3).** One generation cannot bootstrap from a cold prior; the
   improvement is an *iterated* fixed point. C2's honest non-beat is the evidence
   that the single-shot shortcut does not work here, which is exactly what C2 was
   built to determine.

This is a legitimate, informative gate result: **C2's machinery is sound; the
single-generation-from-cold improvement signal is absent; C3 should start from a
trained-warm prior and pull Gumbel forward.** Falling back to the A/B/E spine
options stays on the table if a warm-prior C3 still fails to beat rung 1.
