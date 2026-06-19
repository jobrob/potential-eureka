# Learnings: Sprint 8C/8D "collapse" investigation

**Date:** 2026-06-18/19  **Branch:** `worktree-8c-rootcause`

This document records what was *learned* while investigating why the productionized
Sprint-8C run (`train_8c.py`) produced a below-random agent. It is a findings log,
**not** a fix design. All numbers are from experiments on the current code.

---

## 1. The trigger

Before launching the 8D mass-training campaign, we ran a single full-budget
`train_8c.py` (1.6M steps: solo 300k → weak 300k → mixed 400k → strong 600k) as a
sanity check. It completed cleanly (29.7 min, healthy SB3 curves: `explained_variance`
0.6–0.8, no NaN) but the trained agent was **below random**:

| Metric (head-to-head, USA track) | BEST ckpt | FINAL ckpt |
| --- | --- | --- |
| vs weak heuristic (4p) | 5.0% | 4.0% |
| vs strong heuristic (4p) | 0.0% | 1.0% |
| (random-field baseline) | 25% | 25% |

League ladder (free-for-all on generated tracks) put `8c` dead last. Healthy training
metrics masked a useless policy — the failure was invisible in the SB3 logs.

---

## 2. Method: bisection by controlled experiment

Each step held everything constant except one variable, to localize the defect.

| # | Experiment | Result | Conclusion |
| --- | --- | --- | --- |
| 1 | Prototype `proto_solo`→`proto_finetune` rerun | solo 100% finish; after-weak 85.6% / after-strong **93.1% / 81.9%** (USA) | Shared code path is healthy |
| 2 | Gamma-handoff reload in isolation | policy weights identical (max diff 0.0), gamma flips 0.99→0.999 | The reload is **faithful** |
| 3 | Weak fine-tune forced to gamma 0.999 | 77.5% weak (USA) | Gamma value **exonerated** |
| 4 | Hand-rolled race replay from a *good* solo base (no gate) | after-weak 68%, after-mixed 91%, after-strong **91.7% / 90.8%** (USA) | Race-phase *logic* is healthy |
| 5 | Production solo phase in isolation (WITH clamp, 200k) | 57% finish / 0% transfer (USA) | Solo phase is the first culprit |
| 6 | Prototype-style solo @ matched 200k (signed progress) | **97% finish** | It is the reward, not the budget |
| 7 | Production solo phase WITH clamp fix (200k) | 67% finish / **68% transfer** (USA) | Fix #1 repairs the solo base |
| 8 | SB3 source read of `_setup_learn` / `save` | `save()` nulls `_last_obs`; `set_env` nulls it | `reset_num_timesteps=False` handoff is **correct** |
| 9 | Real `_train_phases` replica (solo200/weak100/mixed100/strong400) | BEST 25% / **FINAL 0.6%** (USA); gate ~0% throughout | Long strong phase collapses; + track confound |
| 10 | Same checkpoints on USA vs generated tracks | see §5 | **Nothing generalizes to generated** |

---

## 3. Bug #1 — solo reward clamp (CONFIRMED, already changed in this branch)

`heat.ml.spaces.step_reward` (Sprint 8C) clamped solo per-step progress to
`max(0.0, progress)` — a "solo is positive-only" Definition-of-Done. That clamp
removed the **negative reward for moving backward**, which is the signal that
teaches the car not to spin out. The prototype (`experiments/proto_solo.py`) never
clamped (it left `REWARD_MODE="race"` so progress stayed signed, plus a wrapper
finish bonus) and produced a far better driver.

Controlled ablation at matched 200k budget:

- signed progress (prototype-style): **97% solo finish-rate**
- clamped progress (production "solo" mode): **57% solo finish-rate**

A weak solo base then can't be recovered by the downstream curriculum.

**Change made in this branch:** removed the clamp in `spaces.py` (signed progress in
both modes); updated `tests/test_solo_reward.py` — the real anti-exploit invariant is
"no negative **terminal** reward", not "no negative per-step reward" (solo has no
negative terminal to escape: placement is 0 for n≤1 and the finish bonus only pays on
a real finish). Unit suite: 80 passed, 1 skipped.

**Effect of fix #1 alone** (full `train_8c`, USA): BEST 5%→**32%**, FINAL 4%→24%.
Real improvement, but far short of the prototype's USA numbers — i.e. **necessary but
not sufficient.**

---

## 4. Bug #2 — long strong phase catastrophically collapses the policy (CONFIRMED)

The faithful real-`_train_phases` replica (§2 #9) degraded a good base on USA:

```
BEST  (gate-selected, mid-run): 25.0% vs weak
FINAL (after the 400k strong):   0.6% vs weak   <- forgot everything
```

A hand-rolled replay with a *short* (200k) strong phase improved to 91% (§2 #4); the
real run with a *long* (400k/600k) strong phase collapses. The gate's best-checkpoint
selection partially mitigates (it saves a pre-collapse snapshot) but the strong-phase
training itself is destructive past some length. Consistent with this project's
long-standing "training against strong opponents collapses" theme.

Not yet localized further (gate cadence vs strong-phase length vs chunking were the
remaining differences between the working replay and the failing real run; strong-phase
length is the leading explanation but unproven in isolation).

---

## 5. Bug #3 / THE REFRAME — nothing generalizes to generated tracks

The decisive finding. `evaluate_ml` defaults to `track="usa"`; **every "good" number
in this investigation was measured on the fixed USA track.** The actual training
objective is procedurally **generated** tracks (`track=None`), which is what the gate
and league evaluate on. Evaluating the same checkpoints on both:

| checkpoint | USA: weak / strong | GENERATED: weak / strong |
| --- | --- | --- |
| `proto_final` (the "validated 93% recipe") | 93.1% / 81.9% | **0.0% / 2.5%** |
| replica BEST | 25.0% / 1.9% | 1.9% / 7.5% |
| solo base | 70.0% / 26.9% | 0.0% / 0.0% |

**The validated solo-pretrain prototype does not generalize — 0% vs weak on generated
tracks.** It overfits to USA. The 8C gate and league (correctly on generated) reported
~0% honestly the whole time; the perceived "8C collapse" was largely an artifact of
comparing USA-measured "successes" against generated-measured reality.

Notable nuance: the solo policy **drives** generated tracks fine on its own (67% solo
finish-rate on held-out generated tracks), but is **worse than random** (0% vs 25%) in
**multiplayer** racing on generated tracks. So the break is specifically
*multiplayer-on-generated*, not basic driving — the policy is actively miscalibrated
there, not merely undertrained.

This matches Sprint B's earlier observation (generated ≈40% best vs USA ≈94%); 8C's
headline "94%" quietly reverted to USA-only measurement and buried the gap.

---

## 6. What was ruled out

- The shared training/env/feature/codec path (the prototype runs through it and works
  on USA).
- Reload weight fidelity (bit-identical across the gamma handoff).
- The gamma value 0.999 (a 0.999 weak fine-tune reaches 77% on USA).
- `_last_obs` / `reset_num_timesteps=False` handoff (SB3 nulls `_last_obs` on save and
  `set_env`, so the env resets after the reload).
- Gate-induced training-env corruption (the gate saves to a temp checkpoint and evals
  with its own envs; it never steps the training venv).
- The solo phase as a generalization cause (it solo-drives generated tracks fine).

---

## 7. Measurement traps observed (record so we don't repeat them)

- **`evaluate_ml` defaults to `track="usa"`.** A USA number says nothing about the
  generated-track objective. Always state the eval track.
- **Healthy SB3 curves ≠ a good policy.** `explained_variance` stayed 0.6–0.8 through a
  run that produced a below-random agent; the value head fit the shaped reward while
  the policy was useless. A head-to-head win-rate gate is the only trustworthy signal.
- **The league/gate (generated) were the honest metrics;** the USA head-to-head numbers
  were the misleading ones.

---

## 8. Scripts / entry-point audit (state of the tree)

The **current 8C path** (`train_8c.py` → `_train_phases`) does **not** use the
flagged-harmful machinery — checkpoint metadata confirms `use_league=false`,
`normalize_obs=false`, `use_track_curriculum=false`. But that machinery still exists in
the codebase and is exposed elsewhere:

| File | Notes |
| --- | --- |
| `scripts/train_ml.py` | Legacy pre-8C entry point; exposes `--use-league` (self-play league), Phase-2 snapshots, `--normalize-obs`, track curriculum — all previously flagged as collapse-prone. Calls `train_self_play` without phases (the legacy path). Coupled to `tests/test_train_cli.py`. |
| `experiments/diag_ablate.py`, `diag_canlearn.py` | Dead Sprint A/B diagnostic one-offs; nothing imports them. (`diag_canlearn` embodies a useful idea: a fast "can it learn at all" check.) |
| `train_and_eval_run.py` (untracked, main checkout) | Old Sprint A launcher. |
| `experiments/proto_solo.py`, `proto_finetune.py` | The reference recipe — but now known to be USA-overfit (§5). |
| `scripts/evaluate_ml.py`, `run_race.py`, `run_simulation.py`, `watch_ml_race.py` | Benign utilities. |
| `experiments/run_campaign.py`, `run_league.py` | Current 8D tooling. |

---

## 9. Architecture observations (information only)

- **Reward config lives in mutable module-level globals** (`heat.ml.spaces`:
  `SHAPING_WEIGHT`, `SHAPING_PROGRESS_COEF`, `REWARD_MODE`, `SOLO_FINISH_BONUS`,
  spinout knobs). They must be re-applied inside every `SubprocVecEnv` worker *and* the
  main process. Bug #1 is an instance of this class (a reward-semantics divergence that
  was easy to introduce silently).
- **Multiple training entry points** (`train_8c.py`, `scripts/train_ml.py`,
  `train_and_eval_run.py`, the proto/diag experiments) with overlapping, differently
  defaulted config surfaces.
- **`CurriculumConfig` carries ~30 fields**, many of them the flagged-harmful levers
  (league, normalize_obs, track curriculum, snapshot self-play, PFSP), still wired into
  `train_self_play`'s legacy branch even though the 8C phase path bypasses them.
- **The gate always scores vs the strong yardstick** (`training.py` ~L1704), even during
  the weak phase, so a good-vs-weak checkpoint can be discarded for a marginally
  better-vs-strong one.

---

## 10. Open questions (for when this is picked up)

1. **Why no generalization to generated tracks (Bug #3, the real blocker)?** Candidates:
   a *features* issue (generated-track + opponent obs malformed/out-of-distribution in
   multiplayer — the solo-vs-multiplayer split points here); a *training-distribution*
   issue (is training actually exposing diverse generated tracks, or a narrow band?); or
   simply *much harder* (needs different reward / far more steps). The "worse than
   random" multiplayer-on-generated result argues for active miscalibration, not just
   undertraining.
2. **Bug #2 isolation:** is the strong-phase collapse purely a function of phase length,
   or does the gate cadence / chunking contribute?
3. **Re-baseline everything on generated tracks**, never USA, before trusting any recipe
   or launching 8D.

---

## 11. Changes already on this branch

- `src/heat/ml/spaces.py` — removed the solo progress clamp (Bug #1).
- `tests/test_solo_reward.py` — corrected the invariant to "no negative terminal reward".
- `docs/ml-learnings-8c-collapse-investigation.md` — this document.

No fixes for Bug #2 or Bug #3 were designed or applied.
