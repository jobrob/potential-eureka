# ML learnings — why generated-track training collapsed, and the recipe that fixed it

> **Status:** empirical learning log (2026-06-18). Captures the diagnosis of the
> Sprint A/B baseline collapse and the validated **solo-pretrain → opponent
> curriculum** training recipe. Intended as the reference for a follow-up sprint
> that productionizes the recipe. All numbers are from prototype runs on an
> RTX 4080 SUPER, `n_envs=8`.

---

## 1. TL;DR

- Training a HEAT policy **from scratch against strong opponents on generated
  tracks collapsed** to ~3–4% win-rate (worse than the equal-field 25%).
- Root cause is **learning-signal density, not opponent strength**: vs strong
  opponents a fresh policy loses almost every race, so the terminal placement
  reward is ~constant → no gradient → collapse. "Opponent strength" was only a
  proxy for how much usable signal the agent gets.
- **Sprint A's train-vs-strong and Sprint B's track curriculum both attacked the
  wrong axis** and made things worse. The one Sprint-A idea that helped was
  `randomize_seat`.
- The fix — **dense solo pretrain, then a curriculum on the *opponent* axis** —
  took a policy to **94% vs weak / 86% vs strong in ~16 min** of training
  (1.2M steps total), vs the ~54-min runs that collapsed.

---

## 2. Timeline / what we ran

Sprints A (trustworthy gate) and B (whole-track v2 obs + track curriculum) were
implemented and the suite was green (678 tests). The Sprint-B baseline
(`sprint_b_curriculum`, 2M steps, generated tracks + strong opponents + track
curriculum + `randomize_seat`) produced a **best checkpoint of 2.7% vs weak /
2.0% vs strong** — worse than random-field. That kicked off the diagnosis below.

---

## 3. The diagnostic ablations

All ablations: generated tracks, **400k** steps, eval vs the **weak** heuristic
on **held-out** generated tracks (point estimates, ~120 games).

| Config | vs weak | Takeaway |
|---|---:|---|
| Fixed `usa` track, weak opp, no extras (control) | **94.2%** | Core v2-obs + training + eval path is HEALTHY (matches the Sprint-5 ~94% benchmark). The collapse was never a code bug. |
| Generated tracks, weak opp, no extras (`gen_base`) | 19.2% | Generated-track *generalization* is the real difficulty (94→19). |
| `gen_base` + `randomize_seat` | **40.0%** | `randomize_seat` is a big, genuine win (+21 pts). |
| `gen_base` + track curriculum (horizon=total) | 7.5% | Track curriculum HURTS. |
| `gen_base` + track curriculum (horizon=0.4×total) | 1.7% | Retuning the horizon makes it WORSE → it's the *mechanism*, not the schedule. |
| `gen_base` + `randomize_seat` + **strong** opp | **4.2%** | Strong opponents COLLAPSE from-scratch training (40→4). |

Plus the two full 2M runs: curriculum+strong = 2.7%/2.0%; no-curriculum+strong =
6.7%/3.3%.

### Why each "improvement" backfired
- **Strong opponents (Sprint A, the dominant killer).** A from-scratch policy
  loses ~every race vs strong opponents, so the terminal placement reward is
  ~constant → zero gradient toward winning → collapse. Classic "opponent too hard
  to bootstrap against."
- **Track curriculum (Sprint B).** Net-negative at every horizon. The staged
  vec-env rebuilds disrupt PPO at each stage boundary and/or the easy-track phase
  teaches degenerate early behavior. Crucially it ramps the *track* axis while the
  *opponent* stays strong — the wrong axis.
- **`randomize_seat` (Sprint A Idea 10).** The one real win: removes the seat-0
  overfit, +21 pts. Keep it.

---

## 4. Root cause — it's about learning signal, not opponents

Our reward is the terminal placement (rank). All the "opponent strength" tuning
was really moving one variable: **how much usable signal does the agent get per
episode?**

- Strong opponents → agent always finishes last → constant reward → no gradient.
- Weak opponents → agent sometimes wins → varying reward → it learns.
- Random opponents → even more wins, but low ceiling (you learn to beat noise).
- **Solo time-trial → maximal, dense, stationary signal** (every step of progress
  is feedback, independent of any opponent).

So the right move is not to keep dialing opponent strength — it is to **give the
agent a dense signal to bootstrap on**, then introduce opponents gradually.

Rejected alternative: **pure self-play from scratch** — already known to collapse
here (Sprint 5 Phase-2 self-play went worse-than-random; 6C/6D league/PFSP
couldn't stabilize it). Self-play needs a competent base first.

---

## 5. The validated recipe

Three stages, each **400k** steps, each **warm-starting the same network** from
the previous stage. `n_envs=8`, `n_steps=1024`, `batch_size=256`, `gamma=0.99`.

| Stage | Steps | Wall-clock | Setup | Result |
|---|---:|---:|---|---|
| 1. Solo pretrain | 400k | 3.3 min | `num_players=1`; dense progress reward (`shaping_weight=1.0`) + `+5.0` terminal finish bonus; speed driven by `gamma<1` | 66 steps/lap (vs 124 random) @ 98% finish |
| 2. Fine-tune vs weak | 400k | 3.9 min | 4p, weak pool (2×Heuristic + Random), `randomize_seat`, `shaping_weight=0.05` | **90.0% weak / 15.0% strong** |
| 3. Ramp to strong | 400k | 8.8 min | 4p, broadened strong pool (2×Strong + Heuristic), `randomize_seat`, `shaping_weight=0.05` | **94.4% weak / 85.6% strong** |
| **Total** | **1.2M** | **~16 min** | | **94.4% weak / 85.6% strong** |

Compare to from-scratch-vs-strong: **~4% / ~3%**. Night and day.

### Why the solo reward induces *speed*
Summed undiscounted progress over an episode is constant (= `track.length × laps`),
so progress alone does not reward speed. With `gamma<1`, the same progress (and
the finish bonus) is worth more the earlier it arrives → the policy minimizes
steps-to-finish. No negative rewards are used, so there is no "crash to end the
episode early" exploit. The finish bonus is paid only on `terminated` (laps
completed), not `truncated` (ran out of steps).

### Engine change required
`HeatEnv` previously hard-capped `num_players ∈ [2, 6]`. The engine supports solo
(`GameState.create(track, 1)` works), so the guard was relaxed to `[1, 6]` to
enable solo time-trial training. This is kept.

---

## 6. Caveats (read before trusting the numbers)

- **Point estimates, not the gate.** Prototype evals are point estimates over
  120–160 games (±~7% CI), **not** Sprint A's Wilson-LB held-out gate. The signal
  is far larger than noise, but the production version must fold the gate back in
  for honest checkpoint selection.
- **Untuned budget.** 400k/stage were round prototype numbers, not tuned. The
  solo stage probably needs less; the strong ramp probably wants more.
- **Wall-clock is not apples-to-apples.** The ~16-min prototype skipped the
  in-training gate entirely; the ~54-min baselines spent ~20 min playing gate
  games (120 strong + 120 weak, serial, every 100k steps × 20). At the 4-player
  level raw throughput is similar (~700–760 steps/s); solo is ~2–3× faster per
  step because there is only one car to simulate. Re-adding the gate will raise
  wall-clock (cadence is tunable).
- **No slipstream/tactics yet.** Solo can't teach drafting/blocking; those are a
  thin tactical layer the opponent stages (and, later, self-play from a strong
  base) would add.

---

## 7. Seed for the next sprint

Productionize the recipe:

1. **Solo pretrain phase** in `train_self_play` (or a preset): `num_players=1`,
   dense progress + finish-bonus reward, `gamma<1`. Add the solo terminal reward
   to `spaces.step_reward` (currently `_placement_reward` returns 0 for n≤1) so it
   is a first-class reward mode, not a wrapper.
2. **Opponent curriculum** weak→strong. **Reuse Sprint B's staged vec-env-rebuild
   machinery, but ramp the OPPONENT pool, not track difficulty.** This is the
   axis that matters. Demote the track curriculum to optional/off.
3. **Keep `randomize_seat`** throughout (the proven Sprint-A win).
4. **Keep Sprint A's Wilson-LB held-out gate** for checkpoint selection across all
   stages; gate vs the opponent of the current stage.
5. **Tune the per-stage budget** (less solo, more strong ramp) and the gate
   cadence vs wall-clock.
6. **Optional stretch:** self-play / league **from the strong-fine-tuned base**
   (the standard recipe: proxy/dense pretrain → harder opponents → league) to add
   slipstream/tactics — now that there is a competent base for it.

### Reusable experiment harnesses (`experiments/`)
- `proto_solo.py` — solo time-trial training + steps-to-finish + transfer eval.
- `proto_finetune.py` — warm-start fine-tune vs weak then ramp to strong.
- `diag_ablate.py` — toggle one feature at a time on generated tracks, eval vs weak.
- `diag_canlearn.py` — learnability control on a fixed track (is the core path healthy?).
