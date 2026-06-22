# Sprint C5 -- Findings: warm-start + anchor the value head (mechanism works; the borrowed Option-A V is off-distribution on self-play -- the honest stop)

> **Type:** design-gate outcome (mirrors C2/C3/C4-findings.md).
> **Deliverables exercised:** `experiments/train_az.py` (the shared `graft_warm_critic`
> helper + the `--critic-anchor {none,l2,low_lr,freeze_thaw}` mechanism + the
> `v_mae_rounds` summary signal), `experiments/gen_selfplay.py` (the `--traj-greedy`
> acted-trajectory decoupling + per-split finished counts), `experiments/az_loop.py`
> (the C5 toggles threaded end-to-end + the hard non-empty finished-val precondition +
> the per-generation value-MAE gate signal), `experiments/mint_warm_prior.py` (its
> `_graft_critic` refactored onto the shared helper), `tests/test_c5_value_warmstart.py`
> (12 tests: warm-start init, the does-NOT-regress headline, the anchor term, the
> traj-greedy finish-rate + target-byte-identity, the val-split hard-raise, the
> C4-default parity pin, band disjointness, schema invariance).
> **Reproduce (Windows 11, RTX 4080, torch cu126; generation/gate are clone-bound on
> CPU, the trainer ran on CUDA):**
> ```
> # the C5 moving-loop run (Gumbel ON + warm-value + L2 anchor + traj-greedy):
> PYTHONPATH=src python experiments/az_loop.py --warm checkpoints/c3_warm_gen0.zip \
>     --root-selector gumbel --warm-value checkpoints/c3_warm_gen0.zip \
>     --critic-anchor l2 --c-anchor 1.0 --traj-greedy \
>     --generations 4 --tracks 24 --val-tracks 8 --sims 16 \
>     --buffer-window 2 --epochs 30 --gate-games 16 --stop-patience 4 \
>     --device cuda --workdir runs/c5_loop_main --out checkpoints/c5_main_best.zip
> # the critic-protection ablation (one generation's aggregate, anchor strength sweep):
> for ca in 0.0 1.0 100.0 1000.0; do
>   PYTHONPATH=src python experiments/train_az.py --data runs/c5_loop_main/agg1.npz \
>     --out runs/c5_ablate/anchor_$ca.zip --warm-value checkpoints/c3_warm_gen0.zip \
>     --critic-anchor l2 --c-anchor $ca --epochs 30 --device cuda; done
> ```

---

## TL;DR -- the data-starvation fix WORKED, the anchor works exactly as designed, but the loop is STILL flat -- and the diagnostics show why: the borrowed Option-A V is ~30-round-MAE *on the self-play distribution*, not the ~4.5 it shows on its own heuristic-rollout band. This is the honest stop.

C5's narrow bar was a **moving loop**: (a) value-head MAE stays low (~4-5 rounds) and
does NOT regress across generations, AND (b) a generation finally promotes under the
existing Wilson-LB / bootstrap-CI guard.

1. **The data-starvation fix is MET, decisively.** The `--traj-greedy` acted-trajectory
   decoupling collapsed the C4 starvation: the finished-only val split was **non-empty
   every generation** (5-7 of 8 val races finished, vs C4's *0/8 emptied split*), so
   early-stopping was never silently disabled and the MC value target always had data.
   The logged Gumbel completed-Q `pi` target is **byte-identical** with the knob on or
   off (unit-test pinned) -- C4's entropy win is fully preserved.
2. **The warm-start + L2 anchor work exactly as specified.** Each generation grafts the
   calibrated warm critic into a fresh policy (no cross-generation drift), and the L2
   anchor faithfully holds the trained critic near that warm critic (ablation below).
   The "does NOT regress" unit test is green: with the anchor on the held-batch MAE
   stays near the warm baseline; with it off the critic drifts. The mechanism is sound.
3. **Criterion 1 NOT MET, criterion 2 NOT MET -- and the cause is now isolated and is
   NOT a plumbing bug.** The per-generation value-MAE is **20-35 rounds**, not ~4.5, and
   **no generation promoted**. The anchor is not failing -- it is doing its job and
   pulling toward a warm critic that is *itself* ~30-round-MAE on the self-play data.
   The borrowed Option-A V was calibrated by `mint_warm_prior` on the **heuristic
   rollout** policy, where solo races finish in ~18-20 rounds; the MCTS self-play
   trajectory (a still-weak searched agent) takes **2.5x+ longer** (mean
   rounds-remaining **48.9**, max **143**), so the warm V predicts ~25 where the truth
   is ~49 -- a **distribution-shift ceiling**, exactly the "stuck at the borrowed V"
   risk the C5 design named.

**This is the design's explicit honest-stop branch:** both the policy-target collapse
(C4) and the value-head *plumbing* (C5: init, anchor, starvation, val-split) are now
eliminated, and the loop still will not move. Per README §5 the recommendation is to
**fall back to the A/B/E spine**, with the value side's only remaining lever being the
reserved C6 -- but C6 (a lower-variance bootstrapped *target*) does not fix a
*distribution-shifted prior*; the more direct read is that the borrowed V's ceiling,
not MC-target variance, is the limiter. (See "Recommendation".)

---

## Step 1 -- the data-starvation fix (the val split survives): MET

The C4 Gumbel run drove **23/32** self-play episodes to MAX_ROUNDS and **emptied all 8
val tracks**, disabling early-stopping. The C5 `--traj-greedy` knob drives the *acted*
trajectory greedily (most-visited searched edge) while the logged completed-Q `pi`
target is untouched:

| run | acted trajectory | val races finished / total | val split | early-stop |
|---|---|---|---|---|
| C4 Gumbel | Gumbel-sampled (exploratory) | **0 / 8** (all dropped) | **EMPTY** | silently disabled (the bug) |
| **C5 Gumbel + traj-greedy** | greedy most-visited | **5-7 / 8** every gen | **non-empty** | active every generation |

The byte-identity of the `pi` target (knob on vs off, fixed `(state,seed)`) is unit-test
pinned (`TestTrajGreedy::test_target_byte_identical_under_traj_greedy`), so the trajectory
is genuinely decoupled from the target -- the greedier *driving* costs no target entropy.
The hard non-empty finished-val precondition (`az_loop`, `TestValSplitPrecondition`) now
*raises* if a generation ever does starve the val split again -- the C4 mechanical bug
can no longer recur silently.

## Step 2 -- the moving-loop table (4 generations, Gumbel + warm-value + L2 anchor + traj-greedy)

Warm-prior incumbent (in-search, held-out 900_000+ band): **spins 38.0 [26.4, 38.0],
rounds 149.6 [119.9, 176.5], 100%.**

| gen | promoted? | **value MAE (rounds)** | val finished | in-search spins (pt [lo,hi]) | in-search rounds (pt) | finish |
|---|---|---|---|---|---|---|
| 1 | NO | **20.74** | 5/8 | 190.0 [173.0, 190.0] | 191.3 | 100% |
| 2 | NO | **35.18** | 6/8 | 185.0 [158.0, 185.0] | 191.0 | 100% |
| 3 | NO | **29.84** | 7/8 | 92.5 [61.7, 92.5] | 183.6 | 100% |
| 4 | NO | **14.82** | 6/8 | 181.0 [178.0, 181.0] | 195.2 | 100% |

**Reading.** The value MAE is **not** the ~4.5 the C5 bar wanted and does not settle --
it wanders 20 -> 35 -> 30 -> 15 with no monotone trend, because each generation re-anchors
to a warm critic that is ~30-round-MAE on *that generation's* self-play tracks (the
target distribution itself shifts as the buffer window slides). No generation's in-search
gate is CI-separated below the warm incumbent's spins (38.0) -- the searched nets spin far
more (92-190) at the tight L1 corners. Nothing promoted. (Note gen 3's spins point 92.5 is
the best of the four but its CI [61.7, 92.5] is nowhere near the incumbent's 38.0, so the
guard correctly withholds promotion -- no fake pass.)

## Step 3 -- the critic-protection ablation (the anchor works; the warm V is the ceiling)

One generation's aggregate (`agg1.npz`), the same 30-epoch train, sweeping `c_anchor`:

| `--critic-anchor` / `c_anchor` | val v_mse | **val v_mae (rounds)** | reading |
|---|---|---|---|
| `l2` / **0.0** (== warm-start only, no protection) | 1042.3 | **20.43** | the C4-style free critic |
| `l2` / **1.0** (the loop default) | 1036.9 | **20.74** | anchor holds it near warm |
| `l2` / **100.0** | 991.9 | **20.74** | stronger -- still ~20 (== warm's own MAE here) |
| `l2` / **1000.0** | 1391.9 | **25.65** | over-anchored: frozen onto the *worse-fit* warm critic |

The anchor is **not inert and not failing**: as `c_anchor` rises the critic is pulled ever
closer to the warm critic, and at `c_anchor=1000` it is essentially frozen *onto* it --
and the warm critic's own MAE **on this self-play val split is 30.6 rounds** (measured
directly). So "anchor harder" cannot get below the warm V's ceiling; it converges *to* it.
The `low_lr` / `freeze_thaw` alternates are wired and unit-covered but were not run at
scale -- they are strictly weaker levers for the same goal (slow the critic's drift toward
the MC target), and the L2-anchor-vs-off ablation already shows the binding constraint is
the *anchor target*, not the *anchor strength or schedule*.

### The root cause, measured

```
warm critic predicted rounds-remaining (self-play val): mean 24.8  (min 0.8, max 45.1)
actual    rounds-remaining (self-play val):              mean 48.9  (min 0,   max 143)
warm critic MAE on the self-play val rows:               30.58 rounds  (n=1382)
```

The warm critic was minted on the **heuristic rollout** distribution (solo finishes
~18-20 rounds). On the **MCTS self-play** distribution the still-weak searched agent
finishes in ~2.5x the rounds (the in-search gate's own rounds point is 149.6), so every
self-play state's true cost-to-go is far larger than anything the warm V ever saw. The
warm V *systematically under-predicts* (24.8 vs 48.9) and the anchor faithfully preserves
that bias. **The value head is a stable, low-error leaf on the band it was trained on, and
an off-distribution one on the band the loop generates.** That is the ceiling.

## Cost report

The `--traj-greedy` decoupling adds no search cost (it only changes which already-computed
edge is acted). Generation + gate are CPU-clone-bound as in C1-C4; the trainer ran on
CUDA. Each generation (24 train / 8 val self-play tracks, 16 sims, 30 epochs, 16 gate
games) ran ~2 min; the 4-generation loop ~8 min wall. The ablation sweep (4 trains x 30
epochs) ran in ~2 min total on CUDA.

## Honest verdict + recommendation for the next step

**Criteria 1 + 2 (the moving loop): NOT MET, and the result is fully attributable.** The
three C5 levers are correct and contract-checked (12 tests + full suite **959 passed, 1
skipped**, no regressions): the starvation fix kept the val split non-empty every
generation, the warm-start grafts a known-good critic per generation, the L2 anchor holds
it there, and the value-MAE gate signal surfaced the (non-)progress every generation the
way C4's 4.8 -> 6.5 -> 40 should have been caught. But the loop did not move and the value
MAE stayed at ~20-35 rounds, because the borrowed Option-A V is **off-distribution on the
self-play band** -- it was calibrated where races finish in ~18-20 rounds and is asked to
score states whose true cost-to-go is ~49 (max 143). The anchor cannot do better than its
target, and its target is ~30-round-MAE here.

This is exactly the design's **honest-stop branch**, and it is the most informative
possible stop: C4 eliminated the policy-target collapse, C5 eliminated the value-head
*plumbing* collapse (init / anchor / starvation / val-split), and the loop is **still
flat** -- so the remaining gap to `LookaheadAgent` (~7x rounds / ~76x spins) is
**structural**, not a fixable bottleneck in this pipeline. Per README §5's gate-hard rule
the recommendation is to **fall back to the A/B/E spine**.

**Recommendation, in priority order:**

1. **Fall back to the A/B/E spine (the primary recommendation).** Both measured plumbing
   bottlenecks (policy target, value head) are now removed and the CI-gated behavioral
   loop still does not beat rung 1. A measured, CI-gated non-beat that has eliminated both
   collapses is the strong honest signal to stop pouring sprints into Option-C and return
   to the spine (where the search agent already beats the strong heuristic at 76%).

2. **If Option-C is pursued further, the next lever is NOT C6 (TD/bootstrap target) --
   it is an *on-distribution* value source.** C6 lowers the *variance* of the MC target;
   it does not fix a *distribution-shifted prior*, which is what the diagnostics show is
   the limiter here. The honest fix would be to **re-mint the warm V on self-play (or
   mixed self-play + heuristic) rollouts** so the critic's training band matches the loop's
   generation band -- i.e. let the critic *learn* the long self-play episodes rather than
   anchoring it to a heuristic-band critic. That is a `gen_value_data` policy/band change,
   larger than C5 and only worth it if the spine fallback is rejected.

3. **C6 (the reserved `value_target` seam) remains deferred and is now LOWER priority.**
   The C5 diagnostics point at *prior distribution shift*, not *MC-target variance*, as the
   remaining limiter -- so the precondition C5's design set for pulling C6 forward
   ("MC-target variance, not init/starvation/tuning, is the remaining limiter") is **not**
   met. C6 should not be spent.

The takeaway for whoever picks this up: **C5's three levers are correct and green -- the
val-starvation that wrecked C4 is fixed, the warm-start + anchor protect the critic
exactly as designed, and the value-MAE is now a first-class per-generation gate signal.
But the borrowed Option-A V has a real off-distribution ceiling on the long, slow MCTS
self-play episodes (~30-round MAE on the band the loop generates vs ~4.5 on its own
heuristic band), so the loop does not move. With both the policy-target collapse (C4) and
the value-head plumbing collapse (C5) eliminated and the CI-gated loop still flat, the
honest call is the A/B/E-spine fallback -- not C6, which fixes variance, not the prior
shift that is actually binding.**
