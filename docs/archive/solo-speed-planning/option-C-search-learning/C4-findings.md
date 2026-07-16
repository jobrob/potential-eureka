# Sprint C4 — Findings: the Gumbel-AlphaZero `RootActionSelector` seam (mechanism fixed, value head is now the binding constraint)

> **Type:** design-gate outcome (mirrors C2-findings.md / C3-findings.md).
> **Deliverables exercised:** `src/heat/agents/mcts_agent.py` (the Gumbel root
> selector behind `MCTSConfig.root_selector`, PUCT default byte-identical),
> `experiments/gen_selfplay.py` (the completed-Q policy target branch),
> `experiments/az_loop.py` (`--root-selector` threaded), `tests/test_c4_gumbel.py`
> (16 tests: non-collapse, low-visit improvement guarantee, PUCT parity,
> determinism/seed-purity, Sequential-Halving budget accounting, schema invariance).
> **Reproduce (this box: Windows 11, RTX 4080, torch cu126; generation/gate are
> clone-bound on CPU, the trainer ran on CUDA):**
> ```
> # 0. the entropy comparison (the headline mechanism number) -- same prior/config,
> #    only --root-selector differs:
> PYTHONPATH=src python experiments/gen_selfplay.py --model checkpoints/c3_warm_gen0.zip \
>     --tracks 12 --val-tracks 4 --sims 16 --root-selector puct   --out runs/c4/puct_targets.npz
> PYTHONPATH=src python experiments/gen_selfplay.py --model checkpoints/c3_warm_gen0.zip \
>     --tracks 12 --val-tracks 4 --sims 16 --root-selector gumbel --out runs/c4/gumbel_targets.npz
> # 1. the one-cycle CI gate, Gumbel ON off the C3 warm prior (the apples-to-apples test):
> PYTHONPATH=src python experiments/az_loop.py --warm checkpoints/c3_warm_gen0.zip \
>     --root-selector gumbel --generations 1 --tracks 24 --val-tracks 8 --sims 16 \
>     --buffer-window 1 --epochs 30 --gate-games 16 --stop-patience 1 --device cuda \
>     --workdir runs/c4_loop_gumbel --out checkpoints/c4_gumbel_best.zip
> # 1-control. the IDENTICAL cycle with PUCT (the "the PUCT loop could not improve" baseline):
> PYTHONPATH=src python experiments/az_loop.py --warm checkpoints/c3_warm_gen0.zip \
>     --root-selector puct --generations 1 --tracks 24 --val-tracks 8 --sims 16 \
>     --buffer-window 1 --epochs 30 --gate-games 16 --stop-patience 1 --device cuda \
>     --workdir runs/c4_loop_puct --out checkpoints/c4_puct_control.zip
> ```

---

## TL;DR — criterion 1 MET (the visit-target collapse is fixed); criterion 2 NOT MET at small scale, and the cause is now isolated to the VALUE head, not the policy target

Sprint C4's narrow, honest bar has two parts:

1. **Non-collapsed target at the C0 budget — MET, decisively.** At 16 sims off the
   C3 warm prior, the Gumbel **completed-Q** policy target's π-entropy is **0.461
   overall / 0.661 CARDS**, versus the PUCT **visit-count** target's **0.024 / 0.039**
   on the *same* prior and config — a **~19×** lift overall and **~17×** on the wide
   CARDS branch, well clear of the documented ~0.04 collapse floor. The mechanism
   the C0/C2/C3 findings reserved Gumbel for is removed: the policy target now
   carries a graded improvement signal at 16 sims, not a near-one-hot argmax/BC.
2. **A CI-separated one-cycle improvement the PUCT loop could not produce — NOT MET
   at this scale.** Off the same warm prior, one generate→train→gate cycle with
   Gumbel targets did **not** clear the promotion guard (the trained net gated
   *worse* than the warm prior in-search), so nothing was promoted. The PUCT control
   cycle (identical config) also did not improve — so C4 did **not** flip rung-3's
   verdict at small scale.

**But the two runs disagree on *why* in a way that is the actionable C4 result.**
The PUCT control trained a clean value head (val vMAE **5.5 rounds**, best-val
early-stop at epoch 26) and still could not improve — the C3 plateau, caused by the
collapsed policy target. The Gumbel run **fixed the policy target** but its
exploratory trajectory drove **23 of 32** self-play episodes into MAX_ROUNDS (vs
PUCT's 12), which **emptied the held-out val split**, disabled early-stopping, and
left the value head badly fit (final-epoch train vMSE **252** vs PUCT's **57**).
The Gumbel net's in-search gate is therefore dominated by a **broken value head**,
not a broken policy target. This is exactly the design's anticipated honest stop:
**the entropy lifts (criterion 1) but the behavioral improvement is absent
(criterion 2) ⇒ the binding constraint is now the value head** — the C3-finding-#2
`LeafEvaluator` / A/B-V warm-start lever (a candidate **C5**), reported plainly,
never papered over with a finish-rate / USA-only / point-estimate "win" (all four
agents finish 100%; the load-bearing CIs say neither trained net beats the prior).

---

## Step 1 — the entropy comparison (criterion 1): the mechanism fix, measured directly

Same warm prior (`checkpoints/c3_warm_gen0.zip`), same 12 train / 4 val tracks, same
16 sims; the **only** difference is `--root-selector`:

| target | overall π-entropy | CARDS π-entropy | verdict |
|---|---|---|---|
| PUCT visit-count (C2/C3) | **0.024** | **0.039** | collapsed (the documented ~0.04 floor — barely above argmax/BC) |
| **Gumbel completed-Q (C4)** | **0.461** | **0.661** | **non-collapsed**, graded — a real distribution |

The unit suite pins the same fact at the decision level without a trained net
(`tests/test_c4_gumbel.py::TestNonCollapse`): on matched GEAR and CARDS decisions the
Gumbel completed-Q entropy is materially above the visit-count entropy (GEAR ~0.22,
CARDS ~0.86 with the retuned σ; the PUCT side stays near the floor). σ is **not**
inert and **not** swamped: the improvement-guarantee test confirms the completed-Q
mass shifts toward the higher-Q action (σ(Q̂) bites), while the entropy stays far
from one-hot (σ no longer dominates the prior). The mechanism is fixed.

### σ constant retune (Sprint C4 decision #1, exercised honestly)

Decision #1 was: start from Danihelka et al. (2022) `c_visit=50, c_scale=1.0` and
**retune only if the non-collapse test fails**. It failed — at our LOW (16) sim
budget with `[0,1]` Q-normalization, `(c_visit + max_n)·c_scale ≈ 58` swamps the
log-prior and σ(Q̂) collapses the completed-Q target to one-hot at narrow (GEAR)
decisions (measured GEAR π-entropy **0.002** — a swamped σ, the documented hazard).
The published constants assume Danihelka's hundreds-of-sims / `[-1,1]`-value regime;
at 16 sims the σ magnitude must shrink to keep the prior in play. Retuned to
**`c_visit=25, c_scale=0.25`**, which restores a non-collapsed, graded target (GEAR
~0.22, CARDS ~0.86) while σ still moves mass toward higher-Q actions. This is a
constant change behind the seam; the algorithm and the PUCT default are untouched.

---

## Step 2 — the one-cycle CI gate (criterion 2): Gumbel vs the PUCT control, off the same warm prior

Both: 1 generation, 24 train / 8 val self-play tracks, 16 sims, 30 epochs, 16 gate
games on the held-out 900_000+ solo band, CUDA trainer. The warm-prior incumbent
gates at **in-search spins 38.0 [26.4, 38.0], rounds 149.6 [119.9, 176.5], 100%**.

| run | self-play π-entropy | races dropped (MAX_ROUNDS) | train val acc / **val vMAE** | trained-net in-search gate (spins / rounds) | promoted? |
|---|---|---|---|---|---|
| **Gumbel (C4)** | **0.468** | **23 / 32** | empty val set → **no early-stop** (final-epoch train vMSE **252**) | 191.0 [188, 191] / 200.0 [200, 200] | **NO** (worse) |
| PUCT control (C3 repro) | 0.045 | 12 / 32 | 0.735 / **5.5 rounds** (best-val ep 26) | 193.0 [25.6, 191*] / 149.9 [118.7, 180.1] | NO (CI-overlap) |

\* the PUCT control's in-search spins CI is very wide (`[25.6, 191]`) — not
CI-separated below the incumbent, hence not promoted, exactly the C3 result.

**Reading.** The PUCT control reproduces C3 faithfully: a clean value head
(vMAE 5.5), a collapsed policy target (entropy 0.045), and no improvement — the
policy target is the bottleneck *for PUCT*. The Gumbel run removes that bottleneck
(entropy 0.468) but exposes the **next** one: its exploratory trajectory spins more
episodes to MAX_ROUNDS, which here **wiped the entire val split** (8 val tracks all
dropped), disabled best-val early-stopping, and left the value head 4–5× worse on
train vMSE. A value head that mis-orders far-from-finish leaves makes the *deeper*
in-search net worse (the C3-observed in-search/net-only divergence amplified), which
is exactly what the gate shows. **The improvement signal is absent because the value
target — not the policy target — is now the binding constraint.**

> **Do NOT paper over with finish-rate.** All agents finish 100%; the Gumbel net
> "finishes" at 200 rounds while spinning 191× at a single L1 corner. Per the README
> this is the number we must not lead with — the load-bearing CIs say neither trained
> net beats the warm prior, and the Gumbel net is dominated by its broken value head.

---

## Cost report (Gumbel changes the root allocation, not the budget)

The Sequential-Halving root allocation runs **exactly** `n_simulations` interior
descents per move (the budget-accounting unit test pins `root.n == n_simulations`
for sims ∈ {1, 4, 8, 16, 32}; `gumbel_m=1` degenerates to the prior-greedy single
arm, the C1 `n_simulations=1` analogue). Measured cost is essentially unchanged
from PUCT: **~9.8 ms/move, ~16 clones/move** (Gumbel) vs **~9.6 ms/move, ~15
clones/move** (PUCT) on the entropy-comparison run — Gumbel reallocates the same
budget across the sampled top-`m`, it does not spend more.

---

## Honest verdict + recommendation for the next step

**Criterion 1 (the mechanism rung): MET.** The visit-count policy-target collapse
that C3 isolated as rung-3's failure cause is fixed — the Gumbel completed-Q target
is non-collapsed (π-entropy ~0.46 vs the ~0.04 PUCT floor) at the C0 16-sim budget,
with no extra clone cost, and the PUCT path is byte-for-byte unchanged (the parity
regression test passes; the agent still acts and is evaluated with PUCT — scope #3).

**Criterion 2 (the CI-separated one-cycle improvement): NOT MET at small scale,**
and that is a **reportable** result per the C4 design, not a fake or a silent fail.
The trained net did not beat the warm prior in-search; the cause is now isolated by
the PUCT control to the **value head**, not the policy target. C4 deliberately
changed only the policy target so this attribution is clean: with a clean value head
(PUCT control) the policy target is the bottleneck; with the policy target fixed
(Gumbel) the value head is. **This is precisely the design's "entropy lifts but no
behavioral improvement ⇒ the value head is the binding constraint" branch**, which
routes to the deferred A/B-V `LeafEvaluator` lever.

**Recommendation, in priority order:**

1. **Pull the `LeafEvaluator` / A/B-V value-head warm-start (a C5).** This is now
   the binding constraint, with two independent tells from this sprint: (a) the
   PUCT control trains a *clean* value head and still cannot improve from the
   collapsed policy target — fixed by C4; (b) the Gumbel run fixes the policy target
   and is then dominated by a *broken* value head. C's `z` is definitionally the
   A/B `−rounds_remaining`, so an A/B V is a drop-in leaf that removes the
   spin-spiral-dominated MC-target variance the value head currently fights. Gumbel
   (policy) + A/B-V (value) are composable and were deliberately kept apart so this
   attribution would be clean — it now is.
2. **Before any scale-up, fix the val-set starvation.** The Gumbel trajectory's
   higher MAX_ROUNDS drop rate (23/32) emptied the val split and disabled
   early-stopping — a confound that on its own can sink a generation. A larger
   self-play track count (so val survives the drops) and/or a less-exploratory
   acting temperature for the *trajectory* (the target is already non-collapsed; the
   trajectory need not spin as much) should precede the scale-up. This is a tuning
   fix, not a mechanism change.
3. **Then the scale-up campaign**, but only after (1)/(2) demonstrate a *monotone*
   per-generation improvement at small scale — C4 is the **enabler** (the policy
   target no longer collapses), not the campaign. Scaling now would buy a
   value-head-limited plateau, the C3-finding-#3 hazard.
4. **Fallback unchanged:** if Gumbel + A/B-V still fails to beat rung 1, fall back to
   the A/B/E spine (README §5 gate-hard rule). The option stays on the table.

The takeaway for whoever picks this up: **C4's seam swap is correct, contract-checked
(16 tests + full suite green), and the policy-target collapse is genuinely fixed —
the headline π-entropy went from ~0.04 to ~0.46 at 16 sims. The one-cycle behavioral
gate did not improve, and the apples-to-apples PUCT control isolates the reason to
the VALUE head (C3 finding #2), not the policy target. The next sprint should pull
the `LeafEvaluator` / A/B-V warm-start (C5) and fix the val-set starvation BEFORE
paying for a scale-up campaign.**
