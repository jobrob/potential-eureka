# Sprint C3 — Findings: the closed self-play loop at small scale + the rung-4 verdict

> **Type:** design-gate outcome (mirrors C2-findings.md).
> **Deliverables exercised:** `experiments/mint_warm_prior.py` (the trained-WARM
> gen-0 prior C2 said C3 must start from), `experiments/az_loop.py` (the iterated
> generate→train→gate loop with the Wilson-LB/bootstrap-CI best-checkpoint
> promotion guard), `experiments/eval_az.py` (the rung-4 CI gate, in-search +
> net-only), `tests/test_az_loop.py` (15 tests: the promotion guard, the recency
> window, the seed-band disjointness, the end-to-end smoke).
> **Reproduce (the small-real run that produced the headline numbers):**
> ```
> # 0. mint the trained-WARM gen-0 prior (BC actor + value critic, grafted)
> PYTHONPATH=src python experiments/mint_warm_prior.py \
>     --train-tracks 80 --val-tracks 20 --rollouts 2 \
>     --bc-epochs 40 --value-epochs 40 --top-k 6 --device cuda \
>     --sims 16 --verify-games 16 --out checkpoints/c3_warm_gen0.zip
> # 1. run the closed loop from the warm prior
> PYTHONPATH=src python experiments/az_loop.py \
>     --warm checkpoints/c3_warm_gen0.zip --generations 4 \
>     --tracks 24 --val-tracks 8 --sims 16 --buffer-window 3 \
>     --epochs 30 --gate-games 12 --stop-patience 2 --device cuda \
>     --out checkpoints/c3_best.zip
> # 2. the rung-4 CI gate on the shipped net (in-search AND net-only)
> PYTHONPATH=src python experiments/eval_az.py \
>     --model checkpoints/c3_best.zip --games 20 --sims 16
> ```
> Numbers below are from one full small-real run on this box (Windows 11, RTX 4080,
> torch cu126; the two trainers ran on **CUDA**, the bottleneck is generation/gate
> clone cost exactly as C0/C2 found). Scale: the `large` net + the full multi-hour
> GPU campaign is OUT of scope (deferred, the S4 posture), as the C3 spec directs.

---

## TL;DR — rung-4 NOT MET; the loop plateaued and STOPPED HONESTLY (a process success, not a fake pass)

C2's #1 recommendation was acted on: **the loop started from a trained-warm v3
prior, not a cold random one.** The warm prior is real — its actor BC-clones the
search teacher (val acc 0.74, CARDS-acc 0.47) and its value head is calibrated to
~4.5-round MAE (vs C2's cold ~14), and it **beats a random net in-search**
(worst-case L1 spins 38 vs 173). The closed loop then ran, generation by
generation, with the bootstrap-CI best-checkpoint guard.

**No trained generation cleared the guard.** Both generations gated *no better*
than the warm prior (CI-overlapping or worse), so neither was promoted; after the
2-generation stop-patience the loop stopped and shipped the warm prior verbatim
(best-checkpoint preservation). On the held-out rung-4 gate the shipped net **does
not beat `LookaheadAgent`** — it trails it by a wide, CI-separated margin on both
axes. This is reported as an **honest plateau**, exactly the outcome README §5 /
the C3 spec call a legitimate stop. The failure mode the sprint exists to avoid —
a fake pass (USA-only, finish-rate-only, point-estimate) — did **not** occur: the
guard refused to promote a non-improving net, and the gate is the behavioral
held-out CI eval, not a training loss.

The *why* is the C2-flagged one, now confirmed under a warm prior: at the C0 visit
budget (16 sims) the self-play **visit targets are near-collapsed** (pi-entropy
~0.04 — barely above argmax/BC), so the policy target the loop trains toward
carries no improvement over the prior. This is precisely the **Gumbel-AlphaZero
seam-pull trigger** the C0/C2 findings reserved.

---

## Step 0 — the trained-warm gen-0 prior (C2's #1 ask, delivered)

The loop cannot bootstrap from nothing (C2 proved single-shot-from-cold futile).
`mint_warm_prior.py` produces a warm gen-0 net by reusing the existing precursor
pipelines and **combining** them — exploiting `share_features_extractor=False`
(the model.py default), under which the actor and critic own byte-disjoint module
sets:

1. **warm actor** — `gen_demos --players 1` (solo search-teacher demos, 100_000
   band) + `train_bc` → a checkpoint whose ACTOR is BC-warm;
2. **warm critic** — `gen_value_data` (solo MC value data, same bands) +
   `train_value` → a checkpoint whose CRITIC is value-warm;
3. **graft** — copy the critic-prefixed state-dict entries
   (`vf_features_extractor.` / `mlp_extractor.value_net.` / `value_net.`, the
   `train_value._critic_parameters` set) from the value net into the BC net, and
   re-save as one MaskablePPO checkpoint. The disjoint graft leaves the BC actor
   exactly intact and replaces only the init critic.

| precursor | metric | value |
|---|---|---|
| BC actor (`train_bc`, 40 ep, early-stopped ep 26) | val acc / CARDS-acc | **0.738 / 0.468** |
| value critic (`train_value`, 40 ep, early-stopped ep 14) | val MSE / **MAE (rounds)** | 53.8 / **~4.5** |
| warm gen-0 (combined) | §3.4 contract tripwire | **PASS** |
| warm gen-0 vs random (in-search, 16 held-out tracks) | worst-case L1 spins | **38 vs 173 → PASS** |

The warm prior carries genuine driving skill (it beats random) and a value head an
order of magnitude better than C2's cold ~14-round MAE. It is a legitimate loop
start — the precondition C2 said C3 needed. Seed bands: BC/value on 100_000 /
500_000, **disjoint** from the C3 self-play band (600_000+, per-generation slices)
and the held-out eval band (900_000) — asserted at startup.

---

## Step 1 — the closed loop (small-real config, the mechanism end-to-end)

Config: 4 generations max, 24 train / 8 val self-play tracks per generation, sims
16, recency window 3, 30 train epochs/gen, 12 gate games (900_000+ band),
stop-patience 2. Total wall time **~5 min** on the RTX 4080.

| gen | self-play targets (pi-entropy) | train val acc / CARDS-acc / V-MAE | **in-search** gate (worst L1 spins / rounds) | **net-only** gate | promoted? |
|---|---|---|---|---|---|
| warm gen-0 (incumbent) | — | — | **38.0** / 158.0 | 191.0 / 95.9 | (baseline) |
| 1 | 2883 (**0.045**) | 0.735 / 0.514 / **4.78** | 193.0 / 147.6 | 19.2 / 139.2 | **NO** (CI-overlap, not separated) |
| 2 | 2611 (**0.038**) | 0.653 / 0.398 / **6.47** | 180.0 / 178.2 | 197.0 / 148.2 | **NO** (worse) |

After gen 2 the loop hit 2 consecutive non-improving generations and **stopped**,
shipping the warm gen-0 prior (`best_is_warm = true`). The promotion guard worked
exactly as intended: a generation whose in-search gate was not **CI-separated**
better than the incumbent was never promoted (the 8C collapse guard). Note the
in-search / net-only **divergence** is visible per the design — gen 1's net-only
spins (19.2) look far better than its in-search (193), but neither is CI-separated
below the incumbent, and the divergence is itself a tell (the value head still
mis-orders far-from-finish leaves, so deeper search amplifies a bad leaf).

**Why no improvement signal:** the self-play **pi-entropy is ~0.04** at both
generations — the visit distribution is near-one-hot (effectively BC, with BC's
distribution gap). At 16 sims the search barely explores past the prior, so the
policy target it produces is no better than the prior it learned from. The MC value
targets are also still spin-spiral-dominated (12–18 of 36 races dropped to
MAX_ROUNDS per generation), so the value head's MAE *regresses* gen-1→gen-2
(4.78→6.47). One warm-prior loop at the C0 budget does not bootstrap a
better-than-prior fixed point — the C2-anticipated branch, now observed under a
warm start.

---

## Step 2 — rung-4 verdict (held-out solo, 20 tracks, 900_000+ band, sims 16)

| agent | finish% | rounds (pt [90% CI]) | worst-case L1 spins/pass (pt [90% CI]) | dist/heat | cooldowns/game | ms/move |
|---|---|---|---|---|---|---|
| Heuristic (rung 0) | 100% | 26.x | ~0.41 | 14.91 | 9.40 | — |
| **Lookahead (rung 1)** | 100% | **19.7 [18.4, 21.0]** | **0.50 [0.25, 0.50]** | 15.32 | 11.05 | 4.63 |
| AZ-MCTS (in-search) | 100% | 132.4 [106.7, 159.4] | 38.0 [26.4, 38.0] | 19.95 | 17.70 | 9.23 |
| AZ-net (net-only) | 100% | 88.6 [61.9, 116.9] | 191.0 [5.3, 191.0] | 27.12 | 14.45 | — |

**Rung-4 criteria (BEAT LookaheadAgent), CI-separated:**

| criterion | AZ-MCTS | Lookahead | result |
|---|---|---|---|
| worst-case L1 spins/pass (lower=better, CI-separated) | 38.0 | 0.50 | **FAIL** |
| rounds-to-finish (lower=better, CI-separated) | 132.4 | 19.7 | **FAIL** |
| 100% finish | 100% | 100% | PASS |

**VERDICT: rung-4 NOT MET.** The shipped net, in-search and net-only, trails
`LookaheadAgent` by a wide, **CI-separated** margin on both load-bearing axes. The
gap to rung 1 is **~6.7× rounds** and **~76× worst-case L1 spins** (in-search).

> **Do NOT paper over with finish-rate.** All four agents finish 100% — and the AZ
> net "finishes" at 132 rounds while spinning 38× at a single L1 corner. Per the
> README this is exactly the number we must not lead with; the load-bearing CIs say
> the AZ net is far below rung 1. The AZ-net's high `dist/heat=27` with 0-spin
> stretches is the same degenerate never-spend-heat tell C2 flagged, not "better
> budgeting".

---

## Honest reading + recommendation for the next step

C3's **machinery is built, correct, and contract-checked** (15 tests in
`tests/test_az_loop.py`, plus the full suite green). The **process worked**: a
trained-warm prior was minted (C2's #1 ask), the loop ran unattended, the
bootstrap-CI best-checkpoint guard refused to promote a non-improving generation,
and the plateau was reported honestly against a CI-gated behavioral eval — no fake
pass. **What is absent is the improvement signal**, and C3 isolates *why*: at the
C0 16-sim budget the self-play visit targets collapse (pi-entropy ~0.04), so the
loop trains toward a target no better than its prior, even from a warm start.

This is a **legitimate honest stop** (README §5: a measured, reported non-beat is a
success of the sprint's process). The recommendation, in priority order:

1. **Pull the Gumbel-AlphaZero seam forward (the `RootActionSelector` seam).** The
   pi-entropy ~0.04 at 16 sims is the exact trigger the C0/C2 findings reserved:
   Gumbel-AlphaZero's low-visit-count improvement guarantee + completed-Q policy
   target would make 16 sims produce a *non-collapsed* visit target before paying
   for more clones. This is the single highest-leverage next move and the seam was
   designed for precisely this case.
2. **Warm-start C's value head from an Option-A/B V (the `LeafEvaluator` seam).**
   The value head's MAE *regressed* across generations (4.78→6.47) under the
   spin-spiral-dominated MC targets; C's `z` is definitionally the A/B
   `−rounds_remaining`, so an A/B V is a drop-in leaf that removes the
   value-bottleneck variance the loop currently fights.
3. **Then the scale-up campaign** (`large` net, many generations, more sims/tracks)
   — but only *after* (1)/(2) demonstrate a *monotone* per-generation improvement
   at small scale. Scaling a loop that has no improvement signal at 16 sims would
   only buy a more expensive plateau.
4. **Fallback:** if a Gumbel + A/B-V-warm-start C3 still fails to beat rung 1, fall
   back to the A/B/E spine — the option remains on the table, exactly as the README
   gate-hard rule requires.

The takeaway for whoever picks this up: **C3's loop and guard are sound and the
warm-prior bootstrap is in place; the next sprint should pull the Gumbel seam (and
optionally the A/B-V leaf seam) BEFORE paying for a scale-up campaign, because the
binding constraint is visit-target collapse at low sims, not loop plumbing.**
