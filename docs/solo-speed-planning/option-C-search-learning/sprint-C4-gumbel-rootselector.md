# Sprint C4 — Pull the Gumbel-AlphaZero `RootActionSelector` seam forward (fix the visit-target collapse)

> The first *post-foundation* sprint, and the one C2 and C3 both reserved by name.
> C0–C3 are built, green, and contract-checked, but C3's closed loop **plateaued**:
> no trained generation cleared the promotion guard, so the warm prior shipped
> verbatim and the net does not beat `LookaheadAgent` (rung 4 NOT met). C3 isolated
> the cause — it is **not** loop plumbing — to a single mechanism: at the C0 16-sim
> budget the self-play **visit-count policy target collapses** (π-entropy ~0.04,
> effectively argmax/BC), so the loop trains toward a target no better than its
> prior. This sprint swaps the reserved **Gumbel-AlphaZero** behaviour into the
> `RootActionSelector` / `SearchPolicy` seam (README §3, §7): Gumbel top-`m`
> sampling + Sequential Halving for the acted action, and a **completed-Q policy
> target** in place of the visit-count target — both giving a *non-collapsed,
> provably-improving* target at small sim counts. It is a **seam swap at the root
> only**: interior PUCT, the chance-node DPW machinery, the frozen codec, the
> `MCTSAgent` acting contract, and leaf evaluation are untouched.

## Goal

Replace, **at the root only and behind a config toggle**, C1/C2's
Dirichlet-root-noise + temperature-sampling + visit-count target with the
Gumbel-AlphaZero `RootActionSelector` (README §3 deferred impl): root action
sampled by the Gumbel-top-`m` trick over `(logits + Gumbel)`, the sim budget
allocated across the `m` sampled actions by **Sequential Halving**, the acted
action chosen by `argmax(g + logits + σ(Q̂))`, and the policy target built as the
**completed-Q** distribution `π = softmax(logits + σ(completedQ))`. The narrow,
honest C4 success is the **mechanism fix**: at the C0 sim budget the Gumbel target
is measurably non-collapsed (π-entropy well above the ~0.04 floor) AND one
generate→train cycle shows a **CI-separated improvement over the prior** that the
C2/C3 PUCT loop could not produce — i.e. the cause of rung-3's failure is removed.
Beating `LookaheadAgent` at small scale (rung 4) is a *stretch*, not the bar.

### Why Gumbel is the right fix (the standard result, applied to our collapse)

Gumbel-AlphaZero (Danihelka et al., 2022) gives a **policy-improvement guarantee
even at very low simulation counts**: the acted action it selects has a Q at least
as high (in expectation) as the prior's, and the completed-Q target it constructs
is a *bounded improvement* over the prior policy regardless of how few simulations
were run — precisely because it never relies on visit counts as the policy signal.
That is the exact property our π-entropy ~0.04 collapse needs: at 16 sims the
visit distribution is near-one-hot (the search barely explores past the prior, so
`N(a)^{1/τ}/ΣN` ≈ argmax = BC, with BC's distribution gap, C3 finding). The
completed-Q target is **not** a count — it is the network logits *completed* with
the searched Q on visited actions and the value-estimate on unvisited ones, then
softmaxed — so it carries a graded improvement signal at any sim budget. This is
the lever to "make 16 sims go further before paying for more clones" the C0/C2/C3
findings reserved.

## Scope

**In.**

- **The Gumbel root selector, as an alternate `RootActionSelector`/`SearchPolicy`
  behind the C1 seam.** A new `MCTSConfig` field `root_selector: str` with values
  `"puct"` (default — the byte-for-byte-unchanged C1 path used for the parity eval)
  and `"gumbel"`. When `"gumbel"`, the **root** is governed by:
  - **Sample top-`m` actions without replacement** via the Gumbel-top-`k` trick:
    draw `g_a ~ Gumbel(0)` per kept candidate action (from the agent's deterministic
    per-turn RNG `_search_rng`, never global RNG — the C1 determinism contract), and
    take the `m` actions with the largest `(g_a + logits_a)`, where `logits_a` is the
    net's **log**-prior over the kept candidate set (the same masked, dedup-by-speed
    kept set `_edges_for` already builds — read the prior, log it, do **not**
    re-prune). `m = config.gumbel_m` (C0/C4-tuned; a small power of two, default 8,
    capped at the kept-candidate count).
  - **Sequential Halving** over the sampled `m` actions: split the
    `n_simulations` budget into `⌈log2(m)⌉` phases; in each phase run an equal share
    of sims through the *currently-surviving* root actions (each sim is a normal C1
    interior descent + chance-DPW + leaf eval **unchanged** — only the *root edge it
    is forced through* is dictated by Sequential Halving, not by PUCT), then keep the
    top-half by `g_a + logits_a + σ(Q̂_a)` and recurse until one survives.
  - **Acted action** = `argmax_a (g_a + logits_a + σ(Q̂_a))` over the survivors,
    where `Q̂_a` is the root edge's normalized mean value (the existing
    `_normalize_q(edge.q())`) and `σ` is the standard Gumbel monotonic transform
    `σ(q) = (c_visit + max_b N_b) · c_scale · q` (Danihelka's `σ`; `c_visit`,
    `c_scale` are C0/C4 constants). This replaces `_best_edge(at_root=True)`'s
    most-visited / temperature-sampling rule **at the root only**.
- **The completed-Q policy target in `gen_selfplay`.** `_search_visit_distribution`
  gains a branch keyed on `agent.config.root_selector`:
  - `"puct"` → the existing visit-distribution target (unchanged).
  - `"gumbel"` → the **completed-Q** target
    `π(a) = softmax_a( logits_a + σ(completedQ_a) )` over the kept candidate set,
    written into the same full-`ACTION_DIM` `pi` vector (zeros elsewhere), where
    `completedQ_a = Q̂_a` for a root action that was searched (`edge.n > 0`) and
    `completedQ_a = v̂` (the root's own value estimate `root.value`, normalized) for
    a sampled-but-unvisited / un-sampled action — the "completion" step. The acted
    action returned alongside is the Gumbel-selected one (above). The **output
    schema is identical** — `(obs, π, mask, z)` with `π` a probability vector whose
    support ⊆ mask and sums to 1 — so `train_az.py`, `_SelfPlayBuffer`, the `.npz`
    contract, and `az_loop`'s aggregation are **unchanged**. The same off-table
    drop-and-count, the same `pi` support-⊆-mask + sums-to-1 contract guards, and
    the same degenerate fallback all apply verbatim.
- **The `z` target is unchanged.** The MC return with the **pre-spin floor**
  (`_floored_rounds_remaining` / `_spin_round`, C2's single most important
  correctness item) is **kept exactly** — C4 changes only the *policy* target, not
  the value target. (The value-bottleneck lever is C3 finding #2, out of scope here
  — see "Relationship to the A/B-V warm-start" below.)
- **The toggle plumbed end-to-end, defaulting OFF.** `generate_dataset` builds its
  `MCTSConfig` from a new `--root-selector` arg (default `"puct"`), and
  `az_loop._gen_selfplay` threads it through its `argparse.Namespace`. With the
  default everything is the C2/C3 path byte-for-byte; `--root-selector gumbel` turns
  C4 on. No change to `eval_mcts.py` / `eval_az.py` / `value_iterate.py`: the eval
  reads a trained checkpoint and runs the **interior** search, which is unchanged;
  the promotion guard is unchanged.

**Out.**

- **The scale-up campaign.** C4 is the *enabler* — its green light is the
  precondition for the deferred `large`-net multi-hour campaign (README §7), NOT the
  campaign itself. Scaling a loop whose policy target collapses only buys a more
  expensive plateau (C3 finding #3); C4 must first show a non-collapsed,
  CI-separated improvement at small scale.
- **The A/B-value-head warm-start (`LeafEvaluator` seam, C3 finding #2).** Adjacent
  and *separable* — see the dedicated note below. C4 stays focused on the
  policy-target collapse, not the value bottleneck. It may become C5 or an optional
  knob; it is deliberately not bundled, so C4's improvement signal is attributable to
  Gumbel alone.
- **Interior-node Gumbel / full Sequential Halving in the whole tree.** Gumbel is a
  **root** selector; interior nodes keep log-PUCT (§4.1) untouched. (Interior Gumbel
  is a further refinement; not now.)
- **KataGo forced-playout + policy-target pruning** (a further `RootActionSelector`
  target-sharpening refinement, README §7) — note as a seam knob, not built.
- Opponents, MuZero, learned chance (the other deferred seams) — unchanged from C1.

## Deliverables

- **`src/heat/agents/mcts_agent.py`** — the Gumbel root selector behind the seam:
  - the new `MCTSConfig` fields `root_selector: str = "puct"`, `gumbel_m: int`,
    and the `σ` constants (`gumbel_c_visit`, `gumbel_c_scale`), all with C0/C4
    defaults and `__post_init__` validation (`root_selector in {"puct","gumbel"}`,
    `gumbel_m >= 1`);
  - a `_gumbel_root_plan` path (parallel to `_plan_turn`'s root loop) that does the
    Gumbel top-`m` sample + Sequential Halving allocation, reusing `_edges_for` for
    the kept set + prior, `_simulate`'s interior descent **unchanged** (only the
    forced root edge differs per phase), and `_normalize_q` for `Q̂`. The `"puct"`
    branch calls the existing `_plan_turn` / `_best_edge` path with **zero**
    behavioural change.
  - the acted-action rule `argmax(g + logits + σ(Q̂))` wired into `_extract_plan`
    behind the `root_selector` toggle (the C1 most-visited / temperature path
    preserved as the `"puct"` default).
- **`experiments/gen_selfplay.py`** — the completed-Q target branch in
  `_search_visit_distribution` (keyed on `agent.config.root_selector`), plus the
  `--root-selector` arg threaded into `generate_dataset`'s `MCTSConfig`. The visit
  target, the off-table drop-and-count, the `z` pre-spin floor, and the `(obs, π,
  mask, z)` schema are otherwise untouched.
- **`experiments/az_loop.py`** — `--root-selector` threaded through `_gen_selfplay`'s
  `Namespace` so a closed-loop run can request Gumbel; default `"puct"`.
- **The C4 experiment**: one generate→train cycle with `--root-selector gumbel` off
  the **C3 trained-warm prior** (`checkpoints/c3_warm_gen0.zip`), re-evaluated with
  `eval_mcts.py --compare-model` (the rung-3 head-to-head, unchanged) against the
  same prior generating PUCT targets — the apples-to-apples "did the target fix
  produce an improvement signal the PUCT loop could not" test.
- **`sprint-C4-gumbel-findings.md`** — the findings deliverable (mirrors
  C2-/C3-findings.md): the measured π-entropy at 16 sims (Gumbel vs the ~0.04 PUCT
  floor), the CI-gated one-cycle improvement verdict, the cost report (ms/move +
  clones/move — Gumbel changes the *root allocation*, so report whether it shifts the
  budget), and the honest go/no-go for the deferred scale-up.
- **Unit tests** in `tests/test_gumbel_root.py` (mirroring the C1/C2/C3 test
  discipline):
  - **non-collapse**: on a fixed forced state at `n_simulations=16`, the completed-Q
    target has **measurably higher entropy** than the visit-count target the same
    search produces (the exact collapse C3 observed, now lifted) — the headline
    correctness check;
  - **improvement-guarantee property on a toy case**: a tiny hand-built tree with
    known leaf values where Gumbel's acted action has Q ≥ the prior-argmax's Q at a
    very low sim count (the low-visit guarantee holds), and the completed-Q target's
    mass shifts toward the higher-Q action vs the prior;
  - **PUCT default unchanged**: with `root_selector="puct"` (the default) the agent
    and `_search_visit_distribution` produce **byte-identical** output to the
    pre-C4 code on a fixed `(state, seed)` (a regression pin — C4 must not perturb
    the parity path);
  - **determinism / seed-purity preserved**: Gumbel sampling draws only from
    `_search_rng` (seeded off the turn signature), so the whole Gumbel search +
    target is a **pure function of `(state, seed)`** — same state + seed ⇒
    byte-stable target and acted action, no global-RNG leak (the C1 contract);
  - **schema invariance**: the completed-Q `π` is a probability vector, support ⊆
    mask, sums to 1, full `ACTION_DIM` width — so the `.npz` / `train_az` contract
    is unchanged (the C2 guards still pass);
  - **Sequential Halving budget accounting**: the total interior sims run equals
    `n_simulations` (no budget leak/overrun) and the survivor set halves per phase as
    specified;
  - **`z` untouched**: the pre-spin-floor value target is identical with Gumbel on
    (re-run the C2 forced-spin-track check unchanged).

## Success criteria (tie to the rung ladder — honestly)

The **primary, narrow C4 bar is the mechanism fix**, not rung 4:

1. **Non-collapsed target at the C0 budget.** With `--root-selector gumbel` at
   `n_simulations=16`, the self-play **policy-target π-entropy is well above the
   ~0.04 floor** C2/C3 measured for the visit-count target (report overall and
   CARDS-only entropy, the same two numbers `gen_selfplay` already prints). This is
   the direct evidence the collapse is fixed.
2. **A CI-separated one-cycle improvement the PUCT loop could not produce.** Off the
   **same C3 warm prior**, one generate→train cycle with Gumbel targets yields a net
   that, plugged back into the interior search, **beats the prior** on the rung-2
   metrics (worst-case L1 spins/pass AND rounds-to-finish), **CI-separated** (the
   bootstrap-CI / Wilson-LB gate, not a point estimate), on disjoint held-out seeds —
   where the C2/C3 PUCT cycle from the same prior did **not** (rung-3 NOT met / the
   loop plateaued). This is "fix rung-3's failure cause", measured against the
   CI-gated behavioral eval.

**Stretch (not the bar): rung 4.** If a Gumbel small-scale loop additionally beats
`LookaheadAgent` (strictly lower worst-case L1 spins/pass AND rounds, CI-separated,
100% finish, heat-efficiency confirming better budgeting), report it — but C4 is
**not** judged on it. The honest-stop rule applies: a *measured, CI-gated* result
against the behavioral held-out eval **is** the success, whether it clears rung 4 or
only criterion 1+2. The failure mode to avoid is a fake pass — a finish-rate-only,
USA-only, or point-estimate "win" (the S3/8C footgun, the C2/C3 discipline). If
Gumbel lifts the entropy (criterion 1) but the one-cycle improvement is still absent
or CI-overlapping (criterion 2 fails), that is a legitimate honest stop: it narrows
the binding constraint to the **value head** (C3 finding #2, the A/B-V warm-start
lever) and/or sim budget, reported as the go/no-go for C5, and the fallback to the
A/B/E spine stays on the table (README §5 gate-hard rule).

**C4 is the enabler, not the campaign.** Criterion 1+2 green is the **precondition**
for the deferred scale-up (README §7) — C4 does not run it. A green C4 is the
evidence that scaling the loop will buy improvement rather than a costlier plateau.

## Relationship to the A/B-V warm-start (C3 finding #2) — adjacent and separable

C3's recommendation #2 is to warm-start C's value head from an Option-A/B V via the
**`LeafEvaluator`** seam (C's `z` is definitionally the A/B `−rounds_remaining`, so
an A/B V is a drop-in leaf). That attacks a *different* bottleneck — the value
head's MAE *regressed* across C3's generations (4.78 → 6.47) under spin-spiral-
dominated MC targets — and lives behind a *different* seam than C4's
`RootActionSelector`. The two are **composable** (Gumbel fixes the policy target;
the A/B-V leaf fixes the value target) but deliberately **kept apart**: bundling
them would make a green result un-attributable. C4 changes only the policy target so
its improvement signal is Gumbel's alone. If C4 lifts the entropy but the one-cycle
improvement is still absent, that is the precise evidence the **value** is now the
binding constraint — which is exactly when the A/B-V warm-start (a possible **C5**,
or an optional `LeafEvaluator` knob) is the next lever. Noted here so the seam order
is explicit; not built in C4.

## Pre-C4 cheap diagnostic (decision-support, not a deliverable)

Before writing any Gumbel code, run a **sim-budget sweep on the existing `az_loop` /
`gen_selfplay`** — higher `--sims` (e.g. 16 → 32 → 64), **no new code** — and read
the reported π-entropy. This is a fast check of whether the π-entropy ~0.04 collapse
is partly a **budget artifact** (entropy climbs materially with sims) versus a
**structural property of the visit-count target at any affordable budget** (entropy
stays near the floor as sims rise). The two outcomes frame C4 differently:

- if entropy climbs sharply with sims, Gumbel is *confirmed necessary* — it is the
  way to get a non-collapsed target *without* paying the clone cost of the higher
  budget (the whole point of the low-visit guarantee);
- if entropy stays near the floor even at 64 sims, Gumbel is confirmed not merely
  *efficient* but *necessary* (more budget alone never un-collapses the count
  target), strengthening the case.

Either way the sweep is cheap, uses code already in the repo, and de-risks the
sprint by confirming the diagnosis before the implementation cost — it is decision
support, not a gated deliverable.

## Risks & mitigations

- **Risk: the completed-Q target is mis-constructed (e.g. completes unvisited
  actions with the wrong baseline), so it is non-collapsed but *wrong* — a confident
  push toward a bad action.** Mitigation: the toy-case improvement-guarantee test
  pins that the completed-Q mass shifts toward the *higher-Q* action vs the prior;
  the one-cycle rung-3 re-eval is the behavioral CI gate that catches a wrong-but-
  confident target as a *non*-improvement (it will not beat the prior). Completion
  uses the root's own normalized value estimate `v̂` for unvisited actions (the
  standard choice), unit-tested.
- **Risk: σ's scaling constants are mis-tuned, so `σ(Q̂)` either swamps the logits
  (search ignores the prior) or is inert (the Q signal never bites) — the C0 §D
  Q-scale hazard, now at the root.** Mitigation: reuse the **already-normalized**
  `_normalize_q(edge.q())` for `Q̂` (the same `[0,1]` MuZero min-max the interior
  uses — the unbounded `−rounds_remaining` scale is already tamed there), and treat
  `gumbel_c_visit`/`gumbel_c_scale` as C0/C4 constants validated by the non-collapse
  + improvement tests; report the entropy so an inert/swamped σ is visible.
- **Risk: Sequential Halving with `m > n_simulations` or a tiny budget degenerates
  (fewer sims than phases).** Mitigation: `gumbel_m` is capped at the kept-candidate
  count and at a value SH can afford given `n_simulations`; the budget-accounting
  test asserts no overrun and a sensible degenerate (`m=1` ⇒ prior-greedy root, the
  C1 `n_simulations=1` analogue).
- **Risk: C4 silently perturbs the C1/C2/C3 PUCT path (a refactor regression in a
  shared method), invalidating the parity eval and the C3 baseline.** Mitigation: the
  **byte-identical PUCT-default regression test** on a fixed `(state, seed)` is the
  hard guard; the `"gumbel"` behaviour is reached only when the toggle is explicitly
  set, default `"puct"`.
- **Risk: a global-RNG leak in the Gumbel draw breaks the determinism contract (the
  reproducibility the whole eval depends on).** Mitigation: Gumbel noise is drawn
  *only* from `_search_rng` (seeded off the turn signature, the C1 scheme); the
  seed-purity test asserts byte-stable output for fixed `(state, seed)`.
- **Risk: the entropy lifts but no behavioral improvement appears (Gumbel fixes the
  policy target, value is the real bottleneck).** Mitigation: this is an *expected,
  reportable* outcome, not a failure of C4's process — it is the honest go/no-go that
  routes to the A/B-V `LeafEvaluator` lever (C5). The CI gate + the value-MSE report
  make the diagnosis explicit rather than papered over.

## Dependencies

- **C1** (`MCTSAgent` + the five seams; the `RootActionSelector`/`SearchPolicy` seam
  is the one C4 swaps — `_edges_for`'s kept set + prior, `_normalize_q`, `_simulate`'s
  interior descent, the deterministic `_search_rng`/`_turn_seed` scheme are all
  reused unchanged).
- **C2** (`gen_selfplay` target generator — C4 adds a target branch and the
  `z`-pre-spin-floor is reused verbatim; `train_az` + the `.npz`/`MLAgent` contract
  are unchanged).
- **C3** (the warm prior `checkpoints/c3_warm_gen0.zip` is the loop start C4 runs off;
  `az_loop` + the `value_iterate` Wilson-LB / best-checkpoint promotion guard
  (`GateMetric.strictly_improves_on`, `run_iteration`'s `best_model`/`best_gate`) are
  unchanged — C4 plugs into them).
- `eval_mcts.py --compare-model` (rung-3 head-to-head) / `eval_az.py` (rung-4 CI gate)
  — both unchanged; the Gumbel-trained checkpoint is a plain `MaskablePPO` they load.
- The GPU (RTX 4080, torch cu126) for `train_az`; `_runlog.run_main`.

## Effort

~1 sprint. The interior search, the chance-node DPW, the leaf discipline, the codec,
the trainer, the loop, and the promotion guard are all C1/C2/C3 reuse and untouched.
The new work is contained: the Gumbel top-`m` + Sequential Halving root allocation,
the completed-Q target construction, getting σ's normalization right (reusing the
existing `_normalize_q`), and the test discipline that pins non-collapse +
improvement-guarantee + PUCT-parity + determinism. The bulk of the cost is the
*tests* and the honest one-cycle CI verdict, not the algorithm — the seam C1
reserved makes the swap a contained addition, not a rewrite.
