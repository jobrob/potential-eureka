# Sprint C5 — Warm-start + anchor the value head (fix the now-binding critic constraint)

> The second *post-foundation* sprint, the one C3 (finding #2) and C4 (its
> "Relationship to the A/B-V warm-start" note) both reserved by name. C0–C4 are
> built, green, and contract-checked. C4 **fixed the policy-target collapse** — the
> Gumbel completed-Q target lifted π-entropy from ~0.04 to ~0.46 at 16 sims — but
> the one-cycle loop still did not improve, and C4's clean PUCT control isolated
> *why*: with the policy target now good, **the value head (the critic predicting
> `z = −rounds_remaining`) is the binding constraint.** In the C4 Gumbel run the
> value head regressed to ~40-round MAE (vs the ~4.5-round critic C3's
> `mint_warm_prior` minted) because the more-exploratory Gumbel *trajectory* spun
> **23 of 32** self-play episodes to `MAX_ROUNDS`: most races never finished, the
> Monte-Carlo "rounds-to-finish" target had nothing to tally, the held-out val
> split emptied, early-stopping was disabled, and the critic trained uncontrolled.
> This sprint fixes the value-head signal so the loop can finally improve
> generation-over-generation. It is the **composition of three levers** — a value
> warm-start/anchor (primary), a data-starvation fix, and the val-split/`c_v`/
> schedule fixes — composed on top of C4's Gumbel target, which stays on.

## Resolved design decisions (2026-06-22)

Three choices are settled before C5 starts; the implementer follows these, not
their alternatives. (Genuinely open sub-choices are listed in **Open questions**.)

1. **Primary lever = the Option-A V warm-start/anchor (the `LeafEvaluator` seam),
   composed with the data-starvation fix and the val-split/tuning fix.** C's `z` is
   *definitionally* Option A's `−rounds_remaining` V, so the separately-trained
   `train_value` critic (already ~4.5-round MAE, already grafted into a checkpoint
   by `mint_warm_prior._graft_critic`) is a drop-in. The three levers ship together;
   they are *not* split into separate sprints, because C4 proved that fixing one
   bottleneck (the policy target) only exposes the next — bundling the value fix
   with the starvation fix is what gives the loop a chance to actually move.

2. **TD/bootstrap value target (the `value_target` seam, README §4.6) is OUT of
   scope — reserved as a possible C6.** It is pulled forward *only if*, after C5,
   the *variance of the MC target* (not starvation, not bad init) is shown to be the
   remaining limiter. C5 attacks init + starvation + tuning first because those are
   the mechanisms C3/C4 actually measured; a bootstrapped target is a larger,
   riskier change that should not be spent until the cheaper, measured causes are
   removed.

3. **Honest success = a *moving loop* (non-regressing value MAE + a generation that
   promotes), NOT a one-cycle rung-4 win.** The gap to rung 1 is ~7× rounds / ~76×
   spins and the borrowed Option-A V has its own plateau ceiling, so a single sprint
   is unlikely to leap to victory. "Beat `LookaheadAgent`" stays the explicit
   *stretch*, not the bar. See **Success criteria**.

## Goal

Make the value head a **stable, low-error, non-regressing** leaf across generations
so the closed loop finally produces a *monotone* per-generation improvement at small
scale. Concretely, three composed changes on top of C4's Gumbel target:

1. **Warm-start + anchor the critic from the Option-A value net** (the `LeafEvaluator`
   seam): each generation's critic is **initialized** from the calibrated warm V
   (not from scratch, not from the previous generation's collapsed critic) and is
   **regularized toward it during training** so self-play cannot wreck it.
2. **Fix the data starvation** that wrecked the C4 value target: a **trajectory
   temperature / exploration control** so the self-play *trajectory* finishes more
   races (rich Gumbel *target*, less reckless *driving*), plus a simple **more-tracks**
   lever so the finished-only val split survives the drops.
3. **Fix the mechanical bug + tuning**: guarantee a non-empty finished-only val set
   (so early-stopping is never silently disabled), retune `c_v` and the critic
   training schedule, and report **value-head MAE per generation as a first-class
   gate signal** (the single number to watch — contrast C4's 4.8 → 6.5 → 40 collapse).

The narrow, honest C5 success is the **moving loop**: value MAE stays ~4–5 rounds
and does not regress across generations, and at least one generation finally
*promotes* under the existing Wilson-LB / bootstrap-CI guard (which never happened
in C2/C3/C4). Beating `LookaheadAgent` (rung 4) is the *stretch*, not the bar.

## Why the value head is the right fix now (the C4 evidence, applied)

C4 ran a deliberately clean control so the attribution would be unambiguous:

- The **PUCT control** trained a *clean* value head (val vMAE **5.5 rounds**,
  best-val early-stop at epoch 26) and still could not improve — the C3 plateau,
  caused by the collapsed *policy* target. C4 fixed that.
- The **Gumbel run** fixed the policy target (π-entropy 0.468) but its exploratory
  trajectory drove **23/32** episodes to `MAX_ROUNDS`, **emptied the held-out val
  split**, disabled early-stopping, and left the value head badly fit (final-epoch
  train vMSE **252** vs PUCT's **57**; MAE ~40 rounds vs ~4.5). The Gumbel net's
  in-search gate (191 spins / 200 rounds) is therefore dominated by a **broken value
  head**, not a broken policy target.

So the binding constraint is now the critic, and it is broken by **two compounding
mechanisms C5 targets directly**: (a) it is trained from a poor init under a
spin-spiral-dominated MC target (init lever), and (b) the trajectory starves the MC
target of finished episodes and empties the val split (starvation + val-split
levers). Because C's `z` is *definitionally* the Option-A `−rounds_remaining` V, the
calibrated `train_value` critic is a ready-made anchor — the same disjoint graft
`mint_warm_prior` already performs, now applied *per generation* and *defended*
during training.

## Scope

**In.**

- **A persistent warm value net + a per-generation critic warm-start (the
  `LeafEvaluator`-seam init).** C5 mints (once, reusing `mint_warm_prior`'s critic
  half) or accepts a path to a calibrated Option-A value net — `gen_value_data` +
  `train_value`, the exact ~4.5-round-MAE critic C3 already grafts. Each generation's
  trainer is then **initialized** with that warm critic grafted in (the disjoint
  state-dict copy `mint_warm_prior._graft_critic` already implements, lifted into a
  reusable helper), rather than building a fresh random critic or inheriting the
  previous generation's. The actor half is initialized as today (from the warm
  prior's BC actor on gen 1, then from the best-promoted net). This severs the
  4.8 → 6.5 → 40 cross-generation critic drift at its root: every generation starts
  from a known-good critic.

- **A critic anchor / regularizer during training (the "don't wreck it" lever).**
  `train_az`'s joint loss currently trains **both** actor and critic with one Adam
  over `policy.parameters()` at a single `lr`, with no protection on the critic. C5
  adds a configurable critic-protection mechanism so self-play cannot drag the warm
  critic back into collapse. The **selected default** (subject to **Open question 1**)
  is an **L2 anchor to the warm critic**: an extra loss term
  `c_anchor · ‖θ_critic − θ_critic^warm‖²` over the critic-only parameter set
  (`train_value._critic_parameters(policy)` — the byte-disjoint
  `vf_features_extractor` / `mlp_extractor.value_net` / `value_net` modules), pulling
  the critic toward the calibrated warm V while still letting the MC targets refine
  it. Two alternates are wired behind the same toggle for the **Open question 1**
  decision: a **low critic learning rate** (a separate, smaller `lr_critic` on a
  second Adam param-group over `_critic_parameters`, the actor keeping the base `lr`)
  and **freeze-then-slow-thaw** (critic frozen for the first `freeze_critic_epochs`,
  then thawed at `lr_critic`). All three are minimal because the critic parameter set
  is already isolated by `share_features_extractor=False` and already enumerated by
  `train_value._critic_parameters`.

- **A trajectory-temperature / exploration control (the data-starvation fix).** The
  Gumbel *target* must stay rich, but the *trajectory* should not crash most races.
  Today `_search_visit_distribution` returns a single `acted_action` selected by the
  Gumbel-aware `_best_edge(root, at_root=True)`, and the trajectory temperature is
  entangled with the target. C5 **decouples trajectory exploration from target
  richness** with a new `gen_selfplay` knob — the **selected default** (subject to
  **Open question 2**) is a **trajectory greediness control**: a `traj_temperature`
  (or, equivalently, a `traj_greedy_after` ply count distinct from
  `temperature_moves`) that makes the *acted* move greedier (lower-variance, more
  finishes) while the *logged* completed-Q `π` target is unchanged. This is a pure
  trajectory/target decoupling — the Gumbel completed-Q target (C4) is untouched, so
  C4's entropy win is preserved while the trajectory stops spinning out. The toggle
  defaults to a value that measurably raises the finish rate vs the C4 Gumbel run
  (the unit test pins this).

- **A more-tracks lever for the val split.** C4 emptied the val split because all 8
  val tracks dropped to `MAX_ROUNDS`. C5 adds a simple **more-self-play-tracks**
  path (raise `--tracks` / `--val-tracks`; the per-generation seed slice
  `_C3_GEN_SLICE = 10_000` already has ample headroom) so that even with some drops
  the finished-only val split is non-empty. Combined with the trajectory fix the drop
  rate falls *and* the surviving sample grows.

- **A non-empty finished-only val-split guarantee + early-stopping safety (the
  mechanical bug).** Today, when every val episode drops, `train_az` silently warns
  "empty val set — early stopping disabled; saving final-epoch weights" and trains
  the critic uncontrolled — exactly the C4 failure. C5 makes this a **hard,
  first-class precondition**: after generation, assert the finished-only val split
  is non-empty (raise a clear error / skip the generation with a logged reason rather
  than silently training uncontrolled), and report the finished-vs-dropped val count
  per generation. The `z`-is-NaN drop already lives in `generate_dataset`
  (`keep = ~np.isnan(z)`); C5 only adds the *guarantee* that what survives includes a
  val split, surfaced loudly.

- **`c_v` + critic-schedule retuning, and value-MAE as a first-class gate signal.**
  `az_loop` already passes `--c-v` into `train_az`; C5 retunes it (and the new
  `c_anchor` / `lr_critic` / schedule knobs) on the track-disjoint val split, and
  **promotes `v_mae_rounds` to a per-generation gate signal** in the `az_loop` report
  (`train_az._evaluate` already computes `v_mae_rounds`; `az_loop` already stores
  `train_summary["best_val"]`). The loop prints and records the critic MAE every
  generation so a regression is caught immediately, the way `pi_entropy_mean` already
  is.

- **The toggles plumbed end-to-end, defaulting to the C4 behaviour OFF.** A new
  `--warm-value <path>` (the calibrated Option-A V), `--critic-anchor` mode +
  `--c-anchor` / `--lr-critic` / `--freeze-critic-epochs`, and `--traj-temperature`
  are threaded `az_loop` → `_train` / `_generate` → `train_az` / `gen_selfplay`. With
  the warm-value path unset and the anchor mode `"none"`, the trainer is the C4 path
  byte-for-byte; setting `--warm-value` + a non-`none` anchor turns C5 on. The codec,
  the `(obs, π, mask, z)` schema, the `MCTSAgent` acting/eval contract, the Gumbel
  target, the `.npz` contract, the promotion guard, and the seed-band disjointness are
  **unchanged**.

**Out.**

- **The TD / n-step / bootstrap value target (the `value_target` seam, README §4.6).**
  Reserved as a possible **C6**, pulled forward *only if* C5's diagnostics show the
  remaining limiter is the *variance of the MC target itself* (not init, not
  starvation). C5 deliberately exhausts the cheaper, measured causes first; see the
  future-extension note. Bundling a new value-target estimator now would make a green
  result un-attributable (the C4 discipline).

- **The scale-up campaign.** C5 is the *enabler* — a moving small-scale loop is the
  green light that earns the deferred `large`-net multi-hour campaign (README §7),
  NOT the campaign itself. Scaling a loop whose critic regresses every generation
  only buys a costlier plateau (the C3 finding #3 / C4 hazard).

- **Any change to the policy target or the Gumbel selector.** C4 owns those and they
  are kept *exactly*: C5 changes only the *value* side (init, anchor, trajectory,
  val-split, `c_v`) so its improvement signal is attributable to the value fix alone,
  the mirror of C4's "policy only" discipline.

- **Inference-time / acting changes.** The agent still acts and is evaluated with the
  unchanged PUCT interior search (C4 scope decision #3 carries over); C5 touches only
  self-play *generation* (trajectory) and *training* (critic), never the eval path.

- Opponents, MuZero, learned chance, interior-node Gumbel, KataGo forced-playout —
  unchanged from C1/C4 (the other deferred seams).

## Data-model / interface changes

- **`MCTSConfig` (`src/heat/agents/mcts_agent.py`)** — *no new fields required* for
  the primary path: the value warm-start lives in the trainer, and the trajectory
  control lives in `gen_selfplay`. (If **Open question 2** resolves toward a
  *search-time* trajectory knob rather than an acting-rule knob, a single
  `traj_temperature: float = <C4 default>` field is added with `__post_init__`
  validation, mirroring `temperature_moves`. Default leaves the C4/PUCT path
  byte-identical.)

- **`train_az.train_az` args** — new optional fields on the `Namespace`:
  `warm_value: str | None` (path to the calibrated Option-A V checkpoint; `None` =
  C4 behaviour), `critic_anchor: str` in `{"none", "l2", "low_lr", "freeze_thaw"}`
  (default `"none"`), `c_anchor: float`, `lr_critic: float`, `freeze_critic_epochs:
  int`. When `warm_value` is set, the trainer grafts the warm critic into the fresh
  `policy` before the optimizer is built (reusing the `_graft_critic` logic), records
  `θ_critic^warm` for the L2 anchor, and builds the optimizer per the anchor mode.

- **`gen_selfplay.generate_dataset` / `_generate_one_track` args** —
  `traj_temperature: float` (or `traj_greedy_after: int`) controlling only the
  *acted* move's greediness; the logged `π` target is unchanged. The
  `.selfplay.json` `search_config` block records the new value (provenance).

- **`az_loop` report (`.loop.json`)** — each generation entry gains
  `value_mae_rounds` (read from `train_summary["best_val"]["v_mae_rounds"]`) and the
  finished-vs-dropped val count, surfaced in `_print_report` next to the existing
  in-search/net-only line. No schema break — additive fields.

- **The `(obs, π, mask, z)` `.npz` contract, the codec, `save_checkpoint` /
  `MLAgent` / `NetAdapter` tripwire, and the seed bands are UNCHANGED.** The warm
  value net is a plain `train_value` checkpoint; the grafted critic is byte-disjoint
  from the actor (`share_features_extractor=False`), so the saved C5 net loads through
  the exact §3.4 contract the loop already consumes.

## Deliverables

- **`experiments/train_az.py`** — the value warm-start + anchor:
  - a `--warm-value <path>` arg and a `_graft_warm_critic(policy, warm_value_path)`
    helper (the `mint_warm_prior._graft_critic` state-dict copy, lifted to a shared
    function so both call sites use one implementation — the `_CRITIC_PREFIXES` /
    `train_value._critic_parameters` set) that initializes the critic from the warm V
    before the optimizer is built;
  - the critic-anchor mechanism behind `--critic-anchor {none,l2,low_lr,freeze_thaw}`
    + `--c-anchor` / `--lr-critic` / `--freeze-critic-epochs`: the L2 term added to
    the joint loss (default), the two-param-group / low-`lr_critic` optimizer
    alternate, and the freeze-then-thaw schedule alternate;
  - `_evaluate` already reports `v_mae_rounds`; surface it (and finished-val count)
    in the `train_az` summary so `az_loop` can gate on it. The `--warm-value`-unset +
    `--critic-anchor none` path is the C4 trainer byte-for-byte (a regression pin).
- **`experiments/gen_selfplay.py`** — the `--traj-temperature` (or
  `--traj-greedy-after`) knob decoupling the *acted* trajectory from the logged
  completed-Q target, plus the provenance record. The completed-Q `π`, the off-table
  drop-and-count, the `z` pre-spin floor, and the schema are otherwise untouched.
- **`experiments/az_loop.py`** — thread `--warm-value`, the critic-anchor knobs, and
  `--traj-temperature` through `_train` / `_generate`; add `value_mae_rounds` +
  finished-val count to the per-generation report and `_print_report`; add the
  **hard non-empty finished-val precondition** (raise / skip-with-reason instead of
  silently training an uncontrolled critic). The default (`--warm-value` unset) is the
  C4 loop unchanged.
- **The C5 experiment**: a small-scale closed-loop run with **C4's Gumbel target ON
  plus** `--warm-value` + the critic anchor + the trajectory fix + more tracks, off
  the C3 warm prior (`checkpoints/c3_warm_gen0.zip`), gated by the existing Wilson-LB
  / bootstrap-CI promotion guard — the apples-to-apples "did the value fix make the
  loop move" test against the C4 Gumbel-only run.
- **`C5-findings.md`** — the findings deliverable (mirrors C2/C3/C4-findings): the
  per-generation value-MAE curve (does it stay ~4–5 and NOT regress?), the
  per-generation in-search CI gate (does a generation finally *promote*?), the
  finish-rate / drop-rate before-and-after the trajectory fix, the cost report, and
  the **honest moving-loop-or-not verdict** — including the explicit fallback call:
  if the loop is *still* flat after both the value warm-start AND the starvation fix,
  that is the strong honest signal the approach won't beat the search agent, and the
  recommendation is to fall back to the A/B/E spine (README §5 gate-hard rule).
- **Unit tests** in `tests/test_c5_value_warmstart.py` (mirroring the C2/C3/C4
  discipline):
  - **critic warm-start init**: after `_graft_warm_critic`, the trained `policy`'s
    critic parameters equal the warm V's critic parameters (byte-equal on the
    `_critic_parameters` set) and the actor is untouched (the disjoint-graft contract,
    the `mint_warm_prior` property re-pinned at the new call site);
  - **the critic does NOT regress across a generation** — the headline correctness
    check: with the anchor on, train one generation on a deliberately
    starvation-prone batch and assert the value-head MAE on a held-out finished batch
    is **no worse** than the warm critic's (the 4.8 → 40 collapse cannot recur with
    the anchor); with the anchor *off* on the same batch, the MAE is allowed to
    regress (the anchor is load-bearing, not inert);
  - **non-empty finished-only val split is enforced**: an all-`MAX_ROUNDS` val band
    raises the new hard precondition (or skips with a logged reason), never silently
    trains an uncontrolled critic (the exact C4 mechanical bug, now guarded);
  - **the trajectory-temperature toggle measurably raises the finish rate**: on a
    fixed seed band, the greedier-trajectory setting finishes strictly more episodes
    than the C4 Gumbel-trajectory default, while the logged completed-Q `π` target
    for a fixed `(state, seed)` is **byte-identical** (trajectory decoupled from
    target — C4's entropy win preserved);
  - **C4 path unchanged**: with `--warm-value` unset + `--critic-anchor none` +
    the default trajectory, `train_az` and `gen_selfplay` produce **byte-identical**
    output to the pre-C5 code on a fixed `(data, seed)` / `(state, seed)` (a
    regression pin — C5 must not perturb the C4 path);
  - **seed-band disjointness preserved**: the warm-value precursor band (the
    `mint_warm_prior` 100_000 / 500_000 bands) stays disjoint from the C5 self-play
    bands (`_C3_SELFPLAY_BASE = 600_000` per-gen slices) and the held-out eval band
    (900_000) — asserted at startup, the existing `_assert_c3_band_disjoint` /
    `_assert_bands_disjoint` discipline re-pinned for the warm-value source;
  - **schema invariance / contract**: the C5-trained checkpoint passes the §3.4
    `MLAgent` / `NetAdapter` tripwire and the `(obs, π, mask, z)` guards (the C2/C4
    guards still pass).

## Success criteria (a moving loop — the honest bar, tied to the ladder)

The **primary C5 bar is a *moving loop*: monotone generation-over-generation
improvement at small scale**, on the held-out generated solo field, gated with the
honest discipline (behavioral held-out CI eval, never finish-rate / USA-only /
point-estimate). Three measured signals:

1. **Value-head MAE stays low (~4–5 rounds) and does NOT regress across
   generations** — the single number to watch. Contrast C4's 4.8 → 6.5 → 40 collapse:
   with the warm-start + anchor + starvation fix, the per-generation
   `v_mae_rounds` (already computed by `train_az._evaluate`, now surfaced in the loop
   report) stays near the warm critic's ~4.5 and is **non-increasing** across
   generations. This is the direct evidence the binding constraint is removed.

2. **A generation finally PROMOTES under the existing guard.** At least one
   generation's CI-gated in-search metric (worst-case L1 spins / rounds) is **no
   worse, and trends better**, generation-over-generation, so a generation clears the
   Wilson-LB / bootstrap-CI promotion guard (`az_loop.strictly_improves`) — which
   **never happened in C2/C3/C4**. This is "the loop can now move", measured against
   the CI-gated behavioral eval, not a training loss.

3. **Reported with the honest-gating discipline.** The verdict is the behavioral
   held-out CI eval (worst-case spins / rounds on the 900_000+ band), with the
   value-MAE curve and the finish/drop-rate before-and-after the trajectory fix. The
   failure mode to avoid is a fake pass — finish-rate-only, USA-only, or
   point-estimate (the S3/8C footgun the whole line gates against).

**A moving small-scale loop is the green light that earns the deferred scale-up
campaign.** Criterion 1 + 2 green is the *precondition* for the `large`-net
multi-hour campaign (README §7); C5 does not run it.

**Stretch (not the bar): rung 4.** If the C5 loop additionally beats
`LookaheadAgent` (strictly lower worst-case L1 spins/pass AND rounds, CI-separated,
100% finish, heat-efficiency confirming better budgeting), report it — but C5 is
**not** judged on it. The gap to rung 1 is ~7× rounds / ~76× spins and the borrowed
Option-A V has its own plateau ceiling, so a one-cycle leap to rung 4 is not
expected.

**Honest-stop criterion (explicitly allowed, and decisive here).** If the loop is
**still flat after C5** — value head warm-started AND data-starvation fixed, yet no
generation promotes and the value MAE is stable but the behavior does not move — that
is the **strong honest signal the approach won't beat the search agent**, and the
recommendation is to fall back to the A/B/E spine (README §5 gate-hard rule). A
measured, CI-gated non-beat that has *eliminated* both the policy-target collapse
(C4) and the value-head collapse (C5) is the most informative possible stop: it says
the remaining gap is structural, not a fixable plumbing bottleneck. The only further
value-side lever is the reserved C6 (the MC-target-variance / TD seam), pulled forward
*only* if C5's diagnostics point there.

## Future extension (reserved C6, behind the `value_target` seam — NOT now)

**TD(λ) / n-step / bootstrapped value target.** README §4.6 fixes the core's value
target as the **unbiased Monte-Carlo return** precisely so C's `z` is definitionally
the Option-A/B V (which is what makes C5's warm-start a drop-in). The cost of that
unbiasedness is **variance**, especially on long, spin-spiral-prone solo episodes. If,
after C5, the value head is well-initialized and well-anchored and the trajectory
finishes most races and the val split is healthy — and the loop *still* will not move
because the surviving MC targets are too high-variance to learn from — then the next
lever is to replace the pure MC return with a lower-variance **bootstrapped** target
(n-step or TD(λ) off the trained critic). That is a `value_target`-seam swap, a
larger and riskier change (it reintroduces bootstrap bias and breaks the clean
A/B-V equivalence), and it is reserved as **C6**, pulled forward **only** when C5's
diagnostics show MC-target *variance* — not init, not starvation, not tuning — is the
remaining limiter.

## Risks & mitigations

- **Risk: the anchor is too strong — the warm critic is frozen so hard the loop
  cannot refine it past Option-A's plateau ceiling (a "stuck at the borrowed V"
  non-beat that looks like the warm critic's own limit, not C's improvement).**
  Mitigation: `c_anchor` / `lr_critic` are tuned on the val split so the critic *can*
  move toward the (good) MC targets while being *prevented* from collapsing on the
  starved ones; the per-generation value-MAE curve + the promotion gate make a
  "stuck at the teacher" outcome visible and *reportable* — it is itself the honest
  signal that routes to the C6 / fallback decision, not a hidden failure.

- **Risk: the anchor is too weak — the starvation-driven gradients still drag the
  critic back to the 40-round collapse (the C4 failure recurs).** Mitigation: the
  "critic does NOT regress across a generation" unit test is the hard guard (anchor
  on ⇒ MAE no worse than warm; anchor off ⇒ allowed to regress, proving the anchor is
  load-bearing); the loop's first-class value-MAE gate catches a regression the
  generation it happens, before the scale-up is paid for.

- **Risk: the trajectory-greediness fix lowers exploration so far that the self-play
  data carries no new information (the trajectory finishes but only ever drives the
  one safe line — a different starvation).** Mitigation: the Gumbel completed-Q
  *target* is untouched (still high-entropy, C4), so the *learning signal* per logged
  state stays rich even when the *acted* line is greedier; the toggle is tuned to the
  *minimum* greediness that lifts the finish rate, and the finish-rate unit test pins
  the floor, not a maximum.

- **Risk: bundling three levers makes a green (or red) result un-attributable — the
  C4 anti-bundling discipline.** Mitigation: the three levers are *all on the value
  side* (init, trajectory, val-split/tuning) and the policy/Gumbel target is held
  fixed, so the C5 signal is attributable to "the value fix" as a unit vs C4's
  "Gumbel only"; the per-lever toggles (warm-value path, anchor mode,
  trajectory-temperature) each default OFF and can be ablated individually in the
  findings if the headline result needs decomposing.

- **Risk: an empty finished-val split silently disables early-stopping again (the
  exact C4 mechanical bug).** Mitigation: C5 makes a non-empty finished-only val
  split a **hard precondition** (raise / skip-with-reason), unit-tested on an
  all-`MAX_ROUNDS` band — the critic is never trained uncontrolled again.

- **Risk: C5 silently perturbs the C4/C3/C2/C1 path (a refactor regression in the
  shared trainer / generator), invalidating the C4 baseline.** Mitigation: the
  **byte-identical C4-default regression test** (`--warm-value` unset + anchor
  `none` + default trajectory) on a fixed `(data, seed)` / `(state, seed)` is the
  hard guard; C5 behaviour is reached only when the toggles are explicitly set.

- **Risk: the warm-value precursor band leaks into the self-play or eval bands (a
  generalization leak — the review's bug #6/#9).** Mitigation: the warm-value source
  is the `mint_warm_prior` 100_000 / 500_000 precursor bands, already asserted
  disjoint from the 600_000 per-gen self-play slices and the 900_000 eval band; the
  seed-band disjointness test re-pins this for the warm-value source.

## Dependencies

- **C4** (the Gumbel completed-Q policy target in `gen_selfplay`; the
  `--root-selector gumbel` path — C5 runs *with Gumbel on*, fixing the value side
  while keeping C4's policy-target win).
- **C3** (`az_loop` + the `value_iterate` Wilson-LB / best-checkpoint promotion guard
  — `strictly_improves` / `LoopGate` — unchanged; the warm prior
  `checkpoints/c3_warm_gen0.zip` is the loop start; `mint_warm_prior._graft_critic` /
  `_CRITIC_PREFIXES` are the warm-start machinery C5 lifts and reuses per generation).
- **Option A** (`experiments/gen_value_data.py` + `experiments/train_value.py` —
  the ~4.5-round-MAE calibrated `−rounds_remaining` critic that is C5's warm-start
  source; `train_value._critic_parameters` is the exact byte-disjoint critic set the
  anchor / graft operate over).
- **C2** (`train_az` — C5 adds the warm-start/anchor branch; the joint AZ loss,
  `_masked_log_probs`, `_policy_ce`, the `(obs, π, mask, z)` schema, and the
  `share_features_extractor=False` actor/critic decoupling are reused verbatim).
- **C1** (`MCTSAgent` + the `LeafEvaluator` seam — `_evaluate_leaf` → `NetAdapter.
  leaf_value` is the seam site whose value the warm-started critic now supplies; the
  interior search, chance-node DPW, leaf floor, and codec are untouched).
- The GPU (RTX 4080, torch cu126) for `train_az`; `_runlog.run_main`.

## Effort

~1 sprint. The warm-start graft, the critic anchor, and the trajectory/val-split
fixes are all reuse over isolated, already-enumerated machinery: the critic
parameter set is byte-disjoint and already named (`train_value._critic_parameters`),
the graft already exists (`mint_warm_prior._graft_critic`), the joint trainer and the
loop are C2/C3, the Gumbel target is C4, and the promotion guard is unchanged. The new
work is contained — the per-generation warm-critic init, the anchor term/optimizer
variants, the trajectory-greediness decoupling, the non-empty-val guarantee, and the
value-MAE gate signal — and, as with C4, the bulk of the cost is the **tests** (the
non-regression critic guard, the trajectory-finish-rate pin, the C4-parity
regression pin) and the **honest moving-loop verdict**, not the algorithm.
