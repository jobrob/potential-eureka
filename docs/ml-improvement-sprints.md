# ML Improvement Sprints — generalization run, sequenced by ROI

> **Status:** planning / design only — **no code yet**.
> **Input:** `docs/ml-performance-improvements.md` (Ideas 1–18, all marked "DECIDED").
> **This doc is deliberately NOT a yes-man to that input.** It re-scores every
> idea on improvement-for-effort, **drops/defers the ones that aren't worth it**,
> and re-sequences the survivors so that **cheap, measurable, high-confidence
> wins land first** — before the expensive architecture change and the long run.
>
> Goal (unchanged): an agent that **wins on unseen procedurally generated
> tracks**, not one memorized track.

---

## TL;DR

The source doc's instinct to fold everything into one giant codec-v2 / extractor
rewrite (its "Suggested build order" step 1) is the main thing this plan
**rejects**. That order sinks the largest, riskiest effort *first*, before a
single number has been measured, and before the training loop is even
trustworthy enough to tell us whether the big change helped.

We invert it:

1. **Sprint A — make the run trustworthy & measurable (cheap).** The Tier-1
   training-loop fixes (Ideas 7/8/9/10) + config levers (3/4/15). These are the
   gate, eval, and budget changes that decide whether *any* later experiment can
   be believed. None of them touch the obs contract.
2. **Sprint B — cheap-obs baseline + curriculum (Option A, NOT Option B).**
   Establish the generalization number with the *cheap* whole-track obs and the
   curriculum, so Option B has a baseline to beat.
3. **Sprint C — the big architecture change (Option B + its companions), but
   only if B's number says we need it.** Codec-v2, CNN/attention extractor, and
   the ideas that genuinely share that bump (11, 12, 13, 14).
4. **Sprint D — net size + the long run + honest eval.**

The single most important critical call: **do not build Idea 1 Option B up front.
Build Option A first (Sprint B), measure, and only escalate to B if the cheap
version plateaus below target.** The source doc rejected A "for headroom and true
skill" — a justification with *zero measured evidence*, when the doc itself
concedes "nothing has been measured yet."

---

## 1. ROI scorecard (independent, skeptical)

Verdict key: **HIGH** = do it, cheap and high-confidence · **MED** = worthwhile
but conditional/later · **LOW** = marginal, do only if it falls out for free ·
**DROP/DEFER** = not needed now; explicitly cut.

| # | Idea | Verdict | Why (skeptical take) |
|---|------|---------|----------------------|
| 7 | In-Phase-1 best-checkpoint + periodic eval | **HIGH** | Real bug. Under Idea 4 the Phase-1 gate fires *once* (`training.py:778`); the periodic gate lives only in the Phase-2 loop (`:914-919`). A Phase-1-heavy run keeps the **final** weights even if the policy peaked and regressed. Without this, the whole expensive run is untrustworthy. Cheapest possible high-value fix. |
| 8 | Gate vs the trained-against opponent | **HIGH** | Confirmed: `_gate_score` calls `evaluate_ml` with no `opponent_factory` (`training.py:666`), defaulting to the **weak** `HeuristicAgent` even when training vs `StrongHeuristicAgent`. Best-checkpoint selection saturates near 1.0 and loses discriminating power. ~tiny effort. |
| 9 | Gate statistical robustness (more games + Wilson LB) | **HIGH** | `gate_games=20` over 3 holdout tracks = ~6/track (`:648`), ±20% CI. Idea-7 selection rides on this noise. `wilson_interval` is already imported in `evaluate.py`. Cheap, and it's the difference between "we picked the best checkpoint" and "we picked a lucky one." |
| 4 | Phase-1-heavy budget / minimal Phase 2 | **HIGH** | Two runs + Sprint-5 collapse confirm Phase 2 is net-negative here. Trivial config. Frees ~half the compute. (Note: it is *because* of this decision that 7/8/9 become mandatory, not optional.) |
| 3 | Stronger/longer reward shaping | **MED→HIGH** | Cheap (config: `shaping_weight_start`, slower anneal) and the sparse generated-track problem is exactly where shaping earns its keep. The optional anti-spinout *term* is MED (new bounded term in `step_reward`). Keep the term optional. |
| 15 | `normalize_obs` OFF for Option B | **HIGH (as a guardrail), but free** | It's already the default (`normalize_obs=False`). This is "don't turn it on," not a build task. Effort ≈ 0; record it as a constraint, not a sprint item. |
| 2 | Difficulty curriculum | **MED** | Genuinely useful for a harder distribution, BUT not "small": `TrackGenParams` is **frozen at `TrackSampler` construction** — there is no step-aware param schedule today. Needs a stateful sampler threaded with a step counter. Worth doing, but it's a real change, and it's only measurable *after* Sprint A makes the gate trustworthy. |
| 10 | Randomize learner seat / start | **MED** | Legitimate overfit risk (`learner_id` pinned to 0 everywhere; eval always seats MLAgent at 0). Obs is already relative, so well-posed. But it's slightly cross-cutting (reset must re-pick seat + rebuild opponents; eval harness must vary seat to *measure* the benefit). Cheap-ish, real, do it in Sprint A. |
| 1B | Whole-track obs + CNN/attention extractor | **DEFER (escalate-only)** | The foundational change — and the one to *not* build first. See §2. Its benefit is **unobservable until Option A + curriculum are measured.** Building B before A is committing the largest effort against an unmeasured hypothesis. |
| 1A | All-corners fixed-slot obs + plain MLP | **HIGH (promoted from "rejected")** | The doc rejected A with no data. A captures whole-track planning for ≤7-corner tracks (the generator caps at 7: `num_corners_range=(3,7)`), stays an MLP, needs only a modest `OBS_DIM` bump, and is **the cheapest way to test the core hypothesis** ("the agent is blind to the track"). This is the right first obs experiment. |
| 11 | Widen REACT action table | **MED (conditional)** | The 8-slot table (`action_codec.py:149-158`) does omit combos. But "no whole-track obs fixes an action it can't emit" cuts both ways: **widening the table is pointless until we know the policy is bottlenecked on react expressiveness**, which we can't know pre-measurement. Defer to the codec-v2 sprint, and only after an audit shows missing combos are *ever optimal*. |
| 12 | Slipstream-context obs features | **MED** | Real gap (single "available" bit, `features.py:213`). Cheap *if* folded into an obs rework. Not worth a standalone codec bump. Rides Sprint C. |
| 13 | Decision-kind conditioning + unshared extractor | **MED (split it)** | Two sub-parts. `share_features_extractor=False` is a **one-line `policy_kwargs` change** — HIGH value, do it early (Sprint A/B, no codec dependency). The skip-connect/FiLM conditioning rides the extractor rewrite (Sprint C). The multi-head contingency is correctly gated behind a diagnostic — **DROP from planning** until that diagnostic fires. |
| 14 | Cache static track encoding | **HIGH — but only conditional on B** | Correct and necessary *if* Option B ships (the per-step grid build would tank the ~1080 steps/s throughput). Irrelevant for Option A (cheap obs). So it's a **Sprint-C design constraint**, not an independent idea. |
| 5 | Grow the net (xlarge profile) | **MED** | `NET_PROFILES` already has `small`/`large`; adding `xlarge` is trivial. But bigger net only pays off with more signal (richer obs) and more steps. Sequence it with the long run (Sprint D), not before. |
| 6 | Minor levers (entropy warm-up, opp mix, longer run) | **MIXED** | "Longer run" = HIGH (it's the dominant lever once structure is fixed → Sprint D). Broader Phase-1 opp mix = LOW-MED (cheap, fold into Sprint A). Entropy warm-up / `reload_best_on_regression` = LOW (only matters if we keep Phase 2, which Idea 4 says we mostly don't). |
| 16 | Partial-progress reward on truncation | **DROP (defer)** | The doc itself calls it "rare at current track lengths." Truncation at `round_num > 200` essentially never fires. This is insurance against a problem we don't have. Revisit *only if* curriculum/long episodes start truncating (watch the truncation rate; it's free to log). |
| 17 | Evaluate sampled *and* deterministic | **LOW (keep, ~free)** | Genuinely cheap (eval already supports a `deterministic` flag via `ml_agent_factory`) and a real exploitability check in a hidden-info game. Fold into Sprint D's eval, don't sprint it. |
| 18 | North-star lookahead/search opponent | **DROP (defer)** | Speculative and the *most* expensive "opportunistic" item (a whole new search agent). We don't yet beat the strong heuristic on generated tracks — measuring headroom *beyond* that ceiling is premature. Build it only once we're saturating the strong-heuristic bar. |

### Ideas explicitly cut from this plan (with one-line rationale)

- **Idea 1B (build-first):** dropped *as the first step*; deferred behind a
  measured Option-A baseline. Built only if A plateaus below target.
- **Idea 13 multi-head head:** dropped from planning — already gated behind a
  concrete diagnostic that hasn't fired.
- **Idea 16 (truncation reward):** dropped — guards a non-occurring failure
  (truncation is effectively never hit). Replaced by a free truncation-rate log.
- **Idea 18 (north-star opponent):** dropped — premature; we don't yet beat the
  opponents we already have on generated tracks.
- **Idea 6 entropy warm-up / `reload_best_on_regression`:** dropped — only
  relevant to a Phase 2 that Idea 4 minimizes away.
- **Idea 15 as a "task":** dropped as a build item — it's already the default;
  recorded as a constraint instead.

---

## 2. The Option A vs Option B call (the central critique)

The source doc commits to **Option B** (full track-map grid + CNN/attention,
`OBS_DIM` 72→~170+, codec v2) and rejects **Option A** (all-corners fixed slots +
MLP) "for headroom and true skill." Three reasons that's the wrong order:

1. **No measurement exists.** The doc's own opening says "nothing has even been
   measured." Choosing the larger, riskier architecture over the cheaper one with
   no baseline is an unforced bet on the most expensive surface area in the bundle.
2. **Option A may already be "the whole track."** The generator caps corners at
   `num_corners_range=(3, 7)` (`generator.py:47`). With `MAX_CORNERS≈8`, Option A's
   padded slots encode *every* corner of *every* generated track, plus inter-corner
   distances = the straights. The doc admits this ("with ≤7 corners this *is* the
   whole track"). The "headroom" B buys is for tracks the generator can't even
   produce yet.
3. **A is the clean experiment for the core hypothesis.** Root cause #1 is "the
   agent is nearly blind to the track (only 4 track dims, next-corner only)."
   Option A directly tests *"does giving it all corners fix generalization?"* with
   an MLP and a small `OBS_DIM` bump — no CNN to tune, no per-step caching problem
   (Idea 14 becomes moot), no attention masks. If A closes most of the gap, B's
   cost is unjustified. If A plateaus, we escalate to B *with a baseline to beat.*

**Decision for this plan:** Sprint B builds **Option A**. Sprint C builds
**Option B only if A's held-out win-rate plateaus below target.** This keeps B —
and its expensive companions (11/13-conditioning/14) — gated behind evidence.

---

## 3. Sprints

Dependencies form a short chain: **A → B → (C if needed) → D**. A is pure
training-loop/config and unblocks trustworthy measurement for everything after.

### Sprint A — Trustworthy & measurable run (cheap, no contract change)

> **Detailed design:** [`docs/sprint-A-trustworthy-run.md`](sprint-A-trustworthy-run.md) — implementation-ready spec (per-idea fix, signatures, tests, DoD).

| | |
|---|---|
| **Goal** | Make the training loop honest: pick the genuinely-best checkpoint, gate against the real opponent with statistical confidence, stop wasting budget on collapsing self-play, and remove the seat-0 overfit — all without touching `OBS_DIM`/`ACTION_DIM`. |
| **Ideas** | 7 (Phase-1 periodic eval + best-checkpoint), 8 (gate vs strong), 9 (more games + Wilson LB), 10 (seat randomization), 4 (phase budget), 3 (shaping config; anti-spinout term optional), 13-part-1 (`share_features_extractor=False`), 6 (broader Phase-1 opp mix). 15 recorded as a constraint. |
| **Files** | `training.py` (Phase-1 periodic gate + best-checkpoint hook around `_gate`/`_save_best`; pass strong factory + Wilson-LB into `_gate_score`; raise `gate_games`; `_scripted_opponents` mix), `evaluate.py` (let `evaluate_ml`/gate take an opponent factory + return Wilson LB; allow seat ≠ 0 for measuring Idea 10), `env.py` (`reset()` re-picks `learner_id` + rebuilds opponents when seat-randomization is on), `model.py`/`PPOConfig` (`share_features_extractor=False`; shaping defaults), optionally `spaces.py` (`step_reward` anti-spinout term, bounded, default-off). |
| **Effort** | Small–moderate (mostly training-loop plumbing; no contract bump). |
| **Risk** | Low. Seat randomization is the only mildly cross-cutting piece — covered by an env test asserting obs/opponents stay consistent across seats. |
| **Dependencies** | None. Must land **before** any long run. |
| **Definition of done** | (a) A Phase-1-only run preserves and reports the best held-out checkpoint, not the final one, with the gate score on TensorBoard. (b) Gate reports **both** weak- and strong-heuristic win-rates and promotes on the **Wilson lower bound**. (c) Gate plays enough games (e.g. ≥40/track) that the LB is usable. (d) Training episodes see randomized learner seats; a seat-swept eval confirms no seat-0 overfit. (e) Full suite green: `PYTHONPATH=src python -m pytest tests/ -q`. |

### Sprint B — Cheap whole-track obs (Option A) + curriculum baseline

> **Detailed design:** [`docs/sprint-B-cheap-obs-curriculum.md`](sprint-B-cheap-obs-curriculum.md) — implementation-ready spec (v2 obs layout, step-aware curriculum, tests, DoD).

| | |
|---|---|
| **Goal** | Test the core hypothesis ("the agent is blind to the track") with the **cheapest** obs that gives global planning, plus the difficulty curriculum, and **establish the generalization number Option B must beat.** |
| **Ideas** | 1A (all-corners fixed-slot obs + MLP, small `OBS_DIM` bump → codec v2), 2 (track-difficulty curriculum). |
| **Files** | `features.py` (replace 4-dim `_track_lookahead` with an all-corners ego-centric block: `MAX_CORNERS × (dist-ahead, speed_limit, corner_len, lanes)` + globals laps-remaining/dist-to-finish/pool/pos-in-lap), `spaces.py` (`OBS_DIM`, block sizes, **`CODEC_VERSION` 1→2**), a new generated-track feature test (shape/bounds, ego-centric + padding correctness), `generator.py`/`TrackSampler` (a **step-aware** params schedule — interpolate `num_corners_range`/`length_range`/`laps` easy→full), `training.py` (thread the curriculum step counter into the sampler; `CurriculumConfig` schedule fields). |
| **Effort** | Moderate. The obs change is real but MLP-only (no extractor rewrite). The curriculum's stateful sampler is the trickier half (the current sampler freezes params at construction). |
| **Risk** | Low–moderate. One codec bump (acceptable; checkpoints are throwaway). Curriculum can mis-shape if the schedule is too slow — keep it bounded and ablate. |
| **Dependencies** | Sprint A (need the trustworthy gate to *believe* the resulting number). |
| **Definition of done** | A medium-length run on generated tracks (curriculum on) reports a held-out generalist win-rate vs strong heuristics that we **record as the Option-A baseline.** Decision gate: if it clears target decisively, Sprint C may be unnecessary; if it plateaus below target, Sprint C is justified *with a baseline to beat.* Feature tests + full suite green. |

### Sprint C — Option B architecture (escalate-only) + obs companions

| | |
|---|---|
| **Goal** | If and only if Option A plateaus: richer whole-track representation (grid/CNN or corner-sequence/attention) for headroom, bundling the companions that genuinely share the codec-v2 / extractor rewrite. |
| **Ideas** | 1B (track-map + CNN/attention extractor), 12 (slipstream-context obs features), 13-part-2 (skip-connect/FiLM decision-kind conditioning), 14 (static-track caching — **design constraint, build in from the start**), 11 (wider REACT table — only if a combo audit shows missing plays are ever optimal). |
| **Files** | `features.py` (track block + reshape contract + **per-`reset()` static-track cache**, ego-shift per step), `spaces.py` (`OBS_DIM`, codec — *already* v2 from Sprint B, so if C immediately follows B, combine into a single v2; if B shipped first, this is **v3**), `model.py` (custom `BaseFeaturesExtractor`: scalar-MLP + CNN/attention track encoder, skip-connect the kind one-hot; `xlarge`-ready widths), `env.py` (hold the cached track encoding across steps), `action_codec.py`/`spaces.py` (REACT table + `REACT_SIZE`/`ACTION_DIM`, only if the audit justifies it), extractor + caching + padding-mask tests. |
| **Effort** | Large. Biggest surface area in the program. |
| **Risk** | Moderate. Isolated behind the extractor + codec bump, but CNN/attention tuning + caching correctness + a possible action-space widening all land together. Mitigated by Sprint A/B having de-risked the loop and given a baseline. |
| **Dependencies** | Sprint B (baseline + decision to escalate). **Skip entirely if B clears target.** |
| **Definition of done** | Option-B run beats the recorded Option-A baseline on held-out generated tracks (Wilson-LB, vs strong) by a margin worth its cost; throughput stays acceptable (caching verified — no per-step grid rebuild); extractor/caching/mask tests + full suite green. If B fails to beat A, **revert to A** (the doc's own honest-ceiling fallback). |

### Sprint D — Net size, the long run, honest eval

| | |
|---|---|
| **Goal** | Spend the big compute on the winning architecture, sized correctly, and report skill honestly. |
| **Ideas** | 5 (`xlarge` net profile), 6 ("longer run" — the dominant lever), 17 (report sampled *and* deterministic). |
| **Files** | `model.py` (`NET_PROFILES["xlarge"]`), a throughput probe + the long-run launch config, `evaluate.py`/eval script (report sampled & deterministic, weak & strong, held-out generated + USA reference; log truncation rate to retire Idea 16). |
| **Effort** | Small build + long wall-clock. |
| **Risk** | Low (the structural risk was retired in A–C). |
| **Dependencies** | The winning obs sprint (B, or C if escalated). |
| **Definition of done** | Final agent **wins decisively on unseen generated tracks** vs strong heuristics under the Wilson-LB gate, reported for both sampled and deterministic policies, with a USA reference. Truncation rate logged (confirms Idea 16 stays unneeded). Full suite green. |

---

## 4. Sequencing rationale (why this order, vs the doc's)

- **Cheap, high-confidence, measurable first.** Sprint A is all training-loop and
  config — no contract bump — and it's what makes every downstream number
  *believable*. The doc buried these (its step 3) *after* the giant obs change;
  but if the gate is selecting on 6 noisy games vs the wrong opponent (Ideas
  8/9) and keeping the final-not-best checkpoint (Idea 7), Sprint C's result is
  uninterpretable. Trust fixes must precede the expensive run. The doc's own
  Tier-1 note agrees they "decide whether the run can be trusted" — this plan
  just refuses to put them *after* the thing they're meant to validate.
- **Baseline before architecture.** Option A (Sprint B) is the controlled
  experiment for root-cause #1 at a fraction of B's cost, and it produces the
  number that decides whether B is even worth building.
- **One codec bump, honestly accounted.** The doc's strongest real argument is
  "several ideas share the codec-v2 bump, so do them together." True — but that's
  an argument for bundling **12/13-conditioning/14/(11)** *with whichever obs
  sprint ships*, **not** for doing the most expensive obs option first. If Option
  A suffices, those companions ride A's v2 and B-the-architecture never happens.
  If we escalate, they ride C. Either way they share exactly one bump.
- **Net size and long run last.** A bigger net (Idea 5) and "just train longer"
  (Idea 6) only pay back once the obs + loop are right. Sizing before that wastes
  the most wall-clock.

---

## 5. Open questions (decide at build time)

- **Sprint B `MAX_CORNERS` / globals → final `OBS_DIM`.** Generator caps at 7
  corners; pick `MAX_CORNERS=8` for slack and confirm padding correctness.
- **Curriculum schedule shape** (linear vs staged) and step horizon — ablate
  against no-curriculum once Sprint A's gate is trustworthy.
- **Shaping magnitude / anti-spinout weight** — keep bounded; gate against
  degenerate play.
- **Sprint A gate cadence vs cost** — the gate plays real games; tie cadence to
  the raised game count (Idea 9) so periodic eval doesn't dominate wall-clock.
- **If escalating to C: grid-CNN vs corner-sequence-attention** — prototype both
  on the feature test before committing the long run.
- **REACT audit (Sprint C, Idea 11)** — confirm against the engine's
  `legal_react_options` which missing combos are *ever* optimal before widening.

---

## 6. Validation (unchanged methodology)

Train on generated tracks; **gate and evaluate on held-out generated tracks**
(seeds disjoint from training, per `_HOLDOUT_TRACK_SEEDS`), vs **both** weak and
strong heuristics, for **both** sampled and deterministic policies, plus a USA
reference. Every sprint ends by writing tests and running the full suite
(`PYTHONPATH=src python -m pytest tests/ -q`). Target: decisive wins on *unseen*
generated tracks under the Wilson lower-bound gate.
