# Option A — Learned leaf value plugged into the existing search

> **Status:** Sprint-broken plan, pre-build. The cheapest, de-risking realization
> of the shared **value-function spine**
> ([`../README.md`](../README.md), [`../../solo-speed-whole-track-planning-options.md`](../../solo-speed-whole-track-planning-options.md)).
> Read those two first for the scope, the shared spine, the hard constraints, and
> the shared evaluation methodology this plan inherits verbatim.
>
> **Input:** the S1/S2 `LookaheadAgent` (built, green — see
> `docs/sprint-search-imitation-design.md` §4), the frozen ML codec, and the
> model's separate critic/value head.

---

## 1. Goal

Replace the `LookaheadAgent`'s weak leaf evaluator — today the lap-aware
`progress` scalar (`_move_eval.race_progress`) or a reckless `move_eval` — with a
**trained rest-of-lap value**

> **V(state) ≈ expected rounds-to-finish** (a cost-to-go), our own future draws
> taken in expectation,

so that **shallow search + a good leaf beats deep search + a bad one**. This
attacks hard constraint #1 head-on ("depth is inert; the lever is leaf quality")
and is the spine every later option (B/C/E) reuses.

The win condition is narrow and measurable: at a **fixed, cheap horizon**
(target horizon 1, fall back to 2), `leaf_value="learned"` must **match or beat**
the current `leaf_value="progress"` agent at its *best* horizon on the shared
gate — same worst-case limit-1 spins, same-or-better rounds-to-finish — at
**lower ms/move** (because the learned leaf removes the need for depth and for
the heuristic rollout's per-step simulation).

## 2. Design decisions (resolved)

### 2.1 What V predicts, and the training signal

- **Target = cost-to-go, expressed as expected rounds-to-finish from this state.**
  Concretely, for a logged state at round `r` whose game finishes the solo car at
  round `R`, the regression target is `R - r` (rounds remaining). We store and
  regress **rounds-remaining** (a non-negative scalar), and the leaf consumes
  **`V_leaf = -rounds_remaining`** so that, like `progress`, **higher is better**
  and it slots into the existing `_leaf_score` sign convention with zero changes
  to the penalty terms. Rationale for "rounds" over "discounted progress": the
  shared spine *is* rounds-to-finish (≈ lap time, the primary gate metric), so
  the leaf is trained directly on the quantity we are optimizing — no
  progress→time proxy to mis-calibrate, and the heat-budgeting signal ("running
  out of heat costs rounds") is captured natively because a state that will stall
  for heat genuinely takes more rounds to finish in the MC rollout.

- **Trained by Monte-Carlo returns, NOT TD bootstrapping.** We own a perfect,
  fast simulator (`GameState.clone()` + `run_round_driver`, profiled at ~16 µs /
  clone, ~64k/s in S2) and the solo horizon is short and bounded (the gate's
  2-lap tight tracks finish in ~20–26 rounds). MC gives **unbiased** targets at
  trivial cost and sidesteps the bootstrapping instabilities that plagued the
  8C/self-play saga (memory `8c-production-collapse`). TD's only advantage —
  sample efficiency on long/expensive-to-roll horizons — does not apply here. We
  therefore generate complete rollouts to the finish and label every visited
  state with its realized rounds-remaining. (TD is noted as a *deferred* lever in
  Open Questions only if MC variance proves too high at the limit-1 corners.)

### 2.2 Own-draw expectation

- **Keep the existing `n_determinizations` averaging at search time; train V on
  the expectation directly via MC.** V is a deterministic `state → scalar` map; it
  does **not** itself average over draws. Two complementary mechanisms cover the
  own-draw chance node, and they compose cleanly:
  1. **At training time**, each MC rollout is one sample of the own-deck draw
     process, so regressing across many rollouts makes V learn the *expected*
     rounds-remaining — the expectation is baked into the target distribution, not
     computed at inference.
  2. **At search time**, the unchanged `n_determinizations` loop in `_score_plan`
     already averages each candidate's *leaf* over independent draw-seeded clones,
     so the one-ply (or two-ply) consequence of *our forced move's* own draw is
     still integrated by the search exactly as it is today.

  This is deliberately the cheap path the options doc names ("own-draw handled by
  the existing `n_determinizations` averaging") — we do **not** build a separate
  expectation head.

### 2.3 State / features for V

- **Reuse `encode_observation` as-is. No codec change.** The frozen v2 codec
  already carries everything the spine names: whole-track ego-centric corner
  block (`_track_block`, all corners by forward distance + speed-limit +
  length + lanes), `heat_available`, one-hot gear, hand histogram, deck
  composition, and the track-global `dist_to_finish` / `laps_remaining`. That is
  exactly `V(position, heat, speed/gear, hand-summary)`. Reusing the codec keeps
  V **codec-compatible** (`CODEC_VERSION == 2`), so a learned-leaf checkpoint
  drops into the same `save_checkpoint` / sidecar-tripwire machinery `MLAgent`
  uses, and a future C can warm-start from it.
- **`decision=None` at leaf encoding.** V scores a *state* (a rolled-out leaf),
  not a pending decision, so we pass `decision=None` to `encode_observation`
  (the zero-filled phase block) — the same convention the env uses at reset.
- **Reuse the model's separate critic/value head.** `PPOConfig.share_features_extractor=False`
  already gives the critic its own trunk ("how's the race going"). The value-net
  trainer builds a `MaskablePPO` via `build_model` and trains **only the critic
  head + its feature extractor** on the MC regression target, leaving the actor
  untouched — mirroring how `train_bc.py` trains only the actor and leaves the
  critic at init. This keeps one architecture/codec across A → C.

### 2.4 Integration with the corner-skill scoring (do not regress S1)

The S1 leaf score is
`value − own_spin_penalty·own_spins − spin_penalty·later_spins`, where `value`
is the `progress` (or `move_eval`) term and the two penalties make the
**controllable own-spin dominate** (own_spin_penalty=1000) with a depth-invariant
**pre-spin progress floor**. The learned leaf must not weaken this.

- **V replaces ONLY the `value` term, never the whole leaf.** Add a third
  `leaf_value` branch, `"learned"`, inside `_leaf_score`:
  - if `own_spins > 0`: keep the **existing `_pre_spin_progress` floor unchanged**
    (do **not** call V) — a line whose forced move spun is still scored by the
    depth-invariant pre-spin floor, so the spin penalty still bites at every
    horizon. V is only consulted on **clean** leaves.
  - else (`own_spins == 0`): `value = V_leaf(clone, player_id)` — the learned
    rounds-remaining estimate replaces `race_progress`.
  - the `− own_spin_penalty·own_spins − spin_penalty·later_spins` terms are
    **untouched**.
- **Keeping own-spins dominant.** Because V is only evaluated on clean leaves and
  the spin path retains the floor + the 1000-space own-spin penalty, a spinning
  candidate is still terminal-dominated relative to any clean one. The one new
  requirement is a **scale guard**: `V_leaf` (in negated-rounds units, roughly
  `[-26, 0]`) must never out-rank the own-spin penalty. Since the penalty is 1000
  and the spin branch never even calls V, this holds by construction; a unit test
  asserts a spinning candidate always scores below a clean one regardless of V's
  output (mirroring the S1 ablation tests).
- **No change to the rollout policy, REACT/slipstream delegation, determinization,
  top-k, sim-budget, or determinism machinery.** `"learned"` is purely a new
  `_leaf_score` branch; every S1/S2 invariant (the per-(candidate,det) reseed, the
  spin-count-once accounting, the prior-ordered budget cut) is inherited unchanged.

### 2.5 Data generation

- **Generator policy = `HeuristicAgent`** (the S1-validated rollout policy — it
  takes spins/limit-1-pass to ~0, and S1 learning #3 proved `StrongHeuristicAgent`
  is the *wrong* policy on this distribution). Each MC trajectory is a full solo
  race driven by `HeuristicAgent` on a generated tight/limit-1-weighted track
  (`_TIGHT_PARAMS`), labeling every learner state with its realized
  rounds-remaining.
- **Track bands kept disjoint, reusing the repo's existing convention:**
  TRAIN `100_000+`, VAL `500_000+`, GATE (held-out) `900_000+` — the exact bands
  `gen_demos.py` and `eval_search.py` already use, so V is never trained or
  validated on a gate track.
- **Why the heuristic and not the search agent first:** the heuristic is ~100×
  cheaper per move than a rollout-search expert, so we can generate a large,
  diverse value dataset cheaply for the *first* fit. The search agent only enters
  in the optional iteration step (§2.6), where its better trajectories refine V.

### 2.6 Iteration (light policy iteration)

- **Sprint A1 fits V once on heuristic trajectories and ships the leaf.** That
  alone tests the central hypothesis ("is the leaf the bottleneck?").
- **Sprint A3 (optional, gated) runs one or two rounds of light policy
  iteration:** generate fresh trajectories with the *current* `leaf_value="learned"`
  `LookaheadAgent` (better-than-heuristic driving), re-fit V on the union of old +
  new data, re-gate. **Stop when the held-out gate (worst-case limit-1 spins AND
  rounds-to-finish) stops improving** between iterations (a one-step
  no-improvement rule), or after 2 iterations — whichever first. This is the
  options doc's "can't exceed the policy that generated it unless iterated" caveat,
  bounded so it never becomes an open-ended self-play loop (the 8C failure mode).
  A3 is **escalate-only**: if A2's gate already matches/beats the `progress` leaf,
  A3 is skipped.

## 3. Ordered sprint list

```
A1  Value-net trainer + MC dataset            (build V, fit it, sanity-check calibration)
  └─ A2  leaf_value="learned" in the search   (plug V in, gate vs progress leaf on the shared eval)
        └─ A3  Light policy iteration (optional, escalate-only)
```

| Sprint | One-line goal | Effort |
|---|---|---|
| [A1](sprint-A1-value-net-and-data.md) | MC value dataset (`gen_value_data.py`) + value-net trainer (`train_value.py`); a calibrated `V` checkpoint. | ~1 sprint |
| [A2](sprint-A2-learned-leaf-in-search.md) | `leaf_value="learned"` path in `LookaheadAgent` (V loaded once, clean-leaf only); gate vs the `progress` leaf. | ~0.5–1 sprint |
| [A3](sprint-A3-policy-iteration.md) | Optional 1–2 rounds of refit-on-search-trajectories; stop-on-no-gate-improvement. | ~0.5 sprint, escalate-only |

Total: **2 sprints firm (A1, A2) + 1 conditional (A3)** — within the options
doc's "1–2 sprints" estimate, with A3 as the only addition and it is gated off by
default. Justification for not collapsing A1+A2: A1 produces a *standalone,
testable artifact* (a calibrated V checkpoint with its own regression metrics)
that B and C also consume, whereas A2 is a search-integration + eval sprint with a
different success bar (the behavioral gate). Splitting them keeps each sprint's
"done" crisp and lets A1's V be reused even if A2's integration is deferred.

## 4. Success ladder (tie to the shared eval)

All measured on **held-out generated tracks (900_000+ band)**, solo, via the
`experiments/eval_search.py` harness (extended, not replaced), reporting
**rounds-to-finish** + **worst-case (p90/max) spins/pass bucketed by corner
speed-limit** + **ms/move + clones/move** (the `SearchProfile` print). Baselines:
`HeuristicAgent` (the bar), the S1/S2 `LookaheadAgent` with `leaf_value="progress"`
(the agent to match/beat).

1. **Rung 0 — V calibrates (A1 gate, regression-level).** On held-out states, V's
   predicted rounds-remaining tracks realized rounds-remaining (val MAE reported;
   monotone-by-distance-to-finish sanity holds). This is a *necessary* check, not
   the behavioral win.
2. **Rung 1 — learned leaf is not worse (A2 floor).** `leaf_value="learned"` at
   horizon 2 ≥ `leaf_value="progress"` at horizon 2 on **all three** S1 criteria
   (worst-case limit-1 spins ≤, finish 100%, rounds ≤). No regression of the
   corner skill — the spin floor + own-spin penalty are intact.
3. **Rung 2 — the leaf buys depth back (the headline win).** `leaf_value="learned"`
   at **horizon 1** matches or beats `leaf_value="progress"` at its *best* horizon
   (2/3/4) on rounds-to-finish and worst-case limit-1 spins, at **lower ms/move**
   (no heuristic rollout to the horizon; one net forward pass per clean leaf). This
   is "shallow search + good value beats deep search + bad value" — the whole point.
4. **Rung 3 — heat budgeting improved (spine-level).** The **heat-efficiency**
   metric (heat spent vs distance + cool-downs taken, added to the harness) shows
   the learned-leaf agent budgets heat better over the lap than the `progress`
   leaf — confirming V encodes the long-horizon resource decision, not just corner
   survival. (Stretch within A2; the metric is a shared-eval requirement.)
5. **Rung 4 — iteration helps (A3, conditional).** If A3 runs, each iteration's
   held-out rounds-to-finish strictly improves until the stop rule fires; the final
   V beats the A2 V on the gate.

A2 **passes** at Rung 2 (Rung 1 is the floor, Rung 3 is the spine confirmation).
A3 is skipped if A2 already clears Rung 2 comfortably.

## 5. Repo assets reused (no reinvention)

- **Search hook:** `LookaheadAgent`'s `leaf_value` knob (`"progress"`/`"move_eval"`
  today) + `_leaf_score` / `_pre_spin_progress` (the corner-skill scoring to
  preserve).
- **Net + codec:** `build_model` / `HeatMLPExtractor` + the separate critic head
  (`share_features_extractor=False`); `encode_observation`; `spaces.CODEC_VERSION`.
- **Checkpoint I/O:** `save_checkpoint` / `load_meta` / the `MLAgent` sidecar
  tripwire (the learned-leaf checkpoint carries the same `obs_dim`/`action_dim`/
  `codec_version` sidecar so a drifted codec fails fast).
- **Simulator + data pattern:** `GameState.clone()` + `run_round_driver`; the
  `gen_demos.py` track-disjoint band + `_TIGHT_PARAMS` + driver-loop pattern
  (A1's data generator mirrors it, labeling rounds-remaining instead of actions).
- **Eval:** `experiments/eval_search.py` (spins-by-limit + p90/max + finish +
  rounds + `SearchProfile`), extended with a `leaf_value="learned"` contender and
  the heat-efficiency metric.
- **Run reporting:** wrap new entry points in `experiments/_runlog.py`'s
  `run_main` (START/DONE/FAILED markers), like every other experiment.

## 6. Explicitly deferred / out of scope

- **Opponents / hidden-info determinization for V.** V is trained and gated solo
  (own-draw uncertainty only). The `determinize_hidden` path is untouched and V
  composes with it later (it scores a state regardless of how the clone was
  determinized), but we do **not** train a 4p-aware value here.
- **TD / n-step bootstrapping.** MC only (§2.1). TD is a fallback noted in Open
  Questions, not built.
- **A learned policy prior / replacing the search.** That is C. A keeps the search
  and only upgrades the leaf.
- **Codec changes / new track features.** None — `encode_observation` as-is.
- **A separate own-draw expectation head.** Covered by MC targets +
  `n_determinizations` (§2.2).

## 7. Open questions

- **MC variance at the limit-1 corners.** If realized rounds-remaining is
  high-variance for states near a tight corner (a single spin adds many rounds),
  V may under-fit exactly where it matters. Mitigation in A1: report per-bucket
  (by corner-limit) regression error; if limit-1 MAE is poor, either (a) increase
  rollouts-per-state, or (b) fall back to a small TD(λ) bootstrap **only** as a
  deferred follow-up.
- **Leaf encoding of a mid-rollout state.** `encode_observation(decision=None)`
  zero-fills the phase block; confirm the critic trained on `decision=None`-encoded
  states (A1 logs states the same way) so train/inference encodings match exactly.
  Asserted by a round-trip test in A1.
- **Whether clean-leaf-only V is enough.** We deliberately keep the spin floor for
  spinning leaves (§2.4). If a future need arises to value *post-spin recovery*
  states (e.g. for C), V would need recovery-state coverage in its data — out of
  scope for A, flagged here.
