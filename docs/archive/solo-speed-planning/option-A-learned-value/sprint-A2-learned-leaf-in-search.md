# Sprint A2 — `leaf_value="learned"` in the search

> Plug A1's trained `V` into the `LookaheadAgent` as a third leaf evaluator,
> **on clean leaves only**, leaving the S1 own-spin penalty + pre-spin progress
> floor untouched. Gate it against the `leaf_value="progress"` agent on the shared
> eval. This sprint delivers the option's headline win: **shallow search + a good
> leaf beats deep search + a bad one.**

## Goal

Add a `leaf_value="learned"` path to `LookaheadAgent` that scores a **clean**
rolled-out leaf with A1's `V` (negated rounds-remaining) in place of
`race_progress`, and demonstrate on held-out generated tracks that the learned
leaf at **horizon 1** matches/beats the `progress` leaf at its best horizon, at
lower ms/move — without regressing the corner-skill scoring.

## Scope

**In:**
- A `leaf_value="learned"` branch in `LookaheadAgent._leaf_score` that calls V on
  clean leaves (`own_spins == 0`) and keeps the existing `_pre_spin_progress` floor
  on spinning leaves.
- One-time, lazy V load on the agent (the model + sidecar-tripwire pattern from
  `MLAgent`), pickling by path for the eval harness's per-game agent factory.
- Eval-harness extension: a `leaf_value="learned"` contender + the
  **heat-efficiency** metric (heat spent vs distance, cool-downs taken).
- Determinism + legality + no-regression unit tests.

**Out:**
- Training V (A1) and refitting it (A3).
- 4p / `determinize_hidden` value training (the branch composes but is not
  trained/gated here).
- Any change to the rollout policy, top-k, sim-budget, or determinization.

## Deliverables (concrete files / functions)

- **`src/heat/agents/search_agent.py`** — extend, do not rewrite:
  - `__init__`: accept `leaf_value="learned"` in the validation set; add a
    `value_model_path: str | None = None` arg (required when
    `leaf_value="learned"`). Store it; **lazy-load** the value model on first use
    (a `_value_model` slot nulled in `__getstate__`, exactly like `MLAgent`), and
    validate its `.meta.json` against the live contract
    (`obs_dim`/`action_dim`/`codec_version`) so a stale V fails fast.
  - new `_value_leaf(self, clone, player_id) -> float`: encode the leaf with
    `encode_observation(clone, player_id, decision=None)`, run the critic
    forward (`policy.predict_values`), and return the **negated** scalar
    (`-rounds_remaining` → higher is better, sign-consistent with `progress`).
  - `_leaf_score`: add the third branch. The control flow becomes, for the
    `"learned"` case:
    - `if own_spins > 0: value = self._pre_spin_progress(clone, player_id)`
      — **unchanged**: spinning leaves never call V, so the depth-invariant floor
      and the dominant own-spin penalty are fully intact.
    - `else: value = self._value_leaf(clone, player_id)`.
    - the `− own_spin_penalty·own_spins − spin_penalty·later_spins` subtraction is
      **byte-for-byte unchanged**.
  - `"progress"` / `"move_eval"` paths and every S1/S2 invariant
    (per-(candidate,det) reseed, count-spins-once, prior-ordered budget cut,
    plan caching) are untouched.
- **Picklable factory** updates in `simulation/runner.py` and `ml/evaluate.py`
  (the existing `lookahead_agent_factory`s): add `leaf_value` + `value_model_path`
  as plain primitives so the learned-leaf agent ships to `ProcessPoolExecutor`
  workers (each reloads V by path), mirroring how S2's `determinize_hidden`/`top_k`
  were plumbed.
- **`experiments/eval_search.py`** — extend (don't replace):
  - a `--value-model PATH` CLI arg; when set, register a `LookaheadLearned`
    contender (`leaf_value="learned"`, `value_model_path=PATH`) alongside the
    existing `Lookahead` (progress) / `Heuristic` rows.
  - the **heat-efficiency** metric (shared-eval requirement): per game, accumulate
    heat spent vs lap-aware distance advanced and count cool-downs (`react`
    cooldown events) from the event log; report mean per agent in the solo field.
  - print the existing `SearchProfile` (ms/move, clones/move) for the learned
    contender so the "cheaper at shallower depth" claim is measured.
- **`tests/test_search_agent_learned.py`** — unit tests:
  - **no-regression / spin dominance:** with a *stub* V (a fixed callable returning
    an arbitrary large value), a candidate whose forced move spins still scores
    strictly below a clean candidate — proves V can never out-rank the own-spin
    penalty (the §2.4 scale guard).
  - **clean-leaf-only:** V is invoked only when `own_spins == 0` (assert via a
    counting stub) — spinning leaves go through `_pre_spin_progress`.
  - **determinism:** fixed state + seed + V → byte-stable plan selection (the V
    forward pass is deterministic; the S1 reseed machinery is unchanged).
  - **legality:** the learned-leaf agent never returns an illegal move (the search
    still only selects among legal candidate plans).
  - **horizon-0 equivalence preserved** for the non-learned paths (the S1 test
    still passes).
  - **contract tripwire:** loading the agent with a V whose sidecar codec_version
    mismatches raises the `MLAgent`-style mismatch error.

## Success criteria (measurable, on the shared eval)

Measured solo on the held-out **900_000+** generated band via the extended
`eval_search.py`, vs `HeuristicAgent` (bar) and `LookaheadAgent`
`leaf_value="progress"` (the agent to match/beat). Maps to the README success
ladder:

- **Rung 1 (floor — must pass):** `leaf_value="learned"` at horizon 2 ≥
  `leaf_value="progress"` at horizon 2 on **all three** S1 criteria — worst-case
  limit-1 spins/pass ≤, solo finish 100%, rounds-to-finish ≤. No corner-skill
  regression.
- **Rung 2 (headline — the sprint passes here):** `leaf_value="learned"` at
  **horizon 1** matches or beats `leaf_value="progress"` at its *best* horizon
  (sweep 2/3/4) on rounds-to-finish and worst-case limit-1 spins, at **lower
  ms/move** (the learned leaf removes the horizon-deep heuristic rollout). This is
  the "good value lets us search shallow" result the whole option exists to prove.
- **Rung 3 (spine confirmation — stretch within A2):** the heat-efficiency metric
  shows the learned-leaf agent spends heat more efficiently over the lap than the
  `progress` leaf (better distance-per-heat and/or better-timed cool-downs),
  evidence V encodes the long-horizon heat budget, not just corner survival.

## Risks & mitigations

- **V mis-calibrated at limit-1 → bad shallow-horizon choices.** Because spinning
  leaves never call V (the floor handles them), V only steers *clean* lines, where
  A1's calibration is best. If Rung 2 fails at horizon 1, fall back to horizon 2
  with the learned leaf (still cheaper than progress-at-best-depth's rollout) —
  Rung 1 is the floor.
- **Scale mismatch (V out-ranking the spin penalty).** Guarded by construction (V
  not called on spins) + the explicit stub-V unit test. The own-spin penalty
  (1000) dwarfs V's `[-26, 0]` range.
- **Per-move net forward-pass cost.** One critic forward per clean leaf. Cheaper
  than the heuristic rollout it replaces at the same depth, and the depth drop
  (h2/3/4 → h1) is the net saving — measured by `SearchProfile`, not assumed. CPU
  inference (`MLAgent` loads `device="cpu"`) is fine at this scale.
- **Pickling V to workers.** Solved by the `MLAgent` path-only `__getstate__`
  pattern; a test asserts the factory pickles.

## Dependencies

- **A1** (the trained `V` checkpoint + its contract sidecar) — hard dependency.
- The `LookaheadAgent` `leaf_value` hook, `_leaf_score`, `_pre_spin_progress`,
  `SearchProfile`; `MLAgent`'s lazy-load + sidecar-tripwire + path-only pickle
  pattern; `eval_search.py`; the `lookahead_agent_factory`s — all built.

## Rough effort

~0.5–1 sprint. The integration is a single new `_leaf_score` branch + a lazy V
loader cloned from `MLAgent`; the bulk is the eval sweep (horizons × learned/progress)
and the heat-efficiency metric. Lower-risk than A1 because the corner-skill scoring
is preserved by construction (clean-leaf-only).
