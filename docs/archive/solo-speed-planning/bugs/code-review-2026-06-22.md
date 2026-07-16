# Pre-Option-C code review — findings (2026-06-22)

> **What this is.** A read-only code review of the whole codebase run before
> building Option C (solo AlphaZero). Five parallel `code-quality-reviewer` agents
> each owned a section (engine+models, agents, ML core/codec, tracks/eval-infra,
> experiment harnesses). **Nothing was changed** — this is a findings register.
> Severity is the reviewers' assessment of *risk*, not confirmed live failure; a
> few items are convention-by-design and need a confirming test, not a fix. Items
> the reviewers explicitly *confirmed sound* are listed at the bottom so we don't
> re-investigate them.
>
> **Scope reminder.** Option C reuses the engine as its tree transition model
> (millions of `clone(reseed=…)` + `run_round_driver` calls), the frozen codec for
> obs/policy/value, the held-out generated-track seed band as its gate, and the
> data-gen / promotion-guard patterns from the experiment harnesses. So
> correctness bugs in those exact surfaces propagate straight into C.

## Triage at a glance

| # | Severity | Area | File:line | Pre-C blocker? |
|---|---|---|---|---|
| 1 | 🔴 HIGH | Codec / value path | `src/heat/ml/features.py:266` | **YES** |
| 2 | 🔴 HIGH | Engine RNG | `src/heat/models/game_state.py:167` | verify |
| 3 | 🔴 HIGH | Engine state | `src/heat/engine/phases.py:290/294, 618` | verify |
| 4 | 🔴 HIGH | Engine deck | `src/heat/models/cards.py:154` / `phases.py:259` | verify |
| 5 | 🔴 HIGH | Codec / data gen | `src/heat/ml/action_codec.py` (`encode_action_index`) | **YES** |
| 6 | 🔴 HIGH | Track seeds | `src/heat/tracks/generator.py:283` | **YES** |
| 7 | 🔴 HIGH | Eval RNG | `src/heat/simulation/runner.py:355` | verify |
| 8 | 🔴 HIGH | Promotion guard | `experiments/dagger.py` (absent) → `experiments/value_iterate.py` | **YES (doc fix)** |
| 9 | 🔴 HIGH | Train/val split | `experiments/gen_*`, `train_*` | **YES** |
| 10 | 🔴 HIGH | Eval metric | `experiments/eval_search.py` (`max_single_turn=12`) | **YES** |
| 11 | 🟡 MED | Search reuse | `src/heat/agents/search_agent.py:423` | knowingly accept or fix |
| 12 | 🟡 MED | Value contract | `src/heat/agents/search_agent.py:361, 307` | recommended for C |
| 13 | 🟡 MED | Harness | several (see below) | recommended |
| 14 | 🟡 MED | Codec normalization | `src/heat/ml/features.py` (several) | recommended |
| 15 | 🟢 LOW | Misc | several | improve-as-you-go |

---

## 🔴 Cross-cutting HIGH

### 1. `round_num` leaks into the observation; `decision=None` zeroes the phase block — `features.py:266`
*(flagged independently by the ML-core and experiments reviewers)*

`round_num` is written into the observation **only when a real `decision` is
present**. Option C's value states call `encode_observation(state, pid,
decision=None)`, which zeroes the whole phase block — so the *same board position*
encodes differently in the policy/prior path vs. the value path, and the
observation is **not a pure function of game state**. Two compounding harms:

- the value net can partly **memorize the round counter** instead of learning a
  genuine cost-to-go, making the `target = −rounds_remaining` label leak its own
  answer (also hits `gen_value_data.py` and Option A's V);
- search-time value encodings won't match train-time encodings.

This is the single most important item to resolve before C, because **both** A's
value-net and C's critic depend on a state-pure observation. Decide the convention
(round_num in or out, but *consistent* across `decision`/`decision=None`) and bump
`CODEC_VERSION` if it changes.

---

## 🔴 Engine correctness (the transition model C clones millions of times)

### 2. `clone(reseed=None)` advances the *parent* RNG — `game_state.py:167`
Cloning without an explicit reseed mutates the original's RNG stream, so cloning N
candidates from the live state couples their seeds to loop order and to how many
draws the live game already made. `LookaheadAgent` already always passes an
explicit `reseed=`; **C must too.** Consider a louder default (or a debug assert)
so a future caller can't silently regress determinism.

### 3. lap-vs-position desync — `phases.py:290/294` (blocking pullback across finish), `phases.py:618` (wrap-around spin-out)
When a car is pulled back across the finish line by blocking, or spins out on a
wrap-around, `lap` and `position` can disagree. A deep tree clones and propagates
these inconsistent states, and `race_progress` / the value target read both — so a
desync here biases rollouts and labels. Validate with targeted tests on a short
(2-lap) track before wiring the engine into MCTS.

### 4. possible card loss / count non-conservation when the deck exhausts mid stress/boost — `cards.py:154` / `phases.py:259`
If the draw pile empties during a multi-card stress/boost draw, cards may be lost
(total deck count not conserved). Rare in normal play, but C's long rollouts
exercise deck exhaustion / reshuffle far more often, and a lost card silently
changes the reachable state space. Add a deck-count-conservation invariant test.

---

## 🔴 Codec / contract

### 5. `encode_action_index` raises on off-table REACT (and out-of-range CARDS/DISCARD) — `action_codec.py`
The engine accepts any valid `ReactDecision`, but the codec has only an 8-slot
REACT table; off-table combos (and out-of-range CARDS/DISCARD) raise. **Rollout /
search is safe** because the legal-action *mask* bounds the choice — but **any data
generation that encodes a chosen action** (DAgger, BC, and C2's `gen_selfplay.py`)
will crash or silently drop samples unless it reuses `gen_demos.py`'s
**drop-and-count** discipline. This confirms why the C2 design mandates inheriting
drop-and-count verbatim; it is load-bearing, not a nicety.

---

## 🔴 Eval / seed integrity (the held-out gate C lives or dies by)

### 6. additive seed-mixing in `TrackSampler` — `generator.py:283`
There is **no structural disjointness** between a self-play seed band and the
held-out eval band (base `900_000`). With additive mixing, a long campaign's
episode seeds can reach into the eval band and **silently regenerate the exact eval
tracks** → data leakage that fakes a rung-4 pass. Fix the band scheme (namespaced /
non-overlapping by construction) before standing up C's gate.

### 7. global `random.seed()` in the sequential runner — `runner.py:355`
The default `parallel=False` eval path seeds global `random`; the
"execution-order-independent" guarantee only holds if the engine/agents never touch
global `random` (related to #2). Verify against current engine RNG usage before
trusting eval reproducibility.

### 8. `dagger.py` has **no** best-checkpoint / Wilson-LB promotion guard — despite being cited as the exemplar
The Option-C design (README §6, C3) points the collapse guard at
`eval_dagger.py` / `dagger.py`, but the reviewer found the actual guard pattern
lives in **`value_iterate.py`** (the A3 orchestrator: strict-improvement stop,
best-checkpoint preservation). **C3 should mirror `value_iterate.py`.** This is a
documentation correction to make in `sprint-C3-closed-loop-small-scale.md` and the
README reuse map regardless of whether anything else changes.

### 9. train/val track-disjointness asserted in docstrings but never verified in code — `gen_*` / `train_*`
The split that prevents overfitting-leakage is *claimed* in comments but never
checked at runtime. C2/C3 should **assert** band/track disjointness programmatically
(self-play vs eval, and train vs val), not trust the comment.

### 10. `max_single_turn=12` cap in `eval_search`'s pass reconstruction can desync pass/spin counts — `eval_search.py`
The headline gate metric (spins bucketed by corner speed-limit) is reconstructed
with a hardcoded `max_single_turn=12` cap that can miscount on large single-turn
moves. Because this *is* the rung-2 / rung-4 verdict, a desync here silently
distorts the bar C is measured against.

---

## 🟡 MEDIUM

### 11. `_candidate_plans` dedup by `(gear, total speed)` is lossy at horizon ≥ 2 — `search_agent.py:423`
Same-speed plays with different card *composition* (stress count, *which* physical
cards leave the hand) are treated as interchangeable. Lossless for the immediate
landing position (horizon 0/1); **diverges for deeper rollouts and for any distilled
training target**. C branches deeper than 1 ply and trains on these states, so
either tighten the dedup key (include stress-count / the value multiset) or accept
the bias as a known approximation explicitly.

### 12. `_value_leaf` sign / target-convention is unguarded — `search_agent.py:361`, `:307`
Correct today (net regresses `−rounds_remaining`, returned as-is), but there is no
`value_sign` / target-convention field in the `.meta.json` tripwire
(`_validate_value_meta`). A retrained V with a flipped target would **silently
invert leaf preferences**. Add a convention marker to the sidecar for C (where V is
retrained repeatedly).

### 13. Harness mediums *(experiments reviewer)*
- omitted **finish-crossing pass** in the pass reconstruction;
- train/val share *game* seeds (only track seeds differ);
- seed-band constants **duplicated across files** with no central disjointness
  assertion (compounds #6/#9);
- `_runlog` treats `SystemExit(0)` as `DONE` (a script that `sys.exit(0)`s early
  before doing its work reports success).

### 14. Codec normalization *(codec reviewer)*
- per-track **max scaling** rather than a fixed scale (a learned net sees
  track-relative magnitudes that shift between tracks);
- `MAX_CORNERS` silent **truncation** of long tracks;
- finished-opponent **double-encoding** in the opponent block.

---

## 🟢 LOW

- `_move_eval` known residual bugs now also feed **candidate ranking / budget-cut
  ordering** via `search_agent._candidate_prior` (`:471`) and the `"move_eval"` leaf
  (`:813`) — wider blast radius than before (ranking-only, lower stakes).
- `choose_cards` stale-recompute path is convoluted/fragile — `search_agent.py:955`.
- `_count_spins` / `_pre_spin_progress` each linearly **rescan the full event_log**
  per leaf — fine at horizon ≤ 2, a hot path at AlphaZero depth (`:685`, `:773`).
- possible **lap off-by-one** in the final-lap boost gate — `heuristic_agent.py:244`
  (consistent with `strong_heuristic.py:137`; confirm with one 2-lap test).
- `IndexError` (rather than a clear error) on an empty legal list —
  `random_agent.py:29`, `heuristic_agent.py:99`.

---

## ✅ Confirmed sound (do not re-investigate)

- **Search determinism**: `_turn_seed` hand-folds signature ints (survives
  `PYTHONHASHSEED`); `_score_plan` reseeds each clone from `(turn_seed, plan_index,
  det)` with `plan_index` = original enumeration index, so top-k reordering never
  perturbs the reseed stream. Scores are a pure function of `(state, seed)`.
- **No double-counting in spin accounting**: `clone` defaults
  `copy_event_log=False`, so the rollout clone's `event_log` starts empty; the
  `round_num == first_round_num` own-vs-later split is exact.
- **`opponent_action`** routes all five `DecisionKind`s correctly.
- **Eval statistics**: Wilson lower bound, ELO/TrueSkill order-independence, seeded
  bootstrap, and `validate_track` are all sound.
- **Contract tripwire**: the `MLAgent` / `LookaheadAgent` pickle-by-path
  `__getstate__` + `CheckpointMismatchError` on `obs_dim`/`action_dim`/`codec_version`
  is solid (the gap is only the *value-sign* convention, #12).
- **`planning/*.py`** is abandoned Option-B spike code (NO-GO) — its bugs don't
  matter unless revived.

---

## Recommended sequencing before / during Option C

1. **Resolve before C1 builds:** #1 (obs purity), #6 (seed-band disjointness),
   #5 (drop-and-count in data gen — already mandated by C2), #8 (point C3 at
   `value_iterate.py`; doc fix now).
2. **Verify with targeted tests before wiring the engine into MCTS:** #2, #3, #4,
   #7 — a deep tree exercises clone-RNG, lap/position edges, deck exhaustion, and
   eval RNG far harder than the current 2-ply lookahead.
3. **Decide knowingly during C1:** #11 (dedup key) and #12 (value-sign guard).
4. **Improve as you go:** #10, #13, #14, and the LOWs.
