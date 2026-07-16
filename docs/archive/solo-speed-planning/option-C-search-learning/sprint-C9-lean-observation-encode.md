# Sprint C9 — lean observation encode (cut the per-leaf obs-build cost)

> **The next throughput sprint, chosen by measurement.** The C8 doc pencilled C9 as
> *batched-GPU leaf eval*; the C8 findings + the engine-clone profiling
> ([`sprint-C10-incremental-round-advance.md`](sprint-C10-incremental-round-advance.md))
> reprioritized. The profile says the biggest **lower-risk** non-`net-fwd` lever is the
> per-leaf **observation build** (`encode`, ~20% of wall), so it takes the C9 slot.
> Batched-GPU (`net-fwd`) is deferred as the conditional GPU sprint; the engine
> incremental-advance rewrite (C10) is deferred as higher-risk for a similar ceiling.

## Profiling motivation (measured, not assumed)

Serial `bench_selfplay.py`, Tier-0 (`c5_main_best.zip`, two-player, 16 sims, 4 games,
codec v3), cProfile `cumtime`, 52.4s wall. `encode_observation` totals **10.7s (~20%)**
over 104,935 calls, and it is **not** evenly spread:

| sub-block (`features.py`) | cumtime | share of encode | what it does per leaf |
|---|---|---|---|
| **`_deck_composition`** (`:86`) | **5.0s** | **~47%** | `list(draw)+list(discard)` alloc, then **4 separate genexpr passes** (`:94`–`:97`) over all ~18 deck cards, one per card type |
| `_track_block` (`:206`) | 1.6s | ~15% | per-call `sorted(corner_table)` + globals (the C7 precompute already handles the invariant maxes) |
| `_opponent_slots` (`:303`) | 0.63s | ~6% | per-opponent rel-pos / lap-delta |
| `_hand_histogram` (`:45`) | 0.61s | ~6% | dict with **f-string keys** (`f"S{value}"`) per card |
| `_adrenaline_context` (`:295`) | 0.59s | ~6% | rebuilds `active_players`, calls `adrenaline_eligible` |
| `_clip01` (`:27`) ×3.1M | 0.32s | ~3% | per-field Python clamp |
| assembly (`asarray`/`clip`/list `+=`) + rest | ~1.9s | ~17% | |

Two structural facts drive the design:

1. **`_deck_composition` walks the deck four times.** It allocates a combined list and
   then runs four independent `sum(1 for c in all_cards if c.card_type == …)` passes —
   ~4× the necessary iteration plus two list allocations, on **every** leaf eval. This
   single block is ~47% of encode and ~9–10% of total wall.
2. **Each clean leaf encodes the state twice** (the C8 finding). `_edges_for` builds the
   **prior** obs `encode_observation(state, mover, decision)` (`:1187`) and
   `_evaluate_leaf` builds the **value** obs `encode_observation(state, to_move_pid,
   None)` (`:1834`). When `mover == to_move_pid` (all **solo** leaves, and the
   learner's-own-seat leaves in two-player) these two vectors are **byte-identical on
   every block except the final phase tail** (decision-context bits 0..8). Everything
   before the phase block — hand, gear, kinematics, **deck**, track, adrenaline,
   opponents — is recomputed identically a second time.

## The wins this sprint takes

### Tier 1 — local, bit-identical rewrites (primary; no cross-method plumbing)

1. **`_deck_composition`: one pass, no alloc.** Replace the two `list(...)` builds and
   four genexpr passes with a **single loop** over the draw + discard piles
   (`itertools.chain` over `deck._draw_pile`/`_discard_pile`, or the existing tuple
   properties without concatenation) accumulating four counts. Same output, ~4× less
   iteration, zero intermediate lists. Expected: ~5.0s → ~1.3s.
2. **`_hand_histogram`: fixed-index counts.** Drop the string-keyed dict and the
   `f"S{value}"` f-strings; count into a small fixed-size list/array indexed by card
   type+value. Same 8 floats. Expected: ~0.61s → ~0.2s.
3. **`_adrenaline_context` / `_opponent_slots` micro-opts.** Avoid rebuilding
   `active_players` per call where a cheap count suffices; hoist `length`/`laps`
   reciprocals. Small but free.

Tier 1 is **purely local** — each function returns the identical vector — so it is
pinned by the existing obs golden tests with **no re-baseline**. Estimated ~3.5–4s.

### Tier 2 — shared state-prefix between the prior and value encodes (structural)

Compute the **seat-state prefix** (every block before the phase tail) **once per leaf**
and reuse it for both the prior obs and the value obs, recomputing only the small phase
block. This directly removes the second full encode the C8 finding identified.

- **Applies when `mover == to_move_pid`**: all solo leaves, and learner-seat leaves in
  two-player. **Does not apply** at an *opponent* decision node in two-player (`mover !=
  to_move_pid` → genuinely different seat → no shared prefix); that path keeps two full
  encodes. Gate the sharing on the seat-equality check so the opponent path is untouched.
- Mechanism: a small helper that returns `(prefix_values, phase_block_writer)` so
  `_expand_and_evaluate` can build both vectors from one prefix. Crosses the
  `_edges_for` / `_evaluate_leaf` boundary, so the plumbing (not the math) is the work.
- Output must be **byte-identical** to two independent `encode_observation` calls.

Estimated additional ~3–4s on the solo / learner-seat path (overlaps Tier 1's deck win,
so the combined saving is sub-additive, not the sum).

### Folded-in free win (from the C10 design — ship it here)

`EngineTransitionModel._advance` forces `clone.logging_enabled = True` on **every**
throwaway replay clone (`mcts_agent.py:461`) purely so leaf spin-accounting can read
`spin_out` events — making ~398k discarded event-log builds run during search. **Detect
spins directly** (read `player.spun_out`, or a lightweight counter) and leave
`logging_enabled` **off** on search clones. ~1–1.5% of wall, no engine rewrite, and it
belongs with the throughput work. (This is the "Tier 1 free win" carved out of C10.)

## Out of scope
- Batched-GPU `net-fwd` (~34%, the conditional GPU sprint — separate; re-measure after C9).
- Engine incremental-advance / round cursor (C10, ~19%, deferred — higher risk, similar ceiling).
- Vectorizing the whole encode assembly into one preallocated `np.ndarray` (drops the
  3.1M `_clip01` calls + per-block list churn) — a real future win but a larger rewrite
  with its own bit-identity proof obligation (division/clip order must match exactly).
  Note it; do not attempt here.
- Incremental deck-composition counts maintained on `Deck` mutations — a stretch that
  touches the deck mutation surface; out unless Tier 1 underdelivers.

## The determinism / equivalence contract (first-class)

`encode_observation` is a **pure function** and the codec is frozen (`spaces.py`); the
contract is **bit-identical output**, not "close."

- **Re-baseline NOTHING.** Tier 1 and Tier 2 must produce the **exact same float32
  vector** as today on every index. The existing obs golden / codec tests
  (`test_ml_features_*`, `test_c7_throughput.py`) are the regression oracle and must stay
  green unchanged.
- **Add an equivalence battery:** the new `_deck_composition` / `_hand_histogram` / the
  shared-prefix path equal the current `encode_observation` **exactly** (`np.array_equal`,
  not `allclose`) over a battery of real search states across both modes and both seats.
- **Reproducibility unchanged:** same `(state, seed)` ⇒ identical move; the C8
  determinism tests stay green.
- **Behaviour re-validation:** a 1–2-generation Tier-0 slice via `az_loop_1v1` gives an
  identical read-(b)/ECE curve (a pure speed change must not move the learning curve).

## Success criteria

- **Primary (measured):** `encode` phase drops from ~20% toward ~6–8% on the same
  `bench_selfplay` config (`_deck_composition` no longer dominant). Realized end-to-end
  speedup reported honestly — **expected ~1.12–1.15×** (the biggest lower-risk
  non-`net-fwd` lever; same ballpark as C8/C10 individually, but cheaper and safer to land).
- **Equivalence:** `np.array_equal` battery green; all obs/codec/C0–C8 tests green with
  **zero golden re-baseline**.
- **Honest reporting:** updated phase breakdown + the new residual wall (almost certainly
  `net-fwd` → the GPU sprint becomes the next go/no-go).

## Detailed changes
- **`src/heat/ml/features.py`**: rewrite `_deck_composition` (single pass, no alloc) and
  `_hand_histogram` (fixed-index); minor `_adrenaline_context`/`_opponent_slots` hoists;
  add the shared-prefix helper (Tier 2) returning the state blocks + a phase-tail writer.
- **`src/heat/agents/mcts_agent.py`**: route `_edges_for` + `_evaluate_leaf` through the
  shared prefix when `mover == to_move_pid` (Tier 2); drop the forced replay-logging in
  `_advance` and replace it with direct spin detection (the folded-in free win).
- **`tests/`**: a `test_c9_lean_encode` equivalence battery (`np.array_equal`, both modes,
  both seats) + a spin-detection-without-logging equivalence check.
- **`experiments/bench_selfplay.py`**: no change; it already reports the `encode` phase —
  use it before/after.

## Why this over C9-batched-GPU / C10-engine (the prioritization)
`encode` (~20%) is a **larger** slice than engine replay (~19%) and **far lower risk**
than either an engine rewrite (C10's byte-identical generator decomposition) or a
cross-process GPU inference server (the original C9, gated behind Windows spawn/IPC and
capped at ~1.2× by Amdahl per the C8 doc). Tier 1 alone is local, bit-identical, and
lands a meaningful chunk with almost no risk; Tier 2 adds the C8 double-encode fix. After
C9, re-measure: `net-fwd` will be the residual wall and the GPU sprint becomes the next
honest go/no-go.

## Dependencies
- C0–C8 built and green; `bench_selfplay.py` is the measurement instrument.
- No GPU needed (encode + search are CPU; orthogonal to the GPU sprint).

## Effort
- Tier 1 (deck/hand rewrites): small, low risk (the equivalence battery is the careful part).
- Tier 2 (shared prefix): small–moderate (cross-method plumbing + the seat-equality gate).
- Free logging win: small. Benchmark + Tier-0 re-validation: ~½ day. Total: smaller and
  safer than C8, with the measured speedup + the GPU go/no-go read as deliverables.
