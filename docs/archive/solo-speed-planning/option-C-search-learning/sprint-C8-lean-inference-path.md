# Sprint C8 — lean inference path (cut the per-leaf neural + plumbing overhead)

> **The second throughput sprint, and a deliberate redirect.** C7 cashed in
> everything the CPU cores can give (~8× via parallel self-play workers,
> [`sprint-C7-selfplay-throughput.md`](sprint-C7-selfplay-throughput.md)). The
> obvious next idea is **batched-GPU leaf evaluation** — but the C7 profile says
> that is the *wrong* next bet right now, and a cheaper, determinism-safe win is
> sitting in plain sight. This sprint takes the cheap win first and **defers the
> batched-GPU inference server to a conditional follow-on (C9)**.

## Why batched-GPU leaf eval is NOT this sprint (the Amdahl argument)

The steady-state per-move profile (C7 `bench_selfplay`, two-player, 16 sims) is:

| phase | share | nature |
|---|---|---|
| `net-fwd` (torch `linear`) | ~16% | the only part GPU-batching speeds up |
| `encode` (obs build) | ~15% | Python; per-leaf |
| **`other`** | **~54%** | **SB3 `get_distribution` Distribution objects, 516-wide masked `logsumexp`, per-call `as_tensor`/`reshape`, double forward, `nn.Module` call overhead** |
| `engine` | ~10% | leave |
| `clone` | ~4% | leave |

Batched-GPU inference attacks only the ~16% `net-fwd` slice. Even driving it to
**zero** caps the speedup at **~1.2×** — for a *complex* change: a cross-worker
inference server (Windows uses spawn, so IPC/queue latency is real) or within-move
**parallel MCTS with virtual loss** (which *changes the search dynamics* and risks
the determinism/behaviour contract). **Not worth it while >50% of the time is
non-`net-fwd` overhead.**

That `other 54%` is the prize, and most of it is **reducible, determinism-safe
plumbing** in how we call the net — not irreducible compute.

## The win this sprint takes (measured root cause)

Every leaf expansion (`NetAdapter.policy_prior` + `NetAdapter.leaf_value`,
`mcts_agent.py:715`/`:734`) currently does, **per leaf**:

1. **Two separate forward passes** — `model.policy.get_distribution(ob, masks)` for
   the prior **and** `model.policy.predict_values(ob)` for the value. With the
   checkpoint's `share_features_extractor=false`, each runs its **own feature
   extractor** → the network is effectively evaluated twice per leaf (~970
   forwards/game for ~485 leaves).
2. **Two per-call tensor creations** (`torch.as_tensor(...).reshape(1, -1)` twice).
3. **The full SB3 `Distribution` machinery** — `get_distribution` builds a
   `MaskableCategorical` (a 516-wide masked `logsumexp` + softmax + a `Distribution`
   object) only for the search to read back `dist.distribution.probs` and use just
   the kept-candidate entries.

A **lean inference path** computes the *same mathematical quantities* with far less
overhead, and **composes** with C7's 8× (it is per-eval, orthogonal to parallelism).

## Goal

A **measured additional ≥1.5×** self-play generation throughput on top of C7 (read by
`bench_selfplay.py` before/after on the same config), achieved by removing redundant
per-leaf neural/plumbing work — with **behaviour equivalent** to the current search
(same probs/value to tight tolerance; the search stays a pure, deterministic function
of `(state, seed)`). Cumulative target with C7: **~12–20×** over the original serial
baseline.

## Scope

### In scope (the cheap, determinism-safe wins — fold them all in)

1. **One combined forward per leaf → both policy logits and value.** Add a single
   `NetAdapter` method (e.g. `evaluate(obs, mask) -> (probs_or_logits, value)`) that
   runs the network **once** and returns both heads, replacing the separate
   `policy_prior` + `leaf_value` double pass at expansion. (Keep the two public
   methods as thin wrappers for any caller that needs only one, but route
   `_expand_and_evaluate` through the combined call.) This is the single biggest
   piece — it roughly halves the forward work and the tensor churn.
2. **Direct masked log-softmax — bypass the SB3 `Distribution` objects.** Compute the
   masked policy probabilities with raw torch ops (mask → `-inf` logits →
   `log_softmax`/`softmax`) instead of constructing a `MaskableCategorical` /
   `Distribution`. Must equal the SB3 result to tight tolerance (the masking + softmax
   are the same operation). Return what the search consumes.
3. **Kill per-call tensor churn.** Use `torch.inference_mode()` (not just
   `no_grad`), a reused/preallocated input buffer, and a single `as_tensor` per leaf
   (one obs, not two).
4. **Per-worker `torch.set_num_threads(1)`** in the pool worker init (a tiny, safe
   anti-oversubscription nudge; C7 showed >8 workers contend on the 8 physical cores).

### Folded-in cheap wins (only if cleanly safe)

- Reuse the encoded obs within a single leaf expansion (one encode feeds both heads —
  this falls out of (1) naturally).
- Micro-opt the player-dependent loop left in `features.py:_track_block` (the C7
  precompute handled only the invariant part), **gated on bit-identical obs**.

### Out of scope — deferred to **C9** (conditional)

- **Batched-GPU leaf evaluation / inference server / parallel-MCTS-with-virtual-loss.**
  Re-measure with `bench_selfplay` *after* C8: take C9 **only if `net-fwd` is then the
  dominant residual phase** and a feasibility spike shows the cross-process batching
  win survives Windows spawn/IPC latency. Virtual-loss within-move batching is
  explicitly algorithm-touching and would require a learning-equivalence re-validation.
- Narrower/factored action head (the 516-wide head) — a codec/model-architecture
  change, separate concern.
- Moving inference to GPU at batch-1 (established *not* a win — transfer dominates).

## Detailed changes

- **`src/heat/agents/mcts_agent.py` (`NetAdapter`)**: add the combined
  `evaluate(obs, mask)` (one forward, both heads, lean masked softmax, `inference_mode`,
  single tensor build); apply the `value_mode` tanh in the same call. Re-point
  `_expand_and_evaluate` to it. Keep `policy_prior`/`leaf_value` as compatibility
  shims (some call sites / tests use them directly).
- **`experiments/gen_selfplay.py`**: pool-worker initializer calling
  `torch.set_num_threads(1)` (and the cached model load).
- **`src/heat/ml/features.py`**: optional `_track_block` player-loop micro-opt behind
  the existing bit-identical guard.
- **`experiments/bench_selfplay.py`**: no change needed; it already reports the
  per-phase breakdown — use it to confirm `net-fwd`+`other` shrink and to read the
  realized speedup.

## The determinism / equivalence contract (the first-class risk)

The lean path computes the **same math** but, replacing SB3's `logsumexp`/softmax with
our own ops, may differ from the historical output by floating-point ULPs. Handle this
explicitly:

- **Preserve the real contract:** same `(state, seed)` ⇒ **identical** move, every run
  (the determinism property is about reproducibility, not a specific golden value).
  The existing "same seed twice → byte-identical" tests must stay green.
- **Re-baseline golden-value assertions** if any test pins a *specific* historical
  prob/value/visit vector that shifts by ULPs — and document it as an intended,
  behaviour-equivalent change of the deterministic function.
- **Add an equivalence test:** the lean `evaluate` probs/value equal the SB3
  `get_distribution`/`predict_values` outputs within a tight `atol` (e.g. 1e-5) over a
  battery of states/masks.
- **Behaviour re-validation:** a short 1–2-generation Tier-0 slice (via `az_loop_1v1`)
  produces an **equivalent** read-(b)/ECE curve to the pre-C8 path — proof the search
  still behaves the same, just faster.

## Success criteria

- **Primary (measured):** ≥1.5× additional generation throughput vs C7 on the Tier-0
  config (`bench_selfplay` before/after), with `net-fwd`+`other` visibly shrunk in the
  phase breakdown. Cumulative ~12–20× over the original serial baseline.
- **Equivalence:** lean `evaluate` ≈ SB3 path within tolerance (test-pinned); a 1–2-gen
  Tier-0 slice gives an equivalent learning curve.
- **Regression:** all C0–C7 tests green (golden-value re-baselines documented if any).
- **Honest reporting:** realized speedup + updated phase breakdown recorded; an explicit
  read on whether `net-fwd` is now the residual wall (the C9 trigger).

## Risks & mitigations

- **ULP drift flips an argmax / breaks a golden test.** Expected and acceptable;
  mitigated by re-baselining golden values, keeping the reproducibility property, and
  the equivalence + behaviour-re-validation tests. If drift is *behaviourally*
  material (it should not be), fall back to mirroring SB3's exact op order.
- **A compatibility shim diverges from the combined path.** Mitigate by implementing
  `policy_prior`/`leaf_value` as wrappers over `evaluate` (single source of truth).
- **Diminishing returns.** If the lean path underdelivers (<1.5×), report it honestly;
  the profile then says the residual is genuinely `net-fwd` → C9 (batched GPU) becomes
  justified, which is itself the useful result.

## Dependencies

- C0–C7 built and green; `bench_selfplay.py` (C7) is the measurement instrument.
- RTX 4080 box; inference stays on CPU across cores (GPU remains the trainer's).

## Effort

- Combined forward + lean masked softmax: small–moderate (the equivalence test is the
  careful part). Thread-pin + micro-opts: small. Benchmark + re-validation run:
  ~½ day compute/analysis. Total: smaller code surface than C7, with the **measured
  speedup and the C9 go/no-go read** as the deliverables.
