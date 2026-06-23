# Sprint C11 — lean network forward (kill the batch-1 dispatch tax + the wasted critic)

> **The `net-fwd` sprint — and, like C8, a deliberate redirect away from batched-GPU.**
> After C9, `net-fwd` is the dominant residual phase (~17% of wall). The reflexive
> answer is the batched-GPU inference server the C8 doc pencilled as the conditional
> "C9-GPU". Profiling says that is *still* the wrong first bet: more than half of
> `net-fwd` is **not matmul** — it is batch-1 `nn.Module` dispatch overhead and a
> **wasted critic forward on the prior path** — and both are removable single-process,
> determinism-safe, and composing with C7's parallelism. This sprint takes those CPU
> wins first and keeps **batched-GPU deferred as a conditional follow-on**.

## Profiling motivation (measured, post-C9)

Serial `bench_selfplay.py`, Tier-0 (`c5_main_best.zip`, two-player, 16 sims, 4 games),
cProfile `tottime`, ~44s wall, ~45s summed. `net-fwd` ≈ 7.5s (~17%) decomposes as:

| component | tottime | nature |
|---|---|---|
| `torch._C._nn.linear` (1,056,348 calls) | **8.0s** | the actual matmul — the **only** part batched-GPU speeds up |
| `_call_impl` + `_wrapped_call_impl` + `_get_tracing_state` (2,518,884 each) | **~4.2s** | **pure `nn.Module` dispatch overhead** — batch-1 Python tax, not compute |
| `torch.relu` / `torch.tanh` (609k / 298k) | ~1.8s | activations |
| `evaluate` / `leaf_value` / obs marshal | ~1.5s | the lean C8 wrappers + `as_tensor` |

### The architecture (why the prior path wastes a forward)

The checkpoint (`HeatMLPExtractor`, `share_features_extractor=False`, `net_arch=[256,256]`):

- **pi feature extractor** = `Linear(104→256)→ReLU→Linear(256→256)→ReLU→Linear(256→128)→ReLU` (3 linears)
- **vf feature extractor** = same shape, **separate weights** (3 linears) — *not shared*
- policy MLP `Linear(128→256)→Tanh→Linear(256→256)→Tanh`; value MLP same
- `action_net` `Linear(256→516)` (the wide head); `value_net` head `Linear(256→1)`

So a full `evaluate` (both heads) = **12 linears**; a critic-only `leaf_value` = **6 linears**.

**The waste:** C8 made `policy_prior` a thin shim over the combined `evaluate`
(`mcts_agent.py:809`), so the **prior path computes the full critic head — the entire
`vf` extractor (3 linears) + value MLP (2) + value head (1) — and discards it**. The C8
note assumed "the extra critic head is cheap next to the feature extraction"; that holds
for a *shared* extractor, but here the extractor is **not shared**, so the discarded
critic re-runs a whole 3-linear `vf` trunk. That is **6 wasted linears per leaf
expansion — ~27% of all 1.06M linears**, plus their dispatch + activations.

## The wins this sprint takes (CPU-side, low-risk, in priority order)

### Win 1 — actor-only prior path (pure, byte-identical, biggest single win)
Make `policy_prior` compute **only the actor head** (`pi` extractor → policy MLP →
`action_net` → masked softmax) and **not** the critic. Keep `evaluate` (the combined
one-obs-both-heads method) for `leaf_value`'s value read and the C8 equivalence battery,
but route `_edges_for`'s prior read through an actor-only path. Removes the discarded
`vf` extractor + value MLP + value head — **~6 linears/leaf, ~27% of all linears**.
Probs are byte-identical (same actor ops, same order).

### Win 2 — kill the `nn.Module` dispatch tax (functional forward)
The ~4.2s in `_call_impl`/`_wrapped_call_impl`/`_get_tracing_state` is the cost of calling
~a dozen small `nn.Module`s (each `Linear`, each activation) per forward, ~100k forwards.
Bypass it: **extract the weight/bias tensors once at model load** and run the actor and
critic pipelines as a plain function using `torch.nn.functional.linear` + `torch.relu` /
`torch.tanh`, inside `inference_mode`. `F.linear` is *exactly* what `nn.Linear.forward`
calls, so this is the **same op in the same order — bit-equivalent**, no Module dispatch.
(Cache the extracted tensors on the `NetAdapter`, rebuilt on model (re)load; null them in
`__getstate__` alongside `_model`/`_obs_buf`.) `torch.jit.script` of the two pure pipelines
is an acceptable alternative if cleaner, but hand-rolled `F.linear` is the simplest exactly
-equivalent path and avoids JIT warmup/caching/determinism questions.

### Folded-in micro-opts (only if cleanly safe)
- One `as_tensor`/obs marshal already exists (C8 reused buffer); reuse the same input
  tensor for the actor-only prior and the (separate-obs) value read where the obs is
  identical — but note the prior obs and value obs **differ** (decision context vs
  `None`), so they remain two marshals; do not force-merge.

## Out of scope — deferred to a conditional GPU follow-on
- **Batched-GPU leaf evaluation / cross-process inference server / parallel-MCTS with
  virtual loss.** The C8 Amdahl argument still binds and *tightens* after Wins 1–2: with
  dispatch + the wasted critic removed, the GPU-addressable matmul is an even smaller
  share, so driving it to zero caps the *additional* speedup well under the C8-quoted
  ~1.2×. It is also the high-complexity / high-risk path (Windows `spawn` IPC latency for
  an inference server; virtual loss *changes the search dynamics* and needs a
  learning-equivalence re-validation). **Re-measure after C11**: take the GPU sprint only
  if matmul is then the dominant residual AND a feasibility spike shows the batched win
  survives. The honest expectation is that it will not clear the bar — which is itself the
  useful result.
- Narrower/factored 516-wide `action_net` head — a model-architecture/codec change.
- Quantization / `torch.compile` of the whole policy — heavier dependency + its own
  numerical-equivalence proof obligation; Win 2's hand-rolled `F.linear` gets most of the
  dispatch win without it.

## The determinism / equivalence contract (first-class)

- **Win 1 is byte-identical** (the actor ops are unchanged; only the unused critic is
  dropped). The prior probs must equal the current `policy_prior` output exactly
  (`np.array_equal`).
- **Win 2 is bit-equivalent** (`F.linear` == `nn.Linear`'s op, same order). Extend the
  C8 `test_c8_lean_inference` battery: the functional `evaluate`/actor-only-prior/`leaf_value`
  equal the current Module-dispatch path within a tight `atol` (1e-6; expect exact). If any
  ULP drift appears, document it as an intended behaviour-equivalent change and keep the
  reproducibility property.
- **Reproducibility unchanged:** same `(state, seed)` ⇒ identical move; the C8 determinism
  tests stay green.
- **Behaviour re-validation:** a 1–2-generation Tier-0 slice via `az_loop_1v1` gives an
  equivalent read-(b)/ECE curve.

## Success criteria
- **Primary (measured):** `net-fwd` phase shrinks from ~17% toward ~9–11% on the same
  `bench_selfplay` config — `linear` call count down ~25%+ (Win 1) and the
  `_call_impl`/`_get_tracing_state` dispatch rows largely **gone** (Win 2). Realized
  end-to-end speedup reported honestly — **expected ~1.08–1.12×** (the residual is
  irreducible-on-CPU matmul; this is the low-risk slice of `net-fwd`).
- **Equivalence:** `np.array_equal` (Win 1) + tight-`atol` battery (Win 2) green; all
  C0–C9 tests green with golden re-baselines documented only if a genuine ULP drift appears.
- **Honest reporting:** updated phase breakdown + an explicit read on whether matmul is now
  the dominant residual → the **go/no-go for the conditional batched-GPU sprint** (with the
  Amdahl caveat that its post-C11 ceiling is small).

## Detailed changes
- **`src/heat/agents/mcts_agent.py` (`NetAdapter`)**: add an actor-only prior method (Win 1)
  and route `_edges_for` through it; implement the functional forward (Win 2) — extract
  `(W,b)` for the 3-linear `pi`/`vf` extractors, the two 2-linear MLPs, and the two heads at
  load; run actor/critic pipelines with `F.linear`+`relu`/`tanh` under `inference_mode`;
  keep `evaluate`/`leaf_value`/`policy_prior` as the public surface (now backed by the
  functional path); cache tensors, null in `__getstate__`.
- **`src/heat/ml/model.py`**: read `HeatMLPExtractor.forward` (`:239`) to mirror its exact
  layer order in the functional path (no change to the module itself).
- **`tests/test_c11_lean_forward.py`** (new): `np.array_equal` for the actor-only prior vs
  current `policy_prior`; tight-`atol` battery for the functional `evaluate`/`leaf_value` vs
  the Module-dispatch path, across both modes/seats; a determinism check.
- **`experiments/bench_selfplay.py`**: no change; use the per-phase breakdown before/after.

## Dependencies
- C0–C9 built and green; `bench_selfplay.py` is the measurement instrument.
- Inference stays on CPU across cores (GPU remains the trainer's); orthogonal to C7.

## Effort
- Win 1 (actor-only prior): small, low risk (the equivalence assert is trivial).
- Win 2 (functional forward): small–moderate (faithfully mirroring the extractor + MLP +
  head order is the careful part; the math is unchanged). Benchmark + Tier-0 re-validation:
  ~½ day. Total: small code surface, with the measured speedup and the batched-GPU go/no-go
  read as deliverables.
