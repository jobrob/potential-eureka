# Sprint C7 — self-play throughput (make a real-scale run feasible)

> **The feasibility sprint.** The C6 postmortem
> ([`C6-findings-scale-and-feasibility.md`](C6-findings-scale-and-feasibility.md))
> established that the flat loop is **not** a method bug — the implementation is a
> faithful Gumbel-AlphaZero — but that we ran at ~0.015% of the games any AlphaZero
> result has needed, because generation is too slow to reach scale on one box. The
> profiled bottleneck is **not** the engine or cloning (~10% of per-move time) but
> the **neural-net evaluation path** (~90%) plus a **per-game model-reload tax**.
> This sprint removes those costs so a real-scale run becomes feasible. It is a
> **pure performance sprint**: it does **not** touch the search algorithm, the
> targets, or the loop's learning logic, and it preserves the determinism contract
> byte-for-byte.
>
> **This repurposes the C7 slot.** The README registered C7 as the hidden-info
> determinization follow-on "only if C6 validates" — C6 did not validate, so that
> bet is moot; throughput is the binding priority instead.

## Measured starting point (the profile this sprint attacks)

Steady-state, two-player 1v1 search at 16 sims, single process, CPU inference
(`cProfile`, warm). Per focal search-move ≈ **17 ms**; the loop spent ≈ **13 s/game**
(vs ≈1 s/game of actual compute — the gap is reload tax).

| Cost | share of per-move compute | this sprint |
|---|---|---|
| Feature encoding (pure-Python obs build) | ~36% | **cache the state-invariant part** |
| Torch `linear` (net fwd, **batch-1, CPU**) | ~34% | parallelize across cores |
| `logsumexp`/Categorical over **516** logits | ~21% | out of scope (model-arch change) |
| `run_round_driver` (engine) | ~6% | leave |
| Clone (GameState/Player/deck) | ~4% | leave |
| **Per-game model reload from disk (~1.3 s)** | **the loop's real killer** | **cache/reuse the model** |

## Goal

Raise self-play **generation throughput by ≥10×** (target ~20–50× with parallelism)
on the RTX 4080 box, with **bit-identical datasets** vs the current serial path, so a
real-scale AlphaZero run on Heat moves from "years" toward "feasible." Success is a
**measured** speedup (a benchmark harness is a first-class deliverable), not an
assumed one.

## Scope

### In scope (the fixes, in priority order)

1. **Model reuse across games — kill the reload tax (the biggest loop-level win,
   lowest risk).** Today `gen_selfplay.py:831` builds a **fresh `MCTSAgent` per
   game**, and each reloads the SB3 model from its `.zip` on first use (~1.3 s,
   measured). Add a **process-level model cache** keyed by `(abspath, mtime)` so the
   loaded SB3 policy is loaded **once per process** and shared read-only across all
   games; per-game mutable state (the agent's `_search_rng`/seed, `_ply`,
   `SearchProfile`) is reset per game, never the weights. The frozen-snapshot
   opponent (`_make_opponent`) uses the same cache. **Determinism is unchanged** —
   the per-game seed is still set, and the weights are read-only.

2. **Parallel self-play workers (near-linear core scaling).** Add a `--workers N`
   option to `gen_selfplay.generate_dataset` that distributes the per-track games
   across a process pool. The `NetAdapter` already pickles **by path** (model nulled
   on pickle), so cross-process workers are already supported; on Windows (spawn)
   each worker loads the model once via the cache from (1). Aggregation must be
   **deterministic**: collect each game's rows keyed by `(split, track_seed,
   mover_seat)` and concatenate in a **fixed sorted order** so the assembled dataset
   is **byte-identical regardless of `--workers`**. Seed bands stay disjoint
   (`_assert_seed_bands_disjoint` unchanged). Default `--workers 1` (serial) keeps
   current behavior exactly.

3. **Cache the state-invariant portion of `encode_observation` (~36% of per-move).**
   `features.py:_track_block` is rebuilt on **every** leaf eval (968×/game profiled)
   though the track is **constant within a game**. Investigate which sub-blocks of
   `encode_observation(state, pid, decision)` depend only on the (track) — or only on
   slow-changing state — and memoize them per game, recomputing only the dynamic
   per-decision features. **Hard gate: the assembled observation vector must remain
   bit-identical** to the un-cached path (pin with a test that the cached encoder
   equals the reference encoder on a battery of states); if a clean invariant split
   is not provable, ship a smaller safe win (e.g. precompute the per-track corner/
   distance arrays once) rather than risk a wrong obs.

4. **Benchmark harness — a first-class deliverable.** A new
   `experiments/bench_selfplay.py` that reports **games/sec, sims/sec, ms/move**, the
   per-phase breakdown (encode / net-fwd / engine / clone), and the model-load
   amortization, for a fixed config and seed. Run it **before and after** each change
   and record the numbers in the sprint's findings. This is how the ≥10× claim is
   substantiated.

### Out of scope (deferred, with reasons)

- **Batched-GPU inference server** (cross-worker leaf-eval batching) — the largest
  *theoretical* per-sim lever, but a major architectural change that fights MCTS's
  sequential nature and the determinism contract. Deferred to a follow-on (**C8**),
  taken only if cores + caching prove insufficient. **Note:** batch-1 GPU inference
  is *not* a win (host↔device transfer dominates a 104-dim/516-logit forward), so
  this sprint keeps **inference on CPU** and parallelizes across host cores; the GPU
  stays dedicated to the trainer.
- **Narrower / factored action head** (the 516-wide softmax, ~21%) — a model-
  architecture change with codec implications; separate concern.
- **Any change to the search, the policy/value targets, the Gumbel root, the chance
  nodes, or the loop's learning logic.** This sprint is throughput-only.

## Detailed changes

- **`src/heat/agents/mcts_agent.py` (`NetAdapter`)**: a module-level
  `_MODEL_CACHE: dict[tuple[str, float], MaskablePPO]` keyed by `(abspath, mtime)`;
  `_get_model` consults it before loading. Inference is read-only, so the cached
  module is shared safely; the per-(node,sample) reseed and the pickle-by-path
  contract are untouched. (If a future need arises to invalidate, the mtime key
  handles a re-trained checkpoint at the same path.)
- **`experiments/gen_selfplay.py`**: (a) construct/seed the agent's per-game state
  without reloading the model (reuse a cached `NetAdapter`); (b) `--workers N` over a
  `multiprocessing` pool with a deterministic, seed-sorted merge of the per-game
  buffers; (c) the `_make_opponent` snapshot uses the cache too. The
  `total_clones/moves/search_s` profile accounting is preserved (summed across
  workers).
- **`src/heat/ml/features.py`**: per-game memoization of the state-invariant encode
  sub-blocks behind a bit-identical-obs guard (see scope 3).
- **`experiments/bench_selfplay.py`** (new): the throughput benchmark.
- **`experiments/az_loop_1v1.py`**: thread `--workers` through to the generation call
  (default 1 = unchanged).

## Determinism — the hard contract (do not break)

The whole search is a **pure function of `(root state, search seed)`**, and a
generation run is a pure function of `(config, seeds, model)`. This sprint must keep
both:

- Same `(state, seed)` ⇒ **byte-identical move** (existing determinism tests stay
  green, unchanged).
- Same generation config ⇒ **byte-identical dataset**, *independent of `--workers`*
  (new test: `generate_dataset(workers=4)` produces the same arrays as
  `workers=1` after the deterministic seed-sorted merge).
- The cached encoder ⇒ **bit-identical observation vector** vs the reference encoder
  (new test over a battery of decision states).
- Model caching is read-only: it must not change any logged target.

## Success criteria

- **Primary (measured):** ≥10× self-play generation throughput on the Tier-0 config
  (`bench_selfplay.py` before/after), with `--workers` scaling toward the host core
  count, and a **bit-identical dataset** vs the serial path.
- **Mechanism (the minimum honest result):** model loaded once per process (reload
  tax gone, shown in the benchmark's load-amortization line); deterministic parallel
  aggregation pinned by test; the encoder cache bit-identical or omitted.
- **Regression:** all C0–C6 tests green; the loop produces the **same** 1–2-gen
  Tier-0 learning curve (read (b)/ECE) as the serial path, just faster (a short
  confirmation run).
- **Honest reporting:** the realized speedup and its decomposition recorded in a
  short findings note; if a lever underdelivers (e.g. encoder caching unsafe), say so
  and ship the rest.

## Risks & mitigations

- **Determinism regression from parallelism or caching** (the first-class risk).
  Mitigated by the three determinism tests above and the bit-identical-dataset gate;
  any lever that cannot be made bit-identical is dropped, not shipped approximate.
- **Windows multiprocessing (spawn) overhead / model reload per worker.** Each worker
  loads the model once (amortized over its share of games); choose a chunked work
  split so the per-worker load is amortized. The benchmark must report the amortized,
  not cold, throughput.
- **Encoder cache correctness** (a wrong obs silently poisons training). Mitigated by
  the bit-identical guard test and the "ship the smaller safe win if no clean
  invariant" instruction.
- **Shared read-only torch module across a process's games.** Safe for inference (no
  in-place state); do not enable grad, keep `torch.no_grad()`/eval mode as today.

## Dependencies

- C0–C6 built and green (they are).
- `NetAdapter` pickle-by-path contract (exists), `_assert_seed_bands_disjoint`
  (exists), the existing determinism tests (exist).
- RTX 4080 + torch cu126 box; this sprint deliberately keeps **inference on CPU
  across cores** and leaves the GPU for training.

## Effort

- Model cache: small. Parallel workers + deterministic merge: moderate (the merge
  determinism is the careful part). Encoder caching: small–moderate (gated on the
  invariant proof). Benchmark harness: small. Total: comparable to a C4/C5 code
  surface, with **the measured speedup as the deliverable**.
