# Sprint C0 — Feasibility & cost spike (design gate)

> **Type:** design-gate / spike (~0.5 sprint). May fold into the first day of C1
> if the answer is obviously "yes". Its job is to *decide the numbers* C1 builds
> against and to kill the option early if per-move MCTS is unaffordable, before any
> real machinery is written.

## Goal

Answer one question with measured numbers, not intuition: **is a net-guided solo
MCTS-with-chance over the real engine affordable at our throughput, and what
(visit count, determinizations, leaf horizon) make it both correct and cheap
enough to self-play with?** Produce the concrete budget C1 targets.

## Scope

**In.**
- Profile the real cost of the building blocks at the budgets MCTS will actually
  use: `GameState.clone()` + one `run_round_driver` step (S2 measured ~16 µs/clone,
  ~64k/s, but re-measure *with* a net forward at the leaf, which dominates), and one
  `MaskableActorCriticPolicy` forward (policy logits + value) on CPU and on the
  RTX 4080 (torch cu126).
- A back-of-envelope (and one tiny end-to-end timing harness) for: clones/move and
  ms/move at candidate `(n_simulations, n_determinizations, leaf_horizon)` settings,
  scaled to the self-play volume C2/C3 need (states/sec → tracks/hour).
- Decide whether the leaf value is the **net's value head** (cheap, one forward) or a
  **short rollout to the S1 pre-spin leaf** (more faithful, costs clones) — or a
  blend — for the foundation.
- Decide the action set the tree branches over: the S1 dedup-by-speed candidates
  (`LookaheadAgent._candidate_plans`) vs. the full 494 CARDS prior. Confirm the wide
  branch does not require Gumbel/sampled-MCTS *yet* (that is a deferred extension).

**Out.** Any actual MCTS implementation, any training, any net checkpoint
production. This sprint writes a short measured memo + the chosen constants, nothing
that ships as an agent.

## Deliverables

- `experiments/spike_mcts_cost.py` — a throwaway-but-committed timing harness:
  times N engine clones+steps, N net forwards (CPU + CUDA), and reports the implied
  ms/move and states/hour at a small grid of `(n_simulations, n_determinizations,
  leaf_horizon)`. Wrapped in `_runlog.run_main`.
- A short measured memo (in this folder, `C0-findings.md`, or appended here as an
  "Outcome" section mirroring the S1/S2 docs) stating the **chosen foundation
  constants**: `n_simulations`, `n_determinizations` (start from S1's 2), leaf =
  net-value vs. short-rollout, branch action set, and the per-move clone/ms budget
  C1 must hit.
- A go / no-go recommendation: if the affordable visit count is too low to beat
  `LookaheadAgent`'s effective lookahead, say so and recommend falling back to the
  spine's A/B options.

## Success criteria

- The chosen constants give a **per-move cost within ~2–5× of `LookaheadAgent`'s
  measured ~4 ms/move solo** (S2 number) — i.e. self-play data generation is
  hours-not-days at the small scale C2/C3 target. (A higher ms/move is acceptable at
  inference if the *trained* net then runs net-only; the binding constraint is
  self-play data volume.)
- A clear, written go/no-go with the numbers behind it.

## Risks & mitigations

- **Risk: the net forward dominates and CPU self-play is too slow.** Mitigation:
  measure CUDA batched forwards; if needed, batch leaf evaluations across the tree's
  frontier (standard AlphaZero virtual-loss batching) — but only *note* it here, do
  not build it (C1 can start unbatched and C3 can batch if throughput binds).
- **Risk: the spike under-counts because a real tree re-clones per simulation.**
  Mitigation: time at the *realistic* `n_simulations × n_determinizations` product,
  not a single clone; the S1 reseed scheme means each simulation forks from the live
  state, so the count is `n_simulations × n_determinizations × leaf_horizon` engine
  steps per move — measure that product directly.
- **Risk: analysis paralysis.** Mitigation: this is a half-sprint with a hard output
  (the constants + a go/no-go). If the spike is trivially affordable, fold it into
  C1 day 1 and move on.

## Dependencies

- The S1/S2 `LookaheadAgent` cost numbers (`SearchProfile`, the eval harness print)
  as the baseline to compare against.
- A GPU box (RTX 4080, torch cu126 — memory `reference-gpu-torch-setup`) for the
  CUDA forward timings.

## Effort

~0.5 sprint. Foldable into C1 if the spike answer is an easy yes.
