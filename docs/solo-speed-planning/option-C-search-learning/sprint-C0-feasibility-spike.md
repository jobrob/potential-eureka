# Sprint C0 — Feasibility, cost & constants spike (design gate)

> **Type:** design-gate / spike (~0.5 sprint). May fold into the first day of C1
> if the answer is obviously "yes". Its job is to *decide the numbers* the complete
> algorithm (README §4) builds against and to kill the option early if per-move
> stochastic MCTS is unaffordable, before any real machinery is written.

## Goal

Answer one question with measured numbers, not intuition: **is a net-guided solo
stochastic-MCTS over the real engine affordable at our throughput, and what
constants make it both correct and cheap enough to self-play with?** Produce the
concrete budget + the full constant set C1 targets.

## Scope

**In.**
- **Profile the building blocks** at the budgets MCTS will actually use:
  `GameState.clone()` + one `run_round_driver` advance to the next decision (S2
  measured ~16 µs/clone, ~64k/s — re-measure *with* a net forward at the leaf, which
  dominates), and one `MaskableActorCriticPolicy` forward (policy logits + value) on
  CPU and on the RTX 4080 (torch cu126).
- **Profile the chance-node cost specifically.** Because chance is now *in-tree*
  (README §4.3), the realistic per-move work is `n_simulations` selections, each
  descending through some decision edges and some chance edges that re-clone+advance.
  Time the realistic product (sims × avg-path-length × clone+advance), **not** a
  single clone — and separately measure how DPW fan-out (`C_pw`, `α_pw`) changes the
  distinct-clone count, since that is what actually sets cost.
- **Fix the complete-algorithm constants** (the deliverable):
  - search budget: `n_simulations`, per-move clone/ms budget;
  - chance: DPW `C_pw`, `α_pw`, a hard outcome cap `K`, and the reseed scheme
    (confirm purity — a fixed `(root, seed)` gives a byte-stable move);
  - selection: `c_init`, `c_base` (the log-PUCT term), `fpu_reduction`, and the
    Q-normalization mode (running tree min-max — confirm the raw `−rounds_remaining`
    scale really does swamp `P` without it, so the requirement is *measured*, not
    asserted);
  - self-play exploration (used by C2/C3, but sized here): Dirichlet `α`, `ε`,
    and the temperature schedule `(τ=1 for T_moves, then τ→0)`;
  - leaf: net value-head vs. a short rollout-to-pre-spin-floor blend (`LeafEvaluator`
    choice for the core);
  - branch action set: confirm the S1 dedup-by-speed candidate prune
    (`LookaheadAgent._candidate_plans`) is the right *lossless* branch and that the
    pruned width does **not** yet force Gumbel/sampled-MCTS (that stays a deferred
    seam — but record the width so the Gumbel go/no-go is data-driven later).
- A back-of-envelope + one tiny end-to-end timing harness for states/sec → tracks/hour
  at the chosen constants, scaled to the self-play volume C2/C3 need.

**Out.** Any real MCTS implementation, any training, any net checkpoint production,
any Gumbel/opponent/MuZero work. This sprint writes a short measured memo + the
chosen constants, nothing that ships as an agent.

## Deliverables

- `experiments/spike_mcts_cost.py` — a throwaway-but-committed timing harness: times
  N engine clones+advances, N net forwards (CPU + CUDA), and the *realistic* in-tree
  product (sims × path × clone+advance) at a small grid of
  `(n_simulations, C_pw, α_pw, K, leaf_mode)`; reports implied ms/move, clones/move,
  and states/hour. Includes a tiny experiment that confirms (or refutes) the
  Q-normalization requirement by comparing PUCT child-selection entropy with raw vs.
  min-max-normalized Q on a handful of real states. Wrapped in `_runlog.run_main`.
- A short measured memo (`C0-findings.md` in this folder, or an "Outcome" section
  appended here mirroring the S1/S2 docs) stating the **chosen constant set** (every
  item in Scope) and the per-move clone/ms budget C1 must hit.
- A go / no-go recommendation: if the affordable simulation count is too low to beat
  `LookaheadAgent`'s effective lookahead, say so and recommend falling back to the
  spine's A/B options (or pulling the Gumbel seam forward).

## Success criteria

- The chosen constants give a **per-move cost within ~2–5× of `LookaheadAgent`'s
  measured ~4 ms/move solo** (S2 number) — i.e. self-play data generation is
  hours-not-days at the small scale C2/C3 target. (A higher ms/move is acceptable at
  inference if the *trained* net then runs net-only; the binding constraint is
  self-play data volume.)
- The Q-normalization requirement is settled with a number (not an assertion), and
  the DPW constants give a chance fan-out that is bounded and affordable.
- A clear, written go/no-go with the numbers behind it.

## Risks & mitigations

- **Risk: the net forward dominates and CPU self-play is too slow.** Mitigation:
  measure CUDA batched forwards; if needed, batch leaf evaluations across the tree's
  frontier (standard AlphaZero virtual-loss batching) — *note* it here, do not build
  it (C1 can start unbatched; C3 can batch if throughput binds).
- **Risk: the spike under-counts because real in-tree chance re-clones per sample.**
  Mitigation: time at the realistic `sims × path × (clone+advance)` product with DPW
  on, exactly the count §4.3 implies — not a single clone.
- **Risk: DPW fan-out explodes (chance node keeps widening).** Mitigation: grid
  `(C_pw, α_pw, K)` and pick the smallest fan-out whose backed-up value is stable
  across re-seeds (a cheap variance check in the harness).
- **Risk: analysis paralysis.** Mitigation: this is a half-sprint with a hard output
  (the constant set + a go/no-go). If trivially affordable, fold into C1 day 1.

## Dependencies

- The S1/S2 `LookaheadAgent` cost numbers (`SearchProfile`, the eval harness print)
  as the baseline to compare against.
- A GPU box (RTX 4080, torch cu126 — memory `reference-gpu-torch-setup`) for the
  CUDA forward timings.

## Effort

~0.5 sprint. Foldable into C1 if the spike answer is an easy yes.
