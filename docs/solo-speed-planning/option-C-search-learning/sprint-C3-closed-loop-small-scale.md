# Sprint C3 — Close the loop at small scale (self-play → train → repeat, with the collapse guard)

> The final foundation sprint. Turn C2's single generation into the iterated
> AlphaZero loop — generate-with-search → train → generate-with-the-better-net →
> … — at SMALL scale, with the Wilson-LB / best-checkpoint guard that prevents a
> healthy-looking collapse from being promoted (the 8C / S3 history). The bar is to
> **BEAT** `LookaheadAgent` (rung 4), or to stop honestly at a reported plateau.

## Goal

A small-scale closed self-play loop that runs N generations unattended, gates each
generation on the shared eval, keeps the best checkpoint by a Wilson lower bound (not
a point estimate), and produces a net that — used as prior+value in `MCTSAgent`, and
ideally net-only at inference — **surpasses `LookaheadAgent`** on the held-out solo
field. If it plateaus at parity, report the gap and the honest stop.

## Scope

**In.**
- **The loop driver** (`experiments/az_loop.py`): for `gen` in `1..N`:
  1. generate self-play targets with the current-best net in `MCTSAgent` (C2's
     `gen_selfplay`, exploration on), on a per-generation self-play seed slice
     (disjoint from eval);
  2. train a new net on the **aggregated** target buffer (C2's `train_az`; aggregate
     across generations DAgger-style, with a cap / recency window so old low-quality
     targets age out);
  3. evaluate the new net (in-search **and** net-only) on the held-out solo field;
  4. **promote** only if it clears the Wilson-LB / best-checkpoint guard, else keep the
     incumbent — never promote a regression (the 8C bug-2 guard).
  `--smoke` shrinks generations / tracks / sims to prove the pipeline end-to-end in
  minutes (the S4 posture: pipeline real, headline numbers may need a real campaign).
- **The collapse guard**: reuse the A3 bounded-policy-iteration guard
  (`experiments/value_iterate.py`: the strict-improvement stop rule
  `GateMetric.strictly_improves_on` + best-checkpoint preservation in
  `run_iteration`, which never ships a checkpoint that gates worse than the
  incumbent) adapted to the solo metric — gate on worst-case L1 spins/pass +
  rounds with a lower bound (bootstrap CI over the held-out tracks), so a small
  lucky sample can't fake a pass. (Note: `dagger.py` / `eval_dagger.py` do NOT
  implement this guard — the reusable pattern lives in `value_iterate.py`.)
- **Net-only inference check**: every promoted net is also evaluated as a plain
  `MLAgent` (no search) on the held-out solo field, to measure how much skill
  distilled into the net itself (the "fast bot" payoff) vs. how much still needs
  search.
- A short **stop rule**: if K consecutive generations fail to improve the Wilson-LB,
  stop and report the plateau + the gap to `LookaheadAgent` (an honest C falls back to
  the A/B spine; it is not papered over).

**Out.**
- The `large` net + the full multi-hour/days GPU campaign (deferred — C3 proves the
  *mechanism* and monotone improvement at small scale; scale-up is a follow-on,
  exactly as S4 split smoke-validation from the full campaign).
- Opponents / multiplayer (deferred `Node` seam), MuZero (deferred `TransitionModel`
  seam), Gumbel sampling / leaf batching (deferred `RootActionSelector` seam — pulled
  forward only if throughput / low-visit quality binds, per the escalation below).

## Deliverables

- `experiments/az_loop.py` — the iterated loop with `--smoke`, the aggregation buffer
  (capped/recency-windowed), the per-generation eval, and the Wilson-LB
  best-checkpoint promotion guard. Wrapped in `_runlog.run_main`.
- `experiments/eval_az.py` (or an `eval_search.py` extension) — the C3 gate: rung-4
  verdict (C net in-search **and** net-only vs. `LookaheadAgent` and `HeuristicAgent`)
  on the held-out solo field, with the worst-case-spins/rounds Wilson-LB and the
  heat-efficiency report (heat-spent-vs-distance, cool-downs).
- The promoted checkpoint(s) + a generation-by-generation report (the improvement
  curve: worst-case L1 spins/pass, rounds, value-head MSE, net-only vs. in-search).
- Unit tests in `tests/test_az_loop.py`: the loop runs ≥2 generations end-to-end at
  `--smoke` and produces a contract-checked checkpoint; a regressed candidate is
  **not** promoted (inject a deliberately worse net and assert the incumbent is kept);
  the aggregation buffer respects its cap / recency window; the self-play and eval seed
  bands are asserted disjoint.

## Success criteria (rung 4 — BEAT `LookaheadAgent`)

On the held-out generated solo field, the best promoted net (in `MCTSAgent`):

- strictly **lower worst-case (max) spins/limit-1-pass** than `LookaheadAgent`, AND
- strictly **lower rounds-to-finish** than `LookaheadAgent`, AND
- 100% solo finish, with the **heat-efficiency** report confirming better *budgeting*
  (lower heat-per-distance / more deliberate cool-downs), not just corner survival,
- gated by a Wilson-LB / bootstrap CI, not a point estimate, on disjoint held-out
  seeds.

**Honest-stop criterion (explicitly allowed):** if the loop plateaus at rung 2/3
(matches but does not beat), C3 is "done" by reporting the plateau, the improvement
curve, the gap to `LookaheadAgent`, and a recommendation (escalate a seam below, or
fall back to the A/B/E spine). A non-beating result that is *measured and reported* is
a success of the sprint's process even though it is not rung 4 — the failure mode to
avoid is a fake pass (USA-only, finish-rate-only, point-estimate), per the parent
README's gate-hard rule.

## Risks & mitigations

- **Risk: self-play collapse that looks healthy (the 8C / Sprint-5 history — training
  curves fine, behavior worse-than-random).** Mitigation: the gate is the *behavioral*
  shared eval (worst-case spins / rounds on held-out generated tracks), not the
  training loss; the Wilson-LB best-checkpoint guard means a collapsed generation is
  never promoted; net-only and in-search are both reported so a divergence is visible.
  The single most important discipline in C3 and the reason it is its own sprint.
- **Risk: the aggregated buffer drowns the good late targets in stale early ones (or,
  conversely, forgets corner discipline).** Mitigation: cap + recency window (tuned at
  `--smoke`); the per-generation eval catches forgetting immediately and the
  best-checkpoint guard rolls back.
- **Risk: the loop is too expensive to iterate enough to beat the baseline, or the
  visit budget is too low to produce better-than-net targets.** Mitigation: C0's
  budget + small scale; the **escalation levers are the deferred seams** — CUDA leaf
  batching, **Gumbel-AlphaZero** (low-visit improvement guarantee, the
  `RootActionSelector` seam), `large`-net-only-at-scale — but C3 first proves monotone
  improvement at small scale before paying for them.
- **Risk: improvement stalls because the value head is the bottleneck.** Mitigation:
  C2 reports value MSE; if V is the limit, the deferred **Option-A/B-V warm-start**
  (the `LeafEvaluator` seam) is the escalation, not built in the foundation.

## Dependencies

- **C2** (target generator + joint policy/value trainer) and **C1** (`MCTSAgent` + the
  seams).
- The A3 strict-improvement / best-checkpoint guard pattern
  (`experiments/value_iterate.py`: `GateMetric.strictly_improves_on` +
  `run_iteration`'s best-checkpoint preservation).
- The held-out eval band + heat-efficiency metric (parent README shared eval); the GPU
  (RTX 4080, torch cu126); `_runlog.run_main`.

## Effort

~1–1.5 sprints. The loop plumbing is mostly C1/C2 + S4-guard reuse; the real cost is
running enough small-scale generations to get a *trustworthy* rung-4 verdict and the
discipline of gating it honestly.
