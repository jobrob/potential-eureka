# Sprint C2 — Self-play target generation + policy/value trainer (one generation)

> The second foundation sprint. Make the search *teach the net*: turn on the
> self-play exploration machinery built in C1 (Dirichlet root noise + the
> temperature schedule), generate `(obs, π_visits, z_MC)` targets from C1's MCTS,
> train the policy+value net toward them once, and prove the trained net — used as
> the prior+value in the same search — **BEATS the net it learned from**. This is
> the single experiment that validates the AlphaZero improvement signal before we
> pay for the full loop in C3.

## Goal

A self-play data generator + a supervised policy/value trainer such that **one**
generation of (generate-with-C1-search → train) yields a net that, plugged back into
`MCTSAgent`, is measurably better on the rung-2 metrics than the net used to generate
the data (rung 3). The targets follow README §4.6 exactly.

## Scope

**In.**
- **Target generation** (`experiments/gen_selfplay.py`): run `MCTSAgent` (C1) over
  many GENERATED solo tracks (the `eval_search` tight params, on a self-play seed band
  **disjoint** from the held-out eval band), driving `run_round_driver` exactly as
  `gen_demos.py` / `HeatEnv` do, with the **self-play exploration ON** (README §4.5):
  - **root Dirichlet noise** `P(root,a)=(1−ε)p_a+ε·Dir(α)` (C0's `ε`,`α`),
  - **temperature schedule** `a∝N^{1/τ}`, `τ=1` for the first `T_moves` plies then
    `τ→0` — so the data carries exploration signal, not a collapsed argmax.
  At every real-choice learner decision (for **all** `DecisionKind`s — GEAR, CARDS,
  REACT, SLIPSTREAM, DISCARD, since C1 searches them all) log:
  - `encode_observation(state, pid, decision)` (the frozen obs),
  - the **MCTS visit distribution** `π(a)=N(a)^{1/τ_t}/ΣN` over the legal/kept flat
    actions — the policy target (never the argmax: that is BC again, with BC's gap),
  - the `action_mask` (`legal_action_mask`),
  - enough episode bookkeeping to compute the **MC value target** `z` post-hoc
    (the realized rounds-to-finish from that state to episode end).
  After each episode finishes, backfill `z` for every logged state as
  `z = −rounds_remaining` **with the pre-spin floor** (`_pre_spin_progress`) applied
  so a spin does not poison the target. MC return is used (not the bootstrapped root
  value) so `z` is unbiased and **definitionally identical to Option A/B's V** — an
  A/B warm-start stays a drop-in (README §4.6).
  Output a compressed `.npz` + a `.selfplay.json` provenance sidecar (codec version,
  full search config incl. DPW/PUCT/Dirichlet/temperature, seed band) — mirror
  `gen_demos.py`'s format and its **drop-and-count** of any off-table REACT action the
  codec can't encode (S3 lesson #2).
- **Trainer** (`experiments/train_az.py`): a standalone supervised trainer (NOT PPO)
  that builds a real `MaskablePPO` via `build_model` (weights live in the exact
  architecture `MLAgent` expects), then minimizes the AlphaZero loss (README §4.6):
  `L = CE(π, masked policy logits) + c_v · MSE(z, value head) + wd·‖θ‖²`. Train **both**
  the actor trunk/head and the **critic** (unlike S3 BC, which left the critic
  uninitialized — here V is a first-class target). Save via `save_checkpoint` (zip +
  `.meta.json` contract sidecar). Hold out tracks for a track-disjoint val split
  (like `gen_demos`). `c_v` tuned on the val split.
- **Re-evaluation**: plug the trained checkpoint back into `MCTSAgent` (prior + leaf
  value) and run the rung-2 eval vs. the **pre-training** net used to generate the
  data, on the held-out solo field — the rung-3 verdict.

**Out.**
- The *repeated* loop (multiple generations) — that is C3. C2 does exactly **one**
  generate→train→evaluate cycle to isolate the improvement signal.
- Opponents, MuZero, Gumbel sampling, the `large` net / multi-hour campaign (deferred
  seams), bootstrapped/TD value targets (the `value_target` seam — MC only now).
- Any PPO / RL reward fine-tune (the AZ target is the search policy/value, not a
  reward — keep the paths separate; PPO fine-tune is the S4 lineage, not this one).

## Deliverables

- `experiments/gen_selfplay.py` — the target generator (reuses C1's `MCTSAgent` with
  exploration on, `gen_demos.py`'s logging/codec machinery, the disjoint seed band,
  the post-hoc MC-return backfill with the pre-spin floor). Wrapped in `run_main`.
- `experiments/train_az.py` — the policy+value supervised trainer (CE policy + MSE
  value + weight decay), saving a contract-checked checkpoint. Wrapped in `run_main`.
- A trained checkpoint + the rung-3 re-eval report (trained-net-in-search vs.
  pre-training-net-in-search, held-out solo, with heat-efficiency).
- Unit tests in `tests/test_az_targets.py` / `tests/test_train_az.py`:
  - logged tuples are codec-valid and mask-consistent (`π` support ⊆ mask, sums to 1);
  - **the value target is the pre-spin cost-to-go** on a known forced-spin track (the
    realized MC return floored, *not* the post-spin recovery) — the single most
    important correctness check in C2;
  - `π` is the *visit distribution*, not one-hot, when sims>1 + temperature>0 (the
    exploration signal is actually present);
  - targets are logged for **all** searched `DecisionKind`s (REACT/SLIP/DISCARD too),
    not just GEAR/CARDS;
  - only real-choice states logged; train/val track-disjoint;
  - the AZ checkpoint loads as a `MaskablePPO`, has the contract sidecar, round-trips
    through `MLAgent` returning only legal moves;
  - the value head is actually trained (its output moved from init on a held batch).

## Success criteria (rung 3 — improvement signal is real)

On the held-out generated solo field:

- The **trained** net used as the search's prior+value **beats the pre-training net**
  used as the search's prior+value on **both** worst-case L1 spins/pass **and**
  rounds-to-finish (strictly lower, on disjoint held-out seeds).
- The trained net's value head is **calibrated** enough to be a useful leaf: report
  its MSE vs. the realized rest-of-lap cost-to-go on the val split, and show that
  `leaf_mode="value_head"` search with the trained net ≥ `leaf_mode="rollout"` with the
  pre-training net (the value is starting to replace the rollout — the spine paying
  off).

If one generation does not beat the generator, report it honestly — it likely means
the search visit budget (C0) is too low to produce better-than-net targets, which is a
go/no-go signal for C3 (and the trigger to pull the Gumbel seam forward), not
something to mask.

## Risks & mitigations

- **Risk: the value target is the post-spin recovery (inflated), poisoning V — the S1
  lesson #2 hazard, now in the *training data*.** Mitigation: the MC return is floored
  with `_pre_spin_progress`, unit-tested on a forced-spin track; the single most
  important correctness check in C2.
- **Risk: the policy target is degenerate (one-hot) because the visit count is low or
  the temperature is off, so CE just clones the argmax (BC again).** Mitigation: log
  the *visit distribution* with a root temperature during generation (built in C1,
  turned on here); if it still collapses, that is the C0 visit-budget go/no-go
  surfacing.
- **Risk: training the critic and actor together destabilizes.** Mitigation: tune
  `c_v` on the val split; the separate actor/critic trunks
  (`share_features_extractor=False`) already decouple them — that is why the default
  exists (`model.py` Idea-13 note).
- **Risk: off-table REACT actions silently corrupt targets.** Mitigation: reuse
  `gen_demos`'s drop-and-count (S3 lesson #2) verbatim; assert every logged target
  sits in its own mask.
- **Risk: codec drift between generation and training.** Mitigation: stamp
  `CODEC_VERSION` in the `.selfplay.json` sidecar and the checkpoint `.meta.json`; the
  `MLAgent` tripwire fails fast on mismatch.

## Dependencies

- **C1** (`MCTSAgent` with the exploration hooks + the net adapter) — the generator
  *is* C1's search with noise on; the re-eval *is* C1's eval.
- `gen_demos.py` machinery (logging, codec snapping, drop-and-count, seed bands),
  `build_model` / `PPOConfig` / `save_checkpoint`, `MLAgent` load path.
- The S1 `_pre_spin_progress` for the value target. The GPU (RTX 4080, torch cu126)
  for the trainer.

## Effort

~1 sprint. Generator is mostly C1-reuse + `gen_demos`-reuse; the new work is the
joint policy+value trainer, the MC-return backfill, and getting the value target's
pre-spin discipline right.
