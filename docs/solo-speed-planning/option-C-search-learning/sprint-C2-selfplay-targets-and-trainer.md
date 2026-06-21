# Sprint C2 — Self-play target generation + policy/value trainer (one generation)

> The second foundation sprint. Make the search *teach the net*: generate
> `(obs, search_policy, search_value)` targets from C1's MCTS, train the
> policy+value net toward them once, and prove the trained net — used as the
> prior+value in the same search — **BEATS the net it learned from**. This is the
> single experiment that validates the AlphaZero improvement signal before we pay
> for the full loop in C3.

## Goal

A self-play data generator + a supervised policy/value trainer such that **one**
generation of (generate-with-C1-search → train) yields a net that, plugged back into
`MCTSAgent`, is measurably better on the rung-2 metrics than the net used to generate
the data (rung 3 of the success ladder).

## Scope

**In.**
- **Target generation** (`experiments/gen_selfplay.py`): run `MCTSAgent` (C1) over
  many GENERATED solo tracks (the `eval_search` tight params, on a self-play seed band
  **disjoint** from the held-out eval band), driving `run_round_driver` exactly as
  `gen_demos.py` / `HeatEnv` do. At every real-choice learner decision log:
  - `encode_observation(state, pid, decision)` (the frozen obs),
  - the **MCTS visit distribution** over the legal flat actions = the policy target
    `π` (the AlphaZero search-policy target, normalized over the masked/kept actions),
  - the `action_mask` (`legal_action_mask`),
  - the **value target** `z` = the search's root value / the realized rest-of-lap
    cost-to-go for that state (the spine V — expected rounds-to-finish), computed with
    the S1 pre-spin discipline so a spin does not inflate the target.
  Output a compressed `.npz` + a `.selfplay.json` provenance sidecar (codec version,
  search config, seed band) — mirror `gen_demos.py`'s format and its **drop-and-count**
  of any off-table REACT action the codec can't encode (S3 lesson #2).
- **Trainer** (`experiments/train_az.py`): a standalone supervised trainer (NOT PPO)
  that builds a real `MaskablePPO` via `build_model` (so weights live in the exact
  architecture `MLAgent` expects), then minimizes the AlphaZero loss on the targets:
  `L = CE(π_target, masked policy logits) + c_v · MSE(z, value_head)` (+ a small L2 /
  the trainer's weight decay). Train **both** the actor head/trunk and the **critic**
  (unlike S3 BC, which left the critic uninitialized) — here V is a first-class target.
  Save via `save_checkpoint` (zip + `.meta.json` contract sidecar). Hold out tracks for
  a validation split (track-disjoint, like `gen_demos`).
- **Re-evaluation**: plug the trained checkpoint back into `MCTSAgent` (as the prior +
  leaf value) and run the rung-2 eval vs. the **pre-training** net used to generate the
  data, on the held-out solo field — the rung-3 verdict.

**Out.**
- The *repeated* loop (multiple generations) — that is C3. C2 does exactly **one**
  generate→train→evaluate cycle to isolate the improvement signal.
- Opponents, MuZero, Gumbel sampling, the `large` net / multi-hour campaign (deferred).
- Any PPO / RL reward fine-tune (the AZ target is the search policy/value, not a reward
  — keep the two paths separate; PPO fine-tune is the S4 lineage, not this one).

## Deliverables

- `experiments/gen_selfplay.py` — the target generator (reuses C1's `MCTSAgent`,
  `gen_demos.py`'s logging/codec machinery, the disjoint seed band). Wrapped in
  `_runlog.run_main`.
- `experiments/train_az.py` — the policy+value supervised trainer (CE policy + MSE
  value), saving a contract-checked checkpoint. Wrapped in `run_main`.
- A trained checkpoint + the rung-3 re-eval report (trained-net-in-search vs.
  pre-training-net-in-search, held-out solo).
- Unit tests in `tests/test_az_targets.py` / `tests/test_train_az.py`: logged tuples
  are codec-valid and mask-consistent (target `π` support ⊆ mask, sums to 1); the
  value target is the pre-spin cost-to-go on a known forced-spin track (not the
  post-spin recovery); only real-choice states logged; train/val track-disjoint; the
  AZ checkpoint loads as a `MaskablePPO`, has the contract sidecar, and round-trips
  through `MLAgent` returning only legal moves; the value head is actually trained
  (its output moved from init on a held batch).

## Success criteria (rung 3 — improvement signal is real)

On the held-out generated solo field:

- The **trained** net used as the search's prior+value **beats the pre-training net**
  used as the search's prior+value on **both** worst-case L1 spins/pass **and**
  rounds-to-finish (strictly lower, on disjoint held-out seeds).
- The trained net's value head is **calibrated** enough to be useful as a leaf: report
  its MSE vs. the realized rest-of-lap cost-to-go on the val split, and show that
  `leaf_mode="value_head"` search with the trained net ≥ `leaf_mode="rollout"` with the
  pre-training net (the value is starting to replace the rollout — the spine paying
  off).

If one generation does not beat the generator, report it honestly — it likely means
the search visit budget (C0) is too low to produce better-than-net targets, which is a
go/no-go signal for C3, not something to mask.

## Risks & mitigations

- **Risk: the value target is the post-spin recovery (inflated), poisoning V — the S1
  lesson #2 hazard, now in the *training data*.** Mitigation: compute `z` with the
  pre-spin progress floor (`_pre_spin_progress`), unit-tested on a forced-spin track;
  this is the single most important correctness check in C2.
- **Risk: the search policy target is degenerate (one-hot) because the visit count is
  low, so CE training just clones the argmax (BC again, with BC's distribution gap).**
  Mitigation: log the *visit distribution*, not the argmax; add a small visit
  temperature at the root during generation (standard AZ) so `π` carries exploration
  signal; if it still collapses, that is the C0 visit-budget go/no-go surfacing.
- **Risk: training the critic and actor together destabilizes (the value loss swamps
  the policy, or vice-versa).** Mitigation: tune `c_v` on the val split; the separate
  actor/critic trunks (`share_features_extractor=False`) already decouple them, which
  is exactly why that default exists (`model.py` Idea-13 note).
- **Risk: off-table REACT actions silently corrupt targets.** Mitigation: reuse
  `gen_demos`'s drop-and-count (S3 lesson #2) verbatim; assert every logged target sets
  in its own mask.
- **Risk: codec drift between generation and training.** Mitigation: stamp
  `CODEC_VERSION` in the `.selfplay.json` sidecar and the checkpoint `.meta.json`; the
  `MLAgent` tripwire fails fast on mismatch.

## Dependencies

- **C1** (`MCTSAgent` + the net adapter) — the generator *is* C1's search; the
  re-eval *is* C1's eval.
- `gen_demos.py` machinery (logging, codec snapping, drop-and-count, seed bands),
  `build_model` / `PPOConfig` / `save_checkpoint`, `MLAgent` load path.
- The S1 `_pre_spin_progress` for the value target. The GPU (RTX 4080, torch cu126)
  for the trainer.

## Effort

~1 sprint. Generator is mostly C1-reuse + `gen_demos`-reuse; the new work is the
joint policy+value trainer and getting the value target's pre-spin discipline right.
