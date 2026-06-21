# Sprint A1 — Value-net trainer + Monte-Carlo dataset

> Build the spine object: a **trained rest-of-lap value `V(state) ≈ expected
> rounds-to-finish`**, fit by Monte-Carlo regression on simulator rollouts, in a
> codec-compatible checkpoint the search (A2) and later options (B/C) reuse.
> This sprint produces a *standalone, testable artifact* — a calibrated `V`
> checkpoint with its own regression metrics — and does **not** touch the search.

## Goal

Generate a track-disjoint Monte-Carlo value dataset (every solo state labeled
with its realized **rounds-remaining**) and train **only the model's critic head**
to regress it, producing a `V` checkpoint that calibrates on held-out states.

## Scope

**In:**
- A data generator that drives full solo races with `HeuristicAgent` over the
  TRAIN/VAL generated-track bands and labels every learner state with
  rounds-remaining.
- A supervised value-regression trainer over the frozen obs codec, training the
  critic head/trunk only (actor left at init), saved via `save_checkpoint`.
- Regression-level evaluation: val MAE overall + **bucketed by corner speed-limit**
  and by distance-to-finish; a calibration sanity check.

**Out:**
- Any change to `LookaheadAgent` / `_leaf_score` (that is A2).
- Opponents / 4p / `determinize_hidden` (solo only).
- TD bootstrapping (MC only; TD deferred per README §7).
- Policy iteration / refitting on search trajectories (A3).

## Deliverables (concrete files / functions)

- **`experiments/gen_value_data.py`** — MC value-dataset generator. Mirrors
  `experiments/gen_demos.py`'s structure (the proven driver-loop + track-disjoint
  band pattern), but labels **rounds-remaining** instead of action targets:
  - reuse `_TIGHT_PARAMS`, the `_TRAIN_SEED_BASE=100_000` / `_VAL_SEED_BASE=500_000`
    bands, and the `run_round_driver` solo loop (`player.lap=1` init, `MAX_ROUNDS`
    guard) verbatim from `gen_demos.py`.
  - drive seat 0 with `HeuristicAgent` (the S1-validated rollout policy); **no
    opponent seats** (solo).
  - at **every** learner decision (not just real-choice — V scores all states),
    record `obs = encode_observation(state, 0, decision=None)` plus the current
    `state.round_num` and a per-state row id, so post-hoc labeling can compute
    `rounds_remaining = finish_round − round_num`. (Encode with `decision=None`
    to match how A2 will encode a leaf — README §2.3.)
  - on race completion, set `finish_round = state.round_num` and backfill the
    `rounds_remaining` label for every row of that race. Drop races that hit
    `MAX_ROUNDS` without finishing (degenerate; would poison the target) and
    count them.
  - output a compressed `.npz` with `obs (N, OBS_DIM) float32`,
    `rounds_remaining (N,) float32`, `corner_limit_at_state (N,) int8` (the
    speed-limit of the next corner ahead, via `rules.distance_to_next_corner` /
    the track, for bucketed error reporting), `track_seed (N,) int64`, and
    `split (N,) S5`; plus a `.value.json` provenance sidecar recording
    `codec_version`, the bands, the generator policy, and rollouts-per-track.
  - wrap `main` in `experiments/_runlog.py`'s `run_main("gen_value_data", main)`.
- **`experiments/train_value.py`** — value-regression trainer:
  - load the `.npz`, assert its `codec_version == spaces.CODEC_VERSION` and
    `obs.shape[1] == OBS_DIM` (fail fast on drift).
  - build a `MaskablePPO` via `heat.ml.model.build_model` (so the weights live in
    the exact `HeatMLPExtractor` + critic architecture `MLAgent`/A2 expect), with
    `PPOConfig(share_features_extractor=False)` so the critic owns its trunk.
  - train **only `policy.value_net` + the critic features extractor** (the actor
    trunk/head frozen) by MSE (or Huber) on the target
    `target = -rounds_remaining` (negated so larger = better, matching the leaf
    sign — README §2.1). A small standalone Adam loop over minibatches, exactly
    the shape of `train_bc.py`'s supervised loop but with a regression loss on the
    value head instead of masked cross-entropy on the policy head.
  - **track-disjoint** train/val split honored (filter rows by `split`).
  - save via `heat.ml.training.save_checkpoint(..., track_name="generated",
    num_players=1, ppo_config=cfg)` so the `.meta.json` carries the contract
    tripwire fields A2's `MLAgent`-style load will check.
  - print per-epoch train/val MSE + **val MAE bucketed by `corner_limit_at_state`**
    and by distance-to-finish bins.
- **`tests/test_value_net.py`** — unit tests:
  - dataset rows are codec-valid (`obs.shape[1] == OBS_DIM`, finite, in `[-1,1]`);
    `rounds_remaining >= 0`; train/val track seeds disjoint.
  - labeling is correct: for a tiny synthetic race, `rounds_remaining` equals
    `finish_round − round_num` for each row; `MAX_ROUNDS`-truncated races are
    dropped.
  - a trained (tiny, few-epoch) checkpoint **loads as a `MaskablePPO`**, carries
    the contract sidecar, and `policy.predict_values(encode_observation(...))`
    returns a finite scalar (the hook A2 calls).
  - **encoding parity:** a state encoded by `gen_value_data` (`decision=None`)
    is byte-identical to the same state encoded at a rolled-out leaf (the
    encoding A2 uses), so train and inference see the same vector.

## Success criteria (measurable, on the shared eval)

This sprint's gate is **regression-level (Rung 0 of the README success ladder)**,
the necessary precondition for A2's behavioral win:

- **Calibration:** on held-out *val* states, V's predicted rounds-remaining tracks
  the realized value — report val MAE; require monotonicity in aggregate (mean
  predicted rounds-remaining decreases as `dist_to_finish` decreases).
- **Corner-bucketed error reported honestly:** val MAE bucketed by corner speed
  limit is printed, with the **limit-1 bucket called out** (the high-variance,
  high-stakes slice — README §7). No hard threshold here (variance is expected);
  the requirement is that it is *measured and reported*, so A2 knows whether to
  trust V near tight corners.
- **Artifact contract:** the checkpoint loads under the `MLAgent` sidecar
  tripwire (matching `obs_dim`/`action_dim`/`codec_version`) — proven by a test —
  so A2 and C can reuse it unchanged.

## Risks & mitigations

- **MC variance at limit-1 states (a spin adds many rounds).** → Report
  per-corner-limit MAE; make rollouts-per-track a CLI knob so A1 can raise sample
  count for the tight bucket if needed. Deeper fix (TD) deferred, not built.
- **Label leakage via `decision`.** Encoding with a real `decision` would leak the
  pending-decision phase block into a *state* value. → Always encode with
  `decision=None`; assert encoding parity with A2 in a test.
- **`HeuristicAgent` ceiling.** V trained on heuristic trajectories can't credit a
  better-than-heuristic line. → Accepted for A1 (it is the cheap first fit); A3
  refits on search-improved trajectories if A2 plateaus.
- **`MAX_ROUNDS` truncated races poisoning targets.** → Dropped + counted; the
  heuristic finishes ~100% solo on the gate band, so the drop rate should be
  near-zero (reported as a sanity check).

## Dependencies

- The frozen codec (`encode_observation`, `spaces.CODEC_VERSION`), `build_model` /
  `HeatMLPExtractor` (the critic head), `save_checkpoint` / `load_meta`,
  `GameState.clone` + `run_round_driver`, `HeuristicAgent`, the generator
  (`TrackGenParams` / `generate_track`), and the `gen_demos.py` band/loop pattern —
  all built.
- No dependency on A2/A3.

## Rough effort

~1 sprint. The data generator is a structural copy of `gen_demos.py` (relabel
rounds-remaining); the trainer is `train_bc.py`'s loop with a regression head; the
risk is concentrated in calibration quality at limit-1, which is measured not
assumed.
