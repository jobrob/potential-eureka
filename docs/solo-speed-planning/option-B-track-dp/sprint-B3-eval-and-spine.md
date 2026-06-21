# Sprint B3 — Eval, tuning & spine hand-off

> Gate `TrackDPAgent` the way 8C/S3 taught us — held-out generated, worst-case by
> corner limit, plus a heat-efficiency metric — tune the abstraction against the
> results, and export `V*` so Options A and C can reuse it as a warm-start / leaf
> prior.

## Goal

Extend the shared eval harness to include `TrackDPAgent` and a **heat-efficiency**
metric; run the full ladder vs `HeuristicAgent` and the S1/S2 `LookaheadAgent`;
tune the model fidelity / penalty knobs to clear the ladder; and produce a
codec-compatible, reusable export of the DP value (`V*`) plus a short report on how
well it predicts realized rounds-to-finish (the spine hand-off to A/C).

## Scope

**In**
- Adding `TrackDPAgent` (and reporting its per-track precompute) to the eval.
- A **heat-efficiency** metric (heat spent per space + cooldowns taken) added to
  the harness, since the standard harness reports spins/finish/rounds but not the
  budgeting signal Option B is built to improve.
- Tuning: model fidelity mode, intent-band thresholds, spin penalty, optional
  re-solve-on-drift trigger — driven by the eval, not by hope.
- Exporting `V*` and documenting the warm-start/leaf-prior interface for A/C.

**Out**
- Building new model/solver/agent machinery (owned by B0–B2; B3 tunes + gates).
- Opponents / multiplayer win-rate (solo scope; the spine is left layerable).
- Actually training Option A/C (B3 only *hands off* `V*`).

## Deliverables (concrete)

- `experiments/eval_track_dp.py` (or a `--track-dp` extension to
  `experiments/eval_search.py`, wrapped in `_runlog.run_main`)
  - Registers `TrackDPAgent` alongside `Heuristic` / `Lookahead` / `LookaheadDet`
    in the existing `_run_field` flow, reusing `_HELDOUT_BASE`, `_TIGHT_PARAMS`,
    and the `_passes_and_spins_by_limit` / `spin_stats` reconstruction unchanged.
  - Prints the standard block (finish%, rounds, spins/pass by limit mean/p90/max,
    pooled spins/passes) **and** the new heat-efficiency rows, **and** the
    `SearchProfile` ms/move plus a separate `precompute ms/track` line.
- `src/heat/planning/heat_metrics.py` (or a helper in the eval)
  - `heat_efficiency(event_log, player_id, track) -> HeatEff` reconstructing from
    the event log: total heat units spent (gear-shift + corner overspeed), heat
    recovered via cooldown, and `heat_spent_per_space`. Mirrors the exact-from-log
    reconstruction discipline of `_passes_and_spins_by_limit`.
- `src/heat/planning/value_export.py`
  - `export_value(planned_track, track) -> ValueTable` and a serialization
    (e.g. `save_value(path)`) keyed so it can be consumed by an Option-A leaf or an
    Option-C value head. Documents the units (**expected rounds-to-finish**,
    `gamma=1.0`) and the `(pos, heat, gear)` → `V*` mapping. Where a
    codec-compatible observation is needed, note how `(pos, heat, gear)` maps onto
    the frozen `features.encode_observation` fields so the table can seed a net.
- `docs/solo-speed-planning/option-B-track-dp/RESULTS.md` (written at sprint end)
  - The eval table, the chosen fidelity/penalty config, the heat-efficiency
    comparison, and the `V*`-vs-realized-rounds correlation.
- `tests/planning/test_value_export.py`
  - Round-trips a `PlannedTrack` through `export_value`/`save_value` and asserts the
    table reproduces `V*` and is keyed consistently.

## Success criteria (measurable, shared eval — solo, held-out generated)

Climbs the README success ladder, reported worst-case by corner limit:

- **Rung 1 (safety):** finish 100% solo, worst-case spins/limit-1-pass ≤
  HeuristicAgent.
- **Rung 2 (budgeting):** a **heat-efficiency win** — fewer heat units/space and/or
  fewer spins than HeuristicAgent at equal-or-better rounds-to-finish. This is the
  headline that B exists to prove (budget, not just survival).
- **Rung 3 (speed/cost):** rounds-to-finish **≤ S1/S2 `LookaheadAgent`** at **a
  fraction of its ms/move** (precompute counted separately and shown to be cheap).
- **Rung 4 (spine):** `V*` exported in a reusable, documented form and shown to
  **correlate with realized rounds-to-finish** (report the correlation), making it
  a usable warm-start for Option A's leaf / Option C's value head.

A clear written verdict per rung (PASS/FAIL), and — if Rung 2 or 3 fails — a
diagnosis pointing at the specific abstraction divergence (traceable back to the
B0 fidelity metrics), with the recommended fix or the honest "fold into A" call.

## Risks & mitigations

- **R1 — Beats heuristic on safety but not on rounds-to-finish** (the plan is too
  conservative). *Mitigation:* tune intent-band thresholds and spin penalty toward
  `push` where `V*` shows slack; the eval is the loss function, so tuning is
  grounded, not guessed.
- **R2 — Heat-efficiency metric is gameable / ambiguous.** *Mitigation:* define it
  purely from the event log (spent vs recovered vs distance) the same way the spin
  metric is reconstructed, and always report it **alongside** rounds and spins so a
  cheap "spend no heat by crawling" degenerate is caught by the rounds column.
- **R3 — `V*` does not transfer to A/C** (units or state mismatch). *Mitigation:*
  fix `gamma=1.0`/rounds units in B1, document the `(pos,heat,gear)`→codec mapping
  here, and validate the correlation before claiming the hand-off; if it does not
  correlate, B3 reports that and scopes the warm-start claim down honestly.
- **R4 — Per-track precompute inflates wall-clock at eval scale** (many tracks).
  *Mitigation:* B1's ≤ ~50 ms/track budget keeps a full held-out sweep cheap;
  precompute is reported separately so it never masquerades as per-move cost.

## Dependencies

- **B2 (`TrackDPAgent`)**, transitively B1 + B0.
- `eval_search.py` harness internals (`_run_field`, `_AgentAgg`,
  `_passes_and_spins_by_limit`, `spin_stats`) — extend, don't replace.
- The frozen codec (`features.encode_observation`, `spaces.CODEC_VERSION`) for the
  `V*` export mapping — read-only reference, no codec change.

## Rough effort

**~1 sprint.** Mostly harness extension, tuning runs, and the export + correlation
report. Lighter than B0–B2 in new code but it is the gate that decides whether B
ships and whether the spine hand-off is real.
