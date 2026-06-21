# Sprint B0 — Feasibility spike: reduced-model fidelity

> **The gate that decides whether Option B is worth building.** Before any solver,
> validate that a reduced `(pos, heat, gear)` model with cards-as-stochastic-
> resource predicts true-engine per-turn outcomes well enough to plan against.
> Fail fast if the abstraction is too lossy.

## Goal

Build (a) the **stochastic resource model** `P(S | gear, intent)` from a deck
composition, and (b) a **reduced one-turn simulator** `step_reduced(s, a) → s'`
that predicts the next `(pos, heat, gear)` and whether a spin occurs — then
**measure its disagreement with the true engine** (`run_round_driver`) on states
drawn from real games, bucketed by corner speed-limit. Produce a single go/no-go
verdict with numbers.

## Scope

**In**
- The resource model at three fidelities (expected-only, banded/few-moment,
  empirical-sample) and a switch to compare them.
- The reduced one-turn transition: gear-shift heat, speed sampling/marginalizing,
  `corners_crossed` + `corner_heat_cost`, spin detection, cooldown-refill estimate.
- A fidelity harness that pairs each reduced prediction against the **true**
  outcome from cloning the live `GameState` and running exactly one turn.
- Error metrics bucketed by corner limit (limit-1 is the headline).

**Out**
- The DP solver (B1) and the online controller (B2) — B0 only proves the model.
- Track-agnostic features, opponents, slipstream/blocking.
- Any agent that plays a full game (B0 produces a report, not a player).

## Deliverables (concrete)

- `src/heat/planning/__init__.py` — new package for Option B.
- `src/heat/planning/resource_model.py`
  - `class SpeedResourceModel` built from a `PlayerState`/deck composition:
    - `basic_value_multiset(player) -> list[int]` (reuse logic behind
      `_move_eval.expected_basic_value`; include owned Basics + stress-as-mean).
    - `speed_distribution(gear, intent) -> dict[int, float]` — `intent ∈
      {"conserve","mean","push"}`; `conserve` uses lower order statistics of the
      `gear`-card draw, `push` the upper, `mean` the centre.
    - `expected_speed(gear, intent) -> float` and `speed_bands(gear, intent) ->
      list[tuple[int, float]]` (the few-moment discretization the DP consumes).
- `src/heat/planning/reduced_model.py`
  - `@dataclass(frozen=True) ReducedState: pos: int; heat: int; gear: int`
  - `@dataclass(frozen=True) ReducedAction: dgear: int; intent: str`
  - `step_reduced(state, action, track, model, *, mode) -> StepOutcome` where
    `StepOutcome` carries `next_pos, next_heat, next_gear, spun: bool,
    p_spin: float, exp_rounds_cost: float`. `mode ∈ {"expected","banded",
    "empirical"}`.
  - Heat/corner arithmetic reuses `rules.corners_crossed`,
    `rules.corner_heat_cost`, `rules.legal_gear_shifts`, `rules.cooldown_amount`.
- `experiments/spike_reduced_model.py` (wrapped in `_runlog.run_main`)
  - Draws `N` `(GameState, player)` snapshots by running `HeuristicAgent` and the
    S1 `LookaheadAgent` on held-out generated tracks (reuse `generate_track` +
    `_TIGHT_PARAMS` so limit-1 corners are well represented), logging the state at
    each turn start.
  - For each snapshot and each candidate `(gear, intent)`: compute the reduced
    prediction **and** the true outcome (clone the state, force that gear and the
    controller-rule card play, run one turn via `run_round_driver`, read true
    `next_pos/next_heat/spun`).
  - Reports per corner-limit bucket: median/p90 absolute error in `next_heat`,
    median/p90 error in `next_pos`, and **spin-agreement** (predicted-spin vs
    actual-spin confusion: precision/recall on the limit-1 bucket especially).
  - Prints a one-line **VERDICT: PASS/FAIL** against the Rung-0 bar, per fidelity
    mode, so we can read which model is good enough.
- `tests/planning/test_resource_model.py`, `tests/planning/test_reduced_model.py`
  - Deterministic unit tests: known deck → known `speed_distribution`; a hand-built
    `(state, action)` whose true engine outcome is hand-checkable matches
    `step_reduced` (heat arithmetic, spin trigger at `cost > heat`, cooldown refill
    in gears 1–2 only).

## Success criteria (measurable, on the shared eval band)

- **PASS** for at least one fidelity mode (prefer the cheapest that passes):
  - **Median absolute end-of-turn heat error ≤ 1** overall and in the **limit-1
    bucket** specifically.
  - **Spin-agreement ≥ ~90%** (predicted spin matches actual) in the limit-1
    bucket — the model must not say "safe" where the engine spins, nor vice versa,
    on tight corners.
  - `next_pos` median error ≤ 1 space.
- The report is produced on **held-out generated tracks**, bucketed by limit,
  worst-case (p90) shown alongside median.
- A written **go/no-go** paragraph: if no mode passes, B0 outputs the *specific*
  divergence (e.g. "hand-memory loss inflates conserve speed at limit-1") and the
  recommendation to either (i) add the mitigation and re-spike, or (ii) abort B and
  fold the resource model into Option A as a feature.

## Risks & mitigations

- **R1 — Expected-only is too optimistic at tight corners** (averaging hides the
  above-mean draw that spins). *Mitigation:* the banded/conditional model is built
  in the same sprint; B0 reports all three so we pick empirically, not by hope.
- **R2 — Dropped hand-memory biases the conserve intent** (the model assumes a
  fresh draw; a real driver holds a known low card). *Mitigation:* measure the bias
  explicitly (limit-1 conserve bucket); the B2 controller's 1-ply lookahead is the
  designed recovery, and B0 quantifies how much it must recover.
- **R3 — Cooldown/clog timing is stochastic and only modeled in expectation.**
  *Mitigation:* heat error is the headline metric precisely because clog shows up
  there; if heat error fails only via clog, B0 recommends a richer clog term before
  B1.
- **R4 — The snapshot distribution is unrepresentative** (only states a heuristic
  visits). *Mitigation:* draw snapshots from **both** HeuristicAgent and the S1
  LookaheadAgent so the model is tested on the recovery states that matter.

## Dependencies

- Engine: `GameState.clone`, `run_round_driver`, `rules.*` (all present).
- `_move_eval.expected_basic_value` / `play_speed` (present) for the speed model.
- `generate_track` + `eval_search._TIGHT_PARAMS` band for snapshots.
- No dependency on B1/B2. **B1 and B2 depend on B0 passing.**

## Rough effort

**~1 sprint, front-loaded.** The model and reduced step are small; most effort is
the fidelity harness and honestly characterizing the gap. This sprint *can* end in
a "no" — that is a successful outcome (cheap failure), not a wasted one.
