# Sprint B1 — Offline DP solver + value/budget map

> Turn the validated reduced model (B0) into a solved plan: `V*(pos, heat, gear)`
> ≈ expected rounds-to-finish (the shared spine) plus an explicit per-sector heat
> budget, computed **once per track** in milliseconds.

## Goal

Implement backward dynamic programming / value iteration over the reduced
`(pos, heat, gear)` MDP for a given `Track`, producing (a) the optimal value map
`V*`, (b) the optimal reduced policy `π*(s) = (Δgear, intent)`, and (c) a derived
**per-sector heat budget**: for each corner, the target heat to carry in and the
cooldown banking the plan relies on to afford it. Verify correctness on small
hand-checkable tracks and measure `precompute ms/track`.

## Scope

**In**
- The DP/value-iteration solver over the B0 reduced transition.
- Extraction of `π*` and the per-sector heat budget from `V*`.
- A `PlannedTrack` artifact (the solved plan) that B2 consumes.
- Correctness tests on tiny tracks where the optimum is hand-derivable.
- Precompute-cost measurement.

**Out**
- The online controller (B2) and full-game eval (B3).
- Track-agnostic/feature-based value (deferred — see README §4).
- Re-deriving the reduced model (owned by B0; B1 imports it).

## Deliverables (concrete)

- `src/heat/planning/dp_solver.py`
  - `@dataclass PlannedTrack` — the solved plan:
    - `value: dict[ReducedState, float]` (or a dense `ndarray` indexed by
      `pos × heat × gear`) — `V*`.
    - `policy: dict[ReducedState, ReducedAction]` — `π*`.
    - `heat_budget: list[SectorBudget]` — per corner: `corner`, `target_heat_in`,
      `cool_window: tuple[int, int]` (position range where the plan banks heat),
      `expected_spend`.
    - `precompute_ms: float`.
  - `solve_track(track, *, model_params, mode, gamma=1.0,
    own_spin_penalty, laps) -> PlannedTrack`:
    - Enumerate states `pos ∈ [0, length·laps)` × `heat ∈ [0, 6]` × `gear ∈
      [1, 4]`.
    - Backward sweep over `pos` (finish→start) since position is near-monotone;
      a small fixed number of value-iteration passes (or in-place Gauss–Seidel)
      settles the **spin back-edges** (spin sends `pos → corner.start−1`).
    - Bellman backup over actions `(Δgear ∈ legal_gear_shifts, intent ∈
      {conserve,mean,push})` using `reduced_model.step_reduced(...)`; for `banded`
      mode, expectation over the speed bands; cost = 1 round/transition (+ shaped
      spin penalty), terminal `V(pos ≥ length·laps) = 0`.
  - `derive_heat_budget(planned, track) -> list[SectorBudget]` — walk `π*` from a
    representative start state to read, per corner, the planned carry-in heat and
    the cooldown window.
- `experiments/solve_track_demo.py` (wrapped in `_runlog.run_main`)
  - Solve a held-out generated track, **print the value-over-position map** and the
    per-sector budget (the interpretability payoff), and report `precompute ms`.
- `tests/planning/test_dp_solver.py`
  - A trivial straight track (no corners): `V*` equals the obvious min-rounds; the
    policy pushes the highest safe gear.
  - A one-tight-corner track: `π*` **conserves into the corner** and `V*` reflects
    the rounds cost of slowing; a hand-derived optimum matches within tolerance.
  - Determinism: same `(track, params)` → byte-identical `PlannedTrack` value map.
  - Monotonicity sanity: `V*(pos)` is non-increasing in `pos` along the optimal
    path (closer to finish ⇒ fewer rounds-to-go).

## Success criteria (measurable)

- Solver returns the **hand-derived optimum** on the tiny test tracks (straight and
  single-tight-corner) within rounding tolerance.
- **`precompute ms/track` ≤ ~50 ms** on the default generated distribution (state
  space ≈ `length·laps × 7 × 4` ≈ a few thousand states) — i.e. cheap enough to
  solve at race start with no perceptible cost. Reported in the run log.
- `V*` is **monotone non-increasing in `pos`** along the optimal path and finite at
  every reachable state (no unreachable/`inf` leaks in the policy support).
- The derived per-sector heat budget is **consistent with `π*`** (carry-in heat at
  each corner equals what `π*` actually accumulates) — checked in a test.
- Determinism: identical plan for identical inputs (the project's standing
  requirement; mirrors S1's byte-stable selection test).

## Risks & mitigations

- **R1 — Spin back-edges create cycles** (spin sends position backward), breaking a
  pure single-pass backward DP. *Mitigation:* a bounded number of value-iteration
  sweeps over the back-edges converges quickly (spins strictly increase rounds, so
  the operator is a contraction under the round cost); cap passes and assert
  convergence (max value change < ε).
- **R2 — `gamma`/penalty mis-scaling** makes `V*` not read as rounds-to-finish.
  *Mitigation:* keep `gamma = 1.0` and **1 round per transition** so `V*` is
  literally expected rounds (the spine units A/C expect); the own-spin penalty is
  additive shaping kept comparable to the S1 `DEFAULT_OWN_SPIN_PENALTY` discipline
  (own-spin dominates), and B3 checks `V*` correlates with realized rounds.
- **R3 — Multi-lap state blow-up** if `pos` spans `length·laps` naively. *Mitigation:*
  per-lap structure is identical except the finish boundary; solve over the full
  `length·laps` range (still only thousands of states) or collapse identical lap
  interiors if the budget metric demands it.
- **R4 — The `intent` action is too coarse** to express the real card choice.
  *Mitigation:* `intent` is a *plan-level* control; the true card resolution lives
  in B2's 1-ply lookahead, so coarseness here is intentional and bounded. B3 tunes
  the band count if needed.

## Dependencies

- **B0 must have PASSED** (the reduced model + chosen fidelity mode).
- `src/heat/planning/resource_model.py`, `reduced_model.py` (from B0).
- `rules.*`, `Track`/`Corner` (present).

## Rough effort

**~1 sprint.** The DP is small and standard once the transition exists; the real
work is the spin back-edge convergence, the budget extraction, and the tiny-track
correctness tests that make the solver trustworthy.
