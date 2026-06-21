# Solo Speed — Whole-Track Planning: Plans (A, B, C)

This folder holds the **sprint-broken implementation plans** for making our car go
fast around the generated-track distribution. It is the detailed follow-up to the
options landscape in
[`../solo-speed-whole-track-planning-options.md`](../solo-speed-whole-track-planning-options.md)
(read that first for the full rationale and the comparison of all six options).

We are expanding the three highest-value options into full plans:

- **[Option A — Learned leaf value](option-A-learned-value/README.md)** — plug a
  trained rest-of-lap value into the existing search. The cheap, de-risking win
  that every other option benefits from.
- **[Option B — Offline whole-track DP](option-B-track-dp/README.md)** — solve
  position×heat once per track, then execute cheaply. The literal "plan the lap,
  drive it cheaply" approach; exploits the known track.
- **[Option C — Search + learning (solo AlphaZero/MuZero)](option-C-search-learning/README.md)**
  — search and a policy+value net co-improve. The only option that can *surpass*
  today's search. Planned **foundations-first**: a minimal, correct core we can
  extend, not a sprawling one-shot.

D / E / F are kept as brief stubs at the bottom for later expansion.

---

## Scope (fixed for all three plans)

**In:** one car, minimize **rounds-to-finish** (≈ lap time) on held-out *generated*
tracks, keep tight (limit-1) corners clean, and budget **heat over the whole lap**.
Our own **card-draw uncertainty is in** (chance over our deck). **Out:** opponent
modeling, hidden-info determinization, multiplayer win-rate (the spine is built so
these can be layered back later — do not design them away, just don't build them
now).

## The shared spine (all three plans align on this)

Every plan, at bottom, produces and/or uses one object:

> **V(track position, heat, speed/gear, hand-summary) ≈ expected rounds to finish**
> (a cost-to-go), with our own future draws taken **in expectation**.

- **A** trains V and plugs it into search as the leaf evaluator.
- **B** computes a V (and an explicit per-sector heat budget) by dynamic
  programming over a reduced position×heat model.
- **C** trains V *and* a policy by self-play search, so both improve past the
  teacher.

Keeping V as the common interface is deliberate: a good V from A or B is a
ready-made warm start / sanity check for C, and C's V can back-fill A's leaf.

## Hard constraints (from the 2026-06-21 runs — every plan must respect)

1. **Depth alone is inert-to-harmful** — the lever is **leaf/position-value
   quality**, not deeper search. (Horizons 2/3/4 tied; naive depth selected spins.)
2. **Imitation caps ~0.6 corner-card accuracy** — to *exceed* the current search we
   must learn from **outcomes/value**, not just labels.
3. **Heat is the long-horizon blind spot** the current 2-ply search cannot see —
   adding whole-lap heat budgeting is the point.
4. **The track is known in advance** (`generate_track(seed, params)`) — a
   precompute lever (central to B).
5. **Keep the corner-skill scoring lessons** — own-spin penalty dominates, pre-spin
   progress floor, `HeuristicAgent` (not `StrongHeuristicAgent`) as the rollout
   policy. Don't regress these.

## Repo assets every plan reuses (do not reinvent)

- Simulator: `GameState.clone()` + `heat.engine.driver.run_round_driver` (the loop
  `HeatEnv` uses).
- Frozen codec: `heat.ml.features.encode_observation`, `heat.ml.action_codec`,
  `heat.ml.spaces.CODEC_VERSION` (a new value/policy net must stay codec-compatible
  so it drops into `MLAgent` / `evaluate_ml`).
- Search: `heat.agents.search_agent.LookaheadAgent` — already has a pluggable
  **`leaf_value`** hook (`"progress"` / `"move_eval"`); A adds `"learned"`.
- Net: the model's **separate critic/value head** (designed as "how's the race
  going") in `heat.ml.model` — the natural home for V.
- Tracks: `heat.tracks.generator` (`TrackGenParams`, `generate_track`).
- Eval: `experiments/eval_search.py` (rounds-to-finish + spins bucketed by corner
  speed-limit, held-out generated band) — extend, don't replace.
- Run reporting: wrap new entry points in `experiments/_runlog.py` `run_main`
  (START/DONE/FAILED markers); pass `--n-envs > 1` for throughput; `verbose=1`.

## Shared evaluation (every sprint that produces an agent must gate on this)

- **Primary:** rounds-to-finish **and** worst-case (p90/max) spins/pass **bucketed
  by corner speed-limit**, on **held-out generated tracks** (disjoint seed band).
  Never finish-rate alone.
- **Heat-efficiency:** heat spent vs. distance/time + cool-downs taken — proves the
  *budgeting* improved, not just corner survival.
- **Cost:** ms/move + clones/move (reuse the `SearchProfile` print).
- **Baselines:** `HeuristicAgent` (the bar), S1/S2 `LookaheadAgent` (to surpass),
  and a model-free control where relevant.

## Dependency / suggested order

```
A (learned V → search)         ← cheapest, de-risks the spine; do first
  └─ feeds → C's value warm-start / leaf back-fill
B (offline track DP)            ← standalone; the known-track "plan the lap" idea
  └─ feeds → C/A a model-based prior
C (search + learning)           ← high ceiling; let A/B de-risk V first
```

## Sprint-file conventions (followed by all three subfolders)

Each option subfolder has a `README.md` (option overview + ordered sprint list +
success ladder) and one file per sprint named `sprint-<X><n>-<slug>.md`
(e.g. `sprint-A1-value-net-and-data.md`). Each sprint file states: **Goal**,
**Scope** (what's in/out), **Deliverables** (concrete files/functions), **Success
criteria** (measurable, on the shared eval), **Risks & mitigations**,
**Dependencies**, and a rough **effort** estimate. Mirror the tone/precision of
`docs/sprint-search-imitation-design.md`.

---

## Deferred options (brief — expand later only if A–C leave a gap)

- **D — Hierarchical RL (manager/worker).** A manager sets a per-sector heat intent
  ("push / conserve / set up the corner"); a worker executes turns toward it. The
  learned version of "plan the lap, re-plan rarely." Expand if a *flat* value (A/C)
  plateaus on long-horizon heat budgeting. ~3–4 sprints; HRL training is finicky.
- **E — MPC / plan-and-cache.** Receding-horizon: re-plan a short trajectory to a
  terminal value only every K turns / at sector boundaries, execute the cached plan
  between. An *inference* layer on top of A's V — your "don't recalculate
  constantly." Expand once V exists and we want faster, steadier inference.
  ~2–3 sprints.
- **F — Model-free RL + whole-track observation.** PPO with richer "what's up the
  road" eyesight and the corner lessons baked in; no search. Keep as the
  **baseline/control** — model-free was already refuted on the limit-1 problem, so
  it is a yardstick, not the main bet. ~1–2 sprints.
