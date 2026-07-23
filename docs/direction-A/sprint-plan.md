# Direction A — sprint plan (PPO self-play workhorse, the Big 2 path)

> **Status:** **implemented** (completed 2026-07-16). A0–A8 are complete;
> G0002 passed A8's frozen multi-ruler gate on an untouched generated-track band. Turns Direction A of
> [`heat-rl-transition-plan.md`](../heat-rl-transition-plan.md) into an ordered set of
> sprints with go/no-go gates. **High-level only** — each sprint gets a detailed design
> when we reach it. Grounded in the current code (`src/heat/ml/{env,model,features,
> action_codec}.py`) and the C6 feasibility findings.
> The committed next phase is
> [Direction D's affordable-substrate design](../direction-D/D0-big-picture-design.html).

## What "complete Direction A" means

A multi-agent PPO self-play loop with the Big 2 architecture that plays **full
multiplayer Heat (2–6 seats), under imperfect information, on any generated track**, and
**clears the held-out skill bar** (beats the strong heuristic by a defined margin across
held-out tracks and seat counts). Affordability of the *full-scale* run is Direction D's
job; Direction A's job is to **prove the method learns and generalizes at small scale**
(transition-plan Phases 0 and 1). Direction B (search) and Direction D (substrate) are
explicitly out of scope here, except where Direction A consumes Direction C's dense
targets as a shared enabler.

## The pivot that drives the sequence

The current learner is single-seat `HeatEnv` + SB3 `MaskablePPO`, a flat 104-float
observation, and a **fixed 516-wide masked `Discrete` head**. Direction A needs three
things SB3-Maskable does not give cleanly: (1) **true N-seat shared-policy self-play**,
(2) a **variable-length legal-action dot-product head** (rank only currently-legal
actions, not a 516-wide masked softmax), and (3) a **structured hidden-info encoder**
(card embeddings + attention, not a flat vector). So the first real decision is the
**learner substrate** — and most early sprints replace pieces of the SB3 stack rather
than extend it. We keep the engine, `action_codec` legality, `features` building blocks,
and the league/Wilson-LB eval gate.

---

## Sprint sequence

### A0 — Substrate decision + skeleton (design + small spike)
**Goal:** decide how we run PPO self-play with a variable action head: fork a **minimal
custom PPO** (Big 2 style, full control of the action head and multi-agent rollout) vs.
bend SB3-Maskable. Spike both far enough to choose.
**Why first:** every later sprint's interface (rollout format, action head, loss)
depends on this. Picking wrong here is the most expensive mistake.
**Deliverable:** decision doc + a runnable training skeleton (random/tiny net) that
collects a self-play rollout and takes one PPO update end-to-end.
**Gate:** skeleton trains without error on a trivial task; chosen substrate can express
a variable-length action head and per-seat rollouts. *(Recommendation going in: a small
custom PPO — SB3-Maskable's fixed `Discrete` head fights both requirements.)*

### A1 — Tiny-Heat proving ground
**Goal:** a deliberately small Heat variant (short track, small deck, 2–3 seats,
imperfect info) exposed through the same engine/driver, with a fast reset/step path —
the "Connect4-equivalent" cheap-iteration bed (transition-plan Phase 0).
**Why:** C6's #1 lesson — prove the method *moves* where iteration is cheap before
paying for scale. Decouples method risk from the real game's cost.
**Deliverable:** Tiny-Heat config + a parametrized env; sanity sims confirm legal play
and termination.
**Gate:** a full self-play game runs in Tiny-Heat at a target games/sec that makes a
Phase-0 method experiment a minutes-to-hours loop, not days.

### A2 — N-agent self-play harness
**Goal:** drive 2–N **shared-policy** seats over the engine, each acting on its **own
per-seat partial observation**; produce per-seat trajectories for PPO. Current-policy
self-play (all seats = the live net). Replaces the single-learner / scripted-opponent
asymmetry of today's `HeatEnv`.
**Depends on:** A0 (rollout format), A1 (cheap bed).
**Deliverable:** multi-seat rollout collector yielding per-seat `(obs, action, logp,
value, reward)` streams; reuses `run_round_driver` advancing.
**Gate:** self-play games complete with correct per-seat credit assignment; throughput
acceptable in Tiny-Heat.

### A3 — Legal-action dot-product head
**Goal:** replace the 516-wide masked softmax with **legal-action scoring**: embed each
*currently legal* gear/card/react/etc. action's features, score by dot-product against
the state embedding, softmax over the legal set only. Reuses `action_codec` to enumerate
the legal set and to decode the chosen action back to an engine action.
**Depends on:** A0, A2.
**Deliverable:** action head + action-feature encoder; round-trips through the existing
codec; variable-length batching handled.
**Gate:** on Tiny-Heat the dot-product head matches or beats the old masked head's
learning curve at equal compute (removes the wide-head generalization tax).

### A4 — Hidden-information encoder
**Goal:** a structured encoder for the imperfect-info, multiplayer state: **own hand via
card embeddings + attention pooling**; a separate **public block** (positions, gears,
laps, seen/played cards, opponent hand *counts*). Track-distribution-general (whole-track
obs, building on the existing Option-A track block). Honest partial observability —
opponents' hands stay hidden.
**Depends on:** A2 (per-seat obs), A3 (state embedding feeds the action head).
**Deliverable:** encoder module + state embedding consumed by both heads; replaces the
flat-vector `features` path for the learner (building blocks reused).
**Gate:** ablation on Tiny-Heat — structured encoder ≥ flat encoder on sample efficiency;
no hidden-info leakage (verified by a leakage unit test).

### A5 — Anti-collapse self-play recipe
**Goal:** the Big 2 non-collapse stack: **entropy regularization** (floor), plus tune the
opponent policy — current-self vs. a small **recent-snapshot pool** (reuse `league.py`).
This is the specific antidote to our recurring PPO/BC→PPO collapse history.
**Depends on:** A2–A4.
**Deliverable:** entropy schedule + snapshot-pool opponent sampler + a **Stage-1
validation** check that must pass before any longer run.
**Gate:** Tiny-Heat self-play **climbs vs. previous-self** and **beats the weak
heuristic** without collapsing across several seeds — the core "does the method learn?"
go/no-go for Phase 0.

### A6 — First dense target (Direction C, shared)
**Goal:** raise value SNR with **one** dense signal in Phase 0 — score/margin-based value
(finishing position/margin or rounds-to-finish) replacing/augmenting the low-SNR ±1
target. Remaining Direction-C auxiliary heads were originally assigned to A8, but the
completion decision deferred them after sparse reward already demonstrated full-rules
learning and untouched-track generalization.
**Depends on:** A5 (a learning loop to improve).
**Deliverable:** margin value target wired into the PPO value loss.
**Gate:** measurable sample-efficiency gain vs. the ±1 target on Tiny-Heat. **End of
Phase 0.**

### A7 — Multi-seat / held-out evaluation harness
**Goal:** extend the existing Wilson-LB gate to score the policy **vs. strong/weak
heuristic across seat counts (2–6) and across held-out generated tracks**, with
confidence bounds — the instrument that defines "done" and guards every later run.
**Depends on:** A2 (multi-seat play), runs alongside A5/A6.
**Deliverable:** eval harness + a single skill-bar report (win-rate per seat count, per
track split, with Wilson-LB).
**Gate:** harness reproduces known baselines (strong beats weak by the expected margin);
held-out track split is genuinely disjoint from any training track.

### [A8 — Full-rules, small-scale, domain-randomized (Phase 1) — complete](A8-full-rules-phase1.html)
**Goal:** lift the proven method from Tiny-Heat to **full Heat rules, full generated-track
distribution, 2–6 seats, imperfect info**, still modest net/games. Train with **domain
randomization over tracks and seat counts from the start** (directly attacks the prior
"USA-only, 0% on generated tracks" failure). Auxiliary heads were explicitly deferred:
they were not needed to establish the Phase-1 learning/generalization claim, and adding
them during the final stabilization comparison would have confounded attribution.
**Depends on:** A0–A7 (the whole method, validated tiny).
**Deliverable:** a full-rules small-scale training run + an A7 skill-bar report.
**Gate (Phase-1, the big one):** **generalizes across held-out tracks and seat counts at
small scale** — beats the weak heuristic comfortably and shows real, non-trivial play vs.
strong. Also emit the **measured games/sec + sample-efficiency curve** that becomes the
**Phase-2 trigger** (the input to deciding whether Direction D is needed for the 48 h
run). **This is the completion of Direction A's mandate.**

**Implemented result:** PASS. G0002's one-use final test covered 40 untouched generated
tracks, four non-transitive rulers, and 2/4/6 seats. Its three aggregate placement rewards
were +0.0890, +0.1805, and +0.1390; the median was +0.1390 and weak-ruler medians were
positive at every seat count. S3 checkpoint averaging (G0004) improved validation skill
but missed its predeclared stabilization gate, so G0002 remained the frozen selection.

---

## Sequencing & dependencies

```
A0 substrate ─┬─> A2 self-play ─┬─> A3 action head ─┐
A1 tiny-heat ─┘                 └─> A4 encoder ──────┴─> A5 anti-collapse ─> A6 dense target ──┐  (Phase 0 done)
                                            A7 eval harness (parallel) ─────────────────────────┴─> A8 full-rules Phase 1
```

- **Phase 0 (cheap method proving):** A0 → A6, with A7 built in parallel. Exit = A5/A6
  gates: the method learns on Tiny-Heat.
- **Phase 1 (generalization at small scale):** A8. Exit = the Phase-1 gate above, which
  also produces the measurement that triggers (or skips) Direction D / Phase 2.

Every gate is a stop/redirect point. We do **not** start A8's full-rules run until
Tiny-Heat has shown the method learns and doesn't collapse, and we do **not** commit to
the expensive 48 h substrate work until A8's measurement says we must.

## What this plan deliberately defers

- **Direction D (vectorized engine / learned dynamics)** — only triggered by A8's
  measured throughput. Not on Direction A's critical path.
- **Direction B (search / PIMC / test-time MCTS)** — a ceiling enhancement applied *after*
  Direction A has a base net; out of scope here.
- **Most Direction-C aux heads** — one dense target in Phase 0 (A6), the rest in Phase 1
  (A8); kept minimal early so collapse vs. method is diagnosable.

## Key risks carried through

- **PPO self-play collapse** (our recurring failure) — A5 is the explicit mitigation;
  the A5 gate is where we find out if the Big 2 recipe holds for Heat.
- **Substrate lock-in** — A0 is a real decision, not a formality; the wrong call taxes
  A2–A4. Spike before committing.
- **Tiny-Heat fidelity** — if Tiny-Heat is *too* easy it won't predict full-rules
  behavior; A1 must keep the hard axes (stochastic, hidden info, multiplayer) intact even
  while shrinking track/deck.
- **"Is 48 h enough?"** stays unanswered until A8 — and that's by design; A8's curve is
  the first honest budget estimate.
