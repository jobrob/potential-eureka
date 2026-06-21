# Option C — Search + learning that co-improve (solo AlphaZero-style)

> **Status:** design-only, foundations-first. This is the detailed expansion of
> **Option C** in
> [`../../solo-speed-whole-track-planning-options.md`](../../solo-speed-whole-track-planning-options.md)
> (§3 Option C) and the parent [`../README.md`](../README.md). Read both first for
> the shared scope, the value-function spine, the eval methodology, and the
> repo-asset reuse rules — they are not repeated here.
>
> **Input:** the S1/S2 `LookaheadAgent` outcomes and their three hard-won leaf
> lessons (`docs/sprint-search-imitation-design.md` §4, S1/S2 "Things … MUST
> inherit"), the BC/DAgger distribution-gap result (S3/S4), and the solo-driving
> decision trace (memory `project-solo-driving-decision-trace`).

---

## 1. What this option is (and the one sentence that constrains everything)

A policy+value net **guides** a tree search; solo time-trial self-play generates
**better-than-net** action/value targets by searching with the real engine; the
net trains toward those targets; the better net guides a better search; repeat.
It is the only option in the landscape that can *surpass* today's `LookaheadAgent`
rather than merely match or distil it, because the training signal is the search's
own improvement over the net, not a fixed teacher (cf. the imitation cap, parent
README hard-constraint #2).

Solo ⇒ **no opponent node.** The search tree has exactly two node types:

- **decision nodes** — our own GEAR / CARDS / REACT / SLIPSTREAM / DISCARD choice
  (the engine's `DecisionKind`s), with the net's policy head as the PUCT prior over
  the legal flat actions;
- **chance nodes** — our **own card draws** (the only stochasticity in solo),
  handled exactly as `LookaheadAgent` already handles them: by averaging over
  determinizations re-seeded from `GameState.clone(reseed=…)` (the
  `n_determinizations` machinery). This is the *expectimax / stochastic-MCTS*
  variant; we do **not** invent new chance machinery.

## 2. THE SCOPE BOUNDARY — minimal foundation vs. future (read this carefully)

This option is the most complex in the landscape. The explicit instruction (and the
repo's history of healthy-looking training collapses — 8C, S3) is: **build the
smallest thing that is genuinely AlphaZero-shaped and gate it hard, then extend.**
The line is drawn here, deliberately and non-negotiably:

### In the minimal foundation (the sprints below)

- **AlphaZero, NOT MuZero.** We OWN a perfect simulator (`GameState.clone()` +
  `heat.engine.driver.run_round_driver`). The foundation uses the **real engine**
  for every state transition in the tree. A *learned dynamics model* (MuZero) is a
  future extension, not the starting point — building a learned model when a perfect
  one is free would be strictly worse and add a large untested surface.
- **Single-agent (solo) MCTS-with-chance** over the real engine: decision nodes +
  own-draw chance nodes, PUCT selection, the net's policy as prior and value at the
  leaf. No opponents in the tree.
- **The net is the existing architecture, reused unchanged:** `HeatMLPExtractor` +
  `MaskableActorCriticPolicy`'s policy head (the 516-wide flat action prior) and its
  **separate critic/value head** (`share_features_extractor=False` is already the
  default — the value head is designed as "how's the race going"). The frozen codec
  (`encode_observation` / `action_codec` / `CODEC_VERSION = 2`) is reused verbatim so
  any net we train drops into `MLAgent` and the eval harness with no contract change.
- **The S1/S2 leaf discipline is reused, not reinvented:** the own-spin-dominates /
  pre-spin-progress-floor scoring is the source of the *value bootstrap target* at a
  terminal/early-stop leaf; the dedup-by-resulting-speed candidate prune
  (`LookaheadAgent._candidate_plans`) is the action-set the tree branches over; the
  own-draw chance handling is the `n_determinizations` reseed scheme.
- **A small-scale self-play → train → repeat loop**, gated each iteration on the
  shared eval, with a Wilson-LB / best-checkpoint guard so a collapse cannot be
  promoted.

### Deferred to "Future extensions" (specced one paragraph each in §6 — NOT sprints)

MuZero (learned dynamics), deeper/larger nets and longer campaigns, re-introducing
opponents into the tree (the multiplayer hidden-info node), Gumbel-AlphaZero /
sampled-MCTS for the wide CARDS branch, search-at-inference as the production agent,
and warm-starting C's value head from an Option-A/B V. **None of these are built in
the foundation.** Each is one paragraph in §6 so the foundation stays small and
genuinely finishable.

> **Why this line.** Each foundation sprint must be *independently valuable and
> verifiable on the shared eval*, not a slice of a big-bang. The cheapest genuine
> AlphaZero core is: (C1) a correct solo MCTS-with-chance over the engine that, with
> a *fixed* net/value, already MATCHES `LookaheadAgent`; (C2) the self-play data +
> training that lets one improvement iteration BEAT the net it learned from; (C3) the
> closed loop at small scale with the collapse guard. Everything ambitious rides on
> that core being correct first — so it is deferred until the core is green.

## 3. Ordered sprint list (the foundation — 3 sprints + 1 gate spike)

```
C0 (spike)  Feasibility + cost spike: is per-move MCTS affordable at our clone
            budget? Pick visit count / determinizations / horizon. (design-gate,
            ~0.5 sprint — may fold into C1 if cheap.)
   └─ C1   Solo MCTS-with-chance over the real engine, net-guided, NO training.
            Bar: MATCH LookaheadAgent on the shared eval with a fixed net/value.
        └─ C2  Self-play target generation + the policy/value trainer (one
                generation, off C1's search). Bar: the trained net BEATS the net
                it learned from (search-improvement signal is real).
            └─ C3  Close the loop: self-play → train → repeat at small scale, with
                    the Wilson-LB / best-checkpoint collapse guard. Bar: BEAT
                    LookaheadAgent on the shared eval, or stop honestly at the
                    plateau with the gap reported.
```

C0 is a half-sprint design gate (it can be folded into C1's first day if the spike
is cheap); C1–C3 are the three real foundation sprints. See each
`sprint-C<n>-*.md` for Goal / Scope / Deliverables / Success criteria / Risks /
Dependencies / effort.

## 4. Measurable success ladder (the shared eval — extend `experiments/eval_search.py`, don't replace)

All bars are on **held-out generated tracks** (the `_HELDOUT_BASE = 900_000` band,
`_TIGHT_PARAMS`, disjoint from any self-play seed band), **solo** field, with the
**worst-case (p90/max) spins/pass bucketed by corner speed-limit** discipline. Never
finish-rate alone (the S3/8C footgun: a crawl-and-spin policy "finishes" at 113
rounds). Three measured quantities per rung, exactly the ones the harness already
prints: **rounds-to-finish**, **worst-case spins/limit-1-pass**, and the cost report
**ms/move + clones/move** (the `SearchProfile`).

| Rung | Who | Bar | Sprint |
|---|---|---|---|
| 0 | `HeuristicAgent` | the honest floor (already measured) | — |
| 1 | `LookaheadAgent` (S1/S2) | **the bar to MATCH then BEAT** | — |
| **2** | C-MCTS, **fixed** net/value | **MATCH** rung 1: worst-case L1 spins/pass ≤ `LookaheadAgent`, rounds ≤ `LookaheadAgent`, at a *reported* (not necessarily smaller) ms/move | **C1** |
| **3** | C net after **one** train generation | the trained net (used as prior+value in the same search) **BEATS** the pre-training net on rung-2 metrics — the improvement signal is real | **C2** |
| **4** | C net after the **looped** self-play | **BEAT** rung 1 (`LookaheadAgent`): strictly lower worst-case L1 spins/pass AND rounds-to-finish on the held-out solo field | **C3** |

Rung 4 is the option's reason to exist. If C3 plateaus at rung 2/3 (matches but does
not beat), that is a legitimate honest stop — report the gap and fall back to the
A/B/E options on the spine, do **not** paper over it with finish-rate or a USA-only
number.

> **Heat-efficiency check (every rung that produces an agent):** also report heat
> spent vs. distance and cool-downs taken (parent README "shared eval"), to confirm
> the *budgeting* improved — the whole point of whole-track planning — and not merely
> corner survival.

## 5. Reuse map (concrete symbols — do not reinvent these)

| Need | Reuse | Where |
|---|---|---|
| Forward transitions in the tree | `GameState.clone(reseed=…)` + `run_round_driver` | `heat.models.game_state`, `heat.engine.driver` |
| Own-draw chance node | the deterministic per-(node, det) `reseed` averaging scheme | `LookaheadAgent._score_plan` / `_turn_seed` |
| Branch action set (tame the 494-wide CARDS) | dedup-by-resulting-speed candidates + the fast `_move_eval` prior ordering | `LookaheadAgent._candidate_plans` / `_candidate_prior` |
| Leaf / bootstrap value when search ends early | own-spin-dominates + **pre-spin progress floor** | `LookaheadAgent._leaf_score` / `_pre_spin_progress` |
| Policy prior + leaf value (the net) | `HeatMLPExtractor` + `MaskableActorCriticPolicy` actor head + **separate critic** | `heat.ml.model` (`build_model`, `PPOConfig`, `share_features_extractor=False`) |
| Obs / action / mask encoding (codec) | `encode_observation`, `legal_action_mask`, `encode_action_index`, `decode_action`, `CODEC_VERSION` | `heat.ml.features`, `heat.ml.action_codec`, `heat.ml.spaces` |
| Drop a trained net into eval / league | `MLAgent` (loads checkpoint + `.meta.json` contract tripwire) | `heat.agents.ml_agent`, `save_checkpoint` |
| Eval gate (spins-by-limit, p90/max, profiling) | extend `eval_search.py` (`_passes_and_spins_by_limit`, `_AgentAgg`, `_success_check`) | `experiments/eval_search.py` |
| Run reporting (START/DONE/FAILED) | wrap entry points in `run_main` | `experiments/_runlog.py` |
| Wilson-LB / best-checkpoint collapse guard | the S4 league-gate pattern | `experiments/eval_dagger.py`, `train_self_play` |

## 6. Future extensions (NOT now — one paragraph each, escalate-only)

- **MuZero (learned dynamics model).** Replace the real-engine transition in the tree
  with a learned `(state, action) → (next-latent, reward, value)` model, so search
  runs in latent space without cloning the engine. Only worth it if engine clone cost
  (profiled in C0; S2 measured ~16 µs/clone) becomes the throughput ceiling at the
  visit counts C3 needs — which the perfect, cheap simulator makes unlikely soon.
  Large untested surface; explicitly *not* the starting point precisely because we own
  a perfect simulator.
- **Re-introduce opponents (multiplayer search node).** Add an opponent decision node
  with hidden-info determinization. S2 already proved the determinization belief
  composes cleanly (`_determinize_opponents`: pool `hand+draw_pile`, keep discard
  fixed, multiset-preserving), so the spine is built for this — but solo isolates the
  driving skill and removes the hardest piece, so opponents stay out of the foundation.
- **Gumbel-AlphaZero / sampled-MCTS for the wide CARDS branch.** The policy prior over
  494 CARDS multisets is wide; Gumbel top-k action sampling (or progressive widening)
  gives a low-visit-count guarantee of improvement and would cut the per-move budget.
  Defer until C0/C1 show the naive PUCT visit budget is the bottleneck.
- **Deeper / larger nets and a full multi-hour GPU campaign.** The foundation uses the
  `small` profile at small scale on the RTX 4080 (torch cu126); the `large` profile and
  a long campaign are the obvious scale-up once C3's loop is proven to improve
  monotonically — the same "mechanism validated at smoke scale, needs a real campaign"
  posture S4 took.
- **Search at inference as the production agent.** C's net can drive a *shallow*
  inference-time MCTS (fast + strong), or run net-only (fastest). Choosing and tuning
  the inference-time search (this is essentially Option E / MPC on C's spine) is a
  separate decision deferred until C3's net exists and is worth deploying.
- **Warm-start C's value head from an Option-A/B V.** The parent README notes a good V
  from A or B is a ready-made warm start / sanity check for C. Wiring that hand-off is
  deferred — the foundation trains V from scratch to keep C self-contained and to
  measure C's own improvement signal cleanly.
