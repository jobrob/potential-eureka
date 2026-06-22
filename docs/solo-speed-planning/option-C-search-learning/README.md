# Option C — Search + learning that co-improve (solo AlphaZero, complete core)

> **Status:** design, complete-core. This is the detailed expansion of **Option C**
> in [`../../solo-speed-whole-track-planning-options.md`](../../solo-speed-whole-track-planning-options.md)
> (§3 Option C) and the parent [`../README.md`](../README.md). Read both first for
> the shared scope, the value-function spine, the eval methodology, and the
> repo-asset reuse rules — they are not repeated here.
>
> **Design stance (set 2026-06-22).** We build the **complete core algorithm**, not
> a skeleton: a genuine single-agent AlphaZero with **in-tree stochastic chance
> nodes** over the real engine, the full self-play exploration machinery
> (Dirichlet root noise, a temperature schedule, MuZero-style Q-normalization,
> log-scaled PUCT with FPU), search over **every** learner decision, and **MC**
> value targets. What is deferred (MuZero, opponents, Gumbel, the large campaign,
> inference-time search) is deferred *behind explicit code seams* — see §3 — so it
> drops in later without a rewrite. The verification ladder (C0→C1→C2→C3, each
> gated on the shared eval) is unchanged: completeness of the *algorithm* does not
> mean a big-bang of *scope*.
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

**Solo is a single-agent stochastic-shortest-path MDP.** There is no opponent, our
current hand is fully known at every decision, and the *only* randomness is our own
future **replenish draws** (and the deck reshuffle they trigger). That structure is
exactly the case where the *correct* search is **expectimax / in-tree stochastic
MCTS**, and where the determinization (PIMC) shortcut `LookaheadAgent` uses is
*biased* (strategy fusion: PIMC plans as if it will know its future draws). So C
uses real in-tree chance nodes — see §4. This is the single most important way C is
a complete algorithm and not a half measure.

The search tree therefore has exactly two node types:

- **decision nodes** — our own GEAR / CARDS / REACT / SLIPSTREAM / DISCARD choice
  (the engine's `DecisionKind`s), with the net's policy head as the PUCT prior over
  the legal flat actions. We branch on **all** of them (not just GEAR+CARDS — the
  codec already encodes every kind, so delegating REACT/SLIP/DISCARD to a heuristic
  would leave driving skill unsearched);
- **chance nodes** — our **own card draws**, realized *in the tree* at the
  transitions that cross a replenish (the round boundary), expanded by **sampled
  outcomes with double progressive widening** and backed up by a running average
  (§4.3). Not PIMC, not a new chance mechanism invented from scratch — the standard
  scalable stochastic-MCTS construction.

## 2. THE SCOPE BOUNDARY — complete core vs. future (read this carefully)

The explicit instruction (and the repo's history of healthy-looking training
collapses — 8C, S3) is: **build the complete AlphaZero core correctly and gate it
hard; defer the genuine forks behind seams, not behind prose.** The line:

### In the complete core (the sprints below)

- **AlphaZero, NOT MuZero.** We OWN a perfect simulator (`GameState.clone()` +
  `heat.engine.driver.run_round_driver`). The core uses the **real engine** for
  every transition in the tree. A *learned dynamics model* (MuZero) is a future
  extension behind the `TransitionModel` seam (§3), not the starting point —
  building a learned model when a perfect one is free would be strictly worse and
  add a large untested surface.
- **Single-agent (solo) stochastic MCTS** over the real engine: decision nodes for
  every learner `DecisionKind`, in-tree chance nodes at draw transitions, log-scaled
  PUCT selection with MuZero Q-normalization and FPU, the net's policy as prior and
  value head at the leaf, Dirichlet root noise + a temperature schedule during
  self-play. No opponents in the tree (the `PlayerNode` seam reserves the slot).
- **The net is the existing architecture, reused unchanged:** `HeatMLPExtractor` +
  `MaskableActorCriticPolicy`'s policy head (the 516-wide flat action prior) and its
  **separate critic/value head** (`share_features_extractor=False` is already the
  default — the value head is designed as "how's the race going"). The frozen codec
  (`encode_observation` / `action_codec` / `CODEC_VERSION = 2`) is reused verbatim so
  any net we train drops into `MLAgent` and the eval harness with no contract change.
- **The S1/S2 leaf discipline is reused, not reinvented:** the
  own-spin-dominates / pre-spin-progress-floor scoring is preserved at leaves and is
  the source of the **MC value target** (§4.5); the dedup-by-resulting-speed
  candidate prune (`LookaheadAgent._candidate_plans`) is a *lossless* equivalence
  (same-speed plays resolve identically) we keep to tame the wide CARDS branch.
- **A small-scale self-play → train → repeat loop**, gated each iteration on the
  shared eval, with a Wilson-LB / best-checkpoint guard so a collapse cannot be
  promoted.

### Deferred — behind the §3 seams, one paragraph each in §7 (NOT sprints)

MuZero (learned dynamics), re-introducing opponents (multiplayer hidden-info node),
**Gumbel-AlphaZero / sampled-MCTS** for the wide CARDS branch, deeper/larger nets and
the long GPU campaign, search-at-inference as the production agent, warm-starting C's
value head from an Option-A/B V, and KataGo-style forced-playout policy-target
pruning. **None are built in the core.** Each maps onto a seam so it is a drop-in
later, which is the whole point: we commit to the complete *core algorithm* and to
clean *interfaces* for the forks — not to the forks themselves.

> **Why this line.** Each core sprint must be *independently valuable and verifiable
> on the shared eval*. The cheapest genuine complete-AlphaZero core is: (C1) a
> correct solo stochastic MCTS over the engine that, with a *fixed* net/value,
> already MATCHES `LookaheadAgent`; (C2) the self-play data + training that lets one
> improvement iteration BEAT the net it learned from; (C3) the closed loop at small
> scale with the collapse guard. Everything ambitious rides on that core being
> correct first — so it is deferred until the core is green.

## 3. The seams (design now, so the deferred forks don't tie our hands)

Five interfaces are defined in C1 and depended on by everything downstream. They are
the concrete answer to "how much can we design now without tying our hands": we fix
the *core* implementation behind each seam and reserve the deferred fork as an
alternate implementation of the *same* interface.

| Seam | Core impl (built now) | Deferred impl (drops in, no rewrite) |
|---|---|---|
| **`TransitionModel`** — `legal_actions(state)`, `step(state, action, rng) → (next, is_chance, terminal)` | real engine: `clone` + `run_round_driver` | **MuZero** learned `(latent, action) → (latent, reward, value)` |
| **`Node` / `to_move`** — node owner + node kind (decision / chance) | solo: every decision node is the learner's | **opponent** decision node + hidden-info determinization (S2's `_determinize_opponents` composes) |
| **`RootActionSelector` / `SearchPolicy`** — prior→selection, target construction | log-PUCT + Dirichlet noise + visit-count target | **Gumbel-AlphaZero** sampled top-k selection + completed-Q target |
| **`LeafEvaluator`** — value at a leaf | net value head (clean) + pre-spin floor (spun) | **A/B V warm-start**; short-rollout-to-floor blend |
| **`ChanceExpansion`** — how chance outcomes are sampled/widened | double progressive widening over engine reseeds | full enumeration; **learned chance** (Stochastic MuZero) |

The MCTS core (C1) is written *against these interfaces only*. A seam that is a
single-line strategy object today is what makes opponents / MuZero / Gumbel a
contained addition rather than a rewrite later.

## 4. The complete algorithm (the spec every sprint builds to)

This section is the normative algorithm. C0 fixes its free constants; C1 implements
it with a fixed net; C2/C3 turn on self-play exploration and training.

### 4.1 Decision-node selection — log-scaled PUCT with Q-normalization + FPU

Select the child maximizing

```
argmax_a  Q̂(s,a)  +  c_puct(s) · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
c_puct(s) = c_init + log((Σ_b N(s,b) + c_base + 1) / c_base)        # AZ/MuZero log term
```

where:

- **`P(s,a)`** is the net's masked policy (`legal_action_mask` + the actor head's
  softmax), read **only over the kept candidate set** (the dedup-by-speed prune for
  CARDS) and renormalized over it — the prior is *read*, the candidate set is
  *fixed*, identical to the S1 prune.
- **`Q̂(s,a)`** is the child value **normalized to `[0,1]` by the tree's running
  `[Qmin, Qmax]`** (MuZero min-max normalization). This is mandatory, not optional:
  our value is *negated rounds-to-finish* — an unbounded scale (~−25…0) that, left
  raw, swamps the `P` exploration term and degenerates the search. Normalization is
  the fix and is a known-required step for non-`[-1,1]` rewards.
- **FPU (first-play urgency):** an unvisited child takes `Q̂ = Q̂(parent) − fpu_reduction`
  rather than `0` or `+∞`, so the prior — not an optimistic default — governs which
  unexplored action to try first.

### 4.2 Decision-node expansion

Advance exactly one engine decision via `run_round_driver` (force the chosen action
through `gen.send(...)`, delegate any *non-searched* decision — none in pure solo —
exactly as `LookaheadAgent._answer_rollout_decision`). The resulting state is either
the next decision node (deterministic intra-round edge) or, if the advance crossed a
replenish/draw, a **chance node** (§4.3). A node is expanded with the net's policy
prior over its legal (kept) actions and evaluated once at creation (§4.4).

### 4.3 Chance nodes — in-tree, sampled, double-progressive-widened

The only stochastic transition in solo is the **replenish draw** at the round
boundary (and the reshuffle it can trigger); all intra-round edges
(GEAR→CARDS→REACT→SLIP→DISCARD) consume no RNG and are deterministic. A transition is
a **chance edge** iff advancing across it consumed engine RNG / changed our draw
state (detected, not assumed — robust to engine changes). At a chance node:

- **outcomes are sampled**, not enumerated (the draw distribution over a reshuffling
  deck is too wide to enumerate). Each outcome is realized by cloning with a
  deterministic per-(node, sample) `reseed` (`GameState.clone(reseed=…)`, the S1
  scheme) and advancing — so the whole search is a pure function of `(root state,
  search seed)`, no global-RNG leak (the S1 determinism contract).
- **double progressive widening (DPW)** bounds the fan-out: a chance node expands a
  new sampled outcome only when `⌈C_pw · N(node)^α_pw⌉` exceeds its current child
  count; otherwise selection re-descends into an existing outcome chosen by visit
  proportion. This is the standard stochastic-MCTS control and the principled
  generalization of "average over N determinizations" (PIMC is the degenerate case of
  fixing the reseed for a whole root-to-leaf path; DPW samples per chance node, which
  is what removes the strategy-fusion bias).
- **backup** through a chance node is the **visit-weighted average** of its outcome
  values (running mean) — the expectation over our own draws, in the tree, correctly.

### 4.4 Leaf evaluation

A newly-expanded leaf is valued by the net's **value head**
(`policy.predict_values` on `encode_observation(state, pid, decision=None)`,
emitting `−rounds_remaining`, higher-is-better — the A1/A2 convention, sign
authoritative from the artifact). The S1 corner discipline is preserved at the leaf
exactly as A2 does it: if reaching this leaf required our own forced move to spin
(`own_spins>0`), the leaf is floored at `_pre_spin_progress` and the net value is
**not** consulted — so a reckless line is never preferred at depth, at any horizon,
before V is even trained. (`LeafEvaluator` seam: a short-rollout-to-floor blend and
the A/B-V warm-start are alternate implementations.)

### 4.5 Self-play exploration (C2/C3) — turned off for C1's parity eval

- **Root Dirichlet noise (self-play only):** `P(root,a) = (1−ε)·p_a + ε·η_a`,
  `η ~ Dir(α)`, default `ε=0.25`, `α` sized to the (pruned) branch width — set in C0.
- **Action selection:** self-play samples `a ∝ N(a)^{1/τ}` with a **temperature
  schedule** (`τ=1` for the first `T_moves` plies of an episode, then `τ→0`); eval is
  greedy (`argmax N`, with an `argmax Q̂` config). Disabling noise + greedy selection
  is exactly C1's deterministic eval mode.

### 4.6 Training targets (C2/C3)

- **Policy target `π`** = the root **visit distribution** `N(a)^{1/τ_t}/ΣN` over the
  masked/kept actions (never the argmax — that would be BC again, with BC's gap).
- **Value target `z`** = the **Monte-Carlo return**: the *realized* rounds-to-finish
  from that state to the end of the self-play episode, expressed as
  `−rounds_remaining`, with the **pre-spin floor** applied so a spin does not poison
  the target (the single most important correctness check in C2). MC return is chosen
  over a bootstrapped root value deliberately: it is unbiased and it makes C's `z`
  **definitionally identical to Option A/B's V**, so an A/B warm-start is a drop-in
  later. (n-step / TD(λ) / bootstrap are a `value_target` seam, not built now.)
- **Loss:** `L = CE(π, masked policy logits) + c_v · MSE(z, value head) + wd·‖θ‖²`,
  training **both** the actor trunk/head and the **critic** (unlike S3 BC, which left
  the critic uninitialized — here V is a first-class target). `c_v` tuned on a
  track-disjoint val split.

## 5. Measurable success ladder (the shared eval — extend `experiments/eval_search.py`, don't replace)

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

## 6. Reuse map (concrete symbols — do not reinvent these)

| Need | Reuse | Where |
|---|---|---|
| Forward transitions in the tree (`TransitionModel` core impl) | `GameState.clone(reseed=…)` + `run_round_driver` | `heat.models.game_state`, `heat.engine.driver` |
| Chance-node outcome sampling (`ChanceExpansion`) | the deterministic per-(node, sample) `reseed` scheme | `LookaheadAgent._score_plan` / `_turn_seed` |
| Branch action set (tame the 494-wide CARDS, losslessly) | dedup-by-resulting-speed candidates + the fast `_move_eval` prior ordering | `LookaheadAgent._candidate_plans` / `_candidate_prior` |
| Leaf floor when our forced move spun (`LeafEvaluator`) | own-spin-dominates + **pre-spin progress floor** | `LookaheadAgent._leaf_score` / `_pre_spin_progress` |
| Policy prior + leaf value (the net) | `HeatMLPExtractor` + `MaskableActorCriticPolicy` actor head + **separate critic** | `heat.ml.model` (`build_model`, `PPOConfig`, `share_features_extractor=False`) |
| Obs / action / mask encoding (codec) | `encode_observation`, `legal_action_mask`, `encode_action_index`, `decode_action`, `CODEC_VERSION` | `heat.ml.features`, `heat.ml.action_codec`, `heat.ml.spaces` |
| Drop a trained net into eval / league | `MLAgent` (loads checkpoint + `.meta.json` contract tripwire) | `heat.agents.ml_agent`, `save_checkpoint` |
| Eval gate (spins-by-limit, p90/max, profiling) | extend `eval_search.py` (`_passes_and_spins_by_limit`, `_AgentAgg`, `_success_check`) | `experiments/eval_search.py` |
| Run reporting (START/DONE/FAILED) | wrap entry points in `run_main` | `experiments/_runlog.py` |
| Wilson-LB / best-checkpoint collapse guard | the A3 bounded-policy-iteration guard: strict-improvement stop rule + best-checkpoint preservation (`GateMetric.strictly_improves_on`, `run_iteration`'s `best_model`/`best_gate`) | `experiments/value_iterate.py` |
| Opponent determinization (deferred `Node` impl) | the multiset-preserving re-deal already built for S2 | `LookaheadAgent._determinize_opponents` |

## 7. Future extensions (behind the §3 seams — NOT now, one paragraph each)

- **MuZero (learned dynamics model).** Alternate `TransitionModel`: replace the
  real-engine transition with a learned `(state, action) → (next-latent, reward,
  value)` so search runs in latent space without cloning. Only worth it if engine
  clone cost (profiled in C0; S2 measured ~16 µs/clone) becomes the throughput
  ceiling — which the perfect, cheap simulator makes unlikely soon. Large untested
  surface; explicitly *not* the start precisely because we own a perfect simulator.
- **Re-introduce opponents (multiplayer node).** Alternate `Node`/`to_move`: an
  opponent decision node with hidden-info determinization. S2 proved the
  determinization belief composes cleanly (`_determinize_opponents`: pool
  `hand+draw_pile`, keep discard fixed, multiset-preserving), so the spine is built
  for this — but solo isolates the driving skill and removes the hardest piece, so
  opponents stay out of the core.
- **Gumbel-AlphaZero / sampled-MCTS.** Alternate `RootActionSelector`: Gumbel top-k
  action sampling + completed-Q policy target gives a low-visit-count guarantee of
  improvement and would cut the per-move budget on the wide CARDS branch. Built
  *after* C0 measures whether the naive PUCT visit budget actually binds — committing
  before then is premature; the seam makes it a drop-in if it does.
- **Deeper / larger nets and a full multi-hour GPU campaign.** The core uses the
  `small` profile at small scale on the RTX 4080 (torch cu126); the `large` profile
  and a long campaign are the scale-up once C3's loop is proven to improve
  monotonically — the same "mechanism validated at smoke scale, needs a real
  campaign" posture S4 took.
- **Search at inference as the production agent (Option E / MPC on C's spine).** C's
  net can drive a *shallow* inference-time MCTS (fast + strong) or run net-only
  (fastest). Choosing/tuning that is deferred until C3's net exists and is worth
  deploying.
- **Warm-start C's value head from an Option-A/B V.** Alternate `LeafEvaluator`
  init: because C's `z` is the *same* `−rounds_remaining` quantity A/B produce, an
  A/B V is a ready-made warm start / sanity check. Deferred so the core trains V from
  scratch and measures C's own improvement signal cleanly.
- **KataGo forced-playout + policy-target pruning.** A refinement of the
  `RootActionSelector` target construction that sharpens the policy target at low
  visits. Deferred; note as a seam knob.
