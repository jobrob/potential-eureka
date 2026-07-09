# Heat RL — transition plan toward multiplayer, imperfect-information, any-track in <48 h GPU

> **Status:** strategic planning (2026-06-23). High-level direction-setting, not an
> implementation spec. Synthesizes the C6 feasibility findings, the C7–C11 throughput
> work, and a survey of how hard games have actually been trained on consumer hardware
> (KataGo, Big 2, Stochastic MuZero, PIMC card-game agents). Each direction below ends
> with the high-level steps it requires; the final section sequences them.

## 0. The target (definition of done)

A single agent that plays **full multiplayer Heat** (up to 6 cars), under **imperfect
information** (hidden hands and deck order), on **any track from the generator**
(not a memorized track), reaching a clear, measured skill bar (beats the strong
heuristic by a defined margin across held-out tracks and seat counts) — and the whole
training run fits in **< 48 hours of GPU time on one consumer GPU (RTX 4080 class)**.

Every choice in this document is judged against that budget.

## 1. Where we are, honestly

- **The engine + ML core are healthy and fast.** C7 (parallel workers + killed the
  per-game model-reload tax) and C8/C9/C11 (lean inference, encode, and network forward)
  bought roughly an order of magnitude of generation throughput. `net-fwd` is no longer
  the bottleneck (~3% of wall); the dominant cost is now the **Python game engine +
  per-leaf plumbing**.
- **But the current approach cannot reach the target.** The loop we have is a
  **1v1, perfect-information, MCTS-AlphaZero** loop. The C6 findings showed it ran at
  ~0.015% of the games even a *trivial* game needs (~6,800× short); after our ~15× we
  are still ~450× short — *and that is for the easier 1v1 perfect-info case*. The actual
  target (multiplayer + imperfect-info + any-track) is strictly harder.
- **The research says the gap is not closed by "more AlphaZero, faster."** On consumer
  hardware, *perfect-information* games (Go 9×9/19×19, Hex, Othello) are the AlphaZero
  success stories. In **Heat's actual class — stochastic, hidden-information, large
  action space — the consumer-hardware successes used something else**: PPO self-play
  ([Big 2], trained to a working agent on a 6-core *laptop*), learned-dynamics MuZero
  (Backgammon/2048), or PIMC+MCTS (Skat, Hearts). That is the central strategic signal.

**Implication:** the project should pivot its *primary* learner away from from-scratch
MCTS-AlphaZero toward a **PPO self-play workhorse** (the Big 2 path), and keep
MCTS/search as a *targeted enhancement* (test-time search and/or a distillation teacher),
on top of a **cheaper environment substrate** and **denser learning signals**. The
directions below make that concrete and keep the alternatives on the table.

## 2. Why the Big 2 recipe fits Heat so well

[Big 2] is the closest published analog to Heat: stochastic, imperfect-information, large
and *variable* legal-action set, multiplayer — learned on hardware weaker than ours. Its
recipe maps onto our hardest requirements almost one-to-one:

| Heat requirement | Big 2 technique that addresses it |
|---|---|
| **Imperfect information** | The policy simply conditions on the *observation* (own hand + public state). No PIMC/determinization machinery — partial observability is handled "for free" by PPO. |
| **Multiplayer (N seats)** | A single shared policy acting per-seat on each seat's own observation. No zero-sum/negamax assumption to break (which AlphaZero's 2-player backup relies on). |
| **516-wide action space** | **Legal-action scoring by dot-product**: embed each *legal* action's features, score against the state embedding — rank only currently-legal actions instead of a fixed wide masked softmax. |
| **Stochastic, luck-heavy outcomes** | **Entropy regularization** (keeps the policy appropriately stochastic) + **current-policy self-play** (beat checkpoint/fixed-opponent curricula). |
| **Limited compute** | Plain PPO self-play was *more* compute-efficient than the value-based / search-heavy alternatives in their budget. |

This is why PPO-first is the recommended spine: the three things that make Heat hard for
AlphaZero (hidden info, >2 players, wide action head) are exactly the things PPO+Big2
handles with less machinery.

## 3. The directions

### Direction A (recommended primary) — PPO self-play workhorse, the Big 2 path

**Bet:** a multi-agent PPO self-play loop with the Big 2 architecture is the most
compute-efficient route to *multiplayer + imperfect-info + any-track*, and the most
likely to fit 48 h.

**High-level steps**
1. Stand up a **true N-agent self-play harness** over the existing `HeatEnv`: 2–6 shared-
   policy seats acting on per-seat partial observations, current-policy self-play.
2. **Redesign the action head** to legal-action dot-product scoring (embed each legal
   gear/card action, score vs state embedding) — removes the fixed 516-wide masked head
   and its generalization tax.
3. **Redesign the encoder** for hidden information: own hand via card embeddings +
   attention pooling; public state (positions, gears, laps, seen cards, opponent hand
   *counts*) as a separate block. Keep it track-distribution-general (whole-track obs).
4. Add **entropy regularization** and tune the self-play opponent policy (current vs a
   small recent-snapshot pool) for non-collapse.
5. Train **across the generated track distribution and across seat counts** from the
   start (domain randomization) — directly attacks the "USA-only, 0% on generated tracks"
   generalization failure.
6. Add the **dense auxiliary targets** from Direction C (shared) to raise value SNR.
7. Gate on the held-out-track / multi-seat skill bar.

**Pros:** handles all three hard axes natively; cheapest; proven on weaker hardware.
**Cons:** PPO self-play can collapse (we have seen BC→PPO collapse before — the Big 2
entropy + current-policy recipe is the specific antidote to try); no built-in lookahead.

### Direction B — cheap-and-dense AlphaZero, reserved for the ceiling / test-time search

**Bet:** search still gives stronger *per-game* play; KataGo's efficiency stack makes it
affordable, and "[MCTS only at test time]" lets us add it *on top of* a PPO-trained net
rather than paying for it during the whole from-scratch run.

**High-level steps**
1. Adopt **playout-cap randomization** (few sims on most moves, full sims only where
   training targets are taken) — we currently pay 16 sims on *every* move.
2. Add **forced playouts + policy-target pruning** (decouple exploration from the trained
   signal).
3. Handle imperfect information via **PIMC determinization** (sample consistent hidden
   hands, search each, aggregate) — only if/where search is used.
4. Use search as a **distillation teacher** for the PPO net and/or as **test-time
   search** for evaluation and final strength, not as the from-scratch generator.

**Pros:** higher skill ceiling; reuses our correct, tested MCTS/chance machinery.
**Cons:** PIMC + >2-player search is heavy; only worth it once Direction A has a base net.

### Direction C (shared enabler) — denser learning signal (KataGo auxiliary targets)

**Bet:** the C6 "low-SNR ±1 target" is a root cause of the flat curve; dense targets are
KataGo's biggest non-domain-specific efficiency lever and reduce the *number of games
needed* — the axis that matters most for a 48 h budget.

**High-level steps**
1. Replace/augment pure win/loss with a **score/margin-based value** (finishing
   position/margin, rounds-to-finish).
2. Add **auxiliary prediction heads**: final finishing order, per-corner spin outcome,
   and the **opponent's next move** (KataGo's auxiliary policy target).
3. Reuse the design's held-back **hybrid progress/anti-spin shaping** as a dense
   intermediate signal.

Shared by A and B; cheap to add; high expected leverage.

### Direction D (shared enabler) — a cheaper environment substrate

**Bet:** after C11 we are engine/plumbing-bound; 48 h at the required game counts needs
the transition itself to be orders cheaper. Two routes, decide by a spike:

**High-level steps**
1. **Spike a vectorized/batched engine** ([Pgx]-style: Heat rules as batched tensor ops
   so thousands of games step in parallel on the GPU) — measure games/sec vs today.
2. **Or spike a learned dynamics model** (Stochastic MuZero-style: cheap learned
   transitions replacing clone+replay; we already use the correct stochastic chance
   handling) — measure fidelity + speed.
3. Pick one based on the spike; fold the engine incremental-advance idea (the deferred
   C10) in only if it remains relevant.

**Pros:** the only thing that makes a real-scale run affordable. **Cons:** the largest
build; a rewrite, not a sprint — sequence it after Direction A proves the *method* works
at small scale.

## 4. Cross-cutting requirements (true regardless of direction)

- **Imperfect-information observation** — extend the encoder to the multiplayer hidden-
  info state (own hand visible, opponents' hands hidden, public derived features).
- **Any-track generalization** — train on the generator's track distribution, evaluate
  on held-out tracks; never single-track.
- **Multiplayer evaluation harness** — skill vs strong/weak heuristic across seat counts
  and tracks, with confidence bounds (extend the existing Wilson-LB gate).
- **Non-collapse discipline** — every self-play recipe needs an explicit anti-collapse
  guard (entropy floor, opponent pool, Stage-1 validation before any long run), given our
  history of collapses.

## 5. Sequencing toward 48 h (phased, with go/no-go gates)

The C6 "shrink the problem first" lever still governs: prove the *method* learns at a
feasible scale before paying for scale.

- **Phase 0 — Tiny-Heat proving ground (days, not GPU-weeks).** A deliberately small Heat
  variant (short track, small deck, 2–3 seats, imperfect info). Stand up **Direction A**
  (PPO self-play + Big 2 action head + entropy) **with one Direction-C dense target**.
  *Gate:* does the loop *learn* (beats weak heuristic, climbs vs prev-self) at small
  scale? If not, fix the method here where iteration is cheap.
- **Phase 1 — Full-rules, small-scale.** Full Heat rules, full track distribution, 2–6
  seats, imperfect info, still modest net/games. Add remaining Direction-C targets.
  *Gate:* generalizes across held-out tracks and seat counts at small scale.
- **Phase 2 — Make it affordable (Direction D).** Only now, if Phase 1's projected
  full-scale run exceeds 48 h, build the cheaper substrate (vectorized engine or learned
  dynamics). *Gate:* measured games/sec implies the target run fits the budget.
- **Phase 3 — The 48 h run + optional Direction B.** Launch the budgeted run; optionally
  add KataGo-cheap MCTS as **test-time search** / distillation to push final strength.
  *Gate:* the definition-of-done skill bar on held-out tracks within 48 h GPU.

Each gate is a stop/redirect point — we never launch the expensive phase until the cheap
phase has proven the method.

## 6. Key risks & open questions

- **PPO self-play collapse** — our recurring failure mode; the Big 2 entropy +
  current-policy recipe is the specific mitigation to validate in Phase 0.
- **Is 48 h actually enough?** Big 2 reached a working agent in ~10 h on a laptop CPU;
  Heat is harder and richer, but an RTX 4080 is far stronger. Phase 1's measured games/sec
  + sample-efficiency curve is what turns this from hope into a budget — that measurement
  is the real Phase-2 trigger.
- **Search vs no-search for final strength** — whether Direction A alone clears the skill
  bar, or whether Direction B's test-time search is required, is an empirical question to
  settle at Phase 1/3, not now.
- **Substrate choice (vectorized engine vs learned dynamics)** — decided by the Phase-2
  spike, not pre-committed.

## Sources / prior art
- [Big 2: Self-Play RL under Imperfect Information](https://arxiv.org/html/2605.28863) — the recommended recipe (PPO, legal-action scoring, entropy, current-policy self-play).
- [KataGo: Accelerating Self-Play Learning in Go](https://arxiv.org/pdf/1902.10565) — playout-cap randomization, forced playouts, auxiliary dense targets (~50× efficiency, non-domain-specific).
- [Stochastic MuZero (ICLR 2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1) — learned dynamics for stochastic games (Backgammon, 2048).
- [Pgx: hardware-accelerated parallel game simulators](https://arxiv.org/pdf/2303.17503) — vectorized engine substrate.
- [AlphaZe**: imperfect-information baselines](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full) and [Student of Games](https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/) — PIMC / unified perfect+imperfect handling.
- Internal: `docs/solo-speed-planning/option-C-search-learning/C6-findings-scale-and-feasibility.md` (the feasibility wall); C7–C11 throughput sprints.

[Big 2]: https://arxiv.org/html/2605.28863
[Pgx]: https://arxiv.org/pdf/2303.17503
[MCTS only at test time]: https://arxiv.org/pdf/2204.13307
