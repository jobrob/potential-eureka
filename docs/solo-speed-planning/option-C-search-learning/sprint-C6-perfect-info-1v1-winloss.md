# Sprint C6 — Perfect-information 1v1, win/loss-valued AlphaZero (does an adversary + a clean value fix the flat loop?)

> The **pivot** sprint. C0–C5 are built, green, and contract-checked, and they
> reached an *honest stop*: with the two plumbing collapses fixed — the policy
> target (C4's Gumbel completed-Q lifted π-entropy ~0.04 → ~0.46) and the value
> head (C5's warm-start + anchor + the data-starvation fix) — the **solo**
> self-play loop is *still flat* and does not beat `LookaheadAgent`. C5's
> diagnosis is that the gap is **structural, not plumbing**: solo time-trial
> supplies almost none of what AlphaZero needs. The README §7 "re-introduce
> opponents" paragraph and the C5 findings name four differences from Go/chess:
> (1) **no adversary → no self-play curriculum**; (2) a **thin search-over-net
> gap** (the corner skill is local); (3) a **low ceiling already reached by a
> cheap heuristic + 1-ply search**; (4) a **noisy, unbounded, frequently-missing
> value** (`−rounds_remaining`, which the still-weak searched agent drives to
> `MAX_ROUNDS`, so the MC target is starved). This sprint tests one hypothesis:
> **moving from solo to a 1v1 competitive game with a win/loss value directly
> fixes (1) and (4)** — the adversary restores the curriculum escalator, and
> win/loss is a clean, bounded ±1 signal *always defined at game end* — **and
> plausibly improves (2)/(3)** because head-to-head racing has real tactical depth
> (slipstreaming, blocking, heat-spend-to-pass, forcing an opponent hot into a
> corner) that deep search can exploit and a snap policy misses. It is the **cheap
> go/no-go** that gates the project's hardest deferred bet (hidden information).

## Resolved design decisions (2026-06-22)

Settled before C6 starts; the implementer follows these, not their alternatives.
Genuinely open sub-choices are in **Open questions** (the C4/C5 discipline:
surface the high-stakes forks, do not silently choose them).

1. **Perfect information ONLY.** Heat is really a hidden-information game
   (opponents' hands are secret → it is poker-like, and vanilla AlphaZero assumes
   perfect info; PIMC determinization is *biased* — the strategy-fusion problem
   Option-C deliberately avoided in solo). Hidden-info search is the project's
   hardest deferred seam **and** competitive self-play is this project's historical
   nemesis (the Sprint-5 / 8C / Sprint-B collapses). We do **not** bet on
   hidden-info *and* competitive-stability at once. C6 runs the search with **both
   hands visible** — a legitimate perfect-info two-player zero-sum AlphaZero
   setting — to isolate the single question: **does an adversarial win/loss signal
   produce a MOVING loop where solo could not?** Hidden-info determinization (the
   `Node` seam, with `LookaheadAgent._determinize_opponents` ready) is the *next*
   bet, pulled forward **only if C6 validates**.

2. **Pure 1v1 (two seats), not 4p.** 1v1 is the minimal adversarial setting: a
   clean zero-sum minimax (negate at the opponent's nodes), parity = 50%, no
   multi-way credit-assignment or kingmaking noise. 4p / league-of-4 is the
   scale-up *after* 1v1 validates, not the test.

3. **Reuse the C-machinery; activate the opponent `Node` seam.** The MCTS core,
   the Gumbel root selector, the chance nodes, the loop, the promotion guard, the
   heuristic opponents, and the seat-neutral eval are reused. The new code is the
   **two-player minimax backup at decision nodes**, the **win/loss `z`**, the
   **opponent node owner** (`to_move` alternates), and the **frozen-snapshot
   league**. This is the seam the whole Option-C design was built to make a
   contained addition rather than a rewrite (README §3, the `Node`/`to_move` row).

4. **Honest success = a MOVING loop, not a one-shot beat of the strong heuristic.**
   The prize is a seat-neutral, CI-gated win-rate that **climbs
   generation-over-generation** (vs a frozen reference, vs earlier selves) **plus a
   calibrating value head**. The compute-tiered plan (Tier 0 go/no-go → Tier 1
   larger run on a stated trigger) is a first-class deliverable, per the user's
   steer that *if compute is plausibly the limiter, a larger test is worth running
   if it could yield results*.

> **Umbrella-naming tension (flagged).** This sprint lives in the
> `solo-speed-planning/option-C-search-learning/` tree but **pivots from solo
> time-trial to competitive racing**. That is intentional — C6 reuses the
> Option-C *machinery* (the whole point of the seams) — but the directory name no
> longer describes the contents. We keep the file here for continuity with C0–C5
> and note the mismatch; if C6 validates and a hidden-info follow-on (C7) is
> taken, the competitive line likely deserves its own doc tree.

## Goal

Determine, cheaply and honestly, whether **an adversary + a clean win/loss value**
turns the flat Option-C loop into a **moving** one. Concretely: build a
perfect-information 1v1 two-player AlphaZero on the C-machinery, run the
compute-tiered experiment, and read three signals — (a) win-rate vs a frozen
reference (the strong heuristic) over generations, (b) win-rate vs earlier
generations of itself (the direct self-play improvement curve), (c) value-head
calibration (does predicted P(win) match realized outcomes). A moving loop is the
green light for the hidden-info bet; a flat loop *even at Tier 1* is the strong
signal the problem is deeper than value/curriculum and routes back to the A/B/E
spine.

## Scope

### In scope

- A **two-player perfect-information minimax** search: decision nodes alternate
  owners (learner vs opponent), value is zero-sum (negate at opponent nodes),
  backed up through the existing chance nodes (card draws stay the only
  randomness). The interior PUCT, the Gumbel root selector, the chance-node DPW,
  the codec, and the determinism scheme are reused unchanged.
- The **win/loss value target**: `z` becomes the realized game outcome (±1) from
  the perspective of the player to move at each logged state, backfilled at game
  end. The value head predicts expected outcome (a P(win) the search consumes).
- A **1v1 self-play loop** with a **frozen-snapshot league** (play vs the current
  net AND periodic frozen past snapshots) and the existing Wilson-LB /
  best-checkpoint promotion guard, gated on a **seat-neutral 1v1 win-rate**.
- The **compute-tiered experiment** (Tier 0 → Tier 1) with concrete numbers, an
  escalation trigger, and a stop trigger.

### Out of scope (deferred, behind the seams)

- **Hidden information / determinization.** The whole point of C6 is to isolate the
  adversary from the hidden-info hardness. `_determinize_opponents` is ready (README
  §6) but is **not** wired here. This is the C7 follow-on, taken only if C6 moves.
- **4p / league-of-4 competitive self-play.** 1v1 isolates the adversary; 4p is a
  scale-up after.
- **MuZero, deeper nets as the default, the multi-hour campaign.** Tier 1 may use
  the `large` net, but the long campaign is post-validation, as in every prior rung.
- **Search-at-inference tuning as the production agent** (Option E). The shipped C6
  artifact is the net + the in-search 1v1 agent for the gate, not a tuned deploy.

## Proposed design

### Overview

Activate the `Node`/`to_move` seam to make decision nodes two-player and the backup
zero-sum minimax; change `z` from the floored `−rounds_remaining` MC return to the
±1 game outcome; and wrap the loop with a frozen-snapshot league and a seat-neutral
1v1 win-rate gate. Everything else in the C-machinery is reused as-is.

### Detailed changes

#### 1. Two-player perfect-info search (`src/heat/agents/mcts_agent.py`, the `Node`/`to_move` seam)

Today the core is solo: `EngineTransitionModel` auto-resolves any non-learner
decision with a legal default (`_legal_default`), and `Node.to_move` is always the
learner (`mcts_agent.py` lines ~316–452, ~680–727). C6 makes the opponent a
**searched** decision node instead of an auto-resolved one:

- **`TwoPlayerTransitionModel`** (a new `TransitionModel` impl, or a `two_player`
  flag on `EngineTransitionModel`): when the engine yields a decision whose
  `decision.player_id` is the *opponent*, it is **not** auto-resolved — it becomes
  a DECISION node with `to_move = opponent_id`, branched on with the opponent's own
  candidate set (the same `_candidate_actions` dedup-by-speed prune) and the
  **same net's** policy prior read from the *opponent's* observation
  (`encode_observation(state, opponent_id, decision)`). Perfect information means
  the opponent's hand is visible, so this is a legitimate full-information branch —
  no determinization.

- **Zero-sum minimax backup.** Selection at a node always maximizes from the
  *mover's* perspective. The simplest correct construction that touches the least
  code: store every value in the **learner's** frame and **negate it when the node
  to move is the opponent**, so the existing `_select_edge` (argmax over
  `Q̂ + U`) and `_backup` keep working unchanged on a single scalar. Concretely,
  the leaf value `v` (learner's expected outcome, see §2) is backed up the path,
  and `_backup` flips the sign of the increment applied to an edge whose parent
  `to_move == opponent_id`. The running `[Qmin, Qmax]` min-max normalization
  (`_normalize_q`) and the FPU baseline stay exactly as in C1 — they already
  operate on a single normalized scalar, which is all minimax needs. (Equivalently,
  negate-at-each-ply "negamax"; storing in the learner frame is the smaller diff to
  the existing `Node.value` / `Edge.w` accounting.)

- **Chance nodes are untouched.** Card draws remain the only stochastic
  transition; the round-boundary chance edge, the DPW widening, and the
  visit-weighted average backup (`_select_outcome` / `_step_chance_child` /
  `_backup`, lines ~1302–1456, ~1585–1613) work identically — a chance node has no
  owner, so no negation. This is the one place the perfect-info two-player tree is
  *easier* than solo was hard: the expectimax-over-own-draws machinery already
  exists and composes.

- **Opponent identity in self-play and at the gate.** In self-play the opponent is
  the **same net** (or a frozen snapshot — §3). At leaves, the S1 own-spin floor
  (`_pre_spin_progress`) is **dropped** for the win/loss frame (it was a
  `−rounds_remaining`-scale device; see §2) — a spin's cost is now expressed
  entirely through whether it loses the race.

- **Acting / extraction.** `_extract_plan` (lines ~1624–1660) is unchanged in
  shape — most-visited root edge for the learner's own gear+cards. The opponent's
  searched plies live *inside* the tree (below the learner's root); the agent still
  only emits the learner's own action.

#### 2. The win/loss value target (`experiments/gen_selfplay.py` → a 1v1 generator)

The solo generator backfills `z = −floored_rounds_remaining` after the episode,
dropping `MAX_ROUNDS`-truncated rows (`gen_selfplay.py` lines ~424–589). C6
replaces the value backfill:

- **`z` = the realized game outcome**, from the perspective of the player to move
  at the logged state: `z = +1` if that player wins the 1v1, `−1` if it loses
  (`0` for the rare double-DNF / tie — see Open questions). Backfilled at game end;
  **every game finishes** (someone crosses first, or `MAX_ROUNDS` decides on
  progress — see Open questions on the tie rule), so the
  `MAX_ROUNDS`-drop-the-whole-episode machinery and the per-row NaN-z drop are
  **removed**. This is the structural win the pivot buys: the value target is
  *always defined*.
- **The pre-spin floor is removed from `z`** (it was the rounds-frame image of the
  leaf floor; in the win/loss frame a spin is "bad" exactly insofar as it loses).
- **Both seats are logged** when self-play is net-vs-net (double the rows per game,
  each from its own mover's perspective). When self-play is net-vs-frozen-snapshot,
  only the *current-net* seat's rows are kept as training targets (the snapshot is
  a fixed adversary, not a learner).
- **The policy target is unchanged in shape** — the Gumbel completed-Q
  distribution (`_completed_q_distribution`) or the visit-count distribution, over
  the kept candidate set, same `(obs, π, mask, z)` schema. Only `z`'s *meaning* and
  *backfill* change. The `--traj-greedy` knob and the Dirichlet/temperature
  exploration carry over.
- **Seed-band disjointness** (`_assert_seed_bands_disjoint`) is kept; the 1v1
  self-play band stays disjoint from the eval band (`900_000+`) and the precursor
  bands.

#### 3. The value loss and leaf evaluation (`experiments/train_az.py`, `NetAdapter`)

- **Loss.** Today `L = CE(π) + c_v·MSE(z, value head)` with `z = −rounds_remaining`
  on an unbounded scale (`train_az.py` line ~468). With `z ∈ {−1, +1}` the value
  head becomes a **P(win) predictor**. Two equally-standard options: keep **MSE on
  ±1** (AlphaZero's original `(z − v)²`, `v ∈ [−1, 1]` via a `tanh` head), or move
  to **BCE on a {0,1} label with a sigmoid head**. **Default: MSE on ±1 with a
  `tanh` value head** — it is AlphaZero's own choice, it is the smallest change to
  the existing critic (the head already emits a scalar; clamp/`tanh` it), and it
  keeps `predict_values` returning a `[−1, 1]` quantity the search consumes
  directly. (Open question if the calibration read argues for BCE.)
- **Leaf evaluation** (`NetAdapter.leaf_value`, `_evaluate_leaf`): the net value
  head now returns a learner-frame expected outcome in `[−1, 1]`; the
  `−rounds_remaining` sign comment and the own-spin floor branch are removed. A
  **terminal leaf** returns `+1` / `−1` directly (replacing `_terminal_value`'s
  `0.0`), which is the cleanest possible value signal — exact at the leaves the
  search can reach.
- **Calibration metric.** Replace the value-MAE-in-rounds gate signal (C5's
  first-class number) with **value-head calibration**: predicted P(win) bucketed vs
  realized win frequency (a reliability read), reported per generation. This is the
  C6 analogue of C5's "the single number to watch."

#### 4. Self-play stability — the frozen-snapshot league (the project's nemesis, designed in)

Competitive self-play collapsed repeatedly here (Sprints 5/8C/Sprint-B). The
win/loss value is a **moving target** and naive self-play-vs-current invites
rock-paper-scissors cycling. C6 makes anti-collapse first-class:

- **A small frozen-snapshot league (AlphaStar-style).** Each generation plays a mix
  of opponents: the **current net** (self-play), **N frozen past snapshots** (the
  best-of-generation checkpoints), and the **strong/weak heuristics** as fixed
  anchors. Training targets come only from the current-net seat. Frozen snapshots
  break the cycle: a net that beats only its immediate predecessor but regresses
  vs an older self is **caught** because the older self is still in the pool. The
  snapshot pool is the loop's existing per-generation checkpoints — no new storage
  mechanism, just retention + a sampling list.
- **The existing promotion guard, re-pointed at win-rate.** Reuse the Wilson-LB /
  best-checkpoint discipline (`az_loop.strictly_improves` + the `LoopGate` pattern,
  lines ~122–183), with the gate metric now the **seat-neutral 1v1 win-rate vs the
  frozen reference**, promoted only when the **Wilson lower bound** clears the
  incumbent (reuse `heat.simulation.stats.wilson_interval`). A candidate that wins
  more vs its predecessor but does not clear the Wilson-LB vs the frozen reference
  is **not** promoted — the anti-fake-pass rule, unchanged in spirit.
- **Seat-neutral evaluation.** 1v1 has a turn-order seat bias (front seats win more,
  all else equal — the documented finding behind `_seat_neutral_win_counts`). The
  existing harness rotates the focal agent through **all four** seats for 4p; C6
  needs the **2-seat analogue**: rotate the focal through **both** seats and pool
  the win indicator. This is a small generalization of `_seat_neutral_win_counts`
  (`eval_search.py` lines ~360–416) to a `num_seats` parameter — *not* a rewrite.

### Data-model / interface changes

- `MCTSConfig`: a `two_player: bool = False` (or `opponent_mode`) flag and an
  `opponent_id`; default off keeps C0–C5 byte-identical.
- `Node.to_move` becomes load-bearing (already a field, line ~696) — opponent
  decision nodes set it to `opponent_id`.
- `gen_selfplay`: `z` dtype/meaning changes to `{−1, 0, +1}`; the
  positive-`z`-is-a-bug assertion (line ~701) is **removed/inverted** (now `z` is
  bounded `[−1, 1]`); the MAX_ROUNDS row-drop is removed.
- `train_az`: a `value_mode ∈ {"rounds", "winloss"}` (default `"rounds"` to keep
  C5 callers working); `winloss` selects the `tanh` head + MSE-on-±1 + the
  calibration metric.
- A new `experiments/eval_1v1.py` (or an `eval_search` mode): the seat-neutral 1v1
  win-rate field + Wilson-LB, the three success reads. Mirrors `eval_dagger._league_gate`
  (lines ~51–90) with `num_seats=2` and parity `0.5`.
- A new `experiments/az_loop_1v1.py` (or a `--competitive` mode on `az_loop`): the
  frozen-snapshot league + the win-rate gate.

### Reuse map (explicit)

| Reused **unchanged** | Where |
|---|---|
| MCTS interior PUCT + Q-norm + FPU | `mcts_agent._select_edge`, `_normalize_q`, `_c_puct` |
| Gumbel root selector + completed-Q target | `mcts_agent._gumbel_root_search`, `gen_selfplay._completed_q_distribution` |
| Chance nodes (DPW, sampled draws, expectimax backup) | `mcts_agent._select_outcome`, `_step_chance_child`, `_backup` |
| Candidate prune (dedup-by-speed) | `mcts_agent._candidate_actions` |
| Determinism scheme (per-(node,sample) reseed, pickle-by-path) | `mcts_agent._chance_reseed`, `_turn_seed`, `NetAdapter.__getstate__` |
| The closed loop + recency-window aggregate | `az_loop.run_loop`, `_aggregate` |
| Wilson-LB / best-checkpoint promotion guard | `az_loop.strictly_improves`, `stats.wilson_interval` |
| Seat-neutral win-rate harness (→ generalize to 2 seats) | `eval_search._seat_neutral_win_counts` |
| Strong / weak heuristic anchors | `StrongHeuristicAgent`, `HeuristicAgent` |
| 1v1 league gate template (→ parity 0.5, num_seats 2) | `eval_dagger._league_gate` |

| **New** | What |
|---|---|
| Two-player minimax backup at decision nodes | sign-flip in `_backup` for opponent-owned edges |
| Opponent decision node owner | `TwoPlayerTransitionModel` does not auto-resolve the opponent; `to_move = opponent_id` |
| Win/loss `z` (±1, always defined, both-seats logging) | `gen_selfplay` value backfill |
| `tanh` value head + MSE-on-±1 + calibration metric | `train_az` `winloss` mode |
| Frozen-snapshot league + 1v1 win-rate gate | `az_loop_1v1` |
| Deferred (ready, NOT wired): hidden-info determinization | `LookaheadAgent._determinize_opponents` |

## The experiment design (compute-tiered) — a first-class deliverable

### Success metric (precise)

The honest prize is a **MOVING loop**, read three ways, all seat-neutral and
CI-gated on the held-out generated `_TIGHT_PARAMS` distribution:

- **(a) vs a frozen reference over time.** Seat-neutral 1v1 win-rate vs
  `StrongHeuristicAgent(strength=2)`, plotted generation-over-generation. The
  **real bar: beat the strong heuristic 1v1, seat-neutral, Wilson-LB > 50%
  (parity)**. (Win-rate vs the *weak* heuristic is a sanity floor.)
- **(b) vs earlier generations of itself.** Seat-neutral 1v1 win-rate of gen *k* vs
  gen *k−1* and vs gen *0* (the warm net). A win-rate **> 50% with Wilson-LB
  separation**, *climbing*, is the direct evidence the adversary is driving
  improvement — the single most important read, because it is exactly the
  curriculum-escalator signal solo could never produce.
- **(c) value-head calibration.** Predicted P(win) bucketed vs realized win
  frequency. A value head that *calibrates* (and tightens over generations) is the
  evidence the win/loss signal is learnable, even if (a)/(b) are still climbing.

Secondary reads (the spin/rounds discipline): the worst-case L1 spins/pass and
rounds-to-finish for the learner's seat, so we confirm a 1v1-winning net is not
winning by crawling. **Never** lead with finish-rate / a USA-only number / a point
estimate (the S3/8C footgun).

### Why this is plausibly compute-limited (the sparsity argument)

Win/loss is **clean but sparse**: one bit per several-hundred-decision game, vs the
solo MC return that labeled *every* logged state with a (noisy, unbounded) scalar.
That sparsity is *exactly* why AlphaZero needs volume — the gradient per game is
thin, so the loop needs many games and enough sims that the search actually
out-resolves the net before the value is informative. So C6 is a genuine candidate
for "compute is the limiter," which is why the tiered plan and the explicit Tier 1
trigger matter. (This also frames the **hybrid** open question below: a small dense
shaping term thickens the early gradient at the cost of a biased value.)

### Tier 0 — the cheap go/no-go (~1–2h on the RTX 4080)

The question: does the loop **move at all** — any CI-separated gen-over-gen
win-rate gain, or a clearly calibrating value head — at a cost of an hour or two.

| Knob | Tier 0 value | Rationale |
|---|---|---|
| Net profile | `small` (codec v3) | the C0–C5 smoke profile; fast clone-bound generation |
| Generations | 6 | enough to see a 3-point trend if one exists |
| Games / generation | 64 1v1 self-play games (≈ a few thousand both-seat rows) | sized so the val split survives and the win-rate gate has ~128 seat-rotated games |
| Sims / move | 16 (the C0 default) | the search must out-resolve the net; 16 is the validated solo budget |
| League pool | current net + 2 frozen snapshots + strong/weak anchors | minimal anti-cycle |
| Gate games | 24 held-out tracks × 2 seats = 48 seat-neutral games vs the strong heuristic; 24×2 vs gen *k−1* | Wilson-LB needs ~50 games for a usable bound |
| Epochs / gen | 30 | the C5 default |
| Value mode | **win/loss, MSE-on-±1, `tanh` head** (the default); the hybrid is the first escalation lever, not the Tier-0 default | isolate the clean signal first |
| Wall-clock (4080) | **~1–2h** | generation/gate are CPU-clone-bound (the C1–C5 cost model: ~10 ms/move, ~16 clones/move; two-player roughly doubles the tree work per move), trainer on CUDA; ~6 gens × (~10 min gen + ~2 min train + ~3 min gate) |

**Tier 0 verdict logic:**
- **MOVE** (escalate-worthy or done): a CI-separated gen-over-gen win-rate gain on
  read (b), OR Wilson-LB vs the strong heuristic crossing 50% by gen 6, OR a
  clearly tightening calibration curve with a positive (not yet CI-separated)
  win-rate trend.
- **FLAT-BUT-PROMISING** (→ Tier 1): a *positive but not CI-separated* win-rate
  trend, OR flat win-rate **but** the value head is calibrating and the games are
  visibly under-sampled (wide Wilson intervals). This is the under-powered case the
  sparsity argument predicts.
- **DEAD-FLAT** (→ STOP, do not spend Tier 1): win-rate flat AND value head **not**
  calibrating AND no gen-over-gen trend. The adversary is not helping; the problem
  is deeper than value/curriculum — fall back to the A/B/E spine.

### Tier 1 — the larger run (overnight, on an explicit trigger)

**Trigger:** Tier 0 lands in **FLAT-BUT-PROMISING** (a positive-but-not-separated
trend, or flat-but-calibrating-and-undersampled). **Do not run Tier 1 if Tier 0 is
DEAD-FLAT.**

| Knob | Tier 1 value | vs Tier 0 |
|---|---|---|
| Net profile | `large` (or `small` if Tier 0 was net-capacity-fine) | more capacity for the win/loss signal |
| Generations | 15–20 | a longer escalator to see a sustained climb |
| Games / generation | 256–512 1v1 self-play games | the volume the sparse signal needs |
| Sims / move | 32 (optionally 64) | widen the search-over-net gap |
| League pool | current + 4–6 frozen snapshots + anchors | a real anti-cycle ladder |
| Gate games | 64 tracks × 2 seats = 128 seat-neutral games per matchup | tight Wilson bounds |
| Value mode | win/loss; **the hybrid shaping is the first thing to try here** if Tier 0's value calibrated slowly | thicken the early gradient |
| Wall-clock (4080) | **~8–12h (overnight)** | volume × sims dominates; clone-bound |

**Expected Tier 1 signal:** a **sustained, CI-separated** gen-over-gen win-rate
climb on read (b) and a Wilson-LB crossing 50% vs the strong heuristic on read (a),
with a tightening calibration curve. That is the green light for C7 (hidden-info
determinization). A Tier 1 that is *still flat* with a calibrated value head is the
strong structural-stop signal: the adversary + clean value did not move it even at
volume, so hidden-info would not rescue it → A/B/E spine.

**Recommendation on pre-committing to Tier 1:** **do not pre-commit.** Run Tier 0
first; it is cheap and its three-way verdict (MOVE / FLAT-BUT-PROMISING /
DEAD-FLAT) is exactly the information needed to decide. Pre-committing to an
overnight run risks paying for the DEAD-FLAT case the sparsity argument cannot rule
out a priori. The one nuance: if Tier 0 is FLAT-BUT-PROMISING **and** the value head
is calibrating, Tier 1 is well-justified and should be launched without further
deliberation — that is the case the user's "compute is plausibly the limiter" steer
is about.

## Success criteria

- **Primary (the moving loop).** Either (a) seat-neutral 1v1 win-rate vs
  `StrongHeuristicAgent`, Wilson-LB > 50%, OR (b) a CI-separated gen-over-gen
  win-rate climb vs earlier selves — achieved at Tier 0 or Tier 1. Plus (c) a value
  head that calibrates and tightens.
- **Mechanism (the minimum honest result).** The two-player minimax search is
  correct (unit-pinned: negation at opponent plies, perfect-info opponent branch,
  determinism preserved), the win/loss `z` is always defined (no episode drops),
  and the frozen-snapshot league + seat-neutral gate run end-to-end. Even a flat
  loop with these green is a *reportable, attributable* result — the C4/C5 posture.
- **Secondary (anti-crawl).** The learner's worst-case L1 spins/pass and
  rounds-to-finish do not blow up as win-rate climbs.
- **Honest stop.** Tier 1 flat + calibrated value + no trend ⇒ the gap is deeper
  than value/curriculum, hidden-info would not rescue it ⇒ A/B/E spine.
- **Green light.** A moving perfect-info loop ⇒ take the next bet: C7, hidden-info
  determinization (the deferred `Node` seam, `_determinize_opponents` ready).

## Risks & mitigations

- **Self-play collapse / cycling (the nemesis).** Mitigated by the frozen-snapshot
  league (an older self in the pool catches a regression), the Wilson-LB
  best-checkpoint guard (a collapse is never promoted), and seat-neutral evaluation
  (so a seat-bias artifact is not read as skill). This is *the* first-class risk and
  is the reason the league is in-scope, not deferred.
- **Win/loss sparsity starves the gradient.** Mitigated by the tiered volume plan
  and the **hybrid-shaping** lever (Open question 1) held in reserve for Tier 1.
- **Perfect-info ≠ the real game.** Acknowledged and *deliberate*: C6 is an
  isolation experiment, not a deployable agent. Its result is informative precisely
  because it removes the hidden-info confound; the determinization follow-on is the
  next bet, not this one.
- **Two-player tree cost.** The opponent plies roughly double the tree depth per
  move (clone-bound). Mitigated by keeping sims at 16 for Tier 0; the cost is
  budgeted into the wall-clock estimates and is the reason Tier 1 is overnight.
- **Value-frame sign bugs (minimax negation).** The single most error-prone change.
  Mitigated by a unit test pinning that a forced-win line backs up to `+1` for the
  learner and `−1` for the opponent, and that a symmetric net-vs-itself game has a
  ~50% seat-neutral win-rate (the calibration sanity).

## Dependencies

- C0–C5 built and green (they are): the MCTS core, Gumbel selector, chance nodes,
  the loop, the promotion guard, the seat-neutral harness, the heuristic opponents.
- `heat.simulation.stats.wilson_interval` (exists), `eval_search._seat_neutral_win_counts`
  (exists, generalize to `num_seats`), `eval_dagger._league_gate` (exists, the 1v1
  template), `StrongHeuristicAgent` / `HeuristicAgent` (exist).
- RTX 4080 + torch cu126 (the C0–C5 box); generation/gate clone-bound on CPU,
  trainer on CUDA.

## Effort

- **Two-player search + minimax backup + opponent node owner:** the core change;
  moderate, behind the `Node` seam. Most risk is the negation sign discipline.
- **Win/loss `z` + `tanh`/MSE value mode + calibration metric:** small-to-moderate;
  mostly removing the rounds-frame machinery (floor, MAX_ROUNDS drop, the
  positive-z assertion) and adding the ±1 backfill + the reliability read.
- **Frozen-snapshot league + 1v1 seat-neutral gate + the loop wrapper:** moderate;
  reuses the harness and the guard, generalizes the seat rotation to 2 seats.
- **Tier 0 run + readout:** ~1–2h compute + analysis. **Tier 1 (if triggered):**
  overnight.

Total: comparable to C4/C5 in code surface, with the experiment (not the code) as
the deliverable — the cheap go/no-go before any hidden-information investment.

## Open questions (for the user to decide before implementation)

1. **Pure win/loss vs a hybrid (win/loss + small dense shaping).** Pure ±1 is clean
   but sparse; a small dense progress/anti-spin shaping term (the project's
   `shaping_weight` precedent) thickens the early gradient at the cost of a biased
   value and a re-introduced reward-design surface. **Recommendation: default pure
   win/loss for Tier 0** (isolate the clean signal — it is the whole point of the
   pivot), and hold the hybrid as the **first Tier-1 lever** if Tier 0's value head
   calibrates slowly. But this is a genuine fork (it changes what "the value" means)
   and is surfaced rather than silently chosen.
2. **Self-play vs current only, vs the frozen-snapshot league from gen 1.** The
   league is the anti-collapse insurance but adds opponent-pool bookkeeping and
   slows each generation (more matchups). **Recommendation: league from gen 1**
   (the nemesis history makes the insurance worth the cost up front), but a leaner
   "current + strong-heuristic anchor only" start is defensible if Tier 0 wall-clock
   is tight — surfaced for the user.
3. **The `MAX_ROUNDS` tie rule.** "Every game finishes" needs a definite 1v1
   outcome when both cars hit `MAX_ROUNDS` without crossing. Options: decide by race
   progress (further-along wins), or score it a draw (`z = 0`). **Recommendation:
   decide by progress** (a definite ±1 keeps the signal binary and avoids a
   z=0-dominated dataset if the early nets stall), with a draw fallback only on an
   exact progress tie — but flagged because it subtly shapes what the value learns.
4. **MSE-on-±1 (`tanh`) vs BCE (sigmoid) value head.** Recommendation MSE-on-±1
   (AlphaZero's own, smallest diff). Revisit only if the calibration read argues a
   probabilistic (BCE) head calibrates materially better.
