# Sprint C0 — Findings: net-guided solo stochastic-MCTS cost & constants

> **Type:** design-gate outcome (mirrors the S1/S2 outcome style).
> **Harness:** `experiments/spike_mcts_cost.py` (throwaway-but-committed).
> **Reproduce:** `python experiments/spike_mcts_cost.py` (full) or
> `python experiments/spike_mcts_cost.py --quick` (seconds-long smoke).
> Numbers below are from a full run on this box (Windows 11, RTX 4080, torch
> cu126; CPU is the binding device — see §A). All timings re-measured here, not
> trusted from S2.
>
> **Caveat (read first):** the net is freshly built with **random weights**. The
> forward *timings* are weight-independent (valid as-is). The Q-normalization
> experiment (§D) does not rely on the untrained value head — it synthesizes a
> realistic `−rounds_remaining` Q scale and exercises the PUCT arithmetic — so its
> conclusion stands. No checkpoint was produced (out of C0 scope).

---

## TL;DR — GO (at a reduced simulation budget)

**GO.** A net-guided solo stochastic-MCTS over the real engine is affordable for
self-play at small scale — **but only at `n_simulations ≈ 16`**, which lands at
**3.2× `LookaheadAgent`'s 4.08 ms/move** (inside the spec's 2–5× bar). 32 sims is
already 6.4× (outside the bar), 64 sims 12.8×. The binding constraint is **engine
clone+advance cost (~120 µs each), not the net forward (~0.33 ms total per leaf)**.
This shapes C1: keep the simulation budget modest, treat the clone as the cost
ceiling, and do **not** reach for CUDA/MuZero to fix a constraint they do not bind.

The per-move budget **C1 must hit: ≤ ~13 ms/move solo (≈256 clones/move)** at the
chosen constants — within 2–5× of the baseline and ~4,600 tracks/hour/core, which
is hours-not-days for the few-thousand-tracks/iteration C2/C3 target.

---

## The chosen constant set (every Scope item)

| Group | Constant | Value | Basis / status |
|---|---|---|---|
| **search budget** | `n_simulations` | **16** (C1 start); ceiling ~24 before >5× | §B/§F: 16 = 3.2×, 32 = 6.4× |
| | per-move budget | **≤ 13 ms / ≤ 256 clones per move** | §F (the bar C1 must hit) |
| **chance (DPW)** | `C_pw` | **1.0** | §C: with α=0.5 reaches the cap by N≈64 |
| | `α_pw` | **0.5** | §C: √N widening, standard stochastic-MCTS |
| | hard cap `K` | **8** | §C variance: value stable by fan-out 4–8 (std 1.26→0.42) |
| | reseed scheme | **per-(node,sample) explicit `reseed=`**, the S1 `_score_plan` scheme | §E: byte-stable; purity confirmed |
| **selection** | `c_init` | **1.25** | AZ/MuZero default; C1 to confirm vs the pruned width |
| | `c_base` | **19652** | AZ/MuZero default (log-PUCT term) |
| | `fpu_reduction` | **0.25** (C1 to confirm) | defensible default; §4.1 FPU on normalized Q |
| | Q-normalization | **min-max `[0,1]`, MANDATORY** | §D: measured, not asserted |
| **self-play explore** (sized here, used C2/C3) | Dirichlet `α` | **0.5** | sized to pruned width ~18 (§branch): `α≈10/width` heuristic |
| | Dirichlet `ε` | **0.25** | §4.5 default |
| | temperature | **`τ=1` for first `T_moves=10` plies, then `τ→0`** | §4.5; `T_moves` a defensible default, C2 to confirm |
| **leaf** | evaluator | **net value head (clean) + pre-spin floor (spun)** for the core | §B: net leaf is *cheaper* than a 3-round rollout AND is the §4.4 design |
| **branch** | candidate prune | **S1 dedup-by-resulting-speed** (`_candidate_plans`) | §branch: lossless, kept |
| | pruned width (recorded) | **mean 18.3, max 33** | §branch: does NOT yet force Gumbel (deferred seam stays deferred) |

Where a value is marked "C1 to confirm" it is a defensible default chosen now so
nothing is left blank; C1 may tune it against the real (trained) net.

---

## A — Building-block costs (measured)

**Clone + advance one round** (`GameState.clone(reseed=) + run_round_driver` to
round end — the `TransitionModel` core edge): **~120 µs/call, ~8,300 calls/s.**
This is ~2× the S2 "~16 µs/clone" figure because S2 timed a bare `clone()`; the
realistic tree edge is clone **plus** a full round advance (gear→cards→react→
slip→discard→replenish), which is the right unit (§B keeps the primitive and the
path length in the same unit — no double-count).

- **chance-edge fraction = 1.000** — every round-advance crosses a replenish (a
  draw fires). This is the §4.3 chance edge.
- **reshuffle fraction = 0.150** — only ~15% of advances actually consume parent
  RNG. Important nuance: a plain draw off an already-ordered deck consumes **no**
  RNG; only a *reshuffle* does. So detecting a chance edge by "RNG changed" would
  miss 85% of draws. **C1 must treat the round boundary itself as the chance edge,
  not the rarer reshuffle.** (The draw still changes the future hand even without
  a reshuffle; see §C.)

**Net forward** (`MaskableActorCriticPolicy`: masked policy logits + value head,
the exact MLAgent/A2 path):

| device | single forward | batched/item (batch=64) |
|---|---|---|
| **CPU** | **0.34 ms** (~2,940/s) | **0.016 ms** |
| **CUDA (RTX 4080, measured)** | **1.07 ms** (~930/s) | **0.017 ms** |

CUDA **was measured** on this box. Note the single-forward CUDA path is *slower*
than CPU (1.07 vs 0.34 ms) — kernel-launch overhead dominates a batch-1 forward of
this tiny MLP — and even batched, CUDA (0.017) ≈ CPU (0.016). **For C's leaf
evaluation the net is not the bottleneck on either device; CPU is the right core
default and CUDA buys nothing until/unless frontier batching at much larger batch
is in play.** (Virtual-loss frontier batching is the README mitigation — noted,
not built; it would only matter if the leaf forward bound, which it does not.)

## B — Realistic in-tree per-move cost (cost model)

Modeled as `sims × avg_path_len × (clone+round) + leaf_forwards`, with explicit,
audit­able assumptions: **`avg_path_len = 4.0` round-advances/sim** (rounds of
lookahead a typical selection descends before an unexpanded leaf — a solo lap is
~10–15 rounds, so 4 is a representative interior depth) and **`rollout_leaf_rounds
= 3`** for the rollout-leaf variant. Measured `plies/round = 4.0` (informational:
the per-round decision granularity, which drives *branching* not wall-clock).

| sims | net leaf ms/move | clones | rollout leaf ms/move | ×LookaheadAgent (net) |
|---|---|---|---|---|
| 16 | **13.0** | 64 | 13.6 | **3.2×** ✅ |
| 32 | 26.0 | 128 | 27.2 | 6.4× |
| 64 | 52.1 | 256 | 54.3 | 12.8× |
| 128 | 104.2 | 512 | 108.6 | 25.5× |

The **net leaf is slightly cheaper than the 3-round rollout leaf** (the rollout
pays 3 extra clones; the net pays one 0.34 ms forward) **and** is the §4.4 design
choice, so the core uses the net value head. Cost scales linearly in sims and is
dominated by the clone term (`sims × 4 × 0.12 ms`); the leaf forward is a few
percent. **This is why only ~16 sims fits the 2–5× bar.**

## C — DPW fan-out + chance-node value variance

**Fan-out** `min(K, ⌈C_pw·N^α_pw⌉)` at sample visit counts:

| (C_pw, α_pw) | N=8 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|
| (1.0, 0.5) | 3 | 4 | 6 | 8(cap) | 8 |
| (2.0, 0.5) | 6 | 8 | 8 | 8 | 8 |
| (1.0, 0.7) | 5 | 7 | 8 | 8 | 8 |

**Variance check** — backed-up chance-node value (mean race-progress over sampled
draw outcomes) vs fan-out, with spread across 5 independent reseed bases. Crucial
methodology point discovered here: a chance outcome's value is **identical across
draws after a one-round advance** (the round plays the *already-known* hand); it
only **diverges after a second round** plays the freshly-drawn hand. The check
therefore advances two rounds, and selects a *draw-sensitive* state (many
corner-forced states are draw-insensitive, which would make the check vacuous).

| fan-out | mean value | std across reseeds |
|---|---|---|
| 1 | 108.0 | 1.26 |
| 2 | 107.3 | 0.93 |
| 4 | 107.6 | 0.46 |
| 8 | 107.1 | 0.42 |
| 16 | 107.1 | 0.29 |

The backed-up value **stabilizes by fan-out 4–8** (std drops below ~0.5). So the
hard cap **K = 8** with `C_pw=1.0, α_pw=0.5` (which reaches 8 only at N≈64 visits)
gives a bounded, affordable, and *stable* chance fan-out. C1 input for C1: the
chance edge fires every round but only matters one round downstream — the tree
must expand the chance node's outcomes far enough to play the new hand.

## D — Q-normalization (settled with a number)

PUCT child selection compared with raw `Q` (the `−rounds_remaining` scale, ~−25…0)
vs min-max-normalized `Q∈[0,1]`, over 20 real states, with realistic synthesized
child-Q on the true scale and the net prior:

- **argmax-flip rate (raw vs norm): 0.10** — in 10% of selections the normalized
  Q picks a *different* child than raw Q, i.e. the exploration/prior term changed
  the decision once Q stopped swamping it.
- **mean selection entropy: raw 0.55 → normalized 0.86** — normalizing Q raises
  selection entropy ~55%, i.e. with raw Q the score is Q-dominated and the prior
  `P` is nearly inert; normalized, the prior competes.

**Conclusion: min-max Q-normalization is mandatory (§4.1 confirmed by measurement,
not assertion).** Raw `−rounds_remaining` swamps the prior exploration term.

## E — Reseed purity (determinism contract)

A fixed `(root state, search seed)` replaying the same per-(node,sample) reseed
sequence (the `LookaheadAgent._score_plan` scheme: `reseed = base + sample·31`,
explicit `reseed=`) yields a **byte-stable descent** (identical per-sample
position/lap/heat). Confirmed `True`. The default `clone(reseed=None)` is
*non-pure* (it advances the parent RNG by a draw), which is exactly why explicit
reseeds are mandatory — C1 must never use the default in-tree.

## Branch width (the Gumbel data point)

Pruned CARDS candidate width via the S1 dedup-by-resulting-speed prune
(`_candidate_plans`): **mean 18.3, max 33** distinct candidates per GEAR node. This
is the lossless branch (same-speed plays resolve identically). A width of ~18–33 is
comfortably handled by naive PUCT at 16 sims/move and **does NOT yet force
Gumbel/sampled-MCTS** — the deferred seam stays deferred. Recorded here so the
Gumbel go/no-go later is data-driven: revisit if a larger sim budget or a wider
effective branch (e.g. branching all DecisionKinds jointly) pushes visits-per-child
too low.

---

## GO / NO-GO recommendation

**GO**, with the binding decision being the simulation budget:

- At **16 sims (net leaf, CPU): 13 ms/move = 3.2× `LookaheadAgent` (4.08 ms)** —
  inside the spec's 2–5× criterion. Throughput ≈ **4,600 tracks/hour/core**, so a
  few-thousand-tracks self-play iteration is **hours on one core and embarrassingly
  parallel** — the hours-not-days bar is met.
- **32+ sims fall outside 2–5×** (32 = 6.4×). If C1 finds 16 sims too shallow to
  *match* `LookaheadAgent`'s effective 2-round lookahead, the honest options are
  (in order): (a) accept a higher inference ms/move — the spec explicitly allows
  this because the *trained* net runs net-only and the binding constraint is
  self-play volume, which parallelizes; (b) pull the **Gumbel seam forward**
  (Gumbel-AlphaZero gives a low-visit improvement guarantee, letting fewer sims go
  further on the width-18 branch); (c) only if neither holds, fall back to the
  spine's A/B options. **C0's recommendation is to proceed to C1 at 16 sims and
  reserve Gumbel as the first lever if the visit budget binds.**

**Why the cost is clone-bound, and what it means:** at 16 sims the per-move work is
64 clone+advances (~7.7 ms) + 16 net forwards (~5.4 ms); the clone dominates and
scales linearly in sims. **CUDA does not help** (the forward is already cheap and
batch-1 CUDA is slower than CPU here). **MuZero would help** (it removes the engine
clone), but the README is right to defer it: a perfect cheap simulator makes a
large untested learned-dynamics surface a poor trade until clone cost is *proven*
to be the throughput ceiling at scale — which at 4,600 tracks/hour/core it is not
yet. The lever that *would* help if needed is fewer/cheaper engine advances
(shallower lookahead, or a coarser advance step), not more compute.
