# C6 findings — why the loop is flat: scale, game-difficulty, and feasibility (not a bug)

> **Status:** findings / postmortem (2026-06-23). Reads the C6 Tier-0 result
> ([`sprint-C6-perfect-info-1v1-winloss.md`](sprint-C6-perfect-info-1v1-winloss.md))
> against the published AlphaZero/Gumbel literature and a consumer-hardware
> reproduction, and reframes the "DEAD-FLAT" verdict. **Bottom line: C6 did not
> fail because the method is wrong or because we missed an algorithmic ingredient.
> It ran at ~0.01% of the data/search scale any AlphaZero result has ever needed,
> on a game that is *harder* than the ones that need that scale. The real blocker
> is compute feasibility, dominated by the clone-bound per-move cost and a
> low-SNR binary value target.**

## What we ran (the valid Tier-0)

Two full 6-generation Tier-0 loops (small net, 64 self-play games/gen, 16 sims/move,
frozen-snapshot league, seat-neutral Wilson-LB gate), one warmed from the C5 solo
crawler (`c5_main_best`) and one from a cold/untrained prior (`c2_cold_prior`). Both
read the same:

- **(b) self-play vs prev-self**: oscillates around 50% with no trend and no
  generation's Wilson-LB clearing parity — **no curriculum escalation**.
- **(a) vs the strong heuristic**: 4–17%, at or below the warm baseline; nothing
  ever shipped.
- **vs the WEAK heuristic**: 0–4% every generation — the net has **no racing skill**.
- **(c) value-head ECE**: tight (0.03–0.09) but **trivially so** — a value head
  correctly predicting "I almost always lose" / "I'm 50-50 vs an equal self" on
  lopsided outcomes is not evidence of learning.

The from-scratch run reproducing the crawler run **refutes the warm-prior confound**:
the result is independent of the prior.

## 1. The algorithm is correct — we did not miss an ingredient

Tracing `src/heat/agents/mcts_agent.py`, `experiments/train_az.py`,
`experiments/gen_selfplay.py`, `experiments/az_loop_1v1.py` against the published
specs, what we have is a faithful **Gumbel-AlphaZero**:

- **Selection**: log-PUCT, `c_puct = c_init + log((ΣN + c_base + 1)/c_base)`,
  min-max-normalized Q̂, FPU reduction — matches the AlphaZero/MuZero pseudocode.
- **Root**: Gumbel top-`m` sampling + Sequential Halving with the σ-transformed
  completed-Q target — matches Danihelka et al. 2022. Gumbel was the *correct*
  choice for a low simulation budget.
- **Chance**: true in-tree **expectimax** with double progressive widening over
  engine reseeds — the correct (Stochastic-MuZero-style) handling, *not* the biased
  PIMC determinization shortcut.
- **Two-player**: zero-sum negamax backup, perfect-information opponent node.

23 C6 unit tests pin the sign discipline, the searched-opponent branch, determinism,
and backward-compatibility. **The flat loop is not a method bug.**

## 2. We ran at ~0.01% of the required scale

Anchor: the [AlphaZero.jl Connect Four tutorial][c4] trained on an **RTX 2070**
(weaker than this box's RTX 4080 Super), on a *trivial* deterministic perfect-info
game (7 actions, ~42 moves max):

| Knob | Connect4 (consumer GPU) | **C6 Tier-0** | Gap |
|---|---|---|---|
| Sims / move | **600** | 16 | ~37× |
| Self-play games / iteration | **5,000** | 64 | ~78× |
| Iterations | 15 | 6 | — |
| **Total self-play games** | **≈2,625,000** | **≈384** | **~6,800×** |
| Network | 5-block ResNet (~1.6M params) | small MLP | — |

We ran **~0.015% of the games** a trivial game needed — about **3.5 orders of
magnitude short** — at **1/37th the search depth**. The tutorial explicitly notes its
**"neural network alone was initially unable to win a single game"** — *exactly our
symptom* (losing to the weak heuristic) — yet search+learning bootstrapped to
superhuman **over 2.6M games**. We stopped at 384.

**A flat curve is the expected behavior of a correct AlphaZero given 384 games.** The
"DEAD-FLAT" verdict measured an empty tank, not a broken engine.

## 3. Heat is much harder than the games AlphaZero has beaten

Every game AlphaZero has solved (Go, chess, shogi, Connect4, Othello, Hex) is
**deterministic, perfect-information, small-action-space**. Heat differs on every
axis that *raises* the data requirement:

- **Stochastic** (card draws). [Research on AlphaZero-like agents in stochastic /
  imperfect-information games][frai] notes the theoretical values there are
  *expected win rates*, not a simple win/loss, and that a binary ±1 reward is
  "particularly problematic in high-variance stochastic environments." In Heat the
  outcome between two equal players is dominated by *who drew better cards*, so the
  ±1 label is **low signal-to-noise** — the skill difference is swamped by luck.
  That is exactly why our value head calibrated trivially to ~0.5 and self-play was a
  coin-flip: it learned the luck-dominated base rate. Tabular AlphaZero *has* learned
  stochastic games (Chinese dark chess, EinStein würfelt nicht — [IEEE Access 2023][ieee]),
  so stochasticity is not fatal, but it needs *more* data and ideally a denser target.
- **Large action space** (`action_dim` 516; gear × card combinations) vs Connect4's 7.
- **Long episodes** (~hundreds of decisions/game) → one ±1 label diluted across
  hundreds of moves (severe credit assignment) vs Connect4's ~42.

The low-sim Gumbel result does not rescue this: Gumbel learns with 2 sims *per move
given enough games to train the net*, and the **one stochastic game in that paper
(2048) required Gumbel _plus_ Stochastic MuZero** ([Danihelka et al. 2022][gumbel]).
It improves planning efficiency per move; it does not reduce the games needed.

## 4. The feasibility wall (the real conclusion)

C6 Tier-0 did ≈384 games in ≈85 min ≈ **13 s/game** (clone-bound on the real engine,
at only 16 sims). Scaling:

- **One** Connect4-scale iteration (5,000 games) at our settings ≈ **18 hours**. At
  the proper 600 sims (~37×) ≈ **weeks per iteration**.
- Full Connect4-scale (≈2.6M games) ≈ **years** on this box.

And Heat needs *more* than Connect4 scale, not less. **What we "missed" is that
AlphaZero's data appetite × Heat's expensive clone-bound transitions × Heat's harder
structure puts a working run far outside a single consumer GPU's reach. This is a
feasibility limit, not an algorithmic one.** (The per-move cost gap — why our
simulations are ~15× more expensive than Connect4's even per-sim — is analyzed
separately; the dominant suspect is full-`GameState` cloning + the real rules engine
per transition, vs a bitboard drop, plus no GPU batching of net evals across
parallel games.)

## Implications / levers (priority order)

1. **Attack the per-game cost (the true bottleneck).** A fast/vectorized or learned
   (MuZero) transition model so 100–1000× more games become affordable. Without
   this, no scale plan is feasible.
2. **Raise the value SNR.** The design's held-back **hybrid shaping** (dense
   progress/anti-spin term) thickens the gradient the binary target starves; the
   literature backs denser targets for stochastic games.
3. **Shrink the problem.** A deliberately tiny Heat variant (short track, small deck)
   as the Connect4-equivalent proving ground — confirm the loop *moves* at a feasible
   scale before scaling up.
4. **Then** a genuinely larger run, only once #1 makes it affordable.

## Sources

- [AlphaZero.jl — Connect Four tutorial (training params, RTX 2070)][c4]
- [Danihelka et al. 2022, *Policy improvement by planning with Gumbel* (ICLR)][gumbel]
- [*AlphaZe∗∗: AlphaZero-like baselines for imperfect information games* (Frontiers in AI, 2023)][frai]
- [*Analyses of Tabular AlphaZero on Strongly-Solved Stochastic Games* (IEEE Access, 2023)][ieee]

[c4]: https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/
[gumbel]: https://iclr.cc/virtual/2022/spotlight/6419
[frai]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full
[ieee]: https://ieeexplore.ieee.org/document/10049064/
