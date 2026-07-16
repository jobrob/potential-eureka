# Sprint C1 — Solo stochastic MCTS over the real engine (net-guided, no training)

> The first real foundation sprint, and the one that builds the **complete search
> core** plus the five seams (README §3). Build a *correct* single-agent
> stochastic MCTS that uses the real engine for transitions, in-tree chance nodes
> for our own draws, and a net for the prior + leaf value — and prove it MATCHES
> `LookaheadAgent` on the shared eval with a **fixed** net. No self-play, no
> training yet. If the search is not correct and competitive with a fixed net, no
> amount of training in C2/C3 will save it.

## Goal

A `MCTSAgent` (`BaseAgent`) that, given a policy+value net, runs the README-§4
algorithm — log-PUCT decision-node selection with Q-normalization + FPU, in-tree
DPW chance nodes over own draws, net policy prior + net value leaf with the S1
pre-spin floor — over the real `run_round_driver` engine, and on the held-out solo
field **matches** `LookaheadAgent` (rung 2). Correctness and parity first; learning
later.

## Scope

**In.**
- **The five seams (README §3), as small strategy objects** so C2/C3/forks compose:
  - `TransitionModel` (core impl = real engine: `clone` + `run_round_driver`),
  - `Node` with a `to_move` + kind (decision / chance) (solo: owner always learner),
  - `RootActionSelector` / `SearchPolicy` (core impl = log-PUCT + optional Dirichlet),
  - `LeafEvaluator` (core impl = net value head + pre-spin floor),
  - `ChanceExpansion` (core impl = DPW over engine reseeds).
  The MCTS core depends on these interfaces only — never on the concrete engine
  directly — which is what makes opponents / MuZero / Gumbel drop-ins later.
- **`MCTSAgent`** implementing `BaseAgent`, with the four MCTS phases adapted to
  HEAT-solo per README §4:
  - **Selection** by log-scaled PUCT (§4.1): `Q̂(s,a) + c_puct(s)·P(s,a)·√ΣN/(1+N)`,
    `c_puct(s)=c_init+log((ΣN+c_base+1)/c_base)`, with **Q̂ min-max normalized** by the
    tree's running `[Qmin,Qmax]` (mandatory — the `−rounds_remaining` value scale
    swamps `P` raw) and **FPU** for unvisited children. `P` is the net's masked
    policy read over the kept candidate set and renormalized.
  - **Decision nodes** = our GEAR / CARDS / REACT / SLIPSTREAM / DISCARD
    (`DecisionKind`) — **all** searched (the codec encodes every kind; nothing is
    delegated to a heuristic). The CARDS branch uses the S1 dedup-by-resulting-speed
    candidates (`_candidate_plans`, a *lossless* equivalence) so the 494-wide branch
    stays tractable.
  - **Chance nodes (own draws)** = in-tree (§4.3): a transition is a chance edge iff
    advancing it consumed engine RNG / changed our draw state (detected, not assumed);
    outcomes are sampled via the S1 deterministic per-(node, sample) `reseed`
    (`GameState.clone(reseed=…)`), fanned out by **double progressive widening**
    (`⌈C_pw·N^{α_pw}⌉`, hard cap `K`), and backed up as a **visit-weighted average**.
    This is expectimax-in-tree, not PIMC — it removes the strategy-fusion bias.
  - **Expansion** advances exactly one engine decision via `run_round_driver` (force
    the chosen action through `gen.send(...)`; in pure solo there is no non-searched
    decision to delegate), creating the next decision or chance node, evaluated once
    at creation.
  - **Evaluation (leaf)** = the net's value head on `encode_observation(state, pid,
    decision=None)` (emitting `−rounds_remaining`, higher-is-better — the A1/A2 sign,
    artifact-authoritative), **floored at `_pre_spin_progress` and not consulting V
    when our forced move spun** (`own_spins>0`) — the S1 discipline, depth-invariant.
  - **Backup** averages the (chance-averaged) leaf value up the path and updates the
    tree `[Qmin,Qmax]`; the acted move is the **most-visited** root child (AZ acting),
    with a config to act on `argmax Q̂` for deterministic eval.
- **Self-play exploration hooks present but OFF for C1's eval** (the mechanism is
  built here, exercised in C2/C3): root Dirichlet noise (`ε`, `α`) and the visit-count
  temperature schedule are config flags; C1's parity eval runs noise-off + greedy, so
  it is a deterministic function of `(state, seed)`.
- **Config** (`MCTSConfig` dataclass): `n_simulations`, DPW `(C_pw, α_pw, K)`,
  `c_init`, `c_base`, `fpu_reduction`, `leaf_mode`, the branch action set,
  `dirichlet_(eps,alpha)` + `temperature_schedule` (default off/greedy), and a `seed`
  (deterministic, like `LookaheadAgent._turn_seed` — never touch global RNG, so the
  determinism test holds). All defaults are C0's chosen constants.
- A `SearchProfile` (reuse the class) on the agent: clones/move, ms/move, always on.
- A **fixed net** to drive C1: a randomly-initialized `build_model` policy (the honest
  cold-start) **and/or** the S3 BC checkpoint as a sane warm prior, loaded the
  `MLAgent` way (checkpoint + `.meta.json` contract tripwire, `CODEC_VERSION` check).

**Out.**
- Any training / self-play target generation (that is C2).
- Opponent nodes / multiplayer determinization (deferred `Node` impl behind the seam).
- Gumbel/sampled-MCTS, virtual-loss leaf batching (deferred; only if C0 says the
  visit budget binds — the `RootActionSelector` seam makes Gumbel a later drop-in).
- A learned dynamics model (MuZero — deferred `TransitionModel` impl; we use the
  real engine).

## Deliverables

- `src/heat/agents/mcts_agent.py` — `MCTSAgent(BaseAgent)` + `MCTSConfig` + the five
  seam interfaces and their core implementations, reusing `SearchProfile`, the S1
  candidate prune, and the S1 reseed/clone scheme. Picklable factory
  (`mcts_agent_factory`) mirroring `lookahead_agent_factory` so it drops into the eval
  harness / league (pickle-by-path for the net, the `MLAgent`/A2 pattern).
- A thin **net adapter** (the `LeafEvaluator`/prior backing): `policy_prior(obs, mask)
  → np.ndarray` and `leaf_value(obs, mask) → float`, backed by a loaded `MaskablePPO`
  (`policy.get_distribution(obs, action_masks=mask)` for the prior; the critic head
  for the value). A small adapter so C2/C3 swap the checkpoint without touching search.
- `experiments/eval_search.py` extension (or a sibling `eval_mcts.py` importing its
  `_passes_and_spins_by_limit` / `_AgentAgg` / `_success_check` verbatim) adding
  `MCTSAgent` as a contender on the solo (primary) field, printing the rung-2 verdict
  vs `LookaheadAgent` + the heat-efficiency report.
- Unit tests in `tests/test_mcts_agent.py`:
  - search never returns an illegal move (re-validated against the handed legal set,
    like `LookaheadAgent.choose_cards`);
  - **determinism** under a fixed seed (byte-stable move for fixed state+seed) — the
    whole search, *including chance sampling*, is a pure function of `(state, seed)`;
  - `n_simulations=1` degenerates to a sensible prior-greedy move;
  - **chance correctness**: a chance node's backed-up value is the visit-weighted
    average of its sampled outcomes (pure function of `(state, seed)`); DPW respects
    its `(C_pw, α_pw, K)` widening rule (child count grows as specified);
  - **Q-normalization**: with the raw `−rounds_remaining` scale, asserting the
    normalized selection actually uses the prior (a regression guard that the swamp
    bug — raw Q dominating `P` — cannot silently return);
  - **leaf discipline**: a forced-spin line scores below a clean line at every
    `leaf_horizon`/sim count (V never consulted on a spun leaf);
  - **all-decision coverage**: REACT/SLIPSTREAM/DISCARD are searched (the tree
    branches on them), not delegated — assert the agent does not call a heuristic
    rollout policy for them.

## Success criteria (rung 2 — MATCH `LookaheadAgent`)

On the held-out generated solo field (`eval_search` `_HELDOUT_BASE`/`_TIGHT_PARAMS`),
with a **fixed** net:

- worst-case (max) spins/limit-1-pass **≤ `LookaheadAgent`** (and ≤ `HeuristicAgent`),
- rounds-to-finish **≤ `LookaheadAgent`**,
- 100% solo finish (necessary, never sufficient — report rounds alongside),
- ms/move + clones/move **reported** (the `SearchProfile`); the cost need not beat
  `LookaheadAgent`, it must be within the C0 budget.

Acceptance is the harness's printed verdict (mirroring `_success_check`), on disjoint
held-out seeds, not a single track.

## Risks & mitigations

- **Risk: the cold (random) net prior makes early search worse than `LookaheadAgent`,
  so rung 2 looks unreachable.** Mitigation: C1's bar is *MATCH with a fixed net* —
  use the S3 BC checkpoint as the fixed prior to demonstrate parity is achievable, and
  separately report the random-net number as the honest cold-start floor C2 will lift.
  Parity with *some* fixed net proves the search is sound; C2 proves training lifts it.
- **Risk: the in-tree chance node is averaged wrong / biased (the S1 "counted spins
  multiply-count" class of bug, now at a chance node).** Mitigation: reuse the *exact*
  S1 reseed + once-after-rollout spin accounting; unit-test the chance backup as a
  pure visit-weighted average of `(state, seed)` outcomes; assert the leaf value on a
  known forced-spin track equals the pre-spin floor.
- **Risk: Q-normalization is forgotten and the unbounded value scale degenerates the
  search (the prior never bites).** Mitigation: it is a first-class config + a
  dedicated regression test (above); C0 already measured that the requirement is real.
- **Risk: the 494-wide CARDS prior dilutes PUCT so search never commits.**
  Mitigation: branch only over the S1 dedup-by-speed candidates and renormalize the
  net prior over that kept set (lossless equivalence; prior is *read*, candidate set
  is *fixed*). Gumbel stays a deferred seam, pulled forward only if C0/C1 show the
  pruned width still binds.
- **Risk: DPW makes the tree too deep/wide to be affordable.** Mitigation: C0's
  `(C_pw, α_pw, K)` + `sim_budget`; profile is always on so a blow-up is visible.
- **Risk: depth re-introduces the reckless-recovery pathology (S1 lesson #2).**
  Mitigation: the leaf uses the pre-spin progress floor; unit-tested depth-invariant.

## Dependencies

- **C0** constants (sims, DPW, PUCT/FPU, Q-norm mode, Dirichlet/temperature sizing,
  leaf mode, branch set, budget).
- S1/S2 `LookaheadAgent` (the baseline + the reused `_candidate_plans` /
  `_leaf_score` / `_pre_spin_progress` / reseed scheme / `SearchProfile`).
- `heat.ml.model.build_model` + the frozen codec; `MLAgent` checkpoint-load path; the
  S3 BC checkpoint (optional, as the fixed warm prior).
- `experiments/eval_search.py` (extend) + `_runlog.run_main`.

## Effort

~1.5 sprints (up from the skeleton's 1 — the complete core + the five seams + the
chance-node correctness are the cost). The bulk is the stochastic-MCTS core and
getting the in-tree chance node, Q-normalization, and leaf discipline provably
correct; parity eval is cheap once the search is sound.
