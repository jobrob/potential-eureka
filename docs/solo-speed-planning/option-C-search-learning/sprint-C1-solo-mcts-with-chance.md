# Sprint C1 — Solo MCTS-with-chance over the real engine (net-guided, no training)

> The first real foundation sprint. Build a *correct* single-agent
> MCTS-with-chance that uses the real engine for transitions and a net for the
> prior + leaf value — and prove it MATCHES `LookaheadAgent` on the shared eval
> with a **fixed** net. No self-play, no training yet. If the search is not correct
> and competitive with a fixed net, no amount of training in C2/C3 will save it.

## Goal

A `MCTSAgent` (`BaseAgent`) that, given a policy+value net, runs PUCT MCTS over
decision nodes + own-draw chance nodes on the real `run_round_driver` engine, and
on the held-out solo field **matches** `LookaheadAgent` (rung 2 of the success
ladder). Correctness and parity first; speed and learning later.

## Scope

**In.**
- `MCTSAgent` implementing `BaseAgent`, with the standard four MCTS phases adapted
  to HEAT-solo:
  - **Selection** by PUCT: `argmax_a Q(s,a) + c_puct · P(s,a) · √ΣN / (1+N(s,a))`,
    where `P` is the **net's masked policy** over the legal flat actions
    (`legal_action_mask` + the actor head's softmax), evaluated from
    `encode_observation(state, pid, decision)`.
  - **Decision nodes** = our GEAR / CARDS / REACT / SLIPSTREAM / DISCARD
    (`DecisionKind`). The branch action set reuses S1's dedup-by-resulting-speed
    candidates for CARDS (`LookaheadAgent._candidate_plans`) so the 494-wide branch
    stays tractable (the prior is read only over those kept actions).
  - **Chance nodes (own draws)** = expectation over our future draws, realized by the
    S1 scheme: clone with a deterministic per-(node, simulation, det) `reseed`
    (`GameState.clone(reseed=…)`) and average — **not** a new chance mechanism.
  - **Expansion** advances exactly one engine decision via `run_round_driver`
    (force the chosen action through `gen.send(...)`, delegate every *non-searched*
    decision — none in pure solo except our own non-branched kinds — exactly as
    `LookaheadAgent._answer_rollout_decision` does).
  - **Evaluation (leaf)** = the net's **value head** (`predict_values`-equivalent on
    the masked obs) at the leaf, OR a short rollout to the S1 **pre-spin leaf**
    (`_leaf_score` / `_pre_spin_progress`) — whichever C0 chose. Either way the
    own-spin-dominates / pre-spin-floor discipline is preserved so a reckless line is
    never preferred at depth (S1 lesson #2).
  - **Backup** averages the (chance-averaged) leaf value up the path; the chosen move
    is the **most-visited** child at the root (AlphaZero acting), with a config to
    act greedily on Q for deterministic eval.
- Config knobs: `n_simulations`, `n_determinizations` (default S1's 2), `c_puct`,
  `leaf_mode` (`"value_head"` / `"rollout"`), `leaf_horizon`, the branch action set,
  and a `seed` (deterministic, like `LookaheadAgent._turn_seed` — never touch global
  RNG, so the determinism unit test holds).
- A `SearchProfile` (reuse the class) on the agent: clones/move, ms/move, always on.
- A **fixed net** to drive C1: either a randomly-initialized `build_model` policy (the
  honest cold-start) **and/or** the S3 BC checkpoint as a sane warm prior, loaded the
  `MLAgent` way (checkpoint + `.meta.json` contract tripwire, `CODEC_VERSION` check).

**Out.**
- Any training / self-play / target generation (that is C2).
- Opponent nodes / multiplayer determinization (deferred extension).
- Gumbel/sampled-MCTS, virtual-loss leaf batching (deferred; only add if C0 says the
  visit budget binds).
- A learned dynamics model (MuZero — deferred, we use the real engine).

## Deliverables

- `src/heat/agents/mcts_agent.py` — `MCTSAgent(BaseAgent)` + a `MCTSConfig`
  dataclass, reusing `SearchProfile`, the S1 candidate prune, and the S1 reseed/clone
  scheme. Picklable factory (`mcts_agent_factory`) mirroring `lookahead_agent_factory`
  so it drops into the eval harness / league.
- A thin **net interface** the agent calls: `policy_prior(obs, mask) → np.ndarray`
  and `leaf_value(obs, mask) → float`, backed by a loaded `MaskablePPO`
  (`policy.get_distribution(obs, action_masks=mask)` for the prior; the critic head
  for the value). Keep this as a small adapter so C2/C3 can swap the checkpoint
  without touching the search.
- `experiments/eval_search.py` extension (or a sibling `eval_mcts.py` that imports
  its `_passes_and_spins_by_limit` / `_AgentAgg` / `_success_check` verbatim) adding
  `MCTSAgent` as a contender on the solo (primary) field, printing the rung-2 verdict
  vs `LookaheadAgent`.
- Unit tests in `tests/test_mcts_agent.py`: search never returns an illegal move
  (re-validated against the handed legal set, like `LookaheadAgent.choose_cards`);
  determinism under a fixed seed (byte-stable move for fixed state+seed);
  `n_simulations=1` degenerates to a sensible prior-greedy move; the chance-node
  average is a pure function of `(state, seed)` (no global-RNG leak); the leaf scoring
  preserves the own-spin/pre-spin discipline (a forced-spin line scores below a clean
  line at every `leaf_horizon`).

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
- **Risk: MCTS-with-chance is subtle — the chance node is averaged wrong and the value
  is biased (the S1 "counted spins per round multiply-counts" class of bug).**
  Mitigation: reuse the *exact* S1 reseed + once-after-rollout spin accounting; unit
  test the chance average as a pure function of `(state, seed)`; assert the leaf value
  on a known forced-spin track equals the pre-spin floor.
- **Risk: the 494-wide CARDS prior dilutes PUCT so the search never commits.**
  Mitigation: branch only over the S1 dedup-by-speed candidates and renormalize the
  net prior over that kept set (the prior is *read*, the candidate set is *fixed*) —
  this is the S1 prune reused, not new machinery. Gumbel sampling stays deferred.
- **Risk: depth re-introduces the reckless-recovery pathology (S1 lesson #2).**
  Mitigation: the leaf uses the pre-spin progress floor; unit-tested to be
  depth-invariant.

## Dependencies

- **C0** constants (visit count, determinizations, leaf mode, branch set, budget).
- S1/S2 `LookaheadAgent` (the baseline + the reused `_candidate_plans` / `_leaf_score`
  / `_pre_spin_progress` / reseed scheme / `SearchProfile`).
- `heat.ml.model.build_model` + the frozen codec; `MLAgent` checkpoint-load path; the
  S3 BC checkpoint (optional, as the fixed warm prior).
- `experiments/eval_search.py` (extend) + `_runlog.run_main`.

## Effort

~1 sprint. The bulk is the MCTS core + getting the chance node and leaf discipline
provably correct; parity eval is cheap once the search is sound.
