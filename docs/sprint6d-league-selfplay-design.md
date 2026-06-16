# Design: Sprint 6D — Opponent League + Prioritized Fictitious Self-Play (PFSP)

> **Status:** planning / design only — no code. Part of Sprint 6 (see
> `docs/sprint6-roadmap.md`). **Depends on 6C** and is **NOT parallel with it**:
> 6D builds directly on 6C's stable, vectorized training loop and extends its
> opponent-sampling machinery. It is a **stretch / strengthening** sub-sprint — the
> base capstone is *not* blocked on 6D, but 6D feeds an even-stronger capstone.
> 6A/6B/6C remain mutually parallel; 6D serializes after 6C.

## Goal

Replace the **FIFO snapshot pool** with a proper **opponent league** governed by
**prioritized fictitious self-play (PFSP)**:

1. A **persistent, diverse pool** of past snapshots that retains *strong and
   diverse* opponents rather than FIFO-evicting them (today snapshots are
   FIFO-capped at 3, `training.py:356-358`).
2. A **priority sampler** that draws opponents the current policy *struggles
   against* (à la AlphaStar PFSP) instead of a fixed `snapshot_mix`
   (`training.py:226`, `training.py:392`).
3. **Per-opponent win-rate bookkeeping** so the priorities are data-driven and
   update as the learner improves.
4. Clean composition with 6C's **gated best-checkpoint** (§2.1) and **stochastic
   snapshots** (§2.2): the league *is* the opponent pool that 6C's stability
   features operate over.

Where 6C makes self-play **survivable** (a collapse can no longer destroy the
good model), 6D attacks self-play **fragility at the root** — a single, identical,
deterministic self-copy gives a degenerate signal; a prioritized league of varied
past selves gives a rich, non-stationary-but-bounded curriculum.

## Motivation

6C diagnosed and patched the acute collapse (value shock, entropy collapse,
deterministic-identical opponents, lost model — `sprint6-roadmap.md §2`). But the
*opponent supply* it leaves in place is still shallow:

- **FIFO eviction throws away strength and diversity.** `train_self_play`
  appends each frozen snapshot and pops the oldest once past `max_snapshots`
  (`training.py:355-358`), so a long Phase 2 can evict the *best* early snapshot
  and keep only recent (possibly weaker/collapsing) ones. There is no notion of
  "keep this one because it is strong / because it is different".
- **Fixed mix, uniform sampling.** `_mixed_opponent_pool` fills `n_snap` seats by
  cycling snapshots in index order with a fixed `snapshot_mix`
  (`training.py:392-397`) — every snapshot is equally likely regardless of whether
  the learner already crushes it or still loses to it. That wastes rollout budget
  on already-beaten opponents and under-trains against the hard ones.
- **No memory of relative strength.** Nothing tracks how the current policy fares
  against each pool member, so the curriculum cannot adapt.

PFSP fixes all three: a retention policy that preserves a strong/diverse frontier,
and a sampler weighted toward opponents that are *informative right now* (close
games or current losses), with per-opponent win-rate stats driving the weights.
This is the same paradigm AlphaStar used to stabilize and strengthen multi-agent
self-play; here it is scaled down to a single-machine `MaskablePPO` loop.

## Files to create / modify

| File | Action | What |
|---|---|---|
| `src/heat/ml/league.py` | **new** | The league: a persistent pool of snapshot entries (path + metadata + per-opponent win-rate stats), a **retention policy** (keep strong/diverse, not FIFO), and a **PFSP sampler** that returns opponent specs weighted by priority. |
| `src/heat/ml/training.py` | **modify** | In Phase 2, use the league in place of `_mixed_opponent_pool` (`training.py:383-399`): add each new snapshot to the league, let the league pick opponent seats, and record game results back into the league. Reuse `FrozenSnapshotAgent` (`training.py:110`, pickles by path) for the chosen seats. Anchors: `train_self_play` Phase-2 loop (`training.py:347-374`), snapshot save (`training.py:349-355`), FIFO cap (`training.py:356-358`), `_mixed_opponent_pool` call (`training.py:361-363`). |
| `src/heat/ml/training.py` (`CurriculumConfig`) | **modify** | Add league knobs (pool capacity, retention mode, PFSP weighting + exponent, min/max sample probability) without disturbing the existing fields (`training.py:222-231`). |
| `tests/test_league.py` | **new** | Retention determinism; PFSP weight correctness (weights ≥ 0, sum to 1); win-rate bookkeeping. |
| `tests/test_ml_training_smoke.py` | **modify** | `slow`: league self-play does **not** collapse below the 6C gated baseline. |

> No change to `models/`, the obs/action codec, or `MaskablePPO`. 6D is an
> **opponent-curriculum** change layered on the 6C loop.

## Design

### League membership & retention (keep strong/diverse, not FIFO)

A `League` holds entries, each describing one frozen snapshot by **path** (so it
stays small and picklable, exactly like `FrozenSnapshotAgent`'s
path-only spec, `training.py:119-121`, `training.py:142-146`):

```python
# ml/league.py (sketch)
@dataclass
class LeagueEntry:
    path: str                      # frozen checkpoint path (FrozenSnapshotAgent loads this)
    snapshot_index: int            # creation order
    gate_score: float | None       # 6C eval-gate score at creation (strength proxy, §2.1)
    games: int = 0                 # games the current learner has played vs this entry
    wins_vs_learner: int = 0       # entry's wins against the current learner
    # win_rate_vs_learner = wins_vs_learner / games  (the PFSP signal)

class League:
    def add(self, entry: LeagueEntry) -> None: ...
    def retain(self) -> None:                 # apply the retention policy when over capacity
    def sample(self, k: int, rng) -> list[str]:   # k opponent paths by PFSP priority
    def record_result(self, path: str, learner_won: bool) -> None:  # update bookkeeping
```

**Retention policy** (replaces the FIFO `pop(0)`, `training.py:357`): when the
pool exceeds capacity, evict by a **keep-strong-and-diverse** rule rather than by
age. Concretely, score each entry by a combination of (a) **strength** — its 6C
`gate_score` and/or its win-rate vs the current learner — and (b) **diversity** —
spread snapshots across the training timeline / strength bands so the pool is not
all near-identical recent copies. Evict the lowest-value entry. **Determinism:**
all tie-breaks use `snapshot_index`, and any randomness uses a seeded RNG so the
retained set is reproducible (test gate).

> Always retain the **current best** snapshot (the 6C `best_path` policy, §2.1) as
> a league anchor so the learner is always measured against the strongest self.

### PFSP sampling (priority, not fixed mix)

Each Phase-2 env build asks the league for `k = n_snap` opponent seats (the same
seat count `_mixed_opponent_pool` computes, `training.py:391-392`). Instead of
cycling in index order, the league assigns each entry a **priority weight** and
samples without/with replacement by those weights:

```
p_i  ∝  f( learner_win_rate_vs_i )          # the PFSP weighting fn
w_i  =  clamp(p_i, p_min, p_max) ;  Σ w_i = 1  (normalized)
```

The weighting function `f` follows AlphaStar PFSP: bias toward opponents the
learner is **not** reliably beating. Two standard variants, both supported via a
`CurriculumConfig` switch:

- **"hard"** — `f(wr) = (1 - wr)^p` (focus on current losses; good for pushing a
  plateaued policy).
- **"even"/"variance"** — `f(wr) = wr·(1 - wr)` (focus on *close* matchups; good
  for stable, informative gradients — closest in spirit to fixing the Sprint-5
  "uniform −1, no gradient" failure, `sprint6-roadmap.md §2 (b)`).

`win_rate_vs_i` comes from the **per-opponent bookkeeping** updated during
rollouts. Entries with too few games default to a neutral prior (e.g. treated as
0.5) so a brand-new snapshot is sampled enough to estimate it.

### Win-rate bookkeeping during rollouts

After each completed game in Phase 2, the learner's result against each
league-member it faced is recorded via `League.record_result(path, learner_won)`,
updating `games`/`wins_vs_learner` for that entry. Under 6C's vectorized envs the
opponents are `FrozenSnapshotAgent` instances built **per worker** from paths the
league chose, so results must be attributed back to the originating path — the
env factory carries the chosen paths, and outcomes are aggregated into the league
(running in the main process) between `learn` chunks. This keeps the league's
state single-source-of-truth in the training process (consistent with the
process-global pattern 6C already uses for shaping, `model.py:91-102`).

### Composition with 6C

6D **slots into** the 6C Phase-2 loop; it does not replace 6C's stability work:

- **Stochastic snapshots (6C §2.2):** league-sampled seats are still built as
  `FrozenSnapshotAgent(path, deterministic=False)` so opponents have variance.
- **Gated best-checkpoint (6C §2.1):** unchanged — the gate still decides what is
  saved to `best_path`; the league additionally *retains* the best as an anchor
  opponent. The gate is the safety net; the league is the curriculum.
- **Opponent-mix ramp (6C §2.4):** the league supplies the *snapshot* seats; the
  remaining seats are still scripted / strong-heuristic (6B), and the ramp still
  governs how many seats are snapshots vs scripted early in Phase 2.
- **Return normalization (6C §2.6):** league snapshots that were trained with
  normalization load their own stats the same way `MLAgent` does — no change
  needed in 6D beyond reusing that loader.

So 6D's surface area is: **add `League`, route Phase-2 opponent selection and
result recording through it, and add `CurriculumConfig` knobs.** Everything else
6C established stays in force.

## Build sequence

1. **`League` data model** (`LeagueEntry`, `League`) with `add` + `record_result`
   + per-opponent win-rate; unit-tested in isolation (no training).
2. **Retention policy** (`retain`) — keep-strong-and-diverse, deterministic
   tie-breaks; test against the FIFO baseline behavior it replaces.
3. **PFSP sampler** (`sample`) — weighting fn variants, clamping, normalization;
   test weights sum to 1 and respond correctly to win-rate inputs.
4. **`CurriculumConfig`** league knobs (`training.py:222-231`).
5. **Wire into `train_self_play`** Phase-2 loop: add snapshot to league, sample
   seats via league, record results between chunks (replacing
   `_mixed_opponent_pool` for the snapshot seats, `training.py:361-363`).
6. **`slow` smoke**: league self-play does not collapse below the 6C gated
   baseline.
7. Full suite: `PYTHONPATH=src python -m pytest tests/ -q` (with `slow`
   deselected by default, `pyproject.toml:24-25`).

## Test gates

### `test_league.py` (fast, default-run)

- **Retention is deterministic & seeded:** building the same league (same adds,
  same seed) and calling `retain` twice yields the same retained set; the policy
  keeps the highest-value entries (a constructed case where the FIFO order and the
  value order disagree proves it is *not* FIFO).
- **Best anchor retained:** the current-best entry is never evicted by `retain`.
- **PFSP weights correct:** for hand-set win-rates, `sample`'s weights are all
  `>= 0`, **sum to 1**, respect `p_min`/`p_max` clamping, and rank correctly —
  a 20%-win-rate opponent gets a higher "hard" weight than an 80% one; the
  "even/variance" variant peaks near 0.5.
- **Win-rate bookkeeping:** `record_result` updates `games`/`wins_vs_learner` and
  the derived win-rate; a never-played entry uses the neutral prior.

### `test_ml_training_smoke.py` (`slow`, deselected by default)

- **League self-play does NOT collapse:** a short Phase-1→Phase-2 run using the
  league asserts the **gated best-checkpoint eval score does not regress below the
  6C gated baseline** (i.e. league + PFSP is at least as safe as 6C alone, never
  worse). Kept fast/small — "best ≥ 6C baseline", not "learns to superhuman".

## Risks + de-risking

| Risk | De-risking |
|---|---|
| PFSP destabilizes training (over-focusing on a single brutal opponent → all-loss, the Sprint-5 failure again). | Clamp sample probabilities (`p_min`/`p_max`) so no opponent dominates; default to the **"even/variance"** weighting (close games, informative gradient) rather than pure "hard"; the 6C best-checkpoint gate (§2.1) is still the safety net. |
| Retention policy silently degenerates to FIFO or to all-identical recent copies. | The diversity term + the deterministic retention test (FIFO vs value disagreement) pin the behavior; the best anchor is always retained. |
| Win-rate attribution wrong under vectorized envs (results credited to the wrong snapshot). | Opponent paths chosen by the league are carried on the env factory and outcomes aggregated back in the main process between `learn` chunks; a unit test on `record_result` plus the smoke test guard it. |
| League state pickled across `SubprocVecEnv` workers (heavy / non-picklable). | The league lives in the **main** training process only; workers receive **paths** (picklable, `training.py:142-146`) and build `FrozenSnapshotAgent`s locally — the live `League` never crosses the process boundary. |
| Added complexity for unclear benefit over 6C alone. | 6D is a **stretch** sub-sprint; the smoke gate requires "≥ 6C baseline" so it can only help or be neutral, never regress the shipped capstone. |

## Non-goals

- **No new RL algorithm.** Still `MaskablePPO` over the flattened masked
  `Discrete(ACTION_DIM)` (`model.py:173`). **PFSP is an opponent-sampling /
  curriculum change, not an algorithm change** — it only decides *which frozen
  opponents fill the seats*, exactly where `_mixed_opponent_pool` does today
  (`training.py:383-399`).
- **No obs / action / codec changes** (`sprint6-roadmap.md §3`); `OBS_DIM=72`,
  `ACTION_DIM=516`, `MAX_PLAYERS=6`, `CODEC_VERSION=1` all unchanged.
- **No distributed / multi-machine training.** Single-machine, on top of 6C's
  multi-core vec envs + one optional GPU.
- **No new agent interface.** League opponents are `FrozenSnapshotAgent`
  (`training.py:110`), the existing path-pickled `BaseAgent`.
- **No co-evolving population of *distinct* policies / exploiters.** 6D is a
  league of **past selves** (snapshots of the one learner), not AlphaStar's full
  main/exploiter/league-exploiter zoo — that is out of scope.

## Open questions / assumptions

- **League capacity.** Assumed larger than today's FIFO 3 (`training.py:356`) but
  modest (e.g. 8–16) to bound disk + sampling cost; a `CurriculumConfig` knob.
- **Default weighting fn.** Assumed **"even/variance"** (`wr·(1-wr)`) as the safe
  default for stability; "hard" `(1-wr)^p` available for a plateaued policy. The
  exponent `p` is a knob.
- **Neutral prior for unplayed entries.** Assumed win-rate 0.5 until a minimum
  game count, so new snapshots get sampled enough to be estimated; the minimum is
  a knob.
- **Retention value function.** Assumed a weighted blend of `gate_score` and
  win-rate-vs-learner plus a diversity term; the exact blend is a tuning detail to
  validate against the smoke gate.
- **Inline result aggregation cost.** Recording per-game results between `learn`
  chunks adds light bookkeeping; assumed negligible vs rollout cost (it reuses
  outcomes already produced). Revisit if it shows up in the 6C measurement plan.
