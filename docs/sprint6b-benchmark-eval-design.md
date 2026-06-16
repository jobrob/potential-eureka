# Design: Sprint 6B — Evaluation Overhaul

> **Status:** planning / design only — no code. Part of Sprint 6 (see
> `docs/sprint6-roadmap.md`). Buildable independently and in parallel with
> 6A/6C/6E.
>
> **Split note:** the *stronger heuristic* originally bundled here is now its own
> sub-sprint, **6E** (`docs/sprint6e-strong-heuristic-design.md`). 6B owns only the
> evaluation overhaul. 6B *consumes* 6E's `StrongHeuristicAgent` as a rung in its
> rating ladder, but the coupling is **soft**: 6B can rank `{Random, Heuristic,
> MLAgent}` and falls back cleanly if 6E has not landed.

## Goal

Replace "win-rate vs one weak heuristic" with an **honest scoreboard**: head-to-head
(2-player) win rates, a **pool ELO / round-robin** across agent types, evaluation vs
*stochastic* and *multiple* opponents, **cross-track** eval (consumes 6A), and
optionally a lightweight lookahead/search baseline as a yardstick.

The *meaningful strength bar* — the `StrongHeuristicAgent` ladder this scoreboard
ranks against — is designed in **6E** (`docs/sprint6e-strong-heuristic-design.md`).

Reuse `simulation/runner.py:run_batch` and `simulation/stats.py:aggregate_stats`;
add new stats types rather than rebuilding the harness.

## Motivation

"94.3% vs heuristic" measures the heuristic's weakness. `HeuristicAgent`
(`heuristic_agent.py:13-334`) is:

- **One-turn greedy:** `choose_gear` (`heuristic_agent.py:79`) scores gears by an
  *estimated* speed from the hand; `choose_cards` (`heuristic_agent.py:143`)
  re-derives speed independently — gear and cards are not jointly optimized.
- **Opponent-blind:** `choose_slipstream` (`heuristic_agent.py:281`) only looks at
  the player's own corner cost; there is no rival-position awareness, no blocking,
  no slipstream *off a specific opponent*. None of the five `choose_*` methods
  reads other players' state.
- **Magic-number tuned:** e.g. `score -= corner_cost * 10` (`heuristic_agent.py:117`),
  `score -= 100` spinout (`heuristic_agent.py:122`), `score -= heat_cost * 15`
  (`heuristic_agent.py:128`) — hand-tuned, not derived.
- **No race-long economy:** `choose_discard` (`heuristic_agent.py:308`) uses a
  local hand-size rule; there is no deck-cycling or heat-budget plan across the
  race.

So beating it proves little, and the policy that overfit to exploit it shattered
in self-play (`sprint6-roadmap.md §2`). The *real opponent* is built in **6E**
(`docs/sprint6e-strong-heuristic-design.md`); **6B builds the eval that can *rank*
agents** rather than just score one vs one weak baseline.

## Files to create / modify

| File | Action | What |
|---|---|---|
| `src/heat/agents/lookahead_agent.py` | **new, optional** | A lightweight 1-2 ply search baseline (yardstick), behind the same `BaseAgent`. Reuses 6E's `evaluate_move`. May be deferred. |
| `src/heat/ml/evaluate.py` | **modify / extend** | Add `head_to_head(...)`, `round_robin_elo(...)` (with a `rating="elo"\|"trueskill"` option), `evaluate_cross_track(...)`; reuse `evaluate_ml` (`evaluate.py:82`) and `run_batch`. |
| `src/heat/simulation/stats.py` | **modify** | Add `EloRating` / `TrueSkillRating` / `RoundRobinStats` / `HeadToHeadStats` dataclasses, **confidence intervals on win-rates and ratings**, ELO computation over `GameOutcome`s, and a minimal/optional TrueSkill path. |
| `pyproject.toml` | **modify (optional dep, maybe)** | If TrueSkill uses the `trueskill` package, add it to `[ml]`; otherwise a minimal in-repo implementation needs no dep. Decision in Open questions. |
| `tests/test_ml_eval.py` | **modify** | ELO determinism, head-to-head symmetry, cross-track aggregation. |

> The `StrongHeuristicAgent` and its `strong_heuristic_agent_factory` (in
> `runner.py`) are created by **6E** (`docs/sprint6e-strong-heuristic-design.md`).
> 6B imports them as ladder rungs once available; until then the round-robin runs
> on `{Random, Heuristic, MLAgent}` (the fallback noted above).

## Part 1 — Stronger heuristic (moved to 6E)

The stronger heuristic — joint gear+cards planning, opponent awareness,
multi-corner lookahead, race-long heat/deck economy, and the difficulty ladder that
makes ELO meaningful — is now designed in full in
**`docs/sprint6e-strong-heuristic-design.md`**. It ships its own
`StrongHeuristicAgent`, factory, and `tests/test_strong_heuristic.py`, gated on
"beats `HeuristicAgent` head-to-head." 6B consumes it as a rating-ladder rung.

## Part 2 — Richer evaluation

### New stats types (in `simulation/stats.py`)

`stats.py` already defines `AgentStats` (`stats.py:42`) and `SimulationStats`
(`stats.py:65`) over `GameOutcome`s. Add:

```python
# simulation/stats.py
@dataclass(frozen=True)
class HeadToHeadStats:
    agent_a: str; agent_b: str
    games: int; a_wins: int; b_wins: int
    a_win_rate: float                      # 2-player, head-to-head
    a_win_rate_ci: tuple[float, float]     # Wilson 95% interval on a_win_rate

@dataclass(frozen=True)
class EloRating:
    agent_type: str; rating: float; games: int
    rating_ci: tuple[float, float]         # bootstrap CI on the ELO (or +/- from K)

@dataclass(frozen=True)
class TrueSkillRating:
    agent_type: str; mu: float; sigma: float; games: int
    # conservative skill estimate = mu - 3*sigma; sigma IS the uncertainty

@dataclass(frozen=True)
class RoundRobinStats:
    ratings: dict[str, EloRating]                   # agent_type -> ELO
    trueskill: dict[str, TrueSkillRating] | None    # populated when rating="trueskill"
    pairwise: dict[tuple[str, str], HeadToHeadStats]
    per_track: dict[str, dict[str, float]] | None   # track_name -> {agent_type: win_rate}
```

ELO is computed **from `GameOutcome`s** (pure aggregation, like the rest of
`stats.py`, `stats.py:1-8` "pure aggregation … no I/O"): replay each game's
`finish_order` (`runner.py:106`) as a sequence of pairwise results (winner beats
each lower finisher) and apply the standard ELO update with a fixed K-factor.
Because it is computed from a fixed, seed-determined list of outcomes, it is
deterministic **if the iteration order is fixed** — sort games by `game_index`
(`runner.py:101`) before updating (see test gate).

### Confidence intervals (so an ordering is not read as significant when it is noise)

A bare "ELO: A 1530 > B 1515 > C 1490" invites over-reading a few points of
spread that is within sampling noise over a finite seeded batch. **Every reported
rate/rating carries an uncertainty:**

- **Win-rate CIs.** Compute a **Wilson score interval** (preferred over the normal
  approximation near 0/1 and for small N) on each win-rate — `AgentStats.win_rate`
  (`stats.py:50`), `HeadToHeadStats.a_win_rate`, and per-track win-rates. The
  interval is a pure function of `(wins, games_played)` (`stats.py:48-50`), so it
  stays in `stats.py`'s pure-aggregation contract.
- **Rating CIs.** For ELO, attach a **bootstrap CI** (resample the
  `game_index`-sorted outcome list with a fixed seed, recompute ELO, take
  percentiles) — deterministic given the bootstrap seed. *Or* prefer TrueSkill,
  whose **σ is a native uncertainty** (no bootstrap needed).
- **Significance flag.** Two agents whose rating/win-rate CIs **overlap** are
  reported as *not significantly different* — surfaced in `format_summary` and
  asserted by the sanity-ranking gate.

### TrueSkill as the principled multiplayer rating (optional, decided)

The ELO-via-pairwise-decomposition is *serviceable* but is a 1-v-1 model bent onto
a 4-player free-for-all. **TrueSkill** is the principled choice for N-way
free-for-alls: it consumes a full `finish_order` (`runner.py:106`) directly as a
ranked outcome and yields, per agent, a skill posterior `(mu, sigma)` —
**rating + uncertainty in one model**. `round_robin_elo` takes a
`rating="elo"|"trueskill"` switch; ELO stays the default so nothing else in 6B
depends on the choice. TrueSkill is computed from the same fixed,
`game_index`-sorted outcome list, so it is equally deterministic. Whether to add
the `trueskill` package or ship a minimal implementation is in Open questions.

### New eval entry points (in `ml/evaluate.py`)

`evaluate.py` already has `ml_agent_factory` (`evaluate.py:61`) and `evaluate_ml`
(`evaluate.py:82`) which wrap `run_batch` + `aggregate_stats`. Add:

```python
# ml/evaluate.py
def head_to_head(factory_a, factory_b, *, num_games=200, track=None,
                 seed=0, parallel=False) -> HeadToHeadStats:
    """2-player A-vs-B over a seeded batch. Runs both seat orders (A at seat 0
    and A at seat 1) to cancel any first-mover advantage."""

def round_robin_elo(factories: dict[str, AgentFactory], *, num_games_per_pair=200,
                    num_players=2, tracks=None, seed=0, rating="elo",
                    bootstrap=1000, parallel=False) -> RoundRobinStats:
    """Round-robin every pair of named agents, compute ratings + pairwise tables,
    optionally across a set/sampler of tracks (cross-track).

    rating in {"elo","trueskill"}: ELO (pairwise decomposition + bootstrap CI) or
    TrueSkill (native N-way mu/sigma). All ratings and win-rates carry CIs; pairs
    whose CIs overlap are flagged not-significant. `bootstrap` is the ELO CI
    resample count (seeded → deterministic)."""

def evaluate_cross_track(factories, *, tracks, num_games=100, seed=0,
                         parallel=False) -> dict[str, dict[str, AgentStats]]:
    """Run a fixed eval across multiple tracks; return per-track stats.
    `tracks` is a list[Track] (e.g. [usa, silverstone] OR generated via 6A)."""
```

### Evaluating vs stochastic and multiple opponents

The Sprint-5 self-play collapse was partly caused by *deterministic identical*
opponents (`sprint6-roadmap.md §2 (c)`). Eval must therefore exercise:

- **Stochastic opponents:** include `RandomAgent` (`runner.py:128-130`) and a
  *stochastic* MLAgent (`ml_agent_factory(..., deterministic=False)`,
  `evaluate.py:61-79`) in the pool, so a brittle policy that only beats a fixed
  opponent is exposed.
- **Multiple distinct opponents in one game:** 3-6 player games with a *mix* of
  agent types (not all-identical seats), via heterogeneous `factories` to
  `run_batch` (the runner already supports one factory per seat,
  `runner.py:212-216`).

### Cross-track (consumes 6A; falls back to static)

`evaluate_cross_track` takes a `tracks` list. With 6A landed, pass tracks from
`generate_track`/`track_sampler` (`sprint6a-track-generation-design.md`). Without
6A, pass `[load_track_by_name("usa"), load_track_by_name("silverstone")]`
(`loader.py:43`). The harness is identical either way — this is the soft edge
(ii) in `sprint6-roadmap.md §5.1`.

### Optional lookahead/search yardstick

A `LookaheadAgent` (1-2 ply over `rules.legal_*` enumerations, evaluating the
resulting state with a simple value) gives an *upper-ish* reference point that is
neither the trained policy nor the scripted heuristic — useful to interpret ELO
("the policy sits between strong-heuristic and 2-ply lookahead"). **Optional and
deferrable**; if the search proves expensive, cap depth or drop it without
affecting the rest of 6B.

### Picklability / parallel caveat (carried from Sprint 5)

`run_batch(parallel=True)` pickles factories to spawned workers
(`runner.py:25-30`, `runner.py:150-167`). The same rules apply to 6B:

- Heuristic/strong-heuristic/random factories are top-level `functools.partial`
  (picklable) — fine in parallel.
- **MLAgent factories** carry a *path*, not a model (`evaluate.py:44-58`,
  `ml_agent.py:29-34`), so they pickle, but each worker reloads the SB3 model;
  `evaluate_ml` defaults `parallel=False` (`evaluate.py:88`) for this reason. ELO
  round-robins that include an MLAgent should default to **sequential** unless the
  batch is large; document this on `round_robin_elo`.
- Never pass a **lambda/closure** factory to `parallel=True` (the runner falls
  back to sequential with a warning, `runner.py:366-375`). The new factories must
  be top-level/partial.

### Part-2 test gates (`test_ml_eval.py` additions)

- **ELO determinism:** the same outcome list → identical ratings (fixed
  `game_index` ordering); permuting the *input* order does not change ratings.
- **Head-to-head symmetry:** running A-vs-B and B-vs-A on the same seeds yields
  consistent `a_win_rate`/`b_win_rate` (sum ≈ 1 minus draws); the dual-seat
  cancellation works (no large seat-0 bias remains).
- **Cross-track aggregation:** `evaluate_cross_track` over 2+ tracks returns a
  per-track stats dict with the right keys; aggregate win-rates are in `[0, 1]`.
- **Confidence intervals are computed and sane:** every win-rate CI satisfies
  `0 <= lo <= rate <= hi <= 1`; a degenerate batch (all wins / all losses) yields
  a valid Wilson interval (not NaN); larger N yields a *narrower* interval than
  small N (monotone-in-N sanity). ELO bootstrap CIs are deterministic given the
  bootstrap seed.
- **Sanity ranking (with significance):** in a round-robin of {Random, Heuristic,
  StrongHeuristic}, ELO (and TrueSkill `mu`) order them
  StrongHeuristic > Heuristic > Random over a sufficiently large seeded batch, and
  the gate asserts the **CIs do not overlap** at that batch size (a real ranking,
  not noise). On a *tiny* batch the gate instead asserts the overlap is correctly
  **flagged not-significant** — i.e. the CI machinery refuses to over-claim.
- **TrueSkill path (when enabled):** `rating="trueskill"` populates
  `RoundRobinStats.trueskill` with finite `mu`/`sigma`, `sigma` shrinks with more
  games, and the `mu` ordering matches the ELO ordering on the sanity batch.
- **Reuse, not rebuild:** the new harness calls `run_batch`/`aggregate_stats`
  (assert via the existing `GameOutcome`/`AgentStats` shapes, `runner.py:86`,
  `stats.py:42`).
- **Full suite green.**

## Build sequence

1. **(6E, parallel)** `StrongHeuristicAgent` + `strong_heuristic_agent_factory`
   land via `docs/sprint6e-strong-heuristic-design.md`. 6B does **not** block on
   this — start at step 2 against `{Random, Heuristic, MLAgent}` and add the strong
   rung when 6E lands.
2. **Stats types** (`HeadToHeadStats`, `EloRating`, `TrueSkillRating`,
   `RoundRobinStats`) + ELO computation over `GameOutcome`s + **Wilson win-rate
   CIs** + **bootstrap ELO CIs** + the **CI-overlap significance flag**.
3. **`head_to_head` + `round_robin_elo` (with `rating="elo"|"trueskill"`) +
   `evaluate_cross_track`** in `evaluate.py`, reusing `run_batch`.
4. **Cross-track** wiring (static fallback now; generated tracks once 6A lands).
5. **Optional `LookaheadAgent`** yardstick.
6. Full suite: `PYTHONPATH=src python -m pytest tests/ -q`.

## Risks + de-risking

| Risk | De-risking |
|---|---|
| Strong-heuristic correctness/speed risks. | Owned by **6E** (`docs/sprint6e-strong-heuristic-design.md` Risks). 6B only consumes the agent through `run_batch`. |
| ELO is non-deterministic or order-sensitive. | Compute from a fixed, `game_index`-sorted outcome list; test that input permutation does not change ratings. |
| Parallel eval flakiness with MLAgent in the pool. | Default round-robins with an MLAgent to sequential; only top-level/partial factories (no lambdas); reuse the runner's pre-flight pickle check (`runner.py:150-167`). |
| "Stronger" is asserted but not real. | The head-to-head win-rate gate (> 50% vs old heuristic by a margin) is a concrete pass/fail, not a vibe. |
| ELO ordering over-read as significant when it is sampling noise. | Every rating/win-rate carries a CI (Wilson for rates, bootstrap for ELO, σ for TrueSkill); overlapping CIs are flagged not-significant and the sanity-ranking gate asserts non-overlap only at adequate N. |
| TrueSkill adds a dependency / hidden non-determinism. | TrueSkill is **optional** (`rating="elo"` default); computed from the fixed `game_index`-sorted outcomes so it is deterministic; the dep-vs-minimal-impl decision is recorded in Open questions (keep it optional, don't force it into `[ml]` if avoidable). |

## Non-goals

- **No engine rule changes** — the strong heuristic and lookahead use only
  existing `rules.*` and public state.
- **No change to `run_batch`/`aggregate_stats` semantics** — only *new* factory
  and *new* stats types are added (`stats.py:18` "single place to extend").
- **No obs/action/codec changes** (`sprint6-roadmap.md §3`).
- **No full MCTS / deep search** — the optional `LookaheadAgent` is shallow (1-2
  ply) and is a yardstick, not a competitor (ruled out for HEAT by hidden info +
  randomness, `sprint5-ml-roadmap.md:561`).
- **No "ML beats X" as a unit gate** — strength of the *trained* agent is the
  capstone's manual/CI-optional metric; 6B provides the *tooling* to measure it.
- **No mandatory new dependency for ratings.** TrueSkill is optional and behind a
  switch; if it ships, it is an *optional* `[ml]` dep or a minimal in-repo
  implementation — ELO + CIs work with zero new deps.

## Open questions / assumptions

- **ELO K-factor and starting rating.** Assumed standard (K≈16-32, start 1500);
  fixed constants so results are reproducible. Surface as a parameter.
- **Draws.** HEAT games produce a strict `finish_order` (`runner.py:106`), so
  pairwise results are decisive; no draw handling needed unless a tie in
  `finish_order` is possible (verify — `game_state.py:97` sorts by
  `finish_order`).
- **Whether the lookahead baseline ships in 6B or is deferred.** Assumed optional;
  the rest of 6B does not depend on it.
- **Cross-track track set size.** Assumed a modest fixed sweep (e.g. the 2 static
  + a handful of generated seeds); the exact count is a tuning detail.
- **TrueSkill: dependency vs minimal in-repo implementation.** Assumed *optional*.
  Adding the `trueskill` package to `[ml]` is the least-code path; a minimal
  Gaussian-update implementation avoids the dep but is more code to test. **Decision:
  keep ELO + CIs as the always-available default; TrueSkill is opt-in** — choose
  package-vs-minimal when implementing (lean to the package if it is light and the
  team is fine with the dep). Either way it must stay deterministic.
- **CI method + confidence level.** Assumed **Wilson 95%** for win-rates and a
  seeded **percentile bootstrap (≈1000 resamples)** for ELO; both surfaced as
  parameters with fixed defaults so results are reproducible.
- **Bootstrap cost.** A large `bootstrap` over a big round-robin is extra compute;
  assumed cheap relative to running the games themselves (it resamples existing
  outcomes, no new games). Cap or lower it if it dominates.
