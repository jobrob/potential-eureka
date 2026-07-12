# Sprint A7 — Multi-seat / held-out evaluation harness

> **Status:** design (2026-07-10). Detailed design for Sprint A7 of the
> [Direction A sprint plan](sprint-plan.md). Implementable spec; read this plus
> `src/heat/ml/selfplay/{recipe,snapshots,multiseat}.py` (the `_EvalCollector`
> pattern and `SnapshotAgent`), `src/heat/tracks/generator.py`
> (`track_sampler` and its **seed-namespace** mechanism), `src/heat/simulation/stats.py`
> (`wilson_interval`), `src/heat/agents/strong_heuristic.py`, and the
> [A5 gate results](A5-anticollapse-recipe.md) §8 (the G2 dispute this harness must
> settle).

## 1. Purpose

A7 builds the instrument that defines "done" for Direction A: score a policy **vs
weak/strong heuristics across seat counts (2–6) and across track splits (fixed /
training-distribution / held-out generated)** with Wilson lower-bound confidence,
plus a **fixed-anchor self-improvement probe** (the statistically honest version of
A5's disputed G2). Every later run — A6's gate re-reads, A4 ablations, A8's Phase-1
verdict — reports through this harness.

## 2. Key design decisions

1. **Reuse the generator's existing seed-namespace split.** `track_sampler(params,
   base_seed=0)` is documented as the *eval namespace* reaching the reserved
   held-out `900_000+` seed band, and any non-zero `base_seed` is a disjoint
   training namespace. A7 does not invent a split: the held-out set is
   `track_sampler(params, base_seed=0)` over seeds `900_000..900_000+n`, training
   samplers must use `base_seed != 0`, and a test *verifies* disjointness by track
   fingerprint over a large sample rather than trusting the doc comment.
2. **Games run through the A2/A5 machinery** — a policy seat rotating through the
   field with every other seat scripted (`_EvalCollector` pattern, promoted into the
   harness); opponents are `HeuristicAgent` / `StrongHeuristicAgent` / any
   `BaseAgent` (including `SnapshotAgent` anchors). No SB3 / `evaluate_ml`
   dependency — that path stays for MLAgent checkpoints.
3. **"Win" = first place; placement reward reported alongside.** At >2 seats a
   binary top-half signal hides skill; the headline metric is first-place rate with
   `wilson_interval` bounds against the `1/num_players` chance line, with mean
   placement reward as the graded secondary.
4. **Policies must be saveable to be evaluable.** Direction-A policies currently
   live only in memory; A7 adds a minimal checkpoint (`save_policy`/`load_policy`:
   `state_dict` + head/hidden/obs/action dims + `spaces.CODEC_VERSION` tripwire) so
   trained runs can be scored later. This is deliberately tiny — not the SB3
   sidecar system.

## 3. Scope

**In scope**
- `src/heat/ml/selfplay/checkpoint.py` — `save_policy` / `load_policy`.
- `src/heat/ml/selfplay/eval_harness.py` — held-out track set, `evaluate_policy`
  (the cell grid), `evaluate_vs_anchor`, report dataclasses + markdown rendering.
- `--save PATH` on `scripts/train_selfplay_a5.py` (one flag; produces the artifact
  the harness consumes).
- `scripts/a7_skillbar.py` — CLI producing the skill-bar report for a checkpoint
  (or for heuristic-only baselines).
- Tests + the §5 gate runs.

**Out of scope**
- PFSP / persisted league integration (A8+, per the A5 §2.2 deferral).
- Any training change; any new observation/reward work.
- Elo / round-robin machinery (`evaluate.py` keeps that for the SB3 world).

## 4. Design

### 4.1 `checkpoint.py`

`save_policy(policy, config, path)` → `torch.save` of
`{"state_dict", "head", "hidden_sizes", "obs_dim", "action_dim", "codec_version"}`.
`load_policy(path) -> PPOPolicy` rebuilds via `build_policy`-equivalent construction
and **fails fast** on any obs/action/codec mismatch with the live contract (mirror
`MLAgent`'s tripwire semantics in one `if`; no sidecar files).

### 4.2 `eval_harness.py`

- `held_out_tracks(n=20, params=None) -> list[Track]` — `track_sampler(params,
  base_seed=0)` over seeds `900_000 + i`. Default `params=None` = the full
  generator distribution; `tiny_heat_params()` may be passed for tiny-bed work.
- `EvalCell` / `EvalReport` dataclasses: per **(opponent, seat_count, split)** cell —
  games, first-place wins, win rate, `wilson_interval(wins, games)` bounds, chance
  line `1/num_players`, mean placement reward. `EvalReport.to_markdown()` renders
  the grid; `EvalReport.clears_bar(opponent, lb_margin)` answers "Wilson-LB above
  chance by `lb_margin` at every seat count" (the A8 skill-bar predicate).
- `evaluate_policy(policy, *, opponents={"weak": HeuristicAgent, "strong":
  StrongHeuristicAgent}, seat_counts=(2,3,4,6), splits={"tiny": ..., "heldout":
  held_out_tracks(...)}, games_per_cell=50, seed) -> EvalReport` — for each cell,
  rotate the policy seat game-to-game, all other seats one scripted opponent
  instance per seat, track cycling through the split's list; the policy **samples**
  (`act`), matching every prior yardstick. Uses the promoted `_EvalCollector`
  (terminal-state capture) from `recipe.py` — move it here, re-export/import back
  into `recipe.py` so there is exactly one implementation.
- `evaluate_vs_anchor(policy, anchor: SnapshotAgent, *, num_players=2, track,
  games=200, seed) -> EvalCell` — the fixed-anchor self-improvement probe: 200
  games (se ≈ 0.035, vs the A5 gate's underpowered 40) and a Wilson-LB verdict
  against 0.5. This is the instrument that settles A5's disputed G2.
- Heuristic-only baselines: `evaluate_policy` must accept a `BaseAgent` in the
  policy seat too (type the parameter `PPOPolicy | BaseAgent`; a `BaseAgent` skips
  encoding and just plays) — that is how the strong-vs-weak baseline grid (§5 G1)
  runs through the *same* code path it certifies.

### 4.3 `scripts/a7_skillbar.py`

Flags: `--checkpoint PATH` (or `--baseline strong|weak|random` for heuristic-only
grids), `--opponents weak strong`, `--seats 2 3 4 6`, `--splits tiny heldout`,
`--games N` (default 50/cell), `--anchor PATH` (optional: adds the
`evaluate_vs_anchor` row against a second checkpoint), `--seed`. Prints the
markdown report; exits non-zero if a requested `--bar` predicate fails (default: no
bar, report-only).

## 5. Acceptance gate (exit criteria)

- **G1 — reproduces known baselines.** The heuristic-only grid shows
  `StrongHeuristicAgent` beating a weak field with Wilson-LB **above** the chance
  line at every seat count in {2,3,4,6} on both splits, and `RandomAgent` **below**
  the weak field's chance-line performance at 2p (sanity direction check). Record
  the measured 2p strong-vs-weak win rate in §8 (the project's historical margin
  reference point).
- **G2 — held-out split verifiably disjoint.** A test generates ≥500 training-
  namespace tracks (`base_seed != 0`) and the held-out set, and asserts zero
  fingerprint overlap (reuse/adapt `features._track_fingerprint`).
- **G3 — the A5 G2 dispute, settled with power.** Train one A5-recipe policy
  (pool arm, seed 2 — the disputed seed, 150k steps), save it, snapshot an anchor
  at ~iteration 15 (~30k steps) via a `--save`-produced early checkpoint or an
  in-run snapshot, and run `evaluate_vs_anchor` at 200 games. Record the Wilson-LB
  verdict in §8: this either confirms seed 2 genuinely fails to climb (recipe
  problem — reopen A5) or clears it (yardstick problem — A5 G2 closed as
  measurement artifact). Either outcome is a pass **of A7** (the instrument
  worked); the *finding* routes A5's status.
- **G4 — end-to-end policy report.** `a7_skillbar.py` on the G3 checkpoint produces
  the full grid (weak+strong × seats × tiny+heldout) without error; cells are
  internally consistent (wins ≤ games, LB ≤ rate ≤ UB).
- **G5 — engineering.** Checkpoint round-trip test (save→load→identical outputs on
  a fixed obs batch; mismatched codec_version raises); new tests green; full suite
  green; `ruff` clean; mypy per repo practice; no behavior change to any training
  path beyond the added `--save` flag.

Runtime note: heuristic-only cells are engine-speed (hundreds of games ≈ seconds);
policy cells are ~50 games × ~8 cells ≈ minutes; the G3 training run is ~3 min.
Chunk invocations to stay inside command timeouts.

## 6. Tests (`tests/test_a7_eval_harness.py`)

1. Disjointness (G2, the ≥500-track fingerprint test — mark it if slow, but it must
   run in the gate).
2. Checkpoint round-trip + codec tripwire (G5).
3. `evaluate_policy` smoke on Tiny-Heat 2p with tiny games_per_cell: report shape,
   cell arithmetic invariants (wins ≤ games, bounds ordered, chance line correct
   per seat count).
4. Win definition: a constructed terminal state where the policy seat finished
   first / not-first scores 1 / 0 respectively.
5. `evaluate_vs_anchor`: an anchor built from the same policy scores ≈ 50% over a
   moderate game count (self-play symmetry sanity, wide tolerance).
6. Heuristic-in-policy-seat path: strong-vs-weak 2p mini-grid runs and strong's
   rate > 0.5 (tiny game count, direction only — the powered version is the gate).

## 7. Notes for the implementer

- Match conventions; only edits to existing files: `recipe.py` (import the promoted
  `_EvalCollector` back from `eval_harness` — no behavior change),
  `train_selfplay_a5.py` (`--save`), `selfplay/__init__.py` exports.
- `StrongHeuristicAgent` may be slower per decision than `HeuristicAgent` — measure,
  and if a full-distribution 6-seat strong cell is slow, reduce that cell's games
  rather than skipping it (note any reduction in the report).
- Run before committing: new tests, full suite, the §5 gate runs; append `## 8.
  Gate results` (baseline grid, disjointness confirmation, the G3 anchor verdict
  with its A5-routing conclusion, the G4 report) to this doc.
- Commit protocol (as A3/A5/A6): commit 1 = this doc alone (`Design Sprint A7:
  multi-seat held-out eval harness`); commit 2 = implementation + tests + gate
  results (`Implement Sprint A7: ...` with the headline verdict in the subject),
  only after full suite green + gate runs. `git add` specific paths; do not push.
  End both messages with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## 8. Gate results (2026-07-10)

Headline: **A7 built and green; the harness is validated correct against an
independent baseline. The G3 anchor probe settles A5's disputed G2 — seed 2
genuinely CLIMBS (final-vs-iter15 anchor 76.5%, Wilson-LB 0.702 > 0.5 over 200
games), so A5 G2 closes as a yardstick artifact, not a recipe failure.**

All runs on CPU. Policy grid via `scripts/a7_skillbar.py`; the two G3 training
runs via `scripts/train_selfplay_a5.py --save`.

### G1 — reproduces known baselines

Heuristic-only grid, `StrongHeuristicAgent` in the policy seat vs a weak
`HeuristicAgent` field, policy seat rotated game-to-game (seat-neutral):

| split | seats | strong win% | Wilson-LB | chance | mean reward | LB>chance |
|---|---|---|---|---|---|---|
| tiny (200/cell) | 2 | 55.0% | 0.481 | 0.500 | +0.100 | no |
| tiny | 3 | 54.0% | 0.471 | 0.333 | +0.370 | yes |
| tiny | 4 | 48.5% | 0.417 | 0.250 | +0.323 | yes |
| tiny | 6 | 16.5% | 0.120 | 0.167 | +0.154 | ~chance |
| heldout (100/cell) | 2 | 37.0% | 0.282 | 0.500 | −0.260 | no |
| heldout | 3 | 33.0% | 0.246 | 0.333 | −0.100 | no |
| heldout | 4 | 32.0% | 0.237 | 0.250 | −0.013 | no |
| heldout | 6 | 12.0% | 0.070 | 0.167 | −0.192 | no |

Random-agent sanity (policy seat = `RandomAgent`, vs weak, 200 games): tiny 2p
**38.0%** (mean −0.240), heldout 2p **0.0%** (mean −0.925) — random is well below
the weak field everywhere (direction check PASS; ranking is random < weak).

**The literal bar ("strong beats weak, Wilson-LB above chance at every seat on
both splits") is NOT met.** Per the gate discipline this is investigated as
harness-bug-vs-true-result, and it is a **true result**, cross-validated against
the independent `run_batch` engine path (fresh per-game agent factories,
seat-neutral by averaging both seat assignments):

- `_play_scripted_game` vs the engine `Game` on identical (track, seed, agents):
  **0 / 40 winner mismatches** — the harness game driver is faithful.
- 2p seat-neutral strong-vs-weak, `run_batch`: **tiny 52.7%** (strong@seat0 24.0%
  vs strong@seat1 **81.5%** — extreme short-track seat bias) vs the harness's
  55.0%; **heldout (same 20 tracks) 40.7%** vs the harness's 37.0%. Both within
  sampling noise.

So `StrongHeuristicAgent` only *marginally* edges the weak heuristic at 2p
first-place on tiny, and **actively loses on the generated-track distribution** —
a real property (the strong agent appears tuned to the static tracks), not a
harness defect. **Recorded 2p strong-vs-weak reference point: 55.0% (tiny) /
37.0% (heldout), matching `run_batch` ground truth (52.7% / 40.7%).**
**Verdict: instrument-reproduces-ground-truth PASS; the literal strong>weak-LB
numeric bar FAILs by a true, cross-validated result — reported as measured, no
game-count adjustment.**

### G2 — held-out split verifiably disjoint

`test_heldout_split_disjoint_from_training_namespaces`: 600 training-namespace
tracks (`base_seed` ∈ {1, 12345}, 300 seeds each) and the 50-track held-out set
(`base_seed=0`, seeds `900_000+`) share **zero** full-structural fingerprints
(the fingerprint adapts `features._track_fingerprint`, adding per-space lanes +
the start grid so only genuinely identical tracks collide). **PASS.**

### G3 — the A5 G2 dispute, settled with power

Trained one A5-recipe policy (pool arm, seed 2, 150k steps, 137.5s) and its
iteration-15 anchor (a second `--save` run at `--timesteps 30720` = 15 iters,
byte-identical to iteration 15 of the full run since the same seed makes the
iteration prefix identical). `evaluate_vs_anchor`, 200 games, Tiny-Heat 2p:

| probe | games | wins | win% | Wilson-LB | Wilson-UB | verdict |
|---|---|---|---|---|---|---|
| final(150k) vs iter-15 anchor | 200 | 153 | 76.5% | **0.702** | 0.818 | **CLIMBS (LB > 0.5)** |

(For reference the same final policy beats the weak heuristic 92.5% at tiny 2p.)

**A5-routing conclusion: A5's disputed G2 is CLOSED as a measurement artifact.**
The A5 gate's "seed 2 fails at 0.450 vs oldest" used a *rolling recent snapshot*
at only 40 games (se ≈ 0.08) — it conflated "still climbing" with a moving
target and was underpowered. With a **fixed** early-training anchor and 200
games, seed 2 climbs decisively (Wilson-LB 0.702 ≫ 0.5). This is exactly the
"yardstick problem, not recipe problem" reading A5 §8 hypothesized. The A5 recipe
is validated; **A5's sprint-plan gate can be recorded as G2-cleared** (a
measurement artifact, not a recipe failure). Either outcome passes A7 — the
instrument worked.

### G4 — end-to-end policy report

`a7_skillbar.py` on the seed-2 checkpoint produced the full grid (weak+strong ×
{2,3,4,6} × {tiny,heldout}, 50 games/cell, 16 cells, 43.7s) without error; every
cell is internally consistent (wins ≤ games, LB ≤ rate ≤ UB, chance = 1/seats).
The trained tiny-2p policy dominates on its training bed (vs weak: 86% / 90% /
72% at 2/3/4 seats, all Wilson-LB above chance; vs strong: 78% / 70% / 62%) and
**collapses on held-out generated tracks** (≈0–10%, strongly negative mean
reward) — the known tiny→generated generalization gap, which A7 now *measures*.
**PASS.**

### G5 — engineering

New tests 6/6 green; full suite **1056 passed / 1 skipped** (~2m51s); `ruff`
clean; `mypy --strict` clean on the new `checkpoint.py` / `eval_harness.py` (and
the edited `recipe.py`). The only training-path change is the additive `--save`
flag; `_EvalCollector` was promoted into `eval_harness.py` and imported back into
`recipe.py` (one implementation, no behavior change). **PASS.**

### Instrument caveats surfaced (for A6/A4/A8 consumers)

1. **Seat bias is large and must be averaged out.** Tiny 2p strong@seat0 24% vs
   strong@seat1 81.5% — the harness's game-to-game policy-seat rotation is
   load-bearing; any fixed-seat read would be badly wrong.
2. **First-place rate saturates toward chance at 6 seats** even when mean
   placement reward stays positive (crowded short-track fields). At high seat
   counts read the graded mean placement reward alongside first-place rate.
3. **`StrongHeuristicAgent` does not dominate the generated-track distribution**
   (loses to the weak heuristic at 2p first-place). A real finding, out of A7
   scope — flagged for a follow-up.

### Deviation from §3/§7

No `--save-at-iter` flag: the anchor is produced by a **second `--save` run at
reduced `--timesteps`** (the spec's explicitly-allowed "`--save`-produced early
checkpoint" route). Because the same `--seed` makes iterations 1..k byte-
identical regardless of the total, `--timesteps 30720 --seed 2` reproduces
exactly the policy the full 150k seed-2 run holds at iteration 15. This is the
smallest mechanism that works *and* keeps §7's "recipe.py: no behavior change
beyond the `_EvalCollector` import" — an in-run policy-snapshot hook would have
required a recipe training-path change. Documented in the `--save` help.
