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
