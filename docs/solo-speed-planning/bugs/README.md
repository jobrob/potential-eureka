# Bugs & risks register (pre-Option-C)

Read-only code-review findings collected **before** building Option C (solo
AlphaZero), so the engine / codec / eval surfaces C reuses are trustworthy first.
Findings only — no fixes applied. See the parent [`../README.md`](../README.md) and
[`../option-C-search-learning/README.md`](../option-C-search-learning/README.md).

## Reviews

- [`code-review-2026-06-22.md`](code-review-2026-06-22.md) — full codebase review by
  five parallel section reviewers (engine+models, agents, ML core/codec,
  tracks/eval-infra, experiment harnesses). 10 HIGH, 4 MEDIUM, 5 LOW, plus a
  "confirmed sound" list.

## Pre-build blockers (the short list)

These four directly determine whether C's training signal and held-out gate are
trustworthy — resolve before C1 builds:

1. **Obs not state-pure** — `round_num` leaks in; `decision=None` zeroes the phase
   block, so value-path and policy-path encodings of the same state differ
   (`features.py:266`). Corrupts both A's V and C's critic.
2. **Seed-band leakage** — additive seed-mixing gives no structural disjointness
   between self-play and the held-out eval band 900,000 (`generator.py:283`). Can
   fake a rung-4 pass.
3. **Data-gen action encoding** — `encode_action_index` raises on off-table
   REACT/out-of-range actions (`action_codec.py`); self-play data gen must inherit
   `gen_demos.py`'s drop-and-count (already mandated in C2).
4. **Promotion guard mis-cited** — the real Wilson-LB / best-checkpoint guard is in
   `value_iterate.py`, not `dagger.py`; C3's design should point there (doc fix).

Then **verify with targeted tests** before wiring the engine into MCTS: clone-RNG
parent advance (`game_state.py:167`), lap/position desync (`phases.py`), deck-count
conservation on exhaustion (`cards.py`/`phases.py`), and sequential-runner global
RNG (`runner.py:355`).

## Status

All items are **open / unverified** as of 2026-06-22. This register records risks;
triage and fixes are a separate, not-yet-started piece of work.
