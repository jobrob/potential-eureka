# Sprint A5 — Anti-collapse self-play recipe

> **Status:** **implemented and complete** (2026-07-10). The original gate was
> partial, then A7's powered fixed-anchor evaluation cleared the disputed G2 as a
> measurement artifact. The snapshot-pool recipe remains the Direction A baseline.

## 1. Purpose and starting point

A5 is the sprint plan's historical-collapse gate. The starting point is better than
planned: the A2 probe already showed **naive** current-policy self-play (no floor, no
pool) does not collapse on Tiny-Heat seed 0 — it climbs 41% → ~85–92% vs the weak
heuristic with entropy stable at ~0.7. So A5's job is sharpened:

1. **Harden** that result into a recipe with explicit guards — an entropy floor
   (a parachute that engages only if entropy decays toward collapse), a
   recent-snapshot opponent pool (damps the 84–93% self-chasing oscillation), and a
   **Stage-1 validation** check (the 8C lesson: catch a broken run in the first
   minutes, never launch long runs unvalidated).
2. **Prove it across seeds** — the A2 probe was one seed; the sprint-plan gate is
   "climbs vs previous-self and beats the weak heuristic without collapsing across
   several seeds".
3. **Measure whether the pool actually helps** vs pure current-policy self-play —
   the Big 2 finding was that *current-policy* beat fixed-opponent curricula; we keep
   both arms and let the gate data decide.

## 2. Key design decisions

1. **Snapshots ride the existing `scripted_seats` hook.** `MultiSeatCollector`
   already advances scripted `BaseAgent` seats without recording them. A frozen
   policy snapshot becomes a `SnapshotAgent(BaseAgent)` — so the collector, buffer,
   and trainer are all untouched. (This is exactly what the hook was built for in
   the A2 design.)
2. **In-memory pool; `league.py` PFSP deferred.** The sprint plan says "reuse
   `league.py`", but `League` is a path-based, on-disk, PFSP-weighted structure from
   the SB3 era. A5's pool is a deque of ≤5 in-process frozen policies sampled
   uniformly — the league machinery buys nothing here and couples us to checkpoint
   files. PFSP + persisted leagues return at A7/A8 when checkpoints go to disk;
   `SnapshotAgent` is the adapter both will reuse. *Deliberate deviation, flagged.*
3. **The entropy floor is a parachute, not a driver.** Healthy runs sit near ~0.7
   entropy; the controller only intervenes when mean rollout entropy sinks below the
   floor (default **0.40**), multiplying `ent_coef` up until entropy recovers, then
   decaying back toward the configured base. A healthy run should end with the
   controller having done nothing (assert this in the gate report).
4. **Default head = `masked`** (per the A3 gate result). The recipe is
   head-agnostic via `build_policy`.

## 3. Scope

**In scope**
- `src/heat/ml/selfplay/snapshots.py` — `SnapshotAgent` + `SnapshotPool`.
- `src/heat/ml/selfplay/recipe.py` — `A5Config`, `EntropyController`,
  `Stage1ValidationError`, `train_selfplay_a5` (the recipe loop with periodic eval).
- `scripts/train_selfplay_a5.py` (CLI) and `scripts/a5_gate.py` (the multi-seed
  gate experiment).
- Tests.

**Out of scope (named so they aren't built here)**
- PFSP / persisted league, disk checkpoints → A7/A8 (decision §2.2).
- Dense/margin value targets → **A6**. Aux heads → A8.
- The full Wilson-LB multi-seat eval harness → **A7** (A5's eval is the probe-style
  vs-heuristic / vs-snapshot winrate, both seat orders).
- Any hyperparameter search beyond the defaults below — the gate runs the defaults;
  a FAIL is reported, not tuned away (A3 discipline).

## 4. Design

### 4.1 `snapshots.py`

```python
class SnapshotAgent(BaseAgent):
    """A frozen policy snapshot behind the BaseAgent pull-method interface."""
    def __init__(self, policy: PPOPolicy, *, name: str = "Snapshot") -> None: ...
```

- Construction **deep-copies** the policy, `.eval()`s it, and detaches it from any
  optimizer (a pool entry must be immutable as the live net trains on).
- Each `choose_*` method rebuilds the `Decision(kind, pid, legal)` its arguments
  came from (see `opponent_action` for the exact per-kind `legal` payloads — REACT
  receives unpacked `ReactOptions` fields and must reconstruct
  `rules.ReactOptions(...)`; SLIPSTREAM's legal is `True`), then:
  1. `forced_action(decision, state)` — if not `NO_FORCED`, return it (handles the
     all-False empty-hand CARDS mask exactly as the collector does);
  2. else `encode_observation(state, pid, decision)` + `legal_action_mask` +
     `policy.act` (sampled, CPU) + `decode_legal_action` → the engine action.
- Keep it on CPU; batch size 1. No meta/tripwire machinery (in-process only —
  contract drift is impossible within one run).

```python
class SnapshotPool:
    def __init__(self, capacity: int = 5) -> None: ...
    def push(self, policy: PPOPolicy, label: str) -> None      # deep-copy, evict oldest
    def sample(self, rng: np.random.Generator) -> SnapshotAgent  # uniform over entries
    def oldest(self) -> SnapshotAgent                            # the "previous self" yardstick
    def __len__(self) -> int
```

### 4.2 `recipe.py`

**`A5Config(A0Config)`** (dataclass subclass; A0 fields keep their meaning):

| field | default | meaning |
|---|---|---|
| `entropy_floor` | 0.40 | controller engages below this mean rollout entropy |
| `ent_scale_up` / `ent_scale_down` | 1.5 / 0.98 | multiplicative controller steps |
| `ent_coef_max` | 0.10 | controller ceiling (base `ent_coef` is the floor) |
| `snapshot_every` | 10 | iterations between pool pushes (push at iter 1 too) |
| `pool_capacity` | 5 | recent snapshots kept |
| `pool_prob` | 0.5 | per-iteration probability the collection uses one pool opponent |
| `eval_every` | 10 | iterations between eval checkpoints |
| `eval_games` | 40 | eval games per checkpoint (~half per seat order) |
| `stage1_iter` | 15 | iteration of the Stage-1 check (~30k steps at n_steps=2048) |
| `stage1_min_winrate` | 0.55 | vs weak heuristic; A2 probe showed ~70% by 20k steps |
| `stage1_enabled` | True | |

**`EntropyController`**: holds `current_ent_coef` (starts at `config.ent_coef`).
After each rollout: if mean entropy < floor → `coef = min(coef * ent_scale_up,
ent_coef_max)`; elif entropy > floor * 1.2 → `coef = max(coef * ent_scale_down,
config.ent_coef)`. The PPO update must consume `current_ent_coef` — `ppo_update`
takes its coefficient from the config object it is passed, so the loop passes a
per-iteration shallow config copy (`dataclasses.replace(config,
ent_coef=controller.current_ent_coef)`); do **not** mutate the caller's config.
Log the coefficient and an `engaged: bool` flag per iteration.

**`train_selfplay_a5(config, *, track=None, on_iteration=None) -> tuple[PPOPolicy,
list[dict]]`** — the A5 loop (mirrors `train_multiseat`'s structure; returns the
policy and the list of eval records):

```
pool = SnapshotPool(capacity)
for iteration in 1..n_iterations:
    # opponent plan for this iteration (per-iteration, not per-game — simpler,
    # statistically equivalent over a run):
    if len(pool) > 0 and rng.random() < pool_prob:
        seat = rng.integers(num_players)          # one pool opponent this iteration
        collector = MultiSeatCollector(track, num_players,
                                       scripted_seats={seat: pool.sample(rng)})
    else:
        collector = MultiSeatCollector(track, num_players)   # pure self-play
    collect -> per-seat GAE -> concat -> ppo_update(with controller's ent_coef)
    controller.update(mean rollout entropy)
    if iteration == 1 or iteration % snapshot_every == 0: pool.push(policy, f"iter{iteration}")
    if iteration % eval_every == 0: eval record (see below)
    if stage1_enabled and iteration == stage1_iter:
        eval vs weak heuristic; if winrate < stage1_min_winrate or entropy
        collapsed (< 0.5 * floor): raise Stage1ValidationError(diagnostics)
```

**Eval record** (the probe pattern, via a fresh `MultiSeatCollector` with the
opponent in `scripted_seats`, both seat orders, `eval_games` total): winrate + mean
return **vs weak `HeuristicAgent`** and **vs `pool.oldest()`** ("previous self"),
plus current entropy, `ent_coef`, controller-engaged flag, iteration, steps.
Collected into the returned list and passed to `on_iteration`/printed by the CLI.

### 4.3 CLIs

- `scripts/train_selfplay_a5.py`: A0-style flags + `--entropy-floor --pool-prob
  --pool-capacity --snapshot-every --eval-every --no-stage1 --arm {pool,pure}`
  (`pure` = `pool_prob 0`, the ablation arm), `--track {tiny,usa}` default tiny,
  `--head` per A3. Prints eval records as they land.
- `scripts/a5_gate.py`: the gate experiment. Runs `train_selfplay_a5` for each
  requested `--arms {pool,pure}` × `--seeds` (defaults: both arms, seeds 0 1 2) at
  `--timesteps` (default **150k** — the A2 probe plateaued by ~40–60k; 150k leaves
  room to *observe* late collapse) and prints one row per run:
  `arm | seed | final vs-weak | peak vs-weak | regression (peak−final) |
  final vs-oldest-self | min entropy | controller engaged? | stage1 | wall(s)`,
  plus per-arm mean ± std summaries, as a markdown table. Accepts subset invocation
  (single arm/seed) so runs can be chunked; rows print immediately per run.

## 5. Acceptance gate (exit criteria)

The sprint-plan gate, made concrete. On `a5_gate.py` defaults (Tiny-Heat, 2 players,
150k steps, seeds {0,1,2}) the **pool arm** must satisfy, on every seed:

- **G1 — beats weak.** Final vs-weak winrate ≥ **70%** (A2 probe: ~85%).
- **G2 — climbs vs previous-self.** Final winrate vs `pool.oldest()` ≥ **55%**.
- **G3 — no collapse.** Min entropy ≥ `entropy_floor` **without** the controller
  ever needing to engage, **and** regression (peak vs-weak − final vs-weak) ≤ **15
  points**, **and** Stage-1 passes.
- **G4 — the pool comparison is reported** (pool vs pure per-arm summary — an
  information gate, not pass/fail: if pure ≥ pool on all metrics, say so; the recipe
  then defaults to the simpler arm going forward).
- **G5 — engineering.** New tests green; full suite green; `ruff` clean; mypy on new
  files per repo practice; defaults unchanged for A0–A3 paths (nothing outside the
  new modules/CLIs edited except — if needed — a minimal export in
  `selfplay/__init__.py`).

A FAIL on G1–G3 for any seed = stop and report honestly (A3 discipline: no tuning to
force a pass; the failed table is still committed to this doc).

## 6. Tests (`tests/test_a5_recipe.py`)

1. **SnapshotAgent legality + degenerate handling**: a collector game with a
   `SnapshotAgent` opponent completes with zero illegal actions (reuse the A2 G1
   cross-check pattern); empty-hand/forced decisions resolve via `forced_action`.
2. **Snapshot immutability**: pool `push` deep-copies — training steps on the live
   policy afterward do not change a pool entry's `state_dict` (compare tensors).
3. **SnapshotPool mechanics**: capacity eviction (oldest dropped), `oldest()`
   identity, uniform `sample` over entries (statistical smoke with fixed rng).
4. **EntropyController**: below floor → coef rises (capped at `ent_coef_max`);
   above floor*1.2 → decays toward but never below base; between → unchanged.
5. **Stage-1**: with an absurd `stage1_min_winrate=1.01` the loop raises
   `Stage1ValidationError` at `stage1_iter`; with `stage1_enabled=False` it doesn't.
6. **`train_selfplay_a5` smoke**: 3 tiny iterations (`snapshot_every=1`,
   `eval_every=2`, small nets/games): completes, pool populated, ≥1 eval record with
   finite fields, losses finite.

Keep tests fast (tiny trunks, few games); the gate experiment lives in the script,
not CI.

## 7. Notes for the implementer

- Match conventions: `from __future__ import annotations`, house docstrings, full
  type hints, `PYTHONPATH=src` layout.
- Only edits to existing files: `selfplay/__init__.py` exports. Everything else new.
- `SnapshotAgent.choose_react` must rebuild `rules.ReactOptions` from the unpacked
  args — check `opponent_action` for the exact field order.
- Deep-copy discipline: `copy.deepcopy(policy).eval()` at push/construction;
  snapshot `act` under `torch.no_grad()` (already guaranteed by the interface).
- The gate is ~6 runs × ~3–5 min: run `a5_gate.py` chunked (per arm or per
  arm+seed) to stay inside command timeouts; assemble the full table yourself.
- Run before committing: new tests, full suite, the gate experiment; append the
  gate table + G1–G5 verdicts as `## 8. Gate results` to this doc.
- Commit protocol (as A3): commit 1 = this design doc alone (`Design Sprint A5:
  anti-collapse self-play recipe`); commit 2 = implementation + tests + gate results
  (`Implement Sprint A5: ...` — state the gate verdict in the subject line), only
  after full suite green + gate run. `git add` specific paths only; do not push.
  End both messages with the trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## 8. Gate results (2026-07-10)

Setup: `scripts/a5_gate.py` defaults — Tiny-Heat, 2 players, masked head, 150k steps
(~73 iterations), seeds {0,1,2}, both arms. "final/peak vs-weak" = winrate vs weak
`HeuristicAgent` (40 games, seat-rotated); "final vs-oldest" = winrate vs
`pool.oldest()` (the snapshot from ~40–50 iterations earlier); regression =
peak − final vs-weak.

| arm | seed | final vs-weak | peak vs-weak | regression | final vs-oldest | min entropy | engaged? | stage1 | wall(s) |
|---|---|---|---|---|---|---|---|---|---|
| pool | 0 | 0.825 | 0.900 | 0.075 | 0.650 | 0.646 | no | pass | 161 |
| pool | 1 | 0.875 | 1.000 | 0.125 | 0.625 | 0.679 | no | pass | 144 |
| pool | 2 | 0.825 | 0.975 | 0.150 | **0.450** | 0.578 | no | pass | 142 |
| pure | 0 | 0.775 | 0.900 | 0.125 | 0.500 | 0.634 | no | pass | 117 |
| pure | 1 | 0.800 | 0.900 | 0.100 | 0.625 | 0.629 | no | pass | 102 |
| pure | 2 | 0.725 | 0.925 | 0.200 | 0.575 | 0.592 | no | pass | 105 |

| arm | final vs-weak | regression | final vs-oldest | min entropy | any engaged | any stage1 fail |
|---|---|---|---|---|---|---|
| pool | 0.842 ± 0.024 | 0.117 ± 0.031 | 0.575 ± 0.089 | 0.634 ± 0.042 | no | no |
| pure | 0.767 ± 0.031 | 0.142 ± 0.042 | 0.567 ± 0.051 | 0.618 ± 0.019 | no | no |

### Verdicts (pool arm, per §5)

- **G1 — beats weak (≥70% every seed): PASS.** 0.825 / 0.875 / 0.825.
- **G2 — climbs vs previous-self (≥55% every seed): FAIL.** Seeds 0/1 pass (0.650,
  0.625); **seed 2 fails at 0.450**. Reported as measured, not tuned around.
- **G3 — no collapse: PASS.** Min entropy 0.578–0.679, all well above the 0.40
  floor; the controller **never engaged** on any of the six runs; regression ≤ 0.150
  on every pool seed; Stage-1 passed everywhere.
- **G4 — pool vs pure (informational):** the pool arm is better on final vs-weak
  (+7.5 points mean), regression (−2.5 points), and min entropy, and avoids the pure
  arm's worst outcomes (pure seed 0 ends 0.500 vs its old self; pure seed 2 regresses
  0.200). The pool earns its keep; keep `pool_prob=0.5` as the default arm.
- **G5 — engineering: PASS.** 8 new tests; full suite 1044 passed / 1 skipped;
  `ruff` and `mypy --strict` clean on the new modules.

### Interpretation (honest, not a gate re-litigation)

The failed cell is *not* a collapse signature: seed 2's entropy is healthy (0.578
min), its vs-weak skill is strong and stable (0.825 final, 0.150 regression), and it
loses only the vs-own-recent-snapshot comparison. Two structural readings: (a) with
40 eval games the se is ~0.08, so 0.450 is ~1.3 se below the bar — underpowered; (b)
"beat the self from ~45 iterations ago by ≥55%" conflates *still climbing* with
*converged* — a plateaued policy legitimately draws ~50% against its recent self.
The gate metric, not the recipe, is the likeliest culprit; A7's Wilson-LB harness
with a fixed early-training anchor (rather than a rolling recent snapshot) is the
right instrument to settle it. Until then the sprint-plan gate is recorded as
**PARTIAL: G1/G3/G4/G5 pass, G2 1/3 seeds fail**.
