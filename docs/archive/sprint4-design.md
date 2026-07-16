# Sprint 4: Simulation — Implementation Plan

## Overview

Sprint 4 adds a batch simulation layer on top of the completed engine (Sprints 1–2) and agents (Sprint 3). Three deliverables, in strict dependency order:

1. `src/heat/simulation/runner.py` — batch runner (sequential + parallel via `concurrent.futures`).
2. `src/heat/simulation/stats.py` — aggregate statistics over a list of per-game outcomes.
3. `scripts/run_simulation.py` — CLI tying both together.

The runner produces a flat, fully-picklable list of `GameOutcome` records; `stats.py` is a pure aggregation layer that takes that list and never touches the engine. This separation is deliberate: it keeps `stats.py` trivially testable on hand-built data and keeps the runner free of presentation/aggregation logic.

**Performance target:** <10 ms per game with random agents, `logging_enabled=False`. The runner must construct every `Game` with `logging_enabled=False` by default.

---

## Key facts established from the codebase (do not re-derive)

- `Game(track, agents, player_names=None, logging_enabled=True)`; `.run() -> GameResult`; `.state` is a `GameState`.
- `GameResult`: `finish_order: list[int]` (player_ids best→worst), `total_rounds: int`, `event_log` (empty when logging disabled). The winner is `finish_order[0]`.
- After `.run()`, `game.state.players` holds final `PlayerState`s. Relevant persistent fields: `player_id`, `name`, `position`, `lap`, `gear`, `heat_available` (= `len(heat_pool)`), `finished`, `finish_order` (1-based rank).
- **There is no cumulative "heat spent" counter anywhere in the engine.** Cooled heat returns to the pool, and spent heat leaves the pool into the deck. The only robust, zero-engine-change signal available after a game is `heat_available` (heat remaining in pool at game end). Heat efficiency must be defined in terms of this remaining-heat figure (see Step 2). Do **not** add a counter to the engine.
- Each player starts with 6 heat cards (`create_heat_cards`).
- Agents carry per-game RNG/state: `RandomAgent(seed=..., name=...)` holds a `random.Random`; reusing the same instance across games would continue its RNG stream. **Therefore the runner must accept agent *factories* (callables returning fresh agents), not agent instances.**
- `Track` is mutable; `track.laps` can be overridden (as `scripts/run_race.py` does). `load_track_by_name("usa")` / `("silverstone")`.
- Tests run via `PYTHONPATH=src python -m pytest tests/ -q`; ~335 existing tests must stay green.
- Conventions to match: `from __future__ import annotations`, module docstring with usage example, plain/frozen dataclasses for result structures, full type hints, `_`-prefixed private helpers, `sys.path.insert(...)` pattern in `scripts/`.

---

## What NOT to do

- **Do not modify the engine or models** (`src/heat/engine/**`, `src/heat/models/**`). No new fields, no new counters. The runner works only with what `GameResult` and the post-game `GameState` already expose.
- **Do not modify the agents** (`src/heat/agents/**`).
- **Do not add any ML, features, training, or numpy modeling** — that is Sprint 5. `numpy` may be used only if convenient for stats math, but plain Python is sufficient and preferred for picklability/simplicity.
- **Do not enable logging in batch runs** (kills the perf target).
- **Do not break the existing ~335 tests.** New modules live under a new `simulation/` package and new test files; nothing existing should need editing.

---

## Deliverable 1 — `src/heat/simulation/runner.py`

### Package setup
Create `src/heat/simulation/__init__.py` (module docstring + re-exports of the public API: `GameOutcome`, `PlayerOutcome`, `run_batch`, `run_single_game`). `src/heat/simulation/__init__.py` already exists but is empty — populate it.

### Data structures (plain frozen dataclasses)

```python
@dataclass(frozen=True)
class PlayerOutcome:
    player_id: int
    name: str
    agent_type: str          # e.g. "RandomAgent" / "HeuristicAgent" (agent.__class__.__name__)
    finish_position: int     # 1-based rank (1 = winner); from finish_order
    final_lap: int
    final_position: int      # track space index at game end
    heat_remaining: int      # heat_available at game end (pool size)

@dataclass(frozen=True)
class GameOutcome:
    game_index: int
    seed: int | None         # the derived per-game seed actually used
    num_players: int
    winner_id: int           # finish_order[0]
    winner_name: str
    finish_order: tuple[int, ...]   # tuple (frozen + picklable)
    total_rounds: int
    players: tuple[PlayerOutcome, ...]   # ordered by player_id
```

All fields are primitives/tuples → fully picklable, safe to return from `ProcessPoolExecutor`.

### Picklability & the factory model

`concurrent.futures.ProcessPoolExecutor` pickles the callable and its args to send to workers, and pickles return values back. Constraints this imposes:

- The **worker function must be a top-level module function** (not a closure/lambda/local) so it is importable by reference in the child process.
- **Agent factories must be picklable.** A factory is a callable returning a fresh agent. Picklable forms: a top-level function, or a `functools.partial` of a top-level constructor. The seed is injected by the runner at call time, not baked into the factory — see signature below. Document in the docstring that lambdas/local closures will fail under `ProcessPoolExecutor` and that such callers must use `parallel=False`.
- `GameOutcome`/`PlayerOutcome` are picklable (primitives only).

### Agent factory signature

A factory is `Callable[[int, int | None], Agent]` taking `(player_id, seed)` and returning a fresh agent:
- `player_id` lets the factory set a sensible default name.
- `seed` is the per-player derived seed the runner supplies; seed-less agents (e.g. `HeuristicAgent`) ignore it.

Provide two small top-level helper factory builders in the module so callers (and the CLI) have picklable factories out of the box:

```python
def random_agent_factory(name=None) -> Callable[[int, int | None], Agent]
def heuristic_agent_factory(name=None) -> Callable[[int, int | None], Agent]
```

Each returns a picklable callable. Implementation note: define a top-level `_make_random(player_id, seed, name)` / `_make_heuristic(player_id, seed, name)` and return `functools.partial(_make_random, name=name)` so the result is picklable.

### Determinism / seeding across workers

- `run_batch(..., seed=None)`: when `seed` is given, the runner derives a **per-game base seed** deterministically, e.g. `game_seed = seed + game_index * 1000` (large stride to avoid overlap between games). Per-player seeds within a game: `game_seed + player_id`.
- The runner seeds the global `random` module **inside the worker** at the start of each game (`random.seed(game_seed)`) because the engine and some agents may use the global RNG (the integration tests do `random.seed(...)` before each game — mirror that). It also passes the derived per-player seed into each factory call so seeded agents are reproducible.
- This makes results **independent of execution order**, so sequential and parallel runs with the same `seed` produce **identical** outcomes (per-game determinism keyed only on `game_index`, not on which worker ran it).
- When `seed is None`, runs are nondeterministic (each game seeded from entropy); `GameOutcome.seed` records `None`.

### Public functions

```python
def run_single_game(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    game_index: int,
    base_seed: int | None,
    laps: int | None = None,
) -> GameOutcome
```
Top-level (picklable) worker. Steps:
1. Compute `game_seed` from `base_seed`/`game_index` (or `None`).
2. If `game_seed is not None`: `random.seed(game_seed)`.
3. Apply `laps` override to the track locally if provided (each process gets its own pickled copy; mutation is process-local — document this).
4. Instantiate fresh agents: `agents = [factory(pid, per_player_seed) for pid, factory in enumerate(agent_factories)]`.
5. `game = Game(track, agents, logging_enabled=False)`.
6. `result = game.run()`.
7. Build `PlayerOutcome`s from `game.state.players` + `result`, and the `GameOutcome`. `agent_type = type(agents[pid]).__name__`. `finish_position` = the player's rank in `result.finish_order` (index+1) or `player.finish_order`.
8. Return the `GameOutcome`.

```python
def run_batch(
    track: Track,
    agent_factories: Sequence[AgentFactory],
    num_games: int,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    seed: int | None = None,
    laps: int | None = None,
    progress: bool = False,
) -> list[GameOutcome]
```
- Validates: `1 <= len(agent_factories) <= 6`; `num_games >= 1`.
- Sequential path (`parallel=False` or `num_games == 1`): a plain loop calling `run_single_game`, returning outcomes ordered by `game_index`.
- Parallel path: `ProcessPoolExecutor(max_workers=max_workers)`, submit one `run_single_game` per `game_index`, collect results with `as_completed`, **re-sort by `game_index`** before returning (futures complete out of order).
- `progress`: optional `tqdm` wrapper (tqdm is already in the tech stack); guard the import so it's optional and never required for tests.
- **Fallback:** wrap the parallel submission in a `try/except` for pickling errors (`TypeError`/`PicklingError`). On failure, log a warning and transparently fall back to the sequential path so non-picklable factories (lambdas) still work. Document this fallback.

### Edge cases the runner must handle
- **Single game** (`num_games == 1`): always sequential regardless of `parallel`.
- **Single player** (1 factory): legal; the lone player is the winner. Assert `1 <= n <= 6`.
- **`parallel=False`**: pure sequential, no executor created.
- **Very large N**: futures collected via `as_completed` and sorted at the end; memory bounded by `num_games` small dataclasses.
- **Agents needing reseeding**: handled by factory-per-game + derived seeds — never reuse an agent instance.
- **`laps` override** applied per game, process-locally.

### Tests for Deliverable 1 → `tests/test_runner.py`
Assert:
- `run_batch` returns exactly `num_games` outcomes, ordered by `game_index`.
- Each `GameOutcome` has exactly one winner; `winner_id == finish_order[0]`; `set(finish_order) == set(range(num_players))`; `len(players) == num_players`.
- **Determinism**: two `run_batch(..., seed=42, parallel=False)` calls produce identical outcomes. Same for `parallel=True`.
- **Sequential ≡ parallel**: `run_batch(seed=7, parallel=False)` and `run_batch(seed=7, parallel=True)` produce the **same per-game `finish_order` and `total_rounds`**. Core correctness guarantee.
- **Single game**: `num_games=1` returns one outcome; works with `parallel=True`.
- **Single player**: 1 factory → winner is player 0, `finish_order == (0,)`.
- **Picklable factories**: `random_agent_factory()` / `heuristic_agent_factory()` survive a `pickle.dumps`/`loads` round-trip.
- **Fallback path**: passing a lambda factory with `parallel=True` does not crash (falls back to sequential) and still returns N valid outcomes. Keep N small (4–8 games) to stay fast.
- **`heat_remaining`** is in range `0..6` for every `PlayerOutcome`.

After writing: **run the full suite to confirm no regressions.**

---

## Deliverable 2 — `src/heat/simulation/stats.py`

Pure aggregation. Takes `list[GameOutcome]` (plus optional config) and returns a summary dataclass. **No engine imports, no I/O.** Easily unit-tested on hand-built outcomes.

### Heat-efficiency definition (important, stated explicitly)
Because the engine exposes no cumulative heat-spent counter, define **heat efficiency** in terms of the post-game `heat_remaining` field already captured in `PlayerOutcome`:

- `avg_heat_remaining` per agent group = mean of `heat_remaining` over all that group's player-games.
- Document in the module docstring that this is "heat left in the pool at game end" — a proxy. Keep the metric simple and honest; do not invent a spent-heat figure. If a future sprint adds a real counter, this is the single place to extend.

### Data structures

```python
@dataclass(frozen=True)
class AgentStats:
    agent_type: str
    games_played: int
    wins: int
    win_rate: float                         # wins / games_played
    finish_position_counts: dict[int, int]  # rank -> count
    avg_finish_position: float
    avg_heat_remaining: float

@dataclass(frozen=True)
class SimulationStats:
    num_games: int
    num_players: int
    avg_rounds: float
    min_rounds: int
    max_rounds: int
    per_agent: dict[str, AgentStats]
    win_counts_by_player_id: dict[int, int]
    finish_distribution_by_player_id: dict[int, dict[int, int]]
```

### Grouping decision
Aggregation supports **two grouping keys**, controlled by a parameter:
- `by="player_id"` — groups by seat (player slot).
- `by="agent_type"` — groups by `PlayerOutcome.agent_type`. Useful for "Heuristic vs Random" headline numbers.

Default `by="agent_type"`. agent_type grouping aggregates across seats (this is what the Heuristic-vs-Random sanity check needs).

### Public functions

```python
def aggregate_stats(
    outcomes: Sequence[GameOutcome],
    *,
    by: str = "agent_type",
) -> SimulationStats
```
- Validates `outcomes` non-empty (raise `ValueError` on empty; test it).
- Computes round stats from `total_rounds`.
- For each grouping key, tallies `games_played`, `wins`, `finish_position_counts`, `avg_finish_position`, `avg_heat_remaining`.
- `win_rate = wins / games_played`.

```python
def format_summary(stats: SimulationStats) -> str
```
- Returns a human-readable multi-line table string (no printing inside — caller prints). Used by the CLI. Keep formatting in `stats.py` so the CLI stays thin and the formatter is testable.

Private helpers `_tally_finishes`, `_mean`, etc. (`_`-prefixed).

### Edge cases
- **Single outcome**: all stats well-defined; `min_rounds == max_rounds == avg_rounds`.
- **Single player games**: every game has one winner = player 0; `win_rate == 1.0` for that group.
- **Empty list**: raise `ValueError`.
- **Ranks**: `finish_position` ranges `1..num_players`; default missing ranks to 0 via `dict.get` and test it.

### Tests for Deliverable 2 → `tests/test_stats.py`
Build `GameOutcome`/`PlayerOutcome` **by hand** (no engine) so math is verifiable:
- **Win-rate math**: 10 hand-built games where player 0 wins 7 → `win_rate == 0.7`.
- **Finish-position distribution** and `avg_finish_position` match hand-computed values.
- **Round stats**: `avg_rounds`, `min_rounds`, `max_rounds` correct on a known list.
- **`avg_heat_remaining`**: correct mean over hand-set values.
- **Grouping**: `by="agent_type"` aggregates two seats of the same type into one group; `by="player_id"` keeps them separate.
- **Empty input** raises `ValueError`.
- **Single outcome / single player** edge cases.
- **`format_summary`** returns a non-empty `str` containing expected labels — smoke-level assertion.
- **Integration**: small real `run_batch` (20 games, fixed seed, `parallel=False`) → `aggregate_stats`; assert `sum(per_agent.wins) == num_games` and total `games_played == num_games * num_players`.
- **Heuristic-beats-random sanity check**: ~50–100 games, 2 heuristic + 2 random seats, fixed seed; aggregate `by="agent_type"`; assert `HeuristicAgent` group `win_rate > 0.55` (robust threshold at the chosen N).

After writing: **run the full suite to confirm no regressions.**

---

## Deliverable 3 — `scripts/run_simulation.py`

CLI mirroring `scripts/run_race.py` conventions: shebang, module docstring with usage examples, `sys.path.insert(0, .../src)`, `argparse`, `main()` + `if __name__ == "__main__"`.

### Arguments
- `--games N` (int, default e.g. 100) — number of games.
- `--track NAME` (default `"usa"`).
- `--laps N` (int, default None → track default).
- `--heuristic N`, `--random N` — agent slot counts (mirror `run_race.py` logic, enforce 1–6 total).
- `--players N` — shortcut for all-heuristic (like `run_race.py`).
- `--names A,B,C` — optional custom names.
- `--seed N` (default None) — base seed for determinism.
- `--no-parallel` (store_true) — force sequential.
- `--workers N` (default None) — `max_workers`.
- `--group {agent_type,player_id}` (default `agent_type`).
- `--progress` (store_true) — show tqdm bar.

### Flow
1. Parse args; validate player counts (1–6) reusing the `run_race.py` pattern (stderr + `sys.exit(1)` on bad input).
2. `track = load_track_by_name(args.track)` (catch `FileNotFoundError` → stderr + exit 1).
3. Build a list of **picklable agent factories** (using `random_agent_factory` / `heuristic_agent_factory`), one per seat, applying names.
4. `outcomes = run_batch(track, factories, args.games, parallel=not args.no_parallel, max_workers=args.workers, seed=args.seed, laps=args.laps, progress=args.progress)`.
5. `stats = aggregate_stats(outcomes, by=args.group)`.
6. `print(format_summary(stats))`. Optionally print timing (`time.perf_counter()` around `run_batch`) and games/sec.

### Tests for Deliverable 3 → `tests/test_run_simulation_cli.py`
- **`main()` invocation**: monkeypatch `sys.argv` to `["run_simulation.py", "--games", "5", "--track", "usa", "--seed", "1", "--no-parallel"]`, call `main()`, assert it runs without error and prints a summary (capture stdout via `capsys`). Keep `--games` tiny for speed.
- **Subprocess smoke test (optional, mark slow)**: `subprocess.run([sys.executable, "scripts/run_simulation.py", "--games", "3", "--no-parallel"], env={PYTHONPATH=src})`, assert returncode 0 and non-empty stdout.
- **Bad input**: `--players 7` → exit code 1 / `SystemExit`.

After writing: **run the full suite to confirm no regressions.**

---

## Ordered implementation checklist

**Step 1 — Package + runner data structures + sequential runner.**
Create/populate `src/heat/simulation/__init__.py`, `src/heat/simulation/runner.py` with `PlayerOutcome`, `GameOutcome`, the factory helpers, `run_single_game`, and `run_batch` with the **sequential path only** plus the parallel path stubbed to fall back to sequential. Implement seeding/determinism. → **Write `tests/test_runner.py` (counts, one-winner, determinism sequential, single game, single player, picklable factories, heat range); run full suite to confirm no regressions.**

**Step 2 — Parallel execution path.**
Add `ProcessPoolExecutor` parallel path with `as_completed` + re-sort by `game_index`, optional `tqdm`, and the pickling-error fallback to sequential. → **Extend `tests/test_runner.py` (sequential≡parallel equivalence, parallel determinism, lambda-factory fallback); run full suite to confirm no regressions.**

**Step 3 — Stats module.**
Create `src/heat/simulation/stats.py` with `AgentStats`, `SimulationStats`, `aggregate_stats(by=...)`, `format_summary`, private helpers, and the documented heat-efficiency-via-`heat_remaining` definition. → **Write `tests/test_stats.py` (win-rate math, finish distribution, round stats, heat mean, grouping, empty/single edge cases, format_summary smoke, real-batch integration, heuristic-beats-random sanity); run full suite to confirm no regressions.**

**Step 4 — CLI.**
Create `scripts/run_simulation.py` wiring track loading, factory construction, `run_batch`, `aggregate_stats`, `format_summary`, plus optional timing. → **Write CLI tests (`main()` via monkeypatched argv + capsys, bad-input `SystemExit`, optional subprocess smoke); run full suite to confirm no regressions.**

**Step 5 — Final verification.**
Run the entire suite (`PYTHONPATH=src python -m pytest tests/ -q`) and confirm all prior tests plus the new ones pass. Optionally run `scripts/run_simulation.py --games 1000 --random 4 --seed 0` and eyeball games/sec against the <10 ms target (informational).

---

## New files created (summary)
- `src/heat/simulation/__init__.py` (populate existing empty file)
- `src/heat/simulation/runner.py`
- `src/heat/simulation/stats.py`
- `scripts/run_simulation.py`
- `tests/test_runner.py`
- `tests/test_stats.py`
- `tests/test_run_simulation_cli.py`
