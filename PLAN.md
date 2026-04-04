# HEAT Board Game AI - Implementation Plan

## Context

Build a simulation engine for the board game **HEAT: Pedal to the Metal** (Days of Wonder, 2022), then layer machine learning (reinforcement learning) on top to discover optimal strategies.

The approach is: **game engine first, ML second** — the standard pattern for game AI research.

---

## Project Structure

```
heat-game/
├── pyproject.toml
├── src/heat/
│   ├── models/        # Data models (cards, player state, track, game state)
│   ├── engine/        # Game logic (orchestrator, phases, rules, events)
│   ├── agents/        # Player strategies (random, heuristic, ML)
│   ├── tracks/        # Track JSON loader
│   ├── simulation/    # Batch runner + stats
│   └── ml/            # Phase 2: features, model, training
├── tracks/            # Track data files (JSON)
├── tests/             # pytest tests
└── scripts/           # CLI entry points
```

---

## Sprint 1: Data Models

1. **`models/cards.py`** — `Card` (frozen dataclass: type, value, id), `CardType` enum (SPEED/HEAT/STRESS), `Deck` (draw/discard piles with reshuffle)
2. **`models/track.py`** + **`tracks/loader.py`** — `Track`, `Space`, `Corner` dataclasses. Tracks loaded from JSON. Linear space sequence with corner overlays and lane counts.
3. **`models/player_state.py`** — Per-player: deck, hand (7 cards), gear (1-4), position, lap, heat_pool (6 heat cards), spun_out flag
4. **`models/game_state.py`** — Full state: track, all players, round number, current phase, turn order, event log

## Sprint 2: Engine

5. **`engine/rules.py`** — Pure functions: `legal_gear_shifts()`, `legal_card_plays()` (itertools.combinations), `corner_heat_cost()`, `slipstream_eligible()`, `cooldown_amount()` (gear 1→3, gear 2→1, else 0)
6. **`engine/events.py`** — `GameEvent` dataclass for full game replay/training data
7. **`engine/phases.py`** — Each phase as a pure function: `(state, player_id, decision) → list[GameEvent]`
   - 9 phases per round: Shift Gears → Play Cards → Reveal & Move → Adrenaline → React → Slipstream → Check Corner → Discard → Replenish
   - Gear shift/card play are **simultaneous** (all players decide secretly, then reveal)
   - Slipstream runs **back-to-front** (trailing players first)
8. **`engine/game.py`** — `Game` orchestrator: round loop, collects decisions from agents, applies via phases, checks win condition

## Sprint 3: Agents

9. **`agents/base.py`** — Abstract `Agent` with methods: `choose_gear()`, `choose_cards()`, `choose_adrenaline()`, `choose_react()`, `choose_slipstream()`, `choose_discard()`. Each receives full `GameState` + list of legal options.
10. **`agents/random_agent.py`** — Uniform random from legal options. First end-to-end validation.
11. **`agents/heuristic_agent.py`** — Rule-based: match gear to hand values, avoid corners when low on heat, cooldown aggressively in low gears, take free slipstreams.

## Sprint 4: Simulation

12. **`simulation/runner.py`** — Batch runner with `concurrent.futures` for parallel games
13. **`simulation/stats.py`** — Win rates, position distributions, heat efficiency
14. **`scripts/run_simulation.py`** — CLI: run N games, configurable agents/tracks

## Sprint 5: ML (Phase 2)

15. **`ml/features.py`** — GameState → fixed-size feature vector (~60-80 floats): hand histogram, gear, position, heat pool, opponent positions, track features (distance to corner, speed limit)
16. **`ml/model.py`** — MLP (3-4 layers, 128-256 units) with separate heads per decision type (gear: 4-way softmax, cards: score-per-card + combination softmax, react: 3-way, slipstream: binary)
17. **Gymnasium wrapper** — `HeatEnv(gym.Env)`: each `step()` = one decision point, auto-advances non-decision phases
18. **`ml/training.py`** — PPO via Stable-Baselines3, self-play curriculum (start vs random, gradually mix trained copies)
19. **`agents/ml_agent.py`** — Loads trained model, uses it for decisions

---

## Key Design Decisions

- **Cards as frozen dataclasses** with unique IDs for ML tracking
- **Heat pool is actual Card objects** (not a counter) — faithfully models hand clogging and running out of heat
- **Event sourcing** — every state change logged for replay + training data (toggleable for speed)
- **Engine always enumerates legal moves** — agents never need to understand rules, invalid moves impossible
- **Simultaneous phases** collected before applying — fair for ML (no peeking at opponents' choices)
- **Performance target**: <10ms per game with random agents → 100K training games in ~17 min

## Tech Stack

- **Core**: Python 3.10+, dataclasses, pytest, numpy, tqdm
- **ML**: PyTorch, gymnasium, stable-baselines3, tensorboard
- **Why PPO**: discrete variable action space, partial observability (hidden hands), moderate episode length, stable training. MCTS/AlphaZero not ideal due to randomness + hidden info.

## Track Data

Create at least one track (e.g., USA) as JSON with space list, corner definitions (start/end/speed_limit), lane counts, and start positions. Transcribed from the physical board.

---

## Verification Plan

1. **Unit tests** for each module (cards, rules, phases, track loading)
2. **Integration test**: Run 1000 games with random agents — zero crashes, all games terminate, exactly one winner per game
3. **Heuristic vs Random**: Heuristic agent should win >60% against random (validates strategy matters)
4. **ML training smoke test**: Train for 100 episodes, verify loss decreases
5. **ML vs Heuristic**: After full training, ML agent should match or beat heuristic win rate
