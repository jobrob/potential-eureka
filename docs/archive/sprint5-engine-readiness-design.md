# Design: Sprint 5 Engine-Readiness (RNG, Clone, Legal-Action API, Step Driver)

## Summary

Before building the Sprint 5 ML layer (`src/heat/ml/`: features, PyTorch model, `HeatEnv`, PPO),
four interdependent engine gaps must be closed. They all touch the same
`GameState` / `Deck` / `PlayerState` surface and together define the substrate a
Gymnasium environment will sit on. This document designs them as **one coherent
change**, in a risk-ordered, incrementally-tested sequence:

- **A. Injectable RNG** — thread a `random.Random` instance owned by `GameState`
  through `Deck` so shuffles are reproducible and isolated per game/env, removing
  reliance on the module-global `random`.
- **B. Clone / reset primitives** — `clone()` for `Deck`, `PlayerState`,
  `GameState` plus read-only deck accessors, so `env.reset()`, rollout buffering,
  and lookahead can deep-copy state faithfully (including the RNG policy).
- **C. Unified legal-action API** — extract the inline React / slipstream /
  discard legality logic from `game.py` into pure `rules.*` functions so a Gym env
  can query the legal action set at any decision point without driving the round
  loop.
- **D. Step-wise game driver** — invert the monolithic `run_round` into a
  pausable driver (generator-based state machine) that yields at each agent
  decision point, accepts an action, and advances; `HeatEnv.step()` will drive it.
  The existing `Game`/agent path is preserved and re-expressed on top of the driver.

This is a **design-only** document. No ML/PyTorch/Gym code is in scope here (that
is the rest of Sprint 5). The deliverable is the engine substrate only.

**Non-negotiable invariant:** all 390 existing tests must continue to pass, and
`simulation/` and `viewer.py` behavior must be unchanged. Each implementation step
ends by adding tests and running `PYTHONPATH=src python -m pytest tests/ -q`.

---

## Motivation

- Sprint 5 adds `HeatEnv(gym.Env)`. A Gym env interleaves many episodes (and,
  under vectorized training, many envs) in **one process**. The current
  determinism strategy seeds the **module-global** `random` once per game
  (`simulation/runner.py:200`, `random.seed(game_seed)`), which cannot give
  reproducible, isolated streams when episodes interleave — episode B's shuffles
  perturb episode A's stream. (Item A)
- `env.reset()`, PPO rollout buffers, and any one-ply lookahead need a **faithful
  deep copy** of state. No copy method exists today, and `Deck` hides its piles
  behind name-mangled `_draw_pile` / `_discard_pile` with no accessor. (Item B)
- The env needs **one function per decision point** returning the legal action
  set, callable without running the round loop. Today React options, slipstream
  eligibility, and discardable-card computation are built **inline** in `game.py`.
  (Item C)
- `run_round` is **monolithic and agent-pull-driven**: it calls
  `agent.choose_*(...)` itself. A Gym env is **push-driven**: the trainer hands an
  action to `step()`. These control-flow models are incompatible without inverting
  the loop into a pausable driver. (Item D)

---

## Current Behavior (state analysis with file:line references)

### A. RNG today
- `Deck.__init__` shuffles via the module-global RNG: `cards.py:53`
  (`random.shuffle(self._draw_pile)`).
- `Deck.add_to_draw_pile`: `cards.py:86` (`random.shuffle`).
- `Deck._reshuffle`: `cards.py:92` (`random.shuffle`).
- `import random` at module scope: `cards.py:5`.
- **The only engine-side global-RNG use is `Deck` shuffling.** A grep of
  `src/heat` for `random.` shows exactly: `cards.py` (3 shuffles), `random_agent.py`
  (its own injected `random.Random`), and `runner.py` (`random.seed`). The
  spin-out stress-card generation in `step_check_corner` (`phases.py:622-628`)
  uses `state.next_stress_id()` — a **deterministic counter**
  (`game_state.py:69-72`), **not** RNG. (The prompt's mention of "spin-out
  stress-card generation also uses global RNG" does not match the current code;
  this design treats `Deck` as the sole RNG site and notes the discrepancy.)
- `RandomAgent` already injects its own stream: `random_agent.py:21`
  (`self._rng = random.Random(seed)`). `HeuristicAgent` uses no RNG (deterministic).
- Ordering problem: `Deck` instances are created inside `PlayerState.create`
  (`player_state.py:77`, `deck = Deck(all_deck_cards)`), which is called from
  `GameState.create` (`game_state.py:147`) — i.e. **decks exist before the
  `GameState` that would own the RNG exists**. The initial 7-card draw also
  happens there (`player_state.py:87`), but `draw()` itself does not shuffle; only
  the `Deck.__init__` shuffle consumes RNG at creation time.
- Determinism is currently achieved by seeding global `random` per game:
  `runner.py:199-200`, and in tests `test_game.py:183/189`,
  `test_agent_integration.py:21/43/68`.

### B. Clone / accessors today
- No `clone()` / `__deepcopy__` on `Deck` (`cards.py:47-100`), `PlayerState`
  (`player_state.py:18-139`), or `GameState` (`game_state.py:45-158`).
- Deck piles are name-mangled with **no public accessor**: `_draw_pile` /
  `_discard_pile` (`cards.py:51-52`). Only size properties exist
  (`draw_pile_size` `cards.py:56`, `discard_pile_size` `cards.py:60`,
  `total_size` `cards.py:64`) and `__iter__` (`cards.py:94`). Tests reach into the
  mangled fields directly: `test_phases.py:66`, `test_rules.py:108`,
  `test_rules.py:432` — so any accessor work must not break those, and ideally
  gives them a supported alternative.
- `Card` is a frozen dataclass (`cards.py:20`) — shared safely across clones (no
  deep copy of cards needed).
- Transient per-turn fields on `PlayerState` (`player_state.py:53-62`:
  `cards_played`, `boost_used_this_turn`, `speed_from_*`, `slipstream_moved`,
  `cluttered`, `turn_start_position`, `turn_start_lap`) and `GameState.event_log`
  (`game_state.py:64`) plus `_stress_counter` (`game_state.py:67`) need an explicit
  clone policy.

### C. Inline legality today (in `game.py`)
- **React options** are computed inline in `_collect_react_decision`
  (`game.py:359-382`): `gear_cooldown = rules.cooldown_amount(player.gear)`
  (`game.py:362`), `has_adrenaline = rules.adrenaline_eligible(...)`
  (`game.py:363-367`), `max_cooldown = gear_cooldown` (`game.py:370`),
  `can_boost = player.heat_available > 0 and not player.boost_used_this_turn`
  (`game.py:372-374`). Note `max_cooldown` here is gear-only; the +1 adrenaline
  cooldown is applied later inside `step_react` (`phases.py:387`).
- **Slipstream eligibility** is queried inline in the loop: `game.py:276-280`
  (`rules.slipstream_eligible(player, list(self._state.active_players),
  self._state.track)`) before calling `choose_slipstream`.
- **Discardable cards** are filtered inline: `game.py:290-294`
  (`[c for c in player.hand if c.card_type in (CardType.SPEED, CardType.UPGRADE)]`).
  `step_discard` re-validates the same rule (`phases.py:708-714`).
- **Gear** and **card** legality already live in `rules` (`legal_gear_shifts`
  `rules.py:30`, `legal_card_plays` `rules.py:69`) and are called from
  `_collect_gear_decisions` (`game.py:327`) and `_collect_card_decisions`
  (`game.py:345`). Those two are already clean; C only needs to lift React,
  slipstream, and discard to match.
- **Value-redundant card plays:** `legal_card_plays` (`rules.py:69-102`) uses
  `itertools.combinations` over distinct `Card` objects, so two equal-value Speed
  cards yield two distinct-but-equivalent tuples. This is correct for the engine
  (cards have unique ids) but matters for the ML action space — captured as
  guidance in §C, not changed.

### D. Monolithic round loop today
- `run_round` (`game.py:184-313`) does, in order: `compute_turn_order`
  (`game.py:194`), simultaneous `_collect_gear_decisions` + `phase_shift_gears`
  (`game.py:197-198`), simultaneous `_collect_card_decisions` + `phase_play_cards`
  (`game.py:207-208`), then a **sequential per-player loop** (`game.py:212-302`)
  running steps 3–9 with three embedded agent calls: React
  (`_collect_react_decision` `game.py:269`), slipstream (`choose_slipstream`
  `game.py:281`), discard (`choose_discard` `game.py:297`). It also handles
  cluttered-hand skip (`game.py:254-257`), finish-during-move early-replenish
  (`game.py:261-264`, `272-274`), and end-of-round `spun_out` reset + round
  increment (`game.py:306-311`).
- The agent calls are **pull** (`self._agents[pid].choose_*`). A Gym env is
  **push**. Inverting this is the largest structural change.

---

## Proposed Design

### Overview

Add an RNG owned by `GameState` and threaded into `Deck` via a deferred-attach
pattern (A). Add `clone()` to the three mutable models with an explicit RNG and
transient-field policy (B). Lift React/slipstream/discard legality into pure
`rules.*` functions and refactor `game.py` to call them (C). Introduce a new
generator-based `RoundDriver` that pauses at each decision point; reimplement
`Game.run_round` on top of it so behavior is identical, and let `HeatEnv` drive
the same generator directly (D).

Files changed: `models/cards.py`, `models/player_state.py`, `models/game_state.py`,
`engine/rules.py`, `engine/game.py`, **new** `engine/driver.py`, plus `simulation/runner.py`
(seed-threading, optional) and new tests. `viewer.py` is untouched (it only calls
`game.run_round()` `viewer.py:550`, which remains).

---

### A. Injectable RNG

**Where the RNG lives.** `GameState` owns the canonical `random.Random`. `Deck`
holds a reference to "its RNG" so its three shuffle sites use it instead of the
global module.

**Solving the creation-ordering problem.** Decks are built in
`PlayerState.create` → `GameState.create`, before the `GameState` exists. Use a
**deferred-attach** pattern: `Deck` accepts an optional `rng` and falls back to a
private local stream; `GameState.create` then re-binds every deck to the game RNG
and re-shuffles deterministically.

```python
# cards.py
class Deck:
    def __init__(
        self,
        cards: list[Card] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._draw_pile: list[Card] = list(cards) if cards else []
        self._discard_pile: list[Card] = []
        self._rng.shuffle(self._draw_pile)

    def attach_rng(self, rng: random.Random, reshuffle: bool = True) -> None:
        """Re-bind this deck to a shared RNG. If reshuffle, re-shuffle the
        draw pile from the new stream so deck order is a deterministic
        function of the game seed (not the pre-attach local stream)."""
        self._rng = rng
        if reshuffle:
            rng.shuffle(self._draw_pile)
```

All three shuffle sites change `random.shuffle(...)` →
`self._rng.shuffle(...)` (`cards.py:53`, `:86`, `:92`). The module-level
`import random` stays (used for the default `random.Random()`).

`GameState` gains an `rng` field and creates/attaches it:

```python
# game_state.py
@dataclass
class GameState:
    ...
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def create(cls, track, num_players, player_names=None,
               logging_enabled=True, seed: int | None = None) -> "GameState":
        rng = random.Random(seed)
        players = []
        for i in range(num_players):
            player = PlayerState.create(i, name=...)   # builds a Deck with a local RNG
            player.deck.attach_rng(rng)                # re-bind + deterministic reshuffle
            if i < len(track.start_positions):
                player.position = track.start_positions[i]
            players.append(player)
        return cls(track=track, players=players, rng=rng,
                   logging_enabled=logging_enabled,
                   starting_player_count=num_players)
```

Key property: after `attach_rng(rng)` re-shuffles, **the entire deck order is a
deterministic function of `seed` alone**, independent of the throwaway local
stream used during `PlayerState.create`. The initial 7-card hand
(`player_state.py:87`) is drawn *after* `attach_rng` runs in `GameState.create`,
so hands are seed-determined too. (Note: `PlayerState.create` currently draws the
hand itself at `:87`. Move that draw to occur after attach — see Implementation
Step A2 — or have `create` accept an optional `rng` so the hand draw uses the
final stream. The design below moves hand-draw responsibility into
`GameState.create` to keep deck order fully seed-determined.)

`Game.__init__` (`game.py:141`) passes a new optional `seed` through to
`GameState.create`. `RandomAgent` keeps its **own independent** stream
(`random_agent.py:21`) — agent decision randomness stays separate from deck
randomness, which is correct (the env will control agent choices itself, and the
runner already seeds agents independently).

**Backward compatibility.** Existing tests seed the **global** `random` before
constructing a `Game` (`test_game.py:183`, `test_agent_integration.py:21/43/68`).
After this change, if `seed=None` is passed (the default), `GameState` builds
`random.Random(None)` — seeded from OS entropy, **not** the global module — so
those tests would lose reproducibility. Two-part compatibility policy:

1. **Preserve the global-seed path during migration.** When `seed is None`,
   `GameState.create` derives its RNG from the *current global random state* so
   that a preceding `random.seed(999)` still determines the game:
   `rng = random.Random(random.random())` when `seed is None`, else
   `random.Random(seed)`. This makes `random.seed(999); Game(...)` reproducible
   exactly as before (the global stream deterministically seeds the game RNG),
   keeping `test_game.py:178-196` and the integration tests green **unchanged**.
2. New code (env, runner) passes an explicit `seed=` and never relies on global
   state. `simulation/runner.py:199-200` is updated to pass `seed=game_seed` to
   `Game(...)` instead of (or in addition to) `random.seed(game_seed)`; the
   `random.seed` line may be kept during transition and removed once all call
   sites are migrated (noted in Step A3).

This satisfies the explicit requirement that existing `random.seed()`-based
determinism tests are **preserved** (option 1) with a documented migration path
(option 2).

---

### B. State clone / reset primitives + deck accessors

**Deck accessors (read-only).**

```python
# cards.py
@property
def draw_pile(self) -> tuple[Card, ...]:
    """Read-only snapshot of the draw pile (bottom..top)."""
    return tuple(self._draw_pile)

@property
def discard_pile(self) -> tuple[Card, ...]:
    return tuple(self._discard_pile)
```

Returning tuples prevents external mutation while exposing contents/counts. The
existing size properties (`draw_pile_size`, etc.) and `__iter__` remain. Tests
that poke `_draw_pile` directly (`test_phases.py:66`, `test_rules.py:108/432`)
keep working (name-mangled attrs still exist); new tests use the accessors.

**Deck.clone.**

```python
# cards.py
def clone(self, rng: random.Random | None = None) -> "Deck":
    """Deep-ish copy: new pile lists (Cards are frozen, shared safely).
    RNG policy: caller supplies the clone's rng (normally the cloned
    GameState's rng). If None, the clone gets a fresh independent
    random.Random() so the two decks do not share a stream."""
    new = Deck.__new__(Deck)
    new._draw_pile = list(self._draw_pile)
    new._discard_pile = list(self._discard_pile)
    new._rng = rng if rng is not None else random.Random()
    return new
```

**PlayerState.clone.** Copies all scalar fields and makes new lists for `hand`,
`heat_pool`, `cooldown_pool`, `cards_played` (Cards shared — frozen). Deck cloned
with the supplied RNG.

```python
# player_state.py
def clone(self, rng: random.Random | None = None) -> "PlayerState":
    return PlayerState(
        player_id=self.player_id, name=self.name,
        deck=self.deck.clone(rng),
        hand=list(self.hand), gear=self.gear, position=self.position,
        lap=self.lap, heat_pool=list(self.heat_pool),
        cooldown_pool=list(self.cooldown_pool),
        spun_out=self.spun_out, finished=self.finished,
        finish_order=self.finish_order,
        cards_played=list(self.cards_played),
        boost_used_this_turn=self.boost_used_this_turn,
        speed_from_cards=self.speed_from_cards,
        speed_from_boost=self.speed_from_boost,
        speed_from_adrenaline=self.speed_from_adrenaline,
        slipstream_moved=self.slipstream_moved,
        cluttered=self.cluttered,
        turn_start_position=self.turn_start_position,
        turn_start_lap=self.turn_start_lap,
    )
```

All per-turn transient fields **are** copied (a clone taken mid-turn must be a
faithful resume point — needed for lookahead). They are not reset.

**GameState.clone.**

```python
# game_state.py
def clone(self, *, copy_event_log: bool = False,
          reseed: int | None = None) -> "GameState":
    """Faithful deep copy.

    RNG policy (explicit, the interaction with item A):
      - reseed is None (default): the clone gets its OWN rng forked
        deterministically from this state's rng via
        random.Random(self.rng.random()). This guarantees the clone is
        independent (mutating one stream never touches the other) AND
        reproducible (forking the same parent rng twice yields the same
        child stream). The parent's rng is advanced by one draw.
      - reseed is an int: the clone's rng = random.Random(reseed), for
        callers (e.g. env.reset()) that want an explicit fresh seed.

    Cloned player decks are bound to the clone's rng.

    event_log: NOT copied by default (it is large, append-only replay
    data; a clone for lookahead/rollout does not need history). When
    copy_event_log=True the list is shallow-copied (GameEvents are
    treated as immutable records and shared).
    """
    new_rng = random.Random(reseed) if reseed is not None \
        else random.Random(self.rng.random())
    new = GameState(
        track=self.track,                       # Track is immutable config; shared
        players=[p.clone(new_rng) for p in self.players],
        round_num=self.round_num,
        current_phase=self.current_phase,
        turn_order=list(self.turn_order),
        event_log=list(self.event_log) if copy_event_log else [],
        logging_enabled=self.logging_enabled,
        starting_player_count=self.starting_player_count,
        rng=new_rng,
    )
    new._stress_counter = self._stress_counter   # preserve id sequence
    return new
```

**RNG-and-clone interaction (the policy item A asks for):** a clone never shares
an RNG object with its parent. By default it gets a deterministically-forked
child stream (`random.Random(parent.rng.random())`), giving independence +
reproducibility. `env.reset()` can instead pass `reseed=<episode_seed>` for an
explicit fresh stream. This is specified, not left implicit.

**`Track` sharing.** `Track` and its `Space`/`Corner` are immutable race
configuration (only `runner.py` mutates `track.laps` once before a game). Clones
**share** the same `Track` object; the design does not deep-copy it. (If a future
caller mutates `track` mid-game this would need revisiting — flagged in Open
Questions.)

`copy.deepcopy` support is **not** added; explicit `clone()` is preferred because
it (a) lets us control the RNG policy, (b) shares frozen `Card`/`Track` objects
instead of needlessly copying them, and (c) is far faster — important for rollout
buffering at training scale.

---

### C. Unified legal-action enumeration API

Add three pure functions to `rules.py`, mirroring the existing
`legal_gear_shifts` / `legal_card_plays` style (stateless, no mutation, return
plain data).

```python
# rules.py

@dataclass(frozen=True)
class ReactOptions:
    """Legal React action envelope for one player at the React decision.
    Mirrors exactly what _collect_react_decision computes today
    (game.py:359-382) so the ML action space and the agent protocol agree."""
    max_cooldown: int        # gear-based cooldown (adrenaline +1 added in step_react)
    can_boost: bool
    has_adrenaline: bool

def legal_react_options(
    player: PlayerState,
    active_players: list[PlayerState],
    starting_player_count: int,
) -> ReactOptions:
    """Pure extraction of game.py:362-374. No behavior change."""
    return ReactOptions(
        max_cooldown=cooldown_amount(player.gear),
        can_boost=(player.heat_available > 0 and not player.boost_used_this_turn),
        has_adrenaline=adrenaline_eligible(
            player, active_players, starting_player_count),
    )

def legal_slipstream(
    player: PlayerState,
    active_players: list[PlayerState],
    track: Track,
) -> bool:
    """Thin pass-through to slipstream_eligible (game.py:276-280), provided
    as the single named decision-point entry the env queries."""
    return slipstream_eligible(player, active_players, track)

def legal_discards(player: PlayerState) -> list[Card]:
    """Speed/Upgrade cards eligible for voluntary discard.
    Pure extraction of game.py:290-294 (and the rule step_discard
    re-validates at phases.py:708-714)."""
    return [c for c in player.hand
            if c.card_type in (CardType.SPEED, CardType.UPGRADE)]
```

**`game.py` refactor (no behavior change).**
- `_collect_react_decision` (`game.py:359-382`) calls `rules.legal_react_options`
  and passes its fields to `choose_react` — identical values, identical
  `max_cooldown` semantics (gear-only; adrenaline +1 still applied in
  `step_react`).
- The slipstream guard at `game.py:276-280` calls `rules.legal_slipstream(...)`.
- The discardable filter at `game.py:290-294` calls `rules.legal_discards(player)`.

Because each function is a literal lift of the current expression, the 390 tests
must still pass; a dedicated **parity test** (Step C2) asserts the new functions
return exactly what the old inline code produced across many random states.

**ML guidance on value-redundant card plays (engine unchanged).**
`legal_card_plays` (`rules.py:69`) returns one tuple per distinct *Card-object*
combination, so two value-3 Speed cards produce two equivalent actions. The
engine is correct (unique ids); the **ML action space** should not treat these as
distinct. Recommended approach for the ML design (not implemented here):
score-per-card with a value-keyed combination head (per `PLAN.md:62`), or
dedupe-by-multiset-of-values when enumerating discrete actions, then map the
chosen value-combination back to concrete cards by picking the
lowest-id representatives. This is **guidance for `ml/model.py` / the env's action
encoding**, recorded so the ML implementer does not double-count, and is
explicitly out of scope for the engine.

---

### D. Step-wise game driver

Introduce a new module `engine/driver.py` providing a **generator-based round
driver** that yields a `Decision` request at each agent decision point and resumes
with the supplied action via `generator.send(...)`. Non-decision phases
auto-advance inside the generator.

**Decision-point protocol.**

```python
# engine/driver.py
from enum import Enum

class DecisionKind(Enum):
    GEAR = "gear"            # simultaneous
    CARDS = "cards"          # simultaneous
    REACT = "react"          # sequential, per player
    SLIPSTREAM = "slipstream"
    DISCARD = "discard"

@dataclass
class Decision:
    """A pause point. The driver yields this; the caller inspects
    `legal` and sends back an action of the matching type."""
    kind: DecisionKind
    player_id: int
    legal: object   # type depends on kind (see table below)

# kind -> legal type -> action type the caller must send():
#   GEAR        legal: list[tuple[int,int]]        action: tuple[int,int]
#   CARDS       legal: list[tuple[Card,...]]       action: tuple[Card,...]
#   REACT       legal: rules.ReactOptions          action: ReactDecision
#   SLIPSTREAM  legal: bool (always True when asked) action: bool
#   DISCARD     legal: list[Card]                  action: list[Card]
```

```python
def run_round_driver(state: GameState):
    """Generator that runs ONE round, pausing at each decision.

    Usage (push model):
        gen = run_round_driver(state)
        decision = next(gen)
        while True:
            action = choose(decision)         # caller supplies
            try:
                decision = gen.send(action)
            except StopIteration:
                break

    Simultaneous phases (gear, cards): the driver yields one Decision per
    active player FIRST (collecting all actions), then applies
    phase_shift_gears / phase_play_cards once with the full decision dict
    — preserving the no-peeking simultaneity guarantee (PLAN.md:75).

    Sequential phases (steps 3-9): the driver runs the existing step_*
    functions in turn order, yielding REACT/SLIPSTREAM/DISCARD Decisions
    only where an agent choice is required, and auto-advancing everything
    else (reveal_and_move, adrenaline, check_corner, replenish, cluttered
    skip, finish-early replenish).
    """
```

The generator body is a **mechanical transcription** of `run_round`
(`game.py:184-313`): same `compute_turn_order`, same gather-then-apply for the two
simultaneous phases, same per-player loop including the cluttered skip
(`game.py:254-257`), finish-early replenish (`game.py:261-264, 272-274`),
slipstream-eligibility guard (now `rules.legal_slipstream`), and end-of-round
`spun_out` reset + `round_num += 1` (`game.py:306-311`). The difference is that
where `run_round` calls `self._agents[pid].choose_*`, the generator `yield`s a
`Decision` and waits for `send`.

**`Game` re-expressed on top of the driver (behavior identical).**

```python
# game.py — run_round becomes a thin pump over the driver
def run_round(self) -> list[GameEvent]:
    log_start = len(self._state.event_log)
    gen = run_round_driver(self._state)
    try:
        decision = next(gen)
        while True:
            action = self._answer(decision)   # dispatch to self._agents[pid].choose_*
            decision = gen.send(action)
    except StopIteration as stop:
        events = stop.value or []             # generator returns its event list
    return events
```

`self._answer(decision)` switches on `decision.kind` and calls the matching agent
method with the same arguments as today. The agent-facing `Agent` protocol
(`game.py:33-101`) and `ReactDecision` (`phases.py:28-42`) are **unchanged**, so
all agents and `viewer.py` (which calls `game.run_round()` at `viewer.py:550`)
work without modification.

**How the env uses it.** `HeatEnv.step(action)` (Sprint 5, not here) holds the
live generator, calls `gen.send(action)` to advance to the next `Decision`,
exposes `decision.player_id` / `decision.legal` for the policy, and starts a new
round generator when one `StopIteration`s. `env.reset()` builds a fresh
`GameState` (with an explicit `seed`) or `state.clone(reseed=...)` (item B) and a
new generator. Non-decision phases auto-advance, exactly matching `PLAN.md:63`.

**Composition with A–C.** The driver uses item C's `rules.legal_*` functions to
populate `Decision.legal` (single source of truth shared with `Game`). It runs
against item A's seeded `state.rng` (shuffles during the round are reproducible).
Item B lets the env reset/fork the `GameState` the driver runs on.

**Equivalence guarantee.** Because `run_round` is reimplemented as a pump over the
generator and the generator is a transcription of the old body, a seeded game run
via the new `Game` must produce a byte-identical event log and finish order to a
game run via the pre-refactor code. Step D2 adds a regression test asserting this
across many seeds (and, transitionally, can compare against a captured golden log).

---

## Alternatives Considered

- **Global-RNG-only with per-episode `random.seed`** (status quo). Rejected:
  interleaved episodes/vector envs in one process cannot get isolated reproducible
  streams; this is the core blocker for item A.
- **`copy.deepcopy` instead of explicit `clone()`.** Rejected: no control over the
  RNG policy, needlessly deep-copies frozen `Card`/`Track`, and is slow at rollout
  scale. Explicit `clone()` is faster and lets us share immutables.
- **Threading RNG as a function parameter through every `Deck` method** rather than
  storing it on the deck. Rejected: `Deck.draw` reshuffles internally
  (`cards.py:74`) deep inside `step_*` call chains; plumbing an `rng` arg through
  every caller is invasive and error-prone. Owning the RNG on the deck (re-bound
  from `GameState`) is localized.
- **Coroutine/`async` step driver** instead of a generator. Rejected: generators
  with `send()` give the exact pause/resume semantics with no event loop; async
  adds complexity SB3/Gym does not need.
- **Explicit state-machine object (enum cursor + step table)** instead of a
  generator. Viable and arguably easier to serialize mid-round, but a far larger
  rewrite and harder to prove equivalent to `run_round`. The generator keeps the
  body a line-by-line transcription, minimizing regression risk. (Recorded as the
  fallback if mid-round serialization is later required.)

---

## Testing Strategy

Each step adds tests and ends by running `PYTHONPATH=src python -m pytest tests/ -q`.

- **A — determinism with injected RNG:**
  - Two `GameState.create(..., seed=42)` produce identical deck orders
    (via new `draw_pile` accessor) and identical initial hands.
  - Different seeds produce different orders (sanity).
  - `Deck.attach_rng` makes deck order a function of the game seed only
    (independent of pre-attach local stream): build the same player twice with
    different transient construction RNG, attach the same seeded rng, assert equal
    order.
  - **Back-compat:** `random.seed(999); Game(...)` twice → identical finish order
    (the existing `test_game.py:178-196` must pass **unchanged**).
- **B — clone independence and equality:**
  - `clone()` deep-equals the source field-by-field (decks, hands, heat pools,
    transient fields, `_stress_counter`).
  - Mutating the clone (draw a card, pay heat, change gear) does **not** affect the
    source, and vice versa.
  - RNG independence: advancing `clone.rng` does not perturb `source.rng`; forking
    the same parent twice yields identical child streams (reproducibility).
  - `reseed=N` gives a deterministic clone stream; `copy_event_log=False` yields an
    empty log while `True` shares the events.
  - Cards are shared (identity), confirming no needless copy.
- **C — legal-action parity:**
  - For many random mid-game states, `rules.legal_react_options` /
    `legal_slipstream` / `legal_discards` return exactly what the original inline
    expressions produced (assert against literal recomputation).
  - After the `game.py` refactor, full suite green (behavior unchanged).
- **D — step-driver vs run_round equivalence:**
  - Same agents + same seed via the new `Game` (pump over driver) produce identical
    `finish_order`, `total_rounds`, and event log to the reference behavior (golden
    log captured before refactor, or compared against a parallel un-refactored copy
    during development).
  - The driver yields the expected `Decision.kind` sequence for a crafted
    deterministic state (e.g. cluttered-hand player skips React/slipstream/discard).
  - Simultaneity preserved: gear/card decisions are all collected before either
    phase applies (no agent sees another's move).
  - Edge cases: finish-during-move triggers early replenish and no further
    decisions for that player; spun-out player forced to gear 1 next round.

Edge cases to cover overall: empty/exhausted deck during clone; clone taken
mid-React (transient fields preserved); single-active-player adrenaline
(`rules.py:417`); slipstream guard on final lap (`rules.py:368`).

---

## Implementation Plan

Ordered to minimize risk: A and B are foundational and additive; C is a pure
refactor; D builds on A–C. Every step ends with new tests + full-suite green.

1. **Step A1 — Deck RNG.** Add `rng` param + `attach_rng` to `Deck`; switch the
   three shuffle sites to `self._rng.shuffle`. Tests: deck-order determinism for a
   given injected `random.Random`; default (no rng) still shuffles. Run suite.
2. **Step A2 — GameState owns RNG.** Add `GameState.rng` field and `seed` param to
   `GameState.create`; attach the game rng to each player's deck and move the
   initial 7-card draw to after attach (so hands are seed-determined). Add `seed`
   passthrough on `Game.__init__`. Implement the `seed=None` → fork-from-global
   back-compat path. Tests: seed reproducibility; existing
   `random.seed()`-based tests unchanged. Run suite.
3. **Step A3 — Runner migration (optional, low-risk).** Update
   `simulation/runner.py:run_single_game` to pass `seed=game_seed` to `Game`;
   keep or remove the `random.seed` line with a comment noting the migration.
   Tests: `test_runner.py` determinism still holds. Run suite.
4. **Step B1 — Deck accessors + `Deck.clone`.** Add `draw_pile` / `discard_pile`
   read-only properties and `clone(rng)`. Tests: accessor snapshots, clone
   independence, card sharing. Run suite.
5. **Step B2 — `PlayerState.clone`.** Tests: field-by-field equality, mutation
   isolation, deck rng binding. Run suite.
6. **Step B3 — `GameState.clone`.** Implement RNG fork/`reseed` policy and
   `copy_event_log` flag; preserve `_stress_counter`. Tests: full-state clone
   equality, rng independence/reproducibility, event-log policy. Run suite.
7. **Step C1 — Add `rules.legal_react_options` / `legal_slipstream` /
   `legal_discards` + `ReactOptions`.** Pure additions, no callers yet. Tests:
   each function vs literal recomputation across random states. Run suite.
8. **Step C2 — Refactor `game.py` to call the new rules functions** at
   `game.py:276-280`, `:290-294`, `:359-382`. No behavior change. Tests: parity +
   full suite green. Run suite.
9. **Step D1 — New `engine/driver.py` with `run_round_driver`.** Transcribe
   `run_round`. Add `Decision` / `DecisionKind`. Drive it from a test harness (not
   yet from `Game`). Tests: decision-sequence and per-decision `legal` correctness
   on deterministic states. Run suite.
10. **Step D2 — Reimplement `Game.run_round` as a pump over the driver.** Add
    `self._answer(decision)` dispatch. Tests: seeded equivalence (finish order,
    rounds, event log) vs golden; `viewer.py` smoke (`test_viewer.py`) unchanged.
    Run suite. **This is the highest-risk step — the equivalence test is the gate.**

---

## Backward-Compatibility Requirements (explicit)

- All 390 existing tests pass unchanged at the end of every step.
- `engine` public behavior (move math, corner/spin-out, finish order) unchanged.
- `simulation/` results unchanged for a given seed (Step A3 must keep
  `test_runner.py` determinism; sequential == parallel still holds).
- `viewer.py` unaffected — it only calls `game.run_round()` (`viewer.py:550`),
  which keeps its signature and behavior.
- `Agent` protocol (`game.py:33-101`) and `ReactDecision` (`phases.py:28-42`)
  unchanged — no agent edits required.
- Existing `random.seed()`-based determinism tests (`test_game.py:178-196`,
  `test_agent_integration.py:21/43/68`) are **preserved** via the `seed=None`
  fork-from-global path; a note in `runner.py` documents the migration toward
  explicit `seed=` for new code.
- Tests that reach into `Deck._draw_pile` / `_discard_pile`
  (`test_phases.py:66`, `test_rules.py:108/432`) keep working (mangled attributes
  remain); new tests prefer the `draw_pile` accessor.

---

## Non-Goals

- **No ML/PyTorch/Gym code.** No `ml/features.py`, `ml/model.py`,
  `ml/training.py`, `HeatEnv`, or `agents/ml_agent.py` here — those are the rest
  of Sprint 5 (`PLAN.md:59-66`). This document delivers only the engine substrate
  the env will sit on.
- **No engine rule changes.** The value-redundant card-play observation is ML
  action-space *guidance*, not an engine change. `legal_card_plays` stays as is.
- **No `copy.deepcopy` protocol.** Explicit `clone()` only.
- **No mid-round serialization** of the driver (the generator is not pickled
  mid-round). If training later needs that, switch to the explicit state-machine
  fallback (see Alternatives) — out of scope now.

---

## Risk / Sequencing Notes

- **Biggest risk is Step D2** (the round-loop inversion). Mitigation: the driver
  is a line-by-line transcription of `run_round`, and the gate is a seeded
  equivalence test comparing finish order, round count, and the full event log
  against a golden captured before the refactor. If it diverges, the diff in the
  event log localizes the exact phase/step that drifted.
- **Subtle RNG-ordering risk (A2):** if the initial hand is drawn before
  `attach_rng` re-shuffles, deck order would depend on the throwaway local stream
  and break seed-determinism. Mitigation: Step A2 explicitly moves the hand draw
  after attach; the A-determinism test (same seed → same hands) catches a
  regression here immediately.
- **Back-compat RNG risk (A2):** the `seed=None` fork-from-global path is what
  keeps the legacy `random.seed()` tests green. If that path is mis-wired, those
  exact tests fail loudly in the same step — a built-in tripwire.
- **Clone-aliasing risk (B):** a shallow copy that shares a `hand`/`heat_pool`
  list would silently corrupt rollouts. Mitigation: the B mutation-isolation tests
  mutate the clone and assert the source is untouched (and vice versa).
- **Parity risk (C):** an off-by-one in `max_cooldown` (gear-only vs +adrenaline)
  would change React behavior. Mitigation: `ReactOptions.max_cooldown` is
  documented as gear-only (adrenaline +1 still added in `step_react`
  `phases.py:387`), and the parity test compares against the literal old
  expression.
- **Sequencing safety:** A and B are purely additive (new params with defaults,
  new methods) — they cannot break existing callers. C is a behavior-preserving
  refactor guarded by a parity test. D is last because it depends on C's
  `rules.legal_*` and A's seeded state. Each step's full-suite gate means a
  regression is caught at the step that introduced it, never compounded.

---

## Open Questions

- Should `Track` ever be cloned? Current design shares it (immutable config; only
  `runner.py` mutates `track.laps` once pre-game). If a future feature mutates
  track state per-game, `GameState.clone` must deep-copy it. Assumed shared for now.
- For the env, is `state.clone(reseed=...)` (fork an initial template) preferred
  over a fresh `GameState.create(..., seed=...)` on every `env.reset()`? Both are
  supported by this design; the ML implementer should pick based on whether per-env
  starting configuration must be held fixed. Assumed `create(seed=...)` is the
  default reset path.
- Should the deprecated `random.seed` line in `runner.py` be removed in Step A3 or
  left until the env lands? Assumed left (with a comment) to avoid churning the
  runner's determinism tests mid-sprint; remove once `HeatEnv` is the only new
  consumer.
