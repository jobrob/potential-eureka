"""H1 deterministic loss replay for the generated-track strong heuristic.

The screen uses a fixed heuristic-development band, selects losses by a
predeclared ordering, and reruns those games with decision traces enabled.
It does not tune or modify either heuristic.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine import rules
from heat.engine.game import Game, GameResult
from heat.engine.phases import ReactDecision
from heat.models.cards import Card, CardType
from heat.models.game_state import GameEvent, GameState
from heat.tracks.generator import generate_track


TRACK_SEED_BASE = 700_000
GAME_SEED_BASE = 710_000
DEFAULT_TRACKS = 12
DEFAULT_GAMES_PER_TRACK = 3
DEFAULT_LOSSES_PER_SEAT = 10
SEAT_COUNTS = (2, 4, 6)
COMMON_SIGNAL_RATE = 0.40


@dataclass(frozen=True)
class GameSpec:
    """A complete deterministic coordinate for one diagnostic race."""

    seats: int
    track_seed: int
    game_seed: int
    strong_seat: int


@dataclass
class ScreenResult:
    """Cheap first-pass result used to select losses without trace bias."""

    spec: GameSpec
    finish_order: list[int]
    strong_place: int
    final_heat: int
    total_rounds: int


class TracingStrongAgent:
    """Delegate to the current strong agent while recording shadow choices."""

    def __init__(self) -> None:
        self.name = "StrongHeuristic-H1"
        self._strong = StrongHeuristicAgent(name=self.name, strength=2)
        self._shadow = HeuristicAgent(name="WeakShadow-H1")
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _distance_to_finish(state: GameState, player_id: int) -> int:
        """Return forward spaces remaining across all unfinished laps."""
        player = state.get_player(player_id)
        laps_remaining = max(0, state.track.laps - player.lap)
        return max(0, state.track.length - player.position) + (
            laps_remaining * state.track.length
        )

    def _context(self, state: GameState, player_id: int, kind: str) -> dict[str, Any]:
        """Capture the public state needed by H1's conservative classifiers."""
        player = state.get_player(player_id)
        return {
            "round": state.round_num,
            "kind": kind,
            "lap": player.lap,
            "position": player.position,
            "gear": player.gear,
            "heat_available": player.heat_available,
            "heat_in_hand": len(player.heat_in_hand),
            "distance_to_finish": self._distance_to_finish(state, player_id),
        }

    @staticmethod
    def _card_speed(cards: tuple[Card, ...]) -> float:
        """Estimate a play's speed using the same stress mean as both agents."""
        return sum(2.5 if card.card_type == CardType.STRESS else card.value for card in cards)

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Record strong and weak-shadow gear choices from the same state."""
        chosen = self._strong.choose_gear(state, player_id, legal_gears)
        shadow = self._shadow.choose_gear(state, player_id, legal_gears)
        record = self._context(state, player_id, "gear")
        record.update({"chosen": list(chosen), "shadow": list(shadow)})
        self.records.append(record)
        return chosen

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Record comparable card-play speeds without serializing card objects."""
        chosen = self._strong.choose_cards(state, player_id, legal_plays)
        shadow = self._shadow.choose_cards(state, player_id, legal_plays)
        record = self._context(state, player_id, "cards")
        record.update(
            {
                "chosen": [card.display_name for card in chosen],
                "shadow": [card.display_name for card in shadow],
                "chosen_speed": self._card_speed(chosen),
                "shadow_speed": self._card_speed(shadow),
            }
        )
        self.records.append(record)
        return chosen

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        """Record react divergences, especially safe-looking late boosts."""
        chosen = self._strong.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )
        shadow = self._shadow.choose_react(
            state, player_id, max_cooldown, can_boost, has_adrenaline
        )
        record = self._context(state, player_id, "react")
        record.update(
            {
                "can_boost": can_boost,
                "has_adrenaline": has_adrenaline,
                "chosen": asdict(chosen),
                "shadow": asdict(shadow),
            }
        )
        self.records.append(record)
        return chosen

    def choose_slipstream(self, state: GameState, player_id: int) -> bool:
        """Record every eligible slipstream decision and weak-agent contrast."""
        chosen = self._strong.choose_slipstream(state, player_id)
        shadow = self._shadow.choose_slipstream(state, player_id)
        record = self._context(state, player_id, "slipstream")
        record.update({"chosen": chosen, "shadow": shadow})
        self.records.append(record)
        return chosen

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        """Delegate discard decisions; H1 does not classify deck cycling."""
        return self._strong.choose_discard(state, player_id, discardable)


def fixed_specs(
    *, tracks: int = DEFAULT_TRACKS, games_per_track: int = DEFAULT_GAMES_PER_TRACK
) -> list[GameSpec]:
    """Build the predeclared development grid in stable selection order."""
    specs: list[GameSpec] = []
    for seats in SEAT_COUNTS:
        for track_offset in range(tracks):
            for repeat in range(games_per_track):
                ordinal = track_offset * games_per_track + repeat
                specs.append(
                    GameSpec(
                        seats=seats,
                        track_seed=TRACK_SEED_BASE + track_offset,
                        game_seed=GAME_SEED_BASE + seats * 10_000 + ordinal,
                        strong_seat=ordinal % seats,
                    )
                )
    return specs


def _agents(spec: GameSpec, tracer: TracingStrongAgent | None = None) -> list[Any]:
    """Create fresh agents so every replay starts with empty agent state."""
    agents: list[Any] = []
    for seat in range(spec.seats):
        if seat == spec.strong_seat:
            agents.append(tracer or StrongHeuristicAgent(strength=2))
        else:
            agents.append(HeuristicAgent(name=f"Heuristic-{seat}"))
    return agents


def run_game(spec: GameSpec, *, trace: bool) -> tuple[ScreenResult, GameResult, Game, list[dict[str, Any]]]:
    """Run one screen or replay game from its exact seed coordinates."""
    track = generate_track(spec.track_seed)
    tracer = TracingStrongAgent() if trace else None
    game = Game(
        track,
        _agents(spec, tracer),
        logging_enabled=trace,
        seed=spec.game_seed,
    )
    result = game.run()
    player = game.state.get_player(spec.strong_seat)
    screen = ScreenResult(
        spec=spec,
        finish_order=list(result.finish_order),
        strong_place=result.finish_order.index(spec.strong_seat) + 1,
        final_heat=player.heat_available,
        total_rounds=result.total_rounds,
    )
    return screen, result, game, tracer.records if tracer is not None else []


def _traffic_blocked_turns(
    records: list[dict[str, Any]], events: list[GameEvent], state: GameState
) -> int:
    """Count card moves whose realised landing differs from the open-track landing."""
    starts = {
        record["round"]: record
        for record in records
        if record["kind"] == "cards"
    }
    blocked = 0
    for event in events:
        if event.event_type != "reveal_and_move" or event.data.get("finished"):
            continue
        start = starts.get(event.round_num)
        if start is None:
            continue
        expected, _ = rules.calculate_move_position(
            start["position"], event.data["speed"], state.track, start["lap"]
        )
        if expected != event.data["new_position"]:
            blocked += 1
    return blocked


def classify_loss(
    records: list[dict[str, Any]],
    events: list[GameEvent],
    state: GameState,
    strong_seat: int,
    final_heat: int,
) -> tuple[list[str], dict[str, int]]:
    """Return conservative, auditable signals for one replayed loss."""
    own_events = [event for event in events if event.player_id == strong_seat]
    spins = sum(event.event_type == "spin_out" for event in own_events)
    blocked = _traffic_blocked_turns(records, own_events, state)
    declined_slips = sum(
        record["kind"] == "slipstream"
        and not record["chosen"]
        and record["shadow"]
        for record in records
    )
    late_boost_refusals = sum(
        record["kind"] == "react"
        and record["lap"] >= state.track.laps
        and record["distance_to_finish"] <= 10
        and record["can_boost"]
        and not record["chosen"]["use_boost"]
        and record["shadow"]["use_boost"]
        for record in records
    )
    late_lower_gears = sum(
        record["kind"] == "gear"
        and record["lap"] >= state.track.laps
        and record["distance_to_finish"] <= 20
        and record["chosen"][0] < record["shadow"][0]
        for record in records
    )
    late_slower_cards = sum(
        record["kind"] == "cards"
        and record["lap"] >= state.track.laps
        and record["distance_to_finish"] <= 20
        and record["chosen_speed"] + 1.5 <= record["shadow_speed"]
        for record in records
    )

    details = {
        "spins": spins,
        "traffic_blocked_turns": blocked,
        "declined_slipstreams": declined_slips,
        "late_boost_refusals": late_boost_refusals,
        "late_lower_gears": late_lower_gears,
        "late_slower_cards": late_slower_cards,
        "final_heat": final_heat,
    }
    signals: list[str] = []
    if spins:
        signals.append("unsafe_corner_entry")
    if late_boost_refusals or late_lower_gears or late_slower_cards:
        signals.append("late_race_conservatism")
    if declined_slips:
        signals.append("declined_slipstream")
    if blocked:
        signals.append("traffic_blocking")
    if not signals:
        signals.append("unclassified")
    return signals, details


def _track_summary(track_seed: int) -> dict[str, Any]:
    """Return the small geometry fingerprint used for loss clustering."""
    track = generate_track(track_seed)
    single_lane = sum(space.lanes == 1 for space in track.spaces) / track.length
    return {
        "track_seed": track_seed,
        "length": track.length,
        "corners": len(track.corners),
        "mean_limit": round(
            sum(corner.speed_limit for corner in track.corners) / len(track.corners), 2
        ),
        "single_lane_fraction": round(single_lane, 3),
    }


def run_diagnostic(
    *,
    tracks: int = DEFAULT_TRACKS,
    games_per_track: int = DEFAULT_GAMES_PER_TRACK,
    losses_per_seat: int = DEFAULT_LOSSES_PER_SEAT,
) -> dict[str, Any]:
    """Screen the fixed grid, replay selected losses, and aggregate H1 evidence."""
    started = time.perf_counter()
    screens: list[ScreenResult] = []
    for seats in SEAT_COUNTS:
        seat_specs = [spec for spec in fixed_specs(tracks=tracks, games_per_track=games_per_track) if spec.seats == seats]
        for spec in seat_specs:
            screen, _, _, _ = run_game(spec, trace=False)
            screens.append(screen)
        losses = sum(screen.strong_place > 1 for screen in screens if screen.spec.seats == seats)
        print(f"screen seats={seats}: games={len(seat_specs)} losses={losses}", flush=True)

    selected: list[ScreenResult] = []
    for seats in SEAT_COUNTS:
        seat_losses = [
            screen for screen in screens
            if screen.spec.seats == seats and screen.strong_place > 1
        ]
        selected.extend(seat_losses[:losses_per_seat])

    replays: list[dict[str, Any]] = []
    signal_counts: Counter[str] = Counter()
    signal_seats: dict[str, set[int]] = defaultdict(set)
    for index, original in enumerate(selected, start=1):
        replay, result, game, records = run_game(original.spec, trace=True)
        if replay.finish_order != original.finish_order:
            raise RuntimeError(f"replay mismatch for {original.spec}")
        signals, details = classify_loss(
            records,
            result.event_log,
            game.state,
            original.spec.strong_seat,
            replay.final_heat,
        )
        for signal in signals:
            signal_counts[signal] += 1
            signal_seats[signal].add(original.spec.seats)
        replays.append(
            {
                "spec": asdict(original.spec),
                "finish_order": replay.finish_order,
                "strong_place": replay.strong_place,
                "total_rounds": replay.total_rounds,
                "signals": signals,
                "details": details,
                "decision_trace": records,
            }
        )
        if index % 5 == 0 or index == len(selected):
            print(f"replay progress: {index}/{len(selected)}", flush=True)

    common_signals = [
        signal
        for signal, count in signal_counts.items()
        if signal != "unclassified"
        and count / max(1, len(selected)) >= COMMON_SIGNAL_RATE
        and len(signal_seats[signal]) >= 2
    ]
    track_losses = Counter(screen.spec.track_seed for screen in screens if screen.strong_place > 1)
    track_games = Counter(screen.spec.track_seed for screen in screens)
    track_rows = []
    for track_seed in sorted(track_games):
        row = _track_summary(track_seed)
        row.update(
            {
                "losses": track_losses[track_seed],
                "games": track_games[track_seed],
                "loss_rate": round(track_losses[track_seed] / track_games[track_seed], 3),
            }
        )
        track_rows.append(row)

    by_seats = {}
    for seats in SEAT_COUNTS:
        cells = [screen for screen in screens if screen.spec.seats == seats]
        wins = sum(screen.strong_place == 1 for screen in cells)
        by_seats[str(seats)] = {
            "games": len(cells),
            "wins": wins,
            "losses": len(cells) - wins,
            "max_rounds": max(cell.total_rounds for cell in cells),
        }
    return {
        "experiment": "H1 heuristic diagnostic",
        "hypothesis": "A small number of recurring decision failures explains most strong-vs-weak losses.",
        "prior_confidence": 0.65,
        "development_band": [TRACK_SEED_BASE, TRACK_SEED_BASE + tracks - 1],
        "configuration": {
            "tracks": tracks,
            "games_per_track": games_per_track,
            "losses_per_seat": losses_per_seat,
            "seat_counts": list(SEAT_COUNTS),
            "common_signal_rate": COMMON_SIGNAL_RATE,
        },
        "screen": by_seats,
        "replays_verified": len(replays),
        "signal_counts": dict(signal_counts),
        "signal_seat_counts": {key: sorted(value) for key, value in signal_seats.items()},
        "common_signals": common_signals,
        "decision": "common_signal_found" if common_signals else "no_common_signal",
        "track_results": track_rows,
        "replays": replays,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    """Run H1 and persist the machine-readable evidence outside source control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/a8_h1_diagnostic.json"))
    parser.add_argument("--tracks", type=int, default=DEFAULT_TRACKS)
    parser.add_argument("--games-per-track", type=int, default=DEFAULT_GAMES_PER_TRACK)
    parser.add_argument("--losses-per-seat", type=int, default=DEFAULT_LOSSES_PER_SEAT)
    args = parser.parse_args()
    report = run_diagnostic(
        tracks=args.tracks,
        games_per_track=args.games_per_track,
        losses_per_seat=args.losses_per_seat,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("screen", "signal_counts", "common_signals", "decision", "elapsed_seconds")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
