"""T1 deterministic repaired-heuristic loss replay and manual narratives.

The diagnostic samples fixed losses from T0, replays them with the existing H1
shadow seam, adds planner-score context, classifies recurring signals, and emits
plain-text turn narratives for manual strategic review. It does not modify an
agent or choose a repair.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from experiments.diag_h1_heuristic import TracingStrongAgent, classify_loss
else:  # Direct ``python experiments/diag_t1_heuristic_losses.py`` execution.
    from diag_h1_heuristic import TracingStrongAgent, classify_loss
from heat.agents import _move_eval as ME
from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.engine import rules
from heat.ml.selfplay.eval_harness import _play_scripted_game
from heat.models.cards import Card, CardType
from heat.models.game_state import GameEvent, GameState
from heat.tracks.generator import TrackGenParams, generate_track


FOCAL_ID = "heuristic_repaired_v1"
OPPONENT_IDS = ("heuristic_weak_v1", "static_search_v1")
FAMILIES = ("default_generated", "tight_generated")
SEAT_COUNTS = (2, 4, 6)
DEFAULT_PER_STRATUM = 5
DEFAULT_SAMPLE_SEED = 1_711
COMMON_SIGNAL_RATE = 0.20
TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),
    laps=2,
)


@dataclass(frozen=True)
class LossSpec:
    """One selected T0 loss and its expected deterministic outcome."""

    source_key: str
    family: str
    opponent: str
    seats: int
    track_seed: int
    game_seed: int
    focal_seat: int
    expected_place: int
    expected_rounds: int
    sample_rank: int

    @property
    def replay_id(self) -> str:
        """Return a readable stable identifier for reports and log files."""
        opponent = "weak" if self.opponent == "heuristic_weak_v1" else "search"
        family = "default" if self.family == "default_generated" else "tight"
        return (
            f"{family}-{opponent}-{self.seats}p-t{self.track_seed}-"
            f"g{self.game_seed}-s{self.focal_seat}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse T0 input, fixed sampling, and output controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t0", type=Path, default=Path("runs/t0_static_matchup_map.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/t1_heuristic_loss_replay.json")
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=Path("runs/t1_heuristic_loss_logs")
    )
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if not 1 <= args.per_stratum <= 20:
        parser.error("--per-stratum must be in 1..20")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def select_losses(
    payload: dict[str, Any],
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> list[LossSpec]:
    """Select a stable random sample from every T1 T0 loss stratum."""
    if payload.get("complete") is not True:
        raise ValueError("T0 artifact must be complete before T1 sampling")
    cells = {
        (str(row["family"]), str(row["opponent"]), int(row["seat_count"])): row
        for row in payload.get("cells", [])
        if row.get("focal") == FOCAL_ID
    }
    selected: list[LossSpec] = []
    ordinal = 0
    for family in FAMILIES:
        for opponent in OPPONENT_IDS:
            for seats in SEAT_COUNTS:
                key = (family, opponent, seats)
                if key not in cells:
                    raise ValueError(f"T0 artifact is missing T1 stratum {key}")
                cell = cells[key]
                losses = sorted(
                    (row for row in cell["races"] if not bool(row["won"])),
                    key=lambda row: (
                        int(row["track_seed"]),
                        int(row["focal_seat"]),
                        int(row["game_seed"]),
                    ),
                )
                if len(losses) < per_stratum:
                    raise ValueError(
                        f"T1 stratum {key} has {len(losses)} losses, "
                        f"needs {per_stratum}"
                    )
                sampled = random.Random(sample_seed + ordinal).sample(
                    losses, per_stratum
                )
                for sample_rank, row in enumerate(sampled):
                    place = row.get("place")
                    if place is None:
                        raise ValueError(f"T0 loss has no focal place in {cell['key']}")
                    selected.append(
                        LossSpec(
                            source_key=str(cell["key"]),
                            family=family,
                            opponent=opponent,
                            seats=seats,
                            track_seed=int(row["track_seed"]),
                            game_seed=int(row["game_seed"]),
                            focal_seat=int(row["focal_seat"]),
                            expected_place=int(place),
                            expected_rounds=int(row["rounds"]),
                            sample_rank=sample_rank,
                        )
                    )
                ordinal += 1
    return selected


def _next_corners(state: GameState, player_id: int) -> list[dict[str, int]]:
    """Describe the next two corner starts from the focal position."""
    player = state.get_player(player_id)
    rows = [
        {
            "start": corner.start,
            "end": corner.end,
            "limit": corner.speed_limit,
            "distance": (corner.start - player.position) % state.track.length,
        }
        for corner in state.track.corners
    ]
    return sorted(rows, key=lambda row: int(row["distance"]))[:2]


def _race_rank(state: GameState, player_id: int) -> int:
    """Return current rank by lap-aware progress, finished order first."""
    ordered = sorted(
        state.players,
        key=lambda player: (
            player.finished,
            -player.finish_order if player.finished else 0,
            ME.race_progress(player, state.track),
        ),
        reverse=True,
    )
    return ordered.index(state.get_player(player_id)) + 1


class TracingRepairedAgent(TracingStrongAgent):
    """Extend H1 tracing with planner math and field context for T1."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "HeuristicRepaired-T1"
        self._strong.name = self.name
        self._audit_by_round: dict[int, list[dict[str, Any]]] = {}

    def _context(self, state: GameState, player_id: int, kind: str) -> dict[str, Any]:
        """Add hand, rank, rivals, and corner geometry to H1 context."""
        context = super()._context(state, player_id, kind)
        player = state.get_player(player_id)
        context.update(
            {
                "rank": _race_rank(state, player_id),
                "hand": [card.display_name for card in player.hand],
                "next_corners": _next_corners(state, player_id),
                "field": [
                    {
                        "seat": other.player_id,
                        "lap": other.lap,
                        "position": other.position,
                        "finished": other.finished,
                    }
                    for other in state.players
                    if other.player_id != player_id
                ],
            }
        )
        return context

    def _planner_audit(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], float, float]:
        """Recompute the planner table using the production evaluator contract."""
        player = state.get_player(player_id)
        heat_price, spin_loss = self._strong._risk_posture(state, player_id)
        expected_stress = ME.expected_basic_value(player)
        max_basic = ME.max_basic_value(player)
        rows: list[dict[str, Any]] = []
        for gear, gear_heat in legal_gears:
            for play in rules.legal_card_plays(player.hand, gear):
                expected_speed = ME.play_speed(play, expected_stress)
                variance = ME.play_speed_variance(
                    play, expected_stress, max_basic
                )
                result = ME.evaluate_move(
                    state,
                    player_id,
                    expected_speed=expected_speed,
                    heat_spent=gear_heat,
                    from_position=player.position,
                    from_lap=player.lap,
                    planned_gear=gear,
                    heat_price=heat_price,
                    horizon_corners=self._strong._horizon,
                    opponent_aware=self._strong._opponent_aware,
                    blocking=self._strong._blocking,
                    enable_solvency=self._strong._solvency,
                    speed_variance=variance,
                    spinout_loss=spin_loss,
                )
                no_future_result = ME.evaluate_move(
                    state,
                    player_id,
                    expected_speed=expected_speed,
                    heat_spent=gear_heat,
                    from_position=player.position,
                    from_lap=player.lap,
                    planned_gear=gear,
                    heat_price=heat_price,
                    horizon_corners=0,
                    opponent_aware=self._strong._opponent_aware,
                    blocking=self._strong._blocking,
                    enable_solvency=self._strong._solvency,
                    speed_variance=variance,
                    spinout_loss=spin_loss,
                )
                rows.append(
                    {
                        "gear": gear,
                        "gear_heat": gear_heat,
                        "cards": [card.display_name for card in play],
                        "has_stress": any(
                            card.card_type == CardType.STRESS for card in play
                        ),
                        "expected_speed": expected_speed,
                        "value": result.value,
                        "no_future_solvency_value": no_future_result.value,
                        "corner_cost": result.corner_cost,
                        "p_spinout": result.p_spinout,
                        "relative_value": result.relative_value,
                        "guaranteed_spin": self._strong._play_guarantees_spin(
                            state, player_id, play, gear_heat
                        ),
                    }
                )
        reject_spins = any(not bool(row["guaranteed_spin"]) for row in rows)
        for row in rows:
            row["eligible"] = not (
                self._strong._avoid_certain_spins
                and reject_spins
                and bool(row["guaranteed_spin"])
            )
        return rows, heat_price, spin_loss

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Record the chosen joint plan, alternatives, and score margin."""
        audit, heat_price, spin_loss = self._planner_audit(
            state, player_id, legal_gears
        )
        chosen_gear = super().choose_gear(state, player_id, legal_gears)
        chosen_cards = [
            card.display_name for card in (self._strong._plan_cards or ())
        ]
        eligible = sorted(
            (row for row in audit if bool(row["eligible"])),
            key=lambda row: float(row["value"]),
            reverse=True,
        )
        no_future_eligible = sorted(
            eligible,
            key=lambda row: float(row["no_future_solvency_value"]),
            reverse=True,
        )
        fastest = max(float(row["expected_speed"]) for row in eligible)
        basic_alternatives = [
            row for row in eligible if not bool(row["has_stress"])
        ]
        cached_match = next(
            (
                row
                for row in eligible
                if int(row["gear"]) == chosen_gear[0]
                and row["cards"] == chosen_cards
            ),
            None,
        )
        # Retain ineligible rows too: a stale production cache can return a
        # plan that the current guaranteed-spin guard would reject.
        self._audit_by_round[state.round_num] = audit
        self.records[-1].update(
            {
                "legal_gears": [list(value) for value in legal_gears],
                "planner": {
                    "heat_price": heat_price,
                    "spinout_loss": spin_loss,
                    "eligible_count": len(eligible),
                    "returned_gear": list(chosen_gear),
                    "cached_cards": chosen_cards,
                    "stale_cached_plan": cached_match is None,
                    "chosen": cached_match,
                    "current_best": eligible[0],
                    "no_future_solvency_best": no_future_eligible[0],
                    "selected_is_current_best": cached_match == eligible[0],
                    "value_regret": (
                        float(eligible[0]["value"])
                        - float(cached_match["value"])
                        if cached_match is not None
                        else None
                    ),
                    "no_future_solvency_regret": (
                        float(no_future_eligible[0]["no_future_solvency_value"])
                        - float(cached_match["no_future_solvency_value"])
                        if cached_match is not None
                        else None
                    ),
                    "fastest_eligible_speed": fastest,
                    "best_basic_alternative": (
                        max(basic_alternatives, key=lambda row: float(row["value"]))
                        if basic_alternatives
                        else None
                    ),
                    "top": eligible[:4],
                },
            }
        )
        return chosen_gear

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Retain the legal speed range alongside H1's strong/shadow choice."""
        chosen = super().choose_cards(state, player_id, legal_plays)
        expected_stress = ME.expected_basic_value(state.get_player(player_id))
        self.records[-1]["legal_play_count"] = len(legal_plays)
        self.records[-1]["legal_expected_speeds"] = sorted(
            {ME.play_speed(play, expected_stress) for play in legal_plays}
        )
        round_num = state.round_num
        audit = self._audit_by_round.pop(round_num)
        eligible = sorted(
            (row for row in audit if bool(row["eligible"])),
            key=lambda row: float(row["value"]),
            reverse=True,
        )
        actual_cards = [card.display_name for card in chosen]
        actual_gear = state.get_player(player_id).gear
        chosen_row = next(
            (
                row
                for row in audit
                if int(row["gear"]) == actual_gear
                and row["cards"] == actual_cards
            ),
            None,
        )
        if chosen_row is None:
            raise RuntimeError(
                "T1 planner audit lost the realised play: "
                f"round={round_num} gear={actual_gear} cards={actual_cards} "
                f"audit={[(row['gear'], row['cards'], row['eligible']) for row in audit]}"
            )
        gear_record = next(
            record
            for record in reversed(self.records[:-1])
            if record["kind"] == "gear" and int(record["round"]) == round_num
        )
        planner = gear_record["planner"]
        planner["chosen"] = chosen_row
        planner["selected_is_current_best"] = chosen_row == eligible[0]
        planner["value_regret"] = max(
            0.0, float(eligible[0]["value"]) - float(chosen_row["value"])
        )
        no_future_best = max(
            eligible, key=lambda row: float(row["no_future_solvency_value"])
        )
        planner["no_future_solvency_regret"] = max(
            0.0,
            float(no_future_best["no_future_solvency_value"])
            - float(chosen_row["no_future_solvency_value"]),
        )
        return chosen

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        """Record deck-cycling disagreement instead of leaving it invisible."""
        chosen = self._strong.choose_discard(state, player_id, discardable)
        shadow = self._shadow.choose_discard(state, player_id, discardable)
        record = self._context(state, player_id, "discard")
        record.update(
            {
                "chosen": [card.display_name for card in chosen],
                "shadow": [card.display_name for card in shadow],
                "discardable": [card.display_name for card in discardable],
            }
        )
        self.records.append(record)
        return chosen


def _opponent_factory(opponent: str) -> BaseAgent:
    """Build one fresh frozen T0 opponent."""
    if opponent == "heuristic_weak_v1":
        return HeuristicAgent(name="HeuristicWeakV1")
    if opponent == "static_search_v1":
        return StaticSearchAgent()
    raise ValueError(f"unknown T1 opponent {opponent!r}")


def _turn_rows(
    records: Sequence[dict[str, Any]],
    events: Sequence[GameEvent],
    focal_seat: int,
    track_length: int,
) -> list[dict[str, Any]]:
    """Combine decisions and realised events into one auditable row per round."""
    grouped_records: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped_records[int(record["round"])][str(record["kind"])] = record
    grouped_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.player_id == focal_seat and event.event_type in {
            "reveal_and_move",
            "boost",
            "cooldown",
            "corner_check",
            "spin_out",
            "slipstream",
            "discard",
        }:
            grouped_events[event.round_num].append(
                {"type": event.event_type, "data": event.data}
            )
    rows: list[dict[str, Any]] = []
    for round_num in sorted(grouped_records):
        decision = grouped_records[round_num]
        gear = decision.get("gear")
        cards = decision.get("cards")
        events_this_round = grouped_events.get(round_num, [])
        flags: list[str] = []
        if gear is not None:
            planner = gear["planner"]
            chosen = planner["chosen"]
            fastest = float(planner["fastest_eligible_speed"])
            chosen_speed = float(chosen["expected_speed"])
            next_corner = gear["next_corners"][0] if gear["next_corners"] else None
            if fastest - chosen_speed >= 3.0:
                flags.append("large_speed_sacrifice")
            if (
                next_corner is not None
                and int(next_corner["distance"]) > fastest + 1.0
                and float(gear["distance_to_finish"]) > fastest + 1.0
                and fastest - chosen_speed >= 2.0
            ):
                flags.append("clear_straight_underpush")
            if (
                int(gear["lap"]) >= 2
                and chosen_speed < float(gear["distance_to_finish"]) <= fastest
            ):
                flags.append("finish_underpush")
            basic = planner["best_basic_alternative"]
            if (
                bool(chosen["has_stress"])
                and basic is not None
                and float(basic["value"]) + 0.25 >= float(chosen["value"])
                and float(basic["expected_speed"]) >= chosen_speed
            ):
                flags.append("stress_over_basic")
            if int(gear["chosen"][0]) < int(gear["shadow"][0]):
                flags.append("lower_gear_than_shadow")
            if bool(planner["stale_cached_plan"]):
                flags.append("stale_cached_gear_plan")
            if not bool(planner["selected_is_current_best"]):
                flags.append("planner_cache_regret")
            no_future_best = planner["no_future_solvency_best"]
            if (
                next_corner is not None
                and int(gear["lap"]) >= 2
                and int(next_corner["distance"])
                >= int(gear["distance_to_finish"])
                and float(planner["no_future_solvency_regret"]) > 0.01
                and float(no_future_best["expected_speed"]) > chosen_speed
            ):
                flags.append("post_finish_solvency_drag")
        if cards is not None and float(cards["chosen_speed"]) + 1.5 <= float(
            cards["shadow_speed"]
        ):
            flags.append("slower_cards_than_shadow")
        react = decision.get("react")
        if (
            react is not None
            and react["can_boost"]
            and not react["chosen"]["use_boost"]
            and react["shadow"]["use_boost"]
        ):
            flags.append("boost_refusal")
        slip = decision.get("slipstream")
        if slip is not None and not slip["chosen"] and slip["shadow"]:
            flags.append("declined_slipstream")
        reveal = next(
            (event for event in events_this_round if event["type"] == "reveal_and_move"),
            None,
        )
        if reveal is not None and cards is not None:
            expected, _ = rules.calculate_move_position(
                int(cards["position"]),
                int(reveal["data"]["speed"]),
                _TrackLengthProxy(track_length),
                int(cards["lap"]),
            )
            if expected != int(reveal["data"]["new_position"]):
                flags.append("traffic_blocking")
        if any(event["type"] == "spin_out" for event in events_this_round):
            flags.append("unsafe_corner_entry")
        rows.append(
            {
                "round": round_num,
                "decisions": decision,
                "events": events_this_round,
                "flags": flags,
            }
        )
    return rows


@dataclass(frozen=True)
class _TrackLengthProxy:
    """Minimal track stand-in for modulo-only movement expectation."""

    length: int


ANOMALY_WEIGHTS = {
    "post_finish_solvency_drag": 5,
    "stale_cached_gear_plan": 4,
    "planner_cache_regret": 4,
    "finish_underpush": 4,
    "clear_straight_underpush": 3,
    "stress_over_basic": 2,
    "large_speed_sacrifice": 1,
    "lower_gear_than_shadow": 1,
    "slower_cards_than_shadow": 1,
    "boost_refusal": 1,
    "declined_slipstream": 1,
    "unsafe_corner_entry": 1,
    # Traffic displacement is an outcome/context signal, not automatically
    # an agent mistake, so it does not increase manual-suspicion ranking.
    "traffic_blocking": 0,
}


def _classify_turns(
    turns: Sequence[dict[str, Any]], final_heat: int
) -> tuple[list[str], int]:
    """Return race-level signals and a manual-review anomaly score."""
    counts = Counter(
        flag for turn in turns for flag in set(str(value) for value in turn["flags"])
    )
    signals = sorted(counts)
    if final_heat >= 4:
        signals.append("stranded_heat")
    if not signals:
        signals.append("unclassified")
    score = sum(ANOMALY_WEIGHTS.get(signal, 0) * count for signal, count in counts.items())
    if final_heat >= 4:
        score += 1
    return signals, score


def _render_event(event: dict[str, Any]) -> str:
    """Render one focal engine event compactly for the manual log."""
    kind = str(event["type"])
    data = event["data"]
    if kind == "reveal_and_move":
        return (
            f"move speed {data['speed']} -> pos {data['new_position']} "
            f"lap {data['lap']}"
        )
    if kind == "corner_check":
        return f"corner speed {data['speed']} heat {data['heat_cost']}"
    if kind == "boost":
        return f"boost +{data['value']} (heat now {data['heat_available']})"
    if kind == "cooldown":
        return f"cool {data['count']} (heat now {data['heat_available']})"
    if kind == "slipstream":
        return f"slipstream -> pos {data['new_position']}"
    if kind == "spin_out":
        return f"SPIN at {data['corner_start']} -> {data['new_position']}"
    if kind == "discard":
        return f"discard {data['cards']}"
    return kind


def render_narrative(replay: dict[str, Any]) -> str:
    """Render one complete T1 replay into a human-readable turn log."""
    spec = replay["spec"]
    lines = [
        f"T1 replay {replay['replay_id']}",
        (
            f"family={spec['family']} opponent={spec['opponent']} "
            f"seats={spec['seats']} track={spec['track_seed']} "
            f"game={spec['game_seed']} focal_seat={spec['focal_seat']}"
        ),
        (
            f"outcome=place {replay['place']}/{spec['seats']} "
            f"rounds={replay['rounds']} final_heat={replay['final_heat']}"
        ),
        f"signals={', '.join(replay['signals'])}",
        f"anomaly_score={replay['anomaly_score']}",
        "",
    ]
    for turn in replay["turns"]:
        decision = turn["decisions"]
        gear = decision.get("gear")
        cards = decision.get("cards")
        if gear is None:
            continue
        corner = gear["next_corners"][0] if gear["next_corners"] else None
        corner_text = (
            f"next corner {corner['start']} limit {corner['limit']} "
            f"in {corner['distance']}"
            if corner is not None
            else "no corner"
        )
        lines.append(
            f"R{turn['round']:02d} lap {gear['lap']} pos {gear['position']} "
            f"rank {gear['rank']}/{spec['seats']} heat {gear['heat_available']} | "
            f"{corner_text}"
        )
        planner = gear["planner"]
        chosen = planner["chosen"]
        lines.append(
            f"  chose gear {chosen['gear']} heat {chosen['gear_heat']} "
            f"cards {chosen['cards']} exp {chosen['expected_speed']:.2f} "
            f"value {chosen['value']:.3f} pspin {chosen['p_spinout']:.2f}; "
            f"weak shadow gear {gear['shadow'][0]}"
        )
        if cards is not None:
            lines.append(
                f"  actual cards {cards['chosen']} exp {cards['chosen_speed']:.2f}; "
                f"shadow {cards['shadow']} exp {cards['shadow_speed']:.2f}"
            )
        alternatives = [
            f"G{row['gear']} {row['cards']} exp={row['expected_speed']:.2f} "
            f"v={row['value']:.3f} p={row['p_spinout']:.2f}"
            for row in planner["top"][1:3]
        ]
        if alternatives:
            lines.append("  alternatives: " + " | ".join(alternatives))
        events = "; ".join(_render_event(event) for event in turn["events"])
        if events:
            lines.append(f"  outcome: {events}")
        if turn["flags"]:
            lines.append("  FLAGS: " + ", ".join(turn["flags"]))
    lines.append("")
    return "\n".join(lines)


def run_replay(spec: LossSpec) -> dict[str, Any]:
    """Replay one selected T0 loss and verify its outcome before classifying."""
    track = generate_track(
        spec.track_seed,
        TIGHT_PARAMS if spec.family == "tight_generated" else None,
    )
    tracer = TracingRepairedAgent()
    agents = [_opponent_factory(spec.opponent) for _ in range(spec.seats)]
    agents[spec.focal_seat] = tracer
    state = _play_scripted_game(
        track, spec.seats, dict(enumerate(agents)), spec.game_seed
    )
    focal = state.get_player(spec.focal_seat)
    place = focal.finish_order if focal.finished else None
    observed_rounds = state.round_num
    if place != spec.expected_place or observed_rounds != spec.expected_rounds:
        raise RuntimeError(
            f"T1 replay mismatch for {spec.replay_id}: "
            f"expected place/rounds {spec.expected_place}/{spec.expected_rounds}, "
            f"got {place}/{observed_rounds}"
        )
    player = state.get_player(spec.focal_seat)
    turns = _turn_rows(
        tracer.records,
        state.event_log,
        spec.focal_seat,
        track.length,
    )
    signals, anomaly_score = _classify_turns(turns, player.heat_available)
    h1_signals, h1_details = classify_loss(
        tracer.records,
        state.event_log,
        state,
        spec.focal_seat,
        player.heat_available,
    )
    return {
        "replay_id": spec.replay_id,
        "spec": asdict(spec),
        "finish_order": [
            finished.player_id for finished in state.finished_players
        ],
        "place": place,
        "rounds": observed_rounds,
        "final_heat": player.heat_available,
        "signals": signals,
        "h1_signals": h1_signals,
        "h1_details": h1_details,
        "anomaly_score": anomaly_score,
        "turns": turns,
    }


def _manual_review_ids(replays: Sequence[dict[str, Any]]) -> list[str]:
    """Select 12 baseline narratives plus six additional suspicious losses."""
    baseline = [
        replay
        for replay in replays
        if int(replay["spec"]["sample_rank"]) == 0
    ]
    baseline_ids = {str(replay["replay_id"]) for replay in baseline}
    extra = sorted(
        (
            replay
            for replay in replays
            if str(replay["replay_id"]) not in baseline_ids
        ),
        key=lambda replay: (
            -int(replay["anomaly_score"]),
            str(replay["replay_id"]),
        ),
    )[:6]
    return [str(replay["replay_id"]) for replay in baseline + extra]


def summarize_replays(replays: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate signal frequency and regime coverage over completed replays."""
    signal_counts: Counter[str] = Counter()
    signal_opponents: dict[str, set[str]] = defaultdict(set)
    signal_seats: dict[str, set[int]] = defaultdict(set)
    signal_families: dict[str, set[str]] = defaultdict(set)
    for replay in replays:
        for signal in set(str(value) for value in replay["signals"]):
            signal_counts[signal] += 1
            signal_opponents[signal].add(str(replay["spec"]["opponent"]))
            signal_seats[signal].add(int(replay["spec"]["seats"]))
            signal_families[signal].add(str(replay["spec"]["family"]))
    common = [
        signal
        for signal, count in signal_counts.most_common()
        if signal != "unclassified"
        and count / max(1, len(replays)) >= COMMON_SIGNAL_RATE
        and len(signal_opponents[signal]) == 2
        and len(signal_seats[signal]) >= 2
    ]
    return {
        "replays": len(replays),
        "signal_counts": dict(signal_counts.most_common()),
        "signal_opponents": {
            key: sorted(value) for key, value in signal_opponents.items()
        },
        "signal_seat_counts": {
            key: sorted(value) for key, value in signal_seats.items()
        },
        "signal_families": {
            key: sorted(value) for key, value in signal_families.items()
        },
        "common_signals": common,
    }


def _write_outputs(
    out: Path,
    logs_dir: Path,
    payload: dict[str, Any],
) -> None:
    """Persist JSON atomically and write one readable log per replay."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(out)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manual_ids = set(str(value) for value in payload["manual_review_ids"])
    index_lines = ["T1 manual review set", ""]
    for replay in payload["replays"]:
        path = logs_dir / f"{replay['replay_id']}.txt"
        path.write_text(render_narrative(replay), encoding="utf-8")
        if replay["replay_id"] in manual_ids:
            index_lines.append(
                f"{replay['replay_id']} score={replay['anomaly_score']} "
                f"signals={','.join(replay['signals'])}"
            )
    (logs_dir / "manual-review-index.txt").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )


def run_t1(
    t0_payload: dict[str, Any],
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    max_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run the bounded T1 sample and return complete or partial evidence."""
    selected = select_losses(
        t0_payload,
        per_stratum=per_stratum,
        sample_seed=sample_seed,
    )
    started = time.perf_counter()
    replays: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for index, spec in enumerate(selected, start=1):
        if time.perf_counter() - started >= max_seconds:
            stop_reason = f"wall-time cap reached before {spec.replay_id}"
            break
        replays.append(run_replay(spec))
        if index % 5 == 0 or index == len(selected):
            print(
                f"T1 replay progress {index}/{len(selected)} "
                f"last={spec.replay_id}",
                flush=True,
            )
    summary = summarize_replays(replays)
    return {
        "complete": len(replays) == len(selected),
        "stop_reason": stop_reason,
        "experiment": "T1 repaired-heuristic loss replay",
        "hypothesis": (
            "A recurring post-spin decision pattern appears in at least 20% of "
            "losses across both opponent styles and multiple field sizes."
        ),
        "prior_confidence": 0.65,
        "configuration": {
            "per_stratum": per_stratum,
            "sample_seed": sample_seed,
            "strata": 12,
            "expected_replays": len(selected),
            "common_signal_rate": COMMON_SIGNAL_RATE,
            "max_seconds": max_seconds,
        },
        "summary": summary,
        "manual_review_ids": _manual_review_ids(replays),
        "replays": replays,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    """Load T0, run T1, and persist machine and human-readable evidence."""
    args = _parse_args(argv)
    t0_payload = json.loads(args.t0.read_text(encoding="utf-8"))
    report = run_t1(
        t0_payload,
        per_stratum=args.per_stratum,
        sample_seed=args.sample_seed,
        max_seconds=args.max_seconds,
    )
    _write_outputs(args.out, args.logs_dir, report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "replays": report["summary"]["replays"],
                "signal_counts": report["summary"]["signal_counts"],
                "common_signals": report["summary"]["common_signals"],
                "manual_review_ids": report["manual_review_ids"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} and {args.logs_dir}", flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
