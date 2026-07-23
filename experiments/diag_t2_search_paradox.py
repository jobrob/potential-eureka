"""T2 paired replay of StaticSearchV1 wins and losses from the T0 map.

The diagnostic keeps the production search choice in control of each race, then
scores every distinct candidate with the same rollout contract.  It also asks
the frozen weak and repaired heuristics for shadow plans on the same state.  No
agent configuration or evaluator is changed by this experiment.
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

from heat.agents import _move_eval as ME
from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine import rules
from heat.ml.selfplay.eval_harness import _play_scripted_game
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.tracks.generator import TrackGenParams, generate_track


FOCAL_ID = "static_search_v1"
OPPONENT_IDS = ("heuristic_weak_v1", "heuristic_repaired_v1")
FAMILIES = ("default_generated", "tight_generated")
SEAT_COUNTS = (2, 4, 6)
DEFAULT_PER_STRATUM = 3
DEFAULT_SAMPLE_SEED = 2_717
TIGHT_PARAMS = TrackGenParams(
    num_corners_range=(4, 7),
    speed_limit_choices=(1, 1, 2, 3),
    laps=2,
)


@dataclass(frozen=True)
class PairSpec:
    """One exact T0 coordinate with opposite outcomes against the two fields."""

    family: str
    seats: int
    track_seed: int
    game_seed: int
    focal_seat: int
    repaired_place: int
    repaired_rounds: int
    weak_place: int
    weak_rounds: int
    sample_rank: int

    @property
    def pair_id(self) -> str:
        """Return a stable coordinate label for artifacts and narratives."""
        family = "default" if self.family == "default_generated" else "tight"
        return (
            f"{family}-{self.seats}p-t{self.track_seed}-"
            f"g{self.game_seed}-s{self.focal_seat}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the frozen T2 sample and bounded execution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t0",
        type=Path,
        default=Path("runs/t0_static_matchup_map_post_t1a.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/t2_search_paradox_replay.json")
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=Path("runs/t2_search_paradox_logs")
    )
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if not 1 <= args.per_stratum <= 10:
        parser.error("--per-stratum must be in 1..10")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def _coordinate(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return the matchup-independent identity of one T0 race."""
    return int(row["track_seed"]), int(row["game_seed"]), int(row["focal_seat"])


def select_pairs(
    payload: dict[str, Any],
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> list[PairSpec]:
    """Select stable paired coordinates from every family and seat stratum."""
    if payload.get("complete") is not True:
        raise ValueError("T0 artifact must be complete before T2 sampling")
    cells = {
        (str(row["opponent"]), str(row["family"]), int(row["seat_count"])): row
        for row in payload.get("cells", [])
        if row.get("focal") == FOCAL_ID
    }
    selected: list[PairSpec] = []
    ordinal = 0
    for family in FAMILIES:
        for seats in SEAT_COUNTS:
            weak_key = ("heuristic_weak_v1", family, seats)
            repaired_key = ("heuristic_repaired_v1", family, seats)
            if weak_key not in cells or repaired_key not in cells:
                raise ValueError(f"T0 artifact is missing T2 stratum {(family, seats)}")
            weak = {_coordinate(row): row for row in cells[weak_key]["races"]}
            repaired = {
                _coordinate(row): row for row in cells[repaired_key]["races"]
            }
            paradox = sorted(
                coordinate
                for coordinate in weak.keys() & repaired.keys()
                if not bool(weak[coordinate]["won"])
                and bool(repaired[coordinate]["won"])
            )
            if len(paradox) < per_stratum:
                raise ValueError(
                    f"T2 stratum {(family, seats)} has {len(paradox)} paired "
                    f"coordinates, needs {per_stratum}"
                )
            sampled = random.Random(sample_seed + ordinal).sample(
                paradox, per_stratum
            )
            for sample_rank, coordinate in enumerate(sampled):
                weak_row = weak[coordinate]
                repaired_row = repaired[coordinate]
                selected.append(
                    PairSpec(
                        family=family,
                        seats=seats,
                        track_seed=coordinate[0],
                        game_seed=coordinate[1],
                        focal_seat=coordinate[2],
                        repaired_place=int(repaired_row["place"]),
                        repaired_rounds=int(repaired_row["rounds"]),
                        weak_place=int(weak_row["place"]),
                        weak_rounds=int(weak_row["rounds"]),
                        sample_rank=sample_rank,
                    )
                )
            ordinal += 1
    return selected


def _race_rank(state: GameState, player_id: int) -> int:
    """Return the exact current place proxy, respecting completed finish order."""
    player = state.get_player(player_id)
    if player.finished and player.finish_order is not None:
        return int(player.finish_order)
    ahead = sum(
        other.finished
        or ME.race_progress(other, state.track) > ME.race_progress(player, state.track)
        for other in state.players
        if other.player_id != player_id
    )
    return int(ahead) + 1


def _plan_key(gear: tuple[int, int], cards: Sequence[Card]) -> str:
    """Identify rollout-equivalent plans by gear, shift heat, and total speed."""
    return f"g{gear[0]}h{gear[1]}s{sum(card.value for card in cards)}"


def _cards(cards: Sequence[Card]) -> list[str]:
    """Render a card play compactly without relying on object identities."""
    return [card.display_name for card in cards]


class TracingStaticSearchAgent(StaticSearchAgent):
    """Run StaticSearchV1 unchanged while auditing candidates and shadows."""

    def __init__(self) -> None:
        super().__init__(name="StaticSearchV1-T2")
        self.records: list[dict[str, Any]] = []

    def _trace_plan(
        self,
        state: GameState,
        player_id: int,
        gear: tuple[int, int],
        cards: tuple[Card, ...],
        turn_seed: int,
        plan_index: int,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Repeat the production clone loop and retain each exact leaf state."""
        rows: list[dict[str, Any]] = []
        for det in range(self.n_determinizations):
            reseed = (turn_seed + plan_index * 100003 + det * 31) & 0x7FFFFFFF
            clone = state.clone(reseed=reseed)
            clone.logging_enabled = True
            if self.determinize_hidden:
                self._determinize_opponents(
                    clone, player_id, random.Random(reseed ^ 0x5DEECE66)
                )
            first_round = clone.round_num
            score = self._rollout_once(clone, player_id, gear, cards)
            own_spins, later_spins = self._count_spins(
                clone, player_id, first_round
            )
            player = clone.get_player(player_id)
            events = [
                {
                    "type": event.event_type,
                    "data": event.data,
                }
                for event in clone.event_log
                if event.round_num == first_round
                and event.player_id == player_id
                and event.event_type
                in {
                    "stress_resolved",
                    "reveal_and_move",
                    "boost",
                    "slipstream",
                    "corner_check",
                    "spin_out",
                }
            ]
            rows.append(
                {
                    "determinization": det,
                    "score": score,
                    "progress": ME.race_progress(player, clone.track),
                    "rank": _race_rank(clone, player_id),
                    "heat": player.heat_available,
                    "finished": player.finished,
                    "place": player.finish_order if player.finished else None,
                    "own_spins": own_spins,
                    "later_spins": later_spins,
                    "events": events,
                }
            )
        return sum(float(row["score"]) for row in rows) / len(rows), rows

    def _shadow_plan(
        self, agent: BaseAgent, state: GameState, player_id: int
    ) -> dict[str, Any]:
        """Ask one persistent frozen heuristic for its legal joint plan."""
        clone = state.clone(reseed=0)
        player = clone.get_player(player_id)
        legal_gears = rules.legal_gear_shifts(player.gear, player.heat_available)
        gear = agent.choose_gear(clone, player_id, legal_gears)
        player.gear = gear[0]
        legal_plays = rules.legal_card_plays(player.hand, player.gear)
        cards = agent.choose_cards(clone, player_id, legal_plays)
        return {"key": _plan_key(gear, cards), "gear": list(gear), "cards": _cards(cards)}

    @staticmethod
    def _dominates(
        preferred: dict[str, Any], alternative: dict[str, Any]
    ) -> bool:
        """Return true only for strict dominance under common random seeds."""
        better = False
        for chosen, shadow in zip(
            preferred["matched_determinizations"],
            alternative["matched_determinizations"],
            strict=True,
        ):
            if (
                float(chosen["score"]) <= float(shadow["score"])
                or int(chosen["rank"]) > int(shadow["rank"])
                or int(chosen["heat"]) < int(shadow["heat"])
                or int(chosen["own_spins"]) > int(shadow["own_spins"])
            ):
                return False
            better = True
        return better

    @staticmethod
    def _resource_dominates(
        preferred: dict[str, Any], alternative: dict[str, Any]
    ) -> bool:
        """Return true for a common-seed tie with no-worse rank/heat/spins."""
        better = False
        for candidate, chosen in zip(
            preferred["matched_determinizations"],
            alternative["matched_determinizations"],
            strict=True,
        ):
            if (
                float(candidate["score"]) < float(chosen["score"])
                or int(candidate["rank"]) > int(chosen["rank"])
                or int(candidate["heat"]) < int(chosen["heat"])
                or int(candidate["own_spins"]) > int(chosen["own_spins"])
            ):
                return False
            better = better or (
                int(candidate["rank"]) < int(chosen["rank"])
                or int(candidate["heat"]) > int(chosen["heat"])
                or int(candidate["own_spins"]) < int(chosen["own_spins"])
            )
        return better

    def _audit_turn(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
        chosen_gear: tuple[int, int],
        chosen_cards: tuple[Card, ...] | None,
        cache_reused: bool,
    ) -> dict[str, Any]:
        """Score all candidates and classify production and shadow choices."""
        candidates = self._candidate_plans(state, player_id, legal_gears)
        ordered = self._prune_top_k(state, player_id, candidates)
        top_keys = {_plan_key(gear, cards) for gear, cards in ordered}
        index_of = {(gear, cards): index for index, (gear, cards) in enumerate(candidates)}
        turn_seed = self._turn_seed(self._turn_signature(state, player_id))
        rows: list[dict[str, Any]] = []
        for gear, cards in candidates:
            plan_index = index_of[(gear, cards)]
            score, det_rows = self._trace_plan(
                state, player_id, gear, cards, turn_seed, plan_index
            )
            matched_score, matched_rows = self._trace_plan(
                state, player_id, gear, cards, turn_seed, 0
            )
            rows.append(
                {
                    "key": _plan_key(gear, cards),
                    "gear": list(gear),
                    "cards": _cards(cards),
                    "speed": sum(card.value for card in cards),
                    "prior": self._candidate_prior(
                        state, player_id, gear, cards
                    ),
                    "in_top_k": _plan_key(gear, cards) in top_keys,
                    "score": score,
                    "determinizations": det_rows,
                    "matched_score": matched_score,
                    "matched_determinizations": matched_rows,
                }
            )
        rows.sort(key=lambda row: float(row["prior"]), reverse=True)
        planned_key = (
            _plan_key(chosen_gear, chosen_cards)
            if chosen_cards is not None
            else None
        )
        by_key = {str(row["key"]): row for row in rows}
        full_best = max(rows, key=lambda row: float(row["score"]))
        top_best = max(
            (row for row in rows if bool(row["in_top_k"])),
            key=lambda row: float(row["score"]),
        )
        flags: list[str] = []
        det_best = []
        top_rows = [row for row in rows if bool(row["in_top_k"])]
        for det in range(self.n_determinizations):
            det_best.append(
                max(
                    top_rows,
                    key=lambda row: float(row["determinizations"][det]["score"]),
                )["key"]
            )
        if len(set(det_best)) > 1:
            flags.append("determinization_argmax_disagreement")

        shadows = {
            "weak": self._shadow_plan(
                HeuristicAgent(name="WeakShadow-T2"), state, player_id
            ),
            "repaired": self._shadow_plan(
                StrongHeuristicAgent(
                    name="RepairedShadow-T2",
                    strength=2,
                    avoid_certain_spins=True,
                ),
                state,
                player_id,
            ),
        }
        for name, shadow in shadows.items():
            shadow_row = by_key.get(str(shadow["key"]))
            shadow["candidate"] = shadow_row

        player = state.get_player(player_id)
        record = {
            "round": state.round_num,
            "lap": player.lap,
            "position": player.position,
            "rank": _race_rank(state, player_id),
            "heat": player.heat_available,
            "hand": _cards(player.hand),
            "track_length": state.track.length,
            "track_laps": state.track.laps,
            "field": [
                {
                    "seat": other.player_id,
                    "lap": other.lap,
                    "position": other.position,
                    "finished": other.finished,
                    "place": other.finish_order,
                }
                for other in state.players
                if other.player_id != player_id
            ],
            "cache_reused": cache_reused,
            "chosen_gear": list(chosen_gear),
            "chosen_key": planned_key,
            "chosen": by_key.get(planned_key or "", top_best),
            "top_k_best": top_best,
            "full_best": full_best,
            "determinization_best_keys": det_best,
            "shadows": shadows,
            "flags": sorted(set(flags)),
            "candidates": rows,
        }
        self._refresh_choice_flags(record, planned_key)
        return record

    def _refresh_choice_flags(
        self, record: dict[str, Any], chosen_key: str | None
    ) -> None:
        """Reclassify choice-dependent flags after card-plan revalidation."""
        preserved = {
            str(flag)
            for flag in record["flags"]
            if flag == "determinization_argmax_disagreement"
        }
        by_key = {str(row["key"]): row for row in record["candidates"]}
        chosen = by_key.get(chosen_key or "")
        if chosen is None:
            preserved.add("cached_plan_not_in_current_candidates")
            chosen = record["top_k_best"]
        full_best = record["full_best"]
        top_best = record["top_k_best"]
        if (
            not bool(full_best["in_top_k"])
            and float(full_best["score"]) > float(chosen["score"])
        ):
            preserved.add("top_k_pruned_better_plan")
        tied = [
            row
            for row in record["candidates"]
            if row["key"] != chosen["key"]
            and abs(
                float(row["matched_score"]) - float(chosen["matched_score"])
            )
            < 1e-9
        ]
        if any(self._resource_dominates(row, chosen) for row in tied):
            preserved.add("progress_tie_resource_dominated")
        if bool(record["cache_reused"]) and float(top_best["score"]) > float(
            chosen["score"]
        ):
            preserved.add("cached_plan_suboptimal")
        for name, shadow in record["shadows"].items():
            shadow_row = shadow["candidate"]
            if shadow_row is None:
                continue
            chosen_dets = chosen["matched_determinizations"]
            shadow_dets = shadow_row["matched_determinizations"]
            if all(int(row["own_spins"]) == 0 for row in chosen_dets) and all(
                int(row["own_spins"]) > 0 for row in shadow_dets
            ):
                preserved.add(f"{name}_shadow_avoidable_spin")
            elif all(bool(row["finished"]) for row in chosen_dets) and all(
                not bool(row["finished"]) for row in shadow_dets
            ):
                preserved.add(f"{name}_shadow_finish_miss")
            elif self._dominates(chosen, shadow_row):
                preserved.add(f"{name}_shadow_strictly_dominated")
        record["chosen_key"] = chosen_key
        record["chosen"] = chosen
        record["flags"] = sorted(preserved)

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Return the production choice, then audit it without steering the race."""
        signature = self._turn_signature(state, player_id)
        cache_reused = self._plan_sig == signature and self._plan_gear is not None
        chosen = super().choose_gear(state, player_id, legal_gears)
        self.records.append(
            self._audit_turn(
                state,
                player_id,
                legal_gears,
                chosen,
                self._plan_cards,
                cache_reused,
            )
        )
        return chosen

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Record whether card revalidation changed the planned search play."""
        cards = super().choose_cards(state, player_id, legal_plays)
        if self.records:
            record = self.records[-1]
            actual_key = _plan_key(
                (int(record["chosen_gear"][0]), int(record["chosen_gear"][1])),
                cards,
            )
            record["actual_cards"] = _cards(cards)
            record["actual_speed"] = sum(card.value for card in cards)
            record["actual_differs_from_plan"] = actual_key != record["chosen_key"]
            self._refresh_choice_flags(record, actual_key)
        return cards


def _opponent_factory(opponent: str) -> BaseAgent:
    """Build one frozen T0 opponent identity."""
    if opponent == "heuristic_weak_v1":
        return HeuristicAgent(name="HeuristicWeakV1")
    if opponent == "heuristic_repaired_v1":
        return StrongHeuristicAgent(
            name="HeuristicRepairedV1", strength=2, avoid_certain_spins=True
        )
    raise ValueError(f"unsupported T2 opponent {opponent!r}")


def run_replay(spec: PairSpec, opponent: str) -> dict[str, Any]:
    """Replay one side of a pair and require the exact T0 terminal result."""
    params = TIGHT_PARAMS if spec.family == "tight_generated" else None
    track = generate_track(spec.track_seed, params)
    tracer = TracingStaticSearchAgent()
    agents = [_opponent_factory(opponent) for _ in range(spec.seats)]
    agents[spec.focal_seat] = tracer
    state = _play_scripted_game(track, spec.seats, dict(enumerate(agents)), spec.game_seed)
    player = state.get_player(spec.focal_seat)
    expected_place = spec.weak_place if opponent == "heuristic_weak_v1" else spec.repaired_place
    expected_rounds = spec.weak_rounds if opponent == "heuristic_weak_v1" else spec.repaired_rounds
    if player.finish_order != expected_place or state.round_num != expected_rounds:
        raise AssertionError(
            f"{spec.pair_id} {opponent} replay mismatch: "
            f"place/rounds={player.finish_order}/{state.round_num}, "
            f"expected={expected_place}/{expected_rounds}"
        )
    return {
        "opponent": opponent,
        "place": player.finish_order,
        "rounds": state.round_num,
        "final_heat": player.heat_available,
        "decisions": tracer.records,
    }


def _pair_score(pair: dict[str, Any]) -> int:
    """Rank manual narratives by semantic flags, not terminal place alone."""
    weights = {
        "top_k_pruned_better_plan": 4,
        "progress_tie_resource_dominated": 4,
        "cached_plan_suboptimal": 4,
        "cached_plan_not_in_current_candidates": 4,
        "determinization_argmax_disagreement": 1,
    }
    score = 0
    for replay in pair["replays"]:
        for decision in replay["decisions"]:
            for flag in decision["flags"]:
                score += weights.get(str(flag), 3 if "shadow_" in str(flag) else 0)
    return score


def _manual_review_ids(pairs: Sequence[dict[str, Any]]) -> list[str]:
    """Keep one baseline per stratum, then add six highest unread pairs."""
    baselines = [
        str(pair["pair_id"])
        for pair in pairs
        if int(pair["spec"]["sample_rank"]) == 0
    ]
    unread = [pair for pair in pairs if str(pair["pair_id"]) not in baselines]
    unread.sort(key=lambda pair: (-_pair_score(pair), str(pair["pair_id"])))
    return baselines + [str(pair["pair_id"]) for pair in unread[:6]]


def summarize_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate flags by opponent and track their cross-regime coverage."""
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    flag_pairs: dict[str, set[str]] = defaultdict(set)
    flag_families: dict[str, set[str]] = defaultdict(set)
    flag_seats: dict[str, set[int]] = defaultdict(set)
    decisions: Counter[str] = Counter()
    for pair in pairs:
        for replay in pair["replays"]:
            opponent = str(replay["opponent"])
            decisions[opponent] += len(replay["decisions"])
            for decision in replay["decisions"]:
                for flag in decision["flags"]:
                    flag = str(flag)
                    field_counts[opponent][flag] += 1
                    flag_pairs[flag].add(str(pair["pair_id"]))
                    flag_families[flag].add(str(pair["spec"]["family"]))
                    flag_seats[flag].add(int(pair["spec"]["seats"]))
    return {
        "pairs": len(pairs),
        "races": len(pairs) * 2,
        "decisions": dict(decisions),
        "flag_counts_by_opponent": {
            opponent: dict(counts) for opponent, counts in field_counts.items()
        },
        "flag_coverage": {
            flag: {
                "pairs": len(pair_ids),
                "families": sorted(flag_families[flag]),
                "seat_counts": sorted(flag_seats[flag]),
            }
            for flag, pair_ids in sorted(flag_pairs.items())
        },
    }


def render_narrative(pair: dict[str, Any]) -> str:
    """Render a complete but compact paired decision narrative."""
    spec = pair["spec"]
    lines = [
        f"T2 PAIR {pair['pair_id']}",
        (
            f"family={spec['family']} seats={spec['seats']} "
            f"track={spec['track_seed']} game={spec['game_seed']} "
            f"focal_seat={spec['focal_seat']} score={pair['manual_score']}"
        ),
    ]
    for replay in pair["replays"]:
        lines.append(
            f"\nFIELD {replay['opponent']} outcome=place "
            f"{replay['place']}/{spec['seats']} rounds={replay['rounds']}"
        )
        for decision in replay["decisions"]:
            chosen = decision["chosen"]
            shadows = decision["shadows"]
            lines.append(
                "  "
                f"r{decision['round']} lap={decision['lap']} pos={decision['position']} "
                f"rank={decision['rank']} heat={decision['heat']} "
                f"pick={chosen['key']} score={chosen['score']:.2f} "
                f"full={decision['full_best']['key']} "
                f"weak={shadows['weak']['key']} repaired={shadows['repaired']['key']} "
                f"flags={','.join(decision['flags']) or '-'}"
            )
    return "\n".join(lines) + "\n"


def _write_outputs(
    path: Path,
    logs_dir: Path,
    *,
    pairs: Sequence[dict[str, Any]],
    configuration: dict[str, Any],
    complete: bool,
    stop_reason: str | None,
    started: float,
) -> None:
    """Atomically persist JSON and refresh the selected manual narratives."""
    summary = summarize_pairs(pairs)
    manual_ids = _manual_review_ids(pairs) if pairs else []
    payload = {
        "complete": complete,
        "stop_reason": stop_reason,
        "experiment": "T2 search-paradox paired replay",
        "hypothesis": (
            "The same one-round raw-progress objective creates recognisable "
            "errors in search wins over repaired play and losses to weak play."
        ),
        "prior_confidence": 0.70,
        "configuration": configuration,
        "summary": summary,
        "manual_review_ids": manual_ids,
        "pairs": list(pairs),
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    logs_dir.mkdir(parents=True, exist_ok=True)
    selected = {str(pair["pair_id"]): pair for pair in pairs}
    for pair_id in manual_ids:
        (logs_dir / f"{pair_id}.txt").write_text(
            render_narrative(selected[pair_id]), encoding="utf-8"
        )


def run_t2(
    t0_path: Path,
    out_path: Path,
    logs_dir: Path,
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    max_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run paired T2 replays with outcome and wall-time gates."""
    payload = json.loads(t0_path.read_text(encoding="utf-8"))
    selected = select_pairs(
        payload, per_stratum=per_stratum, sample_seed=sample_seed
    )
    configuration = {
        "source": str(t0_path),
        "per_stratum": per_stratum,
        "sample_seed": sample_seed,
        "strata": len(FAMILIES) * len(SEAT_COUNTS),
        "expected_pairs": len(selected),
        "expected_races": len(selected) * 2,
        "max_seconds": max_seconds,
    }
    started = time.perf_counter()
    pairs: list[dict[str, Any]] = []
    print(
        f"T2 search-paradox pairs={len(selected)} races={len(selected) * 2} "
        f"max_seconds={max_seconds:.0f}",
        flush=True,
    )
    for index, spec in enumerate(selected, start=1):
        if time.perf_counter() - started >= max_seconds:
            reason = f"wall-time cap reached before {spec.pair_id}"
            _write_outputs(
                out_path,
                logs_dir,
                pairs=pairs,
                configuration=configuration,
                complete=False,
                stop_reason=reason,
                started=started,
            )
            return json.loads(out_path.read_text(encoding="utf-8"))
        replays = [
            run_replay(spec, "heuristic_weak_v1"),
            run_replay(spec, "heuristic_repaired_v1"),
        ]
        pair = {"pair_id": spec.pair_id, "spec": asdict(spec), "replays": replays}
        pair["manual_score"] = _pair_score(pair)
        pairs.append(pair)
        _write_outputs(
            out_path,
            logs_dir,
            pairs=pairs,
            configuration=configuration,
            complete=False,
            stop_reason=None,
            started=started,
        )
        print(
            f"T2 pair {index}/{len(selected)} {spec.pair_id} "
            f"places={replays[0]['place']}/{replays[1]['place']} "
            f"decisions={len(replays[0]['decisions'])}/{len(replays[1]['decisions'])} "
            f"wall={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    _write_outputs(
        out_path,
        logs_dir,
        pairs=pairs,
        configuration=configuration,
        complete=True,
        stop_reason=None,
        started=started,
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Run the command-line T2 diagnostic and print its compact summary."""
    args = _parse_args(argv)
    payload = run_t2(
        args.t0,
        args.out,
        args.logs_dir,
        per_stratum=args.per_stratum,
        sample_seed=args.sample_seed,
        max_seconds=args.max_seconds,
    )
    print(
        f"T2 {'complete' if payload['complete'] else 'stopped'} "
        f"pairs={payload['summary']['pairs']} "
        f"races={payload['summary']['races']} "
        f"wall={payload['elapsed_seconds']:.1f}s out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
