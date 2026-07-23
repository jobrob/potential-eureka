"""Versioned T2 repairs for the static heuristic and search population.

The V1 classes remain frozen for evidence replay.  These V2 subclasses keep
their search depth and heuristic objectives unchanged while repairing the
three concrete defects surfaced by T2: cross-round plan-cache collisions,
unsafe top-k search admission, and avoidable high-risk Stress plays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from heat.agents import _move_eval as ME
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.static_search import StaticSearchAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.models.cards import Card, CardType
from heat.models.game_state import GameState


HIGH_STRESS_SPIN_RISK = 0.5


def _safe_turn_signature(state: GameState, player_id: int) -> tuple[int, ...]:
    """Identify a turn with round, resources, and exact hand contents.

    The stable integer fingerprint also remains compatible with the search
    agent's platform-independent per-turn seed mixer.
    """
    player = state.get_player(player_id)
    hand_fingerprint = 17
    for card_id in sorted(card.id for card in player.hand):
        for byte in card_id.encode("utf-8"):
            hand_fingerprint = (hand_fingerprint * 131 + byte) & 0x7FFFFFFF
    return (
        player_id,
        player.position,
        player.lap,
        len(player.hand),
        player.gear,
        state.round_num,
        player.heat_available,
        hand_fingerprint,
    )


def _move_risk(
    state: GameState,
    player_id: int,
    gear: tuple[int, int],
    cards: tuple[Card, ...],
) -> ME.MoveEval:
    """Return the cheap prior plus high-Stress-flip spin risk for one plan."""
    player = state.get_player(player_id)
    expected_stress = ME.expected_basic_value(player)
    max_basic = ME.max_basic_value(player)
    expected_speed = ME.play_speed(cards, expected_stress)
    variance = ME.play_speed_variance(cards, expected_stress, max_basic)
    return ME.evaluate_move(
        state,
        player_id,
        expected_speed=expected_speed,
        heat_spent=gear[1],
        from_position=player.position,
        from_lap=player.lap,
        planned_gear=gear[0],
        heat_price=ME.DEFAULT_HEAT_PRICE,
        horizon_corners=1,
        opponent_aware=True,
        blocking=False,
        enable_solvency=True,
        speed_variance=variance,
    )


def _contains_stress(cards: tuple[Card, ...]) -> bool:
    """Return whether a play delegates any speed to a Stress flip."""
    return any(card.card_type == CardType.STRESS for card in cards)


def _candidate_cards(candidate: Any) -> tuple[Card, ...]:
    """Extract cards from either a play or a joint ``(gear, play)`` plan."""
    if (
        isinstance(candidate, tuple)
        and len(candidate) == 2
        and isinstance(candidate[0], tuple)
        and isinstance(candidate[1], tuple)
    ):
        return cast(tuple[Card, ...], candidate[1])
    return cast(tuple[Card, ...], candidate)


class HeuristicV2Agent(HeuristicAgent):
    """Weak heuristic with a narrow avoidable Stress-spin safety rule."""

    VERSION = "HeuristicWeakV2"

    def __init__(self, name: str = VERSION) -> None:
        super().__init__(name=name)

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        """Avoid high-risk Stress plays when a zero-risk legal play exists."""
        if len(legal_plays) <= 1:
            return legal_plays[0]
        gear = (state.get_player(player_id).gear, 0)
        assessed = [(_move_risk(state, player_id, gear, play), play) for play in legal_plays]
        if any(result.p_spinout == 0.0 for result, _play in assessed):
            safe_set = {
                play
                for result, play in assessed
                if not (
                    _contains_stress(play)
                    and result.p_spinout >= HIGH_STRESS_SPIN_RISK
                )
            }
            filtered = [play for play in legal_plays if play in safe_set]
            if filtered:
                legal_plays = filtered
        return super().choose_cards(state, player_id, legal_plays)


class RepairedHeuristicV2Agent(StrongHeuristicAgent):
    """Repaired heuristic with collision-safe caching and Stress safety."""

    VERSION = "HeuristicRepairedV2"

    def __init__(self, name: str = VERSION) -> None:
        super().__init__(name=name, strength=2, avoid_certain_spins=True)

    def _turn_signature(self, state: GameState, player_id: int) -> tuple[int, ...]:
        """Prevent a cached plan from crossing a round or hand boundary."""
        return _safe_turn_signature(state, player_id)

    def _reject_avoidable_certain_spins[T](
        self, scored: Sequence[tuple[ME.MoveEval, bool, T]]
    ) -> list[tuple[ME.MoveEval, bool, T]]:
        """Also veto high-risk Stress plans when a zero-risk plan exists."""
        candidates = super()._reject_avoidable_certain_spins(scored)
        if not any(result.p_spinout == 0.0 for result, _certain, _plan in candidates):
            return candidates
        filtered = [
            row
            for row in candidates
            if not (
                _contains_stress(_candidate_cards(row[2]))
                and row[0].p_spinout >= HIGH_STRESS_SPIN_RISK
            )
        ]
        return filtered or candidates


class StaticSearchV2Agent(StaticSearchAgent):
    """StaticSearchV1 with safe cache identity and candidate admission."""

    VERSION = "StaticSearchV2"

    def __init__(self, name: str = VERSION) -> None:
        super().__init__(name=name)

    def _turn_signature(self, state: GameState, player_id: int) -> tuple[int, ...]:
        """Prevent a cached plan from crossing a round or hand boundary."""
        return _safe_turn_signature(state, player_id)

    def _prune_top_k(
        self,
        state: GameState,
        player_id: int,
        candidates: list[tuple[tuple[int, int], tuple[Card, ...]]],
    ) -> list[tuple[tuple[int, int], tuple[Card, ...]]]:
        """Veto certain spins and reserve one zero-risk top-k slot.

        The prior incorporates the existing high-flip risk estimate.  Certain
        spins are removed only when an alternative exists; if all leading
        candidates retain partial risk, the best zero-risk plan replaces the
        last top-k slot so the exact rollout can compare it.
        """
        scored = []
        for index, (gear, cards) in enumerate(candidates):
            result = _move_risk(state, player_id, gear, cards)
            certain = StrongHeuristicAgent._play_guarantees_spin(
                state, player_id, cards, gear[1]
            )
            scored.append((result, certain, index, gear, cards))

        eligible = scored
        if any(not certain for _result, certain, _idx, _gear, _cards in scored):
            eligible = [row for row in scored if not row[1]]
        eligible.sort(key=lambda row: (-row[0].value, row[2]))

        if self.top_k is None or len(eligible) <= self.top_k:
            return [(gear, cards) for _result, _certain, _idx, gear, cards in eligible]

        selected = eligible[: self.top_k]
        safe = [row for row in eligible if row[0].p_spinout == 0.0]
        if safe and not any(row[0].p_spinout == 0.0 for row in selected):
            selected[-1] = safe[0]
        return [(gear, cards) for _result, _certain, _idx, gear, cards in selected]
