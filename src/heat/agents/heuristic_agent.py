"""Heuristic agent for the HEAT board game."""

from __future__ import annotations

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.track import Corner
from heat.engine import rules
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent


class HeuristicAgent(BaseAgent):
    """Agent that uses hand-crafted heuristics for each decision.

    Makes reasonable choices by scoring options based on game-state
    context such as corner proximity, heat conservation, and race
    progress.
    """

    def __init__(self, name: str = "HeuristicAgent") -> None:
        super().__init__(name=name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_corner(
        self, state: GameState, player_id: int
    ) -> tuple[Corner | None, int]:
        """Find the nearest corner ahead and the distance to it.

        Returns (corner, distance). If no corner ahead, returns (None, 999).
        """
        player = state.get_player(player_id)
        track = state.track
        pos = player.position

        best_corner: Corner | None = None
        best_dist = 999

        for corner in track.corners:
            # Distance from current position to corner start
            dist = (corner.start - pos) % track.length
            if dist == 0:
                # Already at/in the corner
                return corner, 0
            if dist < best_dist:
                best_dist = dist
                best_corner = corner

        return best_corner, best_dist

    def _distance_to_finish(self, state: GameState, player_id: int) -> int:
        """Spaces remaining in the race."""
        player = state.get_player(player_id)
        track = state.track
        laps_remaining = track.laps - player.lap
        spaces_this_lap = (track.length - player.position) % track.length
        if spaces_this_lap == 0 and laps_remaining > 0:
            spaces_this_lap = track.length
        return spaces_this_lap + laps_remaining * track.length

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        player = state.get_player(player_id)
        corner, corner_dist = self._next_corner(state, player_id)
        heat_in_hand = len(player.heat_in_hand)

        best_score = -9999
        best_choice = legal_gears[0]

        for new_gear, heat_cost in legal_gears:
            # Skip gears where we don't have enough playable cards
            playable = [c for c in player.hand if c.card_type != CardType.HEAT]
            if len(playable) < new_gear:
                continue

            score = new_gear * 10  # Base: prefer higher gears

            # Corner proximity penalty
            if corner is not None and corner_dist <= new_gear * 2:
                # Estimate max possible speed at this gear
                max_speed = new_gear * 6  # worst case: all 6-value cards
                if max_speed > corner.speed_limit:
                    penalty = (max_speed - corner.speed_limit) * 5
                    if corner_dist <= new_gear:
                        penalty *= 2  # very close, stronger penalty
                    score -= penalty

            # Heat conservation
            if player.heat_available <= 1:
                if new_gear >= 3:
                    score -= 30  # strongly penalize high gears
            elif player.heat_available <= 3:
                if new_gear == 4:
                    score -= 15  # mild penalty for gear 4

            # Gear shift cost
            score -= heat_cost * 15

            # Cooldown bonus: if heat cards in hand, favor low gears
            if heat_in_hand > 0:
                if new_gear == 1:
                    score += min(heat_in_hand, 3) * 8  # up to +24
                elif new_gear == 2:
                    score += min(heat_in_hand, 1) * 8  # up to +8

            if score > best_score:
                best_score = score
                best_choice = (new_gear, heat_cost)

        return best_choice

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        if len(legal_plays) == 1:
            return legal_plays[0]

        player = state.get_player(player_id)
        corner, corner_dist = self._next_corner(state, player_id)

        best_score = -9999
        best_play = legal_plays[0]

        for play in legal_plays:
            # Calculate speed (stress = 0 in hand, estimated ~3.5 for scoring)
            speed = 0
            stress_count = 0
            for card in play:
                if card.card_type == CardType.STRESS:
                    stress_count += 1
                    speed += 3.5  # estimated stress resolution value
                else:
                    speed += card.value

            score = speed * 3  # speed score

            # Stress bonus: stress cards save good cards for later
            score += stress_count * 5

            # Corner penalty
            if corner is not None and corner_dist <= player.gear * 2:
                estimated_total_speed = speed
                if estimated_total_speed > corner.speed_limit:
                    overshoot = estimated_total_speed - corner.speed_limit
                    score -= overshoot * 8
                    if overshoot > player.heat_available:
                        score -= 50  # spin-out risk

            if score > best_score:
                best_score = score
                best_play = play

        return best_play

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        player = state.get_player(player_id)
        heat_in_hand = len(player.heat_in_hand)
        corner, corner_dist = self._next_corner(state, player_id)
        dist_to_finish = self._distance_to_finish(state, player_id)

        # Cooldown: always cool as many heat cards from hand as possible
        base_cooldown = min(max_cooldown, heat_in_hand)
        cooldown_count = base_cooldown

        # Adrenaline cooldown: use if heat in hand exceeds base cooldown
        use_adrenaline_cooldown = False
        if has_adrenaline and heat_in_hand > base_cooldown:
            use_adrenaline_cooldown = True
            cooldown_count = min(max_cooldown + 1, heat_in_hand)

        # Boost decision
        use_boost = False
        if can_boost:
            near_corner_at_limit = (
                corner is not None
                and corner_dist <= 6
                and player.speed_from_cards >= corner.speed_limit
            )
            if player.heat_available >= 3 and not near_corner_at_limit:
                use_boost = True
            elif player.lap >= state.track.laps and dist_to_finish <= 10:
                # Final lap, near finish: boost aggressively
                use_boost = True

        # Adrenaline speed: almost always use (+1 free)
        use_adrenaline_speed = False
        if has_adrenaline:
            at_corner_limit = (
                corner is not None
                and corner_dist == 0
                and player.speed_from_cards == corner.speed_limit
                and player.heat_available <= 1
            )
            if not at_corner_limit:
                use_adrenaline_speed = True

        return ReactDecision(
            cooldown_count=cooldown_count,
            use_boost=use_boost,
            use_adrenaline_speed=use_adrenaline_speed,
            use_adrenaline_cooldown=use_adrenaline_cooldown,
        )

    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        player = state.get_player(player_id)
        track = state.track

        # Check if slipstream +2 would cross into new corners
        new_pos = (player.position + 2) % track.length
        crossed = rules.corners_crossed(player.position, new_pos, track)

        if not crossed:
            # No new corners: always take free speed
            return True

        # New corners: compute additional heat cost
        speed = rules.corner_speed_for_check(player)
        total_cost = 0
        for corner in crossed:
            total_cost += rules.corner_heat_cost(speed, corner)

        # Take if affordable and not too expensive
        if total_cost <= player.heat_available and total_cost <= 2:
            return True
        return False

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        player = state.get_player(player_id)
        playable = [
            c for c in player.hand
            if c.card_type in (CardType.SPEED, CardType.UPGRADE)
        ]

        # Keep everything unless hand has plenty of playable cards
        if len(playable) <= player.gear + 2:
            return []

        # Too many playable cards: discard low-value speed cards to cycle deck
        to_discard: list[Card] = []
        for card in discardable:
            # Never discard upgrade cards
            if card.card_type == CardType.UPGRADE:
                continue
            # Discard value-1 speed cards
            if card.card_type == CardType.SPEED and card.value == 1:
                to_discard.append(card)

        return to_discard
