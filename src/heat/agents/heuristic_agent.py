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

    def _compute_corner_heat_cost(
        self, state: GameState, player_id: int, from_position: int, speed: int
    ) -> tuple[int, list[Corner]]:
        """Compute total corner heat cost if moving `speed` spaces from `from_position`."""
        track = state.track
        player = state.get_player(player_id)
        new_pos, _ = rules.calculate_move_position(from_position, speed, track, player.lap)
        crossed = rules.corners_crossed(from_position, new_pos, track)
        total_cost = sum(rules.corner_heat_cost(speed, corner) for corner in crossed)
        return total_cost, crossed

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
        heat_in_hand = len(player.heat_in_hand)

        # Estimate speed from hand: get playable card values
        playable = [c for c in player.hand if c.card_type != CardType.HEAT]
        playable_values = []
        for c in playable:
            if c.card_type == CardType.STRESS:
                playable_values.append(2.5)  # stress estimate for 1-4 deck
            else:
                playable_values.append(float(c.value))
        playable_values.sort(reverse=True)

        best_score = -9999
        best_choice = legal_gears[0]

        for new_gear, heat_cost in legal_gears:
            # Skip gears where we don't have enough playable cards (cluttered)
            if len(playable) < new_gear:
                continue

            score = new_gear * 10  # Base: prefer higher gears

            # Estimate speed as sum of top N card values (N = gear)
            estimated_speed = int(sum(playable_values[:new_gear]))

            # Compute actual corner heat cost for this estimated speed
            corner_cost, crossed = self._compute_corner_heat_cost(
                state, player_id, player.position, estimated_speed
            )

            # Corner cost penalty
            if corner_cost > 0:
                score -= corner_cost * 10
                # Spinout risk: corner cost exceeds available heat
                if corner_cost > player.heat_available:
                    score -= 100

            # Heat conservation: penalize high gears when heat is critically low
            if player.heat_available <= 2 and new_gear >= 3:
                score -= 40

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

        best_score = -9999
        best_play = legal_plays[0]

        for play in legal_plays:
            # Calculate speed (stress estimated at 2.5 for 1-4 deck)
            speed = 0
            stress_count = 0
            for card in play:
                if card.card_type == CardType.STRESS:
                    stress_count += 1
                    speed += 2.5  # estimated stress resolution for 1-4 deck
                else:
                    speed += card.value

            score = speed * 3  # speed score

            # Stress bonus: stress cards save good cards for later
            score += stress_count * 5

            # Compute actual corner heat cost for this combo
            int_speed = int(speed)
            corner_cost, crossed = self._compute_corner_heat_cost(
                state, player_id, player.position, int_speed
            )

            if crossed:
                # Penalize by heat cost
                score -= corner_cost * 12

                # Spinout risk: corner cost exceeds available heat
                if corner_cost > player.heat_available:
                    score -= 200

                # Reward combos that stay at or under corner speed limits
                all_under = all(
                    int_speed <= corner.speed_limit for corner in crossed
                )
                if all_under:
                    score += 10

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

        # Boost decision — the critical fix
        use_boost = False
        if can_boost:
            # Check corners actually crossed this turn
            crossed = rules.corners_crossed(
                player.turn_start_position, player.position, state.track
            )

            # If speed_from_cards already exceeds ANY crossed corner's limit, DO NOT boost
            over_corner_limit = any(
                player.speed_from_cards > c.speed_limit for c in crossed
            )

            if over_corner_limit:
                use_boost = False
            elif player.heat_available <= 2:
                # Low heat: DO NOT boost
                use_boost = False
            elif not crossed and player.heat_available >= 4:
                # No corners crossed AND plenty of heat: boost
                use_boost = True
            elif player.lap >= state.track.laps and dist_to_finish <= 10:
                # Final lap near finish: boost only if estimated corner cost is affordable
                estimated_boost_speed = player.speed_from_cards + 2  # rough estimate
                cost, _ = self._compute_corner_heat_cost(
                    state, player_id, player.turn_start_position, estimated_boost_speed
                )
                if cost <= player.heat_available:
                    use_boost = True

        # Adrenaline speed: use unless it would increase corner costs with low heat
        use_adrenaline_speed = False
        if has_adrenaline:
            # Check corners from this turn
            crossed = rules.corners_crossed(
                player.turn_start_position, player.position, state.track
            )
            # Compute cost with +1 speed
            current_speed = player.speed_from_cards + player.speed_from_boost
            cost_with_adrenaline = sum(
                rules.corner_heat_cost(current_speed + 1, c) for c in crossed
            )
            cost_without = sum(
                rules.corner_heat_cost(current_speed, c) for c in crossed
            )
            extra_cost = cost_with_adrenaline - cost_without
            if extra_cost > 0 and player.heat_available <= 2:
                use_adrenaline_speed = False
            else:
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
