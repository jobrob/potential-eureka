"""Strong, opponent-aware scripted agent for the HEAT board game (Sprint 6E).

The strength bar the capstone is measured against. Two architectural
commitments make it coherent where :class:`HeuristicAgent` is piecemeal:

(A) **One currency.** Every decision is scored in expected net spaces of
    progress by the pure :func:`heat.agents._move_eval.evaluate_move`. There
    are no per-call-site magic weights -- only ``heat_price`` (spaces per heat)
    and a spin-out loss model, both interpretable.

(B) **Plan once, execute consistently.** At ``choose_gear`` the agent enumerates
    the joint ``legal_gear_shifts x legal_card_plays`` space, scores each
    through the evaluator, and caches the best ``(gear, card_play)`` keyed by a
    turn signature. ``choose_cards`` returns the cached play (re-validated
    against the legal set the engine hands in), and ``choose_react`` /
    ``choose_slipstream`` refine the intent against the now-realised state.

A single ``strength`` integer (0..3) selects a monotone difficulty ladder. The
rungs separate because each turns on a component that changes *decisions*, not
just a tiebreak (Sprint 6E redesign):

    0  myopic single-turn, opponent-blind  (~ old heuristic via new evaluator)
    1  + joint plan + real forward solvency + spin-out aversion
    2  + relative_term (saturating gap change) + rank-adjusted risk posture
       + two-corner horizon + end-game decay (default)
    3  + real block projection + three-corner horizon

The agent is deterministic given the state. An optional ``seed`` seeds a local
``random.Random`` used only to break ties between equal-valued plans, so a
seeded ``run_batch`` is reproducible. No global RNG is touched.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from heat.models.cards import Card, CardType
from heat.models.game_state import GameState
from heat.models.player_state import PlayerState
from heat.models.track import Corner
from heat.engine import rules
from heat.engine.phases import ReactDecision
from heat.agents.base import BaseAgent
from heat.agents import _move_eval as ME


# Turn signature: identifies the planning context so a cached plan is only
# reused within the same turn for the same player.
_TurnSig = tuple[int, ...]


class StrongHeuristicAgent(BaseAgent):
    """Opponent-aware, economy-planning scripted agent (see module docstring)."""

    def __init__(
        self,
        name: str = "StrongHeuristic",
        *,
        strength: int = 2,
        heat_price: float | None = None,
        seed: int | None = None,
        avoid_certain_spins: bool = True,
    ) -> None:
        super().__init__(name=name)
        if not (0 <= strength <= 3):
            raise ValueError(f"strength must be in 0..3, got {strength}")
        self.strength = strength
        self.base_heat_price = (
            ME.DEFAULT_HEAT_PRICE if heat_price is None else float(heat_price)
        )
        self._rng = random.Random(seed) if seed is not None else None
        self._avoid_certain_spins = avoid_certain_spins
        # Cached plan for the current turn.
        self._plan_sig: _TurnSig | None = None
        self._plan_gear: tuple[int, int] | None = None
        self._plan_cards: tuple[Card, ...] | None = None

        # --- Ladder feature flags, derived from strength (monotone) ---
        # strength 0 is the deliberately myopic floor: gear and cards are
        # optimised *separately* (reproducing the old heuristic's defect #1)
        # so the ladder has room to climb. Each flag is set once here so the
        # difficulty rungs are an explicit, inspectable configuration.
        s = strength
        self._joint_plan = s >= 1       # joint gear+cards planning
        self._solvency = s >= 1         # real forward heat-solvency + spin aversion
        self._opponent_aware = s >= 2   # relative_term (saturating gap change, P0)
        self._endgame = s >= 2          # heat-price decay near the finish
        self._position_price = s >= 2   # rank-adjusted risk posture (price + spin, P0)
        self._blocking = s >= 3         # real block projection (P3)
        # Lookahead depth in corners. A single-corner solvency horizon is the
        # sweet spot on the benchmark tracks: the real forward-heat projection
        # (P2) already prices the *next* corner honestly, and deeper horizons
        # proved measurably over-conservative (they cost own-progress without a
        # commensurate solvency gain). The higher rungs separate via the
        # relative objective / risk posture / block projection, not via deeper
        # solvency lookahead.
        self._horizon = {0: 0, 1: 1, 2: 1, 3: 1}[s]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _turn_signature(self, state: GameState, player_id: int) -> _TurnSig:
        player = state.get_player(player_id)
        return (
            player_id,
            player.position,
            player.lap,
            len(player.hand),
            player.gear,
        )

    def _risk_posture(
        self, state: GameState, player_id: int
    ) -> tuple[float, float]:
        """Return ``(heat_price_eff, spinout_loss_eff)`` for the current state.

        At strength >= 2 (``_position_price``) the rank-dependent risk posture
        (:func:`ME.effective_risk`) replaces the WIP's inert +/-10% nudge: a
        leader pays MORE per heat and fears a spin MORE (conserve); a trailer
        pays LESS and discounts spins (push). The end-game heat-price decay
        (heat is use-it-or-lose-it near the line on the final lap) is applied
        *after* the rank adjustment, multiplicatively. Lower rungs keep a flat
        base price and the base spin-out loss.
        """
        player = state.get_player(player_id)
        track = state.track

        if self._position_price:
            price, spin_loss = ME.effective_risk(
                state, player_id, self.base_heat_price
            )
        else:
            price = self.base_heat_price
            spin_loss = ME.SPINOUT_LOSS_SPACES

        # End-game decay on the final lap: heat is use-it-or-lose-it. Applied
        # multiplicatively after the rank adjustment.
        if self._endgame and player.lap >= track.laps and track.length > 0:
            dist_to_finish = track.length - player.position
            # Linear decay over the final ~8 spaces.
            frac = max(0.0, min(1.0, dist_to_finish / 8.0))
            price *= frac

        return max(0.0, price), max(0.0, spin_loss)

    def _effective_heat_price(self, state: GameState, player_id: int) -> float:
        """Backwards-compatible accessor: the rank/end-game adjusted heat price."""
        price, _ = self._risk_posture(state, player_id)
        return price

    def _select_best[T](
        self, scored: Sequence[tuple[float, T]]
    ) -> T | None:
        """Pick the highest-scoring option, breaking ties deterministically.

        Ties are broken by the optional local RNG when present (still
        reproducible under a fixed seed); otherwise by first-seen order.
        """
        if not scored:
            return None
        best_value = max(s for s, _ in scored)
        winners = [obj for s, obj in scored if s == best_value]
        if len(winners) == 1 or self._rng is None:
            return winners[0]
        return winners[self._rng.randrange(len(winners))]

    def _reject_avoidable_certain_spins[T](
        self, scored: Sequence[tuple[ME.MoveEval, bool, T]]
    ) -> list[tuple[ME.MoveEval, bool, T]]:
        """Remove certain-spin plans when any legal non-certain plan exists.

        H1 found tight-corner recovery loops where intended progress outweighed
        the finite spin penalty. This is a safety invariant rather than another
        weight: unavoidable spins and plans with only partial risk remain legal.
        The strength-0 floor stays unchanged because it intentionally disables
        the forward-solvency model that supplies ``p_spinout``.
        """
        candidates = list(scored)
        if not self._avoid_certain_spins or not self._solvency or not any(
            not guaranteed for _result, guaranteed, _candidate in candidates
        ):
            return candidates
        return [
            (result, guaranteed, candidate)
            for result, guaranteed, candidate in candidates
            if not guaranteed
        ]

    @staticmethod
    def _play_guarantees_spin(
        state: GameState,
        player_id: int,
        play: tuple[Card, ...],
        heat_spent: int,
    ) -> bool:
        """Return whether even the play's lowest possible speed must spin.

        Stress cards can flip below their expected value. H1's lock fixtures
        rely on that escape route, so ``MoveEval.p_spinout``—which deliberately
        scores the mean and high-flip case—is not a proof of certainty. The
        invariant instead checks the lowest Basic card the player owns.
        """
        player = state.get_player(player_id)
        basics = [
            card.value
            for card in (*tuple(player.deck), *player.hand)
            if card.card_type == CardType.SPEED
        ]
        min_basic = min(basics) if basics else 1
        stress_count = sum(
            card.card_type == CardType.STRESS for card in play
        )
        min_speed = float(rules.calculate_speed(play) + stress_count * min_basic)
        crossed = ME.corners_crossed_by_move(
            state.track,
            player.position,
            player.lap,
            int(min_speed),
        )
        corner_cost = ME.corner_cost_for_speed(crossed, min_speed)
        return heat_spent + corner_cost > player.heat_available

    # ------------------------------------------------------------------
    # Joint planner (defects #1, #3, #4)
    # ------------------------------------------------------------------

    def _plan_turn(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[Card, ...]]:
        """Score the joint (gear, card_play) space; return the best pair."""
        player = state.get_player(player_id)
        heat_price, spin_loss = self._risk_posture(state, player_id)
        exp_stress = ME.expected_basic_value(player)
        max_basic = ME.max_basic_value(player)

        from_position = player.position
        from_lap = player.lap

        scored: list[
            tuple[ME.MoveEval, bool, tuple[tuple[int, int], tuple[Card, ...]]]
        ] = []

        for gear, gear_heat in legal_gears:
            plays = rules.legal_card_plays(player.hand, gear)
            for play in plays:
                exp_speed = ME.play_speed(play, exp_stress)
                var = (
                    ME.play_speed_variance(play, exp_stress, max_basic)
                    if self._solvency
                    else 0.0
                )
                result = ME.evaluate_move(
                    state,
                    player_id,
                    expected_speed=exp_speed,
                    heat_spent=gear_heat,
                    from_position=from_position,
                    from_lap=from_lap,
                    planned_gear=gear,
                    heat_price=heat_price,
                    horizon_corners=self._horizon,
                    opponent_aware=self._opponent_aware,
                    blocking=self._blocking,
                    enable_solvency=self._solvency,
                    speed_variance=var,
                    spinout_loss=spin_loss,
                )
                scored.append(
                    (
                        result,
                        self._play_guarantees_spin(
                            state, player_id, play, gear_heat
                        ),
                        ((gear, gear_heat), play),
                    )
                )

        eligible = self._reject_avoidable_certain_spins(scored)
        chosen = self._select_best(
            [
                (result.value, candidate)
                for result, _guaranteed, candidate in eligible
            ]
        )
        if chosen is None:
            # Degenerate fallback: stay at the cheapest legal gear with any play.
            fallback_gear = legal_gears[0]
            plays = rules.legal_card_plays(player.hand, fallback_gear[0])
            return fallback_gear, plays[0]
        return chosen

    def _myopic_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Decoupled gear choice for the strength-0 floor.

        Scores each gear on the *estimated* best play (top-``gear`` expected
        card values) rather than fully evaluating every play, then leaves the
        actual card choice to ``choose_cards``. This reproduces the old
        heuristic's gear/cards split (defect #1) so strength 0 ties it.
        """
        player = state.get_player(player_id)
        heat_price = self._effective_heat_price(state, player_id)
        exp_stress = ME.expected_basic_value(player)

        # Expected per-card values, best-first (heat cards contribute 0).
        values: list[float] = []
        for c in player.hand:
            if c.card_type == CardType.STRESS:
                values.append(exp_stress)
            elif c.card_type == CardType.HEAT:
                values.append(0.0)
            else:
                values.append(float(c.value))
        values.sort(reverse=True)

        scored: list[tuple[float, tuple[int, int]]] = []
        for gear, gear_heat in legal_gears:
            est_speed = sum(values[:gear]) if values else 0.0
            result = ME.evaluate_move(
                state,
                player_id,
                expected_speed=est_speed,
                heat_spent=gear_heat,
                from_position=player.position,
                from_lap=player.lap,
                planned_gear=gear,
                heat_price=heat_price,
                horizon_corners=0,
                opponent_aware=False,
                blocking=False,
                enable_solvency=False,
            )
            scored.append((result.value, (gear, gear_heat)))
        chosen = self._select_best(scored)
        return chosen if chosen is not None else legal_gears[0]

    def _ensure_plan(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> None:
        sig = self._turn_signature(state, player_id)
        if self._plan_sig == sig and self._plan_gear is not None:
            return
        if self._joint_plan:
            gear, cards = self._plan_turn(state, player_id, legal_gears)
            self._plan_cards = cards
        else:
            # Myopic floor: choose gear independently, leave cards to recompute.
            gear = self._myopic_gear(state, player_id, legal_gears)
            self._plan_cards = None
        self._plan_sig = sig
        self._plan_gear = gear

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def choose_gear(
        self,
        state: GameState,
        player_id: int,
        legal_gears: list[tuple[int, int]],
    ) -> tuple[int, int]:
        self._ensure_plan(state, player_id, legal_gears)
        gear = self._plan_gear
        if gear is None or gear not in legal_gears:
            # Recompute against the handed legal set (state may have shifted).
            gear, cards = self._plan_turn(state, player_id, legal_gears)
            self._plan_sig = self._turn_signature(state, player_id)
            self._plan_gear = gear
            self._plan_cards = cards
        return gear

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        if len(legal_plays) == 1:
            return legal_plays[0]

        sig = self._turn_signature(state, player_id)
        # Honour the cached plan when it is still valid for this turn.
        if (
            self._plan_sig == sig
            and self._plan_cards is not None
            and self._plan_cards in legal_plays
        ):
            return self._plan_cards

        # Cache miss / stale: recompute the best play at the current gear.
        return self._best_play(state, player_id, legal_plays)

    def _best_play(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        player = state.get_player(player_id)
        heat_price, spin_loss = self._risk_posture(state, player_id)
        exp_stress = ME.expected_basic_value(player)
        max_basic = ME.max_basic_value(player)
        scored: list[tuple[ME.MoveEval, bool, tuple[Card, ...]]] = []
        for play in legal_plays:
            exp_speed = ME.play_speed(play, exp_stress)
            var = (
                ME.play_speed_variance(play, exp_stress, max_basic)
                if self._solvency
                else 0.0
            )
            result = ME.evaluate_move(
                state,
                player_id,
                expected_speed=exp_speed,
                heat_spent=0,
                from_position=player.position,
                from_lap=player.lap,
                planned_gear=player.gear,
                heat_price=heat_price,
                horizon_corners=self._horizon,
                opponent_aware=self._opponent_aware,
                blocking=self._blocking,
                enable_solvency=self._solvency,
                speed_variance=var,
                spinout_loss=spin_loss,
            )
            scored.append(
                (
                    result,
                    self._play_guarantees_spin(state, player_id, play, 0),
                    play,
                )
            )
        eligible = self._reject_avoidable_certain_spins(scored)
        chosen = self._select_best(
            [
                (result.value, play)
                for result, _guaranteed, play in eligible
            ]
        )
        return chosen if chosen is not None else legal_plays[0]

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
        heat_price = self._effective_heat_price(state, player_id)

        # --- Cooldown: nearly free value (un-clogs hand, refills heat pool). ---
        base_cooldown = min(max_cooldown, heat_in_hand)
        cooldown_count = base_cooldown
        use_adrenaline_cooldown = False
        if has_adrenaline and heat_in_hand > base_cooldown:
            use_adrenaline_cooldown = True
            cooldown_count = min(max_cooldown + 1, heat_in_hand)

        card_speed = rules.corner_speed_for_check(player)

        if not self._solvency:
            # Strength-0 floor: myopic "heat economy" -- boost/adrenaline are
            # decided on the *expected* flip with no worst-case/variance guard
            # (the old heuristic's blind spot). The heat-economy rung (>=1)
            # replaces this with the solvent logic below.
            crossed = self._corners_after_extra(state, player, 0)
            use_boost = self._myopic_boost(
                state, player_id, can_boost, card_speed, heat_price
            )
            use_adrenaline_speed = has_adrenaline and self._myopic_adrenaline(
                player, crossed, card_speed, can_boost and use_boost
            )
        else:
            # --- Adrenaline speed: free +1 space; decline only if even its
            # modest extra corner cost cannot be paid (else it just spins us). ---
            use_adrenaline_speed = has_adrenaline and self._adrenaline_safe(
                state, player_id, card_speed
            )

            # --- Boost: pay 1 heat for a random flip (+movement & +corner
            # speed). The single end-of-turn corner check covers the whole
            # boosted path, so only boost when the WORST plausible flip stays
            # solvent (incl. the +1 if adrenaline is taken) and the expected
            # space gain beats the heat. ---
            adr = 1 if use_adrenaline_speed else 0
            use_boost = self._should_boost(
                state, player_id, can_boost, card_speed, heat_price, adr
            )

        return ReactDecision(
            cooldown_count=cooldown_count,
            use_boost=use_boost,
            use_adrenaline_speed=use_adrenaline_speed,
            use_adrenaline_cooldown=use_adrenaline_cooldown,
        )

    def _spaces_moved(self, player: PlayerState, state: GameState) -> int:
        return (
            (player.lap - player.turn_start_lap) * state.track.length
            + (player.position - player.turn_start_position)
        )

    def _corners_after_extra(
        self,
        state: GameState,
        player: PlayerState,
        extra_movement: int,
    ) -> list[Corner]:
        """Corners crossed over the whole turn if the car moves ``extra_movement``
        further from its current (post-card) position.

        The engine checks every corner from ``turn_start_position`` through the
        final position in one pass, so extra react-phase movement can pull in
        corners the card movement alone did not reach.
        """
        track = state.track
        final_pos, _ = rules.calculate_move_position(
            player.position, extra_movement, track, player.lap
        )
        spaces_moved = self._spaces_moved(player, state) + extra_movement
        return rules.corners_crossed(
            player.turn_start_position, final_pos, track, spaces_moved=spaces_moved
        )

    def _adrenaline_safe(
        self,
        state: GameState,
        player_id: int,
        card_speed: int,
    ) -> bool:
        """Take the free +1 speed/movement unless it forces an unaffordable corner.

        Adrenaline adds 1 to both movement and the corner-check speed. Declining
        only matters when the resulting corner cost exceeds the heat pool (the
        car would spin out); otherwise the extra space is pure gain.
        """
        player = state.get_player(player_id)
        crossed = self._corners_after_extra(state, player, 1)
        cost = sum(rules.corner_heat_cost(card_speed + 1, c) for c in crossed)
        return cost <= player.heat_available

    def _should_boost(
        self,
        state: GameState,
        player_id: int,
        can_boost: bool,
        card_speed: int,
        heat_price: float,
        adr: int,
    ) -> bool:
        if not can_boost:
            return False
        player = state.get_player(player_id)
        if player.heat_available <= 0:
            return False  # cannot pay the 1 heat the flip costs

        exp_flip = ME.expected_basic_value(player)
        max_flip = int(round(ME.max_basic_value(player)))

        # Worst case: highest owned flip plus the +1 if adrenaline is taken.
        worst_move = max_flip + adr
        worst_speed = card_speed + max_flip + adr
        worst_crossed = self._corners_after_extra(state, player, worst_move)
        worst_cost = sum(
            rules.corner_heat_cost(worst_speed, c) for c in worst_crossed
        )
        # Boost pays 1 heat for the flip; the rest of the pool must still cover
        # the worst-case corner cost. Refuse if a bad flip could spin us.
        if 1 + worst_cost > player.heat_available:
            return False

        # Expected value: exp_flip extra spaces minus heat priced in (1 for the
        # flip + the expected extra corner overage the flip alone adds).
        base_crossed = self._corners_after_extra(state, player, adr)
        base_cost = sum(
            rules.corner_heat_cost(card_speed + adr, c) for c in base_crossed
        )
        exp_move = int(round(exp_flip)) + adr
        exp_crossed = self._corners_after_extra(state, player, exp_move)
        exp_cost = sum(
            rules.corner_heat_cost(int(card_speed + exp_flip) + adr, c)
            for c in exp_crossed
        )
        exp_extra_overage = max(0, exp_cost - base_cost)
        gain = exp_flip - heat_price * (1 + exp_extra_overage)
        return gain > 0.0

    def _myopic_boost(
        self,
        state: GameState,
        player_id: int,
        can_boost: bool,
        card_speed: int,
        heat_price: float,
    ) -> bool:
        """Strength-0 boost: expected-flip greedy, no worst-case/variance guard.

        Deliberately reproduces the naive boost decision (prices only the
        *expected* extra overage on the already-crossed corners) so the floor
        rung spins out the way the old heuristic does. The solvent rungs use
        :meth:`_should_boost` instead.
        """
        if not can_boost:
            return False
        player = state.get_player(player_id)
        if player.heat_available <= 0:
            return False
        exp_flip = ME.expected_basic_value(player)
        crossed = self._corners_after_extra(state, player, 0)
        extra_overage = 0
        for c in crossed:
            cur = rules.corner_heat_cost(card_speed, c)
            after = rules.corner_heat_cost(int(card_speed + exp_flip), c)
            extra_overage += max(0, after - cur)
        boost_cost_heat = 1 + extra_overage
        if boost_cost_heat > player.heat_available:
            return False
        return exp_flip - heat_price * boost_cost_heat > 0.0

    def _myopic_adrenaline(
        self,
        player: PlayerState,
        crossed: list[Corner],
        card_speed: int,
        boosting: bool,
    ) -> bool:
        """Strength-0 adrenaline: take +1 unless it pushes an unaffordable corner.

        Looks only at the already-crossed corners and the expected speed (no
        worst-case guard) -- the myopic floor's version of :meth:`_adrenaline_safe`.
        """
        speed_now = card_speed + (1 if boosting else 0)
        cost_with = sum(rules.corner_heat_cost(speed_now + 1, c) for c in crossed)
        cost_without = sum(rules.corner_heat_cost(speed_now, c) for c in crossed)
        extra = cost_with - cost_without
        if extra > 0 and player.heat_available <= extra:
            return False
        return True

    def choose_slipstream(
        self,
        state: GameState,
        player_id: int,
    ) -> bool:
        player = state.get_player(player_id)
        track = state.track
        heat_price = self._effective_heat_price(state, player_id)

        # +2 slipstream move; movement is EXCLUDED from the corner-speed check,
        # so the check uses the (lower) card speed -- a genuinely good deal.
        # Lap-aware landing (P1) so the relative value sees the correct lap on a
        # finish-crossing slipstream.
        end_lap, new_pos = ME.landed_lap_and_pos(track, player.position, player.lap, 2)
        crossed = rules.corners_crossed(player.position, new_pos, track)
        card_speed = rules.corner_speed_for_check(player)
        cost = sum(rules.corner_heat_cost(card_speed, c) for c in crossed)

        if cost > player.heat_available:
            return False  # cannot pay -> would spin; decline

        # Spaces gained net of any block at the landing space.
        effect = ME.resolve_landing(state, player_id, new_pos)
        spaces = 2 - effect.spaces_lost_to_block

        value = spaces - heat_price * cost
        if self._opponent_aware:
            # A slipstream that pulls level with / ahead of the car directly
            # ahead is the canonical contested-gap win -- value it with the same
            # saturating relative term used by the planner.
            prog_before = ME.race_progress(player, track)
            prog_after = end_lap * track.length + effect.landed_pos
            value += ME.RELATIVE_WEIGHT * ME.relative_term(
                state, player_id, prog_before, prog_after
            )
        return value > 0.0

    def choose_discard(
        self,
        state: GameState,
        player_id: int,
        discardable: list[Card],
    ) -> list[Card]:
        if not discardable:
            return []
        player = state.get_player(player_id)

        # Never discard upgrades (they are high value / flip fodder).
        speed_discardable = [
            c for c in discardable if c.card_type == CardType.SPEED
        ]
        if not speed_discardable:
            return []

        playable = [
            c for c in player.hand
            if c.card_type in (CardType.SPEED, CardType.UPGRADE)
        ]
        # Never thin the hand into a cluttered state for the upcoming gear.
        # Keep at least gear + 2 playable cards as a buffer.
        keep_floor = player.gear + 2
        room_to_cycle = len(playable) - keep_floor
        if room_to_cycle <= 0:
            return []

        deck_mean = ME.expected_basic_value(player)
        # Discard the lowest-value speed cards that are below the deck mean:
        # removing them raises the expected value of what we redraw. Cap the
        # number discarded by the room available so we never over-thin.
        candidates = sorted(speed_discardable, key=lambda c: c.value)
        to_discard: list[Card] = []
        for card in candidates:
            if len(to_discard) >= room_to_cycle:
                break
            if card.value < deck_mean - 0.5:
                to_discard.append(card)
        return to_discard
