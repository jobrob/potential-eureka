"""Human-readable game viewer for HEAT races.

Usage:
    from heat.viewer import watch_game, format_event_log, save_event_log

    # Live mode — prints each event as the game runs
    result = watch_game(track, agents, player_names=["Alice", "Bob"])

    # Live mode with file logging
    result = watch_game(track, agents, player_names=["Alice", "Bob"], log_file="race.log")

    # Replay mode — format a completed game's event log
    print(format_event_log(result.event_log, game.state))

    # Save formatted event log to a file
    save_event_log(result.event_log, game.state, "replay.log")
"""

from __future__ import annotations

from heat.models.game_state import GameEvent, GameState
from heat.models.track import Track
from heat.engine.game import Agent, Game, GameResult


def format_event(event: GameEvent, state: GameState) -> str | None:
    """Format a single GameEvent into a human-readable line.

    Returns None for events that don't need display.
    """
    pid = event.player_id
    d = event.data
    name = state.get_player(pid).name if pid is not None else "?"

    match event.event_type:
        case "turn_start":
            hand = d.get("hand", [])
            hand_size = d.get("hand_size", len(hand))
            gear = d.get("gear", "?")
            heat = d.get("heat_available", "?")
            pos = d.get("position", "?")
            corner_dist = d.get("next_corner_dist")
            corner_limit = d.get("next_corner_speed_limit")
            hand_str = ", ".join(hand)
            corner_info = ""
            if corner_dist is not None and corner_limit is not None:
                corner_info = f" | next corner in {corner_dist} spaces (limit {corner_limit})"
            return (
                f"  {name} [pos {pos}, gear {gear}, heat {heat}]: "
                f"hand({hand_size}) [{hand_str}]{corner_info}"
            )

        case "gear_shift":
            old = d.get("old_gear", "?")
            new = d["new_gear"]
            cost = d["heat_cost"]
            if d.get("spun_out"):
                return f"  {name}: forced to gear {new} (spun out)"
            shift = ""
            if cost > 0:
                shift = f" (paid {cost} heat)"
            return f"  {name}: gear {old} -> {new}{shift}"

        case "play_cards":
            cards = d["cards"]
            cluttered = d.get("cluttered", False)
            card_str = ", ".join(cards)
            suffix = " [CLUTTERED HAND]" if cluttered else ""
            return f"  {name}: played [{card_str}]{suffix}"

        case "stress_resolved":
            val = d["value"]
            flipped = d["flipped_count"]
            discarded = d.get("discarded", [])
            if discarded:
                kept_val = val
                disc_str = ", ".join(discarded)
                return (
                    f"  {name}: stress resolved -> speed {val} "
                    f"(flipped {flipped}: discarded [{disc_str}], kept Speed({kept_val}))"
                )
            return f"  {name}: stress resolved -> speed {val} (flipped {flipped} cards)"

        case "reveal_and_move":
            spd = d["speed"]
            pos = d["new_position"]
            lap = d["lap"]
            fin = d["finished"]
            fin_str = " *** FINISHED! ***" if fin else ""
            return f"  {name}: speed {spd}, moved to space {pos} (lap {lap}){fin_str}"

        case "cooldown":
            count = d["count"]
            return f"  {name}: cooled {count} heat card(s)"

        case "boost":
            val = d["value"]
            flipped = d["flipped_count"]
            return f"  {name}: boost! speed +{val} (flipped {flipped} cards, paid 1 heat)"

        case "adrenaline_granted":
            return f"  {name}: adrenaline available"

        case "adrenaline_speed":
            return f"  {name}: used adrenaline speed +1"

        case "slipstream":
            pos = d["new_position"]
            return f"  {name}: slipstreamed +2 -> space {pos}"

        case "corner_check":
            corners = d["corners"]
            spd = d["speed"]
            cost = d["heat_cost"]
            if cost == 0:
                return None  # Don't clutter output with "no cost" messages
            return f"  {name}: corner check (speed {spd}, {corners} corner(s)) -> paid {cost} heat"

        case "spin_out":
            pos = d["new_position"]
            stress = d["stress_added"]
            heat_paid = d["heat_paid"]
            return f"  {name}: SPIN OUT! paid {heat_paid} heat, +{stress} stress, back to space {pos}, gear -> 1"

        case "discard":
            count = d["count"]
            return f"  {name}: discarded {count} card(s)"

        case "replenish":
            drawn = d.get("drawn", [])
            if drawn:
                drawn_str = ", ".join(drawn)
                return f"  {name}: drew {len(drawn)} cards [{drawn_str}]"
            return None  # Nothing drawn, skip

        case _:
            return f"  {name}: {event.event_type} {d}"


def format_event_log(event_log: list[GameEvent], state: GameState) -> str:
    """Format an entire event log into a readable string."""
    lines: list[str] = []
    current_round = -1

    for event in event_log:
        if event.round_num != current_round:
            current_round = event.round_num
            if lines:
                lines.append("")
            lines.append(f"=== Round {current_round} ===")

        line = format_event(event, state)
        if line is not None:
            lines.append(line)

    return "\n".join(lines)


def save_event_log(event_log: list[GameEvent], state: GameState, filepath: str) -> None:
    """Save formatted event log to a file."""
    text = format_event_log(event_log, state)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def _format_standings(state: GameState) -> list[str]:
    """Format current race standings as a list of lines."""
    active = sorted(
        state.active_players,
        key=lambda p: (p.lap, p.position),
        reverse=True,
    )
    finished = state.finished_players
    track = state.track

    parts: list[str] = []
    pos = 1
    for p in finished:
        parts.append(f"  {pos}. {p.name} (FINISHED)")
        pos += 1
    for p in active:
        # Compute distance to next corner
        corner_info = ""
        best_dist = None
        for corner in track.corners:
            dist = (corner.start - p.position) % track.length
            if dist == 0:
                dist = track.length
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is not None:
            corner_info = f", corner in {best_dist}"
        parts.append(
            f"  {pos}. {p.name} (space {p.position}, lap {p.lap}, "
            f"gear {p.gear}, heat {p.heat_available}{corner_info})"
        )
        pos += 1

    return ["  --- Standings ---"] + parts


def watch_game(
    track: Track,
    agents: list[Agent],
    player_names: list[str] | None = None,
    show_standings: bool = True,
    log_file: str | None = None,
) -> GameResult:
    """Run a game and print a live play-by-play to stdout.

    When log_file is provided, all output is also written to that file.

    Returns the GameResult when the game finishes.
    """
    _out = None
    if log_file:
        _out = open(log_file, "w", encoding="utf-8")

    def _print(msg: str = "") -> None:
        print(msg)
        if _out:
            _out.write(msg + "\n")

    try:
        game = Game(track, agents, player_names=player_names, logging_enabled=True)
        state = game.state

        # Print header
        _print(f"{'='*60}")
        _print(f"HEAT Race: {track.name}")
        _print(f"Players: {', '.join(p.name for p in state.players)}")
        _print(f"Track: {track.length} spaces, {len(track.corners)} corners, {track.laps} lap(s)")
        _print(f"{'='*60}")
        _print()

        round_num = 0
        while not game.is_over:
            round_num += 1
            log_start = len(state.event_log)

            game.run_round()

            # Print events from this round
            _print(f"=== Round {round_num} ===")
            for event in state.event_log[log_start:]:
                line = format_event(event, state)
                if line is not None:
                    _print(line)

            if show_standings:
                for line in _format_standings(state):
                    _print(line)
            _print()

            if state.round_num > 200:
                _print("MAX ROUNDS reached -- forcing finish.")
                break

        # Print final results
        result = GameResult(
            finish_order=[p.player_id for p in state.finished_players],
            total_rounds=round_num,
            event_log=state.event_log,
        )

        _print(f"{'='*60}")
        _print(f"RACE COMPLETE in {round_num} rounds!")
        _print()
        for i, pid in enumerate(result.finish_order):
            p = state.get_player(pid)
            _print(f"  {i+1}. {p.name}")
        _print(f"{'='*60}")

        return result
    finally:
        if _out:
            _out.close()
