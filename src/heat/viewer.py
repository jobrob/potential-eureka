"""Human-readable game viewer for HEAT races.

Usage:
    from heat.viewer import watch_game, format_event_log, save_event_log

    # Live mode — prints each event as the game runs
    result = watch_game(track, agents, player_names=["Alice", "Bob"])

    # Render the track layout (optionally with current player positions)
    from heat.viewer import render_track
    print(render_track(game.state))

    # Live mode with file logging
    result = watch_game(track, agents, player_names=["Alice", "Bob"], log_file="race.log")

    # Replay mode — format a completed game's event log
    print(format_event_log(result.event_log, game.state))

    # Save formatted event log to a file
    save_event_log(result.event_log, game.state, "replay.log")
"""

from __future__ import annotations

from heat.models.game_state import GameEvent, GameState
from heat.models.player_state import PlayerState
from heat.models.track import Track
from heat.engine import rules
from heat.engine.game import MAX_ROUNDS, Agent, Game, GameResult


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
            if cost > 0:
                heat = d.get("heat_available", "?")
                return f"  {name}: gear {old} -> {new} (paid {cost} heat, heat: {heat})"
            return f"  {name}: gear {old} -> {new}"

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
                    f"(flipped {flipped}: discarded [{disc_str}], kept {kept_val})"
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
            heat = d.get("heat_available", "?")
            return f"  {name}: cooled {count} heat card(s) (heat: {heat})"

        case "boost":
            val = d["value"]
            flipped = d["flipped_count"]
            heat = d.get("heat_available", "?")
            return f"  {name}: boost! +{val} speed (flipped {flipped} cards, paid 1 heat, heat: {heat})"

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
            heat = d.get("heat_available", "?")
            return f"  {name}: corner check (speed {spd}, {corners} corner(s)) -> paid {cost} heat (heat: {heat})"

        case "spin_out":
            pos = d["new_position"]
            stress = d["stress_added"]
            heat_paid = d["heat_paid"]
            heat = d.get("heat_available", "?")
            return f"  {name}: SPIN OUT! paid {heat_paid} heat, +{stress} stress, back to space {pos}, gear -> 1 (heat: {heat})"

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
        next_corner, best_dist = rules.distance_to_next_corner(
            track, p.position,
        )
        if next_corner is not None:
            corner_info = f", corner in {best_dist}"
        parts.append(
            f"  {pos}. {p.name} (space {p.position}, lap {p.lap}, "
            f"gear {p.gear}, heat {p.heat_available}{corner_info})"
        )
        pos += 1

    return ["  --- Standings ---"] + parts


# ---------------------------------------------------------------------------
# Track visualisation
# ---------------------------------------------------------------------------

# Player marker symbols, assigned in player_id order. Falls back to the first
# letter of the name (then a digit) once this pool is exhausted.
_MARKER_SYMBOLS = ["#", "@", "*", "+", "%", "$", "&", "=", "~", "?"]


def _distribute_spaces(total: int) -> tuple[int, int, int, int]:
    """Divide ``total`` spaces around a rectangle as (top, right, bottom, left).

    Aims for a roughly 3:2 width:height look while guaranteeing that every
    edge has at least one space (so the rectangle never collapses) and that the
    four counts always sum back to ``total``.
    """
    if total <= 0:
        return 0, 0, 0, 0
    if total < 4:
        # Too small to form a real rectangle: put everything on the top edge.
        return total, 0, 0, 0

    # Horizontal edges (top/bottom) get the bulk of the spaces.
    top = max(1, round(total * 0.35))
    bottom = max(1, round(total * 0.35))
    # Make sure the verticals get at least one space each.
    while top + bottom > total - 2:
        if top >= bottom and top > 1:
            top -= 1
        elif bottom > 1:
            bottom -= 1
        else:
            break

    remaining = total - top - bottom
    right = remaining // 2
    left = remaining - right
    return top, right, bottom, left


def _assign_markers(players: list[PlayerState]) -> dict[int, str]:
    """Assign a single-character marker symbol to each player by player_id."""
    markers: dict[int, str] = {}
    used: set[str] = set()
    for i, p in enumerate(sorted(players, key=lambda pl: pl.player_id)):
        if i < len(_MARKER_SYMBOLS):
            sym = _MARKER_SYMBOLS[i]
        else:
            first = p.name.strip()[:1].upper() if p.name.strip() else "?"
            sym = first if first not in used else str(p.player_id % 10)
        markers[p.player_id] = sym
        used.add(sym)
    return markers


def _corner_index_map(track: Track) -> dict[int, int]:
    """Map each space index covered by a corner to that corner's 1-based label."""
    mapping: dict[int, int] = {}
    for ci, corner in enumerate(track.corners, start=1):
        for idx in range(corner.start, corner.end + 1):
            mapping[idx] = ci
    return mapping


def _players_by_space(players: list[PlayerState]) -> dict[int, list[PlayerState]]:
    """Group players by their current position index."""
    by_space: dict[int, list[PlayerState]] = {}
    for p in players:
        by_space.setdefault(p.position, []).append(p)
    return by_space


def _build_edges(track: Track) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return the space indices on each edge: (top, right, bottom, left).

    Spaces are walked clockwise starting from index 0:
    top is left-to-right, right is top-to-bottom, bottom is right-to-left
    (so the indices are descending in render order), left is bottom-to-top.
    """
    n = track.length
    top_n, right_n, bottom_n, left_n = _distribute_spaces(n)

    order = list(range(n))
    top = order[:top_n]
    right = order[top_n : top_n + right_n]
    bottom_raw = order[top_n + right_n : top_n + right_n + bottom_n]
    left_raw = order[top_n + right_n + bottom_n :]

    # Bottom is rendered right-to-left; left is rendered bottom-to-top.
    bottom = list(reversed(bottom_raw))
    left = list(reversed(left_raw))
    return top, right, bottom, left


def _format_horizontal_edge(
    indices: list[int],
    corner_map: dict[int, int],
    track: Track,
) -> str:
    """Render a horizontal edge (top or bottom) as a single ``──n──`` line.

    Consecutive spaces belonging to the same corner are grouped into a single
    ``[C1:limit a·b·c]`` token.
    """
    if not indices:
        return "──"

    tokens: list[str] = []
    i = 0
    while i < len(indices):
        idx = indices[i]
        cid = corner_map.get(idx)
        if cid is None:
            tokens.append(str(idx))
            i += 1
            continue

        # Gather the run of consecutive indices that share this corner id.
        run = [idx]
        j = i + 1
        while j < len(indices) and corner_map.get(indices[j]) == cid:
            run.append(indices[j])
            j += 1
        corner = track.corners[cid - 1]
        nums = "·".join(str(x) for x in run)
        tokens.append(f"[C{cid}:{corner.speed_limit} {nums}]")
        i = j

    return "──" + "──".join(tokens) + "──"


def _marker_overlay(
    edge_line: str,
    indices: list[int],
    by_space: dict[int, list[PlayerState]],
    markers: dict[int, str],
    show_lap: bool,
) -> str | None:
    """Build a marker row aligned under/over a horizontal edge line.

    Returns None when no player sits on this edge.
    """
    if not any(idx in by_space for idx in indices):
        return None

    row = [" "] * len(edge_line)
    # Recompute token start columns by re-walking the same structure used in
    # _format_horizontal_edge: a leading "──" then tokens joined by "──".
    # We instead locate each space number by searching for its rendered form.
    cursor = 0
    for idx in indices:
        players_here = by_space.get(idx)
        if not players_here:
            continue
        token = str(idx)
        col = edge_line.find(token, cursor)
        if col == -1:
            col = edge_line.find(token)
        if col == -1:
            continue
        cursor = col + len(token)
        label = "".join(markers[p.player_id] for p in players_here)
        if show_lap:
            laps = {p.lap for p in players_here}
            if len(laps) == 1:
                label += f"L{laps.pop()}"
        for k, ch in enumerate(label):
            pos = col + k
            if pos < len(row):
                row[pos] = ch

    text = "".join(row).rstrip()
    return text if text.strip() else None


def render_track(state: GameState) -> str:
    """Render the track as an ASCII rectangular circuit.

    Shows spaces around the perimeter, corners inline with ``[C1:3 8·9·10]``
    notation (label:speed_limit space·numbers), player position markers, a
    legend mapping markers to names, and a header with the track name and lap
    info. Deterministic and safe for 0 players, shared spaces, lapped players,
    and tracks from ~8 to 80+ spaces.
    """
    track = state.track
    players = list(state.players)
    markers = _assign_markers(players)
    corner_map = _corner_index_map(track)
    by_space = _players_by_space(players)

    # Are any players on different laps? If so, annotate markers with lap nums.
    laps_present = {p.lap for p in players}
    show_lap = len(laps_present) > 1

    top, right, bottom, left = _build_edges(track)

    top_line = _format_horizontal_edge(top, corner_map, track)
    bottom_line = _format_horizontal_edge(bottom, corner_map, track)

    width = max(len(top_line), len(bottom_line), 4)

    lines: list[str] = []

    # --- Header ----------------------------------------------------------
    lap_info = f"Lap 1/{track.laps}"
    if players:
        leading_lap = max(laps_present)
        lap_info = f"Lap {leading_lap}/{track.laps}"
    header = f"{track.name} ({lap_info})"
    lines.append(f"    {header}")

    # --- Legend ----------------------------------------------------------
    if players:
        legend = "  ".join(
            f"{markers[p.player_id]} {p.name}"
            for p in sorted(players, key=lambda pl: pl.player_id)
        )
        lines.append(f"    Legend: {legend}")
    lines.append("")

    indent = "    "

    # --- Top marker row + top edge --------------------------------------
    top_markers = _marker_overlay(top_line, top, by_space, markers, show_lap)
    if top_markers is not None:
        lines.append(indent + top_markers)
    lines.append(indent + top_line)

    # --- Vertical edges --------------------------------------------------
    # right is top-to-bottom, left is bottom-to-top. We render rows pairing the
    # left edge (descending visually) with the right edge.
    rows = max(len(left), len(right))
    # Column where the right edge sits (right border of the rectangle).
    right_col = width - 1

    def _vertical_cell(idx: int | None) -> tuple[str, str]:
        """Return (number_text, marker_text) for a vertical-edge space."""
        if idx is None:
            return "", ""
        num = str(idx)
        ps = by_space.get(idx)
        if not ps:
            return num, ""
        marker = "".join(markers[p.player_id] for p in ps)
        if show_lap:
            laps = {p.lap for p in ps}
            if len(laps) == 1:
                marker += f"L{laps.pop()}"
        return num, marker

    # left is rendered top-to-bottom in visual order: the topmost left-edge
    # space is the one adjacent to the top edge. _build_edges gave us left as
    # bottom-to-top, so reverse for top-to-bottom display.
    left_disp = list(reversed(left))
    for r in range(rows):
        l_idx = left_disp[r] if r < len(left_disp) else None
        r_idx = right[r] if r < len(right) else None

        l_num, l_mark = _vertical_cell(l_idx)
        r_num, r_mark = _vertical_cell(r_idx)

        # Left border: number (or │) at the far left.
        left_label = l_num if l_num else "│"
        # Compose a blank row of the full width.
        row_chars = [" "] * (right_col + 1)
        # Place left label at column 0.
        for k, ch in enumerate(left_label):
            if k < len(row_chars):
                row_chars[k] = ch
        # Left marker just to the right of the left label.
        if l_mark:
            start = len(left_label) + 1
            for k, ch in enumerate(l_mark):
                if start + k < len(row_chars):
                    row_chars[start + k] = ch
        # Right border: number (or │) at the right column.
        right_label = r_num if r_num else "│"
        start = right_col - len(right_label) + 1
        for k, ch in enumerate(right_label):
            if 0 <= start + k < len(row_chars):
                row_chars[start + k] = ch
        # Right marker just to the left of the right label.
        if r_mark:
            mstart = start - len(r_mark) - 1
            for k, ch in enumerate(r_mark):
                if 0 <= mstart + k < len(row_chars):
                    row_chars[mstart + k] = ch

        lines.append(indent + "".join(row_chars).rstrip())

    # --- Bottom edge + bottom marker row --------------------------------
    lines.append(indent + bottom_line)
    bottom_markers = _marker_overlay(bottom_line, bottom, by_space, markers, show_lap)
    if bottom_markers is not None:
        lines.append(indent + bottom_markers)

    return "\n".join(lines)


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

        # Show the initial track layout with starting positions.
        _print(render_track(state))
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

            # Show the track with current player positions after the round.
            _print(render_track(state))
            _print()

            if state.round_num > MAX_ROUNDS:
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
