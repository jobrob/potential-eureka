# Track Visualisation Plan

## Goal

Add an ASCII track visualisation to the game viewer that shows the track layout as a rectangular circuit with player positions, corners, and speed limits. Displayed at the start of the game and after each round.

## Design: Rectangular Circuit

Render the linear space sequence as a closed rectangular loop, viewed from above:

```
    Silverstone (Lap 1/2)                          ■ Max  ● Lewis

    ──0──1──2──3──4──5──6──7──[C1:3 8·9·10]─[C2:2 11·12]──13──
    │                                                          │
   49                                                         14
   48                                                         15
   47                                                         16
    │         ■ Max                           ● Lewis          │
    ──43──42──41──40──39──38──37──36──35──34──33──32──31──30──
                                      [C4:3 29·28]─[C3:4 26·25]
```

Key features:
- Track is a rectangle with spaces distributed around the perimeter
- Top edge: spaces go left-to-right
- Right edge: spaces go top-to-bottom
- Bottom edge: spaces go right-to-left
- Left edge: spaces go bottom-to-top
- Corners shown inline with `[C1:3 8·9·10]` notation (name:speed_limit spaces)
- Player positions shown with single-character symbols (first letter of name, or ■●▲◆)
- Legend at top maps symbols to player names

## Implementation Steps

### Step 1: Space-to-grid mapping

Given N spaces, divide them around a rectangle:

```python
def _distribute_spaces(total: int) -> tuple[int, int, int, int]:
    """Divide spaces into (top, right, bottom, left) counts."""
    # Aim for roughly 3:2 width:height ratio
    perimeter = total
    width_fraction = 0.35  # top and bottom each get 35%
    height_fraction = 0.15  # left and right each get 15%
    
    top = round(perimeter * width_fraction)
    bottom = round(perimeter * width_fraction)
    right = round(perimeter * height_fraction)
    left = perimeter - top - bottom - right
    
    return top, right, bottom, left
```

Then assign each space index an (x, y) grid coordinate by walking around the rectangle.

### Step 2: Corner labelling

For each corner in `track.corners`, find the spaces it covers and render them as a group:

- Horizontal corners (on top/bottom edge): `[C1:3 8·9·10]`
- Vertical corners (on left/right edge): rendered as a bracketed column

Normal spaces on horizontal edges: `──5──6──7──`
Normal spaces on vertical edges: one per row with the space number

### Step 3: Rendering

Build a 2D character grid (list of lists), then:
1. Draw the rectangle edges (─ for horizontal, │ for vertical)
2. Place space numbers along the edges
3. Overlay corner brackets and labels
4. Place player markers
5. Add legend and track info header
6. Convert grid to string

### Step 4: Integration with viewer

Add a `render_track(state: GameState) -> str` function that produces the visualisation.

Call it from `watch_game()`:
- Once at the start of the game (empty track, just layout)
- After each round (showing current player positions)

Also make it available standalone: `from heat.viewer import render_track`

## File changes

- `src/heat/viewer.py`: Add `render_track()` function and call it from `watch_game()`
- No model changes needed — all data is already in Track and PlayerState

## Edge cases to handle

- **Very short tracks** (10 spaces): minimum rectangle size so it doesn't collapse
- **Very long tracks** (80+ spaces): may need wider output, consider terminal width ~100 chars
- **Corners at rectangle bends**: if a corner spans where the top meets the right edge, split the rendering across both edges
- **Multiple players on same space**: show markers side by side, use lane count info
- **Lapped players**: show lap number next to marker if players are on different laps

## Estimated effort

1-1.5 hours for core implementation. The main complexity is the grid coordinate mapping and making corners render cleanly on both horizontal and vertical edges.

## Example outputs

### Small track (USA, 30 spaces)

```
    USA (Lap 1/1)                              ■ Max  ● Lewis

    ──0──1──2──3──4──5──[C1:4 6·7·8]──9──10──
    │                                         │
   29                                        11
   28                                        12
   27                                        13
    │              ●                  ■       │
    ──[C3:5 26·25·24]──23──22──21──20──19──18──
                          [C2:3 17·16·15]
```

### Large track (Silverstone, 50 spaces)

```
    Silverstone (Lap 1/2)                      ■ Max  ● Lewis

    ──0──1──2──3──4──5──6──7──[C1:3 8·9·10]─[C2:2 11·12]──13──
    │                                                          │
   49                                                         14
   48                                                         15
   47                                                         16
    │         ■                                ●               │
    ──43──42──41──40──39──38──37──36──35──34──33──32──31──30──
                                      [C4:3 29·28]─[C3:4 26·25]
```
