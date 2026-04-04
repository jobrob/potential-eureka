"""Track loader: reads track definitions from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from heat.models.track import Corner, Space, Track

# Default directory for track JSON files
_TRACKS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tracks"


def load_track(path: str | Path) -> Track:
    """Load a track from a JSON file.

    Args:
        path: Path to the JSON track file.

    Returns:
        A Track instance.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    spaces = [Space(index=s["index"], lanes=s.get("lanes", 1)) for s in data["spaces"]]
    corners = [
        Corner(start=c["start"], end=c["end"], speed_limit=c["speed_limit"])
        for c in data["corners"]
    ]
    start_positions = data["start_positions"]

    return Track(
        name=data["name"],
        spaces=spaces,
        corners=corners,
        start_positions=start_positions,
        laps=data.get("laps", 1),
    )


def load_track_by_name(name: str) -> Track:
    """Load a track by name from the default tracks directory.

    Args:
        name: Track name (e.g., "usa"). Looks for tracks/{name}.json.

    Returns:
        A Track instance.

    Raises:
        FileNotFoundError: If the track file doesn't exist.
    """
    path = _TRACKS_DIR / f"{name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")
    return load_track(path)
