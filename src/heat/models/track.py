"""Track, Space, and Corner data models for the HEAT board game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Corner:
    """A corner on the track with a speed limit.

    Attributes:
        start: Index of the first space in the corner (inclusive).
        end: Index of the last space in the corner (inclusive).
        speed_limit: Maximum speed (total card value) to pass without paying heat.
    """

    start: int
    end: int
    speed_limit: int


@dataclass(frozen=True)
class Space:
    """A single space on the track.

    Attributes:
        index: Position index along the track (0-based).
        lanes: Number of lanes at this space (how many cars can share it side-by-side).
    """

    index: int
    lanes: int = 1


@dataclass
class Track:
    """A complete race track.

    Attributes:
        name: Display name of the track.
        spaces: Ordered list of spaces making up the track.
        corners: List of corners with speed limits.
        start_positions: Ordered starting grid positions (space indices).
        laps: Number of laps to complete.
    """

    name: str
    spaces: list[Space]
    corners: list[Corner]
    start_positions: list[int]
    laps: int = 1

    @property
    def length(self) -> int:
        """Total number of spaces on the track."""
        return len(self.spaces)

    def get_corner_at(self, position: int) -> Corner | None:
        """Return the corner that covers the given position, or None."""
        for corner in self.corners:
            if corner.start <= position <= corner.end:
                return corner
        return None

    def spaces_in_corner(self, corner: Corner) -> list[Space]:
        """Return all spaces within a given corner."""
        return self.spaces[corner.start : corner.end + 1]
