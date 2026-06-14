"""Game event utilities.

GameEvent is defined in models.game_state and re-exported from models.
This module is reserved for event filtering, replay serialization,
and other engine-level event processing added in later sprints.
"""

from heat.models.game_state import GameEvent

__all__ = ["GameEvent"]
