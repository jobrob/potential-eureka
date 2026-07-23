"""Fixed-shape tensor state for Direction D2's exact engine slices."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import torch

from heat.engine.game import MAX_ROUNDS
from heat.ml.spaces import MAX_CORNERS, MAX_PLAYERS
from heat.models.game_state import Phase


# These capacities cover the default generated-track contract. Wider custom
# tracks must fail at the bridge instead of being silently truncated.
MAX_TRACK_SPACES = 90
MAX_TRACK_STARTS = MAX_PLAYERS
MAX_CARDS_PER_ZONE = 24
MAX_SPIN_RECORDS = MAX_ROUNDS
PYTHON_RNG_WORDS = 625
PHASE_ORDER = tuple(Phase)
PHASE_TO_CODE = {phase: index for index, phase in enumerate(PHASE_ORDER)}

CARD_TYPE_PADDING = 0
CARD_TYPE_SPEED = 1
CARD_TYPE_HEAT = 2
CARD_TYPE_STRESS = 3
CARD_TYPE_UPGRADE = 4


class TensorStateCapacityError(ValueError):
    """Raised when legacy state cannot fit D2's frozen tensor capacities."""


@dataclass(frozen=True)
class TensorCardZone:
    """One padded card zone for every player in every game lane."""

    card_ids: torch.Tensor
    card_types: torch.Tensor
    card_values: torch.Tensor
    lengths: torch.Tensor

    def validate(self, batch_size: int) -> None:
        """Check shapes, dtypes, and recorded lengths for this card zone."""
        _expect_tensor(
            "card_ids",
            self.card_ids,
            (batch_size, MAX_PLAYERS, MAX_CARDS_PER_ZONE),
            torch.int64,
        )
        _expect_tensor(
            "card_types",
            self.card_types,
            (batch_size, MAX_PLAYERS, MAX_CARDS_PER_ZONE),
            torch.int64,
        )
        _expect_tensor(
            "card_values",
            self.card_values,
            (batch_size, MAX_PLAYERS, MAX_CARDS_PER_ZONE),
            torch.int64,
        )
        _expect_tensor(
            "card_lengths", self.lengths, (batch_size, MAX_PLAYERS), torch.int64
        )
        if bool(torch.any(self.lengths < 0)) or bool(
            torch.any(self.lengths > MAX_CARDS_PER_ZONE)
        ):
            raise ValueError("card zone lengths exceed fixed capacity")


@dataclass(frozen=True)
class TensorGameState:
    """A batch of exact legacy states represented by fixed-shape tensors.

    Human-readable names and the card-ID vocabulary are immutable metadata;
    every mutable rule field and every reference into that metadata is a tensor.
    This keeps future kernels tensor-only without discarding the identities that
    the D0 semantic oracle uses for exact comparisons.
    """

    game_ids: torch.Tensor
    game_active: torch.Tensor
    player_present: torch.Tensor
    player_active: torch.Tensor

    round_num: torch.Tensor
    current_phase: torch.Tensor
    turn_order: torch.Tensor
    turn_order_lengths: torch.Tensor
    starting_player_count: torch.Tensor
    stress_counter: torch.Tensor

    rng_version: torch.Tensor
    rng_words: torch.Tensor
    rng_gauss_present: torch.Tensor
    rng_gauss: torch.Tensor

    track_lengths: torch.Tensor
    track_space_indices: torch.Tensor
    track_lanes: torch.Tensor
    track_corner_counts: torch.Tensor
    track_corners: torch.Tensor
    track_start_counts: torch.Tensor
    track_start_positions: torch.Tensor
    track_laps: torch.Tensor

    player_ids: torch.Tensor
    gear: torch.Tensor
    position: torch.Tensor
    lap: torch.Tensor
    spun_out: torch.Tensor
    finished: torch.Tensor
    finish_order: torch.Tensor
    spin_log: torch.Tensor
    spin_log_lengths: torch.Tensor
    boost_used_this_turn: torch.Tensor
    speed_from_cards: torch.Tensor
    speed_from_boost: torch.Tensor
    speed_from_adrenaline: torch.Tensor
    slipstream_moved: torch.Tensor
    cluttered: torch.Tensor
    turn_start_position: torch.Tensor
    turn_start_lap: torch.Tensor

    hand: TensorCardZone
    draw_pile: TensorCardZone
    discard_pile: TensorCardZone
    heat_pool: TensorCardZone
    cooldown_pool: TensorCardZone
    cards_played: TensorCardZone

    track_names: tuple[str, ...]
    player_names: tuple[tuple[str, ...], ...]
    card_id_vocabulary: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of game lanes in this batch."""
        return int(self.game_ids.shape[0])

    def validate(self) -> None:
        """Fail clearly if a tensor has the wrong shape, type, or capacity."""
        batch = self.batch_size
        if batch < 1:
            raise ValueError("tensor state requires at least one game")

        vector_names = (
            "game_ids",
            "game_active",
            "round_num",
            "current_phase",
            "turn_order_lengths",
            "starting_player_count",
            "stress_counter",
            "rng_version",
            "rng_gauss_present",
            "rng_gauss",
            "track_lengths",
            "track_corner_counts",
            "track_start_counts",
            "track_laps",
        )
        bool_vectors = {"game_active", "rng_gauss_present"}
        float_vectors = {"rng_gauss"}
        for name in vector_names:
            tensor = getattr(self, name)
            dtype = (
                torch.bool
                if name in bool_vectors
                else torch.float64
                if name in float_vectors
                else torch.int64
            )
            _expect_tensor(name, tensor, (batch,), dtype)

        player_shape = (batch, MAX_PLAYERS)
        player_names = (
            "player_present",
            "player_active",
            "player_ids",
            "gear",
            "position",
            "lap",
            "spun_out",
            "finished",
            "finish_order",
            "spin_log_lengths",
            "boost_used_this_turn",
            "speed_from_cards",
            "speed_from_boost",
            "speed_from_adrenaline",
            "slipstream_moved",
            "cluttered",
            "turn_start_position",
            "turn_start_lap",
        )
        bool_players = {
            "player_present",
            "player_active",
            "spun_out",
            "finished",
            "boost_used_this_turn",
            "cluttered",
        }
        for name in player_names:
            _expect_tensor(
                name,
                getattr(self, name),
                player_shape,
                torch.bool if name in bool_players else torch.int64,
            )

        _expect_tensor(
            "turn_order", self.turn_order, player_shape, torch.int64
        )
        _expect_tensor(
            "rng_words", self.rng_words, (batch, PYTHON_RNG_WORDS), torch.int64
        )
        _expect_tensor(
            "track_space_indices",
            self.track_space_indices,
            (batch, MAX_TRACK_SPACES),
            torch.int64,
        )
        _expect_tensor(
            "track_lanes",
            self.track_lanes,
            (batch, MAX_TRACK_SPACES),
            torch.int64,
        )
        _expect_tensor(
            "track_corners",
            self.track_corners,
            (batch, MAX_CORNERS, 3),
            torch.int64,
        )
        _expect_tensor(
            "track_start_positions",
            self.track_start_positions,
            (batch, MAX_TRACK_STARTS),
            torch.int64,
        )
        _expect_tensor(
            "spin_log",
            self.spin_log,
            (batch, MAX_PLAYERS, MAX_SPIN_RECORDS, 2),
            torch.int64,
        )

        for zone_name in (
            "hand",
            "draw_pile",
            "discard_pile",
            "heat_pool",
            "cooldown_pool",
            "cards_played",
        ):
            getattr(self, zone_name).validate(batch)

        if len(self.track_names) != batch or len(self.player_names) != batch:
            raise ValueError("metadata lane count does not match tensor batch")
        if any(len(names) != MAX_PLAYERS for names in self.player_names):
            raise ValueError("player name metadata must use the fixed player shape")
        if len(set(int(item) for item in self.game_ids.tolist())) != batch:
            raise ValueError("game_ids must be unique stable identities")
        _bounded_lengths("turn order", self.turn_order_lengths, MAX_PLAYERS)
        _bounded_lengths("track spaces", self.track_lengths, MAX_TRACK_SPACES)
        _bounded_lengths("track corners", self.track_corner_counts, MAX_CORNERS)
        _bounded_lengths("track starts", self.track_start_counts, MAX_TRACK_STARTS)
        _bounded_lengths("spin log", self.spin_log_lengths, MAX_SPIN_RECORDS)

        player_counts = self.player_present.sum(dim=1)
        if bool(torch.any(player_counts < 2)) or bool(
            torch.any(player_counts > MAX_PLAYERS)
        ):
            raise ValueError("Direction D2 tensor lanes require 2-6 players")
        if not torch.equal(self.player_active, self.player_present & ~self.finished):
            raise ValueError("player_active must equal present and not finished")
        if not torch.equal(self.game_active, torch.any(self.player_active, dim=1)):
            raise ValueError("game_active must reflect active player lanes")

    def tensor_fields(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return named tensors recursively for exact determinism assertions."""
        result: list[tuple[str, torch.Tensor]] = []
        for state_field in fields(self):
            value = getattr(self, state_field.name)
            if isinstance(value, torch.Tensor):
                result.append((state_field.name, value))
            elif isinstance(value, TensorCardZone):
                for zone_field in fields(value):
                    tensor = getattr(value, zone_field.name)
                    result.append((f"{state_field.name}.{zone_field.name}", tensor))
        return tuple(result)

    def clone(self) -> TensorGameState:
        """Return an independent tensor copy while sharing immutable metadata."""
        return copy.deepcopy(self)


def _expect_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Validate one fixed tensor field."""
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} dtype {tensor.dtype} != {dtype}")


def _bounded_lengths(name: str, lengths: torch.Tensor, capacity: int) -> None:
    """Validate a padded field's recorded lengths."""
    if bool(torch.any(lengths < 0)) or bool(torch.any(lengths > capacity)):
        raise ValueError(f"{name} lengths exceed fixed capacity {capacity}")
