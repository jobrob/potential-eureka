"""Explicit recorded draw inputs for the Direction D2 vertical slice."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from heat.ml.spaces import MAX_PLAYERS
from heat.ml.vector_env.state import TensorGameState, TensorStateCapacityError


MAX_STRESS_DRAWS = 4
MAX_CARDS_PER_STRESS_DRAW = 24
MAX_REPLENISH_DRAWS = 7


@dataclass(frozen=True)
class RecordedDrawInputs:
    """Expected stress and replenish cards for each tensor lane."""

    card_ids: torch.Tensor
    lengths: torch.Tensor
    replenish_card_ids: torch.Tensor
    replenish_lengths: torch.Tensor

    @classmethod
    def create(
        cls,
        state: TensorGameState,
        records: dict[tuple[int, int], list[list[str]]],
        replenishments: dict[tuple[int, int], list[str]] | None = None,
    ) -> RecordedDrawInputs:
        """Encode human-readable card-ID sequences against a tensor vocabulary."""
        shape = (
            state.batch_size,
            MAX_PLAYERS,
            MAX_STRESS_DRAWS,
            MAX_CARDS_PER_STRESS_DRAW,
        )
        card_ids = torch.zeros(shape, dtype=torch.int64, device=state.game_ids.device)
        lengths = torch.zeros(shape[:-1], dtype=torch.int64, device=state.game_ids.device)
        replenish_card_ids = torch.zeros(
            (state.batch_size, MAX_PLAYERS, MAX_REPLENISH_DRAWS),
            dtype=torch.int64,
            device=state.game_ids.device,
        )
        replenish_lengths = torch.zeros(
            (state.batch_size, MAX_PLAYERS),
            dtype=torch.int64,
            device=state.game_ids.device,
        )
        vocabulary = {
            card_id: index + 1
            for index, card_id in enumerate(state.card_id_vocabulary)
        }
        for (game_id, player_id), stress_records in records.items():
            lane_matches = (state.game_ids == game_id).nonzero(as_tuple=False).flatten()
            if len(lane_matches) != 1:
                raise KeyError(f"unknown recorded-draw game_id {game_id}")
            lane = int(lane_matches[0].item())
            player_matches = (
                (state.player_ids[lane] == player_id) & state.player_present[lane]
            ).nonzero(as_tuple=False).flatten()
            if len(player_matches) != 1:
                raise KeyError(f"unknown recorded-draw player_id {player_id}")
            player = int(player_matches[0].item())
            if len(stress_records) > MAX_STRESS_DRAWS:
                raise TensorStateCapacityError(
                    f"stress count {len(stress_records)} exceeds "
                    f"{MAX_STRESS_DRAWS}"
                )
            for stress_index, card_sequence in enumerate(stress_records):
                if len(card_sequence) > MAX_CARDS_PER_STRESS_DRAW:
                    raise TensorStateCapacityError(
                        f"recorded stress draw count {len(card_sequence)} exceeds "
                        f"{MAX_CARDS_PER_STRESS_DRAW}"
                    )
                lengths[lane, player, stress_index] = len(card_sequence)
                for draw_index, card_id in enumerate(card_sequence):
                    try:
                        encoded = vocabulary[card_id]
                    except KeyError as exc:
                        raise KeyError(
                            f"recorded draw card {card_id!r} is not in state"
                        ) from exc
                    card_ids[lane, player, stress_index, draw_index] = encoded
        for (game_id, player_id), card_sequence in (replenishments or {}).items():
            lane, player = _resolve_identity(state, game_id, player_id)
            if len(card_sequence) > MAX_REPLENISH_DRAWS:
                raise TensorStateCapacityError(
                    f"recorded replenish draw count {len(card_sequence)} exceeds "
                    f"{MAX_REPLENISH_DRAWS}"
                )
            replenish_lengths[lane, player] = len(card_sequence)
            for draw_index, card_id in enumerate(card_sequence):
                try:
                    encoded = vocabulary[card_id]
                except KeyError as exc:
                    raise KeyError(
                        f"recorded replenish card {card_id!r} is not in state"
                    ) from exc
                replenish_card_ids[lane, player, draw_index] = encoded
        result = cls(
            card_ids,
            lengths,
            replenish_card_ids,
            replenish_lengths,
        )
        result.validate(state)
        return result

    def validate(self, state: TensorGameState) -> None:
        """Validate recorded input shapes and bounds against a tensor state."""
        expected_ids = (
            state.batch_size,
            MAX_PLAYERS,
            MAX_STRESS_DRAWS,
            MAX_CARDS_PER_STRESS_DRAW,
        )
        expected_lengths = expected_ids[:-1]
        expected_replenish_ids = (
            state.batch_size,
            MAX_PLAYERS,
            MAX_REPLENISH_DRAWS,
        )
        expected_replenish_lengths = expected_replenish_ids[:-1]
        if tuple(self.card_ids.shape) != expected_ids:
            raise ValueError(
                f"recorded card_ids shape {tuple(self.card_ids.shape)} != {expected_ids}"
            )
        if tuple(self.lengths.shape) != expected_lengths:
            raise ValueError(
                f"recorded lengths shape {tuple(self.lengths.shape)} != "
                f"{expected_lengths}"
            )
        if tuple(self.replenish_card_ids.shape) != expected_replenish_ids:
            raise ValueError(
                "recorded replenish_card_ids shape "
                f"{tuple(self.replenish_card_ids.shape)} != {expected_replenish_ids}"
            )
        if tuple(self.replenish_lengths.shape) != expected_replenish_lengths:
            raise ValueError(
                "recorded replenish_lengths shape "
                f"{tuple(self.replenish_lengths.shape)} != "
                f"{expected_replenish_lengths}"
            )
        tensors = (
            self.card_ids,
            self.lengths,
            self.replenish_card_ids,
            self.replenish_lengths,
        )
        if any(tensor.dtype != torch.int64 for tensor in tensors):
            raise TypeError("recorded draw tensors must use torch.int64")
        if any(tensor.device != state.game_ids.device for tensor in tensors):
            raise ValueError("recorded draws and state must share one device")
        if bool(torch.any(self.lengths < 0)) or bool(
            torch.any(self.lengths > MAX_CARDS_PER_STRESS_DRAW)
        ):
            raise ValueError("recorded draw lengths exceed fixed capacity")
        if bool(torch.any(self.replenish_lengths < 0)) or bool(
            torch.any(self.replenish_lengths > MAX_REPLENISH_DRAWS)
        ):
            raise ValueError("recorded replenish lengths exceed fixed capacity")


def reshuffle_discard_into_draw(
    state: TensorGameState,
    lane: int,
    player: int,
) -> int:
    """Apply the legacy deck reshuffle and advance only this lane's RNG."""
    if int(state.draw_pile.lengths[lane, player].item()) != 0:
        raise ValueError("reshuffle requires an empty draw pile")
    length = int(state.discard_pile.lengths[lane, player].item())
    if length == 0:
        return 0

    rng = _lane_rng(state, lane)
    order = list(range(length))
    rng.shuffle(order)
    order_tensor = torch.tensor(order, dtype=torch.int64, device=state.game_ids.device)
    for draw_values, discard_values in (
        (state.draw_pile.card_ids, state.discard_pile.card_ids),
        (state.draw_pile.card_types, state.discard_pile.card_types),
        (state.draw_pile.card_values, state.discard_pile.card_values),
    ):
        source = discard_values[lane, player, :length].clone()
        draw_values[lane, player, :length] = source[order_tensor]
        discard_values[lane, player].zero_()
    state.draw_pile.lengths[lane, player] = length
    state.discard_pile.lengths[lane, player] = 0
    _store_lane_rng(state, lane, rng)
    return length


def _resolve_identity(
    state: TensorGameState, game_id: int, player_id: int
) -> tuple[int, int]:
    """Resolve stable game/player identities into one tensor lane and slot."""
    lane_matches = (state.game_ids == game_id).nonzero(as_tuple=False).flatten()
    if len(lane_matches) != 1:
        raise KeyError(f"unknown recorded-draw game_id {game_id}")
    lane = int(lane_matches[0].item())
    player_matches = (
        (state.player_ids[lane] == player_id) & state.player_present[lane]
    ).nonzero(as_tuple=False).flatten()
    if len(player_matches) != 1:
        raise KeyError(f"unknown recorded-draw player_id {player_id}")
    return lane, int(player_matches[0].item())


def _lane_rng(state: TensorGameState, lane: int) -> random.Random:
    """Restore one exact Python RNG state from its fixed tensor fields."""
    gauss = (
        float(state.rng_gauss[lane].item())
        if bool(state.rng_gauss_present[lane].item())
        else None
    )
    rng = random.Random()
    rng.setstate(
        (
            int(state.rng_version[lane].item()),
            tuple(int(word) for word in state.rng_words[lane].tolist()),
            gauss,
        )
    )
    return rng


def _store_lane_rng(state: TensorGameState, lane: int, rng: random.Random) -> None:
    """Store one advanced Python RNG state back into its tensor lane."""
    version, words, gauss = rng.getstate()
    state.rng_version[lane] = version
    state.rng_words[lane] = torch.tensor(
        words, dtype=torch.int64, device=state.game_ids.device
    )
    state.rng_gauss_present[lane] = gauss is not None
    state.rng_gauss[lane] = 0.0 if gauss is None else gauss
