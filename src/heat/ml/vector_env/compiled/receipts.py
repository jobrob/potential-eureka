"""Fixed numeric event receipts for the Direction D2-R compiled boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from heat.ml.vector_env.kernels.common import TensorKernelEvent
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    CARD_TYPE_UPGRADE,
    PHASE_TO_CODE,
    TensorGameState,
)
from heat.models.game_state import Phase


MAX_EVENT_RECEIPTS = 64
RECEIPT_FIELD_COUNT = 8
MAX_RECEIPT_CARDS = 24
NO_VALUE = -1


class EventCode(IntEnum):
    """Frozen numeric codes for events emitted by cards-to-REACT."""

    PLAY_CARDS = 1
    TURN_START = 2
    STRESS_RESOLVED = 3
    REVEAL_AND_MOVE = 4
    ADRENALINE_GRANTED = 5
    REPLENISH = 6


EVENT_CODE_BY_NAME = {
    "play_cards": EventCode.PLAY_CARDS,
    "turn_start": EventCode.TURN_START,
    "stress_resolved": EventCode.STRESS_RESOLVED,
    "reveal_and_move": EventCode.REVEAL_AND_MOVE,
    "adrenaline_granted": EventCode.ADRENALINE_GRANTED,
    "replenish": EventCode.REPLENISH,
}
EVENT_NAME_BY_CODE = {int(code): name for name, code in EVENT_CODE_BY_NAME.items()}


@dataclass(frozen=True)
class NumericEventReceipts:
    """One fixed-capacity numeric event buffer per game lane.

    ``order_keys`` preserve the eager kernel's cross-lane event ordering. The
    compiled path writes the same schema directly; ``receipts_from_events`` is
    a cold-path adapter used to freeze and test the ABI against the oracle.
    """

    event_codes: torch.Tensor
    order_keys: torch.Tensor
    round_nums: torch.Tensor
    phase_codes: torch.Tensor
    player_ids: torch.Tensor
    fields: torch.Tensor
    card_types: torch.Tensor
    card_values: torch.Tensor
    card_lengths: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def empty(cls, batch_size: int, *, device: torch.device) -> NumericEventReceipts:
        """Allocate an empty receipt buffer using the frozen ABI."""
        event_shape = (batch_size, MAX_EVENT_RECEIPTS)
        card_shape = (*event_shape, MAX_RECEIPT_CARDS)
        return cls(
            event_codes=torch.zeros(event_shape, dtype=torch.int64, device=device),
            order_keys=torch.full(event_shape, NO_VALUE, dtype=torch.int64, device=device),
            round_nums=torch.zeros(event_shape, dtype=torch.int64, device=device),
            phase_codes=torch.zeros(event_shape, dtype=torch.int64, device=device),
            player_ids=torch.full(event_shape, NO_VALUE, dtype=torch.int64, device=device),
            fields=torch.full(
                (*event_shape, RECEIPT_FIELD_COUNT),
                NO_VALUE,
                dtype=torch.int64,
                device=device,
            ),
            card_types=torch.zeros(card_shape, dtype=torch.int64, device=device),
            card_values=torch.zeros(card_shape, dtype=torch.int64, device=device),
            card_lengths=torch.zeros(event_shape, dtype=torch.int64, device=device),
            lengths=torch.zeros((batch_size,), dtype=torch.int64, device=device),
        )

    @property
    def batch_size(self) -> int:
        """Return the number of game lanes represented by the buffer."""
        return int(self.lengths.shape[0])

    def validate(self) -> None:
        """Check every shape, dtype, and capacity before cold-path use."""
        event_shape = (self.batch_size, MAX_EVENT_RECEIPTS)
        expected = {
            "event_codes": event_shape,
            "order_keys": event_shape,
            "round_nums": event_shape,
            "phase_codes": event_shape,
            "player_ids": event_shape,
            "fields": (*event_shape, RECEIPT_FIELD_COUNT),
            "card_types": (*event_shape, MAX_RECEIPT_CARDS),
            "card_values": (*event_shape, MAX_RECEIPT_CARDS),
            "card_lengths": event_shape,
            "lengths": (self.batch_size,),
        }
        device = self.lengths.device
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            if tensor.dtype != torch.int64:
                raise TypeError(f"{name} dtype {tensor.dtype} != torch.int64")
            if tensor.device != device:
                raise ValueError("receipt tensors must share one device")
        if bool(torch.any(self.lengths < 0)) or bool(
            torch.any(self.lengths > MAX_EVENT_RECEIPTS)
        ):
            raise ValueError("event receipt length exceeds fixed capacity")
        if bool(torch.any(self.card_lengths < 0)) or bool(
            torch.any(self.card_lengths > MAX_RECEIPT_CARDS)
        ):
            raise ValueError("event card payload exceeds fixed capacity")


def receipts_from_events(
    events: tuple[TensorKernelEvent, ...], state: TensorGameState
) -> NumericEventReceipts:
    """Encode eager semantic events into the frozen numeric receipt ABI."""
    receipts = NumericEventReceipts.empty(
        state.batch_size, device=state.game_ids.device
    )
    lane_by_game = {
        int(game_id): lane for lane, game_id in enumerate(state.game_ids.tolist())
    }
    for order, event in enumerate(events):
        try:
            lane = lane_by_game[event.game_id]
        except KeyError as exc:
            raise KeyError(f"event references unknown game_id {event.game_id}") from exc
        slot = int(receipts.lengths[lane].item())
        if slot >= MAX_EVENT_RECEIPTS:
            raise OverflowError(
                f"game {event.game_id} exceeds {MAX_EVENT_RECEIPTS} event receipts"
            )
        try:
            code = EVENT_CODE_BY_NAME[event.event_type]
            phase_code = PHASE_TO_CODE[Phase(event.phase)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unsupported compiled receipt event {event.event_type!r}") from exc
        receipts.event_codes[lane, slot] = int(code)
        receipts.order_keys[lane, slot] = order
        receipts.round_nums[lane, slot] = event.round_num
        receipts.phase_codes[lane, slot] = phase_code
        receipts.player_ids[lane, slot] = (
            NO_VALUE if event.player_id is None else event.player_id
        )
        _encode_payload(receipts, lane, slot, code, event.data)
        receipts.lengths[lane] += 1
    receipts.validate()
    return receipts


def _encode_payload(
    receipts: NumericEventReceipts,
    lane: int,
    slot: int,
    code: EventCode,
    data: dict[str, object],
) -> None:
    """Encode one event payload according to its frozen field schema."""
    fields = receipts.fields[lane, slot]
    cards: object | None = None
    if code is EventCode.PLAY_CARDS:
        fields[0] = int(bool(data["cluttered"]))
        cards = data["cards"]
    elif code is EventCode.TURN_START:
        for index, name in enumerate(
            (
                "hand_size",
                "gear",
                "heat_available",
                "position",
                "next_corner_dist",
                "next_corner_speed_limit",
            )
        ):
            value = data[name]
            fields[index] = NO_VALUE if value is None else _int_field(value, name)
        cards = data["hand"]
    elif code is EventCode.STRESS_RESOLVED:
        fields[0] = _int_field(data["value"], "value")
        fields[1] = _int_field(data["flipped_count"], "flipped_count")
        cards = data["discarded"]
    elif code is EventCode.REVEAL_AND_MOVE:
        fields[0] = _int_field(data["speed"], "speed")
        fields[1] = _int_field(data["new_position"], "new_position")
        fields[2] = _int_field(data["lap"], "lap")
        fields[3] = int(bool(data["finished"]))
    elif code is EventCode.ADRENALINE_GRANTED:
        fields[0] = int(bool(data["eligible"]))
    elif code is EventCode.REPLENISH:
        fields[0] = _int_field(data["hand_size"], "hand_size")
        cards = data["drawn"]
    if cards is not None:
        if not isinstance(cards, list):
            raise TypeError("event card payload must be a list")
        if len(cards) > MAX_RECEIPT_CARDS:
            raise OverflowError("event card payload exceeds fixed receipt capacity")
        receipts.card_lengths[lane, slot] = len(cards)
        for index, display in enumerate(cards):
            card_type, value = _parse_card_display(str(display))
            receipts.card_types[lane, slot, index] = card_type
            receipts.card_values[lane, slot, index] = value


def _parse_card_display(display: str) -> tuple[int, int]:
    """Encode the exact legacy display vocabulary into numeric card fields."""
    if display == "Heat":
        return CARD_TYPE_HEAT, 0
    if display == "Stress":
        return CARD_TYPE_STRESS, 0
    if display.startswith("Upgrade(") and display.endswith(")"):
        return CARD_TYPE_UPGRADE, int(display[8:-1])
    try:
        return CARD_TYPE_SPEED, int(display)
    except ValueError as exc:
        raise ValueError(f"unsupported card display {display!r}") from exc


def _int_field(value: object, name: str) -> int:
    """Validate an integer event field before placing it in the ABI."""
    if not isinstance(value, int):
        raise TypeError(f"event field {name!r} must be an integer")
    return value
