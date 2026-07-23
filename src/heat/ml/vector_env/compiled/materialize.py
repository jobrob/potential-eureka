"""Cold-path conversion from D2-R numeric receipts to semantic events."""

from __future__ import annotations

from heat.ml.vector_env.compiled.receipts import (
    EVENT_NAME_BY_CODE,
    NO_VALUE,
    EventCode,
    NumericEventReceipts,
)
from heat.ml.vector_env.kernels.common import TensorKernelEvent
from heat.ml.vector_env.state import (
    CARD_TYPE_HEAT,
    CARD_TYPE_SPEED,
    CARD_TYPE_STRESS,
    CARD_TYPE_UPGRADE,
    PHASE_ORDER,
    TensorGameState,
)


def materialize_receipts(
    receipts: NumericEventReceipts, state: TensorGameState
) -> tuple[TensorKernelEvent, ...]:
    """Rebuild canonical D2 events and restore their eager global ordering."""
    receipts.validate()
    if receipts.batch_size != state.batch_size:
        raise ValueError("receipt and state batch sizes differ")
    ordered: list[tuple[int, TensorKernelEvent]] = []
    for lane in range(receipts.batch_size):
        for slot in range(int(receipts.lengths[lane].item())):
            code_value = int(receipts.event_codes[lane, slot].item())
            try:
                code = EventCode(code_value)
                event_name = EVENT_NAME_BY_CODE[code_value]
                phase = PHASE_ORDER[int(receipts.phase_codes[lane, slot].item())]
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(f"invalid numeric event code {code_value}") from exc
            player_id = int(receipts.player_ids[lane, slot].item())
            event = TensorKernelEvent(
                game_id=int(state.game_ids[lane].item()),
                round_num=int(receipts.round_nums[lane, slot].item()),
                phase=phase.value,
                player_id=None if player_id == NO_VALUE else player_id,
                event_type=event_name,
                data=_materialize_payload(receipts, lane, slot, code),
            )
            ordered.append((int(receipts.order_keys[lane, slot].item()), event))
    ordered.sort(key=lambda item: item[0])
    orders = [order for order, _event in ordered]
    if any(order < 0 for order in orders) or len(set(orders)) != len(orders):
        raise ValueError("receipt order keys must be non-negative and unique")
    return tuple(event for _order, event in ordered)


def _materialize_payload(
    receipts: NumericEventReceipts, lane: int, slot: int, code: EventCode
) -> dict[str, object]:
    """Decode one event's fixed scalar and card payload fields."""
    fields = receipts.fields[lane, slot]
    cards = _materialize_cards(receipts, lane, slot)
    if code is EventCode.PLAY_CARDS:
        return {"cards": cards, "cluttered": bool(fields[0].item())}
    if code is EventCode.TURN_START:
        return {
            "hand": cards,
            "hand_size": int(fields[0].item()),
            "gear": int(fields[1].item()),
            "heat_available": int(fields[2].item()),
            "position": int(fields[3].item()),
            "next_corner_dist": _optional_int(int(fields[4].item())),
            "next_corner_speed_limit": _optional_int(int(fields[5].item())),
        }
    if code is EventCode.STRESS_RESOLVED:
        return {
            "value": int(fields[0].item()),
            "flipped_count": int(fields[1].item()),
            "discarded": cards,
        }
    if code is EventCode.REVEAL_AND_MOVE:
        return {
            "speed": int(fields[0].item()),
            "new_position": int(fields[1].item()),
            "lap": int(fields[2].item()),
            "finished": bool(fields[3].item()),
        }
    if code is EventCode.ADRENALINE_GRANTED:
        return {"eligible": bool(fields[0].item())}
    if code is EventCode.REPLENISH:
        return {"hand_size": int(fields[0].item()), "drawn": cards}
    raise AssertionError(f"unhandled event code {code}")


def _materialize_cards(
    receipts: NumericEventReceipts, lane: int, slot: int
) -> list[str]:
    """Decode a receipt card payload into exact legacy display strings."""
    result: list[str] = []
    for index in range(int(receipts.card_lengths[lane, slot].item())):
        card_type = int(receipts.card_types[lane, slot, index].item())
        value = int(receipts.card_values[lane, slot, index].item())
        if card_type == CARD_TYPE_SPEED:
            result.append(str(value))
        elif card_type == CARD_TYPE_HEAT:
            result.append("Heat")
        elif card_type == CARD_TYPE_STRESS:
            result.append("Stress")
        elif card_type == CARD_TYPE_UPGRADE:
            result.append(f"Upgrade({value})")
        else:
            raise ValueError(f"invalid numeric card type {card_type}")
    return result


def _optional_int(value: int) -> int | None:
    """Decode the ABI's sentinel for an absent optional integer."""
    return None if value == NO_VALUE else value
