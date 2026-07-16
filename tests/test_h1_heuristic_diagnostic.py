"""Focused tests for the A8 H1 deterministic loss replay."""

from __future__ import annotations

from experiments.diag_h1_heuristic import (
    GameSpec,
    classify_loss,
    fixed_specs,
    run_game,
)
from heat.engine.phases import ReactDecision
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.models.game_state import GameEvent, GameState, Phase
from heat.tracks.generator import generate_track


def test_fixed_specs_are_stable_and_rotate_strong_seat() -> None:
    """The screen coordinates and seat rotation must not depend on RNG state."""
    first = fixed_specs(tracks=2, games_per_track=3)
    assert first == fixed_specs(tracks=2, games_per_track=3)
    assert len(first) == 18
    assert [spec.strong_seat for spec in first if spec.seats == 2][:3] == [0, 1, 0]


def test_seeded_game_replays_with_identical_finish_and_trace() -> None:
    """A selected loss can be reconstructed solely from its GameSpec."""
    spec = GameSpec(seats=2, track_seed=700_000, game_seed=730_000, strong_seat=0)
    first, _, _, trace_a = run_game(spec, trace=True)
    second, _, _, trace_b = run_game(spec, trace=True)
    assert first.finish_order == second.finish_order
    assert first.total_rounds == second.total_rounds
    assert trace_a == trace_b
    assert trace_a


def test_classifier_detects_late_boost_conservatism_and_spin() -> None:
    """H1's two clearest local signals remain explicit and auditable."""
    track = generate_track(700_000)
    state = GameState.create(track, 2, seed=1)
    record = {
        "round": 3,
        "kind": "react",
        "lap": track.laps,
        "position": track.length - 5,
        "distance_to_finish": 5,
        "can_boost": True,
        "chosen": {
            **ReactDecision(0, False, False, False).__dict__,
        },
        "shadow": {
            **ReactDecision(0, True, False, False).__dict__,
        },
    }
    event = GameEvent(3, Phase.CHECK_CORNER, 0, "spin_out", {})
    signals, details = classify_loss([record], [event], state, 0, final_heat=4)
    assert signals == ["unsafe_corner_entry", "late_race_conservatism"]
    assert details["spins"] == 1


def test_certain_spin_guard_breaks_known_recovery_locks() -> None:
    """The five H1 repeated-spin fixtures no longer enter recovery loops."""
    specs = [
        GameSpec(4, 700_004, 750_012, 0),
        GameSpec(2, 700_004, 730_012, 0),
        GameSpec(2, 700_004, 730_013, 1),
        GameSpec(6, 700_002, 770_008, 2),
        GameSpec(6, 700_001, 770_004, 4),
    ]
    for spec in specs:
        _, result, _, _ = run_game(spec, trace=True)
        spins = sum(
            event.event_type == "spin_out"
            and event.player_id == spec.strong_seat
            for event in result.event_log
        )
        assert spins <= 1, f"{spec} still produced {spins} spins"


def test_guard_can_be_disabled_for_legacy_measurement() -> None:
    """The S1 evaluator can reproduce the pre-repair recovery behavior."""
    agent = StrongHeuristicAgent(avoid_certain_spins=False)
    assert not agent._avoid_certain_spins
