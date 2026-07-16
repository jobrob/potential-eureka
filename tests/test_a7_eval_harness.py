"""Tests for the Sprint A7 multi-seat / held-out evaluation harness (§6).

Covers: held-out disjointness (G2), checkpoint round-trip + codec tripwire (G5),
the ``evaluate_policy`` grid smoke + cell invariants, the first-place win
definition, the ``evaluate_vs_anchor`` self-play symmetry sanity, and the
heuristic-in-policy-seat baseline path (the G1 code path).
"""

from __future__ import annotations

import torch

from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.models.game_state import GameState
from heat.ml.selfplay.checkpoint import (
    CheckpointMismatchError,
    load_policy,
    save_policy,
)
from heat.ml.selfplay.eval_harness import (
    EvalReport,
    _first_place,
    evaluate_policy,
    evaluate_vs_anchor,
    held_out_tracks,
)
from heat.ml.selfplay.policy import build_policy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.selfplay.snapshots import SnapshotAgent
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.spaces import ACTION_DIM, CODEC_VERSION, OBS_DIM, _placement_reward
from heat.tracks.generator import track_sampler


def _full_fingerprint(track: object) -> tuple:
    """A full structural identity tag for a track.

    Adapts ``features._track_fingerprint`` to a stricter identity: adds per-space
    lanes and the start grid so only *genuinely identical* generated tracks
    collide -- the real disjointness claim (two different generation seeds
    producing byte-identical geometry is astronomically unlikely).
    """
    t = track
    return (
        len(t.spaces),  # type: ignore[attr-defined]
        t.laps,  # type: ignore[attr-defined]
        tuple(s.lanes for s in t.spaces),  # type: ignore[attr-defined]
        tuple((c.start, c.end, c.speed_limit) for c in t.corners),  # type: ignore[attr-defined]
        tuple(t.start_positions),  # type: ignore[attr-defined]
    )


def test_heldout_split_disjoint_from_training_namespaces() -> None:
    """G2: >=500 training-namespace tracks share no track with the held-out set.

    The generator partitions the track-seed space so ``base_seed == 0`` (the
    identity/eval namespace, seeds ``900_000+``) and any non-zero training
    ``base_seed`` land in structurally disjoint seed regions. This verifies the
    guarantee empirically by fingerprint rather than trusting the doc comment.
    """
    heldout = held_out_tracks(n=50)
    heldout_fps = {_full_fingerprint(t) for t in heldout}
    assert len(heldout_fps) == len(heldout)  # held-out set has no internal dupes

    train_fps: set[tuple] = set()
    # Two disjoint training namespaces, 300 seeds each = 600 training tracks.
    for base_seed in (1, 12345):
        sampler = track_sampler(base_seed=base_seed)
        for seed in range(300):
            train_fps.add(_full_fingerprint(sampler(seed)))

    assert len(train_fps & heldout_fps) == 0


def test_checkpoint_roundtrip_and_codec_tripwire(tmp_path: object) -> None:
    """G5: save->load reproduces outputs; a codec mismatch fails fast."""
    config = A0Config(hidden_sizes=(32, 32))
    policy = build_policy(config)

    torch.manual_seed(0)
    obs = torch.randn(5, OBS_DIM)
    actions = torch.zeros(5, dtype=torch.long)
    mask = torch.ones(5, ACTION_DIM, dtype=torch.bool)

    with torch.no_grad():
        logp0, value0, ent0 = policy.evaluate(obs, actions, mask)

    path = tmp_path / "policy.pt"  # type: ignore[operator]
    save_policy(policy, config, path)
    loaded = load_policy(path)

    with torch.no_grad():
        logp1, value1, ent1 = loaded.evaluate(obs, actions, mask)

    assert torch.allclose(logp0, logp1)
    assert torch.allclose(value0, value1)
    assert torch.allclose(ent0, ent1)
    assert loaded.obs_dim == OBS_DIM
    assert loaded.action_dim == ACTION_DIM

    # Tamper the codec_version -> load_policy must raise before touching weights.
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    blob["codec_version"] = CODEC_VERSION + 99
    bad = tmp_path / "bad.pt"  # type: ignore[operator]
    torch.save(blob, str(bad))
    try:
        load_policy(bad)
    except CheckpointMismatchError as exc:
        assert "codec_version" in str(exc)
    else:  # pragma: no cover - the tripwire must fire
        raise AssertionError("expected CheckpointMismatchError on codec mismatch")


def test_evaluate_policy_grid_smoke_and_invariants() -> None:
    """evaluate_policy on Tiny-Heat: report shape + per-cell arithmetic."""
    policy = build_policy(A0Config(hidden_sizes=(32, 32)))
    report = evaluate_policy(
        policy,
        opponents={"weak": HeuristicAgent},
        seat_counts=(2, 3),
        splits={"tiny": tiny_heat_track()},
        games_per_cell=4,
        seed=0,
    )
    assert isinstance(report, EvalReport)
    assert len(report.cells) == 2  # 1 opp x 2 seats x 1 split

    by_seats = {c.seat_count: c for c in report.cells}
    assert by_seats[2].chance == 0.5
    assert abs(by_seats[3].chance - 1.0 / 3.0) < 1e-9
    for c in report.cells:
        assert 0 <= c.wins <= c.games == 4
        assert c.wilson_lb <= c.win_rate <= c.wilson_ub
        assert c.split == "tiny"


def test_policy_evaluation_is_reproducible_and_rng_neutral() -> None:
    """A7 policy sampling is seeded per game and preserves caller torch RNG."""
    torch.manual_seed(11)
    policy = build_policy(A0Config(hidden_sizes=(16,)))

    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()
    first = evaluate_policy(
        policy,
        opponents={"weak": HeuristicAgent},
        seat_counts=(2,),
        splits={"tiny": tiny_heat_track()},
        games_per_cell=12,
        seed=77,
    )
    assert torch.equal(torch.random.get_rng_state(), state_before)

    # A different ambient RNG state must not change the report.
    torch.manual_seed(999_999)
    second = evaluate_policy(
        policy,
        opponents={"weak": HeuristicAgent},
        seat_counts=(2,),
        splits={"tiny": tiny_heat_track()},
        games_per_cell=12,
        seed=77,
    )
    assert first == second


def test_first_place_win_definition() -> None:
    """A constructed terminal state: first-place seat scores 1, others 0."""
    state = GameState.create(tiny_heat_track(), 3, seed=0)
    state.players[0].finished = True
    state.players[0].finish_order = 1
    state.players[1].finished = True
    state.players[1].finish_order = 2
    state.players[2].finished = True
    state.players[2].finish_order = 3

    assert _first_place(state, 0) is True
    assert _first_place(state, 1) is False
    assert _first_place(state, 2) is False
    # Placement reward grades the same ranking (winner +1, last -1 for n=3).
    assert _placement_reward(state, 0) == 1.0
    assert _placement_reward(state, 2) == -1.0


def test_evaluate_vs_anchor_selfplay_symmetry() -> None:
    """An anchor cloned from the same policy scores ~50% (wide tolerance)."""
    policy = build_policy(A0Config(hidden_sizes=(32, 32)))
    anchor = SnapshotAgent(policy, name="self")
    cell = evaluate_vs_anchor(
        policy, anchor, num_players=2, track=tiny_heat_track(), games=40, seed=0
    )
    assert cell.chance == 0.5
    assert cell.games == 40
    assert 0.25 <= cell.win_rate <= 0.75


def test_heuristic_in_policy_seat_beats_weak() -> None:
    """G1 path: a StrongHeuristicAgent in the policy seat beats a weak field."""
    report = evaluate_policy(
        StrongHeuristicAgent(),
        opponents={"weak": HeuristicAgent},
        seat_counts=(2,),
        splits={"tiny": tiny_heat_track()},
        games_per_cell=30,
        seed=0,
    )
    cell = report.cells[0]
    assert cell.win_rate > 0.5
