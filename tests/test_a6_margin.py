"""Tests for the A6 dense terminal-margin value target (§6 of the A6 design).

Covers:
  1. ``terminal_margin`` arithmetic: leader-ahead-by-k (clipped value +
     antisymmetry), both-finished, finished-vs-not, solo, and >full-lap clipping.
  2. Collector applies the margin only at game end and only when
     ``margin_coef != 0`` (earlier rewards identical; the terminal reward gains
     exactly ``coef * terminal_margin``).
  3. A truncated game pays the margin *and* the §4.6 truncation fold.
  4. ``margin_coef=0.0`` byte-identity: the stored reward stream equals the
     pre-A6 (default-constructor) collector's for the same seed.
  5. Config/CLI plumbing: ``A5Config.margin_coef`` reaches the *training*
     collector; the eval path never sees it.
  6. ``train_selfplay_a5(margin_coef=0.5)`` smoke: tiny run, finite losses.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.selfplay import recipe
from heat.ml.selfplay.multiseat import MultiSeatCollector
from heat.ml.selfplay.policy import HeatPolicy, PPOPolicy
from heat.ml.selfplay.recipe import A5Config, train_selfplay_a5
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.spaces import terminal_margin

DEVICE = torch.device("cpu")
LENGTH = 14  # tiny_heat_track() is 14 spaces, 1 lap.


def _multilap_track() -> Track:
    """A 3-lap Tiny-Heat track.

    On the stock 1-lap Tiny-Heat bed the margin is identically 0: players race
    at ``lap == 1`` so ``abs_pos = lap*length + pos >= laps*length`` and every
    ``remaining`` clips to 0 (this mirrors ``_track_block.dist_to_finish``, which
    is likewise degenerate on a 1-lap track). A multi-lap track keeps
    ``remaining`` positive mid-race, so the collector tests can exercise a
    genuinely non-zero margin term.
    """
    return replace(tiny_heat_track(), laps=3)


def _policy(seed: int = 0) -> HeatPolicy:
    """A tiny fresh policy (small trunk keeps the collection fast)."""
    torch.manual_seed(seed)
    return HeatPolicy(hidden_sizes=(16, 16))


def _make_state(
    track: Track,
    positions: list[int],
    laps: list[int],
    finished: list[bool],
) -> GameState:
    """A hand-built game state with each seat's lap/position/finished forced.

    Built from a real :func:`GameState.create` (so decks/hands are valid) then
    the kinematic fields are overwritten to the controlled test values.
    """
    n = len(positions)
    state = GameState.create(track, n, seed=0)
    for pid in range(n):
        p = state.get_player(pid)
        p.position = positions[pid]
        p.lap = laps[pid]
        p.finished = finished[pid]
    return state


# ---------------------------------------------------------------------------
# 1. terminal_margin arithmetic
# ---------------------------------------------------------------------------


def test_terminal_margin_arithmetic() -> None:
    track = tiny_heat_track()

    # Leader ahead by k=4 spaces (both unfinished, lap 0): remaining(me)=14-10=4,
    # remaining(opp)=14-6=8 -> margin(me) = (8-4)/14 = +4/14; antisymmetric.
    s = _make_state(track, positions=[10, 6], laps=[0, 0], finished=[False, False])
    m0 = terminal_margin(s, 0)
    m1 = terminal_margin(s, 1)
    assert abs(m0 - 4.0 / LENGTH) < 1e-9
    assert abs(m0 + m1) < 1e-9, "unclipped 2-seat margin must be antisymmetric"
    assert m0 > 0.0 > m1

    # Both finished -> zero remaining each -> 0 vs 0.
    s = _make_state(track, positions=[0, 0], laps=[0, 0], finished=[True, True])
    assert terminal_margin(s, 0) == 0.0
    assert terminal_margin(s, 1) == 0.0

    # Finished vs not-finished -> strictly positive for the finisher.
    s = _make_state(track, positions=[0, 6], laps=[0, 0], finished=[True, False])
    assert terminal_margin(s, 0) > 0.0, "finisher must lead the unfinished seat"
    assert terminal_margin(s, 1) < 0.0

    # Solo (n <= 1) -> 0.0 (no opponent to lead).
    solo = GameState.create(track, 1, seed=0)
    assert terminal_margin(solo, 0) == 0.0

    # Margin greater than a full lap clips to +-1. On a 2-lap track (total_len=28)
    # a finished seat vs an opponent at the start line is a 2-lap raw lead (2.0).
    two_lap = replace(track, laps=2)
    s = _make_state(two_lap, positions=[0, 0], laps=[0, 0], finished=[True, False])
    assert terminal_margin(s, 0) == 1.0, "raw margin 2.0 must clip to +1"
    assert terminal_margin(s, 1) == -1.0, "raw margin -2.0 must clip to -1"


# ---------------------------------------------------------------------------
# Collector helpers (capture terminal state + flags)
# ---------------------------------------------------------------------------


class _CapturingCollector(MultiSeatCollector):
    """Collector that records the last game's terminal state + episode flags."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.last_state: GameState | None = None
        self.last_terminated = False
        self.last_truncated = False

    def _on_game_end(
        self, state: GameState, terminated: bool, truncated: bool
    ) -> None:
        self.last_state = state
        self.last_terminated = terminated
        self.last_truncated = truncated


def _collect_one_game(
    policy: PPOPolicy,
    *,
    margin_coef: float,
    seed: int,
    track: Track | None = None,
) -> tuple[_CapturingCollector, list[list[float]], list[list[float]]]:
    """Drive exactly one seeded self-play game (n_steps=1 finishes one game).

    Returns ``(collector, per_seat_rewards, per_seat_dones)`` sliced to each
    seat's recorded length. Defaults to the multi-lap track so the margin term
    is live (see :func:`_multilap_track`).
    """
    torch.manual_seed(seed)
    collector = _CapturingCollector(
        track if track is not None else _multilap_track(), 2,
        margin_coef=margin_coef,
    )
    buffers, _ = collector.collect(
        policy, 1, DEVICE, np.random.default_rng(seed), gamma=0.99
    )
    rewards = [list(buffers[s].rewards[: len(buffers[s])]) for s in range(2)]
    dones = [list(buffers[s].dones[: len(buffers[s])]) for s in range(2)]
    return collector, rewards, dones


# ---------------------------------------------------------------------------
# 2. Collector applies margin only at game end, only when coef != 0
# ---------------------------------------------------------------------------


def test_collector_margin_only_at_game_end(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # At a fully *terminated* game every player has finished, so every
    # ``remaining`` is 0 and the margin is 0 (placement grades finish order).
    # The margin only carries signal on TRUNCATED ends (design §2.2), so force
    # truncation to exercise a genuinely non-zero margin.
    monkeypatch.setattr("heat.ml.selfplay.multiseat.MAX_ROUNDS", 1)
    policy = _policy(0)
    coef = 0.5
    base, b_rew, b_done = _collect_one_game(policy, margin_coef=0.0, seed=7)
    marg, m_rew, _ = _collect_one_game(policy, margin_coef=coef, seed=7)

    assert base.last_state is not None
    assert base.last_truncated and not base.last_terminated, "game must truncate"
    assert any(
        abs(terminal_margin(base.last_state, s)) > 0.0 for s in range(2)
    ), "test needs a non-zero margin to be meaningful"
    # Same trajectory (rewards never feed back into action sampling): identical
    # obs/actions/dones across the two runs, so we can compare rewards elementwise.
    for seat in range(2):
        b, m, dones = b_rew[seat], m_rew[seat], b_done[seat]
        assert len(b) == len(m) == len(dones)
        assert len(b) > 0, "policy seat must have recorded transitions"
        expected_margin = coef * terminal_margin(base.last_state, seat)
        saw_terminal = False
        for i, done in enumerate(dones):
            if done:  # game-end transition: gains exactly coef * margin
                assert abs((m[i] - b[i]) - expected_margin) < 1e-5
                saw_terminal = True
            else:  # intermediate transition: untouched
                assert abs(m[i] - b[i]) < 1e-6
        assert saw_terminal, "each seat stream must end on a done=True transition"


# ---------------------------------------------------------------------------
# 3. Truncated game pays margin AND the truncation fold
# ---------------------------------------------------------------------------


def test_truncated_game_pays_margin_and_fold(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Force truncation: MAX_ROUNDS=1 makes almost every game hit the time limit.
    monkeypatch.setattr("heat.ml.selfplay.multiseat.MAX_ROUNDS", 1)
    policy = _policy(0)
    coef = 0.5
    base, b_rew, b_done = _collect_one_game(policy, margin_coef=0.0, seed=3)
    marg, m_rew, _ = _collect_one_game(policy, margin_coef=coef, seed=3)

    assert base.last_truncated and not base.last_terminated, "game must truncate"
    assert base.last_state is not None
    for seat in range(2):
        b, m, dones = b_rew[seat], m_rew[seat], b_done[seat]
        # Terminal index of this seat's stream.
        term_idx = max(i for i, d in enumerate(dones) if d)
        margin = terminal_margin(base.last_state, seat)
        assert abs(margin) > 0.0, "seat margin must be non-zero at truncation"
        # The margin term is present at the truncated end.
        assert abs((m[term_idx] - b[term_idx]) - coef * margin) < 1e-5
        # The truncation fold is present: on truncation the placement term is 0,
        # so the baseline (coef=0) terminal reward is exactly the gamma*V fold.
        assert abs(b[term_idx]) > 0.0, "truncation fold must be nonzero"


# ---------------------------------------------------------------------------
# 4. margin_coef=0.0 byte-identity to the pre-A6 collector
# ---------------------------------------------------------------------------


def test_margin_coef_zero_byte_identity(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    policy = _policy(0)
    # Default-constructor collector == the pre-A6 API (no margin_coef).
    torch.manual_seed(11)
    default = MultiSeatCollector(tiny_heat_track(), 2)
    d_buf, _ = default.collect(
        policy, 200, DEVICE, np.random.default_rng(11), gamma=0.99
    )
    torch.manual_seed(11)
    explicit = MultiSeatCollector(tiny_heat_track(), 2, margin_coef=0.0)
    e_buf, _ = explicit.collect(
        policy, 200, DEVICE, np.random.default_rng(11), gamma=0.99
    )
    for seat in range(2):
        n = len(d_buf[seat])
        assert n == len(e_buf[seat])
        assert np.array_equal(d_buf[seat].rewards[:n], e_buf[seat].rewards[:n])
        assert np.array_equal(d_buf[seat].dones[:n], e_buf[seat].dones[:n])

    # And a non-zero coef genuinely diverges (guards against a vacuous test).
    # The margin is only non-zero on truncated ends, so force truncation.
    monkeypatch.setattr("heat.ml.selfplay.multiseat.MAX_ROUNDS", 1)
    _, z_rew, _ = _collect_one_game(policy, margin_coef=0.0, seed=11)
    _, nz_rew, _ = _collect_one_game(policy, margin_coef=0.5, seed=11)
    assert z_rew[0] != nz_rew[0], "coef=0.5 must change the terminal reward"


# ---------------------------------------------------------------------------
# 5. Config/CLI plumbing: training collector gets the coef, eval never does
# ---------------------------------------------------------------------------


def test_config_plumbing_training_only(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert A5Config(margin_coef=0.5).margin_coef == 0.5

    train_coefs: list[float] = []
    eval_coefs: list[float] = []
    real_msc = recipe.MultiSeatCollector
    real_eval = recipe._EvalCollector

    class _SpyTrain(real_msc):  # type: ignore[valid-type,misc]
        def __init__(self, *a: object, margin_coef: float = 0.0, **k: object) -> None:
            train_coefs.append(margin_coef)
            super().__init__(*a, margin_coef=margin_coef, **k)  # type: ignore[arg-type]

    class _SpyEval(real_eval):  # type: ignore[valid-type,misc]
        def __init__(self, *a: object, margin_coef: float = 0.0, **k: object) -> None:
            eval_coefs.append(margin_coef)
            super().__init__(*a, margin_coef=margin_coef, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(recipe, "MultiSeatCollector", _SpyTrain)
    monkeypatch.setattr(recipe, "_EvalCollector", _SpyEval)

    config = A5Config(
        n_steps=64, batch_size=64, n_epochs=2, total_timesteps=64 * 2,
        hidden_sizes=(16, 16), num_players=2, seed=0, device="cpu",
        eval_games=2, eval_every=1, snapshot_every=1, stage1_enabled=False,
        margin_coef=0.5,
    )
    train_selfplay_a5(config, track=tiny_heat_track())

    assert train_coefs, "training collector must be constructed"
    assert all(c == 0.5 for c in train_coefs), "training collector needs the coef"
    assert eval_coefs, "eval collector must be constructed (eval ran)"
    assert all(c == 0.0 for c in eval_coefs), "eval must NEVER see the margin"


# ---------------------------------------------------------------------------
# 6. train_selfplay_a5(margin_coef=0.5) smoke
# ---------------------------------------------------------------------------


def test_train_selfplay_a5_margin_smoke() -> None:
    config = A5Config(
        n_steps=128, batch_size=64, n_epochs=2, total_timesteps=128 * 3,
        hidden_sizes=(16, 16), num_players=2, seed=0, device="cpu",
        eval_games=6, snapshot_every=1, eval_every=2, stage1_enabled=False,
        margin_coef=0.5,
    )
    infos: list[dict[str, float]] = []
    policy, records = train_selfplay_a5(
        config, track=tiny_heat_track(),
        on_iteration=lambda it, info: infos.append(info),
    )
    assert isinstance(policy, HeatPolicy)
    assert len(infos) == 3
    for info in infos:
        for key in ("policy_loss", "value_loss", "entropy", "ent_coef"):
            assert np.isfinite(info[key]), f"{key} not finite: {info[key]}"
    assert records
    for rec in records:
        assert np.isfinite(rec["winrate_vs_weak"])
        assert 0.0 <= rec["winrate_vs_weak"] <= 1.0
