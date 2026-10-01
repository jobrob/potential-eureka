"""Tests for the A2 N-agent shared-policy self-play harness (multiseat.py).

Covers the §6 test list of ``docs/direction-A/A2-multiseat-selfplay.md`` and the
acceptance gates (§5):

  1. Legality/termination sweep (G1): fresh policy on Tiny-Heat 2p/4p + a USA
     case; every sampled action is masked-legal, every seat stream ends done=True.
  2. Stream isolation (G2): each transition lands only in its own seat's buffer.
  3. Terminal credit (G2): a seeded 2p game's last per-seat reward equals the
     ``_placement_reward`` of the actual finish order; earlier rewards are 0.
  4. Truncation bootstrap (G2, §4.6): a truncated episode's final reward folds in
     ``gamma * V(s_next)``.
  5. Scripted-seat cross-check: a scripted seat is advanced but never recorded,
     and the recorded seat matches a manually-driven HeatEnv learner (the bridge
     between the env path and the self-play path).
  6. ``train_multiseat`` smoke (G3/G4): 2 iterations, tiny net, finite losses.
  7. Refactor identity (G4): HeatEnv's ``_decode_legal``/``_forced_action`` are
     thin delegations to the shared ``action_codec`` functions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from heat.agents.heuristic_agent import HeuristicAgent
from heat.engine.driver import Decision, DecisionKind, simultaneous_decisions
from heat.models.game_state import GameState
from heat.ml import action_codec
from heat.ml.action_codec import decode_legal_action, forced_action, legal_action_mask
from heat.ml.env import HeatEnv
from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.multiseat import (
    CollectorTiming,
    MultiSeatCollector,
    train_multiseat,
)
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.ppo import A0Config
from heat.ml.selfplay.tiny_heat import tiny_heat_track
from heat.ml.spaces import _placement_reward
from heat.tracks.loader import load_track_by_name

DEVICE = torch.device("cpu")


def _policy(seed: int = 0) -> HeatPolicy:
    """A tiny fresh policy (small trunk keeps the sweep fast)."""
    torch.manual_seed(seed)
    return HeatPolicy(hidden_sizes=(16, 16))


# ---------------------------------------------------------------------------
# 1. Legality / termination sweep (G1)
# ---------------------------------------------------------------------------


class _LegalityCollector(MultiSeatCollector):
    """Collector that asserts every stored action was legal under its mask."""

    def _store(  # type: ignore[override]
        self,
        buffers: list[RolloutBuffer],
        seat: int,
        pending: object,
        reward: float,
        done: bool,
    ) -> None:
        p = pending  # type: ignore[assignment]
        assert bool(p.mask[p.action]), (  # type: ignore[attr-defined]
            f"seat {seat} sampled illegal action {p.action}"  # type: ignore[attr-defined]
        )
        super()._store(buffers, seat, pending, reward, done)  # type: ignore[arg-type]


def test_legality_and_termination_sweep() -> None:
    """Fresh policy over Tiny-Heat 2p/4p and USA 2p: no illegal action ever
    reaches a buffer, and every non-empty seat stream ends on done=True (no
    pending-transition leak)."""
    tiny = tiny_heat_track()
    usa = load_track_by_name("usa")
    cases = [(tiny, 2, 200), (tiny, 4, 200), (usa, 2, 120)]

    for track, players, n_steps in cases:
        policy = _policy()
        collector = _LegalityCollector(track, players)
        buffers, returns = collector.collect(
            policy, n_steps, DEVICE, np.random.default_rng(0), gamma=0.99
        )
        # Policy seats recorded transitions; scripted seats (none) stay empty.
        for seat in collector.policy_seats:
            buf = buffers[seat]
            if len(buf) > 0:
                assert bool(buf.dones[len(buf) - 1]), (
                    f"seat {seat} stream must end on done=True"
                )
        assert sum(len(buffers[s]) for s in collector.policy_seats) >= n_steps
        assert returns  # at least one completed game recorded


# ---------------------------------------------------------------------------
# 2. Stream isolation (G2)
# ---------------------------------------------------------------------------


class _IsolationCollector(MultiSeatCollector):
    """Records, for every ``_store`` call, exactly which buffers grew."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.emits: list[tuple[int, tuple[int, ...]]] = []

    def _store(  # type: ignore[override]
        self,
        buffers: list[RolloutBuffer],
        seat: int,
        pending: object,
        reward: float,
        done: bool,
    ) -> None:
        before = [len(b) for b in buffers]
        super()._store(buffers, seat, pending, reward, done)  # type: ignore[arg-type]
        after = [len(b) for b in buffers]
        grown = tuple(i for i in range(len(buffers)) if after[i] != before[i])
        self.emits.append((seat, grown))


def test_stream_isolation() -> None:
    """Every transition emitted for seat i grows ONLY buffers[i] -- the per-seat
    streams are never interleaved (the GAE single-stream invariant)."""
    policy = _policy()
    collector = _IsolationCollector(tiny_heat_track(), 4)
    buffers, _ = collector.collect(
        policy, 300, DEVICE, np.random.default_rng(1), gamma=0.99
    )

    assert collector.emits
    for seat, grown in collector.emits:
        assert grown == (seat,), (
            f"store for seat {seat} grew buffers {grown}, expected only ({seat},)"
        )
    # Per-seat store counts exactly account for each buffer's length.
    for seat in range(4):
        n_for_seat = sum(1 for s, _ in collector.emits if s == seat)
        assert n_for_seat == len(buffers[seat])
    # Seats collect independently (lengths need not be identical).
    lengths = [len(buffers[s]) for s in collector.policy_seats]
    assert all(n > 0 for n in lengths)


# ---------------------------------------------------------------------------
# 3. Terminal credit (G2)
# ---------------------------------------------------------------------------


class _CaptureCollector(MultiSeatCollector):
    """Captures each game's terminal state + episode flags via the hook."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.ends: list[tuple[object, bool, bool]] = []

    def _on_game_end(  # type: ignore[override]
        self, state: object, terminated: bool, truncated: bool
    ) -> None:
        self.ends.append((state, terminated, truncated))


def test_terminal_placement_credit() -> None:
    """One seeded 2p game: each seat's LAST stored reward equals the placement
    reward of the actual finish order, earlier rewards are 0, and the two seats'
    terminal rewards sum to ~0 (zero-sum placement)."""
    policy = _policy()
    collector = _CaptureCollector(tiny_heat_track(), 2)
    # n_steps=1 -> exactly one complete game (finish the in-flight game rule).
    buffers, _ = collector.collect(
        policy, 1, DEVICE, np.random.default_rng(7), gamma=0.99
    )
    assert len(collector.ends) == 1
    state, terminated, truncated = collector.ends[0]
    assert terminated and not truncated, "Tiny-Heat game should finish, not truncate"

    terminal_rewards = []
    for seat in range(2):
        buf = buffers[seat]
        assert len(buf) > 0
        last = len(buf) - 1
        expected = _placement_reward(state, seat)  # type: ignore[arg-type]
        assert np.isclose(buf.rewards[last], expected, atol=1e-6), (
            f"seat {seat}: last reward {buf.rewards[last]} != placement {expected}"
        )
        # Under default sparse race reward every non-terminal reward is 0.
        assert np.allclose(buf.rewards[:last], 0.0)
        terminal_rewards.append(expected)
    assert np.isclose(sum(terminal_rewards), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Truncation bootstrap (G2, §4.6)
# ---------------------------------------------------------------------------


def test_truncation_bootstrap_fold(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With MAX_ROUNDS forced low the game truncates; each seat's final reward
    folds in ``gamma * V(s_next)`` on top of the (sparse) terminal placement
    reward, per §4.6 -- a truncation must NOT zero the value bootstrap.

    The relation checked is exact:
    ``final_reward == _placement_reward(state, seat) + gamma * V(s_next)``.
    A seat that reached the finish before the cutoff carries its placement term;
    an unfinished seat's placement term is 0, leaving pure ``gamma * V(s_next)``.
    """
    # Patch the name the collector resolves at call time.
    monkeypatch.setattr("heat.ml.selfplay.multiseat.MAX_ROUNDS", 2)

    policy = _policy()
    gamma = 0.97
    collector = _CaptureCollector(tiny_heat_track(), 2)
    buffers, _ = collector.collect(
        policy, 1, DEVICE, np.random.default_rng(3), gamma=gamma
    )
    state, terminated, truncated = collector.ends[0]
    assert truncated and not terminated, "forced-low MAX_ROUNDS must truncate"

    from heat.ml.features import encode_observation

    folded_any = False
    for seat in range(2):
        buf = buffers[seat]
        assert len(buf) > 0
        last = len(buf) - 1
        s_next = encode_observation(state, seat, None)  # type: ignore[arg-type]
        v_next = MultiSeatCollector._seat_value(policy, s_next, DEVICE)
        placement = _placement_reward(state, seat)  # type: ignore[arg-type]
        expected = placement + gamma * v_next
        assert np.isclose(buf.rewards[last], expected, atol=1e-5), (
            f"seat {seat}: final reward {buf.rewards[last]} != "
            f"placement+gamma*V {expected}"
        )
        folded_any = folded_any or abs(gamma * v_next) > 1e-9
    # The whole point of §4.6: the bootstrap is actually non-zero and folded in.
    assert folded_any, "truncation bootstrap fold contributed nothing"


# ---------------------------------------------------------------------------
# 5. Scripted-seat cross-check (bridge to the A0 env path)
# ---------------------------------------------------------------------------


def _drive_env_episode(
    env: HeatEnv, policy: HeatPolicy, seed: int
) -> list[tuple[int, float, bool]]:
    """Drive one HeatEnv episode with ``policy`` (like collect_rollout), returning
    the learner's ``(action, reward, done)`` per step."""
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    out: list[tuple[int, float, bool]] = []
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        action_t, _lp, _v, _e = policy.act(obs_t, mask_t)
        action = int(action_t.item())
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        out.append((action, float(reward), done))
        mask = info["action_mask"]
        if done:
            break
    return out


def test_scripted_seat_matches_env_path(monkeypatch) -> None:
    """Seat 1 scripted (HeuristicAgent): only seat 0 records, and its recorded
    (action, reward, done) sequence matches a manually-driven HeatEnv learner
    under the same game seed + torch seed -- the regression bridge between the
    env path and the self-play path."""
    from heat.ml import spaces

    # Match the collector's explicit zero-shaping contract, independent of prior trainers.
    monkeypatch.setattr(spaces, "SHAPING_WEIGHT", 0.0)
    monkeypatch.setattr(spaces, "SHAPING_SPINOUT_WEIGHT", 0.0)
    track = tiny_heat_track()
    # Deterministic game seed: the collector's first rng draw.
    seed = int(np.random.default_rng(11).integers(0, 2**31 - 1))

    # Collector path (seat 1 scripted). Seed torch identically before driving.
    torch.manual_seed(123)
    policy_c = HeatPolicy(hidden_sizes=(16, 16))
    collector = MultiSeatCollector(
        track, 2, scripted_seats={1: HeuristicAgent()}
    )
    torch.manual_seed(999)
    buffers, _ = collector.collect(
        policy_c, 1, DEVICE, np.random.default_rng(11), gamma=0.99
    )
    assert len(buffers[1]) == 0, "scripted seat 1 must not be recorded"
    assert len(buffers[0]) > 0, "policy seat 0 must be recorded"
    assert bool(buffers[0].dones[len(buffers[0]) - 1])

    # Env path: same seed, same policy weights, same torch RNG before the drive.
    torch.manual_seed(123)
    policy_e = HeatPolicy(hidden_sizes=(16, 16))
    env = HeatEnv(track=track, num_players=2, opponents=HeuristicAgent(),
                  learner_id=0, randomize_seat=False)
    torch.manual_seed(999)
    env_steps = _drive_env_episode(env, policy_e, seed)

    n = len(buffers[0])
    assert n == len(env_steps), (
        f"collector recorded {n} seat-0 steps, env recorded {len(env_steps)}"
    )
    for i in range(n):
        c_action = int(buffers[0].actions[i])
        c_reward = float(buffers[0].rewards[i])
        c_done = bool(buffers[0].dones[i])
        e_action, e_reward, e_done = env_steps[i]
        assert c_action == e_action, f"step {i}: action {c_action} != {e_action}"
        assert np.isclose(c_reward, e_reward, atol=1e-6), (
            f"step {i}: reward {c_reward} != {e_reward}"
        )
        assert c_done == e_done


# ---------------------------------------------------------------------------
# 6. train_multiseat smoke (G3 / G4)
# ---------------------------------------------------------------------------


def test_train_multiseat_smoke() -> None:
    """2 iterations, tiny net, Tiny-Heat: every logged loss is finite and the
    per-iteration returns list is non-empty."""
    config = A0Config(
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        total_timesteps=256,  # -> 2 iterations
        hidden_sizes=(16, 16),
        num_players=2,
        seed=0,
        device="cpu",
    )
    infos: list[dict[str, float]] = []
    policy = train_multiseat(
        config, track=tiny_heat_track(),
        on_iteration=lambda it, info: infos.append(info),
    )
    assert isinstance(policy, HeatPolicy)
    assert len(infos) == 2
    for info in infos:
        for key in (
            "policy_loss",
            "value_loss",
            "update_entropy",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ):
            assert np.isfinite(info[key]), f"{key} not finite: {info[key]}"
        assert "entropy" not in info
        assert info["n_recorded"] >= config.n_steps
        assert info["n_episodes"] >= 1


# ---------------------------------------------------------------------------
# 7. Refactor identity (G4)
# ---------------------------------------------------------------------------


def test_env_delegates_to_action_codec() -> None:
    """HeatEnv's ``_decode_legal``/``_forced_action`` are thin delegations to the
    shared ``action_codec`` functions, so the env path and the self-play path can
    never drift apart."""
    import heat.ml.env as env_mod

    # The sentinel is the SAME object (env re-exports action_codec.NO_FORCED).
    assert env_mod._NO_FORCED is action_codec.NO_FORCED

    env = HeatEnv(track=tiny_heat_track(), num_players=2)
    env.reset(seed=0)

    # forced_action: env method and module function agree object-for-object.
    real_choice = Decision(DecisionKind.GEAR, env.learner_id, [(1, 0), (2, 0)])
    assert env._forced_action(real_choice) is action_codec.NO_FORCED
    assert (
        forced_action(real_choice, env.state) is env._forced_action(real_choice)
    )

    forced = Decision(DecisionKind.GEAR, env.learner_id, [(1, 0)])
    assert env._forced_action(forced) == forced_action(forced, env.state) == (1, 0)

    # decode_legal_action: env method delegates identically. Find a real GEAR
    # decision's legal index and check both paths decode it the same.
    mask = legal_action_mask(real_choice, env.state)
    idx = int(np.flatnonzero(mask)[0])
    assert env._decode_legal(real_choice, idx) == decode_legal_action(
        real_choice, env.state, idx
    )


# ---------------------------------------------------------------------------
# 8. T2 phase-first batching
# ---------------------------------------------------------------------------


class _FirstLegalPolicy:
    """Deterministic fixture policy that records its flattened action trace."""

    def __init__(self) -> None:
        self.trace: list[int] = []

    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose the first legal action independently for every row."""
        actions = mask.to(dtype=torch.int64).argmax(dim=1)
        self.trace.extend(int(action) for action in actions.tolist())
        zeros = torch.zeros(actions.shape[0], dtype=torch.float32, device=obs.device)
        return actions, zeros, zeros, zeros


class _TerminalCollector(MultiSeatCollector):
    """Capture a compact terminal-state fingerprint for mode comparisons."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.terminal: tuple[object, ...] | None = None

    def _on_game_end(
        self, state: GameState, terminated: bool, truncated: bool
    ) -> None:
        self.terminal = (
            terminated,
            truncated,
            state.round_num,
            tuple(
                (
                    player.position,
                    player.lap,
                    player.gear,
                    player.finished,
                    player.finish_order,
                )
                for player in state.players
            ),
        )


def _buffer_hash(buffers: list[RolloutBuffer]) -> str:
    """Hash every populated rollout field in stable seat/field order."""
    digest = hashlib.sha256()
    for buffer in buffers:
        n = len(buffer)
        digest.update(n.to_bytes(8, "little"))
        for name in ("obs", "actions", "logps", "values", "rewards", "dones", "masks"):
            digest.update(getattr(buffer, name)[:n].tobytes())
    return digest.hexdigest()


def test_simultaneous_decision_helper_is_read_only() -> None:
    """Enumerating either simultaneous phase does not alter live state."""
    state = GameState.create(tiny_heat_track(), 6, seed=19)
    before = (
        tuple(repr(player) for player in state.players),
        tuple(state.turn_order),
        state.rng.getstate(),
    )
    gears = simultaneous_decisions(state, DecisionKind.GEAR)
    cards = simultaneous_decisions(state, DecisionKind.CARDS)
    after = (
        tuple(repr(player) for player in state.players),
        tuple(state.turn_order),
        state.rng.getstate(),
    )
    assert [decision.player_id for decision in gears] == list(range(6))
    assert [decision.player_id for decision in cards] == list(range(6))
    assert before == after


def test_phase_mode_matches_scalar_with_deterministic_fixture() -> None:
    """Scalar and phase modes preserve complete traces, buffers, and endings."""
    results: dict[str, tuple[list[int], str, tuple[object, ...] | None]] = {}
    for mode in ("scalar", "phase"):
        policy = _FirstLegalPolicy()
        collector = _TerminalCollector(
            tiny_heat_track(), 6, collector_mode=mode
        )
        buffers, _returns = collector.collect(  # type: ignore[arg-type]
            policy, 1, DEVICE, np.random.default_rng(31), gamma=0.99
        )
        results[mode] = (policy.trace, _buffer_hash(buffers), collector.terminal)
    assert results["scalar"] == results["phase"]


def _phase_rollout(sample_seed: int) -> tuple[str, list[int], CollectorTiming]:
    """Run one reproducible stochastic phase-mode game for T2 checks."""
    policy = _policy(23)
    collector = _LegalityCollector(
        tiny_heat_track(), 4, collector_mode="phase"
    )
    timing = CollectorTiming()
    torch.manual_seed(sample_seed)
    buffers, _returns = collector.collect(
        policy,
        1,
        DEVICE,
        np.random.default_rng(47),
        gamma=0.99,
        timing=timing,
    )
    actions = [
        int(action)
        for buffer in buffers
        for action in buffer.actions[: len(buffer)]
    ]
    return _buffer_hash(buffers), actions, timing


def test_phase_mode_is_reproducible_legal_and_exactly_accounted() -> None:
    """Equal seeds reproduce; every live row enters exactly one action call."""
    first_hash, first_actions, timing = _phase_rollout(101)
    second_hash, second_actions, _second_timing = _phase_rollout(101)
    changed_hash, changed_actions, _changed_timing = _phase_rollout(102)

    assert (first_hash, first_actions) == (second_hash, second_actions)
    assert (changed_hash, changed_actions) != (first_hash, first_actions)
    assert timing.live_decisions == timing.action_inference_rows
    assert timing.live_decisions == len(first_actions)
    assert timing.live_decisions == (
        timing.simultaneous_live_decisions + timing.sequential_live_decisions
    )
    assert timing.action_inference_rows == sum(
        size * calls for size, calls in timing.action_batch_histogram.items()
    )
    assert timing.action_inference_calls == sum(
        timing.action_batch_histogram.values()
    )
    assert timing.available_phase_count > 0
    assert timing.available_phase_rows >= timing.available_phase_count
    assert any(size > 1 for size in timing.action_batch_histogram)
