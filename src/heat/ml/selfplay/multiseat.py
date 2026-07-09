"""N-agent shared-policy self-play collector + trainer (Sprint A2).

A0 (:mod:`heat.ml.selfplay.ppo`) proved the custom PPO loop is correct, but it
trains a **single seat against scripted opponents** -- :class:`heat.ml.env.HeatEnv`
auto-advances every non-learner seat internally, so it structurally cannot produce
per-seat trajectories. A2 replaces that asymmetry with **true current-policy
self-play**: 2-N seats are all driven by the *same live* :class:`HeatPolicy`, each
acting on its **own per-seat partial observation**
(:func:`heat.ml.features.encode_observation`), each producing its **own trajectory
stream** for PPO.

Design (see ``docs/direction-A/A2-multiseat-selfplay.md``):

* **Drive the round generator directly.** :func:`heat.engine.driver.run_round_driver`
  yields a :class:`~heat.engine.driver.Decision` at *every* seat's decision point;
  the collector drives it and never touches ``HeatEnv``.
* **Shared legality plumbing.** Forced/degenerate decisions and flat-index decoding
  reuse the exact functions ``HeatEnv`` uses
  (:func:`heat.ml.action_codec.forced_action` /
  :func:`~heat.ml.action_codec.decode_legal_action`), so the env path and the
  self-play path can never drift apart.
* **One buffer per seat, never interleaved.** GAE assumes a single time-ordered
  stream, so each seat owns a :class:`~heat.ml.selfplay.buffer.RolloutBuffer`; GAE is
  computed per stream and the streams are concatenated only *after* ``compute_gae``
  for the (seat-agnostic) PPO update.
* **Whole-game collection.** Complete games are run until the total recorded
  transitions across policy seats reach ``n_steps`` (always finishing the in-flight
  game), so every seat stream ends on ``done=True`` and the GAE bootstrap is 0.
* **Per-seat span reward.** A seat's transition spans its decision to its next
  decision (or game end), rewarded by :func:`heat.ml.spaces.step_reward` over that
  span -- the hook A6's dense targets will reuse.

A2 is judged on *correctness of per-seat collection*, not on learning stability
(that is A5's gate). In pure self-play with the default sparse placement reward the
mean per-seat return is ~0 by construction; it is not a learning signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from heat.agents.base import BaseAgent
from heat.engine.driver import Decision, run_round_driver
from heat.engine.game import MAX_ROUNDS
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml.action_codec import (
    NO_FORCED,
    decode_legal_action,
    forced_action,
    legal_action_mask,
)
from heat.ml.env import TrackSource
from heat.ml.features import encode_observation
from heat.ml.model import resolve_device
from heat.ml.opponents import opponent_action
from heat.ml.selfplay.buffer import RolloutBuffer
from heat.ml.selfplay.policy import HeatPolicy
from heat.ml.selfplay.ppo import A0Config, ppo_update
from heat.ml.spaces import ACTION_DIM, OBS_DIM, step_reward

#: A single game yields at most 5 recordable decisions per active seat per round
#: (GEAR, CARDS, REACT, SLIPSTREAM, DISCARD), bounded by ``MAX_ROUNDS`` rounds.
#: Per-seat buffers are sized ``n_steps + this`` so the always-finish-the-game
#: rule (design §2.4) can never overflow a buffer even in a worst-case truncated
#: spin-loop game -- ``get()``/``compute_gae`` slice by ``_pos``, so the headroom
#: costs only zero-fill it never reads.
_MAX_RECORDED_PER_GAME: int = 5 * MAX_ROUNDS


@dataclass
class _Pending:
    """One in-flight, not-yet-rewarded transition for a seat.

    Held from the moment a seat acts until its *next* decision (or game end),
    when the reward over that span is known and the transition is stored. The
    ``prev_state`` snapshot (a ``clone(reseed=0)`` taken when the seat acted) is
    the ``prev`` argument to :func:`heat.ml.spaces.step_reward`.
    """

    obs: NDArray[np.float32]
    action: int
    logp: float
    value: float
    mask: NDArray[np.bool_]
    prev_state: GameState


class MultiSeatCollector:
    """Collect per-seat self-play trajectories by driving the round generator.

    Every non-scripted seat is a *policy seat* driven by the shared live policy
    and recorded into its own buffer; scripted seats (if any) are advanced but
    never recorded (the hook A5's snapshot-pool opponents and A7's eval reuse).
    Because every policy seat is the learner, seat-symmetry randomization is moot
    -- ``A0Config.randomize_seat`` is ignored here.

    Args:
        track: the track *source* -- a fixed :class:`~heat.models.track.Track`
            reused every game, or a sampler ``callable(seed) -> Track`` invoked
            once per game (a seed-derived sampler keeps "same seed -> same game").
        num_players: total seats (policy + scripted). Must be ``>= 2``.
        scripted_seats: optional ``{seat_id: BaseAgent}`` for seats driven by a
            scripted agent instead of the policy; default none (pure self-play).
    """

    def __init__(
        self,
        track: Track | TrackSource,
        num_players: int,
        *,
        scripted_seats: dict[int, BaseAgent] | None = None,
    ) -> None:
        if num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {num_players}")
        self._track_source = track
        self.num_players = num_players
        self.scripted_seats: dict[int, BaseAgent] = dict(scripted_seats or {})
        for seat in self.scripted_seats:
            if not (0 <= seat < num_players):
                raise ValueError(f"scripted seat {seat} out of range")
        #: The seats driven (and recorded) by the shared policy.
        self.policy_seats: list[int] = [
            s for s in range(num_players) if s not in self.scripted_seats
        ]
        if not self.policy_seats:
            raise ValueError("at least one seat must be a policy seat")

    # ------------------------------------------------------------------
    # Track / episode-flag helpers
    # ------------------------------------------------------------------

    def _resolve_track(self, seed: int | None) -> Track:
        """Draw the game's track from the source (a sampler is called; a fixed
        :class:`Track` is returned unchanged)."""
        src = self._track_source
        if callable(src) and not isinstance(src, Track):
            return src(seed)
        return src

    @staticmethod
    def _episode_flags(state: GameState) -> tuple[bool, bool]:
        """Return ``(terminated, truncated)`` for ``state`` -- the same rule as
        :meth:`HeatEnv._episode_flags`."""
        terminated = state.is_game_over
        truncated = (not terminated) and (state.round_num > MAX_ROUNDS)
        return terminated, truncated

    # ------------------------------------------------------------------
    # Recording seam (overridable so a test can thread a debug seat-id array)
    # ------------------------------------------------------------------

    def _store(
        self,
        buffers: list[RolloutBuffer],
        seat: int,
        pending: _Pending,
        reward: float,
        done: bool,
    ) -> None:
        """Append ``pending`` (now rewarded) to seat ``seat``'s buffer.

        The single point where a transition reaches a buffer, so a subclass can
        override this to assert stream isolation (that seat ``i``'s transitions
        only ever land in ``buffers[i]``) -- see the A2 §6 tests.
        """
        buffers[seat].add(
            obs=pending.obs,
            action=pending.action,
            logp=pending.logp,
            value=pending.value,
            reward=reward,
            done=done,
            mask=pending.mask,
        )

    def _on_game_end(
        self, state: GameState, terminated: bool, truncated: bool
    ) -> None:
        """Hook called once per game after its last transition is stored.

        A no-op in production; a seam so a test (A2 §6) can capture the terminal
        state to cross-check per-seat placement rewards / the truncation fold.
        """

    # ------------------------------------------------------------------
    # Value bootstrap (truncation fold, §4.6)
    # ------------------------------------------------------------------

    @staticmethod
    def _seat_value(
        policy: HeatPolicy,
        obs: NDArray[np.float32],
        device: torch.device,
    ) -> float:
        """Value ``V(s)`` the policy assigns to ``obs`` (mask-independent).

        Used for the §4.6 truncation bootstrap: an all-legal mask is passed since
        the value head reads only the trunk latent, so the mask never affects it.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.ones((1, ACTION_DIM), dtype=torch.bool, device=device)
        _action, _logp, value_t, _entropy = policy.act(obs_t, mask_t)
        return float(value_t.item())

    # ------------------------------------------------------------------
    # Public collection
    # ------------------------------------------------------------------

    def collect(
        self,
        policy: HeatPolicy,
        n_steps: int,
        device: torch.device,
        rng: np.random.Generator,
        *,
        gamma: float = A0Config.gamma,
    ) -> tuple[list[RolloutBuffer], list[float]]:
        """Run whole games until ``sum(len(buf) for policy seats) >= n_steps``.

        Games are always run to completion (design §2.4), so every seat stream
        ends on ``done=True`` and no obs/mask carries between rollouts. GAE is
        *not* computed here (the caller does that per stream before the update).

        Args:
            policy: the shared acting policy.
            n_steps: target total recorded transitions across policy seats.
            device: torch device for the per-step act tensors.
            rng: numpy RNG for per-game seeds (same seed -> same game).
            gamma: discount for the §4.6 truncation value-bootstrap fold. Defaults
                to the :class:`A0Config` discount (kept as a keyword so
                ``collect`` still matches the design's positional signature while
                enabling the fold the design mandates).

        Returns:
            ``(buffers, episode_returns)`` -- one :class:`RolloutBuffer` per seat
            (GAE not yet computed; scripted seats' buffers stay empty) and the
            flat list of per-seat-per-game returns (one entry per policy seat that
            recorded at least one transition in a game).
        """
        capacity = n_steps + _MAX_RECORDED_PER_GAME
        buffers = [
            RolloutBuffer(
                capacity=capacity,
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                device=device,
            )
            for _ in range(self.num_players)
        ]
        episode_returns: list[float] = []

        recorded = 0
        while recorded < n_steps:
            episode_returns.extend(
                self._play_one_game(policy, buffers, device, rng, gamma)
            )
            recorded = sum(len(buffers[s]) for s in self.policy_seats)

        return buffers, episode_returns

    def _play_one_game(
        self,
        policy: HeatPolicy,
        buffers: list[RolloutBuffer],
        device: torch.device,
        rng: np.random.Generator,
        gamma: float,
    ) -> list[float]:
        """Play one complete self-play game, recording every policy seat's stream.

        Mirrors :meth:`HeatEnv.reset` / :meth:`HeatEnv._advance_to_learner` but for
        *all* policy seats at once. Returns the per-seat game returns (only for
        seats that recorded at least one transition).
        """
        seed = int(rng.integers(0, 2**31 - 1))
        track = self._resolve_track(seed)
        state = GameState.create(track, self.num_players, seed=seed)
        # Mirror Game.__init__: the race has started, everyone is on lap 1.
        for player in state.players:
            player.lap = 1

        gen = run_round_driver(state)
        send_value: object = None
        pending: dict[int, _Pending] = {}
        seat_return: dict[int, float] = {s: 0.0 for s in self.policy_seats}
        seat_recorded: dict[int, int] = {s: 0 for s in self.policy_seats}

        while True:
            terminated, truncated = self._episode_flags(state)
            if terminated or truncated:
                break
            try:
                decision: Decision = gen.send(send_value)
            except StopIteration:
                # One round finished (its counter was advanced inside the
                # generator); start the next round's generator unless the game is
                # now over. Mirrors HeatEnv._advance_to_learner.
                terminated, truncated = self._episode_flags(state)
                if terminated or truncated:
                    break
                gen = run_round_driver(state)
                send_value = None
                continue

            seat = decision.player_id

            # Scripted seat: advance, never record.
            if seat in self.scripted_seats:
                send_value = opponent_action(
                    self.scripted_seats[seat], decision, state
                )
                continue

            # Degenerate/forced policy decision (<= 1 legal action): auto-resolve,
            # never surface it as an RL step -- exactly as HeatEnv does.
            forced = forced_action(decision, state)
            if forced is not NO_FORCED:
                send_value = forced
                continue

            # A real policy decision for this seat. First close out this seat's
            # PENDING transition (its span ran to here), done=False.
            if seat in pending:
                prev = pending.pop(seat)
                reward = step_reward(
                    prev.prev_state, state, seat, done=False, terminated=False
                )
                self._store(buffers, seat, prev, reward, done=False)
                seat_return[seat] += reward
                seat_recorded[seat] += 1

            obs = encode_observation(state, seat, decision)
            mask = legal_action_mask(decision, state)
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            action_t, logp_t, value_t, _entropy = policy.act(obs_t, mask_t)
            action = int(action_t.item())

            pending[seat] = _Pending(
                obs=obs,
                action=action,
                logp=float(logp_t.item()),
                value=float(value_t.item()),
                mask=mask,
                # clone(reseed=0): snapshot without perturbing the live game RNG.
                prev_state=state.clone(reseed=0),
            )
            send_value = decode_legal_action(decision, state, action)

        # Game ended: complete every pending transition with done=True.
        terminated, truncated = self._episode_flags(state)
        for seat, prev in pending.items():
            reward = step_reward(
                prev.prev_state, state, seat, done=True, terminated=terminated
            )
            # truncation != termination (A2 design §4.6): a time-limit cutoff must
            # not zero the value bootstrap the way done=True does in GAE. Fold
            # gamma * V(s_next) into the final reward (s_next = the post-game obs
            # for this seat with no pending decision), keeping done=True.
            if truncated and not terminated:
                s_next = encode_observation(state, seat, None)
                reward += gamma * self._seat_value(policy, s_next, device)
            self._store(buffers, seat, prev, reward, done=True)
            seat_return[seat] += reward
            seat_recorded[seat] += 1

        self._on_game_end(state, terminated, truncated)
        return [
            seat_return[s] for s in self.policy_seats if seat_recorded[s] > 0
        ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _concat_batches(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate per-seat ``get()`` dicts along dim 0 into one PPO batch.

    Advantage normalization then happens over the full concatenated batch inside
    :func:`ppo_update`, which is correct for a shared policy (all seats' data
    train the one network).
    """
    keys = batches[0].keys()
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}


def train_multiseat(
    config: A0Config,
    *,
    track: Track | TrackSource | None = None,
    scripted_seats: dict[int, BaseAgent] | None = None,
    on_iteration: object = None,
) -> HeatPolicy:
    """Run the A2 shared-policy self-play PPO loop and return the trained policy.

    Mirrors :func:`heat.ml.selfplay.ppo.train` (same device/seed handling, same
    ``on_iteration`` info dict + ``n_episodes``) but collects per-seat streams via
    :class:`MultiSeatCollector`: each iteration collects ``n_steps`` transitions,
    computes GAE per seat stream (bootstrap 0 -- whole-game collection), then runs
    one :func:`~heat.ml.selfplay.ppo.ppo_update` over the concatenated batch.

    Log honestly: in pure self-play with the default sparse placement reward the
    mean per-seat return is ~0 by construction (roughly zero-sum); it is *not* a
    learning signal (the A2 smoke gate checks finiteness + mechanics, not skill).

    Args:
        config: hyperparameters (``A0Config`` as-is; ``randomize_seat`` is ignored
            -- every policy seat is already the learner).
        track: track source for the games; defaults to the Tiny-Heat bed (A2's
            proving ground) when omitted.
        scripted_seats: optional ``{seat_id: BaseAgent}`` opponents (advanced but
            not recorded); default none (pure self-play).
        on_iteration: optional callback ``fn(iteration, info)`` invoked after each
            PPO update with the loss terms, ``mean_episode_return``, ``n_episodes``
            and ``n_recorded`` (transitions collected this iteration). Typed
            ``object`` so callers pass any callable without import gymnastics.

    Returns:
        The trained :class:`HeatPolicy`.
    """
    device = torch.device(resolve_device(config.device))

    if track is None:
        # Lazily import the Tiny-Heat bed only when needed (avoids coupling the
        # trainer to the tiny-track helpers at import time).
        from heat.ml.selfplay.tiny_heat import tiny_heat_track

        track = tiny_heat_track()

    collector = MultiSeatCollector(
        track, config.num_players, scripted_seats=scripted_seats
    )

    policy = HeatPolicy(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=config.hidden_sizes
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    if config.seed is not None:
        torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    n_iterations = max(1, config.total_timesteps // config.n_steps)
    for iteration in range(n_iterations):
        buffers, episode_returns = collector.collect(
            policy, config.n_steps, device, rng, gamma=config.gamma
        )

        batches: list[dict[str, torch.Tensor]] = []
        n_recorded = 0
        for buf in buffers:
            if len(buf) == 0:
                continue
            buf.compute_gae(0.0, config.gamma, config.gae_lambda)
            batches.append(buf.get())
            n_recorded += len(buf)

        if not batches:  # pragma: no cover - defensive (a game always records)
            continue

        batch = _concat_batches(batches)
        losses = ppo_update(policy, optimizer, batch, config)

        if callable(on_iteration):
            mean_return = (
                float(np.mean(episode_returns)) if episode_returns else float("nan")
            )
            on_iteration(
                iteration,
                {
                    **losses,
                    "mean_episode_return": mean_return,
                    "n_episodes": len(episode_returns),
                    "n_recorded": n_recorded,
                },
            )

    return policy
