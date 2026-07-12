"""Anti-collapse self-play recipe (Sprint A5).

The A2 probe (``docs/direction-A/A2-findings.md`` §2) showed that *naive*
current-policy self-play already does not collapse on Tiny-Heat seed 0 -- it
climbs 41% -> ~85-92% vs the weak heuristic with entropy holding near ~0.7. A5's
job is therefore to *harden* that into a reproducible recipe with explicit
guards, and to prove it across seeds:

1. **An entropy floor** -- a *parachute* (:class:`EntropyController`) that does
   nothing on a healthy run and only multiplies ``ent_coef`` up if mean rollout
   entropy sinks toward collapse, decaying back to base once it recovers.
2. **A recent-snapshot opponent pool** (:class:`~heat.ml.selfplay.snapshots.SnapshotPool`)
   that the collection occasionally draws an opponent from, damping the self-
   chasing win-rate oscillation.
3. **A Stage-1 validation check** (the 8C lesson): assert the run is on-track in
   its first minutes, and abort loudly (:class:`Stage1ValidationError`) rather
   than let a broken run burn a long budget.

:func:`train_selfplay_a5` is the recipe loop; it mirrors
:func:`heat.ml.selfplay.multiseat.train_multiseat` and returns the trained policy
plus the list of periodic eval records the CLI / gate consume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.models.track import Track
from heat.ml.env import TrackSource
from heat.ml.model import resolve_device
from heat.ml.selfplay.eval_harness import _EvalCollector
from heat.ml.selfplay.multiseat import MultiSeatCollector
from heat.ml.selfplay.policy import PPOPolicy, build_policy
from heat.ml.selfplay.ppo import A0Config, ppo_update
from heat.ml.selfplay.snapshots import SnapshotPool
from heat.ml.spaces import _placement_reward


@dataclass
class A5Config(A0Config):
    """Hyperparameters for the A5 anti-collapse self-play recipe.

    Subclasses :class:`A0Config` (all A0 fields keep their meaning) and adds the
    entropy-floor, snapshot-pool, eval, and Stage-1 knobs. Defaults are the
    design §4.2 recipe defaults.
    """

    # --- entropy floor (parachute) ---
    #: Controller engages below this mean rollout entropy.
    entropy_floor: float = 0.40
    #: Multiplicative controller step up (below floor) / down (recovered).
    ent_scale_up: float = 1.5
    ent_scale_down: float = 0.98
    #: Controller ceiling; base ``ent_coef`` is the floor it decays back to.
    ent_coef_max: float = 0.10

    # --- snapshot pool ---
    #: Iterations between pool pushes (a push also happens at iteration 1).
    snapshot_every: int = 10
    #: Recent snapshots retained.
    pool_capacity: int = 5
    #: Per-iteration probability the collection uses one pool opponent.
    pool_prob: float = 0.5

    # --- eval ---
    #: Iterations between eval checkpoints.
    eval_every: int = 10
    #: Eval games per checkpoint (~half per seat order).
    eval_games: int = 40

    # --- dense terminal-margin target (Sprint A6) ---
    #: Coefficient on :func:`heat.ml.spaces.terminal_margin`, added to each policy
    #: seat's game-end reward during *training* collection (never during eval).
    #: ``0.0`` == off (pre-A6 behavior); the A6 gate decides whether this default
    #: flips to 0.5.
    margin_coef: float = 0.0

    # --- Stage-1 validation ---
    #: Iteration of the Stage-1 check (~30k steps at n_steps=2048).
    stage1_iter: int = 15
    #: Minimum vs-weak winrate required at the Stage-1 check.
    stage1_min_winrate: float = 0.55
    #: Whether the Stage-1 check runs at all.
    stage1_enabled: bool = True


class Stage1ValidationError(RuntimeError):
    """Raised when the Stage-1 check fails -- the run is aborted early.

    Carries a ``diagnostics`` dict (iteration, winrate, entropy, thresholds) so
    the caller can log exactly why the run was judged broken (the 8C lesson:
    fail loud and early, never launch a long run unvalidated).
    """

    def __init__(self, message: str, diagnostics: dict[str, float]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class EntropyController:
    """Entropy-floor parachute for ``ent_coef`` (design §4.2 / §2.3).

    Holds :attr:`current_ent_coef` (initialised to ``config.ent_coef``). After
    each rollout, :meth:`update` is called with the mean rollout entropy:

    * ``entropy < floor``          -> raise the coef by ``ent_scale_up`` (capped
      at ``ent_coef_max``); mark :attr:`engaged` for this iteration.
    * ``entropy > floor * 1.2``    -> decay the coef by ``ent_scale_down`` toward
      but never below the base ``ent_coef``.
    * otherwise (in the ``[floor, floor*1.2]`` band) -> leave the coef unchanged.

    A *healthy* run never drops below the floor, so the controller never engages
    and the coef stays at base -- the A5 G3 gate asserts exactly this.
    """

    def __init__(self, config: A5Config) -> None:
        self._base = config.ent_coef
        self._floor = config.entropy_floor
        self._scale_up = config.ent_scale_up
        self._scale_down = config.ent_scale_down
        self._coef_max = config.ent_coef_max
        self.current_ent_coef = config.ent_coef
        #: Whether the most recent :meth:`update` raised the coef (parachute
        #: deployed on that iteration).
        self.engaged = False

    def update(self, mean_entropy: float) -> float:
        """Adjust :attr:`current_ent_coef` from ``mean_entropy`` and return it."""
        if mean_entropy < self._floor:
            self.current_ent_coef = min(
                self.current_ent_coef * self._scale_up, self._coef_max
            )
            self.engaged = True
        elif mean_entropy > self._floor * 1.2:
            self.current_ent_coef = max(
                self.current_ent_coef * self._scale_down, self._base
            )
            self.engaged = False
        else:
            self.engaged = False
        return self.current_ent_coef


# ---------------------------------------------------------------------------
# Eval (probe-style vs-opponent winrate, both seat orders)
# ---------------------------------------------------------------------------


def _evaluate_vs(
    policy: PPOPolicy,
    opponent: BaseAgent,
    track: Track | TrackSource,
    num_players: int,
    device: torch.device,
    n_games: int,
    base_seed: int,
) -> tuple[float, float]:
    """Play ``n_games`` of ``policy`` vs ``opponent`` and return
    ``(winrate, mean_return)``.

    The policy seat rotates through the field game-to-game (so both seat orders
    are covered evenly); every other seat is driven by the same ``opponent``
    instance. A game is a *win* when the policy seat's placement reward is
    positive (it finished ahead of the field); ``mean_return`` averages that
    signed placement reward. The policy still *samples* (``act``), matching the
    A2 probe's yardstick.
    """
    wins = 0
    total_return = 0.0
    for g in range(n_games):
        policy_seat = g % num_players
        scripted: dict[int, BaseAgent] = {
            s: opponent for s in range(num_players) if s != policy_seat
        }
        collector = _EvalCollector(track, num_players, scripted_seats=scripted)
        rng = np.random.default_rng(base_seed + g)
        # n_steps=1 -> exactly one complete game (finish-the-in-flight-game rule).
        collector.collect(policy, 1, device, rng, gamma=1.0)
        state = collector.last_state
        reward = _placement_reward(state, policy_seat)  # type: ignore[arg-type]
        total_return += reward
        if reward > 0.0:
            wins += 1
    return wins / n_games, total_return / n_games


# ---------------------------------------------------------------------------
# The recipe loop
# ---------------------------------------------------------------------------


def train_selfplay_a5(
    config: A5Config,
    *,
    track: Track | TrackSource | None = None,
    on_iteration: object = None,
) -> tuple[PPOPolicy, list[dict[str, float]]]:
    """Run the A5 anti-collapse self-play loop.

    Each iteration collects one rollout -- pure self-play, or (with probability
    ``pool_prob``, once the pool is non-empty) self-play with a single pool
    opponent occupying one seat -- computes per-seat GAE, and runs one
    :func:`~heat.ml.selfplay.ppo.ppo_update` whose ``ent_coef`` is the entropy
    controller's current value (passed via a per-iteration
    :func:`dataclasses.replace`, so the caller's ``config`` is never mutated).
    Snapshots are pushed to the pool, periodic evals are recorded, and the
    Stage-1 check aborts a broken run early.

    Args:
        config: the :class:`A5Config` recipe hyperparameters.
        track: track source for the games; defaults to the Tiny-Heat bed.
        on_iteration: optional callback ``fn(iteration, info)`` invoked after each
            PPO update with the loss terms plus ``entropy``, ``ent_coef``,
            ``engaged``, ``n_recorded`` and ``steps``. Typed ``object`` so callers
            pass any callable without import gymnastics.

    Returns:
        ``(policy, eval_records)`` -- the trained policy and the list of periodic
        eval-record dicts (each carrying vs-weak / vs-oldest winrate + return,
        current entropy / ent_coef / engaged flag, the running min entropy and
        ever-engaged flag, iteration, and cumulative steps).

    Raises:
        Stage1ValidationError: if the Stage-1 check fails at ``stage1_iter``.
    """
    device = torch.device(resolve_device(config.device))

    if track is None:
        from heat.ml.selfplay.tiny_heat import tiny_heat_track

        track = tiny_heat_track()

    policy = build_policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    if config.seed is not None:
        torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    # A separate RNG stream for eval game seeds keeps eval reproducible and
    # independent of how many pool draws the training stream consumed.
    eval_base = int.from_bytes(b"a5eval", "little") + (config.seed or 0)

    controller = EntropyController(config)
    pool = SnapshotPool(capacity=config.pool_capacity)
    weak_opponent = HeuristicAgent()

    eval_records: list[dict[str, float]] = []
    total_steps = 0
    min_entropy = float("inf")
    ever_engaged = False

    n_iterations = max(1, config.total_timesteps // config.n_steps)
    for iteration in range(1, n_iterations + 1):
        # --- opponent plan for this iteration (per-iteration, not per-game) ---
        scripted_seats: dict[int, BaseAgent] | None = None
        if len(pool) > 0 and rng.random() < config.pool_prob:
            seat = int(rng.integers(config.num_players))
            scripted_seats = {seat: pool.sample(rng)}
        # A6: the dense terminal-margin term is applied only to TRAINING
        # collection. Eval (`_evaluate_vs` -> `_EvalCollector`) never passes
        # `margin_coef`, and scores off the terminal state via `_placement_reward`
        # directly -- so the vs-weak / vs-oldest yardstick is identical across
        # arms regardless of `margin_coef`.
        collector = MultiSeatCollector(
            track, config.num_players, scripted_seats=scripted_seats,
            margin_coef=config.margin_coef,
        )

        # --- collect one rollout, build the concatenated PPO batch ---
        buffers, _episode_returns = collector.collect(
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
        batch = {
            k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]
        }

        # --- mean rollout entropy (of the policy that generated the batch) ---
        with torch.no_grad():
            _logp, _value, ent = policy.evaluate(
                batch["obs"], batch["actions"], batch["masks"]
            )
            mean_entropy = float(ent.mean().item())
        min_entropy = min(min_entropy, mean_entropy)

        # --- controller adjusts ent_coef, then the PPO update consumes it ---
        controller.update(mean_entropy)
        ever_engaged = ever_engaged or controller.engaged
        iter_config = replace(config, ent_coef=controller.current_ent_coef)
        losses = ppo_update(policy, optimizer, batch, iter_config)
        total_steps += n_recorded

        # --- snapshot push (iter 1, then every snapshot_every) ---
        if iteration == 1 or iteration % config.snapshot_every == 0:
            pool.push(policy, f"iter{iteration}")

        # --- Stage-1 validation ---
        if config.stage1_enabled and iteration == config.stage1_iter:
            s1_winrate, _s1_return = _evaluate_vs(
                policy, weak_opponent, track, config.num_players, device,
                config.eval_games, eval_base,
            )
            collapsed = mean_entropy < 0.5 * config.entropy_floor
            if s1_winrate < config.stage1_min_winrate or collapsed:
                diagnostics = {
                    "iteration": float(iteration),
                    "winrate_vs_weak": s1_winrate,
                    "entropy": mean_entropy,
                    "min_winrate": config.stage1_min_winrate,
                    "entropy_floor": config.entropy_floor,
                }
                raise Stage1ValidationError(
                    f"Stage-1 check failed at iter {iteration}: "
                    f"winrate {s1_winrate:.3f} (min {config.stage1_min_winrate}), "
                    f"entropy {mean_entropy:.3f} "
                    f"(collapse<{0.5 * config.entropy_floor:.3f})",
                    diagnostics,
                )

        # --- periodic eval record ---
        if iteration % config.eval_every == 0:
            wr_weak, ret_weak = _evaluate_vs(
                policy, weak_opponent, track, config.num_players, device,
                config.eval_games, eval_base + 1000 * iteration,
            )
            if len(pool) > 0:
                wr_old, ret_old = _evaluate_vs(
                    policy, pool.oldest(), track, config.num_players, device,
                    config.eval_games, eval_base + 1000 * iteration + 500,
                )
            else:  # pragma: no cover - pool is pushed at iter 1
                wr_old, ret_old = float("nan"), float("nan")
            record: dict[str, float] = {
                "iteration": float(iteration),
                "steps": float(total_steps),
                "entropy": mean_entropy,
                "ent_coef": controller.current_ent_coef,
                "engaged": float(controller.engaged),
                "min_entropy": min_entropy,
                "ever_engaged": float(ever_engaged),
                "winrate_vs_weak": wr_weak,
                "return_vs_weak": ret_weak,
                "winrate_vs_oldest": wr_old,
                "return_vs_oldest": ret_old,
            }
            eval_records.append(record)

        # --- per-iteration callback (CLI live print) ---
        if callable(on_iteration):
            on_iteration(
                iteration,
                {
                    **losses,
                    "entropy": mean_entropy,
                    "ent_coef": controller.current_ent_coef,
                    "engaged": float(controller.engaged),
                    "n_recorded": float(n_recorded),
                    "steps": float(total_steps),
                },
            )

    return policy, eval_records
