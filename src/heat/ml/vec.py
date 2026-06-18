"""Vectorized HEAT environment construction (Sprint 6C, Part 1).

The pure-Python engine — not the tiny MLP — is the training throughput
bottleneck (~1,080 steps/s on a single env). The lever is **more envs in
parallel** across CPU cores, which :func:`make_vec_env` provides by wrapping
``HeatEnv`` in SB3's ``SubprocVecEnv`` (true multiprocess parallelism) or
``DummyVecEnv`` (in-process, for ``n_envs == 1`` / debugging / Windows-spawn
troubleshooting).

Design constraints (from the design doc and the frozen contract):

* **Picklable construction.** ``SubprocVecEnv`` on Windows uses the ``spawn``
  start method, so every per-worker callable and argument must pickle. The env
  builder is a *top-level* function (:func:`heat_env_factory`), opponent specs
  are picklable (scripted agent classes, or :class:`FrozenSnapshotAgent` which
  pickles by path and nulls its model), and tracks pickle as plain dataclasses.
  Lambdas/closures are deliberately avoided.
* **Per-worker determinism.** Each sub-env is seeded ``seed + worker_index`` and
  reset with that seed, so the whole vec env is reproducible from one base seed.
* **Per-worker shaping.** Reward shaping is configured by mutating module-level
  globals in :mod:`heat.ml.spaces` (see :func:`heat.ml.model.apply_shaping_config`).
  A spawned worker does **not** inherit the parent's mutated globals, so the
  factory re-applies the shaping weights inside each worker process.
* **No codec change.** Vectorization is purely a construction concern; the env's
  obs/action codec, masking hook, and reward are untouched. ``action_masks()``
  is collected per sub-env by ``MaskablePPO`` exactly as in the single-env case.
"""

from __future__ import annotations

from typing import Callable, Sequence

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from heat.models.track import Track
from heat.ml import spaces
from heat.ml.env import HeatEnv, OpponentSpec

#: A zero-arg callable returning a fresh ``HeatEnv`` (the SB3 env-fn convention).
EnvFn = Callable[[], HeatEnv]


def heat_env_factory(
    *,
    track: Track | None,
    num_players: int,
    opponents: OpponentSpec | Sequence[OpponentSpec] | None,
    learner_id: int,
    seed: int,
    shaping_weight: float = 0.0,
    shaping_progress_coef: float = 1.0,
    shaping_spinout_weight: float = 0.0,
    shaping_spinout_cap: float = 0.05,
    reward_mode: str = "race",
    solo_finish_bonus: float = 5.0,
    randomize_seat: bool = False,
) -> HeatEnv:
    """Build a fresh :class:`HeatEnv` for one (sub-)worker.

    Top-level and picklable so ``SubprocVecEnv`` can ship it across the process
    boundary under Windows ``spawn``. Re-applies the reward-shaping globals
    inside the worker (a spawned process does not inherit the parent's mutated
    :mod:`heat.ml.spaces` globals), then builds the env. ``seed`` is recorded on
    the env so the SB3 ``env_fn`` seeding (and our explicit reset) is
    reproducible per worker.
    """
    # A spawned worker starts from pristine module globals; re-apply shaping so
    # every sub-env computes the same reward the learner is training under.
    spaces.SHAPING_WEIGHT = shaping_weight
    spaces.SHAPING_PROGRESS_COEF = shaping_progress_coef
    spaces.SHAPING_SPINOUT_WEIGHT = shaping_spinout_weight
    spaces.SHAPING_SPINOUT_CAP = shaping_spinout_cap
    # Sprint 8C: the reward MODE + solo finish bonus are plain str/float globals,
    # re-applied per worker exactly like the shaping weights above.
    spaces.REWARD_MODE = reward_mode
    spaces.SOLO_FINISH_BONUS = solo_finish_bonus

    env = HeatEnv(
        track=track,
        num_players=num_players,
        opponents=opponents,
        learner_id=learner_id,
        randomize_seat=randomize_seat,
    )
    # Stash the per-worker seed; SB3 calls env.reset(seed=...) via VecEnv.seed,
    # but we also reset once here so a never-seeded env is still deterministic.
    env.reset(seed=seed)
    return env


class _EnvBuilder:
    """A picklable zero-arg env builder bound to per-worker kwargs.

    Used in place of a lambda/closure (which ``spawn`` cannot pickle) as the
    ``env_fn`` SB3 hands to each sub-process. Calling it builds one ``HeatEnv``
    via :func:`heat_env_factory`.
    """

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def __call__(self) -> HeatEnv:
        return heat_env_factory(**self._kwargs)


def make_vec_env(
    *,
    track: Track | None,
    num_players: int,
    opponents: OpponentSpec | Sequence[OpponentSpec] | None,
    learner_id: int,
    n_envs: int,
    vec_cls: str = "subproc",
    seed: int = 0,
    shaping_weight: float = 0.0,
    shaping_progress_coef: float = 1.0,
    shaping_spinout_weight: float = 0.0,
    shaping_spinout_cap: float = 0.05,
    reward_mode: str = "race",
    solo_finish_bonus: float = 5.0,
    randomize_seat: bool = False,
) -> VecEnv:
    """Build a vectorized HEAT env of ``n_envs`` sub-envs (§6C Part 1).

    Args:
        track / num_players / opponents / learner_id: forwarded to each sub-env's
            :class:`HeatEnv` (see :func:`heat_env_factory`).
        n_envs: number of parallel sub-envs (>= 1).
        vec_cls: ``"subproc"`` for true multiprocess parallelism (the point — to
            beat the Python engine bottleneck), or ``"dummy"`` for an in-process
            vector. ``n_envs == 1`` always uses ``DummyVecEnv`` regardless, since
            a one-process subproc pool only adds IPC overhead.
        seed: base seed; sub-env ``i`` is seeded ``seed + i`` so the whole vector
            is reproducible from this one value.
        shaping_weight / shaping_progress_coef: reward-shaping globals re-applied
            inside each worker (spawn does not inherit parent globals).
        randomize_seat: re-pick the learner seat per episode in each sub-env
            (Idea 10). Off by default. Each sub-env is independently seeded, so
            the seat draw stays reproducible from the single base ``seed``.

    Returns:
        A ``DummyVecEnv`` or ``SubprocVecEnv`` ready to pass to ``build_model`` /
        ``set_env`` (optionally further wrapped in ``VecNormalize`` by the caller).
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    env_fns: list[EnvFn] = [
        _EnvBuilder(
            track=track,
            num_players=num_players,
            opponents=opponents,
            learner_id=learner_id,
            seed=seed + i,
            shaping_weight=shaping_weight,
            shaping_progress_coef=shaping_progress_coef,
            shaping_spinout_weight=shaping_spinout_weight,
            shaping_spinout_cap=shaping_spinout_cap,
            reward_mode=reward_mode,
            solo_finish_bonus=solo_finish_bonus,
            randomize_seat=randomize_seat,
        )
        for i in range(n_envs)
    ]

    use_subproc = vec_cls == "subproc" and n_envs > 1
    if use_subproc:
        # "spawn" is the safe start method cross-platform (required on Windows).
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        venv = DummyVecEnv(env_fns)

    # Deterministic per-sub-env seeding from the single base seed.
    venv.seed(seed)
    return venv
