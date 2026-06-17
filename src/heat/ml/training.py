"""PPO training + self-play curriculum for the HEAT RL layer (Sprint 5c).

This module wires :func:`heat.ml.model.build_model` to the :class:`heat.ml.env.HeatEnv`
and runs a self-play curriculum:

* **Phase 1** -- train the learner against the scripted pool
  (``HeuristicAgent`` + some ``RandomAgent``).
* **Phase 2** -- periodically *freeze* the current policy to a checkpoint
  (``model.save`` + sidecar), and mix those frozen snapshots into the opponent
  pool. Snapshots play through the env's opponent adapter
  (:func:`heat.ml.opponents.opponent_action`) via :class:`FrozenSnapshotAgent`,
  which loads the model **from a path** so the opponent spec stays light and
  picklable (§6.5).

Checkpoint format (§3.4)
------------------------
:func:`save_checkpoint` writes:

* ``<path>.zip`` -- SB3 native ``MaskablePPO.save`` archive.
* ``<path>.meta.json`` -- sidecar with
  ``{obs_dim, action_dim, codec_version, track_name, num_players}`` pulled from
  :mod:`heat.ml.spaces`. 5d's ``MLAgent`` asserts these on load.

This is a self-play training driver, not a CLI -- ``scripts/`` and the eval CLI
are 5d. :func:`smoke_train` is the tiny entry point the slow smoke test uses.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from heat.agents.base import BaseAgent
from heat.agents.heuristic_agent import HeuristicAgent
from heat.agents.random_agent import RandomAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml import spaces
from heat.ml.action_codec import decode_action, legal_action_mask
from heat.ml.env import HeatEnv, _default_track
from heat.ml.features import encode_observation
from heat.ml.league import League, LeagueEntry
from heat.ml.model import (
    PPOConfig,
    apply_shaping_config,
    build_model,
    resolve_device,
)
from heat.ml.vec import make_vec_env

#: Suffix for the SB3 archive and the JSON sidecar.
_META_SUFFIX = ".meta.json"

#: Suffix for the (optional) ``VecNormalize`` running-statistics sidecar (§2.6).
_VECNORM_SUFFIX = ".vecnorm.pkl"


# ---------------------------------------------------------------------------
# Checkpoint save/load (§3.4)
# ---------------------------------------------------------------------------


def _strip_zip(path: str) -> str:
    return path[:-4] if path.endswith(".zip") else path


def meta_path_for(path: str) -> str:
    """Return the sidecar metadata path for a checkpoint ``path`` (the SB3 zip).

    SB3 appends ``.zip`` to ``path`` if missing; the sidecar sits next to the
    archive as ``<path>.meta.json`` (mirroring the un-suffixed ``model.save``
    argument so 5d can locate it from the same base path).
    """
    return _strip_zip(path) + _META_SUFFIX


def vecnorm_path_for(path: str) -> str:
    """Return the ``VecNormalize`` stats sidecar path for a checkpoint (§2.6).

    Mirrors :func:`meta_path_for`: ``<path>.vecnorm.pkl`` next to the SB3 archive.
    The file is only written when the run used reward/obs normalization; its
    absence means "no normalization" (back-compat with Sprint-5 checkpoints).
    """
    return _strip_zip(path) + _VECNORM_SUFFIX


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or ``None`` outside a git checkout.

    Non-fatal: an exported / non-git tree records ``None`` rather than failing
    the save (§3.3 open question).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _dataclass_to_dict(obj) -> dict | None:
    """Serialize a dataclass instance to a plain JSON-able dict, or ``None``."""
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return None


def save_checkpoint(
    model: MaskablePPO,
    path: str,
    *,
    track_name: str,
    num_players: int,
    seed: int | None = None,
    ppo_config: PPOConfig | None = None,
    curriculum_config: "CurriculumConfig | None" = None,
    track_config: dict | None = None,
    normalize: dict | None = None,
    vec_env: VecEnv | None = None,
) -> str:
    """Save ``model`` to ``path`` (SB3 zip) plus the ``.meta.json`` sidecar.

    The sidecar always carries the frozen contract tripwire fields
    (``obs_dim``/``action_dim``/``codec_version``, §3.4) plus the existing
    ``track_name``/``num_players``. Sprint 6C adds *additive* reproducibility
    metadata (§3.3): ``git_sha``, ``seed``, ``ppo_config``, ``curriculum_config``,
    ``track_config`` and a ``normalize`` block. These extra fields are ignored by
    the loader's contract validation (``MLAgent._validate_meta``), so older
    sidecars lacking them still load.

    If ``vec_env`` is a ``VecNormalize`` wrapper, its running statistics are
    saved alongside as ``<path>.vecnorm.pkl`` (:func:`vecnorm_path_for`) so the
    normalization can be reproduced at inference. A non-normalized run writes no
    such file.

    Returns the sidecar (``.meta.json``) path.
    """
    model.save(path)

    meta: dict = {
        "obs_dim": spaces.OBS_DIM,
        "action_dim": spaces.ACTION_DIM,
        "codec_version": spaces.CODEC_VERSION,
        "track_name": track_name,
        "num_players": num_players,
        # --- additive repro metadata (§3.3); ignored by the contract tripwire ---
        "git_sha": _git_sha(),
        "seed": seed,
        "ppo_config": _dataclass_to_dict(ppo_config),
        "curriculum_config": _dataclass_to_dict(curriculum_config),
        "track_config": track_config,
        "normalize": normalize,
    }
    sidecar = meta_path_for(path)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # Persist VecNormalize running stats next to the checkpoint (§2.6).
    if isinstance(vec_env, VecNormalize):
        vec_env.save(vecnorm_path_for(path))

    return sidecar


def load_meta(path: str) -> dict:
    """Load and return the checkpoint sidecar metadata for ``path``."""
    with open(meta_path_for(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Frozen self-play snapshot opponent (§5c, §6.5)
# ---------------------------------------------------------------------------


class FrozenSnapshotAgent(BaseAgent):
    """A :class:`BaseAgent` that acts using a frozen SB3 checkpoint on disk.

    Built for self-play (§5c Phase 2): the env's opponent adapter calls the five
    ``choose_*`` pull-methods, which receive *already-enumerated* legal options
    (not a ``Decision``). This agent reconstructs the matching ``Decision``,
    encodes the observation, predicts a flat action under the legal mask, and
    decodes it back to the concrete engine action -- always legal.

    The model is loaded **lazily from ``model_path``** (not held as a constructor
    arg) so the opponent spec stays small and picklable across process
    boundaries (§6.5). The loaded model is cached on the instance.
    """

    def __init__(
        self,
        model_path: str,
        *,
        deterministic: bool = True,
        name: str = "FrozenSnapshot",
    ) -> None:
        super().__init__(name=name)
        self.model_path = model_path
        self.deterministic = deterministic
        self._model: MaskablePPO | None = None

    # -- lazy model access (keeps the spec picklable) --
    def _get_model(self) -> MaskablePPO:
        if self._model is None:
            # Snapshots always load on CPU even when the learner trains on GPU:
            # opponent inference is tiny and keeping snapshots off the GPU avoids
            # contending for device memory (§6C Part 1). SB3 maps tensors on load,
            # so a GPU-trained checkpoint loads fine on CPU.
            self._model = MaskablePPO.load(self.model_path, device="cpu")
        return self._model

    def __getstate__(self) -> dict:
        # Never pickle the heavy SB3 model; reload from path in the worker.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    # -- shared predict path --
    def _predict_flat(self, decision: Decision, state: GameState) -> int:
        obs = encode_observation(state, decision.player_id, decision)
        mask = legal_action_mask(decision, state)
        model = self._get_model()
        action, _ = model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        return int(np.asarray(action).reshape(-1)[0])

    def _choose(self, decision: Decision, state: GameState) -> object:
        flat = self._predict_flat(decision, state)
        return decode_action(decision, flat, state)

    # -- BaseAgent pull-methods: rebuild a Decision, then predict+decode --
    def choose_gear(
        self, state: GameState, player_id: int, legal_gears: list[tuple[int, int]]
    ) -> tuple[int, int]:
        decision = Decision(DecisionKind.GEAR, player_id, legal_gears)
        return self._choose(decision, state)  # type: ignore[return-value]

    def choose_cards(
        self,
        state: GameState,
        player_id: int,
        legal_plays: list[tuple[Card, ...]],
    ) -> tuple[Card, ...]:
        decision = Decision(DecisionKind.CARDS, player_id, legal_plays)
        return self._choose(decision, state)  # type: ignore[return-value]

    def choose_react(
        self,
        state: GameState,
        player_id: int,
        max_cooldown: int,
        can_boost: bool,
        has_adrenaline: bool,
    ) -> ReactDecision:
        opts = rules.ReactOptions(
            max_cooldown=max_cooldown,
            can_boost=can_boost,
            has_adrenaline=has_adrenaline,
        )
        decision = Decision(DecisionKind.REACT, player_id, opts)
        return self._choose(decision, state)  # type: ignore[return-value]

    def choose_slipstream(self, state: GameState, player_id: int) -> bool:
        decision = Decision(DecisionKind.SLIPSTREAM, player_id, True)
        return self._choose(decision, state)  # type: ignore[return-value]

    def choose_discard(
        self, state: GameState, player_id: int, discardable: list[Card]
    ) -> list[Card]:
        decision = Decision(DecisionKind.DISCARD, player_id, discardable)
        return self._choose(decision, state)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Self-play curriculum + stability settings (§5c, §6C Part 2).

    ``phase1_steps`` are trained vs the scripted pool; the remaining
    ``total_timesteps - phase1_steps`` are Phase 2, where after every
    ``snapshot_every`` timesteps the current policy is frozen to a checkpoint
    and added to the opponent pool (capped at ``max_snapshots``, FIFO).

    Sprint 6C adds the stability machinery that stops Phase 2 from collapsing
    and *losing* the good Phase-1 model:

    * **Eval-gated best-checkpoint preservation** (§2.1) — the canonical
      ``run_name`` checkpoint only ever holds the best-evaluated policy; the
      collapsed final model goes to a separate ``run_name_final`` path.
    * **Stochastic, multi-snapshot opponents** (§2.2).
    * **Shaping schedule** (§2.3) and **opponent-mix ramp** (§2.4).
    * **Critic warm-up + LR/entropy schedule** after the env swap (§2.5).
    * **Return/reward normalization** via ``VecNormalize`` (§2.6).
    """

    total_timesteps: int = 200_000
    phase1_steps: int = 100_000
    snapshot_every: int = 50_000
    max_snapshots: int = 3
    #: Max fraction of opponent seats drawn from the snapshot pool (the ramp
    #: target — Phase 2 ramps from ~0 up to this; §2.4).
    snapshot_mix: float = 0.5
    #: Directory checkpoints (and snapshots) are written to.
    checkpoint_dir: str = "checkpoints"
    #: Base name for the (best) checkpoint. The collapsed final model is saved to
    #: ``f"{run_name}_final"``.
    run_name: str = "heat_ppo"

    # --- §2.2 stochastic snapshots ---
    #: Whether Phase-2 snapshot opponents sample (True) vs argmax (False). False
    #: was the Sprint-5 default that produced zero-variance, identical opponents.
    snapshot_deterministic: bool = False

    # --- §2.3 shaping schedule ---
    #: Dense shaping weight applied during the Phase-1->2 transition / early
    #: Phase 2 to keep a learning signal while everyone is losing. Annealed
    #: linearly to ``shaping_weight_end`` across Phase 2.
    shaping_weight_start: float = 0.0
    shaping_weight_end: float = 0.0

    # --- §2.5 critic warm-up + LR/entropy schedule (applied right after the
    #     env swap into Phase 2) ---
    #: Multiplier on the base learning rate for the first Phase-2 chunk(s).
    phase2_lr_scale: float = 1.0
    #: Multiplier on the base entropy coefficient early in Phase 2 (keep
    #: exploration up to avoid entropy collapse locking in all-loss).
    phase2_ent_scale: float = 1.0
    #: Number of leading Phase-2 chunks the warm-up scaling applies to.
    warmup_chunks: int = 1

    # --- §2.6 return / reward normalization ---
    #: Wrap the vec env in ``VecNormalize`` (norm_reward) to keep critic targets
    #: scale-invariant across the Phase-1->2 return-distribution shift.
    normalize_reward: bool = False
    normalize_obs: bool = False
    #: Reward-clipping range for ``VecNormalize`` (only when normalize_reward).
    clip_reward: float = 10.0

    # --- §2.1 eval gate ---
    #: Number of games the inline win-rate gate plays per evaluation. Kept small
    #: (the gate runs every snapshot_every); the full 6B round-robin runs after.
    gate_games: int = 20
    #: If True, reload the best checkpoint into the live model whenever the gate
    #: score regresses badly, so a collapsing Phase 2 can continue from strength.
    reload_best_on_regression: bool = False
    #: Relative drop below ``best_score`` that triggers a reload (when enabled).
    regression_tol: float = 0.5

    # --- §6D opponent league + PFSP ---
    #: Use the :class:`heat.ml.league.League` (retention + PFSP sampling) to fill
    #: the Phase-2 snapshot seats instead of the FIFO ``_mixed_opponent_pool``
    #: pool. When ``False`` (default) Phase 2 keeps the 6C FIFO behavior exactly,
    #: so 6D is strictly opt-in and 6C runs are byte-for-byte unchanged.
    use_league: bool = False
    #: League capacity (entries retained; the best anchor counts but is never
    #: evicted). Modest by design to bound disk + sampling cost (was FIFO 3).
    league_capacity: int = 8
    #: PFSP weighting family: ``"even"``/``"variance"`` (``wr*(1-wr)``, the stable
    #: default -- close matchups) or ``"hard"`` (``(1-wr)^p`` -- current losses).
    league_pfsp_mode: str = "even"
    #: Exponent ``p`` in the ``"hard"`` PFSP variant.
    league_pfsp_exponent: float = 2.0
    #: Per-entry sample-probability clamps (after normalization); guards against a
    #: single opponent dominating (``p_max``) or being starved (``p_min``).
    league_p_min: float = 0.0
    league_p_max: float = 1.0
    #: Games vs an entry before its observed win-rate is trusted over the neutral
    #: 0.5 prior (so a brand-new snapshot is sampled enough to be estimated).
    league_min_games: int = 1
    #: Retention value blend: strength (gate_score) vs diversity (recency band).
    league_strength_weight: float = 1.0
    league_diversity_weight: float = 1.0
    #: Games the inline league bookkeeping eval plays per sampled opponent each
    #: chunk to estimate ``learner_win_rate_vs_i``. Kept SMALL -- this is
    #: bookkeeping for the PFSP weights, not the safety gate.
    league_eval_games: int = 4


def _scripted_opponents(num_players: int) -> list[type[BaseAgent]]:
    """Phase-1 opponent factories: mostly HeuristicAgent, one RandomAgent."""
    n_opp = num_players - 1
    pool: list[type[BaseAgent]] = [HeuristicAgent] * n_opp
    if n_opp >= 1:
        pool[-1] = RandomAgent  # inject some exploration pressure
    return pool


# ---------------------------------------------------------------------------
# Picklable opponent factory (top-level so SubprocVecEnv can spawn it)
# ---------------------------------------------------------------------------


class _SnapshotFactory:
    """A picklable zero-arg builder for a :class:`FrozenSnapshotAgent`.

    Used in place of a lambda (which Windows ``spawn`` cannot pickle) when an
    opponent spec must cross a ``SubprocVecEnv`` process boundary. Carries only
    the checkpoint path + the determinism flag (§2.2); the heavy model is loaded
    lazily inside the worker.
    """

    def __init__(self, model_path: str, *, deterministic: bool) -> None:
        self.model_path = model_path
        self.deterministic = deterministic

    def __call__(self) -> FrozenSnapshotAgent:
        return FrozenSnapshotAgent(
            self.model_path, deterministic=self.deterministic
        )


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def _build_vec_env(
    *,
    track: Track | None,
    num_players: int,
    opponents,
    learner_id: int,
    config: PPOConfig,
    curriculum: CurriculumConfig,
    shaping_weight: float,
    seed: int,
) -> VecEnv:
    """Build the (optionally normalized) vec env for one training phase.

    Wraps :func:`heat.ml.vec.make_vec_env` and, when the curriculum enables it,
    a fresh ``VecNormalize`` (§2.6). The caller is responsible for transferring
    running stats across phases via :func:`_swap_vec_env`.
    """
    venv = make_vec_env(
        track=track,
        num_players=num_players,
        opponents=opponents,
        learner_id=learner_id,
        n_envs=max(1, config.n_envs),
        vec_cls="subproc" if config.n_envs > 1 else "dummy",
        seed=seed,
        shaping_weight=shaping_weight,
        shaping_progress_coef=config.shaping_progress_coef,
    )
    if curriculum.normalize_reward or curriculum.normalize_obs:
        venv = VecNormalize(
            venv,
            norm_obs=curriculum.normalize_obs,
            norm_reward=curriculum.normalize_reward,
            clip_reward=curriculum.clip_reward,
            gamma=config.gamma,
        )
    return venv


def _normalize_meta(curriculum: CurriculumConfig) -> dict | None:
    """The ``normalize`` meta block (§3.3) describing the stats sidecar, or None."""
    if not (curriculum.normalize_reward or curriculum.normalize_obs):
        return None
    return {
        "norm_obs": curriculum.normalize_obs,
        "norm_reward": curriculum.normalize_reward,
        "clip_reward": curriculum.clip_reward,
    }


def smoke_train(
    *,
    total_timesteps: int = 256,
    num_players: int = 2,
    track: Track | None = None,
    checkpoint_path: str | None = None,
    config: PPOConfig | None = None,
    n_envs: int | None = None,
    vec_cls: str = "dummy",
) -> tuple[MaskablePPO, str | None]:
    """Tiny end-to-end training run for the slow smoke test (§5c, §6.2, §6C).

    Trains a small net for ``total_timesteps`` steps against a scripted opponent
    pool over a (small) vec env, optionally saving a checkpoint (+ sidecar).
    Returns ``(model, checkpoint_path_or_None)``. Keeps everything small/fast:
    no self-play phase, tiny net unless ``config`` overrides. ``n_envs`` /
    ``vec_cls`` let the smoke test exercise both ``DummyVecEnv`` and
    ``SubprocVecEnv`` masking.
    """
    if config is None:
        # A deliberately tiny net + short rollout so the smoke test stays <~30s.
        config = PPOConfig(
            net_arch=[32, 32],
            features_extractor_hidden=[32],
            features_dim=32,
            n_steps=128,
            batch_size=64,
            verbose=0,
            seed=0,
        )
    if n_envs is not None:
        config = dataclasses.replace(config, n_envs=n_envs)

    apply_shaping_config(config)
    opponents = _scripted_opponents(num_players)
    venv = make_vec_env(
        track=track,
        num_players=num_players,
        opponents=opponents,
        learner_id=0,
        n_envs=max(1, config.n_envs),
        vec_cls=vec_cls if config.n_envs > 1 else "dummy",
        seed=config.seed or 0,
        shaping_weight=config.shaping_weight,
        shaping_progress_coef=config.shaping_progress_coef,
    )

    model = build_model(venv, config)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    track_name = (track.name if track is not None else _default_track().name)
    saved: str | None = None
    if checkpoint_path is not None:
        save_checkpoint(
            model,
            checkpoint_path,
            track_name=track_name,
            num_players=num_players,
            seed=config.seed,
            ppo_config=config,
        )
        saved = checkpoint_path

    venv.close()
    return model, saved


def _gate_score(
    model: MaskablePPO,
    *,
    curriculum: CurriculumConfig,
    num_players: int,
    track: Track | None,
    seed: int,
    vec_env: VecEnv | None = None,
    normalize: dict | None = None,
) -> float:
    """Evaluate the live model's win-rate vs the scripted pool (the §2.1 gate).

    Falls back to the existing :func:`heat.ml.evaluate.evaluate_ml` (the design's
    explicit fallback when 6B's richer eval is absent — **no 6B symbols are
    imported**). The model is written to a temp checkpoint and evaluated as an
    :class:`heat.agents.ml_agent.MLAgent` over a few games; returns the MLAgent
    win-rate in ``[0, 1]``. The gate uses the **real (un-normalized) win-rate**,
    so it is immune to ``VecNormalize`` reward scaling (§2.6 caveat). When the
    run normalizes *observations*, the stats sidecar is written next to the
    temp checkpoint so the gate's ``MLAgent`` applies the same obs normalization
    the policy was trained under.
    """
    import tempfile

    from heat.ml.evaluate import evaluate_ml

    with tempfile.TemporaryDirectory() as tmp:
        gate_path = os.path.join(tmp, "gate_model")
        # Save with a contract sidecar so MLAgent's tripwire is satisfied; pass
        # the normalize block + vec stats so an obs-normalized gate is correct.
        save_checkpoint(
            model,
            gate_path,
            track_name=(track.name if track is not None else _default_track().name),
            num_players=num_players,
            normalize=normalize,
            vec_env=vec_env,
        )
        per_agent = evaluate_ml(
            gate_path,
            num_games=curriculum.gate_games,
            num_players=num_players,
            track=track,
            seed=seed,
            parallel=False,
        )
    ml_stats = per_agent.get("MLAgent")
    if ml_stats is None:
        return 0.0
    return float(ml_stats.win_rate)


def train_self_play(
    config: PPOConfig | None = None,
    curriculum: CurriculumConfig | None = None,
    *,
    num_players: int = 4,
    track: Track | None = None,
    learner_id: int = 0,
    gate_fn=None,
) -> tuple[MaskablePPO, str]:
    """Run the Phase 1 -> Phase 2 self-play curriculum with 6C stability (§6C).

    Phase 1 trains vs the scripted pool. Phase 2 periodically freezes the policy
    to a snapshot and rebuilds the (vec) env with a ramped, stochastic,
    multi-snapshot opponent mix. After each Phase-2 chunk the model is scored by
    an eval gate; the **canonical ``run_name`` checkpoint is overwritten only
    when the score strictly improves**, so a collapsing Phase 2 can never destroy
    the best model. The collapsed final policy is saved separately to
    ``run_name_final``.

    Returns ``(model, best_checkpoint_path)`` — the best-evaluated checkpoint,
    not the (possibly collapsed) final model.

    Args:
        gate_fn: optional override for the eval gate (``model -> float``); used
            by tests to inject a deterministic score sequence. Defaults to a
            win-rate gate over :func:`heat.ml.evaluate.evaluate_ml` (§2.1
            fallback — no 6B dependency).
    """
    if config is None:
        config = PPOConfig()
    if curriculum is None:
        curriculum = CurriculumConfig()

    os.makedirs(curriculum.checkpoint_dir, exist_ok=True)
    seed = config.seed if config.seed is not None else 0

    best_path = os.path.join(curriculum.checkpoint_dir, curriculum.run_name)
    final_path = os.path.join(
        curriculum.checkpoint_dir, f"{curriculum.run_name}_final"
    )
    track_name = track.name if track is not None else _default_track().name
    norm_meta = _normalize_meta(curriculum)

    def _gate(m: MaskablePPO, venv: VecEnv | None = None) -> float:
        if gate_fn is not None:
            return float(gate_fn(m))
        return _gate_score(
            m,
            curriculum=curriculum,
            num_players=num_players,
            track=track,
            seed=seed,
            vec_env=venv,
            normalize=norm_meta,
        )

    def _save_best(m: MaskablePPO, venv: VecEnv) -> None:
        save_checkpoint(
            m,
            best_path,
            track_name=track_name,
            num_players=num_players,
            seed=seed,
            ppo_config=config,
            curriculum_config=curriculum,
            track_config={"track_name": track_name},
            normalize=norm_meta,
            vec_env=venv,
        )

    # --- Phase 1: scripted opponents ---
    apply_shaping_config(config)  # base shaping; the schedule overrides per-chunk
    scripted = _scripted_opponents(num_players)
    venv = _build_vec_env(
        track=track,
        num_players=num_players,
        opponents=scripted,
        learner_id=learner_id,
        config=config,
        curriculum=curriculum,
        shaping_weight=config.shaping_weight,
        seed=seed,
    )
    model = build_model(venv, config)
    model.learn(total_timesteps=curriculum.phase1_steps, progress_bar=False)

    # Establish the Phase-1 baseline as the first "best" — the model we must
    # never lose. This is the safety net (§2.1): even a fully collapsing Phase 2
    # cannot overwrite this checkpoint with a worse one.
    best_score = _gate(model, venv)
    _save_best(model, venv)

    base_lr = config.learning_rate

    # --- Phase 2: ramped, stochastic, multi-snapshot self-play ---
    snapshot_paths: list[str] = []
    remaining = max(0, curriculum.total_timesteps - curriculum.phase1_steps)
    step = max(1, curriculum.snapshot_every)
    total_phase2 = max(1, remaining)
    snap_idx = 0
    chunk_idx = 0

    # §6D opponent league + PFSP. The League lives ONLY in this (main) training
    # process; workers receive picklable snapshot *paths* via _SnapshotFactory,
    # never the live League (the design's process-boundary risk mitigation). When
    # ``use_league`` is False the FIFO 6C path is taken unchanged. The PFSP draw
    # uses a dedicated seeded RNG so opponent selection is reproducible (gate).
    league: League | None = None
    league_rng: np.random.Generator | None = None
    if curriculum.use_league:
        league = League(
            capacity=curriculum.league_capacity,
            pfsp_mode=curriculum.league_pfsp_mode,
            pfsp_exponent=curriculum.league_pfsp_exponent,
            p_min=curriculum.league_p_min,
            p_max=curriculum.league_p_max,
            min_games=curriculum.league_min_games,
            strength_weight=curriculum.league_strength_weight,
            diversity_weight=curriculum.league_diversity_weight,
        )
        league_rng = np.random.default_rng(seed)

    while remaining > 0:
        # Freeze the current policy and register it as a (stochastic) opponent.
        snap_path = os.path.join(
            curriculum.checkpoint_dir, f"{curriculum.run_name}_snap{snap_idx}"
        )
        save_checkpoint(
            model,
            snap_path,
            track_name=track_name,
            num_players=num_players,
            seed=seed,
            ppo_config=config,
            vec_env=venv,
        )
        if league is not None:
            # §6D: register the new snapshot in the league (gate_score = current
            # best as a strength proxy), retain by keep-strong-and-diverse rather
            # than FIFO, and always anchor the current best.
            league.add(
                LeagueEntry(
                    path=snap_path,
                    snapshot_index=snap_idx,
                    gate_score=best_score,
                )
            )
            league.set_anchor(snap_path)
            league.retain()
        else:
            snapshot_paths.append(snap_path)
            if len(snapshot_paths) > curriculum.max_snapshots:
                snapshot_paths.pop(0)  # FIFO cap
        snap_idx += 1

        # §2.4 opponent-mix ramp: snapshot fraction grows 0 -> snapshot_mix
        # across Phase 2, limiting the value-function shock at the swap.
        progress = 1.0 - (remaining / total_phase2)
        mix = curriculum.snapshot_mix * progress

        # §2.3 shaping schedule: anneal shaping_weight_start -> _end across P2.
        shaping_w = (
            curriculum.shaping_weight_start
            + (curriculum.shaping_weight_end - curriculum.shaping_weight_start)
            * progress
        )

        sampled_paths: list[str] = []
        if league is not None:
            assert league_rng is not None
            opponents, sampled_paths = _league_opponent_pool(
                num_players,
                league,
                mix,
                league_rng,
                deterministic=curriculum.snapshot_deterministic,
            )
        else:
            opponents = _mixed_opponent_pool(
                num_players,
                snapshot_paths,
                mix,
                deterministic=curriculum.snapshot_deterministic,
            )

        # Rebuild the (vec) env on the new mix, preserving VecNormalize stats.
        new_venv = _build_vec_env(
            track=track,
            num_players=num_players,
            opponents=opponents,
            learner_id=learner_id,
            config=config,
            curriculum=curriculum,
            shaping_weight=shaping_w,
            seed=seed + 1000 + chunk_idx,
        )
        venv = _swap_vec_env(model, venv, new_venv)

        # §2.5 critic warm-up + LR/entropy schedule on the leading chunk(s).
        warmup = chunk_idx < curriculum.warmup_chunks
        lr = base_lr * (curriculum.phase2_lr_scale if warmup else 1.0)
        ent = config.ent_coef * (curriculum.phase2_ent_scale if warmup else 1.0)
        model.learning_rate = lr
        model.lr_schedule = (lambda _progress, _lr=lr: _lr)
        model.ent_coef = ent

        chunk = min(step, remaining)
        model.learn(
            total_timesteps=chunk,
            progress_bar=False,
            reset_num_timesteps=False,
        )
        remaining -= chunk
        chunk_idx += 1

        # §2.1 eval-gated best-checkpoint preservation: only overwrite best_path
        # when the score STRICTLY improves. Log the gate score for TensorBoard
        # so a collapse is visible live (§3.2).
        score = _gate(model, venv)
        model.logger.record("eval/gate_score", score)
        model.logger.record("eval/best_score", best_score)
        if score > best_score:
            best_score = score
            _save_best(model, venv)
        elif (
            curriculum.reload_best_on_regression
            and score < best_score - curriculum.regression_tol
        ):
            # Recovery option (§2.1): reload the best policy weights so a
            # collapsing Phase 2 continues from strength rather than from rubble.
            model.set_parameters(best_path, device=resolve_device(config.device))

        # §6D win-rate bookkeeping (PREFERRED eval-based attribution): between
        # learn chunks, run a SMALL deterministic head-to-head of the current
        # learner against each league member it just faced, and fold the estimate
        # back into the league so the PFSP weights are data-driven. This is
        # bookkeeping (kept small via ``league_eval_games``), not the safety gate.
        # The learner is written to a temp checkpoint so the existing path-pickled
        # eval harness (head_to_head + ml_agent_factory) can load it; the live
        # League never leaves this process.
        if league is not None and sampled_paths:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                learner_path = os.path.join(tmp, "league_learner")
                save_checkpoint(
                    model,
                    learner_path,
                    track_name=track_name,
                    num_players=num_players,
                    seed=seed,
                    normalize=norm_meta,
                    vec_env=venv,
                )
                for j, opp_path in enumerate(dict.fromkeys(sampled_paths)):
                    wr = _estimate_learner_win_rate(
                        learner_path,
                        opp_path,
                        num_games=curriculum.league_eval_games,
                        track=track,
                        seed=seed + 7000 + chunk_idx * 97 + j,
                    )
                    league.record_win_rate(
                        opp_path, wr, curriculum.league_eval_games
                    )

    # The (possibly collapsed) final model goes to a SEPARATE path — never the
    # canonical best_path.
    save_checkpoint(
        model,
        final_path,
        track_name=track_name,
        num_players=num_players,
        seed=seed,
        ppo_config=config,
        curriculum_config=curriculum,
        normalize=norm_meta,
        vec_env=venv,
    )

    venv.close()
    return model, best_path


def _swap_vec_env(model: MaskablePPO, old: VecEnv, new: VecEnv) -> VecEnv:
    """Point ``model`` at ``new``, carrying ``VecNormalize`` stats forward (§2.6).

    The running normalization statistics are *part of the model* — re-creating
    a fresh ``VecNormalize`` per phase would reset them and re-introduce the very
    scale shock §2.6 exists to prevent. So when both envs are ``VecNormalize``
    wrappers, the new one inherits the old one's running means/vars/counts before
    the swap. The old vec env is then closed to release its worker processes.
    """
    if isinstance(old, VecNormalize) and isinstance(new, VecNormalize):
        # Copy whichever running stats this VecNormalize actually maintains.
        # When norm_obs/norm_reward is off, SB3 may not expose obs_rms/ret_rms,
        # so guard each access via the instance __dict__ (avoid the recursive
        # __getattr__ that would delegate to the inner unwrapped VecEnv).
        if "obs_rms" in old.__dict__ and "obs_rms" in new.__dict__:
            new.obs_rms = old.__dict__["obs_rms"]
        if "ret_rms" in old.__dict__ and "ret_rms" in new.__dict__:
            new.ret_rms = old.__dict__["ret_rms"]
    model.set_env(new)
    if old is not new:
        old.close()
    return new


def _mixed_opponent_pool(
    num_players: int,
    snapshot_paths: list[str],
    snapshot_mix: float,
    *,
    deterministic: bool = False,
):
    """Build an opponent-spec list mixing scripted agents and frozen snapshots.

    Snapshot seats are :class:`_SnapshotFactory` instances bound to a path
    (top-level + picklable, so they survive ``SubprocVecEnv`` spawn, §6.5);
    they sample (``deterministic=False``, §2.2) by default so the learner sees
    diverse losses rather than a uniform −1 from identical opponents. Distinct
    snapshots are spread across seats so the pool is varied, and the remaining
    seats stay scripted (``HeuristicAgent``) so the pool is never all
    self-copies (§2.4).
    """
    n_opp = num_players - 1
    n_snap = min(n_opp, int(round(n_opp * snapshot_mix))) if snapshot_paths else 0

    specs: list = []
    for i in range(n_snap):
        # Spread across distinct snapshots (newest-first) for opponent variety.
        path = snapshot_paths[-(1 + (i % len(snapshot_paths)))]
        specs.append(_SnapshotFactory(path, deterministic=deterministic))
    specs.extend(HeuristicAgent for _ in range(n_opp - n_snap))
    return specs


# ---------------------------------------------------------------------------
# Sprint 6D: league-driven opponent pool + win-rate attribution
# ---------------------------------------------------------------------------


def _n_snapshot_seats(num_players: int, snapshot_mix: float, pool_size: int) -> int:
    """Number of opponent seats drawn from the snapshot pool (matches §2.4 ramp).

    Mirrors :func:`_mixed_opponent_pool`'s seat split so the league fills exactly
    the same number of snapshot seats the FIFO path would, keeping the 6C
    opponent-mix ramp semantics intact.
    """
    n_opp = num_players - 1
    if pool_size <= 0:
        return 0
    return min(n_opp, int(round(n_opp * snapshot_mix)))


def _league_opponent_pool(
    num_players: int,
    league: "League",
    snapshot_mix: float,
    rng: "np.random.Generator",
    *,
    deterministic: bool = False,
) -> tuple[list, list[str]]:
    """Build the Phase-2 opponent specs with snapshot seats chosen by the league.

    The same seat count as :func:`_mixed_opponent_pool` (so the §2.4 ramp is
    preserved) is filled by :meth:`League.sample` (PFSP priority) rather than by
    index cycling; the remaining seats stay scripted ``HeuristicAgent`` so the
    pool is never all self-copies. Snapshot seats are :class:`_SnapshotFactory`
    instances bound to a **path** (picklable across ``SubprocVecEnv`` spawn) and
    sample (``deterministic=False``, §2.2) by default.

    Returns ``(specs, sampled_paths)`` -- the sampled paths are returned so the
    caller can attribute results back to the originating league entry (the live
    League never crosses the process boundary; only paths do).
    """
    n_opp = num_players - 1
    n_snap = _n_snapshot_seats(num_players, snapshot_mix, len(league))

    sampled_paths = league.sample(n_snap, rng)
    specs: list = [
        _SnapshotFactory(path, deterministic=deterministic) for path in sampled_paths
    ]
    specs.extend(HeuristicAgent for _ in range(n_opp - len(sampled_paths)))
    return specs, sampled_paths


def _estimate_learner_win_rate(
    learner_path: str,
    opponent_path: str,
    *,
    num_games: int,
    track: Track | None,
    seed: int,
) -> float:
    """Seat-neutral learner-vs-opponent win-rate via :func:`evaluate.head_to_head`.

    The chosen win-rate-attribution strategy (§"Win-rate bookkeeping"): rather
    than instrumenting the vectorized env workers to surface per-game outcomes
    (invasive -- the env build assigns opponent paths per seat), run a SMALL
    deterministic evaluation of the current learner against each sampled league
    member between ``learn`` chunks and fold the estimate into the league. This
    keeps the live :class:`League` single-source-of-truth in the main process
    and reuses the existing 6B eval harness.

    ``head_to_head`` runs half the games with the learner in the front seat and
    half in the back, averaging out the known front-seat positional advantage so
    the win-rate is a real skill signal, not a seat artifact.
    """
    from heat.ml.evaluate import head_to_head, ml_agent_factory

    stats = head_to_head(
        ml_agent_factory(learner_path),
        ml_agent_factory(opponent_path),
        label_a="learner",
        label_b="opponent",
        num_games=num_games,
        track=track,
        seed=seed,
        parallel=False,
    )
    return float(stats.a_win_rate)
