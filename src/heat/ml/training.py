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
from heat.agents.strong_heuristic import StrongHeuristicAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card
from heat.models.game_state import GameState
from heat.models.track import Track
from heat.ml import spaces
from heat.ml.action_codec import decode_action, legal_action_mask
from heat.ml.env import HeatEnv, TrackSource, _default_track
from heat.tracks.generator import (
    CurriculumSchedule,
    StepAwareTrackSampler,
    TrackGenParams,
    TrackSampler,
    default_curriculum_schedule,
    generate_track,
)
from heat.ml.features import encode_observation
from heat.ml.league import League, LeagueEntry
from heat.ml.model import (
    PPOConfig,
    apply_shaping_config,
    build_model,
    resolve_device,
)
from heat.ml.vec import make_vec_env
from heat.simulation.stats import wilson_interval

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

    # --- §6E strong-heuristic curriculum ---
    #: Upgrade the Phase-1 (and scripted-seat) opponents from ``HeuristicAgent``
    #: to the 6E :class:`~heat.agents.strong_heuristic.StrongHeuristicAgent`
    #: strength bar. Off by default so existing runs are unchanged.
    use_strong_heuristic_opponents: bool = False

    # --- Sprint A: trustworthy & measurable run ---
    #: Steps per Phase-1 learn/gate chunk (Idea 7). The held-out gate runs after
    #: each chunk, so a Phase-1-heavy run (Idea 4: ``phase1_steps ==
    #: total_timesteps``) still periodically preserves the BEST checkpoint rather
    #: than keeping the (possibly regressed) final weights. Set so the periodic
    #: eval does not dominate wall-clock -- tie it to ``gate_games``. A value of 0
    #: or ``>= phase1_steps`` collapses Phase 1 to a single gate (the legacy
    #: one-shot behavior).
    phase1_eval_every: int = 50_000
    #: When True, the promotion criterion is the Wilson LOWER bound of the pooled
    #: held-out ``(wins, games)`` vs the trained-against opponent, not the point
    #: estimate (Idea 9). When False the point estimate is used, keeping the
    #: legacy behavior for old runs. ``True`` is the default so a lucky, low-game
    #: checkpoint cannot displace a genuinely-better one.
    gate_use_wilson_lb: bool = True
    #: Randomize the learner's start seat each episode (Idea 10). Off by default so
    #: existing fixed-seat runs are byte-for-byte unchanged. When on, the obs is
    #: unchanged (opponents are encoded relatively) -- only the engine seat the
    #: policy drives moves, removing the seat-0 overfit.
    randomize_seat: bool = False
    #: Broaden the Phase-1 strong-opponent pool (strengths 2 & 3 + heuristic +
    #: random) for curriculum variety (Idea 6). Only takes effect together with
    #: ``use_strong_heuristic_opponents``. Off by default.
    broaden_phase1_mix: bool = False

    # --- Sprint B: step-aware track-difficulty curriculum (Idea 2) ---
    #: Enable the step-aware track-difficulty curriculum: training tracks start
    #: from a narrow "easy" distribution (short, few corners, 1 lap) and widen to
    #: the full generated distribution over ``curriculum_horizon_steps``. Off by
    #: default so existing generated-track runs are byte-for-byte unchanged. Only
    #: takes effect when ``track is None`` (generated-track training); a pinned
    #: fixed Track is left untouched.
    use_track_curriculum: bool = False
    #: Steps over which difficulty ramps easy -> full (the schedule horizon).
    curriculum_horizon_steps: int = 1_000_000
    #: Curriculum shape: "linear" (continuous ramp, realised as fine-grained
    #: staged rebuilds) or "staged" (a few discrete difficulty stages). Both are
    #: implemented as process-safe staged vec-env rebuilds (see §9 resolution);
    #: "linear" simply uses more stages.
    curriculum_shape: str = "linear"
    #: Number of stages when ``curriculum_shape == "staged"`` (and the stage count
    #: cap for "linear"). The Phase-1 chunk loop rebuilds the training env with a
    #: fresh step-pinned sampler at each stage boundary.
    curriculum_stages: int = 4

    # NOTE (Idea 15, recorded constraint -- NOT a knob to flip): keep
    # ``normalize_obs = False``. The obs is already bounded to [-1, 1] by
    # construction (features.py clipping) and the gate uses the real
    # un-normalized win-rate, so observation normalization buys nothing and would
    # only risk a train/inference normalization-stats mismatch.


class _StrongHeuristicFactory:
    """A picklable zero-arg builder for a :class:`StrongHeuristicAgent(strength)`.

    Used in place of a lambda (which Windows ``spawn`` cannot pickle) when the
    broadened Phase-1 pool (Idea 6) needs a strength-3 seat: the env instantiates
    opponent specs by calling them with no args, but ``StrongHeuristicAgent``
    takes ``strength`` as a keyword. Carrying it on a tiny top-level callable
    keeps the spec picklable across a ``SubprocVecEnv`` boundary.
    """

    def __init__(self, strength: int) -> None:
        self.strength = strength

    def __call__(self) -> StrongHeuristicAgent:
        return StrongHeuristicAgent(strength=self.strength)


def _broadened_strong_pool(n_opp: int) -> list:
    """A varied strong Phase-1 opponent pool of exactly ``n_opp`` entries (Idea 6).

    Cycles the template ``[Strong(3), Strong(2), Heuristic, Random]`` to fill the
    available opponent seats, so the learner faces a mix of difficulty rungs plus
    exploration pressure rather than a single repeated class. Each entry is a
    zero-arg picklable callable (a class or :class:`_StrongHeuristicFactory`).
    """
    template: list = [
        _StrongHeuristicFactory(3),
        _StrongHeuristicFactory(2),
        HeuristicAgent,
        RandomAgent,
    ]
    return [template[i % len(template)] for i in range(n_opp)]


def sprint_a_curriculum(
    total_timesteps: int = 2_000_000,
    *,
    run_name: str = "heat_ppo_sprintA",
    checkpoint_dir: str = "checkpoints",
) -> CurriculumConfig:
    """The Sprint A "trustworthy & measurable run" curriculum preset (§5 task 8).

    Bundles the Sprint-A levers into one launch config so the held-out gate is
    honest before any long run / Sprint B:

    * **Idea 4** -- Phase-1-heavy: ``phase1_steps == total_timesteps`` (Phase 2
      skipped entirely), so the periodic Phase-1 gate (Idea 7) is the ONLY thing
      preserving the best model.
    * **Idea 9** -- ``gate_games`` raised to 120 (40 per held-out track) so the
      Wilson lower bound is usable, with ``gate_use_wilson_lb=True``.
    * **Idea 7** -- ``phase1_eval_every`` set to a sensible cadence relative to
      the raised game count (tie cadence to cost, §8).
    * **Idea 8 / 6** -- ``use_strong_heuristic_opponents=True`` (+ broadened mix)
      so we gate against the opponent we train against.
    * **Idea 10** -- ``randomize_seat=True`` to remove the seat-0 overfit.
    * **Idea 15** -- ``normalize_obs=False`` (recorded constraint, kept off).

    The non-zero ``shaping_weight`` lives on the :class:`PPOConfig` (it is a
    model/env knob, applied by ``apply_shaping_config``), so the run script pairs
    this with ``PPOConfig(shaping_weight=...)``.
    """
    return CurriculumConfig(
        total_timesteps=total_timesteps,
        phase1_steps=total_timesteps,  # Idea 4: Phase-1-only run
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        # Idea 7: periodic Phase-1 gate cadence (tie to the raised gate_games).
        phase1_eval_every=100_000,
        # Idea 9: enough games for a usable Wilson LB (40 / held-out track).
        gate_games=120,
        gate_use_wilson_lb=True,
        # Idea 8 / 6: gate against (and broaden) the trained-against opponent.
        use_strong_heuristic_opponents=True,
        broaden_phase1_mix=True,
        # Idea 10: remove the seat-0 overfit.
        randomize_seat=True,
        # Idea 15 (recorded constraint): keep observation normalization OFF.
        normalize_obs=False,
        normalize_reward=False,
    )


def sprint_b_curriculum(
    total_timesteps: int = 2_000_000,
    *,
    run_name: str = "heat_ppo_sprintB",
    checkpoint_dir: str = "checkpoints",
    curriculum_stages: int = 4,
) -> CurriculumConfig:
    """The Sprint B "cheap whole-track obs + curriculum baseline" preset.

    Builds on :func:`sprint_a_curriculum`: keeps the Sprint-A trustworthy gate
    (strong opponents, Wilson LB, seat randomization, Phase-1-heavy chunked
    learn/gate loop) and adds the Sprint B step-aware track-difficulty curriculum
    (Idea 2) on top of the Option-A whole-track obs (which is automatic via the
    v2 codec -- no flag needed). The curriculum horizon is tied to
    ``total_timesteps`` so difficulty ramps easy -> full across (essentially) the
    whole Phase-1-only run.

    Per the Sprint B spec:
      * Option-A obs: automatic (CODEC_VERSION == 2; OBS_DIM == 104).
      * Curriculum: ON (``use_track_curriculum=True``), horizon == total steps.
      * Sprint-A trustworthy gate: ON (inherited).
      * ``normalize_obs=False`` (Idea 15 recorded constraint).
      * ``phase1_steps == total_timesteps`` (Phase-1-only; the periodic gate is
        the only thing preserving the best checkpoint).
    """
    cfg = sprint_a_curriculum(
        total_timesteps,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
    )
    return dataclasses.replace(
        cfg,
        use_track_curriculum=True,
        curriculum_horizon_steps=total_timesteps,
        curriculum_shape="linear",
        curriculum_stages=curriculum_stages,
    )


def sprint_8c_curriculum(
    total_timesteps: int = 1_600_000,
    *,
    run_name: str = "heat_ppo_sprint8C",
    checkpoint_dir: str = "checkpoints",
) -> CurriculumConfig:
    """The Sprint 8C "solo pretrain + opponent curriculum" preset (§8.1).

    Built on :func:`sprint_a_curriculum` (so the trustworthy Wilson-LB gate +
    seat randomization are inherited) with the TRACK curriculum demoted to
    default-off -- the validated recipe ramps the OPPONENT axis instead
    (``docs/ml-learnings-solo-pretrain.md``). The solo phase + opponent ramp are
    expressed as a :class:`TrainingPhase` list (see :func:`default_8c_phases`),
    passed to :func:`train_self_play` via ``phases=``; this preset carries the
    gate/seat/normalization levers those phases run under.

    ``total_timesteps`` is informational here (the phase list owns the per-stage
    budgets); it sets the sidecar metadata + the Phase-1-heavy ``phase1_steps``
    default inherited from Sprint A.
    """
    cfg = sprint_a_curriculum(
        total_timesteps,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
    )
    return dataclasses.replace(
        cfg,
        use_track_curriculum=False,  # demoted (net-negative per learnings)
        randomize_seat=True,  # the proven Sprint-A win, kept
        use_strong_heuristic_opponents=True,
        broaden_phase1_mix=True,
        gate_use_wilson_lb=True,
        normalize_obs=False,  # Idea 15 recorded constraint
        normalize_reward=False,
    )


def _scripted_opponents(
    num_players: int, *, use_strong: bool = False, broaden_mix: bool = False
) -> list:
    """Phase-1 opponent factories.

    Default pool is mostly :class:`HeuristicAgent` with one :class:`RandomAgent`
    for exploration pressure. When ``use_strong`` (§6E), the heuristic seats are
    upgraded to :class:`StrongHeuristicAgent` (the relative-objective strength
    bar, default ``strength=2``) so the learner trains against a far harder
    scripted curriculum; one ``RandomAgent`` seat is retained for exploration.

    When ``use_strong`` *and* ``broaden_mix`` (Idea 6), the pool is broadened to
    a mix of ``StrongHeuristicAgent`` strengths 2 & 3, a plain ``HeuristicAgent``,
    and a ``RandomAgent`` (cycled to fill the seats) for curriculum variety.
    ``broaden_mix`` is a no-op without ``use_strong`` so non-strong runs are
    unchanged.

    The entries are zero-arg callables (a class, or a small picklable factory),
    matching how the env instantiates non-snapshot opponent specs.
    """
    n_opp = num_players - 1
    if use_strong and broaden_mix:
        return _broadened_strong_pool(n_opp)
    base: type[BaseAgent] = StrongHeuristicAgent if use_strong else HeuristicAgent
    pool: list = [base] * n_opp
    if n_opp >= 1:
        pool[-1] = RandomAgent  # inject some exploration pressure
    return pool


def _mixed_strength_pool(num_players: int) -> list:
    """A "mixed" opponent rung between weak and the broadened strong pool (8C).

    Cycles the template ``[Strong(2), Heuristic, Random]`` to fill exactly
    ``num_players - 1`` seats -- one rung of strong pressure, one weak heuristic,
    and exploration -- so the learner ramps through an intermediate difficulty
    before the full broadened-strong pool. Every entry is a zero-arg picklable
    callable (a class or :class:`_StrongHeuristicFactory`), so the pool survives
    ``SubprocVecEnv`` spawn unchanged.
    """
    n_opp = num_players - 1
    template: list = [
        _StrongHeuristicFactory(2),
        HeuristicAgent,
        RandomAgent,
    ]
    return [template[i % len(template)] for i in range(n_opp)]


# ---------------------------------------------------------------------------
# Sprint 8C: opponent-curriculum schedule + multi-phase training plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpponentStage:
    """One opponent-curriculum stage: a pool spec + a step budget (Sprint 8C).

    ``pool_kind`` is one of ``"weak"`` / ``"mixed"`` / ``"strong"``; ``steps`` is
    the learn budget for the stage. Frozen + hashable (plain ``str``/``int``) so
    it pickles into ``SubprocVecEnv`` workers cleanly.
    """

    pool_kind: str
    steps: int


@dataclass(frozen=True)
class OpponentSchedule:
    """Weak -> mixed -> strong opponent ramp across staged vec-env rebuilds (8C).

    Generalizes Sprint B's ``CurriculumSchedule`` from the track axis to the
    OPPONENT axis (the axis ``docs/ml-learnings-solo-pretrain.md`` proved
    matters). Each stage names a pool kind, resolved to picklable opponent
    factories by :meth:`pool_for`. Frozen + hashable; the pools it produces are
    lists of zero-arg picklable factories so they survive ``SubprocVecEnv`` spawn
    (only the resolved list, never the live schedule, crosses the boundary).
    """

    stages: tuple[OpponentStage, ...]

    def pool_for(self, num_players: int, stage: int) -> list:
        """Resolve the opponent factory list for ``stage`` (clamped to range).

        Out-of-range indices clamp to the final stage so a phase list with more
        opponent phases than schedule stages still resolves to the hardest pool.
        """
        if not self.stages:
            raise ValueError("OpponentSchedule has no stages")
        idx = max(0, min(stage, len(self.stages) - 1))
        kind = self.stages[idx].pool_kind
        if kind == "weak":
            # 2x HeuristicAgent + Random (the proto's "weak").
            return _scripted_opponents(num_players, use_strong=False)
        if kind == "strong":
            # Broadened strong pool (Strong3/Strong2/Heuristic/Random).
            return _scripted_opponents(
                num_players, use_strong=True, broaden_mix=True
            )
        if kind == "mixed":
            # Half-strong / half-weak rung between weak and strong.
            return _mixed_strength_pool(num_players)
        raise ValueError(f"unknown opponent stage kind {kind!r}")


@dataclass(frozen=True)
class TrainingPhase:
    """One stage of the 8C recipe (solo pretrain or an opponent-curriculum rung).

    A frozen dataclass of primitives so the whole phase list pickles trivially.
    ``pool_kind`` is ``None`` for the solo phase (which has no opponents) and one
    of ``"weak"``/``"mixed"``/``"strong"`` for the opponent phases.
    """

    name: str
    num_players: int
    reward_mode: str
    gamma: float
    shaping_weight: float
    steps: int
    pool_kind: str | None


#: The default 8C phase list (the validated recipe + the explicit mixed rung).
#: Solo runs at gamma 0.99 (the §4.3 speed gradient); the race phases at 0.999.
#: Per-stage step budgets are PLACEHOLDERS the run tunes (§6.1/§10).
def default_8c_phases(num_players: int = 4) -> list[TrainingPhase]:
    """Return the default solo -> weak -> mixed -> strong phase list (§6.1)."""
    return [
        TrainingPhase("solo", 1, "solo", 0.99, 1.0, 300_000, None),
        TrainingPhase("weak", num_players, "race", 0.999, 0.05, 300_000, "weak"),
        TrainingPhase("mixed", num_players, "race", 0.999, 0.05, 400_000, "mixed"),
        TrainingPhase("strong", num_players, "race", 0.999, 0.05, 600_000, "strong"),
    ]


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
    reward_mode: str | None = None,
    solo_finish_bonus: float | None = None,
) -> VecEnv:
    """Build the (optionally normalized) vec env for one training phase.

    Wraps :func:`heat.ml.vec.make_vec_env` and, when the curriculum enables it,
    a fresh ``VecNormalize`` (§2.6). The caller is responsible for transferring
    running stats across phases via :func:`_swap_vec_env`.

    ``reward_mode`` / ``solo_finish_bonus`` (Sprint 8C) override the values taken
    from ``config`` when set, so a solo phase can build a solo-reward env while a
    race phase uses the race reward, without mutating the shared ``PPOConfig``.
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
        shaping_spinout_weight=config.shaping_spinout_weight,
        shaping_spinout_cap=config.shaping_spinout_cap,
        reward_mode=reward_mode if reward_mode is not None else config.reward_mode,
        solo_finish_bonus=(
            solo_finish_bonus
            if solo_finish_bonus is not None
            else config.solo_finish_bonus
        ),
        randomize_seat=curriculum.randomize_seat,
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


#: Fixed seeds for the held-out generated tracks the eval gate scores against
#: when training on generated tracks. Held constant so the gate is comparable
#: across chunks (and distinct enough from typical training episode seeds that
#: it is a generalization signal, not a memorized-track score).
_HOLDOUT_TRACK_SEEDS: tuple[int, ...] = (90_000_001, 90_000_002, 90_000_003)


def _resolve_track_source(
    track: TrackSource | None,
    seed: int,
    curriculum: "CurriculumConfig | None" = None,
) -> TrackSource:
    """Resolve the training track source.

    ``None`` (the default) -> a :class:`TrackSampler` that draws a fresh
    generated track every episode (training on procedurally generated tracks is
    now the default, §6A). A fixed :class:`Track` pins one specific track; an
    already-built sampler callable is passed through unchanged.

    When ``curriculum.use_track_curriculum`` and ``track is None`` (Sprint B Idea
    2), return a step-aware :class:`StepAwareTrackSampler` pinned to step 0
    instead; the Phase-1 chunk loop advances its step by rebuilding the vec env
    at stage boundaries (process-safe staged rebuilds). The curriculum only
    applies to generated-track training -- a pinned fixed Track is left untouched.
    """
    if track is None:
        if curriculum is not None and curriculum.use_track_curriculum:
            schedule = default_curriculum_schedule(
                curriculum.curriculum_horizon_steps
            )
            return StepAwareTrackSampler(schedule, base_seed=seed, step=0)
        return TrackSampler(base_seed=seed)
    return track


def _track_label(track_source: TrackSource) -> str:
    """Checkpoint-metadata label for a track source (samplers are 'generated')."""
    return track_source.name if isinstance(track_source, Track) else "generated"


def _curriculum_stage_index(step: int, horizon: int, n_stages: int) -> int:
    """Discrete curriculum stage (0..n_stages-1) for a global ``step`` (Sprint B).

    Maps the [0, horizon] ramp into ``n_stages`` equal bands; steps at/after the
    horizon clamp to the final (full-difficulty) stage. Used to decide when the
    Phase-1 chunk loop must rebuild the training vec env with a harder
    step-pinned sampler (process-safe staged rebuilds; see §9 resolution).
    """
    n_stages = max(1, n_stages)
    if horizon <= 0:
        return n_stages - 1
    frac = min(1.0, max(0.0, step / horizon))
    return min(n_stages - 1, int(frac * n_stages))


def _curriculum_stage_step(stage: int, horizon: int, n_stages: int) -> int:
    """Representative global step for a curriculum ``stage`` (Sprint B).

    Returns the step at the *start* of the stage's band, so a fresh step-pinned
    :class:`StepAwareTrackSampler` built for the stage uses that band's
    difficulty. The final stage maps to ``horizon`` (full difficulty).
    """
    n_stages = max(1, n_stages)
    if stage >= n_stages - 1:
        return horizon
    return round(stage * horizon / n_stages)


def _gate_tracks(track_source: TrackSource) -> list[Track]:
    """The track(s) the eval gate scores on for a given training source.

    A fixed track gates on itself; generated-track training gates on a fixed
    held-out set so the gate measures cross-track generalization with a stable,
    reproducible signal.
    """
    if isinstance(track_source, Track):
        return [track_source]
    return [generate_track(s, name=f"holdout-{s}") for s in _HOLDOUT_TRACK_SEEDS]


@dataclass
class GateResult:
    """Outcome of one held-out eval gate (Sprint A, Ideas 8/9).

    The gate now reports BOTH the trained-against (strong) win-rate and the weak
    ``HeuristicAgent`` win-rate, and promotes on the Wilson lower bound of the
    pooled ``(wins, games)`` against the trained-against opponent so a lucky,
    low-game checkpoint cannot displace a genuinely-better one.

    Attributes:
        win_rate_strong: Pooled point-estimate win-rate vs the trained-against
            pool (the strong heuristic when ``use_strong_heuristic_opponents``,
            else the weak heuristic).
        win_rate_weak: Pooled point-estimate win-rate vs the weak
            ``HeuristicAgent`` (reporting only).
        wilson_lb_strong: Wilson 95% LOWER bound of the pooled strong
            ``(wins, games)`` -- the promotion criterion (Idea 9).
        games_strong: Total games played in the strong pass (>= 0).
    """

    win_rate_strong: float
    win_rate_weak: float
    wilson_lb_strong: float
    games_strong: int

    @property
    def promote_score(self) -> float:
        """The scalar the best-checkpoint logic compares (Idea 9 Wilson LB)."""
        return self.wilson_lb_strong


def _pooled_win_counts(
    gate_path: str,
    *,
    gate_tracks: list[Track],
    per_track_games: int,
    num_players: int,
    seed: int,
    opponent_factory,
) -> tuple[int, int]:
    """Pool ``(wins, games)`` for the MLAgent across the held-out gate track(s).

    Aggregates integer ``AgentStats.wins`` / ``AgentStats.games_played`` (not a
    mean of per-track rates) so the pooled counts can feed
    :func:`wilson_interval` directly (Idea 9). ``opponent_factory`` selects the
    opponent the win-rate is measured against (Idea 8).
    """
    from heat.ml.evaluate import evaluate_ml

    total_wins = 0
    total_games = 0
    for i, gt in enumerate(gate_tracks):
        per_agent = evaluate_ml(
            gate_path,
            num_games=per_track_games,
            num_players=num_players,
            track=gt,
            seed=seed + i,
            parallel=False,
            opponent_factory=opponent_factory,
        )
        ml_stats = per_agent.get("MLAgent")
        if ml_stats is not None:
            total_wins += int(ml_stats.wins)
            total_games += int(ml_stats.games_played)
        else:
            total_games += per_track_games
    return total_wins, total_games


def _gate_score(
    model: MaskablePPO,
    *,
    curriculum: CurriculumConfig,
    num_players: int,
    track: TrackSource | None,
    seed: int,
    vec_env: VecEnv | None = None,
    normalize: dict | None = None,
) -> GateResult:
    """Evaluate the live model on the held-out gate (Ideas 8/9; §2.1 fallback).

    Falls back to the existing :func:`heat.ml.evaluate.evaluate_ml` (the design's
    explicit fallback when 6B's richer eval is absent -- **no 6B symbols are
    imported**). The model is written to a temp checkpoint and evaluated as an
    :class:`heat.agents.ml_agent.MLAgent` over a few games per held-out track.

    Sprint A changes (vs the 6C point-estimate gate):

    * **Idea 8** -- score against the opponent we actually train against. When
      ``use_strong_heuristic_opponents``, the strong pass uses
      :func:`heat.ml.evaluate.strong_heuristic_agent_factory`; a separate weak
      pass (vs the default ``HeuristicAgent``) is reported alongside.
    * **Idea 9** -- pool wins/games across the held-out track(s) and promote on
      the Wilson LOWER bound of the pooled strong counts (when
      ``gate_use_wilson_lb``), not the point estimate.

    The gate uses the **real (un-normalized) win-rate**, so it is immune to
    ``VecNormalize`` reward scaling (§2.6 caveat). When the run normalizes
    *observations*, the stats sidecar is written next to the temp checkpoint so
    the gate's ``MLAgent`` applies the same obs normalization the policy was
    trained under.
    """
    import tempfile

    from heat.ml.evaluate import strong_heuristic_agent_factory

    gate_tracks = _gate_tracks(track) if track is not None else _gate_tracks(
        TrackSampler(base_seed=seed)
    )
    per_track_games = max(1, curriculum.gate_games // len(gate_tracks))

    # The trained-against opponent (Idea 8): the strong heuristic when the
    # curriculum trains vs it, else the same weak heuristic the weak pass uses.
    strong_factory = (
        strong_heuristic_agent_factory()
        if curriculum.use_strong_heuristic_opponents
        else None
    )

    with tempfile.TemporaryDirectory() as tmp:
        gate_path = os.path.join(tmp, "gate_model")
        # Save with a contract sidecar so MLAgent's tripwire is satisfied; pass
        # the normalize block + vec stats so an obs-normalized gate is correct.
        save_checkpoint(
            model,
            gate_path,
            track_name=_track_label(track) if track is not None else "generated",
            num_players=num_players,
            normalize=normalize,
            vec_env=vec_env,
        )

        # Strong pass: pooled (wins, games) vs the trained-against opponent.
        strong_wins, strong_games = _pooled_win_counts(
            gate_path,
            gate_tracks=gate_tracks,
            per_track_games=per_track_games,
            num_players=num_players,
            seed=seed,
            opponent_factory=strong_factory,
        )
        win_rate_strong = strong_wins / strong_games if strong_games else 0.0

        # Weak pass (reporting): pooled win-rate vs the weak HeuristicAgent. When
        # the curriculum is not strong, the strong pass already IS the weak pass,
        # so reuse its numbers rather than paying for a second batch.
        if strong_factory is None:
            win_rate_weak = win_rate_strong
        else:
            weak_wins, weak_games = _pooled_win_counts(
                gate_path,
                gate_tracks=gate_tracks,
                per_track_games=per_track_games,
                num_players=num_players,
                seed=seed,
                opponent_factory=None,
            )
            win_rate_weak = weak_wins / weak_games if weak_games else 0.0

    if curriculum.gate_use_wilson_lb:
        wilson_lb_strong, _ = wilson_interval(strong_wins, strong_games)
    else:
        # Legacy point-estimate promotion: the LB field carries the point value
        # so promote_score stays a single comparable scalar.
        wilson_lb_strong = win_rate_strong

    return GateResult(
        win_rate_strong=win_rate_strong,
        win_rate_weak=win_rate_weak,
        wilson_lb_strong=wilson_lb_strong,
        games_strong=strong_games,
    )


def train_self_play(
    config: PPOConfig | None = None,
    curriculum: CurriculumConfig | None = None,
    *,
    num_players: int = 4,
    track: TrackSource | None = None,
    learner_id: int = 0,
    gate_fn=None,
    warm_start_path: str | None = None,
    phases: "list[TrainingPhase] | None" = None,
) -> tuple[MaskablePPO, str]:
    """Run the Phase 1 -> Phase 2 self-play curriculum with 6C stability (§6C).

    Phase 1 trains vs the scripted pool. Phase 2 periodically freezes the policy
    to a snapshot and rebuilds the (vec) env with a ramped, stochastic,
    multi-snapshot opponent mix. After each Phase-2 chunk the model is scored by
    an eval gate; the **canonical ``run_name`` checkpoint is overwritten only
    when the score strictly improves**, so a collapsing Phase 2 can never destroy
    the best model. The collapsed final policy is saved separately to
    ``run_name_final``.

    **Sprint 8C** -- when ``phases`` is supplied, the run is driven by that ordered
    :class:`TrainingPhase` list instead of the legacy Phase-1/Phase-2 split (see
    :func:`_train_phases`): a solo pretrain phase followed by a weak->mixed->strong
    opponent ramp, each phase running the same chunked Wilson-LB gate +
    best-checkpoint preservation, with a per-phase gamma handoff (the model is
    reloaded carrying weights with the phase's gamma, since SB3 bakes gamma at
    construction). ``warm_start_path`` (8C) loads an existing checkpoint instead
    of building fresh, so an externally-pretrained solo policy can seed the run.
    Both default to the legacy behavior (``phases=None`` -> the 6C path;
    ``warm_start_path=None`` -> ``build_model``), so existing runs are unchanged.

    Returns ``(model, best_checkpoint_path)`` — the best-evaluated checkpoint,
    not the (possibly collapsed) final model.

    Args:
        track: the track *source*. ``None`` (default) trains on **procedurally
            generated tracks** -- a fresh one per episode via :class:`TrackSampler`
            (§6A) -- and gates on a held-out generated set. Pass a fixed
            :class:`~heat.models.track.Track` to pin one specific track instead.
        gate_fn: optional override for the eval gate (``model -> float``); used
            by tests to inject a deterministic score sequence. Defaults to a
            win-rate gate over :func:`heat.ml.evaluate.evaluate_ml` (§2.1
            fallback — no 6B dependency).
    """
    if config is None:
        config = PPOConfig()
    if curriculum is None:
        curriculum = CurriculumConfig()

    # Sprint 8C: an explicit phase list drives the solo -> opponent-curriculum run
    # through a separate, self-contained loop. Default (None) keeps the legacy 6C
    # Phase-1/Phase-2 path below byte-for-byte unchanged.
    if phases is not None:
        return _train_phases(
            phases,
            config=config,
            curriculum=curriculum,
            track=track,
            learner_id=learner_id,
            gate_fn=gate_fn,
            warm_start_path=warm_start_path,
        )

    os.makedirs(curriculum.checkpoint_dir, exist_ok=True)
    seed = config.seed if config.seed is not None else 0

    # Default to generated tracks (a sampler); a fixed Track pins one track. With
    # the Sprint B curriculum on, this is a StepAwareTrackSampler the Phase-1
    # chunk loop advances via staged vec-env rebuilds.
    track_source = _resolve_track_source(track, seed, curriculum)

    best_path = os.path.join(curriculum.checkpoint_dir, curriculum.run_name)
    final_path = os.path.join(
        curriculum.checkpoint_dir, f"{curriculum.run_name}_final"
    )
    track_name = _track_label(track_source)
    norm_meta = _normalize_meta(curriculum)

    def _gate(m: MaskablePPO, venv: VecEnv | None = None) -> GateResult:
        if gate_fn is not None:
            # Tests inject ``gate_fn=lambda m: <float>`` (a model -> float). Wrap
            # the float into a GateResult so the rest of the loop (which keys off
            # ``promote_score`` + the strong/weak split) is unchanged.
            s = float(gate_fn(m))
            return GateResult(
                win_rate_strong=s,
                win_rate_weak=s,
                wilson_lb_strong=s,
                games_strong=0,
            )
        return _gate_score(
            m,
            curriculum=curriculum,
            num_players=num_players,
            track=track_source,
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

    # --- Phase 1: scripted opponents (chunked + periodically gated, Idea 7) ---
    apply_shaping_config(config)  # base shaping; the schedule overrides per-chunk
    scripted = _scripted_opponents(
        num_players,
        use_strong=curriculum.use_strong_heuristic_opponents,
        broaden_mix=curriculum.broaden_phase1_mix,
    )
    venv = _build_vec_env(
        track=track_source,
        num_players=num_players,
        opponents=scripted,
        learner_id=learner_id,
        config=config,
        curriculum=curriculum,
        shaping_weight=config.shaping_weight,
        seed=seed,
    )
    # Sprint 8C: warm-start from an existing checkpoint (carrying weights) instead
    # of building fresh, when a path is given. Reuses the gamma-carrying reload
    # helper; falls back to build_model for the default (no warm start) case.
    if warm_start_path is not None:
        model = _load_model_with_gamma(
            warm_start_path, venv, config, gamma=config.gamma
        )
    else:
        model = build_model(venv, config)

    # SB3 ``load`` restores the checkpoint's saved ``verbose`` -- a warm-start BC
    # checkpoint was saved with ``verbose=0``, which silences the fine-tune's
    # rollout tables and makes a 40-min run look hung. Re-apply the live config's
    # verbosity so a warm-started run shows the same SB3 progress a fresh run does
    # (no-op on the build_model path, which already used config.verbose).
    model.verbose = config.verbose

    # Chunk Phase 1 into ``phase1_eval_every`` learn/gate iterations so a
    # Phase-1-heavy run (Idea 4) still preserves the BEST held-out checkpoint, not
    # the (possibly regressed) final weights. After each chunk: run the held-out
    # gate, log the strong/weak split + promotion score to TensorBoard, and
    # overwrite the canonical best checkpoint only when ``promote_score`` strictly
    # improves (Idea 9 Wilson LB). ``best_score`` is the running best promotion
    # score; ``None`` until the first gate so the first chunk always saves.
    best_score: float | None = None
    p1_remaining = curriculum.phase1_steps
    p1_step = max(1, curriculum.phase1_eval_every)
    p1_chunk_idx = 0

    # Sprint B step-aware curriculum (Idea 2): when the training source is a
    # StepAwareTrackSampler, the global step does NOT reach SubprocVecEnv workers
    # through the pickled sampler. So we drive difficulty by REBUILDING the
    # training vec env at curriculum stage boundaries with a fresh, step-pinned
    # sampler (process-safe staged rebuilds). The gate's ``track_source`` keeps
    # the original sampler -- gating always uses the fixed full-difficulty
    # held-out set (``_gate_tracks``), so the gate is comparable across stages.
    use_curriculum = isinstance(track_source, StepAwareTrackSampler)
    cur_stage = 0
    cur_horizon = curriculum.curriculum_horizon_steps
    cur_stages = max(1, curriculum.curriculum_stages)
    # The Phase-1 env was already built from ``track_source`` (a
    # StepAwareTrackSampler pinned to step 0), so it starts at stage 0 -- no
    # initial rebuild needed; stage rebuilds happen on boundary crossings below.

    total_so_far = 0
    while p1_remaining > 0:
        chunk = min(p1_step, p1_remaining)
        model.learn(
            total_timesteps=chunk,
            progress_bar=False,
            reset_num_timesteps=(p1_chunk_idx == 0),
        )
        p1_remaining -= chunk
        p1_chunk_idx += 1
        total_so_far += chunk

        # Advance the curriculum: if this chunk crossed a stage boundary, rebuild
        # the training env with a harder step-pinned sampler and swap it in.
        if use_curriculum:
            new_stage = _curriculum_stage_index(
                total_so_far, cur_horizon, cur_stages
            )
            if new_stage > cur_stage and p1_remaining > 0:
                cur_stage = new_stage
                staged_sampler = StepAwareTrackSampler(
                    track_source.schedule,
                    base_seed=seed,
                    step=_curriculum_stage_step(
                        new_stage, cur_horizon, cur_stages
                    ),
                )
                new_venv = _build_vec_env(
                    track=staged_sampler,
                    num_players=num_players,
                    opponents=scripted,
                    learner_id=learner_id,
                    config=config,
                    curriculum=curriculum,
                    shaping_weight=config.shaping_weight,
                    seed=seed,
                )
                venv = _swap_vec_env(model, venv, new_venv)
                model.logger.record("curriculum/stage", new_stage)

        score = _gate(model, venv)
        model.logger.record("eval/gate_score", score.promote_score)
        model.logger.record("eval/win_rate_strong", score.win_rate_strong)
        model.logger.record("eval/win_rate_weak", score.win_rate_weak)
        if best_score is None or score.promote_score > best_score:
            best_score = score.promote_score
            _save_best(model, venv)

    # Guarantee at least one best save even if phase1_steps < phase1_eval_every
    # (or phase1_steps == 0): this checkpoint is the safety net (§2.1) -- even a
    # fully collapsing Phase 2 cannot overwrite it with a worse one.
    if best_score is None:
        best_score = _gate(model, venv).promote_score
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

        scripted_cls: type[BaseAgent] = (
            StrongHeuristicAgent
            if curriculum.use_strong_heuristic_opponents
            else HeuristicAgent
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
                scripted_cls=scripted_cls,
            )
        else:
            opponents = _mixed_opponent_pool(
                num_players,
                snapshot_paths,
                mix,
                deterministic=curriculum.snapshot_deterministic,
                scripted_cls=scripted_cls,
            )

        # Rebuild the (vec) env on the new mix, preserving VecNormalize stats.
        new_venv = _build_vec_env(
            track=track_source,
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
        # when the promotion score STRICTLY improves. Log the gate score (and the
        # Sprint-A strong/weak split) for TensorBoard so a collapse is visible
        # live (§3.2).
        result = _gate(model, venv)
        promote = result.promote_score
        model.logger.record("eval/gate_score", promote)
        model.logger.record("eval/win_rate_strong", result.win_rate_strong)
        model.logger.record("eval/win_rate_weak", result.win_rate_weak)
        model.logger.record("eval/best_score", best_score)
        if promote > best_score:
            best_score = promote
            _save_best(model, venv)
        elif (
            curriculum.reload_best_on_regression
            and promote < best_score - curriculum.regression_tol
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
                        # PFSP bookkeeping needs a concrete track; use a fixed
                        # representative (the pinned track, or a held-out
                        # generated one) so the win-rate signal is comparable.
                        track=_gate_tracks(track_source)[0],
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


# ---------------------------------------------------------------------------
# Sprint 8C: multi-phase driver (solo pretrain + opponent curriculum)
# ---------------------------------------------------------------------------


def _load_model_with_gamma(
    path: str, venv: VecEnv, config: PPOConfig, *, gamma: float
) -> MaskablePPO:
    """Load an SB3 checkpoint onto ``venv``, overriding its baked-in ``gamma``.

    SB3 bakes ``gamma`` into the model at construction; ``set_env`` / ``learn`` do
    NOT change it. Empirically (verified for this sb3-contrib version),
    ``MaskablePPO.load(path, env=venv, custom_objects={"gamma": gamma})`` loads
    the saved weights AND adopts the new ``gamma`` -- so a phase boundary that
    needs a different discount (solo 0.99 -> race 0.999, §6.3) is handled by a
    single reload that carries the weights forward. The shaping/reward-mode
    globals are re-applied afterwards so this process computes the right reward.
    """
    # ``custom_objects`` overrides values baked into the saved checkpoint at load
    # time. We override ``gamma`` (the discount handoff) AND ``verbose`` -- a
    # warm-start BC checkpoint was saved with ``verbose=0``, and without this each
    # phase reload would re-silence SB3's rollout tables, making a multi-phase
    # fine-tune look hung for its whole run.
    model = MaskablePPO.load(
        path,
        env=venv,
        device=resolve_device(config.device),
        custom_objects={"gamma": gamma, "verbose": config.verbose},
    )
    model.verbose = config.verbose
    apply_shaping_config(config)
    return model


def _train_phases(
    phases: "list[TrainingPhase]",
    *,
    config: PPOConfig,
    curriculum: CurriculumConfig,
    track: TrackSource | None,
    learner_id: int,
    gate_fn=None,
    warm_start_path: str | None = None,
) -> tuple[MaskablePPO, str]:
    """Drive an ordered :class:`TrainingPhase` list (Sprint 8C; §5, §6).

    Each phase runs the chunked learn/gate loop (reusing Sprint A's Wilson-LB
    gate + best-checkpoint preservation): the env is (re)built for the phase's
    opponent pool / reward mode / shaping, and -- because SB3 bakes ``gamma`` at
    construction -- the model is RELOADED carrying weights with the phase's gamma
    whenever it differs from the running model's (the §6.3/§10 gamma handoff,
    resolved via :func:`_load_model_with_gamma`). ``best_score`` is a single
    running scalar across ALL phases, so a collapsing later phase can never
    overwrite a better earlier checkpoint.

    Gate yardstick (§6.4/§10): the gate ALWAYS scores against the fixed final
    strong 4-player target (a single coherent scale across the ramp). The SOLO
    phase is NOT gated against that race yardstick (it is OOD -- a 1-player
    time-trial policy scored in a 4-player race is meaningless, §10 last bullet);
    the solo phase trains ungated and the race-yardstick gate starts at the first
    opponent phase.

    Returns ``(model, best_checkpoint_path)`` -- the best-evaluated checkpoint.
    """
    if not phases:
        raise ValueError("_train_phases requires a non-empty phase list")

    os.makedirs(curriculum.checkpoint_dir, exist_ok=True)
    seed = config.seed if config.seed is not None else 0
    # The opponent curriculum demotes the track curriculum to off; resolve the
    # track source with curriculum=None so a plain generated-track sampler (or a
    # pinned Track) is used regardless of the (legacy) track-curriculum flag.
    track_source = _resolve_track_source(track, seed, curriculum=None)
    track_name = _track_label(track_source)
    norm_meta = _normalize_meta(curriculum)

    best_path = os.path.join(curriculum.checkpoint_dir, curriculum.run_name)
    final_path = os.path.join(
        curriculum.checkpoint_dir, f"{curriculum.run_name}_final"
    )

    # The fixed final-strong yardstick (§6.4): gate every opponent phase against
    # the LAST race phase's player count + the strong pool, so promote_score is a
    # single comparable scale. Built from the last non-solo phase (fallback 4p).
    race_phases = [p for p in phases if p.reward_mode != "solo"]
    yardstick_players = race_phases[-1].num_players if race_phases else 4

    def _gate(m: MaskablePPO, venv: VecEnv | None = None) -> GateResult:
        if gate_fn is not None:
            s = float(gate_fn(m))
            return GateResult(
                win_rate_strong=s,
                win_rate_weak=s,
                wilson_lb_strong=s,
                games_strong=0,
            )
        # Always gate vs the fixed final-strong yardstick (§6.4), independent of
        # the current phase's opponent pool.
        return _gate_score(
            m,
            curriculum=curriculum,
            num_players=yardstick_players,
            track=track_source,
            seed=seed,
            vec_env=venv,
            normalize=norm_meta,
        )

    def _save_best(m: MaskablePPO, venv: VecEnv) -> None:
        save_checkpoint(
            m,
            best_path,
            track_name=track_name,
            num_players=yardstick_players,
            seed=seed,
            ppo_config=config,
            curriculum_config=curriculum,
            track_config={"track_name": track_name},
            normalize=norm_meta,
            vec_env=venv,
        )

    schedule = OpponentSchedule(
        stages=tuple(
            OpponentStage(p.pool_kind, p.steps)
            for p in phases
            if p.pool_kind is not None
        )
    )

    best_score: float | None = None
    model: MaskablePPO | None = None
    venv: VecEnv | None = None
    cur_gamma: float | None = None
    first_learn = True  # reset_num_timesteps only on the very first learn (§11)
    opp_stage_idx = 0  # index into the opponent schedule (solo phases skip it)

    for phase_idx, phase in enumerate(phases):
        # Resolve the phase's opponent pool. The solo phase has no opponents.
        if phase.pool_kind is None:
            opponents = None
        else:
            opponents = schedule.pool_for(phase.num_players, opp_stage_idx)
            opp_stage_idx += 1

        new_venv = _build_vec_env(
            track=track_source,
            num_players=phase.num_players,
            opponents=opponents,
            learner_id=learner_id if phase.num_players > learner_id else 0,
            config=config,
            curriculum=curriculum,
            shaping_weight=phase.shaping_weight,
            seed=seed + 1000 + phase_idx,
            reward_mode=phase.reward_mode,
            solo_finish_bonus=config.solo_finish_bonus,
        )

        if model is None:
            # First phase: warm-start (carrying gamma) or build fresh.
            if warm_start_path is not None:
                model = _load_model_with_gamma(
                    warm_start_path, new_venv, config, gamma=phase.gamma
                )
            else:
                phase_cfg = dataclasses.replace(
                    config,
                    gamma=phase.gamma,
                    reward_mode=phase.reward_mode,
                    shaping_weight=phase.shaping_weight,
                )
                model = build_model(new_venv, phase_cfg)
            venv = new_venv
            cur_gamma = phase.gamma
        elif phase.gamma != cur_gamma:
            # Gamma boundary (§6.3): SB3 bakes gamma at construction, so reload
            # the current weights with the new gamma onto the new env (carrying
            # weights forward). Save the live weights to a temp checkpoint first.
            assert venv is not None
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                handoff = os.path.join(tmp, "phase_handoff")
                model.save(handoff)
                old_venv = venv
                model = _load_model_with_gamma(
                    handoff, new_venv, config, gamma=phase.gamma
                )
                # Carry VecNormalize stats across the reload, then drop old env.
                if isinstance(old_venv, VecNormalize) and isinstance(
                    new_venv, VecNormalize
                ):
                    if (
                        "obs_rms" in old_venv.__dict__
                        and "obs_rms" in new_venv.__dict__
                    ):
                        new_venv.obs_rms = old_venv.__dict__["obs_rms"]
                    if (
                        "ret_rms" in old_venv.__dict__
                        and "ret_rms" in new_venv.__dict__
                    ):
                        new_venv.ret_rms = old_venv.__dict__["ret_rms"]
                if old_venv is not new_venv:
                    old_venv.close()
                venv = new_venv
            cur_gamma = phase.gamma
        else:
            # Same gamma: just swap the env (carries VecNormalize stats).
            assert venv is not None
            venv = _swap_vec_env(model, venv, new_venv)

        # Re-apply the phase's reward/shaping globals in THIS process (the gate +
        # any DummyVecEnv path read them here, not just spawned workers).
        apply_shaping_config(
            dataclasses.replace(
                config,
                reward_mode=phase.reward_mode,
                shaping_weight=phase.shaping_weight,
                solo_finish_bonus=config.solo_finish_bonus,
            )
        )

        # Chunk the phase into learn/gate iterations (§6.1). The SOLO phase is
        # ungated against the race yardstick (§10 last bullet): it trains in one
        # uninterrupted block. Opponent phases gate after every chunk.
        is_solo = phase.reward_mode == "solo"
        chunk_steps = max(1, curriculum.phase1_eval_every)
        remaining = phase.steps
        while remaining > 0:
            chunk = phase.steps if is_solo else min(chunk_steps, remaining)
            model.learn(
                total_timesteps=chunk,
                progress_bar=False,
                reset_num_timesteps=first_learn,
            )
            first_learn = False
            remaining -= chunk

            model.logger.record("phase/index", phase_idx)
            if is_solo:
                # No race-yardstick gate during solo; one block then move on.
                break

            result = _gate(model, venv)
            model.logger.record("eval/gate_score", result.promote_score)
            model.logger.record("eval/win_rate_strong", result.win_rate_strong)
            model.logger.record("eval/win_rate_weak", result.win_rate_weak)
            if best_score is None or result.promote_score > best_score:
                best_score = result.promote_score
                _save_best(model, venv)

    assert model is not None and venv is not None

    # Guarantee at least one best save (e.g. a solo-only phase list never gated).
    if best_score is None:
        best_score = _gate(model, venv).promote_score
        _save_best(model, venv)

    # The (possibly collapsed) final model goes to a SEPARATE path.
    save_checkpoint(
        model,
        final_path,
        track_name=track_name,
        num_players=yardstick_players,
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
    scripted_cls: type[BaseAgent] = HeuristicAgent,
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
    specs.extend(scripted_cls for _ in range(n_opp - n_snap))
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
    scripted_cls: type[BaseAgent] = HeuristicAgent,
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
    specs.extend(scripted_cls for _ in range(n_opp - len(sampled_paths)))
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
