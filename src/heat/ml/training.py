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

import json
import os
from dataclasses import dataclass

import numpy as np

from sb3_contrib import MaskablePPO

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
from heat.ml.env import HeatEnv
from heat.ml.features import encode_observation
from heat.ml.model import PPOConfig, build_model

#: Suffix for the SB3 archive and the JSON sidecar.
_META_SUFFIX = ".meta.json"


# ---------------------------------------------------------------------------
# Checkpoint save/load (§3.4)
# ---------------------------------------------------------------------------


def meta_path_for(path: str) -> str:
    """Return the sidecar metadata path for a checkpoint ``path`` (the SB3 zip).

    SB3 appends ``.zip`` to ``path`` if missing; the sidecar sits next to the
    archive as ``<path>.meta.json`` (mirroring the un-suffixed ``model.save``
    argument so 5d can locate it from the same base path).
    """
    base = path[:-4] if path.endswith(".zip") else path
    return base + _META_SUFFIX


def save_checkpoint(
    model: MaskablePPO,
    path: str,
    *,
    track_name: str,
    num_players: int,
) -> str:
    """Save ``model`` to ``path`` (SB3 zip) plus the ``.meta.json`` sidecar (§3.4).

    Returns the sidecar path. ``obs_dim``/``action_dim``/``codec_version`` come
    from :mod:`heat.ml.spaces` so the sidecar always reflects the live contract.
    """
    model.save(path)
    meta = {
        "obs_dim": spaces.OBS_DIM,
        "action_dim": spaces.ACTION_DIM,
        "codec_version": spaces.CODEC_VERSION,
        "track_name": track_name,
        "num_players": num_players,
    }
    sidecar = meta_path_for(path)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
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
    """Self-play curriculum settings (§5c).

    ``phase1_steps`` are trained vs the scripted pool; the remaining
    ``total_timesteps - phase1_steps`` are Phase 2, where after every
    ``snapshot_every`` timesteps the current policy is frozen to a checkpoint
    and added to the opponent pool (capped at ``max_snapshots``, FIFO).
    """

    total_timesteps: int = 200_000
    phase1_steps: int = 100_000
    snapshot_every: int = 50_000
    max_snapshots: int = 3
    #: Fraction of opponent seats drawn from the frozen snapshot pool in Phase 2.
    snapshot_mix: float = 0.5
    #: Directory checkpoints (and snapshots) are written to.
    checkpoint_dir: str = "checkpoints"
    #: Base name for the final checkpoint.
    run_name: str = "heat_ppo"


def _scripted_opponents(num_players: int) -> list[type[BaseAgent]]:
    """Phase-1 opponent factories: mostly HeuristicAgent, one RandomAgent."""
    n_opp = num_players - 1
    pool: list[type[BaseAgent]] = [HeuristicAgent] * n_opp
    if n_opp >= 1:
        pool[-1] = RandomAgent  # inject some exploration pressure
    return pool


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def _make_env(
    track: Track | None,
    num_players: int,
    opponents,
    learner_id: int,
) -> HeatEnv:
    return HeatEnv(
        track=track,
        num_players=num_players,
        opponents=opponents,
        learner_id=learner_id,
    )


def smoke_train(
    *,
    total_timesteps: int = 256,
    num_players: int = 2,
    track: Track | None = None,
    checkpoint_path: str | None = None,
    config: PPOConfig | None = None,
) -> tuple[MaskablePPO, str | None]:
    """Tiny end-to-end training run for the slow smoke test (§5c, §6.2).

    Trains a small net for ``total_timesteps`` steps against a scripted opponent
    pool, optionally saving a checkpoint (+ sidecar). Returns
    ``(model, checkpoint_path_or_None)``. Keeps everything small/fast: no
    self-play phase, CPU, tiny net unless ``config`` overrides.
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

    opponents = _scripted_opponents(num_players)
    env = _make_env(track, num_players, opponents, learner_id=0)

    model = build_model(env, config)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    saved: str | None = None
    if checkpoint_path is not None:
        save_checkpoint(
            model,
            checkpoint_path,
            track_name=env.track.name,
            num_players=num_players,
        )
        saved = checkpoint_path

    return model, saved


def train_self_play(
    config: PPOConfig | None = None,
    curriculum: CurriculumConfig | None = None,
    *,
    num_players: int = 4,
    track: Track | None = None,
    learner_id: int = 0,
) -> tuple[MaskablePPO, str]:
    """Run the full Phase 1 -> Phase 2 self-play curriculum (§5c).

    Phase 1 trains vs the scripted pool. Phase 2 periodically freezes the policy
    to a checkpoint and rebuilds the env with a mixed opponent pool (scripted +
    frozen snapshots loaded by path). Returns ``(model, final_checkpoint_path)``.

    Snapshots are added as :class:`FrozenSnapshotAgent` *factories* bound to a
    checkpoint path, so the env's opponent specs stay picklable (§6.5).
    """
    if config is None:
        config = PPOConfig()
    if curriculum is None:
        curriculum = CurriculumConfig()

    os.makedirs(curriculum.checkpoint_dir, exist_ok=True)

    # --- Phase 1: scripted opponents ---
    scripted = _scripted_opponents(num_players)
    env = _make_env(track, num_players, scripted, learner_id)
    model = build_model(env, config)

    track_name = env.track.name
    model.learn(total_timesteps=curriculum.phase1_steps, progress_bar=False)

    # --- Phase 2: self-play with frozen snapshots mixed in ---
    snapshot_paths: list[str] = []
    remaining = max(0, curriculum.total_timesteps - curriculum.phase1_steps)
    step = max(1, curriculum.snapshot_every)
    snap_idx = 0

    trained = curriculum.phase1_steps
    while remaining > 0:
        # Freeze the current policy and register it as an opponent.
        snap_path = os.path.join(
            curriculum.checkpoint_dir, f"{curriculum.run_name}_snap{snap_idx}"
        )
        save_checkpoint(
            model, snap_path, track_name=track_name, num_players=num_players
        )
        snapshot_paths.append(snap_path)
        if len(snapshot_paths) > curriculum.max_snapshots:
            snapshot_paths.pop(0)  # FIFO cap
        snap_idx += 1

        # Rebuild the opponent pool: mix scripted + snapshot factories.
        opponents = _mixed_opponent_pool(
            num_players, snapshot_paths, curriculum.snapshot_mix
        )
        env = _make_env(track, num_players, opponents, learner_id)
        model.set_env(env)

        chunk = min(step, remaining)
        model.learn(
            total_timesteps=chunk,
            progress_bar=False,
            reset_num_timesteps=False,
        )
        remaining -= chunk
        trained += chunk

    final_path = os.path.join(curriculum.checkpoint_dir, curriculum.run_name)
    save_checkpoint(
        model, final_path, track_name=track_name, num_players=num_players
    )
    return model, final_path


def _mixed_opponent_pool(
    num_players: int, snapshot_paths: list[str], snapshot_mix: float
):
    """Build an opponent-spec list mixing scripted agents and frozen snapshots.

    Snapshot seats are :class:`FrozenSnapshotAgent` factories bound to a path
    (picklable, §6.5); the rest are ``HeuristicAgent``.
    """
    n_opp = num_players - 1
    n_snap = min(n_opp, int(round(n_opp * snapshot_mix))) if snapshot_paths else 0

    specs: list = []
    for i in range(n_snap):
        path = snapshot_paths[i % len(snapshot_paths)]
        specs.append(lambda p=path: FrozenSnapshotAgent(p))
    specs.extend(HeuristicAgent for _ in range(n_opp - n_snap))
    return specs
