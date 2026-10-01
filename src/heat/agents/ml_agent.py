"""Checkpoint-backed RL agent for HEAT (Sprint 5d).

:class:`MLAgent` is a :class:`~heat.agents.base.BaseAgent` that acts using a
trained SB3 ``MaskablePPO`` checkpoint on disk. It drops into the existing
pull-driven ``Game`` loop, the ``simulation`` runner, and the viewer unchanged.

Acting path (§3.5)
------------------
``BaseAgent``'s five ``choose_*`` methods are *pull* methods: they receive the
*already-enumerated* legal options (not a ``Decision``). For each call, the
agent reconstructs the matching :class:`~heat.engine.driver.Decision`, encodes
the observation, predicts a flat action under the legal mask, and decodes it
back to the concrete engine action -- which is always legal.

This mirrors :class:`heat.ml.training.FrozenSnapshotAgent` (the self-play
opponent) on purpose; the small amount of pull->Decision->predict->decode logic
is replicated here rather than shared, to keep the frozen 5c ``training.py``
behavior untouched (per the 5d scope constraint). The one thing ``MLAgent``
adds over ``FrozenSnapshotAgent`` is the §3.4 sidecar tripwire below.

Meta tripwire (§3.4)
--------------------
On (lazy) model load, ``MLAgent`` reads the checkpoint's ``.meta.json`` sidecar
and asserts ``obs_dim``/``action_dim``/``codec_version`` match the live
:mod:`heat.ml.spaces` contract. A stale checkpoint -- one trained against a now
drifted feature/action layout -- fails fast with a clear error instead of
silently emitting garbage actions.

Picklability (§6.5)
-------------------
The heavy SB3 model is loaded lazily from ``model_path`` and nulled in
``__getstate__`` so the agent pickles by path only. This lets an ``MLAgent``
factory ship to ``ProcessPoolExecutor`` workers (each worker reloads the model),
though the eval harness defaults to sequential -- see :mod:`heat.ml.evaluate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from sb3_contrib import MaskablePPO

from heat.agents.base import BaseAgent
from heat.engine import rules
from heat.engine.driver import Decision, DecisionKind
from heat.engine.phases import ReactDecision
from heat.models.cards import Card
from heat.models.game_state import GameState
import os

from heat.ml.action_codec import decode_action, legal_action_mask
from heat.ml.features import encode_observation
from heat.ml.training import (
    CheckpointMismatchError as CheckpointMismatchError,
    load_compatible_meta,
    load_meta,
    vecnorm_path_for,
)

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecNormalize


class MLAgent(BaseAgent):
    """A :class:`BaseAgent` driven by a trained SB3 ``MaskablePPO`` checkpoint.

    Args:
        model_path: Path to the SB3 ``.zip`` checkpoint (with or without the
            ``.zip`` suffix). The ``.meta.json`` sidecar must sit alongside it
            (written by :func:`heat.ml.training.save_checkpoint`).
        deterministic: If True (default), take the arg-max action; otherwise
            sample from the masked policy distribution.
        name: Display name for the agent.

    The model is loaded lazily on the first decision (and validated against the
    live contract at that point), then cached on the instance.
    """

    def __init__(
        self,
        model_path: str,
        *,
        deterministic: bool = True,
        name: str = "MLAgent",
    ) -> None:
        super().__init__(name=name)
        self.model_path = model_path
        self.deterministic = deterministic
        self._model: MaskablePPO | None = None
        self._codec_version: int | None = None
        #: Loaded ``VecNormalize`` (obs stats only), or ``None`` for an
        #: un-normalized checkpoint (back-compat with Sprint-5 models).
        self._vecnorm: VecNormalize | None = None
        #: Whether the checkpoint was trained with obs normalization. Set on load.
        self._norm_obs = False

    # -- lazy model access with the §3.4 meta tripwire --
    def _validate_meta(self) -> None:
        """Reject a checkpoint whose sidecar does not match the runtime codec."""
        meta = load_compatible_meta(self.model_path)
        self._codec_version = int(meta["codec_version"])

    def _get_model(self) -> MaskablePPO:
        if self._model is None:
            self._validate_meta()
            self._model = MaskablePPO.load(self.model_path, device="cpu")
            self._load_vecnorm()
        return self._model

    def _load_vecnorm(self) -> None:
        """Load the optional ``VecNormalize`` stats sidecar and apply it (§2.6).

        A normalized checkpoint saw normalized **observations**; feeding it raw
        obs would be silent garbage. So when a ``<path>.vecnorm.pkl`` exists and
        the meta ``normalize`` block declares ``norm_obs=True``, load the running
        ``obs_rms`` and apply it before ``predict``. Reward normalization affects
        only the critic, never action selection, so it is ignored at inference.

        Back-compat: no stats file (or ``norm_obs`` off / absent) means "no
        normalization" — exactly the Sprint-5 behavior, not an error.
        """
        try:
            meta = load_meta(self.model_path)
        except FileNotFoundError:  # pragma: no cover - tripwire ran first
            meta = {}
        norm_block = meta.get("normalize") or {}
        self._norm_obs = bool(norm_block.get("norm_obs", False))

        stats_path = vecnorm_path_for(self.model_path)
        if not self._norm_obs or not os.path.exists(stats_path):
            self._vecnorm = None
            self._norm_obs = False
            return

        # Lazy import: VecNormalize lives in the optional [ml] extra. SB3's
        # VecNormalize.load needs a venv to derive observation/action spaces
        # from; supply a throwaway zero-step env with the contract spaces (we
        # only ever call normalize_obs, never step it).
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecNormalize,
        )

        from heat.ml.env import HeatEnv

        dummy = DummyVecEnv([lambda: HeatEnv(num_players=2)])
        self._vecnorm = VecNormalize.load(stats_path, venv=dummy)
        # Inference only: freeze the running stats and disable reward norm.
        self._vecnorm.training = False
        self._vecnorm.norm_reward = False

    def __getstate__(self) -> dict[str, object]:
        # Never pickle the heavy SB3 model / stats; reload from path in the
        # worker (§6.5). The vecnorm sidecar is reloaded alongside the model.
        state = self.__dict__.copy()
        state["_model"] = None
        state["_vecnorm"] = None
        return state

    # -- shared predict path: encode obs, mask, predict, decode --
    def _predict_flat(self, decision: Decision, state: GameState) -> int:
        model = self._get_model()
        assert self._codec_version is not None
        obs = encode_observation(
            state,
            decision.player_id,
            decision,
            codec_version=self._codec_version,
        )
        mask = legal_action_mask(decision, state)
        if self._norm_obs and self._vecnorm is not None:
            # Apply the SAME obs normalization the policy was trained under.
            obs = cast(np.ndarray, self._vecnorm.normalize_obs(obs))
        action, _ = model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        return int(np.asarray(action).reshape(-1)[0])

    def _choose(self, decision: Decision, state: GameState) -> object:
        flat = self._predict_flat(decision, state)
        return decode_action(decision, flat, state)

    # -- BaseAgent pull-methods: rebuild a Decision, then predict + decode --
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
