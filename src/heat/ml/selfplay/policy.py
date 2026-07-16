"""Swappable policy interface for the custom PPO loop (Sprint A0).

:class:`HeatPolicy` is the contract the trainer (:mod:`heat.ml.selfplay.ppo`)
codes against; A3/A4 re-implement only its *internals* (the trunk, the action
head, the encoder) while keeping the ``act`` / ``evaluate`` signatures and return
tuples below frozen. That stability is the whole point of A0: the loop never
needs to change when the architecture does.

A0 implementation
-----------------
A small MLP trunk over the flat ``OBS_DIM`` observation produces (a) policy
logits over the fixed ``ACTION_DIM`` union action space and (b) a value scalar.
Legality is enforced by ``logits.masked_fill(~mask, -inf)`` before building a
:class:`torch.distributions.Categorical`, so **no illegal action can ever
receive probability mass** (the smoke test asserts this against
:func:`heat.ml.action_codec.legal_action_mask`).

When A3 lands, ``mask`` generalizes from "the boolean legal-action mask over a
fixed head" to "the set of currently-legal action feature vectors", and the head
becomes a dot-product over that variable set -- but ``act`` / ``evaluate`` keep
exactly these shapes and return tuples, so the trainer is untouched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.distributions import Categorical

from heat.ml.selfplay.action_features import ACTION_FEAT_DIM, action_feature_table
from heat.ml.spaces import (
    ACTION_DIM,
    BLOCK_HAND_HISTOGRAM,
    OBS_DIM,
)

if TYPE_CHECKING:
    from heat.ml.selfplay.ppo import A0Config


CARD_EMBED_DIM = 32
"""Width of A4's learned card-token embeddings."""


class StructuredObservationEncoder(nn.Module):
    """A4 encoder separating the private own hand from public state.

    The frozen observation contract already has the information boundary A4
    needs: indices ``0:8`` are the acting seat's private hand histogram and the
    remainder contains own/public state, including only public opponent fields.
    The histogram stores counts divided by seven.  We recover those counts and
    use them as multiplicities in attention over eight learned card-token
    embeddings.  Adding ``log(count)`` to a token's attention score is exactly
    equivalent to expanding that token into ``count`` identical card slots, but
    avoids a per-batch ragged representation.

    The public block is encoded by its own MLP.  Hand and public embeddings are
    fused only after their separate encoders, preserving a visible architectural
    boundary that the leakage tests can audit.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        card_embed_dim: int = CARD_EMBED_DIM,
    ) -> None:
        super().__init__()
        if obs_dim < BLOCK_HAND_HISTOGRAM:
            raise ValueError(
                f"obs_dim {obs_dim} is smaller than hand block "
                f"{BLOCK_HAND_HISTOGRAM}"
            )
        if not hidden_sizes:
            raise ValueError("structured encoder requires at least one hidden size")

        self.obs_dim = obs_dim
        self.output_dim = hidden_sizes[-1]
        self.card_embeddings = nn.Embedding(BLOCK_HAND_HISTOGRAM, card_embed_dim)
        self.hand_query = nn.Parameter(torch.empty(card_embed_dim))
        self.empty_hand = nn.Parameter(torch.empty(card_embed_dim))

        public_layers: list[nn.Module] = []
        public_prev = obs_dim - BLOCK_HAND_HISTOGRAM
        for width in hidden_sizes:
            public_layers.append(nn.Linear(public_prev, width))
            public_layers.append(nn.Tanh())
            public_prev = width
        self.public_encoder = nn.Sequential(*public_layers)
        self.fusion = nn.Sequential(
            # Keep the raw multiplicities alongside attention pooling.  Pure
            # softmax attention represents composition but would map one S3 and
            # two S3 cards to the same vector; the histogram preserves that
            # strategically important count information exactly.
            nn.Linear(
                public_prev + card_embed_dim + BLOCK_HAND_HISTOGRAM,
                self.output_dim,
            ),
            nn.Tanh(),
        )

        nn.init.normal_(self.hand_query, std=1.0 / math.sqrt(card_embed_dim))
        nn.init.normal_(self.empty_hand, std=1.0 / math.sqrt(card_embed_dim))

    def _encode_hand(self, hand_histogram: torch.Tensor) -> torch.Tensor:
        """Attention-pool the private hand histogram into ``(B, E)``."""
        # The feature contract stores integer card counts / 7.  Rounding makes
        # the reconstruction robust to float32 representation noise.
        counts = torch.round(hand_histogram.clamp(0.0, 1.0) * 7.0)
        token_emb = self.card_embeddings.weight  # (8, E)
        scores = (token_emb @ self.hand_query) / math.sqrt(token_emb.shape[1])
        scores = scores.unsqueeze(0).expand(counts.shape[0], -1)
        present = counts > 0
        weighted_scores = scores + torch.log(counts.clamp_min(1.0))
        neg_inf = torch.finfo(weighted_scores.dtype).min
        weights = torch.softmax(weighted_scores.masked_fill(~present, neg_inf), dim=-1)
        pooled = weights @ token_emb

        # All-masked softmax is numerically defined here because finite min is
        # used, but it would produce a meaningless uniform mixture.  Empty hands
        # receive an explicit learned representation instead.
        empty = ~present.any(dim=-1)
        return torch.where(empty.unsqueeze(-1), self.empty_hand, pooled)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the fused structured state embedding for ``obs``."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected observation width {self.obs_dim}, got {obs.shape[-1]}"
            )
        hand = self._encode_hand(obs[:, :BLOCK_HAND_HISTOGRAM])
        public = self.public_encoder(obs[:, BLOCK_HAND_HISTOGRAM:])
        histogram = obs[:, :BLOCK_HAND_HISTOGRAM]
        return cast(
            torch.Tensor,
            self.fusion(torch.cat((hand, histogram, public), dim=-1)),
        )


def _build_trunk(
    obs_dim: int,
    hidden_sizes: tuple[int, ...],
    encoder: str,
) -> tuple[nn.Module, int]:
    """Build the selected observation trunk and return it plus output width."""
    if encoder == "structured":
        structured = StructuredObservationEncoder(obs_dim, hidden_sizes)
        return structured, structured.output_dim
    if encoder != "flat":
        raise ValueError(
            f"unknown encoder {encoder!r} (expected 'flat' or 'structured')"
        )

    layers: list[nn.Module] = []
    prev = obs_dim
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.Tanh())
        prev = width
    return nn.Sequential(*layers), prev


class HeatPolicy(nn.Module):
    """Masked-categorical actor-critic over the flat HEAT observation (A0).

    A shared MLP trunk feeds two linear heads: a policy head emitting one logit
    per action in the fixed ``ACTION_DIM`` union space, and a value head emitting
    a single state-value scalar. The trainer calls :meth:`act` during rollout
    collection and :meth:`evaluate` during the PPO update; both apply the legal-
    action ``mask`` so illegal actions never receive probability mass.

    Args:
        obs_dim: observation width (defaults to the frozen ``OBS_DIM``).
        action_dim: action-space width (defaults to the frozen ``ACTION_DIM``).
        hidden_sizes: trunk hidden-layer widths. A small default keeps the smoke
            test fast; real runs pass a larger trunk.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        encoder: str = "flat",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = encoder
        self.trunk, prev = _build_trunk(obs_dim, hidden_sizes, encoder)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    # ------------------------------------------------------------------
    # Internal: build the masked action distribution + value for a batch.
    # ------------------------------------------------------------------

    def _distribution_and_value(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Return ``(masked Categorical, value (B,))`` for a batch of obs.

        Illegal logits are driven to ``-inf`` before the softmax so the
        resulting distribution places exactly zero mass on them. ``mask`` is a
        boolean tensor where ``True`` marks a *legal* action.
        """
        latent = self.trunk(obs)
        logits = self.policy_head(latent)
        # masked_fill(~mask, -inf): illegal actions get -inf logits -> 0 prob.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, neg_inf)
        dist = Categorical(logits=masked_logits)
        value = self.value_head(latent).squeeze(-1)
        return dist, value

    # ------------------------------------------------------------------
    # Frozen interface (A3/A4 swap internals only).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for each obs in the batch (rollout collection).

        Args:
            obs: ``(B, OBS_DIM)`` float32 observations.
            mask: ``(B, ACTION_DIM)`` bool legal-action mask (``True`` == legal).

        Returns:
            ``(action, logp, value, entropy)`` where ``action`` is ``(B,)`` long,
            and ``logp``, ``value``, ``entropy`` are each ``(B,)`` float.
        """
        dist, value = self._distribution_and_value(obs, mask)
        action = dist.sample()  # type: ignore[no-untyped-call]
        logp = dist.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return action, logp, value, entropy

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score given ``actions`` under the current policy (PPO update).

        Unlike :meth:`act`, this keeps the graph so gradients flow through the
        returned tensors.

        Args:
            obs: ``(B, OBS_DIM)`` float32 observations.
            actions: ``(B,)`` long actions to score.
            mask: ``(B, ACTION_DIM)`` bool legal-action mask (``True`` == legal).

        Returns:
            ``(logp, value, entropy)``, each ``(B,)`` float.
        """
        dist, value = self._distribution_and_value(obs, mask)
        logp = dist.log_prob(actions)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return logp, value, entropy


class DotProductPolicy(nn.Module):
    """Feature-derived dot-product actor-critic over the flat action space (A3).

    Structurally identical *contract* to :class:`HeatPolicy` -- same ``act`` /
    ``evaluate`` signatures, same masked-:class:`~torch.distributions.Categorical`
    output, same ``(B, ACTION_DIM)`` bool mask -- but the policy head is different:
    instead of a free ``Linear(hidden, ACTION_DIM)`` with one untied weight vector
    per flat index, each action is embedded from its **static feature row**
    (:func:`heat.ml.selfplay.action_features.action_feature_table`) by a small
    action MLP, and scored by a scaled dot-product against a state embedding::

        E      = action_mlp(action_features)      # (ACTION_DIM, embed_dim)
        e      = state_proj(trunk(obs))           # (B, embed_dim)
        logits = e @ E.T / sqrt(embed_dim)        # (B, ACTION_DIM)

    Actions therefore share statistical strength through their features (the
    wide-head generalization tax A3 removes), yet the trainer, buffer, and both
    collectors are untouched -- only the internals swap.

    All 516 actions are scored and the illegal set is masked to ``-inf`` before
    the softmax (identical result to ranking only the legal set, and trivially
    cheap at this width); the sparse-gather optimization stays easy to add later.

    Args:
        obs_dim: observation width (defaults to the frozen ``OBS_DIM``).
        action_dim: action-space width (defaults to the frozen ``ACTION_DIM``);
            must match the feature table's row count.
        hidden_sizes: trunk hidden-layer widths (same shape as ``HeatPolicy`` so
            the A3 gate compares heads at equal trunk compute).
        embed_dim: width of the shared state/action embedding space.
        action_mlp_hidden: hidden widths of the per-action feature MLP.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        encoder: str = "flat",
        embed_dim: int = 64,
        action_mlp_hidden: tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.encoder = encoder

        # State trunk is shared with HeatPolicy; A4 swaps only this component.
        self.trunk, prev = _build_trunk(obs_dim, hidden_sizes, encoder)

        # Static per-action feature table as a (non-parameter) buffer: it moves
        # with .to(device) and is saved in checkpoints but never optimized.
        table = torch.from_numpy(action_feature_table())
        if table.shape[0] != action_dim:
            raise ValueError(
                f"action feature table has {table.shape[0]} rows but "
                f"action_dim == {action_dim}"
            )
        self.register_buffer("action_features", table)

        # Action MLP: ACTION_FEAT_DIM -> action_mlp_hidden -> embed_dim (Tanh,
        # matching house style). Recomputed each forward (weights change per
        # update; 516x25 is trivially cheap).
        a_layers: list[nn.Module] = []
        a_prev = ACTION_FEAT_DIM
        for h in action_mlp_hidden:
            a_layers.append(nn.Linear(a_prev, h))
            a_layers.append(nn.Tanh())
            a_prev = h
        a_layers.append(nn.Linear(a_prev, embed_dim))
        self.action_mlp = nn.Sequential(*a_layers)

        # State projection into the shared embedding space, and the value head
        # (reads the trunk latent, exactly as HeatPolicy).
        self.state_proj = nn.Linear(prev, embed_dim)
        self.value_head = nn.Linear(prev, 1)
        self._logit_scale = 1.0 / math.sqrt(embed_dim)

    # ------------------------------------------------------------------
    # Internal: build the masked action distribution + value for a batch.
    # ------------------------------------------------------------------

    def _distribution_and_value(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Return ``(masked Categorical, value (B,))`` -- the dot-product head.

        Scores every action by ``e @ E.T / sqrt(embed_dim)`` then drives illegal
        logits to ``-inf`` before the softmax, so exactly zero mass reaches an
        illegal action (same guarantee, and same ``mask`` semantics, as
        :meth:`HeatPolicy._distribution_and_value`).
        """
        latent = self.trunk(obs)
        state_emb = self.state_proj(latent)  # (B, embed_dim)
        action_emb = self.action_mlp(self.action_features)  # (ACTION_DIM, embed_dim)
        logits = (state_emb @ action_emb.t()) * self._logit_scale  # (B, ACTION_DIM)

        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, neg_inf)
        dist = Categorical(logits=masked_logits)
        value = self.value_head(latent).squeeze(-1)
        return dist, value

    # ------------------------------------------------------------------
    # Frozen interface (byte-for-byte the same contract as HeatPolicy).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for each obs (rollout collection); see
        :meth:`HeatPolicy.act` for the frozen shapes/returns."""
        dist, value = self._distribution_and_value(obs, mask)
        action = dist.sample()  # type: ignore[no-untyped-call]
        logp = dist.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return action, logp, value, entropy

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score given ``actions`` under the current policy (PPO update); see
        :meth:`HeatPolicy.evaluate` for the frozen shapes/returns."""
        dist, value = self._distribution_and_value(obs, mask)
        logp = dist.log_prob(actions)  # type: ignore[no-untyped-call]
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        return logp, value, entropy


#: The policy types the PPO loop accepts. Both classes satisfy the same frozen
#: ``act`` / ``evaluate`` contract (a structural equivalence, not a shared base
#: class), so the trainer/collectors annotate against this union.
PPOPolicy = HeatPolicy | DotProductPolicy


def build_policy(config: A0Config) -> PPOPolicy:
    """Construct the policy selected by ``config.head`` (A3 factory).

    Maps ``config.head`` -> ``"masked"`` :class:`HeatPolicy` (default) /
    ``"dotprod"`` :class:`DotProductPolicy`. Both are built over the frozen
    ``OBS_DIM`` / ``ACTION_DIM`` with the config's trunk widths, so the two
    heads compare at equal trunk compute.

    Raises:
        ValueError: if ``config.head`` is neither ``"masked"`` nor ``"dotprod"``.
    """
    if config.head == "masked":
        return HeatPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=config.hidden_sizes,
            encoder=config.encoder,
        )
    if config.head == "dotprod":
        return DotProductPolicy(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=config.hidden_sizes,
            encoder=config.encoder,
        )
    raise ValueError(
        f"unknown head {config.head!r} (expected 'masked' or 'dotprod')"
    )
