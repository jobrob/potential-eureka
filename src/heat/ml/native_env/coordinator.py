"""CPU policy batching and identity-keyed sampling for Direction D3 A7."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import torch

from heat.ml.spaces import ACTION_DIM, OBS_DIM


_U64_MASK = (1 << 64) - 1


class _Distribution(Protocol):
    """Part of ``torch.distributions.Categorical`` used by A7."""

    @property
    def probs(self) -> torch.Tensor: ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...


class BatchedPolicy(Protocol):
    """Frozen actor-critic boundary shared by current and snapshot policies."""

    def _distribution_and_value(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[_Distribution, torch.Tensor]: ...


@dataclass(frozen=True)
class RegisteredPolicy:
    """One immutable policy identity/version entry for an admitted iteration."""

    policy_id: int
    version: int
    policy: BatchedPolicy


@dataclass(frozen=True)
class ReadyBatch:
    """One leased native-ready buffer with rows aligned across every field."""

    observations: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]
    group_tokens: NDArray[np.uint64]
    group_sizes: NDArray[np.uint16]
    game_ids: NDArray[np.uint64]
    seat_ids: NDArray[np.int16]
    decision_sequences: NDArray[np.uint32]
    policy_ids: NDArray[np.uint16]
    policy_versions: NDArray[np.uint32]
    value_only: NDArray[np.bool_]
    lease_token: int

    def __post_init__(self) -> None:
        """Reject implicit copies, invalid rows, and split/non-contiguous groups."""
        rows = self.observations.shape[0]
        expected = (
            (self.observations, np.float32, (rows, OBS_DIM), "observations"),
            (self.legal_masks, np.bool_, (rows, ACTION_DIM), "legal_masks"),
            (self.group_tokens, np.uint64, (rows,), "group_tokens"),
            (self.group_sizes, np.uint16, (rows,), "group_sizes"),
            (self.game_ids, np.uint64, (rows,), "game_ids"),
            (self.seat_ids, np.int16, (rows,), "seat_ids"),
            (self.decision_sequences, np.uint32, (rows,), "decision_sequences"),
            (self.policy_ids, np.uint16, (rows,), "policy_ids"),
            (self.policy_versions, np.uint32, (rows,), "policy_versions"),
            (self.value_only, np.bool_, (rows,), "value_only"),
        )
        for array, dtype, shape, name in expected:
            if array.dtype != dtype:
                raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
        if rows and np.any(~self.value_only & ~self.legal_masks.any(axis=1)):
            raise ValueError("every action row must contain a legal action")
        if self.lease_token < 1:
            raise ValueError("lease_token must be positive")
        closed_tokens: set[int] = set()
        start = 0
        while start < rows:
            token = int(self.group_tokens[start])
            if token in closed_tokens:
                raise ValueError("ready groups must be contiguous and unsplit")
            end = start + 1
            while end < rows and int(self.group_tokens[end]) == token:
                end += 1
            if np.any(self.group_sizes[start:end] != end - start):
                raise ValueError("ready group row count is incomplete")
            closed_tokens.add(token)
            start = end

    @property
    def row_count(self) -> int:
        """Return the number of real and value-only rows in this lease."""
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class PolicySubmission:
    """Actions and actor-critic values aligned to the input ready rows."""

    actions: NDArray[np.int64]
    logps: NDArray[np.float32]
    values: NDArray[np.float32]
    sampling_words: NDArray[np.uint32]
    lease_token: int


def splitmix64(value: int) -> int:
    """Return the platform-independent SplitMix64 permutation of one word."""
    value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (value ^ (value >> 31)) & _U64_MASK


def keyed_sampling_word(
    sampling_seed: int,
    policy_id: int,
    policy_version: int,
    game_id: int,
    seat_id: int,
    decision_sequence: int,
) -> int:
    """Hash immutable decision identity into the keyed-categorical-v1 word."""
    value = sampling_seed & _U64_MASK
    for field in (policy_id, policy_version, game_id, seat_id, decision_sequence):
        value = splitmix64(value ^ (field & _U64_MASK))
    return value & 0xFFFFFFFF


class CpuPolicyCoordinator:
    """Partition ready rows by policy and make one CPU inference call per group."""

    def __init__(
        self,
        policies: tuple[RegisteredPolicy, ...],
        *,
        sampling_seed: int,
    ) -> None:
        if not policies:
            raise ValueError("at least one policy must be registered")
        self._policies: dict[int, RegisteredPolicy] = {}
        for entry in policies:
            if entry.policy_id in self._policies:
                raise ValueError(f"duplicate policy_id {entry.policy_id}")
            self._policies[entry.policy_id] = entry
        self._sampling_seed = sampling_seed
        self._dispatch_calls = 0
        self._inference_calls = 0
        self._rows = 0
        self._batch_sizes: list[int] = []
        self._rows_by_policy: dict[int, int] = {}
        self._inference_ns = 0

    @torch.no_grad()
    def dispatch(self, ready: ReadyBatch) -> PolicySubmission:
        """Evaluate registered CPU policies and sample independently of row order."""
        rows = ready.row_count
        actions = np.full(rows, -1, dtype=np.int64)
        logps = np.zeros(rows, dtype=np.float32)
        values = np.zeros(rows, dtype=np.float32)
        words = np.zeros(rows, dtype=np.uint32)
        started = time.perf_counter_ns()
        for raw_policy_id in np.unique(ready.policy_ids):
            policy_id = int(raw_policy_id)
            entry = self._policies.get(policy_id)
            if entry is None:
                raise KeyError(f"ready row refers to unknown policy_id {policy_id}")
            indices = np.flatnonzero(ready.policy_ids == raw_policy_id)
            if np.any(ready.policy_versions[indices] != entry.version):
                raise ValueError(f"policy version mismatch for policy_id {policy_id}")

            obs = torch.from_numpy(ready.observations[indices])
            masks_np = ready.legal_masks[indices].copy()
            value_only = ready.value_only[indices]
            masks_np[value_only] = True
            distribution, batch_values = entry.policy._distribution_and_value(
                obs, torch.from_numpy(masks_np)
            )
            if distribution.probs.device.type != "cpu":
                raise ValueError("A7 CpuPolicyCoordinator requires CPU policies")
            batch_actions = torch.full((len(indices),), -1, dtype=torch.int64)
            action_positions = np.flatnonzero(~value_only)
            if len(action_positions):
                action_words = np.array(
                    [
                        keyed_sampling_word(
                            self._sampling_seed,
                            policy_id,
                            entry.version,
                            int(ready.game_ids[index]),
                            int(ready.seat_ids[index]),
                            int(ready.decision_sequences[index]),
                        )
                        for index in indices[action_positions]
                    ],
                    dtype=np.uint32,
                )
                uniforms = torch.from_numpy(
                    (action_words.astype(np.float64) + 0.5) / 2**32
                )
                probabilities = distribution.probs[action_positions]
                sampled = torch.sum(
                    torch.cumsum(probabilities, dim=1).to(torch.float64)
                    < uniforms[:, None],
                    dim=1,
                ).clamp_max(ACTION_DIM - 1)
                batch_actions[action_positions] = sampled
                batch_logps = distribution.log_prob(batch_actions.clamp_min(0))
                logps[indices[action_positions]] = (
                    batch_logps[action_positions].cpu().numpy().astype(np.float32)
                )
                words[indices[action_positions]] = action_words

            actions[indices] = batch_actions.cpu().numpy()
            values[indices] = batch_values.cpu().numpy().astype(np.float32)
            self._inference_calls += 1
            self._batch_sizes.append(len(indices))
            self._rows_by_policy[policy_id] = (
                self._rows_by_policy.get(policy_id, 0) + len(indices)
            )
        self._inference_ns += time.perf_counter_ns() - started
        self._dispatch_calls += 1
        self._rows += rows
        return PolicySubmission(actions, logps, values, words, ready.lease_token)

    def stats(self) -> dict[str, object]:
        """Return dispatch counts and a compact observed batch histogram."""
        histogram: dict[int, int] = {}
        for size in self._batch_sizes:
            histogram[size] = histogram.get(size, 0) + 1
        return {
            "dispatch_calls": self._dispatch_calls,
            "inference_calls": self._inference_calls,
            "rows": self._rows,
            "rows_by_policy": dict(sorted(self._rows_by_policy.items())),
            "batch_size_histogram": dict(sorted(histogram.items())),
            "inference_seconds": self._inference_ns / 1_000_000_000,
        }
