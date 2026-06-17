"""Opponent league + Prioritized Fictitious Self-Play (PFSP) for HEAT (Sprint 6D).

This module replaces the Sprint-5/6C **FIFO snapshot pool** in
:func:`heat.ml.training.train_self_play` with a proper *opponent league* governed
by **prioritized fictitious self-play** (the AlphaStar paradigm, scaled down to a
single-machine ``MaskablePPO`` loop). See
``docs/sprint6d-league-selfplay-design.md``.

Three responsibilities (the design's three FIFO defects, fixed):

1. **Retention** (``League.retain``) -- when the pool exceeds capacity, evict by a
   *keep-strong-and-diverse* value rule rather than by age. The current-best
   anchor is **never** evicted. All tie-breaks use ``snapshot_index`` and any
   randomness uses a seeded RNG, so the retained set is reproducible (test gate).
2. **PFSP sampling** (``League.sample``) -- draw the ``k`` opponent seats by a
   priority weight ``f(learner_win_rate_vs_i)`` instead of a fixed, uniform mix.
   ``"even"``/``"variance"`` (``wr*(1-wr)``, the safe default) focuses on *close*
   matchups; ``"hard"`` (``(1-wr)^p``) focuses on *current losses*. Weights are
   clamped to ``[p_min, p_max]`` and normalized to sum 1.
3. **Per-opponent win-rate bookkeeping** (``League.record_result``) -- track how
   the *current learner* fares against each entry so the priorities are
   data-driven. Unplayed entries use a neutral ``0.5`` prior until they reach
   ``min_games`` so a brand-new snapshot is sampled enough to be estimated.

Each :class:`LeagueEntry` describes one frozen snapshot by **path only** (plus
small metadata), exactly like :class:`heat.ml.training.FrozenSnapshotAgent`'s
path-only spec. The live :class:`League` lives **only in the main training
process**; workers receive picklable paths (the design's process-boundary risk
mitigation) and never the live League.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Win-rate (learner's wins / games) attributed to an entry that has not yet been
#: played enough to estimate -- a neutral prior so new snapshots get sampled.
NEUTRAL_PRIOR = 0.5


@dataclass
class LeagueEntry:
    """One frozen snapshot in the league, described by path + bookkeeping.

    The win-rate the PFSP sampler uses is the **learner's** win-rate against this
    entry (``wins_vs_learner`` is the *entry's* wins, so the learner's win-rate is
    ``1 - wins_vs_learner / games``). Until ``min_games`` games are recorded the
    derived win-rate falls back to :data:`NEUTRAL_PRIOR`.

    Attributes:
        path: Frozen checkpoint base path (the SB3 ``.zip`` minus the suffix);
            :class:`FrozenSnapshotAgent` loads from this. The league's identity
            key.
        snapshot_index: Monotonic creation order -- the deterministic tie-break
            for retention and the diversity axis.
        gate_score: The 6C eval-gate score at creation (a strength proxy, §2.1),
            or ``None`` if unknown.
        games: Games the current learner has played against this entry.
        wins_vs_learner: Games this entry won against the current learner.
        is_anchor: When ``True`` this entry is the current-best snapshot and is
            never evicted by :meth:`League.retain` (the §2.1 best-checkpoint
            anchor). Exactly one entry should be the anchor at a time.
    """

    path: str
    snapshot_index: int
    gate_score: float | None = None
    games: int = 0
    wins_vs_learner: int = 0
    is_anchor: bool = False

    def learner_win_rate(self, *, min_games: int = 1) -> float:
        """Learner's win-rate vs this entry, or the neutral prior if under-played.

        ``min_games`` is the minimum number of recorded games before the observed
        rate is trusted; below it the entry returns :data:`NEUTRAL_PRIOR` so a
        new snapshot is still sampled enough to be estimated.
        """
        if self.games < max(1, min_games):
            return NEUTRAL_PRIOR
        return 1.0 - (self.wins_vs_learner / self.games)


def _pfsp_weight(learner_win_rate: float, *, mode: str, exponent: float) -> float:
    """The unnormalized PFSP priority ``f(wr)`` for one entry (>= 0).

    * ``"even"`` / ``"variance"``: ``wr * (1 - wr)`` -- peaks at ``wr = 0.5``
      (close matchups; the stable default).
    * ``"hard"``: ``(1 - wr) ** exponent`` -- monotonically decreasing in ``wr``
      (focus on current losses; a low-win-rate opponent outweighs a high one).
    """
    wr = min(1.0, max(0.0, learner_win_rate))
    if mode in ("even", "variance"):
        return wr * (1.0 - wr)
    if mode == "hard":
        return (1.0 - wr) ** exponent
    raise ValueError(f"unknown PFSP mode {mode!r} (expected 'even'/'variance'/'hard')")


@dataclass
class League:
    """A persistent, prioritized pool of frozen self-play snapshots (Sprint 6D).

    Lives in the **main** training process only. ``add`` registers a new snapshot;
    ``retain`` enforces the capacity by a keep-strong-and-diverse value rule;
    ``sample`` returns ``k`` opponent **paths** by PFSP priority; ``record_result``
    folds per-game outcomes back into the per-opponent win-rate bookkeeping.

    Attributes:
        capacity: Max number of entries retained (the anchor counts toward it but
            is never evicted). The Sprint-5/6C FIFO cap was 3; 6D defaults larger
            but modest.
        pfsp_mode: ``"even"``/``"variance"`` (default) or ``"hard"`` -- the
            weighting-function family for :meth:`sample`.
        pfsp_exponent: ``p`` in the ``"hard"`` variant ``(1 - wr) ** p``.
        p_min, p_max: Per-entry sample-probability clamps (after normalization the
            relative ordering is preserved but no single opponent dominates / is
            starved). ``p_min`` guards against starving a hard opponent; ``p_max``
            against over-focusing on a single brutal one (the Sprint-5 all-loss
            failure mode).
        min_games: Games before an entry's observed win-rate is trusted over the
            neutral prior.
        strength_weight, diversity_weight: Blend of the retention value function
            (see :meth:`_retention_value`).
    """

    capacity: int = 8
    pfsp_mode: str = "even"
    pfsp_exponent: float = 2.0
    p_min: float = 0.0
    p_max: float = 1.0
    min_games: int = 1
    strength_weight: float = 1.0
    diversity_weight: float = 1.0
    entries: list[LeagueEntry] = field(default_factory=list)

    # -- membership ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def _by_path(self, path: str) -> LeagueEntry | None:
        for e in self.entries:
            if e.path == path:
                return e
        return None

    def add(self, entry: LeagueEntry) -> None:
        """Register ``entry`` (replacing any existing entry with the same path).

        Re-adding a path updates its metadata in place but **preserves** the
        accumulated win-rate bookkeeping (a re-saved snapshot at the same path is
        the same opponent). After adding, the caller should call :meth:`retain`
        to enforce capacity.
        """
        existing = self._by_path(entry.path)
        if existing is not None:
            existing.snapshot_index = entry.snapshot_index
            existing.gate_score = entry.gate_score
            existing.is_anchor = entry.is_anchor
            return
        self.entries.append(entry)

    def set_anchor(self, path: str) -> None:
        """Mark ``path`` as the sole current-best anchor (never evicted).

        Clears the anchor flag on all other entries. A no-op if ``path`` is not in
        the league.
        """
        target = self._by_path(path)
        if target is None:
            return
        for e in self.entries:
            e.is_anchor = e is target

    # -- retention ----------------------------------------------------------

    def _retention_value(self, entry: LeagueEntry) -> float:
        """Keep-strong-and-diverse value for ``entry`` (higher = more keepable).

        Blends two terms (the design's §"League membership & retention"):

        * **strength** -- the entry's ``gate_score`` if known (a higher 6C gate
          score means a stronger past self worth keeping), else a neutral 0.5.
        * **diversity** -- a recency term ``snapshot_index`` (normalized): newer
          snapshots are slightly favored so the pool tracks the evolving learner
          rather than collapsing onto one stale band, but never enough to make
          retention pure-FIFO (strength dominates when the weights are equal and
          gate scores differ). The blend is the tuning knob the smoke gate
          validates.

        Deterministic: depends only on stored fields, no RNG.
        """
        strength = entry.gate_score if entry.gate_score is not None else 0.5

        n = len(self.entries)
        if n <= 1:
            diversity = 0.0
        else:
            indices = [e.snapshot_index for e in self.entries]
            lo, hi = min(indices), max(indices)
            span = hi - lo
            diversity = 0.0 if span == 0 else (entry.snapshot_index - lo) / span

        return self.strength_weight * strength + self.diversity_weight * diversity

    def retain(self) -> None:
        """Evict lowest-value entries until ``len(self) <= capacity``.

        Keep-strong-and-diverse, **not** FIFO: entries are ranked by
        :meth:`_retention_value` (desc), with ``snapshot_index`` (asc) as the
        deterministic tie-break. The anchor (``is_anchor``) is always retained
        regardless of value. Reproducible: no randomness, total ordering.
        """
        if len(self.entries) <= self.capacity:
            return

        anchors = [e for e in self.entries if e.is_anchor]
        candidates = [e for e in self.entries if not e.is_anchor]

        # Highest value first; ties broken by smallest snapshot_index (stable,
        # deterministic). We *keep* the top of this order.
        candidates.sort(key=lambda e: (-self._retention_value(e), e.snapshot_index))

        keep_n = max(0, self.capacity - len(anchors))
        kept = anchors + candidates[:keep_n]

        # Preserve a stable storage order (by snapshot_index) for reproducibility.
        kept.sort(key=lambda e: e.snapshot_index)
        self.entries = kept

    # -- PFSP sampling ------------------------------------------------------

    def weights(self) -> dict[str, float]:
        """Return the normalized PFSP sample weights ``{path -> w}`` (sum to 1).

        Each entry's unnormalized priority is ``f(learner_win_rate)`` per
        :func:`_pfsp_weight`; the vector is normalized, **clamped** to
        ``[p_min, p_max]``, then re-normalized so it still sums to 1 while
        honoring the clamps as closely as a valid distribution allows. With an
        empty league this returns ``{}``.

        Determinism: pure function of the stored win-rates + config (no RNG).
        """
        if not self.entries:
            return {}

        raw = np.array(
            [
                _pfsp_weight(
                    e.learner_win_rate(min_games=self.min_games),
                    mode=self.pfsp_mode,
                    exponent=self.pfsp_exponent,
                )
                for e in self.entries
            ],
            dtype=np.float64,
        )

        total = raw.sum()
        if total <= 0.0:
            # All-zero priorities (e.g. every entry at wr=0 or wr=1 under "even"):
            # fall back to uniform so sampling is still well-defined.
            probs = np.full(len(self.entries), 1.0 / len(self.entries))
        else:
            probs = raw / total

        probs = self._project_to_clamped_simplex(probs)
        return {e.path: float(p) for e, p in zip(self.entries, probs)}

    def _project_to_clamped_simplex(self, probs: np.ndarray) -> np.ndarray:
        """Renormalize ``probs`` to sum 1 while honoring ``[p_min, p_max]``.

        A single clamp-then-renormalize pass does NOT guarantee the bounds hold
        after the final division. This iterates clamp-then-redistribute (a
        water-filling / iterative-projection scheme): entries pinned at a bound
        are frozen and the residual mass is renormalized across the free entries
        until every entry is within ``[p_min, p_max]`` (or no free entries
        remain). Deterministic and order-preserving among free entries.

        If the bounds are infeasible for ``n`` entries (e.g. ``n * p_max < 1`` or
        ``n * p_min > 1``) the result is the closest valid clamp -- still a
        well-defined, summed-to-1-where-possible vector for sampling.
        """
        n = len(probs)
        p_min = max(0.0, self.p_min)
        p_max = min(1.0, self.p_max) if self.p_max > 0 else 1.0
        # Guard against an infeasible/degenerate band -> fall back to uniform-clip.
        if p_min > p_max or n == 0:
            return np.clip(probs, 0.0, 1.0)

        out = probs.astype(np.float64).copy()
        free = np.ones(n, dtype=bool)
        for _ in range(n + 1):
            fixed_sum = out[~free].sum()
            free_target = 1.0 - fixed_sum
            free_idx = np.where(free)[0]
            if free_idx.size == 0:
                break
            cur = out[free_idx].sum()
            if cur > 0:
                out[free_idx] = out[free_idx] / cur * max(0.0, free_target)
            else:
                out[free_idx] = free_target / free_idx.size

            lo = out[free_idx] < p_min - 1e-12
            hi = out[free_idx] > p_max + 1e-12
            if not lo.any() and not hi.any():
                break
            out[free_idx[hi]] = p_max
            out[free_idx[lo]] = p_min
            free[free_idx[hi]] = False
            free[free_idx[lo]] = False

        return out

    def sample(self, k: int, rng: np.random.Generator) -> list[str]:
        """Return ``k`` opponent **paths** drawn by PFSP priority.

        Sampling is *with replacement* by the normalized :meth:`weights` (so a
        seat may repeat a strong/close opponent), using the supplied seeded
        ``rng`` for reproducibility (test gate). Returns ``[]`` for an empty
        league or ``k <= 0``.
        """
        if k <= 0 or not self.entries:
            return []
        w = self.weights()
        paths = list(w.keys())
        probs = np.array([w[p] for p in paths], dtype=np.float64)
        probs = probs / probs.sum()  # guard against float drift
        idx = rng.choice(len(paths), size=k, replace=True, p=probs)
        return [paths[i] for i in idx]

    # -- win-rate bookkeeping ----------------------------------------------

    def record_result(self, path: str, learner_won: bool) -> None:
        """Record one game's outcome of the current learner vs entry ``path``.

        Updates ``games`` and (when the entry won) ``wins_vs_learner``, which
        drives the entry's :meth:`LeagueEntry.learner_win_rate`. A no-op if
        ``path`` is not in the league (a stale path whose entry was already
        evicted -- the result is simply dropped).
        """
        entry = self._by_path(path)
        if entry is None:
            return
        entry.games += 1
        if not learner_won:
            entry.wins_vs_learner += 1

    def record_win_rate(self, path: str, learner_win_rate: float, games: int) -> None:
        """Fold an aggregate learner-win-rate estimate over ``games`` into ``path``.

        Convenience for the training loop's *evaluation-based* attribution
        (§"Win-rate bookkeeping"): rather than recording each rollout game
        individually, a small deterministic eval estimates the learner's win-rate
        vs this entry; this method converts that into the equivalent
        ``games``/``wins_vs_learner`` increment. A no-op for unknown paths.
        """
        entry = self._by_path(path)
        if entry is None or games <= 0:
            return
        wr = min(1.0, max(0.0, learner_win_rate))
        learner_wins = int(round(wr * games))
        entry.games += games
        entry.wins_vs_learner += games - learner_wins
