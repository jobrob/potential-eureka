"""Sprint 8D: mass-training sweep + large-league analysis orchestration.

This module is the **thin, pure, unit-tested** half of Sprint 8D (§3): the
declarative factor sweep, the run manifest, the balanced league field sampler,
and the attribution math. The heavy compute (actual ``train_self_play`` runs,
actual league games) lives in ``experiments/run_campaign.py`` /
``experiments/run_league.py``, which call into here.

Everything in this module is deterministic given its inputs and free of SB3 /
env construction, so the test suite (``tests/test_sweep.py``) exercises it with
cheap synthetic data and never trains a model.

Layout
------
* :class:`FactorAxis` / :class:`SweepSpec` / :class:`RunConfig` -- the declarative
  factor grid + its expansion to concrete training configs (§4).
* :func:`_apply_factor` -- the single dotted-path applier that reaches into the
  8C config shape (``ppo.*`` / ``curriculum.*`` / ``phases.i.*`` /
  ``schedule.*`` / ``seed``), via :func:`dataclasses.replace` + list edits (§4.3).
* :class:`ManifestWriter` / :class:`ManifestReader` + :func:`run_campaign` --
  the append-only ``jsonl`` run index + the sequential, failure-isolated,
  resumable run loop (§5).
* :func:`balanced_fields` / :func:`seat_rotations` + the ``GameOutcome`` jsonl
  writer/reader -- the league matchup sampler + per-game persistence (§7).
* :func:`build_attribution` + the ``attribution.md`` / ``ladder.csv`` /
  ``effects.json`` emitters -- the observational factor-effect analysis (§9).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from heat.ml.model import PPOConfig, net_profile_config
from heat.ml.training import (
    CurriculumConfig,
    OpponentSchedule,
    OpponentStage,
    TrainingPhase,
    default_8c_phases,
    sprint_8c_curriculum,
)
from heat.simulation.runner import GameOutcome, PlayerOutcome

# ---------------------------------------------------------------------------
# Declarative sweep spec (§4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorAxis:
    """One swept factor: a name + the discrete levels to try (§4.2).

    ``name`` is a dotted path the expander knows how to apply
    (e.g. ``"phases.0.steps"``, ``"ppo.shaping_weight"``,
    ``"schedule.pool_kinds"``, ``"seed"``, ``"ppo.net_profile"``). Continuous
    factors are pre-discretized to levels so the grid is finite and the
    attribution (§9) has clean groups.
    """

    name: str
    levels: tuple


@dataclass(frozen=True)
class RunConfig:
    """One fully-resolved run: the factor vector + the concrete training configs.

    ``factors`` is the ``{axis_name: level}`` dict (the attribution design-matrix
    row). ``ppo`` / ``curriculum`` / ``phases`` / ``schedule`` are the resolved
    objects passed straight to :func:`heat.ml.training.train_self_play`
    (``schedule`` is informational -- ``train_self_play`` derives its own from
    ``phases`` -- but carried so the sweep can report the pool ramp). ``run_id``
    is a deterministic hash of ``factors`` (stable across campaign restarts ->
    resumability).
    """

    run_id: str
    factors: dict
    ppo: PPOConfig
    curriculum: CurriculumConfig
    phases: list[TrainingPhase]
    schedule: OpponentSchedule


def _stable_run_id(factors: Mapping[str, Any]) -> str:
    """Short deterministic hash of a factor vector (resumability key, §5.1)."""
    payload = json.dumps(factors, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _schedule_from_phases(phases: Sequence[TrainingPhase]) -> OpponentSchedule:
    """Mirror :func:`heat.ml.training._train_phases`'s schedule derivation.

    The opponent schedule is exactly the non-solo phases (those with a
    ``pool_kind``), in order. Kept here so a :class:`RunConfig` can carry the
    schedule it will effectively train under for reporting / sidecar metadata.
    """
    return OpponentSchedule(
        stages=tuple(
            OpponentStage(p.pool_kind, p.steps)
            for p in phases
            if p.pool_kind is not None
        )
    )


# ---------------------------------------------------------------------------
# The dotted-path config applier (§4.3) -- the ONLY place 8D reaches into the
# 8C config shape. If 8C renames a field, exactly this function changes.
# ---------------------------------------------------------------------------


def _apply_factor(
    name: str,
    level: Any,
    *,
    ppo: PPOConfig,
    curriculum: CurriculumConfig,
    phases: list[TrainingPhase],
) -> tuple[PPOConfig, CurriculumConfig, list[TrainingPhase]]:
    """Overlay one ``name=level`` factor onto the resolved configs (§4.3).

    Returns the (possibly new) ``(ppo, curriculum, phases)`` triple. Uses
    :func:`dataclasses.replace` for the frozen/plain dataclasses (so every
    untouched field is preserved exactly) and a copy-then-edit for the phase
    list. Supported dotted paths:

    * ``"seed"`` -> ``ppo.seed``.
    * ``"ppo.<field>"`` -> a field on :class:`PPOConfig`. Special case
      ``"ppo.net_profile"`` applies a named :data:`heat.ml.model.NET_PROFILES`
      profile (net_arch + features sizes together).
    * ``"curriculum.<field>"`` -> a field on :class:`CurriculumConfig`.
    * ``"phases.<i>.<field>"`` -> a field on phase ``i`` of the phase list.
    * ``"phases.steps_ratio"`` -> scale every opponent phase's ``steps`` by a
      per-phase ratio tuple (a step-ratio *profile*, §4.1).
    * ``"schedule.pool_kinds"`` -> redefine the opponent ramp: a comma-separated
      string (e.g. ``"weak,mixed,strong"`` or ``"weak,strong"``) rebuilds the
      opponent phases to that kind sequence, preserving the solo phase and reusing
      the existing opponent phases' budgets/gamma/shaping in order.

    Raises ``ValueError`` / ``AttributeError`` for an unknown path or field, so a
    typo'd axis fails loudly at expand time rather than silently training the base.
    """
    if name == "seed":
        return dataclasses.replace(ppo, seed=int(level)), curriculum, phases

    if name.startswith("ppo."):
        field_name = name[len("ppo.") :]
        if field_name == "net_profile":
            return net_profile_config(str(level), ppo), curriculum, phases
        if not hasattr(ppo, field_name):
            raise AttributeError(f"PPOConfig has no field {field_name!r}")
        return dataclasses.replace(ppo, **{field_name: level}), curriculum, phases

    if name.startswith("curriculum."):
        field_name = name[len("curriculum.") :]
        if not hasattr(curriculum, field_name):
            raise AttributeError(f"CurriculumConfig has no field {field_name!r}")
        return ppo, dataclasses.replace(curriculum, **{field_name: level}), phases

    if name == "phases.steps_ratio":
        # ``level`` is a per-opponent-phase multiplier tuple applied to each
        # opponent phase's steps in order (the solo phase is left untouched). A
        # scalar applies uniformly. Records a clean "step-ratio profile" factor.
        new_phases = list(phases)
        ratios = level if isinstance(level, (tuple, list)) else None
        opp_i = 0
        for i, p in enumerate(new_phases):
            if p.pool_kind is None:
                continue
            mult = ratios[opp_i % len(ratios)] if ratios else float(level)
            new_phases[i] = dataclasses.replace(
                p, steps=max(1, int(round(p.steps * float(mult))))
            )
            opp_i += 1
        return ppo, curriculum, new_phases

    if name == "schedule.pool_kinds":
        kinds = [
            k.strip() for k in str(level).split(",") if k.strip()
        ]
        if not kinds:
            raise ValueError(f"schedule.pool_kinds is empty: {level!r}")
        solo = [p for p in phases if p.pool_kind is None]
        opp = [p for p in phases if p.pool_kind is not None]
        if not opp:
            raise ValueError("cannot set schedule.pool_kinds with no opponent phases")
        # Reuse the existing opponent phases' (steps, gamma, shaping, players) in
        # order; if the new ramp is longer, cycle the last opponent phase as the
        # template so budgets stay sensible.
        new_opp: list[TrainingPhase] = []
        for j, kind in enumerate(kinds):
            template = opp[j] if j < len(opp) else opp[-1]
            new_opp.append(
                dataclasses.replace(template, name=kind, pool_kind=kind)
            )
        return ppo, curriculum, solo + new_opp

    if name.startswith("phases."):
        parts = name.split(".")
        if len(parts) != 3:
            raise ValueError(f"bad phase factor path {name!r} (want phases.<i>.<field>)")
        idx = int(parts[1])
        field_name = parts[2]
        new_phases = list(phases)
        if not (0 <= idx < len(new_phases)):
            raise ValueError(f"phase index {idx} out of range (have {len(new_phases)})")
        if not hasattr(new_phases[idx], field_name):
            raise AttributeError(f"TrainingPhase has no field {field_name!r}")
        new_phases[idx] = dataclasses.replace(
            new_phases[idx], **{field_name: level}
        )
        return ppo, curriculum, new_phases

    raise ValueError(f"unknown sweep factor path {name!r}")


# ---------------------------------------------------------------------------
# Base presets (§4.3) -- where expand() branches from.
# ---------------------------------------------------------------------------

_BASE_PRESETS: dict[str, Callable[[], tuple[PPOConfig, CurriculumConfig, list]]] = {}


def _sprint_8c_base() -> tuple[PPOConfig, CurriculumConfig, list[TrainingPhase]]:
    """The default base: 8C's PPO defaults + curriculum + phase list."""
    return PPOConfig(), sprint_8c_curriculum(), default_8c_phases()


_BASE_PRESETS["sprint_8c"] = _sprint_8c_base


def _base_for(preset: str) -> tuple[PPOConfig, CurriculumConfig, list[TrainingPhase]]:
    if preset not in _BASE_PRESETS:
        raise ValueError(
            f"unknown base_preset {preset!r}; choose from {sorted(_BASE_PRESETS)}"
        )
    return _BASE_PRESETS[preset]()


@dataclass(frozen=True)
class SweepSpec:
    """Declarative factor grid + sampling policy for a campaign (§4.2).

    Data only -- no model, no env -- so it is trivially serializable, diffable,
    and unit-testable.

    Attributes:
        axes: the swept factors. ``seed`` is supplied separately via ``seeds``
            and must NOT also appear as an axis.
        seeds: replication seeds applied within every cell (so within-cell
            variance is estimable for attribution).
        method: ``"grid"`` (full Cartesian product x seeds), ``"random"`` (draw
            ``max_configs`` random cells), or ``"lhs"`` (Latin-hypercube: spread
            ``max_configs`` draws across each axis's marginal). Random/LHS are
            seeded by a stable spec hash -> deterministic.
        max_configs: cap on the number of distinct factor cells (before seed
            replication) for ``"random"``/``"lhs"``. Ignored for ``"grid"``.
        base_preset: which preset :func:`expand` branches from (default
            ``"sprint_8c"``).
    """

    axes: tuple[FactorAxis, ...]
    seeds: tuple[int, ...] = (0,)
    method: str = "grid"
    max_configs: int | None = None
    base_preset: str = "sprint_8c"

    def _spec_hash(self) -> int:
        """Deterministic integer seed for the random/LHS samplers."""
        payload = json.dumps(
            {
                "axes": [(a.name, list(a.levels)) for a in self.axes],
                "seeds": list(self.seeds),
                "method": self.method,
                "max_configs": self.max_configs,
                "base_preset": self.base_preset,
            },
            sort_keys=True,
            default=str,
        )
        return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)

    def _cells(self) -> list[dict]:
        """The list of factor cells (``{axis_name: level}``), pre-replication."""
        if self.method not in ("grid", "random", "lhs"):
            raise ValueError(
                f"method must be 'grid'/'random'/'lhs', got {self.method!r}"
            )
        if any(a.name == "seed" for a in self.axes):
            raise ValueError(
                "'seed' is a replication axis (use SweepSpec.seeds), not a "
                "FactorAxis"
            )
        if not self.axes:
            return [{}]

        if self.method == "grid":
            cells: list[dict] = [{}]
            for axis in self.axes:
                cells = [
                    {**c, axis.name: lvl} for c in cells for lvl in axis.levels
                ]
            return cells

        # random / lhs both honor max_configs deterministically (seeded by the
        # spec hash). Without a cap, fall back to the full grid.
        if self.max_configs is None:
            return self._cells_grid_fallback()
        rng = random.Random(self._spec_hash())
        if self.method == "random":
            return self._cells_random(rng)
        return self._cells_lhs(rng)

    def _cells_grid_fallback(self) -> list[dict]:
        cells: list[dict] = [{}]
        for axis in self.axes:
            cells = [
                {**c, axis.name: lvl} for c in cells for lvl in axis.levels
            ]
        return cells

    def _cells_random(self, rng: random.Random) -> list[dict]:
        """Draw ``max_configs`` distinct random cells (independent per axis)."""
        n = int(self.max_configs)  # type: ignore[arg-type]
        seen: set[str] = set()
        cells: list[dict] = []
        # Bound attempts so a small factor space (fewer combos than max_configs)
        # terminates rather than spinning forever.
        max_attempts = max(1000, n * 50)
        attempts = 0
        while len(cells) < n and attempts < max_attempts:
            attempts += 1
            cell = {a.name: rng.choice(list(a.levels)) for a in self.axes}
            key = json.dumps(cell, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            cells.append(cell)
        return cells

    def _cells_lhs(self, rng: random.Random) -> list[dict]:
        """Latin-hypercube draw: each axis's levels spread evenly across N draws.

        For each axis, build a list of ``max_configs`` level picks where the
        axis's levels appear as equally as possible, then shuffle independently
        per axis and zip into cells. This spreads each axis's marginal (no level
        starved) while decorrelating axes -- the textbook LHS property on a
        discretized grid. Deterministic given the spec hash.
        """
        n = int(self.max_configs)  # type: ignore[arg-type]
        columns: list[list[Any]] = []
        for axis in self.axes:
            levels = list(axis.levels)
            col: list[Any] = []
            i = 0
            while len(col) < n:
                col.append(levels[i % len(levels)])
                i += 1
            rng.shuffle(col)
            columns.append(col)
        cells: list[dict] = []
        for row in range(n):
            cells.append(
                {axis.name: columns[c][row] for c, axis in enumerate(self.axes)}
            )
        return cells

    def expand(self) -> list[RunConfig]:
        """Materialize the (possibly sampled) list of :class:`RunConfig`\\ s.

        grid: Cartesian product of all axis levels x seeds. random/lhs: draw
        ``max_configs`` cells (seeded) then replicate across ``seeds``.
        Deterministic given the spec.
        """
        seeds = self.seeds if self.seeds else (0,)
        run_configs: list[RunConfig] = []
        for cell in self._cells():
            for seed in seeds:
                ppo, curriculum, phases = _base_for(self.base_preset)
                # Apply the cell's factors in a stable (sorted) order so the
                # resolved configs are deterministic regardless of dict order.
                for name in sorted(cell):
                    ppo, curriculum, phases = _apply_factor(
                        name, cell[name],
                        ppo=ppo, curriculum=curriculum, phases=phases,
                    )
                # Seed is a replication factor, applied last.
                ppo, curriculum, phases = _apply_factor(
                    "seed", seed, ppo=ppo, curriculum=curriculum, phases=phases
                )
                factors = {**{k: cell[k] for k in sorted(cell)}, "seed": seed}
                run_id = _stable_run_id(factors)
                run_configs.append(
                    RunConfig(
                        run_id=run_id,
                        factors=factors,
                        ppo=ppo,
                        curriculum=dataclasses.replace(
                            curriculum, run_name=f"campaign_{run_id}"
                        ),
                        phases=phases,
                        schedule=_schedule_from_phases(phases),
                    )
                )
        return run_configs


# ---------------------------------------------------------------------------
# Run manifest (§5.2) -- append-only jsonl index over the checkpoint sidecars.
# ---------------------------------------------------------------------------


class ManifestWriter:
    """Append-only ``jsonl`` writer for the campaign run index (§5.2).

    Each call appends one self-contained row (``started`` / ``done`` /
    ``failed``) and flushes, so a mid-campaign kill loses at most the in-flight
    run. The run loop reads ``completed_run_ids`` (via :class:`ManifestReader`)
    on restart to skip finished runs.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _append(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def mark_started(self, rc: RunConfig) -> None:
        self._append(
            {
                "run_id": rc.run_id,
                "status": "started",
                "factors": rc.factors,
                "ts": time.time(),
            }
        )

    def mark_done(
        self,
        rc: RunConfig,
        *,
        checkpoint: str,
        wall_clock_s: float,
        gate_score: float | None = None,
        git_sha: str | None = None,
    ) -> None:
        self._append(
            {
                "run_id": rc.run_id,
                "status": "done",
                "factors": rc.factors,
                "checkpoint": checkpoint,
                "meta": _meta_sidecar_path(checkpoint),
                "git_sha": git_sha,
                "gate_score": gate_score,
                "wall_clock_s": wall_clock_s,
                "error": None,
                "ts": time.time(),
            }
        )

    def mark_failed(self, rc: RunConfig, *, error: str) -> None:
        self._append(
            {
                "run_id": rc.run_id,
                "status": "failed",
                "factors": rc.factors,
                "checkpoint": None,
                "error": error,
                "ts": time.time(),
            }
        )


def _meta_sidecar_path(checkpoint: str) -> str:
    """Best-effort sidecar path for a checkpoint (mirrors training.meta_path_for).

    Imported lazily so the pure manifest layer has no hard training dependency
    for the common case; falls back to a simple suffix swap.
    """
    base = checkpoint[:-4] if checkpoint.endswith(".zip") else checkpoint
    return base + ".meta.json"


@dataclass
class ManifestRow:
    """A parsed manifest row (the latest status wins per ``run_id``)."""

    run_id: str
    status: str
    factors: dict
    checkpoint: str | None = None
    meta: str | None = None
    git_sha: str | None = None
    gate_score: float | None = None
    wall_clock_s: float | None = None
    error: str | None = None


class ManifestReader:
    """Reader over an append-only manifest ``jsonl`` (§5.2).

    Collapses the append log to one :class:`ManifestRow` per ``run_id`` (the
    last-written status wins, so a ``started`` then ``done`` reads as ``done``).
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _rows(self) -> dict[str, ManifestRow]:
        latest: dict[str, ManifestRow] = {}
        if not os.path.exists(self.path):
            return latest
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                latest[d["run_id"]] = ManifestRow(
                    run_id=d["run_id"],
                    status=d.get("status", ""),
                    factors=d.get("factors", {}),
                    checkpoint=d.get("checkpoint"),
                    meta=d.get("meta"),
                    git_sha=d.get("git_sha"),
                    gate_score=d.get("gate_score"),
                    wall_clock_s=d.get("wall_clock_s"),
                    error=d.get("error"),
                )
        return latest

    def rows(self) -> list[ManifestRow]:
        """All rows (one per run_id, latest status)."""
        return list(self._rows().values())

    def completed_run_ids(self) -> set[str]:
        """Run ids whose latest status is ``done`` (resumability skip set)."""
        return {r.run_id for r in self._rows().values() if r.status == "done"}

    def done_rows(self) -> list[ManifestRow]:
        """Rows whose latest status is ``done`` (the league contender source)."""
        return [r for r in self._rows().values() if r.status == "done"]

    def failed_rows(self) -> list[ManifestRow]:
        return [r for r in self._rows().values() if r.status == "failed"]


def run_campaign(
    spec: SweepSpec,
    *,
    out_dir: str,
    train_fn: Callable[..., tuple[Any, str]] | None = None,
    gate_reader: Callable[[str], float | None] | None = None,
    resume: bool = True,
) -> str:
    """Run a campaign sequentially, isolated per-run, resumable (§5.1).

    Drives one ``train_fn`` (default :func:`heat.ml.training.train_self_play`)
    per :class:`RunConfig` in ``spec.expand()``, writing a ``manifest.jsonl`` row
    for each. **Sequential** (one train in flight), **isolated** (a run's
    exception marks that row ``failed`` and the campaign continues), and
    **resumable** (on restart, ``done`` run_ids are skipped).

    ``train_fn(config, curriculum, *, num_players, phases)`` must return
    ``(model, best_checkpoint_path)`` like ``train_self_play``. ``gate_reader``
    maps a checkpoint path to its final gate score for the manifest row (default:
    read ``promote_score``-equivalent from the sidecar's metadata when present).

    Returns the manifest path. The heavy ``experiments/run_campaign.py`` shell
    just builds the default :class:`SweepSpec` and calls this.
    """
    if train_fn is None:  # pragma: no cover - exercised only in the heavy shell
        from heat.ml.training import train_self_play as train_fn  # type: ignore

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    writer = ManifestWriter(manifest_path)
    done = ManifestReader(manifest_path).completed_run_ids() if resume else set()

    for rc in spec.expand():
        if rc.run_id in done:
            continue
        writer.mark_started(rc)
        try:
            t0 = time.time()
            _model, best_path = train_fn(
                rc.ppo,
                rc.curriculum,
                num_players=_num_players_for(rc),
                phases=rc.phases,
            )
            gate_score = gate_reader(best_path) if gate_reader else None
            writer.mark_done(
                rc,
                checkpoint=best_path,
                wall_clock_s=time.time() - t0,
                gate_score=gate_score,
                git_sha=_safe_git_sha(best_path),
            )
        except Exception as exc:  # noqa: BLE001 - failure isolation (§5.1)
            writer.mark_failed(rc, error=repr(exc))
            continue
    return manifest_path


def _num_players_for(rc: RunConfig) -> int:
    """The race player count for a run (the last non-solo phase, fallback 4)."""
    race = [p for p in rc.phases if p.pool_kind is not None]
    return race[-1].num_players if race else 4


def _safe_git_sha(checkpoint: str) -> str | None:
    """Read the git SHA from a checkpoint sidecar if present (else None)."""
    meta = _meta_sidecar_path(checkpoint)
    try:
        with open(meta, "r", encoding="utf-8") as fh:
            return json.load(fh).get("git_sha")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# League matchup sampling (§7.2) + seat-rotation fairness (§7.4)
# ---------------------------------------------------------------------------


def seat_rotations(field: Sequence[str], num_players: int) -> list[list[str]]:
    """Cyclic seat rotations of ``field`` so each member occupies every seat (§7.4).

    Mirrors :func:`heat.ml.evaluate._seat_rotations` so the sampler and the
    evaluator agree on the fairness convention. Returns ``num_players``
    rotations; over the set each contender sits in every grid slot equally.
    """
    members = list(field)
    n = len(members)
    return [[members[(i + r) % n] for i in range(n)] for r in range(num_players)]


def balanced_fields(
    contenders: Sequence[str],
    *,
    num_players: int = 4,
    fields_per_contender: int = 30,
    seed: int = 0,
) -> list[tuple[str, ...]]:
    """Build a balanced set of distinct-``num_players`` fields (§7.2).

    Each contender appears in **>= fields_per_contender** distinct fields. Fields
    are drawn by repeatedly forming a size-``num_players`` group, preferring the
    most under-represented contenders (lowest appearance count, tie-broken
    deterministically by the seeded RNG) so coverage stays even. Far cheaper than
    enumerating ``C(n, num_players)`` (the scaling win, §7.1).

    Deterministic given ``(contenders, num_players, fields_per_contender, seed)``.

    Returns a list of label tuples (each sorted internally for a canonical form;
    seat order is handled later by :func:`seat_rotations`).
    """
    labels = sorted(contenders)
    n = len(labels)
    if n < num_players:
        raise ValueError(
            f"need >= {num_players} contenders, got {n}"
        )
    if fields_per_contender < 1:
        raise ValueError("fields_per_contender must be >= 1")

    rng = random.Random(seed)
    counts = {lbl: 0 for lbl in labels}
    seen: set[tuple[str, ...]] = set()
    fields: list[tuple[str, ...]] = []

    # Cap the attempts so a tiny pool (few distinct C(n,k) fields) terminates;
    # if the distinct-field space is exhausted before K is met, return what we
    # have rather than spinning forever.
    from math import comb

    max_distinct = comb(n, num_players)
    max_attempts = max(1000, fields_per_contender * n * 50)
    attempts = 0

    def _under_target() -> bool:
        return any(counts[lbl] < fields_per_contender for lbl in labels)

    while _under_target() and len(seen) < max_distinct and attempts < max_attempts:
        attempts += 1
        # Prefer the most under-represented contenders. Sort by (count, random
        # jitter) so ties break deterministically but without a fixed bias.
        jitter = {lbl: rng.random() for lbl in labels}
        ordered = sorted(labels, key=lambda l: (counts[l], jitter[l]))
        field = tuple(sorted(ordered[:num_players]))
        if field in seen:
            # Force diversity: pick a fresh combination by sampling randomly.
            field = tuple(sorted(rng.sample(labels, num_players)))
            if field in seen:
                continue
        seen.add(field)
        fields.append(field)
        for lbl in field:
            counts[lbl] += 1

    return fields


# ---------------------------------------------------------------------------
# Per-game outcome persistence (§7.4) -- the recompute substrate.
# ---------------------------------------------------------------------------


def _outcome_to_dict(o: GameOutcome) -> dict:
    """Serialize a :class:`GameOutcome` to a plain jsonl-able dict (§7.4)."""
    return {
        "game_index": o.game_index,
        "seed": o.seed,
        "num_players": o.num_players,
        "winner_id": o.winner_id,
        "winner_name": o.winner_name,
        "finish_order": list(o.finish_order),
        "total_rounds": o.total_rounds,
        "players": [
            {
                "player_id": p.player_id,
                "name": p.name,
                "agent_type": p.agent_type,
                "finish_position": p.finish_position,
                "final_lap": p.final_lap,
                "final_position": p.final_position,
                "heat_remaining": p.heat_remaining,
            }
            for p in o.players
        ],
    }


def _outcome_from_dict(d: Mapping[str, Any]) -> GameOutcome:
    """Rehydrate a :class:`GameOutcome` from a jsonl dict (inverse of above)."""
    return GameOutcome(
        game_index=d["game_index"],
        seed=d["seed"],
        num_players=d["num_players"],
        winner_id=d["winner_id"],
        winner_name=d["winner_name"],
        finish_order=tuple(d["finish_order"]),
        total_rounds=d["total_rounds"],
        players=tuple(
            PlayerOutcome(
                player_id=p["player_id"],
                name=p["name"],
                agent_type=p["agent_type"],
                finish_position=p["finish_position"],
                final_lap=p["final_lap"],
                final_position=p["final_position"],
                heat_remaining=p["heat_remaining"],
            )
            for p in d["players"]
        ),
    )


def write_outcomes_jsonl(outcomes: Sequence[GameOutcome], path: str) -> str:
    """Write a list of :class:`GameOutcome` to ``path`` as ``jsonl`` (§7.4).

    One line per game. The store is the recompute substrate: ``compute_elo`` /
    ``compute_trueskill`` are pure functions of the (reloaded) list, so ratings
    re-rate without replaying a single race.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(_outcome_to_dict(o)) + "\n")
    return path


def read_outcomes_jsonl(path: str) -> list[GameOutcome]:
    """Reload a per-game outcome store written by :func:`write_outcomes_jsonl`."""
    out: list[GameOutcome] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(_outcome_from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Attribution (§9) -- "what each factor brings", reported honestly.
# ---------------------------------------------------------------------------

#: The honesty prose (§9.3) the report MUST carry. Each entry is a (heading,
#: body) pair; :func:`render_attribution_md` emits them verbatim and the
#: regression guard test asserts the key phrases survive.
_HONESTY_CAVEATS: tuple[tuple[str, str], ...] = (
    (
        "Observational, not causal",
        "We swept a grid and *observed* ratings; we did not randomize at the "
        "unit level beyond seeds. Factor effects are associations under THIS "
        "recipe and track set, not causal guarantees.",
    ),
    (
        "Ratings have CIs",
        "Every effect is reported with uncertainty. Differences inside "
        "overlapping confidence intervals are 'not distinguishable', not 'zero'.",
    ),
    (
        "Factors interact",
        "A marginal effect averages over the other factors' levels; the grouped "
        "diff and the OLS can disagree when interactions are strong. When they "
        "do, inspect the relevant two-way cell means rather than a single "
        "marginal number.",
    ),
    (
        "Sampled-grid imbalance",
        "If random/LHS sampling was used the design is unbalanced and the "
        "grouped diffs inherit that imbalance -- the OLS becomes the primary "
        "read and this report says so.",
    ),
)


@dataclass(frozen=True)
class LadderEntry:
    """One contender's rating row in the joined ladder (§9.1)."""

    label: str
    rating: float
    rating_ci_lo: float
    rating_ci_hi: float
    games: int


@dataclass(frozen=True)
class FactorEffect:
    """The estimated effect of one factor on rating (§9.2)."""

    factor: str
    #: ``level -> (mean_rating, ci_lo, ci_hi)`` grouped marginal means.
    group_means: dict
    #: OLS coefficient per non-baseline level: ``level -> (coef, se)``.
    ols_coefs: dict


@dataclass(frozen=True)
class Attribution:
    """The full attribution result (§9.4): joined ladder + per-factor effects."""

    ladder: list[LadderEntry]
    effects: list[FactorEffect]
    #: ``run_id -> rating`` for the swept runs that joined to the ladder.
    run_ratings: dict


def join_ladder(
    ratings: Mapping[str, Any],
    *,
    rating_attr: str = "rating",
) -> list[LadderEntry]:
    """Build the ranked ladder table from a ratings mapping (§9.1).

    ``ratings`` is ``label -> EloRating`` (``rating`` + ``rating_ci`` + ``games``)
    or ``label -> TrueSkillRating`` (``mu`` + ``sigma`` + ``games``). ``rating_attr``
    selects the headline scalar (``"rating"`` for ELO, ``"mu"`` for TrueSkill).
    Sorted best-rating-first.
    """
    entries: list[LadderEntry] = []
    for label, r in ratings.items():
        value = float(getattr(r, rating_attr))
        if hasattr(r, "rating_ci"):
            lo, hi = r.rating_ci
        elif hasattr(r, "sigma"):
            # TrueSkill: a +/-1 sigma band around mu as the uncertainty interval.
            lo, hi = value - r.sigma, value + r.sigma
        else:
            lo, hi = value, value
        entries.append(
            LadderEntry(
                label=label,
                rating=value,
                rating_ci_lo=float(lo),
                rating_ci_hi=float(hi),
                games=int(getattr(r, "games", 0)),
            )
        )
    entries.sort(key=lambda e: e.rating, reverse=True)
    return entries


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_ci(xs: Sequence[float]) -> tuple[float, float, float]:
    """Mean + a normal-approx 95% CI on the mean (mean, lo, hi).

    A light, assumption-light interval for the grouped marginal means (§9.2a):
    ``mean +/- 1.96 * sd / sqrt(n)``. With ``n == 1`` the CI degenerates to the
    point (we cannot estimate spread from one run).
    """
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = _mean(xs)
    if n == 1:
        return (m, m, m)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    half = 1.96 * se
    return (m, m - half, m + half)


def grouped_diffs(
    rows: Sequence[Mapping[str, Any]],
    *,
    factor: str,
    rating_key: str = "rating",
) -> dict:
    """Grouped marginal means of ``rating_key`` per level of ``factor`` (§9.2a).

    ``rows`` is the joined manifest x ladder table (each row a dict with the
    factor values + a rating). Returns ``level -> (mean, ci_lo, ci_hi)``.
    """
    by_level: dict[Any, list[float]] = {}
    for row in rows:
        if factor not in row or rating_key not in row:
            continue
        by_level.setdefault(row[factor], []).append(float(row[rating_key]))
    return {
        _level_key(level): _mean_ci(vals) for level, vals in by_level.items()
    }


def _level_key(level: Any) -> str:
    """Stable string key for a factor level (tuples/ints/strs all hashable)."""
    if isinstance(level, (list, tuple)):
        return ",".join(str(x) for x in level)
    return str(level)


def ols_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    factors: Sequence[str],
    rating_key: str = "rating",
) -> dict:
    """OLS of rating on one-hot factor levels (§9.2b), via a normal-equation fit.

    Treats every factor as categorical (one-hot, dropping the first level per
    factor as the baseline -> the intercept). Returns
    ``factor -> {level: (coef, se)}`` -- each coefficient the marginal effect of
    that level vs the factor's baseline, holding the other factors fixed.

    A dependency-free least-squares fit (``(X'X)^-1 X'y`` with a small Gaussian
    solver) so the test suite needs no numpy/statsmodels. Falls back to a
    coefficient with ``se = inf`` when the design is rank-deficient for a column.
    """
    # Build the one-hot design. Baseline level per factor = its sorted-first.
    levels_by_factor: dict[str, list[Any]] = {}
    for f in factors:
        seen: list[Any] = []
        for row in rows:
            if f in row and row[f] not in seen:
                seen.append(row[f])
        levels_by_factor[f] = sorted(seen, key=_level_key)

    columns: list[tuple[str, Any]] = []  # (factor, level) for non-baseline levels
    for f in factors:
        for lvl in levels_by_factor[f][1:]:
            columns.append((f, lvl))

    y = [float(row[rating_key]) for row in rows if rating_key in row]
    design_rows = [row for row in rows if rating_key in row]
    n = len(y)
    p = 1 + len(columns)  # +1 intercept
    if n == 0 or n < p:
        # Under-determined: report point coefs with infinite SE (honesty §9.3).
        return {
            f: {
                _level_key(lvl): (0.0, float("inf"))
                for lvl in levels_by_factor[f][1:]
            }
            for f in factors
        }

    X = []
    for row in design_rows:
        vec = [1.0]
        for (f, lvl) in columns:
            vec.append(1.0 if row.get(f) == lvl else 0.0)
        X.append(vec)

    coefs, ses = _ols_fit(X, y)

    out: dict[str, dict] = {f: {} for f in factors}
    for i, (f, lvl) in enumerate(columns, start=1):
        out[f][_level_key(lvl)] = (coefs[i], ses[i])
    return out


def _ols_fit(X: list[list[float]], y: list[float]) -> tuple[list[float], list[float]]:
    """Ordinary least squares via the normal equations (dependency-free).

    Returns ``(coefs, standard_errors)``. Solves ``(X'X) b = X'y`` by Gaussian
    elimination, then ``se_j = sqrt(sigma^2 * (X'X)^-1_jj)`` with
    ``sigma^2 = RSS / (n - p)``.
    """
    n = len(X)
    p = len(X[0])
    # X'X and X'y
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r in range(n):
        for i in range(p):
            xty[i] += X[r][i] * y[r]
            for j in range(p):
                xtx[i][j] += X[r][i] * X[r][j]

    inv = _invert(xtx)
    if inv is None:
        return ([0.0] * p, [float("inf")] * p)
    coefs = [sum(inv[i][j] * xty[j] for j in range(p)) for i in range(p)]

    # Residual variance.
    rss = 0.0
    for r in range(n):
        pred = sum(coefs[i] * X[r][i] for i in range(p))
        rss += (y[r] - pred) ** 2
    dof = n - p
    sigma2 = rss / dof if dof > 0 else float("inf")
    ses = [
        (sigma2 * inv[i][i]) ** 0.5 if sigma2 != float("inf") and inv[i][i] >= 0
        else float("inf")
        for i in range(p)
    ]
    return (coefs, ses)


def _invert(mat: list[list[float]]) -> list[list[float]] | None:
    """Invert a small square matrix via Gauss-Jordan; None if singular."""
    n = len(mat)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
    for col in range(n):
        # Partial pivot.
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        a[col] = [v / piv for v in a[col]]
        for r in range(n):
            if r != col:
                factor = a[r][col]
                a[r] = [a[r][k] - factor * a[col][k] for k in range(2 * n)]
    return [row[n:] for row in a]


def build_attribution(
    manifest_rows: Sequence[ManifestRow],
    ratings: Mapping[str, Any],
    *,
    factors: Sequence[str],
    rating_attr: str = "mu",
) -> Attribution:
    """Join the manifest factor vectors to the ladder ratings + estimate effects.

    For each ``done`` manifest row whose ``run_id`` has a rating, build a tidy
    row of ``{factor: level, ..., "rating": value}``; compute grouped diffs and
    the OLS per factor (§9.2). ``rating_attr`` selects the headline scalar
    (``"mu"`` for TrueSkill, ``"rating"`` for ELO).
    """
    ladder = join_ladder(ratings, rating_attr=("rating" if rating_attr == "rating" else rating_attr))
    run_ratings: dict[str, float] = {}
    tidy: list[dict] = []
    for row in manifest_rows:
        if row.status != "done" or row.run_id not in ratings:
            continue
        value = float(getattr(ratings[row.run_id], rating_attr))
        run_ratings[row.run_id] = value
        tidy.append({**{f: row.factors.get(f) for f in factors}, "rating": value})

    effects: list[FactorEffect] = []
    ols = ols_effects(tidy, factors=factors, rating_key="rating") if tidy else {}
    for f in factors:
        effects.append(
            FactorEffect(
                factor=f,
                group_means=grouped_diffs(tidy, factor=f, rating_key="rating"),
                ols_coefs=ols.get(f, {}),
            )
        )
    return Attribution(ladder=ladder, effects=effects, run_ratings=run_ratings)


# ---- Emitters (§9.4) --------------------------------------------------------


def render_attribution_md(attribution: Attribution, *, rating_label: str = "TrueSkill mu") -> str:
    """Render the human-readable ``attribution.md`` (§9.4), honesty prose included.

    MUST contain the §9.3 caveats verbatim (the regression guard asserts the key
    phrases survive). Emits: the ranked ladder, a per-factor effects table
    (grouped diff + OLS coef, both with uncertainty), and the honesty section.
    """
    lines: list[str] = []
    lines.append("# League attribution report")
    lines.append("")
    lines.append(f"Headline rating: **{rating_label}**.")
    lines.append("")

    lines.append("## Ranked ladder")
    lines.append("")
    lines.append("| rank | contender | rating | ci_lo | ci_hi | games |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, e in enumerate(attribution.ladder, start=1):
        lines.append(
            f"| {rank} | {e.label} | {e.rating:.2f} | {e.rating_ci_lo:.2f} | "
            f"{e.rating_ci_hi:.2f} | {e.games} |"
        )
    lines.append("")

    lines.append("## Per-factor effects")
    lines.append("")
    lines.append(
        "Grouped marginal means (assumption-light headline) and OLS coefficients "
        "(marginal effect holding other factors fixed). Both carry uncertainty."
    )
    lines.append("")
    for eff in attribution.effects:
        lines.append(f"### {eff.factor}")
        lines.append("")
        lines.append("| level | grouped mean | mean ci_lo | mean ci_hi | OLS coef | OLS se |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        all_levels = sorted(
            set(eff.group_means) | set(eff.ols_coefs)
        )
        for lvl in all_levels:
            gm = eff.group_means.get(lvl)
            coef = eff.ols_coefs.get(lvl)
            gm_str = (
                f"{gm[0]:.2f} | {gm[1]:.2f} | {gm[2]:.2f}"
                if gm is not None
                else " | | "
            )
            coef_str = (
                f"{coef[0]:.3f} | {coef[1]:.3f}"
                if coef is not None
                else "(baseline) | "
            )
            lines.append(f"| {lvl} | {gm_str} | {coef_str} |")
        lines.append("")

    lines.append("## How to read this (honest caveats)")
    lines.append("")
    for heading, body in _HONESTY_CAVEATS:
        lines.append(f"- **{heading}.** {body}")
    lines.append("")
    return "\n".join(lines)


def render_ladder_csv(attribution: Attribution) -> str:
    """Render ``ladder.csv`` (§9.4): every contender's rating/CI/games."""
    lines = ["label,rating,rating_ci_lo,rating_ci_hi,games"]
    for e in attribution.ladder:
        lines.append(
            f"{e.label},{e.rating},{e.rating_ci_lo},{e.rating_ci_hi},{e.games}"
        )
    return "\n".join(lines) + "\n"


def effects_to_json(attribution: Attribution) -> str:
    """Render ``effects.json`` (§9.4): machine-readable per-factor effects."""
    payload = {
        "ladder": [dataclasses.asdict(e) for e in attribution.ladder],
        "effects": {
            eff.factor: {
                "group_means": {
                    lvl: {"mean": m, "ci_lo": lo, "ci_hi": hi}
                    for lvl, (m, lo, hi) in eff.group_means.items()
                },
                "ols_coefs": {
                    lvl: {"coef": c, "se": s}
                    for lvl, (c, s) in eff.ols_coefs.items()
                },
            }
            for eff in attribution.effects
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def write_attribution_artifacts(
    attribution: Attribution,
    out_dir: str,
    *,
    rating_label: str = "TrueSkill mu",
) -> dict[str, str]:
    """Write ``attribution.md`` / ``ladder.csv`` / ``effects.json`` to ``out_dir``.

    Returns the ``{name: path}`` map of artifacts written. Used by the heavy
    ``experiments/run_league.py`` shell.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "attribution.md": os.path.join(out_dir, "attribution.md"),
        "ladder.csv": os.path.join(out_dir, "ladder.csv"),
        "effects.json": os.path.join(out_dir, "effects.json"),
    }
    with open(paths["attribution.md"], "w", encoding="utf-8") as fh:
        fh.write(render_attribution_md(attribution, rating_label=rating_label))
    with open(paths["ladder.csv"], "w", encoding="utf-8") as fh:
        fh.write(render_ladder_csv(attribution))
    with open(paths["effects.json"], "w", encoding="utf-8") as fh:
        fh.write(effects_to_json(attribution))
    return paths
