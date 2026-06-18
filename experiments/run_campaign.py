"""Sprint 8D: the mass-training campaign launcher (thin, heavy-compute shell).

Defines the default campaign :class:`~heat.ml.sweep.SweepSpec` (§4.1/§6) and
calls :func:`heat.ml.sweep.run_campaign`, which runs each config sequentially
through ``train_self_play`` (8C), isolates per-run failures, and writes a
resumable ``manifest.jsonl``. All the pure logic (spec expansion, manifest,
config applier) lives in ``src/heat/ml/sweep.py`` and is unit-tested; this file
is just the wiring + the GPU-burning ``main()``.

**This is heavy compute** (~8-12 GPU-hours for the default campaign, §6) -- it is
NOT run by the test suite. Launch manually:

    PYTHONPATH=src python experiments/run_campaign.py

Resumes automatically: re-launching with the same spec skips runs already
``done`` in the manifest.
"""

from __future__ import annotations

import argparse

from heat.ml.sweep import FactorAxis, SweepSpec, run_campaign
from heat.ml.training import load_meta


def default_spec() -> SweepSpec:
    """The campaign-1 grid (§4.1/§6): ~3 primary x ~2 secondary x >= 2 seeds.

    Primary factors (swept, 2 levels each):
      * ``phases.0.steps`` -- solo pretrain step count.
      * ``phases.steps_ratio`` -- the weak/mixed/strong step-budget profile (a
        per-opponent-phase multiplier tuple): a flat ramp vs a strong-heavy one.
      * ``schedule.pool_kinds`` -- opponent ramp composition: the full
        weak->mixed->strong vs the shorter weak->strong.

    Secondary factors (2 levels each; folded in -- the grid keeps them crossed):
      * ``ppo.shaping_weight`` -- dense-shaping weight.
      * ``ppo.net_profile`` -- network size (default vs larger).

    With 2 seeds this is 2*2*2*2*2 * 2 = 64 grid runs. Trim with ``--max-configs``
    + ``--method random`` (or ``lhs``) for a cheaper first pass; gamma and
    ``randomize_seat`` stay pinned at the validated recipe (§4.1).
    """
    return SweepSpec(
        axes=(
            FactorAxis("phases.0.steps", (200_000, 400_000)),
            FactorAxis("phases.steps_ratio", ((1.0, 1.0, 1.0), (1.0, 1.0, 2.0))),
            FactorAxis(
                "schedule.pool_kinds", ("weak,mixed,strong", "weak,strong")
            ),
            FactorAxis("ppo.shaping_weight", (0.0, 0.05)),
            FactorAxis("ppo.net_profile", ("small", "large")),
        ),
        seeds=(0, 1),
        method="grid",
        base_preset="sprint_8c",
    )


def _gate_from_meta(checkpoint: str) -> float | None:
    """Read the run's final gate score from the checkpoint sidecar, if present."""
    try:
        meta = load_meta(checkpoint)
    except OSError:
        return None
    # The sidecar does not currently carry the gate score directly; the manifest
    # records None when unavailable. Hook left for when save_checkpoint grows it.
    return meta.get("gate_score")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 8D mass-training campaign")
    parser.add_argument("--out-dir", default="campaign_out")
    parser.add_argument("--method", default=None, choices=["grid", "random", "lhs"])
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    spec = default_spec()
    if args.method is not None:
        spec = SweepSpec(
            axes=spec.axes,
            seeds=spec.seeds,
            method=args.method,
            max_configs=args.max_configs,
            base_preset=spec.base_preset,
        )

    runs = spec.expand()
    print(f"Campaign: {len(runs)} runs -> {args.out_dir}/manifest.jsonl", flush=True)
    manifest = run_campaign(
        spec,
        out_dir=args.out_dir,
        gate_reader=_gate_from_meta,
        resume=not args.no_resume,
    )
    print(f"Done. Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
