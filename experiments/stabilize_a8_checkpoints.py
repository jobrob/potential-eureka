"""Build the registered A8 S3 late-checkpoint averages (generation G0004)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from heat.ml.policy_registry import load_generation
from heat.ml.selfplay.checkpoint import average_policy_checkpoints, load_policy


@dataclass(frozen=True)
class AverageSpec:
    """Frozen inputs and output identity for one derived G0004 run."""

    agent_id: str
    seed: int
    source_checkpoint_ids: tuple[str, ...]
    sources: tuple[str, ...]
    output: str


def _specs(output_dir: Path) -> tuple[AverageSpec, ...]:
    """Return the recipe-level three-milestone average for every G0002 seed."""
    milestones = (500_000, 750_000, 1_000_000)
    return tuple(
        AverageSpec(
            agent_id=f"G0004-R{seed:02d}",
            seed=seed,
            source_checkpoint_ids=tuple(
                f"G0002-R{seed:02d}@{step // 1000}K" for step in milestones
            ),
            sources=tuple(
                f"runs/a8_s1_5ep_seed{seed}/step_{step}.pt" for step in milestones
            ),
            output=(output_dir / f"g0004_r{seed:02d}.pt").as_posix(),
        )
        for seed in range(3)
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one generated checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the frozen output location and machine-readable receipt path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/a8_s3_average")
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("runs/a8_s3_average_receipt.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build every G0004 run and persist hashes for registry finalization."""
    args = _parse_args(argv)
    generation = load_generation("G0004")
    if generation.get("status") != "planned":
        raise RuntimeError("G0004 must be registered with status='planned' first")

    rows: list[dict[str, object]] = []
    for spec in _specs(args.output_dir):
        output = Path(spec.output)
        average_policy_checkpoints(spec.sources, output)
        # A load round-trip is the cheapest complete architecture/codec check.
        load_policy(output)
        row = asdict(spec)
        row.update({"bytes": output.stat().st_size, "sha256": _sha256(output)})
        rows.append(row)
        print(
            f"built {spec.agent_id} bytes={row['bytes']} sha256={row['sha256']}",
            flush=True,
        )

    payload = {
        "generation_id": "G0004",
        "method": "uniform_parameter_mean",
        "runs": rows,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.receipt)
    print(f"receipt={args.receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
