from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .splits import (
    DEFAULT_SPLIT_SEEDS,
    build_split_assignments,
    build_split_quality_summary,
    build_split_report,
    validate_split_assignments,
)


def build_split_artifacts(
    targets: pd.DataFrame,
    morgan: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    assignments = build_split_assignments(targets, morgan, seeds=seeds)
    errors = validate_split_assignments(assignments, targets, expected_repeats=len(seeds))
    summary = build_split_quality_summary(assignments, targets)
    report = build_split_report(assignments, summary, errors)

    paths = {
        "split_assignments_csv": output_dir / "split_assignments.csv",
        "split_assignments_parquet": output_dir / "split_assignments.parquet",
        "split_quality_csv": reports_dir / "split_quality.csv",
        "split_quality_json": reports_dir / "split_quality.json",
    }
    assignments.to_csv(paths["split_assignments_csv"], index=False)
    assignments.to_parquet(paths["split_assignments_parquet"], index=False, compression="zstd")
    summary.to_csv(paths["split_quality_csv"], index=False)
    paths["split_quality_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_readback = pd.read_csv(
        paths["split_assignments_csv"],
        usecols=["molecule_id", "split_strategy", "repeat", "partition"],
    )
    parquet_readback = pd.read_parquet(
        paths["split_assignments_parquet"],
        columns=["molecule_id", "split_strategy", "repeat", "partition"],
    )
    if not csv_readback.equals(parquet_readback):
        errors.append("CSV and Parquet split-assignment identifiers differ after read-back")

    if errors:
        report["status"] = "fail"
        report["blocking_errors"] = errors
        paths["split_quality_json"].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        joined = "\n- ".join(errors)
        raise RuntimeError(f"Split validation failed:\n- {joined}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible random, scaffold, and low-similarity benchmark splits."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--morgan", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPLIT_SEEDS),
        help="One or more integer seeds; each seed creates one repeat per strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    targets_path = (args.targets or root / "data_processed/modeling_targets.parquet").resolve()
    morgan_path = (args.morgan or root / "data_processed/morgan_fingerprints.parquet").resolve()
    output_dir = (args.output_dir or root / "data_processed").resolve()
    reports_dir = (args.reports_dir or root / "reports").resolve()

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Modeling targets not found: {targets_path}. Run aquasol-build-targets first."
        )
    if not morgan_path.exists():
        raise FileNotFoundError(
            f"Morgan fingerprints not found: {morgan_path}. Run aquasol-build-features first."
        )

    paths = build_split_artifacts(
        targets=pd.read_parquet(targets_path),
        morgan=pd.read_parquet(morgan_path),
        output_dir=output_dir,
        reports_dir=reports_dir,
        seeds=tuple(args.seeds),
    )
    print("Split build completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
