from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .targets import (
    build_modeling_target_report,
    build_modeling_targets,
    build_reliability_summary,
    validate_modeling_targets,
)


def build_target_artifacts(
    observations: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    write_csv: bool = True,
    consistent_range_max: float = 0.5,
    conflict_range_min: float = 1.0,
    review_log_s_min: float = -15.0,
    review_log_s_max: float = 2.5,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    targets = build_modeling_targets(
        observations,
        consistent_range_max=consistent_range_max,
        conflict_range_min=conflict_range_min,
        review_log_s_min=review_log_s_min,
        review_log_s_max=review_log_s_max,
    )
    expected_observations = int(
        observations["model_eligible"].fillna(False).astype(bool).sum()
    )
    errors = validate_modeling_targets(targets, expected_observations)
    report = build_modeling_target_report(targets, expected_observations)
    report["blocking_errors"] = errors
    report["status"] = "fail" if errors else "pass"
    reliability_summary = build_reliability_summary(targets)

    paths = {
        "modeling_targets_parquet": output_dir / "modeling_targets.parquet",
        "modeling_targets_report": reports_dir / "modeling_targets_report.json",
        "modeling_reliability_summary": reports_dir / "modeling_reliability_summary.csv",
    }
    targets.to_parquet(paths["modeling_targets_parquet"], index=False)
    reliability_summary.to_csv(paths["modeling_reliability_summary"], index=False)
    paths["modeling_targets_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if write_csv:
        paths["modeling_targets_csv"] = output_dir / "modeling_targets.csv"
        targets.to_csv(paths["modeling_targets_csv"], index=False)

    if errors:
        joined = "\n- ".join(errors)
        raise RuntimeError(f"Modelling-target validation failed:\n- {joined}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one auditable solubility target per standardized molecule."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--master", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--consistent-range-max", type=float, default=0.5)
    parser.add_argument("--conflict-range-min", type=float, default=1.0)
    parser.add_argument("--review-log-s-min", type=float, default=-15.0)
    parser.add_argument("--review-log-s-max", type=float, default=2.5)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    master_path = (args.master or root / "data_processed/master_observations.parquet").resolve()
    output_dir = (args.output_dir or root / "data_processed").resolve()
    reports_dir = (args.reports_dir or root / "reports").resolve()

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master table not found: {master_path}. Run aquasol-build-master first."
        )
    observations = pd.read_parquet(master_path)
    paths = build_target_artifacts(
        observations,
        output_dir=output_dir,
        reports_dir=reports_dir,
        write_csv=not args.no_csv,
        consistent_range_max=args.consistent_range_max,
        conflict_range_min=args.conflict_range_min,
        review_log_s_min=args.review_log_s_min,
        review_log_s_max=args.review_log_s_max,
    )
    print("Modelling-target build completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
