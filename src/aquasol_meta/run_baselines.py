from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .modeling import (
    ALL_MODELS,
    CORE_MODELS,
    FEATURE_TYPES,
    run_baseline_experiments,
)


def _write_table(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    csv_rows = len(pd.read_csv(csv_path))
    parquet_rows = len(pd.read_parquet(parquet_path))
    if csv_rows != len(frame) or parquet_rows != len(frame):
        raise RuntimeError(
            f"Read-back row mismatch for {csv_path.name}: "
            f"memory={len(frame):,}, csv={csv_rows:,}, parquet={parquet_rows:,}"
        )


def _flatten_report(report: dict[str, object]) -> pd.DataFrame:
    """Represent a nested JSON report as a lossless long-form CSV table."""
    rows: list[dict[str, object]] = []

    def visit(path: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}.{key}" if path else str(key), item)
        elif isinstance(value, list):
            if not value:
                rows.append({"metric": path, "value": "[]"})
            for index, item in enumerate(value):
                item_path = f"{path}.{index}" if isinstance(item, dict) else path
                visit(item_path, item)
        else:
            rows.append({"metric": path, "value": value})

    visit("", report)
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_baseline_artifacts(
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    results_dir: Path,
    reports_dir: Path,
    strategies: tuple[str, ...] = ("random",),
    repeats: tuple[int, ...] = (1,),
    model_names: tuple[str, ...] = CORE_MODELS,
    show_progress: bool = False,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results, predictions, tuning, report = run_baseline_experiments(
        targets,
        assignments,
        feature_tables=feature_tables,
        strategies=strategies,
        repeats=repeats,
        model_names=model_names,
        progress=print if show_progress else None,
    )
    if results.empty or predictions.empty or tuning.empty:
        raise RuntimeError("Baseline run produced one or more empty result tables")

    paths = {
        "model_results_csv": results_dir / "model_results.csv",
        "model_results_parquet": results_dir / "model_results.parquet",
        "test_predictions_csv": results_dir / "test_predictions.csv",
        "test_predictions_parquet": results_dir / "test_predictions.parquet",
        "hyperparameter_results_csv": results_dir / "hyperparameter_results.csv",
        "hyperparameter_results_parquet": results_dir / "hyperparameter_results.parquet",
        "baseline_run_report": reports_dir / "baseline_run_report.json",
        "baseline_run_report_csv": reports_dir / "baseline_run_report.csv",
    }
    _write_table(results, paths["model_results_csv"], paths["model_results_parquet"])
    _write_table(
        predictions,
        paths["test_predictions_csv"],
        paths["test_predictions_parquet"],
    )
    _write_table(
        tuning,
        paths["hyperparameter_results_csv"],
        paths["hyperparameter_results_parquet"],
    )
    report["output_files"] = {
        name: {"path": str(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
        if name not in {"baseline_run_report", "baseline_run_report_csv"}
    }
    paths["baseline_run_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["baseline_run_report_csv"], index=False)
    if report["failed_runs"]:
        failures = "\n- ".join(
            f"{item['run_id']}: {item['reason']}" for item in report["failed_runs"]
        )
        raise RuntimeError(f"One or more baseline runs failed:\n- {failures}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune, fit, and evaluate leakage-safe aqueous-solubility baselines."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_TYPES,
        default=list(FEATURE_TYPES),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("random", "scaffold", "low_similarity"),
        default=["random"],
    )
    parser.add_argument("--repeats", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(CORE_MODELS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    targets_path = (args.targets or root / "data_processed/modeling_targets.parquet").resolve()
    splits_path = (args.splits or root / "data_processed/split_assignments.parquet").resolve()
    results_dir = (args.results_dir or root / "results").resolve()
    reports_dir = (args.reports_dir or root / "reports").resolve()

    required = [targets_path, splits_path]
    feature_paths = {
        feature_type: root / "data_processed" / f"{feature_type}.parquet"
        for feature_type in args.features
    }
    required.extend(feature_paths.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required modeling files are missing:\n- " + "\n- ".join(missing))

    paths = build_baseline_artifacts(
        targets=pd.read_parquet(targets_path),
        assignments=pd.read_parquet(splits_path),
        feature_tables={
            name: pd.read_parquet(path) for name, path in feature_paths.items()
        },
        results_dir=results_dir,
        reports_dir=reports_dir,
        strategies=tuple(args.strategies),
        repeats=tuple(args.repeats),
        model_names=tuple(args.models),
        show_progress=True,
    )
    print("Baseline modeling completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
