from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .failure_meta import (
    CORE_META_PREDICTORS,
    DEFAULT_FAILURE_THRESHOLDS,
    META_DESCRIPTORS,
    PRIMARY_FAILURE_THRESHOLD,
    add_failure_labels,
    build_failure_group_summary,
    build_meta_data_dictionary,
    build_meta_model_dataset,
    build_molecule_failure_summary,
    build_run_failure_summary,
    validate_failure_meta_outputs,
)
from .run_baselines import _flatten_report, _write_table


def build_failure_meta_artifacts(
    predictions: pd.DataFrame,
    model_results: pd.DataFrame,
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
    descriptors: pd.DataFrame,
    data_dir: Path,
    results_dir: Path,
    reports_dir: Path,
    thresholds: tuple[float, ...] = DEFAULT_FAILURE_THRESHOLDS,
    primary_threshold: float = PRIMARY_FAILURE_THRESHOLD,
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    labeled = add_failure_labels(
        predictions, thresholds=thresholds, primary_threshold=primary_threshold
    )
    run_summary = build_run_failure_summary(labeled, thresholds=thresholds)
    molecule_summary = build_molecule_failure_summary(
        labeled,
        thresholds=thresholds,
        primary_threshold=primary_threshold,
        include_dummy=False,
    )
    group_summary = build_failure_group_summary(
        run_summary, primary_threshold=primary_threshold
    )
    meta_dataset = build_meta_model_dataset(
        model_results,
        run_summary,
        assignments,
        targets,
        descriptors,
        descriptor_names=META_DESCRIPTORS,
        primary_threshold=primary_threshold,
    )
    dictionary = build_meta_data_dictionary(meta_dataset)
    errors = validate_failure_meta_outputs(
        labeled, run_summary, meta_dataset, model_results
    )

    paths = {
        "prediction_failures_csv": results_dir / "prediction_failures.csv",
        "prediction_failures_parquet": results_dir / "prediction_failures.parquet",
        "run_failure_summary_csv": results_dir / "run_failure_summary.csv",
        "run_failure_summary_parquet": results_dir / "run_failure_summary.parquet",
        "molecule_failure_summary_csv": results_dir / "molecule_failure_summary.csv",
        "molecule_failure_summary_parquet": results_dir / "molecule_failure_summary.parquet",
        "failure_group_summary_csv": results_dir / "failure_group_summary.csv",
        "meta_model_dataset_csv": data_dir / "meta_model_dataset.csv",
        "meta_model_dataset_parquet": data_dir / "meta_model_dataset.parquet",
        "meta_dataset_dictionary_csv": reports_dir / "meta_dataset_dictionary.csv",
        "failure_analysis_report_json": reports_dir / "failure_analysis_report.json",
        "failure_analysis_report_csv": reports_dir / "failure_analysis_report.csv",
    }
    _write_table(
        labeled,
        paths["prediction_failures_csv"],
        paths["prediction_failures_parquet"],
    )
    _write_table(
        run_summary,
        paths["run_failure_summary_csv"],
        paths["run_failure_summary_parquet"],
    )
    _write_table(
        molecule_summary,
        paths["molecule_failure_summary_csv"],
        paths["molecule_failure_summary_parquet"],
    )
    group_summary.to_csv(paths["failure_group_summary_csv"], index=False)
    _write_table(
        meta_dataset,
        paths["meta_model_dataset_csv"],
        paths["meta_model_dataset_parquet"],
    )
    dictionary.to_csv(paths["meta_dataset_dictionary_csv"], index=False)

    report = {
        "status": "fail" if errors else "pass",
        "blocking_errors": errors,
        "prediction_rows": int(len(labeled)),
        "completed_runs": int(len(run_summary)),
        "molecules_with_held_out_predictions": int(len(molecule_summary)),
        "molecule_summary_excludes_dummy": True,
        "meta_dataset_rows": int(len(meta_dataset)),
        "distinct_tasks": int(meta_dataset["task_id"].nunique()),
        "selection_eligible_rows": int(meta_dataset["selection_eligible"].sum()),
        "predictor_columns": int(sum(column.startswith("x_") for column in meta_dataset)),
        "recommended_initial_predictors": list(CORE_META_PREDICTORS),
        "outcome_columns": int(sum(column.startswith("y_") for column in meta_dataset)),
        "failure_thresholds_log_s": list(thresholds),
        "primary_failure_threshold_log_s": float(primary_threshold),
        "q_i_definition": (
            "q_i = 1 when absolute(true_log_s - predicted_log_s) is strictly greater than "
            f"{primary_threshold:g} logS; otherwise q_i = 0."
        ),
        "primary_threshold_interpretation": (
            "An absolute error above 1.0 log10(mol/L) corresponds to a solubility error larger "
            "than one order of magnitude. Thresholds 0.5 and 2.0 are retained for sensitivity "
            "analysis."
        ),
        "predictor_leakage_policy": (
            "All x_ columns are available before final test evaluation. Test-molecule structure "
            "and train-test chemical similarity may be used, but no held-out test logS value or "
            "test-error statistic is included in X."
        ),
        "selection_outcome_policy": (
            "Dummy regressors are retained as baselines but selection_eligible is false. Among "
            "eligible candidates, y_top3_candidate identifies the three lowest test-RMSE rows "
            "within each task, and y_rmse_regret_vs_best measures distance from the task winner."
        ),
        "validation_policy": (
            "Future meta-model evaluation must hold out complete tasks or datasets. Randomly "
            "splitting the 180 run rows would leak repeated task-level meta-features."
        ),
        "current_limitation": (
            "The table contains 15 correlated tasks from one master benchmark. It is suitable "
            "for exploratory analysis and pipeline validation, not a final publishable "
            "meta-model. Source-holdout and external-dataset tasks must be added."
        ),
        "meta_descriptors": list(META_DESCRIPTORS),
    }
    output_files: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if name.startswith("failure_analysis_report"):
            continue
        output_files[name] = {"path": str(path), "bytes": int(path.stat().st_size)}
    report["output_files"] = output_files
    paths["failure_analysis_report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["failure_analysis_report_csv"], index=False)
    if errors:
        raise RuntimeError("Failure/meta artifacts failed validation:\n- " + "\n- ".join(errors))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label molecule-level errors and build the run-level meta-learning table."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--primary-threshold", type=float, default=PRIMARY_FAILURE_THRESHOLD)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=list(DEFAULT_FAILURE_THRESHOLDS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_paths = {
        "predictions": root / "results/test_predictions.parquet",
        "model_results": root / "results/model_results.parquet",
        "assignments": root / "data_processed/split_assignments.parquet",
        "targets": root / "data_processed/modeling_targets.parquet",
        "descriptors": root / "data_processed/rdkit_descriptors.parquet",
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required failure-analysis inputs are missing:\n- " + "\n- ".join(missing)
        )

    paths = build_failure_meta_artifacts(
        predictions=pd.read_parquet(input_paths["predictions"]),
        model_results=pd.read_parquet(input_paths["model_results"]),
        assignments=pd.read_parquet(input_paths["assignments"]),
        targets=pd.read_parquet(input_paths["targets"]),
        descriptors=pd.read_parquet(input_paths["descriptors"]),
        data_dir=root / "data_processed",
        results_dir=root / "results",
        reports_dir=root / "reports",
        thresholds=tuple(args.thresholds),
        primary_threshold=args.primary_threshold,
    )
    print("Failure characterization and meta-dataset construction completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
