from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .failure_meta import (
    add_failure_labels,
    build_failure_group_summary,
    build_meta_data_dictionary,
    build_meta_model_dataset,
    build_run_failure_summary,
)
from .modeling import ALL_MODELS, CORE_MODELS, FEATURE_TYPES, run_baseline_experiments
from .run_baselines import _flatten_report, _write_table


def _concatenate_parquet_files(sources: list[Path], destination: Path) -> int:
    if destination.exists():
        destination.unlink()
    schema = pa.unify_schemas(
        [pq.ParquetFile(source).schema_arrow for source in sources],
        promote_options="permissive",
    )
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for source in sources:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])
                for field in schema:
                    if field.name not in table.column_names:
                        table = table.append_column(
                            field.name,
                            pa.nulls(table.num_rows, type=field.type),
                        )
                table = table.select(schema.names).cast(schema)
                if writer is None:
                    writer = pq.ParquetWriter(destination, schema, compression="zstd")
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No Parquet rows were available to concatenate")
    return rows


def _write_partitioned_prediction_dataset(
    sources: list[Path],
    destination: Path,
) -> int:
    """Write bounded Parquet parts grouped by held-out source for GitHub compatibility."""
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    datasets = [ds.dataset(source, format="parquet", partitioning="hive") for source in sources]
    schema = pa.unify_schemas(
        [dataset.schema for dataset in datasets],
        promote_options="permissive",
    )
    if "heldout_source" not in schema.names:
        raise ValueError("Source prediction dataset has no heldout_source column")
    rows = 0
    destination.mkdir(parents=True)
    for source_number, dataset in enumerate(datasets):
        for batch_number, batch in enumerate(dataset.to_batches(batch_size=100_000)):
            table = pa.Table.from_batches([batch])
            for field in schema:
                if field.name not in table.column_names:
                    table = table.append_column(
                        field.name,
                        pa.nulls(table.num_rows, type=field.type),
                    )
            table = table.select(schema.names).cast(schema)
            pq.write_to_dataset(
                table,
                root_path=destination,
                partition_cols=["heldout_source"],
                basename_template=(
                    f"part-source{source_number:02d}-batch{batch_number:05d}-{{i}}.parquet"
                ),
                compression="zstd",
            )
            rows += table.num_rows
    if rows == 0:
        raise RuntimeError("No source prediction rows were available to write")
    oversized = [
        path for path in destination.rglob("*.parquet") if path.stat().st_size >= 95_000_000
    ]
    if oversized:
        raise RuntimeError(
            "Partitioned source prediction dataset contains files at or above 95 MB: "
            + ", ".join(str(path) for path in oversized)
        )
    return rows


def _path_size(path: Path) -> int:
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def _single_value(frame: pd.DataFrame, column: str, task_id: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{task_id} has inconsistent {column} values")
    return values.iloc[0]


def _task_inputs(task: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_columns = [
        "molecule_id",
        "split_strategy",
        "repeat",
        "seed",
        "partition",
        "scaffold_group",
        "maximum_train_similarity",
    ]
    target_columns = [
        "molecule_id",
        "log_s_target",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "log_s_range",
        "include_primary_benchmark",
    ]
    return task[assignment_columns].copy(), task[target_columns].copy()


def _rebuild_source_meta(
    tasks: pd.DataFrame,
    results: pd.DataFrame,
    run_summary: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute rankings across every candidate in each complete source task."""
    frames: list[pd.DataFrame] = []
    for task_id, task in tasks.groupby("task_id", sort=True):
        task_results = results.loc[results["task_id"].eq(task_id)]
        task_summary = run_summary.loc[run_summary["task_id"].eq(task_id)]
        if task_results.empty or set(task_results["run_id"]) != set(task_summary["run_id"]):
            raise RuntimeError(f"Source task {task_id} has incomplete result coverage")
        assignments, targets = _task_inputs(task)
        task_meta = build_meta_model_dataset(
            task_results,
            task_summary,
            assignments,
            targets,
            descriptors,
        )
        heldout_source = str(_single_value(task, "heldout_source", task_id))
        task_meta.insert(2, "heldout_source", heldout_source)
        task_meta["x_split_strategy"] = "source_holdout"
        task_meta["x_evaluation_design"] = "source_holdout"
        frames.append(task_meta)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["task_id", "x_feature_type", "x_model"], kind="stable"
    ).reset_index(drop=True)


def _combine_meta_datasets(
    base_meta: pd.DataFrame,
    source_meta: pd.DataFrame,
) -> pd.DataFrame:
    base = base_meta.copy()
    source = source_meta.copy()
    base["heldout_source"] = "not_applicable"
    base["x_evaluation_design"] = "within_benchmark_split"
    source["x_evaluation_design"] = "source_holdout"
    combined = pd.concat([base, source], ignore_index=True, sort=False)
    if combined["run_id"].duplicated().any():
        raise ValueError("Combined meta-dataset contains duplicate run IDs")
    identifier_columns = [
        "run_id",
        "task_id",
        "heldout_source",
        "repeat",
        "seed",
        "selection_eligible",
    ]
    predictor_columns = sorted(column for column in combined if column.startswith("x_"))
    outcome_columns = sorted(column for column in combined if column.startswith("y_"))
    return combined[identifier_columns + predictor_columns + outcome_columns].sort_values(
        ["x_evaluation_design", "task_id", "x_feature_type", "x_model"], kind="stable"
    ).reset_index(drop=True)


def build_source_holdout_model_artifacts(
    tasks: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    descriptors: pd.DataFrame,
    base_meta: pd.DataFrame,
    data_dir: Path,
    results_dir: Path,
    reports_dir: Path,
    model_names: tuple[str, ...] = CORE_MODELS,
    show_progress: bool = False,
    append_existing: bool = False,
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_model_results_csv": results_dir / "source_holdout_model_results.csv",
        "source_model_results_parquet": results_dir / "source_holdout_model_results.parquet",
        "source_predictions_parquet": results_dir / "source_holdout_prediction_failures.parquet",
        "source_tuning_csv": results_dir / "source_holdout_hyperparameter_results.csv",
        "source_tuning_parquet": results_dir / "source_holdout_hyperparameter_results.parquet",
        "source_run_summary_csv": results_dir / "source_holdout_run_failure_summary.csv",
        "source_run_summary_parquet": results_dir / "source_holdout_run_failure_summary.parquet",
        "source_group_summary_csv": results_dir / "source_holdout_failure_group_summary.csv",
        "source_meta_csv": data_dir / "source_holdout_meta_model_dataset.csv",
        "source_meta_parquet": data_dir / "source_holdout_meta_model_dataset.parquet",
        "combined_meta_csv": data_dir / "meta_model_dataset_with_source_holdouts.csv",
        "combined_meta_parquet": data_dir / "meta_model_dataset_with_source_holdouts.parquet",
        "combined_dictionary_csv": reports_dir
        / "meta_dataset_with_source_holdouts_dictionary.csv",
        "source_model_report_json": reports_dir / "source_holdout_model_report.json",
        "source_model_report_csv": reports_dir / "source_holdout_model_report.csv",
    }
    existing: dict[str, pd.DataFrame] = {}
    existing_report: dict[str, Any] = {}
    if append_existing:
        required_existing = [
            paths["source_model_results_parquet"],
            paths["source_predictions_parquet"],
            paths["source_tuning_parquet"],
            paths["source_run_summary_parquet"],
            paths["source_meta_parquet"],
            paths["source_model_report_json"],
        ]
        missing_existing = [str(path) for path in required_existing if not path.exists()]
        if missing_existing:
            raise FileNotFoundError(
                "Append mode requires the existing completed source outputs:\n- "
                + "\n- ".join(missing_existing)
            )
        existing = {
            "results": pd.read_parquet(paths["source_model_results_parquet"]),
            "tuning": pd.read_parquet(paths["source_tuning_parquet"]),
            "run_summary": pd.read_parquet(paths["source_run_summary_parquet"]),
            "source_meta": pd.read_parquet(paths["source_meta_parquet"]),
        }
        existing_report = json.loads(
            paths["source_model_report_json"].read_text(encoding="utf-8")
        )

    temporary_predictions = paths["source_predictions_parquet"].with_suffix(".incomplete.parquet")
    if temporary_predictions.exists():
        temporary_predictions.unlink()
    writer: pq.ParquetWriter | None = None
    result_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    run_summary_frames: list[pd.DataFrame] = []
    source_meta_frames: list[pd.DataFrame] = []
    failure_records: list[dict[str, str]] = []
    skipped_records: list[dict[str, str]] = []
    prediction_rows = 0

    try:
        for task_id, task in tasks.groupby("task_id", sort=True):
            heldout_source = str(_single_value(task, "heldout_source", task_id))
            strategy = str(_single_value(task, "split_strategy", task_id))
            repeat = int(_single_value(task, "repeat", task_id))
            assignments, targets = _task_inputs(task)
            if show_progress:
                print(f"Starting task {task_id} ({len(task):,} molecules)")
            results, predictions, tuning, task_report = run_baseline_experiments(
                targets,
                assignments,
                feature_tables=feature_tables,
                strategies=(strategy,),
                repeats=(repeat,),
                model_names=model_names,
                progress=print if show_progress else None,
            )
            failure_records.extend(task_report["failed_runs"])
            skipped_records.extend(task_report["skipped_runs"])
            if results.empty or predictions.empty or tuning.empty:
                failure_records.append(
                    {"run_id": task_id, "reason": "Task produced an empty result table"}
                )
                continue
            for frame in (results, predictions, tuning):
                frame.insert(1, "task_id", task_id)
                frame.insert(2, "heldout_source", heldout_source)

            labeled = add_failure_labels(predictions)
            arrow_table = pa.Table.from_pandas(labeled, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_predictions,
                    arrow_table.schema,
                    compression="zstd",
                )
            writer.write_table(arrow_table)
            prediction_rows += len(labeled)

            run_summary = build_run_failure_summary(labeled)
            run_summary.insert(1, "task_id", task_id)
            run_summary.insert(2, "heldout_source", heldout_source)
            task_meta = build_meta_model_dataset(
                results,
                run_summary,
                assignments,
                targets,
                descriptors,
            )
            task_meta.insert(2, "heldout_source", heldout_source)
            task_meta["x_split_strategy"] = "source_holdout"

            result_frames.append(results)
            tuning_frames.append(tuning)
            run_summary_frames.append(run_summary)
            source_meta_frames.append(task_meta)
            if show_progress:
                print(f"Completed task {task_id}: {len(results)} model runs")
    finally:
        if writer is not None:
            writer.close()

    if failure_records:
        raise RuntimeError(
            "One or more source-holdout model runs failed:\n- "
            + "\n- ".join(
                f"{item['run_id']}: {item['reason']}" for item in failure_records
            )
        )
    if writer is None or not temporary_predictions.exists():
        raise RuntimeError("Source-holdout modeling produced no prediction file")
    results = pd.concat(result_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    run_summary = pd.concat(run_summary_frames, ignore_index=True)
    source_meta = pd.concat(source_meta_frames, ignore_index=True)
    source_meta["x_evaluation_design"] = "source_holdout"
    if append_existing:
        overlap = sorted(set(existing["results"]["run_id"]).intersection(results["run_id"]))
        if overlap:
            raise ValueError(
                "Append mode found existing source run IDs and refused to overwrite them:\n- "
                + "\n- ".join(overlap[:20])
            )
        results = pd.concat([existing["results"], results], ignore_index=True)
        tuning = pd.concat([existing["tuning"], tuning], ignore_index=True)
        run_summary = pd.concat([existing["run_summary"], run_summary], ignore_index=True)
        source_meta = pd.concat([existing["source_meta"], source_meta], ignore_index=True)

    results = results.sort_values(
        ["task_id", "feature_type", "model"], kind="stable"
    ).reset_index(drop=True)
    tuning = tuning.sort_values(["task_id", "run_id", "candidate"], kind="stable").reset_index(
        drop=True
    )
    run_summary = run_summary.sort_values(["task_id", "run_id"], kind="stable").reset_index(
        drop=True
    )
    source_meta = _rebuild_source_meta(tasks, results, run_summary, descriptors)
    if results["run_id"].duplicated().any():
        raise RuntimeError("Source model results contain duplicate run IDs")
    if tuning.duplicated(["run_id", "candidate"]).any():
        raise RuntimeError("Source tuning results contain duplicate run/candidate rows")
    if run_summary["run_id"].duplicated().any():
        raise RuntimeError("Source run summaries contain duplicate run IDs")
    if source_meta["run_id"].duplicated().any():
        raise RuntimeError("Source meta-data contain duplicate run IDs")

    partitioned_predictions = results_dir / "source_holdout_predictions.incomplete.parquet"
    prediction_sources = [temporary_predictions]
    if append_existing:
        prediction_sources.insert(0, paths["source_predictions_parquet"])
    prediction_rows = _write_partitioned_prediction_dataset(
        prediction_sources,
        partitioned_predictions,
    )
    canonical_predictions = paths["source_predictions_parquet"]
    backup_predictions = results_dir / "source_holdout_predictions.backup.parquet"
    if backup_predictions.exists():
        if backup_predictions.is_dir():
            shutil.rmtree(backup_predictions)
        else:
            backup_predictions.unlink()
    if canonical_predictions.exists():
        canonical_predictions.rename(backup_predictions)
    try:
        partitioned_predictions.rename(canonical_predictions)
    except Exception:
        if backup_predictions.exists():
            backup_predictions.rename(canonical_predictions)
        raise
    if backup_predictions.exists():
        if backup_predictions.is_dir():
            shutil.rmtree(backup_predictions)
        else:
            backup_predictions.unlink()
    temporary_predictions.unlink()

    combined_meta = _combine_meta_datasets(base_meta, source_meta)
    dictionary = build_meta_data_dictionary(combined_meta)
    group_summary = build_failure_group_summary(run_summary).merge(
        run_summary[["split_strategy", "heldout_source"]].drop_duplicates(),
        on="split_strategy",
        how="left",
        validate="many_to_one",
    )
    group_summary = group_summary[
        ["heldout_source", *[column for column in group_summary if column != "heldout_source"]]
    ]

    _write_table(
        results,
        paths["source_model_results_csv"],
        paths["source_model_results_parquet"],
    )
    _write_table(tuning, paths["source_tuning_csv"], paths["source_tuning_parquet"])
    _write_table(
        run_summary,
        paths["source_run_summary_csv"],
        paths["source_run_summary_parquet"],
    )
    group_summary.to_csv(paths["source_group_summary_csv"], index=False)
    _write_table(source_meta, paths["source_meta_csv"], paths["source_meta_parquet"])
    _write_table(combined_meta, paths["combined_meta_csv"], paths["combined_meta_parquet"])
    dictionary.to_csv(paths["combined_dictionary_csv"], index=False)

    parquet_rows = len(pd.read_parquet(paths["source_predictions_parquet"], columns=["run_id"]))
    if parquet_rows != prediction_rows:
        raise RuntimeError(
            "Source prediction Parquet read-back row mismatch: "
            f"expected {prediction_rows:,}, found {parquet_rows:,}"
        )
    report = {
        "status": "pass",
        "heldout_sources": sorted(str(value) for value in tasks["heldout_source"].unique()),
        "task_count": int(tasks["task_id"].nunique()),
        "completed_runs": int(len(results)),
        "prediction_rows": int(prediction_rows),
        "tuning_candidate_rows": int(len(tuning)),
        "source_meta_rows": int(len(source_meta)),
        "combined_meta_rows": int(len(combined_meta)),
        "combined_distinct_tasks": int(combined_meta["task_id"].nunique()),
        "requested_models": sorted(str(value) for value in results["model"].unique()),
        "requested_feature_types": sorted(
            str(value) for value in results["feature_type"].unique()
        ),
        "failed_runs": failure_records,
        "skipped_runs": skipped_records,
        "prediction_storage_policy": (
            "Molecule-level source-holdout predictions and q_i labels are stored as a compressed "
            "Parquet dataset partitioned by held-out source, with bounded part files below 95 MB. "
            "A CSV copy is intentionally omitted because it would exceed GitHub's 100 MB limit."
        ),
        "meta_validation_policy": (
            "Do not randomly split combined meta rows. Hold out complete heldout_source groups "
            "when evaluating generalization across datasets."
        ),
        "remaining_limitation": (
            "The five source datasets overlap in chemical content. Molecule leakage is blocked "
            "within every task, but genuinely external datasets are still needed for the final "
            "publication claim."
        ),
    }
    if append_existing:
        skip_by_run = {
            item["run_id"]: item
            for item in [
                *existing_report.get("skipped_runs", []),
                *skipped_records,
            ]
        }
        report["skipped_runs"] = list(skip_by_run.values())
        report["append_mode"] = True
        report["appended_models"] = list(model_names)
    report["output_files"] = {
        name: {"path": str(path), "bytes": int(_path_size(path))}
        for name, path in paths.items()
        if not name.startswith("source_model_report")
    }
    paths["source_model_report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["source_model_report_csv"], index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core model grid across leakage-safe source-holdout tasks."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=list(CORE_MODELS)
    )
    parser.add_argument(
        "--features", nargs="+", choices=FEATURE_TYPES, default=list(FEATURE_TYPES)
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append disjoint model runs to existing outputs; refuse duplicate run IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stale_marker = root / "reports/downstream_artifacts_stale.json"
    if stale_marker.exists():
        raise RuntimeError(
            "Within-benchmark downstream artifacts are marked stale. Run "
            "aquasol-build-meta-dataset before source-holdout modeling. This only refreshes "
            "failure labels and the meta-dataset; it does not fit a meta-model."
        )
    input_paths = {
        "tasks": root / "data_processed/source_holdout_tasks.parquet",
        "descriptors": root / "data_processed/rdkit_descriptors.parquet",
        "base_meta": root / "data_processed/meta_model_dataset.parquet",
    }
    feature_paths = {
        name: root / "data_processed" / f"{name}.parquet" for name in args.features
    }
    missing = [
        str(path)
        for path in [*input_paths.values(), *feature_paths.values()]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required source-holdout modeling inputs are missing:\n- "
            + "\n- ".join(missing)
        )
    paths = build_source_holdout_model_artifacts(
        tasks=pd.read_parquet(input_paths["tasks"]),
        feature_tables={name: pd.read_parquet(path) for name, path in feature_paths.items()},
        descriptors=pd.read_parquet(input_paths["descriptors"]),
        base_meta=pd.read_parquet(input_paths["base_meta"]),
        data_dir=root / "data_processed",
        results_dir=root / "results",
        reports_dir=root / "reports",
        model_names=tuple(args.models),
        show_progress=True,
        append_existing=args.append,
    )
    print("Source-holdout modeling completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
