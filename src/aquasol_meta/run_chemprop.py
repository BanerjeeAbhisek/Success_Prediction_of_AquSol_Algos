from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import regression_metrics
from .run_baselines import _flatten_report, _write_table

CHEMPROP_MODEL = "chemprop"
CHEMPROP_FEATURE_TYPE = "molecular_graph"
AUDIT_COLUMNS = (
    "molecule_id",
    "scaffold_group",
    "maximum_train_similarity",
    "target_reliability",
    "n_observations",
    "n_source_datasets",
    "log_s_range",
    "log_s_target",
)


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/aquasol-matplotlib")
    return environment


def _run_command(command: list[str], log_path: Path) -> None:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_command_environment(),
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + f"\n\nELAPSED_SECONDS\n{elapsed:.6f}\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"Chemprop command failed with exit code {completed.returncode}. "
            f"See {log_path}."
        )


def _attach_smiles(frame: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    required = {"molecule_id", "partition", "log_s_target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Chemprop task is missing columns: {sorted(missing)}")
    smiles = structures[["molecule_id", "canonical_smiles_parent"]].rename(
        columns={"canonical_smiles_parent": "smiles"}
    )
    merged = frame.merge(smiles, on="molecule_id", how="left", validate="many_to_one")
    if merged["smiles"].isna().any():
        raise ValueError("One or more task molecules have no standardized parent SMILES")
    if merged["molecule_id"].duplicated().any():
        raise ValueError("Chemprop task contains duplicate molecule IDs")
    expected = {"train", "validation", "test"}
    observed = set(merged["partition"].astype(str))
    if observed != expected:
        raise ValueError(f"Expected train/validation/test partitions, found {sorted(observed)}")
    return merged


def _within_tasks(
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    structures: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    target_columns = [
        "molecule_id",
        "log_s_target",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "log_s_range",
    ]
    eligible = targets.loc[targets["include_primary_benchmark"].astype(bool), target_columns]
    tasks: dict[str, pd.DataFrame] = {}
    for (strategy, repeat), split in assignments.groupby(
        ["split_strategy", "repeat"], sort=True
    ):
        task_id = f"{strategy}_r{int(repeat)}"
        task = split.merge(eligible, on="molecule_id", how="inner", validate="one_to_one")
        if len(task) != len(split):
            raise ValueError(f"{task_id} lost molecules while joining targets")
        tasks[task_id] = _attach_smiles(task, structures)
    return tasks


def _source_tasks(
    source_task_rows: pd.DataFrame,
    structures: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        str(task_id): _attach_smiles(task.copy(), structures)
        for task_id, task in source_task_rows.groupby("task_id", sort=True)
    }


def _prediction_values(path: Path, expected_rows: int) -> np.ndarray:
    predictions = pd.read_csv(path)
    if "log_s_target" not in predictions:
        raise ValueError(f"Chemprop prediction file has no log_s_target column: {path}")
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Chemprop prediction row mismatch: expected {expected_rows}, found {len(predictions)}"
        )
    values = predictions["log_s_target"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Chemprop produced non-finite predictions")
    return values


def _build_task_outputs(
    task_id: str,
    task: pd.DataFrame,
    validation_prediction: np.ndarray,
    test_prediction: np.ndarray,
    training_seconds: float,
    validation_predict_seconds: float,
    test_predict_seconds: float,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = task["partition"].eq("train")
    validation = task["partition"].eq("validation")
    test = task["partition"].eq("test")
    strategy = str(task["split_strategy"].iloc[0])
    repeat = int(task["repeat"].iloc[0])
    seed = int(task["seed"].iloc[0])
    run_id = f"{strategy}_r{repeat}_{CHEMPROP_FEATURE_TYPE}_{CHEMPROP_MODEL}"
    y_validation = task.loc[validation, "log_s_target"].to_numpy(dtype=float)
    y_test = task.loc[test, "log_s_target"].to_numpy(dtype=float)
    validation_metrics = regression_metrics(y_validation, validation_prediction)
    test_metrics = regression_metrics(y_test, test_prediction)
    parameters = {
        "architecture": "D-MPNN",
        "batch_size": batch_size,
        "depth": 3,
        "dropout": 0.0,
        "early_stopping_patience": patience,
        "ensemble_size": 1,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "maximum_epochs": epochs,
        "message_hidden_dim": 300,
    }
    result = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": CHEMPROP_MODEL,
                "feature_type": CHEMPROP_FEATURE_TYPE,
                "split_strategy": strategy,
                "repeat": repeat,
                "seed": seed,
                "n_train": int(train.sum()),
                "n_validation": int(validation.sum()),
                "n_final_fit": int(train.sum()),
                "n_test": int(test.sum()),
                "features_before_preprocessing": np.nan,
                "features_after_preprocessing": 300,
                "best_parameters": json.dumps(parameters, sort_keys=True),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "test_mean_predicted_std": np.nan,
                "test_gaussian_nll": np.nan,
                "test_interval_95_coverage": np.nan,
                "tuning_seconds": training_seconds,
                "final_fit_seconds": training_seconds,
                "test_predict_seconds": test_predict_seconds,
            }
        ]
    )
    available_audit = [column for column in AUDIT_COLUMNS if column in task.columns]
    predictions = (
        task.loc[test, available_audit]
        .copy()
        .rename(columns={"log_s_target": "true_log_s"})
        .reset_index(drop=True)
    )
    predictions.insert(0, "run_id", run_id)
    predictions.insert(1, "model", CHEMPROP_MODEL)
    predictions.insert(2, "feature_type", CHEMPROP_FEATURE_TYPE)
    predictions.insert(3, "split_strategy", strategy)
    predictions.insert(4, "repeat", repeat)
    predictions.insert(5, "seed", seed)
    predictions["predicted_log_s"] = test_prediction
    predictions["predicted_log_s_std"] = np.nan
    predictions["predicted_log_s_lower_95"] = np.nan
    predictions["predicted_log_s_upper_95"] = np.nan
    predictions["residual"] = predictions["true_log_s"] - predictions["predicted_log_s"]
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["squared_error"] = predictions["residual"].pow(2)
    tuning = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "candidate": 1,
                "parameters": json.dumps(parameters, sort_keys=True),
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "fit_and_predict_seconds": training_seconds + validation_predict_seconds,
            }
        ]
    )
    if "heldout_source" in task:
        heldout_source = str(task["heldout_source"].iloc[0])
        for frame in (result, predictions, tuning):
            frame.insert(1, "task_id", task_id)
            frame.insert(2, "heldout_source", heldout_source)
    return result, predictions, tuning


def _run_task(
    executable: Path,
    task_id: str,
    task: pd.DataFrame,
    cache_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"aquasol-{task_id}-") as temporary:
        work = Path(temporary)
        paths: dict[str, Path] = {}
        for partition, stem in (("train", "train"), ("validation", "validation"), ("test", "test")):
            path = work / f"{stem}.csv"
            task.loc[task["partition"].eq(partition), ["smiles", "log_s_target"]].to_csv(
                path, index=False
            )
            paths[stem] = path
        output = work / "training"
        seed = int(task["seed"].iloc[0])
        warmup_epochs = min(2, max(0, epochs - 1))
        train_command = [
            str(executable),
            "train",
            "-i",
            str(paths["train"]),
            str(paths["validation"]),
            str(paths["test"]),
            "-o",
            str(output),
            "-s",
            "smiles",
            "--target-columns",
            "log_s_target",
            "--task-type",
            "regression",
            "--metrics",
            "rmse",
            "mae",
            "r2",
            "--epochs",
            str(epochs),
            "--warmup-epochs",
            str(warmup_epochs),
            "--patience",
            str(patience),
            "--accelerator",
            "cpu",
            "--num-workers",
            "0",
            "--batch-size",
            str(batch_size),
            "--pytorch-seed",
            str(seed),
            "--remove-checkpoints",
            "-q",
        ]
        started = time.perf_counter()
        _run_command(train_command, cache_dir / "train.log")
        training_seconds = time.perf_counter() - started
        model_path = output / "model_0" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Chemprop did not save its selected model: {model_path}")

        predicted: dict[str, np.ndarray] = {}
        prediction_seconds: dict[str, float] = {}
        for partition in ("validation", "test"):
            input_path = work / f"predict_{partition}.csv"
            prediction_path = work / f"predicted_{partition}.csv"
            selected = task.loc[
                task["partition"].eq(partition), ["smiles", "molecule_id"]
            ]
            selected.to_csv(input_path, index=False)
            command = [
                str(executable),
                "predict",
                "-i",
                str(input_path),
                "-o",
                str(prediction_path),
                "-s",
                "smiles",
                "--model-paths",
                str(model_path),
                "--accelerator",
                "cpu",
                "--num-workers",
                "0",
                "--batch-size",
                str(batch_size),
                "-q",
            ]
            started = time.perf_counter()
            _run_command(command, cache_dir / f"predict_{partition}.log")
            prediction_seconds[partition] = time.perf_counter() - started
            predicted[partition] = _prediction_values(prediction_path, len(selected))

        result, predictions, tuning = _build_task_outputs(
            task_id,
            task,
            predicted["validation"],
            predicted["test"],
            training_seconds,
            prediction_seconds["validation"],
            prediction_seconds["test"],
            epochs,
            patience,
            batch_size,
        )
        result.to_parquet(cache_dir / "model_result.parquet", index=False, compression="zstd")
        predictions.to_parquet(
            cache_dir / "test_predictions.parquet", index=False, compression="zstd"
        )
        tuning.to_parquet(cache_dir / "hyperparameter_result.parquet", index=False)
        (cache_dir / "complete.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "run_id": str(result["run_id"].iloc[0]),
                    "test_rmse": float(result["test_rmse"].iloc[0]),
                    "configuration": {
                        "epochs": epochs,
                        "patience": patience,
                        "batch_size": batch_size,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _collect(
    cache_root: Path,
    task_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for task_id in task_ids:
        task_dir = cache_root / task_id
        if not (task_dir / "complete.json").exists():
            raise RuntimeError(f"Chemprop task is incomplete: {task_id}")
        result_frames.append(pd.read_parquet(task_dir / "model_result.parquet"))
        prediction_frames.append(pd.read_parquet(task_dir / "test_predictions.parquet"))
        tuning_frames.append(pd.read_parquet(task_dir / "hyperparameter_result.parquet"))
    results = (
        pd.concat(result_frames, ignore_index=True)
        .sort_values("run_id")
        .reset_index(drop=True)
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["run_id", "molecule_id"]
    ).reset_index(drop=True)
    tuning = (
        pd.concat(tuning_frames, ignore_index=True)
        .sort_values("run_id")
        .reset_index(drop=True)
    )
    if results["run_id"].duplicated().any():
        raise RuntimeError("Collected Chemprop results contain duplicate run IDs")
    if predictions.duplicated(["run_id", "molecule_id"]).any():
        raise RuntimeError("Collected Chemprop predictions contain duplicate run/molecule rows")
    return results, predictions, tuning


def _write_outputs(
    design: str,
    results_dir: Path,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    complete_design: bool,
) -> dict[str, Path]:
    prefix = "chemprop_within" if design == "within" else "chemprop_source_holdout"
    if not complete_design:
        prefix += "_partial"
    paths = {
        "results_csv": results_dir / f"{prefix}_model_results.csv",
        "results_parquet": results_dir / f"{prefix}_model_results.parquet",
        "predictions_parquet": results_dir / f"{prefix}_test_predictions.parquet",
        "tuning_csv": results_dir / f"{prefix}_hyperparameter_results.csv",
        "tuning_parquet": results_dir / f"{prefix}_hyperparameter_results.parquet",
        "report": results_dir.parent / "reports" / f"{prefix}_run_report.json",
    }
    _write_table(results, paths["results_csv"], paths["results_parquet"])
    predictions.to_parquet(paths["predictions_parquet"], index=False, compression="zstd")
    _write_table(tuning, paths["tuning_csv"], paths["tuning_parquet"])
    report = {
        "status": "pass",
        "design": design,
        "model": CHEMPROP_MODEL,
        "feature_type": CHEMPROP_FEATURE_TYPE,
        "completed_runs": int(len(results)),
        "complete_design": complete_design,
        "prediction_rows": int(len(predictions)),
        "mean_test_rmse": float(results["test_rmse"].mean()),
        "training_policy": (
            "One fixed Chemprop v2 D-MPNN architecture is trained on each fixed training set. "
            "The validation set controls early stopping; the test set is evaluated once."
        ),
        "final_fit_policy": (
            "Unlike the classical models, Chemprop is not refit on training plus validation "
            "because validation is required for epoch selection and early stopping."
        ),
        "hardware_policy": "CPU execution with one model per task and no ensemble.",
        "output_files": {
            name: {"path": str(path), "bytes": int(path.stat().st_size)}
            for name, path in paths.items()
            if name != "report"
        },
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _append_within_to_canonical(
    root: Path,
    chemprop_results: pd.DataFrame,
    chemprop_predictions: pd.DataFrame,
    chemprop_tuning: pd.DataFrame,
) -> bool:
    """Append a complete Chemprop grid to canonical tables exactly once."""
    results_dir = root / "results"
    reports_dir = root / "reports"
    paths = {
        "results_csv": results_dir / "model_results.csv",
        "results_parquet": results_dir / "model_results.parquet",
        "predictions_csv_gz": results_dir / "test_predictions.csv.gz",
        "predictions_parquet": results_dir / "test_predictions.parquet",
        "tuning_csv": results_dir / "hyperparameter_results.csv",
        "tuning_parquet": results_dir / "hyperparameter_results.parquet",
        "report_json": reports_dir / "baseline_run_report.json",
        "report_csv": reports_dir / "baseline_run_report.csv",
    }
    required = [
        paths["results_parquet"],
        paths["predictions_parquet"],
        paths["tuning_parquet"],
        paths["report_json"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot integrate Chemprop because canonical files are missing:\n- "
            + "\n- ".join(missing)
        )

    existing_results = pd.read_parquet(paths["results_parquet"])
    overlap = set(existing_results["run_id"]).intersection(chemprop_results["run_id"])
    if overlap:
        if overlap == set(chemprop_results["run_id"]):
            return False
        raise ValueError(
            "Canonical results contain only part of the Chemprop grid; refusing a partial merge"
        )
    existing_predictions = pd.read_parquet(paths["predictions_parquet"])
    existing_tuning = pd.read_parquet(paths["tuning_parquet"])
    results = pd.concat([existing_results, chemprop_results], ignore_index=True).sort_values(
        ["split_strategy", "repeat", "feature_type", "model"], kind="stable"
    )
    predictions = pd.concat(
        [existing_predictions, chemprop_predictions], ignore_index=True
    ).sort_values(["run_id", "molecule_id"], kind="stable")
    tuning = pd.concat([existing_tuning, chemprop_tuning], ignore_index=True).sort_values(
        ["run_id", "candidate"], kind="stable"
    )
    if results["run_id"].duplicated().any():
        raise RuntimeError("Integrated model results contain duplicate run IDs")
    if predictions.duplicated(["run_id", "molecule_id"]).any():
        raise RuntimeError("Integrated predictions contain duplicate run/molecule rows")
    if tuning.duplicated(["run_id", "candidate"]).any():
        raise RuntimeError("Integrated tuning results contain duplicate run/candidate rows")

    with tempfile.TemporaryDirectory(prefix="aquasol-chemprop-integration-") as temporary:
        staging = Path(temporary)
        _write_table(
            results,
            staging / paths["results_csv"].name,
            staging / paths["results_parquet"].name,
        )
        _write_table(
            predictions,
            staging / paths["predictions_csv_gz"].name,
            staging / paths["predictions_parquet"].name,
        )
        _write_table(
            tuning,
            staging / paths["tuning_csv"].name,
            staging / paths["tuning_parquet"].name,
        )
        for key in (
            "results_csv",
            "results_parquet",
            "predictions_csv_gz",
            "predictions_parquet",
            "tuning_csv",
            "tuning_parquet",
        ):
            (staging / paths[key].name).replace(paths[key])

    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    appended_models = set(str(value) for value in report.get("appended_models", []))
    appended_models.add(CHEMPROP_MODEL)
    report.update(
        {
            "status": "pass",
            "append_mode": True,
            "appended_models": sorted(appended_models),
            "requested_models": sorted(str(value) for value in results["model"].unique()),
            "requested_feature_types": sorted(
                str(value) for value in results["feature_type"].unique()
            ),
            "completed_runs": int(len(results)),
            "test_prediction_rows": int(len(predictions)),
            "tuning_candidate_rows": int(len(tuning)),
            "chemprop_integration": (
                "Complete fixed-task Chemprop grid merged from the separately resumable runner."
            ),
        }
    )
    report["output_files"] = {
        key: {"path": str(path), "bytes": int(path.stat().st_size)}
        for key, path in paths.items()
        if key not in {"report_json", "report_csv"}
    }
    paths["report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["report_csv"], index=False)
    stale_marker = reports_dir / "downstream_artifacts_stale.json"
    stale_marker.write_text(
        json.dumps(
            {
                "status": "stale",
                "reason": "Chemprop was integrated after downstream tables were last built.",
                "current_baseline_runs": int(len(results)),
                "refresh_command": "aquasol-build-meta-dataset",
                "note": "This refresh does not fit the meta-model.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Chemprop D-MPNN benchmark on the fixed chemical tasks."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--design", choices=("within", "source"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--append-canonical",
        action="store_true",
        help="After a complete within run, merge Chemprop into canonical baseline tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    structures_path = root / "data_processed/rdkit_descriptors.parquet"
    executable = Path(sys.executable).with_name("chemprop")
    if not executable.exists():
        raise FileNotFoundError(
            f"Chemprop executable not found beside the active Python interpreter: {executable}"
        )
    structures = pd.read_parquet(
        structures_path, columns=["molecule_id", "canonical_smiles_parent"]
    )
    if args.design == "within":
        tasks = _within_tasks(
            pd.read_parquet(root / "data_processed/modeling_targets.parquet"),
            pd.read_parquet(root / "data_processed/split_assignments.parquet"),
            structures,
        )
    else:
        tasks = _source_tasks(
            pd.read_parquet(root / "data_processed/source_holdout_tasks.parquet"),
            structures,
        )
    all_task_ids = sorted(tasks)
    selected_ids = list(all_task_ids)
    if args.task_ids:
        unknown = sorted(set(args.task_ids).difference(tasks))
        if unknown:
            raise ValueError(f"Unknown task IDs: {unknown}")
        selected_ids = list(args.task_ids)
    if args.task_limit is not None:
        if args.task_limit < 1:
            raise ValueError("--task-limit must be positive")
        selected_ids = selected_ids[: args.task_limit]

    cache_root = root / "results" / "chemprop_cache" / args.design
    cache_root.mkdir(parents=True, exist_ok=True)
    requested_configuration = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    for index, task_id in enumerate(selected_ids, start=1):
        task_dir = cache_root / task_id
        if args.force and task_dir.exists():
            shutil.rmtree(task_dir)
        completion_path = task_dir / "complete.json"
        if completion_path.exists():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("configuration") == requested_configuration:
                print(f"[{index}/{len(selected_ids)}] Reusing completed {task_id}")
                continue
            print(f"[{index}/{len(selected_ids)}] Configuration changed; rerunning {task_id}")
            shutil.rmtree(task_dir)
        print(f"[{index}/{len(selected_ids)}] Starting {task_id}")
        _run_task(
            executable,
            task_id,
            tasks[task_id],
            task_dir,
            args.epochs,
            args.patience,
            args.batch_size,
        )
        print(f"[{index}/{len(selected_ids)}] Completed {task_id}")

    results, predictions, tuning = _collect(cache_root, selected_ids)
    paths = _write_outputs(
        args.design,
        root / "results",
        results,
        predictions,
        tuning,
        complete_design=set(selected_ids) == set(all_task_ids),
    )
    if args.append_canonical:
        if args.design != "within":
            raise ValueError("--append-canonical currently supports only --design within")
        if set(selected_ids) != set(all_task_ids):
            raise ValueError("--append-canonical requires the complete within-task design")
        appended = _append_within_to_canonical(root, results, predictions, tuning)
        print(
            "Chemprop canonical integration completed."
            if appended
            else "Chemprop was already present in the canonical tables; no rows were changed."
        )
    print("Chemprop modeling completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
