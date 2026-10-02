from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURE_TYPES = ("rdkit_descriptors", "morgan_fingerprints", "maccs_fingerprints")
CORE_MODELS = ("dummy", "ridge", "random_forest", "extra_trees")
ALL_MODELS = CORE_MODELS + ("hist_gradient_boosting", "svr", "mlp")
EXCLUDED_RDKIT_DESCRIPTORS = ("Ipc",)


def _candidate_parameters(model_name: str) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "dummy": [{"strategy": "mean"}, {"strategy": "median"}],
        "ridge": [{"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)],
        "random_forest": [
            {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": 100, "max_features": "log2", "min_samples_leaf": 1},
            {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 3},
        ],
        "extra_trees": [
            {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1},
            {"n_estimators": 100, "max_features": "log2", "min_samples_leaf": 1},
            {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 3},
        ],
        "hist_gradient_boosting": [
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
            {"learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        ],
        "svr": [
            {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
            {"C": 100.0, "epsilon": 0.1, "gamma": "scale"},
            {"C": 10.0, "epsilon": 0.2, "gamma": "scale"},
        ],
        "mlp": [
            {"hidden_layer_sizes": (128,), "alpha": 0.0001},
            {"hidden_layer_sizes": (128,), "alpha": 0.001},
            {"hidden_layer_sizes": (128, 64), "alpha": 0.001},
        ],
    }
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")
    return candidates[model_name]


def _is_compatible(model_name: str, feature_type: str) -> bool:
    return model_name != "hist_gradient_boosting" or feature_type == "rdkit_descriptors"


def _estimator(model_name: str, parameters: dict[str, Any], seed: int) -> RegressorMixin:
    if model_name == "dummy":
        return DummyRegressor(**parameters)
    if model_name == "ridge":
        return Ridge(solver="lsqr", **parameters)
    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            bootstrap=True,
            **parameters,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            random_state=seed,
            n_jobs=-1,
            bootstrap=False,
            **parameters,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=300,
            early_stopping=True,
            **parameters,
        )
    if model_name == "svr":
        return SVR(kernel="rbf", cache_size=2048, **parameters)
    if model_name == "mlp":
        return MLPRegressor(
            random_state=seed,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            **parameters,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _pipeline(
    model_name: str,
    feature_type: str,
    parameters: dict[str, Any],
    seed: int,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_type == "rdkit_descriptors" and model_name != "dummy":
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("variance", VarianceThreshold(threshold=0.0)),
            ]
        )
        if model_name in {"ridge", "svr", "mlp"}:
            steps.append(("scaler", StandardScaler()))
    steps.append(("model", _estimator(model_name, parameters, seed)))
    return Pipeline(steps)


def regression_metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2 or np.unique(y_predicted).size < 2:
        statistic = np.nan
    else:
        statistic = spearmanr(y_true, y_predicted).statistic
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_predicted))),
        "mae": float(mean_absolute_error(y_true, y_predicted)),
        "r2": float(r2_score(y_true, y_predicted)),
        "spearman": float(statistic) if np.isfinite(statistic) else np.nan,
    }


def _feature_columns(feature_type: str, frame: pd.DataFrame) -> list[str]:
    excluded = {"molecule_id", "canonical_smiles_parent"}
    if feature_type == "rdkit_descriptors":
        excluded.update(EXCLUDED_RDKIT_DESCRIPTORS)
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise ValueError(f"{feature_type} contains no predictor columns")
    return columns


def _experiment_frame(
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    feature_type: str,
    split_strategy: str,
    repeat: int,
) -> tuple[pd.DataFrame, list[str]]:
    split = assignments.loc[
        assignments["split_strategy"].eq(split_strategy)
        & assignments["repeat"].eq(repeat)
    ].copy()
    if split.empty:
        raise ValueError(f"No assignments for {split_strategy} repeat {repeat}")
    if split["molecule_id"].duplicated().any():
        raise ValueError(f"Duplicate split rows for {split_strategy} repeat {repeat}")

    audit_columns = [
        "molecule_id",
        "log_s_target",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "log_s_range",
    ]
    available_audit = [column for column in audit_columns if column in targets.columns]
    target_data = targets.loc[targets["include_primary_benchmark"].astype(bool), available_audit]
    frame = split.merge(target_data, on="molecule_id", how="inner", validate="one_to_one")
    frame = frame.merge(features, on="molecule_id", how="inner", validate="one_to_one")
    if len(frame) != len(split):
        raise ValueError(
            f"{feature_type} join produced {len(frame):,} rows; expected {len(split):,}"
        )
    columns = _feature_columns(feature_type, features)
    if frame[columns].select_dtypes(exclude=[np.number]).shape[1]:
        raise ValueError(f"{feature_type} includes non-numeric predictor columns")
    return frame, columns


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _retained_feature_count(pipeline: Pipeline, original_count: int) -> int:
    if "variance" not in pipeline.named_steps:
        return original_count
    return int(pipeline.named_steps["variance"].get_support().sum())


def run_baseline_experiments(
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    strategies: Iterable[str],
    repeats: Iterable[int],
    model_names: Iterable[str],
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    strategies = tuple(strategies)
    repeats = tuple(int(value) for value in repeats)
    model_names = tuple(model_names)
    unknown_features = sorted(set(feature_tables).difference(FEATURE_TYPES))
    if unknown_features:
        raise ValueError(f"Unknown feature tables: {unknown_features}")
    unknown_models = sorted(set(model_names).difference(ALL_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    for feature_type, features in feature_tables.items():
        for split_strategy in strategies:
            for repeat in repeats:
                frame, columns = _experiment_frame(
                    targets,
                    assignments,
                    features,
                    feature_type,
                    split_strategy,
                    repeat,
                )
                seed = int(frame["seed"].iloc[0])
                train = frame["partition"].eq("train")
                validation = frame["partition"].eq("validation")
                test = frame["partition"].eq("test")
                X_train = frame.loc[train, columns]
                y_train = frame.loc[train, "log_s_target"].to_numpy(dtype=float)
                X_validation = frame.loc[validation, columns]
                y_validation = frame.loc[validation, "log_s_target"].to_numpy(dtype=float)
                X_test = frame.loc[test, columns]
                y_test = frame.loc[test, "log_s_target"].to_numpy(dtype=float)

                for model_name in model_names:
                    run_id = f"{split_strategy}_r{repeat}_{feature_type}_{model_name}"
                    if not _is_compatible(model_name, feature_type):
                        skipped.append(
                            {
                                "run_id": run_id,
                                "reason": (
                                    "HistGradientBoosting is restricted to compact RDKit "
                                    "descriptors rather than high-dimensional binary fingerprints."
                                ),
                            }
                        )
                        continue

                    if progress:
                        progress(f"Starting {run_id}")
                    try:
                        candidates = _candidate_parameters(model_name)
                        best: tuple[float, float, dict[str, Any], dict[str, float]] | None = None
                        tuning_start = time.perf_counter()
                        for candidate_number, parameters in enumerate(candidates, start=1):
                            candidate_start = time.perf_counter()
                            pipeline = _pipeline(model_name, feature_type, parameters, seed)
                            pipeline.fit(X_train, y_train)
                            validation_prediction = pipeline.predict(X_validation)
                            candidate_seconds = time.perf_counter() - candidate_start
                            metrics = regression_metrics(y_validation, validation_prediction)
                            tuning_rows.append(
                                {
                                    "run_id": run_id,
                                    "candidate": candidate_number,
                                    "parameters": json.dumps(parameters, sort_keys=True),
                                    **_prefix_metrics("validation", metrics),
                                    "fit_and_predict_seconds": candidate_seconds,
                                }
                            )
                            rank = (metrics["rmse"], metrics["mae"])
                            if best is None or rank < best[:2]:
                                best = (rank[0], rank[1], parameters, metrics)

                        assert best is not None
                        tuning_seconds = time.perf_counter() - tuning_start
                        best_parameters = best[2]
                        best_validation_metrics = best[3]
                        final_pipeline = _pipeline(
                            model_name, feature_type, best_parameters, seed
                        )
                        X_final = pd.concat([X_train, X_validation], axis=0)
                        y_final = np.concatenate([y_train, y_validation])
                        final_fit_start = time.perf_counter()
                        final_pipeline.fit(X_final, y_final)
                        final_fit_seconds = time.perf_counter() - final_fit_start
                        prediction_start = time.perf_counter()
                        test_prediction = final_pipeline.predict(X_test)
                        prediction_seconds = time.perf_counter() - prediction_start
                        test_metrics = regression_metrics(y_test, test_prediction)

                        result_rows.append(
                            {
                                "run_id": run_id,
                                "model": model_name,
                                "feature_type": feature_type,
                                "split_strategy": split_strategy,
                                "repeat": repeat,
                                "seed": seed,
                                "n_train": int(train.sum()),
                                "n_validation": int(validation.sum()),
                                "n_final_fit": int(train.sum() + validation.sum()),
                                "n_test": int(test.sum()),
                                "features_before_preprocessing": len(columns),
                                "features_after_preprocessing": _retained_feature_count(
                                    final_pipeline, len(columns)
                                ),
                                "best_parameters": json.dumps(
                                    best_parameters, sort_keys=True
                                ),
                                **_prefix_metrics(
                                    "validation", best_validation_metrics
                                ),
                                **_prefix_metrics("test", test_metrics),
                                "tuning_seconds": tuning_seconds,
                                "final_fit_seconds": final_fit_seconds,
                                "test_predict_seconds": prediction_seconds,
                            }
                        )

                        prediction_columns = [
                            "molecule_id",
                            "scaffold_group",
                            "maximum_train_similarity",
                            "target_reliability",
                            "n_observations",
                            "n_source_datasets",
                            "log_s_range",
                            "log_s_target",
                        ]
                        available = [
                            column for column in prediction_columns if column in frame.columns
                        ]
                        predictions = frame.loc[test, available].copy()
                        predictions = predictions.rename(
                            columns={"log_s_target": "true_log_s"}
                        )
                        predictions.insert(0, "run_id", run_id)
                        predictions.insert(1, "model", model_name)
                        predictions.insert(2, "feature_type", feature_type)
                        predictions.insert(3, "split_strategy", split_strategy)
                        predictions.insert(4, "repeat", repeat)
                        predictions.insert(5, "seed", seed)
                        predictions["predicted_log_s"] = test_prediction
                        predictions["residual"] = (
                            predictions["true_log_s"] - predictions["predicted_log_s"]
                        )
                        predictions["absolute_error"] = predictions["residual"].abs()
                        predictions["squared_error"] = predictions["residual"].pow(2)
                        prediction_frames.append(predictions)
                        if progress:
                            progress(
                                f"Completed {run_id}: test RMSE={test_metrics['rmse']:.3f}"
                            )
                    except Exception as exc:  # save other successful runs for diagnosis
                        failures.append(
                            {
                                "run_id": run_id,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if progress:
                            progress(f"Failed {run_id}: {type(exc).__name__}: {exc}")

    results = pd.DataFrame(result_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    tuning = pd.DataFrame(tuning_rows)
    report = {
        "status": "fail" if failures else "pass",
        "scikit_learn_version": sklearn.__version__,
        "requested_feature_types": list(feature_tables),
        "requested_strategies": list(strategies),
        "requested_repeats": list(repeats),
        "requested_models": list(model_names),
        "completed_runs": int(len(results)),
        "test_prediction_rows": int(len(predictions)),
        "tuning_candidate_rows": int(len(tuning)),
        "skipped_runs": skipped,
        "failed_runs": failures,
        "selection_policy": "Minimum validation RMSE, with validation MAE as the tie-breaker.",
        "final_fit_policy": (
            "After selection, preprocessing and the model are refit on training plus validation; "
            "the test partition is evaluated once."
        ),
        "residual_definition": "true_log_s - predicted_log_s",
        "failure_label_policy": (
            "No binary success/failure labels are created at this stage; molecule-level residuals "
            "are retained for evidence-based failure definitions."
        ),
    }
    return results, predictions, tuning, report
