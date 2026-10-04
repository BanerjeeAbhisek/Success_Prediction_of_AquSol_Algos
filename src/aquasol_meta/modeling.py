from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from ngboost import NGBRegressor
from ngboost.distns import Normal
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

FEATURE_TYPES = ("rdkit_descriptors", "morgan_fingerprints", "maccs_fingerprints")
CORE_MODELS = ("dummy", "ridge", "random_forest", "extra_trees")
ADVANCED_MODELS = ("hist_gradient_boosting", "svr", "mlp")
ADDITIONAL_MODELS = ("elastic_net", "knn", "xgboost", "ngboost", "esol")
ALL_MODELS = CORE_MODELS + ADVANCED_MODELS + ADDITIONAL_MODELS
EXCLUDED_RDKIT_DESCRIPTORS = ("Ipc",)
ESOL_FEATURE_COLUMNS = (
    "MolLogP",
    "MolWt",
    "NumRotatableBonds",
    "_esol_aromatic_proportion",
)


class ESOLRegressor(RegressorMixin, BaseEstimator):
    """Fixed Delaney ESOL equation used as an interpretable chemistry baseline."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ESOLRegressor:
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(ESOL_FEATURE_COLUMNS):
            raise ValueError(
                "ESOL requires logP, molecular weight, rotors, and aromatic proportion"
            )
        log_p, mol_wt, rotors, aromatic_proportion = values.T
        return (
            0.16
            - 1.5 * log_p
            - 0.01 * mol_wt
            + 0.5 * rotors
            - 1.5 * aromatic_proportion
        )


class NGBoostRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible adapter retaining NGBoost distributions."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        minibatch_frac: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> NGBoostRegressor:
        self.model_ = NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            random_state=self.random_state,
            verbose=False,
        )
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X), dtype=float)

    def pred_dist(self, X: pd.DataFrame) -> Any:
        check_is_fitted(self, "model_")
        return self.model_.pred_dist(X)


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
        "elastic_net": [
            {"alpha": 0.01, "l1_ratio": 0.2},
            {"alpha": 0.1, "l1_ratio": 0.5},
            {"alpha": 1.0, "l1_ratio": 0.8},
        ],
        "knn": [
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 11, "weights": "distance"},
            {"n_neighbors": 25, "weights": "distance"},
        ],
        "xgboost": [
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 1,
            },
            {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 1,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 3,
            },
        ],
        "ngboost": [
            {"n_estimators": 300, "learning_rate": 0.03, "minibatch_frac": 1.0},
            {"n_estimators": 500, "learning_rate": 0.03, "minibatch_frac": 0.8},
            {"n_estimators": 300, "learning_rate": 0.05, "minibatch_frac": 0.8},
        ],
        "esol": [{}],
    }
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")
    return candidates[model_name]


def _is_compatible(model_name: str, feature_type: str) -> bool:
    descriptor_only = {"hist_gradient_boosting", "ngboost", "esol"}
    return model_name not in descriptor_only or feature_type == "rdkit_descriptors"


def _estimator(
    model_name: str,
    feature_type: str,
    parameters: dict[str, Any],
    seed: int,
) -> RegressorMixin:
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
    if model_name == "elastic_net":
        return ElasticNet(random_state=seed, max_iter=20_000, **parameters)
    if model_name == "knn":
        metric = "minkowski" if feature_type == "rdkit_descriptors" else "cosine"
        return KNeighborsRegressor(metric=metric, algorithm="brute", n_jobs=-1, **parameters)
    if model_name == "xgboost":
        return xgboost.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            **parameters,
        )
    if model_name == "ngboost":
        return NGBoostRegressor(random_state=seed, **parameters)
    if model_name == "esol":
        return ESOLRegressor()
    raise ValueError(f"Unknown model: {model_name}")


def _pipeline(
    model_name: str,
    feature_type: str,
    parameters: dict[str, Any],
    seed: int,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_type == "rdkit_descriptors" and model_name not in {"dummy", "esol"}:
        steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("variance", VarianceThreshold(threshold=0.0)),
            ]
        )
        if model_name in {"ridge", "svr", "mlp", "elastic_net", "knn"}:
            steps.append(("scaler", StandardScaler()))
    elif model_name == "elastic_net":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _estimator(model_name, feature_type, parameters, seed)))
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
    excluded = {"molecule_id", "canonical_smiles_parent", "_esol_aromatic_proportion"}
    if feature_type == "rdkit_descriptors":
        excluded.update(EXCLUDED_RDKIT_DESCRIPTORS)
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise ValueError(f"{feature_type} contains no predictor columns")
    return columns


def _add_esol_feature(features: pd.DataFrame) -> pd.DataFrame:
    if "canonical_smiles_parent" not in features:
        raise ValueError("ESOL requires canonical_smiles_parent in the RDKit feature table")
    enriched = features.copy()

    def aromatic_proportion(smiles: str) -> float:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"Could not parse standardized SMILES for ESOL: {smiles}")
        heavy_atoms = molecule.GetNumHeavyAtoms()
        if heavy_atoms == 0:
            return 0.0
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        return float(aromatic_atoms / heavy_atoms)

    enriched["_esol_aromatic_proportion"] = enriched[
        "canonical_smiles_parent"
    ].map(aromatic_proportion)
    return enriched


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


def _predictive_standard_deviation(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray | None:
    model = pipeline.named_steps["model"]
    if not isinstance(model, NGBoostRegressor):
        return None
    transformed: Any = X
    for name, transformer in pipeline.steps[:-1]:
        if name != "model":
            transformed = transformer.transform(transformed)
    distribution = model.pred_dist(transformed)
    return np.asarray(distribution.scale, dtype=float)


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
    feature_tables = dict(feature_tables)
    if "esol" in model_names and "rdkit_descriptors" in feature_tables:
        feature_tables["rdkit_descriptors"] = _add_esol_feature(
            feature_tables["rdkit_descriptors"]
        )
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
                y_train = frame.loc[train, "log_s_target"].to_numpy(dtype=float)
                y_validation = frame.loc[validation, "log_s_target"].to_numpy(dtype=float)
                y_test = frame.loc[test, "log_s_target"].to_numpy(dtype=float)

                for model_name in model_names:
                    run_id = f"{split_strategy}_r{repeat}_{feature_type}_{model_name}"
                    if not _is_compatible(model_name, feature_type):
                        skipped.append(
                            {
                                "run_id": run_id,
                                "reason": (
                                    f"{model_name} is restricted to compact RDKit descriptors "
                                    "rather than high-dimensional binary fingerprints."
                                ),
                            }
                        )
                        continue

                    model_columns = (
                        list(ESOL_FEATURE_COLUMNS) if model_name == "esol" else columns
                    )
                    X_train = frame.loc[train, model_columns]
                    X_validation = frame.loc[validation, model_columns]
                    X_test = frame.loc[test, model_columns]

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
                        test_prediction_std = _predictive_standard_deviation(
                            final_pipeline, X_test
                        )
                        prediction_seconds = time.perf_counter() - prediction_start
                        test_metrics = regression_metrics(y_test, test_prediction)
                        if test_prediction_std is None:
                            mean_prediction_std = np.nan
                            gaussian_nll = np.nan
                            interval_95_coverage = np.nan
                        else:
                            safe_std = np.maximum(test_prediction_std, 1.0e-12)
                            standardized_error = (y_test - test_prediction) / safe_std
                            gaussian_nll = float(
                                np.mean(
                                    np.log(safe_std)
                                    + 0.5 * np.log(2.0 * np.pi)
                                    + 0.5 * standardized_error**2
                                )
                            )
                            mean_prediction_std = float(np.mean(safe_std))
                            interval_95_coverage = float(
                                np.mean(np.abs(y_test - test_prediction) <= 1.96 * safe_std)
                            )

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
                                "features_before_preprocessing": len(model_columns),
                                "features_after_preprocessing": _retained_feature_count(
                                    final_pipeline, len(model_columns)
                                ),
                                "best_parameters": json.dumps(
                                    best_parameters, sort_keys=True
                                ),
                                **_prefix_metrics(
                                    "validation", best_validation_metrics
                                ),
                                **_prefix_metrics("test", test_metrics),
                                "test_mean_predicted_std": mean_prediction_std,
                                "test_gaussian_nll": gaussian_nll,
                                "test_interval_95_coverage": interval_95_coverage,
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
                        if test_prediction_std is None:
                            predictions["predicted_log_s_std"] = np.nan
                            predictions["predicted_log_s_lower_95"] = np.nan
                            predictions["predicted_log_s_upper_95"] = np.nan
                        else:
                            predictions["predicted_log_s_std"] = test_prediction_std
                            predictions["predicted_log_s_lower_95"] = (
                                test_prediction - 1.96 * test_prediction_std
                            )
                            predictions["predicted_log_s_upper_95"] = (
                                test_prediction + 1.96 * test_prediction_std
                            )
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
        "xgboost_version": xgboost.__version__,
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
