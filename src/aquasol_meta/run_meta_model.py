from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .run_baselines import _flatten_report, _write_table

PRIMARY_TARGET = "y_top3_candidate"
RANDOM_SEED = 20260309

CANDIDATE_PREDICTORS = (
    "x_model",
    "x_model_class",
    "x_feature_type",
    "x_features_after_preprocessing",
)

TASK_PREDICTORS = (
    "x_fit_molecules",
    "x_fit_scaffold_count",
    "x_fit_log_s_mean",
    "x_fit_log_s_std",
    "x_fit_log_s_iqr",
    "x_fit_mean_observation_count",
    "x_fit_multiple_source_fraction",
    "x_fit_single_observation_fraction",
    "x_test_molecules",
    "x_test_scaffold_count",
    "x_test_unseen_scaffold_fraction",
    "x_test_max_train_similarity_median",
    "x_test_max_train_similarity_p10",
    "x_test_similarity_below_0p3_fraction",
    "x_test_similarity_below_0p5_fraction",
    "x_shift_mol_wt_smd",
    "x_shift_mol_log_p_smd",
    "x_shift_tpsa_smd",
    "x_shift_fraction_csp3_smd",
    "x_shift_rotatable_bonds_smd",
    "x_shift_ring_count_smd",
)

VALIDATION_PREDICTORS = (
    "x_validation_rmse",
    "x_validation_mae",
    "x_validation_r2",
    "x_validation_spearman",
)

FIXED_BASELINES = {
    "fixed_svr_rdkit": ("svr", "rdkit_descriptors"),
    "fixed_random_forest_rdkit": ("random_forest", "rdkit_descriptors"),
    "fixed_xgboost_rdkit": ("xgboost", "rdkit_descriptors"),
    "fixed_chemprop_graph": ("chemprop", "molecular_graph"),
}

META_MODEL_NAMES = (
    "logistic",
    "bayesian_logistic_laplace",
    "random_forest",
    "hist_gradient_boosting",
)

ABLATION_NAMES = (
    "hgb_identity_only",
    "hgb_identity_validation",
    "hgb_task_chemistry_no_validation",
    "hgb_full",
)


def meta_predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Return a preregistered, leakage-safe predictor list for source meta-learning."""
    required = [*CANDIDATE_PREDICTORS, *TASK_PREDICTORS, *VALIDATION_PREDICTORS]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Source meta-dataset is missing predictors: {missing}")
    hyperparameters = sorted(column for column in frame if column.startswith("x_hp_"))
    columns = [*required, *hyperparameters]
    forbidden = [column for column in columns if column.startswith("y_")]
    if forbidden:
        raise RuntimeError(f"Outcome columns entered the predictor set: {forbidden}")
    return columns


def _validate_source_meta(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_id",
        "task_id",
        "heldout_source",
        "selection_eligible",
        "x_model",
        "x_feature_type",
        "x_validation_rmse",
        PRIMARY_TARGET,
        "y_test_rmse",
        "y_rmse_regret_vs_best",
        "y_within_0p1_log_s_of_best",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source meta-dataset is missing columns: {missing}")
    if frame["run_id"].duplicated().any():
        raise ValueError("Source meta-dataset contains duplicate run IDs")
    source_counts = frame.groupby("task_id")["heldout_source"].nunique()
    if not source_counts.eq(1).all():
        raise ValueError("At least one task maps to more than one held-out source")
    eligible = frame.loc[frame["selection_eligible"].astype(bool)].copy()
    outcomes = [
        PRIMARY_TARGET,
        "y_test_rmse",
        "y_rmse_regret_vs_best",
        "y_within_0p1_log_s_of_best",
    ]
    if eligible[outcomes].isna().any().any():
        raise ValueError("Eligible source rows contain missing meta-model outcomes")
    positive_counts = eligible.groupby("task_id")[PRIMARY_TARGET].sum()
    if not positive_counts.eq(3).all():
        raise ValueError("Every source task must contain exactly three top-three candidates")
    if eligible["heldout_source"].nunique() < 3:
        raise ValueError("At least three held-out sources are required")
    return eligible.reset_index(drop=True)


def _source_folds(frame: pd.DataFrame) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    """Yield outer folds that never divide rows from the same source."""
    sources = sorted(frame["heldout_source"].astype(str).unique())
    for source in sources:
        test_mask = frame["heldout_source"].astype(str).eq(source).to_numpy()
        train_index = np.flatnonzero(~test_mask)
        test_index = np.flatnonzero(test_mask)
        if not len(train_index) or not len(test_index):
            raise RuntimeError(f"Invalid leave-one-source-out fold for {source}")
        yield source, train_index, test_index


def _preprocessor(
    frame: pd.DataFrame, predictors: list[str]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = [
        column
        for column in predictors
        if isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]
    numeric = [column for column in predictors if column not in categorical]
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    return transformer, numeric, categorical


def _laplace_probabilities(
    model: LogisticRegression,
    x_train: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate coefficient uncertainty around a logistic MAP estimate."""
    train_design = np.column_stack([np.ones(len(x_train)), x_train])
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    train_probability = model.predict_proba(x_train)[:, 1]
    weight = np.clip(train_probability * (1.0 - train_probability), 1e-8, None)
    hessian = train_design.T @ (weight[:, None] * train_design)
    prior_precision = np.zeros(hessian.shape[0])
    prior_precision[1:] = 1.0 / c_value
    hessian += np.diag(prior_precision)
    hessian[0, 0] += 1e-8
    covariance = np.linalg.pinv(hessian, hermitian=True)
    coefficients = np.concatenate([model.intercept_, model.coef_.ravel()])
    linear_mean = test_design @ coefficients
    linear_variance = np.einsum(
        "ij,jk,ik->i", test_design, covariance, test_design, optimize=True
    )
    linear_sd = np.sqrt(np.maximum(linear_variance, 0.0))
    probability_mean = expit(linear_mean / np.sqrt(1.0 + np.pi * linear_variance / 8.0))
    probability_lower = expit(linear_mean - 1.96 * linear_sd)
    probability_upper = expit(linear_mean + 1.96 * linear_sd)
    # A skewed logistic-normal distribution can place its mean just outside an
    # equal-tailed interval. Report a conservative interval containing the point estimate.
    probability_lower = np.minimum(probability_lower, probability_mean)
    probability_upper = np.maximum(probability_upper, probability_mean)
    return probability_mean, probability_lower, probability_upper, covariance


def _prediction_frame(
    test: pd.DataFrame,
    source: str,
    meta_model: str,
    probability: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "task_id",
        "heldout_source",
        "x_model",
        "x_model_class",
        "x_feature_type",
        "x_validation_rmse",
        PRIMARY_TARGET,
        "y_test_rmse",
        "y_rmse_regret_vs_best",
        "y_within_0p1_log_s_of_best",
    ]
    output = test[columns].copy()
    output.insert(0, "outer_heldout_source", source)
    output.insert(1, "meta_model", meta_model)
    output["predicted_top3_probability"] = probability
    output["probability_lower_95"] = np.nan if lower is None else lower
    output["probability_upper_95"] = np.nan if upper is None else upper
    return output


def _importance_rows(
    source: str,
    meta_model: str,
    feature_names: np.ndarray,
    importance: np.ndarray,
    posterior_sd: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (feature, value) in enumerate(zip(feature_names, importance, strict=True)):
        rows.append(
            {
                "outer_heldout_source": source,
                "meta_model": meta_model,
                "transformed_feature": str(feature),
                "importance": float(value),
                "absolute_importance": float(abs(value)),
                "posterior_sd": (
                    float(posterior_sd[index]) if posterior_sd is not None else np.nan
                ),
            }
        )
    return rows


def fit_leave_one_source_out(
    frame: pd.DataFrame,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit four fixed meta-models under strict leave-one-source-out evaluation."""
    prediction_frames: list[pd.DataFrame] = []
    importance: list[dict[str, Any]] = []
    c_value = 1.0
    for fold_number, (source, train_index, test_index) in enumerate(_source_folds(frame), 1):
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        train_sources = set(train["heldout_source"].astype(str))
        test_sources = set(test["heldout_source"].astype(str))
        if train_sources.intersection(test_sources):
            raise RuntimeError(f"Source leakage detected in outer fold {source}")

        transformer, _, _ = _preprocessor(train, predictors)
        x_train = np.asarray(transformer.fit_transform(train[predictors]), dtype=float)
        x_test = np.asarray(transformer.transform(test[predictors]), dtype=float)
        y_train = train[PRIMARY_TARGET].astype(int).to_numpy()
        feature_names = transformer.get_feature_names_out()

        logistic = LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")
        logistic.fit(x_train, y_train)
        logistic_probability = logistic.predict_proba(x_test)[:, 1]
        prediction_frames.append(
            _prediction_frame(test, source, "logistic", logistic_probability)
        )
        importance.extend(
            _importance_rows(
                source,
                "logistic",
                feature_names,
                logistic.coef_.ravel(),
            )
        )

        bayes_mean, bayes_lower, bayes_upper, covariance = _laplace_probabilities(
            logistic, x_train, x_test, c_value
        )
        prediction_frames.append(
            _prediction_frame(
                test,
                source,
                "bayesian_logistic_laplace",
                bayes_mean,
                bayes_lower,
                bayes_upper,
            )
        )
        importance.extend(
            _importance_rows(
                source,
                "bayesian_logistic_laplace",
                feature_names,
                logistic.coef_.ravel(),
                np.sqrt(np.maximum(np.diag(covariance)[1:], 0.0)),
            )
        )

        random_forest = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_SEED + fold_number,
            n_jobs=-1,
        )
        random_forest.fit(x_train, y_train)
        prediction_frames.append(
            _prediction_frame(
                test,
                source,
                "random_forest",
                random_forest.predict_proba(x_test)[:, 1],
            )
        )
        importance.extend(
            _importance_rows(
                source,
                "random_forest",
                feature_names,
                random_forest.feature_importances_,
            )
        )

        gradient_boosting = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_SEED + fold_number,
        )
        gradient_boosting.fit(x_train, y_train)
        prediction_frames.append(
            _prediction_frame(
                test,
                source,
                "hist_gradient_boosting",
                gradient_boosting.predict_proba(x_test)[:, 1],
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = len(frame) * len(META_MODEL_NAMES)
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Unexpected OOS prediction count: {len(predictions)} != {expected_rows}"
        )
    if predictions.duplicated(["meta_model", "run_id"]).any():
        raise RuntimeError("A run received multiple OOS predictions from one meta-model")
    return predictions, pd.DataFrame(importance)


def fit_hgb_feature_ablations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Measure what task chemistry, validation results and hyperparameters add."""
    hyperparameters = sorted(column for column in frame if column.startswith("x_hp_"))
    feature_sets = {
        "hgb_identity_only": list(CANDIDATE_PREDICTORS),
        "hgb_identity_validation": [*CANDIDATE_PREDICTORS, *VALIDATION_PREDICTORS],
        "hgb_task_chemistry_no_validation": [
            *CANDIDATE_PREDICTORS,
            *TASK_PREDICTORS,
            *hyperparameters,
        ],
        "hgb_full": [
            *CANDIDATE_PREDICTORS,
            *TASK_PREDICTORS,
            *VALIDATION_PREDICTORS,
            *hyperparameters,
        ],
    }
    frames: list[pd.DataFrame] = []
    for ablation_name, predictors in feature_sets.items():
        for fold_number, (source, train_index, test_index) in enumerate(
            _source_folds(frame), 1
        ):
            train = frame.iloc[train_index]
            test = frame.iloc[test_index]
            transformer, _, _ = _preprocessor(train, predictors)
            x_train = np.asarray(transformer.fit_transform(train[predictors]), dtype=float)
            x_test = np.asarray(transformer.transform(test[predictors]), dtype=float)
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=RANDOM_SEED + fold_number,
            )
            model.fit(x_train, train[PRIMARY_TARGET].astype(int).to_numpy())
            frames.append(
                _prediction_frame(
                    test,
                    source,
                    ablation_name,
                    model.predict_proba(x_test)[:, 1],
                )
            )
    predictions = pd.concat(frames, ignore_index=True)
    expected_rows = len(frame) * len(ABLATION_NAMES)
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Unexpected ablation prediction count: {len(predictions)} != {expected_rows}"
        )
    return predictions


def _selected_row(
    candidates: pd.DataFrame,
    selector: str,
    sort_columns: list[str],
    ascending: list[bool],
    selection_detail: str,
) -> dict[str, Any]:
    chosen = candidates.sort_values(sort_columns, ascending=ascending, kind="stable").iloc[0]
    return {
        "outer_heldout_source": str(chosen["heldout_source"]),
        "task_id": str(chosen["task_id"]),
        "selector": selector,
        "selected_run_id": str(chosen["run_id"]),
        "selected_model": str(chosen["x_model"]),
        "selected_feature_type": str(chosen["x_feature_type"]),
        "predicted_top3_probability": float(
            chosen.get("predicted_top3_probability", np.nan)
        ),
        "probability_lower_95": float(chosen.get("probability_lower_95", np.nan)),
        "probability_upper_95": float(chosen.get("probability_upper_95", np.nan)),
        "actual_top3": int(chosen[PRIMARY_TARGET]),
        "selected_test_rmse": float(chosen["y_test_rmse"]),
        "rmse_regret_vs_best": float(chosen["y_rmse_regret_vs_best"]),
        "within_0p1_log_s_of_best": int(chosen["y_within_0p1_log_s_of_best"]),
        "selection_detail": selection_detail,
    }


def _build_meta_selected(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (meta_model, _task_id), candidates in predictions.groupby(
        ["meta_model", "task_id"], sort=True
    ):
        rows.append(
            _selected_row(
                candidates,
                f"meta_{meta_model}",
                ["predicted_top3_probability", "x_validation_rmse", "run_id"],
                [False, True, True],
                "highest out-of-source predicted top-three probability",
            )
        )
    return pd.DataFrame(rows)


def build_selected_candidates(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Select one candidate per task for every meta-model and comparator."""
    rows = _build_meta_selected(predictions).to_dict("records")

    for source, _, test_index in _source_folds(frame):
        test = frame.iloc[test_index]
        train = frame.loc[~frame["heldout_source"].astype(str).eq(source)]
        training_best = (
            train.groupby(["x_model", "x_feature_type"], as_index=False)["y_test_rmse"]
            .mean()
            .sort_values(
                ["y_test_rmse", "x_model", "x_feature_type"], kind="stable"
            )
            .iloc[0]
        )
        training_key = (
            str(training_best["x_model"]),
            str(training_best["x_feature_type"]),
        )
        for task_id, candidates in test.groupby("task_id", sort=True):
            rows.append(
                _selected_row(
                    candidates,
                    "validation_rmse",
                    ["x_validation_rmse", "run_id"],
                    [True, True],
                    "lowest validation RMSE within the unseen task",
                )
            )
            rows.append(
                _selected_row(
                    candidates,
                    "oracle",
                    ["y_test_rmse", "run_id"],
                    [True, True],
                    "unattainable lower bound using held-out test RMSE",
                )
            )
            training_candidates = candidates.loc[
                candidates["x_model"].eq(training_key[0])
                & candidates["x_feature_type"].eq(training_key[1])
            ]
            if training_candidates.empty:
                raise RuntimeError(
                    f"Training-source best candidate {training_key} is absent from {task_id}"
                )
            rows.append(
                _selected_row(
                    training_candidates,
                    "training_source_best",
                    ["run_id"],
                    [True],
                    f"best mean training-source candidate: {training_key[0]} + {training_key[1]}",
                )
            )
            for selector, (model, feature_type) in FIXED_BASELINES.items():
                fixed = candidates.loc[
                    candidates["x_model"].eq(model)
                    & candidates["x_feature_type"].eq(feature_type)
                ]
                if fixed.empty:
                    raise RuntimeError(
                        f"Fixed candidate {model}/{feature_type} absent from {task_id}"
                    )
                rows.append(
                    _selected_row(
                        fixed,
                        selector,
                        ["run_id"],
                        [True],
                        f"fixed candidate: {model} + {feature_type}",
                    )
                )
    selected = pd.DataFrame(rows).sort_values(
        ["selector", "outer_heldout_source", "task_id"], kind="stable"
    ).reset_index(drop=True)
    counts = selected.groupby("selector")["task_id"].nunique()
    if not counts.eq(frame["task_id"].nunique()).all():
        raise RuntimeError("At least one selector did not choose exactly one row per task")
    if selected.duplicated(["selector", "task_id"]).any():
        raise RuntimeError("A selector chose multiple candidates for one task")
    return selected


def _classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame[PRIMARY_TARGET].astype(int).to_numpy()
    probability = frame["predicted_top3_probability"].to_numpy()
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "brier_score": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
    }


def build_discrimination_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for meta_model, model_rows in predictions.groupby("meta_model", sort=True):
        rows.append(
            {
                "scope": "all_sources",
                "heldout_source": "ALL",
                "meta_model": meta_model,
                "candidate_rows": int(len(model_rows)),
                "positive_rows": int(model_rows[PRIMARY_TARGET].sum()),
                **_classification_metrics(model_rows),
            }
        )
        for source, source_rows in model_rows.groupby("outer_heldout_source", sort=True):
            rows.append(
                {
                    "scope": "heldout_source",
                    "heldout_source": source,
                    "meta_model": meta_model,
                    "candidate_rows": int(len(source_rows)),
                    "positive_rows": int(source_rows[PRIMARY_TARGET].sum()),
                    **_classification_metrics(source_rows),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "meta_model", "heldout_source"], kind="stable"
    ).reset_index(drop=True)


def _selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "tasks": int(frame["task_id"].nunique()),
        "mean_selected_test_rmse": float(frame["selected_test_rmse"].mean()),
        "mean_rmse_regret": float(frame["rmse_regret_vs_best"].mean()),
        "median_rmse_regret": float(frame["rmse_regret_vs_best"].median()),
        "top3_selection_rate": float(frame["actual_top3"].mean()),
        "within_0p1_of_best_rate": float(frame["within_0p1_log_s_of_best"].mean()),
    }


def build_selection_summaries(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = []
    fold = []
    for selector, rows in selected.groupby("selector", sort=True):
        overall.append({"selector": selector, **_selection_metrics(rows)})
        for source, source_rows in rows.groupby("outer_heldout_source", sort=True):
            fold.append(
                {
                    "outer_heldout_source": source,
                    "selector": selector,
                    **_selection_metrics(source_rows),
                }
            )
    overall_frame = pd.DataFrame(overall).sort_values(
        ["mean_rmse_regret", "selector"], kind="stable"
    ).reset_index(drop=True)
    fold_frame = pd.DataFrame(fold).sort_values(
        ["outer_heldout_source", "mean_rmse_regret", "selector"], kind="stable"
    ).reset_index(drop=True)
    return overall_frame, fold_frame


def build_cluster_bootstrap_uncertainty(
    selected: pd.DataFrame,
    iterations: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Bootstrap complete source clusters, preserving all five repeated tasks per source."""
    rng = np.random.default_rng(seed)
    metric_columns = {
        "mean_selected_test_rmse": "selected_test_rmse",
        "mean_rmse_regret": "rmse_regret_vs_best",
        "top3_selection_rate": "actual_top3",
        "within_0p1_of_best_rate": "within_0p1_log_s_of_best",
    }
    output: list[dict[str, Any]] = []
    for selector, rows in selected.groupby("selector", sort=True):
        sources = sorted(rows["outer_heldout_source"].unique())
        by_source = {
            source: rows.loc[rows["outer_heldout_source"].eq(source)] for source in sources
        }
        samples = {metric: [] for metric in metric_columns}
        for _ in range(iterations):
            sampled_sources = rng.choice(sources, size=len(sources), replace=True)
            sample = pd.concat([by_source[source] for source in sampled_sources])
            for metric, column in metric_columns.items():
                samples[metric].append(float(sample[column].mean()))
        for metric, column in metric_columns.items():
            values = np.asarray(samples[metric])
            output.append(
                {
                    "selector": selector,
                    "metric": metric,
                    "estimate": float(rows[column].mean()),
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                    "bootstrap_unit": "heldout_source",
                    "source_clusters": int(len(sources)),
                    "iterations": int(iterations),
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(output).sort_values(["metric", "estimate", "selector"]).reset_index(
        drop=True
    )


def build_paired_selector_comparisons(
    selected: pd.DataFrame,
    iterations: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Estimate paired source-bootstrap differences from practical comparators."""
    rng = np.random.default_rng(seed)
    meta_selectors = sorted(
        selector for selector in selected["selector"].unique() if selector.startswith("meta_")
    )
    references = [
        "fixed_xgboost_rdkit",
        "fixed_svr_rdkit",
        "validation_rmse",
        "training_source_best",
    ]
    metrics = {
        "mean_rmse_regret": ("rmse_regret_vs_best", "lower_is_better"),
        "top3_selection_rate": ("actual_top3", "higher_is_better"),
    }
    output: list[dict[str, Any]] = []
    for selector in meta_selectors:
        candidate = selected.loc[selected["selector"].eq(selector)]
        for reference in references:
            comparator = selected.loc[selected["selector"].eq(reference)]
            paired = candidate.merge(
                comparator,
                on=["outer_heldout_source", "task_id"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            sources = sorted(paired["outer_heldout_source"].unique())
            for metric, (column, direction) in metrics.items():
                paired["difference"] = (
                    paired[f"{column}_candidate"] - paired[f"{column}_reference"]
                )
                source_difference = paired.groupby("outer_heldout_source")[
                    "difference"
                ].mean()
                bootstrap = np.empty(iterations)
                for index in range(iterations):
                    sampled = rng.choice(sources, size=len(sources), replace=True)
                    bootstrap[index] = float(source_difference.loc[sampled].mean())
                probability_better = (
                    float(np.mean(bootstrap < 0.0))
                    if direction == "lower_is_better"
                    else float(np.mean(bootstrap > 0.0))
                )
                output.append(
                    {
                        "selector": selector,
                        "reference": reference,
                        "metric": metric,
                        "direction": direction,
                        "paired_difference": float(paired["difference"].mean()),
                        "lower_95": float(np.quantile(bootstrap, 0.025)),
                        "upper_95": float(np.quantile(bootstrap, 0.975)),
                        "bootstrap_probability_better": probability_better,
                        "source_clusters": int(len(sources)),
                        "iterations": int(iterations),
                        "seed": int(seed),
                    }
                )
    return pd.DataFrame(output).sort_values(
        ["metric", "selector", "reference"], kind="stable"
    ).reset_index(drop=True)


def build_feature_audit(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    rows = []
    for column in predictors:
        if column in CANDIDATE_PREDICTORS:
            group = "candidate_identity"
        elif column in TASK_PREDICTORS:
            group = "task_chemistry"
        elif column in VALIDATION_PREDICTORS:
            group = "validation_performance"
        else:
            group = "selected_hyperparameter"
        categorical = (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
        rows.append(
            {
                "column": column,
                "predictor_group": group,
                "data_type": "categorical" if categorical else "numeric",
                "non_missing_rows": int(frame[column].notna().sum()),
                "missing_fraction": float(frame[column].isna().mean()),
                "unique_non_missing_values": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def build_meta_model_artifacts(
    source_meta: pd.DataFrame,
    results_dir: Path,
    reports_dir: Path,
    bootstrap_iterations: int = 2000,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    eligible = _validate_source_meta(source_meta)
    predictors = meta_predictor_columns(eligible)
    predictions, importance = fit_leave_one_source_out(eligible, predictors)
    ablation_predictions = fit_hgb_feature_ablations(eligible)
    selected = build_selected_candidates(eligible, predictions)
    ablation_selected = _build_meta_selected(ablation_predictions)
    discrimination = build_discrimination_summary(predictions)
    ablation_discrimination = build_discrimination_summary(ablation_predictions)
    performance, fold_performance = build_selection_summaries(selected)
    ablation_performance, ablation_fold_performance = build_selection_summaries(
        ablation_selected
    )
    uncertainty = build_cluster_bootstrap_uncertainty(
        selected, iterations=bootstrap_iterations
    )
    paired_comparisons = build_paired_selector_comparisons(
        selected, iterations=bootstrap_iterations
    )
    feature_audit = build_feature_audit(eligible, predictors)

    paths = {
        "oos_predictions_csv": results_dir / "meta_model_oos_predictions.csv",
        "oos_predictions_parquet": results_dir / "meta_model_oos_predictions.parquet",
        "selected_candidates_csv": results_dir / "meta_model_selected_candidates.csv",
        "selected_candidates_parquet": results_dir
        / "meta_model_selected_candidates.parquet",
        "performance_csv": results_dir / "meta_model_performance_summary.csv",
        "fold_performance_csv": results_dir / "meta_model_fold_performance.csv",
        "discrimination_csv": results_dir / "meta_model_discrimination_summary.csv",
        "uncertainty_csv": results_dir / "meta_model_bootstrap_uncertainty.csv",
        "feature_importance_csv": results_dir / "meta_model_feature_importance.csv",
        "paired_comparisons_csv": results_dir / "meta_model_paired_comparisons.csv",
        "ablation_predictions_parquet": results_dir
        / "meta_model_ablation_oos_predictions.parquet",
        "ablation_selected_csv": results_dir / "meta_model_ablation_selected_candidates.csv",
        "ablation_performance_csv": results_dir / "meta_model_ablation_performance.csv",
        "ablation_fold_performance_csv": results_dir
        / "meta_model_ablation_fold_performance.csv",
        "ablation_discrimination_csv": results_dir
        / "meta_model_ablation_discrimination.csv",
        "feature_audit_csv": reports_dir / "meta_model_feature_audit.csv",
        "report_json": reports_dir / "meta_model_report.json",
        "report_csv": reports_dir / "meta_model_report.csv",
    }
    _write_table(
        predictions,
        paths["oos_predictions_csv"],
        paths["oos_predictions_parquet"],
    )
    _write_table(
        selected,
        paths["selected_candidates_csv"],
        paths["selected_candidates_parquet"],
    )
    performance.to_csv(paths["performance_csv"], index=False)
    fold_performance.to_csv(paths["fold_performance_csv"], index=False)
    discrimination.to_csv(paths["discrimination_csv"], index=False)
    uncertainty.to_csv(paths["uncertainty_csv"], index=False)
    importance.to_csv(paths["feature_importance_csv"], index=False)
    paired_comparisons.to_csv(paths["paired_comparisons_csv"], index=False)
    ablation_predictions.to_parquet(
        paths["ablation_predictions_parquet"], index=False, compression="zstd"
    )
    ablation_selected.to_csv(paths["ablation_selected_csv"], index=False)
    ablation_performance.to_csv(paths["ablation_performance_csv"], index=False)
    ablation_fold_performance.to_csv(
        paths["ablation_fold_performance_csv"], index=False
    )
    ablation_discrimination.to_csv(paths["ablation_discrimination_csv"], index=False)
    feature_audit.to_csv(paths["feature_audit_csv"], index=False)

    non_oracle = performance.loc[~performance["selector"].eq("oracle")]
    best = non_oracle.sort_values(["mean_rmse_regret", "selector"], kind="stable").iloc[0]
    practical_comparators = performance.loc[
        ~performance["selector"].str.startswith("meta_")
        & ~performance["selector"].eq("oracle")
    ]
    strongest_comparator = practical_comparators.sort_values(
        ["mean_rmse_regret", "selector"], kind="stable"
    ).iloc[0]
    best_ablation = ablation_performance.sort_values(
        ["mean_rmse_regret", "selector"], kind="stable"
    ).iloc[0]
    full_ablation = ablation_performance.loc[
        ablation_performance["selector"].eq("meta_hgb_full")
    ].iloc[0]
    identity_ablation = ablation_performance.loc[
        ablation_performance["selector"].eq("meta_hgb_identity_only")
    ].iloc[0]
    primary_pairwise = paired_comparisons.loc[
        paired_comparisons["selector"].eq(str(best["selector"]))
        & paired_comparisons["reference"].eq(str(strongest_comparator["selector"]))
        & paired_comparisons["metric"].eq("mean_rmse_regret")
    ].iloc[0]
    selector_results = {
        str(row["selector"]): {
            "mean_selected_test_rmse": float(row["mean_selected_test_rmse"]),
            "mean_rmse_regret": float(row["mean_rmse_regret"]),
            "top3_selection_rate": float(row["top3_selection_rate"]),
            "within_0p1_of_best_rate": float(row["within_0p1_of_best_rate"]),
        }
        for _, row in performance.iterrows()
    }
    report: dict[str, Any] = {
        "status": "pass",
        "primary_target": PRIMARY_TARGET,
        "primary_target_definition": (
            "One when a selection-eligible candidate is among the three lowest test-RMSE "
            "candidates within its source-holdout task; zero otherwise."
        ),
        "evaluation_design": "leave_one_heldout_source_out",
        "source_rows_before_eligibility_filter": int(len(source_meta)),
        "eligible_candidate_rows": int(len(eligible)),
        "source_tasks": int(eligible["task_id"].nunique()),
        "independent_source_groups": int(eligible["heldout_source"].nunique()),
        "candidates_per_task": sorted(
            int(value) for value in eligible.groupby("task_id").size().unique()
        ),
        "positive_rows": int(eligible[PRIMARY_TARGET].sum()),
        "positive_prevalence": float(eligible[PRIMARY_TARGET].mean()),
        "meta_models": list(META_MODEL_NAMES),
        "predictor_columns": predictors,
        "predictor_count": int(len(predictors)),
        "excluded_identifiers": ["run_id", "task_id", "heldout_source", "repeat", "seed"],
        "excluded_outcomes": sorted(column for column in source_meta if column.startswith("y_")),
        "leakage_policy": (
            "Every candidate and repeat from one source is held out together. Source names, "
            "task identifiers, seeds, repeat numbers and all held-out outcomes are excluded from X."
        ),
        "within_benchmark_rows_used": False,
        "within_benchmark_exclusion_reason": (
            "The primary claim concerns unseen experimental sources; ordinary within-dataset "
            "splits are not independent source groups and are excluded from model assessment."
        ),
        "bayesian_method": (
            "Gaussian Laplace approximation around the L2-regularized logistic MAP estimate. "
            "Reported intervals quantify coefficient uncertainty in the top-three probability "
            "and are conservatively expanded to contain the logistic-normal point estimate."
        ),
        "uncertainty_method": (
            f"Percentile bootstrap with {bootstrap_iterations} resamples of complete held-out "
            "source clusters. With only five source clusters, intervals are exploratory."
        ),
        "best_non_oracle_selector": str(best["selector"]),
        "best_non_oracle_mean_rmse_regret": float(best["mean_rmse_regret"]),
        "strongest_practical_comparator": str(strongest_comparator["selector"]),
        "strongest_practical_comparator_mean_rmse_regret": float(
            strongest_comparator["mean_rmse_regret"]
        ),
        "best_selector_paired_regret_difference_vs_strongest_comparator": float(
            primary_pairwise["paired_difference"]
        ),
        "best_selector_paired_regret_difference_lower_95": float(
            primary_pairwise["lower_95"]
        ),
        "best_selector_paired_regret_difference_upper_95": float(
            primary_pairwise["upper_95"]
        ),
        "best_selector_bootstrap_probability_better_than_strongest_comparator": float(
            primary_pairwise["bootstrap_probability_better"]
        ),
        "ablation_models": list(ABLATION_NAMES),
        "best_ablation_selector": str(best_ablation["selector"]),
        "best_ablation_mean_rmse_regret": float(best_ablation["mean_rmse_regret"]),
        "full_feature_hgb_mean_rmse_regret": float(full_ablation["mean_rmse_regret"]),
        "identity_only_hgb_mean_rmse_regret": float(
            identity_ablation["mean_rmse_regret"]
        ),
        "task_chemistry_improved_regret_over_identity_only": bool(
            full_ablation["mean_rmse_regret"]
            < identity_ablation["mean_rmse_regret"]
        ),
        "selector_results": selector_results,
        "publication_limitation": (
            "Five source groups support a leakage-safe proof of concept but provide limited "
            "independent sample size. A genuinely external sixth dataset remains important."
        ),
    }
    report["output_files"] = {
        key: {"path": str(path), "bytes": int(path.stat().st_size)}
        for key, path in paths.items()
        if key not in {"report_json", "report_csv"}
    }
    paths["report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["report_csv"], index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-success meta-models with complete experimental sources held out."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    root = args.root.resolve()
    source_path = root / "data_processed/source_holdout_meta_model_dataset.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Source meta-dataset not found: {source_path}")
    paths = build_meta_model_artifacts(
        pd.read_parquet(source_path),
        root / "results",
        root / "reports",
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print("Meta-model evaluation completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
