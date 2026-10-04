from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FAILURE_THRESHOLDS = (0.5, 1.0, 2.0)
PRIMARY_FAILURE_THRESHOLD = 1.0
META_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "FractionCSP3",
    "NumRotatableBonds",
    "RingCount",
)

CORE_META_PREDICTORS = (
    "x_model",
    "x_feature_type",
    "x_split_strategy",
    "x_features_after_preprocessing",
    "x_validation_rmse",
    "x_fit_molecules",
    "x_fit_scaffold_count",
    "x_fit_log_s_std",
    "x_fit_log_s_iqr",
    "x_test_unseen_scaffold_fraction",
    "x_test_max_train_similarity_median",
    "x_test_max_train_similarity_p10",
    "x_shift_mol_wt_smd",
    "x_shift_mol_log_p_smd",
    "x_shift_tpsa_smd",
    "x_shift_fraction_csp3_smd",
)

MODEL_CLASSES = {
    "dummy": "constant_baseline",
    "ridge": "linear",
    "random_forest": "tree_ensemble",
    "extra_trees": "tree_ensemble",
    "hist_gradient_boosting": "tree_ensemble",
    "svr": "kernel",
    "mlp": "neural_network",
    "elastic_net": "linear",
    "knn": "local_similarity",
    "xgboost": "tree_ensemble",
    "ngboost": "probabilistic_boosting",
    "esol": "chemistry_equation",
    "chemprop": "graph_neural_network",
}

DESCRIPTOR_SLUGS = {
    "MolWt": "mol_wt",
    "MolLogP": "mol_log_p",
    "TPSA": "tpsa",
    "FractionCSP3": "fraction_csp3",
    "NumRotatableBonds": "rotatable_bonds",
    "RingCount": "ring_count",
}


def _threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):.1f}".replace(".", "p")


def failure_column(threshold: float) -> str:
    return f"q_abs_error_gt_{_threshold_suffix(threshold)}"


def failure_rate_column(threshold: float) -> str:
    return f"failure_rate_gt_{_threshold_suffix(threshold)}"


def add_failure_labels(
    predictions: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_FAILURE_THRESHOLDS,
    primary_threshold: float = PRIMARY_FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Add molecule-level failure indicators without changing the raw prediction table."""
    required = {
        "run_id",
        "molecule_id",
        "true_log_s",
        "predicted_log_s",
        "residual",
        "absolute_error",
        "squared_error",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {sorted(missing)}")
    if predictions.duplicated(["run_id", "molecule_id"]).any():
        raise ValueError("Prediction table contains duplicate run/molecule rows")

    thresholds = tuple(sorted({float(value) for value in thresholds}))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("Failure thresholds must contain positive values")
    primary_threshold = float(primary_threshold)
    if primary_threshold not in thresholds:
        raise ValueError("The primary failure threshold must be included in thresholds")

    frame = predictions.copy()
    expected_residual = frame["true_log_s"] - frame["predicted_log_s"]
    if not np.allclose(frame["residual"], expected_residual, rtol=1e-10, atol=1e-12):
        raise ValueError("Stored residuals do not equal true_log_s - predicted_log_s")
    if not np.allclose(frame["absolute_error"], expected_residual.abs()):
        raise ValueError("Stored absolute errors do not match absolute residuals")

    for threshold in thresholds:
        frame[failure_column(threshold)] = (
            frame["absolute_error"] > threshold
        ).astype("int8")
    frame["q_i"] = frame[failure_column(primary_threshold)].astype("int8")
    frame["primary_failure_threshold_log_s"] = primary_threshold
    frame["error_direction"] = np.select(
        [frame["residual"] > 0, frame["residual"] < 0],
        ["underpredicted_log_s", "overpredicted_log_s"],
        default="exact",
    )
    frame["absolute_error_bin"] = pd.cut(
        frame["absolute_error"],
        bins=[0.0, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.5", "0.5-<1.0", "1.0-<2.0", ">=2.0"],
        right=False,
        include_lowest=True,
    ).astype("string")
    return frame


def build_run_failure_summary(
    labeled_predictions: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_FAILURE_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize molecule-level errors into one outcome row per completed model run."""
    thresholds = tuple(sorted({float(value) for value in thresholds}))
    metadata = ["model", "feature_type", "split_strategy", "repeat", "seed"]
    required = {"run_id", "residual", "absolute_error", "squared_error", *metadata}
    required.update(failure_column(value) for value in thresholds)
    missing = required.difference(labeled_predictions.columns)
    if missing:
        raise ValueError(f"Labeled predictions are missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for run_id, frame in labeled_predictions.groupby("run_id", sort=True):
        row: dict[str, Any] = {"run_id": run_id}
        for column in metadata:
            values = frame[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"{run_id} has inconsistent {column} values")
            row[column] = values.iloc[0]
        absolute_error = frame["absolute_error"]
        residual = frame["residual"]
        row.update(
            {
                "n_test_predictions": int(len(frame)),
                "mean_residual": float(residual.mean()),
                "residual_sd": float(residual.std(ddof=1)),
                "rmse": float(np.sqrt(frame["squared_error"].mean())),
                "mae": float(absolute_error.mean()),
                "median_absolute_error": float(absolute_error.median()),
                "p90_absolute_error": float(absolute_error.quantile(0.90)),
                "p95_absolute_error": float(absolute_error.quantile(0.95)),
                "maximum_absolute_error": float(absolute_error.max()),
                "underprediction_rate": float((residual > 0).mean()),
                "overprediction_rate": float((residual < 0).mean()),
            }
        )
        for threshold in thresholds:
            row[failure_rate_column(threshold)] = float(
                frame[failure_column(threshold)].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["split_strategy", "repeat", "feature_type", "model"], kind="stable"
    ).reset_index(drop=True)


def build_failure_group_summary(
    run_summary: pd.DataFrame,
    primary_threshold: float = PRIMARY_FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate run-level failure measurements across the five split repeats."""
    primary_rate = failure_rate_column(primary_threshold)
    required = {
        "split_strategy",
        "feature_type",
        "model",
        "repeat",
        "n_test_predictions",
        "rmse",
        "mae",
        "median_absolute_error",
        primary_rate,
    }
    missing = required.difference(run_summary.columns)
    if missing:
        raise ValueError(f"Run summary is missing columns: {sorted(missing)}")

    return (
        run_summary.groupby(
            ["split_strategy", "feature_type", "model"], sort=True, as_index=False
        )
        .agg(
            repeats=("repeat", "nunique"),
            total_test_predictions=("n_test_predictions", "sum"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_median_absolute_error=("median_absolute_error", "mean"),
            mean_primary_failure_rate=(primary_rate, "mean"),
            sd_primary_failure_rate=(primary_rate, "std"),
        )
        .sort_values(["split_strategy", "mean_rmse"], kind="stable")
        .reset_index(drop=True)
    )


def build_molecule_failure_summary(
    labeled_predictions: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_FAILURE_THRESHOLDS,
    primary_threshold: float = PRIMARY_FAILURE_THRESHOLD,
    include_dummy: bool = False,
) -> pd.DataFrame:
    """Aggregate repeated held-out evaluations to identify consistently hard molecules."""
    thresholds = tuple(sorted({float(value) for value in thresholds}))
    frame = labeled_predictions.copy()
    if not include_dummy:
        frame = frame.loc[~frame["model"].eq("dummy")].copy()
    if frame.empty:
        raise ValueError("No molecule-level predictions remain for failure summarization")
    frame["task_id"] = (
        frame["split_strategy"].astype(str) + "_r" + frame["repeat"].astype(str)
    )
    audit_columns = [
        "true_log_s",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "log_s_range",
        "scaffold_group",
    ]
    available_audit = [column for column in audit_columns if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for molecule_id, molecule in frame.groupby("molecule_id", sort=True):
        row: dict[str, Any] = {"molecule_id": molecule_id}
        for column in available_audit:
            values = molecule[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"{molecule_id} has inconsistent {column} values")
            row[column] = values.iloc[0]
        similarity = pd.to_numeric(molecule["maximum_train_similarity"], errors="coerce")
        row.update(
            {
                "n_held_out_evaluations": int(len(molecule)),
                "n_test_tasks": int(molecule["task_id"].nunique()),
                "n_models": int(molecule["model"].nunique()),
                "n_feature_types": int(molecule["feature_type"].nunique()),
                "mean_maximum_train_similarity": float(similarity.mean()),
                "minimum_maximum_train_similarity": float(similarity.min()),
                "mean_residual": float(molecule["residual"].mean()),
                "mean_absolute_error": float(molecule["absolute_error"].mean()),
                "median_absolute_error": float(molecule["absolute_error"].median()),
                "p90_absolute_error": float(molecule["absolute_error"].quantile(0.90)),
                "maximum_absolute_error": float(molecule["absolute_error"].max()),
            }
        )
        for threshold in thresholds:
            row[failure_rate_column(threshold)] = float(
                molecule[failure_column(threshold)].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [failure_rate_column(primary_threshold), "mean_absolute_error"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def _numeric_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)),
        "median": float(numeric.median()),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _task_meta_features(
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
    descriptors: pd.DataFrame,
    descriptor_names: Iterable[str],
) -> pd.DataFrame:
    descriptor_names = tuple(descriptor_names)
    required_assignments = {
        "molecule_id",
        "split_strategy",
        "repeat",
        "seed",
        "partition",
        "scaffold_group",
        "maximum_train_similarity",
    }
    missing = required_assignments.difference(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    required_targets = {
        "molecule_id",
        "log_s_target",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "include_primary_benchmark",
    }
    missing = required_targets.difference(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    missing_descriptors = set(descriptor_names).difference(descriptors.columns)
    if missing_descriptors:
        raise ValueError(f"RDKit descriptor table is missing: {sorted(missing_descriptors)}")

    primary_targets = targets.loc[
        targets["include_primary_benchmark"].astype(bool), list(required_targets)
    ]
    descriptor_columns = ["molecule_id", *descriptor_names]
    rows: list[dict[str, Any]] = []
    for (strategy, repeat), split in assignments.groupby(
        ["split_strategy", "repeat"], sort=True
    ):
        frame = split.merge(primary_targets, on="molecule_id", how="left", validate="one_to_one")
        frame = frame.merge(
            descriptors[descriptor_columns], on="molecule_id", how="left", validate="one_to_one"
        )
        if frame["log_s_target"].isna().any():
            raise ValueError(f"{strategy} repeat {repeat} has split molecules without targets")
        fit = frame.loc[frame["partition"].isin({"train", "validation"})]
        test = frame.loc[frame["partition"].eq("test")]
        if fit.empty or test.empty:
            raise ValueError(f"{strategy} repeat {repeat} has an empty fit or test partition")

        seed_values = split["seed"].drop_duplicates()
        if len(seed_values) != 1:
            raise ValueError(f"{strategy} repeat {repeat} has inconsistent seeds")
        fit_target = _numeric_summary(fit["log_s_target"])
        similarity = pd.to_numeric(test["maximum_train_similarity"], errors="coerce")
        if similarity.isna().any():
            raise ValueError(f"{strategy} repeat {repeat} has missing test similarities")
        fit_scaffolds = set(fit["scaffold_group"].astype(str))
        unseen_scaffold = ~test["scaffold_group"].astype(str).isin(fit_scaffolds)

        row: dict[str, Any] = {
            "task_id": f"{strategy}_r{int(repeat)}",
            "x_split_strategy": str(strategy),
            "repeat": int(repeat),
            "seed": int(seed_values.iloc[0]),
            "x_fit_molecules": int(len(fit)),
            "x_test_molecules": int(len(test)),
            "x_fit_scaffold_count": int(fit["scaffold_group"].nunique()),
            "x_test_scaffold_count": int(test["scaffold_group"].nunique()),
            "x_test_unseen_scaffold_fraction": float(unseen_scaffold.mean()),
            "x_fit_log_s_mean": fit_target["mean"],
            "x_fit_log_s_std": fit_target["std"],
            "x_fit_log_s_median": fit_target["median"],
            "x_fit_log_s_iqr": fit_target["q75"] - fit_target["q25"],
            "x_fit_log_s_min": fit_target["min"],
            "x_fit_log_s_max": fit_target["max"],
            "x_fit_single_observation_fraction": float(
                fit["target_reliability"].eq("single_observation").mean()
            ),
            "x_fit_mean_observation_count": float(fit["n_observations"].mean()),
            "x_fit_multiple_source_fraction": float(fit["n_source_datasets"].gt(1).mean()),
            "x_test_max_train_similarity_mean": float(similarity.mean()),
            "x_test_max_train_similarity_std": float(similarity.std(ddof=1)),
            "x_test_max_train_similarity_median": float(similarity.median()),
            "x_test_max_train_similarity_p10": float(similarity.quantile(0.10)),
            "x_test_max_train_similarity_p25": float(similarity.quantile(0.25)),
            "x_test_max_train_similarity_min": float(similarity.min()),
            "x_test_max_train_similarity_max": float(similarity.max()),
            "x_test_similarity_below_0p3_fraction": float((similarity < 0.3).mean()),
            "x_test_similarity_below_0p5_fraction": float((similarity < 0.5).mean()),
            "x_test_similarity_below_0p7_fraction": float((similarity < 0.7).mean()),
        }
        for descriptor in descriptor_names:
            slug = DESCRIPTOR_SLUGS.get(descriptor, descriptor.lower())
            fit_values = pd.to_numeric(fit[descriptor], errors="coerce")
            test_values = pd.to_numeric(test[descriptor], errors="coerce")
            fit_mean = float(fit_values.mean())
            fit_std = float(fit_values.std(ddof=1))
            test_mean = float(test_values.mean())
            row[f"x_fit_{slug}_mean"] = fit_mean
            row[f"x_fit_{slug}_std"] = fit_std
            row[f"x_test_{slug}_mean"] = test_mean
            row[f"x_test_{slug}_std"] = float(test_values.std(ddof=1))
            row[f"x_shift_{slug}_smd"] = (
                (test_mean - fit_mean) / fit_std if np.isfinite(fit_std) and fit_std > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["x_split_strategy", "repeat"]).reset_index(
        drop=True
    )


def _hyperparameter_columns(parameters: pd.Series) -> pd.DataFrame:
    parsed = parameters.map(json.loads)
    keys = sorted({str(key) for item in parsed for key in item})
    output: dict[str, pd.Series] = {}
    for key in keys:
        values = parsed.map(lambda item, selected_key=key: item.get(selected_key, np.nan))
        output[f"x_hp_{key}"] = values.map(
            lambda value: json.dumps(value) if isinstance(value, (list, tuple)) else value
        )
    return pd.DataFrame(output, index=parameters.index)


def build_meta_model_dataset(
    model_results: pd.DataFrame,
    run_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
    descriptors: pd.DataFrame,
    descriptor_names: Iterable[str] = META_DESCRIPTORS,
    primary_threshold: float = PRIMARY_FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Create one row per fitted model with pre-test predictors and held-out outcomes."""
    if model_results["run_id"].duplicated().any():
        raise ValueError("Model results contain duplicate run IDs")
    if run_summary["run_id"].duplicated().any():
        raise ValueError("Run failure summary contains duplicate run IDs")
    if set(model_results["run_id"]) != set(run_summary["run_id"]):
        raise ValueError("Model results and run failure summary cover different runs")

    tasks = _task_meta_features(assignments, targets, descriptors, descriptor_names)
    base = model_results.copy()
    base["task_id"] = (
        base["split_strategy"].astype(str) + "_r" + base["repeat"].astype(str)
    )
    base["selection_eligible"] = ~base["model"].eq("dummy")
    base["x_model"] = base["model"]
    base["x_model_class"] = base["model"].map(MODEL_CLASSES).fillna("other")
    base["x_feature_type"] = base["feature_type"]
    base["x_features_before_preprocessing"] = base["features_before_preprocessing"]
    base["x_features_after_preprocessing"] = base["features_after_preprocessing"]
    for column in ("n_train", "n_validation", "n_final_fit", "n_test"):
        base[f"x_{column}"] = base[column]
    for metric in ("rmse", "mae", "r2", "spearman"):
        base[f"x_validation_{metric}"] = base[f"validation_{metric}"]
        base[f"y_test_{metric}"] = base[f"test_{metric}"]
    base = pd.concat([base, _hyperparameter_columns(base["best_parameters"])], axis=1)

    failure_rate_columns = [
        column for column in run_summary.columns if column.startswith("failure_rate_gt_")
    ]
    run_outcomes = run_summary[
        [
            "run_id",
            "mean_residual",
            "residual_sd",
            "median_absolute_error",
            "p90_absolute_error",
            "p95_absolute_error",
            "maximum_absolute_error",
            "underprediction_rate",
            "overprediction_rate",
            *failure_rate_columns,
        ]
    ].rename(
        columns={
            "mean_residual": "y_mean_residual",
            "residual_sd": "y_residual_sd",
            "median_absolute_error": "y_median_absolute_error",
            "p90_absolute_error": "y_p90_absolute_error",
            "p95_absolute_error": "y_p95_absolute_error",
            "maximum_absolute_error": "y_maximum_absolute_error",
            "underprediction_rate": "y_underprediction_rate",
            "overprediction_rate": "y_overprediction_rate",
            **{column: f"y_{column}" for column in failure_rate_columns},
        }
    )
    base = base.merge(run_outcomes, on="run_id", how="left", validate="one_to_one")
    base = base.merge(tasks, on=["task_id", "repeat", "seed"], how="left", validate="many_to_one")

    eligible = base["selection_eligible"]
    best_by_task = base.loc[eligible].groupby("task_id")["y_test_rmse"].transform("min")
    base.loc[eligible, "y_rmse_regret_vs_best"] = (
        base.loc[eligible, "y_test_rmse"] - best_by_task
    )
    base.loc[eligible, "y_rank_within_task"] = base.loc[eligible].groupby("task_id")[
        "y_test_rmse"
    ].rank(method="min", ascending=True)
    base["y_top3_candidate"] = pd.Series(pd.NA, index=base.index, dtype="Int8")
    base.loc[eligible, "y_top3_candidate"] = (
        base.loc[eligible, "y_rank_within_task"] <= 3
    ).astype("int8")
    base["y_within_0p1_log_s_of_best"] = pd.Series(pd.NA, index=base.index, dtype="Int8")
    base.loc[eligible, "y_within_0p1_log_s_of_best"] = (
        base.loc[eligible, "y_rmse_regret_vs_best"] <= 0.1
    ).astype("int8")
    base["y_primary_failure_rate"] = base[
        f"y_{failure_rate_column(primary_threshold)}"
    ]

    identifier_columns = ["run_id", "task_id", "repeat", "seed", "selection_eligible"]
    predictor_columns = sorted(column for column in base.columns if column.startswith("x_"))
    outcome_columns = sorted(column for column in base.columns if column.startswith("y_"))
    output = base[identifier_columns + predictor_columns + outcome_columns].copy()
    if output[predictor_columns].filter(regex="test_log_s", axis=1).shape[1]:
        raise RuntimeError("A test-label summary leaked into the predictor columns")
    return output.sort_values(["task_id", "x_feature_type", "x_model"]).reset_index(drop=True)


def build_meta_data_dictionary(meta_dataset: pd.DataFrame) -> pd.DataFrame:
    """Describe column roles so X predictors cannot be confused with y outcomes."""
    descriptions = {
        "run_id": "Unique fitted-model run identifier.",
        "task_id": "Chemical evaluation task: split strategy and repeat.",
        "repeat": "Audit-only repeat number; do not use as a scientific predictor.",
        "seed": "Audit-only random seed; do not use as a scientific predictor.",
        "heldout_source": "Dataset-level grouping variable for source-holdout evaluation.",
        "selection_eligible": "True for substantive model candidates; false for dummy baselines.",
        "x_model": "Candidate regression algorithm.",
        "x_model_class": "Broad algorithm family.",
        "x_feature_type": "Molecular representation supplied to the regression model.",
        "x_split_strategy": "Known evaluation condition: random, scaffold, or low similarity.",
        "x_evaluation_design": "Evaluation design used to construct the chemical task.",
        "y_top3_candidate": (
            "Binary outcome: candidate ranks among the three lowest test RMSE values in its task."
        ),
        "y_within_0p1_log_s_of_best": (
            "Binary outcome: test RMSE is within 0.1 logS of the task winner."
        ),
        "y_rmse_regret_vs_best": (
            "Candidate test RMSE minus the best eligible RMSE in the same task."
        ),
        "y_primary_failure_rate": (
            "Fraction of held-out molecules with absolute error above 1.0 logS."
        ),
    }
    rows: list[dict[str, Any]] = []
    for column in meta_dataset.columns:
        if column.startswith("x_"):
            role = "predictor"
            uses_test_labels = False
            available = True
        elif column.startswith("y_"):
            role = "outcome"
            uses_test_labels = True
            available = False
        else:
            role = "identifier_or_filter"
            uses_test_labels = False
            available = True

        description = descriptions.get(column)
        if description is None:
            if column.startswith("x_hp_"):
                description = "Validation-selected model hyperparameter."
            elif column.startswith("x_validation_"):
                description = "Validation-set performance available before final test evaluation."
            elif column.startswith("x_fit_log_s_"):
                description = "Solubility-target summary from the final fitting data only."
            elif column.startswith("x_fit_"):
                description = "Training-plus-validation composition or chemistry summary."
            elif column.startswith("x_test_max_train_similarity_") or column.startswith(
                "x_test_similarity_"
            ):
                description = "Structure-only test-to-training chemical similarity summary."
            elif column.startswith("x_test_"):
                description = (
                    "Structure-only held-out chemical descriptor summary; no test logS used."
                )
            elif column.startswith("x_shift_"):
                description = "Standardized train-to-test chemical descriptor shift."
            elif column.startswith("x_n_") or column.startswith("x_features_"):
                description = "Modeling sample-size or representation-size predictor."
            elif column.startswith("y_failure_rate_"):
                description = (
                    "Held-out molecule failure rate at the named absolute-error threshold."
                )
            elif column.startswith("y_test_"):
                description = "Held-out regression performance outcome."
            elif column.startswith("y_"):
                description = "Held-out prediction-error outcome."
            else:
                description = "Analysis field."
        rows.append(
            {
                "column": column,
                "role": role,
                "uses_test_log_s": uses_test_labels,
                "available_before_test_evaluation": available,
                "recommended_for_initial_meta_model": column in CORE_META_PREDICTORS,
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def validate_failure_meta_outputs(
    labeled_predictions: pd.DataFrame,
    run_summary: pd.DataFrame,
    meta_dataset: pd.DataFrame,
    model_results: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if not labeled_predictions["q_i"].isin({0, 1}).all():
        errors.append("q_i contains values other than 0 and 1")
    if labeled_predictions.duplicated(["run_id", "molecule_id"]).any():
        errors.append("Labeled predictions contain duplicate run/molecule rows")
    if run_summary["run_id"].duplicated().any():
        errors.append("Run failure summary contains duplicate run IDs")
    if meta_dataset["run_id"].duplicated().any():
        errors.append("Meta-dataset contains duplicate run IDs")
    expected_runs = set(model_results["run_id"])
    if set(run_summary["run_id"]) != expected_runs:
        errors.append("Run failure summary does not cover every model result")
    if set(meta_dataset["run_id"]) != expected_runs:
        errors.append("Meta-dataset does not cover every model result")
    predictor_columns = [column for column in meta_dataset if column.startswith("x_")]
    leaked_names = [column for column in predictor_columns if "test_log_s" in column]
    if leaked_names:
        errors.append(f"Predictor columns contain test-label summaries: {leaked_names}")
    return errors
