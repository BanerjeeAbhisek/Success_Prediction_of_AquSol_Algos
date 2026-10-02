from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _join_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna()}))


def _quantile_25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def _quantile_75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def _median_absolute_deviation(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def build_modeling_targets(
    observations: pd.DataFrame,
    consistent_range_max: float = 0.5,
    conflict_range_min: float = 1.0,
    review_log_s_min: float = -15.0,
    review_log_s_max: float = 2.5,
) -> pd.DataFrame:
    """Create one auditable target row per model-eligible standardized molecule.

    The median is the default target because it is robust to outlying repeated measurements. The
    original observations remain in the master table. Inclusion flags expose alternative benchmark
    populations without deleting disputed molecules.
    """
    if consistent_range_max < 0:
        raise ValueError("consistent_range_max must be non-negative")
    if conflict_range_min <= consistent_range_max:
        raise ValueError("conflict_range_min must exceed consistent_range_max")
    if review_log_s_min >= review_log_s_max:
        raise ValueError("review_log_s_min must be less than review_log_s_max")

    eligible = observations.loc[
        observations["model_eligible"].fillna(False).astype(bool)
        & observations["molecule_id"].notna()
    ].copy()
    eligible["log_s"] = pd.to_numeric(eligible["log_s"], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible["log_s"])].copy()

    if eligible.empty:
        raise ValueError("No model-eligible observations were available to construct targets")

    grouped = eligible.groupby("molecule_id", sort=True, dropna=False)
    targets = grouped.agg(
        canonical_smiles_parent=("canonical_smiles_parent", "first"),
        murcko_scaffold=("murcko_scaffold", "first"),
        n_observations=("observation_id", "size"),
        n_source_datasets=("source_dataset", "nunique"),
        source_datasets=("source_dataset", _join_unique),
        log_s_target=("log_s", "median"),
        log_s_mean=("log_s", "mean"),
        log_s_sd=("log_s", "std"),
        log_s_min=("log_s", "min"),
        log_s_q25=("log_s", _quantile_25),
        log_s_q75=("log_s", _quantile_75),
        log_s_max=("log_s", "max"),
        log_s_mad=("log_s", _median_absolute_deviation),
        observations_with_temperature=("temperature", "count"),
        observations_with_ph=("ph", "count"),
        any_multifragment=("has_multiple_fragments", "max"),
    ).reset_index()

    targets["log_s_range"] = targets["log_s_max"] - targets["log_s_min"]
    targets["log_s_iqr"] = targets["log_s_q75"] - targets["log_s_q25"]
    targets["measurement_conflict"] = targets["log_s_range"] > conflict_range_min
    targets["outside_target_review_range"] = (
        (targets["log_s_target"] < review_log_s_min)
        | (targets["log_s_target"] > review_log_s_max)
    )

    targets["target_reliability"] = np.select(
        [
            targets["n_observations"].eq(1),
            targets["log_s_range"].le(consistent_range_max),
            targets["log_s_range"].le(conflict_range_min),
        ],
        ["single_observation", "consistent_replicates", "moderate_disagreement"],
        default="high_conflict",
    )

    targets["include_all_data_benchmark"] = True
    targets["include_primary_benchmark"] = ~(
        targets["measurement_conflict"] | targets["outside_target_review_range"]
    )
    targets["include_conflict_stress_test"] = targets["measurement_conflict"]

    both = targets["measurement_conflict"] & targets["outside_target_review_range"]
    targets["primary_exclusion_reason"] = np.select(
        [both, targets["measurement_conflict"], targets["outside_target_review_range"]],
        [
            "high_conflict_and_outside_review_range",
            "high_measurement_conflict",
            "outside_target_review_range",
        ],
        default="included",
    )
    targets["target_resolution_method"] = "median_of_all_model_eligible_observations"
    targets["target_unit"] = "log10(mol/L)"
    targets["consistent_range_max"] = consistent_range_max
    targets["conflict_range_min"] = conflict_range_min
    targets["review_log_s_min"] = review_log_s_min
    targets["review_log_s_max"] = review_log_s_max

    return targets.sort_values("molecule_id", kind="stable").reset_index(drop=True)


def validate_modeling_targets(
    targets: pd.DataFrame,
    expected_observations: int,
) -> list[str]:
    errors: list[str] = []
    if targets["molecule_id"].duplicated().any():
        errors.append("molecule_id is not unique in the modelling-target table")
    if not np.isfinite(targets["log_s_target"]).all():
        errors.append("log_s_target contains missing or non-finite values")
    represented = int(targets["n_observations"].sum())
    if represented != expected_observations:
        errors.append(
            f"Target table represents {represented:,} observations; "
            f"expected {expected_observations:,}"
        )
    if not targets["include_all_data_benchmark"].all():
        errors.append("The all-data benchmark does not retain every target molecule")
    return errors


def build_modeling_target_report(
    targets: pd.DataFrame,
    represented_observations: int,
) -> dict[str, Any]:
    reliability_counts = targets["target_reliability"].value_counts().sort_index()
    exclusion_counts = targets["primary_exclusion_reason"].value_counts().sort_index()
    return {
        "status": "pass",
        "total_molecules": int(len(targets)),
        "represented_model_eligible_observations": int(represented_observations),
        "primary_benchmark_molecules": int(targets["include_primary_benchmark"].sum()),
        "all_data_benchmark_molecules": int(targets["include_all_data_benchmark"].sum()),
        "conflict_stress_test_molecules": int(targets["include_conflict_stress_test"].sum()),
        "outside_target_review_range": int(targets["outside_target_review_range"].sum()),
        "target_reliability_counts": {
            str(key): int(value) for key, value in reliability_counts.items()
        },
        "primary_exclusion_reason_counts": {
            str(key): int(value) for key, value in exclusion_counts.items()
        },
        "thresholds": {
            "consistent_range_max": float(targets["consistent_range_max"].iloc[0]),
            "conflict_range_min": float(targets["conflict_range_min"].iloc[0]),
            "review_log_s_min": float(targets["review_log_s_min"].iloc[0]),
            "review_log_s_max": float(targets["review_log_s_max"].iloc[0]),
        },
        "target_resolution_method": str(targets["target_resolution_method"].iloc[0]),
        "target_unit": str(targets["target_unit"].iloc[0]),
    }


def build_reliability_summary(targets: pd.DataFrame) -> pd.DataFrame:
    summary = (
        targets.groupby("target_reliability", sort=True)
        .agg(
            molecules=("molecule_id", "size"),
            primary_benchmark_molecules=("include_primary_benchmark", "sum"),
            median_observations=("n_observations", "median"),
            median_log_s_range=("log_s_range", "median"),
            median_log_s_target=("log_s_target", "median"),
        )
        .reset_index()
    )
    summary["percent_of_molecules"] = 100 * summary["molecules"] / len(targets)
    return summary
