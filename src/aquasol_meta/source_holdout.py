from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .splits import (
    DEFAULT_SPLIT_SEEDS,
    _fingerprints_by_id,
    _maximum_train_similarity,
    _safe_train_test_split,
    _target_bins,
)
from .targets import build_modeling_targets


def source_slug(source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(source).lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot create a task name from source {source!r}")
    return slug


def _scaffold_groups(targets: pd.DataFrame) -> pd.Series:
    scaffold = targets["murcko_scaffold"].fillna("").astype(str).str.strip()
    missing = scaffold.isin({"", "[NO_SCAFFOLD]"})
    output = scaffold.copy()
    output.loc[missing] = "ACYCLIC_SINGLETON::" + targets.loc[missing, "molecule_id"].astype(str)
    return output


def _source_targets(
    observations: pd.DataFrame,
    heldout_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    heldout_rows = observations.loc[observations["source_dataset"].eq(heldout_source)].copy()
    heldout_molecule_ids = set(heldout_rows["molecule_id"].dropna().astype(str))
    if not heldout_molecule_ids:
        raise ValueError(f"{heldout_source} contains no standardized molecules")

    nonheldout_rows = observations.loc[
        observations["source_dataset"].ne(heldout_source)
        & ~observations["molecule_id"].astype("string").isin(heldout_molecule_ids)
    ].copy()
    test_all = build_modeling_targets(heldout_rows)
    fit_all = build_modeling_targets(nonheldout_rows)
    test = test_all.loc[test_all["include_primary_benchmark"].astype(bool)].copy()
    fit = fit_all.loc[fit_all["include_primary_benchmark"].astype(bool)].copy()
    if test.empty or fit.empty:
        raise ValueError(f"{heldout_source} produces an empty fit or test target table")
    if set(test["molecule_id"]).intersection(fit["molecule_id"]):
        raise RuntimeError(f"{heldout_source} has molecule leakage between fit and test targets")

    test["target_origin"] = "heldout_source_only"
    fit["target_origin"] = "nonheldout_sources_only"
    test["heldout_source"] = heldout_source
    fit["heldout_source"] = heldout_source
    audit = {
        "heldout_source_rows": int(len(heldout_rows)),
        "heldout_standardized_molecules": int(len(heldout_molecule_ids)),
        "test_targets_before_primary_filter": int(len(test_all)),
        "test_targets": int(len(test)),
        "test_targets_excluded_for_conflict": int(test_all["measurement_conflict"].sum()),
        "test_targets_excluded_outside_review_range": int(
            test_all["outside_target_review_range"].sum()
        ),
        "fit_targets_before_primary_filter": int(len(fit_all)),
        "fit_targets": int(len(fit)),
    }
    return fit, test, audit


def build_source_holdout_tasks(
    observations: pd.DataFrame,
    morgan: pd.DataFrame,
    sources: Iterable[str] | None = None,
    seeds: Iterable[int] = DEFAULT_SPLIT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build leakage-safe source-holdout tasks with source-specific target reconstruction."""
    required = {"source_dataset", "molecule_id", "model_eligible", "log_s"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Master observations are missing columns: {sorted(missing)}")
    available_sources = sorted(
        str(value) for value in observations["source_dataset"].dropna().unique()
    )
    selected_sources = tuple(available_sources if sources is None else sources)
    unknown = sorted(set(selected_sources).difference(available_sources))
    if unknown:
        raise ValueError(f"Requested source datasets do not exist: {unknown}")
    if not selected_sources:
        raise ValueError("At least one held-out source is required")

    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("At least one split seed is required")
    fingerprints = _fingerprints_by_id(morgan)
    target_columns = [
        "molecule_id",
        "log_s_target",
        "target_reliability",
        "n_observations",
        "n_source_datasets",
        "source_datasets",
        "log_s_range",
        "include_primary_benchmark",
        "target_origin",
    ]
    task_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    source_audit: dict[str, dict[str, int]] = {}

    for heldout_source in selected_sources:
        fit_targets, test_targets, audit = _source_targets(observations, heldout_source)
        source_audit[heldout_source] = audit
        fit_targets["scaffold_group"] = _scaffold_groups(fit_targets)
        test_targets["scaffold_group"] = _scaffold_groups(test_targets)
        missing_fingerprints = sorted(
            set(fit_targets["molecule_id"]).union(test_targets["molecule_id"])
            .difference(fingerprints)
        )
        if missing_fingerprints:
            raise ValueError(
                f"{heldout_source} is missing Morgan fingerprints for "
                f"{len(missing_fingerprints)} target molecules"
            )

        strategy = f"source_holdout_{source_slug(heldout_source)}"
        for repeat, seed in enumerate(seeds, start=1):
            indices = np.arange(len(fit_targets))
            bins = _target_bins(fit_targets["log_s_target"])
            train_index, validation_index = _safe_train_test_split(
                indices,
                train_size=0.9,
                seed=seed,
                stratify=bins,
            )
            train_ids = set(fit_targets.iloc[train_index]["molecule_id"].astype(str))
            validation_ids = set(
                fit_targets.iloc[validation_index]["molecule_id"].astype(str)
            )
            test_ids = set(test_targets["molecule_id"].astype(str))
            if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
                raise RuntimeError(f"{heldout_source} repeat {repeat} has partition leakage")

            targets = pd.concat([fit_targets, test_targets], ignore_index=True)
            partition_by_id = {
                **dict.fromkeys(train_ids, "train"),
                **dict.fromkeys(validation_ids, "validation"),
                **dict.fromkeys(test_ids, "test"),
            }
            task = targets[target_columns + ["scaffold_group"]].copy()
            task["task_id"] = f"{strategy}_r{repeat}"
            task["split_strategy"] = strategy
            task["heldout_source"] = heldout_source
            task["repeat"] = repeat
            task["seed"] = seed
            task["partition"] = task["molecule_id"].astype(str).map(partition_by_id)
            task["maximum_train_similarity"] = np.nan
            evaluation_ids = sorted(validation_ids | test_ids)
            similarity = _maximum_train_similarity(
                evaluation_ids, sorted(train_ids), fingerprints
            )
            task["maximum_train_similarity"] = task["molecule_id"].map(similarity)
            ordered_columns = [
                "task_id",
                "split_strategy",
                "heldout_source",
                "repeat",
                "seed",
                "molecule_id",
                "partition",
                "scaffold_group",
                "maximum_train_similarity",
                *target_columns[1:],
            ]
            task_frames.append(task[ordered_columns])

            test_similarity = task.loc[
                task["partition"].eq("test"), "maximum_train_similarity"
            ]
            summary_rows.append(
                {
                    "task_id": f"{strategy}_r{repeat}",
                    "heldout_source": heldout_source,
                    "repeat": repeat,
                    "seed": seed,
                    "train_molecules": len(train_ids),
                    "validation_molecules": len(validation_ids),
                    "test_molecules": len(test_ids),
                    "fit_target_molecules": len(fit_targets),
                    "test_log_s_mean": float(test_targets["log_s_target"].mean()),
                    "test_log_s_std": float(test_targets["log_s_target"].std(ddof=1)),
                    "test_similarity_mean": float(test_similarity.mean()),
                    "test_similarity_median": float(test_similarity.median()),
                    "test_similarity_p10": float(test_similarity.quantile(0.10)),
                    "test_similarity_min": float(test_similarity.min()),
                }
            )

    tasks = pd.concat(task_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    errors = validate_source_holdout_tasks(tasks, expected_repeats=len(seeds))
    report = {
        "status": "fail" if errors else "pass",
        "blocking_errors": errors,
        "heldout_sources": list(selected_sources),
        "source_count": len(selected_sources),
        "repeats": len(seeds),
        "seeds": list(seeds),
        "task_count": int(tasks["task_id"].nunique()),
        "manifest_rows": int(len(tasks)),
        "source_target_counts": source_audit,
        "test_target_policy": (
            "Test targets are medians of model-eligible observations from the held-out source "
            "only, followed by the established conflict and review-range filters."
        ),
        "fit_target_policy": (
            "Fit targets are rebuilt after removing the held-out source and every molecule that "
            "appears in that source. No held-out-source molecule can enter training or validation."
        ),
        "validation_policy": (
            "The non-held-out target pool is split 90/10 into train and validation using logS "
            "quantile stratification and five fixed seeds. The held-out-source test set remains "
            "fixed across repeats."
        ),
        "similarity_policy": (
            "Maximum Morgan radius-2 Tanimoto similarity is computed against the training "
            "partition for every validation and test molecule."
        ),
        "known_dependence": (
            "Some source datasets share many molecules. Cross-source molecule leakage is blocked "
            "within each task, but source-holdout tasks are still correlated and do not replace "
            "evaluation on genuinely external datasets."
        ),
    }
    return tasks, summary, report


def validate_source_holdout_tasks(
    tasks: pd.DataFrame,
    expected_repeats: int,
) -> list[str]:
    errors: list[str] = []
    if tasks.duplicated(["task_id", "molecule_id"]).any():
        errors.append("A task contains duplicate molecule IDs")
    for heldout_source, source_tasks in tasks.groupby("heldout_source", sort=True):
        if source_tasks["repeat"].nunique() != expected_repeats:
            errors.append(f"{heldout_source} does not contain {expected_repeats} repeats")
        test_sets: list[frozenset[str]] = []
        for task_id, task in source_tasks.groupby("task_id", sort=True):
            partitions = set(task["partition"])
            if partitions != {"train", "validation", "test"}:
                errors.append(f"{task_id} does not contain all three partitions")
            train_ids = set(task.loc[task["partition"].eq("train"), "molecule_id"])
            validation_ids = set(
                task.loc[task["partition"].eq("validation"), "molecule_id"]
            )
            test_ids = set(task.loc[task["partition"].eq("test"), "molecule_id"])
            if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
                errors.append(f"{task_id} leaks molecules between partitions")
            test_sets.append(frozenset(str(value) for value in test_ids))
            if not task.loc[task["partition"].eq("test"), "target_origin"].eq(
                "heldout_source_only"
            ).all():
                errors.append(f"{task_id} has a test target not derived from the held-out source")
            fit = task["partition"].isin({"train", "validation"})
            if not task.loc[fit, "target_origin"].eq("nonheldout_sources_only").all():
                errors.append(f"{task_id} has a fit target containing held-out-source data")
            evaluation = task["partition"].isin({"validation", "test"})
            if task.loc[evaluation, "maximum_train_similarity"].isna().any():
                errors.append(f"{task_id} has missing validation/test similarities")
            if task.loc[~evaluation, "maximum_train_similarity"].notna().any():
                errors.append(f"{task_id} stores train-to-self similarities")
        if len(set(test_sets)) != 1:
            errors.append(f"{heldout_source} test molecules change across repeats")
    return errors
