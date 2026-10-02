from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.model_selection import train_test_split

DEFAULT_SPLIT_SEEDS = (13, 37, 73, 101, 137)
PARTITIONS = ("train", "validation", "test")
STRATEGIES = ("random", "scaffold", "low_similarity")


def _primary_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {
        "molecule_id",
        "log_s_target",
        "murcko_scaffold",
        "include_primary_benchmark",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing required columns: {sorted(missing)}")

    primary = targets.loc[targets["include_primary_benchmark"].astype(bool)].copy()
    if primary.empty:
        raise ValueError("The primary benchmark contains no molecules")
    if primary["molecule_id"].duplicated().any():
        raise ValueError("Primary benchmark molecule IDs must be unique")

    primary["molecule_id"] = primary["molecule_id"].astype(str)
    primary["log_s_target"] = pd.to_numeric(primary["log_s_target"], errors="raise")
    scaffold = primary["murcko_scaffold"].fillna("").astype(str).str.strip()
    no_scaffold = scaffold.isin({"", "[NO_SCAFFOLD]"})
    primary["scaffold_group"] = scaffold
    primary.loc[no_scaffold, "scaffold_group"] = (
        "ACYCLIC_SINGLETON::" + primary.loc[no_scaffold, "molecule_id"]
    )
    return primary.sort_values("molecule_id", kind="stable").reset_index(drop=True)


def _fingerprints_by_id(morgan: pd.DataFrame) -> dict[str, Any]:
    if "molecule_id" not in morgan.columns:
        raise ValueError("Morgan table must contain molecule_id")
    bit_columns = [column for column in morgan.columns if column.startswith("morgan_")]
    if not bit_columns:
        raise ValueError("Morgan table contains no morgan_* bit columns")
    if morgan["molecule_id"].duplicated().any():
        raise ValueError("Morgan molecule IDs must be unique")

    matrix = morgan[bit_columns].to_numpy(dtype=np.uint8, copy=False)
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("Morgan fingerprints must contain only 0 and 1")

    fingerprints: dict[str, Any] = {}
    for molecule_id, row in zip(morgan["molecule_id"].astype(str), matrix, strict=True):
        fingerprint = DataStructs.ExplicitBitVect(len(bit_columns))
        fingerprint.SetBitsFromList(np.flatnonzero(row).astype(int).tolist())
        fingerprints[molecule_id] = fingerprint
    return fingerprints


def _target_bins(values: pd.Series, max_bins: int = 10) -> pd.Series | None:
    if len(values) < 20:
        return None
    bins = min(max_bins, max(2, len(values) // 20))
    result = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    if result.nunique() < 2 or result.value_counts().min() < 2:
        return None
    return result


def _safe_train_test_split(
    indices: np.ndarray,
    train_size: float,
    seed: int,
    stratify: pd.Series | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        left, right = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        left, right = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
        )
    return np.asarray(left), np.asarray(right)


def _random_partitions(primary: pd.DataFrame, seed: int) -> dict[str, set[str]]:
    indices = np.arange(len(primary))
    bins = _target_bins(primary["log_s_target"])
    train_index, temporary_index = _safe_train_test_split(
        indices,
        train_size=0.8,
        seed=seed,
        stratify=bins,
    )
    temporary_bins = bins.iloc[temporary_index] if bins is not None else None
    validation_index, test_index = _safe_train_test_split(
        temporary_index,
        train_size=0.5,
        seed=seed + 1,
        stratify=temporary_bins,
    )
    ids = primary["molecule_id"]
    return {
        "train": set(ids.iloc[train_index]),
        "validation": set(ids.iloc[validation_index]),
        "test": set(ids.iloc[test_index]),
    }


def _scaffold_partitions(primary: pd.DataFrame, seed: int) -> dict[str, set[str]]:
    rng = np.random.default_rng(seed)
    groups = (
        primary.groupby("scaffold_group", sort=False)["molecule_id"]
        .agg(list)
        .rename("molecule_ids")
        .reset_index()
    )
    groups["size"] = groups["molecule_ids"].str.len()
    groups["priority"] = groups["size"] * rng.uniform(0.9, 1.1, size=len(groups))
    groups = groups.sort_values(["priority", "size"], ascending=False, kind="stable")

    target_counts = {
        "train": 0.8 * len(primary),
        "validation": 0.1 * len(primary),
        "test": 0.1 * len(primary),
    }
    loads = dict.fromkeys(PARTITIONS, 0)
    assignments: dict[str, set[str]] = {partition: set() for partition in PARTITIONS}

    for molecule_ids in groups["molecule_ids"]:
        fill_ratios = {
            partition: loads[partition] / target_counts[partition]
            for partition in PARTITIONS
        }
        minimum = min(fill_ratios.values())
        candidates = [
            partition
            for partition in PARTITIONS
            if np.isclose(fill_ratios[partition], minimum)
        ]
        partition = max(
            candidates,
            key=lambda value: target_counts[value] - loads[value],
        )
        assignments[partition].update(str(value) for value in molecule_ids)
        loads[partition] += len(molecule_ids)
    return assignments


def _maximum_train_similarity(
    query_ids: Iterable[str],
    train_ids: Iterable[str],
    fingerprints: dict[str, Any],
) -> dict[str, float]:
    train_fingerprints = [fingerprints[molecule_id] for molecule_id in train_ids]
    if not train_fingerprints:
        raise ValueError("A split contains no training fingerprints")
    similarities: dict[str, float] = {}
    for molecule_id in query_ids:
        values = DataStructs.BulkTanimotoSimilarity(
            fingerprints[molecule_id], train_fingerprints
        )
        similarities[molecule_id] = float(max(values)) if values else 0.0
    return similarities


def _low_similarity_partitions(
    primary: pd.DataFrame,
    fingerprints: dict[str, Any],
    seed: int,
) -> dict[str, set[str]]:
    indices = np.arange(len(primary))
    bins = _target_bins(primary["log_s_target"])
    provisional_train_index, candidate_index = _safe_train_test_split(
        indices,
        train_size=0.7,
        seed=seed,
        stratify=bins,
    )
    ids = primary["molecule_id"]
    provisional_train = set(ids.iloc[provisional_train_index])
    candidates = list(ids.iloc[candidate_index])
    similarity = _maximum_train_similarity(candidates, provisional_train, fingerprints)

    rng = np.random.default_rng(seed)
    tie_break = {molecule_id: float(rng.random()) for molecule_id in candidates}
    ordered = sorted(candidates, key=lambda value: (similarity[value], tie_break[value]))
    n_test = round(0.1 * len(primary))
    n_validation = round(0.1 * len(primary))
    test = set(ordered[:n_test])
    validation = set(ordered[n_test : n_test + n_validation])
    train = provisional_train.union(ordered[n_test + n_validation :])
    return {"train": train, "validation": validation, "test": test}


def _assignment_rows(
    primary: pd.DataFrame,
    partitions: dict[str, set[str]],
    fingerprints: dict[str, Any],
    strategy: str,
    repeat: int,
    seed: int,
) -> pd.DataFrame:
    partition_by_id = {
        molecule_id: partition
        for partition, molecule_ids in partitions.items()
        for molecule_id in molecule_ids
    }
    frame = primary[["molecule_id", "scaffold_group"]].copy()
    frame["split_strategy"] = strategy
    frame["repeat"] = repeat
    frame["seed"] = seed
    frame["partition"] = frame["molecule_id"].map(partition_by_id)
    frame["maximum_train_similarity"] = np.nan

    train_ids = sorted(partitions["train"])
    evaluation_ids = sorted(partitions["validation"].union(partitions["test"]))
    similarities = _maximum_train_similarity(evaluation_ids, train_ids, fingerprints)
    frame["maximum_train_similarity"] = frame["molecule_id"].map(similarities)
    return frame[
        [
            "molecule_id",
            "split_strategy",
            "repeat",
            "seed",
            "partition",
            "scaffold_group",
            "maximum_train_similarity",
        ]
    ]


def build_split_assignments(
    targets: pd.DataFrame,
    morgan: pd.DataFrame,
    seeds: Iterable[int] = DEFAULT_SPLIT_SEEDS,
) -> pd.DataFrame:
    primary = _primary_targets(targets)
    fingerprints = _fingerprints_by_id(morgan)
    missing_fingerprints = sorted(set(primary["molecule_id"]).difference(fingerprints))
    if missing_fingerprints:
        raise ValueError(
            f"Morgan fingerprints are missing for {len(missing_fingerprints)} primary molecules"
        )

    frames: list[pd.DataFrame] = []
    for repeat, seed in enumerate(tuple(seeds), start=1):
        strategy_partitions = {
            "random": _random_partitions(primary, seed),
            "scaffold": _scaffold_partitions(primary, seed),
            "low_similarity": _low_similarity_partitions(primary, fingerprints, seed),
        }
        for strategy, partitions in strategy_partitions.items():
            frames.append(
                _assignment_rows(
                    primary,
                    partitions=partitions,
                    fingerprints=fingerprints,
                    strategy=strategy,
                    repeat=repeat,
                    seed=seed,
                )
            )
    return pd.concat(frames, ignore_index=True)


def validate_split_assignments(
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
    expected_repeats: int,
) -> list[str]:
    errors: list[str] = []
    primary = _primary_targets(targets)
    expected_ids = set(primary["molecule_id"])

    if assignments.duplicated(["molecule_id", "split_strategy", "repeat"]).any():
        errors.append("A molecule appears more than once within a strategy/repeat")
    if not set(assignments["partition"]).issubset(PARTITIONS):
        errors.append("Split assignments contain an unknown partition")
    if not set(assignments["split_strategy"]).issubset(STRATEGIES):
        errors.append("Split assignments contain an unknown strategy")

    for strategy in STRATEGIES:
        strategy_frame = assignments.loc[assignments["split_strategy"].eq(strategy)]
        if strategy_frame["repeat"].nunique() != expected_repeats:
            errors.append(f"{strategy} does not contain {expected_repeats} repeats")
        for repeat, frame in strategy_frame.groupby("repeat"):
            if set(frame["molecule_id"]) != expected_ids:
                errors.append(f"{strategy} repeat {repeat} does not cover every primary molecule")
            if set(frame["partition"]) != set(PARTITIONS):
                errors.append(f"{strategy} repeat {repeat} lacks a required partition")
            evaluation = frame["partition"].isin({"validation", "test"})
            if frame.loc[evaluation, "maximum_train_similarity"].isna().any():
                errors.append(f"{strategy} repeat {repeat} has missing evaluation similarities")
            if frame.loc[~evaluation, "maximum_train_similarity"].notna().any():
                errors.append(f"{strategy} repeat {repeat} has train-to-self similarities")

            if strategy == "scaffold":
                scaffold_partitions = frame.groupby("scaffold_group")["partition"].nunique()
                if scaffold_partitions.gt(1).any():
                    errors.append(f"scaffold repeat {repeat} leaks scaffolds across partitions")
    return errors


def build_split_quality_summary(
    assignments: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    target_lookup = targets.set_index("molecule_id")["log_s_target"]
    joined = assignments.copy()
    joined["log_s_target"] = joined["molecule_id"].map(target_lookup)
    summary = (
        joined.groupby(["split_strategy", "repeat", "seed", "partition"], sort=True)
        .agg(
            molecules=("molecule_id", "size"),
            scaffolds=("scaffold_group", "nunique"),
            log_s_mean=("log_s_target", "mean"),
            log_s_median=("log_s_target", "median"),
            log_s_min=("log_s_target", "min"),
            log_s_max=("log_s_target", "max"),
            mean_maximum_train_similarity=("maximum_train_similarity", "mean"),
            median_maximum_train_similarity=("maximum_train_similarity", "median"),
            minimum_maximum_train_similarity=("maximum_train_similarity", "min"),
            maximum_maximum_train_similarity=("maximum_train_similarity", "max"),
        )
        .reset_index()
    )
    totals = summary.groupby(["split_strategy", "repeat"])["molecules"].transform("sum")
    summary["fraction_of_molecules"] = summary["molecules"] / totals
    return summary


def build_split_report(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    errors: list[str],
) -> dict[str, Any]:
    test_summary = summary.loc[summary["partition"].eq("test")]
    similarity_by_strategy = (
        test_summary.groupby("split_strategy")["median_maximum_train_similarity"]
        .agg(["mean", "min", "max"])
        .to_dict(orient="index")
    )
    return {
        "status": "fail" if errors else "pass",
        "blocking_errors": errors,
        "molecules_per_strategy_repeat": int(
            assignments.groupby(["split_strategy", "repeat"]).size().iloc[0]
        ),
        "strategies": list(STRATEGIES),
        "repeats": int(assignments["repeat"].nunique()),
        "seeds": sorted(int(value) for value in assignments["seed"].unique()),
        "target_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "random_policy": "logS-quantile-stratified molecule split",
        "scaffold_policy": (
            "Murcko scaffold groups remain intact; molecules without a ring scaffold are treated "
            "as acyclic singletons to avoid one oversized empty-scaffold group."
        ),
        "low_similarity_policy": (
            "A provisional 70% training set is sampled; the lowest-similarity 10% and next-lowest "
            "10% become test and validation, and the remaining 10% joins training. Similarity is "
            "Morgan radius-2 Tanimoto similarity and is recomputed against the final training set."
        ),
        "similarity_scope": (
            "maximum_train_similarity is populated for validation/test molecules and left missing "
            "for training molecules."
        ),
        "mean_of_repeat_test_median_similarity": {
            strategy: {metric: float(value) for metric, value in metrics.items()}
            for strategy, metrics in similarity_by_strategy.items()
        },
    }
