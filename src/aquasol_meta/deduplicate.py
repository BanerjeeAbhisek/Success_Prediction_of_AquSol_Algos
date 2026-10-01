from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _join_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna()}))


def build_molecule_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize measurements by standardized parent molecule.

    This table is descriptive. It does not decide which measurements should be pooled for model
    fitting because pH, temperature, endpoint definition, and source can make repeated values
    scientifically non-equivalent.
    """
    valid = observations.loc[
        observations["molecule_id"].notna()
        & observations["log_s"].notna()
        & np.isfinite(observations["log_s"])
        & observations["structure_status"].eq("ok")
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby("molecule_id", sort=True, dropna=False)
    summary = grouped.agg(
        canonical_smiles_parent=("canonical_smiles_parent", "first"),
        murcko_scaffold=("murcko_scaffold", "first"),
        n_observations=("observation_id", "size"),
        n_source_datasets=("source_dataset", "nunique"),
        source_datasets=("source_dataset", _join_unique),
        log_s_mean=("log_s", "mean"),
        log_s_median=("log_s", "median"),
        log_s_sd=("log_s", "std"),
        log_s_min=("log_s", "min"),
        log_s_max=("log_s", "max"),
        any_multifragment=("has_multiple_fragments", "max"),
    ).reset_index()

    summary["log_s_range"] = summary["log_s_max"] - summary["log_s_min"]
    summary["measurement_conflict_over_1_log_unit"] = summary["log_s_range"] > 1.0
    return summary.sort_values("molecule_id", kind="stable").reset_index(drop=True)


def build_source_overlap(observations: pd.DataFrame) -> pd.DataFrame:
    """Count standardized molecules shared by each pair of source datasets."""
    valid = observations.loc[
        observations["molecule_id"].notna() & observations["structure_status"].eq("ok"),
        ["source_dataset", "molecule_id"],
    ].drop_duplicates()

    sets = {
        name: set(group["molecule_id"])
        for name, group in valid.groupby("source_dataset", sort=True)
    }
    rows: list[dict[str, object]] = []
    for left, right in combinations(sorted(sets), 2):
        overlap = sets[left] & sets[right]
        smaller = min(len(sets[left]), len(sets[right]))
        rows.append(
            {
                "source_left": left,
                "source_right": right,
                "unique_molecules_left": len(sets[left]),
                "unique_molecules_right": len(sets[right]),
                "shared_molecules": len(overlap),
                "percent_of_smaller_source": 100 * len(overlap) / smaller if smaller else np.nan,
            }
        )
    return pd.DataFrame(rows)
