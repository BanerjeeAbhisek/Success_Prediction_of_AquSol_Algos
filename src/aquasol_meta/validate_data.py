from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DATASET_SPECS


def _python_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def validate_master(observations: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Return a JSON-ready audit report and a list of blocking validation errors."""
    errors: list[str] = []
    expected_total = sum(spec.expected_rows for spec in DATASET_SPECS)

    if len(observations) != expected_total:
        errors.append(
            f"Master row count is {len(observations):,}; expected {expected_total:,} source rows."
        )
    if observations["observation_id"].duplicated().any():
        errors.append("observation_id is not unique.")

    counts = observations["source_dataset"].value_counts().to_dict()
    for spec in DATASET_SPECS:
        actual = int(counts.get(spec.name, 0))
        if actual != spec.expected_rows:
            errors.append(
                f"{spec.name} contributes {actual:,} rows; expected {spec.expected_rows:,}."
            )

    finite_target = observations["log_s"].notna() & np.isfinite(observations["log_s"])
    valid_structure = observations["structure_status"].eq("ok")
    observations_by_source: dict[str, dict[str, Any]] = {}
    for source, frame in observations.groupby("source_dataset", sort=True):
        source_finite = frame["log_s"].notna() & np.isfinite(frame["log_s"])
        source_valid_structure = frame["structure_status"].eq("ok")
        target = frame.loc[source_finite, "log_s"]
        observations_by_source[str(source)] = {
            "rows": int(len(frame)),
            "finite_log_s": int(source_finite.sum()),
            "missing_smiles": int(frame["original_smiles"].isna().sum()),
            "valid_structures": int(source_valid_structure.sum()),
            "invalid_or_missing_structures": int((~source_valid_structure).sum()),
            "unique_parent_molecules": int(
                frame.loc[source_valid_structure, "molecule_id"].nunique(dropna=True)
            ),
            "multifragment_observations": int(
                frame["has_multiple_fragments"].fillna(False).astype(bool).sum()
            ),
            "log_s_min": _python_value(target.min()) if not target.empty else None,
            "log_s_median": _python_value(target.median()) if not target.empty else None,
            "log_s_max": _python_value(target.max()) if not target.empty else None,
        }

    report: dict[str, Any] = {
        "status": "pass" if not errors else "fail",
        "total_observations": int(len(observations)),
        "finite_log_s": int(finite_target.sum()),
        "valid_structures": int(valid_structure.sum()),
        "model_eligible_observations": int((finite_target & valid_structure).sum()),
        "unique_parent_molecules": int(
            observations.loc[valid_structure, "molecule_id"].nunique(dropna=True)
        ),
        "duplicate_observation_ids": int(observations["observation_id"].duplicated().sum()),
        "observations_by_source": observations_by_source,
        "blocking_errors": errors,
    }
    return report, errors
