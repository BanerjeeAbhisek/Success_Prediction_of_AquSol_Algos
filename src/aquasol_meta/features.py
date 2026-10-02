from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator


@dataclass(frozen=True)
class FeatureTables:
    descriptors: pd.DataFrame
    morgan: pd.DataFrame
    maccs: pd.DataFrame
    failures: list[dict[str, str]]
    descriptor_failures: list[dict[str, str]]
    descriptor_names: list[str]


def _ordered_molecules(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"molecule_id", "canonical_smiles_parent"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing required columns: {sorted(missing)}")

    molecules = targets[["molecule_id", "canonical_smiles_parent"]].copy()
    if molecules["molecule_id"].isna().any() or molecules["canonical_smiles_parent"].isna().any():
        raise ValueError("Molecule IDs and canonical parent SMILES must not be missing")
    if molecules["molecule_id"].duplicated().any():
        raise ValueError("molecule_id must be unique before feature generation")
    return molecules.sort_values("molecule_id", kind="stable").reset_index(drop=True)


def _bit_vector_to_array(bit_vector: Any, n_bits: int) -> np.ndarray:
    array = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def build_feature_tables(
    targets: pd.DataFrame,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    include_chirality: bool = True,
) -> FeatureTables:
    """Generate label-free molecular representations for every target molecule."""
    if morgan_radius < 0:
        raise ValueError("morgan_radius must be non-negative")
    if morgan_n_bits <= 0:
        raise ValueError("morgan_n_bits must be positive")

    molecules = _ordered_molecules(targets)
    descriptor_functions = list(Descriptors.descList)
    descriptor_names = [name for name, _ in descriptor_functions]
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius,
        fpSize=morgan_n_bits,
        includeChirality=include_chirality,
    )

    molecule_ids: list[str] = []
    canonical_smiles: list[str] = []
    descriptor_rows: list[tuple[float, ...]] = []
    morgan_rows: list[np.ndarray] = []
    maccs_rows: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    descriptor_failures: list[dict[str, str]] = []

    for row in molecules.itertuples(index=False):
        molecule_id = str(row.molecule_id)
        smiles = str(row.canonical_smiles_parent)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            failures.append(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles_parent": smiles,
                    "reason": "RDKit could not parse canonical_smiles_parent",
                }
            )
            continue

        try:
            morgan = morgan_generator.GetFingerprint(molecule)
            maccs = MACCSkeys.GenMACCSKeys(molecule)
        except Exception as exc:  # pragma: no cover - defensive path for unusual chemistry
            failures.append(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles_parent": smiles,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        descriptor_values: list[float] = []
        for descriptor_name, descriptor_function in descriptor_functions:
            try:
                value = float(descriptor_function(molecule))
                descriptor_values.append(value if np.isfinite(value) else np.nan)
            except Exception as exc:  # individual descriptors may not support unusual molecules
                descriptor_values.append(np.nan)
                descriptor_failures.append(
                    {
                        "molecule_id": molecule_id,
                        "descriptor": descriptor_name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

        molecule_ids.append(molecule_id)
        canonical_smiles.append(smiles)
        descriptor_rows.append(tuple(descriptor_values))
        morgan_rows.append(_bit_vector_to_array(morgan, morgan_n_bits))
        maccs_rows.append(_bit_vector_to_array(maccs, int(maccs.GetNumBits())))

    descriptors = pd.DataFrame(descriptor_rows, columns=descriptor_names, dtype="float64")
    descriptors.insert(0, "canonical_smiles_parent", canonical_smiles)
    descriptors.insert(0, "molecule_id", molecule_ids)
    numeric_descriptor_columns = descriptors.columns[2:]
    descriptors.loc[:, numeric_descriptor_columns] = descriptors.loc[
        :, numeric_descriptor_columns
    ].replace([np.inf, -np.inf], np.nan)

    morgan_columns = [f"morgan_{index:04d}" for index in range(morgan_n_bits)]
    morgan_matrix = (
        np.vstack(morgan_rows)
        if morgan_rows
        else np.empty((0, morgan_n_bits), dtype=np.uint8)
    )
    morgan_frame = pd.DataFrame(morgan_matrix, columns=morgan_columns, dtype=np.uint8)
    morgan_frame.insert(0, "molecule_id", molecule_ids)

    maccs_n_bits = 167
    if maccs_rows:
        maccs_n_bits = int(maccs_rows[0].shape[0])
    maccs_columns = [f"maccs_{index:03d}" for index in range(maccs_n_bits)]
    maccs_matrix = (
        np.vstack(maccs_rows)
        if maccs_rows
        else np.empty((0, maccs_n_bits), dtype=np.uint8)
    )
    maccs_frame = pd.DataFrame(maccs_matrix, columns=maccs_columns, dtype=np.uint8)
    maccs_frame.insert(0, "molecule_id", molecule_ids)

    return FeatureTables(
        descriptors=descriptors,
        morgan=morgan_frame,
        maccs=maccs_frame,
        failures=failures,
        descriptor_failures=descriptor_failures,
        descriptor_names=descriptor_names,
    )


def validate_feature_tables(tables: FeatureTables, expected_ids: pd.Series) -> list[str]:
    errors: list[str] = []
    expected = [str(value) for value in expected_ids]
    expected_set = set(expected)

    if len(expected_set) != len(expected):
        errors.append("Expected molecule IDs are not unique")
    if tables.failures:
        errors.append(f"Feature calculation failed for {len(tables.failures)} molecules")

    for name, frame in (
        ("descriptors", tables.descriptors),
        ("morgan", tables.morgan),
        ("maccs", tables.maccs),
    ):
        if frame["molecule_id"].duplicated().any():
            errors.append(f"{name} contains duplicate molecule IDs")
        actual_set = set(frame["molecule_id"].astype(str))
        if actual_set != expected_set:
            errors.append(
                f"{name} molecule IDs differ from the target table "
                f"({len(actual_set):,} actual; {len(expected_set):,} expected)"
            )

    for name, frame in (("morgan", tables.morgan), ("maccs", tables.maccs)):
        values = frame.iloc[:, 1:].to_numpy(dtype=np.uint8, copy=False)
        if values.size and not np.isin(values, [0, 1]).all():
            errors.append(f"{name} contains values other than 0 and 1")

    forbidden = {
        "log_s_target",
        "include_primary_benchmark",
        "target_reliability",
        "source_datasets",
    }
    for name, frame in (
        ("descriptors", tables.descriptors),
        ("morgan", tables.morgan),
        ("maccs", tables.maccs),
    ):
        leaked = sorted(forbidden.intersection(frame.columns))
        if leaked:
            errors.append(f"{name} includes forbidden target or audit columns: {leaked}")
    return errors


def _identifier_sha256(frame: pd.DataFrame) -> str:
    joined = "\n".join(frame["molecule_id"].astype(str))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_feature_report(
    tables: FeatureTables,
    targets: pd.DataFrame,
    morgan_radius: int,
    morgan_n_bits: int,
    include_chirality: bool,
    errors: list[str],
) -> dict[str, Any]:
    descriptor_values = tables.descriptors.iloc[:, 2:]
    missing_by_descriptor = descriptor_values.isna().sum()
    missing_by_descriptor = missing_by_descriptor.loc[missing_by_descriptor.gt(0)]
    constant_descriptors = descriptor_values.columns[descriptor_values.nunique(dropna=True).le(1)]

    morgan_values = tables.morgan.iloc[:, 1:].to_numpy(dtype=np.uint8, copy=False)
    maccs_values = tables.maccs.iloc[:, 1:].to_numpy(dtype=np.uint8, copy=False)

    return {
        "status": "fail" if errors else "pass",
        "blocking_errors": errors,
        "rdkit_version": rdkit.__version__,
        "molecules": int(len(targets)),
        "primary_benchmark_molecules": int(
            targets.get("include_primary_benchmark", pd.Series(False, index=targets.index)).sum()
        ),
        "molecule_id_sha256": _identifier_sha256(tables.descriptors),
        "representations": {
            "rdkit_2d_descriptors": {
                "rows": int(len(tables.descriptors)),
                "descriptor_columns": int(len(tables.descriptor_names)),
                "missing_values": int(descriptor_values.isna().sum().sum()),
                "descriptors_with_missing_values": {
                    str(key): int(value) for key, value in missing_by_descriptor.items()
                },
                "constant_descriptor_count": int(len(constant_descriptors)),
                "constant_descriptors": [str(value) for value in constant_descriptors],
            },
            "morgan_fingerprint": {
                "rows": int(len(tables.morgan)),
                "radius": int(morgan_radius),
                "diameter": int(2 * morgan_radius),
                "n_bits": int(morgan_n_bits),
                "include_chirality": bool(include_chirality),
                "bit_density": float(morgan_values.mean()) if morgan_values.size else None,
            },
            "maccs_keys": {
                "rows": int(len(tables.maccs)),
                "n_bits": int(tables.maccs.shape[1] - 1),
                "bit_density": float(maccs_values.mean()) if maccs_values.size else None,
            },
        },
        "calculation_failures": tables.failures,
        "descriptor_calculation_failures": tables.descriptor_failures,
        "leakage_control": (
            "Feature tables contain molecule identifiers and structure-derived values only; "
            "logS targets, source provenance, reliability classes, and benchmark flags are "
            "excluded."
        ),
    }
