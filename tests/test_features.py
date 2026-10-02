import numpy as np
import pandas as pd

from aquasol_meta.build_features import build_feature_artifacts
from aquasol_meta.features import build_feature_tables, validate_feature_tables


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": ["ETHANOL", "BENZENE"],
            "canonical_smiles_parent": ["CCO", "c1ccccc1"],
            "log_s_target": [-0.3, -2.1],
            "include_primary_benchmark": [True, True],
            "target_reliability": ["single_observation", "consistent_replicates"],
            "source_datasets": ["A", "A|B"],
        }
    )


def test_feature_tables_have_one_row_per_molecule_and_binary_fingerprints() -> None:
    targets = _targets()
    tables = build_feature_tables(targets, morgan_n_bits=64)

    assert len(tables.descriptors) == 2
    assert tables.morgan.shape == (2, 65)
    assert tables.maccs.shape == (2, 168)
    assert set(tables.descriptors["molecule_id"]) == {"ETHANOL", "BENZENE"}
    assert np.isin(tables.morgan.iloc[:, 1:].to_numpy(), [0, 1]).all()
    assert np.isin(tables.maccs.iloc[:, 1:].to_numpy(), [0, 1]).all()
    assert validate_feature_tables(tables, targets["molecule_id"]) == []


def test_feature_tables_do_not_include_targets_or_audit_columns() -> None:
    tables = build_feature_tables(_targets(), morgan_n_bits=64)
    forbidden = {
        "log_s_target",
        "include_primary_benchmark",
        "target_reliability",
        "source_datasets",
    }

    assert forbidden.isdisjoint(tables.descriptors.columns)
    assert forbidden.isdisjoint(tables.morgan.columns)
    assert forbidden.isdisjoint(tables.maccs.columns)


def test_invalid_structure_is_reported_and_fails_id_validation() -> None:
    targets = pd.DataFrame(
        {
            "molecule_id": ["INVALID"],
            "canonical_smiles_parent": ["not-a-smiles"],
        }
    )
    tables = build_feature_tables(targets, morgan_n_bits=64)
    errors = validate_feature_tables(tables, targets["molecule_id"])

    assert len(tables.failures) == 1
    assert any("failed for 1 molecules" in error for error in errors)


def test_descriptor_exception_becomes_a_reported_missing_value() -> None:
    targets = pd.DataFrame(
        {
            "molecule_id": ["DEUTERIUM"],
            "canonical_smiles_parent": ["[H][2H]"],
        }
    )
    tables = build_feature_tables(targets, morgan_n_bits=64)

    assert pd.isna(tables.descriptors.loc[0, "SPS"])
    assert tables.descriptor_failures[0]["descriptor"] == "SPS"
    assert validate_feature_tables(tables, targets["molecule_id"]) == []


def test_feature_artifacts_include_csv_and_parquet_with_matching_rows(tmp_path) -> None:
    paths = build_feature_artifacts(
        _targets(),
        output_dir=tmp_path / "data_processed",
        reports_dir=tmp_path / "reports",
        morgan_n_bits=64,
    )

    assert pd.read_parquet(paths["rdkit_descriptors"]).shape[0] == 2
    assert pd.read_csv(paths["rdkit_descriptors_csv"]).shape[0] == 2
    assert pd.read_csv(paths["morgan_fingerprints_csv"]).shape == (2, 65)
    assert pd.read_csv(paths["maccs_fingerprints_csv"]).shape == (2, 168)
    quality = pd.read_csv(paths["feature_quality_csv"])
    assert "status" in set(quality["metric"])
    assert "representations.morgan_fingerprint.n_bits" in set(quality["metric"])
