import pandas as pd

from aquasol_meta.targets import build_modeling_targets, validate_modeling_targets


def _observations() -> pd.DataFrame:
    rows = [
        ("A:1", "A", "M1", -1.0),
        ("A:2", "A", "M2", -1.0),
        ("B:2", "B", "M2", -1.4),
        ("A:3", "A", "M3", -1.0),
        ("B:3", "B", "M3", -1.8),
        ("A:4", "A", "M4", -1.0),
        ("B:4", "B", "M4", -2.5),
        ("A:5", "A", "M5", -20.0),
    ]
    return pd.DataFrame(
        {
            "observation_id": [row[0] for row in rows],
            "source_dataset": [row[1] for row in rows],
            "molecule_id": [row[2] for row in rows],
            "log_s": [row[3] for row in rows],
            "canonical_smiles_parent": ["CCO"] * len(rows),
            "murcko_scaffold": ["[NO_SCAFFOLD]"] * len(rows),
            "temperature": [None] * len(rows),
            "ph": [None] * len(rows),
            "has_multiple_fragments": [False] * len(rows),
            "model_eligible": [True] * len(rows),
        }
    )


def test_target_table_uses_robust_median_and_reliability_classes() -> None:
    targets = build_modeling_targets(_observations()).set_index("molecule_id")

    assert targets.loc["M2", "log_s_target"] == -1.2
    assert targets.loc["M2", "source_datasets"] == "A|B"
    assert targets.loc["M1", "target_reliability"] == "single_observation"
    assert targets.loc["M2", "target_reliability"] == "consistent_replicates"
    assert targets.loc["M3", "target_reliability"] == "moderate_disagreement"
    assert targets.loc["M4", "target_reliability"] == "high_conflict"


def test_benchmark_flags_expose_conflict_without_deleting_rows() -> None:
    targets = build_modeling_targets(_observations()).set_index("molecule_id")

    assert len(targets) == 5
    assert targets["include_all_data_benchmark"].all()
    assert bool(targets.loc["M1", "include_primary_benchmark"])
    assert not bool(targets.loc["M4", "include_primary_benchmark"])
    assert bool(targets.loc["M4", "include_conflict_stress_test"])
    assert not bool(targets.loc["M5", "include_primary_benchmark"])
    assert targets.loc["M5", "primary_exclusion_reason"] == "outside_target_review_range"


def test_target_validation_accounts_for_every_eligible_observation() -> None:
    targets = build_modeling_targets(_observations())

    assert validate_modeling_targets(targets, expected_observations=8) == []
