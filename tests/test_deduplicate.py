import pandas as pd

from aquasol_meta.deduplicate import build_molecule_summary, build_source_overlap


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["A:1", "B:1", "A:2"],
            "source_dataset": ["A", "B", "A"],
            "molecule_id": ["KEY-1", "KEY-1", "KEY-2"],
            "canonical_smiles_parent": ["CCO", "CCO", "CC"],
            "murcko_scaffold": ["[NO_SCAFFOLD]"] * 3,
            "log_s": [-1.0, -2.5, -0.5],
            "has_multiple_fragments": [False, False, False],
            "structure_status": ["ok", "ok", "ok"],
        }
    )


def test_molecule_summary_flags_large_measurement_disagreement() -> None:
    summary = build_molecule_summary(_observations()).set_index("molecule_id")

    assert summary.loc["KEY-1", "n_observations"] == 2
    assert summary.loc["KEY-1", "n_source_datasets"] == 2
    assert summary.loc["KEY-1", "log_s_range"] == 1.5
    assert bool(summary.loc["KEY-1", "measurement_conflict_over_1_log_unit"])


def test_source_overlap_uses_standardized_molecule_ids() -> None:
    overlap = build_source_overlap(_observations())

    assert len(overlap) == 1
    assert overlap.loc[0, "shared_molecules"] == 1
    assert overlap.loc[0, "percent_of_smaller_source"] == 100.0

