import numpy as np
import pandas as pd

from aquasol_meta.splits import build_split_assignments, validate_split_assignments


def _inputs(n_molecules: int = 60, n_bits: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecule_ids = [f"M{index:03d}" for index in range(n_molecules)]
    targets = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "log_s_target": np.linspace(-8.0, 1.0, n_molecules),
            "murcko_scaffold": [f"S{index // 3:02d}" for index in range(n_molecules)],
            "include_primary_benchmark": [True] * n_molecules,
        }
    )
    rng = np.random.default_rng(5)
    bits = rng.binomial(1, 0.12, size=(n_molecules, n_bits)).astype(np.uint8)
    morgan = pd.DataFrame(bits, columns=[f"morgan_{index:04d}" for index in range(n_bits)])
    morgan.insert(0, "molecule_id", molecule_ids)
    return targets, morgan


def test_every_strategy_repeat_covers_each_primary_molecule_once() -> None:
    targets, morgan = _inputs()
    assignments = build_split_assignments(targets, morgan, seeds=(11, 23))

    assert len(assignments) == 60 * 3 * 2
    assert not assignments.duplicated(["molecule_id", "split_strategy", "repeat"]).any()
    assert validate_split_assignments(assignments, targets, expected_repeats=2) == []


def test_scaffolds_do_not_cross_partitions() -> None:
    targets, morgan = _inputs()
    assignments = build_split_assignments(targets, morgan, seeds=(11,))
    scaffold = assignments.loc[assignments["split_strategy"].eq("scaffold")]

    assert scaffold.groupby("scaffold_group")["partition"].nunique().max() == 1


def test_similarity_is_defined_only_for_validation_and_test_rows() -> None:
    targets, morgan = _inputs()
    assignments = build_split_assignments(targets, morgan, seeds=(11,))
    evaluation = assignments["partition"].isin({"validation", "test"})

    assert assignments.loc[evaluation, "maximum_train_similarity"].notna().all()
    assert assignments.loc[~evaluation, "maximum_train_similarity"].isna().all()
    assert assignments.loc[evaluation, "maximum_train_similarity"].between(0, 1).all()


def test_split_manifest_does_not_copy_the_response() -> None:
    targets, morgan = _inputs()
    assignments = build_split_assignments(targets, morgan, seeds=(11,))

    assert "log_s_target" not in assignments.columns
    assert "include_primary_benchmark" not in assignments.columns
