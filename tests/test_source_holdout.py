import numpy as np
import pandas as pd

from aquasol_meta.run_source_holdouts import (
    _concatenate_parquet_files,
    _write_partitioned_prediction_dataset,
)
from aquasol_meta.source_holdout import (
    build_source_holdout_tasks,
    validate_source_holdout_tasks,
)


def _observations() -> pd.DataFrame:
    rows = []
    source_molecules = {
        "SOURCE_A": ["M00", *[f"MA{index:02d}" for index in range(10)]],
        "SOURCE_B": ["M00", *[f"MB{index:02d}" for index in range(10)]],
    }
    observation = 0
    for source, molecule_ids in source_molecules.items():
        for index, molecule_id in enumerate(molecule_ids):
            observation += 1
            rows.append(
                {
                    "observation_id": f"O{observation:03d}",
                    "source_dataset": source,
                    "molecule_id": molecule_id,
                    "model_eligible": True,
                    "log_s": -1.0 - 0.2 * index,
                    "canonical_smiles_parent": "C" * (index + 1),
                    "murcko_scaffold": "" if index == 0 else f"S{index:02d}",
                    "temperature": np.nan,
                    "ph": np.nan,
                    "has_multiple_fragments": False,
                }
            )
    return pd.DataFrame(rows)


def _morgan() -> pd.DataFrame:
    molecule_ids = ["M00"] + [f"MA{index:02d}" for index in range(10)] + [
        f"MB{index:02d}" for index in range(10)
    ]
    rows = []
    for index, molecule_id in enumerate(molecule_ids):
        bits = [0] * 16
        bits[index % 16] = 1
        bits[(index * 3 + 1) % 16] = 1
        rows.append(
            {
                "molecule_id": molecule_id,
                **{f"morgan_{bit:04d}": value for bit, value in enumerate(bits)},
            }
        )
    return pd.DataFrame(rows)


def test_source_holdout_targets_remove_every_heldout_molecule_from_fit() -> None:
    tasks, summary, report = build_source_holdout_tasks(
        _observations(), _morgan(), seeds=(13, 37)
    )

    assert report["status"] == "pass"
    assert report["task_count"] == 4
    assert len(summary) == 4
    assert validate_source_holdout_tasks(tasks, expected_repeats=2) == []

    for _, task in tasks.groupby("task_id"):
        fit_ids = set(
            task.loc[task["partition"].isin({"train", "validation"}), "molecule_id"]
        )
        test_ids = set(task.loc[task["partition"].eq("test"), "molecule_id"])
        assert not fit_ids.intersection(test_ids)
        assert "M00" in test_ids
        assert "M00" not in fit_ids
        assert task.loc[task["partition"].eq("test"), "target_origin"].eq(
            "heldout_source_only"
        ).all()


def test_source_holdout_test_set_is_fixed_across_repeats() -> None:
    tasks, _, _ = build_source_holdout_tasks(
        _observations(), _morgan(), sources=("SOURCE_A",), seeds=(13, 37)
    )

    test_sets = [
        frozenset(task.loc[task["partition"].eq("test"), "molecule_id"])
        for _, task in tasks.groupby("task_id")
    ]
    assert len(set(test_sets)) == 1


def test_parquet_append_unifies_old_and_new_prediction_columns(tmp_path) -> None:
    old_path = tmp_path / "old.parquet"
    new_path = tmp_path / "new.parquet"
    combined_path = tmp_path / "combined.parquet"
    pd.DataFrame({"run_id": ["old"], "predicted_log_s": [-1.0]}).to_parquet(old_path)
    pd.DataFrame(
        {
            "run_id": ["new"],
            "predicted_log_s": [-2.0],
            "predicted_log_s_std": [0.3],
        }
    ).to_parquet(new_path)

    rows = _concatenate_parquet_files([old_path, new_path], combined_path)
    combined = pd.read_parquet(combined_path)

    assert rows == 2
    assert combined["run_id"].tolist() == ["old", "new"]
    assert np.isnan(combined.loc[0, "predicted_log_s_std"])
    assert combined.loc[1, "predicted_log_s_std"] == 0.3


def test_source_predictions_are_written_as_readable_partitioned_dataset(tmp_path) -> None:
    old_path = tmp_path / "old.parquet"
    new_path = tmp_path / "new.parquet"
    combined_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "run_id": ["old_a", "old_b"],
            "heldout_source": ["SOURCE_A", "SOURCE_B"],
            "predicted_log_s": [-1.0, -2.0],
        }
    ).to_parquet(old_path)
    pd.DataFrame(
        {
            "run_id": ["new_a"],
            "heldout_source": ["SOURCE_A"],
            "predicted_log_s": [-1.5],
            "predicted_log_s_std": [0.2],
        }
    ).to_parquet(new_path)

    rows = _write_partitioned_prediction_dataset(
        [old_path, new_path], combined_path
    )
    combined = pd.read_parquet(combined_path).sort_values("run_id").reset_index(drop=True)

    assert rows == 3
    assert combined_path.is_dir()
    assert combined["run_id"].tolist() == ["new_a", "old_a", "old_b"]
    assert set(combined["heldout_source"].astype(str)) == {"SOURCE_A", "SOURCE_B"}
