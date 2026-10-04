import numpy as np
import pandas as pd

from aquasol_meta.run_chemprop import (
    CHEMPROP_FEATURE_TYPE,
    _attach_smiles,
    _build_task_outputs,
)


def _task() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "partition": ["train", "train", "validation", "validation", "test", "test"],
            "log_s_target": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            "split_strategy": ["random"] * 6,
            "repeat": [1] * 6,
            "seed": [13] * 6,
            "scaffold_group": ["A", "B", "C", "D", "E", "F"],
            "maximum_train_similarity": [np.nan, np.nan, 0.5, 0.4, 0.3, 0.2],
            "target_reliability": ["single_observation"] * 6,
            "n_observations": [1] * 6,
            "n_source_datasets": [1] * 6,
            "log_s_range": [0.0] * 6,
        }
    )


def test_attach_smiles_preserves_fixed_partitions() -> None:
    task = _task()
    structures = pd.DataFrame(
        {
            "molecule_id": task["molecule_id"],
            "canonical_smiles_parent": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
        }
    )

    attached = _attach_smiles(task, structures)

    assert attached["smiles"].tolist() == ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
    assert attached["partition"].tolist() == task["partition"].tolist()


def test_chemprop_outputs_follow_common_result_schema() -> None:
    task = _task()
    result, predictions, tuning = _build_task_outputs(
        "random_r1",
        task,
        validation_prediction=np.array([-2.5, -3.5]),
        test_prediction=np.array([-4.0, -5.0]),
        training_seconds=1.0,
        validation_predict_seconds=0.1,
        test_predict_seconds=0.1,
        epochs=3,
        patience=1,
        batch_size=16,
    )

    assert result.loc[0, "feature_type"] == CHEMPROP_FEATURE_TYPE
    assert np.isclose(result.loc[0, "test_rmse"], 1.0)
    assert predictions.loc[0, "residual"] == -1.0
    assert predictions.loc[0, "absolute_error"] == 1.0
    assert len(tuning) == 1
