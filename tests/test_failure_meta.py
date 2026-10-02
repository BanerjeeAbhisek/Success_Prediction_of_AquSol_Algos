import numpy as np
import pandas as pd

from aquasol_meta.failure_meta import (
    add_failure_labels,
    build_meta_data_dictionary,
    build_meta_model_dataset,
    build_molecule_failure_summary,
    build_run_failure_summary,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for model, predictions in {
        "dummy": [-2.0, -2.0],
        "ridge": [-0.5, -2.5],
    }.items():
        run_id = f"random_r1_rdkit_descriptors_{model}"
        for molecule_id, truth, prediction, similarity in zip(
            ["M3", "M4"], [-1.0, -3.0], predictions, [0.4, 0.8], strict=True
        ):
            residual = truth - prediction
            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "feature_type": "rdkit_descriptors",
                    "split_strategy": "random",
                    "repeat": 1,
                    "seed": 13,
                    "molecule_id": molecule_id,
                    "scaffold_group": molecule_id,
                    "maximum_train_similarity": similarity,
                    "true_log_s": truth,
                    "predicted_log_s": prediction,
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "squared_error": residual**2,
                }
            )
    return pd.DataFrame(rows)


def _meta_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_results = pd.DataFrame(
        {
            "run_id": [
                "random_r1_rdkit_descriptors_dummy",
                "random_r1_rdkit_descriptors_ridge",
            ],
            "model": ["dummy", "ridge"],
            "feature_type": ["rdkit_descriptors"] * 2,
            "split_strategy": ["random"] * 2,
            "repeat": [1] * 2,
            "seed": [13] * 2,
            "n_train": [1] * 2,
            "n_validation": [1] * 2,
            "n_final_fit": [2] * 2,
            "n_test": [2] * 2,
            "features_before_preprocessing": [6] * 2,
            "features_after_preprocessing": [6] * 2,
            "best_parameters": ['{"strategy": "mean"}', '{"alpha": 1.0}'],
            "validation_rmse": [2.0, 0.8],
            "validation_mae": [2.0, 0.7],
            "validation_r2": [0.0, 0.5],
            "validation_spearman": [np.nan, 1.0],
            "test_rmse": [1.0, 0.5],
            "test_mae": [1.0, 0.5],
            "test_r2": [0.0, 0.75],
            "test_spearman": [np.nan, 1.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "molecule_id": ["M1", "M2", "M3", "M4"],
            "split_strategy": ["random"] * 4,
            "repeat": [1] * 4,
            "seed": [13] * 4,
            "partition": ["train", "validation", "test", "test"],
            "scaffold_group": ["S1", "S2", "S3", "S4"],
            "maximum_train_similarity": [np.nan, 0.6, 0.4, 0.8],
        }
    )
    targets = pd.DataFrame(
        {
            "molecule_id": ["M1", "M2", "M3", "M4"],
            "log_s_target": [-2.0, -2.5, -1.0, -3.0],
            "target_reliability": ["single_observation"] * 4,
            "n_observations": [1] * 4,
            "n_source_datasets": [1] * 4,
            "include_primary_benchmark": [True] * 4,
        }
    )
    descriptors = pd.DataFrame(
        {
            "molecule_id": ["M1", "M2", "M3", "M4"],
            "MolWt": [100.0, 110.0, 120.0, 130.0],
            "MolLogP": [1.0, 1.1, 1.2, 1.3],
            "TPSA": [20.0, 25.0, 30.0, 35.0],
            "FractionCSP3": [0.1, 0.2, 0.3, 0.4],
            "NumRotatableBonds": [1.0, 2.0, 3.0, 4.0],
            "RingCount": [1.0, 1.0, 2.0, 2.0],
        }
    )
    return model_results, assignments, targets, descriptors


def test_q_i_uses_strict_one_log_s_absolute_error_threshold() -> None:
    predictions = _predictions()
    labeled = add_failure_labels(predictions)

    dummy = labeled.loc[labeled["model"].eq("dummy")]
    assert dummy["absolute_error"].tolist() == [1.0, 1.0]
    assert dummy["q_i"].tolist() == [0, 0]

    ridge = labeled.loc[labeled["model"].eq("ridge")]
    assert ridge["q_abs_error_gt_0p5"].tolist() == [0, 0]
    assert set(labeled["absolute_error_bin"]) == {"0.5-<1.0", "1.0-<2.0"}


def test_run_failure_summary_matches_molecule_level_errors() -> None:
    labeled = add_failure_labels(_predictions())
    summary = build_run_failure_summary(labeled)

    assert len(summary) == 2
    assert np.allclose(summary["rmse"], [1.0, 0.5])
    assert summary["failure_rate_gt_1p0"].eq(0.0).all()


def test_molecule_failure_summary_aggregates_across_models() -> None:
    labeled = add_failure_labels(_predictions())
    summary = build_molecule_failure_summary(labeled)

    assert len(summary) == 2
    assert summary["n_held_out_evaluations"].eq(1).all()
    assert summary["n_models"].eq(1).all()


def test_meta_dataset_separates_pretest_predictors_from_test_outcomes() -> None:
    labeled = add_failure_labels(_predictions())
    run_summary = build_run_failure_summary(labeled)
    model_results, assignments, targets, descriptors = _meta_inputs()

    meta = build_meta_model_dataset(
        model_results, run_summary, assignments, targets, descriptors
    )
    dictionary = build_meta_data_dictionary(meta)

    assert len(meta) == 2
    assert meta["run_id"].is_unique
    assert not any("test_log_s" in column for column in meta if column.startswith("x_"))
    assert meta.loc[meta["x_model"].eq("dummy"), "selection_eligible"].eq(False).all()
    assert meta.loc[meta["x_model"].eq("ridge"), "y_rmse_regret_vs_best"].item() == 0
    assert dictionary.loc[dictionary["role"].eq("predictor"), "uses_test_log_s"].eq(
        False
    ).all()
    assert dictionary.loc[dictionary["role"].eq("outcome"), "uses_test_log_s"].all()
    assert dictionary["recommended_for_initial_meta_model"].sum() == 16
