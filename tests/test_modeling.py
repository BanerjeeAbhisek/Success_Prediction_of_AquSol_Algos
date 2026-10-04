import numpy as np
import pandas as pd

from aquasol_meta.modeling import (
    ESOLRegressor,
    _candidate_parameters,
    _feature_columns,
    _is_compatible,
    regression_metrics,
    run_baseline_experiments,
)
from aquasol_meta.run_baselines import _flatten_report


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    molecule_ids = [f"M{index:02d}" for index in range(30)]
    targets = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "log_s_target": np.linspace(-5, 1, 30),
            "include_primary_benchmark": [True] * 30,
            "target_reliability": ["single_observation"] * 30,
            "n_observations": [1] * 30,
            "n_source_datasets": [1] * 30,
            "log_s_range": [0.0] * 30,
        }
    )
    assignments = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "split_strategy": ["random"] * 30,
            "repeat": [1] * 30,
            "seed": [13] * 30,
            "partition": ["train"] * 24 + ["validation"] * 3 + ["test"] * 3,
            "scaffold_group": [f"S{index // 2}" for index in range(30)],
            "maximum_train_similarity": [np.nan] * 24 + [0.5] * 6,
        }
    )
    features = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "maccs_000": [index % 2 for index in range(30)],
            "maccs_001": [int(index > 14) for index in range(30)],
            "maccs_002": np.linspace(0, 1, 30),
        }
    )
    return targets, assignments, {"maccs_fingerprints": features}


def test_regression_metrics_are_computed_from_predictions() -> None:
    metrics = regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0]))

    assert np.isclose(metrics["rmse"], np.sqrt(1 / 3))
    assert np.isclose(metrics["mae"], 1 / 3)


def test_baseline_runner_tunes_on_validation_and_saves_test_residuals() -> None:
    targets, assignments, feature_tables = _inputs()
    results, predictions, tuning, report = run_baseline_experiments(
        targets,
        assignments,
        feature_tables=feature_tables,
        strategies=("random",),
        repeats=(1,),
        model_names=("dummy", "ridge"),
    )

    assert len(results) == 2
    assert len(predictions) == 6
    assert len(tuning) == 6
    assert report["status"] == "pass"
    assert set(predictions["model"]) == {"dummy", "ridge"}
    assert np.allclose(
        predictions["residual"],
        predictions["true_log_s"] - predictions["predicted_log_s"],
    )
    assert predictions["maximum_train_similarity"].notna().all()


def test_response_and_audit_columns_are_not_feature_columns() -> None:
    targets, assignments, feature_tables = _inputs()
    results, _, _, _ = run_baseline_experiments(
        targets,
        assignments,
        feature_tables=feature_tables,
        strategies=("random",),
        repeats=(1,),
        model_names=("ridge",),
    )

    assert results.loc[0, "features_before_preprocessing"] == 3


def test_numerically_unbounded_ipc_descriptor_is_excluded() -> None:
    descriptors = pd.DataFrame(
        {
            "molecule_id": ["M1", "M2"],
            "canonical_smiles_parent": ["CC", "CCC"],
            "Ipc": [1.0e100, 2.0],
            "AvgIpc": [1.1, 1.2],
            "MolWt": [30.0, 44.0],
        }
    )

    assert _feature_columns("rdkit_descriptors", descriptors) == ["AvgIpc", "MolWt"]


def test_additional_model_grid_and_compatibility_are_explicit() -> None:
    for model_name in ("elastic_net", "knn", "xgboost", "ngboost", "esol"):
        assert _candidate_parameters(model_name)

    assert _is_compatible("xgboost", "morgan_fingerprints")
    assert _is_compatible("knn", "maccs_fingerprints")
    assert not _is_compatible("ngboost", "morgan_fingerprints")
    assert not _is_compatible("esol", "maccs_fingerprints")


def test_esol_equation_matches_published_coefficient_form() -> None:
    frame = pd.DataFrame(
        {
            "MolLogP": [2.0],
            "MolWt": [100.0],
            "NumRotatableBonds": [2.0],
            "_esol_aromatic_proportion": [0.5],
        }
    )
    prediction = ESOLRegressor().fit(frame, np.array([-2.0])).predict(frame)

    expected = 0.16 - 1.5 * 2.0 - 0.01 * 100.0 + 0.5 * 2.0 - 1.5 * 0.5
    assert np.isclose(prediction[0], expected)


def test_baseline_report_can_be_flattened_without_losing_nested_values() -> None:
    report = {
        "status": "pass",
        "requested_models": ["ridge", "random_forest"],
        "failed_runs": [],
        "output_files": {"model_results_csv": {"bytes": 42}},
    }

    flattened = _flatten_report(report)

    assert list(flattened.columns) == ["metric", "value"]
    assert flattened.loc[flattened["metric"] == "requested_models", "value"].tolist() == [
        "ridge",
        "random_forest",
    ]
    assert flattened.loc[flattened["metric"] == "failed_runs", "value"].item() == "[]"
    assert (
        flattened.loc[
            flattened["metric"] == "output_files.model_results_csv.bytes", "value"
        ].item()
        == 42
    )
