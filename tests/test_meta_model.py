import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aquasol_meta.run_meta_model import (
    CANDIDATE_PREDICTORS,
    TASK_PREDICTORS,
    VALIDATION_PREDICTORS,
    _laplace_probabilities,
    _source_folds,
    meta_predictor_columns,
)


def test_source_folds_hold_out_complete_sources() -> None:
    frame = pd.DataFrame(
        {
            "heldout_source": ["A", "A", "B", "B", "C", "C"],
            "run_id": [f"R{index}" for index in range(6)],
        }
    )

    folds = list(_source_folds(frame))

    assert [source for source, _, _ in folds] == ["A", "B", "C"]
    for source, train_index, test_index in folds:
        assert set(frame.iloc[test_index]["heldout_source"]) == {source}
        assert source not in set(frame.iloc[train_index]["heldout_source"])


def test_meta_predictor_columns_are_explicit_and_outcome_free() -> None:
    columns = [*CANDIDATE_PREDICTORS, *TASK_PREDICTORS, *VALIDATION_PREDICTORS]
    frame = pd.DataFrame({column: [1.0] for column in columns})
    frame["x_hp_depth"] = 3.0
    frame["y_test_rmse"] = 0.5

    predictors = meta_predictor_columns(frame)

    assert "x_hp_depth" in predictors
    assert "y_test_rmse" not in predictors
    assert all(column.startswith("x_") for column in predictors)


def test_laplace_logistic_probabilities_have_ordered_intervals() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.5],
            [-0.5, -0.5],
            [0.5, 0.2],
            [1.0, -0.2],
            [2.0, 0.0],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[-1.5, 0.0], [0.0, 0.0], [1.5, 0.0]])
    model = LogisticRegression(C=1.0, max_iter=5000).fit(x_train, y_train)

    mean, lower, upper, covariance = _laplace_probabilities(
        model, x_train, x_test, c_value=1.0
    )

    assert np.all((0.0 <= mean) & (mean <= 1.0))
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
    assert covariance.shape == (x_train.shape[1] + 1, x_train.shape[1] + 1)
