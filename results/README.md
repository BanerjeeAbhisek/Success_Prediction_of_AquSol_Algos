# Model outputs

`aquasol-run-baselines` writes model-level metrics, molecule-level test predictions, and validation
hyperparameter results here in both CSV and Parquet formats.

- `model_results.*`: one row per completed model/feature/split/repeat run.
- `test_predictions.*`: one row per held-out molecule and completed run, including residuals and
  audit fields used later to study failure conditions.
- `hyperparameter_results.*`: validation metrics for every candidate configuration.

The test partition is evaluated only after validation-based configuration selection. No binary
success/failure label is created at this stage.

The raw RDKit `Ipc` descriptor is excluded from fitting because its exponential magnitude exceeds
tree-model numeric limits. The normalized `AvgIpc` descriptor is retained.

`aquasol-build-meta-dataset` adds the following derivative analysis products without changing the
raw baseline outputs:

- `prediction_failures.*`: molecule-level prediction rows with primary `q_i` and 0.5-, 1.0-, and
  2.0-logS threshold indicators.
- `run_failure_summary.*`: one row per fitted model with error quantiles, directional bias, and
  failure rates.
- `molecule_failure_summary.*`: one row per molecule that appeared in a test partition, aggregating
  repeated model and task evaluations to identify consistently difficult chemistry.
- `failure_group_summary.csv`: across-repeat results by split strategy, representation, and model.

`aquasol-run-source-holdouts` writes source-level model results, tuning results, run summaries, and
group summaries with the `source_holdout_` prefix. The molecule-level source-holdout predictions and
failure labels are stored only as `source_holdout_prediction_failures.parquet`; a CSV copy would
exceed GitHub's 100 MB per-file limit.
