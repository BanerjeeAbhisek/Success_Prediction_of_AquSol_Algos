# Model outputs

`aquasol-run-baselines` writes model-level metrics, molecule-level test predictions, and validation
hyperparameter results here in CSV and Parquet formats. Molecule-level CSV tables use `.csv.gz`
compression once their expanded model grid would otherwise approach GitHub's file-size limit.

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
failure labels are stored as the partitioned Parquet dataset
`source_holdout_prediction_failures.parquet/`; pandas reads this directory with the same
`pd.read_parquet(...)` call used for a single file. Its bounded part files remain below GitHub's
100 MB per-file limit. A CSV copy is intentionally omitted.

`aquasol-run-chemprop` stores completed task shards in the Git-ignored `chemprop_cache/` directory
so interrupted CPU runs can resume. Complete within-benchmark and source-holdout runs produce
`chemprop_within_*` and `chemprop_source_holdout_*` result tables. Molecule-level Chemprop
predictions are Parquet-only. A run using `--task-limit` writes explicitly named `*_partial_*`
outputs, which are ignored by Git and must not be treated as the complete benchmark. For the
within-benchmark design, `--append-canonical` retains these standalone files while also adding the
15 Chemprop rows to `model_results.*`, `test_predictions.*`, and `hyperparameter_results.*`.
