# Generated reports

The data builds write machine-readable quality, overlap, and target-construction reports here.

- `data_validation.json`: blocking integrity checks and source-level counts.
- `dataset_summary.csv`: observation and structure status by source.
- `source_overlap.csv`: standardized molecules shared across source pairs.
- `modeling_targets_report.json`: target counts, thresholds, exclusions, and validation status.
- `modeling_reliability_summary.csv`: target reliability-class summary.
- `feature_quality.json` and `.csv`: feature settings, RDKit version, molecule coverage,
  descriptor-quality checks, fingerprint density, output sizes, and validation status. The CSV uses
  a long `metric,value` layout to represent the nested JSON report without discarding information.
- `split_quality.csv` and `.json`: split sizes, logS distributions, scaffold coverage, train-set
  similarity distributions, policies, seeds, and leakage-validation status.
- `baseline_run_report.json` and `.csv`: requested and completed baseline runs, software version,
  skipped or failed combinations, output sizes, and validation/test-use policies. The CSV uses a
  long `metric,value` layout so the nested JSON content remains available in a spreadsheet-friendly
  form.
- `failure_analysis_report.json` and `.csv`: failure thresholds, validation results, leakage policy,
  output sizes, and the current meta-learning limitation.
- `meta_dataset_dictionary.csv`: role, test-label dependency, availability timing, initial-model
  recommendation flag, and definition for every meta-dataset column.
- `source_holdout_summary.csv`: source-specific task sizes, held-out target distributions, and
  train-to-test similarity summaries for every repeat.
- `source_holdout_report.json` and `.csv`: target-reconstruction rules, leakage checks, counts, and
  known cross-source dependence.
- `source_holdout_model_report.json` and `.csv`: completed source-level model runs, output sizes,
  prediction-storage policy, and grouped meta-validation requirements.
- `meta_dataset_with_source_holdouts_dictionary.csv`: column roles for the combined meta-dataset.
