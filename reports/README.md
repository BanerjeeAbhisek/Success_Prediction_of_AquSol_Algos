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
