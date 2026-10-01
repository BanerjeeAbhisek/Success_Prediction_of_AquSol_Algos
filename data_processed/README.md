# Generated data products

This directory is populated by `aquasol-build-master`.

- `master_observations.csv` and `.parquet`: one row per original observation.
- `molecule_summary.csv` and `.parquet`: descriptive summaries by standardized parent molecule.

The molecule summary is an audit product. It must not replace the observation-level table when
source, pH, temperature, endpoint, or measurement disagreement matters.

