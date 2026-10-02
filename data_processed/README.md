# Generated data products

This directory is populated by `aquasol-build-master` and `aquasol-build-targets`.

- `master_observations.csv` and `.parquet`: one row per original observation.
- `molecule_summary.csv` and `.parquet`: descriptive summaries by standardized parent molecule.
- `modeling_targets.csv` and `.parquet`: one median target per standardized parent molecule, with
  replicate-disagreement summaries and benchmark inclusion flags.
- `rdkit_descriptors.csv` and `.parquet`: molecule identifiers, standardized parent SMILES, and
  RDKit 2D descriptors.
- `morgan_fingerprints.csv` and `.parquet`: 2,048-bit radius-2 Morgan fingerprints with chirality.
- `maccs_fingerprints.csv` and `.parquet`: 167-bit MACCS structural keys.

The molecule summary is an audit product. It must not replace the observation-level table when
source, pH, temperature, endpoint, or measurement disagreement matters.

For solubility model fitting, use `log_s_target` as the response. Do not use reliability, replicate,
source, or benchmark-flag columns as molecular predictors; they document how the target was built.
Join targets and features by the unique `molecule_id`. Fit descriptor imputation and scaling on each
training fold rather than on the complete feature table.
Parquet is the preferred modeling format; CSV is provided for inspection and interoperability.
