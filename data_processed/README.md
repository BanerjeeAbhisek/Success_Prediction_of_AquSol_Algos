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
- `split_assignments.csv` and `.parquet`: five fixed repeats of random, scaffold, and low-similarity
  train/validation/test assignments for primary-benchmark molecules.
- `meta_model_dataset.csv` and `.parquet`: one row per fitted model run. `x_` columns are predictors
  available before final test evaluation, `y_` columns are held-out outcomes, and identifiers or
  eligibility flags are unprefixed. Complete tasks or datasets, not individual rows, must be held
  out when evaluating a meta-model.
- `source_holdout_tasks.csv` and `.parquet`: task-specific targets and train/validation/test
  assignments for holding out each experimental source. Test targets use only held-out-source
  observations, while fit targets exclude both the source and every molecule appearing in it.
- `source_holdout_meta_model_dataset.*`: one fitted-model row for each source-holdout task.
- `meta_model_dataset_with_source_holdouts.*`: the original split tasks and source-holdout tasks in
  one table for grouped meta-learning experiments.

The molecule summary is an audit product. It must not replace the observation-level table when
source, pH, temperature, endpoint, or measurement disagreement matters.

For solubility model fitting, use `log_s_target` as the response. Do not use reliability, replicate,
source, or benchmark-flag columns as molecular predictors; they document how the target was built.
Join targets and features by the unique `molecule_id`. Fit descriptor imputation and scaling on each
training fold rather than on the complete feature table.
Parquet is the preferred modeling format; CSV is provided for inspection and interoperability.
