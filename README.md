# Aqueous-solubility model success prediction

This repository studies which machine-learning workflows succeed or fail on aqueous-solubility
prediction tasks. The first stage constructs an auditable observation-level database from the raw
source files. Model fitting should begin only after the source overlap, structure quality, endpoint
definitions, and repeated measurements have been reviewed.

## Data design

The pipeline does not blindly average the source files into one training table.

- `Data/` contains the original files and is never modified by the pipeline.
- `data_processed/master_observations.*` retains one row per original measurement.
- `data_processed/molecule_summary.*` describes repeated measurements by standardized molecule.
- `data_processed/modeling_targets.*` gives one auditable median target per standardized molecule.
- `reports/` records data quality and cross-source molecular overlap.

Both the original canonical structure and an RDKit fragment-parent structure are retained. The
parent structure supports duplicate and overlap detection. The original multicomponent form remains
available because salts and charge states can affect measured solubility.

## Set up in VS Code

The recommended environment uses Conda or Mamba because RDKit is distributed reliably through
conda-forge.

```bash
conda env create -f environment.yml
conda activate aquasol-meta
```

Open the repository in VS Code and select the `aquasol-meta` Python interpreter when prompted.

If Conda is unavailable, a Python 3.11 virtual environment can be used:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Build the master database

From the repository root:

```bash
aquasol-build-master
```

Equivalent module command:

```bash
python -m aquasol_meta.build_master
```

The build performs the following operations:

1. checks that all expected raw files and row counts are present;
2. maps all sources to one observation-level schema;
3. converts the OCHEM harmonized `-log(M)` field to `logS` by changing its sign;
4. parses and canonicalizes structures with RDKit;
5. creates full-structure and fragment-parent identifiers;
6. identifies repeated molecules across and within sources;
7. creates molecule-level descriptive summaries;
8. writes dataset-quality and source-overlap reports; and
9. stops with an error if blocking integrity checks fail.

The same command also creates the modelling-target table described below. Generated CSV, Parquet,
JSON, and report files may be committed when a versioned data release is wanted.

## Build or rebuild the modelling targets

If the master database already exists, rebuild only the target-construction stage with:

```bash
aquasol-build-targets
```

Equivalent module command:

```bash
python -m aquasol_meta.build_targets
```

Each row in `data_processed/modeling_targets.*` is one standardized parent molecule. The response
for model fitting is `log_s_target`, the median of all model-eligible measurements for that
molecule. The table also retains the mean, standard deviation, median absolute deviation, range,
quartiles, number of observations, number and names of source datasets, and pH/temperature coverage.
These audit columns describe target reliability; they are not ordinary molecular descriptors and
must not be given to a solubility model as predictors.

Three explicit flags create the initial experimental populations:

- `include_primary_benchmark`: excludes targets with a measurement range greater than 1 log unit
  and targets outside the broad review interval of -15 to 2.5 log10(mol/L);
- `include_all_data_benchmark`: keeps every model-eligible standardized molecule; and
- `include_conflict_stress_test`: selects molecules with high measurement disagreement.

No observation is deleted. The thresholds are recorded in every row and in
`reports/modeling_targets_report.json`. They can be changed for sensitivity analyses, for example:

```bash
aquasol-build-targets --conflict-range-min 1.5 --consistent-range-max 0.5
```

## Generate molecular features

After building the targets, generate label-free molecular representations with:

```bash
aquasol-build-features
```

This creates both Parquet files for efficient modeling and CSV copies for direct inspection:

- `data_processed/rdkit_descriptors.csv` and `.parquet`: RDKit 2D descriptors plus the molecule ID
  and standardized parent SMILES;
- `data_processed/morgan_fingerprints.csv` and `.parquet`: 2,048-bit radius-2 Morgan fingerprints
  (ECFP4) with chirality enabled;
- `data_processed/maccs_fingerprints.csv` and `.parquet`: 167-bit MACCS structural keys; and
- `reports/feature_quality.csv` and `.json`: versions, parameters, row coverage, missing
  descriptors, constant descriptors, fingerprint density, file sizes, and validation results.

Features are generated for all target molecules, not only the primary benchmark, so every
experimental population can reuse the same representation. The feature files deliberately exclude
logS, source provenance, reliability classes, and benchmark flags. Join them to
`modeling_targets.parquet` by `molecule_id` only after choosing an experimental population.

Do not fill missing descriptor values before splitting the data. Any imputer or scaler must be fit
using the training fold only. Fingerprint models normally use the binary bits without scaling.
Parquet is recommended for Python model fitting because it is smaller and preserves data types.
Use the CSV copies for inspection or software that cannot read Parquet. Add `--no-csv` when only
the compact Parquet outputs are needed.

## Run tests

```bash
pytest
```

## Important modelling rule

Do not randomly split the combined master table. Molecules and source measurements overlap across
the input datasets. Later model evaluation must group identical standardized molecules and must hold
out complete experimental sources or carefully defined chemical tasks.

## Current scope

This stage prepares and audits the data, encodes a transparent first target-resolution policy, and
generates reproducible structure-derived features. The next stage will calculate leakage-safe
train/test splits.
Measurement-condition and provenance columns remain audit variables unless a specifically defined
conditional-solubility task makes them available at prediction time.
