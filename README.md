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

Generated CSV and Parquet files are ignored by Git because they can be rebuilt from the raw files.

## Run tests

```bash
pytest
```

## Important modelling rule

Do not randomly split the combined master table. Molecules and source measurements overlap across
the input datasets. Later model evaluation must group identical standardized molecules and must hold
out complete experimental sources or carefully defined chemical tasks.

## Current scope

This stage prepares and audits the data. It does not yet decide whether repeated measurements should
be pooled, which endpoint definitions are scientifically compatible, or which observations enter a
final model-fitting task. Those decisions will be encoded in a separate task-construction stage after
the generated reports are reviewed.
