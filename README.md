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

## Build fixed benchmark splits

After feature generation, create the reproducible split manifest with:

```bash
aquasol-build-splits
```

This creates `data_processed/split_assignments.csv` and `.parquet` with five repeats of three
strategies: logS-stratified random splitting, Murcko-scaffold group splitting, and a deliberately
low-similarity challenge split. Each strategy targets 80% training, 10% validation, and 10% testing.
Validation and test rows record their maximum radius-2 Morgan Tanimoto similarity to the matching
training set. Training rows leave this field missing because self-similarity is not informative.

Scaffold groups never cross partitions. Molecules without a ring scaffold use an explicitly
documented acyclic-singleton policy, avoiding one unusably large empty-scaffold group. Split counts,
target distributions, scaffold counts, and similarity distributions are written to
`reports/split_quality.csv` and `.json`.

The manifest contains no logS response. Join it to targets and features by `molecule_id` only when
running a specified strategy and repeat. Source-holdout evaluation remains a separate task because
it requires source-specific target construction to avoid using held-out measurements.

## Run baseline models

The default command runs the initial smoke test on random split repeat 1, using all three feature
representations and the four core models (Dummy, Ridge, Random Forest, and Extra Trees):

```bash
aquasol-run-baselines
```

For every model/feature/split/repeat combination, candidate settings are selected by validation
RMSE. The selected preprocessing and model are then refit on training plus validation, and the test
partition is evaluated once. Descriptor imputation, variance filtering, and scaling are fitted
inside the relevant training pipeline rather than on the complete dataset.

The raw RDKit `Ipc` descriptor is excluded from model fitting because its exponential magnitude
reaches approximately (10^{100}) in this collection and cannot be represented by the `float32`
arrays used by tree models. The normalized `AvgIpc` descriptor remains available. This fixed,
label-independent rule is applied consistently to every split.

The command writes model-level metrics, molecule-level test residuals, and validation candidate
results to `results/`. Large molecule-level CSV tables use gzip compression (`.csv.gz`) to remain
below GitHub's per-file limit; pandas and spreadsheet import tools can read them directly. Compact
Parquet copies are also retained. The run report is written to `reports/` as both JSON and a
long-form `metric,value` CSV. It does not create binary failure labels.

After reviewing the smoke test, run all fixed split strategies and repeats with the core models:

```bash
aquasol-run-baselines \
  --strategies random scaffold low_similarity \
  --repeats 1 2 3 4 5
```

To run only the advanced models after checking their runtime and convergence, pass
`--models hist_gradient_boosting svr mlp`. To retain the core models too, list all seven model
names. HistGradientBoosting is restricted to the compact RDKit descriptor representation.

Five additional classical or chemistry-aware comparisons are available:

- Elastic Net, as a sparse linear comparator;
- distance-weighted k-nearest neighbors, as a local-similarity method;
- XGBoost, as a modern boosted-tree model;
- NGBoost on RDKit descriptors, providing a predictive distribution as well as a point estimate;
  and
- the fixed Delaney ESOL equation, as an interpretable chemistry baseline rather than a fitted ML
  model.

They can be appended to the completed seven-model benchmark without rerunning or overwriting it:

```bash
aquasol-run-baselines \
  --strategies random scaffold low_similarity \
  --repeats 1 2 3 4 5 \
  --models elastic_net knn xgboost ngboost esol \
  --append
```

Elastic Net, kNN, and XGBoost use all three existing representations. NGBoost and ESOL are
descriptor-only by design. NGBoost writes its predicted standard deviation, Gaussian negative
log-likelihood, and 95% interval coverage; the other models leave those uncertainty fields missing.
Append mode refuses duplicate run identifiers and does not change the existing outputs if a new run
fails. Whenever baseline results change, `reports/downstream_artifacts_stale.json` explicitly marks
the failure summaries and meta-dataset as needing a refresh. Running `aquasol-build-meta-dataset`
clears that marker; it reconstructs tables and does not fit the meta-model.
When source-holdout meta rows already exist, the same refresh also regenerates the combined
within-benchmark plus source-holdout table, without rerunning source models.

Chemprop 2.3 is handled by a separate resumable command because its D-MPNN learns directly from the
molecular graph and uses validation-based early stopping rather than the scikit-learn fitting
protocol:

```bash
aquasol-run-chemprop --design within
aquasol-run-chemprop --design source
```

After the complete within run, merge it into the canonical benchmark without retraining by using:

```bash
aquasol-run-chemprop --design within --append-canonical
```

The default Chemprop experiment is one fixed D-MPNN architecture, 30 maximum epochs, patience 5,
and one model per task. This computer has no available PyTorch GPU backend, so the runner uses CPU
and saves one completed cache shard per task. An interrupted command can be rerun safely; cached
tasks are reused only when epochs, patience, and batch size match. For a
quick pipeline check, use `--task-limit 1 --epochs 2 --patience 1`; partial outputs are explicitly
named and ignored by Git. Complete Chemprop tables remain as auditable standalone outputs;
`--append-canonical` also adds their rows to the canonical benchmark before the failure/meta tables
are refreshed. None of these commands trains the meta-model.

## Characterize failures and build the meta-dataset

After the complete baseline grid finishes, run:

```bash
aquasol-build-meta-dataset
```

The primary molecule-level failure indicator is
`q_i = 1(abs(true_log_s - predicted_log_s) > 1.0)`. One logS unit corresponds to a tenfold
solubility error. Companion indicators at 0.5 and 2.0 logS support sensitivity analysis; the raw
signed and absolute errors remain available.

This command produces molecule-level failure labels, one failure-summary row per fitted model, an
across-repeat summary, and `data_processed/meta_model_dataset.*`. In the meta-dataset, columns
beginning with `x_` are available before final test evaluation, while columns beginning with `y_`
are held-out outcomes. Test-set molecular structures and train-test similarity are valid `x_`
inputs; held-out test logS values and errors are never included in `x_`.

The dummy models are retained for comparison but have `selection_eligible = false`. For substantive
candidates, `y_top3_candidate` is the binary model-selection outcome and
`y_rmse_regret_vs_best` measures continuous distance from the task winner. Do not randomly split
meta-dataset rows: hold out complete `task_id` groups, and ultimately complete external datasets.

The within-benchmark table contains correlated model rows from 15 chemical tasks. It validates the
failure-analysis pipeline but is not sufficient for the final publishable meta-model. Source-holdout
and genuinely external-dataset tasks are required before training the definitive model selector.

## Add source-holdout tasks

Build source-specific targets and assignments with:

```bash
aquasol-build-source-holdouts
```

For each held-out source, the test target is reconstructed exclusively from that source. Every
molecule appearing in the held-out source is removed from training and validation, even if the same
structure occurs in another source. Targets for the fitting pool are then rebuilt using only the
remaining observations. Five fixed 90/10 training/validation splits are created while the source
test set remains fixed.

Run the core model grid over the source tasks with:

```bash
aquasol-run-source-holdouts
```

After the completed core and advanced grids exist, append the five additional classical or
chemistry-aware models with:

```bash
aquasol-run-source-holdouts \
  --models elastic_net knn xgboost ngboost esol \
  --append
```

This creates source-level results and a combined meta-dataset. Molecule-level source predictions
are a Parquet-only dataset partitioned by held-out source; each physical part remains below GitHub's
100 MB per-file limit, and `pandas.read_parquet` reads the directory normally. Evaluate the
meta-model by holding out complete `heldout_source` groups. The source datasets share many
molecules, so genuinely external datasets are still required for the strongest publication claim.

## Run tests

```bash
pytest
```

## Reproduce and verify the current milestone

`requirements-lock.txt` records the exact Python packages used for the archived results. A fresh
Python 3.11 environment can be populated with:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

The complete command order for rebuilding the current milestone from `Data/` is recorded in
`scripts/run_reproducible_pipeline.sh`. It deliberately reproduces the current experimental scope:
all 13 within-benchmark model families and the existing seven-family source-holdout comparison. It
does not train a meta-model.

After any intentional artifact change, regenerate the audit snapshot with:

```bash
aquasol-build-reproducibility
```

This writes `reports/reproducibility_manifest.json` and `.csv`, plus
`reports/artifact_checksums.csv`. The manifest records exact package versions, seeds, row counts,
model counts, Chemprop configuration and SHA-256 checksums, and fails if a checked file reaches the
95 MB GitHub safety threshold.

## Important modelling rule

Do not randomly split the combined master table. Molecules and source measurements overlap across
the input datasets. Later model evaluation must group identical standardized molecules and must hold
out complete experimental sources or carefully defined chemical tasks.

## Current scope

This stage prepares and audits the data, encodes a transparent first target-resolution policy,
generates reproducible structure-derived features, fixes leakage-checked benchmark splits, runs the
complete core baseline grid, constructs molecule- and run-level failure tables, and supports strict
source-holdout evaluation. Genuinely external-dataset tasks remain necessary before definitive
meta-model training.
Measurement-condition and provenance columns remain audit variables unless a specifically defined
conditional-solubility task makes them available at prediction time.
