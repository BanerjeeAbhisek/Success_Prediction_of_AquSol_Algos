# SC2019 external validation data

This directory contains the standardized tight and loose test sets from the Second Solubility
Challenge. The source archive is the MIT-licensed Zenodo record
[`10.5281/zenodo.7130065`](https://doi.org/10.5281/zenodo.7130065). The importer verifies the
published archive MD5 checksum `07ede2fc4ac33adab288dfea1d48682b` before reading it.

Run the reproducible import with:

```bash
aquasol-import-sc2019
```

The 34 MB downloaded archive is cached under `raw/` and ignored by Git. Only `Tight_set.csv` and
`Loose_set.csv` are imported. The archive's training compilation is intentionally ignored because
this project already has a fixed internal training corpus.

- `sc2019_external_inputs.*` contains structures and identifiers but no solubility response.
- `sc2019_external_labels.*` contains intrinsic logS0 in log10(mol/L) and must not be read during
  model selection.
- `sc2019_*_fingerprints.parquet` and `sc2019_rdkit_descriptors.parquet` are regenerated locally
  from standardized parent structures.
- `sc2019_overlap_audit.*` distinguishes overlap with the complete raw collection from overlap with
  the actual 8,969-molecule primary training benchmark.
- `sc2019_training_exclusions.csv` lists the exact primary-training structures that must be removed
  before any external candidate is fitted.

The files remain separate from `data_processed/master_observations.*`; they are never appended to
the internal training master.
