# Raw solubility data

The CSV files in this directory are source files. The data-curation pipeline reads them but does not
modify them. Processed tables are written to `data_processed/` and reports to `reports/`.

## Source inventory and target mapping

| Project source | File | Rows | Harmonized target |
| --- | --- | ---: | --- |
| AqSolDB | `AqSolDB.csv` | 9,982 | `Solubility`, used as `log10(mol/L)` |
| Delaney | `Delaney.csv` | 1,128 | measured log solubility in mol/L |
| AQUA | `aqua_org.csv` | 1,311 | `logS` |
| PHYSPROP | `phys_org.csv` | 2,010 | `logS` |
| OCHEM | `ochem_all.csv` | 36,449 | negative of the converted `-log(M)` field |

The original OCHEM measurement field contains several units. The pipeline therefore does not treat
that raw field as a common target. It uses `Water solubility {measured, converted}`, whose recorded
unit is `-log(M)`, and calculates

```text
logS = -1 * converted_minus_log10_molar
```

## Known issues that must remain visible

- The source collections contain overlapping molecules and cannot be treated as independent without
  molecule-aware and source-aware splitting.
- OCHEM contains repeated measurements, missing structures, mixed original units, and measurement
  disagreement.
- AqSolDB contains aggregated observations. `Ocurrences` records the source count and `SD` records
  the reported variation.
- The repository does not yet document intrinsic versus apparent solubility consistently.
- Temperature and pH are missing for most observations.
- The original publication, download URL, version, retrieval date, and licence for each file still
  need to be recorded below before publication.

## Provenance to complete

| Source | Publication/URL | Version or retrieval date | Licence | Endpoint notes |
| --- | --- | --- | --- | --- |
| AqSolDB | TODO | TODO | TODO | TODO |
| Delaney | TODO | TODO | TODO | TODO |
| AQUA | TODO | TODO | TODO | TODO |
| PHYSPROP | TODO | TODO | TODO | TODO |
| OCHEM | TODO | TODO | TODO | TODO |
