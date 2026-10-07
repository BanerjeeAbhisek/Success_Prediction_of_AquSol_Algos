from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_tables, validate_feature_tables
from .run_baselines import _flatten_report, _write_table
from .standardize import standardize_dataframe

SC2019_RECORD_URL = "https://zenodo.org/records/7130065"
SC2019_ARCHIVE_URL = (
    "https://zenodo.org/api/records/7130065/files/datasets.tar.gz/content"
)
SC2019_ARCHIVE_MD5 = "07ede2fc4ac33adab288dfea1d48682b"
SC2019_ARCHIVE_BYTES = 34_000_436
SC2019_DOI = "10.5281/zenodo.7130065"
SC2019_LICENSE = "MIT"
SC2019_TABLES = {
    "sc2019_tight": "datasets/Tight_set.csv",
    "sc2019_loose": "datasets/Loose_set.csv",
}
EXPECTED_ROWS = {"sc2019_tight": 100, "sc2019_loose": 32}


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sc2019_archive(destination: Path, force: bool = False) -> Path:
    """Download the immutable Zenodo archive and verify its published checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if _file_digest(destination, "md5") == SC2019_ARCHIVE_MD5:
            return destination
        raise RuntimeError(
            f"Existing archive has the wrong MD5 checksum: {destination}. "
            "Use --force-download to replace it."
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        SC2019_ARCHIVE_URL,
        headers={"User-Agent": "aquasol-meta/0.1 reproducible research importer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != SC2019_ARCHIVE_BYTES:
            raise RuntimeError(
                f"Downloaded archive size is {temporary.stat().st_size:,}; "
                f"expected {SC2019_ARCHIVE_BYTES:,} bytes"
            )
        checksum = _file_digest(temporary, "md5")
        if checksum != SC2019_ARCHIVE_MD5:
            raise RuntimeError(
                f"Downloaded archive MD5 is {checksum}; expected {SC2019_ARCHIVE_MD5}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _read_challenge_table(archive: tarfile.TarFile, task: str, member: str) -> pd.DataFrame:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Archive member not found: {member}")
    frame = pd.read_csv(
        handle,
        header=1,
        usecols=["ID", "Name", "SMILES", "Solubility"],
    ).rename(
        columns={
            "ID": "external_id",
            "Name": "compound_name",
            "SMILES": "original_smiles",
            "Solubility": "intrinsic_log_s",
        }
    )
    frame.insert(1, "external_task", task)
    frame.insert(2, "measurement_quality", task.removeprefix("sc2019_"))
    return frame


def read_sc2019_archive(archive_path: Path) -> pd.DataFrame:
    if _file_digest(archive_path, "md5") != SC2019_ARCHIVE_MD5:
        raise RuntimeError(f"SC2019 archive checksum validation failed: {archive_path}")
    frames = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = set(archive.getnames())
        missing = sorted(set(SC2019_TABLES.values()).difference(members))
        if missing:
            raise FileNotFoundError(f"SC2019 archive is missing members: {missing}")
        for task, member in SC2019_TABLES.items():
            frame = _read_challenge_table(archive, task, member)
            expected = EXPECTED_ROWS[task]
            if len(frame) != expected:
                raise RuntimeError(f"{task} contains {len(frame)} rows; expected {expected}")
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["external_id"].duplicated().any():
        raise RuntimeError("SC2019 contains duplicate external IDs")
    combined["intrinsic_log_s"] = pd.to_numeric(
        combined["intrinsic_log_s"], errors="raise"
    )
    return combined


def _nearest_internal_similarity(
    external_morgan: pd.DataFrame,
    internal_morgan: pd.DataFrame,
) -> pd.DataFrame:
    bit_columns = [column for column in external_morgan if column.startswith("morgan_")]
    if bit_columns != [
        column for column in internal_morgan if column.startswith("morgan_")
    ]:
        raise ValueError("External and internal Morgan fingerprint columns differ")
    external_bits = external_morgan[bit_columns].to_numpy(dtype=np.float32)
    internal_bits = internal_morgan[bit_columns].to_numpy(dtype=np.float32)
    intersections = external_bits @ internal_bits.T
    unions = (
        external_bits.sum(axis=1)[:, None]
        + internal_bits.sum(axis=1)[None, :]
        - intersections
    )
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    nearest_index = similarities.argmax(axis=1)
    return pd.DataFrame(
        {
            "molecule_id": external_morgan["molecule_id"].astype(str),
            "maximum_primary_training_tanimoto_similarity": similarities[
                np.arange(len(similarities)), nearest_index
            ],
            "nearest_primary_training_molecule_id": internal_morgan.iloc[nearest_index][
                "molecule_id"
            ].astype(str).to_numpy(),
        }
    )


def _internal_overlap_summary(master: pd.DataFrame) -> pd.DataFrame:
    usable = master.loc[
        master["structure_status"].eq("ok") & master["molecule_id"].notna()
    ].copy()
    return (
        usable.groupby("molecule_id", as_index=False)
        .agg(
            internal_observation_count=("molecule_id", "size"),
            internal_source_count=("source_dataset", "nunique"),
            internal_source_datasets=(
                "source_dataset",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
        )
        .sort_values("molecule_id", kind="stable")
        .reset_index(drop=True)
    )


def _task_summary(audit: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    merged = audit.merge(
        labels[["external_id", "intrinsic_log_s"]],
        on="external_id",
        how="left",
        validate="one_to_one",
    )
    output: dict[str, Any] = {}
    for task, rows in merged.groupby("external_task", sort=True):
        output[str(task)] = {
            "rows": int(len(rows)),
            "exact_raw_collection_overlap_rows": int(
                rows["exact_internal_overlap"].sum()
            ),
            "rows_without_exact_raw_collection_overlap": int(
                (~rows["exact_internal_overlap"]).sum()
            ),
            "intrinsic_log_s_min": float(rows["intrinsic_log_s"].min()),
            "intrinsic_log_s_max": float(rows["intrinsic_log_s"].max()),
            "intrinsic_log_s_mean": float(rows["intrinsic_log_s"].mean()),
            "exact_primary_training_overlap_rows": int(
                rows["exact_primary_training_overlap"].sum()
            ),
            "rows_without_exact_primary_training_overlap": int(
                (~rows["exact_primary_training_overlap"]).sum()
            ),
            "median_maximum_primary_training_tanimoto_similarity": float(
                rows["maximum_primary_training_tanimoto_similarity"].median()
            ),
        }
    return output


def build_sc2019_artifacts(
    archive_path: Path,
    internal_master: pd.DataFrame,
    internal_targets: pd.DataFrame,
    internal_morgan: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    """Create label-separated external inputs, features and an overlap audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw = read_sc2019_archive(archive_path)
    standardized = standardize_dataframe(raw, progress_every=0)
    invalid = standardized.loc[~standardized["structure_status"].eq("ok")]
    if not invalid.empty:
        raise RuntimeError(
            f"SC2019 structure standardization failed for {len(invalid)} compounds"
        )
    if standardized["molecule_id"].duplicated().any():
        raise RuntimeError("SC2019 contains duplicate standardized parent structures")

    input_columns = [
        "external_id",
        "external_task",
        "measurement_quality",
        "compound_name",
        "original_smiles",
        "structure_status",
        "canonical_smiles_full",
        "canonical_smiles_parent",
        "full_inchi_key",
        "parent_inchi_key",
        "molecule_id",
        "murcko_scaffold",
        "fragment_count",
        "has_multiple_fragments",
        "formal_charge_full",
        "heavy_atom_count_parent",
        "molecular_weight_parent",
    ]
    inputs = standardized[input_columns].copy()
    labels = standardized[
        ["external_id", "external_task", "measurement_quality", "intrinsic_log_s"]
    ].copy()
    labels["endpoint"] = "intrinsic_aqueous_solubility_log_s0"
    labels["units"] = "log10(mol/L)"
    labels["locked_for_model_selection"] = True

    feature_input = inputs[["molecule_id", "canonical_smiles_parent"]].copy()
    features = build_feature_tables(feature_input)
    feature_errors = validate_feature_tables(features, feature_input["molecule_id"])
    if feature_errors:
        raise RuntimeError("SC2019 feature validation failed:\n- " + "\n- ".join(feature_errors))

    internal_overlap = _internal_overlap_summary(internal_master)
    audit = inputs[
        [
            "external_id",
            "external_task",
            "measurement_quality",
            "compound_name",
            "molecule_id",
            "canonical_smiles_parent",
        ]
    ].merge(internal_overlap, on="molecule_id", how="left", validate="one_to_one")
    audit["exact_internal_overlap"] = audit["internal_observation_count"].notna()
    audit["internal_observation_count"] = (
        audit["internal_observation_count"].fillna(0).astype(int)
    )
    audit["internal_source_count"] = audit["internal_source_count"].fillna(0).astype(int)
    audit["internal_source_datasets"] = audit["internal_source_datasets"].fillna("")
    if "include_primary_benchmark" in internal_targets:
        primary_targets = internal_targets.loc[
            internal_targets["include_primary_benchmark"].astype(bool)
        ].copy()
    else:
        primary_targets = internal_targets.copy()
    target_ids = set(primary_targets["molecule_id"].astype(str))
    primary_morgan = internal_morgan.loc[
        internal_morgan["molecule_id"].astype(str).isin(target_ids)
    ].copy()
    if len(primary_morgan) != len(target_ids):
        raise RuntimeError("Primary targets and internal Morgan features cover different molecules")
    similarity = _nearest_internal_similarity(features.morgan, primary_morgan)
    audit = audit.merge(similarity, on="molecule_id", how="left", validate="one_to_one")
    audit["exact_primary_training_overlap"] = audit["molecule_id"].astype(str).isin(
        target_ids
    )
    audit["nearest_primary_training_neighbor_is_exact_match"] = audit[
        "maximum_primary_training_tanimoto_similarity"
    ].ge(1.0 - 1e-12)
    audit = audit.sort_values(["external_task", "external_id"], kind="stable").reset_index(
        drop=True
    )

    exclusions = audit.loc[
        audit["exact_primary_training_overlap"]
    ].copy()
    exclusions = (
        exclusions.groupby("molecule_id", as_index=False)
        .agg(
            canonical_smiles_parent=("canonical_smiles_parent", "first"),
            external_ids=("external_id", lambda values: "|".join(sorted(values.astype(str)))),
            external_tasks=(
                "external_task",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
            internal_observation_count=("internal_observation_count", "first"),
            internal_source_count=("internal_source_count", "first"),
            internal_source_datasets=("internal_source_datasets", "first"),
        )
        .sort_values("molecule_id", kind="stable")
        .reset_index(drop=True)
    )

    paths = {
        "inputs_csv": output_dir / "sc2019_external_inputs.csv",
        "inputs_parquet": output_dir / "sc2019_external_inputs.parquet",
        "labels_csv": output_dir / "sc2019_external_labels.csv",
        "labels_parquet": output_dir / "sc2019_external_labels.parquet",
        "descriptors_parquet": output_dir / "sc2019_rdkit_descriptors.parquet",
        "morgan_parquet": output_dir / "sc2019_morgan_fingerprints.parquet",
        "maccs_parquet": output_dir / "sc2019_maccs_fingerprints.parquet",
        "overlap_audit_csv": output_dir / "sc2019_overlap_audit.csv",
        "overlap_audit_parquet": output_dir / "sc2019_overlap_audit.parquet",
        "training_exclusions_csv": output_dir / "sc2019_training_exclusions.csv",
        "report_json": reports_dir / "sc2019_import_report.json",
        "report_csv": reports_dir / "sc2019_import_report.csv",
    }
    _write_table(inputs, paths["inputs_csv"], paths["inputs_parquet"])
    _write_table(labels, paths["labels_csv"], paths["labels_parquet"])
    features.descriptors.to_parquet(
        paths["descriptors_parquet"], index=False, compression="zstd"
    )
    features.morgan.to_parquet(paths["morgan_parquet"], index=False, compression="zstd")
    features.maccs.to_parquet(paths["maccs_parquet"], index=False, compression="zstd")
    _write_table(audit, paths["overlap_audit_csv"], paths["overlap_audit_parquet"])
    exclusions.to_csv(paths["training_exclusions_csv"], index=False)

    overlapping_ids = set(exclusions["molecule_id"].astype(str))
    report: dict[str, Any] = {
        "status": "pass",
        "source": {
            "title": (
                "Blinded Predictions and Post-hoc Analysis of the Second Solubility "
                "Challenge Data"
            ),
            "record_url": SC2019_RECORD_URL,
            "doi": SC2019_DOI,
            "license": SC2019_LICENSE,
            "archive_url": SC2019_ARCHIVE_URL,
            "archive_bytes": int(archive_path.stat().st_size),
            "archive_md5": _file_digest(archive_path, "md5"),
            "archive_sha256": _file_digest(archive_path, "sha256"),
        },
        "endpoint": "intrinsic aqueous solubility logS0",
        "units": "log10(mol/L)",
        "external_rows": int(len(inputs)),
        "valid_structures": int(inputs["structure_status"].eq("ok").sum()),
        "duplicate_external_ids": int(inputs["external_id"].duplicated().sum()),
        "duplicate_standardized_parents": int(inputs["molecule_id"].duplicated().sum()),
        "task_summary": _task_summary(audit, labels),
        "exact_raw_collection_overlap_rows": int(audit["exact_internal_overlap"].sum()),
        "external_rows_without_exact_raw_collection_overlap": int(
            (~audit["exact_internal_overlap"]).sum()
        ),
        "exact_primary_training_overlap_rows": int(
            audit["exact_primary_training_overlap"].sum()
        ),
        "external_rows_without_exact_primary_training_overlap": int(
            (~audit["exact_primary_training_overlap"]).sum()
        ),
        "training_molecules_to_exclude": int(len(overlapping_ids)),
        "primary_training_molecules_before_exclusion": int(len(target_ids)),
        "primary_training_molecules_after_exclusion": int(
            len(target_ids.difference(overlapping_ids))
        ),
        "labels_separated_from_inputs": True,
        "labels_joined_to_internal_master": False,
        "archive_training_table_used": False,
        "archive_training_table_policy": (
            "datasets/Training_sets.csv is intentionally ignored because the repository already "
            "has a fixed internal training corpus and the SC2019 tight/loose sets are reserved "
            "for external evaluation."
        ),
        "overlap_policy": (
            "Every exact standardized-parent match listed in sc2019_training_exclusions.csv "
            "must be removed from all future candidate-model fitting before external evaluation."
        ),
    }
    report["output_files"] = {
        key: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _file_digest(path, "sha256"),
        }
        for key, path in paths.items()
        if key not in {"report_json", "report_csv"}
    }
    paths["report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["report_csv"], index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, standardize and overlap-audit the external SC2019 test sets."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use a local datasets.tar.gz instead of the verified Zenodo download cache.",
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    archive = (
        args.archive.resolve()
        if args.archive is not None
        else root / "external_data/sc2019/raw/datasets.tar.gz"
    )
    if args.archive is None:
        archive = download_sc2019_archive(archive, force=args.force_download)
    elif not archive.exists():
        raise FileNotFoundError(f"SC2019 archive not found: {archive}")
    required = {
        "master": root / "data_processed/master_observations.parquet",
        "targets": root / "data_processed/modeling_targets.parquet",
        "morgan": root / "data_processed/morgan_fingerprints.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Internal data products required for overlap auditing are missing:\n- "
            + "\n- ".join(missing)
        )
    paths = build_sc2019_artifacts(
        archive,
        pd.read_parquet(required["master"]),
        pd.read_parquet(required["targets"]),
        pd.read_parquet(required["morgan"]),
        root / "external_data/sc2019",
        root / "reports",
    )
    print("SC2019 external-data import completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
