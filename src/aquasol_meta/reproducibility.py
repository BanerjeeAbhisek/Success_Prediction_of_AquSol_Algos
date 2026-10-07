from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .run_baselines import _flatten_report

REPORT_NAMES = {
    "artifact_checksums.csv",
    "reproducibility_manifest.csv",
    "reproducibility_manifest.json",
}
ROOT_FILES = (
    ".gitignore",
    "environment.yml",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "requirements-lock.txt",
)
INCLUDED_DIRECTORIES = (
    "Data",
    "data_processed",
    "reports",
    "results",
    "scripts",
    "src",
    "tests",
)
EXCLUDED_PARTS = {
    ".DS_Store",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "chemprop_cache",
    "source_holdout_prediction_failures.parquet",
}
KEY_TABLES = {
    "master_observations": "data_processed/master_observations.parquet",
    "modeling_targets": "data_processed/modeling_targets.parquet",
    "split_assignments": "data_processed/split_assignments.parquet",
    "source_holdout_tasks": "data_processed/source_holdout_tasks.parquet",
    "model_results": "results/model_results.parquet",
    "test_predictions": "results/test_predictions.parquet",
    "prediction_failures": "results/prediction_failures.parquet",
    "run_failure_summary": "results/run_failure_summary.parquet",
    "within_meta_dataset": "data_processed/meta_model_dataset.parquet",
    "source_meta_dataset": "data_processed/source_holdout_meta_model_dataset.parquet",
    "combined_meta_dataset": (
        "data_processed/meta_model_dataset_with_source_holdouts.parquet"
    ),
    "meta_model_oos_predictions": "results/meta_model_oos_predictions.parquet",
    "meta_model_selected_candidates": "results/meta_model_selected_candidates.parquet",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included_files(root: Path) -> list[Path]:
    candidates = [root / name for name in ROOT_FILES]
    for directory_name in INCLUDED_DIRECTORIES:
        directory = root / directory_name
        if directory.exists():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    included: list[Path] = []
    for path in candidates:
        if not path.exists() or path.name in REPORT_NAMES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        included.append(path)
    return sorted(set(included), key=lambda path: path.relative_to(root).as_posix())


def _checksums(root: Path) -> pd.DataFrame:
    rows = []
    for path in _included_files(root):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def _locked_packages(root: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, version = stripped.split("==", maxsplit=1)
        packages[name] = version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _table_counts(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name, relative in KEY_TABLES.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Required reproducibility table is missing: {path}")
        frame = pd.read_parquet(path)
        counts[name] = {"rows": int(len(frame)), "columns": int(len(frame.columns))}
    return counts


def _experiment_summary(root: Path) -> dict[str, Any]:
    results = pd.read_parquet(root / KEY_TABLES["model_results"])
    predictions = pd.read_parquet(root / KEY_TABLES["test_predictions"])
    within_meta = pd.read_parquet(root / KEY_TABLES["within_meta_dataset"])
    source_meta = pd.read_parquet(root / KEY_TABLES["source_meta_dataset"])
    combined_meta = pd.read_parquet(root / KEY_TABLES["combined_meta_dataset"])
    assignments = pd.read_parquet(
        root / KEY_TABLES["split_assignments"],
        columns=["split_strategy", "repeat", "seed"],
    )
    source_tasks = pd.read_parquet(
        root / KEY_TABLES["source_holdout_tasks"],
        columns=["task_id", "heldout_source", "repeat", "seed"],
    )
    meta_predictions = pd.read_parquet(root / KEY_TABLES["meta_model_oos_predictions"])
    meta_selected = pd.read_parquet(root / KEY_TABLES["meta_model_selected_candidates"])
    if results["run_id"].duplicated().any():
        raise RuntimeError("Canonical model results contain duplicate run IDs")
    if predictions.duplicated(["run_id", "molecule_id"]).any():
        raise RuntimeError("Canonical predictions contain duplicate run/molecule rows")
    if set(results["run_id"]) != set(within_meta["run_id"]):
        raise RuntimeError("Canonical results and within meta-dataset cover different runs")
    if meta_predictions.duplicated(["meta_model", "run_id"]).any():
        raise RuntimeError("Meta-model results contain duplicate out-of-source predictions")
    if meta_selected.duplicated(["selector", "task_id"]).any():
        raise RuntimeError("A meta-model selector chose multiple candidates for one task")
    chemprop_parameters = sorted(
        results.loc[results["model"].eq("chemprop"), "best_parameters"].unique()
    )
    return {
        "within_model_runs": int(len(results)),
        "within_prediction_rows": int(len(predictions)),
        "within_tasks": int(within_meta["task_id"].nunique()),
        "source_model_runs": int(len(source_meta)),
        "source_tasks": int(source_meta["task_id"].nunique()),
        "combined_meta_rows": int(len(combined_meta)),
        "combined_tasks": int(combined_meta["task_id"].nunique()),
        "model_run_counts": {
            str(key): int(value)
            for key, value in results["model"].value_counts().sort_index().items()
        },
        "feature_types": sorted(str(value) for value in results["feature_type"].unique()),
        "split_strategies": sorted(
            str(value) for value in assignments["split_strategy"].unique()
        ),
        "within_repeats": sorted(int(value) for value in assignments["repeat"].unique()),
        "within_seeds": sorted(int(value) for value in assignments["seed"].unique()),
        "source_repeats": sorted(int(value) for value in source_tasks["repeat"].unique()),
        "source_seeds": sorted(int(value) for value in source_tasks["seed"].unique()),
        "heldout_sources": sorted(
            str(value) for value in source_tasks["heldout_source"].unique()
        ),
        "meta_model_oos_candidate_rows": int(len(meta_predictions)),
        "meta_model_names": sorted(
            str(value) for value in meta_predictions["meta_model"].unique()
        ),
        "meta_model_outer_sources": sorted(
            str(value) for value in meta_predictions["outer_heldout_source"].unique()
        ),
        "meta_model_selected_rows": int(len(meta_selected)),
        "meta_model_selectors": sorted(
            str(value) for value in meta_selected["selector"].unique()
        ),
        "chemprop_parameters": [json.loads(value) for value in chemprop_parameters],
    }


def build_reproducibility_artifacts(root: Path) -> dict[str, Path]:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checksums = _checksums(root)
    checksum_path = reports_dir / "artifact_checksums.csv"
    manifest_path = reports_dir / "reproducibility_manifest.json"
    checksums.to_csv(checksum_path, index=False)
    oversized = checksums.loc[checksums["bytes"] >= 95_000_000, "path"].tolist()
    failure_report = json.loads(
        (reports_dir / "failure_analysis_report.json").read_text(encoding="utf-8")
    )
    manifest = {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if not oversized else "fail",
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "locked_packages": _locked_packages(root),
        "table_counts": _table_counts(root),
        "experiment": _experiment_summary(root),
        "failure_definition": failure_report["q_i_definition"],
        "failure_thresholds_log_s": failure_report["failure_thresholds_log_s"],
        "checksum_algorithm": "SHA-256",
        "checksum_file": checksum_path.relative_to(root).as_posix(),
        "checksummed_file_count": int(len(checksums)),
        "checksummed_bytes": int(checksums["bytes"].sum()),
        "github_safety_threshold_bytes": 95_000_000,
        "files_at_or_above_safety_threshold": oversized,
        "pipeline_script": "scripts/run_reproducible_pipeline.sh",
        "notes": [
            "Chemprop cache files and checkpoints are excluded because they are regenerable.",
            (
                "The Git-ignored partitioned source prediction dataset is regenerable "
                "and excluded from checksums."
            ),
            "Neural-network results may not be bit-identical across hardware backends.",
            "Meta-model assessment holds out complete experimental sources.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(manifest).to_csv(
        reports_dir / "reproducibility_manifest.csv", index=False
    )
    if oversized:
        raise RuntimeError(
            "Files exceed the 95 MB GitHub safety threshold:\n- " + "\n- ".join(oversized)
        )
    return {
        "manifest_json": manifest_path,
        "manifest_csv": reports_dir / "reproducibility_manifest.csv",
        "checksums_csv": checksum_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record package versions, experiment counts, seeds, and SHA-256 checksums."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_reproducibility_artifacts(args.root.resolve())
    print("Reproducibility snapshot completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
