from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_targets import build_target_artifacts
from .deduplicate import build_molecule_summary, build_source_overlap
from .ingest import load_all_sources
from .standardize import standardize_dataframe
from .validate_data import validate_master


def _dataset_summary(report: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_source = report["observations_by_source"]
    assert isinstance(by_source, dict)
    for source, metrics in by_source.items():
        row = {"source_dataset": source}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)


def _normalize_output_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_columns = {
        "source_row",
        "log_s",
        "temperature",
        "ph",
        "ionic_strength",
        "measurement_sd",
        "replicate_count",
        "reference_prediction",
        "fragment_count",
        "formal_charge_full",
        "heavy_atom_count_parent",
        "molecular_weight_parent",
    }
    boolean_columns = {"has_multiple_fragments", "model_eligible"}
    for column in out.columns:
        if column in numeric_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        elif column in boolean_columns:
            out[column] = out[column].astype("boolean")
        else:
            out[column] = out[column].astype("string")
    return out


def build_master_dataset(
    raw_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    write_csv: bool = True,
) -> dict[str, Path]:
    """Build all curated data products and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    observations = load_all_sources(raw_dir)
    observations = standardize_dataframe(observations)
    observations["model_eligible"] = (
        observations["structure_status"].eq("ok")
        & observations["log_s"].notna()
        & np.isfinite(observations["log_s"])
    )
    observations = _normalize_output_dtypes(observations)

    molecule_summary = build_molecule_summary(observations)
    overlap = build_source_overlap(observations)
    report, errors = validate_master(observations)
    dataset_summary = _dataset_summary(report)

    paths = {
        "master_parquet": output_dir / "master_observations.parquet",
        "molecule_parquet": output_dir / "molecule_summary.parquet",
        "validation_report": reports_dir / "data_validation.json",
        "dataset_summary": reports_dir / "dataset_summary.csv",
        "source_overlap": reports_dir / "source_overlap.csv",
    }

    observations.to_parquet(paths["master_parquet"], index=False)
    molecule_summary.to_parquet(paths["molecule_parquet"], index=False)
    dataset_summary.to_csv(paths["dataset_summary"], index=False)
    overlap.to_csv(paths["source_overlap"], index=False)
    paths["validation_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if write_csv:
        paths["master_csv"] = output_dir / "master_observations.csv"
        paths["molecule_csv"] = output_dir / "molecule_summary.csv"
        observations.to_csv(paths["master_csv"], index=False)
        molecule_summary.to_csv(paths["molecule_csv"], index=False)

    if errors:
        joined = "\n- ".join(errors)
        raise RuntimeError(f"Master-data validation failed:\n- {joined}")

    paths.update(
        build_target_artifacts(
            observations=observations,
            output_dir=output_dir,
            reports_dir=reports_dir,
            write_csv=write_csv,
        )
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the observation-level aqueous-solubility master dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write Parquet and reports only; omit the larger CSV copies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_dir = (args.raw_dir or root / "Data").resolve()
    output_dir = (args.output_dir or root / "data_processed").resolve()
    reports_dir = (args.reports_dir or root / "reports").resolve()

    paths = build_master_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        reports_dir=reports_dir,
        write_csv=not args.no_csv,
    )
    print("Master-data build completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
