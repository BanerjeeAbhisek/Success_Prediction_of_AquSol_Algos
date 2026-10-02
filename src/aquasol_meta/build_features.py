from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import (
    build_feature_report,
    build_feature_tables,
    validate_feature_tables,
)


def _readback_errors(
    paths: dict[str, Path], expected_ids: pd.Series, write_csv: bool
) -> list[str]:
    errors: list[str] = []
    expected = set(expected_ids.astype(str))
    for name in ("rdkit_descriptors", "morgan_fingerprints", "maccs_fingerprints"):
        frame = pd.read_parquet(paths[name], columns=["molecule_id"])
        actual = set(frame["molecule_id"].astype(str))
        if actual != expected:
            errors.append(f"{name} Parquet read-back molecule IDs differ from target table")
        if len(frame) != len(expected):
            errors.append(
                f"{name} Parquet read-back has {len(frame):,} rows; expected {len(expected):,}"
            )

        if write_csv:
            csv_name = f"{name}_csv"
            csv_frame = pd.read_csv(
                paths[csv_name], usecols=["molecule_id"], dtype={"molecule_id": "string"}
            )
            csv_actual = set(csv_frame["molecule_id"].astype(str))
            if csv_actual != expected:
                errors.append(f"{name} CSV read-back molecule IDs differ from target table")
            if len(csv_frame) != len(expected):
                errors.append(
                    f"{name} CSV read-back has {len(csv_frame):,} rows; "
                    f"expected {len(expected):,}"
                )
    return errors


def _flatten_report(report: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def visit(path: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}.{key}" if path else str(key), item)
        elif isinstance(value, list):
            if not value:
                rows.append({"metric": path, "value": "[]"})
            for index, item in enumerate(value):
                item_path = f"{path}.{index}" if isinstance(item, dict) else path
                visit(item_path, item)
        else:
            rows.append({"metric": path, "value": value})

    visit("", report)
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_feature_artifacts(
    targets: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    include_chirality: bool = True,
    write_csv: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tables = build_feature_tables(
        targets,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        include_chirality=include_chirality,
    )
    expected_ids = targets["molecule_id"].astype(str)
    errors = validate_feature_tables(tables, expected_ids)

    paths = {
        "rdkit_descriptors": output_dir / "rdkit_descriptors.parquet",
        "morgan_fingerprints": output_dir / "morgan_fingerprints.parquet",
        "maccs_fingerprints": output_dir / "maccs_fingerprints.parquet",
        "feature_quality": reports_dir / "feature_quality.json",
        "feature_quality_csv": reports_dir / "feature_quality.csv",
    }
    if write_csv:
        paths.update(
            {
                "rdkit_descriptors_csv": output_dir / "rdkit_descriptors.csv",
                "morgan_fingerprints_csv": output_dir / "morgan_fingerprints.csv",
                "maccs_fingerprints_csv": output_dir / "maccs_fingerprints.csv",
            }
        )

    if not errors:
        tables.descriptors.to_parquet(
            paths["rdkit_descriptors"], index=False, compression="zstd"
        )
        tables.morgan.to_parquet(
            paths["morgan_fingerprints"], index=False, compression="zstd"
        )
        tables.maccs.to_parquet(
            paths["maccs_fingerprints"], index=False, compression="zstd"
        )
        if write_csv:
            tables.descriptors.to_csv(paths["rdkit_descriptors_csv"], index=False)
            tables.morgan.to_csv(paths["morgan_fingerprints_csv"], index=False)
            tables.maccs.to_csv(paths["maccs_fingerprints_csv"], index=False)
        errors.extend(_readback_errors(paths, expected_ids, write_csv=write_csv))

    report = build_feature_report(
        tables,
        targets=targets,
        morgan_radius=morgan_radius,
        morgan_n_bits=morgan_n_bits,
        include_chirality=include_chirality,
        errors=errors,
    )
    report["output_files"] = {
        name: {"path": str(path), "bytes": path.stat().st_size if path.exists() else None}
        for name, path in paths.items()
        if name not in {"feature_quality", "feature_quality_csv"}
    }
    paths["feature_quality"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["feature_quality_csv"], index=False)

    if errors:
        joined = "\n- ".join(errors)
        raise RuntimeError(f"Molecular-feature validation failed:\n- {joined}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RDKit descriptors and fingerprints for standardized molecules."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-n-bits", type=int, default=2048)
    parser.add_argument(
        "--ignore-chirality",
        action="store_true",
        help="Generate Morgan fingerprints without chirality information.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write Parquet and the JSON/CSV quality reports, but omit feature CSV copies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    targets_path = (args.targets or root / "data_processed/modeling_targets.parquet").resolve()
    output_dir = (args.output_dir or root / "data_processed").resolve()
    reports_dir = (args.reports_dir or root / "reports").resolve()

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Modeling targets not found: {targets_path}. Run aquasol-build-targets first."
        )

    targets = pd.read_parquet(targets_path)
    paths = build_feature_artifacts(
        targets,
        output_dir=output_dir,
        reports_dir=reports_dir,
        morgan_radius=args.morgan_radius,
        morgan_n_bits=args.morgan_n_bits,
        include_chirality=not args.ignore_chirality,
        write_csv=not args.no_csv,
    )
    print("Molecular-feature build completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
