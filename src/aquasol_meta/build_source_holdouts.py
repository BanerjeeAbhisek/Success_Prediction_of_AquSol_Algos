from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_baselines import _flatten_report, _write_table
from .source_holdout import build_source_holdout_tasks
from .splits import DEFAULT_SPLIT_SEEDS


def build_source_holdout_artifacts(
    observations: pd.DataFrame,
    morgan: pd.DataFrame,
    data_dir: Path,
    reports_dir: Path,
    sources: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tasks, summary, report = build_source_holdout_tasks(
        observations, morgan, sources=sources, seeds=seeds
    )
    paths = {
        "source_holdout_tasks_csv": data_dir / "source_holdout_tasks.csv",
        "source_holdout_tasks_parquet": data_dir / "source_holdout_tasks.parquet",
        "source_holdout_summary_csv": reports_dir / "source_holdout_summary.csv",
        "source_holdout_report_json": reports_dir / "source_holdout_report.json",
        "source_holdout_report_csv": reports_dir / "source_holdout_report.csv",
    }
    _write_table(
        tasks,
        paths["source_holdout_tasks_csv"],
        paths["source_holdout_tasks_parquet"],
    )
    summary.to_csv(paths["source_holdout_summary_csv"], index=False)
    report["output_files"] = {
        name: {"path": str(path), "bytes": int(path.stat().st_size)}
        for name, path in paths.items()
        if not name.startswith("source_holdout_report")
    }
    paths["source_holdout_report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _flatten_report(report).to_csv(paths["source_holdout_report_csv"], index=False)
    if report["blocking_errors"]:
        raise RuntimeError(
            "Source-holdout task construction failed:\n- "
            + "\n- ".join(report["blocking_errors"])
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe source-holdout aqueous-solubility tasks."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    observations_path = root / "data_processed/master_observations.parquet"
    morgan_path = root / "data_processed/morgan_fingerprints.parquet"
    missing = [str(path) for path in (observations_path, morgan_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required source-holdout inputs are missing:\n- " + "\n- ".join(missing)
        )
    paths = build_source_holdout_artifacts(
        observations=pd.read_parquet(observations_path),
        morgan=pd.read_parquet(morgan_path),
        data_dir=root / "data_processed",
        reports_dir=root / "reports",
        sources=tuple(args.sources) if args.sources else None,
        seeds=tuple(args.seeds),
    )
    print("Source-holdout task construction completed.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
