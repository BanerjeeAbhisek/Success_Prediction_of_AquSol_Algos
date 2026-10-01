from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    expected_rows: int


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("AqSolDB", "AqSolDB.csv", 9_982),
    DatasetSpec("Delaney", "Delaney.csv", 1_128),
    DatasetSpec("AQUA", "aqua_org.csv", 1_311),
    DatasetSpec("PHYSPROP", "phys_org.csv", 2_010),
    DatasetSpec("OCHEM", "ochem_all.csv", 36_449),
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "Data"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data_processed"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

