from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_SPECS

MASTER_COLUMNS = [
    "observation_id",
    "source_dataset",
    "source_file",
    "source_row",
    "source_record_id",
    "original_smiles",
    "source_inchi",
    "source_inchi_key",
    "molecule_name",
    "casrn",
    "log_s",
    "target_original_value",
    "target_original_unit",
    "target_source_column",
    "conversion_rule",
    "endpoint_definition",
    "temperature",
    "temperature_unit",
    "ph",
    "ph_unit",
    "ionic_strength",
    "ionic_strength_unit",
    "measurement_sd",
    "replicate_count",
    "article_id",
    "pubmed_id",
    "dataset_partition",
    "quality_code",
    "source_group",
    "source_comment",
    "reference_prediction",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a source CSV without silently discarding undecodable bytes."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        # OCHEM contains legacy single-byte characters that are not valid UTF-8.
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def _blank(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="object")


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else _blank(df.index)


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(df, name), errors="coerce")


def _base(df: pd.DataFrame, dataset: str, filename: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = dataset
    out["source_file"] = filename
    out["source_row"] = np.arange(1, len(df) + 1, dtype=int)
    out["observation_id"] = [f"{dataset}:{row:07d}" for row in out["source_row"]]
    for column in MASTER_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[MASTER_COLUMNS]


def harmonize_aqsoldb(df: pd.DataFrame, filename: str = "AqSolDB.csv") -> pd.DataFrame:
    out = _base(df, "AqSolDB", filename)
    out["source_record_id"] = _column(df, "ID").astype("string")
    out["original_smiles"] = _column(df, "SMILES").astype("string")
    out["source_inchi"] = _column(df, "InChI").astype("string")
    out["source_inchi_key"] = _column(df, "InChIKey").astype("string")
    out["molecule_name"] = _column(df, "Name").astype("string")
    out["log_s"] = _numeric(df, "Solubility")
    out["target_original_value"] = _column(df, "Solubility")
    out["target_original_unit"] = "log10(mol/L)"
    out["target_source_column"] = "Solubility"
    out["conversion_rule"] = "identity"
    out["endpoint_definition"] = "not provided in repository"
    out["measurement_sd"] = _numeric(df, "SD")
    out["replicate_count"] = _numeric(df, "Ocurrences")
    out["source_group"] = _column(df, "Group").astype("string")
    return out


def harmonize_delaney(df: pd.DataFrame, filename: str = "Delaney.csv") -> pd.DataFrame:
    target = "measured log solubility in mols per litre"
    out = _base(df, "Delaney", filename)
    out["source_record_id"] = _column(df, "Compound ID").astype("string")
    out["original_smiles"] = _column(df, "smiles").astype("string").str.strip()
    out["molecule_name"] = _column(df, "Compound ID").astype("string")
    out["log_s"] = _numeric(df, target)
    out["target_original_value"] = _column(df, target)
    out["target_original_unit"] = "log10(mol/L)"
    out["target_source_column"] = target
    out["conversion_rule"] = "identity"
    out["endpoint_definition"] = "aqueous solubility; repository gives no pH metadata"
    out["reference_prediction"] = _numeric(
        df, "ESOL predicted log solubility in mols per litre"
    )
    return out


def _harmonize_two_column(
    df: pd.DataFrame,
    dataset: str,
    filename: str,
) -> pd.DataFrame:
    out = _base(df, dataset, filename)
    out["source_record_id"] = [f"{dataset}-{row:07d}" for row in out["source_row"]]
    out["original_smiles"] = _column(df, "smiles").astype("string").str.strip()
    out["log_s"] = _numeric(df, "logS")
    out["target_original_value"] = _column(df, "logS")
    out["target_original_unit"] = "log10(mol/L)"
    out["target_source_column"] = "logS"
    out["conversion_rule"] = "identity"
    out["endpoint_definition"] = "not provided in repository"
    return out


def harmonize_aqua(df: pd.DataFrame, filename: str = "aqua_org.csv") -> pd.DataFrame:
    return _harmonize_two_column(df, "AQUA", filename)


def harmonize_physprop(df: pd.DataFrame, filename: str = "phys_org.csv") -> pd.DataFrame:
    return _harmonize_two_column(df, "PHYSPROP", filename)


def harmonize_ochem(df: pd.DataFrame, filename: str = "ochem_all.csv") -> pd.DataFrame:
    converted_column = "Water solubility {measured, converted}"
    converted_minus_log_m = _numeric(df, converted_column)

    out = _base(df, "OCHEM", filename)
    out["source_record_id"] = _column(df, "EXTERNALID").astype("string")
    missing_record_id = out["source_record_id"].isna() | (out["source_record_id"] == "")
    out.loc[missing_record_id, "source_record_id"] = (
        _column(df, "CASRN").astype("string").loc[missing_record_id]
    )
    still_missing = out["source_record_id"].isna() | (out["source_record_id"] == "")
    out.loc[still_missing, "source_record_id"] = [
        f"OCHEM-{row:07d}" for row in out.loc[still_missing, "source_row"]
    ]

    out["original_smiles"] = _column(df, "SMILES").astype("string").str.strip()
    out["molecule_name"] = _column(df, "NAME").astype("string")
    out["casrn"] = _column(df, "CASRN").astype("string")

    # OCHEM's harmonized column is -log10(mol/L). The common project endpoint is logS.
    out["log_s"] = -converted_minus_log_m
    out["target_original_value"] = _column(df, "Water solubility {measured}")
    out["target_original_unit"] = _column(df, "UNIT {Water solubility}")
    out["target_source_column"] = converted_column
    out["conversion_rule"] = "log_s = -1 * converted_minus_log10_molar"
    out["endpoint_definition"] = "water solubility; intrinsic/apparent status not provided"

    out["temperature"] = _numeric(df, "Temperature")
    out["temperature_unit"] = _column(df, "UNIT {Temperature}").astype("string")
    out["ph"] = _numeric(df, "pH")
    out["ph_unit"] = _column(df, "UNIT {pH}").astype("string")
    out["ionic_strength"] = _numeric(df, "Ionic strength")
    out["ionic_strength_unit"] = _column(df, "UNIT {Ionic strength}").astype("string")
    out["article_id"] = _column(df, "ARTICLEID").astype("string")
    out["pubmed_id"] = _column(df, "PUBMEDID").astype("string")
    out["dataset_partition"] = _column(df, "Dataset").astype("string")
    out["quality_code"] = _column(df, "Quality code").astype("string")
    out["source_comment"] = _column(df, "comment (chemical)").astype("string")
    return out


HARMONIZERS: dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
    "AqSolDB": harmonize_aqsoldb,
    "Delaney": harmonize_delaney,
    "AQUA": harmonize_aqua,
    "PHYSPROP": harmonize_physprop,
    "OCHEM": harmonize_ochem,
}


def load_all_sources(raw_dir: Path, check_expected_rows: bool = True) -> pd.DataFrame:
    """Load all source files into a common observation-level schema."""
    frames: list[pd.DataFrame] = []
    for spec in DATASET_SPECS:
        path = raw_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(f"Required source file not found: {path}")
        raw = read_csv_robust(path)
        if check_expected_rows and len(raw) != spec.expected_rows:
            raise ValueError(
                f"{spec.name} row count changed: expected {spec.expected_rows:,}, "
                f"found {len(raw):,}. Review the source before continuing."
            )
        frames.append(HARMONIZERS[spec.name](raw, spec.filename))

    master = pd.concat(frames, ignore_index=True)
    master["log_s"] = pd.to_numeric(master["log_s"], errors="coerce")
    master.loc[~np.isfinite(master["log_s"]), "log_s"] = np.nan
    return master
