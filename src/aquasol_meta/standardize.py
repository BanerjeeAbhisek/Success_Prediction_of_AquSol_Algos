from __future__ import annotations

import sys
from typing import Any

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
    Chem = None
    Descriptors = None
    rdMolStandardize = None
    MurckoScaffold = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None


STRUCTURE_COLUMNS = [
    "structure_status",
    "structure_error",
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


def require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RDKit is required for structure standardization. "
            "Create the project environment with `conda env create -f environment.yml`."
        ) from RDKIT_IMPORT_ERROR


def _inchi_key(mol: Any) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def standardize_smiles(smiles: object) -> dict[str, object]:
    """Create stable full-structure and largest-parent identifiers.

    The full canonical structure is retained. A separate fragment-parent form is generated for
    overlap detection and later modelling. Charges and tautomers are not forcibly neutralized or
    canonicalized because those transformations can change the solubility-relevant chemical form.
    """
    require_rdkit()
    result: dict[str, object] = {column: pd.NA for column in STRUCTURE_COLUMNS}

    if smiles is None or pd.isna(smiles) or not str(smiles).strip():
        result["structure_status"] = "missing_smiles"
        result["structure_error"] = "No SMILES was provided"
        return result

    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        result["structure_status"] = "invalid_smiles"
        result["structure_error"] = "RDKit could not parse the SMILES"
        return result

    try:
        canonical_full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        parent = rdMolStandardize.FragmentParent(mol)
        parent.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(parent)
        Chem.GetSymmSSSR(parent)
        canonical_parent = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        full_key = _inchi_key(mol)
        parent_key = _inchi_key(parent)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=True)

        result.update(
            {
                "structure_status": "ok",
                "structure_error": pd.NA,
                "canonical_smiles_full": canonical_full,
                "canonical_smiles_parent": canonical_parent,
                "full_inchi_key": full_key,
                "parent_inchi_key": parent_key,
                "molecule_id": parent_key or f"SMILES:{canonical_parent}",
                "murcko_scaffold": scaffold or "[NO_SCAFFOLD]",
                "fragment_count": len(fragments),
                "has_multiple_fragments": len(fragments) > 1,
                "formal_charge_full": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                "heavy_atom_count_parent": int(parent.GetNumHeavyAtoms()),
                "molecular_weight_parent": float(Descriptors.MolWt(parent)),
            }
        )
    except Exception as exc:
        result["structure_status"] = "standardization_error"
        result["structure_error"] = f"{type(exc).__name__}: {exc}"
    return result


def standardize_dataframe(
    observations: pd.DataFrame,
    smiles_column: str = "original_smiles",
    progress_every: int = 5_000,
) -> pd.DataFrame:
    """Append RDKit-derived structure fields without altering the original SMILES."""
    require_rdkit()
    records: list[dict[str, object]] = []
    total = len(observations)
    for position, smiles in enumerate(observations[smiles_column], start=1):
        records.append(standardize_smiles(smiles))
        if progress_every and position % progress_every == 0:
            print(f"Standardized {position:,}/{total:,} observations", file=sys.stderr)

    structure = pd.DataFrame.from_records(records, columns=STRUCTURE_COLUMNS)
    structure.index = observations.index
    return pd.concat([observations.copy(), structure], axis=1)
