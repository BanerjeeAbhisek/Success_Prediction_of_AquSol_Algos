from aquasol_meta.standardize import standardize_smiles


def test_standardization_retains_full_form_and_creates_fragment_parent() -> None:
    result = standardize_smiles("CC(=O)[O-].[Na+]")

    assert result["structure_status"] == "ok"
    assert result["fragment_count"] == 2
    assert bool(result["has_multiple_fragments"])
    assert "." in str(result["canonical_smiles_full"])
    assert "." not in str(result["canonical_smiles_parent"])
    assert result["molecule_id"]


def test_invalid_smiles_is_reported_without_crashing() -> None:
    result = standardize_smiles("not-a-smiles")

    assert result["structure_status"] == "invalid_smiles"
    assert result["molecule_id"] is not None

