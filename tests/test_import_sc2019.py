import io
import tarfile

import numpy as np
import pandas as pd

from aquasol_meta.import_sc2019 import (
    _nearest_internal_similarity,
    _read_challenge_table,
)


def test_read_challenge_table_uses_second_header_and_four_source_fields(tmp_path) -> None:
    archive_path = tmp_path / "challenge.tar.gz"
    content = (
        b"metadata,metadata,metadata,metadata\n"
        b"ID,Name,SMILES,Solubility\n"
        b"mol-1,Example,CCO,-1.25\n"
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("datasets/Tight_set.csv")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with tarfile.open(archive_path, "r:gz") as archive:
        frame = _read_challenge_table(
            archive, "sc2019_tight", "datasets/Tight_set.csv"
        )

    assert frame.columns.tolist() == [
        "external_id",
        "external_task",
        "measurement_quality",
        "compound_name",
        "original_smiles",
        "intrinsic_log_s",
    ]
    assert frame.loc[0, "external_id"] == "mol-1"
    assert frame.loc[0, "intrinsic_log_s"] == -1.25


def test_nearest_internal_similarity_identifies_exact_and_partial_matches() -> None:
    external = pd.DataFrame(
        {
            "molecule_id": ["external_exact", "external_partial"],
            "morgan_0000": np.array([1, 1], dtype=np.uint8),
            "morgan_0001": np.array([0, 1], dtype=np.uint8),
            "morgan_0002": np.array([1, 0], dtype=np.uint8),
        }
    )
    internal = pd.DataFrame(
        {
            "molecule_id": ["internal_a", "internal_b"],
            "morgan_0000": np.array([1, 0], dtype=np.uint8),
            "morgan_0001": np.array([0, 1], dtype=np.uint8),
            "morgan_0002": np.array([1, 0], dtype=np.uint8),
        }
    )

    similarity = _nearest_internal_similarity(external, internal)

    assert similarity.loc[0, "nearest_primary_training_molecule_id"] == "internal_a"
    assert similarity.loc[0, "maximum_primary_training_tanimoto_similarity"] == 1.0
    assert similarity.loc[1, "maximum_primary_training_tanimoto_similarity"] == 0.5
