import numpy as np
import pandas as pd

from aquasol_meta.ingest import harmonize_ochem


def test_ochem_converted_minus_log_m_is_negated() -> None:
    raw = pd.DataFrame(
        {
            "SMILES": ["CCO", "CC"],
            "CASRN": ["64-17-5", "74-84-0"],
            "Water solubility {measured}": [1.2, 3.4],
            "UNIT {Water solubility}": ["log(mol/L)", "mg/L"],
            "Water solubility {measured, converted}": [-1.2, 2.5],
            "UNIT {Water solubility}.1": ["-log(M)", "-log(M)"],
        }
    )

    result = harmonize_ochem(raw)

    np.testing.assert_allclose(result["log_s"].to_numpy(dtype=float), [1.2, -2.5])
    assert set(result["conversion_rule"]) == {
        "log_s = -1 * converted_minus_log10_molar"
    }
    assert list(result["target_original_unit"]) == ["log(mol/L)", "mg/L"]

