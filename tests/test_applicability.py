import numpy as np
import pytest

from admet_benchmark.applicability import compute_max_similarities
from admet_benchmark.smiles_compat import (
    LEGACY_VALENCE_REPAIRS,
    model_input_smiles,
)


@pytest.mark.parametrize("source", sorted(LEGACY_VALENCE_REPAIRS))
def test_legacy_source_smiles_are_fingerprintable_without_changing_identity(source: str):
    pytest.importorskip("rdkit")
    repaired = model_input_smiles(source)
    similarities = compute_max_similarities([source], [repaired])
    assert np.isfinite(similarities).all()
    assert np.isclose(similarities[0], 1.0)

