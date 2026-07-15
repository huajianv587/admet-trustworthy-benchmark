from __future__ import annotations

import numpy as np


def featurizable_indices(smiles_list: list[str]) -> list[int]:
    """Return the row indices accepted by the RDKit molecule parser.

    This is the inexpensive identity-only counterpart to
    :func:`morgan_count_matrix`.  Artifact audits use it to reconstruct the
    exact rows retained by the prediction runner without calculating hundreds
    of thousands of fingerprints a second time.
    """

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for molecular featurization.") from exc

    return [
        idx
        for idx, smiles in enumerate(smiles_list)
        if Chem.MolFromSmiles(smiles) is not None
    ]


def morgan_count_matrix(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[int]]:
    """Convert SMILES to a binary Morgan fingerprint matrix.

    The historical function name contains ``count`` for API compatibility;
    the returned 2,048-dimensional values are binary bits, not occurrence
    counts.

    Returns the feature matrix and indices that were successfully featurized.
    Invalid molecules are skipped and must be mirrored in labels by caller.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("RDKit is required for Morgan fingerprints.") from exc

    rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        kept_indices.append(idx)
    if not rows:
        raise ValueError("No valid molecules could be featurized.")
    return np.vstack(rows), kept_indices


def align_labels(labels, kept_indices: list[int]) -> np.ndarray:
    values = np.asarray(labels)
    return values[np.asarray(kept_indices, dtype=int)]
