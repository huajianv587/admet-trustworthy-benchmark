from __future__ import annotations

from typing import Iterable

import numpy as np

from .smiles_compat import model_input_smiles


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("RDKit is required for molecular fingerprints.") from exc

    # Fingerprinting is a model-input operation. Apply the audited parser-only
    # repair while callers continue to retain and hash the official source
    # SMILES for row identity and release provenance.
    mol = Chem.MolFromSmiles(model_input_smiles(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def max_tanimoto_similarity(test_fp, train_fps: Iterable) -> float:
    try:
        from rdkit import DataStructs
    except ImportError as exc:
        raise ImportError("RDKit is required for Tanimoto similarity.") from exc

    sims = DataStructs.BulkTanimotoSimilarity(test_fp, list(train_fps))
    return float(max(sims)) if sims else float("nan")


def compute_max_similarities(test_smiles: list[str], train_smiles: list[str]) -> np.ndarray:
    train_fps = [morgan_fingerprint(smi) for smi in train_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    values = []
    for smi in test_smiles:
        fp = morgan_fingerprint(smi)
        values.append(max_tanimoto_similarity(fp, train_fps) if fp is not None else float("nan"))
    return np.asarray(values, dtype=float)


def bin_by_similarity(
    similarities: np.ndarray,
    values: np.ndarray,
    bins: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0),
) -> list[dict[str, float]]:
    similarities = np.asarray(similarities, dtype=float)
    values = np.asarray(values, dtype=float)
    rows: list[dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (similarities >= left) & (similarities <= right)
        else:
            mask = (similarities >= left) & (similarities < right)
        rows.append(
            {
                "similarity_left": float(left),
                "similarity_right": float(right),
                "count": int(np.sum(mask)),
                "mean_value": float(np.mean(values[mask])) if np.any(mask) else float("nan"),
                "mean_similarity": float(np.mean(similarities[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows
