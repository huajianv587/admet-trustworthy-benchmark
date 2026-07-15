from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import clone


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for SMILES canonicalization.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def audit_dataset(smiles_list: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total": len(smiles_list),
        "invalid": 0,
        "salts_or_mixtures": 0,
        "duplicates": 0,
        "unique_canonical": 0,
    }
    seen: set[str] = set()
    for smiles in smiles_list:
        if "." in str(smiles):
            report["salts_or_mixtures"] += 1
        canonical = canonicalize_smiles(str(smiles))
        if canonical is None:
            report["invalid"] += 1
            continue
        if canonical in seen:
            report["duplicates"] += 1
        seen.add(canonical)
    report["unique_canonical"] = len(seen)
    return report


def molecule_overlap(train_smiles: list[str], test_smiles: list[str]) -> set[str]:
    train = {canonicalize_smiles(smi) for smi in train_smiles}
    test = {canonicalize_smiles(smi) for smi in test_smiles}
    train.discard(None)
    test.discard(None)
    return train.intersection(test)


def murcko_scaffold(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise ImportError("RDKit is required for scaffold checks.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def scaffold_overlap(train_smiles: list[str], test_smiles: list[str]) -> set[str]:
    train = {murcko_scaffold(smi) for smi in train_smiles}
    test = {murcko_scaffold(smi) for smi in test_smiles}
    train.discard(None)
    train.discard("")
    test.discard(None)
    test.discard("")
    return train.intersection(test)


def y_randomization_test(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_trials: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Train on shuffled training labels and evaluate on true test labels."""
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    real_model = clone(model)
    real_model.fit(x_train, y_train)
    real_pred = _predict_score(real_model, x_test)
    real_score = float(metric_fn(y_test, real_pred))

    random_scores = []
    for _ in range(n_trials):
        randomized_model = clone(model)
        shuffled = rng.permutation(y_train)
        randomized_model.fit(x_train, shuffled)
        random_pred = _predict_score(randomized_model, x_test)
        random_scores.append(float(metric_fn(y_test, random_pred)))

    return {
        "real_score": real_score,
        "random_mean": float(np.mean(random_scores)),
        "random_std": float(np.std(random_scores, ddof=1)) if len(random_scores) > 1 else 0.0,
        "random_scores": random_scores,
    }


def _predict_score(model, x):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    return model.predict(x)
