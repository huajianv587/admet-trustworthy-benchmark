from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ensemble_mean_and_uncertainty(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and sample standard deviation.

    Expected shape: (n_models, n_samples).
    """
    preds = np.asarray(predictions, dtype=float)
    if preds.ndim != 2:
        raise ValueError("predictions must have shape (n_models, n_samples)")
    return np.mean(preds, axis=0), np.std(preds, axis=0, ddof=1)


def uncertainty_rejection_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    rejection_fracs: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if not (len(y_true) == len(y_pred) == len(uncertainty)):
        raise ValueError("y_true, y_pred, and uncertainty must have the same length")

    order = np.argsort(uncertainty)
    rows: list[dict[str, float]] = []
    n = len(y_true)
    for frac in rejection_fracs:
        if not 0 <= frac < 1:
            raise ValueError("rejection fractions must be in [0, 1)")
        keep = int(round(n * (1 - frac)))
        kept_idx = order[:keep]
        rows.append(
            {
                "rejection_frac": float(frac),
                "kept_count": int(len(kept_idx)),
                "metric": float(metric_fn(y_true[kept_idx], y_pred[kept_idx])),
                "mean_uncertainty_kept": float(np.mean(uncertainty[kept_idx])),
            }
        )
    return rows
