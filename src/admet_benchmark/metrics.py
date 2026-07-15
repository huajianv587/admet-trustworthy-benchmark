from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _safe_metric(fn, *args: Any, default: float = float("nan"), **kwargs: Any) -> float:
    try:
        value = fn(*args, **kwargs)
        return float(value)
    except Exception:
        return default


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    positive_rate = float(np.mean(y_true)) if len(y_true) else float("nan")
    return {
        "auroc": _safe_metric(roc_auc_score, y_true, y_score),
        "auprc": _safe_metric(average_precision_score, y_true, y_score),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "brier": _safe_metric(brier_score_loss, y_true, y_score),
        "positive_rate": positive_rate,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = _safe_metric(mean_squared_error, y_true, y_pred)
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(y_true, y_pred, nan_policy="omit").statistic)
    except Exception:
        spearman = float("nan")
    return {
        "mae": _safe_metric(mean_absolute_error, y_true, y_pred),
        "rmse": float(np.sqrt(mse)) if np.isfinite(mse) else float("nan"),
        "r2": _safe_metric(r2_score, y_true, y_pred),
        "spearman": spearman,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(y_prob[mask])) if np.any(mask) else float("nan"),
                "empirical_positive_rate": float(np.mean(y_true[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows
