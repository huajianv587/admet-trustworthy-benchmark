from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .utils import as_numpy


class BinaryProbabilityCalibrator:
    """Calibrate binary probabilities on validation data.

    The historical method key ``"platt"`` is retained for artifact
    compatibility, but its implementation is L2-regularized logistic
    regression on the one-dimensional probability ``p``. It is therefore a
    logistic-on-probability mapping, not classical Platt scaling on a raw
    decision score or logit.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self.model = (
            IsotonicRegression(out_of_bounds="clip")
            if method == "isotonic"
            else LogisticRegression(max_iter=1000)
        )

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "BinaryProbabilityCalibrator":
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        labels = np.asarray(labels).astype(int)
        if self.method == "isotonic":
            self.model.fit(probs, labels)
        else:
            self.model.fit(probs.reshape(-1, 1), labels)
        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if self.method == "isotonic":
            calibrated = self.model.predict(probs)
        else:
            calibrated = self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


class TemperatureScaler:
    """Temperature scaling for logits, fitted on validation data."""

    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> "TemperatureScaler":
        import torch
        import torch.nn.functional as F

        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels)
        temperature = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            if logits_tensor.ndim == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits_tensor / temperature,
                    labels_tensor.float(),
                )
            else:
                loss = F.cross_entropy(logits_tensor / temperature, labels_tensor.long())
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.detach().cpu().item())
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler must be fitted before calibrate().")
        import torch
        import torch.nn.functional as F

        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        if logits_tensor.ndim == 1:
            probs = torch.sigmoid(logits_tensor / self.temperature)
        else:
            probs = F.softmax(logits_tensor / self.temperature, dim=-1)[:, 1]
        return as_numpy(probs).reshape(-1)


@dataclass
class ConformalRegressor:
    alpha: float = 0.1
    quantile_: float | None = None

    def fit(self, predictions: np.ndarray, labels: np.ndarray) -> "ConformalRegressor":
        preds = np.asarray(predictions, dtype=float)
        labels = np.asarray(labels, dtype=float)
        residuals = np.abs(preds - labels)
        n = len(residuals)
        if n == 0:
            raise ValueError("Cannot fit conformal regressor with no calibration samples.")
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        try:
            self.quantile_ = float(np.quantile(residuals, level, method="higher"))
        except TypeError:
            self.quantile_ = float(np.quantile(residuals, level, interpolation="higher"))
        return self

    def predict_interval(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise RuntimeError("ConformalRegressor must be fitted before predict_interval().")
        preds = np.asarray(predictions, dtype=float)
        return preds - self.quantile_, preds + self.quantile_


def interval_coverage(labels: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((labels >= lower) & (labels <= upper)))
