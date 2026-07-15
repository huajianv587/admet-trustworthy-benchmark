import numpy as np
import pandas as pd

from experiments.analyze_conditional_reliability import (
    auprc,
    auroc,
    conformal_quantile,
    holm_adjust as conditional_holm_adjust,
    spearman,
)
from experiments.analyze_trustworthiness_alignment import rank_correlation


def test_binary_metrics_handle_perfect_reverse_and_tied_predictions():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auprc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auprc(labels, np.full(4, 0.5)) == 0.5


def test_rank_helpers_are_directionally_correct():
    ascending = np.array([1.0, 2.0, 3.0, 4.0])
    descending = ascending[::-1]
    assert np.isclose(spearman(ascending, ascending), 1.0)
    assert np.isclose(spearman(ascending, descending), -1.0)
    assert np.isclose(
        rank_correlation(pd.Series(ascending), pd.Series(descending)), -1.0
    )


def test_conformal_quantile_uses_finite_sample_correction():
    residuals = np.arange(1.0, 11.0)
    assert conformal_quantile(residuals, alpha=0.1) == 10.0


def test_holm_adjustment_is_monotone_in_sorted_pvalues():
    raw = pd.Series([0.01, 0.04, 0.03])
    adjusted = conditional_holm_adjust(raw)
    ordered = adjusted.loc[raw.sort_values().index].to_numpy()
    assert np.all(np.diff(ordered) >= 0)
    assert np.all(adjusted.to_numpy() >= raw.to_numpy())
