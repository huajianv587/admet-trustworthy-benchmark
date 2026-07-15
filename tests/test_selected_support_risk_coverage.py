import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.analyze_reliability_regret import (
    EXPECTED_TASKS,
    FINAL_EXPECTED_MODELS,
    ModelEnsemble,
    aurc_from_curve,
)
from experiments.analyze_selected_support_risk_coverage import (
    evaluate_selected_support,
    latex_summary_table,
    ordered_curve,
    summarize_endpoints,
    validate_frozen_selection,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "tables" / "selected_support_risk_coverage_final"
COMMITTED_SELECTED_SUPPORT_ARTIFACTS = (
    RESULT_DIR / "selected_support_risk_coverage_by_endpoint.csv",
    RESULT_DIR / "selected_support_risk_coverage_summary.csv",
    RESULT_DIR / "selected_support_risk_coverage_gate.json",
)
requires_committed_selected_support_artifacts = pytest.mark.skipif(
    not all(path.is_file() for path in COMMITTED_SELECTED_SUPPORT_ARTIFACTS),
    reason=(
        "requires committed selected-support result artifacts, absent from a "
        "source-only public clone"
    ),
)


def _complete_synthetic_matrix():
    ensembles = []
    selection_rows = []
    pool = ";".join(FINAL_EXPECTED_MODELS)
    for task in EXPECTED_TASKS:
        selection_rows.append(
            {
                "task": task,
                "n_models": 5,
                "model_pool": pool,
                "validation_selected_model": "rf",
                "validation_selected_score_oriented": 0.5,
                "validation_score_margin": 0.01,
                "validation_tie_models": "rf",
            }
        )
        for model in FINAL_EXPECTED_MODELS:
            ensembles.append(
                ModelEnsemble(
                    task=task,
                    model=model,
                    task_type="classification",
                    official_metric="auroc",
                    seeds=(1, 2, 3, 4, 5),
                    validation_scores={seed: 0.5 for seed in range(1, 6)},
                    test=pd.DataFrame(
                        {
                            "sample_index": [0, 1, 2],
                            "smiles": ["C", "CC", "CCC"],
                            "y_true": [0.0, 1.0, 0.0],
                            "prediction": [0.1, 0.8, 0.3],
                            "uncertainty": [0.1, 0.3, 0.2],
                            "max_tanimoto": [0.9, 0.2, 0.5],
                            "n_seeds": [5, 5, 5],
                        }
                    ),
                    conformal_coverage=float("nan"),
                )
            )
    return pd.DataFrame(selection_rows), ensembles


def test_support_curve_uses_high_support_first_and_exact_random_baseline():
    losses = np.array([0.0, 2.0, 4.0])
    support = np.array([0.9, 0.5, 0.1])
    curve = ordered_curve(
        losses,
        1.0 - support,
        np.arange(3),
        ranking="morgan_support",
    )
    assert np.allclose(curve["cumulative_risk"], [0.0, 1.0, 2.0])
    aurc, oracle, excess = aurc_from_curve(curve)
    assert np.isclose(aurc, 1.0)
    assert np.isclose(np.mean(losses), 2.0)
    assert np.isclose((np.mean(losses) - aurc) / np.mean(losses), 0.5)
    assert oracle <= aurc
    assert excess >= 0.0


def test_support_ties_use_expected_risk_not_loss_or_row_order():
    first = ordered_curve(
        np.array([0.0, 2.0]),
        np.array([0.5, 0.5]),
        np.array([0, 1]),
        ranking="morgan_support",
    )
    reversed_rows = ordered_curve(
        np.array([2.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([1, 0]),
        ranking="morgan_support",
    )
    assert np.allclose(first["cumulative_risk"], [1.0, 1.0])
    assert np.allclose(first["cumulative_risk"], reversed_rows["cumulative_risk"])


def test_frozen_validation_choice_is_independent_of_test_predictions():
    selection, ensembles = _complete_synthetic_matrix()
    resolved = validate_frozen_selection(selection, ensembles)
    assert set(resolved) == set(EXPECTED_TASKS)
    assert {item.model for item in resolved.values()} == {"rf"}

    for item in ensembles:
        item.test["prediction"] = 1.0 - item.test["prediction"]
    reversed_resolved = validate_frozen_selection(selection, ensembles)
    assert {task: item.model for task, item in resolved.items()} == {
        task: item.model for task, item in reversed_resolved.items()
    }


def test_frozen_selection_fails_closed_on_incomplete_model_matrix():
    selection, ensembles = _complete_synthetic_matrix()
    with pytest.raises(ValueError, match="exact 22x5 matrix"):
        validate_frozen_selection(selection, ensembles[:-1])


def test_endpoint_arithmetic_uses_selected_model_only():
    selection, ensembles = _complete_synthetic_matrix()
    selected = validate_frozen_selection(selection, ensembles)
    endpoints, curves = evaluate_selected_support(selection, selected)
    assert len(endpoints) == 22
    assert set(endpoints["validation_selected_model"]) == {"rf"}
    assert set(curves["model"]) == {"rf"}
    assert set(curves["ranking"]) == {"morgan_support", "ensemble_uncertainty"}
    expected = (
        endpoints["mean_loss_random_aurc"] - endpoints["support_aurc"]
    ) / endpoints["mean_loss_random_aurc"]
    assert np.allclose(endpoints["support_gain_vs_random"], expected)


def test_latex_table_counts_negative_as_favorable_for_paired_difference():
    endpoints = pd.DataFrame(
        {
            "task_type": ["classification", "regression"],
            "support_gain_vs_random": [0.2, 0.1],
            "uncertainty_gain_vs_random": [0.3, 0.2],
            "support_minus_uncertainty_relative_aurc": [-0.1, 0.1],
        }
    )
    summary = summarize_endpoints(
        endpoints,
        bootstrap_reps=100,
        sign_flip_reps=100,
        random_seed=7,
    )
    latex = latex_summary_table(summary)
    assert "Morgan support vs random ordering" in latex
    assert "Morgan support minus ensemble uncertainty" in latex
    assert "descriptive 95\\% endpoint-bootstrap interval" in latex
    assert "endpoint bootstrap CI" not in latex
    assert "1/2" in latex


@requires_committed_selected_support_artifacts
def test_committed_selected_support_artifacts_reconcile_and_pass():
    endpoints = pd.read_csv(
        RESULT_DIR / "selected_support_risk_coverage_by_endpoint.csv"
    )
    summary = pd.read_csv(
        RESULT_DIR / "selected_support_risk_coverage_summary.csv"
    )
    gate = json.loads(
        (RESULT_DIR / "selected_support_risk_coverage_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "PASS"
    assert gate["zero_refit"] is True
    assert gate["test_outcomes_used_for_model_selection"] is False
    assert all(gate["technical_checks"].values())
    assert len(endpoints) == 22
    assert set(endpoints["task"]) == set(EXPECTED_TASKS)
    assert endpoints["n_seeds"].eq(5).all()

    expected_gain = (
        endpoints["mean_loss_random_aurc"] - endpoints["support_aurc"]
    ) / endpoints["mean_loss_random_aurc"]
    assert np.allclose(endpoints["support_gain_vs_random"], expected_gain)
    primary = summary.query(
        "scope == 'all' and metric == 'support_gain_vs_random'"
    ).iloc[0]
    assert np.isclose(primary["mean"], endpoints["support_gain_vs_random"].mean())
    assert int(primary["positive_endpoints"]) == int(
        endpoints["support_gain_vs_random"].gt(0).sum()
    )
    assert int(primary["positive_endpoints"]) == 20
    assert float(primary["bootstrap_ci_low"]) > 0.0
    assert np.isclose(
        gate["primary_descriptive_result"]["mean"], float(primary["mean"])
    )
