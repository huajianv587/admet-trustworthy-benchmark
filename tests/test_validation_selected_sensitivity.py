import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.analyze_validation_selected_sensitivity import (
    DEFAULT_CALIBRATION,
    DEFAULT_CONDITIONAL,
    DEFAULT_SELECTION,
    run_analysis,
    selected_calibration_endpoint_rows,
    selected_conditional_rows,
    summarize_selected_calibration,
    summarize_selected_conditional,
)


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "class_task",
                "task_type": "classification",
                "official_metric": "auroc",
                "validation_selected_model": "m1",
            },
            {
                "task": "reg_task",
                "task_type": "regression",
                "official_metric": "mae",
                "validation_selected_model": "m2",
            },
        ]
    )


def _conditional() -> pd.DataFrame:
    rows = []
    values = {
        ("class_task", "m1"): (0.30, 0.20, -0.10, 0.04, 0.03, np.nan),
        ("class_task", "m2"): (-0.80, -0.20, 0.40, -0.04, -0.03, np.nan),
        ("reg_task", "m1"): (-0.50, 0.30, 0.20, np.nan, np.nan, -0.02),
        ("reg_task", "m2"): (0.60, -0.10, np.nan, np.nan, np.nan, 0.05),
    }
    for (task, model), metrics in values.items():
        task_type = "classification" if task == "class_task" else "regression"
        official = "auroc" if task_type == "classification" else "mae"
        rows.append(
            {
                "task": task,
                "model": model,
                "task_type": task_type,
                "official_metric": official,
                "n_seeds": 2,
                "threshold": 0.5,
                "inside_count": 30,
                "outside_count": 40,
                "eligible": True,
                "log_error_ratio_outside_inside": metrics[0],
                "uq_correlation_advantage_inside": metrics[1],
                "selective_gain_advantage_inside": metrics[2],
                "ece_gap_outside_inside": metrics[3],
                "brier_gap_outside_inside": metrics[4],
                "coverage_error_gap_outside_inside": metrics[5],
            }
        )
    return pd.DataFrame(rows)


def _calibration() -> pd.DataFrame:
    rows = []
    for model in ("m1", "m2"):
        for seed in (1, 2):
            for method in ("isotonic", "platt"):
                selected = model == "m1"
                if method == "isotonic":
                    ece_after = 0.10 if selected else 0.30
                    brier_after = 0.20 if selected else 0.35
                else:
                    ece_after = 0.25 if selected else 0.15
                    brier_after = 0.30 if selected else 0.20
                rows.append(
                    {
                        "task": "class_task",
                        "task_type": "classification",
                        "official_metric": "auroc",
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "test_ece_before": 0.20,
                        "test_ece_after": ece_after,
                        "test_brier_before": 0.25,
                        "test_brier_after": brier_after,
                        "coverage": np.nan,
                        "mean_interval_width": np.nan,
                    }
                )
    for model in ("m1", "m2"):
        for seed, coverage in ((1, 0.80), (2, 0.90)):
            rows.append(
                {
                    "task": "reg_task",
                    "task_type": "regression",
                    "official_metric": "mae",
                    "model": model,
                    "seed": seed,
                    "method": "conformal_90",
                    "test_ece_before": np.nan,
                    "test_ece_after": np.nan,
                    "test_brier_before": np.nan,
                    "test_brier_after": np.nan,
                    "coverage": coverage if model == "m2" else 0.50,
                    "mean_interval_width": 2.0 if model == "m2" else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selected_conditional_filter_uses_frozen_validation_identity_only() -> None:
    selection = _selection()
    source = _conditional()
    selection_before = selection.copy(deep=True)
    source_before = source.copy(deep=True)
    rows = selected_conditional_rows(selection, source)
    summary = summarize_selected_conditional(
        rows, bootstrap_reps=101, permutation_reps=257, random_seed=42
    )

    pd.testing.assert_frame_equal(selection, selection_before, check_exact=True)
    pd.testing.assert_frame_equal(source, source_before, check_exact=True)
    assert dict(zip(rows["task"], rows["selected_model"])) == {
        "class_task": "m1",
        "reg_task": "m2",
    }
    primary = summary.set_index("metric")
    assert primary.loc["log_error_ratio_outside_inside", "positive_count"] == 2
    assert primary.loc["brier_gap_outside_inside", "positive_count"] == 1
    assert primary.loc["coverage_error_gap_outside_inside", "positive_count"] == 1


def test_selected_calibration_aggregates_seeds_but_never_unselected_models() -> None:
    rows = selected_calibration_endpoint_rows(_selection(), _calibration())
    assert len(rows) == 3
    assert set(rows.loc[rows["task"] == "class_task", "selected_model"]) == {"m1"}
    assert set(rows.loc[rows["task"] == "reg_task", "selected_model"]) == {"m2"}
    summary = summarize_selected_calibration(
        rows, bootstrap_reps=101, permutation_reps=257, random_seed=42
    ).set_index(["method", "metric"])
    assert summary.loc[("isotonic", "ece_gain"), "improved_count"] == 1
    assert summary.loc[("isotonic", "brier_gain"), "improved_count"] == 1
    assert summary.loc[("platt", "ece_gain"), "improved_count"] == 0
    assert summary.loc[("platt", "brier_gain"), "improved_count"] == 0
    conformal = rows.loc[rows["method"] == "conformal_90"].iloc[0]
    assert conformal["coverage"] == pytest.approx(0.85)
    assert conformal["absolute_coverage_error"] == pytest.approx(0.05)


def test_selected_only_inputs_fail_closed_on_missing_identity_or_inconsistent_before() -> None:
    missing = _conditional()
    missing = missing[~((missing["task"] == "reg_task") & (missing["model"] == "m2"))]
    with pytest.raises(ValueError, match="exactly one selected conditional row"):
        selected_conditional_rows(_selection(), missing)

    inconsistent = _calibration()
    mask = (
        (inconsistent["task"] == "class_task")
        & (inconsistent["model"] == "m1")
        & (inconsistent["seed"] == 1)
        & (inconsistent["method"] == "platt")
    )
    inconsistent.loc[mask, "test_ece_before"] = 0.99
    with pytest.raises(ValueError, match="Uncalibrated metrics disagree"):
        selected_calibration_endpoint_rows(_selection(), inconsistent)


def test_nonfinal_file_bound_run_is_deterministic_and_input_immutable(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.csv"
    conditional_path = tmp_path / "conditional.csv"
    calibration_path = tmp_path / "calibration.csv"
    _selection().to_csv(selection_path, index=False)
    _conditional().to_csv(conditional_path, index=False)
    _calibration().to_csv(calibration_path, index=False)
    inputs = [selection_path, conditional_path, calibration_path]
    before = {path: _sha256(path) for path in inputs}

    kwargs = {
        "selection_path": selection_path,
        "conditional_path": conditional_path,
        "calibration_path": calibration_path,
        "primary_threshold": 0.5,
        "bootstrap_reps": 101,
        "permutation_reps": 257,
        "random_seed": 123,
        "strict_final_contract": False,
    }
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_analysis(out_dir=first_dir, **kwargs)
    second = run_analysis(out_dir=second_dir, **kwargs)

    assert before == {path: _sha256(path) for path in inputs}
    assert first["headline_counts"] == second["headline_counts"]
    first_files = sorted(path.name for path in first_dir.iterdir())
    second_files = sorted(path.name for path in second_dir.iterdir())
    assert first_files == second_files
    for filename in first_files:
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    latex = (first_dir / "table_supp_validation_selected_sensitivity.tex").read_text(
        encoding="utf-8"
    )
    interpretation = (first_dir / "VALIDATION_SELECTED_SENSITIVITY.md").read_text(
        encoding="utf-8"
    )
    assert "inside-support utility" in latex
    assert "inside-domain utility" not in latex
    assert "validation-selected models evaluated here" in interpretation
    assert "deployed validation-selected models" not in interpretation


requires_locked_tables = pytest.mark.skipif(
    not all(path.is_file() for path in (DEFAULT_SELECTION, DEFAULT_CONDITIONAL, DEFAULT_CALIBRATION)),
    reason="requires complete locked result tables, absent from a source-only public clone",
)


@requires_locked_tables
def test_locked_selected_only_headlines_match_endpoint_level_recomputation(
    tmp_path: Path,
) -> None:
    before = {
        path: _sha256(path)
        for path in (DEFAULT_SELECTION, DEFAULT_CONDITIONAL, DEFAULT_CALIBRATION)
    }
    manifest = run_analysis(
        out_dir=tmp_path,
        bootstrap_reps=211,
        permutation_reps=503,
        random_seed=20260715,
    )
    assert before == {
        path: _sha256(path)
        for path in (DEFAULT_SELECTION, DEFAULT_CONDITIONAL, DEFAULT_CALIBRATION)
    }
    counts = manifest["headline_counts"]
    assert counts["outside_support_error_positive_endpoints"] == {
        "positive": 20,
        "eligible": 21,
    }
    assert counts["outside_support_brier_positive_endpoints"] == {
        "positive": 11,
        "eligible": 12,
    }
    assert counts["isotonic_ece_improved_endpoints"] == {
        "improved": 11,
        "eligible": 13,
    }
    assert counts["isotonic_brier_improved_endpoints"] == {
        "improved": 8,
        "eligible": 13,
    }
    assert counts["logistic_on_probability_ece_improved_endpoints"] == {
        "improved": 11,
        "eligible": 13,
    }
    assert counts["logistic_on_probability_brier_improved_endpoints"] == {
        "improved": 11,
        "eligible": 13,
    }
