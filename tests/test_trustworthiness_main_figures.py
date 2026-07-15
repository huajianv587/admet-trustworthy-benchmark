import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.make_trustworthiness_main_figures import (
    FINAL_BANNER,
    FINAL_MODELS,
    TASK_ORDER,
    validate_final_five_model_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL_DIR = ROOT / "results" / "tables" / "conditional_reliability_local_prelim"
CALIBRATION = ROOT / "results" / "tables" / "calibration_summary_local_prelim.csv"
FIGURE_DIR = ROOT / "results" / "figures" / "trustworthiness_submission_partial"
TRADITIONAL_MODELS = {"rf", "xgboost", "lightgbm"}
requires_provisional_result_artifacts = pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            CONDITIONAL_DIR / "model_reliability_profiles.csv",
            CONDITIONAL_DIR / "conditional_reliability_by_task_model.csv",
            CALIBRATION,
            FIGURE_DIR / "fig02_source_data.csv",
            FIGURE_DIR / "fig03_source_data.csv",
            FIGURE_DIR / "figure_manifest.json",
        )
    ),
    reason=(
        "requires non-source diagnostic result artifacts; the source-only clone "
        "still runs the final figure-source contract tests"
    ),
)


def _synthetic_final_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile = pd.DataFrame(
        [
            {
                "task": task,
                "task_type": "classification" if task == TASK_ORDER[0] else "regression",
                "model": model,
                "n_seeds": 5,
            }
            for task in TASK_ORDER
            for model in FINAL_MODELS
        ]
    )
    conditional = pd.DataFrame(
        [
            {"task": task, "model": model, "threshold": 0.5, "eligible": True}
            for task in TASK_ORDER
            for model in FINAL_MODELS
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "task": TASK_ORDER[0],
                "task_type": "classification",
                "model": model,
                "seed": seed,
                "method": method,
                "test_ece_before": 0.2,
                "test_ece_after": 0.1,
                "test_brier_before": 0.2,
                "test_brier_after": 0.1,
            }
            for model in FINAL_MODELS
            for seed in range(1, 6)
            for method in ("isotonic", "platt")
        ]
    )
    return profile, conditional, calibration


def test_final_figure_source_gate_requires_exact_calibration_matrix() -> None:
    profile, conditional, calibration = _synthetic_final_sources()
    validate_final_five_model_sources(profile, conditional, calibration)
    with pytest.raises(ValueError, match="exactly one row"):
        validate_final_five_model_sources(profile, conditional.iloc[:-1], calibration)
    with pytest.raises(ValueError, match="calibration matrix mismatch"):
        validate_final_five_model_sources(profile, conditional, calibration.iloc[:-1])


@requires_provisional_result_artifacts
def test_figure_sources_are_complete_traditional_only() -> None:
    profile = pd.read_csv(CONDITIONAL_DIR / "model_reliability_profiles.csv")
    conditional = pd.read_csv(
        CONDITIONAL_DIR / "conditional_reliability_by_task_model.csv"
    )
    calibration = pd.read_csv(CALIBRATION)

    assert set(profile["model"]) == TRADITIONAL_MODELS
    assert set(conditional["model"]) == TRADITIONAL_MODELS
    assert set(calibration["model"]) == TRADITIONAL_MODELS
    coverage = profile.groupby("model")["task"].nunique().to_dict()
    assert coverage == {model: 22 for model in TRADITIONAL_MODELS}
    assert set(profile["n_seeds"]) == {5}
    assert len(profile[["task", "model"]].drop_duplicates()) == 66
    calibration = calibration[
        (calibration["task_type"] == "classification")
        & calibration["method"].isin(["isotonic", "platt"])
    ]
    seed_counts = calibration.groupby(["task", "model", "method"])["seed"].nunique()
    assert set(seed_counts) == {5}
    assert not calibration.duplicated(["task", "model", "method", "seed"]).any()


@requires_provisional_result_artifacts
def test_fig02_export_equals_independent_endpoint_aggregation() -> None:
    raw = pd.read_csv(CONDITIONAL_DIR / "conditional_reliability_by_task_model.csv")
    eligible = raw["eligible"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().isin({"true", "1", "yes"})
    raw = raw[np.isclose(raw["threshold"].astype(float), 0.5) & eligible].copy()
    metrics = [
        "log_error_ratio_outside_inside",
        "uq_correlation_advantage_inside",
        "selective_gain_advantage_inside",
        "ece_gap_outside_inside",
        "brier_gap_outside_inside",
        "coverage_error_gap_outside_inside",
    ]
    expected = (
        raw.groupby(["task", "task_type"], as_index=False)[metrics]
        .mean()
        .sort_values(["task_type", "task"])
        .reset_index(drop=True)
    )
    expected.insert(0, "threshold", 0.5)
    expected["error_change_percent"] = 100.0 * (
        np.exp(expected["log_error_ratio_outside_inside"]) - 1.0
    )
    observed = pd.read_csv(FIGURE_DIR / "fig02_source_data.csv")
    pd.testing.assert_frame_equal(observed, expected, check_exact=False, rtol=1e-12)


@requires_provisional_result_artifacts
def test_fig03_export_equals_independent_calibration_gains() -> None:
    raw = pd.read_csv(CALIBRATION)
    raw = raw[
        (raw["task_type"] == "classification")
        & raw["method"].isin(["isotonic", "platt"])
    ].copy()
    raw["ece_gain"] = raw["test_ece_before"] - raw["test_ece_after"]
    raw["brier_gain"] = raw["test_brier_before"] - raw["test_brier_after"]
    expected = (
        raw.groupby(["task", "method"], as_index=False)[["ece_gain", "brier_gain"]]
        .mean()
        .sort_values(["task", "method"])
        .reset_index(drop=True)
    )
    observed = pd.read_csv(FIGURE_DIR / "fig03_source_data.csv")
    pd.testing.assert_frame_equal(observed, expected, check_exact=False, rtol=1e-12)


@requires_provisional_result_artifacts
def test_manifest_and_vector_outputs_make_provisional_scope_unambiguous() -> None:
    manifest = json.loads((FIGURE_DIR / "figure_manifest.json").read_text())
    assert manifest["status"] == "provisional_traditional_only"
    assert manifest["neural_results_included"] is False
    assert set(manifest["models_included"]) == TRADITIONAL_MODELS
    assert manifest["gate"]["traditional_only_provisional_figures"] == "PASS"
    assert manifest["gate"]["final_five_model_submission_figures"] == "HOLD"
    assert "TRADITIONAL-ONLY" in manifest["visible_label"]
    assert "FINAL FIVE-MODEL ANALYSIS PENDING" in manifest["visible_label"]
    assert all("combined_partial" not in source for source in manifest["sources"].values())
    assert manifest["superseded_artifacts"]["submission_eligible"] is False
    assert not list(FIGURE_DIR.glob("fig01_winner_misalignment.*"))
    for output, expected_digest in manifest["output_sha256"].items():
        observed_digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
        assert observed_digest == expected_digest
    for stem in ["fig02_conditional_reliability", "fig03_calibration_gains_failures"]:
        for suffix in [".png", ".pdf", ".svg"]:
            path = FIGURE_DIR / f"{stem}{suffix}"
            assert path.exists() and path.stat().st_size > 10_000
        svg = (FIGURE_DIR / f"{stem}.svg").read_text()
        assert "PROVISIONAL" in svg
        assert "TRADITIONAL-ONLY" in svg
    fig02_svg = (FIGURE_DIR / "fig02_conditional_reliability.svg").read_text()
    # Matplotlib may split the footnote across SVG text groups when the panel
    # layout changes; assert both semantic fragments instead of byte adjacency.
    assert "predefined minimum" in fig02_svg
    assert "sample count." in fig02_svg
    assert "analysis-locked minimum sample count" not in fig02_svg


def test_final_banner_is_reader_facing_and_not_an_internal_status_label() -> None:
    assert FINAL_BANNER == "COMPLETE FIVE-MODEL MATRIX (5 SEEDS × 22 ENDPOINTS)"
    assert "LOCKED" not in FINAL_BANNER
    assert "GO" not in FINAL_BANNER
