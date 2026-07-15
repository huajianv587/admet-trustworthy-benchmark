from pathlib import Path

import pytest

from experiments.make_reliability_regret_figure import (
    DEFAULT_RESULTS,
    DIMENSIONS,
    build_svg,
    contrast_text,
    file_sha256,
    plotted_practical_signals,
    scope_banner,
    prepare_figure_data,
    regret_color,
)


LOCKED_RESULT_DIR = Path(DEFAULT_RESULTS)
requires_locked_figure_tables = pytest.mark.skipif(
    not all(
        (LOCKED_RESULT_DIR / name).is_file()
        for name in (
            "reliability_regret_by_endpoint.csv",
            "reliability_regret_summary.csv",
            "reliability_regret_gate.json",
            "validation_selection_by_task.csv",
        )
    ),
    reason=(
        "requires non-source result tables; source-only clones still run the "
        "self-contained figure-contract tests"
    ),
)


def test_secondary_log_loss_signal_is_disclosed_but_not_required_in_core_heatmap():
    gate = {
        "effect_gate": {
            "dimensions_with_practical_effect_signal": [
                "ece",
                "selective_relative_excess_aurc_log_loss",
            ]
        }
    }
    assert plotted_practical_signals(gate, ["ece", "brier"]) == ["ece"]


def test_regret_color_scale_is_bounded_and_has_readable_endpoints():
    assert regret_color(-1.0) == regret_color(0.0)
    assert regret_color(2.0) == regret_color(1.0)
    assert regret_color(0.0) != regret_color(1.0)
    assert contrast_text(regret_color(0.0)) == "#FFFFFF"
    assert contrast_text(regret_color(1.0)) != "#FFFFFF"


@requires_locked_figure_tables
def test_locked_traditional_source_builds_complete_explicit_grid():
    grid, summary, gate, selections = prepare_figure_data(Path(DEFAULT_RESULTS))
    assert grid["task"].nunique() == 22
    assert len(grid) == 22 * len(DIMENSIONS)
    assert int(grid["applicable"].sum()) == 79
    assert gate["scope"] == "traditional_only"
    assert gate["status"] == "GO"
    assert not gate["final_submission_gate"]
    assert set(selections["model_pool"]) == {"lightgbm;rf;xgboost"}
    signals = set(
        summary.loc[summary["practical_effect_signal"], "dimension"].astype(str)
    )
    assert signals == {"ece", "selective_relative_excess_aurc"}


@requires_locked_figure_tables
def test_svg_exposes_scope_model_pool_and_marker_semantics():
    result_dir = Path(DEFAULT_RESULTS)
    grid, summary, gate, selections = prepare_figure_data(result_dir)
    source = result_dir / "reliability_regret_by_endpoint.csv"
    svg = build_svg(grid, summary, gate, selections, file_sha256(source))
    assert "TRADITIONAL-MODEL EXPLORATORY MATRIX" in svg
    assert "NOT THE FINAL FIVE-MODEL ANALYSIS" in svg
    assert "Model pool on every endpoint: LGBM + RF + XGB" in svg
    assert "Practical: raw regret exceeds" in svg
    assert "Tie-robust: remains practical" in svg
    assert "Normalized reliability regret" in svg


def test_final_completed_scope_has_reader_facing_descriptive_label():
    text, _ = scope_banner(
        {
            "scope": "final",
            "status": "GO",
            "coverage": {"complete_expected_model_endpoint_matrix": True},
        }
    )
    assert text == "COMPLETE FIVE-MODEL MATRIX · DESCRIPTIVE BENCHMARK AUDIT"
    assert "GO" not in text
