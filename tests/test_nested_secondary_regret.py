from __future__ import annotations

import csv

from experiments.analyze_nested_secondary_regret import build_records, summarize


def _row(task: str, fold: int, model: str, selected: bool, brier: float) -> dict[str, str]:
    return {
        "task": task,
        "role_seed": "1",
        "outer_fold": str(fold),
        "task_type": "classification",
        "model": model,
        "selected_by_inner": str(selected),
        "outer_brier": str(brier),
        "outer_log_loss": str(brier + 0.1),
        "outer_mae": "",
        "outer_rmse": "",
    }


def test_secondary_regret_uses_untouched_outer_candidate_minimum() -> None:
    rows = [
        _row("task_a", 0, "rf", True, 0.30),
        _row("task_a", 0, "extra_trees", False, 0.20),
        _row("task_a", 0, "hist_gb", False, 0.25),
        _row("task_a", 1, "rf", True, 0.10),
        _row("task_a", 1, "extra_trees", False, 0.20),
        _row("task_a", 1, "hist_gb", False, 0.15),
    ]
    records = build_records(rows)
    brier = [row for row in records if row["metric"] == "brier"]
    assert len(brier) == 2
    assert abs(brier[0]["raw_selection_regret"] - 0.1) < 1e-12
    assert brier[0]["selected_matches_candidate_oracle"] is False
    assert abs(brier[1]["raw_selection_regret"]) < 1e-12
    assert brier[1]["selected_matches_candidate_oracle"] is True


def test_summary_reports_endpoint_block_statistics() -> None:
    rows = [
        _row("task_a", 0, "rf", True, 0.30),
        _row("task_a", 0, "extra_trees", False, 0.20),
        _row("task_a", 0, "hist_gb", False, 0.25),
        _row("task_b", 0, "rf", True, 0.20),
        _row("task_b", 0, "extra_trees", False, 0.20),
        _row("task_b", 0, "hist_gb", False, 0.25),
    ]
    summaries = summarize(build_records(rows))
    brier = next(row for row in summaries if row["metric"] == "brier")
    assert brier["outer_fold_records"] == 2
    assert brier["endpoint_count"] == 2
    assert brier["positive_regret_folds"] == 1
    assert brier["oracle_match_folds"] == 1
    assert brier["positive_regret_endpoints"] == 1
    assert 0.0 <= brier["endpoint_block_bootstrap_95_low"] <= brier["endpoint_block_bootstrap_95_high"] <= 1.0
