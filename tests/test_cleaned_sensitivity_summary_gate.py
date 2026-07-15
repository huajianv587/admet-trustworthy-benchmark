from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "experiments" / "summarize_cleaned_sensitivity.py"
    spec = importlib.util.spec_from_file_location("cleaned_summary_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _complete_runs(module) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    maximize_base = {"rf": 0.80, "xgboost": 0.75, "lightgbm": 0.70}
    minimize_base = {"rf": 0.20, "xgboost": 0.25, "lightgbm": 0.30}
    for task in module.DEFAULT_TASKS:
        metadata = module.TDC_TASK_META[task]
        metric = metadata["metric"]
        for model in module.DEFAULT_MODELS:
            for seed in module.DEFAULT_SEEDS:
                official = (
                    minimize_base[model] if metric == "mae" else maximize_base[model]
                ) + seed * 1e-5
                cleaned = official - 0.001 if metric == "mae" else official + 0.001
                raw_delta = cleaned - official
                oriented_delta = -raw_delta if metric == "mae" else raw_delta
                row: dict[str, object] = {
                    "task": task,
                    "task_type": metadata["task_type"],
                    "official_metric": metric,
                    "model": model,
                    "seed": seed,
                    "radius": 2,
                    "n_bits": 2048,
                    "threads_per_model": 1,
                    "official_train_n": 100,
                    "cleaned_train_n": 95,
                    "common_valid_n": 20,
                    "common_test_n": 30,
                    "official_fit_seconds": 0.1,
                    "cleaned_fit_seconds": 0.1,
                    "primary_official_common_test": official,
                    "primary_cleaned_common_test": cleaned,
                    "primary_raw_delta_cleaned_minus_official": raw_delta,
                    "primary_oriented_delta": oriented_delta,
                    "primary_relative_oriented_delta": oriented_delta
                    / max(abs(official), 1e-12),
                    f"official_common_valid_{metric}": official,
                    f"cleaned_common_valid_{metric}": cleaned,
                    f"official_common_test_{metric}": official,
                    f"cleaned_common_test_{metric}": cleaned,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_complete_finite_matrix_is_stable() -> None:
    module = _module()
    runs = _complete_runs(module)
    _, task_summary, gate = module.summarize_runs(
        runs,
        expected_tasks=module.DEFAULT_TASKS,
        expected_models=module.DEFAULT_MODELS,
        expected_seeds=module.DEFAULT_SEEDS,
        thresholds=module.THRESHOLDS,
    )
    assert gate["complete_expected_matrix"] is True
    assert gate["status"] == "SENSITIVITY_STABLE"
    assert gate["run_integrity_gate"]["strict_integer_fields"] is True
    assert int(task_summary["winner_changed_after_cleaning"].sum()) == 0


def test_complete_matrix_with_nan_fails_closed() -> None:
    module = _module()
    runs = _complete_runs(module)
    runs.loc[0, "primary_oriented_delta"] = np.nan
    with pytest.raises(ValueError, match="non-finite core values"):
        module.summarize_runs(
            runs,
            expected_tasks=module.DEFAULT_TASKS,
            expected_models=module.DEFAULT_MODELS,
            expected_seeds=module.DEFAULT_SEEDS,
            thresholds=module.THRESHOLDS,
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [("seed", 1.5), ("common_test_n", 30.9), ("threads_per_model", True)],
)
def test_noninteger_contract_fields_fail_closed(column: str, value: object) -> None:
    module = _module()
    runs = _complete_runs(module)
    runs[column] = runs[column].astype(object)
    runs.loc[0, column] = value
    with pytest.raises(ValueError, match="integer"):
        module.validate_run_integrity(runs)


def test_config_is_required_unique_and_finite(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="config is missing"):
        module._load_config(tmp_path / "missing.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
protocol: paired-common-test-v1
selected_tasks_for_cleaned_rerun: [AMES, AMES]
models: [rf, xgboost, lightgbm]
seeds: [1, 2, 3, 4, 5]
audit:
  scaffold_overlap: false
material_change_thresholds:
  auroc: .nan
  auprc: 0.02
  spearman: 0.03
  mae_relative: 0.05
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tasks must be nonempty and unique"):
        module._load_config(invalid)

    invalid.write_text(
        invalid.read_text(encoding="utf-8").replace("[AMES, AMES]", "[AMES]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        module._load_config(invalid)
