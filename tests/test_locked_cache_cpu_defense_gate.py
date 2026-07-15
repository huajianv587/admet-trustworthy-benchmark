from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    TDC_TASK_META,
    TDC_TASKS,
)
from experiments.run_shap_selected import SELECTED


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "experiments" / "summarize_locked_cache_cpu_defenses.py"
    spec = importlib.util.spec_from_file_location("cpu_defense_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    models = {task: "rf" for task in TDC_TASKS}
    selection = pd.DataFrame(
        [{"task": task, "model": model} for task, model in models.items()]
    )
    selection_path = tmp_path / "selection.csv"
    selection.to_csv(selection_path, index=False)

    summaries = []
    scores = []
    for task in TDC_TASKS:
        metric = TDC_TASK_META[task]["metric"]
        task_scores = [0.1 + trial / 1000 for trial in range(1, 11)]
        random_mean = sum(task_scores) / len(task_scores)
        real_score = 0.05 if metric == "mae" else 0.8
        oriented_margin = (
            random_mean - real_score if metric == "mae" else real_score - random_mean
        )
        summaries.append(
            {
                "task": task,
                "model": models[task],
                "seed": 1,
                "split_source": LOCKED_CACHE_SPLIT_SOURCE,
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "metric": metric,
                "real_score": real_score,
                "random_mean": random_mean,
                "random_std": pd.Series(task_scores).std(ddof=1),
                "oriented_margin": oriented_margin,
                "n_trials": 10,
            }
        )
        scores.extend(
            {
                "task": task,
                "model": models[task],
                "seed": 1,
                "split_source": LOCKED_CACHE_SPLIT_SOURCE,
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "metric": metric,
                "trial": trial,
                "random_score": score,
            }
            for trial, score in enumerate(task_scores, start=1)
        )
    y_summary_path = tmp_path / "y_summary.csv"
    y_scores_path = tmp_path / "y_scores.csv"
    pd.DataFrame(summaries).to_csv(y_summary_path, index=False)
    pd.DataFrame(scores).to_csv(y_scores_path, index=False)

    shap_rows = []
    mapping_rows = []
    for task, _ in SELECTED:
        for rank in range(1, 21):
            feature = f"morgan_bit_{rank}"
            shap_rows.append(
                {
                    "task": task,
                    "model": models[task],
                    "seed": 1,
                    "split_source": LOCKED_CACHE_SPLIT_SOURCE,
                    "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": 1 / rank,
                    "train_n": 100,
                    "test_n": 40,
                    "sample_size_actual": 40,
                }
            )
            if rank <= 10:
                mapping_rows.append(
                    {
                        "task": task,
                        "model": models[task],
                        "seed": 1,
                        "split_source": LOCKED_CACHE_SPLIT_SOURCE,
                        "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                        "rank": rank,
                        "feature": feature,
                        "bit_id": rank,
                        "mean_abs_shap": 1 / rank,
                        "example_smiles": "CCO",
                        "bit_atom_index": 0,
                        "bit_radius": 1,
                        "substructure_smiles": "CC",
                    }
                )
    shap_path = tmp_path / "shap.csv"
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(shap_rows).to_csv(shap_path, index=False)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)
    return y_summary_path, y_scores_path, shap_path, mapping_path, selection_path


def test_cpu_defense_gate_passes_exact_locked_matrix(tmp_path: Path) -> None:
    module = _load_module()
    result = module.summarize(*_write_valid_inputs(tmp_path))
    assert result["status"] == "GO"
    assert result["artifact_gate"] == "PASS"
    assert result["negative_control_signal"] == "PASS"
    assert result["coverage"]["mapped_substructures"] == 80
    assert result["y_randomization"]["positive_margin_count"] == 22


def test_cpu_defense_gate_holds_on_incomplete_mapping(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_valid_inputs(tmp_path)
    mapping = pd.read_csv(paths[3])
    mapping.loc[0, "substructure_smiles"] = ""
    mapping.to_csv(paths[3], index=False)
    result = module.summarize(*paths)
    assert result["status"] == "HOLD"
    assert result["artifact_gate"] == "HOLD"
    assert any("mapping is incomplete" in failure for failure in result["failures"])


def test_cpu_defense_gate_reviews_nonpositive_randomization(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_valid_inputs(tmp_path)
    summary = pd.read_csv(paths[0])
    summary.loc[0, "real_score"] = summary.loc[0, "random_mean"]
    summary.loc[0, "oriented_margin"] = 0.0
    summary.to_csv(paths[0], index=False)
    result = module.summarize(*paths)
    assert result["status"] == "REVIEW"
    assert result["artifact_gate"] == "PASS"
    assert result["negative_control_signal"] == "MIXED"
    assert result["y_randomization"]["positive_margin_count"] == 21


def test_cpu_defense_gate_rejects_metric_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_valid_inputs(tmp_path)
    summary = pd.read_csv(paths[0])
    summary.loc[0, "metric"] = "auroc" if summary.loc[0, "metric"] == "mae" else "mae"
    summary.to_csv(paths[0], index=False)
    result = module.summarize(*paths)
    assert result["status"] == "HOLD"
    assert any("metric mismatch" in failure for failure in result["failures"])


def test_cpu_defense_gate_rejects_fractional_trial(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_valid_inputs(tmp_path)
    scores = pd.read_csv(paths[1])
    scores["trial"] = scores["trial"].astype(object)
    scores.loc[0, "trial"] = 1.5
    scores.to_csv(paths[1], index=False)
    result = module.summarize(*paths)
    assert result["status"] == "HOLD"
    assert any("trial contract" in failure for failure in result["failures"])
