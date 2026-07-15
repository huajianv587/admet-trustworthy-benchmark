from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from admet_benchmark.smiles_compat import SMILES_COMPATIBILITY_POLICY
from experiments import build_reproducibility_release as release


TASK = "Caco2_Wang"
SEED = 1


def test_public_environment_contract_covers_gin_and_documents_chemprop() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    environment = (root / "environment.yml").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    for dependency_text in (requirements, environment, pyproject):
        assert "torch-geometric>=2.4" in dependency_text
    assert "pip install -e '.[chem,models,dev]'" in readme
    assert "Chemprop 2.2.1" in readme
    assert "Python 3.12.10" in readme
    assert "PyTorch 2.7.1+cu128" in readme
    assert "CUDA runtime 12.8" in readme
    assert "--chemprop-bin .venv-chemprop/bin/chemprop" in readme
    assert (
        "release owner's internal `release/package_plan.csv`" in normalized_readme
    )
    assert "is intentionally not bundled" in normalized_readme


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_repeated_csv(
    path: Path, columns: list[str], row: dict, count: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for _ in range(count):
            writer.writerow(row)


def _replace_column(path: Path, column: str, values: list[object]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == len(values)
    for row, value in zip(rows, values):
        row[column] = value
    _write_csv(path, columns, rows)


def _held_out_rows() -> list[dict]:
    return [
        {"split": "valid", "smiles": "CCO", "label": 1.0, "prediction": 1.1},
        {"split": "test", "smiles": "CCN", "label": 2.0, "prediction": 2.1},
        {"split": "test", "smiles": "CCC", "label": 3.0, "prediction": 2.9},
    ]


def _write_neural_run(base: Path, model: str, *, complete_status: bool = False) -> None:
    stem = f"{TASK}_{model}_seed{SEED}"
    rows = _held_out_rows()
    _write_csv(
        base / "predictions" / f"{stem}.csv",
        ["split", "smiles", "label", "prediction"],
        rows,
    )
    metrics = {
        "task": TASK,
        "model": model,
        "seed": SEED,
        "valid_metric": 0.1,
        "test_metric": 0.1,
    }
    if complete_status:
        metrics.update(
            {
                "status": "complete",
                "official_test_preserved": True,
                "chemprop_version": "2.2.1",
                "prediction_rows": 3,
            }
        )
    path = base / "metrics" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))
    checkpoint = base / "checkpoints" / f"{stem}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{model}-checkpoint".encode())


def _approve_policy(path: Path) -> None:
    policy = release.default_release_policy()
    for item in policy["items"].values():
        if item["blocking"]:
            item["status"] = "APPROVED"
            item["decision"] = "MIT"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy))


def _synthetic_project(tmp_path: Path, *, with_legacy: bool = False) -> dict[str, Path]:
    root = tmp_path / "project"
    for path in [
        root / "requirements.txt",
        root / "environment.yml",
        root / "pyproject.toml",
        root / "experiments" / "run_traditional_grid.py",
        root / "experiments" / "run_traditional_predictions.py",
        root / "experiments" / "run_neural_a40.py",
        root / "src" / "admet_benchmark" / "neural_models.py",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {path.name}\n")
    config = root / "experiments" / "config" / "tdc_official.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("seeds: [1]\n")
    (config.parent / "gin.yaml").write_text(
        "data:\n"
        f"  model_input_representation_policy: {SMILES_COMPATIBILITY_POLICY}\n"
    )
    (config.parent / "chemprop_dmpnn.yaml").write_text(
        "data:\n"
        f"  model_input_representation_policy: {SMILES_COMPATIBILITY_POLICY}\n"
    )

    task_data = root / "data" / "tdc_official" / "admet_group" / TASK.lower()
    _write_csv(
        task_data / "train_val.csv",
        ["Drug", "Y"],
        [{"Drug": "C" * (i + 1), "Y": float(i)} for i in range(8)],
    )
    _write_csv(
        task_data / "test.csv",
        ["Drug", "Y"],
        [{"Drug": "CCN", "Y": 2.0}, {"Drug": "CCC", "Y": 3.0}],
    )

    metrics_path = root / "results" / "tables" / "traditional.csv"
    _write_csv(
        metrics_path,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "seed",
            "valid_n",
            "test_n",
            "valid_mae",
            "test_mae",
        ],
        [
            {
                "task": TASK.lower(),
                "task_type": "regression",
                "official_metric": "mae",
                "model": "rf",
                "seed": SEED,
                "valid_n": 1,
                "test_n": 2,
                "valid_mae": 0.1,
                "test_mae": 0.1,
            }
        ],
    )
    trad_predictions = root / "results" / "predictions" / "traditional"
    traditional_rows = []
    split_indices: Counter[str] = Counter()
    for row in _held_out_rows():
        index = split_indices[row["split"]]
        split_indices[row["split"]] += 1
        traditional_rows.append(
            {
                "task": TASK.lower(),
                "task_type": "regression",
                "official_metric": "mae",
                "model": "rf",
                "seed": SEED,
                "split": row["split"],
                "sample_index": index,
                "smiles": row["smiles"],
                "y_true": row["label"],
                "y_pred": row["prediction"],
            }
        )
    _write_csv(
        trad_predictions / TASK.lower() / "rf_seed1.csv",
        sorted(release.TRADITIONAL_PREDICTION_COLUMNS),
        traditional_rows,
    )

    neural = root / "results" / "neural"
    _write_neural_run(neural, "gin")
    if with_legacy:
        _write_neural_run(neural, "dmpnn")

    chemprop = root / "results" / "neural_chemprop"
    _write_neural_run(chemprop, "chemprop_dmpnn", complete_status=True)
    stem = f"{TASK}_chemprop_dmpnn_seed{SEED}"
    config_path = chemprop / "configs" / f"{stem}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": TASK,
                "model": "chemprop_dmpnn",
                "seed": SEED,
                "task_type": "regression",
                "official_metric": "mae",
                "software": {"chemprop": "2.2.1"},
                "training": {"data_seed": SEED, "pytorch_seed": SEED},
                "architecture": {
                    "directed_bond_messages": True,
                    "atom_messages": False,
                    "undirected": False,
                },
                "command": ["/tmp/chemprop/bin/python", "train"],
                "python_executable": "/tmp/chemprop/bin/python",
                "api_token": "must-not-leak",
                "data": {
                    "official_test_preserved": True,
                    "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
                    "model_input_repairs_n": 0,
                    "model_input_repairs": [],
                    "valid_n": 1,
                    "test_n": 2,
                    "train_sha256": "a" * 64,
                    "valid_sha256": "b" * 64,
                    "test_sha256": "c" * 64,
                    "source_train_sha256": "a" * 64,
                    "source_valid_sha256": "b" * 64,
                    "source_test_sha256": "c" * 64,
                },
            }
        )
    )

    output = root / "release"
    policy = output / "release_policy.json"
    _approve_policy(policy)
    return {
        "root": root,
        "output": output,
        "policy": policy,
        "metrics": metrics_path,
        "trad_predictions": trad_predictions,
        "neural": neural,
        "chemprop": chemprop,
    }


def _build(paths: dict[str, Path]) -> dict:
    return release.build_release(
        root=paths["root"],
        output_dir=paths["output"],
        tasks=(TASK,),
        seeds=(SEED,),
        traditional_models=("rf",),
        traditional_metrics_path=paths["metrics"],
        traditional_prediction_dir=paths["trad_predictions"],
        neural_roots=(paths["neural"],),
        chemprop_dir=paths["chemprop"],
        release_policy_path=paths["policy"],
    )


def _file_binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _snapshot(path: Path) -> release.FileSnapshot:
    data = path.read_bytes()
    return release.FileSnapshot(
        path=path,
        relative_path=path.name,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_longtable_shape_counts_only_body_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.tex"
    path.write_text(
        "\\begin{longtable}{lrr}\n"
        "\\caption{Demo}\\\\\n"
        "name & first & second \\\\\n"
        "\\endfirsthead\n"
        "name & first & second \\\\\n"
        "\\endhead\n"
        "\\multicolumn{3}{r}{Continued on next page} \\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        "A \\& B & 1 & 2 \\\\ % escaped ampersand and comment\n"
        "C & 3 & 4 \\\\\n"
        "\\end{longtable}\n",
        encoding="utf-8",
    )

    assert release._latex_longtable_shape(_snapshot(path), "demo") == (2, 3)


@pytest.mark.parametrize(
    "body, expected_error",
    [
        ("A & 1 \\\\\n", "has 2 columns, expected 3"),
        ("A & 1 & 2\n", "is not a complete longtable data row"),
        ("\\multicolumn{3}{l}{A} \\\\\n", "is not a plain data row"),
    ],
)
def test_longtable_shape_rejects_nonplain_or_malformed_body_rows(
    tmp_path: Path, body: str, expected_error: str
) -> None:
    path = tmp_path / "table.tex"
    path.write_text(
        "\\begin{longtable}{lrr}\n"
        "name & first & second \\\\\n"
        "\\endfirsthead\n"
        "name & first & second \\\\\n"
        "\\endhead\n"
        "\\multicolumn{3}{r}{Continued} \\\\\n"
        "\\endfoot\n"
        "\\endlastfoot\n"
        + body
        + "\\end{longtable}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        release._latex_longtable_shape(_snapshot(path), "demo")


def _write_reviewer_defense_supplements(root: Path, tables: Path) -> None:
    out = tables / "reviewer_defense_sensitivities_final"
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(release.normalize_task(task) for task in release.TDC_TASKS)
    metadata = {
        release.normalize_task(task): meta
        for task, meta in release.TDC_TASK_META.items()
    }
    classification = {
        task for task, meta in metadata.items() if meta["task_type"] == "classification"
    }
    regression = set(tasks) - classification
    models = sorted(
        set(release.TRADITIONAL_MODELS) | set(release.SUBMISSION_NEURAL_MODELS)
    )
    families = {
        model: family
        for family, family_models in release.FINAL_REVIEWER_DEFENSE_FAMILIES.items()
        for model in family_models
    }

    endpoint_rows = [
        {
            "endpoint": task,
            "source_row_count": 731,
            "unique_molecule_count": 731,
            "whole_set_sha256": hashlib.sha256(f"set:{task}".encode()).hexdigest(),
            "source_file_sha256": hashlib.sha256(f"file:{task}".encode()).hexdigest(),
        }
        for task in tasks
    ]
    _write_csv(
        out / "cross_endpoint_test_endpoint_summary.csv",
        list(endpoint_rows[0]),
        endpoint_rows,
    )
    overlap_pairs = []
    for index, (left, right) in enumerate(itertools.combinations(tasks, 2)):
        intersection = 538 if index == 0 else 1 if index < 209 else 0
        union = 1462 - intersection
        overlap_pairs.append(
            {
                "endpoint_a": left,
                "endpoint_b": right,
                "endpoint_a_unique_count": 731,
                "endpoint_b_unique_count": 731,
                "intersection_unique_count": intersection,
                "union_unique_count": union,
                "jaccard": intersection / union,
                "has_overlap": intersection > 0,
            }
        )
    _write_csv(
        out / "cross_endpoint_test_overlap_pairwise.csv",
        list(overlap_pairs[0]),
        overlap_pairs,
    )
    _write_csv(
        out / "cross_endpoint_test_overlap_components.csv",
        ["component_id", "endpoint_count", "endpoints"],
        [{"component_id": 1, "endpoint_count": 22, "endpoints": ";".join(tasks)}],
    )
    overlap_checks = {
        "endpoint_count_22": True,
        "pair_count_231": True,
        "endpoint_unique_sum_16082": True,
        "global_union_13818": True,
        "overlapping_pairs_209": True,
        "single_connected_component": True,
        "no_row_level_molecular_output": True,
    }
    (out / "cross_endpoint_test_overlap_summary.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "post_hoc": True,
                "primary_claim": False,
                "endpoint_count": 22,
                "pair_count": 231,
                "endpoint_unique_count_sum": 16082,
                "global_unique_union_count": 13818,
                "cross_endpoint_duplicate_memberships": 2264,
                "pairs_with_overlap": 209,
                "pairs_without_overlap": 22,
                "connected_component_count": 1,
                "largest_component_endpoint_count": 22,
                "maximum_pair_intersection_count": 538,
                "global_whole_set_sha256": "f" * 64,
                "checks": overlap_checks,
            }
        ),
        encoding="utf-8",
    )

    selection_rows = []
    probability_rows = []
    endpoint_regrets = []
    dimension_summaries = []
    dimensions = release.FINAL_REVIEWER_DEFENSE_DIMENSIONS
    tasks_by_dimension = {
        dimension: (
            set(tasks)
            if dimension
            in {"applicability_alignment", "selective_relative_excess_aurc"}
            else classification
            if dimension
            in {"brier", "ece", "selective_relative_excess_aurc_log_loss"}
            else regression
        )
        for dimension in dimensions
    }
    omission_signal_sets: dict[str, dict[str, set[str]]] = {}
    for omitted in models:
        pool = sorted(set(models) - {omitted})
        selected = pool[0]
        pool_text = ";".join(pool)
        for task in tasks:
            meta = metadata[task]
            selection_rows.append(
                {
                    "omitted_model": omitted,
                    "task": task,
                    "task_type": meta["task_type"],
                    "official_metric": meta["metric"],
                    "n_models": 4,
                    "model_pool": pool_text,
                    "validation_selected_model": selected,
                    "validation_selected_score_oriented": 0.8,
                    "validation_runner_up_score_oriented": 0.7,
                    "validation_score_margin": 0.1,
                    "validation_tolerance": 0.005,
                    "validation_tie_models": selected,
                    "validation_tie_count": 1,
                    "selected_exact_bootstrap_probability": 1.0,
                    "selected_tie_bootstrap_probability": 1.0,
                    "shared_seed_bootstrap": True,
                    "bootstrap_seed_count": 5,
                }
            )
            for model in pool:
                probability_rows.append(
                    {
                        "omitted_model": omitted,
                        "task": task,
                        "model": model,
                        "mean_validation_score_oriented": 0.8 if model == selected else 0.7,
                        "exact_selection_probability": 1.0 if model == selected else 0.0,
                        "validation_tie_probability": 1.0 if model == selected else 0.0,
                    }
                )
        if omitted == "gin":
            signals = {
                "effect": {"selective_relative_excess_aurc"},
                "tie": {"selective_relative_excess_aurc"},
                "random": {"applicability_alignment"},
            }
        else:
            signals = {
                "effect": {"applicability_alignment", "ece"},
                "tie": {"applicability_alignment"},
                "random": {"applicability_alignment"},
            }
        omission_signal_sets[omitted] = signals
        for dimension in dimensions:
            for task in sorted(tasks_by_dimension[dimension]):
                higher = dimension == "applicability_alignment"
                if higher:
                    selected_value, oracle_value = 0.4, 0.5
                    loss_min, loss_max, loss_mean = -0.5, -0.1, -0.3
                else:
                    selected_value, oracle_value = 0.2, 0.1
                    loss_min, loss_max, loss_mean = 0.1, 0.5, 0.3
                endpoint_regrets.append(
                    {
                        "omitted_model": omitted,
                        "task": task,
                        "task_type": metadata[task]["task_type"],
                        "official_metric": metadata[task]["metric"],
                        "dimension": dimension,
                        "risk_metric": dimension,
                        "n_models": 4,
                        "model_pool": pool_text,
                        "validation_selected_model": selected,
                        "test_reliability_oracle_model": pool[-1],
                        "selected_reliability_value": selected_value,
                        "oracle_reliability_value": oracle_value,
                        "higher_is_better": higher,
                        "reliability_regret": 0.1,
                        "random_selection_expected_regret": 0.2,
                        "regret_fraction_of_random": 0.5,
                        "regret_minus_random_expected": -0.1,
                        "normalized_regret": 0.25,
                        "practical_tolerance": 0.01,
                        "meaningful_regret": True,
                        "validation_tie_models": selected,
                        "validation_tie_count": 1,
                        "tie_best_case_regret": 0.1,
                        "tie_best_case_normalized_regret": 0.25,
                        "tie_worst_case_regret": 0.1,
                        "tie_robust_meaningful_regret": True,
                        "selected_exact_bootstrap_probability": 1.0,
                        "selected_tie_bootstrap_probability": 1.0,
                        "pool_reliability_loss_min": loss_min,
                        "pool_reliability_loss_max": loss_max,
                        "pool_reliability_loss_range": 0.4,
                        "uniform_random_mean_reliability_loss": loss_mean,
                        "exact_uniform_random_expected_regret": 0.2,
                    }
                )
            dimension_summaries.append(
                {
                    "omitted_model": omitted,
                    "dimension": dimension,
                    "n_endpoints": len(tasks_by_dimension[dimension]),
                    "mean_regret": 0.1,
                    "median_regret": 0.1,
                    "mean_regret_ci_low": 0.09,
                    "mean_regret_ci_high": 0.11,
                    "mean_normalized_regret": 0.25,
                    "median_normalized_regret": 0.25,
                    "mean_normalized_regret_ci_low": 0.2,
                    "mean_normalized_regret_ci_high": 0.3,
                    "nonzero_regret_count": len(tasks_by_dimension[dimension]),
                    "nonzero_regret_fraction": 1.0,
                    "meaningful_regret_count": len(tasks_by_dimension[dimension]),
                    "meaningful_regret_fraction": 1.0,
                    "mean_tie_best_case_regret": 0.1,
                    "mean_tie_best_case_regret_ci_low": 0.09,
                    "mean_tie_best_case_regret_ci_high": 0.11,
                    "mean_tie_best_case_normalized_regret": 0.25,
                    "mean_tie_best_case_normalized_ci_low": 0.2,
                    "mean_tie_best_case_normalized_ci_high": 0.3,
                    "mean_regret_fraction_of_random": 0.5,
                    "median_regret_fraction_of_random": 0.5,
                    "mean_regret_fraction_of_random_ci_low": 0.5,
                    "mean_regret_fraction_of_random_ci_high": 0.6,
                    "fraction_retaining_at_least_half_random_regret": 1.0,
                    "tie_robust_meaningful_count": len(tasks_by_dimension[dimension]),
                    "tie_robust_meaningful_fraction": 1.0,
                    "stable_selection_endpoints": len(tasks_by_dimension[dimension]),
                    "mean_regret_stable_selection": 0.1,
                    "median_selected_exact_bootstrap_probability": 1.0,
                    "median_selected_tie_bootstrap_probability": 1.0,
                    "practical_tolerance": 0.01,
                    "sign_flip_p": 0.001,
                    "holm_p": 0.006,
                    "practical_effect_signal": dimension in signals["effect"],
                    "tie_robust_effect_signal": dimension in signals["tie"],
                    "random_baseline_retention_signal": dimension in signals["random"],
                }
            )
    _write_csv(
        out / "leave_one_model_out_validation_selection.csv",
        list(selection_rows[0]),
        selection_rows,
    )
    _write_csv(
        out / "leave_one_model_out_validation_probabilities.csv",
        list(probability_rows[0]),
        probability_rows,
    )
    _write_csv(
        out / "leave_one_model_out_endpoint.csv",
        list(endpoint_regrets[0]),
        endpoint_regrets,
    )
    _write_csv(
        out / "leave_one_model_out_dimension_summary.csv",
        list(dimension_summaries[0]),
        dimension_summaries,
    )
    lomo_checks = {
        "canonical_reliability_rows_460": True,
        "canonical_validation_rows_550": True,
        "dimension_summary_rows_30": True,
        "endpoint_dimension_rows_460": True,
        "exact_uniform_random_reference_recomputed": True,
        "five_omissions": True,
        "formal_bootstrap_reps_5000": True,
        "formal_permutation_reps_20000": True,
        "formal_selection_bootstrap_reps_1000": True,
        "four_model_pool_every_endpoint": True,
        "no_row_level_molecular_output": True,
        "normalized_regret_recomputed_from_four_model_range": True,
        "oracle_and_pool_range_recomputed": True,
        "selection_rows_110": True,
        "six_dimensions_every_omission": True,
        "validation_reselected_in_each_pool": True,
    }
    omission_gates = {
        omitted: {
            "status": "HOLD" if omitted == "gin" else "GO",
            "frozen_criteria_pass": omitted != "gin",
            "dimensions_with_practical_effect_signal": sorted(signals["effect"]),
            "dimensions_with_tie_robust_signal": sorted(signals["tie"]),
            "dimensions_retaining_random_baseline_regret": sorted(signals["random"]),
        }
        for omitted, signals in omission_signal_sets.items()
    }
    (out / "leave_one_model_out_gate.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "post_hoc": True,
                "primary_claim": False,
                "checks": lomo_checks,
                "omission_effect_gates": omission_gates,
                "frozen_effect_criteria": {"gate": "synthetic frozen criteria"},
            }
        ),
        encoding="utf-8",
    )

    conditional_rows = []
    family_metric_values = {
        "fingerprint": {
            "log_error_ratio_outside_inside": 0.30,
            "uq_correlation_advantage_inside": 0.10,
            "selective_gain_advantage_inside": -0.02,
            "ece_gap_outside_inside": -0.01,
            "brier_gap_outside_inside": 0.05,
            "coverage_error_gap_outside_inside": 0.02,
        },
        "neural": {
            "log_error_ratio_outside_inside": 0.25,
            "uq_correlation_advantage_inside": 0.02,
            "selective_gain_advantage_inside": -0.04,
            "ece_gap_outside_inside": 0.00,
            "brier_gap_outside_inside": 0.03,
            "coverage_error_gap_outside_inside": 0.01,
        },
    }
    classification_metrics = {"ece_gap_outside_inside", "brier_gap_outside_inside"}
    regression_metrics = {"coverage_error_gap_outside_inside"}
    for task in tasks:
        for model in models:
            family = families[model]
            for threshold in release.FINAL_REVIEWER_DEFENSE_THRESHOLDS:
                row = {
                    "task": task,
                    "model": model,
                    "model_family": family,
                    "task_type": metadata[task]["task_type"],
                    "official_metric": metadata[task]["metric"],
                    "n_seeds": 5,
                    "threshold": threshold,
                    "inside_count": 30,
                    "inside_mean_abs_error": 0.1,
                    "inside_uncertainty_error_correlation": 0.2,
                    "inside_selective_gain": 0.1,
                    "inside_ece": 0.1 if task in classification else None,
                    "inside_brier": 0.1 if task in classification else None,
                    "inside_conformal_coverage": 0.9 if task in regression else None,
                    "outside_count": 30,
                    "outside_mean_abs_error": 0.2,
                    "outside_uncertainty_error_correlation": 0.1,
                    "outside_selective_gain": 0.05,
                    "outside_ece": 0.2 if task in classification else None,
                    "outside_brier": 0.2 if task in classification else None,
                    "outside_conformal_coverage": 0.8 if task in regression else None,
                    "eligible": True,
                }
                for metric, value in family_metric_values[family].items():
                    if metric in classification_metrics and task not in classification:
                        row[metric] = None
                    elif metric in regression_metrics and task not in regression:
                        row[metric] = None
                    else:
                        row[metric] = value
                conditional_rows.append(row)
    _write_csv(
        out / "conditional_reliability_by_family.csv",
        list(conditional_rows[0]),
        conditional_rows,
    )

    significant_by_family = {
        "fingerprint": {
            "log_error_ratio_outside_inside",
            "uq_correlation_advantage_inside",
            "brier_gap_outside_inside",
        },
        "neural": {
            "log_error_ratio_outside_inside",
            "brier_gap_outside_inside",
        },
    }
    conditional_summaries = []
    for family in release.FINAL_REVIEWER_DEFENSE_FAMILIES:
        family_models = set(release.FINAL_REVIEWER_DEFENSE_FAMILIES[family])
        for threshold in release.FINAL_REVIEWER_DEFENSE_THRESHOLDS:
            group_rows = [
                row
                for row in conditional_rows
                if row["model"] in family_models and row["threshold"] == threshold
            ]
            pending = []
            for metric in release.FINAL_REVIEWER_DEFENSE_CONDITIONAL_METRICS:
                pairs = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
                by_task: defaultdict[str, list[float]] = defaultdict(list)
                for row in group_rows:
                    if row.get(metric) is not None:
                        by_task[row["task"]].append(float(row[metric]))
                endpoints = [float(np.mean(values)) for values in by_task.values()]
                mean = float(np.mean(endpoints))
                significant = metric in significant_by_family[family]
                pending.append(
                    {
                        "model_family": family,
                        "threshold": threshold,
                        "metric": metric,
                        "n_pairs": len(pairs),
                        "n_endpoints": len(endpoints),
                        "pair_positive_count": sum(value > 0 for value in pairs),
                        "pair_positive_fraction": float(np.mean(np.asarray(pairs) > 0)),
                        "positive_count": sum(value > 0 for value in endpoints),
                        "positive_fraction": float(np.mean(np.asarray(endpoints) > 0)),
                        "mean": mean,
                        "median": float(np.median(endpoints)),
                        "mean_ci_low": (mean - 0.01 if significant else min(-0.01, mean - 0.01)),
                        "mean_ci_high": mean + 0.01,
                        "sign_flip_p": 0.001 if significant else 0.5,
                    }
                )
            order = sorted(range(len(pending)), key=lambda index: pending[index]["sign_flip_p"])
            running = 0.0
            adjusted = {}
            for rank, index in enumerate(order):
                running = max(
                    running,
                    min(1.0, (len(pending) - rank) * pending[index]["sign_flip_p"]),
                )
                adjusted[index] = running
            for index, row in enumerate(pending):
                row["holm_p_within_family_threshold"] = adjusted[index]
                conditional_summaries.append(row)
    _write_csv(
        out / "conditional_reliability_family_summary.csv",
        list(conditional_summaries[0]),
        conditional_summaries,
    )
    conditional_checks = {
        "canonical_source_rows_330": True,
        "frozen_eligibility_preserved": True,
        "five_models_assigned_once": True,
        "two_families": True,
        "three_thresholds_per_family": True,
        "six_metrics_per_family_threshold": True,
        "summary_rows_36": True,
        "holm_within_family_threshold": True,
        "formal_endpoint_bootstrap_reps_5000": True,
        "formal_permutation_reps_20000": True,
        "no_row_level_molecular_output": True,
    }
    family_effect_gates = {}
    effect_metrics = (
        "log_error_ratio_outside_inside",
        "ece_gap_outside_inside",
        "brier_gap_outside_inside",
        "coverage_error_gap_outside_inside",
        "uq_correlation_advantage_inside",
    )
    for family in release.FINAL_REVIEWER_DEFENSE_FAMILIES:
        criteria = {}
        for metric in effect_metrics:
            row = next(
                item
                for item in conditional_summaries
                if item["model_family"] == family
                and item["threshold"] == 0.5
                and item["metric"] == metric
            )
            criteria[metric] = bool(
                row["positive_fraction"] >= 0.6
                and row["mean_ci_low"] > 0
                and row["holm_p_within_family_threshold"] < 0.05
            )
        family_effect_gates[family] = {
            "status": "GO",
            "frozen_original_effect_gate_pass": True,
            "strong_conditional_dimensions": sum(criteria.values()),
            "criteria": criteria,
        }
    (out / "conditional_reliability_family_gate.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "post_hoc": True,
                "primary_claim": False,
                "primary_threshold": 0.5,
                "family_policy": {
                    family: list(family_models)
                    for family, family_models in release.FINAL_REVIEWER_DEFENSE_FAMILIES.items()
                },
                "family_effect_gates": family_effect_gates,
                "checks": conditional_checks,
            }
        ),
        encoding="utf-8",
    )

    for filename in (
        "table_supp_conditional_reliability_families.tex",
        "table_supp_cross_endpoint_test_overlap.tex",
        "table_supp_leave_one_model_out.tex",
    ):
        (out / filename).write_text(
            "\\begin{tabular}{ll}\nAggregate & Value \\\\\n\\end{tabular}\n",
            encoding="utf-8",
        )

    combined_parameters = {
        "bootstrap_reps": release.FINAL_REVIEWER_DEFENSE_BOOTSTRAP_REPS,
        "permutation_reps": release.FINAL_REVIEWER_DEFENSE_PERMUTATION_REPS,
        "selection_bootstrap_reps": release.FINAL_REVIEWER_DEFENSE_SELECTION_BOOTSTRAP_REPS,
        "random_seed": release.FINAL_REVIEWER_DEFENSE_RANDOM_SEED,
    }
    combined_checks = {
        "conditional_family_gate_go": True,
        "cross_endpoint_overlap_gate_go": True,
        "leave_one_model_out_gate_go": True,
        "no_new_primary_claim": True,
        "no_row_level_molecular_output": True,
        "post_hoc_only": True,
        "source_inputs_sha256_and_size_unchanged": True,
        "zero_refit": True,
    }
    (out / "reviewer_defense_sensitivities_gate.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "scope": release.FINAL_REVIEWER_DEFENSE_SCOPE,
                "post_hoc": True,
                "primary_claim": False,
                "parameters": combined_parameters,
                "checks": combined_checks,
            }
        ),
        encoding="utf-8",
    )
    output_names = {
        filename
        for logical_name, (filename, _) in release.FINAL_REVIEWER_DEFENSE_OUTPUTS.items()
        if logical_name != "reviewer_manifest"
    }
    script = root / "experiments" / "analyze_reviewer_defense_sensitivities.py"
    input_paths = [
        *(f"data/tdc_official/admet_group/{task}/test.csv" for task in tasks),
        "results/tables/reliability_regret_final/validation_seed_scores.csv",
        "results/tables/reliability_regret_final/test_reliability_by_task_model.csv",
        "results/tables/conditional_reliability_final/conditional_reliability_by_task_model.csv",
    ]
    manifest = {
        "schema": "reviewer_defense_sensitivities_v1",
        "post_hoc": True,
        "zero_refit": True,
        "primary_claim": False,
        "script": "experiments/analyze_reviewer_defense_sensitivities.py",
        "script_sha256": _file_binding(script)["sha256"],
        "parameters": combined_parameters,
        "input_files": [
            {"path": path, "bytes": 1, "sha256": hashlib.sha256(path.encode()).hexdigest()}
            for path in input_paths
        ],
        "output_files": [
            {
                "path": name,
                "bytes": _file_binding(out / name)["bytes"],
                "sha256": _file_binding(out / name)["sha256"],
            }
            for name in sorted(output_names)
        ],
    }
    (out / "reviewer_defense_sensitivities_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_selected_model_sensitivities(root: Path, tables: Path) -> None:
    selection_path = (
        tables / "reliability_regret_final" / "validation_selection_by_task.csv"
    )
    conditional_path = (
        tables
        / "conditional_reliability_final"
        / "conditional_reliability_by_task_model.csv"
    )
    calibration_path = tables / "calibration_summary_final.csv"
    tasks = sorted(release.normalize_task(task) for task in release.TDC_TASKS)
    models = (*release.TRADITIONAL_MODELS, *release.SUBMISSION_NEURAL_MODELS)
    _write_csv(
        selection_path,
        ["task", "validation_selected_model"],
        [
            {"task": task, "validation_selected_model": models[index % len(models)]}
            for index, task in enumerate(tasks)
        ],
    )
    _write_csv(conditional_path, ["value"], [{"value": 1}])
    _write_csv(calibration_path, ["value"], [{"value": 1}])

    validation_dir = tables / "validation_selected_sensitivity_final"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = {
        "selected_calibration_by_endpoint_method.csv": 35,
        "selected_calibration_gain_summary.csv": 4,
        "selected_conditional_reliability_by_endpoint.csv": 66,
        "selected_conditional_reliability_summary.csv": 18,
        "selected_conformal_summary.csv": 1,
    }
    for filename, rows in validation_rows.items():
        _write_csv(
            validation_dir / filename,
            ["value"],
            [{"value": index} for index in range(rows)],
        )
    (validation_dir / "VALIDATION_SELECTED_SENSITIVITY.md").write_text(
        "# Synthetic post-hoc sensitivity\n\nZero-refit descriptive check.\n",
        encoding="utf-8",
    )
    (validation_dir / "table_supp_validation_selected_sensitivity.tex").write_text(
        "\\begin{tabular}{l}value \\\\ \\end{tabular}\n",
        encoding="utf-8",
    )
    validation_output_names = {
        filename
        for logical_name, (filename, _) in release.FINAL_VALIDATION_SELECTED_OUTPUTS.items()
        if logical_name != "validation_selected_manifest"
    }
    manifest = {
        "technical_status": "PASS",
        "strict_final_contract": True,
        "checks": {
            "endpoint_level_aggregation": True,
            "one_validation_selected_model_per_endpoint": True,
            "source_inputs_sha256_and_size_unchanged": True,
            "zero_refit": True,
            "selected_calibration_endpoint_method_rows": 35,
            "selected_conditional_rows": 66,
        },
        "input_files": [
            {
                **_file_binding(root / relative),
                "path": relative,
            }
            for relative, _ in release.FINAL_VALIDATION_SELECTED_INPUTS.values()
        ],
        "output_files": [
            {
                **_file_binding(validation_dir / filename),
                "path": (
                    "results/tables/validation_selected_sensitivity_final/"
                    + filename
                ),
                **(
                    {"rows": validation_rows[filename]}
                    if filename in validation_rows
                    else {}
                ),
            }
            for filename in sorted(validation_output_names)
        ],
    }
    (validation_dir / "validation_selected_sensitivity_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    support_dir = tables / "selected_support_risk_coverage_final"
    support_dir.mkdir(parents=True, exist_ok=True)
    support_simple_rows = {
        "ensemble_sample_alignment_audit.csv": 110,
        "model_seed_availability.csv": 110,
        "run_sample_alignment_audit.csv": 550,
    }
    for filename, rows in support_simple_rows.items():
        _write_csv(
            support_dir / filename,
            ["value"],
            [{"value": index} for index in range(rows)],
        )
    _write_csv(
        support_dir / "prediction_input_audit.csv",
        ["path", "included"],
        [
            {"path": str((root / f"private/prediction_{index}.csv").resolve()), "included": True}
            for index in range(550)
        ],
    )
    _write_csv(
        support_dir / "validation_seed_scores_recomputed.csv",
        ["source", "value"],
        [
            {"source": str((root / f"private/validation_{index}.csv").resolve()), "value": 1}
            for index in range(550)
        ],
    )
    endpoint_gains = [0.2] * 20 + [-0.1] * 2
    _write_csv(
        support_dir / "selected_support_risk_coverage_by_endpoint.csv",
        ["task", "support_gain_vs_random"],
        [
            {"task": task, "support_gain_vs_random": endpoint_gains[index]}
            for index, task in enumerate(tasks)
        ],
    )
    _write_csv(
        support_dir / "selected_support_risk_coverage_curve.csv",
        ["coverage", "cumulative_risk"],
        [
            {"coverage": (index + 1) / 32758, "cumulative_risk": 0.1}
            for index in range(32758)
        ],
    )
    mean_gain = float(np.mean(endpoint_gains))
    summary_rows = [
        {
            "scope": "all",
            "metric": "support_gain_vs_random",
            "mean": mean_gain,
            "positive_endpoints": 20,
        },
        *[
            {
                "scope": "synthetic",
                "metric": f"metric_{index}",
                "mean": 0.0,
                "positive_endpoints": 0,
            }
            for index in range(8)
        ],
    ]
    _write_csv(
        support_dir / "selected_support_risk_coverage_summary.csv",
        ["scope", "metric", "mean", "positive_endpoints"],
        summary_rows,
    )
    (support_dir / "table_selected_support_risk_coverage.tex").write_text(
        "\\begin{tabular}{l}value \\\\ \\end{tabular}\n",
        encoding="utf-8",
    )
    support_gate = {
        "status": "PASS",
        "zero_refit": True,
        "test_outcomes_used_for_model_selection": False,
        "technical_checks": {
            "exact_22_endpoints": True,
            "exact_one_frozen_choice_per_endpoint": True,
            "exact_five_seeds_per_choice": True,
            "exact_550_locked_prediction_runs": True,
            "run_sample_alignment_exact": True,
            "ensemble_sample_alignment_exact": True,
            "complete_locked_similarity": True,
            "all_endpoint_metrics_finite": True,
        },
        "coverage": {
            "endpoints": 22,
            "classification_endpoints": 13,
            "regression_endpoints": 9,
        },
        "input_sha256": {
            "selection": _file_binding(selection_path)["sha256"],
            "prediction_files_combined": "a" * 64,
            "similarity_caches_combined": "b" * 64,
            "analysis_script": _file_binding(
                root / "experiments/analyze_selected_support_risk_coverage.py"
            )["sha256"],
        },
        "primary_descriptive_result": {
            "mean": mean_gain,
            "positive_endpoints": 20,
            "n_endpoints": 22,
        },
    }
    (support_dir / "selected_support_risk_coverage_gate.json").write_text(
        json.dumps(support_gate), encoding="utf-8"
    )


def _write_final_supplements(paths: dict[str, Path]) -> None:
    root = paths["root"]
    tables = root / "results" / "tables"
    cleaned_dir = tables / "cleaned_sensitivity_locked_cache"
    predictions = root / "results" / "predictions" / "cleaned_sensitivity_locked_cache"

    neural_precheck = (
        tables
        / "final_neural_artifact_precheck_final"
        / "final_neural_precheck_gate.json"
    )
    neural_precheck.parent.mkdir(parents=True, exist_ok=True)
    neural_precheck.write_text(
        json.dumps(
            {
                "status": "GO",
                "expected_total_formal_neural_runs": 220,
                "expected_per_model": 110,
                "required_models": ["gin", "chemprop_dmpnn"],
                "required_tasks": 22,
                "required_seeds": [1, 2, 3, 4, 5],
                "complete_by_model": {"gin": 110, "chemprop_dmpnn": 110},
                "legacy_exclusion_ok": True,
                "sample_alignment": {
                    "official_split_verification_enabled": True,
                    "all_expected_runs_match_official_samples": True,
                },
                "representation_compatibility": {
                    "policy": SMILES_COMPATIBILITY_POLICY,
                    "gin_global_config_ok": True,
                    "chemprop_global_config_ok": True,
                    "chemprop_all_included_runs_valid": True,
                    "source_smiles": "must-not-release",
                },
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    (tables / "final_neural_checkpoint_integrity.json").write_text(
        json.dumps(
            {
                "gin": {"files": 110, "zip_integrity_ok": 110},
                "chemprop_dmpnn": {"files": 110, "zip_integrity_ok": 110},
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    for relative, _ in release.FINAL_PUBLIC_RESULT_OUTPUTS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text('{"status": "GO"}\n', encoding="utf-8")
        else:
            _write_csv(path, ["metric", "value"], [{"metric": "demo", "value": 1}])

    config = root / "experiments" / "config" / "sensitivity.yaml"
    config.write_text(
        "protocol: paired-common-test-v1\n"
        "selected_tasks_for_cleaned_rerun:\n"
        + "".join(f"  - {task}\n" for task in release.FINAL_CLEANED_TASKS)
        + "models: [rf, xgboost, lightgbm]\n"
        + "seeds: [1, 2, 3, 4, 5]\n",
        encoding="utf-8",
    )
    # Materialize the same explicit public code closure required by a formal
    # build. Existing synthetic Python placeholders are overwritten because
    # the AST gate must see valid source, while real test configs are retained.
    for relative in release.REQUIRED_PUBLIC_SOFTWARE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py":
            path.write_text(f"# synthetic {path.name}\n", encoding="utf-8")
        elif path.suffix == ".sh":
            path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        elif path.suffix == ".json" and not path.exists():
            path.write_text("{}\n", encoding="utf-8")
        elif not path.exists():
            path.write_text(f"synthetic {path.name}\n", encoding="utf-8")

    input_hashes = {
        task: {"train_val.csv": "a" * 64, "test.csv": "b" * 64}
        for task in release.FINAL_CLEANED_TASKS
    }
    manifest_path = tables / "cleaned_sensitivity_locked_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": "paired-common-test-v1",
                "config_path": str(config.resolve()),
                "config_logical_path": "experiments/config/sensitivity.yaml",
                "input_sha256": input_hashes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run_rows = [
        {"task": task, "model": model, "seed": seed}
        for task in release.FINAL_CLEANED_TASKS
        for model in release.FINAL_CLEANED_MODELS
        for seed in release.FINAL_CLEANED_SEEDS
    ]
    runs_path = tables / "cleaned_sensitivity_locked_cache_runs.csv"
    _write_csv(runs_path, ["task", "model", "seed"], run_rows)
    raw_audit_path = tables / "cleaned_sensitivity_locked_cache_audit.jsonl"
    raw_audit_path.write_text(
        "\n".join(
            json.dumps({"task": task, "seed": seed})
            for task in release.FINAL_CLEANED_TASKS
            for seed in release.FINAL_CLEANED_SEEDS
        )
        + "\n",
        encoding="utf-8",
    )

    audit_columns = [
        "task",
        "model",
        "seed",
        "prediction_path",
        "prediction_sha256",
        "prediction_bytes",
        "rows",
        *release.CLEANED_PREDICTION_AUDIT_TRUE_COLUMNS,
    ]
    audit_rows = []
    inventory_rows = []
    prediction_paths = []
    for task in release.FINAL_CLEANED_TASKS:
        for model in release.FINAL_CLEANED_MODELS:
            for seed in release.FINAL_CLEANED_SEEDS:
                name = f"{task.lower()}_{model}_seed{seed}_paired.csv"
                path = predictions / name
                _write_csv(
                    path,
                    [
                        "task",
                        "model",
                        "seed",
                        "evaluation_set",
                        "canonical_smiles",
                        "label",
                        "official_prediction",
                        "cleaned_prediction",
                    ],
                    [
                        {
                            "task": task,
                            "model": model,
                            "seed": seed,
                            "evaluation_set": "common_cleaned_test",
                            "canonical_smiles": "CCO",
                            "label": 1,
                            "official_prediction": 0.4,
                            "cleaned_prediction": 0.6,
                        }
                    ],
                )
                binding = _file_binding(path)
                prediction_paths.append(path)
                audit_rows.append(
                    {
                        "task": task.lower(),
                        "model": model,
                        "seed": seed,
                        "prediction_path": str(path.resolve()),
                        "prediction_sha256": binding["sha256"],
                        "prediction_bytes": binding["bytes"],
                        "rows": 1,
                        **{
                            column: True
                            for column in release.CLEANED_PREDICTION_AUDIT_TRUE_COLUMNS
                        },
                    }
                )
                inventory_rows.append(
                    {
                        "filename": name,
                        "task": task.lower(),
                        "model": model,
                        "seed": seed,
                        "sha256": binding["sha256"],
                        "bytes": binding["bytes"],
                        "rows": 1,
                    }
                )
    prediction_audit = cleaned_dir / "cleaned_sensitivity_prediction_audit.csv"
    _write_csv(prediction_audit, audit_columns, audit_rows)
    audit_binding = _file_binding(prediction_audit)
    inventory_sha = release._prediction_inventory_digest(inventory_rows)
    entries = {
        row["filename"]: {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "rows": row["rows"],
        }
        for row in inventory_rows
    }
    prediction_gate = {
        "status": "GO",
        "complete_expected_matrix": True,
        "sample_label_identity_aligned_within_task": True,
        "source_common_test_reconstructed_exactly": True,
        "prediction_metrics_recomputed_match_runs": True,
        "expected_prediction_files": 120,
        "observed_prediction_files": 120,
        "audited_prediction_files": 120,
        "prediction_inventory_sha256": inventory_sha,
        "prediction_inventory": {
            "sha256": inventory_sha,
            "files": 120,
            "bytes": sum(int(row["bytes"]) for row in inventory_rows),
            "rows": 120,
            "entries": entries,
        },
        "audit_csv": {**audit_binding, "rows": 120},
    }
    prediction_gate_path = cleaned_dir / "cleaned_sensitivity_prediction_gate.json"
    prediction_gate_path.write_text(json.dumps(prediction_gate), encoding="utf-8")

    summary_gate = {
        "status": "SENSITIVITY_STABLE",
        "complete_expected_matrix": True,
        "split_source_gate": True,
        "split_source": "locked_cache",
        "expected_rows": 120,
        "observed_rows": 120,
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
        "n_tasks": 8,
        "run_integrity_gate": {
            "finite_core_values": True,
            "strict_integer_fields": True,
            "fingerprint_contract_matches": True,
            "single_thread_cpu_contract_matches": True,
            "task_metadata_matches_tdc": True,
            "primary_metric_columns_match": True,
            "paired_delta_algebra_matches": True,
            "sample_counts_consistent_across_models": True,
            "validated_rows": 120,
        },
        "manifest_provenance_gate": {"checks": {"all": True}},
        "audit_provenance_gate": {
            "exact_task_seed_matrix": True,
            "schema_v2": True,
            "post_precedence_overlap_zero": True,
            "parser_repair_counts_match_sources": True,
            "split_output_counts_match_runs": True,
            "expected_payloads": 40,
        },
        "input_artifacts": {
            "runs": {**_file_binding(runs_path), "rows": 120},
            "audit": {**_file_binding(raw_audit_path), "payloads": 40},
            "manifest": _file_binding(manifest_path),
            "config": _file_binding(config),
        },
    }
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    (cleaned_dir / "cleaned_sensitivity_gate.json").write_text(
        json.dumps(summary_gate), encoding="utf-8"
    )
    for name in (
        "cleaned_sensitivity_by_task.csv",
        "cleaned_sensitivity_by_task_model.csv",
        "cleaned_sensitivity_audit.csv",
    ):
        _write_csv(cleaned_dir / name, ["value"], [{"value": 1}])

    cpu_paths = {
        "y_randomization_summary": tables / "y_randomization_all_tasks_locked_cache.csv",
        "y_randomization_scores": tables
        / "y_randomization_all_tasks_scores_locked_cache.csv",
        "shap": tables / "shap_selected_locked_cache.csv",
        "shap_mapping": tables / "shap_morgan_bit_substructures_locked_cache.csv",
        "traditional_selection": tables
        / "traditional_grid_locked_cache_best_by_task.csv",
    }
    _write_csv(
        cpu_paths["y_randomization_summary"],
        ["value"],
        [{"value": index} for index in range(22)],
    )
    _write_csv(
        cpu_paths["y_randomization_scores"],
        ["value"],
        [{"value": index} for index in range(220)],
    )
    _write_csv(
        cpu_paths["shap"],
        ["value"],
        [{"value": index} for index in range(160)],
    )
    _write_csv(
        cpu_paths["shap_mapping"],
        [
            "task",
            "model",
            "seed",
            "rank",
            "feature",
            "bit_id",
            "mean_abs_shap",
            "example_smiles",
            "bit_atom_index",
            "bit_radius",
            "substructure_smiles",
        ],
        [
            {
                "task": "Caco2_Wang",
                "model": "rf",
                "seed": 1,
                "rank": index + 1,
                "feature": f"morgan_bit_{index}",
                "bit_id": index,
                "mean_abs_shap": 0.1,
                "example_smiles": "CCO",
                "bit_atom_index": 0,
                "bit_radius": 2,
                "substructure_smiles": "CO",
            }
            for index in range(80)
        ],
    )
    _write_csv(
        cpu_paths["traditional_selection"],
        ["value"],
        [{"value": index} for index in range(22)],
    )
    cpu_gate = {
        "status": "GO",
        "artifact_gate": "PASS",
        "negative_control_signal": "PASS",
        "coverage": {
            "y_summary_rows": 22,
            "y_score_rows": 220,
            "shap_rows": 160,
            "shap_mapping_rows": 80,
            "mapped_substructures": 80,
        },
        "y_randomization": {"positive_margin_count": 22},
        "inputs": {name: _file_binding(path) for name, path in cpu_paths.items()},
    }
    (tables / "locked_cache_cpu_defense_gate.json").write_text(
        json.dumps(cpu_gate), encoding="utf-8"
    )

    ablation_dir = tables / "posthoc_ablations_final"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "ablation_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "all_task_seed_model_samples_aligned": True,
                "zero_model_fits": True,
                "expected_runs": 550,
                "observed_runs": 550,
                "ensemble_rows": 2310,
                "calibration_rows": 7800,
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "calibration_binning_sensitivity.csv",
        "ensemble_size_by_seed_subset.csv",
        "ensemble_size_summary.csv",
        "input_prediction_audit.csv",
        "sample_alignment_audit.csv",
    ):
        _write_csv(ablation_dir / name, ["value"], [{"value": 1}])

    crossfit_dir = tables / "crossfit_oracle_sensitivity_final"
    crossfit_dir.mkdir(parents=True, exist_ok=True)
    classification_tasks = sum(
        meta["task_type"] == "classification"
        for meta in release.TDC_TASK_META.values()
    )
    endpoint_dimensions = classification_tasks * 6 + (
        len(release.TDC_TASK_META) - classification_tasks
    ) * 3
    expected_fold_rows = (
        len(release.FINAL_CROSSFIT_GROUPING_SCHEMES)
        * endpoint_dimensions
        * release.FINAL_CROSSFIT_REPEATS
        * 2
    )
    fold_path = crossfit_dir / "crossfit_oracle_fold_level.csv"
    _write_repeated_csv(
        fold_path,
        [
            "task",
            "grouping_scheme",
            "repeat",
            "direction",
            "dimension",
            "usable",
            "signed_d",
            "oracle_tie_count",
            "tie_average_usable",
            "tie_averaged_signed_d",
            "group_overlap_count",
        ],
        {
            "task": "ames",
            "grouping_scheme": "bemis_murcko",
            "repeat": 0,
            "direction": "A_to_B",
            "dimension": "brier",
            "usable": True,
            "signed_d": 0.01,
            "oracle_tie_count": 1,
            "tie_average_usable": True,
            "tie_averaged_signed_d": 0.01,
            "group_overlap_count": 0,
        },
        expected_fold_rows,
    )
    endpoint_path = crossfit_dir / "crossfit_oracle_endpoint_summary.csv"
    locked_task_meta = {
        release.normalize_task(task): meta
        for task, meta in release.TDC_TASK_META.items()
    }
    classification_tasks = {
        task
        for task, meta in locked_task_meta.items()
        if meta["task_type"] == "classification"
    }
    all_tasks = set(locked_task_meta)
    regression_tasks = all_tasks - classification_tasks
    classification_dimensions = {
        "brier",
        "ece_5bin",
        "ece_10bin",
        "selective_relative_excess_aurc_log_loss",
    }
    tasks_by_dimension = {
        dimension: (
            classification_tasks
            if dimension in classification_dimensions
            else regression_tasks
            if dimension == "conformal_coverage_error"
            else all_tasks
        )
        for dimension in release.FINAL_CROSSFIT_DIMENSIONS
    }
    endpoint_rows = [
        {
            "grouping_scheme": scheme,
            "scheme_role": "primary" if scheme == "bemis_murcko" else "secondary",
            "task": task,
            "task_type": locked_task_meta[task]["task_type"],
            "dimension": dimension,
            "evidence_tier": release.FINAL_CROSSFIT_EVIDENCE_TIERS[dimension],
            "complete_repeats": release.FINAL_CROSSFIT_REPEATS,
            "usable_direction_rows": 2 * release.FINAL_CROSSFIT_REPEATS,
            "tie_average_usable_direction_rows": 2
            * release.FINAL_CROSSFIT_REPEATS,
            "mean_signed_d": 0.01,
            "mean_tie_averaged_signed_d": 0.01,
            "mean_fixed_minus_tie_averaged_signed_d": 0.0,
            "oracle_tie_direction_count": 0,
            "oracle_tie_direction_fraction": 0.0,
        }
        for scheme in release.FINAL_CROSSFIT_GROUPING_SCHEMES
        for dimension in release.FINAL_CROSSFIT_DIMENSIONS
        for task in sorted(tasks_by_dimension[dimension])
    ]
    _write_csv(
        endpoint_path,
        [
            "grouping_scheme",
            "scheme_role",
            "task",
            "task_type",
            "dimension",
            "evidence_tier",
            "complete_repeats",
            "usable_direction_rows",
            "tie_average_usable_direction_rows",
            "mean_signed_d",
            "mean_tie_averaged_signed_d",
            "mean_fixed_minus_tie_averaged_signed_d",
            "oracle_tie_direction_count",
            "oracle_tie_direction_fraction",
        ],
        endpoint_rows,
    )
    dimension_path = crossfit_dir / "crossfit_oracle_dimension_summary.csv"
    synthetic_raw_p = {
        dimension: 2.0 / (2 ** len(tasks_by_dimension[dimension]))
        for dimension in release.FINAL_CROSSFIT_DIMENSIONS
    }
    synthetic_holm: dict[str, float] = {}
    running_holm = 0.0
    ordered_synthetic = sorted(
        synthetic_raw_p.items(), key=lambda item: (item[1], item[0])
    )
    for rank, (dimension, pvalue) in enumerate(ordered_synthetic):
        running_holm = max(
            running_holm,
            min(1.0, (len(ordered_synthetic) - rank) * pvalue),
        )
        synthetic_holm[dimension] = running_holm
    _write_csv(
        dimension_path,
        [
            "grouping_scheme",
            "dimension",
            "endpoint_count",
            "mean_endpoint_signed_d",
            "mean_endpoint_signed_d_ci_low",
            "mean_endpoint_signed_d_ci_high",
            "bootstrap_unit",
            "bootstrap_reps",
            "sign_flip_unit",
            "sign_flip_method",
            "sign_flip_assignments",
            "sign_flip_extreme_count",
            "sign_flip_p_exact",
            "holm_family",
            "holm_family_size",
            "holm_p_within_grouping_7",
            "holm_reject_0_05",
            "mean_endpoint_tie_averaged_signed_d",
            "mean_oracle_tie_direction_fraction",
            "tie_averaged_sign_flip_assignments",
            "tie_averaged_sign_flip_extreme_count",
            "tie_averaged_sign_flip_p_exact",
            "tie_averaged_holm_p_within_grouping_7",
            "tie_averaged_holm_reject_0_05",
        ],
        [
            {
                "grouping_scheme": scheme,
                "dimension": dimension,
                "endpoint_count": len(tasks_by_dimension[dimension]),
                "mean_endpoint_signed_d": 0.01,
                "mean_endpoint_signed_d_ci_low": 0.005,
                "mean_endpoint_signed_d_ci_high": 0.015,
                "bootstrap_unit": "endpoint",
                "bootstrap_reps": release.FINAL_CROSSFIT_BOOTSTRAP_REPS,
                "sign_flip_unit": "endpoint_mean",
                "sign_flip_method": "exact_enumeration_two_sided",
                "sign_flip_assignments": 2 ** len(tasks_by_dimension[dimension]),
                "sign_flip_extreme_count": 2,
                "sign_flip_p_exact": f"{synthetic_raw_p[dimension]:.12g}",
                "holm_family": "all_7_dimensions_within_grouping",
                "holm_family_size": len(release.FINAL_CROSSFIT_DIMENSIONS),
                "holm_p_within_grouping_7": f"{synthetic_holm[dimension]:.12g}",
                "holm_reject_0_05": synthetic_holm[dimension] < 0.05,
                "mean_endpoint_tie_averaged_signed_d": 0.01,
                "mean_oracle_tie_direction_fraction": 0.0,
                "tie_averaged_sign_flip_assignments": 2
                ** len(tasks_by_dimension[dimension]),
                "tie_averaged_sign_flip_extreme_count": 2,
                "tie_averaged_sign_flip_p_exact": f"{synthetic_raw_p[dimension]:.12g}",
                "tie_averaged_holm_p_within_grouping_7": f"{synthetic_holm[dimension]:.12g}",
                "tie_averaged_holm_reject_0_05": synthetic_holm[dimension] < 0.05,
            }
            for scheme in release.FINAL_CROSSFIT_GROUPING_SCHEMES
            for dimension in release.FINAL_CROSSFIT_DIMENSIONS
        ],
    )
    overlap_path = crossfit_dir / "locked_split_overlap_audit.csv"
    _write_csv(
        overlap_path,
        [
            "task",
            "canonical_overlap_count",
            "scaffold_group_overlap_count",
            "nonempty_bemis_murcko_overlap_count",
            "prediction_test_matches_locked_source_order_labels",
        ],
        [
            {
                "task": task,
                "canonical_overlap_count": 0,
                "scaffold_group_overlap_count": 0,
                "nonempty_bemis_murcko_overlap_count": 0,
                "prediction_test_matches_locked_source_order_labels": True,
            }
            for task in release.TDC_TASKS
        ],
    )
    source_hash_path = crossfit_dir / "locked_split_source_sha256.csv"
    _write_csv(
        source_hash_path,
        [
            "task",
            "split",
            "path",
            "file_sha256",
            "ordered_smiles_sha256",
            "ordered_labels_sha256",
        ],
        [
            {
                "task": task,
                "split": split,
                "path": f"data/tdc_official/admet_group/{task.lower()}/{split}.csv",
                "file_sha256": "a" * 64,
                "ordered_smiles_sha256": "b" * 64,
                "ordered_labels_sha256": "c" * 64,
            }
            for task in release.TDC_TASKS
            for split in ("train_val", "test")
        ],
    )
    formal_run_count = (
        len(release.TDC_TASKS)
        * len(release.SEEDS)
        * (len(release.TRADITIONAL_MODELS) + len(release.SUBMISSION_NEURAL_MODELS))
    )
    prediction_audit_path = crossfit_dir / "prediction_input_audit.csv"
    _write_repeated_csv(
        prediction_audit_path,
        ["path", "included", "reason"],
        {
            "path": str((root / "results/predictions/formal.csv").resolve()),
            "included": True,
            "reason": "",
        },
        formal_run_count,
    )
    run_alignment_path = crossfit_dir / "run_sample_alignment_audit.csv"
    _write_repeated_csv(
        run_alignment_path,
        ["value"],
        {"value": 1},
        formal_run_count,
    )
    model_count = len(release.TRADITIONAL_MODELS) + len(
        release.SUBMISSION_NEURAL_MODELS
    )
    ensemble_count = len(release.TDC_TASKS) * model_count
    availability_path = crossfit_dir / "model_seed_availability.csv"
    _write_repeated_csv(
        availability_path, ["value"], {"value": 1}, ensemble_count
    )
    ensemble_alignment_path = crossfit_dir / "ensemble_sample_alignment_audit.csv"
    _write_repeated_csv(
        ensemble_alignment_path, ["value"], {"value": 1}, ensemble_count
    )
    validation_scores_path = crossfit_dir / "validation_seed_scores.csv"
    _write_repeated_csv(
        validation_scores_path,
        ["value", "source"],
        {
            "value": 1,
            "source": str((root / "results/predictions/formal.csv").resolve()),
        },
        formal_run_count,
    )
    fixed_selection_path = crossfit_dir / "fixed_validation_selection.csv"
    _write_repeated_csv(
        fixed_selection_path,
        ["value"],
        {"value": 1},
        len(release.TDC_TASKS),
    )
    crossfit_manuscript_dir = tables / "manuscript_final"
    crossfit_manuscript_dir.mkdir(parents=True, exist_ok=True)
    crossfit_summary_csv = (
        crossfit_manuscript_dir / "table_supp_crossfit_oracle_summary.csv"
    )
    _write_csv(
        crossfit_summary_csv,
        [
            "grouping",
            "dimension",
            "n_endpoints",
            "mean_signed_d",
            "sign_flip_p_exact",
            "holm_p_within_grouping_7",
        ],
        [
            {
                "grouping": scheme,
                "dimension": dimension,
                "n_endpoints": len(tasks_by_dimension[dimension]),
                "mean_signed_d": 0.01,
                "sign_flip_p_exact": f"{synthetic_raw_p[dimension]:.12g}",
                "holm_p_within_grouping_7": f"{synthetic_holm[dimension]:.12g}",
            }
            for scheme in release.FINAL_CROSSFIT_GROUPING_SCHEMES
            for dimension in release.FINAL_CROSSFIT_DIMENSIONS
        ],
    )
    crossfit_summary_tex = (
        crossfit_manuscript_dir / "table_supp_crossfit_oracle_summary.tex"
    )
    crossfit_summary_tex.write_text(
        "% synthetic cross-fit summary\n\\begin{tabular}{ll}A&B\\\\\\end{tabular}\n",
        encoding="utf-8",
    )
    crossfit_summary_manifest = (
        crossfit_manuscript_dir
        / "table_supp_crossfit_oracle_summary_manifest.json"
    )
    crossfit_summary_manifest.write_text(
        json.dumps(
            {
                "schema": "crossfit_oracle_supplementary_table_v2",
                "rows": len(release.FINAL_CROSSFIT_GROUPING_SCHEMES)
                * len(release.FINAL_CROSSFIT_DIMENSIONS),
                "csv_path": release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
                    "crossfit_manuscript_summary_csv"
                ][0],
                "tex_path": release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
                    "crossfit_manuscript_summary_tex"
                ][0],
                "csv_sha256": _file_binding(crossfit_summary_csv)["sha256"],
                "tex_sha256": _file_binding(crossfit_summary_tex)["sha256"],
                "source_summary_path": (
                    "results/tables/crossfit_oracle_sensitivity_final/"
                    "crossfit_oracle_dimension_summary.csv"
                ),
                "source_summary_sha256": _file_binding(dimension_path)["sha256"],
            }
        ),
        encoding="utf-8",
    )
    crossfit_output_paths = {
        path.name: path
        for path in (
            fold_path,
            endpoint_path,
            dimension_path,
            overlap_path,
            source_hash_path,
            prediction_audit_path,
            run_alignment_path,
            availability_path,
            ensemble_alignment_path,
            validation_scores_path,
            fixed_selection_path,
        )
    }
    crossfit_output_paths.update(
        {
            release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
                "crossfit_manuscript_summary_csv"
            ][0]: crossfit_summary_csv,
            release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
                "crossfit_manuscript_summary_tex"
            ][0]: crossfit_summary_tex,
            release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
                "crossfit_manuscript_summary_manifest"
            ][0]: crossfit_summary_manifest,
        }
    )
    crossfit_gate = {
        "status": "GO",
        "scope": "zero_retraining_crossfit_oracle_sensitivity_final_locked",
        "analysis_contract": {
            "passes": True,
            "expected_models_in_tie_order": [
                *release.TRADITIONAL_MODELS,
                *release.SUBMISSION_NEURAL_MODELS,
            ],
            "expected_seeds": list(release.SEEDS),
            "expected_tasks": len(release.TDC_TASKS),
            "grouping_schemes": list(release.FINAL_CROSSFIT_GROUPING_SCHEMES),
            "primary_grouping_scheme": "bemis_murcko",
            "repeats_per_endpoint_scheme": release.FINAL_CROSSFIT_REPEATS,
            "directions_per_repeat": 2,
            "expected_fold_rows": expected_fold_rows,
            "observed_fold_rows": expected_fold_rows,
            "expected_endpoint_dimension_rows": len(
                release.FINAL_CROSSFIT_GROUPING_SCHEMES
            )
            * endpoint_dimensions,
            "observed_endpoint_dimension_rows": len(
                release.FINAL_CROSSFIT_GROUPING_SCHEMES
            )
            * endpoint_dimensions,
            "observed_dimension_summary_rows": len(
                release.FINAL_CROSSFIT_GROUPING_SCHEMES
            )
            * len(release.FINAL_CROSSFIT_DIMENSIONS),
            "all_rows_usable": True,
            "no_group_leakage": True,
            "all_models_dimensions_share_fold": True,
            "all_repeats_bidirectionally_complete": True,
            "formal_prediction_sources_complete": True,
            "run_sample_alignment_exact": True,
            "ensemble_sample_alignment_exact": True,
            "prediction_test_matches_locked_source": True,
            "canonical_and_scaffold_overlap_zero_all_endpoints": True,
            "exact_sign_flip_holm_complete": True,
        },
        "signed_difference_policy": {"negative_values_never_truncated": True},
        "overlap_audit": {
            "endpoints": len(release.TDC_TASKS),
            "canonical_overlap_total": 0,
            "scaffold_group_overlap_total": 0,
            "nonempty_bemis_murcko_overlap_total": 0,
        },
        "provenance": {
            "selection_file": (
                "results/tables/reliability_regret_final/"
                "validation_selection_by_task.csv"
            ),
            "locked_source_file_count": 2 * len(release.TDC_TASKS),
            "locked_source_hash_manifest_sha256": "d" * 64,
            "output_sha256": {
                name: _file_binding(path)["sha256"]
                for name, path in crossfit_output_paths.items()
            },
        },
    }
    (crossfit_dir / "crossfit_oracle_sensitivity_gate.json").write_text(
        json.dumps(crossfit_gate), encoding="utf-8"
    )

    influence_dir = tables / "crossfit_influence_sensitivity_final"
    influence_dir.mkdir(parents=True, exist_ok=True)

    def influence_seed(base_seed: int, *parts: object) -> int:
        payload = "|".join([str(base_seed), *[str(part) for part in parts]])
        return int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little"
        )

    task_categories = {
        task: str(meta["category"]).lower() for task, meta in locked_task_meta.items()
    }

    def common_influence_row(
        scheme: str,
        dimension: str,
        remaining_count: int,
        omission_kind: str,
        omitted_value: str,
    ) -> dict[str, object]:
        return {
            "grouping_scheme": scheme,
            "scheme_role": "primary" if scheme == "bemis_murcko" else "secondary",
            "dimension": dimension,
            "evidence_tier": release.FINAL_CROSSFIT_EVIDENCE_TIERS[dimension],
            "full_endpoint_count": len(tasks_by_dimension[dimension]),
            "remaining_endpoint_count": remaining_count,
            "full_mean_signed_d": 0.01,
            "full_ci_low": 0.005,
            "full_ci_high": 0.015,
            "full_mean_sign": "positive",
            "reduced_mean_signed_d": 0.01,
            "reduced_ci_low": 0.004,
            "reduced_ci_high": 0.016,
            "reduced_mean_sign": "positive",
            "mean_sign_stable": True,
            "sign_stability": "preserved",
            "ci_low_gt_zero": True,
            "ci_high_lt_zero": False,
            "ci_excludes_zero": True,
            "ci_supports_full_mean_sign": True,
            "bootstrap_unit": "endpoint",
            "bootstrap_reps": release.FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS,
            "bootstrap_seed": influence_seed(
                release.FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
                "endpoint_bootstrap",
                scheme,
                dimension,
                omission_kind,
                omitted_value,
            ),
            "full_bootstrap_seed": influence_seed(
                release.FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
                "endpoint_bootstrap",
                scheme,
                dimension,
            ),
            "analysis_scope": release.FINAL_CROSSFIT_INFLUENCE_ANALYSIS_SCOPE,
        }

    loeo_rows = []
    loco_rows = []
    summary_rows = []
    for scheme in release.FINAL_CROSSFIT_GROUPING_SCHEMES:
        for dimension in release.FINAL_CROSSFIT_DIMENSIONS:
            tasks = tasks_by_dimension[dimension]
            for task in sorted(tasks):
                loeo_rows.append(
                    {
                        **common_influence_row(
                            scheme,
                            dimension,
                            len(tasks) - 1,
                            "leave_one_endpoint_out",
                            task,
                        ),
                        "omitted_endpoint": task,
                        "omitted_endpoint_category": task_categories[task],
                    }
                )
            categories = sorted({task_categories[task] for task in tasks})
            for category in categories:
                omitted_count = sum(
                    task_categories[task] == category for task in tasks
                )
                loco_rows.append(
                    {
                        **common_influence_row(
                            scheme,
                            dimension,
                            len(tasks) - omitted_count,
                            "leave_one_category_out",
                            category,
                        ),
                        "omitted_category": category,
                        "omitted_endpoint_count": omitted_count,
                    }
                )
            summary_rows.append(
                {
                    "grouping_scheme": scheme,
                    "scheme_role": (
                        "primary" if scheme == "bemis_murcko" else "secondary"
                    ),
                    "dimension": dimension,
                    "evidence_tier": release.FINAL_CROSSFIT_EVIDENCE_TIERS[
                        dimension
                    ],
                    "full_endpoint_count": len(tasks),
                    "full_mean_signed_d": 0.01,
                    "full_ci_low": 0.005,
                    "full_ci_high": 0.015,
                    "full_ci_low_gt_zero": True,
                    "loeo_omission_count": len(tasks),
                    "loeo_reduced_mean_min": 0.01,
                    "loeo_reduced_mean_max": 0.01,
                    "loeo_ci_low_min": 0.004,
                    "loeo_ci_low_max": 0.004,
                    "loeo_ci_high_min": 0.016,
                    "loeo_ci_high_max": 0.016,
                    "loeo_mean_sign_stable_count": len(tasks),
                    "loeo_mean_sign_stable_fraction": 1.0,
                    "loeo_ci_low_gt_zero_count": len(tasks),
                    "loeo_ci_low_gt_zero_fraction": 1.0,
                    "loco_omission_count": len(categories),
                    "loco_reduced_mean_min": 0.01,
                    "loco_reduced_mean_max": 0.01,
                    "loco_ci_low_min": 0.004,
                    "loco_ci_low_max": 0.004,
                    "loco_ci_high_min": 0.016,
                    "loco_ci_high_max": 0.016,
                    "loco_mean_sign_stable_count": len(categories),
                    "loco_mean_sign_stable_fraction": 1.0,
                    "loco_ci_low_gt_zero_count": len(categories),
                    "loco_ci_low_gt_zero_fraction": 1.0,
                    "bootstrap_unit": "endpoint",
                    "bootstrap_reps": release.FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS,
                    "reduced_random_seed": release.FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
                    "full_random_seed": release.FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
                    "analysis_scope": release.FINAL_CROSSFIT_INFLUENCE_ANALYSIS_SCOPE,
                    "independent_validation": False,
                }
            )

    influence_paths = {
        "leave_one_endpoint_out.csv": influence_dir / "leave_one_endpoint_out.csv",
        "leave_one_category_out.csv": influence_dir / "leave_one_category_out.csv",
        "influence_dimension_summary.csv": (
            influence_dir / "influence_dimension_summary.csv"
        ),
        "table_supp_crossfit_influence_primary.tex": (
            influence_dir / "table_supp_crossfit_influence_primary.tex"
        ),
    }
    _write_csv(
        influence_paths["leave_one_endpoint_out.csv"],
        list(loeo_rows[0]),
        loeo_rows,
    )
    _write_csv(
        influence_paths["leave_one_category_out.csv"],
        list(loco_rows[0]),
        loco_rows,
    )
    _write_csv(
        influence_paths["influence_dimension_summary.csv"],
        list(summary_rows[0]),
        summary_rows,
    )
    influence_paths["table_supp_crossfit_influence_primary.tex"].write_text(
        "% Post-hoc, zero-refit influence sensitivity\n"
        "% This is not a new primary result or independent validation.\n"
        "ECE (10 bins)\\\\\nECE (5 bins)\\\\\nMorgan support/error rank association\\\\\n",
        encoding="utf-8",
    )
    influence_checks = {
        "source_hash_and_size_unchanged": True,
        "all_input_tasks_map_to_locked_tdc_categories": True,
        "loeo_has_one_row_per_input_endpoint_mean": True,
        "loco_has_one_row_per_present_category": True,
        "summary_has_every_grouping_dimension": True,
        "reduced_subsets_are_nonempty": True,
        "all_reported_estimates_and_intervals_are_finite": True,
        "bootstrap_contract_is_endpoint_5000_full_seed_20260715_reduced_seed_20260714": True,
        "locked_full_dimension_intervals_reproduced": True,
        "primary_bemis_murcko_focus_complete": True,
        "analysis_is_explicitly_post_hoc_and_not_independent_validation": True,
    }
    summary_by_key = {
        (row["grouping_scheme"], row["dimension"]): row for row in summary_rows
    }
    influence_gate = {
        "schema": "crossfit_influence_sensitivity_v1",
        "status": "GO",
        "scope": release.FINAL_CROSSFIT_INFLUENCE_SCOPE,
        "analysis_contract": {
            "model_refits": 0,
            "input_unit": "cross-fit endpoint mean signed D",
            "bootstrap_unit": "endpoint",
            "bootstrap_method": "percentile_95pct",
            "bootstrap_reps": release.FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS,
            "full_reference_random_seed": release.FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
            "reduced_subset_random_seed": release.FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
            "seed_derivation": "sha256_first_4_bytes_little_endian",
            "multiple_testing_adjustment": "not_applicable_descriptive_sensitivity",
            "confirmatory_inference": False,
            "independent_validation": False,
        },
        "checks": influence_checks,
        "counts": {
            "input_endpoint_rows": len(endpoint_rows),
            "grouping_schemes": len(release.FINAL_CROSSFIT_GROUPING_SCHEMES),
            "dimensions": len(release.FINAL_CROSSFIT_DIMENSIONS),
            "grouping_dimension_cells": len(summary_rows),
            "loeo_rows": len(loeo_rows),
            "loco_rows": len(loco_rows),
            "summary_rows": len(summary_rows),
        },
        "reference_reproduction": {
            "reference_row_count": len(summary_rows),
            "observed_row_count": len(summary_rows),
            "keys_match": True,
            "maximum_absolute_error": 0.0,
            "absolute_error_tolerance": 1e-11,
            "passes": True,
        },
        "primary_bemis_murcko_focus": {
            dimension: {
                key: summary_by_key[("bemis_murcko", dimension)][key]
                for key in (
                    "full_mean_signed_d",
                    "full_ci_low",
                    "full_ci_high",
                    "loeo_ci_low_gt_zero_count",
                    "loeo_omission_count",
                    "loco_ci_low_gt_zero_count",
                    "loco_omission_count",
                )
            }
            for dimension in release.FINAL_CROSSFIT_INFLUENCE_FOCUS_DIMENSIONS
        },
        "provenance": {
            "input_path": (
                "results/tables/crossfit_oracle_sensitivity_final/"
                "crossfit_oracle_endpoint_summary.csv"
            ),
            "input_sha256_before": _file_binding(endpoint_path)["sha256"],
            "input_sha256_after": _file_binding(endpoint_path)["sha256"],
            "input_bytes_before": _file_binding(endpoint_path)["bytes"],
            "input_bytes_after": _file_binding(endpoint_path)["bytes"],
            "reference_dimension_summary_path": (
                "results/tables/crossfit_oracle_sensitivity_final/"
                "crossfit_oracle_dimension_summary.csv"
            ),
            "reference_dimension_summary_sha256": _file_binding(dimension_path)[
                "sha256"
            ],
            "analysis_script_path": (
                "experiments/analyze_crossfit_influence_sensitivity.py"
            ),
            "analysis_script_sha256": _file_binding(
                root / "experiments/analyze_crossfit_influence_sensitivity.py"
            )["sha256"],
            "output_sha256": {
                name: _file_binding(path)["sha256"]
                for name, path in influence_paths.items()
            },
            "gate_self_hash": "excluded_to_avoid_recursive_digest",
        },
    }
    (influence_dir / "crossfit_influence_sensitivity_gate.json").write_text(
        json.dumps(influence_gate), encoding="utf-8"
    )

    _write_reviewer_defense_supplements(root, tables)
    _write_selected_model_sensitivities(root, tables)

    reliability_figures = root / "results" / "figures" / "reliability_regret_final"
    trust_figures = root / "results" / "figures" / "trustworthiness_submission_final"
    reliability_figures.mkdir(parents=True, exist_ok=True)
    trust_figures.mkdir(parents=True, exist_ok=True)
    figure_01 = (
        reliability_figures / "fig01_validation_selected_reliability_regret.pdf"
    )
    figure_02 = trust_figures / "fig02_conditional_reliability.pdf"
    figure_03 = trust_figures / "fig03_calibration_gains_failures.pdf"
    for index, path in enumerate((figure_01, figure_02, figure_03), start=1):
        path.write_bytes(f"%PDF-1.4\nsynthetic-final-figure-{index}\n".encode())
    figure_01_binding = _file_binding(figure_01)
    figure_02_binding = _file_binding(figure_02)
    figure_03_binding = _file_binding(figure_03)
    (reliability_figures / "figure_manifest.json").write_text(
        json.dumps(
            {
                "source_result_dir": str(tables.resolve()),
                "external_source": "/private/nonportable/reliability",
                "outputs": {
                    figure_01.name: {
                        "sha256": figure_01_binding["sha256"],
                        "bytes": figure_01_binding["bytes"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (trust_figures / "figure_manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "alignment_dir": str(tables.resolve()),
                    "outside": "/tmp/nonportable/alignment",
                    "windows_outside": "C:" + r"\private\alignment",
                },
                "superseded_artifacts": {
                    "old": str((root / "results/debug/old").resolve())
                },
                "output_sha256": {
                    str((trust_figures / "fig02.png").resolve()): "a" * 64,
                    str((tables / "fig02.png").resolve()): "b" * 64,
                    str(figure_02.resolve()): figure_02_binding["sha256"],
                    str(figure_03.resolve()): figure_03_binding["sha256"],
                    "/private/nonportable/fig02.png": "c" * 64,
                },
                "api_token": "must-not-leak",
            }
        ),
        encoding="utf-8",
    )
    for path in (
        reliability_figures / "figure_source_statistics.csv",
        reliability_figures / "figure_dimension_statistics.csv",
        trust_figures / "fig02_source_data.csv",
        trust_figures / "fig03_source_data.csv",
        trust_figures / "figure_source_statistics.csv",
    ):
        _write_csv(path, ["value"], [{"value": 1}])

    manuscript_dir = tables / "manuscript_final"
    table_path = manuscript_dir / "table_supp_demo.csv"
    _write_csv(table_path, ["value"], [{"value": 1}])
    latex_table_path = manuscript_dir / "table_supp_demo.tex"
    latex_table_path.write_text(
        "\\begin{longtable}{l}\n"
        "value \\\\\n"
        "\\endfirsthead\n"
        "value \\\\\n"
        "\\endhead\n"
        "\\multicolumn{1}{r}{Continued} \\\\\n"
        "\\endfoot\n"
        "\\endlastfoot\n"
        "1 \\\\\n"
        "\\end{longtable}\n",
        encoding="utf-8",
    )
    table_binding = _file_binding(table_path)
    latex_table_binding = _file_binding(latex_table_path)
    _write_csv(
        manuscript_dir / "table_inventory.csv",
        ["filename", "rows", "columns", "sha256"],
        [
            {
                "filename": table_path.name,
                "rows": 1,
                "columns": 1,
                "sha256": table_binding["sha256"],
            },
            {
                "filename": latex_table_path.name,
                "rows": 1,
                "columns": 1,
                "sha256": latex_table_binding["sha256"],
            },
        ],
    )
    _write_csv(manuscript_dir / "source_audit.csv", ["status"], [{"status": "PASS"}])
    (manuscript_dir / "source_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "mode": "final_locked",
                "protocol_scope": "final_locked_five_model",
                "scientific_gate_statuses": {
                    "ablation": "PASS",
                    "cpu_defense": "GO",
                    "cleaned_prediction_exports": "GO",
                    "cleaned_sensitivity": "SENSITIVITY_STABLE",
                },
            }
        ),
        encoding="utf-8",
    )

    paper = root / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    reference_keys = [f"Ref{index:02d}" for index in range(1, 40)]
    (paper / "references.bib").write_text(
        "\n".join(
            f"@article{{{key}, title={{Synthetic {key}}}, year={{2026}}}}"
            for key in reference_keys
        )
        + "\n",
        encoding="utf-8",
    )
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        + "\\cite{" + ",".join(reference_keys) + "}\n"
        + "\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "supplementary.tex").write_text(
        "\\documentclass{article}\n\\begin{document}Final supplement\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "reference_audit_final_20260715.md").write_text(
        "# Final 39-entry reference audit\n\n"
        + "\n".join(f"- `{key}`: verified" for key in reference_keys)
        + "\n\n39/39 cited; missing 0; duplicate 0.\n",
        encoding="utf-8",
    )
    (paper / "final_numeric_source_audit_20260714.md").write_text(
        "# Final numeric audit\nAll locked sources verified.\n", encoding="utf-8"
    )
    (paper / "supplementary_numeric_source_audit_20260714.md").write_text(
        "# Final supplementary numeric audit\nAll locked sources verified.\n",
        encoding="utf-8",
    )
    main_pdf = paper / "build_final" / "main.pdf"
    supplementary_pdf = (
        paper / "build_supplementary_final" / "supplementary.pdf"
    )
    main_pdf.parent.mkdir(parents=True, exist_ok=True)
    supplementary_pdf.parent.mkdir(parents=True, exist_ok=True)
    main_pdf.write_bytes(b"%PDF-1.4 synthetic final main\n")
    supplementary_pdf.write_bytes(b"%PDF-1.4 synthetic final supplement\n")

    receipt_entries = []
    receipt_sources = [
        ("manifest", manifest_path),
        ("run_table", runs_path),
        *(("prediction", path) for path in prediction_paths),
    ]
    for kind, path in receipt_sources:
        binding = _file_binding(path)
        state = {
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "rows": 1,
            "semantic_payload_sha256": "c" * 64,
        }
        receipt_entries.append(
            {
                "kind": kind,
                "path": str(path.resolve()),
                "old": state,
                "new": state,
                "semantic_payload_unchanged": True,
            }
        )
    receipt_path = (
        tables
        / "cleaned_sensitivity_locked_cache_manifest_provenance_upgrade_receipt.json"
    )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "cleaned_sensitivity_manifest_bound_provenance_upgrade",
                "manifest_binding": {"path": str(manifest_path.resolve())},
                "config_binding": {
                    "path": str(config.resolve()),
                    "logical_path": "experiments/config/sensitivity.yaml",
                    "current_config_sha256": _file_binding(config)["sha256"],
                },
                "files": receipt_entries,
                "ignored_appledouble_sidecars": [str((predictions / "._x.csv").resolve())],
                "aggregate_inventory": {
                    "file_count": 122,
                    "csv_semantic_payloads_unchanged": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _build_final(paths: dict[str, Path]) -> dict:
    return release.build_release(
        root=paths["root"],
        output_dir=paths["output"],
        tasks=(TASK,),
        seeds=(SEED,),
        traditional_models=("rf",),
        traditional_metrics_path=paths["metrics"],
        traditional_prediction_dir=paths["trad_predictions"],
        neural_roots=(paths["neural"],),
        chemprop_dir=paths["chemprop"],
        release_policy_path=paths["policy"],
        require_final_supplements=True,
    )


def test_required_final_supplements_fail_closed_when_absent(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    manifest = _build_final(paths)

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert manifest["gate"]["public_prediction_views_complete"] is False
    assert len(manifest["records"]) == 3
    assert manifest["metadata"]["final_supplements"][
        "primary_submission_records_unchanged"
    ]


def test_final_supplements_build_deidentified_prediction_views_and_index_raw_sources(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    (paths["root"] / "results/predictions/cleaned_sensitivity_locked_cache/._junk.csv").write_bytes(
        b"sidecar"
    )

    manifest = _build_final(paths)
    gate = manifest["gate"]
    assert gate["status"] == "GO"
    assert gate["final_supplements_complete"] is True
    assert gate["public_prediction_views_complete"] is True
    assert gate["public_code_release_complete"] is True
    assert gate["public_code_release"]["status"] == "PASS"
    assert gate["public_code_release"]["required_code_paths_present"] is True
    assert gate["public_code_release"]["required_test_paths_present"] is True
    assert gate["public_code_release"]["local_import_closure_complete"] is True
    assert gate["public_code_release"]["public_readme_present"] is True
    assert gate["checks"]["required_public_code_paths_present"] is True
    assert gate["checks"]["required_public_test_paths_present"] is True
    assert gate["checks"]["public_python_local_import_closure_complete"] is True
    assert gate["checks"]["public_bundle_readme_present"] is True
    assert gate["checks"]["appledouble_excluded_from_include_plan"] is True
    assert gate["checks"]["included_public_package_content_safe"] is True
    package_content_audit = gate["public_package_content_audit"]
    assert package_content_audit["status"] == "PASS"
    assert package_content_audit["forbidden_csv_columns"] == {}
    assert package_content_audit["forbidden_json_keys"] == {}
    assert package_content_audit["secret_assignment_matches"] == 0
    assert package_content_audit["absolute_path_matches"] == 0
    crossfit_gate = gate["final_supplements"]["crossfit_oracle_sensitivity"]
    assert crossfit_gate["status"] == "PASS"
    assert crossfit_gate["required_files"] == len(
        release.FINAL_CROSSFIT_OUTPUTS
    ) + len(release.FINAL_CROSSFIT_DERIVED_OUTPUTS)
    assert crossfit_gate["observed_rows"]["crossfit_fold_level"] == 210000
    assert crossfit_gate["fold_level_evidence_directly_releasable"] is True
    assert set(crossfit_gate["raw_nonportable_files"]) == {
        "prediction_input_audit.csv",
        "validation_seed_scores.csv",
    }
    influence_gate = gate["final_supplements"][
        "crossfit_influence_sensitivity"
    ]
    assert influence_gate["status"] == "PASS"
    assert influence_gate["required_files"] == len(
        release.FINAL_CROSSFIT_INFLUENCE_OUTPUTS
    )
    assert influence_gate["observed_rows"] == {
        "crossfit_influence_loeo": 210,
        "crossfit_influence_loco": 60,
        "crossfit_influence_summary": 14,
    }
    assert influence_gate["zero_refit"] is True
    assert influence_gate["post_hoc_not_independent_validation"] is True
    assert influence_gate["directly_releasable"] is True
    reviewer_gate = gate["final_supplements"]["reviewer_defense_sensitivities"]
    assert reviewer_gate["status"] == "PASS"
    assert reviewer_gate["required_outputs"] == len(
        release.FINAL_REVIEWER_DEFENSE_OUTPUTS
    ) == 17
    assert reviewer_gate["observed_rows"] == (
        release.FINAL_REVIEWER_DEFENSE_EXPECTED_ROWS
    )
    assert reviewer_gate["overlap"] == {
        "endpoints": 22,
        "pairs": 231,
        "pairs_with_overlap": 209,
        "connected_components": 1,
    }
    assert reviewer_gate["leave_one_model_out"] == {
        "four_model_pool_recomputed": True,
        "passing_omissions": ["chemprop_dmpnn", "lightgbm", "rf", "xgboost"],
        "held_omission": "gin",
    }
    assert reviewer_gate["conditional_families"] == ["fingerprint", "neural"]
    assert reviewer_gate["zero_refit"] is True
    assert reviewer_gate["post_hoc_not_primary_claim"] is True
    assert reviewer_gate["no_row_level_molecular_output"] is True
    assert reviewer_gate["directly_releasable"] is True
    support_gate = gate["final_supplements"]["selected_support_risk_coverage"]
    assert support_gate["status"] == "PASS"
    assert support_gate["required_outputs"] == len(
        release.FINAL_SELECTED_SUPPORT_OUTPUTS
    )
    assert support_gate["observed_rows"] == release.FINAL_SELECTED_SUPPORT_EXPECTED_ROWS
    assert support_gate["zero_refit"] is True
    assert support_gate["post_hoc_descriptive"] is True
    assert set(support_gate["raw_nonportable_files"]) == {
        "prediction_input_audit.csv",
        "validation_seed_scores_recomputed.csv",
    }
    selected_gate = gate["final_supplements"]["validation_selected_sensitivity"]
    assert selected_gate["status"] == "PASS"
    assert selected_gate["required_outputs"] == len(
        release.FINAL_VALIDATION_SELECTED_OUTPUTS
    )
    assert selected_gate["required_inputs"] == len(
        release.FINAL_VALIDATION_SELECTED_INPUTS
    )
    assert selected_gate["observed_rows"] == (
        release.FINAL_VALIDATION_SELECTED_EXPECTED_ROWS
    )
    assert selected_gate["zero_refit"] is True
    assert selected_gate["post_hoc_not_primary_claim"] is True
    figure_gate = gate["final_supplements"]["final_manuscript_figures"]
    assert figure_gate["status"] == "PASS"
    assert figure_gate["expected_files"] == 3
    assert figure_gate["observed_files"] == 3
    assert figure_gate["manifest_hashes_match"] is True
    assert gate["public_prediction_views"]["formal_views_generated"] == 3
    assert gate["public_prediction_views"]["cleaned_views_generated"] == 120
    assert gate["public_prediction_views"]["total_prediction_views_generated"] == 123
    public_scan = gate["public_prediction_views"]["public_subtree_scan"]
    assert public_scan["status"] == "PASS"
    assert public_scan["absolute_path_matches"] == 0
    assert public_scan["secret_assignment_matches"] == 0
    assert public_scan["raw_structure_payload_key_matches"] == 0
    assert public_scan["utf8_decode_failures"] == 0
    assert len(manifest["records"]) == 3

    public_files = sorted((paths["output"] / "public/predictions").rglob("*.csv"))
    assert len(public_files) == 123
    forbidden = {"smiles", "canonical_smiles", "label", "y_true", "sample_index"}
    for path in public_files:
        with path.open(newline="", encoding="utf-8") as handle:
            columns = set(csv.DictReader(handle).fieldnames or [])
        assert not columns.intersection(forbidden)
        assert "opaque_sample_id" in columns

    with (paths["output"] / "package_plan.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        package_plan = list(csv.DictReader(handle))
    readme = (paths["output"] / "README.md").read_text(encoding="utf-8")
    assert "Do **not** upload this entire `release/` directory" in readme
    assert "only `package_plan.csv` rows whose action is `include`" in readme
    assert "internal publishability" in readme
    raw_predictions = [
        row
        for row in package_plan
        if "prediction" in row["artifact_roles"].split(";")
        or "cleaned_prediction_raw_index" in row["artifact_roles"].split(";")
    ]
    assert len(raw_predictions) == 123
    assert {row["package_action"] for row in raw_predictions} == {
        "exclude_raw_structure_and_labels"
    }
    public_predictions = [
        row
        for row in package_plan
        if "public_prediction_view" in row["artifact_roles"].split(";")
    ]
    assert len(public_predictions) == 123
    assert {row["package_action"] for row in public_predictions} == {"include"}
    included_paths = {
        row["relative_path"]
        for row in package_plan
        if row["package_action"] == "include"
    }
    assert set(release.REQUIRED_PUBLIC_CODE_PATHS).issubset(included_paths)
    assert set(release.REQUIRED_PUBLIC_TEST_PATHS).issubset(included_paths)
    assert "release/public/README.md" in included_paths
    assert "release/public/metadata/run_seeds_and_hashes.csv" in included_paths
    for publication_document in (
        "paper/main.tex",
        "paper/supplementary.tex",
        "paper/references.bib",
        "paper/reference_audit_final_20260715.md",
        "paper/final_numeric_source_audit_20260714.md",
        "paper/supplementary_numeric_source_audit_20260714.md",
        "paper/build_final/main.pdf",
        "paper/build_supplementary_final/supplementary.pdf",
    ):
        assert publication_document not in included_paths
    assert "paper/reference_audit_final_20260714.md" not in {
        row["relative_path"] for row in package_plan
    }
    crossfit_plan = [
        row
        for row in package_plan
        if row["relative_path"].startswith(
            "results/tables/crossfit_oracle_sensitivity_final/"
        )
    ]
    assert len(crossfit_plan) == len(release.FINAL_CROSSFIT_OUTPUTS)
    assert {
        Path(row["relative_path"]).name: row["package_action"]
        for row in crossfit_plan
    } == {
        filename: (
            "exclude_nonportable"
            if filename
            in {"prediction_input_audit.csv", "validation_seed_scores.csv"}
            else "include"
        )
        for filename, _ in release.FINAL_CROSSFIT_OUTPUTS.values()
    }
    plan_by_path = {row["relative_path"]: row for row in package_plan}
    for relative, _ in release.FINAL_MANUSCRIPT_FIGURES.values():
        figure_row = plan_by_path[relative]
        figure_path = paths["root"] / relative
        assert figure_row["package_action"] == "include"
        assert "final_manuscript_figure" in figure_row["artifact_roles"].split(";")
        assert figure_row["sha256"] == hashlib.sha256(
            figure_path.read_bytes()
        ).hexdigest()
    for relative, role in release.FINAL_CROSSFIT_DERIVED_OUTPUTS.values():
        assert plan_by_path[relative]["package_action"] == "include"
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for filename, role in release.FINAL_CROSSFIT_INFLUENCE_OUTPUTS.values():
        relative = (
            "results/tables/crossfit_influence_sensitivity_final/" + filename
        )
        assert plan_by_path[relative]["package_action"] == "include"
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for filename, role in release.FINAL_REVIEWER_DEFENSE_OUTPUTS.values():
        relative = (
            "results/tables/reviewer_defense_sensitivities_final/" + filename
        )
        assert plan_by_path[relative]["package_action"] == "include"
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for logical_name, (filename, role) in release.FINAL_SELECTED_SUPPORT_OUTPUTS.items():
        relative = "results/tables/selected_support_risk_coverage_final/" + filename
        expected_action = (
            "exclude_nonportable"
            if logical_name in release.FINAL_SELECTED_SUPPORT_NONPORTABLE
            else "include"
        )
        assert plan_by_path[relative]["package_action"] == expected_action
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for filename, role in release.FINAL_VALIDATION_SELECTED_OUTPUTS.values():
        relative = "results/tables/validation_selected_sensitivity_final/" + filename
        assert plan_by_path[relative]["package_action"] == "include"
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for relative, role in release.FINAL_VALIDATION_SELECTED_INPUTS.values():
        assert plan_by_path[relative]["package_action"] == "include"
        assert role in plan_by_path[relative]["artifact_roles"].split(";")
    for name in (
        "crossfit_prediction_input_audit.csv",
        "crossfit_validation_seed_scores.csv",
        "selected_support_prediction_input_audit.csv",
        "selected_support_validation_seed_scores_recomputed.csv",
    ):
        path = paths["output"] / "public/metadata" / name
        assert path.is_file()
        assert str(paths["root"].resolve()) not in path.read_text(encoding="utf-8")
    raw_configs = [
        row for row in package_plan if "config" in row["artifact_roles"].split(";")
    ]
    assert raw_configs
    assert {
        row["package_action"]
        for row in raw_configs
        if "software_or_code" not in row["artifact_roles"].split(";")
    } == {"exclude_nonportable"}
    assert {
        row["package_action"]
        for row in raw_configs
        if "software_or_code" in row["artifact_roles"].split(";")
    } == {"include"}
    public_configs = [
        row
        for row in package_plan
        if "public_sanitized_config" in row["artifact_roles"].split(";")
    ]
    assert len(public_configs) == 3
    assert {row["package_action"] for row in public_configs} == {"include"}
    for path in (paths["output"] / "public/configs").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "/tmp/" not in text
        assert "must-not-leak" not in text
        assert '"command"' not in text
    assert not any(Path(row["relative_path"]).name.startswith("._") for row in package_plan)

    public_readme = (paths["output"] / "public/README.md").read_text(
        encoding="utf-8"
    )
    normalized_public_readme = " ".join(public_readme.split())
    assert "public prediction views alone cannot recompute" in public_readme
    assert "obtain the matching TDC ADMET source" in public_readme
    assert "locally reconstruct the internal" in public_readme
    assert "not a direct runner" in public_readme
    assert "release owner's internal package plan" in normalized_public_readme
    assert "is intentionally not distributed" in normalized_public_readme
    with (paths["output"] / "public/metadata/run_seeds_and_hashes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        public_seed_rows = list(csv.DictReader(handle))
    assert len(public_seed_rows) == 3
    assert {int(row["seed"]) for row in public_seed_rows} == {1}
    assert not forbidden.intersection(public_seed_rows[0])

    sanitized_manifest = (
        paths["output"] / "public/metadata/cleaned_sensitivity_manifest.json"
    ).read_text(encoding="utf-8")
    sanitized_receipt = (
        paths["output"]
        / "public/metadata/cleaned_sensitivity_provenance_upgrade_receipt.json"
    ).read_text(encoding="utf-8")
    assert str(paths["root"].resolve()) not in sanitized_manifest
    assert str(paths["root"].resolve()) not in sanitized_receipt
    assert "config_logical_path" in sanitized_manifest
    assert "logical_path" in sanitized_receipt
    reliability_manifest = json.loads(
        (
            paths["output"]
            / "public/metadata/reliability_regret_final_figure_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert reliability_manifest["source_result_dir"] == "results/tables"
    assert "external_source" not in reliability_manifest
    trust_manifest = json.loads(
        (
            paths["output"]
            / "public/metadata/trustworthiness_submission_final_figure_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert trust_manifest["sources"] == {"alignment_dir": "results/tables"}
    assert trust_manifest["superseded_artifacts"] == {
        "old": "results/debug/old"
    }
    assert trust_manifest["output_sha256"] == {
        "results/figures/trustworthiness_submission_final/fig02.png": "a" * 64,
        "results/tables/fig02.png": "b" * 64,
        "results/figures/trustworthiness_submission_final/"
        "fig02_conditional_reliability.pdf": _file_binding(
            paths["root"]
            / "results/figures/trustworthiness_submission_final/"
            "fig02_conditional_reliability.pdf"
        )["sha256"],
        "results/figures/trustworthiness_submission_final/"
        "fig03_calibration_gains_failures.pdf": _file_binding(
            paths["root"]
            / "results/figures/trustworthiness_submission_final/"
            "fig03_calibration_gains_failures.pdf"
        )["sha256"],
    }
    assert "api_token" not in trust_manifest
    logical_public_payloads = {
        path.relative_to(paths["output"] / "public").as_posix(): path.read_bytes()
        for path in (paths["output"] / "public").rglob("*")
        if path.is_file() and not path.name.startswith("._")
    }
    independent_scan, independent_failures = release._audit_public_payload_map(
        logical_public_payloads
    )
    assert not independent_failures
    assert independent_scan["absolute_path_matches"] == 0
    assert independent_scan["secret_assignment_matches"] == 0
    shap_public = (
        paths["output"]
        / "public/metadata/shap_bit_importance_without_structures.csv"
    )
    with shap_public.open(newline="", encoding="utf-8") as handle:
        shap_columns = set(csv.DictReader(handle).fieldnames or [])
    assert not any("smiles" in column.lower() for column in shap_columns)
    raw_shap = next(
        row
        for row in package_plan
        if row["relative_path"].endswith(
            "shap_morgan_bit_substructures_locked_cache.csv"
        )
    )
    assert raw_shap["package_action"] == "exclude_raw_structure_and_labels"
    neural_public = json.loads(
        (
            paths["output"]
            / "public/metadata/final_neural_completion_gate_sanitized.json"
        ).read_text(encoding="utf-8")
    )
    assert neural_public["complete_by_model"] == {
        "gin": 110,
        "chemprop_dmpnn": 110,
    }
    assert "source_smiles" not in json.dumps(neural_public)
    checkpoint_public = json.loads(
        (
            paths["output"]
            / "public/metadata/final_neural_checkpoint_integrity_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint_public["status"] == "PASS"
    assert not any(
        row["relative_path"].endswith("reference_audit.md")
        or row["relative_path"].endswith(
            "reference_authenticity_audit_20260714.md"
        )
        for row in package_plan
    )


def test_included_public_package_audit_rejects_structure_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    table = root / "results" / "unsafe.csv"
    _write_csv(
        table,
        ["task", "example_smiles", "mean_abs_shap"],
        [{"task": "demo", "example_smiles": "CCO", "mean_abs_shap": 1.0}],
    )
    details, failures = release.audit_included_public_package(
        root=root,
        package_plan=[
            {
                "relative_path": "results/unsafe.csv",
                "package_action": "include",
            }
        ],
        required=True,
    )

    assert details["status"] == "HOLD"
    assert details["forbidden_csv_columns"] == {
        "results/unsafe.csv": ["example_smiles"]
    }
    assert any("raw structure/label/index columns" in failure for failure in failures)


def test_included_public_package_audit_does_not_treat_escaped_latex_as_unc(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    source = root / "experiments" / "safe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'TABLE = "\\\\\\\\begin{tabular}{ll}\\\\n"\n'
        'CITE_RE = r"\\\\\\\\cite[A-Za-z*]*\\\\s*\\\\{([^}]+)\\\\}"\n',
        encoding="utf-8",
    )

    details, failures = release.audit_included_public_package(
        root=root,
        package_plan=[
            {
                "relative_path": "experiments/safe.py",
                "package_action": "include",
            }
        ],
        required=True,
    )

    assert not failures
    assert details["status"] == "PASS"
    assert details["absolute_path_matches"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        r'PATH = r"C:' + r"\private\artifact.csv" + '"\n',
        r'PATH = r"' + "\\\\" + r'server\share\artifact.csv"' + "\n",
    ],
)
def test_included_public_package_audit_still_rejects_windows_absolute_paths(
    tmp_path: Path, payload: str
) -> None:
    root = tmp_path / "project"
    source = root / "experiments" / "unsafe.py"
    source.parent.mkdir(parents=True)
    source.write_text(payload, encoding="utf-8")

    details, failures = release.audit_included_public_package(
        root=root,
        package_plan=[
            {
                "relative_path": "experiments/unsafe.py",
                "package_action": "include",
            }
        ],
        required=True,
    )

    assert details["status"] == "HOLD"
    assert details["absolute_path_matches"] == 1
    assert any("absolute path token" in failure for failure in failures)


def test_formal_release_holds_when_required_public_code_is_missing(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    (paths["root"] / "scripts/run_final_posthoc_locked.sh").unlink()

    manifest = _build_final(paths)
    code_gate = manifest["gate"]["public_code_release"]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["public_code_release_complete"] is False
    assert code_gate["required_code_paths_present"] is False
    assert "scripts/run_final_posthoc_locked.sh" in code_gate[
        "missing_required_code_paths_on_disk"
    ]
    assert "scripts/run_final_posthoc_locked.sh" in code_gate[
        "missing_required_code_paths_from_include_plan"
    ]


def test_formal_release_holds_when_required_public_test_is_missing(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    missing = "tests/test_applicability.py"
    (paths["root"] / missing).unlink()

    manifest = _build_final(paths)
    code_gate = manifest["gate"]["public_code_release"]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["public_code_release_complete"] is False
    assert code_gate["required_code_paths_present"] is True
    assert code_gate["required_test_paths_present"] is False
    assert manifest["gate"]["checks"]["required_public_test_paths_present"] is False
    assert missing in code_gate["missing_required_test_paths_on_disk"]
    assert missing in code_gate["missing_required_test_paths_from_include_plan"]
    assert any(
        "required public test paths missing on disk: 1" in failure
        for failure in code_gate["failures"]
    )


def test_formal_release_holds_when_selected_support_gate_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    gate_path = (
        paths["root"]
        / "results/tables/selected_support_risk_coverage_final/"
        "selected_support_risk_coverage_gate.json"
    )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["primary_descriptive_result"]["mean"] = 0.99
    gate_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = _build_final(paths)
    details = manifest["gate"]["final_supplements"][
        "selected_support_risk_coverage"
    ]
    assert manifest["gate"]["status"] == "HOLD"
    assert details["status"] == "HOLD"
    assert any(
        "selected support gate primary result is stale" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_validation_selected_binding_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    manifest_path = (
        paths["root"]
        / "results/tables/validation_selected_sensitivity_final/"
        "validation_selected_sensitivity_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["output_files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = _build_final(paths)
    details = manifest["gate"]["final_supplements"][
        "validation_selected_sensitivity"
    ]
    assert manifest["gate"]["status"] == "HOLD"
    assert details["status"] == "HOLD"
    assert any(
        "validation selected output" in failure and "SHA-256 is stale" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_crossfit_evidence_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    summary = (
        paths["root"]
        / "results/tables/crossfit_oracle_sensitivity_final/"
        "crossfit_oracle_dimension_summary.csv"
    )
    summary.write_text(
        summary.read_text(encoding="utf-8") + "tampered,row\n",
        encoding="utf-8",
    )

    manifest = _build_final(paths)
    crossfit = manifest["gate"]["final_supplements"][
        "crossfit_oracle_sensitivity"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert crossfit["status"] == "HOLD"
    assert any(
        "cross-fit evidence crossfit_oracle_dimension_summary.csv" in failure
        or "cross-fit gate output SHA-256 inventory is stale" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_recomputes_crossfit_sign_flip_instead_of_trusting_hashes(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    root = paths["root"]
    summary = (
        root
        / "results/tables/crossfit_oracle_sensitivity_final/"
        "crossfit_oracle_dimension_summary.csv"
    )
    with summary.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["sign_flip_p_exact"] = "0.5"
    _write_csv(summary, columns, rows)

    table_manifest = (
        root
        / "results/tables/manuscript_final/"
        "table_supp_crossfit_oracle_summary_manifest.json"
    )
    table_payload = json.loads(table_manifest.read_text(encoding="utf-8"))
    table_payload["source_summary_sha256"] = _file_binding(summary)["sha256"]
    table_manifest.write_text(json.dumps(table_payload), encoding="utf-8")

    gate_path = (
        root
        / "results/tables/crossfit_oracle_sensitivity_final/"
        "crossfit_oracle_sensitivity_gate.json"
    )
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    output_hashes = gate_payload["provenance"]["output_sha256"]
    output_hashes["crossfit_oracle_dimension_summary.csv"] = _file_binding(summary)[
        "sha256"
    ]
    manifest_key = release.FINAL_CROSSFIT_DERIVED_OUTPUTS[
        "crossfit_manuscript_summary_manifest"
    ][0]
    output_hashes[manifest_key] = _file_binding(table_manifest)["sha256"]
    gate_path.write_text(json.dumps(gate_payload), encoding="utf-8")

    manifest = _build_final(paths)
    assert manifest["gate"]["status"] == "HOLD"
    assert any(
        "exact sign-flip value disagrees with endpoint summary" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_crossfit_influence_semantics_are_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    loeo = (
        paths["root"]
        / "results/tables/crossfit_influence_sensitivity_final/"
        "leave_one_endpoint_out.csv"
    )
    with loeo.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["reduced_mean_signed_d"] = "0.9"
    _write_csv(loeo, columns, rows)

    manifest = _build_final(paths)
    influence = manifest["gate"]["final_supplements"][
        "crossfit_influence_sensitivity"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert influence["status"] == "HOLD"
    assert influence["directly_releasable"] is False
    assert any(
        "cross-fit influence LOEO line 2 reduced mean is not source-derived"
        in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_reviewer_lomo_is_not_four_model_reselection(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    selection = (
        paths["root"]
        / "results/tables/reviewer_defense_sensitivities_final/"
        "leave_one_model_out_validation_selection.csv"
    )
    with selection.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["n_models"] = "5"
    _write_csv(selection, columns, rows)

    manifest = _build_final(paths)
    reviewer = manifest["gate"]["final_supplements"][
        "reviewer_defense_sensitivities"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert reviewer["status"] == "HOLD"
    assert reviewer["leave_one_model_out"]["four_model_pool_recomputed"] is False
    assert reviewer["directly_releasable"] is False
    assert any(
        "reviewer LOMO selection line 2 is not a true four-model reselection"
        in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_reviewer_overlap_gate_count_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    gate_path = (
        paths["root"]
        / "results/tables/reviewer_defense_sensitivities_final/"
        "cross_endpoint_test_overlap_summary.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["pairs_with_overlap"] = 208
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    manifest = _build_final(paths)
    reviewer = manifest["gate"]["final_supplements"][
        "reviewer_defense_sensitivities"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert reviewer["status"] == "HOLD"
    assert reviewer["directly_releasable"] is False
    assert any(
        "reviewer overlap gate pairs_with_overlap is stale" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_reviewer_family_split_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    gate_path = (
        paths["root"]
        / "results/tables/reviewer_defense_sensitivities_final/"
        "conditional_reliability_family_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["family_policy"]["neural"] = ["gin"]
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    manifest = _build_final(paths)
    reviewer = manifest["gate"]["final_supplements"][
        "reviewer_defense_sensitivities"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert reviewer["status"] == "HOLD"
    assert reviewer["conditional_families"] == ["fingerprint", "neural"]
    assert reviewer["directly_releasable"] is False
    assert any(
        "reviewer conditional family policy neural is stale" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_final_figure_hash_is_stale(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    figure = (
        paths["root"]
        / "results/figures/reliability_regret_final/"
        "fig01_validation_selected_reliability_regret.pdf"
    )
    figure.write_bytes(figure.read_bytes() + b"tampered")

    manifest = _build_final(paths)
    figure_gate = manifest["gate"]["final_supplements"][
        "final_manuscript_figures"
    ]

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert figure_gate["status"] == "HOLD"
    assert figure_gate["manifest_hashes_match"] is False
    assert any(
        "final figure fig01_validation_selected_reliability_regret.pdf hash is stale"
        in failure
        for failure in manifest["gate"]["failures"]
    )


def test_formal_release_holds_when_local_python_import_is_not_allowlisted(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    importer = paths["root"] / "experiments/analyze_calibration.py"
    importer.write_text(
        "from experiments.private_release_helper import VALUE\n",
        encoding="utf-8",
    )
    helper = paths["root"] / "experiments/private_release_helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    manifest = _build_final(paths)
    code_gate = manifest["gate"]["public_code_release"]

    assert manifest["gate"]["status"] == "HOLD"
    assert code_gate["required_code_paths_present"] is True
    assert code_gate["local_import_closure_complete"] is False
    assert any(
        "local dependency not allowlisted: experiments/private_release_helper.py"
        in failure
        for failure in code_gate["local_import_failures"]
    )


def test_formal_release_holds_when_public_readme_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    monkeypatch.setattr(release, "_public_bundle_readme_text", lambda: "")

    manifest = _build_final(paths)
    code_gate = manifest["gate"]["public_code_release"]

    assert manifest["gate"]["status"] == "HOLD"
    assert code_gate["public_readme_present"] is False
    assert any(
        "public bundle README is missing, empty, or not allowlisted" in failure
        for failure in code_gate["failures"]
    )


def test_final_reference_audit_may_mark_only_legacy_audits_superseded(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    audit = paths["root"] / "paper/reference_audit_final_20260715.md"
    audit.write_text(
        "# Final authoritative reference audit\n\n"
        "> **CANONICAL AUDIT.** This is the final 39-entry audit.\n"
        "> `paper/reference_authenticity_audit_20260714.md` and\n"
        "> `paper/reference_audit.md` are **SUPERSEDED** and excluded.\n\n"
        + "\n".join(f"- `Ref{index:02d}`: verified" for index in range(1, 40))
        + "\n",
        encoding="utf-8",
    )

    manifest = _build_final(paths)

    assert manifest["gate"]["status"] == "GO"
    assert not any(
        "final canonical reference audit is marked superseded" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_final_reference_audit_rejects_self_superseded_marker(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    audit = paths["root"] / "paper/reference_audit_final_20260715.md"
    audit.write_text(
        audit.read_text(encoding="utf-8")
        + "\n> **SUPERSEDED.** This final audit is obsolete.\n",
        encoding="utf-8",
    )

    manifest = _build_final(paths)

    assert manifest["gate"]["status"] == "HOLD"
    assert any(
        "final canonical reference audit is marked superseded" in failure
        for failure in manifest["gate"]["failures"]
    )


def test_public_metadata_path_key_logicalization_collision_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    figure_manifest = (
        paths["root"]
        / "results/figures/trustworthiness_submission_final/figure_manifest.json"
    )
    first = paths["root"] / "results/figures/collision.json"
    second = paths["root"] / "results/figures/other/../collision.json"
    figure_manifest.write_text(
        json.dumps(
            {
                "source": str((paths["root"] / "results/tables").resolve()),
                "output_sha256": {
                    str(first): "a" * 64,
                    str(second): "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = _build_final(paths)

    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert any(
        "key collision after path logicalization" in failure
        for failure in manifest["gate"]["final_supplements"]["failures"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"source": "/etc/passwd"}).encode(),
        json.dumps({"source": "C:" + r"\private\artifact.csv"}).encode(),
        json.dumps({"token": "must-not-leak"}).encode(),
        json.dumps({"nested": {"source_smiles": "CCO"}}).encode(),
    ],
)
def test_public_subtree_atomic_commit_rejects_unsanitized_leakage(
    tmp_path: Path, payload: bytes
) -> None:
    root = tmp_path / "project"
    output = root / "release"

    with pytest.raises(ValueError, match="public subtree payload contains"):
        release._commit_public_payloads(
            root=root,
            output_dir=output,
            payloads={"metadata/leak.json": payload},
        )

    assert not (output / "public").exists()


def test_final_supplement_gate_rehashes_predictions_and_rejects_stale_go(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    prediction = (
        paths["root"]
        / "results/predictions/cleaned_sensitivity_locked_cache"
        / "ames_rf_seed1_paired.csv"
    )
    prediction.write_bytes(prediction.read_bytes() + b"\n")

    manifest = _build_final(paths)
    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["final_supplements_complete"] is False
    assert any(
        "stale" in failure or "blank row" in failure
        for failure in manifest["gate"]["final_supplements"]["failures"]
    )


def test_final_supplement_audit_rejects_fractional_seed_without_truncation(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    _write_final_supplements(paths)
    audit_path = (
        paths["root"]
        / "results/tables/cleaned_sensitivity_locked_cache"
        / "cleaned_sensitivity_prediction_audit.csv"
    )
    with audit_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["seed"] = "1.5"
    _write_csv(audit_path, columns, rows)
    gate_path = audit_path.with_name("cleaned_sensitivity_prediction_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["audit_csv"].update(_file_binding(audit_path))
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    manifest = _build_final(paths)
    assert manifest["gate"]["status"] == "HOLD"
    assert any(
        "canonical integer" in failure
        for failure in manifest["gate"]["final_supplements"]["failures"]
    )


def test_public_view_builder_generates_exact_deterministic_550_plus_120_views(
    tmp_path: Path,
) -> None:
    root = tmp_path / "full"
    output = root / "release"
    records = []
    config_path = root / "raw" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "architecture": {"hidden_size": 128},
                "training": {"epochs": 10},
                "software": {"chemprop": "2.2.1"},
                "data": {
                    "smiles_column": "Drug",
                    "prediction_columns": ["prediction"],
                    "model_input_repairs_n": 2,
                    "model_input_representation_policy": "legacy_valence_repair_v1",
                    "source_test_sha256": "a" * 64,
                    "model_input_repairs": [
                        {
                            "split": "test",
                            "sample_index": 1159,
                            "source_smiles": "CC[NH2+]C",
                            "model_input_smiles": "CCNC",
                            "standard_inchikey": "RAW-INCHIKEY-CANARY",
                        }
                    ],
                    "nested_tamper": {
                        "smiles": "CCC",
                        "canonical_smiles": "CCC",
                        "label": 1,
                        "y_true": 1,
                    },
                },
                "command": ["/tmp/env/bin/python", "train.py"],
                "python_path": "/tmp/env/bin/python",
                "api_token": "must-not-leak",
            }
        ),
        encoding="utf-8",
    )
    config_binding = _file_binding(config_path)
    formal_models = (*release.TRADITIONAL_MODELS, *release.SUBMISSION_NEURAL_MODELS)
    for task in release.TDC_TASKS:
        task_type = release.TDC_TASK_META[task]["task_type"]
        source_sha = hashlib.sha256(task.encode()).hexdigest()
        for model in formal_models:
            for seed in release.SEEDS:
                path = root / "raw" / f"{task.lower()}_{model}_seed{seed}.csv"
                if model in release.TRADITIONAL_MODELS:
                    columns = ["split", "y_pred"]
                    rows = [
                        {"split": "valid", "y_pred": 0.4},
                        {"split": "test", "y_pred": 0.6},
                    ]
                else:
                    columns = ["split", "prediction"]
                    rows = [
                        {"split": "valid", "prediction": 0.4},
                        {"split": "test", "prediction": 0.6},
                    ]
                _write_csv(path, columns, rows)
                binding = _file_binding(path)
                records.append(
                    {
                        "model": model,
                        "task": task,
                        "task_type": task_type,
                        "seed": seed,
                        "included_default": True,
                        "run_status": "complete",
                        "prediction_path": path.relative_to(root).as_posix(),
                        "prediction_sha256": binding["sha256"],
                        "prediction_bytes": binding["bytes"],
                        "prediction_rows": 2,
                        "split_sha256": "b" * 64,
                        "source_data_sha256": source_sha,
                        "execution_spec_sha256": "e" * 64,
                        "software_sha256": "f" * 64,
                        "config_path": config_path.relative_to(root).as_posix(),
                        "config_sha256": config_binding["sha256"],
                        "config_bytes": config_binding["bytes"],
                    }
                )

    cleaned_sources = []
    for task in release.FINAL_CLEANED_TASKS:
        for model in release.FINAL_CLEANED_MODELS:
            for seed in release.FINAL_CLEANED_SEEDS:
                path = root / "cleaned" / f"{task.lower()}_{model}_seed{seed}.csv"
                _write_csv(
                    path,
                    ["evaluation_set", "official_prediction", "cleaned_prediction"],
                    [
                        {
                            "evaluation_set": "common_cleaned_test",
                            "official_prediction": 0.4,
                            "cleaned_prediction": 0.6,
                        }
                    ],
                )
                raw = path.read_bytes()
                snapshot = release.FileSnapshot(
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    data=raw,
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
                cleaned_sources.append(
                    release.CleanedPredictionSource(
                        task.lower(), model, seed, 1, snapshot
                    )
                )
    supplements = release.FinalSupplementAudit(
        complete=True,
        failures=(),
        files=(),
        details={},
        cleaned_predictions=tuple(cleaned_sources),
        cleaned_source_data_sha256="d" * 64,
    )

    first = release.build_public_prediction_views(
        root=root, output_dir=output, records=records, supplements=supplements
    )
    second = release.build_public_prediction_views(
        root=root, output_dir=output, records=records, supplements=supplements
    )

    assert first.complete and second.complete
    assert first.details["formal_views_generated"] == 550
    assert first.details["cleaned_views_generated"] == 120
    assert first.details["total_prediction_views_generated"] == 670
    assert first.details["sanitized_config_views_generated"] == 550
    assert first.details["prediction_inventory_sha256"] == second.details[
        "prediction_inventory_sha256"
    ]
    assert len(list((output / "public/predictions").rglob("*.csv"))) == 670
    public_configs = list((output / "public/configs").glob("*.json"))
    assert len(public_configs) == 550
    assert all("/tmp/" not in path.read_text() for path in public_configs)
    assert all("must-not-leak" not in path.read_text() for path in public_configs)
    assert all('"command"' not in path.read_text() for path in public_configs)
    representative_config = json.loads(public_configs[0].read_text(encoding="utf-8"))
    public_data = representative_config["config"]["data"]
    assert public_data["smiles_column"] == "Drug"
    assert public_data["prediction_columns"] == ["prediction"]
    assert public_data["model_input_repairs_n"] == 2
    assert public_data["model_input_representation_policy"] == (
        "legacy_valence_repair_v1"
    )
    assert public_data["source_test_sha256"] == "a" * 64
    assert "model_input_repairs" not in public_data
    observed_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                observed_keys.add(str(key).strip().lower().replace("-", "_"))
                collect_keys(item)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item)

    collect_keys(representative_config)
    assert not observed_keys.intersection(release.PUBLIC_CONFIG_RAW_PAYLOAD_KEYS)
    assert first.details["public_subtree_scan"][
        "raw_structure_payload_key_matches"
    ] == 0
    assert first.details["raw_structure_and_labels_present_in_public_views"] is False
    assert {
        (item.info.path, item.info.sha256, item.info.bytes) for item in first.files
    } == {
        (item.info.path, item.info.sha256, item.info.bytes) for item in second.files
    }


def test_complete_release_is_go_and_writes_relative_hash_index(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    manifest = _build(paths)
    gate = manifest["gate"]
    assert gate["status"] == "GO"
    assert gate["expected_total_runs"] == 3
    assert gate["complete_by_model"] == {"rf": 1, "gin": 1, "chemprop_dmpnn": 1}
    assert gate["checks"]["cross_model_split_hashes_consistent"] is True
    assert gate["checks"]["formal_neural_representation_compatibility_audited"]
    chemprop = next(
        record for record in manifest["records"] if record["model"] == "chemprop_dmpnn"
    )
    assert chemprop["representation_compatibility_valid"] is True
    assert len(chemprop["model_input_split_sha256"]) == 64
    assert len(chemprop["source_split_sha256"]) == 64
    assert all(not Path(record["prediction_path"]).is_absolute() for record in manifest["records"])
    assert all(len(record["split_sha256"]) == 64 for record in manifest["records"])
    for name in [
        "README.md",
        "manifest.csv",
        "manifest.json",
        "seeds.csv",
        "package_plan.csv",
        "gate.json",
        "LICENSE_AND_REDISTRIBUTION_TODO.md",
    ]:
        assert (paths["output"] / name).is_file()


def test_legacy_dmpnn_is_renamed_excluded_and_cannot_replace_chemprop(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path, with_legacy=True)
    manifest = _build(paths)
    legacy = [record for record in manifest["records"] if record["model"] == "legacy_node_mpnn"]
    assert len(legacy) == 1
    assert legacy[0]["included_default"] is False
    assert legacy[0]["run_status"] == "excluded_legacy"
    assert all("exclude_legacy" in legacy[0][f"{role}_package_action"] for role in ["prediction", "metrics", "config", "checkpoint"])
    assert manifest["gate"]["status"] == "GO"

    (paths["chemprop"] / "metrics" / f"{TASK}_chemprop_dmpnn_seed1.json").unlink()
    held = _build(paths)
    assert held["gate"]["status"] == "HOLD"
    assert held["gate"]["complete_by_model"]["chemprop_dmpnn"] == 0
    assert held["gate"]["legacy_exclusion_ok"] is True


def test_nonfinite_prediction_is_hold_not_partial_success(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    prediction = paths["neural"] / "predictions" / f"{TASK}_gin_seed1.csv"
    rows = _held_out_rows()
    rows[0]["prediction"] = "nan"
    _write_csv(prediction, ["split", "smiles", "label", "prediction"], rows)
    manifest = _build(paths)
    assert manifest["gate"]["status"] == "HOLD"
    gin = next(record for record in manifest["records"] if record["model"] == "gin")
    assert gin["run_status"] == "invalid"
    assert any("non-finite prediction" in error for error in gin["errors"])


def test_unresolved_license_policy_blocks_otherwise_complete_release(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    policy = json.loads(paths["policy"].read_text())
    policy["items"]["code_license"]["status"] = "TODO"
    paths["policy"].write_text(json.dumps(policy))
    manifest = _build(paths)
    assert manifest["gate"]["status"] == "HOLD"
    assert manifest["gate"]["checks"]["license_and_redistribution_policy_resolved"] is False
    assert any("code_license" in item for item in manifest["gate"]["failures"])


def test_resolved_not_released_checkpoint_policy_excludes_formal_checkpoints(
    tmp_path: Path,
) -> None:
    for status in ("NOT_APPLICABLE", "APPROVED"):
        paths = _synthetic_project(tmp_path / status.lower())
        policy = json.loads(paths["policy"].read_text())
        checkpoint = policy["items"]["model_checkpoint_release_license"]
        checkpoint["status"] = status
        checkpoint["decision"] = "NOT_RELEASED"
        paths["policy"].write_text(json.dumps(policy))

        manifest = _build(paths)
        assert manifest["gate"]["status"] == "GO"
        checkpoint_metadata = manifest["metadata"]["checkpoint_release_policy"]
        assert checkpoint_metadata == {
            "status": status,
            "decision": "NOT_RELEASED",
            "policy_resolved": True,
            "checkpoints_excluded_by_policy": True,
            "checkpoint_publication_planned": False,
            "package_action": "exclude_by_release_policy",
        }
        formal_neural = [
            record
            for record in manifest["records"]
            if record["model"] in {"gin", "chemprop_dmpnn"}
        ]
        assert {
            record["checkpoint_package_action"] for record in formal_neural
        } == {"exclude_by_release_policy"}

        with (paths["output"] / "package_plan.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            package_plan = list(csv.DictReader(handle))
        checkpoint_rows = [
            row
            for row in package_plan
            if "checkpoint" in row["artifact_roles"].split(";")
            and set(row["models"].split(";")) & {"gin", "chemprop_dmpnn"}
        ]
        assert checkpoint_rows
        assert {row["package_action"] for row in checkpoint_rows} == {
            "exclude_by_release_policy"
        }
        assert "publish_via_lfs_or_object_store" not in {
            row["package_action"] for row in checkpoint_rows
        }
        readme = (paths["output"] / "README.md").read_text(encoding="utf-8")
        assert "Checkpoint publication is explicitly disabled" in readme
        assert "exclude_by_release_policy" in readme


def test_checkpoint_not_released_with_todo_status_remains_hold(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    policy = json.loads(paths["policy"].read_text())
    checkpoint = policy["items"]["model_checkpoint_release_license"]
    checkpoint["status"] = "TODO"
    checkpoint["decision"] = "NOT_RELEASED"
    paths["policy"].write_text(json.dumps(policy))

    manifest = _build(paths)
    assert manifest["gate"]["status"] == "HOLD"
    assert any(
        "model_checkpoint_release_license" in failure
        for failure in manifest["gate"]["failures"]
    )
    metadata = manifest["metadata"]["checkpoint_release_policy"]
    assert metadata["policy_resolved"] is False
    assert metadata["checkpoints_excluded_by_policy"] is False
    assert metadata["package_action"] == "publish_via_lfs_or_object_store"


def test_release_holds_wrong_gin_policy_and_incomplete_chemprop_hash_audit(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    gin_config = paths["root"] / "experiments" / "config" / "gin.yaml"
    gin_config.write_text("data:\n  model_input_representation_policy: wrong\n")
    manifest = _build(paths)
    gin = next(record for record in manifest["records"] if record["model"] == "gin")
    assert manifest["gate"]["status"] == "HOLD"
    assert gin["representation_compatibility_valid"] is False
    assert any("parser compatibility policy mismatch" in error for error in gin["errors"])

    paths = _synthetic_project(tmp_path / "chemprop_global")
    global_config = (
        paths["root"] / "experiments" / "config" / "chemprop_dmpnn.yaml"
    )
    global_config.write_text(
        "data:\n  model_input_representation_policy: wrong\n"
    )
    manifest = _build(paths)
    chemprop = next(
        record for record in manifest["records"] if record["model"] == "chemprop_dmpnn"
    )
    assert manifest["gate"]["status"] == "HOLD"
    assert chemprop["representation_compatibility_valid"] is False
    assert any(
        "global config parser compatibility policy mismatch" in error
        for error in chemprop["errors"]
    )

    paths = _synthetic_project(tmp_path / "source_hash")
    config_path = (
        paths["chemprop"]
        / "configs"
        / f"{TASK}_chemprop_dmpnn_seed{SEED}.json"
    )
    config = json.loads(config_path.read_text())
    config["data"].pop("source_test_sha256")
    config_path.write_text(json.dumps(config))
    manifest = _build(paths)
    chemprop = next(
        record for record in manifest["records"] if record["model"] == "chemprop_dmpnn"
    )
    assert manifest["gate"]["status"] == "HOLD"
    assert chemprop["representation_compatibility_valid"] is False
    assert any("partial compatibility manifest" in error for error in chemprop["errors"])


def test_release_accepts_pre_solubility_no_repair_manifest_without_rewriting(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    config_path = (
        paths["chemprop"]
        / "configs"
        / f"{TASK}_chemprop_dmpnn_seed{SEED}.json"
    )
    config = json.loads(config_path.read_text())
    for field in (
        "model_input_representation_policy",
        "model_input_repairs_n",
        "model_input_repairs",
        "source_train_sha256",
        "source_valid_sha256",
        "source_test_sha256",
    ):
        config["data"].pop(field)
    config_path.write_text(json.dumps(config))

    manifest = _build(paths)
    chemprop = next(
        record for record in manifest["records"] if record["model"] == "chemprop_dmpnn"
    )
    assert manifest["gate"]["status"] == "GO"
    assert chemprop["representation_compatibility_valid"] is True
    assert chemprop["compatibility_manifest_schema"] == (
        "legacy_no_repair_hash_identity"
    )
    assert chemprop["source_split_sha256"] == chemprop["model_input_split_sha256"]


def test_release_validator_locks_exact_solubility_inchi_mapping() -> None:
    data = {
        "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
        "model_input_repairs_n": 2,
        "model_input_repairs": [
            dict(value) for value in release.SOLUBILITY_COMPATIBILITY_REPAIRS
        ],
        "train_sha256": "a" * 64,
        "valid_sha256": "b" * 64,
        "test_sha256": "c" * 64,
        "source_train_sha256": "a" * 64,
        "source_valid_sha256": "b" * 64,
        "source_test_sha256": "f" * 64,
    }
    assert not release.validate_chemprop_compatibility_manifest(
        "Solubility_AqSolDB", data
    )

    data["model_input_repairs"][0]["standard_inchikey"] = "WRONG"
    errors = release.validate_chemprop_compatibility_manifest(
        "Solubility_AqSolDB", data
    )
    assert errors == ["Chemprop Solubility exact compatibility mapping mismatch"]


def test_float32_gin_labels_align_with_float64_without_weakening_identity(
    tmp_path: Path,
) -> None:
    paths = _synthetic_project(tmp_path)
    exact = [123.123456789, 234.234567891, 345.345678912]
    rounded = [float(np.float32(value)) for value in exact]
    _replace_column(
        paths["trad_predictions"] / TASK.lower() / "rf_seed1.csv",
        "y_true",
        exact,
    )
    _replace_column(
        paths["neural"] / "predictions" / f"{TASK}_gin_seed1.csv",
        "label",
        rounded,
    )
    _replace_column(
        paths["chemprop"] / "predictions" / f"{TASK}_chemprop_dmpnn_seed1.csv",
        "label",
        exact,
    )

    manifest = _build(paths)
    gate = manifest["gate"]

    assert gate["status"] == "GO"
    assert gate["checks"]["cross_model_sample_identity_hashes_consistent"]
    assert gate["checks"]["cross_model_labels_aligned_with_tolerance"]
    assert gate["label_alignment"]["max_abs_difference_across_task_seed_groups"] > 0
    assert all(
        record["label_alignment_within_tolerance"]
        for record in manifest["records"]
        if record["included_default"]
    )


def test_real_label_misalignment_is_an_explicit_hard_hold(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    gin_prediction = paths["neural"] / "predictions" / f"{TASK}_gin_seed1.csv"
    _replace_column(gin_prediction, "label", [1.0, 200.0, 3.0])

    manifest = _build(paths)
    gate = manifest["gate"]

    assert gate["status"] == "HOLD"
    assert gate["checks"]["cross_model_sample_identity_hashes_consistent"]
    assert not gate["checks"]["cross_model_labels_aligned_with_tolerance"]
    assert gate["label_alignment"]["inconsistent_task_seed_groups"]
    assert any("label alignment HARD FAIL" in failure for failure in gate["failures"])


def test_smiles_identity_and_duplicate_sample_indices_remain_strict(tmp_path: Path) -> None:
    paths = _synthetic_project(tmp_path)
    gin_prediction = paths["neural"] / "predictions" / f"{TASK}_gin_seed1.csv"
    _replace_column(gin_prediction, "smiles", ["CCO", "WRONG", "CCC"])
    mismatched = _build(paths)
    assert mismatched["gate"]["status"] == "HOLD"
    assert not mismatched["gate"]["checks"][
        "cross_model_sample_identity_hashes_consistent"
    ]

    paths = _synthetic_project(tmp_path / "duplicate")
    traditional = paths["trad_predictions"] / TASK.lower() / "rf_seed1.csv"
    _replace_column(traditional, "sample_index", [0, 0, 0])
    duplicated = _build(paths)
    rf = next(record for record in duplicated["records"] if record["model"] == "rf")
    assert duplicated["gate"]["status"] == "HOLD"
    assert rf["run_status"] == "invalid"
    assert any("duplicate sample_index" in error for error in rf["errors"])
