import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM
from experiments.make_manuscript_tables import (
    CLEANED_PREDICTION_AUDIT_COLUMNS,
    CLEANED_PREDICTION_COLUMNS,
    DEFAULT_FINAL_CONTRACT,
    FINAL_SOURCE_PATHS,
    METRIC_ATOL,
    METRIC_REPLAY_POLICY_ID,
    METRIC_RTOL,
    MODEL_PREDICTION_DTYPES,
    ROOT,
    FinalContract,
    build_final_locked,
)


TASKS = ("AMES", "Caco2_Wang")
MODELS = ("rf", "gin")
SEEDS = (1, 2)
CONTRACT = FinalContract(
    tasks=TASKS,
    models=MODELS,
    seeds=SEEDS,
    conditional_thresholds=(0.5,),
    ensemble_sizes=(2,),
    ablation_bin_counts=(5,),
    calibration_bin_count=2,
    y_random_seed=1,
    y_random_trials=3,
    shap_tasks=("AMES",),
    shap_top_k=3,
    shap_mapping_top_k=2,
    cleaned_tasks=TASKS,
    cleaned_models=("rf",),
    compatibility_occurrences=(),
)


def _path(root: Path, source_id: str) -> Path:
    configured = FINAL_SOURCE_PATHS[source_id]
    path = (
        ROOT / configured.removeprefix("@project/")
        if configured.startswith("@project/")
        else root / configured
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _csv(root: Path, source_id: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(_path(root, source_id), index=False)


def _json(root: Path, source_id: str, payload: dict) -> None:
    _path(root, source_id).write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_replay_policy() -> dict[str, object]:
    policy: dict[str, object] = {
        "policy_id": METRIC_REPLAY_POLICY_ID,
        "application": (
            "cast parsed official_prediction and cleaned_prediction arrays to "
            "the recorded model-output dtype immediately before metric "
            "recomputation"
        ),
        "prediction_dtype_by_model": dict(MODEL_PREDICTION_DTYPES),
        "labels_dtype": "float64",
        "csv_parser_numeric_dtype": "float64",
        "comparison_tolerances_unchanged": {
            "rtol": METRIC_RTOL,
            "atol": METRIC_ATOL,
        },
    }
    return {**policy, "sha256": _stable_json_sha256(policy)}


def _rewrite_prediction_audit_and_rebind(
    root: Path, audit: pd.DataFrame
) -> None:
    audit_path = _path(root, "cleaned_prediction_audit")
    raw = audit.to_csv(
        index=False,
        columns=CLEANED_PREDICTION_AUDIT_COLUMNS,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    audit_path.write_bytes(raw)
    gate_path = _path(root, "cleaned_prediction_gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["audit_csv"].update(
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": len(audit),
            "columns": CLEANED_PREDICTION_AUDIT_COLUMNS,
        }
    )
    gate_path.write_text(json.dumps(gate), encoding="utf-8")


def _sample_identity(smiles: list[str], labels: np.ndarray) -> str:
    raw = pd.DataFrame(
        {"canonical_smiles": smiles, "label": np.asarray(labels, dtype=float)}
    ).to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_raw(root: Path, source_id: str, payload: bytes) -> None:
    _path(root, source_id).write_bytes(payload)


def _task_type(task: str) -> str:
    return "classification" if task == "AMES" else "regression"


def _metric(task: str) -> str:
    return "auroc" if task == "AMES" else "mae"


def _test_score(task: str, model: str) -> float:
    if task == "AMES":
        return 0.75 if model == "rf" else 0.85
    return 0.25 if model == "rf" else 0.15


def _dimensions(task: str) -> tuple[str, ...]:
    base = ("applicability_alignment", "selective_relative_excess_aurc")
    if task == "AMES":
        return base + ("brier", "ece", "selective_relative_excess_aurc_log_loss")
    return base + ("conformal_coverage_error",)


def _dimension_value(dimension: str) -> float:
    return {
        "applicability_alignment": 0.5,
        "selective_relative_excess_aurc": 0.2,
        "brier": 0.1,
        "ece": 0.05,
        "selective_relative_excess_aurc_log_loss": 0.3,
        "conformal_coverage_error": 0.02,
    }[dimension]


def _write_fixture(
    root: Path,
    *,
    metadata_correction: bool = True,
    data_logical_path: str = "data/tdc_official",
) -> None:
    pool = "gin;rf"
    _csv(
        root,
        "traditional_selection",
        [
            {
                "task": task,
                "task_type": _task_type(task),
                "official_metric": _metric(task),
                "model": "rf",
                "n_seeds": 2,
                "valid_official_mean": 0.8,
                "test_official_mean": 0.75,
            }
            for task in TASKS
        ],
    )
    _json(
        root,
        "reliability_gate",
        {
            "status": "HOLD",
            "scope": "final",
            "final_submission_gate": True,
            "coverage": {
                "complete_expected_model_endpoint_matrix": True,
                "expected_tasks": 2,
                "tasks_present": 2,
                "expected_models": list(MODELS),
                "models_present": list(MODELS),
                "missing_expected_models": [],
                "unexpected_models": [],
                "consistent_selection_pool": True,
                "sample_identity_alignment_exact": True,
                "final_seed_set_required": list(SEEDS),
                "final_seed_sets_exact": True,
            },
        },
    )
    _csv(
        root,
        "validation_selection",
        [
            {
                "task": task,
                "task_type": _task_type(task),
                "official_metric": _metric(task),
                "n_models": 2,
                "model_pool": pool,
                "validation_selected_model": "rf",
                "validation_selected_score_oriented": 0.8,
                "validation_score_margin": 0.1,
                "validation_tolerance": 0.01,
                "validation_tie_models": "rf",
                "validation_tie_count": 1,
                "selected_exact_bootstrap_probability": 0.9,
                "selected_tie_bootstrap_probability": 0.95,
                "bootstrap_seed_count": 2,
            }
            for task in TASKS
        ],
    )
    _csv(
        root,
        "model_availability",
        [
            {
                "task": task,
                "model": model,
                "available_seed_count": 2,
                "available_seeds": "1;2",
                "eligible": True,
            }
            for task, model in itertools.product(TASKS, MODELS)
        ],
    )
    reliability_rows = []
    endpoint_rows = []
    for task in TASKS:
        for dimension in _dimensions(task):
            endpoint_rows.append(
                {
                    "task": task,
                    "task_type": _task_type(task),
                    "official_metric": _metric(task),
                    "dimension": dimension,
                    "n_models": 2,
                    "model_pool": pool,
                    "validation_selected_model": "rf",
                    "test_reliability_oracle_model": "gin",
                    "reliability_regret": 0.01,
                    "normalized_regret": 0.1,
                    "practical_tolerance": 0.005,
                }
            )
            for model in MODELS:
                reliability_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "task_type": _task_type(task),
                        "official_metric": _metric(task),
                        "n_seeds": 2,
                        "test_n": 10,
                        "mean_validation_score_oriented": 0.8,
                        "test_official_metric_raw": _test_score(task, model),
                        "test_official_score_oriented": (
                            -_test_score(task, model)
                            if _metric(task) == "mae"
                            else _test_score(task, model)
                        ),
                        "dimension": dimension,
                        "value": _dimension_value(dimension),
                        "higher_is_better": dimension == "applicability_alignment",
                        "risk_metric": "synthetic",
                        "conformal_coverage": 0.88 if task == "Caco2_Wang" else np.nan,
                    }
                )
    _csv(root, "test_reliability", reliability_rows)
    _csv(root, "reliability_endpoint", endpoint_rows)
    dimensions = sorted({value for task in TASKS for value in _dimensions(task)})
    _csv(
        root,
        "reliability_summary",
        [
            {
                "dimension": dimension,
                "n_endpoints": sum(dimension in _dimensions(task) for task in TASKS),
                "mean_regret": 0.01,
                "mean_regret_ci_low": 0.005,
                "mean_regret_ci_high": 0.015,
                "mean_normalized_regret": 0.1,
                "practical_effect_signal": False,
                "tie_robust_effect_signal": False,
                "random_baseline_retention_signal": False,
            }
            for dimension in dimensions
        ],
    )

    profiles = []
    conditional_rows = []
    for task, model in itertools.product(TASKS, MODELS):
        profiles.append(
            {
                "task": task,
                "model": model,
                "task_type": _task_type(task),
                "official_metric": _metric(task),
                "n_seeds": 2,
                "n_samples": 10,
                "performance": (
                    -_test_score(task, model)
                    if _metric(task) == "mae"
                    else _test_score(task, model)
                ),
                "ece": 0.05 if task == "AMES" else np.nan,
                "brier": 0.1 if task == "AMES" else np.nan,
                "coverage_error": 0.02 if task == "Caco2_Wang" else np.nan,
                "applicability_alignment": 0.5,
                "uncertainty_error_correlation": 0.4,
            }
        )
        conditional_rows.append(
            {
                "task": task,
                "model": model,
                "task_type": _task_type(task),
                "official_metric": _metric(task),
                "n_seeds": 2,
                "threshold": 0.5,
                "inside_count": 5,
                "outside_count": 5,
                "eligible": True,
            }
        )
    _csv(root, "conditional_profiles", profiles)
    _csv(root, "conditional_endpoint_model", conditional_rows)
    _csv(
        root,
        "conditional_summary",
        [
            {
                "threshold": 0.5,
                "metric": metric,
                "n_pairs": 4,
                "n_endpoints": 2,
                "mean": 0.1,
                "median": 0.1,
                "mean_ci_low": 0.01,
                "mean_ci_high": 0.2,
                "sign_flip_p": 0.1,
                "holm_p_within_threshold": 0.2,
            }
            for metric in (
                "log_error_ratio_outside_inside",
                "uq_correlation_advantage_inside",
                "selective_gain_advantage_inside",
                "ece_gap_outside_inside",
                "brier_gap_outside_inside",
                "coverage_error_gap_outside_inside",
            )
        ],
    )
    _json(root, "conditional_gate", {"status": "HOLD", "primary_threshold": 0.5})

    calibration_rows = []
    for task, model, seed in itertools.product(TASKS, MODELS, SEEDS):
        if task == "AMES":
            for method in ("isotonic", "platt"):
                calibration_rows.append(
                    {
                        "task": task,
                        "task_type": "classification",
                        "official_metric": "auroc",
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "test_brier_before": 0.1,
                        "test_brier_after": 0.09,
                        "test_ece_before": 0.05,
                        "test_ece_after": 0.04,
                        "test_mae": np.nan,
                        "test_rmse": np.nan,
                        "test_spearman": np.nan,
                        "coverage": np.nan,
                        "mean_interval_width": np.nan,
                        "error": "",
                    }
                )
        else:
            calibration_rows.append(
                {
                    "task": task,
                    "task_type": "regression",
                    "official_metric": "mae",
                    "model": model,
                    "seed": seed,
                    "method": "conformal_90",
                    "test_brier_before": np.nan,
                    "test_brier_after": np.nan,
                    "test_ece_before": np.nan,
                    "test_ece_after": np.nan,
                    "test_mae": 0.2,
                    "test_rmse": 0.3,
                    "test_spearman": 0.7,
                    "coverage": 0.88,
                    "mean_interval_width": 1.2,
                    "error": "",
                }
            )
    _csv(root, "calibration_summary", calibration_rows)
    calibration_bins = []
    for model, seed, method, bin_index in itertools.product(
        MODELS, SEEDS, ("uncalibrated", "isotonic", "platt"), range(2)
    ):
        calibration_bins.append(
            {
                "task": "AMES",
                "model": model,
                "seed": seed,
                "method": method,
                "bin_left": bin_index / 2,
                "bin_right": (bin_index + 1) / 2,
                "count": 5,
                "empirical_positive_rate": 0.2 + 0.5 * bin_index,
                "mean_confidence": 0.25 + 0.5 * bin_index,
            }
        )
    _csv(root, "calibration_bins", calibration_bins)

    expected_runs = len(TASKS) * len(MODELS) * len(SEEDS)
    calibration_ablation_rows = []
    for model, seed, method, binning in itertools.product(
        MODELS,
        SEEDS,
        ("uncalibrated", "isotonic", "platt"),
        ("equal_width", "equal_mass_tie_preserving"),
    ):
        calibration_ablation_rows.append(
            {
                "task": "AMES",
                "model": model,
                "seed": seed,
                "method": method,
                "binning": binning,
                "n_bins_requested": 5,
                "n_bins_actual": 5,
                "ece": 0.05,
                "brier": 0.1,
                "nll": 0.3,
            }
        )
    ensemble_rows = [
        {
            "task": task,
            "task_type": _task_type(task),
            "official_metric": _metric(task),
            "model": model,
            "ensemble_size": 2,
            "seed_subset": "1;2",
            "test_n": 10,
            "test_official_metric": 0.8,
            "relative_excess_aurc": 0.2,
        }
        for task, model in itertools.product(TASKS, MODELS)
    ]
    _csv(root, "ablation_ensemble_rows", ensemble_rows)
    _csv(
        root,
        "ablation_ensemble_summary",
        [
            {
                "task": row["task"],
                "task_type": row["task_type"],
                "official_metric": row["official_metric"],
                "model": row["model"],
                "ensemble_size": 2,
                "n_seed_subsets": 1,
                "test_official_metric_mean": 0.8,
                "relative_excess_aurc_mean": 0.2,
            }
            for row in ensemble_rows
        ],
    )
    _csv(root, "ablation_calibration_bins", calibration_ablation_rows)
    _csv(
        root,
        "ablation_alignment",
        [
            {
                "task": task,
                "model": model,
                "seed": seed,
                "valid_matches_task_seed_reference": True,
                "test_matches_task_seed_reference": True,
                "task_seed_models_aligned": True,
            }
            for task, model, seed in itertools.product(TASKS, MODELS, SEEDS)
        ],
    )
    _csv(
        root,
        "ablation_input_audit",
        [
            {"path": f"predictions/{task}_{model}_{seed}.csv", "included": True, "reason": "complete"}
            for task, model, seed in itertools.product(TASKS, MODELS, SEEDS)
        ],
    )
    _json(
        root,
        "ablation_gate",
        {
            "schema": "posthoc_ablation_v1",
            "status": "PASS",
            "zero_model_fits": True,
            "all_task_seed_model_samples_aligned": True,
            "expected_runs": expected_runs,
            "observed_runs": expected_runs,
            "ensemble_sizes": [2],
            "seed_subsets_per_task_model": 1,
            "ensemble_rows": 4,
            "bin_counts": [5],
            "binning_schemes": ["equal_width", "equal_mass_tie_preserving"],
            "calibration_methods": ["uncalibrated", "isotonic", "platt"],
            "calibration_rows": len(calibration_ablation_rows),
        },
    )

    y_summary = []
    y_scores = []
    random_scores = (0.4, 0.5, 0.6)
    for task in TASKS:
        real = 0.8 if task == "AMES" else 0.2
        y_summary.append(
            {
                "task": task,
                "task_type": _task_type(task),
                "model": "rf",
                "seed": 1,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "metric": _metric(task),
                "real_score": real,
                "random_mean": 0.5,
                "random_std": 0.1,
                "oriented_margin": 0.3,
                "z_like_margin": 3.0,
                "n_trials": 3,
                "train_n": 20,
                "test_n": 10,
            }
        )
        for trial, score in enumerate(random_scores, start=1):
            y_scores.append(
                {
                    "task": task,
                    "task_type": _task_type(task),
                    "model": "rf",
                    "seed": 1,
                    "split_source": "locked_cache",
                    "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                    "metric": _metric(task),
                    "trial": trial,
                    "random_score": score,
                }
            )
    _csv(root, "y_random_summary", y_summary)
    _csv(root, "y_random_scores", y_scores)
    _json(
        root,
        "cpu_defense_gate",
        {
            "schema_version": 1,
            "status": "GO",
            "artifact_gate": "PASS",
            "negative_control_signal": "PASS",
            "failures": [],
            "y_randomization": {"positive_margin_count": 2, "nonpositive_tasks": []},
        },
    )

    _csv(
        root,
        "shap_selected",
        [
            {
                "task": "AMES",
                "model": "rf",
                "seed": 1,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "rank": rank,
                "feature": f"morgan_bit_{rank}",
                "mean_abs_shap": 1.0 / rank,
                "mean_feature_value": 0.2,
                "train_n": 20,
                "test_n": 10,
                "sample_size_actual": 10,
            }
            for rank in range(1, 4)
        ],
    )
    _csv(
        root,
        "shap_mapping",
        [
            {
                "task": "AMES",
                "model": "rf",
                "seed": 1,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "rank": rank,
                "feature": f"morgan_bit_{rank}",
                "bit_id": rank,
                "mean_abs_shap": 1.0 / rank,
                "example_smiles": "CC",
                "bit_atom_index": 0,
                "bit_radius": 1,
                "substructure_smiles": "C",
            }
            for rank in range(1, 3)
        ],
    )

    fixture_config_path = _path(root, "cleaned_config")
    fixture_config_sha = _sha256(fixture_config_path)
    execution_config_sha = "1" * 64 if metadata_correction else fixture_config_sha
    config_metadata_corrections = (
        [
            {
                "field": "audit.scaffold_overlap",
                "old_value": True,
                "new_value": False,
                "execution_config_sha256": execution_config_sha,
                "corrected_config_sha256": fixture_config_sha,
                "computational_effect": False,
                "reason": "fixture mirrors the reviewed metadata-only correction",
            }
        ]
        if metadata_correction
        else []
    )
    _json(
        root,
        "cleaned_manifest",
        {
            "schema_version": 2,
            "protocol": "paired-common-test-v1",
            "cpu_only": True,
            "split_source": "locked_cache",
            "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
            "parser_compatibility_repairs_count_as_cleaning_deletions": False,
            "tasks": list(TASKS),
            "models": ["rf"],
            "seeds": list(SEEDS),
            "compatibility_source_occurrences": {task: 0 for task in TASKS},
            "config_logical_path": "experiments/config/sensitivity.yaml",
            "config_sha256": fixture_config_sha,
            "execution_config_sha256": execution_config_sha,
            "config_metadata_corrections": config_metadata_corrections,
        },
    )
    cleaned_runs = []
    for task, seed in itertools.product(TASKS, SEEDS):
        metric = _metric(task)
        official = 1.0 if task == "AMES" else 0.2
        cleaned = 1.0 if task == "AMES" else 0.19
        raw_delta = cleaned - official
        oriented_delta = -raw_delta if metric == "mae" else raw_delta
        relative_delta = oriented_delta / abs(official)
        cleaned_runs.append(
            {
                "task": task,
                "task_type": _task_type(task),
                "official_metric": metric,
                "model": "rf",
                "seed": seed,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "radius": 2,
                "n_bits": 2048,
                "threads_per_model": 1,
                "official_train_n": 20,
                "cleaned_train_n": 20,
                "common_valid_n": 4,
                "common_test_n": 10,
                "official_fit_seconds": 1.0,
                "cleaned_fit_seconds": 1.0,
                "primary_official_common_test": official,
                "primary_cleaned_common_test": cleaned,
                "primary_raw_delta_cleaned_minus_official": raw_delta,
                "primary_oriented_delta": oriented_delta,
                "primary_relative_oriented_delta": relative_delta,
                f"official_common_valid_{metric}": official,
                f"cleaned_common_valid_{metric}": cleaned,
                f"official_common_test_{metric}": official,
                f"cleaned_common_test_{metric}": cleaned,
            }
        )
    _csv(root, "cleaned_runs", cleaned_runs)
    _csv(
        root,
        "cleaned_task",
        [
            {
                "task": task,
                "official_metric": _metric(task),
                "official_selected_model": "rf",
                "cleaned_best_model": "rf",
                "winner_changed_after_cleaning": False,
                "n_models": 1,
                "task_material_change": False,
            }
            for task in TASKS
        ],
    )
    _csv(
        root,
        "cleaned_task_model",
        [
            {
                "task": task,
                "model": "rf",
                "official_metric": _metric(task),
                "n_seeds": 2,
                "primary_official_mean": 1.0 if task == "AMES" else 0.2,
                "primary_cleaned_mean": 1.0 if task == "AMES" else 0.19,
                "oriented_delta_mean": 0.0 if task == "AMES" else 0.01,
                "material_change": False,
            }
            for task in TASKS
        ],
    )
    _csv(
        root,
        "cleaned_audit",
        [
            {
                "task": task,
                "seed": seed,
                "split": split,
                "overlap_after_train_test": 0,
                "overlap_after_train_valid": 0,
                "overlap_after_valid_test": 0,
            }
            for task, seed, split in itertools.product(TASKS, SEEDS, ("train", "valid", "test"))
        ],
    )
    manifest_checks = {
        "schema_v2": True,
        "paired_protocol": True,
        "split_source": True,
        "split_algorithm": True,
        "cpu_only": True,
        "task_contract": True,
        "model_contract": True,
        "seed_contract": True,
        "parser_repairs_not_cleaning_deletions": True,
        "compatibility_mapping_hashed": True,
        "source_files_hashed": True,
        "config_sha256_recomputed": True,
        "execution_config_hash_preserved": True,
        "noncomputational_config_correction_transparent": True,
    }
    raw_audit = b"".join(
        (json.dumps({"task": task, "seed": seed}) + "\n").encode("utf-8")
        for task, seed in itertools.product(TASKS, SEEDS)
    )
    _write_raw(root, "cleaned_raw_audit", raw_audit)
    config_path = _path(root, "cleaned_config")
    manifest_path = _path(root, "cleaned_manifest")
    runs_path = _path(root, "cleaned_runs")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_gate = {
        "split_source": "locked_cache",
        "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "config_sha256": _sha256(config_path),
        "execution_config_sha256": manifest_payload["execution_config_sha256"],
        "config_metadata_corrections": manifest_payload[
            "config_metadata_corrections"
        ],
        "config_logical_path": "experiments/config/sensitivity.yaml",
        "data_logical_path": data_logical_path,
        "checks": manifest_checks,
    }
    audit_gate = {
        "payloads": 4,
        "expected_payloads": 4,
        "flattened_rows": 12,
        "exact_task_seed_matrix": True,
        "schema_v2": True,
        "post_precedence_overlap_zero": True,
        "parser_repair_counts_match_sources": True,
        "split_output_counts_match_runs": True,
        "audit_sha256": _sha256(_path(root, "cleaned_raw_audit")),
        "audit_bytes": _path(root, "cleaned_raw_audit").stat().st_size,
    }
    cleaned_output_artifacts = {
        name: {
            "serialization": "utf8-pandas-csv-lf-float17g-v1",
            "sha256": _sha256(_path(root, source_id)),
            "bytes": _path(root, source_id).stat().st_size,
            "rows": len(pd.read_csv(_path(root, source_id))),
        }
        for name, source_id in {
            "task_model": "cleaned_task_model",
            "task": "cleaned_task",
            "audit": "cleaned_audit",
        }.items()
    }
    _json(
        root,
        "cleaned_gate",
        {
            "status": "SENSITIVITY_STABLE",
            "complete_expected_matrix": True,
            "expected_rows": 4,
            "observed_rows": 4,
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "n_tasks": 2,
            "split_source": "locked_cache",
            "split_source_gate": True,
            "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
            "manifest_provenance_gate": manifest_gate,
            "audit_provenance_gate": audit_gate,
            "output_artifacts": cleaned_output_artifacts,
            "run_integrity_gate": {
                "finite_core_values": True,
                "strict_integer_fields": True,
                "fingerprint_contract_matches": True,
                "single_thread_cpu_contract_matches": True,
                "task_metadata_matches_tdc": True,
                "primary_metric_columns_match": True,
                "paired_delta_algebra_matches": True,
                "sample_counts_consistent_across_models": True,
                "validated_rows": 4,
            },
            "input_artifacts": {
                "runs": {
                    "sha256": _sha256(runs_path),
                    "bytes": runs_path.stat().st_size,
                    "rows": 4,
                },
                "audit": {
                    "sha256": _sha256(_path(root, "cleaned_raw_audit")),
                    "bytes": _path(root, "cleaned_raw_audit").stat().st_size,
                    "payloads": 4,
                },
                "manifest": {
                    "sha256": _sha256(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                "config": {
                    "logical_path": "experiments/config/sensitivity.yaml",
                    "sha256": _sha256(config_path),
                    "bytes": config_path.stat().st_size,
                },
            },
        },
    )
    predictions_dir = root.parent / "predictions" / "cleaned_sensitivity_locked_cache"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_audit = []
    inventory_records = []
    for task, seed in itertools.product(TASKS, SEEDS):
        task_slug = task.lower()
        filename = f"{task_slug}_rf_seed{seed}_paired.csv"
        prediction_path = predictions_dir / filename
        if task == "AMES":
            labels = np.asarray([0] * 5 + [1] * 5, dtype=float)
            official_predictions = np.linspace(0.05, 0.95, 10)
            cleaned_predictions = official_predictions.copy()
            recomputed_official = 1.0
            recomputed_cleaned = 1.0
        else:
            labels = np.arange(10, dtype=float)
            official_predictions = labels + 0.2
            cleaned_predictions = labels + 0.19
            recomputed_official = 0.2
            recomputed_cleaned = 0.19
        smiles = [f"{task_slug}_molecule_{index}" for index in range(10)]
        prediction = pd.DataFrame(
            {
                "task": task_slug,
                "model": "rf",
                "seed": seed,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "evaluation_set": "common_cleaned_test",
                "canonical_smiles": smiles,
                "label": labels,
                "official_prediction": official_predictions,
                "cleaned_prediction": cleaned_predictions,
            },
            columns=CLEANED_PREDICTION_COLUMNS,
        )
        prediction.to_csv(prediction_path, index=False)
        prediction_sha = _sha256(prediction_path)
        prediction_bytes = prediction_path.stat().st_size
        identity = _sample_identity(smiles, labels)
        prediction_audit.append(
            {
                "task": task_slug,
                "task_type": _task_type(task),
                "official_metric": _metric(task),
                "model": "rf",
                "metric_replay_policy_id": METRIC_REPLAY_POLICY_ID,
                "metric_prediction_dtype": MODEL_PREDICTION_DTYPES["rf"],
                "seed": seed,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction_sha,
                "prediction_bytes": prediction_bytes,
                "rows": 10,
                "row_count_matches_run": True,
                "finite_labels_and_predictions": True,
                "classification_probability_range_ok": True,
                "canonical_smiles_complete": True,
                "canonical_smiles_unique": True,
                "source_common_test_exact": True,
                "official_metric_matches_run": True,
                "cleaned_metric_matches_run": True,
                "paired_delta_matches_run": True,
                "recomputed_official_common_test": recomputed_official,
                "recomputed_cleaned_common_test": recomputed_cleaned,
                "sample_label_identity_sha256": identity,
                "source_sample_label_identity_sha256": identity,
            }
        )
        inventory_records.append(
            {
                "filename": filename,
                "task": task_slug,
                "model": "rf",
                "seed": seed,
                "sha256": prediction_sha,
                "bytes": prediction_bytes,
                "rows": 10,
            }
        )
    prediction_audit_frame = pd.DataFrame(
        prediction_audit, columns=CLEANED_PREDICTION_AUDIT_COLUMNS
    )
    prediction_audit_raw = prediction_audit_frame.to_csv(
        index=False,
        columns=CLEANED_PREDICTION_AUDIT_COLUMNS,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    _write_raw(root, "cleaned_prediction_audit", prediction_audit_raw)
    inventory_records = sorted(inventory_records, key=lambda row: row["filename"])
    inventory_sha = _stable_json_sha256(inventory_records)
    inventory_entries = {
        row["filename"]: {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "rows": row["rows"],
        }
        for row in inventory_records
    }
    prediction_inventory = {
        "algorithm": "sha256-stable-json-inventory-v1",
        "sha256": inventory_sha,
        "files": 4,
        "bytes": sum(int(row["bytes"]) for row in inventory_records),
        "rows": 40,
        "entries": inventory_entries,
    }
    prediction_audit_path = _path(root, "cleaned_prediction_audit")
    _json(
        root,
        "cleaned_prediction_gate",
        {
            "schema_version": 2,
            "status": "GO",
            "complete_expected_matrix": True,
            "expected_prediction_files": 4,
            "observed_prediction_files": 4,
            "audited_prediction_files": 4,
            "sample_label_identity_aligned_within_task": True,
            "source_common_test_reconstructed_exactly": True,
            "prediction_metrics_recomputed_match_runs": True,
            "unique_sample_label_hashes_by_task": {
                task.lower(): 1 for task in TASKS
            },
            "prediction_inventory": prediction_inventory,
            "prediction_inventory_sha256": inventory_sha,
            "metric_replay_policy": _metric_replay_policy(),
            "manifest_provenance_gate": manifest_gate,
            "failures": [],
            "audit_csv": {
                "serialization": "utf8-pandas-csv-lf-float17g-v1",
                "sha256": _sha256(prediction_audit_path),
                "bytes": prediction_audit_path.stat().st_size,
                "rows": 4,
                "columns": CLEANED_PREDICTION_AUDIT_COLUMNS,
            },
            "inputs": {
                "runs": {
                    "sha256": _sha256(runs_path),
                    "bytes": runs_path.stat().st_size,
                },
                "manifest": {
                    "sha256": _sha256(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                "config": {
                    "sha256": _sha256(config_path),
                    "bytes": config_path.stat().st_size,
                },
                "predictions": prediction_inventory,
            },
        },
    )
    _json(
        root,
        "cpu_defense_gate",
        {
            "schema_version": 1,
            "status": "GO",
            "artifact_gate": "PASS",
            "negative_control_signal": "PASS",
            "coverage": {
                "y_summary_rows": 2,
                "y_score_rows": 6,
                "shap_rows": 3,
                "shap_mapping_rows": 2,
                "mapped_substructures": 2,
            },
            "y_randomization": {"positive_margin_count": 2, "nonpositive_tasks": []},
            "failures": [],
            "inputs": {
                "y_randomization_summary": {
                    "sha256": _sha256(_path(root, "y_random_summary"))
                },
                "y_randomization_scores": {
                    "sha256": _sha256(_path(root, "y_random_scores"))
                },
                "shap": {"sha256": _sha256(_path(root, "shap_selected"))},
                "shap_mapping": {"sha256": _sha256(_path(root, "shap_mapping"))},
                "traditional_selection": {
                    "sha256": _sha256(_path(root, "traditional_selection"))
                },
            },
        },
    )


def test_default_contract_is_the_locked_submission_matrix() -> None:
    assert len(DEFAULT_FINAL_CONTRACT.tasks) == 22
    assert DEFAULT_FINAL_CONTRACT.models == (
        "rf",
        "xgboost",
        "lightgbm",
        "gin",
        "chemprop_dmpnn",
    )
    assert DEFAULT_FINAL_CONTRACT.seeds == (1, 2, 3, 4, 5)
    assert len(DEFAULT_FINAL_CONTRACT.shap_tasks) == 8
    assert len(DEFAULT_FINAL_CONTRACT.cleaned_tasks) == 8


def test_final_locked_builds_transactional_inventory_and_source_audit(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    inventory = build_final_locked(tables, out, contract=CONTRACT)
    assert len(inventory) == 28
    assert set(inventory["role"]) == {"main", "supplementary"}
    assert (out / "table_inventory.csv").is_file()
    assert (out / "source_audit.csv").is_file()
    assert (out / "source_audit.json").is_file()
    audit = json.loads((out / "source_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["scientific_gate_statuses"]["reliability"] == "HOLD"
    assert audit["scientific_gate_statuses"]["conditional"] == "HOLD"
    assert len(audit["source_files"]) == len(FINAL_SOURCE_PATHS)
    assert all(row["validation_status"] == "PASS" for row in audit["source_files"])
    assert not any("local_prelim" in row["path"] for row in audit["source_files"])
    official_gate = audit["reader_visible_table_gates"][
        "official_test_metrics_all_models"
    ]
    assert official_gate["status"] == "PASS"
    assert official_gate["selection_basis"] == "validation_only"
    assert official_gate["expected_rows"] == 4
    assert set(official_gate["outputs"]) == {"csv", "tex"}
    for output_record in official_gate["outputs"].values():
        output_path = out / output_record["filename"]
        assert output_record["rows"] == 4
        assert output_record["sha256"] == _sha256(output_path)
    official = pd.read_csv(out / "table_supp_official_test_metrics_all_models.csv")
    assert list(official.columns) == [
        "task",
        "category",
        "task_type",
        "official_metric",
        "model",
        "n_seeds",
        "test_n",
        "test_score",
        "validation_selected",
    ]
    assert len(official) == 4
    assert int(official["validation_selected"].sum()) == 2
    # The validation-selected RF is deliberately worse on test than GIN in
    # this fixture; the flag must still come from validation, never test rank.
    assert set(official.loc[official["validation_selected"], "model"]) == {"rf"}
    latex = (out / "table_supp_official_test_metrics_all_models.tex").read_text(
        encoding="utf-8"
    )
    assert "\\begin{longtable}" in latex
    assert "validation\\_selected" in latex


def test_final_locked_accepts_current_config_without_metadata_correction(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(
        tables,
        metadata_correction=False,
        data_logical_path="tdc_official",
    )

    inventory = build_final_locked(tables, out, contract=CONTRACT)

    assert len(inventory) == 28
    audit = json.loads((out / "source_audit.json").read_text())
    assert audit["status"] == "PASS"
    manifest_source = next(
        row for row in audit["source_files"] if row["source_id"] == "cleaned_manifest"
    )
    assert manifest_source["validation_status"] == "PASS"


def test_final_locked_never_falls_back_to_local_preview(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame([{"task": "AMES"}]).to_csv(
        tables / "y_randomization_all_tasks_local_prelim.csv", index=False
    )
    out = tmp_path / "manuscript_final"
    with pytest.raises(FileNotFoundError, match="no preview fallback"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


@pytest.mark.parametrize(
    ("source_id", "column", "value", "message"),
    [
        ("y_random_summary", "split_source", "pytdc", "locked-cache provenance"),
        ("shap_selected", "model", "gin", "locked traditional winner"),
    ],
)
def test_final_locked_rejects_defense_provenance_or_selection_without_replacing_output(
    tmp_path: Path, source_id: str, column: str, value: object, message: str
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    out.mkdir()
    (out / "sentinel.txt").write_text("old", encoding="utf-8")
    path = _path(tables, source_id)
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        build_final_locked(tables, out, contract=CONTRACT)
    assert (out / "sentinel.txt").read_text() == "old"
    assert not (out / "source_audit.json").exists()


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        ("bit_1", "canonical Morgan bit identifier"),
        ("morgan_bit_1_suffix", "canonical Morgan bit identifier"),
        ("morgan_bit_2048", "outside the 0-2047 fingerprint range"),
    ],
)
def test_final_locked_rejects_noncanonical_or_out_of_range_morgan_features(
    tmp_path: Path, feature: str, message: str
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    out.mkdir()
    (out / "sentinel.txt").write_text("old", encoding="utf-8")
    path = _path(tables, "shap_selected")
    frame = pd.read_csv(path)
    frame.loc[0, "feature"] = feature
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        build_final_locked(tables, out, contract=CONTRACT)
    assert (out / "sentinel.txt").read_text() == "old"
    assert not (out / "source_audit.json").exists()


def test_final_locked_rejects_shap_mapping_bit_id_tamper(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    out.mkdir()
    (out / "sentinel.txt").write_text("old", encoding="utf-8")
    path = _path(tables, "shap_mapping")
    frame = pd.read_csv(path)
    frame.loc[0, "bit_id"] = 2047
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="feature/bit mismatch"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert (out / "sentinel.txt").read_text() == "old"
    assert not (out / "source_audit.json").exists()


def test_final_locked_rejects_incomplete_reliability_gate(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    path = _path(tables, "reliability_gate")
    gate = json.loads(path.read_text())
    gate["coverage"]["complete_expected_model_endpoint_matrix"] = False
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage gate is incomplete"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_requires_completed_cleaned_manifest_gate(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    path = _path(tables, "cleaned_gate")
    gate = json.loads(path.read_text())
    gate["manifest_provenance_gate"]["checks"]["source_files_hashed"] = False
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance checks"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_requires_aligned_cleaned_prediction_exports(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    path = _path(tables, "cleaned_prediction_gate")
    gate = json.loads(path.read_text())
    gate["sample_label_identity_aligned_within_task"] = False
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_label_identity_aligned_within_task"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rehashes_each_cleaned_prediction_file(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    prediction_path = (
        tables.parent
        / "predictions"
        / "cleaned_sensitivity_locked_cache"
        / "ames_rf_seed1_paired.csv"
    )
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "official_prediction"] = 0.123456789
    prediction.to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="prediction .* SHA is stale"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_tampered_cleaned_prediction_audit_csv(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    audit_path = _path(tables, "cleaned_prediction_audit")
    audit_path.write_bytes(audit_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="audit CSV SHA is stale"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_fractional_cleaned_seed_even_with_rebound_audit(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    audit_path = _path(tables, "cleaned_prediction_audit")
    audit = pd.read_csv(audit_path)
    audit["seed"] = audit["seed"].astype(float)
    audit.loc[0, "seed"] = 1.5
    _rewrite_prediction_audit_and_rebind(tables, audit)
    with pytest.raises(ValueError, match="must be an integer"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


@pytest.mark.parametrize(
    ("column", "tampered_value", "message"),
    (
        (
            "metric_replay_policy_id",
            "unapproved-replay-policy",
            "metric replay policy ID mismatch",
        ),
        (
            "metric_prediction_dtype",
            "float32",
            "metric prediction dtype mismatch for model rf",
        ),
    ),
)
def test_final_locked_rejects_tampered_metric_replay_audit_contract(
    tmp_path: Path,
    column: str,
    tampered_value: str,
    message: str,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    audit = pd.read_csv(_path(tables, "cleaned_prediction_audit"))
    audit.loc[0, column] = tampered_value
    _rewrite_prediction_audit_and_rebind(tables, audit)
    with pytest.raises(ValueError, match=message):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_stale_metric_replay_policy_hash(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    gate_path = _path(tables, "cleaned_prediction_gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["metric_replay_policy"]["sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="metric replay policy SHA is stale"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_self_consistent_unapproved_dtype_policy(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    gate_path = _path(tables, "cleaned_prediction_gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    policy = gate["metric_replay_policy"]
    policy["prediction_dtype_by_model"]["xgboost"] = "float64"
    payload = {key: value for key, value in policy.items() if key != "sha256"}
    policy["sha256"] = _stable_json_sha256(payload)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="metric replay policy contract mismatch"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_fractional_cpu_defense_count(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    summary_path = _path(tables, "y_random_summary")
    summary = pd.read_csv(summary_path)
    summary["n_trials"] = summary["n_trials"].astype(float)
    summary.loc[0, "n_trials"] = 3.5
    summary.to_csv(summary_path, index=False)
    with pytest.raises(ValueError, match="must be an integer"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_stale_prediction_inventory_entry(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    gate_path = _path(tables, "cleaned_prediction_gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["inputs"]["predictions"]["entries"]["ames_rf_seed1_paired.csv"][
        "sha256"
    ] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="entry .* SHA is stale"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_stale_cleaned_summary_output_binding(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    task_summary_path = _path(tables, "cleaned_task")
    task_summary_path.write_bytes(task_summary_path.read_bytes() + b"\n")
    with pytest.raises(
        ValueError, match=r"output_artifacts\.task SHA/serialization is stale"
    ):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_final_locked_rejects_stale_raw_cleaning_audit_binding(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    raw_audit_path = _path(tables, "cleaned_raw_audit")
    raw_audit_path.write_bytes(raw_audit_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"input_artifacts\.audit SHA is stale"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_official_test_metric_table_rejects_cross_source_score_tamper(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    profile_path = _path(tables, "conditional_profiles")
    profiles = pd.read_csv(profile_path)
    profiles.loc[0, "performance"] = float(profiles.loc[0, "performance"]) + 0.01
    profiles.to_csv(profile_path, index=False)
    with pytest.raises(ValueError, match="does not reproduce conditional profile"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()


def test_official_test_metric_table_rejects_fractional_source_count(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables"
    out = tmp_path / "manuscript_final"
    _write_fixture(tables)
    reliability_path = _path(tables, "test_reliability")
    reliability = pd.read_csv(reliability_path)
    reliability["test_n"] = reliability["test_n"].astype(float)
    reliability.loc[0, "test_n"] = 10.5
    reliability.to_csv(reliability_path, index=False)
    with pytest.raises(ValueError, match="must be an integer"):
        build_final_locked(tables, out, contract=CONTRACT)
    assert not out.exists()
