"""Build manuscript table inventories with explicit protocol separation.

``local_prelim`` preserves the historical preview workflow. ``final_locked`` is
fail-closed: it consumes only fixed final/locked artifacts, validates the whole
study contract before writing anything, and records hashes for every source and
generated table.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    TDC_TASK_META,
    TDC_TASKS,
)


FINAL_MODELS = ("rf", "xgboost", "lightgbm", "gin", "chemprop_dmpnn")
TRADITIONAL_MODELS = ("rf", "xgboost", "lightgbm")
FINAL_SEEDS = (1, 2, 3, 4, 5)
CONDITIONAL_THRESHOLDS = (0.3, 0.5, 0.7)
ENSEMBLE_SIZES = (2, 3, 5)
ABLATION_BIN_COUNTS = (5, 10, 15, 20)
MORGAN_FINGERPRINT_N_BITS = 2048
SHAP_TASKS = (
    "Caco2_Wang",
    "Solubility_AqSolDB",
    "BBB_Martins",
    "CYP2C9_Veith",
    "CYP3A4_Veith",
    "AMES",
    "DILI",
    "hERG",
)
CLEANED_TASKS = (
    "Caco2_Wang",
    "BBB_Martins",
    "PPBR_AZ",
    "Solubility_AqSolDB",
    "CYP2C9_Veith",
    "DILI",
    "AMES",
    "LD50_Zhu",
)

BASE_RELIABILITY_DIMENSIONS = (
    "applicability_alignment",
    "selective_relative_excess_aurc",
)
CLASSIFICATION_RELIABILITY_DIMENSIONS = (
    "brier",
    "ece",
    "selective_relative_excess_aurc_log_loss",
)
REGRESSION_RELIABILITY_DIMENSIONS = ("conformal_coverage_error",)
CONDITIONAL_METRICS = (
    "log_error_ratio_outside_inside",
    "uq_correlation_advantage_inside",
    "selective_gain_advantage_inside",
    "ece_gap_outside_inside",
    "brier_gap_outside_inside",
    "coverage_error_gap_outside_inside",
)


@dataclass(frozen=True)
class FinalContract:
    """Injectable final-study contract; production defaults are locked below."""

    tasks: tuple[str, ...] = tuple(TDC_TASKS)
    models: tuple[str, ...] = FINAL_MODELS
    seeds: tuple[int, ...] = FINAL_SEEDS
    conditional_thresholds: tuple[float, ...] = CONDITIONAL_THRESHOLDS
    ensemble_sizes: tuple[int, ...] = ENSEMBLE_SIZES
    ablation_bin_counts: tuple[int, ...] = ABLATION_BIN_COUNTS
    calibration_bin_count: int = 10
    y_random_seed: int = 1
    y_random_trials: int = 10
    shap_tasks: tuple[str, ...] = SHAP_TASKS
    shap_top_k: int = 20
    shap_mapping_top_k: int = 10
    cleaned_tasks: tuple[str, ...] = CLEANED_TASKS
    cleaned_models: tuple[str, ...] = TRADITIONAL_MODELS
    compatibility_occurrences: tuple[tuple[str, int], ...] = (
        ("Solubility_AqSolDB", 2),
    )


DEFAULT_FINAL_CONTRACT = FinalContract()


FINAL_SOURCE_PATHS: dict[str, str] = {
    "traditional_selection": "traditional_grid_locked_cache_best_by_task.csv",
    "cpu_defense_gate": "locked_cache_cpu_defense_gate.json",
    "reliability_gate": "reliability_regret_final/reliability_regret_gate.json",
    "validation_selection": "reliability_regret_final/validation_selection_by_task.csv",
    "reliability_endpoint": "reliability_regret_final/reliability_regret_by_endpoint.csv",
    "reliability_summary": "reliability_regret_final/reliability_regret_summary.csv",
    "test_reliability": "reliability_regret_final/test_reliability_by_task_model.csv",
    "model_availability": "reliability_regret_final/model_seed_availability.csv",
    "conditional_gate": "conditional_reliability_final/conditional_gate.json",
    "conditional_profiles": "conditional_reliability_final/model_reliability_profiles.csv",
    "conditional_endpoint_model": "conditional_reliability_final/conditional_reliability_by_task_model.csv",
    "conditional_summary": "conditional_reliability_final/conditional_reliability_summary.csv",
    "calibration_summary": "calibration_summary_final.csv",
    "calibration_bins": "calibration_reliability_bins_final.csv",
    "ablation_gate": "posthoc_ablations_final/ablation_gate.json",
    "ablation_ensemble_rows": "posthoc_ablations_final/ensemble_size_by_seed_subset.csv",
    "ablation_ensemble_summary": "posthoc_ablations_final/ensemble_size_summary.csv",
    "ablation_calibration_bins": "posthoc_ablations_final/calibration_binning_sensitivity.csv",
    "ablation_alignment": "posthoc_ablations_final/sample_alignment_audit.csv",
    "ablation_input_audit": "posthoc_ablations_final/input_prediction_audit.csv",
    "y_random_summary": "y_randomization_all_tasks_locked_cache.csv",
    "y_random_scores": "y_randomization_all_tasks_scores_locked_cache.csv",
    "shap_selected": "shap_selected_locked_cache.csv",
    "shap_mapping": "shap_morgan_bit_substructures_locked_cache.csv",
    "cleaned_manifest": "cleaned_sensitivity_locked_cache_manifest.json",
    "cleaned_raw_audit": "cleaned_sensitivity_locked_cache_audit.jsonl",
    "cleaned_config": "@project/experiments/config/sensitivity.yaml",
    "cleaned_runs": "cleaned_sensitivity_locked_cache_runs.csv",
    "cleaned_gate": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_gate.json",
    "cleaned_task": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_by_task.csv",
    "cleaned_task_model": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_by_task_model.csv",
    "cleaned_audit": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_audit.csv",
    "cleaned_prediction_gate": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_prediction_gate.json",
    "cleaned_prediction_audit": "cleaned_sensitivity_locked_cache/cleaned_sensitivity_prediction_audit.csv",
}


CLEANED_PREDICTION_COLUMNS = [
    "task",
    "model",
    "seed",
    "split_source",
    "split_algorithm",
    "evaluation_set",
    "canonical_smiles",
    "label",
    "official_prediction",
    "cleaned_prediction",
]
CLEANED_PREDICTION_AUDIT_COLUMNS = [
    "task",
    "task_type",
    "official_metric",
    "model",
    "metric_replay_policy_id",
    "metric_prediction_dtype",
    "seed",
    "split_source",
    "split_algorithm",
    "prediction_path",
    "prediction_sha256",
    "prediction_bytes",
    "rows",
    "row_count_matches_run",
    "finite_labels_and_predictions",
    "classification_probability_range_ok",
    "canonical_smiles_complete",
    "canonical_smiles_unique",
    "source_common_test_exact",
    "official_metric_matches_run",
    "cleaned_metric_matches_run",
    "paired_delta_matches_run",
    "recomputed_official_common_test",
    "recomputed_cleaned_common_test",
    "sample_label_identity_sha256",
    "source_sample_label_identity_sha256",
]
METRIC_RTOL = 1e-10
METRIC_ATOL = 1e-12
METRIC_REPLAY_POLICY_ID = "model-native-output-dtype-replay-v1"
MODEL_PREDICTION_DTYPES = {
    "rf": "float64",
    "xgboost": "float32",
    "lightgbm": "float64",
}
_INTEGER_TEXT = re.compile(r"[+-]?(?:0|[1-9][0-9]*)")
_MORGAN_BIT_FEATURE = re.compile(r"morgan_bit_(0|[1-9][0-9]*)")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}")


def oriented_metric(metric: str, value: float) -> float:
    return -value if str(metric).lower() == "mae" else value


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _task_type(task: str) -> str:
    lookup = {_norm(key): value["task_type"] for key, value in TDC_TASK_META.items()}
    return lookup[_norm(task)]


def _task_metric(task: str) -> str:
    lookup = {_norm(key): value["metric"] for key, value in TDC_TASK_META.items()}
    return lookup[_norm(task)]


def _task_category(task: str) -> str:
    lookup = {_norm(key): str(value["category"]) for key, value in TDC_TASK_META.items()}
    return lookup[_norm(task)]


def _canonical_task_name(task: str) -> str:
    lookup = {_norm(key): str(key) for key in TDC_TASK_META}
    return lookup[_norm(task)]


def _dimensions_for_task(task: str) -> tuple[str, ...]:
    if _task_type(task) == "classification":
        return BASE_RELIABILITY_DIMENSIONS + CLASSIFICATION_RELIABILITY_DIMENSIONS
    return BASE_RELIABILITY_DIMENSIONS + REGRESSION_RELIABILITY_DIMENSIONS


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _strict_int(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    """Parse an integer without accepting booleans or truncating fractions."""

    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not _INTEGER_TEXT.fullmatch(text):
            raise ValueError(f"{label} must be a canonical integer, got {value!r}")
        result = int(text)
    elif isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} must be an integer, got {value!r}")
        result = int(numeric)
    else:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be nonnegative, got {result}")
    return result


def _strict_int_values(
    values: Iterable[object],
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> list[int]:
    return [
        _strict_int(
            value,
            f"{label}[{index}]",
            positive=positive,
            nonnegative=nonnegative,
        )
        for index, value in enumerate(values)
    ]


def _morgan_bit_id(value: object, label: str) -> int:
    """Parse the canonical feature name emitted by ``run_shap_selected.py``."""

    text = str(value)
    match = _MORGAN_BIT_FEATURE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{label} must be a canonical Morgan bit identifier "
            f"morgan_bit_<0-{MORGAN_FINGERPRINT_N_BITS - 1}>"
        )
    bit_id = int(match.group(1))
    if bit_id >= MORGAN_FINGERPRINT_N_BITS:
        raise ValueError(
            f"{label} Morgan bit identifier is outside the "
            f"0-{MORGAN_FINGERPRINT_N_BITS - 1} fingerprint range"
        )
    return bit_id


def _stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _metric_replay_policy_record() -> dict[str, object]:
    """Return the exact prediction-dtype replay contract used by the audit."""

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


def _sample_label_identity_sha256(smiles: Iterable[str], labels: Iterable[float]) -> str:
    payload = pd.DataFrame(
        {
            "canonical_smiles": [str(value) for value in smiles],
            "label": np.asarray(list(labels), dtype=float),
        }
    ).to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _recompute_official_metric(metric: str, labels: np.ndarray, predictions: np.ndarray) -> float:
    metric_key = _norm(metric)
    if metric_key == "auroc":
        value = roc_auc_score(labels, predictions)
    elif metric_key == "auprc":
        value = average_precision_score(labels, predictions)
    elif metric_key == "mae":
        value = np.mean(np.abs(labels - predictions))
    elif metric_key == "spearman":
        value = spearmanr(labels, predictions).statistic
    else:
        raise ValueError(f"Unsupported official metric {metric!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Recomputed {metric_key} is non-finite")
    return result


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _require_nonempty(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} must not be empty")


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    values: list[bool] = []
    for value in series:
        key: object = value
        if isinstance(value, str):
            key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"{label} contains a non-boolean value: {value!r}")
        values.append(mapping[key])
    return pd.Series(values, index=series.index, dtype=bool)


def _require_all_true(series: pd.Series, label: str) -> None:
    parsed = _boolean_series(series, label)
    if not parsed.all():
        raise ValueError(f"{label} contains false values")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label}.{column} contains missing or non-finite values")


def _key_value(column: str, value: object) -> object:
    if column in {
        "task",
        "model",
        "method",
        "dimension",
        "metric",
        "binning",
        "split_source",
        "split_algorithm",
        "feature",
        "seed_subset",
    }:
        return _norm(value)
    if column in {
        "seed",
        "trial",
        "rank",
        "ensemble_size",
        "n_bins_requested",
    }:
        return _strict_int(value, f"key column {column}")
    if column == "threshold":
        return round(float(value), 10)
    return value


def _frame_keys(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[object, ...]]:
    _require_columns(frame, columns, "key frame")
    return [
        tuple(_key_value(column, value) for column, value in zip(columns, row))
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    ]


def _require_exact_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: Iterable[tuple[object, ...]],
    label: str,
) -> None:
    observed_list = _frame_keys(frame, columns)
    observed = set(observed_list)
    expected_set = {
        tuple(_key_value(column, value) for column, value in zip(columns, row))
        for row in expected
    }
    if len(observed_list) != len(observed):
        raise ValueError(f"{label} contains duplicate keys for {list(columns)}")
    missing = sorted(expected_set - observed, key=str)
    unexpected = sorted(observed - expected_set, key=str)
    if missing or unexpected:
        raise ValueError(
            f"{label} key matrix mismatch: missing={len(missing)} {missing[:5]}, "
            f"unexpected={len(unexpected)} {unexpected[:5]}"
        )


def _parse_semicolon_set(value: object) -> set[str]:
    return {_norm(item) for item in str(value).split(";") if str(item).strip()}


def _parse_seed_set(value: object) -> set[int]:
    return {
        _strict_int(item, "semicolon seed", positive=True)
        for item in str(value).split(";")
        if str(item).strip()
    }


def _validate_contract(contract: FinalContract) -> None:
    named_sequences: Mapping[str, Sequence[object]] = {
        "tasks": contract.tasks,
        "models": contract.models,
        "seeds": contract.seeds,
        "conditional_thresholds": contract.conditional_thresholds,
        "ensemble_sizes": contract.ensemble_sizes,
        "ablation_bin_counts": contract.ablation_bin_counts,
        "shap_tasks": contract.shap_tasks,
        "cleaned_tasks": contract.cleaned_tasks,
        "cleaned_models": contract.cleaned_models,
    }
    for name, values in named_sequences.items():
        normalized = [_norm(value) for value in values]
        if not values or len(normalized) != len(set(normalized)):
            raise ValueError(f"Final contract {name} must be non-empty and unique")
    _strict_int_values(contract.seeds, "contract.seeds", positive=True)
    _strict_int_values(contract.ensemble_sizes, "contract.ensemble_sizes", positive=True)
    _strict_int_values(
        contract.ablation_bin_counts,
        "contract.ablation_bin_counts",
        positive=True,
    )
    _strict_int(contract.calibration_bin_count, "contract.calibration_bin_count", positive=True)
    _strict_int(contract.y_random_seed, "contract.y_random_seed", positive=True)
    _strict_int(contract.y_random_trials, "contract.y_random_trials", positive=True)
    _strict_int(contract.shap_top_k, "contract.shap_top_k", positive=True)
    _strict_int(contract.shap_mapping_top_k, "contract.shap_mapping_top_k", positive=True)
    known_tasks = {_norm(task) for task in TDC_TASKS}
    task_set = {_norm(task) for task in contract.tasks}
    if not task_set <= known_tasks:
        raise ValueError(f"Unknown contract tasks: {sorted(task_set - known_tasks)}")
    if not {_norm(task) for task in contract.shap_tasks} <= task_set:
        raise ValueError("SHAP tasks must be a subset of final tasks")
    if not {_norm(task) for task in contract.cleaned_tasks} <= task_set:
        raise ValueError("Cleaned-sensitivity tasks must be a subset of final tasks")
    if not {_norm(model) for model in contract.cleaned_models} <= {
        _norm(model) for model in contract.models
    }:
        raise ValueError("Cleaned models must be a subset of final models")
    if any(size < 2 or size > len(contract.seeds) for size in contract.ensemble_sizes):
        raise ValueError("Ensemble sizes must be between 2 and the final seed count")
    if contract.calibration_bin_count < 1 or contract.y_random_trials < 2:
        raise ValueError("Calibration bins and y-random trials must be positive")


def _validate_task_metadata(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, ["task", "task_type", "official_metric"], label)
    for row in frame[["task", "task_type", "official_metric"]].drop_duplicates().itertuples(index=False):
        task = _norm(row.task)
        if task not in {_norm(item) for item in TDC_TASKS}:
            raise ValueError(f"{label} contains unknown task {row.task!r}")
        if _norm(row.task_type) != _task_type(task):
            raise ValueError(f"{label} task_type mismatch for {row.task}")
        if _norm(row.official_metric) != _task_metric(task):
            raise ValueError(f"{label} official_metric mismatch for {row.task}")


def _load_final_sources(
    tables_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    def resolve(relative: str) -> Path:
        if relative.startswith("@project/"):
            return ROOT / relative.removeprefix("@project/")
        return tables_dir / relative

    paths = {
        source_id: resolve(relative)
        for source_id, relative in FINAL_SOURCE_PATHS.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "final_locked source set is incomplete; no preview fallback is permitted. "
            f"Missing {len(missing)} files: {missing}"
        )
    for source_id, path in paths.items():
        if "local_prelim" in str(path.resolve()).lower():
            raise ValueError(
                f"final_locked source {source_id} resolves to a local_prelim path: {path.resolve()}"
            )

    frames: dict[str, pd.DataFrame] = {}
    payloads: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for source_id, path in paths.items():
        raw = path.read_bytes()
        if path.suffix == ".json":
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{source_id} must contain a JSON object")
            payloads[source_id] = payload
            rows: object = ""
            columns: object = ""
            source_type = "json"
        elif path.suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(raw))
            frames[source_id] = frame
            rows = len(frame)
            columns = len(frame.columns)
            source_type = "csv"
        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - formal environment has PyYAML
                raise ValueError("PyYAML is required to validate final_locked config") from exc
            payload = yaml.safe_load(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{source_id} must contain a YAML mapping")
            payloads[source_id] = payload
            rows = ""
            columns = ""
            source_type = "yaml"
        else:
            rows = ""
            columns = ""
            source_type = path.suffix.removeprefix(".") or "raw"
        configured_path = FINAL_SOURCE_PATHS[source_id]
        audit_path = (
            configured_path.removeprefix("@project/")
            if configured_path.startswith("@project/")
            else configured_path
        )
        audit.append(
            {
                "source_id": source_id,
                "path": audit_path,
                "source_type": source_type,
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "rows": rows,
                "columns": columns,
                "validation_status": "PENDING",
                "protocol_scope": "final_locked_five_model",
                "notes": "fixed required source; no glob or fallback",
            }
        )
    return frames, payloads, audit


def _validate_traditional_selection(
    frame: pd.DataFrame, contract: FinalContract
) -> dict[str, str]:
    label = "traditional_selection"
    _require_nonempty(frame, label)
    _require_columns(
        frame,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "n_seeds",
            "valid_official_mean",
            "test_official_mean",
        ],
        label,
    )
    _require_exact_keys(frame, ["task"], [(task,) for task in contract.tasks], label)
    _validate_task_metadata(frame, label)
    allowed = {_norm(model) for model in contract.cleaned_models}
    observed = {_norm(model) for model in frame["model"]}
    if not observed <= allowed:
        raise ValueError(f"{label} contains non-traditional winners: {sorted(observed - allowed)}")
    if set(
        _strict_int_values(
            frame["n_seeds"], f"{label}.n_seeds", positive=True
        )
    ) != {len(contract.seeds)}:
        raise ValueError(f"{label} must use exactly {len(contract.seeds)} seeds")
    _require_finite(frame, ["valid_official_mean", "test_official_mean"], label)
    return {
        _norm(row.task): _norm(row.model)
        for row in frame[["task", "model"]].itertuples(index=False)
    }


def _validate_reliability(
    frames: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.tasks}
    models = {_norm(model) for model in contract.models}
    seeds = set(contract.seeds)
    if gate.get("scope") != "final" or gate.get("final_submission_gate") is not True:
        raise ValueError("reliability_gate must have scope=final and final_submission_gate=true")
    if gate.get("status") not in {"GO", "HOLD"}:
        raise ValueError("reliability_gate status must be GO or HOLD")
    coverage = gate.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("reliability_gate.coverage must be an object")
    required_true = (
        "complete_expected_model_endpoint_matrix",
        "consistent_selection_pool",
        "sample_identity_alignment_exact",
        "final_seed_sets_exact",
    )
    if any(coverage.get(key) is not True for key in required_true):
        raise ValueError(f"reliability coverage gate is incomplete: required {required_true}")
    if int(coverage.get("expected_tasks", -1)) != len(tasks) or int(
        coverage.get("tasks_present", -1)
    ) != len(tasks):
        raise ValueError("reliability gate task count does not match the final contract")
    if {_norm(value) for value in coverage.get("expected_models", [])} != models:
        raise ValueError("reliability gate expected_models mismatch")
    if {_norm(value) for value in coverage.get("models_present", [])} != models:
        raise ValueError("reliability gate models_present mismatch")
    if {int(value) for value in coverage.get("final_seed_set_required", [])} != seeds:
        raise ValueError("reliability gate final seed set mismatch")
    if coverage.get("missing_expected_models") or coverage.get("unexpected_models"):
        raise ValueError("reliability gate reports missing or unexpected models")

    selection = frames["validation_selection"]
    label = "validation_selection"
    _require_columns(
        selection,
        [
            "task",
            "task_type",
            "official_metric",
            "n_models",
            "model_pool",
            "validation_selected_model",
            "validation_selected_score_oriented",
            "validation_score_margin",
            "validation_tolerance",
            "validation_tie_models",
            "validation_tie_count",
            "selected_exact_bootstrap_probability",
            "selected_tie_bootstrap_probability",
            "bootstrap_seed_count",
        ],
        label,
    )
    _require_exact_keys(selection, ["task"], [(task,) for task in tasks], label)
    _validate_task_metadata(selection, label)
    for row in selection.itertuples(index=False):
        pool = _parse_semicolon_set(row.model_pool)
        ties = _parse_semicolon_set(row.validation_tie_models)
        selected = _norm(row.validation_selected_model)
        if pool != models or selected not in pool or not ties <= pool or selected not in ties:
            raise ValueError(f"{label} has an invalid model pool/tie set for {row.task}")
        if int(row.n_models) != len(models) or int(row.bootstrap_seed_count) != len(seeds):
            raise ValueError(f"{label} model/seed count mismatch for {row.task}")
        if int(row.validation_tie_count) != len(ties):
            raise ValueError(f"{label} tie count mismatch for {row.task}")
    _require_finite(
        selection,
        [
            "validation_selected_score_oriented",
            "validation_score_margin",
            "validation_tolerance",
            "selected_exact_bootstrap_probability",
            "selected_tie_bootstrap_probability",
        ],
        label,
    )
    availability = frames["model_availability"]
    label = "model_availability"
    _require_columns(
        availability,
        ["task", "model", "available_seed_count", "available_seeds", "eligible"],
        label,
    )
    _require_exact_keys(
        availability,
        ["task", "model"],
        itertools.product(tasks, models),
        label,
    )
    _require_all_true(availability["eligible"], f"{label}.eligible")
    for row in availability.itertuples(index=False):
        if int(row.available_seed_count) != len(seeds) or _parse_seed_set(row.available_seeds) != seeds:
            raise ValueError(f"{label} seed coverage mismatch for {(row.task, row.model)}")

    expected_dimension_keys = {
        (task, model, dimension)
        for task in tasks
        for model in models
        for dimension in _dimensions_for_task(task)
    }
    reliability = frames["test_reliability"]
    label = "test_reliability"
    _require_columns(
        reliability,
        [
            "task",
            "model",
            "task_type",
            "official_metric",
            "n_seeds",
            "test_n",
            "mean_validation_score_oriented",
            "test_official_metric_raw",
            "test_official_score_oriented",
            "dimension",
            "value",
            "higher_is_better",
            "risk_metric",
            "conformal_coverage",
        ],
        label,
    )
    _require_exact_keys(
        reliability,
        ["task", "model", "dimension"],
        expected_dimension_keys,
        label,
    )
    _validate_task_metadata(reliability, label)
    if set(
        _strict_int_values(
            reliability["n_seeds"], f"{label}.n_seeds", positive=True
        )
    ) != {len(seeds)}:
        raise ValueError(f"{label} n_seeds mismatch")
    _strict_int_values(reliability["test_n"], f"{label}.test_n", positive=True)
    _require_finite(
        reliability,
        [
            "test_n",
            "mean_validation_score_oriented",
            "test_official_metric_raw",
            "test_official_score_oriented",
            "value",
        ],
        label,
    )
    raw_scores = pd.to_numeric(
        reliability["test_official_metric_raw"], errors="raise"
    ).to_numpy(dtype=float)
    oriented_scores = pd.to_numeric(
        reliability["test_official_score_oriented"], errors="raise"
    ).to_numpy(dtype=float)
    expected_oriented = np.asarray(
        [
            oriented_metric(metric, score)
            for metric, score in zip(reliability["official_metric"], raw_scores)
        ],
        dtype=float,
    )
    if not np.allclose(oriented_scores, expected_oriented, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{label} raw/oriented official test score mismatch")

    endpoint = frames["reliability_endpoint"]
    label = "reliability_endpoint"
    _require_columns(
        endpoint,
        [
            "task",
            "task_type",
            "official_metric",
            "dimension",
            "n_models",
            "model_pool",
            "validation_selected_model",
            "test_reliability_oracle_model",
            "reliability_regret",
            "normalized_regret",
            "practical_tolerance",
        ],
        label,
    )
    expected_endpoint_keys = {
        (task, dimension) for task in tasks for dimension in _dimensions_for_task(task)
    }
    _require_exact_keys(endpoint, ["task", "dimension"], expected_endpoint_keys, label)
    _validate_task_metadata(endpoint, label)
    selected_by_task = {
        _norm(row.task): _norm(row.validation_selected_model)
        for row in selection.itertuples(index=False)
    }
    for row in endpoint.itertuples(index=False):
        if int(row.n_models) != len(models) or _parse_semicolon_set(row.model_pool) != models:
            raise ValueError(f"{label} model pool mismatch for {(row.task, row.dimension)}")
        if _norm(row.validation_selected_model) != selected_by_task[_norm(row.task)]:
            raise ValueError(f"{label} selection mismatch for {row.task}")
        if _norm(row.test_reliability_oracle_model) not in models:
            raise ValueError(f"{label} oracle model is outside the final pool")
    _require_finite(
        endpoint,
        ["reliability_regret", "normalized_regret", "practical_tolerance"],
        label,
    )
    if (pd.to_numeric(endpoint["reliability_regret"]) < -1e-12).any():
        raise ValueError(f"{label} contains negative regret")

    summary = frames["reliability_summary"]
    label = "reliability_summary"
    dimensions = sorted({dimension for task in tasks for dimension in _dimensions_for_task(task)})
    _require_columns(
        summary,
        [
            "dimension",
            "n_endpoints",
            "mean_regret",
            "mean_regret_ci_low",
            "mean_regret_ci_high",
            "mean_normalized_regret",
            "practical_effect_signal",
            "tie_robust_effect_signal",
            "random_baseline_retention_signal",
        ],
        label,
    )
    _require_exact_keys(summary, ["dimension"], [(value,) for value in dimensions], label)
    expected_counts = {
        dimension: sum(dimension in _dimensions_for_task(task) for task in tasks)
        for dimension in dimensions
    }
    for row in summary.itertuples(index=False):
        if int(row.n_endpoints) != expected_counts[_norm(row.dimension)]:
            raise ValueError(f"{label} endpoint count mismatch for {row.dimension}")
    _require_finite(
        summary,
        [
            "mean_regret",
            "mean_regret_ci_low",
            "mean_regret_ci_high",
            "mean_normalized_regret",
        ],
        label,
    )
    for column in (
        "practical_effect_signal",
        "tie_robust_effect_signal",
        "random_baseline_retention_signal",
    ):
        _boolean_series(summary[column], f"{label}.{column}")


def _validate_conditional(
    frames: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.tasks}
    models = {_norm(model) for model in contract.models}
    thresholds = {round(float(value), 10) for value in contract.conditional_thresholds}
    if gate.get("status") not in {"GO", "HOLD"}:
        raise ValueError("conditional_gate status must be GO or HOLD")
    if round(float(gate.get("primary_threshold", float("nan"))), 10) not in thresholds:
        raise ValueError("conditional_gate primary_threshold is outside the locked thresholds")

    profiles = frames["conditional_profiles"]
    label = "conditional_profiles"
    _require_columns(
        profiles,
        [
            "task",
            "model",
            "task_type",
            "official_metric",
            "n_seeds",
            "n_samples",
            "performance",
            "ece",
            "brier",
            "coverage_error",
            "applicability_alignment",
            "uncertainty_error_correlation",
        ],
        label,
    )
    _require_exact_keys(profiles, ["task", "model"], itertools.product(tasks, models), label)
    _validate_task_metadata(profiles, label)
    if set(
        _strict_int_values(
            profiles["n_seeds"], f"{label}.n_seeds", positive=True
        )
    ) != {len(contract.seeds)}:
        raise ValueError(f"{label} n_seeds mismatch")
    _strict_int_values(profiles["n_samples"], f"{label}.n_samples", positive=True)
    _require_finite(
        profiles,
        ["n_samples", "performance", "applicability_alignment", "uncertainty_error_correlation"],
        label,
    )

    conditional = frames["conditional_endpoint_model"]
    label = "conditional_endpoint_model"
    _require_columns(
        conditional,
        [
            "task",
            "model",
            "task_type",
            "official_metric",
            "n_seeds",
            "threshold",
            "inside_count",
            "outside_count",
            "eligible",
        ],
        label,
    )
    _require_exact_keys(
        conditional,
        ["task", "model", "threshold"],
        itertools.product(tasks, models, thresholds),
        label,
    )
    _validate_task_metadata(conditional, label)
    if set(
        _strict_int_values(
            conditional["n_seeds"], f"{label}.n_seeds", positive=True
        )
    ) != {len(contract.seeds)}:
        raise ValueError(f"{label} n_seeds mismatch")
    _boolean_series(conditional["eligible"], f"{label}.eligible")
    _require_finite(conditional, ["inside_count", "outside_count"], label)
    if (pd.to_numeric(conditional["inside_count"]) < 0).any() or (
        pd.to_numeric(conditional["outside_count"]) < 0
    ).any():
        raise ValueError(f"{label} has negative domain counts")
    _strict_int_values(
        conditional["inside_count"], f"{label}.inside_count", nonnegative=True
    )
    _strict_int_values(
        conditional["outside_count"], f"{label}.outside_count", nonnegative=True
    )

    summary = frames["conditional_summary"]
    label = "conditional_summary"
    _require_columns(
        summary,
        [
            "threshold",
            "metric",
            "n_pairs",
            "n_endpoints",
            "mean",
            "median",
            "mean_ci_low",
            "mean_ci_high",
            "sign_flip_p",
            "holm_p_within_threshold",
        ],
        label,
    )
    _require_exact_keys(
        summary,
        ["threshold", "metric"],
        itertools.product(thresholds, CONDITIONAL_METRICS),
        label,
    )
    _require_finite(
        summary,
        [
            "n_pairs",
            "n_endpoints",
            "mean",
            "median",
            "mean_ci_low",
            "mean_ci_high",
            "sign_flip_p",
            "holm_p_within_threshold",
        ],
        label,
    )
    if (pd.to_numeric(summary["n_endpoints"]) <= 0).any() or (
        pd.to_numeric(summary["n_endpoints"]) > len(tasks)
    ).any():
        raise ValueError(f"{label} endpoint counts are outside the final contract")

    profile_index = {
        (_norm(row.task), _norm(row.model)): row
        for row in profiles.itertuples(index=False)
    }
    reliability = frames["test_reliability"]
    for row in reliability.itertuples(index=False):
        key = (_norm(row.task), _norm(row.model))
        profile = profile_index[key]
        dimension = _norm(row.dimension)
        if dimension == "applicability_alignment" and not np.isclose(
            float(row.value), float(profile.applicability_alignment), rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"Conditional/reliability applicability mismatch for {key}")
        if dimension == "ece" and not np.isclose(
            float(row.value), float(profile.ece), rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"Conditional/reliability ECE mismatch for {key}")
        if dimension == "brier" and not np.isclose(
            float(row.value), float(profile.brier), rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"Conditional/reliability Brier mismatch for {key}")
        if dimension == "conformal_coverage_error" and not np.isclose(
            abs(0.9 - float(row.conformal_coverage)),
            float(profile.coverage_error),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(f"Conditional/reliability coverage mismatch for {key}")


def _validate_calibration(
    frames: Mapping[str, pd.DataFrame], contract: FinalContract
) -> None:
    tasks = {_norm(task) for task in contract.tasks}
    models = {_norm(model) for model in contract.models}
    seeds = set(contract.seeds)
    classification_tasks = {task for task in tasks if _task_type(task) == "classification"}
    regression_tasks = tasks - classification_tasks
    summary = frames["calibration_summary"]
    label = "calibration_summary"
    _require_columns(
        summary,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "seed",
            "method",
            "test_brier_before",
            "test_brier_after",
            "test_ece_before",
            "test_ece_after",
            "test_mae",
            "test_rmse",
            "test_spearman",
            "coverage",
            "mean_interval_width",
            "error",
        ],
        label,
    )
    expected_keys = {
        (task, model, seed, method)
        for task in classification_tasks
        for model in models
        for seed in seeds
        for method in ("isotonic", "platt")
    } | {
        (task, model, seed, "conformal_90")
        for task in regression_tasks
        for model in models
        for seed in seeds
    }
    _require_exact_keys(
        summary,
        ["task", "model", "seed", "method"],
        expected_keys,
        label,
    )
    _validate_task_metadata(summary, label)
    errors = summary["error"].fillna("").astype(str).str.strip()
    if (errors != "").any():
        preview = summary.loc[errors != "", ["task", "model", "seed", "method", "error"]]
        raise ValueError(f"{label} contains failed rows: {preview.head().to_dict('records')}")
    cls = summary[summary["task"].map(_norm).isin(classification_tasks)]
    reg = summary[summary["task"].map(_norm).isin(regression_tasks)]
    _require_finite(
        cls,
        [
            "test_brier_before",
            "test_brier_after",
            "test_ece_before",
            "test_ece_after",
        ],
        label,
    )
    _require_finite(
        reg,
        ["test_mae", "test_rmse", "test_spearman", "coverage", "mean_interval_width"],
        label,
    )

    bins = frames["calibration_bins"]
    label = "calibration_bins"
    _require_columns(
        bins,
        [
            "task",
            "model",
            "seed",
            "method",
            "bin_left",
            "bin_right",
            "count",
            "empirical_positive_rate",
            "mean_confidence",
        ],
        label,
    )
    expected_bin_keys = {
        (task, model, seed, method, bin_index)
        for task in classification_tasks
        for model in models
        for seed in seeds
        for method in ("uncalibrated", "isotonic", "platt")
        for bin_index in range(contract.calibration_bin_count)
    }
    observed_bin_keys: list[tuple[object, ...]] = []
    for row in bins.itertuples(index=False):
        left = float(row.bin_left)
        right = float(row.bin_right)
        bin_index = int(round(left * contract.calibration_bin_count))
        expected_left = bin_index / contract.calibration_bin_count
        expected_right = (bin_index + 1) / contract.calibration_bin_count
        if not np.isclose(left, expected_left, atol=1e-12) or not np.isclose(
            right, expected_right, atol=1e-12
        ):
            raise ValueError(f"{label} contains a non-locked bin interval: {(left, right)}")
        observed_bin_keys.append(
            (_norm(row.task), _norm(row.model), int(row.seed), _norm(row.method), bin_index)
        )
    if len(observed_bin_keys) != len(set(observed_bin_keys)):
        raise ValueError(f"{label} contains duplicate task/model/seed/method/bin keys")
    if set(observed_bin_keys) != expected_bin_keys:
        missing = expected_bin_keys - set(observed_bin_keys)
        unexpected = set(observed_bin_keys) - expected_bin_keys
        raise ValueError(
            f"{label} key matrix mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    _require_finite(bins, ["bin_left", "bin_right", "count"], label)
    if (pd.to_numeric(bins["count"]) < 0).any():
        raise ValueError(f"{label} contains negative bin counts")
    nonempty_bins = pd.to_numeric(bins["count"]) > 0
    _require_finite(
        bins.loc[nonempty_bins],
        ["empirical_positive_rate", "mean_confidence"],
        label,
    )
    for column in ("empirical_positive_rate", "mean_confidence"):
        empty_values = pd.to_numeric(bins.loc[~nonempty_bins, column], errors="coerce")
        if empty_values.notna().any():
            raise ValueError(f"{label}.{column} must be empty when count=0")

    test_n = (
        frames["test_reliability"]
        .drop_duplicates(["task", "model"])
        .assign(task=lambda value: value["task"].map(_norm), model=lambda value: value["model"].map(_norm))
        .set_index(["task", "model"])["test_n"]
    )
    for key, group in bins.groupby(
        [bins["task"].map(_norm), bins["model"].map(_norm), "seed", bins["method"].map(_norm)],
        sort=False,
    ):
        task, model, _, _ = key
        if int(pd.to_numeric(group["count"]).sum()) != int(test_n.loc[(task, model)]):
            raise ValueError(f"{label} count sum does not match test_n for {key}")


def _validate_ablation(
    frames: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.tasks}
    models = {_norm(model) for model in contract.models}
    seeds = tuple(sorted(contract.seeds))
    expected_runs = len(tasks) * len(models) * len(seeds)
    combinations_by_size = {
        size: tuple(itertools.combinations(seeds, size)) for size in contract.ensemble_sizes
    }
    subsets_per_task_model = sum(len(value) for value in combinations_by_size.values())
    classification_tasks = {task for task in tasks if _task_type(task) == "classification"}
    expected_ensemble_rows = len(tasks) * len(models) * subsets_per_task_model
    expected_calibration_rows = (
        len(classification_tasks)
        * len(models)
        * len(seeds)
        * 3
        * len(contract.ablation_bin_counts)
        * 2
    )
    required_gate = {
        "schema": "posthoc_ablation_v1",
        "status": "PASS",
        "zero_model_fits": True,
        "all_task_seed_model_samples_aligned": True,
        "expected_runs": expected_runs,
        "observed_runs": expected_runs,
        "ensemble_rows": expected_ensemble_rows,
        "calibration_rows": expected_calibration_rows,
        "seed_subsets_per_task_model": subsets_per_task_model,
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"ablation_gate.{key} mismatch: {gate.get(key)!r} != {expected!r}")
    if {int(value) for value in gate.get("ensemble_sizes", [])} != set(contract.ensemble_sizes):
        raise ValueError("ablation_gate ensemble_sizes mismatch")
    if {int(value) for value in gate.get("bin_counts", [])} != set(contract.ablation_bin_counts):
        raise ValueError("ablation_gate bin_counts mismatch")
    if {_norm(value) for value in gate.get("binning_schemes", [])} != {
        "equal_width",
        "equal_mass_tie_preserving",
    }:
        raise ValueError("ablation_gate binning_schemes mismatch")
    if {_norm(value) for value in gate.get("calibration_methods", [])} != {
        "uncalibrated",
        "isotonic",
        "platt",
    }:
        raise ValueError("ablation_gate calibration_methods mismatch")

    ensemble = frames["ablation_ensemble_rows"]
    label = "ablation_ensemble_rows"
    _require_columns(
        ensemble,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "ensemble_size",
            "seed_subset",
            "test_n",
            "test_official_metric",
            "relative_excess_aurc",
        ],
        label,
    )
    expected_ensemble_keys = {
        (task, model, size, ";".join(str(seed) for seed in subset))
        for task in tasks
        for model in models
        for size, subsets in combinations_by_size.items()
        for subset in subsets
    }
    _require_exact_keys(
        ensemble,
        ["task", "model", "ensemble_size", "seed_subset"],
        expected_ensemble_keys,
        label,
    )
    _validate_task_metadata(ensemble, label)
    _require_finite(ensemble, ["test_n", "test_official_metric", "relative_excess_aurc"], label)

    ensemble_summary = frames["ablation_ensemble_summary"]
    label = "ablation_ensemble_summary"
    _require_columns(
        ensemble_summary,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "ensemble_size",
            "n_seed_subsets",
            "test_official_metric_mean",
            "relative_excess_aurc_mean",
        ],
        label,
    )
    _require_exact_keys(
        ensemble_summary,
        ["task", "model", "ensemble_size"],
        itertools.product(tasks, models, contract.ensemble_sizes),
        label,
    )
    _validate_task_metadata(ensemble_summary, label)
    for row in ensemble_summary.itertuples(index=False):
        if int(row.n_seed_subsets) != len(combinations_by_size[int(row.ensemble_size)]):
            raise ValueError(f"{label} subset count mismatch for {(row.task, row.model)}")
    _require_finite(
        ensemble_summary,
        ["test_official_metric_mean", "relative_excess_aurc_mean"],
        label,
    )

    calibration = frames["ablation_calibration_bins"]
    label = "ablation_calibration_bins"
    _require_columns(
        calibration,
        [
            "task",
            "model",
            "seed",
            "method",
            "binning",
            "n_bins_requested",
            "n_bins_actual",
            "ece",
            "brier",
            "nll",
        ],
        label,
    )
    expected_calibration_keys = itertools.product(
        classification_tasks,
        models,
        seeds,
        ("uncalibrated", "isotonic", "platt"),
        ("equal_width", "equal_mass_tie_preserving"),
        contract.ablation_bin_counts,
    )
    _require_exact_keys(
        calibration,
        ["task", "model", "seed", "method", "binning", "n_bins_requested"],
        expected_calibration_keys,
        label,
    )
    _require_finite(calibration, ["n_bins_actual", "ece", "brier", "nll"], label)
    actual_bins = pd.to_numeric(calibration["n_bins_actual"], errors="raise")
    requested_bins = pd.to_numeric(calibration["n_bins_requested"], errors="raise")
    if ((actual_bins < 1) | (actual_bins > requested_bins)).any():
        raise ValueError(f"{label} realized bin counts are invalid")

    alignment = frames["ablation_alignment"]
    label = "ablation_alignment"
    _require_columns(
        alignment,
        [
            "task",
            "model",
            "seed",
            "valid_matches_task_seed_reference",
            "test_matches_task_seed_reference",
            "task_seed_models_aligned",
        ],
        label,
    )
    _require_exact_keys(
        alignment,
        ["task", "model", "seed"],
        itertools.product(tasks, models, seeds),
        label,
    )
    for column in (
        "valid_matches_task_seed_reference",
        "test_matches_task_seed_reference",
        "task_seed_models_aligned",
    ):
        _require_all_true(alignment[column], f"{label}.{column}")

    input_audit = frames["ablation_input_audit"]
    label = "ablation_input_audit"
    _require_columns(input_audit, ["path", "included", "reason"], label)
    if len(input_audit) != expected_runs or input_audit["path"].astype(str).duplicated().any():
        raise ValueError(f"{label} must contain {expected_runs} unique prediction paths")
    _require_all_true(input_audit["included"], f"{label}.included")
    if {_norm(value) for value in input_audit["reason"]} != {"complete"}:
        raise ValueError(f"{label} contains non-complete inputs")


def _validate_y_randomization(
    frames: Mapping[str, pd.DataFrame],
    traditional_winners: Mapping[str, str],
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.tasks}
    summary = frames["y_random_summary"]
    label = "y_random_summary"
    _require_columns(
        summary,
        [
            "task",
            "task_type",
            "model",
            "seed",
            "split_source",
            "split_algorithm",
            "metric",
            "real_score",
            "random_mean",
            "random_std",
            "oriented_margin",
            "z_like_margin",
            "n_trials",
            "train_n",
            "test_n",
        ],
        label,
    )
    _require_exact_keys(summary, ["task"], [(task,) for task in tasks], label)
    for row in summary.itertuples(index=False):
        task = _norm(row.task)
        if _norm(row.task_type) != _task_type(task) or _norm(row.metric) != _task_metric(task):
            raise ValueError(f"{label} metadata mismatch for {row.task}")
        if _norm(row.model) != traditional_winners[task]:
            raise ValueError(f"{label} model is not the locked traditional winner for {row.task}")
        if _strict_int(row.seed, f"{label}.seed", positive=True) != contract.y_random_seed or _strict_int(
            row.n_trials, f"{label}.n_trials", positive=True
        ) != contract.y_random_trials:
            raise ValueError(f"{label} seed/trial mismatch for {row.task}")
        if _norm(row.split_source) != LOCKED_CACHE_SPLIT_SOURCE or str(
            row.split_algorithm
        ) != LOCKED_CACHE_SPLIT_ALGORITHM:
            raise ValueError(f"{label} does not use locked-cache provenance for {row.task}")
        if _strict_int(row.train_n, f"{label}.train_n", positive=True) <= 0 or _strict_int(
            row.test_n, f"{label}.test_n", positive=True
        ) <= 0:
            raise ValueError(f"{label} has invalid sample counts for {row.task}")
    _require_finite(
        summary,
        [
            "real_score",
            "random_mean",
            "random_std",
            "oriented_margin",
            "z_like_margin",
            "train_n",
            "test_n",
        ],
        label,
    )
    if (pd.to_numeric(summary["oriented_margin"], errors="raise") <= 0).any():
        raise ValueError("y_random_summary contains a nonpositive locked negative-control margin")

    scores = frames["y_random_scores"]
    label = "y_random_scores"
    _require_columns(
        scores,
        [
            "task",
            "task_type",
            "model",
            "seed",
            "split_source",
            "split_algorithm",
            "metric",
            "trial",
            "random_score",
        ],
        label,
    )
    _require_exact_keys(
        scores,
        ["task", "trial"],
        itertools.product(tasks, range(1, contract.y_random_trials + 1)),
        label,
    )
    _require_finite(scores, ["random_score"], label)
    summary_index = {_norm(row.task): row for row in summary.itertuples(index=False)}
    for task_value, group in scores.groupby(scores["task"].map(_norm), sort=False):
        row = summary_index[task_value]
        if {_norm(value) for value in group["model"]} != {_norm(row.model)}:
            raise ValueError(f"{label} model mismatch for {task_value}")
        if set(_strict_int_values(group["seed"], f"{label}.seed", positive=True)) != {
            contract.y_random_seed
        }:
            raise ValueError(f"{label} seed mismatch for {task_value}")
        if {_norm(value) for value in group["split_source"]} != {
            LOCKED_CACHE_SPLIT_SOURCE
        } or set(group["split_algorithm"].astype(str)) != {LOCKED_CACHE_SPLIT_ALGORITHM}:
            raise ValueError(f"{label} provenance mismatch for {task_value}")
        values = pd.to_numeric(group["random_score"], errors="raise").to_numpy(dtype=float)
        random_mean = float(np.mean(values))
        random_std = float(np.std(values, ddof=1))
        margin = oriented_metric(_norm(row.metric), float(row.real_score)) - oriented_metric(
            _norm(row.metric), random_mean
        )
        if not np.isclose(random_mean, float(row.random_mean), rtol=1e-10, atol=1e-12):
            raise ValueError(f"{label} random mean mismatch for {task_value}")
        if not np.isclose(random_std, float(row.random_std), rtol=1e-10, atol=1e-12):
            raise ValueError(f"{label} random std mismatch for {task_value}")
        if not np.isclose(margin, float(row.oriented_margin), rtol=1e-10, atol=1e-12):
            raise ValueError(f"{label} oriented margin mismatch for {task_value}")
        if not np.isclose(
            margin / random_std,
            float(row.z_like_margin),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"{label} z-like margin mismatch for {task_value}")


def _validate_cpu_defense_gate(
    gate: Mapping[str, Any],
    contract: FinalContract,
    source_hashes: Mapping[str, str],
) -> None:
    if _strict_int(gate.get("schema_version"), "cpu_defense_gate.schema_version", positive=True) != 1:
        raise ValueError("cpu_defense_gate.schema_version must be 1")
    required = {
        "status": "GO",
        "artifact_gate": "PASS",
        "negative_control_signal": "PASS",
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise ValueError(f"cpu_defense_gate.{key} mismatch: {gate.get(key)!r} != {expected!r}")
    failures = gate.get("failures")
    if failures != []:
        raise ValueError(f"cpu_defense_gate reports failures: {failures!r}")
    y_randomization = gate.get("y_randomization")
    if not isinstance(y_randomization, dict) or _strict_int(
        y_randomization.get("positive_margin_count"),
        "cpu_defense_gate.positive_margin_count",
        nonnegative=True,
    ) != len(contract.tasks):
        raise ValueError("cpu_defense_gate positive_margin_count mismatch")
    if y_randomization.get("nonpositive_tasks") not in ([], None):
        raise ValueError("cpu_defense_gate reports nonpositive y-randomization tasks")
    coverage = gate.get("coverage")
    expected_coverage = {
        "y_summary_rows": len(contract.tasks),
        "y_score_rows": len(contract.tasks) * contract.y_random_trials,
        "shap_rows": len(contract.shap_tasks) * contract.shap_top_k,
        "shap_mapping_rows": len(contract.shap_tasks) * contract.shap_mapping_top_k,
        "mapped_substructures": len(contract.shap_tasks) * contract.shap_mapping_top_k,
    }
    if not isinstance(coverage, dict) or any(
        _strict_int(
            coverage.get(key),
            f"cpu_defense_gate.coverage.{key}",
            nonnegative=True,
        )
        != value
        for key, value in expected_coverage.items()
    ):
        raise ValueError("cpu_defense_gate coverage mismatch")
    gate_inputs = gate.get("inputs")
    source_map = {
        "y_randomization_summary": "y_random_summary",
        "y_randomization_scores": "y_random_scores",
        "shap": "shap_selected",
        "shap_mapping": "shap_mapping",
        "traditional_selection": "traditional_selection",
    }
    if not isinstance(gate_inputs, dict):
        raise ValueError("cpu_defense_gate input hashes are missing")
    for gate_name, source_id in source_map.items():
        record = gate_inputs.get(gate_name)
        if not isinstance(record, dict) or _norm(record.get("sha256")) != source_hashes[source_id]:
            raise ValueError(f"cpu_defense_gate stale or missing input hash for {gate_name}")


def _validate_shap(
    frames: Mapping[str, pd.DataFrame],
    traditional_winners: Mapping[str, str],
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.shap_tasks}
    selected = frames["shap_selected"]
    label = "shap_selected"
    _require_columns(
        selected,
        [
            "task",
            "model",
            "seed",
            "split_source",
            "split_algorithm",
            "rank",
            "feature",
            "mean_abs_shap",
            "mean_feature_value",
            "train_n",
            "test_n",
            "sample_size_actual",
        ],
        label,
    )
    _require_exact_keys(
        selected,
        ["task", "rank"],
        itertools.product(tasks, range(1, contract.shap_top_k + 1)),
        label,
    )
    for row in selected.itertuples(index=False):
        task = _norm(row.task)
        if _norm(row.model) != traditional_winners[task]:
            raise ValueError(f"{label} model is not the locked traditional winner for {row.task}")
        if _strict_int(row.seed, f"{label}.seed", positive=True) != contract.y_random_seed:
            raise ValueError(f"{label} seed mismatch for {row.task}")
        if _norm(row.split_source) != LOCKED_CACHE_SPLIT_SOURCE or str(
            row.split_algorithm
        ) != LOCKED_CACHE_SPLIT_ALGORITHM:
            raise ValueError(f"{label} provenance mismatch for {row.task}")
        _morgan_bit_id(row.feature, f"{label}.feature")
        if (
            _strict_int(row.train_n, f"{label}.train_n", positive=True) <= 0
            or _strict_int(row.test_n, f"{label}.test_n", positive=True) <= 0
            or _strict_int(
                row.sample_size_actual,
                f"{label}.sample_size_actual",
                positive=True,
            )
            <= 0
        ):
            raise ValueError(f"{label} has invalid sample counts")
    _require_finite(
        selected,
        ["mean_abs_shap", "mean_feature_value", "train_n", "test_n", "sample_size_actual"],
        label,
    )
    _strict_int_values(selected["rank"], f"{label}.rank", positive=True)

    mapping = frames["shap_mapping"]
    label = "shap_mapping"
    _require_columns(
        mapping,
        [
            "task",
            "model",
            "seed",
            "split_source",
            "split_algorithm",
            "rank",
            "feature",
            "bit_id",
            "mean_abs_shap",
            "example_smiles",
            "bit_atom_index",
            "bit_radius",
            "substructure_smiles",
        ],
        label,
    )
    _require_exact_keys(
        mapping,
        ["task", "rank"],
        itertools.product(tasks, range(1, contract.shap_mapping_top_k + 1)),
        label,
    )
    if mapping["substructure_smiles"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains unmapped Morgan bits")
    _require_finite(mapping, ["bit_id", "mean_abs_shap", "bit_atom_index", "bit_radius"], label)
    _strict_int_values(mapping["rank"], f"{label}.rank", positive=True)
    _strict_int_values(mapping["seed"], f"{label}.seed", positive=True)
    _strict_int_values(mapping["bit_id"], f"{label}.bit_id", nonnegative=True)
    _strict_int_values(
        mapping["bit_atom_index"], f"{label}.bit_atom_index", nonnegative=True
    )
    _strict_int_values(mapping["bit_radius"], f"{label}.bit_radius", nonnegative=True)
    selected_index = {
        (_norm(row.task), _strict_int(row.rank, f"{label}.rank", positive=True)): row
        for row in selected.itertuples(index=False)
        if _strict_int(row.rank, f"{label}.rank", positive=True)
        <= contract.shap_mapping_top_k
    }
    for row in mapping.itertuples(index=False):
        key = (_norm(row.task), _strict_int(row.rank, f"{label}.rank", positive=True))
        source = selected_index[key]
        if (
            _norm(row.model) != traditional_winners[key[0]]
            or _strict_int(row.seed, f"{label}.seed", positive=True)
            != contract.y_random_seed
            or _norm(row.split_source) != LOCKED_CACHE_SPLIT_SOURCE
            or str(row.split_algorithm) != LOCKED_CACHE_SPLIT_ALGORITHM
        ):
            raise ValueError(f"{label} provenance/model mismatch for {key}")
        if str(row.feature) != str(source.feature) or _strict_int(
            row.bit_id, f"{label}.bit_id", nonnegative=True
        ) != _morgan_bit_id(row.feature, f"{label}.feature"):
            raise ValueError(f"{label} feature/bit mismatch for {key}")
        if not np.isclose(
            float(row.mean_abs_shap), float(source.mean_abs_shap), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"{label} SHAP value mismatch for {key}")


def _validate_cleaned_sensitivity(
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    traditional_winners: Mapping[str, str],
    contract: FinalContract,
    source_records: Mapping[str, Mapping[str, Any]],
) -> None:
    tasks = {_norm(task) for task in contract.cleaned_tasks}
    models = {_norm(model) for model in contract.cleaned_models}
    seeds = set(contract.seeds)
    expected_rows = len(tasks) * len(models) * len(seeds)
    config_audit = config.get("audit")
    if not isinstance(config_audit, dict) or config_audit.get("scaffold_overlap") is not False:
        raise ValueError("cleaned_config audit.scaffold_overlap must be false")
    config_tasks = [_norm(value) for value in config.get("selected_tasks_for_cleaned_rerun", [])]
    config_models = [_norm(value) for value in config.get("models", [])]
    config_seeds = _strict_int_values(
        config.get("seeds", []), "cleaned_config.seeds", positive=True
    )
    production_contract = contract == DEFAULT_FINAL_CONTRACT
    if config.get("protocol") != "paired-common-test-v1" or (
        production_contract
        and (
            set(config_tasks) != tasks
            or len(config_tasks) != len(tasks)
            or len(set(config_tasks)) != len(config_tasks)
            or set(config_models) != models
            or len(config_models) != len(models)
            or len(set(config_models)) != len(config_models)
            or set(config_seeds) != seeds
            or len(config_seeds) != len(seeds)
            or len(set(config_seeds)) != len(config_seeds)
        )
    ):
        raise ValueError("cleaned_config does not match the final cleaned contract")
    if _strict_int(
        manifest.get("schema_version"),
        "cleaned_manifest.schema_version",
        positive=True,
    ) != 2:
        raise ValueError("cleaned_manifest schema_version must be 2")
    if manifest.get("cpu_only") is not True:
        raise ValueError("cleaned_manifest must declare cpu_only=true")
    if _norm(manifest.get("split_source")) != LOCKED_CACHE_SPLIT_SOURCE or str(
        manifest.get("split_algorithm")
    ) != LOCKED_CACHE_SPLIT_ALGORITHM:
        raise ValueError("cleaned_manifest does not match the locked-cache split protocol")
    if manifest.get("parser_compatibility_repairs_count_as_cleaning_deletions") is not False:
        raise ValueError("cleaned_manifest parser repairs must not count as cleaning deletions")
    manifest_tasks = [_norm(value) for value in manifest.get("tasks", [])]
    if (
        set(manifest_tasks) != tasks
        or len(manifest_tasks) != len(tasks)
        or len(set(manifest_tasks)) != len(manifest_tasks)
    ):
        raise ValueError("cleaned_manifest task set mismatch")
    manifest_models = [_norm(value) for value in manifest.get("models", [])]
    if (
        set(manifest_models) != models
        or len(manifest_models) != len(models)
        or len(set(manifest_models)) != len(manifest_models)
    ):
        raise ValueError("cleaned_manifest model set mismatch")
    manifest_seeds = _strict_int_values(
        manifest.get("seeds", []), "cleaned_manifest.seeds", positive=True
    )
    if set(manifest_seeds) != seeds or len(manifest_seeds) != len(seeds):
        raise ValueError("cleaned_manifest seed set mismatch")
    observed_occurrences = {
        _norm(key): _strict_int(
            value,
            f"cleaned_manifest.compatibility_source_occurrences.{key}",
            nonnegative=True,
        )
        for key, value in manifest.get("compatibility_source_occurrences", {}).items()
    }
    locked_nonzero = {
        _norm(task): _strict_int(
            value,
            f"contract.compatibility_occurrences.{task}",
            nonnegative=True,
        )
        for task, value in contract.compatibility_occurrences
    }
    expected_occurrences = {task: locked_nonzero.get(task, 0) for task in tasks}
    if observed_occurrences != expected_occurrences:
        raise ValueError(
            "cleaned_manifest compatibility occurrence matrix mismatch: "
            f"{observed_occurrences} != {expected_occurrences}"
        )

    allowed_statuses = {
        "SENSITIVITY_STABLE",
        "SENSITIVITY_MIXED",
        "SENSITIVITY_SENSITIVE",
    }
    if gate.get("status") not in allowed_statuses:
        raise ValueError("cleaned_gate status must be a completed sensitivity status")
    required_gate_values = {
        "complete_expected_matrix": True,
        "expected_rows": expected_rows,
        "observed_rows": expected_rows,
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
        "n_tasks": len(tasks),
    }
    for key, expected in required_gate_values.items():
        observed = gate.get(key)
        if isinstance(expected, int) and not isinstance(expected, bool):
            observed = _strict_int(
                observed,
                f"cleaned_gate.{key}",
                nonnegative=True,
            )
        if observed != expected:
            raise ValueError(f"cleaned_gate.{key} mismatch: {gate.get(key)!r} != {expected!r}")
    if _norm(gate.get("split_source")) != LOCKED_CACHE_SPLIT_SOURCE or gate.get(
        "split_source_gate"
    ) is not True:
        raise ValueError("cleaned_gate split provenance did not pass")
    if str(gate.get("split_algorithm")) != LOCKED_CACHE_SPLIT_ALGORITHM:
        raise ValueError("cleaned_gate split_algorithm mismatch")
    manifest_gate = gate.get("manifest_provenance_gate")
    if not isinstance(manifest_gate, dict) or not manifest_gate:
        raise ValueError("cleaned_gate manifest_provenance_gate is missing")
    if _norm(manifest_gate.get("split_source")) != LOCKED_CACHE_SPLIT_SOURCE or str(
        manifest_gate.get("split_algorithm")
    ) != LOCKED_CACHE_SPLIT_ALGORITHM:
        raise ValueError("cleaned_gate manifest provenance protocol mismatch")
    checks = manifest_gate.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("cleaned_gate manifest provenance checks did not all pass")

    run_gate = gate.get("run_integrity_gate")
    required_run_checks = (
        "finite_core_values",
        "strict_integer_fields",
        "fingerprint_contract_matches",
        "single_thread_cpu_contract_matches",
        "task_metadata_matches_tdc",
        "primary_metric_columns_match",
        "paired_delta_algebra_matches",
        "sample_counts_consistent_across_models",
    )
    if not isinstance(run_gate, dict) or any(
        run_gate.get(name) is not True for name in required_run_checks
    ):
        raise ValueError("cleaned_gate run_integrity_gate did not fully pass")
    if any(value is not True for value in run_gate.values() if isinstance(value, bool)):
        raise ValueError("cleaned_gate run_integrity_gate contains a false check")
    if _strict_int(
        run_gate.get("validated_rows"),
        "cleaned_gate.run_integrity_gate.validated_rows",
        nonnegative=True,
    ) != expected_rows:
        raise ValueError("cleaned_gate run_integrity_gate row count mismatch")

    audit_gate = gate.get("audit_provenance_gate")
    required_audit_checks = (
        "exact_task_seed_matrix",
        "schema_v2",
        "post_precedence_overlap_zero",
        "parser_repair_counts_match_sources",
        "split_output_counts_match_runs",
    )
    if not isinstance(audit_gate, dict) or any(
        audit_gate.get(name) is not True for name in required_audit_checks
    ):
        raise ValueError("cleaned_gate audit_provenance_gate did not fully pass")
    if any(value is not True for value in audit_gate.values() if isinstance(value, bool)):
        raise ValueError("cleaned_gate audit_provenance_gate contains a false check")
    expected_payloads = len(tasks) * len(seeds)
    for key, expected in (
        ("payloads", expected_payloads),
        ("expected_payloads", expected_payloads),
        ("flattened_rows", expected_payloads * 3),
    ):
        if _strict_int(
            audit_gate.get(key),
            f"cleaned_gate.audit_provenance_gate.{key}",
            nonnegative=True,
        ) != expected:
            raise ValueError(f"cleaned_gate audit provenance {key} mismatch")

    input_artifacts = gate.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise ValueError("cleaned_gate input_artifacts are missing")
    artifact_sources = {
        "runs": "cleaned_runs",
        "audit": "cleaned_raw_audit",
        "manifest": "cleaned_manifest",
        "config": "cleaned_config",
    }
    for artifact_name, source_id in artifact_sources.items():
        artifact = input_artifacts.get(artifact_name)
        source = source_records[source_id]
        if not isinstance(artifact, dict):
            raise ValueError(f"cleaned_gate input_artifacts.{artifact_name} is missing")
        if str(artifact.get("sha256", "")).lower() != str(source["sha256"]).lower():
            raise ValueError(
                f"cleaned_gate input_artifacts.{artifact_name} SHA is stale"
            )
        if _strict_int(
            artifact.get("bytes"),
            f"cleaned_gate.input_artifacts.{artifact_name}.bytes",
            nonnegative=True,
        ) != _strict_int(
            source["bytes"], f"source {source_id}.bytes", nonnegative=True
        ):
            raise ValueError(
                f"cleaned_gate input_artifacts.{artifact_name} byte count is stale"
            )
    if _strict_int(
        input_artifacts["runs"].get("rows"),
        "cleaned_gate.input_artifacts.runs.rows",
        nonnegative=True,
    ) != expected_rows:
        raise ValueError("cleaned_gate input run row count mismatch")
    if _strict_int(
        input_artifacts["audit"].get("payloads"),
        "cleaned_gate.input_artifacts.audit.payloads",
        nonnegative=True,
    ) != expected_payloads:
        raise ValueError("cleaned_gate input audit payload count mismatch")
    if str(audit_gate.get("audit_sha256", "")).lower() != str(
        source_records["cleaned_raw_audit"]["sha256"]
    ).lower() or _strict_int(
        audit_gate.get("audit_bytes"),
        "cleaned_gate.audit_provenance_gate.audit_bytes",
        nonnegative=True,
    ) != _strict_int(
        source_records["cleaned_raw_audit"]["bytes"],
        "source cleaned_raw_audit.bytes",
        nonnegative=True,
    ):
        raise ValueError("cleaned_gate audit provenance artifact binding is stale")
    if str(manifest_gate.get("manifest_sha256", "")).lower() != str(
        source_records["cleaned_manifest"]["sha256"]
    ).lower() or _strict_int(
        manifest_gate.get("manifest_bytes"),
        "cleaned_gate.manifest_provenance_gate.manifest_bytes",
        nonnegative=True,
    ) != _strict_int(
        source_records["cleaned_manifest"]["bytes"],
        "source cleaned_manifest.bytes",
        nonnegative=True,
    ):
        raise ValueError("cleaned_gate manifest provenance artifact binding is stale")
    if str(manifest_gate.get("config_sha256", "")).lower() != str(
        source_records["cleaned_config"]["sha256"]
    ).lower():
        raise ValueError("cleaned_gate manifest config binding is stale")
    execution_config_sha256 = str(manifest.get("execution_config_sha256", ""))
    corrections = manifest.get("config_metadata_corrections")
    if _norm(manifest.get("config_sha256")) != _norm(
        source_records["cleaned_config"]["sha256"]
    ):
        raise ValueError("cleaned_manifest config_sha256 is stale")
    if not _SHA256_TEXT.fullmatch(execution_config_sha256):
        raise ValueError("cleaned_manifest execution_config_sha256 is invalid")
    current_config_sha256 = str(source_records["cleaned_config"]["sha256"]).lower()
    manifest_config_sha256 = _norm(manifest.get("config_sha256"))
    if not isinstance(corrections, list):
        raise ValueError("cleaned_manifest config metadata corrections must be a list")
    if len(corrections) == 0:
        if execution_config_sha256.lower() != manifest_config_sha256 or (
            execution_config_sha256.lower() != current_config_sha256
        ):
            raise ValueError(
                "cleaned_manifest without metadata correction must bind execution and current config hashes"
            )
    elif len(corrections) == 1:
        correction = corrections[0]
        if not isinstance(correction, dict) or not (
            correction.get("field") == "audit.scaffold_overlap"
            and correction.get("old_value") is True
            and correction.get("new_value") is False
            and correction.get("computational_effect") is False
            and correction.get("execution_config_sha256") == execution_config_sha256
            and correction.get("corrected_config_sha256") == current_config_sha256
            and isinstance(correction.get("reason"), str)
            and bool(str(correction.get("reason")).strip())
        ):
            raise ValueError("cleaned_manifest config metadata correction is invalid")
    else:
        raise ValueError("cleaned_manifest has multiple config metadata corrections")
    if (
        manifest_gate.get("execution_config_sha256") != execution_config_sha256
        or manifest_gate.get("config_metadata_corrections") != corrections
    ):
        raise ValueError("cleaned_gate manifest correction metadata is stale")
    for required_check in (
        "execution_config_hash_preserved",
        "noncomputational_config_correction_transparent",
        "config_sha256_recomputed",
    ):
        if checks.get(required_check) is not True:
            raise ValueError(
                f"cleaned_gate manifest provenance is missing {required_check}"
            )
    output_artifacts = gate.get("output_artifacts")
    output_sources = {
        "task_model": "cleaned_task_model",
        "task": "cleaned_task",
        "audit": "cleaned_audit",
    }
    if not isinstance(output_artifacts, dict) or set(output_artifacts) != set(
        output_sources
    ):
        raise ValueError("cleaned_gate output_artifacts are incomplete")
    for output_name, source_id in output_sources.items():
        output_record = output_artifacts.get(output_name)
        source = source_records[source_id]
        if not isinstance(output_record, dict):
            raise ValueError(f"cleaned_gate output_artifacts.{output_name} is missing")
        if (
            output_record.get("serialization")
            != "utf8-pandas-csv-lf-float17g-v1"
            or _norm(output_record.get("sha256")) != _norm(source["sha256"])
        ):
            raise ValueError(
                f"cleaned_gate output_artifacts.{output_name} SHA/serialization is stale"
            )
        for field in ("bytes", "rows"):
            if _strict_int(
                output_record.get(field),
                f"cleaned_gate.output_artifacts.{output_name}.{field}",
                nonnegative=True,
            ) != _strict_int(
                source[field], f"source {source_id}.{field}", nonnegative=True
            ):
                raise ValueError(
                    f"cleaned_gate output_artifacts.{output_name} {field} is stale"
                )
    expected_config_logical_path = FINAL_SOURCE_PATHS["cleaned_config"].removeprefix(
        "@project/"
    )
    data_logical_path = str(manifest_gate.get("data_logical_path", ""))
    if (
        str(input_artifacts["config"].get("logical_path", ""))
        != expected_config_logical_path
        or str(manifest_gate.get("config_logical_path", ""))
        != expected_config_logical_path
        or str(manifest.get("config_logical_path", ""))
        != expected_config_logical_path
        or data_logical_path not in {"data/tdc_official", "tdc_official"}
    ):
        raise ValueError("cleaned_gate logical config/data paths are not the locked project paths")
    if production_contract:
        input_hashes = manifest.get("input_sha256")
        if not isinstance(input_hashes, dict) or {
            _norm(task) for task in input_hashes
        } != tasks:
            raise ValueError("cleaned_manifest input hash task set is incomplete")
        task_keys = {_norm(task): str(task) for task in input_hashes}
        data_root = ROOT / "data" / "tdc_official" / "admet_group"
        for task in tasks:
            task_hashes = input_hashes[task_keys[task]]
            if not isinstance(task_hashes, dict) or set(task_hashes) != {
                "test.csv",
                "train_val.csv",
            }:
                raise ValueError(
                    f"cleaned_manifest input hashes are incomplete for {task}"
                )
            for filename, expected_hash in task_hashes.items():
                source_path = data_root / task_keys[task].lower() / filename
                if (
                    not source_path.is_file()
                    or not _SHA256_TEXT.fullmatch(str(expected_hash).lower())
                    or _sha256_path(source_path) != str(expected_hash).lower()
                ):
                    raise ValueError(
                        f"cleaned_manifest source hash mismatch for {task}/{filename}"
                    )
    for nonportable_field in (
        "verified_config_path",
        "verified_config_path_nonportable",
        "verified_data_dir_nonportable",
    ):
        if nonportable_field in manifest_gate:
            raise ValueError(
                f"cleaned_gate manifest provenance contains {nonportable_field}"
            )

    runs = frames["cleaned_runs"]
    label = "cleaned_runs"
    _require_columns(
        runs,
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "seed",
            "split_source",
            "split_algorithm",
            "radius",
            "n_bits",
            "threads_per_model",
            "official_train_n",
            "cleaned_train_n",
            "common_valid_n",
            "common_test_n",
            "primary_official_common_test",
            "primary_cleaned_common_test",
            "primary_raw_delta_cleaned_minus_official",
            "primary_oriented_delta",
            "primary_relative_oriented_delta",
        ],
        label,
    )
    _require_exact_keys(
        runs,
        ["task", "model", "seed"],
        itertools.product(tasks, models, seeds),
        label,
    )
    _validate_task_metadata(runs, label)
    for column in (
        "seed",
        "radius",
        "n_bits",
        "threads_per_model",
        "official_train_n",
        "cleaned_train_n",
        "common_valid_n",
        "common_test_n",
    ):
        _strict_int_values(runs[column], f"{label}.{column}", positive=True)
    if set(_strict_int_values(runs["radius"], f"{label}.radius", positive=True)) != {2}:
        raise ValueError(f"{label} radius fingerprint mismatch")
    if set(_strict_int_values(runs["n_bits"], f"{label}.n_bits", positive=True)) != {2048}:
        raise ValueError(f"{label} n_bits fingerprint mismatch")
    if set(
        _strict_int_values(
            runs["threads_per_model"], f"{label}.threads_per_model", positive=True
        )
    ) != {1}:
        raise ValueError(f"{label} CPU thread contract mismatch")
    if {_norm(value) for value in runs["split_source"]} != {LOCKED_CACHE_SPLIT_SOURCE} or set(
        runs["split_algorithm"].astype(str)
    ) != {LOCKED_CACHE_SPLIT_ALGORITHM}:
        raise ValueError(f"{label} split provenance mismatch")
    _require_finite(
        runs,
        [
            "primary_official_common_test",
            "primary_cleaned_common_test",
            "primary_raw_delta_cleaned_minus_official",
            "primary_oriented_delta",
            "primary_relative_oriented_delta",
        ],
        label,
    )

    task_summary = frames["cleaned_task"]
    label = "cleaned_task"
    _require_columns(
        task_summary,
        [
            "task",
            "official_metric",
            "official_selected_model",
            "cleaned_best_model",
            "winner_changed_after_cleaning",
            "n_models",
            "task_material_change",
        ],
        label,
    )
    _require_exact_keys(task_summary, ["task"], [(task,) for task in tasks], label)
    for row in task_summary.itertuples(index=False):
        task = _norm(row.task)
        if _norm(row.official_metric) != _task_metric(task):
            raise ValueError(f"{label} metric mismatch for {row.task}")
        if _norm(row.cleaned_best_model) not in models or _strict_int(
            row.n_models, f"{label}.n_models", positive=True
        ) != len(models):
            raise ValueError(f"{label} model summary mismatch for {row.task}")
    _boolean_series(
        task_summary["winner_changed_after_cleaning"],
        f"{label}.winner_changed_after_cleaning",
    )
    _boolean_series(task_summary["task_material_change"], f"{label}.task_material_change")

    model_summary = frames["cleaned_task_model"]
    label = "cleaned_task_model"
    _require_columns(
        model_summary,
        [
            "task",
            "model",
            "official_metric",
            "n_seeds",
            "primary_official_mean",
            "primary_cleaned_mean",
            "oriented_delta_mean",
            "material_change",
        ],
        label,
    )
    _require_exact_keys(
        model_summary,
        ["task", "model"],
        itertools.product(tasks, models),
        label,
    )
    if set(
        _strict_int_values(
            model_summary["n_seeds"], f"{label}.n_seeds", positive=True
        )
    ) != {len(seeds)}:
        raise ValueError(f"{label} n_seeds mismatch")
    _require_finite(
        model_summary,
        ["primary_official_mean", "primary_cleaned_mean", "oriented_delta_mean"],
        label,
    )
    _boolean_series(model_summary["material_change"], f"{label}.material_change")
    for task, group in runs.groupby(runs["task"].map(_norm), sort=False):
        metric = _task_metric(task)
        official_means = group.groupby(group["model"].map(_norm))[
            "primary_official_common_test"
        ].mean()
        cleaned_means = group.groupby(group["model"].map(_norm))[
            "primary_cleaned_common_test"
        ].mean()
        if metric == "mae":
            official_winner = _norm(official_means.idxmin())
            cleaned_winner = _norm(cleaned_means.idxmin())
        else:
            official_winner = _norm(official_means.idxmax())
            cleaned_winner = _norm(cleaned_means.idxmax())
        row = task_summary[task_summary["task"].map(_norm) == task].iloc[0]
        if _norm(row["official_selected_model"]) != official_winner or _norm(
            row["cleaned_best_model"]
        ) != cleaned_winner:
            raise ValueError(f"cleaned_task winner recomputation mismatch for {task}")

    audit = frames["cleaned_audit"]
    label = "cleaned_audit"
    overlap_columns = (
        "overlap_after_train_test",
        "overlap_after_train_valid",
        "overlap_after_valid_test",
    )
    _require_columns(audit, ["task", "seed", "split", *overlap_columns], label)
    expected_audit_keys = itertools.product(tasks, seeds, ("train", "valid", "test"))
    _require_exact_keys(audit, ["task", "seed", "split"], expected_audit_keys, label)
    _strict_int_values(audit["seed"], f"{label}.seed", positive=True)
    for column in overlap_columns:
        if any(
            value != 0
            for value in _strict_int_values(
                audit[column], f"{label}.{column}", nonnegative=True
            )
        ):
            raise ValueError(f"{label} reports residual post-cleaning overlap")
    if any((pd.to_numeric(audit[column]) != 0).any() for column in overlap_columns):
        raise ValueError(f"{label} reports residual post-cleaning overlap")


def _validate_cleaned_prediction_audit(
    frame: pd.DataFrame,
    gate: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    runs: pd.DataFrame,
    predictions_dir: Path,
    contract: FinalContract,
) -> None:
    tasks = {_norm(task) for task in contract.cleaned_tasks}
    models = {_norm(model) for model in contract.cleaned_models}
    seeds = set(contract.seeds)
    expected_rows = len(tasks) * len(models) * len(seeds)
    unsupported_models = sorted(models - set(MODEL_PREDICTION_DTYPES))
    if unsupported_models:
        raise ValueError(
            "cleaned prediction metric-replay contract has unsupported models: "
            f"{unsupported_models}"
        )
    if _strict_int(
        gate.get("schema_version"),
        "cleaned_prediction_gate.schema_version",
        positive=True,
    ) != 2:
        raise ValueError("cleaned_prediction_gate.schema_version must be 2")
    required_gate = {
        "status": "GO",
        "complete_expected_matrix": True,
        "expected_prediction_files": expected_rows,
        "observed_prediction_files": expected_rows,
        "audited_prediction_files": expected_rows,
        "sample_label_identity_aligned_within_task": True,
        "source_common_test_reconstructed_exactly": True,
        "prediction_metrics_recomputed_match_runs": True,
        "failures": [],
    }
    for key, expected in required_gate.items():
        observed = gate.get(key)
        if isinstance(expected, int) and not isinstance(expected, bool):
            observed = _strict_int(
                observed,
                f"cleaned_prediction_gate.{key}",
                nonnegative=True,
            )
        if observed != expected:
            raise ValueError(
                f"cleaned_prediction_gate.{key} mismatch: {gate.get(key)!r} != {expected!r}"
            )
    manifest_gate = gate.get("manifest_provenance_gate")
    if not isinstance(manifest_gate, dict) or not manifest_gate:
        raise ValueError("cleaned_prediction_gate manifest provenance is missing")
    if _norm(manifest_gate.get("split_source")) != LOCKED_CACHE_SPLIT_SOURCE or str(
        manifest_gate.get("split_algorithm")
    ) != LOCKED_CACHE_SPLIT_ALGORITHM:
        raise ValueError("cleaned_prediction_gate manifest protocol mismatch")
    checks = manifest_gate.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("cleaned_prediction_gate manifest checks did not all pass")
    expected_config_logical_path = FINAL_SOURCE_PATHS["cleaned_config"].removeprefix(
        "@project/"
    )
    if manifest_gate.get("config_logical_path") != expected_config_logical_path:
        raise ValueError("cleaned_prediction_gate config_logical_path mismatch")
    if any(
        field in manifest_gate
        for field in ("verified_config_path", "verified_config_path_nonportable")
    ):
        raise ValueError("cleaned_prediction_gate contains a nonportable config path")
    inputs = gate.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("cleaned_prediction_gate input hashes are missing")
    for gate_name, source_id in (
        ("runs", "cleaned_runs"),
        ("manifest", "cleaned_manifest"),
        ("config", "cleaned_config"),
    ):
        record = inputs.get(gate_name)
        source = source_records[source_id]
        if not isinstance(record, dict) or _norm(record.get("sha256")) != _norm(
            source["sha256"]
        ):
            raise ValueError(f"cleaned_prediction_gate stale or missing hash for {gate_name}")
        if _strict_int(
            record.get("bytes"),
            f"cleaned_prediction_gate.inputs.{gate_name}.bytes",
            nonnegative=True,
        ) != _strict_int(
            source["bytes"], f"source {source_id}.bytes", nonnegative=True
        ):
            raise ValueError(f"cleaned_prediction_gate stale byte count for {gate_name}")

    metric_replay_policy = gate.get("metric_replay_policy")
    expected_metric_replay_policy = _metric_replay_policy_record()
    if not isinstance(metric_replay_policy, dict):
        raise ValueError("cleaned_prediction_gate metric replay policy is missing")
    if set(metric_replay_policy) != set(expected_metric_replay_policy):
        raise ValueError("cleaned_prediction_gate metric replay policy schema mismatch")
    supplied_policy_payload = {
        key: value for key, value in metric_replay_policy.items() if key != "sha256"
    }
    if not _SHA256_TEXT.fullmatch(str(metric_replay_policy.get("sha256", ""))) or (
        metric_replay_policy["sha256"]
        != _stable_json_sha256(supplied_policy_payload)
    ):
        raise ValueError("cleaned_prediction_gate metric replay policy SHA is stale")
    if metric_replay_policy != expected_metric_replay_policy:
        raise ValueError("cleaned_prediction_gate metric replay policy contract mismatch")

    audit_csv = gate.get("audit_csv")
    audit_source = source_records["cleaned_prediction_audit"]
    if not isinstance(audit_csv, dict):
        raise ValueError("cleaned_prediction_gate audit_csv binding is missing")
    if _norm(audit_csv.get("sha256")) != _norm(audit_source["sha256"]):
        raise ValueError("cleaned_prediction_gate audit CSV SHA is stale")
    for field in ("bytes", "rows"):
        if _strict_int(
            audit_csv.get(field),
            f"cleaned_prediction_gate.audit_csv.{field}",
            nonnegative=True,
        ) != _strict_int(
            audit_source[field],
            f"source cleaned_prediction_audit.{field}",
            nonnegative=True,
        ):
            raise ValueError(f"cleaned_prediction_gate audit CSV {field} is stale")
    audit_columns = audit_csv.get("columns")
    if audit_columns != CLEANED_PREDICTION_AUDIT_COLUMNS:
        raise ValueError("cleaned_prediction_gate audit CSV columns are stale")

    label = "cleaned_prediction_audit"
    if list(frame.columns) != CLEANED_PREDICTION_AUDIT_COLUMNS:
        raise ValueError(f"{label} schema/order mismatch")
    _require_exact_keys(
        frame,
        ["task", "model", "seed"],
        itertools.product(tasks, models, seeds),
        label,
    )
    if not (
        frame["metric_replay_policy_id"].astype(str) == METRIC_REPLAY_POLICY_ID
    ).all():
        raise ValueError(f"{label} metric replay policy ID mismatch")
    for row in frame[["model", "metric_prediction_dtype"]].itertuples(index=False):
        model = _norm(row.model)
        if str(row.metric_prediction_dtype) != MODEL_PREDICTION_DTYPES[model]:
            raise ValueError(
                f"{label} metric prediction dtype mismatch for model {model}"
            )
    if {_norm(value) for value in frame["split_source"]} != {LOCKED_CACHE_SPLIT_SOURCE} or set(
        frame["split_algorithm"].astype(str)
    ) != {LOCKED_CACHE_SPLIT_ALGORITHM}:
        raise ValueError(f"{label} split provenance mismatch")
    for row in frame[["task", "task_type"]].drop_duplicates().itertuples(index=False):
        if _norm(row.task_type) != _task_type(_norm(row.task)):
            raise ValueError(f"{label} task_type mismatch for {row.task}")
    for column in (
        "row_count_matches_run",
        "finite_labels_and_predictions",
        "classification_probability_range_ok",
        "canonical_smiles_complete",
        "canonical_smiles_unique",
        "source_common_test_exact",
        "official_metric_matches_run",
        "cleaned_metric_matches_run",
        "paired_delta_matches_run",
    ):
        _require_all_true(frame[column], f"{label}.{column}")
    for column in ("prediction_bytes", "rows"):
        _strict_int_values(frame[column], f"{label}.{column}", positive=True)
    _require_finite(
        frame,
        ["recomputed_official_common_test", "recomputed_cleaned_common_test"],
        label,
    )
    for column in (
        "prediction_sha256",
        "sample_label_identity_sha256",
        "source_sample_label_identity_sha256",
    ):
        values = frame[column].fillna("").astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if not values.all():
            raise ValueError(f"{label}.{column} contains invalid hashes")
    if not (
        frame["sample_label_identity_sha256"].astype(str)
        == frame["source_sample_label_identity_sha256"].astype(str)
    ).all():
        raise ValueError(f"{label} source/sample identity hashes differ")
    if frame["prediction_path"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate prediction paths")
    identity_counts = frame.groupby(frame["task"].map(_norm))[
        "sample_label_identity_sha256"
    ].nunique()
    if not (identity_counts == 1).all():
        raise ValueError(f"{label} sample/label identities are not aligned within tasks")

    unique_hash_counts = gate.get("unique_sample_label_hashes_by_task")
    if not isinstance(unique_hash_counts, dict):
        raise ValueError("cleaned_prediction_gate unique identity counts are missing")
    normalized_unique_counts = {
        _norm(task): _strict_int(
            value,
            f"cleaned_prediction_gate.unique_sample_label_hashes_by_task.{task}",
            nonnegative=True,
        )
        for task, value in unique_hash_counts.items()
    }
    if normalized_unique_counts != {task: 1 for task in tasks}:
        raise ValueError("cleaned_prediction_gate unique identity counts mismatch")

    expected_files = {
        f"{task}_{model}_seed{seed}_paired.csv": (task, model, seed)
        for task, model, seed in itertools.product(tasks, models, seeds)
    }
    if not predictions_dir.is_dir():
        raise ValueError(f"cleaned prediction directory is missing: {predictions_dir}")
    actual_files = {
        path.name: path
        for path in predictions_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and not path.name.startswith("._")
    }
    if set(actual_files) != set(expected_files):
        raise ValueError(
            "cleaned prediction file matrix is stale: "
            f"missing={sorted(set(expected_files) - set(actual_files))[:5]}, "
            f"unexpected={sorted(set(actual_files) - set(expected_files))[:5]}"
        )

    audit_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in frame.to_dict("records"):
        key = (
            _norm(record["task"]),
            _norm(record["model"]),
            _strict_int(record["seed"], f"{label}.seed", positive=True),
        )
        audit_lookup[key] = record
    runs_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in runs.to_dict("records"):
        key = (
            _norm(record["task"]),
            _norm(record["model"]),
            _strict_int(record["seed"], "cleaned_runs.seed", positive=True),
        )
        runs_lookup[key] = record

    inventory_records: list[dict[str, object]] = []
    inventory_entries: dict[str, dict[str, object]] = {}
    total_bytes = 0
    total_rows = 0
    for filename, key in sorted(expected_files.items()):
        audit_row = audit_lookup[key]
        run_row = runs_lookup[key]
        if Path(str(audit_row["prediction_path"])).name != filename:
            raise ValueError(f"{label} filename identity mismatch for {key}")
        # One immutable byte snapshot drives both hashing and CSV parsing.
        raw = actual_files[filename].read_bytes()
        actual_sha = _sha256_bytes(raw)
        try:
            prediction = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(f"cannot parse cleaned prediction {filename}: {exc}") from exc
        if list(prediction.columns) != CLEANED_PREDICTION_COLUMNS:
            raise ValueError(f"cleaned prediction {filename} schema/order mismatch")
        if actual_sha != _norm(audit_row["prediction_sha256"]):
            raise ValueError(f"cleaned prediction {filename} SHA is stale")
        if len(raw) != _strict_int(
            audit_row["prediction_bytes"], f"{label}.prediction_bytes", positive=True
        ):
            raise ValueError(f"cleaned prediction {filename} byte count is stale")
        if len(prediction) != _strict_int(
            audit_row["rows"], f"{label}.rows", positive=True
        ):
            raise ValueError(f"cleaned prediction {filename} row count is stale")
        if len(prediction) != _strict_int(
            run_row["common_test_n"], "cleaned_runs.common_test_n", positive=True
        ):
            raise ValueError(f"cleaned prediction {filename} disagrees with run row count")

        parsed_prediction_seeds = _strict_int_values(
            prediction["seed"], f"prediction {filename}.seed", positive=True
        )
        identity_checks = {
            "task": {_norm(value) for value in prediction["task"]} == {key[0]},
            "model": {_norm(value) for value in prediction["model"]} == {key[1]},
            "seed": set(parsed_prediction_seeds) == {key[2]},
            "split_source": set(prediction["split_source"].astype(str))
            == {LOCKED_CACHE_SPLIT_SOURCE},
            "split_algorithm": set(prediction["split_algorithm"].astype(str))
            == {LOCKED_CACHE_SPLIT_ALGORITHM},
            "evaluation_set": set(prediction["evaluation_set"].astype(str))
            == {"common_cleaned_test"},
        }
        failed_identity = [name for name, passed in identity_checks.items() if not passed]
        if failed_identity:
            raise ValueError(
                f"cleaned prediction {filename} identity mismatch: {failed_identity}"
            )
        numeric = prediction[
            ["label", "official_prediction", "cleaned_prediction"]
        ].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"cleaned prediction {filename} contains non-finite values")
        smiles = prediction["canonical_smiles"].fillna("").astype(str).str.strip()
        if smiles.eq("").any() or smiles.duplicated().any():
            raise ValueError(f"cleaned prediction {filename} has invalid canonical SMILES")
        labels = numeric["label"].to_numpy(dtype=float)
        prediction_dtype = np.dtype(MODEL_PREDICTION_DTYPES[key[1]])
        official_predictions = numeric["official_prediction"].to_numpy(
            dtype=prediction_dtype
        )
        cleaned_predictions = numeric["cleaned_prediction"].to_numpy(
            dtype=prediction_dtype
        )
        if _task_type(key[0]) == "classification" and (
            (official_predictions < 0).any()
            or (official_predictions > 1).any()
            or (cleaned_predictions < 0).any()
            or (cleaned_predictions > 1).any()
        ):
            raise ValueError(f"cleaned prediction {filename} probabilities are out of range")
        identity_sha = _sample_label_identity_sha256(smiles, labels)
        if identity_sha != str(audit_row["sample_label_identity_sha256"]):
            raise ValueError(f"cleaned prediction {filename} sample identity is stale")

        metric = _task_metric(key[0])
        official_metric = _recompute_official_metric(
            metric, labels, official_predictions
        )
        cleaned_metric = _recompute_official_metric(
            metric, labels, cleaned_predictions
        )
        comparisons = (
            (
                official_metric,
                float(audit_row["recomputed_official_common_test"]),
                "audited official metric",
            ),
            (
                cleaned_metric,
                float(audit_row["recomputed_cleaned_common_test"]),
                "audited cleaned metric",
            ),
            (
                official_metric,
                float(run_row["primary_official_common_test"]),
                "run official metric",
            ),
            (
                cleaned_metric,
                float(run_row["primary_cleaned_common_test"]),
                "run cleaned metric",
            ),
        )
        for observed, expected, comparison_label in comparisons:
            if not np.isclose(
                observed, expected, rtol=METRIC_RTOL, atol=METRIC_ATOL
            ):
                raise ValueError(
                    f"cleaned prediction {filename} {comparison_label} mismatch"
                )
        raw_delta = cleaned_metric - official_metric
        oriented_delta = -raw_delta if metric == "mae" else raw_delta
        relative_delta = oriented_delta / max(abs(official_metric), 1e-12)
        for observed, expected, comparison_label in (
            (
                float(run_row["primary_raw_delta_cleaned_minus_official"]),
                raw_delta,
                "raw delta",
            ),
            (
                float(run_row["primary_oriented_delta"]),
                oriented_delta,
                "oriented delta",
            ),
            (
                float(run_row["primary_relative_oriented_delta"]),
                relative_delta,
                "relative delta",
            ),
        ):
            if not np.isclose(
                observed, expected, rtol=METRIC_RTOL, atol=METRIC_ATOL
            ):
                raise ValueError(f"cleaned prediction {filename} {comparison_label} mismatch")

        inventory_records.append(
            {
                "filename": filename,
                "task": key[0],
                "model": key[1],
                "seed": key[2],
                "sha256": actual_sha,
                "bytes": len(raw),
                "rows": len(prediction),
            }
        )
        inventory_entries[filename] = {
            "sha256": actual_sha,
            "bytes": len(raw),
            "rows": len(prediction),
        }
        total_bytes += len(raw)
        total_rows += len(prediction)

    inventory_sha = _stable_json_sha256(
        sorted(inventory_records, key=lambda record: str(record["filename"]))
    )
    if _norm(gate.get("prediction_inventory_sha256")) != inventory_sha:
        raise ValueError("cleaned_prediction_gate top-level prediction inventory is stale")

    def validate_inventory(record: object, record_label: str) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"{record_label} is missing")
        if record.get("algorithm") != "sha256-stable-json-inventory-v1":
            raise ValueError(f"{record_label} algorithm mismatch")
        if _norm(record.get("sha256")) != inventory_sha:
            raise ValueError(f"{record_label} SHA is stale")
        for field, expected in (
            ("files", expected_rows),
            ("bytes", total_bytes),
            ("rows", total_rows),
        ):
            if _strict_int(
                record.get(field), f"{record_label}.{field}", nonnegative=True
            ) != expected:
                raise ValueError(f"{record_label} {field} is stale")
        entries = record.get("entries")
        if not isinstance(entries, dict) or set(entries) != set(inventory_entries):
            raise ValueError(f"{record_label} entries are incomplete")
        for filename, expected_entry in inventory_entries.items():
            entry = entries.get(filename)
            if not isinstance(entry, dict):
                raise ValueError(f"{record_label} entry {filename} is missing")
            if _norm(entry.get("sha256")) != expected_entry["sha256"]:
                raise ValueError(f"{record_label} entry {filename} SHA is stale")
            for field in ("bytes", "rows"):
                if _strict_int(
                    entry.get(field),
                    f"{record_label}.entries.{filename}.{field}",
                    positive=True,
                ) != expected_entry[field]:
                    raise ValueError(
                        f"{record_label} entry {filename} {field} is stale"
                    )

    validate_inventory(gate.get("prediction_inventory"), "prediction_inventory")
    validate_inventory(inputs.get("predictions"), "inputs.predictions")


def _validate_final_sources(
    frames: Mapping[str, pd.DataFrame],
    payloads: Mapping[str, Mapping[str, Any]],
    source_audit: Sequence[Mapping[str, Any]],
    contract: FinalContract,
    predictions_dir: Path,
) -> dict[str, str]:
    source_hashes = {str(row["source_id"]): str(row["sha256"]) for row in source_audit}
    source_records = {str(row["source_id"]): row for row in source_audit}
    traditional_winners = _validate_traditional_selection(
        frames["traditional_selection"], contract
    )
    _validate_reliability(frames, payloads["reliability_gate"], contract)
    _validate_conditional(frames, payloads["conditional_gate"], contract)
    _validate_calibration(frames, contract)
    _validate_ablation(frames, payloads["ablation_gate"], contract)
    _validate_y_randomization(frames, traditional_winners, contract)
    _validate_shap(frames, traditional_winners, contract)
    _validate_cpu_defense_gate(payloads["cpu_defense_gate"], contract, source_hashes)
    _validate_cleaned_sensitivity(
        frames,
        payloads["cleaned_manifest"],
        payloads["cleaned_config"],
        payloads["cleaned_gate"],
        traditional_winners,
        contract,
        source_records,
    )
    _validate_cleaned_prediction_audit(
        frames["cleaned_prediction_audit"],
        payloads["cleaned_prediction_gate"],
        source_records,
        frames["cleaned_runs"],
        predictions_dir,
        contract,
    )
    return traditional_winners


def _calibration_effects(summary: pd.DataFrame) -> pd.DataFrame:
    classification = summary[summary["task_type"].map(_norm) == "classification"].copy()
    rows: list[dict[str, object]] = []
    for method, group in classification.groupby("method", sort=True):
        ece_gain = group["test_ece_before"].astype(float) - group["test_ece_after"].astype(float)
        brier_gain = group["test_brier_before"].astype(float) - group["test_brier_after"].astype(float)
        rows.append(
            {
                "method": method,
                "n_rows": len(group),
                "n_tasks": group["task"].map(_norm).nunique(),
                "n_models": group["model"].map(_norm).nunique(),
                "mean_ece_before": float(group["test_ece_before"].mean()),
                "mean_ece_after": float(group["test_ece_after"].mean()),
                "mean_ece_gain": float(ece_gain.mean()),
                "ece_improved_count": int((ece_gain > 0).sum()),
                "ece_improved_fraction": float((ece_gain > 0).mean()),
                "mean_brier_before": float(group["test_brier_before"].mean()),
                "mean_brier_after": float(group["test_brier_after"].mean()),
                "mean_brier_gain": float(brier_gain.mean()),
                "brier_improved_count": int((brier_gain > 0).sum()),
                "brier_improved_fraction": float((brier_gain > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _conformal_by_task_model(summary: pd.DataFrame) -> pd.DataFrame:
    regression = summary[summary["task_type"].map(_norm) == "regression"].copy()
    return (
        regression.groupby(["task", "model", "official_metric"], sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_interval_width=("mean_interval_width", "mean"),
            std_interval_width=("mean_interval_width", "std"),
            mean_mae=("test_mae", "mean"),
            mean_rmse=("test_rmse", "mean"),
            mean_spearman=("test_spearman", "mean"),
        )
        .reset_index()
    )


def _official_test_metric_matrix(
    frames: Mapping[str, pd.DataFrame], contract: FinalContract
) -> pd.DataFrame:
    """Build the reader-visible test matrix without selecting on test results."""

    tasks = {_norm(task) for task in contract.tasks}
    models = {_norm(model) for model in contract.models}
    seeds = set(contract.seeds)
    expected_keys = set(itertools.product(tasks, models))
    selection = frames["validation_selection"]
    selected_by_task = {
        _norm(row.task): _norm(row.validation_selected_model)
        for row in selection.itertuples(index=False)
    }
    if set(selected_by_task) != tasks or any(
        model not in models for model in selected_by_task.values()
    ):
        raise ValueError("Official test matrix has an invalid validation-selection map")

    profiles = frames["conditional_profiles"]
    profile_lookup = {
        (_norm(row.task), _norm(row.model)): row
        for row in profiles.itertuples(index=False)
    }
    if set(profile_lookup) != expected_keys:
        raise ValueError("Official test matrix conditional-profile keys are incomplete")

    reliability = frames["test_reliability"]
    required_columns = (
        "task",
        "model",
        "task_type",
        "official_metric",
        "n_seeds",
        "test_n",
        "test_official_metric_raw",
        "test_official_score_oriented",
    )
    _require_columns(reliability, required_columns, "official_test_metric_source")
    rows: list[dict[str, object]] = []
    observed_keys: set[tuple[str, str]] = set()
    for (task_value, model_value), group in reliability.groupby(
        [reliability["task"].map(_norm), reliability["model"].map(_norm)],
        sort=False,
    ):
        key = (_norm(task_value), _norm(model_value))
        if key in observed_keys:
            raise ValueError(f"Official test metric source duplicates {key}")
        observed_keys.add(key)
        task, model = key
        if key not in expected_keys:
            raise ValueError(f"Official test metric source has unexpected key {key}")
        task_types = {_norm(value) for value in group["task_type"]}
        metrics = {_norm(value) for value in group["official_metric"]}
        if task_types != {_task_type(task)} or metrics != {_task_metric(task)}:
            raise ValueError(f"Official test metric metadata mismatch for {key}")
        n_seeds_values = set(
            _strict_int_values(
                group["n_seeds"], f"official_test_metric_source.{key}.n_seeds", positive=True
            )
        )
        test_n_values = set(
            _strict_int_values(
                group["test_n"], f"official_test_metric_source.{key}.test_n", positive=True
            )
        )
        if n_seeds_values != {len(seeds)} or len(test_n_values) != 1:
            raise ValueError(f"Official test metric seed/sample counts mismatch for {key}")
        raw_values = pd.to_numeric(
            group["test_official_metric_raw"], errors="coerce"
        ).to_numpy(dtype=float)
        oriented_values = pd.to_numeric(
            group["test_official_score_oriented"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(raw_values).all() or not np.isfinite(oriented_values).all():
            raise ValueError(f"Official test metric is non-finite for {key}")
        if not np.allclose(raw_values, raw_values[0], rtol=1e-12, atol=1e-12) or not np.allclose(
            oriented_values, oriented_values[0], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                f"Official test metric is inconsistent across reliability dimensions for {key}"
            )
        test_score = float(raw_values[0])
        oriented_score = float(oriented_values[0])
        expected_oriented = oriented_metric(_task_metric(task), test_score)
        if not np.isclose(
            oriented_score, expected_oriented, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"Official test metric orientation mismatch for {key}")

        profile = profile_lookup[key]
        profile_n_seeds = _strict_int(
            profile.n_seeds, f"conditional_profiles.{key}.n_seeds", positive=True
        )
        profile_n_samples = _strict_int(
            profile.n_samples, f"conditional_profiles.{key}.n_samples", positive=True
        )
        profile_performance = float(profile.performance)
        if not math.isfinite(profile_performance) or not np.isclose(
            profile_performance, oriented_score, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"Official test metric does not reproduce conditional profile for {key}"
            )
        if profile_n_seeds != len(seeds) or profile_n_samples != next(iter(test_n_values)):
            raise ValueError(
                f"Official test metric counts do not reproduce conditional profile for {key}"
            )
        rows.append(
            {
                "task": _canonical_task_name(task),
                "category": _task_category(task),
                "task_type": _task_type(task),
                "official_metric": _task_metric(task),
                "model": model,
                "n_seeds": len(seeds),
                "test_n": next(iter(test_n_values)),
                "test_score": test_score,
                "validation_selected": model == selected_by_task[task],
            }
        )
    if observed_keys != expected_keys:
        raise ValueError(
            "Official test metric source key matrix mismatch: "
            f"missing={sorted(expected_keys - observed_keys)[:5]}, "
            f"unexpected={sorted(observed_keys - expected_keys)[:5]}"
        )
    matrix = pd.DataFrame(
        rows,
        columns=[
            "task",
            "category",
            "task_type",
            "official_metric",
            "model",
            "n_seeds",
            "test_n",
            "test_score",
            "validation_selected",
        ],
    )
    _require_exact_keys(
        matrix,
        ["task", "model"],
        itertools.product(tasks, models),
        "official_test_metric_matrix",
    )
    _require_finite(matrix, ["n_seeds", "test_n", "test_score"], "official_test_metric_matrix")
    _strict_int_values(matrix["n_seeds"], "official_test_metric_matrix.n_seeds", positive=True)
    _strict_int_values(matrix["test_n"], "official_test_metric_matrix.test_n", positive=True)
    selected = _boolean_series(
        matrix["validation_selected"], "official_test_metric_matrix.validation_selected"
    )
    selected_counts = selected.groupby(matrix["task"].map(_norm)).sum()
    if set(selected_counts.index) != tasks or not (selected_counts == 1).all():
        raise ValueError("Official test metric matrix must mark one validation-selected model per task")
    task_order = {_norm(task): index for index, task in enumerate(contract.tasks)}
    model_order = {_norm(model): index for index, model in enumerate(contract.models)}
    matrix["_task_order"] = matrix["task"].map(lambda value: task_order[_norm(value)])
    matrix["_model_order"] = matrix["model"].map(lambda value: model_order[_norm(value)])
    matrix = (
        matrix.sort_values(["_task_order", "_model_order"], kind="stable")
        .drop(columns=["_task_order", "_model_order"])
        .reset_index(drop=True)
    )
    return matrix


def _official_test_metric_latex(frame: pd.DataFrame) -> bytes:
    def escape(value: object) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(character, character) for character in str(value))

    headers = [escape(column) for column in frame.columns]
    lines = [
        r"\begin{longtable}{lllllrrrl}",
        (
            r"\caption{Official test metrics for every task--model pair. Scores retain "
            r"each endpoint's original metric direction; model-selection flags are based "
            r"only on validation data.}\label{tab:official-test-metrics-all-models}\\"
        ),
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in frame.itertuples(index=False):
        values = [
            escape(row.task),
            escape(row.category),
            escape(row.task_type),
            escape(row.official_metric),
            escape(row.model),
            str(_strict_int(row.n_seeds, "LaTeX n_seeds", positive=True)),
            str(_strict_int(row.test_n, "LaTeX test_n", positive=True)),
            f"{float(row.test_score):.8g}",
            "Yes" if bool(row.validation_selected) else "No",
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{longtable}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _final_output_frames(
    frames: Mapping[str, pd.DataFrame], contract: FinalContract
) -> tuple[
    dict[str, pd.DataFrame], dict[str, tuple[str, ...]]
]:
    official_test_metrics = _official_test_metric_matrix(frames, contract)
    outputs: dict[str, pd.DataFrame] = {
        "table_main_validation_selection_by_task.csv": frames["validation_selection"].copy(),
        "table_main_reliability_regret_summary.csv": frames["reliability_summary"].copy(),
        "table_main_conditional_reliability_summary.csv": frames["conditional_summary"].copy(),
        "table_main_calibration_effects.csv": _calibration_effects(frames["calibration_summary"]),
        "table_main_cleaned_sensitivity_by_task.csv": frames["cleaned_task"].copy(),
        "table_supp_traditional_best_by_task.csv": frames["traditional_selection"].copy(),
        "table_supp_reliability_regret_by_endpoint.csv": frames["reliability_endpoint"].copy(),
        "table_supp_test_reliability_by_task_model.csv": frames["test_reliability"].copy(),
        "table_supp_model_seed_availability.csv": frames["model_availability"].copy(),
        "table_supp_model_reliability_profiles.csv": frames["conditional_profiles"].copy(),
        "table_supp_conditional_reliability_by_task_model.csv": frames[
            "conditional_endpoint_model"
        ].copy(),
        "table_supp_calibration_seed_level.csv": frames["calibration_summary"].copy(),
        "table_supp_calibration_reliability_bins.csv": frames["calibration_bins"].copy(),
        "table_supp_conformal_by_task_model.csv": _conformal_by_task_model(
            frames["calibration_summary"]
        ),
        "table_supp_ensemble_size_by_seed_subset.csv": frames[
            "ablation_ensemble_rows"
        ].copy(),
        "table_supp_ensemble_size_summary.csv": frames["ablation_ensemble_summary"].copy(),
        "table_supp_calibration_binning_sensitivity.csv": frames[
            "ablation_calibration_bins"
        ].copy(),
        "table_supp_ablation_sample_alignment.csv": frames["ablation_alignment"].copy(),
        "table_supp_y_randomization_all_tasks.csv": frames["y_random_summary"].copy(),
        "table_supp_y_randomization_scores.csv": frames["y_random_scores"].copy(),
        "table_supp_shap_selected.csv": frames["shap_selected"].copy(),
        "table_supp_shap_bit_substructures.csv": frames["shap_mapping"].copy(),
        "table_supp_cleaned_sensitivity_by_task_model.csv": frames[
            "cleaned_task_model"
        ].copy(),
        "table_supp_cleaned_sensitivity_runs.csv": frames["cleaned_runs"].copy(),
        "table_supp_cleaned_sensitivity_audit.csv": frames["cleaned_audit"].copy(),
        "table_supp_cleaned_sensitivity_prediction_audit.csv": frames[
            "cleaned_prediction_audit"
        ].copy(),
        "table_supp_official_test_metrics_all_models.csv": official_test_metrics.copy(),
        "table_supp_official_test_metrics_all_models.tex": official_test_metrics.copy(),
    }
    lineage: dict[str, tuple[str, ...]] = {
        "table_main_validation_selection_by_task.csv": ("validation_selection",),
        "table_main_reliability_regret_summary.csv": (
            "reliability_summary",
            "reliability_gate",
        ),
        "table_main_conditional_reliability_summary.csv": (
            "conditional_summary",
            "conditional_gate",
        ),
        "table_main_calibration_effects.csv": ("calibration_summary",),
        "table_main_cleaned_sensitivity_by_task.csv": (
            "cleaned_task",
            "cleaned_gate",
            "cleaned_manifest",
        ),
        "table_supp_traditional_best_by_task.csv": ("traditional_selection",),
        "table_supp_reliability_regret_by_endpoint.csv": ("reliability_endpoint",),
        "table_supp_test_reliability_by_task_model.csv": ("test_reliability",),
        "table_supp_model_seed_availability.csv": ("model_availability",),
        "table_supp_model_reliability_profiles.csv": ("conditional_profiles",),
        "table_supp_conditional_reliability_by_task_model.csv": (
            "conditional_endpoint_model",
        ),
        "table_supp_calibration_seed_level.csv": ("calibration_summary",),
        "table_supp_calibration_reliability_bins.csv": ("calibration_bins",),
        "table_supp_conformal_by_task_model.csv": ("calibration_summary",),
        "table_supp_ensemble_size_by_seed_subset.csv": ("ablation_ensemble_rows",),
        "table_supp_ensemble_size_summary.csv": ("ablation_ensemble_summary",),
        "table_supp_calibration_binning_sensitivity.csv": (
            "ablation_calibration_bins",
        ),
        "table_supp_ablation_sample_alignment.csv": ("ablation_alignment",),
        "table_supp_y_randomization_all_tasks.csv": (
            "y_random_summary",
            "traditional_selection",
            "cpu_defense_gate",
        ),
        "table_supp_y_randomization_scores.csv": ("y_random_scores",),
        "table_supp_shap_selected.csv": (
            "shap_selected",
            "traditional_selection",
            "cpu_defense_gate",
        ),
        "table_supp_shap_bit_substructures.csv": ("shap_mapping", "shap_selected"),
        "table_supp_cleaned_sensitivity_by_task_model.csv": (
            "cleaned_task_model",
            "cleaned_gate",
        ),
        "table_supp_cleaned_sensitivity_runs.csv": (
            "cleaned_runs",
            "cleaned_manifest",
        ),
        "table_supp_cleaned_sensitivity_audit.csv": ("cleaned_audit",),
        "table_supp_cleaned_sensitivity_prediction_audit.csv": (
            "cleaned_prediction_audit",
            "cleaned_prediction_gate",
        ),
        "table_supp_official_test_metrics_all_models.csv": (
            "test_reliability",
            "conditional_profiles",
            "validation_selection",
            "reliability_gate",
            "conditional_gate",
        ),
        "table_supp_official_test_metrics_all_models.tex": (
            "test_reliability",
            "conditional_profiles",
            "validation_selection",
            "reliability_gate",
            "conditional_gate",
        ),
    }
    return outputs, lineage


def _write_final_staging(
    staging: Path,
    outputs: Mapping[str, pd.DataFrame],
    lineage: Mapping[str, tuple[str, ...]],
    source_audit: list[dict[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    contract: FinalContract,
) -> pd.DataFrame:
    staging.mkdir(parents=True, exist_ok=False)
    source_path_by_id = {row["source_id"]: row["path"] for row in source_audit}
    inventory_rows: list[dict[str, object]] = []
    for filename, frame in outputs.items():
        path = staging / filename
        if path.suffix == ".tex":
            if filename != "table_supp_official_test_metrics_all_models.tex":
                raise ValueError(f"No locked LaTeX serializer is defined for {filename}")
            path.write_bytes(_official_test_metric_latex(frame))
        else:
            frame.to_csv(path, index=False)
        source_ids = lineage[filename]
        inventory_rows.append(
            {
                "table": path.stem,
                "filename": filename,
                "role": "main" if filename.startswith("table_main_") else "supplementary",
                "format": path.suffix.removeprefix("."),
                "rows": len(frame),
                "columns": len(frame.columns),
                "source_count": len(source_ids),
                "source_ids": ";".join(source_ids),
                "source_paths": ";".join(source_path_by_id[value] for value in source_ids),
                "sha256": _sha256_path(path),
                "protocol_scope": "final_locked_five_model",
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(staging / "table_inventory.csv", index=False)

    official_filenames = (
        "table_supp_official_test_metrics_all_models.csv",
        "table_supp_official_test_metrics_all_models.tex",
    )
    official_output_records = {
        row["format"]: {
            "filename": row["filename"],
            "rows": row["rows"],
            "columns": row["columns"],
            "sha256": row["sha256"],
        }
        for row in inventory_rows
        if row["filename"] in official_filenames
    }
    expected_official_rows = len(contract.tasks) * len(contract.models)
    if set(official_output_records) != {"csv", "tex"} or any(
        _strict_int(record["rows"], "official output rows", positive=True)
        != expected_official_rows
        for record in official_output_records.values()
    ):
        raise ValueError("Reader-visible official test metric outputs failed row/format gate")
    official_source_ids = (
        "test_reliability",
        "conditional_profiles",
        "validation_selection",
        "reliability_gate",
        "conditional_gate",
    )
    source_record_by_id = {row["source_id"]: row for row in source_audit}
    official_test_metric_gate = {
        "status": "PASS",
        "selection_basis": "validation_only",
        "metric_direction": "original_official_metric_direction",
        "expected_rows": expected_official_rows,
        "observed_rows": expected_official_rows,
        "expected_tasks": len(contract.tasks),
        "expected_models": len(contract.models),
        "exact_task_model_key_matrix": True,
        "finite_test_scores": True,
        "cross_source_recomputation_match": True,
        "source_inputs": {
            source_id: {
                "path": source_record_by_id[source_id]["path"],
                "sha256": source_record_by_id[source_id]["sha256"],
                "bytes": source_record_by_id[source_id]["bytes"],
                "rows": source_record_by_id[source_id]["rows"],
            }
            for source_id in official_source_ids
        },
        "outputs": official_output_records,
    }

    for row in source_audit:
        row["validation_status"] = "PASS"
        row["notes"] = "fixed required source; final contract validated; no glob or fallback"
    pd.DataFrame(source_audit).to_csv(staging / "source_audit.csv", index=False)
    audit_payload = {
        "schema": "manuscript_table_source_audit_v1",
        "mode": "final_locked",
        "status": "PASS",
        "protocol_scope": "final_locked_five_model",
        "contract": {
            "tasks": list(contract.tasks),
            "models": list(contract.models),
            "seeds": list(contract.seeds),
            "conditional_thresholds": list(contract.conditional_thresholds),
            "ensemble_sizes": list(contract.ensemble_sizes),
            "ablation_bin_counts": list(contract.ablation_bin_counts),
            "shap_tasks": list(contract.shap_tasks),
            "cleaned_tasks": list(contract.cleaned_tasks),
            "cleaned_models": list(contract.cleaned_models),
        },
        "scientific_gate_statuses": {
            "reliability": payloads["reliability_gate"].get("status"),
            "conditional": payloads["conditional_gate"].get("status"),
            "ablation": payloads["ablation_gate"].get("status"),
            "cpu_defense": payloads["cpu_defense_gate"].get("status"),
            "cleaned_sensitivity": payloads["cleaned_gate"].get("status"),
            "cleaned_prediction_exports": payloads["cleaned_prediction_gate"].get(
                "status"
            ),
        },
        "reader_visible_table_gates": {
            "official_test_metrics_all_models": official_test_metric_gate,
        },
        "source_files": source_audit,
        "output_tables": inventory_rows,
        "interpretation": (
            "PASS means the locked source and coverage contract passed. Scientific GO/HOLD or "
            "sensitivity status values are preserved verbatim and are not overwritten by this audit."
        ),
    }
    (staging / "source_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return inventory


def _commit_directory(staging: Path, out_dir: Path) -> None:
    backup = out_dir.parent / f".{out_dir.name}.backup.{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    had_existing = out_dir.exists()
    if had_existing:
        out_dir.rename(backup)
    try:
        staging.rename(out_dir)
    except Exception:
        if had_existing and backup.exists() and not out_dir.exists():
            backup.rename(out_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_final_locked(
    tables_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    contract: FinalContract = DEFAULT_FINAL_CONTRACT,
) -> pd.DataFrame:
    """Validate and build the final locked table inventory.

    No output path is created or replaced until every required source passes.
    """

    _validate_contract(contract)
    tables_path = Path(tables_dir)
    output_path = Path(out_dir) if out_dir is not None else tables_path / "manuscript_final"
    if "local_prelim" in str(output_path.resolve()).lower():
        raise ValueError("final_locked output path must not contain local_prelim")
    frames, payloads, source_audit = _load_final_sources(tables_path)
    _validate_final_sources(
        frames,
        payloads,
        source_audit,
        contract,
        tables_path.parent / "predictions" / "cleaned_sensitivity_locked_cache",
    )
    outputs, lineage = _final_output_frames(frames, contract)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        inventory = _write_final_staging(
            staging,
            outputs,
            lineage,
            source_audit,
            payloads,
            contract,
        )
        _commit_directory(staging, output_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return inventory


def build_local_preview(
    tables_dir: str | Path, out_dir: str | Path | None = None
) -> pd.DataFrame:
    """Historical local-preliminary preview path, retained without final claims."""

    tables_path = Path(tables_dir)
    output_path = Path(out_dir) if out_dir is not None else tables_path / "manuscript"
    output_path.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(tables_path / "traditional_grid_local_prelim_best_by_task.csv")
    best_out = best[
        [
            "task",
            "task_type",
            "official_metric",
            "model",
            "test_official_mean",
            "test_official_std",
            "valid_official_mean",
            "valid_official_std",
        ]
    ].copy()
    best_out.to_csv(output_path / "table_main_traditional_best_by_task.csv", index=False)

    audit = pd.read_csv(tables_path / "data_audit_all_tasks.csv")
    audit_summary = (
        audit.groupby("task")
        .agg(
            total_rows=("total", "sum"),
            invalid_smiles=("invalid", "sum"),
            duplicates=("duplicates", "sum"),
            salts_or_mixtures=("salts_or_mixtures", "sum"),
            train_valid_test_molecule_overlap=("train_valid_test_molecule_overlap", "max"),
            train_valid_test_scaffold_overlap=("train_valid_test_scaffold_overlap", "max"),
        )
        .reset_index()
    )
    audit_summary.to_csv(output_path / "table_supp_data_audit_by_task.csv", index=False)

    calibration = pd.read_csv(tables_path / "calibration_summary_local_prelim.csv")
    classification = calibration[calibration["task_type"] == "classification"].copy()
    cal_summary = (
        classification.groupby("method")
        .agg(
            rows=("task", "size"),
            mean_ece_before=("test_ece_before", "mean"),
            mean_ece_after=("test_ece_after", "mean"),
            mean_brier_before=("test_brier_before", "mean"),
            mean_brier_after=("test_brier_after", "mean"),
        )
        .reset_index()
    )
    improved = {
        method: (
            int((group["test_ece_after"] < group["test_ece_before"]).sum()),
            int((group["test_brier_after"] < group["test_brier_before"]).sum()),
        )
        for method, group in classification.groupby("method", sort=True)
    }
    cal_summary["ece_improved_count"] = cal_summary["method"].map(
        lambda value: improved[value][0]
    )
    cal_summary["brier_improved_count"] = cal_summary["method"].map(
        lambda value: improved[value][1]
    )
    cal_summary.to_csv(output_path / "table_supp_calibration_summary.csv", index=False)

    regression = calibration[calibration["task_type"] == "regression"].copy()
    reg_summary = pd.DataFrame(
        [
            {
                "rows": len(regression),
                "mean_coverage": regression["coverage"].mean(),
                "std_coverage": regression["coverage"].std(),
                "min_coverage": regression["coverage"].min(),
                "max_coverage": regression["coverage"].max(),
                "median_interval_width": regression["mean_interval_width"].median(),
            }
        ]
    )
    reg_summary.to_csv(output_path / "table_supp_conformal_summary.csv", index=False)

    uncertainty = pd.read_csv(tables_path / "uncertainty_rejection_local_prelim.csv")
    uncertainty["oriented"] = uncertainty.apply(
        lambda row: oriented_metric(row["official_metric"], row["metric"]), axis=1
    )
    uncertainty_pivot = (
        uncertainty.pivot_table(
            index=["task", "model", "task_type", "official_metric"],
            columns="rejection_frac",
            values="oriented",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    uncertainty_pivot["delta_30_vs_0"] = uncertainty_pivot[0.3] - uncertainty_pivot[0.0]
    uncertainty_pivot.to_csv(
        output_path / "table_supp_uncertainty_rejection_delta.csv", index=False
    )

    applicability = pd.read_csv(tables_path / "applicability_domain_all_seeds_local_prelim.csv")
    applicability["oriented"] = applicability.apply(
        lambda row: oriented_metric(row["official_metric"], row["metric"]), axis=1
    )
    applicability_rows = []
    for key, group in applicability.dropna(subset=["oriented", "mean_similarity"]).groupby(
        ["task", "model", "task_type", "official_metric", "seed"]
    ):
        if len(group) < 3 or group["oriented"].nunique() <= 1:
            continue
        rho = spearmanr(group["mean_similarity"], group["oriented"]).statistic
        applicability_rows.append((*key, rho, len(group)))
    applicability_summary = pd.DataFrame(
        applicability_rows,
        columns=[
            "task",
            "model",
            "task_type",
            "official_metric",
            "seed",
            "similarity_performance_spearman",
            "n_bins",
        ],
    )
    applicability_summary.to_csv(
        output_path / "table_supp_applicability_domain_seed_correlations.csv", index=False
    )

    y_random = pd.read_csv(tables_path / "y_randomization_all_tasks_local_prelim.csv")
    y_random.to_csv(output_path / "table_supp_y_randomization_all_tasks.csv", index=False)
    shap_mapping = pd.read_csv(tables_path / "shap_morgan_bit_substructures_local_prelim.csv")
    shap_mapping.to_csv(output_path / "table_supp_shap_bit_substructures.csv", index=False)

    inventory = pd.DataFrame(
        [
            {"table": "main_traditional_best_by_task", "rows": len(best_out)},
            {"table": "supp_data_audit_by_task", "rows": len(audit_summary)},
            {"table": "supp_calibration_summary", "rows": len(cal_summary)},
            {"table": "supp_conformal_summary", "rows": len(reg_summary)},
            {"table": "supp_uncertainty_rejection_delta", "rows": len(uncertainty_pivot)},
            {
                "table": "supp_applicability_domain_seed_correlations",
                "rows": len(applicability_summary),
            },
            {"table": "supp_y_randomization_all_tasks", "rows": len(y_random)},
            {"table": "supp_shap_bit_substructures", "rows": len(shap_mapping)},
        ]
    )
    inventory.to_csv(output_path / "table_inventory.csv", index=False)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create manuscript-ready tables with explicit preliminary/final isolation."
    )
    parser.add_argument(
        "--mode",
        choices=("local_prelim", "final_locked"),
        default="local_prelim",
        help="Historical preview or strict locked final source contract.",
    )
    parser.add_argument("--tables-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to manuscript for local_prelim and manuscript_final for final_locked.",
    )
    args = parser.parse_args()
    if args.mode == "final_locked":
        inventory = build_final_locked(args.tables_dir, args.out_dir)
        destination = Path(args.out_dir) if args.out_dir else Path(args.tables_dir) / "manuscript_final"
    else:
        inventory = build_local_preview(args.tables_dir, args.out_dir)
        destination = Path(args.out_dir) if args.out_dir else Path(args.tables_dir) / "manuscript"
    print(f"wrote manuscript tables to {destination}")
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
