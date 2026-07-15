"""Validation-selected reliability-regret analysis for the ADMET benchmark.

This module deliberately separates model selection from reliability evaluation:

* models are selected using the mean official validation metric across complete
  seeds (the validation molecules differ by seed and therefore must not be
  naively ensembled by row number);
* test predictions are ensembled across seeds only after selection;
* a test-set reliability oracle is used solely as a descriptive counterfactual,
  never to select the deployed model;
* descriptive aggregation uses endpoints as comparison units. Task-model rows
  remain in the outputs for auditability but are never treated as independent
  replicates; cross-endpoint molecule overlap precludes independent-assay
  inference.

For selective prediction, AUROC and AUPRC are intentionally not placed on a
sample-wise risk-coverage curve: they are ranking/set-level metrics without an
additive per-sample loss.  Classification curves use Brier score and log loss;
regression curves use absolute error.  The script writes every empirical
coverage point and reports AURC and excess AURC relative to an oracle ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASK_META,
)


TASK_META_BY_SLUG = {name.lower(): meta for name, meta in TDC_TASK_META.items()}
EXPECTED_TASKS = tuple(sorted(TASK_META_BY_SLUG))
FINAL_EXPECTED_MODELS = (
    "rf",
    "xgboost",
    "lightgbm",
    "gin",
    "chemprop_dmpnn",
)
FINAL_EXPECTED_SEEDS = (1, 2, 3, 4, 5)
LEGACY_EXCLUDED_MODELS = frozenset({"legacy_node_mpnn"})

PRACTICAL_TOLERANCES = {
    "ece": 0.010,
    "brier": 0.005,
    "conformal_coverage_error": 0.010,
    "applicability_alignment": 0.050,
    "selective_relative_excess_aurc": 0.010,
    "selective_relative_excess_aurc_log_loss": 0.010,
}


@dataclass
class RunPrediction:
    task: str
    model: str
    seed: int
    task_type: str
    official_metric: str
    valid: pd.DataFrame
    test: pd.DataFrame
    source: str


@dataclass
class ModelEnsemble:
    task: str
    model: str
    task_type: str
    official_metric: str
    seeds: tuple[int, ...]
    validation_scores: dict[int, float]
    test: pd.DataFrame
    conformal_coverage: float


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(float)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return float("nan")
    left_rank = rank_values(frame["left"].to_numpy())
    right_rank = rank_values(frame["right"].to_numpy())
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positive = y_true == 1
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rank_values(y_score)
    rank_sum = float(np.sum(ranks[positive]))
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision with tied predictions evaluated at one threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_positive = int(np.sum(y_true == 1))
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ordered_labels = y_true[order]
    ordered_scores = y_score[order]
    cumulative_positive = np.cumsum(ordered_labels == 1)
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0), len(ordered_scores) - 1
    ]
    true_positive = cumulative_positive[threshold_ends]
    predicted_positive = threshold_ends + 1
    precision = true_positive / predicted_positive
    previous_true_positive = np.r_[0, true_positive[:-1]]
    recall_increment = (true_positive - previous_true_positive) / n_positive
    return float(np.sum(recall_increment * precision))


def official_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = str(metric).lower()
    if metric == "mae":
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    if metric == "spearman":
        return spearman(y_true, y_pred)
    if metric == "auroc":
        return auroc(y_true, y_pred)
    if metric == "auprc":
        return auprc(y_true, y_pred)
    raise ValueError(f"Unsupported official metric: {metric}")


def orient_metric(metric: str, value: float) -> float:
    """Orient an official score so that larger is always better."""
    return -float(value) if str(metric).lower() == "mae" else float(value)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask]))
            )
    return float(result)


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n == 0:
        return float("nan")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    try:
        return float(np.quantile(residuals, level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(residuals, level, interpolation="higher"))


def risk_coverage_curve(
    losses: np.ndarray,
    uncertainty: np.ndarray,
    sample_index: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return every empirical risk-coverage point and its oracle counterpart.

    At intermediate coverage points inside an uncertainty tie, risk is the
    expectation under a random within-tie order.  It is therefore invariant to
    CSV row order and never uses observed loss to resolve uncertainty ties. The
    oracle curve sorts by observed loss solely to define excess AURC.
    """
    losses = np.asarray(losses, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if sample_index is None:
        sample_index = np.arange(len(losses))
    sample_index = np.asarray(sample_index)
    valid = np.isfinite(losses) & np.isfinite(uncertainty)
    losses = losses[valid]
    uncertainty = uncertainty[valid]
    sample_index = sample_index[valid]
    if len(losses) == 0:
        return pd.DataFrame(
            columns=[
                "retained_count",
                "coverage",
                "uncertainty_threshold",
                "cumulative_risk",
                "oracle_cumulative_risk",
            ]
        )
    order = np.lexsort((sample_index, uncertainty))
    oracle_order = np.lexsort((sample_index, losses))
    retained = np.arange(1, len(losses) + 1)
    # Within an uncertainty tie there is no defensible ordering.  Use the
    # expected cumulative loss under a random within-tie order, which is
    # invariant to CSV row order and cannot peek at the observed losses.
    ordered_uncertainty = uncertainty[order]
    ordered_losses = losses[order]
    expected_cumulative_loss = np.empty(len(losses), dtype=float)
    cumulative_before = 0.0
    start = 0
    while start < len(losses):
        end = start + 1
        while (
            end < len(losses)
            and ordered_uncertainty[end] == ordered_uncertainty[start]
        ):
            end += 1
        tie_mean = float(np.mean(ordered_losses[start:end]))
        within = np.arange(1, end - start + 1, dtype=float)
        expected_cumulative_loss[start:end] = cumulative_before + within * tie_mean
        cumulative_before += float(np.sum(ordered_losses[start:end]))
        start = end
    cumulative_risk = expected_cumulative_loss / retained
    oracle_risk = np.cumsum(losses[oracle_order]) / retained
    return pd.DataFrame(
        {
            "retained_count": retained,
            "coverage": retained / len(losses),
            "uncertainty_threshold": ordered_uncertainty,
            "cumulative_risk": cumulative_risk,
            "oracle_cumulative_risk": oracle_risk,
        }
    )


def aurc_from_curve(curve: pd.DataFrame) -> tuple[float, float, float]:
    if curve.empty:
        return float("nan"), float("nan"), float("nan")
    aurc = float(curve["cumulative_risk"].mean())
    oracle = float(curve["oracle_cumulative_risk"].mean())
    # Numerical roundoff is the only legitimate source of a negative excess.
    excess = max(0.0, aurc - oracle)
    return aurc, oracle, excess


def bootstrap_interval(
    values: np.ndarray, reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        draws[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_flip_pvalue(values: np.ndarray, reps: int, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    signs = rng.choice((-1.0, 1.0), size=(reps, len(values)))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((1 + np.sum(null >= observed)) / (reps + 1))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().astype(float).sort_values()
    running_max = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        adjusted = min(1.0, (total - rank) * value)
        running_max = max(running_max, adjusted)
        result.loc[index] = running_max
    return result


def _prediction_files(directories: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*.csv"):
            # Ignore macOS AppleDouble files and interrupted hidden temporary
            # writes. Formal artifacts never begin with a dot.
            if path.name.startswith("."):
                continue
            paths.append(path)
    return sorted(set(paths))


def _finite_numeric(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(len(values) and np.isfinite(values).all())


def _validate_normalized_rows(
    frame: pd.DataFrame, task: str, model: str
) -> str | None:
    """Validate the row-level contract shared by every prediction source."""
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    if set(frame["split"]) != {"valid", "test"}:
        return "invalid_or_missing_prediction_split"
    if frame["smiles"].isna().any() or frame["smiles"].astype(str).str.strip().eq("").any():
        return "empty_prediction_smiles"
    if not _finite_numeric(frame["y_true"]):
        return "nonfinite_prediction_label"
    if not _finite_numeric(frame["y_pred"]):
        return "nonfinite_prediction_value"
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(float)
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise").astype(float)
    try:
        sample_index = pd.to_numeric(frame["sample_index"], errors="raise")
    except (TypeError, ValueError):
        return "invalid_sample_index"
    if (sample_index < 0).any() or not np.equal(sample_index, np.floor(sample_index)).all():
        return "invalid_sample_index"
    frame["sample_index"] = sample_index.astype(int)
    if frame.duplicated(["split", "sample_index"]).any():
        return "duplicate_sample_index_within_split"

    meta = TASK_META_BY_SLUG[task]
    if meta["task_type"] == "classification":
        labels = set(frame["y_true"].unique())
        if not labels.issubset({0.0, 1.0}):
            return "nonbinary_classification_label"
        if not frame["y_pred"].between(0.0, 1.0, inclusive="both").all():
            return "classification_prediction_outside_unit_interval"
    frame["task"] = task
    frame["model"] = model
    frame["task_type"] = meta["task_type"]
    frame["official_metric"] = meta["metric"]
    return None


def sample_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered sample identity, never labels or predictions.

    The hash is deliberately sensitive to row order, split-local index, and
    SMILES. Labels are compared separately with a small numeric tolerance
    because the legacy GIN writer serializes batch labels after a float32 round
    trip while the other model writers preserve float64 source labels.
    """
    required = {"sample_index", "smiles"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Cannot fingerprint rows without {sorted(required)}")
    ordered = frame.sort_values("sample_index", kind="mergesort")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False):
        record = [
            int(getattr(row, "sample_index")),
            str(getattr(row, "smiles")),
        ]
        digest.update(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def sample_rows_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare sample identity exactly and labels within serialization error."""
    if len(left) != len(right) or sample_fingerprint(left) != sample_fingerprint(right):
        return False
    left_ordered = left.sort_values("sample_index", kind="mergesort")
    right_ordered = right.sort_values("sample_index", kind="mergesort")
    return bool(
        np.allclose(
            left_ordered["y_true"].to_numpy(dtype=float),
            right_ordered["y_true"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False,
        )
    )


def _normalize_prediction_file(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Normalize one file under a strict, model-aware artifact contract."""
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return None, f"unreadable:{type(exc).__name__}"
    standard = {
        "task",
        "model",
        "seed",
        "split",
        "sample_index",
        "smiles",
        "y_true",
        "y_pred",
        "task_type",
        "official_metric",
    }
    if standard.issubset(frame.columns):
        frame = frame.copy()
        frame["task"] = frame["task"].astype(str).str.strip().str.lower()
        frame["model"] = frame["model"].astype(str).str.strip().str.lower().replace(
            {"dmpnn": "legacy_node_mpnn"}
        )
        if (
            frame["task"].nunique(dropna=False) != 1
            or frame["model"].nunique(dropna=False) != 1
            or frame["seed"].nunique(dropna=False) != 1
        ):
            return None, "mixed_run_metadata_in_prediction_file"
        task = str(frame["task"].iloc[0])
        model = str(frame["model"].iloc[0])
        if task not in TASK_META_BY_SLUG:
            return None, "unknown_tdc_task"
        # Submission neural runs must use their four-artifact neural schema;
        # accepting them through the traditional schema would bypass companion
        # metrics/checkpoint identity checks.
        if model in {"gin", "chemprop_dmpnn"}:
            return None, "neural_standard_schema_forbidden"
        try:
            seed_values = pd.to_numeric(frame["seed"], errors="raise")
        except (TypeError, ValueError):
            return None, "invalid_prediction_seed"
        if not np.equal(seed_values, np.floor(seed_values)).all():
            return None, "invalid_prediction_seed"
        frame["seed"] = seed_values.astype(int)
        meta = TASK_META_BY_SLUG[task]
        if set(frame["task_type"].astype(str).str.lower()) != {meta["task_type"]}:
            return None, "prediction_task_type_mismatch"
        if set(frame["official_metric"].astype(str).str.lower()) != {meta["metric"]}:
            return None, "prediction_official_metric_mismatch"
        reason = _validate_normalized_rows(frame, task, model)
        if reason:
            return None, reason
        return frame, None

    neural = {"split", "smiles", "label", "prediction"}
    if not neural.issubset(frame.columns):
        return None, "unrecognized_schema"
    match = re.match(
        r"^(?P<task>.+?)_(?P<model>chemprop_dmpnn|gin|dmpnn)_seed(?P<seed>\d+)$",
        path.stem,
    )
    if not match:
        return None, "neural_filename_not_parseable"
    task = match.group("task").lower()
    if task not in TASK_META_BY_SLUG:
        return None, "unknown_tdc_task"

    # Neural CSVs are written before their metrics JSON.  The JSON is the
    # durable completion marker and is mandatory for inclusion.
    metrics_path = path.parent.parent / "metrics" / f"{path.stem}.json"
    checkpoint_path = path.parent.parent / "checkpoints" / f"{path.stem}.pt"
    if not metrics_path.exists():
        return None, "missing_neural_metrics_marker"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        return None, "missing_or_empty_neural_checkpoint"
    try:
        marker = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "invalid_neural_metrics_marker"
    required_marker = {
        "task",
        "model",
        "seed",
        "task_type",
        "metric_name",
        "valid_metric",
        "test_metric",
    }
    if not required_marker.issubset(marker):
        return None, "incomplete_neural_metrics_marker"
    try:
        if not all(
            np.isfinite(float(marker[key])) for key in ("valid_metric", "test_metric")
        ):
            return None, "nonfinite_neural_metrics_marker"
    except (TypeError, ValueError):
        return None, "invalid_neural_metrics_marker"

    parsed_model = match.group("model")
    if parsed_model == "chemprop_dmpnn" and marker.get("status") != "complete":
        return None, "chemprop_marker_not_complete"
    if parsed_model == "gin" and marker.get("status", "complete") != "complete":
        return None, "gin_marker_not_complete"
    # The earlier `_dmpnn_` files were produced by a custom node-message-passing
    # implementation, not Chemprop D-MPNN.  Preserve them under an honest legacy
    # name so they can never satisfy the locked Chemprop coverage gate.
    normalized_model = (
        "legacy_node_mpnn" if parsed_model == "dmpnn" else parsed_model
    )
    marker_task = str(marker.get("task", match.group("task"))).lower()
    marker_model = str(marker.get("model", parsed_model)).lower()
    try:
        marker_seed = int(marker["seed"])
    except (TypeError, ValueError):
        return None, "invalid_neural_metrics_marker"
    compatible_marker_models = {parsed_model, normalized_model}
    if (
        marker_task != task
        or marker_model not in compatible_marker_models
        or marker_seed != int(match.group("seed"))
    ):
        return None, "neural_marker_metadata_mismatch"

    meta = TASK_META_BY_SLUG[task]
    if (
        str(marker.get("task_type", "")).lower() != meta["task_type"]
        or str(marker.get("metric_name", "")).lower() != meta["metric"]
    ):
        return None, "neural_marker_task_metadata_mismatch"

    if parsed_model == "chemprop_dmpnn":
        config_path = path.parent.parent / "configs" / f"{path.stem}.json"
        if not config_path.is_file():
            return None, "missing_chemprop_config"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None, "invalid_chemprop_config"
        training = config.get("training", {})
        architecture = config.get("architecture", {})
        software = config.get("software", {})
        config_data = config.get("data", {})
        try:
            config_seed = int(config.get("seed", -1))
            data_seed = int(training.get("data_seed", -1))
            pytorch_seed = int(training.get("pytorch_seed", -1))
        except (TypeError, ValueError, AttributeError):
            return None, "chemprop_config_seed_mismatch"
        if (
            config.get("status") != "complete"
            or str(config.get("task", "")).lower() != task
            or config.get("model") != "chemprop_dmpnn"
            or config_seed != int(match.group("seed"))
            or str(config.get("task_type", "")).lower() != meta["task_type"]
            or str(config.get("official_metric", "")).lower() != meta["metric"]
        ):
            return None, "chemprop_config_metadata_mismatch"
        if (
            not isinstance(training, dict)
            or data_seed != int(match.group("seed"))
            or pytorch_seed != int(match.group("seed"))
        ):
            return None, "chemprop_config_seed_mismatch"
        if (
            not isinstance(architecture, dict)
            or architecture.get("directed_bond_messages") is not True
            or architecture.get("atom_messages") is not False
            or architecture.get("undirected") is not False
        ):
            return None, "chemprop_config_not_directed_bond_dmpnn"
        if not isinstance(software, dict) or software.get("chemprop") != "2.2.1":
            return None, "chemprop_config_version_mismatch"
        if (
            marker.get("chemprop_version") != "2.2.1"
            or marker.get("official_test_preserved") is not True
        ):
            return None, "chemprop_marker_provenance_mismatch"
        if not isinstance(config_data, dict) or config_data.get("official_test_preserved") is not True:
            return None, "chemprop_config_split_provenance_missing"
        for hash_name in ("train_sha256", "valid_sha256", "test_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(config_data.get(hash_name, ""))):
                return None, "chemprop_config_split_provenance_missing"

    frame = frame.rename(columns={"label": "y_true", "prediction": "y_pred"})
    frame["seed"] = int(match.group("seed"))
    frame["sample_index"] = frame.groupby("split", sort=False).cumcount()
    reason = _validate_normalized_rows(frame, task, normalized_model)
    if reason:
        return None, reason

    if parsed_model == "chemprop_dmpnn":
        try:
            expected_valid_n = int(config_data["valid_n"])
            expected_test_n = int(config_data["test_n"])
        except (KeyError, TypeError, ValueError):
            return None, "chemprop_config_split_counts_missing"
        split_counts = frame["split"].value_counts().to_dict()
        if (
            split_counts.get("valid", 0) != expected_valid_n
            or split_counts.get("test", 0) != expected_test_n
        ):
            return None, "chemprop_prediction_split_count_mismatch"

    expected_rows = marker.get("prediction_rows")
    if expected_rows is not None:
        try:
            if int(expected_rows) != len(frame):
                return None, "neural_marker_prediction_row_count_mismatch"
        except (TypeError, ValueError):
            return None, "invalid_neural_metrics_marker"
    for split_name, marker_name in (("valid", "valid_metric"), ("test", "test_metric")):
        part = frame[frame["split"] == split_name]
        recomputed = official_metric(
            meta["metric"],
            part["y_true"].to_numpy(dtype=float),
            part["y_pred"].to_numpy(dtype=float),
        )
        if not np.isclose(
            recomputed,
            float(marker[marker_name]),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=False,
        ):
            return None, f"neural_{split_name}_metric_prediction_mismatch"
    return frame, None


def _similarity_cache_path(
    similarity_dir: Path,
    task: str,
    seed: int,
    split_source: str,
) -> Path:
    """Resolve a split-explicit cache, with legacy PyTDC compatibility only."""
    explicit = (
        similarity_dir
        / f"{task.lower()}_seed{seed}_{split_source}_test_similarity.csv"
    )
    if explicit.exists() or split_source != PYTDC_SPLIT_SOURCE:
        return explicit
    # Historical exploratory caches predate the split-source suffix. Never
    # apply this fallback to locked_cache, because that could silently mix the
    # formal validation protocol with a PyTDC-derived applicability cache.
    return similarity_dir / f"{task.lower()}_seed{seed}_test_similarity.csv"


def _load_similarity_for_test(
    path: Path,
    test: pd.DataFrame,
    expected_split_source: str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Validate exact test-row identity and finite similarity values."""
    try:
        similarity = pd.read_csv(path)
    except Exception as exc:
        return None, f"unreadable_similarity_cache:{type(exc).__name__}"
    required = {"sample_index", "max_tanimoto"}
    if not required.issubset(similarity.columns):
        return None, "similarity_cache_missing_columns"
    if expected_split_source is not None:
        provenance_columns = {
            "cache_schema",
            "split_source",
            "train_smiles_sha256",
            "test_smiles_sha256",
        }
        if not provenance_columns.issubset(similarity.columns):
            return None, "similarity_cache_provenance_missing"
        if (
            set(similarity["cache_schema"].astype(str))
            != {"applicability_similarity_v2"}
            or set(similarity["split_source"].astype(str))
            != {expected_split_source}
            or not all(
                similarity[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
                for column in ("train_smiles_sha256", "test_smiles_sha256")
            )
        ):
            return None, "similarity_cache_provenance_mismatch"
    similarity = similarity.copy()
    try:
        indices = pd.to_numeric(similarity["sample_index"], errors="raise")
        values = pd.to_numeric(similarity["max_tanimoto"], errors="raise")
    except (TypeError, ValueError):
        return None, "similarity_cache_non_numeric"
    if (
        not np.equal(indices, np.floor(indices)).all()
        or (indices < 0).any()
        or indices.duplicated().any()
    ):
        return None, "similarity_cache_invalid_sample_index"
    similarity["sample_index"] = indices.astype(int)
    similarity["max_tanimoto"] = values.astype(float)
    if (
        not np.isfinite(similarity["max_tanimoto"]).all()
        or not similarity["max_tanimoto"].between(0.0, 1.0, inclusive="both").all()
    ):
        return None, "similarity_cache_invalid_values"
    expected_indices = set(test["sample_index"].astype(int))
    if len(similarity) != len(test) or set(similarity["sample_index"]) != expected_indices:
        return None, "similarity_cache_test_index_mismatch"
    if "smiles" in similarity.columns:
        left = test[["sample_index", "smiles"]].sort_values("sample_index")
        right = similarity[["sample_index", "smiles"]].sort_values("sample_index")
        if left["smiles"].astype(str).tolist() != right["smiles"].astype(str).tolist():
            return None, "similarity_cache_test_smiles_mismatch"
    return similarity[["sample_index", "max_tanimoto"]], None


def load_runs(
    prediction_dirs: list[Path],
    similarity_dir: Path,
    excluded_models: frozenset[str] = LEGACY_EXCLUDED_MODELS,
    similarity_split_source: str = PYTDC_SPLIT_SOURCE,
    require_similarity: bool = False,
) -> tuple[dict[tuple[str, str, int], RunPrediction], pd.DataFrame]:
    runs: dict[tuple[str, str, int], RunPrediction] = {}
    audit_rows: list[dict] = []
    for path in _prediction_files(prediction_dirs):
        frame, reason = _normalize_prediction_file(path)
        if frame is None:
            audit_rows.append({"path": str(path), "included": False, "reason": reason})
            continue
        for (task, model, seed), run_frame in frame.groupby(
            ["task", "model", "seed"], sort=True
        ):
            task = str(task).lower()
            model = str(model).lower()
            seed = int(seed)
            if model in excluded_models:
                audit_rows.append(
                    {
                        "path": str(path),
                        "included": False,
                        "reason": f"excluded_legacy_model:{model}",
                    }
                )
                continue
            if task not in TASK_META_BY_SLUG:
                audit_rows.append(
                    {
                        "path": str(path),
                        "included": False,
                        "reason": f"unknown_task:{task}",
                    }
                )
                continue
            meta = TASK_META_BY_SLUG[task]
            valid = run_frame[run_frame["split"] == "valid"].copy()
            test = run_frame[run_frame["split"] == "test"].copy()
            if valid.empty or test.empty:
                audit_rows.append(
                    {
                        "path": str(path),
                        "included": False,
                        "reason": "missing_valid_or_test_split",
                    }
                )
                continue
            similarity_path = _similarity_cache_path(
                similarity_dir, task, seed, similarity_split_source
            )
            if similarity_path.exists():
                similarity, similarity_error = _load_similarity_for_test(
                    similarity_path,
                    test,
                    expected_split_source=(
                        similarity_split_source if require_similarity else None
                    ),
                )
                if similarity_error:
                    if require_similarity:
                        raise ValueError(
                            f"Invalid required similarity cache {similarity_path}: "
                            f"{similarity_error}"
                        )
                    similarity = None
                if similarity is None:
                    test["max_tanimoto"] = np.nan
                else:
                    test = test.merge(similarity, on="sample_index", how="left")
            else:
                similarity_error = "similarity_cache_missing"
                if require_similarity:
                    raise FileNotFoundError(
                        "Required split-explicit applicability cache is missing: "
                        f"{similarity_path}"
                    )
                test["max_tanimoto"] = np.nan
            columns = ["sample_index", "smiles", "y_true", "y_pred", "max_tanimoto"]
            valid = valid[["sample_index", "smiles", "y_true", "y_pred"]].sort_values(
                "sample_index"
            )
            test = test[columns].sort_values("sample_index")
            key = (task, model, seed)
            if key in runs:
                raise ValueError(
                    f"Duplicate complete prediction run {key}: {runs[key].source} and {path}"
                )
            runs[key] = RunPrediction(
                task=task,
                model=model,
                seed=seed,
                task_type=str(run_frame.get("task_type", pd.Series([meta["task_type"]])).iloc[0]),
                official_metric=str(
                    run_frame.get("official_metric", pd.Series([meta["metric"]])).iloc[0]
                ),
                valid=valid.reset_index(drop=True),
                test=test.reset_index(drop=True),
                source=str(path),
            )
            audit_rows.append(
                {
                    "path": str(path),
                    "included": True,
                    "reason": "complete",
                    "similarity_path": str(similarity_path),
                    "similarity_split_source": similarity_split_source,
                    "similarity_complete": bool(test["max_tanimoto"].notna().all()),
                }
            )
    if not runs:
        raise FileNotFoundError(
            "No complete prediction runs found under "
            + ", ".join(str(path) for path in prediction_dirs)
        )
    return runs, pd.DataFrame(audit_rows)


def audit_run_sample_alignment(
    runs: dict[tuple[str, str, int], RunPrediction],
) -> tuple[pd.DataFrame, bool]:
    """Check validation and test identity across models for each task/seed."""
    grouped: dict[tuple[str, int], list[RunPrediction]] = {}
    for run in runs.values():
        grouped.setdefault((run.task, run.seed), []).append(run)
    rows: list[dict] = []
    all_aligned = True
    for (task, seed), task_runs in sorted(grouped.items()):
        task_runs = sorted(task_runs, key=lambda item: item.model)
        reference = task_runs[0]
        valid_aligned = all(
            sample_rows_match(run.valid, reference.valid) for run in task_runs
        )
        test_aligned = all(
            sample_rows_match(run.test, reference.test) for run in task_runs
        )
        all_aligned = all_aligned and valid_aligned and test_aligned
        for run in task_runs:
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "model": run.model,
                    "valid_rows": len(run.valid),
                    "test_rows": len(run.test),
                    "valid_sample_fingerprint": sample_fingerprint(run.valid),
                    "test_sample_fingerprint": sample_fingerprint(run.test),
                    "valid_matches_task_seed_reference": sample_rows_match(
                        run.valid, reference.valid
                    ),
                    "test_matches_task_seed_reference": sample_rows_match(
                        run.test, reference.test
                    ),
                    "task_seed_models_aligned": valid_aligned and test_aligned,
                }
            )
    return pd.DataFrame(rows), all_aligned


def build_ensembles(
    runs: dict[tuple[str, str, int], RunPrediction],
    minimum_seeds: int,
    alpha: float,
) -> tuple[list[ModelEnsemble], pd.DataFrame, pd.DataFrame]:
    grouped: dict[tuple[str, str], list[RunPrediction]] = {}
    for run in runs.values():
        grouped.setdefault((run.task, run.model), []).append(run)

    ensembles: list[ModelEnsemble] = []
    validation_rows: list[dict] = []
    availability_rows: list[dict] = []
    for (task, model), model_runs in sorted(grouped.items()):
        model_runs = sorted(model_runs, key=lambda run: run.seed)
        unique_seeds = tuple(run.seed for run in model_runs)
        availability_rows.append(
            {
                "task": task,
                "model": model,
                "available_seed_count": len(unique_seeds),
                "available_seeds": ";".join(map(str, unique_seeds)),
                "eligible": len(unique_seeds) >= minimum_seeds,
            }
        )
        if len(unique_seeds) < minimum_seeds:
            continue

        validation_scores: dict[int, float] = {}
        stacked_test: list[pd.DataFrame] = []
        coverage_parts: list[pd.DataFrame] = []
        first = model_runs[0]
        reference_test = first.test
        for run in model_runs:
            if not sample_rows_match(run.test, reference_test):
                raise ValueError(
                    "Test sample identity/order/labels disagree across seeds for "
                    f"{(task, model)}; refusing to ensemble misaligned rows"
                )
            raw_valid_score = official_metric(
                run.official_metric,
                run.valid["y_true"].to_numpy(),
                run.valid["y_pred"].to_numpy(),
            )
            oriented = orient_metric(run.official_metric, raw_valid_score)
            validation_scores[run.seed] = oriented
            validation_rows.append(
                {
                    "task": task,
                    "model": model,
                    "seed": run.seed,
                    "official_metric": run.official_metric,
                    "validation_metric_raw": raw_valid_score,
                    "validation_score_oriented": oriented,
                    "source": run.source,
                }
            )
            test_part = run.test.copy()
            test_part["seed"] = run.seed
            stacked_test.append(test_part)
            if run.task_type == "regression":
                residuals = np.abs(
                    run.valid["y_true"].to_numpy(dtype=float)
                    - run.valid["y_pred"].to_numpy(dtype=float)
                )
                quantile = conformal_quantile(residuals, alpha=alpha)
                covered = run.test[["sample_index"]].copy()
                covered["covered"] = (
                    np.abs(
                        run.test["y_true"].to_numpy(dtype=float)
                        - run.test["y_pred"].to_numpy(dtype=float)
                    )
                    <= quantile
                ).astype(float)
                coverage_parts.append(covered.assign(seed=run.seed))

        stacked = pd.concat(stacked_test, ignore_index=True)
        label_counts = stacked.groupby("sample_index")["y_true"].nunique(dropna=False)
        if int(label_counts.max()) > 1:
            raise ValueError(f"Test labels disagree across seeds for {(task, model)}")
        smiles_counts = stacked.groupby("sample_index")["smiles"].nunique(dropna=False)
        if int(smiles_counts.max()) > 1:
            raise ValueError(f"Test SMILES disagree across seeds for {(task, model)}")
        ensemble_test = (
            stacked.groupby("sample_index", as_index=False)
            .agg(
                smiles=("smiles", "first"),
                y_true=("y_true", "first"),
                prediction=("y_pred", "mean"),
                uncertainty=("y_pred", lambda values: float(np.std(values, ddof=1))),
                max_tanimoto=("max_tanimoto", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        ensemble_test = ensemble_test[
            ensemble_test["n_seeds"] == len(unique_seeds)
        ].reset_index(drop=True)
        conformal_coverage = float("nan")
        if coverage_parts:
            coverage = pd.concat(coverage_parts, ignore_index=True)
            coverage = coverage.groupby("sample_index", as_index=False).agg(
                conformal_covered=("covered", "mean"),
                n_seeds=("seed", "nunique"),
            )
            coverage = coverage[coverage["n_seeds"] == len(unique_seeds)]
            conformal_coverage = float(coverage["conformal_covered"].mean())

        ensembles.append(
            ModelEnsemble(
                task=task,
                model=model,
                task_type=first.task_type,
                official_metric=first.official_metric,
                seeds=unique_seeds,
                validation_scores=validation_scores,
                test=ensemble_test,
                conformal_coverage=conformal_coverage,
            )
        )
    return ensembles, pd.DataFrame(validation_rows), pd.DataFrame(availability_rows)


def audit_ensemble_test_alignment(
    ensembles: list[ModelEnsemble],
) -> tuple[pd.DataFrame, bool]:
    """Require one exact official-test sample fingerprint per endpoint.

    This comparison is across models after each model's five seeds have already
    passed the within-model fingerprint check above.  A mismatch is a hard
    analysis error; it cannot be repaired by intersecting rows because that
    would silently change the locked endpoint population.
    """
    rows: list[dict] = []
    all_aligned = True
    by_task: dict[str, list[ModelEnsemble]] = {}
    for item in ensembles:
        by_task.setdefault(item.task, []).append(item)
    for task, task_ensembles in sorted(by_task.items()):
        fingerprints = {item.model: sample_fingerprint(item.test) for item in task_ensembles}
        reference_item = task_ensembles[0] if task_ensembles else None
        reference = fingerprints.get(reference_item.model, "") if reference_item else ""
        aligned = bool(
            reference_item is None
            or all(sample_rows_match(item.test, reference_item.test) for item in task_ensembles)
        )
        all_aligned = all_aligned and aligned
        for item in sorted(task_ensembles, key=lambda value: value.model):
            rows.append(
                {
                    "task": task,
                    "model": item.model,
                    "test_rows": len(item.test),
                    "test_sample_fingerprint": fingerprints[item.model],
                    "matches_task_reference": bool(
                        reference_item is not None
                        and sample_rows_match(item.test, reference_item.test)
                    ),
                    "task_models_aligned": aligned,
                }
            )
    return pd.DataFrame(rows), all_aligned


def select_validation_models(
    ensembles: list[ModelEnsemble],
    seed_bootstrap_reps: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select by validation score and quantify seed-bootstrap stability."""
    by_task: dict[str, list[ModelEnsemble]] = {}
    for ensemble in ensembles:
        by_task.setdefault(ensemble.task, []).append(ensemble)
    selection_rows: list[dict] = []
    probability_rows: list[dict] = []
    for task, task_ensembles in sorted(by_task.items()):
        score_means = {
            item.model: float(np.mean(list(item.validation_scores.values())))
            for item in task_ensembles
        }
        ordered = sorted(score_means, key=lambda model: (-score_means[model], model))
        selected = ordered[0]
        best = score_means[selected]
        runner_up = score_means[ordered[1]] if len(ordered) > 1 else float("nan")
        tolerance = max(absolute_tolerance, relative_tolerance * max(abs(best), 1e-12))
        tie_models = sorted(
            model for model, score in score_means.items() if best - score <= tolerance
        )

        seed_sets = [set(item.validation_scores) for item in task_ensembles]
        shared_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
        exact_counts = {item.model: 0 for item in task_ensembles}
        tie_counts = {item.model: 0 for item in task_ensembles}
        for _ in range(seed_bootstrap_reps):
            boot_scores: dict[str, float] = {}
            if shared_seeds:
                draws = rng.choice(shared_seeds, size=len(shared_seeds), replace=True)
                for item in task_ensembles:
                    boot_scores[item.model] = float(
                        np.mean([item.validation_scores[int(seed)] for seed in draws])
                    )
            else:
                for item in task_ensembles:
                    seeds = list(item.validation_scores)
                    draws = rng.choice(seeds, size=len(seeds), replace=True)
                    boot_scores[item.model] = float(
                        np.mean([item.validation_scores[int(seed)] for seed in draws])
                    )
            boot_ordered = sorted(
                boot_scores, key=lambda model: (-boot_scores[model], model)
            )
            exact_counts[boot_ordered[0]] += 1
            boot_best = boot_scores[boot_ordered[0]]
            boot_tolerance = max(
                absolute_tolerance,
                relative_tolerance * max(abs(boot_best), 1e-12),
            )
            for model, score in boot_scores.items():
                if boot_best - score <= boot_tolerance:
                    tie_counts[model] += 1

        for model in sorted(score_means):
            probability_rows.append(
                {
                    "task": task,
                    "model": model,
                    "mean_validation_score_oriented": score_means[model],
                    "exact_selection_probability": exact_counts[model]
                    / seed_bootstrap_reps,
                    "validation_tie_probability": tie_counts[model]
                    / seed_bootstrap_reps,
                }
            )
        selection_rows.append(
            {
                "task": task,
                "task_type": task_ensembles[0].task_type,
                "official_metric": task_ensembles[0].official_metric,
                "n_models": len(task_ensembles),
                "model_pool": ";".join(sorted(score_means)),
                "validation_selected_model": selected,
                "validation_selected_score_oriented": best,
                "validation_runner_up_score_oriented": runner_up,
                "validation_score_margin": (
                    best - runner_up if np.isfinite(runner_up) else float("nan")
                ),
                "validation_tolerance": tolerance,
                "validation_tie_models": ";".join(tie_models),
                "validation_tie_count": len(tie_models),
                "selected_exact_bootstrap_probability": exact_counts[selected]
                / seed_bootstrap_reps,
                "selected_tie_bootstrap_probability": tie_counts[selected]
                / seed_bootstrap_reps,
                "shared_seed_bootstrap": bool(shared_seeds),
                "bootstrap_seed_count": len(shared_seeds),
            }
        )
    return pd.DataFrame(selection_rows), pd.DataFrame(probability_rows)


def evaluate_reliability(
    ensembles: list[ModelEnsemble], target_coverage: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reliability_rows: list[dict] = []
    curve_parts: list[pd.DataFrame] = []
    for item in ensembles:
        test = item.test
        y_true = test["y_true"].to_numpy(dtype=float)
        prediction = test["prediction"].to_numpy(dtype=float)
        uncertainty = test["uncertainty"].to_numpy(dtype=float)
        sample_index = test["sample_index"].to_numpy()
        absolute_error = np.abs(y_true - prediction)
        test_metric_raw = official_metric(item.official_metric, y_true, prediction)
        base = {
            "task": item.task,
            "model": item.model,
            "task_type": item.task_type,
            "official_metric": item.official_metric,
            "n_seeds": len(item.seeds),
            "test_n": len(test),
            "mean_validation_score_oriented": float(
                np.mean(list(item.validation_scores.values()))
            ),
            "test_official_metric_raw": test_metric_raw,
            "test_official_score_oriented": orient_metric(
                item.official_metric, test_metric_raw
            ),
            "dimension": "applicability_alignment",
            "value": spearman(
                test["max_tanimoto"].to_numpy(dtype=float), -absolute_error
            ),
            "higher_is_better": True,
            "risk_metric": "absolute_error_alignment",
            "mean_sample_loss": float(np.mean(absolute_error)),
        }
        reliability_rows.append(base)

        loss_specs: list[tuple[str, np.ndarray]] = []
        if item.task_type == "classification":
            probability = np.clip(prediction, 1e-12, 1 - 1e-12)
            brier_losses = (y_true - probability) ** 2
            log_losses = -(
                y_true * np.log(probability) + (1 - y_true) * np.log(1 - probability)
            )
            reliability_rows.extend(
                [
                    {
                        **base,
                        "dimension": "ece",
                        "value": expected_calibration_error(y_true, probability),
                        "higher_is_better": False,
                        "risk_metric": "calibration_error",
                        "mean_sample_loss": float(np.mean(brier_losses)),
                    },
                    {
                        **base,
                        "dimension": "brier",
                        "value": float(np.mean(brier_losses)),
                        "higher_is_better": False,
                        "risk_metric": "brier",
                        "mean_sample_loss": float(np.mean(brier_losses)),
                    },
                ]
            )
            loss_specs = [("brier", brier_losses), ("log_loss", log_losses)]
        else:
            reliability_rows.append(
                {
                    **base,
                    "dimension": "conformal_coverage_error",
                    "value": abs(item.conformal_coverage - target_coverage),
                    "higher_is_better": False,
                    "risk_metric": "coverage_error",
                    "mean_sample_loss": float(np.mean(absolute_error)),
                    "conformal_coverage": item.conformal_coverage,
                }
            )
            loss_specs = [("mae", absolute_error)]

        for risk_metric, losses in loss_specs:
            curve = risk_coverage_curve(losses, uncertainty, sample_index)
            aurc, oracle_aurc, excess = aurc_from_curve(curve)
            mean_loss = float(np.mean(losses))
            relative_excess = excess / max(mean_loss, 1e-12)
            curve.insert(0, "risk_metric", risk_metric)
            curve.insert(0, "model", item.model)
            curve.insert(0, "task", item.task)
            curve_parts.append(curve)
            dimension = "selective_relative_excess_aurc"
            if risk_metric == "log_loss":
                dimension = "selective_relative_excess_aurc_log_loss"
            reliability_rows.append(
                {
                    **base,
                    "dimension": dimension,
                    "value": relative_excess,
                    "higher_is_better": False,
                    "risk_metric": risk_metric,
                    "mean_sample_loss": mean_loss,
                    "aurc": aurc,
                    "oracle_aurc": oracle_aurc,
                    "excess_aurc": excess,
                }
            )
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    return pd.DataFrame(reliability_rows), curves


def build_regrets(
    reliability: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    selection_index = selections.set_index("task")
    rows: list[dict] = []
    for (task, dimension), group in reliability.groupby(
        ["task", "dimension"], sort=True
    ):
        if task not in selection_index.index:
            continue
        selection = selection_index.loc[task]
        selected_model = str(selection["validation_selected_model"])
        usable = group[np.isfinite(group["value"].to_numpy(dtype=float))].copy()
        if selected_model not in set(usable["model"]):
            continue
        higher_is_better = bool(usable["higher_is_better"].iloc[0])
        usable["reliability_loss"] = (
            -usable["value"] if higher_is_better else usable["value"]
        )
        oracle_index = usable["reliability_loss"].idxmin()
        oracle = usable.loc[oracle_index]
        selected = usable[usable["model"] == selected_model].iloc[0]
        oracle_loss = float(oracle["reliability_loss"])
        selected_loss = float(selected["reliability_loss"])
        regret = max(0.0, selected_loss - oracle_loss)
        random_expected_regret = max(
            0.0, float(usable["reliability_loss"].mean()) - oracle_loss
        )
        regret_fraction_of_random = (
            regret / random_expected_regret
            if random_expected_regret > 1e-12
            else 0.0
        )
        loss_range = float(
            usable["reliability_loss"].max() - usable["reliability_loss"].min()
        )
        normalized_regret = regret / loss_range if loss_range > 1e-12 else 0.0

        tie_models = str(selection["validation_tie_models"]).split(";")
        tie_usable = usable[usable["model"].isin(tie_models)]
        if tie_usable.empty:
            tie_usable = usable[usable["model"] == selected_model]
        tie_best_regret = max(
            0.0, float(tie_usable["reliability_loss"].min()) - oracle_loss
        )
        tie_worst_regret = max(
            0.0, float(tie_usable["reliability_loss"].max()) - oracle_loss
        )
        tolerance = PRACTICAL_TOLERANCES[dimension]
        rows.append(
            {
                "task": task,
                "task_type": selected["task_type"],
                "official_metric": selected["official_metric"],
                "dimension": dimension,
                "risk_metric": selected["risk_metric"],
                "n_models": len(usable),
                "model_pool": ";".join(sorted(usable["model"].astype(str))),
                "validation_selected_model": selected_model,
                "test_reliability_oracle_model": str(oracle["model"]),
                "selected_reliability_value": float(selected["value"]),
                "oracle_reliability_value": float(oracle["value"]),
                "higher_is_better": higher_is_better,
                "reliability_regret": regret,
                "random_selection_expected_regret": random_expected_regret,
                "regret_fraction_of_random": regret_fraction_of_random,
                "regret_minus_random_expected": regret - random_expected_regret,
                "normalized_regret": normalized_regret,
                "practical_tolerance": tolerance,
                "meaningful_regret": bool(regret > tolerance),
                "validation_tie_models": selection["validation_tie_models"],
                "validation_tie_count": int(selection["validation_tie_count"]),
                "tie_best_case_regret": tie_best_regret,
                "tie_best_case_normalized_regret": (
                    tie_best_regret / loss_range if loss_range > 1e-12 else 0.0
                ),
                "tie_worst_case_regret": tie_worst_regret,
                "tie_robust_meaningful_regret": bool(tie_best_regret > tolerance),
                "selected_exact_bootstrap_probability": float(
                    selection["selected_exact_bootstrap_probability"]
                ),
                "selected_tie_bootstrap_probability": float(
                    selection["selected_tie_bootstrap_probability"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_regrets(
    regrets: pd.DataFrame,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    for dimension, group in regrets.groupby("dimension", sort=True):
        raw = group["reliability_regret"].to_numpy(dtype=float)
        normalized = group["normalized_regret"].to_numpy(dtype=float)
        tie_best = group["tie_best_case_regret"].to_numpy(dtype=float)
        tie_normalized = group["tie_best_case_normalized_regret"].to_numpy(dtype=float)
        random_fraction = group["regret_fraction_of_random"].to_numpy(dtype=float)
        raw_low, raw_high = bootstrap_interval(raw, bootstrap_reps, rng)
        norm_low, norm_high = bootstrap_interval(normalized, bootstrap_reps, rng)
        tie_raw_low, tie_raw_high = bootstrap_interval(tie_best, bootstrap_reps, rng)
        tie_low, tie_high = bootstrap_interval(tie_normalized, bootstrap_reps, rng)
        random_low, random_high = bootstrap_interval(
            random_fraction, bootstrap_reps, rng
        )
        stable = group[group["selected_tie_bootstrap_probability"] >= 0.80]
        rows.append(
            {
                "dimension": dimension,
                "n_endpoints": len(group),
                "mean_regret": float(np.mean(raw)),
                "median_regret": float(np.median(raw)),
                "mean_regret_ci_low": raw_low,
                "mean_regret_ci_high": raw_high,
                "mean_normalized_regret": float(np.mean(normalized)),
                "median_normalized_regret": float(np.median(normalized)),
                "mean_normalized_regret_ci_low": norm_low,
                "mean_normalized_regret_ci_high": norm_high,
                "nonzero_regret_count": int(np.sum(raw > 1e-12)),
                "nonzero_regret_fraction": float(np.mean(raw > 1e-12)),
                "meaningful_regret_count": int(group["meaningful_regret"].sum()),
                "meaningful_regret_fraction": float(group["meaningful_regret"].mean()),
                "mean_tie_best_case_regret": float(np.mean(tie_best)),
                "mean_tie_best_case_regret_ci_low": tie_raw_low,
                "mean_tie_best_case_regret_ci_high": tie_raw_high,
                "mean_tie_best_case_normalized_regret": float(np.mean(tie_normalized)),
                "mean_tie_best_case_normalized_ci_low": tie_low,
                "mean_tie_best_case_normalized_ci_high": tie_high,
                "mean_regret_fraction_of_random": float(np.mean(random_fraction)),
                "median_regret_fraction_of_random": float(np.median(random_fraction)),
                "mean_regret_fraction_of_random_ci_low": random_low,
                "mean_regret_fraction_of_random_ci_high": random_high,
                "fraction_retaining_at_least_half_random_regret": float(
                    np.mean(random_fraction >= 0.5)
                ),
                "tie_robust_meaningful_count": int(
                    group["tie_robust_meaningful_regret"].sum()
                ),
                "tie_robust_meaningful_fraction": float(
                    group["tie_robust_meaningful_regret"].mean()
                ),
                "stable_selection_endpoints": len(stable),
                "mean_regret_stable_selection": (
                    float(stable["reliability_regret"].mean())
                    if len(stable)
                    else float("nan")
                ),
                "median_selected_exact_bootstrap_probability": float(
                    group["selected_exact_bootstrap_probability"].median()
                ),
                "median_selected_tie_bootstrap_probability": float(
                    group["selected_tie_bootstrap_probability"].median()
                ),
                "practical_tolerance": float(group["practical_tolerance"].iloc[0]),
                "sign_flip_p": sign_flip_pvalue(raw, permutation_reps, rng),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["holm_p"] = holm_adjust(summary["sign_flip_p"])
        # The oracle construction makes regret non-negative, so a sign-flip
        # test against exactly zero is reported only as a descriptive legacy
        # diagnostic.  Central evidence uses endpoint-bootstrap magnitude.
        summary["practical_effect_signal"] = (
            (summary["mean_normalized_regret_ci_low"] >= 0.10)
            & (summary["mean_regret_ci_low"] >= summary["practical_tolerance"])
            & (summary["meaningful_regret_fraction"] >= 0.25)
        )
        summary["tie_robust_effect_signal"] = (
            (summary["mean_tie_best_case_normalized_ci_low"] >= 0.05)
            & (
                summary["mean_tie_best_case_regret_ci_low"]
                >= summary["practical_tolerance"]
            )
            & (summary["tie_robust_meaningful_fraction"] >= 0.20)
        )
        # Exact winner mismatch has a high combinatorial null rate.  This
        # magnitude benchmark asks whether validation selection leaves at least
        # half the regret expected from choosing a model uniformly at random.
        summary["random_baseline_retention_signal"] = (
            (summary["mean_regret_fraction_of_random_ci_low"] >= 0.50)
            & (
                summary["fraction_retaining_at_least_half_random_regret"]
                >= 0.40
            )
        )
    return summary


def build_gate(
    ensembles: list[ModelEnsemble],
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    minimum_seeds: int,
    expected_models: tuple[str, ...],
    gate_scope: str,
    sample_alignment_ok: bool = True,
) -> dict:
    coverage_frame = pd.DataFrame(
        [
            {
                "task": item.task,
                "model": item.model,
                "n_seeds": len(item.seeds),
            }
            for item in ensembles
        ]
    )
    model_task_coverage = (
        coverage_frame.groupby("model")["task"].nunique().sort_index().to_dict()
    )
    model_min_seeds = (
        coverage_frame.groupby("model")["n_seeds"].min().sort_index().to_dict()
    )
    tasks_present = sorted(coverage_frame["task"].unique())
    models_present = sorted(coverage_frame["model"].unique())
    pool_signatures = sorted(set(selections["model_pool"].astype(str)))
    exact_final_seed_sets = bool(
        all(set(item.seeds) == set(FINAL_EXPECTED_SEEDS) for item in ensembles)
    )
    complete = bool(
        len(tasks_present) == len(EXPECTED_TASKS)
        and set(tasks_present) == set(EXPECTED_TASKS)
        and set(models_present) == set(expected_models)
        and model_task_coverage
        and all(
            model_task_coverage.get(model, 0) == len(EXPECTED_TASKS)
            for model in expected_models
        )
        and all(
            model_min_seeds.get(model, 0) >= minimum_seeds
            for model in expected_models
        )
        and len(pool_signatures) == 1
        and sample_alignment_ok
        and (gate_scope != "final" or exact_final_seed_sets)
    )
    effect_dimensions = (
        sorted(
            summary.loc[summary["practical_effect_signal"], "dimension"].astype(str)
        )
        if not summary.empty
        else []
    )
    tie_robust_dimensions = (
        sorted(
            summary.loc[summary["tie_robust_effect_signal"], "dimension"].astype(str)
        )
        if not summary.empty
        else []
    )
    random_retention_dimensions = (
        sorted(
            summary.loc[
                summary["random_baseline_retention_signal"], "dimension"
            ].astype(str)
        )
        if not summary.empty
        else []
    )
    evidence_passes = bool(
        len(effect_dimensions) >= 2
        and len(tie_robust_dimensions) >= 1
        and len(random_retention_dimensions) >= 1
    )
    status = "GO" if complete and evidence_passes else "HOLD"
    if not complete:
        interpretation = (
            "Reliability regret is evaluated without test-set model selection, but the "
            "included model matrix is incomplete or selection pools differ by endpoint. "
            "Treat neural results as partial and finish only the locked baseline matrix."
        )
    elif evidence_passes and gate_scope == "traditional_only":
        interpretation = (
            "The locked traditional-only reliability-regret evidence gate passes. "
            "This is not the final submission gate; GIN and Chemprop D-MPNN must still "
            "complete all 22 endpoints before a final model-wide claim is released."
        )
    elif evidence_passes:
        interpretation = (
            "Validation-selected models incur practically meaningful test reliability "
            "regret in multiple dimensions, including at least one signal robust to the "
            "validation winner tolerance. The reliability-audit claim gate passes."
        )
    else:
        interpretation = (
            "The full model matrix is available, but endpoint-level reliability-regret "
            "effects do not meet the locked magnitude, multiplicity, and tie-robustness "
            "criteria. Do not broaden the central winner-regret claim."
        )
    return {
        "status": status,
        "scope": gate_scope,
        "final_submission_gate": gate_scope == "final",
        "interpretation": interpretation,
        "selection_policy": (
            "Mean oriented official validation metric across complete seeds; test data are "
            "never used for model selection. Validation sets differ by seed, so seed metrics "
            "are averaged rather than validation rows being incorrectly ensembled."
        ),
        "oracle_policy": (
            "The test reliability oracle is a non-deployable descriptive counterfactual used "
            "only after validation selection to calculate non-negative regret."
        ),
        "random_baseline_policy": (
            "For each endpoint and dimension, the exact expected reliability regret under "
            "uniform random selection from the available model pool is computed. Endpoint "
            "bootstrap intervals for selected/random regret retention, rather "
            "than winner-mismatch frequency, enter the central gate."
        ),
        "analysis_timeline_policy": (
            "The regret analysis was developed during transparent exploration of the "
            "traditional results, then frozen before the complete neural matrix became "
            "available. Traditional results are exploratory/locking evidence; the completed "
            "GIN plus Chemprop D-MPNN matrix is the locked final model-completion gate. "
            "Because the same public test endpoints and some partial GIN results have already "
            "been inspected, this stage is not independent external confirmation."
        ),
        "model_identity_policy": (
            "Only files explicitly named *_chemprop_dmpnn_seedN.csv count as Chemprop "
            "D-MPNN. Earlier *_dmpnn_seedN.csv files are renamed legacy_node_mpnn "
            "during normalization and excluded from the final model set."
        ),
        "selective_prediction_policy": (
            "Every empirical coverage point is evaluated with per-sample Brier and log loss "
            "for classification or absolute error for regression. AUROC/AUPRC are excluded "
            "from sample-wise risk because they are ranking/set-level metrics."
        ),
        "coverage": {
            "complete_expected_model_endpoint_matrix": complete,
            "expected_tasks": len(EXPECTED_TASKS),
            "tasks_present": len(tasks_present),
            "expected_models": list(expected_models),
            "models_present": models_present,
            "missing_expected_models": sorted(set(expected_models) - set(models_present)),
            "unexpected_models": sorted(set(models_present) - set(expected_models)),
            "model_task_coverage": {str(k): int(v) for k, v in model_task_coverage.items()},
            "model_minimum_complete_seeds": {
                str(k): int(v) for k, v in model_min_seeds.items()
            },
            "consistent_selection_pool": len(pool_signatures) == 1,
            "selection_pool_signatures": pool_signatures,
            "sample_identity_alignment_exact": sample_alignment_ok,
            "final_seed_set_required": list(FINAL_EXPECTED_SEEDS),
            "final_seed_sets_exact": exact_final_seed_sets,
        },
        "effect_gate": {
            "passes": evidence_passes,
            "dimensions_with_practical_effect_signal": effect_dimensions,
            "dimensions_with_tie_robust_signal": tie_robust_dimensions,
            "dimensions_retaining_random_baseline_regret": random_retention_dimensions,
            "criteria": {
                "practical_effect_signal": (
                    "endpoint-bootstrap lower 95% bound for mean normalized regret>=0.10 "
                    "and raw mean regret>=the dimension tolerance, with practical-regret "
                    "fraction>=0.25"
                ),
                "tie_robust_signal": (
                    "endpoint-bootstrap lower 95% bound for best-case validation-tie "
                    "normalized regret>=0.05 and raw regret>=the dimension tolerance, "
                    "with tie-robust practical-regret fraction>=0.20"
                ),
                "random_baseline_retention": (
                    "endpoint-bootstrap lower 95% bound for mean selected/uniform-random "
                    "expected regret>=0.50 and at least 40% of endpoints retain >=50% "
                    "of random-selection regret"
                ),
                "gate": (
                    "at least 2 effect signals, at least 1 tie-robust signal, and at "
                    "least 1 random-baseline-retention signal"
                ),
            },
            "descriptive_inference_only": (
                "sign_flip_p and Holm-adjusted p values are retained for transparency but "
                "are not used by the gate because oracle regret is non-negative by construction."
            ),
        },
        "practical_tolerances": PRACTICAL_TOLERANCES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select ADMET models by validation performance and quantify their test "
            "reliability regret without test-set winner selection."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        default=str(ROOT / "results" / "predictions" / "traditional_local_prelim"),
    )
    parser.add_argument("--additional-prediction-dirs", nargs="*", default=[])
    parser.add_argument(
        "--similarity-dir",
        default=str(ROOT / "results" / "cache" / "applicability"),
    )
    parser.add_argument(
        "--similarity-split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=PYTDC_SPLIT_SOURCE,
        help=(
            "Split protocol encoded in applicability cache filenames. Formal "
            "five-model analysis must pass locked_cache."
        ),
    )
    parser.add_argument(
        "--require-complete-similarity",
        action="store_true",
        help=(
            "Fail if any task/seed cache is missing, malformed, or misaligned. "
            "This is automatically enabled for --gate-scope final."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "tables" / "reliability_regret_local_prelim"),
    )
    parser.add_argument("--minimum-seeds", type=int, default=5)
    parser.add_argument(
        "--expected-models",
        nargs="+",
        default=list(FINAL_EXPECTED_MODELS),
        help=(
            "Models required for a complete gate. The final locked set is: "
            "rf xgboost lightgbm gin chemprop_dmpnn."
        ),
    )
    parser.add_argument(
        "--gate-scope",
        choices=("final", "traditional_only"),
        default="final",
        help="Label whether this is the final submission gate or a preliminary traditional-only gate.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--absolute-score-tolerance", type=float, default=0.005)
    parser.add_argument("--relative-score-tolerance", type=float, default=0.01)
    parser.add_argument("--seed-bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutation-reps", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20260714)
    args = parser.parse_args()

    expected_models = tuple(str(model).lower() for model in args.expected_models)
    if len(set(expected_models)) != len(expected_models):
        raise ValueError("--expected-models contains duplicates")
    if "legacy_node_mpnn" in expected_models or "dmpnn" in expected_models:
        raise ValueError(
            "The legacy custom node MPNN cannot satisfy the Chemprop D-MPNN gate; "
            "use the explicit model name chemprop_dmpnn."
        )
    if args.gate_scope == "final" and set(expected_models) != set(FINAL_EXPECTED_MODELS):
        raise ValueError(
            "The final gate requires exactly: " + ", ".join(FINAL_EXPECTED_MODELS)
        )

    rng = np.random.default_rng(args.random_seed)
    prediction_dirs = [
        Path(args.prediction_dir),
        *[Path(path) for path in args.additional_prediction_dirs],
    ]
    runs, input_audit = load_runs(
        prediction_dirs,
        Path(args.similarity_dir),
        similarity_split_source=args.similarity_split_source,
        require_similarity=(
            args.require_complete_similarity or args.gate_scope == "final"
        ),
    )
    run_sample_alignment, run_sample_alignment_ok = audit_run_sample_alignment(runs)
    if not run_sample_alignment_ok:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_audit.to_csv(out_dir / "prediction_input_audit.csv", index=False)
        run_sample_alignment.to_csv(
            out_dir / "run_sample_alignment_audit.csv", index=False
        )
        raise ValueError(
            "Validation/test sample fingerprints differ across models within at least "
            "one task/seed; refusing to compare misaligned validation scores. Audit "
            f"written to {out_dir / 'run_sample_alignment_audit.csv'}"
        )
    ensembles, validation_scores, availability = build_ensembles(
        runs, args.minimum_seeds, args.alpha
    )
    sample_alignment, sample_alignment_ok = audit_ensemble_test_alignment(ensembles)
    if not sample_alignment_ok:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_audit.to_csv(out_dir / "prediction_input_audit.csv", index=False)
        run_sample_alignment.to_csv(
            out_dir / "run_sample_alignment_audit.csv", index=False
        )
        availability.to_csv(out_dir / "model_seed_availability.csv", index=False)
        sample_alignment.to_csv(out_dir / "sample_alignment_audit.csv", index=False)
        raise ValueError(
            "Test sample fingerprints differ across models for at least one endpoint; "
            f"audit written to {out_dir / 'sample_alignment_audit.csv'}"
        )
    selections, selection_probabilities = select_validation_models(
        ensembles,
        args.seed_bootstrap_reps,
        args.absolute_score_tolerance,
        args.relative_score_tolerance,
        rng,
    )
    reliability, curves = evaluate_reliability(ensembles, 1 - args.alpha)
    regrets = build_regrets(reliability, selections)
    summary = summarize_regrets(
        regrets, args.bootstrap_reps, args.permutation_reps, rng
    )
    gate = build_gate(
        ensembles,
        selections,
        summary,
        args.minimum_seeds,
        expected_models,
        args.gate_scope,
        sample_alignment_ok and run_sample_alignment_ok,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(out_dir / "prediction_input_audit.csv", index=False)
    run_sample_alignment.to_csv(out_dir / "run_sample_alignment_audit.csv", index=False)
    availability.to_csv(out_dir / "model_seed_availability.csv", index=False)
    sample_alignment.to_csv(out_dir / "sample_alignment_audit.csv", index=False)
    validation_scores.to_csv(out_dir / "validation_seed_scores.csv", index=False)
    selections.to_csv(out_dir / "validation_selection_by_task.csv", index=False)
    selection_probabilities.to_csv(
        out_dir / "validation_selection_probabilities.csv", index=False
    )
    reliability.to_csv(out_dir / "test_reliability_by_task_model.csv", index=False)
    curves.to_csv(out_dir / "risk_coverage_by_sample.csv", index=False)
    regrets.to_csv(out_dir / "reliability_regret_by_endpoint.csv", index=False)
    summary.to_csv(out_dir / "reliability_regret_summary.csv", index=False)
    (out_dir / "reliability_regret_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"complete runs: {len(runs)}")
    print(f"eligible task-model ensembles: {len(ensembles)}")
    print(f"selected endpoints: {len(selections)}")
    print(f"endpoint-dimension regrets: {len(regrets)}")
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
