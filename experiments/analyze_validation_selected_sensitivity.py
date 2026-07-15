"""Zero-refit sensitivity restricted to validation-selected models.

The primary manuscript artifacts contain two complementary analyses that average
over all five candidate models within an endpoint: conditional reliability and
validation-fitted calibration.  This post-hoc sensitivity instead freezes the
single model selected from validation performance for each endpoint and filters
the locked result tables to that model.  It performs no fitting, calibration,
prediction, threshold selection, or test-guided model selection.

Outputs are endpoint-level CSV tables, one compact LaTeX table, a technical
manifest, and a citation-ready interpretation with explicit limitations.
"""

from __future__ import annotations

# The portable Python 3.9 environment contains a dataclasses backport whose
# pandas import path expects typing._ClassVar.  This is a no-op elsewhere.
import typing

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT
    / "results"
    / "tables"
    / "reliability_regret_final"
    / "validation_selection_by_task.csv"
)
DEFAULT_CONDITIONAL = (
    ROOT
    / "results"
    / "tables"
    / "conditional_reliability_final"
    / "conditional_reliability_by_task_model.csv"
)
DEFAULT_CALIBRATION = ROOT / "results" / "tables" / "calibration_summary_final.csv"
DEFAULT_OUT_DIR = (
    ROOT / "results" / "tables" / "validation_selected_sensitivity_final"
)

DEFAULT_PRIMARY_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_PERMUTATION_REPS = 20000
DEFAULT_RANDOM_SEED = 20260715
EXPECTED_TASKS = 22
EXPECTED_MODELS = frozenset({"rf", "xgboost", "lightgbm", "gin", "chemprop_dmpnn"})
EXPECTED_SEEDS = frozenset({1, 2, 3, 4, 5})
EXPECTED_THRESHOLDS = (0.3, 0.5, 0.7)
CLASSIFICATION_METHODS = ("isotonic", "platt")
REGRESSION_METHOD = "conformal_90"

CONDITIONAL_METRICS = (
    "log_error_ratio_outside_inside",
    "uq_correlation_advantage_inside",
    "selective_gain_advantage_inside",
    "ece_gap_outside_inside",
    "brier_gap_outside_inside",
    "coverage_error_gap_outside_inside",
)

CONDITIONAL_LABELS = {
    "log_error_ratio_outside_inside": "Log outside/inside absolute error",
    "uq_correlation_advantage_inside": "UQ--error correlation advantage inside",
    "selective_gain_advantage_inside": "Selective-gain advantage inside",
    "ece_gap_outside_inside": "ECE outside--inside",
    "brier_gap_outside_inside": "Brier outside--inside",
    "coverage_error_gap_outside_inside": "Coverage-error outside--inside",
}

CALIBRATION_EFFECTS = (
    ("isotonic", "ece_gain"),
    ("isotonic", "brier_gain"),
    ("platt", "ece_gain"),
    ("platt", "brier_gain"),
)

CALIBRATION_LABELS = {
    ("isotonic", "ece_gain"): "Isotonic ECE gain",
    ("isotonic", "brier_gain"): "Isotonic Brier gain",
    ("platt", "ece_gain"): "Logistic-on-$p$ ECE gain",
    ("platt", "brier_gain"): "Logistic-on-$p$ Brier gain",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_random_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )


def bootstrap_mean_interval(
    values: Iterable[float], reps: int, random_seed: int
) -> tuple[float, float]:
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Cannot bootstrap an empty finite endpoint set")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(array), size=(reps, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_pvalue(values: Iterable[float], reps: int, random_seed: int) -> float:
    if reps <= 0:
        raise ValueError("permutation reps must be positive")
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Cannot sign-flip an empty finite endpoint set")
    observed = abs(float(np.mean(array)))
    rng = np.random.default_rng(random_seed)
    exceedances = 0
    # Chunk to avoid a large reps-by-endpoints allocation on constrained systems.
    remaining = reps
    while remaining:
        chunk = min(remaining, 4096)
        signs = rng.choice((-1.0, 1.0), size=(chunk, len(array)))
        null = np.abs(np.mean(signs * array[None, :], axis=1))
        exceedances += int(np.sum(null >= observed))
        remaining -= chunk
    return float((1 + exceedances) / (reps + 1))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().astype(float).sort_values(kind="mergesort")
    running_max = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        candidate = min(1.0, (total - rank) * float(value))
        running_max = max(running_max, candidate)
        adjusted.loc[index] = running_max
    return adjusted


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _normalize_text(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column in columns:
        result[column] = result[column].astype(str).str.strip().str.lower()
        if result[column].eq("").any():
            raise ValueError(f"Empty normalized value in {column}")
    return result


def normalize_selection(selection: pd.DataFrame) -> pd.DataFrame:
    required = (
        "task",
        "task_type",
        "official_metric",
        "validation_selected_model",
    )
    _require_columns(selection, required, "validation selection")
    frame = _normalize_text(selection, required)
    if frame.duplicated("task").any():
        duplicates = sorted(frame.loc[frame.duplicated("task", keep=False), "task"].unique())
        raise ValueError(f"Validation selection has duplicate tasks: {duplicates}")
    return frame.sort_values("task", kind="mergesort").reset_index(drop=True)


def selected_conditional_rows(
    selection: pd.DataFrame, conditional: pd.DataFrame
) -> pd.DataFrame:
    selected = normalize_selection(selection)
    required = (
        "task",
        "model",
        "task_type",
        "official_metric",
        "n_seeds",
        "threshold",
        "inside_count",
        "outside_count",
        "eligible",
        *CONDITIONAL_METRICS,
    )
    _require_columns(conditional, required, "conditional reliability")
    frame = _normalize_text(
        conditional, ("task", "model", "task_type", "official_metric")
    )
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="raise")
    if not np.isfinite(frame["threshold"].to_numpy(dtype=float)).all():
        raise ValueError("Conditional thresholds must be finite")
    key = ["task", "model", "threshold"]
    if frame.duplicated(key).any():
        raise ValueError("Conditional reliability has duplicate task/model/threshold rows")

    metadata = selected[
        ["task", "task_type", "official_metric", "validation_selected_model"]
    ].rename(
        columns={
            "task_type": "selection_task_type",
            "official_metric": "selection_official_metric",
        }
    )
    merged = frame.merge(metadata, on="task", how="inner", validate="many_to_one")
    if merged["task"].nunique() != len(selected):
        missing = sorted(set(selected["task"]) - set(merged["task"]))
        raise ValueError(f"Selected tasks missing from conditional reliability: {missing}")
    selected_rows = merged[merged["model"] == merged["validation_selected_model"]].copy()
    if selected_rows.empty:
        raise ValueError("No validation-selected models matched conditional reliability")
    if not selected_rows["task_type"].eq(selected_rows["selection_task_type"]).all():
        raise ValueError("Task-type mismatch between selection and conditional tables")
    if not selected_rows["official_metric"].eq(
        selected_rows["selection_official_metric"]
    ).all():
        raise ValueError("Official-metric mismatch between selection and conditional tables")
    thresholds = sorted(selected_rows["threshold"].unique())
    expected_keys = pd.MultiIndex.from_product(
        [selected["task"].tolist(), thresholds], names=["task", "threshold"]
    )
    actual_keys = pd.MultiIndex.from_frame(selected_rows[["task", "threshold"]])
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != set(expected_keys):
        raise ValueError("Each task must have exactly one selected conditional row per threshold")

    selected_rows = selected_rows.drop(
        columns=["selection_task_type", "selection_official_metric"]
    ).rename(columns={"model": "selected_model"})
    ordered = [
        "task",
        "selected_model",
        "task_type",
        "official_metric",
        "n_seeds",
        "threshold",
        "inside_count",
        "outside_count",
        "eligible",
        *CONDITIONAL_METRICS,
    ]
    return selected_rows[ordered].sort_values(
        ["threshold", "task"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_selected_conditional(
    selected_rows: pd.DataFrame,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rows: list[dict] = []
    eligible = selected_rows[selected_rows["eligible"].astype(bool)].copy()
    for threshold, threshold_rows in eligible.groupby("threshold", sort=True):
        selected_at_threshold = selected_rows[np.isclose(selected_rows["threshold"], threshold)]
        for metric in CONDITIONAL_METRICS:
            values = pd.to_numeric(threshold_rows[metric], errors="coerce")
            finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
            if len(finite) == 0:
                continue
            low, high = bootstrap_mean_interval(
                finite,
                bootstrap_reps,
                stable_random_seed(random_seed, "conditional", threshold, metric, "bootstrap"),
            )
            pvalue = sign_flip_pvalue(
                finite,
                permutation_reps,
                stable_random_seed(random_seed, "conditional", threshold, metric, "sign_flip"),
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "metric": metric,
                    "selected_endpoint_rows": int(len(selected_at_threshold)),
                    "eligible_endpoints": int(len(finite)),
                    "excluded_or_undefined_endpoints": int(
                        len(selected_at_threshold) - len(finite)
                    ),
                    "positive_count": int(np.sum(finite > 0)),
                    "positive_fraction": float(np.mean(finite > 0)),
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "sign_flip_p": pvalue,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No finite selected conditional effects were available")
    summary["holm_p_within_threshold"] = summary.groupby(
        "threshold", group_keys=False, sort=True
    )["sign_flip_p"].transform(holm_adjust)
    return summary.sort_values(["threshold", "metric"], kind="mergesort").reset_index(
        drop=True
    )


def selected_calibration_endpoint_rows(
    selection: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    selected = normalize_selection(selection)
    required = (
        "task",
        "task_type",
        "official_metric",
        "model",
        "seed",
        "method",
        "test_ece_before",
        "test_ece_after",
        "test_brier_before",
        "test_brier_after",
        "coverage",
        "mean_interval_width",
    )
    _require_columns(calibration, required, "calibration summary")
    frame = _normalize_text(
        calibration, ("task", "task_type", "official_metric", "model", "method")
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    key = ["task", "model", "seed", "method"]
    if frame.duplicated(key).any():
        raise ValueError("Calibration summary has duplicate task/model/seed/method rows")

    metadata = selected[
        ["task", "task_type", "official_metric", "validation_selected_model"]
    ].rename(
        columns={
            "task_type": "selection_task_type",
            "official_metric": "selection_official_metric",
        }
    )
    merged = frame.merge(metadata, on="task", how="inner", validate="many_to_one")
    if merged["task"].nunique() != len(selected):
        missing = sorted(set(selected["task"]) - set(merged["task"]))
        raise ValueError(f"Selected tasks missing from calibration summary: {missing}")
    selected_runs = merged[merged["model"] == merged["validation_selected_model"]].copy()
    if not selected_runs["task_type"].eq(selected_runs["selection_task_type"]).all():
        raise ValueError("Task-type mismatch between selection and calibration tables")
    if not selected_runs["official_metric"].eq(
        selected_runs["selection_official_metric"]
    ).all():
        raise ValueError("Official-metric mismatch between selection and calibration tables")

    classification = selected_runs[selected_runs["task_type"] == "classification"].copy()
    if set(classification["method"].unique()) != set(CLASSIFICATION_METHODS):
        raise ValueError("Selected classification calibration must contain isotonic and platt")
    # Before-calibration metrics are repeated across the two intervention rows and
    # must agree exactly within every endpoint/model/seed.
    before_consistency = classification.groupby(
        ["task", "model", "seed"], sort=True
    )[["test_ece_before", "test_brier_before"]].nunique(dropna=False)
    if (before_consistency > 1).any().any():
        raise ValueError("Uncalibrated metrics disagree between calibration methods")
    classification["ece_gain"] = (
        classification["test_ece_before"] - classification["test_ece_after"]
    )
    classification["brier_gain"] = (
        classification["test_brier_before"] - classification["test_brier_after"]
    )
    class_grouped = (
        classification.groupby(
            ["task", "task_type", "official_metric", "model", "method"],
            as_index=False,
            sort=True,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            ece_before=("test_ece_before", "mean"),
            ece_after=("test_ece_after", "mean"),
            ece_gain=("ece_gain", "mean"),
            brier_before=("test_brier_before", "mean"),
            brier_after=("test_brier_after", "mean"),
            brier_gain=("brier_gain", "mean"),
        )
        .rename(columns={"model": "selected_model"})
    )

    regression = selected_runs[selected_runs["task_type"] == "regression"].copy()
    if set(regression["method"].unique()) != {REGRESSION_METHOD}:
        raise ValueError("Selected regression calibration must contain conformal_90 only")
    reg_grouped = (
        regression.groupby(
            ["task", "task_type", "official_metric", "model", "method"],
            as_index=False,
            sort=True,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            coverage=("coverage", "mean"),
            mean_interval_width=("mean_interval_width", "mean"),
        )
        .rename(columns={"model": "selected_model"})
    )
    reg_grouped["absolute_coverage_error"] = np.abs(reg_grouped["coverage"] - 0.9)

    columns = [
        "task",
        "selected_model",
        "task_type",
        "official_metric",
        "method",
        "n_seeds",
        "ece_before",
        "ece_after",
        "ece_gain",
        "brier_before",
        "brier_after",
        "brier_gain",
        "coverage",
        "absolute_coverage_error",
        "mean_interval_width",
    ]
    combined = pd.concat([class_grouped, reg_grouped], ignore_index=True, sort=False)
    return combined[columns].sort_values(
        ["task_type", "method", "task"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_selected_calibration(
    endpoint_rows: pd.DataFrame,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rows: list[dict] = []
    classification = endpoint_rows[endpoint_rows["task_type"] == "classification"]
    for method, metric in CALIBRATION_EFFECTS:
        subset = classification[classification["method"] == method]
        values = pd.to_numeric(subset[metric], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
        if len(finite) == 0:
            raise ValueError(f"No finite selected calibration values for {method}/{metric}")
        low, high = bootstrap_mean_interval(
            finite,
            bootstrap_reps,
            stable_random_seed(random_seed, "calibration", method, metric, "bootstrap"),
        )
        pvalue = sign_flip_pvalue(
            finite,
            permutation_reps,
            stable_random_seed(random_seed, "calibration", method, metric, "sign_flip"),
        )
        rows.append(
            {
                "method": method,
                "metric": metric,
                "eligible_endpoints": int(len(finite)),
                "improved_count": int(np.sum(finite > 0)),
                "improved_fraction": float(np.mean(finite > 0)),
                "mean_gain": float(np.mean(finite)),
                "median_gain": float(np.median(finite)),
                "mean_gain_ci_low": low,
                "mean_gain_ci_high": high,
                "sign_flip_p": pvalue,
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p_across_four_effects"] = holm_adjust(summary["sign_flip_p"])
    return summary.reset_index(drop=True)


def summarize_selected_conformal(
    endpoint_rows: pd.DataFrame,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    regression = endpoint_rows[endpoint_rows["task_type"] == "regression"].copy()
    values = pd.to_numeric(regression["absolute_coverage_error"], errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if len(finite) == 0:
        raise ValueError("No finite selected validation-residual coverage values")
    low, high = bootstrap_mean_interval(
        finite,
        bootstrap_reps,
        stable_random_seed(random_seed, "conformal", "coverage_error", "bootstrap"),
    )
    coverage = regression["coverage"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "method": REGRESSION_METHOD,
                "eligible_endpoints": int(len(finite)),
                "mean_coverage": float(np.mean(coverage)),
                "mean_absolute_coverage_error": float(np.mean(finite)),
                "median_absolute_coverage_error": float(np.median(finite)),
                "mean_absolute_coverage_error_ci_low": low,
                "mean_absolute_coverage_error_ci_high": high,
                "coverage_below_0_9_count": int(np.sum(coverage < 0.9)),
                "coverage_equal_0_9_count": int(np.sum(coverage == 0.9)),
                "coverage_above_0_9_count": int(np.sum(coverage > 0.9)),
                "within_0_05_of_0_9_count": int(np.sum(np.abs(coverage - 0.9) <= 0.05)),
            }
        ]
    )


def validate_final_contract(
    selection: pd.DataFrame,
    conditional_source: pd.DataFrame,
    calibration_source: pd.DataFrame,
    selected_conditional: pd.DataFrame,
    selected_calibration: pd.DataFrame,
) -> None:
    selected = normalize_selection(selection)
    if len(selected) != EXPECTED_TASKS or selected["task"].nunique() != EXPECTED_TASKS:
        raise ValueError("Final selection contract requires exactly 22 endpoints")
    if not set(selected["validation_selected_model"]).issubset(EXPECTED_MODELS):
        raise ValueError("Final selection contains a model outside the locked five-model pool")

    conditional_models = set(
        conditional_source["model"].astype(str).str.strip().str.lower().unique()
    )
    if conditional_models != EXPECTED_MODELS or len(conditional_source) != 330:
        raise ValueError("Final conditional source must be the complete 22x5x3 matrix")
    conditional_contract = conditional_source.copy()
    conditional_contract["task"] = conditional_contract["task"].astype(str).str.strip().str.lower()
    conditional_contract["model"] = conditional_contract["model"].astype(str).str.strip().str.lower()
    if conditional_contract["task"].nunique() != EXPECTED_TASKS:
        raise ValueError("Final conditional source must cover exactly 22 endpoints")
    if not conditional_contract.groupby("task")["model"].nunique().eq(5).all():
        raise ValueError("Every final conditional endpoint must contain all five models")
    if not conditional_contract.groupby(["task", "model"]).size().eq(3).all():
        raise ValueError("Every final conditional task/model must contain three thresholds")
    thresholds = sorted(float(value) for value in selected_conditional["threshold"].unique())
    if not np.allclose(thresholds, EXPECTED_THRESHOLDS, rtol=0, atol=1e-12):
        raise ValueError("Final selected conditional thresholds must be 0.3, 0.5, and 0.7")
    if len(selected_conditional) != 66:
        raise ValueError("Final selected conditional table must contain 22x3 rows")
    if set(pd.to_numeric(selected_conditional["n_seeds"], errors="raise")) != {5}:
        raise ValueError("Every selected conditional endpoint must use five seeds")

    calibration_models = set(
        calibration_source["model"].astype(str).str.strip().str.lower().unique()
    )
    if calibration_models != EXPECTED_MODELS or len(calibration_source) != 875:
        raise ValueError("Final calibration source must contain the complete 875-row matrix")
    seed_set = set(pd.to_numeric(calibration_source["seed"], errors="raise").astype(int))
    if seed_set != EXPECTED_SEEDS:
        raise ValueError("Final calibration source must contain seeds 1--5")
    calibration_contract = calibration_source.copy()
    for column in ("task", "task_type", "model", "method"):
        calibration_contract[column] = (
            calibration_contract[column].astype(str).str.strip().str.lower()
        )
    class_contract = calibration_contract[
        calibration_contract["task_type"] == "classification"
    ]
    reg_contract = calibration_contract[calibration_contract["task_type"] == "regression"]
    if class_contract["task"].nunique() != 13 or reg_contract["task"].nunique() != 9:
        raise ValueError("Final calibration source must cover 13 classification and 9 regression endpoints")
    if not class_contract.groupby(["task", "model"]).size().eq(10).all():
        raise ValueError("Every classification task/model must contain five seeds for two mappings")
    if not reg_contract.groupby(["task", "model"]).size().eq(5).all():
        raise ValueError("Every regression task/model must contain five validation-residual rows")
    if len(selected_calibration) != 35:
        raise ValueError("Final selected calibration table must contain 26 mapping and 9 interval rows")
    if set(selected_calibration["n_seeds"].astype(int)) != {5}:
        raise ValueError("Every selected calibration endpoint must use five seeds")


def _format_interval(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


def render_latex(
    conditional_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    primary_threshold: float,
) -> str:
    primary = conditional_summary[
        np.isclose(conditional_summary["threshold"], primary_threshold)
    ].set_index("metric")
    lines = [
        "% Generated by experiments/analyze_validation_selected_sensitivity.py",
        "% Post-hoc zero-refit sensitivity; not independent validation.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Post-hoc sensitivity retaining only the validation-selected model for each endpoint. Positive conditional values mean greater outside-support loss or stronger inside-support utility; positive calibration gains mean lower loss after validation-fitted mapping. Intervals resample endpoints. This zero-refit diagnostic is not independent or prospective validation.}",
        r"\label{tab:validation-selected-sensitivity}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Panel & Effect & $n$ & Positive/improved & Mean & 95\% interval & Holm $p$ \\",
        r"\midrule",
    ]
    for metric in CONDITIONAL_METRICS:
        if metric not in primary.index:
            continue
        row = primary.loc[metric]
        lines.append(
            "Conditional & "
            + CONDITIONAL_LABELS[metric]
            + f" & {int(row['eligible_endpoints'])}"
            + f" & {int(row['positive_count'])}/{int(row['eligible_endpoints'])}"
            + f" & {float(row['mean']):.4f}"
            + f" & {_format_interval(float(row['mean_ci_low']), float(row['mean_ci_high']))}"
            + f" & {float(row['holm_p_within_threshold']):.4g} \\\\"
        )
    lines.append(r"\midrule")
    summary_index = calibration_summary.set_index(["method", "metric"])
    for key in CALIBRATION_EFFECTS:
        row = summary_index.loc[key]
        lines.append(
            "Calibration & "
            + CALIBRATION_LABELS[key]
            + f" & {int(row['eligible_endpoints'])}"
            + f" & {int(row['improved_count'])}/{int(row['eligible_endpoints'])}"
            + f" & {float(row['mean_gain']):.4f}"
            + f" & {_format_interval(float(row['mean_gain_ci_low']), float(row['mean_gain_ci_high']))}"
            + f" & {float(row['holm_p_across_four_effects']):.4g} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_interpretation(
    selected_conditional_rows_frame: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    conformal_summary: pd.DataFrame,
    primary_threshold: float,
) -> str:
    conditional = conditional_summary[
        np.isclose(conditional_summary["threshold"], primary_threshold)
    ].set_index("metric")
    calibration = calibration_summary.set_index(["method", "metric"])
    error = conditional.loc["log_error_ratio_outside_inside"]
    brier = conditional.loc["brier_gap_outside_inside"]
    ece = conditional.loc["ece_gap_outside_inside"]
    coverage = conditional.loc["coverage_error_gap_outside_inside"]
    isotonic_ece = calibration.loc[("isotonic", "ece_gain")]
    isotonic_brier = calibration.loc[("isotonic", "brier_gain")]
    platt_ece = calibration.loc[("platt", "ece_gain")]
    platt_brier = calibration.loc[("platt", "brier_gain")]
    conformal = conformal_summary.iloc[0]
    ratio = math.exp(float(error["mean"]))
    classification_n = int(isotonic_ece["eligible_endpoints"])
    regression_n = int(conformal["eligible_endpoints"])
    primary_rows = selected_conditional_rows_frame[
        np.isclose(selected_conditional_rows_frame["threshold"], primary_threshold)
    ]
    base_ineligible = primary_rows[~primary_rows["eligible"].astype(bool)]
    if base_ineligible.empty:
        eligibility_limit = (
            f"All selected endpoint rows satisfied the group-size rule at threshold "
            f"{primary_threshold:.1f}."
        )
    else:
        details = "; ".join(
            f"{row.task} (inside {int(row.inside_count)}, outside {int(row.outside_count)})"
            for row in base_ineligible.itertuples(index=False)
        )
        eligibility_limit = (
            f"The group-size rule excluded {details} at threshold {primary_threshold:.1f}; "
            f"the absolute-error denominator is therefore {int(error['eligible_endpoints'])} "
            f"rather than {len(primary_rows)}."
        )
    return f"""# Validation-selected-model sensitivity

## Citation-ready result

In a zero-refit sensitivity retaining only the model selected from validation performance for each endpoint, outside-support absolute error was higher for {int(error['positive_count'])}/{int(error['eligible_endpoints'])} eligible endpoints at the locked maximum-Tanimoto threshold {primary_threshold:.1f} (mean log outside/inside ratio {float(error['mean']):.4f}, 95% endpoint-bootstrap interval {_format_interval(float(error['mean_ci_low']), float(error['mean_ci_high']))}; geometric-mean ratio {ratio:.3f}). Brier score was worse outside support for {int(brier['positive_count'])}/{int(brier['eligible_endpoints'])} classification endpoints (mean gap {float(brier['mean']):.4f}, {_format_interval(float(brier['mean_ci_low']), float(brier['mean_ci_high']))}), whereas the ECE contrast was positive for only {int(ece['positive_count'])}/{int(ece['eligible_endpoints'])} and the validation-residual coverage-error contrast for {int(coverage['positive_count'])}/{int(coverage['eligible_endpoints'])}. Among the {classification_n} selected classification models, validation-fitted isotonic mapping improved ten-bin ECE on {int(isotonic_ece['improved_count'])}/{classification_n} endpoint means but Brier score on {int(isotonic_brier['improved_count'])}/{classification_n}; logistic-on-probability mapping improved ECE and Brier score on {int(platt_ece['improved_count'])}/{classification_n} and {int(platt_brier['improved_count'])}/{classification_n}, respectively. Across the {regression_n} selected regression models, mean absolute deviation from 90% validation-residual interval coverage was {float(conformal['mean_absolute_coverage_error']):.4f} ({_format_interval(float(conformal['mean_absolute_coverage_error_ci_low']), float(conformal['mean_absolute_coverage_error_ci_high']))}).

## Interpretation

The selected-only conditional result reproduces the headline directions for outside-support error ({int(error['positive_count'])}/{int(error['eligible_endpoints'])}) and Brier score ({int(brier['positive_count'])}/{int(brier['eligible_endpoints'])}) without averaging unselected candidates. In the locked final matrix, the calibration sensitivity is materially less universal than the all-model average: selected-only isotonic ECE improvement is {int(isotonic_ece['improved_count'])}/{classification_n}. The defensible conclusion is therefore that calibration often helps ECE but is endpoint- and scoring-rule-dependent; it must not be described as universal across the validation-selected models evaluated here.

## Required limitations

- This is an explicitly post-hoc, zero-refit sensitivity computed from locked result tables; it did not alter the selector, predictions, support threshold, or calibration mappings.
- Model identity was fixed by mean validation performance before these test summaries were filtered. The analysis does not quantify uncertainty in that single selected identity; the existing validation-bootstrap selection probabilities and tie sensitivity remain necessary context.
- The endpoint, not the seed or model, is the summary and resampling unit. Endpoints share molecules and curation dependencies, so percentile intervals, sign-flip diagnostics, and Holm values are collection-conditional descriptive quantities rather than independent-assay inference.
- The primary conditional threshold requires at least 20 inside- and 20 outside-support molecules. {eligibility_limit}
- Calibration mappings were fitted on validation labels that were also used for checkpointing and model selection. They are not based on an independent calibration cohort. Scaffold shift challenges exchangeability, and the validation-residual intervals carry no split-conformal finite-sample guarantee here.
- All outcomes come from fixed public benchmark tests. They establish neither prospective, external, experimental, clinical, nor regulatory validity and should not be used to choose among models with test labels.
"""


def _json_default(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("Non-finite numeric value cannot enter formal JSON")
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_json(path: Path, value: Mapping | Sequence) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _logical_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def run_analysis(
    selection_path: Path = DEFAULT_SELECTION,
    conditional_path: Path = DEFAULT_CONDITIONAL,
    calibration_path: Path = DEFAULT_CALIBRATION,
    out_dir: Path = DEFAULT_OUT_DIR,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    strict_final_contract: bool = True,
) -> dict:
    input_paths = [selection_path, conditional_path, calibration_path]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_before = {
        path: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in input_paths
    }
    selection = pd.read_csv(selection_path)
    conditional_source = pd.read_csv(conditional_path)
    calibration_source = pd.read_csv(calibration_path)

    selected_conditional = selected_conditional_rows(selection, conditional_source)
    conditional_summary = summarize_selected_conditional(
        selected_conditional,
        bootstrap_reps=bootstrap_reps,
        permutation_reps=permutation_reps,
        random_seed=random_seed,
    )
    selected_calibration = selected_calibration_endpoint_rows(selection, calibration_source)
    calibration_summary = summarize_selected_calibration(
        selected_calibration,
        bootstrap_reps=bootstrap_reps,
        permutation_reps=permutation_reps,
        random_seed=random_seed,
    )
    conformal_summary = summarize_selected_conformal(
        selected_calibration,
        bootstrap_reps=bootstrap_reps,
        random_seed=random_seed,
    )
    if strict_final_contract:
        validate_final_contract(
            selection,
            conditional_source,
            calibration_source,
            selected_conditional,
            selected_calibration,
        )
    if not np.any(np.isclose(conditional_summary["threshold"], primary_threshold)):
        raise ValueError("Primary threshold is absent from selected conditional summary")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "selected_conditional_reliability_by_endpoint.csv": selected_conditional,
        "selected_conditional_reliability_summary.csv": conditional_summary,
        "selected_calibration_by_endpoint_method.csv": selected_calibration,
        "selected_calibration_gain_summary.csv": calibration_summary,
        "selected_conformal_summary.csv": conformal_summary,
    }
    for filename, frame in output_frames.items():
        write_csv(out_dir / filename, frame)
    latex_path = out_dir / "table_supp_validation_selected_sensitivity.tex"
    latex_path.write_text(
        render_latex(conditional_summary, calibration_summary, primary_threshold),
        encoding="utf-8",
    )
    interpretation_path = out_dir / "VALIDATION_SELECTED_SENSITIVITY.md"
    interpretation_path.write_text(
        build_interpretation(
            selected_conditional,
            conditional_summary,
            calibration_summary,
            conformal_summary,
            primary_threshold,
        ),
        encoding="utf-8",
    )

    input_after = {
        path: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in input_paths
    }
    if input_before != input_after:
        raise RuntimeError("A locked source artifact changed during selected-only analysis")

    primary = conditional_summary[
        np.isclose(conditional_summary["threshold"], primary_threshold)
    ].set_index("metric")
    calibration_index = calibration_summary.set_index(["method", "metric"])
    headline_counts = {
        "outside_support_error_positive_endpoints": {
            "positive": int(primary.loc["log_error_ratio_outside_inside", "positive_count"]),
            "eligible": int(primary.loc["log_error_ratio_outside_inside", "eligible_endpoints"]),
        },
        "outside_support_brier_positive_endpoints": {
            "positive": int(primary.loc["brier_gap_outside_inside", "positive_count"]),
            "eligible": int(primary.loc["brier_gap_outside_inside", "eligible_endpoints"]),
        },
        "isotonic_ece_improved_endpoints": {
            "improved": int(calibration_index.loc[("isotonic", "ece_gain"), "improved_count"]),
            "eligible": int(calibration_index.loc[("isotonic", "ece_gain"), "eligible_endpoints"]),
        },
        "isotonic_brier_improved_endpoints": {
            "improved": int(calibration_index.loc[("isotonic", "brier_gain"), "improved_count"]),
            "eligible": int(calibration_index.loc[("isotonic", "brier_gain"), "eligible_endpoints"]),
        },
        "logistic_on_probability_ece_improved_endpoints": {
            "improved": int(calibration_index.loc[("platt", "ece_gain"), "improved_count"]),
            "eligible": int(calibration_index.loc[("platt", "ece_gain"), "eligible_endpoints"]),
        },
        "logistic_on_probability_brier_improved_endpoints": {
            "improved": int(calibration_index.loc[("platt", "brier_gain"), "improved_count"]),
            "eligible": int(calibration_index.loc[("platt", "brier_gain"), "eligible_endpoints"]),
        },
    }
    manifest_path = out_dir / "validation_selected_sensitivity_manifest.json"
    preliminary_outputs = [
        *[out_dir / filename for filename in output_frames],
        latex_path,
        interpretation_path,
    ]
    manifest = {
        "technical_status": "PASS",
        "evidence_status": "post_hoc_zero_refit_descriptive_sensitivity",
        "selection_policy": (
            "one model per endpoint frozen from mean oriented validation performance; "
            "no test-guided reselection"
        ),
        "analysis_unit": "endpoint",
        "primary_threshold": primary_threshold,
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps": permutation_reps,
        "random_seed": random_seed,
        "strict_final_contract": strict_final_contract,
        "checks": {
            "source_inputs_sha256_and_size_unchanged": True,
            "zero_refit": True,
            "one_validation_selected_model_per_endpoint": True,
            "endpoint_level_aggregation": True,
            "selected_conditional_rows": int(len(selected_conditional)),
            "selected_calibration_endpoint_method_rows": int(len(selected_calibration)),
        },
        "headline_counts": headline_counts,
        "input_files": [
            {
                "path": _logical_path(path),
                "bytes": input_before[path]["bytes"],
                "sha256": input_before[path]["sha256"],
            }
            for path in input_paths
        ],
        "output_files": [
            {
                "path": _logical_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": (
                    int(len(output_frames[path.name]))
                    if path.name in output_frames
                    else None
                ),
            }
            for path in preliminary_outputs
        ],
        "limitations": [
            "post-hoc sensitivity, not a new primary analysis",
            "selection identity uncertainty is not propagated",
            "shared molecules make endpoints statistically dependent",
            "validation labels were reused for checkpointing, selection, and calibration",
            "fixed public tests are neither external nor prospective validation",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter locked conditional-reliability and calibration results to the "
            "validation-selected model for each endpoint; performs no training."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--conditional", type=Path, default=DEFAULT_CONDITIONAL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--permutation-reps", type=int, default=DEFAULT_PERMUTATION_REPS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--allow-nonfinal-contract",
        action="store_true",
        help="Permit synthetic or incomplete inputs; never use for the formal release run.",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        selection_path=args.selection,
        conditional_path=args.conditional,
        calibration_path=args.calibration,
        out_dir=args.out_dir,
        primary_threshold=args.primary_threshold,
        bootstrap_reps=args.bootstrap_reps,
        permutation_reps=args.permutation_reps,
        random_seed=args.random_seed,
        strict_final_contract=not args.allow_nonfinal_contract,
    )
    print(f"technical_status: {manifest['technical_status']}")
    print(f"headline_counts: {json.dumps(manifest['headline_counts'], sort_keys=True)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
