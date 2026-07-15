from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
)
from experiments.analyze_reliability_regret import (
    LEGACY_EXCLUDED_MODELS,
    _load_similarity_for_test,
    _normalize_prediction_file,
    _similarity_cache_path,
    sample_rows_match,
)


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


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
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_positive = int(np.sum(y_true == 1))
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ordered_labels = y_true[order]
    ordered_scores = y_score[order]
    cumulative_positive = np.cumsum(ordered_labels == 1)
    # Average precision must treat tied scores as one threshold. Otherwise the
    # result depends on the arbitrary order of examples within a tie.
    threshold_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0), len(ordered_scores) - 1
    ]
    true_positive = cumulative_positive[threshold_ends]
    predicted_positive = threshold_ends + 1
    precision = true_positive / predicted_positive
    previous_true_positive = np.r_[0, true_positive[:-1]]
    recall_increment = (true_positive - previous_true_positive) / n_positive
    return float(np.sum(recall_increment * precision))


def official_metric(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
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
    return -float(value) if str(metric).lower() == "mae" else float(value)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
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
    n = len(residuals)
    if n == 0:
        return float("nan")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    try:
        return float(np.quantile(residuals, level, method="higher"))
    except TypeError:
        return float(np.quantile(residuals, level, interpolation="higher"))


def selective_gain(
    group: pd.DataFrame,
    metric: str,
    rejection_fraction: float,
) -> float:
    if len(group) < 3:
        return float("nan")
    keep = max(2, int(round(len(group) * (1 - rejection_fraction))))
    selected = group.sort_values("uncertainty", ascending=True).iloc[:keep]
    baseline = official_metric(metric, group["y_true"].to_numpy(), group["prediction"].to_numpy())
    retained = official_metric(
        metric, selected["y_true"].to_numpy(), selected["prediction"].to_numpy()
    )
    if not np.isfinite(baseline) or not np.isfinite(retained):
        return float("nan")
    return orient_metric(metric, retained) - orient_metric(metric, baseline)


def bootstrap_interval(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = np.empty(reps, dtype=float)
    for idx in range(reps):
        draws[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
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


def find_prediction_files(prediction_dir: Path) -> list[Path]:
    files = []
    for path in prediction_dir.rglob("*.csv"):
        if path.name.startswith("."):
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        required = {"task", "model", "seed", "split", "sample_index", "y_true", "y_pred"}
        neural = {"split", "smiles", "label", "prediction"}
        if required.issubset(columns) or neural.issubset(columns):
            files.append(path)
    return sorted(files)


def normalize_prediction_file(path: Path) -> pd.DataFrame:
    frame, reason = _normalize_prediction_file(path)
    if frame is None:
        raise ValueError(f"Invalid prediction artifact {path}: {reason}")
    if set(frame["model"].astype(str)) & LEGACY_EXCLUDED_MODELS:
        return pd.DataFrame()
    return frame


def load_ensemble_groups(
    prediction_dirs: list[Path],
    similarity_dir: Path,
    minimum_seeds: int,
    similarity_split_source: str = PYTDC_SPLIT_SOURCE,
    require_similarity: bool = False,
) -> list[tuple[dict, pd.DataFrame]]:
    frames: list[pd.DataFrame] = []
    observed_runs: dict[tuple[str, str, int], Path] = {}
    for prediction_dir in prediction_dirs:
        for path in find_prediction_files(prediction_dir):
            frame = normalize_prediction_file(path)
            if frame.empty:
                continue
            identities = frame[["task", "model", "seed"]].drop_duplicates()
            if len(identities) != 1:
                raise ValueError(f"Mixed run metadata in {path}")
            identity = identities.iloc[0]
            key = (
                str(identity["task"]).lower(),
                str(identity["model"]).lower(),
                int(identity["seed"]),
            )
            if key in observed_runs:
                raise ValueError(
                    f"Duplicate prediction run {key}: {observed_runs[key]} and {path}"
                )
            observed_runs[key] = path
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No prediction CSV files found under: "
            + ", ".join(str(path) for path in prediction_dirs)
        )
    predictions = pd.concat(frames, ignore_index=True)

    # The formal pipeline must never compare models on intersected or shifted
    # validation/test rows. Check exact identity before constructing ensembles.
    for (task, seed), task_seed in predictions.groupby(["task", "seed"], sort=True):
        model_frames = [
            group for _, group in task_seed.groupby("model", sort=True)
        ]
        reference = model_frames[0]
        for split_name in ("valid", "test"):
            reference_split = reference[reference["split"].eq(split_name)]
            for candidate in model_frames[1:]:
                candidate_split = candidate[candidate["split"].eq(split_name)]
                if not sample_rows_match(candidate_split, reference_split):
                    raise ValueError(
                        "Prediction sample identity mismatch for "
                        f"task={task}, seed={seed}, split={split_name}"
                    )

    groups: list[tuple[dict, pd.DataFrame]] = []
    for (task, model), group in predictions.groupby(["task", "model"], sort=True):
        seeds = sorted(int(value) for value in group["seed"].dropna().unique())
        if len(seeds) < minimum_seeds:
            continue
        test_parts: list[pd.DataFrame] = []
        coverage_parts: list[pd.DataFrame] = []
        first = group.iloc[0]
        task_type = str(first["task_type"])
        metric = str(first["official_metric"])
        for seed in seeds:
            seed_frame = group[group["seed"] == seed]
            test = seed_frame[seed_frame["split"] == "test"].copy()
            if test.empty:
                continue
            similarity_path = _similarity_cache_path(
                similarity_dir,
                str(task),
                seed,
                similarity_split_source,
            )
            if not similarity_path.exists():
                if require_similarity:
                    raise FileNotFoundError(
                        "Required split-explicit applicability cache is missing: "
                        f"{similarity_path}"
                    )
                continue
            similarity, similarity_error = _load_similarity_for_test(
                similarity_path,
                test,
                expected_split_source=(
                    similarity_split_source if require_similarity else None
                ),
            )
            if similarity_error or similarity is None:
                if require_similarity:
                    raise ValueError(
                        f"Invalid required similarity cache {similarity_path}: "
                        f"{similarity_error}"
                    )
                continue
            test = test.merge(similarity, on="sample_index", how="inner")
            test_parts.append(
                test[["sample_index", "y_true", "y_pred", "max_tanimoto"]].assign(seed=seed)
            )
            if task_type == "regression":
                valid = seed_frame[seed_frame["split"] == "valid"]
                quantile = conformal_quantile(
                    np.abs(valid["y_true"].to_numpy(dtype=float) - valid["y_pred"].to_numpy(dtype=float))
                )
                test_coverage = test[["sample_index"]].copy()
                test_coverage["covered"] = (
                    np.abs(test["y_true"].to_numpy(dtype=float) - test["y_pred"].to_numpy(dtype=float))
                    <= quantile
                ).astype(float)
                coverage_parts.append(test_coverage.assign(seed=seed))
        if len(test_parts) < minimum_seeds:
            continue
        stacked = pd.concat(test_parts, ignore_index=True)
        ensemble = (
            stacked.groupby("sample_index", as_index=False)
            .agg(
                y_true=("y_true", "first"),
                prediction=("y_pred", "mean"),
                uncertainty=("y_pred", lambda values: float(np.std(values, ddof=1))),
                max_tanimoto=("max_tanimoto", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        if coverage_parts:
            coverage = pd.concat(coverage_parts, ignore_index=True)
            coverage = coverage.groupby("sample_index", as_index=False).agg(
                conformal_covered=("covered", "mean")
            )
            ensemble = ensemble.merge(coverage, on="sample_index", how="left")
        meta = {
            "task": task,
            "model": model,
            "task_type": task_type,
            "official_metric": metric,
            "n_seeds": int(ensemble["n_seeds"].min()),
        }
        groups.append((meta, ensemble))
    return groups


def domain_metrics(
    group: pd.DataFrame,
    task_type: str,
    official: str,
    rejection_fraction: float,
) -> dict:
    if group.empty:
        return {
            "count": 0,
            "mean_abs_error": float("nan"),
            "uncertainty_error_correlation": float("nan"),
            "selective_gain": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "conformal_coverage": float("nan"),
        }
    error = np.abs(group["y_true"].to_numpy(dtype=float) - group["prediction"].to_numpy(dtype=float))
    result = {
        "count": len(group),
        "mean_abs_error": float(np.mean(error)),
        "uncertainty_error_correlation": spearman(
            group["uncertainty"].to_numpy(dtype=float), error
        ),
        "selective_gain": selective_gain(group, official, rejection_fraction),
        "ece": float("nan"),
        "brier": float("nan"),
        "conformal_coverage": float("nan"),
    }
    if task_type == "classification":
        result["ece"] = expected_calibration_error(
            group["y_true"].to_numpy(dtype=int), group["prediction"].to_numpy(dtype=float)
        )
        result["brier"] = float(np.mean(error**2))
    elif "conformal_covered" in group:
        result["conformal_coverage"] = float(group["conformal_covered"].mean())
    return result


def analyze_groups(
    groups: list[tuple[dict, pd.DataFrame]],
    thresholds: list[float],
    rejection_fraction: float,
    minimum_domain_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for meta, ensemble in groups:
        for threshold in thresholds:
            inside = ensemble[ensemble["max_tanimoto"] >= threshold]
            outside = ensemble[ensemble["max_tanimoto"] < threshold]
            inside_metrics = domain_metrics(
                inside, meta["task_type"], meta["official_metric"], rejection_fraction
            )
            outside_metrics = domain_metrics(
                outside, meta["task_type"], meta["official_metric"], rejection_fraction
            )
            row = {**meta, "threshold": threshold}
            for key, value in inside_metrics.items():
                row[f"inside_{key}"] = value
            for key, value in outside_metrics.items():
                row[f"outside_{key}"] = value
            eligible = (
                inside_metrics["count"] >= minimum_domain_n
                and outside_metrics["count"] >= minimum_domain_n
            )
            row["eligible"] = eligible
            if eligible:
                epsilon = 1e-12
                row["log_error_ratio_outside_inside"] = float(
                    np.log(
                        (outside_metrics["mean_abs_error"] + epsilon)
                        / (inside_metrics["mean_abs_error"] + epsilon)
                    )
                )
                row["uq_correlation_advantage_inside"] = (
                    inside_metrics["uncertainty_error_correlation"]
                    - outside_metrics["uncertainty_error_correlation"]
                )
                row["selective_gain_advantage_inside"] = (
                    inside_metrics["selective_gain"] - outside_metrics["selective_gain"]
                )
                row["ece_gap_outside_inside"] = (
                    outside_metrics["ece"] - inside_metrics["ece"]
                )
                row["brier_gap_outside_inside"] = (
                    outside_metrics["brier"] - inside_metrics["brier"]
                )
                row["coverage_error_gap_outside_inside"] = (
                    abs(outside_metrics["conformal_coverage"] - 0.9)
                    - abs(inside_metrics["conformal_coverage"] - 0.9)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_model_profiles(
    groups: list[tuple[dict, pd.DataFrame]], rejection_fraction: float
) -> pd.DataFrame:
    profiles: list[dict] = []
    for meta, ensemble in groups:
        metrics = domain_metrics(
            ensemble, meta["task_type"], meta["official_metric"], rejection_fraction
        )
        metric_value = official_metric(
            meta["official_metric"],
            ensemble["y_true"].to_numpy(),
            ensemble["prediction"].to_numpy(),
        )
        error = np.abs(
            ensemble["y_true"].to_numpy(dtype=float)
            - ensemble["prediction"].to_numpy(dtype=float)
        )
        profiles.append(
            {
                **meta,
                "n_samples": len(ensemble),
                "performance": orient_metric(meta["official_metric"], metric_value),
                "ece": metrics["ece"],
                "brier": metrics["brier"],
                "coverage": metrics["conformal_coverage"],
                "coverage_error": (
                    abs(metrics["conformal_coverage"] - 0.9)
                    if np.isfinite(metrics["conformal_coverage"])
                    else float("nan")
                ),
                "selective_gain_30pct": metrics["selective_gain"],
                "applicability_alignment": spearman(
                    ensemble["max_tanimoto"].to_numpy(dtype=float), -error
                ),
                "uncertainty_error_correlation": metrics[
                    "uncertainty_error_correlation"
                ],
            }
        )
    return pd.DataFrame(profiles)


def summarize(
    rows: pd.DataFrame,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metrics = (
        "log_error_ratio_outside_inside",
        "uq_correlation_advantage_inside",
        "selective_gain_advantage_inside",
        "ece_gap_outside_inside",
        "brier_gap_outside_inside",
        "coverage_error_gap_outside_inside",
    )
    output: list[dict] = []
    for threshold, threshold_rows in rows[rows["eligible"]].groupby("threshold", sort=True):
        for metric in metrics:
            if metric not in threshold_rows:
                continue
            pair_frame = threshold_rows[["task", metric]].dropna(subset=[metric]).copy()
            pair_values = pair_frame[metric].to_numpy(dtype=float)
            pair_values = pair_values[np.isfinite(pair_values)]
            # Endpoint, rather than task-model pair, is the comparison unit.
            # This avoids pseudo-replication when several models are evaluated on
            # the same molecules. Models remain visible in the pair-level counts,
            # while all descriptive intervals, diagnostics, and gates use endpoint means.
            endpoint_values = (
                pair_frame.groupby("task", sort=True)[metric].mean().to_numpy(dtype=float)
            )
            endpoint_values = endpoint_values[np.isfinite(endpoint_values)]
            if len(endpoint_values) == 0:
                continue
            low, high = bootstrap_interval(endpoint_values, bootstrap_reps, rng)
            output.append(
                {
                    "threshold": threshold,
                    "metric": metric,
                    "n_pairs": len(pair_values),
                    "n_endpoints": len(endpoint_values),
                    "pair_positive_count": int(np.sum(pair_values > 0)),
                    "pair_positive_fraction": float(np.mean(pair_values > 0)),
                    "positive_count": int(np.sum(endpoint_values > 0)),
                    "positive_fraction": float(np.mean(endpoint_values > 0)),
                    "mean": float(np.mean(endpoint_values)),
                    "median": float(np.median(endpoint_values)),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "sign_flip_p": sign_flip_pvalue(
                        endpoint_values, permutation_reps, rng
                    ),
                }
            )
    return pd.DataFrame(output)


def build_gate(summary: pd.DataFrame, primary_threshold: float) -> dict:
    primary = summary[np.isclose(summary["threshold"], primary_threshold)].set_index("metric")
    criteria: dict[str, bool] = {}
    for metric in (
        "log_error_ratio_outside_inside",
        "ece_gap_outside_inside",
        "brier_gap_outside_inside",
        "coverage_error_gap_outside_inside",
        "uq_correlation_advantage_inside",
    ):
        if metric not in primary.index:
            criteria[metric] = False
            continue
        row = primary.loc[metric]
        criteria[metric] = bool(
            float(row["positive_fraction"]) >= 0.6
            and float(row["mean_ci_low"]) > 0
            and float(row["holm_p_within_threshold"]) < 0.05
        )
    strong_dimensions = int(sum(criteria.values()))
    status = "GO" if criteria["log_error_ratio_outside_inside"] and strong_dimensions >= 2 else "HOLD"
    return {
        "status": status,
        "primary_threshold": primary_threshold,
        "strong_conditional_dimensions": strong_dimensions,
        "criteria": criteria,
        "interpretation": (
            "Historical resource-routing GO: the frozen descriptive thresholds identified "
            "multiple conditional contrasts worth carrying into the completed neural matrix. "
            "This collection-conditional gate is not confirmatory evidence and does not "
            "remove cross-endpoint molecule dependence."
            if status == "GO"
            else "Historical resource-routing HOLD: the frozen descriptive thresholds did not "
            "justify expanding the scoped GPU analysis. This is not a confirmatory null result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how calibration and uncertainty reliability change outside the chemical domain."
    )
    parser.add_argument(
        "--prediction-dir",
        default=str(ROOT / "results" / "predictions" / "traditional_local_prelim"),
    )
    parser.add_argument(
        "--additional-prediction-dirs",
        nargs="*",
        default=[],
        help="Additional normalized or neural prediction directories to include.",
    )
    parser.add_argument(
        "--similarity-dir",
        default=str(ROOT / "results" / "cache" / "applicability"),
    )
    parser.add_argument(
        "--similarity-split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=PYTDC_SPLIT_SOURCE,
        help=(
            "Split protocol encoded in applicability cache filenames; pass "
            "locked_cache for formal five-model analysis."
        ),
    )
    parser.add_argument(
        "--require-complete-similarity",
        action="store_true",
        help="Fail on any missing, malformed, or test-misaligned similarity cache.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "tables" / "conditional_reliability_local_prelim"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--primary-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-seeds", type=int, default=5)
    parser.add_argument("--minimum-domain-n", type=int, default=20)
    parser.add_argument("--rejection-fraction", type=float, default=0.3)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutation-reps", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20260714)
    args = parser.parse_args()

    if args.primary_threshold not in args.thresholds:
        raise ValueError("primary-threshold must also be listed in thresholds")
    rng = np.random.default_rng(args.random_seed)
    groups = load_ensemble_groups(
        [Path(args.prediction_dir), *[Path(path) for path in args.additional_prediction_dirs]],
        Path(args.similarity_dir),
        args.minimum_seeds,
        similarity_split_source=args.similarity_split_source,
        require_similarity=args.require_complete_similarity,
    )
    profiles = build_model_profiles(groups, args.rejection_fraction)
    rows = analyze_groups(
        groups,
        args.thresholds,
        args.rejection_fraction,
        args.minimum_domain_n,
    )
    summary = summarize(rows, args.bootstrap_reps, args.permutation_reps, rng)
    summary["holm_p_within_threshold"] = (
        summary.groupby("threshold", group_keys=False)["sign_flip_p"].apply(holm_adjust)
    )
    gate = build_gate(summary, args.primary_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out_dir / "model_reliability_profiles.csv", index=False)
    rows.to_csv(out_dir / "conditional_reliability_by_task_model.csv", index=False)
    summary.to_csv(out_dir / "conditional_reliability_summary.csv", index=False)
    (out_dir / "conditional_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"task-model ensembles: {len(groups)}")
    print(f"rows: {len(rows)}")
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
