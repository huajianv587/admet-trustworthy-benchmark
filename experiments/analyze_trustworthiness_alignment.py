from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dimension:
    name: str
    column: str
    higher_is_better: bool
    task_type: str | None = None


DIMENSIONS = (
    Dimension("uncalibrated_ece", "ece", False, "classification"),
    Dimension("uncalibrated_brier", "brier", False, "classification"),
    Dimension("conformal_coverage_error", "coverage_error", False, "regression"),
    Dimension("selective_gain_30pct", "selective_gain_30pct", True),
    Dimension("applicability_alignment", "applicability_alignment", True),
)


def orient_metric(metric: str, value: float) -> float:
    return -float(value) if str(metric).lower() == "mae" else float(value)


def bootstrap_interval(
    values: np.ndarray,
    statistic,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = np.empty(reps, dtype=float)
    for idx in range(reps):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[idx] = statistic(sample)
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


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return float("nan")
    left_rank = frame["left"].rank(method="average").to_numpy(dtype=float)
    right_rank = frame["right"].rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_performance(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    metric_columns = {
        "auroc": "test_auroc",
        "auprc": "test_auprc",
        "mae": "test_mae",
        "spearman": "test_spearman",
    }
    frame["performance"] = frame.apply(
        lambda row: orient_metric(
            row["official_metric"], row[metric_columns[str(row["official_metric"]).lower()]]
        ),
        axis=1,
    )
    return (
        frame.groupby(["task", "task_type", "official_metric", "model"], as_index=False)
        .agg(
            performance=("performance", "mean"),
            performance_sd=("performance", "std"),
            n_seeds=("seed", "nunique"),
            train_n=("train_n", "median"),
            valid_n=("valid_n", "median"),
            test_n=("test_n", "median"),
        )
    )


def load_calibration(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    classification = frame[frame["task_type"] == "classification"].copy()
    before = classification.drop_duplicates(["task", "model", "seed"])
    before = (
        before.groupby(["task", "model"], as_index=False)
        .agg(ece=("test_ece_before", "mean"), brier=("test_brier_before", "mean"))
    )

    classification["ece_gain"] = (
        classification["test_ece_before"] - classification["test_ece_after"]
    )
    classification["brier_gain"] = (
        classification["test_brier_before"] - classification["test_brier_after"]
    )
    method_effects = (
        classification.groupby(["task", "model", "method"], as_index=False)
        .agg(ece_gain=("ece_gain", "mean"), brier_gain=("brier_gain", "mean"))
    )

    regression = frame[frame["task_type"] == "regression"].copy()
    regression = (
        regression.groupby(["task", "model"], as_index=False)
        .agg(coverage=("coverage", "mean"), interval_width=("mean_interval_width", "mean"))
    )
    regression["coverage_error"] = np.abs(regression["coverage"] - 0.9)
    profiles = before.merge(regression, on=["task", "model"], how="outer")
    return profiles, method_effects


def load_selective_gain(path: Path, performance: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    available = sorted(float(value) for value in frame["rejection_frac"].dropna().unique())
    if 0.0 not in available:
        raise ValueError("Uncertainty-rejection table must include rejection_frac=0.0")
    target = 0.3 if 0.3 in available else max(available)
    pivot = frame.pivot_table(
        index=["task", "model"], columns="rejection_frac", values="metric", aggfunc="first"
    ).reset_index()
    meta = performance[["task", "official_metric"]].drop_duplicates()
    pivot = pivot.merge(meta, on="task", how="left")
    pivot["selective_gain_30pct"] = pivot.apply(
        lambda row: orient_metric(row["official_metric"], row[target])
        - orient_metric(row["official_metric"], row[0.0]),
        axis=1,
    )
    return pivot[["task", "model", "selective_gain_30pct"]]


def load_applicability_alignment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["oriented_bin_metric"] = frame.apply(
        lambda row: orient_metric(row["official_metric"], row["metric"]), axis=1
    )
    rows: list[dict] = []
    for (task, model, seed), group in frame.groupby(["task", "model", "seed"], sort=True):
        usable = group[(group["count"] > 0)].dropna(
            subset=["mean_similarity", "oriented_bin_metric"]
        )
        rows.append(
            {
                "task": task,
                "model": model,
                "seed": seed,
                "applicability_alignment": rank_correlation(
                    usable["mean_similarity"], usable["oriented_bin_metric"]
                ),
            }
        )
    seed_frame = pd.DataFrame(rows)
    return (
        seed_frame.groupby(["task", "model"], as_index=False)
        .agg(
            applicability_alignment=("applicability_alignment", "mean"),
            applicability_positive_seeds=(
                "applicability_alignment",
                lambda values: int(np.sum(np.asarray(values, dtype=float) > 0)),
            ),
        )
    )


def build_profiles(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance = load_performance(Path(args.traditional_grid))
    calibration, method_effects = load_calibration(Path(args.calibration_summary))
    selective = load_selective_gain(Path(args.uncertainty_rejection), performance)
    applicability = load_applicability_alignment(Path(args.applicability_summary))
    profiles = performance.merge(calibration, on=["task", "model"], how="left")
    profiles = profiles.merge(selective, on=["task", "model"], how="left")
    profiles = profiles.merge(applicability, on=["task", "model"], how="left")
    return profiles, method_effects


def analyze_rank_alignment(
    profiles: pd.DataFrame,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dimension in DIMENSIONS:
        eligible = profiles.copy()
        if dimension.task_type:
            eligible = eligible[eligible["task_type"] == dimension.task_type]
        eligible = eligible.dropna(subset=["performance", dimension.column])
        for task, group in eligible.groupby("task", sort=True):
            if group["model"].nunique() < 2:
                continue
            performance_winner = group.loc[group["performance"].idxmax(), "model"]
            if dimension.higher_is_better:
                reliability_winner = group.loc[group[dimension.column].idxmax(), "model"]
                reliability_oriented = group[dimension.column]
            else:
                reliability_winner = group.loc[group[dimension.column].idxmin(), "model"]
                reliability_oriented = -group[dimension.column]
            correlation = rank_correlation(group["performance"], reliability_oriented)
            task_rows.append(
                {
                    "dimension": dimension.name,
                    "task": task,
                    "task_type": group["task_type"].iloc[0],
                    "n_models": group["model"].nunique(),
                    "performance_winner": performance_winner,
                    "reliability_winner": reliability_winner,
                    "same_winner": bool(performance_winner == reliability_winner),
                    "performance_reliability_rank_correlation": correlation,
                }
            )

        dimension_rows = pd.DataFrame(task_rows)
        dimension_rows = dimension_rows[dimension_rows["dimension"] == dimension.name]
        if dimension_rows.empty:
            continue
        same = dimension_rows["same_winner"].astype(float).to_numpy()
        correlations = dimension_rows[
            "performance_reliability_rank_correlation"
        ].to_numpy(dtype=float)
        same_low, same_high = bootstrap_interval(
            same, np.mean, bootstrap_reps, rng
        )
        corr_low, corr_high = bootstrap_interval(
            correlations, np.nanmean, bootstrap_reps, rng
        )
        summary_rows.append(
            {
                "dimension": dimension.name,
                "eligible_tasks": len(dimension_rows),
                "same_winner_count": int(np.sum(same)),
                "same_winner_fraction": float(np.mean(same)),
                "same_winner_ci_low": same_low,
                "same_winner_ci_high": same_high,
                "mismatch_fraction": float(1.0 - np.mean(same)),
                "mean_rank_correlation": float(np.nanmean(correlations)),
                "rank_correlation_ci_low": corr_low,
                "rank_correlation_ci_high": corr_high,
                "rank_correlation_sign_flip_p": sign_flip_pvalue(
                    correlations, permutation_reps, rng
                ),
            }
        )
    return pd.DataFrame(task_rows), pd.DataFrame(summary_rows)


def summarize_effect(
    name: str,
    frame: pd.DataFrame,
    value_column: str,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> dict:
    pair_frame = frame[["task", value_column]].dropna(subset=[value_column]).copy()
    pair_values = pair_frame[value_column].to_numpy(dtype=float)
    pair_values = pair_values[np.isfinite(pair_values)]
    endpoint_values = (
        pair_frame.groupby("task", sort=True)[value_column]
        .mean()
        .to_numpy(dtype=float)
    )
    endpoint_values = endpoint_values[np.isfinite(endpoint_values)]
    low, high = bootstrap_interval(
        endpoint_values, np.mean, bootstrap_reps, rng
    )
    return {
        "effect": name,
        "n_pairs": len(pair_values),
        "n_endpoints": len(endpoint_values),
        "pair_positive_count": int(np.sum(pair_values > 0)),
        "pair_positive_fraction": (
            float(np.mean(pair_values > 0)) if len(pair_values) else float("nan")
        ),
        "positive_count": int(np.sum(endpoint_values > 0)),
        "positive_fraction": (
            float(np.mean(endpoint_values > 0))
            if len(endpoint_values)
            else float("nan")
        ),
        "mean": (
            float(np.mean(endpoint_values)) if len(endpoint_values) else float("nan")
        ),
        "median": (
            float(np.median(endpoint_values)) if len(endpoint_values) else float("nan")
        ),
        "mean_ci_low": low,
        "mean_ci_high": high,
        "sign_flip_p": sign_flip_pvalue(
            endpoint_values, permutation_reps, rng
        ),
    }


def analyze_effects(
    profiles: pd.DataFrame,
    method_effects: pd.DataFrame,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    for method, group in method_effects.groupby("method", sort=True):
        rows.append(
            summarize_effect(
                f"{method}_ece_gain",
                group,
                "ece_gain",
                bootstrap_reps,
                permutation_reps,
                rng,
            )
        )
        rows.append(
            summarize_effect(
                f"{method}_brier_gain",
                group,
                "brier_gain",
                bootstrap_reps,
                permutation_reps,
                rng,
            )
        )
    rows.append(
        summarize_effect(
            "selective_gain_30pct",
            profiles,
            "selective_gain_30pct",
            bootstrap_reps,
            permutation_reps,
            rng,
        )
    )
    rows.append(
        summarize_effect(
            "applicability_alignment",
            profiles,
            "applicability_alignment",
            bootstrap_reps,
            permutation_reps,
            rng,
        )
    )
    return pd.DataFrame(rows)


def build_claim_gate(
    profiles: pd.DataFrame,
    rank_summary: pd.DataFrame,
    effects: pd.DataFrame,
) -> dict:
    rank_mismatch_dimensions = int(
        np.sum(rank_summary["mismatch_fraction"].fillna(0).to_numpy() >= 0.5)
    )
    effect_index = effects.set_index("effect")

    calibration_mixed = False
    for method in ("isotonic", "platt"):
        ece_key = f"{method}_ece_gain"
        brier_key = f"{method}_brier_gain"
        if ece_key in effect_index.index and brier_key in effect_index.index:
            ece_fraction = float(effect_index.loc[ece_key, "positive_fraction"])
            brier_fraction = float(effect_index.loc[brier_key, "positive_fraction"])
            calibration_mixed = calibration_mixed or (
                0.2 <= ece_fraction <= 0.8 and brier_fraction < 0.8
            )

    selective_fraction = float(
        effect_index.loc["selective_gain_30pct", "positive_fraction"]
    )
    applicability_fraction = float(
        effect_index.loc["applicability_alignment", "positive_fraction"]
    )
    model_task_coverage = profiles.groupby("model")["task"].nunique()
    gates = {
        "complete_analyzed_model_matrix": bool(
            profiles["task"].nunique() == 22
            and profiles["model"].nunique() >= 3
            and int(model_task_coverage.min()) == 22
            and int(profiles["n_seeds"].min()) >= 5
        ),
        "rank_mismatch_is_recurrent": bool(rank_mismatch_dimensions >= 3),
        "calibration_is_helpful_but_not_universal": bool(calibration_mixed),
        "selective_prediction_is_helpful_but_not_universal": bool(
            0.5 <= selective_fraction <= 0.9
        ),
        "applicability_signal_is_common": bool(applicability_fraction >= 0.6),
    }
    story_core = [
        gates["rank_mismatch_is_recurrent"],
        gates["calibration_is_helpful_but_not_universal"],
        gates["selective_prediction_is_helpful_but_not_universal"],
    ]
    complete = gates["complete_analyzed_model_matrix"]
    story_passes = all(story_core)
    if complete and story_passes:
        status = "GO"
        interpretation = (
            "The analyzed model matrix supports a performance-versus-reliability "
            "misalignment paper. Finish the prespecified neural baselines, but do not "
            "add expensive model families until the conditional-reliability analysis passes."
        )
    elif story_passes:
        status = "HOLD"
        interpretation = (
            "The empirical story gate passes, but at least one included model lacks full "
            "22-endpoint coverage. Continue only the locked baseline matrix; do not expand "
            "GPU scope until coverage is complete."
        )
    else:
        status = "HOLD"
        interpretation = (
            "The central empirical story is not yet strong enough to justify additional GPU scope."
        )
    return {
        "status": status,
        "interpretation": interpretation,
        "gates": gates,
        "diagnostics": {
            "tasks": int(profiles["task"].nunique()),
            "models": sorted(str(value) for value in profiles["model"].dropna().unique()),
            "task_coverage_by_model": {
                str(model): int(count) for model, count in model_task_coverage.items()
            },
            "rank_mismatch_dimensions_at_least_half": rank_mismatch_dimensions,
            "selective_positive_fraction": selective_fraction,
            "applicability_positive_fraction": applicability_fraction,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether ADMET performance winners are also reliability winners."
    )
    tables = ROOT / "results" / "tables"
    parser.add_argument(
        "--traditional-grid",
        default=str(tables / "traditional_grid_local_prelim.csv"),
    )
    parser.add_argument(
        "--calibration-summary",
        default=str(tables / "calibration_summary_local_prelim.csv"),
    )
    parser.add_argument(
        "--uncertainty-rejection",
        default=str(tables / "uncertainty_rejection_local_prelim.csv"),
    )
    parser.add_argument(
        "--applicability-summary",
        default=str(tables / "applicability_domain_all_seeds_local_prelim.csv"),
    )
    parser.add_argument(
        "--out-dir", default=str(tables / "trustworthiness_alignment_local_prelim")
    )
    parser.add_argument(
        "--profile-table",
        help=(
            "Optional model_reliability_profiles.csv produced by "
            "analyze_conditional_reliability.py. When supplied, ranking analyses "
            "use this table instead of rebuilding traditional-only profiles."
        ),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutation-reps", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20260714)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    if args.profile_table:
        profiles = pd.read_csv(args.profile_table)
        _, method_effects = load_calibration(Path(args.calibration_summary))
    else:
        profiles, method_effects = build_profiles(args)
    rank_by_task, rank_summary = analyze_rank_alignment(
        profiles, args.bootstrap_reps, args.permutation_reps, rng
    )
    effects = analyze_effects(
        profiles,
        method_effects,
        args.bootstrap_reps,
        args.permutation_reps,
        rng,
    )
    rank_summary["rank_correlation_holm_p"] = holm_adjust(
        rank_summary["rank_correlation_sign_flip_p"]
    )
    effects["holm_p"] = holm_adjust(effects["sign_flip_p"])
    gate = build_claim_gate(profiles, rank_summary, effects)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out_dir / "model_reliability_profiles.csv", index=False)
    rank_by_task.to_csv(out_dir / "rank_alignment_by_task.csv", index=False)
    rank_summary.to_csv(out_dir / "rank_alignment_summary.csv", index=False)
    effects.to_csv(out_dir / "method_effects_summary.csv", index=False)
    (out_dir / "claim_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"profiles: {len(profiles)}")
    print(f"rank rows: {len(rank_by_task)}")
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
