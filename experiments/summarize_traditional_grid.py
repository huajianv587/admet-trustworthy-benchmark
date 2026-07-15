from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


HIGHER_IS_BETTER = {"auroc", "auprc", "mcc", "balanced_accuracy", "r2", "spearman"}
LOWER_IS_BETTER = {"mae", "rmse", "brier"}


def summarize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize a grid while treating unavailable legacy fit times as NA."""

    metric_rows = []
    for (task, model), group in df.groupby(["task", "model"], sort=True):
        task_type = group["task_type"].iloc[0]
        official_metric = group["official_metric"].iloc[0]
        official_col = f"test_{official_metric}"
        if official_col not in group:
            raise ValueError(f"Missing official metric column: {official_col}")
        values = pd.to_numeric(group[official_col], errors="raise")
        fit_seconds = pd.to_numeric(group["fit_seconds"], errors="coerce")
        timed_runs = int(fit_seconds.notna().sum())
        metric_rows.append(
            {
                "task": task,
                "task_type": task_type,
                "official_metric": official_metric,
                "model": model,
                "n_seeds": int(group["seed"].nunique()),
                "test_official_mean": float(values.mean()),
                "test_official_std": float(values.std(ddof=1)),
                "valid_official_mean": float(
                    pd.to_numeric(group[f"valid_{official_metric}"], errors="raise").mean()
                ),
                "valid_official_std": float(
                    pd.to_numeric(
                        group[f"valid_{official_metric}"], errors="raise"
                    ).std(ddof=1)
                ),
                "n_fit_seconds": timed_runs,
                "fit_seconds_coverage": float(timed_runs / len(group)),
                "mean_fit_seconds": (
                    float(fit_seconds.mean()) if timed_runs else float("nan")
                ),
            }
        )
    summary = pd.DataFrame(metric_rows)

    best_rows = []
    for task, group in summary.groupby("task", sort=True):
        metric = group["official_metric"].iloc[0]
        ascending = metric in LOWER_IS_BETTER
        # This file is consumed by negative controls and interpretability
        # analyses. Selection must therefore precede test evaluation.
        best = group.sort_values("valid_official_mean", ascending=ascending).iloc[0]
        best_rows.append(best.to_dict())
    best_df = pd.DataFrame(best_rows)
    return summary, best_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize traditional ML grid results.")
    parser.add_argument("--input", default=str(ROOT / "results" / "tables" / "traditional_grid_local_prelim.csv"))
    parser.add_argument("--summary", default=str(ROOT / "results" / "tables" / "traditional_grid_local_prelim_summary.csv"))
    parser.add_argument("--best", default=str(ROOT / "results" / "tables" / "traditional_grid_local_prelim_best_by_task.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary, best_df = summarize_frame(df)
    summary.to_csv(args.summary, index=False)
    best_df.to_csv(args.best, index=False)

    print(f"input_rows: {len(df)}")
    print(f"tasks: {df['task'].nunique()}")
    print(f"models: {sorted(df['model'].unique())}")
    print(f"seeds: {sorted(df['seed'].unique())}")
    print(f"summary_rows: {len(summary)}")
    print("best_by_task:")
    for _, row in best_df.sort_values("task").iterrows():
        print(
            f"{row['task']}: {row['model']} "
            f"validation_{row['official_metric']}={row['valid_official_mean']:.4f}; "
            f"heldout_test={row['test_official_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
