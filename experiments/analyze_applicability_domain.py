from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.applicability import compute_max_similarities
from admet_benchmark.data_loader import TDC_TASKS, extract_xy, load_tdc_official
from admet_benchmark.metrics import classification_metrics, regression_metrics


FIELDS = [
    "task",
    "task_type",
    "official_metric",
    "model",
    "seed",
    "similarity_left",
    "similarity_right",
    "count",
    "mean_similarity",
    "metric",
    "mean_abs_error",
]


def metric_value(task_type: str, official_metric: str, y_true, y_pred) -> float:
    if len(y_true) == 0:
        return float("nan")
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)[official_metric]
    return regression_metrics(y_true, y_pred)[official_metric]


def similarity_cache_path(cache_dir: Path, task: str, seed: int) -> Path:
    return cache_dir / f"{task.lower()}_seed{seed}_test_similarity.csv"


def load_or_compute_similarity(task: str, seed: int, data_dir: str, cache_dir: Path) -> pd.DataFrame:
    path = similarity_cache_path(cache_dir, task, seed)
    if path.exists():
        return pd.read_csv(path)
    split = load_tdc_official(task, data_dir=data_dir, seed=seed)
    train_smiles, _ = extract_xy(split.train)
    test_smiles, _ = extract_xy(split.test)
    sims = compute_max_similarities(test_smiles, train_smiles)
    out = pd.DataFrame(
        {
            "task": split.name,
            "seed": seed,
            "sample_index": np.arange(len(test_smiles)),
            "smiles": test_smiles,
            "max_tanimoto": sims,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Applicability-domain analysis using nearest-neighbor Tanimoto similarity.")
    parser.add_argument("--tasks", nargs="*", default=TDC_TASKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1])
    parser.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions" / "traditional_local_prelim"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument("--cache-dir", default=str(ROOT / "results" / "cache" / "applicability"))
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "applicability_domain_local_prelim.csv"))
    parser.add_argument("--bins", nargs="*", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    args = parser.parse_args()

    pred_files = [Path(p) for p in glob.glob(str(Path(args.pred_dir) / "*" / "*.csv"))]
    predictions = pd.concat([pd.read_csv(path) for path in pred_files], ignore_index=True)
    predictions = predictions[predictions["split"] == "test"].copy()
    rows: list[dict] = []
    cache_dir = Path(args.cache_dir)

    for task in args.tasks:
        for seed in args.seeds:
            print(f"similarity: task={task} seed={seed}", flush=True)
            sims = load_or_compute_similarity(task, seed, args.data_dir, cache_dir)
            task_key = sims["task"].iloc[0]
            subset = predictions[(predictions["task"] == task_key) & (predictions["seed"] == seed)].copy()
            if subset.empty:
                print(f"no predictions for {task_key} seed={seed}", flush=True)
                continue
            merged = subset.merge(sims[["sample_index", "max_tanimoto"]], on="sample_index", how="left")
            for model, group in merged.groupby("model", sort=True):
                task_type = group["task_type"].iloc[0]
                metric = group["official_metric"].iloc[0]
                for left, right in zip(args.bins[:-1], args.bins[1:]):
                    if right == 1.0:
                        mask = (group["max_tanimoto"] >= left) & (group["max_tanimoto"] <= right)
                    else:
                        mask = (group["max_tanimoto"] >= left) & (group["max_tanimoto"] < right)
                    b = group[mask]
                    y_true = b["y_true"].to_numpy()
                    y_pred = b["y_pred"].to_numpy()
                    if task_type == "classification":
                        abs_error = np.abs(y_pred - y_true)
                    else:
                        abs_error = np.abs(y_pred - y_true)
                    rows.append(
                        {
                            "task": task_key,
                            "task_type": task_type,
                            "official_metric": metric,
                            "model": model,
                            "seed": seed,
                            "similarity_left": left,
                            "similarity_right": right,
                            "count": len(b),
                            "mean_similarity": float(b["max_tanimoto"].mean()) if len(b) else float("nan"),
                            "metric": metric_value(task_type, metric, y_true, y_pred) if len(b) else float("nan"),
                            "mean_abs_error": float(np.mean(abs_error)) if len(b) else float("nan"),
                        }
                    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
