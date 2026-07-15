from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASKS,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.defense import y_randomization_test
from admet_benchmark.featurization import align_labels, morgan_count_matrix
from admet_benchmark.models import make_traditional_model
from admet_benchmark.smiles_compat import model_input_smiles
from admet_benchmark.utils import set_seed


SUMMARY_FIELDS = [
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
]

SCORE_FIELDS = [
    "task",
    "task_type",
    "model",
    "seed",
    "split_source",
    "split_algorithm",
    "metric",
    "trial",
    "random_score",
]


def metric_fn(name: str):
    if name == "auroc":
        return roc_auc_score
    if name == "auprc":
        return average_precision_score
    if name == "mae":
        return mean_absolute_error
    if name == "spearman":
        from scipy.stats import spearmanr

        return lambda y, p: float(spearmanr(y, p, nan_policy="omit").statistic)
    raise ValueError(f"Unsupported metric: {name}")


def oriented_margin(metric: str, real_score: float, random_mean: float) -> float:
    if metric == "mae":
        return random_mean - real_score
    return real_score - random_mean


def load_best_models(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "validation_selected_model" in df.columns:
        model_column = "validation_selected_model"
    elif "model" in df.columns:
        model_column = "model"
    else:
        raise ValueError(
            "Selection table requires validation_selected_model or model column"
        )
    task_map = {task.lower(): task for task in TDC_TASKS}
    best: dict[str, str] = {}
    for row in df.itertuples(index=False):
        task = task_map.get(str(row.task).lower())
        if task is None:
            raise ValueError(f"Cannot map task from best-by-task table: {row.task}")
        best[task] = str(getattr(row, model_column))
    missing = [task for task in TDC_TASKS if task not in best]
    if missing:
        raise ValueError(f"Best-model table missing tasks: {missing}")
    return best


def run_one(
    task: str,
    model_name: str,
    seed: int,
    n_trials: int,
    data_dir: str,
    model_n_jobs: int,
    split_source: str = PYTDC_SPLIT_SOURCE,
) -> tuple[dict, list[dict]]:
    print(f"y-randomization-all: {task} {model_name}", flush=True)
    set_seed(seed)
    split = load_tdc_official(
        task, data_dir=data_dir, seed=seed, split_source=split_source
    )
    train_smiles, train_y = extract_xy(split.train)
    test_smiles, test_y = extract_xy(split.test)
    x_train, train_idx = morgan_count_matrix(
        [model_input_smiles(value) for value in train_smiles]
    )
    x_test, test_idx = morgan_count_matrix(
        [model_input_smiles(value) for value in test_smiles]
    )
    y_train = align_labels(train_y, train_idx)
    y_test = align_labels(test_y, test_idx)
    model = make_traditional_model(model_name, split.task_type, seed=seed, n_jobs=model_n_jobs)
    result = y_randomization_test(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        metric_fn(split.metric),
        n_trials=n_trials,
        seed=seed,
    )
    margin = oriented_margin(split.metric, result["real_score"], result["random_mean"])
    z_like = margin / result["random_std"] if result["random_std"] else float("nan")
    summary = {
        "task": split.name,
        "task_type": split.task_type,
        "model": model_name,
        "seed": seed,
        "split_source": split_source,
        "split_algorithm": (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        ),
        "metric": split.metric,
        "real_score": result["real_score"],
        "random_mean": result["random_mean"],
        "random_std": result["random_std"],
        "oriented_margin": margin,
        "z_like_margin": z_like,
        "n_trials": n_trials,
        "train_n": len(y_train),
        "test_n": len(y_test),
    }
    scores = [
        {
            "task": split.name,
            "task_type": split.task_type,
            "model": model_name,
            "seed": seed,
            "split_source": split_source,
            "split_algorithm": (
                LOCKED_CACHE_SPLIT_ALGORITHM
                if split_source == LOCKED_CACHE_SPLIT_SOURCE
                else "pytdc_default_task_specific_runtime"
            ),
            "metric": split.metric,
            "trial": trial,
            "random_score": score,
        }
        for trial, score in enumerate(result["random_scores"], start=1)
    ]
    return summary, scores


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Y-randomization on all TDC tasks using the best traditional model per task.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--model-n-jobs", type=int, default=1)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--split-source", choices=SPLIT_SOURCE_CHOICES, default=PYTDC_SPLIT_SOURCE
    )
    parser.add_argument("--best", default=str(ROOT / "results" / "tables" / "traditional_grid_local_prelim_best_by_task.csv"))
    parser.add_argument("--summary", default=str(ROOT / "results" / "tables" / "y_randomization_all_tasks_local_prelim.csv"))
    parser.add_argument("--scores", default=str(ROOT / "results" / "tables" / "y_randomization_all_tasks_scores_local_prelim.csv"))
    args = parser.parse_args()

    best = load_best_models(Path(args.best))
    jobs = [(task, best[task]) for task in TDC_TASKS]
    summaries: list[dict] = []
    score_rows: list[dict] = []
    if args.parallel_runs <= 1:
        for task, model_name in jobs:
            summary, scores = run_one(
                task,
                model_name,
                args.seed,
                args.n_trials,
                args.data_dir,
                args.model_n_jobs,
                args.split_source,
            )
            summaries.append(summary)
            score_rows.extend(scores)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_runs) as executor:
            future_to_job = {
                executor.submit(
                    run_one,
                    task,
                    model_name,
                    args.seed,
                    args.n_trials,
                    args.data_dir,
                    args.model_n_jobs,
                    args.split_source,
                ): (task, model_name)
                for task, model_name in jobs
            }
            for future in as_completed(future_to_job):
                summary, scores = future.result()
                summaries.append(summary)
                score_rows.extend(scores)

    summaries.sort(
        key=lambda row: TDC_TASKS.index(
            next(
                task
                for task in TDC_TASKS
                if task.lower() == str(row["task"]).lower()
            )
        )
    )
    score_rows.sort(key=lambda row: (row["task"], row["trial"]))
    write_csv(Path(args.summary), summaries, SUMMARY_FIELDS)
    write_csv(Path(args.scores), score_rows, SCORE_FIELDS)
    print(f"summary_rows: {len(summaries)}")
    print(f"score_rows: {len(score_rows)}")
    print(f"wrote: {args.summary}")
    print(f"wrote: {args.scores}")


if __name__ == "__main__":
    main()
