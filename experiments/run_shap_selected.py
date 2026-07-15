from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.featurization import align_labels, morgan_count_matrix
from admet_benchmark.models import make_traditional_model
from admet_benchmark.smiles_compat import model_input_smiles
from admet_benchmark.utils import set_seed


SELECTED = [
    ("Caco2_Wang", "xgboost"),
    ("Solubility_AqSolDB", "xgboost"),
    ("BBB_Martins", "rf"),
    ("CYP2C9_Veith", "xgboost"),
    ("CYP3A4_Veith", "xgboost"),
    ("AMES", "xgboost"),
    ("DILI", "rf"),
    ("hERG", "rf"),
]

FIELDS = [
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
]


def normalize_shap_values(values):
    if isinstance(values, list):
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    return arr


def selected_pairs(selection_path: Path | None) -> list[tuple[str, str]]:
    if selection_path is None:
        return list(SELECTED)
    frame = pd.read_csv(selection_path)
    if "model" not in frame.columns:
        raise ValueError("Traditional selection table requires a model column")
    task_lookup = {str(row.task).lower(): str(row.model) for row in frame.itertuples()}
    pairs = []
    for task, _ in SELECTED:
        if task.lower() not in task_lookup:
            raise ValueError(f"Traditional selection table is missing {task}")
        pairs.append((task, task_lookup[task.lower()]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP top Morgan-bit analysis for selected traditional ML tasks.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--split-source", choices=SPLIT_SOURCE_CHOICES, default=PYTDC_SPLIT_SOURCE
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help=(
            "Optional validation-selected traditional best-by-task table. The eight "
            "prespecified task identities remain fixed; only their model is loaded."
        ),
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "shap_selected_local_prelim.csv"))
    args = parser.parse_args()

    import shap

    rows = []
    rng = np.random.default_rng(args.seed)
    for task, model_name in selected_pairs(args.selection):
        print(f"shap: {task} {model_name}", flush=True)
        set_seed(args.seed)
        split = load_tdc_official(
            task,
            data_dir=args.data_dir,
            seed=args.seed,
            split_source=args.split_source,
        )
        train_smiles, train_y = extract_xy(split.train)
        test_smiles, _ = extract_xy(split.test)
        x_train, train_idx = morgan_count_matrix(
            [model_input_smiles(value) for value in train_smiles]
        )
        y_train = align_labels(train_y, train_idx)
        x_test, _ = morgan_count_matrix(
            [model_input_smiles(value) for value in test_smiles]
        )
        model = make_traditional_model(model_name, split.task_type, seed=args.seed)
        model.fit(x_train, y_train)

        if len(x_test) > args.sample_size:
            sample_idx = rng.choice(len(x_test), size=args.sample_size, replace=False)
            x_explain = x_test[sample_idx]
        else:
            x_explain = x_test
        explainer = shap.TreeExplainer(model)
        shap_values = normalize_shap_values(explainer.shap_values(x_explain))
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        feature_means = np.mean(x_explain, axis=0)
        top_idx = np.argsort(mean_abs)[::-1][: args.top_k]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "task": split.name,
                    "model": model_name,
                    "seed": args.seed,
                    "split_source": args.split_source,
                    "split_algorithm": (
                        LOCKED_CACHE_SPLIT_ALGORITHM
                        if args.split_source == LOCKED_CACHE_SPLIT_SOURCE
                        else "pytdc_default_task_specific_runtime"
                    ),
                    "rank": rank,
                    "feature": f"morgan_bit_{idx}",
                    "mean_abs_shap": float(mean_abs[idx]),
                    "mean_feature_value": float(feature_means[idx]),
                    "train_n": len(x_train),
                    "test_n": len(x_test),
                    "sample_size_actual": len(x_explain),
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
