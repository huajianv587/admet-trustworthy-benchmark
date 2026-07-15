from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASKS,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.featurization import align_labels, morgan_count_matrix
from admet_benchmark.models import make_traditional_model, predict_scores
from admet_benchmark.smiles_compat import (
    SMILES_COMPATIBILITY_POLICY,
    model_input_smiles,
)
from admet_benchmark.utils import set_seed


FIELDNAMES = [
    "task",
    "task_type",
    "official_metric",
    "model",
    "seed",
    "split",
    "sample_index",
    "smiles",
    "y_true",
    "y_pred",
]

PREDICTION_MANIFEST_SCHEMA = "traditional_prediction_run_v1"


def prediction_path(out_dir: Path, task: str, model: str, seed: int) -> Path:
    return out_dir / task.lower() / f"{model}_seed{seed}.csv"


def prediction_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prediction_manifest(
    prediction: Path,
    *,
    task: str,
    task_type: str,
    official_metric: str,
    model: str,
    seed: int,
    split_source: str,
    radius: int,
    n_bits: int,
    train_n: int,
    valid_n: int,
    test_n: int,
    fit_seconds: float,
    representation_repair_count: int,
) -> Path:
    """Atomically record run timing and bind it to the prediction bytes."""

    path = prediction_manifest_path(prediction)
    payload = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "task": task,
        "task_type": task_type,
        "official_metric": official_metric,
        "model": model,
        "seed": int(seed),
        "split_source": split_source,
        "split_algorithm": (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        ),
        "radius": int(radius),
        "n_bits": int(n_bits),
        "train_n": int(train_n),
        "valid_n": int(valid_n),
        "test_n": int(test_n),
        "prediction_rows": int(valid_n + test_n),
        "fit_seconds": float(fit_seconds),
        "prediction_sha256": file_sha256(prediction),
        "smiles_compatibility_policy": SMILES_COMPATIBILITY_POLICY,
        "representation_repair_count": int(representation_repair_count),
    }
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def compatible_model_inputs(smiles: list[str]) -> list[str]:
    return [model_input_smiles(value) for value in smiles]


def representation_repair_count(*smiles_groups: list[str]) -> int:
    return sum(
        model_input_smiles(value) != str(value)
        for group in smiles_groups
        for value in group
    )


def prediction_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            seen_splits = {row.get("split") for row in reader}
    except csv.Error:
        return False
    return {"valid", "test"}.issubset(seen_splits)


def write_predictions(
    path: Path,
    split_name: str,
    split,
    smiles: list[str],
    labels,
    kept_indices: list[int],
    predictions,
    task_name: str,
    task_type: str,
    metric: str,
    model_name: str,
    seed: int,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if mode == "a" else "w"
    write_header = open_mode == "w" or not path.exists()
    with path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for local_idx, original_idx in enumerate(kept_indices):
            writer.writerow(
                {
                    "task": task_name,
                    "task_type": task_type,
                    "official_metric": metric,
                    "model": model_name,
                    "seed": seed,
                    "split": split_name,
                    "sample_index": original_idx,
                    "smiles": smiles[original_idx],
                    "y_true": labels[local_idx],
                    "y_pred": predictions[local_idx],
                }
            )


def run_one(
    task: str,
    model_name: str,
    seed: int,
    out_dir: Path,
    data_dir: str,
    radius: int,
    n_bits: int,
    model_n_jobs: int,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
) -> str:
    out_path = prediction_path(out_dir, task, model_name, seed)
    if prediction_complete(out_path):
        timing = (
            "timing manifest present"
            if prediction_manifest_path(out_path).is_file()
            else "fit_seconds unavailable (legacy prediction without manifest)"
        )
        return f"skip existing: {out_path} [{timing}]"
    if out_path.exists():
        out_path.unlink()
    manifest_path = prediction_manifest_path(out_path)
    if manifest_path.exists():
        manifest_path.unlink()

    set_seed(seed)
    split = load_tdc_official(
        task, data_dir=data_dir, seed=seed, split_source=split_source
    )
    train_smiles, train_y = extract_xy(split.train)
    valid_smiles, valid_y = extract_xy(split.valid)
    test_smiles, test_y = extract_xy(split.test)

    x_train, train_idx = morgan_count_matrix(
        compatible_model_inputs(train_smiles), radius=radius, n_bits=n_bits
    )
    x_valid, valid_idx = morgan_count_matrix(
        compatible_model_inputs(valid_smiles), radius=radius, n_bits=n_bits
    )
    x_test, test_idx = morgan_count_matrix(
        compatible_model_inputs(test_smiles), radius=radius, n_bits=n_bits
    )
    y_train = align_labels(train_y, train_idx)
    y_valid = align_labels(valid_y, valid_idx)
    y_test = align_labels(test_y, test_idx)

    model = make_traditional_model(model_name, split.task_type, seed=seed, n_jobs=model_n_jobs)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    valid_pred = predict_scores(model, x_valid, split.task_type)
    test_pred = predict_scores(model, x_test, split.task_type)

    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    write_predictions(
        tmp_path,
        "valid",
        split.valid,
        valid_smiles,
        y_valid,
        valid_idx,
        valid_pred,
        split.name,
        split.task_type,
        split.metric,
        model_name,
        seed,
        "w",
    )
    write_predictions(
        tmp_path,
        "test",
        split.test,
        test_smiles,
        y_test,
        test_idx,
        test_pred,
        split.name,
        split.task_type,
        split.metric,
        model_name,
        seed,
        "a",
    )
    tmp_path.replace(out_path)
    write_prediction_manifest(
        out_path,
        task=split.name,
        task_type=split.task_type,
        official_metric=split.metric,
        model=model_name,
        seed=seed,
        split_source=split_source,
        radius=radius,
        n_bits=n_bits,
        train_n=len(y_train),
        valid_n=len(y_valid),
        test_n=len(y_test),
        fit_seconds=fit_seconds,
        representation_repair_count=representation_repair_count(
            train_smiles, valid_smiles, test_smiles
        ),
    )
    return f"done: {task} {model_name} seed={seed} fit_seconds={fit_seconds:.2f}"


def run_task_seed(
    task: str,
    model_names: list[str],
    seed: int,
    out_dir: Path,
    data_dir: str,
    radius: int,
    n_bits: int,
    model_n_jobs: int,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
) -> str:
    missing_models = [
        model_name
        for model_name in model_names
        if not prediction_complete(prediction_path(out_dir, task, model_name, seed))
    ]
    if not missing_models:
        return f"skip existing: {task} seed={seed}"

    set_seed(seed)
    split = load_tdc_official(
        task, data_dir=data_dir, seed=seed, split_source=split_source
    )
    train_smiles, train_y = extract_xy(split.train)
    valid_smiles, valid_y = extract_xy(split.valid)
    test_smiles, test_y = extract_xy(split.test)

    x_train, train_idx = morgan_count_matrix(
        compatible_model_inputs(train_smiles), radius=radius, n_bits=n_bits
    )
    x_valid, valid_idx = morgan_count_matrix(
        compatible_model_inputs(valid_smiles), radius=radius, n_bits=n_bits
    )
    x_test, test_idx = morgan_count_matrix(
        compatible_model_inputs(test_smiles), radius=radius, n_bits=n_bits
    )
    y_train = align_labels(train_y, train_idx)
    y_valid = align_labels(valid_y, valid_idx)
    y_test = align_labels(test_y, test_idx)

    messages = []
    for model_name in missing_models:
        out_path = prediction_path(out_dir, task, model_name, seed)
        if out_path.exists():
            out_path.unlink()
        manifest_path = prediction_manifest_path(out_path)
        if manifest_path.exists():
            manifest_path.unlink()

        model = make_traditional_model(model_name, split.task_type, seed=seed, n_jobs=model_n_jobs)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        valid_pred = predict_scores(model, x_valid, split.task_type)
        test_pred = predict_scores(model, x_test, split.task_type)

        tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
        if tmp_path.exists():
            tmp_path.unlink()
        write_predictions(
            tmp_path,
            "valid",
            split.valid,
            valid_smiles,
            y_valid,
            valid_idx,
            valid_pred,
            split.name,
            split.task_type,
            split.metric,
            model_name,
            seed,
            "w",
        )
        write_predictions(
            tmp_path,
            "test",
            split.test,
            test_smiles,
            y_test,
            test_idx,
            test_pred,
            split.name,
            split.task_type,
            split.metric,
            model_name,
            seed,
            "a",
        )
        tmp_path.replace(out_path)
        write_prediction_manifest(
            out_path,
            task=split.name,
            task_type=split.task_type,
            official_metric=split.metric,
            model=model_name,
            seed=seed,
            split_source=split_source,
            radius=radius,
            n_bits=n_bits,
            train_n=len(y_train),
            valid_n=len(y_valid),
            test_n=len(y_test),
            fit_seconds=fit_seconds,
            representation_repair_count=representation_repair_count(
                train_smiles, valid_smiles, test_smiles
            ),
        )
        messages.append(f"done: {task} {model_name} seed={seed} fit_seconds={fit_seconds:.2f}")

    return "\n".join(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save valid/test predictions for traditional ML baselines.")
    parser.add_argument("--tasks", nargs="*", default=TDC_TASKS)
    parser.add_argument("--models", nargs="*", default=["rf", "xgboost", "lightgbm"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=LOCKED_CACHE_SPLIT_SOURCE,
        help="Explicit split protocol; formal five-model runs use locked_cache.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "predictions" / "traditional_locked_cache"),
        help="A separate formal output root; the historical local_prelim tree is preserved.",
    )
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--model-n-jobs", type=int, default=-1)
    parser.add_argument("--group-by-task-seed", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_n_jobs = args.model_n_jobs
    if args.parallel_runs > 1 and model_n_jobs == -1:
        model_n_jobs = max(1, (os.cpu_count() or 1) // args.parallel_runs)

    if args.group_by_task_seed:
        jobs = [(task, seed) for task in args.tasks for seed in args.seeds]
    else:
        jobs = [
            (task, model_name, seed)
            for task in args.tasks
            for seed in args.seeds
            for model_name in args.models
        ]
    print(
        f"scheduled_runs={len(jobs)} parallel_runs={args.parallel_runs} model_n_jobs={model_n_jobs}",
        flush=True,
    )

    if args.parallel_runs <= 1:
        if args.group_by_task_seed:
            for task, seed in jobs:
                print(f"run: task={task} seed={seed}", flush=True)
                print(
                    run_task_seed(
                        task,
                        args.models,
                        seed,
                        out_dir,
                        args.data_dir,
                        args.radius,
                        args.n_bits,
                        model_n_jobs,
                        args.split_source,
                    ),
                    flush=True,
                )
        else:
            for task, model_name, seed in jobs:
                print(f"run: task={task} model={model_name} seed={seed}", flush=True)
                print(
                    run_one(
                        task,
                        model_name,
                        seed,
                        out_dir,
                        args.data_dir,
                        args.radius,
                        args.n_bits,
                        model_n_jobs,
                        args.split_source,
                    ),
                    flush=True,
                )
        return

    with ProcessPoolExecutor(max_workers=args.parallel_runs) as executor:
        if args.group_by_task_seed:
            future_to_job = {
                executor.submit(
                    run_task_seed,
                    task,
                    args.models,
                    seed,
                    out_dir,
                    args.data_dir,
                    args.radius,
                    args.n_bits,
                    model_n_jobs,
                    args.split_source,
                ): (task, "all_models", seed)
                for task, seed in jobs
            }
        else:
            future_to_job = {
                executor.submit(
                    run_one,
                    task,
                    model_name,
                    seed,
                    out_dir,
                    args.data_dir,
                    args.radius,
                    args.n_bits,
                    model_n_jobs,
                    args.split_source,
                ): (task, model_name, seed)
                for task, model_name, seed in jobs
            }
        for future in as_completed(future_to_job):
            task, model_name, seed = future_to_job[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                print(f"failed: {task} {model_name} seed={seed}: {exc}", flush=True)
                raise


if __name__ == "__main__":
    main()
