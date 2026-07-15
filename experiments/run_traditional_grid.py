from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from admet_benchmark.featurization import (
    align_labels,
    featurizable_indices,
    morgan_count_matrix,
)
from admet_benchmark.metrics import classification_metrics, regression_metrics
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
    "split_source",
    "split_algorithm",
    "radius",
    "n_bits",
    "train_n",
    "valid_n",
    "test_n",
    "fit_seconds",
    "valid_auroc",
    "valid_auprc",
    "valid_mcc",
    "valid_balanced_accuracy",
    "valid_brier",
    "valid_positive_rate",
    "valid_mae",
    "valid_rmse",
    "valid_r2",
    "valid_spearman",
    "test_auroc",
    "test_auprc",
    "test_mcc",
    "test_balanced_accuracy",
    "test_brier",
    "test_positive_rate",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_spearman",
]

PREDICTION_FIELDNAMES = [
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
REBUILD_AUDIT_SCHEMA = "traditional_grid_prediction_rebuild_v1"


class PredictionAuditError(ValueError):
    """Raised when cached predictions cannot support a formal grid row."""


def prediction_path(root: Path, task: str, model: str, seed: int) -> Path:
    return root / task.lower() / f"{model}_seed{seed}.csv"


def prediction_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_inputs(smiles: list[str]) -> list[str]:
    return [model_input_smiles(value) for value in smiles]


def _expected_rows(frame: pd.DataFrame) -> dict[str, object]:
    source_smiles, labels = extract_xy(frame)
    kept_indices = featurizable_indices(_model_inputs(source_smiles))
    label_values = align_labels(labels, kept_indices).astype(float)
    if not np.all(np.isfinite(label_values)):
        raise PredictionAuditError("source split contains non-finite labels")
    return {
        "sample_index": np.asarray(kept_indices, dtype=int),
        "smiles": [source_smiles[index] for index in kept_indices],
        "label": label_values,
    }


def _expected_identity(split) -> dict[str, object]:
    train_smiles, _ = extract_xy(split.train)
    valid_smiles, _ = extract_xy(split.valid)
    test_smiles, _ = extract_xy(split.test)
    train_indices = featurizable_indices(_model_inputs(train_smiles))
    return {
        "task": split.name,
        "task_type": split.task_type,
        "official_metric": split.metric,
        "train_n": len(train_indices),
        "valid": _expected_rows(split.valid),
        "test": _expected_rows(split.test),
        "representation_repair_count": sum(
            model_input_smiles(value) != str(value)
            for group in (train_smiles, valid_smiles, test_smiles)
            for value in group
        ),
    }


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise PredictionAuditError(f"{column} is not strictly numeric") from exc
    if not np.all(np.isfinite(values)):
        raise PredictionAuditError(f"{column} contains non-finite values")
    return values


def audit_prediction_file(
    path: Path,
    *,
    identity: dict[str, object],
    model: str,
    seed: int,
) -> pd.DataFrame:
    """Validate schema, metadata, sample identity, labels, order and scores."""

    if not path.is_file() or path.stat().st_size == 0:
        raise PredictionAuditError(f"missing prediction: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise PredictionAuditError(f"unreadable prediction {path}: {exc}") from exc
    if frame.columns.tolist() != PREDICTION_FIELDNAMES:
        raise PredictionAuditError(
            f"prediction schema mismatch for {path}: {frame.columns.tolist()}"
        )

    expected_metadata = {
        "task": str(identity["task"]),
        "task_type": str(identity["task_type"]),
        "official_metric": str(identity["official_metric"]),
        "model": model,
    }
    for column, expected in expected_metadata.items():
        observed = set(frame[column].astype(str))
        if observed != {expected}:
            raise PredictionAuditError(
                f"{path}: {column} mismatch; expected {expected!r}, observed {sorted(observed)!r}"
            )
    seed_values = _numeric_column(frame, "seed")
    if not np.all(seed_values == int(seed)):
        raise PredictionAuditError(f"{path}: seed mismatch; expected {seed}")

    expected_split = []
    expected_indices: list[int] = []
    expected_smiles: list[str] = []
    expected_labels: list[float] = []
    for split_name in ("valid", "test"):
        payload = identity[split_name]
        assert isinstance(payload, dict)
        indices = np.asarray(payload["sample_index"], dtype=int)
        expected_split.extend([split_name] * len(indices))
        expected_indices.extend(indices.tolist())
        expected_smiles.extend(str(value) for value in payload["smiles"])
        expected_labels.extend(np.asarray(payload["label"], dtype=float).tolist())

    if frame["split"].astype(str).tolist() != expected_split:
        raise PredictionAuditError(f"{path}: valid/test membership or row order mismatch")
    sample_indices = _numeric_column(frame, "sample_index")
    if not np.all(sample_indices == np.floor(sample_indices)):
        raise PredictionAuditError(f"{path}: sample_index contains non-integers")
    if sample_indices.astype(int).tolist() != expected_indices:
        raise PredictionAuditError(f"{path}: sample_index identity or order mismatch")
    if frame["smiles"].astype(str).tolist() != expected_smiles:
        raise PredictionAuditError(f"{path}: source SMILES identity or order mismatch")
    labels = _numeric_column(frame, "y_true")
    if not np.allclose(
        labels,
        np.asarray(expected_labels, dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    ):
        raise PredictionAuditError(f"{path}: labels are not aligned to source rows")
    predictions = _numeric_column(frame, "y_pred")
    if identity["task_type"] == "classification" and (
        np.any(predictions < 0.0) or np.any(predictions > 1.0)
    ):
        raise PredictionAuditError(f"{path}: classification scores outside [0, 1]")
    if frame.duplicated(["split", "sample_index"]).any():
        raise PredictionAuditError(f"{path}: duplicate split/sample_index identities")
    return frame


def _load_fit_seconds(
    prediction: Path,
    *,
    identity: dict[str, object],
    model: str,
    seed: int,
    split_source: str,
    radius: int,
    n_bits: int,
) -> tuple[float | str, bool]:
    """Load an optional timing manifest; an absent legacy manifest is explicit NA."""

    path = prediction_manifest_path(prediction)
    if not path.is_file():
        return "NA", False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionAuditError(f"unreadable timing manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredictionAuditError(f"timing manifest is not an object: {path}")

    valid_payload = identity["valid"]
    test_payload = identity["test"]
    assert isinstance(valid_payload, dict) and isinstance(test_payload, dict)
    expected = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "task": identity["task"],
        "task_type": identity["task_type"],
        "official_metric": identity["official_metric"],
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
        "train_n": int(identity["train_n"]),
        "valid_n": len(valid_payload["sample_index"]),
        "test_n": len(test_payload["sample_index"]),
        "prediction_rows": len(valid_payload["sample_index"])
        + len(test_payload["sample_index"]),
        "prediction_sha256": file_sha256(prediction),
        "smiles_compatibility_policy": SMILES_COMPATIBILITY_POLICY,
        "representation_repair_count": int(identity["representation_repair_count"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PredictionAuditError(
                f"{path}: {key} mismatch; expected {value!r}, observed {payload.get(key)!r}"
            )
    try:
        fit_seconds = float(payload["fit_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionAuditError(f"{path}: invalid fit_seconds") from exc
    if not math.isfinite(fit_seconds) or fit_seconds < 0:
        raise PredictionAuditError(f"{path}: invalid fit_seconds")
    return round(fit_seconds, 3), True


def _grid_row_from_prediction(
    frame: pd.DataFrame,
    *,
    identity: dict[str, object],
    model: str,
    seed: int,
    split_source: str,
    radius: int,
    n_bits: int,
    fit_seconds: float | str,
) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDNAMES}
    valid = frame.loc[frame["split"] == "valid"]
    test = frame.loc[frame["split"] == "test"]
    metric_function = (
        classification_metrics
        if identity["task_type"] == "classification"
        else regression_metrics
    )
    valid_metrics = metric_function(
        valid["y_true"].to_numpy(dtype=float), valid["y_pred"].to_numpy(dtype=float)
    )
    test_metrics = metric_function(
        test["y_true"].to_numpy(dtype=float), test["y_pred"].to_numpy(dtype=float)
    )
    row.update(
        {
            "task": identity["task"],
            "task_type": identity["task_type"],
            "official_metric": identity["official_metric"],
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
            "train_n": int(identity["train_n"]),
            "valid_n": len(valid),
            "test_n": len(test),
            "fit_seconds": fit_seconds,
        }
    )
    row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def rebuild_grid_from_predictions(
    *,
    prediction_dir: Path,
    output_path: Path,
    audit_path: Path,
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    radius: int,
    n_bits: int,
    data_dir: str,
    split_source: str,
) -> dict[str, object]:
    """Rebuild all grid metrics from cached predictions without model fitting."""

    expected_predictions = {
        prediction_path(prediction_dir, task, model, seed).resolve()
        for task in tasks
        for model in models
        for seed in seeds
    }
    actual_predictions = {
        path.resolve()
        for path in prediction_dir.rglob("*.csv")
        if not path.name.startswith("._")
    }
    expected_manifests = {
        prediction_manifest_path(path).resolve() for path in expected_predictions
    }
    actual_manifests = {
        path.resolve()
        for path in prediction_dir.rglob("*.manifest.json")
        if not path.name.startswith("._")
    }
    errors: list[dict[str, object]] = []
    for path in sorted(actual_predictions - expected_predictions):
        errors.append({"path": str(path), "reason": "unexpected_prediction"})
    for path in sorted(actual_manifests - expected_manifests):
        errors.append({"path": str(path), "reason": "unexpected_timing_manifest"})

    rows: list[dict[str, object]] = []
    timing_manifest_runs = 0
    missing_timing_manifest_runs = 0
    for task in tasks:
        for seed in seeds:
            try:
                split = load_tdc_official(
                    task,
                    data_dir=data_dir,
                    seed=seed,
                    split_source=split_source,
                )
                identity = _expected_identity(split)
            except Exception as exc:
                errors.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "reason": "source_identity_reconstruction_failed",
                        "detail": repr(exc),
                    }
                )
                continue
            for model in models:
                path = prediction_path(prediction_dir, task, model, seed)
                try:
                    frame = audit_prediction_file(
                        path, identity=identity, model=model, seed=seed
                    )
                    fit_seconds, has_timing = _load_fit_seconds(
                        path,
                        identity=identity,
                        model=model,
                        seed=seed,
                        split_source=split_source,
                        radius=radius,
                        n_bits=n_bits,
                    )
                    timing_manifest_runs += int(has_timing)
                    missing_timing_manifest_runs += int(not has_timing)
                    rows.append(
                        _grid_row_from_prediction(
                            frame,
                            identity=identity,
                            model=model,
                            seed=seed,
                            split_source=split_source,
                            radius=radius,
                            n_bits=n_bits,
                            fit_seconds=fit_seconds,
                        )
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "task": task,
                            "model": model,
                            "seed": int(seed),
                            "path": str(path),
                            "reason": "prediction_audit_failed",
                            "detail": str(exc),
                        }
                    )

    expected_runs = len(tasks) * len(models) * len(seeds)
    if len(rows) != expected_runs and not errors:
        errors.append(
            {
                "reason": "run_count_mismatch",
                "expected": expected_runs,
                "observed": len(rows),
            }
        )
    audit: dict[str, object] = {
        "schema": REBUILD_AUDIT_SCHEMA,
        "status": "PASS" if not errors and len(rows) == expected_runs else "HOLD",
        "prediction_dir": str(prediction_dir),
        "output_path": str(output_path),
        "split_source": split_source,
        "split_algorithm": (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        ),
        "radius": int(radius),
        "n_bits": int(n_bits),
        "expected_runs": expected_runs,
        "validated_runs": len(rows),
        "timing_manifest_runs": timing_manifest_runs,
        "fit_seconds_na_runs": missing_timing_manifest_runs,
        "errors": errors,
    }
    _write_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise PredictionAuditError(
            f"traditional prediction rebuild HOLD: {len(errors)} audit error(s); "
            f"see {audit_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(output_path)
    return audit


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (row["task"].lower(), row["model"], int(row["seed"]))
            for row in reader
            if row.get("task") and row.get("model") and row.get("seed")
        }


def run_one(
    task: str,
    model_name: str,
    seed: int,
    radius: int,
    n_bits: int,
    data_dir: str,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
) -> dict:
    set_seed(seed)
    split = load_tdc_official(
        task, data_dir=data_dir, seed=seed, split_source=split_source
    )
    train_smiles, train_y = extract_xy(split.train)
    valid_smiles, valid_y = extract_xy(split.valid)
    test_smiles, test_y = extract_xy(split.test)

    x_train, train_idx = morgan_count_matrix(
        _model_inputs(train_smiles), radius=radius, n_bits=n_bits
    )
    x_valid, valid_idx = morgan_count_matrix(
        _model_inputs(valid_smiles), radius=radius, n_bits=n_bits
    )
    x_test, test_idx = morgan_count_matrix(
        _model_inputs(test_smiles), radius=radius, n_bits=n_bits
    )
    y_train = align_labels(train_y, train_idx)
    y_valid = align_labels(valid_y, valid_idx)
    y_test = align_labels(test_y, test_idx)

    model = make_traditional_model(model_name, split.task_type, seed=seed)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    valid_pred = predict_scores(model, x_valid, split.task_type)
    test_pred = predict_scores(model, x_test, split.task_type)
    if split.task_type == "classification":
        valid_metrics = classification_metrics(y_valid, valid_pred)
        test_metrics = classification_metrics(y_test, test_pred)
    else:
        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred)

    row = {
        "task": split.name,
        "task_type": split.task_type,
        "official_metric": split.metric,
        "model": model_name,
        "seed": seed,
        "split_source": split_source,
        "split_algorithm": (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        ),
        "radius": radius,
        "n_bits": n_bits,
        "train_n": len(y_train),
        "valid_n": len(y_valid),
        "test_n": len(y_test),
        "fit_seconds": round(fit_seconds, 3),
    }
    for key, value in valid_metrics.items():
        row[f"valid_{key}"] = value
    for key, value in test_metrics.items():
        row[f"test_{key}"] = value
    return row


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_failure(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional ML grid on TDC official splits.")
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
        "--out",
        default=str(ROOT / "results" / "tables" / "traditional_grid_locked_cache.csv"),
    )
    parser.add_argument(
        "--failures",
        default=str(
            ROOT / "results" / "tables" / "traditional_grid_locked_cache_failures.jsonl"
        ),
    )
    parser.add_argument(
        "--from-predictions",
        type=Path,
        help=(
            "Strictly rebuild the grid from this prediction root instead of fitting "
            "models. Missing, extra, misaligned, or non-finite artifacts cause HOLD."
        ),
    )
    parser.add_argument(
        "--rebuild-audit",
        type=Path,
        default=ROOT
        / "results"
        / "tables"
        / "traditional_grid_locked_cache_rebuild_audit.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    failures = Path(args.failures)
    if len(args.tasks) != len(set(args.tasks)):
        raise SystemExit("--tasks contains duplicates")
    if len(args.models) != len(set(args.models)):
        raise SystemExit("--models contains duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("--seeds contains duplicates")
    if args.from_predictions is not None:
        if args.resume:
            raise SystemExit("--resume is not used with --from-predictions")
        try:
            audit = rebuild_grid_from_predictions(
                prediction_dir=args.from_predictions,
                output_path=out,
                audit_path=args.rebuild_audit,
                tasks=list(args.tasks),
                models=list(args.models),
                seeds=list(args.seeds),
                radius=args.radius,
                n_bits=args.n_bits,
                data_dir=args.data_dir,
                split_source=args.split_source,
            )
        except PredictionAuditError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"rebuilt_without_training: {audit['validated_runs']}/{audit['expected_runs']} "
            f"fit_seconds_na={audit['fit_seconds_na_runs']} out={out}",
            flush=True,
        )
        return
    done = existing_keys(out) if args.resume else set()

    for task in args.tasks:
        for seed in args.seeds:
            for model_name in args.models:
                key = (task.lower(), model_name, seed)
                if args.resume and key in done:
                    print(f"skip existing: task={task} model={model_name} seed={seed}")
                    continue
                print(f"run: task={task} model={model_name} seed={seed}", flush=True)
                try:
                    row = run_one(
                        task=task,
                        model_name=model_name,
                        seed=seed,
                        radius=args.radius,
                        n_bits=args.n_bits,
                        data_dir=args.data_dir,
                        split_source=args.split_source,
                    )
                    append_row(out, row)
                    print(f"done: {row['task']} {model_name} seed={seed}", flush=True)
                except Exception as exc:
                    payload = {
                        "task": task,
                        "model": model_name,
                        "seed": seed,
                        "error": repr(exc),
                    }
                    append_failure(failures, payload)
                    print(f"failed: {payload}", flush=True)


if __name__ == "__main__":
    main()
