#!/usr/bin/env python3
"""Freeze a validation-only reliability selector before reading test targets.

The selector is deliberately a retrospective analysis rule.  It is intended
to test whether validation performance and a small set of reliability signals
give the same model-family choice on the ExpansionRx replication data.  The
script fails closed when run artifacts are incomplete or validation identities
are misaligned.  Test labels are optional and, when supplied, are opened only
after ``selection_freeze.json`` has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_index",
    "measurement_identity",
    "connectivity_key",
    "parent_tautomer_key",
    "smiles",
    "split",
    "task",
    "model",
    "seed",
    "prediction",
    "validation_target",
}
MODEL_ORDER = ("chemprop_dmpnn", "gin", "lightgbm", "rf", "xgboost")
LOWER_IS_BETTER = {"mae", "brier", "ece"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rank_score(values: pd.Series, lower_is_better: bool) -> pd.Series:
    """Map a metric to [0, 1], with one as the best family."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Non-finite metric values: {list(values.index[numeric.isna()])}")
    ranks = numeric.rank(method="average", ascending=lower_is_better)
    if len(ranks) <= 1:
        return pd.Series(1.0, index=ranks.index)
    return 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return float("nan")
    left_rank = frame["left"].rank(method="average").to_numpy(float)
    right_rank = frame["right"].rank(method="average").to_numpy(float)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = y_true == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[positive].sum())
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (score >= left) & (score <= right)
        else:
            mask = (score >= left) & (score < right)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(score[mask].mean())
            )
    return float(result)


def infer_prepared_dir(prediction_dirs: list[Path], explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = []
    for directory in prediction_dirs:
        for parent in (directory, *directory.parents):
            candidate = parent / "prepared"
            if candidate.is_dir():
                candidates.append(candidate)
                break
    unique = sorted({path.resolve() for path in candidates})
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(f"Prediction directories imply multiple prepared dirs: {unique}")
    return unique[0]


def find_prediction_files(prediction_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in prediction_dirs:
        for path in sorted(directory.rglob("*.csv")):
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception:
                continue
            if REQUIRED_COLUMNS <= columns:
                files.append(path)
    if not files:
        raise FileNotFoundError("No compatible prediction CSV files were found")
    return files


def load_predictions(prediction_dirs: list[Path]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    files = find_prediction_files(prediction_dirs)
    seen: dict[tuple[str, str, int], Path] = {}
    for path in files:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        keys = frame[["task", "model", "seed"]].drop_duplicates()
        if len(keys) != 1:
            raise ValueError(f"Mixed run metadata in {path}")
        key_row = keys.iloc[0]
        key = (str(key_row.task), str(key_row.model), int(key_row.seed))
        if key in seen:
            raise ValueError(f"Duplicate run {key}: {seen[key]} and {path}")
        seen[key] = path
        frame["task"] = frame["task"].astype(str)
        frame["model"] = frame["model"].astype(str)
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise").astype(float)
        frame["validation_target"] = pd.to_numeric(frame["validation_target"], errors="coerce").astype(float)
        valid = frame[frame["split"].astype(str).str.lower() == "valid"].copy()
        if valid.empty:
            raise ValueError(f"No validation rows in {path}")
        if valid["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate validation sample_index in {path}")
        if valid["validation_target"].isna().any():
            raise ValueError(f"Validation targets missing in {path}")
        if not np.isfinite(valid["prediction"]).all():
            raise ValueError(f"Non-finite validation predictions in {path}")
        frames.append(valid)
    return pd.concat(frames, ignore_index=True), files


def fingerprint_support(task: str, seed: int, validation: pd.DataFrame, prepared_dir: Path) -> np.ndarray:
    """Compute max Morgan similarity to the seed-specific training identities."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    task_dir = prepared_dir / task
    development = pd.read_csv(task_dir / "development.csv")
    split = json.loads((task_dir / "splits.json").read_text())[str(seed)]
    train = development.iloc[list(split["train"])]
    train_fps = []
    for smiles in train["smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))
    if not train_fps:
        raise ValueError(f"No training fingerprints for {task}/seed{seed}")
    values = []
    for smiles in validation["smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            values.append(float("nan"))
            continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        values.append(float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))))
    return np.asarray(values, dtype=float)


def task_type(y_true: pd.Series) -> str:
    values = np.asarray(y_true, dtype=float)
    unique = set(np.unique(values).tolist())
    return "classification" if unique <= {0.0, 1.0} and len(unique) == 2 else "regression"


def metric_rows_for_run(frame: pd.DataFrame, support: np.ndarray) -> dict[str, float | str | int]:
    y_true = frame["validation_target"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    residual = np.abs(y_true - prediction)
    kind = task_type(frame["validation_target"])
    row: dict[str, float | str | int] = {
        "task": str(frame["task"].iloc[0]),
        "model": str(frame["model"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "task_type": kind,
        "n_validation": int(len(frame)),
        "validation_mae": float(np.mean(residual)),
        "support_error_spearman": spearman(support, -residual),
    }
    if kind == "classification":
        clipped = np.clip(prediction, 0.0, 1.0)
        row.update(
            {
                "validation_auroc": auroc(y_true, clipped),
                "validation_brier": float(np.mean((y_true - clipped) ** 2)),
                "validation_ece": expected_calibration_error(y_true, clipped),
            }
        )
    else:
        row.update(
            {
                "validation_auroc": float("nan"),
                "validation_brier": float("nan"),
                "validation_ece": float("nan"),
            }
        )
    return row


def select_family(aggregate: pd.DataFrame, kind: str) -> tuple[str, pd.DataFrame]:
    metrics = ["validation_mae", "support_error_spearman"]
    if kind == "classification":
        metrics = [
            "validation_auroc",
            "validation_brier",
            "validation_ece",
            "support_error_spearman",
        ]
    working = aggregate.copy()
    rank_columns = []
    for metric in metrics:
        if not np.isfinite(working[metric].to_numpy(float)).all():
            raise ValueError(f"Non-finite aggregate {metric} for {kind}: {working[metric].tolist()}")
        column = metric + "_rank_score"
        working[column] = rank_score(working[metric], metric.removeprefix("validation_") in LOWER_IS_BETTER)
        rank_columns.append(column)
    working["vrrs_score"] = working[rank_columns].mean(axis=1)
    # A separate MAE-only selection is retained as the conventional comparator.
    mae_sorted = working.sort_values(["validation_mae", "model"], ascending=[True, True])
    vrrs_sorted = working.sort_values(["vrrs_score", "model"], ascending=[False, True])
    working["validation_mae_selected_model"] = mae_sorted.iloc[0]["model"]
    working["vrrs_selected_model"] = vrrs_sorted.iloc[0]["model"]
    return str(vrrs_sorted.iloc[0]["model"]), working


def validation_analysis(
    predictions: pd.DataFrame,
    prepared_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if prepared_dir is None:
        raise ValueError("A prepared directory is required for support/error analysis")
    run_rows: list[dict] = []
    for (task, model, seed), frame in predictions.groupby(["task", "model", "seed"], sort=True):
        if not (prepared_dir / str(task) / "development.csv").exists():
            raise FileNotFoundError(f"Missing prepared task directory for {task}")
        support = fingerprint_support(str(task), int(seed), frame, prepared_dir)
        run_rows.append(metric_rows_for_run(frame.reset_index(drop=True), support))
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        raise ValueError("No validation run rows")

    selection_rows: list[pd.DataFrame] = []
    selected_map: dict[str, str] = {}
    for task, task_runs in runs.groupby("task", sort=True):
        kind = str(task_runs["task_type"].iloc[0])
        aggregate = task_runs.groupby("model", as_index=False).agg(
            task_type=("task_type", "first"),
            n_seeds=("seed", "nunique"),
            validation_mae=("validation_mae", "mean"),
            support_error_spearman=("support_error_spearman", "mean"),
            validation_auroc=("validation_auroc", "mean"),
            validation_brier=("validation_brier", "mean"),
            validation_ece=("validation_ece", "mean"),
        )
        if (aggregate["n_seeds"] != 5).any():
            raise ValueError(f"Expected five seeds for {task}: {aggregate[['model', 'n_seeds']].to_dict('records')}")
        selected, scored = select_family(aggregate, kind)
        selected_map[str(task)] = selected
        scored.insert(0, "task", str(task))
        scored["vrrs_selected"] = scored["model"].eq(selected)
        selection_rows.append(scored)
    selection = pd.concat(selection_rows, ignore_index=True)

    crossfit_rows: list[dict] = []
    for task, task_runs in runs.groupby("task", sort=True):
        kind = str(task_runs["task_type"].iloc[0])
        for heldout_seed in sorted(task_runs["seed"].unique()):
            development = task_runs[task_runs["seed"] != heldout_seed]
            aggregate = development.groupby("model", as_index=False).agg(
                validation_mae=("validation_mae", "mean"),
                support_error_spearman=("support_error_spearman", "mean"),
                validation_auroc=("validation_auroc", "mean"),
                validation_brier=("validation_brier", "mean"),
                validation_ece=("validation_ece", "mean"),
            )
            selected, _ = select_family(
                aggregate.assign(task_type=kind, n_seeds=4), kind
            )
            heldout = task_runs[task_runs["seed"] == heldout_seed].set_index("model")
            selected_row = heldout.loc[selected]
            best_heldout = (
                heldout.sort_values(["validation_mae", "model"], ascending=[True, True]).index[0]
            )
            crossfit_rows.append(
                {
                    "task": str(task),
                    "task_type": kind,
                    "heldout_seed": int(heldout_seed),
                    "selected_from_other_seeds": selected,
                    "full_vrrs_selected": selected_map[str(task)],
                    "selection_agrees_with_full_vrrs": selected == selected_map[str(task)],
                    "heldout_validation_mae": float(selected_row["validation_mae"]),
                    "heldout_best_mae_model": str(best_heldout),
                    "heldout_selected_is_mae_best": selected == best_heldout,
                }
            )
    crossfit = pd.DataFrame(crossfit_rows)
    summary = {
        "tasks": sorted(selected_map),
        "task_count": len(selected_map),
        "model_families": sorted(selection["model"].unique().tolist()),
        "run_count": int(len(runs)),
        "selection_map": selected_map,
        "vrrs_stability_rate": float(crossfit["selection_agrees_with_full_vrrs"].mean()),
        "crossfit_rows": int(len(crossfit)),
        "test_outcomes_used_for_selection": False,
    }
    return selection, crossfit, {"runs": runs, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_replication_20261026/validation_selector"),
    )
    parser.add_argument("--test-target-dir", type=Path, help="Optional labels opened only after freeze")
    args = parser.parse_args()

    prediction_dirs = [path.resolve() for path in args.prediction_dirs]
    prepared_dir = infer_prepared_dir(prediction_dirs, args.prepared_dir.resolve() if args.prepared_dir else None)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, files = load_predictions(prediction_dirs)
    selection, crossfit, details = validation_analysis(predictions, prepared_dir)
    selection_path = output_dir / "selection_before_test.csv"
    crossfit_path = output_dir / "crossfit_validation_selection.csv"
    selection.to_csv(selection_path, index=False)
    crossfit.to_csv(crossfit_path, index=False)

    freeze_payload = {
        "status": "FROZEN_BEFORE_TEST_TARGET_ACCESS",
        "protocol": "retrospective_validation_reliability_selector_v1",
        "test_outcomes_used_for_selection": False,
        "test_targets_read_before_freeze": False,
        "selection_file": selection_path.name,
        "crossfit_file": crossfit_path.name,
        "selection_sha256": sha256_file(selection_path),
        "crossfit_sha256": sha256_file(crossfit_path),
        "selection_map": details["summary"]["selection_map"],
        "prediction_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in files
        ],
        "prepared_dir": str(prepared_dir) if prepared_dir else None,
        "prepared_manifest_sha256": (
            sha256_file(prepared_dir.parent / "preparation_manifest.json")
            if prepared_dir and (prepared_dir.parent / "preparation_manifest.json").exists()
            else None
        ),
        "summary": details["summary"],
    }
    freeze_path = output_dir / "selection_freeze.json"
    atomic_json(freeze_payload, freeze_path)

    test_comparison = output_dir / "external_test_selector_comparison.csv"
    if args.test_target_dir is None:
        pd.DataFrame(
            [
                {
                    "status": "NOT_EVALUATED_TEST_TARGETS_NOT_PROVIDED",
                    "test_outcomes_used_for_selection": False,
                    "selection_freeze": freeze_path.name,
                }
            ]
        ).to_csv(test_comparison, index=False)
    else:
        raise NotImplementedError(
            "Test-target evaluation is intentionally disabled in this first freeze pass; "
            "use a separate post-freeze evaluator after checking selection_freeze.json."
        )

    manifest = {
        "status": "COMPLETE_VALIDATION_ONLY_FREEZE",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "prediction_dirs": [str(path) for path in prediction_dirs],
        "prepared_dir": str(prepared_dir) if prepared_dir else None,
        "output_files": {
            name: {"path": str(output_dir / name), "sha256": sha256_file(output_dir / name)}
            for name in (
                "selection_before_test.csv",
                "crossfit_validation_selection.csv",
                "selection_freeze.json",
                "external_test_selector_comparison.csv",
            )
        },
        "test_targets_read": False,
        "summary": details["summary"],
    }
    atomic_json(manifest, output_dir / "analysis_manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
