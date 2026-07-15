from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.calibration import (
    BinaryProbabilityCalibrator,
    ConformalRegressor,
    interval_coverage,
)
from admet_benchmark.metrics import (
    classification_metrics,
    expected_calibration_error,
    regression_metrics,
    reliability_bins,
)
from experiments.analyze_reliability_regret import (
    LEGACY_EXCLUDED_MODELS,
    _normalize_prediction_file,
    _prediction_files,
)


SUMMARY_FIELDS = [
    "task",
    "task_type",
    "official_metric",
    "model",
    "seed",
    "method",
    "test_auroc_before",
    "test_auroc_after",
    "test_auprc_before",
    "test_auprc_after",
    "test_brier_before",
    "test_brier_after",
    "test_ece_before",
    "test_ece_after",
    "test_mae",
    "test_rmse",
    "test_spearman",
    "coverage",
    "mean_interval_width",
    "error",
]


def analyze_frame(
    df: pd.DataFrame, source: str | Path = "prediction frame"
) -> tuple[list[dict], list[dict]]:
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()
    if valid.empty or test.empty:
        raise ValueError(f"Missing valid/test predictions in {source}")
    first = test.iloc[0]
    task = first["task"]
    task_type = first["task_type"]
    metric = first["official_metric"]
    model = first["model"]
    seed = int(first["seed"])

    rows: list[dict] = []
    bins_rows: list[dict] = []
    if task_type == "classification":
        y_valid = valid["y_true"].astype(int).to_numpy()
        p_valid = valid["y_pred"].astype(float).to_numpy()
        y_test = test["y_true"].astype(int).to_numpy()
        p_test = test["y_pred"].astype(float).to_numpy()
        before_metrics = classification_metrics(y_test, p_test)
        before_ece = expected_calibration_error(y_test, p_test)

        # The uncalibrated curve is shared by both intervention comparisons and
        # belongs in the artifact exactly once per task/model/seed.
        for item in reliability_bins(y_test, p_test):
            bins_rows.append(
                {
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "method": "uncalibrated",
                    **item,
                }
            )

        for method in ["isotonic", "platt"]:
            row = {
                "task": task,
                "task_type": task_type,
                "official_metric": metric,
                "model": model,
                "seed": seed,
                "method": method,
                "test_auroc_before": before_metrics["auroc"],
                "test_auprc_before": before_metrics["auprc"],
                "test_brier_before": before_metrics["brier"],
                "test_ece_before": before_ece,
            }
            try:
                calibrator = BinaryProbabilityCalibrator(method).fit(p_valid, y_valid)
                p_cal = calibrator.calibrate(p_test)
                after_metrics = classification_metrics(y_test, p_cal)
                row.update(
                    {
                        "test_auroc_after": after_metrics["auroc"],
                        "test_auprc_after": after_metrics["auprc"],
                        "test_brier_after": after_metrics["brier"],
                        "test_ece_after": expected_calibration_error(y_test, p_cal),
                    }
                )
                for item in reliability_bins(y_test, p_cal):
                    bins_rows.append({"task": task, "model": model, "seed": seed, "method": method, **item})
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
    else:
        y_valid = valid["y_true"].astype(float).to_numpy()
        p_valid = valid["y_pred"].astype(float).to_numpy()
        y_test = test["y_true"].astype(float).to_numpy()
        p_test = test["y_pred"].astype(float).to_numpy()
        metrics = regression_metrics(y_test, p_test)
        conformal = ConformalRegressor(alpha=0.1).fit(p_valid, y_valid)
        lower, upper = conformal.predict_interval(p_test)
        rows.append(
            {
                "task": task,
                "task_type": task_type,
                "official_metric": metric,
                "model": model,
                "seed": seed,
                "method": "conformal_90",
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_spearman": metrics["spearman"],
                "coverage": interval_coverage(y_test, lower, upper),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )
    return rows, bins_rows


def analyze_file(path: Path) -> tuple[list[dict], list[dict]]:
    """Backward-compatible helper for already normalized prediction CSVs."""
    return analyze_frame(pd.read_csv(path), path)


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze validation-fitted calibration for normalized traditional and neural predictions."
    )
    parser.add_argument(
        "--pred-dir",
        "--prediction-dir",
        dest="prediction_dir",
        default=str(ROOT / "results" / "predictions" / "traditional_local_prelim"),
    )
    parser.add_argument(
        "--additional-prediction-dirs",
        nargs="*",
        default=[],
        help="Additional formal GIN or Chemprop prediction directories.",
    )
    parser.add_argument("--summary", default=str(ROOT / "results" / "tables" / "calibration_summary_local_prelim.csv"))
    parser.add_argument("--bins", default=str(ROOT / "results" / "tables" / "calibration_reliability_bins_local_prelim.csv"))
    args = parser.parse_args()

    prediction_dirs = [
        Path(args.prediction_dir),
        *[Path(path) for path in args.additional_prediction_dirs],
    ]
    files = _prediction_files(prediction_dirs)
    if not files:
        raise FileNotFoundError(
            "No prediction CSV files found under "
            + ", ".join(str(path) for path in prediction_dirs)
        )
    all_rows: list[dict] = []
    all_bins: list[dict] = []
    included_runs: dict[tuple[str, str, int], Path] = {}
    for path in sorted(files):
        frame, reason = _normalize_prediction_file(path)
        if frame is None:
            raise ValueError(f"Invalid prediction artifact {path}: {reason}")
        if set(frame["model"].astype(str)) & LEGACY_EXCLUDED_MODELS:
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
        if key in included_runs:
            raise ValueError(
                f"Duplicate prediction run {key}: {included_runs[key]} and {path}"
            )
        included_runs[key] = path
        rows, bins = analyze_frame(frame, path)
        all_rows.extend(rows)
        all_bins.extend(bins)
    write_rows(Path(args.summary), all_rows, SUMMARY_FIELDS)
    write_rows(Path(args.bins), all_bins)
    print(f"prediction_files_discovered: {len(files)}")
    print(f"prediction_runs_included: {len(included_runs)}")
    print(f"summary_rows: {len(all_rows)}")
    print(f"reliability_bin_rows: {len(all_bins)}")


if __name__ == "__main__":
    main()
