"""Zero-model-fit sensitivity analyses for ensemble size and ECE binning."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.calibration import BinaryProbabilityCalibrator
from admet_benchmark.data_loader import TDC_TASKS
from admet_benchmark.metrics import classification_metrics, expected_calibration_error
from experiments.analyze_reliability_regret import (
    RunPrediction,
    audit_run_sample_alignment,
    aurc_from_curve,
    load_runs,
    official_metric,
    risk_coverage_curve,
    sample_rows_match,
)


SCHEMA = "posthoc_ablation_v1"
DEFAULT_MODELS = ["rf", "xgboost", "lightgbm"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def equal_mass_ece(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> tuple[float, int]:
    """Tie-preserving quantile-bin ECE and the realized number of bins."""

    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    try:
        edges = np.quantile(probabilities, quantiles, method="linear")
    except TypeError:  # NumPy < 1.22
        edges = np.quantile(probabilities, quantiles, interpolation="linear")
    edges = np.unique(edges)
    if len(edges) <= 1:
        return float(abs(np.mean(labels) - np.mean(probabilities))), 1
    assignments = np.searchsorted(edges[1:-1], probabilities, side="right")
    ece = 0.0
    actual = 0
    for bin_id in np.unique(assignments):
        mask = assignments == bin_id
        if not np.any(mask):
            continue
        actual += 1
        ece += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
        )
    return float(ece), actual


def calibration_binning_rows(
    runs: dict[tuple[str, str, int], RunPrediction], bin_counts: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, run in sorted(runs.items()):
        if run.task_type != "classification":
            continue
        y_valid = run.valid["y_true"].to_numpy(dtype=int)
        p_valid = run.valid["y_pred"].to_numpy(dtype=float)
        y_test = run.test["y_true"].to_numpy(dtype=int)
        p_test = run.test["y_pred"].to_numpy(dtype=float)
        predictions = {"uncalibrated": p_test}
        for method in ("isotonic", "platt"):
            predictions[method] = BinaryProbabilityCalibrator(method).fit(
                p_valid, y_valid
            ).calibrate(p_test)
        for method, probability in predictions.items():
            metrics = classification_metrics(y_test, probability)
            brier = float(metrics["brier"])
            nll = negative_log_likelihood(y_test, probability)
            for n_bins in bin_counts:
                equal_mass, actual_bins = equal_mass_ece(
                    y_test, probability, n_bins
                )
                rows.extend(
                    [
                        {
                            "task": run.task,
                            "model": run.model,
                            "seed": run.seed,
                            "method": method,
                            "binning": "equal_width",
                            "n_bins_requested": n_bins,
                            "n_bins_actual": n_bins,
                            "ece": expected_calibration_error(
                                y_test, probability, n_bins=n_bins
                            ),
                            "brier": brier,
                            "nll": nll,
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                        },
                        {
                            "task": run.task,
                            "model": run.model,
                            "seed": run.seed,
                            "method": method,
                            "binning": "equal_mass_tie_preserving",
                            "n_bins_requested": n_bins,
                            "n_bins_actual": actual_bins,
                            "ece": equal_mass,
                            "brier": brier,
                            "nll": nll,
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                        },
                    ]
                )
    return pd.DataFrame(rows)


def ensemble_size_rows(
    runs: dict[tuple[str, str, int], RunPrediction],
    ensemble_sizes: list[int],
    expected_seeds: list[int],
) -> pd.DataFrame:
    grouped: dict[tuple[str, str], list[RunPrediction]] = {}
    for run in runs.values():
        grouped.setdefault((run.task, run.model), []).append(run)
    rows: list[dict[str, object]] = []
    for (task, model), model_runs in sorted(grouped.items()):
        by_seed = {run.seed: run for run in model_runs}
        if set(by_seed) != set(expected_seeds):
            raise ValueError(
                f"Seed set mismatch for {(task, model)}: {sorted(by_seed)}"
            )
        reference = by_seed[expected_seeds[0]].test
        if not all(sample_rows_match(run.test, reference) for run in model_runs):
            raise ValueError(f"Test identity mismatch across seeds for {(task, model)}")
        reference = reference.sort_values("sample_index").reset_index(drop=True)
        labels = reference["y_true"].to_numpy(dtype=float)
        sample_index = reference["sample_index"].to_numpy(dtype=int)
        for ensemble_size in ensemble_sizes:
            for subset in itertools.combinations(expected_seeds, ensemble_size):
                matrix = np.vstack(
                    [
                        by_seed[seed]
                        .test.sort_values("sample_index")["y_pred"]
                        .to_numpy(dtype=float)
                        for seed in subset
                    ]
                )
                prediction = np.mean(matrix, axis=0)
                uncertainty = np.std(matrix, axis=0, ddof=1)
                if by_seed[subset[0]].task_type == "classification":
                    losses = (prediction - labels) ** 2
                    ece = expected_calibration_error(
                        labels.astype(int), prediction, n_bins=10
                    )
                    brier = float(np.mean(losses))
                    nll = negative_log_likelihood(labels, prediction)
                else:
                    losses = np.abs(prediction - labels)
                    ece = brier = nll = float("nan")
                curve = risk_coverage_curve(losses, uncertainty, sample_index)
                aurc, oracle, excess = aurc_from_curve(curve)
                mean_loss = float(np.mean(losses))
                rows.append(
                    {
                        "task": task,
                        "task_type": by_seed[subset[0]].task_type,
                        "official_metric": by_seed[subset[0]].official_metric,
                        "model": model,
                        "ensemble_size": ensemble_size,
                        "seed_subset": ";".join(str(seed) for seed in subset),
                        "test_n": len(labels),
                        "test_official_metric": official_metric(
                            by_seed[subset[0]].official_metric, labels, prediction
                        ),
                        "ece_10_equal_width": ece,
                        "brier": brier,
                        "nll": nll,
                        "mean_sample_loss": mean_loss,
                        "aurc": aurc,
                        "oracle_aurc": oracle,
                        "excess_aurc": excess,
                        "relative_excess_aurc": excess / max(mean_loss, 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def summarize_ensemble_sizes(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_official_metric",
        "ece_10_equal_width",
        "brier",
        "nll",
        "relative_excess_aurc",
    ]
    output: list[dict[str, object]] = []
    for key, group in rows.groupby(
        ["task", "task_type", "official_metric", "model", "ensemble_size"],
        sort=True,
    ):
        row = dict(
            zip(
                ["task", "task_type", "official_metric", "model", "ensemble_size"],
                key,
            )
        )
        row["n_seed_subsets"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(row)
    return pd.DataFrame(output)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-fit ensemble-size and ECE-binning sensitivity analysis."
    )
    parser.add_argument(
        "--prediction-dir",
        default=str(ROOT / "results" / "predictions" / "traditional_local_prelim"),
    )
    parser.add_argument("--additional-prediction-dirs", nargs="*", default=[])
    parser.add_argument("--tasks", nargs="*", default=TDC_TASKS)
    parser.add_argument("--expected-models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ensemble-sizes", nargs="*", type=int, default=[2, 3, 5])
    parser.add_argument("--bin-counts", nargs="*", type=int, default=[5, 10, 15, 20])
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "tables" / "posthoc_ablations_local_prelim"),
    )
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)) or len(args.expected_models) != len(
        set(args.expected_models)
    ):
        raise ValueError("Seeds and expected models must be unique")
    if any(size < 2 or size > len(args.seeds) for size in args.ensemble_sizes):
        raise ValueError("Ensemble sizes must be between 2 and the seed count")
    prediction_dirs = [
        Path(args.prediction_dir),
        *[Path(path) for path in args.additional_prediction_dirs],
    ]
    out_dir = Path(args.out_dir)
    runs, input_audit = load_runs(
        prediction_dirs,
        out_dir / "_no_similarity_inputs",
        require_similarity=False,
    )
    expected_tasks = {task.lower() for task in args.tasks}
    expected_keys = {
        (task, model.lower(), int(seed))
        for task in expected_tasks
        for model in args.expected_models
        for seed in args.seeds
    }
    observed_keys = set(runs)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    alignment, aligned = audit_run_sample_alignment(runs)
    if missing or unexpected or not aligned:
        raise ValueError(
            "Ablation input gate HOLD: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, aligned={aligned}"
        )

    ensemble = ensemble_size_rows(runs, args.ensemble_sizes, args.seeds)
    ensemble_summary = summarize_ensemble_sizes(ensemble)
    calibration = calibration_binning_rows(runs, args.bin_counts)
    combinations_per_model = sum(
        math.comb(len(args.seeds), size) for size in args.ensemble_sizes
    )
    classification_tasks = sum(
        1
        for task in expected_tasks
        if runs[(task, args.expected_models[0].lower(), args.seeds[0])].task_type
        == "classification"
    )
    expected_ensemble_rows = (
        len(expected_tasks) * len(args.expected_models) * combinations_per_model
    )
    expected_calibration_rows = (
        classification_tasks
        * len(args.expected_models)
        * len(args.seeds)
        * 3
        * len(args.bin_counts)
        * 2
    )
    if len(ensemble) != expected_ensemble_rows or len(calibration) != expected_calibration_rows:
        raise ValueError(
            "Ablation output count mismatch: "
            f"ensemble={len(ensemble)}/{expected_ensemble_rows}, "
            f"calibration={len(calibration)}/{expected_calibration_rows}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(out_dir / "input_prediction_audit.csv", input_audit)
    _atomic_csv(out_dir / "sample_alignment_audit.csv", alignment)
    _atomic_csv(out_dir / "ensemble_size_by_seed_subset.csv", ensemble)
    _atomic_csv(out_dir / "ensemble_size_summary.csv", ensemble_summary)
    _atomic_csv(out_dir / "calibration_binning_sensitivity.csv", calibration)
    gate = {
        "schema": SCHEMA,
        "status": "PASS",
        "zero_model_fits": True,
        "expected_runs": len(expected_keys),
        "observed_runs": len(observed_keys),
        "all_task_seed_model_samples_aligned": bool(aligned),
        "ensemble_sizes": args.ensemble_sizes,
        "seed_subsets_per_task_model": combinations_per_model,
        "ensemble_rows": len(ensemble),
        "bin_counts": args.bin_counts,
        "binning_schemes": ["equal_width", "equal_mass_tie_preserving"],
        "calibration_methods": ["uncalibrated", "isotonic", "platt"],
        "calibration_rows": len(calibration),
        "prediction_dirs": [str(path) for path in prediction_dirs],
    }
    temporary = out_dir / f"ablation_gate.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    temporary.replace(out_dir / "ablation_gate.json")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
