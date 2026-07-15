from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    TDC_TASK_META,
    TDC_TASKS,
)
from experiments.run_shap_selected import SELECTED as SHAP_SELECTED  # noqa: E402


EXPECTED_Y_TRIALS = 10
EXPECTED_SHAP_RANKS = 20
EXPECTED_MAPPING_RANKS = 10
TRADITIONAL_MODELS = {"rf", "xgboost", "lightgbm"}


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer, got {value!r}")
    return int(numeric)


def _read_csv_snapshot(path: Path) -> tuple[bytes, pd.DataFrame]:
    raw = path.read_bytes()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"invalid CSV {path}: {exc}") from exc
    return raw, frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_task(value: object) -> str | None:
    lookup = {task.lower(): task for task in TDC_TASKS}
    return lookup.get(str(value).lower())


def _all_finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    try:
        values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy()
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all())


def _required_columns(
    frame: pd.DataFrame, required: set[str], label: str, failures: list[str]
) -> bool:
    missing = sorted(required.difference(frame.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")
        return False
    return True


def _task_model_map(
    selection: pd.DataFrame, failures: list[str]
) -> dict[str, str]:
    if not _required_columns(selection, {"task", "model"}, "selection", failures):
        return {}
    mapping: dict[str, str] = {}
    for row in selection.itertuples(index=False):
        task = _canonical_task(row.task)
        model = str(row.model)
        if task is None:
            failures.append(f"selection contains unknown task: {row.task}")
            continue
        if task in mapping:
            failures.append(f"selection contains duplicate task: {task}")
            continue
        if model not in TRADITIONAL_MODELS:
            failures.append(f"selection contains non-traditional model for {task}: {model}")
        mapping[task] = model
    missing = sorted(set(TDC_TASKS).difference(mapping))
    if missing:
        failures.append(f"selection missing tasks: {missing}")
    if len(selection) != len(mapping):
        failures.append("selection contains duplicate or unknown rows")
    return mapping


def _validate_common_provenance(
    frame: pd.DataFrame, label: str, failures: list[str]
) -> None:
    if "split_source" in frame and set(frame["split_source"].astype(str)) != {
        LOCKED_CACHE_SPLIT_SOURCE
    }:
        failures.append(f"{label} split_source is not exclusively locked_cache")
    if "split_algorithm" in frame and set(frame["split_algorithm"].astype(str)) != {
        LOCKED_CACHE_SPLIT_ALGORITHM
    }:
        failures.append(f"{label} split_algorithm does not match the locked contract")
    if "seed" in frame:
        try:
            seeds = {
                _strict_int(value, f"{label}.seed row {index}")
                for index, value in enumerate(frame["seed"])
            }
        except ValueError:
            seeds = set()
        if seeds != {1}:
            failures.append(f"{label} seed set is {sorted(seeds)}, expected [1]")


def summarize(
    y_summary_path: Path,
    y_scores_path: Path,
    shap_path: Path,
    mapping_path: Path,
    selection_path: Path,
) -> dict:
    paths = {
        "y_randomization_summary": y_summary_path,
        "y_randomization_scores": y_scores_path,
        "shap": shap_path,
        "shap_mapping": mapping_path,
        "traditional_selection": selection_path,
    }
    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        return {
            "schema_version": 1,
            "status": "HOLD",
            "artifact_gate": "HOLD",
            "negative_control_signal": "NOT_EVALUABLE",
            "failures": [f"missing input files: {missing_files}"],
            "inputs": {name: str(path) for name, path in paths.items()},
        }

    failures: list[str] = []
    try:
        snapshots = {name: _read_csv_snapshot(path) for name, path in paths.items()}
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1,
            "status": "HOLD",
            "artifact_gate": "HOLD",
            "negative_control_signal": "NOT_EVALUABLE",
            "failures": [str(exc)],
            "inputs": {name: str(path) for name, path in paths.items()},
        }
    y_summary = snapshots["y_randomization_summary"][1]
    y_scores = snapshots["y_randomization_scores"][1]
    shap = snapshots["shap"][1]
    mapping = snapshots["shap_mapping"][1]
    selection = snapshots["traditional_selection"][1]
    selected_models = _task_model_map(selection, failures)

    y_summary_required = {
        "task",
        "model",
        "seed",
        "split_source",
        "split_algorithm",
        "metric",
        "real_score",
        "random_mean",
        "random_std",
        "oriented_margin",
        "n_trials",
    }
    if _required_columns(y_summary, y_summary_required, "y summary", failures):
        _validate_common_provenance(y_summary, "y summary", failures)
        canonical = y_summary["task"].map(_canonical_task)
        if len(y_summary) != len(TDC_TASKS):
            failures.append(
                f"y summary has {len(y_summary)} rows, expected {len(TDC_TASKS)}"
            )
        if canonical.isna().any() or set(canonical.dropna()) != set(TDC_TASKS):
            failures.append("y summary does not cover the exact 22-task contract")
        if canonical.duplicated().any():
            failures.append("y summary contains duplicate tasks")
        try:
            trial_counts = {
                _strict_int(value, f"y summary.n_trials row {index}")
                for index, value in enumerate(y_summary["n_trials"])
            }
        except ValueError:
            trial_counts = set()
        if trial_counts != {EXPECTED_Y_TRIALS}:
            failures.append("y summary n_trials is not exclusively 10")
        if not _all_finite(
            y_summary,
            ["real_score", "random_mean", "random_std", "oriented_margin"],
        ):
            failures.append("y summary contains non-finite core statistics")
        for task_value, model in zip(canonical, y_summary["model"].astype(str)):
            if task_value is not None and selected_models.get(task_value) != model:
                failures.append(
                    f"y summary model mismatch for {task_value}: {model} versus "
                    f"{selected_models.get(task_value)}"
                )
        for task_value, metric in zip(canonical, y_summary["metric"].astype(str)):
            if task_value is not None and metric != TDC_TASK_META[task_value]["metric"]:
                failures.append(
                    f"y summary metric mismatch for {task_value}: {metric} versus "
                    f"{TDC_TASK_META[task_value]['metric']}"
                )

    y_scores_required = {
        "task",
        "model",
        "seed",
        "split_source",
        "split_algorithm",
        "metric",
        "trial",
        "random_score",
    }
    if _required_columns(y_scores, y_scores_required, "y scores", failures):
        _validate_common_provenance(y_scores, "y scores", failures)
        if len(y_scores) != len(TDC_TASKS) * EXPECTED_Y_TRIALS:
            failures.append(
                f"y scores has {len(y_scores)} rows, expected "
                f"{len(TDC_TASKS) * EXPECTED_Y_TRIALS}"
            )
        if not _all_finite(y_scores, ["random_score"]):
            failures.append("y scores contains non-finite random scores")
        score_tasks = y_scores["task"].map(_canonical_task)
        if score_tasks.isna().any() or set(score_tasks.dropna()) != set(TDC_TASKS):
            failures.append("y scores does not cover the exact 22-task contract")
        for task, group in y_scores.assign(_task=score_tasks).groupby("_task"):
            try:
                trials = {
                    _strict_int(value, f"y scores.{task}.trial")
                    for value in group["trial"]
                }
            except ValueError:
                trials = set()
            if trials != set(range(1, EXPECTED_Y_TRIALS + 1)) or len(group) != EXPECTED_Y_TRIALS:
                failures.append(f"y scores trial contract failed for {task}")
            if set(group["model"].astype(str)) != {selected_models.get(task)}:
                failures.append(f"y scores model mismatch for {task}")
            summary_group = y_summary[y_summary["task"].map(_canonical_task) == task]
            if len(summary_group) == 1:
                row = summary_group.iloc[0]
                if set(group["metric"].astype(str)) != {str(row["metric"])}:
                    failures.append(f"y scores metric mismatch for {task}")
                observed_mean = float(pd.to_numeric(group["random_score"]).mean())
                if not math.isclose(
                    observed_mean,
                    float(row["random_mean"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    failures.append(f"y scores mean does not reproduce summary for {task}")
                observed_std = float(pd.to_numeric(group["random_score"]).std(ddof=1))
                if not math.isclose(
                    observed_std,
                    float(row["random_std"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    failures.append(f"y scores standard deviation does not reproduce summary for {task}")
                expected_margin = (
                    observed_mean - float(row["real_score"])
                    if str(row["metric"]) == "mae"
                    else float(row["real_score"]) - observed_mean
                )
                if not math.isclose(
                    expected_margin,
                    float(row["oriented_margin"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    failures.append(f"oriented y-randomization margin is inconsistent for {task}")

    expected_shap_tasks = {task for task, _ in SHAP_SELECTED}
    shap_required = {
        "task",
        "model",
        "seed",
        "split_source",
        "split_algorithm",
        "rank",
        "feature",
        "mean_abs_shap",
        "train_n",
        "test_n",
        "sample_size_actual",
    }
    if _required_columns(shap, shap_required, "SHAP", failures):
        _validate_common_provenance(shap, "SHAP", failures)
        shap_tasks = shap["task"].map(_canonical_task)
        if len(shap) != len(expected_shap_tasks) * EXPECTED_SHAP_RANKS:
            failures.append(
                f"SHAP has {len(shap)} rows, expected "
                f"{len(expected_shap_tasks) * EXPECTED_SHAP_RANKS}"
            )
        if shap_tasks.isna().any() or set(shap_tasks.dropna()) != expected_shap_tasks:
            failures.append("SHAP does not cover the exact eight prespecified tasks")
        if not _all_finite(
            shap, ["mean_abs_shap", "train_n", "test_n", "sample_size_actual"]
        ):
            failures.append("SHAP contains non-finite attributions")
        if (
            shap[["train_n", "test_n", "sample_size_actual"]]
            .apply(pd.to_numeric, errors="coerce")
            .le(0)
            .any()
            .any()
        ):
            failures.append("SHAP contains nonpositive sample counts")
        if (pd.to_numeric(shap["mean_abs_shap"], errors="coerce") < 0).any():
            failures.append("SHAP contains negative mean absolute attributions")
        for task, group in shap.assign(_task=shap_tasks).groupby("_task"):
            try:
                ranks = {_strict_int(value, f"SHAP.{task}.rank") for value in group["rank"]}
            except ValueError:
                ranks = set()
            if ranks != set(range(1, EXPECTED_SHAP_RANKS + 1)) or len(group) != EXPECTED_SHAP_RANKS:
                failures.append(f"SHAP rank contract failed for {task}")
            if group["feature"].astype(str).duplicated().any():
                failures.append(f"SHAP contains duplicate features for {task}")
            if not group["feature"].astype(str).str.fullmatch(r"morgan_bit_\d+").all():
                failures.append(f"SHAP feature naming failed for {task}")
            if set(group["model"].astype(str)) != {selected_models.get(task)}:
                failures.append(f"SHAP model mismatch for {task}")

    mapping_required = {
        "task",
        "model",
        "seed",
        "split_source",
        "split_algorithm",
        "rank",
        "feature",
        "bit_id",
        "mean_abs_shap",
        "example_smiles",
        "bit_atom_index",
        "bit_radius",
        "substructure_smiles",
    }
    mapped_count = 0
    if _required_columns(mapping, mapping_required, "SHAP mapping", failures):
        _validate_common_provenance(mapping, "SHAP mapping", failures)
        mapping_tasks = mapping["task"].map(_canonical_task)
        if len(mapping) != len(expected_shap_tasks) * EXPECTED_MAPPING_RANKS:
            failures.append(
                f"SHAP mapping has {len(mapping)} rows, expected "
                f"{len(expected_shap_tasks) * EXPECTED_MAPPING_RANKS}"
            )
        if mapping_tasks.isna().any() or set(mapping_tasks.dropna()) != expected_shap_tasks:
            failures.append("SHAP mapping does not cover the exact eight tasks")
        mapped = mapping["substructure_smiles"].fillna("").astype(str).str.strip() != ""
        examples = mapping["example_smiles"].fillna("").astype(str).str.strip() != ""
        mapped_count = int(mapped.sum())
        if not bool(mapped.all() and examples.all()):
            failures.append(
                f"SHAP mapping is incomplete: {mapped_count}/{len(mapping)} substructures mapped"
            )
        if not _all_finite(
            mapping, ["bit_id", "mean_abs_shap", "bit_atom_index", "bit_radius"]
        ):
            failures.append("SHAP mapping contains non-finite numeric fields")
        feature_bits = mapping["feature"].astype(str).str.extract(r"morgan_bit_(\d+)")[0]
        parsed_bits = pd.to_numeric(feature_bits, errors="coerce").to_numpy(dtype=float)
        stored_bits = pd.to_numeric(mapping["bit_id"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(parsed_bits, stored_bits):
            failures.append("SHAP mapping bit_id does not match feature name")
        for task, group in mapping.assign(_task=mapping_tasks).groupby("_task"):
            try:
                ranks = {
                    _strict_int(value, f"SHAP mapping.{task}.rank")
                    for value in group["rank"]
                }
            except ValueError:
                ranks = set()
            if ranks != set(range(1, EXPECTED_MAPPING_RANKS + 1)) or len(group) != EXPECTED_MAPPING_RANKS:
                failures.append(f"SHAP mapping rank contract failed for {task}")
            if set(group["model"].astype(str)) != {selected_models.get(task)}:
                failures.append(f"SHAP mapping model mismatch for {task}")
            shap_top = shap[
                (shap["task"].map(_canonical_task) == task)
                & (pd.to_numeric(shap["rank"], errors="coerce") <= EXPECTED_MAPPING_RANKS)
            ][["rank", "feature", "mean_abs_shap"]].sort_values("rank")
            map_top = group[["rank", "feature", "mean_abs_shap"]].sort_values("rank")
            if len(shap_top) != len(map_top):
                failures.append(f"SHAP mapping cannot be aligned to top SHAP rows for {task}")
            else:
                keys_equal = (
                    shap_top[["rank", "feature"]].reset_index(drop=True).astype(str)
                    == map_top[["rank", "feature"]].reset_index(drop=True).astype(str)
                ).all().all()
                values_equal = np.allclose(
                    pd.to_numeric(shap_top["mean_abs_shap"]),
                    pd.to_numeric(map_top["mean_abs_shap"]),
                    rtol=1e-12,
                    atol=1e-15,
                )
                if not bool(keys_equal and values_equal):
                    failures.append(f"SHAP mapping values do not reproduce SHAP top rows for {task}")

    margins = pd.to_numeric(
        y_summary.get("oriented_margin", pd.Series(dtype=float)), errors="coerce"
    )
    positive_count = int((margins > 0).sum())
    positive_tasks: list[str] = []
    nonpositive_tasks: list[str] = []
    if "task" in y_summary:
        for task, margin in zip(y_summary["task"], margins):
            (positive_tasks if margin > 0 else nonpositive_tasks).append(str(task))
    artifact_gate = "PASS" if not failures else "HOLD"
    negative_control_signal = (
        "PASS"
        if len(y_summary) == len(TDC_TASKS) and positive_count == len(TDC_TASKS)
        else "MIXED"
    )
    status = (
        "GO"
        if artifact_gate == "PASS" and negative_control_signal == "PASS"
        else ("REVIEW" if artifact_gate == "PASS" else "HOLD")
    )
    return {
        "schema_version": 1,
        "status": status,
        "artifact_gate": artifact_gate,
        "negative_control_signal": negative_control_signal,
        "analysis_lock": {
            "criterion": (
                "all 22 real-label oriented scores must strictly exceed the mean "
                "of ten training-label randomization trials"
            ),
            "expected_tasks": len(TDC_TASKS),
            "expected_trials_per_task": EXPECTED_Y_TRIALS,
            "expected_shap_tasks": sorted(expected_shap_tasks),
            "expected_shap_rows_per_task": EXPECTED_SHAP_RANKS,
            "expected_mapped_rows_per_task": EXPECTED_MAPPING_RANKS,
        },
        "coverage": {
            "y_summary_rows": int(len(y_summary)),
            "y_score_rows": int(len(y_scores)),
            "shap_rows": int(len(shap)),
            "shap_mapping_rows": int(len(mapping)),
            "mapped_substructures": mapped_count,
        },
        "y_randomization": {
            "positive_margin_count": positive_count,
            "positive_margin_fraction": (
                float(positive_count / len(y_summary)) if len(y_summary) else None
            ),
            "minimum_oriented_margin": (
                float(margins.min())
                if len(margins) and np.isfinite(margins).all()
                else None
            ),
            "median_oriented_margin": (
                float(margins.median())
                if len(margins) and np.isfinite(margins).all()
                else None
            ),
            "positive_tasks": positive_tasks,
            "nonpositive_tasks": nonpositive_tasks,
        },
        "failures": failures,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(snapshots[name][0]).hexdigest(),
                "bytes": len(snapshots[name][0]),
            }
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed gate for locked-cache CPU defense artifacts."
    )
    parser.add_argument(
        "--y-summary",
        type=Path,
        default=ROOT / "results/tables/y_randomization_all_tasks_locked_cache.csv",
    )
    parser.add_argument(
        "--y-scores",
        type=Path,
        default=ROOT / "results/tables/y_randomization_all_tasks_scores_locked_cache.csv",
    )
    parser.add_argument(
        "--shap",
        type=Path,
        default=ROOT / "results/tables/shap_selected_locked_cache.csv",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "results/tables/shap_morgan_bit_substructures_locked_cache.csv",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "results/tables/traditional_grid_locked_cache_best_by_task.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/tables/locked_cache_cpu_defense_gate.json",
    )
    args = parser.parse_args()

    result = summarize(
        args.y_summary, args.y_scores, args.shap, args.mapping, args.selection
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "GO" else 1)


if __name__ == "__main__":
    main()
