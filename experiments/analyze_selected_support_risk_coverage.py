"""Zero-refit Morgan-support risk--coverage audit for selected ADMET models.

The analysis deliberately consumes only frozen artifacts:

* the validation-only model choice already recorded by the final reliability
  analysis;
* the complete locked prediction matrix; and
* Morgan/Tanimoto support caches generated from each seed's locked training
  split.

No estimator is fitted here.  For each endpoint, predictions of the model fixed
by validation performance are averaged across the five seeds.  Test molecules
are then retained from highest to lowest mean training-set support.  Additive
test loss is Brier loss for classification and absolute error for regression.
The support AURC is compared with (i) the exact expected AURC under a uniformly
random ordering, which equals the endpoint mean loss, (ii) ensemble-uncertainty
ordering for the same selected model, and (iii) a descriptive observed-loss
oracle.  Ties in support or uncertainty use their expected within-tie risk and
therefore cannot be broken with test outcomes.

All cross-endpoint intervals and sign-flip p-values are descriptive because
TDC endpoints can share molecules.  The JSON gate certifies input completeness
and arithmetic only; it never turns a favorable effect into a submission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_reliability_regret import (
    EXPECTED_TASKS,
    FINAL_EXPECTED_MODELS,
    FINAL_EXPECTED_SEEDS,
    ModelEnsemble,
    audit_ensemble_test_alignment,
    audit_run_sample_alignment,
    aurc_from_curve,
    bootstrap_interval,
    build_ensembles,
    load_runs,
    risk_coverage_curve,
    sample_fingerprint,
    sign_flip_pvalue,
    spearman,
)
from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_SOURCE


DEFAULT_PREDICTION_DIRS = (
    ROOT / "results" / "predictions" / "traditional_locked_cache",
    ROOT / "results" / "neural" / "predictions",
    ROOT / "results" / "neural_chemprop" / "predictions",
)
DEFAULT_SELECTION = (
    ROOT
    / "results"
    / "tables"
    / "reliability_regret_final"
    / "validation_selection_by_task.csv"
)
DEFAULT_SIMILARITY = ROOT / "results" / "cache" / "applicability"
DEFAULT_OUT = (
    ROOT / "results" / "tables" / "selected_support_risk_coverage_final"
)

SUMMARY_METRICS = (
    (
        "support_gain_vs_random",
        "Morgan support vs random ordering",
        "positive means lower AURC under high-to-low Morgan support",
    ),
    (
        "uncertainty_gain_vs_random",
        "Ensemble uncertainty vs random ordering",
        "positive means lower AURC under low-to-high ensemble SD",
    ),
    (
        "support_minus_uncertainty_relative_aurc",
        "Morgan support minus ensemble uncertainty",
        "negative means Morgan support has lower AURC",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def combined_sha256(paths: Iterable[Path]) -> str:
    """Hash logical path plus file hash for a deterministic artifact set."""
    digest = hashlib.sha256()
    for path in sorted({Path(path).resolve() for path in paths}, key=logical_path):
        digest.update(logical_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _exact_model_pool(value: object) -> set[str]:
    return {part.strip().lower() for part in str(value).split(";") if part.strip()}


def validate_frozen_selection(
    selection: pd.DataFrame,
    ensembles: Iterable[ModelEnsemble],
) -> dict[str, ModelEnsemble]:
    """Validate and resolve one pre-existing validation choice per endpoint."""
    required = {
        "task",
        "n_models",
        "model_pool",
        "validation_selected_model",
        "validation_selected_score_oriented",
        "validation_score_margin",
        "validation_tie_models",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(f"Frozen selection table is missing columns: {missing}")
    selection = selection.copy()
    selection["task"] = selection["task"].astype(str).str.strip().str.lower()
    selection["validation_selected_model"] = (
        selection["validation_selected_model"].astype(str).str.strip().str.lower()
    )
    if selection["task"].duplicated().any():
        raise ValueError("Frozen selection table contains duplicate endpoints")
    if set(selection["task"]) != set(EXPECTED_TASKS):
        raise ValueError("Frozen selection table does not contain exactly 22 endpoints")
    expected_pool = set(FINAL_EXPECTED_MODELS)
    if not (
        pd.to_numeric(selection["n_models"], errors="raise").eq(len(expected_pool)).all()
        and selection["model_pool"].map(_exact_model_pool).eq(expected_pool).all()
    ):
        raise ValueError("Frozen selection table does not encode the final five-model pool")

    lookup = {(item.task, item.model): item for item in ensembles}
    expected_pairs = {
        (task, model) for task in EXPECTED_TASKS for model in FINAL_EXPECTED_MODELS
    }
    if set(lookup) != expected_pairs:
        missing_pairs = sorted(expected_pairs - set(lookup))
        extra_pairs = sorted(set(lookup) - expected_pairs)
        raise ValueError(
            "Prediction ensembles do not form the exact 22x5 matrix; "
            f"missing={missing_pairs[:3]}, extra={extra_pairs[:3]}"
        )
    for item in lookup.values():
        if tuple(item.seeds) != tuple(FINAL_EXPECTED_SEEDS):
            raise ValueError(
                f"Selected-support analysis requires seeds 1--5 for {(item.task, item.model)}"
            )

    resolved: dict[str, ModelEnsemble] = {}
    for row in selection.itertuples(index=False):
        key = (str(row.task), str(row.validation_selected_model))
        if key not in lookup:
            raise ValueError(f"Frozen validation choice has no complete ensemble: {key}")
        item = lookup[key]
        recomputed = float(np.mean(list(item.validation_scores.values())))
        recorded = float(row.validation_selected_score_oriented)
        if not np.isclose(recomputed, recorded, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Frozen validation score disagrees with locked predictions for "
                f"{key}: recorded={recorded}, recomputed={recomputed}"
            )
        resolved[item.task] = item
    return resolved


def primary_loss(item: ModelEnsemble) -> tuple[str, np.ndarray]:
    y_true = item.test["y_true"].to_numpy(dtype=float)
    prediction = item.test["prediction"].to_numpy(dtype=float)
    if item.task_type == "classification":
        return "brier", np.square(prediction - y_true)
    return "absolute_error", np.abs(prediction - y_true)


def ordered_curve(
    losses: np.ndarray,
    ranking_score: np.ndarray,
    sample_index: np.ndarray,
    *,
    ranking: str,
) -> pd.DataFrame:
    """Build a curve where a lower ranking score is retained first."""
    curve = risk_coverage_curve(losses, ranking_score, sample_index).copy()
    curve = curve.rename(columns={"uncertainty_threshold": "ranking_threshold"})
    curve.insert(0, "ranking", ranking)
    return curve


def _curve_value_at_coverage(curve: pd.DataFrame, coverage: float) -> float:
    if curve.empty:
        return float("nan")
    target = int(math.ceil(float(coverage) * len(curve)))
    row = curve.loc[curve["retained_count"] == target, "cumulative_risk"]
    if len(row) != 1:
        raise ValueError(f"Risk--coverage curve lacks retained count {target}")
    return float(row.iloc[0])


def evaluate_selected_support(
    selection: pd.DataFrame,
    selected: dict[str, ModelEnsemble],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_index = selection.assign(
        task=selection["task"].astype(str).str.lower()
    ).set_index("task")
    endpoint_rows: list[dict[str, object]] = []
    curve_parts: list[pd.DataFrame] = []
    for task in EXPECTED_TASKS:
        item = selected[task]
        loss_name, losses = primary_loss(item)
        sample_index = item.test["sample_index"].to_numpy(dtype=int)
        support = item.test["max_tanimoto"].to_numpy(dtype=float)
        uncertainty = item.test["uncertainty"].to_numpy(dtype=float)
        if not (
            len(losses)
            and np.isfinite(losses).all()
            and np.isfinite(support).all()
            and np.isfinite(uncertainty).all()
            and np.logical_and(support >= 0.0, support <= 1.0).all()
        ):
            raise ValueError(f"Non-finite loss/support/uncertainty for {task}")

        # risk_coverage_curve retains low scores first; 1-support therefore
        # implements high-to-low chemical support without outcome tie-breaking.
        support_curve = ordered_curve(
            losses, 1.0 - support, sample_index, ranking="morgan_support"
        )
        uncertainty_curve = ordered_curve(
            losses, uncertainty, sample_index, ranking="ensemble_uncertainty"
        )
        support_aurc, support_oracle, _ = aurc_from_curve(support_curve)
        uncertainty_aurc, uncertainty_oracle, _ = aurc_from_curve(uncertainty_curve)
        if not np.isclose(support_oracle, uncertainty_oracle, rtol=0.0, atol=1e-14):
            raise ValueError(f"Oracle AURC disagrees between rankings for {task}")
        mean_loss = float(np.mean(losses))
        if mean_loss <= 0.0 or not np.isfinite(mean_loss):
            raise ValueError(f"Endpoint mean loss cannot normalize AURC for {task}")

        row = selection_index.loc[task]
        endpoint_rows.append(
            {
                "task": task,
                "task_type": item.task_type,
                "official_metric": item.official_metric,
                "validation_selected_model": item.model,
                "validation_selected_score_oriented": float(
                    row["validation_selected_score_oriented"]
                ),
                "validation_score_margin": float(row["validation_score_margin"]),
                "validation_tie_models": str(row["validation_tie_models"]),
                "n_test": len(losses),
                "n_seeds": len(item.seeds),
                "risk_metric": loss_name,
                "mean_loss_random_aurc": mean_loss,
                "oracle_aurc": support_oracle,
                "support_aurc": support_aurc,
                "support_relative_aurc": support_aurc / mean_loss,
                "support_gain_vs_random": (mean_loss - support_aurc) / mean_loss,
                "support_normalized_excess_aurc":
                    (support_aurc - support_oracle) / mean_loss,
                "uncertainty_aurc": uncertainty_aurc,
                "uncertainty_relative_aurc": uncertainty_aurc / mean_loss,
                "uncertainty_gain_vs_random":
                    (mean_loss - uncertainty_aurc) / mean_loss,
                "uncertainty_normalized_excess_aurc":
                    (uncertainty_aurc - uncertainty_oracle) / mean_loss,
                "support_minus_uncertainty_relative_aurc":
                    (support_aurc - uncertainty_aurc) / mean_loss,
                "support_error_spearman": spearman(support, -losses),
                "support_unique_values": int(pd.Series(support).nunique()),
                "support_min": float(np.min(support)),
                "support_median": float(np.median(support)),
                "support_max": float(np.max(support)),
                "support_risk_at_25pct": _curve_value_at_coverage(
                    support_curve, 0.25
                ),
                "support_risk_at_50pct": _curve_value_at_coverage(
                    support_curve, 0.50
                ),
                "support_risk_at_75pct": _curve_value_at_coverage(
                    support_curve, 0.75
                ),
                "test_sample_fingerprint": sample_fingerprint(item.test),
            }
        )
        for curve, score_name in (
            (support_curve, "one_minus_mean_max_tanimoto"),
            (uncertainty_curve, "ensemble_prediction_sd"),
        ):
            part = curve.copy()
            part.insert(0, "task", task)
            part.insert(1, "model", item.model)
            part.insert(2, "task_type", item.task_type)
            part.insert(3, "risk_metric", loss_name)
            part.insert(5, "ranking_score", score_name)
            curve_parts.append(part)
    endpoints = pd.DataFrame(endpoint_rows).sort_values("task").reset_index(drop=True)
    curves = pd.concat(curve_parts, ignore_index=True)
    return endpoints, curves


def summarize_endpoints(
    endpoints: pd.DataFrame,
    *,
    bootstrap_reps: int,
    sign_flip_reps: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = (
        ("all", endpoints),
        ("classification", endpoints[endpoints["task_type"] == "classification"]),
        ("regression", endpoints[endpoints["task_type"] == "regression"]),
    )
    for scope_index, (scope, frame) in enumerate(scopes):
        for metric_index, (metric, label, interpretation) in enumerate(SUMMARY_METRICS):
            values = frame[metric].to_numpy(dtype=float)
            rng = np.random.default_rng(
                int(random_seed) + 1009 * scope_index + 97 * metric_index
            )
            low, high = bootstrap_interval(values, bootstrap_reps, rng)
            pvalue = sign_flip_pvalue(values, sign_flip_reps, rng)
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "comparison": label,
                    "n_endpoints": len(values),
                    "mean": float(np.mean(values)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "median": float(np.median(values)),
                    "positive_endpoints": int(np.sum(values > 0)),
                    "zero_endpoints": int(np.sum(values == 0)),
                    "negative_endpoints": int(np.sum(values < 0)),
                    "descriptive_sign_flip_p": pvalue,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def latex_summary_table(summary: pd.DataFrame) -> str:
    overall = summary[summary["scope"] == "all"].set_index("metric")
    rows: list[str] = []
    for metric, label, interpretation in SUMMARY_METRICS:
        row = overall.loc[metric]
        direction_count = (
            int(row["positive_endpoints"])
            if metric != "support_minus_uncertainty_relative_aurc"
            else int(row["negative_endpoints"])
        )
        rows.append(
            f"{label} & {float(row['mean']):.4f} "
            f"[{float(row['bootstrap_ci_low']):.4f}, "
            f"{float(row['bootstrap_ci_high']):.4f}] & "
            f"{direction_count}/{int(row['n_endpoints'])} \\\\" 
        )
    return "\n".join(
        [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Comparison & Mean [descriptive 95\% endpoint-bootstrap interval] & Favorable endpoints \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "% Positive is favorable in the first two rows; negative is favorable in the third.",
            "% Intervals and sign directions are descriptive because endpoints may share molecules.",
            "",
        ]
    )


def build_gate(
    *,
    selection_path: Path,
    prediction_dirs: list[Path],
    similarity_dir: Path,
    input_audit: pd.DataFrame,
    run_alignment: pd.DataFrame,
    ensemble_alignment: pd.DataFrame,
    endpoints: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    included = input_audit[input_audit["included"].eq(True)]
    prediction_paths = sorted(
        Path(value) for value in included["path"].dropna().astype(str).unique()
    )
    similarity_paths = sorted(
        Path(value) for value in included["similarity_path"].dropna().astype(str).unique()
    )
    overall = summary[summary["scope"] == "all"].set_index("metric")
    support = overall.loc["support_gain_vs_random"]
    technical_checks = {
        "exact_22_endpoints": bool(
            len(endpoints) == 22
            and set(endpoints["task"]) == set(EXPECTED_TASKS)
        ),
        "exact_one_frozen_choice_per_endpoint": bool(endpoints["task"].is_unique),
        "exact_five_seeds_per_choice": bool(endpoints["n_seeds"].eq(5).all()),
        "exact_550_locked_prediction_runs": bool(
            len(prediction_paths) == 22 * 5 * 5
            and included["reason"].eq("complete").all()
        ),
        "run_sample_alignment_exact": bool(
            run_alignment["task_seed_models_aligned"].all()
        ),
        "ensemble_sample_alignment_exact": bool(
            ensemble_alignment["matches_task_reference"].all()
        ),
        "complete_locked_similarity": bool(
            len(similarity_paths) == 22 * 5
            and included["similarity_complete"].eq(True).all()
            and included["similarity_split_source"].eq(LOCKED_CACHE_SPLIT_SOURCE).all()
        ),
        "all_endpoint_metrics_finite": bool(
            np.isfinite(
                endpoints.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            ).all()
        ),
    }
    passes = all(technical_checks.values())
    return {
        "schema_version": 1,
        "status": "PASS" if passes else "FAIL",
        "analysis": "selected_model_morgan_support_risk_coverage",
        "zero_refit": True,
        "test_outcomes_used_for_model_selection": False,
        "selection_source": logical_path(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "prediction_directories": [logical_path(path) for path in prediction_dirs],
        "prediction_file_count": len(prediction_paths),
        "similarity_directory": logical_path(similarity_dir),
        "similarity_cache_count": len(similarity_paths),
        "technical_checks": technical_checks,
        "coverage": {
            "endpoints": len(endpoints),
            "selected_models": {
                str(key): int(value)
                for key, value in endpoints["validation_selected_model"]
                .value_counts()
                .sort_index()
                .items()
            },
            "classification_endpoints": int(endpoints["task_type"].eq("classification").sum()),
            "regression_endpoints": int(endpoints["task_type"].eq("regression").sum()),
        },
        "primary_descriptive_result": {
            "metric": "support_gain_vs_random",
            "mean": float(support["mean"]),
            "endpoint_bootstrap_ci": [
                float(support["bootstrap_ci_low"]),
                float(support["bootstrap_ci_high"]),
            ],
            "positive_endpoints": int(support["positive_endpoints"]),
            "n_endpoints": int(support["n_endpoints"]),
            "descriptive_sign_flip_p": float(support["descriptive_sign_flip_p"]),
            "ci_excludes_zero_in_favorable_direction": bool(
                float(support["bootstrap_ci_low"]) > 0.0
            ),
        },
        "interpretation_contract": {
            "random_aurc": "exact expected AURC under uniform random ordering equals mean endpoint loss",
            "support_gain_vs_random": "(random AURC - support AURC) / random AURC; positive is favorable",
            "model_choice": "read from frozen validation-only selection and reconciled against validation predictions",
            "claim_limit": "descriptive benchmark evidence, not external or prospective validation",
            "dependence_limit": "endpoint bootstrap and sign-flip values are descriptive because endpoints may share molecules",
            "gate_limit": "PASS certifies artifact completeness and arithmetic, not a favorable scientific finding",
        },
        "input_sha256": {
            "selection": sha256_file(selection_path),
            "prediction_files_combined": combined_sha256(prediction_paths),
            "similarity_caches_combined": combined_sha256(similarity_paths),
            "analysis_script": sha256_file(Path(__file__)),
        },
    }


def remove_generated_appledouble_sidecars(paths: Iterable[Path]) -> None:
    """Remove only metadata companions for files generated by this analysis."""
    for path in paths:
        sidecar = path.parent / f"._{path.name}"
        if sidecar.is_file():
            sidecar.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_PREDICTION_DIRS],
    )
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--similarity-dir", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--sign-flip-reps", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100 or args.sign_flip_reps < 100:
        raise ValueError("At least 100 bootstrap and sign-flip repetitions are required")
    prediction_dirs = [Path(path) for path in args.prediction_dirs]
    selection_path = Path(args.selection)
    similarity_dir = Path(args.similarity_dir)
    out_dir = Path(args.out_dir)

    selection = pd.read_csv(selection_path)
    runs, input_audit = load_runs(
        prediction_dirs,
        similarity_dir,
        similarity_split_source=LOCKED_CACHE_SPLIT_SOURCE,
        require_similarity=True,
    )
    run_alignment, run_alignment_ok = audit_run_sample_alignment(runs)
    if not run_alignment_ok:
        raise ValueError("Locked prediction rows are not aligned within task and seed")
    ensembles, validation_scores, availability = build_ensembles(
        runs, minimum_seeds=5, alpha=0.1
    )
    ensemble_alignment, ensemble_alignment_ok = audit_ensemble_test_alignment(ensembles)
    if not ensemble_alignment_ok:
        raise ValueError("Locked ensemble test rows are not aligned across models")
    selected = validate_frozen_selection(selection, ensembles)
    endpoints, curves = evaluate_selected_support(selection, selected)
    summary = summarize_endpoints(
        endpoints,
        bootstrap_reps=args.bootstrap_reps,
        sign_flip_reps=args.sign_flip_reps,
        random_seed=args.random_seed,
    )
    gate = build_gate(
        selection_path=selection_path,
        prediction_dirs=prediction_dirs,
        similarity_dir=similarity_dir,
        input_audit=input_audit,
        run_alignment=run_alignment,
        ensemble_alignment=ensemble_alignment,
        endpoints=endpoints,
        summary=summary,
    )
    if gate["status"] != "PASS":
        raise ValueError(f"Selected-support technical gate failed: {gate['technical_checks']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "prediction_input_audit.csv": input_audit,
        "run_sample_alignment_audit.csv": run_alignment,
        "ensemble_sample_alignment_audit.csv": ensemble_alignment,
        "validation_seed_scores_recomputed.csv": validation_scores,
        "model_seed_availability.csv": availability,
        "selected_support_risk_coverage_by_endpoint.csv": endpoints,
        "selected_support_risk_coverage_curve.csv": curves,
        "selected_support_risk_coverage_summary.csv": summary,
    }
    generated_paths: list[Path] = []
    for name, frame in output_paths.items():
        path = out_dir / name
        frame.to_csv(path, index=False, float_format="%.12g")
        generated_paths.append(path)
    latex_path = out_dir / "table_selected_support_risk_coverage.tex"
    latex_path.write_text(
        latex_summary_table(summary), encoding="utf-8"
    )
    generated_paths.append(latex_path)
    gate_path = out_dir / "selected_support_risk_coverage_gate.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    generated_paths.append(gate_path)
    remove_generated_appledouble_sidecars(generated_paths)
    primary = gate["primary_descriptive_result"]
    print(f"selected endpoints: {len(endpoints)}")
    print(
        "mean support gain vs random: "
        f"{primary['mean']:.6f} "
        f"[{primary['endpoint_bootstrap_ci'][0]:.6f}, "
        f"{primary['endpoint_bootstrap_ci'][1]:.6f}]"
    )
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
