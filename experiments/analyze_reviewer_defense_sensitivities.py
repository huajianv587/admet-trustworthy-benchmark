"""Deterministic zero-refit reviewer-defense sensitivities.

This module adds three explicitly post-hoc checks without changing the locked
training, validation-selection, or primary-claim protocol:

1. aggregate cross-endpoint overlap of the fixed official test molecules;
2. full leave-one-model-out sensitivity of the four-model reliability pool;
3. conditional-reliability stratification by fingerprint and neural families.

No row-level molecular identity, structure, label, prediction, or sample index
is written.  Molecular sets exist only in memory long enough to calculate
aggregate counts and whole-set SHA-256 digests.
"""

from __future__ import annotations

# See the companion cross-fit analysis for the narrow Python 3.9 compatibility
# rationale.  This is a no-op on runtimes that already expose the attribute.
import typing

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import infer_smiles_column  # noqa: E402
from experiments.analyze_conditional_reliability import (  # noqa: E402
    bootstrap_interval,
    holm_adjust,
    sign_flip_pvalue,
)
from experiments.analyze_crossfit_oracle_sensitivity import (  # noqa: E402
    canonical_identity,
)
from experiments.analyze_reliability_regret import (  # noqa: E402
    EXPECTED_TASKS,
    FINAL_EXPECTED_MODELS,
    FINAL_EXPECTED_SEEDS,
    ModelEnsemble,
    build_regrets,
    select_validation_models,
    summarize_regrets,
)


DEFAULT_LOCKED_TEST_ROOT = ROOT / "data" / "tdc_official" / "admet_group"
DEFAULT_RELIABILITY_DIR = ROOT / "results" / "tables" / "reliability_regret_final"
DEFAULT_CONDITIONAL_DIR = ROOT / "results" / "tables" / "conditional_reliability_final"
DEFAULT_OUT_DIR = (
    ROOT / "results" / "tables" / "reviewer_defense_sensitivities_final"
)

DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_PERMUTATION_REPS = 20000
DEFAULT_SELECTION_BOOTSTRAP_REPS = 1000
DEFAULT_RANDOM_SEED = 20260715
DEFAULT_PRIMARY_THRESHOLD = 0.5
DEFAULT_SCORE_ABSOLUTE_TOLERANCE = 0.005
DEFAULT_SCORE_RELATIVE_TOLERANCE = 0.01

EXPECTED_ENDPOINT_UNIQUE_SUM = 16082
EXPECTED_ENDPOINT_UNION = 13818
EXPECTED_OVERLAPPING_PAIRS = 209
EXPECTED_PAIR_COUNT = 231

RELIABILITY_DIMENSIONS = (
    "applicability_alignment",
    "brier",
    "conformal_coverage_error",
    "ece",
    "selective_relative_excess_aurc",
    "selective_relative_excess_aurc_log_loss",
)

CONDITIONAL_METRICS = (
    "log_error_ratio_outside_inside",
    "uq_correlation_advantage_inside",
    "selective_gain_advantage_inside",
    "ece_gap_outside_inside",
    "brier_gap_outside_inside",
    "coverage_error_gap_outside_inside",
)

CONDITIONAL_GATE_METRICS = (
    "log_error_ratio_outside_inside",
    "ece_gap_outside_inside",
    "brier_gap_outside_inside",
    "coverage_error_gap_outside_inside",
    "uq_correlation_advantage_inside",
)

MODEL_FAMILIES = {
    "rf": "fingerprint",
    "xgboost": "fingerprint",
    "lightgbm": "fingerprint",
    "gin": "neural",
    "chemprop_dmpnn": "neural",
}

FORBIDDEN_PUBLIC_COLUMNS = frozenset(
    {
        "smiles",
        "canonical_smiles",
        "molecule_id",
        "sample_index",
        "y_true",
        "label",
        "prediction",
        "y_pred",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_set(values: Iterable[str]) -> str:
    """Hash one complete sorted set; never expose hashes of single molecules."""
    digest = hashlib.sha256()
    for value in sorted(set(str(item) for item in values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_random_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # Tests may bind a temporary input tree.  Keep the manifest deterministic
        # across temporary roots by recording only the stable terminal suffix.
        return path.name


def _json_default(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("Non-finite numeric value cannot enter a formal JSON artifact")
        return converted
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Mapping | Sequence) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _validate_no_forbidden_columns(frame: pd.DataFrame, artifact: str) -> None:
    forbidden = {str(column).lower() for column in frame.columns} & FORBIDDEN_PUBLIC_COLUMNS
    if forbidden:
        raise ValueError(
            f"Reviewer-defense artifact {artifact} contains forbidden row-level columns: "
            + ", ".join(sorted(forbidden))
        )


def _connected_components(
    endpoints: Sequence[str], pairwise: pd.DataFrame
) -> list[list[str]]:
    adjacency = {endpoint: set() for endpoint in endpoints}
    for row in pairwise.itertuples(index=False):
        if int(row.intersection_unique_count) <= 0:
            continue
        adjacency[str(row.endpoint_a)].add(str(row.endpoint_b))
        adjacency[str(row.endpoint_b)].add(str(row.endpoint_a))
    components: list[list[str]] = []
    remaining = set(endpoints)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        remaining -= component
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def cross_endpoint_overlap_audit(
    locked_test_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[Path]]:
    """Audit fixed-test molecular overlap with aggregate-only output."""
    paths = sorted(locked_test_root.glob("*/test.csv"))
    expected_tasks = set(EXPECTED_TASKS)
    observed_tasks = {path.parent.name.lower() for path in paths}
    if observed_tasks != expected_tasks:
        raise ValueError(
            "Locked test endpoint set mismatch: missing="
            f"{sorted(expected_tasks - observed_tasks)}, unexpected="
            f"{sorted(observed_tasks - expected_tasks)}"
        )

    identities: dict[str, set[str]] = {}
    endpoint_rows: list[dict] = []
    for path in paths:
        endpoint = path.parent.name.lower()
        frame = pd.read_csv(path)
        smiles_column = infer_smiles_column(frame)
        molecule_set = {
            canonical_identity(str(value)) for value in frame[smiles_column].tolist()
        }
        identities[endpoint] = molecule_set
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "source_row_count": int(len(frame)),
                "unique_molecule_count": int(len(molecule_set)),
                "whole_set_sha256": sha256_set(molecule_set),
                "source_file_sha256": sha256_file(path),
            }
        )

    pair_rows: list[dict] = []
    for endpoint_a, endpoint_b in itertools.combinations(sorted(identities), 2):
        left = identities[endpoint_a]
        right = identities[endpoint_b]
        intersection = len(left & right)
        union = len(left | right)
        pair_rows.append(
            {
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "endpoint_a_unique_count": int(len(left)),
                "endpoint_b_unique_count": int(len(right)),
                "intersection_unique_count": int(intersection),
                "union_unique_count": int(union),
                "jaccard": float(intersection / union) if union else 0.0,
                "has_overlap": bool(intersection > 0),
            }
        )

    endpoint_summary = pd.DataFrame(endpoint_rows).sort_values("endpoint").reset_index(drop=True)
    pairwise = pd.DataFrame(pair_rows)
    components = _connected_components(sorted(identities), pairwise)
    component_rows = [
        {
            "component_id": index + 1,
            "endpoint_count": len(component),
            "endpoints": ";".join(component),
        }
        for index, component in enumerate(components)
    ]
    component_frame = pd.DataFrame(component_rows)

    union_set = set().union(*identities.values())
    unique_sum = int(sum(len(values) for values in identities.values()))
    overlapping_pairs = int(pairwise["has_overlap"].sum())
    checks = {
        "endpoint_count_22": len(identities) == 22,
        "pair_count_231": len(pairwise) == EXPECTED_PAIR_COUNT,
        "endpoint_unique_sum_16082": unique_sum == EXPECTED_ENDPOINT_UNIQUE_SUM,
        "global_union_13818": len(union_set) == EXPECTED_ENDPOINT_UNION,
        "overlapping_pairs_209": overlapping_pairs == EXPECTED_OVERLAPPING_PAIRS,
        "single_connected_component": len(components) == 1,
        "no_row_level_molecular_output": True,
    }
    summary = {
        "status": "GO" if all(checks.values()) else "HOLD",
        "post_hoc": True,
        "primary_claim": False,
        "identity_policy": (
            "Audited parser compatibility repair followed by RDKit canonical, "
            "isomeric molecular identity; identical to the locked cross-fit audit."
        ),
        "privacy_policy": (
            "Only aggregate counts and complete-set SHA-256 digests are written; "
            "no row-level structures or molecule identifiers are emitted."
        ),
        "endpoint_count": len(identities),
        "pair_count": len(pairwise),
        "endpoint_unique_count_sum": unique_sum,
        "global_unique_union_count": int(len(union_set)),
        "cross_endpoint_duplicate_memberships": unique_sum - len(union_set),
        "pairs_with_overlap": overlapping_pairs,
        "pairs_without_overlap": int(len(pairwise) - overlapping_pairs),
        "connected_component_count": len(components),
        "largest_component_endpoint_count": max(map(len, components), default=0),
        "global_whole_set_sha256": sha256_set(union_set),
        "maximum_pair_intersection_count": int(
            pairwise["intersection_unique_count"].max()
        ),
        "checks": checks,
    }
    for artifact, frame in (
        ("cross_endpoint_test_endpoint_summary.csv", endpoint_summary),
        ("cross_endpoint_test_overlap_pairwise.csv", pairwise),
        ("cross_endpoint_test_overlap_components.csv", component_frame),
    ):
        _validate_no_forbidden_columns(frame, artifact)
    return endpoint_summary, pairwise, component_frame, summary, paths


def _validate_reliability_sources(
    validation_scores: pd.DataFrame, reliability: pd.DataFrame
) -> None:
    validation_required = {
        "task",
        "model",
        "seed",
        "official_metric",
        "validation_score_oriented",
    }
    reliability_required = {
        "task",
        "model",
        "task_type",
        "official_metric",
        "dimension",
        "value",
        "higher_is_better",
        "risk_metric",
    }
    if not validation_required.issubset(validation_scores.columns):
        raise ValueError("Canonical validation table lacks required columns")
    if not reliability_required.issubset(reliability.columns):
        raise ValueError("Canonical reliability table lacks required columns")

    expected_models = set(FINAL_EXPECTED_MODELS)
    expected_tasks = set(EXPECTED_TASKS)
    if set(validation_scores["model"].astype(str)) != expected_models:
        raise ValueError("Canonical validation table does not contain the locked five models")
    if set(validation_scores["task"].astype(str)) != expected_tasks:
        raise ValueError("Canonical validation table does not contain the locked 22 endpoints")
    if set(reliability["model"].astype(str)) != expected_models:
        raise ValueError("Canonical reliability table does not contain the locked five models")
    if set(reliability["task"].astype(str)) != expected_tasks:
        raise ValueError("Canonical reliability table does not contain the locked 22 endpoints")
    if set(reliability["dimension"].astype(str)) != set(RELIABILITY_DIMENSIONS):
        raise ValueError("Canonical reliability table does not contain all six frozen dimensions")

    validation_key = validation_scores[["task", "model", "seed"]]
    if validation_key.duplicated().any():
        raise ValueError("Duplicate task/model/seed row in canonical validation table")
    expected_seeds = set(FINAL_EXPECTED_SEEDS)
    for (_task, _model), group in validation_scores.groupby(["task", "model"], sort=True):
        if set(pd.to_numeric(group["seed"], errors="raise").astype(int)) != expected_seeds:
            raise ValueError("Every task/model must contain exactly the five locked seeds")
    if len(validation_scores) != 22 * 5 * 5:
        raise ValueError("Canonical validation row count must equal 550")
    if len(reliability) != 460:
        raise ValueError("Canonical full-test reliability row count must equal 460")
    if not np.isfinite(
        pd.to_numeric(validation_scores["validation_score_oriented"], errors="coerce")
    ).all():
        raise ValueError("Validation scores must be finite")


def _validation_ensembles(
    validation_scores: pd.DataFrame, reliability: pd.DataFrame
) -> list[ModelEnsemble]:
    metadata = (
        reliability[["task", "model", "task_type", "official_metric"]]
        .drop_duplicates()
        .set_index(["task", "model"])
    )
    ensembles: list[ModelEnsemble] = []
    for (task, model), group in validation_scores.groupby(["task", "model"], sort=True):
        row = metadata.loc[(task, model)]
        seed_scores = {
            int(seed): float(score)
            for seed, score in zip(group["seed"], group["validation_score_oriented"])
        }
        ensembles.append(
            ModelEnsemble(
                task=str(task),
                model=str(model),
                task_type=str(row["task_type"]),
                official_metric=str(row["official_metric"]),
                seeds=tuple(sorted(seed_scores)),
                validation_scores=seed_scores,
                test=pd.DataFrame(),
                conformal_coverage=float("nan"),
            )
        )
    return ensembles


def _augment_pool_quantities(
    regrets: pd.DataFrame, reliability: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    indexed = {
        (str(task), str(dimension)): group.copy()
        for (task, dimension), group in reliability.groupby(
            ["task", "dimension"], sort=True
        )
    }
    for regret in regrets.to_dict(orient="records"):
        group = indexed[(str(regret["task"]), str(regret["dimension"]))]
        higher = bool(group["higher_is_better"].iloc[0])
        values = group["value"].to_numpy(dtype=float)
        losses = -values if higher else values
        minimum = float(np.min(losses))
        maximum = float(np.max(losses))
        mean = float(np.mean(losses))
        loss_range = maximum - minimum
        exact_random = max(0.0, mean - minimum)
        normalized = (
            float(regret["reliability_regret"]) / loss_range
            if loss_range > 1e-12
            else 0.0
        )
        if not np.isclose(
            exact_random,
            float(regret["random_selection_expected_regret"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError("Exact uniform-random reference disagrees with original function")
        if not np.isclose(
            normalized,
            float(regret["normalized_regret"]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError("Four-model normalized regret was not recomputed from pool range")
        rows.append(
            {
                **regret,
                "pool_reliability_loss_min": minimum,
                "pool_reliability_loss_max": maximum,
                "pool_reliability_loss_range": loss_range,
                "uniform_random_mean_reliability_loss": mean,
                "exact_uniform_random_expected_regret": exact_random,
            }
        )
    result = pd.DataFrame(rows)
    _validate_no_forbidden_columns(result, "leave_one_model_out_endpoint.csv")
    return result


def leave_one_model_out_sensitivity(
    validation_scores: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    selection_bootstrap_reps: int = DEFAULT_SELECTION_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    absolute_tolerance: float = DEFAULT_SCORE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_SCORE_RELATIVE_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Re-run selection and every regret quantity in each four-model pool."""
    validation_scores = validation_scores.copy(deep=True)
    reliability = reliability.copy(deep=True)
    _validate_reliability_sources(validation_scores, reliability)
    all_ensembles = _validation_ensembles(validation_scores, reliability)

    selection_parts: list[pd.DataFrame] = []
    probability_parts: list[pd.DataFrame] = []
    regret_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    omitted_gates: dict[str, dict] = {}

    for omitted_model in sorted(FINAL_EXPECTED_MODELS):
        retained = [item for item in all_ensembles if item.model != omitted_model]
        selection_rng = np.random.default_rng(
            stable_random_seed(random_seed, "leave_one_model_out", omitted_model, "selection")
        )
        selections, probabilities = select_validation_models(
            retained,
            selection_bootstrap_reps,
            absolute_tolerance,
            relative_tolerance,
            selection_rng,
        )
        if len(selections) != 22 or set(selections["n_models"]) != {4}:
            raise AssertionError("Each leave-one-model-out selection must use 22 four-model pools")
        expected_pool = sorted(set(FINAL_EXPECTED_MODELS) - {omitted_model})
        if set(selections["model_pool"].astype(str)) != {";".join(expected_pool)}:
            raise AssertionError("Leave-one-model-out validation pool was not reconstructed")

        retained_reliability = reliability[
            reliability["model"].astype(str).ne(omitted_model)
        ].copy()
        regrets = build_regrets(retained_reliability, selections)
        regrets = _augment_pool_quantities(regrets, retained_reliability)
        summary_rng = np.random.default_rng(
            stable_random_seed(random_seed, "leave_one_model_out", omitted_model, "summary")
        )
        summary = summarize_regrets(
            regrets, bootstrap_reps, permutation_reps, summary_rng
        )
        if len(regrets) != 92 or set(regrets["n_models"]) != {4}:
            raise AssertionError("Each leave-one-model-out analysis must recompute 92 regrets")
        if set(summary["dimension"].astype(str)) != set(RELIABILITY_DIMENSIONS):
            raise AssertionError("Each leave-one-model-out analysis must retain six dimensions")

        selections.insert(0, "omitted_model", omitted_model)
        probabilities.insert(0, "omitted_model", omitted_model)
        regrets.insert(0, "omitted_model", omitted_model)
        summary.insert(0, "omitted_model", omitted_model)
        selection_parts.append(selections)
        probability_parts.append(probabilities)
        regret_parts.append(regrets)
        summary_parts.append(summary)

        effect_dimensions = sorted(
            summary.loc[summary["practical_effect_signal"], "dimension"].astype(str)
        )
        tie_dimensions = sorted(
            summary.loc[summary["tie_robust_effect_signal"], "dimension"].astype(str)
        )
        random_dimensions = sorted(
            summary.loc[
                summary["random_baseline_retention_signal"], "dimension"
            ].astype(str)
        )
        passes = (
            len(effect_dimensions) >= 2
            and len(tie_dimensions) >= 1
            and len(random_dimensions) >= 1
        )
        omitted_gates[omitted_model] = {
            "status": "GO" if passes else "HOLD",
            "frozen_criteria_pass": passes,
            "dimensions_with_practical_effect_signal": effect_dimensions,
            "dimensions_with_tie_robust_signal": tie_dimensions,
            "dimensions_retaining_random_baseline_regret": random_dimensions,
        }

    all_selections = pd.concat(selection_parts, ignore_index=True)
    all_probabilities = pd.concat(probability_parts, ignore_index=True)
    all_regrets = pd.concat(regret_parts, ignore_index=True)
    all_summaries = pd.concat(summary_parts, ignore_index=True)
    for artifact, frame in (
        ("leave_one_model_out_validation_selection.csv", all_selections),
        ("leave_one_model_out_validation_probabilities.csv", all_probabilities),
        ("leave_one_model_out_endpoint.csv", all_regrets),
        ("leave_one_model_out_dimension_summary.csv", all_summaries),
    ):
        _validate_no_forbidden_columns(frame, artifact)

    checks = {
        "canonical_validation_rows_550": len(validation_scores) == 550,
        "canonical_reliability_rows_460": len(reliability) == 460,
        "five_omissions": all_selections["omitted_model"].nunique() == 5,
        "selection_rows_110": len(all_selections) == 110,
        "four_model_pool_every_endpoint": set(all_selections["n_models"]) == {4},
        "endpoint_dimension_rows_460": len(all_regrets) == 460,
        "dimension_summary_rows_30": len(all_summaries) == 30,
        "six_dimensions_every_omission": all(
            group["dimension"].nunique() == 6
            for _, group in all_summaries.groupby("omitted_model")
        ),
        "validation_reselected_in_each_pool": True,
        "oracle_and_pool_range_recomputed": True,
        "exact_uniform_random_reference_recomputed": bool(
            np.allclose(
                all_regrets["random_selection_expected_regret"],
                all_regrets["exact_uniform_random_expected_regret"],
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "normalized_regret_recomputed_from_four_model_range": bool(
            np.allclose(
                all_regrets["normalized_regret"],
                np.divide(
                    all_regrets["reliability_regret"],
                    all_regrets["pool_reliability_loss_range"],
                    out=np.zeros(len(all_regrets), dtype=float),
                    where=all_regrets["pool_reliability_loss_range"].to_numpy() > 1e-12,
                ),
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "formal_bootstrap_reps_5000": bootstrap_reps == DEFAULT_BOOTSTRAP_REPS,
        "formal_permutation_reps_20000": permutation_reps == DEFAULT_PERMUTATION_REPS,
        "formal_selection_bootstrap_reps_1000": (
            selection_bootstrap_reps == DEFAULT_SELECTION_BOOTSTRAP_REPS
        ),
        "no_row_level_molecular_output": True,
    }
    gate = {
        "status": "GO" if all(checks.values()) else "HOLD",
        "post_hoc": True,
        "primary_claim": False,
        "interpretation": (
            "This is a robustness analysis of the locked full-test reliability result. "
            "Every omission re-runs validation selection and recomputes the four-model "
            "test oracle, pool range, tie set, best-case tie regret, normalized regret, "
            "and exact uniform-random reference. It is not formed by deleting rows from "
            "the five-model endpoint-regret table."
        ),
        "frozen_effect_criteria": {
            "practical_effect_signal": (
                "endpoint-bootstrap lower 95% bound for mean normalized regret>=0.10 "
                "and raw mean regret>=the frozen dimension tolerance, with practical-"
                "regret fraction>=0.25"
            ),
            "tie_robust_signal": (
                "endpoint-bootstrap lower 95% bound for best-case validation-tie "
                "normalized regret>=0.05 and raw regret>=the dimension tolerance, "
                "with tie-robust practical-regret fraction>=0.20"
            ),
            "random_baseline_retention": (
                "endpoint-bootstrap lower 95% bound for selected/uniform-random regret "
                "retention>=0.50 and at least 40% of endpoints retain >=50%"
            ),
            "gate": (
                "at least two practical-effect dimensions, one tie-robust dimension, "
                "and one random-baseline-retention dimension"
            ),
        },
        "omission_effect_gates": omitted_gates,
        "checks": checks,
    }
    return all_selections, all_probabilities, all_regrets, all_summaries, gate


def _validate_conditional_source(rows: pd.DataFrame) -> None:
    required = {
        "task",
        "model",
        "threshold",
        "eligible",
        *CONDITIONAL_METRICS,
    }
    if not required.issubset(rows.columns):
        raise ValueError("Canonical conditional-reliability table lacks required columns")
    if len(rows) != 330:
        raise ValueError("Canonical conditional-reliability row count must equal 330")
    if set(rows["task"].astype(str)) != set(EXPECTED_TASKS):
        raise ValueError("Conditional source does not contain all 22 locked endpoints")
    if set(rows["model"].astype(str)) != set(FINAL_EXPECTED_MODELS):
        raise ValueError("Conditional source does not contain all five locked models")
    thresholds = sorted(rows["threshold"].astype(float).unique())
    if thresholds != [0.3, 0.5, 0.7]:
        raise ValueError("Conditional source must contain the frozen 0.3/0.5/0.7 thresholds")
    keys = rows[["task", "model", "threshold"]]
    if keys.duplicated().any():
        raise ValueError("Duplicate task/model/threshold conditional-reliability row")


def conditional_family_sensitivity(
    source_rows: pd.DataFrame,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Summarize frozen conditional rows within fingerprint and neural families."""
    source_rows = source_rows.copy(deep=True)
    _validate_conditional_source(source_rows)
    eligible_before = source_rows["eligible"].astype(bool).to_numpy(copy=True)
    source_rows.insert(
        source_rows.columns.get_loc("model") + 1,
        "model_family",
        source_rows["model"].map(MODEL_FAMILIES),
    )
    if source_rows["model_family"].isna().any():
        raise ValueError("Every locked model must have one frozen family assignment")
    if not np.array_equal(eligible_before, source_rows["eligible"].astype(bool)):
        raise AssertionError("Family stratification changed original eligibility flags")

    output: list[dict] = []
    eligible_rows = source_rows[source_rows["eligible"].astype(bool)]
    for (family, threshold), threshold_rows in eligible_rows.groupby(
        ["model_family", "threshold"], sort=True
    ):
        for metric in CONDITIONAL_METRICS:
            pair_frame = threshold_rows[["task", metric]].dropna(subset=[metric]).copy()
            pair_values = pair_frame[metric].to_numpy(dtype=float)
            pair_values = pair_values[np.isfinite(pair_values)]
            endpoint_values = (
                pair_frame.groupby("task", sort=True)[metric].mean().to_numpy(dtype=float)
            )
            endpoint_values = endpoint_values[np.isfinite(endpoint_values)]
            if len(endpoint_values) == 0:
                raise ValueError(
                    f"No eligible endpoint values for family={family}, "
                    f"threshold={threshold}, metric={metric}"
                )
            rng = np.random.default_rng(
                stable_random_seed(
                    random_seed, "conditional_family", family, threshold, metric
                )
            )
            low, high = bootstrap_interval(endpoint_values, bootstrap_reps, rng)
            pvalue = sign_flip_pvalue(endpoint_values, permutation_reps, rng)
            output.append(
                {
                    "model_family": family,
                    "threshold": float(threshold),
                    "metric": metric,
                    "n_pairs": len(pair_values),
                    "n_endpoints": len(endpoint_values),
                    "pair_positive_count": int(np.sum(pair_values > 0)),
                    "pair_positive_fraction": float(np.mean(pair_values > 0)),
                    "positive_count": int(np.sum(endpoint_values > 0)),
                    "positive_fraction": float(np.mean(endpoint_values > 0)),
                    "mean": float(np.mean(endpoint_values)),
                    "median": float(np.median(endpoint_values)),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "sign_flip_p": pvalue,
                }
            )
    summary = pd.DataFrame(output)
    summary["holm_p_within_family_threshold"] = np.nan
    for (_family, _threshold), indices in summary.groupby(
        ["model_family", "threshold"], sort=True
    ).groups.items():
        adjusted = holm_adjust(summary.loc[list(indices), "sign_flip_p"])
        summary.loc[list(indices), "holm_p_within_family_threshold"] = adjusted

    family_effect_gates: dict[str, dict] = {}
    for family in sorted(set(MODEL_FAMILIES.values())):
        primary = summary[
            summary["model_family"].eq(family)
            & np.isclose(summary["threshold"], primary_threshold)
        ].set_index("metric")
        criteria: dict[str, bool] = {}
        for metric in CONDITIONAL_GATE_METRICS:
            if metric not in primary.index:
                criteria[metric] = False
                continue
            row = primary.loc[metric]
            criteria[metric] = bool(
                float(row["positive_fraction"]) >= 0.6
                and float(row["mean_ci_low"]) > 0
                and float(row["holm_p_within_family_threshold"]) < 0.05
            )
        strong_dimensions = int(sum(criteria.values()))
        passes = bool(
            criteria["log_error_ratio_outside_inside"] and strong_dimensions >= 2
        )
        family_effect_gates[family] = {
            "status": "GO" if passes else "HOLD",
            "frozen_original_effect_gate_pass": passes,
            "strong_conditional_dimensions": strong_dimensions,
            "criteria": criteria,
        }

    checks = {
        "canonical_source_rows_330": len(source_rows) == 330,
        "frozen_eligibility_preserved": np.array_equal(
            eligible_before, source_rows["eligible"].astype(bool)
        ),
        "five_models_assigned_once": set(source_rows["model_family"]) == {
            "fingerprint",
            "neural",
        },
        "two_families": summary["model_family"].nunique() == 2,
        "three_thresholds_per_family": all(
            group["threshold"].nunique() == 3
            for _, group in summary.groupby("model_family")
        ),
        "six_metrics_per_family_threshold": all(
            group["metric"].nunique() == 6
            for _, group in summary.groupby(["model_family", "threshold"])
        ),
        "summary_rows_36": len(summary) == 36,
        "holm_within_family_threshold": summary[
            "holm_p_within_family_threshold"
        ].notna().all(),
        "formal_endpoint_bootstrap_reps_5000": bootstrap_reps == DEFAULT_BOOTSTRAP_REPS,
        "formal_permutation_reps_20000": permutation_reps == DEFAULT_PERMUTATION_REPS,
        "no_row_level_molecular_output": True,
    }
    gate = {
        "status": "GO" if all(checks.values()) else "HOLD",
        "post_hoc": True,
        "primary_claim": False,
        "primary_threshold": primary_threshold,
        "family_policy": {
            "fingerprint": ["rf", "xgboost", "lightgbm"],
            "neural": ["gin", "chemprop_dmpnn"],
        },
        "aggregation_policy": (
            "Original task-model-threshold eligibility is preserved. Values are first "
            "averaged across eligible models within endpoint and family; endpoints are "
            "then the descriptive bootstrap and sign-flip comparison units. Shared-molecule "
            "dependence is not removed."
        ),
        "multiplicity_policy": (
            "Holm adjustment is applied across all six frozen metrics separately within "
            "each model-family and threshold family."
        ),
        "effect_policy": (
            "The original conditional-reliability effect rule is shown separately for "
            "each family as a post-hoc sensitivity and cannot replace the pooled primary gate."
        ),
        "family_effect_gates": family_effect_gates,
        "checks": checks,
    }
    for artifact, frame in (
        ("conditional_reliability_by_family.csv", source_rows),
        ("conditional_reliability_family_summary.csv", summary),
    ):
        _validate_no_forbidden_columns(frame, artifact)
    return source_rows, summary, gate


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def overlap_latex(summary: Mapping) -> str:
    rows = (
        ("Endpoints", summary["endpoint_count"]),
        ("Endpoint-unique sum", summary["endpoint_unique_count_sum"]),
        ("Global union", summary["global_unique_union_count"]),
        ("Overlapping endpoint pairs", summary["pairs_with_overlap"]),
        ("All endpoint pairs", summary["pair_count"]),
        ("Connected components", summary["connected_component_count"]),
    )
    body = "\n".join(
        f"{_tex_escape(label)} & {int(value):,} \\\\" for label, value in rows
    )
    return (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        + "Aggregate fixed-test audit & Value "
        + r"\\"
        + "\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def leave_one_model_out_latex(summary: pd.DataFrame) -> str:
    selected = summary[
        summary["dimension"].isin(
            [
                "applicability_alignment",
                "ece",
                "selective_relative_excess_aurc",
                "selective_relative_excess_aurc_log_loss",
            ]
        )
    ].copy()
    dimension_labels = {
        "applicability_alignment": "Morgan support/error rank association",
        "ece": "ECE",
        "selective_relative_excess_aurc": "Primary relative excess AURC",
        "selective_relative_excess_aurc_log_loss": "Log-loss relative excess AURC",
    }
    lines: list[str] = []
    for row in selected.itertuples(index=False):
        lines.append(
            "{} & {} & {} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                _tex_escape(row.omitted_model),
                _tex_escape(dimension_labels[str(row.dimension)]),
                int(row.n_endpoints),
                float(row.mean_normalized_regret),
                float(row.mean_tie_best_case_normalized_regret),
                float(row.mean_regret_fraction_of_random),
            )
        )
    return (
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "Omitted model & Reliability dimension & $n$ & Mean norm. regret & "
        + "Tie-best norm. & Selected/random "
        + r"\\"
        + "\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def conditional_family_latex(summary: pd.DataFrame) -> str:
    primary = summary[np.isclose(summary["threshold"], DEFAULT_PRIMARY_THRESHOLD)]
    lines: list[str] = []
    for row in primary.itertuples(index=False):
        lines.append(
            "{} & {} & {} & {:.3f} & [{:.3f}, {:.3f}] & {:.3g} \\\\".format(
                _tex_escape(row.model_family),
                _tex_escape(row.metric),
                int(row.n_endpoints),
                float(row.mean),
                float(row.mean_ci_low),
                float(row.mean_ci_high),
                float(row.holm_p_within_family_threshold),
            )
        )
    return (
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        + "Family & Conditional metric & $n$ & Mean & 95\\% interval & Descriptive Holm $p$ "
        + r"\\"
        + "\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _input_inventory(paths: Iterable[Path]) -> list[dict]:
    return [
        {
            "path": relative_path(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(set(path.resolve() for path in paths), key=lambda item: str(item))
    ]


def run_analysis(
    locked_test_root: Path = DEFAULT_LOCKED_TEST_ROOT,
    reliability_dir: Path = DEFAULT_RELIABILITY_DIR,
    conditional_dir: Path = DEFAULT_CONDITIONAL_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
    selection_bootstrap_reps: int = DEFAULT_SELECTION_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    reliability_paths = [
        reliability_dir / "validation_seed_scores.csv",
        reliability_dir / "test_reliability_by_task_model.csv",
    ]
    conditional_path = conditional_dir / "conditional_reliability_by_task_model.csv"
    for path in [*reliability_paths, conditional_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Canonical source table is missing: {path}")

    test_paths = sorted(locked_test_root.glob("*/test.csv"))
    input_paths = [*test_paths, *reliability_paths, conditional_path]
    before_inventory = _input_inventory(input_paths)

    endpoint_summary, pairwise, components, overlap_summary, audited_test_paths = (
        cross_endpoint_overlap_audit(locked_test_root)
    )
    if [path.resolve() for path in test_paths] != [
        path.resolve() for path in audited_test_paths
    ]:
        raise AssertionError("Cross-endpoint audit did not consume the complete locked test set")

    validation_scores = pd.read_csv(reliability_paths[0])
    reliability = pd.read_csv(reliability_paths[1])
    (
        leave_selections,
        leave_probabilities,
        leave_endpoints,
        leave_summary,
        leave_gate,
    ) = leave_one_model_out_sensitivity(
        validation_scores,
        reliability,
        bootstrap_reps=bootstrap_reps,
        permutation_reps=permutation_reps,
        selection_bootstrap_reps=selection_bootstrap_reps,
        random_seed=random_seed,
    )

    conditional_source = pd.read_csv(conditional_path)
    conditional_rows, conditional_summary, conditional_gate = (
        conditional_family_sensitivity(
            conditional_source,
            bootstrap_reps=bootstrap_reps,
            permutation_reps=permutation_reps,
            random_seed=random_seed,
        )
    )

    after_inventory = _input_inventory(input_paths)
    inputs_unchanged = before_inventory == after_inventory
    if not inputs_unchanged:
        raise AssertionError("A reviewer-defense source input changed during analysis")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "cross_endpoint_test_endpoint_summary.csv": endpoint_summary,
        "cross_endpoint_test_overlap_pairwise.csv": pairwise,
        "cross_endpoint_test_overlap_components.csv": components,
        "leave_one_model_out_validation_selection.csv": leave_selections,
        "leave_one_model_out_validation_probabilities.csv": leave_probabilities,
        "leave_one_model_out_endpoint.csv": leave_endpoints,
        "leave_one_model_out_dimension_summary.csv": leave_summary,
        "conditional_reliability_by_family.csv": conditional_rows,
        "conditional_reliability_family_summary.csv": conditional_summary,
    }
    for name, frame in frames.items():
        _validate_no_forbidden_columns(frame, name)
        write_csv(out_dir / name, frame)

    write_json(out_dir / "cross_endpoint_test_overlap_summary.json", overlap_summary)
    write_json(out_dir / "leave_one_model_out_gate.json", leave_gate)
    write_json(out_dir / "conditional_reliability_family_gate.json", conditional_gate)

    tex_files = {
        "table_supp_cross_endpoint_test_overlap.tex": overlap_latex(overlap_summary),
        "table_supp_leave_one_model_out.tex": leave_one_model_out_latex(leave_summary),
        "table_supp_conditional_reliability_families.tex": conditional_family_latex(
            conditional_summary
        ),
    }
    for name, value in tex_files.items():
        (out_dir / name).write_text(value, encoding="utf-8")

    combined_checks = {
        "cross_endpoint_overlap_gate_go": overlap_summary["status"] == "GO",
        "leave_one_model_out_gate_go": leave_gate["status"] == "GO",
        "conditional_family_gate_go": conditional_gate["status"] == "GO",
        "source_inputs_sha256_and_size_unchanged": inputs_unchanged,
        "zero_refit": True,
        "post_hoc_only": True,
        "no_new_primary_claim": True,
        "no_row_level_molecular_output": True,
    }
    combined_gate = {
        "status": "GO" if all(combined_checks.values()) else "HOLD",
        "scope": "post_hoc_reviewer_defense_sensitivities",
        "post_hoc": True,
        "primary_claim": False,
        "interpretation": (
            "All three reviewer-defense sensitivities are complete and deterministic. "
            "They diagnose dependence on endpoint overlap, model-pool membership, and "
            "model family; they do not alter the locked primary analysis or authorize "
            "a broader causal/generalization claim."
        ),
        "parameters": {
            "bootstrap_reps": bootstrap_reps,
            "permutation_reps": permutation_reps,
            "selection_bootstrap_reps": selection_bootstrap_reps,
            "random_seed": random_seed,
        },
        "checks": combined_checks,
    }
    write_json(out_dir / "reviewer_defense_sensitivities_gate.json", combined_gate)

    output_paths = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != "reviewer_defense_sensitivities_manifest.json"
    )
    manifest = {
        "schema": "reviewer_defense_sensitivities_v1",
        "post_hoc": True,
        "zero_refit": True,
        "primary_claim": False,
        "script": relative_path(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "parameters": combined_gate["parameters"],
        "input_files": before_inventory,
        "output_files": [
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    write_json(out_dir / "reviewer_defense_sensitivities_manifest.json", manifest)
    return combined_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-test-root", type=Path, default=DEFAULT_LOCKED_TEST_ROOT)
    parser.add_argument("--reliability-dir", type=Path, default=DEFAULT_RELIABILITY_DIR)
    parser.add_argument("--conditional-dir", type=Path, default=DEFAULT_CONDITIONAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--permutation-reps", type=int, default=DEFAULT_PERMUTATION_REPS)
    parser.add_argument(
        "--selection-bootstrap-reps",
        type=int,
        default=DEFAULT_SELECTION_BOOTSTRAP_REPS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    if min(
        args.bootstrap_reps,
        args.permutation_reps,
        args.selection_bootstrap_reps,
    ) <= 0:
        raise ValueError("All resampling counts must be positive")

    gate = run_analysis(
        locked_test_root=args.locked_test_root,
        reliability_dir=args.reliability_dir,
        conditional_dir=args.conditional_dir,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        permutation_reps=args.permutation_reps,
        selection_bootstrap_reps=args.selection_bootstrap_reps,
        random_seed=args.random_seed,
    )
    print(f"status: {gate['status']}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
