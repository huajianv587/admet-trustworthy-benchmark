"""Post-hoc, zero-refit influence sensitivity for cross-fit oracle estimates.

This reviewer-defense analysis starts from the endpoint means already written by
``analyze_crossfit_oracle_sensitivity.py``.  It performs no model fitting and
does not revisit model selection, folds, predictions, or endpoint definitions.
For every grouping scheme and dimension it reports deterministic leave-one-
endpoint-out (LOEO) and leave-one-ADMET-category-out (LOCO) endpoint-bootstrap
intervals.

The full-sample intervals deliberately reproduce the original cross-fit
dimension-summary bootstrap: 5,000 percentile replicates, base seed 20260715,
and the same SHA256-derived random state for each scheme/dimension.  Reduced
subsets use the predeclared influence seed 20260714 and disjoint deterministic
SHA256-derived streams.  These outputs are post-hoc influence diagnostics, not
new primary results or independent validation.
"""

from __future__ import annotations

# The portable Python 3.9 environment contains a dataclasses backport whose
# pandas import path expects typing._ClassVar.  Match the compatibility shim
# used by the locked cross-fit analysis before importing pandas.
import typing

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import TDC_TASK_META  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "results"
    / "tables"
    / "crossfit_oracle_sensitivity_final"
    / "crossfit_oracle_endpoint_summary.csv"
)
DEFAULT_REFERENCE_SUMMARY = DEFAULT_INPUT.parent / "crossfit_oracle_dimension_summary.csv"
DEFAULT_OUT_DIR = (
    ROOT / "results" / "tables" / "crossfit_influence_sensitivity_final"
)
DEFAULT_BOOTSTRAP_REPS = 5000
# The source analysis used 20260715 for its full-sample dimension intervals.
FULL_REFERENCE_RANDOM_SEED = 20260715
# The post-hoc reduced-subset sensitivity was predeclared with this seed.
DEFAULT_RANDOM_SEED = 20260714
PRIMARY_GROUPING_SCHEME = "bemis_murcko"
FOCUS_DIMENSIONS = (
    "ece_10bin",
    "ece_5bin",
    "applicability_alignment",
)
DIMENSION_ORDER = (
    "brier",
    "ece_5bin",
    "ece_10bin",
    "applicability_alignment",
    "selective_relative_excess_aurc",
    "selective_relative_excess_aurc_log_loss",
    "conformal_coverage_error",
)
CATEGORY_ORDER = (
    "absorption",
    "distribution",
    "metabolism",
    "excretion",
    "toxicity",
)
REQUIRED_COLUMNS = {
    "grouping_scheme",
    "task",
    "dimension",
    "mean_signed_d",
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_random_state(base_seed: int, *parts: object) -> int:
    """Reproduce the locked cross-fit SHA256-to-uint32 seed derivation."""
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little"
    )


def _logical_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _dimension_key(value: str) -> tuple[int, str]:
    try:
        return DIMENSION_ORDER.index(value), value
    except ValueError:
        return len(DIMENSION_ORDER), value


def _category_key(value: str) -> tuple[int, str]:
    try:
        return CATEGORY_ORDER.index(value), value
    except ValueError:
        return len(CATEGORY_ORDER), value


def locked_task_category_map(
    task_meta: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Return the case-normalized locked TDC endpoint-to-category mapping."""
    metadata = TDC_TASK_META if task_meta is None else task_meta
    mapping: dict[str, str] = {}
    for task, record in metadata.items():
        slug = str(task).strip().lower()
        category = str(record.get("category", "")).strip().lower()
        if not slug or not category:
            raise ValueError("Every task metadata record must define a category")
        if slug in mapping and mapping[slug] != category:
            raise ValueError(f"Conflicting category metadata for {slug}")
        mapping[slug] = category
    return mapping


def normalize_endpoint_summary(
    endpoint_rows: pd.DataFrame,
    task_meta: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Validate and annotate a deep copy of the endpoint-mean input."""
    missing = sorted(REQUIRED_COLUMNS - set(endpoint_rows.columns))
    if missing:
        raise ValueError(f"Endpoint summary lacks required columns: {missing}")
    frame = endpoint_rows.copy(deep=True)
    for column in ("grouping_scheme", "task", "dimension"):
        frame[column] = frame[column].astype(str).str.strip().str.lower()
        if (frame[column] == "").any():
            raise ValueError(f"Endpoint summary has an empty {column}")
    frame["mean_signed_d"] = pd.to_numeric(frame["mean_signed_d"], errors="raise")
    if not np.isfinite(frame["mean_signed_d"].to_numpy(dtype=float)).all():
        raise ValueError("Endpoint means must all be finite")
    key = ["grouping_scheme", "dimension", "task"]
    if frame.duplicated(key).any():
        duplicates = frame.loc[frame.duplicated(key, keep=False), key]
        raise ValueError(
            "Endpoint summary has duplicate scheme/dimension/task rows: "
            f"{duplicates.to_dict(orient='records')[:5]}"
        )

    category_map = locked_task_category_map(task_meta)
    frame["category"] = frame["task"].map(category_map)
    unknown = sorted(frame.loc[frame["category"].isna(), "task"].unique())
    if unknown:
        raise ValueError(f"Tasks are absent from locked TDC category metadata: {unknown}")

    if "scheme_role" not in frame.columns:
        frame["scheme_role"] = np.where(
            frame["grouping_scheme"] == PRIMARY_GROUPING_SCHEME,
            "primary",
            "secondary",
        )
    else:
        frame["scheme_role"] = (
            frame["scheme_role"].astype(str).str.strip().str.lower()
        )
    if "evidence_tier" not in frame.columns:
        frame["evidence_tier"] = "unspecified"
    else:
        frame["evidence_tier"] = (
            frame["evidence_tier"].astype(str).str.strip().str.lower()
        )

    group_sizes = frame.groupby(
        ["grouping_scheme", "dimension"], sort=False
    ).size()
    if (group_sizes < 2).any():
        bad = group_sizes[group_sizes < 2].to_dict()
        raise ValueError(f"LOEO requires at least two endpoints per group: {bad}")
    category_counts = frame.groupby(
        ["grouping_scheme", "dimension"], sort=False
    )["category"].nunique()
    if (category_counts < 2).any():
        bad = category_counts[category_counts < 2].to_dict()
        raise ValueError(f"LOCO requires at least two categories per group: {bad}")

    dimension_rank = frame["dimension"].map(
        lambda value: _dimension_key(str(value))[0]
    )
    frame = (
        frame.assign(_dimension_rank=dimension_rank)
        .sort_values(
            ["grouping_scheme", "_dimension_rank", "dimension", "task"],
            kind="mergesort",
        )
        .drop(columns="_dimension_rank")
        .reset_index(drop=True)
    )
    return frame


def bootstrap_mean_interval(
    values: Iterable[float],
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile 95% interval for an endpoint-mean bootstrap."""
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Cannot bootstrap an empty finite endpoint set")
    indices = rng.integers(0, len(array), size=(reps, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _mean_sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _single_text(group: pd.DataFrame, column: str) -> str:
    values = sorted(set(group[column].astype(str)))
    if len(values) != 1:
        raise ValueError(
            f"Inconsistent {column} within a scheme/dimension group: {values}"
        )
    return values[0]


def _full_interval(
    values: np.ndarray,
    scheme: str,
    dimension: str,
    bootstrap_reps: int,
    full_random_seed: int,
) -> tuple[float, float, int]:
    # Exact seed contract from summarize_dimensions in the source analysis.
    derived_seed = stable_random_state(
        full_random_seed, "endpoint_bootstrap", scheme, dimension
    )
    low, high = bootstrap_mean_interval(
        values, bootstrap_reps, np.random.default_rng(derived_seed)
    )
    return low, high, derived_seed


def _reduced_record(
    *,
    group: pd.DataFrame,
    reduced: pd.DataFrame,
    scheme: str,
    dimension: str,
    omission_kind: str,
    omitted_value: str,
    bootstrap_reps: int,
    random_seed: int,
    full_mean: float,
    full_ci_low: float,
    full_ci_high: float,
    full_seed: int,
) -> dict[str, object]:
    values = reduced["mean_signed_d"].to_numpy(dtype=float)
    derived_seed = stable_random_state(
        random_seed,
        "endpoint_bootstrap",
        scheme,
        dimension,
        omission_kind,
        omitted_value,
    )
    ci_low, ci_high = bootstrap_mean_interval(
        values, bootstrap_reps, np.random.default_rng(derived_seed)
    )
    reduced_mean = float(np.mean(values))
    full_sign = _mean_sign(full_mean)
    reduced_sign = _mean_sign(reduced_mean)
    sign_stable = reduced_sign == full_sign
    if full_sign == "positive":
        ci_supports_full_sign = ci_low > 0
    elif full_sign == "negative":
        ci_supports_full_sign = ci_high < 0
    else:
        ci_supports_full_sign = ci_low <= 0 <= ci_high
    return {
        "grouping_scheme": scheme,
        "scheme_role": _single_text(group, "scheme_role"),
        "dimension": dimension,
        "evidence_tier": _single_text(group, "evidence_tier"),
        "full_endpoint_count": int(len(group)),
        "remaining_endpoint_count": int(len(reduced)),
        "full_mean_signed_d": full_mean,
        "full_ci_low": full_ci_low,
        "full_ci_high": full_ci_high,
        "full_mean_sign": full_sign,
        "reduced_mean_signed_d": reduced_mean,
        "reduced_ci_low": ci_low,
        "reduced_ci_high": ci_high,
        "reduced_mean_sign": reduced_sign,
        "mean_sign_stable": bool(sign_stable),
        "sign_stability": "preserved" if sign_stable else "changed",
        "ci_low_gt_zero": bool(ci_low > 0),
        "ci_high_lt_zero": bool(ci_high < 0),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "ci_supports_full_mean_sign": bool(ci_supports_full_sign),
        "bootstrap_unit": "endpoint",
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(derived_seed),
        "full_bootstrap_seed": int(full_seed),
        "analysis_scope": "post_hoc_zero_refit_reviewer_defense",
    }


def _summarize_reductions(
    prefix: str,
    rows: pd.DataFrame,
) -> dict[str, object]:
    n = int(len(rows))
    stable_count = int(rows["mean_sign_stable"].sum())
    positive_ci_count = int(rows["ci_low_gt_zero"].sum())
    return {
        f"{prefix}_omission_count": n,
        f"{prefix}_reduced_mean_min": float(rows["reduced_mean_signed_d"].min()),
        f"{prefix}_reduced_mean_max": float(rows["reduced_mean_signed_d"].max()),
        f"{prefix}_ci_low_min": float(rows["reduced_ci_low"].min()),
        f"{prefix}_ci_low_max": float(rows["reduced_ci_low"].max()),
        f"{prefix}_ci_high_min": float(rows["reduced_ci_high"].min()),
        f"{prefix}_ci_high_max": float(rows["reduced_ci_high"].max()),
        f"{prefix}_mean_sign_stable_count": stable_count,
        f"{prefix}_mean_sign_stable_fraction": stable_count / n,
        f"{prefix}_ci_low_gt_zero_count": positive_ci_count,
        f"{prefix}_ci_low_gt_zero_fraction": positive_ci_count / n,
    }


def build_influence_tables(
    endpoint_rows: pd.DataFrame,
    *,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    full_random_seed: int = FULL_REFERENCE_RANDOM_SEED,
    task_meta: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic LOEO, LOCO, and dimension-level summaries."""
    normalized = normalize_endpoint_summary(endpoint_rows, task_meta=task_meta)
    loeo_records: list[dict[str, object]] = []
    loco_records: list[dict[str, object]] = []
    grouped = normalized.groupby(["grouping_scheme", "dimension"], sort=False)
    group_items = sorted(
        grouped,
        key=lambda item: (item[0][0], _dimension_key(item[0][1])),
    )
    for (scheme, dimension), source_group in group_items:
        group = source_group.sort_values("task", kind="mergesort").reset_index(
            drop=True
        )
        full_values = group["mean_signed_d"].to_numpy(dtype=float)
        full_mean = float(np.mean(full_values))
        full_ci_low, full_ci_high, full_seed = _full_interval(
            full_values,
            scheme,
            dimension,
            bootstrap_reps,
            full_random_seed,
        )
        for omitted_endpoint in sorted(group["task"].unique()):
            omitted = group.loc[group["task"] == omitted_endpoint].iloc[0]
            reduced = group.loc[group["task"] != omitted_endpoint]
            record = _reduced_record(
                group=group,
                reduced=reduced,
                scheme=scheme,
                dimension=dimension,
                omission_kind="leave_one_endpoint_out",
                omitted_value=omitted_endpoint,
                bootstrap_reps=bootstrap_reps,
                random_seed=random_seed,
                full_mean=full_mean,
                full_ci_low=full_ci_low,
                full_ci_high=full_ci_high,
                full_seed=full_seed,
            )
            record["omitted_endpoint"] = omitted_endpoint
            record["omitted_endpoint_category"] = str(omitted["category"])
            loeo_records.append(record)

        categories = sorted(
            group["category"].unique(), key=lambda value: _category_key(str(value))
        )
        for omitted_category in categories:
            omitted_count = int((group["category"] == omitted_category).sum())
            reduced = group.loc[group["category"] != omitted_category]
            record = _reduced_record(
                group=group,
                reduced=reduced,
                scheme=scheme,
                dimension=dimension,
                omission_kind="leave_one_category_out",
                omitted_value=str(omitted_category),
                bootstrap_reps=bootstrap_reps,
                random_seed=random_seed,
                full_mean=full_mean,
                full_ci_low=full_ci_low,
                full_ci_high=full_ci_high,
                full_seed=full_seed,
            )
            record["omitted_category"] = str(omitted_category)
            record["omitted_endpoint_count"] = omitted_count
            loco_records.append(record)

    loeo = pd.DataFrame(loeo_records)
    loco = pd.DataFrame(loco_records)
    summary_records: list[dict[str, object]] = []
    summary_keys = sorted(
        set(zip(loeo["grouping_scheme"], loeo["dimension"])),
        key=lambda value: (value[0], _dimension_key(value[1])),
    )
    for scheme, dimension in summary_keys:
        endpoint_subset = loeo.loc[
            (loeo["grouping_scheme"] == scheme)
            & (loeo["dimension"] == dimension)
        ]
        category_subset = loco.loc[
            (loco["grouping_scheme"] == scheme)
            & (loco["dimension"] == dimension)
        ]
        first = endpoint_subset.iloc[0]
        summary_records.append(
            {
                "grouping_scheme": scheme,
                "scheme_role": first["scheme_role"],
                "dimension": dimension,
                "evidence_tier": first["evidence_tier"],
                "full_endpoint_count": int(first["full_endpoint_count"]),
                "full_mean_signed_d": float(first["full_mean_signed_d"]),
                "full_ci_low": float(first["full_ci_low"]),
                "full_ci_high": float(first["full_ci_high"]),
                "full_ci_low_gt_zero": bool(first["full_ci_low"] > 0),
                **_summarize_reductions("loeo", endpoint_subset),
                **_summarize_reductions("loco", category_subset),
                "bootstrap_unit": "endpoint",
                "bootstrap_reps": int(bootstrap_reps),
                "reduced_random_seed": int(random_seed),
                "full_random_seed": int(full_random_seed),
                "analysis_scope": "post_hoc_zero_refit_reviewer_defense",
                "independent_validation": False,
            }
        )
    summary = pd.DataFrame(summary_records)
    return loeo, loco, summary


def compare_reference_dimension_summary(
    influence_summary: pd.DataFrame,
    reference_rows: pd.DataFrame,
) -> dict[str, object]:
    """Check exact reproduction of the 14 locked full-sample estimates/intervals."""
    required = {
        "grouping_scheme",
        "dimension",
        "mean_endpoint_signed_d",
        "mean_endpoint_signed_d_ci_low",
        "mean_endpoint_signed_d_ci_high",
    }
    missing = sorted(required - set(reference_rows.columns))
    if missing:
        raise ValueError(f"Reference dimension summary lacks columns: {missing}")
    reference = reference_rows[list(required)].copy(deep=True)
    reference["grouping_scheme"] = (
        reference["grouping_scheme"].astype(str).str.strip().str.lower()
    )
    reference["dimension"] = reference["dimension"].astype(str).str.strip().str.lower()
    if reference.duplicated(["grouping_scheme", "dimension"]).any():
        raise ValueError("Reference dimension summary has duplicate keys")
    observed = influence_summary[
        [
            "grouping_scheme",
            "dimension",
            "full_mean_signed_d",
            "full_ci_low",
            "full_ci_high",
        ]
    ]
    merged = observed.merge(
        reference,
        on=["grouping_scheme", "dimension"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    keys_match = bool((merged["_merge"] == "both").all())
    complete = merged.loc[merged["_merge"] == "both"].copy()
    error_columns = {
        "mean": ("full_mean_signed_d", "mean_endpoint_signed_d"),
        "ci_low": ("full_ci_low", "mean_endpoint_signed_d_ci_low"),
        "ci_high": ("full_ci_high", "mean_endpoint_signed_d_ci_high"),
    }
    errors: dict[str, float] = {}
    for label, (observed_column, reference_column) in error_columns.items():
        delta = np.abs(
            complete[observed_column].to_numpy(dtype=float)
            - complete[reference_column].to_numpy(dtype=float)
        )
        errors[label] = float(np.max(delta)) if len(delta) else math.inf
    max_error = max(errors.values())
    # The source CSV is formatted to 12 significant digits, so 1e-11 is much
    # tighter than its stored precision while allowing round-trip parsing.
    tolerance = 1e-11
    return {
        "reference_row_count": int(len(reference)),
        "observed_row_count": int(len(observed)),
        "keys_match": keys_match,
        "maximum_absolute_error": max_error,
        "maximum_absolute_error_by_field": errors,
        "absolute_error_tolerance": tolerance,
        "passes": bool(keys_match and max_error <= tolerance),
    }


def write_primary_latex_table(summary: pd.DataFrame, path: Path) -> None:
    """Write a compact, explicitly post-hoc table for three primary results."""
    focus = summary.loc[
        (summary["grouping_scheme"] == PRIMARY_GROUPING_SCHEME)
        & (summary["dimension"].isin(FOCUS_DIMENSIONS))
    ].copy()
    focus["_order"] = focus["dimension"].map(FOCUS_DIMENSIONS.index)
    focus = focus.sort_values("_order", kind="mergesort")
    if list(focus["dimension"]) != list(FOCUS_DIMENSIONS):
        raise ValueError("Primary Bemis--Murcko focus dimensions are incomplete")
    labels = {
        "ece_10bin": "ECE (10 bins)",
        "ece_5bin": "ECE (5 bins)",
        "applicability_alignment": "Applicability alignment",
    }
    lines = [
        "% Generated by experiments/analyze_crossfit_influence_sensitivity.py",
        "% Post-hoc zero-refit reviewer-defense diagnostic; not independent validation.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Post-hoc, zero-refit influence sensitivity for the emphasized primary Bemis--Murcko dimensions. Intervals are deterministic percentile endpoint bootstraps after each omission. This reviewer-defense diagnostic is not a new primary result or independent validation.}",
        "\\label{tab:crossfit-influence-posthoc}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dimension & Full mean signed $D$ [95\\% interval] & LOEO mean range; lower $>0$ & LOCO mean range; lower $>0$ \\\\",
        "\\midrule",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"{labels[str(row.dimension)]} & "
            f"{float(row.full_mean_signed_d):.4f} "
            f"[{float(row.full_ci_low):.4f}, {float(row.full_ci_high):.4f}] & "
            f"[{float(row.loeo_reduced_mean_min):.4f}, "
            f"{float(row.loeo_reduced_mean_max):.4f}]; "
            f"{int(row.loeo_ci_low_gt_zero_count)}/{int(row.loeo_omission_count)} & "
            f"[{float(row.loco_reduced_mean_min):.4f}, "
            f"{float(row.loco_reduced_mean_max):.4f}]; "
            f"{int(row.loco_ci_low_gt_zero_count)}/{int(row.loco_omission_count)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _finite_columns(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(
        np.isfinite(frame[column].to_numpy(dtype=float)).all() for column in columns
    )


def run_analysis(
    input_path: Path,
    out_dir: Path,
    *,
    reference_summary_path: Path | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """Run the file-bound analysis, write outputs, and return its gate."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    before_hash = sha256_file(input_path)
    before_bytes = input_path.stat().st_size
    source = pd.read_csv(input_path)
    normalized = normalize_endpoint_summary(source)
    loeo, loco, summary = build_influence_tables(
        source,
        bootstrap_reps=bootstrap_reps,
        random_seed=random_seed,
    )
    after_hash = sha256_file(input_path)
    after_bytes = input_path.stat().st_size

    if reference_summary_path is None:
        candidate = input_path.parent / "crossfit_oracle_dimension_summary.csv"
        reference_summary_path = candidate if candidate.is_file() else None
    reference_gate: dict[str, object]
    reference_hash: str | None = None
    if reference_summary_path is not None:
        reference_summary_path = Path(reference_summary_path)
        if not reference_summary_path.is_file():
            raise FileNotFoundError(reference_summary_path)
        reference_hash = sha256_file(reference_summary_path)
        reference_gate = compare_reference_dimension_summary(
            summary, pd.read_csv(reference_summary_path)
        )
    else:
        reference_gate = {
            "passes": False,
            "reason": "reference dimension summary not supplied",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "leave_one_endpoint_out.csv": loeo,
        "leave_one_category_out.csv": loco,
        "influence_dimension_summary.csv": summary,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in output_frames.items():
        path = out_dir / name
        frame.to_csv(path, index=False, float_format="%.12g")
        output_hashes[name] = sha256_file(path)
    tex_name = "table_supp_crossfit_influence_primary.tex"
    tex_path = out_dir / tex_name
    write_primary_latex_table(summary, tex_path)
    output_hashes[tex_name] = sha256_file(tex_path)

    expected_loco_rows = int(
        normalized.groupby(["grouping_scheme", "dimension"])["category"]
        .nunique()
        .sum()
    )
    expected_summary_rows = int(
        normalized[["grouping_scheme", "dimension"]].drop_duplicates().shape[0]
    )
    numeric_columns = [
        "full_mean_signed_d",
        "full_ci_low",
        "full_ci_high",
        "reduced_mean_signed_d",
        "reduced_ci_low",
        "reduced_ci_high",
    ]
    focus_rows = summary.loc[
        (summary["grouping_scheme"] == PRIMARY_GROUPING_SCHEME)
        & (summary["dimension"].isin(FOCUS_DIMENSIONS))
    ]
    checks = {
        "source_hash_and_size_unchanged": (
            before_hash == after_hash and before_bytes == after_bytes
        ),
        "all_input_tasks_map_to_locked_tdc_categories": bool(
            normalized["category"].notna().all()
        ),
        "loeo_has_one_row_per_input_endpoint_mean": len(loeo) == len(normalized),
        "loco_has_one_row_per_present_category": len(loco) == expected_loco_rows,
        "summary_has_every_grouping_dimension": len(summary)
        == expected_summary_rows,
        "reduced_subsets_are_nonempty": bool(
            (loeo["remaining_endpoint_count"] > 0).all()
            and (loco["remaining_endpoint_count"] > 0).all()
        ),
        "all_reported_estimates_and_intervals_are_finite": bool(
            _finite_columns(loeo, numeric_columns)
            and _finite_columns(loco, numeric_columns)
        ),
        "bootstrap_contract_is_endpoint_5000_full_seed_20260715_reduced_seed_20260714": bool(
            bootstrap_reps == DEFAULT_BOOTSTRAP_REPS
            and random_seed == DEFAULT_RANDOM_SEED
            and (loeo["bootstrap_unit"] == "endpoint").all()
            and (loco["bootstrap_unit"] == "endpoint").all()
        ),
        "locked_full_dimension_intervals_reproduced": bool(
            reference_gate.get("passes") is True
        ),
        "primary_bemis_murcko_focus_complete": len(focus_rows)
        == len(FOCUS_DIMENSIONS),
        "analysis_is_explicitly_post_hoc_and_not_independent_validation": bool(
            (summary["analysis_scope"] == "post_hoc_zero_refit_reviewer_defense").all()
            and (~summary["independent_validation"]).all()
        ),
    }
    status = "GO" if all(checks.values()) else "HOLD"
    gate: dict[str, object] = {
        "schema": "crossfit_influence_sensitivity_v1",
        "status": status,
        "scope": "post_hoc_zero_refit_crossfit_influence_reviewer_defense",
        "interpretation": (
            "This gate certifies deterministic recomputation, input immutability, "
            "category coverage, and omission-table completeness. LOEO/LOCO are "
            "post-hoc influence diagnostics based on existing endpoint means; "
            "they are neither new primary results nor independent validation."
        ),
        "analysis_contract": {
            "model_refits": 0,
            "input_unit": "cross-fit endpoint mean signed D",
            "bootstrap_unit": "endpoint",
            "bootstrap_method": "percentile_95pct",
            "bootstrap_reps": int(bootstrap_reps),
            "full_reference_random_seed": FULL_REFERENCE_RANDOM_SEED,
            "reduced_subset_random_seed": int(random_seed),
            "seed_derivation": "sha256_first_4_bytes_little_endian",
            "full_interval_seed_parts": (
                "20260715|endpoint_bootstrap|grouping_scheme|dimension"
            ),
            "reduced_interval_streams": "disjoint_by_omission_kind_and_value",
            "multiple_testing_adjustment": "not_applicable_descriptive_sensitivity",
            "confirmatory_inference": False,
            "independent_validation": False,
        },
        "checks": checks,
        "counts": {
            "input_endpoint_rows": int(len(normalized)),
            "grouping_schemes": int(normalized["grouping_scheme"].nunique()),
            "dimensions": int(normalized["dimension"].nunique()),
            "grouping_dimension_cells": expected_summary_rows,
            "loeo_rows": int(len(loeo)),
            "loco_rows": int(len(loco)),
            "summary_rows": int(len(summary)),
        },
        "reference_reproduction": reference_gate,
        "primary_bemis_murcko_focus": {
            str(row.dimension): {
                "full_mean_signed_d": float(row.full_mean_signed_d),
                "full_ci_low": float(row.full_ci_low),
                "full_ci_high": float(row.full_ci_high),
                "loeo_ci_low_gt_zero_count": int(
                    row.loeo_ci_low_gt_zero_count
                ),
                "loeo_omission_count": int(row.loeo_omission_count),
                "loco_ci_low_gt_zero_count": int(
                    row.loco_ci_low_gt_zero_count
                ),
                "loco_omission_count": int(row.loco_omission_count),
            }
            for row in focus_rows.itertuples(index=False)
        },
        "provenance": {
            "input_path": _logical_path(input_path),
            "input_sha256_before": before_hash,
            "input_sha256_after": after_hash,
            "input_bytes_before": int(before_bytes),
            "input_bytes_after": int(after_bytes),
            "category_mapping_source": "src/admet_benchmark/data_loader.py:TDC_TASK_META",
            "reference_dimension_summary_path": (
                _logical_path(reference_summary_path)
                if reference_summary_path is not None
                else None
            ),
            "reference_dimension_summary_sha256": reference_hash,
            "analysis_script_path": _logical_path(Path(__file__)),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "output_sha256": output_hashes,
            "gate_self_hash": "excluded_to_avoid_recursive_digest",
        },
    }
    gate_path = out_dir / "crossfit_influence_sensitivity_gate.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--reference-dimension-summary", default=str(DEFAULT_REFERENCE_SUMMARY)
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    gate = run_analysis(
        Path(args.input),
        Path(args.out_dir),
        reference_summary_path=Path(args.reference_dimension_summary),
        bootstrap_reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )
    print(f"status: {gate['status']}")
    print(f"LOEO rows: {gate['counts']['loeo_rows']}")
    print(f"LOCO rows: {gate['counts']['loco_rows']}")
    print(f"dimension rows: {gate['counts']['summary_rows']}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
