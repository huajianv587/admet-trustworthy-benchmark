#!/usr/bin/env python3
"""Audit secondary-loss regret on untouched outer folds.

The grouped nested refit audit already records every candidate prediction on
each untouched outer fold.  This script reuses those held-out metrics to ask a
narrower question: after inner-fold selection, how often is the selected
family worse than the best candidate on the same outer fold for secondary
losses?  It never reads the official test role and never changes the original
model-selection or power-gate protocols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/nested_grouped_refit_audit_20261027_v2/nested_outer_results.csv"
DEFAULT_OUTPUT = ROOT / "results/nested_grouped_refit_secondary_audit_20261028"
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20261028
EPSILON = 1e-12

METRICS = {
    "classification": {
        "brier": "outer_brier",
        "log_loss": "outer_log_loss",
    },
    "regression": {
        "mae": "outer_mae",
        "rmse": "outer_rmse",
    },
}
GROUP_KEYS = ("task", "role_seed", "outer_fold")
REQUIRED_COLUMNS = {
    *GROUP_KEYS,
    "task_type",
    "model",
    "selected_by_inner",
    "outer_brier",
    "outer_log_loss",
    "outer_mae",
    "outer_rmse",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {column} in {row.get('task')}")
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("outer result file is empty")
    return rows


def build_records(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)

    records: list[dict[str, object]] = []
    for group_key, candidates in sorted(groups.items()):
        task_type = candidates[0]["task_type"]
        if task_type not in METRICS:
            raise ValueError(f"unsupported task_type={task_type!r}")
        if any(row["task_type"] != task_type for row in candidates):
            raise ValueError(f"mixed task types in group {group_key}")
        selected = [row for row in candidates if row["selected_by_inner"].lower() == "true"]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one inner-selected candidate in {group_key}")

        for metric, column in METRICS[task_type].items():
            values = [(row["model"], finite_float(row, column)) for row in candidates]
            selected_value = finite_float(selected[0], column)
            oracle_value = min(value for _, value in values)
            candidate_range = max(value for _, value in values) - oracle_value
            raw_regret = max(0.0, selected_value - oracle_value)
            normalized_regret = (
                0.0 if candidate_range <= EPSILON else raw_regret / candidate_range
            )
            records.append(
                {
                    "task": group_key[0],
                    "role_seed": group_key[1],
                    "outer_fold": int(group_key[2]),
                    "task_type": task_type,
                    "metric": metric,
                    "selected_outer_loss": selected_value,
                    "candidate_oracle_outer_loss": oracle_value,
                    "candidate_range": candidate_range,
                    "raw_selection_regret": raw_regret,
                    "normalized_selection_regret": normalized_regret,
                    "selected_matches_candidate_oracle": raw_regret <= EPSILON,
                }
            )
    if not records:
        raise ValueError("no secondary-loss records were produced")
    return records


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot percentile an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def endpoint_block_interval(
    records: list[dict[str, object]], metric: str, seed: int
) -> tuple[float, float, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["metric"] == metric:
            by_task[str(record["task"])].append(float(record["normalized_selection_regret"]))
    task_means = [statistics.mean(values) for _, values in sorted(by_task.items())]
    if not task_means:
        raise ValueError(f"no task means for metric={metric}")
    observed = statistics.mean(task_means)
    rng = random.Random(seed)
    bootstrap_means = [
        statistics.mean(rng.choice(task_means) for _ in task_means)
        for _ in range(BOOTSTRAP_REPS)
    ]
    bootstrap_means.sort()
    return observed, percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for metric_index, metric in enumerate(sorted({str(r["metric"]) for r in records})):
        subset = [r for r in records if r["metric"] == metric]
        tasks = sorted({str(r["task"]) for r in subset})
        positive_folds = sum(float(r["raw_selection_regret"]) > EPSILON for r in subset)
        matched_folds = sum(bool(r["selected_matches_candidate_oracle"]) for r in subset)
        positive_tasks = sum(
            any(
                str(r["task"]) == task and float(r["raw_selection_regret"]) > EPSILON
                for r in subset
            )
            for task in tasks
        )
        mean_norm, ci_low, ci_high = endpoint_block_interval(
            records, metric, BOOTSTRAP_SEED + metric_index
        )
        summaries.append(
            {
                "metric": metric,
                "task_type": str(subset[0]["task_type"]),
                "outer_fold_records": len(subset),
                "endpoint_count": len(tasks),
                "positive_regret_folds": positive_folds,
                "oracle_match_folds": matched_folds,
                "positive_regret_endpoints": positive_tasks,
                "mean_raw_selection_regret": statistics.mean(
                    float(r["raw_selection_regret"]) for r in subset
                ),
                "mean_endpoint_normalized_regret": mean_norm,
                "endpoint_block_bootstrap_95_low": ci_low,
                "endpoint_block_bootstrap_95_high": ci_high,
                "bootstrap_replicates": BOOTSTRAP_REPS,
                "bootstrap_unit": "endpoint_mean_over_outer_folds",
            }
        )
    return summaries


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "task",
        "role_seed",
        "outer_fold",
        "task_type",
        "metric",
        "selected_outer_loss",
        "candidate_oracle_outer_loss",
        "candidate_range",
        "raw_selection_regret",
        "normalized_selection_regret",
        "selected_matches_candidate_oracle",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: (
                        f"{float(record[field]):.17g}"
                        if isinstance(record[field], float)
                        else record[field]
                    )
                    for field in fields
                }
            )


def write_report(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Nested secondary-loss regret audit",
        "",
        "This is an outcome-separated, grouped outer-fold audit derived from the",
        "nested refit predictions. Inner folds selected the family; every candidate",
        "was then refit on the complete outer-training partition and evaluated on",
        "the untouched outer fold. The official test role is not used here.",
        "",
        "| Metric | Outer folds | Endpoints | Positive-regret folds | Oracle matches | Positive-regret endpoints | Mean normalized regret | Endpoint-block 95% interval |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {metric} | {outer_fold_records} | {endpoint_count} | {positive_regret_folds} | "
            "{oracle_match_folds} | {positive_regret_endpoints} | {mean_endpoint_normalized_regret:.4f} | "
            "{endpoint_block_bootstrap_95_low:.4f}--{endpoint_block_bootstrap_95_high:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "The normalized regret divides the selected outer-fold loss gap by",
            "the candidate range on that same outer fold. The interval resamples",
            "endpoints, not folds or molecules, and is descriptive because endpoints",
            "share chemistry and source lineage. This audit does not establish",
            "independent external validation, a deployable selector, or a new",
            "confirmatory result; the original `0/150 POWER_FAIL_HOLD` is unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> dict[str, object]:
    rows = load_rows(input_path)
    records = build_records(rows)
    summaries = summarize(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nested_secondary_regret.csv"
    summary_path = output_dir / "nested_secondary_regret_summary.json"
    report_path = output_dir / "NESTED_SECONDARY_REGRET_REPORT.md"
    write_csv(csv_path, records)
    write_report(report_path, summaries)
    manifest = {
        "schema": "nested_secondary_regret_audit_v1",
        "status": "COMPLETE",
        "evidence_status": "OUTER_FOLD_SECONDARY_LOSS_AUDIT_RETROSPECTIVE",
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256(input_path),
        "input_rows": len(rows),
        "output_records": len(records),
        "summaries": summaries,
        "official_test_used_for_selection": False,
        "independent_external_validation": False,
        "original_power_gate_unchanged": "0/150 POWER_FAIL_HOLD",
        "protocol": {
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "group_keys": list(GROUP_KEYS),
            "secondary_metrics": METRICS,
            "epsilon": EPSILON,
        },
        "output_sha256": {
            "nested_secondary_regret.csv": sha256(csv_path),
            "NESTED_SECONDARY_REGRET_REPORT.md": sha256(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
