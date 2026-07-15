from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# This experiment is deliberately CPU-only. Traditional estimators receive the
# explicit thread count below; CUDA is hidden before model packages are imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASK_META,
    TDC_TASKS,
)
from admet_benchmark.sensitivity import CleaningPolicy
from admet_benchmark.smiles_compat import (
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)


FALLBACK_CONFIG = {
    "selected_tasks_for_cleaned_rerun": [
        "Caco2_Wang",
        "BBB_Martins",
        "PPBR_AZ",
        "Solubility_AqSolDB",
        "CYP2C9_Veith",
        "DILI",
        "AMES",
        "LD50_Zhu",
    ],
    "models": ["rf", "xgboost", "lightgbm"],
    "seeds": [1, 2, 3, 4, 5],
    "fingerprint": {"radius": 2, "n_bits": 2048},
}


METRICS = [
    "auroc",
    "auprc",
    "mcc",
    "balanced_accuracy",
    "brier",
    "positive_rate",
    "mae",
    "rmse",
    "r2",
    "spearman",
]
METRIC_FIELDS = [
    f"{training}_common_{partition}_{metric}"
    for training in ("official", "cleaned")
    for partition in ("valid", "test")
    for metric in METRICS
]
FIELDNAMES = [
    "task",
    "dataset_name",
    "task_type",
    "official_metric",
    "model",
    "seed",
    "split_source",
    "split_algorithm",
    "radius",
    "n_bits",
    "threads_per_model",
    "official_train_n",
    "cleaned_train_n",
    "common_valid_n",
    "common_test_n",
    "official_fit_seconds",
    "cleaned_fit_seconds",
    "primary_official_common_test",
    "primary_cleaned_common_test",
    "primary_raw_delta_cleaned_minus_official",
    "primary_oriented_delta",
    "primary_relative_oriented_delta",
] + METRIC_FIELDS

MANIFEST_SCHEMA_VERSION = 2
UPGRADE_RECEIPT_SCHEMA_VERSION = 1
KNOWN_MODELS = ("rf", "xgboost", "lightgbm")
PROVENANCE_FIELDS = ("split_source", "split_algorithm")
PREDICTION_CORE_FIELDS = [
    "task",
    "model",
    "seed",
    "evaluation_set",
    "canonical_smiles",
    "label",
    "official_prediction",
    "cleaned_prediction",
]
PREDICTION_FIELDNAMES = (
    PREDICTION_CORE_FIELDS[:3]
    + list(PROVENANCE_FIELDS)
    + PREDICTION_CORE_FIELDS[3:]
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_positive_int(value: object, *, field: str) -> int:
    """Parse a positive integer without accepting bools, floats, or coercions."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be a strict positive integer, not bool")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise ValueError(
            f"{field} must be a strict positive integer; received {value!r}"
        )
    if parsed <= 0:
        raise ValueError(f"{field} must be positive; received {parsed}")
    return parsed


def _positive_int_argument(value: str) -> int:
    try:
        return _strict_positive_int(value, field="command-line integer")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_unique_names(
    values: object,
    *,
    field: str,
    known: set[str],
    case_insensitive: bool,
) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    normalized: list[str] = []
    seen: set[str] = set()
    known_lookup = (
        {item.lower(): item for item in known}
        if case_insensitive
        else {item: item for item in known}
    )
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty, trimmed string")
        lookup = value.lower() if case_insensitive else value
        if lookup not in known_lookup:
            raise ValueError(
                f"Unknown {field} value {value!r}; known values are "
                + ", ".join(sorted(known))
            )
        if lookup in seen:
            raise ValueError(f"{field} contains duplicate value {value!r}")
        seen.add(lookup)
        normalized.append(known_lookup[lookup])
    return normalized


def _validate_seeds(values: object, *, field: str = "seeds") -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    parsed = [
        _strict_positive_int(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    ]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} contains duplicate values")
    return parsed


def _validate_runner_values(
    *,
    tasks: object,
    models: object,
    seeds: object,
    radius: object,
    n_bits: object,
    threads_per_model: object,
) -> tuple[list[str], list[str], list[int], int, int, int]:
    return (
        _validate_unique_names(
            tasks,
            field="tasks",
            known=set(TDC_TASKS),
            case_insensitive=False,
        ),
        _validate_unique_names(
            models,
            field="models",
            known=set(KNOWN_MODELS),
            case_insensitive=True,
        ),
        _validate_seeds(seeds),
        _strict_positive_int(radius, field="radius"),
        _strict_positive_int(n_bits, field="n_bits"),
        _strict_positive_int(threads_per_model, field="threads_per_model"),
    )


def _resolve_and_validate_configuration(
    config: object,
    *,
    tasks_override: list[str] | None,
    models_override: list[str] | None,
    seeds_override: list[int] | None,
    radius_override: int | None,
    n_bits_override: int | None,
    threads_override: int | None,
) -> tuple[list[str], list[str], list[int], int, int, int]:
    if not isinstance(config, dict):
        raise ValueError("Sensitivity config must be a mapping")
    fingerprint = config.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Sensitivity config fingerprint must be a mapping")

    config_threads = config.get("threads_per_model", 4)
    config_values = _validate_runner_values(
        tasks=config.get("selected_tasks_for_cleaned_rerun"),
        models=config.get("models"),
        seeds=config.get("seeds"),
        radius=fingerprint.get("radius"),
        n_bits=fingerprint.get("n_bits"),
        threads_per_model=config_threads,
    )
    config_tasks, config_models, config_seeds, config_radius, config_n_bits, config_threads = (
        config_values
    )
    return _validate_runner_values(
        tasks=config_tasks if tasks_override is None else tasks_override,
        models=config_models if models_override is None else models_override,
        seeds=config_seeds if seeds_override is None else seeds_override,
        radius=config_radius if radius_override is None else radius_override,
        n_bits=config_n_bits if n_bits_override is None else n_bits_override,
        threads_per_model=(
            config_threads if threads_override is None else threads_override
        ),
    )


def _config_logical_path(config_path: Path) -> str:
    resolved_root = ROOT.resolve()
    resolved_config = config_path.resolve()
    try:
        return resolved_config.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Sensitivity config must be inside the project root so "
            "config_logical_path is stable and project-relative: "
            f"{resolved_config}"
        ) from exc


def _manifest_lock_path(manifest_path: Path) -> Path:
    resolved = manifest_path.expanduser().resolve(strict=False)
    return resolved.with_name(resolved.name + ".lock")


@contextmanager
def _exclusive_manifest_lock(manifest_path: Path) -> Iterator[Path]:
    """Hold a non-blocking process lock for the whole mutating invocation."""

    lock_path = _manifest_lock_path(manifest_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another cleaned-sensitivity invocation holds the manifest lock: "
                f"{lock_path}. Refusing concurrent resume/upgrade."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_utc": _utc_now(),
                    "manifest_path": str(manifest_path.expanduser().resolve(strict=False)),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        default_path = ROOT / "experiments" / "config" / "sensitivity.yaml"
        if path.resolve() != default_path.resolve():
            raise ImportError("PyYAML is required for a custom sensitivity config.")
        return FALLBACK_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compatibility_source_counts(
    data_dir: Path, tasks: list[str]
) -> dict[str, int]:
    """Count exact legacy representations in the immutable cached sources."""

    from admet_benchmark.data_loader import infer_smiles_column

    legacy = set(LEGACY_VALENCE_REPAIRS)
    counts: dict[str, int] = {}
    for task in tasks:
        total = 0
        task_dir = data_dir / "admet_group" / task.lower()
        for filename in ("train_val.csv", "test.csv"):
            frame = pd.read_csv(task_dir / filename)
            smiles_column = infer_smiles_column(frame)
            total += int(frame[smiles_column].astype(str).isin(legacy).sum())
        counts[task] = total
    return counts


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _expected_split_algorithm(split_source: str) -> str:
    return (
        LOCKED_CACHE_SPLIT_ALGORITHM
        if split_source == LOCKED_CACHE_SPLIT_SOURCE
        else "pytdc_default_task_specific_runtime"
    )


def _load_upgrade_manifest(manifest_path: Path, *, split_source: str) -> dict:
    if not manifest_path.exists():
        raise ValueError(
            f"Cannot upgrade sensitivity provenance without manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Sensitivity manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Sensitivity manifest must contain a JSON object")
    expected_algorithm = _expected_split_algorithm(split_source)
    if int(manifest.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Sensitivity provenance upgrade requires a schema-v2 manifest")
    if manifest.get("split_source") != split_source:
        raise ValueError("Sensitivity manifest split_source does not bind the requested run")
    if manifest.get("split_algorithm") != expected_algorithm:
        raise ValueError("Sensitivity manifest split_algorithm does not bind the requested run")
    return manifest


@dataclass(frozen=True)
class _CsvDocument:
    fields: list[str]
    rows: list[dict[str, str]]
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _CsvUpgradePlan:
    path: Path
    kind: str
    target_fields: list[str]
    semantic_fields: list[str]
    rows: list[dict[str, str]]
    action: str
    old_state: dict[str, object]


def _read_csv_strict(path: Path, *, artifact: str) -> _CsvDocument:
    if not path.exists():
        raise ValueError(f"Missing {artifact}: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{artifact} is blank or truncated: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} is not valid UTF-8: {path}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = list(reader.fieldnames or [])
    if not fields or any(not field for field in fields):
        raise ValueError(f"{artifact} has a blank or missing CSV schema: {path}")
    if len(fields) != len(set(fields)):
        raise ValueError(f"{artifact} has duplicate CSV columns: {path}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{artifact} has a header but no data rows: {path}")
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                f"{artifact} has a truncated or over-wide row {row_number}: {path}"
            )
    return _CsvDocument(
        fields=fields,
        rows=rows,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def _strict_artifact_seed(value: object, *, path: Path, row_number: int) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(
            f"Artifact seed must be a strict positive integer at row {row_number}: "
            f"{path}; received {value!r}"
        )
    return int(value)


def _manifest_allowed_values(
    manifest: dict,
) -> tuple[set[str] | None, set[str] | None, set[int] | None]:
    tasks_raw = manifest.get("tasks")
    models_raw = manifest.get("models")
    seeds_raw = manifest.get("seeds")
    tasks = (
        {str(value).lower() for value in tasks_raw}
        if isinstance(tasks_raw, list) and tasks_raw
        else None
    )
    models = (
        {str(value).lower() for value in models_raw}
        if isinstance(models_raw, list) and models_raw
        else None
    )
    seeds = (
        {
            _strict_positive_int(value, field="manifest seed")
            for value in seeds_raw
        }
        if isinstance(seeds_raw, list) and seeds_raw
        else None
    )
    return tasks, models, seeds


def _validate_identity_rows(
    rows: list[dict[str, str]],
    *,
    path: Path,
    manifest: dict,
    require_single_key: bool,
) -> set[tuple[str, str, int]]:
    allowed_tasks, allowed_models, allowed_seeds = _manifest_allowed_values(manifest)
    keys: set[tuple[str, str, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        task = row.get("task")
        model = row.get("model")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Artifact task is blank at row {row_number}: {path}")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Artifact model is blank at row {row_number}: {path}")
        seed = _strict_artifact_seed(row.get("seed"), path=path, row_number=row_number)
        task_key = task.lower()
        model_key = model.lower()
        if allowed_tasks is not None and task_key not in allowed_tasks:
            raise ValueError(f"Artifact task is not bound by manifest at row {row_number}: {path}")
        if allowed_models is not None and model_key not in allowed_models:
            raise ValueError(f"Artifact model is not bound by manifest at row {row_number}: {path}")
        if allowed_seeds is not None and seed not in allowed_seeds:
            raise ValueError(f"Artifact seed is not bound by manifest at row {row_number}: {path}")
        keys.add((task_key, model_key, seed))
    if require_single_key and len(keys) != 1:
        raise ValueError(f"Prediction CSV must contain exactly one task/model/seed key: {path}")
    return keys


def _semantic_csv_digest(
    rows: list[dict[str, str]], *, fields: list[str]
) -> str:
    return _stable_json_sha256(
        {
            "fields": fields,
            "rows": [[row[field] for field in fields] for row in rows],
        }
    )


def _csv_state(
    document: _CsvDocument,
    *,
    semantic_fields: list[str],
    observed_utc: str,
) -> dict[str, object]:
    return {
        "sha256": document.sha256,
        "bytes": document.byte_count,
        "rows": len(document.rows),
        "semantic_payload_sha256": _semantic_csv_digest(
            document.rows, fields=semantic_fields
        ),
        "observed_utc": observed_utc,
    }


def _atomic_write_csv(
    path: Path, *, fields: list[str], rows: list[dict[str, str]]
) -> None:
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _plan_run_table_provenance_upgrade(
    path: Path, *, manifest: dict, split_source: str, observed_utc: str
) -> _CsvUpgradePlan | None:
    if not path.exists():
        return None
    document = _read_csv_strict(path, artifact="sensitivity run table")
    final_set = set(FIELDNAMES)
    legacy_fields = [field for field in FIELDNAMES if field not in PROVENANCE_FIELDS]
    legacy_set = set(legacy_fields)
    observed_set = set(document.fields)
    if observed_set not in (final_set, legacy_set):
        raise ValueError(
            "Sensitivity run table schema cannot be upgraded safely: "
            f"missing={sorted(final_set - observed_set)}, "
            f"unexpected={sorted(observed_set - final_set)}"
        )
    rows = [dict(row) for row in document.rows]
    keys = _validate_identity_rows(
        rows, path=path, manifest=manifest, require_single_key=False
    )
    if len(keys) != len(rows):
        raise ValueError(f"Sensitivity run table has duplicate task/model/seed rows: {path}")
    expected_algorithm = _expected_split_algorithm(split_source)
    if observed_set == final_set:
        for row_number, row in enumerate(rows, start=2):
            if (
                row["split_source"] != split_source
                or row["split_algorithm"] != expected_algorithm
            ):
                raise ValueError(
                    "Existing sensitivity run-row provenance is incompatible at "
                    f"row {row_number}: {path}"
                )
        action = (
            "verified_canonical"
            if document.fields == FIELDNAMES
            else "normalized_column_order"
        )
    else:
        for row in rows:
            row["split_source"] = split_source
            row["split_algorithm"] = expected_algorithm
        action = "added_manifest_bound_provenance"
    return _CsvUpgradePlan(
        path=path,
        kind="run_table",
        target_fields=list(FIELDNAMES),
        semantic_fields=legacy_fields,
        rows=rows,
        action=action,
        old_state=_csv_state(
            document,
            semantic_fields=legacy_fields,
            observed_utc=observed_utc,
        ),
    )


def _plan_prediction_provenance_upgrades(
    directory: Path, *, manifest: dict, split_source: str, observed_utc: str
) -> list[_CsvUpgradePlan]:
    if not directory.exists():
        return []
    plans: list[_CsvUpgradePlan] = []
    final_set = set(PREDICTION_FIELDNAMES)
    core_set = set(PREDICTION_CORE_FIELDS)
    expected_algorithm = _expected_split_algorithm(split_source)
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("._"):
            continue
        document = _read_csv_strict(path, artifact="sensitivity prediction table")
        observed_set = set(document.fields)
        if observed_set not in (final_set, core_set):
            raise ValueError(
                f"Prediction schema cannot be upgraded safely: {path}; "
                f"missing={sorted(final_set - observed_set)}, "
                f"unexpected={sorted(observed_set - final_set)}"
            )
        rows = [dict(row) for row in document.rows]
        _validate_identity_rows(
            rows, path=path, manifest=manifest, require_single_key=True
        )
        if observed_set == final_set:
            for row_number, row in enumerate(rows, start=2):
                if (
                    row["split_source"] != split_source
                    or row["split_algorithm"] != expected_algorithm
                ):
                    raise ValueError(
                        f"Prediction provenance mismatch at row {row_number}: {path}"
                    )
            action = (
                "verified_canonical"
                if document.fields == PREDICTION_FIELDNAMES
                else "normalized_column_order"
            )
        else:
            for row in rows:
                row["split_source"] = split_source
                row["split_algorithm"] = expected_algorithm
            action = "added_manifest_bound_provenance"
        plans.append(
            _CsvUpgradePlan(
                path=path,
                kind="prediction",
                target_fields=list(PREDICTION_FIELDNAMES),
                semantic_fields=list(PREDICTION_CORE_FIELDS),
                rows=rows,
                action=action,
                old_state=_csv_state(
                    document,
                    semantic_fields=list(PREDICTION_CORE_FIELDS),
                    observed_utc=observed_utc,
                ),
            )
        )
    return plans


def _apply_csv_upgrade_plan(
    plan: _CsvUpgradePlan, *, completed_utc: str
) -> dict[str, object]:
    if plan.action != "verified_canonical":
        _atomic_write_csv(plan.path, fields=plan.target_fields, rows=plan.rows)
    document = _read_csv_strict(plan.path, artifact=plan.kind)
    if document.fields != plan.target_fields:
        raise RuntimeError(f"Post-upgrade field order verification failed: {plan.path}")
    new_state = _csv_state(
        document,
        semantic_fields=plan.semantic_fields,
        observed_utc=completed_utc,
    )
    if (
        plan.old_state["semantic_payload_sha256"]
        != new_state["semantic_payload_sha256"]
    ):
        raise RuntimeError(f"Semantic payload changed during provenance upgrade: {plan.path}")
    return {
        "kind": plan.kind,
        "path": str(plan.path.resolve()),
        "action": plan.action,
        "old": plan.old_state,
        "new": new_state,
        "semantic_payload_unchanged": True,
    }


def _upgrade_existing_run_table_provenance(
    path: Path, *, manifest_path: Path, split_source: str
) -> dict[str, object] | None:
    """Upgrade/normalize one run table only when a v2 manifest binds it."""

    manifest = _load_upgrade_manifest(manifest_path, split_source=split_source)
    observed_utc = _utc_now()
    plan = _plan_run_table_provenance_upgrade(
        path,
        manifest=manifest,
        split_source=split_source,
        observed_utc=observed_utc,
    )
    if plan is None:
        return None
    return _apply_csv_upgrade_plan(plan, completed_utc=_utc_now())


def _upgrade_existing_prediction_provenance(
    directory: Path, *, manifest_path: Path, split_source: str
) -> list[dict[str, object]]:
    """Upgrade/normalize predictions after prevalidating every CSV in the set."""

    manifest = _load_upgrade_manifest(manifest_path, split_source=split_source)
    plans = _plan_prediction_provenance_upgrades(
        directory,
        manifest=manifest,
        split_source=split_source,
        observed_utc=_utc_now(),
    )
    completed_utc = _utc_now()
    return [
        _apply_csv_upgrade_plan(plan, completed_utc=completed_utc) for plan in plans
    ]


def _upgrade_existing_audit(
    path: Path,
    *,
    tasks_with_compatibility_sources: set[str],
    completed_tasks: set[str] | None = None,
) -> None:
    """Atomically upgrade safe pre-v2 audit rows during a resume.

    A legacy row for a task containing a compatibility source is not safe to
    reinterpret because the old canonicalizer may have counted that source as
    an invalid-SMILES deletion. Rows for all other tasks are behaviorally
    identical and receive explicit zero-valued compatibility fields.
    """

    if not path.exists():
        return
    payloads: list[dict] = []
    changed = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        task = str(payload.get("task", ""))
        if not task:
            raise ValueError(f"Audit line {line_number} has no task: {path}")
        if int(payload.get("audit_schema_version", 1)) < 2:
            if task.lower() in tasks_with_compatibility_sources:
                if task.lower() in {
                    value.lower() for value in (completed_tasks or set())
                }:
                    raise ValueError(
                        "Cannot upgrade a legacy cleaned-sensitivity audit for a "
                        "completed task containing parser compatibility sources: "
                        f"{task}"
                    )
                # Preparation audits are written before any model fit. If a run
                # was stopped at that boundary, discard only this regenerable
                # derived row and let --resume recreate it under schema v2.
                changed = True
                continue
            payload["audit_schema_version"] = 2
            payload["model_input_representation_policy"] = (
                SMILES_COMPATIBILITY_POLICY
            )
            payload["parser_compatibility_repairs_count_as_cleaning_deletions"] = (
                False
            )
            for report in payload.get("splits", {}).values():
                report["parser_compatibility_repairs_applied"] = 0
                report[
                    "parser_compatibility_repairs_count_as_cleaning_deletions"
                ] = False
            changed = True
        payloads.append(payload)
    if not changed:
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe_parameter(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_safe_parameter(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe_parameter(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_parameter(item) for item in value]
    return repr(value)


def _resolved_estimator_summary(
    *, tasks: list[str], models: list[str], seeds: list[int], threads_per_model: int
) -> dict[str, object]:
    """Record the actual estimator classes and get_params output for every seed."""

    from admet_benchmark.models import make_traditional_model

    package_by_model = {
        "rf": "scikit-learn",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
    }
    task_types = sorted({TDC_TASK_META[task]["task_type"] for task in tasks})
    summary: dict[str, object] = {
        "parameter_resolution": (
            "make_traditional_model(...).get_params(deep=False), resolved for "
            "each configured task type and seed"
        ),
        "models": {},
    }
    model_payloads = summary["models"]
    assert isinstance(model_payloads, dict)
    for model in models:
        package = package_by_model[model]
        implementations: dict[str, object] = {}
        for task_type in task_types:
            parameters_by_seed: dict[str, object] = {}
            implementation_class: str | None = None
            for seed in seeds:
                estimator = make_traditional_model(
                    model,
                    task_type,  # type: ignore[arg-type]
                    seed=seed,
                    n_jobs=threads_per_model,
                )
                current_class = (
                    f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__}"
                )
                if implementation_class is None:
                    implementation_class = current_class
                elif implementation_class != current_class:
                    raise RuntimeError(
                        f"Estimator implementation changed across seeds for {model}/{task_type}"
                    )
                parameters_by_seed[str(seed)] = _json_safe_parameter(
                    estimator.get_params(deep=False)
                )
            implementations[task_type] = {
                "class": implementation_class,
                "parameters_by_seed": parameters_by_seed,
            }
        model_payloads[model] = {
            "package": package,
            "implementation_version": _package_version(package),
            "implementations": implementations,
        }
    return summary


def _runner_reproducibility_metadata(
    *, tasks: list[str], models: list[str], seeds: list[int], threads_per_model: int
) -> dict[str, object]:
    model_factory_path = ROOT / "src" / "admet_benchmark" / "models.py"
    return {
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "runner_source_provenance_scope": (
            "attests the runner source used for manifest creation/upgrade and "
            "subsequent resumes; rows completed before this field was introduced "
            "are not retroactively source-attested"
        ),
        "model_factory_source_sha256": _sha256(model_factory_path),
        "resolved_estimator_summary": _resolved_estimator_summary(
            tasks=tasks,
            models=models,
            seeds=seeds,
            threads_per_model=threads_per_model,
        ),
    }


def _metric_values(task_type: str, y_true: np.ndarray, prediction: np.ndarray) -> dict:
    from admet_benchmark.metrics import classification_metrics, regression_metrics

    if task_type == "classification":
        return classification_metrics(y_true, prediction)
    return regression_metrics(y_true, prediction)


def _primary_effect(metric: str, official: float, cleaned: float) -> tuple[float, float, float]:
    raw_delta = float(cleaned - official)
    oriented = float(official - cleaned) if metric in {"mae", "rmse", "brier"} else raw_delta
    relative = oriented / max(abs(float(official)), 1e-12)
    return raw_delta, oriented, float(relative)


def _prediction_path(directory: Path, task: str, model: str, seed: int) -> Path:
    return directory / f"{task.lower()}_{model.lower()}_seed{seed}_paired.csv"


def _existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    document = _read_csv_strict(path, artifact="sensitivity run table")
    legacy_fields = [field for field in FIELDNAMES if field not in PROVENANCE_FIELDS]
    if set(document.fields) not in (set(FIELDNAMES), set(legacy_fields)):
        raise ValueError(f"Sensitivity run table has an unsupported schema: {path}")
    keys = _validate_identity_rows(
        document.rows, path=path, manifest={}, require_single_key=False
    )
    if len(keys) != len(document.rows):
        raise ValueError(f"Sensitivity run table has duplicate task/model/seed rows: {path}")
    return keys


def _existing_audit_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Audit task is blank at line {line_number}: {path}")
        seed = _strict_positive_int(
            payload.get("seed"), field=f"audit seed at line {line_number}"
        )
        key = (task.lower(), seed)
        if key in keys:
            raise ValueError(f"Duplicate audit task/seed at line {line_number}: {path}")
        keys.add(key)
    return keys


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        document = _read_csv_strict(path, artifact="sensitivity run table")
        if document.fields != FIELDNAMES:
            raise ValueError(
                "Refusing to append to a run table whose columns are not in exact "
                f"FIELDNAMES order; run --upgrade-only first: {path}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_failure(path: Path, payload: dict) -> None:
    _append_jsonl(path, payload)


def _featurize(frame: pd.DataFrame, radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from admet_benchmark.data_loader import extract_xy
    from admet_benchmark.featurization import align_labels, morgan_count_matrix

    smiles, labels = extract_xy(frame)
    matrix, kept = morgan_count_matrix(smiles, radius=radius, n_bits=n_bits)
    kept_smiles = [smiles[index] for index in kept]
    return matrix, align_labels(labels, kept), kept_smiles


def _write_predictions(
    path: Path,
    *,
    task: str,
    model: str,
    seed: int,
    split_source: str,
    smiles: list[str],
    labels: np.ndarray,
    official_prediction: np.ndarray,
    cleaned_prediction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "task": task,
            "model": model,
            "seed": seed,
            "split_source": split_source,
            "split_algorithm": (
                LOCKED_CACHE_SPLIT_ALGORITHM
                if split_source == LOCKED_CACHE_SPLIT_SOURCE
                else "pytdc_default_task_specific_runtime"
            ),
            "evaluation_set": "common_cleaned_test",
            "canonical_smiles": smiles,
            "label": labels,
            "official_prediction": official_prediction,
            "cleaned_prediction": cleaned_prediction,
        }
    )
    frame = frame.loc[:, PREDICTION_FIELDNAMES]
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_manifest(
    path: Path,
    *,
    config_path: Path,
    data_dir: Path,
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    radius: int,
    n_bits: int,
    threads_per_model: int,
    save_predictions: bool,
    split_source: str,
    completed_tasks: set[str] | None = None,
) -> None:
    inputs: dict[str, dict[str, str | None]] = {}
    for task in tasks:
        task_dir = data_dir / "admet_group" / task.lower()
        inputs[task] = {}
        for filename in ("train_val.csv", "test.csv"):
            candidate = task_dir / filename
            inputs[task][filename] = _sha256(candidate) if candidate.exists() else None
    compatibility_source_counts = _compatibility_source_counts(data_dir, tasks)
    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    created_utc = _utc_now()
    reproducibility_metadata = _runner_reproducibility_metadata(
        tasks=tasks,
        models=models,
        seeds=seeds,
        threads_per_model=threads_per_model,
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc,
        "protocol": "paired-common-test-v1",
        "main_result_policy": "TDC official split remains the primary analysis",
        "split_source": split_source,
        "split_algorithm": (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        ),
        "cpu_only": True,
        "save_predictions": save_predictions,
        "tasks": tasks,
        "models": models,
        "seeds": seeds,
        "fingerprint": {"radius": radius, "n_bits": n_bits},
        "threads_per_model": threads_per_model,
        "cleaning_policy": CleaningPolicy().to_dict(),
        "canonicalization_input_policy": (
            "apply_enumerated_parser_compatibility_before_rdkit_canonicalization"
        ),
        "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
        "parser_compatibility_repairs_count_as_cleaning_deletions": False,
        "compatibility_mapping_sha256": _stable_json_sha256(
            compatibility_mapping
        ),
        "compatibility_source_occurrences": compatibility_source_counts,
        "config_path": str(config_path.resolve()),
        "config_logical_path": _config_logical_path(config_path),
        "config_sha256": _sha256(config_path),
        "input_sha256": inputs,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "rdkit": _package_version("rdkit"),
            "xgboost": _package_version("xgboost"),
            "lightgbm": _package_version("lightgbm"),
        },
        **reproducibility_metadata,
        "runner_source_recorded_utc": created_utc,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        compatibility_fields = [
            "protocol",
            "cpu_only",
            "save_predictions",
            "tasks",
            "models",
            "seeds",
            "fingerprint",
            "threads_per_model",
            "cleaning_policy",
            "config_sha256",
            "input_sha256",
            "runtime",
            "split_source",
            "split_algorithm",
        ]
        mismatched = [
            field
            for field in compatibility_fields
            if existing.get(
                field,
                (
                    PYTDC_SPLIT_SOURCE
                    if field == "split_source"
                    else "pytdc_default_task_specific_runtime"
                    if field == "split_algorithm"
                    else None
                ),
            )
            != payload.get(field)
        ]
        if mismatched:
            raise ValueError(
                "Existing sensitivity manifest is incompatible with this run: "
                + ", ".join(mismatched)
            )
        existing_schema = int(existing.get("schema_version", 1))
        parser_policy_fields = [
            "canonicalization_input_policy",
            "model_input_representation_policy",
            "parser_compatibility_repairs_count_as_cleaning_deletions",
            "compatibility_mapping_sha256",
            "compatibility_source_occurrences",
        ]
        reproducibility_fields = [
            "config_logical_path",
            "runner_source_sha256",
            "runner_source_provenance_scope",
            "model_factory_source_sha256",
            "resolved_estimator_summary",
        ]
        supplemental_fields = parser_policy_fields + reproducibility_fields
        if existing_schema == 1:
            unsafe_tasks = {
                task.lower()
                for task, count in compatibility_source_counts.items()
                if count
            } & {task.lower() for task in (completed_tasks or set())}
            if unsafe_tasks:
                raise ValueError(
                    "Cannot upgrade the v1 sensitivity manifest because completed "
                    "rows may have treated parser compatibility sources as cleaning "
                    f"deletions: {sorted(unsafe_tasks)}"
                )
            upgraded = dict(existing)
            upgraded["schema_version"] = MANIFEST_SCHEMA_VERSION
            upgraded["upgraded_from_schema_version"] = 1
            upgraded["schema_upgrade_utc"] = datetime.now(timezone.utc).isoformat()
            for field in supplemental_fields:
                upgraded[field] = payload[field]
            upgraded["runner_source_recorded_utc"] = created_utc
            _atomic_write_json(path, upgraded)
            return
        if existing_schema != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sensitivity manifest schema_version={existing_schema}"
            )
        mismatched_new = [
            field
            for field in supplemental_fields
            if field in existing and existing[field] != payload[field]
        ]
        if mismatched_new:
            raise ValueError(
                "Existing sensitivity manifest has incompatible supplemental provenance: "
                + ", ".join(mismatched_new)
            )
        missing_new = [field for field in supplemental_fields if field not in existing]
        if missing_new:
            upgraded = dict(existing)
            for field in missing_new:
                upgraded[field] = payload[field]
            upgraded.setdefault("runner_source_recorded_utc", created_utc)
            _atomic_write_json(path, upgraded)
        return
    _atomic_write_json(path, payload)


def _json_state(path: Path, payload: dict, *, observed_utc: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": 1,
        "semantic_payload_sha256": _stable_json_sha256(payload),
        "observed_utc": observed_utc,
    }


def _resolve_upgrade_config_binding(
    manifest: dict, *, config_path: Path
) -> tuple[str, dict[str, object] | None]:
    """Allow only the reviewed, non-computational scaffold-audit correction.

    The completed runner never reads ``audit.scaffold_overlap``.  If that one
    declarative flag is corrected after execution, reconstructing the exact old
    bytes must reproduce the manifest-bound execution hash.  Any other edit is
    refused.
    """

    execution_hash = manifest.get("config_sha256")
    if not isinstance(execution_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", execution_hash
    ):
        raise ValueError("Manifest config_sha256 is missing or malformed")
    raw = config_path.read_bytes()
    current_hash = hashlib.sha256(raw).hexdigest()
    if current_hash == execution_hash:
        return current_hash, None

    corrected_token = b"  scaffold_overlap: false"
    execution_token = b"  scaffold_overlap: true"
    if raw.count(corrected_token) != 1:
        raise ValueError(
            "Upgrade config content is not bound by manifest config_sha256 and "
            "is not the one permitted audit.scaffold_overlap correction"
        )
    reconstructed = raw.replace(corrected_token, execution_token, 1)
    reconstructed_hash = hashlib.sha256(reconstructed).hexdigest()
    if reconstructed_hash != execution_hash:
        raise ValueError(
            "Config differs from the execution-bound version by more than the "
            "permitted audit.scaffold_overlap true-to-false correction"
        )
    return current_hash, {
        "field": "audit.scaffold_overlap",
        "old_value": True,
        "new_value": False,
        "execution_config_sha256": execution_hash,
        "corrected_config_sha256": current_hash,
        "computational_effect": False,
        "reason": (
            "Correct declarative audit capability: cleaned sensitivity audits "
            "canonical-molecule overlap, not scaffold overlap; the runner never "
            "reads this config field"
        ),
    }


def _plan_upgrade_manifest_supplement(
    manifest_path: Path,
    *,
    manifest: dict,
    config_path: Path,
    observed_utc: str,
) -> tuple[dict, dict[str, object], str, dict[str, object]]:
    """Prepare safe v2 metadata additions without trusting absolute config_path."""

    if not config_path.exists():
        raise ValueError(f"Upgrade config does not exist: {config_path}")
    current_config_hash, metadata_correction = _resolve_upgrade_config_binding(
        manifest, config_path=config_path
    )
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Manifest fingerprint must be a mapping")
    tasks, models, seeds, _, _, threads_per_model = _validate_runner_values(
        tasks=manifest.get("tasks"),
        models=manifest.get("models"),
        seeds=manifest.get("seeds"),
        radius=fingerprint.get("radius"),
        n_bits=fingerprint.get("n_bits"),
        threads_per_model=manifest.get("threads_per_model"),
    )
    desired = {
        "config_logical_path": _config_logical_path(config_path),
        **_runner_reproducibility_metadata(
            tasks=tasks,
            models=models,
            seeds=seeds,
            threads_per_model=threads_per_model,
        ),
    }
    mismatched = [
        field
        for field, value in desired.items()
        if field in manifest and manifest[field] != value
    ]
    if mismatched:
        raise ValueError(
            "Existing manifest has incompatible reproducibility metadata: "
            + ", ".join(mismatched)
        )
    upgraded = dict(manifest)
    execution_config_hash = str(
        manifest.get("execution_config_sha256", manifest["config_sha256"])
    )
    if metadata_correction is not None:
        if execution_config_hash != metadata_correction["execution_config_sha256"]:
            raise ValueError("Existing execution_config_sha256 is incompatible")
        upgraded["config_sha256"] = current_config_hash
        upgraded["execution_config_sha256"] = execution_config_hash
        existing_corrections = upgraded.get("config_metadata_corrections", [])
        if not isinstance(existing_corrections, list):
            raise ValueError("config_metadata_corrections must be a list")
        upgraded["config_metadata_corrections"] = [
            *existing_corrections,
            {**metadata_correction, "recorded_utc": observed_utc},
        ]
    else:
        upgraded.setdefault("execution_config_sha256", execution_config_hash)
        upgraded.setdefault("config_metadata_corrections", [])
    missing = [field for field in desired if field not in upgraded]
    for field in missing:
        upgraded[field] = desired[field]
    if missing:
        upgraded.setdefault("runner_source_recorded_utc", observed_utc)
    changed = upgraded != manifest
    action = "supplemented_reproducibility_metadata" if changed else "verified_canonical"
    config_binding = {
        "path": str(config_path.resolve()),
        "logical_path": _config_logical_path(config_path),
        "manifest_config_sha256_before": manifest["config_sha256"],
        "manifest_config_sha256_after": upgraded["config_sha256"],
        "execution_config_sha256": upgraded["execution_config_sha256"],
        "current_config_sha256": current_config_hash,
        "metadata_only_correction": metadata_correction,
    }
    return (
        upgraded,
        _json_state(manifest_path, manifest, observed_utc=observed_utc),
        action,
        config_binding,
    )


def _apply_manifest_upgrade_plan(
    manifest_path: Path,
    *,
    payload: dict,
    old_state: dict[str, object],
    action: str,
    completed_utc: str,
) -> dict[str, object]:
    if action != "verified_canonical":
        _atomic_write_json(manifest_path, payload)
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    if written != payload:
        raise RuntimeError(f"Post-upgrade manifest verification failed: {manifest_path}")
    return {
        "kind": "manifest",
        "path": str(manifest_path.resolve()),
        "action": action,
        "old": old_state,
        "new": _json_state(
            manifest_path, written, observed_utc=completed_utc
        ),
        "semantic_payload_unchanged": old_state["semantic_payload_sha256"]
        == _stable_json_sha256(written),
    }


def _aggregate_upgrade_inventory(
    entries: list[dict[str, object]],
) -> dict[str, object]:
    def inventory_side(side: str) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for entry in entries:
            state = entry[side]
            assert isinstance(state, dict)
            payloads.append(
                {
                    "kind": entry["kind"],
                    "path": entry["path"],
                    "sha256": state["sha256"],
                    "bytes": state["bytes"],
                    "rows": state["rows"],
                    "semantic_payload_sha256": state[
                        "semantic_payload_sha256"
                    ],
                }
            )
        return payloads

    old_inventory = inventory_side("old")
    new_inventory = inventory_side("new")
    by_kind: dict[str, int] = {}
    for entry in entries:
        kind = str(entry["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
    csv_entries = [entry for entry in entries if entry["kind"] != "manifest"]
    return {
        "file_count": len(entries),
        "files_by_kind": by_kind,
        "changed_file_count": sum(
            1
            for entry in entries
            if entry["old"]["sha256"] != entry["new"]["sha256"]  # type: ignore[index]
        ),
        "old_total_bytes": sum(int(item["bytes"]) for item in old_inventory),
        "new_total_bytes": sum(int(item["bytes"]) for item in new_inventory),
        "old_total_rows": sum(int(item["rows"]) for item in old_inventory),
        "new_total_rows": sum(int(item["rows"]) for item in new_inventory),
        "old_inventory_sha256": _stable_json_sha256(old_inventory),
        "new_inventory_sha256": _stable_json_sha256(new_inventory),
        "old_aggregate_semantic_payload_sha256": _stable_json_sha256(
            [item["semantic_payload_sha256"] for item in old_inventory]
        ),
        "new_aggregate_semantic_payload_sha256": _stable_json_sha256(
            [item["semantic_payload_sha256"] for item in new_inventory]
        ),
        "csv_semantic_payloads_unchanged": all(
            bool(entry["semantic_payload_unchanged"]) for entry in csv_entries
        ),
    }


def _default_upgrade_receipt_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(
        manifest_path.stem + "_provenance_upgrade_receipt.json"
    )


def _run_upgrade_only(args: argparse.Namespace) -> dict:
    """Upgrade provenance metadata only; never load data, fit, or append audits."""

    started_utc = _utc_now()
    manifest_path = Path(args.manifest)
    manifest = _load_upgrade_manifest(
        manifest_path, split_source=args.split_source
    )
    manifest_payload, manifest_old_state, manifest_action, config_binding = (
        _plan_upgrade_manifest_supplement(
            manifest_path,
            manifest=manifest,
            config_path=Path(args.config),
            observed_utc=started_utc,
        )
    )
    run_plan = _plan_run_table_provenance_upgrade(
        Path(args.out),
        manifest=manifest,
        split_source=args.split_source,
        observed_utc=started_utc,
    )
    if run_plan is None:
        raise ValueError("--upgrade-only requires an existing non-empty run CSV")
    prediction_plans = _plan_prediction_provenance_upgrades(
        Path(args.predictions_dir),
        manifest=manifest,
        split_source=args.split_source,
        observed_utc=started_utc,
    )
    ignored_appledouble_sidecars = sorted(
        str(path.resolve())
        for path in Path(args.predictions_dir).glob("._*.csv")
        if path.is_file()
    )
    if bool(manifest.get("save_predictions")) and not prediction_plans:
        raise ValueError(
            "Manifest requires prediction exports, but --upgrade-only found no "
            f"prediction CSV files in {args.predictions_dir}"
        )
    if not bool(manifest.get("save_predictions")) and prediction_plans:
        raise ValueError(
            "Manifest declares save_predictions=false but prediction CSV files exist"
        )

    if args.upgrade_receipt is not None and not str(args.upgrade_receipt).strip():
        raise ValueError("--upgrade-receipt must not be blank")
    receipt_path = (
        Path(args.upgrade_receipt)
        if args.upgrade_receipt is not None
        else _default_upgrade_receipt_path(manifest_path)
    )
    artifact_paths = {
        manifest_path.resolve(),
        Path(args.out).resolve(),
        *(plan.path.resolve() for plan in prediction_plans),
    }
    if receipt_path.resolve() in artifact_paths:
        raise ValueError("Upgrade receipt path must not overwrite an upgraded artifact")
    if receipt_path.exists() and receipt_path.is_dir():
        raise ValueError("Upgrade receipt path must be a file, not a directory")

    completed_utc = _utc_now()
    entries = [
        _apply_manifest_upgrade_plan(
            manifest_path,
            payload=manifest_payload,
            old_state=manifest_old_state,
            action=manifest_action,
            completed_utc=completed_utc,
        ),
        _apply_csv_upgrade_plan(run_plan, completed_utc=completed_utc),
    ]
    entries.extend(
        _apply_csv_upgrade_plan(plan, completed_utc=completed_utc)
        for plan in prediction_plans
    )
    aggregate_inventory = _aggregate_upgrade_inventory(entries)
    aggregate_inventory["ignored_appledouble_sidecars_count"] = len(
        ignored_appledouble_sidecars
    )
    receipt = {
        "schema_version": UPGRADE_RECEIPT_SCHEMA_VERSION,
        "operation": "cleaned_sensitivity_manifest_bound_provenance_upgrade",
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "manifest_binding": {
            "path": str(manifest_path.resolve()),
            "split_source": args.split_source,
            "split_algorithm": _expected_split_algorithm(args.split_source),
        },
        "config_binding": config_binding,
        "files": entries,
        "ignored_appledouble_sidecars": ignored_appledouble_sidecars,
        "aggregate_inventory": aggregate_inventory,
    }
    _atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt["aggregate_inventory"], indent=2, sort_keys=True))
    print(f"upgrade receipt: {receipt_path}")
    return receipt


def _require_canonical_resume_artifacts(
    *,
    out: Path,
    predictions_dir: Path,
    manifest_path: Path,
    split_source: str,
    save_predictions: bool,
) -> None:
    manifest = _load_upgrade_manifest(manifest_path, split_source=split_source)
    run_plan = _plan_run_table_provenance_upgrade(
        out,
        manifest=manifest,
        split_source=split_source,
        observed_utc=_utc_now(),
    )
    if run_plan is not None and run_plan.action != "verified_canonical":
        raise ValueError(
            "Resume requires exact final run-table FIELDNAMES order and provenance; "
            "run --upgrade-only first"
        )
    prediction_plans = _plan_prediction_provenance_upgrades(
        predictions_dir,
        manifest=manifest,
        split_source=split_source,
        observed_utc=_utc_now(),
    )
    noncanonical = [
        str(plan.path)
        for plan in prediction_plans
        if plan.action != "verified_canonical"
    ]
    if noncanonical:
        raise ValueError(
            "Resume requires canonical prediction schemas; run --upgrade-only first: "
            + ", ".join(noncanonical)
        )
    if save_predictions and out.exists() and not prediction_plans:
        raise ValueError("Resume found run rows but no manifest-required prediction CSVs")


def _estimate_cost(tasks: list[str], models: list[str], seeds: list[int], historical: Path) -> dict:
    expected_units = len(tasks) * len(models) * len(seeds)
    estimate: dict[str, object] = {
        "paired_units": expected_units,
        "model_fits": 2 * expected_units,
        "gpu_required": False,
    }
    if historical.exists():
        frame = pd.read_csv(historical)
        mask = (
            frame["task"].astype(str).str.lower().isin({task.lower() for task in tasks})
            & frame["model"].astype(str).str.lower().isin({model.lower() for model in models})
            & frame["seed"].astype(int).isin(seeds)
        )
        matched = frame.loc[mask]
        if len(matched) == expected_units:
            one_side_seconds = float(matched["fit_seconds"].sum())
            estimate.update(
                {
                    "historical_units_matched": int(len(matched)),
                    "estimated_paired_fit_seconds": round(2 * one_side_seconds, 1),
                    "estimated_paired_fit_minutes": round(2 * one_side_seconds / 60, 1),
                    "padded_wall_time_minutes": round(max(30.0, 4 * one_side_seconds / 60), 1),
                    "padded_wall_time_note": "Includes 4x allowance for canonicalization, fingerprints, I/O, and lower CPU concurrency.",
                }
            )
    return estimate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired official-vs-cleaned sensitivity analysis on CPU."
    )
    parser.add_argument("--config", default=str(ROOT / "experiments" / "config" / "sensitivity.yaml"))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=_positive_int_argument)
    parser.add_argument("--radius", type=_positive_int_argument)
    parser.add_argument("--n-bits", type=_positive_int_argument)
    parser.add_argument("--threads-per-model", type=_positive_int_argument)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=PYTDC_SPLIT_SOURCE,
        help="Explicit source for the paired baseline; use locked_cache to match formal models.",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_runs.csv")
    )
    parser.add_argument(
        "--audit", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_audit.jsonl")
    )
    parser.add_argument(
        "--predictions-dir",
        default=str(ROOT / "results" / "predictions" / "cleaned_sensitivity"),
    )
    parser.add_argument(
        "--manifest", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_manifest.json")
    )
    parser.add_argument(
        "--failures", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_failures.jsonl")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--upgrade-only",
        action="store_true",
        help=(
            "After every older runner process has exited, atomically upgrade only "
            "manifest-bound run/prediction provenance and write a receipt; never fit."
        ),
    )
    parser.add_argument(
        "--upgrade-receipt",
        help="Optional receipt JSON path (default: derived from --manifest).",
    )
    return parser


def _run_experiment(args: argparse.Namespace) -> None:

    config_path = Path(args.config)
    config = _load_config(config_path)
    tasks, models, seeds, radius, n_bits, threads_per_model = (
        _resolve_and_validate_configuration(
            config,
            tasks_override=args.tasks,
            models_override=args.models,
            seeds_override=args.seeds,
            radius_override=args.radius,
            n_bits_override=args.n_bits,
            threads_override=args.threads_per_model,
        )
    )

    if args.estimate_only:
        estimate = _estimate_cost(
            tasks,
            models,
            seeds,
            ROOT / "results" / "tables" / "traditional_grid_local_prelim.csv",
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return

    from admet_benchmark.data_loader import load_tdc_official
    from admet_benchmark.models import make_traditional_model, predict_scores
    from admet_benchmark.sensitivity import clean_official_split_for_sensitivity
    from admet_benchmark.utils import set_seed

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    audit_path = Path(args.audit)
    predictions_dir = Path(args.predictions_dir)
    failures = Path(args.failures)
    manifest_path = Path(args.manifest)
    existing_primary_artifacts = [
        path for path in (out, audit_path, manifest_path) if path.exists()
    ]
    if existing_primary_artifacts and not args.resume:
        raise FileExistsError(
            "Sensitivity artifacts already exist: "
            + ", ".join(str(path) for path in existing_primary_artifacts)
            + ". Use --resume or choose new output paths."
        )
    done = _existing_keys(out) if args.resume else set()
    audited = _existing_audit_keys(audit_path) if args.resume else set()
    compatibility_source_counts = _compatibility_source_counts(data_dir, tasks)
    tasks_with_compatibility_sources = {
        task.lower()
        for task, count in compatibility_source_counts.items()
        if count
    }
    if args.resume:
        _upgrade_existing_audit(
            audit_path,
            tasks_with_compatibility_sources=tasks_with_compatibility_sources,
            completed_tasks={task for task, _, _ in done},
        )
        # The atomic audit upgrade may remove a preparation-only row for a
        # compatibility-bearing task.  Re-read it before deciding whether a
        # v1 manifest can be upgraded; using the pre-upgrade set would falsely
        # classify that task as completed and block a safe resume.
        audited = _existing_audit_keys(audit_path)
    completed_tasks = {task for task, _, _ in done} | {
        task for task, _ in audited
    }
    _write_manifest(
        manifest_path,
        config_path=config_path,
        data_dir=data_dir,
        tasks=tasks,
        models=models,
        seeds=seeds,
        radius=radius,
        n_bits=n_bits,
        threads_per_model=threads_per_model,
        save_predictions=not args.no_predictions,
        split_source=args.split_source,
        completed_tasks=completed_tasks,
    )
    if args.resume:
        _require_canonical_resume_artifacts(
            out=out,
            predictions_dir=predictions_dir,
            manifest_path=manifest_path,
            split_source=args.split_source,
            save_predictions=not args.no_predictions,
        )

    for task in tasks:
        for seed in seeds:
            print(f"prepare task={task} seed={seed}", flush=True)
            try:
                set_seed(seed)
                official = load_tdc_official(
                    task,
                    data_dir=data_dir,
                    seed=seed,
                    split_source=args.split_source,
                )
                cleaned = clean_official_split_for_sensitivity(official)
                audit_key = (task.lower(), seed)
                if audit_key not in audited:
                    _append_jsonl(
                        audit_path,
                        {"task": task, "dataset_name": official.name, "seed": seed, **cleaned.audit},
                    )
                    audited.add(audit_key)

                x_official_train, y_official_train, _ = _featurize(
                    official.train, radius, n_bits
                )
                x_cleaned_train, y_cleaned_train, _ = _featurize(cleaned.train, radius, n_bits)
                x_valid, y_valid, _ = _featurize(cleaned.valid, radius, n_bits)
                x_test, y_test, test_smiles = _featurize(cleaned.test, radius, n_bits)
            except Exception as exc:
                payload = {
                    "stage": "prepare",
                    "task": task,
                    "seed": seed,
                    "error": repr(exc),
                }
                _append_failure(failures, payload)
                print(f"failed: {payload}", flush=True)
                continue

            for model_name in models:
                key = (task.lower(), model_name.lower(), seed)
                if args.resume and key in done:
                    print(f"skip existing task={task} model={model_name} seed={seed}")
                    continue
                print(f"fit task={task} model={model_name} seed={seed}", flush=True)
                try:
                    set_seed(seed)
                    official_model = make_traditional_model(
                        model_name, official.task_type, seed=seed, n_jobs=threads_per_model
                    )
                    start = time.perf_counter()
                    official_model.fit(x_official_train, y_official_train)
                    official_fit_seconds = time.perf_counter() - start

                    set_seed(seed)
                    cleaned_model = make_traditional_model(
                        model_name, official.task_type, seed=seed, n_jobs=threads_per_model
                    )
                    start = time.perf_counter()
                    cleaned_model.fit(x_cleaned_train, y_cleaned_train)
                    cleaned_fit_seconds = time.perf_counter() - start

                    predictions = {
                        "official_common_valid": predict_scores(
                            official_model, x_valid, official.task_type
                        ),
                        "cleaned_common_valid": predict_scores(
                            cleaned_model, x_valid, official.task_type
                        ),
                        "official_common_test": predict_scores(
                            official_model, x_test, official.task_type
                        ),
                        "cleaned_common_test": predict_scores(
                            cleaned_model, x_test, official.task_type
                        ),
                    }
                    metrics = {
                        "official_common_valid": _metric_values(
                            official.task_type, y_valid, predictions["official_common_valid"]
                        ),
                        "cleaned_common_valid": _metric_values(
                            official.task_type, y_valid, predictions["cleaned_common_valid"]
                        ),
                        "official_common_test": _metric_values(
                            official.task_type, y_test, predictions["official_common_test"]
                        ),
                        "cleaned_common_test": _metric_values(
                            official.task_type, y_test, predictions["cleaned_common_test"]
                        ),
                    }
                    primary_official = float(
                        metrics["official_common_test"][official.metric]
                    )
                    primary_cleaned = float(metrics["cleaned_common_test"][official.metric])
                    raw_delta, oriented_delta, relative_delta = _primary_effect(
                        official.metric, primary_official, primary_cleaned
                    )
                    row = {
                        "task": task,
                        "dataset_name": official.name,
                        "task_type": official.task_type,
                        "official_metric": official.metric,
                        "model": model_name,
                        "seed": seed,
                        "split_source": args.split_source,
                        "split_algorithm": (
                            LOCKED_CACHE_SPLIT_ALGORITHM
                            if args.split_source == LOCKED_CACHE_SPLIT_SOURCE
                            else "pytdc_default_task_specific_runtime"
                        ),
                        "radius": radius,
                        "n_bits": n_bits,
                        "threads_per_model": threads_per_model,
                        "official_train_n": len(y_official_train),
                        "cleaned_train_n": len(y_cleaned_train),
                        "common_valid_n": len(y_valid),
                        "common_test_n": len(y_test),
                        "official_fit_seconds": round(official_fit_seconds, 3),
                        "cleaned_fit_seconds": round(cleaned_fit_seconds, 3),
                        "primary_official_common_test": primary_official,
                        "primary_cleaned_common_test": primary_cleaned,
                        "primary_raw_delta_cleaned_minus_official": raw_delta,
                        "primary_oriented_delta": oriented_delta,
                        "primary_relative_oriented_delta": relative_delta,
                    }
                    for prefix, values in metrics.items():
                        for metric_name, value in values.items():
                            row[f"{prefix}_{metric_name}"] = value

                    if not args.no_predictions:
                        _write_predictions(
                            _prediction_path(predictions_dir, task, model_name, seed),
                            task=task,
                            model=model_name,
                            seed=seed,
                            split_source=args.split_source,
                            smiles=test_smiles,
                            labels=y_test,
                            official_prediction=predictions["official_common_test"],
                            cleaned_prediction=predictions["cleaned_common_test"],
                        )
                    _append_csv(out, row)
                    done.add(key)
                    print(
                        f"done task={task} model={model_name} seed={seed} "
                        f"oriented_delta={oriented_delta:.6g}",
                        flush=True,
                    )
                except Exception as exc:
                    payload = {
                        "stage": "fit",
                        "task": task,
                        "model": model_name,
                        "seed": seed,
                        "error": repr(exc),
                    }
                    _append_failure(failures, payload)
                    print(f"failed: {payload}", flush=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.estimate_only and args.upgrade_only:
        parser.error("--estimate-only and --upgrade-only are mutually exclusive")
    if args.upgrade_only and args.resume:
        parser.error("--upgrade-only is a separate operation; do not combine it with --resume")
    if args.upgrade_only and args.no_predictions:
        parser.error(
            "--upgrade-only follows manifest save_predictions and cannot be combined "
            "with --no-predictions"
        )
    if args.estimate_only:
        _run_experiment(args)
        return

    # The lock begins before any config/manifest/artifact read-modify-write and
    # remains held through the final fit or atomic upgrade receipt.
    with _exclusive_manifest_lock(Path(args.manifest)):
        if args.upgrade_only:
            _run_upgrade_only(args)
        else:
            _run_experiment(args)


if __name__ == "__main__":
    main()
