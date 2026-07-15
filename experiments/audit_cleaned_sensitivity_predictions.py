from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    OfficialSplit,
    TDC_TASK_META,
    infer_smiles_column,
)
from admet_benchmark.sensitivity import (  # noqa: E402
    CleaningPolicy,
    clean_official_split_for_sensitivity,
)
from admet_benchmark.smiles_compat import (  # noqa: E402
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)
from experiments.run_cleaned_sensitivity_cpu import (  # noqa: E402
    _featurize,
    _metric_values,
    _primary_effect,
)


PREDICTION_COLUMNS = [
    "task",
    "model",
    "seed",
    "split_source",
    "split_algorithm",
    "evaluation_set",
    "canonical_smiles",
    "label",
    "official_prediction",
    "cleaned_prediction",
]
REQUIRED_COLUMNS = set(PREDICTION_COLUMNS)
AUDIT_COLUMNS = [
    "task",
    "task_type",
    "official_metric",
    "model",
    "metric_replay_policy_id",
    "metric_prediction_dtype",
    "seed",
    "split_source",
    "split_algorithm",
    "prediction_path",
    "prediction_sha256",
    "prediction_bytes",
    "rows",
    "row_count_matches_run",
    "finite_labels_and_predictions",
    "classification_probability_range_ok",
    "canonical_smiles_complete",
    "canonical_smiles_unique",
    "source_common_test_exact",
    "official_metric_matches_run",
    "cleaned_metric_matches_run",
    "paired_delta_matches_run",
    "recomputed_official_common_test",
    "recomputed_cleaned_common_test",
    "sample_label_identity_sha256",
    "source_sample_label_identity_sha256",
]
METRIC_RTOL = 1e-10
METRIC_ATOL = 1e-12
METRIC_REPLAY_POLICY_ID = "model-native-output-dtype-replay-v1"
MODEL_PREDICTION_DTYPES = {
    "rf": "float64",
    "xgboost": "float32",
    "lightgbm": "float64",
}
_INTEGER_TEXT = re.compile(r"[+-]?(?:0|[1-9][0-9]*)")
_SHA256_TEXT = re.compile(r"[0-9a-f]{64}")
_TASK_NAME_BY_SLUG = {name.lower(): name for name in TDC_TASK_META}


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str

    @property
    def byte_count(self) -> int:
        return len(self.data)

    def csv(self) -> pd.DataFrame:
        # Parsing and hashing are deliberately derived from exactly the same
        # immutable byte object. Do not replace this with pd.read_csv(path).
        return pd.read_csv(io.BytesIO(self.data))

    def json(self) -> Any:
        return json.loads(self.data.decode("utf-8"))


def _snapshot(path: Path) -> FileSnapshot:
    data = path.read_bytes()
    return FileSnapshot(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _snapshot_record(snapshot: FileSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
    }


def _stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_replay_policy_record() -> dict[str, object]:
    """Describe the exact model-output dtype restored before metric replay.

    Prediction CSV parsing promotes numeric columns to float64. That promotion
    is lossless for the exported decimal text, but it does not reproduce the
    array passed to the runner's metric function for XGBoost, whose sklearn API
    returns float32 predictions. Recasting only the prediction arrays restores
    that model-output boundary; labels, stored values, and comparison
    tolerances are unchanged.
    """

    policy: dict[str, object] = {
        "policy_id": METRIC_REPLAY_POLICY_ID,
        "application": (
            "cast parsed official_prediction and cleaned_prediction arrays to "
            "the recorded model-output dtype immediately before metric "
            "recomputation"
        ),
        "prediction_dtype_by_model": dict(MODEL_PREDICTION_DTYPES),
        "labels_dtype": "float64",
        "csv_parser_numeric_dtype": "float64",
        "comparison_tolerances_unchanged": {
            "rtol": METRIC_RTOL,
            "atol": METRIC_ATOL,
        },
    }
    return {**policy, "sha256": _stable_json_sha256(policy)}


def _strict_int(value: object, label: str, *, positive: bool = False) -> int:
    """Parse an integer without silently truncating 1.5 to 1."""

    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not _INTEGER_TEXT.fullmatch(text):
            raise ValueError(f"{label} must be a canonical integer, got {value!r}")
        result = int(text)
    elif isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} must be an integer, got {value!r}")
        result = int(numeric)
    else:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    return result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric, got {value!r}")
    return result


def _normalized_unique(values: Iterable[object], label: str) -> list[str]:
    normalized = [str(value).strip().lower() for value in values]
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{label} must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate values")
    return normalized


def _sample_identity_hash(smiles: Iterable[str], labels: Iterable[float]) -> str:
    frame = pd.DataFrame(
        {
            "canonical_smiles": [str(value) for value in smiles],
            "label": np.asarray(list(labels), dtype=float),
        }
    )
    payload = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialize_audit_csv(audit: pd.DataFrame) -> bytes:
    return audit.to_csv(
        index=False,
        columns=AUDIT_COLUMNS,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def _finalize_audit_and_gate(
    audit: pd.DataFrame, gate: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    audit = audit.reindex(columns=AUDIT_COLUMNS)
    serialized = _serialize_audit_csv(audit)
    gate["audit_csv"] = {
        "serialization": "utf8-pandas-csv-lf-float17g-v1",
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "rows": len(audit),
        "columns": list(AUDIT_COLUMNS),
    }
    # main() writes these exact bytes. Keeping them in attrs avoids a second
    # serialization that could diverge from the hash recorded in the gate.
    audit.attrs["_serialized_csv_bytes"] = serialized
    return audit, gate


def _load_source_snapshots(
    data_dir: Path, canonical_tasks: list[str]
) -> tuple[
    dict[tuple[str, str], FileSnapshot],
    dict[tuple[str, str], pd.DataFrame],
    dict[str, dict[str, dict[str, object]]],
]:
    snapshots: dict[tuple[str, str], FileSnapshot] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    records: dict[str, dict[str, dict[str, object]]] = {}
    for task in canonical_tasks:
        records[task] = {}
        for filename in ("train_val.csv", "test.csv"):
            path = data_dir / "admet_group" / task.lower() / filename
            snapshot = _snapshot(path)
            frame = snapshot.csv()
            snapshots[(task.lower(), filename)] = snapshot
            frames[(task.lower(), filename)] = frame
            records[task][filename] = _snapshot_record(snapshot)
    return snapshots, frames, records


def _manifest_task_record(
    records: Mapping[str, Any], task: str
) -> Mapping[str, Any]:
    matches = [value for key, value in records.items() if str(key).lower() == task.lower()]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        return {}
    return matches[0]


def _config_locator_matches(manifest: Mapping[str, Any], config_path: Path) -> bool:
    legacy_path = str(manifest.get("config_path", "")).strip()
    if legacy_path and Path(legacy_path).expanduser().resolve() == config_path.resolve():
        return True
    logical = str(manifest.get("config_logical_path", "")).strip()
    if not logical:
        return False
    logical_path = Path(logical)
    if logical_path.is_absolute() or ".." in logical_path.parts:
        return False
    try:
        expected = config_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return False
    return logical_path.as_posix() == expected


def _parse_config_contract(snapshot: FileSnapshot) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - formal environment has PyYAML
        raise ValueError("PyYAML is required to verify the sensitivity config") from exc
    config = yaml.safe_load(snapshot.data.decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Sensitivity config must be a mapping")
    return config


def _validate_manifest_snapshot(
    *,
    manifest_snapshot: FileSnapshot,
    config_snapshot: FileSnapshot,
    source_snapshots: Mapping[tuple[str, str], FileSnapshot],
    source_frames: Mapping[tuple[str, str], pd.DataFrame],
    expected_tasks: list[str],
    expected_models: list[str],
    expected_seeds: list[int],
    expected_radius: int,
    expected_n_bits: int,
    expected_threads_per_model: int,
) -> dict[str, object]:
    manifest = manifest_snapshot.json()
    if not isinstance(manifest, Mapping):
        raise ValueError("Sensitivity manifest must be a JSON object")
    config = _parse_config_contract(config_snapshot)
    task_slugs = [task.lower() for task in expected_tasks]
    model_slugs = [model.lower() for model in expected_models]
    manifest_tasks = _normalized_unique(manifest.get("tasks", []), "manifest tasks")
    manifest_models = _normalized_unique(manifest.get("models", []), "manifest models")
    manifest_seeds = [
        _strict_int(value, "manifest seed", positive=True)
        for value in manifest.get("seeds", [])
    ]
    if len(manifest_seeds) != len(set(manifest_seeds)):
        raise ValueError("manifest seeds contain duplicates")

    config_tasks = _normalized_unique(
        config.get("selected_tasks_for_cleaned_rerun", []), "config tasks"
    )
    config_models = _normalized_unique(config.get("models", []), "config models")
    config_seeds = [
        _strict_int(value, "config seed", positive=True)
        for value in config.get("seeds", [])
    ]
    if len(config_seeds) != len(set(config_seeds)):
        raise ValueError("config seeds contain duplicates")
    fingerprint = config.get("fingerprint", {})
    if not isinstance(fingerprint, Mapping):
        raise ValueError("config fingerprint must be a mapping")
    config_radius = _strict_int(
        fingerprint.get("radius"), "config fingerprint radius", positive=True
    )
    config_n_bits = _strict_int(
        fingerprint.get("n_bits"), "config fingerprint n_bits", positive=True
    )

    manifest_fingerprint = manifest.get("fingerprint", {})
    if not isinstance(manifest_fingerprint, Mapping):
        manifest_fingerprint = {}
    try:
        manifest_radius = _strict_int(
            manifest_fingerprint.get("radius"),
            "manifest fingerprint radius",
            positive=True,
        )
        manifest_n_bits = _strict_int(
            manifest_fingerprint.get("n_bits"),
            "manifest fingerprint n_bits",
            positive=True,
        )
        fingerprint_matches = (
            manifest_radius == config_radius == expected_radius
            and manifest_n_bits == config_n_bits == expected_n_bits
        )
    except ValueError:
        fingerprint_matches = False

    input_hashes = manifest.get("input_sha256", {})
    if not isinstance(input_hashes, Mapping):
        input_hashes = {}
    input_task_slugs = [str(key).strip().lower() for key in input_hashes]
    source_hashes_match = bool(
        len(input_task_slugs) == len(set(input_task_slugs))
        and set(input_task_slugs) == set(task_slugs)
    )
    for task in expected_tasks:
        task_record = _manifest_task_record(input_hashes, task)
        source_hashes_match &= set(task_record) == {"train_val.csv", "test.csv"}
        for filename in ("train_val.csv", "test.csv"):
            recorded = str(task_record.get(filename, ""))
            source_hashes_match &= bool(
                _SHA256_TEXT.fullmatch(recorded)
                and source_snapshots[(task.lower(), filename)].sha256 == recorded
            )

    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    compatibility_occurrences: dict[str, int] = {}
    for task in expected_tasks:
        count = 0
        for filename in ("train_val.csv", "test.csv"):
            frame = source_frames[(task.lower(), filename)]
            smiles_column = infer_smiles_column(frame)
            count += int(
                frame[smiles_column]
                .astype(str)
                .isin(set(LEGACY_VALENCE_REPAIRS))
                .sum()
            )
        compatibility_occurrences[task] = count

    try:
        schema_v2 = _strict_int(manifest.get("schema_version"), "manifest schema") == 2
    except ValueError:
        schema_v2 = False
    try:
        manifest_threads_per_model: int | None = _strict_int(
            manifest.get("threads_per_model"),
            "manifest threads_per_model",
            positive=True,
        )
    except ValueError:
        manifest_threads_per_model = None
    try:
        raw_occurrences = manifest.get("compatibility_source_occurrences", {})
        if not isinstance(raw_occurrences, Mapping):
            raise ValueError("compatibility occurrences must be a mapping")
        normalized_occurrences: dict[str, int] = {}
        for raw_task, raw_count in raw_occurrences.items():
            slug = str(raw_task).strip().lower()
            if not slug or slug in normalized_occurrences:
                raise ValueError("compatibility occurrence task keys are invalid")
            count = _strict_int(
                raw_count, f"compatibility occurrence count for {raw_task}"
            )
            if count < 0:
                raise ValueError("compatibility occurrence count cannot be negative")
            normalized_occurrences[slug] = count
        expected_occurrences = {
            task.lower(): count for task, count in compatibility_occurrences.items()
        }
        compatibility_occurrences_match = (
            normalized_occurrences == expected_occurrences
        )
    except ValueError:
        compatibility_occurrences_match = False
    checks = {
        "schema_v2": schema_v2,
        "paired_protocol": manifest.get("protocol") == "paired-common-test-v1",
        "split_source": manifest.get("split_source") == LOCKED_CACHE_SPLIT_SOURCE,
        "split_algorithm": manifest.get("split_algorithm")
        == LOCKED_CACHE_SPLIT_ALGORITHM,
        "cpu_only": manifest.get("cpu_only") is True,
        "predictions_enabled": manifest.get("save_predictions") is True,
        "task_contract": manifest_tasks == task_slugs == config_tasks,
        "model_contract": manifest_models == model_slugs == config_models,
        "seed_contract": manifest_seeds == expected_seeds == config_seeds,
        "fingerprint_contract": fingerprint_matches,
        "threads_per_model_contract": manifest_threads_per_model
        == expected_threads_per_model,
        "cleaning_policy_contract": manifest.get("cleaning_policy")
        == CleaningPolicy().to_dict(),
        "canonicalization_input_policy_contract": manifest.get(
            "canonicalization_input_policy"
        )
        == "apply_enumerated_parser_compatibility_before_rdkit_canonicalization",
        "model_input_representation_policy_contract": manifest.get(
            "model_input_representation_policy"
        )
        == SMILES_COMPATIBILITY_POLICY,
        "parser_repairs_not_cleaning_deletions": manifest.get(
            "parser_compatibility_repairs_count_as_cleaning_deletions"
        )
        is False,
        "config_path_matches": _config_locator_matches(
            manifest, config_snapshot.path
        ),
        "config_sha256_recomputed": manifest.get("config_sha256")
        == config_snapshot.sha256,
        "source_file_hashes_recomputed": bool(source_hashes_match),
        "compatibility_mapping_hash_recomputed": manifest.get(
            "compatibility_mapping_sha256"
        )
        == _stable_json_sha256(compatibility_mapping),
        "compatibility_occurrences_recomputed": compatibility_occurrences_match,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Sensitivity manifest provenance gate failed: {failed}")
    return {
        "path": str(manifest_snapshot.path),
        "schema_version": 2,
        "split_source": LOCKED_CACHE_SPLIT_SOURCE,
        "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
        "config_logical_path": str(manifest.get("config_logical_path")),
        "checks": checks,
    }


def _reconstruct_common_test(
    *,
    task: str,
    seed: int,
    radius: int,
    n_bits: int,
    source_frames: Mapping[tuple[str, str], pd.DataFrame],
) -> tuple[list[str], np.ndarray, str, str]:
    from sklearn.model_selection import train_test_split

    meta = TDC_TASK_META[task]
    train_val = source_frames[(task.lower(), "train_val.csv")].copy(deep=True)
    test = source_frames[(task.lower(), "test.csv")].copy(deep=True)
    if "Y" not in train_val.columns:
        raise ValueError(f"{task} locked train_val.csv has no Y column")
    stratify = train_val["Y"] if meta["task_type"] == "classification" else None
    train, valid = train_test_split(
        train_val,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    official = OfficialSplit(
        name=task,
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        task_type=meta["task_type"],
        metric=meta["metric"],
        category=meta["category"],
    )
    cleaned = clean_official_split_for_sensitivity(official)
    _, labels, smiles = _featurize(cleaned.test, radius, n_bits)
    if not smiles or len(smiles) != len(labels):
        raise ValueError(f"{task} seed {seed} reconstructed an invalid common test")
    return smiles, np.asarray(labels, dtype=float), official.task_type, official.metric


def _inventory_record(
    *,
    filename: str,
    task: str,
    model: str,
    seed: int,
    snapshot: FileSnapshot,
    rows: int,
) -> dict[str, object]:
    return {
        "filename": filename,
        "task": task,
        "model": model,
        "seed": seed,
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
        "rows": rows,
    }


def _inventory_digest(records: Iterable[Mapping[str, object]]) -> str:
    ordered = sorted((dict(record) for record in records), key=lambda row: row["filename"])
    return _stable_json_sha256(ordered)


def recompute_prediction_inventory_sha256(
    audit: pd.DataFrame, predictions_dir: Path
) -> str:
    """Re-hash an audited inventory so downstream consumers can reject staleness."""

    records: list[dict[str, object]] = []
    for row in audit.sort_values(["task", "model", "seed"]).itertuples(index=False):
        seed = _strict_int(row.seed, "audit seed", positive=True)
        filename = Path(str(row.prediction_path)).name
        snapshot = _snapshot(predictions_dir / filename)
        frame = snapshot.csv()
        records.append(
            _inventory_record(
                filename=filename,
                task=str(row.task).lower(),
                model=str(row.model).lower(),
                seed=seed,
                snapshot=snapshot,
                rows=len(frame),
            )
        )
    return _inventory_digest(records)


def audit_prediction_exports(
    *,
    runs_path: Path,
    predictions_dir: Path,
    manifest_path: Path,
    config_path: Path,
    data_dir: Path,
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    _config_snapshot: FileSnapshot | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    failures: list[str] = []
    inputs: dict[str, object] = {"predictions_dir": str(predictions_dir)}
    audit_rows: list[dict[str, object]] = []
    inventory_records: list[dict[str, object]] = []
    manifest_gate: dict[str, object] | None = None

    try:
        task_slugs = _normalized_unique(tasks, "tasks")
        model_slugs = _normalized_unique(models, "models")
        unsupported_dtype_models = sorted(
            set(model_slugs).difference(MODEL_PREDICTION_DTYPES)
        )
        if unsupported_dtype_models:
            raise ValueError(
                "models have no metric replay dtype policy: "
                + ", ".join(unsupported_dtype_models)
            )
        strict_seeds = [
            _strict_int(value, "expected seed", positive=True) for value in seeds
        ]
        if len(strict_seeds) != len(set(strict_seeds)):
            raise ValueError("expected seeds contain duplicates")
        canonical_tasks = []
        for slug in task_slugs:
            if slug not in _TASK_NAME_BY_SLUG:
                raise ValueError(f"unknown TDC task: {slug}")
            canonical_tasks.append(_TASK_NAME_BY_SLUG[slug])
    except ValueError as exc:
        gate = {
            "schema_version": 2,
            "status": "HOLD",
            "failures": [f"invalid audit contract: {exc}"],
            "metric_replay_policy": _metric_replay_policy_record(),
            "inputs": inputs,
        }
        return _finalize_audit_and_gate(pd.DataFrame(), gate)

    missing_inputs = []
    for label, path, kind in (
        ("runs", runs_path, "file"),
        ("manifest", manifest_path, "file"),
        ("config", config_path, "file"),
        ("predictions directory", predictions_dir, "directory"),
    ):
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            missing_inputs.append(f"missing {label}: {path}")
    if missing_inputs:
        gate = {
            "schema_version": 2,
            "status": "HOLD",
            "failures": missing_inputs,
            "metric_replay_policy": _metric_replay_policy_record(),
            "inputs": inputs,
        }
        return _finalize_audit_and_gate(pd.DataFrame(), gate)

    try:
        runs_snapshot = _snapshot(runs_path)
        manifest_snapshot = _snapshot(manifest_path)
        config_snapshot = _config_snapshot or _snapshot(config_path)
        if config_snapshot.path.resolve() != config_path.resolve():
            raise ValueError("provided config snapshot path does not match config_path")
        runs = runs_snapshot.csv()
        inputs["runs"] = _snapshot_record(runs_snapshot)
        inputs["manifest"] = _snapshot_record(manifest_snapshot)
        inputs["config"] = _snapshot_record(config_snapshot)
    except Exception as exc:
        gate = {
            "schema_version": 2,
            "status": "HOLD",
            "failures": [f"cannot snapshot/parse core input: {exc}"],
            "metric_replay_policy": _metric_replay_policy_record(),
            "inputs": inputs,
        }
        return _finalize_audit_and_gate(pd.DataFrame(), gate)

    try:
        source_snapshots, source_frames, source_records = _load_source_snapshots(
            data_dir, canonical_tasks
        )
        inputs["locked_sources"] = source_records
    except Exception as exc:
        source_snapshots = {}
        source_frames = {}
        failures.append(f"cannot snapshot/parse locked sources: {exc}")

    run_required = {
        "task",
        "task_type",
        "official_metric",
        "model",
        "seed",
        "split_source",
        "split_algorithm",
        "radius",
        "n_bits",
        "threads_per_model",
        "common_test_n",
        "primary_official_common_test",
        "primary_cleaned_common_test",
        "primary_raw_delta_cleaned_minus_official",
        "primary_oriented_delta",
        "primary_relative_oriented_delta",
    }
    missing_run_columns = sorted(run_required.difference(runs.columns))
    run_lookup: dict[tuple[str, str, int], Mapping[str, object]] = {}
    run_radius: int | None = None
    run_n_bits: int | None = None
    run_threads_per_model: int | None = None
    if missing_run_columns:
        failures.append(f"run table missing columns: {missing_run_columns}")
    else:
        for row_number, record in enumerate(runs.to_dict("records"), start=2):
            try:
                task = str(record["task"]).strip().lower()
                model = str(record["model"]).strip().lower()
                seed = _strict_int(record["seed"], f"run row {row_number} seed", positive=True)
                common_test_n = _strict_int(
                    record["common_test_n"],
                    f"run row {row_number} common_test_n",
                    positive=True,
                )
                radius = _strict_int(
                    record["radius"], f"run row {row_number} radius", positive=True
                )
                n_bits = _strict_int(
                    record["n_bits"], f"run row {row_number} n_bits", positive=True
                )
                threads_per_model = _strict_int(
                    record["threads_per_model"],
                    f"run row {row_number} threads_per_model",
                    positive=True,
                )
                if task not in _TASK_NAME_BY_SLUG:
                    raise ValueError(f"unknown task {task!r}")
                meta = TDC_TASK_META[_TASK_NAME_BY_SLUG[task]]
                metric = str(record["official_metric"]).strip().lower()
                if (
                    str(record["task_type"]).strip().lower() != meta["task_type"]
                    or metric != meta["metric"]
                ):
                    raise ValueError("task_type/official_metric metadata mismatch")
                if record["split_source"] != LOCKED_CACHE_SPLIT_SOURCE:
                    raise ValueError("split_source mismatch")
                if record["split_algorithm"] != LOCKED_CACHE_SPLIT_ALGORITHM:
                    raise ValueError("split_algorithm mismatch")
                for column in (
                    "primary_official_common_test",
                    "primary_cleaned_common_test",
                    "primary_raw_delta_cleaned_minus_official",
                    "primary_oriented_delta",
                    "primary_relative_oriented_delta",
                ):
                    _finite_float(record[column], f"run row {row_number} {column}")
                for column in (
                    f"official_common_test_{metric}",
                    f"cleaned_common_test_{metric}",
                ):
                    if column not in runs.columns:
                        raise ValueError(f"missing primary metric column {column}")
                    _finite_float(record[column], f"run row {row_number} {column}")
                key = (task, model, seed)
                if key in run_lookup:
                    raise ValueError(f"duplicate task/model/seed key {key}")
                normalized = dict(record)
                normalized["common_test_n"] = common_test_n
                normalized["radius"] = radius
                normalized["n_bits"] = n_bits
                normalized["threads_per_model"] = threads_per_model
                run_lookup[key] = normalized
                run_radius = radius if run_radius is None else run_radius
                run_n_bits = n_bits if run_n_bits is None else run_n_bits
                run_threads_per_model = (
                    threads_per_model
                    if run_threads_per_model is None
                    else run_threads_per_model
                )
                if (
                    radius != run_radius
                    or n_bits != run_n_bits
                    or threads_per_model != run_threads_per_model
                ):
                    raise ValueError(
                        "fingerprint/thread parameters vary across run rows"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"invalid run row {row_number}: {exc}")

    expected_keys = {
        (task, model, seed)
        for task in task_slugs
        for model in model_slugs
        for seed in strict_seeds
    }
    if set(run_lookup) != expected_keys:
        failures.append(
            "run table key matrix mismatch: "
            f"missing={len(expected_keys - set(run_lookup))}, "
            f"unexpected={len(set(run_lookup) - expected_keys)}"
        )

    if (
        run_radius is not None
        and run_n_bits is not None
        and run_threads_per_model is not None
        and source_snapshots
    ):
        try:
            manifest_gate = _validate_manifest_snapshot(
                manifest_snapshot=manifest_snapshot,
                config_snapshot=config_snapshot,
                source_snapshots=source_snapshots,
                source_frames=source_frames,
                expected_tasks=canonical_tasks,
                expected_models=model_slugs,
                expected_seeds=strict_seeds,
                expected_radius=run_radius,
                expected_n_bits=run_n_bits,
                expected_threads_per_model=run_threads_per_model,
            )
        except Exception as exc:
            failures.append(f"manifest provenance: {exc}")
    else:
        failures.append("manifest provenance cannot be checked without valid runs/sources")

    expected_files = {
        f"{task}_{model}_seed{seed}_paired.csv": (task, model, seed)
        for task in task_slugs
        for model in model_slugs
        for seed in strict_seeds
    }
    ignored_sidecar_names = sorted(
        path.name
        for path in predictions_dir.iterdir()
        if path.is_file()
        and path.name.startswith("._")
        and path.suffix.lower() == ".csv"
    )
    ignored_sidecars = {
        "policy": "ignore-appledouble-basename-prefix-dot-underscore-v1",
        "count": len(ignored_sidecar_names),
        "names_sha256": _stable_json_sha256(ignored_sidecar_names),
    }
    actual_files = {
        path.name: path
        for path in predictions_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and not path.name.startswith("._")
    }
    missing_files = sorted(set(expected_files).difference(actual_files))
    unexpected_files = sorted(set(actual_files).difference(expected_files))
    if missing_files or unexpected_files:
        failures.append(
            "prediction file matrix mismatch: "
            f"missing={len(missing_files)}, unexpected={len(unexpected_files)}"
        )

    reconstructed: dict[tuple[str, int], tuple[list[str], np.ndarray, str, str]] = {}
    if source_frames and run_radius is not None and run_n_bits is not None:
        for task_slug, seed in sorted(
            {(task, seed) for task, _, seed in expected_keys}
        ):
            try:
                reconstructed[(task_slug, seed)] = _reconstruct_common_test(
                    task=_TASK_NAME_BY_SLUG[task_slug],
                    seed=seed,
                    radius=run_radius,
                    n_bits=run_n_bits,
                    source_frames=source_frames,
                )
            except Exception as exc:
                failures.append(
                    f"cannot reconstruct {task_slug} seed {seed} common test: {exc}"
                )

    task_hashes: dict[str, set[str]] = {task: set() for task in task_slugs}
    all_source_exact = True
    all_metrics_match = True
    for filename, key in sorted(expected_files.items()):
        path = actual_files.get(filename)
        if path is None:
            continue
        task, model, seed = key
        try:
            snapshot = _snapshot(path)
            frame = snapshot.csv()
        except Exception as exc:
            failures.append(f"cannot snapshot/parse {filename}: {exc}")
            continue
        if list(frame.columns) != PREDICTION_COLUMNS:
            failures.append(
                f"{filename} prediction schema/order mismatch: {list(frame.columns)}"
            )
            continue
        try:
            parsed_seeds = [
                _strict_int(value, f"{filename} seed", positive=True)
                for value in frame["seed"].tolist()
            ]
        except ValueError as exc:
            parsed_seeds = []
            failures.append(str(exc))
        singleton_checks = {
            "task": {str(value).strip().lower() for value in frame["task"]} == {task},
            "model": {str(value).strip().lower() for value in frame["model"]} == {model},
            "seed": bool(parsed_seeds) and set(parsed_seeds) == {seed},
            "split_source": set(frame["split_source"].astype(str))
            == {LOCKED_CACHE_SPLIT_SOURCE},
            "split_algorithm": set(frame["split_algorithm"].astype(str))
            == {LOCKED_CACHE_SPLIT_ALGORITHM},
            "evaluation_set": set(frame["evaluation_set"].astype(str))
            == {"common_cleaned_test"},
        }
        for name, passed in singleton_checks.items():
            if not passed:
                failures.append(f"{filename} {name} identity mismatch")

        numeric = frame[["label", "official_prediction", "cleaned_prediction"]].apply(
            pd.to_numeric, errors="coerce"
        )
        finite = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
        if not finite:
            failures.append(f"{filename} contains non-finite labels or predictions")
        smiles = frame["canonical_smiles"].fillna("").astype(str).str.strip()
        smiles_complete = bool((smiles != "").all())
        smiles_unique = not bool(smiles.duplicated().any())
        if not smiles_complete:
            failures.append(f"{filename} contains empty canonical SMILES")
        if not smiles_unique:
            failures.append(f"{filename} contains duplicate canonical SMILES")

        run = run_lookup.get(key)
        expected_rows = int(run["common_test_n"]) if run is not None else -1
        row_count_matches = len(frame) == expected_rows
        if not row_count_matches:
            failures.append(
                f"{filename} row count {len(frame)} does not match run table {expected_rows}"
            )
        expected = reconstructed.get((task, seed))
        source_exact = False
        source_identity_hash = ""
        task_type = (
            str(run["task_type"]).strip().lower() if run is not None else "unknown"
        )
        official_metric = (
            str(run["official_metric"]).strip().lower()
            if run is not None
            else "unknown"
        )
        if expected is not None and finite:
            expected_smiles, expected_labels, expected_type, expected_metric = expected
            source_identity_hash = _sample_identity_hash(
                expected_smiles, expected_labels
            )
            source_exact = bool(
                smiles.tolist() == expected_smiles
                and np.array_equal(
                    numeric["label"].to_numpy(dtype=float), expected_labels
                )
                and len(frame) == len(expected_labels)
                and task_type == expected_type
                and official_metric == expected_metric
            )
        if not source_exact:
            failures.append(
                f"{filename} source common-test samples/labels/order mismatch"
            )
            all_source_exact = False

        probability_range_ok = True
        if finite and task_type == "classification":
            prediction_values = numeric[
                ["official_prediction", "cleaned_prediction"]
            ].to_numpy(dtype=float)
            probability_range_ok = bool(
                (prediction_values >= -1e-12).all()
                and (prediction_values <= 1.0 + 1e-12).all()
            )
            if not probability_range_ok:
                failures.append(f"{filename} classification predictions leave [0,1]")

        recomputed_official = float("nan")
        recomputed_cleaned = float("nan")
        official_metric_matches = False
        cleaned_metric_matches = False
        delta_matches = False
        metric_prediction_dtype = MODEL_PREDICTION_DTYPES[model]
        if finite and run is not None and official_metric in {
            "auroc",
            "auprc",
            "mae",
            "spearman",
        }:
            y_true = numeric["label"].to_numpy(dtype=float)
            # Replay the dtype of the estimator output that the runner passed
            # to _metric_values. In particular, XGBoost emits float32 arrays,
            # while CSV parsing promotes the exported decimal text to float64.
            # Casting here restores the original metric input without changing
            # the gate tolerances or mutating the prediction artifact.
            prediction_dtype = np.dtype(metric_prediction_dtype)
            official_prediction = numeric["official_prediction"].to_numpy(
                dtype=prediction_dtype
            )
            cleaned_prediction = numeric["cleaned_prediction"].to_numpy(
                dtype=prediction_dtype
            )
            metric_official = _metric_values(
                task_type, y_true, official_prediction
            ).get(official_metric, float("nan"))
            metric_cleaned = _metric_values(
                task_type, y_true, cleaned_prediction
            ).get(official_metric, float("nan"))
            if np.isfinite([metric_official, metric_cleaned]).all():
                recomputed_official = float(metric_official)
                recomputed_cleaned = float(metric_cleaned)
                raw_delta, oriented_delta, relative_delta = _primary_effect(
                    official_metric, recomputed_official, recomputed_cleaned
                )
                try:
                    observed_official = [
                        _finite_float(
                            run["primary_official_common_test"],
                            f"{filename} run primary official",
                        ),
                        _finite_float(
                            run[f"official_common_test_{official_metric}"],
                            f"{filename} run metric-specific official",
                        ),
                    ]
                    observed_cleaned = [
                        _finite_float(
                            run["primary_cleaned_common_test"],
                            f"{filename} run primary cleaned",
                        ),
                        _finite_float(
                            run[f"cleaned_common_test_{official_metric}"],
                            f"{filename} run metric-specific cleaned",
                        ),
                    ]
                    official_metric_matches = all(
                        np.isclose(
                            value,
                            recomputed_official,
                            rtol=METRIC_RTOL,
                            atol=METRIC_ATOL,
                        )
                        for value in observed_official
                    )
                    cleaned_metric_matches = all(
                        np.isclose(
                            value,
                            recomputed_cleaned,
                            rtol=METRIC_RTOL,
                            atol=METRIC_ATOL,
                        )
                        for value in observed_cleaned
                    )
                    delta_matches = all(
                        np.isclose(
                            observed,
                            expected_value,
                            rtol=METRIC_RTOL,
                            atol=METRIC_ATOL,
                        )
                        for observed, expected_value in (
                            (
                                _finite_float(
                                    run[
                                        "primary_raw_delta_cleaned_minus_official"
                                    ],
                                    f"{filename} run raw delta",
                                ),
                                raw_delta,
                            ),
                            (
                                _finite_float(
                                    run["primary_oriented_delta"],
                                    f"{filename} run oriented delta",
                                ),
                                oriented_delta,
                            ),
                            (
                                _finite_float(
                                    run["primary_relative_oriented_delta"],
                                    f"{filename} run relative delta",
                                ),
                                relative_delta,
                            ),
                        )
                    )
                except (KeyError, ValueError):
                    official_metric_matches = False
                    cleaned_metric_matches = False
                    delta_matches = False
        if not official_metric_matches:
            failures.append(f"{filename} official metric does not match run table")
        if not cleaned_metric_matches:
            failures.append(f"{filename} cleaned metric does not match run table")
        if not delta_matches:
            failures.append(f"{filename} paired delta algebra does not match run table")
        all_metrics_match &= bool(
            official_metric_matches and cleaned_metric_matches and delta_matches
        )

        labels_for_hash = numeric["label"].to_numpy(dtype=float)
        identity_hash = (
            _sample_identity_hash(smiles.tolist(), labels_for_hash)
            if finite and smiles_complete
            else ""
        )
        if identity_hash:
            task_hashes[task].add(identity_hash)
        inventory_records.append(
            _inventory_record(
                filename=filename,
                task=task,
                model=model,
                seed=seed,
                snapshot=snapshot,
                rows=len(frame),
            )
        )
        audit_rows.append(
            {
                "task": task,
                "task_type": task_type,
                "official_metric": official_metric,
                "model": model,
                "metric_replay_policy_id": METRIC_REPLAY_POLICY_ID,
                "metric_prediction_dtype": metric_prediction_dtype,
                "seed": seed,
                "split_source": LOCKED_CACHE_SPLIT_SOURCE,
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "prediction_path": str(path),
                "prediction_sha256": snapshot.sha256,
                "prediction_bytes": snapshot.byte_count,
                "rows": len(frame),
                "row_count_matches_run": row_count_matches,
                "finite_labels_and_predictions": finite,
                "classification_probability_range_ok": probability_range_ok,
                "canonical_smiles_complete": smiles_complete,
                "canonical_smiles_unique": smiles_unique,
                "source_common_test_exact": source_exact,
                "official_metric_matches_run": official_metric_matches,
                "cleaned_metric_matches_run": cleaned_metric_matches,
                "paired_delta_matches_run": delta_matches,
                "recomputed_official_common_test": recomputed_official,
                "recomputed_cleaned_common_test": recomputed_cleaned,
                "sample_label_identity_sha256": identity_hash,
                "source_sample_label_identity_sha256": source_identity_hash,
            }
        )

    inconsistent_tasks = sorted(
        task for task, hashes in task_hashes.items() if len(hashes) != 1
    )
    if inconsistent_tasks:
        failures.append(
            "cross-model/seed sample-label identity mismatch: "
            + ", ".join(inconsistent_tasks)
        )
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    expected_count = len(expected_keys)
    if len(audit) != expected_count:
        failures.append(
            f"prediction audit has {len(audit)} rows, expected {expected_count}"
        )
    inventory_sha256 = _inventory_digest(inventory_records)
    inputs["predictions"] = {
        "algorithm": "sha256-stable-json-inventory-v1",
        "sha256": inventory_sha256,
        "files": len(inventory_records),
        "bytes": int(sum(row["bytes"] for row in inventory_records)),
        "rows": int(sum(row["rows"] for row in inventory_records)),
        "entries": {
            str(row["filename"]): {
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "rows": row["rows"],
            }
            for row in sorted(inventory_records, key=lambda value: value["filename"])
        },
    }
    gate: dict[str, object] = {
        "schema_version": 2,
        "status": "GO" if not failures else "HOLD",
        "complete_expected_matrix": not missing_files
        and not unexpected_files
        and len(audit) == expected_count,
        "expected_prediction_files": expected_count,
        "observed_prediction_files": len(actual_files),
        "audited_prediction_files": len(audit),
        "sample_label_identity_aligned_within_task": not inconsistent_tasks,
        "source_common_test_reconstructed_exactly": bool(
            all_source_exact and len(audit) == expected_count
        ),
        "prediction_metrics_recomputed_match_runs": bool(
            all_metrics_match and len(audit) == expected_count
        ),
        "unique_sample_label_hashes_by_task": {
            task: len(hashes) for task, hashes in sorted(task_hashes.items())
        },
        "prediction_inventory": dict(inputs["predictions"]),
        "prediction_inventory_sha256": inventory_sha256,
        "ignored_appledouble_sidecars": ignored_sidecars,
        "metric_replay_policy": _metric_replay_policy_record(),
        "manifest_provenance_gate": manifest_gate,
        "failures": failures,
        "inputs": inputs,
    }
    return _finalize_audit_and_gate(audit, gate)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all paired cleaned-sensitivity prediction exports."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "results/tables/cleaned_sensitivity_locked_cache_runs.csv",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=ROOT / "results/predictions/cleaned_sensitivity_locked_cache",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/tables/cleaned_sensitivity_locked_cache_manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/config/sensitivity.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/tdc_official",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/tables/cleaned_sensitivity_locked_cache",
    )
    args = parser.parse_args()
    # The CLI snapshots the config once, then derives both its contract and its
    # provenance hash from that byte object. audit_prediction_exports accepts
    # the same snapshot so the CLI never reparses a mutable path.
    config_snapshot = _snapshot(args.config)
    config = _parse_config_contract(config_snapshot)
    tasks = list(config.get("selected_tasks_for_cleaned_rerun", []))
    models = list(config.get("models", []))
    seeds = [
        _strict_int(value, "config seed", positive=True)
        for value in config.get("seeds", [])
    ]
    audit, gate = audit_prediction_exports(
        runs_path=args.runs,
        predictions_dir=args.predictions_dir,
        manifest_path=args.manifest,
        config_path=args.config,
        data_dir=args.data_dir,
        tasks=tasks,
        models=models,
        seeds=seeds,
        _config_snapshot=config_snapshot,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "cleaned_sensitivity_prediction_audit.csv"
    gate_path = args.out_dir / "cleaned_sensitivity_prediction_gate.json"
    audit_bytes = audit.attrs.get("_serialized_csv_bytes")
    if not isinstance(audit_bytes, bytes):  # pragma: no cover - defensive only
        raise RuntimeError("audit CSV was not serialized by the audit function")
    gate_audit = gate.get("audit_csv")
    if not isinstance(gate_audit, dict):  # pragma: no cover - defensive only
        raise RuntimeError("audit CSV binding is missing from the gate")
    gate_audit["path"] = str(audit_path)
    _atomic_write_bytes(audit_path, audit_bytes)
    gate_bytes = (
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(gate_path, gate_bytes)
    print(gate_bytes.decode("utf-8"), end="")
    raise SystemExit(0 if gate["status"] == "GO" else 1)


if __name__ == "__main__":
    main()
