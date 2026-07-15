from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    TDC_TASK_META,
    infer_smiles_column,
)
from admet_benchmark.smiles_compat import (  # noqa: E402
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)
from admet_benchmark.sensitivity import CleaningPolicy  # noqa: E402
DEFAULT_TASKS = [
    "Caco2_Wang",
    "BBB_Martins",
    "PPBR_AZ",
    "Solubility_AqSolDB",
    "CYP2C9_Veith",
    "DILI",
    "AMES",
    "LD50_Zhu",
]
DEFAULT_MODELS = ["rf", "xgboost", "lightgbm"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
THRESHOLDS = {"auroc": 0.02, "auprc": 0.02, "spearman": 0.03, "mae": 0.05}


def _strict_int(value: object, label: str) -> int:
    """Parse an integer without silently truncating floats or booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer, got {value!r}")
    return int(numeric)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _logical_config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


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
) -> dict[str, int] | None:
    legacy = set(LEGACY_VALENCE_REPAIRS)
    counts: dict[str, int] = {}
    try:
        for task in tasks:
            total = 0
            task_dir = data_dir / "admet_group" / task.lower()
            for filename in ("train_val.csv", "test.csv"):
                frame = pd.read_csv(task_dir / filename)
                smiles_column = infer_smiles_column(frame)
                total += int(frame[smiles_column].astype(str).isin(legacy).sum())
            counts[task] = total
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return counts


def validate_manifest(
    path: Path,
    *,
    expected_split_source: str,
    expected_tasks: list[str],
    expected_models: list[str],
    expected_seeds: list[int],
    config_path: Path,
    data_dir: Path,
    manifest_raw: bytes | None = None,
    config_raw: bytes | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Sensitivity manifest is missing: {path}")
    if manifest_raw is None:
        manifest_raw = path.read_bytes()
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Sensitivity manifest is invalid JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Sensitivity manifest must contain one JSON object")
    expected_algorithm = (
        LOCKED_CACHE_SPLIT_ALGORITHM
        if expected_split_source == LOCKED_CACHE_SPLIT_SOURCE
        else "pytdc_default_task_specific_runtime"
    )
    try:
        schema_version = _strict_int(manifest.get("schema_version"), "manifest.schema_version")
        manifest_seeds = [
            _strict_int(value, f"manifest.seeds[{index}]")
            for index, value in enumerate(manifest.get("seeds", []))
        ]
        threads_per_model = _strict_int(
            manifest.get("threads_per_model"), "manifest.threads_per_model"
        )
        fingerprint = manifest.get("fingerprint", {})
        radius = _strict_int(fingerprint.get("radius"), "manifest.fingerprint.radius")
        n_bits = _strict_int(fingerprint.get("n_bits"), "manifest.fingerprint.n_bits")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Sensitivity manifest integer contract failed: {exc}") from exc
    manifest_tasks = [str(value).strip() for value in manifest.get("tasks", [])]
    manifest_models = [str(value).strip() for value in manifest.get("models", [])]
    expected_task_keys = {str(value).lower() for value in expected_tasks}
    expected_model_keys = {str(value).lower() for value in expected_models}
    checks = {
        "schema_v2": schema_version == 2,
        "paired_protocol": manifest.get("protocol") == "paired-common-test-v1",
        "split_source": manifest.get("split_source") == expected_split_source,
        "split_algorithm": manifest.get("split_algorithm") == expected_algorithm,
        "cpu_only": manifest.get("cpu_only") is True,
        "task_contract": (
            {value.lower() for value in manifest_tasks} == expected_task_keys
            and len(manifest_tasks) == len(expected_tasks)
            and len({value.lower() for value in manifest_tasks}) == len(manifest_tasks)
        ),
        "model_contract": (
            {value.lower() for value in manifest_models} == expected_model_keys
            and len(manifest_models) == len(expected_models)
            and len({value.lower() for value in manifest_models}) == len(manifest_models)
        ),
        "seed_contract": (
            set(manifest_seeds) == set(expected_seeds)
            and len(manifest_seeds) == len(expected_seeds)
            and len(set(manifest_seeds)) == len(manifest_seeds)
        ),
        "single_thread_cpu_contract": threads_per_model == 1,
        "fingerprint_contract": radius == 2 and n_bits == 2048,
        "predictions_saved": manifest.get("save_predictions") is True,
        "cleaning_policy_contract": manifest.get("cleaning_policy")
        == CleaningPolicy().to_dict(),
        "canonicalization_input_policy": manifest.get("canonicalization_input_policy")
        == "apply_enumerated_parser_compatibility_before_rdkit_canonicalization",
        "model_input_representation_policy": manifest.get(
            "model_input_representation_policy"
        )
        == SMILES_COMPATIBILITY_POLICY,
        "parser_repairs_not_cleaning_deletions": manifest.get(
            "parser_compatibility_repairs_count_as_cleaning_deletions"
        )
        is False,
        "compatibility_mapping_hashed": bool(
            re.fullmatch(
                r"[0-9a-f]{64}", str(manifest.get("compatibility_mapping_sha256", ""))
            )
        ),
    }
    if not config_path.is_file():
        raise ValueError(f"Sensitivity config is missing: {config_path}")
    if config_raw is None:
        config_raw = config_path.read_bytes()
    checks["config_logical_path_matches"] = (
        manifest.get("config_logical_path") == _logical_config_path(config_path)
    )
    checks["config_sha256_recomputed"] = (
        manifest.get("config_sha256") == _sha256_bytes(config_raw)
    )
    execution_config_sha256 = str(manifest.get("execution_config_sha256", ""))
    corrections = manifest.get("config_metadata_corrections", [])
    expected_correction = None
    if isinstance(corrections, list) and len(corrections) == 1:
        expected_correction = corrections[0]
    checks["execution_config_hash_preserved"] = bool(
        re.fullmatch(r"[0-9a-f]{64}", execution_config_sha256)
    )
    checks["noncomputational_config_correction_transparent"] = bool(
        isinstance(expected_correction, dict)
        and expected_correction.get("field") == "audit.scaffold_overlap"
        and expected_correction.get("old_value") is True
        and expected_correction.get("new_value") is False
        and expected_correction.get("computational_effect") is False
        and expected_correction.get("execution_config_sha256")
        == execution_config_sha256
        and expected_correction.get("corrected_config_sha256")
        == _sha256_bytes(config_raw)
        and isinstance(expected_correction.get("reason"), str)
        and bool(str(expected_correction.get("reason")).strip())
    )
    input_hashes = manifest.get("input_sha256", {})
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    source_checks: list[bool] = []
    observed_occurrences: dict[str, int] = {}
    source_inventory: list[dict[str, object]] = []
    legacy = set(LEGACY_VALENCE_REPAIRS)
    for task in expected_tasks:
        task_occurrences = 0
        for name in ("train_val.csv", "test.csv"):
            source = data_dir / "admet_group" / task.lower() / name
            recorded = str(input_hashes.get(task, {}).get(name, ""))
            if not source.is_file():
                source_checks.append(False)
                continue
            raw = source.read_bytes()
            actual_sha = _sha256_bytes(raw)
            source_checks.append(
                bool(re.fullmatch(r"[0-9a-f]{64}", recorded) and actual_sha == recorded)
            )
            try:
                frame = pd.read_csv(io.BytesIO(raw))
                smiles_column = infer_smiles_column(frame)
            except (KeyError, ValueError, pd.errors.ParserError) as exc:
                raise ValueError(f"Cannot parse locked source {source}: {exc}") from exc
            task_occurrences += int(frame[smiles_column].astype(str).isin(legacy).sum())
            source_inventory.append(
                {
                    "task": task,
                    "name": name,
                    "bytes": len(raw),
                    "sha256": actual_sha,
                    "rows": int(len(frame)),
                }
            )
        observed_occurrences[task] = task_occurrences
    checks["source_task_keys_exact"] = set(input_hashes) == set(expected_tasks)
    checks["source_file_hashes_recomputed"] = all(source_checks)
    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    checks["compatibility_mapping_hash_recomputed"] = (
        manifest.get("compatibility_mapping_sha256")
        == _stable_json_sha256(compatibility_mapping)
    )
    recorded_occurrences = manifest.get("compatibility_source_occurrences", {})
    normalized_occurrences: dict[str, int] = {}
    occurrence_contract_valid = isinstance(recorded_occurrences, dict)
    if occurrence_contract_valid:
        try:
            normalized_occurrences = {
                str(task): _strict_int(value, f"compatibility_source_occurrences.{task}")
                for task, value in recorded_occurrences.items()
            }
        except ValueError:
            occurrence_contract_valid = False
    checks["compatibility_occurrences_recomputed"] = bool(
        occurrence_contract_valid
        and set(normalized_occurrences) == set(expected_tasks)
        and normalized_occurrences == observed_occurrences
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Sensitivity manifest provenance gate failed: {failed}")
    return {
        "path": str(path),
        "schema_version": schema_version,
        "split_source": manifest["split_source"],
        "split_algorithm": manifest["split_algorithm"],
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "manifest_bytes": len(manifest_raw),
        "config_sha256": _sha256_bytes(config_raw),
        "execution_config_sha256": execution_config_sha256,
        "config_metadata_corrections": corrections,
        "config_logical_path": _logical_config_path(config_path),
        "data_logical_path": _logical_config_path(data_dir),
        "source_inventory": source_inventory,
        "compatibility_source_occurrences": observed_occurrences,
        "checks": checks,
    }


def validate_audit(
    path: Path,
    *,
    expected_tasks: list[str],
    expected_seeds: list[int],
    expected_compatibility_occurrences: dict[str, int],
    runs: pd.DataFrame | None = None,
    audit_raw: bytes | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Sensitivity audit JSONL is missing: {path}")
    if audit_raw is None:
        audit_raw = path.read_bytes()
    try:
        audit_text = audit_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Sensitivity audit JSONL is not valid UTF-8") from exc
    payloads: list[dict] = []
    for line_number, line in enumerate(audit_text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Sensitivity audit line {line_number} is blank")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Sensitivity audit line {line_number} is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Sensitivity audit line {line_number} is not a JSON object")
        payloads.append(payload)
    expected_keys = {
        (task.lower(), _strict_int(seed, "expected seed"))
        for task in expected_tasks
        for seed in expected_seeds
    }
    failures: list[str] = []
    observed_keys: list[tuple[str, int]] = []
    for payload in payloads:
        try:
            seed = _strict_int(payload.get("seed"), "audit.seed")
        except ValueError as exc:
            failures.append(str(exc))
            seed = -1
        observed_keys.append((str(payload.get("task", "")).lower(), seed))
    if len(observed_keys) != len(set(observed_keys)):
        failures.append("duplicate task/seed audit payloads")
    if set(observed_keys) != expected_keys:
        failures.append(
            "audit key matrix mismatch: "
            f"missing={len(expected_keys - set(observed_keys))}, "
            f"unexpected={len(set(observed_keys) - expected_keys)}"
        )
    for payload in payloads:
        task = str(payload.get("task", ""))
        try:
            seed = _strict_int(payload.get("seed"), f"{task}.seed")
            schema_version = _strict_int(
                payload.get("audit_schema_version"), f"{task} seed {seed}.audit_schema_version"
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if schema_version != 2:
            failures.append(f"{task} seed {payload.get('seed')} audit schema is not v2")
        if payload.get("protocol") != "paired-common-test-v1":
            failures.append(f"{task} seed {payload.get('seed')} protocol mismatch")
        if payload.get("model_input_representation_policy") != SMILES_COMPATIBILITY_POLICY:
            failures.append(f"{task} seed {payload.get('seed')} representation policy mismatch")
        if payload.get("parser_compatibility_repairs_count_as_cleaning_deletions") is not False:
            failures.append(f"{task} seed {payload.get('seed')} parser-deletion policy mismatch")
        after = payload.get("canonical_molecule_overlap_after_precedence", {})
        overlap_valid = set(after) == {"train_test", "train_valid", "valid_test"}
        if overlap_valid:
            try:
                overlap_valid = all(
                    _strict_int(value, f"{task} seed {seed}.overlap.{name}") == 0
                    for name, value in after.items()
                )
            except ValueError as exc:
                failures.append(str(exc))
                overlap_valid = False
        if not overlap_valid:
            failures.append(f"{task} seed {payload.get('seed')} retains canonical overlap")
        splits = payload.get("splits", {})
        if set(splits) != {"train", "valid", "test"}:
            failures.append(f"{task} seed {payload.get('seed')} split audit is incomplete")
            continue
        repairs = 0
        for split_name, report in splits.items():
            if report.get("split") != split_name:
                failures.append(f"{task} seed {payload.get('seed')} split identity mismatch")
            if report.get("parser_compatibility_repairs_count_as_cleaning_deletions") is not False:
                failures.append(f"{task} seed {payload.get('seed')} split parser policy mismatch")
            try:
                input_rows = _strict_int(
                    report["input_rows"], f"{task} seed {seed}.{split_name}.input_rows"
                )
                output_rows = _strict_int(
                    report["output_rows"], f"{task} seed {seed}.{split_name}.output_rows"
                )
                repairs += _strict_int(
                    report["parser_compatibility_repairs_applied"],
                    f"{task} seed {seed}.{split_name}.parser_repairs",
                )
            except (KeyError, TypeError, ValueError):
                failures.append(f"{task} seed {payload.get('seed')} split counts invalid")
                continue
            if input_rows <= 0 or output_rows <= 0 or output_rows > input_rows:
                failures.append(f"{task} seed {payload.get('seed')} split row counts invalid")
        try:
            expected_repairs = _strict_int(
                expected_compatibility_occurrences.get(task),
                f"expected compatibility occurrences for {task}",
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if repairs != expected_repairs:
            failures.append(
                f"{task} seed {payload.get('seed')} parser repair count "
                f"{repairs} != {expected_repairs}"
            )
        if runs is not None:
            matching = runs[
                runs["task"].astype(str).str.lower().eq(task.lower())
                & runs["seed"].map(lambda value: _strict_int(value, "runs.seed")).eq(seed)
            ]
            if matching.empty:
                failures.append(f"{task} seed {seed} has no matching run rows")
            else:
                expected_counts = {
                    "train": ("official_train_n", "cleaned_train_n"),
                    "valid": (None, "common_valid_n"),
                    "test": (None, "common_test_n"),
                }
                for split_name, (input_column, output_column) in expected_counts.items():
                    report = splits[split_name]
                    try:
                        audit_output = _strict_int(
                            report["output_rows"], f"{task} seed {seed}.{split_name}.output_rows"
                        )
                        run_outputs = {
                            _strict_int(value, f"runs.{output_column}")
                            for value in matching[output_column]
                        }
                        if run_outputs != {audit_output}:
                            failures.append(
                                f"{task} seed {seed} {split_name} output count disagrees with runs"
                            )
                        if input_column is not None:
                            audit_input = _strict_int(
                                report["input_rows"],
                                f"{task} seed {seed}.{split_name}.input_rows",
                            )
                            run_inputs = {
                                _strict_int(value, f"runs.{input_column}")
                                for value in matching[input_column]
                            }
                            if run_inputs != {audit_input}:
                                failures.append(
                                    f"{task} seed {seed} {split_name} input count disagrees with runs"
                                )
                    except (KeyError, ValueError) as exc:
                        failures.append(str(exc))
    if failures:
        raise ValueError("Sensitivity audit provenance gate failed: " + "; ".join(failures[:20]))
    return {
        "path": str(path),
        "payloads": len(payloads),
        "expected_payloads": len(expected_keys),
        "flattened_rows": len(payloads) * 3,
        "exact_task_seed_matrix": True,
        "schema_v2": True,
        "post_precedence_overlap_zero": True,
        "parser_repair_counts_match_sources": True,
        "split_output_counts_match_runs": runs is not None,
        "audit_sha256": _sha256_bytes(audit_raw),
        "audit_bytes": len(audit_raw),
    }


def _load_config(
    path: Path, *, raw: bytes | None = None
) -> tuple[list[str], list[str], list[int], dict[str, float]]:
    if not path.is_file():
        raise ValueError(f"Sensitivity config is missing: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to validate the sensitivity config") from exc
    if raw is None:
        raw = path.read_bytes()
    try:
        config = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Sensitivity config is invalid YAML: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError("Sensitivity config must contain one YAML mapping")
    for key in ("selected_tasks_for_cleaned_rerun", "models", "seeds"):
        if key not in config:
            raise ValueError(f"Sensitivity config is missing required key: {key}")
    tasks = [str(value).strip() for value in config["selected_tasks_for_cleaned_rerun"]]
    models = [str(value).strip() for value in config["models"]]
    seeds = [
        _strict_int(value, f"config.seeds[{index}]")
        for index, value in enumerate(config["seeds"])
    ]
    if not tasks or any(not value for value in tasks) or len({v.lower() for v in tasks}) != len(tasks):
        raise ValueError("Sensitivity config tasks must be nonempty and unique")
    if not models or any(not value for value in models) or len({v.lower() for v in models}) != len(models):
        raise ValueError("Sensitivity config models must be nonempty and unique")
    if not seeds or any(value <= 0 for value in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Sensitivity config seeds must be positive and unique")
    if config.get("protocol") != "paired-common-test-v1":
        raise ValueError("Sensitivity config protocol must be paired-common-test-v1")
    audit_contract = config.get("audit")
    if not isinstance(audit_contract, dict):
        raise ValueError("Sensitivity config audit contract must be a mapping")
    if audit_contract.get("scaffold_overlap") is not False:
        raise ValueError(
            "Sensitivity config must not claim a scaffold-overlap audit that was not run"
        )
    raw = config.get("material_change_thresholds", {})
    if not isinstance(raw, dict):
        raise ValueError("material_change_thresholds must be a mapping")
    required_thresholds = ("auroc", "auprc", "spearman", "mae_relative")
    missing_thresholds = sorted(set(required_thresholds) - set(raw))
    if missing_thresholds:
        raise ValueError(f"Sensitivity config thresholds are missing: {missing_thresholds}")
    thresholds = {
        "auroc": float(raw["auroc"]),
        "auprc": float(raw["auprc"]),
        "spearman": float(raw["spearman"]),
        "mae": float(raw["mae_relative"]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in thresholds.values()):
        raise ValueError("Sensitivity config thresholds must be finite and strictly positive")
    return tasks, models, seeds, thresholds


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 20260714, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    """Two-sided exact sign-flip sensitivity under endpoint sign exchangeability."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = abs(float(np.mean(values)))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def _effect_column(metric: str) -> str:
    return "primary_relative_oriented_delta" if metric == "mae" else "primary_oriented_delta"


def _best_model(frame: pd.DataFrame, value_column: str, metric: str) -> str:
    means = frame.groupby("model", sort=True)[value_column].mean()
    return str(means.idxmin() if metric == "mae" else means.idxmax())


def validate_run_integrity(runs: pd.DataFrame) -> dict[str, object]:
    """Verify row metadata, finite values, metric identity, and paired algebra."""

    count_columns = [
        "official_train_n",
        "cleaned_train_n",
        "common_valid_n",
        "common_test_n",
    ]
    core_numeric = [
        *count_columns,
        "official_fit_seconds",
        "cleaned_fit_seconds",
        "primary_official_common_test",
        "primary_cleaned_common_test",
        "primary_raw_delta_cleaned_minus_official",
        "primary_oriented_delta",
        "primary_relative_oriented_delta",
    ]
    required = {
        "task",
        "task_type",
        "official_metric",
        "model",
        "seed",
        *core_numeric,
    }
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Sensitivity run integrity is missing columns: {missing}")
    if runs.empty:
        raise ValueError("Sensitivity run integrity table is empty")
    strict_integer_columns = [
        "seed",
        "radius",
        "n_bits",
        "threads_per_model",
        *count_columns,
    ]
    missing_integers = sorted(set(strict_integer_columns) - set(runs.columns))
    if missing_integers:
        raise ValueError(
            f"Sensitivity run integrity is missing integer columns: {missing_integers}"
        )
    parsed_integers: dict[str, list[int]] = {}
    for column in strict_integer_columns:
        parsed_integers[column] = [
            _strict_int(value, f"runs.{column} row {index}")
            for index, value in enumerate(runs[column])
        ]
    if set(parsed_integers["radius"]) != {2} or set(parsed_integers["n_bits"]) != {2048}:
        raise ValueError("Sensitivity run fingerprint contract is not radius=2/n_bits=2048")
    if set(parsed_integers["threads_per_model"]) != {1}:
        raise ValueError("Sensitivity run CPU thread contract is not one thread per model")

    numeric = runs[core_numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Sensitivity run integrity contains non-finite core values")
    for column in count_columns:
        values = np.asarray(parsed_integers[column], dtype=int)
        if (values <= 0).any():
            raise ValueError(f"Sensitivity run integrity has invalid counts in {column}")
    if (numeric["cleaned_train_n"] > numeric["official_train_n"]).any():
        raise ValueError("Cleaned training count exceeds official training count")
    if (numeric[["official_fit_seconds", "cleaned_fit_seconds"]] < 0).any().any():
        raise ValueError("Sensitivity run integrity has negative fit time")

    metadata = {
        str(task).lower(): (
            str(values["task_type"]).lower(),
            str(values["metric"]).lower(),
        )
        for task, values in TDC_TASK_META.items()
    }
    algebra_failures: list[str] = []
    for index, row in runs.iterrows():
        task = str(row["task"]).lower()
        if task not in metadata:
            algebra_failures.append(f"row {index}: unknown task {row['task']}")
            continue
        expected_type, expected_metric = metadata[task]
        metric = str(row["official_metric"]).lower()
        if str(row["task_type"]).lower() != expected_type or metric != expected_metric:
            algebra_failures.append(f"row {index}: task metadata mismatch")
            continue
        official_metric_column = f"official_common_test_{metric}"
        cleaned_metric_column = f"cleaned_common_test_{metric}"
        if (
            official_metric_column not in runs.columns
            or cleaned_metric_column not in runs.columns
        ):
            algebra_failures.append(f"row {index}: missing primary metric columns")
            continue
        metric_values = pd.to_numeric(
            pd.Series(
                [row[official_metric_column], row[cleaned_metric_column]]
            ),
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.isfinite(metric_values).all():
            algebra_failures.append(f"row {index}: non-finite primary metric")
            continue
        for split in ("valid", "test"):
            for training in ("official", "cleaned"):
                column = f"{training}_common_{split}_{metric}"
                value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
                if not np.isfinite(float(value)):
                    algebra_failures.append(f"row {index}: non-finite {column}")
        official = float(row["primary_official_common_test"])
        cleaned = float(row["primary_cleaned_common_test"])
        raw = cleaned - official
        oriented = official - cleaned if metric == "mae" else raw
        relative = oriented / max(abs(official), 1e-12)
        comparisons = [
            (official, metric_values[0], "official primary metric"),
            (cleaned, metric_values[1], "cleaned primary metric"),
            (float(row["primary_raw_delta_cleaned_minus_official"]), raw, "raw delta"),
            (float(row["primary_oriented_delta"]), oriented, "oriented delta"),
            (float(row["primary_relative_oriented_delta"]), relative, "relative delta"),
        ]
        for observed, expected, label in comparisons:
            if not np.isclose(observed, expected, rtol=1e-10, atol=1e-12):
                algebra_failures.append(f"row {index}: {label} mismatch")
    if algebra_failures:
        raise ValueError(
            "Sensitivity run integrity algebra/metadata gate failed: "
            + "; ".join(algebra_failures[:20])
        )

    grouped_counts = runs.groupby(["task", "seed"], sort=False)[count_columns].nunique(
        dropna=False
    )
    if (grouped_counts != 1).any().any():
        raise ValueError("Sensitivity sample counts differ across models within task/seed")
    return {
        "finite_core_values": True,
        "strict_integer_fields": True,
        "fingerprint_contract_matches": True,
        "single_thread_cpu_contract_matches": True,
        "task_metadata_matches_tdc": True,
        "primary_metric_columns_match": True,
        "paired_delta_algebra_matches": True,
        "sample_counts_consistent_across_models": True,
        "validated_rows": int(len(runs)),
    }


def summarize_runs(
    runs: pd.DataFrame,
    *,
    expected_tasks: list[str],
    expected_models: list[str],
    expected_seeds: list[int],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    run_integrity_gate = validate_run_integrity(runs)
    required = {
        "task",
        "model",
        "seed",
        "official_metric",
        "primary_official_common_test",
        "primary_cleaned_common_test",
        "primary_oriented_delta",
        "primary_relative_oriented_delta",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Sensitivity run table is missing columns: {sorted(missing)}")
    duplicate_keys = int(runs.duplicated(["task", "model", "seed"]).sum())
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate task/model/seed rows.")

    rows: list[dict[str, object]] = []
    for (task, model), group in runs.groupby(["task", "model"], sort=True):
        metrics = group["official_metric"].dropna().astype(str).unique()
        if len(metrics) != 1:
            raise ValueError(f"Expected one official metric for {task}/{model}, got {metrics}")
        metric = str(metrics[0])
        effect_column = _effect_column(metric)
        effects = group[effect_column].astype(float).to_numpy()
        low, high = _bootstrap_mean_ci(effects)
        threshold = float(thresholds[metric])
        mean_effect = float(np.nanmean(effects))
        rows.append(
            {
                "task": task,
                "model": model,
                "official_metric": metric,
                "n_seeds": int(group["seed"].nunique()),
                "primary_official_mean": float(group["primary_official_common_test"].mean()),
                "primary_cleaned_mean": float(group["primary_cleaned_common_test"].mean()),
                "oriented_delta_mean": float(group["primary_oriented_delta"].mean()),
                "oriented_delta_std_across_seeds": float(
                    group["primary_oriented_delta"].std(ddof=1)
                ),
                "relative_oriented_delta_mean": float(
                    group["primary_relative_oriented_delta"].mean()
                ),
                "material_effect_scale": "relative" if metric == "mae" else "absolute",
                "material_effect_value": mean_effect,
                "material_change_threshold": threshold,
                "material_change": bool(abs(mean_effect) > threshold),
                "descriptive_seed_bootstrap_ci_low": low,
                "descriptive_seed_bootstrap_ci_high": high,
                "official_train_n_mean": float(group["official_train_n"].mean()),
                "cleaned_train_n_mean": float(group["cleaned_train_n"].mean()),
                "common_test_n_mean": float(group["common_test_n"].mean()),
            }
        )
    model_summary = pd.DataFrame(rows)

    task_rows: list[dict[str, object]] = []
    for task, group in runs.groupby("task", sort=True):
        metric_values = group["official_metric"].astype(str).unique()
        if len(metric_values) != 1:
            raise ValueError(f"Expected one metric for task {task}.")
        metric = str(metric_values[0])
        official_winner = _best_model(group, "primary_official_common_test", metric)
        cleaned_winner = _best_model(group, "primary_cleaned_common_test", metric)
        task_models = model_summary[model_summary["task"] == task].copy()
        standardized_effects = (
            task_models["material_effect_value"].astype(float)
            / task_models["material_change_threshold"].astype(float)
        )
        task_rows.append(
            {
                "task": task,
                "official_metric": metric,
                "official_selected_model": official_winner,
                "cleaned_best_model": cleaned_winner,
                "winner_changed_after_cleaning": official_winner != cleaned_winner,
                "n_models": int(len(task_models)),
                "material_models": int(task_models["material_change"].sum()),
                "mean_standardized_oriented_effect_across_models": float(
                    standardized_effects.mean()
                ),
                "median_absolute_standardized_effect_across_models": float(
                    standardized_effects.abs().median()
                ),
                "task_material_change": bool(standardized_effects.abs().median() > 1.0),
            }
        )
    task_summary = pd.DataFrame(task_rows)

    expected_keys = {
        (task.lower(), model.lower(), _strict_int(seed, "expected seed"))
        for task in expected_tasks
        for model in expected_models
        for seed in expected_seeds
    }
    observed_keys = {
        (
            str(row.task).lower(),
            str(row.model).lower(),
            _strict_int(row.seed, "runs.seed"),
        )
        for row in runs.itertuples()
    }
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    complete = not missing_keys and not unexpected_keys
    material_tasks = int(task_summary["task_material_change"].sum())
    winner_flips = int(task_summary["winner_changed_after_cleaning"].sum())
    if not complete:
        status = "INCOMPLETE"
    elif material_tasks <= 2 and winner_flips <= 2:
        status = "SENSITIVITY_STABLE"
    elif material_tasks <= 4 and winner_flips <= 4:
        status = "SENSITIVITY_MIXED"
    else:
        status = "SENSITIVITY_SENSITIVE"

    standardized = task_summary[
        "mean_standardized_oriented_effect_across_models"
    ].astype(float).to_numpy()
    gate: dict[str, object] = {
        "status": status,
        "complete_expected_matrix": complete,
        "expected_rows": len(expected_keys),
        "observed_rows": len(observed_keys),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "missing_keys_preview": [list(key) for key in missing_keys[:20]],
        "materially_changed_tasks": material_tasks,
        "winner_flips": winner_flips,
        "n_tasks": int(task_summary["task"].nunique()),
        "median_standardized_oriented_effect": float(np.nanmedian(standardized)),
        "endpoint_level_exact_sign_flip_pvalue_exploratory": exact_sign_flip_pvalue(
            standardized
        ),
        "decision_rule": {
            "SENSITIVITY_STABLE": "At most 2/8 material task effects and at most 2/8 model-winner flips.",
            "SENSITIVITY_MIXED": "At most 4/8 material task effects and at most 4/8 winner flips.",
            "SENSITIVITY_SENSITIVE": "More than 4/8 material task effects or winner flips.",
        },
        "inference_note": "Seed bootstrap intervals are descriptive because seeds share a test set; endpoint is the comparison unit. The sign-flip value assumes sign exchangeability and is not dependence-aware confirmatory inference.",
        "task_aggregation_note": "A task is material when at least two of three pre-specified models exceed the metric-specific threshold; no model is selected for this gate.",
        "main_result_policy": "Never replace the primary official-TDC tables with cleaned results.",
        "run_integrity_gate": run_integrity_gate,
    }
    return model_summary, task_summary, gate


def flatten_audit(path: Path, *, raw: bytes | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if raw is None:
        raw = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            raise ValueError("Sensitivity audit contains a blank line")
        payload = json.loads(line)
        overlap_before = payload["canonical_molecule_overlap_before_precedence"]
        overlap_after = payload["canonical_molecule_overlap_after_precedence"]
        for split_name, report in payload["splits"].items():
            rows.append(
                {
                    "task": payload["task"],
                    "dataset_name": payload.get("dataset_name"),
                    "seed": payload["seed"],
                    "split": split_name,
                    **report,
                    **{f"overlap_before_{key}": value for key, value in overlap_before.items()},
                    **{f"overlap_after_{key}": value for key, value in overlap_after.items()},
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired cleaned-data sensitivity runs.")
    parser.add_argument(
        "--runs", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_runs.csv")
    )
    parser.add_argument(
        "--audit", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity_audit.jsonl")
    )
    parser.add_argument(
        "--config", default=str(ROOT / "experiments" / "config" / "sensitivity.yaml")
    )
    parser.add_argument(
        "--data-dir", default=str(ROOT / "data" / "tdc_official")
    )
    parser.add_argument(
        "--out-dir", default=str(ROOT / "results" / "tables" / "cleaned_sensitivity")
    )
    parser.add_argument(
        "--expected-split-source",
        choices=("pytdc", "locked_cache"),
        help="Fail if run rows do not all declare this explicit split protocol.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Schema-v2 run manifest. Required when --expected-split-source is set; "
            "provenance is never inferred from an output filename."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config_raw = config_path.read_bytes() if config_path.is_file() else None
    tasks, models, seeds, thresholds = _load_config(config_path, raw=config_raw)
    runs_path = Path(args.runs)
    if not runs_path.is_file():
        raise ValueError(f"Sensitivity run table is missing: {runs_path}")
    runs_raw = runs_path.read_bytes()
    run_lines = runs_raw.decode("utf-8").splitlines()
    if any(not line.strip() for line in run_lines):
        raise ValueError("Sensitivity run table contains a blank row")
    try:
        runs = pd.read_csv(io.BytesIO(runs_raw))
    except pd.errors.ParserError as exc:
        raise ValueError("Sensitivity run table is malformed CSV") from exc
    audit_path = Path(args.audit)
    audit_raw = audit_path.read_bytes() if audit_path.is_file() else None
    manifest_gate = None
    audit_gate = None
    if args.expected_split_source:
        if "split_source" not in runs.columns:
            raise ValueError("Sensitivity run table is missing split_source provenance")
        observed_split_sources = set(runs["split_source"].astype(str))
        if observed_split_sources != {args.expected_split_source}:
            raise ValueError(
                "Sensitivity split_source mismatch: "
                f"expected {args.expected_split_source}, observed "
                f"{sorted(observed_split_sources)}"
            )
        if "split_algorithm" not in runs.columns:
            raise ValueError("Sensitivity run table is missing split_algorithm provenance")
        expected_algorithm = (
            LOCKED_CACHE_SPLIT_ALGORITHM
            if args.expected_split_source == LOCKED_CACHE_SPLIT_SOURCE
            else "pytdc_default_task_specific_runtime"
        )
        observed_algorithms = set(runs["split_algorithm"].astype(str))
        if observed_algorithms != {expected_algorithm}:
            raise ValueError(
                "Sensitivity split_algorithm mismatch: "
                f"expected {expected_algorithm}, observed {sorted(observed_algorithms)}"
            )
        if args.manifest is None:
            raise ValueError(
                "--manifest is required with --expected-split-source; provenance "
                "cannot be inferred from the run-table filename"
            )
        manifest_raw = args.manifest.read_bytes() if args.manifest.is_file() else None
        manifest_gate = validate_manifest(
            args.manifest,
            expected_split_source=args.expected_split_source,
            expected_tasks=tasks,
            expected_models=models,
            expected_seeds=seeds,
            config_path=Path(args.config),
            data_dir=Path(args.data_dir),
            manifest_raw=manifest_raw,
            config_raw=config_raw,
        )
        audit_gate = validate_audit(
            audit_path,
            expected_tasks=tasks,
            expected_seeds=seeds,
            expected_compatibility_occurrences=manifest_gate[
                "compatibility_source_occurrences"
            ],
            runs=runs,
            audit_raw=audit_raw,
        )
    model_summary, task_summary, gate = summarize_runs(
        runs,
        expected_tasks=tasks,
        expected_models=models,
        expected_seeds=seeds,
        thresholds=thresholds,
    )
    gate["split_source"] = args.expected_split_source or "not_cli_gated"
    gate["split_source_gate"] = bool(
        not args.expected_split_source
        or set(runs["split_source"].astype(str)) == {args.expected_split_source}
    )
    gate["split_algorithm"] = (
        manifest_gate["split_algorithm"] if manifest_gate else "not_cli_gated"
    )
    gate["manifest_provenance_gate"] = manifest_gate
    gate["audit_provenance_gate"] = audit_gate
    gate["input_artifacts"] = {
        "runs": {
            "path": str(runs_path),
            "bytes": len(runs_raw),
            "sha256": _sha256_bytes(runs_raw),
            "rows": int(len(runs)),
        },
        "audit": {
            "path": str(audit_path),
            "bytes": len(audit_raw or b""),
            "sha256": _sha256_bytes(audit_raw or b""),
            "payloads": int(audit_gate["payloads"]) if audit_gate else None,
        },
        "manifest": {
            "path": str(args.manifest) if args.manifest else None,
            "bytes": manifest_gate["manifest_bytes"] if manifest_gate else None,
            "sha256": manifest_gate["manifest_sha256"] if manifest_gate else None,
        },
        "config": {
            "logical_path": _logical_config_path(config_path),
            "bytes": len(config_raw or b""),
            "sha256": _sha256_bytes(config_raw or b""),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = flatten_audit(audit_path, raw=audit_raw)
    output_frames = {
        "task_model": (
            out_dir / "cleaned_sensitivity_by_task_model.csv",
            model_summary,
        ),
        "task": (out_dir / "cleaned_sensitivity_by_task.csv", task_summary),
        "audit": (out_dir / "cleaned_sensitivity_audit.csv", audit),
    }
    output_bytes = {
        name: _frame_csv_bytes(frame)
        for name, (_, frame) in output_frames.items()
    }
    gate["output_artifacts"] = {
        name: {
            "path": str(path),
            "serialization": "utf8-pandas-csv-lf-float17g-v1",
            "sha256": _sha256_bytes(output_bytes[name]),
            "bytes": len(output_bytes[name]),
            "rows": int(len(frame)),
        }
        for name, (path, frame) in output_frames.items()
    }
    for name, (path, _) in output_frames.items():
        _atomic_write_bytes(path, output_bytes[name])
    gate_bytes = (json.dumps(gate, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(out_dir / "cleaned_sensitivity_gate.json", gate_bytes)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
