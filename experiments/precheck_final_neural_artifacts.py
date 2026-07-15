#!/usr/bin/env python3
"""Strict 220-run artifact precheck before the five-model final gate.

This command does not train, repair, copy, or reinterpret artifacts.  It
requires each formal GIN and Chemprop D-MPNN run to be self-contained under one
provenance root, validates its prediction/metrics/checkpoint identity contract,
and compares ordered validation/test sample fingerprints with the locked TDC
split loaded for the same task and seed.  Historical ``*_dmpnn_*`` files are
audited separately as ``legacy_node_mpnn`` and never count toward 220/220.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASKS,
    OfficialSplit,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.smiles_compat import (  # noqa: E402
    LEGACY_VALENCE_REPAIR_INCHIKEYS,
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)
from experiments.analyze_reliability_regret import (  # noqa: E402
    FINAL_EXPECTED_SEEDS,
    _normalize_prediction_file,
    sample_fingerprint,
)


FORMAL_NEURAL_MODELS = ("gin", "chemprop_dmpnn")
FORMAL_SOURCE_MODEL = {"gin": "gin", "chemprop_dmpnn": "chemprop_dmpnn"}
FORMAL_FILENAME_RE = re.compile(
    r"^(?P<task>.+?)_(?P<model>gin|chemprop_dmpnn)_seed(?P<seed>\d+)\.csv$"
)
LEGACY_FILENAME_RE = re.compile(r"^.+?_dmpnn_seed\d+\.csv$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOLUBILITY_TASK = "solubility_aqsoldb"
_SOLUBILITY_SOURCE_SMILES = (
    "CC1=CC=C[NH+2]([O-])[CH-]1",
    "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1",
)
SOLUBILITY_COMPATIBILITY_REPAIRS = tuple(
    {
        "split": "test",
        "sample_index": sample_index,
        "source_smiles": source_smiles,
        "model_input_smiles": LEGACY_VALENCE_REPAIRS[source_smiles],
        "standard_inchikey": LEGACY_VALENCE_REPAIR_INCHIKEYS[source_smiles],
        "policy": SMILES_COMPATIBILITY_POLICY,
    }
    for sample_index, source_smiles in zip((1159, 1160), _SOLUBILITY_SOURCE_SMILES)
)
MODEL_INPUT_SPLIT_HASHES = ("train_sha256", "valid_sha256", "test_sha256")
SOURCE_SPLIT_HASHES = (
    "source_train_sha256",
    "source_valid_sha256",
    "source_test_sha256",
)


def _gin_compatibility_policy_error(path: Path) -> str | None:
    """Fail closed unless the published GIN config declares the locked policy."""

    if not path.is_file():
        return "missing_gin_global_config"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_gin_global_config"
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return "gin_config_compatibility_policy_missing_or_mismatch"
    if (
        payload["data"].get("model_input_representation_policy")
        != SMILES_COMPATIBILITY_POLICY
    ):
        return "gin_config_compatibility_policy_missing_or_mismatch"
    return None


def _chemprop_global_compatibility_policy_error(path: Path) -> str | None:
    """Fail closed unless the published Chemprop config declares the policy."""

    if not path.is_file():
        return "missing_chemprop_global_config"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_chemprop_global_config"
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return "chemprop_global_config_compatibility_policy_missing_or_mismatch"
    if (
        payload["data"].get("model_input_representation_policy")
        != SMILES_COMPATIBILITY_POLICY
    ):
        return "chemprop_global_config_compatibility_policy_missing_or_mismatch"
    return None


def _repair_projection(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        sample_index = int(value.get("sample_index"))
    except (TypeError, ValueError):
        return None
    return {
        "split": value.get("split"),
        "sample_index": sample_index,
        "source_smiles": value.get("source_smiles"),
        "model_input_smiles": value.get("model_input_smiles"),
        "standard_inchikey": value.get("standard_inchikey"),
        "policy": value.get("policy"),
    }


def _chemprop_compatibility_config_error(path: Path, task: str) -> str | None:
    """Validate exact parser repairs and both source/model-input split hashes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_chemprop_compatibility_manifest"
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return "invalid_chemprop_compatibility_manifest"
    if not all(
        SHA256_RE.fullmatch(str(data.get(name, "")))
        for name in MODEL_INPUT_SPLIT_HASHES
    ):
        return "chemprop_model_input_split_hash_missing"

    compatibility_fields = (
        "model_input_representation_policy",
        "model_input_repairs_n",
        "model_input_repairs",
        *SOURCE_SPLIT_HASHES,
    )
    is_solubility = str(task).strip().lower() == SOLUBILITY_TASK
    field_presence = [name in data for name in compatibility_fields]
    if not is_solubility and not any(field_presence):
        # The first 105 completed runs predate the explicit compatibility
        # manifest. No representation changed on these tasks, so their locked
        # train/valid/test digests are simultaneously source and model-input
        # digests. The global Chemprop YAML still locks the policy.
        return None
    if not all(field_presence):
        return "chemprop_partial_compatibility_manifest"
    if (
        data.get("model_input_representation_policy")
        != SMILES_COMPATIBILITY_POLICY
    ):
        return "chemprop_compatibility_policy_missing_or_mismatch"
    if not all(
        SHA256_RE.fullmatch(str(data.get(name, ""))) for name in SOURCE_SPLIT_HASHES
    ):
        return "chemprop_source_split_hash_missing"

    repairs = data.get("model_input_repairs")
    try:
        repairs_n = int(data.get("model_input_repairs_n"))
    except (TypeError, ValueError):
        return "chemprop_compatibility_repair_count_invalid"
    if not isinstance(repairs, list) or repairs_n != len(repairs):
        return "chemprop_compatibility_repair_count_invalid"

    if is_solubility:
        observed = [_repair_projection(value) for value in repairs]
        if repairs_n != 2 or any(value is None for value in observed):
            return "chemprop_solubility_exact_compatibility_mapping_mismatch"
        assert all(value is not None for value in observed)
        observed_sorted = sorted(observed, key=lambda value: int(value["sample_index"]))
        expected = [dict(value) for value in SOLUBILITY_COMPATIBILITY_REPAIRS]
        if observed_sorted != expected:
            return "chemprop_solubility_exact_compatibility_mapping_mismatch"
        if (
            data["source_train_sha256"] != data["train_sha256"]
            or data["source_valid_sha256"] != data["valid_sha256"]
            or data["source_test_sha256"] == data["test_sha256"]
        ):
            return "chemprop_solubility_source_model_hash_relation_mismatch"
    elif repairs_n != 0 or repairs != []:
        return "chemprop_unexpected_compatibility_repairs_for_task"
    elif any(
        data[f"source_{name}_sha256"] != data[f"{name}_sha256"]
        for name in ("train", "valid", "test")
    ):
        return "chemprop_no_repair_source_model_hash_mismatch"
    return None


def _chemprop_compatibility_manifest_schema(path: Path, task: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid"
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return "invalid"
    compatibility_fields = (
        "model_input_representation_policy",
        "model_input_repairs_n",
        "model_input_repairs",
        *SOURCE_SPLIT_HASHES,
    )
    if (
        str(task).strip().lower() != SOLUBILITY_TASK
        and not any(name in data for name in compatibility_fields)
    ):
        return "legacy_no_repair_hash_identity"
    return "explicit_v1"


def _artifact_paths(root: Path, task: str, model: str, seed: int) -> dict[str, Path]:
    stem = f"{task}_{FORMAL_SOURCE_MODEL[model]}_seed{seed}"
    paths = {
        "prediction": root / "predictions" / f"{stem}.csv",
        "metrics": root / "metrics" / f"{stem}.json",
        "checkpoint": root / "checkpoints" / f"{stem}.pt",
    }
    if model == "chemprop_dmpnn":
        paths["config"] = root / "configs" / f"{stem}.json"
    return paths


def _frame_for_fingerprint(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return frame.loc[
        frame["split"].eq(split_name), ["sample_index", "smiles", "y_true"]
    ].copy()


def _official_fingerprints(split: OfficialSplit) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for split_name in ("valid", "test"):
        source = getattr(split, split_name)
        smiles, labels = extract_xy(source)
        frame = pd.DataFrame(
            {
                "sample_index": range(len(smiles)),
                "smiles": smiles,
                "y_true": pd.to_numeric(labels, errors="raise").astype(float),
            }
        )
        result[f"official_{split_name}_rows"] = len(frame)
        result[f"official_{split_name}_fingerprint"] = sample_fingerprint(frame)
        result[f"official_{split_name}_labels"] = frame["y_true"].tolist()
    return result


def _existing(path: Path | None) -> bool:
    return bool(path is not None and path.is_file() and path.stat().st_size > 0)


def audit_final_neural_artifacts(
    *,
    gin_dir: Path,
    chemprop_dir: Path,
    data_dir: Path,
    gin_config_path: Path | None = None,
    chemprop_config_path: Path | None = None,
    tasks: Sequence[str] = tuple(TDC_TASKS),
    seeds: Sequence[int] = FINAL_EXPECTED_SEEDS,
    verify_official_splits: bool = True,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
    loader: Callable[..., OfficialSplit] = load_tdc_official,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return the expected-run ledger, legacy ledger, and conservative gate."""
    task_list = list(tasks)
    seed_list = [int(seed) for seed in seeds]
    if len(task_list) != len(set(task_list)):
        raise ValueError("tasks must be unique")
    if len(seed_list) != len(set(seed_list)):
        raise ValueError("seeds must be unique")
    roots = {"gin": Path(gin_dir), "chemprop_dmpnn": Path(chemprop_dir)}
    gin_config_path = Path(
        gin_config_path or ROOT / "experiments" / "config" / "gin.yaml"
    )
    chemprop_config_path = Path(
        chemprop_config_path
        or ROOT / "experiments" / "config" / "chemprop_dmpnn.yaml"
    )
    gin_config_error = _gin_compatibility_policy_error(gin_config_path)
    chemprop_global_config_error = _chemprop_global_compatibility_policy_error(
        chemprop_config_path
    )

    official: dict[tuple[str, int], dict[str, str | int] | str] = {}
    if verify_official_splits:
        for task in task_list:
            for seed in seed_list:
                try:
                    split = loader(
                        task,
                        data_dir=data_dir,
                        seed=seed,
                        split_source=split_source,
                    )
                    official[(task, seed)] = _official_fingerprints(split)
                except Exception as exc:
                    official[(task, seed)] = (
                        f"official_split_unavailable:{type(exc).__name__}:{exc}"
                    )

    records: list[dict] = []
    for model in FORMAL_NEURAL_MODELS:
        root = roots[model]
        for task in task_list:
            for seed in seed_list:
                paths = _artifact_paths(root, task, model, seed)
                row = {
                    "task": task.lower(),
                    "source_task": task,
                    "model": model,
                    "seed": seed,
                    "root": str(root),
                    **{f"{role}_path": str(path) for role, path in paths.items()},
                    **{f"{role}_exists_nonempty": _existing(path) for role, path in paths.items()},
                    "valid_rows": 0,
                    "test_rows": 0,
                    "valid_sample_fingerprint": "",
                    "test_sample_fingerprint": "",
                    "official_valid_sample_match": not verify_official_splits,
                    "official_test_sample_match": not verify_official_splits,
                    "representation_compatibility_ok": False,
                    "compatibility_manifest_schema": (
                        "global_policy_v1" if model == "gin" else ""
                    ),
                    "included": False,
                    "reason": "",
                }
                prediction = paths["prediction"]
                if not prediction.is_file():
                    row["reason"] = "missing_prediction"
                    records.append(row)
                    continue
                frame, reason = _normalize_prediction_file(prediction)
                if frame is None:
                    row["reason"] = reason or "invalid_prediction"
                    records.append(row)
                    continue
                identities = {
                    "task": set(frame["task"].astype(str)),
                    "model": set(frame["model"].astype(str)),
                    "seed": set(frame["seed"].astype(int)),
                }
                if (
                    identities["task"] != {task.lower()}
                    or identities["model"] != {model}
                    or identities["seed"] != {seed}
                ):
                    row["reason"] = "normalized_run_identity_mismatch"
                    records.append(row)
                    continue
                valid = _frame_for_fingerprint(frame, "valid")
                test = _frame_for_fingerprint(frame, "test")
                row["valid_rows"] = len(valid)
                row["test_rows"] = len(test)
                row["valid_sample_fingerprint"] = sample_fingerprint(valid)
                row["test_sample_fingerprint"] = sample_fingerprint(test)

                source = official.get((task, seed))
                if verify_official_splits and isinstance(source, str):
                    row["reason"] = source
                    records.append(row)
                    continue
                if verify_official_splits:
                    assert isinstance(source, dict)
                    row["official_valid_sample_match"] = bool(
                        row["valid_rows"] == source["official_valid_rows"]
                        and row["valid_sample_fingerprint"]
                        == source["official_valid_fingerprint"]
                        and np.allclose(
                            valid["y_true"].to_numpy(dtype=float),
                            np.asarray(source["official_valid_labels"], dtype=float),
                            rtol=1e-6,
                            atol=1e-6,
                            equal_nan=False,
                        )
                    )
                    row["official_test_sample_match"] = bool(
                        row["test_rows"] == source["official_test_rows"]
                        and row["test_sample_fingerprint"]
                        == source["official_test_fingerprint"]
                        and np.allclose(
                            test["y_true"].to_numpy(dtype=float),
                            np.asarray(source["official_test_labels"], dtype=float),
                            rtol=1e-6,
                            atol=1e-6,
                            equal_nan=False,
                        )
                    )
                    if not row["official_valid_sample_match"]:
                        row["reason"] = "official_valid_sample_fingerprint_mismatch"
                    elif not row["official_test_sample_match"]:
                        row["reason"] = "official_test_sample_fingerprint_mismatch"
                compatibility_error = (
                    gin_config_error
                    if model == "gin"
                    else (
                        chemprop_global_config_error
                        or _chemprop_compatibility_config_error(paths["config"], task)
                    )
                )
                if model == "chemprop_dmpnn":
                    row["compatibility_manifest_schema"] = (
                        _chemprop_compatibility_manifest_schema(paths["config"], task)
                    )
                row["representation_compatibility_ok"] = compatibility_error is None
                if not row["reason"] and compatibility_error:
                    row["reason"] = compatibility_error
                if not row["reason"]:
                    row["included"] = True
                    row["reason"] = "complete_and_aligned"
                records.append(row)

    audit = pd.DataFrame(records)
    expected_keys = {
        (task.lower(), model, seed)
        for task in task_list
        for model in FORMAL_NEURAL_MODELS
        for seed in seed_list
    }

    extra_rows: list[dict] = []
    legacy_rows: list[dict] = []
    for model, root in roots.items():
        prediction_dir = root / "predictions"
        if not prediction_dir.exists():
            continue
        for path in sorted(prediction_dir.glob("*.csv")):
            if path.name.startswith("._"):
                continue
            match = FORMAL_FILENAME_RE.match(path.name)
            if match:
                key = (
                    match.group("task").lower(),
                    match.group("model"),
                    int(match.group("seed")),
                )
                if match.group("model") != model:
                    extra_rows.append(
                        {
                            "path": str(path),
                            "reason": "formal_prediction_in_wrong_model_root",
                            "key": key,
                        }
                    )
                elif key not in expected_keys:
                    extra_rows.append(
                        {"path": str(path), "reason": "unexpected_formal_neural_run", "key": key}
                    )
                continue
            if LEGACY_FILENAME_RE.match(path.name):
                frame, reason = _normalize_prediction_file(path)
                normalized_safely = frame is None or set(frame["model"]) == {
                    "legacy_node_mpnn"
                }
                legacy_rows.append(
                    {
                        "path": str(path),
                        "source_root": str(root),
                        "normalized_model": (
                            "legacy_node_mpnn" if normalized_safely else "INVALID"
                        ),
                        "excluded_from_formal_220": normalized_safely,
                        "normalization_reason": reason or "complete_legacy_artifact",
                    }
                )
            elif "gin" in path.name.lower() or "dmpnn" in path.name.lower():
                extra_rows.append(
                    {"path": str(path), "reason": "unparseable_neural_prediction_filename", "key": None}
                )

    legacy = pd.DataFrame(legacy_rows)
    complete_by_model = {
        model: int(audit.loc[(audit["model"] == model) & audit["included"]].shape[0])
        for model in FORMAL_NEURAL_MODELS
    }
    expected_per_model = len(task_list) * len(seed_list)
    complete_fingerprints = audit[audit["included"]]
    test_hashes: defaultdict[str, set[str]] = defaultdict(set)
    valid_hashes: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    for row in complete_fingerprints.itertuples(index=False):
        test_hashes[row.task].add(row.test_sample_fingerprint)
        valid_hashes[(row.task, int(row.seed))].add(row.valid_sample_fingerprint)
    inconsistent_test = {
        task: sorted(values) for task, values in test_hashes.items() if len(values) > 1
    }
    inconsistent_valid = {
        f"{task}/seed{seed}": sorted(values)
        for (task, seed), values in valid_hashes.items()
        if len(values) > 1
    }
    legacy_ok = bool(legacy.empty or legacy["excluded_from_formal_220"].all())
    failures = []
    for model in FORMAL_NEURAL_MODELS:
        if complete_by_model[model] != expected_per_model:
            failures.append(
                f"{model}: complete {complete_by_model[model]}/{expected_per_model}"
            )
    if extra_rows:
        failures.append(f"unexpected or unparseable formal neural predictions: {len(extra_rows)}")
    if inconsistent_test:
        failures.append(f"cross-run test sample fingerprint mismatches: {len(inconsistent_test)}")
    if inconsistent_valid:
        failures.append(f"cross-model validation fingerprint mismatches: {len(inconsistent_valid)}")
    if not legacy_ok:
        failures.append("at least one historical dmpnn artifact was not safely excluded")
    if gin_config_error:
        failures.append(f"GIN representation compatibility config: {gin_config_error}")
    if chemprop_global_config_error:
        failures.append(
            "Chemprop representation compatibility config: "
            f"{chemprop_global_config_error}"
        )
    invalid_reasons = Counter(audit.loc[~audit["included"], "reason"].astype(str))
    gate = {
        "status": "GO" if not failures else "HOLD",
        "expected_total_formal_neural_runs": len(expected_keys),
        "expected_per_model": expected_per_model,
        "required_models": list(FORMAL_NEURAL_MODELS),
        "required_tasks": len(task_list),
        "required_seeds": seed_list,
        "complete_by_model": complete_by_model,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "unexpected_predictions": extra_rows,
        "legacy_artifacts_discovered": len(legacy_rows),
        "legacy_exclusion_ok": legacy_ok,
        "sample_alignment": {
            "official_split_verification_enabled": verify_official_splits,
            "official_split_source": split_source,
            "all_expected_runs_match_official_samples": bool(
                audit["included"].all() and len(audit) == len(expected_keys)
            ),
            "inconsistent_test_fingerprints": inconsistent_test,
            "inconsistent_validation_fingerprints": inconsistent_valid,
        },
        "representation_compatibility": {
            "policy": SMILES_COMPATIBILITY_POLICY,
            "gin_global_config_path": str(gin_config_path),
            "gin_global_config_ok": gin_config_error is None,
            "gin_global_config_error": gin_config_error or "",
            "chemprop_global_config_path": str(chemprop_config_path),
            "chemprop_global_config_ok": chemprop_global_config_error is None,
            "chemprop_global_config_error": chemprop_global_config_error or "",
            "solubility_required_test_indices": [1159, 1160],
            "solubility_required_repairs": [
                dict(value) for value in SOLUBILITY_COMPATIBILITY_REPAIRS
            ],
            "chemprop_all_included_runs_valid": bool(
                audit.loc[
                    audit["model"].eq("chemprop_dmpnn"),
                    "representation_compatibility_ok",
                ].all()
            ),
            "chemprop_manifest_schema_counts": dict(
                Counter(
                    audit.loc[
                        audit["model"].eq("chemprop_dmpnn"),
                        "compatibility_manifest_schema",
                    ].astype(str)
                )
            ),
        },
        "failures": failures,
        "interpretation": (
            "All 220 formal neural runs satisfy the locked artifact, identity, seed, "
            "metric-recomputation, and sample-alignment contract. The five-model "
            "reliability-regret analysis may now run."
            if not failures
            else "Formal neural artifacts are incomplete or invalid. Do not run or report "
            "the final five-model gate as complete."
        ),
    }
    return audit, legacy, gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gin-dir", default=str(ROOT / "results" / "neural"))
    parser.add_argument(
        "--chemprop-dir", default=str(ROOT / "results" / "neural_chemprop")
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--gin-config",
        default=str(ROOT / "experiments" / "config" / "gin.yaml"),
    )
    parser.add_argument(
        "--chemprop-config",
        default=str(ROOT / "experiments" / "config" / "chemprop_dmpnn.yaml"),
    )
    parser.add_argument(
        "--split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=LOCKED_CACHE_SPLIT_SOURCE,
        help="Use the immutable cached split for the formal submission gate.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "tables" / "final_neural_artifact_precheck"),
    )
    parser.add_argument(
        "--skip-official-split-verification",
        action="store_true",
        help="Diagnostic only; a submission precheck should not use this option.",
    )
    args = parser.parse_args()

    audit, legacy, gate = audit_final_neural_artifacts(
        gin_dir=Path(args.gin_dir),
        chemprop_dir=Path(args.chemprop_dir),
        data_dir=Path(args.data_dir),
        gin_config_path=Path(args.gin_config),
        chemprop_config_path=Path(args.chemprop_config),
        verify_official_splits=not args.skip_official_split_verification,
        split_source=args.split_source,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "neural_artifact_audit.csv", index=False)
    legacy.to_csv(out_dir / "legacy_artifact_audit.csv", index=False)
    (out_dir / "final_neural_precheck_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"formal neural runs: {sum(gate['complete_by_model'].values())}/"
        f"{gate['expected_total_formal_neural_runs']}"
    )
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
