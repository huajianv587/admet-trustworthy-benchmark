from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from admet_benchmark.data_loader import OfficialSplit, TDC_TASK_META
from admet_benchmark.sensitivity import (
    CleaningPolicy,
    clean_official_split_for_sensitivity,
    clean_one_split,
)
from admet_benchmark.smiles_compat import LEGACY_VALENCE_REPAIRS


ROOT = Path(__file__).resolve().parents[1]


def _fake_canonicalize(smiles: str) -> str | None:
    if smiles == "bad":
        return None
    return smiles.lower()


def _frame(smiles: list[str], labels: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Drug": smiles, "Y": labels})


def test_cleaning_drops_conflicts_multicomponent_and_cross_split_overlap() -> None:
    train = _frame(
        ["A", "a", "B", "b", "C", "D", "E", "salt.part", "bad", "N"],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, float("nan")],
    )
    valid = _frame(["C", "D", "F"], [1, 1, 0])
    test = _frame(["D", "G", "g", "H", "h", "test.salt"], [1, 0, 0, 0, 1, 1])
    official = OfficialSplit(
        name="synthetic",
        train=train,
        valid=valid,
        test=test,
        task_type="classification",
        metric="auroc",
        category="test",
    )

    cleaned = clean_official_split_for_sensitivity(
        official, canonicalizer=_fake_canonicalize
    )
    assert cleaned.train["Drug"].tolist() == ["a", "e"]
    assert cleaned.valid["Drug"].tolist() == ["c", "f"]
    assert cleaned.test["Drug"].tolist() == ["d", "g"]
    assert set(cleaned.train["Drug"]).isdisjoint(cleaned.valid["Drug"])
    assert set(cleaned.train["Drug"]).isdisjoint(cleaned.test["Drug"])
    assert set(cleaned.valid["Drug"]).isdisjoint(cleaned.test["Drug"])

    # Inputs are evidence and must not be rewritten by the cleaning function.
    assert official.train["Drug"].tolist()[0:2] == ["A", "a"]
    audit = cleaned.audit
    assert audit["splits"]["train"]["conflicting_label_groups_dropped"] == 1
    assert audit["splits"]["train"]["salts_or_mixtures_dropped"] == 1
    assert audit["splits"]["valid"]["cross_split_rows_removed"] == 1
    assert audit["splits"]["train"]["cross_split_rows_removed"] == 2
    assert not any(audit["canonical_molecule_overlap_after_precedence"].values())


def test_regression_conflicting_replicates_are_not_averaged() -> None:
    frame = _frame(["X", "x", "Y", "y", "Z"], [1.0, 1.0, 2.0, 2.1, 3.0])
    cleaned, report = clean_one_split(
        frame,
        split_name="train",
        task_type="regression",
        policy=CleaningPolicy(),
        canonicalizer=_fake_canonicalize,
    )
    assert cleaned["Drug"].tolist() == ["x", "z"]
    assert cleaned["Y"].tolist() == [1.0, 3.0]
    assert report["conflicting_label_groups_dropped"] == 1
    assert report["conflicting_label_rows_dropped"] == 2


def test_parser_compatibility_translation_is_not_an_invalid_smiles_deletion() -> None:
    legacy = next(iter(LEGACY_VALENCE_REPAIRS))
    cleaned, report = clean_one_split(
        _frame([legacy, "CCO"], [1.0, 2.0]),
        split_name="test",
        task_type="regression",
    )
    assert len(cleaned) == 2
    assert report["invalid_smiles_dropped"] == 0
    assert report["parser_compatibility_repairs_applied"] == 1
    assert report["parser_compatibility_repairs_count_as_cleaning_deletions"] is False


def _load_summarizer():
    path = ROOT / "experiments" / "summarize_cleaned_sensitivity.py"
    spec = importlib.util.spec_from_file_location("summarize_cleaned_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_nonrepair_audit_is_atomically_upgradable_on_resume(
    tmp_path: Path,
) -> None:
    from experiments.run_cleaned_sensitivity_cpu import _upgrade_existing_audit

    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps(
            {
                "task": "Caco2_Wang",
                "seed": 1,
                "splits": {name: {} for name in ("train", "valid", "test")},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _upgrade_existing_audit(
        path, tasks_with_compatibility_sources={"solubility_aqsoldb"}
    )
    upgraded = json.loads(path.read_text())
    assert upgraded["audit_schema_version"] == 2
    assert upgraded["splits"]["test"]["parser_compatibility_repairs_applied"] == 0

    path.write_text(
        json.dumps(
            {
                "task": "Solubility_AqSolDB",
                "seed": 1,
                "splits": {name: {} for name in ("train", "valid", "test")},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # A preparation-only row is regenerable and is removed on upgrade.
    _upgrade_existing_audit(
        path,
        tasks_with_compatibility_sources={"solubility_aqsoldb"},
        completed_tasks=set(),
    )
    assert path.read_text() == ""

    path.write_text(
        json.dumps(
            {
                "task": "Solubility_AqSolDB",
                "seed": 1,
                "splits": {name: {} for name in ("train", "valid", "test")},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed task"):
        _upgrade_existing_audit(
            path,
            tasks_with_compatibility_sources={"solubility_aqsoldb"},
            completed_tasks={"solubility_aqsoldb"},
        )


def test_resume_reloads_audit_after_dropping_prepare_only_compatibility_row(
    tmp_path: Path,
) -> None:
    from experiments.run_cleaned_sensitivity_cpu import (
        _existing_audit_keys,
        _upgrade_existing_audit,
    )

    path = tmp_path / "audit.jsonl"
    payloads = [
        {
            "task": "Caco2_Wang",
            "seed": 1,
            "splits": {name: {} for name in ("train", "valid", "test")},
        },
        {
            "task": "Solubility_AqSolDB",
            "seed": 1,
            "splits": {name: {} for name in ("train", "valid", "test")},
        },
    ]
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    done = {("caco2_wang", "rf", 1)}
    _upgrade_existing_audit(
        path,
        tasks_with_compatibility_sources={"solubility_aqsoldb"},
        completed_tasks={task for task, _, _ in done},
    )
    audited_after_upgrade = _existing_audit_keys(path)
    completed_tasks = {task for task, _, _ in done} | {
        task for task, _ in audited_after_upgrade
    }
    assert completed_tasks == {"caco2_wang"}


def test_run_table_provenance_upgrade_requires_and_uses_v2_manifest(
    tmp_path: Path,
) -> None:
    from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM
    from experiments.run_cleaned_sensitivity_cpu import (
        FIELDNAMES,
        _upgrade_existing_run_table_provenance,
    )

    runs = tmp_path / "runs.csv"
    legacy_fields = [
        field for field in FIELDNAMES if field not in {"split_source", "split_algorithm"}
    ]
    with runs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        row = {field: "" for field in legacy_fields}
        row.update({"task": "AMES", "model": "rf", "seed": "1"})
        writer.writerow(row)

    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="without manifest"):
        _upgrade_existing_run_table_provenance(
            runs, manifest_path=manifest, split_source="locked_cache"
        )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
            }
        ),
        encoding="utf-8",
    )
    _upgrade_existing_run_table_provenance(
        runs, manifest_path=manifest, split_source="locked_cache"
    )
    upgraded = pd.read_csv(runs, keep_default_na=False)
    assert list(upgraded.columns) == FIELDNAMES
    assert set(upgraded["split_source"]) == {"locked_cache"}
    assert set(upgraded["split_algorithm"]) == {LOCKED_CACHE_SPLIT_ALGORITHM}


def test_sensitivity_manifest_gate_binds_locked_source_contract(
    tmp_path: Path,
) -> None:
    from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM

    module = _load_summarizer()
    config = tmp_path / "sensitivity.yaml"
    current_config = """
protocol: paired-common-test-v1
selected_tasks_for_cleaned_rerun:
  - Caco2_Wang
  - BBB_Martins
  - PPBR_AZ
  - Solubility_AqSolDB
  - CYP2C9_Veith
  - DILI
  - AMES
  - LD50_Zhu
models: [rf, xgboost, lightgbm]
seeds: [1, 2, 3, 4, 5]
audit:
  scaffold_overlap: false
material_change_thresholds:
  auroc: 0.02
  auprc: 0.02
  spearman: 0.03
  mae_relative: 0.05
""".lstrip()
    config.write_text(current_config, encoding="utf-8")
    execution_config = current_config.replace(
        "  scaffold_overlap: false", "  scaffold_overlap: true"
    )
    current_config_sha = hashlib.sha256(current_config.encode()).hexdigest()
    execution_config_sha = hashlib.sha256(execution_config.encode()).hexdigest()

    data_dir = tmp_path / "data"
    input_sha256: dict[str, dict[str, str]] = {}
    for index, task in enumerate(module.DEFAULT_TASKS):
        task_dir = data_dir / "admet_group" / task.lower()
        task_dir.mkdir(parents=True)
        input_sha256[task] = {}
        for offset, filename in enumerate(("train_val.csv", "test.csv")):
            path = task_dir / filename
            pd.DataFrame(
                {"Drug": ["CCO", "CCN"], "Y": [float(index), float(index + offset + 1)]}
            ).to_csv(path, index=False)
            input_sha256[task][filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    manifest = tmp_path / "manifest.json"
    payload = {
        "schema_version": 2,
        "protocol": "paired-common-test-v1",
        "split_source": "locked_cache",
        "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
        "cpu_only": True,
        "tasks": module.DEFAULT_TASKS,
        "models": module.DEFAULT_MODELS,
        "seeds": module.DEFAULT_SEEDS,
        "threads_per_model": 1,
        "fingerprint": {"radius": 2, "n_bits": 2048},
        "save_predictions": True,
        "cleaning_policy": CleaningPolicy().to_dict(),
        "canonicalization_input_policy": "apply_enumerated_parser_compatibility_before_rdkit_canonicalization",
        "model_input_representation_policy": module.SMILES_COMPATIBILITY_POLICY,
        "parser_compatibility_repairs_count_as_cleaning_deletions": False,
        "compatibility_mapping_sha256": module._stable_json_sha256(
            compatibility_mapping
        ),
        "compatibility_source_occurrences": {
            task: 0 for task in module.DEFAULT_TASKS
        },
        "config_logical_path": config.name,
        "config_sha256": current_config_sha,
        "execution_config_sha256": execution_config_sha,
        "config_metadata_corrections": [
            {
                "field": "audit.scaffold_overlap",
                "old_value": True,
                "new_value": False,
                "execution_config_sha256": execution_config_sha,
                "corrected_config_sha256": current_config_sha,
                "computational_effect": False,
                "reason": "metadata-only test correction",
                "recorded_utc": "2026-07-14T00:00:00+00:00",
            }
        ],
        "input_sha256": input_sha256,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    gate = module.validate_manifest(
        manifest,
        expected_split_source="locked_cache",
        expected_tasks=module.DEFAULT_TASKS,
        expected_models=module.DEFAULT_MODELS,
        expected_seeds=module.DEFAULT_SEEDS,
        config_path=config,
        data_dir=data_dir,
    )
    assert gate["checks"]["source_file_hashes_recomputed"] is True

    payload["split_source"] = "pytdc"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest provenance gate failed"):
        module.validate_manifest(
            manifest,
            expected_split_source="locked_cache",
            expected_tasks=module.DEFAULT_TASKS,
            expected_models=module.DEFAULT_MODELS,
            expected_seeds=module.DEFAULT_SEEDS,
            config_path=config,
            data_dir=data_dir,
        )


def test_prediction_provenance_upgrade_is_manifest_bound(tmp_path: Path) -> None:
    from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM
    from experiments.run_cleaned_sensitivity_cpu import (
        _upgrade_existing_prediction_provenance,
    )

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    path = predictions / "ames_rf_seed1_paired.csv"
    pd.DataFrame(
        {
            "task": ["AMES"],
            "model": ["rf"],
            "seed": [1],
            "evaluation_set": ["common_cleaned_test"],
            "canonical_smiles": ["CCO"],
            "label": [1.0],
            "official_prediction": [0.7],
            "cleaned_prediction": [0.8],
        }
    ).to_csv(path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
            }
        ),
        encoding="utf-8",
    )
    _upgrade_existing_prediction_provenance(
        predictions, manifest_path=manifest, split_source="locked_cache"
    )
    upgraded = pd.read_csv(path)
    assert set(upgraded["split_source"]) == {"locked_cache"}
    assert set(upgraded["split_algorithm"]) == {LOCKED_CACHE_SPLIT_ALGORITHM}
    assert upgraded["official_prediction"].tolist() == [0.7]
    assert upgraded["cleaned_prediction"].tolist() == [0.8]


def test_complete_small_effect_matrix_is_sensitivity_stable() -> None:
    module = _load_summarizer()
    tasks = module.DEFAULT_TASKS
    models = module.DEFAULT_MODELS
    seeds = module.DEFAULT_SEEDS
    rows = []
    maximize_base = {"rf": 0.80, "xgboost": 0.77, "lightgbm": 0.75}
    minimize_base = {"rf": 0.20, "xgboost": 0.23, "lightgbm": 0.25}
    for task in tasks:
        metadata = TDC_TASK_META[task]
        metric = metadata["metric"]
        for model in models:
            for seed in seeds:
                official = (
                    minimize_base[model] if metric == "mae" else maximize_base[model]
                ) + seed * 1e-4
                cleaned = official - 0.001 if metric == "mae" else official + 0.001
                raw_delta = cleaned - official
                oriented_delta = -raw_delta if metric == "mae" else raw_delta
                rows.append(
                    {
                        "task": task,
                        "task_type": metadata["task_type"],
                        "model": model,
                        "seed": seed,
                        "official_metric": metric,
                        "radius": 2,
                        "n_bits": 2048,
                        "threads_per_model": 1,
                        "primary_official_common_test": official,
                        "primary_cleaned_common_test": cleaned,
                        "primary_raw_delta_cleaned_minus_official": raw_delta,
                        "primary_oriented_delta": oriented_delta,
                        "primary_relative_oriented_delta": oriented_delta / official,
                        "official_train_n": 100,
                        "cleaned_train_n": 95,
                        "common_valid_n": 15,
                        "common_test_n": 20,
                        "official_fit_seconds": 0.1,
                        "cleaned_fit_seconds": 0.1,
                        f"official_common_valid_{metric}": official,
                        f"cleaned_common_valid_{metric}": cleaned,
                        f"official_common_test_{metric}": official,
                        f"cleaned_common_test_{metric}": cleaned,
                    }
                )
    _, task_summary, gate = module.summarize_runs(
        pd.DataFrame(rows),
        expected_tasks=tasks,
        expected_models=models,
        expected_seeds=seeds,
        thresholds=module.THRESHOLDS,
    )
    assert gate["complete_expected_matrix"] is True
    assert gate["status"] == "SENSITIVITY_STABLE"
    assert task_summary["winner_changed_after_cleaning"].sum() == 0
