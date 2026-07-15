from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.run_cleaned_sensitivity_cpu as runner
from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM


def _write_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _run_row(fields: list[str]) -> dict[str, str]:
    row = {field: "" for field in fields}
    row.update({"task": "AMES", "model": "rf", "seed": "1"})
    if "split_source" in fields:
        row["split_source"] = "locked_cache"
    if "split_algorithm" in fields:
        row["split_algorithm"] = LOCKED_CACHE_SPLIT_ALGORITHM
    return row


def _prediction_row(fields: list[str]) -> dict[str, str]:
    row = {
        "task": "AMES",
        "model": "rf",
        "seed": "1",
        "evaluation_set": "common_cleaned_test",
        "canonical_smiles": "CCO",
        "label": "1",
        "official_prediction": "0.7",
        "cleaned_prediction": "0.8",
    }
    if "split_source" in fields:
        row["split_source"] = "locked_cache"
    if "split_algorithm" in fields:
        row["split_algorithm"] = LOCKED_CACHE_SPLIT_ALGORITHM
    return row


def _manifest(config: Path, *, save_predictions: bool = True) -> dict:
    return {
        "schema_version": 2,
        "split_source": "locked_cache",
        "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
        "save_predictions": save_predictions,
        "tasks": ["AMES"],
        "models": ["rf"],
        "seeds": [1],
        "fingerprint": {"radius": 2, "n_bits": 2048},
        "threads_per_model": 1,
        "config_path": "/old/machine/admet/config/sensitivity.yaml",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }


def test_manifest_lock_refuses_concurrent_invocation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    with runner._exclusive_manifest_lock(manifest):
        with pytest.raises(RuntimeError, match="Refusing concurrent resume/upgrade"):
            with runner._exclusive_manifest_lock(manifest):
                pass
    # Re-acquisition after release is allowed; the persistent lockfile is harmless.
    with runner._exclusive_manifest_lock(manifest) as lock_path:
        assert lock_path == runner._manifest_lock_path(manifest)


def test_upgrade_only_normalizes_and_receipts_without_touching_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_runner_reproducibility_metadata",
        lambda **_: {
            "runner_source_sha256": "a" * 64,
            "runner_source_provenance_scope": "synthetic-test-scope",
            "model_factory_source_sha256": "b" * 64,
            "resolved_estimator_summary": {"models": {"rf": {"version": "test"}}},
        },
    )
    config = tmp_path / "experiments" / "config" / "sensitivity.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("audit:\n  scaffold_overlap: true\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    initial_manifest = _manifest(config)
    execution_config_hash = initial_manifest["config_sha256"]
    manifest_path.write_text(json.dumps(initial_manifest), encoding="utf-8")
    # The sole post-execution edit corrects a declarative capability flag that
    # the runner never reads; byte reconstruction must prove that exact lineage.
    config.write_text("audit:\n  scaffold_overlap: false\n", encoding="utf-8")
    corrected_config_hash = hashlib.sha256(config.read_bytes()).hexdigest()

    legacy_run_fields = [
        field for field in runner.FIELDNAMES if field not in runner.PROVENANCE_FIELDS
    ]
    # Deliberately shuffle a legacy schema; the upgrade must rewrite exact FIELDNAMES.
    legacy_run_fields = legacy_run_fields[3:] + legacy_run_fields[:3]
    runs = tmp_path / "runs.csv"
    _write_csv(runs, legacy_run_fields, _run_row(legacy_run_fields))

    predictions = tmp_path / "predictions"
    prediction = predictions / "ames_rf_seed1_paired.csv"
    _write_csv(
        prediction,
        list(reversed(runner.PREDICTION_CORE_FIELDS)),
        _prediction_row(list(reversed(runner.PREDICTION_CORE_FIELDS))),
    )
    sidecar = predictions / "._ames_rf_seed1_paired.csv"
    sidecar.write_bytes(b"not-a-csv-appledouble")
    sidecar_before = sidecar.read_bytes()
    receipt_path = tmp_path / "receipt.json"
    args = argparse.Namespace(
        manifest=str(manifest_path),
        split_source="locked_cache",
        config=str(config),
        out=str(runs),
        predictions_dir=str(predictions),
        upgrade_receipt=str(receipt_path),
    )

    with runner._exclusive_manifest_lock(manifest_path):
        receipt = runner._run_upgrade_only(args)

    assert pd.read_csv(runs, nrows=0).columns.tolist() == runner.FIELDNAMES
    assert (
        pd.read_csv(prediction, nrows=0).columns.tolist()
        == runner.PREDICTION_FIELDNAMES
    )
    assert sidecar.read_bytes() == sidecar_before
    upgraded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert upgraded_manifest["config_path"].startswith("/old/machine/")
    assert upgraded_manifest["config_logical_path"] == (
        "experiments/config/sensitivity.yaml"
    )
    assert upgraded_manifest["runner_source_sha256"] == "a" * 64
    assert upgraded_manifest["config_sha256"] == corrected_config_hash
    assert upgraded_manifest["execution_config_sha256"] == execution_config_hash
    assert upgraded_manifest["config_metadata_corrections"][0]["field"] == (
        "audit.scaffold_overlap"
    )

    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == receipt
    assert persisted["aggregate_inventory"]["file_count"] == 3
    assert persisted["aggregate_inventory"]["changed_file_count"] == 3
    assert persisted["aggregate_inventory"]["csv_semantic_payloads_unchanged"]
    assert persisted["aggregate_inventory"]["ignored_appledouble_sidecars_count"] == 1
    assert len(persisted["ignored_appledouble_sidecars"]) == 1
    assert persisted["config_binding"]["manifest_config_sha256_before"] == (
        execution_config_hash
    )
    assert persisted["config_binding"]["manifest_config_sha256_after"] == (
        corrected_config_hash
    )
    assert persisted["config_binding"]["metadata_only_correction"][
        "computational_effect"
    ] is False
    for entry in persisted["files"]:
        for side in ("old", "new"):
            assert set(entry[side]) == {
                "sha256",
                "bytes",
                "rows",
                "semantic_payload_sha256",
                "observed_utc",
            }

    run_hash = hashlib.sha256(runs.read_bytes()).hexdigest()
    prediction_hash = hashlib.sha256(prediction.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with runner._exclusive_manifest_lock(manifest_path):
        second = runner._run_upgrade_only(args)
    assert second["aggregate_inventory"]["changed_file_count"] == 0
    assert hashlib.sha256(runs.read_bytes()).hexdigest() == run_hash
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == prediction_hash
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_hash


def test_upgrade_rejects_any_config_change_beyond_reviewed_metadata_correction(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("audit:\n  scaffold_overlap: true\nseeds: [1]\n", encoding="utf-8")
    manifest = _manifest(config)
    config.write_text("audit:\n  scaffold_overlap: false\nseeds: [2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs.*more than"):
        runner._resolve_upgrade_config_binding(manifest, config_path=config)


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ({"task": "", "model": "rf", "seed": "1"}, "task is blank"),
        ({"task": "AMES", "model": "", "seed": "1"}, "model is blank"),
        ({"task": "AMES", "model": "rf", "seed": "1.0"}, "strict positive integer"),
        ({"task": "AMES", "model": "rf", "seed": " 1"}, "strict positive integer"),
    ],
)
def test_legacy_run_identity_is_fail_closed(
    tmp_path: Path, identity: dict[str, str], message: str
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("x\n", encoding="utf-8")
    manifest = _manifest(config)
    fields = [field for field in runner.FIELDNAMES if field not in runner.PROVENANCE_FIELDS]
    row = _run_row(fields)
    row.update(identity)
    path = tmp_path / "runs.csv"
    _write_csv(path, fields, row)
    with pytest.raises(ValueError, match=message):
        runner._plan_run_table_provenance_upgrade(
            path,
            manifest=manifest,
            split_source="locked_cache",
            observed_utc=runner._utc_now(),
        )


@pytest.mark.parametrize("content", [b"", b"task,model,seed\n"])
def test_blank_or_header_only_run_table_is_rejected(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "runs.csv"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="blank|no data rows"):
        runner._read_csv_strict(path, artifact="sensitivity run table")


def test_truncated_prediction_and_noninteger_seed_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    fields = list(runner.PREDICTION_CORE_FIELDS)
    # A missing trailing value becomes None in DictReader and must fail before upgrade.
    path.write_text(
        ",".join(fields) + "\n" + "AMES,rf,1,common_cleaned_test,CCO,1,0.7\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="truncated"):
        runner._read_csv_strict(path, artifact="sensitivity prediction table")

    row = _prediction_row(fields)
    row["seed"] = "1e0"
    _write_csv(path, fields, row)
    config = tmp_path / "config.yaml"
    config.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict positive integer"):
        runner._plan_prediction_provenance_upgrades(
            tmp_path,
            manifest=_manifest(config),
            split_source="locked_cache",
            observed_utc=runner._utc_now(),
        )


def test_resume_rejects_misordered_final_schema_and_append_never_misaligns(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("x\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(config, save_predictions=False)), encoding="utf-8"
    )
    fields = list(reversed(runner.FIELDNAMES))
    runs = tmp_path / "runs.csv"
    _write_csv(runs, fields, _run_row(fields))
    with pytest.raises(ValueError, match="exact final run-table FIELDNAMES order"):
        runner._require_canonical_resume_artifacts(
            out=runs,
            predictions_dir=tmp_path / "predictions",
            manifest_path=manifest_path,
            split_source="locked_cache",
            save_predictions=False,
        )
    with pytest.raises(ValueError, match="exact FIELDNAMES order"):
        runner._append_csv(runs, _run_row(runner.FIELDNAMES))

    runner._upgrade_existing_run_table_provenance(
        runs, manifest_path=manifest_path, split_source="locked_cache"
    )
    assert pd.read_csv(runs, nrows=0).columns.tolist() == runner.FIELDNAMES


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(selected_tasks_for_cleaned_rerun=[]),
        lambda config: config.update(models=["rf", "RF"]),
        lambda config: config.update(models=["unknown"]),
        lambda config: config.update(seeds=[1, 1]),
        lambda config: config.update(seeds=[1.0]),
        lambda config: config["fingerprint"].update(radius=0),
        lambda config: config["fingerprint"].update(n_bits="2048.0"),
        lambda config: config.update(threads_per_model=False),
    ],
)
def test_config_validation_rejects_empty_duplicate_unknown_or_coerced_values(
    mutation,
) -> None:
    config = {
        "selected_tasks_for_cleaned_rerun": ["AMES"],
        "models": ["rf"],
        "seeds": [1],
        "fingerprint": {"radius": 2, "n_bits": 2048},
        "threads_per_model": 1,
    }
    mutation(config)
    with pytest.raises(ValueError):
        runner._resolve_and_validate_configuration(
            config,
            tasks_override=None,
            models_override=None,
            seeds_override=None,
            radius_override=None,
            n_bits_override=None,
            threads_override=None,
        )


def test_explicit_empty_cli_override_does_not_fall_back_to_config() -> None:
    config = {
        "selected_tasks_for_cleaned_rerun": ["AMES"],
        "models": ["rf"],
        "seeds": [1],
        "fingerprint": {"radius": 2, "n_bits": 2048},
    }
    with pytest.raises(ValueError, match="tasks must be a non-empty list"):
        runner._resolve_and_validate_configuration(
            config,
            tasks_override=[],
            models_override=None,
            seeds_override=None,
            radius_override=None,
            n_bits_override=None,
            threads_override=None,
        )
