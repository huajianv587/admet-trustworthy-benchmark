from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.data_loader import LOCKED_CACHE_SPLIT_ALGORITHM
from admet_benchmark.sensitivity import CleaningPolicy
from admet_benchmark.smiles_compat import (
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)
from experiments.audit_cleaned_sensitivity_predictions import (
    METRIC_ATOL,
    METRIC_REPLAY_POLICY_ID,
    METRIC_RTOL,
    audit_prediction_exports,
    main,
    recompute_prediction_inventory_sha256,
)
from experiments.run_cleaned_sensitivity_cpu import _metric_values, _primary_effect
from experiments.summarize_cleaned_sensitivity import _stable_json_sha256


def _inputs(tmp_path: Path):
    tasks = ["AMES"]
    models = ["rf", "xgboost"]
    seeds = [1, 2]
    radius = 2
    n_bits = 128
    config = tmp_path / "sensitivity.yaml"
    config.write_text(
        json.dumps(
            {
                "selected_tasks_for_cleaned_rerun": tasks,
                "models": models,
                "seeds": seeds,
                "fingerprint": {"radius": radius, "n_bits": n_bits},
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    task_dir = data_dir / "admet_group" / "ames"
    task_dir.mkdir(parents=True)
    # Sixteen balanced, unique training molecules are sufficient for the exact
    # locked 12.5% stratified split at every fixture seed.
    pd.DataFrame(
        {
            "Drug": ["C" * length for length in range(1, 17)],
            "Y": [value % 2 for value in range(16)],
        }
    ).to_csv(task_dir / "train_val.csv", index=False)
    pd.DataFrame({"Drug": ["CCO", "CCN"], "Y": [0, 1]}).to_csv(
        task_dir / "test.csv", index=False
    )
    file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": "paired-common-test-v1",
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "cpu_only": True,
                "save_predictions": True,
                "config_path": str(config.resolve()),
                "config_sha256": file_hash(config),
                "tasks": tasks,
                "models": models,
                "seeds": seeds,
                "fingerprint": {"radius": radius, "n_bits": n_bits},
                "threads_per_model": 1,
                "cleaning_policy": CleaningPolicy().to_dict(),
                "canonicalization_input_policy": (
                    "apply_enumerated_parser_compatibility_before_rdkit_canonicalization"
                ),
                "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
                "parser_compatibility_repairs_count_as_cleaning_deletions": False,
                "compatibility_mapping_sha256": _stable_json_sha256(
                    compatibility_mapping
                ),
                "compatibility_source_occurrences": {"AMES": 0},
                "input_sha256": {
                    "AMES": {
                        "train_val.csv": file_hash(task_dir / "train_val.csv"),
                        "test.csv": file_hash(task_dir / "test.csv"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    run_rows = []
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for model in models:
        for seed in seeds:
            run_rows.append(
                {
                    "task": "AMES",
                    "task_type": "classification",
                    "official_metric": "auroc",
                    "model": model,
                    "seed": seed,
                    "split_source": "locked_cache",
                    "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                    "radius": radius,
                    "n_bits": n_bits,
                    "threads_per_model": 1,
                    "common_test_n": 2,
                    "primary_official_common_test": 1.0,
                    "primary_cleaned_common_test": 1.0,
                    "primary_raw_delta_cleaned_minus_official": 0.0,
                    "primary_oriented_delta": 0.0,
                    "primary_relative_oriented_delta": 0.0,
                    "official_common_test_auroc": 1.0,
                    "cleaned_common_test_auroc": 1.0,
                }
            )
            pd.DataFrame(
                {
                    "task": ["AMES", "AMES"],
                    "model": [model, model],
                    "seed": [seed, seed],
                    "split_source": ["locked_cache", "locked_cache"],
                    "split_algorithm": [
                        LOCKED_CACHE_SPLIT_ALGORITHM,
                        LOCKED_CACHE_SPLIT_ALGORITHM,
                    ],
                    "evaluation_set": ["common_cleaned_test", "common_cleaned_test"],
                    "canonical_smiles": ["CCO", "CCN"],
                    "label": [0.0, 1.0],
                    "official_prediction": [0.2, 0.8],
                    "cleaned_prediction": [0.1, 0.9],
                }
            ).to_csv(predictions / f"ames_{model}_seed{seed}_paired.csv", index=False)
    runs = tmp_path / "runs.csv"
    pd.DataFrame(run_rows).to_csv(runs, index=False)
    return runs, predictions, manifest, config, data_dir, tasks, models, seeds


def _audit(inputs):
    runs, predictions, manifest, config, data_dir, tasks, models, seeds = inputs
    return audit_prediction_exports(
        runs_path=runs,
        predictions_dir=predictions,
        manifest_path=manifest,
        config_path=config,
        data_dir=data_dir,
        tasks=tasks,
        models=models,
        seeds=seeds,
    )


def _xgboost_regression_inputs(tmp_path: Path):
    task = "PPBR_AZ"
    models = ["xgboost"]
    seeds = [3]
    radius = 2
    n_bits = 128
    config = tmp_path / "sensitivity.yaml"
    config.write_text(
        json.dumps(
            {
                "selected_tasks_for_cleaned_rerun": [task],
                "models": models,
                "seeds": seeds,
                "fingerprint": {"radius": radius, "n_bits": n_bits},
            }
        ),
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    task_dir = data_dir / "admet_group" / task.lower()
    task_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Drug": ["C" * length for length in range(1, 17)],
            "Y": np.linspace(0.25, 4.0, 16),
        }
    ).to_csv(task_dir / "train_val.csv", index=False)
    labels = np.array([1.0, 2.0, 3.0], dtype=float)
    test_smiles = ["CCO", "CCN", "CCCl"]
    pd.DataFrame({"Drug": test_smiles, "Y": labels}).to_csv(
        task_dir / "test.csv", index=False
    )

    file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    compatibility_mapping = {
        source: LEGACY_VALENCE_REPAIRS[source]
        for source in sorted(LEGACY_VALENCE_REPAIRS)
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": "paired-common-test-v1",
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "cpu_only": True,
                "save_predictions": True,
                "config_path": str(config.resolve()),
                "config_sha256": file_hash(config),
                "tasks": [task],
                "models": models,
                "seeds": seeds,
                "fingerprint": {"radius": radius, "n_bits": n_bits},
                "threads_per_model": 1,
                "cleaning_policy": CleaningPolicy().to_dict(),
                "canonicalization_input_policy": (
                    "apply_enumerated_parser_compatibility_before_rdkit_canonicalization"
                ),
                "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
                "parser_compatibility_repairs_count_as_cleaning_deletions": False,
                "compatibility_mapping_sha256": _stable_json_sha256(
                    compatibility_mapping
                ),
                "compatibility_source_occurrences": {task: 0},
                "input_sha256": {
                    task: {
                        "train_val.csv": file_hash(task_dir / "train_val.csv"),
                        "test.csv": file_hash(task_dir / "test.csv"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # These mimic the native float32 arrays returned by XGBoost. Pandas writes
    # their compact decimal representation; reading the CSV promotes it to
    # float64, which is precisely the production round-trip being audited.
    official_prediction = np.array(
        [9.223322, 2.123456, 7.765432], dtype=np.float32
    )
    cleaned_prediction = np.array(
        [8.345678, 2.234567, 6.654321], dtype=np.float32
    )
    official_mae = _metric_values(
        "regression", labels, official_prediction
    )["mae"]
    cleaned_mae = _metric_values(
        "regression", labels, cleaned_prediction
    )["mae"]
    raw_delta, oriented_delta, relative_delta = _primary_effect(
        "mae", official_mae, cleaned_mae
    )
    runs = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "task": task,
                "task_type": "regression",
                "official_metric": "mae",
                "model": "xgboost",
                "seed": 3,
                "split_source": "locked_cache",
                "split_algorithm": LOCKED_CACHE_SPLIT_ALGORITHM,
                "radius": radius,
                "n_bits": n_bits,
                "threads_per_model": 1,
                "common_test_n": len(labels),
                "primary_official_common_test": official_mae,
                "primary_cleaned_common_test": cleaned_mae,
                "primary_raw_delta_cleaned_minus_official": raw_delta,
                "primary_oriented_delta": oriented_delta,
                "primary_relative_oriented_delta": relative_delta,
                "official_common_test_mae": official_mae,
                "cleaned_common_test_mae": cleaned_mae,
            }
        ]
    ).to_csv(runs, index=False)

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    prediction_path = predictions / "ppbr_az_xgboost_seed3_paired.csv"
    pd.DataFrame(
        {
            "task": [task] * len(labels),
            "model": ["xgboost"] * len(labels),
            "seed": [3] * len(labels),
            "split_source": ["locked_cache"] * len(labels),
            "split_algorithm": [LOCKED_CACHE_SPLIT_ALGORITHM] * len(labels),
            "evaluation_set": ["common_cleaned_test"] * len(labels),
            "canonical_smiles": test_smiles,
            "label": labels,
            "official_prediction": official_prediction,
            "cleaned_prediction": cleaned_prediction,
        }
    ).to_csv(prediction_path, index=False)
    return (
        runs,
        predictions,
        manifest,
        config,
        data_dir,
        [task],
        models,
        seeds,
    )


def test_prediction_audit_passes_complete_source_bound_matrix(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    audit, gate = _audit(inputs)
    assert gate["status"] == "GO"
    assert gate["complete_expected_matrix"] is True
    assert gate["sample_label_identity_aligned_within_task"] is True
    assert gate["source_common_test_reconstructed_exactly"] is True
    assert gate["prediction_metrics_recomputed_match_runs"] is True
    assert len(audit) == 4
    assert audit["finite_labels_and_predictions"].all()
    assert audit["source_common_test_exact"].all()
    assert audit["paired_delta_matches_run"].all()
    assert set(audit["metric_replay_policy_id"]) == {
        METRIC_REPLAY_POLICY_ID
    }
    assert set(audit.loc[audit["model"] == "rf", "metric_prediction_dtype"]) == {
        "float64"
    }
    assert set(
        audit.loc[audit["model"] == "xgboost", "metric_prediction_dtype"]
    ) == {"float32"}
    policy = gate["metric_replay_policy"]
    assert policy["prediction_dtype_by_model"] == {
        "rf": "float64",
        "xgboost": "float32",
        "lightgbm": "float64",
    }
    assert policy["comparison_tolerances_unchanged"] == {
        "rtol": METRIC_RTOL,
        "atol": METRIC_ATOL,
    }
    policy_payload = {key: value for key, value in policy.items() if key != "sha256"}
    assert policy["sha256"] == _stable_json_sha256(policy_payload)
    serialized = audit.attrs["_serialized_csv_bytes"]
    assert hashlib.sha256(serialized).hexdigest() == gate["audit_csv"]["sha256"]
    assert len(serialized) == gate["audit_csv"]["bytes"]
    assert gate["audit_csv"]["rows"] == 4
    assert gate["audit_csv"]["columns"] == list(audit.columns)
    assert "metric_replay_policy_id" in gate["audit_csv"]["columns"]
    assert "metric_prediction_dtype" in gate["audit_csv"]["columns"]


def test_xgboost_regression_replays_native_float32_before_metric_gate(
    tmp_path: Path,
) -> None:
    inputs = _xgboost_regression_inputs(tmp_path)
    runs, predictions = inputs[:2]
    run = pd.read_csv(runs).iloc[0]
    exported = pd.read_csv(predictions / "ppbr_az_xgboost_seed3_paired.csv")

    # A naive float64 replay is outside the unchanged gate tolerance. Casting
    # the exported decimals back to XGBoost's native float32 exactly recovers
    # the metric inputs used by the runner.
    labels = exported["label"].to_numpy(dtype=float)
    parsed_float64_mae = _metric_values(
        "regression",
        labels,
        exported["official_prediction"].to_numpy(dtype=float),
    )["mae"]
    assert not np.isclose(
        parsed_float64_mae,
        run["primary_official_common_test"],
        rtol=METRIC_RTOL,
        atol=METRIC_ATOL,
    )

    audit, gate = _audit(inputs)
    assert gate["status"] == "GO"
    assert gate["prediction_metrics_recomputed_match_runs"] is True
    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["metric_replay_policy_id"] == METRIC_REPLAY_POLICY_ID
    assert row["metric_prediction_dtype"] == "float32"
    assert bool(row["official_metric_matches_run"]) is True
    assert bool(row["cleaned_metric_matches_run"]) is True
    assert bool(row["paired_delta_matches_run"]) is True
    assert row["recomputed_official_common_test"] == pytest.approx(
        run["primary_official_common_test"], rel=0.0, abs=0.0
    )
    assert row["recomputed_cleaned_common_test"] == pytest.approx(
        run["primary_cleaned_common_test"], rel=0.0, abs=0.0
    )


def test_xgboost_float32_replay_does_not_mask_substantive_tampering(
    tmp_path: Path,
) -> None:
    inputs = _xgboost_regression_inputs(tmp_path)
    path = inputs[1] / "ppbr_az_xgboost_seed3_paired.csv"
    changed = pd.read_csv(path)
    changed.loc[0, "official_prediction"] += 0.25
    changed.to_csv(path, index=False)

    audit, gate = _audit(inputs)
    assert gate["status"] == "HOLD"
    assert gate["prediction_metrics_recomputed_match_runs"] is False
    assert audit.iloc[0]["metric_prediction_dtype"] == "float32"
    assert bool(audit.iloc[0]["official_metric_matches_run"]) is False
    assert any(
        "official metric does not match run table" in failure
        for failure in gate["failures"]
    )


def test_appledouble_prediction_sidecars_are_ignored_but_recorded(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    predictions = inputs[1]
    (predictions / "._ames_rf_seed1_paired.csv").write_bytes(b"appledouble\x00rf")
    (predictions / "._ames_xgboost_seed2_paired.csv").write_bytes(
        b"appledouble\x00xgb"
    )
    audit, gate = _audit(inputs)
    assert gate["status"] == "GO"
    assert gate["observed_prediction_files"] == 4
    assert len(audit) == 4
    ignored = gate["ignored_appledouble_sidecars"]
    assert ignored["count"] == 2
    assert ignored["policy"].startswith("ignore-appledouble")
    assert len(ignored["names_sha256"]) == 64


def test_prediction_audit_holds_cross_seed_identity_change(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    predictions = inputs[1]
    path = predictions / "ames_rf_seed2_paired.csv"
    changed = pd.read_csv(path)
    changed.loc[0, "canonical_smiles"] = "CCC"
    changed.to_csv(path, index=False)
    _, gate = _audit(inputs)
    assert gate["status"] == "HOLD"
    assert gate["sample_label_identity_aligned_within_task"] is False
    assert gate["source_common_test_reconstructed_exactly"] is False
    assert any("identity mismatch" in failure for failure in gate["failures"])


def test_internally_consistent_wrong_samples_are_rejected_by_source_rebuild(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    predictions = inputs[1]
    for path in predictions.glob("*.csv"):
        changed = pd.read_csv(path)
        changed["canonical_smiles"] = ["CCC", "CCCC"]
        changed.to_csv(path, index=False)
    audit, gate = _audit(inputs)
    assert gate["status"] == "HOLD"
    # All prediction files agree with one another; only source reconstruction
    # can detect this coordinated replacement.
    assert gate["sample_label_identity_aligned_within_task"] is True
    assert gate["source_common_test_reconstructed_exactly"] is False
    assert not audit["source_common_test_exact"].any()
    assert any(
        "source common-test samples/labels/order mismatch" in failure
        for failure in gate["failures"]
    )


def test_prediction_metric_mismatch_against_run_table_is_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    path = inputs[1] / "ames_rf_seed1_paired.csv"
    changed = pd.read_csv(path)
    changed["official_prediction"] = [0.8, 0.2]
    changed.to_csv(path, index=False)
    audit, gate = _audit(inputs)
    row = audit[
        (audit["task"] == "ames")
        & (audit["model"] == "rf")
        & (audit["seed"] == 1)
    ].iloc[0]
    assert gate["status"] == "HOLD"
    assert gate["prediction_metrics_recomputed_match_runs"] is False
    assert bool(row["official_metric_matches_run"]) is False
    assert any(
        "official metric does not match run table" in failure
        for failure in gate["failures"]
    )


def test_noninteger_run_seed_is_not_silently_truncated(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    runs = inputs[0]
    changed = pd.read_csv(runs)
    changed["seed"] = changed["seed"].astype(float)
    changed.loc[0, "seed"] = 1.5
    changed.to_csv(runs, index=False)
    _, gate = _audit(inputs)
    assert gate["status"] == "HOLD"
    assert any("must be an integer" in failure for failure in gate["failures"])


def test_inventory_digest_exposes_post_audit_tampering(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    predictions = inputs[1]
    audit, gate = _audit(inputs)
    bound = gate["prediction_inventory_sha256"]
    assert recompute_prediction_inventory_sha256(audit, predictions) == bound
    path = predictions / "ames_rf_seed1_paired.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    assert recompute_prediction_inventory_sha256(audit, predictions) != bound


def test_cli_atomically_writes_the_exact_audit_bytes_bound_by_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, predictions, manifest, config, data_dir, *_ = _inputs(tmp_path)
    out_dir = tmp_path / "audit-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_cleaned_sensitivity_predictions.py",
            "--runs",
            str(runs),
            "--predictions-dir",
            str(predictions),
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    audit_path = out_dir / "cleaned_sensitivity_prediction_audit.csv"
    gate_path = out_dir / "cleaned_sensitivity_prediction_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    audit_bytes = audit_path.read_bytes()
    assert hashlib.sha256(audit_bytes).hexdigest() == gate["audit_csv"]["sha256"]
    assert len(audit_bytes) == gate["audit_csv"]["bytes"]
    assert gate["audit_csv"]["rows"] == 4
    assert not list(out_dir.glob("*.tmp.*"))
