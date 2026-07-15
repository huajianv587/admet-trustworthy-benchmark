from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from admet_benchmark.data_loader import TDC_TASK_META
from admet_benchmark.smiles_compat import SMILES_COMPATIBILITY_POLICY
from experiments.precheck_final_neural_artifacts import (
    SOLUBILITY_COMPATIBILITY_REPAIRS,
    audit_final_neural_artifacts,
)


def _payload(test_smiles=("CCC", "CCCC")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid", "valid", "test", "test"],
            "smiles": ["C", "CC", *test_smiles],
            "label": [0, 1, 0, 1],
            "prediction": [0.2, 0.8, 0.1, 0.9],
        }
    )


def _write_run(
    root: Path,
    model: str,
    *,
    task: str = "AMES",
    checkpoint: bool = True,
    test_smiles=("CCC", "CCCC"),
) -> None:
    stem = f"{task}_{model}_seed1"
    meta = TDC_TASK_META[task]
    for directory in ("predictions", "metrics", "checkpoints", "configs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _payload(test_smiles).to_csv(root / "predictions" / f"{stem}.csv", index=False)
    metrics = {
        "status": "complete",
        "task": task,
        "model": model,
        "seed": 1,
        "task_type": meta["task_type"],
        "metric_name": meta["metric"],
        "valid_metric": 0.2 if meta["metric"] == "mae" else 1.0,
        "test_metric": 0.1 if meta["metric"] == "mae" else 1.0,
        "prediction_rows": 4,
    }
    if model == "chemprop_dmpnn":
        metrics.update(
            {
                "chemprop_version": "2.2.1",
                "official_test_preserved": True,
            }
        )
    (root / "metrics" / f"{stem}.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    if checkpoint:
        (root / "checkpoints" / f"{stem}.pt").write_bytes(b"checkpoint")
    if model == "chemprop_dmpnn":
        config = {
            "status": "complete",
            "task": task,
            "model": model,
            "seed": 1,
            "task_type": meta["task_type"],
            "official_metric": meta["metric"],
            "software": {"chemprop": "2.2.1"},
            "training": {"data_seed": 1, "pytorch_seed": 1},
            "architecture": {
                "directed_bond_messages": True,
                "atom_messages": False,
                "undirected": False,
            },
            "data": {
                "valid_n": 2,
                "test_n": 2,
                "official_test_preserved": True,
                "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
                "model_input_repairs_n": (
                    2 if task == "Solubility_AqSolDB" else 0
                ),
                "model_input_repairs": (
                    [dict(value) for value in SOLUBILITY_COMPATIBILITY_REPAIRS]
                    if task == "Solubility_AqSolDB"
                    else []
                ),
                "train_sha256": "a" * 64,
                "valid_sha256": "b" * 64,
                "test_sha256": "c" * 64,
                "source_train_sha256": "a" * 64,
                "source_valid_sha256": "b" * 64,
                "source_test_sha256": (
                    "f" * 64 if task == "Solubility_AqSolDB" else "c" * 64
                ),
            },
        }
        (root / "configs" / f"{stem}.json").write_text(
            json.dumps(config), encoding="utf-8"
        )


def _audit(
    gin: Path,
    chemprop: Path,
    *,
    tasks=("AMES",),
    gin_config_path: Path | None = None,
    chemprop_config_path: Path | None = None,
):
    return audit_final_neural_artifacts(
        gin_dir=gin,
        chemprop_dir=chemprop,
        data_dir=Path("unused"),
        gin_config_path=gin_config_path,
        chemprop_config_path=chemprop_config_path,
        tasks=tasks,
        seeds=(1,),
        verify_official_splits=False,
    )


def test_minimal_complete_two_model_matrix_passes(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn")

    audit, legacy, gate = _audit(gin, chemprop)

    assert gate["status"] == "GO"
    assert len(audit) == 2
    assert audit["included"].all()
    assert legacy.empty


def test_missing_checkpoint_is_hold_with_exact_reason(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin", checkpoint=False)
    _write_run(chemprop, "chemprop_dmpnn")

    audit, _, gate = _audit(gin, chemprop)

    gin_row = audit[audit["model"] == "gin"].iloc[0]
    assert gate["status"] == "HOLD"
    assert gin_row["reason"] == "missing_or_empty_neural_checkpoint"
    assert gate["complete_by_model"]["gin"] == 0


def test_cross_model_sample_mismatch_and_legacy_exclusion_are_audited(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn", test_smiles=("CCN", "CCCC"))
    (gin / "predictions" / "AMES_dmpnn_seed1.csv").write_text(
        "split,smiles,label,prediction\n", encoding="utf-8"
    )

    _, legacy, gate = _audit(gin, chemprop)

    assert gate["status"] == "HOLD"
    assert gate["sample_alignment"]["inconsistent_test_fingerprints"]
    assert len(legacy) == 1
    assert legacy["excluded_from_formal_220"].all()
    assert gate["legacy_exclusion_ok"]


def test_solubility_requires_exact_two_inchi_audited_repairs(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    task = "Solubility_AqSolDB"
    _write_run(gin, "gin", task=task)
    _write_run(chemprop, "chemprop_dmpnn", task=task)

    audit, _, gate = _audit(gin, chemprop, tasks=(task,))
    assert gate["status"] == "GO"
    assert audit["representation_compatibility_ok"].all()

    config_path = (
        chemprop
        / "configs"
        / "Solubility_AqSolDB_chemprop_dmpnn_seed1.json"
    )
    config = json.loads(config_path.read_text())
    config["data"]["model_input_repairs"][1]["sample_index"] = 1161
    config_path.write_text(json.dumps(config))

    audit, _, gate = _audit(gin, chemprop, tasks=(task,))
    row = audit[audit["model"] == "chemprop_dmpnn"].iloc[0]
    assert gate["status"] == "HOLD"
    assert row["reason"] == "chemprop_solubility_exact_compatibility_mapping_mismatch"


def test_non_solubility_repairs_and_missing_source_hash_are_hard_holds(
    tmp_path: Path,
):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn")
    config_path = chemprop / "configs" / "AMES_chemprop_dmpnn_seed1.json"
    config = json.loads(config_path.read_text())
    config["data"].pop("source_test_sha256")
    config_path.write_text(json.dumps(config))

    audit, _, gate = _audit(gin, chemprop)
    row = audit[audit["model"] == "chemprop_dmpnn"].iloc[0]
    assert gate["status"] == "HOLD"
    assert row["reason"] == "chemprop_partial_compatibility_manifest"

    config["data"]["source_test_sha256"] = "f" * 64
    config["data"]["model_input_repairs_n"] = 1
    config["data"]["model_input_repairs"] = [
        dict(SOLUBILITY_COMPATIBILITY_REPAIRS[0])
    ]
    config_path.write_text(json.dumps(config))
    audit, _, gate = _audit(gin, chemprop)
    row = audit[audit["model"] == "chemprop_dmpnn"].iloc[0]
    assert gate["status"] == "HOLD"
    assert row["reason"] == "chemprop_unexpected_compatibility_repairs_for_task"


def test_gin_global_config_must_declare_locked_policy(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn")
    gin_config = tmp_path / "gin.yaml"
    gin_config.write_text("data:\n  model_input_representation_policy: wrong\n")

    audit, _, gate = _audit(gin, chemprop, gin_config_path=gin_config)
    row = audit[audit["model"] == "gin"].iloc[0]
    assert gate["status"] == "HOLD"
    assert row["reason"] == "gin_config_compatibility_policy_missing_or_mismatch"
    assert not gate["representation_compatibility"]["gin_global_config_ok"]


def test_pre_solubility_nonrepair_schema_remains_auditable(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn")
    config_path = chemprop / "configs" / "AMES_chemprop_dmpnn_seed1.json"
    config = json.loads(config_path.read_text())
    for field in (
        "model_input_representation_policy",
        "model_input_repairs_n",
        "model_input_repairs",
        "source_train_sha256",
        "source_valid_sha256",
        "source_test_sha256",
    ):
        config["data"].pop(field)
    config_path.write_text(json.dumps(config))

    audit, _, gate = _audit(gin, chemprop)
    assert gate["status"] == "GO"
    row = audit[audit["model"] == "chemprop_dmpnn"].iloc[0]
    assert row["representation_compatibility_ok"]
    assert row["compatibility_manifest_schema"] == "legacy_no_repair_hash_identity"


def test_chemprop_global_config_must_declare_locked_policy(tmp_path: Path):
    gin = tmp_path / "gin"
    chemprop = tmp_path / "chemprop"
    _write_run(gin, "gin")
    _write_run(chemprop, "chemprop_dmpnn")
    chemprop_config = tmp_path / "chemprop.yaml"
    chemprop_config.write_text("data:\n  model_input_representation_policy: wrong\n")

    audit, _, gate = _audit(
        gin, chemprop, chemprop_config_path=chemprop_config
    )
    row = audit[audit["model"] == "chemprop_dmpnn"].iloc[0]
    assert gate["status"] == "HOLD"
    assert row["reason"] == (
        "chemprop_global_config_compatibility_policy_missing_or_mismatch"
    )
    assert not gate["representation_compatibility"]["chemprop_global_config_ok"]
