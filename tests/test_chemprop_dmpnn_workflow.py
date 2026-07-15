from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.data_loader import OfficialSplit
from experiments import run_chemprop_dmpnn as workflow


def _split(metric: str = "auroc", task_type: str = "classification") -> OfficialSplit:
    def frame(offset: int, n: int) -> pd.DataFrame:
        labels = (
            [float((offset + i) % 2) for i in range(n)]
            if task_type == "classification"
            else [float(offset + i) for i in range(n)]
        )
        return pd.DataFrame(
            {
                "Drug": ["C" * (offset + i + 1) for i in range(n)],
                "Y": labels,
            }
        )

    return OfficialSplit(
        name="synthetic",
        train=frame(0, 6),
        valid=frame(6, 4),
        test=frame(10, 4),
        task_type=task_type,
        metric=metric,
        category="test",
    )


def test_command_is_official_directed_bond_message_dmpnn() -> None:
    command = workflow.build_train_command(
        Path("/venv/bin/chemprop"),
        Path("/tmp/data.csv"),
        Path("/tmp/output"),
        workflow.selection_policy("classification", "auprc"),
        seed=5,
        epochs=100,
        patience=20,
        batch_size=64,
        num_workers=0,
        accelerator="gpu",
    )
    assert command[1] == "train"
    assert "--atom-messages" not in command
    assert "--undirected" not in command
    assert command[command.index("--pytorch-seed") + 1] == "5"
    assert command[command.index("--data-seed") + 1] == "5"
    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--patience") + 1] == "20"
    assert command[command.index("--tracking-metric") + 1] == "prc"


def test_spearman_tasks_disclose_mae_checkpoint_proxy() -> None:
    policy = workflow.selection_policy("regression", "spearman")
    assert policy.chemprop_metric == "mae"
    assert policy.tracking_metric == "mae"
    assert policy.public_name == "validation_mae_proxy_for_spearman"
    assert "cannot track Spearman" in policy.rationale


def test_completion_marker_is_split_source_aware_with_locked_legacy_default(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "metrics.json"
    marker.write_text(
        json.dumps({"status": "complete", "model": workflow.MODEL_ID}),
        encoding="utf-8",
    )
    assert workflow.is_complete(marker, split_source="locked_cache") is True
    assert workflow.is_complete(marker, split_source="pytdc") is False

    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": workflow.MODEL_ID,
                "split_source": "pytdc",
            }
        ),
        encoding="utf-8",
    )
    assert workflow.is_complete(marker, split_source="locked_cache") is False
    assert workflow.is_complete(marker, split_source="pytdc") is True


def test_locked_csv_preserves_existing_split_membership_and_order(tmp_path: Path) -> None:
    split = _split()
    combined = tmp_path / "combined.csv"
    manifest = workflow.prepare_locked_data(
        split, combined, tmp_path / "valid.csv", tmp_path / "test.csv"
    )
    frame = pd.read_csv(combined)
    assert frame["split"].tolist() == ["train"] * 6 + ["val"] * 4 + ["test"] * 4
    assert frame["smiles"].tolist() == (
        split.train["Drug"].tolist()
        + split.valid["Drug"].tolist()
        + split.test["Drug"].tolist()
    )
    assert manifest["official_test_preserved"] is True
    assert (manifest["train_n"], manifest["valid_n"], manifest["test_n"]) == (6, 4, 4)
    assert all(len(manifest[f"{name}_sha256"]) == 64 for name in ["train", "valid", "test"])


def test_locked_csv_repairs_legacy_valence_without_dropping_official_rows(
    tmp_path: Path,
) -> None:
    split = _split(metric="mae", task_type="regression")
    legacy = "CC1=CC=C[NH+2]([O-])[CH-]1"
    split.test.loc[2, "Drug"] = legacy
    combined = tmp_path / "combined.csv"
    valid = tmp_path / "valid.csv"
    test = tmp_path / "test.csv"

    manifest = workflow.prepare_locked_data(split, combined, valid, test)
    locked = pd.read_csv(combined)
    test_inputs = pd.read_csv(test)

    assert len(locked) == len(split.train) + len(split.valid) + len(split.test)
    assert test_inputs["smiles"].tolist()[2] == "CC1=C[NH+]([O])[CH-]C=C1"
    assert manifest["official_test_preserved"] is True
    assert manifest["official_split_membership_preserved"] is True
    assert manifest["model_input_repairs_n"] == 1
    assert manifest["model_input_repairs"][0]["source_smiles"] == legacy
    assert manifest["model_input_repairs"][0]["sample_index"] == 2
    assert (
        manifest["model_input_repairs"][0]["standard_inchikey"]
        == "FHJYMRFPLWFYPD-UHFFFAOYSA-N"
    )
    assert manifest["source_test_sha256"] != manifest["test_sha256"]


def test_all_audited_compatibility_repairs_parse_with_rdkit() -> None:
    from rdkit import Chem
    from rdkit.Chem import inchi

    from admet_benchmark.smiles_compat import (
        LEGACY_VALENCE_REPAIR_INCHIKEYS,
        LEGACY_VALENCE_REPAIRS,
    )

    assert len(LEGACY_VALENCE_REPAIRS) == 2
    for source, value in LEGACY_VALENCE_REPAIRS.items():
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        assert inchi.MolToInchiKey(molecule) == LEGACY_VALENCE_REPAIR_INCHIKEYS[source]


def test_stubbed_run_writes_standard_artifacts_and_metrics_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split()
    fake_bin = tmp_path / "chemprop"
    fake_bin.write_text("#!/usr/bin/env python\n")
    fake_bin.chmod(0o755)

    monkeypatch.setattr(workflow, "resolve_chemprop_bin", lambda _: fake_bin)
    monkeypatch.setattr(
        workflow,
        "chemprop_runtime",
        lambda *_: {
            "chemprop": "2.2.1",
            "python": "3.12",
            "torch": "test",
            "cuda_runtime": None,
            "cuda_available": False,
            "chemprop_bin": str(fake_bin),
            "chemprop_python": "test",
        },
    )

    def fake_run(command: list[str], _log: Path, _env: dict[str, str]) -> None:
        mode = command[1]
        if mode == "train":
            output = Path(command[command.index("--output-dir") + 1])
            model_dir = output / "model_0"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").write_bytes(b"official-chemprop-stub")
            log_dir = model_dir / "trainer_logs" / "version_0"
            log_dir.mkdir(parents=True)
            pd.DataFrame(
                {"epoch": [0, 1], "val/roc": [0.5, 0.75]}
            ).to_csv(log_dir / "metrics.csv", index=False)
            return
        input_path = Path(command[command.index("--test-path") + 1])
        output_path = Path(command[command.index("--output") + 1])
        inputs = pd.read_csv(input_path)
        # Perfect ranking with both classes in each synthetic held-out split.
        labels = split.valid["Y"] if "valid" in input_path.name else split.test["Y"]
        inputs["target"] = np.asarray(labels, dtype=float) * 0.8 + 0.1
        inputs.to_csv(output_path, index=False)

    monkeypatch.setattr(workflow, "run_command", fake_run)
    output_dir = tmp_path / "results"
    result = workflow.train_single_run(
        task="stub_task",
        seed=3,
        output_dir=output_dir,
        chemprop_bin=str(fake_bin),
        accelerator="cpu",
        split_override=split,
    )
    assert result["status"] == "success"
    paths = workflow.artifact_paths(output_dir, "stub_task", 3)
    prediction = pd.read_csv(paths["prediction"])
    assert prediction.columns.tolist() == ["split", "smiles", "label", "prediction"]
    assert prediction["split"].value_counts().to_dict() == {"valid": 4, "test": 4}
    metrics = json.loads(paths["metrics"].read_text())
    config = json.loads(paths["config"].read_text())
    assert metrics["status"] == "complete"
    assert metrics["model"] == "chemprop_dmpnn"
    assert metrics["valid_metric"] == 1.0
    assert metrics["test_metric"] == 1.0
    assert config["architecture"]["message_type"] == "directed_bond_messages"
    assert config["architecture"]["atom_messages"] is False
    assert config["architecture"]["undirected"] is False
    assert workflow.is_complete(paths["metrics"])


def test_failed_run_never_writes_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split()
    fake_bin = tmp_path / "chemprop"
    fake_bin.write_text("#!/usr/bin/env python\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(workflow, "resolve_chemprop_bin", lambda _: fake_bin)
    monkeypatch.setattr(
        workflow,
        "chemprop_runtime",
        lambda *_: {"chemprop": "2.2.1"},
    )
    monkeypatch.setattr(
        workflow,
        "run_command",
        lambda *_: (_ for _ in ()).throw(RuntimeError("stub failure")),
    )
    output_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="stub failure"):
        workflow.train_single_run(
            task="stub_task",
            seed=4,
            output_dir=output_dir,
            chemprop_bin=str(fake_bin),
            accelerator="cpu",
            split_override=split,
        )
    paths = workflow.artifact_paths(output_dir, "stub_task", 4)
    assert not paths["metrics"].exists()
    assert workflow.is_complete(paths["metrics"]) is False
