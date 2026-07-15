#!/usr/bin/env python3
"""Train the locked, official Chemprop v2 D-MPNN baseline.

This workflow is intentionally separate from ``run_neural_a40.py``.  The
legacy model named ``dmpnn`` in that script is retained for provenance, while
this script uses Chemprop's directed *bond-message* implementation and the
unambiguous model id ``chemprop_dmpnn``.

Completion contract
-------------------
For each task/seed, the metrics JSON is written last and is the sole durable
completion marker.  A completed run has these public artifacts under
``results/neural_chemprop``::

    checkpoints/<task>_chemprop_dmpnn_seed<seed>.pt
    predictions/<task>_chemprop_dmpnn_seed<seed>.csv
    configs/<task>_chemprop_dmpnn_seed<seed>.json
    metrics/<task>_chemprop_dmpnn_seed<seed>.json
    logs/<task>_chemprop_dmpnn_seed<seed>.log

The official TDC test split returned by ``load_tdc_official`` is never split,
cleaned, canonicalized, or used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    OfficialSplit,
    SPLIT_SOURCE_CHOICES,
    TDC_TASK_META,
    TDC_TASKS,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.smiles_compat import (  # noqa: E402
    SMILES_COMPATIBILITY_POLICY,
    model_input_smiles,
    representation_repair_record,
)


MODEL_ID = "chemprop_dmpnn"
REQUIRED_CHEMPROP_VERSION = "2.2.1"
DEFAULT_OUTPUT_DIR = Path("results/neural_chemprop")
SPLIT_COLUMN = "split"
SMILES_COLUMN = "smiles"
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class SelectionPolicy:
    chemprop_task_type: str
    chemprop_metric: str
    tracking_metric: str
    public_name: str
    rationale: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(task: str, seed: int) -> str:
    return f"{task}_{MODEL_ID}_seed{seed}"


def artifact_paths(output_dir: Path, task: str, seed: int) -> dict[str, Path]:
    stem = run_id(task, seed)
    return {
        "checkpoint": output_dir / "checkpoints" / f"{stem}.pt",
        "prediction": output_dir / "predictions" / f"{stem}.csv",
        "config": output_dir / "configs" / f"{stem}.json",
        "metrics": output_dir / "metrics" / f"{stem}.json",
        "log": output_dir / "logs" / f"{stem}.log",
    }


def ensure_output_dirs(output_dir: Path) -> None:
    for name in ["checkpoints", "predictions", "configs", "metrics", "logs"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def is_complete(
    metrics_path: Path,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
) -> bool:
    """Return true only for a parseable, explicit final completion marker."""
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded_split_source = payload.get(
        "split_source", LOCKED_CACHE_SPLIT_SOURCE
    )
    return (
        payload.get("status") == "complete"
        and payload.get("model") == MODEL_ID
        and recorded_split_source == split_source
    )


def selection_policy(task_type: str, official_metric: str) -> SelectionPolicy:
    metric = str(official_metric).lower()
    if task_type == "classification" and metric == "auroc":
        return SelectionPolicy(
            "classification",
            "roc",
            "roc",
            "validation_auroc",
            "Chemprop ROC is the official validation metric; higher is better.",
        )
    if task_type == "classification" and metric == "auprc":
        return SelectionPolicy(
            "classification",
            "prc",
            "prc",
            "validation_auprc",
            "Chemprop PRC is the official validation metric; higher is better.",
        )
    if task_type == "regression" and metric == "mae":
        return SelectionPolicy(
            "regression",
            "mae",
            "mae",
            "validation_mae",
            "Chemprop MAE is the official validation metric; lower is better.",
        )
    if task_type == "regression" and metric == "spearman":
        return SelectionPolicy(
            "regression",
            "mae",
            "mae",
            "validation_mae_proxy_for_spearman",
            "Chemprop 2.2.1 cannot track Spearman. The checkpoint is selected "
            "by validation MAE only; final validation/test reporting uses the "
            "official Spearman metric.",
        )
    raise ValueError(f"Unsupported task/metric combination: {task_type}/{official_metric}")


def _frame_for_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    smiles, labels = extract_xy(frame)
    model_smiles = [model_input_smiles(value) for value in smiles]
    # Fail before launching Chemprop if a future dependency change introduces
    # another parser incompatibility. Silent row dropping is never allowed.
    from rdkit import Chem

    invalid = [
        (index, smiles[index])
        for index, value in enumerate(model_smiles)
        if Chem.MolFromSmiles(value) is None
    ]
    if invalid:
        preview = ", ".join(f"{index}:{value}" for index, value in invalid[:5])
        raise ValueError(
            f"{split} has unparseable SMILES after compatibility policy: {preview}"
        )
    result = pd.DataFrame(
        {
            # Prediction artifacts retain ``smiles`` from the OfficialSplit.
            # Only Chemprop's private model-input CSV receives compatibility
            # repairs for audited legacy valence notation.
            SMILES_COLUMN: pd.Series(
                model_smiles, dtype="string"
            ),
            TARGET_COLUMN: pd.to_numeric(labels, errors="raise").astype(float),
            SPLIT_COLUMN: split,
        }
    )
    if result[SMILES_COLUMN].isna().any() or (result[SMILES_COLUMN].str.len() == 0).any():
        raise ValueError(f"{split} contains missing or empty SMILES")
    if result[TARGET_COLUMN].isna().any():
        raise ValueError(f"{split} contains missing targets")
    return result


def prepare_locked_data(
    split: OfficialSplit,
    combined_path: Path,
    valid_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    """Write Chemprop inputs without changing official sample membership.

    Two legacy pyridine N-oxide strings in Solubility_AqSolDB are translated
    to equivalent parser-compatible model inputs.  The exact translation,
    split, and source index are recorded below; no row is removed or reordered.
    """
    train = _frame_for_split(split.train, "train")
    valid = _frame_for_split(split.valid, "val")
    test = _frame_for_split(split.test, "test")
    combined = pd.concat([train, valid, test], ignore_index=True)

    repairs: list[dict[str, str | int]] = []
    for split_name, source in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        source_smiles, _ = extract_xy(source)
        for sample_index, value in enumerate(source_smiles):
            record = representation_repair_record(
                value, split=split_name, sample_index=sample_index
            )
            if record is not None:
                repairs.append(record)

    combined.to_csv(combined_path, index=False)
    valid[[SMILES_COLUMN]].to_csv(valid_path, index=False)
    test[[SMILES_COLUMN]].to_csv(test_path, index=False)

    def frame_digest(frame: pd.DataFrame) -> str:
        material = frame[[SMILES_COLUMN, TARGET_COLUMN]].to_csv(index=False).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def source_frame_digest(frame: pd.DataFrame) -> str:
        source_smiles, source_labels = extract_xy(frame)
        source = pd.DataFrame(
            {
                SMILES_COLUMN: pd.Series(source_smiles, dtype="string"),
                TARGET_COLUMN: pd.to_numeric(source_labels, errors="raise").astype(float),
            }
        )
        return frame_digest(source)

    return {
        "source": "admet_benchmark.data_loader.load_tdc_official",
        "official_test_preserved": True,
        "official_split_membership_preserved": True,
        "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
        "model_input_repairs_n": len(repairs),
        "model_input_repairs": repairs,
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "train_sha256": frame_digest(train),
        "valid_sha256": frame_digest(valid),
        "test_sha256": frame_digest(test),
        "source_train_sha256": source_frame_digest(split.train),
        "source_valid_sha256": source_frame_digest(split.valid),
        "source_test_sha256": source_frame_digest(split.test),
        "combined_csv_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
    }


def resolve_chemprop_bin(value: str | None) -> Path:
    candidate = value or os.environ.get("CHEMPROP_BIN") or shutil.which("chemprop")
    if not candidate:
        raise FileNotFoundError(
            "Chemprop CLI not found. Install chemprop==2.2.1 and pass --chemprop-bin "
            "or set CHEMPROP_BIN."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Chemprop CLI does not exist: {path}")
    return path


def _python_for_console_script(chemprop_bin: Path, explicit: str | None = None) -> Path:
    if explicit:
        # Keep a venv's ``bin/python`` path intact. Resolving the symlink to the
        # base interpreter drops Python's virtual-environment discovery.
        return Path(explicit).expanduser().absolute()
    try:
        first_line = chemprop_bin.read_text(errors="replace").splitlines()[0]
        if first_line.startswith("#!"):
            tokens = shlex.split(first_line[2:])
            if tokens and Path(tokens[0]).name == "env" and len(tokens) > 1:
                found = shutil.which(tokens[1])
                if found:
                    return Path(found).absolute()
            if tokens and Path(tokens[0]).exists():
                return Path(tokens[0]).absolute()
    except OSError:
        pass
    sibling = chemprop_bin.parent / "python"
    return sibling.absolute() if sibling.exists() else Path(sys.executable).absolute()


def chemprop_runtime(
    chemprop_bin: Path, chemprop_python: str | None = None
) -> dict[str, Any]:
    python = _python_for_console_script(chemprop_bin, chemprop_python)
    code = (
        "import json,platform; from importlib.metadata import version; "
        "import torch,numpy; from rdkit import rdBase; print(json.dumps({"
        "'chemprop':version('chemprop'),'python':platform.python_version(),"
        "'torch':torch.__version__,'numpy':numpy.__version__,"
        "'rdkit':rdBase.rdkitVersion,'cuda_runtime':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(result.stdout.strip())
    runtime["chemprop_bin"] = str(chemprop_bin)
    runtime["chemprop_python"] = str(python)
    if runtime["chemprop"] != REQUIRED_CHEMPROP_VERSION:
        raise RuntimeError(
            f"Expected chemprop=={REQUIRED_CHEMPROP_VERSION}, got {runtime['chemprop']}"
        )
    return runtime


def build_train_command(
    chemprop_bin: Path,
    data_path: Path,
    train_output_dir: Path,
    policy: SelectionPolicy,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
) -> list[str]:
    """Build the locked CLI command for a directed bond-message D-MPNN.

    Do not add ``--atom-messages`` or ``--undirected`` here: their absence is
    what selects Chemprop's default directed bond-message architecture.
    """
    command = [
        str(chemprop_bin),
        "train",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(train_output_dir),
        "--smiles-columns",
        SMILES_COLUMN,
        "--target-columns",
        TARGET_COLUMN,
        "--splits-column",
        SPLIT_COLUMN,
        "--task-type",
        policy.chemprop_task_type,
        "--metrics",
        policy.chemprop_metric,
        "--tracking-metric",
        policy.tracking_metric,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--warmup-epochs",
        "2",
        "--data-seed",
        str(seed),
        "--pytorch-seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--accelerator",
        accelerator,
        "--devices",
        "1",
        "--message-hidden-dim",
        "300",
        "--depth",
        "3",
        "--dropout",
        "0.0",
        "--aggregation",
        "norm",
        "--aggregation-norm",
        "100",
        "--ffn-hidden-dim",
        "300",
        "--ffn-num-layers",
        "1",
        "--remove-checkpoints",
    ]
    forbidden = {"--atom-messages", "--undirected"}
    if forbidden.intersection(command):
        raise AssertionError("Chemprop D-MPNN command changed away from directed bond messages")
    return command


def build_predict_command(
    chemprop_bin: Path,
    input_path: Path,
    model_path: Path,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    accelerator: str,
) -> list[str]:
    return [
        str(chemprop_bin),
        "predict",
        "--test-path",
        str(input_path),
        "--model-path",
        str(model_path),
        "--output",
        str(output_path),
        "--smiles-columns",
        SMILES_COLUMN,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--accelerator",
        accelerator,
        "--devices",
        "1",
    ]


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] $ {_command_text(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
        )


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(staged, destination)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    frame.to_csv(staged, index=False)
    os.replace(staged, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, staged)
    os.replace(staged, destination)


def _read_prediction_values(path: Path, expected_n: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        prediction_columns = [c for c in frame.columns if c.startswith("pred_")]
        if len(prediction_columns) != 1:
            raise ValueError(
                f"Could not identify Chemprop prediction column in {path}: {list(frame.columns)}"
            )
        column = prediction_columns[0]
    else:
        column = TARGET_COLUMN
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if len(values) != expected_n:
        raise ValueError(f"Prediction count mismatch: expected {expected_n}, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite predictions in {path}")
    return values


def compute_official_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    task_type: str,
    metric_name: str,
) -> float:
    metric = str(metric_name).lower()
    if task_type == "classification" and metric == "auroc":
        value = roc_auc_score(labels, predictions)
    elif task_type == "classification" and metric == "auprc":
        value = average_precision_score(labels, predictions)
    elif task_type == "regression" and metric == "mae":
        value = mean_absolute_error(labels, predictions)
    elif task_type == "regression" and metric == "spearman":
        value = spearmanr(labels, predictions, nan_policy="raise").statistic
    else:
        raise ValueError(f"Unsupported official metric: {task_type}/{metric_name}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Official metric is not finite: {value}")
    return value


def _best_epoch(train_output_dir: Path, policy: SelectionPolicy) -> int | None:
    metric_column = f"val/{policy.tracking_metric}"
    candidates: list[tuple[float, int]] = []
    for path in train_output_dir.rglob("metrics.csv"):
        frame = pd.read_csv(path)
        if metric_column not in frame or "epoch" not in frame:
            continue
        subset = frame[["epoch", metric_column]].dropna()
        candidates.extend(
            (float(row[metric_column]), int(row["epoch"])) for _, row in subset.iterrows()
        )
    if not candidates:
        return None
    higher_is_better = policy.tracking_metric in {"roc", "prc"}
    return (max if higher_is_better else min)(candidates, key=lambda item: item[0])[1]


def _labels(frame: pd.DataFrame) -> np.ndarray:
    _, labels = extract_xy(frame)
    return pd.to_numeric(labels, errors="raise").to_numpy(dtype=float)


def _smiles(frame: pd.DataFrame) -> list[str]:
    smiles, _ = extract_xy(frame)
    return smiles


def train_single_run(
    task: str,
    seed: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = "data/tdc_official",
    chemprop_bin: str | None = None,
    chemprop_python: str | None = None,
    accelerator: str = "gpu",
    device_id: int = 0,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 64,
    num_workers: int = 0,
    split_override: OfficialSplit | None = None,
    loader: Callable[..., OfficialSplit] = load_tdc_official,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
) -> dict[str, Any]:
    """Train one task/seed; write metrics last as the completion marker."""
    if task not in TDC_TASK_META and split_override is None:
        raise ValueError(f"Unknown TDC task: {task}")
    output_dir = Path(output_dir).resolve()
    ensure_output_dirs(output_dir)
    paths = artifact_paths(output_dir, task, seed)
    if is_complete(paths["metrics"], split_source=split_source):
        return {"status": "skipped", "run_id": run_id(task, seed)}

    cp_bin = resolve_chemprop_bin(chemprop_bin)
    runtime = chemprop_runtime(cp_bin, chemprop_python)
    split = split_override or loader(
        task,
        data_dir=data_dir,
        seed=seed,
        split_source=split_source,
    )
    policy = selection_policy(split.task_type, split.metric)
    started_at = utc_now()
    started = time.monotonic()

    scratch_parent = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
    if scratch_parent and not Path(scratch_parent).is_dir():
        scratch_parent = None
    scratch = Path(
        tempfile.mkdtemp(prefix=f"{run_id(task, seed)}_", dir=scratch_parent)
    )
    env = os.environ.copy()
    if accelerator == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Chemprop reloads the checkpoint it just created before test prediction.
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects
    # Chemprop metric classes. This official PyTorch escape hatch is safe here
    # because the checkpoint is generated locally by this same run.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env.setdefault(variable, "1")

    try:
        combined_path = scratch / "locked_splits.csv"
        valid_input = scratch / "valid.csv"
        test_input = scratch / "test.csv"
        train_output = scratch / "chemprop_training"
        valid_raw = scratch / "valid_predictions_raw.csv"
        test_raw = scratch / "test_predictions_raw.csv"
        split_manifest = prepare_locked_data(
            split, combined_path, valid_input, test_input
        )
        train_command = build_train_command(
            cp_bin,
            combined_path,
            train_output,
            policy,
            seed,
            epochs,
            patience,
            batch_size,
            num_workers,
            accelerator,
        )

        config: dict[str, Any] = {
            "status": "running",
            "run_id": run_id(task, seed),
            "task": task,
            "model": MODEL_ID,
            "seed": int(seed),
            "split_source": split_source,
            "split_algorithm": (
                LOCKED_CACHE_SPLIT_ALGORITHM
                if split_source == LOCKED_CACHE_SPLIT_SOURCE
                else "pytdc_default_task_specific_runtime"
            ),
            "started_at_utc": started_at,
            "task_type": split.task_type,
            "official_metric": split.metric,
            "selection_policy": asdict(policy),
            "software": runtime,
            "platform": platform.platform(),
            "data": split_manifest,
            "training": {
                "epochs": int(epochs),
                "patience": int(patience),
                "warmup_epochs": 2,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "data_seed": int(seed),
                "pytorch_seed": int(seed),
                "accelerator": accelerator,
                "trusted_self_generated_checkpoint_load": True,
            },
            "architecture": {
                "implementation": "chemprop.cli",
                "message_type": "directed_bond_messages",
                "directed_bond_messages": True,
                "atom_messages": False,
                "undirected": False,
                "message_hidden_dim": 300,
                "depth": 3,
                "dropout": 0.0,
                "aggregation": "norm",
                "aggregation_norm": 100,
                "ffn_hidden_dim": 300,
                "ffn_num_layers": 1,
            },
            "commands": {"train": _command_text(train_command)},
        }
        _atomic_json(config, paths["config"])
        run_command(train_command, paths["log"], env)

        model_source = train_output / "model_0" / "best.pt"
        if not model_source.exists():
            raise FileNotFoundError(f"Chemprop did not produce {model_source}")

        valid_command = build_predict_command(
            cp_bin,
            valid_input,
            model_source,
            valid_raw,
            batch_size,
            num_workers,
            accelerator,
        )
        test_command = build_predict_command(
            cp_bin,
            test_input,
            model_source,
            test_raw,
            batch_size,
            num_workers,
            accelerator,
        )
        config["commands"]["predict_valid"] = _command_text(valid_command)
        config["commands"]["predict_test"] = _command_text(test_command)
        run_command(valid_command, paths["log"], env)
        run_command(test_command, paths["log"], env)

        valid_predictions = _read_prediction_values(valid_raw, len(split.valid))
        test_predictions = _read_prediction_values(test_raw, len(split.test))
        valid_labels = _labels(split.valid)
        test_labels = _labels(split.test)
        predictions = pd.DataFrame(
            {
                "split": ["valid"] * len(split.valid) + ["test"] * len(split.test),
                "smiles": _smiles(split.valid) + _smiles(split.test),
                "label": np.concatenate([valid_labels, test_labels]),
                "prediction": np.concatenate([valid_predictions, test_predictions]),
            }
        )
        valid_metric = compute_official_metric(
            valid_labels, valid_predictions, split.task_type, split.metric
        )
        test_metric = compute_official_metric(
            test_labels, test_predictions, split.task_type, split.metric
        )
        best_epoch = _best_epoch(train_output, policy)

        # Publish all companion artifacts first. The metrics JSON below is the
        # final and only completion marker used by the resume scheduler.
        _atomic_copy(model_source, paths["checkpoint"])
        _atomic_csv(predictions, paths["prediction"])
        config["status"] = "complete"
        config["completed_at_utc"] = utc_now()
        config["best_epoch_zero_based"] = best_epoch
        _atomic_json(config, paths["config"])

        metrics = {
            "status": "complete",
            "task": task,
            "model": MODEL_ID,
            "seed": int(seed),
            "split_source": split_source,
            "task_type": split.task_type,
            "metric_name": split.metric,
            "selection_metric": policy.public_name,
            "selection_metric_chemprop": policy.tracking_metric,
            "selection_rationale": policy.rationale,
            "best_epoch": best_epoch,
            "valid_metric": valid_metric,
            "test_metric": test_metric,
            "train_time_seconds": float(time.monotonic() - started),
            "chemprop_version": runtime["chemprop"],
            "prediction_rows": int(len(predictions)),
            "official_test_preserved": True,
        }
        _atomic_json(metrics, paths["metrics"])
        return {
            "status": "success",
            "run_id": run_id(task, seed),
            "test_metric": test_metric,
        }
    except Exception as exc:
        with paths["log"].open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] FAILED: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def synthetic_smoke_split() -> OfficialSplit:
    # Both validation and test contain both classes so AUROC is defined.
    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "CCCl",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CC(C)O",
        "COC",
        "CNC",
        "O=C=O",
        "N#N",
        "C1CCCCC1",
        "C1CCNCC1",
    ]
    labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]

    def make(indices: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {"Drug": [smiles[i] for i in indices], "Y": [labels[i] for i in indices]}
        )

    return OfficialSplit(
        name="synthetic_chemprop_smoke",
        train=make(list(range(12))),
        valid=make([12, 14, 13, 15]),
        test=make([0, 3, 6, 7]),
        task_type="classification",
        metric="auroc",
        category="smoke",
    )


def run_smoke(
    chemprop_bin: str | None,
    chemprop_python: str | None,
    accelerator: str = "cpu",
) -> None:
    with tempfile.TemporaryDirectory(prefix="admet_chemprop_smoke_") as directory:
        result = train_single_run(
            task="synthetic_chemprop_smoke",
            seed=7,
            output_dir=Path(directory) / "output",
            chemprop_bin=chemprop_bin,
            chemprop_python=chemprop_python,
            accelerator=accelerator,
            epochs=3,
            patience=1,
            batch_size=4,
            split_override=synthetic_smoke_split(),
        )
        paths = artifact_paths(Path(directory) / "output", "synthetic_chemprop_smoke", 7)
        assert result["status"] == "success"
        assert all(paths[key].exists() for key in ["checkpoint", "prediction", "config", "metrics"])
        assert list(pd.read_csv(paths["prediction"]).columns) == [
            "split",
            "smiles",
            "label",
            "prediction",
        ]
        print(f"Chemprop {accelerator.upper()} smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TDC_TASKS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-dir", default="data/tdc_official")
    parser.add_argument(
        "--split-source",
        choices=SPLIT_SOURCE_CHOICES,
        default=LOCKED_CACHE_SPLIT_SOURCE,
        help="Explicit split protocol; formal runs use locked_cache.",
    )
    parser.add_argument("--chemprop-bin", default=None)
    parser.add_argument("--chemprop-python", default=None)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a two-epoch isolated synthetic smoke test; no TDC result is written.",
    )
    parser.add_argument(
        "--smoke-accelerator",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Accelerator for the isolated synthetic smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        run_smoke(args.chemprop_bin, args.chemprop_python, args.smoke_accelerator)
        return 0

    failures = []
    tasks = args.tasks or TDC_TASKS
    for task in tasks:
        for seed in args.seeds:
            try:
                result = train_single_run(
                    task=task,
                    seed=seed,
                    output_dir=args.output_dir,
                    data_dir=args.data_dir,
                    split_source=args.split_source,
                    chemprop_bin=args.chemprop_bin,
                    chemprop_python=args.chemprop_python,
                    accelerator=args.accelerator,
                    device_id=args.device_id,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                print(json.dumps(result, sort_keys=True))
            except Exception as exc:  # continue independent task/seed runs
                failure = {"task": task, "seed": seed, "error": str(exc)}
                failures.append(failure)
                print(json.dumps({"status": "failed", **failure}), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
