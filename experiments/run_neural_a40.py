#!/usr/bin/env python3
"""
A40 Neural Baseline Training Script

Design:
- Resume-friendly: skip completed runs
- Wave-aware: respects wall-time limits
- Checkpoint-based: saves model state, predictions, metrics
- Parallel-friendly: MAX_PARALLEL tasks at once

Usage:
  python experiments/run_neural_a40.py --model gin --tasks Caco2_Wang AMES --seeds 1 2 3
  python experiments/run_neural_a40.py --model dmpnn --resume
  python experiments/run_neural_a40.py --smoke  # smoke test only
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    load_tdc_official,
)
from admet_benchmark.neural_models import GINModel, DMPNNModel, MolecularGraphFeaturizer, collate_graphs
from admet_benchmark.utils import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def publish_checkpoint(local_path, destination, attempts=5):
    """Copy a completed checkpoint to shared storage with atomic replacement."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.copyfile(local_path, staged)
            os.replace(staged, destination)
            return
        except OSError as exc:
            last_error = exc
            staged.unlink(missing_ok=True)
            logger.warning(
                "Checkpoint publish attempt %d/%d failed for %s: %s",
                attempt, attempts, destination, exc,
            )
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Could not publish checkpoint {destination}: {last_error}")


def split_to_frames(data):
    """Normalize the local OfficialSplit dataclass into plain frames."""
    if isinstance(data, dict):
        return (
            data["train"],
            data["valid"],
            data["test"],
            data["task_type"],
            data["metric"],
        )
    return data.train, data.valid, data.test, data.task_type, data.metric


class MolecularGraphDataset(Dataset):
    """Dataset for molecular graphs."""

    def __init__(self, smiles_list, labels, featurizer):
        self.smiles_list = smiles_list
        self.labels = labels
        self.featurizer = featurizer
        # RDKit graph construction is CPU-heavy.  Building graphs in
        # ``__getitem__`` repeated the same work on every epoch and caused the
        # 12 concurrent A40 workers to contend for the 10 allocated CPUs.
        # Cache each molecule once; this changes no features or model inputs.
        self.graphs = [
            self.featurizer.smiles_to_graph(smiles) for smiles in self.smiles_list
        ]
        invalid = [index for index, graph in enumerate(self.graphs) if graph is None]
        if invalid:
            preview = ", ".join(
                f"{index}:{self.smiles_list[index]}" for index in invalid[:5]
            )
            raise ValueError(
                "Unparseable SMILES remain after the locked compatibility policy "
                f"at {preview}"
            )

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        label = self.labels[idx]
        graph = self.graphs[idx]
        if graph is None:
            return None

        # PyG batching mutates bookkeeping attributes, so return a shallow
        # clone rather than the cached object itself.
        graph = graph.clone()
        graph.y = torch.tensor([label], dtype=torch.float32)
        return graph


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch).view(-1)
        target = batch.y.view(-1)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, task_type, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch = batch.to(device)
            output = model(batch).view(-1)

            if task_type == 'classification':
                preds = torch.sigmoid(output).detach().cpu().numpy()
            else:
                preds = output.detach().cpu().numpy()

            all_preds.extend(np.ravel(preds).tolist())
            all_labels.extend(np.ravel(batch.y.detach().cpu().numpy()).tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_preds, all_labels


def compute_metrics(predictions, labels, task_type, metric_name):
    """Compute official metric."""
    metric_name = str(metric_name).lower()
    if task_type == 'classification':
        if metric_name == 'auroc':
            return roc_auc_score(labels, predictions)
        elif metric_name == 'auprc':
            return average_precision_score(labels, predictions)
        else:
            # Default to AUROC
            return roc_auc_score(labels, predictions)
    else:  # regression
        if metric_name == 'mae':
            return mean_absolute_error(labels, predictions)
        elif metric_name == 'rmse':
            return np.sqrt(mean_squared_error(labels, predictions))
        elif metric_name == 'spearman':
            corr, _ = spearmanr(predictions, labels)
            return corr
        else:
            # Default to MAE
            return mean_absolute_error(labels, predictions)


def metric_higher_is_better(task_type, metric_name):
    metric_name = str(metric_name).lower()
    if task_type == "classification":
        return True
    return metric_name == "spearman"


def completed_for_split_source(metrics_path, split_source):
    """Treat pre-provenance artifacts as locked_cache, never as pytdc."""
    try:
        payload = json.loads(Path(metrics_path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded = payload.get("split_source", LOCKED_CACHE_SPLIT_SOURCE)
    return recorded == split_source


def train_single_run(
    task_name,
    model_name,
    seed,
    output_dir,
    device_id=0,
    max_epochs=100,
    patience=20,
    data_dir="data/tdc_official",
    split_source=LOCKED_CACHE_SPLIT_SOURCE,
):
    """Train a single task-model-seed combination."""

    # Setup device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # Setup paths
    run_id = f"{task_name}_{model_name}_seed{seed}"
    checkpoint_path = output_dir / 'checkpoints' / f"{run_id}.pt"
    resume_checkpoint_path = output_dir / 'resume' / f"{run_id}.pt"
    prediction_path = output_dir / 'predictions' / f"{run_id}.csv"
    metrics_path = output_dir / 'metrics' / f"{run_id}.json"
    log_path = output_dir / 'logs' / f"{run_id}.log"

    # Check if already completed
    if (
        checkpoint_path.exists()
        and prediction_path.exists()
        and metrics_path.exists()
        and completed_for_split_source(metrics_path, split_source)
    ):
        logger.info(f"SKIP {run_id}: already completed")
        return {'status': 'skipped', 'run_id': run_id}

    # A checkpoint without both companion artifacts is an interrupted or
    # corrupt partial run, not a resumable completion marker.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    resume_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    try:
        logger.info(f"START {run_id}")
        start_time = time.time()

        # Training checkpoints are updated many times.  Writing every update
        # directly to the shared NFS caused concurrent PytorchStreamWriter
        # failures.  Keep the working checkpoint on node-local storage and
        # publish the final best state once, atomically, with retries.
        local_checkpoint_dir = Path(
            tempfile.gettempdir()
        ) / f"admet_checkpoints_{os.environ.get('SLURM_JOB_ID', 'local')}"
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        working_checkpoint_path = local_checkpoint_dir / f"{run_id}.pt"

        # Set seed
        set_seed(seed)

        # Load data
        data = load_tdc_official(
            task_name,
            data_dir=data_dir,
            seed=seed,
            split_source=split_source,
        )
        train_frame, valid_frame, test_frame, task_type, metric_name = split_to_frames(data)

        train_smiles = train_frame['Drug'].tolist()
        train_labels = train_frame['Y'].tolist()
        valid_smiles = valid_frame['Drug'].tolist()
        valid_labels = valid_frame['Y'].tolist()
        test_smiles = test_frame['Drug'].tolist()
        test_labels = test_frame['Y'].tolist()

        logger.info(f"Task: {task_name}, Type: {task_type}, Metric: {metric_name}")
        logger.info(f"Train: {len(train_smiles)}, Valid: {len(valid_smiles)}, Test: {len(test_smiles)}")

        # Create datasets
        featurizer = MolecularGraphFeaturizer()
        train_dataset = MolecularGraphDataset(train_smiles, train_labels, featurizer)
        valid_dataset = MolecularGraphDataset(valid_smiles, valid_labels, featurizer)
        test_dataset = MolecularGraphDataset(test_smiles, test_labels, featurizer)

        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_graphs, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_graphs, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_graphs, num_workers=0)

        # Create model
        if model_name == 'gin':
            model = GINModel(input_dim=117, hidden_dim=256, num_layers=5,
                           dropout=0.2, task_type=task_type)
        elif model_name == 'dmpnn':
            model = DMPNNModel(atom_dim=117, hidden_dim=300, depth=3,
                             dropout=0.1, task_type=task_type)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)

        # Loss and optimizer
        if task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=10)

        # Training loop
        higher_is_better = metric_higher_is_better(task_type, metric_name)
        best_valid_metric = float('-inf') if higher_is_better else float('inf')
        best_epoch = 0
        patience_counter = 0
        start_epoch = 0
        best_model_state = None

        # Resume an interrupted Slurm wave from its latest durable epoch.
        if resume_checkpoint_path.exists():
            try:
                resume_state = torch.load(
                    resume_checkpoint_path, map_location=device, weights_only=False
                )
                recorded_split_source = resume_state.get(
                    'split_source', LOCKED_CACHE_SPLIT_SOURCE
                )
                if recorded_split_source != split_source:
                    raise ValueError(
                        "resume checkpoint split_source mismatch: "
                        f"{recorded_split_source} != {split_source}"
                    )
                model.load_state_dict(resume_state['model_state_dict'])
                optimizer.load_state_dict(resume_state['optimizer_state_dict'])
                if resume_state.get('scheduler_state_dict'):
                    scheduler.load_state_dict(resume_state['scheduler_state_dict'])
                start_epoch = int(resume_state['epoch']) + 1
                best_epoch = int(resume_state.get('best_epoch', resume_state['epoch']))
                best_valid_metric = float(resume_state['best_valid_metric'])
                patience_counter = int(resume_state.get('patience_counter', 0))
                best_model_state = resume_state.get(
                    'best_model_state_dict', resume_state['model_state_dict']
                )
                shutil.copyfile(resume_checkpoint_path, working_checkpoint_path)
                logger.info(
                    "RESUME %s from epoch %d (best epoch %d, patience %d)",
                    run_id, start_epoch, best_epoch, patience_counter,
                )
            except Exception as exc:
                logger.warning("Ignoring unreadable resume checkpoint %s: %s", run_id, exc)
                resume_checkpoint_path.unlink(missing_ok=True)
                start_epoch = 0
                best_valid_metric = float('-inf') if higher_is_better else float('inf')
                best_epoch = 0
                patience_counter = 0
                best_model_state = None

        for epoch in range(start_epoch, max_epochs):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # Validate
            valid_preds, valid_labels = evaluate(model, valid_loader, task_type, device)
            valid_metric = compute_metrics(valid_preds, valid_labels, task_type, metric_name)

            scheduler.step(train_loss)

            # Check improvement
            improved = valid_metric > best_valid_metric if higher_is_better else valid_metric < best_valid_metric

            if improved:
                best_valid_metric = float(valid_metric)
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                patience_counter += 1

            # Save current optimizer/model state locally every epoch. Every two
            # epochs publish an atomic durable copy so the next 12h wave can
            # continue instead of restarting at epoch zero.
            progress_state = {
                'epoch': epoch,
                'split_source': split_source,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_model_state_dict': best_model_state,
                'best_valid_metric': best_valid_metric,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
            }
            torch.save(progress_state, working_checkpoint_path)
            if epoch % 2 == 0:
                publish_checkpoint(working_checkpoint_path, resume_checkpoint_path)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_{metric_name}={valid_metric:.4f}, best={best_valid_metric:.4f}")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Persist the last epoch before final evaluation, then restore the best
        # validation state for the locked prediction artifacts.
        publish_checkpoint(working_checkpoint_path, resume_checkpoint_path)
        if best_model_state is None:
            raise RuntimeError(f"No valid model state produced for {run_id}")
        model.load_state_dict(best_model_state)

        # Final evaluation
        valid_preds, valid_labels = evaluate(model, valid_loader, task_type, device)
        test_preds, test_labels = evaluate(model, test_loader, task_type, device)

        valid_metric = compute_metrics(valid_preds, valid_labels, task_type, metric_name)
        test_metric = compute_metrics(test_preds, test_labels, task_type, metric_name)

        # Save predictions
        predictions_df = pd.DataFrame({
            'split': ['valid'] * len(valid_preds) + ['test'] * len(test_preds),
            'smiles': valid_smiles + test_smiles,
            'label': list(valid_labels) + list(test_labels),
            'prediction': list(valid_preds) + list(test_preds)
        })
        predictions_df.to_csv(prediction_path, index=False)

        # Publish the model before writing metrics.  The metrics file is the
        # final completion marker used by the resume scanner.
        metrics = {
            'task': task_name,
            'model': model_name,
            'seed': seed,
            'split_source': split_source,
            'split_algorithm': (
                LOCKED_CACHE_SPLIT_ALGORITHM
                if split_source == LOCKED_CACHE_SPLIT_SOURCE
                else 'pytdc_default_task_specific_runtime'
            ),
            'task_type': task_type,
            'metric_name': metric_name,
            'best_epoch': best_epoch,
            'valid_metric': float(valid_metric),
            'test_metric': float(test_metric),
            'train_time_seconds': time.time() - start_time
        }

        final_checkpoint_path = local_checkpoint_dir / f"{run_id}.final.pt"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'best_valid_metric': best_valid_metric,
        }, final_checkpoint_path)
        publish_checkpoint(final_checkpoint_path, checkpoint_path)
        final_checkpoint_path.unlink(missing_ok=True)
        working_checkpoint_path.unlink(missing_ok=True)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        resume_checkpoint_path.unlink(missing_ok=True)

        logger.info(f"COMPLETE {run_id}: valid_{metric_name}={valid_metric:.4f}, test_{metric_name}={test_metric:.4f}")
        logger.removeHandler(file_handler)

        return {'status': 'success', 'run_id': run_id, 'test_metric': float(test_metric)}

    except Exception as e:
        logger.error(f"FAILED {run_id}: {str(e)}", exc_info=True)
        logger.removeHandler(file_handler)
        return {'status': 'failed', 'run_id': run_id, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='A40 Neural Baseline Training')
    parser.add_argument('--model', type=str, choices=['gin', 'dmpnn'], required=False,
                       help='Model type')
    parser.add_argument('--tasks', nargs='+', default=None,
                       help='Task names (default: all 22 TDC tasks)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                       help='Random seeds')
    parser.add_argument('--output-dir', type=str, default='results/neural',
                       help='Output directory')
    parser.add_argument('--data-dir', type=str, default='data/tdc_official',
                       help='TDC cache root')
    parser.add_argument('--split-source', choices=SPLIT_SOURCE_CHOICES,
                       default=LOCKED_CACHE_SPLIT_SOURCE,
                       help='Explicit split protocol; formal runs use locked_cache')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum parallel runs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoints')
    parser.add_argument('--smoke', action='store_true',
                       help='Smoke test: 1 regression + 1 classification task, 1 seed, both models')

    args = parser.parse_args()

    # Setup output directories
    output_dir = Path(args.output_dir)
    for subdir in ['checkpoints', 'predictions', 'metrics', 'logs']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Define task list
    all_tasks = [
        'Caco2_Wang', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma',
        'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'BBB_Martins',
        'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach',
        'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
        'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
        'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
        'CYP3A4_Substrate_CarbonMangels',
        'hERG', 'AMES', 'DILI', 'LD50_Zhu'
    ]

    # Smoke test mode
    if args.smoke:
        logger.info("SMOKE TEST MODE")
        tasks = ['Caco2_Wang', 'AMES']  # 1 regression, 1 classification
        seeds = [1]
        models = ['gin', 'dmpnn']
    else:
        tasks = args.tasks if args.tasks else all_tasks
        seeds = args.seeds
        models = [args.model] if args.model else ['gin', 'dmpnn']

    # Build run queue
    run_queue = []
    for task in tasks:
        for model in models:
            for seed in seeds:
                run_queue.append((task, model, seed))

    logger.info(f"Total runs in queue: {len(run_queue)}")
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Max parallel: {args.max_parallel}")

    # Execute runs (sequential for now, can be parallelized later)
    results = []
    for i, (task, model, seed) in enumerate(run_queue):
        logger.info(f"Progress: {i+1}/{len(run_queue)}")
        result = train_single_run(
            task,
            model,
            seed,
            output_dir,
            device_id=0,
            data_dir=args.data_dir,
            split_source=args.split_source,
        )
        results.append(result)

    # Summary
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_skipped = sum(1 for r in results if r['status'] == 'skipped')
    n_failed = sum(1 for r in results if r['status'] == 'failed')

    logger.info(f"SUMMARY: {n_success} success, {n_skipped} skipped, {n_failed} failed")

    # Save wave summary
    wave_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': len(run_queue),
        'success': n_success,
        'skipped': n_skipped,
        'failed': n_failed,
        'results': results
    }

    wave_summary_path = output_dir / f"wave_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(wave_summary_path, 'w') as f:
        json.dump(wave_summary, f, indent=2)

    logger.info(f"Wave summary saved to {wave_summary_path}")


if __name__ == '__main__':
    main()
