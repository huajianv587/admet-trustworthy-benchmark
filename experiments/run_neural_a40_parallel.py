#!/usr/bin/env python3
"""
A40 High-Throughput Neural Training Script

Optimized for A40 48GB VRAM:
- Multi-GPU parallelism via torch.multiprocessing
- Memory-efficient batching
- Dynamic task scheduling
- Aggressive parallelism to saturate GPU

Design goals:
- MAX_PARALLEL=8-12 concurrent runs (depends on task sizes)
- Estimate: 10-15 runs/hour with A40
- Target: 220 runs in 15-20 hours (1-2 waves)

Usage:
  python experiments/run_neural_a40_parallel.py --model gin --max-parallel 8
  python experiments/run_neural_a40_parallel.py --model dmpnn --max-parallel 10
"""

import argparse
import json
import logging
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from queue import Empty

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_task_size_estimate(task_name):
    """
    Estimate GPU memory requirement based on task size.
    Returns: 'tiny', 'small', 'medium', 'large'
    """
    # Known large tasks
    large_tasks = ['Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ']
    medium_tasks = ['Caco2_Wang', 'LD50_Zhu', 'VDss_Lombardo', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

    if task_name in large_tasks:
        return 'large'
    elif task_name in medium_tasks:
        return 'medium'
    else:
        return 'small'


def estimate_training_time(task_name, model_name):
    """
    Estimate training time in minutes for a single run.
    Based on task size and model complexity.
    """
    size_factor = {
        'tiny': 5,
        'small': 10,
        'medium': 20,
        'large': 40
    }

    model_factor = {
        'gin': 1.0,
        'dmpnn': 1.2  # D-MPNN slightly slower
    }

    size = get_task_size_estimate(task_name)
    base_time = size_factor.get(size, 10)
    model_mult = model_factor.get(model_name, 1.0)

    return base_time * model_mult


def worker_process(
    gpu_id,
    task_queue,
    result_queue,
    output_dir,
    stop_event,
    deadline_ts,
    start_margin_seconds,
    data_dir,
    split_source,
):
    """
    Worker process that consumes tasks from queue.
    Each worker gets assigned a GPU device.
    """
    logger.info(f"Worker started on GPU {gpu_id}")

    # Import here to avoid CUDA init in main process
    from experiments.run_neural_a40 import train_single_run

    while not stop_event.is_set():
        if deadline_ts and time.time() >= deadline_ts - start_margin_seconds:
            logger.info(f"[GPU {gpu_id}] Stop pulling new runs: wave deadline margin reached")
            break
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            continue

        if task is None:  # Poison pill
            break

        task_name, model_name, seed = task
        try:
            logger.info(f"[GPU {gpu_id}] Starting {task_name}_{model_name}_seed{seed}")

            result = train_single_run(
                task_name, model_name, seed,
                output_dir,
                device_id=gpu_id,
                max_epochs=100,
                patience=20,
                data_dir=data_dir,
                split_source=split_source,
            )

            result['gpu_id'] = gpu_id
            result_queue.put(result)

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Worker error: {e}")
            result_queue.put({
                'status': 'failed',
                'run_id': f"{task_name}_{model_name}_seed{seed}",
                'error': str(e),
                'gpu_id': gpu_id
            })


def get_completed_runs(output_dir, split_source=LOCKED_CACHE_SPLIT_SOURCE):
    """Get set of completed run_ids."""
    completed = set()
    checkpoint_dir = output_dir / 'checkpoints'
    prediction_dir = output_dir / 'predictions'
    metrics_dir = output_dir / 'metrics'

    if checkpoint_dir.exists() and prediction_dir.exists() and metrics_dir.exists():
        for checkpoint_file in checkpoint_dir.glob('*.pt'):
            run_id = checkpoint_file.stem
            pred_file = prediction_dir / f"{run_id}.csv"
            metric_file = metrics_dir / f"{run_id}.json"

            if pred_file.exists() and metric_file.exists():
                try:
                    payload = json.loads(metric_file.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                recorded = payload.get("split_source", LOCKED_CACHE_SPLIT_SOURCE)
                if recorded == split_source:
                    completed.add(run_id)

    return completed


def main():
    parser = argparse.ArgumentParser(description='A40 High-Throughput Neural Training')
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
    parser.add_argument('--max-parallel', type=int, default=8,
                       help='Maximum parallel runs (8-12 recommended for A40)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs available')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoints')
    parser.add_argument('--time-budget-hours', type=float, default=12.0,
                       help='Time budget in hours (for wave planning)')
    parser.add_argument('--stop-start-margin-minutes', type=float, default=30.0,
                       help='Do not start new runs this many minutes before the wave budget ends')

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

    tasks = args.tasks if args.tasks else all_tasks
    seeds = args.seeds
    models = [args.model] if args.model else ['gin', 'dmpnn']

    # Get completed runs
    completed = get_completed_runs(output_dir, args.split_source)
    logger.info(f"Found {len(completed)} completed runs")

    # Build run queue with priority (small tasks first for quick wins)
    run_queue_list = []
    for task in tasks:
        for model in models:
            for seed in seeds:
                run_id = f"{task}_{model}_seed{seed}"
                if run_id not in completed:
                    priority = get_task_size_estimate(task)
                    priority_order = {'small': 0, 'medium': 1, 'large': 2, 'tiny': 0}
                    run_queue_list.append((priority_order.get(priority, 1), task, model, seed))

    # Sort by priority (small tasks first)
    run_queue_list.sort(key=lambda x: x[0])

    total_runs = len(run_queue_list)
    logger.info(f"Total runs in queue: {total_runs}")
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Max parallel: {args.max_parallel}")
    logger.info(f"Num GPUs: {args.num_gpus}")
    logger.info(f"Split source: {args.split_source}")
    logger.info(f"Stop-start margin: {args.stop_start_margin_minutes:.1f} minutes")

    if total_runs == 0:
        logger.info("All runs already completed!")
        return

    # Estimate time
    total_time_minutes = sum(estimate_training_time(task, model) for _, task, model, _ in run_queue_list)
    parallel_time_minutes = total_time_minutes / args.max_parallel
    parallel_time_hours = parallel_time_minutes / 60

    logger.info(f"\nTime Estimates:")
    logger.info(f"  Sequential: {total_time_minutes:.0f} minutes ({total_time_minutes/60:.1f} hours)")
    logger.info(f"  Parallel ({args.max_parallel}x): {parallel_time_minutes:.0f} minutes ({parallel_time_hours:.1f} hours)")
    logger.info(f"  Time budget: {args.time_budget_hours:.1f} hours")

    if parallel_time_hours > args.time_budget_hours:
        logger.warning(f"  ⚠ Estimated time exceeds budget. May need multiple waves.")
    else:
        logger.info(f"  ✓ Should complete within time budget")

    # Create multiprocessing queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Fill task queue
    for _, task, model, seed in run_queue_list:
        task_queue.put((task, model, seed))

    # Add poison pills
    for _ in range(args.max_parallel):
        task_queue.put(None)

    # Start worker processes
    workers = []
    start_time = time.time()
    deadline_ts = start_time + args.time_budget_hours * 3600 if args.time_budget_hours else None
    start_margin_seconds = args.stop_start_margin_minutes * 60
    for i in range(args.max_parallel):
        gpu_id = i % args.num_gpus
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                output_dir,
                stop_event,
                deadline_ts,
                start_margin_seconds,
                args.data_dir,
                args.split_source,
            )
        )
        p.start()
        workers.append(p)

    # Monitor results
    results = []
    completed_count = 0

    try:
        while completed_count < total_runs and any(p.is_alive() for p in workers):
            if deadline_ts and time.time() >= deadline_ts:
                logger.warning("Wave time budget reached; stopping workers")
                stop_event.set()
                break

            try:
                result = result_queue.get(timeout=60)
            except Empty:
                elapsed_hours = (time.time() - start_time) / 3600
                alive = sum(1 for p in workers if p.is_alive())
                logger.info(
                    f"Heartbeat: {completed_count}/{total_runs} results received | "
                    f"Elapsed: {elapsed_hours:.1f}h | Alive workers: {alive}"
                )
                continue

            results.append(result)
            completed_count += 1

            elapsed_hours = (time.time() - start_time) / 3600
            remaining = total_runs - completed_count
            rate = completed_count / elapsed_hours if elapsed_hours > 0 else 0
            eta_hours = remaining / rate if rate > 0 else 0

            status_symbol = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"{status_symbol} [{completed_count}/{total_runs}] {result['run_id']} | "
                       f"Elapsed: {elapsed_hours:.1f}h | Rate: {rate:.1f}/h | ETA: {eta_hours:.1f}h")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        stop_event.set()

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Summary
    elapsed_total = (time.time() - start_time) / 3600
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_skipped = sum(1 for r in results if r['status'] == 'skipped')
    n_failed = sum(1 for r in results if r['status'] == 'failed')
    n_not_reported = total_runs - len(results)

    logger.info(f"\n{'='*60}")
    logger.info(f"WAVE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total runs attempted: {total_runs}")
    logger.info(f"Success: {n_success}")
    logger.info(f"Skipped: {n_skipped}")
    logger.info(f"Failed: {n_failed}")
    logger.info(f"Not reported this wave: {n_not_reported}")
    logger.info(f"Total time: {elapsed_total:.2f} hours")
    logger.info(f"Throughput: {(n_success + n_skipped) / elapsed_total:.1f} runs/hour")
    logger.info(f"{'='*60}")

    # Save wave summary
    wave_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': total_runs,
        'success': n_success,
        'skipped': n_skipped,
        'failed': n_failed,
        'not_reported': n_not_reported,
        'elapsed_hours': elapsed_total,
        'throughput_per_hour': (n_success + n_skipped) / elapsed_total if elapsed_total > 0 else 0,
        'max_parallel': args.max_parallel,
        'num_gpus': args.num_gpus,
        'results': results
    }

    wave_summary_path = output_dir / f"wave_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(wave_summary_path, 'w') as f:
        json.dump(wave_summary, f, indent=2)

    logger.info(f"Wave summary saved to {wave_summary_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
