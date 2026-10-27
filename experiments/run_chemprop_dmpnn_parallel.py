#!/usr/bin/env python3
"""Resume-friendly, conservative parallel scheduler for Chemprop D-MPNNN runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    LOCKED_CACHE_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASKS,
)
from experiments.run_chemprop_dmpnn import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    MODEL_ID,
    artifact_paths,
    is_complete,
    run_id,
    train_single_run,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return train_single_run(**payload)
    except Exception as exc:
        return {
            "status": "failed",
            "run_id": run_id(payload["task"], payload["seed"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def completed_runs(
    output_dir: Path,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
    run_mode: str = "formal",
    batch_size: int | None = None,
) -> set[str]:
    complete: set[str] = set()
    metrics_dir = output_dir / "metrics"
    if not metrics_dir.exists():
        return complete
    for path in metrics_dir.glob(f"*_{MODEL_ID}_seed*.json"):
        if not is_complete(
            path,
            split_source=split_source,
            run_mode=run_mode,
            batch_size=batch_size,
        ):
            continue
        complete.add(path.stem)
    return complete


def _task_size(task: str, data_dir: Path) -> int:
    cached = data_dir / "admet_group" / task.lower() / "train_val.csv"
    try:
        return cached.stat().st_size
    except OSError:
        return 0


def build_queue(
    tasks: list[str],
    seeds: list[int],
    output_dir: Path,
    data_dir: Path,
    split_source: str = LOCKED_CACHE_SPLIT_SOURCE,
    run_mode: str = "formal",
    batch_size: int | None = None,
) -> list[tuple[str, int]]:
    complete = completed_runs(
        output_dir,
        split_source=split_source,
        run_mode=run_mode,
        batch_size=batch_size,
    )
    queue = [
        (task, seed)
        for task in tasks
        for seed in seeds
        if run_id(task, seed) not in complete
    ]
    # Short runs first improve wave utilization and produce early coverage.
    return sorted(queue, key=lambda item: (_task_size(item[0], data_dir), item[0], item[1]))


def _atomic_summary(payload: dict[str, Any], output_dir: Path) -> Path:
    directory = output_dir / "wave_summaries"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = os.environ.get("SLURM_JOB_ID", "local")
    destination = directory / f"chemprop_wave_{stamp}_{job}.json"
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(staged, destination)
    return destination


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
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--time-budget-hours", type=float, default=11.5)
    parser.add_argument("--stop-start-margin-minutes", type=float, default=75.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for clarity; resume is always on.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_parallel <= 4:
        raise ValueError("--max-parallel must be 1-4 for the locked conservative workflow")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be positive")
    if args.epochs != 100 or args.patience != 20:
        raise ValueError("Production runner is locked to --epochs 100 --patience 20")

    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    tasks = list(args.tasks or TDC_TASKS)
    run_mode = "profile_only" if args.profile_only else "formal"
    pending = deque(
        build_queue(
            tasks,
            args.seeds,
            output_dir,
            data_dir,
            split_source=args.split_source,
            run_mode=run_mode,
            batch_size=args.batch_size,
        )
    )
    original_queue = list(pending)
    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "already_complete": len(
                    completed_runs(
                        output_dir,
                        split_source=args.split_source,
                        run_mode=run_mode,
                        batch_size=args.batch_size,
                    )
                ),
                "queued": len(pending),
                "max_parallel": args.max_parallel,
                "time_budget_hours": args.time_budget_hours,
                "split_source": args.split_source,
                "batch_size": args.batch_size,
                "run_mode": run_mode,
            },
            sort_keys=True,
        )
    )
    if args.dry_run or not pending:
        for task, seed in list(pending)[:20]:
            print(run_id(task, seed))
        return 0

    started_at = utc_now()
    start = time.monotonic()
    deadline = start + args.time_budget_hours * 3600
    start_cutoff = deadline - args.stop_start_margin_minutes * 60
    results: list[dict[str, Any]] = []
    next_gpu = 0

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        active: dict[Any, tuple[str, int]] = {}
        while pending or active:
            while (
                pending
                and len(active) < args.max_parallel
                and time.monotonic() < start_cutoff
            ):
                task, seed = pending.popleft()
                payload = {
                    "task": task,
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "data_dir": str(data_dir),
                    "split_source": args.split_source,
                    "chemprop_bin": args.chemprop_bin,
                    "chemprop_python": args.chemprop_python,
                    "accelerator": "gpu",
                    "device_id": next_gpu % args.num_gpus,
                    "epochs": 100,
                    "patience": 20,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "profile_only": args.profile_only,
                }
                next_gpu += 1
                future = executor.submit(_worker, payload)
                active[future] = (task, seed)
                print(f"START {run_id(task, seed)} gpu={payload['device_id']}", flush=True)

            if not active:
                if pending:
                    print(
                        f"START CUTOFF reached with {len(pending)} runs left for next wave",
                        flush=True,
                    )
                break

            done, _ = wait(active, timeout=60, return_when=FIRST_COMPLETED)
            if not done:
                elapsed = (time.monotonic() - start) / 3600
                print(
                    f"HEARTBEAT elapsed={elapsed:.2f}h active={len(active)} pending={len(pending)}",
                    flush=True,
                )
                continue
            for future in done:
                task, seed = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # protects scheduler from worker death
                    result = {
                        "status": "failed",
                        "run_id": run_id(task, seed),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    finished_at = utc_now()
    complete_after = completed_runs(
        output_dir,
        split_source=args.split_source,
        run_mode=run_mode,
        batch_size=args.batch_size,
    )
    summary = {
        "model": MODEL_ID,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": time.monotonic() - start,
        "requested_runs": len(original_queue),
        "success": sum(r.get("status") == "success" for r in results),
        "skipped": sum(r.get("status") == "skipped" for r in results),
        "failed": sum(r.get("status") == "failed" for r in results),
        "deferred_to_next_wave": len(pending),
        "durable_complete_total": len(complete_after),
        "expected_complete_total": len(TDC_TASKS) * 5,
        "split_source": args.split_source,
        "batch_size": args.batch_size,
        "run_mode": run_mode,
        "results": results,
    }
    summary_path = _atomic_summary(summary, output_dir)
    print(f"WAVE_SUMMARY {summary_path}")
    # Failed independent runs are retried automatically because they have no
    # metrics completion marker. Return nonzero so Slurm records the issue.
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
