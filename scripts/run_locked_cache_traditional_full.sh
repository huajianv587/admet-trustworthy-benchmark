#!/usr/bin/env bash
# Regenerate the three traditional families on the exact validation split used
# by the formal remote neural artifacts. Historical *_local_prelim outputs are
# deliberately never read, resumed into, or overwritten by this script.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PARALLEL_RUNS="${PARALLEL_RUNS:-8}"
MODEL_N_JOBS="${MODEL_N_JOBS:-2}"

"$PYTHON_BIN" experiments/run_traditional_predictions.py \
  --split-source locked_cache \
  --group-by-task-seed \
  --parallel-runs "$PARALLEL_RUNS" \
  --model-n-jobs "$MODEL_N_JOBS" \
  --out-dir results/predictions/traditional_locked_cache

"$PYTHON_BIN" experiments/run_traditional_grid.py \
  --split-source locked_cache \
  --from-predictions results/predictions/traditional_locked_cache \
  --out results/tables/traditional_grid_locked_cache.csv \
  --rebuild-audit results/tables/traditional_grid_locked_cache_rebuild_audit.json

"$PYTHON_BIN" experiments/summarize_traditional_grid.py \
  --input results/tables/traditional_grid_locked_cache.csv \
  --summary results/tables/traditional_grid_locked_cache_summary.csv \
  --best results/tables/traditional_grid_locked_cache_best_by_task.csv
