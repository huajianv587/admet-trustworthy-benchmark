#!/usr/bin/env bash
# CPU model-refit defenses that are intentionally separate from the zero-fit
# final post-processing pipeline. Historical local_prelim outputs are untouched.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PARALLEL_RUNS="${PARALLEL_RUNS:-4}"
MODEL_N_JOBS="${MODEL_N_JOBS:-1}"
Y_RANDOM_TRIALS="${Y_RANDOM_TRIALS:-10}"
SELECTION=results/tables/traditional_grid_locked_cache_best_by_task.csv

if [[ ! -s "$SELECTION" ]]; then
  echo "missing validation-selected traditional table: $SELECTION" >&2
  exit 1
fi

"$PYTHON_BIN" experiments/run_y_randomization_all.py \
  --split-source locked_cache \
  --best "$SELECTION" \
  --n-trials "$Y_RANDOM_TRIALS" \
  --parallel-runs "$PARALLEL_RUNS" \
  --model-n-jobs "$MODEL_N_JOBS" \
  --summary results/tables/y_randomization_all_tasks_locked_cache.csv \
  --scores results/tables/y_randomization_all_tasks_scores_locked_cache.csv

"$PYTHON_BIN" experiments/run_shap_selected.py \
  --split-source locked_cache \
  --selection "$SELECTION" \
  --out results/tables/shap_selected_locked_cache.csv

"$PYTHON_BIN" experiments/map_shap_morgan_bits.py \
  --split-source locked_cache \
  --shap results/tables/shap_selected_locked_cache.csv \
  --out results/tables/shap_morgan_bit_substructures_locked_cache.csv

"$PYTHON_BIN" experiments/summarize_locked_cache_cpu_defenses.py \
  --y-summary results/tables/y_randomization_all_tasks_locked_cache.csv \
  --y-scores results/tables/y_randomization_all_tasks_scores_locked_cache.csv \
  --shap results/tables/shap_selected_locked_cache.csv \
  --mapping results/tables/shap_morgan_bit_substructures_locked_cache.csv \
  --selection "$SELECTION" \
  --out results/tables/locked_cache_cpu_defense_gate.json

echo "locked CPU defense gate: GO"
