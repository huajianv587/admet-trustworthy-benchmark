#!/usr/bin/env bash
# CPU-only paired cleaned-data sensitivity rerun on the formal locked_cache
# split. All destinations are separate from the completed historical pytdc run.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-1}"
UPGRADE_ONLY="${UPGRADE_ONLY:-0}"
UPGRADE_RECEIPT="${UPGRADE_RECEIPT:-results/tables/cleaned_sensitivity_locked_cache_manifest_provenance_upgrade_receipt.json}"

args=(
  experiments/run_cleaned_sensitivity_cpu.py
  --split-source locked_cache
  --threads-per-model "$THREADS_PER_MODEL"
  --out results/tables/cleaned_sensitivity_locked_cache_runs.csv
  --audit results/tables/cleaned_sensitivity_locked_cache_audit.jsonl
  --predictions-dir results/predictions/cleaned_sensitivity_locked_cache
  --manifest results/tables/cleaned_sensitivity_locked_cache_manifest.json
  --failures results/tables/cleaned_sensitivity_locked_cache_failures.jsonl
)

if [[ "$UPGRADE_ONLY" == "1" ]]; then
  # Use only after every older, pre-lock runner process has completely exited.
  args+=(--upgrade-only --upgrade-receipt "$UPGRADE_RECEIPT")
elif [[ "$UPGRADE_ONLY" == "0" ]]; then
  args+=(--resume)
else
  echo "UPGRADE_ONLY must be 0 or 1" >&2
  exit 2
fi

"$PYTHON_BIN" "${args[@]}"
