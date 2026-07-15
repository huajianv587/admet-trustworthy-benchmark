#!/usr/bin/env bash
# Final, split-locked post-processing pipeline. This script never trains a
# neural model. It fails closed on incomplete artifacts or source matrices.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
RUN_RELEASE="${RUN_RELEASE:-1}"

for toggle_name in RUN_SENSITIVITY RUN_RELEASE; do
  toggle_value="${!toggle_name}"
  if [[ "$toggle_value" != "0" && "$toggle_value" != "1" ]]; then
    echo "$toggle_name must be exactly 0 or 1, got: $toggle_value" >&2
    exit 2
  fi
done

require_json_true() {
  local path="$1"
  local expression="$2"
  "$PYTHON_BIN" - "$path" "$expression" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = sys.argv[2].split(".")
value = json.loads(path.read_text(encoding="utf-8"))
for key in keys:
    value = value[key]
if value is not True:
    raise SystemExit(f"gate failed: {path} {'.'.join(keys)}={value!r}")
PY
}

require_json_equal() {
  local path="$1"
  local expression="$2"
  local expected="$3"
  "$PYTHON_BIN" - "$path" "$expression" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = sys.argv[2].split(".")
expected = sys.argv[3]
value = json.loads(path.read_text(encoding="utf-8"))
for key in keys:
    value = value[key]
if str(value) != expected:
    raise SystemExit(
        f"gate failed: {path} {'.'.join(keys)}={value!r}, expected {expected!r}"
    )
PY
}

# Recompute the traditional metrics from the locked prediction artifacts. The
# --from-predictions route audits all 330 files and performs no model fitting.
"$PYTHON_BIN" experiments/run_traditional_grid.py \
  --split-source locked_cache \
  --from-predictions results/predictions/traditional_locked_cache \
  --out results/tables/traditional_grid_locked_cache.csv \
  --rebuild-audit results/tables/traditional_grid_locked_cache_rebuild_audit.json

"$PYTHON_BIN" experiments/summarize_traditional_grid.py \
  --input results/tables/traditional_grid_locked_cache.csv \
  --summary results/tables/traditional_grid_locked_cache_summary.csv \
  --best results/tables/traditional_grid_locked_cache_best_by_task.csv

# The official-split fingerprint precheck is the hard prerequisite for every
# five-model analysis below.
"$PYTHON_BIN" experiments/precheck_final_neural_artifacts.py \
  --gin-dir results/neural \
  --chemprop-dir results/neural_chemprop \
  --data-dir data/tdc_official \
  --split-source locked_cache \
  --out-dir results/tables/final_neural_artifact_precheck_final

require_json_equal \
  results/tables/final_neural_artifact_precheck_final/final_neural_precheck_gate.json \
  status GO
require_json_true \
  results/tables/final_neural_artifact_precheck_final/final_neural_precheck_gate.json \
  sample_alignment.all_expected_runs_match_official_samples

# Compute split-explicit applicability caches. The output table itself covers
# traditional models; the exact same task/seed caches are consumed by all five
# models because the test rows are hard-aligned in the precheck.
"$PYTHON_BIN" experiments/analyze_applicability_domain.py \
  --pred-dir results/predictions/traditional_locked_cache \
  --data-dir data/tdc_official \
  --split-source locked_cache \
  --seeds 1 2 3 4 5 \
  --cache-dir results/cache/applicability \
  --out results/tables/applicability_domain_locked_cache_all_seeds.csv

"$PYTHON_BIN" experiments/analyze_reliability_regret.py \
  --prediction-dir results/predictions/traditional_locked_cache \
  --additional-prediction-dirs \
    results/neural/predictions \
    results/neural_chemprop/predictions \
  --similarity-dir results/cache/applicability \
  --similarity-split-source locked_cache \
  --require-complete-similarity \
  --gate-scope final \
  --expected-models rf xgboost lightgbm gin chemprop_dmpnn \
  --out-dir results/tables/reliability_regret_final

require_json_true \
  results/tables/reliability_regret_final/reliability_regret_gate.json \
  coverage.complete_expected_model_endpoint_matrix

"$PYTHON_BIN" experiments/analyze_conditional_reliability.py \
  --prediction-dir results/predictions/traditional_locked_cache \
  --additional-prediction-dirs \
    results/neural/predictions \
    results/neural_chemprop/predictions \
  --similarity-dir results/cache/applicability \
  --similarity-split-source locked_cache \
  --require-complete-similarity \
  --out-dir results/tables/conditional_reliability_final

"$PYTHON_BIN" experiments/analyze_calibration.py \
  --prediction-dir results/predictions/traditional_locked_cache \
  --additional-prediction-dirs \
    results/neural/predictions \
    results/neural_chemprop/predictions \
  --summary results/tables/calibration_summary_final.csv \
  --bins results/tables/calibration_reliability_bins_final.csv

# Zero-fit ablations reuse only the already gated predictions: all 2-, 3-, and
# 5-seed subsets plus equal-width/equal-mass ECE at 5/10/15/20 bins.
"$PYTHON_BIN" experiments/analyze_posthoc_ablations.py \
  --prediction-dir results/predictions/traditional_locked_cache \
  --additional-prediction-dirs \
    results/neural/predictions \
    results/neural_chemprop/predictions \
  --expected-models rf xgboost lightgbm gin chemprop_dmpnn \
  --seeds 1 2 3 4 5 \
  --ensemble-sizes 2 3 5 \
  --bin-counts 5 10 15 20 \
  --out-dir results/tables/posthoc_ablations_final

require_json_true \
  results/tables/posthoc_ablations_final/ablation_gate.json \
  all_task_seed_model_samples_aligned

"$PYTHON_BIN" experiments/analyze_trustworthiness_alignment.py \
  --profile-table \
    results/tables/conditional_reliability_final/model_reliability_profiles.csv \
  --calibration-summary results/tables/calibration_summary_final.csv \
  --out-dir results/tables/trustworthiness_alignment_final

"$PYTHON_BIN" experiments/make_reliability_regret_figure.py \
  --result-dir results/tables/reliability_regret_final \
  --out-dir results/figures/reliability_regret_final

"$PYTHON_BIN" experiments/make_trustworthiness_main_figures.py \
  --alignment-dir results/tables/trustworthiness_alignment_final \
  --conditional-dir results/tables/conditional_reliability_final \
  --calibration-summary results/tables/calibration_summary_final.csv \
  --out-dir results/figures/trustworthiness_submission_final

require_json_equal \
  results/figures/trustworthiness_submission_final/figure_manifest.json \
  gate.final_five_model_submission_figures PASS

# Summarize the independent paired cleaned-data sensitivity matrix.  Requesting
# this stage is fail-closed: absence of any primary artifact is not a skip.
if [[ "$RUN_SENSITIVITY" == "1" ]]; then
  for required in \
    results/tables/cleaned_sensitivity_locked_cache_runs.csv \
    results/tables/cleaned_sensitivity_locked_cache_audit.jsonl \
    results/tables/cleaned_sensitivity_locked_cache_manifest.json; do
    if [[ ! -s "$required" ]]; then
      echo "missing required cleaned-sensitivity artifact: $required" >&2
      exit 1
    fi
  done
  if [[ ! -d results/predictions/cleaned_sensitivity_locked_cache ]]; then
    echo "missing cleaned-sensitivity prediction directory" >&2
    exit 1
  fi
  if [[ -s results/tables/cleaned_sensitivity_locked_cache_failures.jsonl ]]; then
    echo "cleaned sensitivity failure log is non-empty; resolve and clear it before final gating" >&2
    exit 1
  fi
  "$PYTHON_BIN" experiments/summarize_cleaned_sensitivity.py \
    --runs results/tables/cleaned_sensitivity_locked_cache_runs.csv \
    --audit results/tables/cleaned_sensitivity_locked_cache_audit.jsonl \
    --expected-split-source locked_cache \
    --manifest results/tables/cleaned_sensitivity_locked_cache_manifest.json \
    --config experiments/config/sensitivity.yaml \
    --data-dir data/tdc_official \
    --out-dir results/tables/cleaned_sensitivity_locked_cache
  require_json_true \
    results/tables/cleaned_sensitivity_locked_cache/cleaned_sensitivity_gate.json \
    complete_expected_matrix
  "$PYTHON_BIN" experiments/audit_cleaned_sensitivity_predictions.py \
    --runs results/tables/cleaned_sensitivity_locked_cache_runs.csv \
    --predictions-dir results/predictions/cleaned_sensitivity_locked_cache \
    --manifest results/tables/cleaned_sensitivity_locked_cache_manifest.json \
    --config experiments/config/sensitivity.yaml \
    --data-dir data/tdc_official \
    --out-dir results/tables/cleaned_sensitivity_locked_cache
  require_json_equal \
    results/tables/cleaned_sensitivity_locked_cache/cleaned_sensitivity_prediction_gate.json \
    status GO
fi

# The release builder indexes predictions/configs/seeds without copying raw TDC
# data or checkpoints. License-policy decisions can legitimately leave HOLD.
if [[ "$RUN_RELEASE" == "1" ]]; then
  "$PYTHON_BIN" experiments/build_reproducibility_release.py \
    --output-dir release \
    --traditional-metrics results/tables/traditional_grid_locked_cache.csv \
    --traditional-predictions results/predictions/traditional_locked_cache \
    --neural-root results/neural \
    --chemprop-dir results/neural_chemprop \
    --release-policy release/release_policy.json
fi
