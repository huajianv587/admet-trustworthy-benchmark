#!/usr/bin/env bash
#SBATCH --job-name=admet-chemprop
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=q_m1x12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda/12.8.0 2>/dev/null || true
module load python312 2>/dev/null || true

mkdir -p logs results/neural_chemprop/{checkpoints,predictions,configs,metrics,logs,wave_summaries}
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Build the exact environment in node-local scratch. This avoids the shared
# home quota failures seen during the legacy neural waves.
ENV_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/admet_chemprop_venv_${SLURM_JOB_ID}"
cleanup_env() {
  rm -rf "$ENV_ROOT"
}
trap cleanup_env EXIT
rm -rf "$ENV_ROOT"
python3 -m venv "$ENV_ROOT"
"$ENV_ROOT/bin/python" -m pip install --upgrade pip wheel "setuptools<81"
"$ENV_ROOT/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128"
"$ENV_ROOT/bin/python" -m pip install "chemprop==2.2.1" "rdkit==2026.3.3"

export CHEMPROP_BIN="$ENV_ROOT/bin/chemprop"
export CHEMPROP_PYTHON="$ENV_ROOT/bin/python"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

"$ENV_ROOT/bin/python" - <<'PY'
import chemprop
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import inchi

from admet_benchmark.smiles_compat import (
    LEGACY_VALENCE_REPAIR_INCHIKEYS,
    LEGACY_VALENCE_REPAIRS,
)

assert chemprop.__version__ == "2.2.1", chemprop.__version__
assert rdBase.rdkitVersion == "2026.03.3", rdBase.rdkitVersion
for source, replacement in LEGACY_VALENCE_REPAIRS.items():
    molecule = Chem.MolFromSmiles(replacement)
    assert molecule is not None, (source, replacement)
    assert inchi.MolToInchiKey(molecule) == LEGACY_VALENCE_REPAIR_INCHIKEYS[source]
print("chemprop", chemprop.__version__)
print("rdkit", rdBase.rdkitVersion)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the Chemprop environment")
PY
nvidia-smi

if [ "${FULL_GPU_SMOKE:-0}" = "1" ]; then
  "$ENV_ROOT/bin/python" experiments/run_chemprop_dmpnn.py \
    --smoke \
    --smoke-accelerator gpu \
    --chemprop-bin "$CHEMPROP_BIN" \
    --chemprop-python "$CHEMPROP_PYTHON"
  echo "isolated full GPU smoke passed; no TDC artifact was written"
  exit 0
fi

echo "job_id=$SLURM_JOB_ID model=chemprop_dmpnn max_parallel=$MAX_PARALLEL"
"$ENV_ROOT/bin/python" experiments/run_chemprop_dmpnn_parallel.py \
  --split-source locked_cache \
  --max-parallel "$MAX_PARALLEL" \
  --num-gpus 1 \
  --time-budget-hours 11.5 \
  --stop-start-margin-minutes 75 \
  --chemprop-bin "$CHEMPROP_BIN" \
  --chemprop-python "$CHEMPROP_PYTHON" \
  --resume

CHEMPROP_DONE="$("$ENV_ROOT/bin/python" - <<'PY'
import json
from pathlib import Path

count = 0
for path in Path("results/neural_chemprop/metrics").glob("*_chemprop_dmpnn_seed*.json"):
    try:
        record = json.loads(path.read_text())
    except Exception:
        continue
    count += (
        record.get("status") == "complete"
        and record.get("model") == "chemprop_dmpnn"
        and record.get("split_source", "locked_cache") == "locked_cache"
    )
print(count)
PY
)"
echo "durable completion: chemprop_dmpnn=${CHEMPROP_DONE}/110"

if [ "$CHEMPROP_DONE" -lt 110 ]; then
  PENDING_COUNT="$(squeue -h -u "$USER" -t PENDING -n admet-chemprop 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${PENDING_COUNT:-0}" -gt 0 ]; then
    echo "successor already pending; no additional job submitted"
  elif SUBMIT_OUTPUT=$(sbatch --export=ALL,MAX_PARALLEL="$MAX_PARALLEL" "$0" 2>&1); then
    echo "submitted successor: $SUBMIT_OUTPUT"
  else
    echo "warning: compute-node successor submission unavailable: $SUBMIT_OUTPUT" >&2
    echo "manual successor: MAX_PARALLEL=$MAX_PARALLEL sbatch $0" >&2
  fi
else
  echo "all 110 official Chemprop D-MPNN runs complete"
fi
