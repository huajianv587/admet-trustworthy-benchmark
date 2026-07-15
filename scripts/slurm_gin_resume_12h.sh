#!/usr/bin/env bash
#SBATCH --job-name=admet-gin
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=q_m1x12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Dedicated production launcher for the GIN half of the locked neural matrix.
# Deliberately does not accept or chain the historical `dmpnn` model name:
# that implementation is a legacy node-message MPNN and is excluded from the
# formal five-family benchmark.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda/12.8.0 2>/dev/null || true
module load python312 2>/dev/null || true

mkdir -p logs results/neural/{checkpoints,predictions,metrics,logs,resume}
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Keep the environment and frequently updated epoch checkpoints on node-local
# storage. Only durable final artifacts are published to the shared project.
ENV_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/admet_gin_venv_${SLURM_JOB_ID}"
LOCAL_CHECKPOINT_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/admet_checkpoints_${SLURM_JOB_ID}"
cleanup_env() {
  rm -rf "$ENV_ROOT" "$LOCAL_CHECKPOINT_ROOT"
}
trap cleanup_env EXIT
rm -rf "$ENV_ROOT"
python3 -m venv "$ENV_ROOT"
"$ENV_ROOT/bin/python" -m pip install --upgrade pip wheel "setuptools<81"
"$ENV_ROOT/bin/python" -m pip install numpy pandas scipy scikit-learn pyyaml "rdkit==2026.3.3" \
  filelock typing-extensions sympy networkx jinja2 fsspec
"$ENV_ROOT/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128"
"$ENV_ROOT/bin/python" -m pip install "torch-geometric==2.8.0"

MAX_PARALLEL="${MAX_PARALLEL:-8}"
echo "job_id=$SLURM_JOB_ID model=gin max_parallel=$MAX_PARALLEL"
"$ENV_ROOT/bin/python" - <<'PY'
import torch
import torch_geometric
from rdkit import Chem, rdBase
from rdkit.Chem import inchi

from admet_benchmark.smiles_compat import (
    LEGACY_VALENCE_REPAIR_INCHIKEYS,
    LEGACY_VALENCE_REPAIRS,
)

assert rdBase.rdkitVersion == "2026.03.3", rdBase.rdkitVersion
for source, replacement in LEGACY_VALENCE_REPAIRS.items():
    molecule = Chem.MolFromSmiles(replacement)
    assert molecule is not None, (source, replacement)
    assert inchi.MolToInchiKey(molecule) == LEGACY_VALENCE_REPAIR_INCHIKEYS[source]
print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("rdkit", rdBase.rdkitVersion)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the GIN environment")
PY
nvidia-smi

"$ENV_ROOT/bin/python" experiments/run_neural_a40_parallel.py \
  --model gin \
  --split-source locked_cache \
  --max-parallel "$MAX_PARALLEL" \
  --num-gpus 1 \
  --time-budget-hours 11.5 \
  --stop-start-margin-minutes 75 \
  --resume

# Count only complete GIN triplets with a parseable, correctly identified
# metric record. Orphan or legacy artifacts never satisfy this gate.
GIN_DONE="$("$ENV_ROOT/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path("results/neural")
count = 0
for metric_path in root.joinpath("metrics").glob("*_gin_seed*.json"):
    run_id = metric_path.stem
    try:
        record = json.loads(metric_path.read_text())
    except Exception:
        continue
    if record.get("model") != "gin":
        continue
    if record.get("split_source", "locked_cache") != "locked_cache":
        continue
    if not root.joinpath("predictions", f"{run_id}.csv").is_file():
        continue
    if not root.joinpath("checkpoints", f"{run_id}.pt").is_file():
        continue
    count += 1
print(count)
PY
)"
echo "durable completion: gin=${GIN_DONE}/110"

if [ "$GIN_DONE" -lt 110 ]; then
  # Preserve the cluster policy of at most one RUNNING plus one PENDING job.
  PENDING_COUNT="$(squeue -h -u "$USER" -t PENDING -n admet-gin 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${PENDING_COUNT:-0}" -gt 0 ]; then
    echo "successor already pending; no additional job submitted"
  elif SUBMIT_OUTPUT=$(sbatch --export=ALL,MAX_PARALLEL="$MAX_PARALLEL" "$0" 2>&1); then
    echo "submitted successor: $SUBMIT_OUTPUT"
  else
    echo "warning: compute-node successor submission unavailable: $SUBMIT_OUTPUT" >&2
    echo "manual successor: MAX_PARALLEL=$MAX_PARALLEL sbatch $0" >&2
  fi
else
  echo "all 110 locked GIN runs complete"
fi
