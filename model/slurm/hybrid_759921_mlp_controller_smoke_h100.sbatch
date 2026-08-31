#!/bin/bash
#SBATCH --job-name=759921-mlp-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=759921-mlp-smoke-%j.out
#SBATCH --error=759921-mlp-smoke-%j.err
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="${AIGC_HF_HOME:-$PWD/.runtime/hf}"
export TORCH_HOME="${AIGC_TORCH_HOME:-$PWD/.runtime/torch}"
export TMPDIR="/tmp/h759mlps-${SLURM_JOB_ID}"
RUN_PYTHON="${AIGC_PYTHON:-python}"
mkdir -p "$TMPDIR" outputs/job-773086-smoke
command -v "$RUN_PYTHON" >/dev/null 2>&1 || test -x "$RUN_PYTHON"
test -s data/fusion_v2/manifests/receipt.json
test -s checkpoint/initial_759921.pt
run_python() {
  "$RUN_PYTHON" "$@"
}
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "outputs/job-773086-smoke/gpu-memory-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT
run_python -m aigc_detector.train --config configs/hybrid_759921_mlp_controller_h100_smoke.yaml --max-steps 3
run_python scripts/verify_mlp_controller_smoke.py --output-dir outputs/job-773086-smoke
