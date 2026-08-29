#!/bin/bash
#SBATCH --job-name=759921-uw-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/759921-uw-smoke-%j.out
#SBATCH --error=logs/759921-uw-smoke-%j.err
set -euo pipefail
PROJECT_ROOT="${AIGC_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${AIGC_PYTHON:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="$PWD/.runtime/hf"
export TORCH_HOME="$PWD/.runtime/torch"
export TMPDIR="${AIGC_TMPDIR:-/tmp/h759uws-${SLURM_JOB_ID}}"
mkdir -p logs "$TMPDIR" outputs/hybrid_759921_uncertainty_h100_smoke
test -s data/fusion_v2/manifests/receipt.json
test -s checkpoint/best.pt
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "outputs/hybrid_759921_uncertainty_h100_smoke/gpu-memory-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT
"$PYTHON_BIN" -m aigc_detector.train --config configs/hybrid_759921_uncertainty_h100_smoke.yaml --max-steps 2
"$PYTHON_BIN" scripts/verify_uncertainty_smoke.py --output-dir outputs/hybrid_759921_uncertainty_h100_smoke
