#!/bin/bash
#SBATCH --job-name=759921-uw-train
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/759921-uw-train-%j.out
#SBATCH --error=logs/759921-uw-train-%j.err
set -euo pipefail
PROJECT_ROOT="${AIGC_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${AIGC_PYTHON:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="$PWD/.runtime/hf"
export TORCH_HOME="$PWD/.runtime/torch"
export TMPDIR="${AIGC_TMPDIR:-/tmp/h759uwt-${SLURM_JOB_ID}}"
mkdir -p logs "$TMPDIR" outputs/hybrid_759921_uncertainty_h100
test -s data/fusion_v2/manifests/receipt.json
test -s checkpoint/best.pt
test -f data/external_eval_only/dalle3_advanced/.EVAL_ONLY_DO_NOT_TRAIN
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "outputs/hybrid_759921_uncertainty_h100/gpu-memory-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT
"$PYTHON_BIN" -m aigc_detector.train --config configs/hybrid_759921_uncertainty_h100.yaml
"$PYTHON_BIN" -m aigc_detector.score --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt --manifest data/fusion_v2/manifests/calibration.jsonl --output outputs/hybrid_759921_uncertainty_h100/calibration_scores.jsonl
"$PYTHON_BIN" scripts/fit_fp32_calibrator.py --scores outputs/hybrid_759921_uncertainty_h100/calibration_scores.jsonl --output outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
"$PYTHON_BIN" -m aigc_detector.evaluate --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt --manifest data/fusion_v2/manifests/test.jsonl --output outputs/hybrid_759921_uncertainty_h100/general_robustness --calibration outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
# DALL-E Advanced is inference-only and opened only after checkpoint and calibration are frozen.
"$PYTHON_BIN" -m aigc_detector.evaluate --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt --manifest data/external_eval_only/dalle3_advanced/manifest.jsonl --output outputs/hybrid_759921_uncertainty_h100/dalle3_observed_diagnostic --calibration outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
"$PYTHON_BIN" -m aigc_detector.score --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt --manifest data/external_eval_only/dalle3_advanced/manifest.jsonl --output outputs/hybrid_759921_uncertainty_h100/dalle3_clean_scores.jsonl
"$PYTHON_BIN" scripts/apply_calibrated_confidence.py --scores outputs/hybrid_759921_uncertainty_h100/dalle3_clean_scores.jsonl --calibration outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json --output outputs/hybrid_759921_uncertainty_h100/dalle3_clean_submission.jsonl --audit-output outputs/hybrid_759921_uncertainty_h100/dalle3_clean_confidence_audit.jsonl
