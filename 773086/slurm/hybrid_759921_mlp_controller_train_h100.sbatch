#!/bin/bash
#SBATCH --job-name=759921-mlp-train
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=759921-mlp-train-%j.out
#SBATCH --error=759921-mlp-train-%j.err
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME="${AIGC_HF_HOME:-$PWD/.runtime/hf}"
export TORCH_HOME="${AIGC_TORCH_HOME:-$PWD/.runtime/torch}"
export TMPDIR="/tmp/h759mlpt-${SLURM_JOB_ID}"
RUN_PYTHON="${AIGC_PYTHON:-python}"
mkdir -p "$TMPDIR" outputs/job-773086-reproduction
command -v "$RUN_PYTHON" >/dev/null 2>&1 || test -x "$RUN_PYTHON"
test -s data/fusion_v2/manifests/receipt.json
test -s checkpoint/initial_759921.pt
test -s data/external_eval_only/dalle3_advanced/manifest.jsonl
test -f data/external_eval_only/dalle3_advanced/.EVAL_ONLY_DO_NOT_TRAIN
run_python() {
  "$RUN_PYTHON" "$@"
}
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "outputs/job-773086-reproduction/gpu-memory-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

# best.pt is selected by AUROC on the leakage-isolated fusion_v2 validation
# split. External test data, including DALL-E Advanced, is opened only after
# training and checkpoint selection have completely finished.
run_python -m aigc_detector.train --config configs/hybrid_759921_mlp_controller_h100.yaml
run_python -m aigc_detector.score --checkpoint outputs/job-773086-reproduction/best.pt --manifest data/fusion_v2/manifests/calibration.jsonl --output outputs/job-773086-reproduction/calibration_scores.jsonl
run_python scripts/fit_fp32_calibrator.py --scores outputs/job-773086-reproduction/calibration_scores.jsonl --output outputs/job-773086-reproduction/calibration_fp32.json
run_python scripts/select_balanced_threshold.py --scores outputs/job-773086-reproduction/calibration_scores.jsonl --calibration outputs/job-773086-reproduction/calibration_fp32.json --output outputs/job-773086-reproduction/calibration_balanced.json
run_python -m aigc_detector.evaluate --checkpoint outputs/job-773086-reproduction/best.pt --manifest data/fusion_v2/manifests/test.jsonl --output outputs/job-773086-reproduction/general_robustness --calibration outputs/job-773086-reproduction/calibration_balanced.json
run_python -m aigc_detector.evaluate --checkpoint outputs/job-773086-reproduction/best.pt --manifest data/external_eval_only/dalle3_advanced/manifest.jsonl --output outputs/job-773086-reproduction/dalle3_post_selection_test --calibration outputs/job-773086-reproduction/calibration_balanced.json
run_python -m aigc_detector.score --checkpoint outputs/job-773086-reproduction/best.pt --manifest data/external_eval_only/dalle3_advanced/manifest.jsonl --output outputs/job-773086-reproduction/dalle3_clean_scores.jsonl
run_python scripts/apply_calibrated_confidence.py --scores outputs/job-773086-reproduction/dalle3_clean_scores.jsonl --calibration outputs/job-773086-reproduction/calibration_balanced.json --output outputs/job-773086-reproduction/dalle3_clean_submission.jsonl --audit-output outputs/job-773086-reproduction/dalle3_clean_confidence_audit.jsonl
