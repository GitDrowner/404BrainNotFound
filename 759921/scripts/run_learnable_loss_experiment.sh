#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${AIGC_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${AIGC_PYTHON:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

test -s checkpoint/best.pt
test -s data/fusion_v2/manifests/receipt.json
test -f data/external_eval_only/dalle3_advanced/.EVAL_ONLY_DO_NOT_TRAIN

"$PYTHON_BIN" -m aigc_detector.train --config configs/hybrid_759921_uncertainty_h100.yaml
"$PYTHON_BIN" -m aigc_detector.score \
  --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt \
  --manifest data/fusion_v2/manifests/calibration.jsonl \
  --output outputs/hybrid_759921_uncertainty_h100/calibration_scores.jsonl
"$PYTHON_BIN" scripts/fit_fp32_calibrator.py \
  --scores outputs/hybrid_759921_uncertainty_h100/calibration_scores.jsonl \
  --output outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
"$PYTHON_BIN" -m aigc_detector.evaluate \
  --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt \
  --manifest data/fusion_v2/manifests/test.jsonl \
  --output outputs/hybrid_759921_uncertainty_h100/general_robustness \
  --calibration outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
# The external set is opened only after checkpoint selection and calibration are frozen.
"$PYTHON_BIN" -m aigc_detector.evaluate \
  --checkpoint outputs/hybrid_759921_uncertainty_h100/best.pt \
  --manifest data/external_eval_only/dalle3_advanced/manifest.jsonl \
  --output outputs/hybrid_759921_uncertainty_h100/dalle3_observed_diagnostic \
  --calibration outputs/hybrid_759921_uncertainty_h100/calibration_fp32.json
