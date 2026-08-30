# Agent guide

## Identity

This is job 773086: the fixed 759921 detector architecture trained on the combined 759921 + 770876
corpus with a shared MLP over EMA-normalized auxiliary losses. Do not describe it as a new detector
architecture. The architectural evidence branches remain DINOv2 global/tile, SigLIP semantic, and
Haar wavelet.

## Non-negotiable evaluation boundary

- DALL-E 3 Advanced is external evaluation only. It must never enter training, validation,
  calibration, hard-negative mining, or checkpoint selection.
- `best.pt` was selected only by AUROC on the 5,298-image held-out split sampled from the training
  data system.
- The 4,700-image calibration split fits Platt temperature/bias and the class-balanced threshold.
- External COCO/DALL-E and COCO/MidJourney results are post-selection diagnostics.
- Report datasets separately; do not pool unrelated external datasets into one headline number.

## Confidence semantics

`probability_fake` is the FP32 Platt-calibrated score. The demo's `aigc_confidence` is a strictly
monotonic, piecewise-linear display mapping that sends the audited threshold `0.2815194250` to
`0.5`. It preserves decisions and AUROC ordering but is not a second probability calibration.

## Key files

- `checkpoint/best.pt`: frozen job-773086 trainable state.
- `checkpoint/calibration_balanced.json`: deployed calibration and operating point.
- `src/aigc_detector/losses.py`: shared normalized MLP loss controller.
- `configs/`: portable training and smoke configurations.
- `results/job-773086/REPORT.md`: statistical report and limitations.
- `docs/ABLATION_773086_PLAN.md`: planned ablations; it does not authorize execution.

Do not commit datasets, generated runtime uploads, new checkpoints, or claims based on a test set
used for model selection.
