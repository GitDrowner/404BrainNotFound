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

`probability_fake` is the FP32 Platt-calibrated score. The demo selects one provisional operating
threshold per transform from `checkpoint/transform_thresholds_aligned_640.json` and maps that
threshold to `0.5`. The mapping preserves decisions and within-transform AUROC ordering but is not
a second probability calibration. These thresholds are post-hoc 640-image test oracles, not unbiased
held-out results; the Platt temperature/bias remain unchanged.

## Progressive transform API

- Obtain the exact transform IDs from `GET /api/v1/transforms`; do not invent aliases.
- `POST /api/v1/predict` accepts `file` and optional `transform`. It synchronously scores the
  selected variant and returns a `scan_id` plus `scan_status_url`.
- Poll `GET /api/v1/transform-scans/{scan_id}` for the remaining variants. Results belong only to
  that upload and are never pooled across images.
- The background scan is opportunistic. A newly waiting detection is scheduled before the next
  background transform, but an already-running model forward is not preempted.
- Gaussian noise is deterministically seeded by image hash and transform ID for reproducibility.
- This scheduling and threshold-policy change does not alter model architecture, checkpoint,
  Platt temperature/bias, preprocessing definitions, or confidence semantics.

## Key files

- `checkpoint/best.pt`: frozen job-773086 trainable state.
- `checkpoint/calibration_balanced.json`: deployed calibration and operating point.
- `checkpoint/transform_thresholds_aligned_640.json`: provisional per-transform demo thresholds.
- `src/aigc_detector/losses.py`: shared normalized MLP loss controller.
- `configs/`: portable training and smoke configurations.
- `results/job-773086/REPORT.md`: statistical report and limitations.
- `docs/ABLATION_773086_PLAN.md`: planned ablations; it does not authorize execution.

Do not commit datasets, generated runtime uploads, new checkpoints, or claims based on a test set
used for model selection.
