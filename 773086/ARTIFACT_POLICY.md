# Artifact policy

This directory contains the portable source, two selected lightweight trainable-state
checkpoints, calibration metadata, statistical results, and the job log needed to audit job
773086. It intentionally excludes:

- all training, validation, calibration, and external-test images;
- manifests containing workstation- or cluster-specific absolute paths;
- per-image scores and augmentation audits;
- five redundant epoch checkpoints, caches, virtual environments, and archives.

The included checkpoints contain the detector's trainable state and configuration. The frozen
DINOv2 and SigLIP backbone weights are fetched by `timm` on first use.

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `checkpoint/best.pt` | selected job-773086 model | `e32b6e19968f3ff28c39cd7f2457a8d1eb3acd10a72618cd8a75071a2914862a` |
| `checkpoint/initial_759921.pt` | exact 759921 initializer | `2df35326275741b3889121f4e8c89b06533aa97194ca4b4faae428d17b03023e` |

`checkpoint/calibration_fp32_original.json` is the original Platt calibrator and conservative
operating threshold emitted by the training pipeline. `checkpoint/calibration_balanced.json` keeps
the same temperature and bias but uses the calibration-only class-balanced reference threshold
`0.2815194250`. The demo now loads separate provisional per-transform operating points from
`checkpoint/transform_thresholds_aligned_640.json`; those values were fitted post hoc on a
640-image internal-test subset and therefore are explicitly test-derived.
