# Agent guide

This package prepares evaluation data only. It contains no images.

Non-negotiable rules:

1. Never use generated manifests or images for training, fine-tuning, hard-negative mining,
   calibration, threshold selection, or checkpoint selection.
2. Keep all seven benchmark manifests separate. Do not concatenate them and do not report a pooled
   AUROC.
3. The Qwen-Image-Bench subset is fake-only. Report fake recall and score distributions; AUROC is
   undefined.
4. Require training/existing-holdout manifests as SHA-256 blocklists for an audited reproduction.
   `--allow-empty-blocklist` creates a non-audited copy and must be labelled accordingly.
5. `seen_in_training` is a normalized generator-name annotation, not a statement of image overlap or
   family-level novelty.
6. Do not commit `data/`, manifests generated after download, caches, or upstream files.
7. For DALL-E preparation, resolve every training image and run the pairwise dHash quarantine; do
   not silently downgrade it to exact-SHA-only checking.
8. The upstream `train` split names in the two Bitmind repositories do not grant a local training
   role. Their selected records are always external-evaluation-only.
9. Fail if the supplied training manifest names DALL-E, Defactify, OpenFake, MidJourney, or the
   historical COCO test source. Do not weaken source-disjointness to exact-file disjointness.

Start with `README.md`, use the revisions in both files under `configs/`, and verify
the result with `scripts/verify_external_generalization_suite.py` before inference.
