# Agent guide

This package prepares evaluation data only. It contains no images.

Non-negotiable rules:

1. Never use generated manifests or images for training, fine-tuning, hard-negative mining,
   calibration, threshold selection, or checkpoint selection.
2. Keep all five benchmark manifests separate. Do not concatenate them and do not report a pooled
   AUROC.
3. The Qwen-Image-Bench subset is fake-only. Report fake recall and score distributions; AUROC is
   undefined.
4. Require training/existing-holdout manifests as SHA-256 blocklists for an audited reproduction.
   `--allow-empty-blocklist` creates a non-audited copy and must be labelled accordingly.
5. `seen_in_training` is a normalized generator-name annotation, not a statement of image overlap or
   family-level novelty.
6. Do not commit `data/`, manifests generated after download, caches, or upstream files.

Start with `README.md`, use the revisions in `configs/external_generalization_suite.json`, and verify
the result with `scripts/verify_external_generalization_suite.py` before inference.
