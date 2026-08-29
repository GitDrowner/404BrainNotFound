# Artifact policy

The repository includes source, documentation, one selected 6.3MB checkpoint, demonstration
images, and statistical experiment results. It intentionally does not duplicate:

- training, validation, calibration, or test images;
- manifests containing machine-specific dataset paths;
- per-image augmentation audit files;
- intermediate epoch checkpoints that duplicate the selected model;
- caches, virtual environments, TensorBoard event files, tarballs, or zip archives.

The selected checkpoint SHA-256 is
`2df35326275741b3889121f4e8c89b06533aa97194ca4b4faae428d17b03023e`. The omitted original
job-759921 output archive remains in the experiment workspace as
`test/results/result-759921-hybrid.tar.gz`. `results/job-759921/data_receipt.json` records the
original role separation; DALL-E Advanced did not participate in training, checkpoint selection, or
threshold selection.

`ORIGINAL_PACKAGE_MANIFEST.json` is retained only as provenance for the earlier explainability zip.
`PACKAGE_MANIFEST.json` describes the current consolidated Git directory.
