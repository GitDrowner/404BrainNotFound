# Provenance and path translation

Job 773086 originally ran from the private experiment workspace root. Its immutable, resolved
configuration is preserved at `results/job-773086/resolved_config.yaml`; paths there describe the
original run and should not be edited.

The two files in `configs/` are portable copies. Their output directories point below this package
and their initializer points to `checkpoint/initial_759921.pt`. Dataset manifests remain external
because images and machine-specific paths are deliberately not committed.

The local backend also resolves `checkpoint/best.pt` and `checkpoint/calibration_balanced.json`
inside this package by default. Environment overrides are documented in `README.md` and
`BACKEND_API.md`.
