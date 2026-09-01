# Job 773086 — MLP-normalized robust AIGC detector

This directory is a portable, auditable snapshot of Slurm job **773086**. It retains the 759921
detector architecture and replaces five hand-set auxiliary-loss ratios with a shared MLP that
operates on EMA-normalized loss statistics. Training uses the combined, leakage-checked
759921 + 770876 corpus.

The selected checkpoint is included for inference. Dataset images are intentionally excluded.

## At a glance

| Item | Value |
|---|---|
| Architecture | DINOv2-B/14 global + native tiles, SigLIP SO400M/14 semantic, Haar wavelet |
| Total / trainable parameters | 515,587,535 / 1,637,007 |
| Training set | 44,377 images: 20,400 real + 23,977 fake |
| Validation / calibration | 5,298 / 4,700 images, isolated from training |
| Selection | highest internal validation AUROC; best epoch 2 |
| Internal validation AUROC | 88.07% |
| COCO/DALL-E 3 clean / mean-16 AUROC | 98.96% / 98.34% |
| COCO/MidJourney clean / mean-16 AUROC | 94.39% / 93.19% |
| End-to-end / training time | 79m10s / 57m24s on H100 NVL MIG 3g.47GB |

The DALL-E 3 and MidJourney figures are post-selection external diagnostics. DALL-E 3 was not used
for training, validation, calibration, hard-negative mining, or checkpoint selection.

## What changed from 759921

The detector graph did **not** change. Five auxiliary losses are individually divided by detached
EMA scale estimates. A shared, task-ID-free MLP assigns a positive simplex weight from their
relative difficulty and trends:

```text
L = L_primary + sum_i stop_gradient(w_i) * L_i / EMA(L_i)
```

`L_primary` remains the Class/Source GroupDRO objective with symmetric hard-example mining and a
real/fake risk-gap term. See `docs/MLP_NORMALIZED_LOSS_EXPERIMENT.md` and
`src/aigc_detector/losses.py` for the exact implementation.

## Repository map

- `checkpoint/best.pt` — selected job-773086 trainable state and config.
- `checkpoint/initial_759921.pt` — exact initializer used for training.
- `checkpoint/calibration_balanced.json` — deployed Platt parameters and calibration-only threshold.
- `configs/` — portable full and smoke configurations.
- `src/aigc_detector/` — training, scoring, evaluation, explanation, and local API code.
- `scripts/` — data preparation, calibration, smoke verification, demo, and helper entry points.
- `slurm/` — original NUS-oriented submission flow; edit paths/runtime before use elsewhere.
- `results/job-773086/` — statistical report, aggregate metrics, loss-weight trace, resource trace, log.
- `docs/ABLATION_773086_PLAN.md` — proposed ablation plan; no ablation is executed by this package.
- `AGENT_GUIDE.md` — evaluation and confidence boundaries for humans and AI assistants.

## Install

Python 3.10–3.12 is supported.

```bash
cd 773086
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

The first inference downloads frozen DINOv2 and SigLIP weights through `timm`. The included
checkpoint stores the compact trainable state, not duplicate backbone weights.

## Local calibrated demo

```bash
PYTHONPATH=src python scripts/serve_demo.py --device auto --port 8000
```

Open <http://127.0.0.1:8000>. The backend defaults to the included job-773086 checkpoint and
calibration. See `BACKEND_API.md` for endpoints and environment overrides.

The demo exposes all 16 competition transforms. When the user presses detect, the selected
transform runs on the foreground path and its result is returned immediately. If no newer
detection is waiting, one background worker then evaluates the remaining transforms in fixed
order and the frontend fills the comparison table progressively. New user detections take
priority between background model forwards; an in-flight forward is allowed to finish.

The API exposes two scores:

- `probability_fake`: FP32 Platt-calibrated score;
- `aigc_confidence`: a monotonic display mapping that sends the audited threshold `0.2815194250`
  to the intuitive display threshold `0.5`.

The mapping preserves classifications and AUROC ordering but is not a second probability
calibration.

## Command-line explanation

```bash
PYTHONPATH=src python scripts/explain_hybrid.py \
  --checkpoint checkpoint/best.pt \
  --calibration checkpoint/calibration_balanced.json \
  --image /absolute/path/example.jpg \
  --output explanations/example \
  --grid 6 --refine-top-k 6 --refine-grid 3 \
  --occlusion blur --device auto
```

Generated heatmaps are counterfactual model evidence, not pixel-level forgery ground truth.

## Training reproduction

The package does not redistribute dataset images or absolute-path manifests. Prepare the data first
so these files exist below the package root:

```text
data/fusion_v2/manifests/train.jsonl
data/fusion_v2/manifests/validation.jsonl
data/fusion_v2/manifests/test.jsonl
data/fusion_v2/manifests/calibration.jsonl
data/external_eval_only/dalle3_advanced/manifest.jsonl  # evaluation only
```

Then run the smoke test before a full run:

```bash
PYTHONPATH=src python -m aigc_detector.train \
  --config configs/hybrid_759921_mlp_controller_h100_smoke.yaml --max-steps 3
PYTHONPATH=src python scripts/verify_mlp_controller_smoke.py \
  --output-dir outputs/job-773086-smoke

PYTHONPATH=src python -m aigc_detector.train \
  --config configs/hybrid_759921_mlp_controller_h100.yaml
```

For Slurm, review the cluster-specific interpreter and paths in `slurm/` before `sbatch`. The exact
configuration resolved during the historical run is preserved separately under
`results/job-773086/resolved_config.yaml`.

## Result interpretation

The deployed operating threshold is `0.2815194250`, selected only on the calibration split by
class-balanced accuracy. AUROC is threshold-independent. Full per-transform accuracy, recall,
AUROC, AP, data composition, training loss, and limitations are in
`results/job-773086/REPORT.md`.

Checkpoint hashes and intentionally omitted artifacts are listed in `ARTIFACT_POLICY.md`.
