# 773086 calibrated local demo backend

The backend serves both `demos-v2` and the frozen job 773086 model from one local
FastAPI process. Job 773086 retains the 759921 detector architecture and adds the
MLP-normalized auxiliary-loss training. The checkpoint is loaded lazily once and retained in memory.
Full explanation jobs are serialized because the wavelet-only counterfactual
uses a temporary inference hook on the shared fusion module.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/serve_demo.py --device auto --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

Device selection order for `auto` is CUDA, Apple MPS, then CPU. Override paths
without editing code:

```bash
AIGC_CHECKPOINT=/absolute/path/best.pt \
AIGC_CALIBRATION=/absolute/path/calibration_fp32.json \
AIGC_RESULTS_ROOT=/absolute/path/results \
PYTHONPATH=src python scripts/serve_demo.py --device mps
```

The checkpoint and calibration must be changed together. By default both resolve inside this
package as `checkpoint/best.pt` and the audited `checkpoint/calibration_balanced.json`. The decision threshold is always read from that JSON;
there is no independent display-threshold override.

## API

### `GET /api/health`

Does not force model loading. Returns service status, selected device, and queue
depth.

### `GET /api/v1/model`

Returns checkpoint SHA-256, backbone names, source epoch, device, and confidence
semantics.

### `POST /api/v1/predict`

Multipart form field: `file`. Accepted formats are JPEG, PNG, and WEBP, up to
20 MiB and 25 megapixels.

```bash
curl -F file=@/path/image.png http://127.0.0.1:8000/api/v1/predict
```

The response includes raw logit, FP32 Platt-calibrated logit and `probability_fake`, image
hash/dimensions, and the decision at the calibration file's threshold. The default values are
temperature `3.1648788452`, bias `-2.4065542221`, and threshold `0.2815194250`. The temperature and
bias are the frozen job 773086 Platt calibrator; the threshold maximizes class-balanced accuracy on
the same independent 4,700-image calibration set and uses no external test labels.

For presentation, the response also includes `aigc_confidence`, a strictly increasing piecewise
linear remapping that sends calibrated probability `0 -> 0`, threshold `0.2815194250 -> 0.5`, and
`1 -> 1`:

`p <= t: confidence = 0.5 * p / t`

`p > t: confidence = 0.5 + 0.5 * (p - t) / (1 - t)`

The displayed decision is `aigc_confidence >= 0.5`. This is exactly equivalent to
`probability_fake >= 0.2815194250` and preserves AUROC ordering. `aigc_confidence` is an intuitive
operating score, not an additional probability calibration; `probability_fake` retains the original
Platt value for audit.

### `POST /api/v1/analyses`

Multipart fields:

- `file`: image;
- `mode`: `fast` or `full`;
- `occlusion`: `blur` or `mean`.

```bash
curl -F file=@/path/image.png -F mode=fast -F occlusion=blur \
  http://127.0.0.1:8000/api/v1/analyses
```

`fast` uses a 4×4 coarse grid, refines the three strongest cells into 2×2, and
produces 28 counterfactual regions. `full` reproduces demos-v2: 6×6, top six,
3×3, for 90 regions. The endpoint returns HTTP 202 and a job ID.

### `GET /api/v1/analyses/{job_id}`

Poll until `status` becomes `completed` or `failed`.

### `GET /api/v1/analyses/{job_id}/result`

Returns `explanation.json` plus URLs for the generated dashboard, attribution
heatmaps, frequency counterfactuals, transform trajectory, and branch chart.
Static assets are served below `/results/{job_id}/`.

## Operational behavior

- Uploaded files and results stay below `runtime_results/` by default.
- Model parameters are never modified; every request runs in inference mode.
- Explanation jobs run one at a time. Fast prediction waits if a full explanation
  currently owns the model lock.
- Jobs are process-local and are not restored after a server restart. Generated
  artifacts remain on disk.
- The API is a local demo service, not an internet-facing hardened deployment.
  Authentication, quotas, persistence, and TLS would be required before exposing
  it outside a trusted machine.
