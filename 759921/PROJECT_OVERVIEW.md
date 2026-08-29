# Job 759921: consolidated project

This directory is the single Git-tracked home for the job-759921 AIGC detector and its follow-up
work. The deployed architecture is fixed to DINOv2-B/14 forensic features, SigLIP SO400M semantic
features, four native-resolution tiles, and a Haar high-frequency branch. The checkpoint contains
516M total model parameters and stays below the 2B competition limit.

## Directory map

- `src/aigc_detector/`: inference, training, evaluation, calibration, and explainability code;
- `checkpoint/`: the selected job-759921 checkpoint and its historical classification threshold;
- `demos/` and `demos-v2/`: static HTML explainability examples;
- `MODEL_ARCHITECTURE.*`: human-readable model diagram;
- `configs/`: original fixed-loss config and the independent learnable-loss configs;
- `slurm/`: smoke and formal H100 submission scripts for the learnable-loss experiment;
- `results/job-759921/`: statistical outputs and logs from the original completed run;
- `docs/`: uncertainty-weighting design and experiment boundary;
- `AGENT_GUIDE.md`: field semantics and safe comparison rules.

## Quick inference demo

```bash
cd 759921
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/explain_hybrid.py \
  --checkpoint checkpoint/best.pt \
  --image /path/to/image.png \
  --output explanations/example \
  --grid 6 --refine-top-k 6 --refine-grid 3 \
  --occlusion blur --device auto
```

Open `explanations/example/index.html` after completion. For existing examples, open
`demos-v2/index.html` directly.

## Learnable-loss experiment

The follow-up experiment does not change the 759921 inference architecture. GroupDRO real/fake
classification remains the fixed primary objective; each of the five original auxiliary losses has
an independent homoscedastic uncertainty weight. See
[`docs/HYBRID_759921_UNCERTAINTY_EXPERIMENT.md`](docs/HYBRID_759921_UNCERTAINTY_EXPERIMENT.md).

Local execution:

```bash
cd 759921
AIGC_PYTHON=python bash scripts/run_learnable_loss_experiment.sh
```

Slurm execution:

```bash
cd 759921
smoke=$(sbatch --parsable slurm/hybrid_759921_uncertainty_smoke_h100.sbatch)
sbatch --dependency=afterok:${smoke} slurm/hybrid_759921_uncertainty_train_h100.sbatch
```

The experiment expects leakage-checked manifests under `data/fusion_v2/manifests/` and keeps
`data/external_eval_only/dalle3_advanced/` inference-only. Dataset files are intentionally absent
from Git; see [`ARTIFACT_POLICY.md`](ARTIFACT_POLICY.md).
