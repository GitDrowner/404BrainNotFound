# Fusion-v2 Uncertainty Weighting

Fusion-v2 keeps the GroupDRO real/fake classification objective as a fixed anchor and learns one
homoscedastic uncertainty weight for each of five human-readable auxiliary groups:

- `clean_robustness`: clean classification and final clean/augmented logit consistency;
- `augmentation_invariance`: consistency-head and feature consistency;
- `degradation_awareness`: degradation type and severity prediction;
- `local_evidence`: native-tile auxiliary classification;
- `moe_regularization`: expert gate balance and anti-collapse diversity.

For group loss `L_i` and learned log variance `s_i`, its objective contribution is
`exp(-s_i) * L_i + s_i`. The effective weight is therefore `exp(-s_i)`. The first 10% of optimizer
steps use the configured initial weights; learning is then enabled. Log variances are constrained to
`[-3, 3]` as a numerical safety bound. This module is training-only and does not alter the detector's
inference architecture or parameter count.

Audit artifacts are written below the configured output directory:

- `history.jsonl`: raw losses, group losses, effective weights, and weighted contributions per step;
- `loss_weight_history.jsonl`: a compact trajectory of all learned weights;
- `epoch-*.pt` and `best.pt`: `uncertainty_weighting` state and summary at checkpoint time;
- `summary.json`: final learned weights.

The weights explain optimization allocation, not per-image causal evidence. Per-image explanations
remain the responsibility of patch counterfactuals, expert weighted logits, and frequency ablations.
