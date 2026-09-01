# MLP-normalized auxiliary-loss experiment

## Purpose

This experiment keeps the 759921 detector architecture and trains on the complete
`fusion_v2` manifest: all 39,985 images used by 759921 plus the 4,392 leakage-checked
OpenFake, TIGAS, and locally generated additions used by 770876. It removes the five
task-specific auxiliary coefficients inherited from 759921.

## Loss controller

For auxiliary task `i`, the raw loss is normalized by a detached exponential moving
average:

`r_i = L_i / EMA(L_i)`

A shared MLP receives scale-free loss trend features for every task. The same MLP is
applied to every task and has no task ID or task-specific bias. Its scores are mapped
to a positive simplex:

`w_i = epsilon + (1 - K * epsilon) * softmax(score)_i`

The detector minimizes:

`L_primary + auxiliary_budget * sum(stop_gradient(w_i) * r_i)`

The controller maximizes the weighted normalized auxiliary loss, implemented with a
zero-valued gradient proxy. This makes it emphasize currently hard tasks instead of
minimizing the objective by assigning all mass to the easiest task. The first 10% of
steps use uniform weights while EMA statistics stabilize. Every weight, normalized
loss, MLP score, and weighted contribution is written to `history.jsonl` and
`loss_weight_history.jsonl`.

`classification_anchor=1` and `auxiliary_budget=1` express only the division between
the primary task and the aggregate auxiliary regularizer. There are no per-task
initial coefficients. `min_weight=0.05` is a task-agnostic anti-collapse bound.

## Sampling

Training uses class-balanced sampling rather than equal `(label, source)` group mass.
The previous sampler caused a 12-image local source to occupy approximately 5.6% of
all draws. Source robustness remains handled by Class/Source GroupDRO inside the
primary classification objective.

## Checkpoint selection boundary

`best.pt` is selected by AUROC on
`data/fusion_v2/manifests/validation.jsonl`, a 5,298-image held-out split created from
the training-data system and isolated from the training manifest by SHA-256. This
split is inference-only and does not participate in back-propagation.

External test datasets are not evaluated inside the training loop. DALL-E Advanced
and the robustness grids are opened only after `best.pt` has been frozen. Their
AUROC values therefore measure post-selection test behaviour and never control
checkpoint saving, early stopping, loss-controller learning, or calibration fitting.
