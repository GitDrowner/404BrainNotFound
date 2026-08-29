# 759921 Architecture + Learnable Loss Weights

This experiment freezes the detector architecture to job 759921: DINOv2-B/14 forensic features,
SigLIP SO400M semantic features, four native-resolution tiles, and the Haar high-frequency branch.
It does not enable the Fusion-v2 three-expert MoE or quality gate.

The initial model weights come from `outputs/hybrid_groupdro_h100/best.pt`. Training, validation,
calibration, and general-test manifests are the leakage-checked Fusion-v2 manifests built for job
770418. DALL-E Advanced remains external evaluation only and is opened after checkpoint selection
and calibration are frozen.

The GroupDRO real/fake objective stays fixed at weight 1.0. Each of the five original 759921
auxiliary losses receives an independent homoscedastic uncertainty parameter: consistency, feature
consistency, degradation classification, degradation severity, and tile auxiliary classification.
The first 10% of optimizer steps are a fixed-weight warm-up. Weight trajectories are saved in
`loss_weight_history.jsonl` and every checkpoint.
