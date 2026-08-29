# Agent guide：759921 explanation

入口：`scripts/explain_hybrid.py`，运行时需要兼容 runtime 中的 `aigc_detector` 包。
统一打包版和原仓库都通过 `PYTHONPATH=src` 使用兼容推理代码。

关键语义：

- `model.method = 759921_hybrid_legacy`；
- `prediction.probability_fake` 是未校准 sigmoid，不等同于可靠概率；
- `attribution.patches[].raw_logit_contribution` 是原 raw logit 减去区域遮挡后的 raw logit；
- `attribution.refinement` 记录 coarse top-K 选择与局部细分方式；
- `confidence_contribution` 仍被保留，但饱和样本应优先解释 raw-logit 贡献；
- `visualizations.heatmap_attribution` 记录多尺度 raw-logit 图的信号、归一化和数值范围；
- `visualizations.heatmap_texture` 记录 99th-percentile 归一化的高频残差信号，它不是模型输出；
- `frequency_attribution` 记录局部低通与 wavelet-only fusion counterfactual 的严格语义；
- `high_frequency_ablation.wavelet_only_raw_logit_contribution` 固定其他 fusion 特征，只改变
  checkpoint 实际计算的 `wavelet_similarity`；
- `branches[].confidence_delta` 是输入分支 neutralization 的概率差；
- `tiles[].attention` 与 `tiles[].raw_logit` 分别是聚合权重和 auxiliary tile 证据；
- `transforms` 是同图固定变换推理。

比较规则：固定 checkpoint、grid/refinement、occlusion；不得跨 759921 与 Fusion v2 直接比较未校准
概率；不得将 attention 或 occlusion map 称作像素级 ground truth；引用结果时保留输入与
checkpoint SHA。
