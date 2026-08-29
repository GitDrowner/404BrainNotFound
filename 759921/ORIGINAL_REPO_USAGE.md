# 759921 可解释展示功能

这是对 759921 DINOv2-B/14 + SigLIP SO400M + native tiles + wavelet 模型的继续开发，
不会训练或改变原 checkpoint。它增加单图 patch 遮挡贡献、global/semantic/tiles 分支
neutralization、tile attention/tile logits 和 16 条件置信度轨迹。

## 在原仓库运行

从仓库根目录执行：

```bash
PYTHONPATH=src python test/results/result-759921-hybrid/scripts/explain_hybrid.py \
  --checkpoint test/results/result-759921-hybrid/outputs/hybrid_groupdro_h100/best.pt \
  --image /path/to/image.png \
  --output test/results/result-759921-hybrid/explanations/example \
  --grid 6 --refine-top-k 6 --refine-grid 3 \
  --occlusion blur --device auto
```

759921 的 `calibration.json` 只有分类阈值，并不是概率校准器，因此不应作为
`--calibration` 传入。输出概率是 checkpoint 原始 sigmoid confidence；展示时要明确它
未经 Platt/temperature calibration。

## 展示内容

- `patch_attribution.png`：coarse-to-fine 区域的 raw-logit 遮挡贡献；
- `heatmap_attribution.png`：独立高对比 raw-logit 热图，避免 sigmoid 饱和掩盖贡献；
- `heatmap_attribution_overlay.png`：上述贡献与原图的强叠加版本；
- `heatmap_texture.png`：逐像素输入高频残差强度，完全不使用模型输出；
- `heatmap_frequency_contribution.png`：固定其他 fusion 特征，只改变模型实际
  `wavelet_similarity` 后的最终 raw-logit 贡献；
- `heatmap_frequency_overlay.png`：wavelet-only 贡献与原图的叠加版本；
- `components.svg`：global+wavelet、semantic、native tiles 被中性化后的 confidence delta；
- `explanation.json/tiles`：四个 native tile 的 attention 和 auxiliary raw logit；
- `transform_trajectory.svg`：完整 16 条件轨迹；
- `index.html`：单页展示；`patches.jsonl` 和 `schema.json`：机器读取。

注意：tile attention 是模型的聚合权重，patch attribution 是输入遮挡反事实，不能把前者
冒充后者。输入高频强度不等于模型贡献；只有 wavelet-only 图经过 checkpoint forward。
红色区域只表示局部反事实下提高 raw logit，不是伪造区域真值。
