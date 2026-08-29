# 759921 portable explainability extension

该包保留 759921 checkpoint、配置与训练扩展代码，并附带兼容推理 runtime。它新增
patch attribution、分支 neutralization、tile 证据和 16 变换轨迹，不会更新权重。
当前版本使用 coarse-to-fine 多尺度遮挡和 raw-logit 贡献，另提供模型相关的 wavelet-only
局部高频反事实图，以及与模型输出无关的输入高频纹理图作为对照。

## 运行

```bash
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

打开 `explanations/example/index.html`。759921 的 confidence 是原始 sigmoid，没有经过
Platt/temperature calibration；不要把 `checkpoint/calibration.json` 的分类阈值误当成
概率校准参数。

新版配对样例见 [`demos-v2/index.html`](demos-v2/index.html)。整体实际模型路径见
[`MODEL_ARCHITECTURE.svg`](MODEL_ARCHITECTURE.svg)。输入纹理图与 checkpoint 无关；
wavelet-only 图则固定其他 fusion 特征，仅允许模型真实的 `wavelet_similarity` 变化。

更完整的字段语义见 `AGENT_GUIDE.md`，历史训练代码在 `legacy_training_src`，训练配置在
`configs`。所有解释图均为模型局部反事实，不是像素级伪造真值。
