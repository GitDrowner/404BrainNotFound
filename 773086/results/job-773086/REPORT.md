# 759921 架构 + MLP 归一化动态损失：作业 773086

## 结论

作业 773086 正常完成（`COMPLETED`, ExitCode `0:0`）。模型保持 759921 的检测架构不变，
在 759921 + 770876 的 44,377 张训练集上，用共享 MLP 动态分配五项辅助任务的权重，并按
训练数据系统内部 held-out validation 的 AUROC 保存 checkpoint。最佳 checkpoint 为
**epoch 2**，validation AUROC 为 **88.07%**。

以官方主要指标 AUROC 看，本轮在 COCO/DALL·E 3 外部诊断集上表现最好：clean AUROC
**98.96%**，16 项条件平均 AUROC **98.34%**。相比原始 759921，分别提升约 0.49 和
0.70 个百分点；相比 770876，提升约 0.02 和 0.14 个百分点。但在 COCO/MidJourney
上 clean/平均 AUROC 为 **94.39% / 93.19%**，低于原始 759921 的
96.18% / 95.35%。因此它强化了目标 DALL·E 3 域的排序泛化，但没有在所有生成器域上统一
超过 759921。

原始 calibration 脚本给出的阈值 `0.700067` 仍然过于保守：COCO/DALL·E clean 的
COCO real recall 为 99.93%，DALL·E recall 只有 59.63%。只在独立 calibration 集上
最大化 class-balanced accuracy，可得到复核阈值 **0.281519**；此时 COCO/DALL·E clean
balanced accuracy 为 **94.65%**，real/fake recall 为 **93.38% / 95.92%**。
AUROC 是阈值无关指标，因此原阈值问题不影响以置信度排序计算的官方 AUROC，但会影响前端
二分类标签、accuracy 和 recall 的展示。

## 方法与数据边界

- 架构：DINOv2-B/14 forensic backbone + SigLIP SO400M/14 semantic backbone +
  四个 native-resolution tiles + Haar 高频分支；与 759921 相同。
- 初始化：从原始 759921 `hybrid_groupdro_h100/best.pt` 初始化，不恢复 optimizer。
- 参数：515,587,535 总参数，1,637,007 可训练参数，513,950,528 冻结参数。
- 主目标：Class/Source GroupDRO、real/fake top-25% hard-example objective 和 class
  risk-gap；主分类 anchor 固定为 1.0。
- 辅助目标：consistency、feature consistency、degradation classification、
  degradation severity、tile auxiliary classification。
- 训练集：44,377 张，20,400 real + 23,977 fake；其中为 759921 的 39,985 张去重语料，
  加上 770876 使用的 4,392 张泄漏检查通过的 OpenFake/TIGAS/本地生成补充数据。
- Validation：5,298 张（2,650 real + 2,648 fake）；calibration：4,700 张；
  COCO/MidJourney test：2,000 张；COCO/DALL·E 3 external diagnostic：2,992 张。
- DALL·E 3 不参与训练、validation、calibration、MLP 学习或 checkpoint 选择。Slurm
  流程在 `best.pt` 冻结和 calibration 完成后才打开 DALL·E 3 外部集。
- 训练 manifest 明确拒绝 DALL·E 3 generator 名称及 DALL·E Advanced 精确 SHA-256
  重叠；本轮训练中没有 DALL·E 3 条目。其他非 DALL·E 3 的 OpenAI-family 生成器不是
  一并排除对象。

| 训练来源组 | Real | Fake | 合计 |
|---|---:|---:|---:|
| CIFAKE | 4,856 | 4,709 | 9,565 |
| WildFake | 4,855 | 4,722 | 9,577 |
| SID | 289 | 554 | 843 |
| 多领域真实图（Food101/Flickr30k/EuroSAT/Cars/WikiArt/Pets） | 10,000 | 0 | 10,000 |
| 早期生成器（BigGAN/ADM） | 0 | 10,000 | 10,000 |
| OpenFake core train | 400 | 3,600 | 4,000 |
| TIGAS 多生成器 | 0 | 380 | 380 |
| 本地 train-only 生成图 | 0 | 12 | 12 |
| **总计** | **20,400** | **23,977** | **44,377** |

虽然 manifest 的 fake 数略多，训练 sampler 按 label 平衡采样；六轮实际读取的
real/fake 数分别为 133,135 / 132,977，避免类别数量差异直接决定梯度方向。

Validation 使用按样本下标确定的 `deterministic_eval_variant`：整个 5,298 张数据集稳定地
轮转覆盖 clean 与 15 项单变换，同一 manifest 重复评测一致。需要注意，每张 validation
图片只评一个固定 variant，而不是每张图片都复制为 16 个版本；每轮审计为 5,298 条，
不是 84,768 条。因此 checkpoint 选择指标是“固定混合鲁棒性 validation AUROC”。

## MLP 动态损失权重

每项辅助 loss 先除以自身 detached EMA，得到无量纲相对难度。一个无 task ID、无
task-specific bias 的共享 MLP 根据当前相对难度、相对初始 loss 趋势和训练进度打分，再经
带 0.05 下限的 softmax 映射到总和为 1 的权重单纯形。前 10% optimizer steps 使用均匀
权重，之后 controller 通过 adversarial gradient proxy 强调当前较难的辅助任务。

整体目标可写为：

`L = 1.0 × L_primary + 1.0 × Σ stop_grad(w_i) × L_i / EMA(L_i)`

这里两个 `1.0` 仅定义主任务与“全部辅助任务预算”的整体比例；它们不是五项辅助 loss 的
人工相对系数。五项相对权重由 MLP 按 batch 动态产生。

| Loss group | 初始权重 | Best epoch 2 整轮平均 | Best epoch 2 最后一批 | 全训练最后一批 |
|---|---:|---:|---:|---:|
| Consistency | 20.00% | 14.24% | 5.08% | 5.00% |
| Feature consistency | 20.00% | 16.92% | 78.48% | 5.00% |
| Degradation classification | 20.00% | 20.09% | 6.24% | 79.99% |
| Degradation severity | 20.00% | 24.60% | 5.07% | 5.00% |
| Tile auxiliary | 20.00% | 24.16% | 5.13% | 5.00% |

权重存在明显的逐 batch 切换：MLP 经常把约 80% 辅助预算放到当前最难的一项，其余各项停在
5% anti-collapse 下限。因此 `summary.json` 中“最后一批 degradation classification
79.99%”不能解释为一个全局静态最优系数；更合理的可解释口径是报告权重轨迹、epoch 均值和
分布。它消除了五个人工 task-specific 系数，但仍保留三个任务无关设计先验：主分类 anchor、
总辅助预算和 5% 最小权重。

## 作业与资源

- 作业：773086，`gpu-long`，H100 NVL MIG 3g.47gb。
- 端到端耗时：**1小时19分10秒**；纯训练耗时：**57分24秒**。
- batch size 192，16 workers，6 epochs；每轮 231 steps、44,352 次取样，总计
  1,386 optimizer steps 和 266,112 条训练增强审计。
- 训练结束吞吐约 78.6 samples/s。
- PyTorch peak allocated/reserved：33.19 / 37.11 GiB。
- `nvidia-smi` 设备级采样峰值为 79,358 MiB / 95,830 MiB；MIG 环境未返回可用的
  utilization 数值，因此不报告平均 GPU utilization。

## 训练过程

| Epoch | Train loss | Val acc. @ 0.5 | Val real recall | Val fake recall | Val AUROC | Val AP | 选择 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.5640 | **86.11%** | **75.28%** | **96.94%** | 87.82% | 84.88% |  |
| 1 | 1.3953 | 84.84% | 75.09% | 94.60% | 86.86% | 83.17% |  |
| 2 | 1.2995 | 84.67% | 72.68% | 96.68% | **88.07%** | **85.45%** | **best** |
| 3 | 1.2619 | 85.09% | 74.79% | 95.39% | 87.73% | 85.00% |  |
| 4 | 1.2484 | 85.24% | 74.38% | 96.11% | 87.89% | 85.19% |  |
| 5 | **1.2430** | 85.13% | 74.08% | 96.19% | 87.96% | 85.32% |  |

Train loss 从 1.5640 降到 1.2430，下降 20.5%。Validation AUROC 在 86.86%–88.07%
之间波动，epoch 2 达峰后没有继续改善；validation cross-entropy loss 同时较高且不单调，
说明排序质量较稳定，但 raw logit 的 operating point/校准并未随训练 loss 同步改善。
`best.pt` 严格按内部 validation AUROC 选择，未使用外部测试结果。

## Calibration 与阈值复核

独立 Platt calibration 为：

`p_calibrated = sigmoid(raw_logit / 3.164879 - 2.406554)`

原脚本为照顾 calibration 中最差 source recall 选择阈值 `0.700067`；但 X-ray real 域存在
严重 domain shift，导致该阈值在外部测试上过度偏向 real。保持 calibrator 和 checkpoint
不变，仅在同一 calibration 集上最大化：

`0.5 × (real recall + fake recall)`

得到复核阈值 `0.281519`。它在 calibration 上的 real/fake recall 为
76.51% / 96.55%，balanced accuracy 为 86.53%。这个复核没有使用 COCO/MidJourney 或
DALL·E 3 标签。

| 测试集 | 阈值 | Clean balanced acc. | Clean real recall | Clean fake recall | 16项平均 balanced acc. |
|---|---:|---:|---:|---:|---:|
| COCO/MidJourney | 原始 0.700067 | 54.90% | 99.80% | 10.00% | 53.81% |
| COCO/MidJourney | **复核 0.281519** | **82.75%** | 96.70% | **68.80%** | **79.12%** |
| COCO/DALL·E 3 | 原始 0.700067 | 79.78% | 99.93% | 59.63% | 77.49% |
| COCO/DALL·E 3 | **复核 0.281519** | **94.65%** | 93.38% | **95.92%** | **93.29%** |

## COCO/DALL·E 3 全变换

下表 accuracy/recall 使用 calibration-only 复核阈值 0.281519；AUROC 与 AP 不依赖阈值。

| 变换 | Balanced acc. | COCO real recall | DALL·E recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|
| Clean | 94.65% | 93.38% | 95.92% | 98.96% | 99.05% |
| JPEG 90 | 94.82% | 93.38% | 96.26% | 99.10% | 99.17% |
| JPEG 70 | 94.32% | 94.65% | 93.98% | 98.84% | 98.94% |
| JPEG 50 | 95.19% | 94.12% | 96.26% | 99.15% | 99.21% |
| JPEG 30 | 94.92% | 96.86% | 92.98% | 99.07% | 99.14% |
| Gaussian blur sigma=0.5 | 94.52% | 93.45% | 95.59% | 98.90% | 99.00% |
| Gaussian blur sigma=1.0 | 93.32% | 92.05% | 94.59% | 98.53% | 98.67% |
| Gaussian blur sigma=2.0 | 92.28% | 93.72% | 90.84% | 97.89% | 98.11% |
| Resize 0.5x | 94.12% | 91.71% | 96.52% | 98.87% | 98.99% |
| Resize 0.25x | 93.68% | 95.72% | 91.64% | 98.45% | 98.63% |
| Gaussian noise sigma=0.02 | 93.75% | 94.39% | 93.11% | 98.56% | 98.75% |
| Gaussian noise sigma=0.05 | 92.68% | 94.52% | 90.84% | 98.19% | 98.45% |
| Gaussian noise sigma=0.10 | 91.38% | 92.51% | 90.24% | 97.50% | 97.89% |
| Color jitter -20% | 90.51% | 85.76% | 95.25% | 97.58% | 97.81% |
| Color jitter +20% | **95.59%** | 95.19% | 95.99% | **99.24%** | **99.32%** |
| Center crop 80% | **87.00%** | 85.90% | 88.10% | **94.54%** | **95.26%** |

16 项平均：balanced accuracy **93.29%**、COCO real recall 92.96%、DALL·E recall
93.63%、AUROC **98.34%**、AP 98.52%。最弱项为 center crop 80%，说明丢失外围内容与
重新放大会同时削弱两类 recall；强噪声其次。

## COCO/MidJourney 全变换

| 变换 | Balanced acc. | COCO real recall | MidJourney recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|
| Clean | 82.75% | 96.70% | 68.80% | 94.39% | 93.93% |
| JPEG 90 | 81.15% | 96.10% | 66.20% | 93.35% | 92.73% |
| JPEG 70 | 80.80% | 95.80% | 65.80% | 92.75% | 92.46% |
| JPEG 50 | 79.55% | 96.70% | 62.40% | 93.70% | 93.36% |
| JPEG 30 | 72.60% | 98.10% | 47.10% | 93.09% | 92.16% |
| Gaussian blur sigma=0.5 | 82.85% | 97.00% | 68.70% | 95.29% | 94.97% |
| Gaussian blur sigma=1.0 | 83.45% | 95.80% | 71.10% | 94.93% | 94.50% |
| Gaussian blur sigma=2.0 | 84.15% | 95.70% | 72.60% | 95.24% | 95.00% |
| Resize 0.5x | 83.80% | 95.10% | 72.50% | 94.19% | 93.82% |
| Resize 0.25x | 84.45% | 96.50% | 72.40% | **95.80%** | **95.30%** |
| Gaussian noise sigma=0.02 | 73.30% | 97.20% | 49.40% | 91.92% | 91.26% |
| Gaussian noise sigma=0.05 | 67.30% | 97.40% | 37.20% | 89.77% | 88.47% |
| Gaussian noise sigma=0.10 | **63.35%** | 95.10% | **31.60%** | **86.34%** | **82.96%** |
| Color jitter -20% | 84.65% | 91.40% | 77.90% | 92.68% | 92.00% |
| Color jitter +20% | 77.05% | 97.40% | 56.70% | 93.81% | 93.32% |
| Center crop 80% | **84.65%** | 94.70% | 74.60% | 93.87% | 93.77% |

16 项平均：balanced accuracy **79.12%**、COCO real recall 96.04%、MidJourney recall
62.19%、AUROC 93.19%、AP 92.50%。主要短板仍是 MidJourney 假阴性，Gaussian noise
sigma=0.10 下 recall 降到 31.60%。

## 与 759921、770876 对比

以下 AUROC 全部来自冻结 checkpoint 的 post-selection test，不受报告中的阈值复核影响。

| 测试集 | 模型 | Clean AUROC | 16项平均 AUROC | Clean BA | 16项平均 BA |
|---|---|---:|---:|---:|---:|
| COCO/DALL·E 3 | 原始 759921 | 98.48% | 97.64% | **94.99%** | 93.09% |
| COCO/DALL·E 3 | 770876 + calibration-balanced 阈值 | 98.94% | 98.20% | 94.08% | 92.60% |
| COCO/DALL·E 3 | **773086 + 0.281519** | **98.96%** | **98.34%** | 94.65% | **93.29%** |
| COCO/MidJourney | 原始 759921 | **96.18%** | **95.35%** | 76.40% | 73.28% |
| COCO/MidJourney | 770876 + calibration-balanced 阈值 | 94.70% | 93.27% | 82.25% | **79.67%** |
| COCO/MidJourney | **773086 + 0.281519** | 94.39% | 93.19% | **82.75%** | 79.12% |

773086 的清晰收益是：去除五项人工相对系数后，内部 validation AUROC 和目标
COCO/DALL·E 3 AUROC 均达到三版最高；代价是 COCO/MidJourney 排序性能没有恢复到原始
759921。按赛题 AUROC 目标，773086 是当前 DALL·E 3 目标域上更合适的单模型候选；若强调
多生成器统一泛化，759921 仍是有价值的 ensemble 成员。

## 审计与产物

- Best checkpoint SHA-256：
  `e32b6e19968f3ff28c39cd7f2457a8d1eb3acd10a72618cd8a75071a2914862a`。
- 初始 759921 checkpoint SHA-256：
  `2df35326275741b3889121f4e8c89b06533aa97194ca4b4faae428d17b03023e`。
- 训练增强审计：266,112 条；validation 审计：31,788 条。
- COCO/MidJourney 固定变换审计：32,000 条；COCO/DALL·E 3 固定变换审计：
  47,872 条。
- 本地结果归档 SHA-256：
  `bc438539cc9df182a84ed360dd9fb87c5c481d28490c4b0612ec8e90975a7659`。
- 原始结果位于 `outputs/remote-773086/extracted/outputs/hybrid_759921_mlp_controller_h100/`；
  Slurm 日志位于 `outputs/remote-773086/extracted/logs/`。

报告中的复核阈值只重新解释冻结的逐图 calibrated probability，没有改动 checkpoint、
calibrator、图像变换或样本顺序。DALL·E 3 已在此前实验中被观察，因此仍应称为外部诊断集，
而不是新的严格盲测集。
