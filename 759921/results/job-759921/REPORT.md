# 组合版 Class/Source GroupDRO：作业 759921

## 结论

组合训练成功解决了 DALL·E 配对集上两版模型相反的错误模式：clean balanced accuracy
达到 **94.99%**，COCO real recall 为 **98.66%**，DALL·E recall 为 **91.31%**。
16 项变换平均 balanced accuracy 为 **93.09%**，最差 noise σ=0.10 仍有
**87.80%**。

但该 checkpoint 不是对所有生成器统一占优：在 COCO/MidJourney 上 clean balanced
accuracy 只有 76.40%，主要仍是假阴性；16 项平均为 73.28%。因此它适合作为
DALL·E/OOD 专家或后续集成成员，不应直接替换此前 MidJourney GroupDRO 模型。

## 方法与数据

- 主干：初版 DINOv2-B/14 + SigLIP SO400M/14 + native tiles + wavelet 分支。
- 参数：515,587,535 总参数，1,637,007 可训练参数；从初版 job 756627 的 epoch 6
  checkpoint 初始化，不恢复 optimizer。
- 优化：按 `(label, source)` 做 GroupDRO；real/fake 分别进行 top-25% 难例强化；
  加入两类 surrogate risk gap 惩罚，防止假阳性或假阴性单边塌缩。
- 训练数据：初版 20k 与第二版 20k 合并。逐文件 SHA-256 去重后为 39,985 张：
  20,000 real + 19,985 fake。
- validation：5,298；calibration：4,700；旧测试：3,999；COCO/MidJourney：2,000。
- role 内去除重复 10 项，跨 role 从低优先级 role 隔离 8 项；最终跨 role SHA-256
  重叠为 0。
- DALL·E 未参与训练、checkpoint 选择或阈值选择。由于其结果在前序实验中已被观察，
  本报告将本轮结果称为诊断，而不是新的严格盲测。

## Smoke 与资源

- 端到端 smoke 759915：COMPLETED，3分24秒。
- batch=192 精确显存 smoke 759918：COMPLETED，35秒；allocated/reserved
  31.98/36.79 GiB。
- 正式 job 759921：COMPLETED，ExitCode 0，端到端 58分12秒；纯训练 30分44秒。
- 正式训练 PyTorch peak allocated/reserved：32.87/36.79 GiB；`nvidia-smi`
  采样峰值 80,698 MiB / 95,830 MiB。

## 训练过程

| Epoch | Train loss | Val AUROC | Val balanced acc. @ balanced threshold | Worst-class recall | 选择 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.8394 | 86.99% | 81.69% | 80.79% |  |
| 1 | 0.4517 | 87.34% | 81.28% | 80.75% |  |
| 2 | 0.4414 | **87.87%** | **81.82%** | **80.98%** | best |
| 3 | 0.4707 | 87.80% | 81.45% | 80.98% |  |

独立 calibration 的阈值为 1.0；模型 BF16 logits 有大量概率精确饱和为 1，因此该阈值
仍会产生正预测。Calibration real/fake recall 为 77.70% / 85.53%，balanced accuracy
81.62%。X-ray real recall 仍只有 0.20%，说明医学域的表示偏移没有被单阈值修复。

## DALL·E 三版对比

| 模型 | Clean balanced acc. | COCO real recall | DALL·E recall | AUROC | 主要错误 |
|---|---:|---:|---:|---:|---|
| 初版双分支 | 66.71% | 33.42% | 100.00% | 92.93% | 假阳性过多 |
| 第二版小型 GroupDRO | 48.80% | 94.18% | 3.41% | 42.41% | 假阴性过多 |
| **组合版** | **94.99%** | **98.66%** | **91.31%** | **98.48%** | 两类基本平衡 |

## DALL·E 全变换

| 变换 | Balanced acc. | Real recall | DALL·E recall | AUROC |
|---|---:|---:|---:|---:|
| clean | 94.99% | 98.66% | 91.31% | 98.48% |
| JPEG 90 | 95.19% | 98.26% | 92.11% | 98.40% |
| JPEG 70 | 94.72% | 98.86% | 90.57% | 98.50% |
| JPEG 50 | 96.59% | 98.06% | 95.12% | 98.58% |
| JPEG 30 | 96.12% | 99.20% | 93.05% | 99.05% |
| Blur 0.5 | 95.22% | 98.40% | 92.05% | 98.44% |
| Blur 1.0 | 92.35% | 98.20% | 86.50% | 97.62% |
| Blur 2.0 | 90.88% | 98.20% | 83.56% | 97.32% |
| Resize 0.5× | 95.76% | 98.20% | 93.32% | 98.29% |
| Resize 0.25× | 93.42% | 98.93% | 87.90% | 98.44% |
| Noise 0.02 | 90.44% | 98.93% | 81.95% | 97.25% |
| Noise 0.05 | 87.83% | 98.86% | 76.80% | 96.50% |
| Noise 0.10 | **87.80%** | 98.13% | 77.47% | 95.40% |
| Color −20% | 93.18% | 95.66% | 90.71% | 96.39% |
| Color +20% | 94.45% | 98.53% | 90.37% | 98.30% |
| Center crop 80% | 90.54% | 95.66% | 85.43% | 95.30% |

16 项平均：balanced accuracy 93.09%、real recall 98.17%、DALL·E recall 88.01%、
AUROC 97.64%。

## 其他测试集

| 测试集 | Clean BA | Clean real recall | Clean fake recall | Mean BA | Worst BA | Mean AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CIFAKE/WildFake | 88.27% | 99.95% | 76.59% | 81.62% | 72.39% | 99.50% |
| COCO/MidJourney | 76.40% | 99.20% | 53.60% | 73.28% | 60.10% | 95.35% |
| COCO/DALL·E | **94.99%** | **98.66%** | **91.31%** | **93.09%** | **87.80%** | **97.64%** |

## 审计文件

- 训练增强记录：159,744 条；validation 记录：21,192 条。
- 逐图固定变换：旧测试 63,984 条、COCO/MidJourney 32,000 条、DALL·E 47,872 条。
- checkpoint SHA-256：
  `2df35326275741b3889121f4e8c89b06533aa97194ca4b4faae428d17b03023e`。
- 原始 checkpoint、所有 epoch、loss、逐图增强/预测、manifest、receipt、脚本和 Slurm
  日志均保存在本结果目录及对应归档中。
