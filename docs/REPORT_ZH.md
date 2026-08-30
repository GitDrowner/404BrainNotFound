# test2 模型 DALL·E 3 单类外部验证报告

## 1. 评测口径

| 项目 | 取值 |
| --- | --- |
| 任务类型 | 纯推理外部验证；`training=false`，未创建 optimizer，未修改权重 |
| 模型 | DINOv2 ViT-S/14 + Fourier 风格随机化一致性学习（`fda_consistency`） |
| 总参数 / 可训练参数 | 21,936,407 / 3,858,839 |
| Checkpoint | `/home/mingjun/Tiktok_Hackthon/test-2/outputs/main/best.pt`（epoch 0） |
| Checkpoint SHA-256 | `b8d0a9a05779aa970da59235c2af41b73c376b074a33e2232f65fab79193350a` |
| 主判定阈值 | 0.235；来自 test2 内部 calibration，未在本外部集上重定阈值 |
| 对照阈值 | 固定 0.5；仅从同一批预测附带统计 |
| 正类 | DALL·E 3 / AI，label=1 |
| 外部验证集 | Defactify 官方 validation split 中隔离后的 1,496 张 DALL·E 3 |
| 评测规模 | 16 variants × 1,496 = 23,936 次图像推理 |

本报告只评测 DALL·E 3 正类，没有加入 COCO 或其他负类。因此可以报告 AI Recall/检出率、漏检率和分数分布，不能从这份单类结果计算 Accuracy、specificity、balanced accuracy 或 AUROC。

## 2. 数据来源、隔离与完整性

数据来自 `Rajarshi-Roy-research/Defactify_Image_Dataset`，固定 revision `787334f7857fa54f29027a7f09c30e895ad486ef`。复用参考报告已完成的配对隔离：从 1,500 对候选中隔离 4 对后，保留 1,496 张 DALL·E 3。本次只读取这些 DALL·E 3 图像，不读取配对 COCO 图像。

| 检查 | 结果 |
| --- | ---: |
| 外部数据 gate | `PASS_WITH_PAIRED_QUARANTINE` |
| DALL·E 3 图像数 | 1,496 |
| 缺失文件 | 0 |
| SHA-256 不符 | 0 |
| 与正式训练集 SHA-256 精确重叠 | 0 |
| 与正式训练集 dHash 近重复（半径 <=4） | 0 |
| test2 正式训练 manifest SHA-256 | `b1e04b16ce7313806483f687854a2a796412eb65671984a3aeb9078604b646bd` |
| DALL·E 3 单类 manifest SHA-256 | `9168caa58742c23b9489e7113a6b4e24d2bb44251dbd85f51b26a6e1ed4f1373` |

## 3. 固定阈值结果

主阈值为 0.235；固定 0.5 仅作对照。

| 变换 | 参数 | 检出 / 1,496 | Recall@0.235 | 漏检率 | Recall@0.5 | 平均分数 | P05 / P50 / P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 无 | 46 | 3.075% | 96.925% | 2.473% | 0.0261 | 0.0001 / 0.0002 / 0.0208 |
| jpeg_q90 | quality=90 | 46 | 3.075% | 96.925% | 2.340% | 0.0259 | 0.0001 / 0.0002 / 0.0194 |
| jpeg_q70 | quality=70 | 47 | 3.142% | 96.858% | 2.406% | 0.0265 | 0.0001 / 0.0002 / 0.0223 |
| jpeg_q50 | quality=50 | 47 | 3.142% | 96.858% | 2.406% | 0.0264 | 0.0001 / 0.0002 / 0.0228 |
| jpeg_q30 | quality=30 | 51 | 3.409% | 96.591% | 3.008% | 0.0291 | 0.0001 / 0.0002 / 0.0289 |
| blur_s0.5 | sigma=0.5 | 47 | 3.142% | 96.858% | 2.473% | 0.0263 | 0.0001 / 0.0002 / 0.0219 |
| blur_s1.0 | sigma=1.0 | 47 | 3.142% | 96.858% | 2.406% | 0.0268 | 0.0001 / 0.0002 / 0.0269 |
| blur_s2.0 | sigma=2.0 | 56 | 3.743% | 96.257% | 3.075% | 0.0315 | 0.0001 / 0.0002 / 0.0549 |
| resize_x0.5 | 0.5x 后恢复 | 47 | 3.142% | 96.858% | 2.674% | 0.0275 | 0.0001 / 0.0002 / 0.0320 |
| resize_x0.25 | 0.25x 后恢复 | 61 | 4.078% | 95.922% | 3.275% | 0.0352 | 0.0001 / 0.0003 / 0.0869 |
| noise_s0.02 | sigma=0.02 | 47 | 3.142% | 96.858% | 2.473% | 0.0269 | 0.0001 / 0.0002 / 0.0257 |
| noise_s0.05 | sigma=0.05 | 54 | 3.610% | 96.390% | 2.741% | 0.0295 | 0.0001 / 0.0002 / 0.0394 |
| noise_s0.10 | sigma=0.10 | 62 | 4.144% | 95.856% | 3.342% | 0.0341 | 0.0001 / 0.0002 / 0.0878 |
| color_jitter_minus20 | 亮度/对比度/饱和度=0.8 | 47 | 3.142% | 96.858% | 2.473% | 0.0260 | 0.0001 / 0.0002 / 0.0191 |
| color_jitter_plus20 | 亮度/对比度/饱和度=1.2 | 46 | 3.075% | 96.925% | 2.540% | 0.0269 | 0.0001 / 0.0002 / 0.0238 |
| center_crop_80 | 中心裁剪 80% 后恢复 | 47 | 3.142% | 96.858% | 2.674% | 0.0282 | 0.0001 / 0.0002 / 0.0428 |

15 个失真设置的平均 DALL·E 3 Recall：主阈值 3.351%，固定 0.5 为 2.687%。主阈值下最低 Recall 出现在 `jpeg_q90`（3.075%），最高出现在 `noise_s0.10`（4.144%）。

## 4. 统计解读

Clean 下检出 46/1,496 张 DALL·E 3，Recall 为 3.075%，漏检 1,450 张（96.925%）。预测为 AI 的概率均值为 0.0261，中位数为 0.0002。固定 0.5 时 Recall 为 2.473%。

同一 checkpoint 在内部 test 的 clean fake recall 为 88.700%，而 DALL·E 3 外部 clean recall 为 3.075%，相差 -85.625 个百分点。内部 fake 来源与 DALL·E 3 不同，这一差异应解释为跨生成器 domain shift，不应把外部集用于事后调阈值后再声称无偏结果。

由于本次没有负类，仅凭 DALL·E 3 Recall 不能证明整体分类性能；是否以真图高误报为代价，需要结合原内部测试或另一个严格隔离的 real 集评估。

## 5. 时间、资源与审计文件

| 项目 | 结果 |
| --- | ---: |
| 退出状态 | `DALLE3_ONLY_RC=0` |
| 评测 wall time | 176.1 秒 |
| GPU | NVIDIA RTX A5000 |
| PyTorch 进程峰值 CUDA allocation | 254.8 MiB |
| 逐图片预测审计记录 | 23,936 条 |

结果目录 `/home/mingjun/Tiktok_Hackthon/test-2/external_eval_defactify_dalle3_only/evaluation`：

- `dalle3_only_metrics.csv`：16 个 variants 的检出率、漏检率与概率分布；
- `dalle3_only_predictions.jsonl`：每张图片、每个 variant 的预测分数与增强记录；
- `evaluation_metadata.json`：checkpoint、manifest、隔离 gate、运行环境与纯推理状态；
- `summary.json`：机器可读汇总；
- `gpu_memory.csv`：每 10 秒 GPU 采样；
- `REPORT_ZH.md`：本报告。

核心审计文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `dalle3_only_metrics.csv` | `0e75247f70c74732dfde18b48f268ed650937929a72098e290f1980f31b039ad` |
| `dalle3_only_predictions.jsonl` | `3eb81b273dbd55e6f5e4ee1461e04773f2f5731b6a3a4ba9a1598b64ab8c3c28` |
| `evaluation_metadata.json` | `e8fbda623aad9c4c42b96e558bcc88c39b87b0edecd4fbf0d790138800772328` |
| `summary.json` | `006a9d240ac85085fe6a2329c03a47494df1f7cda68e024cb9b1b5d2948643fa` |
