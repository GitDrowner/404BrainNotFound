# 773086 消融实验计划

状态：**仅设计，暂不执行**。本文档不授权创建训练配置、修改训练代码、上传集群或提交
Slurm 作业。

## 1. 目标

以 job 773086 为唯一完整基线，通过单变量实验回答四个问题：

1. SigLIP 语义、DINOv2 全局/局部 tile、Haar wavelet 各自贡献多少？
2. MLP-normalized 动态辅助损失是否优于无人工比例的简单方案？
3. GroupDRO/难例/类别风险平衡与竞赛增强是否真正提高变换鲁棒性？
4. 759921 之后增加的 4,392 张多生成器数据是否提高跨生成器泛化？

官方主要指标为 AUROC。Accuracy、balanced accuracy、real recall、fake recall 和 FPR
作为 operating-point 指标，不替代 AUROC。

## 2. 固定基线

基线 `B0-773086` 使用冻结产物，不重新训练：

| 项目 | 固定值 |
|---|---|
| Checkpoint | job 773086 best epoch 2 |
| Checkpoint SHA-256 | `e32b6e19968f3ff28c39cd7f2457a8d1eb3acd10a72618cd8a75071a2914862a` |
| 架构 | DINOv2-B/14 + SigLIP SO400M/14 + 4 tiles + Haar wavelet |
| 总/可训练参数 | 515,587,535 / 1,637,007 |
| Train | 44,377：20,400 real + 23,977 fake |
| Validation | 5,298，内部 held-out，固定混合 16 variants |
| Calibration | 4,700，与 train/validation/test 隔离 |
| 训练 | 6 epochs，1,386 optimizer steps，batch 192，BF16 |
| Checkpoint selection | 内部 validation AUROC |
| Baseline val AUROC | 88.07% |
| COCO/DALL·E clean / 16项平均 AUROC | 98.96% / 98.34% |
| COCO/MidJourney clean / 16项平均 AUROC | 94.39% / 93.19% |

所有消融都从原始 759921 checkpoint 初始化，和 773086 一样不恢复 optimizer。外部
DALL·E 3、MidJourney 和其他 OOD 测试集只允许在 best checkpoint 冻结后打开。

## 3. 公平性约束

除被消融因素外，以下内容必须完全一致：

- train/validation/calibration/test manifest 及其 SHA-256；
- 随机种子 `20260830`、样本顺序和 augmentation RNG；
- batch 192、6 epochs、学习率、scheduler、weight decay、gradient clipping；
- LoRA blocks/rank/alpha、dropout、输入尺寸和 tile 切分；
- 1,386 optimizer steps。数据消融也固定 optimizer steps，不能因 manifest 变小而少训练；
- checkpoint 一律按内部 validation AUROC 选择；
- 每个实验使用自己的 calibration 分数，在同一 4,700 张 calibration 集上重新拟合
  Platt temperature/bias，并以 class-balanced accuracy 选择阈值；
- 展示层将各自 calibration 阈值线性映射为 0.5。该映射严格保序，不纳入模型消融结论；
- 外部测试数据不得参与 early stopping、checkpoint、阈值或实验取舍。

每个配置应由基线 YAML 机械生成，并附一份 machine-readable config diff。若一次实验改变
两个非耦合因素，该结果不得称为单因素消融。

## 4. 分支消融的实现口径

直接删除分支会改变 fusion 维度，使 759921 初始化权重无法公平加载。因此第一轮采用
**functional masking**：保持模块、tensor shape、参数量和初始化一致，只在训练与推理时将
目标信息通路替换为固定中性值。

| 分支 | 中性化方式 | 仍保留内容 |
|---|---|---|
| Semantic | fusion 前将 SigLIP embedding 置零 | DINO global、tiles、wavelet |
| Wavelet | 将 similarity 固定为 1.0 | DINO、tiles、SigLIP |
| Tile fusion | 将 attention-pooled tile embedding 置零 | global DINO、SigLIP、wavelet；tile auxiliary 暂保留 |

这种设计回答“该信息通路是否有用”，而不是“删掉模块后能省多少算力”。若某分支无显著
收益，再单独做 structural removal 测参数、延迟和显存；两类结论不可混为一谈。

## 5. 第一阶段：最小完整消融集

第一阶段共 9 次新训练。它们是最终报告至少应包含的主表。

| ID | 唯一变化 | 回答的问题 | 预期重点 |
|---|---|---|---|
| A1-SEM | semantic embedding 中性化 | SigLIP 是否贡献跨域语义信息 | DALL·E、语义复杂图、center crop |
| A2-WAV | wavelet similarity 固定为 1 | 高频扰动相似度是否贡献 forensic 信号 | JPEG、blur、resize、noise |
| A3-TILE | tile pooled feature 中性化 | 局部证据是否改善全图判断 | 局部缺陷、crop、可解释性 |
| A4-MLP | MLP controller 改为 EMA-normalized 等权 `0.2` | 动态权重是否优于无人工比例的简单基线 | AUROC、权重稳定性、训练方差 |
| A5-AUX | 移除五项辅助目标，仅保留同一 robust primary loss | 辅助任务整体是否有效 | clean 与变换平均 AUROC |
| A6-ROBUST | primary 改为普通 sample-mean BCE，其他不变 | GroupDRO + hard mining + risk gap 整体价值 | real/fake 错误平衡、OOD |
| A7-AUG-CLEAN | 训练只看 clean 图 | 竞赛增强的总贡献 | 所有 15 项变换的性能下降 |
| A8-AUG-SINGLE | 保留 clean+单项增强，禁用 compound augmentation | 复合增强是否带来额外鲁棒性 | 强噪声、crop、组合域偏移 |
| A9-DATA-BASE | train 仅保留原始 759921 的 39,985 张，固定 1,386 steps | 新增 4,392 张多生成器数据的价值 | DALL·E、MidJourney、OOD suite |

### A4 的严格定义

A4 仍对每项辅助 loss 除以 detached EMA，随后固定等权：

`L = L_primary + (1/5) × Σ L_i / EMA(L_i)`

这样只移除 MLP 的动态调度能力，不重新引入五个人工系数，也不把 loss 量纲差异误认为任务
重要性。不能用 759921 的固定 raw-loss 系数作为 A4，因为那会同时改变归一化和权重策略。

### A5 与 A6 的区别

- A5 保留 Class/Source GroupDRO、难例和 class-risk gap，只删除辅助任务。
- A6 保留 MLP 辅助任务，只把主分类目标换为普通 BCE。

两者分别测量 auxiliary regularization 和 robust primary objective，不能合并。

## 6. 第二阶段：细粒度消融

只有第一阶段结果表明对应模块重要时，才执行本阶段。

### 6.1 五项辅助 loss leave-one-out

| ID | 删除项 |
|---|---|
| L1 | consistency |
| L2 | feature consistency |
| L3 | degradation classification |
| L4 | degradation severity |
| L5 | tile auxiliary |

删除后 MLP 在剩余四项上重新形成带相同 5% 下限的 simplex；总辅助预算仍为 1.0。报告
epoch 均值、分位数和权重切换频率，不能使用“最后一批权重”代表全局重要性。

### 6.2 Robust primary objective 拆解

| ID | 唯一变化 |
|---|---|
| R1 | 去掉 source GroupDRO，保留 class-hard 与 risk gap |
| R2 | `hard_weight: 0` |
| R3 | `class_balance_weight: 0` |

R1 需要显式 sample-mean BCE 路径，不能简单设置 `eta: 0`，因为均匀 group mean 与 sample
mean 并不等价。

### 6.3 训练增强 leave-one-family-out

分别禁止 JPEG、blur、resize、noise、color、crop 六个 family，其余增强概率重新归一化。
评测时仍运行完整 16 variants。核心结果是“删除某训练增强后，在同名测试变换上的 AUROC
下降”以及对其他变换的迁移效应。

### 6.4 数据来源消融

按数据块分别移除：

- 多领域真实图 10,000 张；
- BigGAN/ADM 10,000 张；
- OpenFake 4,000 张；
- TIGAS + 本地 train-only 392 张。

每个实验继续固定 1,386 optimizer steps与 label-balanced sampler。数据消融报告必须同时给出
real FPR 与 fake recall，避免“删除真实图后 fake recall 上升”被错误叙述为整体提升。

## 7. 第三阶段：重复种子与结构效率

第一、二阶段先用固定 seed 做 screening。根据内部 validation 决定要进入重复实验的配置，
但不得查看 DALL·E 3 后再挑选。

建议对以下对象运行 3 seeds：

- 完整 B0 的一次独立复现；
- validation AUROC 下降最大的两项消融；
- 一个性能近似但可能节省计算的分支消融。

报告 mean ± standard deviation，并在相同测试图片上对 AUROC 差值做 paired bootstrap
95% CI。若 functional mask 表明某分支贡献很小，再做 structural removal，报告真实参数量、
峰值显存、单图 latency 和 images/s。

## 8. 评测矩阵

测试集必须逐个报告，不得混合成一个总 manifest：

1. 内部 validation：只用于 checkpoint selection 和训练稳定性；
2. COCO real / DALL·E 3：目标外部诊断集；
3. COCO real / MidJourney：跨生成器诊断；
4. 已整理的 external generalization suite：每个 dataset 单独一行；
5. 手机/相机实拍图：作为 qualitative stress test；样本量扩充前不宣称统计结论。

每个二类测试集都输出：

- clean AUROC、AP；
- 16 variants 的逐项 AUROC/AP；
- 16 项 mean AUROC、worst AUROC；
- `clean AUROC - mean transformed AUROC`；
- calibration-only 阈值下的 balanced accuracy、real recall、fake recall、FPR；
- paired bootstrap 95% CI；
- 参数量、可训练参数、显存、训练时间和推理延迟。

官方主排序使用 `mean AUROC` 与 `worst-transform AUROC`。线性化 0.5 confidence 只用于
演示，不可作为性能提升项，因为它不改变 AUROC 或分类。

## 9. 结果判读规则

建议在运行前冻结以下判读方式：

- 某消融使 target mean AUROC 明显下降且 paired CI 不跨 0：该组件有正贡献；
- target 提升但 MidJourney/OOD 明显下降：记录为域专用 trade-off，不称为普遍提升；
- AUROC 近似但 FPR 大幅变化：优先检查 calibration 与 score distribution，不直接归因于
  表征能力；
- 分支消融 AUROC 近似且 structural removal 显著降低延迟：可考虑精简；
- tile/patch 图“看起来合理”只能作为定性证据，不能替代分类指标；可选 deletion/insertion
  faithfulness 指标作为辅助。

不以单个 seed 的 0.1 个百分点差异宣称胜负，不以外部 DALL·E 3 结果决定保留哪个
checkpoint，也不把 linearized confidence 当成新的概率校准。

## 10. 执行前应准备的产物

未来真正执行前，再创建以下内容：

- `configs/ablations/773086/*.yaml`：每项一个 config；
- `scripts/build_773086_ablation_configs.py`：从 baseline 生成并验证单变量 diff；
- `scripts/verify_773086_ablation.py`：manifest、DALL·E 3 gate、参数状态、step budget 检查；
- `slurm/773086_ablation_smoke_h100.sbatch`：每项 3 steps；
- `slurm/773086_ablation_train_h100.sbatch`：正式训练与 post-selection eval；
- 同时提供直接运行脚本，不把 Slurm 作为唯一入口；
- `outputs/ablations_773086/<experiment_id>/`：隔离输出目录；
- `results/ABLATION_773086.csv/json/md`：统一统计、置信区间和结论；
- 图表：component delta bar、transform heatmap、AUROC/latency Pareto、real/fake error trade-off。

每个正式实验前必须 smoke test，并验证：loss 有限、MLP 权重和为 1、目标分支确实被中性化、
训练/validation 增强审计存在、显存不超过资源限制。当前阶段不执行这些步骤。

## 11. 预计工作量

773086 单次完整端到端耗时约 79 分钟。按同等资源粗略估计：

- 第一阶段 9 次：约 12 GPU-hours；
- 第二阶段全部执行：约 18–24 GPU-hours；
- 3-seed 复核与结构效率：约 8–12 GPU-hours。

建议先完成第一阶段并冻结主表，再根据内部 validation 结果决定第二阶段范围。不要为了补齐
矩阵而运行信息增益很低的组合实验。
