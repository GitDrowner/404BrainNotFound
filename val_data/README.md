# External validation data preparation

本目录只包含最新一批外部泛化测试集的下载、确定性抽样、预处理、去重和完整性验证代码，
**不包含任何图片、生成后的 manifest、模型权重或推理结果**。

对应的已审计本地快照名称为 `generalization_suite_20260830`。它由 5 个互相隔离的
benchmark 构成：

| Benchmark | 内容 | 样本数 | 主要用途 |
|---|---|---:|---|
| `ditfake_flux1_schnell` | COCO real 100 + FLUX.1-schnell 100 | 200 | FLUX 跨生成器迁移 |
| `ditfake_pixart_sigma` | COCO real 100 + PixArt-Sigma 100 | 200 | PixArt DiT 迁移 |
| `ditfake_sd3_medium` | COCO real 100 + SD3 Medium 100 | 200 | Stable Diffusion 3 迁移 |
| `frontier_small_2026` | real 82 + 多生成器 fake 58 | 140 | 小型多域补充压力测试 |
| `qwen_image_bench_frontier_fake_only` | 18 个近期生成器，各 10 张 | 180 | 最新生成器 fake recall |

库存合计为 920 张，但这些数据集在评测时不得合并计算一个总指标。前三项和
`frontier_small_2026` 支持各自的 AUROC；Qwen-Image-Bench 子集没有 real 图片，只能报告
fake recall 和置信度分布，不能与其他数据集拼接后制造 AUROC。

## 数据来源

- `lioooox/DiTFake`：固定 revision，抽取 FLUX.1-schnell、PixArt-Sigma 和 SD3 Medium；
- `ash12321/ai-detector-benchmark-test-data`：固定 140 条的小型社区 benchmark；
- `Qwen/Qwen-Image-Bench`：固定 revision，抽取 18 个生成器，每个 10 张。

精确 revision、生成器列表、历史样本统计和 manifest 哈希见
[`metadata/DATASET_CATALOG.json`](metadata/DATASET_CATALOG.json)。来源局限见
[`PROVENANCE.md`](PROVENANCE.md)。

## 安装

```bash
cd val_data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 推荐的一键准备方式

为了复现经过污染检查的版本，需要提供训练 manifest 以及已有外部 holdout manifest。
这些文件只用于读取 SHA-256 和训练生成器名称，不会被复制到本目录：

```bash
HF_HUB_DISABLE_XET=1 python scripts/prepare_and_verify.py \
  --training-manifest /path/to/job_773086_train.jsonl \
  --blocked-manifest /path/to/dalle3_advanced_manifest.jsonl \
  --blocked-manifest /path/to/commercial_frontier_manifest.jsonl \
  --workers 4
```

默认输出到：

```text
data/external_eval_only/generalization_suite_20260830/
```

该路径已被 `.gitignore` 排除。根目录和每个 benchmark 都会生成
`.EVAL_ONLY_DO_NOT_TRAIN`，每个 benchmark 分别拥有自己的 `manifest.jsonl` 和
`receipt.json`，根目录只生成索引，不合并样本。

如果操作者没有任何污染检查 manifest，可以显式加入 `--allow-empty-blocklist`。这样能够
下载数据，但结果不等价于已审计快照，报告时必须标为 `non-audited copy`。

## 分步执行

```bash
python scripts/prepare_external_generalization_suite.py \
  --config configs/external_generalization_suite.json \
  --training-manifest /path/to/train.jsonl \
  --blocked-manifest /path/to/existing_holdout.jsonl \
  --workers 4

python scripts/verify_external_generalization_suite.py \
  --index data/external_eval_only/generalization_suite_20260830/benchmark_index.json \
  --blocked-manifest /path/to/train.jsonl \
  --blocked-manifest /path/to/existing_holdout.jsonl
```

## 关键边界

- 所有数据都是 post-selection evaluation only；禁止进入训练、难例挖掘、校准或选模。
- DALL-E Advanced 和 13 张商用盲测图不是这一批下载内容；它们只在原始准备流程中作为
  exact-SHA blocklist。
- `seen_in_training` 只表示生成器名称与训练 manifest 的规范化名称匹配，不表示图片重复。
- DiTFake 的三个子集分别评测，不能因为都来自 DiTFake 就合并。
- 下载脚本会固定 revision、确定性抽样、验证图片格式和尺寸、计算 SHA-256、拒绝黑名单
  重叠和单个 benchmark 内的重复内容。

完整文件哈希见 `PACKAGE_MANIFEST.json`。
