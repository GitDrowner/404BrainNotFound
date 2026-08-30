# External validation data preparation

本目录只包含外部泛化测试集的下载、确定性抽样、预处理、去重和完整性验证代码，
**不包含任何图片、生成后的 manifest、模型权重或推理结果**。

它覆盖最新五套 OOD benchmark，以及项目历史上使用的 DALL·E 3 和 MidJourney
benchmark，共 7 套。每套数据保留独立 manifest 和指标：

| Benchmark | 内容 | 样本数 | 主要用途 |
|---|---|---:|---|
| `ditfake_flux1_schnell` | COCO real 100 + FLUX.1-schnell 100 | 200 | FLUX 跨生成器迁移 |
| `ditfake_pixart_sigma` | COCO real 100 + PixArt-Sigma 100 | 200 | PixArt DiT 迁移 |
| `ditfake_sd3_medium` | COCO real 100 + SD3 Medium 100 | 200 | Stable Diffusion 3 迁移 |
| `frontier_small_2026` | real 82 + 多生成器 fake 58 | 140 | 小型多域补充压力测试 |
| `qwen_image_bench_frontier_fake_only` | 18 个近期生成器，各 10 张 | 180 | 最新生成器 fake recall |
| `coco_dalle3_advanced` | caption-paired COCO real 1,496 + DALL·E 3 1,496 | 2,992 | 历史 DALL·E 3 外部诊断 |
| `coco_midjourney_historical` | MS-COCO-unique real 1,000 + MidJourney 1,000 | 2,000 | 历史跨生成器测试 |

历史已审计库存合计为 5,912 张，但这些数据集在评测时不得合并计算一个总指标。Qwen
子集没有 real 图片，只能报告 fake recall 和置信度分布，不能与其他数据集拼接后制造
AUROC。

## 数据来源

- `lioooox/DiTFake`：固定 revision，抽取 FLUX.1-schnell、PixArt-Sigma 和 SD3 Medium；
- `ash12321/ai-detector-benchmark-test-data`：固定 140 条的小型社区 benchmark；
- `Qwen/Qwen-Image-Bench`：固定 revision，抽取 18 个生成器，每个 10 张；
- `Rajarshi-Roy-research/Defactify_Image_Dataset`：只下载固定 revision 的 validation
  parquet，按 caption 配对 COCO real 与 DALL·E 3，并按整对隔离近重复；
- `bitmind/MS-COCO-unique` 与 `bitmind/GenImage_MidJourney`：固定 revision 和历史 seed，
  各确定性抽样 1,000 张。

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

## 推荐：一次准备全部 7 套

```bash
HF_HUB_DISABLE_XET=1 python scripts/prepare_all_validation_data.py \
  --training-manifest /path/to/job_773086_train.jsonl \
  --blocked-manifest /path/to/commercial_frontier_manifest.jsonl \
  --manifest-image-root /path/to/project/root \
  --workers 4
```

默认输出到两个隔离根目录：

```text
data/external_eval_only/historical_benchmarks/
data/external_eval_only/generalization_suite_20260830/
```

这些路径已被 `.gitignore` 排除。根目录和每个 benchmark 都会生成
`.EVAL_ONLY_DO_NOT_TRAIN`；每个 benchmark 分别拥有自己的 `manifest.jsonl` 和
`receipt.json`，根目录只生成索引，不合并样本。

`--manifest-image-root` 用来解析训练 manifest 中的相对图片路径，供 DALL·E 3 的 dHash
近重复隔离使用。历史 benchmark 会先完成；随后它们的 manifest 自动加入最新五套数据的
SHA-256 blocklist，避免跨套完全相同的图片。

如果没有污染检查 manifest，可显式加入 `--allow-empty-blocklist`。结果只能标为
`non-audited copy`，不等价于已审计快照。

## 只准备历史 DALL·E 3 / MidJourney

```bash
python scripts/prepare_historical_benchmarks.py \
  --config configs/historical_benchmarks.json \
  --training-manifest /path/to/train.jsonl \
  --manifest-image-root /path/to/project/root

python scripts/verify_external_generalization_suite.py \
  --index data/external_eval_only/historical_benchmarks/benchmark_index.json \
  --blocked-manifest /path/to/train.jsonl
```

可用 `--only dalle3` 或 `--only coco-midjourney` 只准备其中一套。请勿把已经生成的
DALL·E manifest 作为准备它自身时的 `--blocked-manifest`。

## 只准备最新五套 OOD 数据

```bash
python scripts/prepare_and_verify.py \
  --training-manifest /path/to/train.jsonl \
  --blocked-manifest /path/to/existing_holdout.jsonl \
  --workers 4
```

## 关键边界

- 所有数据都是 post-selection evaluation only；禁止进入训练、难例挖掘、校准或选模。
- 13 张商用盲测图不随本包下载，只能作为 exact-SHA blocklist。
- `seen_in_training` 只表示生成器名称匹配，不表示图片重复。
- DiTFake 的三个子集分别评测，不能因为都来自 DiTFake 就合并。
- COCO/DALL·E 3 与 COCO/MidJourney 是两套独立 benchmark；COCO 来源不同，不能合并。
- 训练 manifest 若声明 DALL·E、Defactify、OpenFake、MidJourney 或历史 COCO test 来源，
  历史准备流程会直接失败；不会用“同家族但不同图片”规避来源泄漏。
- 下载脚本会固定 revision、确定性抽样、验证图片格式和尺寸、计算 SHA-256、拒绝黑名单
  重叠和单个 benchmark 内的重复内容。

完整文件哈希见 `PACKAGE_MANIFEST.json`。
