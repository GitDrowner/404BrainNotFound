# 759921 explainability demos

打开 `index.html` 查看四张带标签图片的并排展示。每个子目录都包含完整的 11 项解释产物，
包括原图、两类 patch 图、高频纹理图、颜色图例、失真轨迹、分支贡献、HTML 和原始 JSON。

共同设置：checkpoint `checkpoint/best.pt`，`grid=4`，`occlusion=blur`，完整 16 条赛题变换，
未使用概率校准。

| Demo | Ground truth | Source/generator | P(AIGC) | Nominal result @ 0.5 |
| --- | ---: | --- | ---: | --- |
| `dalle_pair_real_00000` | real | COCO / Defactify pair 0 | 0.982151 | false positive |
| `dalle_pair_fake_00000` | AIGC | DALL·E 3 / Defactify pair 0 | 0.999995 | correct |
| `wildfake_real_f5a51` | real | WildFake `real_cat` | 0.000045 | correct |
| `cifake_sd14_fake_26167` | AIGC | CIFAKE / Stable Diffusion 1.4 | 1.000000 | correct |

这些样例用于验证展示链路，不构成数据集级性能评测。表内结果只按名义阈值 0.5 标注，
没有套用另存的分类阈值；当前 sigmoid confidence 也未经概率校准。热力图是遮挡反事实，
纹理图是输入的高频残差，二者都不是像素级伪造真值。
