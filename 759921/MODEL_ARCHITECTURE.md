# 759921 模型架构说明

整体图见 [`MODEL_ARCHITECTURE.svg`](MODEL_ARCHITECTURE.svg)。图的上半部分严格对应
checkpoint 的实际 forward；下方虚线区域是推理期解释层，不属于模型结构。

759921 是 legacy joint-fusion 模型，不是三专家 MoE。它将 DINOv2 全局 embedding、四个
native tile 的 attention-pooled embedding、SigLIP semantic embedding 和一个
`wavelet_similarity` 标量拼接后，经 512 维 fusion MLP 和二分类头输出 raw logit。

`wavelet_similarity` 的计算过程是：对 224×224 global view 注入 Haar-like 高频扰动，
再次通过共享 DINOv2，并计算原图与扰动图全局 embedding 的 cosine similarity。当前新增
的 wavelet-only 解释会固定 fusion 输入中的其他维度，只改变这一真实模型标量；它不会
修改网络层、参数或 checkpoint。
