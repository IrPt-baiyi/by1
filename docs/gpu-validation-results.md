# GPU 验证结果

这些记录来自 2026-10-13 的 NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB）实例，环境为 PyTorch 2.8.0、CUDA 12.8、Transformers 5.19.0。JSON 原始报告与摘要一同保存在本仓库的 [`gpu-results/`](gpu-results/) 目录；复现命令和判据见 [GPU 验证说明](../gpu/README.md)。

## 真实 checkpoint 前向对比

对照组加载发布方配置，by1 组从 `.by1` 编译模型配置；两边使用相同 checkpoint、Transformers 模型实现和算子。BF16、batch 1、eager attention，输入长度 32 和 128，预热 2 次、每项计时 5 次；贪心生成 16 个新 token。六个模型的前向 logits 最大绝对误差均为 0，greedy 输出 token 均完全相同。

| 模型 | by1 / 发布方前向耗时比，长度 128 | by1 前向峰值显存 | 额外检查 |
|---|---:|---:|---|
| GPT-2 | 0.990× | 0.27 GB | 16 token 生成一致 |
| Qwen3-0.6B | 0.996× | 1.24 GB | 16 token 生成一致；另有真实权重训练步 |
| Qwen2.5-7B-Instruct | 0.998× | 15.28 GB | 16 token 生成一致 |
| Qwen3-8B | 0.997× | 16.43 GB | 16 token 生成一致；另测长上下文 |
| gpt-oss-20b | 1.054× | 41.90 GB | 发布的 MXFP4 checkpoint 在两边都解量化为 BF16；16 token 生成一致 |
| Qwen3.8-27B | 0.991× | 55.64 GB | 16 token 生成一致；图像输入和 MTP 分支另行核验 |

耗时比是这次单机、短序列运行的观测值，不是加速承诺。此 benchmark 复用 Transformers 算子，测的是配置转换、权重加载、计算结果和额外运行开销；它不比较 by1 自研 CUDA 内核，也不衡量模型任务能力。

Qwen3.8 的文本、MTP 和图像输入均用真实 checkpoint 权重。图像项使用合成的 128×128 输入，经该模型 processor 处理；原始与 by1 logits 最大绝对误差为 0。MTP 输出与独立连接公式对拍，最大误差为 0。MTP 计时不是 vLLM speculative decoding 的端到端 benchmark。合成图像只验证分支计算与连接，不代表视觉能力评测。

## 长上下文和训练步

| 模型 | 输入长度 | logits 最大绝对误差 | by1 峰值显存 | 生成对比 |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 2,048 | 0 | 16.52 GB | 8 token 一致 |
| Qwen2.5-7B-Instruct | 4,096 | 0 | 20.16 GB | 8 token 一致 |
| Qwen3-8B | 2,048 | 0 | 17.85 GB | 8 token 一致 |
| Qwen3-8B | 4,096 | 0 | 22.01 GB | 8 token 一致 |
| Qwen3-8B | 8,192 | 0 | 38.44 GB | 4 token 一致 |

此外，Qwen3-0.6B 的真实权重 BF16 训练步跑过一次全参数反向传播和 AdamW 更新。官方与 by1 两侧的 596,049,920 个梯度值及同数量的更新后参数逐元素一致（最大绝对误差均为 0）；峰值显存约 5.98 GB。这验证一个训练步，不等同于多步训练或收敛性评估。

随机权重的小尺寸 CUDA smoke tests 也在 FP32 和 BF16 下通过，覆盖原生模型、六种缩小架构、视觉/视频分支、MTP 及优化器更新。它们用于补足组件覆盖，不代替上述真实 checkpoint 检查。

## 原始报告

- [六模型前向与生成对比](gpu-results/gpu-benchmark.json)
- [gpt-oss-20b 最终报告](gpu-results/gpu-benchmark-gptoss-final.json)
- [Qwen3.8-27B 最终报告（含图像与 MTP）](gpu-results/gpu-benchmark-qwen38-final.json)
- [Qwen2.5-7B 与 Qwen3-8B 长上下文](gpu-results/gpu-long-context-large.json)
- [Qwen3-8B 8K 上下文](gpu-results/gpu-long-context-8k.json)
- [Qwen3-0.6B 真实权重训练步](gpu-results/gpu-real-traincheck.json)
- [FP32 smoke report](gpu-results/gpu-smoke-fp32.json) · [BF16 smoke report](gpu-results/gpu-smoke-bf16.json)
