# GPU Validation Results / GPU 验证结果

## English

These measurements were collected on October 13, 2026, using an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). The environment was PyTorch 2.8.0, CUDA 12.8, and Transformers 5.19.0. Raw JSON reports are in [`gpu-results/`](gpu-results/); reproduction commands and criteria are in the [GPU validation guide](../gpu/README.md).

### Published-checkpoint forward comparison

The reference loaded the publisher's configuration; by1 compiled the configuration from each `.by1` file. Both sides used the same checkpoint, Transformers model implementation, and operators. Settings: BF16, batch size 1, eager attention, input lengths 32 and 128, 2 warmups, 5 timed runs, and greedy generation of 16 new tokens. All six models had a maximum absolute logit difference of 0, and all generated token sequences matched.

| Model | by1 / reference forward latency at length 128 | by1 peak forward memory | Additional check |
|---|---:|---:|---|
| GPT-2 | 0.990× | 0.27 GB | 16 generated tokens matched |
| Qwen3-0.6B | 0.996× | 1.24 GB | 16 tokens matched; one real-weight training step also tested |
| Qwen2.5-7B-Instruct | 0.998× | 15.28 GB | 16 generated tokens matched |
| Qwen3-8B | 0.997× | 16.43 GB | 16 tokens matched; longer contexts also tested |
| gpt-oss-20b | 1.054× | 41.90 GB | Published MXFP4 checkpoint dequantized to BF16 on both sides; 16 tokens matched |
| Qwen3.8-27B | 0.991× | 55.64 GB | 16 tokens matched; image input and MTP checked separately |

These latency ratios are observations from one machine and short sequences, not speedup claims. The benchmark reuses Transformers operators. It checks configuration conversion, weight loading, numerical behavior, and runtime overhead; it does not compare a by1 CUDA kernel or evaluate task quality.

Qwen3.8 text, MTP, and image paths used real checkpoint weights. The image case used a synthetic 128×128 image processed by the model's processor; the maximum absolute logit difference was 0. MTP output matched an independent connection formula with maximum error 0. The MTP timing is not an end-to-end vLLM speculative-decoding benchmark. The synthetic image validates execution and wiring, not visual capability.

### Longer contexts and a training step

| Model | Input length | Maximum absolute logit difference | by1 peak memory | Generation comparison |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 2,048 | 0 | 16.52 GB | 8 tokens matched |
| Qwen2.5-7B-Instruct | 4,096 | 0 | 20.16 GB | 8 tokens matched |
| Qwen3-8B | 2,048 | 0 | 17.85 GB | 8 tokens matched |
| Qwen3-8B | 4,096 | 0 | 22.01 GB | 8 tokens matched |
| Qwen3-8B | 8,192 | 0 | 38.44 GB | 4 tokens matched |

One BF16 training step used real Qwen3-0.6B weights: full-parameter backpropagation followed by one AdamW update. The official and by1 models matched elementwise across 596,049,920 gradients and the same number of updated parameter values (both maximum absolute differences were 0); peak memory was about 5.98 GB. This validates one step, not multi-step training or convergence.

Small CUDA smoke tests with random weights also passed in FP32 and BF16. They cover the native model, six scaled-down architectures, vision/video branches, MTP, and optimizer updates. A fresh run on 2026-10-13 at 20:26–20:27 (UTC+08) passed all seven cases in both dtypes against the current working-tree source; its source hashes match the checked-in files. These tests add component coverage; they do not replace the real-checkpoint comparisons above. The published-checkpoint measurements above were collected from earlier source hashes. A current-source real-checkpoint rerun is still pending: the Qwen3-0.6B download stalled at 50 MiB without progress, so no new pretrained-weight comparison was reported.

### Raw reports

- [Six-model forward and generation comparison](gpu-results/gpu-benchmark.json)
- [Final gpt-oss-20b report](gpu-results/gpu-benchmark-gptoss-final.json)
- [Final Qwen3.8-27B report, including image and MTP](gpu-results/gpu-benchmark-qwen38-final.json)
- [Qwen2.5-7B and Qwen3-8B longer contexts](gpu-results/gpu-long-context-large.json)
- [Qwen3-8B at 8K context](gpu-results/gpu-long-context-8k.json)
- [Qwen3-0.6B real-weight training step](gpu-results/gpu-real-traincheck.json)
- [FP32 smoke report](gpu-results/gpu-smoke-fp32.json) · [BF16 smoke report](gpu-results/gpu-smoke-bf16.json)
- [Current-source FP32 smoke rerun](gpu-results/gpu-smoke-fp32-20261013-2026.json) · [Current-source BF16 smoke rerun](gpu-results/gpu-smoke-bf16-20261013-2027.json)

## 中文

这些数据于 2026 年 10 月 13 日在 NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB）上采集。环境为 PyTorch 2.8.0、CUDA 12.8 和 Transformers 5.19.0。原始 JSON 报告见 [`gpu-results/`](gpu-results/)，复现命令与判据见 [GPU 验证说明](../gpu/README.md)。

### 发布方真实权重前向对比

对照组加载发布方配置，by1 组从 `.by1` 编译配置；两边使用相同 checkpoint、Transformers 模型实现和算子。设置为 BF16、batch 1、eager attention，输入长度 32 和 128，预热 2 次、计时 5 次，贪心生成 16 个新 token。六个模型的 logits 最大绝对误差均为 0，生成的 token 序列也都一致。

| 模型 | 长度 128 时 by1 / 对照组前向耗时比 | by1 前向峰值显存 | 额外检查 |
|---|---:|---:|---|
| GPT-2 | 0.990× | 0.27 GB | 16 个生成 token 一致 |
| Qwen3-0.6B | 0.996× | 1.24 GB | 16 个 token 一致；另测一次真实权重训练步 |
| Qwen2.5-7B-Instruct | 0.998× | 15.28 GB | 16 个生成 token 一致 |
| Qwen3-8B | 0.997× | 16.43 GB | 16 个 token 一致；另测更长上下文 |
| gpt-oss-20b | 1.054× | 41.90 GB | 两边都将发布的 MXFP4 checkpoint 解量化为 BF16；16 个 token 一致 |
| Qwen3.8-27B | 0.991× | 55.64 GB | 16 个 token 一致；另验图像输入和 MTP |

这些耗时比只是单台机器、短序列下的观测值，不代表加速。benchmark 复用 Transformers 算子，核验配置转换、权重加载、数值行为和运行开销；它没有比较 by1 CUDA 内核，也没有评测模型任务能力。

Qwen3.8 的文本、MTP 和图像分支使用真实 checkpoint 权重。图像测试使用合成的 128×128 图像并交给模型 processor 处理，logits 最大绝对误差为 0。MTP 输出与独立连接公式对拍，最大误差为 0。MTP 计时不是 vLLM speculative decoding 的端到端 benchmark。合成图像只验证执行和连接，不代表视觉能力评测。

### 更长上下文与训练步

| 模型 | 输入长度 | logits 最大绝对误差 | by1 峰值显存 | 生成对比 |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 2,048 | 0 | 16.52 GB | 8 个 token 一致 |
| Qwen2.5-7B-Instruct | 4,096 | 0 | 20.16 GB | 8 个 token 一致 |
| Qwen3-8B | 2,048 | 0 | 17.85 GB | 8 个 token 一致 |
| Qwen3-8B | 4,096 | 0 | 22.01 GB | 8 个 token 一致 |
| Qwen3-8B | 8,192 | 0 | 38.44 GB | 4 个 token 一致 |

Qwen3-0.6B 真实 BF16 权重还跑过一次全参数反向传播和 AdamW 更新。官方模型和 by1 模型的 596,049,920 个梯度及同数量的更新后参数逐元素一致（两项最大绝对误差均为 0）；峰值显存约 5.98 GB。这只验证一步，不代表多步训练或收敛性。

随机权重的小尺寸 CUDA smoke test 也在 FP32 和 BF16 下通过，覆盖原生模型、六种缩小架构、视觉/视频分支、MTP 和优化器更新。2026-10-13 20:26–20:27（UTC+08）用当前工作区源码重新运行时，两种精度下的 7 个用例全部通过，报告中的源码哈希与仓库文件一致。这些测试补充组件覆盖，不代替上面的真实权重对比。上方已发布 checkpoint 的性能数据来自较早的源码版本；当前源码的真实权重复跑仍待完成：Qwen3-0.6B 下载到 50 MiB 后停滞且没有进展，因此本次没有生成新的预训练权重对比结果。

### 原始报告

- [六模型前向与生成对比](gpu-results/gpu-benchmark.json)
- [gpt-oss-20b 最终报告](gpu-results/gpu-benchmark-gptoss-final.json)
- [Qwen3.8-27B 最终报告（含图像与 MTP）](gpu-results/gpu-benchmark-qwen38-final.json)
- [Qwen2.5-7B 与 Qwen3-8B 长上下文](gpu-results/gpu-long-context-large.json)
- [Qwen3-8B 的 8K 上下文](gpu-results/gpu-long-context-8k.json)

## T4 首次实测（2026-10-18，AutoDL）

环境：**Tesla T4 16 GB**（compute 7.5）· torch **2.6.0+cu124** · transformers **5.19.0** · Python 3.12 · Ubuntu 22.04。

| 判据 | 结果 | 报告 |
|---|---|---|
| GPU 冒烟（10 个 case，**随机权重**，bf16） | **10/10 PASS**，`gpu_verified: true` | [gpu-smoke-t4.json](gpu-results/gpu-smoke-t4.json) |
| 训练步（影子 Qwen3-0.6B，bf16 反向 + AdamW） | **PASS**，完整梯度与更新参数逐位一致 | `gputrain-t4.json`（未入库，见下） |
| 影子 bench（0.6B，seq 128，atol/rtol 0.002） | **PASS** | [gpu-benchmark-t4.json](gpu-results/gpu-benchmark-t4.json) |

冒烟的 10 个 case：native-hello · Qwen3-0.6B · Qwen3-8B · Qwen2.5-7B · gpt2 · gpt-oss-20b · Qwen3.8-27B ·
granite-4.0-h-350m · granite-4.0-h-1b · Falcon-H1-0.5B。**它们是随机权重**（报告的 `scope` 写明），
验的是"接线"：by1 生成的原生模型 与 同一架构的官方模型，在 CUDA bf16 下前向/反向/更新逐位一致。

**当天修掉/记档的三个环境坑**：

1. transformers 5.19+ 要求 **torch ≥ 2.6**；镜像自带 2.5.1，于是所有判卷人报 "PyTorch was not found"。
   升到 2.6.0+cu124 后恢复；
2. **torchvision 与 torch 2.6 不兼容**（`torchvision::nms` 不存在）→ 卸掉 torchvision 后 transformers 正常。
   代价：**视觉相关的 import 这台机器上没验**（只影响需要 torchvision 的路径）；
3. `git archive` 上传的树缺 `.by1-ignored`，`by1docs` 把 `.gitignore` 挡住的历史脚本判成"不存在"（9 处假红）；
   改传 `by1pack` 产物后 PASS。**包里必须带 `.by1-ignored`**。

**明确没验的**（不写成通过）：


### 真权重（2026-10-19，同一台 T4）

权重：`Qwen/Qwen3-0.6B` 的**真 checkpoint**（1.50 GB，`model.safetensors`），经 `hf-mirror.com` 下载到本地目录，
用 `--checkpoint` 指向它 —— 不走 Hub 的在线查询（`by1gpubench` 为此补了本地目录分支）。

| 判据 | 结果 | 报告 |
|---|---|---|
| 前向（bf16） | **max\|Δ\| = 0.0**，`allclose: true` | [gputrain-real-t4.json](gpu-results/gputrain-real-t4.json) |
| 梯度（596,049,920 个值） | **max\|Δ\| = 0.0**，relative_l2 = 0.0 | 同上 |
| AdamW 一步后的参数 | **逐位一致**（watched 参数 max update 1.22e-04） | 同上 |
| bench：text-128 / text-512 | **PASS** | [gpu-benchmark-real-t4.json](gpu-results/gpu-benchmark-real-t4.json) |
| 生成 | **`identical_tokens: true`**（同一 prompt、贪心） | 同上 |

吞吐（同一台 T4、bf16、贪心、单流）：

    original  612.4 ms（26.13 tok/s）   peak allocated 1.22 GB
    by1       628.8 ms                 load 0.66 s（original 1.27 s）

**口径**：这是"同一份真权重、两条独立实现"的逐位对拍（前向/反向/更新），加上生成 token 序列相同；
吞吐是**单机单次观测**，不当作跨机器/跨版本的性能结论（项目规矩：不把一次性观测写成普遍加速）。


### 上游 llama.cpp：CUDA 构建 + 数值（2026-10-19，同一台 T4）

**构建**：`/root/llama.cpp`，tag `v0.6.0`（sha `d812350`），
`cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=75`，
`llama-cli` / `llama-bench` 都产出。**没有改一行源码。**

| 判据 | 结果 |
|---|---|
| `by1llama.py pin-check --build-dir /root/llama.cpp` | **PASS**（两个二进制的版本串都带 `d812350`） |
| `llama-bench`（Qwen3-0.6B-Q8_0，CUDA，pp32/tg16） | **两次观测**：pp32 **1898 / 2424 t/s**，tg16 **174 / 190 t/s** |
| `llama-cli` 生成（贪心，24 token） | prompt **621 t/s** · generation **120 t/s**，输出正常（Qwen3 thinking 模式） |

报告：[llama-bench-t4.txt](gpu-results/llama-bench-t4.txt)（第二次观测原文）。

**为什么给区间不给单值**：同一台机器、同一构建、同一 GGUF，两次跑出来的 pp32 是 1898 和 2424（+28%）。
GPU 预热/频率/容器负载都会影响它 —— 项目规矩是**不把一次性观测写成性能结论**。

**当天修的一处判据**：`by1llama.py pin-check` 原来只读 `--version` 的**第一行**；CUDA 构建的
`llama-bench --version` 第一行是 `ggml_cuda_init: found 1 CUDA devices ...`（初始化日志），
版本行在后面 —— 于是 CUDA 构建上假红。改成在**所有输出行**里找钉住的 commit 前缀，修后 PASS。

**仍未验**：其它量化档（Q4/Q2）· 视觉/MTP 路径 · 多卡。

~~上游 GGUF 与 by1 产物的 logits 逐位对拍~~ ✅ 2026-10-19 已做：5 对 527,802 值逐位相同（
口径：同一运行时跑两个 GGUF，dump 是 eval-callback 调试视图 4 位小数，见 docs/convert-diff.md）

**仍未验**：27B/8B 的真权重前向（显存不够）、多卡。

- **真权重 27B / 8B 前向**：T4 只有 16 GB，27B bf16 约 55 GB —— 放不下；
- ~~真权重 bench/train~~ ✅ 已验（镜像下载 + `--checkpoint` 本地目录）；
- **llama.cpp CUDA 构建**：镜像里没有 `nvcc`；
- 多卡 / 张量并行：没有第二块卡。
- [Qwen3-0.6B 真实权重训练步](gpu-results/gpu-real-traincheck.json)
- [FP32 smoke 报告](gpu-results/gpu-smoke-fp32.json) · [BF16 smoke 报告](gpu-results/gpu-smoke-bf16.json)
- [当前源码 FP32 smoke 复跑](gpu-results/gpu-smoke-fp32-20261013-2026.json) · [当前源码 BF16 smoke 复跑](gpu-results/gpu-smoke-bf16-20261013-2027.json)
