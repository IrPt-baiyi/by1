# 后端 × 机制矩阵（生成物，别手改）

> 这个文件是 [`src/checks/by1backends.py`](../src/checks/by1backends.py) 的产物：
> 跑 `python src/checks/by1backends.py --write` 重新生成。
> 实测与它不一致时，`by1all` 判卷人段会红 —— **所以它不会漂**。

## 一、矩阵

| 探针 | 机制 | `codegen` | `c` | `ggml` | `exec` | `full` |
|---|---|---|---|---|---|---|
| `hello.by1` | Attention + FFN（最小完整） | ✅ | ✅ | ✅ | ✅ | ✗ |
| `attn-fused-shaped.by1` | Attention（融合 qkv + 门） | ✅ | ✅ | ✅ | ✅ | ✗ |
| `mla-shaped.by1` | MLA | ✅ | ✅ | ✗ | ✅ | ✗ |
| `mla-sparse-shaped.by1` | MLA + 稀疏索引（DSA） | ✅ | ✅ | ✗ | ✅ | ✗ |
| `moe-routing-shaped.by1` | MoE（路由） | ✅ | ✅ | ✗ | ✅ | ✗ |
| `kda-shaped.by1` | KDA | ✅ | ✅ | ✗ | ✅ | ✗ |
| `kda-gva-shaped.by1` | KDA（GVA） | ✅ | ✅ | ✗ | ✅ | ✗ |
| `kda-lowrank-shaped.by1` | KDA（低秩） | ✅ | ✅ | ✗ | ✅ | ✗ |
| `ssm-shaped.by1` | SSM（Mamba2/Nemotron-H 参数化） | ✅ | ✅ | ✅ | ✅ | ✗ |
| `raw-escape.by1` | Raw（逃生舱） | ✅ | ✗ | ✗ | ✗ | ✗ |
| `complete-granite-4.0-h-350m.by1` | SSM2 + Attention（逐层混排） | ✗ | — | — | — | ✅ |
| `complete-Falcon-H1-0.5B-Instruct.by1` | Parallel（层内并行：SSM2 + Attention） | ✗ | — | — | — | ✅ |

**读法**：✅ = 后端接受并给出产物；✗ = 明确拒绝（原文见第二节）；— = 根本没走到（上游 `codegen` 先拒，本后端无从表态）；? = 没验（缺依赖，**不算通过**）。

## 二、每一个拒绝的**原文**

> 这张表是"要么实现，要么拒绝"的落点：越界不是沉默，是一句能贴进 issue 的话。

| 探针 | 后端 | 拒绝原文 |
|---|---|---|
| `hello.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `attn-fused-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `mla-shaped.by1` | `ggml` | 第 0 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 1 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 2 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 3 层 op1（MLA）：算子 MLA 还没有 ggml 映射 |
| `mla-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `mla-sparse-shaped.by1` | `ggml` | 第 0 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 1 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 2 层 op1（MLA）：算子 MLA 还没有 ggml 映射；第 3 层 op1（MLA）：算子 MLA 还没有 ggml 映射 |
| `mla-sparse-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `moe-routing-shaped.by1` | `ggml` | 第 0 层 op4（MoE）：算子 MoE 还没有 ggml 映射；第 1 层 op4（MoE）：算子 MoE 还没有 ggml 映射；第 2 层 op4（MoE）：算子 MoE 还没有 ggml 映射 |
| `moe-routing-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `kda-shaped.by1` | `ggml` | 第 0 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 1 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 2 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 3 层 op1（KDA）：算子 KDA 还没有 ggml 映射 |
| `kda-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `kda-gva-shaped.by1` | `ggml` | 第 0 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 1 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 2 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 3 层 op1（KDA）：算子 KDA 还没有 ggml 映射 |
| `kda-gva-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `kda-lowrank-shaped.by1` | `ggml` | 第 0 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 1 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 2 层 op1（KDA）：算子 KDA 还没有 ggml 映射；第 3 层 op1（KDA）：算子 KDA 还没有 ggml 映射 |
| `kda-lowrank-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `ssm-shaped.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `raw-escape.by1` | `c` | [不支持] C 后端还没有 Raw |
| `raw-escape.by1` | `ggml` | 第 0 层 op1（Raw）：算子 Raw 还没有 ggml 映射；第 1 层 op1（Raw）：算子 Raw 还没有 ggml 映射 |
| `raw-escape.by1` | `exec` | 这个后端还没有实现 Raw |
| `raw-escape.by1` | `full` | 完整后端的 arch 必须是 qwen3, qwen2, gpt2, gpt_oss, qwen3_5, mamba, granitemoehybrid, falcon_h1 |
| `complete-granite-4.0-h-350m.by1` | `codegen` | 完整模型使用 transformers 后端；请用 by1 build，不能把多分支模型降成串行 IR |
| `complete-Falcon-H1-0.5B-Instruct.by1` | `codegen` | 完整模型使用 transformers 后端；请用 by1 build，不能把多分支模型降成串行 IR |

## 三、**公共子集**（两扇门分开算）

### 3.1 IR 路（`codegen` / `c` / `ggml` / `exec` 四后端交集）

这是**机制层面的承诺**：这些声明在四个后端上都出得了产物。

- `hello.by1` —— Attention + FFN（最小完整）
- `attn-fused-shaped.by1` —— Attention（融合 qkv + 门）
- `ssm-shaped.by1` —— SSM（Mamba2/Nemotron-H 参数化）

### 3.2 五后端交集（含 `full`）

**一个都没有。** 这不奇怪：`full` 是**另一扇门**（吃 `arch = qwen3 / granitemoehybrid / …` 的官方家族），而上面那些 shaped 探针走的是串行 IR 门。**但对外措辞要照实**：不存在"一份声明五后端通吃"，"多后端已支持"这句话现在不成立。

生成时间：2026-10-16（仅作记录，判据不看它）
