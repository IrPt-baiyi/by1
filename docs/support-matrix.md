# 支持矩阵（四态）/ Support matrix (four states)

> **四态**（1.0 阶段 E / 发布门的口径，逐模型逐功能分开说）：
>
> | 态 | 含义 | 证据长什么样 |
> |---|---|---|
> | **规格检查** | config 逐字段 + 张量名/形状对着官方产物验过 | `by1all` 的 config/tensor 对拍行 |
> | **加载** | 真 checkpoint 能被 by1 loader 完整装载（零 missing/unexpected） | GPU bench 的装载段报告 |
> | **运行** | 真权重前向/生成对拍过 | GPU bench / modelcheck 判卷人的数值行 |
> | **产物** | 生成的 GGUF/量化产物被未修改上游加载 | llama.cpp / convert 的 round-trip 报告 |
>
> **没有独立判据的标「实验」，不计入稳定覆盖数。** 影子（随机值）对拍
> 是**接线证据**，记在备注里，**不升"运行"态** —— 混了就是夸大。

## A. 完整模型（**真实权重对拍过**：6 个走 GPU 轮次 + mamba-130m 纯 CPU）

| 模型 | 规格检查 | 加载 | 运行 | 产物 | 证据 / 备注 |
|---|---|---|---|---|---|
| gpt2 | ✅ | ✅ | ✅ | ✅ | 运行：[六模型对比](gpu-results/gpu-benchmark.json) + [by1gpt2 前向](../history/ir.md)；产物：标准 GGUF 走 llama.cpp（[基线](llama.cpp-baseline.md)） |
| Qwen3-0.6B | ✅ | ✅ | ✅ | ✅ [闭环报告](p0-loop-report.md) | 运行：六模型对比 + [训练步](gpu-results/) + **真权重双 loader fp32 前向逐位（CPU，max\|Δ\|=0）** + **CPU/bf16 训练步逐位（梯度与 AdamW 更新各 596,049,920 个值 max\|Δ\|=0）**；**产物**：by1 从 HF 权重出 F32 GGUF，未修改的上游 llama.cpp 加载后 logits 与官方 convert 产物**逐位相同（106949 值）**、生成同文 |
| Qwen2.5-7B-Instruct | ✅ | ✅ | ✅ | — 未验 | 六模型对比；影子冒烟超本机内存（GPU 日补） |
| Qwen3-8B | ✅ | ✅ | ✅ | — 未验 | 六模型对比 |
| gpt-oss-20b（MXFP4） | ✅ | ✅ | ✅ | — 未验 | [原生 MXFP4](gpu-results/gpu-benchmark-gptoss-final.json)；量化存储审计过 |
| Qwen3.8-27B（BF16，含视觉+MTP） | ✅ | ✅ | ✅ | 🟡 实验 | [含图像与 MTP](gpu-results/gpu-benchmark-qwen38-final.json)；产物 = Qwen3.8 IQ3_S GGUF 基线配方**待 GPU 日跑** |
| Qwen3.8-27B-FP8 | ✅ | 🟡 装载路径已接通 | — 未验 | — 未验 | Q2 **GPU 实测前不标完成**（[量化边界](quantization.md)）；存储审计 407 对 |
| **mamba-130m-hf**（**第二架构**：SSM，无 KV cache） | ✅ | ✅ | ✅ | ✅ [T26 记录](t26-plan.md) | 真权重双 loader 前向**逐位（max\|Δ\|=0，纯 CPU）**；递归状态增量 == 全量 **1.497e-03**（判据 2e-3，余量 1.3× 已登记进[精度预算](precision-budget.md)）；**产物**：by1 出 GGUF 被未修改的上游 llama.cpp 加载，logits **逐位（30604 值）** |
| **granite-4.0-h-350m**（**第三架构**：混合栈 —— 28 × Mamba2 + 4 × Attention 同栈） | ✅ | ✅ | ✅ | ✅ [T27 记录](t27-plan.md) | 真权重双 loader 前向**逐位（max\|Δ\|=0）**；**混合状态**增量 == 全量 **9.918e-05**（一个 `DynamicCache` 里 28 个递归层 + 4 个注意力层）；config 逐字段 40 个 · 340.3 M 与真权重同；**产物**：402 张量与上游**逐位相同**、上游未修改的 llama.cpp 加载后 logits **逐位（89474 值）** |

## B. refs 语料（**36** 个 config / **31** 个有逐字段行）

**口径**（2026-10-16 重数）：以 `refs/*.config.json` 为准 = **36** 个；
其中 **31** 个有 `by1all` 档 3 的 config 逐字段行（即逐个证据）。
剩下 5 个**只有语料、config 逐字段行还没接**：DeepSeek-V4.1-Flash ·
LLaDA2.2-flash · Laguna-XS_2 · Step-5-Preview-BF16 · Qwen3-0.6B。
其中 **Qwen3-0.6B 是 `complete-*`（Transformers 后端）** —— 它**不该有**
config 逐字段行（那套映射属于原生 IR 路径）：等价判据是"生成 config → HF 装载
零 missing/unexpected"与"双 loader 装同一份真权重前向逐位"，两条都已过
（[P0 闭环报告](p0-loop-report.md) 第五、六节）。Step-5-Preview-BF16 同理但仍是
`drafts/` 里的反推草稿，**没有判卷人，不算数** —— 而且 2026-10-16 实测：
上游 transformers（5.15.1 / 5.20.0）与 llama.cpp 检出**都没有 step5 实现**，
所以连"独立实现当判卷人"这条路也是断的（见
[Step-5 工作单](step5-contract-plan.md) ③）。

**加载/运行/产物：未验**（要显存与真权重轮次），除两处例外：

| 例外 | 运行 | 证据 |
|---|---|---|
| Instella-3B | ✅ | `by1instella` 前向与官方实现一致（判卷人） |
| GPT-2 | ✅ | `by1gpt2` 前向与官方实现一致（判卷人） |

31 个有逐字段行的名字（`by1all` 的 config 行即逐个证据）：gpt-oss-120b ·
gpt-oss-20b · gemma-4-31B · Laguna-XS-2_1 ·
Instella-3B ·
Ling-3.0-tiny · minimind-3 · clef · Qwen3.8-27B · Qwen3.6-35B-A3B ·
Step-3.7-Flash · gpt2 · NVIDIA-Nemotron-3.5-Lightning · GLM-5.3-Flash ·
Qwen3-VL-2B/30B/235B-Thinking · Qwen3-Omni-30B-A3B-Thinking ·
DeepSeek-R1-Distill-Qwen-1.5B/7B · Qwen3.5-0.8B/9B/397B-A17B ·
gemma-4-12B/26B-A4B/E4B · mamba-130m-hf · Qwen3-Coder-Next ·
Qwen3.8-Flash-Next · GLM-4.7-Flash · embeddinggemma-2。

**已从语料移除**：Step-3_7-Flash-180B（社区剪枝版，上游仓库下架；
当年 25 张量缺口的记档保留在 INVENTORY）。

## C. 功能维度（分开说，不用"模型已支持"糊弄）

| 功能 | 状态 | 证据 / 边界 |
|---|---|---|
| 文本前向/生成 | 完整模型 **8** 个 ✅（含第二架构 mamba-130m 与第三架构 granite-4.0-h-350m，**均纯 CPU**）；refs 36 个有语料 / 31 个 config 逐字段 ✅，运行未验 | 见 A/B |
| **状态契约** | **三本账都验过**：KV cache ✅（Qwen3-0.6B 增量 == 全量 **2.289e-05**）· **递归状态 ✅**（mamba **1.497e-03**，判据 2e-3，**余量 1.3× 已登记**）· **混合状态 ✅**（granite：一个 cache 里 28 递归层 + 4 注意力层，**9.918e-05**，余量 20×） | `by1shadow --smoke-state` / `--smoke-state-ssm` / `--smoke-state-hybrid`；[精度预算](precision-budget.md) |
| 视觉塔 | **接线级** ✅（qwen3_5 影子含视觉塔，逐位 0）；真权重值 ✅（Qwen3.8 图像对比）；其余塔未验 | [多模态缺口](INVENTORY.txt)【F】 |
| MTP | ✅（Qwen3.8 含 MTP 对拍 + mtp.* 命名对 refs 15/15） | 六模型对比含图像与 MTP |
| 音频塔 | 未建模 | INVENTORY【F】：完全没建模 |
| 量化（Q1 规格） | ✅ MXFP4/FP8 存储合同 + oracle + manifest 对拍 | [量化边界](quantization.md) |
| 量化（Q2 运行） | 🟡 装载路径接通；GPU 执行待验 | **实测前不标完成** |
| 量化（Q3 产物） | 未开始 | Q3 前置评审五条已立（quantization.md） |
| 训练步 | ✅（Qwen3-0.6B 真权重梯度+AdamW 对拍）；影子模式已备 | [训练检查](gpu-results/) |

## D. 稳定覆盖数的算法

**稳定覆盖 = 四态里至少"规格检查 ✅"且无「实验」标记的模型数。**
当前：refs 32 + 完整 6（gpt2 重叠）= **37 个模型规格级稳定**；
运行级 = 6 + 2 = **8 个**；产物级 = **1 个**（gpt2 的标准 GGUF 链）。
FP8/Q3/音频塔不计。**数字只许随着证据升，不许跟着宣传涨。**
