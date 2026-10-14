# llama.cpp 集成策略

> 当前执行依据是[By1 长期路线图](1.0-roadmap.md)。旧版分阶段移植提案保存在 history/llamacpp-plan-2026-10.md，此文取代旧计划，不再以“把架构生成进 llama.cpp 本体”为发布目标。

## 目标

让 By1 的架构和权重合同在**未修改的 llama.cpp** 上接受真实运行验证；标准 GGUF 能直接使用原版 CLI。By1 生成的 ggml 代码继续承担独立数值判据，不等同于可直接合并进 llama.cpp 的完整架构适配器。

## 先复用上游

截至 2026-10-13，llama.cpp 上游已合入 Qwen3.8-Flash-Next / qwen4exp 支持，包含转换、文本图、稀疏注意力和视觉路径；v0.6.0 发布说明列有 Qwen4Exp MTP 支持。新接模型前先核对所需功能是否已进入目标上游版本：

- [Qwen3.8-Flash-Next 支持 PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [Qwen4Exp MTP PR #29761](https://github.com/ggml-org/llama.cpp/pull/29761)
- [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)

上游支持随版本变化。验证报告必须写明使用的 tag/commit、模型文件 revision、构建选项和 GPU；“支持该架构”不代表每种后端、量化类型、视觉路径和并发配置都已验证。

下一轮的执行 pin 已选为上游 `v0.6.0`（发布 commit 前缀 `d812350`），不跟随每日变化的 master。模型基线选用固定 revision 的 Qwen3.8-27B IQ3_S GGUF，同时锁定 MTP 变体和 BF16 视觉 mmproj。完整 clone、CUDA 构建、文本/图像/MTP 命令与 SHA-256 记录见[llama.cpp 基线配方](llama.cpp-baseline.md)。当前工作区忽略目录里的旧源码没有 Git 来源证明，GPU 实测时必须从该 tag 干净 clone。

The next execution pin is upstream `v0.6.0` (release commit prefix `d812350`), not the daily-moving master branch. The model baseline is a pinned Qwen3.8-27B IQ3_S GGUF together with its MTP variant and BF16 vision mmproj. The clean clone, CUDA build, text/image/MTP commands, and SHA-256 are recorded in the [llama.cpp baseline recipe](llama.cpp-baseline.md). The ignored source snapshot in the current workspace has no verified Git provenance; the GPU run must use a fresh clone at the fixed tag.

## 社区改版的定位

社区实现值得先查，尤其是目标量化格式或硬件路径已有实现时，但不默认把整条分支变成依赖：

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) 提供额外量化格式和 CPU/CUDA 性能改动。采用其中任何实现前，逐项确认 GGUF 类型、硬件后端、质量结果和维护范围是否符合 By1 的目标。
- [Qwen3.8 专用 llama.cpp 分支](https://github.com/holden093/llama.cpp-qwen3.8-flash-next) 针对 DwarfStar 打包的 Qwen3.8 Q2 GGUF 加了专门加载处理。仓库自述这是面向单机的专用改版；适合研究特殊打包与 loader 的差异，不应直接作为通用 By1 后端。

上游已经有该架构和 MTP 后，模型支持补丁就优先回到原版验证；如果 By1 要处理的只是特定打包布局，隔离布局适配并保留标准 GGUF 路径。

## 执行顺序

1. 固定一个上游版本，提供中英双语的 PowerShell 与 Linux 构建命令。
2. 用上游原版加载代表性标准 GGUF，跑文本生成；对含图像或 MTP 的模型分别跑对应路径。
3. 记录同机、同权重、同量化类型的原版基准与结果；再和 Transformers 发布方参考实现对拍 logits 和生成。
4. 将 By1 manifest、共享权重、量化 scale 与 GGUF 张量对应关系加入报告，区分可加载与数值正确。
5. 若遇到上游缺口，先查已公开的活跃派生版并核验补丁；优先向上游提交通用修复。暂时必须使用本地补丁时，保持小范围、可单独构建，并测试原版与补丁版。

## 验收条件

- 干净 clone 可按固定版本构建 llama.cpp。
- 标准产物能被原版加载，能完成生成；支持视觉/MTP 的模型分别通过其对应路径。
- 原版 llama.cpp、By1 和发布方参考实现的对照配置可复现；容差、准确度、速度和显存分项报告。
- 报告明确标注未测试的量化类型、硬件后端和多模态路径，不能把“模型能启动”写成完整支持。

## decode / 状态（T17 切片，2026-10-14 立判据）

rc1 的另一半是 **decode/状态**：生成循环里的 KV cache 与增量解码。两条判据
（缺一不算过）：

1. **增量与全量 logits 一致** —— 同权重同输入：一次全量前向的每步 logits，
   必须与"带 cache 逐 token 增量"的每步 logits 对得上（容差走
   [precision-budget](precision-budget.md) 的 1e-4 一族）。**这是状态
   正确性的唯一判据**：cache 写错一格，全量对拍照样绿，增量当场红。
2. **状态 round-trip** —— cache 可以原样导出再导入，继续 decode 的输出
   与不中断的 decode **逐 token 相同**（save/restore 不许改数值）。

切片顺序（每片都带自己的判卷人，不整包推进）：

| 切片 | 内容 | 判卷人 |
|---|---|---|
| S1 | `by1state` 的声明 -> KV 形状/预算核算复核（`MEM_PER_PAIR` 对照声明） | by1state 输出 vs 手算（30 秒那个任务的机器版） |
| S2 | 生成 C 增加 **decode 模式**：per-layer K/V 缓冲 + 单步函数 | 判据 1（增量 vs 全量）在 by1ggml 自检里 |
| S3 | cache 导出/导入（序列化 + 校验和） | 判据 2（round-trip 逐 token） |
| S4 | 接进 llama.cpp 本体（`llamacpp/` 那批参考实现的路数） | rc1 总判据：`llama-cli` logits 逐位 |

**已知边界**：S2 的 cache 维度按 `by1state` 声明走（kv_cache/recurrent/
conv_history 三类里先做 kv_cache）；递推类（recurrent/conv_history）是
delta_rule/ssm_scan 同族问题，归融合核路线，不在 S2 硬塞。

## T18 地形判定（2026-10-14 实测）

`Instella 真 GGUF 全链`（判据：`llama-cli` logits 逐位）的**前置事实**：

1. **上游 v0.6.0 不认识 instella 架构** —— `convert_hf_to_gguf.py` 与
   `src/` 全文无 `instella`（本地 grep 证据）。`amd/Instella-3B` 是
   `InstellaForCausalLM` / `model_type=instella`（32 头 MHA、head_dim=80、
   vocab 50304、bf16），**原版 llama-cli 连 load 都过不去**。
2. 因此全链 = **补丁版 llama.cpp**（本文件"社区改版的定位"一节的路）：
   查公开派生版 -> 核验补丁 -> 本地小补丁可单独构建 -> **原版与补丁版
   都测**。线索：AMD dev 社区「Hacking AMD's Instella-3B into llama.cpp」
   一帖；派生版 `kingjones30/ROCmFPX`。
   **补丁面核验（2026-10-14，克隆 `llamacpp/ROCmFPX` @ dfeacaf）**：
   它的 `LLM_ARCH_INSTELLA` 是 **Instella-MoE-16B**（DeepSeek-V3 MLA +
   sigmoid MoE + FarSkip，converter 注册的是 `InstellaMoEForCausalLM`），
   **不是 T18 的 dense Instella-3B**（`InstellaForCausalLM`）—— dense 3B
   在这个派生版里同样无支持。它点名 oracle 为
   `csabakecskemeti/llama.cpp@instella-moe`（下一步查它 + AMD 帖）。
   两支 Instella 是**两个模型**，不许混记成一个。
3. 工具链已就位：`llamacpp/build-cli/bin/` 三件（`llama-cli` ·
   `llama-eval-callback`（logits dump 的产者）· `llama-gguf`）。
   构建补记：MinGW 下 cpp-httplib 要 `-D_WIN32_WINNT=0x0A00`
   （`CreateFile2` 未声明）。
4. **判据形态**：`logits-run` 产两份 dump（同 GGUF 同参）逐位比 =
   基线自比对；真正的全链对照是 **补丁版产物 vs 官方产物** 的 logits
   逐位（同权重同实现下 bitwise 是可达的）。原版/补丁版差异单独记账。

**未验**（不吹）：补丁内容未核验、convert 未跑、logits dump 未产。
下一步顺序：核验 ROCmFPX 的 instella 面 -> 本地小补丁 -> 原版/补丁版
双构建 -> convert（上游工具）-> 双 dump 逐位。
