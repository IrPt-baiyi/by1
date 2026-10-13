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
