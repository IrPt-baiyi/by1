# 多模态缺口单（T21 判据之一）/ Multimodal gap list

> 判据"端到端逐位 + **缺口单**"的后者。**只记在案的缺口，不圆场**：
> 每条带证据出处；修一条销一条（记录日期与判卷人）。
> 建单：2026-10-14（第 21 轮）。

## A. 夹具缺口（挡住 T21 边界对拍的直接原因）

| # | 缺口 | 证据 | 状态 |
|---|---|---|---|
| A1 | **微型音频夹具缺失**：llama.cpp 侧有 10 种音频预处理器（whisper / conformer / d1omni / granite-speech / gemma4a / qwen3a / dots3note / mimo / qwen3tts / pockettts / parakeet，`tools/mtmd/mtmd-audio.h`），但判据需要的"小音频模型 + mmproj + HF 源孪生"三件套没有现成的：`mradermacher/tiny-audio-GGUF` 只有文本 GGUF（**无 mmproj**）；`gemma-3n-audio` 转换仓是空壳（仅 .gitattributes）；HF 搜索接口 2026-10-14 被限流（504） | 本轮实测 | **半销（同日）**：官方夹具清单在 `tools/mtmd/tests.sh` 的 `add_test_audio` 里 —— **`ggml-org/Qwen3-ASR-0.6B-GGUF`**（0.6B，Q8_0+bf16 双 mmproj）与 `ultravox-v0_5-llama-3_2-1b`（mmproj-f16）；HF 源孪生 `Qwen/Qwen3-ASR-0.6B` / `fixie-ai/ultravox-v0_5-llama-3_2-1b` 均在。夹具已入库 `model-cache/audio` + `model-cache/qwen3asr-hf`，**边界探针未跑 = 未全销** |
| A2 | 大号候选在案但不划算：Qwen3-Omni-30B-A3B（refs 有契约，音频塔 = conv2d×3 + 32 层 Whisper 式 encoder，见 `src/checks/by1all.py` 注释）体积 60GB 级，不适合边界对拍轮 | by1all 注释 + refs 清单 | 未销 |

## B. by1 侧建模缺口

| # | 缺口 | 证据 | 状态 |
|---|---|---|---|
| B1 | **音频塔完全没建模**（INVENTORY 【F】原话）；by1verify 的多模态张量走排除通道（"Gemma 的 vision/audio、Qwen3.5 的 model.visual.*"） | INVENTORY.txt 【F】· by1verify 注释 | 未销 |
| B2 | 双塔同名消歧只有反推侧处理过（E4B 的 audio/vision 双塔，by1revname 339）；IR 侧 1.1 已有 STACKS 登记表（T22），但**没有任何模型真的登记过 audio 栈** | by1revname · by1ir STACKS | 未销 |
| B3 | 视觉塔"层外改名"一族（Omni 的 `merger.ln_q` / `mlp.0` / `mlp.2`）没有逐张量对照表 | by1all 注释 | 未销 |

## C. 端到端未验清单（"能跑"≠"验过"）

| # | 项 | 现状 | 状态 |
|---|---|---|---|
| C1 | 音频 -> 塔 -> 文本端到端逐位 | 未开跑（被 A1 挡） | 未销 |
| C2 | 视觉端到端（T20 的塔边界已验到同精度等价，但**端到端生成**没跑） | T20 ✅ 仅塔边界 | 未销 |
| C3 | 多模态 KV cache / decode（T17 只做了文本） | 文本 ✅ 多模态 ✗ | 未销 |
| C4 | mmproj 产物链（by1 写手出 mmproj、双产物逐位——T18 的多模态版） | 未开跑 | 未销 |

## 销账规则

销一条 = 补上判卷人与数字（例："A1 销：夹具 X 入库，边界对拍 max|Δ| …，判卷人 tools/audio-*-probe"）。
**没有数字的销账不算销。**
