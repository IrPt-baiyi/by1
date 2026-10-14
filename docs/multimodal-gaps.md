# 多模态缺口单（T21 判据之一）/ Multimodal gap list

> 判据"端到端逐位 + **缺口单**"的后者。**只记在案的缺口，不圆场**：
> 每条带证据出处；修一条销一条（记录日期与判卷人）。
> 建单：2026-10-14（第 21 轮）。

## A. 夹具缺口（挡住 T21 边界对拍的直接原因）

| # | 缺口 | 证据 | 状态 |
|---|---|---|---|
| A1 | **微型音频夹具缺失**：llama.cpp 侧有 10 种音频预处理器（whisper / conformer / d1omni / granite-speech / gemma4a / qwen3a / dots3note / mimo / qwen3tts / pockettts / parakeet，`tools/mtmd/mtmd-audio.h`），但判据需要的"小音频模型 + mmproj + HF 源孪生"三件套没有现成的：`mradermacher/tiny-audio-GGUF` 只有文本 GGUF（**无 mmproj**）；`gemma-3n-audio` 转换仓是空壳（仅 .gitattributes）；HF 搜索接口 2026-10-14 被限流（504） | 本轮实测 | **半销（同日）**：官方夹具清单在 `tools/mtmd/tests.sh` 的 `add_test_audio` 里 —— **`ggml-org/Qwen3-ASR-0.6B-GGUF`**（0.6B，Q8_0+bf16 双 mmproj）与 `ultravox-v0_5-llama-3_2-1b`（mmproj-f16）；HF 源孪生 `Qwen/Qwen3-ASR-0.6B` / `fixie-ai/ultravox-v0_5-llama-3_2-1b` 均在。夹具已入库 `model-cache/audio` + `model-cache/qwen3asr-hf`，**边界探针未跑 = 未全销** |
| A2 | 大号候选在案但不划算：Qwen3-Omni-30B-A3B（refs 有契约，音频塔 = conv2d×3 + 32 层 Whisper 式 encoder，见 `src/checks/by1all.py` 注释）体积 60GB 级，不适合边界对拍轮 | by1all 注释 + refs 清单 | 未销 |

| A3 | **HF 参照侧 Qwen3-ASR 装载映射错位**（2026-10-14 第 25 轮实测）：checkpoint 是 thinker.*（audio_tower 18 层 + proj1/proj2，d_model 896 = mmproj 的 embedding_length 一致）；transformers 侧 Qwen3ASRForConditionalGeneration 实例化出 model.*（24 层 + multi_modal_projector.linear_1/2）—— 全量 MISSING/UNEXPECTED，权重全是随机的（此前若用它的前向等于没验）。根因：真配置在嵌套的 thinker_config（audio_config 为 Whisper 式），类读的是空顶层走默认。mel 对照不受影响（只用 feature_extractor，与权重无关） | LOAD REPORT 逐条 + config 实测 | **修复路径定案（同日）**：checkpoint 自报 transformers_version: 4.57.6，本地 5.15.1 跨两代 —— 类架构在两代间改过（4.57 代 thinker.* + audio_tower.proj1/proj2 + 18 层；5.x 代 model.* + multi_modal_projector.linear_1/2 + 24 层默认），无法忠实映射。正解 = 按版本钉住建独立 venv（transformers==4.57.6），与 llama.cpp 的 pin 同一纪律。venv 建设中（build/hf457），下轮在钉住版本下重跑塔边界 |

## B. by1 侧建模缺口

| # | 缺口 | 证据 | 状态 |
|---|---|---|---|
| B1 | **音频塔完全没建模**（INVENTORY 【F】原话）；by1verify 的多模态张量走排除通道（"Gemma 的 vision/audio、Qwen3.5 的 model.visual.*"） | INVENTORY.txt 【F】· by1verify 注释 | 未销 |
| B2 | 双塔同名消歧只有反推侧处理过（E4B 的 audio/vision 双塔，by1revname 339）；IR 侧 1.1 已有 STACKS 登记表（T22），但**没有任何模型真的登记过 audio 栈** | by1revname · by1ir STACKS | 未销 |
| B3 | 视觉塔"层外改名"一族（Omni 的 `merger.ln_q` / `mlp.0` / `mlp.2`）没有逐张量对照表 | by1all 注释 | 未销 |

## C. 端到端未验清单（"能跑"≠"验过"）

| # | 项 | 现状 | 状态 |
|---|---|---|---|
| C1 | 音频 -> 塔 -> 文本端到端逐位 | 未开跑（被 A1 挡） | **半销（2026-10-14 第 23 轮）**：mel 预处理边界**数值等价** —— 同一 440Hz 正弦下 mtmd 前 100 帧 vs HF Whisper 特征 **mean 4.7e-6 / max 5.8e-3**（fp 内核容差），列 0 逐位同；差异仅分块策略（mtmd 流式 1s 块 + flush 尾块：0.5/1/2s -> 1/2/3 块 x 100 帧；HF 整段一次 30s 填充）。判卷人 tools/audio-mel-probe.cpp + tools/qwen3asr-mel-ref.py。****塔边界首数（2026-10-14 第 27 轮）**：13x1024 @ audio_tower.proj2 —— max|d| 7.13e-02 / mean 6.05e-03 / token 余弦 mean 0.912（min 0.36）；参照侧在 transformers==4.57.6 钉住 venv 下装载（thinker.* 剥前缀 612 张量 unexpected=0，351 missing 全属 visual 未用支路）。漂移**同日归因翻案**：两侧权重同为 bf16（checkpoint bf16、mmproj 110 张 BF16）—— **漂移不是精度**（T20 的 bf16 故事在这里不成立）。新假设：mtmd 是一次过 200 帧出 26 token（预处理分块 + 整段编码），参照是 100 帧出 13 token —— **注意力长度不同**（26 长上下文 vs 13）。**终审判决（同日，第 29 轮）：塔边界等价成立** —— mtmd 的 200 帧 mel 直喂 HF 塔，**26 vs 26 同长对拍：max|d| 6.08e-03 / mean 1.42e-04 / 余弦 min 0.99698 mean 0.99987**（fp 内核容差族）。此前 6e-3/0.91 的漂移**全部是长度差**（26-token 通路 vs 13-token 通路的注意力长度不同），归因闭环。判卷人 tools/audio-mel-probe.cpp + tools/qwen3asr-embd-ref.py --mel。**精度对照 + 端到端首轮（2026-10-14 第 30 轮）**：双 mmproj 同长对拍 —— bf16 mean 1.42e-04 / cos 0.99987；**Q8_0 mean 6.36e-04 / cos 0.99906（4.5 倍劣化）** —— T20 的量化故事音频侧复刻 ✓。端到端双侧已出字：HF 贪心 32 token 出真文本（[English] The voice is a little bit too loud...）；mtmd-cli 走 chat 模板出 language/asr 标签流 —— **输入构造不同（模板/占位）生成自然不同**，端到端逐 token 对照的最后一格 = 两侧输入模板对齐（apply_chat_template vs mtmd 的模板引擎）。**模板取证（第 31 轮）**：GGUF 自带 chat 模板已提取（Qwen 式 role/content 标签流）——mtmd-cli 的输入=chat 模板+媒体标记，**e2e next-token 首判（第 101-105 轮）**：裸序列（151669+26x媒体+151670）两侧 next-token logits：**argmax 全同（151645）**、top5 2/5；分布漂移 mean 4.0 / cos 0.705 —— **非量化**（Q8_0 与 bf16 文本模型漂移几乎相同）、**非位置**（扁平 position_ids 无效）。剩余嫌疑：音频特征的注入点/池化链（HF 侧 get_audio_features 之后是否还有变换）**注入侧已验（同日）**：LM 首层输入的占位行 vs proj2 输出 **max 0.000e+00**（注入=裸 proj2 输出，无隐藏变换）。**新嫌疑=模型身份错配**：ggml-org 的 Qwen3-ASR-0.6B GGUF 文本塔 attn_q 形状 (2048,2048)，HF 的 Qwen/Qwen3-ASR-0.6B 是 (2048,1024) —— 两侧疑似**不同变体**；且我探针的 embd 步长按 1024 喂 2048 隐宽的 GGUF（若其 n_embd=2048 则整段注入是垃圾值）。**身份核对（同日）**：GGUF 元数据 = **qwen3vl** 块 28 层 / 隐宽 **1024** / ffn 3072 / 头 16-8 / head_dim 128 —— 与 HF checkpoint **超参全同**（2048 = 16x128 q 头输出 ✓ 之前形状惊魂是读法误会）。但 arch 是 **qwen3vl（带 n_deepstack_layers）**，我装载的是 Qwen3OmniMoeThinker —— 文本栈实现体不同源（deepstack 等内部件），这与 cos 0.705 的 e2e 漂移相符。**步长裁决（同日）**：llama_model_n_embd_inp=1024 = n_embd=1024（print_info 实测）—— 探针注入宽度无罪；(2048,2048) 形状读数归 gguf-py 的存储布局/读法（qwen3vl 可能存融合张量），留线。**决定性二分（同日）**：文本无音频对拍（ids 785 6722 315 9625 374）—— **max|d| 6.0e-02 / mean 1.0e-02 / cos 0.999998 / argmax 全同**（12095）—— **文本栈等价**（qwen3vl 与 OmniMoeThinker 文本体同源），e2e 的 cos 0.705 漂移**全部锁进音频整合段**（音频特征 -> LM 的那一段），尽管注入值与 proj2 逐位同、塔边界 1.4e-04 同。终审候选：音频行的注意力掩码/位置**位置网格落锤（同日）**：钩 rotary_emb 实测 position 网格 = (3, 1, 28) **三段全为扁平 [0..27]** —— 纯音频输入下 M-RoPE 退化为扁平位置，与 llama 侧 slot 序一致（又一排除）。至此七项测量（塔边界/注入/文本栈/步长/量化/位置/宽度）全部相合，而 e2e 分布仍 cos 0.705 —— **终审候选收窄为「对照运行一致性」**（两次运行的 dump/段落错位嫌疑）与音频注意力掩码细节；**单进程四查（同日）**：embd 段与 HF proj2 逐值核对 1.42e-04 ✓、dump 段落解析一致 ✓、文本栈等价 ✓、位置扁平 ✓ —— 运行一致性无罪；next-token 漂移（mean 4.0 / cos 0.705）确证在 **LM 对 26 行音频的注意力处理**。终审候选：**音频块内注意力模式**（因果 vs 块内双向—— Whisper 式前缀若在 HF 侧是块内双向，llama 侧逐批因果即分叉）。**掩码裁决（同日）**：HF 侧 _prepare_4d_causal_attention_mask = 标准因果 4D（无媒体块特例）—— 双向假设出局。终审余两支：(a) llama 侧分批 decode 的批内因果掩码语义（embd 行）(b) 文本栈 1e-2 级基底噪声经音频路径放大。终审设计：llama 侧单批全 28 行（文本行也走 embd）复算对照。**放大说证伪（同日）**：HF 侧换喂 llama 的 embd（输入差 1.4e-04）——logits 仅动 max 8.0e-02 / mean 1.8e-02 / cos 0.999986，**无放大**；e2e 的 4.0 漂移是噪声解释不了的 100 倍 —— 剩余唯一支：**llama 侧音频段的批内计算语义**（分批 decode 的掩码/位置在 embd 行上的行为）。终审=单批全 28 行复算。llama 侧探针新能力：分块混合 batch（token/embd 分开喂，llama_decode 拒混合 batch；get_logits_ith 下标是批内下标）|探针实测 | 未全销 |
| C2 | 视觉端到端（T20 的塔边界已验到同精度等价，但**端到端生成**没跑） | T20 ✅ 仅塔边界 | 未销 |
| C3 | 多模态 KV cache / decode（T17 只做了文本） | 文本 ✅ 多模态 ✗ | 未销 |
| C4 | mmproj 产物链（by1 写手出 mmproj、双产物逐位——T18 的多模态版） | 未开跑 | 未销 |

## 销账规则

销一条 = 补上判卷人与数字（例："A1 销：夹具 X 入库，边界对拍 max|Δ| …，判卷人 tools/audio-*-probe"）。
**没有数字的销账不算销。**
