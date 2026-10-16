# P0 闭环报告 · Qwen3-0.6B

> 日期 **2026-10-16** · 机器：本机 Windows · **CPU-only（无显卡）** ·
> 代表模型选择记录见 [p0-model-selection.md](p0-model-selection.md)。
>
> 本报告只写**今天真正跑出来的那几段**。没跑的单独列在第四节 ——
> **不写成通过**，这是这个仓库对"证据"的最低要求。

## 结论（一句话）

`.by1 声明 → 真实产物对账 → 标准 GGUF → 未修改的上游 llama.cpp 加载与生成`
四段，在 Qwen3-0.6B 上都有可重跑证据：**311/311 张量命中**、
**上游 logits 逐位相同（106949 个值）**、**生成文本完全相同**。

## 一、输入（每项可核对）

| 项 | 值 | 来源 / 命令 |
|---|---|---|
| 模型 | `Qwen/Qwen3-0.6B`（base，文本） | p0-model-selection.md |
| 许可 | Apache-2.0 | refs/SOURCES.tsv |
| 声明 | `models/complete-Qwen3-0.6B.by1` | `by1check` 0 错 / 0 警告 |
| config | `refs/Qwen__Qwen3-0.6B.config.json`（726 B） | `python src/data/by1fetch.py --only Qwen3-0.6B`（8.4 s） |
| 张量清单 | `refs/Qwen__Qwen3-0.6B.tensors.json`（**311** 张） | 同上 |
| 真权重 | `model-cache/qwen3-0.6b/model.safetensors`（**1433.7 MB**） | HF resolve（hf-mirror），含 tokenizer/config 全套 |
| 上游运行时 | `llamacpp/build-cli/bin/llama-cli.exe` · `llama-eval-callback.exe` | 本机构建 |

config 与声明的关键字段逐项吻合：`qwen3` · 28 层 · 16:8 头 · `head_dim` 128 ·
`tie_word_embeddings=true` · `rope_theta` 1e6 · `max_position_embeddings` 40960。

## 二、四段证据

### ① 声明 ↔ 发布方产物（T12）

```bash
python src/modelcheck/by1shadow.py --smoke-refs --by1 complete-Qwen3-0.6B.by1
```

```
源二影子 …：311 个真名（2.80 GB）
源二 e2e OK：双 loader 零 missing/unexpected，前向 max|Δ| = 0.000e+00
```

权重文件本身也逐张量核过（对照同一份清单）：

```
文件大小 1433.7 MB · 头里张量 311 · refs 清单 311
名字集合相同: True · 形状不一致: 0 · 按头计算权重字节 1433.6 MB（自洽）
```

**没销的问题是 tie 语义**：清单里 `lm_head.weight` 与
`model.embed_tokens.weight` 同时存在，而 config 说 `tie_word_embeddings=true`；
Transformers 加载时给出警告"两者都在且值不同，故不绑"。
**存储布局 vs 共享语义要写清** —— 这正是"同名不同义"的高发区，见第四节。

### ② 标准产物（T16）

```bash
# 基准：上游自己的转换器（tokenizer KV 的来源，也是逐位基准）
python llamacpp/llama.cpp/convert_hf_to_gguf.py model-cache/qwen3-0.6b \
    --outfile build/qwen3-0.6b-ref-f32.gguf --outtype f32        # 3.01 GB

# by1 产物线（KV 表驱动；tokenizer KV 从 ref 整拷 —— S1 记档）
python src/checks/by1llama.py build-gguf model-cache/qwen3-0.6b \
    build/qwen3-0.6b-by1-f32.gguf --ref build/qwen3-0.6b-ref-f32.gguf \
    --arch qwen3 --n-blocks 28                                    # 311 张量 · 2.81 GB
```

> **本次顺带修掉的入口缺口**：`build_gguf` 原来是个**没有调用者的函数**
> （`main()` 的 7 个子命令里没有它，runbook 里那条命令跑不通），
> 而且 `arch` 被硬编成 `LLAMA`、`n_blocks` 默认 24 —— 那正是
> gap **S2**"只 qwen2 档实证"的字面来源。现在入口按参数走。

### ③ 未修改的上游运行时加载 + logits 逐位（T17）

```bash
CLI=llamacpp/build-cli/bin/llama-eval-callback.exe
python src/checks/by1llama.py logits-run --gguf build/qwen3-0.6b-ref-f32.gguf \
    -p "The capital of France is" -o cgen/qwen3-ref.logits --llama-cli $CLI
python src/checks/by1llama.py logits-run --gguf build/qwen3-0.6b-by1-f32.gguf \
    -p "The capital of France is" -o cgen/qwen3-by1.logits --llama-cli $CLI
python src/checks/by1llama.py logits-compare cgen/qwen3-ref.logits cgen/qwen3-by1.logits
```

```
报告写了 cgen/llama-logits-compare.json
[PASS] logits 逐位相同（106949 个值）
```

### ④ 短序列生成（T14 的一部分）

```bash
llamacpp/build-cli/bin/llama-cli.exe -m <gguf> -p "The capital of France is" \
    -n 16 --seed 42 --temp 0 -st
```

两份产物（上游 ref 与 by1 产物）输出**同一段文本**：

```
> The capital of France is
Okay, the user is asking about the capital of France. Let me
```

## 三、这次销掉的缺口（docs/gap-analysis.md 已同步）

- **S2**（build-gguf 族覆盖只 qwen2）：**qwen3 档已实证** —— 311 张量全映射、
  logits 逐位、生成同文。其余族仍不外推。
- **G2**（外部环境复现无记录）：**外部复现 #1** 已在
  `history/external-repro/2026-10-16-colab-hello.md`（Colab CPU 入门全对，
  并抓到 `head_dim=31` 缺口）。

## 四、没跑的（**不是通过**）

| 项 | 状态 | 原因 / 收口路径 |
|---|---|---|
| **T13 真权重数值对拍** | **未跑** | `by1gputrain.py:149` 硬要 `torch.cuda.is_available()`、`by1gpubench.py` 写死 `device_map={'':'cuda:0'}` —— **本机无卡跑不了**。两条路：给工具加 `--device cpu`（0.6B fp32 约 2.4 GB 内存，可行），或留租卡日 |
| **tie 语义** | 未定论 | 清单同时含 embed 与 lm_head，config 说 tie=true，加载器未绑。要给结论（绑 / 证明不绑） |
| **S1 tokenizer 组装线** | 仍未实现 | 本次 GGUF 的 tokenizer KV 从上游 ref 整拷 —— 按记档，非缺陷伪装 |
| 其余模型族 | 不外推 | Qwen3-0.6B 一个新档，不代表别的族 |

## 五、复现（按顺序，全程 CPU）

```bash
python src/data/by1fetch.py --only Qwen3-0.6B          # refs 语料
# 权重：HF Qwen/Qwen3-0.6B 的 model.safetensors + config.json + tokenizer 全套
python src/modelcheck/by1shadow.py --smoke-refs --by1 complete-Qwen3-0.6B.by1
python llamacpp/llama.cpp/convert_hf_to_gguf.py <hf-dir> --outfile ref-f32.gguf --outtype f32
python src/checks/by1llama.py build-gguf <hf-dir> by1-f32.gguf --ref ref-f32.gguf \
    --arch qwen3 --n-blocks 28
python src/checks/by1llama.py logits-run --gguf ref-f32.gguf -o ref.logits --llama-cli <eval-callback>
python src/checks/by1llama.py logits-run --gguf by1-f32.gguf -o by1.logits --llama-cli <eval-callback>
python src/checks/by1llama.py logits-compare ref.logits by1.logits
```

> 数字跟着机器走；本报告的值出自上面这台机器的那次运行。
> 上游版本与构建参数仍按 [llama.cpp 基线](llama.cpp-baseline.md) 钉。
