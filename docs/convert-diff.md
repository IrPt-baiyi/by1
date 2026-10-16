# by1 build-gguf 与上游 convert_hf_to_gguf 的结构差异（T24）

判据：**与上游 convert 的结构差异逐条解释**。产物级证据（T23 已判）：
F32 张量区 290/290 逐字节全同、Q8_0 块 290/290 逐字节全同、
llama-cli logits 逐位 max|Δ|=0 —— **产物等价，差异全在过程**。

## 机器取证（2026-10-14）

| 量 | by1 build-gguf | 上游 convert | 裁决 |
|---|---|---|---|
| GGUF 版本 | 3 | 3 | 同 |
| KV | 27 键（原字节整拷 ref） | 27 键（按 config 生成） | **逐字节同** |
| 张量名集 | 290 | 290 | 对称差 0 |
| 数据型 | 全 F32（Q8_0 可选） | 全 F32（`--outtype` 可选多型） | 同档可比 |

## 结构差异逐条

| # | 差异 | by1 侧 | 上游侧 | 解释 |
|---|---|---|---|---|
| 1 | **驱动源** | `.by1` 声明（KV 表 `BY1_TO_GGML_KV`） | `config.json` | by1 是声明式规格，config 是框架产物；两者的超参可互验（by1check 对 Qwen2.5-0.5B 0/0/0） |
| 2 | **KV 生成** | 张量区自建；KV 原字节整拷 ref（tokenizer 等） | 全部生成 | 分词器组装是独立生产线（by1 未实现）——范围记档，非缺陷伪装 |
| 3 | **张量名映射** | `TensorNameMap.get_name(try_suffixes=('.weight','.bias'))` 直查 | 同一 TensorNameMap（converter 内用法） | **同一份上游映射**，调用面不同 |
| 4 | **数值型** | F32 + 自建 Q8_0 编码器（`_q8_0_encode`） | f16/bf16/f32/Q8_0/Q4 等多型 | by1 只实现判据需要的两档；多型是"要么实现要么拒绝"的拒绝面 |
| 5 | **量化块语义** | 1D 张量保留 F32、2D+ 逐块（32 值/块） | 同（实测 181 张 1D 保 F32） | 与 llama-quantize 行为逐位对齐（290/290） |
| 6 | **张量 info 形状** | 逻辑形单传（块数组形不入 info） | 逻辑形 | 初版坑：块数组形入 info = 168 张量假异（已修，坑账在提交 9317ec8→38800fd） |
| 7 | **旋转频率** | 不生成（ref 有 inv_freq 则拷贝） | 生成 `rope_freqs` | by1 的产物链以 ref 为 KV 底座；自生成是待接线项 |
| 8 | **模型族覆盖** | Qwen2.5-0.5B（qwen2）+ **Qwen3-0.6B（qwen3，2026-10-16）** 实证 | 全族 | 两族已跑：qwen3 档 311 张量全映射、上游 logits **逐位相同（106949 值）**、llama-cli 生成同文（docs/p0-loop-report.md）；**其余族仍不外推** |

## 反例必红

- KV 合成表对未知 arch 必红（垃圾文件 → `未知 arch 前缀`）；
- qwen3vl 产物过表绿（跨架构建绿 9/9）——表的边界即判据。

## 判卷人

`src/checks/by1llama.py`（build_gguf / _gguf_write / selftest_kv_map）·
`llamacpp/build-cli/bin/llama-eval-callback.exe` · 上游 `convert_hf_to_gguf.py` / `llama-quantize`。

## 怎么跑（2026-10-16 接上入口）

```bash
# 0) tokenizer KV 的来源 / 逐位基准：上游自己的转换器
python llamacpp/llama.cpp/convert_hf_to_gguf.py <hf-dir> \
    --outfile build/ref-f32.gguf --outtype f32

# 1) by1 的产物线（KV 表驱动；tokenizer KV 从 --ref 整拷 —— 第 2 行的范围记档）
python src/checks/by1llama.py build-gguf <hf-dir> build/by1-f32.gguf \
    --ref build/ref-f32.gguf --arch qwen3 --n-blocks 28

# 2) 未修改的上游运行时加载 + logits 逐位（两份产物各产一份 dump）
python src/checks/by1llama.py logits-run --gguf build/ref-f32.gguf \
    -o cgen/ref.logits --llama-cli <llama-eval-callback>
python src/checks/by1llama.py logits-run --gguf build/by1-f32.gguf \
    -o cgen/by1.logits --llama-cli <llama-eval-callback>
python src/checks/by1llama.py logits-compare cgen/ref.logits cgen/by1.logits
```

**两个前置**（都不在库里，按 upstream-first 另取）：

- `gguf-py` 来自 **llama.cpp 检出**（`llamacpp/` 被 `.gitignore` 挡着）。
  在仓库里跑会自动用它；在别处跑用 `PYTHONPATH=<llamacpp>/gguf-py`。
  缺了会给出明确提示（不是裸 traceback）。
- `--ref` 必须是**真 GGUF**：S1 记档说 tokenizer KV 由上游产物整拷。

**入口这条账**：`build_gguf` 原来是库里一个**没有调用者**的函数
（`main()` 的子命令里没有它，runbook 里那条命令跑不通），且 `arch` 硬编成
`LLAMA`、`n_blocks` 默认 24 —— 那正是第 8 行"只 qwen2 档实证"的字面来源。
现在按参数走，qwen3 档已实证（[P0 闭环报告](p0-loop-report.md)）。
