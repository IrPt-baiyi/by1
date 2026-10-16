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
| 8 | **模型族覆盖** | Qwen2.5-0.5B（qwen2）+ **Qwen3-0.6B（qwen3，2026-10-16）** + **mamba-130m（mamba，2026-10-16）** 实证 | 全族 | 三族已跑：qwen3 档 311 张量、mamba 档 242 张量，上游 logits 均**逐位相同**（见 docs/p0-loop-report.md、docs/t26-plan.md）；**其余族仍不外推** |
| 9 | **转换期改写**（`modify_tensors`） | 按 arch 的表内规则改**值或形状**（mamba 两条，见下） | `conversion/<arch>.py` 里的同名钩子 | **名字映射只解决"叫什么"**；这两件事不同。少了改写，产物要么装不上、要么**形状全对而算的是另一个东西** |

## 反例必红

- KV 合成表对未知 arch 必红（垃圾文件 → `未知 arch 前缀`）；
- qwen3vl 产物过表绿（跨架构建绿 9/9）——表的边界即判据。

## 转换期改写（`modify_tensors`）：mamba 这一族逼出来的第 9 行

**名字映射不等于搬运。** 上游 `conversion/<arch>.py` 里有一个 `modify_tensors` 钩子，
它会在写盘前**改值或改形状**；`gguf-py` 的 `TensorNameMap` 只给名字，不给这个。
2026-10-16 实测（`--arch mamba`，242 张量）：

| 规则 | 上游出处 | 少了它的后果 |
|---|---|---|
| `A_log → A = -exp(A_log)`（**改值**） | `conversion/mamba.py:84-86` | 形状全对、装得上、**算的是另一个模型**（把 log 值当 A 用） |
| `ssm_conv1d.weight` → `squeeze()`（**改形状**） | `conversion/mamba.py:89-90` | llama.cpp 拒绝加载：`check_tensor_dims: ... expected 4, 1536, got 4, 1, 1536, 1` |

**一条算子级的教训**：第一版用 `numpy.exp`，与上游 `torch.exp` 差 ~1 ulp，
于是 `ssm_a` 在 24 层上各差一点点、logits 首个分歧 @1034（max\|Δ\|=0.625）。
**判卷人是上游那一行，算子也得是同一个** —— 换成 `torch.exp` 后：
242/242 张量逐位相同、上游 logits **逐位相同（30604 值）**。

（**递归模型对这个尤其敏感**：SSM 每步都乘 A，1 ulp 会沿序列放大；
KV-cache 那类模型没有这条放大链。）

## 混合栈（granite 4.0-h）逼出来的另外四条（2026-10-16，T27）

同一件事在混合栈上更露骨：**370 个 HF 张量要变成 402 个 GGUF 张量**，
而且中间有三种"看不见的改写"。四条都在上游的钩子里，映射表一条都没有：

| 规则 | 上游出处 | 少了它的后果 |
|---|---|---|
| `mamba.dt_bias` → 先改名成 `.dt_proj.bias` 再查表 | `conversion/mamba.py` 的 `filter_tensors` | 该张量查不到名字 → **静默丢掉**（包能打出来、装不上） |
| `shared_mlp.input_linear.weight` → **拆成** `ffn_gate` + `ffn_up`（无专家时） | `conversion/granite.py:211-222` | 张量数 370 ≠ 上游 402（32 层各少一个）；名字也不对（映射表给的是 `*_shexp`） |
| `self_attn.q_proj` / `k_proj` → **置换**（`undo_permute = True`） | `conversion/llama.py` 的 `permute` | **形状一个不差、名字一个不差，算出来不一样**（实测 max\|Δ\| 0.18~0.42，四个注意力层全中）——最难从症状上看出来的一类 |
| `shared_mlp.output_linear.weight` → **无专家**时落 `ffn_down` | `conversion/granite.py:223` | 映射表按"有专家"给 `ffn_down_shexp`，与上游对不上 |

外加 **Mamba2 的三条形状规则**（`conversion/mamba.py:184-194`，Mamba1 没有）：
`ssm_conv1d` squeeze、`ssm_a` / `ssm_d` 各 unsqueeze 一维、
`ssm_norm` reshape 成 `(n_groups, d_inner // n_groups)`；**顺序**也要照上游
（先 reshape，再对 `A_log` 取负 exp）。

**收口之后**：402/402 张量逐位相同 · 上游未修改的 llama.cpp 加载 by1 的产物，
logits **逐位相同（89474 值）**。

**一条纪律也在这里落地**：上游对"有专家"的形态用的是另一套名字
（`*_shexp`）。by1 的产物线**只实现了无专家那一支**，遇到
`num_local_experts > 0` 时**直接报错拒绝** ——
「要么实现，要么拒绝」，不产出一个"名字看着对、算的是另一个模型"的包。

## 把标量**折进权重**（Falcon-H1，2026-10-16，T29）

上面那些改写改的是**名字**与**形状**。Falcon-H1 逼出第三类，而且是最隐蔽的一类：
**名字一个不差、形状一个不差，只是每个数都乘了一个系数。**

实测：不折的话，**579 个张量里 218 个值不同**（`attn_k`/`attn_output`/`ssm_in`/
`ssm_out`/`ffn_gate`/`ffn_down` 全中）。上游 `conversion/falcon_h1.py:60-96`
把这一族的九个缩放**乘进对应的权重**：

| HF 张量（名字含） | 乘的系数 |
|---|---|
| `down_proj` / `gate_proj` | `mlp_multipliers[1]` / `[0]`（`up_proj` **不乘**） |
| `k_proj` | `key_multiplier` × `attention_in_multiplier` |
| `q_proj` / `v_proj` | `attention_in_multiplier` |
| `o_proj` | `attention_out_multiplier` |
| `out_proj` | `ssm_out_multiplier` |
| `in_proj` | 先乘 `ssm_in_multiplier`，再**按行分五段**乘 `ssm_multipliers[0..4]`（z / x / B / C / dt；段宽 = `d_ssm` 与 `n_groups × d_state`） |
| `lm_head` / `embed_tokens` | `lm_head_multiplier` / `embedding_multiplier` |

**顺序也照抄**：先跑父类（`Mamba2Model` 的三条形状规则 + `A_log` 取负 exp），
**再**乘系数 —— 反过来的话 `A_log` 那一支会算错。

**收口之后**：579/579 张量逐位相同 · 上游未修改的 llama.cpp 加载 by1 的产物，
logits **逐位相同（188445 值）**。

> 这一类改写对"自研后端"是个提醒：**运行时缩放**与**折进权重**在数学上等价，
> 但产物里只能有一种。判卷人是上游那一行代码，不是"我觉得等价"。

## 改写在代码里成了一张**表**（2026-10-16，T30 的 E3）

三类八条（含九个折标量）原来散在 `by1llama` 的五个函数里、只活在注释中。
现在它们是 `CONVERT_RULES` 里的一行行，每条带三样东西：

| 字段 | 是什么 |
|---|---|
| `cite` | **上游那一行**：`conversion/<arch>.py:行号` 或 `:符号`。行号核行、符号核"那个名字还在文件里"（符号比行号耐用） |
| `probe` | 一个小 fixture（HF 名 + GGML 名 + 形状 + 是否要 torch） |
| `check` | 拿**真实管线**（`_gguf_name` / `_tensors_for`）跑出来的结果断言；期望值是**独立写一遍**的上游公式，不是把实现抄一遍 |

判卷人是 `by1rules.py`（**已进 `by1all` 判卷人表，档 3**）：

```bash
python src/checks/by1rules.py        # 或 by1llama.py selftest-rules
```

三态：**通过 / 失败 / 没验**（缺 numpy、torch、gguf-py，或上游检出不在 ——
"没验"会写出来，不混进"通过"）。

**当前 16 条：破 16 条出处全部核对通过、探针全过。**

为什么要它：这个仓库栽过同一类跟头 —— **没进门禁的检查会烂**
（`by1langspec` 那条欠账；`by1sample` 不在表里时 min_p 那 8 格红没人知道）。
改写表是产物线的核心知识，不能只活在注释里。

加一条规则 = 加一行；**加一行而自检没红，才算真的加上了**。
重构后的回归：五个模型（311 / 242 / 402 / 506 / 579 个张量）**逐位一致**，
最大差 `0.000e+00`。

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
