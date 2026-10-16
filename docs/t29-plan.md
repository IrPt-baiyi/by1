# T29：另一根轴 —— **层内并行**（Falcon-H1）

> ⚠ **编号对照**：本文的"T29"是**工作稿编号**。路线图里 **T29 = 模型毕业流程**、
> **T28 = 多后端一致性**（都是另外两件事，还没做）；本文内容属于路线图
> **[T27「混合栈与特殊机制」](1.0-roadmap.md)** 里"特殊机制"那一半 ——
> 层内并行是一种**机制组合方式**，不是逐层混排。
>
> T27/T28 验的是"**层与层不同**"（这一层 SSM、那一层注意力）。
> 这份计划面对的是另一个方向：**同一层里两种机制并存**。
> 它不是"再加一个家族分支"能解决的 —— **卡在语言上**。

## 一、判卷人（2026-10-16 实测）

| 侧 | 证据 |
|---|---|
| transformers 5.15.1 | `AutoModelForCausalLM` → **`FalconH1ForCausalLM`** · **521.4 M 参数** · 前向 logits `(1, 4, 32784)`、`max\|.\| = 42.334` · 全部 **579 个张量** |
| llama.cpp | `MODEL_ARCH.FALCON_H1`（`"falcon-h1"`）· `conversion/falcon_h1.py` 存在 |

权重已在 `model-cache/Falcon-H1-0.5B-Instruct`（0.97 GB，CPU 跑得动）。

## 二、结构证据（实测，不是读文档猜的）

**每一层里三样东西并存**（layer 0 为例，579 个张量）：

| 属于 | 张量 | 形状 | 读出来的宽度 |
|---|---|---|---|
| SSM（Mamba2） | `mamba.A_log` / `mamba.D` / `mamba.dt_bias` | 24 | = `n_heads` |
| | `mamba.conv1d.weight` / `.bias` | 1792 × 1 × 4 / 1792 | 1792 = 1536 + 2×128×1 |
| | `mamba.in_proj.weight` | 3352 × 1024 | 3352 = 1536 + 1536 + 24 + 256 |
| | `mamba.out_proj.weight` | 1024 × 1536 | d_ssm = 24 × 64 = **1536** |
| 注意力 | `self_attn.q_proj` | 512 × 1024 | 512 = 8 × 64 |
| | `self_attn.k_proj` / `v_proj` | 128 × 1024 | 128 = 2 × 64 |
| | `self_attn.o_proj` | 1024 × 512 | —— |
| 前馈 | `feed_forward.gate_proj` / `up_proj` / `down_proj` | 2048 × 1024 / 1024 × 2048 | SwiGLU |
| 归一化 | `input_layernorm` · **`pre_ff_layernorm`** | 1024 | **多一个**（FFN 前那道） |

外加 `lm_head.weight`（**独立**，`tie_word_embeddings = false`）与 `model.final_layernorm`。

**接线**（读 `FalconH1DecoderLayer.forward` 得来，逐行对照）：

```
residual = x
x   = input_layernorm(x)
a   = mamba(x) * ssm_out_multiplier                 # 支路一
b,_ = self_attn(x * attention_in_multiplier, …) * attn_out_multiplier   # 支路二
x   = a + b                                          # ← **层内并行相加**
x   = residual + x
residual = x
x   = pre_ff_layernorm(x)
x   = feed_forward(x)
x   = residual + x
```

**注意 `a + b` 这一行**：两条支路**读同一个归一化输入**、输出**相加**——
不是"先 SSM 再注意力"，也不是"二选一"。所以逐层类型表（T27/T28 那条路）
在这里**根本不适用**：每一层的类型是"两个都要"。

## 三、语言缺口（这是这份文档的重点）

**by1 现在的 `pattern` 轴是"一层一个混合器"。** 证据：

- 语法上 `pattern` 是机制序列：`*` 重复、`+` 拼接（`drafts/step5-schedule-probe.by1`
  里就用了 `slide + full`），展开后 **`layer_seq` 每层一条记录、只带一个机制**；
- 每层**再挂一个 feed**（`main[:] >> FFN(...)`），这是唯一的"第二个东西"；
- 把仓库里 49 份 `.by1` 全扫了一遍：**没有任何一份出现过"一层挂多个机制"**
  （`layer_seq` 里 attached 只有 feed，且长度恒为 1 或 0）；
- 语言里另有一条 `interop` / residual 轴，但它是给**层间检索边**用的
  （[pattern-design](pattern-design.md) 里 AttnRes 那种），不是层内并行。

结论：**今天写不出 Falcon-H1 的声明** —— 不是后端不支持，是**语言没有这个词**。
这正是"提升上限"这句话最实的一处：**上限不在支持多少个模型，在语言能描述什么。**

## 四、最小扩展的三个选项

| 选项 | 写法（示意） | 爆炸半径 |
|---|---|---|
| **A. 一层多个混合器** | `pattern = 36 * (Mixer \| Attn)` | 改 `layer_seq` 的形状 → **IR + 三个后端 + 张量契约 + by1full 全要动** |
| **B. 组合机制**（推荐先做） | `mech Both : Parallel { a = Mixer; b = Attn }`，再 `pattern = 36 * Both` | 语言层新增一种**机制的展开**（把子机制展开成一段并行子图）；IR 的结构不变，但校验、后端求值、张量契约都要认这种新算子 |
| **C. 停在规格层** | 只写文档（像 [Step-5](step5-contract-plan.md) 那样） | 零改动，但这一类模型永远进不来 |

**建议的走法：先做 B 的"最小可用版本"，而且只做到能验一步为止**：

1. **先让"描述 + 配置"这条路通**：`by1check` 能收这份声明、`by1full` 能出
   `FalconH1Config`（config 逐字段对拍）——这一层不需要动后端；
2. **重后端暂时明确拒绝**：`by1exec` / `by1c` / `by1ggml` 遇到 `Parallel`
   就报"这一类还没实现"（**要么实现，要么拒绝**，不假装）；
3. **判据用现成的**：真权重双 loader 前向 + 混合状态增量 == 全量
   （两个判卷人都在，与 T26/T27 同一把尺）。

这样做的理由是**纪律一致性**：T26 的 mamba、T27 的混合栈，都是
"先声明能过 + 一个后端能算 + 判据逐位"，再谈别的后端。T29 照抄这条路。

### 落点清单（2026-10-16 实测，实施时按这张表走）

| 文件:行 | 是什么 | 加 `Parallel` 要做什么 |
|---|---|---|
| `src/by1ir.py:411` | `KINDS = sorted(KIND_ATTRS)` —— **IR 的 kind 闭集，唯一真相源** | 加一项（含它的属性表） |
| `src/by1ir.py:1195` | 校验：kind 不在闭集里就报错 | 自动生效 |
| `src/by1check.py:729` | `BUILTIN_MECHS`（声明里能不能直接用内置名） | 决定 `Parallel` 是内置名还是必须显式声明 |
| `src/by1check.py:701` | `TOKEN_MIXER`（哪些 kind 算 token 混合器 → 决定"每层要不要挂 FFN"） | **关键判断**：`Parallel` 自己就是一个混合器，不该再要求挂 FFN |
| `src/lang/by1blocks.py:153` | 用 `TOKEN_MIXER` 判层 | 同上 |
| `src/by1codegen.py:95` / `:439` / `:497` | `SUPPORTED_KINDS` · `MIXER_KINDS` · 拒绝 | 不认识就拒绝（**现成**） |
| `src/by1c.py:56` / `:1486` | `C_KINDS` · 拒绝 | 同上 |
| `src/by1ggml.py:68` / `:83` / `:115` | `SUPPORTED_KINDS` · 拒绝 | 同上 |
| `src/by1exec.py:1070` / `:1135` | `OPS` 表：每个 kind 都要有实现，缺了就列出来 | **"要么实现要么拒绝"的现成机制** |
| `src/by1emit.py:97` | ggml 建图映射 | 以后再说 |
| `src/by1tens.py:304` | 张量契约（按 `mtype in TOKEN_MIXER` 走） | 契约要能描述"两条支路的张量" |
| `src/by1full.py` | `MECH_FIELDS` + 家族分支（T26/T27 加的那两处） | 加 `falcon_h1` 家族分支 |

**最重要的一个发现**：四个后端**都已经有"不认识就拒绝"的表**（`OPS` /
`C_KINDS` / `SUPPORTED_KINDS` ×2）。所以"先让声明与配置这条路通、重后端明确拒绝"
**不需要新造机制**，它本来就是这套东西的默认行为 —— 这显著压低了选项 B 的风险。

## 五、完成定义（如果做）

- [x] **语言的 `Parallel` 组合能写出来，`by1check` 收得下** —— 而且**语言层一行没改**：
      `mech Both : Parallel { a = Mixer; b = Attn }` 本来就解析得过（解析器收裸名），
      不认识的 kind 由各后端明确拒绝；
- [x] **`by1full` 出 `FalconH1Config`**：44 个字段，真权重装载 **579/579 零 missing**
      （`official_configs` 的逐字段判据守住）；
- [x] **真权重双 loader 前向**：**max\|Δ\| = 0.000e+00（逐位）**；**增量 vs 全量**
      官方 **4.959e-05** / by1 **4.768e-05**（判据 2e-3）；
- [x] **未被实现的后端明确拒绝**（实测原文：`机制 'Both' 的类型是 'Parallel'，codegen 还不支持（目前支持：…）`）；
- [x] `by1complete`：GALLERY 前向/反向 + config 逐字段 + **5 条新反例**（少一条支路 /
      指向不存在的机制 / 两条支路同类 / 换成单机制 / 缺一个缩放标量）；
- [x] 产物腿（llama.cpp `FALCON_H1`）：**579/579 张量逐位** + 上游 logits **逐位（188445 值）**；
- [x] 报告 + 支持矩阵。

**T29 一句话结论**：by1 现在能处理的**三根轴**都有完整闭环（描述 → 配置 → 前向 → 产物）：
**逐层混排**（granite 350m/1b）· **同类第二成员**（换图案零改动）· **层内并行**
（Falcon-H1，`Parallel` 组合机制）。而且层内并行这条**语言零改动** ——
省下来的工程量正好来自那条纪律："不认识就精确拒绝"。

### 进度（2026-10-16 当天）

| 步 | 状态 | 证据 |
|---|---|---|
| 语言 | **✅ 零改动** | `mech Both : Parallel { a = Mixer; b = Attn }` 直接可用；`pattern = 36 * Both`；`by1check` **0 错 0 警** |
| 配置 | **✅** | 44 个字段 · meta 建模 `FalconH1ForCausalLM` **521.4 M**（与真权重**完全一致**）· 真权重装载 579/579 |
| 判据 | **✅** | 双 loader 前向**逐位（0.000e+00）** · 增量 == 全量 4.959e-05 / 4.768e-05 |
| 夹具 | **✅** | `by1complete` 全绿（含 5 条新反例） |
| 产物腿 | **✅** | `--arch falcon_h1 --n-blocks 36`：**579 个张量**与上游**逐位相同**（补了"把九个缩放折进权重"那条改写 —— 不折的话 **218 个张量值不同**）· 上游未修改的 llama.cpp 加载后 logits **逐位（188445 值）** |

**一个比预期好的结果**：原以为要在 IR 里新增一种算子（选项 A/B 的爆炸半径），
实际**语言层不用动** —— `Parallel` 只是一个 kind 名，`a`/`b` 只是裸名属性，
而"不认识就拒绝"是四个后端本来就有的默认行为。**选项 B 的成本因此接近于零**。

**顺带把一条规则从"某家族"提升为"某一代"**：`dt_bias` 改名那条 T27 就写好了，
但 scope 写死在 `granite_hybrid` 上 —— 探 Falcon-H1 时，579 个张量里有 **36 个
（每层的 `mamba.dt_bias`）映射不上**。上游的证据很清楚：
`GraniteHybridModel(Mamba2Model)` 与 `FalconH1Model(Mamba2Model)` **继承同一套**
`filter_tensors`。所以 by1 这边的规则也改成按 **`MAMBA2_ARCHES`** 划。
改完：579 / 370 / 466 **三个族全部 100% 命中**（旧的没有回归）。

## 六、边界

- **只做 0.5B 这一个形态**；`mamba_d_ssm ≠ expand × hidden`（1536 vs 2048）
  说明这个家族的 SSM 宽度与主干**解耦**，声明里要按实测写，不许"按公式推"；
- `attn_layer_indices` 在 raw config 与 `to_dict()` 里都是 `None`，
  且**建模源码里一次都没出现**（`src.count(...) == 0`）—— 那是个没用的字段，
  不要把它当混合图案的来源；
- 这一轮**不动** `pattern` 的语义：`Parallel` 是**新增**一种机制组合，
  不是把 `+` 的意思改掉（改语义会让 49 份声明全部需要重读）。
