# T27：第三个机制族 —— **混合栈**（Attention 与 SSM 在同一个模型里）

> 这份文档是**计划**，也是记录：写下的数字都是**实测**的，没跑过的都标着"未验"。
> 格式沿用 [T26 那份](t26-plan.md)：判卷人先立，缺什么写清楚，四件事排出来。

## 一、为什么是这件事（不是"再加一个模型"）

T26 让 by1 有了第二个机制族（mamba / SSM），但它是**早退分支**：
那条路要求"**主层全是 SSM**"（`_compile_mamba` 里那句
`kinds == {'SSM'}`），因为 Mamba1 的 mixer 就是整层、没有独立的 FFN。

真实世界的模型不是这么切的。**混合栈**——同一份声明里既有权重注意力层、
又有状态空间层——是 2025 年以后的主流做法之一（Granite 4、Zamba2、Nemotron-H、
Qwen3-Next 都是）。要支持这一类，`by1full` 必须从
"**一个家族一种机制**"走到"**逐层类型表**"（`gpt_oss` 那条分支里已经有
`layer_types` 的雏形，但它只服务一个家族）。

一句话：**这一步把 by1 从"支持几个模型"推向"支持一类模型"。**

## 二、选型与判卷人（2026-10-16 实测）

选 **`ibm-granite/granite-4.0-h-350m`**（0.64 GB）。理由首先是**体积** ——
它要在 CPU 上做双 loader 对拍，两个 loader 都得装得下：

| 候选 | 体积 | 为什么不是它 |
|---|---|---|
| **granite-4.0-h-350m** | **0.64 GB** | ✅ 选它 |
| Zyphra/Zamba2-1.2B | 4.53 GB | fp32 约 4.8 GB／loader，双 loader 太挤 |
| nvidia/Nemotron-H-4B-Base-8K | 8.38 GB | CPU 上 fp32 放不下 |
| state-spaces/mamba2-130m | 0.24 GB | 只有 SSM，**不是混合栈**（没有注意力层） |

**判卷人两侧都在**（都实测过，不是"应该可以"）：

| 侧 | 证据 |
|---|---|
| transformers 5.15.1 | `AutoModelForCausalLM` → **`GraniteMoeHybridForCausalLM`** · **340.3 M 参数** · 前向 logits `(1, 5, 100352)`、`max\|.\| = 53.653` · **顶层有导出**（`hasattr(T, 'GraniteMoeHybridForCausalLM')` 为真，所以不需要 mamba 那次的 `_tclass` 回退） |
| llama.cpp | `MODEL_ARCH.GRANITE_HYBRID` 在 arch 表里 · `src/models/granite-hybrid.cpp` 在 · `conversion/granite.py` 在（这一族的产物线**具备条件**，能不能真跑通要实测） |

## 三、结构证据（实测，不是读文档猜的）

**逐层类型**（32 层）：

```
M M M M M M M M M M A M M A M M M A M M M M M M M M M A M M M M
```

- 权重注意力层在 **0 基 `[10, 13, 17, 27]`**（4 层），其余 **28 层是 mamba**；
- 类型集合只有 `{'mamba', 'attention'}` —— 没有第三类。

**每层张量数**：mamba 层 **12 个** · attention 层 **8 个**。

| mamba 层（12） | 形状 | 说明 |
|---|---|---|
| `input_layernorm` / `post_attention_layernorm` | 768 | 两层 norm 都在（混合栈的公共骨架） |
| `mamba.in_proj` | 3376 × 768 | 3376 = z(1536) + x(1536) + dt(48) + B,C(2×128) |
| `mamba.conv1d.weight` | 1792 × 1 × 4 | 1792 = 1536 + 2×128；`d_conv = 4` |
| `mamba.conv1d.bias` | 1792 | —— |
| `mamba.A_log` / `mamba.D` / `mamba.dt_bias` | 48 | **= n_heads**（不是 d_state） |
| `mamba.norm` | 1536 | mixer 内部 gated norm |
| `mamba.out_proj` | 768 × 1536 | —— |
| `shared_mlp.input_linear` | 4096 × 768 | **4096 = 2 × 2048**：gate/up 融合 |
| `shared_mlp.output_linear` | 768 × 2048 | —— |

| attention 层（8） | 形状 |
|---|---|
| `self_attn.q_proj` / `o_proj` | 768 × 768 |
| `self_attn.k_proj` / `v_proj` | 256 × 768 |
| 两层 norm + 同一个 `shared_mlp` | 见上 |

**这一族的"另一套东西"**（都要如实建模，不能只说能描述）：

1. **Mamba2 参数化**（与 T26 的 Mamba1 不同）：`n_heads = 48`、`d_head = 32`、
   `d_state = 128`、`n_groups = 1`、`chunk_size = 256`。**A_log/D/dt_bias 的长度是
   heads 不是 state** —— 和 Mamba1 的 `(d_inner, state)` 不是一回事。
   by1 侧要有 `SSM2` 这一类机制字段（T26 加的 `SSM` 是 Mamba1 那一套）。
2. **NoPE**：`position_embedding_type = "nope"` —— 这个模型的注意力层**不加位置**
   （by1 的 `position { default = none }` 表达得出，但要显式声明、并让配置生成器照做）。
3. **μP 缩放**：`embedding_multiplier = 12`、`residual_multiplier = 0.246`、
   `logits_scaling = 3`、`attention_multiplier = 0.015625`（= 1/64）。
   这些是**改数值**的字段：少一个，形状全对、算的是另一个模型
   （T26 踩过同型的坑：`A_log → -exp(A_log)`）。
4. **共享 MLP**：`num_local_experts = 0` —— 它是 dense，不是 MoE；
   `shared_mlp` 每层都有（连注意力层也有）。

## 四、四件事（判据沿用 P0/T26 那一套，不新造尺子）

1. **`by1full` 加 `granitemoehybrid` 家族 + `SSM2` 字段表 + 逐层类型表**
   （这一步是能力跃迁：不再"一个家族一种机制"）；
2. **complete 声明**（计划名 `models/complete-granite-4.0-h-350m`
   加上 `.by1` 后缀 —— 还没建，所以这里不写成完整文件名），过 `by1check`
   与 `by1complete`（含 **config 逐字段** —— μP 那四个标量必须逐个对上）；
3. **真权重双 loader 前向**（判据 2e-3，争取 0）+ **混合状态的"增量 == 全量"**：
   同一个模型里，注意力层是 KV cache、mamba 层是定长递归状态 ——
   这是**新的一本账**（T14 是纯 KV、T26 是纯递归，这里是两者混合）；
4. **标准产物**：`by1llama build-gguf --arch granite_hybrid` → 上游
   `llama-eval-callback` logits **逐位**。

## 五、完成定义

- [x] complete 声明过 `by1check` + `by1complete`（含 config 逐字段）；
- [x] 真权重双 loader：零 missing/unexpected + 前向 max|Δ| **0.000e+00**；
- [x] 混合状态的增量 == 全量（KV 与递归两本账在同一个模型里，**分开报**：9.918e-05）；
- [x] 上游加载 + logits 逐位：**402/402 张量逐位 · 上游 logits 逐位（89474 值）**；
- [ ] 报告进 `docs/`，更新[支持矩阵](support-matrix.md)与[精度预算](precision-budget.md)。

**T27 一句话结论**：by1 现在能描述、能算、能出标准产物、能被上游加载的，
是**一整类"逐层机制混排"的模型**了 —— 形状来自逐层类型表而不是家族名，
判据两侧齐全（transformers + llama.cpp），而且**同一套产线**同时覆盖
纯注意力、纯递归、混合三种主干。**从"支持几个模型"到"支持一类模型"。**

### 进度（2026-10-16 当天）

| 步 | 状态 | 证据 |
|---|---|---|
| 1 | **✅** | `by1full`：`FAMILIES` 加 `granitemoehybrid` · `MECH_FIELDS` 加 **`SSM2`**（Mamba2 参数化：`n_heads × d_head`，与 T26 的 `SSM` 并存）· 新增 `_compile_moe_hybrid` —— **第一次按逐层类型表建配置**（`layer_types` 由逐层机制映射而来，不再"一个家族一种机制"） |
| 2 | **✅** | 声明入库（`models/` 下 `complete-granite-4.0-h-350m`，`models.tsv` 加了行，`by1fetch` 抓回 refs）：`by1check` **0 错 0 警** · `compile_config` 出 **40 个字段** · meta 建模 `GraniteMoeHybridForCausalLM` **340.3 M**（与真权重同）· 真权重装载 **370/370 零 missing** · `by1complete` 的 config 逐字段已接上（**PASS**） |
| 3 | **✅（前向那半）** | `by1shadow --smoke-refs` → **370 个真名 · 双 loader 零 missing/unexpected · 前向 max\|Δ\| = 0.000e+00（逐位）** |
| 3' | **✅** | `by1shadow --smoke-state-hybrid`（新加的第三条判据）：**一个 `DynamicCache` 里 28 个递归层 + 4 个注意力层** · 官方 **9.918e-05** · by1 **9.918e-05**（判据 2e-3，余量约 20×）· 逐位置曲线 **不随序列增长**（1e-5 ~ 1e-4 平着走）—— 比 T26 的纯递归（1.497e-03）还小 15 倍，所以**别把纯递归的余量外推给别的递归形态**（已登记进[精度预算](precision-budget.md)） |
| 4 | **✅** | `by1llama build-gguf --arch granite_hybrid --n-blocks 32`：**402 个张量**（比 HF 的 370 多 32 —— 融合的 gate+up 要拆开），与上游产物**逐位相同（402/402）**；上游未修改的 llama.cpp 加载后 logits **逐位相同（89474 值）**。路上补了 6 条转换期改写（3 条在钩子里、Mamba2 的 3 条形状规则），记在 [convert-diff](convert-diff.md) |
| — | **✅** | 已进 `by1complete` 的 **GALLERY**（`tiny`/`config_of`/`reference` 三条分支都写了）：那一步现在跑 **8 个完整声明**的前向/反向 + config 逐字段 + **13 条必红反例**（含 T27 新加的 5 条）→ `[PASS] 完整模型：全部通过` |

**夹具这一步踩的三个坑（都记进代码注释了）**：

1. **`tiny()` 要"按类型取样"，不能按层号截断**：通用那段只留前 2 层，
   而这个模型前 10 层全是 Mamba2 —— 截断后连一个注意力层都不剩，
   于是触发"两种机制都要出现"。所以混合栈那一段从**截断前的层表**里
   各取 2 个递归层 + 1 个注意力层。（第一次就是按截断后的表取，取了个空。）
2. **从原始层表取回来的层，属性还是大尺寸**：注意力层带着 12 × 64（= 768），
   而夹具的 `d_model` 是 32 —— 采样后必须**显式缩小** `q/kv/head_dim`
   与 `n_heads/d_head/d_state/chunk_size`，两处（`layer_seq` 与 `mech_attrs`）都要改。
3. **词表缩了，`pad_token_id` 也要缩**：声明里是 100256，而夹具词表只有 64，
   PyTorch 直接报 `Padding_idx must be within num_embeddings`。

**这一轮踩过的两个坑（都记进代码注释了）**：

1. **`layer_types` 在 transformers 里被规范化**：发布方 `config.json` 写的是
   `'mamba'`，而 `to_dict()` 给的是 **`'linear_attention'`** —— 逐字段对拍要比的是
   规范化后的那一份（差一个字符串就红）。
2. **`time_step_limit` 故意不声明**：它是 dataclass 的 `tuple[float, float]`，
   而 `to_dict()` 给 **list** —— 声明进去要同时满足"构造时 tuple、比对时等于 list"，
   而且 `inf` 在生成的 Python 里不是个名字。实测类默认值就是 checkpoint 里的
   `(0.0, inf)`，所以不碰它最干净。

## 六、边界（先说清楚，免得后面含糊）

- **只做 350m 这一个形态**：同族的 1B/更大版本不因此自动成立；
- llama.cpp 的 `granite-hybrid.cpp` 是 C++ 实现、`conversion/granite.py` 是转换器 ——
  **它们存在不等于这个 350m 形态能过**，要实测；
- 混合栈的"增量 == 全量"里，注意力层与递归层的误差**不共享同一条预算**
  （T26 已经量过：递归那条余量只有 1.3 倍），所以这一条要**分开报**。
