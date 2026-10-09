# `pattern` 属性方案（设计草案 v0）

> 目的：把 `pattern` 的元素从「机制名」升级为「机制实例」，让调度能排**属性**而不只是排机制。
> 起因：写 `deepseek-v4.1-flash.by1` 时，20 层全是同一个 `CSA2`，变的只是层上的 `mode ∈ {Full, Reindex, Reuse}`——现有 pattern 表达不了。
> 状态：**待审**。定稿前不要写后面 4 个模型。
>
> **审查意见见 [`pattern-design-review.md`](pattern-design-review.md)。**
> 一句话结论：立论对，但它关掉的是一个缺口不是十六个 ——
> §5 的两个原语（`prev` / `shared_from`）代码里没有，
> §7「六个全部通过」里有三个没有判卷人，§6④ 的 `KDA : Linear` 和代码冲突。
> **照做之前先读那份审查。**

---

## 0. 问题

现有 `pattern = 12 × [ 3 × GDN + 1 × QSA ]` 的元素是**机制名**，所以它只能回答「第 i 层算什么机制」。

但 1.md 自己说：「真正新的东西是这些机制**怎么组合、按什么比例排列**」。而真实模型里的"按比例排列"有两种：

| 类型 | 例子 | 现有语法 |
|---|---|---|
| **排机制** | Qwen3.8：3 GDN + 1 QSA | ✅ 能表达 |
| **排属性** | DeepSeek：20 层同机制，`mode` 不同 | ❌ 不能表达 |

而且这不是 DeepSeek 独有——你自己的验收表里埋了三处：Gemma 4 的「5:1 + 后缀强制 global」、GPT-OSS 的「交替 SWA/full」、Qwen3.8 的 `l[:] >> MoE`。**六分之四的验收模型都撞这堵墙。**

---

## 1. 核心改动

元素从「机制名」变成「机制实例」。

```ebnf
pattern   = term ( "+" term )*
term      = INTEGER "*" factor | factor
factor    = element | ident | "[" pattern "]"
element   = IDENT ( "(" attr ("," attr)* ")" )?
attr      = IDENT "=" literal
```

| 记号 | 语义 |
|---|---|
| `+` | 顺序拼接 |
| `*` | 重复拼接（`n * X` = X 连续 n 次） |
| `[...]` | 分组，只影响结合与可读性 |
| `ident` | 引用命名子序列（§3） |
| `mech(a = v, ...)` | 机制实例 + 属性覆盖 |

**注意：源码用 ASCII 的 `*`，文档与报错里显示为 `×`。** 现有草案里的 `×`（U+00D7）在机器解析时是个坑。

---

## 2. 属性三分类（本方案最关键的一条）

属性不是平权的一堆键值，它分三类，**因为它们对下游的影响完全不同**：

| 类 | 例子 | 影响 |
|---|---|---|
| **行为属性** | `mode`, `sink`, `window`, `gate` | 只改计算方式，**不改张量形状与张量集合** |
| **结构属性** | `head_dim`, `kv_heads`, `kv_tie` | **改张量形状与张量集合** |
| **调度属性** | `kv_source`, `index_source` | 指向**另一个实例**的状态 |

由此推出三条硬规则：

1. **`mech` 声明的是「默认值 + 不变式」，不是唯一真相。** 元素上的属性覆盖默认值。
2. **行为属性可自由覆盖。** 一个 `GQA` 机制，第 0 层 `window = 128`、第 1 层 `window = none`，仍是同一个 `mech`。
3. **结构属性被覆盖时，该实例必须重新通过形状校验，且 `tensors` 必须能按实例展开。**

第 3 条是连带代价：`tensors` 不能再写成 `layer(i) { ... }` 一刀切（那份 Qwen3.8 草案里 `in_proj_qkvz` 会被发给全部 48 层，包括 12 个不该有的 QSA 层）。张量契约的作用域必须跟着 `pattern` 的实例走。

### 判族规则（连带结论）

一旦承认"属性可以是标量参数"，就会立刻撞上一个问题：**`SWA` 到底是不是一个独立的机制？** 如果 `GQA(window = 128)` 和 `GQA(window = none)` 是同一个机制的两个实例，那 `SWA` 这个 `mech` 就不该存在。

于是必须有一条**判族规则**，否则族表会随模型数量线性膨胀——这正是 llama.cpp 现在的问题。

建议的规则，直接从 1.md 自己的主张（"状态比计算更重要"）推出来：

> **两个变体同族 ⟺ 它们的状态契约相同**（同样的状态种类、生命周期、可复用性）。
> 差异只是标量参数或 mask 语义的，同族；一旦多出一份状态，就分族。

按这条规则重切：

| 变体 | 状态契约 | 族 |
|---|---|---|
| MHA / GQA / MQA / SWA / sink | KV cache，无附加状态 | **Attention**（同族，window/sink 是属性） |
| MLA | 压缩潜向量 KV | **Latent Attention**（分族——状态不同构） |
| QSA / DSA / MSA / CSA2 | KV cache **+ index cache** | **Sparse Attention**（分族——多一份状态） |
| Mamba2 / GDN | 固定递归状态 | **Linear / Recurrent** |

这条规则有两个好处：**它把族数往下压**（`SWA` 不再是族，`MLA` 与 `GQA` 分开），而且它给出的是**可判定的**边界——不需要争论"像不像"，只需要看状态契约。

代价是：我上一轮给你草拟的那张 11 族表**作废了**，它按"功能像不像"切，切得太松。族表应该由这条规则重新推一遍，而且**数量是推导结果，不是事先定的**。

---

## 3. 命名子序列

没有 `let`，Gemma 4 的 pattern 会变成一行读不懂的东西。

```
schedule {
  block     = 5 * GQA(window = 1024, head_dim = 256) + 1 * GQA(head_dim = 512)
  full_tail = 2 * GQA(head_dim = 512)
}
```

命名子序列**不是**纯语法糖——K3 的 AttnRes 推翻了这一点（见 §6 ④）：

- 如果 `block` 只作文本宏，展开后**块的边界就丢了**；
- 而 AttnRes 要求「每层可以直接检索**前序所有 block** 的输出」——它必须知道哪些层构成一个块。

所以命名子序列要**保留身份**：展开成层表时，块边界（`block[0] = l[0..11]`）必须留在结果里，
供 `state` / `interop` / residual 引用。

这也给出了三个模型共用的一个概念：**block**。

| 模型 | 块的用途 |
|---|---|
| Kimi K3 | AttnRes 的检索单元 |
| DeepSeek V4.1 | `Full / Reindex / Reuse` 的段 |
| Gemma 4 | 5:1 块 + 后缀 |

---

## 4. 层数由 pattern 推导，stack 不再写死

```
stack encoder { pattern = ... }        # 长度 = 展开结果
stack encoder[20] { pattern = ... }    # 可选：断言，不匹配就报错
```

这消除双真相源——你那份草案里 `stack l[48]` 和 `pattern = 12 * [...]` 同时声明了 48。同理，`params` 应该由 hparams + 机制维度**算出来**，声明的数字降级为断言。

---

## 5. 选择器语言统一

`>>` / `::` / `state` / `position` / `tensors` 五个块都要能寻址层，所以**只定义一套选择器**：

| 选择器 | 含义 |
|---|---|
| `e[:]` | 全部 |
| `e[0..1]` | 前缀 |
| `e[last]` | 末层 |
| `e[step 4]` | 周期 |
| `e[where mode = reuse]` | 按属性 |

再加一个**唯一的跨层引用原语**：

```
prev(PREDICATE)     # 沿层序向前，最近一个满足条件的实例
```

这一个原语同时解决两个缺口：

```
// DeepSeek：Reuse 层复用最近一个 Reindex 层的 KV
d[where mode = reuse].CSA2.kv_cache : shared_from(prev(CSA2 where mode = reindex))

// DeepSeek：分层索引器的候选集来自前一个稀疏层
d[where mode = reindex].CSA2.index  : inherit(candidates_from = prev(Sparse))
```

---

## 6. 六个验收模型的 pattern

### ① GPT-OSS-120B — 排属性，不排机制

```
mech GQA : Attention {
  sink = learned(per_head)          # 行为属性，全局默认
}

stack main {
  pattern = 18 * [ GQA(window = 128) + GQA(window = none) ]   # 层数待核
}
```

**注意这里没有 `SWA` 这个机制。** `SWA` 和 full attention 是同一个机制的 `window` 属性。你的验收表写的是"交替 SWA(128)/full"——如果语言里真的需要两个不同的 `mech`，那说明属性切分错了。

### ② Gemma 4 31B — 结构属性 + 后缀例外

```
schedule {
  block     = 5 * GQA(window = 1024, head_dim = 256) + 1 * GQA(head_dim = 512)
  full_tail = 2 * GQA(head_dim = 512)                          # 后缀强制 global
}

mech GQA : Attention {
  kv_tie = true                     # K = V，结构属性：无 v_proj
}

stack main {
  pattern = 6 * block + full_tail
}

state {
  GQA[where head_dim = 512].kv_cache : shared_from(prev(GQA))  # 复用什么粒度待核
}
```

这一个模型同时压测：**结构属性覆盖**（两种 head_dim 并存 → 两种张量形状）、**后缀例外**（`+ full_tail`）、**结构属性影响张量集合**（`kv_tie` 少一个投影）。

### ③ Qwen3.8-Flash-Next — 向后兼容

```
stack l {
  pattern = 12 * [ 3 * GDN + 1 * QSA ]
  l[:] >> MoE
  l[:] :: GatedResidual
}
```

**一个字都不用改。** 新方案是现有语法的超集，这是必须守住的底线。

### ④ Kimi K3 — 已核实，而且是本方案最强的压力测试

公开规格：93 层（其中稠密层 1 层）、**69 KDA + 24 Gated MLA**、896 专家激活 16、2 个共享专家、
96 个注意力头、hidden 7168、词表 160K、**完全取消位置编码**（靠 KDA 衰减隐式建模位置）、
MoonViT-V2 视觉编码器（401M）。

**「8×12 分组」是误记。** 真实的是 AttnRes 的 block 划分：

> 93 层划分为 9 个 block（7×12 层 + 1×9 层 + embedding），仅在 block 间做跨层检索。

**「深度维度注意力」= AttnRes**：每层可直接检索**前序所有 block** 的输出，突破残差稀释。

```
mech KDA : Linear { ... }        # Kimi Delta Attention
mech MLA : Latent { ... }        # Gated MLA

schedule {
  # 每 3 层 KDA 后接 1 层 MLA，但 93 不能被 4 整除 —— 余数必须能表达
  body = 21 * [ 3 * KDA + 1 * MLA ] + 3 * [ 2 * KDA + 1 * MLA ]
}

stack main {
  pattern = 1 * DenseFFN + body     # 前缀例外：那 1 层稠密层
  block(12)                          # AttnRes 的检索单元，见 §3
}
```

`21×3 + 3×2 = 69` KDA、`21 + 3 = 24` MLA、`84 + 9 = 93` 层——层数与族计数同时对得上。
**但这是我反推的，不是公布的划分方式，需要核对。**

**结论：本方案存活。** 三个原因：

1. K3 的注意力仍是 `3 : 1` 周期，`pattern` 的代数直接覆盖；
2. 它多出来的是**余数**（93 不是 4 的倍数）——而 `+` 天生能表达，这正是 DeepSeek
   那三个 mode 段用到的同一个东西；
3. AttnRes 是**层间检索边**，属于 `interop` / residual 轴，**不在 pattern 轴上**——
   与 §8 预先划的边界一致。

代价只有一条：**§3 的命名子序列必须保留块身份。**

顺带：K3 用的是 **Gated MLA**，于是判族规则又过了一次测试——MLA 的压缩潜向量 KV 与
KDA 的固定递归状态不同构，两者分族，而它们能在同一个模型里混排。

### ⑤ DeepSeek V4.1-Flash — 本方案的直接受益者

```
mech CSA2 : Sparse {
  indexer = hierarchical(depth = ?)
}

stack encoder {
  pattern = 20 * GQA(mode = full)
}

stack decoder {
  pattern =  6 * CSA2(mode = full)
          +  8 * CSA2(mode = reindex)       # 各 mode 层数待核
          +  6 * CSA2(mode = reuse)
}
```

**这就是全部。** 上一版草案需要新造 `attr mode on d[0..5]` 才能表达的东西，现在只是三个连续的 run。

原因很简单：`Full / Reindex / Reuse` 本来就是**连续的段**，不是散落的例外。而能表达"段"的最小语法就是 `+` 和 `*`——你已经有了。

### ⑥ Nemotron 3（建议补）— 新族，不是新语法

```
mech Mamba2 : SSM { ... }

stack main {
  pattern = 6 * [ 3 * Mamba2 + 1 * GQA ]      # 混排比例待核
}
```

SSM 是**族表第 6 族**的补充，pattern 层面无新需求。但它会压测 `state`——Mamba-2 的递归状态和 KV 完全不同构。

---

## 7. 覆盖性检验

| 验收模型 | 需要的新能力 | 本方案 |
|---|---|---|
| GPT-OSS-120B | 行为属性逐层不同 | ✅ |
| Gemma 4 | 结构属性逐层不同 + 后缀例外 | ✅（张量契约需要跟着改，见 §2 规则 3） |
| Qwen3.8 | 无 | ✅ 向后兼容 |
| Kimi K3 | 非整除的 3:1 周期 + AttnRes 块 | ✅ pattern 层通过；块身份需在 §3 补 |
| DeepSeek V4.1 | 行为属性按段排列 | ✅ |
| Nemotron 3 | 新增 SSM 族 | ✅ pattern 层无需求 |

**六个全部通过。** 唯一遗留是 §3 的命名子序列要保留块身份——那是实现细节，不是方案缺口。

---

## 8. 本方案明确不解决的问题

写清楚边界，避免下次误以为它能包打天下：

1. **跨栈接线**（DeepSeek 的 `encoder.last.hidden → decoder.global_kv`）。这是 `interop` 的事，不是 `pattern` 的事。而且它会**引入可训练权重**（那个投影矩阵），所以 `state` 块和 `tensors` 块之间必须定义边界——现在没有。
2. **`state` 的持久化语义**（SWA Bounded Replay：不落盘 + miss 时重放窗口）。这动摇了"状态是全模型属性、无法事后补救"这句前提。
3. **深度维度注意力**（K3）。如果注意力要跨层作用，`stack` 就不是一条链了，是图。
4. **张量契约的实例化**（§2 规则 3 的连带代价）。这是下一个必须单独定的东西。
5. **`optimizer` 与非目标的冲突**。

---

## 9. 待你拍板

| # | 问题 | 我的倾向 |
|---|---|---|
| Q1 | `SWA` 要不要作为独立机制存在？ | **不要。** 它是 `GQA(window = n)`。留两个 `mech` 就等于承认属性切分失败 |
| Q2 | 结构属性覆盖后，`tensors` 按实例展开还是按 (机制, 结构属性) 去重？ | 去重。12 个 QSA 层应共享一份契约 |
| Q3 | ~~K3 的「8×12 分组」指什么？~~ | **已核实**：是 AttnRes 的 block 划分（7×12 + 1×9），不是注意力分组，见 §6 ④ |
| Q4 | 长度：只推导，还是允许断言？ | 允许 `stack e[20]` 作断言 |
| Q5 | `×` 与 `*` | 源码 `*`，显示 `×` |

---

## 10. 落地顺序

1. 你答 Q1–Q5
2. 按本方案重写 `deepseek-v4.1-flash.by1`（去掉全部 16 个 ⚠️ 中与 pattern 相关的那些）
3. 用 Gemma 4 压测**结构属性**这条路径——它是唯一能验证 §2 规则 3 的模型
4. 张量契约的实例化单独定一版
5. 回头改 1.md 的能力清单第 2 条（调度）
