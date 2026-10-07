# Block 0 · 中间表示（IR）规格

> **状态**：设计稿。下一步是让 `by1codegen` 吐这个，而不是直接吐 PyTorch 模块。
>
> **为什么先做这个**：现在 `compile_spec` 的输出**本身就是 PyTorch 形状的**——
> `{"kind": "attn", "q": 4, ...}` 直接喂给 `RUNTIME` 里的 `Attention` 类。
> 再加 MoE、GDN、多栈，它会往 PyTorch 的形状上长；一旦要换后端（Block 3），全废。

---

## 1. 目标

一句判据：

> **同一份 IR，两个后端（PyTorch / 自建执行器）跑出同一个前向。**

只要 IR 里出现任何一个 PyTorch 概念（`nn.Module`、张量形状、in-place），这个判据就不成立。

---

## 2. 结构

```
IR {
  vocab      : int
  ctx        : int
  d_model    : int
  globals    : [ Op ]        # 词嵌入、最终 norm、输出头 —— 整个模型一次
  layers     : [ Layer ]
}

Layer {
  index : int
  attrs : { 名 -> 值 }        # 该层解析后的全部属性（window / q / head_dim / kv_tie …）
  ops   : [ Op ]             # 按顺序执行
  state : [ StateDecl ]      # 这层持有的状态
}

Op {
  mech    : str               # 机制名，如 "GQA" / "MoE" / "RMSNorm"
  kind    : str               # 机制类型，如 "Attention" / "MoE" / "FFN"
  attrs   : { 名 -> 值 }
  inputs  : [ ValueRef ]
  outputs : [ ValueRef ]
}

StateDecl {
  kind       : "kv_cache" | "recurrent" | "index_cache" | "none"
  bounded_by : int | null     # 来自 .by1 的 bounded_by
  dtype      : str
  reuse      : "none" | "prefix" | "layer:<n>"
}
```

### ValueRef —— 唯一的"数据流"概念

```
"hidden"              该层的主输入（残差流）
"op:<i>.out"          本层第 i 个算子的输出
"global:<name>"       全局张量（词嵌入表、输出头权重）
":<const>"            常量
```

**没有"张量形状"这个概念。** 形状由后端从 `attrs + d_model` 推——因为形状是后端的知识，不是架构的。

---

## 3. 现有四个模型的映射（Block 0 的验收）

### llama-shaped / hello（最简单）

```
layers[i].ops = [
  Op{ mech:"RMSNorm", inputs:["hidden"],       outputs:["op:0.out"] },
  Op{ mech:"GQA",     inputs:["op:0.out"],     outputs:["op:1.out"] },
  Op{ mech:"Add",     inputs:["hidden","op:1.out"], outputs:["op:2.out"] },
  Op{ mech:"RMSNorm", inputs:["op:2.out"],     outputs:["op:3.out"] },
  Op{ mech:"FFN",     inputs:["op:3.out"],     outputs:["op:4.out"] },
  Op{ mech:"Add",     inputs:["op:2.out","op:4.out"], outputs:["hidden"] },
]
state = [ StateDecl{kind:"kv_cache", bounded_by:null} ]
```

### GPT-OSS（滑窗 + 每头 sink + MoE）

同上，差别全在 `attrs` 和一处 `state`：

```
Op{ mech:"GQA", attrs:{ q:64, kv:8, head_dim:64, window:128, sink:"per_head" } }
state = [ StateDecl{ kind:"kv_cache", bounded_by:128-1 } ]     # 全量层为 null
```

**滑窗层与全量层共用同一个 `layers` 结构，只是 `attrs.window` 和 `state.bounded_by` 不同。** 这正是等价类给出的东西。

### Laguna（两级头的滑窗 + 稠密/稀疏混排）

```
layers[0].ops  += Op{ mech:"FFN",  attrs:{hidden:8192} }          # 第 0 层稠密
layers[1..].ops += Op{ mech:"MoE", attrs:{experts:256, top_k:8, shared:1, hidden:512} }
```

**"稠密 vs 稀疏"在这里第一次变成一个真正的 IR 差异**——不是命名差异。这是 Block 1.1 的核心考点。

### Gemma 4（两种 head_dim + K=V）

```
Op{ mech:"GQA", attrs:{ q:32, kv:16, head_dim:256, kv_tie:false } }
Op{ mech:"GQA", attrs:{ q:32, kv:4,  head_dim:512, kv_tie:true  } }
```

`kv_tie` 让后端不建 `v_proj`。**IR 里没有"少一个张量"这回事，只有属性**——张量集由后端推。

---

## 4. Block 1.1（MoE）在 IR 里长什么样

这是 Block 0 之后的第一个机制。它逼出两件现在没有的东西：

**① 一个算子可以有多个输入**

```
Op{ mech:"MoE",
    inputs : ["op:3.out"],
    outputs: ["op:4.out"],
    attrs  : { experts:256, top_k:8, shared:1, hidden:512,
               score_bias:true, routed_scale:2.5 } }
```

路由、共享专家、score bias 全是 `attrs`——**它们是同一个机制的参数，不是三种机制**。这一条如果不成立，就说明"参数化机制族"在这里已经破了。

**② 状态可以没有**

MoE 层的 `state = []`。IR 必须允许空，而现在的 `state_src` 是"按机制名查表"，查不到就当没有——那是巧合，不是设计。

---

## 5. 验收（可证伪）

Block 0 + 1.1 做完，必须**同时**满足：

| # | 判据 | 怎么测 |
|---|---|---|
| 1 | 四个现有模型降级成同一份 IR，无特例分支 | IR 里没有 `if kind == "..."` 之外的后端知识 |
| 2 | **PyTorch 后端的前向仍然对到 ≤1e-6** | `by1diff llama-shaped.by1` |
| 3 | MoE 前向对上参考实现 | 新增：微型 `GptOssForCausalLM` 对拍 |
| 4 | 稠密/稀疏混排表达得出来 | Laguna 的 Layer 0 与 1 结构不同但用同一个 `MoE`/`FFN` 算子 |

**第 2 条是护栏。** 重构过程中它一旦变红，立刻回退——不能为了 IR 的漂亮牺牲唯一已验证的前向。

---

## 6. 实现步骤（按顺序，每步都能回退）

1. 定义 IR 的 Python 数据结构（纯 dict，几十行）
2. `compile_spec` 改为吐 IR；**同时保留旧路径**，让 `by1diff` 跑在旧路径上作对照
3. `RUNTIME` 改为消费 IR；`by1diff` 切到新路径 → **必须仍然 PASS**
4. 加 `MoE` 算子到 IR + RUNTIME
5. 新增 `by1diff` 的 MoE 判卷人（微型 `GptOssForCausalLM`）
6. 让 Laguna 的 `.by1` 走一遍 codegen（它同时有稠密层和稀疏层）

**第 2、3 步之间是本块唯一的风险点。** 所以第 3 步做完之前不碰 MoE。

---

## 7. 明确的非目标

- **不做** Block 3（换后端）。IR 只保证"换得动"，不保证"换了能跑"。
- **不做** GDN。它是 Block 1.3，且它的状态语义会反过来压 IR——但那要在 IR 站稳之后。
- **不做** 量化布局。它是导出参数（Block 4）。

---

## 8. 进度（随做随更新）

| | 状态 | 证据 |
|---|---|---|
| **Block 0** 显式 IR | ✅ | `by1diff llama-shaped` = **4.470e-07**，IR 重构前后**同一个数** |
| **Block 1.1** MoE | ✅ | `by1diff mixtral-shaped` = **4.172e-07** |
| **Block 1.2** 注意力机制 + YaRN | ✅ | `by1diff gpt-oss-shaped` = **2.384e-07** |
| Block 1.3 GDN | 🟡 | **机制本身 ✅**：对 `Qwen3NextGatedDeltaNet` 精确到 **2.384e-07**，去掉 delta 项偏差 3.12e-02（测试非空过）。**整模型 ❌ 见下** |
| Block 2 状态与内存 | ⬜ | |
| Block 3a 自建执行器 | ⬜ | 用来检验 IR 够不够，比 3b 便宜一个数量级 |
| Block 3b 目标 llama.cpp | ⬜ | **1.md 说的"上千行"就是这里** |
| Block 4 量化布局 | ⬜ | 导出参数：`--quant mxfp4` |

### Block 1.2 覆盖到的机制属性

```
Attention  sink · kv_tie · head_gate(per_head/per_element) · rope_scale · qk_norm · attn_bias
           YaRN(factor / original / beta_fast / beta_slow / truncate)
FFN / MoE  swiglu_limit · alpha · act(silu|gptoss) · routing(softmax_topk|topk_softmax)
           expert_bias · router_bias · score_bias · routed_scale · shared
```

### 三条前向对拍（护栏）

**任何重构之后这三条必须全绿，否则回退。**

```
python by1diff.py llama-shaped.by1        # 4.470e-07
python by1diff.py mixtral-shaped.by1      # 4.172e-07
python by1diff.py gpt-oss-shaped.by1      # 2.384e-07
```

### 已知待办

0. **⚑ 卡住的地方：`qwen3-next-shaped.by1` 的整模型对拍不通过。**
   - 权重现在 **70/70 全部搬到**（加了覆盖面护栏之后），名字与形状全对
   - 但前向**输出几乎为零**，相对差恰好 1.000；而且补上共享专家映射后**数值一模一样**，
     说明共享专家那条路径根本没参与计算
   - 已经实现但**未验证**：`partial`（部分 RoPE）、`q_gate`（查询门控）、`shared_gate`
   - **下一步**：查 `own` 里到底有没有 `sw1/sw3/sw2/shared_gate`（如果 MoE 的 `shared` 属性没解析成 1，
     这些模块根本不会建，而覆盖面检查会**平凡通过**）。然后抓每层 hidden state 看第几层塌。
   - 机制本身已经隔离验证过（`Qwen3NextGatedDeltaNet` 2.384e-07），所以**差异在组装层，不在机制里**。
1. **Block 1.3 GDN 的整模型验证**（见上）。
2. **`swiglu_limit` 在 GPT-OSS 上还没验**（`gpt-oss-shaped` 里开了，但参考的 `limit` 是硬编码 7.0）。
3. **反向没做**：1.md 说"前向与反向必须数值一致"，目前只对了前向。
4. **多栈 / 跨栈投影**没有判卷人（本地没有 DeepSeek 参考实现）。

### 判卷人踩过的坑（写下来免得再踩）

- **standalone 模块的 dispatch 陷阱**：`MixtralSparseMoeBlock` / `GptOssAttention` 单独拿出来用时
  `_implementation=None`，**会静默地不算**。必须用完整模型里的块，或显式设 `_attn_implementation='eager'`。
- **参考实现的默认值散落在各处**：`rope_theta`（Llama 1e4 / Mixtral 1e6）、YaRN 的 `attention_factor`、
  `rope_scaling.rope_theta`（GPT-OSS 150000）、`truncate` 默认 True。
  **每次都没有结构性征兆，全靠外部判卷人。**
- **自己写的探针也会坏**：出现过"两个零在互相比"和"设了一边没设另一边"，两次都差点得出错误结论。
  **加可证伪检查：把被测的那条路径关掉，必须对不上。**

