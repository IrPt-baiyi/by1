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
| Block 1.3 GDN | ✅ | `by1diff qwen3-next-shaped` = **1.788e-07**，对 `Qwen3NextForCausalLM`。3 层线性 + 1 层全量的混合栈，**第一个验证过端到端的带状态模型** |
| Block 2 状态与内存 | ✅ | `by1mem.py` 从 IR 推逐层内存计划；`qwen3-next-shaped` **4/4 层推算 = 实测**。GPT-OSS-120B @128K = **4.6 GiB**，把滑窗当全量会高估 **2.00×** |
| Block 3a 第二个后端（NumPy） | ✅ | `by1exec.py` 纯 NumPy、不共享一行代码，**四个模型与 PyTorch 后端全一致**（2.5e-07 / 2.3e-07 / 1.0e-07 / 2.5e-06）。IR 的"后端无关"第一次被真正检验 |
| Block 3b 目标 llama.cpp | 🟡 | **执行不了**：这环境没有 C 编译器、没有 llama.cpp、github 不通。但**可验的那一半做完了**：`by1emit.py` 生成的图与 GGUF manifest **双向一致 612=612**，且列出了后端必须实现的算子（含唯一需要自定义 kernel 的 delta 规则） |
| Block 4 量化布局 | ⬜ | 导出参数：`--quant mxfp4` |

### Block 1.2 覆盖到的机制属性

```
Attention  sink · kv_tie · head_gate(per_head/per_element) · rope_scale · qk_norm · attn_bias
           YaRN(factor / original / beta_fast / beta_slow / truncate)
FFN / MoE  swiglu_limit · alpha · act(silu|gptoss) · routing(softmax_topk|topk_softmax)
           expert_bias · router_bias · score_bias · routed_scale · shared
```

### 四条前向对拍（护栏）

**任何重构之后这四条必须全绿，否则回退。**

```
python by1diff.py llama-shaped.by1        # 4.470e-07   稠密
python by1diff.py mixtral-shaped.by1      # 4.172e-07   MoE
python by1diff.py gpt-oss-shaped.by1      # 2.384e-07   sink + YaRN + topk_softmax
python by1diff.py qwen3-next-shaped.by1   # 1.788e-07   线性注意力 + 混合栈（带状态）
```

### 已知待办

1. **反向没做**：1.md 说"前向与反向必须数值一致"，目前只对了前向。
2. **多栈 / 跨栈投影**没有判卷人（本地没有 DeepSeek 参考实现）。
3. **`swiglu_limit` 在 GPT-OSS 上还没验**（参考的 `limit` 是硬编码 7.0）。
4. **`by1mem` 只接了 qwen3_next 的量测路径**，其它族只出计划不核对。
5. **Block 3b（目标 llama.cpp）还没开始** —— 3a 已经证明 IR 装得下第二个后端。

### 内存计划（Block 2 的产出）

```
python by1mem.py qwen3-next-shaped.by1 --seq 4096 --seq2 16384 --measure
python by1mem.py gpt-oss-120b.by1 --seq 131072
```

两种状态，两种增长方式：

| | |
|---|---|
| `kv_cache` | 每 token 线性增长；滑窗时封顶在 `window-1` |
| `recurrent` + `conv_history` | **固定大小，与序列长度无关** —— 线性注意力的全部含义 |

**GDN 层有两个状态**，这是实测逼出来的：delta 规则的矩阵 **加上** 短卷积的历史
（而且是按 `conv_kernel` 步分配，不是 `kernel-1` 步）。只算前者会少 64 个元素/层。


### 判卷人踩过的坑（写下来免得再踩）

- **standalone 模块的 dispatch 陷阱**：`MixtralSparseMoeBlock` / `GptOssAttention` 单独拿出来用时
  `_implementation=None`，**会静默地不算**。必须用完整模型里的块，或显式设 `_attn_implementation='eager'`。
- **参考实现的默认值散落在各处**：`rope_theta`（Llama 1e4 / Mixtral 1e6）、YaRN 的 `attention_factor`、
  `rope_scaling.rope_theta`（GPT-OSS 150000）、`truncate` 默认 True。
  **每次都没有结构性征兆，全靠外部判卷人。**
- **自己写的探针也会坏**：出现过"两个零在互相比"和"设了一边没设另一边"，两次都差点得出错误结论。
  **加可证伪检查：把被测的那条路径关掉，必须对不上。**


---

## 9. 三个后端（Block 3a / 3b 之后）

> **"后端无关"这句话只有一个后端的时候是没被检验过的** —— 它可能只是恰好长得像 PyTorch。

| 后端 | 判卷人 | 说明 |
|---|---|---|
| `by1codegen.py` | transformers 的参考实现 | 生成真 `nn.Module`，可 `.backward()` |
| `by1exec.py` | 与 PyTorch 后端逐位对拍 | 纯 NumPy，没有框架 |
| `by1c.py` | 与 NumPy 后端逐位对拍 | 生成 C → gcc 编译 → 运行 |

**五个模型 × 三个后端：**

| 模型 | C↔NumPy | NumPy↔PyTorch | PyTorch↔参考 |
|---|---|---|---|
| llama-shaped | 2.241e-07 | 2.505e-07 | 4.470e-07 |
| mixtral-shaped | 1.744e-07 | 2.340e-07 | 4.172e-07 |
| gpt-oss-shaped | 7.407e-08 | 1.028e-07 | 2.384e-07 |
| qwen3-next-shaped | 2.990e-06 | 2.513e-06 | 1.788e-07 |
| **mla-shaped** | **3.945e-07** | **4.992e-07** | **0.000e+00** |

反向：四模型梯度全部在阈值内（39/39 · 35/35 · 31/31 · 62/62）。

---

## 10. 真实模型的产物对拍（`.by1` 只是描述，产物才算数）

| 模型 | 产物 | 结果 |
|---|---|---|
| ling-3.0-tiny | HF 张量 | **9283 / 9283** |
| laguna-xs-2.1 | HF 张量 | **30430 / 30430** |
| gpt-oss-120b | GGUF + HF | **687 / 687** 两边 |
| gemma-4-31b | config + GGUF | **36/36** · **530/530** |
| instella-3b | config + HF 张量 | **21/21** · **399/399** |
| step-3.7-flash | HF 张量 | **728 / 753**（缺的 25 个在坏分片里） |

---

## 11. MLA —— 第一族"要新增机制"而不是加属性的东西

判卷人是 **transformers 的 `DeepseekV3Attention`**。它和 Ling 的 MLA 结构同构，
7 个张量形状逐个相同。**别人写的，不是我自己写的参考。**

```
python by1mla.py
  pairing=interleaved  绝对差 0.000e+00   [一致]
  pairing=half         绝对差 6.298e-02   [不一致]
```

MLA 与 GQA 的根本区别：**`k_rot` 是单头的**，RoPE 只作用在解耦出来的
`qk_rope` 维上，然后广播到所有头。cache 也是压缩的（`c_kv + k_rot`）。

### 这一族压出来的三个 bug

1. **C 的 `attention()` 假定 Q/K 头宽 = V 头宽。** MLA 是 48 vs 32，于是 V 被按
   Q/K 的步长读 —— **堆越界，0xC0000005**。加了独立的 `vd` 参数。
2. **`rope_rows` 拿行号当位置。** q 的行号是 `t*nh + h`，位置应该是 `t`（行号除以每位置行数）。
3. **`rmsnorm` 把 eps 写死 1e-5**，而 mla-shaped 是 1e-6 —— 差 1.5e-04。

### 还有一个不报错的

**三个后端的层归一化全都无视 `.by1` 里的 `rms_eps`，各自写死 1e-5。**
所以它们"一致地错"，互相对拍全绿。
实测 llama/mixtral/gpt-oss/qwen3-next 恰好都是 1e-5，**而 gemma / laguna / ling 声明的是 1e-6**。
现在 `Norm` op 带上 `eps`，三个后端都从描述里取。

---

## 12. 属性表在长（最该盯的一个数）

```
Attention  sink · kv_tie · head_gate · gate_act · q_gate · partial · qk_norm · rope_scale
MoE        routing · expert_bias · router_bias · shared_gate · score_bias
           n_group · topk_group（分组路由，只进契约）
Linear     decay · shortconv · bounded_by · conv_kernel
MLA        q_lora · kv_lora · qk_nope · qk_rope · v_dim
```

> **如果它不收敛，"族"就退化成"每个模型一套参数块"，只是换了个更好看的写法。**

现在有六个真实模型的数据。**MLO 是第一个真的新增了一族、而不是加属性的。**

---

## 13. 明确拒绝的东西（绝不悄悄生成错的）

```
qk_norm = full       Instella 是整宽归一化 [2560]，不是按头 [80]
KDA                  Ling 的线性注意力 —— 当成 Linear 会生成一个 GDN
noaux_tc 分组路由     Ling 的 sigmoid 打分 + 分组 top-k
MTP / NextN          Step-3.7 的三个多步预测头
```

**「名字一样含义不同」是今晚最反复出现的一类 bug：**

| | 为什么一直没露 |
|---|---|
| YaRN 的插值项写成 `1/(fac·p)` | 非 YaRN 时恰好退化成 `p` |
| GDN 的卷积就地覆写了还要读的缓冲 | 另一种写法也对 |
| 部分 RoPE 的频率按 `head_dim` 算 | 非 partial 时两者相同 |
| `interleaved` 指**输入**交错而非输出 | 只有 Ling 用这个值 |
| 门控激活：Laguna `softplus` / Ling `sigmoid` | 都叫 `head_gate` |
| `render_e` 的闭包晚绑定 | 只有同时声明 `expert_name` 和 `global_name` 才触发 |

**共同点：全都是"看着对"，而且都能编译、能运行、能产出看起来正常的结果。**

---

## 14. `norm_eps` 这个 bug 出现了三次

1. GDN 的 `norm_eps` 写死 1e-6，而 RMSNorm 那边是 1e-5 —— 修了，**注释还留着**
2. MLA 的 compile_ir 分支又写死 1e-5 —— **在注释旁边重犯**
3. C 的 `rmsnorm` 写死 1e-5

> **知道一个 bug 存在、甚至把教训写成了注释，都不妨碍在新代码里重犯。**
> 注释救不了，只有"从模型配置取值"这个结构性做法能救。

---

## 15. `noaux_tc` 分组路由（Ling 压出来的第二个缺口）—— ✅ 已实现并验证

```
python by1moe.py
  选中的专家集合一致: 是
  权重（按专家对齐后）最大绝对差: 0.000e+00
  去掉偏置后有 12/12 行的选择变了  （偏置确实在起作用）
  [PASS] noaux_tc 与参考一致
```

**判卷人**：transformers 的 `DeepseekV3TopkRouter`。它和 Ling 的 `BailingMoeV3Gate`
语义逐行相同（sigmoid 打分、偏置只影响选择、每组前 2 之和选组、组内 top-k、
权重归一化 × routed_scaling_factor），**但是两个团队写的**。

**语义里最容易写错的一条**：`expert_bias` **只参与选择，不参与权重**。
选择用 `scores + bias`，权重用原始 `scores`。写成两者都用 bias 的话，
前向看不出明显异常，只有对拍才抓得到。

**可证伪控制**：把偏置设成非零并检查"去掉它之后选择必须变"。
否则偏置是全零的话，这个测试会**空过** —— 那是今晚第 N 次遇到
"一个不报错的检查被当成通过的检查"。

---

## 16. KDA —— 明确不做，原因是没有判卷人

Ling 那 18 层 KDA 的核心不在 `modeling_bailing_moe_v3.py` 里：

```python
from fla.ops.kda import chunk_kda, fused_recurrent_kda    # flash-linear-attention
```

`fla` 要 **Triton**，而这台机器只有 Iris Xe、没有 CUDA（实测装上后 import 就失败）。

**按项目原则：没有能跑的判卷人的块一律不写。** KDA 归到和 Kimi K3、
DeepSeek V4.1 同一类。codegen 对它**明确拒绝**，不会按 GDN 生成一个
"看起来对的错模型"。

**如果想做，需要的东西是明确的**：一台有 CUDA 的机器，或者 llama.cpp 的
`bailingmoe3.cpp` 能在本地跑起来当判卷人。
