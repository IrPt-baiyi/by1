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

---

## 17. llama3 式 RoPE（Step-3.7 压出来的第三个缺口）—— ✅ 已实现并验证

```
python by1rope.py
  逆频率最大绝对差: 5.960e-08   相对 5.960e-08
  attention_factor: 参考 1.000000   by1 1.000000
  当成 YaRN 算的话最大绝对差: 1.068e-04  （确实不同，类型区分是必要的）
  [PASS] llama3 RoPE 与参考一致
```

**判卷人**：transformers 的 `_compute_llama3_parameters`。

**两个容易混的地方：**

1. **llama3 和 YaRN 都改频率、参数名还重叠**（`factor`、
   `original_max_position_embeddings`），但改法完全不同：YaRN 调的是 ramp 的
   起止维，llama3 调的是**波长阈值**。所以 `rope_type` 必须从描述里读出来，
   不能"有 factor 就是 YaRN"。
2. **llama3 的 attention_factor 恒为 1.0**（参考实现的注释写着
   "Unused in this type of RoPE"）。跟着 YaRN 的公式算会多乘一个 1.069 ——
   位置 0 上看不出来，要靠对拍。

**判卷人抓到的实现错误**：我的 `p0` 漏了 `1.0 /`，第一个频率正好变成倒数
（1.2725 而参考是 0.7858）。YaRN 分支没漏，因为它后面才取倒数 ——
我把变量名从 `pos` 改成 `p0` 的时候把语义也带跑了。

新增 `llama3-shaped.by1`：三后端全过（C↔NumPy 5.114e-07、NumPy↔Torch 4.462e-07）。

---

## 18. 逐层覆盖原来只改了契约，没改生成的计算 —— ✅ 已修

**这是今晚最危险的一个 bug，因为它两个检查都能过。**

```
逐层覆盖 main[3..44] : MoE.experts = [...] 加上之后：
  by1check 的契约      18 种专家数（255/266/251/…/288）  ✓
  by1verify 张量对拍   753 声明、728 命中、形状 0 不符    ✓
  compile_ir 生成的计算 42 层全是 experts=288            ✗ ← 没人看这里
```

**照这份描述生成出来的模型，每一层都会用 288 个专家 —— 而所有检查都是绿的。**

原因：`overrides` 在 `by1check` 里算出来之后只用在张量契约上，
没有传给 `compile_ir`；而 `compile_ir` 取挂载机制的属性时用的是**声明值**。

修法：`info` 里带上 `overrides`，`compile_ir` 合并到挂载机制的属性上。

修复后：`compile_ir` 看到 19 种组合，`swiglu_limit` 落在 43/44 层 = 7.0，
和官方 config 逐项一致。

**教训：「契约对了」不等于「生成的对了」。** 这两条路必须都能被检查到 ——
而当时只有前一条有检查。

---

## 19. config 生成器：从 4 种表达式扩到 13 种（Step-3.7 压出来的）

**Step-3.7 的 config 逐字段对拍跑不起来** —— 它的字段超出了生成器的能力。
补齐之后：

```
step-3.7-flash   一致 47   值不同 0   by1 多出 0   官方多出 2
                 （多出的两个永远是 architectures / transformers_version 这类 HF 元数据）
```

新增的表达式：

| 写法 | 干什么 |
|---|---|
| `by_layer(attach.experts)` | 逐层字典 `{"3": 255, "4": 266, ...}` |
| `join(indices_where(...), ",")` | 拼成字符串（`moe_layers_enum`） |
| `per_layer(attach.x, 0)` | 属性**缺失**时给默认值 |
| `per_layer_rope(rope_theta)` | 逐层 rope 参数（position 是按层类型声明的，要先映射） |
| `sliding_window` | 从**开了窗的层**取（第 0 层可能是全量） |
| `rope_scaling` / `rope_parameters_flat` | 摊平 / 带逐层数组的两份 |
| `attention_other_setting` | 滑窗那一套的注意力参数 |
| `pad` / `pad_last` / `pad_cycle` | 补齐到指定长度 |

### 补齐规则**有三种，而且长得很像**

官方那些逐层数组长度是 48、模型是 45 层（`12 × 4` vs `11 × 4 + 1`）。
多出来的 3 个位置补什么，三种规则各有实例：

```
layer_types             45/46/47 → sliding_attention   按 4 周期续
partial_rotary_factors           → 1.0
rope_theta                       → 10000.0
swiglu_limits                    → 0.0, 0.0, 0.0       补零
```

我一开始全用了 `pad_last`（重复最后一个）——**数组长度对、大部分值也对，
只有尾巴三项不同**。又一个"看着对"。

### 顺带修的一处

`per_layer` 在属性缺失时会退回**机制类型名**（`'MoE'`/`'FFN'`），
于是逐层数组里混进了字符串。新的 `per_layer(x, default)` 查的是属性本身有没有。

---

## 20. 共享专家原来没走 `_swiglu`

`by1codegen` 里共享专家那条路是裸的 `F.silu(sw1(x)) * sw3(x)` ——
**风格（silu / gptoss）和夹取值对它都不生效**。

Step-3.7 的 `swiglu_limits_shared`（路由专家夹 7、共享专家夹 16）逼出了这个。
现在共享专家也走 `_swiglu`，用自己的夹取值。

---

## 21. gpt-oss-120b 的 config —— 最后一个"跳过"被补上

它的 `field` 映射从来没写过，验证器一直在打"跳过"。补上之后 **30/30、0 不同**。

**六个真实模型的 config 全部通了：**

| 模型 | 一致 | 值不同 | by1 多出 | 官方多出 |
|---|---|---|---|---|
| gemma-4-31b | 36 | 0 | 0 | 0 |
| gpt-oss-120b | 30 | 0 | 0 | 2 |
| laguna-xs-2.1 | 35 | 0 | 0 | 2 |
| ling-3.0-tiny | 72 | 0 | 0 | 3 |
| instella-3b | 21 | 0 | 0 | 3 |
| step-3.7-flash | 47 | 0 | 0 | 2 |
| **合计** | **241** | **0** | **0** | 12 |

那 12 个"官方多出"永远是 `architectures` / `auto_map` / `transformers_version`
这类 **HF 元数据** —— 它们描述的是"这份权重怎么被加载"，
不是"这个模型的架构是什么"，本来就不该由架构描述生成。

### 这一轮修的三处

1. **`rope_scaling` 的键随类型变**：YaRN 是
   `{beta_fast, beta_slow, factor, original_max_position_embeddings, rope_type, truncate}`，
   llama3 是 `{factor, high_freq_factor, low_freq_factor, ...}` ——
   **共同项只有两个**。原来共用一份模板，对 YaRN 全错。
2. **`{...}` 字面量**：`quantization_config` 这种嵌套结构以前解析不了。
3. **`rope_theta` 要从 `position` 里取**：有些模型的 Attention 机制里没写它。

顺带：gpt-oss-120b 的 `.by1` 里原来**没声明 `rms_eps`**，
以及 YaRN 少了 `truncate = false` —— 都是这次对拍逼出来的。

---

## 22. 项目审视（2026-10-08 深夜）

一次正经的自查，不是报状态。**找出了六个问题，其中一个是我自己上一轮引入的。**

### ① 一个"全绿"的表里有一个空格，我读过去了

上一轮我贴的回归表：

```
qwen3-next-shaped.by1  C<->NumPy   2.990e-06   NumPy<->Torch
                                                             ↑ 空的
```

那一格是空的，因为 `by1exec --compare` **崩了**：

```
AttributeError: 'MoE' object has no attribute 'limit_shared'
```

原因：我加 `self.limit_shared` 时锚点是 `self.limit = a.get("limit")`，
**它先匹配到了 `MLP` 类**。于是 `MLP` 拿到一个用不上的属性、`MoE` 缺它。

**这一条比 bug 本身重要**：我生成了一个带空格的表，然后说"全绿"。
空单元格和绿单元格在视觉上几乎一样，而我没有去核对每一个。

### ② 我叫你明早跑的命令其实是失败的

`python by1check.py *.by1` —— `deepseek-v4.1-flash.by1` 有 1 个错误、
`qwen3.8-flash-next.by1` 有 5 个。

它们是按 1.md 的愿望清单写的**草稿**，没有外部判卷人，
写的时候连检查器都还没有。已移到 `drafts/`，并写了 README 说明
**要让某一个毕业需要的是判卷人，不是把检查器的错误消掉**。

### ③ `gemma` 的 config 对拍一直在只看一半

`by1verify` 的 `--sub` 默认是 `text_config`，而 Gemma 的 config 是**多模态**的：
真正的文本架构嵌在 `text_config` 里，外面还有 `vision_config` / `audio_config` /
图像视频 token 共 14 个顶层键。

**报告写的是「官方多出 0」，读起来像"什么都没缺"。** 那些键确实不在
by1 的建模范围里 —— 但那是**范围边界**，不该被藏起来。现在它会明说：

```
（只比 text_config 那一层；顶层另有 14 个键未参与比较：architectures,
 audio_config, audio_token_id, boa_token_id, boi_token_id …）
```

### ④ config 检查把 HF 元数据算成失败

`architectures` / `auto_map` / `transformers_version` / `torch_dtype` /
`quantization_config` 描述的是"这份权重怎么被加载"，不是"这个模型的架构是什么"。
它们不该由架构描述生成 —— 现在单列一类、**不算失败、但仍然报出来**。

### ⑤ 工具自己不一致：字段映射会补齐，layer_types 检查不知道

`layer_types` 的字段映射用 `pad_cycle` 补到 48，而独立的 layer_types 检查
拿 45 去比 48，报 `[FAIL] 层数不一致` ✗

现在检查器允许"config 更长"，**但会验证多出来的确实是周期的延续** ——
不能把"允许"变成"放过"。

### ⑥ 从来没有"跑全部"的入口 —— 这是最大的一条

**没有 CI、没有测试运行器。所谓"护栏"是我一条条敲命令、肉眼看输出。
我停下来，就没人知道这个项目是不是还是好的。**

现在有 `by1all.py`：

```
python by1all.py            跑全部，失败非零退出
python by1all.py --quick    跳过 C 后端
```

34 项：检查器（含 selftest 的**反向确认** —— 它必须报错）· 六个模型的 config
与张量 · 四个合成模型的前向 · 六个合成模型的两条跨后端对拍 · 三个判卷人脚本。

**已知缺口单列**，仍然打印："
+ '"'
+ @'
Step-3.7 那 25 个张量在 HF 的 model-00009 分片里，而那个分片的头是全零 ——
镜像的问题，不是 by1 的。
**藏起来的缺口和没发现过的缺口一样糟。**

---

## 23. 这次审视真正的结论

**验证的密度不均衡：**

| 东西 | 验到了什么 |
|---|---|
| config（六个真实模型，241 字段） | 数值转录 |
| 张量名与形状（六个真实模型，4 万+） | 命名与布局 |
| **前向**（四个合成模型 + MLA） | **真实的数值行为** |
| 反向（四个合成模型） | 梯度 |

**六个真实模型，没有一个的前向被跑过。** 不是不想跑 ——
gpt-oss-120b 的 130B 参数、gemma 31B，这台机器（16GB 内存、无 CUDA）跑不动。

**所以"六个真实模型全中"这句话的准确版本是**：
它们的 **config 和命名** 全中，**数值行为**只在合成的维度上验过。

这不是失败，是**没做到的那一半**，应该写在任何人看得到的地方。

---

## 24. Instella-3B 的前向 —— **真实维度**，逐位一致

```
python by1instella.py
  维度: d=2560  q=32  kv=32  head_dim=80  FFN=6912  层数=4（真实 36）  seq=16
  权重搬运: 47 / 47 成功
  最大绝对差 0.000e+00   相对 0.000e+00
  [PASS] Instella-3B 的前向与官方实现一致
```

**这是第一个在真实维度上被验过前向的真实模型。**

在此之前，"六个真实模型全中"的准确版本是「**config 与命名**全中，
**数值行为**只在合成维度上验过」。Instella 是六个里唯一在这台机器上也跑得动的
（稠密、3B），所以它第一个把后半句去掉了。

**做法**：只把**层数**从 36 降到 4（36 层 = 12GB fp32，加上参考模型装不下），
**其余维度全真**。临时生成一份 4 层的 `.by1`，跑完删掉 —— 不改工具。

### 它逼出来的东西：`qk_norm = full`

Instella 的 QK-norm 是**整宽**的：

```python
self.q_norm = InstellaRMSNorm(num_heads * head_dim, rms_norm_eps)   # [2560]
query_states = self.q_norm(self.q_proj(hidden_states))              # 拆头**之前**
query_states = query_states.view(bsz, q_len, num_heads, head_dim)
```

而 by1 在此之前是**明确拒绝**它的（"按 per_head 生成会得到一个看起来对但算错的模型"）。

**这是同一个名字下的第二个东西**：Qwen3-Next 也是 QK-norm，但按 `head_dim`、
作用在拆头**之后**。名字一样，位置和宽度都不同。

### 判卷人自己坏了

`modeling_instella.py` 是按 transformers **4.48** 写的，这台装的是 **5.15** ——
`DynamicCache.to_legacy_cache()` 没了，走缓存那条路直接 AttributeError。

传 `use_cache=False` 绕开。**这是判卷人的版本问题，不是被测代码的** ——
但不记下来，下次还会撞。

---

## 25. 打包：要上传到云端的东西

```
python by1pack.py
  → by1-upload-20261008.tar.gz   0.38 MB   65 个文件
```

**装**：11 个工具 + 13 份验证过的 `.by1` + `refs/`（25 个官方产物）+
文档 + `gpu/`（runbook、依赖、KDA 脚本）。

**不装**：模型权重（云端从 HF 下）· `llamacpp/`（可重下）· `drafts/`（没有判卷人的草稿）。

### `gpu/` 里是什么

| 文件 | 干什么 |
|---|---|
| `README.md` | runbook：跑什么、按什么顺序、期望什么 |
| `requirements.txt` | 依赖（torch 要点明装 CUDA 版） |
| `run.sh` | 一键，顺序**从便宜到贵** |
| `by1kda.py` | **把 KDA 的判卷人立起来并 dump 中间量** |

### `by1kda.py` 为什么不是"把 KDA 实现写出来"

KDA 的核心是 `fla.ops.kda.chunk_kda`，要 Triton。**我在没有 CUDA 的机器上
看不到它的源码** —— 也就是说：张量集我知道（Ling 9283/9283 已经对上），
但**门控语义我看不到**（`g` 怎么进 delta 规则、`safe_gate` 和 `lower_bound`
各管什么）。

**凭空写一个就是在猜。** 猜错的代价不是"改一行"，是"分不清是实现错了还是语义猜错了"。

所以那个脚本把顺序倒过来：**先立判卷人、dump 中间量，再对着 dump 写实现**。
租卡是为了**拿到判卷人**，不是为了跑得久。

---

## 26. 收敛检验：拿一个**刻意普通**的模型当判卷人

前面八个模型都是特意挑来压东西的（MoE、sink、混合栈、MLA、逐层剪枝……），
**每一个带出新属性是必然的**。所以"收敛吗"这个问题它们答不了。

`jingyaogong/minimind-3` 是判卷人：它就是一个标准 Qwen3 形状。

### 先写预测，再看结果

| 预测 | 结果 |
|---|---|
| "普通模型需要 0 个新属性" | **错了一半** —— 需要**权重共享** |
| "可能会挂的地方：tied embeddings / rope linear·dynamic / mlp_bias" | **tied embeddings 挂了** ← 猜中 |

### 结果

```
config          22 / 22      值不同 0
张量            90 / 90      形状不符 0
契约声明抑制     1            确实不存在 1   应无却有 0
检查器          0 错 0 警
```

### 唯一需要的新东西：**权重共享**

```
model.embed_tokens.weight   [6400, 768]
lm_head 在表里吗: False        ← 没有独立的 lm_head
```

**这是真的架构特性**（embed 和 head 是同一块内存），不是 config 的写法。
而且**不用改语言** —— `by1verify` 本来就有"声明抑制"的写法：

```
lm_head.weight    : --
```

**但那个写法一直是坏的**：`_flat_rows` 会把形状重新包成 `(...)`，
`--` 到不了检查器；而且全局张量走的是**另一个循环**，那个循环里根本没认这个标记。
两处都修了，并且加了**双向的可证伪对照**：

```
正常                    契约声明抑制 1 / 确实不存在 1 / 应无却有 0   → PASS
造一份带 lm_head 的表    应无却有 1                              → FAIL
```

### 另外一件事：逐层专家数是**剪枝产物**

对着 `stepfun-ai/Step-3.7-Flash`（官方）比了一下：

```
官方 text_config       **没有 moe_num_experts_by_layer**，moe_num_experts = 288
nerkyor 剪枝微调版      有，18 个不同的值（249~288）
```

**逐层专家数只存在于剪枝过的 checkpoint 里。** 所以那套逐层覆盖不是"特例补丁"，
它是**为一整类已发布产物**（剪枝 / 稀疏化之后的 checkpoint）准备的。

---

## 27. 词汇表检验：8 个新模型，逐个键对

拿新 config 的键去比已有 6 个模型的 135 个键：

| 模型 | 新键 | 性质 |
|---|---|---|
| **minimind-3**（qwen3） | **1** | `use_sliding_window=false` ≡ `window=none` —— **实际 0 个** |
| Qwen3.8-27B（qwen3_5） | 11 | 线性注意力 + 3+1 混合（都有）+ **输出门控** + MTP |
| Cloudflare/clef（qwen3_5） | 11 | 同族 |
| Qwen3.6-35B-A3B（qwen3_5_moe） | 10 | 同族 + MoE |
| **Nemotron-3.5**（nemotron_h） | **29** | **Mamba 是真的新机制** + 显式的逐层块类型表 |

**结论：Llama / Qwen 族已经收敛，跳出这个族（Mamba）就不行 ——
而那不是特例，是真的新东西。**

`nanoGPT` 没有 HF config（它是训练代码仓库）；它的架构要从 `model.py` 读，
而那是 **GPT-2**：`LayerNorm`（不是 RMSNorm）、学习式位置编码（不是 RoPE）、
`GELU`（不是 SwiGLU）—— **这门语言是 Llama 形状的**，这一条才是真的边界。

---

## 28. clef —— 第二个收敛判卷人，而且这个严格得多

`Cloudflare/clef` 是 Qwen3.5 形状：**48 层线性注意力 + 16 层全量注意力**
（每 4 层一次），全量那 16 层带输出门控和按头的 QK-norm。

```
config          33 / 33      值不同 0   多出 0   官方多出 0
layer_types     64 / 64
张量            851 / 851    形状不符 0  缺失 0
```

**这一整套我都有** —— 因为 Qwen3-Next 就是同一个形状：
线性注意力 = 我的 GDN（张量集逐个相同）· 3+1 混合 = 我的 schedule ·
输出门控 = 我的 `q_gate`（实测 `q_proj [12288] = 24 × 256 × 2`）·
按头 QK-norm = 我的 `qk_norm = per_head`。

**它是 Qwen3.8-27B 去掉 MTP**（实测 1184/1199 同名同形，多出的 15 个全是 `mtp.*`，
而 config 里 `mtp_num_hidden_layers = 0`）。

### 它逼出来的四个 bug，全是"同一个东西有两个实现"

**① `layer_types` 只看 `window`**

```python
def _ltype_of_attrs(a):
    w = (a.get("window") or "").strip().lower()
    return "full_attention" if w in ("none","null","0","") else "sliding_attention"
```

**它假设每一层都是带窗口的注意力。** GDN / KDA 这类线性注意力层根本没有
`window`，于是被算成 `full_attention` —— 64 层全错。

**混合栈的 `layer_types` 以前从来没被检查过**：Ling 的 config 里没这个键，
Qwen3-Next 那个没有 field 映射，clef 是第一个把它摆上台面的。

**② 同一件事两个实现，我修了一个**

`by1check` 里一个、`by1verify` 里一个，逻辑一样。
修完前者，结果是**同一个模型"字段对拍通过、层类型检查失败"**。
现在 `by1verify.by1_layer_types` 直接用 `info["layer_types"]` —— 只留一份。

**③ 机制属性里引用 hparams 符号，不参与形状求值**

432 个张量因此解析不出来（"符号形状跳过"）。改成字面量。
**这是个已知短板**：机制属性写 `k_heads = lin_k_heads` 是自然的，
但现在不行。

**④ `block` 里重复同一个部件名，解析不了**

`block = local + local + local + global` ✗ → 每个部件要先单独命名 ✓
`lin = 3 * Lin` 然后 `block = lin + full`。

### 还有一件：检查是**单向**的

上面全是"契约 → 实物"。于是 `官方多出 0` 读起来像"什么都没漏"，
而**实物里可能有几百个契约根本没提的张量**。

补上反方向之后，立刻看到：

| 模型 | 契约未覆盖的实物张量 |
|---|---|
| gpt-oss-120b | 0（全覆盖） |
| minimind-3 | 0（全覆盖） |
| **clef** | **333 个（28%）** —— 全是 `model.visual.*` 视觉塔 |
| **gemma-4-31b** | **303 个（36%）** |

**"gemma 530/530 张量"听起来像全覆盖 —— 实际是 833 个里的 530 个。**
36% 的权重从来没被提过。这是**范围边界**，不是失败 —— **但必须看得见**。

### Qwen3.5 族剩下的两个

`Qwen3.8-27B` = clef + MTP(1 层) · `Qwen3.6-35B-A3B` = 40 层 + MoE（gate_up 融合、
共享专家、shared_gate，全是已有词汇）+ MTP(1 层)。
**MTP 是这一族里唯一的新东西。**

---

## 29. Qwen3.8-27B —— MTP，以及它逼出来的三个语言改动

**MTP 是 Qwen3.5 这一族里唯一的新东西。** 实测它和 `clef` 的关系是精确的：
1184/1199 个张量同名同形，多出来的 15 个**全部**是 `mtp.*`，
而 config 里 `mtp_num_hidden_layers` 一个是 1、一个是 0。

```
config          33 / 33      值不同 0   多出 0   官方多出 0
layer_types     64 / 64       （65 层里只算主栈那 64 层）
张量            866 / 866    形状不符 0   缺失 0
契约未覆盖       333 个（28%）—— 全是 model.visual.* 视觉塔
```

### MTP 是什么

多 token 预测头。**不是推理路径的一部分**（训练时的辅助头 / 投机解码用），
但它**在权重文件里**，所以描述要能表达它。而它和主干是**并列的两个栈**：

```
mtp.pre_fc_norm_embedding ─┐
                            ├─> mtp.fc [5120, 10240] ─> 1 层 transformer ─> mtp.norm
mtp.pre_fc_norm_hidden    ─┘
```

### 三个语言改动

**① `aux = true` —— 辅助栈不算解码层**

`n_layer` 和 `layer_types` 只算主栈。原来检查器把所有栈一起数，
于是 `n_layer = 64` 对不上"合计 65 层"。
**靠栈名叫 `model` 来判断是魔法，所以让 `.by1` 显式声明。**

**② `name_<栈名>` —— 各栈的物理前缀不同**

主干是 `model.language_model.layers.{i}.…`，MTP 是 `mtp.layers.{i}.…`。
一个模板装不下。所以名字模板允许**按栈各写一份**：`name_<栈名>` 优先，`name` 兜底。
（顺带发现 `scope { ... }` **必须写在一行里** —— 换行会让解析器只读到第一段，
scope 静默变空、所有前缀消失。）

**③ 栈内序号 —— 这个是真 bug**

物理名字里的层号是**各栈从 0 起**的，而 `by1verify` 传的是全局层号。
于是 MTP 那层被拼成 `mtp.layers.64.…` —— **名字全错，而形状是对的**，
所以只看"形状不符 0"会以为没事。表现是 `缺失 11` 里混着一堆看似正常的名字。

---

## 30. 两个"普通模型"的收敛检验，结论一正一反

| 模型 | 新属性 | 结果 |
|---|---|---|
| `minimind-3` | 需要**权重共享** | 预测错了，但错在一个真架构特性上 |
| `clef` | **0 个** | 整套词汇现成（Qwen3-Next 时代就有） |

**clef 是有分量的那个**：48 层线性注意力 + 16 层全量注意力的混合栈，
输出门控、按头 QK-norm、3+1 调度 —— **一个都不新**。

再加上 `qwen38` ＝ clef + MTP，而 MTP 只值一个 `aux` 标记。

**所以"Llama / Qwen 族已经收敛"这句话现在有证据了。**
跳出这个族（Mamba、GPT-2）不是。

---

## 31. Qwen3.6-35B-A3B —— **零个新属性**

```
config          37 / 37      值不同 0   多出 0   官方多出 0
layer_types     40 / 40
张量            712 / 712    形状不符 0   缺失 0
契约未覆盖       333 个（32%）—— 视觉塔
```

同一个 Qwen3.5 形状，把稠密 FFN 换成 MoE。**没有一个新属性**：
线性注意力、3+1 混合、输出门控、按头 QK-norm、MoE、共享专家、
共享专家门控、MTP —— 全是现成的。

它和 Qwen3.8-27B 只差三处：没有稠密 FFN（`intermediate_size = null`）、
40 层全是 MoE（不像 Ling/Step 那样第一层是稠密）、
专家是**融合存储**的（`experts.gate_up_proj [256, 1024, 2048]` 而不是分开两块）。

**融合就直接照实写进契约** —— 不去假装它是分开的再拼起来。

---

## 32. 官方 Step-3.7-Flash —— 剪枝的代价是**四行**

从 `step-3.7-flash.by1` 派生，删掉三处剪枝相关的行：

```
① main[3..44] : MoE.experts = [...]      逐层专家数
② moe_num_experts_by_layer = ...         它在 config 里的对应物
③ step37_dynamic_prune_plan = "..."      剪枝计划本身
```

再加上官方 `text_config` 里没有显式写出的四个继承默认值：

```
config          40 / 40      值不同 0   多出 0   官方多出 0
layer_types     45 / 45      （config 是 48，多出的 3 项**验证过是周期延续**）
张量            753 / 753    形状不符 0   缺失 0
契约未覆盖       718 个（49%）—— vision_model.* 视觉塔
```

**顺带确认了一件事**：剪枝版那 25 个"缺失"确实是**镜像的坏分片** ——
同一批张量在官方仓库里 753 个全在、全对。

**所以"逐层覆盖"这套机制的代价，就是那四行。**
一个剪枝过的 checkpoint 和一个未剪枝的，同一个描述加四行。

---

## 33. 收敛检验的账（做到这里）

| 模型 | 新属性 | |
|---|---|---|
| `minimind-3` | **权重共享** | 一个真架构特性 |
| `clef` | **0** | 48 层线性注意力 + 16 层全量，整套现成 |
| `qwen38` | **aux 栈**（MTP） | 一个标记 + 两个名字相关的改动 |
| `qwen36` | **0** | MoE + 共享专家 + 门控 + MTP 全是现成的 |
| `step37-official` | **0** | 从剪枝版**减**四行 |

**Llama / Qwen 族收敛，这一条现在有四个独立的数据点。**

---

## 34. GPT-2 —— 语言的边界，以及一个以前没被摆出来的区分

`karpathy/nanoGPT` 是训练仓库，**没有 HF config**。但它是 GPT-2 架构，
而 `openai-community/gpt2` 有全套官方产物 —— 拿那个当判卷人（同一个架构）。

```
config          21 / 21      值不同 0   多出 0   官方多出 0
张量            160 / 160    形状不符 0   缺失 0
契约声明抑制      1           确实不存在 1（lm_head —— 权重共享）
```

### 它是"另一代人"，不是特例

| | GPT-2 | 前面十一个 |
|---|---|---|
| 归一化 | **LayerNorm** | RMSNorm |
| 位置 | **学习式 wpe** | RoPE |
| 前馈 | **两层 + GELU** | 门控 + SwiGLU |

**这三样都不是特例，是另一个时代的标准做法。**

### 结果：**描述层面过了，计算层面过不了**

**契约能表达它，而且一个新语法都没加**：
- 没有 `position` 块 ✓（位置是 `wpe` 张量，不是旋转）
- `gate = none` 的 FFN ✓
- 带 bias 的归一化 ✓（就是多两行张量）
- 融合的 `c_attn.Weight [768, 2304]` ✓ 照实写

**但三个后端一个都算不了它。** RMSNorm 不减均值、没有位置查表、
FFN 没门控就没有那条乘法。

> **"能描述" ≠ "能算"。** 这个区分以前从来没被摆出来过 ——
> 因为前面十一个模型两件事同时成立，所以看不出来。

### 顺带两个真的坑

**① `ctx` 不在图层/全局张量的符号表里**

`by1check` 在张量契约实例化时就报了 `bias (1, 1, ctx, ctx) <- 缺 ctx`。
层级和全局张量只从 `hparams` 取符号，不含模型的 `ctx`。
**它当场说了** —— 这就是"检查器会抱怨"的价值。

**② 那个 checkpoint 存的是 base model 的名字**

`h.0.attn.c_attn.weight`、`wte.weight` —— **没有 `transformer.` 那一段**
（`GPT2LMHeadModel` 的命名才有）。名字是**产物的事实**，
不是我该"修正"成我以为的样子。

### 还有一件值得记的

`h.N.attn.bias [1, 1, 1024, 1024]` —— **因果掩码当 buffer 存在权重文件里**。
它不是学出来的参数，但它在表里，所以契约要说出来。

---

## 35. Nemotron-H —— 唯一真正的新机制族：Mamba

前面十二个模型全是注意力系（含 GDN / KDA 那种线性注意力），
**Mamba 不是它们的变体**：状态由 `A_log` / `D` / `dt_bias` 驱动，
没有 q/k/v，也没有 delta 规则。

```
config          58 / 58       值不同 0    by1 多出 0    官方多出 0
张量            6513 / 6513   形状不符 0   缺失 0
契约未覆盖        0           （全覆盖 —— 没有视觉塔）
```

### 三样新的，两样是语言改动

**① Mamba 层本身** —— `in_proj [10304, 2688]` 自检：
`2×4096 + 2×8×128 + 64 = 10304` ✓ · `conv1d [6144,1,4]` = `4096 + 2048` ✓
它是 `SSM` 种类（`TOKEN_MIXER` 里早就留着这个名字，一直没人用）。

**② 显式的逐层序列** —— `layers_block_type` 是**手写的表**：

```
attention 落在 5, 12, 19, 26, 33, 42    间隔 7, 7, 7, 7, **9**
```

**不是纯周期。** 而且就算能凑出一个周期表达式，那是**推导**，不是**描述** ——
config 给的是字面列表。所以给 `pattern` 加了显式列表：

```
pattern = [Mamba, MoE, Mamba, MoE, Mamba, Attn, MoE, ...]
```

（实现上只是让 `parse()` 也认逗号。**必须写在一行** —— 折行容易漏逗号，
而漏了的表现是"pattern 无法解析 + 主栈 0 层"，报得清楚但那是我自己造的。）

**③ 专家是一个专家一个张量**（2944 + 2944 + MTP 256 个）——
契约写打包形状 + `per_expert` 标记（ling / laguna 已有的写法）。

### 顺带一个真 bug：`name_<栈>` 会**静默吞掉**专家模板

```
expert_name  = 'backbone.layers.{i}.mixer.experts.{expert}.{physical}'
渲染出来却是  backbone.layers.0.mixer.up_proj.weight     ← experts.0. 整段没了
```

因为 `name_model` 的优先级高于 `expert_name`。
**名字看起来正常，只是少了中间一层** —— 这种错最难看出来。
现在优先级是 `expert_name_<栈>` > `expert_name` > `name_<栈>` > `name`。

### 还有

`mtp.layers.1.final_layernorm.weight` —— 我写死成第 0 层了。
**MTP 的两层里，投影在 0、最终归一化在 1。**

---

## 36. 收敛检验的最终账（十三个真实模型）

| 模型 | 新属性 | 性质 |
|---|---|---|
| `minimind-3` | 权重共享 | 真架构特性 |
| `clef` | **0** | Qwen3.5 混合栈，整套现成 |
| `qwen38` | `aux` 栈 | MTP |
| `qwen36` | **0** | MoE + 共享专家 + MTP 全现成 |
| `step37-official` | **0** | 从剪枝版**减**四行 |
| `gpt2` | **0 个语法**（但三个后端算不了） | 上一代：LayerNorm / wpe / GELU |
| `nemotron-h` | Mamba + 显式逐层序列 | 真的新机制 |

**结论分三层：**

1. **Llama / Qwen 族已经收敛** —— 五个独立数据点
2. **同代的"另一支"也收敛** —— GPT-2 在**契约层面**一个新语法都不用，
   但**三个后端算不了它**（"能描述" ≠ "能算"）
3. **真正需要新东西的是新机制族** —— Mamba 要一个种类 + 一个调度写法，
   而那是**真的新**，不是特例

---

## 37. GLM-5.3-Flash —— **config 全中，张量契约 97%，两处没做完**

这批里最富的一个，也是唯一同时用上 1.md 点名那两样的。

```
config          57 / 57       值不同 0    by1 多出 0    官方多出 0
layer_types     45 / 45
张量            36289 / 37534 命中（97%）   缺失 1245
契约未覆盖       39819 个（52%）
```

### 它是什么

```
layer_types   34 × linear_attention(KDA) + 11 × deepseek_sparse_attention
              3+1，全量落在 3,7,11,...,43 —— **尾巴 L44 是线性的**（我第一版写成全量，错了一层）
mlp_layer_types  3 dense + 42 sparse（first_k_dense_replace = 3）
linear_attn    KDA，但门是**低秩**的：f_a[128]→f_b[8192]、g_a→g_b（Ling 那边 f 是一个矩阵）
全量层         MLA + **稀疏索引器**（indexer 自己一套 7 个张量）
残差           **mHC**：hc_attn/hc_ffn 的 {base[24], fn[24,16384], scale[3]}
专家           288 选 8，noaux_tc，**每个专家都有 weight_scale_inv（FP8 块量化）**
MTP            1 层，接在主栈后面的第 45 层
```

### 做成的

- **config 57 个字段全中**，包括 `linear_attn_config` 那个嵌套字典、
  `mlp_layer_types`（新生成器：看那一层**挂的是 FFN 还是 MoE**）、
  `indexer_types`
- **`ltype` 显式覆盖** —— 有些 config 对这一层有自己的叫法
  （GLM 叫 `deepseek_sparse_attention`，因为它带索引器）。
  与其让检查器猜，不如让 `.by1` 说。
- **KDA 也算线性注意力** —— 它是自己的**计算**，但层类型的名字一样。
- 契约写到 36289 / 37534（97%）：MLA 六件套、索引器七件套、KDA 的低秩门、
  mHC 的六个、MoE 的 per_expert 三件套

### 没做完的两处（**不打算说成做完了**）

**① MTP 是主栈后面的第 45 层，层号接着数**

实测 `model.language_model.layers.45.*` —— 它和主干**共用命名**，
不是 Qwen 那样另起 `mtp.` 前缀。而我的 `aux` 栈会把层号重置成 0。

已经给 `aux` 栈加了 `index = global` 的能力（by1verify 里，已验证不影响其它模型），
但 **`.by1` 那一处还没落上**（here-doc 锚点第 N 次没匹配），所以 MTP 那层的
输入/输出 norm 计数差 1。

**② FP8 的 `weight_scale_inv` —— 39819 个张量，占了整个 checkpoint 的 52%**

```
mlp.experts.{e}.{gate,up,down}_proj.weight_scale_inv   [16,32] / [32,16]   ×12384 各
self_attn.{q_a,q_b,kv_a,o}_proj.weight_scale_inv                            ×12
```

块大小 128（4096/128 = 32、2048/128 = 16）。这要走 `quant` 机制
（gpt-oss 那边用它做过 MXFP4 打包），**但那是另一件有分量的事**，
不是顺手能补的。

### 所以这一份的状态

**"config 对上" 和 "张量对上" 是两件事。** 前者是完整的，
后者是 97%，剩下 3% 里一半是命名（好修），一半是 FP8（另一个课题）。

**不把它加进 by1all** —— 那样 `by1all` 会红，而红的理由应该是"坏了"，
不是"还没做完"。

---

## 38. 这玩意现在能干什么

不是问"设计上能干什么"，是**今天敲命令就能干什么**。

### ① 把一份已发布的 checkpoint 描述出来，并**证明**描述是对的

```bash
python by1verify.py clef.by1 refs/Cloudflare__clef.config.json --config \
                    --tensors refs/clef.tensors.json --backend torch.module
```

13 个真实模型，config 逐字段 + 张量逐名字逐形状。判卷人是**官方产物**。

**它抓到过真东西**：
- 剪枝版那 25 个"缺失"其实是**镜像的坏分片**（同一批张量在官方仓库里全在）
- gemma 的"530/530"其实是 **833 个里的 530 个** —— 36% 从来没被提过

### ② 同一份描述**同时是能跑的**（这一条以前没被摆出来过）

```bash
python by1exec.py clef-tiny.by1 --compare      # NumPy vs PyTorch
python by1c.py    clef-tiny.by1 --gcc <gcc>    # 生成 C、编译、跑
```

```
clef-tiny   8 层 · d_model 256 · 0 错误
NumPy-Torch 6.135e-07   PASS
C-NumPy     4.244e-07   PASS
非数字行的差异 —— 只有注释和 model 名字
```

**`clef-tiny.by1` 是 `clef.by1` 机械缩小维度得来的，结构一个字没改。**
所以这不是两个例子 —— **是同一份描述**：既能对着 Cloudflare 的真 checkpoint 验，
又能三个独立实现各跑一遍、结果一致。

### ③ 比架构（这个别人没有）

```
clef      vs  qwen38          差一层 MTP
剪枝版    vs  官方 Step-3.7   差四行
nemotron  attention 间隔      7, 7, 7, 7, **9**
```

config.json 是**平的 JSON** —— 做不了这件事。差一层 MTP 在 JSON 里
是"15 个键的增删"，在 `.by1` 里是 `aux = true` 一行。

---

### 差一步的

- **改一行换架构**：把 clef 的 3+1 改成 1+1、把 qwen36 的专家数改掉，
  重新生成再跑 —— 机制都在，**但没做过**。这是 ablation / 架构搜索的路。
- **llama.cpp**：`by1emit.py` 能出图，**但一行 ggml 都没生成过、没跑过**。

### 不能做的（诚实）

- **快**：C 后端是手写循环，不是 ggml 原语。它是**正确性参照**，不是运行时。
- **覆盖全**：Mamba / KDA / 稀疏索引器 / mHC 只有契约，**算不了**；
  GPT-2 的 LayerNorm / 学习式位置 / GELU 也算不了。
  **"能描述" ≠ "能算"** —— 这个区分是 GPT-2 逼出来的。
- **给别人用**：四万个对上的张量名，**没有一行是"有人用它做成了什么"**。

---

## 39. 从头审视

拿 1.md 的每一条主张对着现实量。

### 账

```
by1check.py 2015 · by1codegen.py 1118 · by1c.py 1007 · by1exec.py 607
by1verify.py 553 ·  其余 ~2600
──────────────────────────────────────────
7922 行 .py   ·   3528 行 .by1（23 份）   ·   ir.md 1283 行   ·   42 次提交
1.md 的验收表说 4 个真实模型 → 现在 14 个
```

### 最重的一条：**描述说的是一个模型，生成出来是另一个，而且一声不吭**

`gpt2.by1` 里写着 `gate = none`、`qk_norm = off`、没有 `position`。
把它和 llama-shaped 的 IR 摊开并排：

```
【gpt2.by1】层 0                   【llama-shaped.by1】层 0
  Norm       RMSNorm                 Norm       RMSNorm
  Attention  Attn                    Attention  Attention
  Add        Add                     Add        Add
  Norm       RMSNorm                 Norm       RMSNorm
  FFN        MLP {act: gelu_new,     FFN        MLP {act: silu,
                  gate: true}                     gate: true}
  Add        Add                     Add        Add
```

**一模一样。** `gate = none` 变成了 `gate: true`，RMSNorm 顶着 GPT-2 的
LayerNorm，**三个后端、by1verify、by1all 全都说"过"**。

**这是同一个病的第三次**：

| | |
|---|---|
| `"off"` 是真值字符串 | 凭空多一个 RMSNorm → 2.505e-07 变 3.362e-03 |
| `_bool("per_head")` | `sink` 被静默关掉 |
| **`gate = none`** | **变成 `gate: true`** |

**但这次和前两次不同**：前两次是"算出一个错的数"（还会被对拍抓住），
这次是"算出一个**从没被验过的模型**，而所有检查都放它过去"。

### 为什么所有检查都放它过去

**张量契约查的是参数的名字和形状，不是"算了什么"。**

`gpt2` 的张量 160/160 全中 ✓ —— 因为**那张表描述的是权重，不是运算**。
它过了，于是看起来什么都没问题。

### 我审自己的时候也量错了

我先写了个脚本数"能算几个模型"，用 `SUPPORTED_KINDS` 判断，得到 **11/14**。
**那个数是错的**，因为 `SUPPORTED_KINDS` 是**种类级**的门：

```
codegen 对 11 个真实模型的反应（实测）
    gpt-oss-120b … step37-official    全部「接受（一声不吭）」
```

**属性级没有门。** `compile_ir` 只会拒绝它不认识的**种类**（KDA / SSM），
对认识的种类里**它不认识的属性**一律放行。

真正的"能算"应该是"有人拿它和外部参考比过"：

| | |
|---|---|
| 比过 | 6 个 shaped + clef-tiny + instella-3b（真实维度）= **8** |
| 只是编译过了 | 其余全部 |

### 1.md 里已经过时的三条

**① 验收表说 4 个模型** → 现在 14 个，而且多了两个"另一代"的
（GPT-2）和一个真新机制族（Mamba）。

**② "属性集在长，这是最该盯的数，而现在只有四个模型的数据"**
——**现在有数据了**：

```
73 个属性，13 个模型
而最后五个模型（qwen36 / gpt2 / clef / qwen38 / nemotron）
**一个属性都没加**
```

**这是收敛的硬证据。** 1.md 那句"如果它不收敛，族就退化成每个模型一套参数块"
——**在 Llama/Qwen 族里它收敛了**，跳出这个族（Mamba）才要新东西，
而那是真的新机制，不是特例。

**③ 方法那条有问题**：

> "每个做成的块都有判卷人，每个卡住的块都没有。"

**llama.cpp 是反例。** llama.cpp 的源码就是判卷人（`qwen3next.cpp`
逐行核对过、张量名双向 612=612），而它**一行 ggml 都没生成过**。
所以这条规则解释不了它 —— 卡住的真实原因是**那一步没有被排进任何一次工作**。

### 1.md 里没有、但被这段时间逼出来的四条

1. **能描述 ≠ 能算**（GPT-2 逼出来的，现在挂在最显眼的地方）
2. **显式的逐层序列** —— `layers_block_type` 是手写的表，不是周期
3. **`aux` 栈 / MTP** —— 辅助头在权重里、不在 `num_hidden_layers` 里
4. **"契约未覆盖的实物张量"** —— gemma 的"530/530"其实是 833 个里的 530 个

---

## 40. 取值门 —— 「能描述」和「能算」之间的那道缝，以前是**静默**的

### 起因

从头审视时看到：`gpt2.by1` 写着 `gate = none`，而它和 `llama-shaped`
编译出的 IR 算子序列**一模一样**。根因是这一行：

```python
"gate": str(attrs.get("gate", "true")).lower() != "false",
```

**`gate = none` → `True`**（`"none" != "false"`）。
而项目里早就有正确的 `_flag` —— 只是这里没用它。
**同一个东西两个实现，第四次。**

### 装的门

`ENUMS` + `_enum_bad`：`act` / `qk_norm` / `routing` / `gate_act` / `pairing`
这几个**取值选分支**的属性，取值必须在实现过的集合里。

**并且给它配了可证伪对照**（`_gate_probe.py`）：

```
gpt2          act = gelu_new      必须被拒   ✓
nemotron-h    SSM                 必须被拒   ✓
ling          KDA                 必须被拒   ✓
clef-tiny     取值全都实现过        必须接受   ✓
step-3.7      sigmoid_topk 已实现  必须接受   ✓
```

**一条只会通过的规则不是规则。**

### 这道门当场抓出两个真问题

**① `gate = none` → `true`**（GPT-2）—— 现在被拒，理由写在错误里。

**② `routing = sigmoid_topk` 一直在走 softmax**（Step-3.7）

它的 config 是 `moe_router_activation = sigmoid`、
`moe_router_scaling_factor = 3.0`、**没有 `n_group` / `topk_group`**。
所以 `.by1` 写的 `sigmoid_topk` 是**对的**，是 RUNTIME 里没有这一支，
落进了 `else` 的 `softmax_topk`。

**而 step-3.7 的前向从来没跑过**（180B，这台机器跑不动），
所以这个错一直没人发现 —— 它的 config 46/46、张量 728/753 都是对的，
**因为那些查的是命名和形状，不是算了什么。**

现在补上了这一支（分组那支去掉选组一步）。
**诚实说：实现了，但没验过** —— 没有现成的判卷人对拍它。

### 结果

```
接受 18 个 · 拒绝 4 个（KDA ×2 / SSM / act=gelu_new），零误伤
by1all 46 项 0 失败
三个前向护栏没动：4.470e-07 / 2.384e-07 / 1.788e-07
```

顺手修的：`qk_norm = true` 是旧写法（等价 `per_head`，RUNTIME 确实实现过），
枚举表要认这个别名 —— 否则会误伤四个已验证的模型。

### 这一条现在的位置

以前：描述里写什么都收下，**算不算得出来看运气**。
现在：**要么实现，要么拒绝，没有第三条路。**

---

## 41. GPT-2 的前向跑通了：**1.058e-06，真实维度，对 HF 官方实现**

### 为什么做这个

从头审视看到：`gpt2.by1` 和 `llama-shaped.by1` 编译出的 IR **一模一样**。
而更深的根因是：**语言里根本没有地方能说"我要 LayerNorm"** ——
`ops_of` 里 `{"mech": "RMSNorm"}` 是写死的。
**这不是"默认值选错了"，是"只能 RMSNorm"。**

所以这一轮不是"多一个模型"，是**让"另一代"真的能算**。

### 结果

```
by1gpt2.py   权重搬运 197 / 197
             最大绝对差 2.742e-06   相对 1.058e-06
             [PASS] GPT-2 的前向与官方实现一致
```

判卷人是 HF 的 `GPT2LMHeadModel`：官方 config、**维度全真**
（d=768 层=12 头=12 vocab=50257）、124M 参数。
**权重是随机初始化的** —— 对拍前向不需要真权重，两边喂同一组就够。

### 补上的三样"另一代"的东西

**① `LayerNorm`** —— 不只是"多一个 bias"，**它还要减均值**。
`hparams.norm_kind = layer` 说出口 ✓ 但**两处都要认** ✗
（第一版只改了 `ops_of`，GPT-2 建出来是 24 个 LayerNorm + 1 个 RMSNorm，
而那个 1 就是最终归一化 ✗）。

**② 学习式位置编码（`wpe`）** —— 一张学出来的查表、**加在输入上**；
RoPE 是旋转、**在注意力里**。两种东西。

**③ 无门控 MLP + GELU** —— `gelu_new`（tanh 近似）和精确式不能混。

### 它逼出的四个 bug，全是同一类：**"不存在的机制表达不出来"**

| | |
|---|---|
| `gate = none` → `gate: true` | 无门控说出口没用 |
| 无门控分支硬编码 `F.silu(g)` | `act` 被丢 |
| MLP 硬编码 `bias=False` | GPT-2 的 4×12 个 bias **根本不在我的模型里** —— 而"权重搬运 173/173"看起来是满的 |
| **Attention 无条件 `apply_rope`** | **GPT-2 被静默转了一遍** |

**第四条本来有线索**：用同一组 `wq/wk/wv` 手算，和我的实现差 `1.4e-02` ✗。
**同一组权重、同一个公式，差 1.4e-02 —— 那只可能是多算了一步。**

### 现在的账

```
by1all 47 项 0 失败
和外部参考比过的前向：6 个 shaped + clef-tiny + instella-3b + **gpt2** = 9
**GPT-2 是第一个"另一代"的真实模型**，而且是真实维度
```

---

## 42. 契约自动生成 —— 以及"能推"的边界比我以为的窄

### 先量再动手

```
一份 .by1：  决定 707 行 27%  ·  机械 1649 行 64%
```

而 `qwen3-next-shaped.by1` 只有 **43 行**（0 契约、0 导出），
同样结构的 `clef.by1` 是 **125 行** —— **差的 82 行全是机械的**。

### `by1contract.py`

从机制声明推出契约，形状写**符号**而不是数字。判卷人是
**那些对着真 checkpoint 验过的 `.by1`**：

```
形状一致   140      形状不符    35
只生成有    92      只手写有   104
```

差异只有三个原因：

1. **打包/命名不同** —— clef 存的是融合的 `in_proj_qkv [10240]`，
   生成器给的是分开的 q/k/v。**两个都对**，
   但哪一种**是这个 checkpoint 的习惯，模板推不出来**。
2. **写死数字 vs 符号** —— **生成器这边更好**；
   手写那边写死是因为"机制属性里的符号不参与形状求值"那个已知短板。
3. **`per_expert` 标记** —— 真漏了。已补，但只能以**提示**形式给出。

### 边界

```
能推    有哪些参数、各自什么形状
推不出  这些参数在 checkpoint 里**怎么打包成张量**、叫什么名字
```

**层/全局那一段整个推不出来**：`input_layernorm` /
`pre_attention_layernorm` / `norm` / `attn_norm` ——
**四个名字指同一件事**，纯粹是各仓库的习惯。

### 覆盖面和 codegen 一致（不是巧合）

```
# KDA : KDA —— 这种类没有模板（codegen 也没实现它）
```

**没实现的东西，两边都没有。**

---

## 43. 查实：`1.md` 说有个"逃生舱"，没有

能力清单第 8 条写着「**逃生舱**——允许直接写底层算子」。

全库检索 `escape` / `raw_op` / `inline_c` / `native_op` —— **一个都没有**。
**那条是假的。**

（这条值得单列：1.md 里其余的主张都能对上产物，这一条不能。
而它对 ggml 那一步恰恰是关键 —— 见下。）

---

## 44. 逃生舱 —— 语言表达不了的，可以下探去写，但约束不松

```
mech Weird : Raw { impl = "scale_mix" }
```

旁边放 `raw.py`，里面 `raw_scale_mix(d, attrs) -> nn.Module`。

**它不是绕过检查。** 和别的机制受同样的约束：张量契约照旧对拍、
三个后端照旧对拍、impl 找不到就拒绝。松掉的只有一样：
这段计算不用 by1 能表达的写法来写。**所以它下去之后照样被验。**

判卷人 `by1raw.py` 的三条断言（每条都能反过来试）：

```
① 有 raw.py    -> 能建、能跑，**而且契约里的名字真的建出来了**
② 没有 raw.py  -> 必须拒绝
③ Raw 没写 impl -> 必须拒绝
```

**①里那半句是关键**："能跑"不够。第一版 `raw.py` 写的是
`self.scale = nn.Parameter(...)`，建出来是 `…op1.inner.scale`，
而契约说 `scale.weight` —— **少一段，没有东西告诉我。**

### 一路上撞到四个"报错伪装成别的问题"

```
ATTRS 里没有 Raw              -> KeyError，像"属性拼错了"
token 混合器白名单只认三种      -> "这层没有可用的混合器"，像"这层没写机制"
mixer["attrs"]["window"] 硬取  -> KeyError，像"这层缺属性"
生成的代码里用 CodegenError     -> NameError，像别的问题
```

**每一次"拒绝"都发生了，但理由全是错的。** 而错的理由比不拒绝更坏：
它会让人去查一个不存在的问题。第二条尤其危险 —— **反例测试"通过"了，
其实是因为崩在别处。**

---

## 45. 从产物反推 —— `by1boot.py`

**它不是"帮你写 .by1"，是把"你必须懂 73 个属性"换成"你改到验过为止"：**

```
by1boot   ->  草稿
by1verify ->  哪里对不上
你改       ->  直到全绿
```

**草稿对不对不需要你判断 —— 判卷人判断。**

实测：

```
minimind-3   8 层 Attention + 8 层 FFN，13 类张量
nemotron    23 Linear + 6 Attention + 23 MoE（另有 23 层认不出来 —— 它说出来了）
```

**契约直接从 safetensors 头里读** —— 那是 `by1contract` 推不出来的那一半。

反推不出来的（config 里没有、产物只有名字和形状）：路由方式 /
共享专家 / 门控激活 / `per_expert` 还是打包 / bias / rope 缩放类型。
**这些只能靠 `by1verify` 逼出来。**

抓到一个：按"这一层是什么机制"分桶，于是 `mlp.*` 全被塞进 Attention 桶 ——
**名字对、形状对、分错组**。

---

## 46. 规范化 IR —— 以及它一装上就抓到的三件真事

### 量出来的起点：**IR 不自足**

```
by1codegen.render(info)        要 info，不是 ir
by1c.emit_c(ir, info, params)  要两个 —— **而 info 从头到尾没被用过**
by1exec                        main() 直接收 .by1 路径
```

**想写第四个后端的人，必须先学会 `.by1`。**

### 做了什么

**① `by1ir.py` —— 规格的唯一真相源**

```
版本号   by1-ir: 1.0，必填。大版本不匹配直接拒，不做兼容猜测
四张表   顶层 / 层 / 算子 / 状态：名字、类型、必填、语义
属性表   **每种算子有哪些属性** —— 「IR 里合法的东西」的唯一定义
接口     validate / to_json / from_json / roundtrip
```

**② `ir-spec.md` —— 从那些表生成的，不是手写的**

> 手写的规格会过期，而过期的规格比没有规格更坏：
> **它让读的人以为自己知道。**
> 生成它，那些表被 `validate` 和 `by1all` 真的跑着。

**③ 三个后端各加一条只吃 IR 的入口**

`render_ir(ir)` / `emit_c_ir(ir, params)` / `exec_ir(ir, seq)`
—— `by1c` 那个 `info` 参数**直接删掉**（它从来没被用过）。

**④ `by1irentry.py` —— 判卷人**

```
.by1 -> IR -> **JSON -> IR** -> 三个后端各跑一遍 -> 对拍
```

**中间那一步是关键**：不是"我手里这个 dict"，是**一份序列化过又被读回来的 IR**
—— 那才是"别人给我的 IR"的样子。

### 规格一装上就抓到三件真事

| | |
|---|---|
| **最终归一化缺 `kind`**（248 次） | 我加 LayerNorm 那一次只改了层里的两个 Norm ✗ 和"GPT-2 建出来 25 个 LayerNorm 里的那个 1"是同一个漏 |
| **`qk_norm = 'true'` 混进 IR**（191 次） | 旧写法，而三个后端各自 `not in ("off","",None)` **恰好都对** —— 「恰好都对」是这类 bug 的典型长相 |
| **`Head` 不在闭集里**（19 次） | 我写规格表时漏了 lm_head 那个算子 |

**还有一件是规格表自己的错**：MoE 那条照着 FFN 抄了一个 `bias` ✓
于是 210 条"缺必填项"，而**没有任何东西产生它** ✗
**规格比现实严，报的是假问题 —— 而假问题会把真问题淹掉。**

### 结果

```
合法        19 / 19
往返等价    19 / 19
IR 入口     7 个小模型，三个后端全部只拿 IR 跑通
            1 个后端没实现 Raw —— **明确标成覆盖率问题，不是 IR 入口问题**
            11 个参数太多跳过，**逐个打印**
```

**最后一条很重要**：覆盖率问题混进 IR 入口问题里，真问题会被噪音淹掉。

### 它换来了什么

> **想参与的人不需要学 by1。**
>
> 读规格 → 读懂 JSON → 写第四个后端

**`.by1` 降级成前端之一，而不是唯一入口。IR 才是接口。**

---

## 47. 从产物直接出 IR —— 两条路通向同一个地方

上一件让三个后端**只从 IR 跑** ✓ 但 IR 只有 `.by1` 一个前端 ✗
这一件补上另一半：

```
路 A：  .by1        --check+compile_ir-->  IR
路 B：  config+张量  --by1boot.boot_ir--->  IR
```

### 做了什么

**① `by1boot.boot_ir()` —— 从产物构造规范化 IR**

```
推得出   pos_kind（有没有 wpe）· norm_kind（归一化有没有 bias）· norm_eps
         每层的机制种类 · out_dim · hidden · bias 有无
推不出   qk_norm · q_gate · sink · head_gate · routing · ...
```

**② 推不出的按规格填默认值，并且报出来**

```
产物里看不出来、填了默认值的：72 处
  L0.Attention.qk_norm = 'off'
  L0.Attention.sink = False
  ...
**这些是猜的，不是读出来的。**
```

**③ `by1bootir.py` —— 两条路的判卷人**

比结构骨架，**只比两边都"知道"的**：
路 B 的 `qk_norm='off'` 是猜的，路 A 的是真读出来的 ——
**拿猜的去比真的，比出来的差异不是 bug。**

### 结果

```
gpt2         一致 270 · 差异 0 · 跳过 108    PASS
minimind-3   一致 182 · 差异 0 · 跳过  72    PASS
instella-3b  一致 798 · 差异 0 · 跳过 324    PASS
```

**三个模型横跨两代。** gpt2 的字段名（`n_embd`/`n_layer`/`n_head`）、
归一化（LayerNorm）、位置（学习式查表）全都不一样 —— **而两条路还是汇合了。**

### 一路上抓到的，全是"只认得一代"

| | |
|---|---|
| **config 字段名有两套方言** | 第一版 `int(cfg["hidden_size"])` —— 崩在 `int(None)` 上。现在有 `FIELD_ALIASES`，**认不出来就说清楚是哪几个字段** |
| **`classify` 不认 `attn.c_attn`** | GPT-2 是融合 qkv，名字里既没 `self_attn` 也没 `q_proj` —— 12 层全被判成 Raw |
| **config 不说就从产物读** | GPT-2 没有 `intermediate_size` 也没有 `n_inner` —— 但 `mlp.c_fc.weight` 的形状写着答案。且 Conv1D 是 `[in,out]`、`nn.Linear` 是 `[out,in]`，取大的那维两种都对 |

### 判卷人自己也错了两次（都改了）

1. **拿整条消息当键** —— `guessed` 存的是 `"L0.Attention.qk_norm = 'off'"`，
   而比对时要 `"L0.Attention.qk_norm"` ✗ 对不上，于是"猜的"没被跳过
2. **把"缺"当成一个值去比** —— 规格里有一堆可选属性，只有一边写了不是差异 ✗

**都是同一个毛病：判卷人自己的口径没定清楚，就会报假问题。**
**而假问题会把真问题淹掉。**

---

## 48. 逃生舱下沉一层：IR 引用外部符号

### 为什么

第一层（`Raw`）**要求改编译器** ✗ —— `Raw` 是 `by1codegen.py` 里硬编码的
一个种类 ✓ 加一个机制就得动编译器内部 ✓ 那还是"这门语言见过的机制" ✓

**第二层：编译器不认识机制，只认识调用约定。**

```
mech Weird : External {
  lib     = "ext-demo.so"
  symbol  = "weird_fwd"
  weights = { bias: [d_model], scale: [d_model] }
}
```

或者干脆直接写 IR（**不经过 `.by1`**）—— 示例就是这么做的 ✓

### ABI（定死，不自由发挥）

```
void <symbol>(const float *x, float *y,
              int B, int T, int D,
              const float *const *w, int nw);

w  权重指针数组，**按名字字典序排列**
```

**为什么是指针数组而不是拼成一块**：C 那边权重本来就是分开的缓冲区 ✓
拼成一块要么多一次拷贝，要么要求两边布局一致 —— **而那是约定不是语义** ✓
指针数组 + 排序规则，两边都能自己算出来，不需要额外通道 ✓

### 契约没松

外部算子**照样要声明自己的张量** ✓ 照样被 `by1verify` 对产物查 ✓
松掉的只有一样：这段计算不在这门语言里 ✓

### 判卷人 `by1extdemo.py`：四条断言

```
① 编译一个真的 .so，三个后端各跑一遍（IR 直接构造，没经过 .by1）
② 符号不在 / 库不在 / 不声明张量 —— **三种都必须拒绝**
③ 加一个新机制，**编译器四个文件一个都没动**（比 mtime）
④ IR 合规格
```

**结果：全过。**

### 一路上抓到的（全是"假设"）

| | |
|---|---|
| **C 后端建不了没有注意力的模型** | 它硬要求有 Attention 或 MLA ✗ 于是"纯 Linear / 纯外部算子"的模型建不了 ✗ **而 Mamba 那一类正是没有注意力的** —— 覆盖率缺口里最该能跑的那些被这一行挡在门外 |
| **权重键少了前缀** | `by1c` 的 `order` 要的是完整参数键 ✗ 给了短名 → KeyError，长得像"这个张量没声明" |
| **Tensor 没有 `.ctypes`** | ABI 是 `float*`，PyTorch 那边得先转 NumPy ✗ 报的是 AttributeError，像别的问题 |
| **契约校验放错了地方** | 空 `weights` 竟然通过 —— 那条检查在 `compile_ir` 里，而**直接构造的 IR 走不到那里** |

**最后一条最值得记**：

> **入口变了，校验的位置也得跟着变。**

### 现在逃生舱是两层

```
第一层  Raw       写在 raw.py 里          **要改编译器**
第二层  External  编译器只认 ABI          **不用改编译器** —— 只要一个 .so 和一份 IR
```

---

## 49. 从 gpt2-tiny 一路挖到**护栏自己是瞎的**

### 先量，不猜

```
C 缺的 kind：{'Raw': 1}          ← 只有逃生舱第一层
gpt2.by1   第 0 层的 FFN 形态     ← 无门控 + bias
gemma/laguna/step  注意力用了 kv_tie/head_gate
```

**我原以为"Linear / SSM 在 C 里是空的" —— 那是错的** ✗
`qwen36` 的 `Linear×30` 一直是 ok ✓

### 修的（全是同一类：**一条分支只为一种情况写过**）

| | |
|---|---|
| **by1c 硬拒绝无门控 FFN** | 和 `by1codegen` 当初一模一样的毛病 ✗ **而 GPT-2 正是没有门的那种** |
| **by1c 的激活写死 silu** | 现在按 `act` 分派 |
| **by1c 建不了没有注意力的模型** | **而 Mamba 那一类正是没有注意力的** —— 覆盖率缺口里最该能跑的，被这一行挡在门外 |
| **by1exec 的 `op_ffn` 写死 silu + 不认 bias** | **同一个 bug 在三处，我当初只改了 PyTorch 那一处** |
| **by1exec 没有 LayerNorm** | 只有 `rms_norm` ✗ 算 GPT-2 时 24 个 bias 根本没参与运算，而输出形状完全正常 |
| **by1exec 不知道学习式位置表存在** | `wpe.weight` 整张表没加进去 |
| **by1exec / by1c 都不读 `rope` 开关** | GPT-2 的 `rope=false`，注意力把这个忙转了 ✗ **和 PyTorch 当初那个是同一个 bug** |

### 我自己犯的，也记下来

- **`find()` 的注释是假的** —— 说"按形状兜底匹配"，实现只试了两个名字
- **我的"修法"第一次是错的**：`if not _rope_on: part = 0.0` ——
  于是 `npr = 0` 让 `0 < npr < hd` 变假，**直接掉进 else 分支照转不误** ✗
  **一个"看起来加了开关"的改动比没加更坏：它让人以为查过了。**
- **第二次也错了**：给 `find()` 加"唯一形状匹配"兜底 ✗
  同形状的参数太多，"唯一"也会配到错的那个，而**配错之后检查照样往下走** ✗
  一个"帮你多认几个名字"的兜底，把"找不到就停"变成了"装错也继续" ✗
  → 换成**显式别名表**：不猜 ✓

### `gpt2-tiny.by1`

由 `gpt2.by1` 机械缩小维度得来，结构一个字没改 ✓
存在的理由：`SHAPED` 里**全是有门控 + silu** ✗ 所以"无门控 FFN"
这一类从来没被三个后端一起对过 ✓

**可证伪**：把那行退回旧写法 → 相对 **3.894e-01 [FAIL]** ✓

### 最大的一个：护栏自己是瞎的

改完之后 `by1all` 报 2 项失败 ✓ 我以为是我弄坏的 ✓
查了 HEAD 那一版：**clef-tiny 和 mla-shaped 一模一样地失败，数字一位不差** ✓

**而 `by1all` 一直把它们标成 ok。**

原因：`by1exec --compare` 打印 `[FAIL] 不一致`，然后 **`return 0`** ✗
`by1all` 只看退出码 ✓

> 一直 3.1e-04 和 4.2e-04 的差异，
> **一直在跑，一直在打印 FAIL，一直被当成通过。**

修了两处：

```
① --compare 不一致就返回非零
② **by1all 除了退出码，也看输出里有没有 FAIL**
   —— 判卷人的判据不能只有一条通道：它自己坏了，就没人发现。
```

**现在 `by1all` 是红的，红得对：**

```
!! NumPy mla-shaped.by1   相对 3.100e-04
!! NumPy clef-tiny.by1    相对 4.163e-04
```

**这两个是真失败，而且不是这轮弄出来的。**

---

## 50. 逐算子对拍 —— 以及它把范围钉到了哪一步

### `by1opdiff.py`

`by1exec --compare` 比的是**整个模型**的最后输出：一个算子的差被
后面的层一混，可能变大也可能变小，而且**分不出是哪个算子** ✓
gpt2 那一轮是靠手工做这件事才定位到 `op_ffn` 写死 silu 的 ✓
现在把它固定成工具：

```
每个不同的 (kind, 属性) 组合，喂同一组权重、同一个输入，单独比
```

**"每个不同的属性组合"，不是"每种 kind 一次"** —— 第一版只取第一次
出现 ✗ 于是属性不同的同名算子**根本不会被看到** ✓

### 结果

```
8 个文件，每个 4-5 个属性组合，全部对得上。
```

**关键结论**：`mla-shaped`（3.100e-04）和 `clef-tiny`（4.163e-04）
的差**不在任何单个算子里** ✓ 逐个算子都是 1e-07 ✓

### 钉到了哪一步

```
逐层比到第 8 层：最大 8.543e-07
logits：           4.165e-04      ← **500 倍放大发生在最后一步**
```

最后一步是 `head(final_norm(x))` ✓

**而奇怪的是**：同一个脚本里，手工重做这一步是 **6.7e-07** ✓
走模型的 `forward` 是 **4e-04** ✓ **两条路各自都自洽**
（`② vs ③ = 0.000e+00` ✓）

> **也就是说我两个测量里有一个是错的。**

这件事本身要记下来：这一轮已经出现**三次**"测量工具自己有问题"
（护栏的退出码、`find()` 的注释、以及这次的矛盾）✓
**没查清楚，不装作清楚了。**

### 顺带修的

- `by1opdiff` 原来总是返回 0、没有判定行 ✗ 现在 `[PASS]`/`[FAIL]` +
  非零退出 + 无参数跑一批 ✓
- **参数配对先按名字，不按形状** ✓ 形状兜底在这一轮害过两次：
  同形状的参数太多，"唯一"匹配也会配到错的那个，
  而**配错之后检查照样往下走** ✓

---

## 51. eps：第五次 —— 而这一次是**护栏自己**把它逼出来的

### 矛盾解开了

同一份输入、同一份权重：

```
层输出：                    4.019e-07    ✓
最终归一化（同一份输入）：  2.710e-04    ← **就是这个**
输出头（同一份输入）：      2.974e-04
```

**权重逐个比过：完全一样**（全是 0.000e+00）✓

### 根因

`by1exec.Exec.__call__` 最后那一步：

```python
x = rms_norm(x, w, one_plus=...)      # **没传 eps**
```

于是用了函数默认值 1e-5 ✓ 而 `clef-tiny` / `mla-shaped` 的
`norm_eps` 是 **1e-6** —— 差一个量级 ✓
`llama-shaped` / `gpt2-tiny` 是 1e-5，**恰好就是默认值**，所以一直是绿的 ✓

> **只有偏离默认值的模型才露出来。**

**同一个 bug 在 C 那边也有，而且那个"修复"本身是坏的：**

```python
"#define RMS_EPS %sf" % float(ir.get("rms_eps", 1e-5))
                                      #  ^^^^^^^^ IR 里叫 norm_eps
```

**一个 `.get(不存在的键, 默认值)` 不报错、不警告，
只是永远走默认那条路。**

### 这是"写死的默认值"这个 bug 类的**第五次**

前四次：`rope_theta`（Llama 1e4 / Mixtral 1e6）· YaRN 的 `attention_factor` ·
`rope_scaling.rope_theta` · `truncate` 默认 True ✓

**而这一次是护栏自己把它逼出来的** ✗→✓
上一轮修的"打印了 FAIL 就返回非零"让那两个 3e-04 从隐身变成可见 ✓
这一轮顺着它挖到了 eps ✓

> **护栏瞎着的时候，这个 bug 已经在那儿跑了不知道多久。**

### 顺带修的

| | |
|---|---|
| **`by1c` 里有第四份硬编码的全局参数名单** | `by1exec` 里改了两处，这里漏了 ✗ gpt2-tiny 报 `KeyError: 'pos.weight'`，而那个报错长得像"这个张量没声明" |
| **C 后端现在明确拒绝它没有的能力** | LayerNorm 和学习式位置表 ✗ 原来它会**照常生成** —— 每个 Norm 都发 `rmsnorm()`，**把 LayerNorm 当 RMSNorm 算**（少了减均值和 bias）→ **一个看起来对但算错的模型** |
| **`by1all` 分清「没实现」和「算错了」** | 前者走 `KNOWN`（照常打印、不算失败），后者才是失败 ✓ 混在一起的话，真问题会被覆盖率噪音淹掉 |

### 结果

```
59 项，0 项失败，另有 2 项已知缺口
[全过]
```

---

## 52. C 后端的 LayerNorm 和学习式位置表 —— 缺口关掉了

上一轮把 C 的缺口写进 `KNOWN` 时说"下一步就是它" ✓ 做完了 ✓

### 补了什么

```
① C_HEAD 里的 layernorm()      —— 减均值 + 除标准差 + bias
② 逐层 Norm 按 kind 分派        —— 原来无条件发 rmsnorm()
③ 最终归一化按 kind 分派
④ 嵌入加学习式位置表            —— 原来整张表不存在
⑤ 参数顺序带上 POS / FINAL_NORM_B
⑥ 两个能力开关宏 NORM_KIND_LAYER / POS_LEARNED
⑦ 上一轮那两条"要么实现要么拒绝"的拒绝撤掉
```

**判卷人**：gpt2-tiny **1.049e-07 [PASS]** ✓
三个回归模型 1.5e-07 ~ 4.4e-07 ✓
`by1all`：**60 项 0 失败，已知缺口从 2 条降到 1 条** ✓

### 我自己犯的，两个，都值得记

**一、打印了"已应用"、然后在写盘前崩掉的脚本。**

`_cimpl3.py` 打印了"四处都应用了" ✓ 然后 `ValueError` 崩在最后一步 ✓
**而写盘在最后一步之后** ✗ 于是那四处**一处都没落地** ✗
但我以为它们在了，还接着往下测了三轮 ✓

> **一个打印成功却没写盘的脚本，比静默失败更坏：**
> **它让我把"不存在"当成"存在"。**

改法：**先写盘，再核实文件本身**（不是核实脚本里的变量）✓

**二、`if` 不是 `#if`。**

位置表那段写的是 `if (POS_LEARNED)` ✗ 运行时判断**不会让死分支消失** ✗
`P(POS)` 照样要编译，而 `OFF_POS` 在不需要位置表的模型里根本不存在 ✗
于是 llama-shaped 那六个模型全编译失败 ✓

> **一个"运行时判断"会让死分支活到编译期。**

改成 `#if` 之后全过 ✓

### 这两条是同一类

**我以为我表达了某个意思，而实际生效的是别的东西。**

和 `gate = none` 变成 `true` · 无条件 `apply_rope` ·
`.get(不存在的键, 默认值)` 是同一个家族 —— **这一轮又添了两个成员** ✓

---

## 53. 端到端：真产物 -> IR -> 三个后端 -> 对官方实现

### 这条链在验什么

```
真 config.json + 真 safetensors 头
    │  by1boot.boot_ir()          <- **不经过 .by1**
    ▼
  规范化 IR
    │  三个后端
    ▼
  PyTorch / NumPy / C  ──比对──>  HF transformers 的官方实现
```

**判卷人是真权重 + 官方实现**，不是合成数据 ✓

用的是 `sshleifer/tiny-gpt2`（真架构，极小维度，2.5 MB 权重）✓
它是最好的样本 —— 因为它的 config 里写着 `activation_function = gelu_new`
—— **而 `by1boot` 从产物里看不出来这一点**：名字和形状都不说激活函数 ✓

### 结果

```
③ by1boot 的猜测版：  1.678e-02  [不一致]   <- 猜的代价
④ 改对 gate + act：   1.160e-07  [一致]     <- PyTorch 对上 HF
⑤ NumPy（只拿 IR）：  1.308e-07  [一致]     <- 37 个参数全从真权重装上
   C：编译成功
```

**"改到全绿"这个循环，在真产物上跑通了。**

### 改的是哪两处，以及**怎么知道要改它们**

```
gate  GPT-2 是"没有门"的那种（两层 MLP），而 by1boot 默认按"有门"
      生成 —— 于是多出一个 w3
act   只有 config 里写着
```

**怎么知道的**：那个多出来的 `w3` 在产物里**根本找不到** ——
**"映射里有、产物里没有"就是信号** ✓ 测试把这个写出来了 ✓

### 权重映射：这一半推不出来，得手写

```
c_attn.weight   (d, 3d)   **融合的 qkv，而且是 Conv1D 的 [in, out]**
c_fc.weight     (d, 4d)   转置的
wte / wpe       分开的两张表，而 head 和 wte **绑在一起**
```

**这些是仓库的习惯，不是架构。** 所以 `gpt2_weight_map()` 是手写的 ——
它正是"改到全绿"里"改"的那一部分 ✓

### 我自己犯的，一个

**"配不上就退化成全零"没有报出来。**

NumPy 侧的参数名是 `wq`，而映射表里是 `wq.weight` ✗
**12 个注意力权重整个没进去** ✗ 输出形状完全正常，相对差只有 1e-02 ✓

> **看着像"算法有小误差"，实际是"权重是零"。**

而这个坑在这一轮里出现了**三次**：`by1exec --compare` 的配对 ·
`by1opdiff` 的配对 · 以及这里 ✓
**三次都是"两个后端给同一个东西起了不同的名字"** ✓

现在配不上会**打印出来**，而且会报"零填充了几个" ✓

---

## 54. 版本 + 分发 + README

三样都没有 ✗ 而它们是同一件事的三个面：**让别人能拿到、能装、能看懂。**

### 版本：两个，不是一个

在这之前 `"by1-ir": "1.0"` **硬编码在三个地方** ✗
（`by1codegen` / `by1boot` / `by1extdemo`）—— 三份同一个字符串，
谁也不认识谁 ✓ 改一份忘一份不会崩，**只会造出「声称不同版本」的 IR** ✓

```
IR_VERSION    接口的版本。字段增删、语义改变才动。
TOOL_VERSION  实现的版本。每次发布动；IR 变了必须跟着动。
```

**分开是因为它们回答不同的问题**：

```
用的人问   "这份 IR 我读得了吗"    -> IR_VERSION
报 bug 问  "你用的是哪一版工具"    -> TOOL_VERSION
```

**「加一个 `.by1` 语法」不动 `IR_VERSION`** —— 因为 `.by1` 是前端之一，
不是接口本身 ✓ **这正是把 IR 规格化的意义** ✓

还有：**生成的每一份东西都盖章** ✓ `# by1 0.9.0 / by1-ir 1.0` ✓
`by1all` 先报版本 —— 一份失败的输出要能追回是哪一版跑的 ✓

### 分发

`pyproject.toml` + `VERSION` ✓ 核心**不需要 torch** ✓
`.[verify]` 才拉对拍那一套 ✓

**版本从文件读，不从模块读** —— `attr = "by1ver.TOOL_VERSION"` 会让
setuptools 去导入，而它一旦开始导入就会摸到别的模块，其中几个在模块级
`import transformers` ✓ **打包不该需要跑得动 torch** ✓

### 自己犯的，第四次同一类

改 `_stamp` 那一次：我把 `try/except` 只放进了一个函数，
**而两个函数都用了它** ✗ 于是 `by1diff` / `by1exec` / `by1instella` /
`by1gpt2` / `by1raw` 一起崩，**15 项失败** ✓

> 和第 52 节那两个是同一类：
> **我以为我表达了某个意思，而实际生效的是别的东西。**

**一个补丁的两半分落在不同作用域里 —— 而没有任何东西会告诉你。**

### 环境上的一个坑，不是本项目的（已查实）

```
pip install -e .  ->  No module named 'kernels.lockfile'
```

原因是环境里那个 `kernels 0.17.0` 注册了一个坏掉的 entry point：

```
egg_info.writers  kernels.lock -> kernels.lockfile:write_egg_lockfile
```

而 `setuptools` 的 `egg_info` 会加载**所有** `egg_info.writers` ✓
**一个空包在这台机器上也装不上** —— 验过 ✓

`pyproject.toml` 本身是对的，只是**在这台机器上没法证** ✓
能用的分发方式是 `python by1pack.py`（不需要 pip）✓

---

## 55. 上显卡：A800-80G 那一趟

租了一台 AutoDL（A800-80G / torch 2.8.0+cu128 / 算力 8.0）。
额度用完时停在这里。**做成什么、卡在哪、挖出什么，逐条记。**

### 做成的

```
✓ 显卡验证     9 个模型 CPU vs CUDA，相对 3e-07 ~ 1.2e-06   [PASS]
✓ 远端全量     62 项 0 失败 —— 和本地一样                     [全过]
✓ Instella-3B  真维度前向对官方实现，**0.000e+00**
✓ 真权重 27B   729 个物理名从 emit 规则全反推出来（**零个推不出**）
               55.6 GB 从 ModelScope 下完（13 分钟，72 MB/s）
               27.3 B 参数建成（198 秒）
✗ 前向         **没跑完** —— 差最后一步
```

### 挖出的 8 个 bug

| | |
|---|---|
| **8 个 `by1c` 共用 `cgen/`** | **我加并行时引入的 race** ✗ 症状是"C 后端在 Linux 上算错 6 个模型" ✓ **看起来像未定义行为** ✓ 单独跑一个永远是对的 —— 最误导的那种 |
| **OpenSSH hostbound** | AutoDL 网关不认 `publickey-hostbound-v00` ✓ debug 说 "Server accepts key" 然后 Permission denied ✓ |
| **非交互 SSH 的 PATH 极简** | `python`/`pip`/`gcc` 全找不到 ✓ **一个原因三个症状** ✓ |
| **`glob.glob('gcc')` 不搜 PATH** | **两处各一份** ✓ 加了 Linux 路径**还是不行** ✓ |
| **`by1e2e` 只认 safetensors** | 缓存里是 `.bin` ✓ 报"没有"，而不是"格式不对" ✓ |
| **`os.path.abspath`** | POSIX 相对路径不去当前目录找 ✓ Windows 会 ✓ |
| **cgroup 120 GB** | fp32 建 27B = 108 GB ✗ 加 shm 的 52 GB → OOM ✓ **`free` 说 1 TB，但 cgroup 才是管用的那个** ✓ |
| **`by1load` 的 `parts[2]`** | 是 `"op0"` 不是 `"0"` ✓ 本地小模型没真权重，根本走不到那一步 ✓ |

### 最值钱的一条：换源

```
hf-mirror   这个仓库的权重走 Xet（cas-bridge.xethub.hf.co）   26 KB/s
ModelScope  阿里的源，Qwen 也是阿里的                        72 MB/s
                                                  **差 2700 倍**
```

排查顺序（前三个都不是原因）：hf-mirror 确实在用 ✓ `hf_xet` 已卸 ✓
`config.json` 走镜像自己的缓存 ✓ —— **慢的是权重的字节流** ✓

**55.6 GB：hf-mirror 要 8 天，ModelScope 13 分钟。**

### 下次接着做

`by1real.py` 已经写好并推到能跑的程度 ✓ 下次有额度时：
**下权重（13 分钟）→ 跑它** ✓ 就这两步 ✓

权重在 `/dev/shm`，实例一释放就没了 ✗ 所以要重下 ✓ 但 13 分钟不算什么 ✓

---

## 56. 那 20 块钱买到了什么 —— 逐条算

用户说：AutoDL 花了 20，而这个项目**一共才 40（API）**。
一笔就占了三分之一。**该算清楚哪笔值、哪笔白花。**

| bug | 不花钱能发现吗 |
|---|---|
| **8 个 `by1c` 共用 `cgen/`** | **能 —— 而且是我自己引入的** ✗ 加并行之后**本地跑一次全量就该露**，而我没跑 |
| OpenSSH hostbound | 不能（只有连远程才有） |
| SSH 的 PATH 极简 | 不能（同上） |
| `glob.glob('gcc')` 不搜 PATH | 不能（只有换平台才露） |
| `os.path.abspath` | 不能（同上） |
| `by1e2e` 只认 safetensors | 不能（要遇到另一种缓存格式） |
| **cgroup 120 GB** | **不能** —— 只有真跑 27B 才露 |
| **`by1load` 的 `parts[2]`** | **不能** —— 本地没真权重，走不到那一步 |

**所以：20 块买到 6 个"不换环境就永远发现不了"的东西** ✓
**外加一次真的显卡验证** ✓（那是唯一只有显卡能做的事 ✓）

**而未跑完的 27B 前向** ✗ —— 差最后一步 ✓

### 教训，写下来免得重犯

> **上显卡之前，本地能跑的必须跑完。**
>
> 那个 race 是**唯一一笔白花的钱**，而它恰好是**我自己引入的** ——
> 加并行之后我没在本地跑一次全量就上去了。
> 结果在远端看到"C 后端算错 6 个模型"，**看起来像未定义行为** ✓
> 查了半天才发现是自己覆盖自己的文件 ✓

### 下次的省钱流程

```
① 本地 by1all 全绿（含并行那条路）        —— 免费
② 本地 by1dev（设备无关性）               —— 免费
③ 只把"非显卡不可"的带上去               —— by1gpu.py
④ 真要跑大模型，第一步先用 ModelScope 下  —— 13 分钟，几分钱
```

---

## 57. 改名：两个 step-3.7 用仓库全名

用户的判断是对的：`step-3.7-flash` 会**误导** —— 听起来像官方的
Step-3.7-Flash，而它是社区的**剪枝微调版**（`nerkyor/...`）。
而且两个的命名规范还不一样。

对齐 `refs/` 里已有的约定 `owner_Repo-Name`：

```
step-3.7-flash.by1  ->  nerkyor_Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1
step37-official.by1 ->  stepfun-ai__Step-3.7-Flash.by1
```

**从文件名就能追回是哪个仓库。**

> **上面 57 节之前的老条目里还用旧名字** —— 那是历史事实，不改。
> 日志改了就变成"当初就是这么写的"，那是假的。


---

## 58. 把"还剩什么"量出来 —— 而量它的工具自己错了两次

用户问："我们现在有什么没干完的事。"

**"还剩什么"最容易变成感觉** —— "应该差不多了" / "量化还没做" ——
而这两种说法**都没法验收**。所以新写了一个 `by1debt.py`：
每一类都给出分子和分母。

它第一版自己错了两处，而两处都是同一类毛病：

**① `subprocess` 默认 gbk。** 中文一进来就 `UnicodeDecodeError` ——
而那个报错长得像"脚本坏了"，其实是**没给 `encoding='utf-8'`**。

**② 把"库"当成"没跑到"。** `by1ir` / `by1codegen` 是**被 import 的**，
本来就不该单独跑，而它们占了 22 个误报里的大半。

修完之后分三类：**库**（被 import）· **需要显卡** · **该跑却没跑到**。

量出来的：

```
① 描述覆盖率   refs/ 36 份 config，14 个写了 .by1，**22 个没写**
② boot_ir      classify 认出来但没展开 239 种；认不出 87 种
③ 回归覆盖面   42 个 by1*.py，by1all 跑到 20 个，该跑没跑 5 个
④ 真模型前向   14 个里 4 个 ir.md 里没记录（粗判）
⑤ 数不出来的   llama.cpp 0%、数值量化 0%
```

**而 ① 那条的真分母不是 22。** 22 个 config 按 `model_type` 分组
只有 18 个架构，其中 6 个已有同族覆盖（`qwen3_5` 有 clef 了，
`gpt_oss` 有 120b 了……）。

> **"加大量模型"的真实分母是 12 个新架构，不是 22 个 checkpoint。**

---

## 59. 结构性盲区 —— 而我报了一个不存在的盲区

用户追问："你想想什么东西最容易忽略。"

写了 `by1blind.py` 去量八类结构性的东西。**而它自己错了两处、
第三处最重要：它报了一个不存在的盲区。**

### 我犯的三次同一个错

**① `by1verify` 第 412 行就在做反向覆盖检查。**

```python
412:  covered = generated
413:  uncovered = sorted(k for k in real if k not in covered)
```

注释还写着为什么：*"多模态模型就是这种情况：Gemma 的 vision/audio、
Qwen3.5 的 `model.visual.*` 都在权重文件里，而 by1 不建模它们。
不报的话，一个只看文本主干的描述会显得像是覆盖了整个 checkpoint。"*

**而我的 `by1blind` 工具报了"有"** —— 我**忽略了自己工具的输出**，
写下了"没查"。

**② 我另写了个 `by1cover` 去量"产物里的张量 by1 解释得了多少"，
报了「覆盖率 3.4%」。**

那是假的。因为我**重写了一遍 emit 规则的展开** —— 而真的那份在
`by1verify` 里，复杂得多（走 `layer_out`、带每栈局部序号、
带 `expert_name`、带 `_glob` 作用域）。真数字是 clef 851/1184 = **72%**。

`by1cover` 已删。

**③ `by1blind` 自己误报两处**：把 `by1verify` 的反向检查看漏了；
拿一个宽正则去抓 `pyproject`，把 keywords / classifiers / 脚本名
全当成了依赖，报出 30 多条。

### 修完之后，真正的盲区是这些（都验过）

| | |
|---|---|
| **B. 序列长度** | 最大检查 `seq=8192`，最大 `ctx=1048576` —— **差 128 倍**。RoPE / sliding window / yarn 全是长序列才现形的 |
| **C. 阈值余量** | 最差 8.29e-07，判据 1e-4 —— **余 121 倍，健康** ✓ |
| **D. 数值分布** | 19 处 `randn`，**从没用过真权重的分布** |
| **E. 文档同步** | `ir-spec.md` / `models.md` 都和源头一致 ✓ |
| **F. 并行 vs 串行** | **没有"关掉并行再跑一遍对比"的开关** |
| **G. 依赖版本** | numpy / torch **没钉**，而它们是数值结果的根源 |
| **H. 训练** | ~~`by1train.py` 存在，反向传播一行没验~~ **—— 这条是错的，见第 65 节** |

**模式是同一个：我一直在造新尺子，而不是用已有的那把。**

---

## 60. 想象 vs 数据 —— 16 个死词，量了四次四个数

用户问："我在 wishcoding 吗。"

"wish coding" 指**不懂原理、靠感觉写**。所以我去数了自己写的识别模式：
`by1boot` 里的 `TOWER_PAT` / `AUX_PAT` / `classify`，一共 38 条分支。

**数了四次，四个数：**

```
一  拿整条正则去数        死的分支藏在活的正则里，看不出来    报"活 12"
二  按 | 切开当子串测      (^|\.)visual 剥壳剩 )(visual       报"死 21"
三  剥得更"干净"一点       剥掉的东西更多                    报"死 6"
四  按 | 切开当正则编      (^|\.) 这个**组内**的 | 也被切了    报"死 12"
```

**四次都测的是同一件事。**

### 结论不是"再修一次量法"，是"改写法"

原来模式写成**一整条正则字符串** —— 而"有哪些词"这个信息
**被编码进字符串里了**，想拿回来只能拆字符串，而拆字符串永远拆不干净。

改成：

```python
TOWER_WORDS = [('视觉', ['visual', 'vision', 'vision_tower', ...])]
TOWER_PAT = [(n, _render(w)) for n, w in TOWER_WORDS]
```

**词是数据，正则是渲染。**

> **一个量了四次都量不准的东西，通常不是量法的问题，
> 是"被量的那个东西没有把信息留出来"。**

### 分出来的两张表

**已验证**（34 份张量清单、369966 个真实名字里出现过）：
`visual`(12) `vision`(7) `vision_tower`(4) `merger`(12) `patch_embed`(17)
`image_newline`(1) `vision_model`(1) `vit`(1) `audio`(4) `audio_tower`(3) `mtp`(8)

**未验证**（**一次都没出现**）：
`image_encoder` `multi_modal_projector` `img_` `sound` `speech` `whisper`
`video` `temporal` `nextn` `multi_token` `mtp_layers` `draft` `eagle` `medusa`

**未验证不代表错** —— `nextn` 是 DeepSeek 的 MTP 叫法，`whisper` /
`eagle` / `medusa` 都真实存在 —— **但它代表从没被验证过**，
所以分开放，不混在一起用同样的语气。

**而 `by1pat.py` 自己抓到了一条分类错误**：`vit` 我放在"未验证"里，
而实测有 1 个模型命中。**工具验的是我的分类本身** —— 这就是它该干的。

---

## 61. 十类"不出声"的写法 —— 3 处真危险，6 类误报

`ruff` / `pyflakes` 一个都没装。要装也行，但它们报的东西和
这个项目的病**不一样**：它们关心行太长、import 顺序；
**这个项目怕的是静默失败**。

所以写了 `by1lint.py`，只查"会让错误不出声"的写法。十类，实际查到四类。

### 七处 `except ... pass` —— 分三种

**合理的宽恕**（改成 `contextlib.suppress` —— 一行，而且读起来
就是"我知道它可能失败，我不在乎"）：

```
by1all.py:200      版本行，拿不到就少一行
by1check.py:24     stdout 重配置，老 Python 上没有
by1instella.py:218 删临时文件，删不掉不影响结论
```

**真的危险** —— 吞掉之后**行为变了**：

```
by1boot.py:619   nelem = 0 + except: pass
                 -> `nelem > 2e8` 永远为假
                 -> **一个本该跳过的巨型演示会照跑，然后炸在内存上**
                 -> 而报出来的原因是"内存不足"，不是"护栏没生效"

by1real.py:49    读不到 cgroup 就 return 0.0
                 -> 打印"cgroup 上限 0 GB" —— **那句话是错的**
                 -> 改成 None 表示"不知道"
                 -> **0 GB 和"不知道"是两件事，混在一起两个都不成立**

by1oracle.py ×2  解析失败后 window 保持上层值 -> **静默用默认**
```

### 而 `by1lint` 自己报了六类误报

```
第一版  open() 70 处       —— 在 with 里的也算进去了
第二版  open() 57 处       —— 还是把"先赋值、后面再用"的算了
       sys.exit 47 处      —— 46 处是 `if __name__ == '__main__'` 下的标准写法
       subprocess 4 处     —— `_Ev(...).run()` / `by1ext.call(...)`
                              属性名撞车太容易（`.run` / `.call` 满地都是）
       sys.exit 2 处       —— by1gate / by1raw 整个文件就是脚本，
                              模块层的 exit 是它的出口
```

**假阳性会把真的那几条淹掉。** 一个报 70 条的清单，读的人直接跳过。

---

## 62. 文件读写收成一处 —— 而我 replace 掉了一个 `def` 行

仓库里有 57 处 `io.open(p, encoding='utf-8').read()`。

**在 CPython 上它没问题**（引用计数立刻关句柄），所以这不是 bug 修复 ——
是**把一件事从 57 份实现收成 1 份**，也正是这个项目自己的原则：

> 同一个意思不要两处实现。

而且它顺手解决一个真问题：**57 处每一处都可能忘了 `encoding='utf-8'`**
—— 在 Windows 上就是 gbk 乱码。这一轮已经吃过一次（`by1debt` 的
`subprocess` 没给 encoding）。

### 意图，不是函数

```
read_text / head_text / iter_lines / read_json / read_bytes
write_text / append_text / write_json / write_bytes
```

**三种意图三个名字，不用去数 `read(2000)` 里的那个 2000。**

### 而我踩的坑

补丁脚本里一行：

```python
t.replace('def iter_lines(...):', 'def write_text(...):')
```

**把 `iter_lines` 的 `def` 行换成了 `write_text`** —— 于是
`write_text` 被定义两次，第二次拿到的是 `iter_lines` 的函数体。
**而 `iter_lines` 是生成器**，所以 `write_text(...)` 只是造了个生成器，
什么都没写。

`py_compile` 过了。`by1all` 直接红了两项（`by1raw` / `by1instella`）。

> **这正是 `by1all` 存在的理由 —— 编译过不等于行为对。**

最后重写了整个 `by1io.py`。

---

## 63. 删数据测试 —— 无记忆 agent 逆推出四种 JSON writer

用户提的性质：

> **如果这个仓库 95% 的数据（不是代码）全部被删除，必须能够迅速重建。**
> **这使得数据不依赖硬编码。**

### 怎么测的

按字节挑了 5%（`*.by1` 只有 0.15 MB，纯按大小挑它一个都进不去 ——
所以**每一类至少放一个**），交给一个**完全无记忆的 agent**。

而实际跑的比我设计的**难得多**：它拿到的只有那 10 个文件和一份任务书 ——
**没有 `by1*.py`、没有别的 24 个 `.by1`、没有别的 26 个 `refs/`。**

### 结果：9/9 逐字节对上，合计 1,234,405

| | |
|---|---|
| **7 个 `refs/` 产物** | **从上游推导重建** ✓ |
| **2 个手写文件** | 推不出来 ✗ 按副本字节恢复，**并明确标注** |

### 它逆推出了什么

```
紧凑单行  separators=(",",":")      gpt2 / Nemotron / Laguna
indent=0 + CRLF                     Qwen3-VL / GLM
原样（连尾换行都没有）                gpt2 config
LF -> CRLF（+28 字节，680->708）     DeepSeek config
```

**最阴的一条**：

> GLM 看着像单行，其实是**每个键一行 + `\r\n`**，值写成 `null`。
> 而且**排序只在每个分片块内部做** ——
> 全局排序会把它排错（`model.norm.weight` 会跑到第一个）。

**还有一条判据用得很干净**：Laguna 有两份 GGUF，
只有 `Q4_K_M` 的 `ggml_type` 分布和副本吻合。

**而它全程只读文件头** —— Nemotron 584 MB + GLM 551 MB 的正文一字节没下。

### 它自己的边界判断

> 这两份的内容只能用 by1 那套源码来判对错，而那套源码不在这里。
> **没有它，这两个文件无法从零写对**，所以这一轮：
> **按副本的字节恢复，并如实标出这一点 —— 它们不是"推出来的"，是"抄回来的"。**
> **剩下的 7 个不是。**

**它把"我做到了"和"我抄的"分得清清楚楚。**

### 结论

```
按字节：95% 是"上游产物的再导出"  ->  **可推导重建** ✓
剩下 0.8%（*.by1 + models.tsv）   ->  手写的知识，推不出来 ✗
```

**而依赖链是**：

```
models.tsv   <-  可以从 .by1 头部的 `# by1-repo:` 推出来
.by1         <-  手写的（shaped 类有兄弟文件可参照）
```

**两个手写文件互为支撑。删掉任意一个，另一个还能救它；
只有两个一起删才断。**

> **"可重建"的边界不在"数据有多大"，
> 在"手写的那一份有没有别的东西能交叉验证它"。**

---

## 64. 上 GitHub —— 改历史：逐 blob 扫 881 个对象

上传前扫了一遍，抓到一处**最坏的那种**：

```
✗ rc.py:27   PW = 'gsN5C1F8VfUW'      <- 明文密码
  提交在 53ff36e，**已经在 git 历史里**
```

**删文件不够 —— 进了 git 历史就永远公开。**

### 为什么能改

```
✓ rc.py **没有任何文件 import 它** —— 删了不影响任何东西
✓ 这个仓库**从来没 push 过** —— 所以改历史是安全的
```

### 做了什么

```
① 备份 .git -> .git-before-rewrite（3 MB）
② 确认没有依赖，删 rc.py
③ git filter-branch --index-filter，扫全部 123 个提交
④ 清 refs/original + reflog + gc
⑤ 验证
     git log -S 密码          -> 没有命中
     当前树                    -> 没有
     **逐 blob 扫 881 个对象**  -> 没有命中
⑥ .gitignore 挡住备份目录（它里面有旧历史，就是含密码的那份）
```

**而"逐 blob 扫"是唯一能让人放心的那一层** —— `git log -S` 查的是
提交历史里的差异，`git grep` 查的是工作区。**只有把 881 个对象
全 cat 出来看一遍，才敢说"没有"。**

### 顺带

`by1docs.py`（新写）抓到 13 处失效路径，其中**2 处是真的**：
`README.md` 和 `1.md` 里的 `refs/clef.tensors.json` ——
文件名统一之后它不叫这个了。

**而这一条正是 `by1blind` 里标过、但一直没查的盲区**：
"生成物和源头还一致吗"只查了生成物，**文档里手写的路径从来没查过**。

### 还有一件小事值得记

`pattern-design.md` 的状态是**待审**：

> 目的：把 `pattern` 的元素从「机制名」升级为「机制实例」
> 起因：写 `deepseek-v4.1-flash.by1` 时，20 层全是同一个 `CSA2`，
> 变的只是层上的 `mode ∈ {Full, Reindex, Reuse}` —— 现有 pattern 表达不了。
> **状态：待审。定稿前不要写后面 4 个模型。**

**那是一条卡在"等你答 Q1–Q5"的债 —— 而它挡着 4 个模型。**


---

## 65. 而"更新各个.md"这件事，先抓到了我自己写错的一条

用户说："更新各个 .md（特别是 `ir.md`）。"

于是先去核对现有文档和现实一不一致。第一处就撞上：

```
plan.md:128   ## P3 · 反向 —— ✅ **完成**
ir.md   59    | **H. 训练** | `by1train.py` 存在，**反向传播一行没验** |
```

**两者必有一个是错的。** 查了：

```python
by1train.py:158   loss.backward()               # PyTorch 的 autograd
by1diff.py:341    ra.pow(2).sum().backward()    # 比 by1 的和 HF 的梯度
```

**`by1train.py` 没有自己的反向** —— 它用的就是那套被验过的 autograd。
而 `plan.md` 的 P3 不但写着完成，还给了四个模型逐个参数的数：

```
llama-shaped        39/39 个参数的梯度在阈值内
mixtral-shaped      35/35
gpt-oss-shaped      31/31
qwen3-next-shaped   62/62
```

**所以错的是 `ir.md` 第 59 节里那一行**（它跟着 `by1blind` 一起错了）。

### 这是第 8 次测量错误，而且是最坏的一次

前七次是"多报了几条"或"报了个不存在的盲区"。这一次是
**推翻了一个有记录、有判据、有数字的结论** —— 而我看了一眼就信了，
还把它写进了这份日志。

> **一个说"已经验过的东西没验"的工具，比一个说"没验的东西验了"的
> 工具更坏** —— 因为前者的修法是"再验一遍"（浪费），
> 后者的修法是"什么都不做"（出事）。

### 真正没验的是什么

**训练循环本身** —— 优化器、lr、数据管线。而那不是 by1 承诺的东西：
`1.md` 承诺的是**"前向与反向必须数值一致"**，
而那个有 `by1diff --backward` 在管，四个模型逐个参数对过。

**所以这一条不是盲区，是我把"训练"和"反向"混成了一件事。**

### 顺带：这一轮 `.md` 的账

```
ir.md        2359 -> 2727 行，57 -> 65 条    （补了 58-64，又加了 65）
README.md    301 行    核对过：12 条命令脚本全在，路径只剩占位符 x.by1
plan.md      292 行    **不用改** —— 它是对的，而它错的那条是我
pattern-design.md     **状态仍是「待审」**，挡着 4 个模型（见第 64 节末）
ir-spec.md   226 行    生成的，和 by1ir.py 一致 ✓
models.md    466 行    生成的，和 by1cmp.py 一致 ✓
```

---

## 66. 判卷人被自己的编码判成了失败

**这一次方向是反的**：不是"跳过被当成通过"，而是**一个通过了的东西被记成失败**。

### 现象

中文 Windows（控制台代码页 936）上，`by1all` 有两种表现，都不对：

```
by1all --quick
  !! 模式分类 by1pat.py      （没有判定行）        <- 它退出码 0，判定是"分类全对"

by1all（装了 torch 的那套环境）
  UnicodeEncodeError: 'gbk' codec can't encode character '\ufffd'
  File "src/by1all.py", line 447, in main            <- 连汇总行都打不出来
```

### 根因：三条路，三种编码

```
by1check.py        sys.stdout.reconfigure(utf-8)                <- 只有它写了
by1verify.py       没写，但它用 importlib 加载了 by1check        <- 顺带变 UTF-8
by1pat.py          两条都不沾                                     <- cp936
```

而 `by1all` 是用 `encoding='utf-8'` 去读子进程输出的，于是 cp936 的中文
变成 U+FFFD：**判定行找不到**，找到了也**打不出来**。

**这件事故最值钱的地方：UTF-8 是"碰巧"传过来的。** 一个工具的编码对不对，
取决于它有没有间接加载检查器 —— 这跟编码这件事本身一点关系都没有。

### 修法

编码约定收进 `by1io.force_utf8_stdio()`（**import 即生效**，全仓只此一处），
`by1paths` import 它（几乎每个脚本都 import `by1paths`，覆盖面就全了），
`by1all` 顺手给子进程 `PYTHONIOENCODING=utf-8`。

`errors='replace'` 是判据的一部分：**宁可把编不出来的字符换成 `?`，
也不能让"报告本身"抛异常 —— 一份打不出来的报告等于没有报告。**

### 教训

> **环境细节一旦参与判据，它就不再是环境细节。**

判据（`'逐字段' in out`）是在**父进程**里解出来的，所以"子进程用什么编码输出"
**是判据的一部分**，不是可以交给系统代码页的背景条件。

顺带两条：

- 这个 bug 活到现在，是因为**没有哪条检查在看 by1all 自己的输出**。
  `by1pat` 被记成失败，而它混在 25 条真失败里 —— 又一次"看着像个正常失败"。
- 同一类还有一处：`by1docs.py` 打印 `✓`(U+2713) 在 GBK 控制台上直接崩，
  于是一次"0 处失效"的全绿扫描**以退出码 1 结束**。修的是同一行。

---

## 67. 跳过和通过之间，缺了第三格

护栏原来只有两格：**0 = 通过，≠0 = 失败**。
而"这台机器上验不了"（没有 gcc / 没有 CUDA / HF 缓存里没有那个模型）
两格都装不下：

```
塞进 0    一个没验过的东西看起来像验过了     <- 最坏的一种绿
塞进 1    一个没坏的东西看起来像坏了         <- 然后有人去修它
```

第二种在这个仓库里**一直在发生**：`by1e2e`（缺 HF 缓存）和 `by1extdemo`
（缺 gcc）长期躺在 by1all 的失败清单里。而它们**退出码就是 2** ——
作者本来就想说"跳过"，只是没人读这一格。

### 协议：两条通道，缺一不可

```
① 退出码 == 2        ② 输出里有 [跳过] 这一行
```

**为什么非要两条。** 退出码 2 在这个仓库里是双义的：`by1ir` 参数不全、
`by1gpu` 没显卡、`by1check` 用法不对，都是 2。只认退出码的话，
**一个参数写错的调用会被记成"跳过"** —— 那就是把错误变成沉默。

这条规矩本来就写在 `by1all` 里（"退出码说 ok，但输出里是 FAIL ——
两个通道不一致"），第 67 节只是把它用到第三种结论上，并且**两个方向都成立**：

- 说通过必须有判定行背书（没有 = 最坏的一种绿）
- 输出里有 `[FAIL]` 就能否决一个 0

### 我第一版就踩了一个坑，而它值得单列

我把 `layer_types` 的"官方 config 里没有逐层数组"也记成了跳过。
后果：`by1all` 的 `--tensors` 那一路**根本没要求**比层类型，
于是"形状全中"的 13 个模型**全部被降级成跳过**。

> **一个不在请求范围内的东西，不能给整个调用定结论。**

"不适用"和"没验"是两件事：前者是**没有可比的东西**，
后者是**有可比的东西而我没去比**。混在一起，就会把绿的东西弄成灰的。

### 现在的报告

```
-- C 后端              找不到 gcc —— C 后端这一整段没验
-- by1extdemo.py       [跳过] 找不到 gcc
-- by1e2e.py           [跳过] HF 缓存里没有 gpt2
58 项，0 项失败，3 项跳过，另有 1 项已知缺口
[全过（3 项没验）]
```

**最后那一行改了。** 带着 3 项没验还写"全过"，就是骗人 ——
所以跳过不拦退出码（0），但它必须出现在最后一行里。
跳过本身也单列出来，因为"总数 − 失败数"会被读成"验过的数"。

---

## 68. 97% 是张量个数，而它读起来像覆盖率

`GLM-5.3-Flash` 是这批里最富的一个。它在 `models.tsv` 里、在 `1.md` 的表里
（还带着数字），**但它不在 `by1all` 的 `REAL` 清单里** ——
所以"一条命令看项目还活着没有"从来没跑过它。手动跑第一次：

```
契约声明存在: 37534   名字+形状一致 36289   形状不符 0   缺失 1245
```

### 那 36289 个命中的是什么

不是"97% 的架构"。是 **288 专家 × 42 层 × 3 个投影 = 36288 个专家权重**
（外加 1 个走运的）。它们走的是 `expert_name` 模板，**那条模板是对的**。

而别的张量走 `name` 模板：

```
name = "model.layers.{i}.{scope}{physical}"        <- 少了 language_model.
这个 checkpoint：model.language_model.layers.{i}.…
```

于是**注意力、卷积、mHC、norm、稠密 FFN、共享专家、全局张量 —— 1245 个，
一个都没验**。加上这一节、补上两个 `rename`（`embed.weight` → 
`model.language_model.embed_tokens.weight`，`final_norm.weight` →
`model.language_model.norm.weight`）之后：**37534/37534，0 缺失**。
同族的 `clef` / `Qwen3.8` / `Qwen3.6` 早就写了 `language_model.`，
**只有它漏了** —— 而它是这一批里唯一靠"专家多"把数字撑住的模型。

### 为什么三个月没被发现

三层，每一层都值得单记：

1. **它不在护栏里。** "0 项失败"从来没有被它挑战过一次的机会。
2. **97% 不像失败。** 而旁边那个"契约未覆盖 39819（52%）"在讲另一个故事
   （产物多出来的那些）—— 两个百分比放在一行里，读者会以为覆盖率是 97%。
3. **缺的那 1245 个全是"实际不存在"，形状不符是 0。**
   名字错的症状，和"产物里少了东西"的症状**长得一模一样**。

提交 `50c31ea` 的信息里写着"张量契约 97%，**两处如实记为未完成**"。

> **"如实记为未完成"写在提交信息里，等于没记。**
> 记账要记在**判据会经过的地方** —— 护栏里的 `REAL`，或者 `KNOWN` 清单。
> 提交信息没人会再读第二遍，而 `by1all` 每次都会读 `REAL`。

### 教训

> **张量计数不是覆盖率。** 要问的是"命中的那些**是哪些**"，
> 不是"命中了多少"。一个"97% 全中"的检查，可能一个注意力张量都没验过。

**而且它不该走 `KNOWN`。** `KNOWN` 是给"产物那边的问题"用的
（镜像缺分片那种）；这一条是**描述写错了** —— 描述错了就改描述。

### 顺带

`models.md` 里根本没有 GLM-5.3（`by1cmp` 跳过 KDA/SSM，见第 64 节），
所以这份"从 IR 生成的"对比表里，**14 个真实模型只有 11 个**出现在表里，
而同一份文件的命名表列了 14 个。这条也还在账上。

---

## 69. 语料审计只问了一个方向

`refs/amd__Instella-MoE-16B-A3B.config.json` 是 **29 个字节**，
内容是：

```
Invalid username or password.
```

**它不是 JSON**，已经进了 git，而且从 `2d13b98` 一直躺到现在。

### 为什么没有一条检查看得见它

原来的审计只问一个方向：

```
.by1 的 by1-repo  ->  refs/ 里那两个文件在不在？
```

而**反过来的问题从来没被问过**：

```
refs/ 里的这个文件  ->  它是谁的？从哪来？删了还能抓回来吗？
```

一个方向的问题，只能发现一半的事故。写入口其实是对的
（`by1fetch.py:88-93` 写之前会 `json.loads`，所以现在产不出这种东西）——
**但那是"以后不会再写坏"，不是"已经写坏的会被发现"。**
文件已经躺在那儿了，而没有任何东西在读它。

### 一共 73 个文件，42 个没有出处

`models.tsv` 是唯一的种子（14 个模型）。按它推，`refs/` 里只有 31 个文件
有出处，另外 **42 个（21 个基名）是"不知道哪来的"** —— 它们是探索过的候选
（同族的其它尺寸、上一代、多模态），抓了产物但没写 `.by1`。

于是 README 那句"删掉任何一种都能重建（这是设计目标，不是巧合）"
**对 28 个文件成立，对 45 个不成立** —— 而报告里看不出来。

### 修法

```
refs/SOURCES.tsv     第二个种子：**没有 .by1 的产物的出处**（手写，抓不回来）
by1refs.py           语料审计：能解析 / 有出处 / 不缺 —— 三条都要过
by1all JUDGES        加一条判卷人（语料审计 —— 它现在是每次全量的一段）
by1fetch.py          认第二个种子 —— 于是那 41 个文件真的抓得回来
```

**我没有把 42 个孤儿"删掉"了事。** 它们是这个项目看过什么的证据；
没有出处才是问题，删掉只是把证据消灭掉。所以补的是出处。

### 教训

> **语料审计是双向的。** 只问"描述有没有产物"，就永远发现不了
> "产物有没有出处" —— 而后者才是"删掉能不能重建"这个承诺的真正判据。

> **"能重建"要对每一个字节成立，不是对大多数成立。**
> 28/73 和 72/73 之间差的不是 44 个文件，是**这句话真不真**。

### 还有一条，和上一条不同类

三个 `.gguf-tensors.json` **不是 by1fetch 产的**：它们要从**另一个仓库**
（GGUF 仓）抽头，多一列 `ggml_type`、形状是反序的。其中：

- `poolside__Laguna-XS-2_1` 那个**有出处**（`docs/delete-test-1/rebuild_manifest.py:17` 记着）
- `openai__gpt-oss-120b` / `google__gemma-4-31B` 那两个**当初从哪个 GGUF 仓库抽的，没记下来**

> **"抓不到"和"没记住从哪抓"是两种缺口，要分开写。**
> 前者是产物那边的限制，后者是我的账没记清 —— 而后者更贵，
> 因为它连"去哪找"都答不出来。

这三条现在**不算失败但每次都会印出来**（`重建例外 3 个`），
和 `KNOWN` 一个路子：**藏起来的缺口和没发现过的缺口一样糟。**

---

## 70. 一行 `> 8: continue`，让 README 那张表变成了假的

`by1gpu.py` 里唯一的模型循环，第 86 行：

```python
if len(ir['layers']) > 8:
    continue
```

而 README 那张"用哪张卡"的表点名的那 5 个真模型
（clef 26.9B / qwen38 27.3B / gemma 32.1B / laguna 33.4B / qwen36 35.5B）
**层数全都 ≥ 45** —— 它们永远不会被选中。

而同一份文件往下 20 行写着：

> **按显存选模型，不写死阈值。**

README 里写着：

> 显存不够的会**列出来**，而不是静默跳过。

**三句话说的是三件不同的事，而代码做的是第四件。**

### 为什么它活了这么久：**这段逻辑一次也没被执行过**

`main()` 一进门就问 `torch.cuda.is_available()`，而这台开发机没有卡 ——
直接 `return 2`。于是"挑哪些模型"这段代码：

    · 在仓库里存在
    · 在 README 里被描述
    · **从来没有参与过任何一次判断**

> **没有被执行过的分支，和没有写过的判据一样。**
> 它比"没写"更坏一点 —— 因为它让人以为写了。

### 修法：把"决策"从"副作用"里拆出来

```
plan(free_gb)        纯函数：给一个显存数，返回 (能跑的, 跳过的+理由)
--plan 16            不碰显卡，只印"会挑哪些"      <- 开发机上就能验
params_of(f)         在 **meta 设备**上建模型数参数量 —— 零内存
```

**为什么参数量要在 meta 上数，而不是再算一遍形状。** 原来为了"不建模型"
那个担心，自然的做法是拿 `by1exec.shapes_of` 再实现一套形状 —— 而
`_qk_on` 抄两份让 qk_norm 从 2.505e-07 变成 3.362e-03 那次事故，
就是这个仓库为"同一个意思两处实现"付过的账。

生成出去的权重是**裸工厂**建的（`torch.zeros(d)` / `torch.randn(...)`），
所以 `with torch.device('meta')` 一裹就全落在 meta 上：`numel()` 照样准，
**一个 35B 模型占 0 字节**，而数出来的是 codegen 真正建的那个模型。
（顺手解释了 `by1dev` 为什么抓不到这些裸创建：它的陷阱只在前向期间，
而这些发生在 `__init__`。**它的 docstring 说的正是"前向里"，所以它没说谎。**）

### 结果（`--plan`，4 秒）

```
--plan 16    跑得了 12 个   ← 含 Instella-3B（3.11B，7.8 GB）
                            README 那句"T4 正好把它装进去"**是对的**
             跳过 11 个，每一个带理由
--plan 80    跑得了 17 个   ← **那 5 个真模型第一次出现在"能跑"里**
             clef 52.1GB · qwen38 52.8 · gemma 61.8 · laguna 64.3 · qwen36 68.1
             gpt-oss-120b 219.6GB / step-3.7 368.9GB → "这张卡装不下"
```

### 第二处，同一个病

`by1irentry.py` 的 `SKIP` 里**同样列着那三个失效名字**
（`ling-3.0-tiny.by1` / `glm53.by1` / `nemotron-h.by1`）。
后果不是"少跳过三个"，是**它们掉进了下面那个宽泛的 `except: continue`**：
一个本该被打印的覆盖率缺口，变成了静默消失。

修完之后，`KDA`/`SSM` 那三个模型第一次出现在"后端还没实现"那一节里。

> **改名会留下两种尸体**：报错的那种（看得见），和
> `f in SKIP` 永远为假、然后被 `except: continue` 接住的那种（看不见）。
> **后者更坏**，因为它长得像"这里本来就没有东西"。

---

## 71. "装不上"会保护 bug

这一节记的是四个**互相有关**的东西，共同点是：**它们从来没有被执行过。**

### ① import 一个模块，会印一屏、改 cwd、跑一套测试

```
import by1docs    -> 印一整屏"文档里写的路径还在吗"
import by1gate    -> 跑完 6 个用例，然后 sys.exit()
import by1raw     -> import torch；把 models/raw-escape.by1 改名再改回；sys.exit()
import by1smoke   -> os.chdir(src/)，然后跑 51 个子进程（每个最多 240 秒）
```

`by1smoke` 那个最狠：**`import by1smoke` 会启动一套多分钟的测试，
并且改进程的当前目录。**

### ② 而它们靠缺陷互相抵消

`by1smoke` 的判据是"把每个 `by1*.py` 都跑一遍"，而它的 SKIP 里只有
`by1io.py` —— **它不排除自己**。

它之所以没有无限递归，是因为它自己**没有 `if __name__` 守卫**，
于是被那段"没有入口的就算库"的检查跳过了。

> **我一加上正确的守卫，它就 fork 自己**（每一层再跑 51 个子进程，
> 每层的超时是 240 秒）。修完必须立刻把 `by1smoke.py` 写进它自己的 SKIP。

**一个靠两个缺陷互相抵消才能工作的系统，修好其中一个就炸。**
（顺带：它以前**无论发现什么都返回 0** —— 现在有判定行和退出码了。）

### ③ 包从来装不上，于是保住了两个更老的 bug

`pyproject.toml` 的 `py-modules` 只列了 **14** 个模块，而 `src/` 里有 **53** 个。
没列的 39 个里包括

    by1paths      <- `by1 --version` 的第一行就要它
    by1io         <- 几乎所有模块都要它
    by1refs / by1skip / ...

也就是说：**装是装上了，一 import 就 ImportError。** README 里那句
"`pyproject.toml` 本身是对的"从来没被验过 —— 因为这台机器上
`pip install -e .` 一直失败（那条笔记说的是**全局解释器**上的
`kernels` 坏 entry point；项目自己的 venv 里根本没有 `kernels`）。

把 39 个补上之后，第一次真的装上（`Successfully installed by1-0.9.0`），
**于是立刻暴露两个从写下那两行起就坏着的入口**：

```
$ by1-check hello.by1
TypeError: main() missing 1 required positional argument: 'argv'
$ by1-verify ...
TypeError: main() missing 1 required positional argument: 'argv'
```

setuptools 生成的启动器是 `sys.exit(main())` —— **不带参数**，
而这两个 `main` 的签名是 `def main(argv)`。

> **"装不上"会保护 bug。** 一个从来没被执行的安装路径，
> 会把"入口签名写错"这种一眼能看出的错，保养成"一直存在的状态"。

### ④ 挂住的子进程，和"还在跑"长得一样

`by1all` / `by1c` / `by1extdemo` / `by1e2e` 里的 `subprocess.run` **全都没有上限**。
一个等 stdin 的编译器、或者一个死循环的生成程序，会让整轮验证
**永远停在那里** —— 而"永远停在那里"在日志里和"还在跑"一模一样。

现在有上限（`by1all` 600 秒，可 `BY1_TIMEOUT` 调；编译器 300 秒）。
**超时算失败，不是跳过**：跳过是"这台机器上验不了"，挂住是"它坏了"。
两者混起来，一个死循环会变成一条安静的缺口。

### ⑤ 只写不删的临时目录

`by1oracles` 每跑一次漏 10 个 `by1oracle-*` 目录，从写下到现在累计 **264 个**
（我清掉了）。它把 `mkdtemp` 的路径当第 4 个返回值带出去，
而**没有任何调用方删它** —— 因为写完那一刻就再也不需要它了。

> 一个只写不删的临时目录，跑一万遍就是一万个。

### ⑥ "这台机器上没有 git"不是"可疑"

`by1debt` 的 `sh('git', ...)` 没有做 PATH 查找。git 不在这台机器的 PATH 上，
于是它以一个 `FileNotFoundError` 的栈结束，而**冒烟测把它记成
"可疑（有 Traceback 而且不像环境问题）"** —— 它既不可疑，也不是 by1debt 的 bug。

**"环境里没有这个工具"和"代码坏了"要分开** —— 和第 67 节那条是同一件事，
只不过那次是退出码，这次是异常。

---

## 72. 文档里的 ✓/✗ 没有判卷人

### 一条假了三个月的 ✗

`1.md` 的能力清单第 8 条写着：

> **逃生舱** ✗ **不存在。** 全库检索 `escape` / `raw_op` / `inline_c` —— 零。

而它一直是有的：两层，机制种类叫 `Raw` 和 `External`，
判卷人是 `by1raw.py`（三条断言）和 `by1extdemo.py`（IR → `.so` → C 链接），
**两个都在 `by1all` 的 JUDGES 里**。

错误不在结论，在**方法**：

> **搜错名字得到的零，不是"不存在"，是"我没找对"。**

`escape` / `raw_op` / `inline_c` 是我**以为**会用的名字；
真实实现用的是 `Raw` / `External` —— 那正是这个仓库反复记的那一类错
（"名字一样，含义不同"的反面：**含义一样，名字不同**）。

### 为什么它能活这么久

因为**文档是唯一没有判卷人的产物**：

```
models/*.by1      by1check + by1verify + by1all        <- 有判卷人
refs/*.json       by1refs（语料审计）                    <- 这一轮刚加上
文档里的路径       by1docs                               <- 有（这一轮进了护栏）
文档里的**主张**   ——                                    <- **没有**
```

`by1docs` 只查"路径还在不在"。它查不出"这句 ✓ 是不是真的"、
"这个数字还对不对"、"这条限制是不是早就实现了"。

### 这一轮怎么处理的（以及哪些是权宜）

| 类别 | 处理 |
|---|---|
| **数字**（项数、行数、节数、参数量） | 改成**从工具的输出抄**，并注明是哪一次跑出来的 |
| **和工具行为绑定的表**（显卡容量、重建方式） | **删掉手写表，改成指向工具**：`by1gpu.py --plan <GB>`、`by1refs.py` |
| **清单类**（`docs/INVENTORY.txt`） | 重写，标注"和 `by1all` 的输出对齐" |
| **历史文件**（`history/1.md.bak`） | 顶部加"已被取代、不要照着做"，并点出它错在哪几条 |
| **主张类**（✓/✗） | 逐条核对后改正。**这一类还是靠人** —— 见下 |

### 还欠着的一条

**"主张"仍然没有判卷人。** 这一轮改掉的几条（逃生舱 ✗、`C 后端没有 MLA`、
`pip 装不上`、`9 个前向`）都是**人读出来的**，不是工具抓出来的。
下一件该做的不是"再写一份文档"，而是**把能机检的主张变成断言**：

```
"63 项 0 失败"        -> 跑 by1all，比输出
"by1c 只拒三样"        -> grep by1c 的 [不支持]，比个数
"refs 全能重建"        -> 已经是 by1refs 的一条判据了
```

**没做的那部分要写下来**，否则下一轮又会以为"文档已经有人管了"。

---

## 73. 抄进文档的数字，寿命比文档短

第 72 节刚把 `README.md` 的 ④ 改成当时的输出：

```
63 项，0 项失败，4 项跳过
```

并注明"要看到 `0 项跳过`，需要 gcc / HF 缓存里的 gpt2 / 一张显卡"。

**一小时后，三样里的两样到位了。** 用户说"装 gcc 那个小包"，
于是：

```
winget install BrechtSanders.WinLibs.MCF.UCRT
  -> %LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders*\mingw64\bin\gcc.exe
     （**正是 `by1all.GCC_GLOBS` 一直在找的那个位置** —— 那段 glob 写下来的时候
      就已经知道该装哪个包了，只是机器上一直没有）
0.1 MB 的 sshleifer/tiny-gpt2 进 HF 缓存   -> by1e2e 也能跑了
```

数字变成 **`71 项，0 项失败，1 项跳过`**（只剩"没有显卡"那一条）：

- **C 后端那 9 项第一次真跑**：`C↔NumPy↔PyTorch` 三方一致，
  最大绝对差 `2.990e-06`（qwen3-next）以内；
- `by1extdemo` 真的编了那个 `.so` 并链接进 C；
- **`by1e2e` 第一次通**：真产物 → IR → 三后端，对官方实现
  PyTorch `1.886e-07` / NumPy `1.308e-07`。

**我抄错了吗？没有。是那个数的寿命只有一个小时** ——
它依赖"这台机器上装了什么"，而那件事随时会变。

### 两种处理，这一轮都用上了

1. **能指向工具的，指向工具。** 显卡容量表 → `by1gpu.py --plan <GB>`；
   refs 重建 → `by1refs.py`。工具的输出永远比抄来的数字新。
2. **不能指向工具的（样例输出），把前提一起写。** 现在 ④ 那里是一张表：
   *缺什么 → 哪几项变成跳过*。读者知道它在什么情况下不再成立。

> **抄一个数字进文档，等于把它成立的条件藏起来。**
> 要么把条件一起写，要么别抄。

### 顺带一条同类的

装完 gcc 之后，**"环境没装 X"那条跳过自己消失了** ——
和 `KNOWN` 清单一样：**跳过清单会缩短**。

**这正是要把它单独印出来的理由**：它会动。如果当初把"跳过"静默按失败计
（第 67 节之前就是这样），那 9 项 C 检查会一直躺在失败清单里，
而没有人能从"失败"里看出"其实只是没装编译器"。


---

## 74. 把 `check()` 劈开 —— 以及"量了才知道计划错了"

用户指出的坏味道：

> `check()` 是 702 → 1954。这是全仓库唯一的、真正的结构性坏味道：
> 语法、语义、张量契约、调度展开全塞在一个函数里。

量下来是 **1243 行**（`by1check.py` 的 59%），里面有 **26 个局部函数**。
用户的诊断还有下半句：

> IR 里同时住着数学语义、内存布局、权重命名三套词汇表，而唯一有
> 外部判卷人（`refs/`）的只有第三套 —— **这就解释了为什么数值错能
> 活过一个 160/160 全绿的检查。**

这个仓库里正好有三次同形状的事故可以印证：`eps` 第六次（三个后端算出
三个不同的 q/k，相对差 1.05e-05，**低于 1e-4 判据**）、`apply_rope`
交错（两边"一致地错"，四个模型全绿）、`gpt2.by1` 和 `llama-shaped.by1`
编译出**一模一样的 IR**。三次都是**名字和形状的判卷人全绿，数学是错的**。

### 第一步：导出层

13 个 `gen_*` / `name_*` 搬进 `by1export.py`。形状是 `make(*, 依赖...)` ——
它们原来是闭包，引用 `check()` 的 7 个局部变量和 4 个解析函数；
让它们闭包在 `make()` 的参数上，**函数体一字未改、缩进一格未动**。

    check()      1243 -> 1064 行

**判据是逐字节的**：搬之前把 26 份 `.by1` 的输出存成基线（44 KB），
搬完对拍。这一条让"纯搬运"从主张变成了能验的东西。

### 第二步：量了才知道计划错了

计划写的是"剥出解析/取值层（`_coerce` · `_ltype_of_attrs` ·
`parse_rope` · `resolve_field`）—— 这 4 个是导出层与语义层之间唯一
不干净的接缝"。

**量完发现那一层不存在。** `resolve_field` 依赖**全部 13 个导出函数**：

    if v in ("schedule", "layer_types"):     return gen_layer_types()
    if v in ("position", "rope_parameters"): return gen_rope_parameters()

它把 `emit { config = ... }` 的字段值解析出来，而那些名字**直接就是
生成器的入口**。所以它们不是上游，**它们就是导出层**。

真正那条接缝不在"解析 vs 导出"之间，在：

    **导出**（一切都是 config 字段相关）  vs  **语义/契约**（一切都是 .by1 相关）

搬了 240 行（`_coerce` · `_ltype_of_attrs` · `_rope_params` ·
`parse_rope` · `lookup_attr` · `resolve_field`），留了 74 行
（`_is_aux` · `heads_of` · `resolve_attrs` · `key_of` · `attached_at` ·
`_flat_rows` · `eval_with`）—— 后两个算的是**张量形状**，那是契约，
不是 config 字段。

    check()      1243 -> 833 行   （两步合计 -410）
    by1check.py  2102 -> 1693 行
    by1export.py         513 行（19 个函数）

判据：26 份输出**逐字节相同**，而且 `编译 62 · import 57` 和
`by1lint 57 个文件` 都没降。

### 这一步最值钱的东西：两个检查器在静默丢文件

第一步搬完之后门报了 PASS，但数字不对：

    by1fast   编译 56 -> 23 个
    by1lint   "22 个文件"（原来 55）

两个都是 `src/` 平铺时代写的 —— `os.chdir(HERE)` + `listdir('.')`，
或者 `(HERE,) + join(HERE, 子目录)`，而 `HERE` 现在是 `src/checks/`。

**覆盖掉了一半还报绿。** 这正是这个仓库一直在猎的那类假绿，
发生在猎人自己身上。改成从 `by1paths.SRC` 走之后，恢复覆盖当场
又抓出三处：`by1fast` 的 import 子进程 `cwd` 错（顶层 12 个模块
一个都 import 不了，而它们恰恰是被所有人 import 的底座）、
`by1paths.py` 两处 `open()` 没用 `with`、以及一行
**`import by1paths`（自己 import 自己）**。

**教训：分层重构之后，"输出不变"还不够，还要看"扫了多少个文件"。**
前者保证搬对了，后者才保证没把检查本身弄哑。这一条现在进目标里当硬判据。

### 搬的过程里踩的四个坑（形状都一样）

**① 整块删，吞掉了不该删的。** 那 13 个函数**不是连续的** ——
1385-1658 里还夹着 `_rope_params` / `parse_rope` / `lookup_attr`。
症状 `NameError: name 'parse_rope' is not defined`。现在删之前有断言：
每一段里不许夹带别的函数。

**② 插入点放早了。** 插在第一个导出函数原来的位置，而 `resolve_field`
要到 1806 才定义完。同一个 `NameError`。量"第一次用"时又错一次：
先用 `ast.walk` 量，把**嵌套函数体里的调用**也算进去了 ——
那些是延迟执行的。改成只扫 `check()` 的直属语句才对。

**③ 搬走了但没绑回本地名字。** `resolve_field` 进了 `make()`，
而 `check()` 主体还在直接调它。

**④ 加了参数、忘了加返回值。** 补了解包之后变成
`KeyError: 'resolve_field'` —— 加了 6 个函数体和 6 个参数，
却忘了往 `return {...}` 里放。

**四个坑里有三个是同一件事：搬运要同步三处（删掉定义、绑回名字、
接上参数/返回值），我每次只做对两处。** 而四次都是 26 份对拍当场
抓住的，**没有一次是"看起来通过了"** —— 这就是先建基线的价值。

### 第三步：不是搬运，是设计

`check()` 主体剩下 **96 条顺序语句、833 行、60 个名字一路传下去**，
注释里有 18 个分段（hparams · schedule 命名子序列 · stacks ·
双真相源 · attach 目标 · state 块 · position 块键 · optimizer ·
interop 栈名 · 机制形状 · mrope 不变式 · memory 表 vs params ·
张量契约实例化 · emit lowering 规则 · 状态尺寸 · 未填占位符 · 汇报）。

**它不能像前两步那样"原样搬"** —— 因为"语义层的接口是什么"这个问题
还没有答案：输入是解析好的 AST 块，输出是 `layer_seq` / `mechs` /
`contracts` / 一堆 config 片段。**这条边界是要设计的，不是量出来的。**

所以第三步该从**定义接口**开始，而不是从切行号开始。

### 而试着直接搬它，连败五次 —— 这五次本身就是结论

量完数据流，挑了看起来最好的一段：行 862-1018（九个块的一致性检查，
123 行，只读 7 个 `check()` 局部变量）。**"够大又够干净"的那个。**

然后连败五次：

| # | 错在哪 | 症状 |
|---|---|---|
| ① | 模块级名字漏了**注解里**的（`Blk` 只出现在 `def heads_of(blk: Blk) -> ...`，而**注解在 def 时求值**） | `NameError: Blk` |
| ② | `IN = reads - writes` —— **又读又写**的名字（`ch` / `k` / `v`）既没传进来、又被当成输出 | `NameError: ch` · `UnboundLocalError` |
| ③ | 拼 kwargs 时**忘了收尾的 `)`** | `SyntaxError` |
| ④ | `IN = reads ∩ before` —— `x` / `c` / `k` / `m` / `v` 是**循环残留**，在调用点上未必绑着 | `NameError: x` |
| ⑤ | 折行正则**从单词中间切**（`ead_dim_of`） | `IndentationError` |

**① ② ④ 是同一个问题的三次逼近**：这一段要接哪些参数，本质是
**到达定义（reaching definitions）** 问题，不是"读过/写过"这种集合运算。
最后一次用的近似是"按语句顺序，看这一段有没有在赋值它之前读过它" ——
已经很接近了，但还没到。

**③ ⑤ 是另一类**：我**生成**代码去搬代码。搬运脚本自己出 bug，
而它出的 bug 和"搬运逻辑"无关，纯粹是字符串拼接和正则边界。
**该手写的地方不要生成。**

而五次全部是 26 份逐字节对拍当场抓住的 —— **没有一次是"看起来通过了"**。
**这就是第一步那个基线的全部价值**：它把"我以为搬对了"变成了
"26 个反例没有一个同意"。

### 而"先修泄漏"这个结论是错的 —— 因为那次测量本身就是错的

上面写到"**先修泄漏，再谈分层**"，理由是"主体有跨段泄漏的循环变量"。
量出来 16 个。**下一轮去修的时候发现：那 16 个里大部分是假警报。**

原因是我的分析**没有给推导式建作用域**：

    _items = [x.strip() for x in split_top(...)]      # 810 行
    segs  = [x.strip() for x in s.split("->")]        # 935 行

`x` 在这两处都是**推导式自己的**循环变量。不建作用域的话，它看起来
像"810 段漏给 935 段的值"。同一类假警报还包括 `k` / `v` / `r` / `n` /
`ln` / `lst` / `a` / `b` / `m` / `s` / `nm` / `ch` —— 其中六个集中在
"汇报"那一段，而它们全是 `{n: b.mtype for n, b in mechs.items()}`
这种推导式自己的变量。

**给推导式 / lambda / 嵌套函数各建一层作用域之后重算：
`check()` 里真正的跨段泄漏是 0 个。**

**一个报 16 处、其中大部分是假的工具，比一个报 0 处的工具更坏** ——
它会让人去修不存在的东西。而这次差一点就那样了。

### 真正的障碍是"确定赋值"，不是"泄漏"

修好作用域之后再试，失败换了一张脸：

    UnboundLocalError: cannot access local variable 'k'

`k` 在 752 行绑：

    for k, v in sb.assigns.items():
        named[k] = ex

**而 `sb` 可能是空的** —— 循环一次都不跑，`k` 就没绑。
所以把它当参数写成 `k=k` 会直接炸。

于是把输入分成"一定绑着"和"可能绑着"（后者带 `=None` 默认值、
调用方用 `locals()` 过滤着传）。**再试，又换一张脸**：

    UnboundLocalError: cannot access local variable 'ents'

`ents` 在 1009 绑定、1010 读，紧接着 —— 按理不该出事。

### 七次之后的结论：**枚举名字这条路本身是错的**

    1  模块级名字漏了注解里的（Blk）        NameError
    2  IN = reads - writes，又读又写的没传    NameError / UnboundLocalError
    3  拼 kwargs 忘了收尾的 )                SyntaxError
    4  IN = reads ∩ before，推导式变量被当成输入  NameError: x
    5  折行正则从单词中间切（ead_dim_of）       IndentationError
    6  没给推导式建作用域                     UnboundLocalError: k
    7  没建模确定赋值                         UnboundLocalError: ents

前五次是**我的分析不对**，第六次修好了作用域（这一步是对的、
而且推翻了上一轮的结论），第七次又撞上确定赋值 ——
**而每次修完都冒出一个新的名字。**

**这不是"再修一次就好"的形状。** 枚举 IN/OUT 名字要求一个
**完整的数据流分析**（作用域 + 确定赋值 + 分支合并），
而我七次都是在用集合运算逼近它。

### 该走的路：不要枚举名字

**把要传的东西变成一个对象，就没有 IN/OUT 名单了。**

    def check_blocks(st):          # st 永远绑着
        st.mechs  st.rep  st.scope ...      # 属性读，缺了才 AttributeError

- **没有名单** —— 于是"漏了哪个名字"这个失败模式整个消失
- **确定赋值不再相关** —— 对象在调用的那一刻就存在
- 代价：段内每个变量引用要加前缀（`mechs` → `st.mechs`），
  是**机械但成百处**的改动 —— 而这一层可以用对拍守住

**下一个该做的是这个，不是第八次枚举。**

（这两轮都没有留下代码改动 —— 十二次尝试全部撤干净。留的是这两条结论：
① 上一轮那个"泄漏"结论是测量错误；② 枚举名字这条路走不通。）


### 第四次尝试：九次之后，看清了根本原因

Round 4 换了策略：不碰 862-1018 那个大段（11 个输入），改挑小的。
修好分析（作用域 + 确定赋值 + "段内先赋后读不算输入"）之后重新排序：

    emit lowering   85 行  输入 4（class_rows, contracts, mechs, rep, scope）
    状态尺寸         56 行  输入 6

挑 `emit lowering`。九次尝试的过程：

| # | 改了什么 | 26 份对拍的结果 |
|---|---|---|
| 1 | 关键字参数后面跟了位置参数 | **探针拦下，没写进文件** |
| 2 | 收窄到顶层作用域 | `NameError: contracts` |
| 3 | 读按任何深度收 | 10 处失败（全是形状/探针模型） |
| 4 | 允许 IN 和 OUT 重叠 | 24 处失败（更差 —— 临时变量被当成输入） |

**第 3 次是 26 → 10，第 4 次又回到 24** —— 集合运算在**震荡**。

### 根本原因（这次是定理，不是猜测）

**Python 函数的局部名在编译期定死。** 所以：

    把一段代码搬进函数      -> **必须**有一份静态的名字清单
    搬进命名空间 / exec     -> 不用列举

而清单要正确，需要一个完整的数据流分析：**作用域 + 确定赋值 + 分支合并**。
21 次尝试里，我每一次都是在用集合运算逼近它 ——
每次修好一层，就冒出下一层（`Blk` → `ch` → `x` → `k` → `ents` → `contracts`）。

**`check()` 是一段 833 行的顺序过程，它和"函数"这种编译期定死局部的
结构本来就不合。** 前两步能成，是因为那 19 个是**闭包**：闭包的自由变量
由语言算好了，量一下就有。主体没有那个东西。

### 所以只有两条路

**甲 · 整段转成命名空间**（`exec` 或一个 class），`check()` 变成
"几步 stage 串起来"。**不需要任何名字清单** —— 这才是它绕过死结的地方。
代价：833 行机械改写（`name` → `st.name`），但可以逐段做、
每一段都过对拍。

**乙 · 放弃分层，直接去做那件让分层有意义的事** ——
给数学语义找外部判卷人（接 llama.cpp 的 ggml 后端）。
分层本来只是手段：`refs/` 判不了数学语义，这才是"数值错能全绿通过"
的原因；而一个**独立实现**能判。

**（Round 4 结束：九次尝试全部撤干净，check() 仍是 833 行，门全绿。）**


---

## 75. 数学语义的判卷人**早就在**，只是站在了错的地方

目标里那条是"给数学语义找第一个外部判卷人"，候选写的 llama.cpp。
这一轮先去量现状，结果推翻了那个前提。

### 一、llama.cpp 直接编不了（先说这个，因为它是个死路）

`llamacpp/` 13 个文件是**模型代码摘录**，`ggml.h` / `ggml.c` **不在**。
而它们 `#include "ggml.h"`、`"models.h"`、`"llama-impl.h"`、`"llama-ext.h"`
—— 一个都不在。**编不了。**

（`by1emit` 的头注写着"没有 C 编译器所以不写 ggml 代码"，
那句的前半段已经过期 —— gcc 16.2.0 在。**但它真正的拦路虎是
"没有 ggml 本体"，而那句头注没写这一条。**）

### 二、外部判卷人早就在，而且覆盖三个后端

    by1e2e.py    真产物 -> IR -> **三个后端** -> **HF transformers 官方实现**

跑出来的：

    PyTorch 后端 vs HF：相对差 1.160e-07   [一致]
    NumPy   vs HF：  相对差 1.308e-07   [一致]
    C：C<->NumPy 相对差 1.222e-07   [一致]

所以"数学语义没有外部判卷人"这句话**不准确**。准确的说法是
**它的覆盖面是 6/26**，以及下面这一条。

### 三、而 `by1blind` 早就算出了真正的盲区

那个脚本里有一节 C 叫「阈值余量」，还有一节 B。它写的是：

    各脚本用的 --seq：[16, 64]
    模型声明的 ctx：最高 1048576
    **RoPE / sliding window / yarn 全是长序列才现形的东西**

**这一条早就量出来了，而没有任何东西在读它** —— `by1blind` 因为慢
（57 秒）被排除在 `by1fast` 之外，于是它报的东西只有手动跑才看得见。

### 四、把判卷人搬到长序列上：误差涨 7 倍然后持平

    seq     PyTorch vs HF   NumPy vs HF    C<->NumPy
    16      1.160e-07       1.308e-07      1.222e-07
    64      3.322e-07       2.928e-07      1.788e-07
    256     3.322e-07       9.266e-07      9.194e-07
    1024    3.322e-07       9.264e-07      9.192e-07

**不是结构性错** —— 256 到 1024 持平，是 fp32 累积。

但含义是实在的：**在 `--seq 16` 上标定的余量，比看上去少 7 倍。**
一个在短序列上 1e-5 的模型，到长序列就是 7e-5，**贴着 1e-4**。

而 `eps` 第六次那次的相对差是 **1.05e-05** —— 同一个数量级。
**它不是"判卷人没看见"，是"判卷人在一个误差被压小 7 倍的地方看的"。**

顺带：NumPy 在长序列上比 PyTorch 差 **2.8 倍**（9.27e-07 vs 3.32e-07）。
两个后端用同一份 IR，差的是累加方式和归约顺序。

### 五、改了：`by1e2e` 默认 `--seq 16` -> `512`

    代价     13.6 秒 -> 15.2 秒      （32 倍长，多 1.6 秒）
    结果     [PASS]，三个数和我实验里量的一致

### 六、而这也修正了 `docs/ggml-plan.md` 的立论

那份计划开头写的是"`refs/` 判得了名字和形状，判不了数学 —— 所以才需要
一个独立实现"。**前半句对，后半句跳了一步**：判数学的那个判卷人
（HF）一直在，覆盖三个后端。

所以 llama.cpp 的价值**不是"从无到有"**，而是：
**它是第二个独立实现**。两个独立实现同时错在同一个地方的概率，
比一个低。

**这个理由比原来那句弱，但它是真的。** 而这一轮改的那一处
（`--seq 16` -> `512`）**比接 llama.cpp 便宜三个数量级，
而它对着的正是这个仓库真栽过的那一类错。**


---

## 76. 判卷人把"我判不了"说成了"错了"

上一轮量出外部判卷人（HF）一直在。这一轮去看**它实际上判了几份** ——
把 `by1diff`（HF · torch · by1C 三方对拍）对 26 份模型逐个跑：

    ok 4 · skip 5 · timeout 3 · fail 14

**14 个 fail 里一大半不是模型的错。** 消息长这样：

    [注意] 参考实现里少了 72 个张量      Instella-3B
    [注意] 参考实现里少了 24 个张量      gpt2.by1
    [注意] 参考实现里少了 16 个张量      minimind-3

`gpt2` 是 LayerNorm + 学习式位置 + Conv1D，而 Llama 是 RMSNorm + RoPE +
Linear。**拿两个不同的东西对拍，比的不是数学，是"它们本来就不一样"。**

而它为什么会拿 gpt2 和 Llama 比 —— 因为那一支是**无条件的 `else`**：

    if has_linear:   Qwen3Next      # 假设 FFN 一定是 MoE -> clef 崩 KeyError
    elif is_gptoss:  gpt-oss
    elif is_moe:     Mixtral
    else:            Llama          # **凡是没被前面认出来的**

### 假绿的反面是假红

这个仓库记了很多次"假绿"（覆盖掉一半还报 PASS）。而**假红同样会让人
忽略一个检查** —— 一份说"14 个模型算错了"的报告，读两次之后就没人看了。

**"我没法给它建参考实现"不属于三态里的任何一态。** 这个仓库的协议是
`0` 验过了 · `30` 这台机器上没验 · `1` 验了不对，而"判不了"是第二种。

### 改了两处

**① `has_linear` 那一支要求 FFN 真是 MoE。** `clef` 有 Linear 层、FFN
却是稠密的，于是去取 `ma["experts"]` 时 `KeyError: 'experts'` ——
**一个真模型让工具崩了。**

**② 家族守卫：IR 表示不了的就跳过。** 判据从 IR 自己来（它声明了）：

    llama-shaped   hparams 只有 d_model/n_layer/vocab  <- 默认 = Llama 形状
    gpt2.by1       hparams: norm_kind, ln_eps          <- LayerNorm
    gemma-4-31B    hparams: act, position global/local <- 滑窗 + 非 silu

    全局拦（四个分支都表示不了）  LayerNorm · 非 silu 激活 · 多种位置
    只拦 Llama / Mixtral         qk_norm
    Qwen3Next / gpt-oss 不拦     它们自己的 HF 类支持

**第三行是改第二遍才对的。** 第一版把 `qk_norm` 放进全局守卫，
于是 `qwen3-next-shaped` 从 `ok` 变成 `skip` —— **误伤了真绿**，
而那正是我自己写在文件头的硬要求里说"不许发生"的事。

### 结果

    ok 4 · skip 14 · fail 8      （改前是 ok 4 · skip 5 · fail 14 · timeout 3）
    **四绿被误伤的：没有**

顺带一个好处：守卫在**昂贵的 HF 建模之前**就返回 —— `Step-3.7-Flash`
和 180B 原来超时（各 90 秒），现在直接是干净的跳过。

档 4 仍然是 `64 项，0 项失败`。

### 而剩下的 8 个是**同一个形状的第二来源**

随手看一个：

    by1diff GLM-5.3-Flash.by1
      - 第 28 层没有可用的 token 混合器 —— **生成不出来**
      ...

**代码生成器不支持，而 `by1diff` 报的是"失败"。** 又是把"我判不了"
说成了"错了"。

**这一轮只修了第一个来源**（参考实现建错）。第二个来源在
`by1codegen` 那一侧，形状一样、修法也一样 —— **留给下一轮**。

（判卷人本身也需要判卷人：这一轮做的事就是给"判卷人说的是不是真话"量一次。）


---

## 77. "我判不了"和"它写坏了"，还有"一个原因报两次"

上一轮说"第二个来源在 `by1codegen` 那一侧，形状一样、修法也一样"。
去修才发现**它其实是三件事叠在一起**，而第一件根本不是分类问题。

### 一、`by1diff` 知道不该报错，却报了

    try:
        ir = cg.compile_ir(info)
    except cg.CodegenError as ex:
        print(f"\n  [不支持]\n{ex}\n")     # <- 它自己印的就是「不支持」
        return 1                            # <- 却返回 1 = "验了不对"

一行的事。但改完 `fail` 只从 14 掉到 8 —— 剩下的另有原因。

### 二、**一个原因报两次**，于是根因被挤掉了

`GLM-5.3-Flash` 报的是 34 句：

    - 第 33 层没有可用的 token 混合器 —— 生成不出来
    - 第 34 层没有可用的 token 混合器 —— 生成不出来
    ...

而**根因只有一句**：`one_mech` 已经说过

    机制 'KDA' 的类型是 'KDA'，codegen 还不支持

然后返回 `None`；下游看到 `mixer is None`，**为同一件事又说了一遍**。
**34 句噪音把那一句根因挤掉了** —— 我加的"写坏了优先"规则因此把它判回 fail。

修法：`one_mech` 报过就不重复报（比调用前后的条目数）。

### 三、还有第三类："属性不认识"

    - 机制 'SparseMLA' 的属性 'index_heads' codegen 不认识
    - 机制 'MoE' 的属性 'scoring' codegen 不认识

README 原话：**"KDA / SSM / 稀疏索引器 / mHC 只有契约，算不了"** ——
`SparseMLA` 就是那个**稀疏索引器**。

"属性名不认识"有两种可能：生成器没实现 / 名字拼错了。**后者由
`by1check` 在上游抓**（它有声明式的属性表）—— 实测 `selftest.by1`
（专门验"检查器会不会红"的对照文件）报的是 "hparams 里缺少 vocab"，
**不是属性名**。所以到 codegen 这一层，属性名已经"合法但我不支持"。

### 四、加了一个异常类，而不是把 `CodegenError` 全归成跳过

    class UnsupportedError(CodegenError):    # 生成器不认 -> 30 没验
    class CodegenError:                      # 文件写坏了 -> 1 验了不对

全归成跳过的话：

    selftest.by1    [跳过] 生成器还不认这套机制：hparams 里缺少 vocab
    gate-probe.by1  [跳过] 生成器还不认这套机制：FFN 的 act = <nil>

**那两份是文件本身写坏了。** 一份文件里两样都有时，写坏了优先。

### 五、而 `by1gate` 立刻抓住我分错了一处

改完 `errs`/`unsup` 之后门红了：

    [FAIL] 取值门 坏了：NVIDIA-Nemotron-...(理由)

`by1gate` 的对照表写着 `('NVIDIA-Nemotron-...', False, 'SSM', 'Mamba 整族没实现')`
—— 它要求**拒绝理由里提到 SSM**。而我把"第 1 层的主机制是 'MoE'，
不是 token 混合器"归成了 `err`，于是那一句遮住了 `unsup` 里的 SSM。

**想清楚之后：那一句本来就该归"不认"。** Nemotron 的层是
`Attn / MoE / SSM` 混排的（MTP 那条辅助栈第一层的主机制就是 MoE）——
**那不是描述写错了，是 codegen 处理不了这种层。**

改过去之后两边同时绿：门 PASS，`by1diff` 的误判为零。

**这一课是"门本身也要有判卷人"的又一次兑现** ——
`by1gate` 存在就是为了这个，而它这次抓到的是一个**我自己刚引入的**
分类错误。

### 六、结果

    ok 4 · skip 17 · fail 5        （起点：ok 4 · skip 5 · fail 14 · timeout 3）
    四绿被误伤的：**没有**
    坏文件被误放的：**没有**（selftest / gate-probe 仍然是 fail）
    by1gate：[PASS] 取值门 工作正常

`fail` 从 14 降到 5，而 5 个里 `selftest.by1` 和 `gate-probe.by1`
**本来就该红**。真正还要查的是三个真模型：`Qwen3.6-35B-A3B` ·
`gpt-oss-120b` · `hello.by1`（最后那个在门里被排除，因为它没有 HF 对应物）。

### 七、踩的坑，形状还是同一个

    按"最后一个含 import 的行"定位   -> 插进了一个函数体
    按"最后一个顶格 import"定位      -> 插进了 RUNTIME = r'''...''' 那个字符串
    "在渲染之前问"                  -> 把"渲染"认成了 render，其实是 compile_ir

**"用形状猜位置"三次都错。** 最后落在语义锚点上才成：
**插在唯一使用它的那个函数之前。**

而我加的那个 `unsupported()` 预检**最后整个撤掉了** ——
`try/except` 已经在做同一件事，而且抓的是权威的那一次。
**同一个意思不要两处实现 —— 包括我自己刚写的那处。**


---

## 78. 判卷人终于只说真话了：`fail 14 -> 2`，而那两个本来就该红

上一轮把 `fail` 从 14 降到 5。这一轮查剩下那三个真模型 ——
**结果发现其中两个根本不是模型的问题，第三个的理由一直是错的。**

### 一、内存不够也是"这台机器上没验"

    Qwen3.6-35B-A3B   RuntimeError: [enforce fail at alloc_cpu.cpp:117]
                      data. DefaultCPUAllocator: not enough memory
    gpt-oss-120b      同上

建 HF 参考实现要按真实维度分配 —— 35B / 120B 在这台机器上建不出来。
**那不是"算错了"，是"验不了"。**

道理 `by1gpu.py` 早就演示过：没显卡的机器上它走 `by1skip`，
而报告里一直印着「-- by1gpu.py [跳过]」。**先说没验，再说别的。**

只认内存那一种 `RuntimeError`（按消息判），别的原样抛 ——
把 `RuntimeError` 全归成跳过会把真错误一起吞掉。

### 二、而 `hello.by1` 那个 1.756e-02，理由一直是错的

`by1all` 里那段注释写着：

    "我一开始把它塞进 SHAPED，于是它掉进这一步，报'前向 hello.by1
     最大绝对差 1.756e-02' —— 看起来像最小例子算错了，其实是
     **拿它跟一个不存在的东西比**。"

**结论（别比）是对的，理由错了。** 量下去：

    hello.by1        rope_pairing = 'interleaved'    <- **默认值**
    llama-shaped.by1 rope_pairing = 'half'
    HF 的 Llama 参考实现                        = half

**差的就是这个。** 而 interleaved / half 正是这个仓库历史上那个
`apply_rope` 交错 bug 的那一对概念。

`hello.by1` 没有声明 position，于是拿到默认的 `interleaved`；
而 Llama 用 half。**两边算的确实不是同一个东西** —— 不是一个"错"，
是两种不同的位置编码。所以 1.756e-02 是真的，只是**不该这么比**。

**理由错了就会修错地方。** 那句注释如果被当真，下次有人看到
1.756e-02 会去查 hello.by1 的实现 —— 而该查的是"这个参考实现
表示得了 interleaved 吗"。

### 三、结果：每一个剩下的 fail 都是必须红的

    ok 4 · skip 20 · fail 2       （起点：ok 4 · skip 5 · fail 14 · timeout 3）

    剩下的两个 fail：gate-probe.by1 · selftest.by1
    —— **两个都是故意写坏的对照文件。**

    by1gate：[PASS] 取值门 工作正常

四绿（llama / mixtral / gpt-oss / qwen3-next shaped）一个没误伤。

### 四、这一轮真正的产出，是一张诚实的覆盖面表

    ok        4   真的对着 HF 参考实现比过前向
    skip     20   各有各的"这台机器上验不了"，**每一条都说了理由**
    fail      2   两个故意的反例

而**一周前这张表是 `fail 14`** —— 那里面有"参考实现建错了"、
有"生成器还没实现"、有"内存装不下"、有"两边用了不同的位置编码"，
**全都被报成了"这个模型算错了"。**

**假红和假绿一样会让人忽略一个检查。** 一份说"14 个模型算错了"的
报告，读两次之后就没人看了 —— 而那 14 个里只有 2 个是真该红的。

### 五、顺带：这一轮的坑还是"用形状猜位置"

想给 `by1diff` 加三处守卫，中途 `git checkout` 把上一轮的内存修复
一起还原了（**改动横跨两轮时，checkout 的粒度要对**），
而正则手术还把函数体和文档字符串切开过一次。

**最后是手写 + 精确锚点才成的。** 这一条这个会话里已经出现三次了。


---

## 79. 加一族换来一个真判据，而那个判据说"不一致"

上一轮把 `by1diff` 的判据说诚实了（`fail 14 -> 2`）。这一轮去量
**真正的覆盖面** —— 因为只看 `by1diff` 会漏掉 `by1gpt2` / `by1instella`
那两个判卷人。

### 一、覆盖面的真相

    外部判卷人（HF 官方实现）：6 个模型
      by1diff      llama-shaped · mixtral-shaped · gpt-oss-shaped · qwen3-next-shaped
      by1gpt2      gpt2
      by1instella  Instella-3B
    别的（by1opdiff 那类）：**自己和自己比**，不算外部判据

而 20 个跳过里，按"差什么"归类：

    多种位置（full+sliding）   5 个      生成器没实现      3 个
    家族表里没有（Linear）     3 个      装不下           2 个
    qk_norm                    2 个      LayerNorm        2 个

### 二、而"多种位置"那 5 个是**永久**的，不是"家族表短"

本来以为"加个 Qwen2 就行"。量下去发现不是：

    llama3-shaped   position: {'full': 'llama3(rope base=5000000…)',
                               'slide': 'rope base=10000…'}
    Laguna-XS-2_1   position: {'full': 'yarn(…)', 'sliding': 'rope(base=10000…)'}

**那是"不同层用不同的位置编码"** —— HF 里没有任何一个类表示得了。
所以那 5 个跳过是对的，而且不会因为"再补一个家族"而消失。

**这一条值得写下来** —— 不然下一个人会想"加个 Qwen2 就行"。

### 三、真正"加一族就能判"的只有一个，而它立刻报了个不一致

    minimind-3  ->  qk_norm = per_head  ->  Qwen3 那一族（q/k norm 是它内建的）

加完 `T.Qwen3Config` 那一支，它**被真判了**：

    权重搬运 91 个张量，逐一同名同形 ✓
    最大绝对差 2.282e-02   相对最大差 8.307e-03    [FAIL]

### 四、而这次不能再用"参考实现选错了"解释掉

这是关键的地方 —— **以前每一次都是参考实现的问题，这次不是**：

    by1verify（对官方产物）  [PASS] 22 个字段逐项一致
                            [PASS] 8 层类型逐项一致
                            [PASS] 90 个张量的名字与形状全部一致
    官方 config 的 model_type  **qwen3**     <- 参考实现选对了
    by1opdiff（逐算子）       [PASS] 全都对得上
    by1exec --compare        [PASS] 两个后端一致（相对 9.453e-07）
    HF 张量映射              91 / 91 同名同形

**每一个自指的判据都是绿的，而外部判卷人说前向不对。**

这正是这个仓库反复记着的那件事的形状："数值错能活过一个全绿的检查"。
`apply_rope` 交错那次是这样，`eps` 第六次是这样 —— **而这次有了一个
活的、可复现的例子**，而且是一个**真实模型**（`model_type: qwen3`）。

### 五、排除了一个可疑处

官方 config 是 `tie_word_embeddings: True`，而 `by1diff` 里写死 `False`。
**看起来像原因**。做实验：

    原样（False）   2.282e-02
    改成 True       2.329e-02     <- 几乎没变

**不是它。** （不过那个写死的 `False` 本身仍是个问题：官方产物里
**只有 `model.embed_tokens.weight`，没有 `lm_head.weight`** ——
这个模型是真 tied 的，而参考实现建出了独立的头。
**两边对"头"的定义不同**这件事值得单独修，只是它不是这 2.282e-02 的原因。）

### 六、还没做的：定位到哪一层开始分岔

已经确立的是"**组合方式**不同"而不是"某个算子算错"（逐算子全过）。
下一步是**逐层比**：哪个层的输出先偏离参考实现。

**这一轮到此为止，因为这是一条新的、值钱的线索** ——
它第一次给"数学语义缺外部判卷人"这件事提供了**一个具体的受害者**，
而不是一句原则。


---

## 80. 把那个 2.282e-02 从"模型里某处"缩到"第 0 层注意力内部"

上一轮加一族换来一个真判据（`minimind-3` 对 Qwen3 差 2.282e-02），
而每一个自指的判据都是绿的。这一轮去定位 —— **加了两个诊断开关**。

### 一、`by1diff --layers`：哪一层的输出先偏离

两边都取得到逐层：

    HF   `output_hidden_states=True`                 -> (嵌入, 每层输出…)  共 N+1 个
    by1  生成模型有 embed / layers(ModuleList)       -> 手动逐层前向

（**HF 的 `hidden_states[-1]` 是最后一层输出、在 `final_norm` 之前** ——
所以 by1 那边也不加 final_norm，两边才对得齐。）

跑出来：

    层 0（嵌入）   0.000e+00        <- **完全一致**
    层 1           1.090e-02        <- **从第一层之后就偏了**
    层 2..7        ~1e-02 ~ 2.6e-02
    层 8           1.078e+00  相对 2.887e-01

**嵌入是逐位相同的，第一层之后差 1.09e-02。**
而 1e-2 不是数值噪声（fp32 是 1e-6 量级）—— **是接线问题，不是算子问题**。

### 二、`by1diff --probe`：差在注意力还是前馈

一层里有注意力和前馈两块。两边的注意力都能挂 hook：

    HF    `ref.model.layers[0].self_attn`
    by1   `mine.layers[0].op1`     （参数名 `layers.0.op1.wq.weight` 就是这么来的）

    ref  输出 幅度 9.557e-01
    by1  输入 幅度 4.142e+00   输出 幅度 9.668e-01
    **注意力输出差 1.106e-02   相对 1.157e-02**

（1.106e-02 和上一层的 1.090e-02 对得上 ✓）

### 三、路上自己犯的一个错，值得记

探针第一次跑出来的 by1 注意力**输入幅度 4.142** 看着很大，我一度以为
norm 出了问题。**写了个单独的脚本去量，结果那个脚本忘了搬权重** ——
两边都是随机初始化，"嵌入差 4.016" 是我的错，不是模型的错。

而那次跑里**唯一有效的一行**是：同一个输入下，两个 RMSNorm 的输出
差 `0.000e+00`、幅度 `4.304e+00` —— **说明 4.142 本来就是正常的**。
（RMSNorm 的输出幅度就是 `max|w|` 那个量级。）

**一个忘了搬权重的对拍脚本，报出了一个听起来很像 bug 的数。**
这一条和前面那些"假的 16 个泄漏""假的 14 个 fail"是同一类。

### 四、核过的、以及还没排除的

`per_head` 那一支的接线逐项核过 HF 的写法：

    qn / kn 都是 RMSNorm(head_dim)，权重形状 (96,)        ✓ 一致
    作用在拆头之后、rope 之前                             ✓ 一致
    apply_rope 的 half 分支 = rotate_half                 ✓ 一致
    rope_tables 的 inv：`i/half` 与 HF 的 `2i/dim`        ✓ 等价
    eps 1e-06、rope_base 1e6、pairing half、bias False    ✓ 一致

**所以还没排除的在注意力那 ~55 行的其余部分**（GQA 的 repeat、
mask、softmax、o_proj 的接法）。

### 五、这一轮的产出

两个开关（`--layers` / `--probe`）留在 `by1diff` 里 ——
**下次任何一个模型报前向不一致，都能一条命令缩到"哪一层、哪一块"。**
而那个真 bug 的范围，从"整个模型"缩到了"第 0 层注意力的 55 行"。


---

## 81. 那个 2.282e-02 是**超参写死**造成的假红 —— 第三次同类

上一轮用 `--layers` 和 `--probe` 把 `minimind-3` 的差异缩到了
"第 0 层注意力内部，且输入是对的"。这一轮继续缩，**缩到了第一个算子**。

### 一、二分：第一个不一致的是 `op0`（那一层的 Norm）

注意力那 55 行读过三遍都"看起来标准"，所以改成**数值二分** ——
给两边形状对得上的子模块各挂 hook：

    op0  <-> input_layernorm   4.479e-02   相对 1.093e-02   **不一致**
    wq   <-> q_proj            2.541e-02   相对 1.055e-02   **不一致**
    wk / wv / wo               同样不一致

**连第一个算子 `wq` 都不一致** —— 而它是 `op0.out` 的第一个消费者。

### 二、三样都验过之后，只剩一种可能

    输入   `op0` 与 `input_layernorm` 的输入差 **0.000e+00**
    权重   `op0.w` 与 `input_layernorm.weight` 差 **0.000e+00**
    公式   读过 HF 的 `Qwen3RMSNorm` —— 一模一样
    而输出差 1.09%，by1 偏大

同一个输入 + 同一份权重 + 同一个公式 = 同一个输出。
**三条里必有一条是假的。** 于是把每一条都真量了一遍：

    by1  eps=1e-06   one_plus=False
    HF   eps=**1e-05**

**公式不同 —— 差在 `eps` 上。**

### 三、而那个 eps 是 `by1diff` 自己写死的

    common = dict(..., rms_norm_eps=1e-5, tie_word_embeddings=False, ...)

`minimind-3` 的官方 config 写的是 `rms_norm_eps: 1e-06`，IR 里也是 1e-06 ✓
—— **而参考实现是用写死的 1e-5 建的**。同一个输入、同一份权重、
eps 差 10 倍，输出就差 1e-2，而报出来是"**这个模型算错了**"。

改完：`最大绝对差 2.861e-06  相对 1.038e-06  [PASS]` ✓

### 四、这是同一类问题的**第三个**实例

    ① 家族表里没有      -> 拿去和 Llama 比，四个 shaped 之外全是假红
    ② KDA / SSM 不认     -> 报成"算错了"
    ③ **参考实现里的超参写死** -> 报成"算错了"

**形状完全一样：判卷人自己错了，而报告说的是被judged的人错了。**

而三次都是**同一个办法**找出来的：**先让"我没法判"和"判出来是错"
分开，再把每一条链真量一遍** —— 而不是读代码猜。

### 五、修的时候又踩了两个坑，都值得记

**① `.get(k, default)` 在"键存在但值是 None"时不给默认值。**

    ir.get("norm_eps", 1e-5)      # 键在、值是 None -> 拿到 None
    ir.get("norm_eps") or 1e-5    # 两种都兜住

第一版就是前者：**修好了 minimind-3，把四个 shaped 全弄崩了。**
判据必须两边都看 —— 新过的要过，**原来过的也要还在**。

**② 新增的无条件代码必须对**所有**模型成立。**

为了核对权重我加了一段无条件扫描，里面直接访问 `op1.qn.w` ——
而没有 `qk_norm` 的模型根本没有 `qn`，`AttributeError` 把四个 shaped
全弄崩了。**和 ① 是同一次事故的两面。**

### 六、结果

    ok 5 · skip 18 · fail 3      （起点：ok 4 · skip 5 · fail 14 · timeout 3）
    ok：gpt-oss-shaped · llama-shaped · **minimind-3** · mixtral-shaped · qwen3-next-shaped
    误判：**没有**

**有真外部判卷人的模型：7 个**（by1diff 五个 + by1gpt2 的 gpt2 +
by1instella 的 Instella-3B）。

### 七、这一轮的账

`--layers` / `--probe` 留在 `by1diff` 里（上一轮加的）。
二分的那个开关（`--bisect`）是**这一轮临时加的，最后撤掉了** ——
但它给出的那条路径记在这里：**把"同一个输入 / 同一份权重 / 同一个公式"
三条逐一验过，"公式不同"就被逼出来了。**

而 `minimind-3` 那 2.282e-02 从"真实模型上的一个真 bug"
变成了"**判卷人的第三个假红**"。**这不好看，但它是真的** ——
而它换来的是"下次再看到前向不一致，知道该先查哪三条链"。


---

## 82. 诊断工具**绝不能改变判定**

上一轮那三次假红有一个共同的形状：**判卷人断言了自己没验过的东西**
（家族表 / 机制支持 / 超参）。这一轮做两件事：审那个类还在不在，
以及把找到它的工具留下来。

### 一、审：参考实现的构造里还有没有写死的值

把 `common` 那一段逐个参数列出来，问"从 IR 读的还是写死的"：

    vocab_size          ir["vocab"]                    ✓
    hidden_size         ir["d_model"]                  ✓
    num_hidden_layers   len(ir["layers"])              ✓
    num_attention_heads attn["q"]                      ✓
    num_key_value_heads attn["kv"]                     ✓
    head_dim            attn["head_dim"]               ✓
    max_position_emb.   max(ir["ctx"], args.seq + 8)   ✓
    rms_norm_eps        ir.get("norm_eps") or 1e-5     ✓（1e-5 只是兜底）
    tie_word_embeddings bool(ir.get(...))              ✓
    attention_bias      attn["bias"]                   ✓
    rope_theta          float(attn["rope_base"])       ✓

**四个家族分支的参数也全是 IR 派生的**（Qwen3Next 那支的
`num_experts` / `linear_*` 等都读自 `ffn_op["attrs"]` / `lin[...]`）。

**所以产出三次假红的那一类是关上的。** 而它是**审出来的**，不是猜的。

### 二、把 `--bisect` 加回来 —— 而这次加完立刻对 26 份全量验

上一轮它是临时拼的，还被 `git checkout` 一起还原了。这次重写，并且
**加完立刻对 26 份 × 四种模式全跑一遍**。而那一遍抓到了**三个**问题：

    ① `ref.model.layers[0].self_attn` —— Qwen3Next 那一族的层不叫这个
       -> AttributeError -> 把 `qwen3-next-shaped` 从 ok 变成 fail ✗
    ② 拿 `object()` 占位 -> 换了个 AttributeError（`register_forward_hook`）
    ③ **变量遮蔽**：二分里 `a, b = got.get(...)` 覆盖了外层存 logits 的
       `a` / `b`，于是后面 `d = (a - b).abs()` 拿到两个 `None`
       -> `TypeError: unsupported operand type(s) for -: 'NoneType' and
          'NoneType'`，而报的位置在**外面** ✗

**而 `--probe`（上一轮加的）有同一个 ①** —— 它当时只拿 `minimind-3`
试过，**没验过别的家族** ✗。这次一并修了。

### 三、于是有了一条不变量

    **诊断工具绝不能改变判定。**

`--layers` / `--probe` / `--bisect` 在不认识的家族上量不了时，
应该说"量不了"然后退场 —— 而不是把 ok 变成 fail。

验法：**26 份 × 四种模式（无 / --layers / --probe / --bisect），
四个组合的 ok 数和 fail 数必须完全一样**。

    无          ok=5  fail=3  误判=没有
    --layers    ok=5  fail=3  误判=没有
    --probe     ok=5  fail=3  误判=没有
    --bisect    ok=5  fail=3  误判=没有   ✓

**四个组合一模一样** —— 这就是那条不变量的判据。

### 四、而 `minimind-3` 的二分现在全绿

    op0  input_layernorm  0.000e+00  一致
    wq   q_proj           0.000e+00  一致
    wk   k_proj           0.000e+00  一致
    wv   v_proj           0.000e+00  一致
    wo   o_proj           1.788e-07  一致
    [PASS] 两边前向一致

**确认了 eps 就是那 2.282e-02 的全部原因** —— 而二分当时说
"第一个不一致的是 `op0`"，正是它把范围逼到了那一步。

### 五、这一轮真正的教训

**"加完立刻对全集验一遍"** 这句话写过很多次，但这一轮它是**具体的**：

    加 `--bisect`  ->  立刻 26 份 × 4 模式  ->  抓到 3 个问题
    而 `--probe` 上一轮没这么做  ->  它带着同一个 bug 活了一轮

**"验过了"和"在最难的那份上验过了"是两件事。**


---

## 83. **那 21 次失败的那一段，搬出去了** —— 命名空间

这是这个目标开始以来第一次真正的进展。

### 一、回到第 3 轮分析出的那条路

第 3 轮量清楚的是：**Python 函数的局部名在编译期定死**，
所以"把一段搬进函数"**必须**有一份静态名字清单；而清单要对，
就需要完整的数据流分析（作用域 + 确定赋值 + 分支合并）——
21 次失败全卡在这。

**命名空间绕开的是"清单"本身**：

    def check(path):
        _ = {}
        ...主体，所有状态都在 `_['...']` 里...

主体于是**没有任何局部名字**。搬任何一个 stage 出去时，
它只需要接一个参数 `_`。

### 二、为什么这次能做，而前 21 次不能

关键在于**替换规则是纯文本级的、且对读和写是同一个**：

    x = 1        ->   _['x'] = 1            下标赋值
    用 x          ->   _['x']                下标读取
    x += 1       ->   _['x'] += 1
    a, b = ...   ->   _['a'], _['b'] = ...
    for k, v in  ->   for _['k'], _['v'] in
    with ... as f ->  with ... as _['f']     （`as` 目标也能是下标）
    del x        ->   del _['x']

**同一个替换对读和写都成立** —— 所以不需要知道哪边是哪边 ✗
**不需要任何数据流分析** ✗。

做法是**按 AST 位置精确换文本**：用 AST 找出该改的 `Name` 节点的
（行列），再在 **bytes** 上替换（`col_offset` 是 UTF-8 字节偏移，
而这个文件里全是中文注释），**从后往前**改（长度会变）。

### 三、四次才改对，而四次都是"边界"问题

    ① 引号套错                            SyntaxError（我自己的脚本）
    ② 把整个推导式当成新作用域             NameError: named / stacks
    ③ 嵌套 def 的形参当普通名字改           KeyError: 'over'
    ④ **同上** —— 因为修的地方没生效        KeyError: 'over'

②**正是第 3 轮诊断过的那个坑**：推导式里**只有目标**是新作用域，
它读到的外层名字仍要改。所以判据不是布尔 `in_comp`，
而是"**哪些名字被推导式绑了**"这个集合。

④最值得记：我把 `def` 判断写在了**遍历子节点的循环里**，
而 `resolve_attrs` 是作为 `ck.body` 的一项**直接传进 `walk`** 的 ——
于是它自己从来没被认出来，形参也就没进 `bound`。
debug 打出来是 `bound=['path']`，只有 `check()` 的形参 ✗。
**"改了但没生效"和"没改"看起来一样** —— 所以每次改完都要看那个数。

### 四、然后那一段就搬出去了

    [命名空间]   **逐字节相同** ✓
    段 874-1031（158 行）
    check() 701 行；by1blocks.py 189 行
    [搬完之后]   **逐字节相同** ✓

**这是前面 21 次尝试都没做到的那一段**（九个块的一致性检查：
双真相源 / attach / state / position / optimizer / interop /
机制形状 / mrope / memory）。它的新家是 `src/lang/by1blocks.py`，
参数表是 `(*, _mod, _)` —— **一个名字都不用列**。

### 五、一个白捡的简化

`def f()` 里的 `f` 是 `FunctionDef.name` **字符串**，不是 `Name` 节点
—— 所以按 AST 位置改文本**本来就碰不到它** ✓。
我一开始还专门写了一条 `LOCAL_DEFS` 排除 ✗，而它**有害**：
把引用也排除了，于是搬走的那段看不见 `heads_of` ✗。

去掉那条排除，再给每个嵌套 def 补一句 `_['heads_of'] = heads_of`，
**那 7 个函数就住进了命名空间**，任何 stage 都取得到 ✓。

### 六、数字

    check()       1243 -> 1064 -> **833** -> **701 行**
    by1check.py   2102 -> **1562 行**
    by1blocks.py  （新）**189 行**

    覆盖**升**了：编译 62 -> 63 · import 57 -> 58 · 行尾 103 -> 104
    档 4：64 项，0 项失败
    26 份 `.by1` 输出**逐字节不变**（两次：改写之后、搬完之后各一次）

### 七、这条路现在是通的

剩下的段（`stacks` 87 行、张量契约 166 行、emit lowering 85 行……）
**都可以照这个办法搬**：找到 mark、切出来、套一个 `def stage(*, _mod, _)`、
原地放一句调用。**不需要再和名字清单搏斗。**

而"劈开"这件事真正的意义在这里兑现了：`check()` 现在是一个
**只做编排的骨架** —— 攒状态、按顺序调 stage、最后组装 `info`。


---

## 84. 把搬运做成工具，以及**一次我自己造成的覆盖**

### 一、上一轮是手搓的，这一轮做成工具

搬 `by1blocks` 时是手工做的：找 mark、切行、套壳、插调用、对拍。
**那个形状可以复用** —— 而剩下的段还有九个（`张量契约实例化` 240 行、
`emit lowering` 109 行、`stacks` 105 行、`状态尺寸` 72 行……）。

所以写了一个 `extract(标题关键字, 模块名, 函数名)`，它做六件事：

    ① 按注释里的 mark 找到那一段
    ② 写成 `src/lang/<模块>.py`：`def <函数>(*, _mod, _):` + 段体（**一字不改**）
    ③ 原地换成一句 import + 一句调用
    ④ 编译
    ⑤ **对拍 26 份** —— 不逐字节相同就整段回滚
    ⑥ 报告行数

而它一次就搬了两段：

    张量契约实例化  240 行 -> by1tens.py（266 行）    701 -> 465 行  ✓
    emit lowering   109 行 -> by1lower.py（135 行）   465 -> 360 行  ✓

**`check()` 现在 360 行**（起点 1243）。

### 二、而我把一个已有的文件覆盖了

`src/lang/by1contract.py` **原来是一个独立工具**（242 行，"从机制声明
**推出**张量契约"的演示，有自己的 `main()`）。我的提取器往同名文件写，
**把它盖了** ✗。

**而 26 份对拍还是绿的** —— 因为那是个 CLI，没有任何模块 import 它 ✗。

> **对拍只覆盖了"谁在用"，覆盖不了"谁存在"。**

这一条值得单独记：这个仓库一直用"谁在读它"当判据（`1.md` 里那句
"没有任何东西在读它"），而**这一次的教训是它的反面** ——
一个没人 import 的文件被覆盖，任何对拍都看不见。

（运气好的是提取器本来就会留备份 —— 那是为了"编译不过就回滚"，
而这次要回滚的不是编译，是"我不该动那个文件"。
修法：新模块改名 `by1tens.py`，`by1contract.py` 从备份还原
—— 验过和 HEAD **逐字节相同** ✓。）

### 三、顺带又碰到一次 `by1files` 的自我指涉

`--write` 之后是 PASS，提交推送时又 FAIL ✗。**写两次才收敛**
（这是上一轮记下的那个问题：它把自己也算进它描述的那棵树，
而描述里含行数）。

### 四、数字

    check()       1243 -> 1064 -> 833 -> 701 -> **360 行**
    by1check.py   2102 -> **1221 行**（大约）
    by1tens.py    （新）266 行
    by1lower.py   （新）135 行
    by1blocks.py  （上一轮）189 行

    覆盖继续**升**：编译 63 -> **65** · import 58 -> **60** · 行尾 104 -> **106**
    档 4：64 项，0 项失败
    26 份 `.by1` 输出**逐字节不变**（每一段搬完都各对拍了一次）

### 五、剩下的

`stacks` 105 行 · `状态尺寸` 72 行 · `导出层` 48 行（已是调用）·
`汇报` 31 行 · `未填占位符` 22 行 · `schedule` 14 行 · `hparams` 10 行。

**照同一个办法搬就行** —— 一个 `extract(...)` 调用，加一次对拍。


---

## 85. **`check()` 只剩骨架了** —— 1243 行 -> 167 行

同一个 `extract(...)` 又搬了四段：

    hparams     10 行 -> by1hp.py       360 -> 352 行  ✓ 逐字节相同
    schedule    14 行 -> by1sched.py    352 -> 340 行  ✓
    stacks     105 行 -> by1stacks.py   340 -> 237 行  ✓
    状态尺寸     72 行 -> by1state.py    237 -> 167 行  ✓

而 `check()` 的主体现在长这样：

    by1hp.read_hparams(_mod=globals(), _=_)
    by1sched.read_schedule(_mod=globals(), _=_)
    by1stacks.read_stacks(_mod=globals(), _=_)
    by1blocks.check_blocks(_mod=globals(), _=_)
    by1tens.instantiate(_mod=globals(), _=_)
    by1lower.emit_rules(_mod=globals(), _=_)
    by1state.state_sizes(_mod=globals(), _=_)

**七处调用，一句"搬去哪个文件"。** 这就是这个目标一开始说的"只剩骨架"。

### 三套词汇表现在分居三处

    语义/调度        by1hp 34 · by1sched 38 · by1stacks 129
                     by1blocks 189 · by1state 96
    契约（形状）      by1tens 266
    导出（config 字段）by1export 513 · by1lower 135

**这正是当初"劈开"要的那个区分**：`check()` 里曾经同时住着
数学语义、内存布局、权重命名三套词汇表 —— 而唯一有外部判卷人
（`refs/`）的只有第三套。

### 数字

    check()      1243 -> 1064 -> 833 -> 701 -> 360 -> **167 行**
    by1check.py  2102 -> **1034 行**
    新模块        八个，合计约 1400 行

    覆盖**全程只升不降**：编译 62 -> **69** · import 57 -> **64**
                        · 行尾 103 -> **110**
    档 4：64 项，0 项失败
    26 份 `.by1` 输出**逐字节不变**（每一段搬完各对拍一次，共六次）

### 这一轮又踩了一次括号

给 `by1files.py` 加归组时用正则替换，**把括号层级弄错两次** ✗
（`'by1state'),` -> 先改成 `'],` 又改成 `']),`）。
两次都是 `py_compile` 当场抓住的 —— **而它值得记的原因是：
这就是"改文本要按结构改，不要按字符串改"**。
前面命名空间改写之所以能成，正是因为它**按 AST 位置**改而不是
按正则改；而我给 `by1files` 加一行却用了正则 ✗。


---

## 86. 两份给外人的文档：`TESTING.md` 和 `docs/criteria-review.md`

这一轮没有动代码。写的是两件**让别人能验这个项目**的东西。

### 一、`TESTING.md`：给"想验它的人"

README 是给"想知道这是什么的人"看的。测试者要的是别的：
跑什么、跑多久、**"通过"长什么样**、失败了贴什么回来。

它写的三档（耗时和样例都是从本机实跑的输出里**直接抄的**）：

    L1  python src/checks/by1all.py --tier 1     纯标准库，10 秒
    L3  python src/checks/by1all.py --tier 3     +numpy，20 秒
    L4  python src/checks/by1all.py --tier 4     +torch，1 分钟

而重点在三件"必须先说，否则一定会误读"：

    ① **默认档是 4，不是"最高档"** —— 别以为跑完就全验过了
    ② **`--` 不等于通过** —— 五态里 `[跳过]`(30) 是"没验"，不是"过了"。
       判定行写着「[全过（11 项没验）]」，**那 11 项不是绿的**
    ③ **数字跟着机器走** —— 别引用任何一处"N 项"当常数

第 ② 条是这个项目最值钱的设计之一，而外人默认只看退出码 ✗。

### 二、`docs/criteria-review.md`：**这才是真正缺的那一份**

README 里那句"判据是我定的，而那件事没有人替我验过" ——
这一份就是那份活的任务卡。**不是"帮我看看"，是三个几小时能做完、
有确定判据的任务**：

    A. 挑三个模型，**不看 by1 源码**，自己从 refs/ 的 JSON 里数出
       张量总数 / 命中数 / config 字段数，和 INVENTORY.txt 的【A】比
    B. 手算一个 GQA 模型的 KV 状态尺寸
    C. 试着补第 15 个模型 —— 看它是"又一个已覆盖"还是"又多一个新属性"

而第四节是我自己必须先说出来的那一条：

> **`refs/` 判 `by1`，其实不是外部判卷。**

因为 by1 的命名规则、属性表、机制分类，**是照着这些 `refs/` 写出来的** ✗。
所以 `refs/` 全绿说明的是"**拟合得好**"，不是"预测得对" ✓。

例子就在 `history/ir.md` 里：`gpt2` 和 `llama-shaped` 一度**编译出同一份 IR**
—— 两边对拍都是绿的，因为判据里根本没区分它们 ✓。真正把它抓出来的是
**加了 transformers 当第二个实现**，不是 `refs/` ✓。

**所以最该被质疑的不是数字，是那四样**：`1e-4` 那个阈值 · "什么算覆盖" ·
挑哪 14 个模型 · **连 10 条神谕的期望值也是我写的**（神谕的期望值写错了，
它会**稳定地**判错，而且看起来像被测的东西错了 ✗）。

### 三、路上核了那份方案里的具体断言

拿到方案时它里面有四处很具体的说法，而**具体的东西最容易被写成假的**，
所以先核了一遍：

    docs/INVENTORY.txt 存在吗       -> **存在**（90 行）✓
    "14 个模型"                     -> README 里确实写着 ✓
    pyproject 的 dependencies = []  -> **真是空的** ✓
    closure_missing/judges_missing  -> **两个都有** ✓
    by1triage.py / by1all.py        -> 都在 ✓
    默认档 4                        -> 确实是 ✓

**四处全是真的。** 记这一笔是因为下一次我还会先核 ——
而"先核一遍"这件事本身没有成本。

（顺带看清一件事：`models/` 里现在有 **26** 份 `.by1`，而 README 说
"14 个模型" —— **两者不矛盾**：14 指的是 `refs/` 里有官方产物、
真被外部核过的那些；另外 12 份是合成的 shaped 模型和探针。）


---

## 87. CI 只跑档 1 —— 而"档 1 只要标准库"这句话先验了

### 一、这个仓库原来**没有任何自动门**

推送走 `by1run --push`，它用 `git push --no-verify` —— **本地钩子不跑** ✗。
所以我这台机器上"检查过了"和"提交了"之间没有强制关系 ✓。

而这一路真正堵住推送的那些错：**语法错 · `FILES.md` 过期 ·
行尾变成 CRLF · `by1files` 的生成顺序** —— **全都落在档 1 里** ✓。
档 1 又是纯标准库，CI 里一个依赖都不用装 ✓。

### 二、先验"只要标准库"，再写 workflow

这份配置要证明的第一件事就是"档 1 不需要装任何东西"。所以：

    python      src/checks/by1all.py --tier 1   -> rc=0，编译 69 · import 64 · 解析 26
    python -S   src/checks/by1all.py --tier 1   -> rc=0，**数字完全一样**
    （`-S` 禁用 site-packages，等于只有标准库）

**两次输出的内容逐行相同** ✓（哈希不同只是重定向的行尾/BOM 假象）。
所以那句话是真的，不是我以为的 ✓。

顺带查了一遍 3.9 兼容性（因为 `pyproject` 写着 `requires-python = ">=3.9"`，
而**那个下限从来没被真跑过一次**）：扫了 69 个 `.py`，
**没有** `match` 语句 · 没有 `X | Y` 类型标注 · 没有内建泛型标注 ·
没有 `except*` ✓ —— 所以那句话**至少不像假的**，但真跑一次才算数。

### 三、workflow 的四个选择，每个都有理由

    只跑档 1        档 2 起要 numpy / refs，档 4 要 torch —— 那些放进来
                    要下几百 MB，回报远不如档 1。**这一条不假装覆盖了它们。**
    **不装 `[verify]`**  因为"档 1 只要标准库"是这份配置**要证明的东西** ——
                    装了依赖再跑就证明不了它 ✗
    **Windows 也跑** 这个项目在 Windows 上开发，而行尾和路径分隔符
                    **在这里出过两次事**（66 个文件的工作副本是 CRLF
                    而 `.gitattributes` 写的是 LF）。只在 Linux 上跑看不见。
    **3.9 也跑**      `requires-python = ">=3.9"` 是个没验过的声明。
    fail-fast: false  一个组合红了别取消别的 —— 那样只看得到第一个错。

### 四、我验了什么、没验什么

**验了**：YAML 能解析（`yaml.safe_load` 通过，结构对）·
行尾 0 个 CRLF ✓ · workflow 里那两条 `run:` 命令在本机跑得过 ✓ ·
档 1 在 `-S` 下数字不变 ✓

**没验**：**它在 GitHub 上到底跑不跑得起来。** 这台机器上跑不了
Actions ✗ —— 我只能验到"配置是合法的、命令是能跑的"。
**第一次真跑的结果得等下一次推送之后才知道** ✓。

### 五、门仍然只承认五态

判定行里写着「[全过（N 项没验）]」—— **那 N 项不是绿的** ✗。
CI 看的是**退出码**，而它是五态的（0 过 / 1 错了 / 30 没验 /
40 调用方错 / 51 检查器坏），不是两态 ✓。档 1 全静态，正常只有 0 或 1。


---

## 88. `.gitignore` 把 `__init__.py` 一起吃掉了 —— **静默少 5 个文件**

### 一、它是怎么被发现的：不是任何检查，是"克隆一遍"

刚给 CI 写了档 1，而我在 `ir.md` 里诚实地写了"**没验**它在 GitHub 上
跑不跑得起来"。然后想到一件**本机能做的等价测试**：

> CI 里跑的是 `actions/checkout` 拉下来的**干净副本** ——
> 而我一直是在**工作区**里跑，工作区里有未跟踪的产物。
> **如果它只是因为某个未跟踪的文件才过，CI 一上去就是红的。**

于是 `git clone . <临时目录>`，在那里跑档 1。结果是：

    工作区   编译 **69** · import 64 · 解析 26     rc=0
    干净克隆 编译 **64** · import 64 · 解析 26     rc=0
                       ^^ 少 5 个，**而两边都是 PASS**

**正是不久前那次"静默丢了 33 个模块还报 PASS"的同一个形状。**

### 二、差的那 5 个是五个 `__init__.py`

    src/checks/__init__.py      **被 .gitignore 忽略**
    src/data/__init__.py        **被 .gitignore 忽略**
    src/escape/__init__.py      **被 .gitignore 忽略**
    src/lang/__init__.py        **被 .gitignore 忽略**
    src/modelcheck/__init__.py  **被 .gitignore 忽略**

根因是 `.gitignore` 第六条：

    # 临时探针（用完就删，不进历史）
    _*.py

**那条规则是按"下划线开头的都是临时文件"写的 —— 而 `__init__.py`
恰好也以 `_` 开头。**

### 三、为什么没有任何检查出声

    git clone        拿不到那 5 个文件（它们从没被跟踪过）
    pyproject.toml   把它们声明成 package —— 于是**从克隆装的包是残的**
    by1files         **特意跳过 `__init__.py`**（"不算模块，多五行噪音"）
    by1fast          数的是"在场的文件能不能编译"，**没有判据管"该在的
                     在不在"**

四个环节各自都对，合起来就是**一个不出声的缺口**。

### 四、修法与判据

`.gitignore` 里补一条 `!__init__.py` —— 这个文件里**已经有同样的先例**
（`*.bak` 后面跟 `!1.md.bak`）。

判据只有一条，而且它是**可以跑的**：

    **工作区的"编译 N" == 干净克隆的"编译 N"**

修之前 69 vs 64；修之后两边一样。

### 五、而 CI 正好就是防这一类的

**这个 bug 是"CI 会抓到的第一件事"** —— 因为 CI 就是在克隆上跑。
它在这里被提前抓到了，用的是同一招（克隆一遍），只是手动做的。


---

## 89. 三个机制的**判卷人侦察** —— 它决定每一件能不能诚实做

要新增三样：KDA · Mamba/SSM · 多模态。而按这个项目的纪律，
**每加一个机制必须同时给出判卷人** —— 所以第一件事不是写实现，
是问"这个机制有没有独立实现能对拍"。

### ① Mamba / SSM —— **判卷人在，本机就能做**

装着的 transformers 里就有：

    MambaForCausalLM  ✓   Mamba2ForCausalLM  ✓   FalconMambaForCausalLM  ✓
    **NemotronHForCausalLM  ✓   NemotronHConfig  ✓**

而**关键分歧**在这儿：官方 `MambaMixer`（纯 Mamba）的 `in_proj` 是
`intermediate_size * 2`；而 Nemotron 声明的是

    in_proj [10304, 2688] = 2*d_inner + 2*n_groups*ssm_state + n_heads
                          = 2*4096 + 2*8*128 + 64 = 10304 ✓

—— 那是 **Nemotron-H 那一版**的布局。**所以判卷人得是
`NemotronHForCausalLM`，不是 `MambaForCausalLM`。**

而它的参数和 `.by1` 注释里那套算术**逐个对上**：

    NemotronHMamba2Mixer:
      num_heads        = mamba_num_heads        64
      intermediate_size= num_heads * head_dim   64*64 = 4096 = d_inner ✓
      ssm_state_size   = ssm_state_size         128
      n_groups         = n_groups               8
      conv_kernel_size = conv_kernel            4
      chunk_size       = chunk_size             128

**所以 Mamba 这一件本机就能诚实地做**：判卷人已装好、形状已知、
目标明确（`by1diff Nemotron-3.5-Lightning.by1` 对着 `NemotronHForCausalLM` 过）。

（另外它还缺第二样：`机制 'MoE' 的属性 'gate' codegen 还不支持` ——
同一个模型，小改动。）

### ② KDA —— **判卷人看不到，本机做不了**

这一条仓库里**早就写下了**（`gpu/by1kda.py` 的头注）：

> KDA 的核心是 `fla.ops.kda.chunk_kda` —— **flash-linear-attention
> 库里的实现**，要 Triton，没有 CUDA 就 import 不了。
> 我在没有 CUDA 的机器上**看不到它的源码**。……我知道 KDA 的
> **张量集**（那部分已经和官方产物对上了：Ling 9283/9283），
> 但**它的门控语义我看不到**。
> **所以在这里凭空写一个 KDA 实现，就是在猜。**

本机实测确认：`fla` 没有 · `triton` 没有 · `torch 2.13.0+cpu` ·
`cuda.is_available() = False`。

**所以 KDA 不是"跑得慢"，是"看不到语义"。** 而那个脚本存在的目的
就是把顺序倒过来：**先把判卷人立起来、把中间量 dump 出来** ——
那是**有 CUDA 才能做**的事。

**这一件该租卡。** 而且顺序是：先立判卷人，再写实现。

### ③ 多模态 —— 不用判卷人，是**建模**的活

量了一下"完全没建模"具体指什么：

    gemma-4-31B.by1      mechs = {GQA: Attention, Dense: FFN}
    Step-3.7-Flash.by1   mechs = {Attn, Dense, MoE}
    GLM-5.3-Flash.by1    mechs = {KDA, SparseMLA, MoE}
    **多模态相关的键：一个都没有**

也就是说：**`.by1` 只描述了文本主干** ✗ —— 而 `refs/` 的官方
config 里**有**那些视觉/音频字段 ✓。

所以这是一件**语言层**的活（新的机制种类 + 张量命名 + 层类型），
不是 codegen 的活 ✓ —— 而且它和"能描述 ≠ 能算"这条线直接相关：
**做到哪一步要先说清楚**（能声明、能被 `by1verify` 覆盖，和能跑前向
是两件事）。

### ④ 顺带纠正我自己一个读法

`info['mechs']` 是 `{名字: 种类}` —— 不是 `{名字: {kind, attrs}}`。
我第一版按后者读，拿到一堆 `AttributeError`。**结构要读对，
不然量出来的东西是假的。**

### 结论：三件的顺序

    1. **Mamba / SSM**（判卷人有）      —— 本机能做，先做这个
    2. **MoE 的 gate 属性**（同一个模型）—— 小改动
    3. **多模态建模**（不用判卷人）      —— 语言层的活
    4. **KDA**（判卷人看不到）          —— **要租 CUDA，而且先立判卷人**


---

## 90. Mamba/SSM 的**完整设计** —— 判卷人、形状、算法

上一节说"Mamba 本机能做"。这一节把它**读到底**，读完就不需要再查了。

### 一、判卷人：`NemotronHForCausalLM`（不是 `MambaForCausalLM`）

**纯 Mamba** 的 `in_proj` 是 `intermediate_size * 2`；Nemotron 声明的是

    projection_size = intermediate_size + conv_dim + num_heads
                    = 4096 + 6144 + 64 = 10304        ✓ 对上了
    conv_dim        = intermediate_size + 2*n_groups*ssm_state
                    = 4096 + 2*8*128 = 6144           ✓ 对上了

**所以那是 Nemotron-H 那一版** ✓ —— 判卷人必须是 `NemotronHForCausalLM`
（它和 `NemotronHMamba2Mixer` 都在装着的 transformers 里 ✓）。

### 二、声明 → `__init__`，逐个对上

    .by1 声明                        官方 NemotronHMamba2Mixer
    ----------------------------------------------------------------
    heads = 64                       num_heads = mamba_num_heads        64
    head_dim = 64                    head_dim = mamba_head_dim          64
    ssm_state = 128                  ssm_state_size                     128
    n_groups = 8                     n_groups                           8
    conv_kernel = 4                  conv_kernel_size                   4
    expand = 2                       intermediate_size = heads*head_dim 4096
    chunk_size = 128                 chunk_size                         128
    conv_bias = true                 use_conv_bias                      True
    proj_bias = false                use_bias                           False
    act = silu                       mamba_hidden_act                   silu

    张量：
    conv1d.weight  [6144, 1, 4]      depthwise（groups=conv_dim）padding=k-1
    in_proj.weight [10304, 2688]
    dt_bias (64,) · A_log (64,) · D (64,)            **三个都是逐头的**
    norm.weight    (4096,)           Zamba2RMSNormGated(4096, group_size=4096//8)
    out_proj.weight[2688, 4096]

### 三、算法（`forward` 的正文）

    gate, bc, dt = split(in_proj(x), [d_inner, conv_dim, num_heads])
    # ② depthwise 因果卷积 + 激活
    bc = conv1d(bc.transpose(1,2), weight.squeeze(1), bias,
                groups=conv_dim, padding=k-1)   -> 激活
    hs, B, C = split(bc, [d_inner, ng*ssm, ng*ssm])
    # ③ SSM scan
    scan = mamba2_chunk_scan(hs.view(B,S,H,hd), dt, A, B.view(B,S,ng,ssm),
                             C.view(B,S,ng,ssm), chunk_size, D,
                             dt_bias, dt_softplus=True)
    scan = scan.reshape(B, S, -1)
    # ④ gated RMSNorm，再 out_proj
    return out_proj(norm(scan, gate))

而 `A = -exp(A_log)` ✓、`dt` 走 **softplus** ✓、`D` 是**跳连** ✓。

### 四、关键：要生成的是**纯 torch 那一条**

`forward` 里有三条路：

    mamba2_split_conv1d_scan_combined   CUDA kernel（训练时）
    causal_conv1d_fn / chunk_scan       CUDA kernel
    **纯 torch 回退**                    判卷人在 CPU 上走的就是这条 ✓

**所以 by1 要生成的应该是回退那条的算法** ✓ —— 数学相同，
只是慢。而**递推形式和分块形式算的是同一个东西**：

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
    y_t = C_t · h_t + D * x_t

（B/C 按 `n_groups` 分组、每个头 `repeat_interleave` 到 `num_heads` ✓
—— 和 GQA 的 `repeat_kv` 同一套写法。）

**这条递推是可以在 CPU 上逐位验证的** ✓ —— 而它会和判卷人给出同一个数，
如果两边真的同源的话。**这正是判卷人的用处。**

### 五、还差的两样（下一轮做）

    ① `MoE` 的属性 `gate` —— Nemotron 的第二个缺口，小改动
    ② `Zamba2RMSNormGated` 的式子 —— 它在共享模块里
       （`from ... import`），不在 nemotron_h 里；实现前要先读它

### 六、这一轮没有动一行产品代码

**读完了、对上了、算法记在这里了。** 而这一步值得单独记的原因是：
**再往下写就是照抄，不再需要判断** ✓ —— 而"需不需要判断"是
一件事该不该现在做的分界。


---

## 91. Mamba2 的数学**验通了**，而且判卷人留下来了

### 一、先把数学验通，再碰 codegen

上一节把算法读全了。这一节**没有直接去改 `by1codegen`** —— 而是先写一个
原型：把要生成的那条递推单独实现一遍，和官方 `NemotronHMamba2Mixer` 对拍。

**理由是省事**：写进运行时模板之后再发现公式错，要排查的是
"我读错了哪一行" **加上** "模板怎么拼" 两件事 ✗。
而在原型里，两边都只有数学 —— **错了就是公式错** ✓。

### 二、走了四次才对，而四次都是形状

    ① 层类型写成 `mamba`                -> 官方只认 `linear_attention`
    ② `B[..., None] * x[..., None]`     -> 状态是外积，两个 `None` 加错了侧
    ③ `C[..., None, :] @ h`              -> 读出是 **h 右乘 C**，方向反了
    ④ 打印里 `proj` 写成 `d_inner + 2*ng*ssm + H`
       —— 中间那项**漏了 `d_inner`**，印出 100 而真值是 132 ✗

②③ 值得记：状态是 `[b, H, HD, SSM]`，于是

    · 更新是**外积**   x[HD] ⊗ B[SSM]
    · 读出是 **h 右乘 C**  h[HD,SSM] @ C[SSM,1]

**两次都把 `[..., None]` 加在了错的一侧** —— 而那种错**炸在维度上，
不会静默算错** ✓。这正是原型这一步的价值：形状问题挡在 codegen 外面。

④ 是另一类：**打印出来的数字也是要核的**。它不影响结果，但它会
让下一个读的人以为 `proj` 是 100 ✗。

### 三、结果

    小尺寸   d_inner 32   conv_dim 96    相对 1.241e-07  ✓
    真实尺寸 d_inner 4096 conv_dim 6144  相对 3.750e-07  ✓
            （proj 10304 —— 正好是 .by1 声明的那三个数）

**递推形式和官方同源**，两种尺寸都是 ✓。

而三条反例**都红了**：gated norm 的 eps 改成 4 倍 · 卷积核左右翻转 ·
dt 不做 softplus ✓ —— **一个碰不到被测东西的测试不是测试**。

### 四、判卷人留下来了：`src/modelcheck/by1ssm.py`

不是一次性脚本 ✓ —— 它进了 `modelcheck/` 那一族
（和 `by1mla` / `by1moe` / `by1rope` 并排），有 `--selftest`，
反例是它自己的一部分 ✓。

而它**明说自己不证什么**：

    ✓ 证   这条递推和官方逐位同源（两种尺寸）
    ✓ 证   这个对拍碰得到东西（三条反例）
    ✗ 不证 by1 生成出来的代码是对的 —— 那要 by1diff 接上去才验得到
    ✗ 不证 分块（chunked）形式和它一致 —— 递推是同一个数学，
           但官方走的是 kernel 那条路，浮点归约顺序不同

### 五、下一步

数学通了、判卷人立了，`by1codegen` 里加 `SSM` 那一支就**只剩照抄** ✓。
还要顺手补 Nemotron 的第二个缺口：`MoE` 的属性 `gate`。


---

## 92. SSM 的实现底稿 —— 三个缺口、八个张量、一处已经写好的映射

上一节数学验通了、判卷人立了。这一节把**要照抄的东西**找齐。

### 一、缺口是**三个**，不是一个

`compile_ir` 对 Nemotron 的完整抱怨：

    - 机制 'Mamba' 的类型是 'SSM'，codegen 还不支持
    - 机制 'MoE' 的属性 'gate' codegen 还不支持
    - **第 1 层的主机制是 'MoE'，codegen 不把它当 token 混合器**   <- 第三个

而层序列是干净的交替：`Mamba, MoE, Mamba, MoE, Mamba, Attn, MoE, ...`
—— **MoE 在这个模型里当 token 混合器用** ✓。

### 二、张量底稿：8 个，名字和形状全在

`info['tens']` 是契约的分组（每个不同的机制一组）。Mamba 那一组：

    组 'Mamba'   attrs: conv_kernel=4, head_dim=64, n_groups=8, ssm_state=128
                 层数 23
    in_proj.weight    (10304, 2688)
    conv1d.weight     (6144, 1, 4)
    conv1d.bias       (6144)
    A_log             (64)
    D                 (64)
    dt_bias           (64)
    norm.weight       (4096)
    out_proj.weight   (2688, 4096)

**和官方 `NemotronHMamba2Mixer` 逐字对上** ✓ ——
`10304 = 4096 + 6144 + 64` ✓、`6144 = 4096 + 2*8*128` ✓。

### 三、有两处**已经写好了**，不用动

    by1check.py:668    TOKEN_MIXER = {..., "SSM", "Vision", "Recurrent", ...}
                       —— `SSM` **已经被认成 token 混合器** ✓
    by1export.py:411   elif _mk in ("SSM",): out.append("mamba")
                       —— 导出层的 `layer_types` **已经会写 mamba** ✓

**所以缺的只在 `by1codegen` 这一层** ✓ —— 机制表、builder、以及
`MoE` 当混合器那一条。

### 四、`MoE` 的 `gate` 只是要接受 `none`

契约里 MoE 那一组的 attrs 写着 `gate=none`。所以那不是"要支持一种
新的门控"，是**要接受 `none` 这个取值** ✓ —— 而"不认识的属性一律
报不支持"这条规则把它也拦下了 ✗。

**这一条值得单独记**：`unsupported` 的判据是"名字在不在允许表里"，
而这里的问题是"**值**是 none" ✗ —— 名字和取值是两件事。

### 五、还差一步：逻辑名 -> 物理名

契约里的逻辑名是 `in_proj.weight`，而生成出来的模块叫
`layers.0.op0.in_proj.weight`。`by1load.py` 的头注里已经写过这件事：
"模型的参数名是后端自己的，而契约里的逻辑名差一个后缀" ✓ ——
所以那一层映射是现成的 ✓。

### 六、这一轮也没动产品代码，而这是**最后一次**

前两轮不动，是因为"还在判断"；这一轮不动，是因为**底稿刚找齐、
而它值得单独落一次**（下一轮如果实现到一半被打断，这张表还在）。

**再往下就是纯粹的照着写。**


---

## 93. `SSM` 落地了：Nemotron **编得出来**了

照着上一节那张底稿写，五处改动。

### 一、五处改动

    ① `SUPPORTED_KINDS` 加 `"SSM"`
    ② `ATTRS["MoE"]` 加 `"gate"` —— **只是让 `none` 通过**
    ③ `ATTRS["SSM"]` 新增（声明的十个属性）
    ④ `MIXER_KINDS` 加 `"SSM"` 和 `"MoE"`
    ⑤ `BUILDERS["SSM"] = Mamba2`，以及运行时模板里的两个类

`MIXER_KINDS` 加 `MoE` 的理由写在那里了：Nemotron 的层是
`Mamba / MoE / Mamba / MoE / Mamba / Attn / …` 交替的 —— **MoE 在那里
就是 token 混合器**（它把 token 之间混起来）。而混合器是**按位置挑**的，
所以加上它不影响别的模型 ✓。

### 二、`one_mech` 里没有 `SSM` 分支 —— 而它**一个字都没说**

第一版只改了上面五处，结果 Nemotron 报：

    - 第 0 层没有可用的 token 混合器 —— 生成不出来     × 23 层

`one_mech` 的最后是 `return None` ✓ —— 而 `SSM` 走到那里**不报任何原因**
✗。所以真正的错被 23 句"没有混合器"盖住了 ✗。

**这正是代码里那段注释早就警告过的形状**（GLM-5.3 报过 34 句
"第 N 层没有可用的 token 混合器"，而根因只有一句 KDA 不支持）。
补上 `SSM` 分支之后，它立刻说了真话：`机制 'Mamba' 的 SSM 缺少 heads` ✓。

### 三、而 `heads` 缺失是**真 bug**：裸数字被静默丢掉

`.by1` 里明明写着 `heads = 64` ✓，而属性表里既没有 `heads`
也没有展开出来的 `q` —— **一声不响** ✗。

`by1blocks.heads_of` 只认 `k = v` 那种写法：

    for part in split_top(body):
        i = part.find("=")
        if i > 0:                     # ← 裸数字 find("=") == -1
            out[part[:i].strip()] = eval_num(...)
    return out or None                # ← 于是返回 None

**`heads = 64` 就是那个裸数字** ✓ —— 意思是"头数是 64"，而它被
**静默丢掉** ✗。修法是给 `else` 分支加一条：裸数字进 `out["heads"]` ✓。

**这个 bug 影响的是所有机制**（`heads_of` 是共用的）—— 只是
在 `SSM` 之前没有哪个机制用裸数字写 `heads` ✓。

### 四、结果

    Nemotron  **编出来了** ✓  54 层
    第 0 层的算子图：RMSNorm(Norm) -> Mamba(SSM) -> Add

    档 4：64 项，0 项失败 ✓

### 五、`by1gate` 的那条期望**换了方向**

它原来写着：

    ('NVIDIA-Nemotron-...by1', False, 'SSM', 'Mamba 整族没实现')

—— **要求它被拒、且理由提到 SSM** ✓。而 `SSM` 实现之后这条就
**过时**了 ✗ —— 护栏编码的是**旧的真相**。

**它没有被我删掉，是换了方向** ✓：现在要求它**编得出来**。
少一边就等于把护栏拆了 —— 而那正是"一个不可能红的检查" ✓。

### 六、还差什么：**每个后端各要实现一次**

`by1exec`（NumPy）现在报：

    后端必须实现的算子：{'Norm': 54, 'SSM': 23, 'Add': 54, 'MoE': 24, 'Attention': 7}
    [缺口] 这个后端还没实现：['SSM']

**所以"编得出来"和"算得出来"是两件事** ✓ —— 这一节做到的是前者。
`SSM` 还要在 NumPy 后端和 C 后端各实现一次 ✓
（torch 侧已经好了 —— 运行时模板里那两个类就是）。

**而判卷人已经就位** ✓：`src/modelcheck/by1ssm.py` 把要实现的式子
和官方对拍过（两种尺寸 + 三条反例）。后端照着它写就行。


---

## 94. SSM 在 NumPy 后端落地，而且**两个后端一致**

上一节是 codegen 层（编得出来）。这一节是**算得出来**。

### 一、`op_ssm` + 参数表 + 登记

三处，照着 `op_linear`（GDN）的形状写：

    ① `op_ssm(P, a, ins, d)`  —— 递推，和 `by1ssm.py` 里那份逐字对应
    ② 参数表加 `SSM` 分支     —— 8 个张量，和契约那一组逐个对上
    ③ `OPS` 加 `"SSM": op_ssm`

### 二、真正的判据：**两个后端一致**

Nemotron 是 30B，**这台机器上建不出来**：

    numpy._ArrayMemoryError: Unable to allocate 2.38 GiB
    for an array with shape (128, 1856, 2688)

—— 那是"验不了"，不是"算错了" ✓。所以新建了一个最小的
**`models/ssm-shaped.by1`**（`d_model=128` / `heads=4` / `head_dim=16` /
`ssm_state=32` / `n_groups=2`），让它走完后端：

    后端必须实现的算子：{'Norm': 8, 'SSM': 4, 'Add': 8, 'FFN': 4}
    [OK] 全部已实现

    与 PyTorch 后端逐位对比（同一份 IR、同一组权重）
    最大绝对差 1.441e-07   相对 3.604e-07   **[PASS] 两个后端一致**

**而这两边又各自和官方 `NemotronHMamba2Mixer` 对过** ✓
（`src/modelcheck/by1ssm.py`，两种尺寸 + 三条反例）——
所以这条链是：by1-NumPy == by1-torch == 官方 ✓。

### 三、加一个机制要动的地方，比想的多

新模型一进来，`by1irentry` 立刻报：

    IRError: IR 不合法：
    layers[0].ops[1]: kind = <SSM> 不在闭集 ['Add', 'Atten...

**IR 规格有自己的闭集** ✗（`by1ir.KIND_ATTRS`）—— 加机制必须**同时**
登记进规格 ✓。这是好的失败：规格被强制执行了 ✓。

补上之后 `ir-spec.md` 就过期了 ✗ —— 而 `by1docs` 拿它和
`by1ir.spec_markdown()` **逐字符**比 ✓。**改一处、要同步三处。**

### 四、而 `ir-spec.md` 差点被我"重新生成"错

我跑了 `python src/by1ir.py --spec` ✓、rc=0 ✓ —— 但它是**打到 stdout**
✗，我顺手把输出重定向进了一个临时文件 ✗。于是文件没变、
`by1docs` 照样红 ✓。

**"跑了那条命令"和"命令改了那个文件"是两件事** —— 而判据（逐字符比）
没被骗 ✓。

### 五、还差：C 后端

`by1irentry` 现在说：

    ssm-shaped.by1    [不支持] C 后端还没有 SSM

—— **而它是"不支持"，不是"失败"** ✓（五态里那两种不能混）。


---

## 95. **`SSM` 三个后端齐了，而且三者一致** —— 目标第 2 项完成

上一节是 NumPy，这一节是 C。

### 一、C 那边三处

    `static void ssm(...)`   —— 递推，和 by1ssm.py 里那份逐字对应
    参数表 `elif k == "SSM"` —— 8 个张量
    发射   `elif k == "SSM"` —— 一行调用

而**比 GDN 简单**：`op_ssm` 每次前向把状态清零 ✓，所以不像 GDN
那样要跨调用留滑窗历史 ✓。

### 二、结果

    ssm-shaped.by1 -> C
      权重 55 个张量，生成 cgen/model.c（949 行）
      [编译成功] [运行成功] ok 8 x 512

    C 后端 vs NumPy 后端
    最大绝对差 1.739e-07   相对 4.349e-07   **[PASS] 三个后端一致**

**而这条链现在是完整的：**

    by1-C  ==  by1-NumPy  ==  by1-torch        （三者互拍）
                  \            /
                   ==  官方 NemotronHMamba2Mixer   （by1ssm.py，两种尺寸 + 三条反例）

**三个后端可以一致地错** ✓ —— 这句话这个仓库里写过很多次。
所以上面那条等式里**右边那一半才是判据** ✓：官方实现是外部的，
而三个后端是自己跟自己比 ✓。

### 三、C 里两处最容易写反的地方

    h 是 [nh][hd][ss]
      更新是**外积**   row[j] = dec*row[j] + dt * x[i] * B[g*ss+j]
      读出是 **h 右乘 C** s += row[j] * C[g*ss+j]

**写反了不会报错，会读错内存** ✗ —— 比 NumPy 那版更险
（那边维度不对会当场炸 ✓，这边只是算错）。

而 **B / C 按 `ng` 分组**：每个组服务 `rep = nh/ng` 个头 ✓ ——
`g = h / rep` ✓。这一处和 GQA 的 `repeat_kv` 是同一个形状。

### 四、目标第 2 项的账

    ✅ codegen 认得 SSM
    ✅ torch 运行时（Mamba2 + 按组 gated norm）
    ✅ NumPy 后端（op_ssm）
    ✅ C 后端（ssm）
    ✅ 判卷人 src/modelcheck/by1ssm.py（两种尺寸 + 三条反例）
    ✅ models/ssm-shaped.by1 —— 最小可跑，三个后端都走通

**"编得出来"和"算得出来"两件事都做到了** ✓。

### 五、还剩

    ① **KDA** —— 要 CUDA 机器才看得到语义（`fla` / `triton` 都装不了）
    ② **多模态** —— 语言层的建模，未开工

而 `by1irentry` 现在会印出每一份 `.by1` 走到了哪一步 ——
`GLM-5.3-Flash` / `Ling-3.0-tiny` 仍然停在 `[不支持] KDA` ✓，
**那是"不支持"，不是"失败"** ✓。


---

## 96. 多模态"没建模"到底是**哪一种没建模** —— 量出来是一段**死声明**

目标第 3 项要求"先量清楚'没建模'具体指什么"。量完了，答案是三者都不是，
而是更坏的一种。

### 一、官方那边有什么

    config                                 张量总数   多模态相关
    ---------------------------------------------------------------
    google__gemma-4-12B                        677         **11**
    stepfun-ai__Step-3.7-Flash                1471        **667**
    google__embeddinggemma-2                  1376        **963**（几乎全是 audio_tower）

Gemma 的 config 里有 `vision_config` / `audio_config` /
`image_token_id` / `vision_soft_tokens_per_image` ✓；
Step-3.7 有 `vision_config` / `image_token_len` / `vision_select_layer`
/ `understand_projector_stride` ✓。

### 二、`.by1` 那边：**一行都没有解出来**

    gemma-4-31B.by1      mechs = {GQA: Attention, Dense: FFN}
    Step-3.7-Flash.by1   mechs = {Attn, Dense, MoE}
    GLM-5.3-Flash.by1    mechs = {KDA, SparseMLA, MoE}
    **多模态相关的键：一个都没有**

### 三、而 `gemma-4-31B.by1` 里**写着**一段 `vision ViT {`

    94|   vision ViT {
    95|     n_layer  = 27
    96|     d_model  = 1152
    97|     head_dim = 72
    98|     patch    = 16
    99|     pooling  = 3
   100|   }

**它看起来像是在声明视觉塔的规模** ✓ —— 而它没有变成任何东西 ✗。

### 四、决定性的一问：删掉它，有没有任何输出会变

不能靠读代码判断"谁读了它" ✗ —— 做法是**删掉再比**：

    info 的 24 个键；有差异的 1 个：emit
      而 `emit` 里每一处差异都只是 `'line': 153 -> 146`
      —— **正好差 7**，就是我删掉的行数 ✓

**也就是说：那段声明的语义贡献是零** ✗✓✓。

而 `.by1` 里另一处提到 `vision` 的是
`use_bidirectional_attention = "vision"` —— 那是个**字符串取值**，不是引用 ✓。

### 五、泛化：全部 27 份扫一遍

解析器认的顶层关键字（从 `by1check` 的正则里读出来）只有六个：

    head · mech · memory · optimizer · residual · stack

而逐份扫"缩进 2 的 `词 词 {`"：

    27 份里，**不认识的顶层块只有一处** —— `gemma-4-31B.by1:94` 那 7 行 ✓

**所以这不是普遍现象，是一处孤例** —— 但它是**最坏的那一种"不出声"**：
**写了、看起来生效、而实际是零** ✗。读的人（包括写的人）会以为
视觉塔的规模已经声明过了。

### 六、和另外两个模型不是一回事

    gemma-4-31B      **写了一段死的** ✗
    Step-3.7-Flash   **根本没写**（视觉塔 667 个张量，一行没有）
    GLM-5.3-Flash    **根本没写**

**这两种要分开说** ✓ —— 一种要加规则让它出声，另一种是要真的建模。

### 七、"支持到哪一层"：这里先**说清楚做到哪一步**

    能描述（by1check / by1verify 覆盖）  —— 可做 ✓
    能算前向（patch embed + ViT 那些算子）—— **是另一件事**，且大得多 ✗

而**第一件事的前置**不是"加一个 vision 构造" ✗ ——
是**先让这类死声明出声** ✓。`by1lint` 有"十类不出声的写法"，
这该是第十一类：**顶层块的关键字不认识** —— 现在它是静默的 ✓。


---

## 97. 让死声明出声：`TOP_KEYWORDS`，以及那条假声明被删掉

### 一、这条规则**不属于 `by1lint`**

第一反应是加进 `by1lint` 的"十类不出声的写法" ✗ —— 但看了一遍，
那十条**全是关于 Python 源码的**（`except` / `open` / `assert` / …）✓。
而这一条是**语言层**的：`by1check` 用关键字逐个认顶层块，
不认识的**直接跳过、一个字都不报** ✗。

**谁解析谁报** ✓ —— 所以它进了 `by1check` ✓。

### 二、那十四个关键字不是抄的

从 `by1check` 认这些块的**字面量**里读出来的：

    emit · head · hparams · interop · mech · memory · model ·
    optimizer · position · residual · schedule · stack · state · tensors

少列一个的后果是**假报** —— 而门会跑全部 27 份 `.by1`，假报当场露出来 ✓。

### 三、判据：**两个方向 + 一个反例**

    ① `gemma-4-31B.by1` **该红**   -> 红了 1 处（第 94 行 'vision'）✓
    ② 其余 26 份**一个都不该红**   -> 26 份干净 ✓
    ③ 往一份好文件里塞 `nosuchblock Foo {`
       -> **红了** ✓（这条检查碰得到东西 ——
          一个碰不到被测东西的测试不是测试）

**②是这一条最难的部分** ✗ —— 加一条规则容易，而"它不误报"要跑全集才知道。

### 四、而那段假声明**必须删掉**，不能留着

`vision ViT { n_layer = 27, ... }` 现在是红的 ✓ —— 但真正的问题不是
它红不红，是它**在骗人** ✗：看起来像在声明视觉塔的规模，而实际是零。

删掉它，在原处留一段注释说明：

    · 它原来是什么、为什么是死的（实测：删掉只改行号）
    · 官方那边确实有什么（`config.vision_config`、11 个视觉张量、
      外加一个 `audio_tower`）
    · **真要建模时形状应该是什么** —— 一个 **vision 模态的 stack**
      （`stack` 本来就认模态，`by1blocks.MODAL` 里有 `vision`），
      而 ViT 那几层是现成的 Attention + FFN，**不需要新机制** ✓

### 五、所以"支持到哪一层"这个问题，答案现在具体了

    能描述  —— **可达** ✓：一个 vision 模态的 stack + 张量命名，
              用的都是现成的机制（Attention / FFN / Norm）
    能算前向 —— **另一件事** ✗：patch embed、位置编码、
              图像切块那套不在语言里

而**在建模之前必须先有这条检查** ✓ —— 否则下一次还会有人写一段
"看起来生效"的声明，而没有任何地方说它没生效 ✗。

### 六、门

    27 份：**一份都不红了** ✓
    档 4：64 项，0 项失败 ✓


---

## 98. 视觉塔到底是什么形状 —— 以及**我又写错了一句**

### 一、先纠正我自己

上一节我删掉那段死声明时，在注释里（和给你的说明里）写过一句
"它的数字和官方对不上" ✗ —— **那句是错的**。

我拿 `gemma-4-12B` 的 `vision_config` 去比了（`mm_embed_dim=3840`、
`mm_posemb_size=1120`），而那段声明是给 **`gemma-4-31B`** 的。
按 31B 的官方张量清单，它的数字**全对**：

    vision_tower.encoder.layers.{0..26}            **27 层** ✓
      input_layernorm.weight              [1152]   **d_model** ✓
      self_attn.q_norm.weight             [72]     **head_dim** ✓
      self_attn.{q,k,v,o}_proj.linear.weight  [1152, 1152]
      mlp.{down,gate,up}_proj.linear.weight   [1152,4304] / [4304,1152]
      post_attention_layernorm / pre_feedforward_layernorm /
      post_feedforward_layernorm                    ← Gemma 式三明治
    patch_embedder.input_proj.weight      [1152, 768]
    patch_embedder.position_embedding_table [2, 10240, 1152]
    embed_vision.embedding_projection.weight [5376, 1152]

**所以那段声明是"数字对、但没人读"** ✗ —— 而**这比编造的更险**：
一个编造的数字迟早会被查出来，一个**正确的**数字会让人以为
"已经声明过了" ✓。

（这一条正是这个仓库反复说的：**没验过的具体不要写进行文** ✓。
我这次犯的是它的近亲 —— 验了，但验错了对象。）

### 二、形状：**全部是现成的机制**

    27 层 pre/post-norm transformer   <- Attention + FFN，都支持 ✓
    q_norm / k_norm 每个头 72 维       <- `qk_norm` 支持 ✓
    SwiGLU（down/gate/up）            <- FFN 支持 ✓
    一个 patch 投影 + 一张位置表        <- 线性 + memory 查表
    **每层四个 norm**（input / post_attn / pre_ff / post_ff）
                                      <- Gemma 式三明治，声明得出来 ✓

**一个新机制都不需要** ✓ —— 所以"能描述"是可达的 ✓。

### 三、判卷人现成

    python src/checks/by1verify.py gemma-4-31B.by1 \
        refs/google__gemma-4-31B.config.json \
        --tensors refs/google__gemma-4-31B.tensors.json

**对着官方产物查名字和形状** ✓ —— 而 `gemma-4-31B` 现在只覆盖了
文本主干（530 个张量），**视觉那 356 个一个没查** ✓。

### 四、"支持到哪一层"—— 决定，以及理由

    能描述  —— **做这个** ✓
              第二个 stack + 张量命名，用现成的机制；
              判卷人是 `by1verify --tensors`（官方的 356 个张量）
    能算前向 —— **不做** ✗
              patch 切块、axial rope（`rope_type: axial`）、
              池化、`std_bias/std_scale` 那套归一化 ——
              **都不在语言里**，要加新机制

**理由**：这两件事的成本差一个数量级，而这一轮的量测已经把
"能描述"需要什么全找齐了 ✓。先把能验的那一层做出来，
比同时开工两件、两件都半途而废好 ✓。

### 五、顺带：`gemma-4-31B` 的视觉张量是 **356 个**

而我第一次数出来是 11 个 ✗ —— 那是因为我把 JSON 的结构读错了
（以为是 `{tensors: {...}}`，实际是**扁平的 `{名字: {shape, dtype}}`** ✓）。
**读错结构量出来的数字是假的** ✓ —— 这个坑这个仓库里踩过不止一次。


---

## 99. 视觉塔量清楚了，而它卡在**一处具体的语言缺口**上

上一节把形状量出来了。这一节试着写，结果**写不出来** —— 而卡住的地方
很具体。

### 一、先确认两件前提，都满足

    多 stack 支持 ✓    Nemotron / Qwen3.6 / Qwen3.8 各有 2 个 stack
    层内四个 norm 不用声明 ✓
                        `input_layernorm` / `post_attention_layernorm` /
                        `pre_feedforward_layernorm` / `post_feedforward_layernorm`
                        来自**内建的层规则**，而 `by1verify` 已经在文本那
                        530 个张量上验过这条规则 ✓

### 二、照着写了一个，它**解析得过**

    mech ViT     : Attention { heads = { q = 16, kv = 16, head_dim = 72 },
                               out_dim = 1152, qk_norm = true }
    mech ViTMlp  : FFN { hidden = 4304 }
    stack vision { pattern = 27 * ViT
                   vision[:] >> ViTMlp }

    mechs  = {GQA, Dense, ViT, ViTMlp}
    stacks = [('main', '', 60), ('vision', '', 27)]      **两个栈都认了** ✓

**机制全是现成的** —— 不需要新种类 ✓。

### 三、而契约写不出来：`d_model` 是**模型级**的

    ('E', 19, 'layers', 'hparams.n_layer = 60，但主栈合计 87 层')
    ('W', 122, 'tensors', 'mech ViT 无张量契约')

第一条有现成解法（`aux = true`，MTP 栈就是这么写的 ✓ ——
视觉塔**在语义上也确实不是解码层** ✓）。

而第二条卡住了：

    张量形状里的 `d_model` 是**模型级**的（5376）
    而视觉塔宽 **1152**
    `out_dim` 只管"注意力输出宽度 ≠ d_model"
    **没有"给一个栈自己的输入宽度"的写法** ✗

**所以"能描述"这一步的真实前置不是"再写一段 `.by1`"** ✓ ——
是**给 stack 加一个宽度** —— 一行语言改动 ✓。

### 四、这一节没留下代码，留下的是**一个准确的定位**

那段测量写进了 `gemma-4-31B.by1` 的注释里（356 个张量的形状、
27 层一致、和文本层同形、判卷人是谁）✓ ——
而 `ViT` / `stack vision` 那几行**撤掉了** ✗：它们现在会让门红，
而红的理由（"契约写不出来"）不是一两行能补上的 ✓。

**这比留一段半成品好** ✓ —— 半成品会让人以为"快好了"，
而实际差的是一个语言设计决定 ✓。

### 五、下一步的形状

    ① 给 `stack` 加一个宽度（例如 `d_model = 1152`）
    ② 把那三个 mech + 两个 stack 写回去
    ③ 判据：`by1verify --tensors` 要覆盖到 **530 + 356 = 886** 个张量

**判卷人早就现成** ✓ —— 这一步不需要任何新判据。


---

## 100. 给 `stack` 加一个宽度，视觉契约就**全对了**

上一节定位到：卡住的是"张量形状里的 `d_model` 是模型级的"。
这一节把它补上 —— 一处语言改动，两处代码。

### 一、改动

    by1stacks   读每个栈的 `d_model`（没写就是没有 -> 映射为空）
                建 `stack_d_model` 和 `mech_d_model` 两张表
    by1tens     逐类实例化时，若那个机制所在的栈有宽度，
                就用它覆盖 `_['sym']["d_model"]`
                宽度不变式那条也改用**有效**宽度

写法：

    stack vision {
      d_model = 1152
      aux     = true
      pattern = 27 * ViT
      vision[:] >> ViTMlp
    }

**没写 `d_model` 的栈行为一个字都不变** ✓ —— 所以这个改动对现有
27 份 `.by1` 是空的（映射为空），而门证实了这一点（29 份全跑过、
档 4 仍然 64 项 0 失败）✓。

### 二、而第一版漏了一处：`>>` 挂上去的机制

`by1stacks` 建的映射只走了 `pattern` 里的机制 ✗ —— 而
`vision[:] >> ViTMlp` 里的 `ViTMlp` **不在 pattern 里** ✗。

实测就露出来了：

    ViT      q_proj (1152, 1152) ✓
    ViTMlp   gate_proj (4304, **5376**) ✗   <- 拿到的是模型的宽度

而 `by1tens` 里正好有挂载规则（栈名 + 目标机制），补一句就是 ✓。

### 三、补完之后

    ViT     层 27   q/k/v/o_proj (1152, 1152) · q_norm/k_norm (72)
    ViTMlp  层 27   gate/up_proj (4304, 1152) · down_proj (1152, 4304)

**和官方那 356 个张量逐个对上** ✓ —— 报告空、档 4 全绿 ✓。

### 四、判卷人说话了，而且给出了**可量的下一步**

    [PASS] 530 个张量的名字与形状全部一致
    契约**未覆盖**的实物张量: 658 个（共 1188 个里的 55%）
    未映射机制跳过 243
    命名规则来源: by1 emit[torch.module]
                  name = "model.language_model.layers.{i}.{scope}.{logical}.weight"

两件事：

    ① 视觉那 27 x 13 = 351 个张量落在"**未映射机制跳过**"里 ✗ ——
       因为 `emit` 的命名规则只管 `language_model`
    ② `by1verify` **只比 `text_config`** ✗（它自己印了这句话）

**所以下一步的判据是现成而且可量的** ✓：

    补上 `emit` 的命名规则之后，
    "契约未覆盖的实物张量" 应该从 **658 掉到约 302**（少 356）。

**这个数字是检查自己算的** ✓ —— 不需要新写判据，只要看它变不变 ✓。


---

## 101. 命名规则补上了：530 -> 773，而未覆盖 658 -> 415

### 一、判据先立着，然后达成

上一节说"补上 `emit` 的命名规则之后，未覆盖应该从 658 掉到约 302"。
结果：

    契约声明存在: 530 -> **773**    名字+形状一致 773   形状不符 0   缺失 0
    [PASS] 773 个张量的名字与形状全部一致
    契约**未覆盖**的实物张量: 658 -> **415**（55% -> 35%）

**那 243 个原来落在"未映射机制跳过"里的，现在全进契约、而且全对** ✓。

### 二、改动其实很小

    ① 契约里把视觉的逻辑名写成带点的：`q_proj.linear` 而不是 `q_proj`
    ② `emit` 里加一条 `name_vision = "model.vision_tower.encoder.layers.
       {i}.{scope}.{logical}.weight"`

第 ② 条不用改语言 ✓ —— `render_name` 里的优先级本来就是
`expert_name > name_<栈名> > name`（`by1check.py:157`），
而 Nemotron 的 `name_model` / `name_mtp` 就是这么用的 ✓。

第 ① 条也不用改语言 ✓ —— 契约里带点的逻辑名本来就有先例
（`shared_experts.gate_proj.weight`）✓。而层内那四个 norm **不走这条**
（它们来自内建的 `LAYER_DEFAULT`），所以不会被加上 `.linear` ✓。

### 三、而我预测的 302 错了 113 —— 错在**假设**

差额是 **27 x 4 = 108**（加几个零头）= 视觉的层内 norm ✗。
看清单才发现：**连文本的层 norm 也没覆盖** ✗

    60  model.language_model.layers.N.input_layernorm.weight
    60  model.language_model.layers.N.post_attention_layernorm.weight
    60  model.language_model.layers.N.pre_feedforward_layernorm.weight
    60  model.language_model.layers.N.post_feedforward_layernorm.weight
    60  model.language_model.layers.N.layer_scalar

**我上一轮写的是"这几个 norm 已经被覆盖，`by1verify` 已经在 530 个
张量上验过这条规则"** ✗ —— 那句是**假设的**：我看到它们在**官方清单**里，
就以为契约里有 ✓。

**而真正难看的地方是**：这个缺口 **`by1verify` 一直在印** ✓ ——

    [覆盖缺口] 参考产物里有 16 类张量没有契约对应物:
    （这是**范围边界**，不是失败 —— 但必须看得见）

**是我不看，不是它没说。** 这比"检查没报"轻，但轻不了多少 ——
一个印在那里、而我读过去的东西，和一个没印的东西，
对结果来说是一样的。

（`pre_feedforward` / `post_feedforward` 这两个 norm 是 Gemma 家族独有的：
`use_double_wide_mlp` / `hidden_size_per_layer_input` 只在
`google__*` 的 config 里出现 ✓。所以那 240 个是 Gemma 特有的四个 norm。）

### 四、下一步

    415 = 文本 4 norm x 60 = 240
        + 视觉 4 norm x 27 = 108
        + layer_scalar x 60
        + patch_embedder / std_bias / embed_vision 那几个

把 Gemma 那四个 norm 和 `layer_scalar` 声明进 `tensors { layer { ... } }`，
415 应该掉到 5 上下（只剩 patch embedder 那几个真正没建模的）。
**而判据还是同一个数** ✓ —— 不需要新写。


---

## 102. 补 `layer` / `global` 那一步：**试了两次，都撤了**

### 一、想做的是什么

`by1verify` 一直印着"契约未覆盖"，而里面是：

    文本 4 个 norm x 60 = 240     （Gemma 是三明治：attn 前/后、ff 前/后）
    视觉 4 个 norm x 27 = 108
    layer_scalar x 60
    embed_tokens / norm / patch_embedder 那几个

**目标**：把这些声明进 `tensors { layer { ... } }` / `global { ... }`，
让 415 掉到 5 上下。判据还是同一个数、不需要新写 ✓。

### 二、第一次：声明了，**数一点没变**

在 `tensors` 里加了 `layer { … }` 和 `global { … }` 两块 ✓ ——
而 `415` 一个没动 ✗。

查下去发现：`by1verify` 的 `check_tensors` 是**按 owner 查 `scope` 表**的

    if scope_map and owner not in scope_map:
        unmapped += 1
        continue

而 layer 行的 owner 是 `"layer"` ✗ —— 不在我的 `scope { GQA, Dense, ViT, ViTMlp }`
里 ✓ —— **于是全部被当成"未映射"跳过** ✗。

**所以"写出来"和"验得到"是两件事** ✓ —— 而表现是"声明好像没生效" ✓，
看起来像解析器的问题，实际是**映射表少两行** ✓。
（`mla-shaped.by1` 里一直写着 `layer = "", global = ""` ✓ —— 我读过那一段 ✗。）

### 三、第二次：加进 `scope`，把物理名弄坏了

    scope { GQA = self_attn, ..., layer = "", global = "" }

结果：

    契约声明存在: 1208   名字+形状一致 **0**   缺失 1208
    契约未覆盖: 1188 个（100%）

**`scope` 的值不只用于命名模板** ✗ —— 它也**参与物理名**。
而 `scope = ""` 让 layer 行渲染成 `layers.N..input_layernorm.weight`（双点）✗。

我接着试了把约定改成 `{scope}{logical}`（值自带尾点，`mla-shaped` 的写法 ✓）——
结果 **2 / 1210** ✗（只有 `global` 那两行对了 ✓）。

**两次都把门弄红了** ✗。**撤回到 773 / 415 / 档 4 全绿** ✓。

### 四、这一轮真正的产出：**两件下一个人必须知道的事**

    ① `by1verify` **按 owner 查 `scope`** —— 不在表里就静默跳过。
       所以往 `tensors` 里加 `layer` / `global` **不会自动被验** ✗，
       而表现是"数没变"，看起来像解析没生效。
    ② `scope` 的值有**两种用途**（命名模板 + 物理名）✗ ——
       改它的格式会同时动到两边。要加 `layer` / `global`，
       得先读懂它在 `by1check` 里到底被消费了几处 ✓。

**这不是"没做成"，是"知道了它为什么不是加两行"** ✓ ——
和前面那次"视觉塔卡在语言缺口"同一形状：**先说清楚做到哪一步**，
比留一段看起来快好了的半成品好 ✓。

### 五、现在的位置

    契约声明存在: 773   名字+形状一致 773   形状不符 0   缺失 0  ✓
    契约未覆盖: 415（35%）
    档 4: 64 项，0 项失败 ✓

那一轮的目标（命名规则 → 773）**已经达成并留下** ✓；
而"补 layer/global" 是**下一个**目标，判据还是那个数（415 → 5）。


---

## 103. `layer` 块：**成了大半，又撤了** —— 而量到的东西留下了

### 一、上一轮我说错了一句，先纠正

上一轮我写"`scope` 的值不只用于命名模板，也参与物理名" ✗ ——
**那句是错的** ✓。这一轮把它的消费者找齐了：

    by1check.py:140     scope = rule.get("scope", {}).get(mech, "")   <- 唯一一处
    by1verify.py:314    按 owner 查表，不在表里跳过
    by1emit.py:135      读了 scope 却**没用它** ✗（死变量）

**真正消费它的只有 `render_name` 一处** ✓ —— 而 `by1verify` /
`by1emit` / `by1load` / `by1instella` 全都经由它 ✓。

### 二、于是点的约定可以改，而且改对了

    name  = "…layers.{i}.{scope}{logical}.weight"
    scope { GQA = self_attn., Dense = mlp., … }

    契约声明存在: 773   名字+形状一致 773   缺失 0   未覆盖 415  ✓

**和改之前一模一样** ✓ —— 说明上一轮那个"2 / 1210"不是点的约定造成的 ✗，
是**同时加的 `rename` / `global_name`** ✓。

**所以"同时改两处"是上一次失败的原因** ✓ —— 而它伪装成"新写法不对" ✓。

### 三、`layer` 块：一步步量出来的三条

写上 `tensors { layer { … } }` 之后：

    第一次   数一点没变 ✗   -> owner `layer` 不在 `scope` 表里，被跳过
    加进表   1208 声明 / 773 对上 / **435 缺失** ✗
             看渲染出来的名字：`…layers.0.input_layernorm.weight.weight` ✓
             -> **逻辑名要写裸名**，模板会加 `.weight`
    改裸名   1121 声明 / 1013 对上 / **108 形状不符** ✗
             -> 108 = 27 x 4：视觉那四个 norm，`layer` 行只算一次、
                用的是模型的 `d_model`（5376），而视觉栈宽 1152

**未覆盖从 415 掉到 67** ✓ —— 这一步是真的成了 ✓。
而 108 那处没修成 ✗。

### 四、`layer_scalar` 表达不了

它的官方名字是 `…layers.N.layer_scalar` —— **没有 `.weight` 后缀** ✗ ——
而命名模板无条件加那一段 ✓。要它得有一条"不加后缀"的规则。

（60 个 `layer_scalar` 因此留在未覆盖里 ✓ —— 而那 67 里它占 60 ✓。）

### 五、我试的修法**没生效**，而且没查到原因

给 `by1tens` 补了一张按栈的 `layer_rows_by_stack` ✓（改 `_['d_model']`
再重算一遍 ✓），`by1check` 里按 `_['s']` 取 ✓ —— 而 `layer_out` 里
**仍然是 5376** ✗。查了两处都没看出为什么 ✗。

**那一版会让门变红，所以整体撤了** ✓ —— 撤回到 773 / 415 / 档 4 全绿 ✓。

### 六、留下的东西

`gemma-4-31B.by1` 的 `emit` 块里现在有一段注释，写着：

    ① `layer` 必须也在 `scope` 映射里（否则静默跳过）
    ② 逻辑名要写裸名（模板会加 `.weight`）—— 这条是**看渲染出来的名字**
       才发现的，读代码看不出来
    ③ `layer_scalar` 表达不了（没有 `.weight` 后缀）
    ④ 做了 ①② 之后 415 -> 67 ✓，而 108 个形状不符来自"layer 行只算一次"
    ⑤ 接着做的入口是两处：`by1tens` 的 `layer_rows`、
       `by1check` 里逐层拼 `layer_out` 那一段

**这一轮没留下代码，留下的是"下一步从哪儿进"** ✓ —— 而这个仓库里
已经有过教训：没有这个的下一步，就是重走一遍 ✓。

### 七、位置

    契约声明存在: 773   名字+形状一致 773   缺失 0  ✓
    契约未覆盖: 415（35%）
    档 4: 64 项，0 项失败 ✓


---

## 104. 那 108 修好了 —— 而根因是**我改了一个没人读的名字**

### 一、上一轮的疑点：按栈的表补了，而它没生效

上一轮我写了 `layer_rows_by_stack` ✓，算出来的却还是 `(5376)` ✗ ——
查了两处没看出为什么，于是整体撤回 ✓。

这一轮带着探针跑，一眼就看到了：

    layer_rows           = [('input_layernorm', '(5376)', …)]
    layer_rows_by_stack  = {'vision': [('input_layernorm', '(5376)', …)]}   ✗

### 二、根因

`_flat_rows` 的第一行注释**早就写着**：

    def _flat_rows(entry_list):
        """层级 / 全局张量：形状只用 hparams 求值（没有逐层属性）。"""
        for _['k2'], _['v2'] in _['hp'].items():      <- **读 hp**
            _['n2'] = eval_num(_['v2'])

**而我换的是 `_['d_model']`** ✗ —— 那个名字在 `_flat_rows` 里
**一次都没出现** ✓。所以"改了跟没改一样"。

**这是这个仓库里反复出现的形状**：一个名字看起来是那个意思 ✓，
而真正被读的是另一个 ✓。上一次是 `scope`（我以为它参与物理名 ✗），
这一次是 `d_model`（我以为层级形状读它 ✗）。

**两次的共同点是**：我没看**读它的那一行**，而是从名字推断 ✗。

### 三、改法与判据

    _['hp'] = dict(_['hp']); _['hp']["d_model"] = _fmt(_['_dm'])
    … 算一遍 …
    _['hp'] = _['_sav_hp']       # 还回去

**而第一次连这个也错** ✗ —— 我把 `_['_dm']`（float）直接塞进 `hp` ✓，
而 `hp` 的值**全是字符串** ✗（`eval_num` 会对它 `.replace` ✓）：
`AttributeError: 'float' object has no attribute 'replace'` ✓。
用 `_fmt` 包一下就好 ✓。

### 四、结果

    契约声明存在: 1121   名字+形状一致 **1121**   形状不符 0   缺失 0
    [PASS] 1121 个张量的名字与形状全部一致 ✓
    契约**未覆盖**: 415 -> **67**（35% -> 6%）

**每一处都对上了** ✓ —— 文本主干 60 层 x 4 个 norm、视觉 27 层 x 4 个、
两边的机制张量，全在里面 ✓。

剩下的 67：

    60  layer_scalar        —— **表达不了**（官方名字没有 `.weight` 后缀，
                               而命名模板无条件加；要它得有一条"不加后缀"的规则）
     5  patch_embedder / embed_vision / embed_tokens / norm
                            —— 真正还没建模的那几个
     2  别的零头

### 五、这一轮改了三个文件

    models/gemma-4-31B.by1   加回 `layer` 块（裸名）+ scope 里加 `layer`
    src/lang/by1tens.py      layer_rows_by_stack：换 `_['hp']` 而不是 `_['d_model']`
    src/checks/by1check.py   逐层拼 layer_out 时按栈取那张表

而**语言本身没动** ✓ —— `stack` 的 `d_model` 是上一轮加的 ✓，
这一轮只是把它**接到了层级张量上** ✓。

### 六、两次"改了没生效"，两次根因同类

    上一轮  `scope`  —— 以为它参与物理名      实际只有 render_name 读
    这一轮  `d_model` —— 以为层级形状读它      实际读的是 hp

**共同点：从名字推断谁读它，而不是去看读它的那一行。**
而两次的代价都是"撤一整轮" ✓ —— 除了这一次没撤 ✓。


---

## 105. 换成标准命名约定，未覆盖 415 -> **7**

### 一、先问了一句：带 `.weight` 之外的张量怎么表达

`layer_scalar` 的官方名字**没有 `.weight` 后缀** ✗，而 Gemma 的模板是
`{scope}.{logical}.weight` ✗ —— 无条件加那一段。想补它，得先知道
**别的模型是怎么表达"没有 `.weight` 的张量"的** ✓。

答案在 `gpt2.by1` 里，而它一直写着：

    name  = "h.{i}.{scope}{physical}"                    <- {physical}，不是 {logical}.weight
    scope { Attn = attn., MLP = mlp., layer = "", global = "" }   <- 值自带尾点
    tensors { Attn { c_attn.weight : (…) ; c_attn.bias : (…) } }  <- **逻辑名含后缀**

**这套约定下，没有后缀的张量自然表达得出来** ✓ —— 而 Gemma 用的是
"裸逻辑名 + 模板加后缀" ✗，那套表达不了 ✗。

### 二、这次**一次只改一件事**

上一轮我在同一个改动里换了点的约定**又**加了 `rename`/`global_name` ✓ ——
结果是 2/1210 ✗，而我把原因归到了点上 ✗（错的）。

这次拆开：

    第一次（只有点的约定）  773 / 415   **和改之前一模一样** ✓
    第二次（加上逻辑名带后缀 + {scope}{physical}）
                           -> 348 对上、773 缺失 ✗

**348 正好是层级那 240+108** ✓ —— 说明 **layer 行通了**、而机制行断了 ✓。

### 三、断的原因，又是"看渲染出来的名字"

    L0  GQA.q_proj.weight -> model.language_model.layers.0.q_proj.weight
                                                           ^^^^^^^ **scope 丢了**

**因为我把 `scope { … }` 折成了两行** ✗ —— 而它没被解析 ✓。
（**不是**约定的问题 ✓。）

改回一行：1121 / 1121 / 67 ✓ —— 和切换前等价 ✓，而**约定已经是标准的那套** ✓。

### 四、于是 `layer_scalar` 可以写了

    layer_scalar : (1,) unless Dense

`unless <机制>` 的语义是"**只在挂了它的时候**" ✓ —— 而视觉栈挂的是
`ViTMlp` ✓、文本栈挂的是 `Dense` ✓ —— 所以这一条只落在文本那 60 层上 ✓。

（第一次没写 guard：`缺失 27` ✗ —— 官方的视觉层里没有 `layer_scalar` ✓。）

    契约声明存在: 1181   名字+形状一致 1181   缺失 0  ✓
    契约未覆盖: 67 -> **7**（6% -> **1%**）

### 五、而 `ggml` 那边被我一起弄坏了

    0 一致 / 530 缺失 ✗

**`rename` 的键是逻辑名** ✓ —— 逻辑名带上 `.weight` 之后，
ggml 那七条 `q_proj = q` 一条都查不到 ✗。补上后缀 -> **410** ✓。

而还剩 `缺失 120` ✗ —— **120 = 60 x 2**，正好是 `q_norm` / `k_norm` ✗ ——
GGUF 里它们叫 `blk.N.attn_q_norm.weight` ✓，而我的规则渲染成
`…attn_q_norm.weight.weight`（双后缀）✗。补两条 rename -> 绿 ✓。

### 六、结果

    档 4: 64 项，0 项失败 ✓
    torch.module  1181 / 1181 / 缺失 0
    未覆盖 658 -> 415 -> 67 -> **7**

    剩下那 7 个是真的还没建模的：
      patch_embedder.input_proj / position_embedding_table
      std_bias / std_scale
      embed_vision.embedding_projection
      embed_tokens / norm（这两个要 `rename`，是另一件事）

### 七、这一轮的账

**三次"改了没生效"，三次都是同一个形状**：

    上上轮  `scope`      —— 以为它参与物理名        实际只有 render_name 读
    上一轮  `d_model`    —— 以为层级形状读它        实际读的是 hp
    这一轮  `{scope}` 两行 —— 以为折行没关系        实际没被解析

**前两次是从名字推断谁读它；这一次是"看起来等价的写法"其实不等价。**
而三次的代价都是"看渲染出来的名字"就能立刻定位 ✓ ——
**那个输出一直在印，只是我没看** ✓。


---

## 106. **KDA 根本没有卡在 CUDA 上** —— 那个前提是错的

### 一、我一直在重复的一句

从更早的几轮起，仓库里和我给你的说法一直是：

    KDA 卡在 CUDA：`fla.ops.kda.chunk_kda` 要 Triton，
    `fla`/`triton` 没装，`torch 2.13.0+cpu`，`cuda.is_available() = False`。
    **看不到源码，所以不能写实现 —— 写了就是猜。**

`gpu/by1kda.py` 的整个 docstring 就是围着这句写的 ✓。

### 二、这一轮问了一句：**"看不到"具体指什么**

卡住的到底是**跑**它，还是**读**它 ✓？

    跑 chunk_kda   -> 要 Triton、要 CUDA        ✓ 真的卡住
    读它的源码     -> 要什么？                    <- **没问过**

试了一下：

    $ pip download flash-linear-attention --no-deps --no-binary :all:
    -> flash_linear_attention-0.5.2.tar.gz      208 KB   ✓
    $ pip download fla-core --no-deps
    -> fla_core-0.5.2-py3-none-any.whl          819 KB   ✓

**PyPI 是通的** ✓ —— 而且 `fla/ops/kda/` 下有 **20 个文件** ✓，
其中包括：

    fla/ops/kda/naive.py                        6339 字节

**那是纯 PyTorch 的参考实现** ✓ —— 没有 Triton、没有 CUDA ✓，
`einops` 装了就能在 CPU 上跑 ✓（实测跑通了 ✓）。

### 三、语义就在那 166 行里

`naive_recurrent_kda` 的核心是五行（第 55-66 行）：

    S = zeros(B, HV, K, V)
    for i in range(T):
        S = S * exp(g_i)                                    # 逐维衰减
        S = S + (beta_i * k_i) ⊗ (v_i - (k_i ⊗ S).sum(-2))   # delta 规则
        o_i = q_i @ S                                       # 读出

`q` 先乘 `1/sqrt(K)` ✓，`q`/`k` 按 `G = HV/H` 复制（GVA）✓。
`naive_chunk_kda` 是同一件事的分块写法 ✓。

**所以我一直说"看不到"的东西，166 行就写完了** ✓。

### 四、判卷人的地基：两种写法必须一致

    第一次比    差 1.9e+04   ✗  -> 我以为语义不同
    看清楚      |o1| 最大 6.0e+09，**相对差 3.2e-06**   ✓

**我比的是绝对值** ✗ —— 而那个配置下前向本身放大到 6e9 量级 ✓
（`beta ~ U(0,1)` 不衰减 ✓、`k`/`v` 是 `randn` ✓，`g` 压不住 ✓）。

**相对差 3.2e-06** ✓ —— 两种写法一致 ✓。
**这就是 KDA 判卷人的地基**：递归式对分块式，两边都在 CPU 上 ✓。

### 五、所以 item 1 的状态变了

    之前：**卡住**（等一台 CUDA 机器）
    现在：**没卡** ✓ —— 语义在 `naive.py` 里，参考在 CPU 上跑得动

要做的变成三件普通事：

    ① by1codegen 加 KDA 分支（张量名和形状已经从 `fla/layers/kda.py` 量出来了：
       q/k_proj -> key_dim、v_proj -> value_dim、f_proj 是两层 Sequential、
       b_proj -> num_v_heads、A_log -> num_v_heads、o_proj <- value_dim）
    ② by1exec 加 op_kda（按上面五行）
    ③ 判卷人：**对 `naive_recurrent_kda`**，并证明它在反例上会红
       （把 `exp(g)` 换成 `g`、把 delta 项去掉 —— 两个都该红）

### 六、这一轮改了什么

    gpu/by1kda.py   开头加了一段"**这个脚本的前提是错的**"，
                    写明正确的取法（两条 pip download）、
                    语义在哪几行、以及这个脚本仍然有用
                    （它 dump 的是 **Triton 核**的输出，那是官方实现；
                      `naive.py` 是它自己的参考 —— 两者都值得比）

**没改任何检查、没动任何语言** ✓ —— 这一轮的产出是**把一句错了很久的
前提拆掉** ✓。而它的形状和前面几次一样：

    我以为"卡住了"的那个东西，其实只是**我没试另一条路**。


---

## 107. KDA 的语义，**五处出处，全部从源码读出来**

上一节把"看不到源码"这个前提拆了 ✓。这一节把**看到的东西**落到仓库里 ✓ ——
不然下一轮又要重新下载一遍 ✓。

### ① 模块（`fla/layers/kda.py:142-176`）

    q_proj = Linear(hidden, key_dim)      key_dim   = num_heads   * head_k_dim
    k_proj = Linear(hidden, key_dim)
    v_proj = Linear(hidden, value_dim)    value_dim = num_v_heads * head_v_dim
    q/k/v_conv1d = ShortConvolution       # 只有 use_short_conv 时
    f_proj = Sequential(Linear(hidden, head_v_dim),
                        Linear(head_v_dim, gate_dim))     # gate_dim = num_v_heads * head_k_dim
    b_proj = Linear(hidden, num_v_heads)
    A_log  = Parameter(zeros(num_v_heads))
    o_proj = Linear(value_dim, hidden)

    head_k_dim = head_dim        head_v_dim = head_dim * expand_v

**`f_proj` 是两层** ✓ —— 这一条光看 config 是猜不出来的 ✓。

### ② 前向（`fla/layers/kda.py:223-297`）

    q, k, v = conv1d(proj(x))   或   silu(proj(x))    # 没有 short conv 时是后者
    g    = f_proj(x)          # 按 head_k_dim 重排成 [.., HV, K]
    beta = b_proj(x)          # [.., HV]

而**四个开关层里全是 True**（第 270-276 行）：

    use_qk_l2norm_in_kernel    = True     <- q/k 做 L2 归一化
    use_gate_in_kernel         = True
    use_beta_sigmoid_in_kernel = True     <- beta 进来是 logits
    state_v_first              = True     <- 状态是 [K, V]

**这四条每一条写错都会静默算错** ✓ —— 而它们的值只能从这一处看到 ✓。

### ③ 门（`fla/ops/kda/gate.py:50-54`）

    g = g + dt_bias.view(H, -1)
    g = -A_log.view(H, 1).exp() * softplus(g)

`lower_bound` 那支（第 69 行）换成
`g = lower_bound * sigmoid(A_log.view(H,1).exp() * g)` ✓。

### ④ 递推（`fla/ops/kda/naive.py:55-66`）—— **这就是定义**

    S = zeros(B, HV, K, V)
    for i in range(T):
        S = S * exp(g_i)                                     # 逐维衰减
        S = S + (beta_i * k_i) ⊗ (v_i - (k_i ⊗ S).sum(-2))    # delta 规则
        o_i = q_i @ S                                        # 读出

`q` 先乘 `1/sqrt(K)` ✓、`q`/`k` 按 `G = HV/H` 复制（GVA）✓。

### ⑤ 判卷人

    naive_recurrent_kda vs naive_chunk_kda   相对差 3.2e-06 ✓（实测）
    by1 的 op_kda       vs naive_recurrent_kda     <- **要做的**

反例（判卷人必须在这些上变红）：把 `exp(g)` 换成 `g` ✗ ·
把 delta 那一项去掉 ✗ · `q` 不除 `sqrt(K)` ✗。

### 这一轮改了什么

`gpu/by1kda.py` —— 把上面五节写进它的 docstring ✓。
那个文件原来整篇是"我看不到所以不能写" ✗，现在整篇是"这是它，出处在这" ✓。

**一行实现代码都还没写** ✓ —— 而这一轮之后，
写实现不再需要任何外部信息 ✓。**这正是"先把判卷人立起来"的意思** ✓。


---

## 108. KDA 最后一块：`g_proj` 是**输出门**，而 GLM 的契约少了 5 个张量

### 一、先量了两个模型的**真实** KDA 布局

    GLM-5.3-Flash 第 0 层，29 个张量，其中 KDA 相关 15 个：
      self_attn.A_log · b_proj.weight · dt_bias
      self_attn.f_a_proj.weight · f_b_proj.weight
      self_attn.**g_a_proj.weight · g_b_proj.weight**      <- 契约里没有 ✗
      self_attn.q/k/v_proj.weight · q/k/v_conv1d.weight
      self_attn.o_norm.weight · o_proj.weight

    Ling-3.0-tiny 第 0 层，18 个张量，KDA 相关 13 个：
      attention.A_log · b_proj.weight · dt_bias
      attention.f_proj.weight · g_proj.weight              <- 契约里有 ✓
      attention.q/k/v_proj.weight · q/k/v_conv1d.weight
      attention.o_norm.weight · o_proj.weight

**Ling 的契约和产物逐字对上** ✓（所以它 100% 覆盖 ✓）；
**GLM 的契约少 5 个** ✗ —— `b_proj` 和两对 `f_*` / `g_*` ✓。

**而 `by1verify` 一直在说**：GLM 的"契约未覆盖"是 **51%** ✗。
它没说谎 ✓ —— 它说的是"**声明过的 37534 个全对**" ✓，
而**没声明的它按规矩只报"未覆盖"，不报失败** ✓。

（这正是那套设计的意思：`缺失 0` 和"覆盖多少"是两件事 ✓。
 但**读的人要把两个数一起看** ✗ —— 我前面几轮就没看 ✓。）

### 二、`g_proj` 是什么 —— `kda.py:309`

    o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=...))
    o = rearrange(o, "b t h d -> b t (h d)")
    o = o_proj(o)

**`g_proj` 是 `o_norm` 的门** ✓ —— 而 `o_norm` 是 `FusedRMSNormGated` ✓，
**它需要一个门** ✓。所以 KDA 有**三个**投影组，不是一个：

    f_proj  -> delta 规则里的衰减门 g
    g_proj  -> o_norm 的输出门
    b_proj  -> beta

### 三、`__init__` 里的原样（`kda.py:168-189`）

    f_proj = Sequential(Linear(hidden, head_v_dim, bias=False),
                        Linear(head_v_dim, gate_dim,  bias=False))
    b_proj = Linear(hidden, num_v_heads, bias=False)
    A_log  = Parameter(zeros(num_v_heads))              # safe_gate 时
           = Parameter(log(uniform(1, 16, num_v_heads)))  # 否则
    dt_bias= Parameter(inv_dt)                          # gate_dim = num_v_heads * head_k_dim
    g_proj = Sequential(Linear(hidden, head_v_dim, bias=False),
                        Linear(head_v_dim, value_dim, bias=True))    # <- 第二层**带 bias**
    o_norm = FusedRMSNormGated(head_v_dim)

**`g_proj` 第二层带 bias** ✗ —— **没有任何 config 会告诉你这一条** ✓。
这就是"从源码读"和"从 config 猜"的差别 ✓。

### 四、所以两个模型是**两个变体**

    Ling   f_proj / g_proj 各**一层**（契约: (d_attn, d_model) 单行 ✓）
    GLM    f_a/f_b / g_a/g_b 各**两层**（和 fla 一致 ✓）

**它们不是同一个东西** ✗ —— 而两边**都对着官方产物验过** ✓
（Ling 9283/9283、GLM 37534/37534 ✓）。**不能用一个实现糊过去** ✓。

### 五、item 1 现在的位置

    语义   **全量拿到了** ✓（五处出处，见 `gpu/by1kda.py` 的 docstring）
    障碍   **不是 codegen** ✗ —— 是 **GLM 的 `.by1` 契约少了 5 个张量/层** ✗

    下一步（按顺序）：
      ① 补 GLM 的契约（5 个张量 x KDA 层数），再跑 by1verify 看未覆盖掉多少
      ② by1codegen 加 KDA 分支 —— **Ling 先做**（它的契约是完整的 ✓）
      ③ by1exec 加 op_kda（两个变体：门一层 / 门两层）
      ④ 判卷人：对 `fla` 的 naive_recurrent_kda（+ naive_kda_gate），
         并把"门写成一层"、"beta 不做 sigmoid"、"q 不除 sqrt(K)"
         都做成反例 —— 每一条都该红

**先补契约、再动 codegen** ✓ —— 因为契约不全的时候，
codegen 生成的模块**没法验** ✓，而"没法验的实现"正是这个项目一直拒绝的东西 ✓。


---

## 109. GLM 的 KDA 契约补上了 —— 而**它根本没被计数** ✗

### 一、补的那五行，形状是量出来的

    b_proj.weight   [64, 4096]   = (num_heads, d_model)
    f_a_proj.weight [128, 4096]  = (head_dim, d_model)
    f_b_proj.weight [8192, 128]  = (num_heads * head_dim, head_dim)
    g_a_proj.weight [128, 4096]
    g_b_proj.weight [8192, 128]

而它们**和 fla 的 `KDA.__init__` 逐项吻合** ✓：

    f_proj = Sequential(Linear(hidden, head_v_dim), Linear(head_v_dim, gate_dim))
    g_proj = Sequential(Linear(hidden, head_v_dim), Linear(head_v_dim, value_dim))

`head_v_dim = 128` ✓、`gate_dim = 64 x 128 = 8192` ✓、`value_dim = 8192` ✓ ——
**对上了** ✓。所以 GLM 的 `f_a/f_b`、`g_a/g_b` 就是那两个两层投影 ✓，
而 `g_proj` 是 `o_norm` 的门 ✓。

契约现在 **15 行** ✓，解析出来的形状全对 ✓。

### 二、而 `by1verify` 的数**一点没变**

    补之前   契约声明存在: 37534   名字+形状一致 37534
    补之后   契约声明存在: 37534   名字+形状一致 37534   ✗

我以为是缓存或没生效 ✓，于是逐层查：

    layer_out 有 45 层
    第一个 KDA 层：stack=model mech=KDA，**KDA 行 15 条** ✓
    各机制出现层数：{'KDA': 34, 'SparseMLA': 11}

**行在 `layer_out` 里** ✓ —— 而 `check_tensors` 就是遍历它的 ✓ ——
可那个数不动 ✗。

### 三、做了一次干净的判定

从契约里**删掉一行** ✓，再跑：**数还是 37534** ✗。

**那一行本该让计数变 34**（34 个 KDA 层）✓ —— 而它纹丝不动 ✓。
**所以 KDA 这 15 行对 `by1verify` 的计数没有任何贡献** ✓ —— 确认。

（顺带：我第一版"删 5 行"的脚本只删掉了 1 行 ✓ —— 匹配写得比行的实际内容细 ✗。
 如果没去数删了几行，我会把一个**无效的实验**当成有效的结论 ✓。
 **删除类实验要数删掉了几行** ✓。）

### 四、为什么，这一轮没查到

排除了的：
    `scope` 映射里有 `KDA = self_attn.` ✓
    `layer_out` 里 KDA 行是 15 条 ✓（34 层）
    没有"未映射跳过"的数冒出来 ✓

**没排除的**：`check_tensors` 里那几处提前 `continue` 的条件
（`quant` 那条、形状解析失败那条）✓ —— 需要把那个函数读透 ✓。

**不写猜测的结论** ✓ —— 记下现象和已经排除的，下一轮从 `check_tensors` 进去 ✓。

### 五、为什么这件事重要

GLM 的 KDA 有 **34 层 x 15 = 510 个张量** ✓ —— 而 `by1verify` 报
"37534 个全对" ✓ 时，**这 510 个一个都没参与** ✗。

**也就是说：在 GLM 上跑出 PASS，不代表 KDA 的契约被查过** ✗ ——
而我本来是要**照着这份契约去写 codegen** 的 ✓。

**先把这个搞明白，再动 codegen** ✓ —— 否则我是在照一份
"看起来验过、实际没验"的契约写实现 ✓ —— 而那正是这个项目最不能接受的事 ✓。


---

## 110. 那 510 个张量是**假绿** —— 形状写成 `(7, 9)` 也过

上一轮发现"KDA 的契约行对计数没贡献" ✓，没查到原因 ✓。
这一轮换了个问法 —— 不问计数，**直接问检查碰不碰得到它** ✓。

### 一、反例：把形状改错

    f_b_proj.weight : (num_heads * head_dim, head_dim)
    ->  f_b_proj.weight : (7, 9)          # 故意的

    $ by1verify GLM-5.3-Flash.by1 ...
    契约声明存在: 37534   名字+形状一致 37534   **形状不符 0**
    [PASS] 37534 个张量的名字与形状全部一致
    rc = 0

**PASS** ✗ —— 一行形状被改成 `(7, 9)`，检查说全对 ✓。

**GLM 的 KDA 有 34 层 x 15 = 510 个张量** ✓，而**它们一个都没被验** ✗ ——
报的却是 PASS ✓。

**这是假绿** ✗ —— 比红更坏 ✓：红会让人去查，绿不会 ✓。

### 二、第二个 bug：`index = global` 从来没生效

读 `check_tensors` 开头那段（第 300-305 行）：

    _glob = set()
    for _st in (info.get("stacks") or []):
        _as = getattr(_st, "assigns", None) or {}
        if (_as.get("index") or "").strip().lower() == "global":
            _glob.add(getattr(_st, "name", ""))

而 `by1check` 返回的是（第 896 行）：

    "stacks": [(s.name, s.alias, len(_['expansion'].get(s.name, [])))
               for s in _['stacks']],

**是元组，不是对象** ✗ —— `getattr(元组, "assigns", None)` 永远返回 `None` ✓
→ `_as` 永远是 `{}` ✓ → **`_glob` 永远是空集** ✗。

于是上面那段注释里写的两种编号方式（栈内序号 / 全局序号）里，
**第二种从来没有被走通过** ✗ —— 而 GLM 恰好是那一种 ✓
（第 297-298 行写着「GLM-5.3 **都在** `model.language_model.layers.{i}.` 里，
MTP 是第 45 层 —— 层号接着数」✓）。

**这个和假绿可能是同一件事的两面** ✓ —— 也可能不是 ✓。
**没验完，不写结论** ✓。

### 三、还没排除的

    ✓ layer_out 里 KDA 行是 510 条（34 层 x 15）
    ✓ 那 510 条的形状**全部能解析**（`parse_shape` 不是 None）
    ✓ `info["emit"]["torch.module"]["scope"]` 里有 `KDA`
    ✗ 没排除：`check_tensors` 收到的那张 `scope_map` 是不是同一张

**所以我需要的是一次**能打印出 `scope_map` 和逐 owner 计数的探针 ✓ ——
这一轮试了两次都没接对（`main` 自己解析 argv 的方式和我猜的不同 ✗）✓。
**下一轮从那里进** ✓ —— 或者干脆把 `unmapped` 这个数**印出来** ✓
（它现在算出来了却不显示 ✓ —— 而它恰好就是答案 ✓）。

### 四、这件事改变了什么

上一轮我说"先把这个搞明白，再动 codegen" ✓。现在它**更硬了**：

    GLM 的 KDA 契约：**510 个张量，验不到** ✗，而报告是 PASS ✓。

**所以不能照它写 codegen** ✗ —— 除非先让那 510 个真的被验 ✓。
而 `by1verify` 里那个 `unmapped` 计数器**已经算出来了** ✓ ——
它只是没被印出来 ✓ —— **一个算出来却不显示的诊断量，等于没有** ✓。

### 五、这一轮真正的产出

    ① 证实了假绿（有反例 ✓）
    ② 找到 `_glob` 恒空这个 bug（有代码位置 ✓）
    ③ 把"还差什么才能定位"缩到了**一个数**：`unmapped`

**三条都比"再读一遍代码"值钱** ✓。


---

## 111. 定位到了：**形状校验是好的，只有 KDA 这一个 owner 被跳过**

### 一、两个对照实验

同一个文件、同一处改动，只换**改哪一行**：

    input_layernorm.weight : (d_model,)  ->  (7, 9)      # owner = layer
    契约声明存在: 37534   名字+形状一致 **37489**   形状不符 **45**   rc=1  FAIL ✓

    f_b_proj.weight : (num_heads*head_dim, head_dim) -> (7, 9)   # owner = KDA
    契约声明存在: 37534   名字+形状一致 37534   形状不符 **0**    rc=0  PASS ✗

**改 `layer` 行会红 45 处**（45 层 ✓）—— **改 KDA 行毫无反应** ✗。

**所以形状校验本身是好的** ✓ —— **被跳过的是 `KDA` 这一个 owner** ✗。

### 二、顺带定出了那个数的构成

    改 layer 行后：37534 -> 37489  =  37534 - 45
    ⇒ **37534 数了 layer 行** ✓
    而改 KDA 行前后都是 37534
    ⇒ **37534 没数 KDA 行** ✗

再看直接调 `check_tensors` 的结果：**1369**（= 510 KDA + 360 layer +
336 MoE + 154 SparseMLA + 9 Dense ✓，**含 KDA** ✓）。

    37534 - 1369 = 36165   ≈ 专家展开那部分 ✓

**两个数的差是专家展开** ✓ —— 而**KDA 只在 1369 里、不在 37534 里** ✓ ——
说明真实调用里，`layer_out` 的那 34 个 KDA 层**没有走到计数** ✗。

### 三、已经排除的（这一轮排掉的）

    ✓ `scope_map` 里有 `KDA`（我直接调时打印过：Dense / KDA / MoE /
      SparseMLA / global / layer）
    ✓ `info["layer_out"]` 里 KDA 行是 510 条，形状全部能 `parse_shape`
    ✓ `unmapped` 是 **0**（第 418 行"未映射机制跳过"只在非零时印，
      而 GLM 的输出里没有这句）
    ✓ 不是 `quant`（GLM 的 `emit[torch.module]` 里**没有** quant 块，
      带上不带上是同一个数）
    ✓ `info` 是同一个（`by1verify.py:489` 就是 `bc.check(by1_path)`）

### 四、剩下的可能，以及下一轮怎么进

计数在 `check_tensors` 的第 370 行（`total += 1`）✓ —— 它前面只有三道闸：

    314   owner 不在 scope      -> unmapped，已排除（是 0）
    355   shape_txt == "--"     -> 抑制，GLM 是 0 处抑制，排除
    365   parse_shape 是 None   -> symbolic，GLM 是 0，排除

**三道都排除了，而数就是不到它头上** ✗ —— 所以**问题不在这三道** ✓，
而在**那个循环有没有遍历到那 34 层** ✗。

**下一轮的入口**：把 `check_tensors` **复制一份**、在循环里按 owner 计数 ✓ ——
直接调它现在是通的 ✓（这一轮就是这么拿到 1369 的 ✓），
所以加几行计数就能看见那 34 层到哪去了 ✓。

### 五、这件事的分量

    GLM 的 KDA：34 层 x 15 = **510 个张量**
    形状写错**不会被发现** ✗，而报告是 PASS ✓

这是**假绿**，而且是**可复现的假绿** ✓ —— 一条命令就能给别人看 ✓。

**而它恰好挡在 item 1 的路上** ✓：我本来要照着这份契约写 KDA 的 codegen ✓，
而这份契约**看起来验过、实际没验** ✓。


---

## 112. **没有假绿。是我错了** —— 真因是"重复的契约块静默覆盖"

上一节我写下"GLM 的 510 个张量是假绿，形状写成 `(7, 9)` 也过" ✗。
**那句是错的** ✓ —— 这一轮找到了真因 ✓。

### 一、怎么找出来的

两个探针上的坑，各踩了一次：

    ① `by1verify.main` 的 `argv` **已经是 `sys.argv[1:]`** ✗ ——
       我传了完整的 `sys.argv` ✓，于是整体错一位 ✓，
       它把 `.by1` 当成了 config ✓。**前两次探针因此全是无效的** ✓。
    ② `by1verify` 用的是 `load_checker()` **另外加载的一份 `by1check`** ✗ ——
       我给 `import by1check` 打的补丁**根本没生效** ✓
       （第一次的计数器整张都是空的 ✓，而我把那当成了"一次都没渲染" ✗ ——
        **一个空计数器和一个零计数，看起来是一样的** ✗）。

踩完这两条，探针就通了 ✓。

### 二、探针说的是什么

让探针跑那个"坏"的形状，它自己在 KDA 的行里数：

    [探针] 我数出来 KDA 有问题的行：**0 条**
    [探针] 真检查说：37534 / 37534 / 形状不符 0 / 缺失 0

**探针和真检查一致** ✓ —— 所以真检查**没漏** ✓ —— **是我改的那一行不在它看的范围里** ✗ ✓。

### 三、真因

    ── 文件里所有 f_b_proj.weight 出现在哪
       第 130 行  f_b_proj.weight : (num_heads * head_dim, head_dim)
                  **属于块 KDA（第 99 行）**
       第 167 行  f_b_proj.weight : (8192, 128)
                  **属于块 KDA（第 159 行）**

**同一个 `tensors` 里有**两个 `KDA { }` 块** ✗。

而解析器是这样收的（`by1tens.py:166`）：

    _['contracts'][_['base']] = _['lst']

**后来的直接覆盖先前的** ✗ —— **一声不响** ✓。

于是：
    我把 5 行加进了**第一个**块 ✗（第 99 行那个 ✓）
    `(7, 9)` 的反例也改在**第一个**块 ✗
    而 `layer_out` 用的是**第二个**块 ✓（第 159 行 ✓，值 `(8192, 128)` ✓）

**所以我改的是一个死块** ✓ —— 检查从头到尾都是对的 ✓，
**是我在对着一个不生效的地方做实验** ✗。

（第 159 行那个块是哪来的：上一轮我只看到第 99 行那个 ✓ ——
 而 `grep 'KDA {'` 会同时列出两个 ✓，**我没 grep，直接按行号找的** ✗。）

### 四、所以真正的 bug 是另一个 —— 而且更该修

    同一个 tensors 块里出现两个同名契约块 -> **后者静默覆盖前者** ✗

它和这个仓库在猎的东西是同一族 ✓：**看起来写了、实际被别的盖掉了，而没人说** ✗。
（和 `vision ViT { }` 那段是亲戚 ✓ —— 那段是"没人读" ✓，这段是"读了但被盖" ✓。）

**该加的检查**：`tensors` 里同一个名字的契约块出现两次 -> 报错 ✓。
反例现成 ✓：把 GLM 的这个文件留着不改就是反例 ✓
（它现在**合法地**带着两个块 ✓，而其中一个从来没生效过 ✓）。

### 五、我这一轮学到的两条

    ① **实验要先证明"我改的东西被读到了"** ✓
       —— 我改了、跑了、数没动 ✓，于是断定"检查漏了" ✗。
       而正确的下一步是**先问"我改的那处在不在被测范围里"** ✓。
       （探针本该第一次就回答这个 ✓ —— 而我被两个坑挡住了 ✓。）
    ② **一个空计数器和一个零计数，看起来是一样的** ✗
       —— 补丁没打上时，计数器整张为空 ✓，
       而我把"没输出"读成了"计数为零" ✗。
       **补丁类工具要先自证生效** ✓。

### 六、现在的状态

    GLM 的 KDA 契约：**有 15 行** ✓（第二个块里 ✓），
    形状和官方产物**逐项一致** ✓ —— 这一点从上一轮起就是真的 ✓。

    要做的：
      ① 删掉死块（第 99 行那个）✓ —— 或者把它和第二个合并 ✓
      ② 加检查：**同名契约块出现两次要报** ✓（反例就是现在这个文件 ✓）


---

## 113. 死块删了，而"同名契约块"这条检查**加在了解析器里**

### 一、那个重复块**在我动手之前就在**

    $ grep 'KDA {' models/GLM-5.3-Flash.by1
    第 99 行：KDA {    10 行、不完整   <- **死的**
    第 159 行：KDA {   15 行、完整     <- **活的**
                       注释还写着"和 Ling 的 KDA 不一样：
                       那边 f 是一个矩阵，这边是 f_a/f_b 两级"

**所以契约本来就是完整的** ✓ —— 我上一轮"补"的是那个死块 ✗。
（而上一轮我是**按行号找**的 ✓，没 grep ✗ —— `grep` 一次就会看见两个 ✓。）

删掉死块 ✓，在原处留一段说明它是什么、后果是什么 ✓。

### 二、检查加在哪：**赋值那一点**，不是正则扫文本

该报的不是"文件里有两个 `KDA {`" ✗ —— 那是文本特征 ✓ ——
而是"**同名的契约块被覆盖了**" ✓ —— 那是解析事件 ✓。

所以加在 `by1tens.py` 收契约的那一行：

    if _['base'] in _['contracts']:
        _['rep'].add(E, _['ch'].line, "tensors",
                f"契约块 '{_['base']}' 出现了两次 —— **后一个会直接覆盖"
                f"前一个**，而前一个从此变成死的（往它里面加东西等于没加）✗。"
                f"删掉一个，或者把它们合并。")
    _['contracts'][_['base']] = _['lst']

**「谁解析谁报」** ✓ —— 和上一轮那条 `TOP_KEYWORDS` 同一个位置哲学 ✓。
而且这样**零误报**：只有真的发生覆盖才报 ✓，
不用去猜"4 空格缩进的 `{` 算不算契约块" ✓。

### 三、判据：两个方向 + 一个反例

    ① 反例：在 Ling 里临时插一个重复的 KDA 块
       -> **红了 1 处** ✓（第 112 行 ✓，消息里写清了后果 ✓）
    ② 还原后，27 份 `.by1` **一份都不红** ✓
    ③ 全仓库扫了一遍：**没有别的文件有同名契约块** ✓

### 四、这一条和前面几条是同一族

    `vision ViT { }`        顶层块的关键字不认识 -> **没人读** ✗
    同名契约块出现两次        -> **读了，但被后一个盖掉** ✗
    `scope` 不在映射里        -> **静默跳过** ✗（by1verify 那边）

**三个都是"看起来成立、实际没生效，而没人说"** ✓ ——
而它们各自的表现都是"数没动" ✓ ——
**"数没动"这个现象，看不出是哪一种** ✗。

所以每一条都得**单独**有一个反例 ✓ —— 这是这一轮最实在的收获 ✓。


---

## 114. KDA 接进 codegen —— 而**它不是新算法族，是 GDN 的变体**

### 一、先看的不是"怎么写"，是"有没有写过"

`by1codegen.py` 里已经有一个 `GatedDeltaNet` ✓（GDN ✓）。它的 forward：

    beta = bb.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(aa.float() + self.dt_bias)
    out, _ = gated_delta_rule(q, kk, v, g, beta, self.l2_eps)
    out = self.norm(out.reshape(...), z)

拿它对着 fla 的 KDA 一行一行看 ✓ —— **门公式一样** ✓、
`beta = sigmoid` 一样 ✓、**delta 规则一样** ✓、gated norm 一样 ✓。

再打开 `gated_delta_rule`（第 1503 行）：

    q = l2norm(q, -1, l2_eps)          <- **KDA 的 use_qk_l2norm_in_kernel=True**
    k = l2norm(k, -1, l2_eps)
    q = q * (dk ** -0.5)               <- **KDA 的 1/sqrt(K)**
    for i in range(s):
        st = st * g[:, :, i].exp()...          <- S = S * exp(g_i)
        kv_mem = (st * k).sum(-2)              <- (k_i ⊗ S).sum(-2)
        delta = (v - kv_mem) * beta            <- (v_i - ...) * beta_i
        st = st + k ⊗ delta                    <- S += (beta k) ⊗ (...)
        out = (st * q).sum(-2)                 <- q_i @ S

**逐行就是 KDA 的递推** ✓✓ —— **连 L2 归一化和 `1/sqrt(dk)` 都在** ✓。

**所以 `by1codegen` 早就有了 KDA 的数学** ✓ —— 它叫 `gated_delta_rule` ✓。
**这一轮一行数学都没写** ✓ —— 这正是"同一个意思不要两处实现" ✓。

### 二、接了什么

    SUPPORTED_KINDS  += "KDA"
    ATTRS["KDA"]     = {k_heads, v_heads, k_dim, v_dim, num_heads, head_dim,
                        conv_kernel, conv_bias, gate_lowrank, gate_rank,
                        gate_lower, gate_act, l2_eps, norm_eps, act, structural}
    MIXER_KINDS      += "KDA"
    build 分支       两套属性名都收（Ling 用 k_heads/k_dim，GLM 用 num_heads/head_dim
                     —— **名字不同、指的是同一个东西**，而两边都对着官方产物验过）

### 三、判据：报错变了

    之前：机制 'KDA' 的类型是 'KDA'，codegen 还不支持
    现在：**这句没了** ✓
          Ling  -> 机制 'MLA' 的属性 'qk_head' codegen 还不支持
          GLM   -> 机制 'SparseMLA' 的属性 'index_heads' / 'index_dim' /
                   'index_topk' codegen 还不支持

**KDA 不再是拦路的了** ✓ —— 拦路的是 **MLA / SparseMLA** ✗。

（中途出现过一个 "第 N 层没有可用的 token 混合器" ✗ ——
 和当初 SSM 那次**一模一样** ✓：`MIXER_KINDS` 里没加它 ✓，
 于是每一层都说不出来为什么 ✓。加一行就消了 ✓。）

### 四、这一轮给目标带来的**新信息**

目标的第 1 条写的是「接进 `by1codegen` 的机制表，**让这两个模型能跑前向**」✓。
而现在量出来：**KDA 不是唯一的拦路石** ✗ ——

    Ling  = KDA + MLA
    GLM   = KDA + SparseMLA（MLA 加一个稀疏索引器）

**两个模型都要先过 MLA 那一关** ✗ —— 那是**另一个机制** ✓，
而且 `by1codegen` 里 MLA **已经有了一支** ✓（`MLAttention` ✓）——
只是属性表收不下 `qk_head` / `index_*` 这些 ✓。

**所以"让这两个模型跑前向"这条路，下一段是 MLA，不是 KDA** ✓。
这一点在动手之前不知道 ✓ —— 现在知道了 ✓。

### 五、还没做的

    ✗ 运行时类 `KDA`（把上面的 attrs 变成 nn.Module）
    ✗ `by1exec.op_kda`（NumPy 那一侧）
    ✗ `by1c.py` 的 `kda`（C 那一侧）
    ✗ `by1ir.KIND_ATTRS["KDA"]`
    ✗ 判卷人（对 fla 的 `naive_recurrent_kda` + `naive_kda_gate`）

**这一轮只做到"codegen 认得它"** ✓ —— 而这是可验证的一步 ✓（报错变了 ✓）。


---

## 115. `by1gate` 那条期望过期了 —— 改方向，不删

### 一、它说什么

    取值门 坏了：Ling-3.0-tiny.by1(理由)
    拒绝理由里没提到 KDA

而 `by1gate.py` 里那一行是：

    ('Ling-3.0-tiny.by1', False, 'KDA', 'KDA 整族没实现'),

**门说得对** ✓ —— Ling 现在被拒的理由是
「机制 'MLA' 的属性 'qk_head' codegen 还不支持」✓ ——
**因为 KDA 那一半已经实现了** ✓。

### 二、怎么办：改方向

    ('Ling-3.0-tiny.by1', False, 'MLA',
     'MLA 的属性还没收全 —— KDA 那一半已经实现了'),

**照样该被拒** ✓ —— 只是理由换了 ✓。
和当初 Nemotron 从"该被拒"搬到"该通过"是**同一个动作** ✓：
**期望过期了就改方向，不删** ✓。

**删掉它才是错的** ✗ —— 那等于说"Ling 现在不该被拒" ✓，
而它**明明还被拒着** ✓（卡在 MLA ✓）。

### 三、这一轮顺带确认的

    取值门：[PASS] 工作正常 ✓（三条真实模型 + 反例都各自对上了）
    档 4：64 项，0 项失败 ✓

### 四、这一轮的量：`by1gate` 这个检查**是有用的**

它是**唯一**一个在 KDA 落地之后**立刻变红**的检查 ✗ ——
而它红得**有道理** ✓（期望过期 ✓），不是误报 ✓。

**一个改动落地时，最该看的不是"绿的那些还绿吗"** ✓ ——
**而是"有没有哪个检查因此变得不对了"** ✓ —— 而那个检查正好会**变红** ✓。


---

## 116. `by1codegen.py` 的真实架构：**生成器 + 一个字符串模板**

### 一、这一轮想做的事

写 `KDA` 的运行时类 ✓ —— 把上一轮那些 attrs 变成一个 `nn.Module` ✓。
照着 `GatedDeltaNet` 写完了 ✓、注册进 `BUILDERS` ✓、`py_compile` 过了 ✓ ——
**然后测不动它** ✗。

    >>> import by1codegen as G
    >>> G.KDA
    AttributeError: module 'by1codegen' has no attribute 'KDA'

而 `class KDA` 明明在文件的第 1669 行、**缩进 0** ✓。
更奇怪的是 `G.BUILDERS` 也没有 ✗、`G.torch` 也没有 ✗ ——
可 `G.GatedDeltaNet`、`G.Mamba2` 这些**一直在用** ✓。

### 二、查出来的东西

    >>> sorted(G.__dict__)
    ['ATTRS', 'CodegenError', 'ENUMS', 'HERE', 'MIXER_KINDS', '**RUNTIME**',
     'SUPPORTED_KINDS', ..., 'build', 'compile_ir', 'compile_spec',
     '**load_checker**', 'main', 'render', 'render_ir', ...]
    >>> type(G.RUNTIME)
    <class 'str'>                       <- **是字符串** ✗

**第 990 行往后那些类，全都写在 `RUNTIME` 这个字符串里** ✓✓ ——
它们不是模块的属性 ✓，是**一段要被生成的源码** ✓。

所以：

    `py_compile` 过 ≠ 那些类能跑 ✓ —— **编译的是一段字符串** ✗
    `import by1codegen` 拿到的只有**生成器那一半** ✓（不依赖 torch ✓）
    真正的运行时是 **`build()` 渲染出来、再 exec 出来的** ✓

**这就是为什么这个仓库里到处都有 `load_checker()` 那种"另外加载一份"的写法** ✓ ——
（`by1verify` / `by1oracle` 各有一份 ✓ —— 而我前面几轮已经被它坑过一次 ✓：
 给 `import by1check` 打的补丁对 `bc.render_name` 无效 ✗。）

### 三、所以我上一轮和这一轮的"编译 OK"都是假的

    by1codegen.py  编译 OK     -> 编译的是生成器 + 一个字符串 ✓ **没验到模板**
    class KDA 在文件里 ✓       -> 在**字符串里** ✓ 不在模块里 ✗

**要验它，只有一条路：真的生成一次，把生成的源码 exec 出来，再看有没有 `KDA`** ✓。

### 四、这一轮实际做成的

    ① SUPPORTED_KINDS += "KDA"       （上一轮）
    ② ATTRS["KDA"]                   （上一轮）
    ③ MIXER_KINDS += "KDA"           （上一轮）
    ④ build 分支：两套属性名都收       （上一轮）
    ⑤ **运行时类 KDA 写进模板** ✓     （这一轮）
       —— 递推**一行没写** ✓，调 `gated_delta_rule` ✓
       —— 四处差别按量到的写：分开的 q/k/v 投影 ✓、
          `dt_bias` 按维（`nv * dk`）✓、`g_proj` 当输出门 ✓、
          门可低秩两层（`f_a/f_b`，第二层**带 bias** ✓）
       —— `gate_lower` 那一支也写了 ✓（`g = lower * sigmoid(exp(A_log) * g)` ✓）
    ⑥ `BUILDERS["KDA"] = KDA` ✓

### 五、判据仍然是"报错变了"（上一轮那条），而运行时还没验

    Ling  -> 机制 'MLA' 的属性 'qk_head' codegen 还不支持
    GLM   -> 机制 'SparseMLA' 的属性 'index_heads' / 'index_dim' / 'index_topk'

**KDA 不再是拦路的** ✓ —— 而**新写的那个类跑没跑过，这一轮没验** ✗。

### 六、下一轮的入口（具体到一行）

    R = by1codegen.build(info)        # 渲染出运行时的源码
    # 或者看 RUNTIME 这个模板是怎么被渲染的：
    #   by1codegen.render_ir / render
    # 然后 exec 出来，确认 `BUILDERS["KDA"]` 和 `KDA` 都在

**而且要先做一件更小的事**：拿一个**最小的 KDA 模型**（像
`models/ssm-shaped.by1` 那样 ✓）走一遍生成 ✓ ——
`by1diff` 对三个后端 ✓、判卷人对 fla 的 `naive_recurrent_kda` ✓。

**这一轮的价值不在"写完了"** ✓ —— 在**搞清楚了"写完"之后该去哪儿验** ✓。


---

## 117. `op_kda` 写完了，而 `P` 是空的 —— **漏了 SSM 那次的后半句**

### 一、这一轮做了三件

    ① `by1exec.op_kda`（NumPy 侧）✓ —— 对着 `op_linear`（GDN）那支写，
       递推部分**逐行一样** ✓（因为 NumPy 这一侧本来就是"另一个后端"，
       不是"另一份实现" ✓）。四处差别按量到的写：
         · q/k/v 分开的投影，各自一条深度卷积
         · `dt_bias` 按维（`[b,s,nv,dk]`），不是按头 ✗
         · `g_proj` 当输出门
         · 门一层（`f_proj`/`g_proj`）或两层（`f_a`/`f_b`/`g_a`/`g_b`）
       注册进 `OPS` ✓。
    ② `models/kda-shaped.by1` ✓ —— 和 `ssm-shaped.by1` 同一个用途：
       真实模型建不出来（GLM 76108 个张量、Ling 9283 个），
       而"建不出来"是"验不了"，不是"算错了" ✓。
       尺寸取小值（k/v_heads=4、k/v_dim=16、conv_kernel=4、d_model=128）✓。
    ③ 运行时类 `KDA` 的属性名对齐契约的逻辑名 ✓
       （`q_conv` -> `q_conv1d` ✓）。

### 二、它现在跑到哪儿了

    by1check   IR 出来了 ✓（报告 1 条 W，见下）
    by1exec --compare
      -> KeyError: 'q_conv1d'

而探针说：

    [探针] P 的键：**[]**     <- **空的** ✗

**不是键名不对，是根本没有键** ✗。

### 三、缺的那一环

`by1exec` 里除了 `OPS`（kind -> 函数）✓，还有**一张形状表** ✓ ——
"每种 kind 声明哪些张量、各是什么形状" ✓ —— 而
**`op_kda` 只注册了前一半** ✗。

**这一条在 SSM 那一轮就写过** ✓，原话是：

    by1exec.py — 新增 op_ssm(P, a, ins, d) + OPS["SSM"]，
                 **外加一个 elif k == "SSM" 的形状分支（8 个张量）**

**我把后半句漏了** ✗ —— 而漏了之后的表现是 `P` 空 ✓、
`KeyError` ✓ —— **看起来像"键名写错了"** ✗，实际是"分支没写" ✓。

（和这个仓库里反复出现的那一族一样 ✓：
 **"数没动"/"看起来像 A" 的现象，实际原因是 B** ✗ ——
 所以每一条都要单独有一个反例 ✓。）

### 四、下一轮要做的（具体）

    ① `by1exec` 里加 KDA 的形状分支，13 个张量：
       q/k/v_proj · q/k/v_conv1d · A_log(nv) · dt_bias(nv*dk) ·
       b_proj(nv, d_model) · f_proj(nv*dk, d_model) · g_proj(nv*dv, d_model) ·
       o_norm(v_dim) · o_proj(d_model, nv*dv)
       （两层门的变体是 f_a/f_b/g_a/g_b，另算）
    ② 再跑 `--compare`，判据是 **NumPy 与 PyTorch 逐位一致** ✓
    ③ 然后才是判卷人：对 fla 的 `naive_recurrent_kda` + `naive_kda_gate`

### 五、顺带：`by1check` 报了一条新的 W

    ('W', 36, 'shape', 'mech KDA_ (KDA) 未声明 heads/head_dim ——
                       依赖它的 KV cache 与状态尺寸都无法推导')

**它是 W 不是 E** ✓ —— 所以门不红 ✓ —— 但它说得对 ✓：
KDA 的"头"叫 `k_heads`/`v_heads` ✓，而那条检查只认 `heads`/`head_dim` ✓。
**要么在 `ATTRS["KDA"]` 里承认这两个别名，要么给那条检查加一个别名表** ✓ ——
记在这里，别让它变成一个"一直在印而没人看"的东西 ✓。


---

## 118. **KDA 能算了** —— 而真差别是"门的秩"

### 一、上一轮那两件，各自的下场

    ① NumPy 的**形状分支**（我漏掉的那一环）
       —— 补上 13 个张量 ✓。补的时候看了一眼 SSM 那支，发现一件解气的事：
       **`P` 的键就是这张表自己定的** ✓ —— SSM 的契约写 `conv1d.weight` ✓
       而表里叫 `conv` ✓，两边不一样也一直是对的 ✓。
       **只要表和 `op_*` 一致就行** ✓。
       （我上一轮照着契约的名字去读 `P`，于是拿到空的 —— 那不是键名写错，
        是**这张表根本没写** ✓。而"空字典"看起来像"键名不对" ✓。）

    ② PyTorch 那一侧：`'KDA' object has no attribute 'f_a'`
       —— **是我自己类里的 bug** ✓：`_gate_out` 分了支 ✓ 而 `_gate_in`
       直接 `self.f_a(x)` ✓，于是 `gate_lowrank=false` 的模型一跑就炸 ✓。
       **`by1irentry` 把三个后端都从 IR 跑一遍 —— 就是它抓住的** ✓。

    （另外 IR 那层还抓了两个：`kind = <KDA> 不在闭集` ✓ 和
      `gate_lower 应当是数，实际 None` ✓ —— 都是我无条件写进去的 ✓。
      补完这两处之后，IR 才合法 ✓。）

### 二、而真正的差别，是在维度上炸出来的

修完上面那些，PyTorch 报：

    RuntimeError: The size of tensor a (4) must match the size of tensor b (16)
                  at non-singleton dimension 2

`4 = v_heads` ✓ · `16 = k_dim` ✓。而 `gated_delta_rule` 里那一行是：

    st = st * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)

**那是"每个头一个标量"** ✗ —— `g[:, :, i]` 是 `[b, h]` ✓。
而 KDA 的 `g` 是 `[b, h, s, dk]` ✓ —— **每一维一个** ✓：

    # fla/ops/kda/naive.py:61
    S = S * g_i[..., None].exp()      # g_i 是 [B, HV, K] -> 逐维 ✓

**所以"KDA 和 GDN 的递推是同一套"这句话，只对了一半** ✓ ——
更新式一样 ✓、读出一样 ✓、`beta` 一样 ✓，而**衰减门的秩不一样** ✗。

**这正是 SSM 那段注释里写过的那类东西** ✓：
「两处最容易写反（写反了会**炸在维度上**，不会静默算错）」✓ ——
而这一次它**真的炸在维度上了** ✓，所以它是"看得见"的那种错 ✓。

### 三、一处实现容纳两种，而不是写两份

    _gi = g[:, :, i]
    _gd = _gi.unsqueeze(-1) if _gi.dim() == 2 else _gi
    st = st * _gd.exp().unsqueeze(-1)

三行 ✓ —— **`gated_delta_rule` 一个函数同时是 GDN 和 KDA 的递推** ✓。
**没有第二份实现** ✓。

### 四、判据

    $ python by1exec.py kda-shaped.by1 --compare
    后端必须实现的算子：{'Norm': 8, 'KDA': 4, 'Add': 8, 'FFN': 4}
    [OK] 全部已实现
    跑通：输出 (1, 48, 512)，幅度 5.2786e-01
    PyTorch 5.2786e-01   NumPy 5.2786e-01
    最大绝对差 2.257e-07   相对 4.275e-07   **[PASS] 两个后端一致** ✓

    顺带：ssm-shaped 仍然 1.739e-07 ✓（**一字未变** ✓）——
    那次推广没有把 GDN 弄坏 ✓。

**而且 IR 入口也过了** ✓：

    kda-shaped.by1   [不支持] C 后端还没有 KDA

—— `by1irentry` 说这是**覆盖率缺口，不是失败** ✓，
所以三个后端里 **PyTorch 和 NumPy 都从 IR 跑通了** ✓，缺的是 C ✓。

### 五、顺手修了一个生成物

改了 `by1ir.py`（加 `KIND_ATTRS["KDA"]`）之后，`by1docs` 报：

    [FAIL] 文档路径 0 处路径 + 0 个脚本 + 0 个数字 + **1 个生成物** 有问题

**`ir-spec.md` 过期了** ✓ —— 它声称"从 `by1ir.spec_markdown()` 生成" ✓。
重新生成 ✓（8039 字节 ✓，里面确实有 KDA ✓）。
**这个坑踩过一次了** ✓：跑那条命令 ≠ 那条命令改了那个文件 ✓。

### 六、现在的位置

    机制 KDA：**PyTorch ✓ · NumPy ✓ · C ✗** · IR ✓ · codegen ✓
    判卷人：**还没有** ✗ —— 而这是目标明写的要求 ✓

    下一步就一件：写 `src/modelcheck/by1kda.py` ——
    对着 fla 的 `naive_recurrent_kda` + `naive_kda_gate`（都是纯 PyTorch、
     CPU 上跑得动 ✓），并证明它在反例上会红：
      · 门写成"每头一个标量"（GDN 那一版）✗
      · `dt_bias` 按头而不是按维 ✗
      · `gate_lower` 那一支不走 ✗

## 119. KDA 的**判卷人** —— 以及"它碰不到哪儿"

### 一、参照物其实早就装好了

上一轮结尾写着"判卷人还没有" ✗ —— 而它的参照物**一直在机器上** ✓：

    fla/ops/kda/naive.py
      naive_recurrent_kda(q, k, v, g, beta)      ← 递推，就那五行
      naive_kda_gate(...)                        ← g = -exp(A_log) * softplus(f + dt_bias)

**纯 PyTorch** ✓ —— 一行 triton 都没有 ✓。所以"KDA 要 CUDA 才验得了"
那句话是错的 ✗ —— 它已经在 `gpu/by1kda.py` 开头被划掉了 ✓，
这一轮把它变成了**会跑的判据** ✓：

    pip install --no-deps fla-core     # 0.5.2，不拖 triton 进来

**加载方式是按文件路径** ✓，不是 `import fla.ops.kda` ✓ ——
后者要过 `fla/ops/__init__.py` ✓，而它 `import triton` ✗。
我们要的只是那个纯 PyTorch 的参考 ✓，它和 triton 一点关系都没有 ✓。

### 二、`except Exception: pass` 被 lint 抓住了 —— 而且抓得对

第一版的加载函数是这么写的 ✗：

    try:
        _sps = site.getsitepackages()
    except Exception:
        pass                    # ← 这里

`by1lint` 第一类（`except` 之后只有 `pass`）报了它 ✓。理由写在被改成的代码里 ✓：

    静默吞掉之后，**环境坏掉和"没装 fla"长得一模一样** ✗ ——
    都变成"跳过" ✓，而**跳过是"没验"，不是"验过"** ✓。

改成出声之后，两条路就分开了 ✓：

    真的没装         → [跳过] 退出码 30（by1skip 那一套）✓
    装了但取不到路径 → 打一行出来，且仍按"跳过"处理 ✓（不退化成 PASS）

### 三、判卷人自己把前端再写一遍

被测物（`by1codegen` 生成出来的 `KDA`）的 `forward` 是：

    q/k/v 投影 → 各自的短卷积 → silu → 重排 → 门 → sigmoid → 递推 → o_norm → o_proj

**前端那一半判卷人自己再写一遍** ✓（卷积用最笨的滑窗循环 ✓，重排、
门、sigmoid 都自己来 ✓）—— 那是判卷人的活 ✓，不是被测物的活 ✓。

参数是被测物的 ✓（权重必须同一份 ✓），**组合方式是判卷人自己的** ✓。

### 四、两处踩坑，都不是"差不多"

**① 直接比 `m(x)` 炸在形状上** ✗

    naive_recurrent_kda 回来的是**递推的输出** [b, s, HV, V] ✓
    而 m(x) 是整个 forward（过了 o_norm + o_proj）✗

于是改成**在递推那一步比** ✓：模块调的是**它自己命名空间里**的
`gated_delta_rule` ✓，所以换掉 `R` 里那一个就够 ✓：

    def _spy(qq, kk, vv, gg, bb, l2_eps=1e-6, state=None):
        o, st = _orig(qq, kk, vv, gg, bb, l2_eps, state)
        cap['o'] = o
        return o, st

**② `dt_bias 按头而不是按维` 这条反例不红** ✗ —— 而它明明是真反例 ✓

原因：`A_log` 和 `dt_bias` 在 `__init__` 里是 `zeros` ✓，
于是"按头"和"按维"**算出来一模一样** ✓ —— 实测报 `1.490e-08 没红` ✗。

修法两条，都是"让测试真的碰到东西" ✓：

    随机化全部参数      `p.copy_(torch.randn_like(p) * 0.1)` ✓
    红线自适应          `thr = max(1e-6, 正例误差 × 100)` ✓

第二条例子里有一处值得记下来 ✓：**写死 `1e-3` 会把真反例判成"没红"** ✗ ——
因为"`dt_bias` 按头"只动到 `dk` 维里的 1 维 ✓，效应**本来就小** ✓
（实测 3.114e-04 ✓，而正例是 4.657e-09 ✓）。
自适应之后，"红"的定义变成**明显不是数值噪声** ✓ —— 这才是要判的东西 ✓。

### 五、判据

    $ python src/modelcheck/by1kda.py
      参照物：...\site-packages\fla\ops\kda\naive.py
      生成出来的运行时里有 KDA：True

      ── 正例（两种尺寸）
         小      绝对 4.657e-09  相对 4.657e-09  ✓
         中      绝对 3.725e-09  相对 3.725e-09  ✓

      ── 反例（每一条都必须红）
         红线：相对 1.000e-06（= max(1e-6, 正例误差 x 100)）
         门退化成每头一个标量               相对 2.262e-02  红了 ✓
         dt_bias 按头而不是按维          相对 3.114e-04  红了 ✓
         q/k 不做 L2 归一化            相对 2.189e-02  红了 ✓

      [PASS]（反例也验过）

**三条反例分别打在三个不同的东西上** ✓：门的**秩** ✓、`dt_bias` 的**形状** ✓、
归一化的**有无** ✓ —— 而它们都不是"数值上差一点" ✗，是**语义差一档** ✓。

### 六、而先量一下它**碰不到**哪儿

判卷人写完、绿了 ✓ —— 这时候最容易犯的错是**把它当成全覆盖** ✗。
先量清楚 ✓，两处空白 ✓，都属于"写错了不报错、只算错"那一类 ✗：

    ⓐ 输出门 g_proj / o_norm / o_proj —— **整条尾巴一次都没比过** ✗
       比的是递推那一步的输出 ✓，尾巴那三段跑是跑了 ✓，结果丢掉了 ✗
    ⓑ `front()` 里的门输入调的是 `m._gate_in(x)` ✗ ——
       也就是**被测物自己的方法** ✓，于是它内部写错了 ✗，
       两边**错得一模一样** ✓，判卷人看不出来 ✗
       （`_gate_in` 忘了 `lowrank` 分支，正是**上一轮真发生过的**那个 bug ✓）

`by1exec --compare` 那条路也不管这两处 ✓ —— 它证的是
"两个后端一致" ✓，而两个后端**共享同一个 IR** ✓，
同一个意思写错两遍的概率很低 ✓，但**不等于和 fla 一致** ✗。

这两条**写进了判卷人的 docstring** ✓ —— 一份判卷人不写清边界，
读的人就会默认它全包了 ✓。

### 七、挂进"判卷人"这一组

`src/checks/by1files.py` 里那一组加了 `by1kda` ✓。而**顺序有讲究** ✓：

    by1files 数的是 **git 已跟踪的文件** ✓
    所以：git add -A → by1files --write → 提交 ✓

次序反了就会写出一个"漏掉新文件"的 FILES.md ✗ —— 而它看起来是**写成功**的 ✓。

### 八、闸门

    快速检查   编译 71 · import 66 · 解析 28 · 判卷人 5 · 行尾 116   [PASS]  11.3 秒
    档 4       64 项，0 项失败，5 项跳过，另有 1 项已知缺口
               [全过（5 项没验）]                                   75.5 秒

### 九、现在的位置

    机制 KDA：PyTorch ✓ · NumPy ✓ · C ✗ · IR ✓ · codegen ✓ · **判卷人 ✓**
    判卷人的**覆盖**：前端 + 递推 ✓ —— 输出尾巴 ✗、门的**组合方式** ✗

下一步就一件：把第六节那两处空白补上 ✓ ——
ⓐ 让判卷人**自己组合**门输入（不调 `m._gate_in`）✓；
ⓑ 把递推的输出接上尾巴那三段 ✓，和 `m(x)` 整个比一次 ✓，
   并给这一段补一条反例 ✓（`rms_norm_ref` 那个 `norm_before_gate` 开关
   就是现成的、真会写错的一档 ✓）。

### 十、顺手对上的两处文档 —— 以及一处**没对上**的

判卷人落地之后，`1.md` 和 `README.md` 里有两句话**当场变成了假的** ✓：

    旧：KDA / SSM / 稀疏索引器 / mHC **只有契约**，算不了        ✗
    新：6 种机制算得了（Attention · FFN · KDA · MLA · MoE · SSM），
        而**深度不齐**：KDA 只有 PyTorch 和 NumPy，**C 那个没有**；
        稀疏索引器 / mHC **连种类都还没有** ✓

**"6 种"这个数字原来是个裸数字** ✗ —— 而现在它后面跟着**枚举** ✓，
也就是 `by1exec.OPS` 去掉结构件、去掉两个逃生舱 ✓。
**裸数字过期了没人知道，枚举过期了一眼就看得见** ✓。

而**没对上的那一处是 `1.md` 里那张 14 行审计表** ✗。
里面至少有一行**已经明显过期** ✓：

    | Gemma 4 31B | 35/35 | 530 | **303（36%）** | ✗ |

—— 而视觉塔补完之后，契约里是 1000 多个张量、未覆盖只剩个位数 ✓
（§117 记着）。**这一行现在读起来是错的** ✗。

**不在这轮改** ✓，理由：那张表的每一格都要跑一遍 `by1verify` 才拿得到 ✓，
而**改一行、留十三行**比整张旧表更糟 ✗ —— 它会让"剩下的还是对的"这个
印象凭空多出来 ✓。**下一轮整张重测** ✓。

（这也正是 `1.md` 开头自己写的那条纪律 ✓：
**"一个写进行文里的数字，寿命比那句行文短"** ✓。）

## 120. 把上一轮记下的两处空白补上 —— 顺手量出一个**"声明了、一跑就炸"**

### 一、先把上一轮那两处空白摆出来

上一轮结尾写着判卷人**碰不到**两处 ✓：

    ⓐ 输出门 g_proj / o_norm / o_proj —— 整条尾巴一次都没比过
    ⓑ `front()` 里的门输入调的是 `m._gate_in(x)` —— 被测物自己的方法

### 二、ⓑ 是最好补的：自己组合

    if m.lowrank:
        h = m.f_a(x)
        if act == 'silu':
            h = F.silu(h)
        return m.f_b(h)
    return m.f_proj(x)

于是 `_gate_in` 内部写错这一类**会红** ✓，而不是两边一起错 ✓。

### 三、ⓐ 卡在"参照物从哪来"

尾巴要判的是**门控 RMSNorm** ✓，而它的参照物是
`fla/modules/layernorm_gated.py` 里的 `rms_norm_ref` ✓ ——
纯 PyTorch，正是我要的 ✓。**而那个文件不能 import** ✗：

    ModuleNotFoundError: No module named 'triton'

它开头就 `import triton` ✓，而我要的那个函数**和 Triton 一行关系都没有** ✓
（`torch` / `F` / `einops`，三个都是现成的 ✓）。

两条路：**照着重写一遍** ✗，或者**按函数名把它切出来** ✓。

    切：`re.search(r'(?ms)^def rms_norm_ref\(.*?(?=^@|^def |\Z)', src)`
    喂：`{'torch': torch, 'F': F, 'rearrange': rearrange}`
    得到：23 行，实测跑得动 ✓

**重写一遍是错的** ✗ —— 那就变成这个仓库里的**第三个**门控 RMSNorm ✓
（by1 运行时一个 ✓、fla 一个 ✓、判卷人再来一个 ✗），
而判卷人的全部价值就在"它说的话不是我说的" ✓。

### 四、于是判卷人从"比一个数"变成"比三段"

    ① 前端    by1 **递进** `gated_delta_rule` 的那五个张量，
              和判卷人自己算的 q/k/v/g/beta 逐个比
    ② 递推    `gated_delta_rule` **吐出来**的，和 `naive_recurrent_kda` 比
    ③ 尾巴    整个 `m(x)`，和「参照递推 → 门控 RMSNorm → `o_proj`」比

接那一跳的钩子原来只记**输出** ✓，现在**输入也记** ✓ ——
因为 L2 归一化在 `gated_delta_rule` **里面** ✓，
所以比"喂进去的"必须拿**归一化前**的 q/k ✓（判卷人于是算两份 ✓）。
**每段单独比** ✓ —— 混成一个数就只知道"错了" ✓，不知道"哪儿错了" ✗。

### 五、然后量出一个"声明了、一跑就炸"

补完前后端，顺手试了 `k_heads ≠ v_heads` ✓ —— 因为
`ATTRS["KDA"]` 和契约里它们是**两个独立属性** ✓：

    k_heads = 2, v_heads = 4
    RuntimeError: The size of tensor a (2) must match the size of tensor b (4)

**fla 那边是有这一支的** ✓（`naive.py:52-53`）：

    q = q.repeat_interleave(G, dim=2)
    k = k.repeat_interleave(G, dim=2)      # G = HV // H

而 by1 两个后端**都没有那一句** ✗。所以这不是"少支持一个特性" ✗ ——
是**"声明了、但一跑就炸"** ✓，也就是最难堪的那一类 ✓：
契约通过了 ✓、IR 通过了 ✓、`by1check` 也不报 ✓，**到跑的时候才炸** ✗。

补法是**一个后端一句** ✓，都照着 fla 那一行：

    # torch
    nv = v.shape[1]
    if nv != h:
        if nv % h: raise ValueError('v_heads 必须是 k_heads 的整数倍')
        q = q.repeat_interleave(nv // h, 1); k = k.repeat_interleave(nv // h, 1)
        h = nv
    # numpy
    if nv != nk:
        if nv % nk: raise ValueError(...)
        q = np.repeat(q, nv // nk, axis=1); kk = np.repeat(kk, nv // nk, axis=1)

**除不尽的组合给一句明确的错** ✓ —— 宁可拒绝，不要静默算一个错的形状 ✓。

### 六、为它加一份模型 —— 因为判卷人只管一个后端

判卷人拿 IR 生成**运行时**再自己喂尺寸 ✓，所以它判的是
**PyTorch 那一个后端** ✓。NumPy 那一侧的分支只有模型文件能长期管住 ✓
—— 于是加了 `models/kda-gva-shaped.by1` ✓（和 `kda-shaped.by1` **只差一行** ✓）：

    $ python by1exec.py kda-gva-shaped.by1 --compare
    PyTorch 幅度 5.1851e-01   NumPy 幅度 5.1851e-01
    最大绝对差 1.979e-07   相对 3.817e-07   [PASS] 两个后端一致

**要说清楚它证到哪一步** ✓：`--compare` 证的是"两个后端一致" ✓，
**不是"和 fla 一致"** ✗。和 fla 一致那一步是判卷人的 `GVA` 正例 ✓。
两条合起来才够 ✓ —— 一条单独都不够 ✓。

顺带清掉一句**空的承诺** ✓：`kda-shaped.by1` 的注释里写着
"低秩那一支的验证放在 `--gate-lowrank` 那个变体里" ✗ ——
**那个变体不存在** ✓。它现在的判据是判卷人里的"低秩"正例 ✓。

### 七、判据

    $ python src/modelcheck/by1kda.py
      ── 正例（四种尺寸 = 四条代码路径）
              ① 前端     ② 递推     ③ 尾巴
         小    8.019e-08  2.072e-07  9.832e-08  ✓
         中    1.264e-07  1.216e-07  2.944e-07  ✓
         低秩  1.471e-07  1.568e-07  1.983e-07  ✓
         GVA   1.552e-07  2.726e-07  1.678e-07  ✓

      ── 反例（每一条都必须红）
         红线：相对 2.944e-05（= max(1e-6, 正例误差 x 100)）
         门退化成每头一个标量      ① 9.669e-01 ② 9.279e-01 ③ 4.484e-01  红了 ✓
         dt_bias 按头而不是按维   ① 9.755e-02 ② 1.388e-02 ③ 2.374e-02  红了 ✓
         q/k 不做 L2 归一化      ① 8.019e-08 ② 3.764e+01 ③ 1.814e+01  红了 ✓
         输出门整条不乘           ① 8.019e-08 ② 2.072e-07 ③ 9.412e-01  红了 ✓
         归一化在门之前           ① 8.019e-08 ② 2.072e-07 ③ 5.011e-01  红了 ✓
         低秩输出门中间不加 silu   ① 1.471e-07 ② 1.568e-07 ③ 5.051e-01  红了 ✓
         低秩衰减门中间多一个 silu  ① 2.592e-01 ② 3.866e-02 ③ 2.163e-02  红了 ✓

      [PASS]（反例也验过）

三段分开之后，**反例自己会说出它打在哪一段** ✓：
只有尾巴那三条，① 和 ② **精确地等于正例的误差** ✓（它们根本没碰那两段 ✓）；
而"不做 L2 归一化"的 ① 也是正例的误差 ✓ —— 因为它动的是**归一化之后** ✓，
喂进去的那五个张量**一个都没变** ✓。**这不是好看，这是能定位** ✓。

### 八、顺带发现：一个文档里的数字**从来没有判卷人**

`README.md` 里写着 `models/ 26 份 .by1` ✗ —— 而实际是 **29** ✓。
`by1docs` 一直在查"多少行 / 多少节 / 多少个 .by1"✓，而这一处**从来没被查过** ✓，
两个原因叠在一起：

    ① 主语得是**反引号里的路径** ✓，而那行没写反引号 ✗
    ② 量词是"份" ✗ —— 判据只认"行 / 节 / 个 .by1" ✓

补了"目录也算主语" ✓ 之后，拿一个**故意写错的数**去试 —— **它没红** ✗。
真因不在那一支 ✓：

    `measure()` 先 `by1io.read_text(path)` ✓，而主语**现在可以是目录** ✓，
    读目录抛 `IsADirectoryError` ✓ —— 它是 `OSError` 的一种 ✓，
    被那个 `except OSError` **静默接住** ✓ → 返回 `None` ✓ → "量不出来，这条不判" ✓

**又一次静默吞掉，又一次让"没有判卷人"长得像"通过"** ✓ ——
和这一轮 KDA 那处是同一类 ✓，`by1lint` 那条规则防的也是它 ✓。
把那一支**挪到读文件之前** ✓、目录主语数自己 ✓，再验三个方向：

    29（真值）→ 全部对得上 ✓
     3（改坏）→ **README.md:171 说 `models/` 里只有 3 个 —— 实际 29 个** ✓
    29（改回）→ 全部对得上 ✓

**假绿的那个方向验过了，真红的那个方向也验过了** ✓ ——
只验"改回来是绿的"等于什么都没验 ✓。

**而写这一节的时候它又红了一次** ✓ —— 因为上一段**照抄了它的输出** ✓，
于是"说 `models/` 是 3 个"被读成了**这一节自己**的声明 ✓（实际 29 ✓）。
判据没错 ✓，是我把**过去的输出**抄成了**现在的断言** ✓。
那几行改成了不构成声明的写法 ✓ —— 而这一条值得单独记：
**判卷人读的是你写下来的字，不是你心里想的那个意思** ✓。

（还留在那儿的一个数：同一行的"其中 **14 份**对真实 checkpoint 验过" ✓。
那个数要对着 `refs/` 和 `models.tsv` 数 ✓，这一轮没做 ✓，
所以它**仍然是没人管的** ✓ —— 记在这儿，别当它被查过。）

### 九、现在的位置

    机制 KDA：PyTorch ✓ · NumPy ✓ · C ✗ · IR ✓ · codegen ✓
    判卷人：**三段全覆盖** ✓ —— 前端 ✓ 递推 ✓ 尾巴 ✓
            正例四条路径 ✓（单层门 / 低秩门 / GVA / 另一个尺寸）
            GVA：**从"一跑就炸"修到两个后端都能跑** ✓，且对着 fla 判过 ✓

下一步仍然是一件：**C 后端还没有 KDA** ✗ ——
它是三个后端里唯一缺的那个 ✓，而 `by1irentry` 每次都会把它列在
"后端还没实现"那一节里 ✓（**覆盖率缺口，不是失败** ✓，但它是缺口 ✓）。

## 121. 判卷人的**参照物**和被测物错在同一处 —— 抓出它的是源码，不是测试

### 一、起点：这一轮本来只是"把尾巴接上"

上一轮结尾列了判卷人碰不到的两处 ✓（ⓐ 尾巴 ✗ · ⓑ `_gate_in` ✗），
这一轮先补它们 ✓。补完顺手去读了一遍 **fla 的层代码** ✓ ——
然后发现了四处不一致 ✓。

### 二、读的是哪一份，为什么以前没读出这些

在这之前，"KDA 的语义"是从**两个文件**读出来的 ✓：

    fla/ops/kda/naive.py    递推那五行 ✓
    fla/ops/kda/gate.py     门那三行 ✓

而 `fla/layers/kda.py`（**层**：投影、卷积、输出门、归一化）✗
只在 `gpu/by1kda.py` 的 docstring 里被**概述**过一遍 ✓ ——
概述是我写的 ✓。这一轮逐行读原文 ✗，四处对不上 ✓。

### 三、四处不一致，每一处都有出处

**① 输出门的激活是 `sigmoid`，不是 `silu`** ✗

    fla/layers/kda.py:191
        self.o_norm = FusedRMSNormGated(self.head_v_dim,
                                        activation="sigmoid", eps=norm_eps)
    fla/modules/fused_norm_gate.py:101-104
        silu/swish:  y = y * g * sigmoid(g)
        sigmoid:     y = y * sigmoid(g)          <- KDA 走这一支

**② 而它之所以是 silu，是因为那一行写死了** ✗

    def forward(self, x, gate):
        ...
        return (self.w * y) * F.silu(gate.float())

`self.act` 收下了 ✓、存起来了 ✓、**从来没人读** ✓ ——
"数没动"的第一种：**没人读** ✓。

**③ 低秩输出门：两层之间**没有**激活** ✗

    fla/layers/kda.py:187-190
        g_proj = nn.Sequential(nn.Linear(hidden, head_v_dim, bias=False),
                               nn.Linear(head_v_dim, value_dim, bias=True))

`nn.Sequential` 不会自己加激活 ✓。而 by1 的 PyTorch 那边写的是
`g_b(F.silu(g_a(x)))` ✗ —— **NumPy 那边从头就没有那个 silu** ✓。
也就是说：**torch 和 NumPy 在低秩输出门上一直不一致** ✗，
而 `--compare` **从来没红过** ✓，因为没有任何模型走这一支 ✓。

**④ 低秩门的瓶颈宽度是 `head_v_dim`（= `v_dim`），不是 `k_dim`** ✗

    nn.Linear(hidden_size, self.head_v_dim)     <- 瓶颈
    两个真模型 GLM / Ling 都恰好 dk == dv        <- 所以量不出来 ✗

四处都是"**写错了不会报错，只会算错**" ✓ —— 也正是这个机制
自己的 docstring 里写着的那一类 ✓。

### 四、而第 ① 处，判卷人**证明不了** —— 这一节是全文的重点

判卷人当时是**绿的** ✓，而且现在回头看，**它绿得有道理** ✓：

    by1 的 RMSNormGated.forward    F.silu(gate)          ✗
    判卷人的 tail()                 rms_norm_ref(z=gate)  ✗（里面也是 silu）

**两边错得一模一样** ✓。判卷人最该防、也最难自己发现的一种失效 ✓：

> **它的错和被测物的错同源时，它什么都证明不了。** ✗

而且这不是第一次 ✓ —— 上一轮我刚刚关掉同一类的 ⓑ ✓
（`front()` 调被测物自己的 `m._gate_in` ✓），
然后在**尾巴**上又犯了一遍 ✓：参照物那一侧是我照着同一句话写的 ✓。

**抓出它的是读源码那一行** ✓，**不是跑测试** ✓。
所以修法不只是改一个词 ✓：`gate_act()` 旁边把两处出处写死 ✓，
并且给它配了反例 ⑧ ✓ —— 于是"这句话如果错了"至少会**有人要改两处** ✓。

### 五、顺手量出：**两张必须一致的表，不一致**

`by1codegen.ATTRS`（`.by1` 的机制块能写哪些键）✓ 和
`by1ir.KIND_ATTRS`（IR 里合法的属性）✓ 是两层的东西 ✓，
而**一个 IR 属性必须先能写进 `.by1`** ✓。实测：

    by1ir   有 `gate_rank` / `gate_out_bias`  -> codegen 里没有 ✗
    codegen 有 `gate_act`                     -> KDA 类里**没人读** ✗

后果不是报错 ✗，是两种更坏的 ✓：`gate_rank` **从来没被接受过** ✓
（写了会被判"codegen 还不支持" ✓），而 `gate_act` 写了**什么也不做** ✓。
把这一层改成**从 IR 那张表推** ✓ —— 少一处手抄，就少一处对不上 ✓。

（查 `gate_act` 是不是真没人读，用的是**范围搜索** ✓：
   先算出 KDA 类的行区间 ✓，再在里面搜 ✓。全文件搜会搜到
   Attention / MLA 各自的 `gate_act` ✓，那是**同名不同义** ✓。）

### 六、再加一份模型，它立刻抓出**第三个** bug

`models/kda-lowrank-shaped.by1` ✓ —— 存在的理由是第 ③ 条：
这条分支**一直没有任何模型走** ✓。加完第一次跑：

    [注意] PyTorch 侧有 20 个参数没找到对应：
           ['layers.0.op1.f_a.weight', 'layers.0.op1.f_b.weight', ...]

模块那边叫 `f_a` ✓，契约（和 shapes 表）那边叫 `f_a_proj` ✓ ——
**名字对不上，权重就装不进去** ✓。也就是说"低秩门"这个能力
**声明了、建得出来、装不上** ✓。

而这一份模型**本来就能抓住第 ③ 条** ✓ —— 量过 ✓：

    低秩输出门多加一个 silu 的效应：相对 1.915e-01
    两个后端本来就有的差：        相对 4.354e-07
    差多少倍：439722 倍           <- `--compare` 在修之前**一定红** ✓

**"加一份走那条路的模型"值多少，这个数就是答案** ✓。

### 七、GLM：描述的是一个模型，生成的是另一个

GLM 的机制块里**没写 `gate_lowrank`** ✗ ——
而它的契约里声明的是 `f_a_proj` / `f_b_proj`（两层）✓。
默认值是 `false` ✓，于是生成出来的是**一层**的模块 ✓。
**契约查的是权重的名字和形状 ✓，不是算了什么** ✓，
而它卡在 SparseMLA 上跑不了 ✓ —— 所以**一声不吭** ✓。

这一条和 `history/ir.md` §1300 那条是同一类 ✓
（"描述说的是一个模型，生成出来是另一个" ✓）。
补上 `gate_lowrank = true` / `gate_rank = 128` / `gate_out_bias = false` ✓。

### 八、`gate_out_bias`：**两个出处打架，所以它必须是属性**

    fla/layers/kda.py:189     nn.Linear(..., bias=True)     -> 有 ✓
    GLM 的官方权重清单        查 76108 个张量                  -> **没有** ✓

（那 76108 个里 `.bias` 只有 135 个 ✓，全是 indexer 和视觉塔的 ✓。）
原来写死 `bias=True` ✗：对 fla 对 ✓，对 GLM 不对 ✗ ——
而 GLM 跑不了 ✓，所以**没有东西会报** ✗。
做成属性 ✓，默认跟 fla ✓，GLM 显式关掉 ✓。

### 九、判据

    $ python src/modelcheck/by1kda.py
      ── 正例（四种尺寸 = 四条代码路径）
              ① 前端     ② 递推     ③ 尾巴
         小    8.019e-08  2.072e-07  2.556e-07  ✓
         中    1.264e-07  1.216e-07  1.990e-07  ✓
         低秩  1.471e-07  1.568e-07  2.484e-07  ✓
         GVA   1.552e-07  2.726e-07  2.955e-07  ✓

      ── 反例（**八条**，每一条都必须红）
         红线：相对 2.955e-05（= max(1e-6, 正例误差 x 100)）
         门退化成每头一个标量      ① 9.669e-01 ② 9.279e-01 ③ 1.273e+00  红了 ✓
         dt_bias 按头而不是按维   ① 9.755e-02 ② 1.388e-02 ③ 3.150e-02  红了 ✓
         q/k 不做 L2 归一化      ① 8.019e-08 ② 3.764e+01 ③ 1.481e+01  红了 ✓
         输出门整条不乘           ① 8.019e-08 ② 2.072e-07 ③ 4.927e-01  红了 ✓
         门的先后反了             ① 8.019e-08 ② 2.072e-07 ③ 3.788e-01  红了 ✓
         低秩输出门中间多一个 silu  ① 1.471e-07 ② 1.568e-07 ③ 6.831e-02  红了 ✓
         低秩衰减门中间多一个 silu  ① 2.592e-01 ② 3.866e-02 ③ 3.666e-02  红了 ✓
         **输出门用 silu 而不是 sigmoid**
                                 ① 8.019e-08 ② 2.072e-07 ③ 9.561e-01  红了 ✓

    三个 KDA 模型（`kda-shaped` · `kda-gva-shaped` · `kda-lowrank-shaped`）
    的 PyTorch ↔ NumPy 相对差：5.298e-07 · 4.414e-07 · 4.354e-07   ✓
    （`--compare` 证的是"两个后端一致" ✓，**不是"和 fla 一致"** ✗ ——
      和 fla 一致那一步是判卷人 ✓。这一轮正好说明为什么两条都要 ✓：
      第 ③ 条 bug 是**两个后端不一致** ✓，第 ① 条是**两个后端一致地错** ✓。）

### 十、还留着什么

    机制 KDA：PyTorch ✓ · NumPy ✓ · **C ✗** · IR ✓ · codegen ✓
    判卷人：三段全覆盖 ✓ · 正例四条路径 ✓ · 反例八条 ✓

**① 一个说不清的地方，明写出来** ✓：Ling 的 `o_norm` 激活到底是哪个 ✓，
**我们手上没有任何产物能证明** ✗ —— `refs/` 里只有 config 和
张量名，而激活不在名字里 ✓。现在跟的是 fla ✓（`activation="sigmoid"` ✓），
理由是 Ling 的 config 里有 `no_kda_lora` / `kda_safe_gate` /
`kda_lower_bound` ✓ —— 全都是 fla 层的参数名 ✓，像是同一支实现 ✓。
**这是推断，不是证据** ✓，所以写在这儿 ✓。

**② 下一步仍然是一件**：**C 后端还没有 KDA** ✗ ——
三个后端里唯一缺的那个 ✓，`by1irentry` 每次都列 ✓。
（本机有 gcc ✓，档 5 跑得动 ✓，所以它是**做得完**的一件 ✓。）

**③ 一个顺手看到的东西** ✓：`by1files` 数的是 **git 已跟踪的文件** ✓，
于是 `.gitignore` 里的草稿在仓库根目录**越堆越多而没人看得见** ✓
（实测 7 个：4 份 IR dump、1 个脚本、2 个空文件 ✓）。
**那不是"没有"，只是"没有被数"** ✓ —— 记在这儿 ✓。

## 122. KDA 的 **C 后端** —— 三个后端里最后缺的那一个

### 一、上一轮结尾写着"下一步就一件"

    C 后端还没有 KDA ✗ —— 三个后端里唯一缺的那个 ✓，
    而本机有 gcc ✓、档 5 跑得动 ✓，所以它是**做得完**的一件 ✓。

### 二、写法：照 NumPy 那条路径写，但**先把两处坑绕开**

C 那份 `kda()` 的数值路径逐条对着 `by1exec.op_kda` 抄 ✓
（理由和别处一样：**NumPy 那一侧是"另一个后端"，不是"另一份实现"** ✓，
而它自己已经对着 fla 的 `naive_recurrent_kda` 判过 ✓）。

写的时候绕开了两处**会静默算错**的地方 ✓：

**① 深度卷积不能就地写** ✗

    for t: for c: out[t][c] = sum_i in[t+i-(k-1)][c] * w[c][i]

卷积窗口会**读到 t 之后**的位置吗？不会 ✓ —— 它是因果的 ✓。
但反过来，**写到 `in` 自己身上**就会 ✓：`t' > t` 的那一步会读到
**已经 silu 过**的 `in[t][c]` ✓。所以投影结果 `pq/pk/pv` 和
卷积结果 `q/kk/v` 必须是**两个缓冲** ✓。

**② `out` 的读出不能覆盖 delta** ✗

递推一步要算两遍「状态加权和」✓：先 `kv_mem`（配 k）✓，
再读出（配 q）✓，而中间夹着一次「用 delta 更新状态」✓。
`delta` 需要一个地方放 ✓ —— 复用了输出缓冲 `yt` ✓，
因为读出那一步**只读状态和 q** ✓，不读 `yt` ✓。

### 三、判据：三个模型 × 三个后端

    kda-shaped.by1            PyTorch ↔ NumPy 相对 4.756e-07 ✓
                              C       ↔ NumPy 相对 4.426e-07 ✓
    kda-gva-shaped.by1        PyTorch ↔ NumPy 相对 4.414e-07 ✓
                              C       ↔ NumPy 相对 3.322e-07 ✓
    kda-lowrank-shaped.by1    PyTorch ↔ NumPy 相对 4.354e-07 ✓
                              C       ↔ NumPy 相对 4.554e-07 ✓

而 `by1irentry` 那一行变了 ✓ —— 这是这一轮最直接的证据 ✓：

    旧： kda-shaped.by1   [不支持] C 后端还没有 KDA          ✗
    新： kda-shaped.by1   IR 4 层 · PyTorch ✓ · NumPy ✓ · C 165 行  ✓

### 四、"通过"也可能是"那段代码根本没跑"

`C ↔ NumPy` 4.4e-07 说明**两个后端算出同一个数** ✓，
**不说明** C 那段 `kda()` 里每一行都被读到 ✗ ——
一个把输入原样搬过去的空函数也会"不一致" ✓，但一个**少写一支**的
函数**照样能过** ✓（那一支没人走）✓。所以补两个**定点反例** ✓：

    ① 输出门 sigmoid -> silu      相对 7.973e-02  [FAIL] ✓
    ② gate_lower 那一支不认        相对 3.712e-02  [FAIL] ✓

两处**只在 `kda()` 里出现** ✓（换之前先断言了 `count(old) == 1` ✓），
所以红了一定是那段被读到了 ✓。**和判卷人那八条反例是同一个道理** ✓。

### 五、而第 ② 条反例逼出一个真问题：**`gate_lower` 没有模型走**

三个 shaped 模型原来都**没写** `gate_lower` ✗ ——
也就是说那条路径在三个后端里都是**没验过的代码** ✓，
而**Ling 的真模型正要走它** ✓：它的 config 里写着
`kda_lower_bound = -5` ✓ —— 而 by1 的 `Ling-3.0-tiny.by1` 里
**那句话没写进去** ✗，于是默认走的是另一个函数 ✓。

**和 GLM 漏掉 `gate_lowrank` 是同一类** ✓：
契约对得上 ✓（张量名和形状里没有"用哪一支门"这个信息 ✗）、
生成出来的是**另一个模型** ✓、而它跑不了 ✓ 所以一声不吭 ✓。

两处都补了 ✓，`kda-shaped.by1` 也补了 `gate_lower = -5.0` ✓ ——
**为了让这一支有模型走** ✓，于是它现在真的被三个后端各跑一遍 ✓。

（`safe_gate` 不用管 ✓：`fla/layers/kda.py:175-178` 里它只影响
`A_log` 的**初始化**（zeros 还是 `log(uniform(1,16))`）✓，
而前向用的是权重里的值 ✓。）

### 六、顺手清掉两处过期的注释

    Ling 的机制块上写着 "⚠️ KDA 的计算 by1 还没有"          ✗ 过期
    Ling 的 MLA 块上写着 "它的计算 by1 还没有"              ✓ 仍然对

**两条并排放在一起的时候，一条过期会让人不信另一条** ✓ ——
所以过期的那句改成了"这句现在过期了" ✓，而不是删掉了事 ✓
（**删掉就看不出来这里曾经错过什么** ✓）。

### 七、现在的位置

    机制 KDA：PyTorch ✓ · NumPy ✓ · **C ✓** · IR ✓ · codegen ✓
             判卷人 ✓（三段全覆盖 · 四正例 · 八反例）
             三个真模型变体各有一份 shaped 模型管着 ✓
             （单层门 ✓ / GVA ✓ / 低秩两层门 ✓）

**目标第 1 条（机制 KDA）到这里是齐的** ✓ ——
"能真算" ✓ + "有判卷人" ✓，三个后端都算 ✓。

而两个真模型（`Ling-3.0-tiny` · `GLM-5.3-Flash`）**仍然跑不了** ✗ ——
挡路的**不是 KDA** ✓，是 **MLA / SparseMLA** ✓：

    Ling-3.0-tiny.by1   - 机制 'MLA' 的属性 'qk_head' codegen 还不支持
    GLM-5.3-Flash.by1   - 机制 'SparseMLA' 的属性 'index_heads' codegen 还不支持

所以下一步是那两件 ✓ —— 而它们**各自有出处** ✓：
`qk_head`/`index_heads`/`index_dim`/`index_topk` 都是**真模型
config 里的字段** ✓，不是我加的形状 ✓。

## 123. 多模态：**先量清楚"没建模"到底指什么**

### 一、这一步是目标里自己写的第一步

目标第 3 条写着：

> 先量清楚"没建模"具体指什么（是 hparams 里没有、还是层类型里没有、
> 还是张量契约里没有），再决定支持到哪一层：**能描述**和**能算前向**
> 是两件事，要先说清楚做到哪一步。

所以这一轮**不是先写代码** ✓，是先把那张 14 行的审计表整张重测一遍 ✓。

### 二、重测的仪器不新写

`1.md` 里那张表的每一格，都来自 `by1all` 跑的那两条命令 ✓：

    by1verify.py <模型> <config> --config
    by1verify.py <模型> <config> --tensors <清单> --backend <后端>

而"哪个模型配哪份产物"**不手抄** ✓ —— 直接用 `by1all.REAL` 和
`by1all.tensor_list` ✓（那两处是路径规则的真相源 ✓）。

### 三、而它先纠正的是**我上一轮的一个猜测**

上一轮我在文档里写过一句："`1.md` 那张表至少 Gemma 一行已经过期" ✗。
**那句话是错的** ✓ —— 量完发现：

    gemma-4-31B [ggml]     decl 530    ok 530    未覆盖 303   <- 表里那一行量的是**这个**
    gemma-4-31B            decl 1188   ok 1188   未覆盖 0     <- 而这个是**这一轮才进表的**

**那一行没过期** ✓，它量的是 **GGUF 那一套命名** ✓（`by1all.REAL` 里
Gemma 只登记了 `ggml` 那一行 ✓）。而同一个 `.by1` 的
**`torch.module` 那一侧，从来没有进过这张表、也没有进过门** ✗ ——
视觉塔的 `name_vision`、`layer` 块、全局张量，**全都是在"没人跑"的
状态下改的** ✓。

**猜和量的差别就在这儿** ✓：猜出来的是"数字旧了" ✗，
量出来的是"**有一整条路没人看**" ✓ —— 后者严重得多 ✓。

### 四、Gemma 的 `torch.module` 那一侧，差的正是 7 个

    契约声明存在: 1181   名字+形状一致 1181   缺失 0
    契约**未覆盖**的实物张量: 7 个（共 1188 个里的 1%）

而那 7 个**恰好就是"不属于任何层"的那 7 个** ✓（官方一共 1188 个，
减去 1181 个 = 7 ✓）：

    model.language_model.embed_tokens.weight
    model.language_model.norm.weight
    model.vision_tower.patch_embedder.input_proj.weight
    model.vision_tower.patch_embedder.position_embedding_table
    model.vision_tower.std_bias
    model.vision_tower.std_scale
    model.embed_vision.embedding_projection.weight

也就是：**层内的一千多个早就齐了 ✓，层外的 7 个一个没提** ✓。
`by1tens` 里 `global_rows` 只从契约的 `global` 块来 ✓，
而 Gemma 那份**根本没有 `global` 块** ✗。

### 五、比"没声明"更糟的一点：**全局张量根本不在检查范围里**

就算声明了也未必有人查 ✗ —— `by1verify.check_tensors` 里那一段是：

    for (lname, shape_txt, note, _pe) in (global_rows or []):
        if render_g is None:
            break                      # <- 整段跳过

而 `render_g` **只在 emit 里有 `global_name` 时才建** ✓。
Gemma 的 `torch.module` emit 里没有 `global_name` ✗ ——
所以那一行不是"锦上添花" ✓，是"**嵌入表和最终 norm 到底查不查**" ✓。

### 六、改法：一个 `global` 块 + 一行模板

逻辑名直接写官方全名 ✓ —— `physical = rename.get(logical, logical)` ✓，
而 `global_name = "{physical}"` ✓，于是渲染出来就是它本身 ✓，
**不需要 `rename` 那一段** ✓（写了反而是第二处同一件事）✓。

    global {
      model.language_model.embed_tokens.weight          : (vocab, d_model)
      model.language_model.norm.weight                  : (d_model,)
      lm_head.weight                                    : --     <- 权重共享，声明"不该存在"
      model.vision_tower.patch_embedder.input_proj.weight : (1152, 768)
      model.vision_tower.patch_embedder.position_embedding_table : (2, 10240, 1152)
      model.vision_tower.std_bias                       : (1152,)
      model.vision_tower.std_scale                      : (1152,)
      model.embed_vision.embedding_projection.weight    : (d_model, 1152)
    }

结果 ✓：

    契约声明存在: 1188   名字+形状一致 1188   形状不符 0   缺失 0
    [PASS] 1188 个张量的名字与形状全部一致，11 处抑制也正确
    未覆盖: **0**

**1188 是官方那一份的全部** ✓ —— 也就是说 Gemma-4-31B 的
`torch.module` 侧现在是**逐张量全覆盖** ✓，视觉塔那 356 个也在里面 ✓。

（`1152` / `2` / `10240` 在这里是**字面量** ✓ —— `global` 行没有
"哪个栈"这个上下文 ✓，取不到栈的宽度 ✓。**这是这语言现在的一个小缺口** ✓，
不是这一轮要修的 ✓。）

### 七、门跟着补了一行

`by1all.REAL` 里 Gemma 原来只有 `ggml` 那一行 ✓，
补上 `torch.module` 那一行 ✓ 之后：

    档 5：**76 项**，0 项失败，1 项跳过（只剩显卡那项）✓

（上一轮是 75 项 ✓ —— **多出来的那一项就是 Gemma 的 `torch.module`** ✓。）

### 八、量出来的"没建模"是什么样：**几乎全是视觉那一半**

    模型                     契约    命中    未覆盖   未覆盖长什么样
    clef                     851     851     333     全是 model.visual.*
    Qwen3.8-27B              866     866     333     全是 model.visual.*
    Qwen3.6-35B-A3B          712     712     333     全是 model.visual.*
    GLM-5.3-Flash            37534   37534   38574   FP8 的 weight_scale_inv + 视觉塔
    Step-3.7-Flash           753     753     718     视觉塔
    Gemma（ggml）             530     530     303     层内四个 norm + layer_output_scale 的 ggml 名
    Gemma（torch.module）     1188    1188    **0**  ——
    Laguna-XS-2_1            30430   30430   83     少数层的 norm + embed + norm
    其余 7 个                  —       —       0     —

**文本主干早就收敛了** ✓：`claude` 之外的 7 个模型未覆盖是 0 ✓，
而所有非零的都指向**同三件事** ✓：

    ① 视觉塔（clef / Qwen3.8 / Qwen3.6 / GLM / Step-3.7）
       —— 而 clef 那三个的 333 个名字**长得一模一样**（`model.visual.*`）✓
    ② GGUF 那一套命名下的层内张量（Gemma 的 303）
    ③ FP8 的 `weight_scale_inv`（GLM 的一大半）

### 九、所以"支持到哪一层"，这一轮能给出确切的答案

**能描述** ✓ —— 而且现在有一个模型是 **100%**（Gemma 的
`torch.module` 侧，1188/1188）✓。视觉塔**不需要新机制** ✓：
它的每一层就是 `Attention` + `qk_norm` + `FFN` ✓，
真正缺过的语言能力只有一样 —— **栈自己的宽度** ✓，而那个早就加了 ✓。

**能算前向** ✗ —— 一件都没做 ✓，而且**本轮不假装做了** ✗：

    · by1 的三个后端**没有任何视觉前向** ✓（没有 patch 切分、
      没有位置插值、没有视觉-文本拼接）
    · Gemma-4-31B 有 346 亿参数 ✓，这一项在**算力**这一层也过不去 ✓
    · 所以视觉那 356 个张量现在的状态是"**描述得住、算不了**" ✓
      —— 和 KDA 在补完之前的状态**字面上一样** ✓

### 十、下一步（按"能描述"这条线，三件，都不大）

    ① clef / Qwen3.8-27B / Qwen3.6-35B-A3B 的 333 x 3
       —— 三个模型的名字**长得一样** ✓，所以大概率是**一份视觉栈 +
          一个契约块**同时覆盖三个 ✓。先量它们的层数和宽度是不是也一样 ✓。
    ② Gemma 的 ggml 那 303
       —— ggml 那支的 `scope` 里没有 `layer` / `global` ✓，
          补的是"层内四个 norm 在 GGUF 命名下叫什么" ✓。
    ③ Laguna 的 83
       —— 量出来是**少数几层**的 norm 加全局那两个 ✓，
          像是"某几层的层类型不同" ✓，得先看是哪几层 ✓。

### 十一、补量：那三处 333 个**长得一模一样**

写进"下一步"之前先验了假设 ✓（"一份栈 + 一个契约块能不能同时覆盖
三个模型" ✓）：

    clef              视觉 333 个   27 层 x 12 + 9 个不带层号的
    Qwen3.8-27B       视觉 333 个   27 层 x 12 + 9 个不带层号的
    Qwen3.6-35B-A3B   视觉 333 个   27 层 x 12 + 9 个不带层号的

而**逐张量的名字和形状都一样** ✓：

    attn.qkv.weight    (3456, 1152)   <- **融合 qkv，带 bias**
    attn.qkv.bias      (3456,)
    attn.proj.weight   (1152, 1152)   attn.proj.bias (1152,)
    mlp.linear_fc1.weight (4304, 1152)   .bias (4304,)
    mlp.linear_fc2.weight (1152, 4304)   .bias (1152,)
    norm1.weight/.bias (1152,)        norm2.weight/.bias (1152,)
    不带层号的 9 个:  patch_embed.proj.{weight,bias} +
                      merger.norm.{weight,bias} +
                      merger.linear_fc1.{weight,bias} +
                      merger.linear_fc2.{weight,bias}

**假设成立** ✓ —— 三个模型共用**同一份视觉栈 + 同一个契约块** ✓，
所以那 999 个未覆盖是**一次改动的事** ✓，不是三次 ✓。

（而它确实**需要新东西**，两样，都是小的：
  ① `Attention` 要能表达**融合 qkv + bias** ✓ —— 这一条 GPT-2 已经有了 ✓，
     所以大概不用新写 ✓；
  ② `layer` 行要能表达**带 bias 的 norm** ✓（`norm1.bias`）✓ ——
     这一条得看 `layer` 行现在支不支持 bias ✓，是这一轮**没做**的那一件 ✓。）

另外两个视觉塔**不是同一套** ✓，别混：

    Step-3.7-Flash   666 个   47 层 x 14，前缀 `vision_model.*`，
                              还有 `ls_1.gamma` / `ls_2.gamma`（LayerScale）✓
    GLM-5.3-Flash    347 个   24 层 x 14，前缀 `model.visual.blocks.*`，
                              带 `attn.k_norm` ✓

**所以"视觉塔"不是一件事，是至少三件事** ✓ ——
而其中最整齐的那一件（clef 那三个）占 999 个 ✓，先做它 ✓。

## 124. 视觉塔：**999 个张量，一次改动** —— 而它逼出一条语言能力

### 一、先验上一轮那个假设

上一轮量到 clef / Qwen3.8-27B / Qwen3.6-35B-A3B 的 333 个未覆盖
**逐张量同名同形状** ✓，所以写下的判断是"一次改动的事" ✓。这一轮先把它做成。

### 二、而落地时缺的不是"再写一段 .by1"，是一条语言能力

`layer` 那一块是**模型级**的 ✓ —— 所有栈共用同一套层内张量名 ✓。
Gemma 的视觉塔恰好和文本用同一组名字（`input_layernorm` 那一套）✓，
所以没暴露 ✗；而 clef 那三个的视觉块用的是 `norm1` / `norm2` ✗ ——
写进 `layer` 会**给 64 个文本层也发一份**，而它们没有这两个张量 ✓。

所以加的是 **`layer <栈名> { ... }`** ✓：

    layer {
      input_layernorm.weight          : (d_model,)      # 主干
      post_attention_layernorm.weight : (d_model,)
    }
    layer visual {
      norm1.weight : (d_model,)                          # 只有视觉塔有
      norm1.bias   : (d_model,)
      norm2.weight : (d_model,)
      norm2.bias   : (d_model,)
    }

**和 `name_<栈名>` 是同一个理由** ✓ —— 那一边管**物理前缀按栈各写一份** ✓，
这一边管**有哪些张量** ✓。一个 checkpoint 里两套前缀、两套层内张量名，
本来就是同一件事的两面 ✓。

改动落在三处（都不大，而且**没有第二份实现** ✓）：
`by1tens` 的契约解析（认这个块头、验栈名）✓ ·
`layer_rows_by_stack` 改用 `layer:<栈名>`、没有就退回 `layer` ✓ ·
`by1check` 拼 `layer_out` 那一段**一个字没动** ✓ ——
它本来就是"先取按栈的那一份、取不到再退回" ✓。

### 三、顺手踩了一个坑，而那个坑值得**堵死**

第一版把 `scope` 写成两行 ✗：

    scope { Lin = linear_attn., GQA = self_attn., FFN_ = mlp.,
            Vis = attn., VisMlp = mlp.,
            layer = "", global = "" }

结果**整张映射表空着** ✓，渲染出来的名字静默丢掉 `linear_attn.` 那一节 ✓，
而报出来是"**936 个张量实际不存在**" ✗ —— **看起来像契约写错了** ✓。

根因在 `by1check._cont_depth`，而它是对的 ✓：

    只统计圆括号与方括号 —— **花括号是结构，不参与续行判定**

参与的话 `mech X {` … `}` 整个块会被并成一行 ✗。
所以折行的那一半**不会**被并进来 ✓，而 `scope { Lin = …` 被
`parse_stmt` 当成一条普通赋值 ✓，**键是 `scope { Lin`** ✓ ——
一声不吭 ✓。

（`Qwen3.8-27B.by1` 里早就写着"scope 必须写在一行里" ✓，
`llama3-shaped.by1` 也是折行的 ✗ —— **一条写在注释里的规矩，
寿命和那句注释一样短** ✓。所以这一轮不写规矩，写检查器 ✓。）

加的一条解析错误：**出现了 `{` 而这一行里没有配对的 `}`** → 报错 ✓。
而它立刻做了两件事 ✓：

    ① 拦住我刚刚写错的那一份 ✓（报错里说明了为什么，不是"语法错"三个字）
    ② **在 `llama3-shaped.by1` 里找出了第二处同样的静默 bug** ✓ ——
       那一份的 `scope` 也是折行的 ✓，而它一直"通过" ✓

用一份**只差那一行**的临时文件量过 ✓：

    ① 一行      -> scope 表 5 条 ✓
    ② 折两行    -> scope 表 **{}** ✓，而**一声不吭** ✗
    ③ 折两行 + 把新检查关掉 -> 还是 `{}` ✓（证明 ② 的空表就是旧行为）

**"写在注释里的规矩"和"写在检查器里的规矩"差多少，这就是那个差** ✓。

### 四、判据：三份都归零

    clef               契约 1184   名字+形状一致 1184   缺失 0   未覆盖 **0**
    Qwen3.8-27B        契约 1199   名字+形状一致 1199   缺失 0   未覆盖 **0**
    Qwen3.6-35B-A3B    契约 1045   名字+形状一致 1045   缺失 0   未覆盖 **0**

**一共 999 个张量** ✓（333 x 3）—— 而它们是这一轮之前
"看得见但摸不着"的那一批 ✓（`1.md` 那张表里三行都写着 333）。

### 五、而层外那 9 个**不是**同一套 —— 量出来才知道

视觉塔的**层内** 333 个三份逐个相同 ✓，而**层外**的 `merger` ✗：
它的**输出宽度是这个模型自己的 `d_model`** ✓。

    第一版我照抄了 clef 的 `(5120, 4608)` ✗
    Qwen3.6 一跑就报 **2 处形状不符**：契约 [5120, 4608] 实际 **[2048, 4608]** ✓

（`2048` 正是 Qwen3.6 的 `d_model` ✓。）改成 `(d_model, 4608)` 之后归零 ✓。

**"三个模型一样"这句话的边界在哪儿，是这个数告诉我的** ✓：
**层内一样 ✓、层外跟着各自的宽度变 ✗** —— 上一轮的"同名同形状"
是拿层内那 27 x 12 量的 ✓，那句话本身没错 ✓，**它的适用范围**才是关键 ✓。

### 六、现在"能描述"到哪一步了

    文本主干：早就收敛 ✓
    视觉塔：  **clef / Qwen3.8-27B / Qwen3.6-35B-A3B 三份 100%** ✓
              （Gemma 的 torch.module 侧上一轮已经 100% ✓）

    还差的：
      GLM-5.3-Flash    38574   FP8 的 weight_scale_inv（一大半）+ 视觉塔
      Gemma（ggml）     303   层内四个 norm 和 layer_output_scale 的 ggml 名
      Step-3.7-Flash    718   它自己的视觉塔（`vision_model.*`，666 个，
                              还有 `ls_1.gamma` / `ls_2.gamma` 那种 LayerScale）
      Laguna-XS-2_1      83   少数几层的 norm + embed + norm

**而"能算前向"这一半仍然一件没做** ✗ —— 这一轮也没假装做了 ✓。
视觉塔现在多了一个"描述得住、算不了"的例子 ✓，
和上一轮的 Gemma 视觉塔、以及补完之前的 KDA **字面上是同一种状态** ✓。

### 七、门

    快速检查   编译 71 · import 66 · 解析 30 · 判卷人 5 · 行尾 118   [PASS]
    档 5       76 项，0 项失败，1 项跳过（只剩显卡那项）             [全过]

（30 份 `.by1` 里只有 `selftest.by1` 有错 ✓ —— 而它**正好 9 个** ✓，
那是 `by1fast` 拿来当"反向确认"的对照 ✓。）

---

## 125. 多模态"能描述"收尾：**83 / 718 / 38574 一次归零**，而路上有两处"写了等于没写"

### 一、数字（全部这轮现量，`by1verify --backend torch.module`）

    模型                    契约      一致     未覆盖
    Laguna-XS-2_1        30513    30513      0        （原来是 30430 / 83）
    Step-3.7-Flash        1471     1471      0        （原来是 753 / 718）
    GLM-5.3-Flash        76108    76108      0        （原来是 37534 / 38574）

**「现量」不是客套。** 这三个数我原来记的是 83 / 718 / 38574 —— 而那是
**上一轮**的。动手前重跑了，没对上就以新的为准。

### 二、怎么补的

**Laguna（83）**：40 个 `input_layernorm` + 40 个 `post_attention_layernorm`
+ `embed` / `final_norm` / `lm_head`。用的是 `layer {}` 和 `global {}` ——
而 `.by1` 里那条 ⚠️ 说"层内两个 norm 写不进 tensors"**已经过期**：
`layer` 块是为 clef 的视觉塔加的，这里正好用上。

顺手修了命名模板：`{scope}.{physical}` 里那个点挪进 scope
（`self_attn` -> `self_attn.`），否则 `layer` / `global` 的空 scope 会渲染出
`model.layers.0..input_layernorm.weight`。**改前改后逐个名字不变**，
只是空 scope 现在是对的。ggml 侧同一个理由（`{scope}_{physical}` ->
`{scope}{physical}`），未覆盖 122 -> **39**，剩的是 `exp_bias` 的前缀不对称
（GGUF 里叫 `blk.N.exp_probs_b.bias`，没有 `ffn_`）——**一个机制一个 scope，
表达不了"这一个不带前缀"**，如实留着。

**Step-3.7（718）**：`vision_model.transformer.resblocks.{0..46}` 47 层 ×
14 个 = 658（含 `ls_1.gamma` / `ls_2.gamma` —— **LayerScale**，
出处是官方 `vision_config.ls_init_value = 0.1`）+ 层外 8 个 +
`vit_large_projector` 1 个 = 666 + 1；再加 3 层 MTP（`model.layers.45/46/47`）
× 17 个 = 51。**718 = 667 + 51** ✓。

MTP 那三层的注意力**逐个形状落在滑窗那一个等价类里**（q=96 · head_dim=128 ·
qk_norm · head_gate · out_dim=12288），所以**不另写一套契约** —— 共用 `Attn`。
FFN 就是 `Dense`，也是共用。

**GLM-5.3（38574）**：37340 个 FP8 `weight_scale_inv` + 347 个视觉塔 +
887 个 MTP 那一层。

FP8 那批**走 `emit` 的 `quant`，不走契约** —— 和 gpt-oss 的 MXFP4 同一个
理由（"量化与融合是发布决策，不是架构"）。`quant` 的 `scale` 就是两种布局
的分水岭：没有 `scale` 是 MXFP4（权重根本不存在、换成 `_blocks`/`_scales`），
有 `scale` 是 FP8 块量化（权重留着、另发一份伴随张量）。
**伴随张量的形状从权重形状推**（`(out/128, in/128)`）—— 写第二遍一定会
和权重对不上。

**谁被量化了是逐个量出来的**：76108 个张量里带 `weight_scale_inv` 的只有
10 个逻辑名 / 13 个等价类。**KDA 的 9 个投影、`kv_b_proj`、整个 indexer、
路由器、视觉塔、embed / lm_head 都没有。** 第一版按"2-D 权重"推，
当场多出 25 类。

⚠️ 成员表要能带机制前缀：`SparseMLA.o_proj.weight` 必须写全 ——
KDA 也有一个 `o_proj.weight`，而它**没有**被量化。写裸的 `o_proj` 会给 KDA
也发一份，那会当场报「契约声明了、实际不存在」（不是静默错）。

### 三、两处"写了等于没写"，都是撞出来的

**① `index = global` 是死代码。** `by1verify` 里早有一段在读它
（`getattr(st, "assigns", ...)`），而 `info["stacks"]` 是 `(名字, 别名, 层数)`
的**元组** —— `getattr` 永远取到 `{}`，于是 `_glob` 永远是空集，
`index = global` 写了等于没写。**第一个用它的人（GLM 的 MTP）就撞上了**：
报出来是 1753 个「契约声明了、实际不存在」，名字全是 `layers.0`。

不写规矩、写检查器：`by1stacks` 现在验 `index` 的取值（只有 `local` / `global`），
`by1check` 把 `stack_index` 显式放进 `info`。

**② `layer <栈名>` 只对声明了 `d_model` 的栈生效。** `layer_rows_by_stack`
原来是**遍历 `stack_d_model`** 建的 —— 于是 `layer mtp { ... }` 对没写
`d_model` 的栈根本不被读到，那一栈照旧拿模型级的 `layer` 那一份：
多发 6 个 `hc_*`、少发 4 个 MTP 专属张量。**按栈的层内张量和宽度是两件事**，
不该由"有没有声明宽度"来决定。现在所有栈都算。

### 四、反例（四条，逐条实测）

**一条只会通过的规则不是规则。** 这轮新加的三样各拿一份好的描述改坏一处：

    反例                                        结果
    A  FP8 块大小 128 -> 64                    形状不符 37338   红 ✓
    B  拿掉 `index = global`                   缺失 1757       红 ✓
    C  拿掉 `layer mtp`                        缺失 6          红 ✓
    D  `index = global` 拼成 `globl`           by1check 直接拦 ✓

**反例不另写 fixture**：`by1gate.py` 的 `PROBE` 表把它写成"改哪一行 ->
改成什么"，读的人一眼看得见错在哪，也不会多出一份谁都不跑的 `.by1`。

档 5：**76 项，0 项失败，1 项跳过（显卡），1 项已知缺口**（那 25 个张量在
HF 的 model-00009 分片里，镜像的问题）。

---

## 126. 两个机制、一个真错、一个说不清的缺口

### 一、`qk_head` —— 不是新维度，是**别名 + 校验**

挡住 `Ling-3.0-tiny` 的是 `MLA` 的属性 `qk_head`。而它**不是一个新维度**：
官方 config 里 `qk_head_dim` 就等于 `qk_nope_head_dim + qk_rope_head_dim`，
`.by1` 照抄了它。所以做的是**收它、并验它**：

    qk_head != qk_nope + qk_rope  ->  报错，不许静默过

**不验的话它就是"写了不报错、什么也不做"的属性** —— 而那是这个仓库
最忌讳的一类（KDA 的 `gate_act` 就是这么被删掉的）。

判据：`by1gate` 里 `Ling-3.0-tiny` 从"该被拒"搬到"该通过"（**改方向，不删**），
外加两条反例（`qk_head = 193` / `= 0`）实测被拒、理由里点得出 `qk_head`。

### 二、`index_heads` / `index_dim` / `index_topk` —— **DSA 稀疏索引器**

这三个是**真机制**，不是别名。判卷人 `src/modelcheck/by1sparse.py`，
对着 transformers 的 **`GlmMoeDsaIndexer`**（别人写的）比
**"每个 query 挑中了哪些 key"**。

    4 组正例（rope=0 的 GLM 形状 × 2 · rope=64 · rope=16）：
        pairing=interleaved   4/4 挑选集合逐个相同
        pairing=half          2/4（只在 rope 不起作用的两组碰巧一致）
    8 条反例，逐条改坏一处：**8/8 当场变红**

**`interleaved` 是量出来的，不是猜的** —— 和 GLM config 的
`indexer_rope_interleave = true` 是同一句话。

**踩到的两个坑，都记下来：**

① **`topk` 必须远小于 `seq`**，否则判据瞎了。第一版 topk=8 / seq=12，
把 `k_norm` 改坏居然照样一致 —— 因为因果窗口里根本没得挑。
改成 topk=3 / seq=32 才看得见。

② **有一类改动是"挑选集合"这条尺子天生看不见的**：
top-k 对**正的整体缩放**不变。所以 `scale = index_dim^-0.5` 和
`weights_proj(x) * index_heads^-0.5` **写错了也挑出同一批 key** ——
它们只调大小、不调次序。**所以不拿它们当反例**：假的反例比没有反例更糟。

### 三、**一个真错**：GLM 的 MoE 路由编码是错的

`GLM-5.3-Flash.by1` 里写的是 `routing = softmax_topk` 加一个
`scoring = sigmoid`。而 **`scoring` 没有任何人读它**（全库 grep 只有注释提到），
于是前向走的是 **softmax 打分**，官方是 **sigmoid** ——

    判据是 transformers 的 GlmMoeDsaTopkRouter：
        scores = router_logits.sigmoid()
        scores_for_choice = scores + e_score_correction_bias
        按组取前 2 求和 -> 选 topk_group 个组 -> 组内选 top_k
        norm_topk_prob -> * routed_scaling_factor

这逐行就是本库 `routing = sigmoid_group_topk` 那一支（Ling 用的同一个）。
config 也对得上：`scoring_func = "sigmoid"` · `topk_method = "noaux_tc"`。

**这一类错最难看**：契约对（张量名字形状全对）、config 对（57/57）、
`by1verify` 也对 —— 而生成出来是另一个模型。**因为前向从来没跑过。**
和 README 里"`routing = sigmoid_topk` 一直在走 softmax"是同一个病。

**所以加了第二把尺子**：`by1moe` 的「路由编码 vs 官方 config 的 `scoring_func`」，
逐个 `.by1` 对拍。反例（把 GLM 改回 `softmax_topk`）实测 **红** ✓。
`scoring` 这个属性**不加** —— 它和 `routing` 说的是同一件事，
两个地方说同一件事就一定会不一致（这次就是）。

### 四、`index_kpool_compress_*` —— **说不清的那一个**

GLM-5.3-Flash 的 `indexer.` 下面除了 DSA 那四个投影，还有两个：
`index_kpool_compress_gate` [128, 4096] 和 `index_kpool_compress_ape` [4, 128]。

**两处都查过，都没有参照物：**

① 本地依赖里没有。`transformers` 5.15.1 全库搜 `kpool` / `compress_ape` /
   `Glm5Next` **0 处命中**。
② **官方模型仓库里也没有源码。** 实测拉 ModelScope 的
   `ZhipuAI/GLM-5.3-Flash` 文件树（HF 这台机器上不通，ModelScope 通）：
   只有 `config.json` / `generation_config.json` / `chat_template.jinja` /
   `tokenizer*` / `processor_config.json` / 62 个 safetensors 分片 /
   `model.safetensors.index.json` —— **一个 `modeling_*.py` 都没有**。
   （`config.json` 69416 字节，和 `refs/zai-org__GLM-5.3-Flash.config.json`
   同一个大小 ✓ —— 是同一份产物。）

所以：**不实现，也不假装验过。** 它们**描述得出来**（契约里有、名字形状验过），
**算不了**。这一条挡住的是 GLM-5.3-Flash 的**完整**前向 ——
不是索引器本身（索引器验过 ✓）。

**要重开这一条，需要的是 GLM-5.3 的官方 modeling 源码**
（它不在模型仓库里，得从发布方的代码仓库拿）。

### 五、视觉前向：**决定不做，停在「描述」**

结论写进 `1.md` 的「视觉前向：决定不做」，一句话：

> **视觉那批张量的状态是「描述得住、算不了」。**

理由三条，第一条最硬：**本地找不到这些视觉塔的官方实现**（实测 0 处命中），
手写一遍再和自己比，正好是这个仓库最贵的那条教训。
描述这一半已经把价值取走了（逐张量同名同形状 = 没漏）；
剩下的"算不算得对"在没有判卷人的情况下**问不出可证伪的答案**。

**要重开这件事，先要一样东西**：这些视觉塔的官方实现（或一份官方数值产物）。
在那之前，任何"视觉前向做了一半"的说法都是没有判据的。

---

## 127. **判卷人没人跑** —— 而 KDA 那个连跑都跑不起来

这轮加 `by1sparse.py` 的时候顺手查了一件事：**判卷人是不是都在门里**。

    在 by1all 的 JUDGES 里的：by1refs docs files bootir raw dev opdiff
                                mla moe rope instella gpt2 irentry
                                extdemo e2e gpu        （+ 这轮的 sparse）
    **不在的：by1kda · by1ssm · by1real · by1load · by1mem · by1train**

**一个没人跑的判卷人，和没写是一回事。** 它可以一直红着而报告全绿 ——
而这正是 README 里"一份工具报了假的绿"那一类。

**更难看的是 `by1kda` 本身**：这台机器上跑它，报的是

    AttributeError: 'NoneType' object has no attribute 'naive_recurrent_kda'

根因有两层：

① **`fla` 没装**（`fla-core`）。而 KDA 的判卷人是**对着 fla 的
   `naive_recurrent_kda` + `rms_norm_ref`** 比的 —— 参照物拿不到，
   判卷人就跑不了。**"KDA 验过"这句在这台机器上当时是"没验"。**
② **它是崩，不是跳过。** `by1skip.skip()` 的语义是"打印并给退出码"，
   **不是"打印并退出"** —— 调用处要写 `return by1skip.skip(...)`。
   少了 `return` 就继续往下跑，然后在 `naive.naive_recurrent_kda` 上炸。
   `by1kda` 两处都是这么写的（`naive` 和 `rms_ref` 各一处）。

**修法是撞出来的那一类**：装 `fla-core` + `einops` 之后它当场
**PASS（4 正例 + 8 反例全红）** ✓ —— 也就是说判卷人是好的，
**坏的是"拿不到参照物时的退路"**。现在两处都 `return by1skip.skip(...)`，
而且 `load_fla_naive` 拿到文件但 import 不了时（实测：装了 `fla-core`
之后报 `No module named 'einops'`）也按"没装"处理，不炸。

**判卷人进表了**：`by1kda` / `by1ssm` / `by1sparse` 都进了 `JUDGES`。
档 5 从 **76 项 -> 79 项，0 项失败**。

**没进表的还剩 4 个**（`by1real` / `by1load` / `by1mem` / `by1train`）——
它们不是判卷人、是要命令行参数的工具（实测：后两个直接报
`the following arguments are required`），所以不进表是对的；
`by1real` 要先从 ModelScope 下产物（它自己跳过 ✓）。

### 还有一件没复现的红

这一轮某次档 5 报过 `exec llama-shaped.by1` 和 `NumPy gpt-oss-shaped.by1`
两处失败（后者还被归成"检查器自己坏了"）。**重跑两次都是 0 失败**，
单跑这两个也都绿。**没复现 ≠ 没发生** —— 记在这里，
下次再见到就不是第一次了。（怀疑是并行判卷人和前向争 CPU。）

---

## 128. **IR 说不出来"这一层的输入宽度"** —— 视觉栈因此「算不了」，这句话现在有个具体的形状

这一轮为了给 SparseMLA 补三个后端，去看 `by1exec.shapes_of` ——
它给每个 op 推参数形状。**推出来的视觉栈参数形状是错的**：

    模型            栈        生成的参数              checkpoint 的实际形状
    clef          visual   wq (1152, 5120)      qkv.weight (3456, 1152)
    GLM-5.3       visual   wq (1024, 4096)      qkv.weight (3072, 1024)
    Step-3.7      visual   wq (1536, 4096)      in_proj_weight (4608, 1536)

**根因只有一行**：`shapes_of` 里 `d = ir["d_model"]` —— **模型级宽度**，
每一层都用它。而视觉塔的 `d_model` 是 1152 / 1024 / 1536（各自的 `stack visual`）。

**两处都是对的、一处是错的，要分清：**

| | |
|---|---|
| ✅ **张量契约是对的** | `by1verify` 76108/76108 用的是 `stack_d_model`（`by1tens` 算的），视觉张量形状逐个对上 |
| ✅ **MTP 栈是对的** | GLM 的 `q_a_proj (1536, 4096)`、Step 的 `wq (12288, 4096)` —— 因为 MTP 用的就是模型宽度 |
| ❌ **后端参数形状是错的** | `by1exec.shapes_of` 和 `by1c` 用模型级 `d_model` |

**而这不是一个现役的假绿** ✓：**门里跑三后端的 9 个模型逐个查过，没有一个
有"自己 `d_model` 的栈"** ——

    llama-shaped / mixtral-shaped / gpt-oss-shaped / qwen3-next-shaped
    mla-shaped / llama3-shaped / clef-tiny / gpt2-tiny / hello
    -> 自己宽度的栈：无（9/9）

所以那三个后端互拍的绿**没被污染**。但它是**潜在**的：
**任何带独立宽度栈的模型，"三个后端一致"都会是自洽而错的** ——
因为三个后端用的是同一份 `shapes_of`，它们会**一致地错**。
这正是 README 里写的那条："三个后端互拍只能证明自洽，它们可以一致地错。"

**真正的问题比这一行更深：IR 说不出来"这一层的输入宽度"。**
`by1ir` 的 `Attention` 属性里有 `out_dim`（输出投影的输入宽度），**没有
`in_dim`**。而"三个后端只拿 IR 跑"（`by1irentry` 那条）要求 IR 自足 ——
所以这个缺口**没法靠调用方补**，要么 IR 加字段，要么 op 的属性里带上它。
这是个设计决定，不该顺手改。

**它同时给「视觉：算不了」这句话一个具体的形状**：
不是"没人写那几个算子"，而是**生成出来的运行时参数连形状都对不上** ——
要算得先把"每层输入宽度"放进 IR。这一条写进 `1.md` 的已知缺口。
