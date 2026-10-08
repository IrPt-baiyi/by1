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
