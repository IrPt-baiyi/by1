# by1 IR 规格

> **这份文档是从 `src/by1ir.py` 里的表生成的**，不是手写的。
> 生成命令：`python src/by1ir.py --spec`
>
> 理由：手写的规格会过期，而过期的规格比没有规格更坏 ——
> 它让读的人以为自己知道。这些表被 `validate()` 和
> `by1all` 真的跑着，所以它们跟得上代码。

**版本：`by1-ir: 1.0`**

---

## 怎么用它

```bash
python src/by1ir.py --spec                  # 生成这份文档
python src/by1ir.py --emit models/<文件>.by1  # 出一份规范化 JSON
python src/by1ir.py --check <文件>.ir.json  # 校验
python src/by1irentry.py                    # 三个后端只从 IR 跑
```

**想写第四个后端的人从这里开始** —— 不需要先学 `.by1`。

---

## 顶层

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `by1-ir` | `str` | 是 | IR 版本。**读的一方必须拒绝不认识的大版本。** |
| `vocab` | `int` | 是 | 词表大小 |
| `ctx` | `int` | 是 | 上下文长度（位置能达到的最大值） |
| `d_model` | `int` | 是 | 残差流宽度 |
| `pos_kind` | `enum:rope,learned` | 是 | 位置信息的种类。`rope`=在注意力里旋转；`learned`=在输入上加一张查表。**这两件事不一样**，所以是一个字段而不是一个布尔。 |
| `n_pos` | `int` | 否 | pos_kind=learned 时必填：查表有多少行 |
| `norm_kind` | `enum:rms,layer` | 是 | 归一化的种类。`rms`=只除均方根；`layer`=减均值+除标准差+有 bias。**名字像，算的不是一回事。** |
| `norm_eps` | `float` | 是 | 归一化的 eps。**必须来自这里，不许各后端写死**（这个 bug 犯过三次） |
| `norm_one_plus` | `bool` | 是 | 归一化权重是乘 `w` 还是乘 `(1+w)` |
| `globals` | `list[Op]` | 是 | 不属于任何层的算子（embed / final_norm / head） |
| `layers` | `list[Layer]` | 是 | 逐层 |

## 层

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `index` | `int` | 是 | 层号。**多栈时是各栈从 0 起**，不是全局序号 |
| `attrs` | `dict` | 是 | 这一层的属性（逐层覆盖合并之后的结果） |
| `ops` | `list[Op]` | 是 | 这一层的算子序列，**顺序有意义** |
| `state` | `list[State]` | 否 | 这一层需要什么状态（KV cache / recurrent / conv） |

## 算子

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `kind` | `str` | 是 | 算子种类，**闭集** —— 见 KIND_ATTRS |
| `mech` | `str` | 是 | 机制名（.by1 里写的那个名字）。只用于报错和取名 |
| `attrs` | `dict` | 是 | 属性。**每种 kind 有自己的必填项** —— 见 KIND_ATTRS |
| `inputs` | `list[str]` | 是 | 输入的值引用（`hidden` 或 `opN.out`） |
| `outputs` | `list[str]` | 是 | 输出的值引用 |

`kind` 是**闭集**：`Add`, `Attention`, `Embed`, `External`, `FFN`, `Head`, `KDA`, `Linear`, `MLA`, `MoE`, `Norm`, `Raw`, `SSM`

## 状态

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `kind` | `enum:recurrent,conv_history,kv_cache,declared_by_impl` | 是 | recurrent / conv_history / kv_cache / declared_by_impl |
| `bounded_by` | `int?` | 是 | 状态有没有上界。None 表示无界，不是「没有」 |
| `shape` | `list[int]?` | 否 | 状态张量的形状 |
| `dtype` | `str?` | 否 | 状态存什么精度（`bf16` / `fp32` …） |
| `reuse` | `enum:none,prefix,unknown` | 是 | none / prefix / unknown —— 能不能跨请求复用 |
| `note` | `str?` | 否 | 自由说明（人看的，不参与计算） |

## 每种算子的属性

**这张表就是「IR 里合法的东西」的唯一定义。**
不在表里的属性 = 不合法；表里必填而缺的 = 不合法。

### `Add`

（没有属性）

### `Attention`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `q` | `int` | 是 | 查询头数 |
| `kv` | `int` | 是 | KV 头数 |
| `head_dim` | `int` | 是 | 每个头的宽度 |
| `out_dim` | `int` | 是 | 输出投影的输入宽度。**不一定等于 q*head_dim**（GPT-OSS 是 4096 而 q*head_dim=4096，Gemma 不是） |
| `bias` | `bool` | 是 | 投影有没有 bias |
| `window` | `int?` | 是 | 滑动窗口。**None=全量** |
| `qk_norm` | `enum:off,per_head,full` | 是 | 拆头前还是后、按 head_dim 还是整宽 —— **三种，不是布尔** |
| `norm_eps` | `float` | 否 |  |
| `norm_one_plus` | `bool` | 否 |  |
| `rope` | `bool` | 是 | **要不要施加旋转**。false 时位置在输入上 |
| `rope_base` | `int` | 是 |  |
| `rope_pairing` | `enum:half,interleaved` | 是 | 配对约定 |
| `rope_partial` | `float` | 是 | 只转前多少比例 |
| `rope_scale` | `float` | 是 | YaRN 的 attention_factor 之类 |
| `yarn` | `dict?` | 是 |  |
| `q_gate` | `bool` | 是 | q 投影输出翻倍，一半当门 |
| `sink` | `bool` | 是 | 每个头一个可学标量参与 softmax |
| `kv_tie` | `bool` | 是 | k 和 v 共用同一块 |
| `head_gate` | `enum:off,per_head` | 是 |  |
| `gate_act` | `enum:softplus,sigmoid` | 是 | **同名不同义的高发区** |

### `Embed`

（没有属性）

### `External`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `lib` | `str` | 是 | 动态库路径（.so / .dylib / .dll / .o）。**相对路径按 IR 文件所在目录解析** —— 否则换个目录就跑不了 |
| `symbol` | `str` | 是 | 符号名。找不到就拒绝，**不静默给个恒等** |
| `weights` | `dict` | 是 | 这个算子自己的张量：{逻辑名: 形状}。**契约照样查** —— 这就是「下沉一层」没有放松的地方 |
| `io` | `enum:same` | 是 | 输出和输入同形。先只支持这一种 —— 多一种就要多一条约定，而约定越多越像糊 |
| `note` | `str?` | 否 | 给人看的说明 |

### `FFN`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `hidden` | `int` | 是 |  |
| `act` | `enum:silu,gptoss,gelu,gelu_new,relu2` | 是 | **闭集** —— 取值不在里面必须拒绝，不许静默走默认分支 |
| `gate` | `bool` | 是 | 有门=三块矩阵；没门=两块。**GPT-2 是没有的那种** |
| `bias` | `bool` | 是 |  |
| `limit` | `float?` | 是 | SwiGLU 的夹取 |
| `alpha` | `float` | 是 | gptoss 那个 1.702 |

### `Head`

（没有属性）

### `KDA`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `k_heads` | `int` | 否 | k 的头数（GLM 那边叫 num_heads） |
| `v_heads` | `int` | 否 | v 的头数。和 k_heads 不等就是 GVA |
| `k_dim` | `int` | 否 | k 每头的宽度（GLM 那边叫 head_dim） |
| `v_dim` | `int` | 否 | v 每头的宽度 |
| `num_heads` | `int` | 否 | alias：k_heads / v_heads |
| `head_dim` | `int` | 否 | alias：k_dim / v_dim |
| `conv_kernel` | `int` | 是 | 深度因果卷积的核长 |
| `conv_bias` | `bool` | 是 |  |
| `gate_lowrank` | `bool` | 是 | 门是不是两层（GLM 的 f_a/f_b）。默认一层（Ling 的 f_proj） |
| `gate_rank` | `int` | 否 | 低秩门的瓶颈宽度 |
| `gate_out_bias` | `bool` | 否 | 输出门第二层有没有 bias。**两个出处不一致**：fla 的层写的是 `bias=True`，而 GLM 的官方权重清单里**没有** `g_b_proj.bias`。所以它必须是属性，不能写死 —— 默认 true（跟 fla） |
| `gate_lower` | `float` | 否 | 给定时走 lower_bound 那一支：g = lower * sigmoid(exp(A_log) * g)（GLM 是 -5.0） |
| `l2_eps` | `float` | 否 | q/k 做 L2 归一化时的 eps |
| `act` | `enum:silu,gptoss,gelu,gelu_new,relu2` | 是 |  |
| `norm_eps` | `float` | 否 |  |

### `Linear`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `k_heads` | `int` | 是 |  |
| `k_dim` | `int` | 是 |  |
| `v_heads` | `int` | 是 |  |
| `v_dim` | `int` | 是 |  |
| `conv_kernel` | `int` | 是 |  |
| `out_dim` | `int` | 是 |  |
| `act` | `enum:silu,gptoss,gelu,gelu_new,relu2` | 是 |  |
| `norm_eps` | `float` | 否 |  |
| `l2_eps` | `float` | 否 |  |

### `MLA`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `q` | `int` | 是 |  |
| `v_dim` | `int` | 是 |  |
| `qk_nope` | `int` | 是 |  |
| `qk_rope` | `int` | 是 |  |
| `q_lora` | `int` | 是 |  |
| `kv_lora` | `int` | 是 |  |
| `head_dim` | `int` | 是 |  |
| `out_dim` | `int` | 是 |  |
| `bias` | `bool` | 是 |  |
| `rope_base` | `int` | 是 |  |
| `pairing` | `enum:half,interleaved` | 是 |  |
| `norm_eps` | `float` | 否 |  |
| `norm_one_plus` | `bool` | 否 |  |
| `head_gate` | `enum:off,per_head` | 是 |  |
| `gate_act` | `enum:softplus,sigmoid` | 是 |  |
| `index_heads` | `int` | 否 | 索引器的头数（GLM-5.3 是 32） |
| `index_dim` | `int` | 否 | 索引器的头维（128） |
| `index_topk` | `int` | 否 | 每层挑多少个 key（2048） |

### `MoE`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `experts` | `int` | 是 |  |
| `top_k` | `int` | 是 |  |
| `hidden` | `int` | 是 | 专家的中间宽度 |
| `shared` | `int` | 是 | 共享专家个数 |
| `shared_hidden` | `int` | 是 |  |
| `shared_gate` | `bool` | 是 |  |
| `routing` | `enum:softmax_topk,topk_softmax,sigmoid_group_topk,sigmoid_topk` | 是 | **四种，名字只差一个词** |
| `router_bias` | `bool` | 是 |  |
| `expert_bias` | `bool` | 是 |  |
| `score_bias` | `bool` | 是 | noaux_tc 的 e_score_correction_bias |
| `n_group` | `int` | 是 | 分组路由 |
| `topk_group` | `int` | 是 |  |
| `routed_scale` | `float` | 是 |  |
| `act` | `enum:silu,gptoss,gelu,gelu_new,relu2` | 是 |  |
| `limit` | `float?` | 是 |  |
| `limit_shared` | `float?` | 是 |  |
| `alpha` | `float` | 是 |  |

### `Norm`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `kind` | `enum:rms,layer` | 是 | 和顶层 norm_kind 一致（冗余但显式） |
| `eps` | `float` | 是 | 来自顶层 norm_eps |
| `one_plus` | `bool` | 是 |  |

### `Raw`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `impl` | `str` | 是 | 逃生舱第一层：raw.py 里的工厂函数名。**这一层要改编译器** —— 加一个机制就得动 src/by1codegen.py。第二层见 External |

### `SSM`

| 属性 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `heads` | `int` | 是 | 头数。**可以写成裸数字** `heads = 64` |
| `head_dim` | `int` | 是 | 每个头的宽度 |
| `ssm_state` | `int` | 是 | 状态维 —— 和 head_dim 是两个东西 |
| `n_groups` | `int` | 是 | B / C 的分组数。和 GQA 的 kv 头同一个意思 |
| `conv_kernel` | `int` | 是 | 深度因果卷积的核长 |
| `expand` | `int` | 是 | d_inner = heads * head_dim 的那个倍数来源 |
| `chunk_size` | `int` | 是 | 分块扫描的块长。**by1 走的是递推** —— 这个属性只进契约，不进计算 |
| `conv_bias` | `bool` | 是 |  |
| `proj_bias` | `bool` | 是 |  |
| `act` | `enum:silu,gptoss,gelu,gelu_new,relu2` | 是 |  |
| `norm_eps` | `float` | 否 |  |

---

## 几条**为什么这样定**

- **`pos_kind` 是一个枚举，不是一个布尔。**
  `rope` 是在注意力里旋转，`learned` 是在输入上加一张查表 ——
  这是两种东西，不是「有没有位置编码」。
- **`norm_kind` 同理。** `rms` 只除均方根，`layer` 还要减均值、
  还有 bias。名字像，算的不是一回事。
- **`Attention.rope` 是一个独立字段。**
  它回答「要不要转」。GPT-2 的教训：**「没有这个机制」以前
  表达不出来**，于是注意力无条件转了一遍，而所有检查都说「过」。
- **闭集就是闭集。** `act` / `qk_norm` / `routing` / `gate_act` /
  `pairing` 只认列出来的取值。取值多一个必须**拒绝**，
  不许静默走默认分支 —— 那会生成一个「看起来对但算错」的模型。
- **`by1-ir` 是必填的。** 没有版本号的 IR 不该被接受：
  读的一方无从判断自己理解的是哪一版。大版本不匹配直接拒，
  不做兼容猜测。
