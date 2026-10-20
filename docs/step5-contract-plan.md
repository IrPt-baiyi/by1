# Step-5 旗舰合同：工作单（A 线）

> 日期 **2026-10-16** · 目标：把 `drafts` 里的 step5 预览草稿（by1boot 从产物
> **反推**的草稿，by1check 0/0/0 但**没有判卷人，不算数**）升格成 `models/` 里
> 过档 3 的合同。
>
> 本文只放**工作单与证据**，不放"已完成"的错觉。所有字段出自发布方产物
> `refs/stepfun-ai__Step-5-Preview-BF16.config.json` + `.tensors.json`
> （2026-10-15 发布日抓取，2449 张量逐项实测）。
>
> **口径纪律**：Step-5 是**发布日结构对账**，不是预测（见
> [SOURCES.tsv](../refs/SOURCES.tsv) 那一行的性质标注）。
>
> **2026-10-16 结论：停在"规格级"，不升格。** 原因是实测出来的 —— 两条后端
> 都缺判卷人（见 ③）。本文从此是**"记档等判卷人"**的一份：判卷人一出现就重启。

---

## ① 95 层调度 —— **证据已就位 + 已落笔验证**

> 落笔产物：step5 调度探针草稿（2.2 KB，**只验调度、不碰契约**；已移出仓库）。

```bash
python src/checks/by1check.py <step5 调度探针草稿>  # 草稿已移出仓库，见 git 历史
#   展开层表：main 92 层 = 23 x [ 3 x Attn(window=512) + Attn(window=none) ]
#             mtp   3 层
#   合计 95 层，d_model = 4096，vocab = 128896（0 错误 / 1 警告）

python src/checks/by1verify.py <step5 调度探针草稿> \
    refs/stepfun-ai__Step-5-Preview-BF16.config.json --sub text_config --config
#   [2] layer_types：层数 by1 = 92 · config = 95（多出 3 项，是周期的延续）
#       [PASS] 92 层类型逐项一致
#       等价类：window=512 → 72 层 · window=none → 23 层
```

**探针里真正声明的东西**（每条都有 config 出处）：

| 声明 | 出处 |
|---|---|
| `schedule { full = 1 * Attn(window=none); slide = 3 * Attn(window=512); block = slide + full }` | `layer_types`：full 落在 3,7,…,91（**周期 4、相位 3**） |
| `pattern = 23 * block` | 23 × 4 = 92 = `num_hidden_layers` |
| `main[0..2] >> Dense` · `main[91] >> Dense` | `moe_layer_list` 的补集（主干稠密 0/1/2/91） |
| `main[3..90] >> MoE` | `moe_layer_list` = 88 层（3…90） |
| `stack mtp { aux = true; index = global; pattern = 3 * Attn(window=512) }` | `num_nextn_predict_layers = 3`，且那三层是 sliding + 稠密 |

> ⚠ **别把这次 PASS 读成全绿**：`by1verify` 的 [1] config 逐字段与 [3] emit
> 仍是 **[跳过]**（探针没有张量契约、也没有 field 映射），所以它整体 exit 1。
> 过的是 **[2] layer_types** —— 判卷人就是 config 自己的逐层数组。

**副产品**：`schedule` / `main[a..b] >>` / `stack mtp { index = global }`
这套写法在同族的 `models/Step-3.7-Flash.by1` 上已验过，在 Step-5 上
**直接成立** —— 也就是说这族的调度**不需要新语言特性**。

| config 字段 | 值 | 用途 |
|---|---|---|
| `num_hidden_layers` | 92 | 主干层数 |
| `num_nextn_predict_layers` | 3 | MTP 层 → **`layer_types` 长度 95** |
| `layer_types` | 95 项：`sliding_attention` **72** / `full_attention` **23** | 注意力类型逐层 |
| `moe_layer_list` | **88** 层（3…90） | MoE 覆盖层 |
| `sliding_window` | 512 | 滑窗宽度 |
| `num_attention_heads` / `head_dim` | 64 / 192 | Q 投影 64×192 = 12288 |
| `num_attention_groups_sliding_attention` | 4 | KV 4 组 → 4×192 = 768 |
| `use_head_wise_attn_gate` | true | Q 投影翻倍（草稿里的 `g_proj (64, 4096)`） |
| `norm_dtype` | float32 | norm 权重按 F32 存储 |
| `intermediate_size` | 13824 | **稠密** FFN 宽 |
| `moe_num_experts` / `moe_top_k` / `moe_intermediate_size` / `share_expert_dim` | 352 / 8 / 1536 / 1536 | MoE 形状（有共享专家） |
| `use_moe_router_bias` / `norm_expert_weight` / `moe_router_scaling_factor` | true / true / 3.0 | 路由侧 |
| `rope_theta` | **95 项**：sliding **10000** / full **10000000** | **逐层** rope 基（按下标给） |
| `rope_scaling` | `llama3` · factor 1.0 · original 1048576 · low 1.0 · **high 32.0** | 频率缩放 |
| `sparse_config` | 见下 | 稀疏索引器 |

```
sparse_config = enabled, 16 头, rope_dim 32, topk 512, softmax ssmax,
                csa_block_compress, region_block_size 8, num_provider_groups 4,
                apply_to_layer_types = ['full_attention']
```

### 调度（可直接写成 `stack` / `pattern`）

- **主干 92 层**：full 层 = **每 4 层的第 4 个**：`3, 7, 11, …, 87, 91`（23 个，
  `rope_theta = 1e7`）；其余 **69 个主干层** sliding（窗口 512，`rope_theta = 1e4`）
  —— **周期 4、相位 3**，是干净的规律，不是逐层列表。
- **FFN / MoE 挂载**：MoE 在层 **3…90**（88 层）；稠密 FFN 在 **0、1、2、91**（4 层）。
- **MTP 3 层（92、93、94）**：**全部 sliding + 稠密**，`rope_theta = 1e4`，
  **不挂稀疏索引器**。
- **稀疏索引器只挂 full 的 23 层**（`apply_to_layer_types = ['full_attention']`；
  MTP 不在其中）。

### 对账：一处"口径差"，已解

反推草稿的注释写着"挂在层上的 `{'FFN': 7, 'MoE': 88}`"，而 config 只给
88 层 MoE + 4 层稠密 —— 看着矛盾，其实是**两个口径**：

> **4（主干稠密：0/1/2/91）+ 3（MTP 层，稠密）= 7** ✓

**反推草稿把 MTP 层算进了 FFN。** 这个口径必须在合同里写死，
否则"看见 7"和"看见 4"的两个人会各写一半、还都自称对。

### 复现

```bash
python -c "import json; c=json.load(open('refs/stepfun-ai__Step-5-Preview-BF16.config.json',encoding='utf-8'))['text_config']; \
print(len(c['layer_types']), c['layer_types'].count('full_attention'), len(c['moe_layer_list'])); \
print([i for i,t in enumerate(c['layer_types']) if t=='full_attention']); \
print(sorted(set(c['rope_theta'])))"
# 95 23 88 / [3,7,...,91] / [10000.0, 10000000.0]
```

---

## ② 703 条字面形状 → 符号（待做）

草稿里的形状是**从 safetensors 头抄下来的字面值**。换成符号才是"我理解了这个模型"，
而 `by1check` 会告诉你换得对不对。

### 单层清单只有**四档**（产物实测，`refs/stepfun-ai__Step-5-Preview-BF16.tensors.json`）

| 档 | 层 | 张量数 | 构成 |
|---|---|---|---|
| 稠密 · 滑窗 | 0, 1, 2 | **12** | 2 norm + 7 attn（`q/k/v/o/g_proj` + `q_norm/k_norm`）+ 3 稠密 mlp |
| MoE · 滑窗 | 69 层 | **17** | 上行 + `moe.{gate,up,down}_proj`(3D) + `share_expert.{gate,up,down}` + `moe.gate.weight` + `moe.router_bias` |
| MoE · 全量 | 22 层 | **25** | 上行 + 7 索引器/`ssmax_s` |
| 稠密 · 全量 | 91 | **20** | 12 + 8 索引器/`ssmax_s` |

**这张表是调度的第二个独立判据**：逐层张量数分布实测
`{12: 3, 17: 69, 25: 22, 20: 1}` —— 3 稠密滑窗 / 69 MoE 滑窗 / 22 MoE 全量 /
1 稠密全量，与 `layer_types` + `moe_layer_list` 的推算**逐层吻合**。

**总量对账**：2449 = 视觉塔 `vision_model.*` **666** + 主干 `model.*` **1781**
+ 全局 2（`lm_head.weight`、`vit_large_projector.weight`）；
1781 = 95 层的逐层张量 + `model.embed_tokens.weight` + `model.norm.weight` ✓。

### 现成换算（从 config 推，可直接用）

| 草稿里的字面 | 符号 |
|---|---|
| `self_attn.q_proj.weight (12288, 4096)` | `(q * head_dim, d_model)`，q=64、head_dim=192 |
| `self_attn.k_proj.weight (768, 4096)` | `(kv * head_dim, d_model)`，kv=4 |
| `self_attn.o_proj.weight (4096, 12288)` | `(d_model, q * head_dim)` |
| `mlp.gate_proj.weight (13824, 4096)` | `(intermediate, d_model)` |
| `moe.gate_proj.weight (352, 1536, 4096)` | `(experts, moe_hidden, d_model)`，**3D 打包** |
| `self_attn.ssmax_s (64,)` | `(q,)`，每 full 层一个 |

（其余按同法；**换不动的那些**才是真未知，见 ⑤。）

## ③ 后端选择 —— **实测后：现在两条都缺判卷人，建议停在规格级**

2026-10-16 实测（这一步把"方向题"变成了"证据题"）：

| 运行时 | Step-5（`step5` / `step5_vl`） | 证据 |
|---|---|---|
| transformers **5.15.1**（本机 CPU 腿） | **没有**：`Step*` 类 0 个、`models/step*` 包 0 个、`CONFIG_MAPPING` 无 `step` 键 | `python -c "import transformers as T; [n for n in dir(T) if 'Step' in n]"` |
| transformers **5.20.0**（`.[full]` 今天解析到的） | **仍然没有 Step5** —— 有的是 **`Step3p7*`**（上一代 Step-3.7：Config/ForConditionalGeneration/Processor/…） | 克隆 venv 实测 |
| llama.cpp（本地检出） | **没有 step5**；架构表里只有 `LLM_ARCH_STEP35`（`"step35"`） | `llamacpp/llama.cpp/src/llama-arch.cpp:152` |

**两条后端都缺判卷人**，于是：

- **走 Transformers 后端**：得先在 `by1full` 里自己写一个 `step5` 家族，
  而**没有任何官方实现可以当判卷人** —— 正是这个仓库最忌讳的那条
  （"没有独立判据的标实验，不计入稳定覆盖数"）。
- **走原生 IR 路径**：写 `emit[torch.module]` + field 映射，判据是 `by1verify`
  的 [1]/[3] —— 但 [3] 的"对不对"最终仍要**独立实现**来判；
  没有它，我们只能自证（`refs/` 判 by1，见
  [criteria-review.md](criteria-review.md) 第四节）。

### 结论与建议

**现在就停在"规格级"**，不再往上堆：

| 已经成立的（有判卷人） | 判卷人 |
|---|---|
| 95 层调度（step5 调度探针草稿，已移出仓库） | **config 自己的 `layer_types`**（`by1verify [2]` PASS） |
| 逐层张量数四档 / 总量 2449 对账 | **发布方产物**（tensors.json） |
| 专家 3D 打包、bias 三处、`ssmax_s`、MTP 自带 shared_head | **发布方产物** |
| 转换器要改的 9 处 | [速查](step5-convert-cheatsheet.md)（对外产物，已按"发布日对账"口径标注） |

**触发条件（watch）**：任一出现就重启本合同 —— ① 上游 transformers 合入
`Step5*`；② llama.cpp 合入 `step5` 架构；③ 官方 modeling 代码公开
（`# ?` 里最后那个"门控激活"也一起解）。在那之前，**不升格进 `models/`**。

> 这条决定本身是可被推翻的：谁要是找到 Step-5 的独立实现（哪怕是一份官方
> modeling 文件），上面每一条都该重排 —— 这正是 `OPEN-` 式的"记档等判卷人"，
> 而不是"做不了所以不做"。

## ④ emit / 命名规则（待做，随 ③）

## ⑤ 六个语义未知项：**只剩一个**（2026-10-16 用产物结掉两项）

| 未知项 | 现状 | 判卷人 / 证据 |
|---|---|---|
| 路由方式 / 打分函数 | **已答**：`use_moe_router_bias=true`、`moe_router_scaling_factor=3.0`、`norm_expert_weight=true`、352 专家选 8 | config |
| 有没有共享专家 | **已答**：`share_expert_dim=1536`，产物里**独立命名** `share_expert.{gate,up,down}_proj.weight`（2D，88 层 × 3） | config + 产物 |
| rope 缩放类型 | **已答**：`rope_scaling = llama3`（factor 1.0 / high_freq 32.0）+ 逐层 `rope_theta` | config |
| **专家打包方式** | **已答（产物）**：`moe.{gate,up,down}_proj.weight` 是 **3D 打包**（352, ·, ·），**没有**逐专家命名（`experts.N.` 命中 0 条）；88 层 × 3 = **264** 条 3D | 产物 |
| **bias 有无** | **已答（产物）**：全库 261 条 `.bias` 只落在三处 —— `moe.router_bias (352,)`（88 层，对上 `use_moe_router_bias`）、`sparse_indexer_k_norm.bias (256,)`（23 层）、视觉塔 238 条；**注意力 7 个投影与主干 norm 一律无 bias** | 产物 |
| 门控激活（head gate 的 act） | **仍未答** | 官方 modeling 代码 |

**顺带两处硬事实（写合同前必须先知道）**：

1. **`ssmax_s (64,)` 是每 full 层一个** —— 索引器张量族共
   **8 种 × 23 层 = 184 条**（`sparse_indexer_{q,k,k_norm.bias,k_norm,q_norm,w,z}` + `ssmax_s`）。
   > 这一条同时**核实了**转换速查里那句"8 种张量 × 23 层"
   > （只数含 `sparse` 的名字会得 7 种 —— `ssmax_s` 不含这个子串）。
2. **每个 MTP 层自带一份 `transformer.shared_head.{norm,output}`**
   （`output` 是 `(128896, 4096)`，3 层各一份）—— 合同里**不能**把它写成
   全局唯一的那一份，否则张量数会对不上（这正是差距表里那类"同名不同义"）。

---

## 顺序与判据

| 步 | 做什么 | 判据 | 状态 |
|---|---|---|---|
| 1 | ① 写成 `stack` / `pattern` | `by1check` 展开层表（95 层）+ `by1verify --layer_types` | **✅ 已落笔并 PASS**（step5 调度探针草稿，已移出仓库） |
| 2 | ⑤ 里"产物可判"的两项 | 直接对 `refs/stepfun-ai__Step-5-Preview-BF16.tensors.json` 判（名字与形状） | **✅ 已结**（专家 3D 打包 + bias 三处，见 ⑤） |
| 3 | ② 符号化 | `by1check` 换算；换不动的登记为未知 | 待做 |
| 4 | ③ 后端选择 | ~~人定~~ → **实测后定**：两条都缺判卷人 → 停在规格级，记 watch 触发条件 | **✅ 已决定（见 ③）** |
| 5 | 升格 `models/` + 过档 3 | `by1all --tier 3` | **暂缓**：等 ③ 的触发条件之一出现 |

**不许做的**：把 `drafts/` 那份直接复制进 `models/` —— 它没有判卷人，
`by1check` 0/0/0 只是因为它是"形状清单"而不是"架构声明"（`.by1` 里没有层）。
