# Step-5 旗舰合同：工作单（A 线）

> 日期 **2026-10-16** · 目标：把 `drafts/step5-preview-draft.by1`（by1boot 从产物
> **反推**的草稿，by1check 0/0/0 但**没有判卷人，不算数**）升格成 `models/` 里
> 过档 3 的合同。
>
> 本文只放**工作单与证据**，不放"已完成"的错觉。所有字段出自发布方产物
> `refs/stepfun-ai__Step-5-Preview-BF16.config.json` + `.tensors.json`
> （2026-10-15 发布日抓取，2449 张量逐项实测）。
>
> **口径纪律**：Step-5 是**发布日结构对账**，不是预测（见
> [SOURCES.tsv](../refs/SOURCES.tsv) 那一行的性质标注）。

---

## ① 95 层调度 —— **证据已就位**（config 逐字段，可重跑）

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
而 `by1check` 会告诉你换得对不对。现成换算（从 config 推，可直接用）：

| 草稿里的字面 | 符号 |
|---|---|
| `self_attn.q_proj.weight (12288, 4096)` | `(q * head_dim, d_model)`，q=64、head_dim=192 |
| `self_attn.k_proj.weight (768, 4096)` | `(kv * head_dim, d_model)`，kv=4 |
| `self_attn.o_proj.weight (4096, 12288)` | `(d_model, q * head_dim)` |
| `mlp.gate_proj.weight (13824, 4096)` | `(intermediate, d_model)` |

（其余按同法；**换不动的那些**才是真未知，见 ⑤。）

## ③ 后端选择（**要人定**，然后才有 config 映射与 emit 规则）

`complete-*` 走 **Transformers 后端**（`by1full` 直接生成 config，不需要
field 映射、也不需要 `emit[torch.module]`）；**原生 IR 路径**才需要这两样。
Step-5 要走哪条，决定了 ③④ 是"补两个表"还是"补一个家族"：

- **走 Transformers 后端**：要在 `by1full.FAMILIES` 里加 `step5` 家族
  （sparse indexer + 逐层 rope + 逐层窗口），config 由声明生成后仍可用
  `by1complete` 的 `official_configs` 对发布方 config 逐字段判；
- **走 IR 路径**：要写 `emit[torch.module]` 命名规则 + config field 映射，
  判据是 `by1verify` 的 [1]/[3] 两格（现在对 Step-5 是 [跳过]）。

**这是一个方向题，不是实现题** —— 先定它，再动 ④。

## ④ emit / 命名规则（待做，随 ③）

## ⑤ 六个语义未知项：现在还剩几个？

| 未知项 | 现状 | 判卷人 |
|---|---|---|
| 路由方式 / 打分函数 | **部分已答**：`use_moe_router_bias=true`、`moe_router_scaling_factor=3.0`、`norm_expert_weight=true`、352 专家 top-8 | config（已实测） |
| 有没有共享专家 | **已答**：`share_expert_dim=1536` → 有 | config |
| rope 缩放类型 | **已答**：`rope_scaling = llama3`（factor 1.0 / high_freq 32.0） | config |
| 门控激活（head gate 的 act） | **未答** | 官方 modeling 代码 |
| 专家打包（3D 打包 vs 逐专家） | **产物可判**：tensors.json 里那 88×3 个张量的形状 | 产物 |
| bias 有无 | **产物可判**：名字里有没有 `.bias` | 产物 |

**结论**：六个里三个已经从 config 结掉，两个产物可判，**只剩"门控激活"** 需要
官方 modeling 代码 —— 也就是说，"没有判卷人"的那一格比草稿头部写的要小得多。

---

## 顺序与判据

| 步 | 做什么 | 判据 |
|---|---|---|
| 1 | ① 写成 `stack` / `pattern` | `by1check` 展开层表（95 层、23 full/72 sliding）+ `by1verify --layer_types` |
| 2 | ⑤ 里"产物可判"的两项 | 直接对 `refs/stepfun-ai__Step-5-Preview-BF16.tensors.json` 判（名字与形状） |
| 3 | ② 符号化 | `by1check` 换算；换不动的登记为未知 |
| 4 | ③ 后端选择 | 人定；然后再排 ④ |
| 5 | 升格 `models/` + 过档 3 | `by1all --tier 3` |

**不许做的**：把 `drafts/` 那份直接复制进 `models/` —— 它没有判卷人，
`by1check` 0/0/0 只是因为它是"形状清单"而不是"架构声明"（`.by1` 里没有层）。
