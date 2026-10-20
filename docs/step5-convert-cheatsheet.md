# Step-5 转换速查（llama.cpp / 转换器社区）

> **TL;DR**：`stepfun-ai/Step-5-Preview-BF16` 是 Step-3.7 同族的双塔 VL 模型（`perception_encoder` 视觉塔 + `step5` 文本塔）。
> 对转换器来说真正**全新**的东西只有一族：**full-attention 层上的稀疏索引注意力**（`sparse_indexer_*` / `ssmax_s`，8 种张量 × 23 层）。
> 其余差异是**参数/形状/精度变化**（head_dim 128→192、MoE 288→352 专家、norm 全部改 F32 存储）和**元数据表达力**（逐层数组、稀疏参数）问题。
> MTP 3 层、混合滑窗、逐层数组 RoPE、sigmoid 路由 + router_bias 在 Step-3.x 里已有先例，映射可以复用。
>
> 数据出处：HF `stepfun-ai/Step-5-Preview-BF16` @ `7734b7ffac118d70d8398bfa3eb0fab007eb45b1`（repo 最后更新 2026-10-15 01:31）。
> 依据 = `config.json` + `model.safetensors.index.json` + 26 个分片的 safetensors 头（**2449 个张量的 dtype/shape 全部实测**，非推测）。
> 抓取复现见文末。文中"推断"字样的角色说明来自 config 字段名对照，确切计算图以官方 modeling 代码为准。
>
> **性质**：发布日当天的**结构对账 + 转换速查**（2449 张量逐项实测）—— **不是发布前预测**。
> by1 的预测范式（先冻结规则 → 公开预测 → 发布后对答案）尚未用于这份工作；本文件是发布后对账。

---

## 1. 模型速览

| 项 | 值 |
|---|---|
| 架构 | `Step5ForConditionalGeneration`（`model_type: step5_vl`）/ 文本塔 `Step5ForCausalLM`（`model_type: step5`） |
| 文本层 | 92 层（+ 3 个 MTP 层，张量层号 92–94） |
| hidden / vocab | 4096 / 128896（`embed_tokens` 与 `lm_head` **不共享**，各一份 [128896, 4096]） |
| 注意力 | GQA：q 64 头 × head_dim 192；kv 4 组 × 192（q `[12288,4096]`，k/v `[768,4096]`，o `[4096,12288]`） |
| 层型 | 3×sliding（窗口 512）+ 1×full 循环；full 层 = 3,7,11,…,91（共 23 层）；MTP 3 层是 sliding |
| FFN | dense 7 层（0,1,2,91,MTP 92–94），intermediate 13824；MoE 88 层（层 3–90） |
| MoE | 352 专家 top-8，moe_intermediate 1536，share_expert 1536，router_bias（F32），sigmoid 路由 ×3.0，`norm_expert_weight`，`need_fp32_gate` |
| RoPE | **逐层数组**（95 项）：sliding 层 θ=1e4 全旋转；full 层 θ=1e7 旋转 1/3（192→64 维）；llama3 缩放（factor 1.0 / high_freq 32）；上限 1M |
| 数值 | norm 与残差走 F32（`norm_dtype: float32`，`fp32_residual_connection: true`）；权重 BF16 |
| MTP | `num_nextn_predict_layers: 3`，独立分片 `model-mtp3-full-00001.safetensors` |
| 视觉塔 | `perception_encoder`：728px / patch 14，width 1536，47 层，layer scale 0.1，quick_gelu，无 CLS / 无 ln_post；后接 2 个 conv 下采样 + 无偏置 projector |
| 分片 | 26 个：`model-00001..00024` + `model-mtp3-full-00001` + `model-vit-00001` |
| dtype 分布 | BF16 1856 个 / F32 593 个（所有 norm、`moe.router_bias`、`sparse_indexer_w`、`ssmax_s`） |

---

## 2. 全新张量：稀疏索引注意力（8 种 × 23 层 = 184 个）

**只出现在 full-attention 层**（层号 3,7,…,91，与 `sparse_config.apply_to_layer_types: ["full_attention"]` 一致），MTP 层没有。这是本次转换唯一需要"新增张量映射"的地方：

| HF 张量名 | dtype | shape | 角色 |
|---|---|---|---|
| `self_attn.sparse_indexer_q.weight` | BF16 | [4096, 4096] | indexer query：16 头 × proxy_dim 256（`sparse_indexer_num_heads: 16`） |
| `self_attn.sparse_indexer_q_norm.weight` | F32 | [256] | q 侧 rmsnorm（`q_norm_type: rmsnorm`，无 bias） |
| `self_attn.sparse_indexer_k.weight` | BF16 | [256, 4096] | indexer key：1 头 × 256（`num_k_heads: 1`） |
| `self_attn.sparse_indexer_k_norm.weight` | F32 | [256] | k 侧 **layernorm**（`k_norm_type: layernorm`） |
| `self_attn.sparse_indexer_k_norm.bias` | F32 | [256] | 同上，带 bias —— 注意与 q_norm 不对称，别按同一模板映射 |
| `self_attn.sparse_indexer_z.weight` | BF16 | [256, 4096] | CSA z 分支（推断，对应 `compression_method: csa_block_compress`）；`csa_z_norm_type: none` → **没有** z 的 norm 张量 |
| `self_attn.sparse_indexer_w.weight` | **F32** | [16, 4096] | 每个 indexer 头一条 4096 维 F32 权重（推断为块压缩用）；全模型最"重"的非权重张量，别丢进量化路径 |
| `self_attn.ssmax_s` | **F32** | [64] | `ssmax` softmax 的可学习 s（`ssmax_s_granularity: q_head`）；**无 `.weight` 后缀**，转换器按裸张量名匹配 |

配套的稀疏参数（不是张量，是 GGUF 元数据）：`topk: 512`、`region_block_size: 8`、`proxy_dim: 256`、`sparse_indexer_rope_dim: 32`、`sparse_indexer_use_rope: true`、`num_provider_groups: 4`、`attention_impl: sparse_gqa`。
注意 **indexer 内部有独立 RoPE 且只作用前 32 维**（`rope_dim: 32`），与主注意力的 partial rotary 是两套。

## 3. 同名但变了的张量（映射可复用，参数要改）

| 张量模式 | Step-3.7 | Step-5 | 备注 |
|---|---|---|---|
| `self_attn.q_proj.weight` | [8192, 4096] / [12288, 4096] | **[12288, 4096]** | head_dim 128→192，64 头 |
| `self_attn.k_proj.weight` / `v_proj` | [1024, 4096] | **[768, 4096]** | kv 组 8→4 |
| `self_attn.o_proj.weight` | [4096, 8192] / [4096, 12288] | **[4096, 12288]** | |
| `self_attn.q_norm.weight` / `k_norm` | BF16 [128] | **F32 [192]** | dtype + 长度都变了（head_dim 级 norm） |
| `self_attn.g_proj.weight` | [64, 4096] / [96, 4096] | **[64, 4096]** | head-wise attention gate，同族继承（`use_head_wise_attn_gate: true`） |
| `mlp.{gate,up}_proj` / `down_proj` | [11264, 4096] | **[13824, 4096]** | 仅 7 个 dense 层 |
| `moe.{gate,up,down}_proj` | 3D [288, …] | **3D [352, …]** | 打包 3D（`gate_proj: [352,1536,4096]`）；不是 per-expert 1D 列表 |
| `moe.gate.weight` | [288, 4096] | **[352, 4096]** | router（sigmoid + ×3.0） |
| `moe.router_bias` | F32 [288] | **F32 [352]** | 同族继承 |
| `share_expert.{gate,up,down}_proj` | 1280 | **1536** | 共享专家 |
| 所有 norm（`input_layernorm` 等 7 种） | BF16 | **F32** | `norm_dtype: float32` —— 转换时别统一 cast BF16 |
| MTP 三件套 `eh_proj`/`enorm`/`hnorm` + `transformer.shared_head.*` | 同名 | 同名 | 各 3 个；`eh_proj: [4096, 8192]`（embed+hidden 拼接），`shared_head.output: [128896, 4096]` |
| 视觉塔全部（`vision_model.*`、`vit_large_projector`） | 同名同形 | 同名同形 | 与 3.7 **完全一致**，映射照抄 |

## 4. 新机制 ↔ config 字段对照

| 机制 | config 字段（值） | 转换影响 |
|---|---|---|
| **稀疏索引注意力** | `sparse_config` 整块（enabled, topk 512, region_block_size 8, proxy_dim 256, num_heads 16, num_k_heads 1, rope_dim 32, use_rope, q_norm rmsnorm / k_norm layernorm / z_norm none, ssmax + s granularity q_head, num_provider_groups 4, csa_block_compress, sparse_gqa, apply_to full_attention） | 新张量映射 + 新 GGUF 元数据 + 运行时需 sparse decode 路径 |
| 混合滑窗 | `layer_types`（3 sliding + 1 full）+ `sliding_window: 512` + `head_dim_full/sliding_attention: 192` + `num_attention_groups_full/sliding: 4` + `num_sliding_attention_heads: 64` | 层型要按层存；3.7 只有全局 head_dim/组数，5 把 full/sliding 拆成两套字段 |
| **逐层数组**（Step 家族惯例"数组短则周期续"） | `rope_theta[95]`、`partial_rotary_factors[95]`、`swiglu_limits[95]`、`swiglu_limits_shared[95]` | 数组长度 **95 = 92 + 3 MTP**，别按 `num_hidden_layers` 截断；本模型数组已满长，无需周期续 |
| RoPE 细节 | `rope_scaling: {llama3, factor 1.0, original_max 1048576, low_freq 1.0, high_freq 32.0}`、`yarn_only_types: [full_attention]` | factor=1.0 但 **high_freq_factor 32 仍进频率公式**，不能按"无缩放"短路；yarn 只用于 full 层 |
| fp32 通道 | `norm_dtype: float32`、`fp32_residual_connection: true`、`rms_norm_eps: 1e-5` | 3.7 config 里没有这几个字段 |
| MoE 路由 | `moe_layer_list`（显式列表 3–90，**取代** 3.7 的 `moe_every_n_layer`/`moe_layer_offset`）、`num_experts_per_tok: 8`、`use_moe_router_bias: true`、`moe_router_scaling_factor: 3.0`、`norm_expert_weight: true`、`need_fp32_gate: true`；`moe_router_activation` 字段**消失**（3.7 显式写 sigmoid） | 层表从"每 N 层"变显式列表；路由激活别默认 softmax |
| SwiGLU 限幅 | `swiglu_limits` / `swiglu_limits_shared`（逐层，全部 7.0） | clamp 7.0 进算子语义，不是训练超参摆设 |
| MTP | `num_nextn_predict_layers: 3` | 独立分片；层号 92–94 |
| 长上下文 | `max_seq_len` / `max_position_embeddings`: 1048576（3.7 为 262144） | |
| VL 接口 | `image_token_len: 169`、`patch_token_len: 81`、`understand_projector_stride: 2`、`projector_bias: false`、`im_start/im_end/im_patch`、`image_token_id: 128001` | 与 3.7 相同 |
| 特殊 token | bos 0、eos [1, 2]（3.7 是 [1, 2, 128007]） | tokenizer 侧注意 |

## 5. 转换器要改哪几处（checklist）

1. **张量映射表**：新增 §2 的 8 个名字。建议 GGUF 侧用 `blk.{i}.attn.sparse_indexer_{q,k,z,w}.weight`、`blk.{i}.attn.sparse_indexer_{q,k}_norm.{weight,bias}`、`blk.{i}.attn.ssmax_s`（命名随上游惯例，关键是别和主注意力 q/k 混前缀）。
2. **F32 张量单列**：所有 norm、`moe.router_bias`、`sparse_indexer_w`、`ssmax_s` 共 593 个 F32 —— 量化/转储路径按 F32 走，别吃进 BF16 批处理。
3. **GGUF 元数据扩容**：现在只有全局标量 `rope_theta`/`sliding_window` 不够。至少要能表达：per-layer 的 θ 与 partial-rotary、per-layer 层型（sliding/full）、per-layer swiglu 限幅、以及 §4 稀疏一整组（topk/block/proxy/rope_dim/norm types/ssmax/compression/provider groups）。
4. **RoPE 处理**：每层 θ 不同（sliding 1e4 / full 1e7）+ full 层只转 64/192 维；indexer 内部另有 32 维 RoPE；llama3 缩放保留 high_freq_factor=32 的频率计算。
5. **稀疏注意力运行时**：转换器只负责搬张量 + 写元数据；`sparse_gqa` 的 top-512 块选择（块大小 8）是 ggml/后端的新 decode 路径（与 DeepSeek DSA 类似），23 个 full 层启用。
6. **MoE**：3D 打包张量 `[352, 1536, 4096]`（in/out 维度方向与常见 per-expert 1D 命名不同，转置/拆分时看清 layout）；router 用 sigmoid ×3.0 + bias，gate 计算 fp32，SwiGLU clamp 7.0，专家输出 norm（`norm_expert_weight`）。
7. **MTP 3 层**：层号 92–94 在 `model-mtp3-full-00001.safetensors` 里。可映射为 draft head 或跳过，但**别当"未知多余张量"报错中断**；它们是完整层（attention + dense MLP + eh_proj/enorm/hnorm + shared_head）。
8. **纯文本转换的最小跳过集**：`vision_model.*`（含 47 个 resblocks）与 `vit_large_projector` 可整体跳过；`embed_tokens` 与 `lm_head` **必须都转**（不共享）。视觉塔命名是 OpenCLIP 风格（`conv1`/`resblocks.*.attn.in_proj`/`c_fc`/`ls_1.gamma`），与 3.7 完全一致，可直接复用既有 VL 映射。
9. **tokenizer**：vocab 128896，bos 0，eos [1, 2]，`<im_start>/<im_end>/<im_patch>`，image_token_id 128001。

## 6. 坑与待确认（首发版，欢迎补刀）

- **`ssmax_s` 长度 64 vs indexer 16 头**：`ssmax_s_granularity: q_head` 的 "q_head" 指主注意力的 64 个 q 头还是 indexer 的 16 头，形状上是 64 —— 以官方 modeling 为准。
- **`sparse_indexer_w` / `z` 的确切计算**（`csa_block_compress` 的块压缩公式）只能按字段名推断，config 里没有算子描述。
- **config 里没有 `auto_map`**（3.7 有 `modeling_step3p7` 等）——建模代码没有随 config 指明，上述"推断"项都等 modeling 文件落地后复核。
- **没有 `generation_config.json`**（404）——采样默认值要自定。
- **`rope_scaling.factor = 1.0` 但带 high_freq_factor 32**：别按"factor=1 就不用管"短路掉频率计算。
- **逐层数组长度 95**（含 MTP 3 层）：转换器按 92 层分配数组会截断 MTP 层的 θ/限幅。
- **3D 打包 MoE**（`[E, out, in]`）与 llama.cpp per-expert 1D 命名的互转是纯 layout 工作，但容易转置错，建议转完立刻做 per-expert 形状断言。

---

## 附录 A · 逐层结构速记

```
层 0,1,2        sliding(512) + dense MLP(13824)            无 MoE 无 indexer
层 3..90        i%4==3 -> full_attention + sparse_indexer   其余 sliding(512)
                全部 MoE(352 专家 top-8 + share_expert 1536)
层 91           full_attention + sparse_indexer + dense MLP 无 MoE
层 92,93,94     sliding + dense MLP + eh_proj/enorm/hnorm + shared_head   (MTP ×3)
全局            embed_tokens + lm_head(不共享) + model.norm
视觉塔          conv1 -> resblocks×47 -> vit_downsampler1 -> vit_downsampler2 -> vit_large_projector
```

RoPE 逐层数组（95 项）：sliding 层 θ=10000、partial=1.0；full 层 θ=1e7、partial=1/3。
`swiglu_limits`/`swiglu_limits_shared`：95 项全 7.0。

## 附录 B · 数据复现

```powershell
# config + 张量索引（小文件，秒回）
curl -L https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/raw/main/config.json
curl -L https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/raw/main/model.safetensors.index.json

# 2449 个张量的 dtype/shape：逐分片读 safetensors 头（Range 请求）
python src/data/by1fetch.py --only Step-5-Preview   # 产出 refs/stepfun-ai__Step-5-Preview-BF16.tensors.json
```

本地已落库：
- `refs/stepfun-ai__Step-5-Preview-BF16.config.json`
- `refs/stepfun-ai__Step-5-Preview-BF16.tensors.json`（2449 个，含 dtype/shape）
- `refs/stepfun-ai__Step-5-Preview-BF16.index.json`（工作缓存，不入语料）
- step5 预览草稿（by1boot 反推：95 层注意力 / FFN×7 / MoE×88，与本文一致；已移出仓库）

## 附录 C · 与 Step-3.7 的张量模式差异统计

- 59 种模式 / 2449 个张量（Step-3.7：51 种 / 1471 个）
- **新增 8 种**：全部是 `sparse_indexer_*` 与 `ssmax_s`（×23 层）
- **消失 0 种**：3.7 的所有张量模式在 5 里都有对应（含 MTP 三件套与 VL 塔）
- **同名变形 18 种**：见 §3（形状 12 种 + norm dtype 6 种合并计）
