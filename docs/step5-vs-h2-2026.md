# 2026 H2 四模型架构大表：Step-5 vs Step-3.7 / DSV4.1-Flash / Qwen3.8-Flash-Next / GLM-5.3-Flash

> **数据纪律**：全部字段来自 refs 下的 config.json 与 tensors.json（实测，非推测）；
> 张量数 = safetensors 清单实数。语义存疑处标「⚠」，角色推断标「推断」。
> 抓取复现：python src/data/by1fetch.py；出处审计见 refs/SOURCES.tsv。

## 一、基本盘

| | **Step-5-Preview** | Step-3.7-Flash | DSV4.1-Flash | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|---|---|---|
| model_type | step5_vl / step5 | step3p7 / step3p5 | deepseek_v41 | qwen4_exp | glm5_next |
| 文本层 | **92 + MTP 3** | 45 + MTP 3 | 40（20+20 ⚠草稿口径）+ MTP 3 | 48 + MTP 1 | 45 + MTP 1 |
| hidden | 4096 | 4096 | 5120 | 2560 | 4096 |
| vocab | 128896 | 128896 | 129280 | 248320（+20M ngram 底表） | 154880 |
| 上下文 | **1M** | 262144 | 1M | 262144 | 1M |
| embed / lm_head | 不共享 | 不共享 | 不共享 | 不共享 | 不共享 |
| 视觉塔 | perception_encoder 47 层/1536 | 同左（同塔） | deepseek_v41_vision 32 层/1024 | 27 层/1152 | 24 层/1024 |
| 张量总数 | 2449 | 1471 | **96085** | 1658 | **76108** |

### 表一 · 读表注意

1. **「+MTP」不占主干层号**：MTP 是额外的预测层（3/3/3/1/1 家家不同），Step-5 的张量层号是 92–94（独立分片），DSV4.1 是 mtp.* 命名——别把「92 + MTP 3」读成 92 里含 3 层。
2. **DSV4.1 的 20+20 是草稿口径**：出处是 drafts 里 DSV4.1 草稿的公开摘要（已移出仓库；CED 编解码拆分），config 实测只有 num_hidden_layers=40。字段旁证（kv_source_layer_ids 到 20 为止、candidate_source_layer_id=20、dspark 在 37–39）与 20+20 **相容但不证实**——⚠ 未销。
3. **vocab 已经分叉成两层**：Q3.8-FN 的 248320 之外还有 20M ngram 底表（ngram_vocab_size_base=20000000）；DSV4.1 另有 Engram 16M（engram_vocab_size=16000000，压缩后 99092）。表里只报主词表。
4. **上下文是声明值不是实测值**：1M 来自 max_position_embeddings / max_seq_len；外推机制各家不同（Step-5 llama3 缩放 factor=1.0 但 high_freq_factor=32 仍进频率公式；DSV4.1 YaRN factor=16、基 65536），实际可达长度未验证。
5. **张量总数 ≠ 规模**：它数的是存储布局。DSV4.1 的 96085 里 94821 个是逐专家量化张量（w1/w2/w3 + scale × 专家项）；Step-5 的 352 个专家被 3D 打包成 88×3 个张量。比参数量要另算。

## 二、注意力与稀疏（这代模型真正的分岔口）

| | **Step-5-Preview** | Step-3.7-Flash | DSV4.1-Flash | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|---|---|---|
| 主注意力类型 | GQA | GQA | MLA 系 | GDN 线性（full 层另有注意力） | MLA-NoPE（full 层；线性层另算） |
| q 头 × 头维 | 64 × 192 | 64 × 128 | 64 × 512 | 线性 qk 16 / v 48 × 128；full 24 × 256 | 线性 64 × 128；full qk 256（nope 256 + rope 0）/ v 256 |
| KV 口径 | 4 组 | 8 组 | **1**（MLA 潜向量；q/o LoRA 1280/1024） | full kv 2 组；线性 key 16 / value 48 | full kv 64 头 / kv_lora 512；q_lora 1536 |
| 层型周期 | 3 滑 : 1 全 | 全滑窗 | 滑窗为主（Bounded Replay ⚠） | 每 4 层 1 全 | 每 4 层 1 全（[3,7,…,43]） |
| 滑窗窗口 | 512 | 512 | 128 | —— | —— |
| 索引器头 × 维 | 16 × 256 proxy | —— | 32 × 128 | 4 × 128 | 32 × 128 |
| 稀疏预算 | top-**512** 块 | —— | top-512 块 | 预算 2048 | top-**2048** |
| 块与压缩 | 块 8 · CSA 压缩 · ssmax | —— | 块 8 | 压 4× | kpool 4 压缩 |
| 索引 RoPE | 独立 32 维 | —— | compress_rope θ=160k | —— ⚠ | indexer_rope_interleave |
| 索引共享 | —— | —— | **KV 跨层共享**（source [2,8,14,20]） | —— | MTP 共享索引 |
| head gate | g_proj（head-wise gate） | g_proj | —— | output gate sigmoid | —— |
| mHC 超连接 | —— | —— | ✅ | ✅ | ✅ |
| mHC 参数 | —— | —— | mult 4 / sinkhorn 20 | count 4 / lowrank 320 | mult 4 / sinkhorn 20（逐层 base/scale/fn） |

### 表二 · 读表注意

1. **跨家比 KV 组数没有意义**：DSV4.1 的「kv 1」是 MLA 的压缩潜向量口径（num_key_value_heads=1 + q_lora_rank 1280 + o_lora_rank 1024），GLM 的 num_key_value_heads=64 但真实 KV 存在 kv_lora_rank=512 的潜空间——名字相同语义不同。
2. **GLM 的 head_dim=0 是字段习惯**：实际走 qk_head_dim 256（qk_nope 256 + qk_rope 0）与 v_head_dim 256；主注意力 NoPE，rope 只装在 indexer 上。
3. **topk 的单位是块不是 token**：Step-5 块 8（top-512 块 ≈ 4096 token 窗口）、DSV4.1 candidate_block_size=8、GLM index_kpool=4——512 与 2048 不能直接比预算。
4. **稀疏参数是调度属性不是张量**：kv_source_layer_ids（DSV4.1 跨层 KV 共享 [2,8,14,20]）、index_share_for_mtp_iteration（GLM 索引跨 MTP 共享）都指向别的组件，转换时进元数据，丢了张量映射照样绿。
5. **mHC 三家参数不可混用**：mult 4 / count 4 + lowrank 320 / mult 4 + 逐层 base·scale·fn；表中角色说明是 config 字段名对照，确切计算图待各家官方 modeling。
6. **「每 4 层 1 全」有三种表达**：Q3.8FN 是规则（full_attention_interval=4），GLM 是显式列表（full_attn_layers=[3,7,…,43]），Step-5 是 3:1 循环——转换器都要落成逐层层型，别按一种写法推全局。
7. **Q3.8FN 与 GLM 是双栈模型**：线性层与 full 层的头数/头维是两套（Q3.8FN 线性 qk16/v48×128 vs full 24×256；GLM 线性 64×128 vs full qk256/v256），表中用分号并列——别当同一栈的参数读。

## 三、FFN / MoE / 数值

| | **Step-5-Preview** | Step-3.7-Flash | DSV4.1-Flash | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|---|---|---|
| MoE | 352 专家 top-8 | 288 专家 top-8 | 384 专家 top-6 | 512 专家 top-10 | 288 专家 top-8 |
| 共享专家 | 1536 | 1280 | 1 个（2304） | 640 | 1 个（2048） |
| 路由 | sigmoid ×3.0 + bias | sigmoid ×3.0 + bias | **sqrtsoftplus** + noaux_tc ×1.5 | sigmoid（output gate） | sigmoid + noaux_tc ×2.5 |
| 专家权重布局 | **3D 打包** [E,out,in] | 3D 打包 | **逐专家 1D**（w1/w2/w3 + scale） | gate_up / down 打包 | 逐专家 1D（gate/up/down + scale_inv） |
| dense FFN | 7 层 13824 | 同族 | —— | —— | 3 层（first_k_dense） |
| 激活限幅 | swiglu 7.0（逐层 95 项） | —— | swiglu 10.0 | —— | swiglu 10.0 |
| 权重存储 | BF16 | BF16 | **I8 + F8_E8M0 scale**（专家） | BF16 | **F8_E4M3 + F32 scale_inv**（专家） |
| F32 张量 | 593（norm / indexer_w / ssmax_s） | 42 | 387 | 3（另有 I64 索引） | 37629（scale_inv 为主） |
| norm / eps | RMS 1e-5，norm 全 F32 | RMS，norm BF16 | RMS **1e-20** ⚠ | RMS 1e-6，SSM 走 F32 | RMS 1e-5 |
| F32 残差 | ✅ | —— | —— ⚠ | —— | —— |
| RoPE | **逐层数组 95 项**（1e4 / 1e7 + partial 1/3） | llama3 缩放 ×2.0 | YaRN ×16（基 65536） | θ=1e7，partial 0.25，mrope [11,11,10] | rope 仅 indexer（主注意力 NoPE） |
| MTP | 3 层（独立分片） | 3 层 | 3 层（mtp.*） | 1 层（hybrid full） | 1 层（共享索引） |
| 特殊件 | —— | —— | **Engram**（16M ngram 词表，layer [1,14]）+ **dspark**（target [37-39]） | **PLE**（layer [2]）+ ngram 词表 20M | index_kpool + MTP 索引共享 |

### 表三 · 读表注意

1. **专家数与 top-k 必须合看**：活跃参数 ≈ top-k × moe_intermediate × hidden + 共享专家；且路由分数尺度三家不同（sigmoid×3.0 / sqrtsoftplus×1.5 / sigmoid×2.5），路由 logits 绝不可互换。
2. **布局三派的坑是转置方向**：3D 打包 [E,out,in] 与 llama.cpp 的 per-expert 1D 命名互转最容易错维——转完立刻做 per-expert 形状断言（速查 §5.6）。
3. **dtype 分布别误读**：GLM 的 37629 个 F32 大多是 weight_scale_inv（scale 不是权重）；DSV4.1 的 I8 47232 + F8_E8M0 47589 是专家权重与块 scale 成对出现。真做到「权重全 BF16」的只有 Step 家和 Q3.8FN。
4. **DSV4.1 的 eps=1e-20 是极端值**：config 实测如此（疑似有意配合 F32 gate/残差，⚠ 待官方确认）——跨家比数值行为时别假设 eps 无关紧要。
5. **RoPE 行混着两类东西**：频率方案（llama3 / YaRN / partial）与逐层覆盖（Step-5 的 95 项数组 = 92 + MTP 3，按 92 截断就丢 MTP 的 θ 与 swiglu 限幅）；GLM 主注意力根本没有 rope。
6. **MTP 的「层数」只是预测头个数**：连接公式各家不同（Step-5 按 vLLM 公式且已对拍；GLM 的 MTP 还共享稀疏索引）——不能按同一个 draft 头模板套五家。

## 四、一页结论

1. **2026 H2 的收敛形状**：稀疏索引注意力（top-k 块选）+ 每 N 层 1 个 full 的混合层型 + sigmoid 路由 MoE + MTP + mHC 超连接——五项里 Step-5 / DSV4.1 / Q3.8FN / GLM5.3 四家全中（Step-5 是唯一没上 mHC 的）。
2. **真正的分岔**在主注意力与稀疏的耦合方式：Step-5「GQA full + 自带 indexer」、DSV4.1「MLA + 跨层 KV 共享的 indexer」、Q3.8FN「线性为主 + 附赠小 indexer」、GLM5.3「NoPE-MLA + DSA 大预算 indexer」。
3. **转换器侧最大坑**：① 专家布局三派（3D 打包 / 逐专家 1D + scale / gate_up 打包）；② 量化存储三派（BF16 / I8+F8_E8M0 / F8 block scale_inv）；③ 逐层数组（Step 家）与显式层表（DS / GLM）并存；④ 同名 norm 的 dtype 两家 F32 一家 BF16。
4. **对 by1 的含义**：每家的稀疏索引都是「机制 + 调度属性」的组合（indexer 头数 / 预算 / 块大小 / 来源层各不相同）——正是 pattern 设计里「排属性」问题的压力测试集。
