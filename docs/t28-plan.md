# T28：同一类的**第二个成员** —— 不同混合图案的 granite-4.0-h-1b

> T27 让 by1 能处理"逐层机制混排"这一类，但**当时只有一个成员**（350m）。
> "支持一类模型"这句话，一个样本证不了。这份计划是**通用性检验**：
> 换一个**图案不同**的同类模型，看那条新路是真通用，还是给 350m 量身定做的。
>
> **红是结果，不是事故** —— 如果它需要改代码，那就是"那一类"还没成立。

## 一、选型（2026-10-16 实测）

| 候选 | 体积 | 判卷人 | 结论 |
|---|---|---|---|
| **granite-4.0-h-1b** | **2.73 GB** | 同族同一套（transformers `GraniteMoeHybridForCausalLM` + llama.cpp `GRANITE_HYBRID`） | ✅ **选它**：图案不同、尺寸不同、μP 不同 |
| granite-4.0-h-tiny | **12.93 GB** | 同上 | ❌ 名字骗人：64 个专家，CPU 上 fp32 放不下（**它才是"有专家"那一支的考验**，留待有卡时） |
| Falcon-H1-0.5B | 0.97 GB | transformers `FalconH1ForCausalLM` · llama.cpp `FALCON_H1` | ⏭ **不是同一根轴**，单独立项（见文末 T29） |

## 二、为什么它算"另一个成员"（实测的结构差异）

| | granite-4.0-h-**350m**（T27 已做） | granite-4.0-h-**1b**（本任务） |
|---|---|---|
| 层数 | 32 | **40** |
| 注意力层（0 基） | 10, 13, 17, 27（**不规则**） | **5, 15, 25, 35**（每 10 层一个） |
| `hidden_size` | 768 | **1536** |
| 注意力头 | 12 / 4，头维 64 | 12 / 4，头维 **128** |
| SSM2 宽度 | 48 × 32 = 1536 | 48 × **64** = 3072（= 2 × 1536） |
| 前馈 | `intermediate_size` = `shared_` = 2048 | **4096 / 4096** |
| μP `residual_multiplier` | 0.246 | **0.22** |
| μP `logits_scaling` | 3 | **6** |
| μP `attention_multiplier` | 0.015625（= 1/64） | **0.0078125**（= 1/128 = 1/头维） |
| `tie_word_embeddings` | true | true |

**共同的**：`position_embedding_type = nope` · `init_method = mup` · `mamba_*` 那一套 ·
`num_local_experts = 0`（dense）· `embedding_multiplier = 12`。

所以这一轮真正在验的是：**逐层类型表**（`layer_types` 由声明里的 `Mixer`/`Attn` 序列映射而来）
对"另一种图案"是不是照样成立；以及那几个"改数值不改形状"的 μP 标量
**换个值也拦得住**（缺一个照样必须报错）。

## 三、四件事（判据沿用 P0/T26/T27 那一套）

1. **声明**：`models/` 下加 `complete-granite-4.0-h-1b`（逐层类型表按
   `5*Mixer + 1*Attn + 9*Mixer + 1*Attn + 9*Mixer + 1*Attn + 9*Mixer + 1*Attn + 4*Mixer`），
   过 `by1check`；`models.tsv` 加行 → `by1fetch` 抓 refs；
2. **配置**：`by1complete` 的 config 逐字段接上（**不改 `by1full`** —— 如果必须改，
   那就是 T27 那条路不够通用的证据，如实记下来）；
3. **真权重**：双 loader 前向（判据 2e-3，争取与 350m 一样逐位 **0**）+
   **混合状态**增量 == 全量（同一个 `--smoke-state-hybrid` 判据，两本账分开报）；
4. **产物腿**：`--arch granite_hybrid --n-blocks 40` → 上游 logits 逐位。
   这一条同时检验 **T27 补的那 6 条转换期改写是不是"只对 350m 有效"**
   （拆分、改名、q/k 置换、Mamba2 三条形状规则）。

## 四、完成定义

- [ ] 声明过 `by1check` + `by1complete`（config 逐字段含 μP 四个标量）；
- [ ] 真权重双 loader：零 missing/unexpected + 前向 max|Δ|（力争 0）；
- [ ] 混合状态增量 == 全量（与 T27 同一个判据、分开报余量）；
- [ ] 上游加载 + logits 逐位（若 llama.cpp 对这个 1b 形态不成立，**如实记为"未验"**）；
- [ ] 报告进 `docs/`；支持矩阵加一行；若为通过它改了 `by1full`，**改动要点写进 T27 那份**。

## 五、边界

- **只做这一个成员**：1b 过了，不等于"所有 granite 形态都过"，更不等于别的族；
- 这一轮**不碰**专家形态（`num_local_experts > 0`）：`by1full` 与产物线现在都
  **明确拒绝**它（"要么实现，要么拒绝"）。要支持它得先把 `granite-4.0-h-tiny`
  那一类搬上显卡，那是另一件事；
- 如果 1b 需要改代码，**不许把改动写成"特判 1b"** —— 要么是通用规则，
  要么明确记为"这一类还没支持"。

---

## 附：T29 的发现（层内**并行** hybrid，先记档）

找第二个成员时顺手查了 **Falcon-H1-0.5B**（0.97 GB，已下载到 `model-cache/`）。
它**不是 granite 那种逐层混排**，读建模源码得到的证据：

```
FalconH1DecoderLayer.__init__:
    self.mamba = FalconH1Mixer(config=config, layer_idx=layer_idx)
    self.self_attn = FalconH1Attention(config, layer_idx)
```

**每一层里同时有 `mamba` 与 `self_attn`** —— 也就是"**层内并行** hybrid"，
而不是"这一层是 SSM、那一层是注意力"。另外：

- `attn_layer_indices` 在 raw config 里是 `None`，在 `to_dict()` 里也是 `None`，
  而且**建模源码里一次都没出现**（`src.count('attn_layer_indices') == 0`）——
  那是个没用的字段；
- `mamba_d_ssm = 1536` **不等于** `expand × hidden = 2048`（SSM 的宽度是
  `n_heads × d_head = 24 × 64`）；
- 十来个缩放标量是无理数（`ssm_multipliers = [0.3535533905932738, …]`，
  `embedding_multiplier = 5.656854249492381`）；
- `tie_word_embeddings = False`（有独立的 `lm_head`）。

**结论**：这是**另一根轴**（层内组合，不是逐层选择），要单独立项。
它的价值同样明确：T27/T28 验的是"层与层不同"，它验的是"**层内两种机制并存**"。
