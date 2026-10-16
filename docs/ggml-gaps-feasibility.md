# ggml 缺口可行性（2026-10-17）

> 这份记档的**第一步就推翻了自己的前提**：原本要分析"上游缺原语怎么办"，
> 逐个数过 `enum ggml_op`（**101 个**）之后发现 —— **原语都在**。
> 矩阵里那 11 个"卡上游"格全部改判"可补"（见 [backend-matrix](backend-matrix.md)）。

## 一、事实底座（每条都可复核）

复核命令（仓库根目录）：

```bash
# ggml 的算子全集（101 个）
grep -A 220 'enum ggml_op {' llamacpp/llama.cpp/ggml/include/ggml.h | grep -o 'GGML_OP_[A-Z0-9_]*' | sort -u
```

与缺口相关的原语（都在）：

| 原语 | 覆盖 by1 的哪些缺口 |
|---|---|
| `GGML_OP_TOP_K` / `ARGSORT` | MoE 路由（softmax_topk / sigmoid 两种）· SparseMLA 的 top-k |
| `GGML_OP_LIGHTNING_INDEXER` | SparseMLA 的 **DSA 索引器**（`index_heads`） |
| `GGML_OP_GATED_DELTA_NET` | `Linear`（GDN 线性注意力）· **也覆盖 KDA**（见下） |
| `GGML_OP_GATED_LINEAR_ATTN` / `RWKV_WKV7` | 同族的另两种参数化 |
| `GGML_OP_SSM_CONV` / `SSM_SCAN` | SSM（**已在用**，T 已销） |

**已过时的摸底结论**（by1ggml 注释里曾写"这版 ggml 没有 top_k 和比较算子"，
2026-10-14）：已改成实测口径。教训写在原处 —— **摸底结论也要有判据**：
一句"我查过"写在注释里没人复核，就烂成了错误的前提。

## 二、一个关键发现：`GATED_DELTA_NET` **一个原语覆盖两种模式**

它的衰减门 `g` 的形状决定语义（ops.cpp 的 `const bool kda = (neg0 == S_v)`）：

    g 的 ne[0] == 1      -> GDN 模式（逐头标量衰减）   <- by1 的 Linear
    g 的 ne[0] == S_v    -> KDA 模式（**逐维**衰减）   <- by1 的 KDA

也就是说 `Linear` 与 `KDA` 两个族**共用同一条映射路线**（都要折算：
`g = -exp(A_log)·softplus(a + dt_bias)`、`beta = sigmoid(b)`、
q/k 的 L2 归一在 op 外面做）。`qwen3-next`/`clef-tiny` 的 `Linear` 与
`kda-*` 三格，是同一件工程的四个面。

## 三、本轮已经落地的一格（样例：q_gate）

`clef-tiny` 的 Attention 阻塞里有一条 `q_gate` —— **已实现**（探针
`attn-qgate-shaped.by1`，ggml vs NumPy **7.464e-08**）。踩的坑值得留档：

> `q_gate` 的 q 投影宽一倍，但**是逐头的 `[q_hd | gate_hd]` 块**
> （参照实现 `qq.reshape(b, n, q, head_dim*2)` 再按 `:head_dim` 切）——
> 第一版按"两大半"切，`max|Δ| = 1.9e-02`，形状与名字全对、只有数值错。
> **布局类 bug 只有数值判据抓得住**（qkv_fused 的 bias 偏移那次同款）。

## 三·五、路线 ① 的调试台账（`Linear`，**未收口**）

发射器写好了（`by1ggml` 的 `elif k == "Linear"`），但数值差 **~2.7e-3** ——
按"数值没对上的实现不许冒充支持"，`check_support` **拒绝 `Linear`**。
探针 `linear-gdn-shaped.by1`（隔离用：clef 混着 q_gate，二分不了）。

已测的事实（下轮从这接）：

| 实验 | 结果 | 说明 |
|---|---|---|
| 序列长度 16 / 64 / 256 | 2.46e-3 / 2.46e-3 / 2.94e-3 | **不随长度变** —— 不是递推累积，是结构/逐元素错 |
| conv 核 1 / 2 / 4 | 1.28e-3 / 3.43e-3 / 2.7e-3 | **随核变** —— conv 的 tap 布局有嫌疑；但核 1 也错 -> 还有第二个来源 |
| rep=1（k_heads=v_heads） | 9.98e-4 | 头广播（interleave）洗清 |
| 去掉 q 的 `dk^-0.5` 预乘 | 误差变大 | 原语内部有 `scale = 1/sqrt(S_v)`；**但 RMS 归一是尺度不变的**，理论两者等价 —— 这条不一致本身也是线索 |
| `ggml_softplus` | 精确 `log1p(exp(x))`（>20 截断） | 洗清 |
| `ggml_l2_norm` | `1/max(sqrt(sum), eps)` ≠ 参照的 `1/sqrt(sum+eps)` | 差异 ~1e-8 量级，洗清 |
| 形状契约（in_proj_qkvz/ba/conv/A_log/dt_bias/norm.w/out_proj） | 逐个对上 | `norm.w` 是 `(dv,)` **逐维共享跨头**，不是 `(vd,)` |

剩余嫌疑（按优先级）：
1. **conv 的 tap 顺序/布局**（`ggml_ssm_conv` 的核序 vs 参照 `P["conv"][:, i]`；
   核 1 的残差说明还有别的）；
2. `beta`/`g` 的逐头索引（`(1, nv, n)` 的 nb1 步长）；
3. 状态转置约定（内核注释 `s_out[j*S_v+i] = S[i][j]`）与参照 `(dk, dv)` 的对齐。

**收口判据**：`python src/by1ggml.py --from-ir linear-gdn-shaped.by1`
对拍进 1e-6 以内，且 `clef-tiny` 同绿（那时 `check_support` 的拒绝一起撤）。

## 四、剩余 11 格的实施路线（按"一处实现关几格"排）

| 路线 | 关掉的格 | 依赖 |
|---|---|---|
| ① `Linear` -> `GATED_DELTA_NET`（GDN 模式） | clef-tiny · qwen3-next 的一半 | conv/gate/z/norm 的外围照 `op_linear` 拼 |
| ② KDA -> 同原语（KDA 模式） | kda×3 · ling-tiny · glm-text-tiny | 与 ① 共用折算（低秩门是外围） |
| ③ MoE 路由 -> `TOP_K`（+ `ARGSORT`） | moe-routing · mixtral · gpt-oss · qwen3-next 的另一半 | 两种路由（softmax_topk / sigmoid）各一段 |
| ④ SparseMLA 的 `index_heads` -> `LIGHTNING_INDEXER` | mla-sparse · glm-text-tiny 的索引路 | 判据是 `by1sparse.py` 对官方 `GlmMoeDsaIndexer` |

**推荐顺序 ① -> ③ -> ② -> ④**：① 与 ③ 各自独立可判（数值对拍现成），
② 强依赖 ①，④ 有专门判卷人但最复杂。

## 五、什么会推翻这份分析

- ggml 上游**改掉**这些算子的签名（那就重新对表）；
- `op_linear`/`op_kda` 的语义与原语**差一截**（折算不出来）——
  那就按"要么实现，要么拒绝"记回矩阵，理由写**实测差异**，不再写"缺原语"。
