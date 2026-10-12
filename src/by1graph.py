#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1graph -- **原语图：IR kind -> 原语节点的唯一语义定义**（1.0.0-rc0 部件 A）。

## 为什么要有这一层

今天一个算子的语义写在**四个后端各一遍**（by1exec / by1codegen /
by1c / by1ggml），于是这个仓库最贵的 bug 类反复出现：

> **同一个公式用两种语言各写一遍时，变量名相同不代表含义相同。**
> YaRN 的 `1/(fac·p)` 反了 · eps 写死六次 · `qk_norm="off"` 当真值 ·
> 三个 gelu 同名不同义 · kv_tie 的 v 分叉点（记录在案 7+ 次）

原语图把语义**收敛到一处**：IR kind 先降成**原语节点的图**（本文），
四个后端退化成「图 -> 语言」的机械翻译。图里的每个原语只有：

    ① 参数表（闭集，像 by1ir 的 KIND_ATTRS）
    ② 一条公式（下面每个 PRIMS 条目自带）
    ③ 一个 numpy 参考实现 `ref()`（**可执行规格**，B1 的神谕就吃它）

## 判卷人

  · golden 快照：每个可降模型的图**逐字节稳定**（sha256 进
    `graphs-golden.json`），后端重写前后必须同图
  · `--selftest`：参考实现对**手算值**过（ggml-smoke 的思路）
  · 拒绝纪律照旧：降不动的 kind/属性**明确报错**，不许悄悄降错

用法:
  python by1graph.py <model.by1>          # 打印该模型的图 + sha
  python by1graph.py --golden [--write]   # 生成/校验 golden 快照
  python by1graph.py --selftest           # 参考实现对拍手算
  python by1graph.py --e2e [模型...] [--mem-gb N]
                                          # 图求值 vs by1exec（按内存预算跳过）
"""
import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import hashlib
import json

import by1io      # noqa: F401  —— import 即钉 UTF-8

try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:
    _DEFAULT_EPS = 1e-5


# ════════════════════════════════════════════════════════════════════
# 原语表 —— **每条只有：参数表 + 公式 + 参考实现。**
# `src` 列记的是这条语义今天住在哪（搬家前的旧家），出处可追。
# ════════════════════════════════════════════════════════════════════

def _np():
    """numpy 只在参考实现里用 —— **模块本体保持纯标准库**（档 1 要 import）。

    语义的可执行性（B1 的神谕）不等于模块的依赖面：公式是文档，
    ref() 是判卷人，两者同一个家。"""
    import numpy
    return numpy


def _ref_add():
    np = _np()
    return lambda a, b: a + b


def _ref_mul():
    np = _np()
    return lambda a, b: a * b


def _ref_scale():
    np = _np()
    return lambda a, s: a * s


def _ref_matmul():
    # by1exec 的约定：x [..., n, in]，w [out, in]，out = x @ wᵀ [..., n, out]。
    # **ref() 全部用这套约定**（样本在前、特征在最后轴）；图里节点形状标
    # (特征, n) 是 ggml 视角 —— 同一块内存的两种读法，这正是布局原语
    # 存在的原因。两种读法不许混在一条公式里（混过的人出过三次错）。
    #
    # ⚠ **用 `swapaxes(w, -1, -2)`，不是 `w.T`**。`w.T` 在 >2 维时反转
    # **所有**轴：MoE 的专家权重是 (n,K,out,in)，`w.T` 给出 (in,out,K,n)，
    # 于是 `x @ w.T` 报 "core dimension 0 mismatch (size 2 vs 128)" ——
    # 而同一个式子对 2 维权重是对的，所以它**只在 MoE 那一支露头**。
    # 2 维时两者完全一样（swapaxes(-1,-2) == .T），所以这个改动
    # 对既有模型**逐位不变**。
    np = _np()
    return lambda w, x: x @ np.swapaxes(w, -1, -2)


def _ref_batch_matmul():
    # out[β,j,i] = Σ_k a[β//r,i,k] · b[β,j,k]，r = B/b_a（GQA 广播，
    # 和 by1exec 的 np.repeat 同一语义：查询头 h 用 kv 头 h//r）。
    # exec 约定：a [ba, m, k]、b [bb, n, k] -> out [B, n, m]。
    np = _np()

    def f(a, b):
        B = max(a.shape[0], b.shape[0])
        if a.shape[0] < B:
            a = np.repeat(a, B // a.shape[0], axis=0)
        if b.shape[0] < B:
            b = np.repeat(b, B // b.shape[0], axis=0)
        return np.matmul(b, np.swapaxes(a, -1, -2))
    return f


def _ref_rms_norm():
    # y = x * rsqrt(mean(x^2) + eps) * (one_plus ? 1+w : w)
    np = _np()

    def f(x, w, eps, one_plus):
        s = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
        return x * s * ((1.0 + w) if one_plus else w)
    return f


def _ref_layer_norm():
    # y = (x - mean) / sqrt(var + eps) * (one_plus ? 1+w : w) + b
    np = _np()

    def f(x, w, b, eps, one_plus):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        z = (x - mu) / np.sqrt(var + eps)
        return z * ((1.0 + w) if one_plus else w) + b
    return f


def _ref_silu():
    np = _np()
    return lambda x: x / (1.0 + np.exp(-x))


def _ref_gelu_tanh():
    # 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))) —— GPT-2 的 gelu_new。
    # **ggml 的 GELU 是 fp16 查表，不能当这个的判卷人**（rc0 记录在案）。
    np = _np()
    return lambda x: 0.5 * x * (1.0 + np.tanh(
        0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _ref_gelu():
    # erf 精确版：0.5·x·(1+erf(x/√2))
    np = _np()
    return lambda x: 0.5 * x * (1.0 + np.vectorize(__import__("math").erf)(
        x * 0.7071067811865476))


def rope_inv(n_dims, base, yarn=None):
    """θ 的**逆频率表**（长度 `n_dims // 2`）—— 只有这一份。

    三支：普通 · **YaRN** · **llama3**（YaRN 的另一种刻度）。
    逐字搬自 `by1exec.rope_tables` —— 而 `by1codegen` 另有一份 torch 版，
    图的 rope 原语以前又把普通那一支**内联**算了一遍：**一条规则三处实现**。
    搬到这里之后图与 by1exec 共用这一份；torch 那一份仍由
    `src/modelcheck/by1rope.py` 盯着（它对 `_compute_llama3_parameters`）。

    `n_dims` 就是参与旋转的维数（partial rope 时是 `head_dim * rope_partial`）。
    """
    np = _np()
    half = n_dims // 2
    if yarn and (yarn.get("type") or "yarn") == "llama3":
        fac = float(yarn["factor"])
        hf, lf = float(yarn["high_freq"]), float(yarn["low_freq"])
        old = float(yarn["original"])
        p0 = 1.0 / (base ** (np.arange(0, half) / half))
        wl = 2 * np.pi / p0
        lo_wl, hi_wl = old / lf, old / hf
        out = np.where(wl > lo_wl, p0 / fac, p0)
        smooth = (old / wl - lf) / (hf - lf)
        return np.where((wl >= hi_wl) & (wl <= lo_wl),
                        (1 - smooth) * out / fac + smooth * out, out)
    if yarn:
        fac, orig = yarn["factor"], yarn["original"]
        bf, bs = yarn["beta_fast"], yarn["beta_slow"]

        def corr_dim(nr):
            return (n_dims * np.log(orig / (nr * 2 * np.pi))
                    / (2 * np.log(base)))
        lo, hi = corr_dim(bf), corr_dim(bs)
        if yarn.get("truncate", True):
            lo, hi = np.floor(lo), np.ceil(hi)
        lo, hi = max(lo, 0.0), min(hi, n_dims - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(half) - lo) / (hi - lo), 0, 1)
        pos = base ** (np.arange(0, n_dims, 2) / n_dims)
        return (1.0 / (fac * pos)) * ramp + (1.0 / pos) * (1 - ramp)
    return 1.0 / (base ** (np.arange(0, half) / half))


def _ref_rope():
    # 位置 t 对每对维度 (u, v) 转 theta_t = t · inv[i]：
    #   half        -> 配对 (i, i+half)
    #   interleaved -> 配对 (2i, 2i+1)，**输出拼接**
    # 约定：x [..., n, hd]（特征在最后轴、位置在倒数第二轴）、pos [n]。
    # `inv` 由 `rope_inv` 给（普通 / YaRN / llama3 三种都在那一处）；
    # `rope_scale` 是 **cos/sin 的缩放**（和 YaRN 是两件事，by1exec 也分开）。
    np = _np()

    def f(x, pos, base, pairing, n_dims, yarn=None, rope_scale=1.0):
        out = np.array(x, copy=True)
        inv = rope_inv(n_dims, base, yarn)
        half = n_dims // 2
        for i in range(half):
            th = pos * inv[i]                              # [n]
            c, s = np.cos(th) * rope_scale, np.sin(th) * rope_scale
            if pairing == "half":
                u, v = x[..., i], x[..., i + half]
                out[..., i] = u * c - v * s
                out[..., i + half] = u * s + v * c
            else:
                u, v = x[..., 2 * i], x[..., 2 * i + 1]
                # 奇偶进、**拼接**着出
                out[..., i] = u * c - v * s
                out[..., i + half] = u * s + v * c
        return out
    return f


def _ref_soft_max():
    # softmax(a·scale + mask)，沿 **key 轴 = 最后轴**（exec 约定：
    # out [头, query, key]）。
    np = _np()

    def f(a, mask, scale):
        z = a * scale + (mask if mask is not None else 0.0)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    return f


def _ref_soft_max_sink():
    """`soft_max` + **一个 sink 槽**（attention sink）。

    语义逐字照 `by1exec`：`concat([att, sk], -1)` -> softmax -> `[..., :-1]`。
    也就是「在 key 维尾巴上拼一列 sink 再 softmax，然后扔掉那一列」——
    于是注意力权重**不再和为 1**，sink 吸走一部分质量。
    sink 是**逐头**的标量（形状 [头]），广播到 (头, query, 1)。
    **顺序要紧**：先加 mask，再拼 sink —— sink 那一列没有 mask。
    """
    np = _np()

    def f(a, mask, sink, scale):
        z = a * scale + (mask if mask is not None else 0.0)
        sk = np.broadcast_to(np.asarray(sink).reshape(-1, 1, 1),
                             (z.shape[0], z.shape[1], 1))
        z = np.concatenate([z, sk], axis=-1)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        e = e / e.sum(axis=-1, keepdims=True)
        return e[..., :-1]
    return f


def _ref_score(kind):
    """打分函数原语（`softmax` / `sigmoid` / `softplus` / `sqrtsoftplus`）。

    **委托给 `by1ir.moe_scores_array`** —— NumPy 侧唯一的那一份定义
    （取值表 `SCORE_KINDS` 也在那里）。图的 ref **不另写一遍公式**：
    「一条规则三处实现」正是这个仓库在猎的病（`ir.md 130`），
    而这一族刚在 `ir.md 132` 里被拆成独立原语，更不能当场长出第二份。
    """
    def f(x):
        from by1ir import moe_scores_array
        return moe_scores_array(kind, x, _np())
    return f


def _ref_top_k():
    """每行前 k 大的**下标**（不是值）。并列取较小下标（稳定排序）——
    和 `by1exec.op_moe._topk` 同一句：并列处取谁没有定义，两个后端会挑出
    不同的，而那不是 bug（见 `ir.md 129`）。"""
    np = _np()

    def f(x, k):
        return np.argsort(-x, axis=-1, kind="stable")[..., :int(k)]
    return f


def _ref_gather():
    """`np.take_along_axis`：沿**最后轴**按 idx 取值。

    MoE 用它从"选出来的下标"回取分数（`take_along_axis(sc, topi, -1)`）——
    和 `get_rows` 分工不同：那个按**第 0 轴**整行取（专家权重就是它）。"""
    np = _np()
    return lambda x, idx: np.take_along_axis(x, np.asarray(idx).astype(np.int64),
                                             axis=-1)


def _ref_sum():
    """沿 **ggml 视角**的 `axis` 求和（`keepdims` 由 attr 给）。

    attr 用 ggml 视角写（和 `permute` / `slice` 同一套），ref 自己换算
    exec 轴 —— 节点上不出现两种轴。ggml 那边对应 `ggml_sum_rows`（末轴）。"""
    np = _np()

    def f(x, axis, keepdims=True):
        ax = x.ndim - 1 - int(axis)
        return x.sum(axis=ax, keepdims=bool(keepdims))
    return f


def _ref_clamp_min():
    """`max(x, m)`。MoE 归一化的分母要夹一下（`by1exec` 是 `max(sum,1e-9)`，
    sigmoid 那两支是 `sum + 1e-20` —— 两种写法都在账上，别混）。"""
    np = _np()
    return lambda x, m: np.maximum(x, float(m))


def _ref_div():
    np = _np()
    return lambda a, b: a / b


def _ref_get_rows():
    np = _np()
    return lambda table, ids: table[ids]


def _ref_slice():
    """沿 **ggml 视角**的 `axis` 取 `[start, stop)`。

    ggml 轴和 exec 轴是倒序的（`_exec_view` 那一课），所以 ref 自己换算：
    `exec_axis = nd - 1 - axis`。**attr 用 ggml 视角写**，和 `permute` 的
    `axes` 同一套约定 —— 节点上不出现两种轴。"""
    np = _np()

    def f(x, axis, start, stop):
        ax = x.ndim - 1 - int(axis)
        return np.take(x, np.arange(int(start), int(stop)), axis=ax)
    return f


#: **原语表（闭集）。**
#:   f   公式（文档级，唯一）
#:   ref 参考实现（可执行规格；B1 的神谕吃它）
#:   src 这条语义今天的旧家（搬家后仍可追溯）
PRIMS = {
    "add":       {"ins": 2, "f": "a + b（广播）", "ref": _ref_add(),
                  "src": "by1exec.op_add / by1c LAYER_BODY"},
    "mul":       {"ins": 2, "f": "a * b（广播）", "ref": _ref_mul(),
                  "src": "by1exec / by1c 的门控乘法"},
    "scale":     {"ins": 1, "f": "a * s", "ref": _ref_scale(),
                  "src": "gelu_tanh 公式的 0.5 与系数"},
    "matmul":    {"ins": 2,
                  "f": "out[...,i,j] = Σ_k x[...,i,k]·w[j,k] = x @ w.T",
                  "ref": _ref_matmul(),
                  "src": "by1c.linear / ggml_mul_mat / torch F.linear"},
    "batch_matmul": {"ins": 2,
                     "f": "out[β,j,i] = Σ_k a[β//r,i,k]·b[β,j,k]，"
                          "r = B/b_a（GQA 广播）",
                     "ref": _ref_batch_matmul(),
                     "src": "by1exec.op_attention 的 q@kᵀ 与 att@v"},
    "rms_norm":  {"ins": 2, "f": "x · rsqrt(mean(x²)+eps) · (one_plus?1+w:w)",
                  "ref": _ref_rms_norm(),
                  "src": "by1c.rmsnorm / ggml_rms_norm / by1exec.rms_norm"},
    "layer_norm": {"ins": 3,
                   "f": "(x-μ)/sqrt(var+eps) · (one_plus?1+w:w) + b",
                   "ref": _ref_layer_norm(),
                   "src": "by1c.layernorm / ggml_norm / by1exec"},
    "silu":      {"ins": 1, "f": "x · σ(x)", "ref": _ref_silu(),
                  "src": "四后端各一份 —— 收敛于此"},
    "gelu_tanh": {"ins": 1,
                  "f": "0.5·x·(1+tanh(√(2/π)·(x+0.044715x³)))",
                  "ref": _ref_gelu_tanh(),
                  "src": "by1c.gelu_tanh_ / by1exec（ggml 那边要拼原语）"},
    "gelu":      {"ins": 1, "f": "0.5·x·(1+erf(x/√2))", "ref": _ref_gelu(),
                  "src": "by1c.gelu_erf_ / ggml_gelu_erf"},
    "sigmoid":   {"ins": 1, "f": "σ(x)（= `moe_scores_array('sigmoid')`）",
                  "ref": _ref_score("sigmoid"),
                  "src": "by1ir.moe_scores_array / ACT2FN"},
    # **打分函数那一族**（MoE 路由的第一步：logits -> scores）。
    # 取值照 `by1ir.SCORE_KINDS`，语义委托给 `moe_scores_array` ——
    # `routing` 说"怎么挑专家"，`score` 说"logits 怎么变成分数"，正交。
    # 现在**还没有 lowering 用它们**（MoE 的 lowering 是下一个增量），
    # 但它们在闭集里各就各位，且被 `--selftest` 对着手算盯着。
    "softmax":   {"ins": 1, "f": "exp(x-max)/Σ，沿最后轴",
                  "ref": _ref_score("softmax"),
                  "src": "by1ir.moe_scores_array / ACT2FN"},
    "softplus":  {"ins": 1, "f": "log(1+eˣ)（数值稳定：max(x,0)+log1p(e^-|x|)）",
                  "ref": _ref_score("softplus"),
                  "src": "by1ir.moe_scores_array / ACT2FN"},
    "sqrtsoftplus": {"ins": 1, "f": "√softplus(x)（DeepSeek V4 的打分函数）",
                     "ref": _ref_score("sqrtsoftplus"),
                     "src": "by1ir.moe_scores_array / ACT2FN"},
    # **路由 / 归约那一族**（MoE 的"选专家"与"加权求和"）。
    "top_k":     {"ins": 1, "f": "每行前 k 大的**下标**（并列取小）",
                  "ref": _ref_top_k(),
                  "src": "by1exec.op_moe._topk / ggml_top_k"},
    "gather":    {"ins": 2, "f": "沿最后轴按 idx 取值（take_along_axis）",
                  "ref": _ref_gather(),
                  "src": "by1exec 的 np.take_along_axis"},
    "sum":       {"ins": 1, "f": "沿 ggml 轴 axis 求和（keepdims 可关）",
                  "ref": _ref_sum(),
                  "src": "ggml_sum_rows / numpy .sum"},
    "clamp_min": {"ins": 1, "f": "max(x, m)", "ref": _ref_clamp_min(),
                  "src": "by1exec 的 np.maximum(分母, eps)"},
    "div":       {"ins": 2, "f": "a / b（广播）", "ref": _ref_div(),
                  "src": "ggml_div / numpy"},
    "rope":      {"ins": 2, "f": "每对维度转 θ_t = t·inv[i]（inv 见 rope_inv："
                                "普通 / YaRN / llama3）；"
                                "half 配对 (i,i+h)，interleaved 配对 (2i,2i+1)"
                                "且**拼接输出**",
                  "ref": _ref_rope(),
                  "src": "by1c.rope / by1exec.apply_rope / ggml_rope_ext"},
    "soft_max":  {"ins": 2, "f": "softmax(a·scale + mask)，沿 key 轴",
                  "ref": _ref_soft_max(),
                  "src": "by1c.attention / ggml_soft_max_ext"},
    "soft_max_sink": {"ins": 3,
                      "f": "softmax(a·scale + mask)，但尾巴上多一个 sink 槽",
                      "ref": _ref_soft_max_sink(),
                      "src": "by1exec.attention 的 concat+softmax / by1c 的 sink 分支"},
    "get_rows":  {"ins": 2, "f": "out[i] = table[ids[i]]", "ref": _ref_get_rows(),
                  "src": "ggml_get_rows / torch nn.Embedding"},
    # 布局原语：语义是**数据重解释**，不是数学。
    "view":      {"ins": 1, "f": "重解释形状（元素序不变）", "ref": None,
                  "src": "reshape / ggml_reshape"},
    "permute":   {"ins": 1, "f": "轴置换（带步长的视图）", "ref": None,
                  "src": "transpose / ggml_permute"},
    "cont":      {"ins": 1, "f": "按当前视图物化为连续内存", "ref": None,
                  "src": "ggml_cont / numpy .copy()"},
    # **切片**：融合权重拆回分块要它（`qkv` -> wq/wk/wv）。
    # attr 的 `axis` 按 **ggml 视角**（和 permute 同一套），ref 自己换算。
    "slice":     {"ins": 1, "f": "沿 ggml 轴 axis 取 [start, stop)",
                  "ref": _ref_slice(),
                  "src": "ggml_view_* 的 offset / numpy 切片"},
}

# **还没收编的原语**（各就各位，谁也不冒充谁）：
#   MoE 路由：**score 系已收编**（softmax/sigmoid/softplus/sqrtsoftplus，
#             委托 `by1ir.moe_scores_array`）；还缺 topk_ids /
#             expert_matmul（ggml_mul_mat_id）/ 分组选法那几步
#   GDN/Linear：conv1d_causal / delta_rule（delta-net-base.cpp 三个实现）
#   KDA/SSM：conv1d_causal / 递推扫描（by1kda/by1ssm 已各自验过）
#   MLA/DSA：dsa_index（by1sparse 的判卷人是 GlmMoeDsaIndexer）
# 每条下沉时：公式 + ref() + 反例一起进来，别只搬代码。
UNLOWERED = ("MoE", "MLA", "KDA", "SSM", "Linear", "Raw", "External")


# ════════════════════════════════════════════════════════════════════
# 图容器 —— 值有名有形状；节点有序（顺序即计算顺序）
# ════════════════════════════════════════════════════════════════════

class Graph:
    def __init__(self, name):
        self.name = name
        self.vals = {}          # 值名 -> 形状
        self.nodes = []         # {prim, ins, outs, attrs}

    def add(self, prim, ins, outs, attrs=None, shapes=None):
        if prim not in PRIMS:
            raise ValueError("原语 %r 不在闭集里" % prim)
        n = {"prim": prim, "ins": list(ins), "outs": list(outs),
             "attrs": dict(attrs or {})}
        if shapes:
            for v, s in zip(outs, shapes):
                self.vals[v] = tuple(s)
        self.nodes.append(n)
        return outs[0] if outs else None

    def to_json(self):
        """**规范化**（键排序），同图必同字节 —— golden 就比这个。"""
        return json.dumps({"name": self.name,
                           "vals": {k: list(v) for k, v in
                                    sorted(self.vals.items())},
                           "nodes": self.nodes},
                          ensure_ascii=False, sort_keys=True, indent=1)

    def sha(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


def _qk_on(v):
    if v is None or v is False:
        return False
    return str(v).strip().lower() not in ("off", "", "none", "false", "0")


# ════════════════════════════════════════════════════════════════════
# lowering：IR op -> 原语图。**降不动就报错**（拒绝纪律）。
# ════════════════════════════════════════════════════════════════════

def lower_op(g, i, j, o, d, pname, refmap):
    """把 IR 的一个算子降成原语节点，追加进 g。

    `pname`：短名 -> **真参数键**（`layers.0.op1.wq` 这种）。图里的权重
    叶子必须是真名 —— 局部名（L0O1_wq）在求值器里根本找不到
    （第一版就是这么断的）。中间值才用 p 前缀防撞。
    `refmap`：值引用 -> 产出它的节点名（`opK.out` / `normed` 都在里面）——
    **不靠名字模式匹配**：模式匹配在 globals 链上当场断过一次。"""
    a, k = o["attrs"], o["kind"]
    p = "L%dO%d_" % (i, j)

    def _src(r):
        if r == "hidden":
            return "x"
        if r not in refmap:
            raise SystemExit("  [FAIL] %s 引用了没产出的值 %r" % (p, r))
        return refmap[r]
    src = [_src(r) for r in o["inputs"]]
    dst = p + o["outputs"][0].replace(".", "_")
    for r in o["outputs"]:
        refmap[r] = dst
    w = lambda nm: pname(nm)

    if k not in ("Embed", "Norm", "Head", "Attention", "FFN", "Add", "MoE"):
        raise SystemExit("  [不支持] %s 还没有 lowering（原语未收编，"
                         "见 by1graph.UNLOWERED）" % k)

    if k == "Embed":
        g.add("get_rows", [w("embed.weight"), "tok"], [dst],
              shapes=[(d, None)])
    elif k == "Add":
        g.add("add", [src[0], src[1]], [dst], shapes=[(d, None)])
    elif k == "Head":
        t = g.add("matmul", [w("head.weight"), src[0]], [p + "logits0"],
                  shapes=[(None, None)])
        # 保持 IR 的输出名
        g.nodes[-1]["outs"] = [dst]
    elif k == "Norm":
        one = 1 if a.get("one_plus") else 0
        fn = "layer_norm" if str(a.get("kind", "rms")).lower() == "layer" \
            else "rms_norm"
        ins = [src[0], w("w")] + ([w("b")] if fn == "layer_norm" else [])
        g.add(fn, ins, [dst],
              attrs={"eps": float(a.get("eps", _DEFAULT_EPS)),
                     "one_plus": bool(one)}, shapes=[(d, None)])
    elif k == "FFN":
        hid = int(a["hidden"])
        bias = bool(a.get("bias"))
        gate = bool(a.get("gate", True))
        act = {"gelu_new": "gelu_tanh", "gelu_tanh": "gelu_tanh",
               "tanh": "gelu_tanh", "gelu": "gelu",
               "gelu_erf": "gelu"}.get(str(a.get("act", "silu")), "silu")
        if act == "silu" and str(a.get("act", "silu")) != "silu":
            raise SystemExit("  [不支持] FFN.act = %r 还没有原语"
                             % a.get("act"))
        hb = g.add("matmul", [w("w1"), src[0]], [p + "hb0"],
                   shapes=[(hid, None)])
        if bias:
            hb = g.add("add", [hb, w("w1.bias")], [p + "hb1"],
                       shapes=[(hid, None)])
        if gate:
            ob = g.add("matmul", [w("w3"), src[0]], [p + "ob0"],
                       shapes=[(hid, None)])
            if bias:
                ob = g.add("add", [ob, w("w3.bias")], [p + "ob1"],
                           shapes=[(hid, None)])
            hb = g.add(act, [hb], [p + "hb2"], shapes=[(hid, None)])
            hb = g.add("mul", [hb, ob], [p + "hb3"], shapes=[(hid, None)])
        else:
            hb = g.add(act, [hb], [p + "hb2"], shapes=[(hid, None)])
        out = g.add("matmul", [w("w2"), hb], [p + "o0"], shapes=[(d, None)])
        if bias:
            out = g.add("add", [out, w("w2.bias")], [dst], shapes=[(d, None)])
        else:
            g.nodes[-1]["outs"] = [dst]
    elif k == "Attention":
        _lower_attn(g, p, src[0], dst, w, a, d)
    elif k == "MoE":
        _lower_moe(g, p, src[0], dst, w, a, d)
    return dst


def _lower_moe(g, p, x, dst, w, a, d):
    """MoE 的 lowering —— 语义照 `by1exec.op_moe`（e2e 的判据就是它）。

    **这一刀只切 `softmax_topk`**（Mixtral / Qwen3 系 / Laguna /
    gemma-4-26B，普查里 11 份，且全是 `act=silu`、无 `expert_bias`）。
    另外三种选法**明确拒绝**：

        sigmoid_topk        3 份（Step-3.7）—— 先打分再直接 top-k
        sigmoid_group_topk  6 份（GLM / Ling / Nemotron）—— noaux_tc，
                            要"每组前 2 之和排名选组"，还得加分组原语
        topk_softmax        3 份（gpt-oss）—— 要 gptoss 激活与 expert_bias

    `softmax_topk` 的三步（照 by1exec 逐句）：

        sel  = logits + score_bias          # **只影响选择**
        topi = top_k(softmax(sel))          # 先全体 softmax 再选
        base = logits if score_bias else sel
        tv   = gather(softmax(base), topi)  # **权重用未加偏置的 logits 重算**
        topv = tv / max(sum(tv), 1e-9) * routed_scale
        out  = Σ_k topv[:,k] · Expert_k(x)
    """
    rt = a.get("routing") or "softmax_topk"
    if rt != "softmax_topk":
        raise SystemExit("  [不支持] MoE.routing=%s 还没有 lowering"
                         "（这一刀只切 softmax_topk）" % rt)
    if a.get("expert_bias"):
        raise SystemExit("  [不支持] MoE.expert_bias 还没有 lowering")
    if str(a.get("act", "silu")) != "silu":
        raise SystemExit("  [不支持] MoE.act=%r 还没有 lowering" % a.get("act"))
    E, K, H = int(a["experts"]), int(a["top_k"]), int(a["hidden"])
    sb_on = bool(a.get("score_bias"))
    # ── ① 路由 logits = x @ routerᵀ（+ bias） ───────────────────────
    lg = g.add("matmul", [w("router"), x], [p + "lg"], shapes=[(E, None)])
    if a.get("router_bias"):
        lg = g.add("add", [lg, w("router.bias")], [p + "lg1"],
                   shapes=[(E, None)])
    sel = lg
    if sb_on:
        sel = g.add("add", [lg, w("score_bias")], [p + "sel"],
                    shapes=[(E, None)])
    sm = g.add("softmax", [sel], [p + "sm"], shapes=[(E, None)])
    topi = g.add("top_k", [sm], [p + "topi"], attrs={"k": K},
                 shapes=[(K, None)])
    if sb_on:
        # **权重用未加偏置的 logits 重算**（by1exec 的 `base`）
        sm2 = g.add("softmax", [lg], [p + "sm2"], shapes=[(E, None)])
        tv = g.add("gather", [sm2, topi], [p + "tv"], shapes=[(K, None)])
    else:
        tv = g.add("gather", [sm, topi], [p + "tv"], shapes=[(K, None)])
    den = g.add("sum", [tv], [p + "den"], attrs={"axis": 0, "keepdims": True},
                shapes=[(1, None)])
    den = g.add("clamp_min", [den], [p + "den1"], attrs={"min": 1e-9},
                shapes=[(1, None)])
    topv = g.add("div", [tv, den], [p + "topv"], shapes=[(K, None)])
    if float(a.get("routed_scale", 1.0)) != 1.0:
        topv = g.add("scale", [topv], [p + "topv1"],
                     attrs={"s": float(a["routed_scale"])}, shapes=[(K, None)])
    # ── ② 逐专家 SwiGLU ───────────────────────────────────────────
    # 专家权重按下标**整行取**（`get_rows` 按第 0 轴）：(E,out,in) + ids(n,K)
    # -> (n,K,out,in)。输入广播成 (n,K,1,in)，`matmul` 的 `x @ wᵀ` 于是给出
    # (n,K,1,out) —— 再把那个 1 挤掉。**能拼出来就不加原语。**
    # ⚠ **两个矩阵乘的收缩维不同**：w1/w3 收 `d`，w2 收 `H`
    # （第一版把 x 直接喂给 w2，于是 (n,1,128) 去乘 (n,K,256,d) —— 当场报
    #  "size 2 is different from 128"）。
    # ⚠ **x 要 view 成 4 维 `(n,1,1,d)`**：numpy 的 matmul 批维**从右对齐**，
    # (n,1,d) 的批维是 `(n,)`，和 (n,K) 对不齐（1 vs K 在错的位置上）。
    # 4 维之后批维是 `(n,1)`，与 `(n,K)` 广播出 `(n,K)` —— 正好。
    xr = g.add("view", [x], [p + "xr"], attrs={"shape": [d, 1, 1, None]},
               shapes=[(d, 1, 1, None)])

    def _expert(wname, inp, out_w, in_w):
        gv = g.add("get_rows", [w(wname), topi], [p + wname + "_g"],
                   shapes=[(in_w, out_w, K, None)])
        mm = g.add("matmul", [gv, inp], [p + wname + "_m"])
        return g.add("view", [mm], [p + wname + "_v"],
                     attrs={"shape": [out_w, K, None]},
                     shapes=[(out_w, K, None)])
    hb = _expert("w1", xr, H, d)
    ob = _expert("w3", xr, H, d)
    hb = g.add("silu", [hb], [p + "hb"], shapes=[(H, K, None)])
    acc = g.add("mul", [hb, ob], [p + "acc0"], shapes=[(H, K, None)])
    acc4 = g.add("view", [acc], [p + "acc4"],
                 attrs={"shape": [H, 1, K, None]}, shapes=[(H, 1, K, None)])
    ye = _expert("w2", acc4, d, H)      # (n,K,d)
    # 加权：topv (n,K) -> (n,K,1)，乘完沿 K 求和
    pv = g.add("view", [topv], [p + "pv"], attrs={"shape": [1, K, None]},
               shapes=[(1, K, None)])
    wy = g.add("mul", [ye, pv], [p + "wy"], shapes=[(d, K, None)])
    s = g.add("sum", [wy], [p + "s"], attrs={"axis": 1, "keepdims": True},
              shapes=[(1, d, None)])
    # ⚠ **残差不在这个算子里** —— `by1exec.op_moe` 返回的就是专家和
    # （`out = zeros_like(xf)` 起手），残差是外面那个独立的 `Add` 算子。
    # 第一版在这里 `add(x, s)`，于是残差被加了两次（实测 7.756e-01）。
    out = g.add("view", [s], [p + "s2"], attrs={"shape": [d, None]},
                shapes=[(d, None)])
    # ── ③ 共享专家（Mixtral 没有，Qwen3 / GLM 有） ─────────────────
    if a.get("shared"):
        sh = g.add("matmul", [w("sw1"), x], [p + "sh1"], shapes=[(None, None)])
        sh3 = g.add("matmul", [w("sw3"), x], [p + "sh3"], shapes=[(None, None)])
        sh = g.add("silu", [sh], [p + "sh2"], shapes=[(None, None)])
        se = g.add("mul", [sh, sh3], [p + "sh4"], shapes=[(None, None)])
        se = g.add("matmul", [w("sw2"), se], [p + "se"], shapes=[(d, None)])
        if a.get("shared_gate"):
            gs = g.add("matmul", [w("shared_gate"), x], [p + "gs"],
                       shapes=[(1, None)])
            gs = g.add("sigmoid", [gs], [p + "gs1"], shapes=[(1, None)])
            se = g.add("mul", [se, gs], [p + "se1"], shapes=[(d, None)])
        out = g.add("add", [out, se], [p + "out2"], shapes=[(d, None)])
    g.nodes[-1]["outs"] = [dst]
    return dst


def _lower_attn(g, p, x, dst, w, a, d):
    hd, nh, nkv = int(a["head_dim"]), int(a["q"]), int(a["kv"])
    if int(a["out_dim"]) != nh * hd:
        raise SystemExit("  [不支持] out_dim != q*head_dim 的注意力"
                         "（输出重排还没收编）")
    for flag, what in (("q_gate", "q_gate"), ("head_gate", "head_gate")):
        if a.get(flag) and (flag != "head_gate" or a.get(flag) != "off"):
            raise SystemExit("  [不支持] Attention.%s 还没有 lowering" % what)
    bias = bool(a.get("bias"))
    # **融合 q/k/v 只是布局**：`by1exec` 的做法是把它拆回 wq/wk/wv/wo，
    # 「后面一格不动」—— 这里同样拆，拆的是**权重叶子**（`slice` 原语）。
    # 权重在 exec 视角是 (out, in)，节点 attr 按 ggml 视角写（[in, out]），
    # 所以**输出那条轴是 axis 1**；一维的 bias 两条视角同轴，是 axis 0。
    if a.get("qkv_fused"):
        if a.get("kv_tie"):
            raise SystemExit("  [不支持] qkv_fused 与 kv_tie 同时出现")
        _qa, _kb = nh * hd, nkv * hd

        def _cut(src, s0, s1, dst, ax):
            return g.add("slice", [w(src)], [p + dst],
                         attrs={"axis": ax, "start": s0, "stop": s1})
        WQ = _cut("qkv", 0, _qa, "wqs", 1)
        WK = _cut("qkv", _qa, _qa + _kb, "wks", 1)
        WV = _cut("qkv", _qa + _kb, _qa + 2 * _kb, "wvs", 1)
        WO = w("proj")
        BQ = _cut("qkv.bias", 0, _qa, "wqbs", 0) if bias else None
        BK = _cut("qkv.bias", _qa, _qa + _kb, "wkbs", 0) if bias else None
        BV = (_cut("qkv.bias", _qa + _kb, _qa + 2 * _kb, "wvbs", 0)
              if bias else None)
        BO = w("proj.bias") if bias else None
    else:
        WQ, WK, WV, WO = w("wq"), w("wk"), w("wv"), w("wo")
        BQ = w("wq.bias") if bias else None
        BK = w("wk.bias") if bias else None
        BV = w("wv.bias") if bias else None
        BO = w("wo.bias") if bias else None
    q = g.add("matmul", [WQ, x], [p + "q0"], shapes=[(nh * hd, None)])
    kq_ = g.add("matmul", [WK, x], [p + "k0"], shapes=[(nkv * hd, None)])
    if bias:
        q = g.add("add", [q, BQ], [p + "q1"], shapes=[(nh * hd, None)])
        kq_ = g.add("add", [kq_, BK], [p + "k1"], shapes=[(nkv * hd, None)])
    # **kv_tie：v 是 k 的原始投影** —— 分叉在 norm/rope 之前（判卷人
    # by1exec：`v = kraw`）。
    if a.get("kv_tie"):
        v = g.add("view", [kq_], [p + "v3"],
                  attrs={"shape": [hd, nkv, None]}, shapes=[(hd, nkv, None)])
    else:
        v = g.add("matmul", [WV, x], [p + "v0"], shapes=[(nkv * hd, None)])
        if bias:
            v = g.add("add", [v, BV], [p + "v1"],
                      shapes=[(nkv * hd, None)])
        v = g.add("view", [v], [p + "v3"], attrs={"shape": [hd, nkv, None]},
                  shapes=[(hd, nkv, None)])
    q3 = g.add("view", [q], [p + "q3"], attrs={"shape": [hd, nh, None]},
               shapes=[(hd, nh, None)])
    k3 = g.add("view", [kq_], [p + "k3"], attrs={"shape": [hd, nkv, None]},
               shapes=[(hd, nkv, None)])
    if _qk_on(a.get("qk_norm")):
        if str(a.get("qk_norm")).lower() != "per_head":
            raise SystemExit("  [不支持] qk_norm=full 还没有 lowering")
        eps = float(a.get("norm_eps", _DEFAULT_EPS))
        # **norm 的公式自带乘 w**（PRIMS["rms_norm"] 就是那一条）——
        # 不是 norm 一个节点、mul 一个节点：两个节点就是两份语义。
        q3 = g.add("rms_norm", [q3, w("qn.w")], [p + "q3n"],
                   attrs={"eps": eps, "one_plus": False},
                   shapes=[(hd, nh, None)])
        k3 = g.add("rms_norm", [k3, w("kn.w")], [p + "k3n"],
                   attrs={"eps": eps, "one_plus": False},
                   shapes=[(hd, nkv, None)])
    if a.get("rope", True):
        nrot = int(hd * float(a.get("rope_partial", 1.0)))
        rp = {"base": float(a["rope_base"]),
              "pairing": str(a.get("rope_pairing", "half")),
              "n_dims": nrot,
              # **YaRN / llama3 与 rope_scale 都走这一条**（`rope_inv` 一处定义）
              "yarn": a.get("yarn"),
              "rope_scale": float(a.get("rope_scale", 1.0))}
        q3 = g.add("rope", [q3, "pos"], [p + "q3r"], attrs=rp,
                   shapes=[(hd, nh, None)])
        k3 = g.add("rope", [k3, "pos"], [p + "k3r"], attrs=rp,
                   shapes=[(hd, nkv, None)])
    # 进 mul_mat 前换轴：[hd, 头, token] -> [hd, token, 头]
    q3 = g.add("permute", [q3], [p + "q3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nh)])
    k3 = g.add("permute", [k3], [p + "k3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nkv)])
    kq = g.add("batch_matmul", [k3, q3], [p + "kq"],
               shapes=[(None, None, nh)])
    win = int(a.get("window") or 0)
    _sc = 1.0 / (hd ** 0.5)
    if a.get("sink"):
        # **attention sink**：softmax 的 key 维尾巴上多一个槽
        # （by1exec 的 `concat([att, sk], -1)` -> softmax -> `[..., :-1]`）。
        # 它让权重**不再和为 1** —— 所以不能拿普通 soft_max 冒充。
        kq = g.add("soft_max_sink", [kq, p + "mask_%d" % win, w("sink")],
                   [p + "sm"], attrs={"scale": _sc},
                   shapes=[(None, None, nh)])
    else:
        kq = g.add("soft_max", [kq, p + "mask_%d" % win], [p + "sm"],
                   attrs={"scale": _sc}, shapes=[(None, None, nh)])
    # v^T：[hd, 头, token] -> [token, hd, 头]
    vt = g.add("permute", [v], [p + "v3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nkv)])
    vt = g.add("cont", [g.add("permute", [vt], [p + "v3t"],
                              attrs={"axes": [1, 0, 2]},
                              shapes=[(None, hd, nkv)])],
               [p + "v3c"], shapes=[(None, hd, nkv)])
    kqv = g.add("batch_matmul", [vt, p + "sm"], [p + "kqv"],
                shapes=[(hd, None, nh)])
    att = g.add("cont", [g.add("permute", [kqv], [p + "kqvp"],
                               attrs={"axes": [0, 2, 1]},
                               shapes=[(hd, nh, None)])],
                [p + "att"], attrs={"shape": [nh * hd, None]},
                shapes=[(nh * hd, None)])
    out = g.add("matmul", [WO, att], [p + "o0"], shapes=[(d, None)])
    if bias:
        out = g.add("add", [out, BO], [dst], shapes=[(d, None)])
    else:
        g.nodes[-1]["outs"] = [dst]


def lower_model(ir, name="model"):
    """整份 IR -> 一张图。globals 在层前后各一截（ggml 惰性图那一课：
    最终 norm/head **必须在层之后**接线）。"""
    d = ir["d_model"]
    g = Graph(name)
    g.vals["tok"] = (None,)
    g.vals["pos"] = (None,)
    g.vals["x"] = (d, None)
    pre = [o for o in ir["globals"] if o["kind"] == "Embed"]
    post = [o for o in ir["globals"] if o["kind"] != "Embed"]
    # Embed 的输出就是 `x`（残差流起点）
    for o in pre:
        g.add("get_rows", ["embed.weight", "tok"], ["x"], shapes=[(d, None)])
    # **学习式位置表**（GPT-2 那一支）：embed 之后一次加法。
    # 语义照 `by1exec`（`x = x + pos.weight[:n]`）—— 求值器给的
    # `pos` 就是 `arange(n)`，所以 `get_rows(pos.weight, pos)` 与
    # `pos.weight[:n]` 逐位相同。**这一条以前是直接拒绝的**
    # （"还没收编"），于是 gpt2-tiny 降不了 —— 而它是 by1ggml 自检的
    # ② 号判据（layernorm / 学习位置 / gelu_new / 无门控）。
    if str(ir.get("pos_kind", "rope")).lower() == "learned":
        pe = g.add("get_rows", ["pos.weight", "pos"], ["pos_emb"],
                   shapes=[(d, None)])
        g.add("add", ["x", pe], ["x"], shapes=[(d, None)])
    for L in ir["layers"]:
        i = L["index"]
        din = L.get("in_dim") or d
        prev = None
        refmap = {}
        for j, o in enumerate(L["ops"]):
            # 层内值名重绑定：hidden -> x；权重叶子用真参数键
            pname = (lambda nm, i=i, j=j:
                     "layers.%d.op%d.%s" % (i, j, nm))
            dst = lower_op(g, i, j, o, din, pname, refmap)
            prev = dst
        # 层输出 -> 下一层的 hidden
        g.add("cont", [prev], ["x"], shapes=[(din, None)])
    gmap = {"Norm": {"w": "final_norm.w", "b": "final_norm.b"},
            "Head": {"head.weight": "head.weight"}}
    refmap = {}
    for j, o in enumerate(post):
        m = gmap[o["kind"]]
        lower_op(g, len(ir["layers"]) + 1, j, o, d,
                 lambda nm, m=m: m[nm], refmap)
    return g


# ════════════════════════════════════════════════════════════════════
# 图求值器 —— **唯一语义 = 现有语义** 的判卷人（A3 的第一块砖）。
# 用 PRIMS 的参考实现把图跑出来，和 by1exec 逐位对拍。
# 在这之前"图是对的"只是一句话；在这之后它是一个数。
# ════════════════════════════════════════════════════════════════════

def _exec_view(x, gshape):
    """ggml 视角的 view（特征在前）-> by1exec 视角（样本在前）。

    ggml [d0, d1, n] 的内存 == 行主序 [n, d1, d0] —— **同一块内存的
    两种读法**，翻转形状就是全部。"""
    import numpy as np
    t = tuple(reversed([s if s is not None else -1 for s in gshape]))
    return np.reshape(x, t)


def _exec_permute(x, axes):
    """ggml_permute（输出轴 j = 输入轴 axes[j]，ggml 视角）-> exec transpose。"""
    import numpy as np
    nd = len(axes)
    # exec 轴 = ggml 轴倒序。输出 exec 轴 k 对应 ggml 输出轴 nd-1-k
    # = ggml 输入轴 axes[nd-1-k] = exec 输入轴 nd-1-axes[nd-1-k]。
    tp = tuple(nd - 1 - axes[nd - 1 - k] for k in range(nd))
    return np.transpose(x, tp)


def run_graph(g, params, ids):
    """按图求值（by1exec 约定：x [n, d]）。判卷人是 by1exec 自己。"""
    import numpy as np
    V = {}
    for k, v in params.items():
        V[k.replace(".", "_")] = v
    V["tok"] = np.asarray(ids)
    V["pos"] = np.arange(len(ids))
    for n in g.nodes:
        prim, ins, outs, at = n["prim"], n["ins"], n["outs"], n["attrs"]
        # mask_* 是**叶子**：值表里没有，soft_max 分支按名字里的
        # window 现场构造（掩码只由 n 和 window 决定，确定性）
        a = [None if "mask_" in i else V[i.replace(".", "_")] for i in ins]
        if prim == "get_rows":
            out = a[0][a[1]]
        elif prim == "add":
            out = a[0] + a[1]
        elif prim == "mul":
            out = a[0] * a[1]
        elif prim == "scale":
            out = a[0] * at["s"]
        elif prim in ("matmul", "batch_matmul"):
            out = PRIMS[prim]["ref"](a[0], a[1])
        elif prim == "rms_norm":
            out = PRIMS["rms_norm"]["ref"](
                a[0], a[1], at["eps"], bool(at.get("one_plus")))
        elif prim == "layer_norm":
            out = PRIMS["layer_norm"]["ref"](
                a[0], a[1], a[2], at["eps"], bool(at.get("one_plus")))
        elif prim in ("silu", "gelu_tanh", "gelu", "sigmoid",
                      "softmax", "softplus", "sqrtsoftplus"):
            out = PRIMS[prim]["ref"](a[0])
        elif prim == "div":
            out = PRIMS["div"]["ref"](a[0], a[1])
        elif prim == "top_k":
            out = PRIMS["top_k"]["ref"](a[0], at["k"])
        elif prim == "gather":
            out = PRIMS["gather"]["ref"](a[0], a[1])
        elif prim == "sum":
            out = PRIMS["sum"]["ref"](a[0], at["axis"],
                                      bool(at.get("keepdims", True)))
        elif prim == "clamp_min":
            out = PRIMS["clamp_min"]["ref"](a[0], at["min"])
        elif prim == "rope":
            # [n, nh, hd]：位置在轴 -2 —— 转成 [..., n, hd] 再转回
            x = np.transpose(a[0], (1, 0, 2))
            out = PRIMS["rope"]["ref"](x, a[1], at["base"], at["pairing"],
                                       at["n_dims"], at.get("yarn"),
                                       at.get("rope_scale", 1.0))
            out = np.transpose(out, (1, 0, 2))
        elif prim == "soft_max":
            # a: [头, query, key]（exec 约定），mask [query, key]
            w = int(n["ins"][1].rsplit("_", 1)[-1])
            m = _mask(a[0].shape[1], w)
            out = PRIMS["soft_max"]["ref"](a[0], m, at["scale"])
        elif prim == "soft_max_sink":
            # 同 soft_max，但第三个输入是逐头的 sink 标量
            w = int(n["ins"][1].rsplit("_", 1)[-1])
            m = _mask(a[0].shape[1], w)
            out = PRIMS["soft_max_sink"]["ref"](a[0], m, a[2], at["scale"])
        elif prim == "view":
            out = _exec_view(a[0], at["shape"])
        elif prim == "permute":
            out = _exec_permute(a[0], at["axes"])
        elif prim == "slice":
            out = PRIMS["slice"]["ref"](a[0], at["axis"], at["start"],
                                        at["stop"])
        elif prim == "cont":
            # cont 不带 shape = 纯物化；带 shape = 物化 + 按 ggml 视角
            # 重排（合头那一步的 cont_2d 就是它）
            out = a[0]
            if at.get("shape"):
                out = _exec_view(out, at["shape"])
            out = np.ascontiguousarray(out)
        else:
            raise SystemExit("  [不支持] 求值器还没有 %s" % prim)
        V[outs[0].replace(".", "_")] = out
    return V


def _mask(n, window):
    """因果 + 滑窗掩码（by1exec 的规则：j <= i 且 j > i - window）。"""
    import numpy as np
    idx = np.arange(n)
    m = idx[None, :] <= idx[:, None]
    if window:
        m = m & (idx[None, :] > idx[:, None] - window)
    return np.where(m, 0.0, -np.inf)


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def param_gb(ir, bir=None):
    """模型的**参数内存足迹**（float32，GB）—— 只读形状，不分配。

    e2e 要给每份模型按全尺寸铺随机权重，**先算足迹再决定跑不跑**。
    golden 里混着真实大模型（`gemma-4-31B` 单份 130.6 GB），
    在 16 GB 的机器上直接分配会打爆页面文件、把系统冻住
    （2026-10-12 实测：这条命令冻了两次机器）。"""
    if bir is None:
        import by1ir as bir
    return sum(_prod(s) for s in bir.params_of(ir).values()) * 4 / 1e9


# e2e 的默认内存预算（GB）。**这不是性能参数，是安全阀**：
# 超过它就跳过并印出来，宁可不跑，不许把机器打死。
# `--mem-gb N` 显式覆盖（换一台大内存机器时才该抬它）。
E2E_MEM_GB = 4.0


def e2e(models=None, mem_gb=None):
    """**图求值 vs by1exec**：同一批随机权重、同一条输入，逐位对拍。
    这是"lowering 降对了"和"唯一语义真的是唯一"的判卷人。

    **按参数足迹过预算**：超预算的模型**显式跳过**（印出需多少 GB），
    不静默丢 —— 静默跳过在这个仓库是记过案的那一类（`ir.md 133`）。"""
    import importlib.util
    import numpy as np
    sp = importlib.util.spec_from_file_location(
        "bcm", by1paths.tool("by1check.py"))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    import by1codegen as cg
    import by1ir as bir
    ex = _load("by1exec")
    budget = E2E_MEM_GB if mem_gb is None else float(mem_gb)
    names = models or sorted(by1io.read_json(GOLDEN).keys())
    bad, skipped, ran, unlowered = [], [], [], []
    for f in names:
        path = _os.path.join(by1paths.MODELS, f)
        try:
            _r, info = bc.check(path)
            ir = cg.compile_ir(info)
        except Exception as e:
            unlowered.append((f, "%s: %s" % (type(e).__name__, e)))
            continue
        try:
            g = lower_model(ir, f)
        except SystemExit as e:
            unlowered.append((f, str(e).strip().splitlines()[0]))
            continue
        gb = param_gb(ir, bir)
        if gb > budget:
            skipped.append((f, gb))
            print("     - %-28s 跳过：需 %.2f GB > 预算 %.2f GB"
                  % (f, gb, budget))
            continue
        rng = np.random.default_rng(0)
        params = {k: rng.normal(0, 0.1, s).astype(np.float32)
                  for k, s in bir.params_of(ir).items()}
        ids = rng.integers(0, ir["vocab"], 16).astype(np.int32)
        e = ex.Exec(ir, params)
        e.P["_g"] = {k: params[k] for k in
                     ("embed.weight", "pos.weight", "final_norm.w",
                      "final_norm.b", "head.weight") if k in params}
        want = e(ids[None, :])[0]
        V = run_graph(g, params, ids)
        got = next(v for k, v in V.items() if k.endswith("_logits"))
        dd = np.abs(got - want).max()
        amp = max(np.abs(want).max(), 1e-9)
        if dd / amp >= 1e-4:
            bad.append("%s 相对差 %.3e" % (f, dd / amp))
        else:
            ran.append(f)
            print("     ✓ %-28s 相对差 %.3e（%.2f GB）"
                  % (f, dd / amp, gb))
    tail = ""
    if skipped:
        tail = "（%d 份超预算跳过，合计需 %.2f GB；`--mem-gb N` 可抬预算）" % (
            len(skipped), sum(g for _f, g in skipped))
    # **降不动也要说，而且算失败**：golden 里列的模型降不动，说明快照
    # 过期了；显式点名的那几个更不用说。**静默丢弃会让"我验了 A 和 B"
    # 悄悄变成"我验了 A"** —— 和 `ir.md 133` 那个"32 个只验了 30 个"
    # 是同一族。第一版就是这样：点名的 `gpt-oss-shaped` 被吞掉了。
    for f, why in unlowered:
        print("  [FAIL] %s 降不动：%s" % (f, why))
    bad.extend(f for f, _w in unlowered)
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        print("  [FAIL] 图求值 vs by1exec：%d 处%s" % (len(bad), tail))
        return 1
    if not ran:
        print("  [FAIL] 图求值 vs by1exec：0 份跑成%s —— 预算太小，"
              "等于没验" % tail)
        return 1
    print("  [PASS] 图求值 vs by1exec：%d 份模型同权重同输入一致%s"
          "（lowering 降对了，唯一语义是真的唯一）" % (len(ran), tail))
    return 0


# ════════════════════════════════════════════════════════════════════
# golden 快照 + 自检
# ════════════════════════════════════════════════════════════════════

GOLDEN = by1paths.root("graphs-golden.json")


def golden(write=False):
    """每个可降模型的图 sha256。**逐字节稳定**是 A3 的护栏：
    后端机械化重写前后，图必须一模一样。"""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "bcm", by1paths.tool("by1check.py"))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    import by1codegen as cg
    got, skipped = {}, []
    for f in sorted(_os.listdir(by1paths.MODELS)):
        if not f.endswith(".by1") or f.endswith("selftest.by1"):
            continue
        path = _os.path.join(by1paths.MODELS, f)
        try:
            _r, info = bc.check(path)
            ir = cg.compile_ir(info)
        except Exception:
            skipped.append(f)
            continue
        try:
            got[f] = lower_model(ir, f).sha()
        except SystemExit:
            skipped.append(f)
    if write:
        by1io.write_text(GOLDEN, json.dumps(got, ensure_ascii=False,
                                            indent=1, sort_keys=True) + "\n")
        print("  写了 %s：%d 张图" % (by1paths.rel(GOLDEN), len(got)))
        print("  （降不动的 %d 份跳过：%s）" % (len(skipped), ", ".join(skipped[:6])))
        return 0
    want = by1io.read_json(GOLDEN) if _os.path.exists(GOLDEN) else {}
    bad = [k for k in got if want.get(k) != got[k]]
    if bad:
        for k in bad[:6]:
            print("  [FAIL] %s 图变了：golden %s != 现在 %s"
                  % (k, want.get(k), got[k]))
        return 1
    print("  [PASS] golden：%d 张图逐字节稳定（%d 份降不动的在账上）"
          % (len(got), len(skipped)))
    return 0


def selftest():
    """参考实现对手算 —— **语义的公式和执行是一致的**（B1 的第一块砖）。"""
    import numpy as np
    bad = []
    x = np.array([3.0, 4.0])
    r = PRIMS["rms_norm"]["ref"](x, np.ones(2), 0.0, False)
    s = 1.0 / np.sqrt((9.0 + 16.0) / 2)
    if abs(r[0] - 3 * s) > 1e-6 or abs(r[1] - 4 * s) > 1e-6:
        bad.append("rms_norm 对不上手算")
    w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # (out=3, in=2)
    xv = np.array([[1.0, 1.0]])                          # (n=1, in=2)
    y = PRIMS["matmul"]["ref"](w, xv)
    if not np.allclose(y, [[3.0, 7.0, 11.0]]):
        bad.append("matmul 对不上手算（ggml-smoke 的同一批数）")
    z = PRIMS["silu"]["ref"](np.array([0.0, 2.0]))
    if abs(z[0]) > 1e-6 or abs(z[1] - 2.0 / (1.0 + np.exp(-2.0))) > 1e-6:
        bad.append("silu 对不上手算")
    # rope：位置 0 不动；half 与 interleaved 的**输入配对不同**
    # （i,i+2 vs 2i,2i+1），各自对公式。
    xr = np.zeros((1, 4))          # [n=1, hd=4]
    xr[0] = [1.0, 0.0, 0.0, 1.0]
    rr = PRIMS["rope"]["ref"](xr, np.array([0.0]), 10000.0, "half", 4)
    if not np.allclose(rr, xr):
        bad.append("rope 位置 0 应当不动")
    # ── 打分函数那一族（MoE 路由的第一步）── 手算 + **与 by1ir 同源** ──
    for k, want in (("softplus", np.log(2.0)),
                    ("sqrtsoftplus", np.sqrt(np.log(2.0))),
                    ("sigmoid", 0.5)):
        z = PRIMS[k]["ref"](np.array([0.0]))
        if abs(float(z[0]) - want) > 1e-9:
            bad.append("%s(0) 对不上手算（应 %r）" % (k, want))
    sm = PRIMS["softmax"]["ref"](np.array([[0.0, 0.0], [1.0, 1.0]]))
    if not np.allclose(sm, 0.5):
        bad.append("softmax 对等值应当均匀")
    from by1ir import moe_scores_array
    _xs = np.array([[-3.0, 0.5, 2.0]])
    for k in ("softmax", "sigmoid", "softplus", "sqrtsoftplus"):
        if not np.array_equal(PRIMS[k]["ref"](_xs),
                              moe_scores_array(k, _xs, np)):
            bad.append("打分函数 %s 与 by1ir 那份**不同源**" % k)
    # ── 路由 / 归约那一族（MoE 的选专家与加权求和）── 手算 ──
    _v = np.array([[1.0, 3.0, 2.0]])
    if not np.array_equal(PRIMS["top_k"]["ref"](_v, 2), [[1, 2]]):
        bad.append("top_k 对不上手算（[1,3,2] 的前 2 大是下标 1,2）")
    if not np.array_equal(PRIMS["gather"]["ref"](_v, np.array([[0, 2]])),
                          [[1.0, 2.0]]):
        bad.append("gather 对不上手算")
    if float(PRIMS["sum"]["ref"](_v, 0)[0][0]) != 6.0:
        bad.append("sum（ggml 轴 0 = 最后轴）对不上手算")
    if not np.array_equal(PRIMS["clamp_min"]["ref"](np.array([-1.0, 2.0]), 0.5),
                          [0.5, 2.0]):
        bad.append("clamp_min 对不上手算")
    if not np.allclose(PRIMS["div"]["ref"](np.array([4.0, 6.0]),
                                           np.array([2.0, 3.0])), [2.0, 2.0]):
        bad.append("div 对不上手算")
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    print("  [PASS] 原语参考实现对拍手算（rms_norm / matmul / silu / rope · "
          "打分族 softmax/softplus/sqrtsoftplus/sigmoid 并与 by1ir 同源 · "
          "路由归约 top_k/gather/sum/clamp_min/div）")
    return 0


def _load(name):
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, by1paths.tool(name + ".py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main(argv=None):
    argv = argv if argv is not None else _sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    if "--golden" in argv:
        return golden(write="--write" in argv)
    if "--e2e" in argv:
        mem, rest, i = None, [], 0
        while i < len(argv):
            if argv[i] == "--mem-gb":
                if i + 1 >= len(argv):
                    print("  [FAIL] --mem-gb 缺参数")
                    return 2
                mem = float(argv[i + 1])
                i += 2
                continue
            if argv[i] != "--e2e":
                rest.append(argv[i])
            i += 1
        return e2e(models=rest or None, mem_gb=mem)
    if not argv:
        print(__doc__)
        return 0
    bc, cg = _load("by1check"), _load("by1codegen")
    _r, info = bc.check(argv[0])
    ir = cg.compile_ir(info)
    g = lower_model(ir, _os.path.basename(argv[0]))
    print("  %d 个节点，%d 个值，sha %s" % (len(g.nodes), len(g.vals), g.sha()))
    for n in g.nodes[:12]:
        print("    %-10s %s -> %s" % (n["prim"], n["ins"], n["outs"]))
    if len(g.nodes) > 12:
        print("    ...（共 %d 个）" % len(g.nodes))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
