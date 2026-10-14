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


def _ref_clamp_max():
    """`min(x, m)` —— `clamp_min` 的对称面。gptoss 激活的 limit 钳制就是它
    （`by1exec.op_moe` 的 `np.minimum(g, lim)` / `np.clip(u, -lim, lim)`）。"""
    np = _np()
    return lambda x, m: np.minimum(x, float(m))


def _ref_add_scalar():
    """`a + c`（c 是 attr）。**不是 clamp**：sigmoid 那两支的归一守卫是
    `tv / (Σtv + 1e-20)`（`by1exec.op_moe` 612 / 604），和 softmax_topk 的
    `max(Σtv, 1e-9)` 是两个公式 —— `+` 和 `max` 在 Σ < c 时不一样，
    描述层不许合并成一条。"""
    np = _np()
    return lambda a, c: a + float(c)


def _ref_one_hot():
    """把下标行散成 0/1 行：`out[..., j] = 1` 若 `j ∈ ids[..., :]`，否则 0。

    **宽度由第二个输入（参照）的最后一维给** —— 它可能是静态的（MoE 的
    组数）也可能是运行期的 n（DSA 的 key 全集），attr 表达不了运行期。
    并列/重复的下标就置 1 两次（无害）。"""
    np = _np()

    def f(ids, ref):
        ids = np.asarray(ids).astype(np.int64)
        out = np.zeros(ids.shape[:-1] + (int(np.asarray(ref).shape[-1]),),
                       dtype=np.float64)
        np.put_along_axis(out, ids, 1.0, axis=-1)
        return out
    return f


def _ref_masked_fill():
    """`out = m > 0 ? x : v`（v 是 attr，典型是 -inf）——
    `by1exec.op_moe` 的 `np.where(mask > 0, choice, -np.inf)`。"""
    np = _np()
    return lambda x, m, v: np.where(np.asarray(m) > 0, x, float(v))


def _ref_le():
    """`out = 1` 若 `a <= b` 否则 0（广播二元，掩码族的第三个成员）。

    DSA 索引器的因果掩码就是它（`by1exec._dsa_index` 的
    `np.where(ar[None,:] > ar[:,None], -inf, ·)` —— 取反一下就是 `<=`）。
    1/0 而不是 bool：掩码要能进 `masked_fill` 的 `m > 0`，也要能乘。"""
    np = _np()
    return lambda a, b: (np.asarray(a) <= np.asarray(b)).astype(np.float64)


def _ref_div():
    np = _np()
    return lambda a, b: a / b


def _ref_concat():
    """接两条 I/O（沿 **ggml 视角**的 axis）。

    Linear 的 `q|k|v` 拼一边进卷积、GGUF 的融合权重都是它。attr 用 ggml
    视角写（和 slice / sum 同一套），ref 自己换算 exec 轴。"""
    np = _np()

    def f(a, b, axis):
        return np.concatenate([a, b], axis=a.ndim - 1 - int(axis))
    return f


def _ref_conv1d_causal():
    """**深度可分离因果卷积**（GDN 的短卷积）。

    `x` [n, C]（样本在前）、`w` [C, kernel]（逐通道一份小核）：

        xpad = [0, x]                              # 左边补 kernel-1
        out[:, t] = Σ_i w[:, i] · xpad[:, t + i]    # 卷积核按 i 走

    ⚠ 这条公式以前写作 `xpad[:, t + (kernel-1-i)]` —— **和下面的实现
    对不上**（实现是 `mp[:, i:i+s] * w[:, i]`，核没翻转）。判卷人是
    `by1exec`，实现是对的那一个；注释错的那版被 by1c 的图路径照着
    翻成 C，当场差 3e-01 —— **注释也是规格，写错了照样咬人**。

    **激活不在这里** —— by1exec 是 `silu(conv)`，图里 silu 是独立节点：
    一个节点一个公式（两个节点就是两份语义）。
    """
    np = _np()

    def f(x, w, kernel):
        b, s, C = x.shape
        pad = np.zeros((b, int(kernel) - 1, C), dtype=x.dtype)
        mp = np.concatenate([pad, x], 1)
        out = np.zeros((b, s, C), dtype=x.dtype)
        for i in range(int(kernel)):
            out += mp[:, i:i + s, :] * w[:, i][None, None, :]
        return out
    return f


def _ref_delta_rule():
    """**delta 规则**（GDN / KDA 的递推扫描）—— Linear / KDA 的心脏。

    exec 约定：k/q [b, 头, n, dk] · v [b, 头, n, dv] · beta [b, 头, n, 1] ·
    decay [b, 头, n, 1 **或** dk]（GDN 的衰减按头一个、**KDA 的按维** ——
    `[b,h,n,dk]` 乘 `st` 时是逐 d 广播；写错形状**不报错**，只是算错）。

        st = 0                                              # 状态，后端自持
        逐 token t：
            st  *= exp(decay[t])                            # **先衰减**
            kvi  = Σ_d st[·,d,v] · k[t,d]                   # 当前读出来的值
            δ    = (v[t] − kvi) · beta[t]
            st  += k[t] ⊗ δ                                  # 写入
            out[t] = Σ_d st[·,d,v] · q[t,d]                  # 输出读的是**更新后**的

    **状态是一步一步的，没有并行公式** —— 所以它是原语而不是 matmul 的组合
    （by1exec 就写成一个 for）。判卷人是 by1exec 的这个 for 本身。
    `q` 的 `dk^-0.5` 由 lowering 用一个 scale 节点做，不在这个公式里。
    """
    np = _np()

    def f(q, k, v, beta, decay):
        dk, dv = k.shape[-1], v.shape[-1]
        st = np.zeros(k.shape[:-2] + (dk, dv), dtype=np.float64)
        out = np.empty(v.shape, dtype=np.float64)
        q64, k64 = q.astype(np.float64), k.astype(np.float64)
        v64, b64 = v.astype(np.float64), beta.astype(np.float64)
        g64 = decay.astype(np.float64)
        # ⚠ **token 轴是第 2 轴**（b, 头, n, dim）—— `x[:, i]` 切的是**头**。
        # 第一版就写成 `g64[:, i]`：当头数=1 时不越界，于是静默算出别的数
        # （手推 [1.5,-0.5]，它给 [1.5,4.0]）。by1exec 写的是 `kk[:, :, i]`。
        for i in range(k.shape[2]):
            st = st * np.exp(g64[:, :, i])[..., None]              # 先衰减
            kvi = (st * k64[:, :, i][..., None]).sum(-2)           # 读
            delta = (v64[:, :, i] - kvi) * b64[:, :, i]            # δ
            st = st + k64[:, :, i][..., None] * delta[..., None, :]   # 写
            out[:, :, i] = (st * q64[:, :, i][..., None]).sum(-2)   # 输出读更新后的
        return out
    return f


def _ref_exp():
    np = _np()
    return np.exp


def _ref_ssm_scan():
    """**Mamba2 的递推扫描** —— `by1exec.op_ssm` 的 876-883 逐句：

        st = exp(d·A)·st + d·(x ⊗ B)      # 先衰减，再**外积**写入
        y  = st·C + D·x                    # 读出是 **h 右乘 C**

    状态 `[b, 头, hd, ssm]` 是一步一步走的，**没有并行公式** ——
    和 delta_rule 同一个理由：递推就是原语，不是 matmul 的组合。
    内部 float64 累（op_ssm 的 st/out 都是 float64）。

    约定：d `[b,头,n,1]` · A `[头]` · x `[b,头,n,hd]` · B/C `[b,头,n,ssm]`
    · D `[头]` -> y `[b,头,n,hd]`。
    """
    np = _np()

    def f(d, A, x, B, C, D):
        b, h, n = int(d.shape[0]), int(d.shape[1]), int(d.shape[2])
        hd, ssm = int(x.shape[-1]), int(B.shape[-1])
        st = np.zeros((b, h, hd, ssm), dtype=np.float64)
        out = np.empty((b, h, n, hd), dtype=np.float64)
        d64 = d.astype(np.float64)[:, :, :, 0]
        A64 = np.asarray(A, dtype=np.float64)
        D64 = np.asarray(D, dtype=np.float64)
        x64, B64, C64 = (x.astype(np.float64), B.astype(np.float64),
                         C.astype(np.float64))
        for i in range(n):
            dd = d64[:, :, i]
            st = (np.exp(dd[:, :, None, None] * A64[None, :, None, None]) * st
                  + dd[:, :, None, None]
                  * (x64[:, :, i][:, :, :, None] * B64[:, :, i][:, :, None, :]))
            out[:, :, i] = ((st * C64[:, :, i][:, :, None, :]).sum(-1)
                            + D64[None, :, None] * x64[:, :, i])
        return out
    return f


def _ref_log1p():
    np = _np()
    return np.log1p


def _ref_sqrt():
    np = _np()
    return np.sqrt


def _ref_l2_norm():
    """`x / √(Σx² + eps)`，沿**最后轴**（sum，不是 mean）。

    GDN 的 q/k 归一就是这一条（`by1exec.op_linear` 的
    `q * (1/sqrt((q*q).sum(-1) + l2_eps))`）。**注意是 sum**：输出那一支
    用的是 mean，而 `l2_norm(x, vd·ε)` 乘上 `√vd` 正好就是 mean 那一支 ——
    所以图的闭集只需要这一个归一（另一支由 lowering 现拼）。
    """
    np = _np()

    def f(x, eps):
        return x / np.sqrt((x * x).sum(-1, keepdims=True) + float(eps))
    return f


def _ref_repeat():
    """沿 **ggml 视角**的 axis 把每一份重复 reps 次（`np.repeat` 的**整块**重复）。

    GDN 的 GQA 广播要它：q/k 是 nk 个头，状态是 nv 个头，映射是 `头 → 头//rep`
    —— **整块**重复才是 `[q0,q0,q1,q1]`；而"平铺"给出 `[q0,q1,q0,q1]`，
    和 v 的配对对不上（判别过：两者不是一回事，不是等价改写）。
    attr 用 ggml 视角写（和 slice / sum / concat 同一套），ref 自己换算。
    """
    np = _np()

    def f(x, reps, axis):
        return np.repeat(x, int(reps), axis=x.ndim - 1 - int(axis))
    return f


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
    "clamp_max": {"ins": 1, "f": "min(x, m)", "ref": _ref_clamp_max(),
                  "src": "by1exec.op_moe 的 np.minimum(g, limit)"},
    "add_scalar": {"ins": 1, "f": "a + c", "ref": _ref_add_scalar(),
                   "src": "by1exec.op_moe 的归一守卫 `sum + 1e-20`"},
    "one_hot":   {"ins": 2, "f": "out[...,j] = 1 若 j ∈ ids 否则 0"
                                "（宽度 = 参照的最后一维）",
                  "ref": _ref_one_hot(),
                  "src": "by1exec.op_moe 的 put_along_axis(gm, gi, 1.0)"},
    "masked_fill": {"ins": 2, "f": "out = m > 0 ? x : v",
                    "ref": _ref_masked_fill(),
                    "src": "by1exec.op_moe 的 np.where(mask, choice, -inf)"},
    "le":        {"ins": 2, "f": "out = 1 若 a <= b 否则 0（广播二元）",
                  "ref": _ref_le(),
                  "src": "by1exec._dsa_index 的因果比较 ar[None,:] <= ar[:,None]"},
    "div":       {"ins": 2, "f": "a / b（广播）", "ref": _ref_div(),
                  "src": "ggml_div / numpy"},
    # ── GDN / delta 规则那一族（Linear 与 KDA 共用）──
    "concat":    {"ins": 2, "f": "沿 ggml 轴 axis 接两条",
                  "ref": _ref_concat(),
                  "src": "torch.cat / ggml_concat / by1exec 的 np.concatenate"},
    "conv1d_causal": {"ins": 2, "f": "深度可分离因果卷积（逐通道一份核，"
                                     "左边补 kernel-1）；**激活不在这里**",
                      "ref": _ref_conv1d_causal(),
                      "src": "by1exec.op_linear 的短卷积 / causal_conv1d"},
    "delta_rule": {"ins": 5,
                   "f": "逐 token：st*=exp(decay)；δ=(v−(st·k))·beta；"
                        "st+=k⊗δ；out=(st·q)",
                   "ref": _ref_delta_rule(),
                   "src": "by1exec.op_linear / op_kda 的 for"},
    "ssm_scan":  {"ins": 6,
                  "f": "逐 token：st=exp(d·A)·st + d·(x⊗B)；y=st·C + D·x",
                  "ref": _ref_ssm_scan(),
                  "src": "by1exec.op_ssm 的递推 for（Mamba2）"},
    "exp":       {"ins": 1, "f": "eˣ", "ref": _ref_exp(),
                  "src": "by1exec 的 np.exp（衰减门要用）"},
    "log1p":     {"ins": 1, "f": "log(1+x)", "ref": _ref_log1p(),
                  "src": "by1exec.op_linear 的 np.log1p（衰减门要用）"},
    "sqrt":      {"ins": 1, "f": "√x", "ref": _ref_sqrt(),
                  "src": "by1exec 的 1/sqrt(·)（L2 归一 / 输出归一）"},
    # GDN 的归一与 GQA 广播。
    # `l2_norm(x, vd·eps) · √vd` **就是** mean 那一支（图里现拼），
    # 所以闭集只需要这一个归一。
    "l2_norm":   {"ins": 1, "f": "x / √(Σx²+eps)，沿最后轴（**sum 不是 mean**）",
                  "ref": _ref_l2_norm(),
                  "src": "by1exec.op_linear 的 q/k 归一 / torch l2norm"},
    "repeat":    {"ins": 1, "f": "沿 ggml 轴 axis 整块重复 reps 次",
                  "ref": _ref_repeat(),
                  "src": "np.repeat / by1exec.op_linear 的 GQA 广播"},
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

#: **B1 数值规格：形状 / dtype / 边界**（公式与状态更新在上面每条的 `f` 里）。
#:
#: 这三样原来只活在 ref() 的代码里 —— 代码是**给机器的规格**，人读的
#: 规格没有，于是"eps>0 这种前提在哪儿"只能翻实现。现在每个原语都有：
#:   shape  输出形状规则（相对输入）
#:   dtype  dtype 传播规则
#:   bound  数值前提 / 边界 / 不变量（写得出反例的那半句）
#:
#: **拼进 PRIMS，不当第二份事实** —— 拼完立刻对账（硬断言）：
#: 键集必须逐个相等，多一个少一个都当场炸。A3-1 的先例：两份事实
#: 的漂移只能是检查错误，不能是运行时惊讶。
_SPEC = {
    "add": {"shape": "out = broadcast(ins0, ins1)", "dtype": "浮点逐元，两入同 dtype",
            "bound": "广播规则按 numpy；不接受 dtype 提升"},
    "mul": {"shape": "out = broadcast(ins0, ins1)", "dtype": "浮点逐元，两入同 dtype",
            "bound": "同 add"},
    "scale": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "s 为标量属性"},
    "matmul": {"shape": "x[..,i,k] · w[j,k]ᵀ -> out[..,i,j]；w 恒为 2-D [out,in]",
               "dtype": "浮点；乘加语义按 ref 的 float32 累加",
               "bound": "k 维必须相等（不等是形状错，不是数值差）"},
    "batch_matmul": {"shape": "a[β//r,i,k]·b[β,j,k] -> out[β,i,j]，r = B/b_a",
                     "dtype": "浮点；同 matmul",
                     "bound": "r 必须整除；r>1 即 GQA 广播"},
    "rms_norm": {"shape": "w = [d]；out = ins0", "dtype": "浮点逐元",
                 "bound": "eps > 0；mean 沿最后一轴；one_plus 二选一见 f"},
    "layer_norm": {"shape": "w = b = [d]；out = ins0", "dtype": "浮点逐元",
                   "bound": "eps > 0；μ/var 沿最后一轴"},
    "silu": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "无前提"},
    "gelu_tanh": {"shape": "out = ins0", "dtype": "浮点逐元",
                  "bound": "近似式系数是规格的一部分（0.044715, √(2/π)）"},
    "gelu": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "erf 精确式"},
    "sigmoid": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "无前提"},
    "softmax": {"shape": "out = ins0（沿最后轴归一）", "dtype": "浮点逐元",
                "bound": "减 max 保稳；Σ 规范化"},
    "softplus": {"shape": "out = ins0", "dtype": "浮点逐元",
                 "bound": "数值稳定式（max(x,0)+log1p(e^-|x|)）是规格的一部分"},
    "sqrtsoftplus": {"shape": "out = ins0", "dtype": "浮点逐元",
                     "bound": "√(softplus(x))；softplus 稳定式同上"},
    "top_k": {"shape": "out = ins0[.., :k] 的**下标**", "dtype": "出 int 下标",
              "bound": "1 ≤ k ≤ 尾维；**并列取小下标**（三后端同一句）"},
    "gather": {"shape": "沿最后轴对齐 idx 形状", "dtype": "出同 ins0 的值",
               "bound": "idx ∈ [0, 尾维)"},
    "sum": {"shape": "沿 ggml 轴 axis 归约；keepdims 决定保不保轴",
            "dtype": "浮点；累加语义按 ref",
            "bound": "axis 合法；keepdims 默认开"},
    "clamp_min": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "m 为标量属性"},
    "clamp_max": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "m 为标量属性"},
    "add_scalar": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "c 为标量属性"},
    "one_hot": {"shape": "宽度 = ins1 的最后一维；ids 形状在前",
                "dtype": "出 float（1.0 / 0.0）",
                "bound": "ids ∈ [0, 宽度)"},
    "masked_fill": {"shape": "out = ins0（m 广播对齐）", "dtype": "值 v 的 dtype",
                    "bound": "m > 0 取 x，否则取 v —— **边界是 0 不是 0.5**"},
    "le": {"shape": "out = broadcast(ins0, ins1)", "dtype": "出 float（1.0 / 0.0）",
           "bound": "<= 是含等号的（因果比较靠它）"},
    "div": {"shape": "out = broadcast(ins0, ins1)", "dtype": "浮点逐元",
            "bound": "b ≠ 0；用途里配 clamp_min 护栏"},
    "concat": {"shape": "沿 ggml 轴 axis 接；其余轴必须相等",
               "dtype": "浮点；两入同 dtype", "bound": "轴不等是形状错"},
    "conv1d_causal": {"shape": "核 [c,k]；out[..,t,c]，左补 k-1 后取有效段",
                      "dtype": "浮点；逐通道一份核",
                      "bound": "**激活不在这里**（下游原语的事）"},
    "delta_rule": {"shape": "q/k/v [t,c]；beta/decay [t,c] 或 [t,1]；out [t,c]",
                   "dtype": "浮点；状态 st 在 float 语义下逐 token 更新",
                   "bound": "状态更新见 f 的逐 token 式；零初始"},
    "ssm_scan": {"shape": "d [t,c]；A [c]；x [t,c]；B/C [t,c]；D [c]；out [t,c]",
                 "dtype": "浮点；状态 st 逐 token 更新",
                 "bound": "状态更新见 f 的递推式；零初始"},
    "exp": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "无前提（溢出即真溢出）"},
    "log1p": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "x > -1"},
    "sqrt": {"shape": "out = ins0", "dtype": "浮点逐元", "bound": "x ≥ 0"},
    "l2_norm": {"shape": "out = ins0（沿最后轴）", "dtype": "浮点逐元",
                "bound": "eps > 0；**sum 不是 mean**（均值那支是现拼的）"},
    "repeat": {"shape": "沿 ggml 轴 axis 整块重复 reps 次", "dtype": "同 ins0",
               "bound": "reps ≥ 1；axis 合法"},
    "rope": {"shape": "x [n, nh, hd]（exec 视角转置进出）；inv 长度 hd/2",
             "dtype": "浮点逐元",
             "bound": "n_dims ≤ hd；pairing ∈ {half, interleaved}；"
                      "half=(i,i+h)、interleaved=(2i,2i+1) 且拼接输出"},
    "soft_max": {"shape": "a [头,q,k] -> out 同形；mask [q,k]",
                 "dtype": "浮点逐元",
                 "bound": "scale > 0；掩码规则 j<=i 且 j>i-window（window 取自 mask 名）"},
    "soft_max_sink": {"shape": "同 soft_max；sink 为逐头标量（多一槽）",
                      "dtype": "浮点逐元",
                      "bound": "同 soft_max；sink 槽参与归一"},
    "get_rows": {"shape": "out = ids 形状 + table 行宽", "dtype": "出同 table",
                 "bound": "ids ∈ [0, 行数)"},
    "view": {"shape": "总数不变、元素序不变", "dtype": "同 ins0",
             "bound": "总数必须相等（不等是形状错）"},
    "permute": {"shape": "轴置换（带步长的视图）", "dtype": "同 ins0",
                "bound": "axes 必须是 0..ndim-1 的排列"},
    "cont": {"shape": "物化为连续；带 shape 时同时重解释",
             "dtype": "同 ins0", "bound": "无前提"},
    "slice": {"shape": "沿 ggml 轴 axis 取 [start, stop)", "dtype": "同 ins0",
              "bound": "0 ≤ start < stop ≤ 轴长"},
}
for _name, _spec in _SPEC.items():
    PRIMS[_name].update(_spec)
_missing_spec = set(PRIMS) - set(_SPEC)
_extra_spec = set(_SPEC) - set(PRIMS)
# **不用 assert** —— `python -O` 会把它整条删掉，不变量就静默消失了
# （by1lint 的 ⑤ 号规则盯的就是这个）。要炸就真炸。
if _missing_spec or _extra_spec:
    raise RuntimeError(
        "B1 规格与原语表对不上账 —— 缺 %s 多 %s（两份事实漂移了，"
        "这正是 B1 要灭的病）" % (sorted(_missing_spec), sorted(_extra_spec)))
#   MoE   ：四种选法 + expert_bias + gptoss 激活全收编
#   KDA   ：已收编（低秩门 / GVA / gate_lower 三支）
#   MLA   ：已收编（非稀疏）；**SparseMLA（index_heads）拒绝** ——
#           dsa_index 的判卷人是 transformers 的 GlmMoeDsaIndexer
#   SSM   ：已收编（Mamba2 递推 = `ssm_scan` 原语）
#   Linear：GDN 已收编（conv1d_causal + delta_rule）
#   Raw / External：逃生舱，按设计不收编
# 每条下沉时：公式 + ref() + 反例一起进来，别只搬代码。
UNLOWERED = ("Raw", "External")


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

def lower_op(g, i, j, o, d, pname, refmap, cache=None):
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

    if k not in ("Embed", "Norm", "Head", "Attention", "FFN", "Add", "MoE",
                 "Linear", "KDA", "MLA", "SSM"):
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
        _lower_attn(g, p, src[0], dst, w, a, d, cache=cache)
    elif k == "KDA":
        _lower_kda(g, p, src[0], dst, w, a, d)
    elif k == "MLA":
        _lower_mla(g, p, src[0], dst, w, a, d)
    elif k == "SSM":
        _lower_ssm(g, p, src[0], dst, w, a, d)
    elif k == "MoE":
        _lower_moe(g, p, src[0], dst, w, a, d)
    elif k == "Linear":
        _lower_linear(g, p, src[0], dst, w, a, d)
    return dst


def _lower_linear(g, p, x, dst, w, a, d):
    """GDN（Linear）的 lowering —— 语义照 `by1exec.op_linear`（e2e 的判据）。

    一条链：融合投影 → 按头切开 → β / 衰减门 → 拼回去进**短卷积** →
    切回 → per-head L2 归一 → **delta_rule** 递推 → 输出 RMS + 门 + out_proj。

    三处 layout 的关键（都写在这，别再错第三遍）：

    ① **exec 约定、ggml 视角标形状**。图里的值是 (n, …) 在前，节点上的
       shape 是 ggml 视角（特征在前）—— 同一块内存的两种读法；
    ② **slice/concat 的 axis 按 ggml 视角写**（axis 0 = 最后那条轴）；
    ③ **delta_rule 吃 [b, 头, n, dim]**（四个轴），所以要给 3 维张量补一个
       batch 轴 —— 用 view，不是 reshape 之外的东西。
    """
    nk, nv = int(a["k_heads"]), int(a["v_heads"])
    dk, dv = int(a["k_dim"]), int(a["v_dim"])
    rep = nv // nk
    kd, vd = nk * dk, nv * dv
    C = 2 * dk + 2 * dv * rep          # in_proj_qkvz 的每头宽度
    R = 2 * rep                         # in_proj_ba 的每头宽度
    L = 2 * kd + vd                     # 卷积那一支的通道数（q|k|v）
    eps_l2 = float(a.get("l2_eps", 1e-6))
    eps_nm = float(a.get("norm_eps", _DEFAULT_EPS))
    kern = int(a["conv_kernel"])
    if str(a.get("act", "silu")) != "silu":
        raise SystemExit("  [不支持] Linear.act=%r 还没有 lowering" % a.get("act"))

    def _cut(t, s0, s1, nm, shape=None):
        return g.add("slice", [t], [p + nm],
                     attrs={"axis": 0, "start": int(s0), "stop": int(s1)},
                     shapes=shape)
    # ── ① 融合投影，按头（nk）切开 ─────────────────────────────────
    qz = g.add("matmul", [w("in_proj_qkvz"), x], [p + "qz"], shapes=[(None, None)])
    qz = g.add("view", [qz], [p + "qz3"], attrs={"shape": [C, nk, None]},
               shapes=[(C, nk, None)])
    q0 = _cut(qz, 0, dk, "q0")                                  # (n,nk,dk)
    k0 = _cut(qz, dk, 2 * dk, "k0")
    v0 = _cut(qz, 2 * dk, 2 * dk + dv * rep, "v0")              # (n,nk,dv·rep)
    z0 = _cut(qz, 2 * dk + dv * rep, C, "z0")
    v0 = g.add("view", [v0], [p + "v1"], attrs={"shape": [dv, nv, None]},
               shapes=[(dv, nv, None)])                         # z 的门在后面才用
    z0 = g.add("view", [z0], [p + "z1"], attrs={"shape": [dv, nv, None]},
               shapes=[(dv, nv, None)])
    q1 = g.add("view", [q0], [p + "q1"], attrs={"shape": [kd, None]},
               shapes=[(kd, None)])
    k1 = g.add("view", [k0], [p + "k1"], attrs={"shape": [kd, None]},
               shapes=[(kd, None)])
    v1 = g.add("view", [v0], [p + "v2"], attrs={"shape": [vd, None]},
               shapes=[(vd, None)])
    # ── ② β 与衰减门 g = −exp(A_log)·log1p(exp(aa + dt_bias)) ──────
    ba = g.add("matmul", [w("in_proj_ba"), x], [p + "ba"], shapes=[(None, None)])
    ba = g.add("view", [ba], [p + "ba3"], attrs={"shape": [R, nk, None]},
               shapes=[(R, nk, None)])
    bb = _cut(ba, 0, rep, "bb")
    # (n, nk, rep) -> (n, nv)：ggml 视角就是 (nv, n)
    bb = g.add("view", [bb], [p + "bb2"], attrs={"shape": [nv, None]},
               shapes=[(nv, None)])
    aa = _cut(ba, rep, 2 * rep, "aa")
    aa = g.add("view", [aa], [p + "aa2"], attrs={"shape": [nv, None]},
               shapes=[(nv, None)])
    beta = g.add("sigmoid", [bb], [p + "beta"], shapes=[(nv, None)])
    t1 = g.add("add", [aa, w("dt_bias")], [p + "t1"], shapes=[(nv, None)])
    t2 = g.add("exp", [t1], [p + "t2"], shapes=[(nv, None)])
    t3 = g.add("log1p", [t2], [p + "t3"], shapes=[(nv, None)])
    t4 = g.add("exp", [w("A_log")], [p + "t4"], shapes=[(nv,)])
    t4 = g.add("scale", [t4], [p + "t5"], attrs={"s": -1.0}, shapes=[(nv,)])
    gm = g.add("mul", [t3, t4], [p + "g"], shapes=[(nv, None)])
    # ── ③ 拼回 q|k|v 进**因果短卷积**，silu，再切回 ─────────────────
    mk = g.add("concat", [q1, k1], [p + "mk0"], attrs={"axis": 0},
               shapes=[(2 * kd, None)])
    mk = g.add("concat", [mk, v1], [p + "mk"], attrs={"axis": 0},
               shapes=[(L, None)])
    # conv1d_causal 的 ref 吃 [b, n, C] —— ggml 视角是 [C, n, b]，
    # 所以这里要写 **[L, None, 1]**（第一版写成 [L, 1, None]：把 batch 轴
    # 和 token 轴对调，于是 b=n、s=1，卷积算了个完全不是那个的东西）。
    mk3 = g.add("view", [mk], [p + "mk3"], attrs={"shape": [L, None, 1]},
                shapes=[(L, None, 1)])
    c3 = g.add("conv1d_causal", [mk3, w("conv")], [p + "c3"],
               attrs={"kernel": kern}, shapes=[(L, 1, None)])
    cm = g.add("view", [c3], [p + "cm"], attrs={"shape": [L, None]},
               shapes=[(L, None)])
    cm = g.add("silu", [cm], [p + "cms"], shapes=[(L, None)])
    q2 = _cut(cm, 0, kd, "q2")
    k2 = _cut(cm, kd, 2 * kd, "k2")
    v2 = _cut(cm, 2 * kd, L, "v3")
    # ── ④ 归头 + per-head L2 归一（GQA 广播在最前） ─────────────────
    q3 = g.add("view", [q2], [p + "q4"], attrs={"shape": [dk, nk, None]},
               shapes=[(dk, nk, None)])
    k3 = g.add("view", [k2], [p + "k4"], attrs={"shape": [dk, nk, None]},
               shapes=[(dk, nk, None)])
    v3 = g.add("view", [v2], [p + "v4"], attrs={"shape": [dv, nv, None]},
               shapes=[(dv, nv, None)])
    if rep > 1:
        # **整块**重复 [q0,q0,q1,q1]：映射是"头 → 头//rep"。
        # 平铺会给出 [q0,q1,q0,q1]，和 v 的配对对不上（不是等价改写）。
        q3 = g.add("repeat", [q3], [p + "q5"], attrs={"reps": rep, "axis": 1},
                   shapes=[(dk, nv, None)])
        k3 = g.add("repeat", [k3], [p + "k5"], attrs={"reps": rep, "axis": 1},
                   shapes=[(dk, nv, None)])
    q3 = g.add("l2_norm", [q3], [p + "qn"], attrs={"eps": eps_l2},
               shapes=[(dk, nv, None)])
    k3 = g.add("l2_norm", [k3], [p + "kn"], attrs={"eps": eps_l2},
               shapes=[(dk, nv, None)])
    # ── ⑤ delta_rule：q 先乘 dk^-0.5；递推吃 [b, 头, n, dim] ──────
    q3 = g.add("scale", [q3], [p + "qs"], attrs={"s": float(dk) ** -0.5},
               shapes=[(dk, nv, None)])
    # **ggml 视角要把"头"挪到 n 前面**：手上是 ggml [dim, 头, n] /
    # [头, n]，而 delta_rule 要 exec [1, 头, n, dim] = ggml [dim, n, 头, 1]。
    # 所以**先 permute 补视图**（第一版少了这一步：直接 view 成 4 维，
    # 于是 batch 轴和 token 轴对调，递推算在"每头一个 token"上）。
    def _b4(t, nm, d0, h1, two):
        ax = [1, 0] if two else [0, 2, 1]
        pp = g.add("permute", [t], [p + nm + "p"], attrs={"axes": ax},
                   shapes=[(None, None)])
        shp = ([1, None, h1, 1] if two else [d0, None, h1, 1])
        return g.add("view", [pp], [p + nm], attrs={"shape": shp},
                     shapes=[tuple(shp[:2]) + (h1, shp[-1])])
    o = g.add("delta_rule", [_b4(q3, "qh", dk, nv, False),
                             _b4(k3, "kh", dk, nv, False),
                             _b4(v3, "vh", dv, nv, False),
                             _b4(beta, "bh", 1, nv, True),
                             _b4(gm, "gh", 1, nv, True)],
              [p + "dr"], shapes=[(dv, None, nv, 1)])
    # 出来是 exec [1, 头, n, dv]；ggml 视角 [dv, n, 头, 1] -> permute ->
    # [dv, 头, n, 1] -> view 掉 batch 轴 -> exec (n, 头, dv)
    o = g.add("permute", [o], [p + "drp"], attrs={"axes": [0, 2, 1, 3]},
              shapes=[(dv, None, nv, 1)])
    o = g.add("view", [o], [p + "o2"], attrs={"shape": [dv, nv, None]},
              shapes=[(dv, nv, None)])
    # ── ⑥ 输出 RMS + 门 ───────────────────────────────────────────
    # mean 那一支现拼：`l2_norm(x, dv·ε) · √dv` **就是** `x/√(mean(x²)+ε)`
    #   √dv·x/√(Σ+dv·ε) = x/√((Σ+dvε)/dv) = x/√(mean+ε)
    y = g.add("l2_norm", [o], [p + "yn"], attrs={"eps": eps_nm * dv},
              shapes=[(dv, nv, None)])
    y = g.add("scale", [y], [p + "ys"], attrs={"s": float(dv) ** 0.5},
              shapes=[(dv, nv, None)])
    y = g.add("mul", [y, w("norm.w")], [p + "yw"], shapes=[(dv, nv, None)])
    zs = g.add("silu", [z0], [p + "zs"], shapes=[(dv, nv, None)])
    y = g.add("mul", [y, zs], [p + "yz"], shapes=[(dv, nv, None)])
    y = g.add("view", [y], [p + "y2"], attrs={"shape": [vd, None]},
              shapes=[(vd, None)])
    g.add("matmul", [w("out_proj"), y], [p + "o"], shapes=[(None, None)])
    g.nodes[-1]["outs"] = [dst]
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
    if rt not in ("softmax_topk", "sigmoid_topk", "sigmoid_group_topk",
                  "topk_softmax"):
        raise SystemExit("  [不支持] MoE.routing=%r 不在闭集里" % rt)
    _act = str(a.get("act", "silu"))
    if _act not in ("silu", "gptoss", "gelu", "gelu_new", "gelu_tanh",
                    "tanh", "relu2"):
        # 表外取值拒绝 —— op_moe 那边的静默 silu 是记过案的 bug
        raise SystemExit("  [不支持] MoE.act=%r 还没有 lowering" % a.get("act"))
    _APRIM = {"silu": "silu", "gelu": "gelu", "gelu_new": "gelu_tanh",
              "gelu_tanh": "gelu_tanh", "tanh": "gelu_tanh"}

    def _act_node(inp, nm, shape):
        """激活分派（照 `_act_apply` / `_act`）；relu2 = relu²，两个节点拼。"""
        if _act == "relu2":
            r = g.add("clamp_min", [inp], [p + nm + "0"],
                      attrs={"min": 0.0}, shapes=[shape])
            return g.add("mul", [r, r], [p + nm], shapes=[shape])
        return g.add(_APRIM[_act], [inp], [p + nm], shapes=[shape])
    E, K, H = int(a["experts"]), int(a["top_k"]), int(a["hidden"])
    sb_on = bool(a.get("score_bias"))
    # **打分函数和选法正交**，解析走 `by1ir.moe_score` 那一份（唯一定义）。
    from by1ir import moe_score as _moe_score
    _sc = _moe_score(a)
    if _sc is None:
        raise SystemExit("  [不支持] MoE 的打分函数推不出来（routing=%r）" % rt)
    # ── ① 路由 logits = x @ routerᵀ（+ bias） ───────────────────────
    lg = g.add("matmul", [w("router"), x], [p + "lg"], shapes=[(E, None)])
    if a.get("router_bias"):
        lg = g.add("add", [lg, w("router.bias")], [p + "lg1"],
                   shapes=[(E, None)])
    if rt == "softmax_topk":
        # 全体 softmax -> top-k -> **权重用未加偏置的 logits 重算**（Mixtral）。
        # score_bias 只进选择（by1exec 的 `sel` / `base`）。
        if _sc != "softmax":
            raise SystemExit(
                "  [不支持] routing=softmax_topk 不经过打分函数，score=%r "
                "不会被读 —— 要么去掉，要么换 sigmoid_topk" % _sc)
        sel = lg
        if sb_on:
            sel = g.add("add", [lg, w("score_bias")], [p + "sel"],
                        shapes=[(E, None)])
        sm = g.add("softmax", [sel], [p + "sm"], shapes=[(E, None)])
        topi = g.add("top_k", [sm], [p + "topi"], attrs={"k": K},
                     shapes=[(K, None)])
        if sb_on:
            sm2 = g.add("softmax", [lg], [p + "sm2"], shapes=[(E, None)])
            tv = g.add("gather", [sm2, topi], [p + "tv"], shapes=[(K, None)])
        else:
            tv = g.add("gather", [sm, topi], [p + "tv"], shapes=[(K, None)])
        den = g.add("sum", [tv], [p + "den"], attrs={"axis": 0,
                                                     "keepdims": True},
                    shapes=[(1, None)])
        den = g.add("clamp_min", [den], [p + "den1"], attrs={"min": 1e-9},
                    shapes=[(1, None)])
    elif rt == "sigmoid_topk":
        # **先打分再直接 top-k**，不分组（Step-3.7）——照 `op_moe` 605-612：
        #   sc = score(logits)        # 打分吃**未加偏置**的 logits
        #   choice = sc + score_bias  # 只影响选择
        #   topi = top_k(choice)；tv = gather(sc, topi)；tv / (Σtv + 1e-20)
        # 归一守卫是 `+ 1e-20`，**不是** softmax_topk 的 `max(·, 1e-9)`。
        sc = g.add(_sc, [lg], [p + "sc"], shapes=[(E, None)])
        sel = sc
        if sb_on:
            sel = g.add("add", [sc, w("score_bias")], [p + "sel"],
                        shapes=[(E, None)])
        topi = g.add("top_k", [sel], [p + "topi"], attrs={"k": K},
                     shapes=[(K, None)])
        tv = g.add("gather", [sc, topi], [p + "tv"], shapes=[(K, None)])
        den = g.add("sum", [tv], [p + "den"], attrs={"axis": 0,
                                                     "keepdims": True},
                    shapes=[(1, None)])
        den = g.add("add_scalar", [den], [p + "den1"], attrs={"c": 1e-20},
                    shapes=[(1, None)])
    elif rt == "sigmoid_group_topk":
        # noaux_tc（Ling / DeepSeek V3 / GLM / DeepSeek V4）——
        # 照 `op_moe` 585-604 逐句：**先打分**，组内前 2 之和排名选组，
        # 组外填 -inf 再 top-k；`score_bias` **只进选择**，权重用未加偏置的 sc。
        ng, tg = int(a.get("n_group") or 0), int(a.get("topk_group") or 0)
        if not ng or not tg:
            raise SystemExit("  [FAIL] routing=sigmoid_group_topk 需要 "
                             "n_group 与 topk_group")
        if E % ng:
            raise SystemExit("  [FAIL] experts=%d 不是 n_group=%d 的整数倍"
                             % (E, ng))
        eg = E // ng
        sc = g.add(_sc, [lg], [p + "sc"], shapes=[(E, None)])
        sel = sc
        if sb_on:
            sel = g.add("add", [sc, w("score_bias")], [p + "sel"],
                        shapes=[(E, None)])
        # 每组取**前 2 之和**排名：sort[-2:].sum == top_k(2) -> gather -> sum
        # （两个加数交换次序在 IEEE 里逐位相同）—— **不加排序原语**。
        gs = g.add("view", [sel], [p + "gs0"],
                   attrs={"shape": [eg, ng, None]}, shapes=[(eg, ng, None)])
        t2i = g.add("top_k", [gs], [p + "t2i"], attrs={"k": 2},
                    shapes=[(2, ng, None)])
        t2 = g.add("gather", [gs, t2i], [p + "t2"], shapes=[(2, ng, None)])
        gsum = g.add("sum", [t2], [p + "gsum"],
                     attrs={"axis": 0, "keepdims": False},
                     shapes=[(ng, None)])
        gi = g.add("top_k", [gsum], [p + "gi"], attrs={"k": tg},
                   shapes=[(tg, None)])
        gm = g.add("one_hot", [gi, gsum], [p + "gm"], shapes=[(ng, None)])
        # 组选择 -> 专家掩码：[n, ng] -> [n, ng, 1] -> 每组整块重复 eg 次
        gm3 = g.add("view", [gm], [p + "gm3"],
                    attrs={"shape": [1, ng, None]}, shapes=[(1, ng, None)])
        mk = g.add("repeat", [gm3], [p + "mk"],
                   attrs={"reps": eg, "axis": 0}, shapes=[(eg, ng, None)])
        mk = g.add("view", [mk], [p + "mk2"], attrs={"shape": [E, None]},
                   shapes=[(E, None)])
        ch = g.add("masked_fill", [sel, mk], [p + "ch"],
                   attrs={"value": float("-inf")}, shapes=[(E, None)])
        topi = g.add("top_k", [ch], [p + "topi"], attrs={"k": K},
                     shapes=[(K, None)])
        tv = g.add("gather", [sc, topi], [p + "tv"], shapes=[(K, None)])
        den = g.add("sum", [tv], [p + "den"], attrs={"axis": 0,
                                                     "keepdims": True},
                    shapes=[(1, None)])
        den = g.add("add_scalar", [den], [p + "den1"], attrs={"c": 1e-20},
                    shapes=[(1, None)])
    else:
        # topk_softmax（GPT-OSS）：先 top-k，再**只对这 k 个 logits** 做
        # softmax —— 选和权重都用 `logits`，不看 score_bias，不经过打分函数。
        if _sc != "softmax":
            raise SystemExit(
                "  [不支持] routing=topk_softmax 不经过打分函数，score=%r "
                "不会被读 —— 要么去掉，要么换 sigmoid_topk" % _sc)
        topi = g.add("top_k", [lg], [p + "topi"], attrs={"k": K},
                     shapes=[(K, None)])
        tv = g.add("gather", [lg, topi], [p + "tv"], shapes=[(K, None)])
        tv = g.add("softmax", [tv], [p + "sm"], shapes=[(K, None)])
    topv = g.add("div", [tv, den], [p + "topv"], shapes=[(K, None)]) \
        if rt != "topk_softmax" else tv
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
        yv = g.add("view", [mm], [p + wname + "_v"],
                   attrs={"shape": [out_w, K, None]},
                   shapes=[(out_w, K, None)])
        if a.get("expert_bias"):
            # **专家偏置按同一个下标整行取**（op_moe 647-649 / 659-660）：
            # b1/b3 加在激活前，b2 加在 w2 之后 —— 都是 get_rows + add。
            br = g.add("get_rows",
                       [w({"w1": "b1", "w2": "b2", "w3": "b3"}[wname]), topi],
                       [p + wname + "_b"], shapes=[(out_w, K, None)])
            yv = g.add("add", [yv, br], [p + wname + "_b1"],
                       shapes=[(out_w, K, None)])
        return yv
    hb = _expert("w1", xr, H, d)
    ob = _expert("w3", xr, H, d)
    if _act == "gptoss":
        # **gptoss 激活**（op_moe 650-655 逐句）：
        #   g 先 min(g, lim)、u 先 clip(u, -lim, lim)
        #   h = (u + 1) · (g · sigmoid(α·g))
        # `min` 是 `clamp_max`（clamp_min 的对称面），`clip` 两个夹。
        _lim = a.get("limit")
        if _lim is not None:
            hb = g.add("clamp_max", [hb], [p + "hb0"],
                       attrs={"max": float(_lim)}, shapes=[(H, K, None)])
            ob = g.add("clamp_min", [ob], [p + "ob0"],
                       attrs={"min": -float(_lim)}, shapes=[(H, K, None)])
            ob = g.add("clamp_max", [ob], [p + "ob1"],
                       attrs={"max": float(_lim)}, shapes=[(H, K, None)])
        t1 = g.add("scale", [hb], [p + "g1"],
                   attrs={"s": float(a.get("alpha", 1.702))},
                   shapes=[(H, K, None)])
        t2 = g.add("sigmoid", [t1], [p + "g2"], shapes=[(H, K, None)])
        t3 = g.add("mul", [hb, t2], [p + "g3"], shapes=[(H, K, None)])
        u1 = g.add("add_scalar", [ob], [p + "u1"], attrs={"c": 1.0},
                   shapes=[(H, K, None)])
        acc = g.add("mul", [t3, u1], [p + "acc0"], shapes=[(H, K, None)])
    else:
        hb = _act_node(hb, "hb", (H, K, None))
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
        if _act == "gptoss":
            # 共享专家同样走 style + limit_shared（torch 的 `_swiglu` 同句）
            _lm = a.get("limit_shared")
            if _lm is not None:
                sh = g.add("clamp_max", [sh], [p + "sh0"],
                           attrs={"max": float(_lm)}, shapes=[(None, None)])
                sh3 = g.add("clamp_min", [sh3], [p + "sh30"],
                            attrs={"min": -float(_lm)}, shapes=[(None, None)])
                sh3 = g.add("clamp_max", [sh3], [p + "sh31"],
                            attrs={"max": float(_lm)}, shapes=[(None, None)])
            t1 = g.add("scale", [sh], [p + "sh10"],
                       attrs={"s": float(a.get("alpha", 1.702))},
                       shapes=[(None, None)])
            t2 = g.add("sigmoid", [t1], [p + "sh11"], shapes=[(None, None)])
            t3 = g.add("mul", [sh, t2], [p + "sh12"], shapes=[(None, None)])
            u1 = g.add("add_scalar", [sh3], [p + "sh13"], attrs={"c": 1.0},
                       shapes=[(None, None)])
            se = g.add("mul", [t3, u1], [p + "sh4"], shapes=[(None, None)])
        else:
            sh = _act_node(sh, "sh2", (None, None))
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


def _lower_kda(g, p, x, dst, w, a, d):
    """KDA（Kimi Delta Attention）的 lowering —— 语义照 `by1exec.op_kda`
    （e2e 的判据就是它）。和 GDN 的四处差别（`op_kda` 的注释里逐条量过）：

      ① q / k / v **分开**投影，各带一条深度卷积（+ silu）
      ② 衰减 g 是**逐维**的 `[n, nv, dk]`（GDN 按头）—— **写错了不报错**，
         所以 dt_bias 要 view 成 `[nv, dk]`、A_log 要 view 成 `[nv, 1]`
         （它是**逐头**的 `(nv,)`，沿 dk 广播）
      ③ 输出门 g_proj 喂给 o_norm（激活是 **sigmoid**，不是 silu —— 这个
         错误 NumPy / PyTorch 两边一起犯过，`--compare` 一直绿）
      ④ 门低秩两层（f_a/f_b，GLM）还是一层（f_proj，Ling）——
         **契约里有 f_b_proj 才走低秩**（`by1ir.params_of` 同一条规则，
         bias 也只在低秩支、跟 `gate_out_bias` 走）

    gate_lower 支路：`g = lower · sigmoid(exp(A_log)·(fg+dt_bias))`；
    否则 `g = −exp(A_log)·softplus(fg+dt_bias)`（`softplus` 原语就是
    log1p(exp)，数值稳定版、同一个公式）。
    """
    nk, nv = int(a["k_heads"]), int(a["v_heads"])
    dk, dv = int(a["k_dim"]), int(a["v_dim"])
    kd, vd = nk * dk, nv * dv
    ck = int(a["conv_kernel"])
    eps_l2 = float(a.get("l2_eps", 1e-6))
    eps_nm = float(a.get("norm_eps", _DEFAULT_EPS))
    if str(a.get("act", "silu")) != "silu":
        raise SystemExit("  [不支持] KDA.act=%r 还没有 lowering" % a.get("act"))
    if a.get("conv_bias"):
        # `op_kda` 的 dconv **不带** conv bias —— 声明了还不加就是静默算错
        raise SystemExit("  [不支持] KDA.conv_bias 还没有 lowering")
    if nv % nk:
        raise SystemExit("  [FAIL] KDA v_heads=%d 不是 k_heads=%d 的整数倍"
                         % (nv, nk))

    def _dconv(z, cw, chan, nm):
        """深度卷积 + silu：[n, chan] -> conv1d_causal([1, n, chan]) -> silu。"""
        z3 = g.add("view", [z], [p + nm + "3"],
                   attrs={"shape": [chan, None, 1]},
                   shapes=[(chan, None, 1)])
        c = g.add("conv1d_causal", [z3, w(cw)], [p + nm + "c"],
                  attrs={"kernel": ck}, shapes=[(chan, 1, None)])
        c = g.add("view", [c], [p + nm + "2"], attrs={"shape": [chan, None]},
                  shapes=[(chan, None)])
        return g.add("silu", [c], [p + nm + "s"], shapes=[(chan, None)])

    def _b4(t, nm, d0, h1, two):
        # 2/3 维 -> exec [1, 头, n, dim]（和 GDN 的 `_b4` 同一个意思）
        ax = [1, 0] if two else [0, 2, 1]
        pp = g.add("permute", [t], [p + nm + "p"], attrs={"axes": ax},
                   shapes=[(None, None)])
        shp = ([1, None, h1, 1] if two else [d0, None, h1, 1])
        return g.add("view", [pp], [p + nm], attrs={"shape": shp},
                     shapes=[tuple(shp[:2]) + (h1, shp[-1])])

    q = _dconv(g.add("matmul", [w("q_proj"), x], [p + "qz"],
                     shapes=[(kd, None)]), "q_conv1d", kd, "q")
    k = _dconv(g.add("matmul", [w("k_proj"), x], [p + "kz"],
                     shapes=[(kd, None)]), "k_conv1d", kd, "k")
    v = _dconv(g.add("matmul", [w("v_proj"), x], [p + "vz"],
                     shapes=[(vd, None)]), "v_conv1d", vd, "v")
    q3 = g.add("view", [q], [p + "q4"], attrs={"shape": [dk, nk, None]},
               shapes=[(dk, nk, None)])
    k3 = g.add("view", [k], [p + "k4"], attrs={"shape": [dk, nk, None]},
               shapes=[(dk, nk, None)])
    v3 = g.add("view", [v], [p + "v4"], attrs={"shape": [dv, nv, None]},
               shapes=[(dv, nv, None)])
    # ── 门：fg（衰减门）/ gt（输出门） ─────────────────────────────
    if a.get("gate_lowrank"):
        _r = int(a["gate_rank"])
        fg = g.add("matmul", [w("f_b_proj"),
                              g.add("matmul", [w("f_a_proj"), x], [p + "fa"],
                                    shapes=[(_r, None)])],
                   [p + "fg"], shapes=[(nv * dk, None)])
        gt = g.add("matmul", [w("g_b_proj"),
                              g.add("matmul", [w("g_a_proj"), x], [p + "ga"],
                                    shapes=[(_r, None)])],
                   [p + "gt"], shapes=[(vd, None)])
        if a.get("gate_out_bias", True):
            gt = g.add("add", [gt, w("g_b_proj.bias")], [p + "gt1"],
                       shapes=[(vd, None)])
    else:
        fg = g.add("matmul", [w("f_proj"), x], [p + "fg"],
                   shapes=[(nv * dk, None)])
        gt = g.add("matmul", [w("g_proj"), x], [p + "gt"],
                   shapes=[(vd, None)])
    fg4 = g.add("view", [fg], [p + "fg4"], attrs={"shape": [dk, nv, None]},
                shapes=[(dk, nv, None)])
    gt4 = g.add("view", [gt], [p + "gt4"], attrs={"shape": [dv, nv, None]},
                shapes=[(dv, nv, None)])
    # ── 衰减 g：**逐维** [n, nv, dk] ──────────────────────────────
    av = g.add("view", [w("A_log")], [p + "a1"], attrs={"shape": [1, nv]},
               shapes=[(1, nv)])
    dtb = g.add("view", [w("dt_bias")], [p + "dt"],
                attrs={"shape": [dk, nv]}, shapes=[(dk, nv)])
    tt = g.add("add", [fg4, dtb], [p + "gt0"], shapes=[(dk, nv, None)])
    if a.get("gate_lower") is not None:
        # g = lower * sigmoid(exp(A_log) * (fg + dt_bias))
        ea = g.add("exp", [av], [p + "ea"], shapes=[(1, nv)])
        z = g.add("mul", [ea, tt], [p + "gz"], shapes=[(dk, nv, None)])
        z = g.add("sigmoid", [z], [p + "gs"], shapes=[(dk, nv, None)])
        gm = g.add("scale", [z], [p + "g"],
                   attrs={"s": float(a["gate_lower"])}, shapes=[(dk, nv, None)])
    else:
        # g = -exp(A_log) * softplus(fg + dt_bias)（fla/ops/kda/gate.py:50-54）
        sp = g.add("softplus", [tt], [p + "sp"], shapes=[(dk, nv, None)])
        ea = g.add("exp", [av], [p + "ea"], shapes=[(1, nv)])
        z = g.add("mul", [ea, sp], [p + "gz"], shapes=[(dk, nv, None)])
        gm = g.add("scale", [z], [p + "g"], attrs={"s": -1.0},
                   shapes=[(dk, nv, None)])
    # ── L2 归一 -> GVA 广播 -> dk^-0.5 ────────────────────────────
    q3 = g.add("l2_norm", [q3], [p + "qn"], attrs={"eps": eps_l2},
               shapes=[(dk, nk, None)])
    k3 = g.add("l2_norm", [k3], [p + "kn"], attrs={"eps": eps_l2},
               shapes=[(dk, nk, None)])
    if nv != nk:
        rep = nv // nk
        q3 = g.add("repeat", [q3], [p + "q5"], attrs={"reps": rep, "axis": 1},
                   shapes=[(dk, nv, None)])
        k3 = g.add("repeat", [k3], [p + "k5"], attrs={"reps": rep, "axis": 1},
                   shapes=[(dk, nv, None)])
    q3 = g.add("scale", [q3], [p + "qs"], attrs={"s": float(dk) ** -0.5},
               shapes=[(dk, nv, None)])
    # ── beta -> delta_rule（decay 逐维） ──────────────────────────
    # ⚠ beta 是 **sigmoid(b_proj @ x)**，不是裸投影 —— 第一版漏了 sigmoid，
    # 递推照跑、形状全对，差 2e-03 起步（逐值定位：bh 是第一个分叉点）。
    beta = g.add("matmul", [w("b_proj"), x], [p + "bp"], shapes=[(nv, None)])
    beta = g.add("sigmoid", [beta], [p + "b1"], shapes=[(nv, None)])
    o = g.add("delta_rule", [_b4(q3, "qh", dk, nv, False),
                             _b4(k3, "kh", dk, nv, False),
                             _b4(v3, "vh", dv, nv, False),
                             _b4(beta, "bh", 1, nv, True),
                             _b4(gm, "gh", dk, nv, False)],
              [p + "dr"], shapes=[(dv, None, nv, 1)])
    o = g.add("permute", [o], [p + "drp"], attrs={"axes": [0, 2, 1, 3]},
              shapes=[(dv, None, nv, 1)])
    o = g.add("view", [o], [p + "o2"], attrs={"shape": [dv, nv, None]},
              shapes=[(dv, nv, None)])
    # ── 输出 norm（mean 版 = l2_norm(x, dv·ε)·√dv）+ 输出门 sigmoid ──
    y = g.add("l2_norm", [o], [p + "yn"], attrs={"eps": eps_nm * dv},
              shapes=[(dv, nv, None)])
    y = g.add("scale", [y], [p + "ys"], attrs={"s": float(dv) ** 0.5},
              shapes=[(dv, nv, None)])
    y = g.add("mul", [y, w("o_norm.w")], [p + "yw"], shapes=[(dv, nv, None)])
    gs2 = g.add("sigmoid", [gt4], [p + "gt5"], shapes=[(dv, nv, None)])
    y = g.add("mul", [y, gs2], [p + "yz"], shapes=[(dv, nv, None)])
    y = g.add("view", [y], [p + "y2"], attrs={"shape": [vd, None]},
              shapes=[(vd, None)])
    g.add("matmul", [w("o_proj"), y], [p + "o"], shapes=[(None, None)])
    g.nodes[-1]["outs"] = [dst]
    return dst


def _lower_dsa(g, p, x, q_resid, w, a, nq):
    """SparseMLA 的 DSA 索引器（**非池路径**）—— 用原语拼出来，照
    `by1exec._dsa_index` 的数学（那份是循环 + 散射写的；这里机械化翻译，
    两份实现独立，e2e 就是它们的对拍）：

        q = view(q_resid @ wq_b)；**rope 那一半在前**（和主 MLA 的
            `[qk_nope | qk_rope]` 正好相反 —— 抄错过一次的地方）；
            索引器的 rope 配对**固定 interleaved**，和主 MLA 的 pairing 无关
        k = **LayerNorm**(x @ wk, k_norm.w, k_norm.b, eps=1e-6)（带 bias，
            不是 RMSNorm —— 参考就是这么写的）
        scores = max(q·kᵀ, 0) · hd^-0.5；sc = Σ_h w·scores，
            w = (x @ weights_proj) · nh^-0.5
        sc 因果掩掉未来（`le` 外积）-> top_k -> **one_hot 散成 keep**
            （循环版的那两层 `keep[bi, ti, sel] = True` 就是这个散射）

    `topk >= n` 的特例不用单写：top_k 钳到行宽，one_hot 全 1 ——
    和参考的 `keep[:] = True` 同一个结果。
    """
    nh, hd = int(a["index_heads"]), int(a["index_dim"])
    topk = int(a["index_topk"])
    nr = int(a.get("qk_rope") or 0)
    if nr > hd:
        raise SystemExit("  [FAIL] index qk_rope=%d > index_dim=%d" % (nr, hd))
    # q 侧
    iq = g.add("matmul", [w("indexer.wq_b.weight"), q_resid], [p + "iq0"],
               shapes=[(nh * hd, None)])
    iq = g.add("view", [iq], [p + "iq3"], attrs={"shape": [hd, nh, None]},
               shapes=[(hd, nh, None)])
    iq_rot = g.add("slice", [iq], [p + "iqr"],
                   attrs={"axis": 0, "start": 0, "stop": nr},
                   shapes=[(nr, nh, None)])
    iq_pass = g.add("slice", [iq], [p + "iqp"],
                    attrs={"axis": 0, "start": nr, "stop": hd},
                    shapes=[(hd - nr, nh, None)])
    # k 侧
    ik = g.add("matmul", [w("indexer.wk.weight"), x], [p + "ik0"],
               shapes=[(hd, None)])
    ik = g.add("layer_norm", [ik, w("indexer.k_norm.weight"),
                              w("indexer.k_norm.bias")], [p + "ikn"],
               attrs={"eps": 1e-6, "one_plus": False}, shapes=[(hd, None)])
    ik_rot = g.add("slice", [ik], [p + "ikr0"],
                   attrs={"axis": 0, "start": 0, "stop": nr},
                   shapes=[(nr, None)])
    ik_pass = g.add("slice", [ik], [p + "ikp"],
                    attrs={"axis": 0, "start": nr, "stop": hd},
                    shapes=[(hd - nr, None)])
    if nr:
        rp = {"base": float(a["rope_base"]), "pairing": "interleaved",
              "n_dims": nr, "yarn": None, "rope_scale": 1.0}
        iq_rot = g.add("rope", [iq_rot, "pos"], [p + "iqr2"], attrs=rp,
                       shapes=[(nr, nh, None)])
        ik3 = g.add("view", [ik_rot], [p + "ikr1"],
                    attrs={"shape": [nr, 1, None]}, shapes=[(nr, 1, None)])
        ik3 = g.add("rope", [ik3, "pos"], [p + "ikr2"], attrs=rp,
                    shapes=[(nr, 1, None)])
        ik_rot = g.add("view", [ik3], [p + "ikr3"],
                       attrs={"shape": [nr, None]}, shapes=[(nr, None)])
    iq = (g.add("concat", [iq_rot, iq_pass], [p + "iq4"], attrs={"axis": 0},
                shapes=[(hd, nh, None)]) if nr else iq_pass)
    ik = (g.add("concat", [ik_rot, ik_pass], [p + "ik4"], attrs={"axis": 0},
                shapes=[(hd, None)]) if nr else ik_pass)
    # 打分：scores[h, ti, tj] = max(Σ_d q·k, 0) · hd^-0.5
    # k 是单头的：view 成 [1, n, hd]（批维 1，batch_matmul 的 GQA 重复
    # 把它摊给每个查询头）—— 写成 [n, 1, hd] 批维就对到 token 上了
    ikk = g.add("view", [ik], [p + "ik5"], attrs={"shape": [hd, None, 1]},
                shapes=[(hd, None, 1)])
    qp = g.add("permute", [iq], [p + "iq5"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nh)])
    sc0 = g.add("batch_matmul", [ikk, qp], [p + "isc0"],
                shapes=[(None, None, nh)])
    sc0 = g.add("clamp_min", [sc0], [p + "isc1"], attrs={"min": 0.0},
                shapes=[(None, None, nh)])
    sc0 = g.add("scale", [sc0], [p + "isc2"],
                attrs={"s": float(hd) ** -0.5}, shapes=[(None, None, nh)])
    wv = g.add("matmul", [w("indexer.weights_proj.weight"), x], [p + "iw0"],
               shapes=[(nh, None)])
    wv = g.add("scale", [wv], [p + "iw1"], attrs={"s": float(nh) ** -0.5},
               shapes=[(nh, None)])
    # sc[ti, tj] = Σ_h w[ti,h] · scores[h, ti, tj]
    s3 = g.add("permute", [sc0], [p + "isc3"], attrs={"axes": [2, 0, 1]},
               shapes=[(None, None, nh)])
    w2 = g.add("view", [wv], [p + "iw2"], attrs={"shape": [nh, 1, None]},
               shapes=[(nh, 1, None)])
    sc = g.add("batch_matmul", [w2, s3], [p + "isc4"],
               shapes=[(None, None, 1)])
    # 挤掉尾部那个 1：对长度 1 的轴求和 == squeeze（浮点逐位相同）——
    # `view` 两个未知维是推不出的（reshape 只许一个 -1）
    sc = g.add("sum", [sc], [p + "isc5"], attrs={"axis": 0, "keepdims": False},
               shapes=[(None, None)])
    # 因果：`le` 的外积广播（key[1,n] <= query[n,1]）-> 未来填 -inf
    pcol = g.add("view", ["pos"], [p + "pcol"], attrs={"shape": [None, 1]},
                 shapes=[(None, 1)])
    prow = g.add("view", ["pos"], [p + "prow"], attrs={"shape": [1, None]},
                 shapes=[(1, None)])
    cm = g.add("le", [pcol, prow], [p + "cm"], shapes=[(None, None)])
    sc = g.add("masked_fill", [sc, cm], [p + "sc2"],
               attrs={"value": float("-inf")}, shapes=[(None, None)])
    topi = g.add("top_k", [sc], [p + "itopi"], attrs={"k": topk},
                 shapes=[(topk, None)])
    # keep [n, n] 直接交给 masked_fill —— 掩码**右对齐广播**到每个查询头
    # （view 成 [1,n,n] + repeat 那条路要两个未知维的 reshape，推不出）
    return g.add("one_hot", [topi, pcol], [p + "keep"],
                 shapes=[(None, None)])


def _lower_mla(g, p, x, dst, w, a, d):
    """MLA（Multi-Latent Attention，低秩压缩注意力）的 lowering ——
    语义照 `by1exec.op_mla`（e2e 的判据就是它）。

    链条：q/kv 各一条**低秩分解**（a_proj -> layernorm -> b_proj），kv_a
    出口切出**单头**的 k_rot 走 rope 再广播到 nh 头；拼回
    `[q_pass | q_rot]` / `[k_pass | k_rot]` 走普通因果注意力；输出门
    head_gate（per_head）+ dense。

    **SparseMLA（`index_heads`）明确拒绝** —— DSA 索引器还没收编
    （判卷人是 transformers 的 `GlmMoeDsaIndexer`），要么实现要么拒绝。
    """
    nh, nope, nr = int(a["q"]), int(a["qk_nope"]), int(a["qk_rope"])
    vd, kvl = int(a["v_dim"]), int(a["kv_lora"])
    ql = int(a["q_lora"])
    hd = nope + nr
    _op = bool(a.get("norm_one_plus", False))
    _eps = float(a.get("norm_eps", _DEFAULT_EPS))
    if a.get("index_heads") and int(a.get("index_kpool") or 1) > 1:
        # k-pool 那条路（GLM-5.3 的池打分）还没收编 —— 要么实现要么拒绝
        raise SystemExit("  [不支持] MLA.index_kpool > 1"
                         "（k-pool 路径还没收编）")
    if a.get("head_gate", "off") not in ("off", "per_head"):
        raise SystemExit("  [不支持] MLA.head_gate=%r 只实现了 per_head"
                         % a["head_gate"])

    def _cut(t, s0, s1, nm, shape):
        return g.add("slice", [t], [p + nm],
                     attrs={"axis": 0, "start": int(s0), "stop": int(s1)},
                     shapes=[shape])

    # ── q 链：q_a_proj -> rms_norm -> q_b_proj ──────────────────────
    qr = g.add("matmul", [w("q_a_proj"), x], [p + "qa"],
               shapes=[(ql, None)])
    qr = g.add("rms_norm", [qr, w("q_a_layernorm.w")], [p + "qan"],
               attrs={"eps": _eps, "one_plus": _op}, shapes=[(ql, None)])
    keep = (_lower_dsa(g, p, x, qr, w, a, nh) if a.get("index_heads")
            else None)
    q = g.add("matmul", [w("q_b_proj"), qr], [p + "qb"],
              shapes=[(nh * hd, None)])
    q = g.add("view", [q], [p + "q3"], attrs={"shape": [hd, nh, None]},
              shapes=[(hd, nh, None)])
    q_pass = _cut(q, 0, nope, "qp", (nope, nh, None))
    q_rot = _cut(q, nope, hd, "qr", (nr, nh, None))
    # ── kv 链：kv_a_proj_with_mqa 出口切 c_kv / k_rot，再 kv_b_proj ──
    ckv = g.add("matmul", [w("kv_a_proj_with_mqa"), x], [p + "ka"],
                shapes=[(kvl + nr, None)])
    c_kv = _cut(ckv, 0, kvl, "ckv", (kvl, None))
    k_rot = _cut(ckv, kvl, kvl + nr, "kr0", (nr, None))
    kvn = g.add("rms_norm", [c_kv, w("kv_a_layernorm.w")], [p + "kan"],
                attrs={"eps": _eps, "one_plus": _op}, shapes=[(kvl, None)])
    kv = g.add("matmul", [w("kv_b_proj"), kvn], [p + "kb"],
               shapes=[(nh * (nope + vd), None)])
    kv = g.add("view", [kv], [p + "kv3"],
               attrs={"shape": [nope + vd, nh, None]},
               shapes=[(nope + vd, nh, None)])
    k_pass = _cut(kv, 0, nope, "kp", (nope, nh, None))
    v = _cut(kv, nope, nope + vd, "vv", (vd, nh, None))
    # ── rope：q_rot 每头；k_rot **单头** rope 后广播到 nh 头 ─────────
    # `qk_rope = 0` 是合法配置（纯 nope）—— 空切片不 rope 不拼，
    # 和 op_mla 的 `if nr:` 同构（空切片 + 带未知维的 reshape 会炸）。
    rp = {"base": float(a["rope_base"]),
          "pairing": str(a.get("pairing", "interleaved")),
          "n_dims": nr, "yarn": None, "rope_scale": 1.0}
    if nr:
        q_rot = g.add("rope", [q_rot, "pos"], [p + "qr2"], attrs=rp,
                      shapes=[(nr, nh, None)])
        k_rot = g.add("view", [k_rot], [p + "kr1"],
                      attrs={"shape": [nr, 1, None]}, shapes=[(nr, 1, None)])
        k_rot = g.add("rope", [k_rot, "pos"], [p + "kr2"], attrs=rp,
                      shapes=[(nr, 1, None)])
        k_rot = g.add("repeat", [k_rot], [p + "kr3"],
                      attrs={"reps": nh, "axis": 1}, shapes=[(nr, nh, None)])
    # ── 拼回 -> 因果注意力（scale = 1/√(nope+nr)）───────────────────
    qq = (g.add("concat", [q_pass, q_rot], [p + "qq"], attrs={"axis": 0},
                shapes=[(hd, nh, None)]) if nr else q_pass)
    kk = (g.add("concat", [k_pass, k_rot], [p + "kk"], attrs={"axis": 0},
                shapes=[(hd, nh, None)]) if nr else k_pass)
    q3p = g.add("permute", [qq], [p + "q3p"], attrs={"axes": [0, 2, 1]},
                shapes=[(hd, None, nh)])
    k3p = g.add("permute", [kk], [p + "k3p"], attrs={"axes": [0, 2, 1]},
                shapes=[(hd, None, nh)])
    kq = g.add("batch_matmul", [k3p, q3p], [p + "kq"],
               shapes=[(None, None, nh)])
    if keep is not None:
        # DSA 的 keep：未选中的 (query, key) 填 -inf。因果那半在 soft_max
        # 的 mask_0 里 —— 两重掩码结果一样（-inf 还是 -inf），
        # 和 op_mla 的 `causal & keep` 同义。
        kq = g.add("masked_fill", [kq, keep], [p + "kqm"],
                   attrs={"value": float("-inf")}, shapes=[(None, None, nh)])
    sm = g.add("soft_max", [kq, p + "mask_0"], [p + "sm"],
               attrs={"scale": 1.0 / (hd ** 0.5)}, shapes=[(None, None, nh)])
    vt = g.add("permute", [v], [p + "v3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(vd, None, nh)])
    vt = g.add("cont", [g.add("permute", [vt], [p + "v3t"],
                              attrs={"axes": [1, 0, 2]},
                              shapes=[(None, vd, nh)])],
               [p + "v3c"], shapes=[(None, vd, nh)])
    kqv = g.add("batch_matmul", [vt, sm], [p + "kqv"],
                shapes=[(vd, None, nh)])
    att = g.add("cont", [g.add("permute", [kqv], [p + "kqvp"],
                               attrs={"axes": [0, 2, 1]},
                               shapes=[(vd, nh, None)])],
                [p + "att"], attrs={"shape": [nh * vd, None]},
                shapes=[(nh * vd, None)])
    if a.get("head_gate", "off") == "per_head":
        # 和 `_lower_attn` 的 head_gate 同一条公式（op_mla 486-491 同句）
        _act = "sigmoid" if a.get("gate_act") == "sigmoid" else "softplus"
        gg = g.add("matmul", [w("g_proj"), x], [p + "hg0"],
                   shapes=[(nh, None)])
        gg = g.add(_act, [gg], [p + "hg1"], shapes=[(nh, None)])
        gg = g.add("view", [gg], [p + "hg2"],
                   attrs={"shape": [1, nh, None]}, shapes=[(1, nh, None)])
        att3 = g.add("view", [att], [p + "hg3"],
                     attrs={"shape": [vd, nh, None]}, shapes=[(vd, nh, None)])
        att3 = g.add("mul", [att3, gg], [p + "hg4"],
                     shapes=[(vd, nh, None)])
        att = g.add("view", [att3], [p + "hg5"],
                    attrs={"shape": [nh * vd, None]},
                    shapes=[(nh * vd, None)])
    out = g.add("matmul", [w("dense"), att], [p + "o"], shapes=[(d, None)])
    if a.get("bias"):
        out = g.add("add", [out, w("dense.bias")], [p + "o1"],
                    shapes=[(d, None)])
    g.nodes[-1]["outs"] = [dst]
    return dst


def _lower_ssm(g, p, x, dst, w, a, d):
    """SSM（选择性状态空间，Mamba2 · Nemotron-H 那一版）的 lowering ——
    语义照 `by1exec.op_ssm`（判卷人 `by1ssm` 对官方 NemotronHMamba2Mixer
    验过）。链条：融合投影切三段（gate / bc / dt）-> 因果深度卷积 + silu ->
    切 xs / B / C（B、C 从组广播到头）-> **`ssm_scan` 递推** ->
    门乘 -> 分组 gated norm -> out_proj。

    递推是原语（`ssm_scan`），理由和 delta_rule 一样：状态一步一步走，
    没有并行公式。
    """
    h, hd = int(a["heads"]), int(a["head_dim"])
    ng, ssm, ck = (int(a["n_groups"]), int(a["ssm_state"]),
                   int(a["conv_kernel"]))
    di = h * hd
    cd = di + 2 * ng * ssm
    rep = h // ng
    eps_nm = float(a.get("norm_eps", _DEFAULT_EPS))
    if str(a.get("act", "silu")) != "silu":
        raise SystemExit("  [不支持] SSM.act=%r 还没有 lowering" % a.get("act"))
    if a.get("proj_bias"):
        # op_ssm 的 `proj = x @ in_proj.T` **不带 bias** —— 声明了不许静默算
        raise SystemExit("  [不支持] SSM.proj_bias 还没有 lowering")
    if h % ng:
        raise SystemExit("  [FAIL] SSM heads=%d 不是 n_groups=%d 的整数倍"
                         % (h, ng))

    def _cut(t, s0, s1, nm, shape):
        return g.add("slice", [t], [p + nm],
                     attrs={"axis": 0, "start": int(s0), "stop": int(s1)},
                     shapes=[shape])

    # ── ① 融合投影，切 gate / bc / dt ───────────────────────────────
    proj = g.add("matmul", [w("in_proj"), x], [p + "pj"],
                 shapes=[(di + cd + h, None)])
    gate = _cut(proj, 0, di, "gate", (di, None))
    bc = _cut(proj, di, di + cd, "bc", (cd, None))
    dt = _cut(proj, di + cd, di + cd + h, "dt", (h, None))
    # ── ② 因果深度卷积（+ 可选 conv.bias）+ silu ────────────────────
    bc3 = g.add("view", [bc], [p + "bc3"], attrs={"shape": [cd, None, 1]},
                shapes=[(cd, None, 1)])
    conv = g.add("conv1d_causal", [bc3, w("conv")], [p + "cv"],
                 attrs={"kernel": ck}, shapes=[(cd, 1, None)])
    conv = g.add("view", [conv], [p + "cv2"], attrs={"shape": [cd, None]},
                 shapes=[(cd, None)])
    if a.get("conv_bias"):
        conv = g.add("add", [conv, w("conv.bias")], [p + "cv3"],
                     shapes=[(cd, None)])
    bc2 = g.add("silu", [conv], [p + "bc4"], shapes=[(cd, None)])
    # ── ③ 切 xs / B / C；B、C 从组广播到头（np.repeat 的整块重复） ──
    xs = _cut(bc2, 0, di, "xs", (di, None))
    B = _cut(bc2, di, di + ng * ssm, "B0", (ng * ssm, None))
    C = _cut(bc2, di + ng * ssm, di + 2 * ng * ssm, "C0", (ng * ssm, None))
    xs = g.add("view", [xs], [p + "xs3"], attrs={"shape": [hd, h, None]},
               shapes=[(hd, h, None)])
    B = g.add("view", [B], [p + "B1"], attrs={"shape": [ssm, ng, None]},
              shapes=[(ssm, ng, None)])
    C = g.add("view", [C], [p + "C1"], attrs={"shape": [ssm, ng, None]},
              shapes=[(ssm, ng, None)])
    if rep > 1:
        B = g.add("repeat", [B], [p + "B2"], attrs={"reps": rep, "axis": 1},
                  shapes=[(ssm, h, None)])
        C = g.add("repeat", [C], [p + "C2"], attrs={"reps": rep, "axis": 1},
                  shapes=[(ssm, h, None)])
    # ── ④ A = -exp(A_log)；d = softplus(dt + dt_bias)；ssm_scan 递推 ──
    # A_log / dt_bias / D 都是**逐头**的 (h,)
    A = g.add("exp", [w("A_log")], [p + "A1"], shapes=[(h,)])
    A = g.add("scale", [A], [p + "A2"], attrs={"s": -1.0}, shapes=[(h,)])
    dtt = g.add("add", [dt, w("dt_bias")], [p + "d0"], shapes=[(h, None)])
    dtt = g.add("softplus", [dtt], [p + "d1"], shapes=[(h, None)])
    o = g.add("ssm_scan", [_b4ish(g, p, dtt, "dd", 1, h, True),
                           A,
                           _b4ish(g, p, xs, "x4", hd, h, False),
                           _b4ish(g, p, B, "B4", ssm, h, False),
                           _b4ish(g, p, C, "C4", ssm, h, False),
                           w("D")],
              [p + "sc"], shapes=[(hd, None, h, 1)])
    o = g.add("permute", [o], [p + "scp"], attrs={"axes": [0, 2, 1, 3]},
              shapes=[(hd, None, h, 1)])
    o = g.add("view", [o], [p + "o2"], attrs={"shape": [di, None]},
              shapes=[(di, None)])
    # ── ⑤ 门乘 -> 分组 gated norm（mean 版 = l2_norm(x, gs·ε)·√gs） ──
    y = g.add("mul", [o, g.add("silu", [gate], [p + "gt"],
                               shapes=[(di, None)])],
              [p + "y0"], shapes=[(di, None)])
    gs_ = di // ng
    g4 = g.add("view", [y], [p + "g4"], attrs={"shape": [gs_, ng, None]},
               shapes=[(gs_, ng, None)])
    g4 = g.add("l2_norm", [g4], [p + "gn"], attrs={"eps": eps_nm * gs_},
               shapes=[(gs_, ng, None)])
    g4 = g.add("scale", [g4], [p + "gs2"], attrs={"s": float(gs_) ** 0.5},
               shapes=[(gs_, ng, None)])
    y = g.add("view", [g4], [p + "y2"], attrs={"shape": [di, None]},
              shapes=[(di, None)])
    y = g.add("mul", [y, w("norm.w")], [p + "y3"], shapes=[(di, None)])
    g.add("matmul", [w("out_proj"), y], [p + "o"], shapes=[(None, None)])
    g.nodes[-1]["outs"] = [dst]
    return dst


def _b4ish(g, p, t, nm, d0, h1, two):
    """2/3 维 -> exec [1, 头, n, dim]（GDN/KDA 的 `_b4` 同一个意思）。
    这个放模块级，SSM 和别处共用。"""
    ax = [1, 0] if two else [0, 2, 1]
    pp = g.add("permute", [t], [p + nm + "p"], attrs={"axes": ax},
               shapes=[(None, None)])
    shp = ([1, None, h1, 1] if two else [d0, None, h1, 1])
    return g.add("view", [pp], [p + nm], attrs={"shape": shp},
                 shapes=[tuple(shp[:2]) + (h1, shp[-1])])


def _lower_attn(g, p, x, dst, w, a, d, cache=None):
    hd, nh, nkv = int(a["head_dim"]), int(a["q"]), int(a["kv"])
    if int(a["out_dim"]) != nh * hd:
        raise SystemExit("  [不支持] out_dim != q*head_dim 的注意力"
                         "（输出重排还没收编）")
    # 这里原来有一条 `head_gate 还没有 lowering` 的预拒绝 —— **已收编**
    # （见下面 attention 输出段的 head_gate 分支；非 per_head 仍然拒绝）。
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
    # **q_gate**：`wq` 出**双倍宽**（每头 2·hd），前半是 q、后半是门 ——
    # `sigmoid(门)` 乘到**注意力输出**上（不是乘到 q 上）。照 by1exec：
    # `q, gate = qq[..., :hd], qq[..., hd:]`，然后 `o = o * sigmoid(gate)`。
    # bias 加在**切分之前**的整条 qq 上（by1exec 的顺序）。
    qg = None
    if a.get("q_gate"):
        qq = g.add("matmul", [WQ, x], [p + "qg0"],
                   shapes=[(2 * nh * hd, None)])
        if bias:
            qq = g.add("add", [qq, BQ], [p + "qg1"],
                       shapes=[(2 * nh * hd, None)])
        qq3 = g.add("view", [qq], [p + "qg2"],
                    attrs={"shape": [2 * hd, nh, None]},
                    shapes=[(2 * hd, nh, None)])
        q = g.add("slice", [qq3], [p + "qg3"],
                  attrs={"axis": 0, "start": 0, "stop": hd},
                  shapes=[(hd, nh, None)])
        qg = g.add("slice", [qq3], [p + "qg4"],
                   attrs={"axis": 0, "start": hd, "stop": 2 * hd},
                   shapes=[(hd, nh, None)])
        qg = g.add("view", [qg], [p + "qg5"],
                   attrs={"shape": [nh * hd, None]}, shapes=[(nh * hd, None)])
    else:
        q = g.add("matmul", [WQ, x], [p + "q0"], shapes=[(nh * hd, None)])
        if bias:
            q = g.add("add", [q, BQ], [p + "q1"], shapes=[(nh * hd, None)])
    kq_ = g.add("matmul", [WK, x], [p + "k0"], shapes=[(nkv * hd, None)])
    if bias:
        kq_ = g.add("add", [kq_, BK], [p + "k1"], shapes=[(nkv * hd, None)])
    if _qk_on(a.get("qk_norm")) and str(a.get("qk_norm")).lower() == "full":
        # **full：拆头之前按整宽归一**（torch 同句；Instella 实测
        # `q_norm.weight` [q*head_dim]）。per_head 在拆头之后按 head_dim ——
        # 名字一样、位置不同、宽度也不同，不许混。
        if qg is not None:
            raise SystemExit("  [不支持] qk_norm=full 与 q_gate 同时出现")
        eps = float(a.get("norm_eps", _DEFAULT_EPS))
        _one = bool(a.get("norm_one_plus", False))
        q = g.add("rms_norm", [q, w("qn.w")], [p + "q1n"],
                  attrs={"eps": eps, "one_plus": _one},
                  shapes=[(nh * hd, None)])
        kq_ = g.add("rms_norm", [kq_, w("kn.w")], [p + "k1n"],
                    attrs={"eps": eps, "one_plus": _one},
                    shapes=[(nkv * hd, None)])
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
    if _qk_on(a.get("qk_norm")) and str(a.get("qk_norm")).lower() == "per_head":
        # full 在上面（拆头之前）已经归过了 —— 这里只管 per_head
        eps = float(a.get("norm_eps", _DEFAULT_EPS))
        # **one_plus 是属性，不是常量** —— 第一版这里写死 False，于是
        # 所有 one_plus=True 的模型（qwen3-next 的注意力就是）算出别的数：
        # qk_norm 少乘了个 1，而判据只会说"差 0.19"。
        _one = bool(a.get("norm_one_plus", False))
        # **norm 的公式自带乘 w**（PRIMS["rms_norm"] 就是那一条）——
        # 不是 norm 一个节点、mul 一个节点：两个节点就是两份语义。
        q3 = g.add("rms_norm", [q3, w("qn.w")], [p + "q3n"],
                   attrs={"eps": eps, "one_plus": _one},
                   shapes=[(hd, nh, None)])
        k3 = g.add("rms_norm", [k3, w("kn.w")], [p + "k3n"],
                   attrs={"eps": eps, "one_plus": _one},
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
    # ── T17-S2 的 cache 挂点：**post-rope 的 k 与 v**（写进 cache 就是
    # 这两个 —— rope 依赖位置，每个 token 只能做一次）。decode 图把
    # cache 里过去的 k/v 和这一步新的 concat 起来再进注意力。attr 的
    # axis=2 是 ggml 视角的 token 轴（ref 自己换算到 exec 轴 0）。
    if cache is not None:
        k3 = g.add("concat", [cache["k_in"], k3], [p + "k3x"],
                   attrs={"axis": 2}, shapes=[(hd, nkv, None)])
        v = g.add("concat", [cache["v_in"], v], [p + "vx"],
                  attrs={"axis": 2}, shapes=[(hd, nkv, None)])
        # cache 出去的 = **全量 cache 状态**（过去 + 这一步）——
        # 就是 concat 的输出。第一版存的是这一步的新 k/v，于是累加器
        # 永远只有最后一个 token 的 KV，t>=2 当场对不上（实测差 30）。
        cache["k_out"], cache["v_out"] = k3, v
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
    if qg is not None:
        # **门乘在注意力输出上**（合头之后、wo 之前）—— 不是乘到 q 上。
        gs = g.add("sigmoid", [qg], [p + "gs"], shapes=[(nh * hd, None)])
        att = g.add("mul", [att, gs], [p + "am"], shapes=[(nh * hd, None)])
    if a.get("head_gate", "off") != "off":
        # **head_gate（per_head）**：`by1exec.op_attention` 292-298 逐句 ——
        #   g = act(g_proj @ x)         # act：sigmoid（Ling）/ softplus（Laguna）
        #   o[b,n,h,d] *= g[b,n,h]      # **每头一个门**，乘在注意力输出上
        # 也在 wo 之前。非 per_head（整个 o 一个门）**明确拒绝** ——
        # 模型里还没有那一支，实现它之前不许假装算得了。
        if a["head_gate"] != "per_head":
            raise SystemExit("  [不支持] head_gate=%r 只实现了 per_head"
                             % a["head_gate"])
        _act = "sigmoid" if a.get("gate_act") == "sigmoid" else "softplus"
        gg = g.add("matmul", [w("g_proj"), x], [p + "hg0"],
                   shapes=[(nh, None)])
        gg = g.add(_act, [gg], [p + "hg1"], shapes=[(nh, None)])
        # [n, nh] -> [n, nh, 1]，和 [n, nh, hd] 的注意力输出按头相乘
        gg = g.add("view", [gg], [p + "hg2"],
                   attrs={"shape": [1, nh, None]}, shapes=[(1, nh, None)])
        att3 = g.add("view", [att], [p + "hg3"],
                     attrs={"shape": [hd, nh, None]}, shapes=[(hd, nh, None)])
        att3 = g.add("mul", [att3, gg], [p + "hg4"],
                     shapes=[(hd, nh, None)])
        att = g.add("view", [att3], [p + "hg5"],
                    attrs={"shape": [nh * hd, None]},
                    shapes=[(nh * hd, None)])
    out = g.add("matmul", [WO, att], [p + "o0"], shapes=[(d, None)])
    if bias:
        out = g.add("add", [out, BO], [dst], shapes=[(d, None)])
    else:
        g.nodes[-1]["outs"] = [dst]


def lower_model(ir, name="model", decode=False):
    """整份 IR -> 一张图。globals 在层前后各一截（ggml 惰性图那一课：
    最终 norm/head **必须在层之后**接线）。

    `decode=True`（T17-S2）：**增量解码图** —— 每层 Attention 的 k/v 从
    cache 输入接（`cache.L{i}.k/v`），新 k/v 作为 `cache["k_out/v_out"]`
    出去；注意力看到的是 cache+当前一步。只接 Attention；带递推/自带
    KV 的机制（Linear/KDA/MLA/SSM）**点名拒绝**，归 S4 前不接。"""
    d = ir["d_model"]
    g = Graph(name)
    g.decode_layers = []          # T17-S2：decode 图的 cache 挂点登记簿
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
            cache = None
            if decode:
                if o["kind"] == "Attention":
                    ci = len(g.decode_layers)
                    cache = {"k_in": "cache.L%d.k" % ci,
                             "v_in": "cache.L%d.v" % ci}
                elif o["kind"] in ("Linear", "KDA", "MLA", "SSM"):
                    raise SystemExit(
                        "  [不支持] decode 模式：%s（递推/自带 KV）归 S4 前不接"
                        % o["kind"])
            dst = lower_op(g, i, j, o, din, pname, refmap, cache=cache)
            if cache is not None:
                g.decode_layers.append(
                    {"k_in": cache["k_in"], "v_in": cache["v_in"],
                     "k_out": cache["k_out"], "v_out": cache["v_out"]})
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
    if sum(1 for s in gshape if s is None) > 1:
        # 两个未知维 numpy 推不出（(1, n, n) 从 n² 是歧义的）——
        # 要么改写 lowering，要么走广播，别在这儿赌
        raise SystemExit("  [FAIL] view 的目标形状 %r 有两个未知维"
                         "（一次 reshape 只许一个 -1）" % (list(gshape),))
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


def run_graph(g, params, ids, pos=None, extra=None):
    """按图求值（by1exec 约定：x [n, d]）。判卷人是 by1exec 自己。

    `pos` / `extra`（T17-S2）：decode 图要给绝对位置和 cache 输入 ——
    `pos=[t]`、`extra={"cache_L0_k": 数组, ...}`。默认行为不变。"""
    import numpy as np
    V = {}
    for k, v in params.items():
        V[k.replace(".", "_")] = v
    V["tok"] = np.asarray(ids)
    V["pos"] = np.asarray(pos) if pos is not None else np.arange(len(ids))
    if extra:
        V.update(extra)
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
        elif prim == "clamp_max":
            out = PRIMS["clamp_max"]["ref"](a[0], at["max"])
        elif prim == "add_scalar":
            out = PRIMS["add_scalar"]["ref"](a[0], at["c"])
        elif prim == "one_hot":
            out = PRIMS["one_hot"]["ref"](a[0], a[1])
        elif prim == "masked_fill":
            out = PRIMS["masked_fill"]["ref"](a[0], a[1], at["value"])
        elif prim == "le":
            out = PRIMS["le"]["ref"](a[0], a[1])
        elif prim == "concat":
            out = PRIMS["concat"]["ref"](a[0], a[1], at["axis"])
        elif prim == "conv1d_causal":
            out = PRIMS["conv1d_causal"]["ref"](a[0], a[1], at["kernel"])
        elif prim == "delta_rule":
            out = PRIMS["delta_rule"]["ref"](a[0], a[1], a[2], a[3], a[4])
        elif prim == "ssm_scan":
            out = PRIMS["ssm_scan"]["ref"](a[0], a[1], a[2], a[3], a[4],
                                           a[5])
        elif prim in ("exp", "log1p", "sqrt"):
            out = PRIMS[prim]["ref"](a[0])
        elif prim == "l2_norm":
            out = PRIMS["l2_norm"]["ref"](a[0], at["eps"])
        elif prim == "repeat":
            out = PRIMS["repeat"]["ref"](a[0], at["reps"], at["axis"])
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
            m = _mask(a[0].shape[2], w)[-a[0].shape[1]:]
            out = PRIMS["soft_max"]["ref"](a[0], m, at["scale"])
        elif prim == "soft_max_sink":
            # 同 soft_max，但第三个输入是逐头的 sink 标量
            w = int(n["ins"][1].rsplit("_", 1)[-1])
            m = _mask(a[0].shape[2], w)[-a[0].shape[1]:]
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
    bad, skipped, ran, unlowered, broken = [], [], [], [], []
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
        # **两侧求值都可能跑不动**（多宽度栈衔接还没实现之类的
        # 能力缺口）——裸崩把整个 run 打死，后面的模型全都不跑了，
        # 而"3 个通过"其实只验了字母序前面的 3 个。进账，继续跑。
        try:
            e = ex.Exec(ir, params)
            e.P["_g"] = {k: params[k] for k in
                         ("embed.weight", "pos.weight", "final_norm.w",
                          "final_norm.b", "head.weight") if k in params}
            want = e(ids[None, :])[0]
        except Exception as err:
            broken.append((f, "by1exec 侧：%s: %s"
                           % (type(err).__name__, str(err).splitlines()[0])))
            continue
        try:
            V = run_graph(g, params, ids)
            got = next(v for k, v in V.items() if k.endswith("_logits"))
        except Exception as err:
            broken.append((f, "图侧：%s: %s"
                           % (type(err).__name__, str(err).splitlines()[0])))
            continue
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
    # **对拍无法进行的要进账**，照 tier-3「已知缺口（不算失败，但仍然
    # 存在）」的先例：能力缺口不是回归，但它**没有被 e2e 作证** ——
    # 不印出来的话，"N 份一致"会被读成"全部一致"。
    for f, why in broken:
        print("  [已知缺口] %-28s 对拍无法进行：%s" % (f, why))
    if broken:
        tail += "（另有 %d 份对拍无法进行，在账）" % len(broken)
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
    # **add_scalar 不是 clamp_min** —— Σ < c 时 `+c` 和 `max(·, c)` 分叉，
    # 两条各是各的公式（sigmoid 归一守卫用的是 `+ 1e-20`）。
    _z = float(PRIMS["add_scalar"]["ref"](np.array([5e-21]), 1e-20)[0])
    if abs(_z - 1.5e-20) > 1e-32:
        bad.append("add_scalar 对不上手算（5e-21 + 1e-20 = 1.5e-20）")
    if float(PRIMS["clamp_min"]["ref"](np.array([5e-21]), 1e-20)[0]) == _z:
        bad.append("add_scalar 和 clamp_min 分不开 —— 有一个是错的")
    if not np.array_equal(PRIMS["clamp_max"]["ref"](np.array([-1.0, 2.0]), 0.5),
                          [-1.0, 0.5]):
        bad.append("clamp_max 对不上手算")
    if not np.array_equal(PRIMS["one_hot"]["ref"](np.array([[1, 2]]),
                                                  np.zeros(4)),
                          [[0.0, 1.0, 1.0, 0.0]]):
        bad.append("one_hot 对不上手算（下标 [1,2] 散成 0/1 行）")
    _mf = PRIMS["masked_fill"]["ref"](np.array([1.0, 2.0]),
                                      np.array([1.0, 0.0]), float("-inf"))
    if not (float(_mf[0]) == 1.0 and np.isneginf(_mf[1])):
        bad.append("masked_fill 对不上手算（m=0 处该填 -inf）")
    # ssm_scan：手推两步的状态更新（b=h=hd=ssm=1，d=[1,1]，x=[2,3]）
    _ss = PRIMS["ssm_scan"]["ref"](
        np.ones((1, 1, 2, 1)), np.array([-1.0]),
        np.array([[[[2.0], [3.0]]]]), np.ones((1, 1, 2, 1)),
        np.ones((1, 1, 2, 1)), np.array([0.5]))
    if (abs(float(_ss[0, 0, 0, 0]) - 3.0) > 1e-9
            or abs(float(_ss[0, 0, 1, 0]) - 5.2357588824) > 1e-9):
        bad.append("ssm_scan 对不上手算（手推两步的状态更新）")
    if not np.allclose(PRIMS["div"]["ref"](np.array([4.0, 6.0]),
                                           np.array([2.0, 3.0])), [2.0, 2.0]):
        bad.append("div 对不上手算")
    # ── GDN / delta 规则那一族（Linear 与 KDA 共用）── 手算 ──
    for k, want in (("exp", 1.0), ("log1p", 0.0), ("sqrt", 2.0)):
        z = PRIMS[k]["ref"](np.array([0.0 if k != "sqrt" else 4.0]))
        if abs(float(z[0]) - want) > 1e-12:
            bad.append("%s 对不上手算（应 %r）" % (k, want))
    if not np.array_equal(PRIMS["concat"]["ref"](np.array([[1.0, 2.0]]),
                                                 np.array([[3.0]]), 0),
                          [[1.0, 2.0, 3.0]]):
        bad.append("concat（ggml 轴 0 = 最后轴）对不上手算")
    xc = np.array([[[1.0], [2.0], [3.0]]])           # (b,n,C)
    wc = np.array([[1.0, 1.0]])                      # (C,kernel)
    cc = PRIMS["conv1d_causal"]["ref"](xc, wc, 2)
    if not np.array_equal(cc, [[[1.0], [3.0], [5.0]]]):
        bad.append("conv1d_causal 对不上手算（补 1 个的滑动窗）")
    # **因果性**：out[0] 不许受后面的 token 影响（这是"因果"两个字的意思）
    # 注意**改 x[1]（第二个 token）**，不是 x[0] —— 第一版改的是 x[0]，
    # 于是测的是"输出依赖第一个输入"，那当然成立（我自己的注释都写错了）。
    xc2 = np.array([[[1.0], [9.0], [3.0]]])
    cc2 = PRIMS["conv1d_causal"]["ref"](xc2, wc, 2)
    if not np.array_equal(cc2[0, 0], cc[0, 0]):
        bad.append("conv1d_causal 不因果：改了 x[1]，out[0] 跟着变")
    dq = np.full((1, 1, 2, 1), 0.5)
    dk_ = np.array([1.0, 2.0]).reshape(1, 1, 2, 1)
    dv_ = np.array([3.0, 4.0]).reshape(1, 1, 2, 1)
    dbeta = np.ones((1, 1, 2, 1))
    dg = np.zeros((1, 1, 2, 1))
    dr = PRIMS["delta_rule"]["ref"](dq, dk_, dv_, dbeta, dg)
    if not np.allclose(dr, [[[[1.5], [-0.5]]]]):
        bad.append("delta_rule 对不上手算（手推两步的状态更新）")
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    print("  [PASS] 原语参考实现对拍手算（rms_norm / matmul / silu / rope · "
          "打分族 softmax/softplus/sqrtsoftplus/sigmoid 并与 by1ir 同源 · "
          "路由归约 top_k/gather/sum/clamp_min/clamp_max/add_scalar/div · "
          "GDN concat/conv1d_causal〔含因果性〕/delta_rule/ssm_scan/exp/log1p/sqrt）")
    return 0


def _load(name):
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, by1paths.tool(name + ".py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def spec(write=False):
    """B1 规格文档：从 PRIMS（f/shape/dtype/bound/src）生成人读的规格表。

    **生成物** —— 手改会过期；改措辞改 `PRIMS` / `_SPEC`。
    对账是硬断言（import 时），所以这份文档**不可能**比原语表少条目。
    """
    out = ["# by1 原语数值规格（生成物）", "",
           "> **这份文档是生成的**：`python src/by1graph.py --spec`（`--write` 落盘）。",
           "> 改措辞改 `src/by1graph.py` 的 `PRIMS` / `_SPEC`；键集对账是 import 时的",
           "> 硬断言，缺一个条目整个库都跑不起来 —— 所以它不可能比原语表少。",
           ">", "> 每条的**公式**与**状态更新**在 `f`；`ref()` 是可执行规格",
           "> （`by1graph.py --selftest` 对手算逐条盯着）。", "",
           "| 原语 | 入 | 公式（f） | 形状 | dtype | 边界 / 不变量 | 出处（搬家前的旧家） |",
           "|---|---|---|---|---|---|---|"]
    for name in PRIMS:
        p = PRIMS[name]
        row = [name, str(p["ins"]), p["f"], p["shape"], p["dtype"],
               p["bound"], p["src"]]
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    out += ["", "共 %d 个原语（闭集）。" % len(PRIMS),
            "布局原语（view/permute/cont）的语义是**数据重解释**，不是数学。",
            "还没收编的机制见 `UNLOWERED`（Raw / External 逃生舱按设计不收编）。", ""]
    text = "\n".join(out)
    if write:
        path = by1paths.root("docs", "prims-spec.md")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("  写了 %s（%d 个原语）" % (path, len(PRIMS)))
        return 0
    print(text)
    return 0


def decode_check(path):
    """T17-S2 判据：**增量与全量 logits 一致**。

    同权重同输入：全量前向的每步 logits，对照"带 cache 逐 token 增量"
    的每步 logits。cache 写错一格，全量对拍照样绿、增量当场红 ——
    这是状态正确性的**唯一判据**（llamacpp-plan 的 decode/状态一节）。"""
    import numpy as np
    bc, cg = _load("by1check"), _load("by1codegen")
    _r, info = bc.check(path)
    ir = cg.compile_ir(info)
    shapes = _load("by1exec").shapes_of(ir)
    rng = np.random.default_rng(20261014)
    params = {k: rng.standard_normal(s).astype(np.float32)
              for k, s in shapes.items()}
    ids = np.asarray([5, 3, 9, 1])       # run_graph 的约定：ids 一维
    name = _os.path.basename(path)

    # 全量：一次前向
    g_full = lower_model(ir, name)
    Vf = run_graph(g_full, params, ids)
    lf = np.asarray(next(v for k, v in Vf.items() if k.endswith("_logits")))

    # 增量：逐 token，k/v 进 cache、下一步接回来
    g_dec = lower_model(ir, name, decode=True)
    dims = [(int(o["attrs"]["kv"]), int(o["attrs"]["head_dim"]))
            for L in ir["layers"] for o in L["ops"] if o["kind"] == "Attention"]
    caches = {("cache_L%d_k" % ci): np.zeros((0, nkv, hd), np.float32)
              for ci, (nkv, hd) in enumerate(dims)}
    caches.update({("cache_L%d_v" % ci): np.zeros((0, nkv, hd), np.float32)
                   for ci, (nkv, hd) in enumerate(dims)})
    steps = []
    for t in range(int(ids.shape[0])):
        V = run_graph(g_dec, params, ids[t:t + 1],
                      pos=np.asarray([t]), extra=caches)
        steps.append(np.asarray(next(v for k, v in V.items()
                                     if k.endswith("_logits"))))
        caches = {}
        for lay in g_dec.decode_layers:
            caches[lay["k_in"].replace(".", "_")] = \
                np.asarray(V[lay["k_out"].replace(".", "_")])
            caches[lay["v_in"].replace(".", "_")] = \
                np.asarray(V[lay["v_out"].replace(".", "_")])
    # token 轴：全量是 n、单步是 1 的那一根 —— 不写死轴号
    tok_axis = next(i for i, (a, b) in enumerate(zip(lf.shape, steps[0].shape))
                    if a == int(ids.shape[0]) and b == 1)
    li = np.concatenate(steps, axis=tok_axis)
    diff = float(np.max(np.abs(lf - li)))
    tol = 1e-4
    print("  增量 vs 全量 logits：最大绝对差 %.3e（容差 %g）%s"
          % (diff, tol, "  [PASS]" if diff < tol else "  [FAIL]"))

    # **负样必红**：cache 每步丢掉最老的一个 token（ring buffer 溢出
    # 那一族的病）—— 必须超出容差，否则这个判据是瞎的。
    caches = {("cache_L%d_k" % ci): np.zeros((0, nkv, hd), np.float32)
              for ci, (nkv, hd) in enumerate(dims)}
    caches.update({("cache_L%d_v" % ci): np.zeros((0, nkv, hd), np.float32)
                   for ci, (nkv, hd) in enumerate(dims)})
    steps = []
    for t in range(int(ids.shape[0])):
        V = run_graph(g_dec, params, ids[t:t + 1],
                      pos=np.asarray([t]), extra=caches)
        s = np.asarray(next(v for k, v in V.items() if k.endswith("_logits")))
        steps.append(s)
        caches = {}
        for lay in g_dec.decode_layers:
            for kind in ("k", "v"):
                inn = lay["%s_in" % kind].replace(".", "_")
                outn = lay["%s_out" % kind].replace(".", "_")
                arr = np.asarray(V[outn])
                caches[inn] = arr[1:] if arr.shape[0] > 1 else arr
    li_bad = np.concatenate(steps, axis=tok_axis)
    diff_bad = float(np.max(np.abs(lf - li_bad)))
    if diff_bad < tol:
        print("  [FAIL] 丢老 token 的坏 cache 没被抓住（差 %.3e）—— 判据瞎了"
              % diff_bad)
        return 1
    print("  负样（cache 丢最老一步）：最大绝对差 %.3e > 容差  [红 ✓]" % diff_bad)
    return 0 if diff < tol else 1


def main(argv=None):
    argv = argv if argv is not None else _sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    if "--decode-check" in argv:
        rest = [a for a in argv if a != "--decode-check"]
        if not rest:
            print("  [FAIL] --decode-check 要一份 .by1")
            return 2
        return decode_check(rest[0])
    if "--spec" in argv:
        return spec(write="--write" in argv)
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
