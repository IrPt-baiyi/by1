#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 exec -- 第二个后端：纯 NumPy 执行器。

为什么需要它：在它之前只有**一个**后端（by1codegen 里的 PyTorch RUNTIME）。
只有一个后端，"IR 是后端无关的"这句话就**从来没被检验过** ——
它可能只是恰好长得像 PyTorch。

这个执行器：
  · 不 import torch，不和 PyTorch 后端共享任何一行代码
  · 形状完全由 IR 的 attrs + d_model **自己推**（IR 里刻意没有张量形状）
  · 把 IR 当成真正的程序：按 ops 顺序求值 ValueRef，Add 就是 Add

判定：同一份 IR、同一组权重，它和 PyTorch 后端的前向必须一致。
不一致的地方，就是 IR 里漏掉的信息 —— 那才是 3a 要找的东西。

用法:
  python by1exec.py llama-shaped.by1            # 跑一遍并自检
  python by1exec.py llama-shaped.by1 --compare  # 和 PyTorch 后端逐位对比
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import by1paths

# **eps 的默认值只有一个地方写**（`by1ir.EPS_DEFAULT`）。
# 这个文件里原来有 11 处 `1e-5`；那些形状本身没错，
# 错在"哪一处漏了传参"没有结构性征兆 —— 见 by1ir 里那段注释。
try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:                     # 单独拷一个文件出去时兜底
    _DEFAULT_EPS = 1e-5

# **MoE 打分函数的解析和计算只有一份实现**（`by1ir`）——
# 三个后端都从这儿取。这条规则原来在三个后端里各写各的，于是
# `sigmoid_group_topk` 曾经在两个后端里都落进 else（ir.md 130），
# 而这一轮补 `sqrtsoftplus` 时又来了一次：只在 `compile_ir` 里填 `_score`，
# NumPy 这条"直读 IR"的路看不见它，两后端差了 1.6e-02。
# **一条规则三处实现 = 迟早对不上**，所以这里不再自己算。
try:
    from by1ir import moe_score, moe_scores_array
except ImportError:                     # 兜底：和 by1ir 里那两份逐字相同
    _SCORE_FROM_ROUTING = {"sigmoid_topk": "sigmoid",
                           "sigmoid_group_topk": "sigmoid",
                           "softmax_topk": "softmax",
                           "topk_softmax": "softmax"}

    def moe_score(attrs):
        a = attrs or {}
        for key in ("score", "_score"):
            v = a.get(key)
            if v is not None and str(v).strip():
                return str(v).strip().lower()
        return _SCORE_FROM_ROUTING.get(
            str(a.get("routing", "")).strip().lower())

    def moe_scores_array(kind, logits, np):
        if kind == "softmax":
            e = np.exp(logits - logits.max(-1, keepdims=True))
            return e / e.sum(-1, keepdims=True)
        if kind == "sigmoid":
            return 1.0 / (1.0 + np.exp(-logits))
        sp = np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits)))
        if kind == "softplus":
            return sp
        if kind == "sqrtsoftplus":
            return np.sqrt(sp)
        raise ValueError("不认识的打分函数 <%s> —— 不许静默退回 softmax"
                         % (kind,))

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 基础算子（都是裸函数，没有模块/类层次） ────────────────────────

def rms_norm(x, w, eps=1e-5, one_plus=False):
    y = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
    return y * ((1.0 + w) if one_plus else w)


def layer_norm(x, w, b, eps=1e-5, one_plus=False):
    """**LayerNorm 和 RMSNorm 不是一个东西。**

    这里原来只有 rms_norm —— 于是 NumPy 后端算 GPT-2 的时候，
    24 个 LayerNorm bias **根本没有参与运算**，而输出形状完全正常。
    "看起来对但算错"，正是这个项目里最贵的那一类。
    （PyTorch 那边早就两样都有了。）
    """
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    y = y * ((1.0 + w) if one_plus else w)
    return y + b if b is not None else y


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = np.max(np.where(np.isfinite(x), x, -np.inf), axis=axis, keepdims=True)
    e = np.exp(np.where(np.isfinite(x), x - m, -np.inf))
    s = e.sum(axis=axis, keepdims=True)
    return np.where(np.isfinite(x), e / np.maximum(s, 1e-30), 0.0)


def rope_tables(head_dim, n, base, yarn=None):
    """θ 的 cos/sin 表。**公式不在这里** —— 在 `by1graph.rope_inv`。

    2026-10-12 搬家：这段原来住在这里，而 `by1codegen` 另有一份 torch 版、
    图的 rope 原语又把普通那一支内联了一遍 —— **一条规则三处实现**。
    现在图与这里共用一份（YaRN / llama3 两支也在那一处）；
    torch 那一份仍由 `src/modelcheck/by1rope.py` 盯着
    （它对 transformers 的 `_compute_llama3_parameters`）。

    **判据是逐位相同**：搬之前把每个 (dim, base, yarn 类型) 组合的
    cos/sin 表都取出来对过，一个字节都不许变。
    """
    import by1graph
    inv = by1graph.rope_inv(head_dim, base, yarn)
    t = np.arange(n)
    f = np.outer(t, inv)
    return np.cos(f), np.sin(f)


def apply_rope(x, cos, sin, pairing="interleaved"):
    """两种配对约定，**必须和 by1codegen 那份逐字一致**：

      half         对半进（x[i] 配 x[i+d/2]）、拼接着出
      interleaved  奇偶进（x[2i] 配 x[2i+1]）、**拼接着出**（不是交错回去）

    曾经这里写成 np.stack（交错出）而 PyTorch 那边也写成 stack —— 两边"一致地错"，
    所以四个模型的 NumPy↔PyTorch 对拍全是绿的。改了一边之后护栏立刻变红。
    """
    if pairing == "half":
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def _qk_on(v):
    """qk_norm 的取值不只有真假：off / per_head / full。

    "off" 是**真值字符串** —— 直接做真值判断会永远成立，于是给 q/k
    悄悄加一层本不该有的 RMSNorm。PyTorch 那边写的是 not in (off, "", None)，
    所以只有别的后端会错。实测 llama-shaped 的 NumPy↔PyTorch
    从 2.505e-07 变成 3.362e-03。
    """
    if v is None or v is False:
        return False
    return str(v).strip().lower() not in ("off", "", "none", "false", "0")


# ── 每个 op 一个函数：签名统一 (params, attrs, inputs, d_model) ────

def op_norm(P, a, ins, d):
    # **按 kind 分派。** 以前无条件走 rms —— `kind` 这个属性
    # 在 IR 里一直有，这个后端从来没读它。
    if a.get("kind") == "layer":
        return layer_norm(ins[0], P["w"], P.get("b"), a.get("eps", _DEFAULT_EPS),
                          a.get("one_plus", False))
    return rms_norm(ins[0], P["w"], a.get("eps", _DEFAULT_EPS),
                    a.get("one_plus", False))


def op_add(P, a, ins, d):
    return ins[0] + ins[1]


def op_attention(P, a, ins, d):
    x = ins[0]
    b, n, _ = x.shape
    if a.get("qkv_fused"):
        # **融合 q/k/v**（视觉塔那种）：`qkv` 是 [q | k | v] 三截拼起来的
        # 一个张量，`proj` 是输出投影。**只是布局不同，算法一模一样** ——
        # 所以这里把它拆回 `wq`/`wk`/`wv`/`wo`，后面一格不动。
        # 不拆的话形状和产物对不上（clef 契约 `qkv.weight (3456, 1152)`）。
        _hd, _q, _kv = a["head_dim"], a["q"], a["kv"]
        _a, _b = _q * _hd, _kv * _hd
        P = dict(P)
        P["wq"] = P["qkv"][:_a]
        P["wk"] = P["qkv"][_a:_a + _b]
        P["wv"] = P["qkv"][_a + _b:_a + 2 * _b]
        P["wo"] = P["proj"]
        if "qkv.bias" in P:
            P["wq.bias"] = P["qkv.bias"][:_a]
            P["wk.bias"] = P["qkv.bias"][_a:_a + _b]
            P["wv.bias"] = P["qkv.bias"][_a + _b:_a + 2 * _b]
        if "proj.bias" in P:
            P["wo.bias"] = P["proj.bias"]
    q_gate = a.get("q_gate", False)
    w = a["head_dim"] * (2 if q_gate else 1)
    # **两种 qk_norm 的区别是"在哪一步归一化"**（torch 同句）：
    #   per_head  拆头之后，按 head_dim
    #   full      拆头**之前**，按 q*head_dim 整宽（Instella 实测
    #             `q_norm.weight` 是 [2560] 不是 [80]）
    # 这里原来**不分 full**，一律按头算 —— 和契约的 (80,) 一致地错。
    _qfull = str(a.get("qk_norm", "")).strip().lower() == "full"
    _qeps = float(a.get("norm_eps", _DEFAULT_EPS))
    _qop = a.get("norm_one_plus", False)
    qq = x @ P["wq"].T
    if "wq.bias" in P:
        qq = qq + P["wq.bias"]
    if _qfull:
        qq = rms_norm(qq, P["qn.w"], eps=_qeps, one_plus=_qop)
    qq = qq.reshape(b, n, a["q"], w)
    if q_gate:
        q, gate = qq[..., :a["head_dim"]], qq[..., a["head_dim"]:]
        gate = gate.reshape(b, n, -1)
    else:
        q, gate = qq, None
    q = q.transpose(0, 2, 1, 3)
    def _b(y, nm):
        return y + P[nm][None, None, :] if nm in P else y
    kraw = x @ P["wk"].T
    if "wk.bias" in P:
        kraw = kraw + P["wk.bias"]
    if _qfull:
        kraw = rms_norm(kraw, P["kn.w"], eps=_qeps, one_plus=_qop)
    kraw = kraw.reshape(b, n, a["kv"], a["head_dim"]).transpose(0, 2, 1, 3)
    if a.get("kv_tie"):
        v = kraw
    else:
        v = x @ P["wv"].T
        if "wv.bias" in P:
            v = v + P["wv.bias"]
        v = v.reshape(b, n, a["kv"], a["head_dim"]).transpose(0, 2, 1, 3)
    k = kraw
    if _qk_on(a.get("qk_norm")) and not _qfull:
        # **eps 必须传，不能吃默认参数。**
        # 这里原来调 `rms_norm(q, P["qn.w"], one_plus=...)` —— **不传 eps**，
        # 于是吃 `rms_norm` 的默认 1e-5。而 by1codegen 那边读的是
        # `a.get("norm_eps", _DEFAULT_EPS)`。clef-tiny 的 qk_norm 是 per_head、
        # norm_eps 是 1e-6 —— **同一份 IR，两个后端两个答案**
        # （相对差 1.05e-05，低于 1e-4 判据所以一直绿）。
        # 这正是 by1c 的 C_HEAD 里记着"犯过五次"的那一类。
        q = rms_norm(q, P["qn.w"], eps=_qeps, one_plus=_qop)
        k = rms_norm(k, P["kn.w"], eps=_qeps, one_plus=_qop)
    hd, part = a["head_dim"], a.get("rope_partial", 1.0)
    # **`rope` 是一个开关，不是一个常量。**
    # 这一段以前无条件转 —— 而 GPT-2 的 `rope = false`
    # （它的位置信息是外面加的 wpe）。
    # 这个后端**从来没读过这个属性**，而 SHAPED 里那些模型的
    # rope 全是开着的，所以一直看不见。
    #
    # 和 by1codegen 当初那个"Attention 无条件 apply_rope"是同一个 bug，
    # 只是那个是在 PyTorch 后端、且是被 GPT-2 逼出来的。
    _rope_on = bool(a.get("rope", True))
    npr = int(hd * part)
    # **开关要真的管住两个分支。**
    # 若写成 `if not _rope_on: part = 0.0`，则 `npr = 0`
    # 让 `0 < npr < hd` 变假，**直接掉进 else 分支照转不误**。
    # 一个"看起来加了开关"的改动比没加更坏：它让人以为查过了。
    if _rope_on and 0 < npr < hd:
        c, s = rope_tables(npr, n, a["rope_base"], a.get("yarn"))
        if a.get("rope_scale", 1.0) != 1.0:
            c, s = c * a["rope_scale"], s * a["rope_scale"]
        q = np.concatenate([apply_rope(q[..., :npr], c, s, a["rope_pairing"]),
                            q[..., npr:]], -1)
        k = np.concatenate([apply_rope(k[..., :npr], c, s, a["rope_pairing"]),
                            k[..., npr:]], -1)
    elif _rope_on:
        c, s = rope_tables(hd, n, a["rope_base"], a.get("yarn"))
        if a.get("rope_scale", 1.0) != 1.0:
            c, s = c * a["rope_scale"], s * a["rope_scale"]
        q = apply_rope(q, c, s, a["rope_pairing"])
        k = apply_rope(k, c, s, a["rope_pairing"])
    if a["q"] != a["kv"]:
        rep = a["q"] // a["kv"]
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
    idx = np.arange(n)
    m = idx[None, :] <= idx[:, None]
    if a["window"]:
        m = m & (idx[None, :] > idx[:, None] - a["window"])
    att = np.where(m, att, -np.inf)
    if a.get("sink"):
        sk = np.broadcast_to(P["sink"].reshape(1, -1, 1, 1), (b, a["q"], n, 1))
        att = softmax(np.concatenate([att, sk], -1), -1)[..., :-1]
    else:
        att = softmax(att, -1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
    if gate is not None:
        o = o * (1.0 / (1.0 + np.exp(-gate)))
    if a.get("head_gate", "off") != "off":
        # 激活是机制属性：Laguna 用 softplus、Ling 用 sigmoid
        _gg = P["g_proj"] @ x.reshape(-1, d).T
        g = ((1.0 / (1.0 + np.exp(-_gg))) if a.get("gate_act") == "sigmoid"
             else np.log1p(np.exp(_gg))).T.reshape(b, n, -1)
        o = (o.reshape(b, n, a["q"], hd) * g[..., None]).reshape(b, n, -1) \
            if a["head_gate"] == "per_head" else o * g
    o = o @ P["wo"].T
    if "wo.bias" in P:
        o = o + P["wo.bias"]
    return o


def _dsa_index(P, a, x, q_resid):
    """DSA 稀疏索引器：返回 [b, n, n] 的布尔掩码（query t 留下哪些 key）。

    **和 `by1codegen.DSAIndexer`（PyTorch 侧）逐行对应**；判卷人是
    transformers 的 `GlmMoeDsaIndexer`，见 `src/modelcheck/by1sparse.py`。
    三个后端都按这一份数学算 —— **不是各写各的**。

    两处容易抄错，先写在这：

    ① **rope 那一半在前**：`split([qk_rope, hd - qk_rope])`，和主 MLA 的
       `[qk_nope | qk_rope]` **正好相反**。
    ② **索引器的 rope 配对约定固定是 `interleaved`**，和主 MLA 的 `pairing`
       无关 —— 参考实现的注释就写着 "the indexer applies interleaved RoPE"。
       实测 interleaved 4/4 组和参考逐个相同，half 只在 rope 不起作用时碰巧一样。

    **并列怎么取 —— 这一条是量出来的，不是想出来的。**
    DSA 打分要过 ReLU，所以大量分数**恰好是 0**（实测：48 个 key 里有 49 个
    是精确的 0），于是 top-k 的边界上很容易撞并列。实测 44 个"有得挑"的
    query 里，**1 个第 4/5 名间隙正好是 0，4 个间隙小于 1e-7**。
    **并列处挑谁没有定义** —— 参考那边是 `torch.topk` 的行为，
    这里是 `argpartition` 的行为，两个都会"挑出一批"，而**不是同一批**。
    后果不是小数值差：挑中的 key 不同，注意力输出当场不同（实测 2.7e-01）。
    所以定死一条：**并列取较小的 key 下标**（稳定排序）。
    三个后端都按这条算，"三个后端一致"才是个有意义的判据。
    """
    b, n, _ = x.shape
    nh, hd = int(a["index_heads"]), int(a["index_dim"])
    topk = int(a["index_topk"])
    nr = int(a.get("qk_rope") or 0)

    q = (q_resid @ P["indexer.wq_b.weight"].T).reshape(b, n, nh, hd)
    q_rot, q_pass = q[..., :nr], q[..., nr:]
    # **k 过 LayerNorm（带 bias，eps=1e-6），不是 RMSNorm** —— 参考就是这么写的
    k = layer_norm(x @ P["indexer.wk.weight"].T,
                   P["indexer.k_norm.weight"], P["indexer.k_norm.bias"], 1e-6)
    k_rot, k_pass = k[..., :nr], k[..., nr:]
    if nr:
        c, s = rope_tables(nr, n, a["rope_base"])
        q_rot = apply_rope(q_rot.transpose(0, 2, 1, 3), c, s,
                           "interleaved").transpose(0, 2, 1, 3)
        k_rot = apply_rope(k_rot, c, s, "interleaved")
    q = np.concatenate([q_rot, q_pass], -1)            # [b, n, nh, hd]
    k = np.concatenate([k_rot, k_pass], -1)            # [b, n, hd]
    if int(a.get("index_kpool") or 1) > 1:
        return _dsa_index_pool(P, a, x, q, k)
    # [b,n,nh,hd] @ [b,1,hd,n] -> [b,n,nh,n]（每头一个分数）
    scores = np.maximum(q @ k.transpose(0, 2, 1)[:, None], 0.0) * (hd ** -0.5)
    w = (x @ P["indexer.weights_proj.weight"].T) * (nh ** -0.5)  # [b, n, nh]
    sc = (w[..., None, :] @ scores).squeeze(-2)          # [b, n, n]
    ar = np.arange(n)
    sc = np.where(ar[None, :] > ar[:, None], -np.inf, sc)   # 因果
    t = min(topk, n)
    keep = np.zeros((b, n, n), dtype=bool)
    if t >= n:
        keep[:] = True
    else:
        for bi in range(b):
            for ti in range(n):
                # **并列取较小的 key 下标** —— 见下面那段注释。
                sel = np.argsort(-sc[bi, ti], kind="stable")[:t]
                keep[bi, ti, sel] = True
    return keep


def _dsa_index_pool(P, a, x, q, k):
    """k-pool：**池**打分，选中的池摊回原始 token。返回 [b, n, n] 布尔掩码。

    **和 `by1codegen.DSAIndexer._pool_select`（PyTorch 侧）逐行对应**，
    判卷人是 transformers 5.19.0 的 `Glm5NextTextIndexer`
    （refs/glm5-next-indexer.reference.txt）。三后端都按这一份数学算。

    两处容易漏：池从**第一个有效 token** 起算（不是槽位 0）；
    池可选当且仅当**它最后一个 token 对 query 可见**，且**整个池都有效**。
    """
    b, n, _ = x.shape
    P_ = int(a["index_kpool"])
    tail = bool(a.get("index_kpool_tail"))
    nh, hd = int(a["index_heads"]), int(a["index_dim"])
    topk = int(a["index_topk"])
    gate = x @ P["indexer.index_kpool_compress_gate"].T        # [b, n, hd]
    ape = P["indexer.index_kpool_compress_ape"]                # [P, hd]
    valid = np.ones((b, n), dtype=bool)
    first = np.where(valid.any(-1), valid.argmax(-1), n)
    np_ = (n + P_ - 1) // P_
    off = np.arange(np_ * P_).reshape(1, np_, P_)
    pidx = first[:, None, None] + off                          # [b, np, P_]
    bidx = np.arange(b).reshape(b, 1, 1)
    safe = np.clip(pidx, 0, n - 1)
    gk = k[bidx, safe]                                         # [b, np, P_, hd]
    gg = gate[bidx, safe]
    gv = valid[bidx, safe] & (pidx < n)
    pool_valid = gv.all(-1)
    pidx = np.where(gv, pidx, -1)
    lg = gg.astype(np.float64) + ape.astype(np.float64)[None, None]
    lg = np.where(gv[..., None], lg, -np.inf)
    e = np.exp(lg - lg.max(axis=2, keepdims=True))
    e = np.where(np.isfinite(lg), e, 0.0)
    prob = e / np.maximum(e.sum(axis=2, keepdims=True), 1e-30)
    pool_key = (prob * gk).sum(axis=2)                          # [b, np, hd]
    keepm = pool_valid.any(0)
    pool_key, pidx, pool_valid = pool_key[:, keepm], pidx[:, keepm], pool_valid[:, keepm]
    s = np.maximum(q @ pool_key.transpose(0, 2, 1)[:, None], 0.0) * (hd ** -0.5)
    w = (x @ P["indexer.weights_proj.weight"].T) * (nh ** -0.5)
    sc = (w[..., None, :] @ s).squeeze(-2)                      # [b, n, np']
    pend = np.clip(pidx[..., -1], 0, n - 1)
    ar = np.arange(n)
    vis = pend[:, None, :] <= ar[None, :, None]
    cand = vis & pool_valid[:, None, :]
    sc = np.where(cand, sc, -np.inf)
    sk = min(topk // P_, sc.shape[-1])
    keep = np.zeros((b, n, n), dtype=bool)
    for bi in range(b):
        for ti in range(n):
            sel = np.argsort(-sc[bi, ti], kind="stable")[:sk]
            for s2 in sel:
                for v in pidx[bi, s2]:
                    if v >= 0 and cand[bi, ti, s2]:
                        keep[bi, ti, v] = True
    if tail and P_ > 1:
        bidx2 = np.arange(b).reshape(b, 1)
        pos = ar
        tv = (pos[None, :, None] >= pos[None, None, :]) & valid[:, None, :]
        vc = tv.sum(-1)
        tc = vc % P_
        offs = np.arange(P_ - 1)
        tstart = first[:, None] + vc - tc
        tidx = tstart[..., None] + offs
        tok = (offs[None, None, :] < tc[..., None]) & (tidx < n)
        kv = np.clip(tidx, 0, n - 1)
        tvis = np.take_along_axis(tv, kv, axis=-1)
        tidx = np.where(tok & tvis, tidx, -1)
        for bi in range(b):
            for ti in range(n):
                for v in tidx[bi, ti]:
                    if v >= 0:
                        keep[bi, ti, v] = True
    return keep


def op_mla(P, a, ins, d):
    """MLA：低秩压缩的注意力。和 PyTorch 后端逐行对应。

    带 `index_heads` 的时候是 **SparseMLA**：先用 DSA 索引器挑出 top-k 个 key，
    注意力只在被挑中的那些上算。索引器的数学见 `by1codegen.DSAIndexer`，
    判卷人是 transformers 的 `GlmMoeDsaIndexer`（`by1sparse.py`）。
    """
    x = ins[0]
    b, n, _ = x.shape
    nh, nope, nr = a["q"], a["qk_nope"], a["qk_rope"]
    vd, kvl = a["v_dim"], a["kv_lora"]
    _op = a.get("norm_one_plus", False)
    _eps = a.get("norm_eps", _DEFAULT_EPS)

    q_resid = rms_norm(x @ P["q_a_proj"].T, P["q_a_layernorm.w"], _eps, _op)
    q = (q_resid @ P["q_b_proj"].T).reshape(b, n, nh, nope + nr).transpose(0, 2, 1, 3)
    q_pass, q_rot = q[..., :nope], q[..., nope:]

    ckv = x @ P["kv_a_proj_with_mqa"].T
    c_kv, k_rot = ckv[..., :kvl], ckv[..., kvl:]
    kv = rms_norm(c_kv, P["kv_a_layernorm.w"], _eps, _op) @ P["kv_b_proj"].T
    kv = kv.reshape(b, n, nh, nope + vd).transpose(0, 2, 1, 3)
    k_pass, v = kv[..., :nope], kv[..., nope:]

    # k_rot 是单头的：RoPE 只作用在这 nr 维上，然后广播到所有头
    k_rot = k_rot.reshape(b, 1, n, nr)
    c, s = rope_tables(nr, n, a["rope_base"])
    q_rot = apply_rope(q_rot, c, s, a.get("pairing", "interleaved"))
    k_rot = apply_rope(k_rot, c, s, a.get("pairing", "interleaved"))
    k_rot = np.broadcast_to(k_rot, k_pass.shape[:-1] + (nr,))

    qq = np.concatenate([q_pass, q_rot], -1)
    kk = np.concatenate([k_pass, k_rot], -1)
    att = (qq @ kk.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(nope + nr))
    idx = np.arange(n)
    causal = idx[None, :] <= idx[:, None]
    if a.get("index_heads"):
        keep = _dsa_index(P, a, x, q_resid)          # [b, n, n] 布尔
        causal = causal[None] & keep
    att = softmax(np.where(causal, att, -np.inf), -1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, nh * vd)

    if a.get("head_gate", "off") != "off":
        g = P["g_proj"] @ x.reshape(-1, d).T
        g = (1.0 / (1.0 + np.exp(-g))) if a.get("gate_act") == "sigmoid" \
            else np.log1p(np.exp(g))
        o = o.reshape(b, n, nh, vd) * g.T.reshape(b, n, nh, 1)
        o = o.reshape(b, n, nh * vd)

    o = o @ P["dense"].T
    if a.get("bias") and "dense.bias" in P:
        o = o + P["dense.bias"]
    return o


def _act_apply(v, style):
    """**激活要分派，不能写死 silu。**

    这里原来那条无门控分支写死 `silu` —— 和 by1codegen 那边当初
    一模一样的毛病：一条分支只为"有门 + silu"那一种情况写过。
    GPT-2 是无门控 + gelu_new，于是拿到的是 silu，
    而 `.by1` 里写的 `act` 一声不吭地被忽略。

    **而且是三个后端里最后一个修好的** —— PyTorch 那边改过了，
    C 那边刚改，这里是漏的那个。同一个 bug 在三处，
    改的时候只改了一处。
    """
    if style in ("gelu_new", "gelu_tanh", "tanh"):
        return 0.5 * v * (1.0 + np.tanh(0.7978845608028654
                                        * (v + 0.044715 * v ** 3)))
    if style == "gelu":
        from math import erf
        _e = np.vectorize(erf)
        return 0.5 * v * (1.0 + _e(v * 0.7071067811865476))
    if style == "relu2":
        return np.maximum(v, 0.0) ** 2
    return silu(v)


def _bias(y, P, key):
    """**bias 也要认。** 这里以前一条都没有 —— GPT-2 的 4×12 个 bias
    在后端层面根本不存在，而"张量搬运 173/173"看起来是满的。"""
    return y + P[key] if key in P else y


def op_ffn(P, a, ins, d):
    x = ins[0]
    _act = a.get("act", "silu")
    g = _bias(x @ P["w1"].T, P, "w1.bias")
    if not a.get("gate", True):
        # 无门控：**一块上投影**。这就是 GPT-2 的 MLP。
        return _bias(_act_apply(g, _act) @ P["w2"].T, P, "w2.bias")
    u = _bias(x @ P["w3"].T, P, "w3.bias")
    if _act == "gptoss":
        lim = a.get("limit")
        if lim is not None:
            g = np.minimum(g, lim)
            u = np.clip(u, -lim, lim)
        h = (u + 1) * (g / (1.0 + np.exp(-a.get("alpha", 1.702) * g)))
    else:
        h = _act_apply(g, _act) * u
    return _bias(h @ P["w2"].T, P, "w2.bias")


def op_moe(P, a, ins, d):
    x = ins[0]
    b, n, _ = x.shape
    xf = x.reshape(-1, d)
    logits = xf @ P["router"].T
    if a.get("router_bias") and "router.bias" in P:
        logits = logits + P["router.bias"]
    sb = P.get("score_bias") if a.get("score_bias") else None

    def _topk(v, k):
        """取每行前 k 大的下标。**并列取较小下标**（稳定排序）——
        和稀疏索引器同一个理由：并列处取谁没有定义，两个后端会挑出不同的，
        而那不是 bug。见 history/ir.md 129。"""
        return np.argsort(-v, axis=-1, kind="stable")[:, :k]

    # ── 四种路由，四种选法。**这四条各是各的机制** ────────────────
    # 这里原来只有 topk_softmax / softmax_topk 两支，而 `sigmoid_topk`
    # 和 `sigmoid_group_topk` **落进了 else**，走成 softmax_topk ——
    # 也就是说 Step-3.7 和 Ling/DeepSeek 的路由**一直在算另一个模型**。
    # 实测（把 routing 换掉再对拍 PyTorch）：
    #     topk_softmax        3.0e-07  过
    #     softmax_topk        2.6e-07  过
    #     sigmoid_topk        3.8e-02  不过     <- 落进 else
    #     sigmoid_group_topk  6.4e-01  不过     <- 落进 else
    # 门里的 shaped 模型只用了前两种，所以从来没红过。见 history/ir.md 130。
    rt = (a.get("routing") or "softmax_topk")
    # **打分函数：和选法正交。** 解析走 `by1ir.moe_score` 那一份实现
    # （`by1codegen` 也从它取）—— 这里**不能**读编译期填的 `_score`：
    # 这一条走的是 IR 直读，`compile_ir` 根本没跑过，`_score` 不存在。
    # 第一版就是只在 `compile_ir` 里填 `_score`，于是 torch 走 sqrtsoftplus、
    # NumPy 退回"按前缀猜"走 sigmoid，两后端差 1.6e-02 ——
    # **正是这条规则不该有两份实现的原因。**
    _sc = moe_score(a)
    if _sc is None:
        raise ValueError(
            "MoE 的打分函数推不出来：routing = <%r> 不在闭集里，"
            "而 score 也没写 —— **不许静默退回 softmax**" % (rt,))
    if rt == "sigmoid_group_topk":
        # noaux_tc（Ling / DeepSeek V3 / GLM / **DeepSeek V4**）：
        # **先打分**再分组 top-k。
        # 关键：`score_bias` **只影响选择**，权重用未加偏置的 sc。
        ng, tg = a.get("n_group") or 0, a.get("topk_group") or 0
        if not ng or not tg:
            raise ValueError("routing = sigmoid_group_topk 需要 n_group 与 topk_group")
        sc = moe_scores_array(_sc, logits, np)
        choice = sc + sb if sb is not None else sc
        gs = choice.reshape(-1, ng, choice.shape[-1] // ng)
        # 每组取前 2 之和排名，选 topk_group 个组
        gs = np.sort(gs, -1)[..., -2:].sum(-1)
        gi = _topk(gs, tg)
        gm = np.zeros_like(gs)
        np.put_along_axis(gm, gi, 1.0, axis=-1)
        mask = np.repeat(gm[:, :, None], choice.shape[-1] // ng, 2)
        choice = np.where(mask.reshape(choice.shape) > 0, choice, -np.inf)
        topi = _topk(choice, a["top_k"])
        tv = np.take_along_axis(sc, topi, -1)         # **未加偏置**的分数
        topv = tv / (tv.sum(-1, keepdims=True) + 1e-20)
    elif rt == "sigmoid_topk":
        # **先打分**再直接 top-k，不分组（Step-3.7）。
        # 和 sigmoid_group_topk 只差"先选组"那一步，名字只差一个词。
        sc = moe_scores_array(_sc, logits, np)
        choice = sc + sb if sb is not None else sc
        topi = _topk(choice, a["top_k"])
        tv = np.take_along_axis(sc, topi, -1)
        topv = tv / (tv.sum(-1, keepdims=True) + 1e-20)
    elif rt == "topk_softmax":
        # 先 top-k，再只对这 k 个 logits 做 softmax（GPT-OSS）。
        # **选和权重都用 `logits`，不看 score_bias** —— 参考就是这么写的。
        # （原来这里用的是加了偏置的 sel，两处都不一样。）
        # **这一支不经过打分函数** —— 配非 softmax 的 score 是矛盾。
        if _sc != "softmax":
            raise ValueError(
                "routing = topk_softmax 不经过打分函数，score = <%r> 不会被读 "
                "—— 要么去掉，要么换 sigmoid_topk" % (_sc,))
        topi = _topk(logits, a["top_k"])
        topv = softmax(np.take_along_axis(logits, topi, -1), -1)
    else:
        # 全体 softmax -> top-k -> 重新归一（Mixtral）。
        # `score_bias` **只影响选择**：权重要重算，用未加偏置的 logits。
        # 同上：这一支的 score 就是 softmax 本身。
        if _sc != "softmax":
            raise ValueError(
                "routing = softmax_topk 不经过打分函数，score = <%r> 不会被读 "
                "—— 要么去掉，要么换 sigmoid_topk" % (_sc,))
        sel = logits + sb if sb is not None else logits
        topi = _topk(softmax(sel, -1), a["top_k"])
        base = logits if sb is not None else sel
        tv = np.take_along_axis(softmax(base, -1), topi, -1)
        topv = tv / np.maximum(tv.sum(-1, keepdims=True), 1e-9)
    topv = topv * a.get("routed_scale", 1.0)
    out = np.zeros_like(xf)
    for e in range(a["experts"]):
        hit = topi == e
        rows = hit.any(-1)
        if not rows.any():
            continue
        xe = xf[rows]
        g = xe @ P["w1"][e].T
        u = xe @ P["w3"][e].T
        if a.get("expert_bias") and "b1" in P:
            g = g + P["b1"][e]
            u = u + P["b3"][e]
        if a.get("act") == "gptoss":
            lim = a.get("limit")
            if lim is not None:
                g = np.minimum(g, lim)
                u = np.clip(u, -lim, lim)
            h = (u + 1) * (g / (1.0 + np.exp(-a.get("alpha", 1.702) * g)))
        else:
            # **激活要分派，不能写死 silu** —— `act = relu2`（Nemotron 的
            # MoE）原来在这里被算成 silu，而 PyTorch 的 `_swiglu` 也写死
            # silu：**两个后端一致，而两个都错**。现在走 `_act_apply`
            # （和 op_ffn 同一份）。
            h = _act_apply(g, a.get("act", "silu")) * u
        ye = h @ P["w2"][e].T
        if a.get("expert_bias") and "b2" in P:
            ye = ye + P["b2"][e]
        out[rows] += topv[rows][hit[rows]][:, None] * ye
    if a.get("shared"):
        # **共享专家走同一个激活与夹取值**（torch 的
        # `_swiglu(sw1·x, sw3·x, style, limit_shared, alpha)`）——
        # 这里原来写死 silu 且不夹：gptoss + 共享的模型两边会对不上。
        _g2, _u2 = xf @ P["sw1"].T, xf @ P["sw3"].T
        if a.get("act") == "gptoss":
            _lm = a.get("limit_shared")
            if _lm is not None:
                _g2 = np.minimum(_g2, _lm)
                _u2 = np.clip(_u2, -_lm, _lm)
            sh = ((_u2 + 1)
                  * (_g2 / (1.0 + np.exp(-a.get("alpha", 1.702) * _g2))))
        else:
            sh = _act_apply(_g2, a.get("act", "silu")) * _u2
        se = sh @ P["sw2"].T
        if a.get("shared_gate") and "shared_gate" in P:
            se = se * (1.0 / (1.0 + np.exp(-(xf @ P["shared_gate"].T))))
        out = out + se
    return out.reshape(b, n, d)


def op_linear(P, a, ins, d):
    """GDN：短卷积 + delta 规则 + 门控归一化。状态由后端自己持有。"""
    x = ins[0]
    b, s, _ = x.shape
    nk, nv, dk, dv = a["k_heads"], a["v_heads"], a["k_dim"], a["v_dim"]
    rep, kd, vd = nv // nk, nk * a["k_dim"], nv * a["v_dim"]
    qkvz = (x @ P["in_proj_qkvz"].T).reshape(b, s, nk, 2 * dk + 2 * dv * rep)
    ba = (x @ P["in_proj_ba"].T).reshape(b, s, nk, 2 * rep)
    q, kk, v, z = np.split(qkvz, [dk, 2 * dk, 2 * dk + dv * rep], axis=3)
    bb, aa = np.split(ba, [rep], axis=3)
    v = v.reshape(b, s, nv, dv)
    z = z.reshape(b, s, nv, dv)
    bb = bb.reshape(b, s, nv)
    aa = aa.reshape(b, s, nv)
    mix = np.concatenate([q.reshape(b, s, kd), kk.reshape(b, s, kd),
                          v.reshape(b, s, vd)], -1)
    # 深度可分离因果卷积
    pad = np.zeros((b, a["conv_kernel"] - 1, mix.shape[-1]))
    mp = np.concatenate([pad, mix], 1)
    conv = np.zeros_like(mix)
    for i in range(a["conv_kernel"]):
        conv += mp[:, i:i + s, :] * P["conv"][:, i][None, None, :]
    mix = silu(conv)
    q, kk, v = np.split(mix, [kd, 2 * kd], -1)
    q = q.reshape(b, s, nk, dk)
    kk = kk.reshape(b, s, nk, dk)
    v = v.reshape(b, s, nv, dv)
    beta = 1.0 / (1.0 + np.exp(-bb))
    g = -np.exp(P["A_log"].astype(np.float64)) * np.log1p(
        np.exp(aa.astype(np.float64) + P["dt_bias"]))
    if rep > 1:
        q = np.repeat(q, rep, axis=2)
        kk = np.repeat(kk, rep, axis=2)
    eps = a.get("l2_eps", 1e-6)
    q = q * (1.0 / np.sqrt((q * q).sum(-1, keepdims=True) + eps))
    kk = kk * (1.0 / np.sqrt((kk * kk).sum(-1, keepdims=True) + eps))
    q, kk, v, beta, g = [t.transpose(0, 2, 1, 3).astype(np.float64)
                         for t in (q, kk, v, beta[..., None], g[..., None])]
    q = q * (dk ** -0.5)
    st = np.zeros((b, nv, dk, dv))
    out = np.zeros((b, nv, s, dv))
    for i in range(s):
        st = st * np.exp(g[:, :, i])[..., None]
        kvi = (st * kk[:, :, i][..., None]).sum(-2)
        delta = (v[:, :, i] - kvi) * beta[:, :, i]
        st = st + kk[:, :, i][..., None] * delta[..., None, :]
        out[:, :, i] = (st * q[:, :, i][..., None]).sum(-2)
    out = out.transpose(0, 2, 1, 3).reshape(b, s, nv, dv)
    var = out.astype(np.float64).__pow__(2).mean(-1, keepdims=True)
    y = out * (1.0 / np.sqrt(var + a.get("norm_eps", _DEFAULT_EPS)))
    y = (P["norm.w"] * y) * silu(z)
    return y.reshape(b, s, vd) @ P["out_proj"].T


def op_kda(P, a, ins, d):
    """**KDA（Kimi Delta Attention）** —— delta 规则 + 逐维衰减门。

    ⚠ **递推和 GDN 是同一套** ✓ —— 代码gen 那边更是**直接调
    `gated_delta_rule`** ✓，一行都没重写 ✓。这里照抄它的数值路径，
    因为 NumPy 这一侧本来就是"另一个后端"，不是"另一份实现" ✓。

    和 GDN 的四处差别（全部从 fla 源码量出来，见 `gpu/by1kda.py`）：

      ① q / k / v 是**分开**的投影，且各自一条深度卷积
      ② `dt_bias` 按**维**（`nv * dk`），GDN 按头 —— 所以 `g` 是
         `[b, s, nv, dk]` 而不是 `[b, s, nv, 1]` ✗。**写错了不会报错** ✓。
      ③ 多一个输出门 `g_proj`，喂给 `o_norm`（GDN 用的是融在 qkvz 里的 z）
      ④ 门可以是**低秩两层**（`f_a`/`f_b`，GLM），也可以一层（`f_proj`，Ling）
    """
    x = ins[0]
    b, s, _ = x.shape
    nk, nv = a["k_heads"], a["v_heads"]
    dk, dv = a["k_dim"], a["v_dim"]
    kd, vd = nk * dk, nv * dv
    ck = a["conv_kernel"]

    def dconv(w, z):
        """深度可分离因果卷积。和 GDN 那支同样的写法（同一个意思不写两遍）。"""
        pad = np.zeros((b, ck - 1, z.shape[-1]))
        mp = np.concatenate([pad, z], 1)
        c = np.zeros_like(z)
        for i in range(ck):
            c += mp[:, i:i + s, :] * w[:, i][None, None, :]
        return silu(c)

    q = dconv(P["q_conv1d"], x @ P["q_proj"].T).reshape(b, s, nk, dk)
    kk = dconv(P["k_conv1d"], x @ P["k_proj"].T).reshape(b, s, nk, dk)
    v = dconv(P["v_conv1d"], x @ P["v_proj"].T).reshape(b, s, nv, dv)

    # 门：一层还是两层，由契约里有没有 f_b_proj 决定 —— **不猜**
    if "f_b_proj" in P:
        fg = (x @ P["f_a_proj"].T) @ P["f_b_proj"].T
        gtw = (x @ P["g_a_proj"].T) @ P["g_b_proj"].T
        # **bias 是可选的** ✓ —— fla 的层有（`bias=True`）✓，
        # 而 GLM 的官方权重清单里**没有** `g_b_proj.bias` ✓。
        # 原来这里是无条件加的 ✗，于是"没有 bias"那份契约会
        # `ndarray + None` 直接炸 ✓ —— 或者更糟：如果有人给它补一个
        # 全零的占位，它会**静默算对** ✓，然后永远没人发现契约是错的 ✗。
        if P.get("g_b_proj.bias") is not None:
            gtw = gtw + P["g_b_proj.bias"]
    else:
        fg = x @ P["f_proj"].T
        gtw = x @ P["g_proj"].T
    fg = fg.reshape(b, s, nv, dk)
    gtw = gtw.reshape(b, s, nv, dv)

    beta = 1.0 / (1.0 + np.exp(-(x @ P["b_proj"].T)))
    A = P["A_log"].astype(np.float64)
    dtb = P["dt_bias"].reshape(nv, dk)
    if a.get("gate_lower") is not None:
        # lower_bound 那一支：g = lower * sigmoid(exp(A_log) * g)
        g = float(a["gate_lower"]) * (
            1.0 / (1.0 + np.exp(-(np.exp(A)[None, None, :, None]
                                  * (fg.astype(np.float64)
                                     + dtb[None, None, :, :])))))
    else:
        # g = -exp(A_log) * softplus(g + dt_bias)（fla/ops/kda/gate.py:50-54）
        g = -np.exp(A)[None, None, :, None] * np.log1p(
            np.exp(fg.astype(np.float64) + dtb[None, None, :, :]))

    eps = a.get("l2_eps", 1e-6)
    q = q * (1.0 / np.sqrt((q * q).sum(-1, keepdims=True) + eps))
    kk = kk * (1.0 / np.sqrt((kk * kk).sum(-1, keepdims=True) + eps))
    q, kk, v, beta, g = [t.transpose(0, 2, 1, 3).astype(np.float64)
                         for t in (q, kk, v, beta[..., None], g)]
    # **GVA：q/k 的头数可以少于 v 的头数** ✓ —— 和 codegen 那边
    # `gated_delta_rule` 里那句 `repeat_interleave` **同一个意思** ✓
    # （`fla/ops/kda/naive.py:52-53`）。两个后端都得有 ✓，
    # 否则 `--compare` 会把"两边都错"读成"两边一致" ✗。
    if nv != nk:
        if nv % nk:
            raise ValueError('v_heads (%d) 必须是 k_heads (%d) 的整数倍'
                             % (nv, nk))
        q = np.repeat(q, nv // nk, axis=1)
        kk = np.repeat(kk, nv // nk, axis=1)
    q = q * (dk ** -0.5)
    st = np.zeros((b, nv, dk, dv))
    out = np.zeros((b, nv, s, dv))
    for i in range(s):
        st = st * np.exp(g[:, :, i])[..., None]
        kvi = (st * kk[:, :, i][..., None]).sum(-2)
        delta = (v[:, :, i] - kvi) * beta[:, :, i]
        st = st + kk[:, :, i][..., None] * delta[..., None, :]
        out[:, :, i] = (st * q[:, :, i][..., None]).sum(-2)
    out = out.transpose(0, 2, 1, 3).reshape(b, s, nv, dv)

    var = out.astype(np.float64).__pow__(2).mean(-1, keepdims=True)
    y = out * (1.0 / np.sqrt(var + a.get("norm_eps", _DEFAULT_EPS)))
    # **输出门的激活是 `sigmoid`，不是 `silu`** ✓ ——
    # `fla/layers/kda.py:191` 的 `activation="sigmoid"` ✓，
    # 而它在核里就是 `y = y * sigmoid(g)` ✓
    # （`fla/modules/fused_norm_gate.py:104`）✓。
    # 这里原来写的是 `silu(gtw)` ✗ —— 和 PyTorch 那边**错得一模一样** ✓，
    # 所以 `--compare` 一直是绿的 ✓：**两个后端一致，而两个都错** ✗。
    y = (P["o_norm.w"] * y) * (1.0 / (1.0 + np.exp(-gtw.astype(np.float64))))
    return y.reshape(b, s, vd) @ P["o_proj"].T


def op_ssm(P, a, ins, d):
    """**选择性状态空间**（Mamba2 · Nemotron-H 那一版）。

    和注意力是不同的算法族 —— 没有 q/k/v，状态由 `A_log` / `D` /
    `dt_bias` 驱动、随 token 递推：

        h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t ⊗ B_t)
        y_t = C_t · h_t + D * x_t

    **判卷人**：`src/modelcheck/by1ssm.py` —— 对着官方
    `NemotronHMamba2Mixer` 验过（两种尺寸 + 三条反例）。

    状态是 `[b, heads, head_dim, ssm_state]`，所以两处最容易写反
    （写反了会炸在维度上，不会静默算错）：

        更新是**外积**   x[head_dim] ⊗ B[ssm_state]
        读出是 **h 右乘 C**
    """
    x = ins[0]
    b, s, _ = x.shape
    h, hd = a["heads"], a["head_dim"]
    ng, ssm, k = a["n_groups"], a["ssm_state"], a["conv_kernel"]
    di = h * hd
    cd = di + 2 * ng * ssm
    rep = h // ng

    proj = x @ P["in_proj"].T
    gate, bc, dt = np.split(proj, [di, di + cd], -1)

    # ② 因果深度卷积 + 激活
    pad = np.zeros((b, k - 1, cd))
    mp = np.concatenate([pad, bc], 1)
    conv = np.zeros_like(bc)
    for i in range(k):
        conv += mp[:, i:i + s, :] * P["conv"][:, i][None, None, :]
    if a.get("conv_bias"):
        conv = conv + P["conv.bias"]
    bc = silu(conv)

    xs, B, C = np.split(bc, [di, di + ng * ssm], -1)
    xs = xs.reshape(b, s, h, hd)
    B = np.repeat(B.reshape(b, s, ng, ssm), rep, axis=2)
    C = np.repeat(C.reshape(b, s, ng, ssm), rep, axis=2)

    # ③ SSM 递推
    A = -np.exp(P["A_log"].astype(np.float64))
    dtt = np.log1p(np.exp(dt.astype(np.float64) + P["dt_bias"]))   # softplus
    st = np.zeros((b, h, hd, ssm))
    out = np.zeros((b, h, s, hd))
    for i in range(s):
        d_i = dtt[:, i][..., None, None]
        st = (np.exp(d_i * A[None, :, None, None]) * st
              + d_i * (xs[:, i][..., None] * B[:, i][:, :, None, :]))
        out[:, :, i] = ((st @ C[:, i][..., None])[..., 0]
                        + P["D"][None, :, None] * xs[:, i])
    y = out.transpose(0, 2, 1, 3).reshape(b, s, di)

    # ④ 按组 gated norm（**gate 乘在 norm 之前**），再输出投影
    y = y * silu(gate)
    gs = di // ng
    g = y.reshape(b, s, di // gs, gs)
    g = g * (1.0 / np.sqrt((g * g).mean(-1, keepdims=True)
                           + a.get("norm_eps", _DEFAULT_EPS)))
    y = P["norm.w"] * g.reshape(b, s, di)
    return y @ P["out_proj"].T


def op_external(P, a, ins, d):
    """**逃生舱第二层：调外部符号。**

    和别的算子**受同样的约束** —— 它的张量照样从 P 里取（`by1verify`
    已经对着产物查过了），输入输出照旧接进数据流。
    区别只是这段计算不在这门语言里。
    """
    import by1ext
    # **`ins` 已经是数组了**，不是引用 —— 别的算子都这么收。
    # 第一版写成 `P["_v"][ins[0]]`，于是永远拿不到输入。
    x = np.asarray(ins[0], dtype=np.float32)
    names = by1ext.sorted_weight_names(a["weights"].keys())
    fn = by1ext.bind(by1ext.load_lib(a["lib"], P.get("_base")), a["symbol"])
    B, T, D = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    ws = [np.ascontiguousarray(P["ext:" + nm], dtype=np.float32)
          for nm in names]
    by1ext.call(fn, np.ascontiguousarray(x, dtype=np.float32), y,
                int(B), int(T), int(D), ws)
    return y


OPS = {"Norm": op_norm, "Add": op_add, "Attention": op_attention,
       "FFN": op_ffn, "MoE": op_moe, "Linear": op_linear, "MLA": op_mla,
       "SSM": op_ssm, "KDA": op_kda,
       "External": op_external}


# ── 执行器：按 IR 求值，不认机制名，只认 kind ──────────────────────

def shapes_of(ir):
    """**唯一定义已搬到 `by1ir.params_of`**（1.0.0-rc0 的 T0.2）——
    这里只是向后兼容的别名，后端们调的还是这个名字。

    搬家的理由：张量形状表是 **IR 规格的一部分**，而它原来住在 numpy
    后端里 —— 于是"形状的真相"有两个家（模型 `tensors{}` 的契约声明
    + 这张表），gemma-4-31B 的全局宽度 bug（契约 5376、推成 1152）
    就是两个家不一致漏出来的。**单一家之后，这类漂移只能是编译错误。**"""
    from by1ir import params_of
    return params_of(ir)


def forward_graph(ir, params, ids):
    """**图路径**（A3-3 的 by1exec 吃图）：lower_model + run_graph。

    语义唯一在 `by1graph.PRIMS` 的 ref() —— 这条路不带任何自己的算子，
    所以"NumPy 后端的语义"和"图的语义"不可能再漂移（判卷人是
    `--compare --from-graph`：图求值 vs PyTorch 官方路径逐位）。

    迁移样板照 `by1c.main --from-graph`：图路径与 OPS 路径并行，
    默认仍是 OPS（迁移期间的判卷人）。**降不动的模型明确报错**，
    不静默回退到 OPS —— 静默回退 = "我验了图路径"其实没验。
    """
    import by1graph as _bg
    g = _bg.lower_model(ir, "model")
    a = np.asarray(ids)
    if a.ndim == 2:
        if a.shape[0] != 1:
            raise ValueError("图路径只支持 batch=1（got %s）——批维还没进图约定"
                             % (a.shape,))
        a = a[0]
    V = _bg.run_graph(g, {k: np.asarray(v) for k, v in params.items()}, a)
    logits = np.asarray(next(v for k, v in V.items() if k.endswith("_logits")))
    return logits[None, ...] if logits.ndim == 2 else logits


class Exec:
    def __init__(self, ir, params, from_graph=False):
        self.ir = ir
        self.from_graph = from_graph
        self._flat = {k: np.asarray(v) for k, v in params.items()}
        self.P = {}
        for L in ir["layers"]:
            self.P[L["index"]] = {}
            for j, o in enumerate(L["ops"]):
                pre = f"layers.{L['index']}.op{j}."
                self.P[L["index"]][j] = {
                    k[len(pre):]: np.asarray(v)
                    for k, v in params.items() if k.startswith(pre)}

    def __call__(self, ids):
        if self.from_graph:
            return forward_graph(self.ir, self._flat, ids)
        d = self.ir["d_model"]
        x = self.params_global("embed.weight")[ids]
        # **学习式位置表。** 这一段以前不存在 —— 于是 GPT-2 的
        # `wpe.weight`（整张位置表）在这个后端里**根本没被加进去**，
        # 而输出形状完全正常。位置信息全靠外面加进来，不加就少了一样东西。
        if self.ir.get("pos_kind") == "learned":
            n = x.shape[1]
            x = x + self.params_global("pos.weight")[:n][None, :, :]
        for L in self.ir["layers"]:
            # **宽度按层取。** 原来这里传的是模型级 `d_model`，
            # 而独立宽度的栈（视觉塔）不是它 —— 参数形状改对了、
            # 计算宽度没跟着改，就是"形状对、算出来是另一个东西"。
            d = L.get("in_dim") or self.ir["d_model"]
            if x.shape[-1] != d:
                # **宽度衔接不上的时候说清楚，不要让 matmul 裸崩。**
                # 这句话正是多宽度模型现在缺的那一半：栈与栈之间
                # 残差流要换宽度，而那一步还没人做（见 history/ir.md 128）。
                raise ValueError(
                    f"层 {L['index']} 的输入宽度 {x.shape[-1]} != "
                    f"in_dim {d} —— **宽度在这里接不上**。"
                    f"（独立宽度的栈之间要换宽度，那一步还没实现。）")
            env = {"hidden": x}
            for j, o in enumerate(L["ops"]):
                ins = [env[r] for r in o["inputs"]]
                fn = OPS.get(o["kind"])
                if fn is None:
                    raise SystemExit(f"  [不支持] 这个后端还没有实现 {o['kind']}")
                env[o["outputs"][0]] = fn(self.P[L["index"]][j], o["attrs"], ins, d)
            x = env[L["ops"][-1]["outputs"][0]]
        # **最终归一化也要按 norm_kind 分派** —— 以前无条件走 rms，
        # 于是 LayerNorm 模型的最后一步也是错的。
        # **eps 要从 IR 里读，不能用函数的默认值。**
        # `rms_norm` 的 eps **必须显式传入**，否则吃默认值 —— 
        # 于是用了默认的 1e-5。而 clef-tiny / mla-shaped 的
        # `norm_eps` 是 **1e-6**，差一个量级，最后差出 2.7e-04。
        #
        # 这就是那个"犯过三次"的 bug 的**第五次**：
        # eps 写死在某个地方，平时看不出来 —— 因为大多数模型的 eps
        # 恰好就是 1e-5。llama-shaped 和 gpt2-tiny 都是 1e-5，
        # 所以它们一直是绿的，**只有偏离默认值的模型才露出来**。
        _eps = float(self.ir.get("norm_eps", _DEFAULT_EPS))
        if self.ir.get("norm_kind") == "layer":
            x = layer_norm(x, self.params_global("final_norm.w"),
                           self.params_global("final_norm.b"),
                           eps=_eps,
                           one_plus=self.ir.get("norm_one_plus", False))
        else:
            x = rms_norm(x, self.params_global("final_norm.w"), eps=_eps,
                         one_plus=self.ir.get("norm_one_plus", False))
        return x @ self.params_global("head.weight").T

    def params_global(self, k):
        return self.P["_g"][k]


def exec_ir(ir, seq=48, seed=0, from_graph=False):
    """**只吃 IR 的入口。** 返回 (输出, 参数字典)。

    原来这条路埋在 `main` 里 —— 想跑 NumPy 后端就得给一个 `.by1` 路径。
    规格说 IR 是接口，那就得有这条。`from_graph=True` 走图路径（A3-3）。
    """
    import by1ir as _ir
    errs = _ir.validate(ir)
    if errs:
        raise SystemExit("IR 不合法：\n  " + "\n  ".join(errs[:10]))
    shapes = shapes_of(ir)
    kinds = {}
    for L in ir["layers"]:
        for o in L["ops"]:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    miss = [k for k in kinds if k not in OPS]
    if miss:
        raise SystemExit("这个后端还没实现：%s" % miss)
    rng = np.random.default_rng(seed)
    params = {k: rng.normal(0, 0.1, s).astype(np.float32)
              for k, s in shapes.items()}
    ex = Exec(ir, params, from_graph=from_graph)
    # **按 shapes_of 实际给出的来** —— 硬编码三个名字的话，
    # 位置表和最终归一化的 bias 永远不会进 `_g`，
    # 而 `params_global` 会 KeyError（或者更糟：被静默跳过）。
    ex.P["_g"] = {k: params[k] for k in
                  ("embed.weight", "pos.weight", "final_norm.w",
                   "final_norm.b", "head.weight") if k in params}
    ids = rng.integers(0, ir["vocab"], (1, seq))
    return ex(ids), params


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 NumPy 执行器（第二个后端）")
    ap.add_argument("by1")
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--from-graph", action="store_true",
                    help="走图路径（A3-3 的 PRIMS 求值）；默认还是 OPS 路径")
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    name = os.path.basename(args.by1)
    rep, info = bc.check(args.by1)
    if rep.count(bc.E):
        return bc.run_one(args.by1)
    try:
        ir = cg.compile_ir(info)
    except cg.CodegenError as ex:
        print(f"[不支持] {name}: {ex}")
        return 1
    # **IR 是合同，run 这条路也要过它。** 这里原来编译完直接开跑 ——
    # `exec_ir` 有 validate，main 没有，于是「结构合法、算子组合没有定义」
    # 的声明一路走到 NumPy 广播里裸崩（2026-10-16 外部复现抓的
    # hello head_dim=31）。拒绝必须发生在前向之前、带诊断、exit 非零 ——
    # 不是崩给你看（「要么实现，要么拒绝」的第三条路）。
    import by1ir as _bir
    _errs = _bir.validate(ir)
    if _errs:
        print(f"  [拒绝] {name}: IR 不合法 —— 前向不会开始")
        for e in _errs[:10]:
            print("    " + e)
        return 1
    shapes = shapes_of(ir)
    print(f"\n{'='*72}\n  {name}   ·   NumPy 后端\n{'='*72}")
    n_params = sum(int(np.prod(shape)) for shape in shapes.values())
    print(f"\n  架构：{len(ir['layers'])} 层 · {n_params:,} 个参数 · 随机权重")
    print(f"\n  从 IR 推出的参数清单（{len(shapes)} 个）—— IR 里没有形状，这是后端自己算的")
    ks = sorted(shapes)
    for k in ks[:6]:
        print(f"    {k:<44} {shapes[k]}")
    if len(ks) > 8:
        print(f"    ...（共 {len(ks)} 个）")
    for k in ks[-3:]:
        print(f"    {k:<44} {shapes[k]}")

    kinds = {}
    for L in ir["layers"]:
        for o in L["ops"]:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    print(f"\n  后端必须实现的算子：{kinds}")
    miss = [k for k in kinds if k not in OPS]
    if miss:
        print(f"  [缺口] 这个后端还没实现：{miss}")
        return 1
    print("  [OK] 全部已实现")

    rng = np.random.default_rng(0)
    params = {k: rng.normal(0, 0.1, s).astype(np.float32) for k, s in shapes.items()}
    ex = Exec(ir, params, from_graph=args.from_graph)
    print("  执行路径：" + ("图（PRIMS 求值，A3-3）" if args.from_graph
                          else "OPS 逐算子（迁移期间的判卷人）"))
    # **按 shapes_of 实际给出的来** —— 硬编码三个名字的话，
    # 位置表和最终归一化的 bias 永远不会进 `_g`，
    # 而 `params_global` 会 KeyError（或者更糟：被静默跳过）。
    ex.P["_g"] = {k: params[k] for k in
                  ("embed.weight", "pos.weight", "final_norm.w",
                   "final_norm.b", "head.weight") if k in params}
    ids = rng.integers(0, ir["vocab"], (1, args.seq))
    out = ex(ids)
    print(f"\n  跑通：输出 {out.shape}，幅度 {np.abs(out).max():.4e}")
    if not np.isfinite(out).all():
        print("  [FAIL] 输出里有 NaN/Inf")
        return 1

    if args.compare:
        print("\n  与 PyTorch 后端逐位对比（同一份 IR、同一组权重）")
        import torch
        ns = {}
        exec(compile(cg.render(info, name), "<g>", "exec"), ns)
        m = ns["build"]().eval()
        sd = m.state_dict()
        import re as _re

        # **两个后端对同一个东西可以用不同的名字，那是事实，不是要猜的谜。**
        # 所以这里是一张**显式别名表** —— 而不是"按形状兜底猜一个"。
        #
        # 我试过按形状兜底（唯一匹配才认）。它把 clef-tiny 和 mla-shaped
        # 从 PASS 变成了 ~3e-04 —— 因为同形状的参数太多，
        # "唯一"匹配也会配到错的那个，而**配错之后检查照样往下走**。
        # 一个"帮你多认几个名字"的兜底，把"找不到就停"变成了"装错也继续"。
        NAME_ALIAS = {
            # PyTorch 那边叫 wpe（GPT-2 的历史名字），
            # IR 里叫 pos（`pos_kind = learned`，与具体模型无关）。
            "wpe.weight": "pos.weight",
            "pos.weight": "wpe.weight",
        }

        def find(nm, v):
            """按**名字**配对。配不上就返回 None —— 让它报"没找到"，
            而不是猜一个然后算出一个错的数。"""
            for cand in (nm,
                         NAME_ALIAS.get(nm),
                         _re.sub(r"\.(weight|bias)$", "", nm)):
                if cand and cand in params:
                    return cand
            return None

        tgt, left, used = {}, [], set()
        for k, v in sd.items():
            c = find(k, v)
            if c is None:
                left.append(k)
            elif int(np.prod(params[c].shape)) != int(v.numel()):
                # **元素总数对不上就是错了。** 这里原来无条件
                # `reshape(v.shape)` —— 形状不一样也能塞进去，于是
                # **宽度算错也照样绿**（`in_dim` 写错正好走这条路）。
                #
                # 判**总数**而不是判**形状**，是因为两边存在**合法的布局差**：
                # 卷积核 `shapes_of` 给 2-D `(64, 4)`、PyTorch 给 3-D `(64, 1, 4)`；
                # 注意力那边是融合 qkv vs 拆开。**总数一样、布局不同**，
                # reshape 干的就是这件事。宽度错了总数就变，所以这条尺子够用。
                left.append('%s 个数 %d != PyTorch 的 %d（形状 %s vs %s）'
                            % (k, int(np.prod(params[c].shape)), int(v.numel()),
                               tuple(params[c].shape), tuple(v.shape)))
            else:
                # 键必须是 PyTorch 的名字 k，不能是 numpy 的名字 c ——
                # 用错的话 strict=False 会**静默忽略全部**，模型还是随机初始化。
                tgt[k] = torch.tensor(params[c]).reshape(v.shape)
        if left:
            print(f"    [注意] PyTorch 侧有 {len(left)} 个参数没找到对应：{left[:4]}")
            return 1
        m.load_state_dict(tgt)
        # load 之后再验一次：确认权重真的进去了，而不是被静默忽略
        bad = [k for k, v in tgt.items()
               if not torch.equal(m.state_dict()[k], v)]
        if bad:
            print(f"    [FAIL] {len(bad)} 个参数 load 之后对不上，例如 {bad[:3]}")
            return 1
        with torch.no_grad():
            ref = m(torch.tensor(ids)).numpy()
        dd = np.abs(ref - out).max()
        amp = max(np.abs(ref).max(), 1e-9)
        print(f"    PyTorch 幅度 {np.abs(ref).max():.4e}   NumPy 幅度 "
              f"{np.abs(out).max():.4e}")
        _ok2 = dd / amp < 1e-5
        print(f"    最大绝对差 {dd:.3e}   相对 {dd/amp:.3e}   "
              + ("[PASS] 两个后端一致" if _ok2 else "[FAIL] 不一致"))
        # **打印了 FAIL 就得返回非零。**
        # 若这里 `return 0`，by1all 看退出码会把
        # `[FAIL] 不一致` 标成 **ok**。护栏瞎了多久没人知道：
        # mla-shaped 差 3.1e-04、clef-tiny 差 4.2e-04，
        # 一直在跑、一直在打印 FAIL、一直被当成通过。
        if not _ok2:
            print()
            return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
