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
    """和 by1codegen 那份逐字对应。yarn 这个参数其实是「缩放参数」，
    里面带 type 字段区分 yarn / llama3。"""
    half = head_dim // 2
    if yarn and (yarn.get("type") or "yarn") == "llama3":
        fac = float(yarn["factor"])
        hf, lf = float(yarn["high_freq"]), float(yarn["low_freq"])
        old = float(yarn["original"])
        p0 = 1.0 / (base ** (np.arange(0, half) / half))
        wl = 2 * np.pi / p0
        lo_wl, hi_wl = old / lf, old / hf
        out = np.where(wl > lo_wl, p0 / fac, p0)
        smooth = (old / wl - lf) / (hf - lf)
        inv = np.where((wl >= hi_wl) & (wl <= lo_wl),
                       (1 - smooth) * out / fac + smooth * out, out)
    elif yarn:
        fac, orig = yarn["factor"], yarn["original"]
        bf, bs = yarn["beta_fast"], yarn["beta_slow"]

        def corr_dim(nr):
            return (head_dim * np.log(orig / (nr * 2 * np.pi))) / (2 * np.log(base))
        lo, hi = corr_dim(bf), corr_dim(bs)
        if yarn.get("truncate", True):
            lo, hi = np.floor(lo), np.ceil(hi)
        lo, hi = max(lo, 0.0), min(hi, head_dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = np.clip((np.arange(half) - lo) / (hi - lo), 0, 1)
        pos = base ** (np.arange(0, head_dim, 2) / head_dim)
        inv = (1.0 / (fac * pos)) * ramp + (1.0 / pos) * (1 - ramp)
    else:
        inv = 1.0 / (base ** (np.arange(0, half) / half))
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
    q_gate = a.get("q_gate", False)
    w = a["head_dim"] * (2 if q_gate else 1)
    qq = x @ P["wq"].T
    if "wq.bias" in P:
        qq = qq + P["wq.bias"]
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
    kraw = kraw.reshape(b, n, a["kv"], a["head_dim"]).transpose(0, 2, 1, 3)
    if a.get("kv_tie"):
        v = kraw
    else:
        v = x @ P["wv"].T
        if "wv.bias" in P:
            v = v + P["wv.bias"]
        v = v.reshape(b, n, a["kv"], a["head_dim"]).transpose(0, 2, 1, 3)
    k = kraw
    if _qk_on(a.get("qk_norm")):
        # **eps 必须传，不能吃默认参数。**
        # 这里原来调 `rms_norm(q, P["qn.w"], one_plus=...)` —— **不传 eps**，
        # 于是吃 `rms_norm` 的默认 1e-5。而 by1codegen 那边读的是
        # `a.get("norm_eps", _DEFAULT_EPS)`。clef-tiny 的 qk_norm 是 per_head、
        # norm_eps 是 1e-6 —— **同一份 IR，两个后端两个答案**
        # （相对差 1.05e-05，低于 1e-4 判据所以一直绿）。
        # 这正是 by1c 的 C_HEAD 里记着"犯过五次"的那一类。
        _qeps = float(a.get("norm_eps", _DEFAULT_EPS))
        q = rms_norm(q, P["qn.w"], eps=_qeps,
                     one_plus=a.get("norm_one_plus", False))
        k = rms_norm(k, P["kn.w"], eps=_qeps,
                     one_plus=a.get("norm_one_plus", False))
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
    # 第一版我写的是 `if not _rope_on: part = 0.0` —— 于是 `npr = 0`
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


def op_mla(P, a, ins, d):
    """MLA：低秩压缩的注意力。和 PyTorch 后端逐行对应。"""
    x = ins[0]
    b, n, _ = x.shape
    nh, nope, nr = a["q"], a["qk_nope"], a["qk_rope"]
    vd, kvl = a["v_dim"], a["kv_lora"]
    _op = a.get("norm_one_plus", False)
    _eps = a.get("norm_eps", _DEFAULT_EPS)

    q = rms_norm(x @ P["q_a_proj"].T, P["q_a_layernorm.w"], _eps, _op)
    q = (q @ P["q_b_proj"].T).reshape(b, n, nh, nope + nr).transpose(0, 2, 1, 3)
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
    att = softmax(np.where(idx[None, :] <= idx[:, None], att, -np.inf), -1)
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
    if a.get("score_bias") and "score_bias" in P:
        sel = logits + P["score_bias"]
    else:
        sel = logits
    if a.get("routing") == "topk_softmax":
        part = np.argpartition(-sel, a["top_k"] - 1, axis=-1)[:, :a["top_k"]]
        tv = np.take_along_axis(sel, part, -1)
        topv = softmax(tv, -1)
        topi = part
    else:
        probs = softmax(sel, -1)
        part = np.argpartition(-probs, a["top_k"] - 1, axis=-1)[:, :a["top_k"]]
        tv = np.take_along_axis(probs, part, -1)
        topv = tv / np.maximum(tv.sum(-1, keepdims=True), 1e-9)
        topi = part
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
            h = silu(g) * u
        ye = h @ P["w2"][e].T
        if a.get("expert_bias") and "b2" in P:
            ye = ye + P["b2"][e]
        out[rows] += topv[rows][hit[rows]][:, None] * ye
    if a.get("shared"):
        sh = silu(xf @ P["sw1"].T) * (xf @ P["sw3"].T)
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
       "External": op_external}


# ── 执行器：按 IR 求值，不认机制名，只认 kind ──────────────────────

def shapes_of(ir):
    """**从 IR 自己推**每个 op 需要的参数形状。IR 里没有形状，这是后端的活。"""
    d = ir["d_model"]
    out = {}
    for L in ir["layers"]:
        for j, o in enumerate(L["ops"]):
            a, k = o["attrs"], o["kind"]
            pre = f"layers.{L['index']}.op{j}."
            if k == "Norm":
                out[pre + "w"] = (d,)
                # **LayerNorm 有 bias。** 以前这里一条都没有 ——
                # GPT-2 的 ln_1.bias / ln_2.bias 在后端层面不存在。
                if a.get("kind") == "layer":
                    out[pre + "b"] = (d,)
            elif k == "Attention":
                hd, q, kv = a["head_dim"], a["q"], a["kv"]
                out[pre + "wq"] = (q * hd * (2 if a.get("q_gate") else 1), d)
                out[pre + "wk"] = (kv * hd, d)
                if not a.get("kv_tie"):
                    out[pre + "wv"] = (kv * hd, d)
                out[pre + "wo"] = (d, a["out_dim"])
                if a["bias"]:
                    out[pre + "wq.bias"] = (q * hd * (2 if a.get("q_gate") else 1),)
                    out[pre + "wk.bias"] = (kv * hd,)
                    if not a.get("kv_tie"):
                        out[pre + "wv.bias"] = (kv * hd,)
                    out[pre + "wo.bias"] = (d,)
                if _qk_on(a.get("qk_norm")):
                    out[pre + "qn.w"] = (hd,)
                    out[pre + "kn.w"] = (hd,)
                if a.get("sink"):
                    out[pre + "sink"] = (q,)
                if a.get("head_gate", "off") != "off":
                    out[pre + "g_proj"] = ((q if a["head_gate"] == "per_head"
                                              else q * hd), d)
            elif k == "External":
                # **外部算子照样贡献张量形状。** 契约不松 ——
                # 它的权重和别的机制一样进参数表、一样被 by1verify 对产物查。
                # 前缀 `ext:` 是为了在 C 那边和内部名区分开。
                for nm, shp in (a.get("weights") or {}).items():
                    out[pre + "ext:" + nm] = tuple(shp)
            elif k == "MLA":
                out[pre + "q_a_proj"] = (a["q_lora"], d)
                out[pre + "q_a_layernorm.w"] = (a["q_lora"],)
                out[pre + "q_b_proj"] = (a["q"] * a["head_dim"], a["q_lora"])
                out[pre + "kv_a_proj_with_mqa"] = (a["kv_lora"] + a["qk_rope"], d)
                out[pre + "kv_a_layernorm.w"] = (a["kv_lora"],)
                out[pre + "kv_b_proj"] = (
                    a["q"] * (a["qk_nope"] + a["v_dim"]), a["kv_lora"])
                out[pre + "dense"] = (d, a["out_dim"])
                if a["bias"]:
                    out[pre + "dense.bias"] = (d,)
                if a.get("head_gate", "off") != "off":
                    out[pre + "g_proj"] = (a["q"], d)
            elif k == "FFN":
                out[pre + "w1"] = (a["hidden"], d)
                out[pre + "w2"] = (d, a["hidden"])
                if a.get("gate", True):
                    out[pre + "w3"] = (a["hidden"], d)
                # **bias。** 以前这里一条都没有 —— 于是 GPT-2 的
                # 4×12 个 bias 在后端层面**根本不存在**，
                # 而"张量搬运 173/173"看起来是满的。
                if a.get("bias"):
                    out[pre + "w1.bias"] = (a["hidden"],)
                    out[pre + "w2.bias"] = (d,)
                    if a.get("gate", True):
                        out[pre + "w3.bias"] = (a["hidden"],)
            elif k == "MoE":
                E, H = a["experts"], a["hidden"]
                out[pre + "router"] = (E, d)
                if a.get("router_bias"):
                    out[pre + "router.bias"] = (E,)
                out[pre + "w1"] = (E, H, d)
                out[pre + "w3"] = (E, H, d)
                out[pre + "w2"] = (E, d, H)
                if a.get("expert_bias"):
                    out[pre + "b1"] = (E, H)
                    out[pre + "b3"] = (E, H)
                    out[pre + "b2"] = (E, d)
                if a.get("score_bias"):
                    out[pre + "score_bias"] = (E,)
                if a.get("shared"):
                    sh = a["shared_hidden"]
                    out[pre + "sw1"] = (sh, d)
                    out[pre + "sw3"] = (sh, d)
                    out[pre + "sw2"] = (d, sh)
                    if a.get("shared_gate"):
                        out[pre + "shared_gate"] = (1, d)
            elif k == "Linear":
                kd, vd, nk, nv = (a["k_heads"] * a["k_dim"], a["v_heads"] * a["v_dim"],
                                  a["k_heads"], a["v_heads"])
                out[pre + "in_proj_qkvz"] = (kd * 2 + vd * 2, d)
                out[pre + "in_proj_ba"] = (nv * 2, d)
                out[pre + "conv"] = (kd * 2 + vd, a["conv_kernel"])
                out[pre + "dt_bias"] = (nv,)
                out[pre + "A_log"] = (nv,)
                out[pre + "norm.w"] = (a["v_dim"],)
                out[pre + "out_proj"] = (d, vd)
    out["embed.weight"] = (ir["vocab"], d)
    # **学习式位置表。** pos_kind=learned 的模型有一张 (n_pos, d) 的查表 ——
    # 它不在任何一层里，也不是 embed，所以以前这个后端**完全不知道它存在**。
    if ir.get("pos_kind") == "learned":
        out["pos.weight"] = (int(ir.get("n_pos") or ir.get("ctx") or 1024), d)
    out["final_norm.w"] = (d,)
    if ir.get("norm_kind") == "layer":
        out["final_norm.b"] = (d,)
    out["head.weight"] = (ir["vocab"], d)
    return out


class Exec:
    def __init__(self, ir, params):
        self.ir = ir
        self.P = {}
        for L in ir["layers"]:
            self.P[L["index"]] = {}
            for j, o in enumerate(L["ops"]):
                pre = f"layers.{L['index']}.op{j}."
                self.P[L["index"]][j] = {
                    k[len(pre):]: np.asarray(v)
                    for k, v in params.items() if k.startswith(pre)}

    def __call__(self, ids):
        d = self.ir["d_model"]
        x = self.params_global("embed.weight")[ids]
        # **学习式位置表。** 这一段以前不存在 —— 于是 GPT-2 的
        # `wpe.weight`（整张位置表）在这个后端里**根本没被加进去**，
        # 而输出形状完全正常。位置信息全靠外面加进来，不加就少了一样东西。
        if self.ir.get("pos_kind") == "learned":
            n = x.shape[1]
            x = x + self.params_global("pos.weight")[:n][None, :, :]
        for L in self.ir["layers"]:
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
        # 这里原来是 `rms_norm(x, w, one_plus=…)` —— **没传 eps**，
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


def exec_ir(ir, seq=48, seed=0):
    """**只吃 IR 的入口。** 返回 (输出, 参数字典)。

    原来这条路埋在 `main` 里 —— 想跑 NumPy 后端就得给一个 `.by1` 路径。
    规格说 IR 是接口，那就得有这条。
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
    ex = Exec(ir, params)
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
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    name = os.path.basename(args.by1)
    _r, info = bc.check(args.by1)
    ir = cg.compile_ir(info)
    shapes = shapes_of(ir)
    print(f"\n{'='*72}\n  {name}   ·   NumPy 后端\n{'='*72}")
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
    ex = Exec(ir, params)
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
        # 这里原来是 `return 0` —— 于是 by1all 看退出码，把
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
