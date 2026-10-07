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

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 基础算子（都是裸函数，没有模块/类层次） ────────────────────────

def rms_norm(x, w, eps=1e-5, one_plus=False):
    y = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
    return y * ((1.0 + w) if one_plus else w)


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
    return rms_norm(ins[0], P["w"], a.get("eps", 1e-5),
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
        q = rms_norm(q, P["qn.w"], one_plus=a.get("norm_one_plus", False))
        k = rms_norm(k, P["kn.w"], one_plus=a.get("norm_one_plus", False))
    hd, part = a["head_dim"], a.get("rope_partial", 1.0)
    npr = int(hd * part)
    if 0 < npr < hd:
        c, s = rope_tables(npr, n, a["rope_base"], a.get("yarn"))
        if a.get("rope_scale", 1.0) != 1.0:
            c, s = c * a["rope_scale"], s * a["rope_scale"]
        q = np.concatenate([apply_rope(q[..., :npr], c, s, a["rope_pairing"]),
                            q[..., npr:]], -1)
        k = np.concatenate([apply_rope(k[..., :npr], c, s, a["rope_pairing"]),
                            k[..., npr:]], -1)
    else:
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
    _eps = a.get("norm_eps", 1e-5)

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


def op_ffn(P, a, ins, d):
    x = ins[0]
    g = x @ P["w1"].T
    if not a.get("gate", True):
        return silu(g) @ P["w2"].T
    u = x @ P["w3"].T
    if a.get("act") == "gptoss":
        lim = a.get("limit")
        if lim is not None:
            g = np.minimum(g, lim)
            u = np.clip(u, -lim, lim)
        h = (u + 1) * (g / (1.0 + np.exp(-a.get("alpha", 1.702) * g)))
    else:
        h = silu(g) * u
    return h @ P["w2"].T


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
    y = out * (1.0 / np.sqrt(var + a.get("norm_eps", 1e-5)))
    y = (P["norm.w"] * y) * silu(z)
    return y.reshape(b, s, vd) @ P["out_proj"].T


OPS = {"Norm": op_norm, "Add": op_add, "Attention": op_attention,
       "FFN": op_ffn, "MoE": op_moe, "Linear": op_linear, "MLA": op_mla}


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
    out["final_norm.w"] = (d,)
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
        for L in self.ir["layers"]:
            env = {"hidden": x}
            for j, o in enumerate(L["ops"]):
                ins = [env[r] for r in o["inputs"]]
                fn = OPS.get(o["kind"])
                if fn is None:
                    raise SystemExit(f"  [不支持] 这个后端还没有实现 {o['kind']}")
                env[o["outputs"][0]] = fn(self.P[L["index"]][j], o["attrs"], ins, d)
            x = env[L["ops"][-1]["outputs"][0]]
        x = rms_norm(x, self.params_global("final_norm.w"),
                     one_plus=self.ir.get("norm_one_plus", False))
        return x @ self.params_global("head.weight").T

    def params_global(self, k):
        return self.P["_g"][k]


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
    ex.P["_g"] = {k: params[k] for k in ("embed.weight", "final_norm.w", "head.weight")}
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

        def find(nm, v):
            """两个后端各自命名参数（PyTorch 的 nn.Linear 会加 .weight），
            IR 不管这件事 —— 所以搬运时按形状兜底匹配。"""
            for cand in (nm, _re.sub(r"\.(weight|bias)$", "", nm)):
                if cand in params and (params[cand].shape == tuple(v.shape)
                                  or params[cand].shape == tuple(s for s in v.shape if s != 1)):
                    return cand
            return None
        tgt, left = {}, []
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
        print(f"    最大绝对差 {dd:.3e}   相对 {dd/amp:.3e}   "
              + ("[PASS] 两个后端一致" if dd / amp < 1e-5 else "[FAIL] 不一致"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
