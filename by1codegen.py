#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 codegen -- 把 .by1 编译成**后端无关的 IR**，再由后端生成可训练模型。

Block 0：IR 里不出现任何 PyTorch 概念。
  判据：同一份 IR，两个后端能跑出同一个前向。

Block 1.1：MoE 作为**一个算子 + 一组 attrs**，不是一族特例。

支持的子集（超出就明确报错，绝不静默生成错的模型）:
  机制类型   Attention  heads={q,kv,head_dim} / out_dim / window / qk_norm / bias / rope_base
             FFN        hidden / act / gate
             MoE        experts / top_k / shared / hidden / shared_hidden
  调度       stack { pattern = ... ; sel >> Mech }
  超参       hparams { d_model, vocab }
  位置       position { ... rope(base=…, pairing=…) ... }
"""

import importlib.util
import math
import os
import pprint
import re
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "by1check", os.path.join(HERE, "by1check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 支持矩阵 ────────────────────────────────────────────────────────

SUPPORTED_KINDS = {"Attention", "FFN", "MoE", "Linear", "MLA"}

ATTRS = {
    "Attention": {"heads", "q", "kv", "head_dim", "v", "qk",
                  "out_dim", "window", "qk_norm", "bias",
                  "attn_bias", "rope", "rope_base", "structural",
                  # Block 1.2：注意力侧的四个机制属性
                  "sink", "kv_tie", "head_gate", "rope_scale",
                  # Block 1.3：Qwen3-Next 逼出来的两个
                  "q_gate", "partial",
                  # 门控的激活函数。Laguna 用 softplus、Ling 用 sigmoid ——
                  # **同名不同义**，必须写出来，不能靠默认值猜。
                  "gate_act"},
    "FFN": {"hidden", "act", "gate", "bias", "layer_kind", "structural",
            "swiglu_limit", "alpha"},
    "MoE": {"experts", "top_k", "shared", "hidden", "shared_hidden",
            "score_bias", "routed_scale", "router_bias", "bias",
            "layer_kind", "structural",
            # Block 1.2：MoE 侧的机制属性
            "routing", "act", "swiglu_limit", "alpha", "expert_bias",
            "shared_gate",
            # 分组路由（noaux_tc）的结构属性：只决定契约的等价类划分，
            # 计算还没实现 —— 但声明出来不该报错
            "n_group", "topk_group"},
    # MLA：低秩压缩的注意力（DeepSeek 提出）。
    # kv 被压到 c_kv，k_rot 是**单头**共享的 —— 这是它和 GQA 的根本区别。
    "MLA": {"heads", "q", "head_dim", "q_lora", "kv_lora",
            "qk_nope", "qk_rope", "v_dim", "out_dim",
            "head_gate", "gate_act", "bias", "structural",
            "rope", "rope_base", "pairing", "norm_eps"},
    # Block 1.3：线性 / 递归注意力。第一个**真的带状态**的机制。
    "Linear": {"k_heads", "v_heads", "k_dim", "v_dim", "conv_kernel",
               "act", "norm_eps", "l2_eps", "out_dim", "structural",
               "decay", "shortconv"},
}


class CodegenError(Exception):
    pass


def _num(v, default=None):
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def _bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def _flag(v, default=False):
    """像 `sink = per_head` 这种描述性取值：只要不是明确的否定，就算开着。
    （旧版用 _bool 会把 per_head 判成 False —— 属性开着却被静默关掉。）"""
    if v is None:
        return default
    s = str(v).strip().lower()
    return s not in ("", "false", "no", "off", "none", "null", "0")


def _heads(raw):
    if not raw:
        return {}
    body = str(raw).strip()
    if body.startswith("{") and "}" in body:
        body = body[1: body.rfind("}")]
    out = {}
    for part in body.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            n = _num(v)
            if n is not None:
                out[k.strip()] = int(n)
    return out


def _rope_decl(info, key, default=None):
    for v in (info.get("position") or {}).values():
        m = re.search(r"\b" + key + r"\s*=\s*([\w.+-]+)", str(v))
        if m:
            return m.group(1)
    return default


def _rope_spec(info, li):
    """取出**第 li 层**该用的 RoPE 参数。

    优先走 by1check 那条已验证的路径（position 键 -> 层类型 -> 逐层类型），
    因为官方 config 的 rope_parameters 就是这么生成的，Gemma 上已经 36/36 对过。
    没有按类型声明时（只有 position { default = ... }）才退回全局扫描。
    """
    base = int(_num(_rope_decl(info, "base", "10000"), 10000))
    pairing = _rope_decl(info, "pairing", "interleaved")
    lts = info.get("layer_types") or []
    rbt = info.get("rope_by_type") or {}
    d = rbt.get(lts[li]) if li < len(lts) else None
    if not d:
        return {"base": base, "pairing": pairing, "scale": 1.0,
                "yarn": _yarn_params(info),
                "partial": float(_num(_rope_decl(info, "partial", "1"), 1.0))}
    yarn = None
    if d.get("factor") is not None:
        yarn = {"factor": float(d.get("factor", 1)),
                "original": int(d.get("original_max_position_embeddings", 4096)),
                "beta_fast": float(d.get("beta_fast", 32)),
                "beta_slow": float(d.get("beta_slow", 1)),
                "truncate": bool(d.get("truncate", True))}
    # attention_factor 是 YaRN 对 cos/sin 的整体缩放（位置 0 上就看得出来）
    fac = d.get("attention_factor")
    if fac is None and yarn:
        fac = 0.1 * math.log(yarn["factor"]) + 1.0 if yarn["factor"] > 1 else 1.0
    return {"base": int(d.get("rope_theta", base)), "pairing": pairing,
            "scale": float(fac if fac is not None else 1.0), "yarn": yarn,
            "partial": float(d.get("partial_rotary_factor", 1.0))}


def _yarn_params(info):
    """从 position 里抽 YaRN 参数。YaRN 不只是缩放，它改的是**频率本身**：
    低频段做插值、高频段外推，中间用一条 ramp 过渡。"""
    for v in (info.get("position") or {}).values():
        s = str(v)
        if "factor" not in s:
            continue

        def g(k, d):
            m = re.search(r"\b" + k + r"\s*=\s*([\w.+-]+)", s)
            return float(m.group(1)) if m else float(d)
        mt = re.search(r"\btruncate\s*=\s*(\w+)", s)
        return {"factor": g("factor", 1),
                "original": int(g("original", 4096)),
                "beta_fast": g("beta_fast", 32),
                "beta_slow": g("beta_slow", 1),
                # truncate=False 时不做 floor/ceil —— 过渡带的位置会因此不同
                "truncate": (mt.group(1).lower() not in ("false", "no", "off"))
                            if mt else True}
    return None


# ── 编译：.by1 -> IR ───────────────────────────────────────────────

def compile_ir(info: Dict[str, Any]) -> Dict[str, Any]:
    errs: List[str] = []

    def err(msg):
        if msg not in errs:
            errs.append(msg)

    hp = info.get("hparams") or {}
    d_model = _num(hp.get("d_model"))
    vocab = _num(hp.get("vocab"))
    if not d_model:
        raise CodegenError("hparams 里缺少 d_model —— 不知道模型多宽")
    if not vocab:
        raise CodegenError("hparams 里缺少 vocab —— 不知道词表多大")
    d_model, vocab = int(d_model), int(vocab)
    ctx = int(_num(info.get("ctx"), 256) or 256)
    # 归一化的两种约定：w 直接乘，还是 (1+w)。
    # Qwen3-Next 里两种并存 —— RMSNorm 是 zero-centered，RMSNormGated 不是。
    nv = str((info.get("hparams") or {}).get("norm", "")).strip().lower()
    norm_1p = nv in ("zero_centered", "one_plus", "1+", "gemma")
    base = int(_num(_rope_decl(info, "base", "10000"), 10000))
    pairing = _rope_decl(info, "pairing", "interleaved")
    kinds = info.get("mechs") or {}
    defaults = info.get("mech_attrs") or {}

    def one_mech(name, attrs, li=None):
        kind = kinds.get(name)
        if kind is None:
            err(f"机制 '{name}' 没有类型，无法生成")
            return None
        if kind not in SUPPORTED_KINDS:
            err(f"机制 '{name}' 的类型是 '{kind}'，codegen 还不支持"
                f"（目前支持：{', '.join(sorted(SUPPORTED_KINDS))}）")
            return None
        for k in attrs:
            if k not in ATTRS[kind]:
                err(f"机制 '{name}' 的属性 '{k}' codegen 不认识"
                    f"（{kind} 支持：{', '.join(sorted(ATTRS[kind]))}）")

        if kind == "Attention":
            h = {k: _num(attrs.get(k)) for k in ("q", "kv", "head_dim")}
            if any(v is None for v in h.values()):
                flat = _heads(attrs.get("heads"))
                for k in h:
                    if h[k] is None:
                        h[k] = flat.get(k)
            miss = [k for k, v in h.items() if v is None]
            if miss:
                err(f"机制 '{name}' 缺少 {', '.join(miss)}"
                    f" —— codegen 需要 q / kv / head_dim")
                return None
            q, kv, hd = int(h["q"]), int(h["kv"]), int(h["head_dim"])
            if kv <= 0 or q % kv:
                err(f"机制 '{name}': q={q} 不是 kv={kv} 的整数倍")
                return None
            win = (attrs.get("window") or "").strip().lower()
            window = None if win in ("", "none", "null", "0") else int(_num(win, 0))
            rs = (_rope_spec(info, li) if li is not None else
                  {"base": base, "pairing": pairing, "scale": 1.0,
                   "yarn": _yarn_params(info)})
            return {"kind": "Attention", "mech": name, "attrs": {
                "q": q, "kv": kv, "head_dim": hd,
                "out_dim": int(_num(attrs.get("out_dim")) or q * hd),
                "window": window,
                # 取值不只有真假：per_head（按头，默认）与 full（整宽）。
                # Instella 是 full —— 名字一样但含义不同。
                "qk_norm": (str(attrs.get("qk_norm", "")).strip().lower()
                            if attrs.get("qk_norm") is not None else "off"),
                "bias": _flag(attrs.get("bias", attrs.get("attn_bias"))),
                "rope_base": int(_num(attrs.get("rope_base"), rs["base"])),
                "rope_pairing": rs["pairing"],
                # YaRN 这类缩放会把 cos/sin 整体乘一个系数 —— 位置 0 上也看得出来
                "rope_scale": float(_num(attrs.get("rope_scale"), rs["scale"])),
                "yarn": rs["yarn"],
                "rope_partial": float(_num(attrs.get("partial"), rs["partial"])),
                "q_gate": _flag(attrs.get("q_gate")),
                "norm_one_plus": norm_1p,
                # Block 1.2
                "sink": _flag(attrs.get("sink")),
                "kv_tie": _flag(attrs.get("kv_tie")),
                "head_gate": (str(attrs.get("head_gate", "off")).strip().lower()
                              or "off"),
                "gate_act": (str(attrs.get("gate_act", "softplus")).strip().lower()
                             or "softplus")}}

        if kind == "MLA":
            h = {k: _num(attrs.get(k)) for k in
                 ("q", "q_lora", "kv_lora", "qk_nope", "qk_rope",
                  "v_dim", "head_dim")}
            if h["q"] is None:
                h["q"] = _heads(attrs.get("heads")).get("q")
            if h["head_dim"] is None:
                h["head_dim"] = _heads(attrs.get("heads")).get("head_dim")
            miss = [k for k, v in h.items()
                    if v is None and k != "head_dim"]
            if miss:
                err(f"机制 '{name}' 缺少 {', '.join(miss)} —— codegen 需要 "
                    f"q / q_lora / kv_lora / qk_nope / qk_rope / v_dim")
                return None
            q, qn, kr = int(h["q"]), int(h["qk_nope"]), int(h["qk_rope"])
            # qk_head_dim = nope + rope，和参考实现一致；给了就校验一下
            if h["head_dim"] is not None and int(h["head_dim"]) != qn + kr:
                err(f"机制 '{name}': head_dim={int(h['head_dim'])} 但 "
                    f"qk_nope({qn}) + qk_rope({kr}) = {qn + kr}")
                return None
            rs = (_rope_spec(info, li) if li is not None else
                  {"base": base, "pairing": pairing})
            return {"kind": "MLA", "mech": name, "attrs": {
                "q": q, "q_lora": int(h["q_lora"]), "kv_lora": int(h["kv_lora"]),
                "qk_nope": qn, "qk_rope": kr, "v_dim": int(h["v_dim"]),
                "head_dim": qn + kr,
                "out_dim": int(_num(attrs.get("out_dim")) or q * int(h["v_dim"])),
                "bias": _flag(attrs.get("bias", attrs.get("attn_bias"))),
                "rope_base": int(_num(attrs.get("rope_base"), rs["base"])),
                "pairing": attrs.get("pairing", rs["pairing"]),
                "head_gate": (str(attrs.get("head_gate", "off")).strip().lower()
                              or "off"),
                # 默认 sigmoid —— Ling 的参考就是这么写的（Laguna 的普通
                # Attention 默认 softplus，两者不同名同义，所以分开写）
                "gate_act": (str(attrs.get("gate_act", "sigmoid")).strip().lower()
                             or "sigmoid"),
                # 跟着模型的 rms_norm_eps 走**必须**。GDN 那边为这个 bug 修过一次
                # （原来写死 1e-6，而 RMSNorm 是 1e-5），MLA 这里又犯了一遍。
                "norm_eps": float(_num(attrs.get("norm_eps"),
                                       _num(hp.get("rms_eps"), 1e-5))),
                "norm_one_plus": norm_1p}}

        if kind == "FFN":
            hid = _num(attrs.get("hidden"))
            if not hid:
                err(f"机制 '{name}' 的 FFN 缺少 hidden（中间层多宽）")
                return None
            lim = _num(attrs.get("swiglu_limit"))
            return {"kind": "FFN", "mech": name, "attrs": {
                "hidden": int(hid), "act": str(attrs.get("act", "silu")),
                "gate": str(attrs.get("gate", "true")).lower() != "false",
                "limit": lim, "alpha": float(_num(attrs.get("alpha"), 1.702))}}

        if kind == "MoE":
            ne = _num(attrs.get("experts"))
            nk = _num(attrs.get("top_k"))
            hid = _num(attrs.get("hidden"))
            if not ne or not nk or not hid:
                err(f"机制 '{name}' 的 MoE 缺少 experts / top_k / hidden")
                return None
            shared = int(_num(attrs.get("shared"), 0) or 0)
            return {"kind": "MoE", "mech": name, "attrs": {
                "experts": int(ne), "top_k": int(nk), "hidden": int(hid),
                "shared": shared,
                "shared_hidden": int(_num(attrs.get("shared_hidden"),
                                          hid if shared else 0) or 0),
                "score_bias": _flag(attrs.get("score_bias")),
                "routed_scale": float(_num(attrs.get("routed_scale"), 1.0)),
                "router_bias": _flag(attrs.get("router_bias")),
                # Block 1.2 —— GPT-OSS 的路由是「先 top-k 再 softmax」，
                # Mixtral 的是「先 softmax 再 top-k 再归一」。这是两个机制。
                "routing": (str(attrs.get("routing", "softmax_topk")).strip()
                            .lower() or "softmax_topk"),
                "act": str(attrs.get("act", "silu")),
                "limit": _num(attrs.get("swiglu_limit")),
                "alpha": float(_num(attrs.get("alpha"), 1.702)),
                "expert_bias": _flag(attrs.get("expert_bias")),
                "shared_gate": _flag(attrs.get("shared_gate")),
                # noaux_tc 的分组路由：每组取前 2 之和排名，选 topk_group 个组，
                # 再在组内取 top_k。（Ling / DeepSeek V3 同一套）
                "n_group": int(_num(attrs.get("n_group"), 0) or 0),
                "topk_group": int(_num(attrs.get("topk_group"), 0) or 0)}}

        if kind == "Linear":
            nk = _num(attrs.get("k_heads"))
            nv = _num(attrs.get("v_heads"))
            dk = _num(attrs.get("k_dim"))
            dv = _num(attrs.get("v_dim"))
            miss = [k for k, v in (("k_heads", nk), ("v_heads", nv),
                                   ("k_dim", dk), ("v_dim", dv)) if not v]
            if miss:
                err(f"机制 '{name}' 的 Linear 缺少 {', '.join(miss)}")
                return None
            nk, nv, dk, dv = int(nk), int(nv), int(dk), int(dv)
            if nv % nk:
                err(f"机制 '{name}': v_heads={nv} 不是 k_heads={nk} 的整数倍")
                return None
            return {"kind": "Linear", "mech": name, "attrs": {
                "k_heads": nk, "v_heads": nv, "k_dim": dk, "v_dim": dv,
                "conv_kernel": int(_num(attrs.get("conv_kernel"), 4)),
                "act": str(attrs.get("act", "silu")),
                # 跟着模型的 rms_norm_eps 走。之前这里写死 1e-6，而 RMSNorm 那边是 1e-5 ——
                # 同一个模型里两个 eps，隔离测试因为显式传了值所以没暴露。
                "norm_eps": float(_num(attrs.get("norm_eps"),
                                       _num(hp.get("rms_eps"), 1e-5))),
                "l2_eps": float(_num(attrs.get("l2_eps"), 1e-6)),
                "out_dim": int(_num(attrs.get("out_dim")) or nv * dv)}}
        return None

    def ops_of(mixer, attached, layer_attrs):
        """把一层展开成算子序列。数据流用 ValueRef 表示，没有张量形状。"""
        ops: List[dict] = []
        env = "hidden"

        def emit(mk, ins):
            j = len(ops)
            out = f"op{j}.out"
            ops.append({"mech": mk["mech"], "kind": mk["kind"],
                        "attrs": mk["attrs"], "inputs": ins, "outputs": [out]})
            return out

        if mixer is not None:
            n = {"mech": "RMSNorm", "kind": "Norm",
     "attrs": {"one_plus": norm_1p,
               # 层归一化的 eps **必须跟着模型的 rms_eps 走**。
               # 三个后端原来各自写死 1e-5 —— 于是它们"一致地错"，
               # 互相对拍全绿，却都不符合 .by1 里写的值。
               "eps": float(_num(hp.get("rms_eps"), 1e-5))}}
            v = emit(n, [env])
            a = emit(mixer, [v])
            j = len(ops)
            ops.append({"mech": "Add", "kind": "Add", "attrs": {},
                        "inputs": [env, a], "outputs": [f"op{j}.out"]})
            env = f"op{j}.out"
        for am in attached:
            n = {"mech": "RMSNorm", "kind": "Norm",
     "attrs": {"one_plus": norm_1p,
               # 层归一化的 eps **必须跟着模型的 rms_eps 走**。
               # 三个后端原来各自写死 1e-5 —— 于是它们"一致地错"，
               # 互相对拍全绿，却都不符合 .by1 里写的值。
               "eps": float(_num(hp.get("rms_eps"), 1e-5))}}
            v = emit(n, [env])
            m = emit(am, [v])
            j = len(ops)
            ops.append({"mech": "Add", "kind": "Add", "attrs": {},
                        "inputs": [env, m], "outputs": [f"op{j}.out"]})
            env = f"op{j}.out"
        return ops

    layers = []
    for li, (_s, m, a, _k, atts) in enumerate(info["layer_seq"]):
        mixer = one_mech(m, a, li)
        attached = [one_mech(am, defaults.get(am, {}), li) for am in atts]
        attached = [x for x in attached if x]
        if mixer is None or mixer["kind"] not in ("Attention", "Linear", "MLA"):
            err(f"第 {li} 层没有可用的 token 混合器 —— 生成不出来")
            continue
        state = []
        if mixer["kind"] == "MLA":
            # MLA 的 cache 是**压缩的**：每个 token 只存 c_kv + k_rot，
            # 不随头数增长 —— 这正是 MLA 存在的理由。
            a = mixer["attrs"]
            state.append({"kind": "kv_cache", "bounded_by": None,
                          "dtype": "bf16", "reuse": "prefix",
                          "shape": [a["kv_lora"] + a["qk_rope"]],
                          "note": "压缩潜伏"})
        elif mixer["kind"] == "Linear":
            # 第一个**真的带状态**的机制。而且它有**两个**状态：
            #   recurrent     delta 规则维护的矩阵 (k_dim, v_dim)
            #   conv_history  短卷积的滑动历史（kernel-1 步）
            # 只算 delta 矩阵会漏掉后者 —— 实测才发现的。
            a = mixer["attrs"]
            state.append({"kind": "recurrent", "bounded_by": None,
                          "shape": [a["v_heads"], a["k_dim"], a["v_dim"]],
                          "dtype": "fp32", "reuse": "none"})
            conv_dim = 2 * a["k_heads"] * a["k_dim"] + a["v_heads"] * a["v_dim"]
            # 逻辑上因果卷积只需要 kernel-1 步历史，但实现按 kernel 步分配 —— 实测为准
            state.append({"kind": "conv_history",
                          "bounded_by": a["conv_kernel"],
                          "shape": [conv_dim, a["conv_kernel"]],
                          "dtype": "fp32", "reuse": "none"})
        elif mixer["attrs"]["window"]:
            state.append({"kind": "kv_cache",
                          "bounded_by": mixer["attrs"]["window"] - 1,
                          "dtype": "bf16", "reuse": "prefix"})
        else:
            state.append({"kind": "kv_cache", "bounded_by": None,
                          "dtype": "bf16", "reuse": "prefix"})
        layers.append({"index": li,
                       "attrs": dict(a),
                       "ops": ops_of(mixer, attached, a),
                       "state": state})

    if errs:
        raise CodegenError("\n".join("  - " + e for e in errs))

    return {
        "vocab": vocab, "ctx": ctx, "d_model": d_model,
        "norm_one_plus": norm_1p,
        "globals": [
            {"mech": "Embed", "kind": "Embed", "attrs": {},
             "inputs": [], "outputs": ["hidden"]},
            {"mech": "FinalNorm", "kind": "Norm",
             "attrs": {"one_plus": norm_1p},
             "inputs": ["hidden"], "outputs": ["normed"]},
            {"mech": "Head", "kind": "Head", "attrs": {},
             "inputs": ["normed"], "outputs": ["logits"]},
        ],
        "layers": layers,
    }


# 旧名字保留一个薄封装，避免外部调用点忽然断掉
def compile_spec(info):
    return compile_ir(info)


# ── 运行时：消费 IR，不假设任何架构 ────────────────────────────────

RUNTIME = r'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, one_plus=False):
        super().__init__()
        # 两种约定：直接乘 w，或者乘 (1+w)。后者是 zero-centered，w 初值为 0。
        # 同一个模型里两种可以并存 —— Qwen3-Next 就是这样。
        self.one_plus = one_plus
        self.w = nn.Parameter(torch.zeros(d) if one_plus else torch.ones(d))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * ((1.0 + self.w) if self.one_plus else self.w)


def rope_tables(head_dim, n, base, device, yarn=None):
    half = head_dim // 2
    if yarn:
        # YaRN：低频插值 / 高频外推，中间一条 ramp 过渡
        fac, orig = yarn["factor"], yarn["original"]
        bf, bs = yarn["beta_fast"], yarn["beta_slow"]

        def corr_dim(nr):
            return (head_dim * math.log(orig / (nr * 2 * math.pi))) / (2 * math.log(base))
        lo, hi = corr_dim(bf), corr_dim(bs)
        if yarn.get("truncate", True):
            lo, hi = math.floor(lo), math.ceil(hi)
        lo, hi = max(lo, 0.0), min(hi, head_dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = torch.clamp(
            (torch.arange(half, device=device).float() - lo) / (hi - lo), 0, 1)
        pos = base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        # 索引小 = 频率高 → ramp=0 → 外推（不动）
        # 索引大 = 频率低 → ramp=1 → 插值（按 factor 拉伸）
        inv = (1.0 / (fac * pos)) * ramp + (1.0 / pos) * (1 - ramp)
    else:
        inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(n, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin, pairing="interleaved"):
    """两种配对约定。**名字一样，含义不同，所以写清楚**：

      half         对半进（x[i] 配 x[i+d/2]）、拼接着出  —— 多数模型
      interleaved  奇偶进（x[2i] 配 x[2i+1]）、**拼接着出** —— DeepSeek/Ling

    注意 interleaved 说的是**输入**交错，输出仍是前半/后半 —— 这是
    transformers 的 apply_rotary_pos_emb_interleave 的实际行为（它的文档也写了
    "与 de-interleaved 的 rotate_half 写法逐位相同"）。
    曾经把它写成"输出也交错"（stack 而非 cat），随机输入下差 5.0 —— 看着像对的。
    """
    c, s = cos[None, None], sin[None, None]
    if pairing == "half":
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class MLAttention(nn.Module):
    """MLA：低秩压缩的注意力（DeepSeek 提出，Ling 用同一套）。

        q  = q_b_proj(q_a_layernorm(q_a_proj(x)))        拆 [qk_nope | qk_rope]
        kv = kv_a_proj_with_mqa(x)                       拆 [c_kv | k_rot]
        k_nope, v = split(kv_b_proj(kv_a_layernorm(c_kv)), [qk_nope, v_dim])

    和 GQA 的根本区别：**k_rot 是单头的**，RoPE 只作用在那 qk_rope 维上，
    然后广播到所有头。qk_head_dim = qk_nope + qk_rope，scaling 用前者。
    """

    def __init__(self, a, d):
        super().__init__()
        self.nh = a["q"]
        self.qk_nope, self.qk_rope = a["qk_nope"], a["qk_rope"]
        self.qk_head = self.qk_nope + self.qk_rope
        self.v_dim, self.kv_lora = a["v_dim"], a["kv_lora"]
        _op = a.get("norm_one_plus", False)
        _eps = a.get("norm_eps", 1e-5)
        self.q_a_proj = nn.Linear(d, a["q_lora"], bias=a["bias"])
        self.q_a_layernorm = RMSNorm(a["q_lora"], eps=_eps, one_plus=_op)
        self.q_b_proj = nn.Linear(a["q_lora"], self.nh * self.qk_head, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            d, self.kv_lora + self.qk_rope, bias=a["bias"])
        self.kv_a_layernorm = RMSNorm(self.kv_lora, eps=_eps, one_plus=_op)
        self.kv_b_proj = nn.Linear(
            self.kv_lora, self.nh * (self.qk_nope + self.v_dim), bias=False)
        self.dense = nn.Linear(self.nh * self.v_dim, d, bias=a["bias"])
        self.base = a["rope_base"]
        self.pairing = a["pairing"]
        self.head_gate = a.get("head_gate", "off")
        self.gate_act = a.get("gate_act", "sigmoid")
        if self.head_gate != "off":
            self.g_proj = nn.Linear(d, self.nh, bias=False)
        self.scaling = self.qk_head ** -0.5

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, n, self.nh, self.qk_head).transpose(1, 2)
        q_pass, q_rot = q.split([self.qk_nope, self.qk_rope], dim=-1)
        c_kv, k_rot = self.kv_a_proj_with_mqa(x).split(
            [self.kv_lora, self.qk_rope], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv))
        kv = kv.view(b, n, self.nh, self.qk_nope + self.v_dim).transpose(1, 2)
        k_pass, v = kv.split([self.qk_nope, self.v_dim], dim=-1)
        k_rot = k_rot.view(b, 1, n, self.qk_rope)
        cos, sin = rope_tables(self.qk_rope, n, self.base, x.device)
        q_rot = apply_rope(q_rot, cos, sin, self.pairing)
        k_rot = apply_rope(k_rot, cos, sin, self.pairing)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        qq = torch.cat([q_pass, q_rot], dim=-1)
        kk = torch.cat([k_pass, k_rot], dim=-1)
        att = (qq @ kk.transpose(-2, -1)) * self.scaling
        idx = torch.arange(n, device=x.device)
        att = att.masked_fill(~(idx[None, :] <= idx[:, None]),
                              float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, n, self.nh * self.v_dim)
        if self.head_gate != "off":
            g = self.g_proj(x).float()
            g = torch.sigmoid(g) if self.gate_act == "sigmoid" else F.softplus(g)
            o = (o.view(b, n, self.nh, self.v_dim)
                 * g.to(o.dtype).unsqueeze(-1)).reshape(b, n, -1)
        return self.dense(o)


class Attention(nn.Module):
    def __init__(self, a, d):
        super().__init__()
        self.q, self.kv, self.hd = a["q"], a["kv"], a["head_dim"]
        # q_gate：q_proj 的输出是 2 倍，一半当 query、一半当门（Qwen3-Next）
        self.q_gate = a.get("q_gate", False)
        self.wq = nn.Linear(d, self.q * self.hd * (2 if self.q_gate else 1),
                            bias=a["bias"])
        self.wk = nn.Linear(d, self.kv * self.hd, bias=a["bias"])
        # kv_tie: V 就是 K 的那份投影，不另建张量
        self.kv_tie = a.get("kv_tie", False)
        if not self.kv_tie:
            self.wv = nn.Linear(d, self.kv * self.hd, bias=a["bias"])
        self.wo = nn.Linear(a["out_dim"], d, bias=a["bias"])
        self.window = a["window"]
        self.base = a["rope_base"]
        self.pairing = a["rope_pairing"]
        self.scale = a.get("rope_scale", 1.0)
        self.yarn = a.get("yarn")
        self.rope_partial = a.get("rope_partial", 1.0)
        self.qk_norm = a["qk_norm"]
        if self.qk_norm == "full":
            raise CodegenError(
                "qk_norm = full（整宽归一化，如 Instella）codegen 还没实现 —— "
                "按 per_head 生成会得到一个**看起来对但算错**的模型，所以这里直接拒绝")
        if self.qk_norm not in ("off", "", None):
            _op = a.get("norm_one_plus", False)
            self.qn = RMSNorm(self.hd, one_plus=_op)
            self.kn = RMSNorm(self.hd, one_plus=_op)
        # sink: 每个头一个可学标量，作为**额外一列 logit** 参与 softmax，之后丢掉
        self.sink = nn.Parameter(torch.zeros(self.q)) if a.get("sink") else None
        self.head_gate = a.get("head_gate", "off")
        self.gate_per_head = (self.head_gate == "per_head")
        if self.head_gate != "off":
            self.g_proj = nn.Linear(
                d, self.q if self.gate_per_head else self.q * self.hd, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        w = self.hd * (2 if self.q_gate else 1)
        qq = self.wq(x).view(b, n, self.q, w)
        gate = None
        if self.q_gate:
            q, gate = torch.chunk(qq, 2, dim=-1)
        else:
            q = qq
        q = q.transpose(1, 2)
        kraw = self.wk(x).view(b, n, self.kv, self.hd).transpose(1, 2)
        v = kraw if self.kv_tie else \
            self.wv(x).view(b, n, self.kv, self.hd).transpose(1, 2)
        k = kraw
        if self.qk_norm not in ("off", "", None):
            q, k = self.qn(q), self.kn(k)
        # partial：只转前一段维度，剩下的原样带走
        np_ = int(self.hd * self.rope_partial)
        if 0 < np_ < self.hd:
            cos, sin = rope_tables(np_, n, self.base, x.device, self.yarn)
            if self.scale != 1.0:
                cos, sin = cos * self.scale, sin * self.scale
            q = torch.cat([apply_rope(q[..., :np_], cos, sin, self.pairing),
                           q[..., np_:]], dim=-1)
            k = torch.cat([apply_rope(k[..., :np_], cos, sin, self.pairing),
                           k[..., np_:]], dim=-1)
        else:
            cos, sin = rope_tables(self.hd, n, self.base, x.device, self.yarn)
            if self.scale != 1.0:
                # YaRN 的 attention_factor：cos 与 sin 一起缩放。
                cos, sin = cos * self.scale, sin * self.scale
            q = apply_rope(q, cos, sin, self.pairing)
            k = apply_rope(k, cos, sin, self.pairing)
        if self.q != self.kv:
            rep = self.q // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        idx = torch.arange(n, device=x.device)
        m = idx[None, :] <= idx[:, None]
        if self.window:
            m = m & (idx[None, :] > idx[:, None] - self.window)
        att = att.masked_fill(~m, float("-inf"))
        if self.sink is not None:
            s = self.sink.view(1, -1, 1, 1).expand(b, -1, n, -1)
            att = torch.cat([att, s], dim=-1).softmax(-1)[..., :-1]
        else:
            att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, n, self.q * self.hd)
        if gate is not None:
            o = o * torch.sigmoid(gate.reshape(b, n, -1))
        if self.head_gate != "off":
            # 激活函数是**机制属性**，不是默认值：Laguna 用 softplus、Ling 用 sigmoid
            _g = self.g_proj(x).float()
            g = (torch.sigmoid(_g) if self.gate_act == "sigmoid"
                 else F.softplus(_g)).to(o.dtype)
            if self.gate_per_head:
                o = (o.view(b, n, self.q, self.hd) * g.unsqueeze(-1)).view(b, n, -1)
            else:
                o = o * g
        return self.wo(o)


def _swiglu(gate, up, style, limit, alpha):
    """两个都叫 SwiGLU，但不是一回事：
         标准   silu(gate) * up
         GPT-OSS gate*sigmoid(alpha*gate) * (up+1)，并且两侧都 clamp"""
    if style == "gptoss":
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return (up + 1) * (gate * torch.sigmoid(gate * alpha))
    return F.silu(gate) * up


class MLP(nn.Module):
    def __init__(self, a, d):
        super().__init__()
        self.gate = a["gate"]
        self.style = a.get("act", "silu")
        self.limit = a.get("limit")
        self.alpha = a.get("alpha", 1.702)
        self.w1 = nn.Linear(d, a["hidden"], bias=False)
        self.w2 = nn.Linear(a["hidden"], d, bias=False)
        if self.gate:
            self.w3 = nn.Linear(d, a["hidden"], bias=False)

    def forward(self, x):
        g = self.w1(x)
        if self.gate:
            return self.w2(_swiglu(g, self.w3(x), self.style, self.limit, self.alpha))
        return self.w2(F.silu(g))


class MoE(nn.Module):
    """一个算子 + 一组 attrs。路由 / 共享专家 / score bias / 专家偏置都是参数，
    不是三种算子 —— 但它确实逼出了不少参数。"""

    def __init__(self, a, d):
        super().__init__()
        self.n_exp, self.k = a["experts"], a["top_k"]
        hid = a["hidden"]
        self.routing = a.get("routing", "softmax_topk")
        self.style = a.get("act", "silu")
        self.limit = a.get("limit")
        self.alpha = a.get("alpha", 1.702)
        self.router = nn.Linear(d, self.n_exp, bias=a["router_bias"])
        self.w1 = nn.Parameter(torch.randn(self.n_exp, hid, d) * 0.02)
        self.w3 = nn.Parameter(torch.randn(self.n_exp, hid, d) * 0.02)
        self.w2 = nn.Parameter(torch.randn(self.n_exp, d, hid) * 0.02)
        if a.get("expert_bias"):
            self.b1 = nn.Parameter(torch.zeros(self.n_exp, hid))
            self.b3 = nn.Parameter(torch.zeros(self.n_exp, hid))
            self.b2 = nn.Parameter(torch.zeros(self.n_exp, d))
        else:
            self.b1 = self.b3 = self.b2 = None
        self.score_bias = (nn.Parameter(torch.zeros(self.n_exp))
                           if a["score_bias"] else None)
        self.routed_scale = a["routed_scale"]
        self.n_group = a.get("n_group", 0)
        self.topk_group = a.get("topk_group", 0)
        self.n_shared = a["shared"]
        if self.n_shared:
            sh = a["shared_hidden"]
            self.sw1 = nn.Linear(d, sh, bias=False)
            self.sw3 = nn.Linear(d, sh, bias=False)
            self.sw2 = nn.Linear(sh, d, bias=False)
            self.shared_gate = (nn.Linear(d, 1, bias=False)
                                if a.get("shared_gate") else None)

    def forward(self, x):
        b, n, d = x.shape
        xf = x.reshape(-1, d)
        logits = self.router(xf)
        if self.routing == "sigmoid_group_topk":
            # noaux_tc（Ling / DeepSeek V3）：sigmoid 打分 + 分组 top-k。
            # 关键：expert_bias **只影响选择**，权重用未加偏置的 scores。
            ng, tg = self.n_group, self.topk_group
            if not ng or not tg:
                raise CodegenError(
                    "routing = sigmoid_group_topk 需要 n_group 与 topk_group")
            sc = logits.sigmoid()
            choice = sc + self.score_bias if self.score_bias is not None else sc
            gs = choice.view(-1, ng, self.n_exp // ng).topk(2, dim=-1)[0].sum(-1)
            gi = gs.topk(tg, dim=-1, sorted=False)[1]
            gm = torch.zeros_like(gs).scatter_(1, gi, 1)
            mask = (gm.unsqueeze(-1)
                    .expand(-1, ng, self.n_exp // ng)
                    .reshape(-1, self.n_exp))
            choice = choice.masked_fill(~mask.bool(), float("-inf"))
            topi = choice.topk(self.k, dim=-1, sorted=False)[1]
            tv = sc.gather(1, topi)
            topv = tv / (tv.sum(-1, keepdim=True) + 1e-20)
        elif self.routing == "topk_softmax":
            # 先 top-k，再只对这 k 个 logits 做 softmax（GPT-OSS）
            tv, topi = logits.topk(self.k, dim=-1)
            topv = tv.softmax(-1)
        else:
            # 全体 softmax -> top-k -> 重新归一（Mixtral）
            sel = logits + self.score_bias if self.score_bias is not None else logits
            tv, topi = sel.softmax(-1).topk(self.k, dim=-1)
            if self.score_bias is not None:
                tv = logits.gather(-1, topi).softmax(-1)
            topv = tv / (tv.sum(-1, keepdim=True) + 1e-9)
        topv = topv * self.routed_scale
        out = torch.zeros_like(xf)
        for e in range(self.n_exp):
            hit = (topi == e)
            rows = hit.any(-1)
            if not rows.any():
                continue
            xe = xf[rows]
            g = xe @ self.w1[e].t()
            u = xe @ self.w3[e].t()
            if self.b1 is not None:
                g = g + self.b1[e]
                u = u + self.b3[e]
            ye = _swiglu(g, u, self.style, self.limit, self.alpha) @ self.w2[e].t()
            if self.b2 is not None:
                ye = ye + self.b2[e]
            out[rows] += topv[rows][hit[rows]].unsqueeze(-1) * ye
        if self.n_shared:
            se = self.sw2(F.silu(self.sw1(xf)) * self.sw3(xf))
            if getattr(self, "shared_gate", None) is not None:
                se = se * torch.sigmoid(self.shared_gate(xf))
            out = out + se
        return out.reshape(b, n, d)


def _mk_norm(attrs, d):
    return RMSNorm(d, eps=attrs.get("eps", 1e-5),
                   one_plus=attrs.get("one_plus", False))


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def gated_delta_rule(q, k, v, g, beta, l2_eps=1e-6, state=None):
    """GDN 的顺序形式。分块版是同一个东西的快速等价实现。

    状态就是那个 delta 规则维护的矩阵 (k_dim, v_dim) —— **大小与序列长度无关**，
    这是「线性注意力」这个说法的全部含义。
    """
    q = l2norm(q, -1, l2_eps)
    k = l2norm(k, -1, l2_eps)
    q, k, v, beta, g = [t.transpose(1, 2).contiguous().float()
                        for t in (q, k, v, beta, g)]
    b, h, s, dk = k.shape
    dv = v.shape[-1]
    q = q * (dk ** -0.5)
    st = (torch.zeros(b, h, dk, dv, dtype=v.dtype, device=v.device)
          if state is None else state)
    out = torch.zeros(b, h, s, dv, dtype=v.dtype, device=v.device)
    for i in range(s):
        st = st * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (st * k[:, :, i].unsqueeze(-1)).sum(-2)
        delta = (v[:, :, i] - kv_mem) * beta[:, :, i].unsqueeze(-1)
        st = st + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (st * q[:, :, i].unsqueeze(-1)).sum(-2)
    return out.transpose(1, 2).contiguous(), st


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-6, act="silu"):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.act = act

    def forward(self, x, gate):
        v = x.float().pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(v + self.eps)
        return (self.w * y) * F.silu(gate.float())


class GatedDeltaNet(nn.Module):
    """GDN：短卷积 + delta 规则 + 门控归一化。带一个大小固定的递归状态。"""

    def __init__(self, a, d):
        super().__init__()
        self.nk, self.nv = a["k_heads"], a["v_heads"]
        self.dk, self.dv = a["k_dim"], a["v_dim"]
        self.kd, self.vd = self.dk * self.nk, self.dv * self.nv
        self.rep = self.nv // self.nk
        k = a["conv_kernel"]
        c = self.kd * 2 + self.vd
        self.conv = nn.Conv1d(c, c, k, groups=c, bias=False, padding=k - 1)
        self.in_proj_qkvz = nn.Linear(d, self.kd * 2 + self.vd * 2, bias=False)
        self.in_proj_ba = nn.Linear(d, self.nv * 2, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.A_log = nn.Parameter(torch.zeros(self.nv))
        self.norm = RMSNormGated(self.dv, a["norm_eps"], a["act"])
        self.out_proj = nn.Linear(self.vd, d, bias=False)
        self.l2_eps = a["l2_eps"]

    def forward(self, x):
        b, s, _ = x.shape
        rep, nk = self.rep, self.nk
        qkvz = self.in_proj_qkvz(x).view(
            b, s, nk, 2 * self.dk + 2 * self.dv * rep)
        ba = self.in_proj_ba(x).view(b, s, nk, 2 * rep)
        q, kk, v, z = torch.split(
            qkvz, [self.dk, self.dk, self.dv * rep, self.dv * rep], dim=3)
        bb, aa = torch.split(ba, [rep, rep], dim=3)
        v = v.reshape(b, s, self.nv, self.dv)
        z = z.reshape(b, s, self.nv, self.dv)
        bb = bb.reshape(b, s, self.nv)
        aa = aa.reshape(b, s, self.nv)
        mixed = torch.cat([q.reshape(b, s, self.kd), kk.reshape(b, s, self.kd),
                           v.reshape(b, s, self.vd)], dim=-1).transpose(1, 2)
        mixed = F.silu(self.conv(mixed)[:, :, :s]).transpose(1, 2)
        q, kk, v = torch.split(mixed, [self.kd, self.kd, self.vd], dim=-1)
        q = q.reshape(b, s, nk, self.dk)
        kk = kk.reshape(b, s, nk, self.dk)
        v = v.reshape(b, s, self.nv, self.dv)
        beta = bb.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(aa.float() + self.dt_bias)
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            kk = kk.repeat_interleave(rep, dim=2)
        out, _ = gated_delta_rule(q, kk, v, g, beta, self.l2_eps)
        out = self.norm(out.reshape(b, s, self.nv, self.dv), z)
        return self.out_proj(out.reshape(b, s, self.vd))


BUILDERS = {"Norm": _mk_norm, "Attention": Attention,
            "FFN": MLP, "MoE": MoE, "Linear": GatedDeltaNet,
            "MLA": MLAttention}


class LayerMod(nn.Module):
    def __init__(self, layer, d):
        super().__init__()
        self.plan = []
        for j, op in enumerate(layer["ops"]):
            b = BUILDERS.get(op["kind"])
            if b is not None:
                self.add_module("op%d" % j, b(op["attrs"], d))
            self.plan.append((op["kind"], j, op["inputs"], op["outputs"][0],
                              b is not None))

    def forward(self, x):
        env = {"hidden": x}
        for (kind, j, ins, out, has_mod) in self.plan:
            if kind == "Add":
                v = env[ins[0]] + env[ins[1]]
            else:
                v = getattr(self, "op%d" % j)(env[ins[0]])
            env[out] = v
        return env[self.plan[-1][3]]


class By1Model(nn.Module):
    def __init__(self, ir):
        super().__init__()
        self.ir = ir
        d = ir["d_model"]
        self.embed = nn.Embedding(ir["vocab"], d)
        self.layers = nn.ModuleList([LayerMod(L, d) for L in ir["layers"]])
        self.final_norm = RMSNorm(d, one_plus=ir.get("norm_one_plus", False))
        self.head = nn.Linear(d, ir["vocab"], bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.layers:
            x = blk(x)
        return self.head(self.final_norm(x))


def build():
    return By1Model(IR)
'''


def render(info: Dict[str, Any], by1_name: str = "model.by1") -> str:
    ir = compile_ir(info)
    return (
        f"# 由 by1 从 {by1_name} 生成 —— 改 .by1 再重新生成，不要手改这个文件\n"
        f"# 后端无关的 IR；下面的 RUNTIME 只是它的一个后端（PyTorch）\n\n"
        f"IR = {pprint.pformat(ir, indent=2, width=86, sort_dicts=False)}\n"
        + RUNTIME
        + '\n\nif __name__ == "__main__":\n'
          '    m = build()\n'
          '    n = sum(p.numel() for p in m.parameters())\n'
          '    print(f"层数 {len(m.layers)}   参数 {n:,}")\n'
    )


def build(info: Dict[str, Any]):
    src = render(info)
    ns: Dict[str, Any] = {}
    exec(compile(src, "<by1-generated>", "exec"), ns)
    return ns["build"](), src


def main(argv):
    if len(argv) < 1:
        print(__doc__)
        return 2
    out = None
    args = list(argv)
    if "-o" in args:
        i = args.index("-o")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    bc = load_checker()
    _rep, info = bc.check(args[0])
    try:
        src = render(info, os.path.basename(args[0]))
    except CodegenError as ex:
        print(f"[不支持] {os.path.basename(args[0])} 超出了 codegen 的子集：")
        print(str(ex))
        return 1
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"已生成 {out}")
    else:
        print(src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
