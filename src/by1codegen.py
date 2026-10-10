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
from typing import Any, Dict, List

# **版本号从 by1ver 来。** 原来这里写死 "1.0"，而 by1boot / by1extdemo
# 也各写了一遍 —— 三份同一个字符串，谁也不认识谁。
# 改一份忘一份不会崩，只会造出「声称不同版本」的 IR。
try:
    from by1ver import IR_VERSION as _IR_VER
except ImportError:                     # 单独拷一个文件出去时兜底
    _IR_VER = "1.0"

# **eps 的默认值只有一个地方写**（`by1ir.EPS_DEFAULT`）。
# 这个文件里原来有 17 处 `1e-5` —— 形状没错，错在"哪一处漏了传参"
# 没有结构性征兆（这一类在 history/ir.md 里记着犯过五次）。
#
# **但这里有两种用法，不能一把替换**：
#   · `compile_ir` 里那些是**现在**算的 -> 直接用常量
#   · 生成器模板里那些会被**写进生成的代码**，而那段代码在一个
#     空命名空间里 exec，看不见这个模块的全局名 ——
#     所以必须内插成**字面量**（`1e-05`）。
#     第一版我一把全换成常量名，生成出来的代码当场 NameError。
try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:                     # 单独拷一个文件出去时兜底
    _DEFAULT_EPS = 1e-5


def _stamp():
    """生成物的版本章。**每一份生成的东西都该能追回是哪版生成的。**"""
    try:
        from by1ver import stamp as _s
        return _s()
    except ImportError:
        return "by1"

HERE = os.path.dirname(os.path.abspath(__file__))


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "by1check", by1paths.tool("by1check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 支持矩阵 ────────────────────────────────────────────────────────

SUPPORTED_KINDS = {"Attention", "FFN", "MoE", "Linear", "MLA",
                    # 选择性状态空间（Mamba2 / Nemotron-H 那一版）。
                    # **它不是注意力的变体** —— 没有 q/k/v，状态是
                    # `A_log` / `D` / `dt_bias` 驱动的。
                    # 判卷人：`src/modelcheck/by1ssm.py`（对官方的
                    # `NemotronHMamba2Mixer`，两种尺寸 + 三条反例）。
                    "SSM",
                    # 逃生舱第一层 —— 计算在 raw.py 里，契约照旧。
                    # **这一层要改编译器。**
                    "Raw",
                    # 逃生舱第二层 —— 引用外部符号。
                    # **这一层不用改编译器**，只要有一个 .so。
                    "External"}

ATTRS = {
    # **逃生舱。** 只认一个属性：impl（raw.py 里的工厂函数名）。
    # 别的都不该有 —— 逃生舱只该有一条路。
    "Raw": {"impl", "structural"},
    "External": {"lib", "symbol", "weights", "io", "note",
                 "structural"},

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
            "shared_gate", "swiglu_limit_shared",
            # **`gate` 是"名字"还是"取值"？** Nemotron 声明的是
            # `gate=none` —— 那不是一种新的门控，是**这个属性取空**。
            # 而"名字不在允许表里就报不支持"这一条把它也拦下了 ✗。
            # 加进来只是让 `none` 通过；非 `none` 的取值由下面
            # `SSM`/`MoE` 的语义各自决定。
            "gate",
            # 分组路由（noaux_tc）的结构属性：只决定契约的等价类划分，
            # 计算还没实现 —— 但声明出来不该报错
            "n_group", "topk_group"},
    # 选择性状态空间。**和注意力是不同的算法族** —— 没有 q/k/v。
    "SSM": {"heads", "head_dim", "ssm_state", "n_groups", "conv_kernel",
            "expand", "chunk_size", "conv_bias", "proj_bias", "act",
            "norm_eps", "structural"},
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
    """**这份描述写坏了** —— 属性缺失、取值非法、机制没有类型。"""


class UnsupportedError(CodegenError):
    """**生成器还不认这个机制** —— 和"写坏了"是两回事。

    两者以前都走 `CodegenError`，于是调用方只能一起处理：`by1diff`
    把它们都算成"验了不对"。而 KDA / SSM 那些
    "只有契约，算不了"（README 原话）其实是"这台机器上没验"。

    **假红和假绿一样会让人忽略一个检查** —— 所以让它们分家。

    继承 `CodegenError` 是为了不让老的 `except CodegenError` 断掉；
    调用方要先捕这个子类，再捕父类。
    """


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


# ── 取值必须在**实现过的**集合里 ────────────────────────────────────
# 一个"看起来对"的默认值比一个报错糟得多：它生成的是**另一个模型**，
# 而所有检查都会放它过去 —— 张量契约查的是权重的名字和形状，不是算了什么。
#
#   act       RUNTIME 只实现了两支：silu 和 gptoss，其余全落回 silu
#   qk_norm   off / per_head / full
#   routing   softmax_topk / topk_softmax / sigmoid_group_topk
#   gate_act  softplus / sigmoid
ENUMS = {
    # gelu 是**无门控**那一路用的（GPT-2 的 MLP 只有两层）。
    # gelu_new 是 tanh 近似，和精确式差 1e-3 量级 —— 不能混。
    "act": {"silu", "gptoss", "gelu", "gelu_new", "relu2"},
    # `true` / `on` 是旧写法，等价于 per_head —— RUNTIME 判的是
    # `not in ("off", "", None)`，所以它确实实现过。不列进来会误伤。
    "qk_norm": {"off", "per_head", "full", "true", "on", "yes", "1"},
    "routing": {"softmax_topk", "topk_softmax", "sigmoid_group_topk",
                "sigmoid_topk"},
    "gate_act": {"softplus", "sigmoid"},
    "pairing": {"half", "interleaved"},
}


def _enum_bad(key, v, name, kind):
    """取值不在实现集合里 -> 返回错误消息；没问题 -> 返回 None。"""
    allow = ENUMS.get(key)
    if allow is None or v is None:
        return None
    s = str(v).strip().lower()
    if s and s not in allow:
        return ("机制 '%s'（%s）的 %s = <%s> —— codegen 只实现了 %s。"
                "按默认值生成会得到一个**看起来对但算错**的模型，所以拒绝"
                % (name, kind, key, v, " / ".join(sorted(allow))))
    return None


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
    # **`pairing` 原来没走取值校验。** `ENUMS` 里声明了它的闭集，
    # 但那个声明**一个调用点都没有** —— 于是 `pairing = totally_bogus`
    # 安静地落进 `apply_rope` 的 `else` 分支（interleaved）：
    # 对 half 配对的模型，两边前向差 1.8e-02，而张量契约全中、
    # `by1check` 报 0 错。这正是「能描述 ≠ 能算」那个 bug 类的新入口。
    # `_rope_spec` 没有 errs 通道（它是独立函数），所以在这里直接抛。
    pairing = _rope_decl(info, "pairing", "interleaved")
    _p = _enum_bad("pairing", pairing, "position", "rope")
    if _p:
        raise CodegenError(_p)
    lts = info.get("layer_types") or []
    rbt = info.get("rope_by_type") or {}
    d = rbt.get(lts[li]) if li < len(lts) else None
    if not d:
        return {"base": base, "pairing": pairing, "scale": 1.0,
                "yarn": _yarn_params(info),
                "partial": float(_num(_rope_decl(info, "partial", "1"), 1.0))}
    # **类型必须从描述里读出来，不能按"有 factor 就是 YaRN"猜。**
    # yarn 和 llama3 都改频率、参数名还重叠，混起来会静默算错。
    rtype = str(d.get("rope_type") or "default").strip().lower()
    yarn = None
    if d.get("factor") is not None:
        yarn = {"type": rtype if rtype in ("yarn", "llama3") else "yarn",
                "factor": float(d.get("factor", 1)),
                "original": int(d.get("original_max_position_embeddings", 4096)),
                "beta_fast": float(d.get("beta_fast", 32)),
                "beta_slow": float(d.get("beta_slow", 1)),
                "high_freq": float(d.get("high_freq_factor",
                                          d.get("high_freq", 4))),
                "low_freq": float(d.get("low_freq_factor",
                                        d.get("low_freq", 1))),
                "truncate": bool(d.get("truncate", True))}
    # attention_factor 是 YaRN 对 cos/sin 的整体缩放（位置 0 上就看得出来）。
    # **llama3 没有这一项** —— 它的 attention_factor 恒为 1.0（参考实现的注释
    # 就写着 "Unused in this type of RoPE"）。跟着 YaRN 的公式算是错的。
    fac = d.get("attention_factor")
    if fac is None and yarn and yarn["type"] == "yarn":
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
        # 包装器的名字就是类型：yarn(...) / llama3(...)。
        # **两者都改频率，但改法完全不同** —— 名字一样、参数名还重叠，
        # 不区分就会把 llama3 当 YaRN 算。
        ty = "yarn"
        for _w in ("llama3", "yarn"):
            if re.search(r"\b" + _w + r"\s*\(", s):
                ty = _w
                break
        return {"type": ty,
                "factor": g("factor", 1),
                "original": int(g("original", 4096)),
                "beta_fast": g("beta_fast", 32),
                "beta_slow": g("beta_slow", 1),
                "high_freq": g("high_freq_factor", g("high_freq", 4)),
                "low_freq": g("low_freq_factor", g("low_freq", 1)),
                # truncate=False 时不做 floor/ceil —— 过渡带的位置会因此不同
                "truncate": (mt.group(1).lower() not in ("false", "no", "off"))
                            if mt else True}
    return None


# ── 编译：.by1 -> IR ───────────────────────────────────────────────

#: **生成器认得哪些 token 混合器。** 只有这一处定义 —— 判定也只用它。
#:
#: 不在里面的（`KDA` / 稀疏索引器等）**不是模型的错**，
#: 是生成器还没实现。README 里就是这么写的：
#: "KDA / SSM / 稀疏索引器 / mHC 只有契约，算不了"。
#:
#: （`SSM` 原来在这个名单**外面** —— 而这一节改完之后，那句话里
#:   只剩 `KDA` / 稀疏索引器 / mHC 还成立 ✓。）
#:
#: 这个常量的用处是让调用方**在渲染之前**就知道，从而按三态协议
#: 说"这台机器上没验"，而不是报一个看起来像"模型算错了"的失败。
#:
#: **`MoE` 也在这里。** Nemotron-H 的层是 `Mamba / MoE / Mamba / MoE /
#: Mamba / Attn / …` 交替的 —— **MoE 在那里就是 token 混合器**
#: （它把 token 之间混起来，而不是只逐 token 变宽）。
#: 加上它不影响别的模型：混合器是**按位置挑**的，只有那一层的主机制
#: 才会走到这个判断。
MIXER_KINDS = ("Attention", "Linear", "MLA", "MoE", "SSM", "Raw",
               "External")


def compile_ir(info: Dict[str, Any]) -> Dict[str, Any]:
    errs: List[str] = []
    #: **"生成器还不认"单独一个桶。** 它和"写坏了"要分开报 ——
    #: 一份文件里两样都有时，**写坏了优先**（那就是一份坏文件）。
    unsup: List[str] = []

    def unsupported(msg):
        unsup.append(msg)

    def _ck(key, value, name, kind):
        """取值校验：不合法就 err 并返回 False。

        装在每个"取值选分支"的属性前面 —— act / qk_norm / routing /
        gate_act 都是这种，取值多一个不会报错，只会**悄悄走默认分支**。
        """
        _m = _enum_bad(key, value, name, kind)
        if _m:
            err(_m)
            return False
        return True

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
    # 同 `_rope_spec`：全局声明的 `pairing` 也要过取值门，
    # 不能静默落进 `apply_rope` 的默认分支。
    pairing = _rope_decl(info, "pairing", "interleaved")
    if not _ck("pairing", pairing, "position", "rope"):
        pairing = "interleaved"     # 已经记了错，值随便取；下面会 raise
    kinds = info.get("mechs") or {}
    defaults = info.get("mech_attrs") or {}

    def one_mech(name, attrs, li=None):
        kind = kinds.get(name)
        if kind is None:
            err(f"机制 '{name}' 没有类型，无法生成")
            return None
        if kind not in SUPPORTED_KINDS:
            unsupported(f"机制 '{name}' 的类型是 '{kind}'，codegen 还不支持"
                        f"（目前支持：{', '.join(sorted(SUPPORTED_KINDS))}）")
            return None
        for k in attrs:
            if k not in ATTRS[kind]:
                # **"属性不认识"也是"不认"，不是"写坏了"。**
                #
                # 它有两种可能：① 生成器没实现这个特性；② 属性名拼错了。
                # **② 由 `by1check` 在上游抓**（它有声明式的属性表），
                # 所以到这一层时属性名已经是"合法但我不支持"。
                #
                # 实测：`SparseMLA` 的 index_heads / index_dim / index_topk、
                # `MoE` 的 scoring 都落在这里 —— 而 README 原话就是
                # "KDA / SSM / **稀疏索引器** / mHC 只有契约，算不了"。
                unsupported(f"机制 '{name}' 的属性 '{k}' codegen 还不支持"
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
            if not (_ck("qk_norm", attrs.get("qk_norm"), name, kind)
                    and _ck("gate_act", attrs.get("gate_act"), name, kind)):
                return None
            # **模型用学习式位置编码时，注意力不做旋转。**
            # 位置信息在输入上（wpe），不在 q/k 上 —— 两件事。
            # 以前这里无条件 apply_rope：GPT-2 被转了一遍，而 .by1 里
            # **没有地方能说"别转"**。和 gate = none 变成 true 是同一类：
            # "不存在的机制"表达不出来。
            _rope_on = not _pos_learned
            rs = (_rope_spec(info, li) if li is not None else
                  {"base": base, "pairing": pairing, "scale": 1.0,
                   "yarn": _yarn_params(info)})
            return {"kind": "Attention", "mech": name, "attrs": {
                "q": q, "kv": kv, "head_dim": hd,
                "out_dim": int(_num(attrs.get("out_dim")) or q * hd),
                "window": window,
                # 取值不只有真假：per_head（按头，默认）与 full（整宽）。
                # Instella 是 full —— 名字一样但含义不同。
                # **IR 里存规范值。** `true` / `on` / `yes` / `1` 是语言层的
                # 旧写法，等价于 per_head —— 规格说 IR 只该有三种取值。
                # 以前直接把原文透传，于是 IR 里混着 `true`，
                # 而三个后端各自 `not in ("off","",None)` 恰好都对。
                # **恰好都对**是这类 bug 的典型长相。
                "qk_norm": ("per_head"
                            if str(attrs.get("qk_norm", "")).strip().lower()
                            in ("true", "on", "yes", "1") else
                            (str(attrs.get("qk_norm", "")).strip().lower()
                             if attrs.get("qk_norm") is not None else "off")),
                # 归一化的 eps 跟着模型的 rms_norm_eps 走（这个 bug 犯过三次）
                "norm_eps": float(_num(attrs.get("norm_eps"),
                                       _num(hp.get("rms_eps"), _DEFAULT_EPS))),
                "bias": _flag(attrs.get("bias", attrs.get("attn_bias"))),
                "rope": _rope_on,
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
            # **逐机制也可以改 `pairing`**（MLA 的 ATTRS 里有它）——
            # 这里也要过取值门，和全局声明那条一样。原来只有全局那条，
            # 而逐机制的非法值会静默落进 `apply_rope` 的默认分支。
            if not _ck("pairing", attrs.get("pairing"), name, kind):
                return None
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
                                       _num(hp.get("rms_eps"), _DEFAULT_EPS))),
                "norm_one_plus": norm_1p}}

        if kind == "FFN":
            hid = _num(attrs.get("hidden"))
            if not hid:
                err(f"机制 '{name}' 的 FFN 缺少 hidden（中间层多宽）")
                return None
            lim = _num(attrs.get("swiglu_limit"))
            # **`gate` 原来写的是 `str(...).lower() != "false"`** ——
            # 于是 `gate = none` 变成 `True`（"none" != "false"）。
            # 项目里早就有正确的 `_flag`，只是这里没用它。
            _m = _enum_bad("act", attrs.get("act"), name, kind)
            if _m:
                err(_m)
                return None
            _gv = attrs.get("gate")
            return {"kind": "FFN", "mech": name, "attrs": {
                "hidden": int(hid),
                "act": str(attrs.get("act", "silu")).strip().lower(),
                "gate": _flag(_gv, True) if _gv is not None else True,
                "limit": lim, "alpha": float(_num(attrs.get("alpha"), 1.702)),
                "bias": _flag(attrs.get("bias"))}}

        if kind == "MoE":
            ne = _num(attrs.get("experts"))
            nk = _num(attrs.get("top_k"))
            hid = _num(attrs.get("hidden"))
            if not ne or not nk or not hid:
                err(f"机制 '{name}' 的 MoE 缺少 experts / top_k / hidden")
                return None
            shared = int(_num(attrs.get("shared"), 0) or 0)
            if not (_ck("routing", attrs.get("routing"), name, kind)
                    and _ck("act", attrs.get("act"), name, kind)
                    and _ck("gate_act", attrs.get("gate_act"), name, kind)):
                return None
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
                "act": str(attrs.get("act", "silu")).strip().lower(),
                "limit": _num(attrs.get("swiglu_limit")),
                # 共享专家的夹取值是**独立**的：Step-3.7 的路由专家夹 7、
                # 共享专家夹 16。原来共享专家那条路是裸的 F.silu，夹取对它不生效。
                "limit_shared": _num(attrs.get("swiglu_limit_shared")),
                "alpha": float(_num(attrs.get("alpha"), 1.702)),
                "expert_bias": _flag(attrs.get("expert_bias")),
                "shared_gate": _flag(attrs.get("shared_gate")),
                # noaux_tc 的分组路由：每组取前 2 之和排名，选 topk_group 个组，
                # 再在组内取 top_k。（Ling / DeepSeek V3 同一套）
                "n_group": int(_num(attrs.get("n_group"), 0) or 0),
                "topk_group": int(_num(attrs.get("topk_group"), 0) or 0)}}

        if kind == "External":
            # **逃生舱第二层。** 编译器不认识这个机制 —— 只校验 ABI 要的东西
            # 齐不齐。所以加一个新机制不用动这里。
            lib = str(attrs.get("lib", "")).strip().strip('"')
            sym = str(attrs.get("symbol", "")).strip().strip('"')
            wts = attrs.get("weights")
            if not lib or not sym:
                err("机制 '%s' 是 External（外部符号），必须写 lib 和 symbol" % name)
                return None
            if not isinstance(wts, dict) or not wts:
                err("机制 '%s' 是 External —— **必须声明自己的张量**。"
                    "契约不松：下沉一层只是计算不在语言里，"
                    "张量照样要对产物查" % name)
                return None
            return {"kind": "External", "mech": name,
                    "attrs": {"lib": lib, "symbol": sym,
                              "weights": wts,
                              "io": str(attrs.get("io", "same")).strip(),
                              "note": str(attrs.get("note", ""))}}

        if kind == "Raw":
            # **逃生舱。** impl 指向 raw.py 里的工厂函数 raw_<impl>。
            # 别的什么都不认 —— 逃生舱只该有一条路。
            impl = str(attrs.get("impl", "")).strip().strip('"')
            if not impl:
                err("机制 '%s' 是 Raw（逃生舱），必须写 impl = <工厂函数名>" % name)
                return None
            return {"kind": "Raw", "mech": name,
                    "attrs": {"impl": impl}}

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
            if not _ck("act", attrs.get("act"), name, kind):
                return None
            return {"kind": "Linear", "mech": name, "attrs": {
                "k_heads": nk, "v_heads": nv, "k_dim": dk, "v_dim": dv,
                "conv_kernel": int(_num(attrs.get("conv_kernel"), 4)),
                "act": str(attrs.get("act", "silu")),
                # 跟着模型的 rms_norm_eps 走。之前这里写死 1e-6，而 RMSNorm 那边是 1e-5 ——
                # 同一个模型里两个 eps，隔离测试因为显式传了值所以没暴露。
                "norm_eps": float(_num(attrs.get("norm_eps"),
                                       _num(hp.get("rms_eps"), _DEFAULT_EPS))),
                "l2_eps": float(_num(attrs.get("l2_eps"), 1e-6)),
                "out_dim": int(_num(attrs.get("out_dim")) or nv * dv)}}

        if kind == "SSM":
            # **选择性状态空间。** 和 Linear 那一支是同一形状（非注意力的
            # 带状态混合器），但参数完全不同 —— 那里是 k/v 头 + delta 规则，
            # 这里是逐头的 A_log / D / dt_bias + 一个深度卷积。
            #
            # 契约里的 8 个张量（Nemotron 那一族，23 层）：
            #   in_proj.weight (10304, 2688)  = d_inner + conv_dim + heads
            #   conv1d.weight  (6144, 1, 4)   = conv_dim, 深度卷积
            #   conv1d.bias    (6144)
            #   A_log / D / dt_bias  (64,)    逐头
            #   norm.weight    (4096,)        d_inner
            #   out_proj.weight (2688, 4096)
            h = _num(attrs.get("heads"))
            hd = _num(attrs.get("head_dim"))
            ss = _num(attrs.get("ssm_state"))
            ng = _num(attrs.get("n_groups"))
            miss = [k for k, v in (("heads", h), ("head_dim", hd),
                                   ("ssm_state", ss), ("n_groups", ng))
                    if not v]
            if miss:
                err(f"机制 '{name}' 的 SSM 缺少 {', '.join(miss)}")
                return None
            h, hd, ss, ng = int(h), int(hd), int(ss), int(ng)
            if h % ng:
                err(f"机制 '{name}': heads={h} 不是 n_groups={ng} 的整数倍")
                return None
            if not _ck("act", attrs.get("act"), name, kind):
                return None
            return {"kind": "SSM", "mech": name, "attrs": {
                "heads": h, "head_dim": hd, "ssm_state": ss, "n_groups": ng,
                "conv_kernel": int(_num(attrs.get("conv_kernel"), 4)),
                "expand": int(_num(attrs.get("expand"), 2)),
                "chunk_size": int(_num(attrs.get("chunk_size"), 128)),
                "conv_bias": _flag(attrs.get("conv_bias"), True),
                "proj_bias": _flag(attrs.get("proj_bias")),
                "act": str(attrs.get("act", "silu")),
                # 跟着模型的 rms_norm_eps 走 —— 这个 bug 在 GDN 和 MLA
                # 上各犯过一次，写死 1e-6 而模型是 1e-5。
                "norm_eps": float(_num(attrs.get("norm_eps"),
                                       _num(hp.get("rms_eps"), _DEFAULT_EPS)))}}
        return None

    # position 块里写 `learned(N)` 就是"学一张表"，不是 RoPE。
    # 解析器把它当成一个叫做 learned 的 rope_type —— 这里认这个名字。
    _pos_learned = False
    _n_pos_decl = None
    # `info["position"]` 是 {名字: 声明原文}（by1check 那边已经解出来了）。
    # **不要用 try/except 包住** —— 上一版引用了不存在的变量（position_block），
    # 异常被吞掉，于是 pos_kind 一直是 rope，而没有任何地方说不对劲。
    for _pv in ((info.get("position") or {}).values()):
        # 直接看原文 —— parse_rope 在 by1check 里，codegen 这边没有它。
        # 上一版就是引用了不存在的名字，而 try/except 把 NameError 吞了，
        # 于是 pos_kind 一直是 rope 而没有任何地方说不对劲。
        _s = str(_pv)
        if "learned" in _s.lower():
            _pos_learned = True
            _mm = re.search(r"learned\s*\(\s*(\d+)", _s)
            if _mm:
                _n_pos_decl = _mm.group(1)

    def ops_of(mixer, attached, layer_attrs):
        """把一层展开成算子序列。数据流用 ValueRef 表示，没有张量形状。"""
        ops: List[dict] = []
        env = "hidden"

        # **用哪种归一化是模型级的选择。**
        # 以前这里两处都写死 "RMSNorm" —— 于是 GPT-2 被生成成一个用
        # RMSNorm 的模型，而 .by1 里**根本没有地方能说出口**。
        # 这不是"默认值选错了"，是"只能 RMSNorm"。
        _nk = str(hp.get("norm_kind", "rms")).strip().lower()
        if _nk not in ("rms", "layer"):
            err("hparams.norm_kind = <%s> —— 只有 rms 和 layer 两种"
                % hp.get("norm_kind"))
            return None
        # 层归一化的 eps **必须跟着模型的 rms_eps 走**。
        # 三个后端原来各自写死 1e-5 —— 于是它们"一致地错"，
        # 互相对拍全绿，却都不符合 .by1 里写的值。
        _n_attrs = {"one_plus": norm_1p, "kind": _nk,
                    "eps": float(_num(hp.get("rms_eps"), _DEFAULT_EPS))}
        _n_mech = "LayerNorm" if _nk == "layer" else "RMSNorm"

        def emit(mk, ins):
            j = len(ops)
            out = f"op{j}.out"
            ops.append({"mech": mk["mech"], "kind": mk["kind"],
                        "attrs": mk["attrs"], "inputs": ins, "outputs": [out]})
            return out

        if mixer is not None:
            n = {"mech": _n_mech, "kind": "Norm", "attrs": dict(_n_attrs)}
            v = emit(n, [env])
            a = emit(mixer, [v])
            j = len(ops)
            ops.append({"mech": "Add", "kind": "Add", "attrs": {},
                        "inputs": [env, a], "outputs": [f"op{j}.out"]})
            env = f"op{j}.out"
        for am in attached:
            n = {"mech": _n_mech, "kind": "Norm", "attrs": dict(_n_attrs)}
            v = emit(n, [env])
            m = emit(am, [v])
            j = len(ops)
            ops.append({"mech": "Add", "kind": "Add", "attrs": {},
                        "inputs": [env, m], "outputs": [f"op{j}.out"]})
            env = f"op{j}.out"
        return ops

    layers = []
    _ov_all = info.get("overrides") or {}
    for li, (_s, m, a, _k, atts) in enumerate(info["layer_seq"]):
        # **记下"报之前"的条数** —— 用来判 `one_mech` 有没有说话。
        _before = len(errs) + len(unsup)
        mixer = one_mech(m, a, li)
        _spoke = (len(errs) + len(unsup)) > _before

        def _with_ov(am, _s=_s, li=li):
            """挂在层上的机制，属性要合并这一层的逐层覆盖。
            不做的话：契约按逐层算、生成的计算全用默认值 —— 一个
            **看起来对的错模型**（张量检查全过，跑起来宽度都一样）。"""
            base_a = dict(defaults.get(am, {}))
            base_a.update(_ov_all.get("%s|%d|%s" % (_s, li, am), {}))
            return base_a

        attached = [one_mech(am, _with_ov(am), li) for am in atts]
        attached = [x for x in attached if x]
        # **逃生舱也算 token 混合器。** 这里原来只认 Attention / Linear / MLA
        # —— 于是 `Raw` 会被判成"没有混合器"，而理由是错的
        # （看起来像"这层没写混合器"，实际是"codegen 不认这个种类"）。
        if mixer is None:
            # **`one_mech` 已经说过原因了，就别再说一遍。**
            #
            # 它返回 None 有两种可能：
            #   ① 它报过（机制种类不认 / 属性不认识 / 取值非法）—— 那原因
            #      已经在 errs 或 unsup 里了，这里再说一句是**一个原因报两次**
            #   ② 它一个字都没说 —— 机制名在 `mechs` 里找不到，
            #      那"这层没有混合器"才是真的新信息
            #
            # 不分开的后果实测过：GLM-5.3-Flash 报了 34 句
            # "第 N 层没有可用的 token 混合器"，而根因只有一句
            # "机制 'KDA' 的类型是 'KDA'，codegen 还不支持" ——
            # 那一句在 unsup 桶里，被这 34 句挤掉了。
            if not _spoke:
                err(f"第 {li} 层没有可用的 token 混合器 —— 生成不出来")
            continue
        if mixer["kind"] not in MIXER_KINDS:
            # **种类认得出来、但它不是 token 混合器 —— 这是"处理不了"，
            # 不是"描述写错了"。**
            #
            # 我一开始把它归成 err（描述的问题），于是 `by1gate` 立刻红了：
            # 它要求 Nemotron 被拒时**理由里提到 SSM**，而这一句
            # "第 1 层的主机制是 'MoE'，不是 token 混合器" 把
            # `unsup` 里那句 SSM 遮住了（`errs` 优先）。
            #
            # 想清楚是这样：Nemotron 的层是 Attn / MoE / SSM 混排的
            # （MTP 那条辅助栈的第一层主机制就是 MoE）——**那不是写错，
            # 是 codegen 处理不了这种层。**
            unsupported(f"第 {li} 层的主机制是 '{mixer['kind']}'，"
                        f"codegen 不把它当 token 混合器")
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
        elif mixer["kind"] == "Raw":
            # **逃生舱的状态由实现方声明，codegen 推不出来。**
            # 这里原来是 `mixer["attrs"]["window"]` —— 无条件假设每个混合器
            # 都有 window，于是 Raw 直接 KeyError，而报错长得像"这层缺属性"。
            # 说"不知道"比崩掉诚实。
            state.append({"kind": "declared_by_impl", "bounded_by": None,
                          "shape": None, "dtype": None, "reuse": "unknown"})
        elif mixer["attrs"].get("window"):
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
        # **写坏了优先。** 一份文件里既有语法/属性错、又有不认的机制时，
        # 它是"坏文件"，不是"没验"。
        raise CodegenError("\n".join("  - " + e for e in errs))
    if unsup:
        raise UnsupportedError("\n".join("  - " + e for e in unsup))

    return {
        # **版本号。** 没有它的 IR 不该被接受 —— 读的一方无从判断
        # 自己理解的是哪一版。见 by1ir.py。
        "by1-ir": _IR_VER,
        "vocab": vocab, "ctx": ctx, "d_model": d_model,
        "norm_one_plus": norm_1p,
        # 最终归一化在 ops_of 之外建，所以这两个要放到 IR 顶层
        "norm_kind": str(hp.get("norm_kind", "rms")).strip().lower(),
        # **学习式位置编码**（GPT-2 的 wpe）。它和 RoPE 是两种东西：
        # 一个是一张学出来的查表、加在输入上；另一个是旋转，在注意力里。
        # 前者是**输入的一部分**，所以放在 IR 顶层。
        "pos_kind": "learned" if _pos_learned else "rope",
        "n_pos": int(_num(_n_pos_decl, _num(hp.get("n_pos"), ctx)) or ctx),
        "norm_eps": float(_num(hp.get("rms_eps"), _DEFAULT_EPS)),
        "globals": [
            {"mech": "Embed", "kind": "Embed", "attrs": {},
             "inputs": [], "outputs": ["hidden"]},
            # **最终归一化也得带 kind。** 我加 LayerNorm 那一次只改了层里的
            # 两个 Norm，这个漏了 —— 于是 GPT-2 建出来是 25 个 LayerNorm，
            # 而 IR 里这一个说是 rms。规格校验一跑就露出来了。
            {"mech": "FinalNorm", "kind": "Norm",
             "attrs": {"kind": str(hp.get("norm_kind", "rms")).strip().lower(),
                       "eps": float(_num(hp.get("rms_eps"), _DEFAULT_EPS)),
                       "one_plus": norm_1p},
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
import by1skip


# ⚠️ **从这里往下的 `1e-05` 是"写进产物"的字面量，不是可引用的常量。**
#
# 这一整段会被渲染成文本、在一个**空命名空间**里 exec ——
# 它看不见本模块的任何全局名（`_DEFAULT_EPS` 也不例外）。
# 所以这些位置只能是字面量。值本身仍然是 `by1ir.EPS_DEFAULT`；
# 两者一致由 `by1lint` 之外的这条约定守着：**生成器里的 eps 只能出现在这里**，
# 而 `compile_ir`（真正决定 IR 那几个 eps 的地方）用的是常量名。
# （第一版我把这些也换成常量名，生成出来的代码当场 NameError；
#   第二版换成 `{_DEFAULT_EPS!r}`，而这几行在普通字符串里，原样进了产物 ——
#   两次都是"看着像改了"。）


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-05, one_plus=False):
        super().__init__()
        # 两种约定：直接乘 w，或者乘 (1+w)。后者是 zero-centered，w 初值为 0。
        # 同一个模型里两种可以并存 —— Qwen3-Next 就是这样。
        self.one_plus = one_plus
        self.w = nn.Parameter(torch.zeros(d) if one_plus else torch.ones(d))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * ((1.0 + self.w) if self.one_plus else self.w)


class LayerNorm(nn.Module):
    """GPT-2 那种归一化。

    和 RMSNorm 的差别**不只是多一个 bias** —— 它还要**减均值**。
    名字都叫"层归一化"，算的不是一回事；而 RMSNorm 那边
    `one_plus` 又是第三种约定。三个东西顶着相近的名字。
    """
    def __init__(self, d, eps=1e-05):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.w + self.b


def rope_tables(head_dim, n, base, device, yarn=None):
    half = head_dim // 2
    if yarn and (yarn.get("type") or "yarn") == "llama3":
        # llama3 式缩放（Llama 3.1）：按**波长**分三段，中间那段做平滑插值。
        # 和 YaRN 不是一回事 —— YaRN 调的是 ramp 的起止维，llama3 调的是波长阈值。
        fac = float(yarn["factor"])
        hf, lf = float(yarn["high_freq"]), float(yarn["low_freq"])
        old = float(yarn["original"])
        # **必须取倒数**：参考是 inv_freq = 1/(base**(arange(0,dim,2)/dim))。
        # 漏了 1.0/ 的话，第一个频率正好变成它的倒数（1.27 而参考是 0.786）。
        p0 = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
        wl = 2 * math.pi / p0
        lo_wl, hi_wl = old / lf, old / hf
        out = torch.where(wl > lo_wl, p0 / fac, p0)
        smooth = (old / wl - lf) / (hf - lf)
        out = torch.where((wl >= hi_wl) & (wl <= lo_wl),
                          (1 - smooth) * out / fac + smooth * out, out)
        inv = out
    elif yarn:
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
        _eps = a.get("norm_eps", 1e-05)
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
        # **模型用学习式位置编码时，注意力不做旋转**（GPT-2）。
        # 位置在输入上，不在 q/k 上。以前无条件转 —— 而 .by1 里
        # 没有地方能说"别转"，所以 GPT-2 被静默转了一遍。
        self.rope = bool(a.get("rope", True))
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
        if self.qk_norm not in ("off", "", None):
            _op = a.get("norm_one_plus", False)
            _eps = a.get("norm_eps", 1e-05)
            # **两种 qk_norm 的区别是"在哪一步归一化"**：
            #   per_head  拆头之后，按 head_dim（多数模型）
            #   full      拆头**之前**，按 q*head_dim 整宽（Instella 是 [2560] 而不是 [80]）
            # 名字一样、位置不同、宽度也不同。
            if self.qk_norm == "full":
                self.qn = RMSNorm(self.q * self.hd, one_plus=_op, eps=_eps)
                self.kn = RMSNorm(self.kv * self.hd, one_plus=_op, eps=_eps)
            else:
                self.qn = RMSNorm(self.hd, one_plus=_op, eps=_eps)
                self.kn = RMSNorm(self.hd, one_plus=_op, eps=_eps)
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
        qflat = self.wq(x)
        kflat = self.wk(x)
        if self.qk_norm == "full":
            # 整宽：作用在**扁平化之后、拆头之前**的输出上
            qflat = self.qn(qflat)
            kflat = self.kn(kflat)
        qq = qflat.view(b, n, self.q, w)
        gate = None
        if self.q_gate:
            q, gate = torch.chunk(qq, 2, dim=-1)
        else:
            q = qq
        q = q.transpose(1, 2)
        kraw = kflat.view(b, n, self.kv, self.hd).transpose(1, 2)
        v = kraw if self.kv_tie else \
            self.wv(x).view(b, n, self.kv, self.hd).transpose(1, 2)
        k = kraw
        if self.qk_norm not in ("off", "", None) and self.qk_norm != "full":
            q, k = self.qn(q), self.kn(k)
        # partial：只转前一段维度，剩下的原样带走
        np_ = int(self.hd * self.rope_partial)
        if not self.rope:
            pass                        # 不转：位置在输入上（wpe）
        elif 0 < np_ < self.hd:
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


def _gelu(x, style="gelu"):
    """GPT-2 用的是 tanh 近似（HF 里叫 gelu_new）。

    精确式和近似式**不是一回事** —— 差在 1e-3 量级，
    比前面所有对拍的阈值都大。所以两个都要有，不能混。
    """
    if style in ("gelu_new", "gelu_tanh", "tanh"):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return F.gelu(x)


def _act(x, style):
    """**无门控**那一层用的激活。

    原来那条分支写死 `F.silu` —— style 直接丢了。
    GPT-2 的 MLP 是无门控 + gelu，于是拿到的是 silu，
    而 .by1 里写的 `act = gelu_new` 一声不吭地被忽略。
    （和 gate = none 变成 true 是同一类：一条分支只为一种情况写过。）
    """
    if style in ("gelu", "gelu_new", "gelu_tanh", "tanh"):
        return _gelu(x, style)
    if style == "relu2":
        return F.relu(x) ** 2
    return F.silu(x)


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


def _load_raw(impl):
    """从 raw.py 里取工厂函数 raw_<impl>。

    **找不到就报错，不静默给个恒等。** 一个"看起来对但什么都没算"的模型
    比一个报错糟得多。
    """
    import importlib.util as _ilu
    import os as _os
    # **`__file__` 在 exec 出来的代码里不存在** —— 生成的模块是被 exec 的，
    # 不是从磁盘 import 的。所以这里不能直接用它，否则会 NameError，
    # 而那会伪装成"找不到 raw.py"（我第一次就是被这个骗过去的：
    # 反例测试"通过"了，其实是因为崩在别处）。
    # **候选按"离模型有多远"排。**
    #
    # ① 模型文件旁边 —— `raw.py` 跟着 `raw-escape.by1` 走，两个都在
    #    `models/`。这个目录由 `render_ir()` / `render()` 注入成字面量
    #    （生成物里的 `BY1_MODEL_DIR`），因为**被 exec 的代码里没有
    #    `__file__`**，这里自己算不出来。
    cands = []
    _d = globals().get("BY1_MODEL_DIR")
    if _d:
        cands.append(_os.path.join(_d, "raw.py"))
    # ② cwd —— 自己手跑、或者把 raw.py 放在别处做实验时用。
    cands.append("raw.py")
    try:
        cands.append(_os.path.join(_os.path.dirname(__file__), "raw.py"))
    except NameError:
        pass
    cands.append(_os.path.join(_os.getcwd(), "raw.py"))
    # 去重但**保序** —— 报错里列出的顺序就是查找顺序。
    uniq = []
    for c in cands:
        if c not in uniq:
            uniq.append(c)
    cands = uniq
    for cand in cands:
        if not _os.path.exists(cand):
            continue
        spec = _ilu.spec_from_file_location("by1raw", cand)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "raw_" + impl, None)
        if fn is None:
            # 用 RuntimeError，不用 CodegenError —— 这段是**生成出去的代码**，
            # 在那边 CodegenError 根本不存在。第一版就是栽在这儿：
            # 报错变成了 NameError，而 NameError 看起来像别的问题。
            raise RuntimeError(
                "raw.py 里没有 raw_%s —— 逃生舱的工厂函数必须叫这个名字"
                % impl)
        return fn
    raise RuntimeError(
        "机制用了逃生舱（impl = %s），但找不到 raw.py（找过：%s）"
        % (impl, ", ".join(cands)))


class ExternalMech(nn.Module):
    """**逃生舱第二层：引用一个外部符号。**

    和 RawMech 的差别是**编译器不认识这个机制** —— 它只认识 ABI
    （见 by1ext.py）。所以加一个新机制不需要动编译器，
    只要有一个 .so 和一份 IR。

    契约没松：这个算子的张量照样声明、照样被 by1verify 对产物查。
    """
    def __init__(self, a, d):
        super().__init__()
        import by1ext as _ext
        self._ext = _ext
        self._lib_path = a["lib"]
        self._symbol = a["symbol"]
        names = _ext.sorted_weight_names(a["weights"].keys())
        self._names = names
        # 参数挂成**和契约一样的路径**（`scale.weight` -> 子模块 scale 的 weight）
        for nm in names:
            parts = nm.split(".")
            mod = self
            for q in parts[:-1]:
                if not hasattr(mod, q):
                    setattr(mod, q, nn.Module())
                mod = getattr(mod, q)
            setattr(mod, parts[-1],
                    nn.Parameter(torch.zeros(*a["weights"][nm])))

    def forward(self, x):
        import numpy as _np
        fn = self._ext.bind(self._ext.load_lib(self._lib_path), self._symbol)
        B, T, D = x.shape
        # **ABI 是 float*，所以这边必须给 NumPy 数组。**
        # 第一版直接把 torch.Tensor 传过去 —— 而 Tensor 没有 .ctypes，
        # 报的是 AttributeError，长得像别的问题。
        xn = _np.ascontiguousarray(x.detach().float().numpy())
        yn = _np.zeros_like(xn)
        ws = []
        for nm in self._names:
            v = self
            for q in nm.split("."):
                v = getattr(v, q)
            ws.append(_np.ascontiguousarray(v.detach().float().numpy()))
        self._ext.call(fn, xn, yn, int(B), int(T), int(D), ws)
        return torch.from_numpy(yn).to(x.dtype)


class RawMech(nn.Module):
    """**逃生舱。**

    它和别的机制**受同样的约束**：张量契约要声明、三个后端要对拍、
    取值门要过。区别只是"这一段计算不在这门语言能表达的范围内"，
    所以写在 raw.py 里。

    所以它下去之后**照样被验** —— 这是它和"绕过检查"的区别。
    """
    def __init__(self, a, d):
        super().__init__()
        self.inner = _load_raw(a["impl"])(d, dict(a))

    def forward(self, x):
        return self.inner(x)


class MLP(nn.Module):
    def __init__(self, a, d):
        super().__init__()
        self.gate = a["gate"]
        self.style = a.get("act", "silu")
        self.limit = a.get("limit")
        self.alpha = a.get("alpha", 1.702)
        _b = bool(a.get("bias", False))
        self.w1 = nn.Linear(d, a["hidden"], bias=_b)
        self.w2 = nn.Linear(a["hidden"], d, bias=_b)
        if self.gate:
            self.w3 = nn.Linear(d, a["hidden"], bias=_b)

    def forward(self, x):
        g = self.w1(x)
        if self.gate:
            return self.w2(_swiglu(g, self.w3(x), self.style, self.limit, self.alpha))
        # style 要传下去 —— 见 _act 的注释
        return self.w2(_act(g, self.style))


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
        # 共享专家的夹取值是独立的（Step-3.7：路由专家 7、共享专家 16）。
        # **这行原来被我加到了 MLP 类上** —— 锚点 self.limit 先匹配到那里，
        # 于是 MoE.forward 用 self.limit_shared 时 AttributeError。
        self.limit_shared = a.get("limit_shared")
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
        elif self.routing == "sigmoid_topk":
            # sigmoid 打分 + 直接 top-k，**不分组**（Step-3.7）。
            # 和 sigmoid_group_topk 的差别就是少了"先选组"那一步 ——
            # 名字只差一个词，但选出来的专家不一样。
            # **这一支原来没有**：`sigmoid_topk` 会落进下面的 else，
            # 走成 softmax_topk。而 step-3.7 的前向从来没跑过，所以没人发现。
            sc = logits.sigmoid()
            choice = sc + self.score_bias if self.score_bias is not None else sc
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
            # 共享专家也要走 _swiglu —— 它有自己的夹取值，而且风格（silu / gptoss）
            # 必须和路由专家一致。原来这里是裸的 F.silu。
            se = self.sw2(_swiglu(self.sw1(xf), self.sw3(xf), self.style,
                                  self.limit_shared, self.alpha))
            if getattr(self, "shared_gate", None) is not None:
                se = se * torch.sigmoid(self.shared_gate(xf))
            out = out + se
        return out.reshape(b, n, d)


def _mk_norm(attrs, d):
    """归一化：**两种不是一回事**，只能按 attrs 里说的来。

    rms   只除均方根 —— 不减均值，没有 bias
    layer 减均值、除标准差、还有一个 bias（GPT-2 那种）

    参数名对应关系也不同（`w` vs `w`+`b`），所以选错了张量契约就对不上 ——
    但契约查不出**算错了**，所以这里必须是显式的。
    """
    if str(attrs.get("kind", "rms")).lower() == "layer":
        return LayerNorm(d, eps=attrs.get("eps", 1e-05))
    return RMSNorm(d, eps=attrs.get("eps", 1e-05),
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


class RMSNormGatedGrouped(nn.Module):
    """**按组**的 gated RMSNorm（Mamba2 那一版）。

    和上面那个 `RMSNormGated` **三处都不同**，所以是两个东西：

        · 归一化在 `group_size = d / n_groups` 那一维上，不是整个 d
        · **gate 乘在 norm 之前**（`x * silu(gate)` 再归一化），
          而上面那个是归一化之后再乘
        · 权重照旧乘在最后

    **同名不同义** —— 这个仓库里已经栽过几次（`qk_norm` 的 full/per_head、
    `one_plus` 的三种约定）。所以宁可两个类，不合并成一个带开关的。
    """

    def __init__(self, d, n_groups, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.gs = d // n_groups

    def forward(self, x, gate):
        y = x * F.silu(gate.float())
        sh = y.shape
        g = y.reshape(*sh[:-1], sh[-1] // self.gs, self.gs)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * g.reshape(*sh).to(x.dtype)


class Mamba2(nn.Module):
    """**选择性状态空间**（Mamba2 · Nemotron-H 那一版）。

    和注意力是**不同的算法族** —— 没有 q/k/v，状态是 `A_log` / `D` /
    `dt_bias` 驱动的，随 token 递推：

        h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t ⊗ B_t)
        y_t = C_t · h_t + D * x_t

    **判卷人**：`src/modelcheck/by1ssm.py` —— 对着官方
    `NemotronHMamba2Mixer`，两种尺寸 + 三条反例。

    **注意判卷人是 `NemotronH*` 而不是 `MambaForCausalLM`** ——
    纯 Mamba 的 `in_proj` 是 `intermediate_size * 2`，而这一版是

        d_inner + conv_dim + heads = 4096 + 6144 + 64 = 10304

    **这一支走的是递推，不是官方的分块扫描** —— 官方有 CUDA kernel，
    这里只有 CPU。数学相同，浮点归约顺序不同。
    """

    def __init__(self, a, d):
        super().__init__()
        self.h = a["heads"]
        self.hd = a["head_dim"]
        self.ssm = a["ssm_state"]
        self.ng = a["n_groups"]
        self.k = a["conv_kernel"]
        self.act = a["act"]
        self.d_inner = self.h * self.hd
        self.conv_dim = self.d_inner + 2 * self.ng * self.ssm
        self.rep = self.h // self.ng
        _b = a["proj_bias"]
        self.in_proj = nn.Linear(d, self.d_inner + self.conv_dim + self.h,
                                 bias=_b)
        # **深度卷积**：groups = conv_dim，padding = k-1（因果）。
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, self.k,
                              groups=self.conv_dim, bias=a["conv_bias"],
                              padding=self.k - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.h))
        self.A_log = nn.Parameter(torch.zeros(self.h))
        self.D = nn.Parameter(torch.ones(self.h))
        self.norm = RMSNormGatedGrouped(self.d_inner, self.ng,
                                        eps=a.get("norm_eps", 1e-6))
        self.out_proj = nn.Linear(self.d_inner, d, bias=_b)

    def forward(self, x):
        b, s, _ = x.shape
        h, hd, ng, ssm = self.h, self.hd, self.ng, self.ssm
        proj = self.in_proj(x)
        gate, bc, dt = torch.split(
            proj, [self.d_inner, self.conv_dim, h], dim=-1)

        # ② 因果深度卷积 + 激活
        bc = self.conv(bc.transpose(1, 2))[:, :, :s].transpose(1, 2)
        bc = F.silu(bc)

        hs, B, C = torch.split(bc, [self.d_inner, ng * ssm, ng * ssm], dim=-1)
        xs = hs.view(b, s, h, hd)
        B = B.view(b, s, ng, ssm).repeat_interleave(self.rep, dim=2)
        C = C.view(b, s, ng, ssm).repeat_interleave(self.rep, dim=2)

        # ③ SSM 递推。状态是 [b, h, hd, ssm]，所以
        #    更新是**外积**（x 在 hd 维、B 在 ssm 维），
        #    读出是 **h 右乘 C**。这两处最容易写反 ——
        #    写反了会炸在维度上，不会静默算错（第一次就是这么发现的）。
        A = -torch.exp(self.A_log.float())
        dtt = F.softplus(dt + self.dt_bias)
        st = x.new_zeros(b, h, hd, ssm)
        ys = []
        for t in range(s):
            d_t = dtt[:, t][..., None, None]
            st = (torch.exp(d_t * A[None, :, None, None]) * st
                  + d_t * (xs[:, t][..., None] * B[:, t][:, :, None, :]))
            ys.append((st @ C[:, t][..., None]).squeeze(-1)
                      + self.D[None, :, None] * xs[:, t])
        y = torch.stack(ys, 1).reshape(b, s, self.d_inner)

        # ④ 按组 gated norm（gate 在 norm **之前**）+ 输出投影
        return self.out_proj(self.norm(y, gate))


BUILDERS = {"Norm": _mk_norm, "Attention": Attention,
            "FFN": MLP, "MoE": MoE, "Linear": GatedDeltaNet,
            "MLA": MLAttention, "SSM": Mamba2,
            "Raw": RawMech, "External": ExternalMech}


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
        self.wpe = (nn.Embedding(int(ir.get("n_pos", ir.get("ctx", 1024))), d)
                    if ir.get("pos_kind") == "learned" else None)
        self.layers = nn.ModuleList([LayerMod(L, d) for L in ir["layers"]])
        # **最终归一化也得跟着模型的选择走。** 它不在 ops_of 里，
        # 所以刚才那一处改动没带上它 —— GPT-2 建出来是 24 个 LayerNorm
        # 加 1 个 RMSNorm，而那个 1 就是这里。
        _fk = str(ir.get("norm_kind", "rms")).lower()
        self.final_norm = (LayerNorm(d, eps=ir.get("norm_eps", 1e-05))
                           if _fk == "layer"
                           else RMSNorm(d, one_plus=ir.get("norm_one_plus", False),
                                        eps=ir.get("norm_eps", 1e-05)))
        self.head = nn.Linear(d, ir["vocab"], bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        if self.wpe is not None:
            # 学习式位置编码：**加在输入上**，不是旋转。
            pos = torch.arange(idx.size(1), device=idx.device)
            x = x + self.wpe(pos)
        for blk in self.layers:
            x = blk(x)
        return self.head(self.final_norm(x))


def build():
    return By1Model(IR)
'''


def _model_dir(by1_name):
    """**`raw.py` 就在模型文件旁边。**

    生成出去的代码是被 `exec` 的 —— **那里没有 `__file__`**，所以
    `_load_raw` 自己算不出模型在哪。那个目录必须由这里算好、注入成
    字面量（生成物里的 `BY1_MODEL_DIR`）。

    拿不到真实路径时（`render_ir(ir, 'booted.ir')` 这类合成名字）
    退回 `models/` —— 那正是模型和 `raw.py` 该在的地方。
    """
    if by1_name and os.path.exists(by1_name):
        return os.path.dirname(os.path.abspath(by1_name))
    try:
        import by1paths
        return by1paths.MODELS
    except ImportError:
        return os.getcwd()


def render_ir(ir: Dict[str, Any], by1_name: str = "model.ir.json") -> str:
    """**只吃 IR 的入口。** 语言层（.by1）到 IR 是另一件事。

    规格说 IR 是接口 —— 那就得有一条路，不经过 .by1 也能跑。
    想写第四个后端的人从这里开始，不需要先学 by1。
    """
    import by1ir as _ir
    _errs = _ir.validate(ir)
    if _errs:
        raise CodegenError("IR 不合法：\n  " + "\n  ".join(_errs[:10]))
    return (
        f"# {_stamp()}\n"
        f"# 由 by1 从 {by1_name} 生成 —— 改 IR 再重新生成，不要手改这个文件\n"
        f"# 后端无关的 IR；下面的 RUNTIME 只是它的一个后端（PyTorch）\n\n"
        f"IR = {pprint.pformat(ir, indent=2, width=86, sort_dicts=False)}\n"
        f"# **逃生舱（Raw）的 `raw.py` 就在这里** —— 就在模型文件旁边。\n"
        f"BY1_MODEL_DIR = {_model_dir(by1_name)!r}\n"
        + RUNTIME
    )


def render(info: Dict[str, Any], by1_name: str = "model.by1") -> str:
    ir = compile_ir(info)
    return (
        f"# {_stamp()}\n"
        f"# 由 by1 从 {by1_name} 生成 —— 改 .by1 再重新生成，不要手改这个文件\n"
        f"# 后端无关的 IR；下面的 RUNTIME 只是它的一个后端（PyTorch）\n\n"
        f"IR = {pprint.pformat(ir, indent=2, width=86, sort_dicts=False)}\n"
        f"# **逃生舱（Raw）的 `raw.py` 就在这里** —— 就在模型文件旁边。\n"
        f"BY1_MODEL_DIR = {_model_dir(by1_name)!r}\n"
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
        # 没给 .by1 -> by1skip.CALLER
        return by1skip.CALLER
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
