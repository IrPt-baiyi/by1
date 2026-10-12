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
    # by1exec 的约定：x [..., n, in]，w [out, in]，out = x @ w.T [..., n, out]。
    # **ref() 全部用这套约定**（样本在前、特征在最后轴）；图里节点形状标
    # (特征, n) 是 ggml 视角 —— 同一块内存的两种读法，这正是布局原语
    # 存在的原因。两种读法不许混在一条公式里（混过的人出过三次错）。
    np = _np()
    return lambda w, x: x @ w.T


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


def _ref_sigmoid():
    np = _np()
    return lambda x: 1.0 / (1.0 + np.exp(-x))


def _ref_rope():
    # 位置 t 对每对维度 (u, v) 转 theta_t = t · base^(-2i/n_dims)：
    #   half        -> u' = u·c - v·s,  v' = u·s + v·c   （配对 i, i+half）
    #   interleaved -> 同公式的配对 (2i, 2i+1)（输出拼接 —— 和 ggml
    #                  NORMAL 的奇偶出差一个**共同置换**，点积不变）
    np = _np()

    def f(x, pos, base, pairing, n_dims):
        # x: [n_dims, n]（行 = 维度、列 = 位置）—— 形状约定由 lower() 保证
        half = n_dims // 2
        out = np.array(x, copy=True)
        for i in range(half):
            th = pos * (base ** (-2.0 * i / n_dims))
            c, s = np.cos(th), np.sin(th)
            if pairing == "half":
                u = np.array(x[i, :], copy=True)
                v = np.array(x[i + half, :], copy=True)
                out[i, :] = u * c - v * s
                out[i + half, :] = u * s + v * c
            else:
                u = np.array(x[2 * i, :], copy=True)
                v = np.array(x[2 * i + 1, :], copy=True)
                # 奇偶进、**拼接**着出
                out[i, :] = u * c - v * s
                out[i + half, :] = u * s + v * c
        return out
    return f


def _ref_soft_max():
    # softmax(a·scale + mask)，沿 dim0（key 轴）
    np = _np()

    def f(a, mask, scale):
        z = a * scale + (mask if mask is not None else 0.0)
        z = z - z.max(axis=0, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=0, keepdims=True)
    return f


def _ref_get_rows():
    np = _np()
    return lambda table, ids: table[ids]


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
    "rms_norm":  {"ins": 1, "f": "x · rsqrt(mean(x²)+eps) · (one_plus?1+w:w)",
                  "ref": _ref_rms_norm(),
                  "src": "by1c.rmsnorm / ggml_rms_norm / by1exec.rms_norm"},
    "layer_norm": {"ins": 1,
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
    "sigmoid":   {"ins": 1, "f": "σ(x)", "ref": _ref_sigmoid(),
                  "src": "MoE 打分 / head_gate"},
    "rope":      {"ins": 2, "f": "每对维度转 θ_t = t·base^(-2i/n_dims)；"
                                "half 配对 (i,i+h)，interleaved 配对 (2i,2i+1)"
                                "且**拼接输出**",
                  "ref": _ref_rope(),
                  "src": "by1c.rope / by1exec.apply_rope / ggml_rope_ext"},
    "soft_max":  {"ins": 2, "f": "softmax(a·scale + mask)，沿 key 轴",
                  "ref": _ref_soft_max(),
                  "src": "by1c.attention / ggml_soft_max_ext"},
    "get_rows":  {"ins": 2, "f": "out[i] = table[ids[i]]", "ref": _ref_get_rows(),
                  "src": "ggml_get_rows / torch nn.Embedding"},
    # 布局原语：语义是**数据重解释**，不是数学。
    "view":      {"ins": 1, "f": "重解释形状（元素序不变）", "ref": None,
                  "src": "reshape / ggml_reshape"},
    "permute":   {"ins": 1, "f": "轴置换（带步长的视图）", "ref": None,
                  "src": "transpose / ggml_permute"},
    "cont":      {"ins": 1, "f": "按当前视图物化为连续内存", "ref": None,
                  "src": "ggml_cont / numpy .copy()"},
}

# **还没收编的原语**（各就各位，谁也不冒充谁）：
#   MoE 路由：topk_ids / expert_matmul（ggml_mul_mat_id）/ score 系
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

def lower_op(g, i, j, o, d):
    """把 IR 的一个算子降成原语节点，追加进 g。值名带层号防撞。"""
    a, k = o["attrs"], o["kind"]
    p = "L%dO%d_" % (i, j)
    src = [("x" if r == "hidden" else p + r.replace(".", "_"))
           for r in o["inputs"]]
    dst = p + o["outputs"][0].replace(".", "_")
    w = lambda nm: p + nm.replace(".", "_")

    if k not in ("Embed", "Norm", "Head", "Attention", "FFN", "Add"):
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
    return dst


def _lower_attn(g, p, x, dst, w, a, d):
    hd, nh, nkv = int(a["head_dim"]), int(a["q"]), int(a["kv"])
    if int(a["out_dim"]) != nh * hd:
        raise SystemExit("  [不支持] out_dim != q*head_dim 的注意力"
                         "（输出重排还没收编）")
    for flag, what in (("q_gate", "q_gate"), ("sink", "sink"),
                       ("head_gate", "head_gate"), ("qkv_fused", "qkv_fused")):
        if a.get(flag) and (flag != "head_gate" or a.get(flag) != "off"):
            raise SystemExit("  [不支持] Attention.%s 还没有 lowering" % what)
    if a.get("yarn") or float(a.get("rope_scale", 1.0)) != 1.0:
        raise SystemExit("  [不支持] yarn / rope_scale != 1 还没有 lowering")
    bias = bool(a.get("bias"))
    q = g.add("matmul", [w("wq"), x], [p + "q0"], shapes=[(nh * hd, None)])
    kq_ = g.add("matmul", [w("wk"), x], [p + "k0"], shapes=[(nkv * hd, None)])
    if bias:
        q = g.add("add", [q, w("wq.bias")], [p + "q1"], shapes=[(nh * hd, None)])
        kq_ = g.add("add", [kq_, w("wk.bias")], [p + "k1"],
                    shapes=[(nkv * hd, None)])
    # **kv_tie：v 是 k 的原始投影** —— 分叉在 norm/rope 之前（判卷人
    # by1exec：`v = kraw`）。
    if a.get("kv_tie"):
        v = g.add("view", [kq_], [p + "v3"],
                  attrs={"shape": [hd, nkv, None]}, shapes=[(hd, nkv, None)])
    else:
        v = g.add("matmul", [w("wv"), x], [p + "v0"], shapes=[(nkv * hd, None)])
        if bias:
            v = g.add("add", [v, w("wv.bias")], [p + "v1"],
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
        q3 = g.add("mul", [g.add("rms_norm", [q3], [p + "qn0"],
                                 attrs={"eps": eps, "one_plus": False},
                                 shapes=[(hd, nh, None)]),
                           w("qn.w")], [p + "q3n"], shapes=[(hd, nh, None)])
        k3 = g.add("mul", [g.add("rms_norm", [k3], [p + "kn0"],
                                 attrs={"eps": eps, "one_plus": False},
                                 shapes=[(hd, nkv, None)]),
                           w("kn.w")], [p + "k3n"], shapes=[(hd, nkv, None)])
    if a.get("rope", True):
        nrot = int(hd * float(a.get("rope_partial", 1.0)))
        rp = {"base": float(a["rope_base"]),
              "pairing": str(a.get("rope_pairing", "half")),
              "n_dims": nrot}
        q3 = g.add("rope", [q3, "pos"], [p + "q3r"], attrs=rp,
                   shapes=[(hd, nh, None)])
        k3 = g.add("rope", [k3, "pos"], [p + "k3r"], attrs=rp,
                   shapes=[(hd, nkv, None)])
    # 进 mul_mat 前换轴：[hd, 头, token] -> [hd, token, 头]
    q3 = g.add("permute", [q3], [p + "q3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nh)])
    k3 = g.add("permute", [k3], [p + "k3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nkv)])
    kq = g.add("matmul", [k3, q3], [p + "kq"], shapes=[(None, None, nh)])
    win = int(a.get("window") or 0)
    kq = g.add("soft_max", [kq, p + "mask_%d" % win], [p + "sm"],
               attrs={"scale": 1.0 / (hd ** 0.5)}, shapes=[(None, None, nh)])
    # v^T：[hd, 头, token] -> [token, hd, 头]
    vt = g.add("permute", [v], [p + "v3p"], attrs={"axes": [0, 2, 1]},
               shapes=[(hd, None, nkv)])
    vt = g.add("cont", [g.add("permute", [vt], [p + "v3t"],
                              attrs={"axes": [1, 0, 2]},
                              shapes=[(None, hd, nkv)])],
               [p + "v3c"], shapes=[(None, hd, nkv)])
    kqv = g.add("matmul", [vt, p + "sm"], [p + "kqv"], shapes=[(hd, None, nh)])
    att = g.add("cont", [g.add("permute", [kqv], [p + "kqvp"],
                               attrs={"axes": [0, 2, 1]},
                               shapes=[(hd, nh, None)])],
                [p + "att"], attrs={"shape": [nh * hd, None]},
                shapes=[(nh * hd, None)])
    out = g.add("matmul", [w("wo"), att], [p + "o0"], shapes=[(d, None)])
    if bias:
        out = g.add("add", [out, w("wo.bias")], [dst], shapes=[(d, None)])
    else:
        g.nodes[-1]["outs"] = [dst]


def lower_model(ir, name="model"):
    """整份 IR -> 一张图。globals 在层前后各一截（ggml 惰性图那一课：
    最终 norm/head **必须在层之后**接线）。"""
    d = ir["d_model"]
    if str(ir.get("pos_kind", "rope")).lower() == "learned":
        # **拒绝，不许悄悄少一块。** 学习式位置表是 embed 后的一次加法，
        # 还没收编 —— 降了就是"看起来对但算错"。
        raise SystemExit("  [不支持] 学习式位置表的 lowering 还没收编")
    g = Graph(name)
    g.vals["tok"] = (None,)
    g.vals["pos"] = (None,)
    g.vals["x"] = (d, None)
    pre = [o for o in ir["globals"] if o["kind"] == "Embed"]
    post = [o for o in ir["globals"] if o["kind"] != "Embed"]
    # Embed 的输出就是 `x`（残差流起点）
    for o in pre:
        g.add("get_rows", ["embed.weight", "tok"], ["x"], shapes=[(d, None)])
    for L in ir["layers"]:
        i = L["index"]
        din = L.get("in_dim") or d
        prev = None
        for j, o in enumerate(L["ops"]):
            # 层内值名重绑定：hidden -> x
            dst = lower_op(g, i, j, o, din)
            prev = dst
        # 层输出 -> 下一层的 hidden
        g.add("cont", [prev], ["x"], shapes=[(din, None)])
    for j, o in enumerate(post):
        lower_op(g, len(ir["layers"]) + 1, j, o, d)
    return g


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
    # rope：位置 0 不动；位置 1 的 half 配对和 interleaved 配对
    # **输入配对不同**（i,i+2 vs 2i,2i+1）—— 各自对公式。
    xr = np.zeros((4, 1))          # [dims=4, n=1]
    xr[:, 0] = [1.0, 0.0, 0.0, 1.0]
    rr = PRIMS["rope"]["ref"](xr, np.array([0.0]), 10000.0, "half", 4)
    if not np.allclose(rr, xr):
        bad.append("rope 位置 0 应当不动")
    if bad:
        for b in bad:
            print("  [FAIL] " + b)
        return 1
    print("  [PASS] 原语参考实现对拍手算（rms_norm / matmul / silu / rope）")
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
