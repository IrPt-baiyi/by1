#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 ggml -- 第四个后端：从 IR 生成 **调 ggml 原语的 C 代码**，编译，
跑起来，和 NumPy 后端对拍。

这是 `docs/llamacpp-plan.md` 的 M1+M2。和 `by1c.py`（第三个后端，手写循环）
的区别只剩一件事：**数学是 ggml 算的** —— 于是 ggml 成为第二个独立实现，
"两个独立实现同时错在同一个地方的概率比一个低"。

三条纪律，全部继承自 by1c，一条不松：

  ① **要么实现，要么拒绝。** 支持集以外的算子/属性直接 `[不支持]`，
     绝不悄悄生成一个算错的模型（`check_support`）。
  ② **w.bin 布局只有一份定义。** 直接调 `by1c.emit_c` 拿 `order` ——
     同一个 `w.bin` 能喂 by1c 和本后端。张量绑定一致不是方便，是判据。
  ③ **判卷人是 by1exec（NumPy）**，它已经和 PyTorch / HF 对过。

用法:
  python by1ggml.py models/llama-shaped.by1 --gcc <gcc.exe> [--seq 16]
  python by1ggml.py --selftest          # 全套：数值对拍 + 真实模型编译
  python by1ggml.py models/gemma-4-31B.by1 --no-run   # 只生成 + 编译

ggml 本体的构建（可重下，见 docs/llamacpp-plan.md M0）:
  cmake -S llamacpp/llama.cpp -B llamacpp/build-ggml -G Ninja \\
        -DCMAKE_BUILD_TYPE=Release -DGGML_STATIC=ON \\
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF
  cmake --build llamacpp/build-ggml --target ggml ggml-base ggml-cpu
"""

import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np
import by1io
import by1paths

MANGLE = __import__("re").compile(r"[^0-9A-Za-z]")

try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:
    _DEFAULT_EPS = 1e-5


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _qk_on(v):
    """qk_norm 三态：off / per_head / full。"off" 是**真值字符串** ——
    直接布尔判断会永远成立（by1c 同名函数就是为此存在）。"""
    if v is None or v is False:
        return False
    return str(v).strip().lower() not in ("off", "", "none", "false", "0")


# ── 支持集：要么实现，要么拒绝 ──────────────────────────────────────

SUPPORTED_KINDS = ("Embed", "Norm", "Head", "Attention", "FFN", "Add")
SUPPORTED_ACT = ("silu", "gelu", "gelu_new", "gelu_tanh", "tanh")


def check_support(ir):
    """**不支持就拒绝，绝不悄悄生成算错的 C。** 返回问题列表（空 = 支持）。"""
    bad = []
    if str(ir.get("pos_kind", "rope")).lower() not in ("rope", "learned"):
        bad.append(f"pos_kind={ir.get('pos_kind')!r} 不在支持集里")
    if str(ir.get("norm_kind", "rms")).lower() not in ("rms", "layer"):
        bad.append(f"norm_kind={ir.get('norm_kind')!r} 不在支持集里")
    for L in ir["layers"]:
        for j, o in enumerate(L["ops"]):
            k, a = o["kind"], o["attrs"]
            where = f"第 {L['index']} 层 op{j}（{k}）"
            if k not in SUPPORTED_KINDS:
                bad.append(f"{where}：算子 {k} 还没有 ggml 映射")
                continue
            if k == "Attention":
                if a.get("q_gate"):
                    bad.append(f"{where}：q_gate 还没实现")
                if a.get("head_gate", "off") != "off":
                    bad.append(f"{where}：head_gate 还没实现")
                if a.get("sink"):
                    bad.append(f"{where}：sink 还没实现（M3）")
                if a.get("qkv_fused"):
                    bad.append(f"{where}：qkv_fused（融合 q/k/v）还没实现")
                _y = a.get("yarn")
                if _y:
                    bad.append(f"{where}：yarn={_y!r} 还没实现（M3）")
                if float(a.get("rope_scale", 1.0)) != 1.0:
                    bad.append(f"{where}：rope_scale={a.get('rope_scale')} "
                               "≠ 1.0 还没实现（M3）")
                if _qk_on(a.get("qk_norm")) and \
                        str(a.get("qk_norm")).lower() == "full":
                    bad.append(f"{where}：qk_norm=full 还没实现（per_head 支持）")
                if int(a["out_dim"]) != int(a["q"]) * int(a["head_dim"]):
                    bad.append(f"{where}：out_dim={a['out_dim']} ≠ "
                               f"q*head_dim={int(a['q'])*int(a['head_dim'])}"
                               "（输出投影前的重排还没实现）")
            elif k == "FFN":
                if str(a.get("act", "silu")) not in SUPPORTED_ACT:
                    bad.append(f"{where}：act={a.get('act')!r} 还没实现")
                if a.get("limit"):
                    bad.append(f"{where}：limit={a.get('limit')} "
                               "（SwiGLU 夹取）还没实现")
    for o in ir["globals"]:
        if o["kind"] not in SUPPORTED_KINDS:
            bad.append(f"全局算子 {o['kind']} 还没有 ggml 映射")
    return bad


# ── C 源生成 ────────────────────────────────────────────────────────

GGML_HEAD = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-cpu.h"

static float *W;                 /* 所有权重，扁平存放（w.bin 原样载入） */
#define P(nm) (W + OFF_##nm)

/* ggml 的张量是**列主序**：by1 的 [n, d] 行主序缓冲正好是 ggml 的 [d, n]，
   所以 w.bin 里 `linear()` 的那套布局（dout 行 × din 列行主序）直接就是
   ggml_mul_mat 要的 [din, dout] —— **零转换**。这就是"同一个 w.bin 喂
   两个后端"能成立的原因。 */
static struct ggml_tensor * wt1(struct ggml_context * ctx, const float * data,
                                int64_t n, const char * nm) {
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, nm);
    /* **这版 ggml 没有 ggml_set_data** —— 结构体是公开的，直接挂指针。
       数据在 w.bin 那块 malloc 里，ctx 的那份配额照旧占着（够用就行）。 */
    t->data = (void *) data;
    return t;
}
static struct ggml_tensor * wt2(struct ggml_context * ctx, const float * data,
                                int64_t ne0, int64_t ne1, const char * nm) {
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, nm);
    t->data = (void *) data;
    return t;
}
'''


def _f(x):
    """float 字面量。eps 这类值必须原样进 C —— 写死 1e-5 的 bug 仓库里出过六次。"""
    return repr(float(x)) + "f"


def _key(i, j, nm):
    return f"L{i}_OP{j}_{MANGLE.sub('_', nm).upper()}"


def _var(ref):
    return "cur" if ref == "hidden" else "v_" + ref.replace(".", "_")


def emit_model_c(ir, order, params):
    """把 IR 展开成一份调 ggml 的 model.c。返回 (源码, N_NODES, MEM_PER_TOKEN,
    MEM_PER_PAIR)。**不支持的东西在 check_support 里就拦掉了** ——
    这里只管已经承诺的那部分，一格都不许糊。"""
    d_model = ir["d_model"]
    vocab = ir["vocab"]
    learned_pos = str(ir.get("pos_kind", "rope")).lower() == "learned"

    # ── 常量与 OFF_/LEN_ 表（布局 = by1c 的 order，一字不差） ────────
    head = [f"#define D_MODEL {d_model}", f"#define VOCAB {vocab}"]
    if learned_pos:
        head.append("#define N_POS %d" % int(ir.get("n_pos") or ir.get("ctx") or 1024))
    off = 0
    for macro, pname in order:
        n = int(np.prod(params[pname]))
        head.append(f"#define OFF_{macro} {off}")
        head.append(f"#define LEN_{macro} {n}")
        off += n
    head.append(f"#define TOTAL_W {off}")

    # 节点数与内存预算的估计（n 是运行期才知道的，所以预算分两段算）
    n_nodes, mem_tok, mem_pair = 64, 4096, 0
    for L in ir["layers"]:
        for o in L["ops"]:
            k, a = o["kind"], o["attrs"]
            if k == "Attention":
                hd, nh, nkv = a["head_dim"], a["q"], a["kv"]
                qw = nh * hd
                n_nodes += 16
                mem_tok += 4 * 4 * (qw + 2 * nkv * hd + nh * hd + 256)
                mem_pair += 4 * 4 * nh * 3
            elif k == "FFN":
                n_nodes += 10
                mem_tok += 4 * 4 * (3 * a["hidden"] + 256)
            elif k == "Norm":
                n_nodes += 4
                mem_tok += 4 * 4 * 256
            else:
                n_nodes += 3
                mem_tok += 4 * 4 * 256
    for o in ir["globals"]:
        n_nodes += 3
    mem_tok += 4 * 4 * (2 * vocab + 4 * d_model)   # embed/head/最终 norm
    head.append(f"#define N_NODES {n_nodes}")
    head.append(f"#define MEM_PER_TOKEN {mem_tok}ULL")
    head.append(f"#define MEM_PER_PAIR {mem_pair}ULL")

    # ── 权重张量（一次性建好，层里引用） ─────────────────────────────
    body = ["", "    /* ---- 权重张量（数据指针指向 w.bin，不拷贝） ---- */"]
    for macro, pname in order:
        shp = params[pname]
        nm = pname.replace('"', "")
        if len(shp) == 1:
            body.append(f"    struct ggml_tensor * t_{macro.lower()} = "
                        f"wt1(ctx, P({macro}), {int(shp[0])}, \"{nm}\");")
        elif len(shp) == 2:
            body.append(f"    struct ggml_tensor * t_{macro.lower()} = "
                        f"wt2(ctx, P({macro}), {int(shp[1])}, {int(shp[0])}, "
                        f"\"{nm}\");")
        else:
            raise SystemExit(f"  [不支持] {pname} 是 {len(shp)} 维张量 —— "
                             "只实现了 1D/2D")

    # ── 注意力掩码：按 window 取值去重，一值一张 ─────────────────────
    wins = sorted({int((o["attrs"].get("window") or 0))
                   for L in ir["layers"] for o in L["ops"]
                   if o["kind"] == "Attention"})
    mask_var = {}
    if wins:
        body.append("")
        body.append("    /* ---- 因果 + 滑窗掩码（host 侧填好，图里当叶子用） ---- */")
        body.append("    /* by1 的规则：j <= i 且 j > i - window（window 个 key） */")
    for w in wins:
        v = f"msk_{w}"
        mask_var[w] = v
        body.append(f"    float * {v}_buf = (float *) malloc(sizeof(float) * (size_t) n * n);")
        body.append(f"    for (int i = 0; i < n; i++)")
        body.append(f"        for (int j = 0; j < n; j++) {{")
        body.append(f"            int ok = (j <= i) && ({w} == 0 || j > i - {w});")
        body.append(f"            {v}_buf[j + (size_t) i * n] = ok ? 0.0f : -INFINITY;")
        body.append(f"        }}")
        body.append(f"    struct ggml_tensor * {v} = ggml_new_tensor_2d(ctx, "
                    "GGML_TYPE_F32, n, n);")
        body.append(f"    {v}->data = {v}_buf;")

    # ── 逐算子发射 ──────────────────────────────────────────────────
    def emit_op(lines, i, j, o, indent="        ", wmap=None):
        """wmap：短参数名 -> order 宏名。层里的算子走 `_key(i, j, nm)`；
        全局算子的宏是 EMBED / FINAL_NORM / HEAD / POS —— **和层里的
        L{i}_OP{j}_ 命名是两套**，混用会引用不存在的变量（第一版就这么错的）。"""
        k, a = o["kind"], o["attrs"]
        src = [_var(r) for r in o["inputs"]]
        dst = _var(o["outputs"][0])
        decl = "" if dst == "cur" else "struct ggml_tensor * "
        # 中间量带算子序号前缀 —— 一层里有两个 Norm（op0/op3），
        # 都叫 `_r` 会重定义（第一版就是这么编译失败的）。
        p = f"_o{j}_"

        def W_(nm):
            return f"t_{(wmap[nm] if wmap else _key(i, j, nm)).lower()}"

        if k == "Norm":
            one = 1 if a.get("one_plus") else 0
            fn = "ggml_norm" if str(a.get("kind", "rms")).lower() == "layer" \
                else "ggml_rms_norm"
            lines.append(f"{indent}struct ggml_tensor * {p}r = {fn}(ctx, {src[0]}, "
                         f"{_f(a.get('eps', _DEFAULT_EPS))});")
            if one:
                lines.append(f"{indent}{decl}{dst} = "
                             f"ggml_add(ctx, ggml_mul(ctx, {p}r, {W_('w')}), {p}r);")
            else:
                lines.append(f"{indent}{decl}{dst} = "
                             f"ggml_mul(ctx, {p}r, {W_('w')});")
            if str(a.get("kind", "rms")).lower() == "layer":
                lines.append(f"{indent}{dst} = ggml_add(ctx, {dst}, {W_('b')});")
        elif k == "Attention":
            hd, nh, nkv = int(a["head_dim"]), int(a["q"]), int(a["kv"])
            qw = nh * hd
            eps = _f(a.get("norm_eps", _DEFAULT_EPS))
            one = 1 if a.get("norm_one_plus") else 0
            bias = bool(a.get("bias"))
            lines.append(f"{indent}struct ggml_tensor * {p}aq = ggml_mul_mat(ctx, "
                         f"{W_('wq')}, {src[0]});")
            lines.append(f"{indent}struct ggml_tensor * {p}ak = ggml_mul_mat(ctx, "
                         f"{W_('wk')}, {src[0]});")
            if bias:
                lines.append(f"{indent}{p}aq = ggml_add(ctx, {p}aq, {W_('wq.bias')});")
                lines.append(f"{indent}{p}ak = ggml_add(ctx, {p}ak, {W_('wk.bias')});")
            # **kv_tie：v 是 k 的原始投影** —— 必须在 qk_norm / rope 之前分叉
            if a.get("kv_tie"):
                lines.append(f"{indent}struct ggml_tensor * {p}v3 = "
                             f"ggml_reshape_3d(ctx, {p}ak, {hd}, {nkv}, n);")
            else:
                lines.append(f"{indent}struct ggml_tensor * {p}av = ggml_mul_mat(ctx, "
                             f"{W_('wv')}, {src[0]});")
                if bias:
                    lines.append(f"{indent}{p}av = ggml_add(ctx, {p}av, {W_('wv.bias')});")
                lines.append(f"{indent}struct ggml_tensor * {p}v3 = "
                             f"ggml_reshape_3d(ctx, {p}av, {hd}, {nkv}, n);")
            lines.append(f"{indent}struct ggml_tensor * {p}q3 = "
                         f"ggml_reshape_3d(ctx, {p}aq, {hd}, {nh}, n);")
            lines.append(f"{indent}struct ggml_tensor * {p}k3 = "
                         f"ggml_reshape_3d(ctx, {p}ak, {hd}, {nkv}, n);")
            if _qk_on(a.get("qk_norm")):
                # per_head：ggml_rms_norm 沿 dim0 归一 —— [hd, nh, n] 上
                # 正好是"每个 (头, 位置) 归一 hd"，和 by1exec 的逐头 rms 一致。
                lines.append(f"{indent}{p}q3 = ggml_mul(ctx, ggml_rms_norm(ctx, {p}q3, "
                             f"{eps}), {W_('qn.w')});")
                lines.append(f"{indent}{p}k3 = ggml_mul(ctx, ggml_rms_norm(ctx, {p}k3, "
                             f"{eps}), {W_('kn.w')});")
            if a.get("rope", True):
                nrot = int(hd * float(a.get("rope_partial", 1.0)))
                # pairing: half -> NEOX（半进半出，逐位对应）；
                #          interleaved -> NORMAL（奇偶进、奇偶出）。
                # by1 的 interleaved 是"奇偶进、**拼接**着出" —— 输出布局
                # 和 NORMAL 差一个 q/k **共同**置换，点积不变。
                # **这条不许当假设**，由 --selftest 的孪生模型数值证明。
                mode = 2 if str(a.get("rope_pairing", "half")) == "half" else 0
                rope = (f"ggml_rope_ext(ctx, %s, pos, NULL, {nrot}, {mode}, 0, "
                        f"{_f(a['rope_base'])}, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)")
                lines.append(f"{indent}{p}q3 = {rope % (p + 'q3')};")
                lines.append(f"{indent}{p}k3 = {rope % (p + 'k3')};")
            w = int(a.get("window") or 0)
            # **进 mul_mat 前要 permute(0,2,1,3)** —— llama.cpp 的
            # build_attn 幂等式：把 [hd, 头, token] 换成 [hd, token, 头]。
            # 不换的话 KQ 出来是 [头, 头, token]，掩码当场 GGML_ASSERT
            # （第一版就是这么炸的）。
            lines.append(f"{indent}{p}q3 = ggml_permute(ctx, {p}q3, 0, 2, 1, 3);")
            lines.append(f"{indent}{p}k3 = ggml_permute(ctx, {p}k3, 0, 2, 1, 3);")
            lines.append(f"{indent}struct ggml_tensor * {p}kq = "
                         f"ggml_mul_mat(ctx, {p}k3, {p}q3);")
            # scale = 1/sqrt(hd)：和 by1exec 的 `att / np.sqrt(hd)` 一致，
            # 由 soft_max_ext 的融合 scale 提供。
            lines.append(f"{indent}{p}kq = ggml_soft_max_ext(ctx, {p}kq, {mask_var[w]}, "
                         f"1.0f / sqrtf((float) {hd}), 0.0f);")
            # v^T：v3 [hd, 头, token] -> permute -> [hd, token, 头] ->
            # transpose -> [token, hd, 头] -> cont（mul_mat 要的左矩阵）
            lines.append(f"{indent}struct ggml_tensor * {p}v3t = "
                         f"ggml_cont(ctx, ggml_transpose(ctx, "
                         f"ggml_permute(ctx, {p}v3, 0, 2, 1, 3)));")
            lines.append(f"{indent}struct ggml_tensor * {p}kqv = "
                         f"ggml_mul_mat(ctx, {p}v3t, {p}kq);")
            # 合头：[hd, n, nh] -> [nh*hd, n]（permute 后 cont_2d 重排连续）
            lines.append(f"{indent}struct ggml_tensor * {p}att = ggml_cont_2d(ctx, "
                         f"ggml_permute(ctx, {p}kqv, 0, 2, 1, 3), {qw}, n);")
            lines.append(f"{indent}struct ggml_tensor * {dst} = ggml_mul_mat(ctx, "
                         f"{W_('wo')}, {p}att);")
            if bias:
                lines.append(f"{indent}{dst} = ggml_add(ctx, {dst}, {W_('wo.bias')});")
        elif k == "FFN":
            hid = int(a["hidden"])
            bias = bool(a.get("bias"))
            act = str(a.get("act", "silu"))
            # **激活映射：三个 gelu 同名不同义，而且 ggml 的 GELU 不能直接用。**
            #   by1 的 gelu_new（tanh 近似） ↔ ggml_gelu      —— **但它是
            #       fp16 查表**（vec.h 无条件 `#define GGML_GELU_FP16`），
            #       实测和公式差 1.3e-4，把 1e-4 的判据顶爆。用原语拼公式。
            #   by1 的 gelu（erf 精确）      ↔ ggml_gelu_erf  —— erff，精确 ✓
            #   ggml_gelu_quick = x*σ(1.702x) —— 那是 gptoss，**不是任何
            #       一个 gelu**（第一版把 gelu_new 映到这里，差 4.041e-03）。
            def _act(vv):
                if act == "gelu":
                    lines.append(f"{indent}{vv} = ggml_gelu_erf(ctx, {vv});")
                elif act in ("gelu_new", "gelu_tanh", "tanh"):
                    # 0.5·x·(1+tanh(c·(x+a·x³))) = 0.5·(x + x·tanh(c·(x+a·x³)))
                    lines.append(f"{indent}struct ggml_tensor * {vv}_t = "
                                 f"ggml_tanh(ctx, ggml_scale(ctx, "
                                 f"ggml_add(ctx, {vv}, ggml_scale(ctx, "
                                 f"ggml_mul(ctx, ggml_mul(ctx, {vv}, {vv}), {vv}), "
                                 f"{_f(0.044715)})), {_f(0.7978845608028654)}));")
                    lines.append(f"{indent}{vv} = ggml_scale(ctx, ggml_add(ctx, "
                                 f"{vv}, ggml_mul(ctx, {vv}, {vv}_t)), 0.5f);")
                else:
                    lines.append(f"{indent}{vv} = ggml_silu(ctx, {vv});")
            lines.append(f"{indent}struct ggml_tensor * {p}hb = ggml_mul_mat(ctx, "
                         f"{W_('w1')}, {src[0]});")
            if bias:
                lines.append(f"{indent}{p}hb = ggml_add(ctx, {p}hb, {W_('w1.bias')});")
            if a.get("gate", True):
                lines.append(f"{indent}struct ggml_tensor * {p}ob = ggml_mul_mat(ctx, "
                             f"{W_('w3')}, {src[0]});")
                if bias:
                    lines.append(f"{indent}{p}ob = ggml_add(ctx, {p}ob, {W_('w3.bias')});")
                _act(p + "hb")
                lines.append(f"{indent}{p}hb = ggml_mul(ctx, {p}hb, {p}ob);")
            else:
                _act(p + "hb")
            lines.append(f"{indent}struct ggml_tensor * {dst} = ggml_mul_mat(ctx, "
                         f"{W_('w2')}, {p}hb);")
            if bias:
                lines.append(f"{indent}{dst} = ggml_add(ctx, {dst}, {W_('w2.bias')});")
        elif k == "Add":
            lines.append(f"{indent}{decl}{dst} = "
                         f"ggml_add(ctx, {src[0]}, {src[1]});")
        elif k == "Embed":
            lines.append(f"{indent}{decl}{dst} = "
                         f"ggml_get_rows(ctx, {W_('embed.weight')}, tok);")
        elif k == "Head":
            lines.append(f"{indent}struct ggml_tensor * {dst} = ggml_mul_mat(ctx, "
                         f"{W_('head.weight')}, {src[0]});")

    body.append("")
    body.append("    /* ---- embed（+ 学习式位置表）：**在层之前** ---- */")
    # **全局算子要拆成两截。** ggml 是惰性图 —— 张量在**构建那一刻**
    # 就把输入钉死了。原来 embed / 最终 norm / head 全排在层之前发射，
    # 于是 norm/head 吃到的是 **embed 的输出**，层算了等于没算
    # （实测：输出和"零层模型"逐位相关 0.99999，而判卷人报 FAIL）。
    # by1exec 的语义是 embed -> 逐层 -> 最终 norm -> head，照它排。
    gmap = {
        "Embed": {"embed.weight": "EMBED"},
        "Norm": {"w": "FINAL_NORM", "b": "FINAL_NORM_B"},
        "Head": {"head.weight": "HEAD"},
    }
    gpre = [(j, o) for j, o in enumerate(ir["globals"]) if o["kind"] == "Embed"]
    gpost = [(j, o) for j, o in enumerate(ir["globals"]) if o["kind"] != "Embed"]
    for j, o in gpre:
        emit_op(body, 0, j, o, indent="    ", wmap=gmap[o["kind"]])
        if learned_pos:
            body.append("    struct ggml_tensor * _posrow = "
                        "ggml_get_rows(ctx, t_pos, pos);")
            body.append("    cur = ggml_add(ctx, cur, _posrow);")
    logits = _var(ir["globals"][-1]["outputs"][0])

    for L in ir["layers"]:
        i = L["index"]
        body.append("")
        body.append(f"    /* ---- layer {i} ---- */")
        body.append("    {")
        for j, o in enumerate(L["ops"]):
            emit_op(body, i, j, o)
        last = _var(L["ops"][-1]["outputs"][0])
        body.append(f"        cur = {last};")
        body.append("    }")

    if gpost:
        body.append("")
        body.append("    /* ---- 最终 norm + head：**在层之后**（吃的是层跑完的残差）---- */")
        for j, o in gpost:
            emit_op(body, 0, j, o, indent="    ", wmap=gmap[o["kind"]])

    main = f"""
int main(int argc, char **argv) {{
    if (argc < 5) {{
        fprintf(stderr, "usage: model n w.bin ids.bin logits.bin\\n");
        return 2;
    }}
    const int n = atoi(argv[1]);
    if (n <= 0) {{ fprintf(stderr, "bad n\\n"); return 2; }}
    FILE * f = fopen(argv[2], "rb");
    if (!f) {{ fprintf(stderr, "cannot open w.bin\\n"); return 2; }}
    W = (float *) malloc(sizeof(float) * (size_t) TOTAL_W);
    if (fread(W, sizeof(float), (size_t) TOTAL_W, f) != (size_t) TOTAL_W) {{
        fprintf(stderr, "w.bin short read\\n"); return 2;
    }}
    fclose(f);
    int32_t * ids = (int32_t *) malloc(sizeof(int32_t) * (size_t) n);
    f = fopen(argv[3], "rb");
    if (!f) {{ fprintf(stderr, "cannot open ids.bin\\n"); return 2; }}
    if (fread(ids, sizeof(int32_t), (size_t) n, f) != (size_t) n) {{
        fprintf(stderr, "ids.bin short read\\n"); return 2;
    }}
    fclose(f);

    /* 内存预算：权重 + 每 token 的中间量 + n*n 的注意力矩阵 + 固定余量。
       估小了会在 ggml_new_tensor 里断言 —— 宁可多给。 */
    size_t mem = (size_t) TOTAL_W * sizeof(float)
               + (size_t) n * MEM_PER_TOKEN
               + (size_t) n * (size_t) n * MEM_PER_PAIR
               + 96ULL * 1024 * 1024;
    struct ggml_init_params ip = {{ mem, NULL, false }};
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {{ fprintf(stderr, "ggml_init failed (%zu bytes)\\n", mem); return 1; }}

    struct ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    memcpy(tok->data, ids, sizeof(int32_t) * (size_t) n);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    for (int i = 0; i < n; i++) ((int32_t *) pos->data)[i] = i;
    struct ggml_tensor * cur;
{chr(10).join(body)}

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, N_NODES, false);
    ggml_build_forward_expand(gf, {logits});
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {{
        fprintf(stderr, "graph compute failed\\n"); return 1;
    }}
    f = fopen(argv[4], "wb");
    fwrite({logits}->data, sizeof(float), (size_t) n * VOCAB, f);
    fclose(f);
    printf("ok %d x %d\\n", n, VOCAB);
    return 0;
}}
"""
    stamp = ""
    try:
        from by1ver import stamp as _stamp
        stamp = "/* %s */" % _stamp()
    except ImportError:
        # **不能 `except: pass`** —— by1lint 的十类静默写法之一：
        # 错误被吃掉，调用方看到"一切正常"，而 stamp 根本没打上。
        stamp = "/* by1ggml */"
    return stamp + GGML_HEAD + "\n".join(head) + main, \
        n_nodes, mem_tok, mem_pair


# ── 孪生模型：gemma-4-31B 的机制组合，小维度 ─────────────────────────
#
# 为什么需要它：真实 gemma-4-31B 的随机权重是 **121.7 GB f32**
# （vocab 262144 · 87 层），数值对拍跑不动。孪生**只缩维度**，
# 机制组合逐项相同：qk_norm=per_head · pairing=interleaved · 滑窗 +
# 全量两种注意力 · kv_tie · partial rope · eps=1e-6 · silu 门控 FFN。
TWIN_BY1 = '''model GemmaShaped {
  ctx 512

  hparams {
    d_model = 32
    n_layer = 6
    vocab   = 64
    d_ff    = 64
    rms_eps = 1e-06
  }

  mech GQA : Attention {
    heads      = { q = 4, kv = 2, head_dim = 8 }
    out_dim    = 32
    qk_norm    = true
    kv_tie     = false
    attn_bias  = false
    structural = [ kv, head_dim, out_dim, kv_tie ]
  }

  mech Dense : FFN { hidden = 64 }

  schedule {
    local  = 5 * GQA(window = 4, kv = 2, head_dim = 8, out_dim = 32,
                     kv_tie = false)
    global = 1 * GQA(window = none, kv = 1, head_dim = 8, out_dim = 32,
                     kv_tie = true)
    block  = local + global
  }

  stack main {
    pattern = 1 * block
    main[:] >> Dense
  }

  position {
    local  = rope(base = 10000, type = default)
    global = partial(rope(base = 1000000, type = proportional), ratio = 0.25)
  }
}
'''


# ── 驱动：生成 -> 编译 -> 跑 -> 和 NumPy 对拍 ─────────────────────────

def ggml_dirs(args):
    src_dir = args.ggml_src or by1paths.root("llamacpp", "llama.cpp", "ggml")
    if not os.path.isdir(src_dir):
        # **ggml 也可以来自独立仓**（`llamacpp/ggml`）。2026-10-12 实测：
        # 本机到 GitHub 只有 ~42 KB/s，llama.cpp 的 tarball 要 20–30 分钟，
        # 而 **ggml 早已独立成仓**（4.1 MB，一分钟拿到）。判卷人不变 ——
        # `tools/ggml-smoke.c` 判的还是"ggml 在这台机器上算的是这个数学"。
        alt = by1paths.root("llamacpp", "ggml")
        if os.path.isdir(alt):
            src_dir = alt
    build = args.ggml_build or by1paths.root("llamacpp", "build-ggml")
    return src_dir, build


def find_libs(build):
    want = ["ggml-cpu", "ggml", "ggml-base"]
    libs = []
    for name in want:
        hit = None
        for root, _dirs, files in os.walk(build):
            for fn in files:
                if fn.startswith(name) and fn.endswith((".a", ".lib")) \
                        and "llama" not in fn:
                    hit = os.path.join(root, fn)
                    break
            if hit:
                break
        if not hit:
            return None
        libs.append(hit)
    return libs


def run_one(by1_path, args, run=True, expect="pass"):
    bc, cg = load("by1check"), load("by1codegen")
    ex_mod = load("by1exec")
    name = os.path.basename(by1_path)

    print(f"\n{'='*72}\n  {name}  ->  ggml  ({'数值对拍' if run else '仅编译'})\n{'='*72}")
    _r, info = bc.check(by1_path)
    ir = cg.compile_ir(info)
    params_shapes = ex_mod.shapes_of(ir)

    bad = check_support(ir)
    if bad:
        print("  [不支持] 这份 .by1 落在支持集外（要么实现，要么拒绝）：")
        for b in bad[:8]:
            print("    · " + b)
        if expect == "reject":
            print("\n  [PASS] 按承诺拒绝")
            return "reject-ok"
        return "fail"

    # **w.bin 布局的唯一定义在 `by1ir.weight_plan`**（1.0.0-rc0 · A3-1）。
    # 以前这里反向 import by1c 只为拿 `order`（下一行的 `del decls` 就是
    # 那个证据）—— 布局是 **IR 的事实**，不该由某个后端代管。
    _bir = load("by1ir")
    order = _bir.weight_plan(ir)
    total = sum(int(np.prod(s)) for s in _bir.params_of(ir).values())

    src, n_nodes, mem_tok, mem_pair = emit_model_c(ir, order, params_shapes)
    if expect == "reject":
        print("\n  [FAIL] 期望拒绝，结果支持集放行了")
        return "fail"

    workdir = args.workdir or ("cgen-ggml-" + os.path.splitext(name)[0])
    os.makedirs(workdir, exist_ok=True)
    cpath = os.path.join(workdir, "model.c")
    by1io.write_text(cpath, src)
    print(f"  生成 {cpath}  （{len(src.splitlines())} 行，"
          f"{len(order)} 个张量 {total*4/1024:.1f} KiB，图预算 {n_nodes} 节点）")

    if run:
        # **和 by1c 的 harness 一字不差**：同一个种子、同一套扁平布局
        rng = np.random.default_rng(0)
        params = {k: rng.normal(0, 0.1, s).astype(np.float32)
                  for k, s in params_shapes.items()}
        flat = np.concatenate([params[p].ravel() for _m, p in order])
        by1io.write_bytes(os.path.join(workdir, "w.bin"), flat.tobytes())
        ids = rng.integers(0, ir["vocab"], args.seq).astype(np.int32)
        by1io.write_bytes(os.path.join(workdir, "ids.bin"), ids.tobytes())

    src_dir, build = ggml_dirs(args)
    libs = find_libs(build)
    if not libs:
        print("  [跳过] 找不到 ggml 静态库。先构建（见本文件头注 / "
              "docs/llamacpp-plan.md M0）：")
        print(f"    cmake -S {os.path.join(by1paths.root(), 'llamacpp', 'llama.cpp')} "
              f"-B {build} -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_STATIC=ON")
        return "skip"

    exe = os.path.abspath(os.path.join(workdir, "model.exe"))
    cmd = [args.gcc, "-O2", "-fopenmp", "-o", exe, cpath,
           "-I", os.path.join(src_dir, "include")] + libs + \
          ["-lm", "-lstdc++", "-lpthread"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print("  [编译失败]")
        print((p.stderr or "")[:1600])
        return "fail"
    print("  [编译成功]")

    if not run:
        print("\n  [PASS] 真实 IR 生成 + 编译（--no-run，不跑数值）")
        return "ok"

    r = subprocess.run([exe, str(args.seq),
                        os.path.join(workdir, "w.bin"),
                        os.path.join(workdir, "ids.bin"),
                        os.path.join(workdir, "logits.bin")],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  [运行失败] {r.stdout} {r.stderr}")
        return "fail"
    print(f"  [运行成功] {r.stdout.strip()}")

    got = np.fromfile(os.path.join(workdir, "logits.bin"), dtype=np.float32)
    got = got.reshape(args.seq, ir["vocab"])
    ex = ex_mod.Exec(ir, params)
    ex.P["_g"] = {k: params[k] for k in
                  ("embed.weight", "pos.weight", "final_norm.w",
                   "final_norm.b", "head.weight") if k in params}
    want = ex(ids[None, :])[0]
    dd = np.abs(got - want).max()
    amp = max(np.abs(want).max(), 1e-9)
    ok = dd / amp < 1e-4
    print(f"\n  ggml 后端 vs NumPy 后端")
    print(f"    幅度 ggml {np.abs(got).max():.4e}   NumPy {np.abs(want).max():.4e}")
    print(f"    最大绝对差 {dd:.3e}   相对 {dd/amp:.3e}   "
          + ("[PASS]" if ok else "[FAIL]"))
    return "ok" if ok else "fail"


def selftest(args):
    """M2 的验收判据，一条命令跑完：
       ① llama-shaped     基础四算子 · pairing=half          数值对拍
       ② gpt2-tiny        layernorm / 学习位置 / gelu_new / 无门控 数值对拍
       ③ gemma 孪生        qk_norm+interleaved+window+kv_tie  数值对拍
       ④ gemma-4-31B       真实 IR（87 层，含视觉塔）          仅编译
       ⑤ qwen3-next-shaped  GDN/MoE 落在支持集外               必须拒绝
    """
    root = by1paths.root()
    twin_path = os.path.join(root, "_gemma-shaped.by1")
    by1io.write_text(twin_path, TWIN_BY1)

    cases = [
        (os.path.join(root, "models", "llama-shaped.by1"), True, "pass"),
        (os.path.join(root, "models", "gpt2-tiny.by1"), True, "pass"),
        (twin_path, True, "pass"),
        (os.path.join(root, "models", "gemma-4-31B.by1"), False, "pass"),
        (os.path.join(root, "models", "qwen3-next-shaped.by1"), False, "reject"),
    ]
    results = []
    for path, run, expect in cases:
        try:
            results.append((os.path.basename(path), run_one(path, args, run, expect)))
        except SystemExit as e:
            print(str(e))
            results.append((os.path.basename(path), "fail"))

    print(f"\n{'='*72}\n  by1ggml 自检汇总\n{'='*72}")
    ok, any_fail, any_skip = True, False, False
    for nm, res in results:
        good = res in ("ok", "reject-ok")
        ok = ok and good
        any_fail = any_fail or res == "fail"
        any_skip = any_skip or res == "skip"
        print(f"    {nm:<28} {'✓' if good else '✗'}  ({res})")
    if ok:
        print("\n  [PASS] ggml 后端：数值对拍 + 真实 IR 编译 + 拒绝纪律，全绿")
        return 0
    if any_skip and not any_fail:
        # **ggml 本体没构建 = 这台机器上没验**，不是失败（五态里的 30）。
        print("\n  [跳过] ggml 静态库不在 —— 这一段没验")
        return 30
    print("\n  [FAIL] 有失败项，见上")
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 ggml 后端")
    ap.add_argument("by1", nargs="?")
    ap.add_argument("--gcc", default="gcc")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--no-run", action="store_true",
                    help="只生成 + 编译（真实大模型用这个）")
    ap.add_argument("--expect-reject", action="store_true",
                    help="这份 .by1 应当被支持集拒绝")
    ap.add_argument("--ggml-src", default=None)
    ap.add_argument("--ggml-build", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args)
    if not args.by1:
        ap.error("要么给一份 .by1，要么 --selftest")
    expect = "reject" if args.expect_reject else "pass"
    res = run_one(args.by1, args, run=not args.no_run, expect=expect)
    return {"ok": 0, "reject-ok": 0, "skip": 30}.get(res, 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
