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
     绝不悄悄生成一个算错的模型 —— 门在**lowering 与发射表**（A3-3 之后
     没有单独的 `check_support` 白名单了）。
  ② **w.bin 布局只有一份定义**（`by1ir.weight_plan`）—— 同一个 `w.bin`
     能喂 by1c / by1exec 和本后端。张量绑定一致不是方便，是判据。
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
import re
import subprocess
import sys

import numpy as np
import by1io
import by1paths
from by1ir import ROPE_SCALE_DEFAULT, as_bool  # noqa: E402

MANGLE = __import__("re").compile(r"[^0-9A-Za-z]")

def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

static struct ggml_tensor * wt3(struct ggml_context * ctx, const float * data,
                                int64_t ne0, int64_t ne1, int64_t ne2,
                                const char * nm) {
    struct ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, nm);
    t->data = (void *) data;
    return t;
}

/* ── T16 第二批：轴约定族的运行时 helper ─────────────────────────────
   逐节点发射在编译期不知道动态轴（seq 那一维是 None），所以这几个
   走**运行时元数据**（照 wt1 的先例：结构体是公开的）。轴都是 ggml
   视角（PRIMS 规格原文：「attr 的 axis 按 ggml 视角」）。 */

/* 沿 ggml 轴 axis 整块重复 reps 次。pat 只借形状，不借数据。 */
static struct ggml_tensor * rep_ax(struct ggml_context * ctx,
                                   struct ggml_tensor * x, int axis, int reps) {
    int64_t ne[GGML_MAX_DIMS];
    int nd = ggml_n_dims(x);
    for (int i = 0; i < nd; i++) ne[i] = x->ne[i];
    ne[axis] = ne[axis] * reps;
    struct ggml_tensor * pat = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
    return ggml_repeat(ctx, x, pat);
}

/* 沿 ggml 轴 axis 取 [start, stop)。**用真视图算子** ggml_view_4d ——
   结构体诡计（view_tensor + 改 data/ne）只对叶子张量灵，对 op 输出
   不灵（data 不指回源，实测读出零/垃圾）。 */
static struct ggml_tensor * sl_ax(struct ggml_context * ctx,
                                  struct ggml_tensor * x, int axis,
                                  int64_t start, int64_t stop) {
    int64_t ne[4];
    for (int i = 0; i < 4; i++) ne[i] = (i < ggml_n_dims(x)) ? x->ne[i] : 1;
    size_t off = x->nb[axis] * (size_t) start;
    ne[axis] = stop - start;
    struct ggml_tensor * v = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], off);
    return ggml_cont(ctx, v);
}

/* 沿 ggml 轴 axis 求和。keepdims=0 时挤掉该轴的 1（走 reshape 算子）。 */
static struct ggml_tensor * sum_ax(struct ggml_context * ctx,
                                   struct ggml_tensor * x, int axis,
                                   int keepdims) {
    int nd = ggml_n_dims(x);
    struct ggml_tensor * r;
    if (axis == 0) {
        r = ggml_sum_rows(ctx, x);         /* [1, ne1, …] */
    } else {
        struct ggml_tensor * t = ggml_cont(ctx, ggml_transpose(ctx, x));
        r = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, t)));
    }
    if (!keepdims) {
        int64_t ne[4];
        int m = 0;
        for (int i = 0; i < nd; i++) {
            if (i != axis) ne[m++] = r->ne[i];
        }
        for (int i = m; i < 4; i++) ne[i] = 1;
        r = ggml_reshape_4d(ctx, r, ne[0], ne[1], ne[2], ne[3]);
    }
    return r;
}

/* 沿 ggml 轴 axis 把每一份**整块**重复 reps 次（np.repeat 语义：
   [q0,q0,q1,q1]）。**不是 ggml_repeat 的平铺**（平铺给 [q0,q1,q0,q1]）。

   做法：在 axis **前面**插一条长度 1 的轴 -> 平铺到 reps（长度 1 的平铺
   就是"整份复制"）-> 把这两条轴折回一条。插入点在 axis 之前是关键：
   折回时内层是 reps、外层是原轴，得到 (i_axis*reps + r)，正是 np.repeat。
   （插在 axis 之后会得到 (r*ne_axis + i_axis) = 平铺 —— 实测踩过两次。）

   历史：这里原有一条 `axis == 0 -> 整块平铺` 的捷径，理由是"最外轴两者
   相同"—— **只在 exec 序最外轴成立**，而 ggml 的 axis 0 是 exec 的**最后**
   一轴。k-pool 的池->token 展开（axis 0、ne=池数>1）第一次踩到它，
   结果是 [p0..pN,p0..pN] 而不是 [p0,p0,p1,p1..]，keep 掩码直接错。 */
static struct ggml_tensor * blk_rep_axis(struct ggml_context * ctx,
                                         struct ggml_tensor * x, int axis,
                                         int reps) {
    if (reps <= 1) return x;
    const int nd = ggml_n_dims(x);
    if (axis < 0 || axis >= nd) return NULL;
    if (nd >= 4) {
        /* 插一个维就超 4 维了：退到逐片切片 + 拼接（慢但语义等价）。 */
        struct ggml_tensor * acc = NULL;
        for (int64_t k = 0; k < x->ne[axis]; k++) {
            struct ggml_tensor * sl = sl_ax(ctx, x, axis, k, k + 1);
            for (int r = 0; r < reps; r++) {
                acc = acc ? ggml_concat(ctx, acc, sl, axis) : sl;
            }
        }
        return acc;
    }
    int64_t ins[4] = {1, 1, 1, 1};
    int m = 0;
    for (int i = 0; i < nd; i++) {
        if (i == axis) ins[m++] = 1;
        ins[m++] = x->ne[i];
    }
    struct ggml_tensor * v = ggml_reshape_4d(ctx, ggml_cont(ctx, x),
                                             ins[0], ins[1], ins[2], ins[3]);
    int64_t pat[4] = {ins[0], ins[1], ins[2], ins[3]};
    const int64_t na = pat[axis + 1];
    pat[axis] = reps;
    struct ggml_tensor * pt = ggml_new_tensor(ctx, x->type, 4, pat);
    struct ggml_tensor * r = ggml_repeat(ctx, v, pt);
    int64_t out[4] = {1, 1, 1, 1};
    int kk = 0;
    for (int i = 0; i < 4; i++) {
        if (i == axis) { out[kk++] = na * reps; continue; }
        if (i == axis + 1) continue;
        out[kk++] = pat[i];
    }
    return ggml_reshape_4d(ctx, r, out[0], out[1], out[2], out[3]);
}

/* numpy 风格的**双向广播**二元运算。ggml 的原生二元算子只允许第二个
   参数被广播（第一个参数必须是结果形状）—— 而图的 ref 是 numpy 的
   `a op b`（两边都可能小）。于是：
     · 加/乘（可交换）：谁大谁当第一个；
     · 减/除：小的那条先 repeat 到大的形状（repeat 是整块平铺，广播语义
       下小的那条在这个轴上只有 1 份，所以平铺 == 广播）。
   第一版直接 `ggml_mul(ctx, ea, tt)`（ea 是 [1, nv] 的衰减系数）——
   KDA 的 `gate_lower` 支当场撞在 ggml_can_repeat 上。 */
enum { BY1_BIN_ADD = 0, BY1_BIN_SUB = 1, BY1_BIN_MUL = 2, BY1_BIN_DIV = 3 };
static struct ggml_tensor * by1_bin_bcast(struct ggml_context * ctx,
                                          struct ggml_tensor * a,
                                          struct ggml_tensor * b, int op) {
    if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) {
        if (op == BY1_BIN_ADD || op == BY1_BIN_MUL) {
            struct ggml_tensor * t = a; a = b; b = t;
        } else {
            a = ggml_repeat(ctx, a, b);
        }
    }
    switch (op) {
        case BY1_BIN_ADD: return ggml_add(ctx, a, b);
        case BY1_BIN_SUB: return ggml_sub(ctx, a, b);
        case BY1_BIN_MUL: return ggml_mul(ctx, a, b);
        default:          return ggml_div(ctx, a, b);
    }
}

/* 深度可分离因果卷积，**F32 im2col**。
   ggml_conv_1d_dw 的 im2col 输出类型是 F16（除非核是 BF16）—— 它把输入
   先round 到 fp16，实测 conv 输出相对差 ~1e-4，经过几层放大后把
   1e-4 的判据顶爆（qwen3-next 5.5e-4 / ling-tiny 2e-4）。
   这里照 conv_1d_dw 的三步自己走一遍，只把 dst_type 换成 F32：
     im2col(kernel, data, s0=1, s1=0, p0=k-1, p1=0, d0=d1=1, is_2D=false, F32)
     -> mul_mat(im2col, kernel) -> reshape_3d [ne0, ne2, 1]
   （p1=0 就是"只在左边补 kernel-1"的因果约定。） */
static struct ggml_tensor * by1_conv1d_dw_f32(struct ggml_context * ctx,
                                              struct ggml_tensor * w,
                                              struct ggml_tensor * x,
                                              int pad) {
    struct ggml_tensor * nb = ggml_reshape_4d(ctx, x, x->ne[0], 1,
                                              x->ne[1], x->ne[2]);
    struct ggml_tensor * im = ggml_im2col(ctx, w, nb, 1, 0, pad, 0, 1, 0,
                                          false, GGML_TYPE_F32);
    struct ggml_tensor * r = ggml_mul_mat(ctx, im, w);
    return ggml_reshape_3d(ctx, r, r->ne[0], r->ne[2], 1);
}

/* le：out = (a <= b) ? 1 : 0（PRIMS 原文；**1/0 不是 bool** —— 它要能
   进 masked_fill 的 `m > 0`，也要能乘）。ggml 这版没有比较算子，走
   map_custom2；两边**先都 repeat 到广播后的形状**（map_custom2 的结果
   形状取第一个参数，而 DSA 的 le 是对 [n,1] 与 [1,n] 做外积，两边都不是、
   也都不等于结果形状）。 */
static void by1_le_f32(struct ggml_tensor * dst, const struct ggml_tensor * a,
                       const struct ggml_tensor * b, int ith, int nth,
                       void * userdata) {
    (void) nth; (void) userdata;
    if (ith != 0) return;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++)
     for (int64_t i2 = 0; i2 < dst->ne[2]; i2++)
      for (int64_t i1 = 0; i1 < dst->ne[1]; i1++)
       for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
           const float va = *(const float *) ((const char *) a->data
               + i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
           const float vb = *(const float *) ((const char *) b->data
               + i0 * b->nb[0] + i1 * b->nb[1] + i2 * b->nb[2] + i3 * b->nb[3]);
           *(float *) ((char *) dst->data + i0 * dst->nb[0] + i1 * dst->nb[1]
               + i2 * dst->nb[2] + i3 * dst->nb[3]) = (va <= vb) ? 1.0f : 0.0f;
       }
}

/* ssm_scan 的**逆 softplus**：图的 ssm_scan 原语（ref）把 softplus 放在
   原语**外面**（输入 d = softplus(dt + dt_bias)），而 ggml_ssm_scan 在内部又做
   一次 softplus（ops.cpp: softplus(dt[h])）—— 直接映射会算两次。
   逆函数 x = log(expm1(d)) 在 float32 下**绝对误差 ~1 ulp**（log 把 expm1 的
   相对误差转成绝对误差），d = 0 -> -inf 也对（内核 softplus(-inf) = 0）；
   d > 80 时 expm1 会溢出，直接取 x = d（softplus 在 x>20 已是恒等）。
   这不是"猜"，是**契约折算**：往返 softplus(log(expm1(d))) = d 逐值可测。 */
static void by1_inv_softplus_f32(struct ggml_tensor * dst,
                                 const struct ggml_tensor * a,
                                 int ith, int nth, void * userdata) {
    (void) nth; (void) userdata;
    if (ith != 0) return;
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(a));
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; i++) {
        const float d = ((const float *) a->data)[i];
        if (!(d >= 0.0f)) {
            GGML_ABORT("by1: ssm_scan 的 dt 应当是 softplus 的输出（>= 0）");
        }
        ((float *) dst->data)[i] = (d > 80.0f) ? d : logf(expm1f(d));
    }
}
/* ── masked_fill 的逐元算子（out = m > 0 ? x : v）─────────────────────
   ggml 没有 select；用算术拼（x*m + v*(1-m)）在 v = -inf 时出 nan
   （-inf·0），而 MoE 的分组路由正是要把非选中组填 -inf —— 所以这条走
   map_custom2，语义**逐字**照 PRIMS（单线程：n_tasks=1）。 */
static void by1_masked_fill_f32(struct ggml_tensor * dst,
                                const struct ggml_tensor * a,
                                const struct ggml_tensor * b,
                                int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const float v = *(const float *) userdata;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++)
     for (int64_t i2 = 0; i2 < dst->ne[2]; i2++)
      for (int64_t i1 = 0; i1 < dst->ne[1]; i1++)
       for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
           const float * pa = (const float *) ((const char *) a->data
               + (i0 % a->ne[0]) * a->nb[0] + (i1 % a->ne[1]) * a->nb[1]
               + (i2 % a->ne[2]) * a->nb[2] + (i3 % a->ne[3]) * a->nb[3]);
           const float * pb = (const float *) ((const char *) b->data
               + (i0 % b->ne[0]) * b->nb[0] + (i1 % b->ne[1]) * b->nb[1]
               + (i2 % b->ne[2]) * b->nb[2] + (i3 % b->ne[3]) * b->nb[3]);
           *(float *) ((char *) dst->data + i0 * dst->nb[0] + i1 * dst->nb[1]
               + i2 * dst->nb[2] + i3 * dst->nb[3]) = (*pb > 0.0f) ? *pa : v;
       }
}
'''


def _f(x):
    """float 字面量。eps 这类值必须原样进 C —— 写死 1e-5 的 bug 仓库里出过六次。"""
    return repr(float(x)) + "f"


def _budget(ir):
    """N_NODES / MEM_PER_TOKEN / MEM_PER_PAIR 的估计。

    **这是 harness 的预算，不是发射逻辑** —— 所以 IR 路径和图路径共用
    同一份，对拍才是同条件（预算只影响 ggml 的 arena 大小，估小了会在
    `ggml_new_tensor` 里断言）。"""
    d_model, vocab = ir["d_model"], ir["vocab"]
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
    for _o in ir["globals"]:
        n_nodes += 3
    mem_tok += 4 * 4 * (2 * vocab + 4 * d_model)
    return n_nodes, mem_tok, mem_pair


def _head_lines(ir, order, params):
    """常量 + `OFF_`/`LEN_` 表（布局来自 `by1ir.weight_plan`）。"""
    d_model, vocab = ir["d_model"], ir["vocab"]
    head = [f"#define D_MODEL {d_model}", f"#define VOCAB {vocab}"]
    if str(ir.get("pos_kind", "rope")).lower() == "learned":
        head.append("#define N_POS %d"
                    % int(ir.get("n_pos") or ir.get("ctx") or 1024))
    off = 0
    for macro, pname in order:
        n = int(np.prod(params[pname]))
        head.append(f"#define OFF_{macro} {off}")
        head.append(f"#define LEN_{macro} {n}")
        off += n
    head.append(f"#define TOTAL_W {off}")
    n_nodes, mem_tok, mem_pair = _budget(ir)
    head.append(f"#define N_NODES {n_nodes}")
    head.append(f"#define MEM_PER_TOKEN {mem_tok}ULL")
    head.append(f"#define MEM_PER_PAIR {mem_pair}ULL")
    return head


def _weight_lines(order, params):
    """权重张量：数据指针指向 w.bin，不拷贝。"""
    out = ["", "    /* ---- 权重张量（数据指针指向 w.bin，不拷贝） ---- */"]
    for macro, pname in order:
        shp = params[pname]
        nm = pname.replace('"', "")
        if len(shp) == 1:
            out.append(f"    struct ggml_tensor * t_{macro.lower()} = "
                       f"wt1(ctx, P({macro}), {int(shp[0])}, \"{nm}\");")
        elif len(shp) == 2:
            out.append(f"    struct ggml_tensor * t_{macro.lower()} = "
                       f"wt2(ctx, P({macro}), {int(shp[1])}, {int(shp[0])}, "
                       f"\"{nm}\");")
        elif len(shp) == 3:
            # 3 维（MoE 专家权重 (E, out, in)）：by1 的 [E][out][in] 行主序
            # 就是 ggml 的 [in, out, E] —— 和前两维同一个"零转换"约定。
            out.append(f"    struct ggml_tensor * t_{macro.lower()} = "
                       f"wt3(ctx, P({macro}), {int(shp[2])}, {int(shp[1])}, "
                       f"{int(shp[0])}, \"{nm}\");")
        else:
            raise SystemExit(f"  [不支持] {pname} 是 {len(shp)} 维张量 —— "
                             "只实现了 1D/2D")
    return out


def _rope_call(p, indent, x_tmpl, pos_expr, nrot, mode, base_c, base_f, yarn):
    """rope 的三支（普通 / YaRN / llama3）**统一成逐对维 `freq_factors`**。

    ggml 的角 = `pos · base^(-2i/n) / ff[i]`（ops.cpp：`theta/ff`），
    by1 的角 = `pos · inv[i]`（**唯一公式源 `by1graph.rope_inv`**，
    三支都在里面）—— 所以 `ff[i] = base^(-2i/n) / inv[i]`。

    普通支 `ff ≡ 1`（不传 c）；YaRN/llama3 的斜坡**已经折进 `inv` 里**，
    所以 ggml 侧不需要它自己的 yarn ramp（`ext_factor = 0`）。
    这样"一条规则三处实现"的老毛病不会再犯：公式只在 `rope_inv`，
    这里只做**刻度换算**。

    返回 (要先发射的行, 带 `%s` 的调用模板)。
    """
    lines = []
    tail = (f"{pos_expr}, %s, {nrot}, {mode}, 0, "
            f"{base_c}, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)")
    if not yarn:
        return lines, "ggml_rope_ext(ctx, %s, " + tail % 'NULL'
    import numpy as _np
    import by1graph as _bg
    half = nrot // 2
    inv = _np.asarray(_bg.rope_inv(nrot, base_f, yarn), dtype=_np.float64)
    ff = (base_f ** (-2.0 * _np.arange(half) / nrot)) / inv
    lines.append(f"{indent}float * {p}ff_buf = (float *) malloc(sizeof(float) * {half});")
    for _i, _v in enumerate(ff):
        lines.append(f"{indent}{p}ff_buf[{_i}] = {_f(float(_v))};")
    lines.append(f"{indent}struct ggml_tensor * {p}ff = ggml_new_tensor_1d("
                 f"ctx, GGML_TYPE_F32, {half});")
    lines.append(f"{indent}{p}ff->data = {p}ff_buf;")
    return lines, "ggml_rope_ext(ctx, %s, " + tail % f'{p}ff'


def _mask_lines(wins):
    """因果 + 滑窗掩码（host 侧填好，图里当叶子用）。
    by1 的规则：j <= i 且 j > i - window（window 个 key）。"""
    out, mask_var = [], {}
    if wins:
        out.append("")
        out.append("    /* ---- 因果 + 滑窗掩码（host 侧填好，图里当叶子用） ---- */")
        out.append("    /* by1 的规则：j <= i 且 j > i - window（window 个 key） */")
    for w in wins:
        v = f"msk_{w}"
        mask_var[w] = v
        out.append(f"    float * {v}_buf = (float *) malloc(sizeof(float) * "
                   "(size_t) n * n);")
        out.append("    for (int i = 0; i < n; i++)")
        out.append("        for (int j = 0; j < n; j++) {")
        out.append(f"            int ok = (j <= i) && ({w} == 0 || j > i - {w});")
        out.append(f"            {v}_buf[j + (size_t) i * n] = ok ? 0.0f : -INFINITY;")
        out.append("        }")
        out.append(f"    struct ggml_tensor * {v} = ggml_new_tensor_2d(ctx, "
                   "GGML_TYPE_F32, n, n);")
        out.append(f"    {v}->data = {v}_buf;")
    return out, mask_var


# ── 从图发射（A3-2） ────────────────────────────────────────────────

_GNAME = re.compile(r"[^0-9A-Za-z_]")


def _gvar(name):
    """图里的值名 -> C 变量名。"""
    return "v_" + _GNAME.sub("_", name)


def _shp(shape, n_expr="n"):
    """图里的 ggml 视角形状 -> C 表达式（None 就是运行期的 n）。"""
    return [n_expr if s is None else str(int(s)) for s in shape]


def _shp_inferred(shape, src):
    """形状里的 None = **推断维**（`ggml_nelements(src) / 静态维之积`）。

    图语言里 None 的语义一直是 numpy 的 `-1`（`by1graph._exec_view` 就是这么
    实现、且只许一个 None）。旧版这里把它写死成 `n` —— 对绝大多数视图恰好
    相等（被推断的那一维就是序列长），但 k-pool 要 `view(pos, [P, n/P])`，
    写死就错了。改成真的按元素数推：对旧图**逐位不变**（推断出来仍是 n）。
    """
    k = sum(1 for t in shape if t is None)
    if k != 1:
        return _shp(shape)
    prod = 1
    for t in shape:
        if t is not None:
            prod *= int(t)
    return [f"(ggml_nelements({src}) / {prod})" if t is None else str(int(t))
            for t in shape]


def emit_model_c_graph(g, ir, order, params, dump=()):
    """**从图发射**：逐节点把 PRIMS 翻成 ggml 调用（1.0.0-rc0 · A3-2）。

    **A3-3（2026-10-17）之后这是唯一的一条发射路**：按 IR kind 手写的旧
    发射器（`emit_model_c`）已删。图里没有"机制"分支 —— Attention / FFN /
    MoE / KDA 早被 lowering 拆成了 matmul / rope / soft_max / top_k 这样的
    原语，所以这里只有一张 **PRIMS -> ggml** 的机械表，没有一处
    "这一层是什么" 的判断。

    **能力判断只在 lowering 那一层**：图降不动就点名拒绝（`by1graph`
    的 `[不支持]`），走到这里的图只含上表覆盖到的原语；遇到表外的
    **当场报错**，不许悄悄糊。
    """
    pmap = {pname: "t_" + macro.lower() for macro, pname in order}
    head = _head_lines(ir, order, params)
    n_nodes, mem_tok, mem_pair = _budget(ir)
    body = _weight_lines(order, params)

    # 掩码：从**节点**里认（`soft_max` 的第二个输入叫 mask_<window>）
    # **soft_max_sink 也要收**（第一版只扫 soft_max，gpt-oss 的 sink 那条
    # 引用到一个从来没发射过的 msk_0 —— 编译当场报 undeclared）。
    wins = sorted({int(n["ins"][1].rsplit("_", 1)[-1]) for n in g.nodes
                   if n["prim"] in ("soft_max", "soft_max_sink")
                   and len(n["ins"]) > 1 and "mask_" in n["ins"][1]})
    _mlines, mask_var = _mask_lines(wins)
    body += _mlines
    # 标量组合族（log1p / softplus）要的 1 元素常量张量；wt1 挂指针，
    # 值活过计算期（static）。
    body.append("    static float _one_v = 1.0f;")
    body.append("    struct ggml_tensor * t_one = wt1(ctx, &_one_v, 1, \"t_one\");")

    declared = set()

    def _ref(name):
        if name in pmap:
            return pmap[name]
        if name == "tok" or name == "pos":
            return name
        # 掩码叶子的名字是**带前缀**的（`L0O1_mask_0`）—— 第一版只认
        # `mask_*` 开头，于是 soft_max_sink 的 a[1] 变成未定义变量。
        if "mask_" in name:
            return mask_var[int(name.rsplit("_", 1)[-1])]
        return _gvar(name)

    def _out(name):
        v = _gvar(name)
        if v in declared:
            return v, ""                     # 第二次产出同名值：只赋值
        declared.add(v)
        return v, "struct ggml_tensor * "

    def _put(lines, prim, node, expr, ind="    "):
        dst, decl = _out(node["outs"][0])
        lines.append(f"{ind}{decl}{dst} = {expr};")
        return dst

    def _reshape(fn, x, shape):
        dims = _shp(shape)
        return f"{fn}(ctx, {x}, " + ", ".join(dims) + ")"

    body.append("")
    body.append("    /* ---- 逐节点发射（PRIMS -> ggml） ---- */")
    # **非连续值的账本**：permute 的产物是视图，ggml_reshape 要求连续 ——
    # 图里 `_b4` 那种 "permute 再 view" 的组合会在 reshape 的 assert 上炸
    # （qwen3-next 实测）。view 的 ref 是 np.reshape（逻辑序，必要时自己拷），
    # 所以这里对非连续输入先 cont 一次，语义逐位相同。
    noncont = set()
    # **I32 的账本**：pos / tok / top_k 的输出及其视图是整数张量。它们可以
    # 当 ids 用（get_rows/one_hot 内部会 cast），但**不能直接进 F32 算术** ——
    # ggml 的二元算子不许混类型（k-pool 的池起点/尾池掩码第一次撞上）。
    ivals = {"pos", "tok"}

    def _f32(expr, valname=None):
        """整数张量进浮点上下文前先 cast。"""
        if valname is not None and valname in ivals:
            return f"ggml_cast(ctx, {expr}, GGML_TYPE_F32)"
        return expr

    def _cont_if_needed(expr, valname=None):
        """非连续的值先物化（ggml 的 soft_max 之类的断言要求连续）。"""
        if valname is None:
            return expr
        u = _gvar(valname)
        body.append(f"    struct ggml_tensor * {u}_cc = ggml_cont(ctx, {expr});")
        return f"{u}_cc"
    for node in g.nodes:
        prim, ins, at = node["prim"], node["ins"], node["attrs"]
        a = [_ref(x) for x in ins]
        if prim == "get_rows":
            # numpy 的 `table[ids]` = **沿表的第一条 exec 轴（= ggml 的最后一条）**
            # 取整块。两条路：
            #   ① ids 是 1 维（embedding 的 tok）-> ggml_get_rows（原语）；
            #   ② ids 是多维（MoE 的 (K, n)）-> 用乘法写成加权和
            #      out[io, m] = Σ_e table[io, e] · EQ[e, m]：
            #      ggml_get_rows 的契约是 idx 只在 dim1 上取、dim2/3 整块搬
            #      （src0->ne2 == src1->ne1），铺在 (K, n) 上的下标它接不住
            #      —— 实测 assert 在 ne2 上就炸。
            tab, ids_ = a[0], a[1]
            # 表的形状：权重在 params 里，中间值在 g.vals 里 —— 两处都查。
            shp_t = params.get(node["ins"][0]) or g.vals.get(node["ins"][0]) or ()
            shp_i = g.vals.get(node["ins"][1]) or ()
            if len(shp_i) <= 1:
                _put(body, prim, node,
                     f"ggml_get_rows(ctx, {tab}, {ids_})")
            else:
                if len(shp_t) not in (2, 3):
                    raise SystemExit("  [不支持] get_rows 的表是 %d 维" % len(shp_t))
                u = _gvar(node["outs"][0])
                # **维数只在运行期取**：权重（`params`）是 exec 序、图值
                # （`g.vals`）是 ggml 序 —— 静态推会推错轴。取整块的轴 =
                # 表的**最后一条 ggml 轴**，行宽 = 其余维之积。
                body.append(f"    int64_t {u}_nd = ggml_n_dims({tab});")
                body.append(f"    int64_t {u}_e = {tab}->ne[{u}_nd - 1];")
                body.append(f"    int64_t {u}_io = ggml_nelements({tab}) / {u}_e;")
                body.append(f"    int64_t {u}_m = ggml_nelements({ids_});")
                body.append(f"    struct ggml_tensor * {u}_t = ggml_reshape_2d("
                            f"ctx, {tab}, {u}_io, {u}_e);")
                body.append(f"    struct ggml_tensor * {u}_a = ggml_cont(ctx, "
                            f"ggml_transpose(ctx, {u}_t));")
                body.append(f"    int64_t {u}_p[2] = {{ {u}_e, {u}_m }};")
                body.append(f"    struct ggml_tensor * {u}_pat = ggml_new_tensor("
                            f"ctx, GGML_TYPE_F32, 2, {u}_p);")
                body.append(f"    struct ggml_tensor * {u}_A = ggml_repeat(ctx, "
                            f"ggml_reshape_2d(ctx, ggml_arange(ctx, 0.0f, "
                            f"(float) {u}_e, 1.0f), {u}_e, 1), {u}_pat);")
                body.append(f"    struct ggml_tensor * {u}_B = ggml_repeat(ctx, "
                            f"ggml_reshape_2d(ctx, ggml_cast(ctx, "
                            f"ggml_cont(ctx, {ids_}), GGML_TYPE_F32), "
                            f"1, {u}_m), {u}_pat);")
                body.append(f"    struct ggml_tensor * {u}_EQ = ggml_clamp(ctx, "
                            f"ggml_add1(ctx, ggml_neg(ctx, ggml_sqrt(ctx, "
                            f"ggml_sqr(ctx, ggml_sub(ctx, {u}_A, {u}_B)))), "
                            f"t_one), 0.0f, 1.0f);")
                body.append(f"    struct ggml_tensor * {u}_r = ggml_mul_mat(ctx, "
                            f"{u}_a, {u}_EQ);")
                # 输出形状 = 表的**其余维**（ggml 序）+ 下标自身的维（ggml 序）
                body.append(f"    int64_t {u}_on[4] = {{ 1, 1, 1, 1 }};")
                body.append(f"    for (int64_t _i = 0; _i < {u}_nd - 1; _i++) "
                            f"{u}_on[_i] = {tab}->ne[_i];")
                body.append(f"    for (int64_t _i = 0; _i < ggml_n_dims({ids_}); _i++) "
                            f"{u}_on[{u}_nd - 1 + _i] = {ids_}->ne[_i];")
                _put(body, prim, node,
                     f"ggml_reshape_4d(ctx, {u}_r, {u}_on[0], {u}_on[1], "
                     f"{u}_on[2], {u}_on[3])")
        elif prim in ("add", "mul"):
            # **双向广播**：ggml 只许第二个参数被广播（见 by1_bin_bcast）。
            _op = "BY1_BIN_ADD" if prim == "add" else "BY1_BIN_MUL"
            _put(body, prim, node,
                 f"by1_bin_bcast(ctx, {_f32(a[0], ins[0])}, {_f32(a[1], ins[1])}, {_op})")
        elif prim == "scale":
            _put(body, prim, node,
                 f"ggml_scale(ctx, {_f32(a[0], ins[0])}, {_f(at['s'])})")
        elif prim in ("matmul", "batch_matmul"):
            # ggml_mul_mat(W, x) —— 图里的 matmul 也是 (权重, 输入)。
            #
            # **批维方向不一致时要先补**：numpy 的广播是双向的，而
            # ggml_mul_mat 只允许"第一个参数的批维整除第二个"（a 被摊到 b 上）。
            # MoE 的专家乘是 wir 反的那一支：权重带批维 (K, n)、输入是 (1, n)
            # —— 实测 ggml_can_mul_mat 在 ne2 上就炸。这里按**静态形状**
            # 把小的那条 rep_ax 到大的（运行期 n 两条相等，不用碰）。
            u = _gvar(node["outs"][0])

            def _shp_of(nm):
                v = params.get(nm)
                if v is None:
                    v = g.vals.get(nm)
                return v or ()

            def _dim(sh, k):
                return int(sh[k]) if len(sh) > k and sh[k] is not None else None

            xa, xb = a[0], a[1]
            # **mul_mat 不吃 permuted 的操作数**（ops.cpp 的注释与断言：
            # nb10 必须是元素大小）。图里的 permute 是视图 —— 要 positional
            # 语义就先物化。（DSA 的 qp 就是这么撞上 ggml-cpu.c:1289 的。）
            if ins[0] in noncont:
                body.append(f"    struct ggml_tensor * {u}_ca = ggml_cont(ctx, {xa});")
                xa = f"{u}_ca"
            if ins[1] in noncont:
                body.append(f"    struct ggml_tensor * {u}_cb = ggml_cont(ctx, {xb});")
                xb = f"{u}_cb"
            sa, sb = _shp_of(ins[0]), _shp_of(ins[1])
            a2, a3 = _dim(sa, 2), _dim(sa, 3)
            b2, b3 = _dim(sb, 2), _dim(sb, 3)
            if a2 and b2 and a2 > b2 and a2 % b2 == 0:
                body.append(f"    struct ggml_tensor * {u}_b2 = rep_ax(ctx, "
                            f"{xb}, 2, {a2 // b2});")
                xb = f"{u}_b2"
            if a3 and b3 and a3 > b3 and a3 % b3 == 0:
                body.append(f"    struct ggml_tensor * {u}_b3 = rep_ax(ctx, "
                            f"{xb}, 3, {a3 // b3});")
                xb = f"{u}_b3"
            _put(body, prim, node, f"ggml_mul_mat(ctx, {xa}, {xb})")
        elif prim == "rms_norm":
            eps, one = _f(at["eps"]), as_bool(at.get("one_plus"))
            r = f"_r{len(body)}"
            body.append(f"    struct ggml_tensor * {r} = "
                        f"ggml_rms_norm(ctx, {a[0]}, {eps});")
            if one:
                _put(body, prim, node,
                     f"ggml_add(ctx, ggml_mul(ctx, {r}, {a[1]}), {r})")
            else:
                _put(body, prim, node, f"ggml_mul(ctx, {r}, {a[1]})")
        elif prim == "layer_norm":
            eps, one = _f(at["eps"]), as_bool(at.get("one_plus"))
            r = f"_r{len(body)}"
            body.append(f"    struct ggml_tensor * {r} = "
                        f"ggml_norm(ctx, {a[0]}, {eps});")
            if one:
                z = f"_z{len(body)}"
                body.append(f"    struct ggml_tensor * {z} = "
                            f"ggml_add(ctx, ggml_mul(ctx, {r}, {a[1]}), {r});")
            else:
                z = f"_z{len(body)}"
                body.append(f"    struct ggml_tensor * {z} = "
                            f"ggml_mul(ctx, {r}, {a[1]});")
            _put(body, prim, node, f"ggml_add(ctx, {z}, {a[2]})")
        elif prim == "silu":
            _put(body, prim, node, f"ggml_silu(ctx, {a[0]})")
        elif prim == "gelu":
            # by1 的 gelu = erf 精确版 ↔ ggml_gelu_erf（erff，精确）
            _put(body, prim, node, f"ggml_gelu_erf(ctx, {a[0]})")
        elif prim == "sigmoid":
            _put(body, prim, node, f"ggml_sigmoid(ctx, {a[0]})")
        elif prim == "gelu_tanh":
            # **ggml 的 GELU 不能直接用**：它是 fp16 查表（`vec.h` 无条件
            # `#define GGML_GELU_FP16`），实测和公式差 1.3e-4 —— 顶爆 1e-4
            # 的判据。所以按公式用原语拼（M2 记在案）。
            x = a[0]
            t = f"_g{len(body)}"
            body.append(
                f"    struct ggml_tensor * {t} = ggml_tanh(ctx, ggml_scale(ctx, "
                f"ggml_add(ctx, {x}, ggml_scale(ctx, ggml_mul(ctx, "
                f"ggml_mul(ctx, {x}, {x}), {x}), {_f(0.044715)})), "
                f"{_f(0.7978845608028654)}));")
            _put(body, prim, node,
                 f"ggml_scale(ctx, ggml_add(ctx, {x}, ggml_mul(ctx, {x}, {t})), "
                 "0.5f)")
        elif prim == "rope":
            nrot = int(at["n_dims"])
            mode = 2 if str(at["pairing"]) == "half" else 0
            if float(at.get("rope_scale", ROPE_SCALE_DEFAULT)) != 1.0:
                # **这里原来是静默算错**：ref 是 `cos/sin · rope_scale`，
                # 而发射出去的是 mscale=1.0 —— rope_scale≠1 的模型会
                # "看起来对、算的是另一个模型"。要么实现（attn_factor -> mscale
                # 的折算还没数值验过），**要么拒绝**。
                raise SystemExit(
                    "  [不支持] 图路的 rope_scale（cos/sin 缩放）还没接 "
                    "mscale —— **拒绝，不猜**（折算未数值验过）")
            _rl, _tmpl = _rope_call(f"_g{len(body)}", "    ", "%s", a[1],
                                    nrot, mode, _f(float(at["base"])),
                                    float(at["base"]), at.get("yarn"))
            body += _rl
            _put(body, prim, node, _tmpl % a[0])
        elif prim == "soft_max":
            w = int(node["ins"][1].rsplit("_", 1)[-1])
            _put(body, prim, node,
                 f"ggml_soft_max_ext(ctx, {_cont_if_needed(a[0], ins[0])}, {mask_var[w]}, "
                 f"{_f(at['scale'])}, 0.0f)")
        elif prim == "view":
            src = a[0]
            if node["ins"][0] in noncont:
                u = _gvar(node["outs"][0])
                body.append(f"    struct ggml_tensor * {u}_c = ggml_cont(ctx, {src});")
                src = f"{u}_c"
            _put(body, prim, node,
                 "ggml_reshape_%dd(ctx, %s, " % (len(at["shape"]), src)
                 + ", ".join(_shp_inferred(at["shape"], src)) + ")")
            if node["ins"][0] in ivals:
                ivals.add(node["outs"][0])
        elif prim == "permute":
            ax = list(at["axes"]) + [3, 2, 1][:4 - len(at["axes"])]
            # **轴要取逆**：图的约定是「结果轴 j = 输入轴 axes[j]」
            # （PRIMS 的 ref 原文，by1c 的图路径就是这么做的），而
            # ggml_permute 的约定是「结果轴 axes[j] = 输入轴 j」——
            # 两者只在**对合**置换上相同（[0,2,1]/[1,0] 都是），所以
            # 此前一直没露；DSA 的 [2,0,1] 是三循环，实测轴位错了
            # （mul_mat 的收缩维 4 vs 16 当场炸）。
            inv = [0, 0, 0, 0]
            for _j, _a in enumerate(ax):
                inv[_a] = _j
            _put(body, prim, node,
                 f"ggml_permute(ctx, {a[0]}, {inv[0]}, {inv[1]}, "
                 f"{inv[2]}, {inv[3]})")
            noncont.add(node["outs"][0])
            if node["ins"][0] in ivals:
                ivals.add(node["outs"][0])
        elif prim == "cont":
            sh = at.get("shape")
            if sh:
                _put(body, prim, node,
                     "ggml_cont_%dd(ctx, %s, " % (len(sh), a[0])
                     + ", ".join(_shp_inferred(sh, a[0])) + ")")
            else:
                _put(body, prim, node, f"ggml_cont(ctx, {a[0]})")
            if node["ins"][0] in ivals:
                ivals.add(node["outs"][0])
        # ── T16 第一批补口（2026-10-14）：对 ggml.h 逐个摸底后接上的 ────
        elif prim in ("exp", "sqrt", "neg", "log"):
            _put(body, prim, node, f"ggml_{prim}(ctx, {a[0]})")
        elif prim == "div":
            _put(body, prim, node,
                 f"by1_bin_bcast(ctx, {_f32(a[0], ins[0])}, {_f32(a[1], ins[1])}, BY1_BIN_DIV)")
        elif prim == "clamp_min":
            # PRIMS 的 bound：`max(x, m)` —— 上界给满
            _put(body, prim, node,
                 f"ggml_clamp(ctx, {_f32(a[0], ins[0])}, {_f(at['min'])}, 1e38f)")
        elif prim == "clamp_max":
            _put(body, prim, node,
                 f"ggml_clamp(ctx, {_f32(a[0], ins[0])}, -1e38f, {_f(at['max'])})")
        elif prim == "add_scalar":
            # a + c（c 是 attr）—— **不是 clamp**（by1c 1999 的注释记着
            # `+1e-20` 和 `max(·, 1e-9)` 是两个公式）。c 发成 1 元素张量。
            sv = _gvar(node["outs"][0]) + "_c"
            body.append(f"    static float {sv} = {_f(at['c'])};")
            _put(body, prim, node,
                 f"ggml_add(ctx, {a[0]}, wt1(ctx, &{sv}, 1, \"{sv}\"))")
        elif prim in ("log1p", "softplus", "sqrtsoftplus"):
            # 标量组合族。softplus 用**数值稳定式**（PRIMS 的 f 原文）：
            #   softplus(x) = max(x,0) + log1p(e^{-|x|})，
            #   |x| = max(x,0) + max(-x,0)（ggml 无 abs，用 clamp+neg 拼）
            u = _gvar(node["outs"][0])
            if prim == "log1p":
                _put(body, prim, node,
                     f"ggml_log(ctx, ggml_add(ctx, {a[0]}, t_one))")
            else:
                body.append(f"    struct ggml_tensor * {u}_mx = "
                            f"ggml_clamp(ctx, {a[0]}, 0.f, 1e38f);")
                body.append(f"    struct ggml_tensor * {u}_ax = "
                            f"ggml_add(ctx, ggml_clamp(ctx, ggml_neg(ctx, {a[0]}), 0.f, 1e38f), {u}_mx);")
                sp = (f"ggml_add(ctx, {u}_mx, ggml_log(ctx, ggml_add(ctx, "
                      f"ggml_exp(ctx, ggml_neg(ctx, {u}_ax)), t_one)))")
                if prim == "softplus":
                    _put(body, prim, node, sp)
                else:
                    _put(body, prim, node, f"ggml_sqrt(ctx, {sp})")
        # ── T16 第二批：轴约定族（attr 的 axis 是 ggml 视角 —— PRIMS
        #    规格原文；而 exec 尾轴 = ggml dim0，两张图在软最大处对过）──
        elif prim == "softmax":
            # PRIMS「沿最后轴」= ggml dim0；ggml_soft_max 正是沿 dim0
            # 归一（自带 max-subtraction 稳定式）—— 直接映射。
            # **输入要连续**（ggml_soft_max 的断言）：k-pool 的池内 softmax
            # 吃的是 permute 出来的视图，第一版就撞在这条上。
            _put(body, prim, node, f"ggml_soft_max(ctx, {_cont_if_needed(a[0], ins[0])})")
        elif prim == "l2_norm":
            # x / sqrt(Σx² + eps)，**sum 不是 mean**（PRIMS 原文），
            # 沿最后轴 = ggml dim0 = sum_rows 的归约轴。
            u = _gvar(node["outs"][0])
            ev = u + "_e"
            body.append(f"    static float {ev} = {_f(at['eps'])};")
            body.append(f"    struct ggml_tensor * {u}_s = "
                        f"ggml_sum_rows(ctx, ggml_mul(ctx, {a[0]}, {a[0]}));")
            body.append(f"    struct ggml_tensor * {u}_d = ggml_sqrt(ctx, "
                        f"ggml_add(ctx, {u}_s, wt1(ctx, &{ev}, 1, \"{ev}\")));")
            _put(body, prim, node, f"ggml_div(ctx, {a[0]}, {u}_d)")
        elif prim == "sum":
            _put(body, prim, node,
                 f"sum_ax(ctx, {a[0]}, {int(at['axis'])}, "
                 f"{1 if at.get('keepdims', True) else 0})")
        elif prim == "repeat":
            # **np.repeat 的整块重复**，不是 ggml_repeat 的平铺（见 blk_rep_axis）。
            u = _gvar(node["outs"][0])
            body.append(f"    struct ggml_tensor * {u}_br = blk_rep_axis(ctx, "
                        f"{a[0]}, {int(at['axis'])}, {int(at['reps'])});")
            body.append(f"    if (!{u}_br) GGML_ABORT(\"blk_rep_axis: axis range\");")
            _put(body, prim, node, f"{u}_br")
            if node["ins"][0] in ivals:
                ivals.add(node["outs"][0])
        elif prim == "concat":
            _put(body, prim, node,
                 f"ggml_concat(ctx, {a[0]}, {a[1]}, {int(at['axis'])})")
        elif prim == "slice":
            _put(body, prim, node,
                 f"sl_ax(ctx, {a[0]}, {int(at['axis'])}, "
                 f"{int(at['start'])}, {int(at['stop'])})")
            if node["ins"][0] in ivals:
                ivals.add(node["outs"][0])
        elif prim == "soft_max_sink":
            # 语义逐字照 ref：z = a·scale + mask -> **尾巴拼 sink 槽** ->
            # softmax -> 去掉最后一列。exec 尾轴（key）= ggml dim0。
            # **顺序要紧**：先加 mask 再拼 sink —— sink 那列没有 mask。
            u = _gvar(node["outs"][0])
            x0 = _cont_if_needed(a[0], ins[0])
            body.append(f"    struct ggml_tensor * {u}_z = "
                        f"ggml_scale(ctx, {x0}, {_f(at['scale'])});")
            body.append(f"    {u}_z = ggml_add(ctx, {u}_z, {a[1]});")
            # sink [头] -> exec (头, q, 1) = ggml [1, n, 头]
            body.append(f"    struct ggml_tensor * {u}_sk0 = "
                        f"ggml_reshape_3d(ctx, {a[2]}, 1, 1, {x0}->ne[2]);")
            body.append(f"    int64_t {u}_ne[3] = {{1, {x0}->ne[1], {x0}->ne[2]}};")
            body.append(f"    struct ggml_tensor * {u}_pat = "
                        f"ggml_new_tensor(ctx, GGML_TYPE_F32, 3, {u}_ne);")
            body.append(f"    struct ggml_tensor * {u}_sk = "
                        f"ggml_repeat(ctx, {u}_sk0, {u}_pat);")
            body.append(f"    struct ggml_tensor * {u}_z2 = "
                        f"ggml_concat(ctx, {u}_z, {u}_sk, 0);")
            body.append(f"    struct ggml_tensor * {u}_sm = "
                        f"ggml_soft_max(ctx, {u}_z2);")
            body.append(f"    int64_t {u}_k = {x0}->ne[0];")
            _put(body, prim, node, f"sl_ax(ctx, {u}_sm, 0, 0, {u}_k)")
        elif prim == "conv1d_causal":
            # 深度可分离因果卷积（ref 的**不翻转**公式）：左补 kernel-1，
            # out[t] = Σ_i w[:,i] · xpad[:,t+i]。这版 ggml 的 conv_1d_dw
            # 是 im2col 布局：**参数序 (核, 输入)**、输入 [L, IC, N]、
            # 核 [K, 1, IC]（探针实证：非对称核 [1,10] 出 [10,21,32] =
            # 不翻转 ✓ 与 ref 同义）。图值是 exec 倒序布局，两头各转一次。
            u = _gvar(node["outs"][0])
            k = int(at["kernel"])
            body.append(f"    struct ggml_tensor * {u}_x = "
                        f"ggml_cont(ctx, ggml_permute(ctx, {a[0]}, 1, 0, 2, 3));")
            body.append(f"    struct ggml_tensor * {u}_w = "
                        f"ggml_reshape_3d(ctx, {a[1]}, {k}, 1, {a[0]}->ne[0]);")
            # **F32 im2col**（ggml_conv_1d_dw 会落到 F16，实测差 1e-4）
            body.append(f"    struct ggml_tensor * {u}_c = "
                        f"by1_conv1d_dw_f32(ctx, {u}_w, {u}_x, {k - 1});")
            body.append(f"    struct ggml_tensor * {u}_b = "
                        f"ggml_cont(ctx, ggml_permute(ctx, {u}_c, 1, 0, 2, 3));")
            body.append(f"    int64_t {u}_s = {a[0]}->ne[1];")
            _put(body, prim, node, f"sl_ax(ctx, {u}_b, 1, 0, {u}_s)")
        elif prim == "top_k":
            # ggml_top_k 正是「沿 ne0 取每行前 k 大的下标」（PRIMS 的 ref
            # 是 `argsort(-x, stable)[..., :k]`）。**并列处取谁没有定义**
            # （PRIMS 原文）—— ggml 用不稳定的 partial_sort 且故意把前两个
            # 换序（ops.cpp 8785：「emphasize that the order is not important」），
            # 这与承诺不冲突：并列的选择本来就不是语义的一部分。
            u = _gvar(node["outs"][0])
            body.append(f"    int {u}_k = {int(at['k'])};")
            body.append(f"    if ({u}_k > (int) {a[0]}->ne[0]) "
                        f"{u}_k = (int) {a[0]}->ne[0];")
            # **并列取较小下标**（by1exec 定死的那条：DSA 的 ReLU 会产生
            # 大量精确 0，top-k 边界天天撞并列；三条后端都按这条算，
            # "三后端一致"才是有意义的判据）。ggml_top_k 用的是**不稳定**
            # partial_sort，还故意换序前两个 —— 所以给它喂一条**denormal
            # 量级的递增偏移** i·1e-30：对任何可表示的正分数都小于 1 ulp
            # （不改变顺序），但把精确相等的 0 排成小下标优先。
            body.append(f"    struct ggml_tensor * {u}_ix = ggml_reshape_2d("
                        f"ctx, ggml_arange(ctx, 0.0f, (float) {a[0]}->ne[0], "
                        f"1.0f), {a[0]}->ne[0], 1);")
            body.append(f"    int64_t {u}_bn[4] = {{ {a[0]}->ne[0], "
                        f"{a[0]}->ne[1], {a[0]}->ne[2], {a[0]}->ne[3] }};")
            body.append(f"    struct ggml_tensor * {u}_pat = ggml_new_tensor("
                        f"ctx, GGML_TYPE_F32, 4, {u}_bn);")
            body.append(f"    struct ggml_tensor * {u}_eps = ggml_repeat(ctx, "
                        f"ggml_scale(ctx, {u}_ix, 1e-30f), {u}_pat);")
            body.append(f"    struct ggml_tensor * {u}_adj = ggml_sub(ctx, "
                        f"{a[0]}, {u}_eps);")
            _put(body, prim, node, f"ggml_top_k(ctx, {u}_adj, {u}_k)")
            ivals.add(node["outs"][0])
        elif prim in ("one_hot", "gather"):
            # 这两条都靠**一个加法拼出来的相等判定**（ggml 这版没有 EQ/LE
            # 逐元算子；GGML_OP_LE 最近的邻居是 ARGSORT，不是比较）：
            #   eq(a,b) = clamp(1 - |a-b|, 0, 1),  |d| = sqrt(d²)（无 abs 算子）
            # 取值是**整数**（下标），所以 |a-b| ∈ {0, 1, 2, …}，判定精确。
            #
            # one_hot: out[W, I, d1, d2] = Σ_i eq(arange(W), ids[i, d1, d2])
            # gather : out[K, d1, d2]    = Σ_m eq(arange(M), idx[k, d1, d2])·x[m]
            #   （gather = take_along_axis 沿 dim0；用 one-hot 矩阵把"取值"
            #    写成"加权求和"—— 图的维度里没有逐列 gather 原语。）
            u = _gvar(node["outs"][0])
            if prim == "one_hot":
                ids, ref = a[0], a[1]
                body.append(f"    int64_t {u}_w = {ref}->ne[0];")
                body.append(f"    int64_t {u}_i = {ids}->ne[0];")
                body.append(f"    int64_t {u}_d[4] = {{ {u}_w, {u}_i, "
                            f"{ids}->ne[1], {ids}->ne[2] }};")
                ids4 = (f"ggml_reshape_4d(ctx, {ids}, 1, {u}_i, "
                        f"{ids}->ne[1], {ids}->ne[2])")
                ar4 = (f"ggml_reshape_4d(ctx, ggml_arange(ctx, 0.0f, "
                       f"(float) {u}_w, 1.0f), {u}_w, 1, 1, 1)")
                ax = 1
            else:
                x, idx = a[0], a[1]
                body.append(f"    int64_t {u}_m = {x}->ne[0];")
                body.append(f"    int64_t {u}_d[4] = {{ {u}_m, {idx}->ne[0], "
                            f"{x}->ne[1], {x}->ne[2] }};")
                ids4 = (f"ggml_reshape_4d(ctx, {idx}, 1, {idx}->ne[0], "
                        f"{x}->ne[1], {x}->ne[2])")
                ar4 = (f"ggml_reshape_4d(ctx, ggml_arange(ctx, 0.0f, "
                       f"(float) {u}_m, 1.0f), {u}_m, 1, 1, 1)")
                ax = 0
            body.append(f"    struct ggml_tensor * {u}_pat = ggml_new_tensor(ctx, "
                        f"GGML_TYPE_F32, 4, {u}_d);")
            body.append(f"    struct ggml_tensor * {u}_A = ggml_repeat(ctx, "
                        f"{ar4}, {u}_pat);")
            # 下标是 I32（top_k 的输出），比对要在 F32 上做 —— 先 cast。
            # \`ggml_sub\` 的 assert 是"同类型"，混着会当场炸（不是静默）。
            body.append(f"    struct ggml_tensor * {u}_B = ggml_repeat(ctx, "
                        f"ggml_cast(ctx, {ids4}, GGML_TYPE_F32), {u}_pat);")
            body.append(f"    struct ggml_tensor * {u}_EQ = ggml_clamp(ctx, "
                        f"ggml_add1(ctx, ggml_neg(ctx, ggml_sqrt(ctx, "
                        f"ggml_sqr(ctx, ggml_sub(ctx, {u}_A, {u}_B)))), t_one), "
                        f"0.0f, 1.0f);")
            if prim == "gather":
                body.append(f"    struct ggml_tensor * {u}_xv = ggml_reshape_4d("
                            f"ctx, ggml_cont(ctx, {x}), {u}_m, 1, "
                            f"{x}->ne[1], {x}->ne[2]);")
                _put(body, prim, node,
                     f"sum_ax(ctx, ggml_mul(ctx, {u}_EQ, {u}_xv), 0, 0)")
            else:
                _put(body, prim, node, f"sum_ax(ctx, {u}_EQ, {ax}, 0)")
        elif prim == "le":
            # out = (a <= b) ? 1 : 0 —— ggml 没有比较算子，走 map_custom2。
            # 两边先 repeat 到**广播后的形状**。形状**运行期取 max**（不是
            # 声明的形状）：k-pool 的可见性掩码是 [n_query, n_pool] 的外积，
            # 两条轴都是运行期的，声明形状里写不出来。
            u = _gvar(node["outs"][0])
            body.append(f"    int64_t {u}_bn[4];")
            body.append(f"    for (int _i = 0; _i < 4; _i++) "
                        f"{u}_bn[_i] = ({a[0]}->ne[_i] > {a[1]}->ne[_i])"
                        f" ? {a[0]}->ne[_i] : {a[1]}->ne[_i];")
            body.append(f"    struct ggml_tensor * {u}_pat = ggml_new_tensor(ctx, "
                        f"GGML_TYPE_F32, 4, {u}_bn);")
            body.append(f"    struct ggml_tensor * {u}_A = ggml_repeat(ctx, "
                        f"{_f32(a[0], ins[0])}, {u}_pat);")
            body.append(f"    struct ggml_tensor * {u}_B = ggml_repeat(ctx, "
                        f"{_f32(a[1], ins[1])}, {u}_pat);")
            _put(body, prim, node,
                 f"ggml_map_custom2(ctx, {u}_A, {u}_B, by1_le_f32, 1, NULL)")
        elif prim == "masked_fill":
            # out = m > 0 ? x : v（PRIMS 原文）。**ggml 没有 select**，而
            # `x*m + v*(1-m)` 在 v = -inf 时出 nan（-inf·0）——所以这条走
            # map_custom2 落**精确**语义（单线程，n_tasks=1）。
            u = _gvar(node["outs"][0])
            sv = u + "_v"
            # -inf 不能写成 -inff（C 里没有这个字面量，实测编译不过）。
            _vv = float(at["value"])
            if _vv == float("-inf"):
                _vs = "(-INFINITY)"
            elif _vv == float("inf"):
                _vs = "INFINITY"
            else:
                _vs = _f(_vv)
            body.append(f"    static float {sv} = {_vs};")
            _put(body, prim, node,
                 f"ggml_map_custom2(ctx, {_f32(a[0], ins[0])}, "
                 f"{_f32(a[1], ins[1])}, by1_masked_fill_f32, 1, &{sv})")
        elif prim == "ssm_scan":
            # **Mamba2 的递推 -> ggml_ssm_scan**（迁移的最后一格）。三处折算：
            #   ① 图把 softplus 放在**原语外面**（ref 吃 d = softplus(dt+bias)），
            #      而内核内部又做一次 softplus —— 先做**逆 softplus**（见
            #      by1_inv_softplus_f32）；
            #   ② 布局：图是 [dim, n, 头, 1]，内核要 [dim, 头, n_tokens, 1]；
            #   ③ 内核的 y 只有 Σ s·C，**D 跳连在外围补**（ref 是 y = s·C + D·x）。
            u = _gvar(node["outs"][0])
            _sx = g.vals.get(node["ins"][2]) or (None, None, None, None)
            _sb = g.vals.get(node["ins"][3]) or (None, None, None, None)
            if _sx[0] is None or _sx[2] is None or _sb[0] is None:
                raise SystemExit("  [不支持] ssm_scan 的静态维推不出来")
            _hd, _h, _ssm = int(_sx[0]), int(_sx[2]), int(_sb[0])
            for _nm, _xx in (("x", a[2]), ("B", a[3]), ("C", a[4])):
                body.append(f"    struct ggml_tensor * {u}_{_nm} = ggml_cont(ctx, "
                            f"ggml_permute(ctx, {_xx}, 0, 2, 1, 3));")
            body.append(f"    struct ggml_tensor * {u}_d0 = ggml_cont(ctx, "
                        f"ggml_permute(ctx, {a[0]}, 0, 2, 1, 3));")
            body.append(f"    struct ggml_tensor * {u}_d1 = ggml_reshape_3d(ctx, "
                        f"{u}_d0, {_h}, n, 1);")
            body.append(f"    struct ggml_tensor * {u}_dt = ggml_map_custom1(ctx, "
                        f"{u}_d1, by1_inv_softplus_f32, 1, NULL);")
            body.append(f"    struct ggml_tensor * {u}_A = ggml_reshape_2d(ctx, "
                        f"{a[1]}, 1, {_h});")
            body.append(f"    float * {u}_s0b = (float *) calloc("
                        f"(size_t) {_ssm} * {_hd} * {_h}, sizeof(float));")
            body.append(f"    int64_t {u}_sn[4] = {{ {_ssm}, {_hd}, {_h}, 1 }};")
            body.append(f"    struct ggml_tensor * {u}_s0 = ggml_new_tensor(ctx, "
                        f"GGML_TYPE_F32, 4, {u}_sn);")
            body.append(f"    {u}_s0->data = {u}_s0b;")
            body.append(f"    int32_t * {u}_idb = (int32_t *) calloc(1, sizeof(int32_t));")
            body.append(f"    struct ggml_tensor * {u}_ids = ggml_new_tensor_1d(ctx, "
                        f"GGML_TYPE_I32, 1);")
            body.append(f"    {u}_ids->data = {u}_idb;")
            body.append(f"    struct ggml_tensor * {u}_res = ggml_ssm_scan(ctx, "
                        f"{u}_s0, {u}_x, {u}_dt, {u}_A, {u}_B, {u}_C, {u}_ids, 1);")
            body.append(f"    struct ggml_tensor * {u}_yv = ggml_view_4d(ctx, {u}_res, "
                        f"{_hd}, {_h}, n, 1, (size_t) {_hd} * {u}_res->nb[0], "
                        f"(size_t) {_hd} * {_h} * {u}_res->nb[0], "
                        f"(size_t) {_hd} * {_h} * n * {u}_res->nb[0], 0);")
            body.append(f"    struct ggml_tensor * {u}_D = ggml_reshape_4d(ctx, "
                        f"{a[5]}, 1, {_h}, 1, 1);")
            body.append(f"    struct ggml_tensor * {u}_y2 = ggml_add(ctx, {u}_yv, "
                        f"ggml_mul(ctx, {u}_x, {u}_D));")
            _put(body, prim, node,
                 f"ggml_cont(ctx, ggml_permute(ctx, {u}_y2, 0, 2, 1, 3))")
        elif prim == "delta_rule":
            # **delta 规则 -> ggml_gated_delta_net**（Linear 的 GDN 与 KDA
            # 共用这一条；内核按 g->ne[0] 自己分派：1=逐头、S_v=逐维）。
            #
            # 布局：图里是 exec [1, 头, n, dim] = ggml [dim, n, 头, 1]，
            # 内核要 [S, 头, n_tokens, n_seqs] —— **头和 token 要对调**。
            # 输出内核给 [S_v, H, n]（与 IR 路径的 y4 视图同一个约定），
            # 再对调回图的 [dim, n, 头, 1]。
            #
            # ⚠ 内核内部乘 `scale = 1/sqrt(S_v)`（ops.cpp 11193），而图的
            # q 已经带 lowering 给的 `dk^-0.5` —— **这里补乘 sqrt(S_v)
            # 抵消**，否则 q 被缩放两次（差 1/S_v）。
            sq = _gvar(node["ins"][2])
            shp_v = g.vals.get(node["ins"][2]) or (None, None, None, None)
            shp_q = g.vals.get(node["ins"][0]) or (None, None, None, None)
            if shp_q[0] is None or shp_v[0] is None or shp_v[2] is None:
                raise SystemExit("  [不支持] delta_rule 的静态维推不出来")
            dk_, dv_, nv_ = int(shp_q[0]), int(shp_v[0]), int(shp_v[2])
            if dk_ != dv_:
                raise SystemExit(
                    "  [不支持] delta_rule：ggml 内核要求 k_dim == v_dim"
                    "（%d vs %d）" % (dk_, dv_))
            u = _gvar(node["outs"][0])
            for nm, xx in (("q", a[0]), ("k", a[1]), ("v", a[2]),
                           ("be", a[3]), ("g", a[4])):
                body.append(f"    struct ggml_tensor * {u}_{nm} = "
                            f"ggml_cont(ctx, ggml_permute(ctx, {xx}, 0, 2, 1, 3));")
            body.append(f"    float * {u}_s0b = (float *) calloc("
                        f"(size_t) {dv_} * {dv_} * {nv_}, sizeof(float));")
            body.append(f"    int64_t {u}_sn[4] = {{ {dv_}, {dv_}, {nv_}, 1 }};")
            body.append(f"    struct ggml_tensor * {u}_s0 = ggml_new_tensor(ctx, "
                        f"GGML_TYPE_F32, 4, {u}_sn);")
            body.append(f"    {u}_s0->data = {u}_s0b;")
            body.append(f"    struct ggml_tensor * {u}_res = ggml_gated_delta_net("
                        f"ctx, {u}_q, {u}_k, {u}_v, {u}_g, {u}_be, {u}_s0, 1);")
            body.append(f"    struct ggml_tensor * {u}_y4 = ggml_view_4d(ctx, "
                        f"{u}_res, {dv_}, {nv_}, n, 1, "
                        f"(size_t) {dv_} * {u}_res->nb[0], "
                        f"(size_t) {dv_} * {nv_} * {u}_res->nb[0], 0, 0);")
            body.append(f"    struct ggml_tensor * {u}_yp = ggml_cont(ctx, "
                        f"ggml_permute(ctx, {u}_y4, 0, 2, 1, 3));")
            _put(body, prim, node,
                 f"ggml_scale(ctx, {u}_yp, sqrtf((float) {dv_}))")
        else:
            # **要么实现，要么拒绝。** 这几族现在仍然拒绝，但 2026-10-17
            # **修正了一条过时的摸底结论** —— 原来写"这版 ggml 没有 top_k
            # 和比较算子（2026-10-14 逐个查过）"，而当天逐个数
            # `enum ggml_op`（101 个）之后：
            #   top_k / gather  -> `GGML_OP_TOP_K`、`ARGSORT`、`GET_ROWS`
            #   delta_rule      -> `GGML_OP_GATED_DELTA_NET`、
            #                      `GATED_LINEAR_ATTN`、`RWKV_WKV7`
            #   DSA 索引        -> `GGML_OP_LIGHTNING_INDEXER`
            # **原语都在**；缺的是 by1 语义到它们的映射实现（可能要折算，
            # 像 SSM 那次）。教训：**摸底结论也要有判据** —— 一句
            # "我查过"写在注释里没人复核，就烂成了错误的前提。
            raise SystemExit(
                "  [不支持] 图的 %r 原语还没有 ggml 映射 —— "
                "**要么实现、要么在 lowering 里拒绝**" % prim)

    # 图预算按**实际发射的调用数**定，不按 IR kind 估 —— 复合式一个
    # 原语顶好几个 ggml 节点（softplus 7 个），按估的数会 GGML_ASSERT
    # `cgraph->n_nodes < cgraph->size`（mla-shaped 实测炸在这）。
    # 宁多勿少：图句柄就是几 KB 的指针数组。
    _calls = sum(str(line).count("ggml_") + 2 * sum(
        str(line).count(h) for h in ("rep_ax(", "sl_ax(", "sum_ax("))
        for line in body)
    n_nodes = max(n_nodes, 3 * _calls + 64)
    # N_NODES 的 define 在头段（发射初）就写死了 —— 重算后要**回写**，
    # 否则改的是变量、跑的是旧预算（实测 define 还停在 181）。
    for _i, _ln in enumerate(head):
        if str(_ln).startswith("#define N_NODES "):
            head[_i] = f"#define N_NODES {n_nodes}"

    logits = _gvar(g.nodes[-1]["outs"][0])
    return _assemble(head, body, logits, n_nodes, mem_tok, mem_pair, dump)


def _assemble(head, body, logits, n_nodes, mem_tok, mem_pair, dump=()):
    """骨架：`GGML_HEAD` + 常量表 + body + `main`。**两条发射路径共用**
    （IR 路径 / 图路径）—— 骨架一样，对拍才是同条件。

    `dump`：要打印的中间节点名（`--dump`）。**打印按张量的逻辑步长读**
    （`nb`）—— 视图张量的 `->data + 扁平下标` 不等于逻辑位置，这是
    调试 Linear 时连踩五个假象换来的教训。
    """
    dump_c = f"""
static struct ggml_tensor * dbg_nodes[64];
static const char * dbg_names[64];
static int dbg_count = 0;
static void dbg_reg(const char * nm, struct ggml_tensor * t) {{
    if (dbg_count < 64) {{ dbg_names[dbg_count] = nm; dbg_nodes[dbg_count] = t; dbg_count++; }}
}}
static void dbg_show(int nshow) {{
    for (int k = 0; k < dbg_count; k++) {{
        struct ggml_tensor * t = dbg_nodes[k];
        printf("[dump] %s (%lld,%lld,%lld,%lld):", dbg_names[k],
               (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        int64_t tot = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        int64_t show = nshow < tot ? nshow : tot;
        for (int64_t i = 0; i < show; i++) {{
            int64_t i0 = i % t->ne[0];
            int64_t i1 = (i / t->ne[0]) % t->ne[1];
            int64_t i2 = (i / (t->ne[0] * t->ne[1])) % t->ne[2];
            int64_t i3 = i / (t->ne[0] * t->ne[1] * t->ne[2]);
            float v = *(float *) ((char *) t->data + i0 * t->nb[0] + i1 * t->nb[1]
                                               + i2 * t->nb[2] + i3 * t->nb[3]);
            printf(" %.8f", v);
        }}
        printf("\\n");
    }}
}}
"""
    # 在**声明点**注册（层循环里的变量只在那层作用域内）。
    import re as _re
    body = list(body)
    if dump:
        out = []
        for ln in body:
            out.append(ln)
            m = _re.search(r"struct ggml_tensor \* (\w+) =", ln)
            if m and any(pat in m.group(1) for pat in dump):
                ind = ln[:len(ln) - len(ln.lstrip())]
                out.append('%sdbg_reg("%s", %s);' % (ind, m.group(1), m.group(1)))
        body = out
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
    dbg_show(16);
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
        # **不能 `except: pass`** —— by1lint 的静默写法之一：
        # 错误被吃掉，调用方看到"一切正常"，而 stamp 根本没打上。
        stamp = "/* by1ggml */"
    return stamp + GGML_HEAD + dump_c + "\n".join(head) + main, \
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

    # **没有"支持集预检"这一层了**（A3-3，2026-10-17）：门就是发射表自己 ——
    # 要么映射、要么在 lowering/发射时点名拒绝。原来的 check_support
    # 是按 IR kind 列白名单，它会把"能降成图"的机制挡在门口
    # （实测 mla-shaped / MoE / KDA 都这么被挡过）。
    # **w.bin 布局的唯一定义在 `by1ir.weight_plan`**（1.0.0-rc0 · A3-1）。
    # 以前这里反向 import by1c 只为拿 `order`（下一行的 `del decls` 就是
    # 那个证据）—— 布局是 **IR 的事实**，不该由某个后端代管。
    _bir = load("by1ir")
    order = _bir.weight_plan(ir)
    total = sum(int(np.prod(s)) for s in _bir.params_of(ir).values())

    try:
        # **图发射是唯一的一条路**（A3-2 起默认，A3-3 把旧路删掉）。
        # 逐节点 PRIMS -> ggml；判据是 --selftest 与矩阵上的数值对拍。
        _g = load("by1graph").lower_model(ir, name)
        src, n_nodes, mem_tok, mem_pair = emit_model_c_graph(
            _g, ir, order, params_shapes, dump=args.dump)
    except SystemExit as e:
        # 降不动 / 图里有原语没映射 —— **点名拒绝**，语义和融合门同款
        print(str(e))
        if expect == "reject":
            print("\n  [PASS] 按承诺拒绝（图路径点名拒绝）")
            return "reject-ok"
        return "fail"
    if expect == "reject":
        print("\n  [FAIL] 期望拒绝，结果支持集放行了")
        return "fail"

    workdir = args.workdir or "cgen-ggmlg-" + os.path.splitext(name)[0]
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
       ④ mla-shaped        轴约定族（slice/repeat/sum/concat/l2_norm）数值对拍
       ⑤ gemma-4-31B       真实 IR（87 层，含视觉塔）          仅编译
       ⑥ moe-routing/kda/glm-text-tiny/qwen3-next  MoE·KDA·Linear  数值对拍
          （A3-4 起这三族在图路径上"要么实现，要么拒绝"变成了**数值判据**）
    """
    root = by1paths.root()
    twin_path = os.path.join(root, "_gemma-shaped.by1")
    by1io.write_text(twin_path, TWIN_BY1)

    cases = [
        (os.path.join(root, "models", "llama-shaped.by1"), True, "pass"),
        (os.path.join(root, "models", "gpt2-tiny.by1"), True, "pass"),
        (twin_path, True, "pass"),
        # T16 第二批的整模判卷人：MLA 图用满 slice/repeat/sum/concat/l2_norm
        (os.path.join(root, "models", "mla-shaped.by1"), True, "pass"),
        (os.path.join(root, "models", "gemma-4-31B.by1"), False, "pass"),
        # **A3-4（2026-10-17）：MoE / KDA / Linear 进了图路径的支持集。**
        # 这三条原来写的是"必须拒绝"—— 图路径只会点名拒绝。现在它们是
        # **数值判据**（判卷人是 by1exec）：拒绝不再是正确行为，
        # 数值对不上才是不支持。
        (os.path.join(root, "models", "moe-routing-shaped.by1"), True, "pass"),
        (os.path.join(root, "models", "kda-shaped.by1"), True, "pass"),
        (os.path.join(root, "models", "glm-text-tiny.by1"), True, "pass"),
        (os.path.join(root, "models", "qwen3-next-shaped.by1"), True, "pass"),
        # **SSM 的图路**（`ssm_scan` -> ggml_ssm_scan）。它此前只由 IR 路覆盖，
        # 图路少这一格 IR 路就删不掉；中间那次逆 softplus 折算见
        # `by1_inv_softplus_f32`。
        (os.path.join(root, "models", "ssm-shaped.by1"), True, "pass"),
    ]
    results = []
    for path, run, expect in cases:
        try:
            results.append((os.path.basename(path), run_one(path, args, run, expect)))
        except SystemExit as e:
            print(str(e))
            results.append((os.path.basename(path), "fail"))

    # T17-S2 判据：**增量与全量 logits 一致**（by1graph 的 decode 判卷人，
    # 带"cache 丢老 token"负样必红）
    try:
        _bg = load("by1graph")
        _rc = _bg.decode_check(os.path.join(root, "models", "llama-shaped.by1"))
        results.append(("decode-equivalence (T17-S2)",
                        "ok" if _rc == 0 else "fail"))
    except Exception as e:
        print(str(e)[:120])
        results.append(("decode-equivalence (T17-S2)", "fail"))

    # T17-S3 判据：**状态 round-trip**（cache 导出/导入逐 token bit 级
    # 相同 + 翻一字节必红）
    try:
        _bg2 = load("by1graph")
        _rc2 = _bg2.roundtrip_check(os.path.join(root, "models",
                                                 "llama-shaped.by1"))
        results.append(("state-roundtrip (T17-S3)",
                        "ok" if _rc2 == 0 else "fail"))
    except Exception as e:
        print(str(e)[:120])
        results.append(("state-roundtrip (T17-S3)", "fail"))

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
                    help="这份 .by1 应当被发射器/lowering 拒绝")
    ap.add_argument("--ggml-src", default=None)
    ap.add_argument("--dump", action="append", default=[],
                    help="打印名字含此子串的中间节点（ne + 前 16 个值，"
                         "按张量逻辑步长读）—— 调试用，可重复")
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
