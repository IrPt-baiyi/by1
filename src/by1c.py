#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 c -- 第三个后端：从 IR 生成 **C 代码**，编译，跑起来，和 NumPy 后端对拍。

这是 Block 3b 的最后一公里。和 llama.cpp 的区别只剩「用 ggml 的原语还是手写循环」——
图的结构、张量的绑定、状态的形状，都是同一套（已经和 qwen3next.cpp 的源码核过）。

纪律：**能生成的才生成，不支持就明确报错**。绝不悄悄生成一个算错的 C。

用法:
  python by1c.py llama-shaped.by1 --gcc <gcc.exe> [--seq 16]
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

# **eps 的默认值只有一个地方写**（`by1ir.EPS_DEFAULT`）。
# 这个文件里原来有 9 处 `1e-5`；C_HEAD 里那段"eps 必须由调用方传进来"
# 的注释记的就是这一类 —— 而它当时没管住 `qk_norm()`。
try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:                     # 单独拷一个文件出去时兜底
    _DEFAULT_EPS = 1e-5

# **打分函数的序数表只有一份**（`by1ir.SCORE_KINDS`）——
# C 端只收整数，而"哪个整数是哪个函数"必须和 by1ir 那张表同源。
try:
    from by1ir import _score_kind_id
    from by1ir import SCORE_KINDS as _SCORE_KINDS
except ImportError:                     # 兜底：和 by1ir 里那份逐字相同
    _SCORE_KINDS = ("softmax", "sigmoid", "sqrtsoftplus", "softplus")

    def _score_kind_id(kind):
        try:
            return _SCORE_KINDS.index(str(kind).strip().lower())
        except ValueError:
            return None

HERE = os.path.dirname(os.path.abspath(__file__))
MANGLE = re.compile(r"[^0-9A-Za-z]")

# **C 后端的支持集**（1.0.0-rc0 · A3-1）。以前这个集合是**隐式**的 ——
# 它藏在 `_emit_c_body` 那条 if 链的最后一个 `else` 里。布局表搬去
# `by1ir.weight_plan` 之后，它必须显式写出来：这是一条**能力声明**
# （"这个后端会算哪些 kind"），不是布局的一部分。
# 判据：`by1all` 的编译/数值档不退；`raw-escape.by1` 仍按承诺拒绝。
C_KINDS = ("Norm", "Attention", "FFN", "Linear", "SSM", "KDA",
           "MLA", "MoE", "External", "Add")



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

def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C_HEAD = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *W;                 /* 所有权重，扁平存放 */

#define P(nm) (W + OFF_##nm)
#define N(nm) (LEN_##nm)

/* eps 必须由调用方传进来 —— 原来写死 1e-5，而 mla-shaped 的 rms_eps 是 1e-6。
   误差 1.5e-04，而其它四个模型恰好都是 1e-5 所以一直没露。 */
static void rmsnorm(float *o, const float *x, const float *w, int n, int d,
                    int op, float eps) {
    for (int i = 0; i < n; i++) {
        const float *r = x + (size_t)i * d;
        float s = 0.f;
        for (int j = 0; j < d; j++) s += r[j] * r[j];
        s = 1.0f / sqrtf(s / d + eps);
        float *q = o + (size_t)i * d;
        for (int j = 0; j < d; j++) q[j] = r[j] * s * (op ? (1.f + w[j]) : w[j]);
    }
}

/* **LayerNorm 和 RMSNorm 不是一个东西。**
 *
 * 这里原来只有 rmsnorm —— 而 C 后端对每个 Norm 都发 rmsnorm()，
 * 也就是**把 LayerNorm 当 RMSNorm 算**（少了减均值和 bias）。
 * 那会生成一个看起来对但算错的模型，所以这里必须**拒绝**。
 * 现在补上。
 *
 * 三个东西顶着相近的名字，算的不是一回事：
 *     rmsnorm    只除均方根
 *     layernorm  减均值、除标准差、还有一个 bias
 *     one_plus   权重是乘 w 还是乘 (1+w) —— 这是第三个约定
 */
static void layernorm(float *o, const float *x, const float *w,
                      const float *b, int n, int d, int op, float eps) {
    for (int i = 0; i < n; i++) {
        const float *r = x + (size_t)i * d;
        float mu = 0.f;
        for (int j = 0; j < d; j++) mu += r[j];
        mu /= d;
        float var = 0.f;
        for (int j = 0; j < d; j++) { float u = r[j] - mu; var += u * u; }
        var /= d;
        float s = 1.0f / sqrtf(var + eps);
        float *q = o + (size_t)i * d;
        for (int j = 0; j < d; j++)
            q[j] = (r[j] - mu) * s * (op ? (1.f + w[j]) : w[j])
                   + (b ? b[j] : 0.f);
    }
}

static void linear(float *o, const float *x, const float *w, int n, int di, int dout) {
    for (int i = 0; i < n; i++) {
        const float *r = x + (size_t)i * di;
        float *q = o + (size_t)i * dout;
        for (int j = 0; j < dout; j++) {
            const float *wr = w + (size_t)j * di;
            float s = 0.f;
            for (int k = 0; k < di; k++) s += r[k] * wr[k];
            q[j] = s;
        }
    }
}

static void linear_b(float *o, const float *x, const float *w, const float *b,
                     int n, int di, int dout) {
    linear(o, x, w, n, di, dout);
    if (b) for (int i = 0; i < n; i++)
        for (int j = 0; j < dout; j++) o[(size_t)i * dout + j] += b[j];
}

/* **无门控那一层用的激活。**
 *
 * 这里原来只有 silu_ —— 而 C 后端的无门控分支写死用它。
 * 和 by1codegen 那边当初的 bug 一模一样：一条分支只为一种情况写过，
 * 于是 GPT-2 的 `act = gelu_new` 一声不吭地被忽略。
 * （PyTorch 那边修过了，C 这边没有。） */
static void gelu_tanh_(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                        (v + 0.044715f * v * v * v)));
    }
}
static void gelu_erf_(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
}
static void relu2_(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i] < 0 ? 0 : x[i];
        x[i] = v * v;
    }
}
static void silu_(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* pairing: 0 = half, 1 = interleaved
   YaRN 不只是缩放：它改的是**频率本身**（低频插值、高频外推、中间 ramp）。
   truncate=0 时校正区间不取整 —— 过渡带的位置会因此不同。 */
static void rope(float *q, float *k, int n, int nh, int nkv, int hd,
                 float base, int pairing, float scale, int nrot,
                 int rtype, float yfac, int yorig, float ybf, float ybs,
                 int ytrunc, float yhf, float ylf) {
    /* rtype: 0 = 不缩放  1 = yarn  2 = llama3
       **两种缩放都改频率、参数名还重叠，所以类型必须显式传进来。**
       yhf/ylf 是 llama3 的 high/low_freq_factor —— 不复用 ybf/ybs，
       因为它们含义不同（YaRN 的 beta 是 ramp 起止维，llama3 的是波长阈值）。 */
    /* 全程 float64 —— numpy 那边是 float64，用 float32 算会把 ramp 的
       过渡带挪一点点，位置 0 上看不出来，多了就累积成 1e-3 量级。 */
    const double PI2 = 6.283185307179586;
    int rdim = nrot;                 /* 频率按实际旋转的那一段算 */
    int half = rdim / 2;
    float *inv = (float *)malloc(sizeof(float) * half);
    double dlo = 0.0, dhi = 1.0;
    if (rtype == 1) {
        dlo = ((double)rdim * log((double)yorig / ((double)ybf * PI2)))
              / (2.0 * log((double)base));
        dhi = ((double)rdim * log((double)yorig / ((double)ybs * PI2)))
              / (2.0 * log((double)base));
        if (ytrunc) { dlo = floor(dlo); dhi = ceil(dhi); }
        if (dlo < 0.0) dlo = 0.0;
        if (dhi > (double)(rdim - 1)) dhi = (double)(rdim - 1);
        if (dlo == dhi) dhi += 0.001;
    }
    for (int i = 0; i < half; i++) {
        double p = pow((double)base, -2.0 * (double)i / (double)rdim);
        if (rtype == 0) { inv[i] = (float)p; continue; }
        if (rtype == 2) {
            /* llama3（Llama 3.1）：按**波长**分三段，中间那段平滑插值。
               p 就是 inv_freq（已经取过倒数）。 */
            double wl = PI2 / p;
            double lo_wl = (double)yorig / (double)ylf;
            double hi_wl = (double)yorig / (double)yhf;
            double out = (wl > lo_wl) ? (p / (double)yfac) : p;
            if (wl >= hi_wl && wl <= lo_wl) {
                double sm = ((double)yorig / wl - (double)ylf)
                            / ((double)yhf - (double)ylf);
                out = (1.0 - sm) * out / (double)yfac + sm * out;
            }
            inv[i] = (float)out;
            continue;
        }
        double r = ((double)i - dlo) / (dhi - dlo);
        if (r < 0.0) r = 0.0;
        if (r > 1.0) r = 1.0;
        /* p 已经是 1/pos（pos = base^(2i/hd)），所以插值项是 p/fac，
           不是 1/(fac*p) —— 后者等于 pos/fac，整个反了。
           非 YaRN 的情况恰好退化成 p，所以只有 YaRN 的模型才暴露这个错。 */
        inv[i] = (float)((p / (double)yfac) * r + p * (1.0 - r));
    }
    int nhalf = half;
    for (int pos = 0; pos < n; pos++) {
        for (int h = 0; h < nh + nkv; h++) {
            int isq = h < nh;
            if (!isq && h - nh >= nkv) continue;
            float *v = isq ? q + ((size_t)pos * nh + h) * hd
                           : k + ((size_t)pos * nkv + (h - nh)) * hd;
            if (pairing == 0) {
                for (int i = 0; i < nhalf; i++) {
                    float f = inv[i] * pos;
                    float c = cosf(f) * scale, s = sinf(f) * scale;
                    float a = v[i], b = v[i + nrot / 2];
                    v[i] = a * c - b * s;
                    v[i + nrot / 2] = a * s + b * c;
                }
            } else {
                for (int i = 0; i < nhalf; i++) {
                    float f = inv[i] * pos;
                    float c = cosf(f) * scale, s = sinf(f) * scale;
                    float a = v[2 * i], b = v[2 * i + 1];
                    v[2 * i] = a * c - b * s;
                    v[2 * i + 1] = a * s + b * c;
                }
            }
        }
    }
    free(inv);
}


/* ── MoE：一个算子 + 一组 attrs ─────────────────────────────────── */
static float dot_(const float *a, const float *b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static float act_gate_(float g, float u, int gptoss, float lim, float alpha) {
    if (gptoss) {
        if (lim > 0.f) { if (g > lim) g = lim; if (u > lim) u = lim; if (u < -lim) u = -lim; }
        return (u + 1.f) * (g / (1.f + expf(-alpha * g)));
    }
    return (g / (1.f + expf(-g))) * u;
}

static void moe(float *o, const float *x, int n, int d, int E, int K, int H,
                const float *router, const float *rb, const float *sb,
                const float *w1, const float *w3,
                const float *w2, const float *b1, const float *b2,
                const float *b3, int sh, int shh, const float *sw1,
                const float *sw3, const float *sw2, const float *sg,
                int routing, int score, int ng, int tg,
                int gptoss, float lim, float alpha, float scale) {
    /* routing: 0 = softmax_topk（Mixtral）  1 = topk_softmax（GPT-OSS）
                2 = sigmoid_topk（Step-3.7） 3 = sigmoid_group_topk（noaux_tc，
                    Ling / DeepSeek V3 / GLM）
       **四种是四个机制。** 这里原来只有 0 / 1 两种，`sigmoid_topk` 和
       `sigmoid_group_topk` 落进 else —— 那一支一直是 softmax_topk。
       见 history/ir.md 130。

       score: **打分函数，和 routing 正交** —— 0 = softmax, 1 = sigmoid,
              2 = sqrtsoftplus, 3 = softplus。
              取值和 by1ir.SCORE_KINDS 同序（由 codegen 传序数，不传字符串）。
              `routing` 2/3 两支读它；0/1 两支**不经过打分函数**（top-k 直接
              做在 logits / softmax(logits) 上），所以那两支忽略 score。 */
    int *idx = (int *)malloc(sizeof(int) * n * K);
    float *wts = (float *)malloc(sizeof(float) * n * K);
    float *lg = (float *)malloc(sizeof(float) * E);
    float *sc = (float *)malloc(sizeof(float) * E);
    float *ch = (float *)malloc(sizeof(float) * E);
    float *gs = (float *)malloc(sizeof(float) * (ng > 0 ? ng : 1));
    float *gu = (float *)malloc(sizeof(float) * H);
    float *uu = (float *)malloc(sizeof(float) * H);
    float *hh = (float *)malloc(sizeof(float) * H);
    for (int i = 0; i < n; i++) {
        const float *xi = x + (size_t)i * d;
        for (int e = 0; e < E; e++)
            lg[e] = dot_(xi, router + (size_t)e * d, d) + (rb ? rb[e] : 0.f);
        if (routing == 1) {
            for (int k = 0; k < K; k++) {
                int best = -1; float bv = -INFINITY;
                for (int e = 0; e < E; e++) {
                    int used = 0;
                    for (int q = 0; q < k; q++) if (idx[i * K + q] == e) used = 1;
                    if (!used && lg[e] > bv) { bv = lg[e]; best = e; }
                }
                idx[i * K + k] = best;
                wts[i * K + k] = bv;
            }
            float mx = -INFINITY;
            for (int k = 0; k < K; k++) if (wts[i * K + k] > mx) mx = wts[i * K + k];
            float s = 0.f;
            for (int k = 0; k < K; k++) { wts[i * K + k] = expf(wts[i * K + k] - mx); s += wts[i * K + k]; }
            for (int k = 0; k < K; k++) wts[i * K + k] = wts[i * K + k] / s * scale;
        } else if (routing == 2 || routing == 3) {
            /* **先打分，再选** —— 打分函数是 score 那个参数，不是写死的 sigmoid。
               sqrtsoftplus（DeepSeek V4）= sqrt(softplus(x))，
               softplus 用数值稳定写法 max(x,0) + log1p(exp(-|x|))。
               闭集由上层（codegen 的 `_score_kind_id`）卡死，这里只认 0..3。 */
            if (score == 0) {
                /* softmax 打分：整行归一化。这一支理论上不该出现在
                   routing 2/3 下（by1ir 会拦），照算以免 C 端和 torch 端分叉。 */
                float mx = -INFINITY;
                for (int e = 0; e < E; e++) if (lg[e] > mx) mx = lg[e];
                float s0 = 0.f;
                for (int e = 0; e < E; e++) { sc[e] = expf(lg[e] - mx); s0 += sc[e]; }
                for (int e = 0; e < E; e++) sc[e] = sc[e] / s0;
            } else {
                for (int e = 0; e < E; e++) {
                    float v = lg[e];
                    if (score == 1) {
                        sc[e] = 1.f / (1.f + expf(-v));
                    } else {
                        /* softplus（score 2 再开方）—— 两个共用这一段 */
                        float sp = (v > 0.f ? v : 0.f)
                                 + log1pf(expf(-(v > 0.f ? v : -v)));
                        sc[e] = (score == 2) ? sqrtf(sp) : sp;
                    }
                }
            }
            /* **sb 只影响选择** —— 权重用未加偏置的 sc。 */
            for (int e = 0; e < E; e++) ch[e] = sc[e] + (sb ? sb[e] : 0.f);
            if (routing == 3) {
                int per = E / ng;
                for (int g = 0; g < ng; g++) {
                    float a1 = -INFINITY, a2 = -INFINITY;
                    for (int e = 0; e < per; e++) {
                        float v = ch[g * per + e];
                        if (v > a1) { a2 = a1; a1 = v; }
                        else if (v > a2) a2 = v;
                    }
                    gs[g] = a1 + a2;          /* 每组前 2 之和 */
                }
                /* 选 topk_group 个组，再把别的组整组屏蔽掉 */
                for (int k = 0; k < tg; k++) {
                    int best = -1; float bv = -INFINITY;
                    for (int g = 0; g < ng; g++) {
                        int used = 0;
                        for (int q = 0; q < k; q++) if (idx[i * K + q] == -(g + 1)) used = 1;
                        if (!used && gs[g] > bv) { bv = gs[g]; best = g; }
                    }
                    idx[i * K + k] = -(best + 1);   /* 暂存选中的组 */
                }
                char *ok = (char *)calloc((size_t)E, 1);
                for (int k = 0; k < tg; k++) {
                    int g = -idx[i * K + k] - 1;
                    for (int e = 0; e < per; e++) ok[g * per + e] = 1;
                }
                for (int e = 0; e < E; e++) if (!ok[e]) ch[e] = -INFINITY;
                free(ok);
            }
            for (int k = 0; k < K; k++) {
                int best = -1; float bv = -INFINITY;
                for (int e = 0; e < E; e++) {
                    int used = 0;
                    for (int q = 0; q < k; q++) if (idx[i * K + q] == e) used = 1;
                    if (!used && ch[e] > bv) { bv = ch[e]; best = e; }
                }
                idx[i * K + k] = best;
            }
            float s = 0.f;
            for (int k = 0; k < K; k++) s += sc[idx[i * K + k]];
            for (int k = 0; k < K; k++)
                wts[i * K + k] = sc[idx[i * K + k]] / (s + 1e-20f) * scale;
        } else {
            float mx = -INFINITY;
            for (int e = 0; e < E; e++) if (lg[e] > mx) mx = lg[e];
            float s = 0.f;
            for (int e = 0; e < E; e++) { lg[e] = expf(lg[e] - mx); s += lg[e]; }
            for (int k = 0; k < K; k++) {
                int best = -1; float bv = -INFINITY;
                for (int e = 0; e < E; e++) {
                    int used = 0;
                    for (int q = 0; q < k; q++) if (idx[i * K + q] == e) used = 1;
                    if (!used && lg[e] > bv) { bv = lg[e]; best = e; }
                }
                idx[i * K + k] = best;
                wts[i * K + k] = bv / s;
            }
            float ss = 0.f;
            for (int k = 0; k < K; k++) ss += wts[i * K + k];
            for (int k = 0; k < K; k++) wts[i * K + k] = wts[i * K + k] / ss * scale;
        }
    }
    for (int i = 0; i < n; i++) {
        const float *xi = x + (size_t)i * d;
        float *oi = o + (size_t)i * d;
        for (int j = 0; j < d; j++) oi[j] = 0.f;
        for (int k = 0; k < K; k++) {
            int e = idx[i * K + k];
            const float *W1 = w1 + (size_t)e * H * d;
            const float *W3 = w3 + (size_t)e * H * d;
            const float *W2 = w2 + (size_t)e * d * H;
            for (int h = 0; h < H; h++) {
                gu[h] = dot_(xi, W1 + (size_t)h * d, d) + (b1 ? b1[(size_t)e * H + h] : 0.f);
                uu[h] = dot_(xi, W3 + (size_t)h * d, d) + (b3 ? b3[(size_t)e * H + h] : 0.f);
                hh[h] = act_gate_(gu[h], uu[h], gptoss, lim, alpha);
            }
            for (int j = 0; j < d; j++) {
                float s = dot_(hh, W2 + (size_t)j * H, H);
                if (b2) s += b2[(size_t)e * d + j];
                oi[j] += wts[i * K + k] * s;
            }
        }
        if (sh) {
            float *sh1 = (float *)malloc(sizeof(float) * shh);
            float *sh3 = (float *)malloc(sizeof(float) * shh);
            for (int h = 0; h < shh; h++) sh1[h] = dot_(xi, sw1 + (size_t)h * d, d);
            for (int h = 0; h < shh; h++) sh3[h] = dot_(xi, sw3 + (size_t)h * d, d);
            for (int h = 0; h < shh; h++) sh1[h] = (sh1[h] / (1.f + expf(-sh1[h]))) * sh3[h];
            float g = 1.f;
            if (sg) g = 1.f / (1.f + expf(-dot_(xi, sg, d)));
            for (int j = 0; j < d; j++) oi[j] += g * dot_(sh1, sw2 + (size_t)j * shh, shh);
            free(sh1); free(sh3);
        }
    }
    free(idx); free(wts); free(lg); free(sc); free(ch); free(gs);
    free(gu); free(uu); free(hh);
}


/* GDN：短卷积 + delta 规则 + 门控归一化。带一个大小固定的递归状态。
   两个状态：delta 规则的矩阵，和短卷积的滑动历史。 */
static void gdn(float *o, const float *x, int n, int d,
                int nk, int nv, int dk, int dv, int ck,
                const float *wqkvz, const float *wba, const float *convw,
                const float *dtb, const float *alog, const float *nw,
                const float *wout, float l2eps, float neps) {
    int rep = nv / nk, kd = nk * dk, vd = nv * dv, cd = 2 * kd + vd;
    int zz = 2 * dk + 2 * dv * rep;          /* 每个 k 头在 qkvz 里的宽度 */
    float *qz = (float *)malloc(sizeof(float) * n * (2 * kd + 2 * vd));
    float *ba = (float *)malloc(sizeof(float) * n * 2 * nv);
    float *mix = (float *)malloc(sizeof(float) * n * cd);
    float *cnv = (float *)malloc(sizeof(float) * n * cd);
    float *Q = (float *)malloc(sizeof(float) * n * nv * dk);
    float *K = (float *)malloc(sizeof(float) * n * nv * dk);
    float *V = (float *)malloc(sizeof(float) * n * nv * dv);
    float *Z = (float *)malloc(sizeof(float) * n * nv * dv);
    float *B = (float *)malloc(sizeof(float) * n * nv);
    float *A = (float *)malloc(sizeof(float) * n * nv);
    float *G = (float *)malloc(sizeof(float) * n * nv);
    float *st = (float *)calloc((size_t)nv * dk * dv, sizeof(float));
    float *kvi = (float *)malloc(sizeof(float) * dv);
    float *dl = (float *)malloc(sizeof(float) * dv);
    float *hd = (float *)malloc(sizeof(float) * dv);

    linear(qz, x, wqkvz, n, d, 2 * kd + 2 * vd);
    linear(ba, x, wba, n, d, 2 * nv);
    for (int t = 0; t < n; t++) {
        const float *r = qz + (size_t)t * (2 * kd + 2 * vd);
        const float *b2 = ba + (size_t)t * 2 * nv;
        for (int h = 0; h < nk; h++) {
            const float *rh = r + (size_t)h * zz;
            for (int i = 0; i < dk; i++) {
                Q[((size_t)t * nv + h * rep) * dk + i] = rh[i];
                for (int u = 0; u < rep; u++)
                    Q[((size_t)t * nv + h * rep + u) * dk + i] = rh[i];
                for (int u = 0; u < rep; u++)
                    K[((size_t)t * nv + h * rep + u) * dk + i] = rh[dk + i];
            }
            for (int u = 0; u < rep; u++) {
                for (int i = 0; i < dv; i++) {
                    V[((size_t)t * nv + h * rep + u) * dv + i] =
                        rh[2 * dk + u * dv + i];
                    Z[((size_t)t * nv + h * rep + u) * dv + i] =
                        rh[2 * dk + dv * rep + u * dv + i];
                }
                B[(size_t)t * nv + h * rep + u] = b2[h * 2 * rep + u];
                A[(size_t)t * nv + h * rep + u] = b2[h * 2 * rep + rep + u];
            }
        }
        /* 拼成 q|k|v 做深度可分离因果卷积 */
        for (int h = 0; h < nk; h++) {
            for (int i = 0; i < dk; i++) {
                mix[(size_t)t * cd + h * dk + i] =
                    Q[((size_t)t * nv + h * rep) * dk + i];
                mix[(size_t)t * cd + kd + h * dk + i] = K[((size_t)t * nv + h * rep) * dk + i];
            }
        }
        for (int h = 0; h < nv; h++)
            for (int i = 0; i < dv; i++)
                mix[(size_t)t * cd + 2 * kd + h * dv + i] =
                    V[((size_t)t * nv + h) * dv + i];
    }
    /* 因果卷积：左边补 ck-1 个零 */
    for (int t = 0; t < n; t++) {
        for (int c = 0; c < cd; c++) {
            float s = 0.f;
            for (int j = 0; j < ck; j++) {
                int tt = t - (ck - 1) + j;
                if (tt < 0) continue;
                s += mix[(size_t)tt * cd + c] * convw[(size_t)c * ck + j];
            }
            cnv[(size_t)t * cd + c] = s / (1.f + expf(-s));
        }
    }
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < nv; h++) {
            for (int i = 0; i < dk; i++) {
                Q[((size_t)t * nv + h) * dk + i] = cnv[(size_t)t * cd + (h / rep) * dk + i];
                K[((size_t)t * nv + h) * dk + i] = cnv[(size_t)t * cd + kd + (h / rep) * dk + i];
            }
            for (int i = 0; i < dv; i++)
                V[((size_t)t * nv + h) * dv + i] = cnv[(size_t)t * cd + 2 * kd + h * dv + i];
        }
    }
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < nv; h++) {
            const float *qh = Q + ((size_t)t * nv + h) * dk;
            const float *kh = K + ((size_t)t * nv + h) * dk;
            float nq = 0.f, nkv = 0.f;
            for (int i = 0; i < dk; i++) { nq += qh[i] * qh[i]; nkv += kh[i] * kh[i]; }
            nq = 1.f / sqrtf(nq + l2eps); nkv = 1.f / sqrtf(nkv + l2eps);
            for (int i = 0; i < dk; i++) {
                ((float *)Q)[((size_t)t * nv + h) * dk + i] = qh[i] * nq;
                ((float *)K)[((size_t)t * nv + h) * dk + i] = kh[i] * nkv;
            }
            float bb = 1.f / (1.f + expf(-B[(size_t)t * nv + h]));
            B[(size_t)t * nv + h] = bb;
            G[(size_t)t * nv + h] = -expf(alog[h]) *
                log1pf(expf(A[(size_t)t * nv + h] + dtb[h]));
        }
    }
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < nv; h++) {
            float *S = st + ((size_t)h * dk) * dv;
            float gt = expf(G[(size_t)t * nv + h]);
            const float *kh = K + ((size_t)t * nv + h) * dk;
            const float *vh = V + ((size_t)t * nv + h) * dv;
            for (int i = 0; i < dk * dv; i++) S[i] *= gt;
            for (int j = 0; j < dv; j++) {
                float s = 0.f;
                for (int i = 0; i < dk; i++) s += S[i * dv + j] * kh[i];
                kvi[j] = s;
            }
            for (int j = 0; j < dv; j++)
                dl[j] = (vh[j] - kvi[j]) * B[(size_t)t * nv + h];
            for (int i = 0; i < dk; i++)
                for (int j = 0; j < dv; j++) S[i * dv + j] += kh[i] * dl[j];
            const float *qh = Q + ((size_t)t * nv + h) * dk;
            float sc = 1.f / sqrtf((float)dk);
            for (int j = 0; j < dv; j++) {
                float s = 0.f;
                for (int i = 0; i < dk; i++) s += S[i * dv + j] * qh[i];
                hd[j] = s * sc;
            }
            /* 门控 RMSNorm：先归一化，再乘 silu(z) */
            float vv = 0.f;
            for (int j = 0; j < dv; j++) vv += hd[j] * hd[j];
            vv = 1.f / sqrtf(vv / dv + neps);
            for (int j = 0; j < dv; j++) {
                float z = Z[((size_t)t * nv + h) * dv + j];
                hd[j] = (nw[j] * hd[j] * vv) * (z / (1.f + expf(-z)));
            }
            memcpy(o + (size_t)t * vd + h * dv, hd, sizeof(float) * dv);
        }
    }
    /* out_proj 就地覆盖前先把结果搬到临时区 */
    {
        float *tmp = (float *)malloc(sizeof(float) * n * vd);
        memcpy(tmp, o, sizeof(float) * n * vd);
        linear(o, tmp, wout, n, vd, d);
        free(tmp);
    }
    free(qz); free(ba); free(mix); free(cnv); free(Q); free(K); free(V); free(Z);
    free(B); free(A); free(G); free(st); free(kvi); free(dl); free(hd);
}


/* SSM：选择性状态空间（Mamba2 · Nemotron-H 那一版）。

   **没有 q/k/v** —— 和注意力是不同的算法族。状态由 A_log / D / dt_bias
   驱动，随 token 递推：

       h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t ⊗ B_t)
       y_t = C_t · h_t + D * x_t

   状态是 [nh, hd, ss]，**每次前向从零开始**（不做持久 state）——
   和 GDN 不同，那里跨调用留滑窗历史。

   两处最容易写反（写反了会读错内存、不会报错）：
       更新是**外积** x[hd] ⊗ B[ss]
       读出是 **h 右乘 C**（h[hd][ss] 逐行点 C[ss]）
   B / C 按 ng 分组，每个组服务 rep = nh/ng 个头。

   判卷人：src/modelcheck/by1ssm.py（对官方 NemotronHMamba2Mixer，
   两种尺寸 + 三条反例）。 */
static void ssm(float *o, const float *x, int n, int d,
                int nh, int hd, int ss, int ng, int ck,
                const float *win, const float *cw, const float *cb,
                const float *dtb, const float *alog, const float *Dw,
                const float *nw, const float *wout, float neps) {
    int di = nh * hd, cd = di + 2 * ng * ss, rep = nh / ng;
    int proj = di + cd + nh;
    int gs = di / ng;
    float *pr = (float *)malloc(sizeof(float) * n * proj);
    float *cv = (float *)malloc(sizeof(float) * n * cd);
    float *st = (float *)calloc((size_t)nh * hd * ss, sizeof(float));
    float *y = (float *)malloc(sizeof(float) * n * di);

    linear(pr, x, win, n, d, proj);

    /* ② 因果深度卷积 + silu：out[t] = sum_i in[t+i-(k-1)] * cw[:, i] */
    for (int t = 0; t < n; t++)
        for (int c = 0; c < cd; c++) {
            float s = cb ? cb[c] : 0.f;
            for (int i = 0; i < ck; i++) {
                int u = t + i - (ck - 1);
                if (u >= 0)
                    s += pr[(size_t)u * proj + di + c] * cw[(size_t)c * ck + i];
            }
            cv[(size_t)t * cd + c] = s / (1.f + expf(-s));
        }

    /* ③ SSM 递推 */
    for (int t = 0; t < n; t++) {
        const float *cvt = cv + (size_t)t * cd;
        const float *prt = pr + (size_t)t * proj;
        const float *Bv = cvt + di;
        const float *Cv = cvt + di + ng * ss;
        float *yt = y + (size_t)t * di;
        for (int h = 0; h < nh; h++) {
            int g = h / rep;
            float dt = prt[di + cd + h] + dtb[h];
            dt = logf(1.f + expf(dt));              /* softplus */
            float dec = expf(dt * (-expf(alog[h])));
            float *hh = st + (size_t)h * hd * ss;
            for (int i = 0; i < hd; i++) {
                float xi = dt * cvt[h * hd + i];
                float *row = hh + (size_t)i * ss;
                for (int j = 0; j < ss; j++)
                    row[j] = dec * row[j] + xi * Bv[g * ss + j];
            }
            for (int i = 0; i < hd; i++) {
                const float *row = hh + (size_t)i * ss;
                float s = 0.f;
                for (int j = 0; j < ss; j++) s += row[j] * Cv[g * ss + j];
                yt[h * hd + i] = s + Dw[h] * cvt[h * hd + i];
            }
        }
    }

    /* ④ 按组 gated RMSNorm：**gate 乘在 norm 之前**，且是 silu */
    for (int t = 0; t < n; t++) {
        float *yt = y + (size_t)t * di;
        const float *prt = pr + (size_t)t * proj;
        for (int q = 0; q < ng; q++) {
            float vv = 0.f;
            for (int i = 0; i < gs; i++) {
                float gv = prt[q * gs + i];
                gv = gv / (1.f + expf(-gv));
                yt[q * gs + i] *= gv;
                vv += yt[q * gs + i] * yt[q * gs + i];
            }
            vv = 1.f / sqrtf(vv / gs + neps);
            for (int i = 0; i < gs; i++)
                yt[q * gs + i] *= vv * nw[q * gs + i];
        }
    }

    {
        float *tmp = (float *)malloc(sizeof(float) * n * di);
        memcpy(tmp, y, sizeof(float) * n * di);
        linear(o, tmp, wout, n, di, d);
        free(tmp);
    }
    free(pr); free(cv); free(st); free(y);
}


/* KDA（Kimi Delta Attention）：短卷积 + **逐维**衰减门 + delta 规则 + 输出门。

   和 GDN 是**同一个递推**，差别全在门和投影上：

     · q / k / v 是**分开**的投影，各自一条深度卷积
     · `dt_bias` 按**维**（nv * dk），不是按头 —— 所以 g 的形状是 [nv, dk]
     · 多一个输出门 g_proj，喂给最后的 RMSNorm
     · 那个 RMSNorm 的激活是 **sigmoid**（不是 GDN 的 silu）
     · 门可以低秩两层（`f_a/f_b`、`g_a/g_b`），也可以一层（`f_proj`）

   两处最容易写反，而**写反了不会报错、只会算错**：
     · `g` 的秩：**每维一个**（KDA），不是每头一个（GDN）
     · `dt_bias` 的长度：`nv * dk`，不是 `nv`
   还有一处**不报错但会算错**：低秩的两层之间**没有激活**
   （fla 是 `nn.Sequential(Linear, Linear)`）。

   GVA：`nk < nv` 时 q/k 按 `rep = nv / nk` 复用（`fla/ops/kda/naive.py:52-53`）。

   和这个文件里别的算子一样：**照 `by1exec.op_kda` 的数值路径写** ——
   NumPy 那一侧是"另一个后端"，不是"另一份实现"，
   而它自己已经对着 fla 的 `naive_recurrent_kda` 判过
   （`src/modelcheck/by1kda.py`，四个正例 + 八条反例）。

   判卷人：`by1all --tier 5` 的 C 后端那一段（C ↔ NumPy 逐位对拍）。 */
static void kda(float *o, const float *x, int n, int d,
                int nk, int nv, int dk, int dv, int ck, int rank,
                int has_lower, float lower, float l2eps, float neps,
                const float *wq, const float *wk, const float *wv,
                const float *cwq, const float *cwk, const float *cwv,
                const float *alog, const float *dtb, const float *bproj,
                const float *fa, const float *fb,
                const float *ga, const float *gb, const float *gbb,
                const float *nw, const float *wout) {
    int kd = nk * dk, vd = nv * dv, gd = nv * dk, rep = nv / nk;
    float qscale = 1.f / sqrtf((float)dk);
    float *pq = (float *)malloc(sizeof(float) * (size_t)n * kd);
    float *pk = (float *)malloc(sizeof(float) * (size_t)n * kd);
    float *pv = (float *)malloc(sizeof(float) * (size_t)n * vd);
    float *q = (float *)malloc(sizeof(float) * (size_t)n * nk * dk);
    float *kk = (float *)malloc(sizeof(float) * (size_t)n * nk * dk);
    float *v = (float *)malloc(sizeof(float) * (size_t)n * nv * dv);
    float *fg = (float *)malloc(sizeof(float) * (size_t)n * gd);
    float *gt = (float *)malloc(sizeof(float) * (size_t)n * vd);
    float *be = (float *)malloc(sizeof(float) * (size_t)n * nv);
    float *st = (float *)calloc((size_t)nv * dk * dv, sizeof(float));
    float *y = (float *)malloc(sizeof(float) * (size_t)n * nv * dv);

    linear(pq, x, wq, n, d, kd);
    linear(pk, x, wk, n, d, kd);
    linear(pv, x, wv, n, d, vd);

    /* ① q / k 各自的深度卷积 + silu，然后**逐头**做 L2 归一化；
          q 再乘 dk^-0.5（和 naive_recurrent_kda 里的 scale 是同一个数）。
          **卷积读的是投影结果 pq/pk，写的是 q/kk —— 不能就地写**：
          卷积的窗口会读到已经 silu 过的值。 */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nk; h++) {
            float *qr = q + ((size_t)t * nk + h) * dk;
            float *kr = kk + ((size_t)t * nk + h) * dk;
            for (int i = 0; i < dk; i++) {
                int c = h * dk + i;
                float sq = 0.f, sk = 0.f;
                for (int j = 0; j < ck; j++) {
                    int u = t + j - (ck - 1);
                    if (u >= 0) {
                        sq += pq[(size_t)u * kd + c] * cwq[(size_t)c * ck + j];
                        sk += pk[(size_t)u * kd + c] * cwk[(size_t)c * ck + j];
                    }
                }
                qr[i] = sq / (1.f + expf(-sq));
                kr[i] = sk / (1.f + expf(-sk));
            }
            float nq = 0.f, nk2 = 0.f;
            for (int i = 0; i < dk; i++) { nq += qr[i] * qr[i]; nk2 += kr[i] * kr[i]; }
            nq = 1.f / sqrtf(nq + l2eps);
            nk2 = 1.f / sqrtf(nk2 + l2eps);
            for (int i = 0; i < dk; i++) { qr[i] *= nq * qscale; kr[i] *= nk2; }
        }

    /* ② v 的深度卷积 + silu（同样换一个缓冲写） */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nv; h++)
            for (int i = 0; i < dv; i++) {
                int c = h * dv + i;
                float s = 0.f;
                for (int j = 0; j < ck; j++) {
                    int u = t + j - (ck - 1);
                    if (u >= 0) s += pv[(size_t)u * vd + c] * cwv[(size_t)c * ck + j];
                }
                v[(size_t)t * vd + c] = s / (1.f + expf(-s));
            }

    /* ③ 衰减门的**输入**：两层（f_a/f_b）或一层（f_proj）——
          **两层之间没有激活**（fla 是 nn.Sequential）。 */
    if (fb) {
        float *tmp = (float *)malloc(sizeof(float) * (size_t)n * rank);
        linear(tmp, x, fa, n, d, rank);
        linear(fg, tmp, fb, n, rank, gd);
        free(tmp);
    } else {
        linear(fg, x, fa, n, d, gd);
    }

    /* ④ 输出门：同样两层或一层，然后**乘 sigmoid**。
          `fla/layers/kda.py:191` 的 `activation="sigmoid"`，
          展开在 `fused_norm_gate.py:104`：`y = y * sigmoid(g)`。 */
    if (gb) {
        float *tmp = (float *)malloc(sizeof(float) * (size_t)n * rank);
        linear(tmp, x, ga, n, d, rank);
        linear_b(gt, tmp, gb, gbb, n, rank, vd);
        free(tmp);
    } else {
        linear(gt, x, ga, n, d, vd);
    }
    for (size_t i = 0; i < (size_t)n * vd; i++)
        gt[i] = 1.f / (1.f + expf(-gt[i]));

    /* ⑤ beta（`use_beta_sigmoid_in_kernel`，所以核里拿到的是概率） */
    linear(be, x, bproj, n, d, nv);
    for (size_t i = 0; i < (size_t)n * nv; i++)
        be[i] = 1.f / (1.f + expf(-be[i]));

    /* ⑥ delta 规则递推。状态是 [nv, dk, dv]，与序列长度无关。 */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nv; h++) {
            int kh = h / rep;                     /* GVA：q/k 复用 */
            const float *qt = q + ((size_t)t * nk + kh) * dk;
            const float *kt = kk + ((size_t)t * nk + kh) * dk;
            const float *vt = v + ((size_t)t * nv + h) * dv;
            const float *fgt = fg + (size_t)t * gd + h * dk;
            float bt = be[(size_t)t * nv + h];
            float *hh = st + (size_t)h * dk * dv;
            float *yt = y + ((size_t)t * nv + h) * dv;
            for (int i = 0; i < dk; i++) {
                float g = fgt[i] + dtb[h * dk + i];
                if (has_lower)
                    g = lower / (1.f + expf(-(expf(alog[h]) * g)));
                else
                    g = -expf(alog[h]) * logf(1.f + expf(g));
                float dd = expf(g);
                float *row = hh + (size_t)i * dv;
                for (int j = 0; j < dv; j++) row[j] *= dd;
            }
            for (int j = 0; j < dv; j++) {
                float s = 0.f;
                for (int i = 0; i < dk; i++) s += hh[(size_t)i * dv + j] * kt[i];
                yt[j] = (vt[j] - s) * bt;          /* 先当 delta 用 */
            }
            for (int i = 0; i < dk; i++) {
                float ki = kt[i];
                float *row = hh + (size_t)i * dv;
                for (int j = 0; j < dv; j++) row[j] += ki * yt[j];
            }
            for (int j = 0; j < dv; j++) {
                float s = 0.f;
                for (int i = 0; i < dk; i++) s += hh[(size_t)i * dv + j] * qt[i];
                yt[j] = s;
            }
        }

    /* ⑦ 逐头 RMSNorm，再乘输出门，最后 o_proj */
    {
        float *tmp = (float *)malloc(sizeof(float) * (size_t)n * vd);
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nv; h++) {
                const float *yt = y + ((size_t)t * nv + h) * dv;
                const float *gtw = gt + (size_t)t * vd + h * dv;
                float *ot = tmp + (size_t)t * vd + h * dv;
                float vv = 0.f;
                for (int i = 0; i < dv; i++) vv += yt[i] * yt[i];
                vv = 1.f / sqrtf(vv / dv + neps);
                for (int i = 0; i < dv; i++)
                    ot[i] = (nw[i] * yt[i] * vv) * gtw[i];
            }
        linear(o, tmp, wout, n, vd, d);
        free(tmp);
    }
    free(pq); free(pk); free(pv); free(q); free(kk); free(v);
    free(fg); free(gt); free(be); free(st); free(y);
}


/* q_gate：wq 的输出是 2 倍，前半是 query，后半是门 */
static void split_qg(float *q, float *gt, const float *qf, int n, int nh, int hd) {
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nh; h++) {
            memcpy(q + ((size_t)t * nh + h) * hd,
                   qf + ((size_t)t * nh + h) * 2 * hd, sizeof(float) * hd);
            memcpy(gt + ((size_t)t * nh + h) * hd,
                   qf + ((size_t)t * nh + h) * 2 * hd + hd, sizeof(float) * hd);
        }
}
static void apply_qg(float *o, const float *gt, int n, int nq) {
    for (size_t i = 0; i < (size_t)n * nq; i++)
        o[i] *= 1.f / (1.f + expf(-gt[i]));
}

/* qk_norm：按头做 RMSNorm，在 RoPE 之前。
 *
 * **eps 由调用方传进来**，不许写死 —— 否则与
 * `RMS_EPS` 这个宏上面已经按 `ir["norm_eps"]` 生成好了，
 * 也就是说**同一个文件里有两套 eps**：一处读 IR、一处写死。
 * clef-tiny 的 qk_norm 是 per_head、norm_eps 是 1e-6，
 * 于是 C 和 by1codegen 对同一份 IR 给出不同的 q/k。
 * 相对差只有 1.05e-05 —— **低于 1e-4 的判据，所以三个后端互拍是绿的。 */
static void qk_norm(float *q, float *k, const float *qn, const float *kn,
                    int n, int nh, int nkv, int hd, int one_plus, float eps) {
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < nh; h++) {
            float *v = q + ((size_t)t * nh + h) * hd;
            float s = 0.f;
            for (int i = 0; i < hd; i++) s += v[i] * v[i];
            s = 1.f / sqrtf(s / hd + eps);
            for (int i = 0; i < hd; i++) v[i] *= s * (one_plus ? (1.f + qn[i]) : qn[i]);
        }
        for (int h = 0; h < nkv; h++) {
            float *v = k + ((size_t)t * nkv + h) * hd;
            float s = 0.f;
            for (int i = 0; i < hd; i++) s += v[i] * v[i];
            s = 1.f / sqrtf(s / hd + eps);
            for (int i = 0; i < hd; i++) v[i] *= s * (one_plus ? (1.f + kn[i]) : kn[i]);
        }
    }
}


/* mla() 会用到 attention()，而它定义在后面 —— 前置声明。
   最后那个 `keep` 是稀疏掩码（NULL = 全要），SparseMLA 用。 */
static void attention(float *o, float *q, const float *k, const float *v,
                      int n, int nh, int nkv, int hd, int vd, int window,
                      const float *sink, const unsigned char *keep);
static void dsa_index(unsigned char *keep, const float *x, const float *qres,
                      int n, int d, int ql, int nh, int hd, int nr, int topk,
                      const float *wqb, const float *wk, const float *knw,
                      const float *knb, const float *wp,
                      float base, int interleaved,
                      int ix_kpool, int ix_tail,
                      const float *ix_kgate, const float *ix_kape);

/* 把 [rows, len] 的每一行做 RoPE。MLA 的 rope 只作用在解耦出来的那一段。 */
static void rope_rows(float *x, int rows, int per, int len, float base,
                      int interleaved) {
    /* per = 每个位置占几行：q 是 [n, nh, nr] 所以 per=nh；k_rot 单头 per=1。
       位置是**行号除以 per**，不是行号本身。
       频率用 double 算 —— 主 rope() 也是这么做的，用 float 会差到 1e-4。 */
    int half = len / 2;
    double *inv = (double *)malloc(sizeof(double) * half);
    for (int i = 0; i < half; i++)
        inv[i] = 1.0 / pow((double)base, 2.0 * (double)i / (double)len);
    for (int r = 0; r < rows; r++) {
        float *v = x + (size_t)r * len;
        float *tmp = (float *)malloc(sizeof(float) * len);
        int pos = r / per;
        for (int i = 0; i < half; i++) {
            double ang = (double)pos * inv[i];
            float ci = (float)cos(ang), si = (float)sin(ang);
            /* interleaved：奇偶成对；否则前半配后半。两种都是串接着写回。 */
            float x0 = interleaved ? v[2 * i] : v[i];
            float x1 = interleaved ? v[2 * i + 1] : v[i + half];
            tmp[i] = x0 * ci - x1 * si;
            tmp[half + i] = x0 * si + x1 * ci;
        }
        memcpy(v, tmp, sizeof(float) * len);
        free(tmp);
    }
    free(inv);
}

/* MLA：低秩压缩的注意力。k_rot 是**单头**的，转完广播到所有头。 */
static void mla(float *o, const float *x, int n, int d,
                int nh, int ql, int kl, int nope, int nr, int vd,
                const float *wqa, const float *nqa, const float *wqb,
                const float *wkva, const float *nkva, const float *wkvb,
                const float *wd, const float *wg,
                float neps, int one_plus, float base, int interleaved,
                int sigmoid_gate,
                int ix_nh, int ix_hd, int ix_topk,
                const float *ix_wqb, const float *ix_wk, const float *ix_knw,
                const float *ix_knb, const float *ix_wp,
                int ix_kpool, int ix_tail,
                const float *ix_kgate, const float *ix_kape) {
    int qk = nope + nr;
    float *qa  = (float *)malloc(sizeof(float) * n * ql);
    float *qb  = (float *)malloc(sizeof(float) * n * nh * qk);
    float *kva = (float *)malloc(sizeof(float) * n * (kl + nr));
    float *ckv = (float *)malloc(sizeof(float) * n * kl);
    float *kb  = (float *)malloc(sizeof(float) * n * nh * (nope + vd));
    float *qr  = (float *)malloc(sizeof(float) * n * nh * nr);   /* 每个头一个 rope 段 */
    float *kr  = (float *)malloc(sizeof(float) * n * nr);
    float *qq  = (float *)malloc(sizeof(float) * n * nh * qk);
    float *kk  = (float *)malloc(sizeof(float) * n * nh * qk);
    float *vv  = (float *)malloc(sizeof(float) * n * nh * vd);

    linear(qa, x, wqa, n, d, ql);
    rmsnorm(qa, qa, nqa, n, ql, one_plus, neps);
    linear(qb, qa, wqb, n, ql, nh * qk);

    linear(kva, x, wkva, n, d, kl + nr);
    for (int t = 0; t < n; t++) {
        const float *r = kva + (size_t)t * (kl + nr);
        memcpy(ckv + (size_t)t * kl, r, sizeof(float) * kl);
        memcpy(kr + (size_t)t * nr, r + kl, sizeof(float) * nr);
    }
    rmsnorm(ckv, ckv, nkva, n, kl, one_plus, neps);
    linear(kb, ckv, wkvb, n, kl, nh * (nope + vd));

    /* 抠出 q 的 rope 段（k_rot 本来就是单头，不用抠） */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nh; h++)
            memcpy(qr + ((size_t)t * nh + h) * nr,
                   qb + ((size_t)t * nh + h) * qk + nope, sizeof(float) * nr);
    rope_rows(qr, n * nh, nh, nr, base, interleaved);
    rope_rows(kr, n, 1, nr, base, interleaved);

    /* 拼 q=[nope|rope]、k=[nope|rope]，k_rot 广播到所有头 */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nh; h++) {
            const float *pb = qb + ((size_t)t * nh + h) * qk;
            float *qd = qq + ((size_t)t * nh + h) * qk;
            memcpy(qd, pb, sizeof(float) * nope);
            memcpy(qd + nope, qr + ((size_t)t * nh + h) * nr, sizeof(float) * nr);
            const float *kb2 = kb + ((size_t)t * nh + h) * (nope + vd);
            float *kd = kk + ((size_t)t * nh + h) * qk;
            memcpy(kd, kb2, sizeof(float) * nope);
            memcpy(kd + nope, kr + (size_t)t * nr, sizeof(float) * nr);
            memcpy(vv + ((size_t)t * nh + h) * vd, kb2 + nope,
                   sizeof(float) * vd);
        }

    /* SparseMLA：先挑 top-k 个 key，只在挑中的那些上算注意力。
       `qa` 正是索引器要的 q_resid（q_a_layernorm(q_a_proj(x))）。 */
    unsigned char *keep = NULL;
    if (ix_nh > 0) {
        keep = (unsigned char *)malloc((size_t)n * n);
        dsa_index(keep, x, qa, n, d, ql, ix_nh, ix_hd, nr, ix_topk,
                  ix_wqb, ix_wk, ix_knw, ix_knb, ix_wp, base, interleaved,
                  ix_kpool, ix_tail, ix_kgate, ix_kape);
    }
    attention(o, qq, kk, vv, n, nh, nh, qk, vd, 0, NULL, keep);
    free(keep);

    if (wg) {
        float *g = (float *)malloc(sizeof(float) * n * nh);
        linear(g, x, wg, n, d, nh);
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nh; h++) {
                float z = g[(size_t)t * nh + h];
                float gv = sigmoid_gate ? 1.f / (1.f + expf(-z)) : log1pf(expf(z));
                for (int c = 0; c < vd; c++)
                    o[(size_t)t * nh * vd + h * vd + c] *= gv;
            }
        free(g);
    }

    {
        float *tmp = (float *)malloc(sizeof(float) * n * nh * vd);
        memcpy(tmp, o, sizeof(float) * n * nh * vd);
        linear(o, tmp, wd, n, nh * vd, d);
        free(tmp);
    }
    free(qa); free(qb); free(kva); free(ckv); free(kb);
    free(qr); free(kr); free(qq); free(kk); free(vv);
}

/* hd 是 Q/K 的头宽，vd 是 V 的头宽。
   **两者不一定相同** —— MLA 就是 qk_head=48 而 v_dim=32。
   原来只有一个 hd，于是 V 被按 Q/K 的步长读，堆越界（0xC0000005）。 */
static void attention(float *o, float *q, const float *k, const float *v,
                      int n, int nh, int nkv, int hd, int vd, int window,
                      const float *sink, const unsigned char *keep) {
    int rep = nh / nkv;
    float sc = 1.0f / sqrtf((float)hd);
    float *att = (float *)malloc(sizeof(float) * n);
    float *vb = (float *)malloc(sizeof(float) * vd);
    /* keep：稀疏掩码，[n*n]，0 = 这个 key 不要。NULL = 全要。
       和 window 用的是同一条"跳过"路径，所以三条循环都改到了 ——
       少改一条就是"算了一半"，而且看起来还是对的。 */
    for (int h = 0; h < nh; h++) {
        int kh = h / rep;
        for (int i = 0; i < n; i++) {
            const float *qi = q + ((size_t)i * nh + h) * hd;
            float mx = -INFINITY;
            for (int j = 0; j <= i; j++) {
                if ((window && j <= i - window) ||
                    (keep && !keep[(size_t)i * n + j])) { att[j] = 0.f; continue; }
                const float *kj = k + ((size_t)j * nkv + kh) * hd;
                float s = 0.f;
                for (int t = 0; t < hd; t++) s += qi[t] * kj[t];
                s *= sc;
                att[j] = s;
                if (s > mx) mx = s;
            }
            if (sink && sink[h] > mx) mx = sink[h];
            float sum = 0.f;
            for (int j = 0; j <= i; j++) {
                if ((window && j <= i - window) ||
                    (keep && !keep[(size_t)i * n + j])) continue;
                att[j] = expf(att[j] - mx);
                sum += att[j];
            }
            /* sink：一列额外的 logit 参与 softmax，之后丢掉 ——
               剩下的概率和小于 1，它专门用来吸收质量 */
            if (sink) sum += expf(sink[h] - mx);
            float *oi = o + ((size_t)i * nh + h) * vd;
            for (int t = 0; t < vd; t++) vb[t] = 0.f;
            for (int j = 0; j <= i; j++) {
                if ((window && j <= i - window) ||
                    (keep && !keep[(size_t)i * n + j])) continue;
                float a = att[j] / sum;
                const float *vj = v + ((size_t)j * nkv + kh) * vd;
                for (int t = 0; t < vd; t++) vb[t] += a * vj[t];
            }
            memcpy(oi, vb, sizeof(float) * vd);
        }
    }
    free(att);
    free(vb);
}

/* DSA 稀疏索引器：给每个 query 挑 top-k 个 key，写进 keep[n*n]。
   **和 `by1codegen.DSAIndexer`（PyTorch）/ `by1exec._dsa_index`（NumPy）
   逐行对应**；判卷人是 transformers 的 `GlmMoeDsaIndexer`（by1sparse.py）。

   两处容易抄错：
   ① rope 那一半**在前**（`[qk_rope | hd - qk_rope]`），和主 MLA 的
      `[nope | rope]` 相反。
   ② 索引器的 rope 配对约定**固定 interleaved**，和主 MLA 的 pairing 无关。

   并列取**较小的 key 下标**（下面那个插入排序用严格 `>`，所以是稳定的）。
   —— 实测：并列处挑谁没有定义，两个后端会挑出不同的 key，
   而那会让输出当场差 1e-1。见 history/ir.md 129。 */
static void dsa_index(unsigned char *keep, const float *x, const float *qres,
                      int n, int d, int ql, int nh, int hd, int nr, int topk,
                      const float *wqb, const float *wk, const float *knw,
                      const float *knb, const float *wp,
                      float base, int interleaved,
                      int ix_kpool, int ix_tail,
                      const float *ix_kgate, const float *ix_kape) {
    float *q = (float *)malloc(sizeof(float) * (size_t)n * nh * hd);
    float *k0 = (float *)malloc(sizeof(float) * (size_t)n * hd);
    float *w = (float *)malloc(sizeof(float) * (size_t)n * nh);
    float *sc = (float *)malloc(sizeof(float) * n);
    int *ord = (int *)malloc(sizeof(int) * n);

    linear(q, qres, wqb, n, ql, nh * hd);
    linear(k0, x, wk, n, d, hd);
    /* LayerNorm（带 bias，eps=1e-6）—— 参考就是这么写的，不是 RMSNorm */
    for (int t = 0; t < n; t++) {
        float *r = k0 + (size_t)t * hd;
        float mu = 0.f;
        for (int c = 0; c < hd; c++) mu += r[c];
        mu /= (float)hd;
        float var = 0.f;
        for (int c = 0; c < hd; c++) { float z = r[c] - mu; var += z * z; }
        var /= (float)hd;
        float inv = 1.f / sqrtf(var + 1e-6f);
        for (int c = 0; c < hd; c++) r[c] = (r[c] - mu) * inv * knw[c] + knb[c];
    }
    /* rope 只作用在**前 nr 维**（索引器里 rope 段在前，和主 MLA 相反）。
       `rope_rows` 按"整行"转，所以先把那一段抠出来单独转，再放回去。 */
    {
        float *qrot = (float *)malloc(sizeof(float) * (size_t)n * nh * nr);
        float *krot = (float *)malloc(sizeof(float) * (size_t)n * nr);
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nh; h++)
                memcpy(qrot + ((size_t)t * nh + h) * nr,
                       q + ((size_t)t * nh + h) * hd, sizeof(float) * nr);
        for (int t = 0; t < n; t++)
            memcpy(krot + (size_t)t * nr, k0 + (size_t)t * hd, sizeof(float) * nr);
        rope_rows(qrot, (size_t)n * nh, nh, nr, base, interleaved);
        rope_rows(krot, n, 1, nr, base, interleaved);
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nh; h++)
                memcpy(q + ((size_t)t * nh + h) * hd,
                       qrot + ((size_t)t * nh + h) * nr, sizeof(float) * nr);
        for (int t = 0; t < n; t++)
            memcpy(k0 + (size_t)t * hd, krot + (size_t)t * nr, sizeof(float) * nr);
        free(qrot); free(krot);
    }
    {
        float *tmp = (float *)malloc(sizeof(float) * (size_t)n * nh);
        linear(tmp, x, wp, n, d, nh);
        for (int i = 0; i < n * nh; i++) w[i] = tmp[i] * powf((float)nh, -0.5f);
        free(tmp);
    }
    if (ix_kpool > 1) {
        /* ── k-pool：**池**打分，选中的池摊回原始 token ──────────────
           逐行对着 Glm5NextTextIndexer（refs/glm5-next-indexer.reference.txt）。
           两处容易漏：
             ① 池从**第一个有效 token** 起算（这里 n 个全有效，所以是 0）；
             ② 池可选当且仅当**它最后一个 token 对 query 可见**，
                且**整个池都有效**。
           `ix_kgate` [hd, d] 管池内加权、`ix_kape` [P, hd] 管池内位置。 */
        int P = ix_kpool;
        float *gate = (float *)malloc(sizeof(float) * (size_t)n * hd);
        linear(gate, x, ix_kgate, n, d, hd);
        int np_ = (n + P - 1) / P;
        float *pk = (float *)malloc(sizeof(float) * (size_t)np_ * hd);
        /* ⚠ **`_ape` 只影响"挑中哪些池"，而输出只看挑选结果。**
           实测：把 `+ ix_kape[...]` **整个去掉**，`logits.bin`
           **逐字节不变**（差 0.0）；而乘 1000 就变（差 1.7e-01）。
           量了挑选掩码才看清：丢掉 `_ape` -> **0 格不同**、×1000 -> **32 格不同**。
           所以**判 `_ape` 不能看输出，要看挑选集合**（`by1sparse` 就是这么判的）。
           这里用 double 累是为了和 NumPy 那边同精度，不是为了修这个。 */
        double *lg = (double *)malloc(sizeof(double) * P * hd);
        double *pb = (double *)malloc(sizeof(double) * P * hd);
        for (int p = 0; p < np_; p++) {
            int lo = p * P;
            for (int j = 0; j < P; j++) {
                int id = lo + j;
                for (int c = 0; c < hd; c++)
                    lg[j * hd + c] = (id < n)
                        ? (double)gate[(size_t)id * hd + c]
                          + (double)ix_kape[j * hd + c]
                        : -INFINITY;
            }
            /* softmax 沿池内那一维（每个通道各一次） */
            for (int c = 0; c < hd; c++) {
                double mx = -INFINITY;
                for (int j = 0; j < P; j++) if (lg[j * hd + c] > mx) mx = lg[j * hd + c];
                if (mx == -INFINITY) { for (int j = 0; j < P; j++) pb[j * hd + c] = 0.0; continue; }
                double s = 0.0;
                for (int j = 0; j < P; j++) { double e = exp(lg[j * hd + c] - mx); pb[j * hd + c] = e; s += e; }
                for (int j = 0; j < P; j++) pb[j * hd + c] = pb[j * hd + c] / s;
            }
            for (int c = 0; c < hd; c++) {
                float acc = 0.f;
                for (int j = 0; j < P; j++) {
                    int id = lo + j;
                    if (id < n) acc += pb[j * hd + c] * k0[(size_t)id * hd + c];
                }
                pk[(size_t)p * hd + c] = acc;
            }
        }
        /* 每个 query：给池打分 -> 并列取较小下标 -> 摊回 token */
        float *psc = (float *)malloc(sizeof(float) * np_);
        int *pord = (int *)malloc(sizeof(int) * np_);
        for (int t = 0; t < n; t++) {
            for (int p = 0; p < np_; p++) {
                float acc = 0.f;
                for (int h = 0; h < nh; h++) {
                    const float *qi = q + ((size_t)t * nh + h) * hd;
                    const float *ks = pk + (size_t)p * hd;
                    float dot = 0.f;
                    for (int c = 0; c < hd; c++) dot += qi[c] * ks[c];
                    dot *= powf((float)hd, -0.5f);
                    if (dot < 0.f) dot = 0.f;
                    acc += w[(size_t)t * nh + h] * dot;
                }
                /* 池可选：**整池有效**（所有槽都在范围内）且**末槽对 t 可见**。
                   末槽 = p*P + P-1，clamp 到 n-1。 */
                int full = ((p + 1) * P) <= n;
                int pend = p * P + P - 1; if (pend > n - 1) pend = n - 1;
                psc[p] = (full && pend <= t) ? acc : -INFINITY;
            }
            for (int p = 0; p < np_; p++) pord[p] = p;
            for (int a = 1; a < np_; a++) {
                int v = pord[a], b = a - 1;
                while (b >= 0 && psc[pord[b]] < psc[v]) { pord[b + 1] = pord[b]; b--; }
                pord[b + 1] = v;
            }
            int sk = topk / P; if (sk > np_) sk = np_; if (sk < 0) sk = 0;
            for (int s = 0; s < n; s++) keep[(size_t)t * n + s] = 0;
            for (int z = 0; z < sk; z++) {
                int p = pord[z];
                for (int j = 0; j < P; j++) {
                    int id = p * P + j;
                    if (id < n && id <= t) keep[(size_t)t * n + id] = 1;
                }
            }
            if (ix_tail && P > 1) {
                int vis = t + 1;                 /* 可见 key 数（因果、无 padding） */
                int tc = vis % P;
                int start = vis - tc;
                for (int j = 0; j < P - 1; j++) {
                    int id = start + j;
                    if (j < tc && id < n && id <= t) keep[(size_t)t * n + id] = 1;
                }
            }
        }
        free(gate); free(pk); free(lg); free(pb); free(psc); free(pord);
        free(q); free(k0); free(w); free(sc); free(ord);
        return;
    }
    for (int t = 0; t < n; t++) {
        for (int s = 0; s < n; s++) sc[s] = -INFINITY;
        for (int s = 0; s <= t; s++) {
            float acc = 0.f;
            for (int h = 0; h < nh; h++) {
                const float *qi = q + ((size_t)t * nh + h) * hd;
                const float *ks = k0 + (size_t)s * hd;
                float dot = 0.f;
                for (int c = 0; c < hd; c++) dot += qi[c] * ks[c];
                dot *= powf((float)hd, -0.5f);
                if (dot < 0.f) dot = 0.f;             /* ReLU */
                acc += w[(size_t)t * nh + h] * dot;
            }
            sc[s] = acc;
        }
        for (int s = 0; s < n; s++) ord[s] = s;
        /* 稳定插入排序：严格 `>`，相等时保持原序 -> 取较小下标 */
        for (int a = 1; a < n; a++) {
            int v = ord[a], b = a - 1;
            while (b >= 0 && sc[ord[b]] < sc[v]) { ord[b + 1] = ord[b]; b--; }
            ord[b + 1] = v;
        }
        int take = topk < n ? topk : n;
        for (int s = 0; s < n; s++) keep[(size_t)t * n + s] = 0;
        for (int s = 0; s < take; s++) keep[(size_t)t * n + ord[s]] = 1;
    }
    free(q); free(k0); free(w); free(sc); free(ord);
}
'''

C_MAIN = r'''
int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    FILE *f = fopen(argv[2], "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    W = (float *)malloc(sz);
    if (fread(W, 1, sz, f) != (size_t)sz) { fprintf(stderr, "weights read fail\n"); return 1; }
    fclose(f);

    int *ids = (int *)malloc(sizeof(int) * n);
    f = fopen(argv[3], "rb");
    if (fread(ids, sizeof(int), n, f) != (size_t)n) { fprintf(stderr, "ids read fail\n"); return 1; }
    fclose(f);

    const int D = D_MODEL, V = VOCAB;
    float *x  = (float *)calloc((size_t)n * D, sizeof(float));
    float *xn = (float *)calloc((size_t)n * D, sizeof(float));
    float *y  = (float *)calloc((size_t)n * D, sizeof(float));
    float *q  = (float *)calloc((size_t)n * Q_DIM, sizeof(float));
    float *kk = (float *)calloc((size_t)n * KV_DIM, sizeof(float));
    float *vv = (float *)calloc((size_t)n * KV_DIM, sizeof(float));
    float *ob = (float *)calloc((size_t)n * OB_MAX, sizeof(float));   /* FFN 也会用到它，得按最宽的那个分配 */
    float *hb = (float *)calloc((size_t)n * HID_MAX, sizeof(float));   /* FFN 中间层比 D 宽 */
    float *qf = (float *)calloc((size_t)n * 2 * OB_MAX, sizeof(float));   /* q_gate 时 wq 输出 2 倍 */
    float *gt = (float *)calloc((size_t)n * OB_MAX, sizeof(float));
    /* 每个算子一个输出缓冲 —— 严格对应 IR 的 ValueRef，不做别名优化 */
    float *tb[64];
    for (int z = 0; z < N_OPS; z++) tb[z] = (float *)calloc((size_t)n * D, sizeof(float));
    float *t0 = tb[0], *t1 = tb[1], *t2 = tb[2], *t3 = tb[3], *t4 = tb[4], *t5 = tb[5];

    for (int i = 0; i < n; i++)
        memcpy(x + (size_t)i * D, P(EMBED) + (size_t)ids[i] * D, sizeof(float) * D);
    /* **学习式位置表：整张加在嵌入上。**
       以前这一段不存在 —— 于是 GPT-2 的位置信息整个丢了，
       而输出形状完全正常。 */
    /* **用 `#if` 不用 `if`。**
       写成运行时的 `if (POS_LEARNED)` 时，那个分支**照样要编译** ——
       而 `P(POS)` 展开成 `OFF_POS`，在不需要位置表的模型里
       这个宏根本不存在，于是编译失败。
       **一个"运行时判断"会让死分支活到编译期。** */
#if POS_LEARNED
    for (int i = 0; i < n; i++)
        for (int j = 0; j < D; j++)
            x[(size_t)i * D + j] += P(POS)[(size_t)i * D + j];
#endif

    for (int L = 0; L < N_LAYER; L++) {
        /* 每个 op 一个作用域：OFF_/LEN_ 宏按层号拼出来 */
        LAYER_BODY(L)
    }

    /* **最终归一化也要按 kind 分派。** */
#if NORM_KIND_LAYER
    layernorm(xn, x, P(FINAL_NORM), P(FINAL_NORM_B), n, D,
              NORM_ONE_PLUS, RMS_EPS);
#else
    rmsnorm(xn, x, P(FINAL_NORM), n, D, NORM_ONE_PLUS, RMS_EPS);
#endif
    float *logits = (float *)calloc((size_t)n * V, sizeof(float));
    for (int i = 0; i < n; i++) {
        const float *r = xn + (size_t)i * D;
        float *o = logits + (size_t)i * V;
        for (int j = 0; j < V; j++) {
            const float *wr = P(HEAD) + (size_t)j * D;
            float s = 0.f;
            for (int t = 0; t < D; t++) s += r[t] * wr[t];
            o[j] = s;
        }
    }
    f = fopen(argv[4], "wb");
    fwrite(logits, sizeof(float), (size_t)n * V, f);
    fclose(f);
    printf("ok %d x %d\n", n, V);
    return 0;
}
'''


def emit_c_ir(ir, params):
    """**只吃 IR 的入口。** `info` 那个参数在这个后端里**从头到尾没被用过** ——
    它是历史遗留。去掉它，"从 IR 入口跑"就是真的。"""
    import by1ir as _ir
    _errs = _ir.validate(ir)
    if _errs:
        raise SystemExit("IR 不合法：\n  " + "\n  ".join(_errs[:10]))
    return _emit_c_body(ir, params)


def emit_c(ir, info, params):
    """旧入口，保留。**info 没用。**"""
    return _emit_c_body(ir, params)


def _emit_c_body(ir, params):
    """把 IR 展开成 C。**不支持就报错，绝不悄悄生成错的。**"""
    d = ir["d_model"]
    # **LayerNorm 和学习式位置表已实现**，所以下面那两条
    # 「要么实现，要么拒绝」的拒绝没有存在的理由了。
    # 但那段注释留着 —— 它记的是"不实现就该拒绝"这个规矩，
    # 而不是"永远不实现"。
    lines, need = [], {}
    att = next((o["attrs"] for L in ir["layers"] for o in L["ops"]
                if o["kind"] == "Attention"), None)
    mla_att = next((o["attrs"] for L in ir["layers"] for o in L["ops"]
                    if o["kind"] == "MLA"), None)
    if att is None and mla_att is None:
        # **没有注意力的模型也是模型。**
        # 若硬拒绝 —— 则"纯 Linear / 纯外部算子"的模型
        # 根本建不了。而 Mamba 那一类**正是没有注意力的**：
        # 覆盖率缺口里最该能跑的那些，被这一行挡在门外。
        #
        # 改成：没有注意力就不发 Q_DIM / KV_DIM —— 那些宏只在
        # 注意力那一段用得到，没有注意力就没人引用它们。
        att = {"q": 1, "kv": 1, "head_dim": 1}
        _no_att = True
    else:
        _no_att = False
    if att is None:
        # 纯 MLA 模型：Q_DIM / KV_DIM 这些宏按 MLA 的宽度来
        att = {"q": mla_att["q"], "kv": mla_att["q"],
               "head_dim": mla_att["head_dim"]}

    def key(i, j, nm):
        return f"L{i}_OP{j}_{MANGLE.sub('_', nm).upper()}"

    # **能力拒绝**（布局表不管这些 —— 它只说"有哪些张量"）。
    # 和布局分开写，理由：`by1exec` 支持 `head_gate` 而 C 不支持；
    # 把"后端能做到哪一步"混进"有哪些张量"，会让两个后端互相绑死
    # （`by1ggml` 的拒绝分散在 lowering/发射表里，也是同一个道理）。
    #
    # **顺序要紧**：预扫**在布局之前**。布局表对 `head_gate` 是照实
    # 声明的（IR 有那个张量），所以它建得出来；而"C 不支持"必须**先**报
    # ——否则一份 head_gate 模型会以别的方式失败，报错长得像 IR 有问题。
    for L in ir["layers"]:
        for j, o in enumerate(L["ops"]):
            k, a = o["kind"], o["attrs"]
            if k == "Attention" and a["head_gate"] != "off":
                raise SystemExit(
                    f"  [不支持] 第 {L['index']} 层的注意力用了 head_gate"
                    " —— C 后端还没实现")
            if k == "Attention" and str(a.get("qk_norm", "")
                                        ).strip().lower() == "full":
                # 融合核 `qk_norm()` 是按头的 —— full 是拆头前整宽，
                # **不许静默算成 per_head**（NumPy 曾经就这么错着）
                raise SystemExit(
                    f"  [不支持] 第 {L['index']} 层 qk_norm=full —— "
                    "C 融合核的 qk_norm 是按头的")
            if k == "FFN" and a.get("act") == "gptoss":
                raise SystemExit(
                    f"  [不支持] 第 {L['index']} 层的 FFN 用了 gptoss 激活")
            if k == "MoE" and a.get("act", "silu") not in ("silu", "gptoss"):
                # 融合核 `act_gate_` 只有 silu/gptoss 两支 —— 表外的取值
                # **不许静默算成 silu**（NumPy/PyTorch 两边就是这么错的，
                # relu2 的 Nemotron 三后端一致地错；见 by1graph._lower_moe）
                raise SystemExit(
                    f"  [不支持] 第 {L['index']} 层的 MoE 用了 act="
                    f"{a.get('act')!r} —— C 融合核只有 silu / gptoss")
            if k == "MoE" and a.get("act") == "gptoss" and a.get("shared"):
                # 共享专家那条支在融合核里没有 limit_shared / 激活分派
                raise SystemExit(
                    f"  [不支持] 第 {L['index']} 层 gptoss + 共享专家 —— "
                    "融合核的共享支还没有 limit_shared / 激活分派")
            if k not in C_KINDS:
                raise SystemExit(f"  [不支持] C 后端还没有 {k}")

    # **权重布局的唯一定义在 `by1ir.weight_plan`**（1.0.0-rc0 · A3-1）。
    # 以前这张表长在这个函数里，于是 `by1ggml` 为了拿同一份 `w.bin`
    # 布局，只能反过来 import 本模块（`emit_c(...)` 然后 `del decls`）。
    # 布局是 **IR 的事实**，不是某个后端的事实 —— 所以它搬到了 IR 那边。
    import by1ir as _bir
    order = _bir.weight_plan(ir)

    # **外部符号的声明。** ABI 见 by1ext.py —— 编译器不认识机制，
    # 只认识这个签名。所以加一个新机制不用动这个文件。
    _extsyms = []
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] == "External":
                _extsyms.append(o["attrs"])
    for a in _extsyms:
        lines.append("extern void %s(const float *x, float *y, int B, int T,"
                     " int D, const float *const *w, int nw);" % a["symbol"])

    try:
        from by1ver import stamp as _stamp2
    except ImportError:
        def _stamp2():
            return "by1"
    lines.insert(0, "/* %s */" % _stamp2())
    lines.append("#define D_MODEL %d" % d)
    lines.append("#define VOCAB %d" % ir["vocab"])
    lines.append("#define N_LAYER %d" % len(ir["layers"]))
    lines.append("#define Q_DIM %d" % (att["q"] * att["head_dim"]))
    lines.append("#define KV_DIM %d" % (att["kv"] * att["head_dim"]))
    lines.append("#define NORM_ONE_PLUS %d" % (1 if ir.get("norm_one_plus") else 0))
    # **能力开关。** 这两个以前不存在 —— 于是 C 那边"不知道自己不知道"。
    lines.append("#define NORM_KIND_LAYER %d"
                 % (1 if str(ir.get("norm_kind", "rms")).lower() == "layer" else 0))
    lines.append("#define POS_LEARNED %d"
                 % (1 if str(ir.get("pos_kind", "rope")).lower() == "learned" else 0))
    # **IR 里这个字段叫 `norm_eps`，不叫 `rms_eps`。**
    # 原来写的是 `ir.get("rms_eps", 1e-5)` —— 那个名字**在 IR 里
    # 根本不存在**，于是永远回退到 1e-5。
    # 而 clef-tiny / mla-shaped 的 `norm_eps` 是 1e-6。
    # 一个 `.get(不存在的键, 默认值)` 就是这类的经典长相：
    # 它不报错、不警告，只是**永远走默认那条路**。
    lines.append("#define RMS_EPS %sf" % float(ir.get("norm_eps", _DEFAULT_EPS)))
    _hids = [o["attrs"]["hidden"] for L in ir["layers"] for o in L["ops"]
             if o["kind"] in ("FFN", "MoE")]
    lines.append("#define HID_MAX %d" % (max(_hids) if _hids else 1))
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] == "Linear":
                _hids.append(o["attrs"]["v_heads"] * o["attrs"]["v_dim"])
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] == "MLA":
                # kv_b 的输出 nh*(nope+v_dim) 是最宽的中间缓冲
                _hids.append(o["attrs"]["q"] * (o["attrs"]["qk_nope"]
                                                + o["attrs"]["v_dim"]))
    lines.append("#define OB_MAX %d" % max(att["q"] * att["head_dim"], max(_hids) if _hids else 1))
    lines.append("#define N_OPS %d" % (len(ir["layers"][0]["ops"]) if ir["layers"] else 0))
    off = 0
    for macro, pname in order:
        n = int(np.prod(params[pname]))   # shapes_of 返回的是形状元组
        lines.append(f"#define OFF_{macro} {off}")
        lines.append(f"#define LEN_{macro} {n}")
        off += n
    lines.append(f"#define TOTAL_W {off}")

    # 逐层的 C 语句
    # 严格按 IR 的 ValueRef 生成：每个算子一个独立缓冲，读谁写谁。
    # 之前图省事把残差写成"加 xn"，结果加的是归一化后的值；层输出也没写回 x。
    n_ops = len(ir["layers"][0]["ops"]) if ir["layers"] else 0
    lb = []
    for L in ir["layers"]:
        i = L["index"]
        lb.append(f"case {i}: {{")
        for j, o in enumerate(L["ops"]):
            k, a = o["kind"], o["attrs"]
            def R(r):
                # "op1.out" -> "t1"；序号在下标 2，不是 3
                return "x" if r == "hidden" else "t" + r[2:r.index(".")]
            src = [R(r) for r in o["inputs"]]
            dst = R(o["outputs"][0])
            if k == "Norm":
                # **按 kind 分派。** 以前无条件发 rmsnorm() ——
                # 也就是把 LayerNorm 当 RMSNorm 算（少了减均值和 bias）。
                # 见 C_HEAD 里 layernorm 的注释。
                _eps = float(a.get("eps", _DEFAULT_EPS))
                _op = 1 if a.get("one_plus") else 0
                if str(a.get("kind", "rms")).lower() == "layer":
                    lb.append(
                        f" layernorm({dst}, {src[0]}, P({key(i,j,'w')}),"
                        f" P({key(i,j,'b')}), n, D, {_op}, {_eps}f);")
                else:
                    lb.append(
                        f" rmsnorm({dst}, {src[0]}, P({key(i,j,'w')}), n, D,"
                        f" {_op}, {_eps}f);")
            elif k == "Attention":
                hd, nh, nkv = a["head_dim"], a["q"], a["kv"]
                B = "linear_b" if a["bias"] else "linear"
                def BA(nm):
                    # key() 会把点也换成下划线，所以整个 "wq.bias" 一起传进去
                    return (f", P({key(i,j, nm + '.bias')})" if a["bias"] else "")
                qw = nh * hd * (2 if a["q_gate"] else 1)
                # **融合 q/k/v**（视觉塔）：一个 `qkv` 装下 [q | k | v] 三截，
                # 所以 k / v 就是同一块内存的偏移 —— 指针加法，不用拷。
                # 这是**布局**不是算法，和拆开的算出来必须一样。
                _fz = bool(a.get("qkv_fused"))
                _qn = "qkv" if _fz else "wq"
                # ⚠ **偏移要乘这一层的输入宽度。** `P(nm)` 展开成
                # `W + OFF_nm` —— 那是**元素**指针；而 k 那一截从第 qw **行**
                # 开始，一行是 `in_dim` 个元素。第一版写成 `P(qkv) + qw`，
                # 等于把 k 接到了第 qw 个 float 上（实测 5.6e-01 不过）。
                _ld = int(L.get("in_dim") or d)
                _kn = (f"P({key(i,j,'qkv')}) + {qw * _ld}" if _fz
                       else f"P({key(i,j,'wk')})")
                _vn = (f"P({key(i,j,'qkv')}) + {(qw + nkv * hd) * _ld}" if _fz
                       else f"P({key(i,j,'wv')})")
                _on = "proj" if _fz else "wo"
                lb.append(f" {B}({'qf' if a['q_gate'] else 'q'}, {src[0]},"
                          f" P({key(i,j,_qn)}){BA(_qn)}, n, D, {qw});")
                if a["q_gate"]:
                    lb.append(f" split_qg(q, gt, qf, n, {nh}, {hd});")
                _bn = (lambda off: (f", P({key(i,j,'qkv.bias')}) + {off}"
                                    if a["bias"] else ""))
                lb.append(f" {B}(kk, {src[0]}, {_kn}{_bn(qw) if _fz else BA('wk')}, n, D, {nkv*hd});")
                if a["kv_tie"] and not _fz:
                    # **K=V：v 就是 k 的原始投影。** 分叉要在 qk_norm / rope
                    # 就地改写 kk **之前**（判卷人 by1exec：`v = kraw`，
                    # 而 kraw 是 norm/rope 之前的那一份）。
                    lb.append(f" memcpy(vv, kk, sizeof(float) * (size_t)n * {nkv*hd});")
                else:
                    lb.append(f" {B}(vv, {src[0]}, {_vn}{_bn(qw + nkv*hd) if _fz else BA('wv')}, n, D, {nkv*hd});")
                if _qk_on(a.get("qk_norm")):
                    # **eps 传 IR 里的那个**（宏也是按它生成的）——
                    # 这一行以前没有，于是 qk_norm 里那个写死的 1e-5f
                    # 和 IR 说的 norm_eps 可以不一致，而没人发现。
                    lb.append(f" qk_norm(q, kk, P({key(i,j,'qn.w')}),"
                              f" P({key(i,j,'kn.w')}), n, {nh}, {nkv}, {hd},"
                              f" {1 if a.get('norm_one_plus') else 0},"
                              f" {float(a.get('norm_eps', _DEFAULT_EPS))}f);")
                _y = a.get("yarn") or {}
                _ty = _y.get("type") or "yarn"
                # llama3 的 attention_factor 恒为 1.0，别跟着 YaRN 的公式算
                _sc = a["rope_scale"] if (not _y or _ty == "yarn") else 1.0
                _rt = 0 if not _y else (2 if _ty == "llama3" else 1)
                _nrot = int(hd * a.get("rope_partial", 1.0))
                # **`rope` 是开关。** 这里以前无条件发这行 ——
                # GPT-2 的 `rope = false`，于是它也把这个忙转了。
                if a.get("rope", True):
                    lb.append(f" rope(q, kk, n, {nh}, {nkv}, {hd},"
                          f" {float(a['rope_base'])}f,"
                          f" {0 if a['rope_pairing']=='half' else 1},"
                          f" {float(_sc)}f, {_nrot},"
                          f" {_rt}, {float(_y.get('factor',1))}f,"
                          f" {int(_y.get('original',4096))},"
                          f" {float(_y.get('beta_fast',32))}f,"
                          f" {float(_y.get('beta_slow',1))}f,"
                          f" {1 if _y.get('truncate',True) else 0},"
                          f" {float(_y.get('high_freq',4))}f,"
                          f" {float(_y.get('low_freq',1))}f);")
                lb.append(f" attention(ob, q, kk, vv, n, {nh}, {nkv}, {hd}, {hd},"
                          f" {a['window'] if a['window'] else 0},"
                          f" {'P(' + key(i,j,'sink') + ')' if a['sink'] else 'NULL'},"
                          f" NULL);")
                if a["q_gate"]:
                    lb.append(f" apply_qg(ob, gt, n, {nh*hd});")
                lb.append(f" {B}({dst}, ob, P({key(i,j,_on)}){BA(_on)}, n, {nh*hd}, D);")
            elif k == "FFN":
                hid = a["hidden"]
                _b = bool(a.get("bias"))
                # **参数顺序要和 order 里注册的一致** —— 不一致的话
                # 权重会串位，而算出来的数看着仍然"像那么回事"。
                # **有 bias 用 linear_b，没有用 linear** —— 两个不同的函数，
                # 不是一个函数多一个参数。第一版写成了一个带 NULL 的调用，
                # 那个签名根本不存在。
                def _lin(dst_, x_, w_, b_, di_, do_):
                    if _b:
                        return (f" linear_b({dst_}, {x_}, P({w_}), P({b_}),"
                                f" n, {di_}, {do_});")
                    return f" linear({dst_}, {x_}, P({w_}), n, {di_}, {do_});"
                lb.append(_lin("hb", src[0], key(i,j,'w1'), key(i,j,'w1.bias'),
                               "D", hid))
                # **激活要分派，不能写死 silu。**
                # GPT-2 是 gelu_new —— 写死 silu 会生成一个
                # 看起来对但算错的模型（PyTorch 那边踩过这个坑）。
                _act = a.get("act", "silu")
                _fn = {"gelu_new": "gelu_tanh_", "gelu": "gelu_erf_",
                       "gelu_tanh": "gelu_tanh_", "tanh": "gelu_tanh_",
                       "relu2": "relu2_"}.get(_act, "silu_")
                if a.get("gate", True):
                    lb.append(f" {_fn}(hb, (size_t)n * {hid});")
                    lb.append(_lin("ob", src[0], key(i,j,'w3'),
                                   key(i,j,'w3.bias'), "D", hid))
                    lb.append(f" for (size_t z = 0; z < (size_t)n * {hid}; z++)"
                              f" hb[z] *= ob[z];")
                else:
                    # **无门控：只有一块上投影。** 这就是 GPT-2 的 MLP。
                    lb.append(f" {_fn}(hb, (size_t)n * {hid});")
                lb.append(_lin(dst, "hb", key(i,j,'w2'), key(i,j,'w2.bias'),
                               hid, "D"))
            elif k == "Linear":
                lb.append(
                    f" gdn({dst}, {src[0]}, n, D, {a['k_heads']}, {a['v_heads']},"
                    f" {a['k_dim']}, {a['v_dim']}, {a['conv_kernel']},"
                    f" P({key(i,j,'in_proj_qkvz')}), P({key(i,j,'in_proj_ba')}),"
                    f" P({key(i,j,'conv')}), P({key(i,j,'dt_bias')}),"
                    f" P({key(i,j,'A_log')}), P({key(i,j,'norm.w')}),"
                    f" P({key(i,j,'out_proj')}),"
                    f" {float(a['l2_eps'])}f, {float(a['norm_eps'])}f);")
            elif k == "SSM":
                _cb = (f"P({key(i, j, 'conv.bias')})"
                       if a.get("conv_bias") else "0")
                lb.append(
                    f" ssm({dst}, {src[0]}, n, D, {a['heads']},"
                    f" {a['head_dim']}, {a['ssm_state']}, {a['n_groups']},"
                    f" {a['conv_kernel']},"
                    f" P({key(i, j, 'in_proj')}), P({key(i, j, 'conv')}),"
                    f" {_cb},"
                    f" P({key(i, j, 'dt_bias')}), P({key(i, j, 'A_log')}),"
                    f" P({key(i, j, 'D')}), P({key(i, j, 'norm.w')}),"
                    f" P({key(i, j, 'out_proj')}),"
                    f" {float(a['norm_eps'])}f);")
            elif k == "KDA":
                _low = bool(a.get("gate_lowrank"))
                _ob = bool(a.get("gate_out_bias", True))
                _lo = a.get("gate_lower")

                def _p(nm, when=True):
                    return ("P(%s)" % key(i, j, nm)) if when else "NULL"

                lb.append(
                    f" kda({dst}, {src[0]}, n, D, {a['k_heads']},"
                    f" {a['v_heads']}, {a['k_dim']}, {a['v_dim']},"
                    f" {a['conv_kernel']}, {a['gate_rank']},"
                    f" {1 if _lo is not None else 0},"
                    f" {float(_lo) if _lo is not None else 0.0}f,"
                    f" {float(a['l2_eps'])}f, {float(a['norm_eps'])}f,"
                    f" {_p('q_proj')}, {_p('k_proj')}, {_p('v_proj')},"
                    f" {_p('q_conv1d')}, {_p('k_conv1d')}, {_p('v_conv1d')},"
                    f" {_p('A_log')}, {_p('dt_bias')}, {_p('b_proj')},"
                    f" {_p('f_a_proj', _low) if _low else _p('f_proj')},"
                    f" {_p('f_b_proj', _low)},"
                    f" {_p('g_a_proj', _low) if _low else _p('g_proj')},"
                    f" {_p('g_b_proj', _low)},"
                    f" {_p('g_b_proj.bias', _low and _ob)},"
                    f" {_p('o_norm.w')}, {_p('o_proj')});")
            elif k == "MLA":
                hg = a.get("head_gate", "off") != "off"
                ix = a.get("index_heads")
                _p = (lambda nm: "P(" + key(i, j, nm) + ")") if ix else (lambda nm: "NULL")
                lb.append(
                    f" mla({dst}, {src[0]}, n, D, {a['q']}, {a['q_lora']},"
                    f" {a['kv_lora']}, {a['qk_nope']}, {a['qk_rope']},"
                    f" {a['v_dim']},"
                    f" P({key(i,j,'q_a_proj')}), P({key(i,j,'q_a_layernorm.w')}),"
                    f" P({key(i,j,'q_b_proj')}),"
                    f" P({key(i,j,'kv_a_proj_with_mqa')}),"
                    f" P({key(i,j,'kv_a_layernorm.w')}),"
                    f" P({key(i,j,'kv_b_proj')}), P({key(i,j,'dense')}),"
                    f" {'P(' + key(i,j,'g_proj') + ')' if hg else 'NULL'},"
                    f" {float(a.get('norm_eps', _DEFAULT_EPS))}f,"
                    f" {1 if a.get('norm_one_plus') else 0},"
                    f" {float(a['rope_base'])}f,"
                    f" {0 if a.get('pairing') == 'half' else 1},"
                    f" {1 if a.get('gate_act') == 'sigmoid' else 0},"
                    f" {int(ix or 0)}, {int(a.get('index_dim') or 0)},"
                    f" {int(a.get('index_topk') or 0)},"
                    f" {_p('indexer.wq_b.weight')}, {_p('indexer.wk.weight')},"
                    f" {_p('indexer.k_norm.weight')}, {_p('indexer.k_norm.bias')},"
                    f" {_p('indexer.weights_proj.weight')},"
                    f" {int(a.get('index_kpool') or 1)},"
                    f" {1 if a.get('index_kpool_tail') else 0},"
                    f" {_p('indexer.index_kpool_compress_gate') if int(a.get('index_kpool') or 1) > 1 else 'NULL'},"
                    f" {_p('indexer.index_kpool_compress_ape') if int(a.get('index_kpool') or 1) > 1 else 'NULL'});")
            elif k == "MoE":
                E, K, hid = a["experts"], a["top_k"], a["hidden"]
                sh = a["shared"]
                # 路由：0 = softmax_topk, 1 = topk_softmax,
                #      2 = sigmoid_topk, 3 = sigmoid_group_topk（noaux_tc）
                _RT = {"softmax_topk": 0, "topk_softmax": 1,
                       "sigmoid_topk": 2, "sigmoid_group_topk": 3}
                # 打分函数：**和路由正交**，序数取自 `by1ir.SCORE_KINDS`
                # （0=softmax 1=sigmoid 2=sqrtsoftplus 3=softplus）。
                # 取不到就把 score 当 0 —— 但先在这里拦一道：推断不出来
                # 说明 routing 不在闭集里，那种 IR 本来就不该编译到这一步。
                _rt = str(a.get("routing", "softmax_topk")).strip().lower()
                _sc_name = a.get("score") or a.get("_score") or (
                    "sigmoid" if _rt.startswith("sigmoid") else "softmax")
                _SC = _score_kind_id(_sc_name)
                if _SC is None:
                    raise SystemExit(
                        f"  [不支持] MoE 的打分函数 <%s> 不在 %s 里 —— "
                        "C 后端不知道该怎么算它，**不许当 softmax 混过去**"
                        % (_sc_name, list(_SCORE_KINDS)))
                lb.append(
                    f" moe({dst}, {src[0]}, n, D, {E}, {K}, {hid},"
                    f" P({key(i,j,'router')}),"
                    f" {'P(' + key(i,j,'router.bias') + ')' if a['router_bias'] else 'NULL'},"
                    f" {'P(' + key(i,j,'score_bias') + ')' if a['score_bias'] else 'NULL'},"
                    f" P({key(i,j,'w1')}),"
                    f" P({key(i,j,'w3')}), P({key(i,j,'w2')}),"
                    f" {'P(' + key(i,j,'b1') + ')' if a['expert_bias'] else 'NULL'},"
                    f" {'P(' + key(i,j,'b2') + ')' if a['expert_bias'] else 'NULL'},"
                    f" {'P(' + key(i,j,'b3') + ')' if a['expert_bias'] else 'NULL'},"
                    f" {sh}, {a['shared_hidden'] if sh else 0},"
                    f" {'P(' + key(i,j,'sw1') + ')' if sh else 'NULL'},"
                    f" {'P(' + key(i,j,'sw3') + ')' if sh else 'NULL'},"
                    f" {'P(' + key(i,j,'sw2') + ')' if sh else 'NULL'},"
                    f" {'P(' + key(i,j,'shared_gate') + ')' if a['shared_gate'] else 'NULL'},"
                    f" {_RT.get(a['routing'], 0)}, {_SC}, {int(a.get('n_group') or 0)},"
                    f" {int(a.get('topk_group') or 0)},"
                    f" {1 if a['act'] == 'gptoss' else 0},"
                    f" {float(a['limit']) if a['limit'] else 0.0}f,"
                    f" {float(a['alpha'])}f, {float(a['routed_scale'])}f);")
            elif k == "External":
                # **逃生舱第二层：C 这边直接调外部符号。**
                # ABI 和 Python 那边一模一样（见 by1ext.py）：
                #     void sym(const float *x, float *y, int B, int T, int D,
                #              const float *const *w, int nw);
                # 权重按**名字字典序**传 —— 排序规则是 ABI 的一部分，
                # 两边都得自己算出来。
                _ws = "{" + ", ".join("P(%s)" % key(i, j, "ext:" + nm)
                                      for nm in sorted(a["weights"])) + "}"
                lb.append(f" {{ const float *w_[{max(len(a['weights']), 1)}] = "
                          f"{_ws}; {a['symbol']}({src[0]}, {dst}, (int)n, 1, "
                          f"(int)D, w_, {len(a['weights'])}); }}")
            elif k == "Add":
                lb.append(f" for (size_t z = 0; z < (size_t)n * D; z++)"
                          f" {dst}[z] = {src[0]}[z] + {src[1]}[z];")
        _lo = L["ops"][-1]["outputs"][0]
        last = "t" + _lo[2:_lo.index(".")]
        lb.append(f" memcpy(x, {last}, sizeof(float) * (size_t)n * D); break; }}")
    # 宏必须是**一行** —— 没有续行符的话预处理器只吃第一行
    lines.append("#define LAYER_BODY(L) switch (L) { "
                 + " ".join(lb) + " default: break; }")
    return "\n".join(lines), order, off


# ════════════════════════════════════════════════════════════════════
# 从图发射（1.0.0-rc0 · A3-3）—— **PRIMS -> C 的机械表**。
#
# 和 `by1ggml.emit_model_c_graph` 同一个做法：图里没有"机制"分支 ——
# Attention / FFN 早被 lowering 拆成 matmul / rope / soft_max 这样的原语，
# 所以这里只有一张逐节点的机械表。**内核逐条对着 `by1graph.PRIMS` 的
# ref()**（可执行规格）——公式在那一边，这里只做机械化翻译。
#
# 和 `_emit_c_body`（按 IR kind 调融合核）是**两条发射路径**：
#   --from-graph  走这条（迁移中的新路）
#   默认          走融合核那条（迁移完成前的判卷人，A3-3 收尾时互换）
# 两条路的数字都要记在 `docs/1.0-roadmap.md` —— 不许只记一条。
#
# **形状从 PRIMS 的语义推，不读 `g.vals` 的声明**：声明里有推不出东西的
# （Head 的 `shapes=[(None, None)]` —— vocab 是多少它没说），两份事实
# 并存就是 `ir.md 130` 那个病。唯一事实是 ref() 的 numpy 广播语义，
# 发射器按同一套规则推形状，推不出来的**当场拒绝**。
# ════════════════════════════════════════════════════════════════════

C_PRIMS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *W;                 /* 所有权重，扁平存放 */
#define P(nm) (W + OFF_##nm)

/* ── 原语内核 ─────────────────────────────────────────────────────
   每条内核**逐字对着 `by1graph.PRIMS` 的 ref()**。约定：exec 视角、
   行主序（样本在前、特征在最后轴）。dims 从发射器来（n 是运行期序列长）。
   ⚠ Windows 是 LLP64：`long` 只有 32 位，尺寸一律 `long long`。 */

static void p_copy(float *o, const float *x, long long tot) {
    memcpy(o, x, sizeof(float) * (size_t)tot);
}

static void p_get_rows(float *o, const float *tab, const int *ids,
                       long long n, long long d) {
    /* out = table[ids]：ids 可以是多维（MoE 的 (n,K) 下标），整行取
       `prod(tab.shape[1:])` 个 float —— 平铺后和 ref 的 `table[ids]`
       逐元素一样。 */
    for (long long i = 0; i < n; i++)
        memcpy(o + (size_t)i * d, tab + (size_t)ids[i] * d,
               sizeof(float) * (size_t)d);
}
static void p_get_rows_f(float *o, const float *tab, const float *ids,
                        long long n, long long d) {
    /* `p_get_rows` 的 **float 下标**孪生：图里的下标常常是 `pos` 的视图
       （p_copy 出来是 float 缓冲），值是整数、只是类型是 float。为了它把
       整套值的类型翻成 int 会连累所有旧图（buffer 声明跟着类型走），所以
       就地取整。 */
    for (long long i = 0; i < n; i++)
        memcpy(o + (size_t)i * d,
               tab + (size_t)((long long) ids[i]) * d,
               sizeof(float) * (size_t)d);
}

/* top_k：每行前 k 大的**下标**（不是值），并列取小 —— `_ref_top_k` 的
   `argsort(-x, kind="stable")`：稳定插入排序，严格 `<`，相等保持原序。
   输出是 int（索引），和 `by1exec.op_moe._topk` 同一句：并列处取谁
   没有定义，两个后端挑出不同的不算 bug（`ir.md 129`）。 */
static void p_top_k(int *o, const float *x, long long rows, long long m,
                    long long k) {
    /* k > m 时 ref 的 `[..., :k]` 只会给 m 个 —— 这里同样钳住 */
    if (k > m) k = m;
    long long *ord = (long long *) malloc(sizeof(long long) * (size_t)m);
    for (long long r = 0; r < rows; r++) {
        const float *in = x + r * m;
        for (long long j = 0; j < m; j++) ord[j] = j;
        for (long long a = 1; a < m; a++) {
            long long v = ord[a], b = a - 1;
            while (b >= 0 && in[ord[b]] < in[v]) { ord[b + 1] = ord[b]; b--; }
            ord[b + 1] = v;
        }
        for (long long j = 0; j < k; j++) o[r * k + j] = (int)ord[j];
    }
    free(ord);
}

/* take_along_axis(x, idx, -1)：out[r, j] = x[r, idx[r, j]]。
   前缀维必须一样（numpy 也只广播前缀；这里**不实现就拒绝**）。 */
static void p_gather(float *o, const float *x, const int *idx,
                     long long rows, long long m, long long k) {
    for (long long r = 0; r < rows; r++)
        for (long long j = 0; j < k; j++)
            o[r * k + j] = x[r * m + (long long)idx[r * k + j]];
}

/* 沿 exec 轴求和：x [rows, m, tail] -> o [rows, tail]。
   keepdims 只改形状元数据，不改布局（[rows,1,tail] 和 [rows,tail]
   的行主序一样）—— 所以一个内核两用。 */
static void p_sum(float *o, const float *x, long long rows, long long m,
                  long long tail) {
    for (long long r = 0; r < rows; r++)
        for (long long u = 0; u < tail; u++) {
            float s = 0.f;
            for (long long j = 0; j < m; j++) s += x[(r * m + j) * tail + u];
            o[r * tail + u] = s;
        }
}

/* 广播二元（numpy 右对齐；dim==1 的输入 stride 记 0）。
   op: 0=a+b  1=a*b  2=a/b  3=(a<=b ? 1 : 0) */
static void p_bin(float *o, const float *a, const float *b, int nd,
                  const long long *od, const long long *ad,
                  const long long *bd, int op) {
    long long tot = 1, os[6], as[6], bs[6], idx[6];
    int i; long long t, acc;
    for (i = 0; i < nd; i++) tot *= od[i];
    acc = 1; for (i = nd - 1; i >= 0; i--) { os[i] = acc; acc *= od[i]; }
    acc = 1; for (i = nd - 1; i >= 0; i--) { as[i] = ad[i] == 1 ? 0 : acc; acc *= ad[i]; }
    acc = 1; for (i = nd - 1; i >= 0; i--) { bs[i] = bd[i] == 1 ? 0 : acc; acc *= bd[i]; }
    for (i = 0; i < nd; i++) idx[i] = 0;
    for (t = 0; t < tot; t++) {
        long long ao = 0, bo = 0;
        for (i = 0; i < nd; i++) { ao += idx[i] * as[i]; bo += idx[i] * bs[i]; }
        float u = a[ao], v = b[bo];
        o[t] = op == 0 ? u + v
             : op == 1 ? u * v
             : op == 2 ? u / v
             : (u <= v ? 1.f : 0.f);
        for (i = nd - 1; i >= 0; i--) { if (++idx[i] < od[i]) break; idx[i] = 0; }
    }
}

static void p_add(float *o, const float *a, const float *b, int nd,
                  const long long *od, const long long *ad,
                  const long long *bd) {
    p_bin(o, a, b, nd, od, ad, bd, 0);
}
static void p_mul(float *o, const float *a, const float *b, int nd,
                  const long long *od, const long long *ad,
                  const long long *bd) {
    p_bin(o, a, b, nd, od, ad, bd, 1);
}
static void p_div(float *o, const float *a, const float *b, int nd,
                  const long long *od, const long long *ad,
                  const long long *bd) {
    p_bin(o, a, b, nd, od, ad, bd, 2);
}
static void p_le(float *o, const float *a, const float *b, int nd,
                 const long long *od, const long long *ad,
                 const long long *bd) {
    /* out = 1 若 a <= b 否则 0 —— `_ref_le`（DSA 的因果掩码） */
    p_bin(o, a, b, nd, od, ad, bd, 3);
}

/* 一元（保形）。op: 0=silu 1=gelu_tanh 2=gelu 3=sigmoid 4=exp
   5=log1p 6=sqrt 7=softplus 8=sqrtsoftplus —— 公式照 PRIMS 的 f 列。 */
static void p_un(float *o, const float *x, long long tot, int op) {
    for (long long t = 0; t < tot; t++) {
        float u = x[t], r;
        switch (op) {
        case 0: r = u / (1.0f + expf(-u)); break;
        case 1: r = 0.5f * u * (1.0f + tanhf(
                    (float)0.7978845608028654 * (u + 0.044715f * u * u * u)));
                break;
        case 2: r = 0.5f * u * (1.0f + erff(
                    (float)0.7071067811865476 * u)); break;
        case 3: r = 1.0f / (1.0f + expf(-u)); break;
        case 4: r = expf(u); break;
        case 5: r = log1pf(u); break;
        case 6: r = sqrtf(u); break;
        case 7: r = (u > 0.f ? u : 0.f) + log1pf(expf(-fabsf(u))); break;
        case 8: r = sqrtf((u > 0.f ? u : 0.f) + log1pf(expf(-fabsf(u)))); break;
        default: r = u; break;
        }
        o[t] = r;
    }
}

static void p_scale(float *o, const float *x, long long tot, float s) {
    for (long long t = 0; t < tot; t++) o[t] = x[t] * s;
}

static void p_clamp_min(float *o, const float *x, long long tot, float m) {
    for (long long t = 0; t < tot; t++) o[t] = x[t] > m ? x[t] : m;
}

/* min(x, m) —— clamp_min 的对称面（gptoss 激活的 limit 钳制）。 */
static void p_clamp_max(float *o, const float *x, long long tot, float m) {
    for (long long t = 0; t < tot; t++) o[t] = x[t] < m ? x[t] : m;
}

/* a + c（c 是 attr）—— **不是 clamp**：`+ 1e-20` 和 `max(·, 1e-9)` 是两个
   公式（Σ < c 时不一样），别在机械表里合并成一条。 */
static void p_add_scalar(float *o, const float *x, long long tot, float c) {
    for (long long t = 0; t < tot; t++) o[t] = x[t] + c;
}

/* 把 int 下标行散成 0/1 行：`o[.., j] = 1` 若 j ∈ ids[.., :] —— `_ref_one_hot`。 */
static void p_one_hot(float *o, const int *ids, long long rows,
                      long long cols, long long nidx) {
    memset(o, 0, sizeof(float) * (size_t)(rows * cols));
    for (long long r = 0; r < rows; r++)
        for (long long t = 0; t < nidx; t++) {
            long long j = ids[r * nidx + t];
            if (j >= 0 && j < cols) o[r * cols + j] = 1.f;
        }
}
static void p_one_hot_f(float *o, const float *ids, long long rows,
                         long long cols, long long nidx) {
    /* `p_one_hot` 的 float 下标孪生（下标是 pos 的视图时走这条）。 */
    memset(o, 0, sizeof(float) * (size_t)(rows * cols));
    for (long long r = 0; r < rows; r++)
        for (long long t = 0; t < nidx; t++) {
            long long j = (long long) ids[r * nidx + t];
            if (j >= 0 && j < cols) o[r * cols + j] = 1.f;
        }
}

/* out = m > 0 ? x : v —— `_ref_masked_fill`（v 常是 -inf）。
   m **右对齐广播**到 x 的形状（np.where 本来就播：DSA 的 keep [n,n]
   摊到 [nh,n,n] 的分数上就是这一条）。 */
static void p_masked_fill(float *o, const float *x, const float *m,
                          int nd, const long long *od, const long long *md,
                          float v) {
    long long tot = 1, ms[6], idx[6];
    int i; long long t, acc;
    for (i = 0; i < nd; i++) tot *= od[i];
    acc = 1; for (i = nd - 1; i >= 0; i--) { ms[i] = md[i] == 1 ? 0 : acc; acc *= md[i]; }
    for (i = 0; i < nd; i++) idx[i] = 0;
    for (t = 0; t < tot; t++) {
        long long mo = 0;
        for (i = 0; i < nd; i++) mo += idx[i] * ms[i];
        o[t] = m[mo] > 0.f ? x[t] : v;
        for (i = nd - 1; i >= 0; i--) { if (++idx[i] < od[i]) break; idx[i] = 0; }
    }
}

/* softmax 沿最后轴（`by1ir.moe_scores_array` 的 softmax 那一支）。 */
static void p_softmax(float *o, const float *x, long long rows, long long k) {
    for (long long r = 0; r < rows; r++) {
        const float *in = x + r * k;
        float *ou = o + r * k;
        float mx = -INFINITY;
        for (long long j = 0; j < k; j++) if (in[j] > mx) mx = in[j];
        float s = 0.f;
        for (long long j = 0; j < k; j++) { ou[j] = expf(in[j] - mx); s += ou[j]; }
        for (long long j = 0; j < k; j++) ou[j] /= s;
    }
}

/* x / sqrt(Σx² + eps)，沿最后轴（sum 不是 mean —— PRIMS["l2_norm"]）。 */
static void p_l2_norm(float *o, const float *x, float eps,
                      long long rows, long long d) {
    for (long long i = 0; i < rows; i++) {
        const float *r = x + i * d;
        float *q = o + i * d;
        float s = 0.f;
        for (long long j = 0; j < d; j++) s += r[j] * r[j];
        s = 1.0f / sqrtf(s + eps);
        for (long long j = 0; j < d; j++) q[j] = r[j] * s;
    }
}

/* x · rsqrt(mean(x²)+eps) · (one_plus ? 1+w : w) */
static void p_rms_norm(float *o, const float *x, const float *w, float eps,
                       int one_plus, long long rows, long long d) {
    for (long long i = 0; i < rows; i++) {
        const float *r = x + i * d;
        float *q = o + i * d;
        float s = 0.f;
        for (long long j = 0; j < d; j++) s += r[j] * r[j];
        s = 1.0f / sqrtf(s / (float)d + eps);
        for (long long j = 0; j < d; j++)
            q[j] = r[j] * s * (one_plus ? (1.f + w[j]) : w[j]);
    }
}

/* (x-μ)/sqrt(var+eps) · (one_plus ? 1+w : w) + b */
static void p_layer_norm(float *o, const float *x, const float *w,
                         const float *b, float eps, int one_plus,
                         long long rows, long long d) {
    for (long long i = 0; i < rows; i++) {
        const float *r = x + i * d;
        float *q = o + i * d;
        float mu = 0.f;
        for (long long j = 0; j < d; j++) mu += r[j];
        mu /= (float)d;
        float var = 0.f;
        for (long long j = 0; j < d; j++) { float u = r[j] - mu; var += u * u; }
        var /= (float)d;
        float s = 1.0f / sqrtf(var + eps);
        for (long long j = 0; j < d; j++)
            q[j] = (r[j] - mu) * s * (one_plus ? (1.f + w[j]) : w[j])
                   + (b ? b[j] : 0.f);
    }
}

/* out = x @ swapaxes(w,-1,-2)：w [..., j, k]、x [..., m, k] -> o [..., m, j]。
   批维 numpy 右对齐广播（dim==1 -> stride 0）—— MoE 的专家权重就是
   4 维的（`_ref_matmul` 的 swapaxes 那一课）。
   ⚠ dims 是**形状**，索引用的是**步长**（后缀积）—— 第一版把形状当步长
   用，2 维时 (n,128) 恰好一路带病到段错误。形状 != 步长，两回事。 */
static void p_matmul(float *o, const float *w, const float *x, int nd,
                     const long long *od, const long long *wd,
                     const long long *xd) {
    long long os[6], ws[6], xs[6], idx[6], btot = 1;
    int i; long long t, acc;
    acc = 1; for (i = nd - 1; i >= 0; i--) { os[i] = acc; acc *= od[i]; }
    acc = 1; for (i = nd - 1; i >= 0; i--) { ws[i] = wd[i] == 1 ? 0 : acc; acc *= wd[i]; }
    acc = 1; for (i = nd - 1; i >= 0; i--) { xs[i] = xd[i] == 1 ? 0 : acc; acc *= xd[i]; }
    for (i = 0; i < nd - 2; i++) btot *= od[i];
    for (i = 0; i < nd; i++) idx[i] = 0;
    for (t = 0; t < btot; t++) {
        long long wo = 0, xo = 0, oo = 0;
        for (i = 0; i < nd - 2; i++) {
            wo += idx[i] * ws[i]; xo += idx[i] * xs[i]; oo += idx[i] * os[i];
        }
        long long m = od[nd-2], j = od[nd-1], k = xd[nd-1];
        for (long long p2 = 0; p2 < m; p2++)
            for (long long q2 = 0; q2 < j; q2++) {
                float s = 0.f;
                for (long long c2 = 0; c2 < k; c2++)
                    s += x[xo + p2 * xs[nd-2] + c2 * xs[nd-1]]
                       * w[wo + q2 * ws[nd-2] + c2 * ws[nd-1]];
                o[oo + p2 * os[nd-2] + q2 * os[nd-1]] = s;
            }
        for (i = nd - 3; i >= 0; i--) { if (++idx[i] < od[i]) break; idx[i] = 0; }
    }
}

/* out[β,j,i] = Σ_k a[β//ra,i,k] · b[β//rb,j,k]（GQA 广播照
   `_ref_batch_matmul` 的 np.repeat 语义：头 h 用 kv 头 h//r）。 */
static void p_bmm(float *o, const float *a, const float *b,
                  long long ba, long long bb, long long m, long long nn,
                  long long k) {
    long long B = ba > bb ? ba : bb;
    long long ra = B / ba, rb = B / bb;
    for (long long be = 0; be < B; be++) {
        const float *aa = a + (be / ra) * m * k;
        const float *bb2 = b + (be / rb) * nn * k;
        float *oo = o + be * nn * m;
        for (long long j = 0; j < nn; j++)
            for (long long i = 0; i < m; i++) {
                float s = 0.f;
                for (long long c = 0; c < k; c++)
                    s += aa[i * k + c] * bb2[j * k + c];
                oo[j * m + i] = s;
            }
    }
}

/* rope：x [n, nh, hd]（exec）—— 位置在轴 0、特征在最后轴。
   配对 half (i, i+h2) / interleaved (2i, 2i+1)，**输出都拼接**
   （out[..., i] / out[..., i+h2]），h2 = n_dims/2 —— 公式照
   `_ref_rope`（partial rope 时配对间距是 n_dims/2，不是 hd/2）。
   `inv` 是发射器按 `by1graph.rope_inv`（普通/YaRN/llama3 一支定义）
   算好烤进来的 —— **不在这里再算一遍**。 */
static void p_rope(float *o, const float *x, const int *pos,
                   long long n, int nh, int hd, int n_dims, int half_pair,
                   double rscale, const double *inv) {
    int h2 = n_dims / 2;
    /* **先把 x 拷进来** —— `_ref_rope` 是 `out = 复制 x` 再改旋转对，
       partial rope（n_dims < hd）时**没转的维度带着原值**。第一版直接
       写输出、其余留 calloc 的 0：全量 rope 看不出来，partial 一来
       就差 1e-2 ~ 1e-4（llama3-shaped / _gemma-shaped 两处同时露头）。 */
    memcpy(o, x, sizeof(float) * (size_t)n * nh * hd);
    for (long long t = 0; t < n; t++)
        for (int h = 0; h < nh; h++) {
            const float *xx = x + (t * nh + h) * hd;
            float *oo = o + (t * nh + h) * hd;
            for (int i = 0; i < h2; i++) {
                double th = (double)pos[t] * inv[i];
                float c = (float)(cos(th) * rscale), s = (float)(sin(th) * rscale);
                float u, v;
                if (half_pair) { u = xx[i]; v = xx[i + h2]; }
                else           { u = xx[2 * i]; v = xx[2 * i + 1]; }
                oo[i] = u * c - v * s;
                oo[i + h2] = u * s + v * c;
            }
        }
}

/* softmax(a·scale + mask)，a [头, query, key]，mask 是因果+滑窗
   （by1 的规则：j <= i 且 j > i - window —— 照 `by1graph._mask`）。 */
static void p_soft_max(float *o, const float *a, float scale, int win,
                       long long h, long long q, long long k) {
    for (long long hh = 0; hh < h; hh++)
        for (long long i = 0; i < q; i++) {
            const float *in = a + (hh * q + i) * k;
            float *ou = o + (hh * q + i) * k;
            float mx = -INFINITY;
            for (long long j = 0; j < k; j++) {
                int ok = (j <= i) && (win == 0 || j > i - win);
                float z = in[j] * scale + (ok ? 0.f : -INFINITY);
                ou[j] = z;
                if (z > mx) mx = z;
            }
            float s = 0.f;
            for (long long j = 0; j < k; j++) { ou[j] = expf(ou[j] - mx); s += ou[j]; }
            for (long long j = 0; j < k; j++) ou[j] /= s;
        }
}

/* soft_max + 一个 sink 槽：key 维尾巴拼一列（逐头 sink 标量，无 mask）
   再 softmax，然后**扔掉那一列** —— 于是权重不再和为 1
   （照 `_ref_soft_max_sink`；先加 mask 再拼 sink，顺序要紧）。 */
static void p_soft_max_sink(float *o, const float *a, const float *sink,
                            float scale, int win,
                            long long h, long long q, long long k) {
    for (long long hh = 0; hh < h; hh++)
        for (long long i = 0; i < q; i++) {
            const float *in = a + (hh * q + i) * k;
            float *ou = o + (hh * q + i) * k;
            float mx = sink[hh];
            for (long long j = 0; j < k; j++) {
                int ok = (j <= i) && (win == 0 || j > i - win);
                float z = in[j] * scale + (ok ? 0.f : -INFINITY);
                ou[j] = z;
                if (z > mx) mx = z;
            }
            float s = expf(sink[hh] - mx);
            for (long long j = 0; j < k; j++) { ou[j] = expf(ou[j] - mx); s += ou[j]; }
            for (long long j = 0; j < k; j++) ou[j] /= s;
        }
}

/* np.transpose(x, tp)：输出轴 k 取输入轴 tp[k]。od 是输出形状，
   输入形状 = od 按 tp 重排（ish[tp[k]] = od[k]）。 */
static void p_perm(float *o, const float *x, int nd, const long long *od,
                   const int *tp) {
    long long tot = 1, ish[6], xs[6], idx[6];
    int i, k; long long t, acc;
    for (i = 0; i < nd; i++) tot *= od[i];
    for (i = 0; i < nd; i++) ish[tp[i]] = od[i];
    acc = 1; for (i = nd - 1; i >= 0; i--) { xs[i] = acc; acc *= ish[i]; }
    for (i = 0; i < nd; i++) idx[i] = 0;
    for (t = 0; t < tot; t++) {
        long long xo = 0;
        for (k = 0; k < nd; k++) xo += idx[k] * xs[tp[k]];
        o[t] = x[xo];
        for (i = nd - 1; i >= 0; i--) { if (++idx[i] < od[i]) break; idx[i] = 0; }
    }
}

/* 沿 exec 轴 ax 取 [start, start+d)：rows = 轴前元素数、tail = 轴后元素数。 */
static void p_slice(float *o, const float *x, long long rows, long long d,
                    long long tail, long long start, long long in_d) {
    for (long long r = 0; r < rows; r++)
        for (long long s = 0; s < d; s++)
            memcpy(o + (r * d + s) * tail,
                   x + (r * in_d + (start + s)) * tail,
                   sizeof(float) * (size_t)tail);
}

/* 接两条（沿 exec 轴；前后 rows / tail 块）—— `_ref_concat`。 */
static void p_concat(float *o, const float *a, const float *b,
                     long long rows, long long da, long long db,
                     long long tail) {
    long long d = da + db;
    for (long long r = 0; r < rows; r++)
        for (long long u = 0; u < tail; u++) {
            for (long long j = 0; j < da; j++)
                o[(r * d + j) * tail + u] = a[(r * da + j) * tail + u];
            for (long long j = 0; j < db; j++)
                o[(r * d + da + j) * tail + u] = b[(r * db + j) * tail + u];
        }
}

/* np.repeat 的**整块**重复 [q0,q0,q1,q1]（平铺是 [q0,q1,q0,q1]，和 v 的
   配对对不上，不是一回事）—— `_ref_repeat`。
   x [rows, m, tail] -> o [rows, m*reps, tail]。 */
static void p_repeat(float *o, const float *x, long long rows, long long m,
                     long long tail, int reps) {
    for (long long r = 0; r < rows; r++)
        for (long long j = 0; j < m; j++)
            for (long long z = 0; z < reps; z++)
                for (long long u = 0; u < tail; u++)
                    o[(r * m * reps + j * reps + z) * tail + u]
                        = x[(r * m + j) * tail + u];
}

/* 深度可分离因果卷积（GDN 的短卷积）—— `_ref_conv1d_causal` 的**代码**
   逐句（注意：那段注释里原来的公式 `xpad[:, t+(kernel-1-i)]` 和实现
   `mp[:, i:i+s]` **对不上**，差一次核的翻转；判卷人是 by1exec，实现为准，
   注释已在 by1graph 里改对）。
   x [b, s, C]、w [C, kernel]：out[b,t,c] = Σ_i w[c,i] · x[b, t+i-(kernel-1), c]，
   t+i-(kernel-1) < 0 处补 0。**激活不在这里**（图里 silu 是独立节点）。 */
static void p_conv1d_causal(float *o, const float *x, const float *w,
                            long long b, long long s, long long C,
                            int kernel) {
    for (long long bi = 0; bi < b; bi++)
        for (long long t = 0; t < s; t++)
            for (long long c = 0; c < C; c++) {
                double acc = 0.0;
                for (int i = 0; i < kernel; i++) {
                    long long src = t + i - (kernel - 1);
                    if (src >= 0)
                        acc += (double)w[c * kernel + i]
                             * (double)x[(bi * s + src) * C + c];
                }
                o[(bi * s + t) * C + c] = (float)acc;
            }
}

/* delta 规则（GDN/KDA 的递推扫描）—— `_ref_delta_rule` 逐句：
   st *= exp(decay[t])；δ = (v[t] − (st·k[t])) · beta[t]；st += k[t]⊗δ；
   out[t] = st·q[t]（读的是**更新后**的 st）。状态一步一步走，没有并行
   公式；内部 double 累（和 ref 同精度）。
   `gk` 是 decay 的尾维：**1**（GDN 按头一个）或 **dk**（KDA 按维）——
   写错了不报错、只是算错，所以由发射器核过再传进来。 */
static void p_delta_rule(float *o, const float *q, const float *k,
                         const float *v, const float *beta,
                         const float *decay, long long b, long long h,
                         long long n, long long dk, long long dv,
                         long long gk) {
    double *st = (double *) calloc((size_t)b * h * dk * dv, sizeof(double));
    for (long long t = 0; t < n; t++)
        for (long long bh = 0; bh < b * h; bh++) {
            double *S = st + (size_t)bh * dk * dv;
            const float *kt = k + ((size_t)bh * n + t) * dk;
            const float *qt = q + ((size_t)bh * n + t) * dk;
            const float *vt = v + ((size_t)bh * n + t) * dv;
            double bt = (double)beta[(size_t)bh * n + t];
            long long di, dj;
            for (di = 0; di < dk; di++) {
                double g = exp((double)decay[((size_t)bh * n + t) * gk
                                             + (gk == 1 ? 0 : di)]);
                for (dj = 0; dj < dv; dj++) S[di * dv + dj] *= g;
            }
            for (dj = 0; dj < dv; dj++) {
                double kv = 0.;
                for (di = 0; di < dk; di++)
                    kv += S[di * dv + dj] * (double)kt[di];
                double del = ((double)vt[dj] - kv) * bt;
                for (di = 0; di < dk; di++)
                    S[di * dv + dj] += (double)kt[di] * del;
                double ou = 0.;
                for (di = 0; di < dk; di++)
                    ou += S[di * dv + dj] * (double)qt[di];
                o[((size_t)bh * n + t) * dv + dj] = (float)ou;
            }
        }
    free(st);
}

/* Mamba2 的递推扫描 —— `_ref_ssm_scan` 逐句：st = exp(d·A)·st + d·(x⊗B)；
   y = st·C + D·x。布局 [b,h,n,*]；内部 double 累（和 ref 同精度）。 */
static void p_ssm_scan(float *o, const float *d, const float *A,
                       const float *x, const float *B, const float *C,
                       const float *D, long long b, long long h,
                       long long n, long long hd, long long ssm) {
    double *st = (double *) calloc((size_t)b * h * hd * ssm, sizeof(double));
    for (long long t = 0; t < n; t++)
        for (long long bh = 0; bh < b * h; bh++) {
            double dd = (double)d[(size_t)bh * n + t];
            double dec = exp(dd * (double)A[bh % h]);
            const float *xt = x + ((size_t)bh * n + t) * hd;
            const float *bt = B + ((size_t)bh * n + t) * ssm;
            const float *ct = C + ((size_t)bh * n + t) * ssm;
            double *S = st + (size_t)bh * hd * ssm;
            long long i, s2;
            for (i = 0; i < hd; i++)
                for (s2 = 0; s2 < ssm; s2++)
                    S[i * ssm + s2] = dec * S[i * ssm + s2]
                        + dd * (double)xt[i] * (double)bt[s2];
            for (i = 0; i < hd; i++) {
                double acc = 0.0;
                for (s2 = 0; s2 < ssm; s2++)
                    acc += S[i * ssm + s2] * (double)ct[s2];
                o[((size_t)bh * n + t) * hd + i] =
                    (float)(acc + (double)D[bh % h] * (double)xt[i]);
            }
        }
    free(st);
}
'''


# 一元原语 -> p_un 的 op 序（和 C_PRIMS 里的 switch 一一对应）
_UN_OP = {"silu": 0, "gelu_tanh": 1, "gelu": 2, "sigmoid": 3, "exp": 4,
          "log1p": 5, "sqrt": 6, "softplus": 7, "sqrtsoftplus": 8}

_UN_SHAPE_OK = set(_UN_OP) | {"softmax"}


def emit_c_graph(ir, params, g=None):
    """**从图发射 C**：逐节点把 PRIMS 翻成 `C_PRIMS` 的内核调用。

    返回 (源码, order, 总长) —— 和 `_emit_c_body` 同一个接口。
    图外的装配（w.bin 读入 / ids / logits 写出）在 `C_MAIN_G` 里，
    和融合路径的 `C_MAIN` 同一个命令行契约。
    """
    import by1graph as _bg
    import by1ir as _bir
    if g is None:
        g = _bg.lower_model(ir, "model")
    order = _bir.weight_plan(ir)
    pmap = {pname: macro for macro, pname in order}

    # ── 形状表：值名 -> exec 形状（C 表达式字符串的列表） ──────────
    # typ：值的类型（"f" float / "i" int）—— top_k 出的是**下标**，
    # get_rows / gather 的第二输入只吃 int，混了就要报错，不许将就。
    shp = {"tok": ["n"], "pos": ["n"]}
    typ = {"tok": "i", "pos": "i"}
    for nm, s in params.items():
        shp[nm] = [str(int(x)) for x in s]
        typ[nm] = "f"

    def _view_shp(sh, src):
        """图视角形状 -> exec 形状字符串（None = **推断维**）。

        旧版把 None 写死成 "n"；对绝大多数视图恰好相等，但 k-pool 的
        `view(pos, [P, n/P])` 不行。这里算 `源元素数 / 静态维之积`，
        能约回 "n" 就约（旧图一个字符串都不变），约不掉才出表达式。"""
        out = [None if t is None else str(int(t)) for t in reversed(sh)]
        if sum(1 for t in out if t is None) != 1:
            return ["n" if t is None else t for t in out]
        stat = 1
        for t in out:
            if t is not None:
                stat *= int(t)
        nums, dyn = 1, []
        for t in src:
            if t.lstrip("-").isdigit():
                nums *= int(t)
            else:
                dyn.append(t)
        expr = None
        if len(dyn) == 0 and nums % stat == 0:
            expr = str(nums // stat)
        elif len(dyn) == 1:
            if nums % stat == 0:
                q = nums // stat
                expr = dyn[0] if q == 1 else "(%s) * %d" % (dyn[0], q)
            else:
                expr = "(%s) / %d" % (dyn[0], stat)
        if expr is None:
            raise SystemExit("  [不支持] 图->C：视图的推断维约不出来（%s -> %s）"
                             % (src, out))
        for i, t in enumerate(out):
            if t is None:
                out[i] = expr
        return out

    def _mul_expr(a, k):
        """形状轴乘以常数：数字直接乘；`(X) / k` 乘 k **约回 X**（形状字符串
        要拿来比，`((n) / 4) * 4` 与 `n` 必须比成同一个）。"""
        if a.lstrip("-").isdigit():
            return str(int(a) * k)
        m = re.fullmatch(r"\((.*)\) / (\d+)", a)
        if m and int(m.group(2)) == k:
            return m.group(1)
        return "(%s) * %d" % (a, k)

    def _add_expr(a, b):
        if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
            return str(int(a) + int(b))
        return "(%s) + (%s)" % (a, b)

    def _bc(a, b, what):
        """numpy 右对齐广播。推不出来就**拒绝**，不猜。"""
        nd = max(len(a), len(b))
        A = ["1"] * (nd - len(a)) + list(a)
        B = ["1"] * (nd - len(b)) + list(b)
        O = []
        for u, v in zip(A, B):
            if u == v or v == "1":
                O.append(u)
            elif u == "1":
                O.append(v)
            else:
                raise SystemExit("  [不支持] %s 的广播对不上：%s vs %s"
                                 % (what, a, b))
        return O, A, B

    def _dims(s):
        return "(const long long[]){" + ", ".join(s) + "}"

    def _prod(s):
        return " * ".join("(%s)" % x for x in s) or "1"

    def _ax(xshape, axis):
        """attr 的 ggml 轴 -> exec 轴（和 ref 们同一句换算）。"""
        return len(xshape) - 1 - int(axis)

    # ── 逐节点 ────────────────────────────────────────────────────
    decls, body = [], []
    decls.append("/* %s */" % _stamp())
    decls.append("#define D_MODEL %d" % ir["d_model"])
    decls.append("#define VOCAB %d" % ir["vocab"])
    off = 0
    for macro, pname in order:
        n2 = int(np.prod(params[pname]))
        decls.append("#define OFF_%s %d" % (macro, off))
        decls.append("#define LEN_%s %d" % (macro, n2))
        off += n2
    decls.append("#define TOTAL_W %d" % off)

    used = set()
    rp_n = 0

    def _ref(name):
        """输入 -> C 表达式（float 上下文）。mask_* 是**叶子**（值表里没有，
        soft_max 分支按名字里的 window 现场构造）。`pos` 在 float 上下文里
        用 float 孪生（int 那份给 get_rows / rope 用）。"""
        if name in pmap:
            return "P(%s)" % pmap[name]
        if "mask_" in name:
            return None
        return "v_pos_f" if name == "pos" else "v_" + name.replace(".", "_")

    def _iref(name):
        """int 上下文的输入（get_rows 的下标 / one_hot 的 ids / gather 的
        idx）—— tok / pos 是 int，top_k 的输出也是 int。"""
        return "v_pos" if name == "pos" else _ref(name)

    for node in g.nodes:
        prim, ins, outs, at = (node["prim"], node["ins"], node["outs"],
                               node["attrs"])
        a = [_ref(x) for x in ins]
        dst = "v_" + outs[0].replace(".", "_")
        used.add(outs[0])
        shp_in = []
        for x in ins:
            if "mask_" in x:
                shp_in.append(None)
                continue
            if x not in shp:
                raise SystemExit("  [FAIL] %s 的输入 %r 推不出形状" % (prim, x))
            shp_in.append(shp[x])
        body.append("    /* %s -> %s */" % (prim, outs[0]))

        if prim == "get_rows":
            tab, ids_ = shp_in[0], shp_in[1]
            if len(tab) < 2:
                raise SystemExit("  [不支持] get_rows 的表是 %d 维 —— "
                                 "至少要 2 维" % len(tab))
            # out = table[ids]：ids 形状 + 表的行宽（MoE 的 (n,K) 下标取
            # (E,out,in) 的专家整行 -> (n,K,out,in) 就是这一条；k-pool 的池
            # gather 是按**图值**取行，同一条式）。
            out = ids_ + tab[1:]
            if typ.get(ins[1]) == "i":
                body.append("    p_get_rows(%s, %s, %s, %s, %s);"
                            % (dst, a[0], _iref(ins[1]), _prod(ids_),
                               _prod(tab[1:])))
            else:
                # 下标是 **float 视图**（pos 的视图；池路的 pidx 就是它）
                body.append("    p_get_rows_f(%s, %s, %s, %s, %s);"
                            % (dst, a[0], a[1], _prod(ids_),
                               _prod(tab[1:])))
        elif prim == "top_k":
            out = list(shp_in[0])
            _m = shp_in[0][-1]
            # ref 的 `[..., :k]` 自动钳到行宽，这里同规则（否则 C 越界）；
            # 行宽是 n 时按 k 铺缓冲 —— 内核运行期钳，只会少写不多写
            if _m.lstrip("-").isdigit():
                out[-1] = str(min(int(at["k"]), int(_m)))
            else:
                # 行宽要么是 n、要么是表达式（池轴 n/kpool）：缓冲按 k 铺，
                # 内核运行期钳 —— 只会少写不多写
                out[-1] = str(int(at["k"]))
            body.append("    p_top_k(%s, %s, %s, %s, %s);"
                        % (dst, a[0], _prod(shp_in[0][:-1]),
                           _m, out[-1]))
        elif prim == "one_hot":
            x, rf = shp_in[0], shp_in[1]
            out = x[:-1] + [rf[-1]]       # 宽度 = 参照的最后一维
            if typ.get(ins[0]) == "i":
                body.append("    p_one_hot(%s, %s, %s, %s, %s);"
                            % (dst, _iref(ins[0]), _prod(x[:-1]), out[-1],
                               x[-1]))
            else:
                body.append("    p_one_hot_f(%s, %s, %s, %s, %s);"
                            % (dst, a[0], _prod(x[:-1]), out[-1], x[-1]))
        elif prim == "masked_fill":
            out, _A, B = _bc(shp_in[0], shp_in[1], "masked_fill")
            _v = float(at["value"])
            _vs = ("-INFINITY" if _v == float("-inf") else
                   "INFINITY" if _v == float("inf") else repr(_v) + "f")
            body.append("    p_masked_fill(%s, %s, %s, %d, %s, %s, %s);"
                        % (dst, a[0], a[1], len(out), _dims(out), _dims(B),
                           _vs))
        elif prim == "gather":
            x, ii = shp_in[0], shp_in[1]
            if typ.get(ins[1]) != "i":
                raise SystemExit("  [FAIL] gather 的下标 %r 不是 int" % ins[1])
            if x[:-1] != ii[:-1]:
                raise SystemExit("  [不支持] gather 的前缀维对不上：%s vs %s"
                                 "（numpy 只广播前缀，这里不实现就拒绝）"
                                 % (x, ii))
            out = x[:-1] + [ii[-1]]
            body.append("    p_gather(%s, %s, %s, %s, %s, %s);"
                        % (dst, a[0], _iref(ins[1]), _prod(x[:-1]), x[-1],
                           ii[-1]))
        elif prim == "sum":
            x = shp_in[0]
            ax = _ax(x, at["axis"])
            out = list(x)
            if bool(at.get("keepdims", True)):
                out[ax] = "1"
            else:
                out = out[:ax] + out[ax + 1:]
            body.append("    p_sum(%s, %s, %s, %s, %s);"
                        % (dst, a[0], _prod(x[:ax]), x[ax], _prod(x[ax + 1:])))
        elif prim in ("add", "mul", "div", "le"):
            out, A, B = _bc(shp_in[0], shp_in[1], prim)
            nd = len(out)
            body.append("    p_%s(%s, %s, %s, %d, %s, %s, %s);"
                        % (prim, dst, a[0], a[1], nd, _dims(out),
                           _dims(A), _dims(B)))
        elif prim in ("scale", "clamp_min", "clamp_max", "add_scalar"):
            out = list(shp_in[0])
            _k = {"scale": "s", "clamp_min": "min", "clamp_max": "max",
                  "add_scalar": "c"}[prim]
            body.append("    p_%s(%s, %s, %s, %sf);"
                        % (prim, dst, a[0], _prod(out),
                           repr(float(at[_k]))))
        elif prim in _UN_SHAPE_OK:
            out = list(shp_in[0])
            if prim == "softmax":
                k = out[-1]
                rows = _prod(out[:-1])
                body.append("    p_softmax(%s, %s, %s, %s);"
                            % (dst, a[0], rows, k))
            else:
                body.append("    p_un(%s, %s, %s, %d);"
                            % (dst, a[0], _prod(out), _UN_OP[prim]))
        elif prim == "l2_norm":
            out = list(shp_in[0])
            body.append("    p_l2_norm(%s, %s, %sf, %s, %s);"
                        % (dst, a[0], repr(float(at["eps"])),
                           _prod(out[:-1]), out[-1]))
        elif prim == "rms_norm":
            out = list(shp_in[0])
            body.append("    p_rms_norm(%s, %s, %s, %sf, %d, %s, %s);"
                        % (dst, a[0], a[1], repr(float(at["eps"])),
                           1 if at.get("one_plus") else 0,
                           _prod(out[:-1]), out[-1]))
        elif prim == "layer_norm":
            out = list(shp_in[0])
            body.append("    p_layer_norm(%s, %s, %s, %s, %sf, %d, %s, %s);"
                        % (dst, a[0], a[1], a[2], repr(float(at["eps"])),
                           1 if at.get("one_plus") else 0,
                           _prod(out[:-1]), out[-1]))
        elif prim == "matmul":
            w, x = shp_in[0], shp_in[1]
            if len(w) < 2 or len(x) < 2:
                raise SystemExit("  [不支持] matmul 的输入维度 < 2")
            if w[-1] != x[-1]:
                raise SystemExit("  [FAIL] matmul 的 k 对不上：%s vs %s" % (w, x))
            ob, A, B = _bc(x[:-2], w[:-2], "matmul 批维")
            out = ob + [x[-2], w[-2]]
            nd = len(out)
            # 参数顺序：dst, w, x, nd, od, wd, xd —— A 是 x 的批维、
            # B 是 w 的批维（第一版在这儿把两边传反了，2 维看不出来）
            body.append("    p_matmul(%s, %s, %s, %d, %s, %s, %s);"
                        % (dst, a[0], a[1], nd, _dims(out),
                           _dims(B + [w[-2], w[-1]]),
                           _dims(A + [x[-2], x[-1]])))
        elif prim == "batch_matmul":
            A3, B3 = shp_in[0], shp_in[1]
            if len(A3) != 3 or len(B3) != 3 or A3[2] != B3[2]:
                raise SystemExit("  [不支持] batch_matmul 只吃两个 3 维、k 相等"
                                 "（%s / %s）" % (A3, B3))
            # 批维可以是 n（DSA 的按-token 加权和）—— B = max(ba, bb) 由
            # 内核算，缓冲按**运行期 max 表达式**铺，别赌 n 一定更大
            if A3[0] == B3[0]:
                _B = A3[0]
            elif A3[0] == "1":
                _B = B3[0]
            elif B3[0] == "1":
                _B = A3[0]
            elif A3[0] != "n" and B3[0] != "n":
                _B = str(max(int(A3[0]), int(B3[0])))
            else:
                _B = ("((%s) > (%s) ? (%s) : (%s))"
                      % (A3[0], B3[0], A3[0], B3[0]))
            out = [_B, B3[1], A3[1]]
            body.append("    p_bmm(%s, %s, %s, %s, %s, %s, %s, %s);"
                        % (dst, a[0], a[1], A3[0], B3[0], A3[1], B3[1], A3[2]))
        elif prim == "rope":
            x = shp_in[0]
            if len(x) != 3:
                raise SystemExit("  [不支持] rope 只吃 [n, 头, hd]（现在是 %s）" % x)
            out = list(x)
            inv = _bg.rope_inv(int(at["n_dims"]), float(at["base"]),
                               at.get("yarn"))
            rn = "RP_%d" % rp_n
            rp_n += 1
            decls.append("static const double %s[] = {%s};"
                         % (rn, ", ".join(repr(float(v)) for v in inv)))
            body.append("    p_rope(%s, %s, v_pos, n, %s, %s, %d, %d, %r, %s);"
                        % (dst, a[0], x[1], x[2], int(at["n_dims"]),
                           1 if str(at.get("pairing")) == "half" else 0,
                           float(at.get("rope_scale", 1.0)), rn))
        elif prim in ("soft_max", "soft_max_sink"):
            x = shp_in[0]
            if len(x) != 3:
                raise SystemExit("  [不支持] %s 只吃 [头, query, key]" % prim)
            out = list(x)
            win = int(ins[1].rsplit("_", 1)[-1])
            if prim == "soft_max":
                body.append("    p_soft_max(%s, %s, %r, %d, %s, %s, %s);"
                            % (dst, a[0], float(at["scale"]), win,
                               x[0], x[1], x[2]))
            else:
                body.append("    p_soft_max_sink(%s, %s, %s, %r, %d,"
                            " %s, %s, %s);"
                            % (dst, a[0], a[2], float(at["scale"]), win,
                               x[0], x[1], x[2]))
        elif prim in ("view", "cont"):
            # view / cont 带 shape：**按 ggml 视角逐字换算成 exec 形状**
            # （`_exec_view` 那一课）；cont 不带 shape 就是纯物化。
            # 两者都只是**按行主序拷贝** —— reshape 不改元素序。
            if typ.get(ins[0]) == "i" and ins[0] != "pos":
                # pos 的 float 孪生走 _ref；别的 int 值（top_k 输出）的
                # 视图没实现就拒绝，不许默默按 float 拷
                raise SystemExit("  [不支持] %s 的输入 %r 是 int" % (prim, ins[0]))
            if prim == "view" or at.get("shape"):
                sh = at["shape"]
                out = _view_shp(sh, shp_in[0])
            else:
                out = list(shp_in[0])
            body.append("    p_copy(%s, %s, %s);" % (dst, a[0], _prod(out)))
        elif prim == "permute":
            if typ.get(ins[0]) == "i":
                raise SystemExit("  [不支持] permute 的输入 %r 是 int" % ins[0])
            x = shp_in[0]
            axes = list(at["axes"])
            nd = len(x)
            tp = tuple(nd - 1 - axes[nd - 1 - k] for k in range(nd))
            out = [x[tp[k]] for k in range(nd)]
            body.append("    p_perm(%s, %s, %d, %s, (const int[]){%s});"
                        % (dst, a[0], nd, _dims(out),
                           ", ".join(str(t) for t in tp)))
        elif prim == "slice":
            x = shp_in[0]
            ax = _ax(x, at["axis"])
            d = int(at["stop"]) - int(at["start"])
            out = list(x)
            out[ax] = str(d)
            body.append("    p_slice(%s, %s, %s, %d, %s, %d, %s);"
                        % (dst, a[0], _prod(x[:ax]), d, _prod(x[ax + 1:]),
                           int(at["start"]), x[ax]))
        elif prim == "concat":
            A3, B3 = shp_in[0], shp_in[1]
            ax = _ax(A3, at["axis"])
            if A3[:ax] + A3[ax + 1:] != B3[:ax] + B3[ax + 1:]:
                raise SystemExit("  [不支持] concat 的非拼接维对不上：%s vs %s"
                                 % (A3, B3))
            if A3[ax] == "n" or B3[ax] == "n":
                raise SystemExit("  [不支持] concat 沿 n 轴拼 —— 没推过")
            out = list(A3)
            out[ax] = _add_expr(A3[ax], B3[ax])
            body.append("    p_concat(%s, %s, %s, %s, %s, %s, %s);"
                        % (dst, a[0], a[1], _prod(A3[:ax]), A3[ax], B3[ax],
                           _prod(A3[ax + 1:])))
        elif prim == "repeat":
            x = shp_in[0]
            ax = _ax(x, at["axis"])
            reps = int(at["reps"])
            if x[ax] == "n":
                raise SystemExit("  [不支持] repeat 沿 n 轴 —— 没推过")
            out = list(x)
            out[ax] = _mul_expr(x[ax], reps)
            body.append("    p_repeat(%s, %s, %s, %s, %s, %d);"
                        % (dst, a[0], _prod(x[:ax]), x[ax],
                           _prod(x[ax + 1:]), reps))
        elif prim == "conv1d_causal":
            x, w = shp_in[0], shp_in[1]
            if len(x) != 3 or len(w) != 2:
                raise SystemExit("  [不支持] conv1d_causal 吃 x[b,s,C] + "
                                 "w[C,kernel]（现在 %s / %s）" % (x, w))
            if x[2] != w[0]:
                raise SystemExit("  [FAIL] conv1d_causal 的通道对不上：%s vs %s"
                                 % (x, w))
            out = list(x)
            body.append("    p_conv1d_causal(%s, %s, %s, %s, %s, %s, %d);"
                        % (dst, a[0], a[1], x[0], x[1], x[2],
                           int(at["kernel"])))
        elif prim == "delta_rule":
            Q, K, V, B5, G5 = shp_in
            if any(len(s) != 4 for s in (Q, K, V, B5, G5)):
                raise SystemExit("  [不支持] delta_rule 的输入都要 4 维 "
                                 "[b, 头, n, dim]（现在 %s）" % (shp_in,))
            if not (Q[:3] == K[:3] == V[:3] == B5[:3] == G5[:3]):
                raise SystemExit("  [FAIL] delta_rule 的 (b,头,n) 对不上")
            if B5[3] != "1":
                raise SystemExit("  [不支持] delta_rule 的 beta 尾维 != 1（%s）"
                                 % B5)
            if G5[3] != "1" and G5[3] != Q[3]:
                raise SystemExit("  [不支持] delta_rule 的 decay 尾维 %s —— "
                                 "只实现了 1（GDN 按头）或 dk（KDA 按维）"
                                 % G5[3])
            if Q[3] != K[3]:
                raise SystemExit("  [FAIL] delta_rule 的 dk 对不上：%s vs %s"
                                 % (Q, K))
            out = list(V)
            body.append("    p_delta_rule(%s, %s, %s, %s, %s, %s,"
                        " %s, %s, %s, %s, %s, %s);"
                        % (dst, a[0], a[1], a[2], a[3], a[4],
                           Q[0], Q[1], Q[2], Q[3], V[3], G5[3]))
        elif prim == "ssm_scan":
            DD, A1, X4, B4, C4, D1 = shp_in
            if len(X4) != 4 or any(len(s) != 4 for s in (DD, B4, C4)):
                raise SystemExit("  [不支持] ssm_scan 吃 d/x/B/C 四维 "
                                 "[b,头,n,*]（现在 %s）" % (shp_in,))
            if not (DD[:3] == X4[:3] == B4[:3] == C4[:3]) or DD[3] != "1":
                raise SystemExit("  [FAIL] ssm_scan 的 (b,头,n) 对不上"
                                 "（%s）" % (shp_in,))
            if B4[3] != C4[3]:
                raise SystemExit("  [FAIL] ssm_scan 的 ssm 维对不上")
            if A1 != [X4[1]] or D1 != [X4[1]]:
                raise SystemExit("  [不支持] ssm_scan 的 A/D 要逐头 [头]"
                                 "（%s / %s）" % (A1, D1))
            out = list(X4)
            body.append("    p_ssm_scan(%s, %s, %s, %s, %s, %s, %s,"
                        " %s, %s, %s, %s, %s);"
                        % (dst, a[0], a[1], a[2], a[3], a[4], a[5],
                           X4[0], X4[1], X4[2], X4[3], B4[3]))
        else:
            raise SystemExit(
                "  [不支持] 图的 %r 原语还没有 C 映射 —— "
                "**要么实现、要么拒绝**（A3-3 的下一刀）" % prim)
        shp[outs[0]] = out
        typ[outs[0]] = "i" if prim == "top_k" else "f"

    # ── 装配：值缓冲 + 逐节点 + 同一个命令行契约 ──────────────────
    logit = None
    for n2 in reversed(g.nodes):
        if n2["outs"][0].replace(".", "_").endswith("logits"):
            logit = n2["outs"][0]
            break
    if logit is None:
        logit = g.nodes[-1]["outs"][0]
    alloc = ["", "    /* ---- 值缓冲（一个值一块，exec 约定行主序） ---- */"]
    alloc.append("    int *v_pos = (int *) malloc(sizeof(int) * (size_t) n);")
    alloc.append("    float *v_pos_f = (float *) malloc(sizeof(float)"
                 " * (size_t) n);")
    alloc.append("    for (int i = 0; i < n; i++) { v_pos[i] = i;"
                 " v_pos_f[i] = (float) i; }")
    for v in sorted(used):
        nm = "v_" + v.replace(".", "_")
        if typ.get(v) == "i":
            alloc.append("    int *%s = (int *) calloc((size_t)(%s),"
                         " sizeof(int));" % (nm, _prod(shp[v])))
        else:
            alloc.append("    float *%s = (float *) calloc((size_t)(%s),"
                         " sizeof(float));" % (nm, _prod(shp[v])))
    main = [
        "",
        "int main(int argc, char **argv) {",
        "    int n = atoi(argv[1]);",
        "    FILE *f = fopen(argv[2], \"rb\");",
        "    if (!f) { fprintf(stderr, \"w.bin open fail\\n\"); return 1; }",
        "    fseek(f, 0, SEEK_END);",
        "    long sz = ftell(f);",
        "    fseek(f, 0, SEEK_SET);",
        "    W = (float *) malloc(sz);",
        "    if (fread(W, 1, sz, f) != (size_t) sz) {"
        " fprintf(stderr, \"weights read fail\\n\"); return 1; }",
        "    fclose(f);",
        "    int *ids = (int *) malloc(sizeof(int) * (size_t) n);",
        "    f = fopen(argv[3], \"rb\");",
        "    if (fread(ids, sizeof(int), (size_t) n, f) != (size_t) n) {"
        " fprintf(stderr, \"ids read fail\\n\"); return 1; }",
        "    fclose(f);",
        "    /* tok 就是读进来的 ids —— 图里 get_rows 的第二输入 */",
        "    int *v_tok = ids;",
    ]
    tail = [
        "    f = fopen(argv[4], \"wb\");",
        "    fwrite(%s, sizeof(float), (size_t)(%s), f);"
        % ("v_" + logit.replace(".", "_"), _prod(shp[logit])),
        "    fclose(f);",
        "    printf(\"ok %d x %d\\n\", n, VOCAB);",
        "    return 0;",
        "}",
    ]
    src = "\n".join([C_PRIMS] + decls + main + alloc + body + tail)
    # 形状表留给调用方（差分定位 / 调试要看每个值的尺寸）
    emit_c_graph.last_shp = shp
    emit_c_graph.last_typ = typ
    return src, order, off


def _stamp():
    try:
        from by1ver import stamp as _s
    except ImportError:
        def _s():
            return "by1"
    return _s()


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 C 后端")
    ap.add_argument("by1")
    ap.add_argument("--gcc", required=True)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--workdir", default="cgen")
    ap.add_argument("--from-graph", action="store_true",
                    help="走图路径（A3-3 的 PRIMS->C 机械表）；"
                         "默认还是融合核路径")
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    ex_mod = load("by1exec")
    name = os.path.basename(args.by1)
    _r, info = bc.check(args.by1)
    ir = cg.compile_ir(info)

    print(f"\n{'='*72}\n  {name}  ->  C\n{'='*72}")
    try:
        if args.from_graph:
            # **图路径**（A3-3）：整份源码由 emit_c_graph 装配好
            src, order, total = emit_c_graph(ir, ex_mod.shapes_of(ir))
            print("  发射路径：图（PRIMS -> C 机械表）")
        else:
            decls, order, total = emit_c(ir, info, ex_mod.shapes_of(ir))
            src = C_HEAD + "\n" + decls + "\n" + C_MAIN
            print("  发射路径：IR 融合核（迁移期间的判卷人）")
    except SystemExit as e:
        print(str(e))
        return 1

    os.makedirs(args.workdir, exist_ok=True)
    rng = np.random.default_rng(0)
    params = {k: rng.normal(0, 0.1, s).astype(np.float32)
              for k, s in ex_mod.shapes_of(ir).items()}
    flat = np.concatenate([params[p].ravel() for _m, p in order])
    print(f"  权重 {total:,} 个 float = {total*4/1024:.1f} KiB，"
          f"{len(order)} 个张量")
    by1io.write_bytes(os.path.join(args.workdir, "w.bin"), flat.tobytes())
    ids = rng.integers(0, ir["vocab"], args.seq).astype(np.int32)
    by1io.write_bytes(os.path.join(args.workdir, "ids.bin"), ids.tobytes())

    cpath = os.path.join(args.workdir, "model.c")
    by1io.write_text(cpath, src)
    print(f"  生成 {cpath}  （{len(src.splitlines())} 行）")

    # **绝对路径，不是相对路径。**
    # POSIX 上 `subprocess.run(["cgen/model.exe"])` 会 FileNotFoundError ——
    # 相对路径没有 `/` 开头，系统不去当前目录找（Windows 会找）。
    # 这个后端只在 Windows/mingw 上跑过，所以一直没露。
    exe = os.path.abspath(os.path.join(args.workdir, "model.exe"))
    # **外部符号要链接进去。** 路径按 IR 的说法解析（相对当前目录），
    # 而 `-Wl,-rpath` 让运行的时候也找得到 —— 否则编译过了跑不起来，
    # 而那个报错长得像别的问题。
    _link = []
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] == "External":
                _lib = os.path.abspath(o["attrs"]["lib"])
                _link += [_lib, "-Wl,-rpath,%s" % os.path.dirname(_lib)]
    # **两个 subprocess 都要有上限。** 一个卡住的编译器（或者生成的程序
    # 自己死循环）会让这一整轮验证永远停在这里。
    p = subprocess.run([args.gcc, "-O2", "-o", exe, cpath, "-lm"] + _link,
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        print("  [编译失败]")
        print((p.stderr or "")[:1600])
        return 1
    print("  [编译成功]")

    r = subprocess.run([exe, str(args.seq),
                        os.path.join(args.workdir, "w.bin"),
                        os.path.join(args.workdir, "ids.bin"),
                        os.path.join(args.workdir, "logits.bin")],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  [运行失败] {r.stdout} {r.stderr}")
        return 1
    print(f"  [运行成功] {r.stdout.strip()}")
    got = np.fromfile(os.path.join(args.workdir, "logits.bin"), dtype=np.float32)
    got = got.reshape(args.seq, ir["vocab"])

    ex = ex_mod.Exec(ir, params)
    # **第四份硬编码的全局参数名单。** by1exec 里改了两处，
    # 这里漏了 —— 于是 gpt2-tiny 报 `KeyError: 'pos.weight'`，
    # 而那个报错长得像"这个张量没声明"。
    ex.P["_g"] = {k: params[k] for k in
                  ("embed.weight", "pos.weight", "final_norm.w",
                   "final_norm.b", "head.weight") if k in params}
    want = ex(ids[None, :])[0]
    dd = np.abs(got - want).max()
    amp = max(np.abs(want).max(), 1e-9)
    print(f"\n  C 后端 vs NumPy 后端")
    print(f"    幅度 C {np.abs(got).max():.4e}   NumPy {np.abs(want).max():.4e}")
    print(f"    最大绝对差 {dd:.3e}   相对 {dd/amp:.3e}   "
          + ("[PASS] 三个后端一致" if dd / amp < 1e-4 else "[FAIL] 不一致"))
    print()
    return 0 if dd / amp < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
