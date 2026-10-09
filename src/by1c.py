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

# **eps 的默认值只有一个地方写**（`by1ir.EPS_DEFAULT`）。
# 这个文件里原来有 9 处 `1e-5`；C_HEAD 里那段"eps 必须由调用方传进来"
# 的注释记的就是这一类 —— 而它当时没管住 `qk_norm()`。
try:
    from by1ir import EPS_DEFAULT as _DEFAULT_EPS
except ImportError:                     # 单独拷一个文件出去时兜底
    _DEFAULT_EPS = 1e-5

HERE = os.path.dirname(os.path.abspath(__file__))
MANGLE = re.compile(r"[^0-9A-Za-z]")



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
        name, os.path.join(HERE, name + ".py"))
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
 * 那会生成一个看起来对但算错的模型，所以上一轮先让它**拒绝**。
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
                const float *router, const float *rb, const float *w1, const float *w3,
                const float *w2, const float *b1, const float *b2,
                const float *b3, int sh, int shh, const float *sw1,
                const float *sw3, const float *sw2, const float *sg,
                int routing, int gptoss, float lim, float alpha, float scale) {
    int *idx = (int *)malloc(sizeof(int) * n * K);
    float *wts = (float *)malloc(sizeof(float) * n * K);
    float *lg = (float *)malloc(sizeof(float) * E);
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
    free(idx); free(wts); free(lg); free(gu); free(uu); free(hh);
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
 * **eps 由调用方传进来。** 这里原来是写死的 `1e-5f` —— 而
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


/* mla() 会用到 attention()，而它定义在后面 —— 前置声明。 */
static void attention(float *o, float *q, const float *k, const float *v,
                      int n, int nh, int nkv, int hd, int vd, int window,
                      const float *sink);

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
                int sigmoid_gate) {
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

    attention(o, qq, kk, vv, n, nh, nh, qk, vd, 0, NULL);

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
                      const float *sink) {
    int rep = nh / nkv;
    float sc = 1.0f / sqrtf((float)hd);
    float *att = (float *)malloc(sizeof(float) * n);
    float *vb = (float *)malloc(sizeof(float) * vd);
    for (int h = 0; h < nh; h++) {
        int kh = h / rep;
        for (int i = 0; i < n; i++) {
            const float *qi = q + ((size_t)i * nh + h) * hd;
            float mx = -INFINITY;
            for (int j = 0; j <= i; j++) {
                if (window && j <= i - window) { att[j] = 0.f; continue; }
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
                if (window && j <= i - window) continue;
                att[j] = expf(att[j] - mx);
                sum += att[j];
            }
            /* sink：一列额外的 logit 参与 softmax，之后丢掉 ——
               剩下的概率和小于 1，它专门用来吸收质量 */
            if (sink) sum += expf(sink[h] - mx);
            float *oi = o + ((size_t)i * nh + h) * vd;
            for (int t = 0; t < vd; t++) vb[t] = 0.f;
            for (int j = 0; j <= i; j++) {
                if (window && j <= i - window) continue;
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
    # **LayerNorm 和学习式位置表现在实现了**，所以上一轮那两条
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
        # 这里原来是硬拒绝 —— 于是"纯 Linear / 纯外部算子"的模型
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

    # 收集参数（顺序确定，C 与 Python 用同一份）
    # **全局参数按 IR 实际有哪些来。**
    # 学习式位置表和最终归一化的 bias 以前不在这里 ——
    # 于是 C 那边根本不知道它们存在。
    order = [("EMBED", "embed.weight")]
    if str(ir.get("pos_kind", "rope")).lower() == "learned":
        order.append(("POS", "pos.weight"))
    order.append(("FINAL_NORM", "final_norm.w"))
    if str(ir.get("norm_kind", "rms")).lower() == "layer":
        order.append(("FINAL_NORM_B", "final_norm.b"))
    order.append(("HEAD", "head.weight"))
    body = []
    for L in ir["layers"]:
        i = L["index"]
        for j, o in enumerate(L["ops"]):
            k, a, pre = o["kind"], o["attrs"], f"layers.{i}.op{j}."
            if k == "Norm":
                order.append((key(i, j, "w"), pre + "w"))
                # **LayerNorm 有 bias。** 见 C_HEAD 里 layernorm 的注释。
                # 少了这一行，生成出来的 C 会引用一个没有的
                # `OFF_L0_OP0_B` 宏 —— 编译期就报，算是运气好。
                if str(a.get("kind", "rms")).lower() == "layer":
                    order.append((key(i, j, "b"), pre + "b"))
            elif k == "Attention":
                if a["kv_tie"] or a["head_gate"] != "off":
                    raise SystemExit(
                        f"  [不支持] 第 {i} 层的注意力用了 kv_tie/"
                        f"head_gate —— C 后端还没实现")
                for nm in ("wq", "wk", "wv", "wo"):
                    order.append((key(i, j, nm), pre + nm))
                if a["bias"]:
                    for nm in ("wq.bias", "wk.bias", "wv.bias", "wo.bias"):
                        order.append((key(i, j, nm), pre + nm))
                if a["sink"]:
                    order.append((key(i, j, "sink"), pre + "sink"))
                if _qk_on(a.get("qk_norm")):
                    for nm in ("qn.w", "kn.w"):
                        order.append((key(i, j, nm), pre + nm))
                body.append((i, j, "Attention", a))
            elif k == "FFN":
                # **无门控的 FFN 也是 FFN。**
                # 这里原来是 `if not gate: raise` —— 和 by1codegen 那边当初
                # 一模一样的毛病：一条分支只为"有门"那一种情况写过。
                # **而 GPT-2 正是没有门的那种**，于是 C 后端建不了它。
                gated = a.get("gate", True)
                for nm in (("w1", "w2", "w3") if gated else ("w1", "w2")):
                    order.append((key(i, j, nm), pre + nm))
                if a.get("bias"):
                    for nm in (("w1.bias", "w2.bias", "w3.bias") if gated
                               else ("w1.bias", "w2.bias")):
                        order.append((key(i, j, nm), pre + nm))
                if a.get("act") == "gptoss":
                    raise SystemExit(f"  [不支持] 第 {i} 层的 FFN 用了 gptoss 激活")
                body.append((i, j, "FFN", a))
            elif k == "Linear":
                for nm in ("in_proj_qkvz", "in_proj_ba", "conv", "dt_bias",
                           "A_log", "norm.w", "out_proj"):
                    order.append((key(i, j, nm), pre + nm))
                body.append((i, j, "Linear", a))
            elif k == "MLA":
                for nm in ("q_a_proj", "q_a_layernorm.w", "q_b_proj",
                           "kv_a_proj_with_mqa", "kv_a_layernorm.w",
                           "kv_b_proj", "dense"):
                    order.append((key(i, j, nm), pre + nm))
                if a.get("head_gate", "off") != "off":
                    order.append((key(i, j, "g_proj"), pre + "g_proj"))
                body.append((i, j, "MLA", a))
            elif k == "MoE":
                for nm in ("router", "w1", "w3", "w2"):
                    order.append((key(i, j, nm), pre + nm))
                if a["router_bias"]:
                    order.append((key(i, j, "router.bias"), pre + "router.bias"))
                if a["expert_bias"]:
                    for nm in ("b1", "b3", "b2"):
                        order.append((key(i, j, nm), pre + nm))
                if a["shared"]:
                    for nm in ("sw1", "sw3", "sw2"):
                        order.append((key(i, j, nm), pre + nm))
                    if a["shared_gate"]:
                        order.append((key(i, j, "shared_gate"), pre + "shared_gate"))
                body.append((i, j, "MoE", a))
            elif k == "External":
                # **逃生舱第二层：C 后端链接一个外部符号。**
                # 编译器不认识这个机制 —— 只认 ABI（见 by1ext.py）。
                # 所以加一个新机制不用动这个文件。
                # 参数名要带 `ext:` 前缀 —— `shapes_of` 就是这么命名的，
                # 名字对不上会在后面 `params[pname]` 那里 KeyError。
                for nm in sorted(a["weights"]):
                    # 第二个元素是**完整的参数键**（`pre` = `layers.N.opM.`），
                    # 不是短名 —— 少了前缀会在后面 `params[pname]` 那里 KeyError，
                    # 而那个报错长得像"这个张量没声明"。
                    order.append((key(i, j, "ext:" + nm), pre + "ext:" + nm))
                body.append((i, j, "External", a))
            elif k == "Add":
                body.append((i, j, "Add", a))
            else:
                raise SystemExit(f"  [不支持] C 后端还没有 {k}")

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
                lb.append(f" {B}({'qf' if a['q_gate'] else 'q'}, {src[0]},"
                          f" P({key(i,j,'wq')}){BA('wq')}, n, D, {qw});")
                if a["q_gate"]:
                    lb.append(f" split_qg(q, gt, qf, n, {nh}, {hd});")
                lb.append(f" {B}(kk, {src[0]}, P({key(i,j,'wk')}){BA('wk')}, n, D, {nkv*hd});")
                lb.append(f" {B}(vv, {src[0]}, P({key(i,j,'wv')}){BA('wv')}, n, D, {nkv*hd});")
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
                          f" {'P(' + key(i,j,'sink') + ')' if a['sink'] else 'NULL'});")
                if a["q_gate"]:
                    lb.append(f" apply_qg(ob, gt, n, {nh*hd});")
                lb.append(f" {B}({dst}, ob, P({key(i,j,'wo')}){BA('wo')}, n, {nh*hd}, D);")
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
            elif k == "MLA":
                hg = a.get("head_gate", "off") != "off"
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
                    f" {1 if a.get('gate_act') == 'sigmoid' else 0});")
            elif k == "MoE":
                E, K, hid = a["experts"], a["top_k"], a["hidden"]
                sh = a["shared"]
                lb.append(
                    f" moe({dst}, {src[0]}, n, D, {E}, {K}, {hid},"
                    f" P({key(i,j,'router')}),"
                    f" {'P(' + key(i,j,'router.bias') + ')' if a['router_bias'] else 'NULL'},"
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
                    f" {1 if a['routing'] == 'topk_softmax' else 0},"
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 C 后端")
    ap.add_argument("by1")
    ap.add_argument("--gcc", required=True)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--workdir", default="cgen")
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    ex_mod = load("by1exec")
    name = os.path.basename(args.by1)
    _r, info = bc.check(args.by1)
    ir = cg.compile_ir(info)

    print(f"\n{'='*72}\n  {name}  ->  C\n{'='*72}")
    try:
        decls, order, total = emit_c(ir, info, ex_mod.shapes_of(ir))
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

    src = C_HEAD + "\n" + decls + "\n" + C_MAIN
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
