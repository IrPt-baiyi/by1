
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

/* qk_norm：按头做 RMSNorm，在 RoPE 之前 */
static void qk_norm(float *q, float *k, const float *qn, const float *kn,
                    int n, int nh, int nkv, int hd, int one_plus) {
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < nh; h++) {
            float *v = q + ((size_t)t * nh + h) * hd;
            float s = 0.f;
            for (int i = 0; i < hd; i++) s += v[i] * v[i];
            s = 1.f / sqrtf(s / hd + 1e-5f);
            for (int i = 0; i < hd; i++) v[i] *= s * (one_plus ? (1.f + qn[i]) : qn[i]);
        }
        for (int h = 0; h < nkv; h++) {
            float *v = k + ((size_t)t * nkv + h) * hd;
            float s = 0.f;
            for (int i = 0; i < hd; i++) s += v[i] * v[i];
            s = 1.f / sqrtf(s / hd + 1e-5f);
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

extern void weird_fwd(const float *x, float *y, int B, int T, int D, const float *const *w, int nw);
extern void weird_fwd(const float *x, float *y, int B, int T, int D, const float *const *w, int nw);
#define D_MODEL 16
#define VOCAB 64
#define N_LAYER 2
#define Q_DIM 1
#define KV_DIM 1
#define NORM_ONE_PLUS 0
#define RMS_EPS 1e-05f
#define HID_MAX 1
#define OB_MAX 1
#define N_OPS 3
#define OFF_EMBED 0
#define LEN_EMBED 1024
#define OFF_FINAL_NORM 1024
#define LEN_FINAL_NORM 16
#define OFF_HEAD 1040
#define LEN_HEAD 1024
#define OFF_L0_OP0_W 2064
#define LEN_L0_OP0_W 16
#define OFF_L0_OP1_EXT_BIAS 2080
#define LEN_L0_OP1_EXT_BIAS 16
#define OFF_L0_OP1_EXT_SCALE 2096
#define LEN_L0_OP1_EXT_SCALE 16
#define OFF_L1_OP0_W 2112
#define LEN_L1_OP0_W 16
#define OFF_L1_OP1_EXT_BIAS 2128
#define LEN_L1_OP1_EXT_BIAS 16
#define OFF_L1_OP1_EXT_SCALE 2144
#define LEN_L1_OP1_EXT_SCALE 16
#define TOTAL_W 2160
#define LAYER_BODY(L) switch (L) { case 0: {  rmsnorm(t0, x, P(L0_OP0_W), n, D, 0, 1e-05f);  { const float *w_[2] = {P(L0_OP1_EXT_BIAS), P(L0_OP1_EXT_SCALE)}; weird_fwd(t0, t1, (int)n, 1, (int)D, w_, 2); }  for (size_t z = 0; z < (size_t)n * D; z++) t2[z] = x[z] + t1[z];  memcpy(x, t2, sizeof(float) * (size_t)n * D); break; } case 1: {  rmsnorm(t0, x, P(L1_OP0_W), n, D, 0, 1e-05f);  { const float *w_[2] = {P(L1_OP1_EXT_BIAS), P(L1_OP1_EXT_SCALE)}; weird_fwd(t0, t1, (int)n, 1, (int)D, w_, 2); }  for (size_t z = 0; z < (size_t)n * D; z++) t2[z] = x[z] + t1[z];  memcpy(x, t2, sizeof(float) * (size_t)n * D); break; } default: break; }

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

    for (int L = 0; L < N_LAYER; L++) {
        /* 每个 op 一个作用域：OFF_/LEN_ 宏按层号拼出来 */
        LAYER_BODY(L)
    }

    rmsnorm(xn, x, P(FINAL_NORM), n, D, NORM_ONE_PLUS, RMS_EPS);
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
