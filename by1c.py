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

HERE = os.path.dirname(os.path.abspath(__file__))
MANGLE = re.compile(r"[^0-9A-Za-z]")


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

static void rmsnorm(float *o, const float *x, const float *w, int n, int d, int op) {
    for (int i = 0; i < n; i++) {
        const float *r = x + (size_t)i * d;
        float s = 0.f;
        for (int j = 0; j < d; j++) s += r[j] * r[j];
        s = 1.0f / sqrtf(s / d + 1e-5f);
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

static void silu_(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* pairing: 0 = half, 1 = interleaved */
static void rope(float *q, float *k, int n, int nh, int nkv, int hd,
                 float base, int pairing, float scale) {
    for (int pos = 0; pos < n; pos++) {
        for (int h = 0; h < nh + nkv; h++) {
            int isq = h < nh;
            if (!isq && h - nh >= nkv) continue;
            float *v = isq ? q + ((size_t)pos * nh + h) * hd
                           : k + ((size_t)pos * nkv + (h - nh)) * hd;
            if (pairing == 0) {
                int half = hd / 2;
                for (int i = 0; i < half; i++) {
                    float f = powf(base, -2.0f * i / hd) * pos;
                    float c = cosf(f) * scale, s = sinf(f) * scale;
                    float a = v[i], b = v[i + half];
                    v[i] = a * c - b * s;
                    v[i + half] = a * s + b * c;
                }
            } else {
                for (int i = 0; i < hd / 2; i++) {
                    float f = powf(base, -2.0f * i / hd) * pos;
                    float c = cosf(f) * scale, s = sinf(f) * scale;
                    float a = v[2 * i], b = v[2 * i + 1];
                    v[2 * i] = a * c - b * s;
                    v[2 * i + 1] = a * s + b * c;
                }
            }
        }
    }
}

static void attention(float *o, float *q, const float *k, const float *v,
                      int n, int nh, int nkv, int hd, int od, int window) {
    int rep = nh / nkv;
    float sc = 1.0f / sqrtf((float)hd);
    float *att = (float *)malloc(sizeof(float) * n);
    float *vb = (float *)malloc(sizeof(float) * hd);
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
            float sum = 0.f;
            for (int j = 0; j <= i; j++) {
                if (window && j <= i - window) continue;
                att[j] = expf(att[j] - mx);
                sum += att[j];
            }
            float *oi = o + ((size_t)i * nh + h) * hd;
            for (int t = 0; t < hd; t++) vb[t] = 0.f;
            for (int j = 0; j <= i; j++) {
                if (window && j <= i - window) continue;
                float a = att[j] / sum;
                const float *vj = v + ((size_t)j * nkv + kh) * hd;
                for (int t = 0; t < hd; t++) vb[t] += a * vj[t];
            }
            memcpy(oi, vb, sizeof(float) * hd);
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

    rmsnorm(xn, x, P(FINAL_NORM), n, D, NORM_ONE_PLUS);
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


def emit_c(ir, info, params):
    """把 IR 展开成 C。**不支持就报错，绝不悄悄生成错的。**"""
    d = ir["d_model"]
    lines, need = [], {}
    att = next((o["attrs"] for L in ir["layers"] for o in L["ops"]
                if o["kind"] == "Attention"), None)
    if att is None:
        raise SystemExit("  这个后端目前只支持带全量注意力的模型")

    def key(i, j, nm):
        return f"L{i}_OP{j}_{MANGLE.sub('_', nm).upper()}"

    # 收集参数（顺序确定，C 与 Python 用同一份）
    order = [("EMBED", "embed.weight"), ("FINAL_NORM", "final_norm.w"),
             ("HEAD", "head.weight")]
    body = []
    for L in ir["layers"]:
        i = L["index"]
        for j, o in enumerate(L["ops"]):
            k, a, pre = o["kind"], o["attrs"], f"layers.{i}.op{j}."
            if k == "Norm":
                order.append((key(i, j, "w"), pre + "w"))
            elif k == "Attention":
                if a["q_gate"] or a["kv_tie"] or a["sink"] or a["bias"] \
                        or a["qk_norm"] or a["head_gate"] != "off" \
                        or a.get("yarn") or a.get("rope_partial", 1.0) != 1.0:
                    raise SystemExit(
                        f"  [不支持] 第 {i} 层的注意力用了 q_gate/kv_tie/sink/"
                        f"bias/qk_norm/head_gate/yarn/partial —— C 后端还没实现")
                for nm in ("wq", "wk", "wv", "wo"):
                    order.append((key(i, j, nm), pre + nm))
                body.append((i, j, "Attention", a))
            elif k == "FFN":
                if not a.get("gate", True) or a.get("act") == "gptoss":
                    raise SystemExit(f"  [不支持] 第 {i} 层的 FFN 形态")
                for nm in ("w1", "w2", "w3"):
                    order.append((key(i, j, nm), pre + nm))
                body.append((i, j, "FFN", a))
            elif k == "Add":
                body.append((i, j, "Add", a))
            else:
                raise SystemExit(f"  [不支持] C 后端还没有 {k}")

    lines.append("#define D_MODEL %d" % d)
    lines.append("#define VOCAB %d" % ir["vocab"])
    lines.append("#define N_LAYER %d" % len(ir["layers"]))
    lines.append("#define Q_DIM %d" % (att["q"] * att["head_dim"]))
    lines.append("#define KV_DIM %d" % (att["kv"] * att["head_dim"]))
    lines.append("#define NORM_ONE_PLUS %d" % (1 if ir.get("norm_one_plus") else 0))
    _hids = [o["attrs"]["hidden"] for L in ir["layers"] for o in L["ops"]
             if o["kind"] in ("FFN", "MoE")]
    lines.append("#define HID_MAX %d" % (max(_hids) if _hids else 1))
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
                lb.append(f" rmsnorm({dst}, {src[0]}, P({key(i,j,'w')}), n, D,"
                          f" {1 if a.get('one_plus') else 0});")
            elif k == "Attention":
                hd, nh, nkv = a["head_dim"], a["q"], a["kv"]
                lb.append(f" linear(q, {src[0]}, P({key(i,j,'wq')}), n, D, {nh*hd});")
                lb.append(f" linear(kk, {src[0]}, P({key(i,j,'wk')}), n, D, {nkv*hd});")
                lb.append(f" linear(vv, {src[0]}, P({key(i,j,'wv')}), n, D, {nkv*hd});")
                lb.append(f" rope(q, kk, n, {nh}, {nkv}, {hd},"
                          f" {float(a['rope_base'])}f,"
                          f" {0 if a['rope_pairing']=='half' else 1},"
                          f" {float(a['rope_scale'])}f);")
                lb.append(f" attention(ob, q, kk, vv, n, {nh}, {nkv}, {hd},"
                          f" {nh*hd}, {a['window'] if a['window'] else 0});")
                lb.append(f" linear({dst}, ob, P({key(i,j,'wo')}), n, {nh*hd}, D);")
            elif k == "FFN":
                hid = a["hidden"]
                lb.append(f" linear(hb, {src[0]}, P({key(i,j,'w1')}), n, D, {hid});")
                lb.append(f" silu_(hb, (size_t)n * {hid});")
                lb.append(f" linear(ob, {src[0]}, P({key(i,j,'w3')}), n, D, {hid});")
                lb.append(f" for (size_t z = 0; z < (size_t)n * {hid}; z++)"
                          f" hb[z] *= ob[z];")
                lb.append(f" linear({dst}, hb, P({key(i,j,'w2')}), n, {hid}, D);")
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
    open(os.path.join(args.workdir, "w.bin"), "wb").write(flat.tobytes())
    ids = rng.integers(0, ir["vocab"], args.seq).astype(np.int32)
    open(os.path.join(args.workdir, "ids.bin"), "wb").write(ids.tobytes())

    src = C_HEAD + "\n" + decls + "\n" + C_MAIN
    cpath = os.path.join(args.workdir, "model.c")
    open(cpath, "w", encoding="utf-8").write(src)
    print(f"  生成 {cpath}  （{len(src.splitlines())} 行）")

    exe = os.path.join(args.workdir, "model.exe")
    p = subprocess.run([args.gcc, "-O2", "-o", exe, cpath, "-lm"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("  [编译失败]")
        print((p.stderr or "")[:1600])
        return 1
    print("  [编译成功]")

    r = subprocess.run([exe, str(args.seq),
                        os.path.join(args.workdir, "w.bin"),
                        os.path.join(args.workdir, "ids.bin"),
                        os.path.join(args.workdir, "logits.bin")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [运行失败] {r.stdout} {r.stderr}")
        return 1
    print(f"  [运行成功] {r.stdout.strip()}")
    got = np.fromfile(os.path.join(args.workdir, "logits.bin"), dtype=np.float32)
    got = got.reshape(args.seq, ir["vocab"])

    ex = ex_mod.Exec(ir, params)
    ex.P["_g"] = {k: params[k] for k in
                  ("embed.weight", "final_norm.w", "head.weight")}
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
