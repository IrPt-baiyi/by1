/* ggml-smoke.c —— M0 的判卷人：**ggml 在这台机器上算的是这个数学**。
 *
 * 不是"编译过了"就算数。三个小检查，全部对照**手算**的值：
 *   ① ggml_mul_mat   A(3x2) @ x(2) = [3, 7, 11]
 *   ② ggml_rms_norm  [3,4] -> [3,4] / sqrt((9+16)/2)
 *   ③ ggml_silu      [0, 2] -> [0, 2/(1+e^-2)]
 *
 * 编译（在仓库根）：
 *   gcc -O2 -fopenmp -o ggml-smoke.exe tools/ggml-smoke.c \
 *       -I llamacpp/llama.cpp/ggml/include \
 *       llamacpp/build-ggml/ggml/src/ggml-cpu.a \
 *       llamacpp/build-ggml/ggml/src/ggml.a \
 *       llamacpp/build-ggml/ggml/src/ggml-base.a \
 *       -lm -lstdc++ -lpthread
 *
 * **`-fopenmp` 不能少**：ggml-cpu.a 是 OpenMP 构建的，少了它链接报
 * 一堆 `GOMP_barrier` 未定义（2026-10-14 头注命令漂了，实测补上）。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"

static int near(float a, float b, float tol) { return fabsf(a - b) < tol; }

int main(void) {
    struct ggml_init_params ip = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    int bad = 0;

    /* ① mul_mat: A [ne0=2, ne1=3]（行主序 [3,2] 的同一批数） */
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    float a[6] = { 1, 2, 3, 4, 5, 6 };
    memcpy(A->data, a, sizeof(a));
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float xv[2] = { 1, 1 };
    memcpy(x->data, xv, sizeof(xv));
    struct ggml_tensor * y = ggml_mul_mat(ctx, A, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }
    float want[3] = { 3, 7, 11 };
    for (int i = 0; i < 3; i++) {
        float got = ggml_get_f32_1d(y, i);
        if (!near(got, want[i], 1e-6f)) {
            printf("  [FAIL] mul_mat[%d] = %.6f, 手算 %.6f\n", i, got, want[i]);
            bad = 1;
        }
    }

    /* ② rms_norm: v=[3,4] -> v / sqrt(12.5)，权重恒 1 */
    struct ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    float vv[2] = { 3, 4 };
    memcpy(v->data, vv, sizeof(vv));
    struct ggml_tensor * r = ggml_rms_norm(ctx, v, 0.0f);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }
    float s = 1.0f / sqrtf(12.5f);
    if (!near(ggml_get_f32_1d(r, 0), 3 * s, 1e-6f) ||
        !near(ggml_get_f32_1d(r, 1), 4 * s, 1e-6f)) {
        printf("  [FAIL] rms_norm = [%.6f, %.6f], 手算 [%.6f, %.6f]\n",
               ggml_get_f32_1d(r, 0), ggml_get_f32_1d(r, 1), 3 * s, 4 * s);
        bad = 1;
    }

    /* ③ silu: [0,2] -> [0, 2/(1+e^-2)] */
    struct ggml_tensor * u = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float uv[2] = { 0, 2 };
    memcpy(u->data, uv, sizeof(uv));
    struct ggml_tensor * z = ggml_silu(ctx, u);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, z);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }
    float w2 = 2.0f / (1.0f + expf(-2.0f));
    if (!near(ggml_get_f32_1d(z, 0), 0.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(z, 1), w2, 1e-6f)) {
        printf("  [FAIL] silu = [%.6f, %.6f], 手算 [0, %.6f]\n",
               ggml_get_f32_1d(z, 0), ggml_get_f32_1d(z, 1), w2);
        bad = 1;
    }

    /* ── T16 第一批补口的神谕（④-⑪）：每个新映射一个手算对拍 ───────── */

    /* ④ exp: [0, 1] -> [1, e] */
    struct ggml_tensor * e4 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e4v[2] = { 0, 1 };
    memcpy(e4->data, e4v, sizeof(e4v));
    struct ggml_tensor * e4o = ggml_exp(ctx, e4);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e4o);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e4o, 0), 1.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e4o, 1), expf(1.0f), 1e-5f)) {
        printf("  [FAIL] exp\n"); bad = 1;
    }

    /* ⑤ sqrt: [4, 9] -> [2, 3] */
    struct ggml_tensor * e5 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e5v[2] = { 4, 9 };
    memcpy(e5->data, e5v, sizeof(e5v));
    struct ggml_tensor * e5o = ggml_sqrt(ctx, e5);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e5o);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e5o, 0), 2.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e5o, 1), 3.0f, 1e-6f)) {
        printf("  [FAIL] sqrt\n"); bad = 1;
    }

    /* ⑥ div: [3, 8] / [1.5, 2] -> [2, 4] */
    struct ggml_tensor * e6a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor * e6b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e6av[2] = { 3, 8 }, e6bv[2] = { 1.5f, 2 };
    memcpy(e6a->data, e6av, sizeof(e6av));
    memcpy(e6b->data, e6bv, sizeof(e6bv));
    struct ggml_tensor * e6o = ggml_div(ctx, e6a, e6b);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e6o);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e6o, 0), 2.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e6o, 1), 4.0f, 1e-6f)) {
        printf("  [FAIL] div\n"); bad = 1;
    }

    /* ⑦ add_scalar = add1（1 元素张量）: [1,2] + 0.5 -> [1.5, 2.5] */
    struct ggml_tensor * e7 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e7v[2] = { 1, 2 };
    memcpy(e7->data, e7v, sizeof(e7v));
    struct ggml_tensor * one1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    float onev[1] = { 0.5f };
    memcpy(one1->data, onev, sizeof(onev));
    struct ggml_tensor * e7o = ggml_add(ctx, e7, one1);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e7o);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e7o, 0), 1.5f, 1e-6f) ||
        !near(ggml_get_f32_1d(e7o, 1), 2.5f, 1e-6f)) {
        printf("  [FAIL] add_scalar\n"); bad = 1;
    }

    /* ⑧ clamp_min max(x,0) / clamp_max min(x,1): [-1,2] -> [0,2]；[2,-3] -> [1,-3] */
    struct ggml_tensor * e8 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e8v[2] = { -1, 2 };
    memcpy(e8->data, e8v, sizeof(e8v));
    struct ggml_tensor * e8o = ggml_clamp(ctx, e8, 0.0f, 1e38f);
    struct ggml_tensor * e8b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e8bv[2] = { 2, -3 };
    memcpy(e8b->data, e8bv, sizeof(e8bv));
    struct ggml_tensor * e8ob = ggml_clamp(ctx, e8b, -1e38f, 1.0f);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e8o);
    ggml_build_forward_expand(gf, e8ob);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e8o, 0), 0.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e8o, 1), 2.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e8ob, 0), 1.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e8ob, 1), -3.0f, 1e-6f)) {
        printf("  [FAIL] clamp_min/clamp_max\n"); bad = 1;
    }

    /* ⑨ log1p = log(add1(x, 1)): [0, e-1] -> [0, 1] */
    struct ggml_tensor * e9 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    float e9v[2] = { 0, 2.718281828f - 1 };
    memcpy(e9->data, e9v, sizeof(e9v));
    struct ggml_tensor * one9 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    float one9v[1] = { 1.0f };
    memcpy(one9->data, one9v, sizeof(one9v));
    struct ggml_tensor * e9o = ggml_log(ctx, ggml_add(ctx, e9, one9));
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e9o);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
    if (!near(ggml_get_f32_1d(e9o, 0), 0.0f, 1e-6f) ||
        !near(ggml_get_f32_1d(e9o, 1), 1.0f, 1e-5f)) {
        printf("  [FAIL] log1p\n"); bad = 1;
    }

    /* ⑩ softplus **稳定式**（与 by1ggml 发射的组合完全相同）：
        softplus(x) = max(x,0) + log1p(e^{-|x|})
        x = 0 -> log2；x = 2 -> 2+log1p(e^-2)；x = -2 -> log1p(e^-2) */
    struct ggml_tensor * e10 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    float e10v[3] = { 0, 2, -2 };
    memcpy(e10->data, e10v, sizeof(e10v));
    struct ggml_tensor * one10 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    float one10v[1] = { 1.0f };
    memcpy(one10->data, one10v, sizeof(one10v));
    {
        struct ggml_tensor * mx = ggml_clamp(ctx, e10, 0.0f, 1e38f);
        struct ggml_tensor * ax = ggml_add(ctx,
            ggml_clamp(ctx, ggml_neg(ctx, e10), 0.0f, 1e38f), mx);
        struct ggml_tensor * sp = ggml_add(ctx, mx, ggml_log(ctx,
            ggml_add(ctx, ggml_exp(ctx, ggml_neg(ctx, ax)), one10)));
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sp);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
        float spw[3] = { logf(2.0f),
                         2.0f + log1pf(expf(-2.0f)),
                         log1pf(expf(-2.0f)) };
        for (int i = 0; i < 3; i++) {
            if (!near(ggml_get_f32_1d(sp, i), spw[i], 1e-5f)) {
                printf("  [FAIL] softplus[%d] = %.6f, 手算 %.6f\n",
                       i, ggml_get_f32_1d(sp, i), spw[i]);
                bad = 1;
            }
        }
    }

    /* ⑪ sqrtsoftplus: x=0 -> sqrt(log2) */
    {
        struct ggml_tensor * u11 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        float u11v[1] = { 0 };
        memcpy(u11->data, u11v, sizeof(u11v));
        struct ggml_tensor * one11 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        float one11v[1] = { 1.0f };
        memcpy(one11->data, one11v, sizeof(one11v));
        struct ggml_tensor * mx = ggml_clamp(ctx, u11, 0.0f, 1e38f);
        struct ggml_tensor * ax = ggml_add(ctx,
            ggml_clamp(ctx, ggml_neg(ctx, u11), 0.0f, 1e38f), mx);
        struct ggml_tensor * ssp = ggml_sqrt(ctx, ggml_add(ctx, mx,
            ggml_log(ctx, ggml_add(ctx, ggml_exp(ctx, ggml_neg(ctx, ax)), one11))));
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ssp);
        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { return 1; }
        if (!near(ggml_get_f32_1d(ssp, 0), sqrtf(logf(2.0f)), 1e-5f)) {
            printf("  [FAIL] sqrtsoftplus = %.6f, 手算 %.6f\n",
                   ggml_get_f32_1d(ssp, 0), sqrtf(logf(2.0f)));
            bad = 1;
        }
    }

    ggml_free(ctx);
    printf(bad ? "[FAIL] ggml 手算对拍不一致\n"
               : "[PASS] mul_mat / rms_norm / silu / exp / sqrt / div / "
                 "add_scalar / clamp / log1p / softplus / sqrtsoftplus 与手算一致\n");
    return bad;
}
