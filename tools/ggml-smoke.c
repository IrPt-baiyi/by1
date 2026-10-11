/* ggml-smoke.c —— M0 的判卷人：**ggml 在这台机器上算的是这个数学**。
 *
 * 不是"编译过了"就算数。三个小检查，全部对照**手算**的值：
 *   ① ggml_mul_mat   A(3x2) @ x(2) = [3, 7, 11]
 *   ② ggml_rms_norm  [3,4] -> [3,4] / sqrt((9+16)/2)
 *   ③ ggml_silu      [0, 2] -> [0, 2/(1+e^-2)]
 *
 * 编译（在仓库根）：
 *   gcc -O2 -o ggml-smoke.exe tools/ggml-smoke.c \
 *       -I llamacpp/llama.cpp/ggml/include \
 *       llamacpp/build-ggml/ggml/src/ggml-cpu.a \
 *       llamacpp/build-ggml/ggml/src/ggml.a \
 *       llamacpp/build-ggml/ggml/src/ggml-base.a \
 *       -lm -lstdc++ -lpthread
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

    ggml_free(ctx);
    printf(bad ? "[FAIL] ggml 手算对拍不一致\n"
               : "[PASS] mul_mat / rms_norm / silu 与手算一致\n");
    return bad;
}
