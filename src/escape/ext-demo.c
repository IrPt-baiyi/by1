/* ext-demo.c —— 逃生舱第二层的示例：两个语言里没有的机制。
 *
 * ABI（和 by1ext.py 里定的一模一样）：
 *
 *     void <symbol>(const float *x, float *y,
 *                   int B, int T, int D,
 *                   const float *const *w, int nw);
 *
 * 这里放**两个**符号。第二个是给"不用改编译器"那条主张做证的：
 * 加它的时候，by1codegen.py / by1exec.py / by1c.py / by1ir.py
 * **一个字都没动** —— 编译器不认识机制，只认识 ABI。
 *
 * 权重顺序是 ABI 的一部分：**按名字字典序**。
 * 两边各自算得出来，不靠额外通道传。
 *
 * 编译：
 *     gcc -O2 -shared -fPIC -o ext-demo.so ext-demo.c
 */
#include <stddef.h>
#include <math.h>

/* 机制一：逐通道缩放 + 偏置。两个权重。
 *     w[0] = bias  (D,)      "bias"  < "scale"
 *     w[1] = scale (D,)
 */
void weird_fwd(const float *x, float *y, int B, int T, int D,
               const float *const *w, int nw)
{
    if (nw < 2) return;
    const float *bias = w[0];
    const float *scale = w[1];
    size_t n = (size_t)B * (size_t)T;
    for (size_t t = 0; t < n; t++) {
        const float *xr = x + t * (size_t)D;
        float *yr = y + t * (size_t)D;
        for (int c = 0; c < D; c++)
            yr[c] = xr[c] * scale[c] + bias[c];
    }
}

/* 机制二：一个 2x2 的逐通道旋转混合。三个权重。
 *     w[0] = a (D,)   "a" < "b" < "theta"
 *     w[1] = b (D,)
 *     w[2] = theta (D,)
 *
 * **加这个符号的时候，编译器一个字都没改。** 这就是那一层的全部意义。
 */
void spin_fwd(const float *x, float *y, int B, int T, int D,
              const float *const *w, int nw)
{
    if (nw < 3) return;
    const float *a = w[0], *b = w[1], *th = w[2];
    size_t n = (size_t)B * (size_t)T;
    for (size_t t = 0; t < n; t++) {
        const float *xr = x + t * (size_t)D;
        float *yr = y + t * (size_t)D;
        for (int c = 0; c < D; c++) {
            float ct = cosf(th[c]), st = sinf(th[c]);
            yr[c] = ct * xr[c] + st * a[c] * xr[(c + 1) % D] + b[c];
        }
    }
}
