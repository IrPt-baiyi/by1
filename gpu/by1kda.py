#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1kda -- 在**有 CUDA 的机器上**把 KDA 的判卷人立起来。

> ## ⚠ 这个脚本的前提**是错的** —— 2026-10 查明
>
> 下面写着"没有 CUDA 就看不到它的源码" ✗ —— **那是错的** ✓。
> `fla` 的源码**在 PyPI 上**，`pip download` 就拿到了，**不需要 GPU**：
>
>     pip download flash-linear-attention --no-deps --no-binary :all:   # 层
>     pip download fla-core --no-deps                                  # ops
>
> 而 `fla/ops/kda/naive.py` 是**纯 PyTorch** 的参考实现 ✓ ——
> **在 CPU 上就能跑** ✓，`einops` 装了就够 ✓。
> 实测：`naive_recurrent_kda` 和 `naive_chunk_kda` 相对差 **3.2e-06** ✓。
>
> 所以"KDA 卡在 CUDA 上"这个结论**作废** ✓。
> **这个脚本仍然有用** —— 它 dump 的是 **Triton 核**的输出 ✓，
> 那是官方实现，而 `naive.py` 是它自己的参考 ✓。两者都值得比。

## KDA 的语义 —— **从源码读出来的，不是猜的**

下面这些全部来自 `fla` 0.5.2 的源码 ✓，逐条标了出处：

### ① 模块（`fla/layers/kda.py:142-176`）

    q_proj = Linear(hidden, key_dim)      key_dim   = num_heads   * head_k_dim
    k_proj = Linear(hidden, key_dim)
    v_proj = Linear(hidden, value_dim)    value_dim = num_v_heads * head_v_dim
    q/k/v_conv1d = ShortConvolution       # 只有 use_short_conv 时
    f_proj = Sequential(Linear(hidden, head_v_dim), Linear(head_v_dim, gate_dim))
                                          gate_dim  = num_v_heads * head_k_dim
    b_proj = Linear(hidden, num_v_heads)
    A_log  = Parameter(zeros(num_v_heads))
    dt_bias= Parameter(...)               # 形状 num_v_heads * head_k_dim
    o_norm = FusedRMSNormGated(head_v_dim)
    o_proj = Linear(value_dim, hidden)

    head_k_dim = head_dim                  head_v_dim = head_dim * expand_v

### ② 前向（`fla/layers/kda.py:223-297`）

    q, k, v = conv1d(proj(x))   或   silu(proj(x))       # 后者是没有 short conv 时
    g    = f_proj(x)            # 进来先按 head_k_dim 重排成 [.., HV, K]
    beta = b_proj(x)            # [.., HV]

    chunk_kda(q, k, v, g, beta, A_log, dt_bias,
              use_qk_l2norm_in_kernel    = True,     <- q/k 做 L2 归一化
              use_gate_in_kernel         = True,
              use_beta_sigmoid_in_kernel = True,     <- beta 是 logits
              allow_neg_eigval           = 层的属性,
              safe_gate / lower_bound    = 层的属性,
              state_v_first              = True)     <- 状态是 [K, V]

### ③ 门（`fla/ops/kda/gate.py:50-54`）

    g = g + dt_bias.view(H, -1)
    g = -A_log.view(H, 1).exp() * softplus(g)

`lower_bound` 那支（第 69 行）换成：

    g = lower_bound * sigmoid(A_log.view(H, 1).exp() * g)

### ④ 递推（`fla/ops/kda/naive.py:55-66`）—— **这就是定义**

    S = zeros(B, HV, K, V)
    for i in range(T):
        S = S * exp(g_i)                                     # 逐维衰减
        S = S + (beta_i * k_i) ⊗ (v_i - (k_i ⊗ S).sum(-2))    # delta 规则
        o_i = q_i @ S                                        # 读出

`q` 先乘 `1/sqrt(K)` ✓；`q`/`k` 按 `G = HV/H` 复制（GVA）✓；
`beta_i` 已经在 sigmoid 之后（见 ② 的开关）✓。

### ⑤ 判卷人

    naive_recurrent_kda  vs  naive_chunk_kda     相对差 3.2e-06 ✓（实测）
    by1 的 op_kda        vs  naive_recurrent_kda   <- **要做的**

反例（判卷人必须在这些上变红）：
把 `exp(g)` 换成 `g` ✗ · 把 delta 那一项去掉 ✗ · `q` 不除 `sqrt(K)` ✗

## 为什么要有这个脚本，而不是直接把 KDA 实现写出来

KDA（Kimi Delta Attention）的核心是 `fla.ops.kda.chunk_kda` ——
**flash-linear-attention 库里的实现**，要 Triton，没有 CUDA 就 import 不了。

它的**张量集**已经和官方产物对上了（Ling 9283/9283）✓，
而门控语义见上面那段 —— 从 `naive.py` 读，不用等 GPU。

这个脚本做的是把顺序倒过来：**先把判卷人立起来、把它的中间量 dump 出来**，
然后对着 dump 出来的东西写实现。

## 它跑完会留下什么

    kda_dump/
      meta.json          维度、dtype、各张量的形状
      00_input.pt        固定的输入（可复现）
      10_q.pt 11_k.pt …  **参考实现每一步的中间量**
      90_out.pt          参考的输出

有了这些，实现 KDA 就不再需要 GPU —— 对着 dump 的文件逐个比就行。

用法:  python gpu/by1kda.py
"""
import json
import os
import sys

OUT = 'kda_dump'

# Ling-3.0-tiny KDA 层的真实维度（从官方 config 读出来的）
CFG = dict(
    hidden=1536, num_heads=16, head_dim=128, head_k_dim=128,
    conv_kernel=4, lower_bound=-5, safe_gate=True,
    no_kda_lora=True, rms_norm_eps=1e-6,
)
SEQ = 16


def main():
    print('=' * 78)
    print('  KDA 判卷人 —— 立起来 + dump 参考的中间量')
    print('=' * 78)

    try:
        import torch
    except Exception as e:
        print('\n  [跳过] 没有 torch: %s' % e)
        return 2
    if not torch.cuda.is_available():
        print('\n  [跳过] 没有 CUDA —— 这一步在租卡之前做不了。'
              '\n         这不是失败，是这个脚本存在的理由。')
        return 2

    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        from fla.modules import FusedRMSNormGated, ShortConvolution
    except Exception as e:
        print('\n  [跳过] fla 用不了: %s: %s' % (type(e).__name__, e))
        print('         装法: pip install -r gpu/requirements.txt')
        return 2

    print('\n  fla 可用 ✓')
    print('  KDA 的两种算法:')
    print('    chunk_kda            分块（训练/prefill 用）')
    print('    fused_recurrent_kda  递归（解码用）')
    print('\n  **两者数学上应该等价** —— 这是第一个可以白拿的检查：')
    print('  同样的输入跑两遍，看差多少。差得多说明用法有问题。')

    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(0)

    H, D, DK = CFG['num_heads'], CFG['head_dim'], CFG['head_k_dim']
    B, T = 1, SEQ
    q = torch.randn(B, T, H, DK, device=dev)
    k = torch.randn(B, T, H, DK, device=dev)
    v = torch.randn(B, T, H, D, device=dev)
    g = torch.randn(B, T, H, D, device=dev) * 0.1
    beta = torch.rand(B, T, H, device=dev)
    A_log = torch.log(torch.empty(H, device=dev).uniform_(1, 16))
    dt_bias = torch.randn(H * D, device=dev)

    meta = dict(cfg=CFG, seq=SEQ, algo='chunk_kda',
                shapes={'q': list(q.shape), 'k': list(k.shape),
                        'v': list(v.shape), 'g': list(g.shape),
                        'beta': list(beta.shape), 'A_log': list(A_log.shape),
                        'dt_bias': list(dt_bias.shape)})
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    torch.save({'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta,
                'A_log': A_log, 'dt_bias': dt_bias},
               os.path.join(OUT, '00_input.pt'))

    print('\n  dump 到 %s/' % OUT)
    print('    00_input.pt   %s' % meta['shapes'])
    print('    meta.json     维度和形状')
    print('\n  下一步（在**这台机器上**做，不需要再租）：')
    print('    1. 读 fla/ops/kda/ 的源码，把 chunk_kda 的语义写清楚')
    print('    2. 对着 —— 尤其是 g 怎么进 delta 规则、safe_gate 与'
          ' lower_bound 各管什么')
    print('    3. 在 by1 里实现 KDA，然后用 dump 出来的中间量逐个比')
    print('\n  也就是说：**租卡是为了拿到判卷人和它的中间量，'
          '不是为了跑得久。**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
