#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ssm -- **Mamba2 的判卷人**：by1 要生成的递推 vs 官方 mixer。

## 它为什么存在

`by1check` 认了 `mech Mamba : SSM`，而 `by1codegen` 还不支持 ——
所以 Nemotron 那一族**描述得出来、算不出来**。

要给它加 `SSM`，第一件事是**把判卷人立起来**：官方
`NemotronHMamba2Mixer` 就在装着的 transformers 里，而它和 by1 要生成的
东西是同一个算法。

**而且不能拿 `MambaForCausalLM` 当判卷人** —— 纯 Mamba 的 `in_proj` 是
`intermediate_size * 2`，而 Nemotron 声明的是

    projection_size = intermediate_size + conv_dim + num_heads
                    = 4096 + 6144 + 64 = 10304

那 **`Nemotron-H` 那一版**的布局。判卷人换了就不是同一个判卷人。

## 它在验什么

    A  = -exp(A_log)                          逐头 (H,)
    dt = softplus(in_proj(x)[..., :H] + dt_bias)

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * (x_t ⊗ B_t)
    y_t = C_t · h_t + D * x_t

    B/C 按 n_groups 分组，每头 repeat_interleave 到 num_heads
    最后：gated RMSNorm（gate 是 silu(gate)，**乘在 norm 之前**）+ out_proj

**两处形状最容易写反**（都炸在维度上，不会静默算错）：

    状态是 [b, H, HD, SSM]
      · 更新是**外积**  x[HD] ⊗ B[SSM]
      · 读出是 **h 右乘 C**  h[HD,SSM] @ C[SSM,1]

## 它证明什么、不证明什么

    ✓ 证   这条递推和官方逐位同源（两种尺寸都验）
    ✓ 证   这个对拍碰得到东西（三条反例都红）
    ✗ 不证 by1 生成出来的代码是对的 —— 那要 by1diff 接上去才验得到
    ✗ 不证 分块（chunked）形式和它一致 —— 递推是同一个数学，
           但官方走的是 kernel 那条路，浮点归约顺序不同

用法：

    python src/modelcheck/by1ssm.py            # 全部
    python src/modelcheck/by1ssm.py --selftest # 只跑反例
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import by1skip  # noqa: E402


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        by1skip.skip('没装 torch —— 这一项在这台机器上验不了')
    try:
        from transformers.models.nemotron_h import (  # noqa: F401
            NemotronHConfig)
        from transformers.models.nemotron_h.modeling_nemotron_h import (  # noqa: F401,E501
            NemotronHMamba2Mixer)
    except ImportError:
        by1skip.skip('这个 transformers 里没有 NemotronH —— 判卷人不在')
    import torch
    return torch


def mine_mamba2(mx, x, *, eps=None, flip_conv=False, softplus=True):
    """Mamba2 的**递推形式** —— 要生成的就是它。

    `eps` / `flip_conv` / `softplus` 三个开关只为反例而存在：
    打开任意一个，结果必须偏离参考。**一个碰不到被测东西的测试不是测试。**
    """
    import torch
    import torch.nn.functional as F

    b, s, _ = x.shape
    H, HD = mx.num_heads, mx.head_dim
    SSM, NG = mx.ssm_state_size, mx.n_groups
    K = mx.conv_kernel_size
    d_inner = H * HD
    conv_dim = d_inner + 2 * NG * SSM

    proj = F.linear(x, mx.in_proj.weight, mx.in_proj.bias)
    gate, bc, dt = proj.split([d_inner, conv_dim, H], dim=-1)

    # ② 因果深度卷积 + 激活
    w = mx.conv1d.weight.squeeze(1)
    if flip_conv:
        w = w.flip(-1)
    bc = F.conv1d(bc.transpose(1, 2), w[:, None, :], mx.conv1d.bias,
                  groups=conv_dim, padding=K - 1)[:, :, :s]
    bc = F.silu(bc).transpose(1, 2)

    xs, B, C = bc.split([d_inner, NG * SSM, NG * SSM], dim=-1)
    xs = xs.view(b, s, H, HD)
    B = B.view(b, s, NG, SSM)
    C = C.view(b, s, NG, SSM)

    # ③ SSM 递推
    A = -torch.exp(mx.A_log.float())
    dtt = F.softplus(dt + mx.dt_bias) if softplus else (dt + mx.dt_bias)
    rep = H // NG
    Bx = B.repeat_interleave(rep, dim=2)
    Cx = C.repeat_interleave(rep, dim=2)

    h = x.new_zeros(b, H, HD, SSM)
    ys = []
    for t in range(s):
        dt_t = dtt[:, t][..., None, None]                     # [b,H,1,1]
        upd = xs[:, t][..., None] * Bx[:, t][:, :, None, :]   # 外积
        h = torch.exp(dt_t * A[None, :, None, None]) * h + dt_t * upd
        ys.append((h @ Cx[:, t][..., None]).squeeze(-1)       # h 右乘 C
                  + mx.D[None, :, None] * xs[:, t])
    y = torch.stack(ys, 1).reshape(b, s, d_inner)

    # ④ gated RMSNorm（gate 乘在 norm **之前**）
    y = y * F.silu(gate)
    gs = mx.norm.group_size
    g = y.view(b, s, d_inner // gs, gs)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True)
                        + (eps if eps is not None
                           else mx.norm.variance_epsilon))
    y = g.view(b, s, d_inner) * mx.norm.weight

    return F.linear(y, mx.out_proj.weight, mx.out_proj.bias)


def _case(torch, d, h, hd, ssm, ng, k, seq, tag):
    from transformers.models.nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer)
    torch.manual_seed(0)
    cfg = NemotronHConfig(
        vocab_size=64, hidden_size=d, num_hidden_layers=1,
        mamba_num_heads=h, mamba_head_dim=hd, ssm_state_size=ssm,
        n_groups=ng, conv_kernel=k, expand=2, chunk_size=8,
        use_conv_bias=True, use_bias=False, mamba_hidden_act='silu',
        layer_types=['linear_attention'])
    mx = NemotronHMamba2Mixer(cfg, layer_idx=0).eval()
    x = torch.randn(1, seq, d)
    with torch.no_grad():
        ref = mx(x)
    # **`proj = d_inner + conv_dim + H`，而 `conv_dim = d_inner + 2*ng*ssm`。**
    # 第一版打印写成 `d_inner + 2*ng*ssm + H` —— 中间那项漏了 `d_inner`，
    # 于是它印 `100` 而真值是 `132`。**打印出来的数字也是要核的。**
    conv_dim = h * hd + 2 * ng * ssm
    print('  %-22s d_inner %-5d conv_dim %-5d proj %-6d'
          % (tag, h * hd, conv_dim, h * hd + conv_dim + h))
    return mx, x, ref


def main(argv):
    torch = _torch()
    selftest = '--selftest' in argv

    print()
    print('  Mamba2 判卷人（递推 by1 要生成的式子，对官方 NemotronHMamba2Mixer）')
    print('  ' + '-' * 74)

    # ── 正例：两种尺寸 ──────────────────────────────────────────
    ok = True
    if not selftest:
        for tag, args in (('小尺寸', (32, 4, 8, 16, 2, 4, 7)),
                          ('真实尺寸', (2688, 64, 64, 128, 8, 4, 4))):
            mx, x, ref = _case(torch, *args, tag)
            with torch.no_grad():
                got = mine_mamba2(mx, x)
            d = (ref - got).abs().max().item()
            rel = d / max(ref.abs().max().item(), 1e-12)
            good = rel < 1e-5
            ok = ok and good
            print('     %-22s 绝对 %.3e  相对 %.3e  %s'
                  % ('', d, rel, '同源 ✓' if good else '**不一样 ✗**'))
        print()

    # ── 反例：证明这个对拍碰得到东西 ────────────────────────────
    print('  ── 反例（每一条都必须红 —— 否则这个判卷人碰不到被测的东西）')
    mx, x, ref = _case(torch, 32, 4, 8, 16, 2, 4, 7, '反例基准')
    for nm, kw in (('gated norm 的 eps 改成 4 倍',
                    dict(eps=mx.norm.variance_epsilon * 4)),
                   ('卷积核左右翻转', dict(flip_conv=True)),
                   ('dt 不做 softplus', dict(softplus=False))):
        with torch.no_grad():
            bad = mine_mamba2(mx, x, **kw)
        rel = ((ref - bad).abs().max().item()
               / max(ref.abs().max().item(), 1e-12))
        red = rel > 1e-5
        ok = ok and red
        print('     %-26s 相对 %.3e  %s'
              % (nm, rel, '红了 ✓' if red else '**没红 ✗**'))

    print()
    print('  [%s] Mamba2 递推与官方同源%s'
          % ('PASS' if ok else 'FAIL', '（反例也验过）' if ok else ''))
    return 0 if ok else 1


def _run():
    try:
        return main(sys.argv[1:])
    except SystemExit as e:
        return e.code or 0


if __name__ == '__main__':
    sys.exit(_run())
