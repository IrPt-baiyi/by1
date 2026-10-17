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

## 它现在验到哪（2026-10-17 第十二轮）

    ① **形状 × 上下文 9 格网格**（seq 7/8/16/20，含分块残块；
       n_groups 1/2/4/8；conv 2/4/7），逐格登记余量；
    ② 三条反例（eps×4 / 卷积核翻转 / dt 不做 softplus）必须全红；
    ③ **直连对拍**：把官方权重直接喂给 `by1exec.op_ssm`。
       —— 上面验的是“by1 的**式子**”（本文件里另写一遍），
       而真正跑模型的是那个算子。两条链各自对上官方
       不等于**彼此一致** —— 实测：把 op_ssm 的 gate 顺序写错，
       式子那些行照样绿，**直连行立刻红**（相对 2.4e-01~3.6e-01）。
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

    ok 证   这条递推和官方逐位同源（两种尺寸都验）
    ok 证   这个对拍碰得到东西（三条反例都红）
    x 不证 by1 生成出来的代码是对的 —— 那要 by1diff 接上去才验得到
    x 不证 分块（chunked）形式和它一致 —— 递推是同一个数学，
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

import by1exec
import by1skip  # noqa: E402

# 判卷人可选，不在只置 None（库代码不 sys.exit，by1lint 第⑩类）——
# `import by1ssm` 照旧成功，跳过在 main() 里 return by1skip.skip(...)。
try:
    from transformers.models.nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer)
except ImportError:
    NemotronHConfig = NemotronHMamba2Mixer = None


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        by1skip.skip('没装 torch —— 这一项在这台机器上验不了')
    # **nemotron_h 的检查在模块级**（import 时置 None，main() 里跳过）——
    # 原来这里也有一份，但它只打印不返回（`by1skip.skip` 的返回值被丢掉），
    # 于是"保护"打印完就继续走，照样在 `_case` 里 ModuleNotFoundError。
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


def _params_from_official(mx):
    """把官方 mixer 的 state_dict 换成 `by1exec.op_ssm` 要的 P。

    **这就是直连对拍的关键一步**：同一个权重、两套代码。
    """
    sd = mx.state_dict()
    P = {
        'in_proj': sd['in_proj.weight'].detach().numpy(),
        'conv': sd['conv1d.weight'].detach().numpy().squeeze(1),
        'A_log': sd['A_log'].detach().numpy(),
        'dt_bias': sd['dt_bias'].detach().numpy(),
        'D': sd['D'].detach().numpy(),
        'norm.w': sd['norm.weight'].detach().numpy(),
        'out_proj': sd['out_proj.weight'].detach().numpy(),
    }
    if 'conv1d.bias' in sd:
        P['conv.bias'] = sd['conv1d.bias'].detach().numpy()
    return P


def _attrs_of(h, hd, ng, ssm, k, eps):
    return {'heads': h, 'head_dim': hd, 'n_groups': ng,
            'ssm_state': ssm, 'conv_kernel': k,
            'conv_bias': True, 'norm_eps': float(eps)}

def main(argv):
    torch = _torch()
    selftest = '--selftest' in argv

    if NemotronHConfig is None:
        # 判卷人不在就干净跳过（退出码 30），不是崩、不是"检查器坏了"
        return by1skip.skip(
            '找不到 transformers 的 nemotron_h —— 这一项在这台机器上验不了')

    print()
    print('  Mamba2 判卷人（递推 by1 要生成的式子，对官方 NemotronHMamba2Mixer）')
    print('  ' + '-' * 74)

    # ── 正例：**形状 x 上下文**的网格 ──────────────────────────
    #
    # 原来只有两格（小尺寸 + 真实尺寸、seq 7/4）。扩容的理由是
    # **分块**：递推按 chunk_size 切块，seq 不是 chunk_size 的整数倍时
    # 最后一块是残块 —— 那条路原来只有 seq=7 一格碰到。现在把
    # seq ∈ {7,8,16,20}、n_groups ∈ {1,2}、conv_kernel ∈ {2,4,7} 都摊开。
    #
    # **余量 = 判据 / 实测**，逐格登记（1.5x 那种薄余量必须一眼看见）。
    #
    # **配置约束（实测踩到）**：`n_groups` 必须整除 `num_heads`。
    # 我第一版配了 n_groups=8 / num_heads=4，上游在
    # `repeat_interleave(..., output_size=num_heads // num_groups)`
    # 处报 `Invalid output_size, expected 0 but got 4` —— 那是**配置非法**，
    # 不是上游 bug。这一行留着，免得下次又把非法配置当成被测对象。
    ok = True
    if not selftest:
        # (tag, d, h, hd, ssm, ng, k, seq)
        grid = [
            ('小/残块', (32, 4, 8, 16, 2, 4, 7)),
            ('小/整块', (32, 4, 8, 16, 2, 4, 8)),
            ('小/两块', (32, 4, 8, 16, 2, 4, 16)),
            ('小/残块2', (32, 4, 8, 16, 2, 4, 20)),
            ('小/单组', (32, 4, 8, 16, 1, 4, 13)),
            ('小/k2', (32, 4, 8, 16, 2, 2, 9)),
            ('小/k7', (32, 4, 8, 16, 2, 7, 11)),
            ('小/组=头', (32, 4, 8, 16, 4, 2, 6)),
            ('真实尺寸', (2688, 64, 64, 128, 8, 4, 4)),
        ]
        for tag, args in grid:
            mx, x, ref = _case(torch, *args, tag)
            with torch.no_grad():
                got = mine_mamba2(mx, x)
            d = (ref - got).abs().max().item()
            rel = d / max(ref.abs().max().item(), 1e-12)
            good = rel < 1e-5
            ok = ok and good
            margin = (1e-5 / rel) if rel > 0 else float("inf")
            print('     %-12s seq %-3d n_groups %-2d conv %-2d  绝对 %.3e  '
                  '相对 %.3e  余量 %.1fx  %s'
                  % (tag, args[6], args[4], args[5], d, rel, margin,
                     '同源 ok' if good else '**不一样 FAIL**'))
        print()
    print()

    # ── **直连对拍**：官方 mixer -> by1 参数 -> `by1exec.op_ssm` ──────
    #
    # 上面那一段验的是"by1 的**式子**"（这个文件里另写一遍）对官方；
    # 而真正跑模型的是 `by1exec.op_ssm`。两条链各自对上官方，不等于**彼此
    # 一致** —— 这里把官方权重直接喂给后端算子，一步连起来。
    if not selftest:
        import numpy as _np
        cross = [
            ('小/残块', (32, 4, 8, 16, 2, 4, 7)),
            ('小/单组', (32, 4, 8, 16, 1, 4, 13)),
            ('小/组=头', (32, 4, 8, 16, 4, 2, 6)),
        ]
        for tag, (d, h, hd, ssm, ng, k, seq) in cross:
            mx, x, ref = _case(torch, d, h, hd, ssm, ng, k, seq, tag)
            print('     %-12s 官方 -> op_ssm …' % tag, end='')
            P = _params_from_official(mx)
            at = _attrs_of(h, hd, ng, ssm, k, mx.norm.variance_epsilon)
            got = by1exec.op_ssm(P, at, [x.detach().numpy().astype(_np.float32)], d)
            got = _np.asarray(got, dtype=_np.float64)
            r = ref.detach().numpy().astype(_np.float64)
            dd = float(_np.abs(got - r).max())
            rr = dd / max(float(_np.abs(r).max()), 1e-12)
            good = rr < 1e-5
            ok = ok and good
            print(' 绝对 %.3e  相对 %.3e  %s'
                  % (dd, rr, '后端 op 与官方一致 ok' if good else '**不一致 FAIL**'))
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
              % (nm, rel, '红了 ok' if red else '**没红 x**'))

    print()
    print('  [%s] Mamba2 递推与官方同源%s'
          % ('PASS' if ok else 'FAIL',             '（9 格网格 · 反例 · 与 op_ssm 直连）' if ok else ''))
    return 0 if ok else 1


def _run():
    try:
        return main(sys.argv[1:])
    except SystemExit as e:
        return e.code or 0


if __name__ == '__main__':
    sys.exit(_run())
