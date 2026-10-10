#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1kda -- KDA 的判卷人：**by1 算的，对着 fla 的 `naive_recurrent_kda`**。

## 为什么是它

KDA 的语义**不是我猜的** ✓ —— 是从 `fla` 的源码里读出来的，
五处出处写在 `gpu/by1kda.py` 的 docstring 里 ✓。这个脚本把那份
"读出来的东西"变成一条**会跑的判据** ✓。

参照物是 `fla/ops/kda/naive.py` —— **纯 PyTorch，CPU 上跑得动** ✓。
（`fla` 那套 Triton 核要 CUDA ✓，**参考实现不要** ✓ ——
 这一点我一开始判断错了 ✗，`gpu/by1kda.py` 开头记着。）

## 它比什么

    同一组权重，两条路算同一个东西：

    by1 侧    `by1codegen` 生成出来的那个 `KDA` 模块，调**它自己的 forward**
    参照侧    `naive_recurrent_kda(q, k, v, g, beta)` —— 那五行

    S = S * exp(g_i)
    S = S + (beta_i * k_i) ⊗ (v_i - (k_i ⊗ S).sum(-2))
    o_i = q_i @ S

**前端的管线判卷人自己再写一遍**（卷积 · silu · 重排 · 门 · sigmoid）✓ ——
那是判卷人的活 ✓，不是被测物的活 ✓。

## 反例（每一条都必须红 —— 否则这个判卷人碰不到被测的东西）

    ① 门退化成**每头一个标量**（GDN 的写法）✗ —— KDA 是逐维的
    ② `dt_bias` 按**头**而不是按**维** ✗
    ③ `q/k` 不做 **L2 归一化** ✗

## 这个判卷人**碰不到**的地方（写出来，免得它被当成全覆盖）

比的是**递推那一步的输出** —— 所以被真正判到的只有：

    投影 · 短卷积 · silu · 重排 · 门 · sigmoid → **递推**

**没判到的有两处** ✓，而两处都属于"写错了不报错、只算错"那一类 ✗：

    ⓐ 输出门 `g_proj` / `o_norm` / `o_proj` —— 整条尾巴一次都没比过 ✗
    ⓑ `front()` 里的门输入调的是 `m._gate_in(x)` ✓，
       也就是**被测物自己的方法** ✓ —— 于是它内部要是写错了 ✗，
       两边**错得一模一样** ✓，判卷人看不出来 ✗
       （`_gate_in` 忘记 `lowrank` 分支，正是**真发生过**的那个 bug ✓）

`by1exec --compare` 那条路（PyTorch 对 NumPy）也不管这两处 ✓ ——
两个后端**共享同一个 IR** ✓，同一个意思写错两遍的概率很低 ✓，
而它证的是"两边一致"，不是"和 fla 一致" ✗。

## 用法

    python src/modelcheck/by1kda.py
    python src/modelcheck/by1kda.py --selftest

参照物不在就 `[跳过]`、退出码 30（见 `by1skip`）：

    pip install --no-deps fla-core
"""
import importlib.util
import os
import site
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import by1paths  # noqa: E402
import by1skip  # noqa: E402

#: 尺寸取小值 —— 验的是语义，不是规模 ✓（和 `models/kda-shaped.by1` 同一个理由）
BASE = {'conv_kernel': 4, 'conv_bias': False, 'gate_lowrank': False,
        'act': 'silu', 'l2_eps': 1e-6, 'norm_eps': 1e-5}


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        by1skip.skip('没装 torch —— 这一项在这台机器上验不了')


def load_fla_naive():
    """**按文件路径**加载 `fla/ops/kda/naive.py`。

    不走 `import fla.ops.kda` —— 那条路要过 `fla/ops/__init__.py` ✓，
    而它 `import triton` ✗。我们要的只是那个纯 PyTorch 的参考 ✓，
    它和 Triton 一点关系都没有 ✓。
    """
    cands = []
    # **这里原来是个 `except Exception: pass`** ✗ —— 而被 `by1lint` 第一类
    # 抓住了 ✓（"except 之后只有 pass" ✓）。它说得对：静默吞掉之后，
    # **环境坏掉和没装 fla 长得一模一样** ✓ —— 都变成"跳过" ✓，
    # 而跳过是"没验"，不是"验过" ✓。所以让它出声 ✓。
    try:
        _sps = site.getsitepackages()
    except Exception as e:
        print('  （取 site-packages 失败：%s: %s —— 当成"没装 fla"处理）'
              % (type(e).__name__, e))
        _sps = []
    cands += [os.path.join(p, 'fla', 'ops', 'kda', 'naive.py') for p in _sps]
    for p in cands:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location('kda_naive', p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m, p
    return None, None


def runtime(torch):
    """把 `by1codegen` **生成出来**的那份运行时 exec 出来。

    ⚠ **不能 `import by1codegen` 然后取 `KDA`** ✗ ——
    `by1codegen.py` 是"生成器 + 一个字符串模板" ✓，那些类写在
    `RUNTIME` 这个字符串里 ✓，不是模块的属性 ✓。
    """
    import by1check
    import by1codegen as G
    _r, info = by1check.check(os.path.join(by1paths.MODELS, 'kda-shaped.by1'))
    _m, src = G.build(info)
    ns = {}
    exec(compile(src, '<by1-generated>', 'exec'), ns)
    return ns


def front(torch, m, x, *, per_dim_gate=True, per_dim_dt=True, l2norm=True):
    """判卷人自己再写一遍前端 —— 回来的五样就是 `naive_recurrent_kda` 要的。"""
    F = torch.nn.functional
    b, s, _ = x.shape
    nk, nv, dk, dv = m.nk, m.nv, m.dk, m.dv
    ck = m.q_conv1d.kernel_size[0]

    def dconv(conv, z):
        zz = z.transpose(1, 2)
        pad = torch.zeros(b, zz.shape[1], ck - 1, dtype=z.dtype)
        mp = torch.cat([pad, zz], 2)
        c = torch.zeros_like(zz)
        for i in range(ck):
            c = c + mp[:, :, i:i + s] * conv.weight[:, 0, i].view(1, -1, 1)
        return F.silu(c).transpose(1, 2)

    q = dconv(m.q_conv1d, m.q_proj(x)).reshape(b, s, nk, dk)
    k = dconv(m.k_conv1d, m.k_proj(x)).reshape(b, s, nk, dk)
    v = dconv(m.v_conv1d, m.v_proj(x)).reshape(b, s, nv, dv)

    # 门：g = -exp(A_log) * softplus(f(x) + dt_bias)（fla/ops/kda/gate.py:50-54）
    fg = m._gate_in(x).float().reshape(b, s, nv, dk)
    if per_dim_dt:
        bias = m.dt_bias.view(1, 1, nv, dk)          # 按**维** ✓
    else:
        # 反例②：按**头** —— 每头只取第一维那个偏置，摊到整头 ✗
        bias = m.dt_bias.view(1, 1, nv, dk)[..., :1]
    g = -m.A_log.float().exp().view(1, 1, -1, 1) * F.softplus(fg + bias)
    if not per_dim_gate:
        # 反例①：门退化成每头一个标量（对 k_dim 求和再摊开）✗
        g = g.sum(-1, keepdim=True).expand(-1, -1, -1, dk).contiguous()
    beta = m.b_proj(x).sigmoid()

    if l2norm:
        # 判卷人这边**自己做** L2 归一化 ✓（by1 那边在 gated_delta_rule 里 ✓）
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + m.l2_eps)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + m.l2_eps)
    return q, k, v, g, beta


def _norm(x):
    return x / max(1e-9, float(x.abs().max()))


def one(torch, naive, R, seed, s, d, nk, nv, dk, dv, ck, **kw):
    a = dict(BASE, k_heads=nk, v_heads=nv, k_dim=dk, v_dim=dv,
             conv_kernel=ck, gate_rank=dk)
    torch.manual_seed(seed)
    m = R['KDA'](a, d)
    m.eval()
    # **把参数随机化。**
    #
    # 不随机化的话，反例**碰不到东西** ✗：`dt_bias` 和 `A_log` 在
    # `__init__` 里是 `zeros` ✓，于是"按头"和"按维"**算出来一模一样** ✓ ——
    # 那一条反例就永远不红 ✓，而它看起来"通过了" ✓。
    # （实测踩到过：`dt_bias 按头而不是按维` 报 1.490e-08「没红」✗。）
    #
    # **这和"一个碰不到被测东西的测试不是测试"是同一条** ✓。
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    x = torch.randn(1, s, d)
    q, k, v, g, beta = front(torch, m, x, **kw)
    # **在同一个阶段比。**
    #
    # `naive_recurrent_kda` 回来的是**递推的输出** `[b, s, HV, V]` ✓，
    # 而 `m(x)` 是整个 forward（过了 `o_norm` + `o_proj`）✗ ——
    # 第一版直接比，于是炸在形状上 ✓。
    #
    # 接住递推那一步：模块调的是**它自己命名空间里**的 `gated_delta_rule` ✓，
    # 所以换掉 `R` 里那一个就够 ✓。
    cap = {}
    _orig = R['gated_delta_rule']

    def _spy(qq, kk, vv, gg, bb, l2_eps=1e-6, state=None):
        o, st = _orig(qq, kk, vv, gg, bb, l2_eps, state)
        cap['o'] = o
        return o, st

    R['gated_delta_rule'] = _spy
    try:
        with torch.no_grad():
            m(x)
            ref, _ = naive.naive_recurrent_kda(q, k, v, g, beta)
    finally:
        R['gated_delta_rule'] = _orig
    y = cap['o']
    return float((y - ref).abs().max()), float(_norm(ref).abs().max())


def main(argv):
    torch = _torch()
    only_red = '--selftest' in argv
    naive, path = load_fla_naive()
    if naive is None:
        by1skip.skip('找不到 fla 的 naive 参考 —— 这一项在这台机器上验不了\n'
                     '    装它：pip install --no-deps fla-core')
    R = runtime(torch)
    if not only_red:
        print('  参照物：%s' % path)
        print('  生成出来的运行时里有 KDA：%s' % ('KDA' in R))

    ok = True
    base = 0.0
    if not only_red:
        print()
        print('  ── 正例（两种尺寸）')
        for seed, s, d, nk, nv, dk, dv, ck, tag in (
                (0, 16, 32, 2, 2, 8, 8, 4, '小'),
                (1, 32, 48, 3, 3, 16, 16, 4, '中')):
            ad, sc = one(torch, naive, R, seed, s, d, nk, nv, dk, dv, ck)
            rl = ad / sc if sc > 0 else ad
            base = max(base, rl)
            good = rl < 1e-4
            ok = ok and good
            print('     %-6s 绝对 %.3e  相对 %.3e  %s'
                  % (tag, ad, rl, '✓' if good else '✗'))

    # **红线是"比正例的误差大两个数量级"** ✓ ——
    # 不是写死一个数 ✗。理由：反例的效应**本来就可能小** ✓ ——
    # 比如"`dt_bias` 按头而不是按维"只动到 `dk` 维里的 1 维 ✓，
    # 实测 3.1e-04 ✓ —— 而正例的误差是 4.7e-09 ✓。
    # 写死 `1e-3` 会把这条真反例判成"没红" ✗，而它**明明红了** ✓。
    # 自适应之后，"红"的定义变成：**明显不是数值噪声** ✓。
    thr = max(1e-6, base * 100)

    print()
    print('  ── 反例（每一条都必须红 —— 否则这个判卷人碰不到被测的东西）')
    print('     红线：相对 %.3e（= max(1e-6, 正例误差 x 100)）' % thr)
    reds = []
    for tag, kw in (('门退化成每头一个标量', dict(per_dim_gate=False)),
                    ('dt_bias 按头而不是按维', dict(per_dim_dt=False)),
                    ('q/k 不做 L2 归一化', dict(l2norm=False))):
        ad, sc = one(torch, naive, R, 0, 16, 32, 2, 2, 8, 8, 4, **kw)
        rl = ad / sc if sc > 0 else ad
        red = rl > thr
        reds.append(red)
        print('     %-24s 相对 %.3e  %s' % (tag, rl, '红了 ✓' if red else '**没红** ✗'))

    allok = ok and all(reds)
    print()
    print('  [%s]%s' % ('PASS' if allok else 'FAIL',
                        '（反例也验过）' if allok else ''))
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
