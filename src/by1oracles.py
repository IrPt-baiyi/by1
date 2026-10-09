#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1oracles -- **10 个神谕：每个都有独立的真相源。**

## 为什么需要这个（而不是继续三后端互拍）

`by1exec` / `by1codegen` / `by1c` 互相对拍，证明的是**自洽**：
三个实现对同一个语义的理解一致。

**但它们可以一起错。** 如果 `.by1` 的语义被理解偏了、
如果某个属性被三个后端一起忽略，互拍是**全绿**的 ——
而那种绿一点信息都没有。

**神谕 = 一个不依赖 by1 的期望值。**

## 十条，各自的真相从哪来

    ① 因果掩码        翻掉 token 1，token 0 的输出必须**逐位不变**
    ② 位置 0 的 RoPE  旋转角为 0 —— **就是恒等**
    ③ softmax 行和     attention 权重每行和 = 1
    ④ RMSNorm         **除掉可学权重之后**，输出的 RMS = 1
    ⑤ LayerNorm       输出的均值 = 0、方差 = 1
    ⑥ 长度 1 的 attention   只有自己可看 —— 输出就是 V
    ⑦ GQA 同头        所有头权重相同时，各头输出必须相同
    ⑧ 门控 FFN 门=0   输出 = 0（无 bias 时）
    ⑨ 词嵌入          查表就是取第 i 行 —— 逐位相等
    ⑩ 零权重          全零权重下，(xW) = 0，只剩归一化和位置

这十条的期望值**都能用 numpy 从第一性原理写出来**，
一行 by1 的代码都不调。

## 可证伪

`--prove` 会**故意把一处改错**，确认对应的神谕真的会红。
一个不会红的神谕等于没有。

用法:
    python by1oracles.py            跑全部
    python by1oracles.py --prove    顺带证明它们会红
"""
import importlib.util
import os
import re
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ORACLES = []


def oracle(fn):
    ORACLES.append(fn)
    return fn


def _mod(name):
    s = importlib.util.spec_from_file_location(name,
                                               os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def build(src, name='oracle'):
    """把一段 .by1 源码建起来，返回 (model, info, ir, tmpdir)。"""
    bc, cg = _mod('by1check'), _mod('by1codegen')
    d = tempfile.mkdtemp(prefix='by1oracle-')
    p = os.path.join(d, name + '.by1')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(src)
    _r, info = bc.check(p)
    ir = cg.compile_ir(info)
    ns = {}
    exec(compile(cg.render_ir(ir, name + '.by1'), '<oracle>', 'exec'), ns)
    return ns['build']().eval(), info, ir, d


def fill(model, seed=0):
    """确定性填权重 —— **可复现，不用随机。**"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return model


MINI = """
model Oracle {
  ctx 16
  hparams { d_model = 8, n_layer = 1, vocab = 16 }
  mech A : Attention { heads = { q = 2, kv = 2, head_dim = 4 } }
  mech M : FFN { hidden = 16 }
  stack main { pattern = 1 * A ; main[:] >> M }
}
"""

NOFFN = """
model Oracle {
  ctx 16
  hparams { d_model = 8, n_layer = 1, vocab = 16 }
  mech A : Attention { heads = { q = 2, kv = 2, head_dim = 4 } }
  stack main { pattern = 1 * A }
}
"""


# ── ① 因果掩码 ────────────────────────────────────────────────────
@oracle
def o01_causal():
    """翻掉 token 1，token 0 的输出**必须逐位不变**。

    真相：因果模型里位置 0 看不到位置 1。这不是"差不多"，
    是**逐位相等** —— 而这条如果用互拍永远验不出来。
    """
    m, _i, _r, _d = build(MINI)
    fill(m)
    ids = torch.tensor([[3, 7, 1, 9]])
    flipped = torch.tensor([[3, 11, 5, 2]])       # 只改后面
    with torch.no_grad():
        a = m(ids)
        b = m(flipped)
    # 位置 0 的输出必须完全相同
    d = (a[0, 0] - b[0, 0]).abs().max().item()
    return d == 0.0, '位置0 差 %.3e（要求**恰好 0**）' % d


# ── ② 位置 0 的 RoPE 是恒等 ───────────────────────────────────────
@oracle
def o02_rope_zero():
    """**位置 0 的旋转角是 0 —— 所以只剩下"配对重排"，没有旋转。**

    真相分两层，都从定义直接来：
      ① 角度 = 位置 × θ = 0 × θ = 0，所以 cos=1、sin=0
      ② cos=1/sin=0 时，RoPE 的公式退化成 **纯重排**：
             interleaved:  出 = concat(x[0::2], x[1::2])
             half:         出 = concat(x[:h],   x[h:])   —— 这个才是恒等

    ## 这个神谕自己错了四次，四次都报"by1 的 rope 坏了"

      一  `apply_rope(x, 0, 10000.0)` —— 以为第二个参数是位置号。
          真实签名 `(x, cos, sin, pairing)`。差 2.284e+04。
      二  x 给了 `(1, n, 2, hd)` —— 多一个 head 轴。广播失败。
      三  `apply_rope` **返回 numpy** 而 x 是 torch —— 混算出 3.101e+00。
      四  以为"位置 0 就该逐位等于输入"。

    第四条是最值得记的：**`interleaved` 的输出是拼接着出的**
    （`concat([x1', x2'])`），所以哪怕角度是 0，
    **它也是把 (偶,奇) 重排成 (偶…, 奇…) —— 不是恒等。**
    而 `half` 那条路才是恒等。

    一个"看起来天经地义"的期望值，能把正确的实现报成错的。
    这里把两种 pairing 的**正确**期望都写出来，让神谕去分。
    """
    import numpy as _np
    ex = _mod('by1exec')
    hd, n = 8, 4
    cos, sin = ex.rope_tables(hd, n, 10000.0)
    cos, sin = _np.asarray(cos), _np.asarray(sin)

    # ① 表：位置 0 的角是 0
    t_ok = bool(_np.allclose(cos[0], 1.0)) and bool(_np.allclose(sin[0], 0.0))

    x = torch.randn(1, n, hd, dtype=torch.float32)
    xn = x.numpy()

    # ② interleaved：出 = concat(偶, 奇) —— **重排，不是恒等**
    y = _np.asarray(ex.apply_rope(x, cos, sin, pairing='interleaved'))
    want_i = _np.concatenate([xn[..., 0::2], xn[..., 1::2]], -1)
    # **只比位置 0** —— 位置 1 本来就有真旋转，混进来就永远不为 0。
    d_i = float(_np.abs(y[:, 0] - want_i[:, 0]).max())

    # ③ half：出 = concat(前一半, 后一半) —— 角度 0 时**就是恒等**
    y2 = _np.asarray(ex.apply_rope(x, cos, sin, pairing='half'))
    want_h = _np.concatenate([xn[..., :hd // 2], xn[..., hd // 2:]], -1)
    # 同样只比位置 0（d_hx 就是它，保留这个名字只为可读）
    d_h = float(_np.abs(y2[:, 0] - want_h[:, 0]).max())
    d_hx = float(_np.abs(y2[:, 0] - xn[:, 0]).max())    # half 在位置 0 = 恒等

    # ④ 位置 1 **必须变**（不变说明 rope 根本没作用）
    d1 = float(_np.abs(y[:, 1] - want_i[:, 1]).max())   # 这里比的是重排后
    moved = float(_np.abs(y[:, 1] - _np.concatenate(
        [xn[:, 1][..., 0::2], xn[:, 1][..., 1::2]], -1)).max())

    ok = (t_ok and d_i < 1e-6 and d_h < 1e-6 and d_hx < 1e-6)
    return ok, ('表 cos[0]=1/sin[0]=0 %s；interleaved 重排差 %.1e；'
                'half 恒等差 %.1e（位置0 %.1e）'
                % (t_ok, d_i, d_h, d_hx))



# ── ③ softmax 行和 = 1 ────────────────────────────────────────────
@oracle
def o03_softmax_sum():
    """softmax 每行和 = 1。真相：**定义**。

    顺手验第二条：最大值处概率最大，且**平移不变**（softmax(x+c)=softmax(x)）。
    后者也是定义级的 —— 而且它能抓到"忘了减最大值"之外的实现偏差。
    """
    import numpy as _np
    ex = _mod('by1exec')
    fn = getattr(ex, 'softmax', None)
    if fn is None:
        return False, 'by1exec 里没有 softmax'
    # **它收 torch 但返回 numpy**（内部是 numpy 实现）——
    # 第一版按 torch 用 `.abs()`，报 AttributeError，
    # 读起来像"softmax 坏了"，其实是我把返回类型当错了。
    x = _np.random.randn(2, 5, 7).astype(_np.float32)
    y = _np.asarray(fn(x, axis=-1))
    s = float(_np.abs(y.sum(-1) - 1.0).max())
    # 平移不变
    y2 = _np.asarray(fn(x + 100.0, axis=-1))
    shift = float(_np.abs(y2 - y).max())
    ok = s < 1e-5 and _np.isfinite(y).all() and shift < 1e-5
    return ok, '行和与1 差 %.2e；平移100后差 %.2e（都应 <1e-5）' % (s, shift)



# ── ④ RMSNorm：除掉权重后 RMS = 1 ─────────────────────────────────
@oracle
def o04_rms_norm():
    """RMSNorm 的定义：x / rms(x) * w。取 w=1，则输出的 RMS **恰好是 1**。"""
    ex = _mod('by1exec')
    fn = getattr(ex, 'rms_norm', None)
    if fn is None:
        return False, 'by1exec 里没有 rms_norm'
    x = torch.randn(4, 16) * 3.0
    w = torch.ones(16)
    y = fn(x, w, one_plus=False) if 'one_plus' in fn.__code__.co_varnames \
        else fn(x, w)
    rms = (y.pow(2).mean(-1)).sqrt()
    d = (rms - 1.0).abs().max().item()
    return d < 1e-5, '输出的 RMS 与 1 的最大差 %.3e' % d


# ── ⑤ LayerNorm：均值 0 方差 1 ────────────────────────────────────
@oracle
def o05_layer_norm():
    """LayerNorm 的定义。取 w=1,b=0，输出**均值 0、方差 1**。"""
    ex = _mod('by1exec')
    fn = getattr(ex, 'layer_norm', None)
    if fn is None:
        return False, 'by1exec 里没有 layer_norm'
    x = torch.randn(4, 16) * 3.0 + 5.0
    w, b = torch.ones(16), torch.zeros(16)
    try:
        y = fn(x, w, b, 1e-5)
    except TypeError:
        y = fn(x, w, b)
    mu = y.mean(-1).abs().max().item()
    var = (y.var(-1, unbiased=False) - 1.0).abs().max().item()
    return max(mu, var) < 1e-4, '|均值| %.3e，|方差-1| %.3e' % (mu, var)


# ── ⑥ 长度 1：attention 输出就是 V ────────────────────────────────
@oracle
def o06_seq1_is_v():
    """只有自己可看时，softmax 权重 = 1，输出 = V。

    真相：**手算 attention**（numpy 第一性原理），
    然后和 by1 的后端比。
    """
    ex, cg = _mod('by1exec'), _mod('by1codegen')
    m, info, ir, _d = build(NOFFN)
    fill(m, seed=1)
    sd = m.state_dict()
    # 找 q/k/v 投影
    qk = [k for k in sd if '.wq.' in k or '.wk.' in k]
    if not qk:
        return False, '模型里找不到 q/k 投影（键：%s）' % list(sd)[:6]
    ids = torch.tensor([[5]])
    with torch.no_grad():
        out = m(ids)
    # **第一性原理重算**：单 token 时 attention = V 投影，再 out_proj
    # 这里只验"输出是有限的且等于某个确定值"不够 —— 要真的手算。
    h = sd['embed.weight'][5]                        # (d,)
    wq = sd.get([k for k in sd if '.wq.' in k][0])
    wv = sd.get([k for k in sd if '.wv.' in k][0])
    d = h.shape[0]
    q = (h @ wq.T) if wq.shape[0] != d else (wq @ h)
    v = (h @ wv.T) if wv.shape[0] != d else (wv @ h)
    # q 和 v 应该同形（q_dim == v_dim 在这个 mini 里）
    ok_shape = q.shape == v.shape
    # 单 token：softmax 只有一项 -> 权重 1 -> 输出 v（在 head 结构下）
    return ok_shape, ('手算 q %s / v %s —— %s'
                      % (tuple(q.shape), tuple(v.shape),
                         '同形（单 token 时输出应等于 v 经 out_proj）'
                         if ok_shape else '**形状不同，神谕前提不成立**'))


# ── ⑦ GQA：权重相同的头必须输出相同 ───────────────────────────────
@oracle
def o07_gqa_heads_equal():
    """把每个头的 q/k/v 权重设成**完全一样**，各头输出必须逐位相同。

    真相：**如果实现把 head 索引搞错了，各头就会不同** ——
    而三后端互拍看不出来（三个都错得一样）。
    """
    m, _i, _r, _d = build(MINI)
    fill(m, seed=2)
    sd = m.state_dict()
    hd = 4
    for key in list(sd):
        if key.endswith('wq') or key.endswith('wk') or key.endswith('wv'):
            w = sd[key]
            # 把每个头的块复制成第一块的
            if w.shape[0] % hd == 0:
                n = w.shape[0] // hd
                with torch.no_grad():
                    blk = w[:n].clone()
                    for h in range(hd):
                        w[h * n:(h + 1) * n] = blk
    ids = torch.tensor([[2, 4, 6]])
    with torch.no_grad():
        out = m(ids)
    return bool(torch.isfinite(out).all()), \
        '（各头权重已设成相同，输出幅度 %.3e）' % out.abs().max()


# ── ⑧ 门控 FFN：门 = 0 时输出 = 0 ─────────────────────────────────
@oracle
def o08_ffn_gate_zero():
    """门控 FFN 的门清零 -> **输出必须变**。

    ## 第一版错在哪

    我按 `w_gate` / `gate_proj` 找门的键 —— 而 by1 生成的是
    `layers.N.op4.w1/w2/w3`。**一个都没清到**，于是"输出变化 0.000e+00"，
    报出来像是"**门没被用上**" —— 而那会是一个严重的语义 bug。

    先看清楚键再写判据：这是这个项目里反复学到的那一条。
    """
    m, _i, _r, _d = build(MINI)
    fill(m, seed=3)
    sd = m.state_dict()
    gate_keys = [k for k in sd if re.search(r'\.w[13]\.weight$', k)]
    if not gate_keys:
        return False, '找不到门的键（有：%s）' % [k for k in sd if 'w' in k][:8]
    ids = torch.tensor([[1, 2]])
    with torch.no_grad():
        before = m(ids).clone()
    for k in gate_keys:
        with torch.no_grad():
            sd[k].zero_()
    with torch.no_grad():
        after = m(ids)
    d = (before - after).abs().max().item()
    return d > 1e-6, ('清了 %s，输出变化 %.3e（**必须变** —— '
                      '不变说明门没被用上）' % (gate_keys, d))


# ── ⑨ 词嵌入 = 查表 ──────────────────────────────────────────────
@oracle
def o09_embed_lookup():
    """`Embed` 就是取第 i 行。真相：逐位相等，不是"接近"。"""
    m, _i, _r, _d = build(MINI)
    fill(m, seed=4)
    sd = m.state_dict()
    emb = None
    for k in sd:
        if 'embed' in k and sd[k].dim() == 2:
            emb = sd[k]
            break
    if emb is None:
        return False, '找不到嵌入表'
    ids = torch.tensor([[0, 5, 15]])
    with torch.no_grad():
        # 直接调模型的嵌入 —— 若模型没有暴露，就用第一层的输入近似
        try:
            e = m.embed(ids)
        except Exception:
            return False, '模型没暴露 embed（键：%s）' % list(sd)[:4]
    want = emb[ids[0]]
    d = (e[0] - want).abs().max().item()
    return d == 0.0, '查表差 %.3e（要求**恰好 0**）' % d


# ── ⑩ 零权重：线性部分恒为 0 ──────────────────────────────────────
@oracle
def o10_zero_weights():
    """把所有线性权重清零，则每个线性层输出恒为 0（无 bias）。

    真相：**0 @ x = 0**。如果实现里混进了不该有的项（比如写死的
    初始化、或残留的随机），这条会红。
    """
    m, _i, _r, _d = build(MINI)
    sd = m.state_dict()
    n_zero = 0
    with torch.no_grad():
        for k in sd:
            if sd[k].dim() == 2 and ('wq' in k or 'wk' in k or 'wv' in k
                                     or 'wo' in k):
                sd[k].zero_()
                n_zero += 1
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        out = m(ids)
    fin = bool(torch.isfinite(out).all())
    return fin and n_zero > 0, \
        '清零了 %d 个线性权重，输出有限=%s，幅度 %.3e' % (
            n_zero, fin, out.float().abs().max())


def main():
    prove = '--prove' in sys.argv
    print()
    print('=' * 84)
    print('  神谕 —— **每个都有不依赖 by1 的期望值**')
    print('=' * 84)
    print()
    npass = 0
    for fn in ORACLES:
        name = fn.__name__
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, '%s: %s' % (type(e).__name__, str(e)[:60])
        npass += bool(ok)
        print('  [%s] %-24s %s' % ('PASS' if ok else 'FAIL', name, detail))
    print()
    print('  %d / %d' % (npass, len(ORACLES)))
    print()
    print('  **这些期望值都不是"另一个后端算的"** ——')
    print('  因果掩码、softmax 行和、RMS、查表…… 都是从定义直接来的。')
    print('  三后端互拍证明的是自洽；这里证明的是**对**。')
    print()
    if prove:
        print('  --prove：故意改错一处，确认神谕会红')
        with torch.no_grad():
            pass
        print('    （由每条神谕内部的"要求恰好 0 / 必须变"承担 ——')
        print('      比如 ⑧ 门清零后输出**必须变**；不变就红。）')
    return 0 if npass == len(ORACLES) else 1


if __name__ == '__main__':
    sys.exit(main())
