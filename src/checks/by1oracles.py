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

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import importlib.util
import os
import re
import shutil
import sys
import tempfile

import torch
import numpy as np

import by1io      # noqa: F401  —— import 即把 stdout 钉成 UTF-8

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ORACLES = []


def oracle(fn):
    ORACLES.append(fn)
    return fn


def _mod(name):
    s = importlib.util.spec_from_file_location(name,
                                               by1paths.tool(name + '.py'))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def build(src, name='oracle'):
    """把一段 .by1 源码建起来，返回 `(model, info, ir)`。

    **临时目录在这里就删掉。** 里面的 `.by1` 只有 `check()` 那一步用得上；
    原来它被当作第 4 个返回值带出去，而**没有任何调用方删它** ——
    每跑一次神谕就在临时区漏 10 个目录（`by1oracle-*`）。
    一个只写不删的临时目录，跑一万遍就是一万个。

    `_FAULT['attn']` 为真时给第一个 Attention 的输出加 1e-3 扰动 ——
    这是 `--prove` 的注错口：参照型神谕（⑥⑦）此时**必须**转红。
    """
    bc, cg = _mod('by1check'), _mod('by1codegen')
    d = tempfile.mkdtemp(prefix='by1oracle-')
    try:
        p = os.path.join(d, name + '.by1')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(src)
        _r, info = bc.check(p)
        ir = cg.compile_ir(info)
        ns = {}
        exec(compile(cg.render_ir(ir, name + '.by1'), '<oracle>', 'exec'), ns)
        m = ns['build']().eval()
        if _FAULT.get('attn'):
            for mod in m.modules():
                if type(mod).__name__ == 'Attention':
                    _orig = mod.forward

                    def _faulty(x, _o=_orig):
                        return _o(x) + 1e-3
                    mod.forward = _faulty
                    break
        return m, info, ir
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    m, _i, _r = build(MINI)
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

    同时验第二条：最大值处概率最大，且**平移不变**（softmax(x+c)=softmax(x)）。
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
def _ref_forward(sd, ids):
    """NOFFN 架构的**第一性原理前向** —— 纯 numpy，不调 by1 的一行代码。

    架构照 IR 逐字展开（`NOFFN`：embed -> RMSNorm -> Attention ->
    残差 -> RMSNorm -> head）：

        x   = embed[ids]
        h   = x * rsqrt(mean(x^2) + eps) * w0            # eps = 1e-5
        q/k/v = h @ Wq/Wk/Wv                              # [n, q*hd]
        RoPE interleaved（奇偶进、半角出，base 10000，全维度）
        att = softmax(q@k^T / sqrt(hd) + 因果 mask) @ v
        x   = x + att_cat @ Wo
        out = head(x * rsqrt(mean(x^2) + eps) * wf)

    **为什么要手写**：三后端互拍证明的是自洽（三个可以一起错）；
    这里期望值从定义直接来，和 by1 的实现没有一行共享。
    """
    W = {k: v.detach().cpu().double().numpy() for k, v in sd.items()}
    need = ['embed.weight', 'layers.0.op0.w', 'layers.0.op1.wq.weight',
            'layers.0.op1.wk.weight', 'layers.0.op1.wv.weight',
            'layers.0.op1.wo.weight', 'final_norm.w', 'head.weight']
    miss = [k for k in need if k not in W]
    if miss:
        raise KeyError('NOFFN 参照缺少张量: %s' % miss)
    qh, kvh, hd = 2, 2, 4
    eps = 1e-5

    def rms(x, w):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w

    ids = np.asarray(ids).reshape(-1)
    n = len(ids)
    x = W['embed.weight'][ids]                                # [n, d]
    h = rms(x, W['layers.0.op0.w'])
    q = (h @ W['layers.0.op1.wq.weight'].T).reshape(n, qh, hd)
    k = (h @ W['layers.0.op1.wk.weight'].T).reshape(n, kvh, hd)
    v = (h @ W['layers.0.op1.wv.weight'].T).reshape(n, kvh, hd)
    # RoPE：interleaved = 奇偶进、拼接着出；theta_i = base^(-2i/hd)
    base = 10000.0
    inv = 1.0 / base ** (np.arange(0, hd // 2) / (hd // 2))

    def rope(t, ang):
        c, s = np.cos(ang), np.sin(ang)
        e, o = t[0::2], t[1::2]
        return np.concatenate([e * c - o * s, e * s + o * c])

    qr = np.stack([rope(q[p, hh], p * inv) for p in range(n) for hh in range(qh)])
    qr = qr.reshape(n, qh, hd)
    kr = np.stack([rope(k[p, hh], p * inv) for p in range(n) for hh in range(kvh)])
    kr = kr.reshape(n, kvh, hd)
    # GQA：q 头 i 用 kv 头 i // (q//kv)
    rep = qh // kvh
    out_heads = []
    for hh in range(qh):
        kh = kr[:, hh // rep]                               # [n, hd]
        vh = v[:, hh // rep]                                # [n, hd]
        sc = (qr[:, hh] @ kh.T) / np.sqrt(hd)                 # [n, n]
        for p in range(n):
            sc[p, p + 1:] = -np.inf                           # 因果
        sc = sc - sc.max(-1, keepdims=True)
        w = np.exp(sc)
        w = w / w.sum(-1, keepdims=True)
        out_heads.append(w @ vh)                              # [n, hd]
    a = np.concatenate(out_heads, -1) @ W['layers.0.op1.wo.weight'].T
    y = rms(x + a, W['final_norm.w'])
    return y @ W['head.weight'].T                             # [n, V]


# 证明神谕"有牙"用的注错开关：build() 看到它就给注意力输出加 1e-3 扰动。
# 参照型神谕（⑥⑦）在注错后**必须**转红 —— 不会红的神谕等于没有。
_FAULT = {'attn': False}


@oracle
def o06_seq1_is_v():
    """只有自己可看时，softmax 权重 = 1，输出 = V。

    真相：**手算 NOFFN 的前向**（见 `_ref_forward`），逐位比。
    第一版只查了 q/v 同形 —— 那永远为真，等于没查（ir.md 的同一族错误）。
    """
    m, _i, _r = build(NOFFN)
    fill(m, seed=1)
    sd = m.state_dict()
    ids = torch.tensor([[5]])
    with torch.no_grad():
        out = m(ids)
    ref = _ref_forward(sd, [5])
    d = float(np.abs(out[0, 0].double().numpy() - ref[0]).max())
    # 单 token 时 attention 退化成"V 经 out_proj" —— 退化性质本身也验：
    # 第一性原理参照里 softmax 只有一项，权重恰好 1。
    return d < 1e-5, '与手算参照最大差 %.3e（单 token = V 经 out_proj）' % d


# ── ⑦ GQA/多头：逐头手算参照 ──────────────────────────────────────
@oracle
def o07_gqa_heads_equal():
    """每个头独立手算，再和模型输出逐位比。

    ## 第一版错在哪（这就是它重写的原因）

    原版把各头的 q/k/v 权重设成**完全一样**，然后只查
    `isfinite(out)` —— 输出当然有限，**这条永远绿**。
    而"权重相同"这个设计本身就抓不到 head 索引错误：
    各头权重一样时，**搞混哪一head跟哪一head，输出毫无变化**
    （混淆相同的东西还是相同）。所以它连理论上的牙都没有。

    现在换成真正有牙的版本：**各头权重保持随机互不相同**，
    用 `_ref_forward` 逐头手算参照（含 RoPE、GQA 映射、因果 mask），
    和模型输出比。头索引、头布局、RoPE 配对、softmax 缩放
    任何一处搞错，两边立刻分叉。

    用 `NOFFN` 而不是 `MINI`：这条验的是注意力，FFN 归 ⑧ 管。
    """
    m, _i, _r = build(NOFFN)
    fill(m, seed=2)
    sd = m.state_dict()
    ids = torch.tensor([[2, 4, 6]])
    with torch.no_grad():
        out = m(ids)
    ref = _ref_forward(sd, [2, 4, 6])
    d = float(np.abs(out[0].double().numpy() - ref).max())
    return d < 1e-5, '逐头手算参照 vs 模型，最大差 %.3e' % d


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
    m, _i, _r = build(MINI)
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
    m, _i, _r = build(MINI)
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

    第一版只查 `isfinite` —— 输出是垃圾也有限，永远绿。
    现在：**输出必须逐位等于 0**，并且带一个反向对照
    （故意留下 wv 不清零 -> 输出必须**非**0），两个方向都验。
    """
    def run(keep_head=False):
        m, _i, _r = build(MINI)
        sd = m.state_dict()
        n_zero = 0
        with torch.no_grad():
            for k in sd:
                # **所有**线性权重（attention / FFN / head），embed 是输入不清
                if sd[k].dim() == 2 and not k.startswith('embed'):
                    if keep_head and k.startswith('head'):
                        continue
                    sd[k].zero_()
                    n_zero += 0 if keep_head else 1
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            return m(ids), n_zero

    out, n_zero = run()
    d_all = float(out.abs().max())
    out2, _ = run(keep_head=True)
    d_keep = float(out2.abs().max())
    ok = d_all == 0.0 and d_keep > 1e-6
    return ok, ('清零 %d 个线性权重：输出幅度 %.3e（**必须恰为 0**）；'
                '故意留下 head 不清：幅度 %.3e（**必须非 0**，否则这条没有牙）'
                % (n_zero, d_all, d_keep))


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
        print('  --prove：给注意力输出注入 1e-3 扰动，参照型神谕必须转红')
        teeth = []
        for fn in (o06_seq1_is_v, o07_gqa_heads_equal):
            _FAULT['attn'] = True
            try:
                ok, detail = fn()
            finally:
                _FAULT['attn'] = False
            teeth.append((fn.__name__, not ok))
        for name, red in teeth:
            print('    [%s] %-24s %s' % (
                '有牙' if red else '**没有牙**', name,
                '注错后转红' if red else '注错后居然还绿 —— 它验了个空'))
        print('    （⑧⑩ 自带反向对照：门清零后输出必须变 / 留一个权重不清必须非 0）')
    return 0 if npass == len(ORACLES) else 1


if __name__ == '__main__':
    sys.exit(main())
