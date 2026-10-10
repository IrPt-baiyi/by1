#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1kda -- KDA 的判卷人：**by1 算的，对着 fla 的参考实现**。

## 为什么是它

KDA 的语义**不是我猜的** —— 是从 `fla` 的源码里读出来的，
五处出处写在 `gpu/by1kda.py` 的 docstring 里。这个脚本把那份
"读出来的东西"变成一条**会跑的判据**。

参照物有两个，都是 fla 自己的**纯 PyTorch** 实现（CPU 上跑得动）：

    ops/kda/naive.py        naive_recurrent_kda   —— 递推那五行
    modules/layernorm_gated.py  rms_norm_ref      —— 门控 RMSNorm

（`fla` 那套 Triton 核要 CUDA，**参考实现不要** ——
 这一点我一开始判断错了，`gpu/by1kda.py` 开头记着。）

## 它比什么 —— **三段，各自单独比**

    ① 前端    by1 递进 `gated_delta_rule` 的那五个张量，
              和判卷人自己算的 q/k/v/g/beta 逐个比
    ② 递推    `gated_delta_rule` 吐出来的，和 `naive_recurrent_kda` 比
    ③ 尾巴    整个 `m(x)`，和「参照递推 → 门控 RMSNorm → `o_proj`」比

**每段单独比**，因为三段会被不同的东西弄错 —— 混成一个数就只能知道
"错了"，不知道"哪儿错了"。

## 判卷人自己写的那一半

前端管线（卷积 · silu · 重排 · 门 · sigmoid）**判卷人自己再写一遍** ——
那是判卷人的活，不是被测物的活。

**而且不调被测物的方法**：门的输入是判卷人自己拿
`f_proj` / `f_b(f_a(x))` 组合出来的，**不调 `m._gate_in`**。
（第一版调了 —— 那是被测物自己的方法，它内部写错了，
两边**错得一模一样**，判卷人看不出来。`_gate_in` 忘了 `lowrank`
分支正是真发生过的那个 bug。）

## 反例（每一条都必须红 —— 否则这个判卷人碰不到被测的东西）

    ① 门退化成**每头一个标量**（GDN 的写法）—— KDA 是逐维的
    ② `dt_bias` 按**头**而不是按**维**
    ③ `q/k` 不做 **L2 归一化**
    ④ 输出门**整条不乘**
    ⑤ 门的**先后反了**（乘在归一化之前）
    ⑥ 低秩输出门中间**多一个 silu**
    ⑦ 低秩衰减门中间**多一个 silu**
    ⑧ 输出门用 **silu 而不是 sigmoid**

## 而它自己**盲过一次** —— 这一条比上面八条都重要

第 ⑧ 条是**事后补的** ✓，因为 by1 和这个判卷人**原来都写成 silu** ✗：

    by1 的 RMSNormGated.forward   F.silu(gate)          ✗
    判卷人的 tail()                rms_norm_ref(z=gate)  ✗（里面也是 silu）

**两边错得一模一样，判卷人一直是绿的** ✓ —— 判卷人最该防、
也最难自己发现的一种失效 ✓：**它的错和被测物的错同源时，它什么都证明不了** ✗。

抓出它的是**读源码那一行** ✓（`fla/layers/kda.py:191` 的
`activation="sigmoid"` ✓），**不是跑测试** ✓。
所以 `gate_act()` 旁边把两处出处写死了 ✓，并且给它配了反例 ⑧ ✓。

**这一节不是自我批评，是使用说明** ✓：这个判卷人的每一条断言，
可信度**不超过它旁边那句出处** ✓。出处错的，它跟着错 ✓。

## 用法

    python src/modelcheck/by1kda.py
    python src/modelcheck/by1kda.py --selftest

参照物不在就 `[跳过]`、退出码 30（见 `by1skip`）：

    pip install --no-deps fla-core
"""
import importlib.util
import os
import re
import site
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import by1paths  # noqa: E402
import by1skip  # noqa: E402

#: 尺寸取小值 —— 验的是语义，不是规模（和 `models/kda-shaped.by1` 同一个理由）
BASE = {'conv_kernel': 4, 'conv_bias': False, 'gate_lowrank': False,
        'act': 'silu', 'l2_eps': 1e-6, 'norm_eps': 1e-5}

#: 正例的尺寸表。**四种**，因为它们是四条不同的代码路径：
#: 单层门 / 单层门（大一点）/ 低秩两层门 / GVA（q/k 头数少于 v）
CASES = [
    ('小', dict(seed=0, s=16, d=32, nk=2, nv=2, dk=8, dv=8, ck=4,
                lowrank=False)),
    ('中', dict(seed=1, s=32, d=48, nk=3, nv=3, dk=16, dv=16, ck=4,
                lowrank=False)),
    ('低秩', dict(seed=2, s=24, d=40, nk=2, nv=2, dk=8, dv=8, ck=4,
                  lowrank=True)),
    ('GVA', dict(seed=3, s=16, d=32, nk=2, nv=4, dk=8, dv=8, ck=4,
                 lowrank=False)),
]

#: 反例打在**哪个尺寸**上。低秩那两条只能打在低秩上。
_PLAIN = CASES[0][1]
_LOWRANK = CASES[2][1]

REDS = [
    ('门退化成每头一个标量', _PLAIN, dict(fkw=dict(per_dim_gate=False))),
    ('dt_bias 按头而不是按维', _PLAIN, dict(fkw=dict(per_dim_dt=False))),
    ('q/k 不做 L2 归一化', _PLAIN, dict(fkw=dict(l2norm=False))),
    ('输出门整条不乘', _PLAIN, dict(tkw=dict(use_gate=False))),
    ('门的先后反了', _PLAIN, dict(tkw=dict(gate_before_norm=True))),
    # **这一条原来的名字是"低秩输出门中间不加 silu"** ✗ ——
    # 也就是说，它把"两层之间有 silu"**当成了标准答案** ✗，
    # 而真答案是反过来 ✓（fla 是 `nn.Sequential`，两层之间没有激活）。
    #
    # 坏的不在反例的机制 ✓（加 silu 和不加 silu 都会红 ✓），
    # 坏的是**默认值** `gate_out(act='silu')` ✓：
    # 于是就算有人把 by1 修对了 ✓，"低秩"那个**正例**也会红 ✗ ——
    # 而正例红了，要改的是**判卷人** ✓。
    # **一条指向反方向的判据，比没有判据更坏** ✓：它会挡住修复 ✓。
    ('低秩输出门中间多一个 silu', _LOWRANK, dict(tkw=dict(out_act='silu'))),
    ('低秩衰减门中间多一个 silu', _LOWRANK, dict(fkw=dict(gate_act='silu'))),
    # **这一条是这一轮真正的收获** ✓ —— 见 `gate_act` 的 docstring：
    # by1 和参照物原来**都**把输出门的激活写成 silu ✗，
    # 于是判卷人一直绿 ✓。抓出它的是**读源码** ✓，不是跑测试 ✓。
    ('输出门用 silu 而不是 sigmoid', _PLAIN, dict(tkw=dict(act='silu'))),
]


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        by1skip.skip('没装 torch —— 这一项在这台机器上验不了')


def _site_file(*parts):
    """在 site-packages 里找 fla 的某个文件。找不到返回 None。

    **这里原来是个 `except Exception: pass`** —— 而被 `by1lint` 第一类
    抓住了（"except 之后只有 pass"）。它说得对：静默吞掉之后，
    **环境坏掉和没装 fla 长得一模一样** —— 都变成"跳过"，
    而跳过是"没验"，不是"验过"。所以让它出声。
    """
    try:
        _sps = site.getsitepackages()
    except Exception as e:
        print('  （取 site-packages 失败：%s: %s —— 当成"没装 fla"处理）'
              % (type(e).__name__, e))
        _sps = []
    for p in _sps:
        q = os.path.join(p, 'fla', *parts)
        if os.path.exists(q):
            return q
    return None


def load_fla_naive():
    """**按文件路径**加载 `fla/ops/kda/naive.py`。

    不走 `import fla.ops.kda` —— 那条路要过 `fla/ops/__init__.py`，
    而它 `import triton`。我们要的只是那个纯 PyTorch 的参考，
    它和 Triton 一点关系都没有。
    """
    p = _site_file('ops', 'kda', 'naive.py')
    if p is None:
        return None, None
    spec = importlib.util.spec_from_file_location('kda_naive', p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        # **装了 fla 不等于这份参考能跑。** 实测过一次：装上 `fla-core`
        # 之后这里报 `No module named 'einops'` —— 那是崩，不是"没验"。
        # 按"参考拿不到"处理，让调用方走 by1skip。
        print('  （fla 的 naive 参考 import 不了：%s: %s —— 当成"没装"处理）'
              % (type(e).__name__, e))
        return None, None
    return m, p


def load_fla_rms_norm_ref():
    """把 `fla/modules/layernorm_gated.py` 里的 `rms_norm_ref` **切出来**。

    ⚠ 那个文件**不能 import** —— 它开头就 `import triton`
    （实测：`ModuleNotFoundError: No module named 'triton'`）。
    而我们要的那个函数是**纯 PyTorch 的参考实现**，
    和 Triton 一行关系都没有。

    所以按函数名从源码里切出那一段再 `exec` —— 这仍然是
    **fla 自己写的那份参照物**，不是我照着重写的一遍。
    （重写一遍就变成"第三个实现"了，而那正是判卷人要避免的。）
    """
    p = _site_file('modules', 'layernorm_gated.py')
    if p is None:
        return None, None
    with open(p, encoding='utf-8') as f:
        src = f.read()
    m = re.search(r'(?ms)^def rms_norm_ref\(.*?(?=^@|^def |\Z)', src)
    if m is None:
        print('  （%s 里找不到 rms_norm_ref —— fla 换版本了？）' % p)
        return None, p
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    ns = {'torch': torch, 'F': F, 'rearrange': rearrange}
    exec(compile(m.group(0), '<fla-rms_norm_ref>', 'exec'), ns)
    return ns['rms_norm_ref'], p


def runtime(torch):
    """把 `by1codegen` **生成出来**的那份运行时 exec 出来。

    ⚠ **不能 `import by1codegen` 然后取 `KDA`** ——
    `by1codegen.py` 是"生成器 + 一个字符串模板"，那些类写在
    `RUNTIME` 这个字符串里，不是模块的属性。
    """
    import by1check
    import by1codegen as G
    _r, info = by1check.check(os.path.join(by1paths.MODELS, 'kda-shaped.by1'))
    _m, src = G.build(info)
    ns = {}
    exec(compile(src, '<by1-generated>', 'exec'), ns)
    return ns


# ── 判卷人自己写的前端 ───────────────────────────────────────────────

def gate_in(torch, m, x, *, act=None):
    """衰减门的**输入** —— 判卷人自己组合，**不调 `m._gate_in`**。

    by1 那边是 `f_b(f_a(x))`（低秩）或 `f_proj(x)`（一层），
    中间**没有激活**。这里照源码写一遍，于是"`_gate_in` 内部写错了"
    这一类**会红**，而不是两边一起错。
    """
    F = torch.nn.functional
    if m.lowrank:
        h = m.f_a_proj(x)
        if act == 'silu':
            h = F.silu(h)
        return m.f_b_proj(h)
    return m.f_proj(x)


def gate_out(torch, m, x, *, act=None):
    """输出门 —— **低秩那支两层之间没有激活** ✓。

    `fla/layers/kda.py:187-190`：

        g_proj = nn.Sequential(nn.Linear(hidden_size, self.head_v_dim, bias=False),
                               nn.Linear(self.head_v_dim, self.value_dim, bias=True))

    `nn.Sequential` **不会自己加激活** ✓。这里原来写的注释是
    "低秩那支中间**有** silu" ✗ —— 那是我凭"门一般配 silu"想出来的 ✗，
    而 `act='silu'` 这个开关留着，专门用来当**反例** ✓。
    """
    F = torch.nn.functional
    if m.lowrank:
        h = m.g_a_proj(x)
        if act == 'silu':
            h = F.silu(h)
        return m.g_b_proj(h)
    return m.g_proj(x)


def front(torch, m, x, *, per_dim_gate=True, per_dim_dt=True, l2norm=True,
          gate_act=None):
    """前端全套。回来的 q/k 有**两份**：归一化前的和归一化后的。

    归一化前的那份用来和 by1 **递进递推**的输入逐个比（那一步 by1
    还没做 L2），归一化后的那份喂给参照递推。
    """
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
    fg = gate_in(torch, m, x, act=gate_act).float().reshape(b, s, nv, dk)
    if per_dim_dt:
        bias = m.dt_bias.view(1, 1, nv, dk)          # 按**维**
    else:
        # 反例②：按**头** —— 每头只取第一维那个偏置，摊到整头
        bias = m.dt_bias.view(1, 1, nv, dk)[..., :1]
    g = -m.A_log.float().exp().view(1, 1, -1, 1) * F.softplus(fg + bias)
    if not per_dim_gate:
        # 反例①：门退化成每头一个标量（对 k_dim 求和再摊开）
        g = g.sum(-1, keepdim=True).expand(-1, -1, -1, dk).contiguous()
    beta = m.b_proj(x).sigmoid()

    qn, kn = q, k
    if l2norm:
        # 判卷人这边**自己做** L2 归一化（by1 那边在 gated_delta_rule 里）
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + m.l2_eps)
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + m.l2_eps)
    return dict(q=q, k=k, v=v, g=g, beta=beta, qn=qn, kn=kn)


def gate_act(torch, g, act="sigmoid"):
    """输出门的**激活** —— 两处出处都在 fla 里，不是选的。

        `fla/layers/kda.py:191`            activation="sigmoid"
        `fla/modules/fused_norm_gate.py`   silu/swish: y * g * sigmoid(g)
                                           sigmoid:   y * sigmoid(g)

    **这一条原来两边都写成了 silu** ✗ —— 也就是说这个判卷人
    **本来是有盲区的** ✓，而且盲的正是它自己要判的那一处 ✓：
    参照物那一侧是我照着同一句话写的 ✓，于是"by1 错、参照物也错" ✓，
    两边**错得一模一样** ✓，判卷人**一直是绿的** ✓。

    这就是判卷人最该防、也最难自己发现的一种失效 ✓ ——
    **判卷人的错和被测物的错同源时，它证明不了任何东西** ✓。
    抓出它的是**读源码**（`activation="sigmoid"` 那一行）✓，
    不是跑测试 ✓。所以这里把出处写死在旁边 ✓，并且给它配一条反例 ✓。
    """
    if act in ("silu", "swish"):
        return g * torch.sigmoid(g)
    if act == "sigmoid":
        return torch.sigmoid(g)
    raise ValueError('不认识的门激活：%r' % (act,))


def tail(torch, m, rms_ref, rec, x, *, use_gate=True, gate_before_norm=False,
         out_act=None, act="sigmoid"):
    """尾巴：输出门 → 门控 RMSNorm → `o_proj`。

    **归一化那一半用 fla 自己的 `rms_norm_ref`** ✓（传 `z=None`，
    它只做 `x * rstd * w`）✓；门的激活**单独乘** ✓ ——
    因为 `rms_norm_ref` 里的 `F.silu(z)` 是**写死的** ✗，
    而 KDA 要的是 sigmoid ✓（见 `gate_act` 的 docstring）。
    """
    b, s, _ = x.shape
    g = gate_out(torch, m, x, act=out_act).reshape(b, s, m.nv, m.dv)
    z = gate_act(torch, g.float(), act) if use_gate else None
    if z is not None and gate_before_norm:
        # 反例：门的先后反了
        rec = rec * z
        z = None
    y = rms_ref(rec, m.o_norm.w, None, z=None, eps=m.o_norm.eps)
    if z is not None:
        y = y * z
    return m.o_proj(y.reshape(b, s, m.vd))


def _rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max()) / max(1e-9, float(b.abs().max()))


def one(torch, naive, rms_ref, R, *, fkw=None, tkw=None, **case):
    """跑一次，回来**三段各自的相对误差**。"""
    fkw = dict(fkw or {})
    tkw = dict(tkw or {})
    a = dict(BASE, k_heads=case['nk'], v_heads=case['nv'],
             k_dim=case['dk'], v_dim=case['dv'],
             conv_kernel=case['ck'], gate_rank=case['dk'],
             gate_lowrank=case['lowrank'])
    torch.manual_seed(case['seed'])
    m = R['KDA'](a, case['d'])
    m.eval()
    # **把参数随机化。**
    #
    # 不随机化的话，反例**碰不到东西**：`dt_bias` 和 `A_log` 在
    # `__init__` 里是 `zeros`，于是"按头"和"按维"**算出来一模一样** ——
    # 那一条反例就永远不红，而它看起来"通过了"。
    # （实测踩到过：`dt_bias 按头而不是按维` 报 1.490e-08「没红」。）
    #
    # **这和"一个碰不到被测东西的测试不是测试"是同一条。**
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    x = torch.randn(1, case['s'], case['d'])
    with torch.no_grad():
        ref = front(torch, m, x, **fkw)

        # 接过 `gated_delta_rule` 那一跳：既要它吐出来的，也要喂进去的。
        cap = {}
        _orig = R['gated_delta_rule']

        def _spy(qq, kk, vv, gg, bb, l2_eps=1e-6, state=None):
            cap['in'] = (qq, kk, vv, gg, bb)
            o, st = _orig(qq, kk, vv, gg, bb, l2_eps, state)
            cap['o'] = o
            return o, st

        R['gated_delta_rule'] = _spy
        try:
            y_by1 = m(x)
            rec, _ = naive.naive_recurrent_kda(
                ref['qn'], ref['kn'], ref['v'], ref['g'], ref['beta'])
        finally:
            R['gated_delta_rule'] = _orig

        y_ref = tail(torch, m, rms_ref, rec, x, **tkw)
    gin = cap['in']
    return {
        # ① 前端：by1 递进递推的那五个，和判卷人自己算的比
        'in': max(_rel(gin[i], ref[n]) for i, n in
                  enumerate(('q', 'k', 'v', 'g', 'beta'))),
        # ② 递推
        'rec': _rel(cap['o'], rec),
        # ③ 尾巴
        'full': _rel(y_by1, y_ref),
    }


def main(argv):
    torch = _torch()
    only_red = '--selftest' in argv
    naive, p_naive = load_fla_naive()
    if naive is None:
        # **`by1skip.skip` 是"打印并给退出码"，不是"打印并退出"。**
        # 少一个 `return` 的话它照样往下跑，然后在 `naive.naive_recurrent_kda`
        # 上 AttributeError —— 报出来是崩溃，不是"这台机器上没验"。
        return by1skip.skip(
            '找不到 fla 的 naive 参考 —— 这一项在这台机器上验不了\n'
            '    装它：pip install --no-deps fla-core einops')
    rms_ref, p_norm = load_fla_rms_norm_ref()
    if rms_ref is None:
        return by1skip.skip(
            '找不到 fla 的 rms_norm_ref —— 这一项在这台机器上验不了\n'
            '    装它：pip install --no-deps fla-core einops')
    R = runtime(torch)
    if not only_red:
        print('  参照物 ①  %s' % p_naive)
        print('  参照物 ②  %s  （切出 rms_norm_ref —— 那个文件 import triton）'
              % p_norm)
        print('  生成出来的运行时里有 KDA：%s' % ('KDA' in R))

    ok = True
    base = 0.0
    if not only_red:
        print()
        print('  ── 正例（四种尺寸 = 四条代码路径）')
        print('     %-4s %-10s %-10s %-10s' % ('', '① 前端', '② 递推', '③ 尾巴'))
        for tag, case in CASES:
            r = one(torch, naive, rms_ref, R, **case)
            base = max(base, r['in'], r['rec'], r['full'])
            good = max(r.values()) < 1e-4
            ok = ok and good
            print('     %-4s %-10.3e %-10.3e %-10.3e  %s'
                  % (tag, r['in'], r['rec'], r['full'],
                     '✓' if good else '✗ 这一段对不上'))

    # **红线是"比正例的误差大两个数量级"**，不是写死一个数。
    # 理由：反例的效应**本来就可能小** —— 比如"`dt_bias` 按头而不是按维"
    # 只动到 `dk` 维里的 1 维，实测 3.1e-04，而正例的误差是 4.7e-09。
    # 写死 `1e-3` 会把这条真反例判成"没红"，而它**明明红了**。
    # 自适应之后，"红"的定义变成：**明显不是数值噪声**。
    thr = max(1e-6, base * 100)

    print()
    print('  ── 反例（每一条都必须红 —— 否则这个判卷人碰不到被测的东西）')
    print('     红线：相对 %.3e（= max(1e-6, 正例误差 x 100)）' % thr)
    reds = []
    for tag, case, kw in REDS:
        r = one(torch, naive, rms_ref, R, **dict(case, **kw))
        worst = max(r.values())
        red = worst > thr
        reds.append(red)
        print('     %-26s ① %-9.3e ② %-9.3e ③ %-9.3e  %s'
              % (tag, r['in'], r['rec'], r['full'],
                 '红了 ✓' if red else '**没红** ✗'))

    allok = ok and all(reds)
    print()
    print('  [%s]%s' % ('PASS' if allok else 'FAIL',
                        '（反例也验过）' if allok else ''))
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
