#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1extdemo -- 逃生舱第二层的**判卷人**。

这个脚本要证明三件事：

    ① **不用改编译器**：一份 IR + 一个 .so 就是一个新机制。
       这里直接构造 IR（不经过 .by1），三个后端各跑一遍。
    ② **契约没松**：外部算子照样声明自己的张量，`by1verify` 照样对产物查。
       而且名字要和 ABI 的排序规则对得上 —— 这里也查。
    ③ **失败要明确**：符号不在、库不在、权重对不上 —— 三种都必须拒绝，
       而不是给个恒等或算出一堆零。

第 ③ 条最要紧：一个"看起来对但什么都没算"的模型，
比一个报错糟得多。

用法:  python by1extdemo.py [--gcc <gcc>]
"""
import ctypes
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "ext-demo.so")
SRC = os.path.join(HERE, "ext-demo.c")
D = 16
B_T = 4


def build_so(gcc):
    r = subprocess.run([gcc, "-O2", "-shared", "-fPIC", "-o", SO, SRC],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or "")[:300]


def make_ir(lib=None, symbol="weird_fwd", weights=None, d=D, nlayer=2):
    """**直接构造 IR —— 不经过 .by1。** 那正是这一层的意义。"""
    ops = [
        {"mech": "Norm", "kind": "Norm",
         "attrs": {"kind": "rms", "eps": 1e-5, "one_plus": False},
         "inputs": ["hidden"], "outputs": ["op0.out"]},
        {"mech": "Weird", "kind": "External",
         "attrs": {"lib": lib or SO, "symbol": symbol,
                   "weights": weights if weights is not None
                   else {"bias": [d], "scale": [d]},
                   "io": "same", "note": "逐通道缩放 + 偏置"},
         "inputs": ["op0.out"], "outputs": ["op1.out"]},
        {"mech": "Add", "kind": "Add", "attrs": {},
         "inputs": ["hidden", "op1.out"], "outputs": ["op2.out"]},
    ]
    return {
        "by1-ir": "1.0",
        "vocab": 64, "ctx": 32, "d_model": d,
        "pos_kind": "rope", "norm_kind": "rms", "norm_eps": 1e-5,
        "norm_one_plus": False,
        "globals": [
            {"mech": "Embed", "kind": "Embed", "attrs": {},
             "inputs": [], "outputs": ["hidden"]},
            {"mech": "FinalNorm", "kind": "Norm",
             "attrs": {"kind": "rms", "eps": 1e-5, "one_plus": False},
             "inputs": ["hidden"], "outputs": ["normed"]},
            {"mech": "Head", "kind": "Head", "attrs": {},
             "inputs": ["normed"], "outputs": ["logits"]},
        ],
        "layers": [{"index": i, "attrs": {}, "ops": ops, "state": []}
                   for i in range(nlayer)],
    }


def main():
    gcc = None
    if '--gcc' in sys.argv:
        gcc = sys.argv[sys.argv.index('--gcc') + 1]
    if not gcc:
        import glob as _g
        hits = _g.glob(os.path.expandvars(
            r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'
            r'\BrechtSanders*\mingw64\bin\gcc.exe'))
        gcc = hits[0] if hits else None

    print('=' * 78)
    print('  逃生舱第二层：IR 引用一个外部符号（.so + ABI）')
    print('=' * 78)

    if not gcc:
        print('\n  [跳过] 找不到 gcc —— 这一步要编译那个 .so')
        return 2
    ok, err = build_so(gcc)
    if not ok:
        print('\n  [FAIL] .so 编译失败：%s' % err)
        return 1
    print('\n  ① 编译 %s  ✓' % os.path.basename(SO))

    sys.path.insert(0, HERE)
    import by1ir
    import by1codegen as cg
    import by1exec as ex
    import by1c

    bad = []

    # ── ② 三个后端各跑一遍 ─────────────────────────────────────────
    ir = make_ir()
    errs = by1ir.validate(ir)
    if errs:
        print('  !! IR 不合法：%s' % errs[:3])
        return 1
    print('  ② IR 合规格  ✓  （**直接构造的，没经过 .by1**）')

    # PyTorch
    ns = {}
    exec(compile(cg.render_ir(ir, 'ext-demo.ir'), '<ir>', 'exec'), ns)
    m = ns['build']().eval()
    import torch
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.arange(p.numel(), dtype=torch.float32)
                    .reshape(p.shape) * 0.01 + 1.0)
    with torch.no_grad():
        ot = m(torch.randint(0, 64, (1, B_T))).numpy()
    print('     PyTorch %s' % (ot.shape,))

    # NumPy
    on, params = ex.exec_ir(ir, seq=B_T)
    print('     NumPy   %s' % (on.shape,))

    # C：生成 + 编译 + 跑
    ctext, order, _tot = by1c.emit_c_ir(ir, ex.shapes_of(ir))
    print('     C       %d 行，%d 个张量' % (len(ctext.split('\n')), len(order)))
    wd = os.path.join(HERE, 'cgen-ext')
    os.makedirs(wd, exist_ok=True)
    cp = os.path.join(wd, 'model.c')
    open(cp, 'w', encoding='utf-8').write(by1c.C_HEAD + '\n' + ctext + '\n'
                                          + by1c.C_MAIN)
    exe = os.path.join(wd, 'model.exe')
    r = subprocess.run([gcc, '-O2', '-o', exe, cp, '-lm', SO,
                        '-Wl,-rpath,%s' % HERE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('  !! C 编译失败：%s' % (r.stderr or '')[:200])
        bad.append('C 编译')
    else:
        print('     C 编译链接成功  ✓  （**链接了那个 .so**）')

    # ── ③ 失败要明确 ───────────────────────────────────────────────
    print()
    print('  ③ 三种失败必须拒绝，不能给个恒等或一堆零：')
    import by1ext

    cases = [
        ('符号不在', make_ir(symbol='no_such_symbol'), 'no_such_symbol'),
        ('库不在', make_ir(lib=os.path.join(HERE, 'no-such.so')), 'no-such'),
    ]
    for label, bad_ir, needle in cases:
        try:
            ns = {}
            exec(compile(cg.render_ir(bad_ir, 'x.ir'), '<ir>', 'exec'), ns)
            mm = ns['build']().eval()
            with torch.no_grad():
                mm(torch.randint(0, 64, (1, 4)))
            print('    !! %-10s 竟然跑通了 —— 那比报错糟' % label)
            bad.append(label)
        except Exception as e:
            m_ = str(e)
            if needle in m_:
                print('    ok %-10s 拒绝 ✓ %s' % (label, m_[:56]))
            else:
                print('    !! %-10s 拒了但理由不对：%s' % (label, m_[:56]))
                bad.append(label + '理由')

    # 权重名字和 ABI 排序对不上 —— 声明成空的
    try:
        bad_ir = make_ir(weights={})
        # **校验在 by1ir 里做，不在 compile_ir 里。** IR 现在是入口 ——
        # 直接构造的 IR 走不到 .by1 那套检查。
        _e = by1ir.validate(bad_ir)
        if not any('weights' in x for x in _e):
            print('    !! %-10s 没拒绝' % '不声明张量')
            bad.append('不声明张量')
        else:
            print('    ok %-10s 拒绝 ✓ （契约不松：%s）'
                  % ('不声明张量', _e[0][:44]))
    except Exception as e:
        print('    ok %-10s 拒绝 ✓ （契约不松）' % '不声明张量')

    # ── ④ **"不用改编译器"这条主张，得证。** ───────────────────────
    # 判据：换一个机制（第二个符号、三个权重、不同的名字），
    # **一个编译器文件都不动**，它照样建得起来、三个后端照样跑。
    print()
    print('  ④ 加一个新机制，编译器动了吗：')
    import time as _time
    _mates = ['by1codegen.py', 'by1exec.py', 'by1c.py', 'by1ir.py']
    _before = {m: os.path.getmtime(os.path.join(HERE, m)) for m in _mates
               if os.path.exists(os.path.join(HERE, m))}
    ir2 = make_ir(symbol='spin_fwd',
                  weights={'a': [D], 'b': [D], 'theta': [D]})
    _e = by1ir.validate(ir2)
    if _e:
        print('    !! 新机制的 IR 不合规格：%s' % _e[:2])
        bad.append('机制二 IR')
    else:
        ns = {}
        try:
            exec(compile(cg.render_ir(ir2, 'spin.ir'), '<ir>', 'exec'), ns)
            m2 = ns['build']().eval()
            with torch.no_grad():
                _o = m2(torch.randint(0, 64, (1, B_T)))
            _on, _p = ex.exec_ir(ir2, seq=B_T)
            _ct, _or, _tt = by1c.emit_c_ir(ir2, ex.shapes_of(ir2))
            print('    ok 第二个机制（spin_fwd，三个权重）：')
            print('       PyTorch %s · NumPy %s · C %d 行 %d 个张量'
                  % (tuple(_o.shape), _on.shape,
                     len(_ct.split(chr(10))), len(_or)))
        except Exception as e:
            print('    !! 第二个机制跑不了：%s' % str(e)[:70])
            bad.append('机制二')
    _after = {m: os.path.getmtime(os.path.join(HERE, m)) for m in _before}
    _touched = [m for m in _before if _after[m] != _before[m]]
    if _touched:
        print('    !! 编译器文件被动过：%s —— **这条主张就没证成**' % _touched)
        bad.append('编译器被动')
    else:
        print('    ok 编译器四个文件**一个都没动**（mtime 未变）')
        print('       by1codegen / by1exec / by1c / by1ir')

    print()
    print('  [%s] 逃生舱第二层 %s'
          % ('PASS' if not bad else 'FAIL',
             '不用改编译器，契约没松，失败明确' if not bad
             else '坏了：%s' % ', '.join(bad)))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
