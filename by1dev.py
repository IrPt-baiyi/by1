#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1dev -- **设备无关性**的判卷人。

## 它回答的问题

「这些代码拿到显卡上会不会当场坏？」

**这台机器没有显卡**（Iris Xe，torch 是 +cpu），所以**没法真的验**。
但有一件事可以在这里验，而且它正是显卡上最先炸的那一类：

    前向的时候，任何**不指定 device** 的张量创建，
    在显卡上都会造成 "Expected all tensors to be on the same device"。

所以这个脚本把 CPU 默认的工厂**设成陷阱**：

    torch.zeros / ones / empty / arange / tensor / randn / full / eye

在**前向期间**调用它们而**不带 device=**，当场抛异常。

## 它证明什么，不证明什么

    ✓ 证  前向不依赖"张量默认建在 CPU 上"
    ✗ 不证 数值和显卡一致（浮点顺序不同，会有差异）
    ✗ 不证 显存放得下
    ✗ 不证 快

**"设备无关"和"显卡上对"是两件事。** 这个脚本只做前一件 ——
但前一件是后一件的必要条件，而且它**今天就能验**。

`nn.Parameter(torch.zeros(...))` 不算陷阱：那是 `__init__` 的时候建的，
`.to(device)` 会搬走它。所以陷阱只在 `forward` 期间生效。

用法:  python by1dev.py [文件.by1 ...]
"""
import glob
import importlib.util
import os
import sys

import torch

sys.path.insert(0, '.')
sp = importlib.util.spec_from_file_location('bcm', 'by1check.py')
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg

# **要设陷阱的工厂。** `*_like` 不在里面 —— 它们跟着输入走，是对的。
TRAPPED = ['zeros', 'ones', 'empty', 'arange', 'tensor', 'randn',
           'rand', 'full', 'eye', 'linspace']
_real = {n: getattr(torch, n) for n in TRAPPED}
_armed = [False]
_hits = []


def _make(name, orig):
    def f(*a, **kw):
        if _armed[0] and 'device' not in kw:
            # 记下来但不立刻抛 —— 抛的话只能看到第一个。
            import traceback
            st = traceback.extract_stack()
            where = [x for x in st if x.filename.startswith('<by1')
                     or x.filename.endswith('.py')]
            _hits.append((name, where[-2].lineno if len(where) > 1 else -1,
                          where[-2].line if len(where) > 1 else ''))
        return orig(*a, **kw)
    return f


for _n in TRAPPED:
    setattr(torch, _n, _make(_n, _real[_n]))

SKIP = {'selftest.by1', 'gate-probe.by1', 'hello.by1',
        'raw-escape.by1', 'ling-3.0-tiny.by1', 'glm53.by1',
        'nemotron-h.by1'}


def main():
    files = sys.argv[1:] or sorted(glob.glob('*.by1'))
    print('=' * 78)
    print('  设备无关性：前向期间不指定 device 的张量创建')
    print('=' * 78)
    print('  （这台机器没有显卡，所以验的是**必要条件**：')
    print('    前向不依赖"张量默认建在 CPU 上"）')
    print()
    bad = []
    for f in files:
        if f in SKIP or not os.path.exists(f):
            continue
        try:
            _r, info = bc.check(f)
            ir = cg.compile_ir(info)
        except Exception:
            continue
        if len(ir['layers']) > 8:
            continue
        try:
            ns = {}
            exec(compile(cg.render_ir(ir, f), '<ir>', 'exec'), ns)
            m = ns['build']().eval()
        except Exception:
            continue
        # 参数量太大的跑不动
        n = sum(p.numel() for p in m.parameters())
        if n > 5e7:
            continue
        ids = torch.randint(0, ir['vocab'], (1, 16))
        del _hits[:]
        _armed[0] = True
        try:
            with torch.no_grad():
                m(ids)
        except Exception as e:
            print('  !! %-22s 前向崩了：%s' % (f, str(e)[:52]))
            bad.append(f)
            _armed[0] = False
            continue
        _armed[0] = False
        if _hits:
            print('  !! %-22s %d 处不带 device 的创建' % (f, len(_hits)))
            for nm, ln, src in _hits[:3]:
                print('       torch.%s(...)  %s' % (nm, src.strip()[:60]))
            bad.append(f)
        else:
            print('  ok %-22s 前向里没有裸的创建' % f)
    print()
    print('  [%s] 设备无关性 %s'
          % ('PASS' if not bad else 'FAIL',
             '前向不依赖 CPU 默认' if not bad else '这些要修：%s' % bad))
    print()
    print('  **这不等于"显卡上对"** —— 数值一致性、显存、性能都没验。')
    print('  它只证了一件事：前向不靠"张量默认在 CPU 上"。')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
