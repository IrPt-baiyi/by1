#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1raw -- 逃生舱的**判卷人**。

一条只会通过的规则不是规则。逃生舱有三条断言，每条都要能反过来试：

    ① 有 raw.py       -> 能建、能跑，**而且契约里的名字真的建出来了**
    ② 没有 raw.py     -> 必须拒绝（不能静默给个恒等）
    ③ Raw 没写 impl    -> 必须拒绝

## ①里那半句是关键

"能跑"不够 —— 还要**契约里声明的张量真的出现在建出来的模型里**。
第一版我没查这个，于是 `raw.py` 里写的是 `self.scale = nn.Parameter(...)`，
建出来是 `…op1.inner.scale`，而契约说的是 `scale.weight` ——
**少了一段，没有东西告诉我。**

用法:  python by1raw.py
"""
import importlib.util
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sp = importlib.util.spec_from_file_location('bcm',
                                             os.path.join(HERE, 'by1check.py'))
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
import by1io
import by1paths

# 裸名交给 `by1paths` 解析 —— 于是打印出来还是那个短名字，
# 而真正打开的路径在 models/。
F = 'raw-escape.by1'
# **`raw.py` 跟着模型走。** 断言②要把它挪走再挪回来。
RAW = by1paths.model('raw.py')
def main():
    bad = []

    # ── ① 有 raw.py ────────────────────────────────────────────────────
    _r, info = bc.check(F)
    ns = {}
    exec(compile(cg.render(info, F), '<by1-generated>', 'exec'), ns)
    m = ns['build']()
    names = [k for k, _ in m.named_parameters()]
    # 契约里 Weird 声明的逻辑名
    want = ['scale.weight']
    hit = [w for w in want if any(k.endswith(w) for k in names)]
    if len(hit) == len(want):
        print('  ok ① 有 raw.py：建起来了，契约里的名字也在（%s）' % ', '.join(hit))
    else:
        print('  !! ① 契约说 %s，建出来的里面没有 —— 现在的名字：%s'
              % (want, [k for k in names if 'scale' in k]))
        bad.append('①')
    try:
        m(torch.randint(0, 256, (1, 8)))
        print('     前向跑得动 ✓ 形状 %s'
              % (tuple(m(torch.randint(0, 256, (1, 8))).shape),))
    except Exception as e:
        print('  !! ① 前向崩了:', str(e)[:60])
        bad.append('①前向')

    # ── ② 没有 raw.py ─────────────────────────────────────────────────
    bak = RAW + '.by1bak'
    shutil.move(RAW, bak)
    try:
        _r, info = bc.check(F)
        ns = {}
        try:
            exec(compile(cg.render(info, F), '<by1-generated>', 'exec'), ns)
            ns['build']()
            print('  !! ② 没有 raw.py 却建起来了 —— 那比报错糟得多')
            bad.append('②')
        except Exception as e:
            msg = str(e)
            if 'raw.py' in msg:
                print('  ok ② 没有 raw.py：拒绝 ✓ 理由对（%s）' % msg[:48])
            else:
                print('  !! ② 拒了，但理由不是"找不到 raw.py"：%s' % msg[:60])
                bad.append('②理由')
    finally:
        shutil.move(bak, RAW)

    # ── ③ Raw 没写 impl ───────────────────────────────────────────────
    src = by1io.read_text(by1paths.model(F), encoding='utf-8')
    src = src.replace('    impl       = "scale_mix"\n', '')
    src = src.replace('model raw-escape', 'model raw-noimpl')
    tmp = by1paths.root('_noimpl.by1')
    by1io.write_text(tmp, src)
    try:
        _r, info = bc.check(tmp)
        try:
            cg.compile_ir(info)
            print('  !! ③ 没写 impl 却编译过了')
            bad.append('③')
        except cg.CodegenError as e:
            msg = str(e)
            if 'impl' in msg:
                print('  ok ③ 没写 impl：拒绝 ✓ 理由对')
            else:
                print('  !! ③ 拒了，但理由里没提 impl：%s' % msg[:60])
                bad.append('③理由')
    finally:
        os.remove(tmp)

    print()
    print('  [%s] 逃生舱 %s' % ('PASS' if not bad else 'FAIL',
                                '三条断言都成立' if not bad else '坏了：%s' % ', '.join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
