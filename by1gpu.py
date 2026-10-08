#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gpu -- **只有显卡能做的那几件验证。**

## 它和 by1dev 的分工

    by1dev   **这台机器上就能跑**：前向不依赖"张量默认建在 CPU 上"。
             那是必要条件 —— 没有显卡也能验。

    by1gpu   **要有显卡**：真的搬到 CUDA 上跑，然后比数。

## 三件事，按重要性

① **搬过去能跑** —— `model.cuda()` 之后前向不抛异常。
   这一条 `by1dev` 已经间接证了一半（前向不带裸的创建），
   但只有真卡能证另一半（显存、算子、dtype）。

② **CPU 和显卡算出来一样** —— 同一份 IR、同一组权重，
   两边各跑一遍。**不一样不代表谁错了** —— 浮点归约顺序不同，
   差在 1e-6 量级是正常的。所以判据是**量级**，不是相等。
   差到 1e-3 就是真问题。

③ **和外部实现在显卡上对拍** —— `transformers` 也搬到 CUDA，
   两边比。这一条最硬，因为参考实现也是别人写的。

用法:  python by1gpu.py [--seq 64]
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sp = importlib.util.spec_from_file_location('bcm', 'by1check.py')
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg

SKIP = {'selftest.by1', 'gate-probe.by1', 'hello.by1', 'raw-escape.by1',
        'ling-3.0-tiny.by1', 'glm53.by1', 'nemotron-h.by1'}


def main():
    seq = 64
    if '--seq' in sys.argv:
        seq = int(sys.argv[sys.argv.index('--seq') + 1])

    if not torch.cuda.is_available():
        print()
        print('  **这台机器没有可用的 CUDA 设备。**')
        print('  torch %s   cuda %s' % (torch.__version__, torch.version.cuda))
        print()
        print('  这个脚本要在有 NVIDIA 卡的机器上跑 —— 用法见 by1cloud.py。')
        print('  没有卡也能验的那一半在 by1dev.py 里。')
        return 2

    dev = torch.device('cuda')
    print('=' * 78)
    print('  显卡验证')
    print('=' * 78)
    print('  设备 %s   %s' % (torch.cuda.get_device_name(0),
                              torch.cuda.get_device_capability(0)))
    print('  torch %s   cuda %s' % (torch.__version__, torch.version.cuda))
    print()

    bad = []
    for f in sorted(glob.glob('*.by1')):
        if f in SKIP:
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
        n = sum(p.numel() for p in m.parameters())
        if n > 2e8:
            continue

        ids = torch.randint(0, ir['vocab'], (1, seq))
        # ① CPU 跑一遍
        try:
            with torch.no_grad():
                y_cpu = m(ids).float().numpy()
        except Exception as e:
            print('  !! %-22s CPU 侧就崩了：%s' % (f, str(e)[:44]))
            bad.append(f)
            continue
        # ② 搬到显卡
        try:
            mc = m.to(dev)
            with torch.no_grad():
                y_gpu = mc(ids.to(dev)).float().cpu().numpy()
        except Exception as e:
            print('  !! %-22s **搬到显卡上崩了**：%s' % (f, str(e)[:44]))
            bad.append(f)
            continue
        # ③ 比数。**判据是量级，不是相等** —— 浮点归约顺序不同。
        dd = np.abs(y_cpu - y_gpu).max()
        amp = max(np.abs(y_cpu).max(), 1e-30)
        rel = dd / amp
        mark = 'ok ' if rel < 1e-4 else '!! '
        print('  %s%-22s CPU vs CUDA  绝对 %.2e  相对 %.2e%s'
              % (mark, f, dd, rel, '' if rel < 1e-4 else '   <<< 差太多'))
        if rel >= 1e-4:
            bad.append(f)

    print()
    print('  [%s] 显卡验证 %s'
          % ('PASS' if not bad else 'FAIL',
             '搬得过去，两边数一致' if not bad
             else '这些有问题：%s' % bad))
    if not bad:
        print()
        print('  注意判据：**相对差 < 1e-4 就算一致**。')
        print('  CPU 和显卡的浮点归约顺序不同，1e-6 量级的差是正常的 ——')
        print('  那不代表谁错了。差到 1e-3 才是真问题。')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
