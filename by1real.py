#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1real -- **真权重、真维度，跑一个 27B 模型的前向。**

这是第一次能做这件事。以前所有"真前向"都是：
  - instella-3b —— 真维度，但**随机权重**（对官方实现逐位比，0.000e+00）
  - gpt2 —— 真权重，但只有 163M

这个脚本用 `by1load` 从 `.by1` 的 emit 规则反推出 729 个物理名，
把 ModelScope 下的 55 GB 真权重装进 by1 生成的模型，跑一次前向。

## 它验什么

    ✓ .by1 的命名规则**和真实产物对得上**（729 个，逐个）
    ✓ 55 GB 权重能装进 by1 的模型（形状全对）
    ✓ 前向跑得出来、不是 NaN
    ✓ **在 A800 上是多少显存**

## 它不验什么

    ✗ 数值对不对 —— 那要有参考实现（Qwen3.5 的自定义 modeling 文件）
    ✗ 和 llama.cpp 一致

第一步先证明"能装上、能跑"，那已经是没做过的事。

用法:  python by1real.py [--by1 qwen38.by1] [--dir /dev/shm/qwen38] [--seq 4]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    by1f = 'qwen38.by1'
    root = '/dev/shm/qwen38'
    seq = 4
    for i, a in enumerate(sys.argv):
        if a == '--by1':
            by1f = sys.argv[i + 1]
        if a == '--dir':
            root = sys.argv[i + 1]
        if a == '--seq':
            seq = int(sys.argv[i + 1])

    import importlib.util as iu

    def load(name):
        s = iu.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
        m = iu.module_from_spec(s)
        s.loader.exec_module(m)
        return m

    bc, cg, ld = load('by1check'), load('by1codegen'), load('by1load')

    print('=' * 78)
    print('  真权重前向：%s' % by1f)
    print('=' * 78)

    _r, info = bc.check(by1f)
    ir = cg.compile_ir(info)
    print('  %d 层，d_model=%d，vocab=%d' % (len(ir['layers']),
                                             ir['d_model'], ir['vocab']))

    # ── ① 映射 ────────────────────────────────────────────────────
    pmap, unresolved = ld.plan(info)
    print()
    print('  ① 从 .by1 的 emit 规则反推物理名：%d 个' % len(pmap))
    if unresolved:
        print('     **推不出的 %d 个**（融合/布局/量化）：' % len(unresolved))
        for mech, why in unresolved[:4]:
            print('       %-10s %s' % (mech, why))

    # ── ② 找权重文件 ──────────────────────────────────────────────
    shards = sorted(glob.glob(os.path.join(root, '*.safetensors')))
    if not shards:
        print()
        print('  [跳过] %s 里没有 safetensors —— 先从 ModelScope 下' % root)
        return 2
    tot = sum(os.path.getsize(f) for f in shards)
    print()
    print('  ② 权重：%d 个分片，%.1f GB' % (len(shards), tot / 1e9))

    # ── ③ 建模型 ──────────────────────────────────────────────────
    print()
    print('  ③ 建模型（PyTorch 后端）')
    t0 = time.time()
    ns = {}
    exec(compile(cg.render_ir(ir, by1f), '<ir>', 'exec'), ns)
    model = ns['build']().eval()
    npar = sum(p.numel() for p in model.parameters())
    print('     %d 个参数（%.1f B）%.0f 秒' % (npar, npar / 1e9, time.time() - t0))
    print('     显存/内存占用（bfloat16 约 %.1f GB）'
          % (npar * 2 / 1e9))

    # ── ④ 装权重 ──────────────────────────────────────────────────
    # 真张量按需从分片里取 —— **不一次全读进内存**（55 GB）。
    from safetensors import safe_open
    handles = {f: safe_open(f, framework='pt', device='cpu') for f in shards}
    keymap = {}
    for f, h in handles.items():
        for k in h.keys():
            keymap[k] = f

    def fetch(phys):
        f = keymap.get(phys)
        if f is None:
            return None
        return handles[f].get_tensor(phys)

    print()
    print('  ④ 装权重（%d 个物理名，从 %d 个分片按需取）'
          % (len(keymap), len(shards)))
    t0 = time.time()
    done, missing, skipped, _ = ld.load(model, info, fetch)
    print('     装上 %d 个，%.0f 秒' % (done, time.time() - t0))
    if missing:
        print('     **缺 %d 个**（产物里没有）：' % len(missing))
        for k, phys in missing[:5]:
            print('       %-40s <- %s' % (k[:40], phys))
    if skipped:
        print('     跳过 %d 个：%s' % (len(skipped), skipped[:3]))

    # ── ⑤ 跑到显卡上，前向 ────────────────────────────────────────
    if not torch.cuda.is_available():
        print()
        print('  [跳过] 没有 CUDA')
        return 1
    dev = torch.device('cuda')
    print()
    print('  ⑤ 搬到显卡 + 前向')
    t0 = time.time()
    try:
        model = model.to(dev).to(torch.bfloat16)
        print('     搬到显卡 %.0f 秒，占用 %.1f GB'
              % (time.time() - t0,
                 torch.cuda.memory_allocated() / 1e9))
    except Exception as e:
        print('     **搬不过去**：%s' % str(e)[:100])
        return 1

    ids = torch.tensor([[100 + i for i in range(seq)]], device=dev)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids)
    dt = time.time() - t0
    print('     前向 %d 个 token，%.2f 秒' % (seq, dt))
    print('     输出 %s，幅度 %.4e' % (tuple(out.shape), out.float().abs().max()))
    print('     峰值显存 %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9))

    ok = bool(torch.isfinite(out).all())
    print()
    print('  [%s] 真权重 27B 前向 %s'
          % ('PASS' if ok else 'FAIL',
             '跑通了，输出有限' if ok else '**输出里有 NaN/Inf**'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
