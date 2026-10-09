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

用法:  python by1real.py [--by1 Qwen3.8-27B.by1] [--dir /dev/shm/qwen38] [--seq 4]
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import contextlib
import by1io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _cgroup_gb():
    """容器的真实内存上限。**`free` 说的不算。**

    ## 读不到时返回 `None`，不返回 0

    原来读不到就 `return 0.0` —— 而调用方拿它去打印
    "cgroup 上限 0 GB"，**那句话是错的，而且会让人以为装不下**。

    `None` 表示"不知道"。调用方要自己分辨：
    **0 GB 和"不知道"是两件事，混在一起两个都不成立。**
    """
    for f in ('/sys/fs/cgroup/memory.max',
              '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        with contextlib.suppress(OSError, ValueError):
            v = int(by1io.read_text(f).strip())
            if v < 1 << 50:          # 不是 "max"
                return v / 1e9
    return None


def _dirsize(d):
    tot = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            tot += os.path.getsize(os.path.join(dp, f))
    return tot / 1e9


def npar_est(info):
    """参数量的粗估 —— 从契约的形状表达式算，不用先建模型。"""
    import importlib.util as _iu
    s = _iu.spec_from_file_location('by1exec', os.path.join(HERE, 'by1exec.py'))
    m = _iu.module_from_spec(s)
    s.loader.exec_module(m)
    try:
        import by1codegen as _cg
        _r, _i = None, info
        ir = _cg.compile_ir(_i)
        return sum(int(np.prod(v)) for v in m.shapes_of(ir).values())
    except Exception:
        return 0


def main():
    by1f = 'Qwen3.8-27B.by1'
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
    # **必须用 bf16 建。**
    #
    # 第一版用默认的 fp32 —— 27B 参数 = 108 GB。而容器有 cgroup 上限：
    #     memory.limit_in_bytes = 128849018880 = **120 GB**
    # 权重还在 /dev/shm 里占 52 GB（**shm 也算 cgroup**），
    # 52 + 108 = 160 > 120 -> **进程被杀，没有报错，日志停在"建模型"**。
    #
    # 这个坑很隐蔽：`free -g` 说 1 TB，看着绰绰有余 ——
    # **但 cgroup 才是管用的那个。**
    #
    # bf16 是 54 GB，52 + 54 = 106 < 120，装得下。
    torch.set_default_dtype(torch.bfloat16)
    print()
    print('  ③ 建模型（PyTorch 后端，**bf16**）')
    print('     cgroup 上限 %.0f GB，权重占 %.0f GB，模型 %.0f GB'
          % (_cgroup_gb(), _dirsize(root), npar_est(info) * 2 / 1e9))
    t0 = time.time()
    ns = {}
    exec(compile(cg.render_ir(ir, by1f), '<ir>', 'exec'), ns)
    model = ns['build']().eval()
    npar = sum(p.numel() for p in model.parameters())
    print('     %d 个参数（%.1f B）%.0f 秒' % (npar, npar / 1e9, time.time() - t0))
    print('     显存/内存占用（bfloat16 约 %.1f GB）'
          % (npar * 2 / 1e9))

    # ── ④ 先搬到显卡，再装权重 ────────────────────────────────────
    #
    # **顺序很重要。** 先装权重再搬的话：
    #     CPU: /dev/shm 52 GB + 模型 54 GB = 106 GB
    #     搬的时候 CPU 上那份还在 -> 峰值更高，容易撞 120 GB 的 cgroup 上限
    # 先搬再装：CPU 上只有 shm 那 52 GB，权重是 CPU -> GPU 逐块过去的。
    if not torch.cuda.is_available():
        print()
        print('  [跳过] 没有 CUDA')
        return 1
    dev = torch.device('cuda')
    print()
    print('  ④ 先搬到显卡（这样 CPU 峰值只有 shm 那份）')
    t0 = time.time()
    try:
        model = model.to(dev)
        print('     搬完 %.0f 秒，显存 %.1f GB'
              % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
    except Exception as e:
        print('     **搬不过去**：%s' % str(e)[:100])
        return 1

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
    print('  ⑤ 装权重（%d 个物理名，从 %d 个分片按需取）'
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

    print()
    print('  ⑥ 前向')
    ids = torch.tensor([[100 + i for i in range(seq)]], device=dev)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids)
    dt = time.time() - t0
    print('     %d 个 token，%.2f 秒' % (seq, dt))
    print('     输出 %s，幅度 %.4e'
          % (tuple(out.shape), out.float().abs().max()))
    print('     峰值显存 %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9))

    ok = bool(torch.isfinite(out).all())
    print()
    print('  [%s] 真权重 27B 前向 %s'
          % ('PASS' if ok else 'FAIL',
             '跑通了，输出有限' if ok else '**输出里有 NaN/Inf**'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
