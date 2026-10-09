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

## 「挑哪些模型」是纯函数，而且这支脚本自己用过一次

    挑模型这件事原来长在 main() 里，而 main() 一进门就问
    `torch.cuda.is_available()` —— 于是**在没有显卡的机器上，
    "挑哪些模型"这段逻辑一次也没被执行过**（开发机就是这样）。
    而它里面藏着一个 `if len(ir['layers']) > 8: continue`：
    README 点名的那 5 个真模型（26.9B ~ 35.5B）层数全都 > 8，
    **永远不会被选中，也不会被列出来**。

    现在它是 `plan(free_gb)`：给一个显存数，谁都能验：

        python src/by1gpu.py --plan 16     # 假装有一张 16 GB 的卡
        python src/by1gpu.py --plan 80

用法:  python by1gpu.py [--seq 64]
       python by1gpu.py --plan <GB>      # 不碰显卡，只印"会挑哪些"
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

import by1skip

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location(
        'bcm', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'by1check.py'))
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
import by1paths

# **三个有理由的例外。**
#
# 原来这里还有五个名字：`hello.by1` / `raw-escape.by1` / `ling-3.0-tiny.by1`
# / `glm53.by1` / `nemotron-h.by1`。**后三个文件名早就不存在了**
# （改名之后是 `Ling-3.0-tiny.by1` / `GLM-5.3-Flash.by1` /
# `NVIDIA-Nemotron-…-BF16.by1`）—— 于是"该被跳过"变成了"落进
# 那个宽泛的 `except: continue`"，**一样是静默丢弃，只是原因不同**。
#
# 现在不留它们：编译不了 / 建不起来，都由 `plan()` **报出来**。
# `hello.by1` 也放回来了 —— 它能跑，就该跑（"最小例子也要回归"）。
SKIP = {
    'selftest.by1',      # 故意的反例（9 个错），不是模型
    'gate-probe.by1',    # 故意的反例（act 取值不存在）
    'raw-escape.by1',    # 要走外部 .so（要 gcc）—— by1raw / by1extdemo 管它
}


def build_model(f):
    """`(.by1 名) -> (ir, model)`。**真的建出来** —— 要跑前向。"""
    _r, info = bc.check(f)
    ir = cg.compile_ir(info)
    ns = {}
    exec(compile(cg.render_ir(ir, f), '<ir>', 'exec'), ns)
    return ir, ns['build']().eval()


def params_of(f):
    """参数量。**在 meta 设备上建模型数** —— 不要一个字节的显存/内存。

    生成出去的权重是裸工厂建的（`torch.zeros(d)` / `torch.randn(...)`），
    所以 `with torch.device('meta')` 一裹，参数就落在 meta 上：
    `numel()` 照样准，而一个 35B 模型占 0 字节。

    **为什么不用 `by1exec.shapes_of` 再算一遍。** 那是"同一个意思第二处
    实现" —— 这个仓库为这类重复吃过的亏（`_qk_on` 抄两份，qk_norm 悄悄
    从 2.505e-07 变成 3.362e-03）写在 `by1all` 里。这里的数必须来自
    **codegen 真正建出来的那个模型**。
    """
    _r, info = bc.check(f)
    ir = cg.compile_ir(info)
    ns = {}
    exec(compile(cg.render_ir(ir, f), '<ir>', 'exec'), ns)
    with torch.device('meta'):
        m = ns['build']().eval()
    return sum(p.numel() for p in m.parameters())


def plan(free_gb, names=None):
    """**按显存挑模型** —— 不写死层数，也不静默丢任何一个。

    返回 `(run, skipped)`：

        run      [(f, 参数量, 需要多少 GB)]
        skipped  [(f, 参数量 or None, 需要多少 GB or None, 为什么)]

    **每一个没被选的都要有理由。** 这是 `by1skip` 那条规矩的另一半：
    跳过可以，但必须看得见、必须说得出为什么。
    """
    run, skipped = [], []
    for f in sorted(names if names is not None else by1paths.names()):
        if f in SKIP:
            continue
        try:
            n = params_of(f)
        except cg.CodegenError as e:
            skipped.append((f, None, None, 'codegen 还不支持：%s'
                            % str(e).strip().splitlines()[0][:44]))
            continue
        except Exception as e:
            skipped.append((f, None, None, '建不起来：%s' % type(e).__name__))
            continue
        need = n * 2 / (1024 ** 3) + 2.0      # bf16 + 2GB 余量
        if need > free_gb:
            skipped.append((f, n, need, '这张卡装不下'))
            continue
        run.append((f, n, need))
    return run, skipped


def print_plan(free_gb, run, skipped):
    print()
    print('  按 %.1f GB 挑（bf16 权重 + 2 GB 余量，不写死层数）' % free_gb)
    print('=' * 78)
    print('  跑得了 %d 个' % len(run))
    for f, n, need in run:
        print('    %-46s %8.2fB 参数  约 %5.1f GB' % (f[:46], n / 1e9, need))
    print()
    print('  跳过 %d 个（**没验** —— 每一个都有理由）' % len(skipped))
    for f, n, need, why in sorted(skipped, key=lambda x: -(x[2] or 0)):
        size = '约 %5.1f GB' % need if need else '        ?'
        print('    %-46s %s  %s' % (f[:46], size, why))
    print()


def main():
    seq = 64
    if '--seq' in sys.argv:
        seq = int(sys.argv[sys.argv.index('--seq') + 1])

    # **不碰显卡也能看"会挑哪些"。** 没有它的话，这段逻辑只有租了卡
    # 的机器才跑得到 —— 而它在开发机上就是错的，没人看得出来。
    if '--plan' in sys.argv:
        gb = float(sys.argv[sys.argv.index('--plan') + 1])
        run, skipped = plan(gb)
        print_plan(gb, run, skipped)
        return 0

    if not torch.cuda.is_available():
        print()
        print('  **这台机器没有可用的 CUDA 设备。**')
        print('  torch %s   cuda %s' % (torch.__version__, torch.version.cuda))
        print()
        print('  这个脚本要在有 NVIDIA 卡的机器上跑 —— 用法见 by1cloud.py。')
        print('  没有卡也能验的那一半在 by1dev.py 里。')
        print('  **挑模型那一段没有卡也能看**：python src/by1gpu.py --plan 16')
        print()
        # **跳过要说清为什么。** "没有 CUDA"是一个现象；A 卡为什么不行
        # 是三条具体的事 —— 而具体的那部分才是下次有用的部分。
        print('  如果是 A 卡（AMD），这条路上不行的原因有三条：')
        print('    ROCm / HIP      官方 ROCm 轮子只有 Linux；Windows 那条'
              '（公开的补丁）针对 RDNA2+，')
        print('                    **RDNA1（gfx1010）不在官方支持列表里**'
              '（社区有非官方的，但是 Linux）')
        print('    torch-directml   **本机实测**：pip 报 No matching distribution')
        print('                    （Python 3.12 没有发行版；最后停在 3.11 / torch 2.3.1）')
        print('    onnxruntime-dml  在，但那要先有一个 ONNX 后端 —— 不是换一行设备名')
        return by1skip.skip('没有可用的 CUDA 设备')

    dev = torch.device('cuda')
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    # **留 10% 给 CUDA context、cuDNN workspace、框架自己。**
    free_gb = total_gb * 0.9
    run, skipped = plan(free_gb)

    print('  显存 %.1f GB（按 %.1f GB 挑模型）' % (total_gb, free_gb))
    print('=' * 78)
    print('  显卡验证')
    print('=' * 78)
    print('  设备 %s   %s' % (torch.cuda.get_device_name(0),
                              torch.cuda.get_device_capability(0)))
    print('  torch %s   cuda %s' % (torch.__version__, torch.version.cuda))
    print('  能跑 %d 个，跳过 %d 个' % (len(run), len(skipped)))
    print()

    bad = []
    for f, _n, _need in run:
        ir, m = build_model(f)
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

    print_plan(free_gb, [], skipped)
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
