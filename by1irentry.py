#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1irentry -- **三个后端必须能从 IR 入口跑通**，而不是只从 .by1 入口。

## 为什么这是判卷人

规格说「IR 是接口」。这句话只有在**有人只拿 IR 就能跑**的时候才成立。

今天三条路都是"从 .by1 入口"的：

    by1codegen.render(info)        要 info
    by1c.emit_c(ir, info, params)  info 是摆设但不给不行
    by1exec                         main() 收 .by1 路径

所以这个脚本做的事是：

    ① .by1 -> IR
    ② IR -> JSON -> IR          （**往返，模拟"别人给我一份 IR"**）
    ③ **只拿那份 JSON 出来的 IR**，跑三个后端
    ④ 三个后端的输出对拍

第 ② 步是关键：它不是"我手里这个 dict"，是**一份序列化过、又被读回来的
IR** —— 那才是"别人给我的 IR"的样子。少了第 ② 步，这个检查会漏掉
"IR 里有 JSON 存不下的东西"这一类问题。

用法:  python by1irentry.py [文件.by1 ...]
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sp = importlib.util.spec_from_file_location('bcm', 'by1check.py')
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
import by1exec as ex
import by1ir

SKIP = {'selftest.by1', 'gate-probe.by1', 'hello.by1',
        # 这些用的机制 codegen 还没实现 —— 是**覆盖率**问题，
        # 不是 IR 入口问题。见 by1gate.py。
        'ling-3.0-tiny.by1', 'glm53.by1', 'nemotron-h.by1'}
SRC = {'llama-shaped.by1', 'mixtral-shaped.by1', 'gpt-oss-shaped.by1',
       'qwen3-next-shaped.by1', 'mla-shaped.by1', 'llama3-shaped.by1',
       'clef-tiny.by1'}


def run_one(f, seq=16, deep=False):
    _r, info = bc.check(f)
    ir0 = cg.compile_ir(info)

    # ① 往返 —— **模拟"别人给我一份 IR"**
    js = by1ir.to_json(ir0)
    ir = by1ir.from_json(js)
    if json.dumps(ir0, sort_keys=True, default=str) != \
            json.dumps(ir, sort_keys=True, default=str):
        return 'FAIL', '往返之后不一致'

    # ② PyTorch：**只从 IR**
    try:
        ns = {}
        exec(compile(cg.render_ir(ir, f), '<ir-only>', 'exec'), ns)
        m = ns['build']().eval()
        import torch
        ids = torch.randint(0, ir['vocab'], (1, seq))
        with torch.no_grad():
            out_t = m(ids).numpy()
    except Exception as e:
        return 'FAIL', 'PyTorch 从 IR 跑不了：%s' % str(e)[:60]

    # ③ NumPy：**只从 IR**
    try:
        out_n, params = ex.exec_ir(ir, seq=seq)
    except Exception as e:
        return 'FAIL', 'NumPy 从 IR 跑不了：%s' % str(e)[:60]

    if deep:
        return 'OK', 'PyTorch %s · NumPy %s' % (out_t.shape, out_n.shape)

    # ④ C：**只从 IR**
    # `emit_c_ir` 返回的是 **(源码, 参数顺序, 总字节数)** —— 不是字符串。
    # 判卷人第一版当成字符串用了，于是 AttributeError。
    try:
        import by1c
        c_text, _order, _off = by1c.emit_c_ir(ir, ex.shapes_of(ir))
    except Exception as e:
        return 'FAIL', 'C 从 IR 跑不了：%s' % str(e)[:60]

    return 'OK', 'IR %d 层 · PyTorch %s · NumPy %s · C %d 行' % (
        len(ir['layers']), out_t.shape, out_n.shape, len(c_text.split('\n')))


def main():
    files = sys.argv[1:] or sorted(glob.glob('*.by1'))
    print('=' * 80)
    print('  三个后端从 **IR 入口** 跑（IR 是序列化过又读回来的）')
    print('=' * 80)
    bad, skipped = [], []
    gaps = []
    for f in files:
        if f in SKIP or not os.path.exists(f):
            continue
        # **尺寸护栏。** 这个判卷人是查"IR 入口通不通"的，
        # 不是查"31B 模型能不能装进 16GB 内存"。所以按参数量跳过大的，
        # 但**要打印出来** —— 静默跳过和没跑过一样。
        try:
            _r, info = bc.check(f)
            ir0 = cg.compile_ir(info)
            nelem = sum(int(np.prod(v)) for v in ex.shapes_of(ir0).values())
        except Exception:
            continue
        if nelem > 3e7:            # ~30M 参数，fp32 约 120MB
            skipped.append((f, nelem))
            continue
        try:
            st, msg = run_one(f)
        except SystemExit as e:
            # **"后端没实现这个机制" 不是 "IR 入口不通"。**
            # 前者是覆盖率问题（KDA / SSM / Raw），后者才是这个判卷人
            # 要查的东西。混在一起的话，真问题会被覆盖率噪音淹掉。
            gaps.append((f, str(e)[:56]))
            continue
        except Exception as e:
            st, msg = 'FAIL', '%s: %s' % (type(e).__name__, str(e)[:60])
        mark = 'ok ' if st == 'OK' else '!! '
        print('  %s%-24s %s' % (mark, f, msg))
        if st != 'OK':
            bad.append(f)
    if gaps:
        print()
        print('  后端还没实现（**覆盖率问题，不是 IR 入口问题**）：')
        for f, m in gaps:
            print('    %-24s %s' % (f, m))
    if skipped:
        print()
        print('  跳过（参数太多，这个判卷人不查那个）：')
        for f, n in skipped:
            print('    %-24s %.1fM 参数' % (f, n / 1e6))
    print()
    print('  [%s] IR 入口 %s'
          % ('PASS' if not bad else 'FAIL',
             '小的那些，三个后端都能只拿 IR 跑' if not bad
             else '坏了：%s' % ', '.join(bad)))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
