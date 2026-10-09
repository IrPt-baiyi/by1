#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1triage -- **把一批新 config 过一遍 by1，看谁需要新原语。**

这是"加模型"唯一靠谱的起点：**先让它拒，再决定做什么。**
不预判哪个模型"应该有"什么机制 —— 让工具说。

## 判据

    ✓ 已覆盖    boot 能出 IR，而且全是现成机制
    ! 猜了      boot 出得来，但有 guessed 的字段（**产物里看不出来的**）
    ✗ 拒了      缺机制 / 缺类型 / 解析不了 —— **这才是要干活的地方**
    ✗ 文件错    config 本身有问题（比如 gated repo 下下来的是错误页）

用法:  python by1triage.py [refs/xxx.config.json ...]
"""
import glob
import importlib.util
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name):
    s = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + '.py'))
    m = importlib.module_from_spec(s) if hasattr(importlib, 'module_from_spec') \
        else importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    import importlib.util as iu

    def ld(name):
        s = iu.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
        m = iu.module_from_spec(s)
        s.loader.exec_module(m)
        return m

    boot = ld('by1boot')

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # 只要**新抓的**：时间上最新的那些，或者全都要
        files = sorted(glob.glob(os.path.join(HERE, 'refs', '*.config.json')))

    print()
    print('=' * 92)
    print('  新 config 过 by1 —— **谁需要新原语**')
    print('=' * 92)
    print()

    rows = []
    for f in files:
        name = os.path.basename(f).replace('.config.json', '')
        try:
            cfg = json.load(io.open(f, encoding='utf-8'))
        except Exception as e:
            rows.append((name, '文件错', str(e)[:40]))
            continue
        if not isinstance(cfg, dict) or 'model_type' not in cfg:
            rows.append((name, '文件错', '没有 model_type（可能是错误页）'))
            continue
        try:
            out = boot.boot_ir(cfg, {}, name)
            ir, guessed, gkeys = out if isinstance(out, tuple) else (out, [], [])
            kinds = set()
            for L in (ir.get('layers') or []):
                for o in (L.get('ops') or []):
                    kinds.add(o.get('kind'))
            # **"没崩"不等于"覆盖了"。**
            #
            # `classify()` 是从**张量名**认机制的。只给 config 的话，
            # 它认不出任何机制，退化成 `Add+Norm+Raw` —— 而第一版
            # triage 把这个当成了"✓ 已覆盖"，于是 37/39 全绿，
            # 连 gpt-oss-120b 都是 `Add+Norm+Raw`。
            #
            # **一个什么机制都没认出来的 IR，和"不支持"没区别，
            # 但报的是"✓"。** 那正是这个项目一直在防的那种谎。
            mechs = {k for k in kinds
                     if k not in ('Add', 'Norm', 'Raw', 'Embed', 'Head')}
            note = '+'.join(sorted(k for k in kinds if k))[:40]
            if not mechs:
                rows.append((name, '空壳拒了',
                             '只有 %s —— **没给张量名，认不出机制**'
                             % (note or '无')))
            elif gkeys:
                rows.append((name, '猜了%d' % len(gkeys), note))
            else:
                rows.append((name, '✓', note))
        except SystemExit as e:
            rows.append((name, '拒了', str(e)[:56]))
        except Exception as e:
            rows.append((name, '拒了', '%s: %s' % (type(e).__name__, str(e)[:44])))

    order = {'✓': 0, '猜了': 1}
    rows.sort(key=lambda r: (0 if r[1] == '✓' else
                             1 if r[1].startswith('猜') else 2, r[0]))
    n_ok = sum(1 for r in rows if r[1] == '✓')
    for name, verdict, note in rows:
        mark = {'✓': ' ✓ ', '文件错': ' ✗ '}.get(verdict, ' ! ')
        print('  %s %-46s %-8s %s' % (mark, name[:46], verdict[:8], note))

    print()
    print('  %d 个 config：%d 个已覆盖，%d 个要干活'
          % (len(rows), n_ok, len(rows) - n_ok))
    print()
    print('  **"✓" 不等于"能算"** —— 那只是 boot 能出 IR。')
    print('  要证明能算，还得走 codegen / exec / C 三后端。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
