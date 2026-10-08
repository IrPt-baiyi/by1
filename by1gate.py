#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gate -- 取值门的**可证伪对照**。

规则是：「声明了一个 codegen 没实现的取值，编译必须被拒」。

一条只会通过的规则不是规则 —— 所以这里拿**已知该被拒**的和
**已知该通过**的各试一遍。

这个脚本存在的理由和 `selftest.by1` 一样：**门本身也要有判卷人。**

    2026-10 的一次真实教训：GPT-2 原来在这个列表里是"必须被拒"
    （act = gelu_new 没实现）。后来把 gelu 实现了，它变成"必须通过" ——
    **而这个脚本立刻红了**，因为预期没跟着改。那是它该干的事。
"""
import importlib.util
import sys

sp = importlib.util.spec_from_file_location('bcm', 'by1check.py')
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg

# (文件, 是否应当接受, 拒绝理由里应当出现的字样, 为什么)
CASE = [
    # ── 该被拒的 ──────────────────────────────────────────────────
    ('gate-probe.by1', False, 'nosuchactivation',
     '故意的反例：act 是个不存在的取值'),
    ('nemotron-h.by1', False, 'SSM', 'Mamba 整族没实现'),
    ('ling-3.0-tiny.by1', False, 'KDA', 'KDA 整族没实现'),
    # ── 该通过的 ──────────────────────────────────────────────────
    ('clef-tiny.by1', True, '', '取值全都实现过'),
    ('step-3.7-flash.by1', True, '', 'sigmoid_topk 已实现'),
    ('gpt2.by1', True, '',
     'gelu / LayerNorm / 学习式位置 / 无门控 MLP 现在都实现了'),
]

bad = []
for f, want_ok, needle, why in CASE:
    try:
        _r, info = bc.check(f)
    except Exception as e:
        bad.append('%s: 检查器就没过 %s' % (f, str(e)[:40]))
        continue
    try:
        cg.compile_ir(info)
        got, msg = True, ''
    except cg.CodegenError as e:
        got, msg = False, str(e)

    if got == want_ok:
        detail = '接受' if got else '拒绝：' + msg.strip().replace('\n', ' ')[:52]
        print('  ok %-22s %s' % (f, detail))
    else:
        print('  !! %-22s **预期%s，实际%s** —— %s'
              % (f, '接受' if want_ok else '拒绝',
                 '接受' if got else '拒绝', why))
        bad.append(f)

    if not got and needle and needle not in msg:
        print('         拒绝理由里没提到 %s' % needle)
        bad.append(f + '(理由)')

print()
print('  [%s] 取值门 %s'
      % ('PASS' if not bad else 'FAIL',
         '工作正常' if not bad else '坏了：%s' % ', '.join(bad)))
sys.exit(0 if not bad else 1)
