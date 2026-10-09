#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1bootir -- **两条路的判卷人**：从 .by1 出来的 IR，和从产物反推出来的 IR。

    路 A：  .by1        --by1check+compile_ir-->  IR
    路 B：  config+张量  --by1boot.boot_ir------>  IR

两条路都应该通向**同一个东西** —— 因为它们描述的是同一个模型。

## 比什么

不比字节（路 B 填了一堆默认值，路 A 是从语言里读的）。比**结构骨架**：

    ✓ 顶层：vocab / d_model / ctx / pos_kind / norm_kind / norm_eps
    ✓ 每层有哪些算子（kind 序列）
    ✓ 每一个算子的属性 —— **但只比两边都"知道"的那些**

第三项是关键：路 B 里 `qk_norm = 'off'` 是**猜的**（产物里看不出来），
路 A 里是真读出来的。拿猜的去比真的，比出来的差异不是 bug。
所以：**只比路 A 有的、且路 B 不是猜的那些。**

用法:  python by1bootir.py <文件>.by1 <config.json> <tensors.json>
"""
import importlib.util
import json
import sys

sys.path.insert(0, '.')
sp = importlib.util.spec_from_file_location('bcm', 'by1check.py')
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
import by1boot


def main():
    # **无参数 = 跑默认那三对**（给 by1all 用）。
    # 三个模型横跨两代：minimind-3 / instella-3b 是现代 config，
    # gpt2 是上一代 —— 字段名、归一化、位置全都不同。
    #
    # **路径不再写死，从 `.by1` 头部的 `# by1-repo:` 推。**
    # 写死的话会各自过期：我把 `refs/minimind-3.tensors.json` 改名成
    # `refs/jingyaogong__minimind-3.tensors.json`，改名脚本知道，
    # **但这里的三行字符串不知道** ——
    # 症状是 `FileNotFoundError: refs/minimind-3.tensors.json`，
    # **看起来像文件丢了，其实是引用没跟上。**
    import os as _os
    import by1refs as _refs

    def _pair(by1):
        c, t = _refs.paths(by1, 'config'), _refs.paths(by1, 'tensors')
        return (by1, c, t) if (c and t) else None

    PAIRS = [p for p in (_pair('minimind-3.by1'), _pair('Instella-3B.by1'),
                         _pair('gpt2.by1')) if p]
    pairs = ([tuple(sys.argv[1:4])] if len(sys.argv) >= 4 else PAIRS)
    bad = 0
    for f, cfg_p, ten_p in pairs:
        bad |= _one(f, cfg_p, ten_p)
    print()
    print('  [%s] 两条路 %s' % ('PASS' if not bad else 'FAIL',
                                '通向同一个 IR' if not bad else '有对不上的'))
    return bad


def _one(f, cfg_p, ten_p):

    _r, info = bc.check(f)
    ir_a = cg.compile_ir(info)
    cfg = json.load(open(cfg_p, encoding='utf-8'))
    cfg = cfg.get('text_config', cfg)
    real = json.load(open(ten_p, encoding='utf-8'))
    ir_b, guessed, gset = by1boot.boot_ir(cfg, real)

    print('=' * 80)
    print('  两条路的 IR 对拍：%s' % f)
    print('=' * 80)
    print()

    diffs, same = [], 0

    # ① 顶层
    for k in ('vocab', 'd_model', 'ctx', 'pos_kind', 'norm_kind', 'norm_eps'):
        a, b = ir_a.get(k), ir_b.get(k)
        if a == b:
            same += 1
        else:
            diffs.append(('顶层', k, a, b))

    # ② 逐层算子 kind 序列
    na, nb = len(ir_a['layers']), len(ir_b['layers'])
    if na != nb:
        diffs.append(('层数', '', na, nb))
    for i in range(min(na, nb)):
        ka = [o['kind'] for o in ir_a['layers'][i]['ops']]
        kb = [o['kind'] for o in ir_b['layers'][i]['ops']]
        if ka == kb:
            same += 1
        else:
            diffs.append(('L%d 算子序列' % i, '', ','.join(ka), ','.join(kb)))

    # ③ 属性：**只比路 A 有的、且路 B 不是猜的**
    skipped = 0
    for i in range(min(na, nb)):
        for oa, ob in zip(ir_a['layers'][i]['ops'], ir_b['layers'][i]['ops']):
            if oa['kind'] != ob['kind']:
                continue
            for k, va in (oa.get('attrs') or {}).items():
                if ("L%d.%s.%s" % (i, oa['kind'], k)) in gset:
                    skipped += 1
                    continue
                if k not in (ob.get('attrs') or {}):
                    # **路 B 没写这个字段就不比。** 规格里有一堆可选属性
                    # （norm_eps / norm_one_plus 之类），只有一边写了不是差异。
                    # 第一版把"缺"当成一个值去比，于是满屏假差异。
                    continue
                vb = ob['attrs'][k]
                if va == vb:
                    same += 1
                else:
                    diffs.append(('L%d %s' % (i, oa['kind']), k, va, vb))

    print('  一致 %d 处 · 差异 %d 处 · 跳过（路 B 是猜的）%d 处' %
          (same, len(diffs), skipped))
    print()
    if diffs:
        print('  ── 差异 ──')
        for where, k, a, b in diffs[:20]:
            print('    %-22s %-14s .by1 %-22s 产物 %s'
                  % (where, k, repr(a)[:22], repr(b)[:22]))
        if len(diffs) > 20:
            print('    …另有 %d 处' % (len(diffs) - 20))
    print()
    return 0 if not diffs else 1


if __name__ == '__main__':
    sys.exit(main())
