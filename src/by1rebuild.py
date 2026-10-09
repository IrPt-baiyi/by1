#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1rebuild -- **删掉数据，能不能重建。**

## 用户要的性质

> 如果这个仓库 95% 的数据（不是代码）全部被删除，
> **必须能够迅速重建**。
> 这使得我们的数据**不依赖硬编码**。

## 判据：每个数据文件都要能回答三个问题

    ① 它从哪来？（URL 规则 / 生成器 / 手写）
    ② 怎么重建？（一条命令）
    ③ 重建要多久？

**答不出①的就是硬编码的知识。**

## "代码 vs 数据"的界线

    代码   by1*.py           —— 概念、算法、规则
    数据   refs/*.json       —— 从 HF / ModelScope 抓的产物描述
           *.by1             —— **手写的模型声明**
           models.tsv        —— **手写的短名表**
           *.md / VERSION    —— 生成物或文档

## 最重要的那一格

`*.by1` 是**手写的** —— 而它不是随便写的，它是这个项目的知识：

    mech Attn : Attention { heads = {kv=8, head_dim=128}, qk_norm = true }

**这一行是人的判断**，产物里推不出来（`qk_norm` 在 config 里可能写作
`q_norm`，也可能根本没有）。

所以真正该问的不是"能不能重建 `.by1`"，而是：

    **`.by1` 里有多少是"从产物可推的"，多少是"人的判断"？**

如果可推的占大多数，那"重建"就是：**推导 + 补上少数判断**。

用法:  python by1rebuild.py           只报告
       python by1rebuild.py --measure 顺带量重建耗时
"""
import glob
import io
import os
import re
import sys
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def size(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def main():
    print()
    print('=' * 84)
    print('  删掉数据，能不能重建')
    print('=' * 84)

    # ── 数据分类 ──────────────────────────────────────────────────
    kinds = {
        'refs/*.config.json': ('抓', 'HF / ModelScope 的 config.json'),
        'refs/*.tensors.json': ('抓', 'HF 的 index.json 或 safetensors 头'),
        'refs/*.gguf-tensors.json': ('抓', 'llama.cpp 的 GGUF 张量表'),
        '*.by1': ('手写', '**模型声明 —— 这个项目的知识本身**'),
        'models.tsv': ('手写', '长名/短名/HF id 对照'),
        # **这两个原本写的是 `ir.md` / `plan.md`** —— 它们早就在
        # `history/` 里了，于是 glob 落空、整行被静默跳过：
        # 报表里少了一整类"不可重建"的数据，而没人看得出来。
        'history/ir.md': ('日志', '**过程记录，不可重建**'),
        'history/plan.md': ('文档', '计划'),
        'README.md': ('文档', '门面'),
        '1.md': ('文档', '宣言'),
    }
    gen = {
        'models.md': 'python src/by1cmp.py --md',
        'ir-spec.md': 'python src/by1ir.py --spec',
        'VERSION': 'python src/by1ver.py --write',
        'models.tsv': None,
    }

    print()
    print('  %-28s %10s  %-6s %s' % ('数据', '大小', '来源', '怎么重建'))
    print('  ' + '-' * 96)
    total = 0
    hand = 0
    for pat, (src, note) in kinds.items():
        # **数据在哪：`.by1` 进了 `models/`，其余在仓库根。**
        # 键名是给人看的标签，不是路径 —— 解析规则只写这一处。
        if pat.endswith('.by1'):
            fs = by1paths.models(pat)
        elif pat.startswith('refs/'):
            fs = glob.glob(by1paths.ref(pat[len('refs/'):]))
        else:
            fs = glob.glob(by1paths.root(pat))
        if not fs:
            continue
        sz = sum(size(f) for f in fs)
        total += sz
        if src == '手写':
            hand += sz
        how = {'抓': '按 by1refs 的规则重抓（URL 可推）',
               '手写': '**只能重写**',
               '日志': '**不可重建**',
               '文档': '**只能重写**'}[src]
        print('  %-28s %8.1f MB  %-6s %s'
              % ('%s（%d 个）' % (pat, len(fs)), sz / 1e6, src, how))
    print('  ' + '-' * 96)
    print('  %-28s %8.1f MB' % ('合计', total / 1e6))
    print('  %-28s %8.1f MB  （手写 %d%%）'
          % ('其中手写的', hand / 1e6, 100 * hand / max(total, 1)))

    # ── 抓得到吗：URL 规则是不是可推 ─────────────────────────────
    print()
    print('  ── 抓的那部分，URL 能不能从 .by1 推出来')
    n_ok, n_bad = 0, []
    for f in sorted(by1paths.models()):
        head = by1io.head_text(f)
        if 'by1-repo' not in head:
            continue
        m = re.search(r'^#\s*by1-repo:\s*(\S+)', head, re.M)
        if m:
            n_ok += 1
        else:
            n_bad.append(f)
    print('     能推的：%d 个' % n_ok)
    if n_bad:
        print('     **推不出的：%d 个**：%s' % (len(n_bad), ', '.join(n_bad)))

    # ── 手写那部分：有多少是可推的 ───────────────────────────────
    print()
    print('  ── `.by1` 里有多少是"从产物可推"、多少是"人的判断"')
    print('     这是整件事的核心：可推的多，重建就是"推导 + 补少量判断"')
    tot_bytes = sum(size(f) for f in by1paths.models())
    # 粗判：含 `#` 注释的行 = 人的解释；含 `mech`/`stack` 的行 = 声明
    n_comment, n_decl, n_attr = 0, 0, 0
    for f in by1paths.models():
        for line in by1io.iter_lines(f, encoding='utf-8'):
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                n_comment += 1
            elif s.startswith('mech') or s.startswith('stack'):
                n_decl += 1
            else:
                n_attr += 1
    print('     .by1 共 %.2f MB，%d 行' % (tot_bytes / 1e6,
                                           n_comment + n_decl + n_attr))
    print('       注释（人的解释）  %5d 行' % n_comment)
    print('       声明（mech/stack）%5d 行' % n_decl)
    print('       属性（值）        %5d 行' % n_attr)
    print()
    print('     **属性那部分，大部分 `by1boot` 能从 config+tensors 推出来** ——')
    print('     推不出来的（`qk_norm` 这种产物里没有的）才是真判断。')
    print('     怎么量：拿 by1boot 从 config+tensors 生成 IR，')
    print('     和手写 .by1 的 IR 对比 —— **那个 diff 就是"人的判断"**。')
    print('     这正是 `by1bootir.py` 在做的事，只是没人把它当"重建率"看。')

    # ── 重建的实际命令 ───────────────────────────────────────────
    print()
    print('  ── 重建的命令（如果数据被删了，按这个顺序跑）')
    print('''
     ① 抓 config + 张量表          python src/by1fetch.py
        （按 .by1 头部的 by1-repo 推 URL；.by1 也删了就按 models.tsv）
     ② 重建 .by1 里可推的部分      python src/by1boot.py <config> <tensors> --emit-ir
        （by1boot 已有；**缺的是 IR -> .by1 的写回**）
     ③ 补上人的判断                （对照 src/by1bootir.py 的 diff）
     ④ 重新生成派生文件            python src/by1cmp.py --md
                                   python src/by1ir.py --spec
                                   python src/by1ver.py --write
     ⑤ 全量验证                    python src/by1all.py
''')
    print('  **② 是缺的那一环**：`by1boot` 是 config -> IR，')
    print('  而要把 IR 写回 `.by1` 没有工具。**那是"迅速重建"的瓶颈。**')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
