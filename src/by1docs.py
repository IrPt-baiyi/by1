#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文档里提到的每一个路径，都还在吗？

## 为什么

`by1blind.py` 里我列过一条盲区：

    E. 文档同步：生成物（ir-spec / models.md）和源头还一致吗

而那一条只查了**生成物**。文档里**手写的路径**从来没查过 ——
于是 README 里 `refs/clef.tensors.json` 这个例子在文件名统一之后
就失效了，而没人知道。

## 搬家之后补上的两件事

**① 路径按新布局解析。** 代码在 `src/`、模型在 `models/`。
规则不在这里重写一遍 —— 问 `by1paths`（同一个意思不写两处）。

**② markdown 链接也要查。** 原来只认 `refs/xxx.json` 和 `xxx.by1`
这两种字面量，于是 README 里 `[ir.md](ir.md)` 这种**链接**一直是坏的：
`ir.md` 实际在 `history/` 里。**一个只认两种写法的检查器，第三种写法
就是它的盲区** —— 而这一条盲区是我自己写下的那句话的现成例子。

## 扫什么

仓库里所有 `.md`（根目录 + `docs/` `history/` `drafts/` `gpu/`），
每一处路径、链接、`python xxx.py` 调用，逐条看目标在不在。
**只报告，不改。**
"""
import os
import re
import sys

import by1io
import by1paths

ROOT = by1paths.ROOT

# 文档里写的路径（字面量）
PAT_PATH = re.compile(r"(refs/[\w.\-]+\.json|[\w.\-]+\.by1)")
# markdown 链接
PAT_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# 文档里对脚本的调用
PAT_CALL = re.compile(r"python\s+([\w./\\-]+\.py)")

SKIP_LINK = ('http://', 'https://', '#', 'mailto:')


def md_files():
    """仓库里所有 .md，含子目录。**原来只扫代码旁边那一层。**

    `history/` **不扫**：那是日志，老条目里的老路径是历史事实
    （`by1name.py` 里写着这条纪律）。把它扫进来，只会给这个检查器
    加一层永远清不掉的噪音 —— 而一个总有几条红的检查器，
    读的人会连真的那条一起跳过。
    """
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in ('.git', '__pycache__', 'history')]
        for f in files:
            if f.endswith('.md'):
                out.append(os.path.join(base, f))
    return sorted(out)


def find(target, near=None):
    """这条引用指的是哪个真实文件？找不到返回 None。

    顺序：文档旁边的相对路径 -> 仓库根 -> `models/`（`*.by1`）
          -> `refs/`（`refs/xxx.json`）-> `drafts/` -> `src/` `gpu/`（脚本）
    """
    cands = []
    base = os.path.dirname(target)
    if base.startswith('refs/'):
        cands.append(by1paths.ref(base[len('refs/'):]))
        cands.append(by1paths.ref(os.path.basename(target)))
    if near:
        cands.append(os.path.normpath(os.path.join(near, target)))
    cands.append(os.path.join(ROOT, target))
    if target.endswith('.by1'):
        cands.append(by1paths.model(target))
        cands.append(os.path.join(ROOT, 'drafts', os.path.basename(target)))
    if target.endswith('.py'):
        cands.append(by1paths.tool(os.path.basename(target)))
        cands.append(os.path.join(ROOT, 'gpu', os.path.basename(target)))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    # **这两行原来在模块级** —— import 这个模块会先印一个横幅出来。
    # 它就是"import 即执行"最轻的一种：不 crash，但谁 import 谁脏输出。
    print()
    print('=' * 76)
    print('  文档里写的路径，还在吗')
    print('=' * 76)
    print()

    files = md_files()
    total, bad = 0, []
    for p in files:
        rel = by1paths.rel(p)
        near = os.path.dirname(p)
        t = by1io.read_text(p, errors='replace')
        seen = {}
        for m in PAT_PATH.finditer(t):
            seen.setdefault(m.group(1), ('路径', t[:m.start()].count('\n') + 1))
        # **链接也要看** —— 这是原来漏掉的那一类。
        for m in PAT_LINK.finditer(t):
            tgt = m.group(1).strip()
            if tgt.startswith(SKIP_LINK):
                continue
            seen.setdefault(tgt.split('#')[0],
                            ('链接', t[:m.start()].count('\n') + 1))
        if not seen:
            continue
        miss = [(k, ln, kind) for k, (kind, ln) in seen.items()
                if not find(k, near)]
        total += len(seen)
        if miss:
            print('  %s（%d 处引用）' % (rel, len(seen)))
            for k, ln, kind in sorted(miss, key=lambda x: x[1]):
                print('     **%s:%d**  %s  %s  ← 不存在' % (rel, ln, kind, k))
            print()
            bad.extend((rel, ln, k) for k, ln, _kind in miss)

    print('  扫了 %d 个 .md、%d 处引用，**%d 处指向不存在的文件**'
          % (len(files), total, len(bad)))
    print('  （`history/` 故意不扫：那是日志，里面的老路径是历史事实。）')
    print()

    # 顺带：文档里对脚本的调用，脚本在吗
    print('  ── 顺带：文档里的 `python xxx.py` 调用')
    calls = {}
    for p in files:
        t = by1io.read_text(p, errors='replace')
        for m in PAT_CALL.finditer(t):
            # **把哪个文件提到的也记下来** —— 脚本可能就在那份文档旁边
            # （docs/delete-test-1/rebuild_manifest.py 就是这种）。
            # 不带目录去查会把它误报成不存在。
            calls.setdefault(m.group(1), {})[by1paths.rel(p)] = os.path.dirname(p)
    missing = [k for k in calls
               if not any(find(k, d) for d in calls[k].values())]
    print('     %d 个不同的脚本被提到' % len(calls))
    if missing:
        for k in sorted(missing):
            print('     **%s** 不存在（在 %s 里被提到）'
                  % (k, ', '.join(sorted(calls[k]))))
    else:
        print('     全部都存在 ✓')
    print()
    print('  [%s] 文档路径 %s'
          % ('PASS' if not bad and not missing else 'FAIL',
             '路径、链接、脚本调用都在' if not bad and not missing
             else '%d 处路径 + %d 个脚本 有问题' % (len(bad), len(missing))))
    print()
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
