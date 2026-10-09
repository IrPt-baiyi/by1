#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1pat -- **哪些词是数据里的，哪些是我的想象。**

## 它量的是什么

`by1boot` 里的识别词（塔名 / 辅助栈名）分两张表：

    TOWER_WORDS             在 34 份张量清单里**出现过**
    TOWER_WORDS_UNVERIFIED  **一次都没出现过** —— 是我的想象

这个脚本去数每一个词，确认那张分类是对的。
**分类一旦错了（把想象的放进已验证），它就永远不会被发现。**

## 这个脚本自己错了四次 —— 而那是这件事最值得记的部分

原来的写法是一整条正则字符串，我想知道"哪几个词是我想象的"，
于是去数。**数了四次，四个数：**

    一  拿整条正则去数        死的分支藏在活的正则里。报"活 12"
    二  按 `|` 切开当子串测     `(^|\\.)visual` 剥壳剩 `)(visual`。
                              报"死 21"
    三  剥得更"干净"一点       剥掉的东西更多。报"死 6"
    四  按 `|` 切开当正则编     **`(^|\\.)` 这个组内的 `|` 也被切了**。
                              报"死 12"

**四次都测的是同一件事。**

## 结论不是"再修一次量法"，是"改写法"

**词是数据，正则是渲染。**

`by1boot` 现在写的是 `TOWER_WORDS = [(名字, [词...])]`，
正则由 `_render()` 拼出来。"有哪些词"永远是数据 ——
**谁都不用拆字符串，也就没有拆不干净的问题。**

> 一个量了四次都量不准的东西，通常不是量法的问题，
> 是**被量的那个东西没有把信息留出来**。
"""
import glob
import importlib.util
import os
import sys
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def main():
    s = importlib.util.spec_from_file_location(
        'by1boot', os.path.join(HERE, 'by1boot.py'))
    boot = importlib.util.module_from_spec(s)
    s.loader.exec_module(boot)

    alls = {}
    for f in sorted(glob.glob(by1paths.ref('*.tensors.json'))):
        try:
            alls[os.path.basename(f).replace('.tensors.json', '')] = \
                list(by1io.read_json(f, encoding='utf-8'))
        except Exception:
            continue
    total = sum(len(v) for v in alls.values())

    print()
    print('=' * 80)
    print('  哪些词是数据里的，哪些是我的想象')
    print('=' * 80)
    print()
    print('  底数：%d 份张量清单，%d 个物理名' % (len(alls), total))
    print()

    bad = []
    for tbl, claim in (('TOWER_WORDS', '已验证'),
                       ('TOWER_WORDS_UNVERIFIED', '未验证'),
                       ('AUX_WORDS', '已验证'),
                       ('AUX_WORDS_UNVERIFIED', '未验证')):
        groups = getattr(boot, tbl, []) or []
        if not groups:
            continue
        print('  ── %s（声称：%s，%d 组）' % (tbl, claim, len(groups)))
        for name, words in groups:
            for w in words:
                n = sum(1 for names in alls.values()
                        if any(w in x for x in names))
                if claim == '已验证' and n == 0:
                    bad.append((tbl, w, '声称已验证', '实测一次都没出现'))
                if claim == '未验证' and n > 0:
                    bad.append((tbl, w, '声称未验证',
                                '实测 %d 个模型有' % n))
                mark = '%2d 个模型' % n if n else '**一次都没出现**'
                print('      %-10s %-24s %s' % (name, w, mark))
        print()

    print('=' * 80)
    if bad:
        print('  **分类错了 %d 条：**' % len(bad))
        for tbl, w, c, m in bad:
            print('    %-24s %-22s %s，%s' % (tbl, w, c, m))
    else:
        print('  **分类全对**：说已验证的都有命中，说未验证的都是 0。')
    print('=' * 80)
    print()
    print('  "未验证"不代表错 —— `nextn`（DeepSeek 的 MTP 叫法）、')
    print('  `whisper` / `eagle` / `medusa` 都真实存在，只是这 34 个模型里没有。')
    print()
    print('  但它代表**从没被验证过** —— 所以分开放，不混在一起用同样的语气。')
    print()
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
