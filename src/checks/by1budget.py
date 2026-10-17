#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1budget -- **门禁耗时的判据**：跑得慢过上限就不再是"通过"。

## 为什么要它

by1all 一直在算各段耗时、也一直打印"合计 N 秒"，但**没有任何上限**。
一个判卷人从 3 秒变成 30 秒，门禁照样绿 —— 只是没人再愿意跑它。
"慢到没人跑"和"没写"是同一种结果。

## 判据（表里的数是量出来的，不是拍的）

    档 1（纯标准库）      实测 18.0 秒 -> 上限  54 秒（3x）
    档 3（+ numpy）       实测 48.1 秒 -> 上限 144 秒（3x）
    档 5（+ gcc / 下载）  实测 115.4 秒 -> 上限 346 秒（3x）

留 3 倍是给别的机器的：CI 的 ubuntu / windows 都比本机慢，而这条判据要抓的是
**数量级回归**（某一段从 3 秒变 30 秒），不是 10% 的抖动。真要临时放宽，设
环境变量 `BY1_BUDGET_SCALE`（默认 1.0）。

## 两条规则

    B1 总量    总耗时超过上限 -> 红，并指出**最慢的那一段**（它占多少）
    B2 单段    单段超过总量的 70% -> 红（一段吃掉大半，说明并行/结构坏了）

档 2 / 4 / 6 没有实测基线：**不假装有上限**，只印一行"未登记"，不算通过也不算失败。

## 自检

`--selftest` 用假数据把 B1 / B2 各弄红一次，再验三条正例（刚好在上限内、
未知档位、单段占 60%）。判据自己得先有判据。
"""

import os
import sys

#: 实测基线（本机，2026-10-17，档 3 = 48.1 秒 / 档 5 = 115.4 秒 / 档 1 = 18.0 秒）。
MEASURED = {1: 18.0, 3: 48.1, 5: 115.4}

#: 上限 = 实测 x FACTOR。
FACTOR = 3.0

#: 单段占比红线。
SECTION_SHARE = 0.70


def limit(tier, scale=None):
    """该档的总耗时上限（秒）。没登记的档位返回 None。"""
    base = MEASURED.get(tier)
    if base is None:
        return None
    if scale is None:
        try:
            scale = float(os.environ.get('BY1_BUDGET_SCALE', '1.0'))
        except ValueError:
            scale = 1.0
    return base * FACTOR * max(scale, 0.01)


def evaluate(tier, total, sections=None, scale=None):
    """返回 (problems, notes)。problems 非空 = 红；notes 是"没验"的说明。"""
    problems = []
    notes = []
    cap = limit(tier, scale)
    if cap is None:
        notes.append('档 %d 没有实测基线 —— 不假装有上限' % tier)
        return problems, notes
    sections = list(sections or ())
    slowest = max(sections, key=lambda x: x[1]) if sections else None
    if total > cap:
        extra = ''
        if slowest is not None:
            extra = '；最慢的一段是 %s（%.1f 秒，占 %.0f%%）' % (
                slowest[0], slowest[1],
                100.0 * slowest[1] / max(total, 0.01))
        problems.append('B1 总耗时 %.1f 秒 > 上限 %.1f 秒（实测基线 %.1f x %.0f）%s'
                        % (total, cap, MEASURED[tier], FACTOR, extra))
    # **B2 只在有两段以上时成立**：档 1 只有"0. 静态"一段，它占 100% 是应该的，
    # 不是"一段吃掉大半"。CI 的档 1 腿就是这么红的（实测）。
    if len(sections) >= 2 and total > 0:
        for label, sec in sections:
            share = sec / total
            if share > SECTION_SHARE:
                problems.append('B2 单段 %s 占 %.0f%%（> %.0f%%）—— 一段吃掉大半，'
                                '说明并行或分段坏了' % (label, 100 * share,
                                                        100 * SECTION_SHARE))
    return problems, notes


def _selftest():
    bad = []
    # B1：超上限
    probs, _notes = evaluate(3, 500.0, [('8. 判卷人', 400.0), ('0. 静态', 100.0)])
    if not any('B1' in p for p in probs):
        bad.append('B1 超上限没红：%r' % (probs,))
    if not any('最慢的一段' in p for p in probs):
        bad.append('B1 红了但没指出最慢的一段：%r' % (probs,))
    # B2：单段占 80%
    probs, _notes = evaluate(5, 100.0, [('8. 判卷人', 80.0), ('0. 静态', 20.0)])
    if not any('B2' in p for p in probs):
        bad.append('B2 单段独占没红：%r' % (probs,))
    # 正例 1：刚好在上限内
    cap = limit(3, scale=1.0)
    probs, _notes = evaluate(3, cap - 0.1,
                             [('8. 判卷人', (cap - 0.1) * 0.6), ('0. 静态', (cap - 0.1) * 0.4)],
                             scale=1.0)
    if probs:
        bad.append('上限内的总量被误判：%r' % (probs,))
    # 正例 2：未知档位不假装
    probs, notes = evaluate(2, 10_000.0, [('x', 10_000.0)])
    if probs or not notes:
        bad.append('未知档位应当只给说明：problems=%r notes=%r' % (probs, notes))
    # 正例 4：**只有一段**时 100%% 也不算独占（档 1 就是这种形状）
    probs, _notes = evaluate(1, 20.0, [('0. 静态', 20.0)], scale=1.0)
    if probs:
        bad.append('只有一段时被误判成独占：%r' % (probs,))
    # 正例 3：单段 60% 不算独占
    probs, _notes = evaluate(5, 100.0, [('8. 判卷人', 60.0), ('0. 静态', 40.0)])
    if probs:
        bad.append('单段 60%% 被误判成独占：%r' % (probs,))
    for b in bad:
        print('     ! ' + b)
    print('  [%s] 耗时判据自检：B1/B2 各红一次，三条正例不误伤'
          % ('PASS' if not bad else 'FAIL'))
    return 0 if not bad else 1


def main(argv):
    if '--selftest' in argv:
        return _selftest()
    print('  实测基线：%s' % ', '.join('档 %d = %.1f 秒' % (k, v)
                                       for k, v in sorted(MEASURED.items())))
    print('  上限 = 基线 x %.0f；可用 BY1_BUDGET_SCALE 临时放宽' % FACTOR)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
