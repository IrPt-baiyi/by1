#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1run -- **按轮换跑检查：一次全量，然后四次快速。**

    python by1run.py            按轮换决定跑哪个
    python by1run.py --status   现在轮到谁了
    python by1run.py --reset    重置（下一次是全量）
    python by1run.py --full     强制全量
    python by1run.py --quick    强制快速
    python by1run.py --deep     全量时连真权重 / 显卡那部分一起跑

## 为什么要轮换，而不是每次都全量

全量（`by1all --quick`）是分钟级的，改一行字也跑一遍不现实；
而纯快速（`by1fast`，10 秒）碰不到数值 —— 它能告诉你"结构没坏"，
不能告诉你"算得还对"。

**两个都要，但不必每次都要。** 轮换的账是这样：

    全量 → 快 · 快 · 快 · 快 → 全量 → …

也就是"每五次里至少有一次真跑"。快速检查负责**立刻知道改坏了没有**，
全量负责**最终不会漏**。

## 状态存在哪

仓库根目录的 `.by1run-state`，一个整数：**还剩几次快速就该全量了**。
它是 gitignore 的 —— 这是本机的节奏，不该跟着仓库走。

没有这个文件时视为 0，也就是**第一次跑的是全量**。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import by1io                                   # noqa: E402

STATE = os.path.join(ROOT, '.by1run-state')

# 全量之后安排几次快速
QUICKS_PER_FULL = 4


def read_state():
    try:
        return max(0, int(by1io.read_text(STATE).strip()))
    except (OSError, ValueError):
        return 0


def write_state(n):
    by1io.write_text(STATE, '%d\n' % n)


def run(script, args=()):
    print()
    print('  ── 跑 %s %s' % (script, ' '.join(args)))
    return subprocess.run([sys.executable, script] + list(args), cwd=HERE).returncode


def main():
    argv = sys.argv[1:]
    n = read_state()

    if '--status' in argv:
        print()
        if n == 0:
            print('  下一次：**全量**（之后安排 %d 次快速）' % QUICKS_PER_FULL)
        else:
            print('  下一次：**快速**（还剩 %d 次，之后全量）' % n)
        print()
        return 0

    if '--reset' in argv:
        write_state(0)
        print()
        print('  重置了：下一次是全量')
        print()
        return 0

    force_full = '--full' in argv or '--deep' in argv
    force_quick = '--quick' in argv
    forced = force_full or force_quick

    if force_full:
        which = 'full'
    elif force_quick:
        which = 'quick'
    else:
        which = 'full' if n == 0 else 'quick'

    print()
    print('=' * 74)
    if forced:
        # **强制跑不动轮换。** "强制"的意思就是"这一次按我说的来，
        # 别记账" —— 否则 `--quick` 会把计数吃掉，跑着跑着就
        # 莫名其妙轮到全量，而人不知道自己什么时候消耗的。
        print('  %s（强制 —— **不动轮换**，还剩 %d 次快速）'
              % ('全量' if which == 'full' else '快速', n))
    elif which == 'full':
        print('  全量（轮换：之后安排 %d 次快速）' % QUICKS_PER_FULL)
    else:
        print('  快速（轮换：还剩 %d 次，之后全量）' % n)
    print('=' * 74)

    if which == 'full':
        args = [] if '--deep' in argv else ['--quick']
        rc = run('by1all.py', args)
        # 全量跑完就重置计数。**失败也重置** —— 因为失败本来就该
        # 让人多跑几次全量，而不是卡在"每次都全量"。
        if not forced:
            write_state(QUICKS_PER_FULL)
            print()
            print('  下一次是快速（还剩 %d 次）' % QUICKS_PER_FULL)
    else:
        rc = run('by1fast.py')
        if not forced:
            write_state(max(0, n - 1))
            left = max(0, n - 1)
            print()
            print('  下一次：%s'
                  % ('快速（还剩 %d 次）' % left if left else '**全量**'))

    print()
    print('  [%s] by1run —— %s%s'
          % ('PASS' if rc == 0 else 'FAIL',
             '全量' if which == 'full' else '快速',
             '（强制）' if forced else ''))
    print()
    return rc


if __name__ == '__main__':
    sys.exit(main())
