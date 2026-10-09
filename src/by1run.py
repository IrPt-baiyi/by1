#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1run -- **推送前的门，以及全量的轮换。**

    python by1run.py --push     跑快速检查 → 推 → 记账（**推之前用这个**）
    python by1run.py            只跑检查（按轮换决定跑哪个）
    python by1run.py --status   现在轮到谁
    python by1run.py --reset    重置（下一次是全量）
    python by1run.py --full     强制全量
    python by1run.py --quick    强制快速
    python by1run.py --deep     全量时连真权重 / 显卡一起
    python by1run.py --install-hook   装 git pre-push 钩子

## 节奏

**每一次推送之前跑一次快速检查**（`by1fast`，10 秒）。
**每第 5 次推送改成跑全量**（`by1all --quick`，分钟级）。

    推送1  快
    推送2  快
    推送3  快
    推送4  快
    推送5  **全量**   ← 然后重新数
    …

也就是"每五次推送里至少有一次真跑"。快速检查负责**立刻知道改坏了
没有**，全量负责**最终不会漏** —— 它碰得到数值，快速检查碰不到。

## 为什么要装钩子

"每次推送前跑"如果只靠人记得，那它迟早会变成"我记得的时候跑"。
`--install-hook` 往 `.git/hooks/pre-push` 写一个脚本，
**手动 `git push` 也拦得住** —— 钩子没过，推不出去。

钩子不进仓库（`.git/hooks/` 是本机的），所以这里存的是安装器。

## 状态存在哪

仓库根的 `.by1run-state`，一个整数：**这是第几次推送**（0..4）。
gitignore 的 —— 这是本机的节奏，不该跟着仓库走。
"""
import contextlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import by1io                                   # noqa: E402

STATE = os.path.join(ROOT, '.by1run-state')
HOOK = os.path.join(ROOT, '.git', 'hooks', 'pre-push')

# 每几次推送里安排一次全量
PUSHES_PER_FULL = 5


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


def check(forced=None, deep=False):
    """按轮换跑检查。返回退出码。**不动计数** —— 记账是 push 的事。"""
    n = read_state()
    if forced:
        which = forced
    else:
        # 第 PUSHES_PER_FULL 次推送（计数到顶）时跑全量，其余跑快速。
        which = 'full' if n >= PUSHES_PER_FULL - 1 else 'quick'
    print()
    print('=' * 74)
    if which == 'full':
        print('  全量（这是第 %d 次推送，每 %d 次一次全量）'
              % (n + 1, PUSHES_PER_FULL))
    else:
        print('  快速（第 %d 次推送，离全量还有 %d 次）'
              % (n + 1, PUSHES_PER_FULL - 1 - n))
    print('=' * 74)
    if which == 'full':
        return run('by1all.py', [] if deep else ['--quick'])
    return run('by1fast.py')


def push(args, deep=False):
    """**推送前的门。** 检查不过就不推。"""
    rc = check(deep=deep)
    if rc != 0:
        print()
        print('  **检查没过，不推。** 修完再来。')
        print()
        return rc

    n = read_state()
    print()
    print('  ── git push %s' % ' '.join(args))
    r = subprocess.run(['git', 'push'] + list(args), cwd=ROOT)
    if r.returncode == 0:
        # 只有推成功了才记账 —— 推失败不该消耗轮换。
        write_state(0 if n >= PUSHES_PER_FULL - 1 else n + 1)
        nxt = read_state()
        print()
        if nxt == 0:
            print('  推上去了。下一次是**全量**。')
        else:
            print('  推上去了。下一次是快速（离全量还有 %d 次）'
                  % (PUSHES_PER_FULL - 1 - nxt))
    else:
        print()
        print('  **推送失败**（网络？）—— 不记账，重推即可。')
    print()
    return r.returncode


HOOK_BODY = '''#!/bin/sh
# 由 `python src/by1run.py --install-hook` 装。
#
# **推送前的门。** 检查不过就推不出去 —— 这样"每次推送前跑一次"
# 不依赖"我记得"。
#
# 想临时绕过：git push --no-verify
cd "$(git rev-parse --show-toplevel)" || exit 1

# 装的时候把解释器的绝对路径记在这里。**不能直接写 `python`** ——
# 这个仓库的开发机上 `python` 常常不在 PATH 里（用的是绝对路径调用的），
# 而钩子失败时报的是"找不到 python"，看起来像钩子坏了，
# 不像"环境没配好"。
PY="{py}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "  !! 找不到 python，推送前的检查没跑成"
  exit 1
fi

exec "$PY" src/by1run.py --hook-check
'''


def install_hook():
    d = os.path.dirname(HOOK)
    if not os.path.isdir(d):
        print('  找不到 .git/hooks/ —— 这不是一个 git 仓库？')
        return 1
    by1io.write_text(HOOK, HOOK_BODY.replace('{py}', sys.executable))
    # Windows 上 chmod 基本是空操作，失败不影响 git 执行 ——
    # 用 `suppress` 而不是 `try/except/pass`：后者读起来像"没想好"。
    with contextlib.suppress(OSError):
        os.chmod(HOOK, 0o755)
    print('  装了 %s' % os.path.relpath(HOOK, ROOT))
    print('  解释器记的是：%s' % sys.executable)
    print('  之后每次 `git push`（含手动）都会先跑快速检查。')
    print('  临时绕过：`git push --no-verify`')
    return 0


def main():
    argv = sys.argv[1:]

    if '--install-hook' in argv:
        return install_hook()

    if '--status' in argv:
        n = read_state()
        print()
        print('  这是第 %d 次推送；下一次：%s'
              % (n + 1,
                 '**全量**' if n >= PUSHES_PER_FULL - 1 else
                 '快速（离全量还有 %d 次）' % (PUSHES_PER_FULL - 1 - n)))
        print('  钩子：%s' % ('装了' if os.path.exists(HOOK) else '**没装**'))
        print()
        return 0

    if '--reset' in argv:
        write_state(0)
        print()
        print('  重置了：下一次是快速（离全量还有 %d 次）' % (PUSHES_PER_FULL - 1))
        print()
        return 0

    if '--hook-check' in argv:
        # 钩子调的就是这个：只跑快速，不推、不记账。
        # **钩子里不跑全量** —— 一次 push 卡几分钟没人受得了。
        # 全量由 `--push` 那条路负责。
        return run('by1fast.py')

    if '--push' in argv:
        rest = [a for a in argv if a not in ('--push', '--deep')]
        return push(rest or ['origin', 'main'], deep='--deep' in argv)

    forced = ('full' if '--full' in argv or '--deep' in argv else
              'quick' if '--quick' in argv else None)
    return check(forced=forced, deep='--deep' in argv)


if __name__ == '__main__':
    sys.exit(main())
