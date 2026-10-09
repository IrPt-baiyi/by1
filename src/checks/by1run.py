#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1run -- **推送前的门，以及逐次加深的档位。**

    python by1run.py --push          门（档 1）+ 加深 + 推 + 记账
    python by1run.py --tier 3        只跑档 3
    python by1run.py --status        现在轮到哪一档
    python by1run.py --reset         重新数
    python by1run.py --install-hook  装 pre-push 钩子

## 六档：**每加一档，多要一样东西**

    档1  10.7 秒   静态        什么都不用       编译 · import · 解析 · 文档
    档2  16.7 秒   结构        + numpy          检查器 · 模式 · lint · 取值门 · 结构判卷人
    档3  18.6 秒   契约        + refs/ 产物      config 逐字段 · 张量名与形状
    档4  50.0 秒   数值        + torch           神谕 · 三后端 · 数值判卷人
    档5  68.3 秒   环境        + gcc / 下载      前向对拍 · C 后端 · 逃生舱第二层 · 端到端
    档6  69.6 秒   真机        + 显卡 / 真权重    （这台机器上和档 5 逐项相同）

（时间是这台机器实测的，不是估的。**档 4 那个大跳**（18.6 → 50.0）
就是那 6 个要 torch 前向的判卷人。）

## 轮换只走到档 4 —— **档 5/6 要手动**

    LADDER = [2, 3, 4]

档 5 要 gcc、要下载，档 6 要显卡 —— **它们不是"每次都该跑"的东西**，
而是"这次想跑就跑"。自动轮换把它们排进去的话，一次推送会从 17 秒
变成 68 秒，而多跑到的那些**并不对应你刚改的东西**。

要全量就明说：

    python src/by1run.py --tier 5          只跑档 5
    python src/by1run.py --push --tier 5   推之前先跑档 5

**档位是默认值，不是上限。** 改了大东西、或者只是心里没底，
手动要一次更高的档永远可以。

## 推送时的节奏

**门永远是档 1** —— 10 秒，每次都跑，不过就不推。
**加深的那部分逐次往上走**：

    推送1   门 + 档2
    推送2   门 + 档3
    推送3   门 + 档4      ← 正常用到这里
    推送4   门 + 档5（全量）
    → 重新数

也就是"每四次推送里至少有一次全量"。门负责**立刻知道改坏了没有**，
加深负责**最终不会漏** —— 它碰得到数值，门碰不到。

## 为什么要装钩子

"每次推送前跑"如果只靠人记得，迟早会变成"我记得的时候跑"。
`--install-hook` 往 `.git/hooks/pre-push` 写脚本，
**手动 `git push` 也拦得住** —— 门没过，推不出去。

钩子不进仓库（`.git/hooks/` 是本机的），所以这里存的是安装器。
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# **ROOT 从 `by1paths` 拿，不从自己的位置推。**
# 这三处原来是 `ROOT = os.path.dirname(HERE)` —— 在 `src/checks/` 里
# 那个式子的答案是 `src/`，不是仓库根。而症状是"文件列表是空的"
# 或者"找不到 models/"，看起来像数据没了，不像路径算错了。
ROOT = by1paths.ROOT
sys.path.insert(0, HERE)

import by1io                                   # noqa: E402
import by1paths                                # noqa: E402

STATE = os.path.join(ROOT, '.by1run-state')
HOOK = os.path.join(ROOT, '.git', 'hooks', 'pre-push')

# 门：每一次推送都跑这一档
GATE_TIER = 1
# 加深：逐次往上走的那几档，走完一轮回到第一个。
#
# **只到档 4。** 档 5 要 gcc、要下载，档 6 要显卡 —— 它们不是
# "每次都该跑"的东西。自动轮换把它们排进去的话，一次推送会从
# 17 秒变成 68 秒，而多跑到的那些**并不对应你刚改的东西**。
# 要全量手动要：--tier 5 / --push --tier 5。
LADDER = [2, 3, 4]

TIER_NAME = {
    1: '静态', 2: '结构', 3: '契约', 4: '数值', 5: '环境', 6: '真机',
}
TIER_SECS = {1: 10.7, 2: 16.7, 3: 18.6, 4: 50.0, 5: 68.3, 6: 69.6}


def read_state():
    """还差几步走完这一轮。0 = 下一次是 LADDER[0]。"""
    try:
        return max(0, min(len(LADDER) - 1,
                          int(by1io.read_text(STATE).strip())))
    except (OSError, ValueError):
        return 0


def write_state(n):
    by1io.write_text(STATE, '%d\n' % n)


def run(script, args=()):
    print()
    print('  ── 跑 %s %s' % (os.path.basename(script), ' '.join(args)))
    return subprocess.run([sys.executable, script] + list(args), cwd=HERE).returncode


def run_tier(n):
    """跑某一档。档 1 只是 `by1all --tier 1`（它内部调 `by1fast`），
    所以这里没有第二条路 —— 同一个意思不写两遍。"""
    return run('by1all.py', ['--tier', str(n)])


def show(n):
    tier = LADDER[n]
    print('  这一次加深到 **档 %d（%s）**，约 %.0f 秒'
          % (tier, TIER_NAME[tier], TIER_SECS[tier]))
    print('  门是档 %d（%s），每次都跑，约 %.0f 秒'
          % (GATE_TIER, TIER_NAME[GATE_TIER], TIER_SECS[GATE_TIER]))


def check(tier=None):
    """按轮换跑。`tier` 给定时只跑那一档（不跑门 —— 手动指名就是要那一档）。"""
    n = read_state()
    if tier is not None:
        print()
        print('=' * 74)
        print('  手动指定：档 %d（%s）' % (tier, TIER_NAME.get(tier, '?')))
        print('=' * 74)
        return run_tier(tier)
    print()
    print('=' * 74)
    show(n)
    print('=' * 74)
    # **门先跑。** 它便宜，而且它失败时没必要再花 50 秒。
    rc = run_tier(GATE_TIER)
    if rc != 0:
        return rc
    deep = LADDER[n]
    if deep <= GATE_TIER:
        return 0
    return run_tier(deep)


def push(args, tier=None):
    """**推送前的门。** 检查不过就不推。"""
    n = read_state()
    rc = check(tier=tier)
    if rc != 0:
        print()
        print('  **检查没过，不推。** 修完再来。')
        print()
        return rc

    print()
    # **`--no-verify`：钩子会再跑一遍同一个门。**
    #
    # 上面刚跑过档 1，而裸 `git push` 会触发 pre-push 钩子 ——
    # 不挡住的话门跑两次，10 秒变 20 秒（第一次实测就是这样）。
    #
    # 钩子仍然有用：**手动 `git push` 走的正是钩子那条路**，
    # 它管的就是"忘了用 `--push` 的时候"。
    print('  ── git push --no-verify %s' % ' '.join(args))
    r = subprocess.run(['git', 'push', '--no-verify'] + list(args), cwd=ROOT)
    if r.returncode == 0:
        # 只有推成功了才记账 —— 推失败不该消耗轮换。
        # **手动指定档位也不记账**：那是一次性的，不改变节奏。
        if tier is None:
            write_state(0 if n >= len(LADDER) - 1 else n + 1)
        nxt = LADDER[read_state()]
        print()
        print('  推上去了。下一次加深到档 %d（%s）。'
              % (nxt, TIER_NAME[nxt]))
    else:
        print()
        print('  **推送失败**（网络？）—— 不记账，重推即可。')
    print()
    return r.returncode


HOOK_BODY = '''#!/bin/sh
# 由 `python src/by1run.py --install-hook` 装。
#
# **推送前的门。** 门不过就推不出去 —— 这样"每次推送前跑档 1"
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
  echo "  !! 找不到 python，推送前的门没跑成"
  exit 1
fi

exec "$PY" src/by1all.py --tier 1
'''


def install_hook():
    d = os.path.dirname(HOOK)
    if not os.path.isdir(d):
        print('  找不到 .git/hooks/ —— 这不是一个 git 仓库？')
        return 1
    by1io.write_text(HOOK, HOOK_BODY.replace('{py}', sys.executable))
    print('  装了 %s' % os.path.relpath(HOOK, ROOT))
    print('  解释器记的是：%s' % sys.executable)
    print('  之后每次 `git push`（含手动）都会先跑档 %d。' % GATE_TIER)
    print('  临时绕过：`git push --no-verify`')
    return 0


def main():
    argv = sys.argv[1:]

    if '--install-hook' in argv:
        return install_hook()

    if '--status' in argv:
        n = read_state()
        print()
        print('  门：档 %d（%s，%.0f 秒，每次推送都跑）'
              % (GATE_TIER, TIER_NAME[GATE_TIER], TIER_SECS[GATE_TIER]))
        print('  加深：下一次是档 %d（%s）；这一轮还剩 %d 次'
              % (LADDER[n], TIER_NAME[LADDER[n]], len(LADDER) - n))
        print('  梯队：%s' % ' → '.join('档%d' % x for x in LADDER))
        print('  钩子：%s' % ('装了' if os.path.exists(HOOK) else '**没装**'))
        print()
        return 0

    if '--reset' in argv:
        write_state(0)
        print()
        print('  重新数了：下一次加深到档 %d' % LADDER[0])
        print()
        return 0

    tier = None
    if '--tier' in argv:
        tier = max(1, min(6, int(argv[argv.index('--tier') + 1])))

    if '--push' in argv:
        FLAGS = ('--push', '--tier', str(tier) if tier else '--tier')
        rest = [a for a in argv
                if a not in FLAGS and not a.startswith('--')]
        return push(rest or ['origin', 'main'], tier=tier)

    return check(tier=tier)


if __name__ == '__main__':
    sys.exit(main())
