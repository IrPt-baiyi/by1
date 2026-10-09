#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1fast -- **快速检查：30 秒内告诉你有没有改坏。**

    python by1fast.py

## 它和 `by1all --quick` 的分工

    by1fast          秒级    编译、import、解析、文档一致性
    by1all --quick   分钟级  66 项，含前向对拍、C 后端编译、神谕
    by1all           更久    含真权重 / 显卡

**快速检查不碰数值。** 它查的是"结构上还成立吗" ——
少了一个文件、多了一个没归类的文件、文档里的路径过期了、
某个模块语法错了、某份 `.by1` 解析不过了。

这类错误占了日常改动的绝大多数，而且**它们全都便宜**。
数值对不对是 `by1all` 的事，那是另一把尺子。

## 为什么不做抽样

抽样会带来一个坏性质：**同一个提交跑两次结果不一样**。
而这套检查的存在意义就是"改完立刻知道有没有坏" ——
一个会随机放过的检查，比没有检查更坏，因为它给你假的安心。

**要快就选"便宜但每次全做"的那几类，而不是"每次随机做一部分"。**
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
import json
import os
import py_compile
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# **ROOT 从 `by1paths` 拿，不从自己的位置推。**
# 这三处原来是 `ROOT = os.path.dirname(HERE)` —— 在 `src/checks/` 里
# 那个式子的答案是 `src/`，不是仓库根。而症状是"文件列表是空的"
# 或者"找不到 models/"，看起来像数据没了，不像路径算错了。
ROOT = by1paths.ROOT
sys.path.insert(0, HERE)

import by1paths                                # noqa: E402

# `selftest.by1` 里每一处都是**故意的错**，用来证明检查器不是空过。
# 所以它不是"要没有错误"，是"要正好这么多错误"。
SELFTEST = ('selftest.by1', 9, 4)

# 便宜的那几个判卷人。判据一律是**退出码** ——
# 不要在源码里 grep `[PASS]`：那儿的写法是 `'[%s]' % ('PASS' if ...)`，
# 字面量根本不在源码里。
#
# **`by1blind.py` 故意不在这一组。** 它要读完 34 份张量清单、37 万个
# 名字，实测 57 秒 —— 而其余五个加起来 1.6 秒。它是**分析工具**
# （量结构性盲区），不判通过失败，只报告看见了什么。
# 那种东西属于 `by1all`，不属于"30 秒内知道有没有改坏"。
FAST_JUDGES = ['by1docs.py', 'by1files.py', 'by1refs.py', 'by1lint.py',
               'by1pat.py']

# 一次子进程里 import 完全部模块 —— 55 次各起一个进程要 99 秒，
# 而其中大头是每个都重新 import 一次 torch。合成一次之后就几秒。
IMPORT_ALL = r'''
import importlib, sys, json
mods = json.loads(sys.argv[1])
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except BaseException as e:
        bad.append([m, type(e).__name__, str(e)[:60]])
print(json.dumps(bad))
'''


def main():
    t0 = time.time()
    bad = []
    print()
    print('=' * 74)
    print('  快速检查 —— 结构上还成立吗')
    print('=' * 74)

    # ── ① 每个模块都能编译 ──────────────────────────────────────
    print()
    print('  ① 编译 src/ 下每个 .py')
    # **走 `src/` 和每个子目录** —— 搬家之后模块不都在同一层了。
    # **从 `by1paths.SRC` 走，不是 `HERE`。**
    #
    # `HERE` 现在是自己那个目录（`src/checks/`）—— 而"所有模块住在
    # 哪些目录"这件事只有 `by1paths` 知道。写成 `HERE` 的后果是
    # **静默丢掉大部分模块**：编译数从 56 掉到 23，而判定行照旧 PASS。
    # 一个覆盖掉了一半还报绿的检查，比没有检查更坏。
    py_files = []
    for d in (by1paths.SRC,) + tuple(os.path.join(by1paths.SRC, s)
                                     for s in by1paths.SUBDIRS):
        if os.path.isdir(d):
            py_files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                         if f.endswith('.py')]
    py_files.sort()
    py_mods = [(p, os.path.basename(p)[:-3]) for p in py_files]
    n_ok = 0
    for p, _m in py_mods:
        try:
            py_compile.compile(p, doraise=True)
            n_ok += 1
        except py_compile.PyCompileError as e:
            bad.append('%s 编译不过：%s'
                       % (by1paths.rel(p), str(e).split('\n')[-1][:70]))
    print('     %d / %d 个模块编译通过' % (n_ok, len(py_files)))

    # ── ② 每个模块都能 import ───────────────────────────────────
    #
    # 编译只查语法。**import 才查得出"少了一个依赖"** ——
    # 而这个仓库真出过：打出来的包里少了 by1skip.py，
    # 8 个模块在模块级 import 它，gpu/run.sh 第一行就 ModuleNotFoundError。
    print()
    print('  ② import 每个模块（查得出"少了一个依赖"）')
    mods = [m for _p, m in py_mods
            if not m.startswith('_') and m != '__init__']
    # **`cwd` 要是 `src/`，不是自己那个目录。**
    #
    # 子进程跑的是 `import by1ir` 这种裸名 —— 它靠 `cwd` 在 `sys.path`
    # 上找。写成 `HERE`（= `src/checks/`）的话，顶层那 12 个模块
    # 一个都 import 不了，**而它们恰恰是被所有人 import 的底座**。
    r = subprocess.run([sys.executable, '-c', IMPORT_ALL, json.dumps(mods)],
                       cwd=by1paths.SRC, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=300)
    try:
        fails = json.loads((r.stdout or '').strip().split('\n')[-1])
    except Exception:
        fails = [['(整批)', 'Unknown', (r.stderr or '')[-70:]]]
    for m, kind, msg in fails:
        bad.append('%s import 不了：%s: %s' % (m, kind, msg))
    print('     %d / %d 个模块 import 成功' % (len(mods) - len(fails), len(mods)))

    # ── ③ 每份 .by1 都能解析 ────────────────────────────────────
    print()
    print('  ③ 解析 models/ 下每份 .by1')
    import by1check as bc
    models = sorted(f for f in os.listdir(by1paths.MODELS)
                    if f.endswith('.by1'))
    n_parse = 0
    for m in models:
        name, want_e, want_w = (SELFTEST if m == SELFTEST[0]
                                else (m, 0, None))
        try:
            rep, _info = bc.check(os.path.join(by1paths.MODELS, m))
            n_e = sum(1 for it in rep.items if it[0] == 'E')
            n_w = sum(1 for it in rep.items if it[0] == 'W')
            if m == SELFTEST[0]:
                # **对照**：这份文件必须**正好**这么多错。
                # 检查器哪天瞎了，这里就会变成 0。
                if (n_e, n_w) != (want_e, want_w):
                    bad.append('%s 期望 %d 错 %d 警，实际 %d 错 %d 警'
                               % (m, want_e, want_w, n_e, n_w))
                else:
                    n_parse += 1
            elif n_e:
                bad.append('%s 有 %d 个 E 级错误' % (m, n_e))
            else:
                n_parse += 1
        except Exception as e:
            bad.append('%s 解析崩了：%s: %s' % (m, type(e).__name__, str(e)[:50]))
    print('     %d / %d 份没问题（含 selftest 的"正好 %d 错"对照）'
          % (n_parse, len(models), SELFTEST[1]))

    # ── ④ 便宜的那几个判卷人 ────────────────────────────────────
    print()
    print('  ④ 文档 / 引用 / 静默写法')
    for script in FAST_JUDGES:
        # **路径走 `by1paths.tool`。** 裸名按自己的目录找是找不到的 ——
        # 这些判卷人散在 `src/` 和 `src/checks/` 两处，而 `by1refs.py`
        # 恰好是留在顶层的那 12 个底座之一。（症状是
        # "can't open file ... src/checks/by1refs.py"，
        # 看起来像那个文件丢了，不像"它在上一层"。）
        r = subprocess.run([sys.executable, by1paths.tool(script)], cwd=HERE,
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=300)
        out = (r.stdout or '') + (r.stderr or '')
        # **退出码 2 = 跳过，不是失败。** 这是这个仓库的三态协议
        # （`by1skip.py`），而这里原来只判了 `== 0` ——
        # 于是"这台机器验不了"被算成了"坏了"。
        if r.returncode == 2:
            tail = [l.strip() for l in out.splitlines() if l.strip()]
            print('     --   %-16s %s'
                  % (script, (tail[-1] if tail else '')[:56]))
        elif r.returncode == 0:
            tail = [l.strip() for l in out.splitlines() if l.strip()]
            print('     ok   %-16s %s' % (script, (tail[-1] if tail else '')[:56]))
        else:
            # **判据是退出码。** 不要去看有没有 `[PASS]` 字样 ——
            # 那只在运行时才拼出来，源码里 grep 不到。
            #
            # 报出来的时候要抓**具体那几条**，不要抓末尾那句总结 ——
            # 总结里写的是"这些不是风格问题"，对定位毫无用处。
            # 命中行长得像 `     by1run.py(1)` 或 `     x.py:12`。
            hits = [l.strip() for l in out.splitlines()
                    if re.search(r'\S+\.py[:(]\d*\)?', l.strip())]
            bad.append('%s 退出码 %d%s'
                       % (script, r.returncode,
                          '：' + '；'.join(hits[:3]) if hits else ''))

    # ── 判定 ────────────────────────────────────────────────────
    dt = time.time() - t0
    print()
    print('  ' + '-' * 72)
    if bad:
        print('  [FAIL] 快速检查：%d 处' % len(bad))
        for b in bad:
            print('     %s' % b)
    else:
        print('  [PASS] 快速检查：编译 %d · import %d · 解析 %d · 判卷人 %d'
              % (len(py_mods), len(mods), len(models), len(FAST_JUDGES)))
    print('         %.1f 秒' % dt)
    print()
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
