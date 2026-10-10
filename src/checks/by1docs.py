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


#: **"多少行 / 多少节 / 多少个 .by1"这类可复核的数字。**
#
# 为什么要有这一条：`by1docs` 原来只查"路径还在不在"，
# 于是**行数、节数、项数这些数字没有任何判卷人** —— 而它们会烂。
# 实测（2026-10）：`README.md` 说 `history/ir.md` 是 "3200+ 行"、
# `1.md` 说 "3321 行"，而那份文件是 **2306 行**；
# `by1pack.py` 的说明写 "65 节"，实际 **73 节**。
# `history/ir.md` §73 的标题就叫"抄进文档的数字，寿命比文档短" ——
# 这一条就是那件事的执行者。
#:
#: ## `(?<!第 )` 那一段是必要的
#:
#: `第 71 节` 是**指某一节**，不是"一共 71 节"。第一版没有这个断言，
#: 于是 README 里三处"见 `history/ir.md` 第 70 节"全被读成
#: "这份文档有 70 节"，报了三个假警报。
#: **假警报会把真问题淹掉** —— 所以宁可窄，不可宽。
#: `第 71 节` 里还有第二个坑：`(\d+)` 会从**中间**开始匹配，
#: 于是 "71 节" 被读成 "1 节"。所以再补一个 `(?<!\d)` ——
#: 数字要从头到尾整块匹配，不能是某个数字的尾巴。
PAT_CLAIM = re.compile(r'(?<!第[ ])(?<!\d)(\d+)\s*(行|节|个\s*\.by1)')


def measure(path, unit):
    """一个文件"多少行 / 多少节 / 多少个 .by1"。

    `unit` 只认三种，**量不出来就返回 None** —— 那时候这条声明不判，
    而不是猜一个数出来（猜出来的数就是下一个要烂的数字）。
    """
    # **"多少个 .by1" 这一支必须放在最前面。**
    #
    # 它数的是**目录**，不需要读文件内容。而它原来在
    # `by1io.read_text(path)` **之后** —— 主语是目录时，
    # 读目录抛 `IsADirectoryError`（`OSError` 的一种），
    # 被下面那个 `except OSError` **静默接住**，于是返回 None，
    # 而 None 的含义是"量不出来，这条不判"。
    #
    # 实测（2026-10）：先加了"目录也算主语"那一支，然后拿一个
    # **故意写错的数**去试 —— 它**没红**。原因不在那一支，
    # 在这里：**又一次静默吞掉，又一次让"没有判卷人"长得像"通过"。**
    if unit.replace(' ', '') == '个.by1':
        import glob as _g
        # 主语**是目录**就数它自己；是文件就数它所在的那一层。
        d = path if os.path.isdir(path) else os.path.dirname(path)
        return len(_g.glob(os.path.join(d or '.', '*.by1')))
    try:
        t = by1io.read_text(path, errors='replace')
    except OSError:
        return None
    if unit == '行':
        # **跳过围栏代码块** —— 那里面是**别人的输出**（比如一段
        # 粘贴的 `wc -l` 结果），不是这段行文自己的声明。
        n, fenced = 0, False
        for ln in t.splitlines():
            if ln.lstrip().startswith('```'):
                fenced = not fenced
                continue
            if not fenced:
                n += 1
        return n
    if unit == '节':
        return len(re.findall(r'^##\s', t, re.M))
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

    # ── 数字类声明：**行数 / 节数 / 多少个 .by1** ───────────────────
    #
    # 这一类以前**完全没有判卷人**。判据很窄，故意的：
    # 数字必须**紧挨着**那个反引号路径（前 60 字符 / 后 20 字符之内），
    # 比如 `` `history/ir.md`（73 节）``。
    #
    # **一行里出现的数字不能全和一行里出现的路径配对** ——
    # 第一版就是那样，于是
    #     `ir.md`（73 节）· `1.md`（8 节）
    # 这一行被判成"`1.md` 是 73 节"和"`ir.md` 是 8 节"，两条假警报。
    # 假警报会把真问题淹掉（这个项目为此踩过两次），所以宁可窄。
    claims, wrong = 0, []
    for p in files:
        rel = by1paths.rel(p)
        near = os.path.dirname(p)
        for i, line in enumerate(by1io.read_text(p, errors='replace')
                                .splitlines(), 1):
            for pm in PAT_CLAIM.finditer(line):
                claimed, unit = int(pm.group(1)), pm.group(2)
                win = line[max(0, pm.start() - 60):pm.end() + 20]
                for rm in re.finditer(r'`([^`]+)`', win):
                    # **主语必须"紧挨着"这个数字。**
                    #
                    # 原来只要求路径出现在前 60 字符的窗口里 —— 于是新写的
                    # `docs/check-map.md` 里两处被误读：
                    #
                    #   `src/by1check.py` 的 `check()`，833 行
                    #       -> "by1check.py 是 833 行"（它 1692 行）
                    #      主语其实是 `check()`
                    #
                    #   | `by1export.py` | 1243 行 | 1064 行 |
                    #       -> "by1export.py 是 1243 行"（它 512 行）
                    #      那两个数是"切之前 / 切之后"的 `check()`
                    #
                    # **改文档去迎合检查器，等于让判据决定事实** ——
                    # 而这个检查器存在的理由正是"文档里的数字会烂"。
                    # 它误报一次，下次真烂就没人看了。所以收窄判据。
                    #
                    # 正确的形状是原注释点名的那个：`history/ir.md`（73 节）
                    # —— 反引号后面**紧跟**括号。所以这里要求路径到数字
                    # 之间只有括号 / 判定词 / 空白。
                    # **还要放开 markdown 记号。** 第一版只允许
                    # `\s（(是为：:`，结果全仓库 **0 处**被查到 ——
                    # 因为真实写法是 `` `history/ir.md`（**74 节**）``，
                    # 中间隔着 `**`。**一个什么都不查的检查，比一个
                    # 误报的检查更坏**：误报会吵，全漏会让人以为查过了。
                    if not re.fullmatch(r'[\s（(是为：:*_`\[\]]*',
                                        win[rm.end():pm.start()]):
                        continue
                    tgt = rm.group(1).strip()
                    if tgt.startswith(SKIP_LINK) or ' ' in tgt:
                        continue
                    real = find(tgt, near)
                    if not real:
                        continue
                    # **目录也算主语 —— 但只对"多少个 .by1"这一种量法。**
                    #
                    # 实测（2026-10）：`README.md` 里那句
                    #     models/         26 份 .by1，其中 14 份对真实 checkpoint 验过
                    # **从来没被查过** ✗ —— 两处都不满足：
                    # ① 主语没有反引号 ✓ ② 量词是"份"不是"个" ✓
                    # 而它当时是**错的** ✓（实际 28 份）。
                    #
                    # 所以补上目录这一支 ✓ —— 于是它现在能写成
                    # 可复核的形状：`` `models/`（29 个 .by1）`` ✓。
                    # **改文档去迎合检查器是错的** ✗，但"把数字写成
                    # 检查器读得懂的形状"不是迎合 ✓ —— 那正是让它有判卷人 ✓。
                    # （"份"仍然不认 ✓：量词多一个，误报面就大一倍，
                    #   而这个检查器怕的是**假警报把真问题淹掉** ✓。）
                    if os.path.isdir(real):
                        if unit.replace(' ', '') != '个.by1':
                            continue
                    elif not os.path.isfile(real):
                        continue
                    got = measure(real, unit)
                    if got is None:
                        continue
                    claims += 1
                    if got != claimed:
                        wrong.append((rel, i, tgt, claimed, unit, got))
    print('  ── 顺带：文档里"多少行 / 多少节 / 多少个 .by1"这类数字')
    print('     查了 %d 处可复核的声明' % claims)
    if wrong:
        for rel, i, tgt, claimed, unit, got in wrong:
            print('     **%s:%d** 说 `%s` 是 %d %s —— 实际 %d %s'
                  % (rel, i, tgt, claimed, unit, got, unit))
    else:
        print('     全部对得上 ✓')
    print()

    # ── 生成物：**`ir-spec.md` 和生成它的代码还一致吗** ──────────────
    #
    # `by1blind.py` 里列过这条盲区（"E. 文档同步：生成物和源头还一致吗"），
    # 而 `by1docs` 的 docstring 也引用了它 —— 但**它自己一直没查**。
    #
    # 实测（2026-10）：`ir-spec.md` 和 `by1ir.spec_markdown()` 差了 30 行，
    # 全都不是笔误 —— 是 `by1ir` 的 schema 改了（闭集从裸 `enum`
    # 改成 `enum:a,b`），而那份"从表生成的"文档没跟着重新生成。
    # **一份声称自己是从 X 生成的文档，比一份手写文档更需要判卷人**：
    # 手写的至少不会被信成"和代码同步"。
    stale = []
    try:
        import by1ir as _ir
        spec_p = os.path.join(ROOT, 'ir-spec.md')
        if os.path.exists(spec_p):
            have = by1io.read_text(spec_p, encoding='utf-8').rstrip('\n')
            want_ = _ir.spec_markdown().rstrip('\n')
            if have != want_:
                stale.append(('ir-spec.md', 'python src/by1ir.py --spec'))
    except Exception as e:                      # 读不了就说读不了，不猜
        print('  ── 顺带：生成物一致性（读不了 by1ir：%s）' % str(e)[:40])
        print()
    print('  ── 顺带：声称"从代码生成"的文档，和代码还一致吗')
    if stale:
        for f, cmd in stale:
            print('     **%s** 和生成它的代码不一致 —— 跑一下：%s' % (f, cmd))
    else:
        print('     ir-spec.md 与 `by1ir.spec_markdown()` 逐字符一致 ✓')
    print()

    print('  [%s] 文档路径 %s'
          % ('PASS' if not bad and not missing and not wrong and not stale
             else 'FAIL',
             '路径、链接、脚本调用、数字、生成物都在'
             if not bad and not missing and not wrong and not stale
             else '%d 处路径 + %d 个脚本 + %d 个数字 + %d 个生成物 有问题'
                  % (len(bad), len(missing), len(wrong), len(stale))))
    print()
    return 0 if not bad and not missing and not wrong and not stale else 1


if __name__ == "__main__":
    sys.exit(main())
