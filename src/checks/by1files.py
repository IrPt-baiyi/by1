#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1files -- **生成并核对 `FILES.md`。**

    python by1files.py            # 看现在一致不一致
    python by1files.py --write    # 重新生成
    python by1files.py --check    # 不一致就非零退出（给 by1all 用）

## 为什么要有这个

`FILES.md` 是手写的，所以它**一定会过期** —— 加一个文件、挪一个目录、
改一个模块的用途，那份文档都不会自己知道。而这个仓库里已经有过一次
同类事故：`README` 说 `ir.md` 是 3200+ 行（实际 2306）。

## 分工：谁生成，谁手写

    **从目录树生成**：有哪些文件、在哪个组、一句话说明（取 docstring）、
                      每组的个数、总数
    **留在本文件里**：组名、每组的引言、开头的"先读哪三份"、结尾的两条约定

所以加一个文件不需要改文档，但**要在这个文件里给它归组** ——
没归组的会被报成"未归类"，这是故意的：逼着人想一下它属于哪。

## 一句话说明从哪来

优先取该文件 docstring 里第一行带 ` -- ` 的（这个仓库的写法是
`by1xxx -- **一句话**`）。取不到就用 `OVERRIDE` 里手写的。
两个都没有，就报出来 —— **宁可报错，不要编一句像样的废话。**
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
import subprocess
import sys

import by1io

HERE = os.path.dirname(os.path.abspath(__file__))
# **ROOT 从 `by1paths` 拿，不从自己的位置推。**
# 这三处原来是 `ROOT = os.path.dirname(HERE)` —— 在 `src/checks/` 里
# 那个式子的答案是 `src/`，不是仓库根。而症状是"文件列表是空的"
# 或者"找不到 models/"，看起来像数据没了，不像路径算错了。
ROOT = by1paths.ROOT
OUT = os.path.join(ROOT, 'FILES.md')

# ════════════════════════════════════════════════════════════════════
# 组 —— 顺序就是文档里的顺序。**没列进来的文件会被报成"未归类"。**
# ════════════════════════════════════════════════════════════════════
GROUPS = [
    ('底座（留在 src/ 顶层）', 'src', '''**被别的模块 import 的那 12 个。**

它们必须在 `src/` 顶层，因为 `by1paths.py` 自己也在这里 ——
它是引导，搬它就得先有引导，而引导又得先找得到它。

| 被谁 import |
|---|
| `by1io` 39 个模块 · `by1paths` 33 个 · `by1codegen` 17 个 · `by1skip` 11 个 |
| `by1ir` 8 个 · `by1ver` 6 个 · `by1exec` 5 个 · `by1c` 4 个 |
| `by1refs` 3 个 · `by1check` 2 个 · `by1boot` 2 个 · `by1ext` 2 个 |

**其余 44 个是叶子**（没人 import 它们），所以能进子目录。

每个入口脚本开头那几行引导是**生成的**（判据在 `by1paths.check_boot`）——
重复的东西必须能被检查，否则新加一个脚本忘了加，症状是
`No module named 'by1paths'`，看起来像"文件丢了"。''', [
        'by1paths', 'by1io', 'by1codegen', 'by1skip', 'by1ir', 'by1ver',
        'by1exec', 'by1c', 'by1refs', 'by1check', 'by1boot', 'by1ext']),

    ('`lang/` —— 语言本身', 'lang', '''把 `.by1` 变成一份能算的东西。

`by1export` 是**导出策略层** —— 把算好的语义变成 HF config 的字段。
它原来是 `by1check.check()` 里嵌着的 13 个 `gen_*` / `name_*`
（那个函数因此有 1243 行）。**劈开的理由写在那个文件的头注里。**''', [
        'by1', 'by1contract', 'by1emit', 'by1name', 'by1vocab',
        'by1export']),

    ('`checks/` —— 判卷人 · 神谕 · 跑检查', 'checks', '''这个项目最核心的资产。

判卷人回答"**我怎么知道它是对的**"；神谕回答"**是不是一起错了**"——
三个后端互拍只能证明自洽，而它们可以一致地错（见 `history/ir.md`）。

跑检查的两个入口也在这里（`by1run` · `by1fast`）。''', [
        'by1all', 'by1verify', 'by1diff', 'by1opdiff', 'by1e2e',
        'by1irentry', 'by1gate', 'by1docs', 'by1files', 'by1lint',
        'by1blind', 'by1debt', 'by1pat', 'by1smoke', 'by1pack',
        'by1rebuild', 'by1bootir', 'by1oracles', 'by1oracle',
        'by1run', 'by1fast', 'by1gpu']),

    ('`modelcheck/` —— 一个模型一个判卷人', 'modelcheck', '''每个都用自己的官方产物当真相源。''', [
        'by1gpt2', 'by1instella', 'by1mla', 'by1moe', 'by1rope',
        'by1real', 'by1load', 'by1mem', 'by1train', 'by1dev']),

    ('`data/` —— 抓数据 / 推模型', 'data', '''从一个已发布的 checkpoint 反推描述，以及把 `refs/` 抓回来。''', [
        'by1fetch', 'by1index', 'by1boot', 'by1triage', 'by1cmp',
        'by1cloud']),

    ('`escape/` —— 逃生舱', 'escape', '''语言表达不了的东西有两个出口，**两个都不是"绕过检查"**：
张量契约、三后端对拍、取值门照旧生效。''', [
        'by1raw', 'by1extdemo', 'ext-demo.c']),
]

# 各目录的引言
DIR_INTRO = {
    'models': '''命名规矩：**文件名 = 长名 = HF 的模型名（去掉 owner）**，
`model` 声明 = 短名（手写的）。两列的对照见 `models.tsv`。''',
    'refs': '''从 HuggingFace / ModelScope 抓的**模型元数据**（配置 + 张量名/形状清单），
**不是权重**。版权属各发布方，各自适用各自的许可证。''',
    'docs': '''活的文档 —— 还没做完的事记在这里，做完的挪去 `history/`。''',
    'history': '''以前的东西。留着的理由是：能看出当时是怎么想的。''',
    'drafts': '''还没进 `models/` 的草稿。**没有判卷人，所以不算数。**''',
    'gpu': '''只有显卡上能跑的那部分。''',
    'tools': '''一次性脚本。**没有任何代码引用它们。**''',
    'src': '''代码。''',
}

# ════════════════════════════════════════════════════════════════════
# 手写的一句话 —— 只在 docstring 取不到时用。
# **不要为了让文档好看而往这里塞。** 取不到就报出来。
# ════════════════════════════════════════════════════════════════════
OVERRIDE = {
    'ext-demo.c': '逃生舱第二层的示例：两个语言里没有的机制。`by1extdemo` 编它',
    '1.md': '立项宣言：这门语言要解决什么、不解决什么',
    'FILES.md': '**这份文档本身**，由 `src/by1files.py` 生成',
    'LICENSE': 'MIT。里面另有一节说明 `refs/` 不是本项目的作品',
    'VERSION': '版本号。**由 `by1ver.py --write` 生成**，不要手改',
    'pyproject.toml': '打包元数据。`py-modules` + `data-files`（没有 package）',
    '.gitattributes': '行尾一律 LF。没有它的后果见文件里的注释',
    '.gitignore': '产物、权重、缓存、临时探针',
    'models.tsv': '**手写的模型清单**：长名 / 短名 / HF id / 说明。命名规矩写在文件头',
    'ir-spec.md': '**生成物**：IR 规格（字段、必填、语义、闭集）。`by1ir.py --spec`',
    'models.md': '**生成物**：所有模型的架构对比表。`by1cmp.py --md`',
    'README.md': '门面。是什么、怎么跑、数据从哪来',
    'SOURCES.tsv': '**没有对应 `.by1` 的那些产物的出处** —— 抓了但还没描述的模型记在这里',
    'raw.py': '**不是模型，是逃生舱的工厂函数实现**。必须和 `raw-escape.by1` '
              '挨着（`by1codegen` 按名字找它）',
    'run.sh': '入口',
    'requirements.txt': '依赖（和本地不同：要 CUDA 版的 torch）',
    'by1kda.py': 'KDA（Kimi Delta Attention）的验证。它在 `by1check` 里是独立种类，不是 GDN',
    'rebuild_manifest.py': '复现脚本',
    'INVENTORY.txt': '验证清单：每个模型验到了什么、判卷人是谁',
    'pattern-design.md': '设计草案 v0，**状态：待审**。它挡着 4 个模型',
    'pattern-design-review.md': '对上面那份草案的审查 —— **照做之前先读它**',
}

HEAD = '''\
# 这个仓库里每个文件是干什么的

> {n} 个跟踪文件。按"你会想找什么"分组，不按字母表。
>
> **这份文档是生成的**：`python src/by1files.py --write`。
> 文件清单、每组的个数、每个文件的一句话都从目录树和 docstring 来 ——
> 所以它不会过期。要改措辞，改 `src/by1files.py` 里的 `GROUPS`。

## 先读哪三份

| 想干什么 | 读哪份 |
|---|---|
| **它到底是什么东西** | `README.md` |
| **当初为什么这么做、错过什么** | `history/ir.md` |
| **每个文件干嘛的** | 就是这份 |

`1.md` 是立项时的宣言，写的是**要做什么**。`history/ir.md` 写的是
**做的时候撞上了什么** —— 后者比前者值钱，因为成功的部分哪份 README
都能写，失败的部分只有日志有。

---
'''

TAIL = '''\

---

## 磁盘上但没被跟踪的

| 目录 | 是什么 | 为什么没跟踪 |
|---|---|---|
| `llamacpp/` | 从 llama.cpp 拉的参照源码 | 可重新下载；`by1instella.py` 会读它 |
| `__delete_test/files/` | 删数据测试那 5% 的副本 | 和仓库里的原件是同一批字节，跟踪会存两遍 |
| `__pycache__/` | Python 字节码 | 每次运行重新生成 |

---

## 两个约定

**一、同一个意思不要两处实现。**
路径规则在 `src/by1refs.py`，仓库布局在 `src/by1paths.py`，版本号在
`src/by1ver.py`，文件读写在 `src/by1io.py`。"顺手再写一遍"是这个仓库里
反复出现过的那类 bug。

**二、能描述 ≠ 能算。**
三个后端对拍只能证明自洽。真正的判据在 `src/by1oracles.py` 那 10 个神谕里，
以及 `src/by1verify.py` 对着官方产物的对拍里。
'''


def tracked():
    r = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    # **看返回码。** 不在 git 仓库里跑时 `ls-files` 会失败，
    # 而那时 stdout 是空的 —— 空文件清单会生成一份空文档，
    # 看起来像"这个仓库没有文件"。
    if r.returncode != 0:
        raise SystemExit('  git ls-files 失败（rc=%d）：%s'
                         % (r.returncode, (r.stderr or '').strip()[:80]))
    return [x for x in r.stdout.split('\n') if x.strip()]


def summary(path):
    """docstring 里第一行带 ` -- ` 的。取不到返回 None。"""
    full = os.path.join(ROOT, path)
    try:
        lines = by1io.read_text(full, errors='replace').split('\n')
    except OSError:
        return None
    for ln in lines[:60]:
        s = ln.strip().strip('"').strip()
        if ' -- ' in s and not s.startswith('#'):
            return s.split(' -- ', 1)[1].strip().strip('*').rstrip('。')
    return None


def describe(path):
    """一句话说明。取不到就返回 None —— **不编。**"""
    base = os.path.basename(path)
    if base in OVERRIDE:
        return OVERRIDE[base]
    s = summary(path)
    if s:
        return s
    # md / by1 / 脚本：试头部注释
    full = os.path.join(ROOT, path)
    try:
        head = by1io.head_text(full, 2000)
    except OSError:
        return None
    # `.by1` 先看头部那两个标签 —— 那才是它的身份，
    # 而不是文件开头那行装饰。
    if path.endswith('.by1'):
        m = re.search(r'^#\s*by1-repo:\s*(\S+)', head, re.M)
        mo = re.search(r'^#\s*by1-model:\s*(\S+)', head, re.M)
        if m or mo:
            return ('真实模型 · HF id `%s`' % m.group(1)) if m else \
                   ('短名 `%s`（合成 / 形状测试，没有 HF 对照）' % mo.group(1))
    for ln in head.split('\n')[:12]:
        t = ln.strip().lstrip('#').strip().strip('/*').strip()
        # **跳过带自己文件名的行。**
        #
        # 那些是文件开头的标题装饰（`# ── glm53.by1 ────────`），
        # 不是说明。原来没有这一步，于是它们被当成说明抄进文档 ——
        # 而那些标题里写的是**旧文件名**（改名之前的），
        # 让 by1docs 报出一串不存在的路径。
        #
        # 判据用"含不含自己的名字"而不是"是不是全符号"：
        # `── gpt2-tiny.by1 ────` 里是有字母的，纯符号那套抓不住。
        if base in t or os.path.splitext(base)[0] in t:
            continue
        if t and not t.startswith('-*-') and not t.startswith('!') and len(t) > 6:
            return t.rstrip('。')
    return None


def generate():
    files = tracked()
    by_dir = {}
    for f in files:
        d = os.path.dirname(f) or '.'
        by_dir.setdefault(d, []).append(f)

    problems = []
    out = [HEAD.format(n=len(files))]

    # ── 根目录 ──────────────────────────────────────────────────
    roots = sorted(by_dir.get('.', []))
    out.append('## 根目录（%d 个）\n' % len(roots))
    out.append('| 文件 | 用途 |')
    out.append('|---|---|')
    for f in roots:
        d = describe(f)
        if not d:
            problems.append(f)
            d = '**（没写说明）**'
        out.append('| `%s` | %s |' % (f, d))
    out.append('\n生成物有两份判卷人看着（`by1docs.py`），'
               '所以改了 schema 忘了重新生成会被抓出来。\n')
    out.append('---\n')

    # ── src/ 按组 ───────────────────────────────────────────────
    # **src/ 递归一层。** 搬家之后模块不在同一层了。
    src_list = []
    for sub in ('',) + tuple(by1paths.SUBDIRS):
        d = 'src/' + sub if sub else 'src'
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for f in sorted(os.listdir(full)):
            # **`__init__.py` 不算模块。** 它只有一个用途：
            # 让 setuptools 装得上那个子目录（见 pyproject 里的
            # `packages`）。把它列进 FILES.md 只会让"这个文件干什么的"
            # 多五行噪音。
            if f == '__init__.py':
                continue
            if f.endswith('.py') or f.endswith('.c'):
                src_list.append(d + '/' + f)
    by_dir['src'] = src_list
    # 名字 -> 路径。`by1check.py` 和 `ext-demo.c` 都按去扩展名的名字索引。
    src_map = {}
    for f in src_list:
        b = os.path.basename(f)
        src_map[b[:-3] if b.endswith('.py') else b] = f
    src_files = set(src_map)
    assigned = set()
    n_src_py = len([f for f in src_list if f.endswith('.py')])
    out.append('## `src/` —— %d 个模块 + %d 个非 .py\n'
               % (n_src_py, len(src_list) - n_src_py))

    for title, _d, intro, members in GROUPS:
        present = [m for m in members if m in src_files]
        missing = [m for m in members if m not in src_files]
        assigned.update(present)
        if missing:
            problems.append('GROUPS 里的 %s 不存在' % ', '.join(missing))
        out.append('### %s\n' % title)
        if intro:
            out.append(intro + '\n')
        out.append('| 文件 | 用途 |')
        out.append('|---|---|')
        for m in present:
            path = src_map[m]
            d = describe(path)
            if not d:
                problems.append(path)
                d = '**（没写说明）**'
            out.append('| `%s` | %s |' % (os.path.basename(path), d))
        out.append('')

    unassigned = sorted(src_files - assigned)
    if unassigned:
        for u in unassigned:
            problems.append('src/%s 没有归组' % u)
        out.append('### **未归类**\n')
        out.append('这一组的文件还没被分到任何一类 —— '
                   '要改的是 `src/by1files.py` 里的 `GROUPS`。\n')
        out.append('| 文件 | 用途 |')
        out.append('|---|---|')
        for u in unassigned:
            out.append('| `%s` | %s |' % (u, describe('src/' + u) or '—'))
        out.append('')
    out.append('---\n')

    # ── 其余目录 ────────────────────────────────────────────────
    for d in sorted(k for k in by_dir if k not in ('.', 'src')):
        fs = sorted(by_dir[d])
        out.append('## `%s/` —— %d 个\n' % (d, len(fs)))
        if d in DIR_INTRO:
            out.append(DIR_INTRO[d] + '\n')
        if d == 'refs':
            kinds = {}
            for f in fs:
                b = os.path.basename(f)
                k = ('config' if b.endswith('.config.json') else
                     'tensors' if b.endswith('.tensors.json') else
                     'gguf' if b.endswith('.gguf-tensors.json') else '其它')
                kinds.setdefault(k, []).append(b)
            out.append('| 种类 | 个数 | 是什么 |')
            out.append('|---|---|---|')
            for k, v in sorted(kinds.items()):
                out.append('| `*.%s.json` | %d | |' % (k, len(v))
                           if k != '其它' else '| `%s` | %d | |' % (v[0], len(v)))
            out.append('')
        else:
            out.append('| 文件 | 用途 |')
            out.append('|---|---|')
            for f in fs:
                dsc = describe(f)
                if not dsc:
                    problems.append(f)
                    dsc = '**（没写说明）**'
                out.append('| `%s` | %s |' % (os.path.basename(f), dsc))
        out.append('')

    out.append(TAIL)
    return '\n'.join(out), problems


def main():
    text, problems = generate()
    stale = False
    if '--write' in sys.argv:
        by1io.write_text(OUT, text)
        print('  写了 FILES.md：%d 行' % len(text.split('\n')))
    else:
        cur = by1io.read_text(OUT) if os.path.exists(OUT) else ''
        if cur.strip() == text.strip():
            print('  FILES.md 和目录树一致（%d 行）' % len(text.split('\n')))
        else:
            stale = True
            print('  **FILES.md 和目录树不一致** —— 跑 `by1files.py --write`')

    if problems:
        print()
        print('  **%d 个问题：**' % len(problems))
        for p in problems:
            print('     %s' % p)
    else:
        print('  每个文件都有说明，也都归了组')

    # 判定行 —— `by1all` 靠它认结果（JUDGES 是不带参数跑的）
    bad = stale or bool(problems)
    print()
    print('  [%s] FILES.md %s'
          % ('PASS' if not bad else 'FAIL',
             '和目录树一致，每个文件都有说明也都归了组'
             if not bad else
             '%d 处（%s）'
             % (len(problems) + (1 if stale else 0),
                '，'.join(filter(None, ['文档过期' if stale else '',
                                        '%d 个文件没说明或没归组' % len(problems)
                                        if problems else ''])))))
    print()
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
