#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1paths -- **仓库布局的唯一真相源。**

## 为什么要有这个文件

在这之前，"仓库在哪"这件事**在 40 个文件里各写了一遍**：

    HERE = os.path.dirname(os.path.abspath(__file__))

那时 `HERE` 恰好**等于**仓库根 —— 所以 `os.path.join(HERE, 'refs')`、
`glob.glob('*.by1')`、`open('models.tsv')` 全都碰巧是对的。

代码搬进 `src/` 之后，`HERE` 变成 `src/`，**那些"碰巧对"的地方一起失效**。
而它们失效的方式很像"文件丢了"，不像"引用没跟上" ——
和 `by1refs.py` 里记的那次 `refs/minimind-3.tensors.json` 是同一类事故：

    改成 `refs/jingyaogong__minimind-3.tensors.json` 之后，
    改名脚本知道，**但四个文件里的 28 处字符串不知道**。

所以布局也只写一处。`by1refs.py` 管的是 **refs/ 里面的命名规则**，
这里管的是**这些东西各自在哪个目录**。两件事，两个文件。

## 三条规矩

① **`HERE` 现在只能回答一件事："我的代码在哪"。**

    HERE       我的代码在哪      -> 只用来拼**和自己同目录**的东西
    by1paths   别人在哪、仓库在哪  -> 其余一律问它

    `src/` 平铺的时候这两件事的答案是同一个目录，所以从来没人需要
    区分它们。**分了子目录之后又裂开一层**：

        os.path.join(HERE, 'by1check.py')      # 错 —— 兄弟在别的目录

    而这个错**不是 `ImportError`，是 `FileNotFoundError`**，
    路径里写的是 `src/checks/by1check.py` —— 看起来像"那个文件丢了"，
    不像"它在 `src/` 而不在 `src/checks/`"。所以兄弟一律用 `tool()`。
    （全仓库 28 处一起改的，见 `ir.md`。）

② **`model()` 收裸名，也收路径。**

    python src/by1exec.py hello.by1            # 裸名 -> models/hello.by1
    python src/by1exec.py models/hello.by1     # 路径 -> 原样

   于是 `by1all` 内部传的仍然是裸名 —— **输出里的模型名不会突然多一截
   `models/`**，而搬家前的输出和搬家后的输出可以直接逐行对。

③ **每个入口脚本开头那几行引导是生成的，别手改。**

    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(...)))

   为什么需要它：**Python 没有"在 import 之前就生效"的钩子**，
   而入口脚本必须能被直接 `python src/checks/by1diff.py` 跑起来。
   包（`from by1.core import ...`）能免掉它，但那要改 200 处 import，
   而这个仓库的入口就是"跑一个脚本"。

   判据在 `check_boot()` —— **重复的东西必须能被检查**，
   否则新加一个脚本忘了加，症状是 `No module named 'by1paths'`。

## 只依赖标准库

这个模块会被几乎所有 `by1*.py` import，所以它和 `by1io.py` 一样
**不能**引入 numpy / torch / transformers。
"""
import glob as _glob
import io
import os

# **只为编码。** `by1io` 在 import 时把 stdout/stderr 钉成 UTF-8，
# 而这个模块几乎每个脚本都会 import —— 所以放在这里，钉一次全都沾光。
# （不要因此把 by1io 变成"什么都往里塞"的模块：它只管读写和这一条约定。）
import by1io  # noqa: F401

# ── src/ 下的子目录 ────────────────────────────────────────────────
#
# **每一个都要在 `sys.path` 上。** 这个仓库的模块互相 `import by1check`
# 这样的裸名 —— 平铺的时候它们碰巧都在一个目录里，所以能用。
# 搬进子目录之后，"碰巧"没有了，得显式放上去。
#
# 顶层那 12 个**必须**留在 `SRC` 本身：它们是**被 import 的底座**
# （`by1io` 被 39 个模块用、`by1paths` 被 33 个用）。
# 搬它们就得先有引导，而引导又得先找得到它们。
SUBDIRS = ('checks', 'modelcheck', 'lang', 'data', 'escape')

# ── 五个目录，各一个名字 ────────────────────────────────────────────
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
MODELS = os.path.join(ROOT, 'models')
REFS = os.path.join(ROOT, 'refs')
DOCS = os.path.join(ROOT, 'docs')
HISTORY = os.path.join(ROOT, 'history')


def _bootstrap():
    """把 `SRC` 和每个子目录放上 `sys.path`。**import 本模块时自动跑。**

    顺序无所谓 —— 这些目录里的模块名不重名（有重名的话，
    谁在前面谁赢，那才是真的会静默出错）。
    """
    import sys
    for d in (SRC,) + tuple(os.path.join(SRC, s) for s in SUBDIRS):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


_bootstrap()


def check_boot():
    """每个"入口脚本"开头有没有那两行引导。**返回缺的那些。**

    那两行是重复的（44 份），所以必须**能被检查** ——
    否则哪天有人新加一个脚本、忘了加引导，症状是
    `ModuleNotFoundError: No module named 'by1paths'`，
    看起来像"文件丢了"，不像"少了引导"。
    """
    missing = []
    for d in (SRC,) + tuple(os.path.join(SRC, s) for s in SUBDIRS):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.py'):
                continue
            p = os.path.join(d, f)
            try:
                with io.open(p, encoding='utf-8', errors='replace') as fh:
                    body = fh.read()
            except OSError:
                continue
            if "__main__" not in body:
                continue                      # 不是入口，不需要
            head = body[:2000]
            if '_sys.path.insert' not in head:
                missing.append(rel(p))
    return missing


def root(*parts):
    """仓库根下的路径。"""
    return os.path.join(ROOT, *parts)


def tool(name):
    """兄弟脚本的路径。**在 `src/` 和每个子目录里找。**

    调用处写的是裸名（`by1check.py`），**不关心它现在住在哪个子目录** ——
    搬过一次家就是这个理由：路径只在这一处推。
    """
    for d in (SRC,) + tuple(os.path.join(SRC, s) for s in SUBDIRS):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    # **找不到就给出期望的位置**，不是 None ——
    # 一个指向错误路径的报错比一个指向 None 的强。
    return os.path.join(SRC, name)


def ref(*parts):
    """`refs/` 下的路径。"""
    return os.path.join(REFS, *parts)


def find_model(name):
    """解析一个模型文件。**找不到返回 None。**

    顺序：
      ① 绝对路径                    -> 原样
      ② 相对 cwd 存在                -> 原样（`models/hello.by1` 走这条）
      ③ `models/<basename>` 存在     -> 那条（主路）
      ④ 都不在                       -> None
    """
    if not name:
        return None
    if os.path.isabs(name):
        return name
    if os.path.exists(name):
        return name
    p = os.path.join(MODELS, os.path.basename(name))
    return p if os.path.exists(p) else None


def model(name):
    """同 `find_model`，但**总是给出一条路径**。

    找不到时给的是**期望的位置**（`models/<basename>`）——
    这样报错会指向"它该在哪"，而不是指向 cwd。
    一个指向 cwd 的 `FileNotFoundError` 看起来像"文件丢了"。
    """
    if not name:
        return name
    if os.path.isabs(name):
        return name
    if os.path.exists(name):
        return name
    return os.path.join(MODELS, os.path.basename(name))


def models(pattern='*.by1'):
    """`models/` 下匹配 pattern 的路径，排序。

    **返回绝对路径。** 要显示给人看的名字用 `os.path.basename()` ——
    判断"有几个模型"和"打印哪个模型"是两件事。
    """
    return sorted(_glob.glob(os.path.join(MODELS, pattern)))


def names(pattern='*.by1'):
    """`models/` 下匹配 pattern 的**基名**，排序。

    给"要拿到裸名去传子进程/打印"的地方用 —— 这样输出和搬家前一致。
    """
    return [os.path.basename(p) for p in models(pattern)]


def rel(path):
    """相对仓库根的路径，用来显示。根外的路径原样返回。"""
    try:
        r = os.path.relpath(path, ROOT)
    except ValueError:
        return path
    return path if r.startswith('..') else r


# ── 找 gcc ──────────────────────────────────────────────────────────
#: 候选项。**裸名字要用 `shutil.which` 搜 PATH** —— `glob.glob('gcc')`
#: 只找当前目录，by1all 那边已经为这个踩过一次坑（"看起来改了，其实没生效"）。
GCC_CANDIDATES = (
    'gcc', 'cc',                                   # PATH 上先找，两个平台都对
    '/usr/bin/gcc', '/usr/local/bin/gcc',          # Linux
    '/usr/bin/cc', '/opt/homebrew/bin/gcc',        # cc / macOS
)

#: Windows：WinGet 装的 mingw（**相对路径要展开**，`%LOCALAPPDATA%`）
GCC_WIN_GLOBS = (
    r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'
    r'\BrechtSanders*\mingw64\bin\gcc.exe',
)


def find_gcc():
    """找一个可用的 C 编译器。**找不到返回 None。**

    这件事原来在**三个脚本里各写了一遍、规则还不一样**：
    `by1all` 是对的（`shutil.which` + Linux 路径 + WinGet glob），
    `by1extdemo` 是第二版（有 which，WinGet 兜底），
    而 `by1e2e` **只剩 WinGet 那一条** —— 于是在 Linux 上它
    "找不到 gcc"，整段 C 后端被静默跳过，**而判定行照样印
    「三个后端 [PASS]」**。同一件事三份实现，坏的那份没人发现。

    所以放一处。谁要用谁 import —— `by1paths` 本来就是每个脚本
    都会 import 的那个模块，加在这里不引入新依赖。
    """
    import glob as _glob
    import shutil as _sh
    for c in GCC_CANDIDATES:
        if os.sep in c or '/' in c:
            if os.path.exists(c):
                return c
            continue
        hit = _sh.which(c)
        if hit:
            return hit
    for g in GCC_WIN_GLOBS:
        hits = _glob.glob(os.path.expandvars(g))
        if hits:
            return hits[0]
    return None


def _check():
    print()
    print('  仓库布局')
    print('  ' + '-' * 62)
    for label, p in (('ROOT', ROOT), ('SRC', SRC), ('MODELS', MODELS),
                     ('REFS', REFS), ('DOCS', DOCS), ('HISTORY', HISTORY)):
        n = len(_glob.glob(os.path.join(p, '*'))) if os.path.isdir(p) else 0
        print('  %-8s %-42s %s' % (
            label, rel(p) or '.', '%d 项' % n if os.path.isdir(p) else '**不在**'))
    print()
    print('  models/ 下 %d 个 .by1，其中 %d 个带 by1-repo 头'
          % (len(models()), len([f for f in models()
                                 if '# by1-repo:' in _head(f)])))
    print('  refs/   下 %d 个产物' % len(_glob.glob(os.path.join(REFS, '*.json'))))
    print()


def _head(p):
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            return f.read(2000)
    except OSError:
        return ''


if __name__ == '__main__':
    _check()
