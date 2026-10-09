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

## 两条规矩

① **`HERE` 仍然是对的 —— 只是它的意思变了。**

    HERE   回答"我的代码在哪"          -> 兄弟脚本用 HERE 拼，**不用改**
    ROOT   回答"仓库在哪"              -> 仓库数据用 by1paths，**要改**

   搬家前这两件事的答案是同一个目录，所以从来没人需要区分它们。
   现在不是了。

② **`model()` 收裸名，也收路径。**

    python src/by1exec.py hello.by1            # 裸名 -> models/hello.by1
    python src/by1exec.py models/hello.by1     # 路径 -> 原样

   于是 `by1all` 内部传的仍然是裸名 —— **输出里的模型名不会突然多一截
   `models/`**，而搬家前的输出和搬家后的输出可以直接逐行对。

## 只依赖标准库

这个模块会被几乎所有 `by1*.py` import，所以它和 `by1io.py` 一样
**不能**引入 numpy / torch / transformers。
"""
import glob as _glob
import os

# **只为编码。** `by1io` 在 import 时把 stdout/stderr 钉成 UTF-8，
# 而这个模块几乎每个脚本都会 import —— 所以放在这里，钉一次全都沾光。
# （不要因此把 by1io 变成"什么都往里塞"的模块：它只管读写和这一条约定。）
import by1io  # noqa: F401

# ── 五个目录，各一个名字 ────────────────────────────────────────────
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
MODELS = os.path.join(ROOT, 'models')
REFS = os.path.join(ROOT, 'refs')
DOCS = os.path.join(ROOT, 'docs')
HISTORY = os.path.join(ROOT, 'history')


def root(*parts):
    """仓库根下的路径。"""
    return os.path.join(ROOT, *parts)


def tool(name):
    """兄弟脚本的路径。

    **不是 `HERE` 的替代品** —— 只是让调用处读起来是意图而不是拼接。
    """
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
