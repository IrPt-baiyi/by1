# -*- coding: utf-8 -*-
"""by1 的内部包（迁移计划见 docs/TEMP-rename-plan.md）。

**这个 __init__ 只做一件事**：注册一个 import 钩子，让**旧名**
`import by1io` / `from by1check import ...` 仍然可用 —— 它们被转发到
包内真身（`by1.io` / `by1.check`），**是同一个模块对象**，不是副本
（副本会让状态分叉：by1check 里的缓存、by1paths 的 sys.path 各有一份）。

为什么保留旧名而不是改几百处 import：旧名是**文档、判卷人的命令行、
外部脚本都在用的拼写**；改一遍只会把风险摊到几百处，而名字并不会变短。
钩子是懒的：真身只在第一次 import 旧名时才载入。
"""
import importlib.abc as _abc
import importlib.util as _util
import os as _os
import sys as _sys

_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_CACHE = {}


def _resolve(stem):
    """旧名 `by1<stem>` → 包内模块的完整名；找不到返回 None。"""
    if stem in _CACHE:
        return _CACHE[stem]
    found = None
    # **关键字的逃逸名**：`by1exec` → `exec_`（`exec` 不能做模块名）。
    names = (stem + '.py', stem + '_.py')
    for base, dirs, files in _os.walk(_PKG_DIR):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        hit = next((n for n in names if n in files), None)
        if hit:
            stem = hit[:-3]
            rel = _os.path.relpath(_os.path.join(base, stem), _PKG_DIR)
            found = 'by1.' + rel.replace(_os.sep, '.')
            break
    _CACHE[stem] = found
    return found


class _AliasLoader(_abc.Loader):
    """把 `import by1<stem>` 变成 `import by1.<pkg>.<stem>`（同一对象）。"""

    def __init__(self, full):
        self.full = full

    def create_module(self, spec):
        import importlib as _il
        return _il.import_module(self.full)

    def exec_module(self, module):
        pass


class _LegacyFinder(_abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if '.' in name or not name.startswith('by1') or len(name) <= 3:
            return None
        full = _resolve(name[3:])
        if not full:
            return None
        return _util.spec_from_loader(name, _AliasLoader(full))


_sys.meta_path.append(_LegacyFinder())

__all__ = []
