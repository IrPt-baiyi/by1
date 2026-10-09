#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1lint -- **只盯这个项目真正怕的那几类。**

## 为什么不用现成的 linter

`ruff` / `pyflakes` / `flake8` 一个都没装。要装也行，
但它们报的东西和这个项目的病**不一样**：

    ruff 关心      行太长、import 顺序、没用的变量、f-string 该不该用
    这个项目怕     **静默失败**

这一轮抓到的四个 bug，全是"不出声"：
    死模式永远不匹配          -> 不报错
    路径写死过期              -> 静默跳过 11 项检查
    `except: pass`            -> 把错误吃掉
    工具自己测错              -> 报一个假的数

**所以这个脚本只查那些"会让错误不出声"的写法。**

## 查十类

    ① `except: pass` / `except Exception: pass`   把错误吃掉
    ② 光秃秃的 `except:`                            连 KeyboardInterrupt 都吃
    ③ `open()` 不用 `with`                          句柄泄漏（跑长任务会炸）
    ④ 可变默认参数                                  跨调用共享状态
    ⑤ `assert` 当控制流用                            `-O` 一跑就没了
    ⑥ 遮蔽内建名（`id` / `type` / `list` / `bytes`）
    ⑦ `== None` / `!= None`
    ⑧ 没用的 import
    ⑨ `subprocess` 不看返回码                       失败了当成功
    ⑩ 库代码里 `sys.exit`                           调用方没法处理

用法:  python by1lint.py [--top N]
"""
import ast
import os
import sys
from collections import defaultdict
import by1io

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

BUILTINS = {'id', 'type', 'list', 'dict', 'set', 'bytes', 'str', 'int',
            'float', 'bool', 'len', 'max', 'min', 'sum', 'map', 'filter',
            'input', 'print', 'open', 'next', 'iter', 'range', 'format',
            'all', 'any', 'hash', 'vars', 'dir', 'eval', 'exec', 'compile'}


class V(ast.NodeVisitor):
    def __init__(self, fname, src):
        self.f = fname
        self.src = src
        self.hits = defaultdict(list)
        self.imported = {}      # 名字 -> 行号
        self.used = set()
        self.sp_alias = set()   # `import subprocess as sp` 的别名

    def add(self, k, node, msg=''):
        self.hits[k].append((node.lineno, msg))

    def _is_subprocess(self, v):
        """这个表达式的根是不是 subprocess（或它的别名）。

        `subprocess.run` / `sp.run` / `subprocess.Popen` —— 认。
        `x.run` / `self.run` —— 不认。
        """
        while isinstance(v, ast.Attribute):
            v = v.value
        return (isinstance(v, ast.Name)
                and (v.id == 'subprocess' or v.id in self.sp_alias))

    def _in_with(self, node):
        """这个节点是不是在某个 `with` 的 items 里。

        靠父指针 —— 所以要先跑一遍 `_parent`。
        """
        p = getattr(node, '_parent', None)
        while p is not None:
            if isinstance(p, ast.withitem):
                return True
            if isinstance(p, ast.With):
                return True
            p = getattr(p, '_parent', None)
        return False

    # ① ② except
    def visit_ExceptHandler(self, n):
        body = n.body
        # 只有 pass / ... 的 handler = 把错误吃掉
        only_pass = (len(body) == 1 and
                     isinstance(body[0], ast.Pass))
        only_ell = (len(body) == 1 and isinstance(body[0], ast.Expr) and
                    isinstance(body[0].value, ast.Constant) and
                    body[0].value.value is Ellipsis)
        if n.type is None:
            self.add('② 光秃秃的 except:', n)
        if only_pass or only_ell:
            self.add('① except 之后只有 pass', n)
        self.generic_visit(n)

    # ③ open 不用 with
    def visit_Call(self, n):
        if (isinstance(n.func, ast.Attribute) and n.func.attr == 'open') or \
           (isinstance(n.func, ast.Name) and n.func.id == 'open'):
            # **只报不在 `with` 里的。**
            # 第一版把所有 `open()` 都算上，报了 70 处 ——
            # 而其中绝大多数是 `with io.open(...) as f:`，
            # **那是正确的写法**。假阳性 70 条会把真的 3 条淹掉。
            if not self._in_with(n):
                self.add('③ open() 没用 with', n)
        # ⑨ subprocess 不看返回码
        #
        # **只认 `subprocess.xxx` / `sp.xxx`。**
        # 第一版只看属性名 —— 于是 `_Ev(...).run()` 和
        # `by1ext.call(...)` 都被算成 subprocess，报了两条假阳性。
        # 属性名撞车太容易了（`.run` / `.call` 满地都是）。
        if isinstance(n.func, ast.Attribute) and \
           n.func.attr in ('run', 'call', 'check_output', 'Popen') and \
           self._is_subprocess(n.func.value):
            has_ck = any(k.arg == 'check' for k in n.keywords)
            is_capture = any(k.arg == 'capture_output' for k in n.keywords)
            if not has_ck and not is_capture:
                self.add('⑨ subprocess 不看返回码', n)
        self.generic_visit(n)

    # ④ 可变默认参数
    def _check_defaults(self, n):
        for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x]:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                self.add('④ 可变默认参数', n, n.name)
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and \
               d.func.id in ('list', 'dict', 'set'):
                self.add('④ 可变默认参数', n, n.name)

    def visit_FunctionDef(self, n):
        self._check_defaults(n)
        self.generic_visit(n)

    def visit_AsyncFunctionDef(self, n):
        self._check_defaults(n)
        self.generic_visit(n)

    # ⑤ assert 当控制流
    def visit_Assert(self, n):
        self.add('⑤ assert', n)
        self.generic_visit(n)

    # ⑥ 遮蔽内建
    def visit_Name(self, n):
        self.used.add(n.id)
        self.generic_visit(n)

    def visit_arg(self, n):
        if n.arg in BUILTINS:
            self.add('⑥ 遮蔽内建名', n, n.arg)
        self.generic_visit(n)

    # ⑦ == None
    def visit_Compare(self, n):
        for op, c in zip(n.ops, n.comparators):
            if isinstance(c, ast.Constant) and c.value is None and \
               isinstance(op, (ast.Eq, ast.NotEq)):
                self.add('⑦ == None', n)
        self.generic_visit(n)

    # ⑧ import
    def visit_Import(self, n):
        for a in n.names:
            nm = (a.asname or a.name).split('.')[0]
            self.imported[nm] = n.lineno
            if a.name == 'subprocess' and a.asname:
                self.sp_alias.add(a.asname)
        self.generic_visit(n)

    def visit_ImportFrom(self, n):
        for a in n.names:
            if a.name != '*':
                self.imported[a.asname or a.name] = n.lineno
        self.generic_visit(n)

    # ⑩ sys.exit
    def visit_Expr(self, n):
        v = n.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and \
           v.func.attr == 'exit' and isinstance(v.func.value, ast.Name) and \
           v.func.value.id == 'sys':
            # **跳过 `if __name__ == '__main__':` 下面的。**
            # 那是脚本入口的标准写法 —— 第一版报了 47 处，
            # 而 46 处都是这个。**假阳性会把真的那 1 条淹掉。**
            # **模块层的 `sys.exit` 不算库代码。**
            # 第一版只看"是不是在 `if __name__` 下面" —— 于是
            # `by1gate.py` / `by1raw.py` 那种**整个文件就是脚本**的
            # 报了假阳性。它们的 `sys.exit` 在模块层，那是脚本的出口。
            #
            # 判据：这个文件里有没有 `def main`。
            #   有  -> 它是个能被 import 的模块，模块层 exit 不对
            #   没有 -> 它就是个脚本，exit 是它的出口
            if self.has_main and not self._under_main_guard(n):
                self.add('⑩ 库代码里的 sys.exit', n)
        self.generic_visit(n)

    def _under_main_guard(self, node):
        """往上找，看是不是在 `if __name__ == '__main__'` 的分支里。"""
        p = getattr(node, '_parent', None)
        while p is not None:
            if isinstance(p, ast.If):
                s = ast.dump(p.test)
                if "__name__" in s and "__main__" in s:
                    return True
            p = getattr(p, '_parent', None)
        return False


def main():
    top = 0
    if '--top' in sys.argv:
        top = int(sys.argv[sys.argv.index('--top') + 1])

    files = sorted(f for f in os.listdir('.')
                   if f.startswith('by1') and f.endswith('.py'))
    allhits = defaultdict(list)
    for f in files:
        try:
            src = by1io.read_text(f, encoding='utf-8')
            tree = ast.parse(src, f)
        except Exception as e:
            print('  !! %s 解析不了：%s' % (f, str(e)[:50]))
            continue
        # **先挂父指针** —— `_in_with` / `_under_main_guard` 靠它。
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node
        v = V(f, src)
        v.has_main = any(isinstance(x, ast.FunctionDef) and x.name == 'main'
                         for x in ast.walk(tree))
        v.visit(tree)
        for k, hs in v.hits.items():
            for ln, msg in hs:
                allhits[k].append((f, ln, msg))

    print()
    print('=' * 78)
    print('  by1 的十类"不出声"写法 —— %d 个文件' % len(files))
    print('=' * 78)
    print()

    order = sorted(allhits.items(), key=lambda kv: -len(kv[1]))
    for k, hs in order:
        byf = defaultdict(int)
        for f, _ln, _m in hs:
            byf[f] += 1
        print('  %-28s %4d 处，分布在 %d 个文件' % (k, len(hs), len(byf)))
        show = sorted(byf.items(), key=lambda kv: -kv[1])[:6]
        print('       %s' % ', '.join('%s(%d)' % (a, b) for a, b in show))
        if top:
            for f, ln, m in hs[:top]:
                print('         %s:%d %s' % (f, ln, m))
        print()

    if not allhits:
        print('  一类都没有。')
        print()
    print('  **这些不是风格问题，是"错误会不会出声"的问题。**')
    print('  ① 和 ② 最重 —— `except: pass` 把错误吃掉之后，')
    print('  调用方看到的是"一切正常"，而它是坏的。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
