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

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
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
        # **逐行的原文。** 规则⑧ 要认 `# noqa` —— 那是标准逃生口。
        # 不认它的后果不是"多报一条"，是**逼人把故意的 import 删掉**
        # （`by1paths` 里 `import by1io` 就是靠它的副作用把 stdout
        # 钉成 UTF-8）。删掉之后没有检查会红 —— 那个副作用就静默没了。
        self.lines = src.split('\n')
        self.hits = defaultdict(list)
        self.imported = {}      # 名字 -> 行号
        self.used = set()
        self.sp_alias = set()   # `import subprocess as sp` 的别名
        self._owner = None      # 当前在哪个函数体里（规则⑨ 要看"这个函数读过结果吗"）

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
        """这个节点是不是**某个 `with` 的 item 表达式**。

        ## 原来它是错的

        第一版往上找，只要祖先是 `ast.With` 就返回 True ——
        而**`with` 的 body 的祖先也是那个 `With`**。于是：

            with open('a') as f:      # 这个报
                g = open('b')         # 这个**不报**（它是 body 里的）

        也就是说，规则只在"文件里一个 with 都没有"时才有效。
        正确判据：往上找到的 `withitem` 必须**真的包含**这个节点
        （`withitem.context_expr` 是它，或者它是那个表达式的子节点）。
        """
        p = getattr(node, '_parent', None)
        while p is not None:
            if isinstance(p, ast.withitem):
                # 确认这个 withitem 的表达式确实包着 node
                for sub in ast.walk(p.context_expr):
                    if sub is node:
                        return True
                return False
            # 撞到语句边界就停：`with` 的 body 从这里开始，
            # 里面的 `open()` 不该被算成"在 with 里"。
            if isinstance(p, (ast.With, ast.AsyncWith, ast.FunctionDef,
                              ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
                return False
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
        #
        # **`capture_output=True` 不等于"看了返回码"。**
        # 原来这里是 `if not has_ck and not is_capture` —— 于是
        #     r = subprocess.run([...], capture_output=True, text=True)
        #     return r.stdout          # returncode 一次都没读
        # 被判成"没问题"。**这个判据自己就是一个静默失败**：
        # 它把"我要了输出"读成了"我检查了结果"。
        # 现在改成看**这个函数里有没有碰过 returncode / stdout / stderr**，
        # 或者调用点允不允许失败（`check=True` / `check=False` 显式表态）。
        if isinstance(n.func, ast.Attribute) and \
           n.func.attr in ('run', 'call', 'check_output', 'Popen') and \
           self._is_subprocess(n.func.value):
            has_ck = any(k.arg == 'check' for k in n.keywords)
            if not has_ck and not self._reads_result(n):
                self.add('⑨ subprocess 不看返回码', n)
        self.generic_visit(n)

    def _reads_result(self, call):
        """调用之后，**同一个函数里**有没有人读过 `returncode`。

        ## 只看 `returncode`，不看 `stdout` / `stderr`

        第一版把三者并列，于是这条形状被判成"没问题"：

            r = subprocess.run([...], capture_output=True, text=True)
            return r.stdout          # returncode 一次都没读

        **读输出 ≠ 知道失败了没有。** 一个退出码 1 的程序照样有 stdout，
        而 `r.stdout` 读得越顺，越像是用成功了。
        这正是这个脚本要抓的那类静默失败 —— 它自己先犯了一次。

        ## 为什么要"同一函数 + 调用点之后"

        `_owner` 限定在**它所在的那个函数**（不限定的话，
        隔壁函数读过 returncode 会把这里盖过去）；
        行号限定在**调用点之后**（不限定的话，调用自己的
        `Attribute(attr='run', lineno=调用行)` 会把自己数进来）。
        """
        owner = self._owner
        if owner is None:
            return False
        # **调用自己的那个属性不算。**
        # `subprocess.run(...)` 里也有一个 `Attribute(attr='run')`，
        # 它的 lineno 就是调用行 —— 天真的 `>= call.lineno` 会数到它。
        skip = {id(x) for x in ast.walk(call) if isinstance(x, ast.Attribute)}
        base = call.func.value if isinstance(call.func, ast.Attribute) else None
        if base is not None:
            skip.add(id(base))
        for fn in ast.walk(owner):
            if not isinstance(fn, ast.Attribute) or fn.attr != 'returncode':
                continue
            if id(fn) in skip:
                continue
            if fn.lineno > call.lineno or (
                    fn.lineno == call.lineno and self._in_stmt_after(fn, call)):
                return True
        return False

    @staticmethod
    def _in_stmt_after(node, call):
        """同一行的情况：这个属性是不是在**调用之后的另一个语句**里。

        `r = subprocess.run(...); print(r.returncode)` 这种一行两句
        也该算"看过了"。简单判据：它不在 call 这棵子树里。
        """
        for s in ast.walk(call):
            if s is node:
                return False
        return True

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
        # **进函数体时换 `_owner`，出来要还原。**
        # 规则⑨ 问的是"这个 subprocess 调用的结果，在**它所在的这个函数**里
        # 有没有人读过"。不还原的话，前一个函数读过 returncode
        # 会把后一个函数的漏检盖成"没问题"。
        prev, self._owner = self._owner, n
        self.generic_visit(n)
        self._owner = prev

    def visit_AsyncFunctionDef(self, n):
        self._check_defaults(n)
        prev, self._owner = self._owner, n
        self.generic_visit(n)
        self._owner = prev

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
            # **`# noqa` 那一行不算。** 标准逃生口 —— 不认它的后果
            # 不是"多报一条"，是逼人把故意的 import 删掉
            # （`by1paths` 里 `import by1io` 就是靠它的副作用）。
            if 'noqa' in self.lines[n.lineno - 1]:
                return
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


def lint_text(fname, src):
    """对一个源码字符串跑一遍全部规则。返回 `{规则: [(行, 说明)]}`。

    抽成函数是为了让下面的**反例自检**能用同一套规则 ——
    一个判据如果不能证明它会红，那它就不是判据。
    """
    tree = ast.parse(src, fname)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    v = V(fname, src)
    v.has_main = any(isinstance(x, ast.FunctionDef) and x.name == 'main'
                     for x in ast.walk(tree))
    v.visit(tree)
    # ⑧ 没用的 import（**收尾才判** —— 要先收集完整份文件的用途）
    #
    # 三种"看起来没用、其实有用"的写法都要认，否则这条规则会被
    # 假阳性淹掉（而这个项目已经为"假阳性淹掉真问题"踩过两次）：
    #   · `import x  # noqa: F401` —— **显式豁免**，作者写了理由的
    #   · 字符串注解 `-> "Foo"` 和 `typing.get_type_hints`
    #   · `__all__` 里列出来的名字
    noqa_lines = {i + 1 for i, ln in enumerate(src.splitlines())
                  if 'noqa' in ln.lower()}
    # **只在注解 / `__all__` 里认字符串名字。**
    # 一开始我写的是"文件里任何字符串等于这个名字就算用了" ——
    # 那太松了：一句注释里的 'os' 就能把 `import os` 洗白，
    # 这个规则会变成一个永远不报的规则（= 又一个静默失败）。
    str_names = set()
    for n in ast.walk(tree):
        ann = getattr(n, 'annotation', None)
        if ann is not None:
            for e in ast.walk(ann):
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    str_names.add(e.value)
                elif isinstance(e, ast.Name):
                    # **注解里的真名字也算用了。**
                    # `visit_Name` 只挂在表达式上，而注解不是表达式 ——
                    # 漏了这一步，`from typing import Optional` 后面
                    # 写了 `Optional[str]` 都会被报成"没用的 import"。
                    str_names.add(e.id)
    allnames = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '__all__' for t in n.targets):
            for e in ast.walk(n.value):
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    allnames.add(e.value)
    # `os.path` 这种：`import os` 之后只以属性根的身份出现
    used_attr = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            r = n
            while isinstance(r, ast.Attribute):
                r = r.value
            if isinstance(r, ast.Name):
                used_attr.add(r.id)
    for nm, ln in sorted(v.imported.items(), key=lambda kv: kv[1]):
        if ln in noqa_lines:
            continue
        if nm in v.used or nm in used_attr or nm in str_names or nm in allnames:
            continue
        v.hits['⑧ 没用的 import'].append((ln, nm))
    return dict(v.hits)


def main():
    top = 0
    if '--top' in sys.argv:
        top = int(sys.argv[sys.argv.index('--top') + 1])

    if '--selftest' in sys.argv:
        return _selftest()

    # **从 `by1paths` 走全部子目录，不是 `listdir('.')`。**
    #
    # 这个脚本原来 `os.chdir(HERE)` 然后 `listdir('.')` ——
    # 在 `src/` 平铺时代那是对的。搬进 `src/checks/` 之后
    # 它只看得见自己那个目录：**扫的文件从 55 个掉到 22 个，
    # 而判定行照旧 PASS。**
    import by1paths as _p
    files = []
    for _d in (_p.SRC,) + tuple(os.path.join(_p.SRC, _s)
                                  for _s in _p.SUBDIRS):
        if not os.path.isdir(_d):
            continue
        files += [_d + os.sep + f for f in os.listdir(_d)
                  if f.startswith('by1') and f.endswith('.py')]
    files.sort()
    allhits = defaultdict(list)
    for f in files:
        try:
            src = by1io.read_text(f, encoding='utf-8')
            hits = lint_text(f, src)
        except Exception as e:
            print('  !! %s 解析不了：%s' % (f, str(e)[:50]))
            continue
        for k, hs in hits.items():
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

    # ── 判定 ────────────────────────────────────────────────────────
    #
    # **这个脚本原来 `return 0`，无条件。** 于是它报了什么都无所谓 ——
    # 一个不可能红的检查，等于一个不存在的检查（而这个仓库的
    # `KNOWN` 清单、`by1skip` 三态、`confirmed()` 都是为了反这一条）。
    #
    # 现在它有一条判定线：**这十类的命中数必须是 0**。
    # 之所以能要求 0，是因为把这条线接进 `by1all` 之前，
    # 树上已经清干净了 —— 而"清干净"这件事本身要能被复查。
    # 要豁免某一条，就写进 `ALLOW`，**并且写清楚为什么**。
    total = sum(len(v) for v in allhits.values())
    print('  **这些不是风格问题，是"错误会不会出声"的问题。**')
    print('  ① 和 ② 最重 —— `except: pass` 把错误吃掉之后，')
    print('  调用方看到的是"一切正常"，而它是坏的。')
    print()
    ok = total == 0
    print('  [%s] lint 十类静默写法 %d 处%s'
          % ('PASS' if ok else 'FAIL', total,
             '（这一类必须一直是 0）' if ok else ' —— 上面逐条列出来了'))
    return 0 if ok else 1


#: 反例：每一条都要**在合成代码上真的红一次**。
#: 判据不是"我以为它查得到"，是"它确实报出来了"。
_SELFTEST = (
    ('① except 之后只有 pass', 'try:\n    x()\nexcept ValueError:\n    pass\n'),
    ('② 光秃秃的 except:', 'try:\n    x()\nexcept:\n    y()\n'),
    ('③ open() 没用 with', 'f = open("a")\n'),
    ('④ 可变默认参数', 'def f(a=[]):\n    return a\n'),
    ('⑤ assert', 'def f():\n    assert 1\n'),
    ('⑥ 遮蔽内建名', 'def f(id):\n    return id\n'),
    ('⑦ == None', 'def f(x):\n    return x == None\n'),
    ('⑧ 没用的 import', 'import os\n\n\ndef f():\n    return 1\n'),
    ('⑨ subprocess 不看返回码',
     'import subprocess\n\n\ndef f():\n'
     '    r = subprocess.run(["x"], capture_output=True, text=True)\n'
     '    return r.stdout\n'),
    ('⑩ 库代码里的 sys.exit',
     'import sys\n\n\ndef main():\n    return 0\n\n\nsys.exit(main())\n'),
)


def _selftest():
    """**十条规则各有一个必须红的反例。** 外加两个必须不红的正例。"""
    print()
    print('=' * 78)
    print('  by1lint 自检：每条规则都要在反例上真的红一次')
    print('=' * 78)
    print()
    bad = 0
    for want, code in _SELFTEST:
        hits = lint_text('<selftest>', code)
        got = want in hits
        bad += 0 if got else 1
        print('  %s %-28s %s' % ('ok ' if got else '!! ', want,
                                 '报出来了' if got else '**没有报 —— 这条规则是死的**'))
    print()
    # **正例**：这两段是正确的写法，报了才是假阳性。
    GOOD = (
        ('with + open 不该报',
         'def f():\n    with open("a") as fh:\n        return fh.read()\n'),
        ('读了 returncode 不该报',
         'import subprocess\n\n\ndef f():\n'
         '    r = subprocess.run(["x"], capture_output=True, text=True)\n'
         '    return 0 if r.returncode == 0 else 1\n'),
    )
    for label, code in GOOD:
        hits = lint_text('<selftest>', code)
        got = not hits
        bad += 0 if got else 1
        print('  %s %-28s %s' % ('ok ' if got else '!! ', label,
                                 '没报（正确）' if got else
                                 '**报了假阳性：%s**' % list(hits)))
    print()
    print('  [%s] lint 自检 %s'
          % ('PASS' if not bad else 'FAIL',
             '十条规则都会红，两个正例都没误报'
             if not bad else '%d 条不对' % bad))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
