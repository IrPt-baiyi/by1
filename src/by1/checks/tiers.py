#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1tiers -- **档位归属的自动判据**：判卷人声明的档位 vs 它真的需要什么。

## 为什么需要它

`by1all` 的 `JUDGES` 表把每个判卷人钉在某一档（2 = 只要标准库，3 = 加 numpy，
4 = 加 torch/transformers，5 = 要 gcc / 下载，6 = 要显卡）。这条规矩一直**只写在
注释里**，靠人守 —— 而它守的是**整个门禁结构**：一个要 torch 的脚本挂在档 2，
CI 的"纯标准库"那条腿就会红；反过来一个只要标准库的挂在档 4，它就**永远跑不到**。

## 五条规则

    T1 存在      `JUDGES` 里的文件必须在 `src/` 下找得到；
    T2 依赖      模块级 import 决定最小档：torch/transformers -> 4，numpy -> 3，其它 -> 2。
                 **只算模块级**（函数里的 import 是懒加载，不挡门），并且对同目录的
                 `by1*` 模块**再跟一层**（大多数依赖是这么进来的）；
    T3 范围      档位必须是 2..6 的整数；
    T4 命令名    `JUDGE_CMDS` 的键必须是 `<文件> <子命令>`，且那个 `<文件>` 必须在 `JUDGES` 里；
    T5 全覆盖    `src/modelcheck/*.py` 每个都要么被 `JUDGES` 跑，要么在 `EXEMPT` 里**写明理由**。
    T6 参照物    判卷人的 docstring 里必须有一行 `参照物：...`，
                 且它里面的至少一个 **CamelCase 名字**要真的出现在那个文件里（防“文档写的参照物早就不存在了”）。
                 不是参照对比类的记在 `NO_REF` 里。
    T7 上游      上面那行里点名的官方名字，要能在**本机装的 transformers 里找到**（没装就跳过）。
                 “文档写着、上游已删”与“名字拼错”都在这一条上现形。
                 （modelcheck 是"一个模型一个判卷人"，漏一个就是漏一整条判据。）

## 自检

`--selftest` 用**临时目录里的一套假树 + 假表**把五条规则各弄红一次，再跑一遍真树。
判据必须两边都验：只验真树的话，"规则从没生效"和"规则全过"长得一样。
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块可能在包内深处（src/by1/checks/…），固定层数会插错目录。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.exists(_os.path.join(_p, 'by1paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import ast
import pathlib
import re
import sys

import by1paths

# 迁移期（docs/TEMP-rename-plan.md）：基准改问 by1paths（包内层数会变）。
ROOT = pathlib.Path(by1paths.ROOT)

#: 模块级 import -> 最小档位。只列"会挡门"的那几个；其余按 2 算。
MIN_TIER = {'torch': 4, 'transformers': 4, 'numpy': 3}

#: `src/modelcheck/` 里**不是判卷人**的东西，逐条写理由（T5 要它）。
#: 不是“对着官方实现比”的判卷人：它们自带自测。
#: 参照物不在 transformers 里的判卷人（T7 要跳过它们）。
UPSTREAM_EXEMPT = {
    'by1instella.py': '参照物是 AMD 官方仓库里的 modeling_instella.py，不在 transformers 里',
}

_UPSTREAM = {}

def upstream_names():
    """本机 transformers 里出现过的标识符集合。没装 -> None（T7 跳过）。cache 一次。"""
    if 'names' in _UPSTREAM:
        return _UPSTREAM['names']
    try:
        import transformers
    except Exception:
        _UPSTREAM['names'] = None
        return None
    import pathlib as _pl
    root = _pl.Path(transformers.__file__).parent
    names = set()
    for f in root.rglob('*.py'):
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        names.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]{4,}', text))
    _UPSTREAM['names'] = names
    return names
NO_REF = {
    'by1dev.py': '设备无关性：验的是前向不依赖 CPU 默认，没有官方对照物',
}

EXEMPT = {
    'by1load.py': '库：量化权重的装载路径，由 by1gpubench / by1full 调',
    'by1mem.py': '库：显存估算，由 GPU 日的脚本调',
    'by1real.py': '库：真权重检查的公共部分，由 GPU 日脚本调',
    'by1shadow.py': '档 6：影子模式的生成器（by1gpuday / by1gpubench 调）',
    'by1train.py': '档 6：训练步（真权重 + autograd），本机没有 GPU 轮次',
    'by1quant.py': '库：它的 selftest 由 by1fast 在档 1/3 里调（不单列）',
    'by1q2schema.py': '库：Q2 报告 schema，自检由 by1fast 调',
}


def load_tables(path):
    """从 `by1all.py` **静态**取出 JUDGES / JUDGE_CMDS。

    不 import 它：那会把整棵依赖树拖进来（而本检查器跑在档 2，连 numpy 都不该要）。
    """
    tree = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('JUDGES', 'JUDGE_CMDS'):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def top_imports(path):
    """**模块级** import 的顶层名。函数里的不算（懒加载不挡门）。"""
    tree = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split('.')[0])
    return names


def find_file(root, name):
    """判卷人脚本可能在 `src/checks` / `src/modelcheck` / `src/escape` / `src/data`。

    迁移期（docs/TEMP-rename-plan.md）：`by1<X>.py` 先问包里真身
    `src/by1/**/<X>.py` —— 旧路径只剩 shim，**档位要从真身的 import 算**。
    """
    src = pathlib.Path(root) / 'src'
    if name.startswith('by1') and name.endswith('.py') and len(name) > 7:
        pkg = src / 'by1'
        if pkg.is_dir():
            hits = [p for p in sorted(pkg.rglob(name[3:]))
                    if p.is_file() and p.suffix == '.py']
            if hits:
                return hits[0]
    for cand in sorted(src.rglob(name)):
        if cand.is_file():
            return cand
    return None


def min_tier_of(root, path, _seen=None):
    """这个脚本最小的档位。**跟一层**同目录的 `by1*` 模块。"""
    _seen = set() if _seen is None else _seen
    path = pathlib.Path(path)
    if path in _seen:
        return 2
    _seen.add(path)
    need = 2
    for name in top_imports(path):
        if name in MIN_TIER:
            need = max(need, MIN_TIER[name])
            continue
        if not name.startswith('by1'):
            continue
        sib = path.parent / (name + '.py')
        if not sib.is_file():
            sib = find_file(root, name + '.py')
        if sib is not None and sib.is_file():
            need = max(need, min_tier_of(root, sib, _seen))
    return need


def check(root=ROOT, tiers_path=None, modelcheck_dir=None, resolve=None):
    """返回问题列表（空 = 过）。`root`/路径可换 —— 自检就靠这个。"""
    root = pathlib.Path(root)
    tiers_path = pathlib.Path(tiers_path or by1paths.tool('by1all.py'))
    mc = pathlib.Path(modelcheck_dir or (root / 'src/modelcheck'))
    problems = []
    # T7 的上游名字集：默认扫本机的 transformers（缓存一次）；没装 -> None -> 跳过
    upstream = upstream_names() if resolve is None else (resolve or None)
    tb = load_tables(tiers_path)
    judges = tb.get('JUDGES') or []
    cmds = tb.get('JUDGE_CMDS') or {}
    if not judges:
        return ['JUDGES 表是空的（或取不出来）—— 这条检查会永远绿，那比红更糟']
    names = set()
    for item in judges:
        if not (isinstance(item, tuple) and len(item) == 2):
            problems.append('T3 JUDGES 的项必须是 (文件名, 档位)：%r' % (item,))
            continue
        name, tier = item
        names.add(str(name).split()[0])
        if not isinstance(tier, int) or not 2 <= tier <= 6:
            problems.append('T3 %s 的档位 %r 不在 2..6' % (name, tier))
            continue
        path = find_file(root, str(name).split()[0])
        if path is None:
            problems.append('T1 %s 在 src/ 下找不到' % name)
            continue
        need = min_tier_of(root, path)
        if tier < need:
            problems.append(
                'T2 %s 声明档 %d，但它的模块级依赖要求至少档 %d'
                % (name, tier, need))
    for key, argv in cmds.items():
        parts = str(key).split()
        if len(parts) != 2:
            problems.append('T4 JUDGE_CMDS 的键应是 "<文件> <子命令>"：%r' % (key,))
            continue
        if parts[0] not in names:
            problems.append('T4 %s 的命令 %r 没在 JUDGES 里登记（它永远不会跑）'
                            % (parts[0], key))
        if not (isinstance(argv, list) and argv and argv[0] == parts[0]):
            problems.append('T4 %s 的 argv %r 与命令名对不上' % (key, argv))
    # T6：参照对比类判卷人要写出参照物，且名字真在文件里。
    for name in sorted(names):
        base = str(name).split()[0]
        if base in NO_REF:
            continue
        path = find_file(root, base)
        if path is None or pathlib.Path(path).parent.name != 'modelcheck':
            continue        # T1 已报；非 modelcheck 的不强求
        text = pathlib.Path(path).read_text(encoding='utf-8', errors='replace')
        try:
            doc = ast.get_docstring(ast.parse(text)) or ''
        except SyntaxError:
            problems.append('T6 %s 语法错，读不到 docstring' % base)
            continue
        line = next((ln.strip() for ln in doc.splitlines()
                     if ln.strip().startswith('参照物')), None)
        if line is None:
            problems.append('T6 %s 的 docstring 里没有 <参照物：...> —— 判卷人得说清它在对着谁比' % base)
            continue
        caps = re.findall(r'[A-Z][A-Za-z0-9]{4,}', line)
        # 也接受 `某某.py` 这种文件名形式的参照物（by1instella 就是这种）
        caps += re.findall(r'[A-Za-z_][A-Za-z0-9_]*\.py', line)
        # **必须在 docstring 之外**——否则“参照物那行自己”就能通过
        code_only = text.replace(doc, '')
        if not caps or not any(c in code_only for c in caps):
            problems.append('T6 %s 的参照物名字 %r 在文件里找不到（文档过期）' % (base, caps[:3]))
            continue
        # T7：这些名字要能**在本机装的 transformers 里找到**（没装就不查）
        if upstream is None or base in UPSTREAM_EXEMPT:
            continue
        if not any(c in upstream for c in caps):
            problems.append('T7 %s 的参照物 %r 在本机 transformers 里一个都找不到'
                            '（上游改过名？名字拼错？）' % (base, caps[:3]))
    if mc.is_dir():
        for f in sorted(mc.glob('*.py')):
            if f.name == '__init__.py' or f.name in names:
                continue
            if f.name not in EXEMPT:
                problems.append('T5 %s 既没被 JUDGES 跑，也没在 EXEMPT 里写明理由'
                                % f.name)
    return problems


def _expect7(d, tag, table_text, want, known):
    p = d / 'src/checks/by1all.py'
    # 自检里用一行字符串写表更省事，这里把字面 \n 当真换行处理
    table_text = table_text.replace(chr(92) + 'n', chr(10))
    p.write_text(table_text, encoding='utf-8')
    probs = check(root=d, tiers_path=p,
                  modelcheck_dir=d / 'src/modelcheck', resolve=known)
    return [x for x in probs if want in x]

def _selftest():
    """假树 + 假表：七条规则各弄红一次，再验真树。"""
    import shutil
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix='by1tiers-'))
    bad = []
    try:
        (d / 'src/checks').mkdir(parents=True)
        (d / 'src/modelcheck').mkdir(parents=True)
        (d / 'src/checks/by1plain.py').write_text('import os\n', encoding='utf-8')
        (d / 'src/checks/by1np.py').write_text('import numpy\n', encoding='utf-8')
        (d / 'src/checks/by1torchy.py').write_text('import torch\n',
                                                   encoding='utf-8')
        (d / 'src/checks/by1lazy.py').write_text('def main():\n    import torch\n',
                                                 encoding='utf-8')
        (d / 'src/checks/by1helper.py').write_text('import numpy\n',
                                                   encoding='utf-8')
        (d / 'src/modelcheck/by1judge.py').write_text(
            '"""by1judge -- fake.\n参照物：官方 FakeJudge\n"""\n'
            'import os  # FakeJudge\n',
            encoding='utf-8')

        def table(text):
            p = d / 'src/checks/by1all.py'
            p.write_text(text, encoding='utf-8')
            return p

        def expect(tag, text, want):
            p = table(text)
            probs = check(root=d, tiers_path=p, resolve=frozenset(),
                          modelcheck_dir=d / 'src/modelcheck')
            if not [x for x in probs if want in x]:
                bad.append('%s 没被抓住（问题表：%r）' % (tag, probs))

        good = ("JUDGES = [('by1plain.py', 2), ('by1np.py', 3),"
                " ('by1torchy.py', 4), ('by1lazy.py', 2),"
                " ('by1judge.py', 2)]\n"
                "JUDGE_CMDS = {'by1torchy.py sub': ['by1torchy.py', 'sub']}\n")
        p = table(good)
        probs = check(root=d, tiers_path=p, modelcheck_dir=d / 'src/modelcheck',
                      resolve=frozenset())
        if probs:
            bad.append('正例竟然红了：%r' % (probs,))
        expect('T1 找不到文件',
               "JUDGES = [('by1nothere.py', 2)]\nJUDGE_CMDS = {}\n", 'T1')
        expect('T2 numpy 挂在档 2',
               "JUDGES = [('by1np.py', 2)]\nJUDGE_CMDS = {}\n", 'T2')
        expect('T2 torch 挂在档 3',
               "JUDGES = [('by1torchy.py', 3)]\nJUDGE_CMDS = {}\n", 'T2')
        (d / 'src/checks/by1useshelper.py').write_text('import by1helper\n',
                                                       encoding='utf-8')
        expect('T2 间接（跟一层）',
               "JUDGES = [('by1useshelper.py', 2)]\nJUDGE_CMDS = {}\n", 'T2')
        expect('T3 档位越界',
               "JUDGES = [('by1plain.py', 9)]\nJUDGE_CMDS = {}\n", 'T3')
        expect('T4 命令没登记',
               "JUDGES = [('by1plain.py', 2)]\n"
               "JUDGE_CMDS = {'by1ghost.py sub': ['by1ghost.py', 'sub']}\n",
               'T4')
        (d / 'src/modelcheck/by1orphan.py').write_text('import os',
                                                       encoding='utf-8')
        (d / 'src/modelcheck/by1noref.py').write_text('import os\n',
                                                       encoding='utf-8')
        expect('T6 没写参照物',
               "JUDGES = [('by1noref.py', 2), ('by1judge.py', 2)]\n"
               "JUDGE_CMDS = {}\n", 'T6')
        (d / 'src/modelcheck/by1stale.py').write_text(
            '"""by1stale -- fake.\n参照物：官方 NoSuchClassAtAll\n"""\nimport os\n',
            encoding='utf-8')
        expect('T6 参照物名字过期',
               "JUDGES = [('by1stale.py', 2), ('by1judge.py', 2)]\n"
               "JUDGE_CMDS = {}\n", 'T6')
        t7_tab = ("JUDGES = [('by1stale.py', 2), ('by1judge.py', 2)]\n"
                  "JUDGE_CMDS = {}\n")
        # 反例：参照物名字在本机上游找不到 -> T7 必须红
        if not _expect7(d, 'T7 上游找不到', t7_tab, 'T7',
                        frozenset({'SomethingElse'})):
            bad.append('T7 反例没红（上游没有那个名字却放行了）')
        # 正例：名字在上游里 -> 不许报 T7
        t7_pos = "JUDGES = [('by1stale.py', 2)]\nJUDGE_CMDS = {}\n"
        if _expect7(d, 'T7 正例', t7_pos, 'T7',
                    frozenset({'NoSuchClassAtAll'})):
            bad.append('T7 正例被误判（名字明明在已知集合里）')
        expect('T5 modelcheck 漏了',
               "JUDGES = [('by1plain.py', 2), ('by1judge.py', 2)]\n"
               "JUDGE_CMDS = {}\n", 'T5')
    finally:
        shutil.rmtree(d, ignore_errors=True)
    real = check()
    for x in real:
        bad.append('真树：' + x)
    for x in bad:
        print('     ! ' + x)
    print('  [%s] 档位归属：七条规则各红一次，真树 %s'
          % ('PASS' if not bad else 'FAIL',
             '也过' if not real else '有 %d 处问题' % len(real)))
    return 0 if not bad else 1


def _count():
    tb = load_tables(by1paths.tool('by1all.py'))
    return len(tb.get('JUDGES') or [])


def main(argv):
    if '--selftest' in argv:
        return _selftest()
    problems = check()
    for x in problems:
        print('     ! ' + x)
    ok = not problems
    print('  [%s] 档位归属 %s'
          % ('PASS' if ok else 'FAIL',
             '七条规则都成立（%d 个判卷人）' % _count()
             if ok else '%d 处不成立' % len(problems)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
