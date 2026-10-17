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
                 （modelcheck 是"一个模型一个判卷人"，漏一个就是漏一整条判据。）

## 自检

`--selftest` 用**临时目录里的一套假树 + 假表**把五条规则各弄红一次，再跑一遍真树。
判据必须两边都验：只验真树的话，"规则从没生效"和"规则全过"长得一样。
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: 模块级 import -> 最小档位。只列"会挡门"的那几个；其余按 2 算。
MIN_TIER = {'torch': 4, 'transformers': 4, 'numpy': 3}

#: `src/modelcheck/` 里**不是判卷人**的东西，逐条写理由（T5 要它）。
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
    """判卷人脚本可能在 `src/checks` / `src/modelcheck` / `src/escape` / `src/data`。"""
    src = pathlib.Path(root) / 'src'
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


def check(root=ROOT, tiers_path=None, modelcheck_dir=None):
    """返回问题列表（空 = 过）。`root`/路径可换 —— 自检就靠这个。"""
    root = pathlib.Path(root)
    tiers_path = pathlib.Path(tiers_path or (root / 'src/checks/by1all.py'))
    mc = pathlib.Path(modelcheck_dir or (root / 'src/modelcheck'))
    problems = []
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
    if mc.is_dir():
        for f in sorted(mc.glob('*.py')):
            if f.name == '__init__.py' or f.name in names:
                continue
            if f.name not in EXEMPT:
                problems.append('T5 %s 既没被 JUDGES 跑，也没在 EXEMPT 里写明理由'
                                % f.name)
    return problems


def _selftest():
    """假树 + 假表：五条规则各弄红一次，再验真树。"""
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
        (d / 'src/modelcheck/by1judge.py').write_text('import os\n',
                                                      encoding='utf-8')

        def table(text):
            p = d / 'src/checks/by1all.py'
            p.write_text(text, encoding='utf-8')
            return p

        def expect(tag, text, want):
            p = table(text)
            probs = check(root=d, tiers_path=p,
                          modelcheck_dir=d / 'src/modelcheck')
            if not [x for x in probs if want in x]:
                bad.append('%s 没被抓住（问题表：%r）' % (tag, probs))

        good = ("JUDGES = [('by1plain.py', 2), ('by1np.py', 3),"
                " ('by1torchy.py', 4), ('by1lazy.py', 2),"
                " ('by1judge.py', 2)]\n"
                "JUDGE_CMDS = {'by1torchy.py sub': ['by1torchy.py', 'sub']}\n")
        p = table(good)
        probs = check(root=d, tiers_path=p, modelcheck_dir=d / 'src/modelcheck')
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
    print('  [%s] 档位归属：五条规则各红一次，真树 %s'
          % ('PASS' if not bad else 'FAIL',
             '也过' if not real else '有 %d 处问题' % len(real)))
    return 0 if not bad else 1


def _count():
    tb = load_tables(ROOT / 'src/checks/by1all.py')
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
             '五条规则都成立（%d 个判卷人）' % _count()
             if ok else '%d 处不成立' % len(problems)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
