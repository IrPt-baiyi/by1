#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1pack -- 打出要上传到云端的那一包。

**只装"描述 + 判卷人 + 工具"，不装权重和可重新下载的东西。**

    不装：模型权重（在 GPU 机器上从 HF 下）· llamacpp/ 里的参考源码
          drafts/（没有判卷人的草稿）· cgen/ __pycache__

用法:  python src/by1pack.py [输出目录]
       默认输出 ../by1-upload/

## 搬家之后

包里的目录结构和仓库**一样**（`src/` `models/` `refs/` + 根下的文档）。
原来是把所有东西摊在包的根目录，而 gpu/ 那一层又是平铺进来的 ——
于是打包完印出来的那句 `bash gpu/run.sh` 指的路径**根本不存在**。
现在包里长什么样、仓库里就长什么样。
"""
import glob
import os
import shutil
import sys
import tarfile
import time

import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = by1paths.ROOT
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(ROOT), 'by1-upload')

# 要装的东西。每条是 (来源, 说明) —— **来源写裸名**，"它在哪个目录"
# 由 `_where()` 一处决定。原来这里把 30 个脚本名和 20 个文档名写在一起、
# 还各自带目录前缀，搬一次家就要改 50 行。
WANT = [
    ('by1.py', '入口'),
    ('by1check.py', '不变式检查器（解析 + 契约 + 等价类）'),
    ('by1codegen.py', '后端一：PyTorch nn.Module'),
    ('by1exec.py', '后端二：纯 NumPy'),
    ('by1c.py', '后端三：生成 C，编译，运行'),
    ('by1verify.py', '产物对拍：config / 张量名与形状'),
    ('by1diff.py', '对 transformers 的前向与反向'),
    ('by1mem.py', '内存计划'),
    ('by1emit.py', 'ggml 建图与 GGUF manifest'),
    ('by1oracle.py', '状态尺寸神谕'),
    ('by1train.py', '训练脚手架'),
    ('by1mla.py', '判卷人：MLA'),
    ('by1moe.py', '判卷人：noaux_tc 分组路由'),
    ('by1rope.py', '判卷人：llama3 RoPE'),
    ('by1all.py', '一次跑完全部验证'),
    ('by1instella.py', '真实模型前向：Instella-3B，**真实维度**'),
    ('by1pack.py', '打这一包'),
    ('by1contract.py', '从机制声明推出张量契约'),
    ('by1boot.py', '从产物反推一份 .by1 草稿'),
    ('by1gate.py', '取值门的可证伪对照'),
    ('by1raw.py', '逃生舱的判卷人'),
    ('by1ir.py', 'IR 规格 + 校验 + JSON 往返（**规格的唯一真相源**）'),
    ('by1irentry.py', '三个后端只从 IR 跑'),
    ('by1bootir.py', '两条路的 IR 对拍（.by1 vs 产物）'),
    ('by1ext.py', '逃生舱第二层：外部符号的加载与 ABI'),
    ('by1extdemo.py', '逃生舱第二层的判卷人'),
    ('by1opdiff.py', '逐算子比 NumPy 和 PyTorch'),
    ('by1e2e.py', '**端到端**：真产物 -> IR -> 三个后端 -> 对官方实现'),
    ('by1dev.py', '设备无关性（显卡上能跑的必要条件）'),
    # **`by1lint.py` 是 `by1all` 的判卷人之一（两条：本体 + 自检）。**
    # 它进来的时候我漏了这一行 —— 而 `closure_missing()` **看不见**它：
    # 那个检查算的是 import 闭包，而 `by1lint` 是被**当脚本调用**的，
    # 没人 import 它。**一个判卷人不在包里 = 那一项在云端变成"文件不存在"**，
    # 而症状长得像"这一包本身有问题"（gpu/run.sh 第一行就会停）。
    ('by1lint.py', '十类"不出声"的写法 + 自检（by1all 的判卷人）'),
    # **这四个也是 `by1all` 当脚本调的，而清单里一直没有。**
    # 加 `judges_missing()` 那一条检查时它一次报出四个 ——
    # 也就是说"上一包"在云端跑到这几项时会报"文件不存在"，
    # 而那个症状是"这一包本身有问题"（gpu/run.sh 第一行就停）。
    ('by1pat.py', '模式分类：说已验证的都有命中，说未验证的都是 0'),
    ('by1oracles.py', '十个神谕：不依赖 by1 的期望值'),
    ('by1docs.py', '文档里的路径**和数字**'),
    ('by1gpt2.py', '真实维度前向：GPT-2（另一代）'),
    # **下面三个是"文档叫你跑，而包里没有"的那一类。**
    # `by1docs.py` 在包里自己把它们抓出来了 —— 它扫 README 里
    # `python xxx.py` 的调用，而包里没有这些文件。
    # 也就是说：**新读者照 README 敲的第一条命令就会 FileNotFoundError**，
    # 而那个报错长得像"这一包坏了"。
    #
    # 这三条是"打包清单"和"文档"之间的缝 —— 两边都没有引用的地方，
    # 所以谁也没发现。判据：文档里提到的脚本，包里有；反之亦然的那些
    # 由 `judges_missing()` 守。
    ('by1fetch.py', 'refs/ 抓取（README 的"数据从哪来"第一条命令）'),
    ('by1gpu.py', '显卡/显存规划（GPU runbook §3 的第一条命令）'),
    ('by1real.py', '真权重真维度前向（GPU runbook §3 的最后一条命令）'),
    ('by1refs.py', 'refs/ 的路径规则'),
    ('by1io.py', '文件读写的每一种意图'),
    # **`by1skip.py` 原来不在这个清单里，而 8 个已打包的模块在模块级
    # `import by1skip`** —— 于是包打出来了、`gpu/run.sh` 第一行
    # （`src/by1all.py --quick`）就 `ModuleNotFoundError: No module named
    # 'by1skip'`。**这个包从来没能跑起来过**，而打包脚本一声不吭。
    # 下面的 `closure_missing()` 就是这条的判卷人：以后少一个就红。
    ('by1skip.py', '三种结论（ok / 跳过 / 失败）的协议'),
    ('by1paths.py', '仓库布局的唯一真相源'),
    ('by1ver.py', '版本号的唯一真相源'),
    ('by1name.py', '改名：models.tsv <-> 磁盘'),
    ('gpt2-tiny.by1', '上一代的缩小版（无门控 FFN + LayerNorm + 查表位置）'),
    ('ext-demo.c', '示例外部机制：两个符号，证「不用改编译器」'),
    ('raw.py', '逃生舱的示例实现（配 raw-escape.by1）'),
    ('gate-probe.by1', '取值门的反例（故意写错，别修）'),
    ('ir-spec.md', 'IR 规格（从 src/by1ir.py 生成）'),
    ('README.md', '门面'),
    ('pyproject.toml', '装得上'),
    ('VERSION', '版本的投影（打包读它，不导入模块）'),
    ('1.md', '项目说明（宣言）'),
    ('history/ir.md', '开发日志'),
    ('history/plan.md', '2026-10-08 的计划'),
    ('docs/INVENTORY.txt', '验证清单：每个模型验到了什么'),
    ('.gitignore', ''),
    # **`.gitattributes` 也要进包。** 它是那份"文本一律 LF"的规则 ——
    # 拿到包的人在**那棵树里**继续改动时，规则要跟着走；
    # 不然他本地一提交，行尾又乱回去（原来就没这个文件，见它的注释）。
    ('.gitattributes', '行尾规范：文本一律 LF'),
]

# refs/ 全要（官方产物，判卷人的原材料）
# models/*.by1 全要（不含 drafts/）


def _where(name):
    """清单里的**裸名** -> 仓库根相对的来源路径。

    **规则只写这一处。** `.by1`（还有 `raw.py`）在 `models/`，
    代码在 `src/`，其余按写的来。
    """
    if name.endswith('.by1') or name == 'raw.py':
        return 'models/' + name
    if name.endswith(('.py', '.c')):
        return 'src/' + name
    return name


def disk_modules():
    """`src/` 下所有模块的裸名。"""
    import glob as _g
    return {os.path.basename(p)
            for p in _g.glob(norm('src/*.py'))}


def closure_missing():
    """**WANT 的传递依赖是不是都装进去了。**

    这是"包打出来却起不来"那一类 bug 的判卷人。

    在这之前，`WANT` 是一张手写清单，少写一个模块的后果是：
    包照常打出来、体积照常报、`gpu/run.sh` 第一行就 `ModuleNotFoundError`
    —— 而**打包脚本什么都不说**（`if not os.path.exists(s): continue`）。
    一个不会失败的检查不是检查；一个不会报错的清单不是清单。

    判据：把 WANT 里每个 `.py` 的**模块级 import**（AST 取，不用正则）
    连成闭包；闭包里凡是在 `src/` 下、却没进 WANT 的，都是缺口。
    """
    import ast
    disk = disk_modules()
    want = {w[0] for w in WANT if w[0].endswith('.py')} & disk
    graph = {}
    for f in sorted(want):
        try:
            tree = ast.parse(by1io.read_text(norm('src/' + f), encoding='utf-8'))
        except SyntaxError:
            graph[f] = set()
            continue
        deps = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or '']
            else:
                continue
            for m in mods:
                base = m.split('.')[0] + '.py'
                if base in disk:
                    deps.add(base)
        graph[f] = deps
    need, frontier = set(want), set(want)
    while frontier:
        nxt = set()
        for f in frontier:
            for d in graph.get(f, ()):
                if d not in need:
                    need.add(d)
                    nxt.add(d)
        frontier = nxt
    return sorted(need - want)


def judges_missing():
    """`by1all` **当脚本调用**的那些 `.py`，有没有漏出这个包。

    ## 为什么 import 闭包不够

    `closure_missing()` 算的是"谁 import 谁" —— 而护栏里有一半的东西
    是**被 subprocess 调用**的（`by1all.run(['by1lint.py'])`），
    没有任何模块 import 它们。于是：

        `by1lint.py` 成了 `by1all` 的判卷人，而它没进 `WANT`
        -> `closure_missing()` **一声不吭**
        -> 到了云端那一项变成"文件不存在"，
           而症状长得像"这一包本身有问题"

    这就是"两个清单各说各话"的老毛病。所以这里**从 `by1all` 自己读**：
    `JUDGES` 加那几个 `run(['xxx.py'])` 的调用点。
    它是 `by1all` 的一部分，改了这边就跟着改。
    """
    import re as _re
    p = norm('src/by1all.py')
    if not os.path.exists(p):
        return []
    src = by1io.read_text(p, encoding='utf-8')
    names = set()
    # `JUDGES = [...]` 里的裸名
    m = _re.search(r'^JUDGES\s*=\s*\[(.*?)\]', src, _re.S | _re.M)
    if m:
        names |= set(_re.findall(r"'([\w.]+\.py)'", m.group(1)))
    # `run(['xxx.py', ...])` 这类调用点
    names |= set(_re.findall(r"run\(\[\s*'([\w.]+\.py)'", src))
    want = {w[0] for w in WANT}
    return sorted(n for n in names if n not in want and n in disk_modules())


def norm(p):
    return os.path.join(ROOT, p)


def main():
    # **先验清单，再动磁盘。** 校验失败时不该留下一个半成品目录
    # （调用方看到的应该是"没产出"，不是"产出了一份坏的"）。
    absent = [_where(name) for name, _why in WANT
              if not os.path.exists(norm(_where(name)))]
    gaps = closure_missing()
    judges = judges_missing()
    if absent or gaps or judges:
        print('=' * 70)
        print('  **打包失败 —— 这一包跑不起来**')
        print('=' * 70)
        if absent:
            print('  清单里写了、磁盘上没有（%d 个）：' % len(absent))
            for r in absent:
                print('    · %s' % r)
        if gaps:
            print('  清单漏了传递依赖（%d 个）—— 装进包里才能 import：'
                  % len(gaps))
            for g in gaps:
                print('    · %s' % g)
        if judges:
            print('  清单漏了 `by1all` 会调用的判卷人（%d 个）——'
                  ' 它们不是被 import 的，所以闭包查不到：' % len(judges))
            for j in judges:
                print('    · %s' % j)
        print()
        print('  清单在 %s 的 `WANT` 里。'
              % os.path.relpath(__file__, ROOT))
        return 1

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    n, total = 0, 0

    def put(src_abs, rel):
        """拷进包里，**保持仓库里的相对路径**。"""
        nonlocal n, total
        d = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src_abs, d)
        n += 1
        total += os.path.getsize(d)

    # 工具与文档（上面已经验过"清单里的东西都在"，所以这里不会静默跳过）
    for name, _why in WANT:
        rel = _where(name)
        put(norm(rel), rel)

    # models/ 的 .by1 描述
    for s in sorted(glob.glob(norm('models/*.by1'))):
        put(s, 'models/' + os.path.basename(s))

    # refs/（官方产物）
    for s in sorted(glob.glob(norm('refs/*'))):
        if os.path.isfile(s):
            put(s, 'refs/' + os.path.basename(s))

    # gpu/（云端专属：runbook + 依赖 + 需要 CUDA 的检查）
    gsrc = norm('gpu')
    if os.path.isdir(gsrc):
        for f in sorted(os.listdir(gsrc)):
            s = os.path.join(gsrc, f)
            if os.path.isfile(s):
                put(s, 'gpu/' + f)

    # 打成一个 tar.gz（保留目录结构）
    stamp = time.strftime('%Y%m%d')
    tgz = os.path.join(os.path.dirname(OUT), 'by1-upload-%s.tar.gz' % stamp)
    with tarfile.open(tgz, 'w:gz') as tf:
        for base, _dirs, files in os.walk(OUT):
            for f in files:
                full = os.path.join(base, f)
                tf.add(full, os.path.relpath(full, os.path.dirname(OUT)))
    tsize = os.path.getsize(tgz)

    print('=' * 70)
    print('  打包完成')
    print('=' * 70)
    print('  目录   %s' % OUT)
    print('  压缩包 %s  (%.2f MB)' % (tgz, tsize / 1048576))
    print('  %d 个文件，原始 %.2f MB' % (n, total / 1048576))
    print()
    print('  **没有装**（在云端重新拿）：')
    print('    · 模型权重 —— 从 HF 下，下多少取决于跑哪个模型')
    print('    · llamacpp/ 里的参考源码 —— 可重新下载')
    print('    · drafts/ —— 没有判卷人的草稿')
    print('    · cgen/ __pycache__ —— 生成物')
    print()
    print('  包里的结构和仓库一样（src/ models/ refs/ + 根下的文档）。')
    print('  上传后：  tar xzf %s && cd by1-upload && bash gpu/run.sh'
          % os.path.basename(tgz))
    return 0


if __name__ == '__main__':
    sys.exit(main())
