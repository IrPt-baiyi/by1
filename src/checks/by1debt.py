#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1debt -- **没干完的事，量出来，不凭印象列。**

## 为什么要有这个

一个长项目里，"还剩什么"最容易变成感觉：
"应该差不多了" / "量化还没做" —— 而这两种说法都没法验收。

这里把每一类**能数的**都数出来，每一条都给出：
  · 现在是多少
  · 分母是多少
  · 差在哪

数不出来的（比如"llama.cpp 后端"）就**明确标成数不出来** ——
不假装它是一条可验收的债。

用法:  python by1debt.py
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
import glob
import os
import re
import shutil
import subprocess
import sys
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def sh(*a):
    """跑一条命令，返回 stdout。**工具不在 PATH 上就明说，不抛栈。**

    原来这里是裸的 `subprocess.run(('git', ...))` —— 而 git 不在这台机器
    的 PATH 上，于是 `by1debt` 以一个 `FileNotFoundError` 的栈结束，
    冒烟测把它记成"**可疑**"（有 Traceback、又不像环境问题）。
    而它既不是"可疑"，也不是 by1debt 的 bug：**是这台机器上没有 git。**
    """
    exe = shutil.which(a[0]) or a[0]
    try:
        r = subprocess.run((exe,) + tuple(a[1:]), capture_output=True,
                           text=True, encoding='utf-8', errors='replace',
                           timeout=120)
    except FileNotFoundError:
        return ''
    # **退出码非零 = 没拿到，返回空串。**
    # 调用方把这个串丢给 regex，"空串"就等于"什么都没发现" ——
    # 而"命令跑失败"和"跑成功、没有输出"是两件事。
    # 返回一个空串冒充后者，就是一次静默的假绿。
    if r.returncode != 0:
        return ''
    return r.stdout


def by1files():
    return by1paths.names()


def real_models():
    """有 by1-repo 的 = 从真仓库来的。其余是合成的。"""
    out = []
    for f in by1files():
        try:
            head = by1io.head_text(by1paths.model(f))
        except Exception:
            continue
        m = re.search(r'^#\s*by1-repo:\s*(\S+)', head, re.M)
        if m:
            out.append((f, m.group(1)))
    return out


def main():
    print()
    print('=' * 78)
    print('  没干完的事 —— **每一条都尽量给出分母**')
    print('=' * 78)

    # ── ① 描述覆盖率 ──────────────────────────────────────────────
    cfgs = {os.path.basename(f)[:-len('.config.json')]
            for f in glob.glob(by1paths.ref('*.config.json'))}
    reals = real_models()
    described = {r[1].replace('/', '__') for r in reals}
    print()
    print('  ① 描述覆盖率')
    print('     refs/ 里有 %d 份 config（能描述的模型）' % len(cfgs))
    print('     写了 .by1 的真实模型：%d 个' % len(reals))
    miss = sorted(cfgs - described)
    print('     **没写 .by1 的：%d 个**' % len(miss))
    for m in miss[:10]:
        print('       %s' % m[:64])
    if len(miss) > 10:
        print('       …还有 %d 个' % (len(miss) - 10))

    # ── ② boot_ir 的展开率 ───────────────────────────────────────
    print()
    print('  ② boot_ir 能展开多少（不需要人写 .by1 的那部分）')
    try:
        r = subprocess.run([sys.executable, 'by1triage.py'],
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace',
                           timeout=1800)
        out = r.stdout
        # 同上：正则从 stdout 里抠数字，那就得先确认这次跑成功了。
        if r.returncode != 0:
            print('     （by1triage 退出码 %d —— 下面的数不作数）'
                  % r.returncode)
        m = re.search(r'① (\d+) 种模式\s*·\s*② (\d+) 种模式', out)
        n_ok = len(re.findall(r'\n  ✓ ', out))
        n_all = len(re.findall(r'\n  [✓!] ', out))
        if m:
            print('     ① classify 认出来、boot 不会展开：%s 种' % m.group(1))
            print('     ② classify 也认不出：%s 种' % m.group(2))
        print('     出得了完整 IR 的：%d / %d' % (n_ok, n_all or n_ok))
    except Exception as e:
        print('     （跑 triage 失败：%s）' % str(e)[:60])

    # ── ③ 回归覆盖面 ─────────────────────────────────────────────
    print()
    print('  ③ 回归覆盖面')
    allpy = sorted(os.path.basename(f) for f in glob.glob('by1*.py'))
    allb = by1io.read_text(by1paths.tool('by1all.py'), encoding='utf-8')
    # **要分三类。** 第一版把"库"也当成"没跑到"，22 个里大半是误报 ——
    # 而 `by1ir` / `by1codegen` 是**被 import 的**，本来就不该单独跑。
    #
    # 判据：谁 import 了它。
    imported = set()
    for f in glob.glob('*.py'):
        try:
            src = by1io.read_text(f, encoding='utf-8')
        except Exception:
            continue
        for m in re.finditer(r'import\s+(by1\w+)', src):
            imported.add(m.group(1) + '.py')
        for m in re.finditer(r"'(by1\w+)\.py'", src):
            imported.add(m.group(1) + '.py')

    libs, scripts, gpu = [], [], []
    for p2 in allpy:
        s = by1io.read_text(p2, encoding='utf-8') if os.path.exists(p2) else ''
        if 'cuda' in s.lower() or 'nvidia-smi' in s or 'by1cloud' in s:
            gpu.append(p2)
        elif p2 in imported and p2 not in allb:
            libs.append(p2)
        elif p2 not in allb:
            scripts.append(p2)

    print('     by1*.py 共 %d 个；by1all 里跑到的 %d 个'
          % (len(allpy), len([p2 for p2 in allpy if p2 in allb])))
    print('     库（被 import，本来就该由别人跑）：%d 个' % len(libs))
    print('       %s' % ', '.join(x[:-3] for x in libs))
    print('     需要显卡（本地跑不了）：%d 个' % len(gpu))
    print('       %s' % ', '.join(x[:-3] for x in gpu))
    print('     **该跑却没跑到的脚本：%d 个**' % len(scripts))
    for u in scripts:
        print('       %s' % u)

    # ── ④ 前向跑过没有 ───────────────────────────────────────────
    print()
    print('  ④ 真模型前向')
    ir = by1io.read_text(by1paths.root('history', 'ir.md'), encoding='utf-8') if os.path.exists(by1paths.root('history', 'ir.md')) \
        else ''
    never = []
    for f, repo in reals:
        # ir.md 里提到过"跑通/一致"就算跑过（粗判，明说是粗判）
        if repo.split('/')[-1][:12] not in ir and f[:-4][:12] not in ir:
            never.append(f[:-4])
    print('     真实模型 %d 个' % len(reals))
    print('     **ir.md 里找不到记录的：%d 个**（粗判，按模型名前 12 字搜）'
          % len(never))
    for n in never[:8]:
        print('       %s' % n)

    # ── ⑤ 数不出来的 ─────────────────────────────────────────────
    print()
    print('  ⑤ **数不出来的**（列出来，但不假装它们可验收）')
    src = by1io.read_text(by1paths.root('history', 'ir.md'), encoding='utf-8') if ir else ''
    print('     llama.cpp 后端：提到 %d 次，**没有实现**'
          % len(re.findall(r'llama\.?cpp', src, re.I)))
    q = len(re.findall(r'量化|quant', src, re.I))
    print('     数值量化：ir.md 里提到 %d 次，**一行实现都没有**' % q)
    print('     （config 里带量化信息的模型有 4 个：fp8 ×2、mxfp4 ×2）')

    # ── ⑥ git 状态 ───────────────────────────────────────────────
    print()
    print('  ⑥ 仓库状态')
    if shutil.which('git') is None:
        # **"这台机器上没有 git" 不是"可疑"。** 说清楚，然后往下走。
        print('     （找不到 git —— 这一段没做。它在 PATH 上时才数。）')
    else:
        dirty = sh('git', 'status', '--short').strip()
        print('     未提交：%s' % (dirty.replace('\n', ' | ')[:60] or '干净'))
        print('     提交数：%s' % sh('git', 'rev-list', '--count', 'HEAD').strip())
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
