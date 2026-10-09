#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1name -- **模型命名的唯一执行者。**

## 分工（用户定的）

    长名   = HF 的**模型名**（不带 owner）   -> **文件名**
    短名   = **手动起的**                    -> model 声明、表格、显示

    文件名     Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1
    model 声明 step37-pruned

## 四条规矩

    ① 文件名只要**模型名称**，不带 owner
       （`Qwen3.8-27B.by1`，不是 `Qwen__Qwen3.8-27B.by1`）
    ② 长名和短名都在 `models.tsv` 里 —— **那是唯一的真相源**
    ③ 这个脚本**批量生成**：改名、改 model 声明、改头部注释、改引用
    ④ 格式统一按 HF 的命名（owner 只记在 manifest 里，**不进文件名**）

## 为什么要有这个

在这之前，文件名是**随手起**的：

    qwen38.by1        vs  gemma-4-31b.by1      一个缩写一个全名
    step-3.7-flash.by1                         **误导**：听起来像官方，
                                               实际是社区的剪枝版

而"从文件名追回是哪个仓库"这件事，**不该靠记性**。

## 一条纪律

**用显式表，不推断。** 第一版想从 `refs/` 的 config 名反查"它现在叫什么"
—— 找不到 8 个，而报错长得像"manifest 写错了"。
**推断在这一步不值得**：改名是一次性的，写清楚就行。

用法:
    python by1name.py            只报告
    python by1name.py --apply    真改
"""
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, 'models.tsv')

# 引用会出现在哪些文件里。
# **ir.md 不在里面** —— 那是日志，老条目里的老名字是历史事实。
REF_FILES = ['by1all.py', 'by1gate.py', 'by1pack.py', 'plan.md', 'README.md',
             'by1e2e.py', 'by1real.py']

# **现名 -> 长名**。一次性：改完之后这些键就不存在了，留着无害。
# 显式写出来，不推断 —— 见文件头那条纪律。
CURRENT = {
    'gemma-4-31b.by1': 'gemma-4-31B',
    'laguna-xs-2.1.by1': 'Laguna-XS-2_1',
    'instella-3b.by1': 'Instella-3B',
    'nerkyor_Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.by1':
        'Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL',
    'stepfun-ai__Step-3.7-Flash.by1': 'Step-3.7-Flash',
    'ling-3.0-tiny.by1': 'Ling-3.0-tiny',
    'nemotron-h.by1': 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    'qwen38.by1': 'Qwen3.8-27B',
    'qwen36.by1': 'Qwen3.6-35B-A3B',
    'glm53.by1': 'GLM-5.3-Flash',
}


def read_manifest():
    out = []
    for line in io.open(TSV, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.lstrip().startswith('#'):
            continue
        p = line.split('\t')
        if len(p) >= 4:
            out.append((p[0], p[1], p[2], p[3]))
    return out


def files_on_disk():
    return {f for f in os.listdir(HERE) if f.endswith('.by1')}


def fix_file(old, 长名, 短名, hf, do):
    """改一个文件：改名 + model 声明 + 头部三行标签。"""
    new = 长名 + '.by1'
    if old != new:
        if do:
            r = subprocess.run(['git', 'mv', old, new], cwd=HERE,
                               capture_output=True, text=True)
            if r.returncode != 0:
                # git mv 对未跟踪的文件会失败 —— 退化成普通改名
                os.rename(os.path.join(HERE, old), os.path.join(HERE, new))
        print('    %s %-42s -> %s' % ('改名' if do else '[要改]',
                                      old[:42], new))
    path = os.path.join(HERE, new)
    if not do:
        # **只报告的时候不能检查新路径** —— 还没改名呢，它当然不存在。
        # 第一版就是在这里对每个文件都打了一行"!! 不存在"，
        # 看起来像 manifest 写错了。**假警报比不报还坏。**
        # 所以：读**老**文件，照常算出要改什么，只是不写盘。
        path = os.path.join(HERE, old)
    if not os.path.exists(path):
        print('    !! %s 不存在' % os.path.basename(path))
        return

    t = io.open(path, encoding='utf-8').read()
    before = t

    # model 声明 -> 短名
    t = re.sub(r'^model\s+[\w.\-]+\s*\{', 'model %s {' % 短名, t,
               count=1, flags=re.M)

    # 头部三行标签（**文件名是长名，看的人未必知道它指什么**）
    tag = ('# by1-model: %s\n# by1-repo:  %s\n# by1-short: %s\n'
           % (长名, hf, 短名))
    if '# by1-model:' in t[:2000]:
        t = re.sub(r'# by1-model:.*\n(# by1-repo:.*\n)?(# by1-short:.*\n)?',
                   tag, t, count=1)
    else:
        m = re.match(r'((?:#[^\n]*\n)+)', t)
        t = (t[:m.end()] + tag + t[m.end():]) if m else (tag + t)

    if t != before:
        if do:
            io.open(path, 'w', encoding='utf-8').write(t)
        print('        %s model %-16s repo %s'
              % ('改' if do else '[要改]', 短名, hf))


def main():
    do = '--apply' in sys.argv
    man = read_manifest()
    files = files_on_disk()

    plan = []
    missing = []
    for 长名, 短名, hf, why in man:
        want = 长名 + '.by1'
        if want in files:
            plan.append((want, 长名, 短名, hf))
            continue
        old = None
        for c, ln in CURRENT.items():
            if ln == 长名 and c in files:
                old = c
                break
        if old:
            plan.append((old, 长名, 短名, hf))
        else:
            missing.append((长名, 短名, hf))

    print()
    print('  %s' % ('**真改**' if do else '只报告（加 --apply 才真改）'))
    print()
    print('  ── 逐条（manifest %d 条）' % len(man))
    for old, 长名, 短名, hf in plan:
        fix_file(old, 长名, 短名, hf, do)
    if missing:
        print()
        print('  ── **manifest 里写了、但本地没有的**（%d 个）' % len(missing))
        for 长名, 短名, hf in missing:
            print('    %-42s %s' % (长名[:42], hf))

    # 引用
    ren = [(old, 长名 + '.by1') for old, 长名, _s, _h in plan
           if old != 长名 + '.by1']
    if ren:
        print()
        print('  ── 引用（%d 对改名）' % len(ren))
        for fn in REF_FILES:
            p = os.path.join(HERE, fn)
            if not os.path.exists(p):
                continue
            t = io.open(p, encoding='utf-8').read()
            n = 0
            for old, new in ren:
                n += t.count(old)
                t = t.replace(old, new)
            if n:
                if do:
                    io.open(p, 'w', encoding='utf-8').write(t)
                print('    %s %s：%d 处' % ('改' if do else '[要改]', fn, n))

    # 本地有、manifest 里没有的
    known = {长名 + '.by1' for 长名, _s, _h, _w in man}
    known |= {old for old, _l, _s, _h in plan}
    extra = sorted(files - known)
    if extra:
        print()
        print('  ── 本地有、manifest 里没写的（%d 个，不是错误）' % len(extra))
        print('    %s' % ', '.join(extra))
        print('    这些不是从 HF 仓库来的（shaped / 探针 / 自测），')
        print('    命名本来就不适用那四条规矩。')

    print()
    print('  ir.md **不动** —— 那是日志。老条目里的老名字是历史事实；')
    print('  日志改了就变成"当初就是这么写的"，那是假的。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
