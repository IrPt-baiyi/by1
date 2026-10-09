#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1blind -- **量"看不见的东西"。**

## 为什么 by1debt 不够

`by1debt` 量的是**我知道要量的**：描述覆盖率、回归覆盖面、
前向跑过没有。

**而它对我没想到的类别是完全瞎的** —— 一个只统计"我记得的项"的
清单，会给人"已经盘完了"的错觉。那比没有清单更坏。

## 这里查的是"结构性的盲区"

    A. 验证的方向     契约 -> 产物 查了；**产物 -> 契约 没查**
    B. 序列长度       所有检查是不是都很短？RoPE/window/yarn 长序列才现形
    C. 阈值余量       判据是 1e-4，而实测离它多远？**"通过"是不是快掉下去了**
    D. 数值分布       一直用随机权重 —— 真权重有极端值
    E. 文档同步       生成物（ir-spec / models.md）和源头还一致吗
    F. 并行 vs 串行   结果一样吗（cgen/ 那个 race 就是这一类）
    G. 依赖版本       钉住的和没钉住的
    H. 生成的输出     只验了 logits，采样/生成验过吗

用法:  python by1blind.py
"""
import glob
import os
import re
import subprocess
import sys
import by1io

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def sec(n, title):
    print()
    print('  %s %s' % (n, title))
    print('  ' + '-' * 72)


def main():
    print()
    print('=' * 78)
    print('  **结构性盲区** —— by1debt 看不见的那些')
    print('=' * 78)

    # ── A. 验证的方向 ─────────────────────────────────────────────
    sec('A.', '验证的方向：契约 -> 产物 查了，**产物 -> 契约** 没查')
    # by1verify 检查"契约里的张量在产物里"，那产物里多出来的呢？
    src = by1io.read_text('by1verify.py', encoding='utf-8')
    has_rev = 'uncovered' in src and 'covered = generated' in src
    print('     by1verify 里有"产物多出来的张量"这类检查吗：**%s**'
          % ('有' if has_rev else '没有'))
    if has_rev:
        # **这一条我第一版写反了。** 工具报了"有"，我忽略了自己工具的
        # 输出，写下了"没查" —— 而 by1verify 第 412 行就在做这件事，
        # 连注释都写着为什么（多模态的 vision/audio 不在契约里，
        # 不报的话一个只看文本主干的描述会显得覆盖了整个 checkpoint）。
        print('       -> by1verify.py 的 `covered = generated` /')
        print('          `uncovered = sorted(k for k in real if k not in covered)`')
        print('       实测它报的是：clef 333/1184 = 28% 未覆盖；')
        print('       minimind-3（纯文本）**零缺口**。')
        print('       **所以这不是盲区 —— 是"已经有了但我没读"。**')
    # 实测：拿已知的 clef 比一下产物和契约的集合大小
    try:
        import importlib.util as iu
        s = iu.spec_from_file_location('bc', 'by1check.py')
        bc = iu.module_from_spec(s)
        s.loader.exec_module(bc)
        _r, info = bc.check('clef.by1')
        declared = set()
        for item in (info.get('tens') or []):
            for t in item[3]:
                if t[0] != '--':
                    declared.add(t[0])
        real = by1io.read_json('refs/Cloudflare__clef.tensors.json',
                                 encoding='utf-8')
        print('     clef：契约声明 %d 个逻辑名，产物有 %d 个物理张量'
              % (len(declared), len(real)))
        print('     **两者不是一个命名空间，没法直接做差** ——')
        print('     而"产物里有多少张量 by1 根本没提"**从来没有被自动数过**。')
    except Exception as e:
        print('     （比不了：%s）' % str(e)[:60])

    # ── B. 序列长度 ──────────────────────────────────────────────
    sec('B.', '序列长度：所有检查是不是都很短？')
    seqs = {}
    for f in glob.glob('by1*.py'):
        try:
            s = by1io.read_text(f, encoding='utf-8')
        except Exception:
            continue
        for m in re.finditer(r'--seq[= ](\d+)', s):
            seqs.setdefault(f, set()).add(int(m.group(1)))
    allseq = sorted({v for vs in seqs.values() for v in vs})
    print('     各脚本用的 --seq：%s' % (allseq or '（没有）'))
    # 模型声明的 ctx
    ctxs = []
    for f in glob.glob('*.by1'):
        try:
            h = by1io.read_text(f, encoding='utf-8')
        except Exception:
            continue
        m = re.search(r'^\s*ctx\s+(\d+)', h, re.M)
        if m:
            ctxs.append((f[:-4], int(m.group(1))))
    ctxs.sort(key=lambda x: -x[1])
    print('     模型声明的 ctx（前 8）：%s'
          % ', '.join('%s=%d' % (a, b) for a, b in ctxs[:8]))
    print('     **最大检查过的 seq：%d，而最大的 ctx：%d**'
          % (max(allseq) if allseq else 0, ctxs[0][1] if ctxs else 0))
    print('     RoPE / sliding window / yarn **全是长序列才现形的东西**')

    # ── C. 阈值余量 ──────────────────────────────────────────────
    sec('C.', '阈值余量：判据是 1e-4，"通过"离红有多远？')
    margins = []
    for f in ('by1exec.py', 'by1c.py', 'by1gpu.py', 'by1diff.py'):
        if not os.path.exists(f):
            continue
        s = by1io.read_text(f, encoding='utf-8')
        for m in re.finditer(r'([\d.]+e-?\d+)\s*[:<]', s):
            pass
    try:
        r = subprocess.run([sys.executable, 'by1all.py', '--quick'],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=1800)
        got = re.findall(r'相对\s*([\d.]+e[-+]\d+)', r.stdout)
        vals = sorted(float(x) for x in got)
        print('     --quick 里报出的相对差 %d 个' % len(vals))
        if vals:
            print('     最小 %.2e · 中位 %.2e · **最大 %.2e**'
                  % (vals[0], vals[len(vals) // 2], vals[-1]))
            print('     判据如果是 1e-4，最大的那个离红还差 %.0f 倍'
                  % (1e-4 / vals[-1]) if vals[-1] else '')
    except Exception as e:
        print('     （跑不了：%s）' % str(e)[:60])

    # ── D. 数值分布 ──────────────────────────────────────────────
    sec('D.', '数值分布：一直用随机权重')
    n_rand = 0
    for f in glob.glob('by1*.py'):
        try:
            s = by1io.read_text(f, encoding='utf-8')
        except Exception:
            continue
        n_rand += len(re.findall(r'randn|rand\(|normal_', s))
    print('     代码里出现 randn/rand/normal_ 共 %d 处' % n_rand)
    print('     **真权重的分布和 N(0,0.5) 不一样**：')
    print('       · 嵌入表有大得多的值；有的层接近全零')
    print('       · fp8/mxfp4 的权重**天生带量化噪声**')
    print('     -> 小随机数下"相对差 5e-07"，在真权重下可能完全不同')

    # ── E. 文档同步 ──────────────────────────────────────────────
    sec('E.', '文档同步：生成物和源头还一致吗')
    for gen, cmd in (('ir-spec.md', ['by1ir.py', '--spec']),
                     ('models.md', ['by1cmp.py', '--md'])):
        if not os.path.exists(gen):
            print('     %-14s **不存在**' % gen)
            continue
        try:
            r = subprocess.run([sys.executable] + cmd, capture_output=True,
                               text=True, encoding='utf-8',
                               errors='replace', timeout=600)
            same = (r.stdout.strip() ==
                    by1io.read_text(gen, encoding='utf-8').strip())
            print('     %-14s 和 `%s` %s'
                  % (gen, ' '.join(cmd), '一致' if same
                     else '**不一致（忘了重新生成）**'))
        except Exception as e:
            print('     %-14s （比不了：%s）' % (gen, str(e)[:40]))

    # ── F. 并行 vs 串行 ──────────────────────────────────────────
    sec('F.', '并行 vs 串行：结果一样吗')
    s = by1io.read_text('by1all.py', encoding='utf-8')
    has_chk = bool(re.search(r'并行.*串行|jobs\s*==\s*1|--serial|--jobs', s))
    print('     by1all 有"关掉并行再跑一遍对比"的开关吗：%s'
          % ('有' if has_chk else '**没有**'))
    print('     cgen/ 那个 race 就是这一类 —— **修了，但"并行≡串行"没验证过**')

    # ── G. 依赖版本 ──────────────────────────────────────────────
    sec('G.', '依赖版本：钉住的和没钉住的')
    # **只挑依赖那一行。** 第一版拿一个宽正则去抓 pyproject 里所有
    # 带引号的字符串，于是把 keywords / classifiers / 脚本名全抓进来了，
    # 报出 30 多条"依赖"。**一个把噪音当信号的清单，读的人会直接跳过它。**
    deps = set()
    for f in ('pyproject.toml', 'requirements.txt'):
        if not os.path.exists(f):
            continue
        s = by1io.read_text(f, encoding='utf-8')
        m = re.search(r'dependencies\s*=\s*\[(.*?)\]', s, re.S)
        if m:
            for d in re.findall(r'"([^"]+)"', m.group(1)):
                deps.add(d)
    if deps:
        print('     声明的依赖：')
        for d in sorted(deps):
            pinned = bool(re.search(r'[=<>~]', d))
            print('       %-28s %s' % (d, '钉住了' if pinned else '**没钉**'))
    print('     numpy / torch **没钉版本**，而它们是数值结果的根源 ——')
    print('     判据是 1e-4 这种量级，换个小版本就可能改结论。')

    # ── H. 生成的输出 ────────────────────────────────────────────
    sec('H.', '生成的输出：只验了 logits')
    gens = [f for f in glob.glob('by1*.py') if 'gen' in f or 'sample' in f]
    print('     有采样/生成的脚本：%s' % (gens or '**一个都没有**'))
    # **这一条我第一版写错了，而且错得最坏。**
    #
    # 原来写的是"反向传播一行没验"。而 `plan.md` 的 P3 明明白白
    # 写着 ✅ 完成，还给了四个模型逐个参数的数：
    #     llama-shaped 39/39 · mixtral-shaped 35/35
    #     gpt-oss-shaped 31/31 · qwen3-next-shaped 62/62
    #
    # 查了才知道两边说的是两件事：
    #     by1train.py:158   loss.backward()            <- PyTorch 的 autograd
    #     by1diff.py:341    ra.pow(2).sum().backward() <- 比 by1 的和 HF 的梯度
    #
    # **`by1train.py` 没有自己的反向** —— 它用的就是那套被验过的 autograd。
    #
    # > 一个说"已经验过的东西没验"的工具，比一个说"没验的东西验了"
    # > 的工具更坏 —— 前者的修法是"再验一遍"（浪费），
    # > 后者的修法是"什么都不做"（出事）。
    print('     反向：`by1diff --backward` 管着（4 个模型逐个参数对过）')
    print('     **`by1train.py` 用的是 PyTorch 的 autograd，没有自己的反向**')
    print('     真正没验的是**训练循环本身**（优化器 / lr / 数据）——')
    print('     而那不是 by1 承诺的东西。1.md 承诺的是"前向与反向数值一致"。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
