#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1cmp -- **把所有模型的架构摆成一张表。**

## 为什么这个值得做

`1.md` 里写过一句：

> config.json 是平的 JSON，做不了这件事。
> 差一层 MTP 在 JSON 里是"15 个键的增删"，在 `.by1` 里是 `aux = true` 一行。

**这张表就是那句话的兑现。** 而且它是**从 IR 生成的** ——
手写的对比表会过期，而过期的规格比没有规格更坏。

## 它回答什么

    ✗ 哪个模型更强        —— 那不是这张表的事
    ✓ 哪些机制是真不一样的 —— nanoGPT 那一代的 LayerNorm/查表位置/无门控，
                            和 Llama 那代的 RMSNorm/RoPE/SwiGLU，不是"参数不同"
    ✓ 哪些属性是"全都在用"的 —— 那些是收敛了的，不值得再看
    ✓ 哪几个模型是独苗     —— 独苗才是需要新原语的地方

## 判据

**只列在模型之间真的不一样的属性。** 全都一样的列出来是噪音 ——
而噪音会把信号淹掉（这个项目里反复吃过这个亏）。

用法:  python by1cmp.py [--md] [--only 属性名]
"""
import glob
import importlib.util
import io
import os
import sys
from collections import Counter, defaultdict
import by1io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SKIP = {'selftest.by1', 'gate-probe.by1', 'hello.by1', 'raw-escape.by1',
        'llama-shaped.by1', 'mixtral-shaped.by1', 'gpt-oss-shaped.by1',
        'qwen3-next-shaped.by1', 'mla-shaped.by1', 'llama3-shaped.by1',
        'clef-tiny.by1', 'gpt2-tiny.by1'}


def load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    md = '--md' in sys.argv
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1]

    bc, cg = load('by1check'), load('by1codegen')

    rows = {}          # 文件 -> 扁平化的属性
    for f in sorted(glob.glob(os.path.join(HERE, '*.by1'))):
        name = os.path.basename(f)
        if name in SKIP:
            continue
        try:
            _r, info = bc.check(f)
            ir = cg.compile_ir(info)
        except Exception:
            continue
        d = {}
        # 顶层
        for k in ('d_model', 'vocab', 'ctx', 'pos_kind', 'norm_kind',
                  'norm_eps', 'norm_one_plus'):
            d[k] = ir.get(k)
        d['n_layer'] = len(ir['layers'])
        # 机制：每一层用了哪些 kind
        kinds = Counter()
        for L in ir['layers']:
            for o in L['ops']:
                if o['kind'] not in ('Norm', 'Add'):
                    kinds[o['kind']] += 1
        d['机制'] = '+'.join('%s×%d' % (k, v) for k, v in sorted(kinds.items()))
        # 每种 kind 的属性（取第一个出现的）
        seen = {}
        for L in ir['layers']:
            for o in L['ops']:
                if o['kind'] in ('Norm', 'Add'):
                    continue
                seen.setdefault(o['kind'], o['attrs'])
        for kind, a in seen.items():
            for k, v in a.items():
                d['%s.%s' % (kind, k)] = v
        # 参数量
        try:
            ex = load('by1exec')
            d['参数'] = sum(int(__import__('numpy').prod(s))
                            for s in ex.shapes_of(ir).values())
        except Exception:
            d['参数'] = None
        rows[name] = d

    # **只留有差异的列。**
    allkeys = sorted({k for d in rows.values() for k in d})
    varies = []
    for k in allkeys:
        vals = {repr(d.get(k, '—')) for d in rows.values()}
        if len(vals) > 1 or k in ('d_model', 'n_layer', '参数', '机制',
                                  'pos_kind', 'norm_kind'):
            varies.append(k)
    if only:
        varies = [k for k in varies if only in k]

    names = sorted(rows, key=lambda n: (rows[n].get('参数') or 0))

    # **表里显示短名，不显示文件名。**
    #
    # 文件名现在是 HF 的模型名（按用户定的四条），有的 53 个字符 ——
    # 塞进表格会把表撑坏，可读性没了。
    # 而每个 `.by1` 里的 `model <短名> {` 就是给人看的那个名字，
    # 由 `by1name.py` 从 `models.tsv` 统一生成，不会漂。
    import re as _re

    def label_of(fname):
        """短名 —— 读文件里的 model 声明，不另存一份。

        **正则要允许 `/`** —— 同名不同源时短名带 owner
        （`nerkyor/Step-3_7-Flash`）。
        """
        try:
            src = by1io.read_text(os.path.join(HERE, fname))
            m = _re.search(r'^model\s+([\w.\-/]+)\s*\{', src, _re.M)
            return m.group(1) if m else fname[:-4]
        except Exception:
            return fname[:-4]

    # 连续的数值属性每人一个值，"独苗"没有意义 —— 排除掉，
    # 否则满屏独苗，真独苗就被淹了。
    NUMERIC = {'d_model', 'n_layer', '参数', 'vocab', 'ctx',
               'Attention.head_dim', 'Attention.q', 'Attention.kv',
               'Attention.out_dim', 'FFN.hidden', 'MoE.experts',
               'MoE.hidden', 'MoE.shared_hidden', 'Linear.out_dim',
               'Linear.k_dim', 'Linear.k_heads', 'Linear.v_dim',
               'Linear.v_heads', 'norm_eps', 'Attention.norm_eps',
               'Attention.rope_base', 'MoE.routed_scale', 'MoE.top_k'}
    # 机制那一列，展示用
    def short_mech(s):
        return (s.replace('Attention', 'Attn').replace('Linear', 'Lin')
                 .replace('FFN×', 'FFN×').replace('MoE', 'MoE'))

    def solo():
        """哪些 (属性, 取值) 只有**一个**模型有。**独苗才是要新原语的地方。**"""
        out = []
        for k in varies:
            if k in NUMERIC:
                continue              # 连续的，不比
            if k == '机制':
                # **每个模型的机制组合本来就唯一** —— 列出来是废话，
                # 而且会把真独苗淹掉（28 行里有 10 行是它）。
                continue
            vals = defaultdict(list)
            for n in names:
                vals[repr(rows[n].get(k, '—'))].append(label_of(n))
            if len(vals) < 2:
                continue
            for v, ns in vals.items():
                if len(ns) == 1:
                    out.append((k, v, ns[0]))
        return sorted(out)

    if md:
        print('# by1 支持的模型 —— 架构对比')
        print()
        print('> **从 IR 生成的**（`python by1cmp.py --md`），不是手写的。')
        print('> 手写的对比表会过期，而过期的表比没有表更坏。')
        print('>')
        print('> **只列在模型之间真的不一样的属性。** 全都一样的列出来是噪音，')
        print('> 而噪音会把信号淹掉 —— 这个项目里反复吃过这个亏。')
        print('>')
        print('> 表里是**短名**（可读）；**文件名是 HF 的模型名**（可追溯）。')
        print('> 两者都由 `models.tsv` + `by1name.py` 生成，不会漂。')
        print()
        # ── 命名对照：短名 / 长名 / HF id ──────────────────────────
        # **必须放在前面。** 表里只有短名，看的人要能查到它是谁。
        mtsv = os.path.join(HERE, 'models.tsv')
        if os.path.exists(mtsv):
            print('### 命名对照（`models.tsv` 是唯一真相源）')
            print()
            print('| 短名（表里用的） | 长名（= 文件名） | HF 仓库 |')
            print('|---|---|---|')
            for line in by1io.iter_lines(mtsv, encoding='utf-8'):
                line = line.rstrip('\n')
                if not line or line.lstrip().startswith('#'):
                    continue
                p = line.split('\t')
                if len(p) >= 3:
                    print('| `%s` | `%s` | [%s](https://hf-mirror.com/%s) |'
                          % (p[1], p[0], p[2], p[2]))
            print()
        print('## 一、谁是谁')
        print()
        print('| 模型 | 参数 | 层 | d_model | 词表 | 机制 |')
        print('|---|---|---|---|---|---|')
        for n in names:
            d = rows[n]
            par = d.get('参数') or 0
            ps = ('%.1fB' % (par / 1e9) if par >= 1e9 else
                  '%.0fM' % (par / 1e6) if par >= 1e6 else str(par))
            print('| `%s` | %s | %d | %d | %s | %s |'
                  % (label_of(n), ps, d.get('n_layer'),
                     d.get('d_model'), d.get('vocab'),
                     short_mech(d.get('机制', ''))))
        print()
        print('## 二、**独苗** —— 只有它这样的')
        print()
        print('**独苗才是需要新原语的地方。** 别的都只是参数不同。')
        print()
        print('| 属性 | 取值 | 只有 |')
        print('|---|---|---|')
        for k, v, n in solo():
            print('| `%s` | `%s` | **%s** |' % (k, v, n))
        print()
        # **落点：有多少模型一个独苗都没有。**
        # 那才是"原语集合收敛"这句话的可数形式 ——
        # 说"收敛了"是感觉，说"N 个模型贡献 0 个新原语"是账。
        solo_models = {n for _k, _v, n in solo()}
        clean = [n.replace('.by1', '') for n in names
                 if n.replace('.by1', '') not in solo_models]
        print('**%d 个模型里，%d 个一个独苗都没有**：%s'
              % (len(names), len(clean), ', '.join(clean)))
        print()
        print('> 这就是"原语集合收敛"这句话的可数形式 ——')
        print('> 说"收敛了"是感觉，说"这几个模型贡献 0 个新原语"是账。')
        print()
        print('> 而独苗集中在 **gpt2**（另一个时代）和 **gpt-oss-120b**')
        print('> （最特殊的一个）—— **新原语是从"另一个时代"和"最特殊的那个"')
        print('> 来的，不是从"又一个 Llama 变体"来的。**')
        print()
        print('## 三、逐属性的全部取值')
        print()
        for k in varies:
            vals = defaultdict(list)
            for n in names:
                vals[repr(rows[n].get(k, '—'))].append(label_of(n))
            if len(vals) == 1:
                continue
            print('### `%s`' % k)
            print()
            for v, ns in sorted(vals.items(), key=lambda kv: -len(kv[1])):
                mark = ' ← **独苗**' if len(ns) == 1 else ''
                print('- `%s` — %s%s' % (v, ', '.join(ns), mark))
            print()
        return 0

    # 文本模式：分组打印
    print('=' * 78)
    print('  所有模型 —— 只列**真的在变**的属性')
    print('=' * 78)
    print()
    print('  %d 个模型，%d 个有差异的属性' % (len(rows), len(varies)))
    print()
    for k in varies:
        vals = defaultdict(list)
        for n in names:
            vals[repr(rows[n].get(k, '—'))].append(label_of(n))
        if len(vals) == 1:
            continue
        print('  ── %s' % k)
        for v, ns in sorted(vals.items(), key=lambda kv: -len(kv[1])):
            tag = '' if len(ns) > 1 else '   ← **独苗**'
            print('    %-22s %s%s' % (v[:22], ', '.join(ns)[:52], tag))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
