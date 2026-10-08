#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1boot -- 从一个已发布的 checkpoint **反推**一份 .by1 草稿。

## 为什么是这一件

量过：一份 `.by1` 里 27% 是决定、64% 是机械。而机械那一半里：

    张量契约   机制 + 属性 -> 名字和形状     by1contract.py 已经能推
    导出命名   「这个仓库怎么命名」           **推不出来 —— 除非问产物**

**而产物就在那儿。** config 里有维度，safetensors 的头里有名字和形状 ——
`by1verify` 每天在读它们，只是**只拿去比对、不拿去生成**。

## 这一件的真正形状

它**不是**"帮你写 .by1"，它是**把"你必须懂 73 个属性"换成"你改到验过为止"**：

    by1boot  ->  草稿 .by1
    by1verify ->  哪里对不上
    你改      ->  直到全绿

**草稿对不对不需要你判断 —— 判卷人判断。** 这才是"小学生也能写"
那条路的真身：不是让他写，是**让他改，而且改到哪儿算对由机器说了算**。

用法:  python by1boot.py <config.json> <tensors.json> [--name X] [--out f.by1]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict


def norm(k):
    return re.sub(r"\.\d+\.", ".N.", k)


def layer_of(k):
    m = re.search(r"\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def classify(names):
    """一层里的张量名 -> 大概是哪种机制。**说不出就说不说。**"""
    s = " ".join(names)
    if re.search(r"linear_attn|\.mixer\.(A_log|D|conv1d)", s):
        return "Linear"
    if re.search(r"kv_a_proj|q_a_proj", s):
        return "MLA"
    if re.search(r"self_attn\.(q_proj|wq|c_attn)|attn\.qkv", s):
        return "Attention"
    if re.search(r"mixer\.(q_proj|k_proj)", s):
        return "Attention"
    return None


def ffn_of(names):
    s = " ".join(names)
    if re.search(r"experts\.|expert", s):
        return "MoE"
    if re.search(r"mlp\.(gate_proj|c_fc|up_proj)", s):
        return "FFN"
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cfg_p, ten_p = sys.argv[1], sys.argv[2]
    name = None
    out_p = None
    if '--name' in sys.argv:
        name = sys.argv[sys.argv.index('--name') + 1]
    if '--out' in sys.argv:
        out_p = sys.argv[sys.argv.index('--out') + 1]

    cfg = json.load(open(cfg_p, encoding='utf-8'))
    cfg = cfg.get('text_config', cfg)
    real = json.load(open(ten_p, encoding='utf-8'))

    name = name or cfg.get('model_type', 'booted').replace('.', '-')
    d = cfg.get('hidden_size')
    L = cfg.get('num_hidden_layers')
    V = cfg.get('vocab_size')

    # ── 按层分组 ───────────────────────────────────────────────────
    per_layer = defaultdict(list)
    globals_ = []
    for k in real:
        li = layer_of(k)
        (per_layer[li] if li is not None else globals_).append(k)

    # ── 每层是什么 ─────────────────────────────────────────────────
    kinds = {}
    ffns = {}
    for li in sorted(per_layer):
        ks = per_layer[li]
        kinds[li] = classify(ks)
        ffns[li] = ffn_of(ks)
    kc = Counter(kinds.values())
    fc = Counter(ffns.values())

    L_ = ['# ── 由 by1boot 从产物反推的草稿 ──────────────────────────────',
          '#',
          '# **它不是"帮你写"，是"让你改到验过为止"**：',
          '#     python by1verify.py <这个文件> %s --config --tensors %s --backend torch.module'
          % (os.path.basename(cfg_p), os.path.basename(ten_p)),
          '#',
          '# 反推不出来的地方，下面都标了 `# ?`。改到 by1verify 全绿为止。',
          '# ──────────────────────────────────────────────────────────',
          '',
          'model %s {' % name,
          '  arch  %s' % cfg.get('model_type', '?'),
          '  ctx   %d' % cfg.get('max_position_embeddings', 0),
          '',
          '  hparams {',
          '    d_model = %s' % d,
          '    n_layer = %s' % L,
          '    vocab   = %s' % V,
          '    rms_eps = %s' % cfg.get('rms_norm_eps', 1e-5),
          '  }',
          '']
    print('\n'.join(L_))
    print('  # 反推结果：')
    print('  #   层数 %d，机制分布 %s' % (len(per_layer), dict(kc)))
    print('  #   挂在层上的 %s' % dict(fc))
    print('  #   全局张量 %d 个：%s' % (len(globals_), ', '.join(sorted(globals_)[:6])))
    print('  #')
    print('  # ? 反推不出来的：路由方式 / 有没有共享专家 / 门控激活 /')
    print('  #   per_expert 还是打包 / bias / rope 的缩放类型 —— 这些**config 里没有**，')
    print('  #   而产物只有名字和形状。改到 by1verify 全绿为止。')
    print()

    # ── 契约：**直接从产物里读** ───────────────────────────────────
    # 这是这一件里最值钱的部分，而它推不出来 —— 除非问产物。
    # 形状写成**字面数字**（不是符号）：符号需要先知道哪个数是哪个维度，
    # 而那正是你要决定的。字面数字至少是**对的**，能立刻验过。
    print('  # ── 契约（从 safetensors 头里读的，形状是字面值）──────────')
    print('  # 把 (4096, 2688) 换成 (q * head_dim, d_model) 这类符号，')
    print('  # 就是"我理解了这个模型" —— 而 by1check 会告诉你换得对不对。')
    print()
    groups = defaultdict(dict)
    layern = {}
    # **一层里同时有混合器和 FFN，得分桶。**
    # 第一版按"这一层是什么机制"分，于是 mlp.* 全被塞进了 Attention 桶里 ——
    # 名字是对的、形状是对的、**分错了组**。而分错组的表现是
    # "Attention 里有 mlp"，一眼能看出来，前提是你去看。
    MIX = re.compile(r'^(self_attn|linear_attn|mixer|attn|attention)\.')
    FFN = re.compile(r'^(mlp|feed_forward|ffn)\.')
    for li in sorted(per_layer):
        kmain = kinds[li] or 'Raw'
        kffn = ffns[li]
        for nm in per_layer[li]:
            short = re.sub(r'^.*layers\.\d+\.', '', nm)
            short = re.sub(r'^(model|backbone|language_model)\.', '', short)
            short = re.sub(r'^layers\.\d+\.', '', short)
            if MIX.match(short):
                groups[kmain].setdefault(short, real[nm]['shape'])
            elif FFN.match(short) and kffn:
                groups[kffn].setdefault(short, real[nm]['shape'])
            elif re.search(r'norm|layernorm', short):
                layern.setdefault(short, real[nm]['shape'])
            else:
                layern.setdefault(short, real[nm]['shape'])
    for key in sorted(groups):
        print('    %s {' % key)
        for nm in sorted(groups[key]):
            shp = ', '.join(str(x) for x in groups[key][nm])
            print('      %-40s (%s)' % (nm + ' :', shp))
        print('    }')
    print('    layer {')
    for nm in sorted(layern):
        shp = ', '.join(str(x) for x in layern[nm])
        print('      %-40s (%s)' % (nm + ' :', shp))
    print('    }')
    print('    global {')
    for nm in sorted(globals_):
        shp = ', '.join(str(x) for x in real[nm]['shape'])
        print('      %-40s (%s)' % (nm + ' :', shp))
    print('    }')
    print('  }')
    return 0


if __name__ == '__main__':
    sys.exit(main())
