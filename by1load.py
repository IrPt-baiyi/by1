#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1load -- 把**真权重**装进 by1 生成的模型。

## 为什么这个能做，而 gpt2 那次要手写

`gpt2.by1` 那一轮，映射是**手写的** —— 因为 GPT-2 的 checkpoint 是
**融合 qkv + Conv1D `[in, out]`**，那是仓库的习惯，推不出来。

而 clef / qwen3.8 这类不一样：`.by1` 里**已经把命名规则写下来了**：

    name        = "model.language_model.layers.{i}.{scope}{physical}"
    global_name = "{physical}"
    scope { Lin = linear_attn., GQA = self_attn., FFN_ = mlp., ... }
    rename { embed.weight = model.language_model.embed_tokens.weight, ... }

**那是描述的一部分，不是仓库的习惯。** `by1verify` 每天在用这个规则
（它就是这么验 851/851 的）。这个模块**把同一个规则反过来用**：
从契约名找到物理名，把权重装进模型。

## 一条纪律

物理名一律走 `by1check.render_name` —— **和 by1verify 用的同一个函数**。
这个项目里"同一个意思两处实现"出过好几次（`by1_layer_types` 就重复过），
每次都是两边慢慢漂开。所以这里不抄一份。

## 它不解的

    ✗ 融合存储（c_attn / gate_up_proj）—— 仓库的习惯
    ✗ 布局转置（Conv1D 的 [in, out]）—— 同上
    ✗ 量化格式（fp8 的 weight_scale_inv）—— 契约里没有

碰到会**明确报出来**，不静默跳过。
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    p = os.path.join(HERE, name + '.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plan(info, stack='main'):
    """算出 {(层号, 机制, 逻辑名) -> 物理张量名}。**不加载权重。**

    用的是 `.by1` 的 `emit { train -> torch.module { ... } }` ——
    和 `by1verify` 同一个 `render_name`。
    """
    bc = _load('by1check')
    rule = ((info.get('emit') or {}).get('torch.module') or {})
    if not rule.get('name'):
        raise SystemExit(
            '  这个 .by1 没有 emit { train -> torch.module { name = ... } }\n'
            '  **没有命名规则就推不出物理名** —— 那正是 gpt2 要手写的原因。')

    out, unresolved = {}, []
    for item in (info.get('tens') or []):
        mech, _attrs, nlay, tensors = item[0], item[1], item[2], item[3]
        # 这个机制出现在哪些层 —— 从 IR 里问，不猜
        layers = _layers_of(info, mech, nlay)
        if not layers:
            unresolved.append((mech, '在层序里找不到这个机制'))
            continue
        for t in tensors:
            logical = t[0]
            if logical == '--':
                continue          # 显式声明"不该存在"
            for i in layers:
                try:
                    phys = bc.render_name(rule, i, stack, mech, logical)
                except Exception as e:
                    unresolved.append((mech, '%s: %s' % (logical, str(e)[:40])))
                    continue
                out[(i, mech, logical)] = phys
    return out, unresolved


def _layers_of(info, mech, nlay):
    """哪些层用了这个机制。**从 layer_seq 里问，不按周期猜。**

    `layer_seq` 一项是 5 个字段：
        (栈名, 主机制, 属性, 结构属性, **挂载的机制列表**)

    **挂载的也得算。** 我第一版只看第二个字段 —— 于是 `FFN_` 这种
    「每层都挂一个」的机制被判成"在层序里找不到"，
    而它明明在每一层。那种报错会让人去查错的地方。
    """
    seq = info.get('layer_seq') or []
    hits = []
    for i, t in enumerate(seq):
        if len(t) > 1 and t[1] == mech:
            hits.append(i)
            continue
        # 挂载的机制在第五个字段里
        if len(t) > 4 and t[4] and mech in t[4]:
            hits.append(i)
    return hits


def internal_index(info):
    """{(层号, 机制, 逻辑名去掉 .weight) -> 模型内部的参数名}。

    模型的参数名是后端自己的（`layers.0.op1.in_proj_qkv`），
    而契约里的逻辑名是 `in_proj_qkv.weight` —— 差一个后缀。
    所以按"去掉 .weight"建索引。
    """
    out = {}
    for item in (info.get('tens') or []):
        mech, tensors = item[0], item[3]
        for i in _layers_of(info, mech, item[2]):
            for t in tensors:
                logical = t[0]
                if logical == '--':
                    continue
                key = logical.replace('.weight', '').replace('.', '_')
                out[(i, mech, logical)] = None      # 由调用方按 op 序号补
    return out


def load(model, info, fetch, stack='main'):
    """把真权重装进去。

    `fetch(physical_name) -> torch.Tensor 或 None`

    返回 (装上了几个, 缺哪些, 装不上的原因)。
    **缺的会列出来** —— 静默跳过是最坏的结果：模型照跑，数是错的。
    """
    pmap, unresolved = plan(info, stack)
    sd = model.state_dict()

    # 模型内部名 -> 契约逻辑名。按"去后缀 + 去下划线"比。
    by_short = {}
    for (i, mech, logical) in pmap:
        short = logical.replace('.weight', '')
        by_short.setdefault((i, short), []).append((mech, logical))

    done, missing, skipped = 0, [], []
    for key, tensor in sd.items():
        # 内部名形如 `layers.{i}.op{j}.{名字}`
        #
        # **`parts[2]` 是 `"op0"` 不是 `"0"`。** 第一版直接 `int(parts[2])`
        # —— 27B 那个模型跑到"装权重"才炸，报的是
        # `invalid literal for int() with base 10: 'op0'`。
        # 本地那些小模型没露出来，因为…… 它们走的是同一段代码 ——
        # **是我的测试里没有真权重，根本没走到这里。**
        parts = key.split('.')
        if len(parts) < 4 or parts[0] != 'layers' or not parts[2].startswith('op'):
            continue                     # 全局的下面单独处理
        try:
            i, opj = int(parts[1]), int(parts[2][2:])
        except ValueError:
            skipped.append(key)
            continue
        short = '.'.join(parts[3:]).replace('.weight', '')
        cands = by_short.get((i, short))
        if not cands:
            skipped.append(key)
            continue
        mech, logical = cands[0]
        phys = pmap[(i, mech, logical)]
        w = fetch(phys)
        if w is None:
            missing.append((key, phys))
            continue
        if tuple(w.shape) != tuple(tensor.shape):
            skipped.append('%s <- %s 形状 %s vs %s'
                           % (key, phys, tuple(w.shape), tuple(tensor.shape)))
            continue
        with torch.no_grad():
            tensor.copy_(w.to(tensor.dtype))
        done += 1

    return done, missing, skipped + [u[1] for u in unresolved], pmap


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print('  用法见 by1load.load()。单独跑这个文件只做静态检查：')
        print()
        print('    python by1load.py <文件>.by1     # 只算映射，不加载')
        return 0
    bc = _load('by1check')
    _r, info = bc.check(sys.argv[1])
    pmap, unresolved = plan(info)
    print()
    print('  %s：算出 %d 个 (层, 机制, 逻辑名) -> 物理名'
          % (sys.argv[1], len(pmap)))
    for k in list(sorted(pmap))[:6]:
        print('    L%-3d %-8s %-24s -> %s' % (k[0], k[1], k[2], pmap[k]))
    if len(pmap) > 6:
        print('    …共 %d 个' % len(pmap))
    if unresolved:
        print()
        print('  **推不出来的 %d 个**（融合存储 / 布局 / 量化 —— 契约里没有）：'
              % len(unresolved))
        for mech, why in unresolved[:6]:
            print('    %-10s %s' % (mech, why))
    return 0


if __name__ == '__main__':
    sys.exit(main())
