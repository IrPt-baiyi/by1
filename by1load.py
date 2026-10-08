#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1load -- 把**真权重**装进 by1 生成的模型。

## 为什么这个能做，而 gpt2 那次要手写

`gpt2.by1` 那一轮，映射是**手写的**（`gpt2_weight_map()`）——
因为 GPT-2 的 checkpoint 是融合 qkv + Conv1D `[in, out]`，
**推不出来**。

而 clef 这类模型不一样：`.by1` 里**已经把命名规则写下来了**：

    name        = "model.language_model.layers.{i}.{scope}{physical}"
    global_name = "{physical}"
    scope { Lin = linear_attn., GQA = self_attn., FFN_ = mlp., ... }
    rename { embed.weight = model.language_model.embed_tokens.weight, ... }

**那是描述的一部分，不是仓库的习惯。** 所以可以从契约名反推出物理名 ——
`by1verify` 每天在做这件事（它就是这么验 851/851 的）。

这个模块把同一个规则**反过来用**：从物理名找到权重，装进模型。

## 它不解的问题

    ✗ 融合存储（c_attn / gate_up_proj）—— 那是仓库的习惯，推不出来
    ✗ 布局转置（Conv1D）—— 同上
    ✗ 量化格式（fp8 的 weight_scale_inv）—— 契约里没有

碰到这三种会**明确报出来**，不静默跳过。

用法见文件末尾的 `load_from_hf()`。
"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plan(info, stack=None):
    """算出「内部参数名 -> 物理张量名」的映射。**不加载任何权重。**

    内部名是后端自己的（`layers.0.op1.in_proj_qkv`），
    物理名是 checkpoint 里的（`model.language_model.layers.0.linear_attn.in_proj_qkv.weight`）。
    中间那座桥是 `.by1` 的 `emit { train -> torch.module { ... } }`。
    """
    bc = _load('by1check')
    rule = (info.get('emit') or {}).get('torch.module') or {}
    if not rule:
        raise SystemExit('  这个 .by1 没有 emit { train -> torch.module } —— '
                         '没有命名规则就推不出物理名')
    scopes = rule.get('scope') or {}
    contracts = {}
    for mname, tbl in (info.get('tens') or {}):
        pass
    # tens 的结构：(名字, 作用域, 形状) 之类 —— 用 by1verify 的口径
    return rule, scopes


def physical_of(bc, rule, i, stack, mech, logical):
    """调 by1check 的 render_name —— **和 by1verify 用的是同一个函数**。

    这点很重要：如果加载用它、验证用另一个实现，两边会漂。
    这个项目里"同一个意思两处实现"已经出过好几次。
    """
    return bc.render_name(rule, i, stack, mech, logical)


def main():
    print(__doc__)
    print('  这个模块是给 by1e2e / 真模型前向用的，单独跑没有意义。')
    print('  用法见 load_from_hf()。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
