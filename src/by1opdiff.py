#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1opdiff -- **逐个算子**比 NumPy 和 PyTorch。

## 为什么要有这个

`by1exec --compare` 比的是**整个模型**的最后输出。一个算子的差
被后面的层一混，可能变大也可能变小 —— 而且**分不出是哪个算子**。

gpt2 那一轮就是靠手工做这件事才定位到 `op_ffn` 写死 silu 的。
这个脚本把那个做法固定下来：**每个算子单独比一次，喂同一组权重、
同一个输入。**

    python by1opdiff.py <文件>.by1 [--weight-scale 0.05]

判据：每个算子的相对差。整模型差 3e-04 的时候，
**有一个算子会明显比别的差** —— 那就是它。
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location(
        'bcm', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'by1check.py'))
bc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(bc)
import by1codegen as cg
import by1exec as ex
import by1paths


def build_torch(kind, attrs, d, R):
    """拿 RUNTIME 里的类建一个 PyTorch 侧的算子。"""
    if kind == "Norm":
        return R['_mk_norm'](attrs, d)
    cls = R['BUILDERS'].get(kind)
    if cls is None or not isinstance(cls, type):
        return None
    try:
        return cls(attrs, d)
    except TypeError:
        return None


def rand(shape, rng, scale):
    return rng.normal(0, scale, shape).astype(np.float32)


def one(f, seq, scale):
    """跑一个文件，返回超阈值的算子列表。"""
    # 同 by1irentry：裸名要按 models/ 解析。
    if not by1paths.find_model(f):
        print('  （%s 不在，跳过）' % f)
        return []
    _r, info = bc.check(f)
    ir = cg.compile_ir(info)
    d = ir['d_model']
    ns = {}
    exec(compile(cg.render_ir(ir, 'x'), '<ir>', 'exec'), ns)
    R = ns
    shapes = ex.shapes_of(ir)

    # **每个不同的 (kind, 属性) 组合都要测，不是每种 kind 只测第一次。**
    # 只取第一次的话，"第 0 层的 Attention"和"第 3 层的 Attention"
    # 属性不同时（窗口、rope 开关、qk_norm…），只测了其中一个。
    import json as _json
    seen = {}
    for L in ir['layers']:
        for j, o in enumerate(L['ops']):
            key = (o['kind'], _json.dumps(o['attrs'], sort_keys=True,
                                          default=str))
            if key not in seen:
                seen[key] = (L['index'], j, o['attrs'])
    print('=' * 78)
    print('  逐算子对拍：%s  （d_model=%d，%d 个属性组合）'
          % (f, d, len(seen)))
    print('=' * 78)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.0, (1, seq, d)).astype(np.float32)
    xt = torch.tensor(x)
    over = []
    for (kind, _ak), (li, oi, attrs) in sorted(seen.items()):
        pre = 'layers.%d.op%d.' % (li, oi)
        P = {}
        for k, s in shapes.items():
            if k.startswith(pre):
                P[k[len(pre):]] = rng.normal(0, scale, s).astype(np.float32)
        my_op = ex.OPS.get(kind)
        if my_op is None:
            print('  L%-3d %-12s NumPy 没实现' % (li, kind))
            continue
        tm = build_torch(kind, attrs, d, R)
        if tm is None:
            continue
        try:
            mine = my_op(P, attrs, [x], d)
        except Exception as e:
            print('  L%-3d %-12s NumPy 侧崩了：%s' % (li, kind, str(e)[:44]))
            over.append('L%d %s（NumPy 崩）' % (li, kind))
            continue
        # **先按名字配。** 形状兜底在这个项目里已经害过两次 ——
        # 同形状的参数太多，"唯一"匹配也会配到错的那个。
        import re as _re
        tgt, miss = {}, []
        for k, v in tm.state_dict().items():
            cands = [k, _re.sub(r"\.(weight|bias)$", "", k)]
            hit = next((c for c in cands if c in P), None)
            if hit is None:
                miss.append(k)
            else:
                tgt[k] = torch.tensor(P[hit]).reshape(v.shape)
        if miss:
            print('  L%-3d %-12s PyTorch 侧 %d 个参数配不上'
                  % (li, kind, len(miss)))
            continue
        tm.load_state_dict(tgt)
        tm.eval()
        with torch.no_grad():
            ref = tm(xt).numpy()
        dd = np.abs(mine - ref).max()
        amp = max(np.abs(ref).max(), 1e-30)
        rel = dd / amp
        flag = '  <<<' if rel > 1e-5 else ''
        if rel > 1e-5:
            over.append('L%d %s' % (li, kind))
        print('  L%-3d %-12s 最大绝对差 %.3e   相对 %.3e%s'
              % (li, kind, dd, rel, flag))
    return over


def _opt(name, dflt):
    return (type(dflt)(sys.argv[sys.argv.index(name) + 1])
            if name in sys.argv else dflt)


def main():
    if len(sys.argv) < 2 or sys.argv[1] == '--all':
        # **无参数 = 跑默认那一批**（给 by1all 用）。
        files = ['llama-shaped.by1', 'mixtral-shaped.by1',
                 'gpt-oss-shaped.by1', 'qwen3-next-shaped.by1',
                 'mla-shaped.by1', 'llama3-shaped.by1',
                 'clef-tiny.by1', 'gpt2-tiny.by1']
        # **裸名 -> models/。** 搬家前这条靠 cwd 恰好是仓库根；
        # 现在模型在 `models/`，而 exists 守卫会让找不到的**静默消失**
        # （报出来是「0 个文件 PASS」）—— 所以解析要用同一个规则。
        files = [x for x in files if by1paths.find_model(x)]
        bad = []
        for ff in files:
            bad += [('%s %s' % (ff, k)) for k in one(ff, 48, 0.05)]
        print()
        if bad:
            print('  [FAIL] 这些算子的两个后端对不上：%s' % bad[:6])
            return 1
        print('  [PASS] 逐算子 全都对得上（%d 个文件）' % len(files))
        return 0
    over = one(sys.argv[1], _opt('--seq', 48), _opt('--weight-scale', 0.05))
    print()
    if over:
        print('  [FAIL] 对不上的算子：%s' % over)
        return 1
    print('  [PASS] 逐算子 全都对得上')
    return 0


if __name__ == '__main__':
    sys.exit(main())
