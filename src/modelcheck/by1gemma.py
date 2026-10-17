#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gemma -- **GeGLU 那一支 FFN**：门控 + tanh 近似激活，对官方 GemmaMLP。

## 为什么单开一条

by1 的 MLP（RUNTIME）有两条支：`gate = true` 的 SwiGLU 与 `gate = none` 的直连。
门控那一支的激活**原来写死 F.silu**，`act = relu2` / `gelu` 全被静默算成 silu
（见 by1codegen._swiglu 的注释）。现在按 style 分派了，但**没有判据钉住 GeGLU**：

  · act = silu      -> SwiGLU（Llama / Qwen 那一类）
  · act = gelu_new  -> **GeGLU**（Gemma 那一类；HF 的 gelu_pytorch_tanh）
  · act = gelu      -> 精确 erf 式，**和 tanh 近似差 1e-3 量级，不能混**

这条判卷人把三件事分开验：正例（GeGLU 对官方 GemmaMLP 逐值一致）+ 两条可证伪
（写成 silu 要差、写成精确 gelu 也要差）。

参照物：transformers 的 `GemmaMLP`（models/gemma/modeling_gemma.py）：
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`，`act_fn = ACT2FN["gelu_pytorch_tanh"]`
（= tanh 近似）。本机 5.20.0 实测可取。
"""

import torch

torch.manual_seed(0)

D, H, T = 16, 32, 5


_NS = {}


def _by1_mlp(**attrs):
    """by1 的 RUNTIME MLP（从 cg.RUNTIME 的源码执行出来）。"""
    import by1codegen as cg
    if not _NS:
        exec(cg.RUNTIME, _NS)
    return _NS["MLP"](dict(attrs), D)



def _official():
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaMLP
    cfg = GemmaConfig(hidden_size=D, intermediate_size=H)
    return GemmaMLP(cfg)


def _tie(m, ref):
    """把 by1 的三个权重装进官方模块：w1->gate_proj, w3->up_proj, w2->down_proj。"""
    with torch.no_grad():
        ref.gate_proj.weight.copy_(m.w1.weight)
        ref.up_proj.weight.copy_(m.w3.weight)
        ref.down_proj.weight.copy_(m.w2.weight)


def _diff(a, b):
    return float((a - b).abs().max())


def main():
    bad = []
    x = torch.randn(1, T, D)

    # ---- ① 正例：GeGLU 对官方 ----
    m = _by1_mlp(gate=True, hidden=H, act="gelu_new")
    ref = _official()
    _tie(m, ref)
    y1 = m(x).detach()
    y2 = ref(x).detach()
    d1 = _diff(y1, y2)
    print("  \u2460 GeGLU      by1 vs \u5b98\u65b9 GemmaMLP  \u6700\u5927\u5dee %.3e  [%s]"
          % (d1, "PASS" if d1 < 1e-6 else "FAIL"))
    if d1 >= 1e-6:
        bad.append("GeGLU \u4e0e\u5b98\u65b9 GemmaMLP \u4e0d\u4e00\u81f4\uff08%.3e\uff09" % d1)

    # ---- ② 可证伪：写成 silu 就不是 GeGLU ----
    ms = _by1_mlp(gate=True, hidden=H, act="silu")
    with torch.no_grad():
        ms.w1.weight.copy_(m.w1.weight)
        ms.w2.weight.copy_(m.w2.weight)
        ms.w3.weight.copy_(m.w3.weight)
    d2 = _diff(ms(x).detach(), y2)
    print("  \u2461 \u8bc1\u4f2a act=silu          \u4e0e\u5b98\u65b9\u5dee    %.3e  [%s]"
          % (d2, "PASS" if d2 > 1e-3 else "FAIL"))
    if d2 <= 1e-3:
        bad.append("act=silu \u7adf\u7136\u548c GeGLU \u4e00\u6837\uff08%.3e\uff09\u2014\u2014 \u5206\u6d3e\u6ca1\u751f\u6548" % d2)

    # ---- ③ 可证伪：精确 gelu 与 tanh 近似不能混 ----
    me = _by1_mlp(gate=True, hidden=H, act="gelu")
    with torch.no_grad():
        me.w1.weight.copy_(m.w1.weight)
        me.w2.weight.copy_(m.w2.weight)
        me.w3.weight.copy_(m.w3.weight)
    d3 = _diff(me(x).detach(), y1)
    print("  \u2462 \u8bc1\u4f2a act=gelu\uff08\u7cbe\u786e\uff09  \u4e0e gelu_new  %.3e  [%s]"
          % (d3, "PASS" if d3 > 1e-4 else "FAIL"))
    if d3 <= 1e-4:
        bad.append("\u7cbe\u786e gelu \u4e0e tanh \u8fd1\u4f3c\u5206\u4e0d\u5f00\uff08%.3e\uff09\u2014\u2014 \u4e24\u5f0f\u88ab\u6df7\u6210\u4e00\u4e2a" % d3)

    for b in bad:
        print("     ! " + b)
    print("  [%s] GeGLU\uff1a\u6b63\u4f8b\u9010\u503c\u4e00\u81f4\uff08%.2e\uff09\uff0csilu / \u7cbe\u786e gelu \u4e24\u6761\u53ef\u8bc1\u4f2a\u90fd\u5dee\u5f97\u51fa\u6765"
          % ("PASS" if not bad else "FAIL", d1))
    return 0 if not bad else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())