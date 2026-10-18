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

参照物：transformers 的 `GemmaMLP`（models/gemma/modeling_gemma.py，GeGLU）、
`NemotronMLP`（models/nemotron/modeling_nemotron.py，无门控·relu2）、
`GptOssExperts._apply_gate`（models/gpt_oss/modeling_gpt_oss.py，clamped swiglu）、
`LlamaMLP`（models/llama/modeling_llama.py，silu）。
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


def _official_nemotron():
    """官方的无门控 MLP（`NemotronMLP`：up_proj -> act_fn -> down_proj，

    `hidden_act` 默认就是 `relu2`）。对应 by1 的 `gate=False`。
    """
    from transformers import NemotronConfig
    from transformers.models.nemotron.modeling_nemotron import NemotronMLP
    cfg = NemotronConfig(hidden_size=D, intermediate_size=H)
    return NemotronMLP(cfg)


def _official_gptoss_gate(gate, up, limit=7.0, alpha=1.702):
    """官方 GPT-OSS 的 clamped swiglu：**直接调 GptOssExperts._apply_gate**。

    官方那四行（modeling_gpt_oss.py）：
        gate = gate.clamp(min=None, max=self.limit)
        up   = up.clamp(min=-self.limit, max=self.limit)
        glu  = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu
    它吃的是交错排布的 gate_up（[..., ::2] 是 gate、[..., 1::2] 是 up），所以这边
    按同样的布局拼一份送进去 —— **不自己重写公式**，只用官方那个模块。
    """
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    cfg = GptOssConfig(hidden_size=D, num_experts=1, swiglu_limit=limit,
                       swiglu_alpha=alpha, num_experts_per_tok=1)
    experts = GptOssExperts(cfg)
    inter = torch.stack([gate, up], dim=-1).flatten(-2)   # [..., 2*H] 交错
    return experts._apply_gate(inter)


def _by1_swiglu():
    """by1 RUNTIME 里的 `_swiglu`（FFN 门控支用的就是它）。"""
    if not _NS:
        import by1codegen as cg
        exec(cg.RUNTIME, _NS)
    return _NS["_swiglu"]

def _tie_ungated(m, ref):
    """无门控那一支：w1->up_proj, w2->down_proj。"""
    with torch.no_grad():
        ref.up_proj.weight.copy_(m.w1.weight)
        ref.down_proj.weight.copy_(m.w2.weight)

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

    # ---- ④ 正例：relu2（Nemotron）对官方 NemotronMLP ----
    # 这也是历史上被静默算成 silu 的取值之一。
    mu = _by1_mlp(gate=False, hidden=H, act="relu2")
    ref_n = _official_nemotron()
    _tie_ungated(mu, ref_n)
    yu = mu(x).detach()
    yn = ref_n(x).detach()
    d4 = _diff(yu, yn)
    print("  ④ relu2      by1 vs 官方 NemotronMLP 最大差 %.3e  [%s]"
          % (d4, "PASS" if d4 < 1e-6 else "FAIL"))
    if d4 >= 1e-6:
        bad.append("relu2 与官方 NemotronMLP 不一致（%.3e）" % d4)

    # ---- ⑤ 可证伪：relu2 写成 silu 就不是它 ----
    ms2 = _by1_mlp(gate=False, hidden=H, act="silu")
    with torch.no_grad():
        ms2.w1.weight.copy_(mu.w1.weight)
        ms2.w2.weight.copy_(mu.w2.weight)
    d5 = _diff(ms2(x).detach(), yn)
    print("  ⑤ 证伪 无门控 act=silu     vs 官方差       %.3e  [%s]"
          % (d5, "PASS" if d5 > 1e-3 else "FAIL"))
    if d5 <= 1e-3:
        bad.append("无门控 act=silu 竟然和 relu2 一样（%.3e）—— 分派没生效" % d5)

    # ---- ⑥ 正例：gptoss 的 clamped swiglu（_swiglu 的第三个分支）----
    sw = _by1_swiglu()
    gg = torch.randn(2, 3, H) * 4.0       # *4 是为了让一部分值越过 limit=7
    gu = torch.randn(2, 3, H) * 4.0
    y6 = sw(gg, gu, "gptoss", 7.0, 1.702).detach()
    r6 = _official_gptoss_gate(gg, gu).detach()
    d6 = _diff(y6, r6)
    print("  ⑥ gptoss    by1 _swiglu vs 官方 _apply_gate    %.3e  [%s]"
          % (d6, "PASS" if d6 < 1e-6 else "FAIL"))
    if d6 >= 1e-6:
        bad.append("gptoss 分支与官方 _apply_gate 不一致（%.3e）" % d6)

    # ---- ⑦ 可证伪：不 clamp 就不是它（输入必须越过 limit）----
    #
    # **用 by1 自己的路径**（`limit = None` -> 那一支不发 clamp），不是在判卷人里
    # 手写一遍公式 —— 手写等于只测自己的转写，测不到实现里的 if 分支。
    noclamp = sw(gg, gu, "gptoss", None, 1.702).detach()
    d7 = _diff(noclamp, r6)
    print("  ⑦ 证伪 by1 limit=None   与官方差         %.3e  [%s]"
          % (d7, "PASS" if d7 > 1e-2 else "FAIL"))
    if d7 <= 1e-2:
        bad.append("去掉 clamp 竟然和官方一样（%.3e）—— 输入没越过 limit，这条判据是空的" % d7)

    # ---- ⑧ 正例：silu（门控支最常见的取值）对官方 LlamaMLP ----
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    lcfg = LlamaConfig(hidden_size=D, intermediate_size=H, num_attention_heads=4,
                       num_key_value_heads=2, num_hidden_layers=1,
                       hidden_act="silu")
    ref_l = LlamaMLP(lcfg).eval()
    m_silu = _by1_mlp(gate=True, hidden=H, act="silu")
    with torch.no_grad():
        ref_l.gate_proj.weight.copy_(m_silu.w1.weight)
        ref_l.up_proj.weight.copy_(m_silu.w3.weight)
        ref_l.down_proj.weight.copy_(m_silu.w2.weight)
    y8 = m_silu(x).detach()
    r8 = ref_l(x).detach()
    d8 = _diff(y8, r8)
    print("  ⑧ silu       by1 vs 官方 LlamaMLP        最大差 %.3e  [%s]"
          % (d8, "PASS" if d8 < 1e-6 else "FAIL"))
    if d8 >= 1e-6:
        bad.append("silu 与官方 LlamaMLP 不一致（%.3e）" % d8)

    # ---- ⑨ 同一支的另一个方向：by1 发 clamp 与不发 clamp 必须差 ----
    d9 = _diff(noclamp, y6)
    print("  ⑨ limit=None vs limit=7.0 by1 自比      %.3e  [%s]"
          % (d9, "PASS" if d9 > 1e-2 else "FAIL"))
    if d9 <= 1e-2:
        bad.append("by1 的 limit=None 与 limit=7.0 算出一模一样（%.3e）—— limit 没生效" % d9)

    for b in bad:
        print("     ! " + b)
    print("  [%s] FFN 激活分派：GeGLU %.2e、relu2 %.2e、gptoss %.2e、silu %.2e；"
          "五条可证伪都差得出来（含 by1 自己的 limit 开关）"
          % ("PASS" if not bad else "FAIL", d1, d4, d6, d8))
    return 0 if not bad else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())