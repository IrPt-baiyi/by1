#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gptoss -- **注意力 sink** 的判卷人：by1 的 Attention(sink) vs 官方 GptOssAttention。

## 为什么值得单独一个判卷人

sink 是**每个头一个可学标量**，作为**额外一列 logit** 参与 softmax、之后丢掉：

    att = [q·k / √d  |  sink]  →  softmax(-1)[..., :-1]

这一列**改变归一化分母**，所以它不是"多看一眼"那种可有可无的东西 —— 但写错了
（比如 softmax 之后才加、或者把 sink 也乘进输出）在多数样本上**看起来正常**，
只是数值偏一点。by1 的三个后端都有 sink（`sink = per_head`），
而在这之前**没有一个判卷人**。

## 验什么

  ① 同一批权重（含非零 sink）下，by1 `Attention` 与官方逐位一致；
  ② 把 sink 关掉**必须**让输出变 —— 否则这一列根本没参与；
  ③ 把两边的 sink 都置零 -> 应当和"没有 sink"的 softmax 一致（钉住语义）。
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import sys  # noqa: E402
import torch  # noqa: E402
import by1codegen as cg  # noqa: E402

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention, GptOssRotaryEmbedding)
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
except ImportError:
    GptOssAttention = GptOssRotaryEmbedding = GptOssConfig = None

import by1skip  # noqa: E402

D, NH, KV, HD, SEQ = 64, 4, 2, 16, 12


def _build():
    cfg = GptOssConfig(
        hidden_size=D, num_attention_heads=NH, num_key_value_heads=KV,
        head_dim=HD, num_hidden_layers=1, num_local_experts=2,
        num_experts_per_tok=1, intermediate_size=64, max_position_embeddings=64,
        rope_theta=10000.0, layer_types=["full_attention"],
        # **默认带 YaRN**（factor=32）：两边 rope 不同源的话，比出来的差说明不了谁错。
        rope_parameters={"rope_type": "default"},
        attention_bias=False, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return cfg


def _by1_attn(ns, sink=True, window=None):
    a = {"q": NH, "kv": KV, "head_dim": HD, "v": HD, "bias": False,
         "out_dim": NH * HD, "window": window, "rope_base": 10000,
         "rope_pairing": "half", "qk_norm": "off", "sink": sink,
         "head_gate": "off"}
    return ns["Attention"](a, D).eval()


def _load(target, ref, torch, with_sink=True):
    sd = {
        "wq.weight": ref.q_proj.weight.detach().clone(),
        "wk.weight": ref.k_proj.weight.detach().clone(),
        "wv.weight": ref.v_proj.weight.detach().clone(),
        "wo.weight": ref.o_proj.weight.detach().clone(),
    }
    if with_sink:
        sd["sink"] = ref.sinks.detach().clone()
    miss, unexp = target.load_state_dict(sd, strict=False)
    return miss, unexp


def main():
    if GptOssAttention is None:
        return by1skip.skip("找不到 transformers 的 gpt_oss —— 这一项在这台机器上验不了")
    torch.manual_seed(0)
    cfg = _build()
    ref = GptOssAttention(cfg, 0).eval()
    # **官方 `sinks` 是 `torch.empty`** —— 独立构造时没初始化，实测可能全零，
    # 那样 sink 那一列等于常数、"关掉 sink"也就测不出来（MoE 判卷人踩过同一个坑）。
    with torch.no_grad():
        ref.sinks.copy_(torch.randn(NH) * 0.5)
    rrope = GptOssRotaryEmbedding(cfg).eval()

    ns = {}
    exec(cg.RUNTIME, ns)
    mine = _by1_attn(ns)
    _load(mine, ref, torch)
    x = torch.randn(1, SEQ, D)
    pos = torch.arange(SEQ)[None]
    # **官方要显式因果掩码**：传 None 时它不做因果（实测与带掩码差 0.65），
    # 那不是"两种等价写法"，是**算了两件事**。
    mask = torch.full((1, 1, SEQ, SEQ), float("-inf")).triu(1)
    with torch.no_grad():
        cos, sin = rrope(x, pos)
        r_out = ref(x, (cos, sin), mask)[0]
        m_out = mine(x)
    dd = (r_out - m_out).abs().max().item()
    rel = dd / max(r_out.abs().max().item(), 1e-12)

    print("=" * 74)
    print("  注意力 sink   by1 的 Attention  vs  transformers 的 GptOssAttention")
    print("=" * 74)
    print("\n  维度: d=%d  q=%d  kv=%d  head_dim=%d  seq=%d  sink=%r"
          % (D, NH, KV, HD, SEQ, ref.sinks.tolist()))
    print("\n  ① 同一批权重（sink 非零）:")
    print("     绝对差 %.3e  相对 %.3e  %s"
          % (dd, rel, "逐位一致 ok" if rel < 1e-6 else "**对不上**"))
    ok = rel < 1e-6

    # ② sink 关掉必须变
    no_sink = _by1_attn(ns, sink=False)
    _load(no_sink, ref, torch, with_sink=False)
    with torch.no_grad():
        n_out = no_sink(x)
    d2 = (n_out - m_out).abs().max().item()
    rel2 = d2 / max(m_out.abs().max().item(), 1e-12)
    print("\n  ② 把 sink 关掉:")
    print("     与带 sink 的相对差 %.3e  %s"
          % (rel2, "（sink 确实参与计算）" if rel2 > 1e-3 else "**（!! sink 没起作用）**"))
    ok = ok and rel2 > 1e-3

    # ③ 两边 sink 都清零 -> 等于没有 sink 的 softmax
    with torch.no_grad():
        ref.sinks.zero_()
        _load(mine, ref, torch)
        r0 = ref(x, (cos, sin), mask)[0]
        m0 = mine(x)
        n0 = no_sink(x)
    d3 = (m0 - n0).abs().max().item()
    rel3 = d3 / max(m0.abs().max().item(), 1e-12)
    print("\n  ③ 两边 sink 清零:")
    # **sink 清零不等于没有 sink**：它仍在 softmax 的分母里多添一个 e^0 = 1。
    # 这一条我第一版猜错了（以为清零就等价）；现在它变成一条可证伪的判据。
    print("     与\"没有 sink\"的相对差 %.3e  %s"
          % (rel3, "（确实不等价 —— 分母多了一个 e^0）" if rel3 > 1e-3
             else "**竟然一样：那 sink 没真正参与**"))
    ok = ok and rel3 > 1e-3
    d4 = (r0 - m0).abs().max().item()
    rel4 = d4 / max(r0.abs().max().item(), 1e-12)
    print("     [调试] abs=%.3e  |r0|=%.3e  |m0|=%.3e"
          % (d4, r0.abs().max().item(), m0.abs().max().item()))
    print("     与官方（清零后）相对差 %.3e  %s"
          % (rel4, "ok" if rel4 < 1e-6 else "**对不上**"))
    ok = ok and rel4 < 1e-6

    print("\n  [%s] 注意力 sink %s"
          % ("PASS" if ok else "FAIL", "三问都过" if ok else "有问题"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())