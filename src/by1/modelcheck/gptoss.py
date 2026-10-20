#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gptoss -- GPT-OSS 的三处易错点：**注意力 sink / 滑窗 / YaRN**。
参照物：transformers 的 GptOssAttention / GptOssRotaryEmbedding

## 为什么值得单独一个判卷人

这三处都是"写错了照样跑、只是数字偏一点"的东西：

  ① **sink**   每个头一个可学标量，作为**额外一列 logit** 参与 softmax、之后丢掉。
               它改变归一化分母，所以不是"多看一眼"；
  ② **滑窗**   交替层里每 K 层是 sliding_attention，掩码是 `j > i - W` 与因果的交；
               少减一个 W、或者把 `>` 写成 `>=`，只有边界上的 token 会错；
  ③ **YaRN**   频率缩放 + `attention_factor`（0.1*ln(factor)+1）。后者**乘进 cos/sin**，
               漏了它整体幅度就偏 7%。

三处在 by1 的三个后端里都有实现（`sink = per_head` / `window` / `rope_scale`），
而在这之前**一个判卷人都没有**。

## 判据（每一条都要有可证伪的对照）

  ① 同一批权重（sink 非零）-> 官方 `GptOssAttention` 逐位一致；
     把 sink 关掉必须变；两边 sink 清零 != 没有 sink（分母多一个 e^0）。
  ② 带滑窗掩码 -> 逐位一致；把 by1 的 window 去掉必须变。
  ③ 带 YaRN（含 attention_factor）-> 逐位一致；退回普通 rope 必须变。

## 踩过的坑（写在这里，免得下一个人再踩）

  · 官方 attention **要显式因果掩码**：传 None 它不做因果（实测差 0.65）；
  · `GptOssConfig` **默认带 YaRN**（factor 32）：两边 rope 不同源时比出来的差说明不了谁错；
  · 官方 `sinks` 是 `torch.empty`（独立构造未初始化，实测全零）—— 必须自己给权重，
    否则 sink 是常数、恒等测不出来。
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import sys  # noqa: E402
import torch  # noqa: E402
import by1codegen as cg  # noqa: E402
import by1skip  # noqa: E402

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention, GptOssRotaryEmbedding)
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except ImportError:
    GptOssAttention = GptOssRotaryEmbedding = None
    GptOssConfig = ROPE_INIT_FUNCTIONS = None

D, NH, KV, HD, SEQ = 64, 4, 2, 16, 12
WINDOW = 4


def _cfg(rope_parameters, layer_types):
    cfg = GptOssConfig(
        hidden_size=D, num_attention_heads=NH, num_key_value_heads=KV,
        head_dim=HD, num_hidden_layers=len(layer_types), num_local_experts=2,
        num_experts_per_tok=1, intermediate_size=64,
        max_position_embeddings=64, rope_theta=10000.0, sliding_window=WINDOW,
        layer_types=list(layer_types), attention_bias=False,
        attention_dropout=0.0, rope_parameters=rope_parameters)
    cfg._attn_implementation = "eager"
    return cfg


def _by1_attn(ns, sink=True, window=None, yarn=None, rope_scale=1.0):
    a = {"q": NH, "kv": KV, "head_dim": HD, "v": HD, "bias": False,
         "out_dim": NH * HD, "window": window, "rope_base": 10000,
         "rope_pairing": "half", "qk_norm": "off", "sink": sink,
         "rope_scale": rope_scale, "yarn": yarn, "head_gate": "off"}
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
    return target.load_state_dict(sd, strict=False)


def _mask(seq, window=None):
    """因果掩码（可选滑窗）：不许看的位置是 -inf。"""
    idx = torch.arange(seq)
    keep = idx[None, :] <= idx[:, None]
    if window:
        keep = keep & (idx[None, :] > idx[:, None] - window)
    m = torch.zeros(1, 1, seq, seq)
    return m.masked_fill(~keep[None, None], float("-inf"))


def _rel(a, b):
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-12)


def main():
    if GptOssAttention is None:
        return by1skip.skip("找不到 transformers 的 gpt_oss —— 这一项在这台机器上验不了")
    torch.manual_seed(0)
    ns = {}
    exec(cg.RUNTIME, ns)
    x = torch.randn(1, SEQ, D)
    pos = torch.arange(SEQ)[None]
    ok_all = True

    print("=" * 76)
    print("  GPT-OSS 三处易错点   by1 的 Attention  vs  transformers 的 GptOssAttention")
    print("=" * 76)
    print("\n  维度: d=%d  q=%d  kv=%d  head_dim=%d  seq=%d  window=%d"
          % (D, NH, KV, HD, SEQ, WINDOW))

    cfg = _cfg({"rope_type": "default"}, ["full_attention"])
    ref = GptOssAttention(cfg, 0).eval()
    with torch.no_grad():
        ref.sinks.copy_(torch.randn(NH) * 0.5)
    rrope = GptOssRotaryEmbedding(cfg).eval()
    with torch.no_grad():
        cos, sin = rrope(x, pos)
        r_out = ref(x, (cos, sin), _mask(SEQ))[0]
    mine = _by1_attn(ns)
    _load(mine, ref, torch)
    with torch.no_grad():
        m_out = mine(x)
    dd, rel = _rel(m_out, r_out)
    print("\n  ① sink（非零）: 绝对 %.3e  相对 %.3e  %s"
          % (dd, rel, "逐位一致 ok" if rel < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel < 1e-6

    no_sink = _by1_attn(ns, sink=False)
    _load(no_sink, ref, torch, with_sink=False)
    with torch.no_grad():
        n_out = no_sink(x)
    _, rel_off = _rel(n_out, m_out)
    print("     关掉 sink: 相对 %.3e  %s"
          % (rel_off, "（sink 确实参与）" if rel_off > 1e-3
             else "**（!! sink 没起作用）**"))
    ok_all = ok_all and rel_off > 1e-3
    with torch.no_grad():
        ref.sinks.zero_()
        _load(mine, ref, torch)
        r0 = ref(x, (cos, sin), _mask(SEQ))[0]
        m0 = mine(x)
    _, rel_zero_vs_none = _rel(m0, n_out)
    _, rel_zero_vs_ref = _rel(m0, r0)
    print("     清零 vs 没有 sink: 相对 %.3e  %s"
          % (rel_zero_vs_none,
             "（确实不等价 —— 分母多一个 e^0）" if rel_zero_vs_none > 1e-3
             else "**（竟然一样：sink 没真正参与）**"))
    print("     清零后 vs 官方: 相对 %.3e  %s"
          % (rel_zero_vs_ref, "ok" if rel_zero_vs_ref < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel_zero_vs_none > 1e-3 and rel_zero_vs_ref < 1e-6

    cfg2 = _cfg({"rope_type": "default"},
                ["full_attention", "sliding_attention"])
    ref2 = GptOssAttention(cfg2, 1).eval()
    with torch.no_grad():
        ref2.sinks.copy_(torch.randn(NH) * 0.5)
    win_mask = _mask(SEQ, WINDOW)
    with torch.no_grad():
        r2 = ref2(x, (cos, sin), win_mask)[0]
    mine2 = _by1_attn(ns, window=WINDOW)
    _load(mine2, ref2, torch)
    with torch.no_grad():
        m2 = mine2(x)
    dd, rel = _rel(m2, r2)
    print("\n  ② 滑窗 W=%d: 绝对 %.3e  相对 %.3e  %s"
          % (WINDOW, dd, rel, "逐位一致 ok" if rel < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel < 1e-6
    mine_no_win = _by1_attn(ns, window=None)
    _load(mine_no_win, ref2, torch)
    with torch.no_grad():
        m2_no = mine_no_win(x)
    _, rel_nw = _rel(m2_no, m2)
    print("     去掉 window: 相对 %.3e  %s"
          % (rel_nw, "（滑窗确实在起作用）" if rel_nw > 1e-3
             else "**（!! window 被忽略）**"))
    ok_all = ok_all and rel_nw > 1e-3

    yarn_params = {"rope_type": "yarn", "factor": 2.0,
                   "original_max_position_embeddings": 32,
                   "beta_fast": 32.0, "beta_slow": 1.0, "truncate": True,
                   "attention_factor": None}
    cfg3 = _cfg(yarn_params, ["full_attention"])
    _inv, attn_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg3)
    ref3 = GptOssAttention(cfg3, 0).eval()
    with torch.no_grad():
        ref3.sinks.copy_(torch.randn(NH) * 0.5)
    rrope3 = GptOssRotaryEmbedding(cfg3).eval()
    with torch.no_grad():
        cos3, sin3 = rrope3(x, pos)
        r3 = ref3(x, (cos3, sin3), _mask(SEQ))[0]
    mine3 = _by1_attn(ns, yarn={"type": "yarn", "factor": 2.0, "original": 32,
                                "beta_fast": 32.0, "beta_slow": 1.0,
                                "truncate": True},
                      rope_scale=float(attn_factor))
    _load(mine3, ref3, torch)
    with torch.no_grad():
        m3 = mine3(x)
    dd, rel = _rel(m3, r3)
    print("\n  ③ YaRN(factor=2, attention_factor=%.4f): 绝对 %.3e  相对 %.3e  %s"
          % (attn_factor, dd, rel, "逐位一致 ok" if rel < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel < 1e-6
    mine_plain = _by1_attn(ns)
    _load(mine_plain, ref3, torch)
    with torch.no_grad():
        m3_plain = mine_plain(x)
    _, rel_plain = _rel(m3_plain, m3)
    print("     退回普通 rope: 相对 %.3e  %s"
          % (rel_plain, "（YaRN 确实在起作用）" if rel_plain > 1e-3
             else "**（!! YaRN 被忽略）**"))
    ok_all = ok_all and rel_plain > 1e-3

    print("\n  [%s] GPT-OSS 三处（sink / 滑窗 / YaRN）%s"
          % ("PASS" if ok_all else "FAIL",
             "各一条正例 + 一条可证伪，全过" if ok_all else "有问题"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())