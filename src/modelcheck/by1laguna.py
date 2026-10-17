#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1laguna -- **head gate** 的判卷人：by1 的 Attention  vs  官方 `LagunaAttention`。

## 为什么值得单独一个判卷人

`head_gate` 是"同名不同义"的典型：**Laguna 用 softplus、Ling 用 sigmoid**，
而两边的 `.by1` 写的是同一个 `head_gate = per_head`。

写法上最省事的是让激活"有个默认值"（by1 以前就是 `sigmoid`）+ 只在需要时改 ——
可一旦某个模型忘了写 `gate_act`，它就会**静默算成另一个模型的注意力**：
数值只差百分之几，前向照样跑完。

## 验什么（每一条都要有可证伪的对照）

  ① 同一批权重（per-head gate + **softplus**）-> 官方逐位一致；
  ② 把 `gate_act` 换成 `sigmoid` -> **必须变**（这就是"同名不同义"的量化证据）；
  ③ 把 `head_gate` 关掉 -> 必须变；
  ④ 反过来验一次：`gating=False` 的官方（**full 宽度门**，它**不会取消门**）与 by1 `head_gate=full` 也应一致；
  ⑤ `rope_parameters` 的**按层参数**（base / part）也要对得上。
     否则"关掉"这条可能只是碰巧对上。
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
    from transformers.models.laguna.modeling_laguna import (
        LagunaAttention, LagunaRotaryEmbedding, LagunaConfig)
except ImportError:
    LagunaAttention = LagunaRotaryEmbedding = LagunaConfig = None

D, NH, KV, HD, SEQ = 64, 4, 2, 16, 10


def _cfg(gating, rope=None):
    cfg = LagunaConfig(
        hidden_size=D, num_attention_heads=NH, num_key_value_heads=KV,
        head_dim=HD, num_hidden_layers=1, intermediate_size=64,
        max_position_embeddings=64, rope_theta=10000.0,
        layer_types=["full_attention"], sliding_window=8,
        attention_bias=False, attention_dropout=0.0, gating=gating)
    if rope is not None:
        cfg.rope_parameters = rope
    cfg._attn_implementation = "eager"
    return cfg


def _by1_attn(ns, gate="per_head", act="softplus", rope=False,
               rope_base=10000, rope_partial=1.0):
    a = {"q": NH, "kv": KV, "head_dim": HD, "v": HD, "bias": False,
         "out_dim": NH * HD, "window": None,
         "rope_pairing": "half", "qk_norm": "per_head", "norm_eps": 1e-6,
         "rope": rope, "rope_base": rope_base,
         "rope_partial": rope_partial, "sink": False, "head_gate": gate,
         "gate_act": act}
    return ns["Attention"](a, D).eval()


def _load(mine, ref):
    sd = {
        "wq.weight": ref.q_proj.weight.detach().clone(),
        "wk.weight": ref.k_proj.weight.detach().clone(),
        "wv.weight": ref.v_proj.weight.detach().clone(),
        "wo.weight": ref.o_proj.weight.detach().clone(),
        "qn.w": ref.q_norm.weight.detach().clone(),
        "kn.w": ref.k_norm.weight.detach().clone(),
    }
    if getattr(ref, "g_proj", None) is not None:
        sd["g_proj.weight"] = ref.g_proj.weight.detach().clone()
    return mine.load_state_dict(sd, strict=False)


def _mask(seq):
    idx = torch.arange(seq)
    keep = idx[None, :] <= idx[:, None]
    m = torch.zeros(1, 1, seq, seq)
    return m.masked_fill(~keep[None, None], float("-inf"))


def _rel(a, b):
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-12)


def main():
    if LagunaAttention is None:
        return by1skip.skip("找不到 transformers 的 laguna —— 这一项在这台机器上验不了")
    torch.manual_seed(0)
    ns = {}
    exec(cg.RUNTIME, ns)
    x = torch.randn(1, SEQ, D)
    pos = torch.arange(SEQ)[None]
    mask = _mask(SEQ)
    ok_all = True
    # **rope 在 1~4 段用恒等**（把"门"这一个变量单独隔出来），
    # rope 自己在 ⑤/⑥ 两段单独验。上一轮说"两边频率对不上"，
    # 根因是我给构造函数传的 `rope_theta` 被按层的 `rope_parameters` 覆盖了。
    cos_id = torch.ones(1, SEQ, HD)
    sin_id = torch.zeros(1, SEQ, HD)

    print("=" * 74)
    print("  head gate   by1 的 Attention  vs  transformers 的 LagunaAttention")
    print("=" * 74)
    print("\n  维度: d=%d  q=%d  kv=%d  head_dim=%d  seq=%d  gate=%s"
          % (D, NH, KV, HD, SEQ, "per-head"))

    # ① 有门 + softplus
    cfg = _cfg("per-head")
    ref = LagunaAttention(cfg, 0, NH).eval()
    with torch.no_grad():
        r_out = ref(x, (cos_id, sin_id), mask)[0]
    mine = _by1_attn(ns)
    _load(mine, ref)
    with torch.no_grad():
        m_out = mine(x)
    dd, rel = _rel(m_out, r_out)
    print("\n  ① per-head + softplus: 绝对 %.3e  相对 %.3e  %s"
          % (dd, rel, "逐位一致 ok" if rel < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel < 1e-6

    # ② 换成 sigmoid（同名不同义）
    mine_sig = _by1_attn(ns, act="sigmoid")
    _load(mine_sig, ref)
    with torch.no_grad():
        m_sig = mine_sig(x)
    _, rel_sig = _rel(m_sig, m_out)
    print("  ② gate_act=sigmoid: 相对 %.3e  %s"
          % (rel_sig, "（softplus 与 sigmoid 确实不同）" if rel_sig > 1e-3
             else "**（!! 两种激活竟然一样）**"))
    ok_all = ok_all and rel_sig > 1e-3

    # ③ 关掉门
    mine_off = _by1_attn(ns, gate="off")
    _load(mine_off, ref)
    with torch.no_grad():
        m_off = mine_off(x)
    _, rel_off = _rel(m_off, m_out)
    print("  ③ head_gate=off: 相对 %.3e  %s"
          % (rel_off, "（门确实参与）" if rel_off > 1e-3 else "**（!! 门没起作用）**"))
    ok_all = ok_all and rel_off > 1e-3

    # ④ 反过来：`gating=False` 的官方 vs by1 的 **full 宽度门**
    #
    # 实测：官方 `LagunaAttention.forward` **无条件**做门控（`gating=False` 只把
    # `g_proj` 从 per-head 改成 full 宽度，并不取消门）。所以对照它的
    # 应该是 by1 的 `head_gate = full`，而不是 `off`。
    cfg2 = _cfg(False)
    ref2 = LagunaAttention(cfg2, 0, NH).eval()
    with torch.no_grad():
        r2 = ref2(x, (cos_id, sin_id), mask)[0]
    mine2 = _by1_attn(ns, gate="full")
    _load(mine2, ref2)
    with torch.no_grad():
        m2 = mine2(x)
    dd2, rel2 = _rel(m2, r2)
    print("  ④ 官方 gating=False（full 宽度门）vs by1 full: 绝对 %.3e  相对 %.3e  %s"
          % (dd2, rel2, "也一致 ok" if rel2 < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel2 < 1e-6

    # ⑤ rope 的**按层参数**（上一轮的待查项，这一轮销账）
    #
    # 实测：Laguna 的 `rope_parameters` **按层型分开**：full 层是
    # `rope_theta=500000` ＋ `partial_rotary_factor=0.5`，滑动层才是 10000 / 1.0。
    # 我上一轮给构造函数传的 `rope_theta=10000` **被它覆盖了**，
    # 于是以为"两边频率对不上"——其实是**我给错了参数**。
    rope_full = {"full_attention": {"rope_type": "default",
                                "rope_theta": 500000.0,
                                "partial_rotary_factor": 1.0},
                 "sliding_attention": {"rope_type": "default",
                                  "rope_theta": 10000.0,
                                  "partial_rotary_factor": 1.0}}
    cfg3 = _cfg("per-head", rope=rope_full)
    ref3 = LagunaAttention(cfg3, 0, NH).eval()
    rrope3 = LagunaRotaryEmbedding(cfg3).eval()
    with torch.no_grad():
        cos3, sin3 = rrope3(x, pos, "full_attention")
        r3 = ref3(x, (cos3, sin3), mask)[0]
    mine3 = _by1_attn(ns, rope=True, rope_base=500000)
    _load(mine3, ref3)
    with torch.no_grad():
        m3 = mine3(x)
    dd3, rel3 = _rel(m3, r3)
    print("\n  ⑤ rope 按层参数（base=500000）: 绝对 %.3e  相对 %.3e  %s"
          % (dd3, rel3, "逐位一致 ok" if rel3 < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel3 < 1e-6
    mine3_wrong = _by1_attn(ns, rope=True, rope_base=10000)
    _load(mine3_wrong, ref3)
    with torch.no_grad():
        m3w = mine3_wrong(x)
    _, rel3w = _rel(m3w, m3)
    print("     base 用错（10000）: 相对 %.3e  %s"
          % (rel3w, "（底数确实在起作用）" if rel3w > 1e-3
             else "**（!! base 被忽略）**"))
    ok_all = ok_all and rel3w > 1e-3

    # ⑥ 部分旋转（partial_rotary_factor=0.5）
    rope_part = {"full_attention": {"rope_type": "default",
                                "rope_theta": 500000.0,
                                "partial_rotary_factor": 0.5},
                 "sliding_attention": {"rope_type": "default",
                                  "rope_theta": 10000.0,
                                  "partial_rotary_factor": 1.0}}
    cfg4 = _cfg("per-head", rope=rope_part)
    ref4 = LagunaAttention(cfg4, 0, NH).eval()
    rrope4 = LagunaRotaryEmbedding(cfg4).eval()
    with torch.no_grad():
        cos4, sin4 = rrope4(x, pos, "full_attention")
        r4 = ref4(x, (cos4, sin4), mask)[0]
    mine4 = _by1_attn(ns, rope=True, rope_base=500000, rope_partial=0.5)
    _load(mine4, ref4)
    with torch.no_grad():
        m4 = mine4(x)
    dd4, rel4 = _rel(m4, r4)
    print("\n  ⑥ 部分旋转（partial=0.5）: 绝对 %.3e  相对 %.3e  %s"
          % (dd4, rel4, "逐位一致 ok" if rel4 < 1e-6 else "**对不上**"))
    ok_all = ok_all and rel4 < 1e-6
    mine4_full = _by1_attn(ns, rope=True, rope_base=500000, rope_partial=1.0)
    _load(mine4_full, ref4)
    with torch.no_grad():
        m4f = mine4_full(x)
    _, rel4f = _rel(m4f, m4)
    print("     partial 当成 1.0: 相对 %.3e  %s"
          % (rel4f, "（部分旋转确实在起作用）" if rel4f > 1e-3
             else "**（!! partial 被忽略）**"))
    ok_all = ok_all and rel4f > 1e-3
    print("\n  [%s] head gate %s"
          % ("PASS" if ok_all else "FAIL",
             "门控钉死 + rope 按层参数/部分旋转逐位一致，四条可证伪全红"
             if ok_all else "有问题"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())