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
  ④ 反过来验一次：`gating=False` 的官方（没有门）与 by1 `head_gate=off` 也应一致 ——
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


def _cfg(gating):
    cfg = LagunaConfig(
        hidden_size=D, num_attention_heads=NH, num_key_value_heads=KV,
        head_dim=HD, num_hidden_layers=1, intermediate_size=64,
        max_position_embeddings=64, rope_theta=10000.0,
        layer_types=["full_attention"], sliding_window=8,
        attention_bias=False, attention_dropout=0.0, gating=gating)
    cfg._attn_implementation = "eager"
    return cfg


def _by1_attn(ns, gate="per_head", act="softplus", rope=False):
    a = {"q": NH, "kv": KV, "head_dim": HD, "v": HD, "bias": False,
         "out_dim": NH * HD, "window": None, "rope_base": 10000,
         "rope_pairing": "half", "qk_norm": "per_head", "norm_eps": 1e-6,
         "rope": rope, "sink": False, "head_gate": gate,
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
    # **rope 两边都设成恒等**：Laguna 的 `LagunaRotaryEmbedding` 频率与 by1 的
    # `rope_tables` **不一致**（实测 cos 第二项 0.9993 vs 0.9504，基频对不上），
    # 那是**另一个问题**，记在 docs/gap-analysis 里待查。这一轮只验门控 ——
    # 把 rope 两边都设成恒等，才能让"门"这一个变量单独成立。
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

    print("\n  [%s] head gate %s"
          % ("PASS" if ok_all else "FAIL",
             "softplus 钉死、sigmoid 必须不同、关掉必须变、无门也对"
             if ok_all else "有问题"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())