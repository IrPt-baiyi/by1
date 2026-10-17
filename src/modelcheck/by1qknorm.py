#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1qknorm -- **qk_norm 的三个取值是不是三件事**：per_head / full / off。
参照物：transformers 的 Qwen3Attention（per-head RMSNorm）与 Olmo2Attention（整宽 RMSNorm）

## 为什么值得单独一个判卷人

`qk_norm` 在闭集里有 7 个写法（`off` / `per_head` / `full` / 旧写法 `true` `on` `yes` `1`）。
但**没有一条判据**证明：

  · `per_head` 与 `full` 真的是两件事（一个按 head_dim 归一、一个在拆头之前按整宽归一）；
  · `off` 与 `per_head` 真的不同；
  · 旧写法 `true` 与 `per_head` 真的是**同一个**（by1langspec 记档说等价，得验）。

如果实现里两条分支写成一样，配置写错**不会有任何症状** —— 前向照样跑完，只是算的是
另一个模型。那正是 ir.md 130 记下的那类错（契约全对、config 逐字段全对、前向算另一个）。

## 验什么

  ① `per_head` vs 官方 `Qwen3Attention`（rope 恒等、因果掩码）-> 逐值一致；
  ② 旧写法 `true` 与 `per_head` -> **必须逐位相同**（别名，不是两件事）；
  ③ `off` 与 `per_head` -> 必须变；
  ④ `full` 与 `per_head`（把 qn/kn 的权重按头数重复，两边的**唯一差别**就是归一轴）-> 必须变。

④ 的构造是关键：传同一组 qn 权重（重复到整宽）之后，若两种实现其实是一回事，输出会一模一样。
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import torch  # noqa: E402
import by1codegen as cg  # noqa: E402
import by1skip  # noqa: E402

try:
    from transformers import Qwen3Config, Olmo2Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention
except ImportError:
    Qwen3Config = Qwen3Attention = None
    Olmo2Config = Olmo2Attention = None

D, NH, KV, HD, SEQ = 64, 4, 2, 16, 8


def _mask(seq):
    idx = torch.arange(seq)
    keep = idx[None, :] <= idx[:, None]
    m = torch.zeros(1, 1, seq, seq)
    return m.masked_fill(~keep[None, None], float("-inf"))


def _by1(ns, qk="per_head", qn=None, kn=None):
    a = {"q": NH, "kv": KV, "head_dim": HD, "v": HD, "bias": False,
         "out_dim": NH * HD, "window": None, "rope_pairing": "half",
         "qk_norm": qk, "norm_eps": 1e-6, "rope": False, "rope_base": 10000,
         "rope_partial": 1.0, "sink": False, "head_gate": "off"}
    m = ns["Attention"](a, D).eval()
    if qn is not None:
        m.load_state_dict({"qn.w": qn, "kn.w": kn}, strict=False)
    return m


def _official():
    cfg = Qwen3Config(hidden_size=D, num_attention_heads=NH,
                      num_key_value_heads=KV, head_dim=HD,
                      num_hidden_layers=1, intermediate_size=64,
                      max_position_embeddings=64, rms_norm_eps=1e-6,
                      attention_bias=False, attention_dropout=0.0,
                      rope_theta=10000.0, layer_types=["full_attention"])
    cfg._attn_implementation = "eager"
    return Qwen3Attention(cfg, layer_idx=0).eval()


def _official_olmo2():
    """官方 `Olmo2Attention`：`q_norm`/`k_norm` 是**整宽** RMSNorm

    （`num_attention_heads * head_dim`），作用在**拆头之前**的扁平 q/k 上 ——
    正是 by1 的 `qk_norm = full`。"""
    cfg = Olmo2Config(hidden_size=D, num_attention_heads=NH,
                        num_key_value_heads=KV, head_dim=HD,
                        num_hidden_layers=1, intermediate_size=64,
                        max_position_embeddings=64, rms_norm_eps=1e-6,
                        attention_bias=False, attention_dropout=0.0,
                        rope_theta=10000.0)
    cfg._attn_implementation = "eager"
    return Olmo2Attention(cfg, layer_idx=0).eval()

def _load(mine, ref):
    sd = {
        "wq.weight": ref.q_proj.weight.detach().clone(),
        "wk.weight": ref.k_proj.weight.detach().clone(),
        "wv.weight": ref.v_proj.weight.detach().clone(),
        "wo.weight": ref.o_proj.weight.detach().clone(),
        "qn.w": ref.q_norm.weight.detach().clone(),
        "kn.w": ref.k_norm.weight.detach().clone(),
    }
    return mine.load_state_dict(sd, strict=False)


def _rel(a, b):
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-12)


def main():
    if Qwen3Attention is None:
        return by1skip.skip("找不到 transformers 的 qwen3 —— 这一项在这台机器上验不了")
    torch.manual_seed(0)
    ns = {}
    exec(cg.RUNTIME, ns)
    x = torch.randn(1, SEQ, D)
    mask = _mask(SEQ)
    cos = torch.ones(1, SEQ, HD)
    sin = torch.zeros(1, SEQ, HD)
    bad = []

    ref = _official()
    m_per = _by1(ns, "per_head")
    _load(m_per, ref)
    y_by1 = m_per(x).detach()
    y_ref = ref(x, position_embeddings=(cos, sin), attention_mask=mask)[0].detach()
    d1, r1 = _rel(y_by1, y_ref)
    print("  ① per_head   by1 vs 官方 Qwen3Attention   绝对 %.3e  相对 %.3e  [%s]"
          % (d1, r1, "PASS" if r1 < 1e-5 else "FAIL"))
    if r1 >= 1e-5:
        bad.append("per_head 与官方 Qwen3Attention 不一致（相对 %.3e）" % r1)

    m_true = _by1(ns, "true")
    _load(m_true, ref)
    d2, _ = _rel(m_true(x).detach(), y_by1)
    print("  ② 别名 true vs per_head                绝对 %.3e  [%s]"
          % (d2, "PASS" if d2 == 0.0 else "FAIL"))
    if d2 != 0.0:
        bad.append("旧写法 true 与 per_head 不等价（绝对 %.3e）—— by1langspec 的记档是错的" % d2)

    m_off = _by1(ns, "off")
    _load(m_off, ref)
    d3, _ = _rel(m_off(x).detach(), y_by1)
    print("  ③ 证伪 off  vs per_head               绝对 %.3e  [%s]"
          % (d3, "PASS" if d3 > 1e-3 else "FAIL"))
    if d3 <= 1e-3:
        bad.append("off 与 per_head 算出来一样（%.3e）—— 归一没生效" % d3)

    qn_rep = ref.q_norm.weight.detach().repeat(NH)
    kn_rep = ref.k_norm.weight.detach().repeat(KV)
    m_full = _by1(ns, "full", qn_rep, kn_rep)
    d4, _ = _rel(m_full(x).detach(), y_by1)
    print("  ④ 证伪 full vs per_head               绝对 %.3e  [%s]"
          % (d4, "PASS" if d4 > 1e-3 else "FAIL"))
    if d4 <= 1e-3:
        bad.append("full 与 per_head 算出来一样（%.3e）—— 两个取值被实现成一件事" % d4)

    # ---- ⑤ 正例：full（整宽）对官方 Olmo2Attention ----
    if Olmo2Attention is not None:
        ref_o = _official_olmo2()
        m_full_o = _by1(ns, "full")
        with torch.no_grad():
            m_full_o.wq.weight.copy_(ref_o.q_proj.weight)
            m_full_o.wk.weight.copy_(ref_o.k_proj.weight)
            m_full_o.wv.weight.copy_(ref_o.v_proj.weight)
            m_full_o.wo.weight.copy_(ref_o.o_proj.weight)
            m_full_o.qn.w.copy_(ref_o.q_norm.weight)
            m_full_o.kn.w.copy_(ref_o.k_norm.weight)
        yf = m_full_o(x).detach()
        yfo = ref_o(x, position_embeddings=(cos, sin),
                    attention_mask=mask)[0].detach()
        d5, r5 = _rel(yf, yfo)
        print("  ⑤ full       by1 vs 官方 Olmo2Attention  绝对 %.3e  相对 %.3e  [%s]"
              % (d5, r5, "PASS" if r5 < 1e-5 else "FAIL"))
        if r5 >= 1e-5:
            bad.append("full 与官方 Olmo2Attention 不一致（相对 %.3e）" % r5)

        # ---- ⑥ 可证伪：同一批权重，只把归一轴换成 per_head ----
        m_ph = _by1(ns, "per_head")
        with torch.no_grad():
            m_ph.wq.weight.copy_(ref_o.q_proj.weight)
            m_ph.wk.weight.copy_(ref_o.k_proj.weight)
            m_ph.wv.weight.copy_(ref_o.v_proj.weight)
            m_ph.wo.weight.copy_(ref_o.o_proj.weight)
            m_ph.qn.w.copy_(ref_o.q_norm.weight[:HD])
            m_ph.kn.w.copy_(ref_o.k_norm.weight[:HD])
        d6, _r6 = _rel(m_ph(x).detach(), yfo)
        print("  ⑥ 可证伪 per_head vs 官方 full         绝对 %.3e  [%s]"
              % (d6, "PASS" if d6 > 1e-3 else "FAIL"))
        if d6 <= 1e-3:
            bad.append("per_head 与 full 在同一批权重下算出一模一样（%.3e）" % d6)

    for b in bad:
        print("     ! " + b)
    print("  [%s] qk_norm：per_head 对 Qwen3、full 对 OLMo2（均逐位）；"
          "true 等价 per_head；off / 换轴后的 full 都差得出来"
          % ("PASS" if not bad else "FAIL"))
    return 0 if not bad else 1


if __name__ == "__main__":
    _sys.exit(main())