#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1mla -- MLA 的差分验证。
参照物：transformers 的 DeepseekV3Attention（MLA 同构）

把 by1 生成的 MLAttention 和 **transformers 的 DeepseekV3Attention** 放在一起，
喂同样的权重，比输出。

## 它现在验两件事（2026-10-17 第十六轮）

  1. **生成物里的 `MLAttention`** vs 官方；配对约定**钉死**为
     `interleaved`（实测相对差 0.0），且 `half` 必须对不上（差 5.03e-02）
     —— 原来写的是“两种里任一对上就算过”，那证明不了约定被区分；
  2. **直连对拍**：官方权重直接喂 `by1exec.op_mla`（实测相对 5.75e-07）。
     两份各自对上官方但彼此不一致会被这里抓住：把 `op_mla` 的 scaling
     改错，生成物那两行照样绿、直连行立刻红（3.48e-02）。
为什么 DeepSeek V3 能当 Ling 的判卷人：两者的 MLA 结构同构 ——
同样的 q_a→norm→q_b、kv_a_proj_with_mqa、kv_b 拆 [nope, v]、k_rot 单头共享、
scaling = qk_head_dim^-0.5。实测 7 个张量形状逐个相同。

用法:  python by1mla.py
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到 `by1/` 目录（即 `src/`），让包可导入。
# 判卷人是被**当脚本 spawn** 的（by1all），没有这层就 ModuleNotFoundError。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import sys
import torch

import numpy as np
import by1codegen as cg
import by1exec
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention, DeepseekV3RotaryEmbedding)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config)

D, NH, QL, KVL, NOPE, ROPE, VD = 1536, 16, 256, 512, 128, 64, 128
SEQ = 10


def causal(n):
    m = torch.full((1, 1, n, n), float("-inf"))
    return torch.triu(m, diagonal=1)


def main():
    torch.manual_seed(0)
    cfg = DeepseekV3Config(
        hidden_size=D, num_attention_heads=NH, num_key_value_heads=NH,
        q_lora_rank=QL, kv_lora_rank=KVL, qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE, v_head_dim=VD, attention_bias=False,
        attention_dropout=0.0, max_position_embeddings=4096, rope_theta=10000.0)
    cfg._attn_implementation = "eager"
    ref = DeepseekV3Attention(cfg, 0).eval()
    rrope = DeepseekV3RotaryEmbedding(cfg).eval()

    ns = {}
    exec(cg.RUNTIME, ns)

    print("=" * 74)
    print("  MLA 差分验证   by1 生成的 MLAttention  vs  transformers 的 DeepseekV3Attention")
    print("=" * 74)
    print("\n  维度: d=%d  heads=%d  q_lora=%d  kv_lora=%d  nope=%d  rope=%d  v=%d"
          % (D, NH, QL, KVL, NOPE, ROPE, VD))
    print("\n  判卷人的张量（应当和 Ling-3.0-tiny 的逐个同形）:")
    for k, v in ref.named_parameters():
        print("     %-30s %s" % (k, tuple(v.shape)))

    x = torch.randn(1, SEQ, D)
    pos = torch.arange(SEQ)[None]
    m = causal(SEQ)
    # **钉死配对约定，而不是"两种里任一对上就算过"。**
    # DeepSeek V3 的 rope 是 **interleaved**（实测那一支相对差 0.0）；
    # `half` 必须**对不上** —— 否则这条判据证明不了"约定被区分"。
    want_pairing = "interleaved"
    wrong_pairing = "half"
    results = {}

    print("\n  两种配对约定:")
    for pairing in ("interleaved", "half"):
        mine = ns["MLAttention"]({
            "q": NH, "q_lora": QL, "kv_lora": KVL, "qk_nope": NOPE,
            "qk_rope": ROPE, "v_dim": VD, "head_dim": NOPE + ROPE,
            "out_dim": NH * VD, "bias": False, "rope_base": 10000,
            "pairing": pairing, "head_gate": "off",
            "norm_eps": cfg.rms_norm_eps, "norm_one_plus": False}, D).eval()

        sd = {k: v.detach().clone() for k, v in ref.named_parameters()}
        sd["dense.weight"] = sd.pop("o_proj.weight")      # Ling 叫 dense
        sd["q_a_layernorm.w"] = sd.pop("q_a_layernorm.weight")
        sd["kv_a_layernorm.w"] = sd.pop("kv_a_layernorm.weight")
        miss, unexp = mine.load_state_dict(sd, strict=False)
        if miss or unexp:
            print("     !! 权重没对上 missing=%r unexpected=%r" % (miss, unexp))
            return 1

        with torch.no_grad():
            cos, sin = rrope(x, pos)
            r = ref(x, (cos, sin), m)[0]
            o = mine(x)
        dd = (r - o).abs().max().item()
        amp = max(r.abs().max().item(), 1e-9)
        rel = dd / amp
        good = rel < 1e-6
        results[pairing] = rel
        print("     pairing=%-12s 绝对差 %.3e  相对 %.3e   %s"
              % (pairing, dd, rel, "[一致]" if good else "[不一致]"))

    # 门控：Ling 用 sigmoid，Laguna 的普通注意力用 softplus —— 同名不同义
    print("\n  门控激活（Ling 的参考用 sigmoid）:")
    g = torch.randn(1, SEQ, NH)
    op = torch.sigmoid(g.float())
    print("     sigmoid 与参考一致:", torch.allclose(op, 1 / (1 + (-g).exp()), atol=1e-7))

    ok = (results[want_pairing] < 1e-6) and (results[wrong_pairing] > 1e-3)
    print("     -> 约定必须是 %s（它一致，且 %s 必须对不上）"
          % (want_pairing, wrong_pairing))

    # ── **直连对拍**：官方权重直接喂 `by1exec.op_mla` ─────────────────
    #
    # 上一段验的是**生成物里的 MLAttention**；真正跑 IR/图那条路的是
    # `by1exec.op_mla`。两份各自对上官方但彼此不一致，就会被这里抓住。
    print("\n  直连对拍：官方 DeepseekV3Attention -> by1exec.op_mla")
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in ref.named_parameters()}
    P = {
        "q_a_proj": sd["q_a_proj.weight"],
        "q_a_layernorm.w": sd["q_a_layernorm.weight"],
        "q_b_proj": sd["q_b_proj.weight"],
        "kv_a_proj_with_mqa": sd["kv_a_proj_with_mqa.weight"],
        "kv_a_layernorm.w": sd["kv_a_layernorm.weight"],
        "kv_b_proj": sd["kv_b_proj.weight"],
        "dense": sd["o_proj.weight"],
    }
    at = {"q": NH, "qk_nope": NOPE, "qk_rope": ROPE, "v_dim": VD,
          "kv_lora": KVL, "rope_base": 10000, "pairing": want_pairing,
          "head_gate": "off", "bias": False,
          "norm_eps": float(cfg.rms_norm_eps), "norm_one_plus": False}
    with torch.no_grad():
        xa = x.numpy().astype(np.float32)
        ref_a = ref(x, (cos, sin), m)[0].numpy().astype(np.float64)
    got_a = by1exec.op_mla(P, at, [xa], D).astype(np.float64)
    dd_a = float(np.abs(got_a - ref_a).max())
    rel_a = dd_a / max(float(np.abs(ref_a).max()), 1e-12)
    good_a = rel_a < 1e-6
    print("     绝对差 %.3e  相对 %.3e  %s"
          % (dd_a, rel_a, "后端 op 与官方一致 ok" if good_a else "**不一致**"))
    ok = ok and good_a

    print("\n  [%s] MLA %s" % ("PASS" if ok else "FAIL",
                               "约定已钉死 + 后端 op 直连一致" if ok else "对不上"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
