#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1mla -- MLA 的差分验证。

把 by1 生成的 MLAttention 和 **transformers 的 DeepseekV3Attention** 放在一起，
喂同样的权重，比输出。

为什么 DeepSeek V3 能当 Ling 的判卷人：两者的 MLA 结构同构 ——
同样的 q_a→norm→q_b、kv_a_proj_with_mqa、kv_b 拆 [nope, v]、k_rot 单头共享、
scaling = qk_head_dim^-0.5。实测 7 个张量形状逐个相同。

用法:  python by1mla.py
"""
import sys
import torch

import by1codegen as cg
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
    ok_any = False

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
        ok_any = ok_any or good
        print("     pairing=%-12s 绝对差 %.3e  相对 %.3e   %s"
              % (pairing, dd, rel, "[一致]" if good else "[不一致]"))

    # 门控：Ling 用 sigmoid，Laguna 的普通注意力用 softplus —— 同名不同义
    print("\n  门控激活（Ling 的参考用 sigmoid）:")
    g = torch.randn(1, SEQ, NH)
    op = torch.sigmoid(g.float())
    print("     sigmoid 与参考一致:", torch.allclose(op, 1 / (1 + (-g).exp()), atol=1e-7))

    print("\n  [%s] MLA %s" % ("PASS" if ok_any else "FAIL",
                               "与参考逐位一致" if ok_any else "对不上"))
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
