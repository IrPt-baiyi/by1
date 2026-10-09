#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1rope -- llama3 式 RoPE 缩放的差分验证。

把 by1 算出来的逆频率和 **transformers 的 `_compute_llama3_parameters`** 逐项比。

为什么它值得单独一个判卷人：llama3 和 YaRN **都改频率、参数名还重叠**
（factor / original_max_position_embeddings），但改法完全不同 ——
YaRN 调的是 ramp 的起止维，llama3 调的是**波长阈值**。
混起来前向不会报错，只会默默地算错。

而且 llama3 的 attention_factor 恒为 1.0（参考实现的注释写着 "Unused in this
type of RoPE"），跟着 YaRN 的公式算会多乘一个 1.069。

用法:  python by1rope.py
"""

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

import by1codegen as cg
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig


def main():
    # Step-3.7-Flash 全量层的实际参数
    D, NH, HD = 4096, 64, 128
    BASE, FAC, ORIG, HF, LF = 5000000, 2.0, 131072, 32.0, 1.0

    cfg = LlamaConfig(
        hidden_size=D, num_attention_heads=NH, head_dim=HD,
        max_position_embeddings=ORIG * 2, rope_theta=BASE,
        rope_parameters={"rope_type": "llama3", "factor": FAC,
                         "original_max_position_embeddings": ORIG,
                         "high_freq_factor": HF, "low_freq_factor": LF})
    inv_ref, attn_factor = ROPE_INIT_FUNCTIONS["llama3"](cfg)

    ns = {}
    exec(cg.RUNTIME, ns)
    cos, sin = ns["rope_tables"](HD, 2, BASE, torch.device("cpu"),
                                 {"type": "llama3", "factor": FAC,
                                  "original": ORIG, "high_freq": HF,
                                  "low_freq": LF})
    # cos/sin 在位置 1 上就是 inv 的余弦/正弦 —— 反解出逆频率
    inv_mine = torch.atan2(sin[1], cos[1])

    print("=" * 74)
    print("  llama3 RoPE   by1 的 rope_tables  vs  transformers 的 _compute_llama3_parameters")
    print("=" * 74)
    print("\n  base=%d  factor=%g  original=%d  high_freq=%g  low_freq=%g  head_dim=%d"
          % (BASE, FAC, ORIG, HF, LF, HD))
    print("  频率个数: 参考 %d   by1 %d" % (len(inv_ref), len(inv_mine)))

    dd = (inv_ref.float() - inv_mine).abs().max().item()
    scale = inv_ref.abs().max().item()
    print("  逆频率最大绝对差: %.3e   相对 %.3e" % (dd, dd / scale))
    print("  attention_factor: 参考 %.6f   by1 %.6f" % (attn_factor, 1.0))

    # 可证伪：把类型当成 yarn 算，必须对不上
    try:
        c2, s2 = ns["rope_tables"](HD, 2, BASE, torch.device("cpu"),
                                   {"type": "yarn", "factor": FAC,
                                    "original": ORIG, "beta_fast": 32,
                                    "beta_slow": 1, "truncate": True})
        inv_yarn = torch.atan2(s2[1], c2[1])
        dy = (inv_ref.float() - inv_yarn).abs().max().item()
        print("  当成 YaRN 算的话最大绝对差: %.3e  %s"
              % (dy, "（确实不同，类型区分是必要的）" if dy > 1e-6
                 else "（!! 两种算法居然一样）"))
    except Exception as e:
        print("  当成 YaRN 算会报错:", type(e).__name__)

    ok = dd / scale < 1e-6 and abs(attn_factor - 1.0) < 1e-9
    print("\n  [%s] llama3 RoPE %s" % ("PASS" if ok else "FAIL",
                                       "与参考一致" if ok else "对不上"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
