#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1rope -- **RoPE 逆频率三支的网格判卷人**（default / yarn / llama3）。
参照物：transformers 的 ROPE_INIT_FUNCTIONS 与 LlamaRotaryEmbedding

## 为什么值得单独一个判卷人

三支都改频率，名字还重叠，而**参考实现在 transformers 里**：
`ROPE_INIT_FUNCTIONS[type](cfg)`（`_compute_*_parameters`）。
llama3 和 YaRN 不是一回事 —— YaRN 调的是 ramp 的起止维，llama3 调的是**波长阈值**；
写成一支就会「看起来对、算的是另一个模型」。

## 判据

每一格同时比**两份 by1 实现**：

  ① 生成物里的 `rope_tables`（codegen 的 RUNTIME，是**产物**）
  ② `by1graph.rope_inv`（IR/lowering 与 by1exec 共用的**唯一源头**）

对手是同一个官方 `ROPE_INIT_FUNCTIONS[type]`。两份都对上，才算这一格过；
任一份漂了都会红 —— 省得「一份错、另一份掩盖」。

外加三条**可证伪**：把 llama3 参数当 yarn 算、把 yarn 参数当 llama3 算、
把 factor≠1 的当 default 算 —— 三组都必须**对不上**。
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
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import sys  # noqa: E402
import torch  # noqa: E402
import by1codegen as cg  # noqa: E402
import by1graph  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # noqa: E402


#: 网格：(tag, 官方 rope_type, head_dim, base, 官方 rope_parameters, by1 的 yarn 字典)
#: by1 的字典键名不同（`type`/`original`/`high_freq`/`low_freq`）—— 两套命名是
#: 历史事实，判卷人负责**翻译**，不强行统一（统一要改 IR，那是另一件事）。
def _grid():
    ORIG, HF, LF = 131072, 32.0, 1.0
    cases = []
    for dim, base in ((64, 10000.0), (128, 5000000.0), (256, 10000.0)):
        cases.append(("default d%d b%g" % (dim, base), "default", dim, base,
                      {"rope_type": "default"}, None))
    for fac in (2.0, 4.0):
        for bf, bs in ((32.0, 1.0), (64.0, 1.0)):
            for tr in (True, False):
                cases.append((
                    "yarn f%g bf%g bs%g tr%d" % (fac, bf, bs, int(tr)),
                    "yarn", 128, 5000000.0,
                    {"rope_type": "yarn", "factor": fac,
                     "original_max_position_embeddings": ORIG,
                     "beta_fast": bf, "beta_slow": bs, "truncate": tr,
                     "attention_factor": None},
                    {"type": "yarn", "factor": fac, "original": ORIG,
                     "beta_fast": bf, "beta_slow": bs, "truncate": tr}))
    for fac, hf, lf2 in ((2.0, HF, LF), (8.0, 4.0, 1.0)):
        cases.append((
            "llama3 f%g hf%g lf%g" % (fac, hf, lf2),
            "llama3", 128, 1000000.0,
            {"rope_type": "llama3", "factor": fac,
             "original_max_position_embeddings": ORIG,
             "high_freq_factor": hf, "low_freq_factor": lf2},
            {"type": "llama3", "factor": fac, "original": ORIG,
             "high_freq": hf, "low_freq": lf2}))
    return cases


def _ref_inv(rtype, dim, base, params):
    """官方参考。**`default` 没有注册函数**（5.20 把它内联进各模型了）——
    那就用官方类本身：`LlamaRotaryEmbedding(cfg).inv_freq`。别退化成"自己再
    把公式写一遍"，那等于没有参考。"""
    # **`max_position_embeddings` 要和 factor 自洽**：transformers 会拿它跟
    # `original_max_position_embeddings` 算一个"隐含 factor"，跟显式 factor 不符就
    # 打一整屏警告（而警告会被人当成"判据有问题"）。这里按 factor 设。
    _fac = float(params.get("factor", 1.0) or 1.0)
    _orig = int(params.get("original_max_position_embeddings", 0) or 0)
    _maxpos = int(_orig * _fac) if _orig else 256
    cfg = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=dim,
                      max_position_embeddings=_maxpos, rope_theta=base,
                      rope_parameters=params)
    if rtype == "default":
        return LlamaRotaryEmbedding(cfg).inv_freq.float(), 1.0
    inv, attn = ROPE_INIT_FUNCTIONS[rtype](cfg)
    return inv.float(), attn


def main():
    ns = {}
    exec(cg.RUNTIME, ns)
    rope_tables = ns["rope_tables"]

    print("=" * 78)
    print("  RoPE 逆频率：by1 的两份实现 vs transformers 的 ROPE_INIT_FUNCTIONS")
    print("=" * 78)
    print()
    print("  %-26s %8s %8s %10s %10s" % ("case", "n", "abs", "rel", "margin"))
    print("  " + "-" * 74)

    bad = []
    worst = 0.0
    for tag, rtype, dim, base, params, yarn in _grid():
        inv_ref, _attn = _ref_inv(rtype, dim, base, params)
        # ① 生成物里的 rope_tables：pos=1 的 cos/sin 反解出逆频率
        c, s = rope_tables(dim, 2, base, torch.device("cpu"), yarn)
        inv_runtime = torch.atan2(s[1], c[1]).float()
        # ② 唯一源头
        inv_graph = torch.as_tensor(by1graph.rope_inv(dim, base, yarn)).float()
        if len(inv_runtime) != len(inv_ref) or len(inv_graph) != len(inv_ref):
            bad.append("%s：频率个数不一致 %d/%d/%d"
                       % (tag, len(inv_ref), len(inv_runtime), len(inv_graph)))
            continue
        scale = max(inv_ref.abs().max().item(), 1e-12)
        d1 = (inv_runtime - inv_ref).abs().max().item()
        d2 = (inv_graph - inv_ref).abs().max().item()
        d = max(d1, d2)
        rel = d / scale
        margin = (1e-6 / rel) if rel > 0 else float("inf")
        worst = max(worst, rel)
        ok = rel < 1e-6
        if not ok:
            bad.append("%s：相对 %.3e（runtime %.3e / graph %.3e）"
                       % (tag, rel, d1, d2))
        mark = "" if ok else "**红**"
        print("  %-26s %8d %8.1e %10.2e %10.1f %s"
              % (tag, len(inv_ref), d, rel, margin, mark))

    print()
    print("  ── 可证伪：三种「用错类型」都必须对不上")
    O = 131072
    f = 2.0
    pairs = (
        ("llama3 参数当 yarn 算",
         {"rope_type": "llama3", "factor": f,
          "original_max_position_embeddings": O,
          "high_freq_factor": 32.0, "low_freq_factor": 1.0},
         {"type": "yarn", "factor": f, "original": O,
          "beta_fast": 32.0, "beta_slow": 1.0, "truncate": True}),
        ("yarn 参数当 llama3 算",
         {"rope_type": "yarn", "factor": f,
          "original_max_position_embeddings": O,
          "beta_fast": 32.0, "beta_slow": 1.0, "truncate": True,
          "attention_factor": None},
         {"type": "llama3", "factor": f, "original": O,
          "high_freq": 32.0, "low_freq": 1.0}),
        ("factor≠1 的当 default 算",
         {"rope_type": "yarn", "factor": f,
          "original_max_position_embeddings": O,
          "beta_fast": 32.0, "beta_slow": 1.0, "truncate": True,
          "attention_factor": None},
         None),
    )
    for label, ref_params, yarn in pairs:
        rtype = ref_params["rope_type"]
        inv_ref, _a = _ref_inv(rtype, 128, 5000000.0, ref_params)
        c, s = rope_tables(128, 2, 5000000.0, torch.device("cpu"), yarn)
        inv_mine = torch.atan2(s[1], c[1]).float()
        # 只有**同类型**才逐个比；类型不同时长度/语义都可能不同
        if len(inv_ref) != len(inv_mine):
            print("  %-26s 长度不同（%d vs %d）—— 类型不通用"
                  % (label, len(inv_ref), len(inv_mine)))
            continue
        d = (inv_mine - inv_ref).abs().max().item()
        rel = d / max(inv_ref.abs().max().item(), 1e-12)
        red = rel > 1e-6
        if not red:
            bad.append("%s 居然对上了（可证伪那条失效）" % label)
        print("  %-26s 相对 %.3e  %s"
              % (label, rel, "对不上 ok" if red else "**对上了，可证伪失效**"))

    print()
    ok = not bad
    print("  [%s] RoPE 逆频率 %d 格 + 3 条可证伪：最差相对 %.2e"
          % ("PASS" if ok else "FAIL", len(_grid()), worst))
    for b in bad:
        print("     ! " + b)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())