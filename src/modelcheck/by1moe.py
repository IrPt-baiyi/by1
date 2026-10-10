#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1moe -- noaux_tc 分组路由的差分验证。

把 by1 的 MoE 路由和 **transformers 的 DeepseekV3TopkRouter** 放在一起，
喂同样的权重，比选中的专家下标和权重。

判卷人成立的理由：DeepseekV3TopkRouter 和 Ling 的 BailingMoeV3Gate
语义逐行相同（sigmoid 打分、偏置只影响选择、每组前 2 之和选组、
组内 top-k、权重归一化 × routed_scaling_factor），但是**两个团队写的**。

用法:  python by1moe.py
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
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3TopkRouter)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config)

D, E, K, NG, TG = 256, 32, 4, 8, 4
NT = 12


def main():
    torch.manual_seed(0)
    cfg = DeepseekV3Config(
        hidden_size=D, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=128, intermediate_size=64,
        n_routed_experts=E, n_shared_experts=1, num_experts_per_tok=K,
        n_group=NG, topk_group=TG, routed_scaling_factor=2.5,
        norm_topk_prob=True, scoring_func="sigmoid", topk_method="noaux_tc",
        moe_intermediate_size=32, first_k_dense_replace=0,
        num_nextn_predict_layers=0)
    ref = DeepseekV3TopkRouter(cfg).eval()

    ns = {}
    exec(cg.RUNTIME, ns)
    MoE = ns["MoE"]
    mine = MoE({"experts": E, "top_k": K, "hidden": 32, "shared": 0,
                "shared_hidden": 0, "score_bias": True, "routed_scale": 2.5,
                "router_bias": False, "routing": "sigmoid_group_topk",
                "act": "silu", "limit": None, "alpha": 1.702,
                "expert_bias": False, "shared_gate": False,
                "n_group": NG, "topk_group": TG}, D).eval()

    print("=" * 74)
    print("  noaux_tc 分组路由   by1 的 MoE  vs  transformers 的 DeepseekV3TopkRouter")
    print("=" * 74)
    print("\n  维度: d=%d  专家=%d  top_k=%d  组=%d  选组=%d  缩放=2.5"
          % (D, E, K, NG, TG))

    with torch.no_grad():
        mine.router.weight.copy_(ref.weight)
        mine.score_bias.copy_(ref.e_score_correction_bias)

    # 给偏置一些非零值 —— 否则"偏置只影响选择"这条根本测不出来
    with torch.no_grad():
        mine.score_bias.copy_(torch.randn(E) * 0.5)
        ref.e_score_correction_bias.copy_(mine.score_bias)

    x = torch.randn(1, NT, D)
    with torch.no_grad():
        _, rw, ri = ref(x.view(-1, D))
        logits = mine.router(x.view(-1, D))
        # 复用 MoE 的路由部分（把专家算的部分跳过）
        sc = logits.sigmoid()
        choice = sc + mine.score_bias
        ng, tg = NG, TG
        gs = choice.view(-1, ng, E // ng).topk(2, dim=-1)[0].sum(-1)
        gi = gs.topk(tg, dim=-1, sorted=False)[1]
        gm = torch.zeros_like(gs).scatter_(1, gi, 1)
        mask = gm.unsqueeze(-1).expand(-1, ng, E // ng).reshape(-1, E)
        choice = choice.masked_fill(~mask.bool(), float("-inf"))
        mi = choice.topk(K, dim=-1, sorted=False)[1]
        mv = sc.gather(1, mi)
        mw = mv / (mv.sum(-1, keepdim=True) + 1e-20) * 2.5

    # 顺序可能不同（sorted=False），按行比集合
    same = True
    for r in range(NT):
        if set(ri[r].tolist()) != set(mi[r].tolist()):
            same = False
            print("     第 %d 行选中的专家不同: %r vs %r"
                  % (r, sorted(ri[r].tolist()), sorted(mi[r].tolist())))
    print("\n  选中的专家集合一致: %s" % ("是" if same else "否"))

    # 权重按专家对齐后比
    rmap = torch.zeros(NT, E)
    rmap.scatter_(1, ri, rw)
    mmap = torch.zeros(NT, E)
    mmap.scatter_(1, mi, mw)
    dd = (rmap - mmap).abs().max().item()
    print("  权重（按专家对齐后）最大绝对差: %.3e" % dd)

    # 可证伪：把偏置关掉，选择**必须**变（否则这个偏置根本没起作用）
    with torch.no_grad():
        c2 = sc.clone()
        gs2 = c2.view(-1, ng, E // ng).topk(2, dim=-1)[0].sum(-1)
        gi2 = gs2.topk(tg, dim=-1, sorted=False)[1]
        gm2 = torch.zeros_like(gs2).scatter_(1, gi2, 1)
        m2 = gm2.unsqueeze(-1).expand(-1, ng, E // ng).reshape(-1, E)
        c2 = c2.masked_fill(~m2.bool(), float("-inf"))
        i2 = c2.topk(K, dim=-1, sorted=False)[1]
    changed = sum(1 for r in range(NT)
                  if set(i2[r].tolist()) != set(mi[r].tolist()))
    print("  去掉偏置后有 %d/%d 行的选择变了  %s"
          % (changed, NT, "（偏置确实在起作用）" if changed else "（!! 偏置没起作用）"))

    ok = same and dd < 1e-6
    print("\n  [%s] noaux_tc %s" % ("PASS" if ok else "FAIL",
                                    "与参考一致" if ok else "对不上"))
    ok2 = routing_vs_config()
    return 0 if (ok and ok2) else 1


def routing_vs_config():
    """第二把尺子：**`.by1` 的路由编码 vs 官方 config 的 `scoring_func`**。

    上一把尺子只证明"`sigmoid_group_topk` 这一支的数学对"。它证明不了
    **模型写的是不是这一支** —— 而 GLM-5.3-Flash 就是写错的那个：
    `.by1` 里 `routing = softmax_topk` 加一个 `scoring = sigmoid`，
    而 `scoring` **没有任何人读它**（全库只有注释提到），于是前向走的是
    softmax 打分，官方却是 sigmoid（config `scoring_func = "sigmoid"`，
    实现 `GlmMoeDsaTopkRouter`：`scores = router_logits.sigmoid()`）。

    这一类错**契约看不出来、config 逐字段也看不出来** —— 那两处都是对的。
    只有"编码 ↔ 官方字段"这条对拍看得见。所以它进判卷人，不进注释。
    """
    print("\n  路由编码 vs 官方 config 的 scoring_func:")
    import importlib.util
    import by1io
    import by1refs
    sp = importlib.util.spec_from_file_location('bcm', by1paths.tool('by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    bad, seen = [], 0
    for f in by1paths.names():
        try:
            p = by1refs.paths(f, 'config')   # 返回字符串或 None，不是列表
        except Exception:
            continue
        if not p:
            continue
        raw = by1io.read_json(p, encoding='utf-8')
        tc = raw.get('text_config', raw)
        sf = str(tc.get('scoring_func', '')).strip().lower()
        if not sf:
            continue
        seen += 1
        _r, info = bc.check(f)
        for mech, a in (info.get('mech_attrs') or {}).items():
            rt = str(a.get('routing', '')).strip().lower()
            if not rt:
                continue
            # `routing` 的前缀就是打分函数：sigmoid_* -> sigmoid，
            # softmax_topk / topk_softmax -> softmax
            got = 'sigmoid' if rt.startswith('sigmoid') else 'softmax'
            mark = 'ok' if got == sf else '!!'
            if got != sf:
                bad.append(f)
            print("     %s %-26s %-22s routing=%-20s -> %s / config %s"
                  % (mark, f, mech, rt, got, sf))
    if not seen:
        print("     （没有带 scoring_func 的官方 config —— 这一项没验）")
        return True
    print("     -> %s" % ("逐个一致" if not bad else
                          "**不一致：%s**" % ', '.join(sorted(set(bad)))))
    return not bad


if __name__ == "__main__":
    sys.exit(main())
