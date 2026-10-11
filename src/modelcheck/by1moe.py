#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1moe -- MoE 路由的差分验证：**分组选法** + **打分函数**。

两把尺子，各自对一层的不同东西：

  ① **选法的数学**：`noaux_tc` 分组路由 vs transformers 的
     `DeepseekV3TopkRouter`（sigmoid 打分）。
     判卷人成立的理由：它和 Ling 的 `BailingMoeV3Gate` 语义逐行相同
     （sigmoid 打分、偏置只影响选择、每组前 2 之和选组、组内 top-k、
     权重归一化 × routed_scaling_factor），但是**两个团队写的**。

  ② **打分函数**（2026-10-11 新增原语）：`sqrtsoftplus` vs transformers 的
     `DeepseekV4TopKRouter`。V4 的 `scoring_func = "sqrtsoftplus"`
     （= `sqrt(softplus(x))`），而 by1 原来**没有位置放它** ——
     打分函数是从 `routing` 的名字前缀推的（`sigmoid_*` -> sigmoid），
     推不出来的就静默走 softmax。

  ③ **编码 vs 官方 config**（`routing_vs_config`）：`.by1` 写的打分函数
     和官方 `scoring_func` 对不对得上。**这一项原来靠名字前缀猜**
     （`rt.startswith('sigmoid')`），是错的来源之一 —— 现在读 `score`。

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
import by1ir
import by1skip
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3TopkRouter)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config)
# **判卷人可选，不在就跳过 —— 但退出是调用方的事，库代码不 sys.exit**
# （by1lint 第⑩类）。所以这里只把名字置 None，跳过放到 main() 里
# `return by1skip.skip(...)`：`import by1moe` 照旧成功（by1fast 的
# import 冒烟测试要的就是这个），跑起来才退出码 30。
try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4TopKRouter)
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config)
except ImportError:
    DeepseekV4TopKRouter = DeepseekV4Config = None

D, E, K, NG, TG = 256, 32, 4, 8, 4
NT = 12


def main():
    if DeepseekV4TopKRouter is None:
        # 判卷人不在就干净跳过（退出码 30），不是崩、更不是"检查器坏了"
        return by1skip.skip(
            '找不到 transformers 的 deepseek_v4 —— 这一项在这台机器上验不了')
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
    ok2 = sqrtsoftplus_vs_v4()
    ok3 = routing_vs_config()
    return 0 if (ok and ok2 and ok3) else 1


def sqrtsoftplus_vs_v4():
    """**打分函数独立于选法** —— 拿 DeepSeek V4 的 `sqrtsoftplus` 验一遍。

    V4 的 config `scoring_func = "sqrtsoftplus"`（`SqrtSoftplusActivation`：
    `nn.functional.softplus(x).sqrt()`），而 by1 原来把这个信息**塞在
    `routing` 的名字里**（`sigmoid_topk` / `softmax_topk` …）——
    于是 `sqrtsoftplus` 根本没有位置放。

    这一节验两件事：
      ① **打分函数的数学**：by1 的 `_score(logits, "sqrtsoftplus")`
         和 `DeepseekV4TopKRouter.score_fn` 逐点一致
      ② **选法没被动过**：同一组 logits 换成 sqrtsoftplus 之后，
         选出来的专家**和 sigmoid 不同** —— 否则这个属性白加了
         （那正是这个仓库最忌讳的一类："声明了、不报错、什么也不做"）
      ③ **解析只有一份**：`by1ir.moe_score`（三个后端都从它取）
         对同一组 attrs 给出的答案，必须和编译器填进 `_score` 的**一致** ——
         这一条单独列出来，是因为"一条规则三处实现"正是上一次静默错的根因。
    """
    print("\n" + "=" * 74)
    print("  sqrtsoftplus 打分   by1  vs  transformers 的 DeepseekV4TopKRouter")
    print("=" * 74)

    # ③ 先验"解析只有一份"—— 它不依赖 transformers，先跑
    _cases = [
        ({"routing": "sigmoid_topk"}, "sigmoid"),
        ({"routing": "sigmoid_group_topk"}, "sigmoid"),
        ({"routing": "softmax_topk"}, "softmax"),
        ({"routing": "topk_softmax"}, "softmax"),
        ({"routing": "sigmoid_topk", "score": "sqrtsoftplus"}, "sqrtsoftplus"),
        ({"routing": "sigmoid_group_topk", "score": "softplus"}, "softplus"),
        # `_score` 是编译期填的 —— 必须优先于"按前缀推"
        ({"routing": "sigmoid_topk", "score": None,
          "_score": "sqrtsoftplus"}, "sqrtsoftplus"),
    ]
    bad_resolve = []
    for a, want in _cases:
        got = by1ir.moe_score(a)
        if got != want:
            bad_resolve.append("%r -> %r（期望 %r）" % (a, got, want))
    # 闭集外的 routing 必须返回 None（而不是退回 softmax）
    if by1ir.moe_score({"routing": "nonsense"}) is not None:
        bad_resolve.append("闭集外的 routing 竟然推出了一个值")
    print("\n  ③ 解析一致性（by1ir.moe_score，%d 组）: %s"
          % (len(_cases) + 1,
             "全部一致" if not bad_resolve else "**不一致**"))
    for b in bad_resolve:
        print("       %s" % b)
    if bad_resolve:
        print("\n  [FAIL] sqrtsoftplus 的解析就不对，后面不用比了")
        return False
    torch.manual_seed(0)
    cfg = DeepseekV4Config(
        hidden_size=D, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=128, intermediate_size=64,
        num_local_experts=E, num_experts_per_tok=K, routed_scaling_factor=1.5,
        norm_topk_prob=True, scoring_func="sqrtsoftplus")
    ref = DeepseekV4TopKRouter(cfg).eval()

    ns = {}
    exec(cg.RUNTIME, ns)
    _score = ns["_score"]

    # ① 打分函数的数学：逐点比
    logits = torch.randn(NT, E) * 3.0
    with torch.no_grad():
        a = _score(logits, "sqrtsoftplus")
        b = ref.score_fn(logits)
    dscore = (a - b).abs().max().item()
    print("\n  ① 打分函数逐点最大差: %.3e   %s"
          % (dscore, "一致" if dscore < 1e-6 else "**对不上**"))

    # ② 选法没被换掉：sqrtsoftplus 与 sigmoid 选出来的专家**必须不同**
    with torch.no_grad():
        sb = torch.randn(E) * 0.5
        i_sig = (_score(logits, "sigmoid") + sb).topk(K, dim=-1,
                                                      sorted=False)[1]
        i_sqr = (_score(logits, "sqrtsoftplus") + sb).topk(K, dim=-1,
                                                           sorted=False)[1]
    n_diff = sum(1 for r in range(NT)
                 if set(i_sig[r].tolist()) != set(i_sqr[r].tolist()))
    print("  ② sqrtsoftplus 与 sigmoid 的选择有 %d/%d 行不同   %s"
          % (n_diff, NT,
             "（这个属性确实在改前向）" if n_diff else
             "（!! 选了等价的专家 —— 这个属性没起作用）"))

    with torch.no_grad():
        xh = torch.randn(NT, D)
        # V4 的偏置是 buffer（初始为 0），喂一点非零值才测得出"只影响选择"
        ref.e_score_correction_bias.copy_(torch.randn(E) * 0.5)
        _, rw, ri = ref(xh)
        lg = ref.weight @ xh.T                       # [E, NT]
        sc = _score(lg.T, "sqrtsoftplus")
        mi = (sc + ref.e_score_correction_bias).topk(
            K, dim=-1, sorted=False)[1]
        mv = sc.gather(1, mi)
        mw = mv / (mv.sum(-1, keepdim=True) + 1e-20) * 1.5
    same = all(set(ri[r].tolist()) == set(mi[r].tolist()) for r in range(NT))
    rmap = torch.zeros(NT, E); rmap.scatter_(1, ri, rw)
    mmap = torch.zeros(NT, E); mmap.scatter_(1, mi, mw)
    dw = (rmap - mmap).abs().max().item()
    print("  ③ 选中专家集合一致: %s   权重最大绝对差: %.3e"
          % ("是" if same else "否", dw))

    ok = dscore < 1e-6 and same and dw < 1e-6 and n_diff > 0
    print("\n  [%s] sqrtsoftplus %s" % ("PASS" if ok else "FAIL",
                                        "与参考一致，且确实改了前向"
                                        if ok else "对不上"))
    return ok


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

    ── 2026-10-11：**这一节原来自己是错的来源** ──────────────────
    它靠 `routing` 的名字前缀猜打分函数：
        got = 'sigmoid' if rt.startswith('sigmoid') else 'softmax'
    这在两种情况下失明：
      · `sqrtsoftplus`（V4）—— 前缀里根本没有它，猜成 softmax
      · 选法叫 `softmax_topk` 而 `scoring_func = "sigmoid"`
        （GLM-5.3 的真错）—— 猜成 softmax，**和写错的那份对上了**，
        于是这一项当时是绿的，而模型是错的
    现在 `score` 是**显式属性**，这一节直接读它（含编译期填的 `_score`），
    **不再猜**。读不到（老模型只写了 `routing`）才退回前缀——并把
    那种情况标成 `~`（"推的，不是写的"），让它和"显式写对了"在输出上分开。
    """
    print("\n  路由编码 vs 官方 config 的 scoring_func:")
    import importlib.util
    import by1io
    import by1refs
    sp = importlib.util.spec_from_file_location('bcm', by1paths.tool('by1check.py'))
    bc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(bc)
    bad, guessed, seen = [], [], 0
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
            # **显式写的优先**；没有才退回按前缀推（并且标出来）
            explicit = str(a.get('score') or '').strip().lower()
            if explicit:
                got, how = explicit, '写'
            else:
                got = 'sigmoid' if rt.startswith('sigmoid') else 'softmax'
                how = '推'
                guessed.append(f)
            mark = 'ok' if got == sf else '!!'
            if got != sf:
                bad.append(f)
            print("     %s %-26s %-22s score=%-14s(%s) / config %s"
                  % (mark, f, mech, got, how, sf))
    if not seen:
        print("     （没有带 scoring_func 的官方 config —— 这一项没验）")
        return True
    if guessed:
        print("     ~  有 %d 处是**按前缀推的**（模型没写 score）：%s"
              % (len(guessed), ', '.join(sorted(set(guessed)))))
        print("        → 能用，但推不出 `sqrtsoftplus` 这类值；")
        print("          新模型请显式写 `score = ...`。")
    print("     -> %s" % ("逐个一致" if not bad else
                          "**不一致：%s**" % ', '.join(sorted(set(bad)))))
    return not bad


if __name__ == "__main__":
    sys.exit(main())
