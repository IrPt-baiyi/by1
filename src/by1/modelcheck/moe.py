#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1moe -- MoE 路由的差分验证：**分组选法** + **打分函数**。
参照物：transformers 的 DeepseekV3TopkRouter / DeepseekV4TopKRouter / DeepseekV3MoE / GptOssTopKRouter

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

  ④ **四个 routing 取值的关系**（routing_kinds）：同一批 router 权重上一起跑，     拍下 by1 真实选出的专家与权重。实测：softmax_topk 与 topk_softmax     **是同一个函数**（权重差 8.9e-08 = 那个 1e-9 eps）；topk_softmax 对官方     GptOssTopKRouter 逐位一致；sigmoid_topk 与 softmax_topk 选中的专家**相同**     但权重不同（输出差 1.2e-02）；sigmoid_group_topk 比 sigmoid_topk 多 6/12 行不同。


用法:  python by1moe.py
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

import by1codegen as cg
import by1ir
import by1skip
import numpy as np
import by1exec
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE, DeepseekV3TopkRouter)
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


# 判卷人现在验三层（2026-10-17 第十七轮）：
#
#   ① 路由式子 vs 官方 router（原来就有）；
#   ② **整块**：官方 `DeepseekV3MoE`（路由 + 专家 + 共享 + 缩放）-> `by1exec.op_moe`；
#   ③ **专家路径单独对拍**：官方权重手算逐专家 MLP（`shared=0`）。
#
# **一个很隐蔽的坑**：官方模块独立构造时
# `experts.gate_up_proj` / `down_proj` / `shared_experts.*` 是 `torch.empty`（未初始化），
# 实测恰好全零 —— 于是“官方 experts 返回 0”、“换掉 w1/w3 结果也不变”，
# 看上去还很一致（0 == 0）。判卷人必须**自己给权重**；
# 另外专家参考要**逐 token** 写（批量写法会轰出错形状，而结果又恰好是 0）。

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
    ok4 = block_direct_link(cfg, torch)
    ok5 = expert_direct_link(cfg, torch)
    ok6 = routing_kinds()
    return 0 if (ok and ok2 and ok3 and ok4 and ok5 and ok6) else 1


def block_direct_link(cfg, torch):
    """**整块直连**：官方 `DeepseekV3MoE`（路由 + 专家 + 共享 + 缩放）-> `by1exec.op_moe`。

    上面那两段验的是"式子"（判卷人里另写一遍）与 config 对账；真正跑 IR/图那条路的是
    `by1exec.op_moe`。两条链各自对上官方，不等于**彼此**一致。

    官方把专家权重**融合**了：`experts.gate_up_proj[e]` 是 `(2*hidden, d)`，
    `linear(x, W).chunk(2, -1)` 给出 `(gate, up)` —— 也就是**前半行是 gate、后半行是 up**。
    by1 的 `op_moe` 吃的是分开的 `w1/w3`，映射就落在这条分界上。
    """
    torch.manual_seed(1)
    cfg = cfg.__class__(**{**cfg.to_dict(), "_experts_implementation": "eager"})
    blk = DeepseekV3MoE(cfg).eval()
    # **官方模块独立构造时专家权重是 `torch.empty`（未初始化）** ——
    # 实测恰好是**全零**，于是"官方 experts 返回 0"、"换 w1/w3 也不变"。
    # 那不是上游的 bug，是**没人调 post_init**；判卷人自己给权重。
    with torch.no_grad():
        for _p in (blk.experts.gate_up_proj, blk.experts.down_proj,
                   blk.shared_experts.gate_proj.weight,
                   blk.shared_experts.up_proj.weight,
                   blk.shared_experts.down_proj.weight):
            _p.normal_(0.0, 0.05)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in blk.state_dict().items()}
    mi = int(cfg.moe_intermediate_size)
    gu = sd["experts.gate_up_proj"]
    P = {
        "router": sd["gate.weight"],
        "score_bias": sd["gate.e_score_correction_bias"],
        "w1": gu[:, :mi, :],
        "w3": gu[:, mi:, :],
        "w2": sd["experts.down_proj"],
        "sw1": sd["shared_experts.gate_proj.weight"],
        "sw3": sd["shared_experts.up_proj.weight"],
        "sw2": sd["shared_experts.down_proj.weight"],
    }
    at = {"experts": int(cfg.n_routed_experts), "top_k": int(cfg.num_experts_per_tok),
          "hidden": mi, "shared": 1, "shared_hidden": mi,
          "score_bias": True, "router_bias": False,
          "routed_scale": float(cfg.routed_scaling_factor),
          "routing": "sigmoid_group_topk", "act": "silu",
          "limit": None, "alpha": 1.702, "expert_bias": False,
          "shared_gate": False, "n_group": int(cfg.n_group),
          "topk_group": int(cfg.topk_group)}
    x = torch.randn(1, NT, D)
    with torch.no_grad():
        ref = blk(x).numpy().astype(np.float64)
    got = by1exec.op_moe(P, at, [x.numpy().astype(np.float32)], D)
    got = np.asarray(got, dtype=np.float64)
    dd = float(np.abs(got - ref).max())
    rel = dd / max(float(np.abs(ref).max()), 1e-12)
    good = rel < 1e-6
    print("\n  整块直连：官方 DeepseekV3MoE -> by1exec.op_moe")
    print("     绝对差 %.3e  相对 %.3e  %s"
          % (dd, rel, "后端 op 与官方一致 ok" if good else "**不一致**"))
    return good

def expert_direct_link(cfg, torch):
    """**专家路径单独对拍**（整块对拍覆盖不到它）。

    坑：本环境里官方 `DeepseekV3Experts.forward` **返回全零** ——
    `@use_experts_implementation` 的专家实现没装/没分派上，装饰器把整段
    算成 0（实测：`m.experts(...)` 的 absmax = 0.0，换 w1/w3 结果都不变）。
    所以"整块对拍"只证明路由 + 共享 + 缩放；**专家路径要单独比**：用官方权重
    （`gate_up_proj` 前半 gate、后半 up）手算逐专家 MLP，再乘官方路由权重。
    """
    torch.manual_seed(2)
    blk = DeepseekV3MoE(cfg).eval()
    # **官方模块独立构造时专家权重是 `torch.empty`（未初始化）** ——
    # 实测恰好是**全零**，于是"官方 experts 返回 0"、"换 w1/w3 也不变"。
    # 那不是上游的 bug，是**没人调 post_init**；判卷人自己给权重。
    with torch.no_grad():
        for _p in (blk.experts.gate_up_proj, blk.experts.down_proj,
                   blk.shared_experts.gate_proj.weight,
                   blk.shared_experts.up_proj.weight,
                   blk.shared_experts.down_proj.weight):
            _p.normal_(0.0, 0.05)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in blk.state_dict().items()}
    mi = int(cfg.moe_intermediate_size)
    gu = sd["experts.gate_up_proj"]
    P = {
        "router": sd["gate.weight"],
        "score_bias": sd["gate.e_score_correction_bias"],
        "w1": gu[:, :mi, :],
        "w3": gu[:, mi:, :],
        "w2": sd["experts.down_proj"],
    }
    at = {"experts": int(cfg.n_routed_experts),
          "top_k": int(cfg.num_experts_per_tok),
          "hidden": mi, "shared": 0, "shared_hidden": 0,
          "score_bias": True, "router_bias": False,
          "routed_scale": float(cfg.routed_scaling_factor),
          "routing": "sigmoid_group_topk", "act": "silu",
          "limit": None, "alpha": 1.702, "expert_bias": False,
          "shared_gate": False, "n_group": int(cfg.n_group),
          "topk_group": int(cfg.topk_group)}
    x = torch.randn(1, NT, D)
    with torch.no_grad():
        # 官方路由（权重**是**官方模块算的；只有专家前向是坏的）
        _, wgt, idx = blk.gate(x)
        xf = x.view(-1, D)
        # **逐 token 的朴素写法**：每个 token 的每个被选专家各自一遍。
        # 第一版写成 `xe @ gu[e, :mi, :].transpose(...)` —— 两边都是**批量**，
        # torch 广播出一个错形状，而结果恰好是 0，看上去"一致"（0 == 0）。
        # 判据里最危险的正是这种"全零也对得上"。
        gu_t = blk.experts.gate_up_proj
        dp_t = blk.experts.down_proj
        ref = torch.zeros_like(xf)
        for t in range(xf.shape[0]):
            for j in range(idx.shape[1]):
                e = int(idx[t, j])
                g = xf[t] @ gu_t[e, :mi, :].T
                u = xf[t] @ gu_t[e, mi:, :].T
                y = (torch.nn.functional.silu(g) * u) @ dp_t[e].T
                ref[t] = ref[t] + y * wgt[t, j]
        if float(ref.abs().max()) == 0.0:
            # **不许用 assert 当控制流**（lint ⑤：`-O` 一跑就没了）
            raise RuntimeError("专家参考全零 —— 这条判据是空的")
        ref = ref.numpy().astype(np.float64)
    got = by1exec.op_moe(P, at, [x.numpy().astype(np.float32)], D)
    got = np.asarray(got, dtype=np.float64).reshape(-1, D)  # op_moe 出 [b,n,d]
    dd = float(np.abs(got - ref).max())
    rel = dd / max(float(np.abs(ref).max()), 1e-12)
    good = rel < 1e-6
    print("\n  专家路径直连：官方权重手算逐专家 MLP -> by1exec.op_moe（shared=0）")
    print("     绝对差 %.3e  相对 %.3e  %s"
          % (dd, rel, "后端专家路径一致 ok" if good else "**不一致**"))
    return good

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


def routing_kinds():
    """四个 `routing` 取值：**到底有几个是真不同的函数**。

    闭集里有四个写法（`softmax_topk` / `topk_softmax` / `sigmoid_topk` /
    `sigmoid_group_topk`），但没有任何判据说过它们之间的关系。这一条把它们放在
    **同一批 router 权重**上跑，抓 by1 真实选出的下标与权重（替换 RUNTIME 命名空间里的
    `_stable_topk` / `_stable_topk2` 来录制，**不重写公式**），然后：

      ① `topk_softmax` 对官方 `GptOssTopKRouter`（logits -> top-k -> 只对 k 个 softmax）
         —— 下标与权重都要一致；
      ② `softmax_topk` 与 `topk_softmax`：**数学上应当是同一个函数**
         （softmax 全体取 top-k 再归一 ≡ 只对那 k 个 logits 做 softmax）。
         如果这条**没通过**，说明两支里有一支算错了；
      ③ `sigmoid_topk` 与 `softmax_topk`：**选中的专家相同、权重不同**（两者都单调，
         所以 top-k 选出的集合必然一样；差在 sigmoid 归一 vs softmax 归一）；
      ④ `sigmoid_group_topk` 与 `sigmoid_topk`：必须不同（多了"先选组"）。

    ②的措辞是刻意的：它不是"两个名字应该一样所以算它过"，而是
    "**如果它们不一样，就有一个是错的**"。
    """
    ns = {}
    exec(cg.RUNTIME, ns)
    MoE = ns["MoE"]
    orig1, orig2 = ns["_stable_topk"], ns["_stable_topk2"]
    cap = {"idx": None, "val": None}

    def rec1(v, k, dim=-1):
        out = orig1(v, k, dim)
        cap["idx"] = out.detach().clone()
        return out

    def rec2(v, k, dim=-1):
        tv, ti = orig2(v, k, dim)
        cap["idx"] = ti.detach().clone()
        cap["val"] = tv.detach().clone()
        return tv, ti

    ns["_stable_topk"] = rec1
    ns["_stable_topk2"] = rec2

    torch.manual_seed(0)
    gw = torch.randn(E, D) * 0.1        # router 权重（四支共用）
    x = torch.randn(1, NT, D)
    base = {"experts": E, "top_k": K, "hidden": 8, "shared": 0,
            "shared_hidden": 0, "score_bias": False, "routed_scale": 1.0,
            "router_bias": False, "act": "silu", "limit": None,
            "alpha": 1.702, "expert_bias": False, "shared_gate": False,
            "n_group": NG, "topk_group": TG, "score": "softmax"}

    def run(kind, score="softmax", score_bias=False):
        at = dict(base)
        at["routing"] = kind
        at["score"] = score
        at["score_bias"] = score_bias
        torch.manual_seed(0)          # 四支专家权重逐位相同
        m = MoE(at, D).eval()
        with torch.no_grad():
            m.router.weight.copy_(gw)
        cap["idx"] = cap["val"] = None
        with torch.no_grad():
            out = m(x).detach()
        return cap["idx"], cap["val"], out

    ok = True
    i_ts, v_ts, o_ts = run("topk_softmax")
    i_st, v_st, o_st = run("softmax_topk")
    i_sg, _, o_sg = run("sigmoid_topk", score="sigmoid")
    i_gr, _, o_gr = run("sigmoid_group_topk", score="sigmoid")

    # ---- ① topk_softmax 对官方 GptOssTopKRouter ----
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
    cfg = GptOssConfig(hidden_size=D, num_local_experts=E,
                       num_experts_per_tok=K, intermediate_size=8,
                       num_hidden_layers=1)
    ref = GptOssTopKRouter(cfg).eval()
    with torch.no_grad():
        ref.weight.copy_(gw)
        ref.bias.zero_()
        _, r_scores, r_idx = ref(x.view(-1, D))
    same_idx = torch.equal(r_idx.sort(-1)[0], i_ts.sort(-1)[0])
    rmap = torch.zeros(NT, E).scatter_(1, r_idx, r_scores)
    mmap = torch.zeros(NT, E).scatter_(1, i_ts, v_ts.softmax(-1))
    d1 = (rmap - mmap).abs().max().item()
    print("  ① topk_softmax vs 官方 GptOssTopKRouter: 下标%s  权重差 %.3e  [%s]"
          % ("一致" if same_idx else "不同", d1,
             "PASS" if (same_idx and d1 < 1e-6) else "FAIL"))
    if not (same_idx and d1 < 1e-6):
        ok = False

    # ---- ② softmax_topk 与 topk_softmax 应当是同一个函数 ----
    same2 = torch.equal(i_st.sort(-1)[0], i_ts.sort(-1)[0])
    smap = torch.zeros(NT, E).scatter_(
        1, i_st, v_st / (v_st.sum(-1, keepdim=True) + 1e-9))
    d2 = (smap - mmap).abs().max().item()
    print("  ② softmax_topk 与 topk_softmax: 下标%s  权重差 %.3e  [%s]"
          % ("一致" if same2 else "不同", d2,
             "PASS" if (same2 and d2 < 1e-6) else "FAIL"))
    if not (same2 and d2 < 1e-6):
        print("     ! 这两个写法算出来不一样 —— 其中一支的实现是错的")
        ok = False

    # ---- ③ sigmoid_topk vs softmax_topk：**选出来的专家相同、权重不同** ----
    # sigmoid 与 softmax 都是单调的，所以 top-k 选中的专家**必然相同**
    # （这一点原来被我写成"选择必须不同"，跑出来 0/12 —— 我的假设错了）。
    # 真正不同的是**权重**：sigmoid 打分归一 vs softmax 归一。所以比整块输出。
    d3same = sum(1 for r in range(NT)
                 if set(i_sg[r].tolist()) != set(i_ts[r].tolist()))
    d3 = float((o_sg - o_st).abs().max())
    ok3 = (d3same == 0) and d3 > 1e-3
    print("  ③ sigmoid_topk vs softmax_topk: 选中专家相同(%d/%d 处不同)"
          " · 输出差 %.3e  [%s]"
          % (d3same, NT, d3, "PASS" if ok3 else "FAIL"))
    if not ok3:
        print("     ! 打分函数换了却没改变权重（或者连选中的专家都变了）")
        ok = False

    # ---- ④ 分组 vs 不分组必须不同 ----
    d4 = sum(1 for r in range(NT)
             if set(i_gr[r].tolist()) != set(i_sg[r].tolist()))
    print("  ④ sigmoid_group_topk vs sigmoid_topk: 有 %d/%d 行选择不同  [%s]"
          % (d4, NT, "PASS" if d4 else "FAIL"))
    if not d4:
        print("     ! 分组与不分组选出同样的专家 —— 分组没生效")
        ok = False

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
