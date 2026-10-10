#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1sparse -- **DSA 稀疏索引器**的差分验证。

把 by1 生成的 `DSAIndexer` 和 **transformers 的 `GlmMoeDsaIndexer`** 放在一起，
喂同样的权重、同样的输入，比"每个 query 挑中了哪些 key"。

判卷人是**别人写的**：`transformers.models.glm_moe_dsa`（DeepSeek Sparse
Attention 那一支）。不是自己和自己比。

## 它验到哪一步

**验过**：索引器的核心 —— `wq_b` / `wk` / `k_norm` / `weights_proj` 四个投影、
ReLU、按头加权求和、因果、top-k。GLM-5.3-Flash 的 `index_heads=32` /
`index_dim=128` / `index_topk=2048` 走的就是这条路。

**没验（已知缺口）**：GLM-5.3-Flash 的 checkpoint 上还有两个张量
`index_kpool_compress_gate` / `index_kpool_compress_ape`，而**整套依赖里
找不到任何官方实现可比**（实测 transformers 5.15.1 全库 0 处命中
`kpool` / `compress_ape`）。所以那两个**不实现、也不假装验过** ——
它挡住的是 GLM-5.3-Flash 的**完整**前向，不是索引器本身。

## 反例

一条只会通过的规则不是规则。所以末尾逐个把实现改坏一处，
要求判卷人**当场变红**（见 `NEG`）。

用法:  python by1sparse.py
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
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig)


# ── 几组维度 ────────────────────────────────────────────────────────
# (名字, d_model, q_lora, index_heads, index_dim, qk_rope, index_topk, seq)
# `qk_rope=0` 是 **GLM-5.3-Flash 自己的取值**（qk_rope_head_dim = 0，
# 它的 MLA 完全不用 RoPE）；`qk_rope>0` 是 DSA 的默认，用来**把 RoPE
# 那条路也走到** —— 只测 0 的话，RoPE 写错了也看不出来。
#
# **`topk` 必须远小于 seq**，否则"挑中了哪些"退化成"全都要"，
# 判据就瞎了（踩过一次：topk=8 / seq=12，改坏 k_norm 居然照样一致）。
CASES = [
    ("GLM-5.3 形状·全取", 1536, 1536, 32, 128, 0, 2048, 12),
    ("GLM-5.3 形状·topk=3", 1536, 1536, 32, 128, 0, 3, 32),
    ("DSA 默认(rope=64)", 512, 256, 8, 64, 64, 3, 32),
    ("小尺寸(rope=16)", 64, 32, 4, 32, 16, 4, 24),
]


def build(nh, hd, qk_rope, topk, d, ql, seq, seed=0):
    """一对同权重的 (by1, 参考) 索引器。"""
    torch.manual_seed(seed)
    cfg = GlmMoeDsaConfig(
        hidden_size=d, q_lora_rank=ql, qk_rope_head_dim=qk_rope,
        index_n_heads=nh, index_head_dim=hd, index_topk=topk,
        num_attention_heads=nh, num_hidden_layers=1)
    ref = GlmMoeDsaIndexer(cfg, 0).eval()
    ns = {}
    exec(cg.RUNTIME, ns)
    mine = ns["DSAIndexer"]({
        "index_heads": nh, "index_dim": hd, "index_topk": topk,
        "q_lora": ql, "qk_rope": qk_rope,
        "pairing": "interleaved", "rope_base": 10000}, d).eval()
    mine.load_state_dict(ref.state_dict(), strict=True)
    return mine, ref, cfg


def run_ref(ref, x, q_resid, seq):
    # **cos/sin 要给到 rope 维那么宽，参考自己会再折一半。**
    # `apply_rotary_pos_emb_interleave` 第一件事就是
    # `cos = cos[..., : cos.shape[-1] // 2]`（配对角在前一半里）。
    # 给成 `// 2` 会让它折成 1/4 —— 实测当场 RuntimeError
    # `size of tensor a (32) must match ... b (16)`。
    # 而 `qk_rope_head_dim = 0`（GLM-5.3-Flash 自己的取值）宽度就是 0，
    # 折完还是 0，跑得动。
    w = ref.qk_rope_head_dim
    cos = torch.randn(1, seq, w)
    sin = torch.randn(1, seq, w)
    pos = torch.arange(seq)[None]
    with torch.no_grad():
        out = ref(x, q_resid, (cos, sin), None, pos)
    return out, (cos, sin), pos


def agree(a, b):
    """两个 [1, S, K] 的挑选集合：每个 query 的选中集合都要相同。

    **K 不同就是不同**（"topk 取满"那条反例正是靠它变红的）——
    不做形状判断的话，sort 那一步会当场崩，而崩不是"变红"。"""
    if tuple(a.shape) != tuple(b.shape):
        return False
    return bool((torch.sort(a[0], dim=-1).values
                 == torch.sort(b[0], dim=-1).values).all())


def run_pair(mine, ref, seq, pairing, seed=0):
    torch.manual_seed(seed + 1)
    x = torch.randn(1, seq, ref.hidden_size)
    q_resid = torch.randn(1, seq, ref.q_lora_rank)
    ref_out, (cos, sin), pos = run_ref(ref, x, q_resid, seq)
    mine.pairing = pairing
    with torch.no_grad():
        # 参考把 cos 折一半（配对角在前一半里），而我们这边 `rope_tables`
        # 给的就是折过的；`apply_rope` 还要求是 [seq, half]（它自己补前两维）。
        my_out = mine(x, q_resid, cos[0, :, : cos.shape[-1] // 2],
                      sin[0, :, : sin.shape[-1] // 2])
    return agree(my_out, ref_out), my_out, ref_out


# ── 反例：改坏一处，判卷人必须红 ───────────────────────────────────
#
# **有一类改动是这个判据天生看不见的，先说清楚**：top-k 的"挑中集合"
# 对**正的整体缩放**不变。所以 `softmax_scale = index_dim^-0.5` 和
# `weights_proj(x) * index_heads^-0.5` 这两个系数**写错了也挑出同一批 key** ——
# 它们只调大小、不调次序。这不等于它们不重要（下游加法掩码用的是绝对值），
# 但**"挑选集合"这条尺子量不到它们**。所以不拿它们当反例 ——
# 假的反例比没有反例更糟。
#
# 能看见的是**会改次序 / 改集合**的那些：
#
#   (标题, 为什么它该被看见, 把源码换成什么)
NEG = [
    ("去掉 ReLU",
     "DSA 的打分非负 —— 不 ReLU 会把负相关也当证据，次序就变了",
     "scores = torch.relu(scores)",
     "scores = scores"),
    ("rope 配对约定用错",
     "参考是 `apply_rotary_pos_emb_interleave`（奇偶配对）；"
     "换成对半配对，同一组 cos/sin 就转出另一套坐标",
     "self.pairing).transpose(1, 2)",
     '"half").transpose(1, 2)'),
    ("不套因果",
     "不套因果会让 query 挑到未来的 key —— 集合当场不同",
     'idx = idx.masked_fill(pos[None, :] > pos[:, None], float("-inf"))',
     "idx = idx"),
    ("topk 取满",
     "取满等于没稀疏，集合大小都不一样",
     "t = min(self.topk, n)",
     "t = n"),
    ("权重取负",
     "按头加权求和，权重反号就是另一个排序",
     "w = self.weights_proj(x).float() * (self.nh ** -0.5)",
     "w = -self.weights_proj(x).float() * (self.nh ** -0.5)"),
    ("丢掉第一个头",
     "少算一个头的证据，排序会挪",
     "w = self.weights_proj(x).float() * (self.nh ** -0.5)",
     "w = self.weights_proj(x).float().index_fill("
     "-1, torch.tensor([0], device=x.device), 0.0) * (self.nh ** -0.5)"),
    ("k_norm 只留仿射",
     "参考里 k 先过 LayerNorm 再用；只留仿射、不归一化，尺度就变了"
     "（**不能整个拿掉** —— 那样 load_state_dict(strict=True) 会先崩）",
     "k = self.k_norm(self.wk(x)).unsqueeze(2)",
     "k = (self.wk(x) * self.k_norm.weight"
     " + self.k_norm.bias).unsqueeze(2)"),
    ("对头直接求和",
     "参考是按 `weights_proj(x)` 给每个头加权再求和；直接求和是另一套证据",
     "w = self.weights_proj(x).float() * (self.nh ** -0.5)",
     "w = torch.ones_like(self.weights_proj(x).float())"),
]


def negatives():
    """逐条把实现改坏一处，要求判卷人**当场变红**。"""
    print("\n  反例（**改坏一处，判卷人必须红**）:")
    bad = []
    src = cg.RUNTIME
    # **反例的配置要让判据看得见**：topk 远小于 seq；rope 两半**不等宽**
    # （qk_rope != index_dim - qk_rope），否则"rope 那一半放后面"是个空操作
    # —— 第一版就是 [32,32] 对 [32,32]，怎么改都一样，白测一场。
    nh, hd, qk_rope, topk, d, ql, seq = 8, 64, 16, 3, 512, 256, 32
    for title, why, old, new in NEG:
        mutated = src.replace(old, new)
        if mutated == src:
            print("  !! %-22s 反例的原句找不到" % title)
            bad.append(title)
            continue
        ns = {}
        exec(mutated, ns)
        torch.manual_seed(0)
        cfg = GlmMoeDsaConfig(
            hidden_size=d, q_lora_rank=ql, qk_rope_head_dim=qk_rope,
            index_n_heads=nh, index_head_dim=hd, index_topk=topk,
            num_attention_heads=nh, num_hidden_layers=1)
        ref = GlmMoeDsaIndexer(cfg, 0).eval()
        mine = ns["DSAIndexer"]({
            "index_heads": nh, "index_dim": hd, "index_topk": topk,
            "q_lora": ql, "qk_rope": qk_rope,
            "pairing": "interleaved", "rope_base": 10000}, d).eval()
        mine.load_state_dict(ref.state_dict(), strict=True)
        try:
            ok, _a, _b = run_pair(mine, ref, seq, "interleaved")
        except Exception as e:
            # 崩了也算"没对上"，但**要分清是崩还是红** —— 崩可能是反例
            # 自己写坏了。所以单独说一句。
            print("  !! %-22s 改坏后跑崩了（%s）—— 这条反例没证到什么"
                  % (title, type(e).__name__))
            bad.append(title + '(崩)')
            continue
        if ok:
            print("  !! %-22s **改坏了却还一致** —— %s" % (title, why))
            bad.append(title)
        else:
            print("  ok %-22s 判卷人当场变红（挑选集合不同）" % title)
    return bad


def main():
    print("=" * 74)
    print("  DSA 稀疏索引器差分验证   by1 的 DSAIndexer  vs  "
          "transformers 的 GlmMoeDsaIndexer")
    print("=" * 74)
    print("\n  判卷人是 transformers 里别人写的实现 —— 不是自己和自己比。")
    print("  比的是**每个 query 挑中了哪些 key**（挑选集合逐个相同）。")

    ns = {}
    exec(cg.RUNTIME, ns)
    ok_all = True
    pairing_seen = {}
    for label, d, ql, nh, hd, qk_rope, topk, seq in CASES:
        print("\n  %-22s d=%d q_lora=%d heads=%d dim=%d rope=%d topk=%d seq=%d"
              % (label, d, ql, nh, hd, qk_rope, topk, seq))
        mine, ref, _cfg = build(nh, hd, qk_rope, topk, d, ql, seq)
        ok_i = False
        for pairing in ("interleaved", "half"):
            ok, a, b = run_pair(mine, ref, seq, pairing)
            pairing_seen.setdefault(pairing, 0)
            pairing_seen[pairing] += 1 if ok else 0
            if pairing == "interleaved":
                ok_i = ok
            print("     pairing=%-12s 挑选集合 %s"
                  % (pairing, "逐个相同" if ok else "**不同**"))
        # **判据取 `interleaved`** —— 实测 4/4 组它全对，`half` 只在
        # rope 不起作用的两组（rope=0）碰巧一致。和 GLM 的
        # `indexer_rope_interleave = true` 也是同一句话。
        ok_all = ok_all and ok_i

    print("\n  哪个 pairing 对得上参考（计数 / %d 组）:" % len(CASES))
    for k, v in pairing_seen.items():
        print("     %-12s %d" % (k, v))

    bad = negatives()

    print("\n  [%s] DSA 索引器 %s"
          % ("PASS" if ok_all and not bad else "FAIL",
             "与参考一致（挑选集合逐个相同），反例全红"
             if (ok_all and not bad) else
             ("对不上参考" if not ok_all else "反例没全红")))
    return 0 if (ok_all and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
