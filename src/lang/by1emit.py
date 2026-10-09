#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 emit -- 从 IR 生成 **ggml 侧的后端需求 + 声明式图**，并和 GGUF manifest 双向对拍。

背景：这个环境里没有 C 编译器、没有 llama.cpp、github 不通。所以「生成 ggml 代码」
这一步**无法验证**，写出来只能看起来对 —— 那不写。

能验的是另一半：**图引用哪些张量**这件事，可以拿已经验证过的 GGUF manifest 来核。
两个独立生成器、同一份 IR，引用的名字必须**双向**一致：
  · 图引用的每个张量都在 manifest 里
  · manifest 里的每个张量都被图引用

这条通过，3b 剩下的就只是「把这些原语写成 C」——苦工，不是未知。

用法:
  python by1emit.py gpt-oss-120b.by1
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

import argparse
import importlib.util
import os
import sys

import by1io      # noqa: F401  —— import 即把 stdout 钉成 UTF-8

HERE = os.path.dirname(os.path.abspath(__file__))

# IR 的算子 -> ggml 原语序列。这张表就是 3b 要写的东西。
GGML = {
    "Norm": ["ggml_rms_norm", "ggml_mul"],
    "Add": ["ggml_add"],
    "FFN": ["ggml_mul_mat", "ggml_silu", "ggml_mul", "ggml_mul_mat"],
    "MoE": ["ggml_mul_mat(router)", "ggml_top_k", "ggml_soft_max",
            "ggml_mul_mat(专家, 逐专家)", "ggml_mul_mat(down)", "ggml_add"],
    "Attention": ["ggml_mul_mat(qkv)", "ggml_rope_ext", "ggml_mul_mat(qk)",
                  "ggml_soft_max_ext(掩码, sink)", "ggml_mul_mat(av)",
                  "ggml_mul_mat(o)"],
    # 来自 src/models/delta-net-base.cpp：delta 规则有三个实现
    #   build_delta_net_chunking      分块（prefill）
    #   build_delta_net_autoregressive 顺序（decode，显式 state）
    #   build_delta_net_fused          融合 kernel
    # 核心递推 state = state * g + kgdmulvnew，与 IR 一致；
    # 分块版只用标准原语（cumsum / tri / exp / sub / pad）就能表达。
    # 状态类型是 llama-memory-recurrent.h —— 正对应 IR 的 "recurrent" 声明。
    "Linear": ["ggml_mul_mat(qkvz)", "ggml_mul_mat(ba)", "ggml_ssm_conv(短卷积)",
               "ggml_silu", "delta 规则: cumsum/tri/exp/sub (分块) 或顺序 state",
               "build_norm_gated", "ggml_mul_mat(out)"],
}
# 这些不是「写几行」能解决的，得单独说清楚
CUSTOM = {
}


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 ggml 后端需求与图")
    ap.add_argument("by1")
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    name = os.path.basename(args.by1)
    _r, info = bc.check(args.by1)
    ir = cg.compile_ir(info)

    print(f"\n{'='*76}\n  {name}  ->  ggml 后端\n{'='*76}")

    # ── 1. 后端必须实现什么 ────────────────────────────────────────
    kinds, prims, custom = {}, [], {}
    for L in ir["layers"]:
        for o in L["ops"]:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    print("\n  [1] 后端必须实现的算子")
    for k in sorted(kinds):
        mark = "  ⚠ 需要自定义 kernel" if k in CUSTOM else ""
        print(f"    {k:<10} {kinds[k]:>4} 个    {', '.join(GGML.get(k, ['?']))}{mark}")
        if k in CUSTOM:
            custom[k] = CUSTOM[k]
    for k, v in custom.items():
        print(f"      {k}: {v}")
    miss = [k for k in kinds if k not in GGML]
    if miss:
        print(f"    [缺口] 还没有对应原语的算子：{miss}")

    # ── 2. 声明式图（逐层，只画前两层示意 + 最后一层） ──────────────
    print("\n  [2] 声明式图（数据流 + 每步调用的原语）")
    shown = 0
    for L in ir["layers"]:
        for j, o in enumerate(L["ops"]):
            if shown < 14:
                ins = " , ".join(o["inputs"])
                pr = GGML.get(o["kind"], ["?"])
                print(f"    L{L['index']:<3} op{j}  {ins:<22} -> "
                      f"{o['outputs'][0]:<10} {o['kind']:<10} [{len(pr)} 步]")
                shown += 1
    if len(ir["layers"]) * 6 > 14:
        print(f"    ...（{len(ir['layers'])} 层 × 每层 "
              f"{len(ir['layers'][0]['ops'])} 个算子，此处省略）")

    # ── 3. 与 GGUF manifest 双向对拍 ───────────────────────────────
    print("\n  [3] 与 GGUF manifest 双向对拍")
    rules = info.get("emit") or {}
    rule = rules.get("ggml")
    if not rule:
        print("    (这份 .by1 里没有 emit[ggml] 规则，跳过)")
        return 0
    scope = rule.get("scope", {})
    # 图会引用哪些张量：每个算子的机制名 -> 该机制在契约里声明的逻辑张量
    # info["tens"] 是 [(机制, 标签, 层数, [(逻辑名, 形状, 注, per_expert), ...]), ...]
    contracts = {}
    for ent in (info.get("tens") or []):
        blk, rows = ent[0], ent[3]
        contracts.setdefault(blk, []).extend(t[0] for t in rows)
    refs = set()
    mech_of = {}
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] in ("Norm", "Add"):
                continue
            mech_of.setdefault(L["index"], set()).add(o["mech"])
            for t in contracts.get(o["mech"], []):
                refs.add((L["index"], o["mech"], t))
    # 渲染成 ggml 名字
    named = set()
    for (i, mech, t) in refs:
        named.add(bc.render_name(rule, i, "main", mech, t))
    # manifest：用同一套规则把契约里**所有**张量渲染出来
    manifest = set()
    for ent in (info.get("tens") or []):
        blk, rows = ent[0], ent[3]
        for L in ir["layers"]:
            if blk in mech_of.get(L["index"], set()):
                for (t, _shp, _note, _pe) in rows:
                    manifest.add(bc.render_name(rule, L["index"], "main", blk, t))
    a, b = named - manifest, manifest - named
    print(f"    图引用的张量      {len(named):>7}")
    print(f"    manifest 里的张量 {len(manifest):>7}")
    print(f"    图引用但 manifest 没有  {len(a):>5}"
          + (f"   例如 {sorted(a)[:3]}" if a else "   ✓"))
    print(f"    manifest 有但图没引用  {len(b):>5}"
          + (f"   例如 {sorted(b)[:3]}" if b else "   ✓"))
    ok = not a and not b and not miss
    print(f"\n  [{'PASS' if ok else 'FAIL'}] "
          + ("图与 manifest 双向一致 —— 3b 剩下的只是把原语写成 C"
             if ok else "两边对不上，说明绑定还没做完"))
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
