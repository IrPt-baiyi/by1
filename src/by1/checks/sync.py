#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1sync -- **推导 ↔ 契约 的同步报表**（1.0.0-rc0 的 T0.2）。

同一个事实有两个家：

  ① `by1ir.params_of(ir)`   从 kind+attrs **推导**（后端吃的那份）
  ② `.by1` 的 `tensors{}`    模型**声明**的契约（对 refs 验过的那份）

历史教训：gemma-4-31B 的全局张量被推成 (262144, **1152**)
（尾层视觉塔的宽度），而契约说 (vocab, **5376**) —— 两个家不一致，
**谁都没报警**，直到 GGUF 判卷人把 5376 摆出来（见
`docs/1.0-roadmap.md` T0.2 与 1.md 的 shapes_of bug）。

本工具把两个家摆在一起 diff。**第一版是报表，不是门禁** ——
因为契约是**产物布局**（gpt2 的 `c_attn.weight (64,192)` 是融合+转置的
Conv1D），推导是**计算布局**（拆开的 wq/wk/wv）：布局变换
（fuse / transpose / quant-block）还没逐个编码。编码完才配进门禁。

用法:
  python by1sync.py                # models/ 全部报表
  python by1sync.py <model.by1>    # 单个
  python by1sync.py --selftest     # 反例必红：复现 1152 bug
"""
import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块可能在包内深处（src/by1/checks/…），固定层数会插错目录。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.exists(_os.path.join(_p, 'by1paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import importlib.util

import by1io      # noqa: F401  —— import 即钉 UTF-8


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_shape(s):
    """契约里的形状串 -> 元组。`--`（被抑制/权重共享）-> None。"""
    s = (s or "").strip()
    if s in ("--", ""):
        return None
    inner = s.strip("()").replace(" ", "")
    if not inner:
        return None
    return tuple(int(x) for x in inner.split(",") if x != "")


def derived(ir, params):
    """推导侧：按机制名分组的形状集合。globals 单列、层级 norm 归 `layer`。"""
    groups = {"@global": set(), "layer": set()}
    for name, shp in params.items():
        if not name.startswith("layers."):
            groups["@global"].add(tuple(shp))
    for L in ir["layers"]:
        for j, o in enumerate(L["ops"]):
            pre = "layers.%d.op%d." % (L["index"], j)
            # **层级 norm 归 `layer` 组** —— 契约那边它们写在 `layer {}`
            # 块里（ln_1/ln_2 这类），块名和机制名不是一回事。
            key = "layer" if o.get("kind") == "Norm" else \
                (o.get("mech") or o.get("kind"))
            g = groups.setdefault(key, set())
            for name, shp in params.items():
                if name.startswith(pre):
                    g.add(tuple(shp))
    return groups


def contracted(info):
    """契约侧：按块分组的形状集合。`--` 行不算（被抑制/权重共享）。"""
    groups = {"@global": set(), "layer": set()}
    for ent in (info.get("tens") or []):
        blk, rows = ent[0], ent[3]
        g = groups.setdefault(blk, set())
        for t in rows:
            shp = parse_shape(t[1])
            if shp is not None:
                g.add(shp)
    # **层级行**（`layer {}` 块：ln_1 / ln_2 / input_layernorm 这些）——
    # 藏在 layer_out 的 (blk, row) 对里，块名统一归 `layer`。
    for ent in (info.get("layer_out") or []):
        for item in (ent[3] if len(ent) > 3 else []):
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            blk, row = item[0], item[1]
            key = "layer" if str(blk).lower() in ("layer", "norm",
                                                  "layernorm") else blk
            shp = parse_shape(row[1] if isinstance(row, (list, tuple))
                              and len(row) > 1 else "")
            if shp is not None:
                groups.setdefault(key, set()).add(shp)
    # 全局行（`global { ... }` 块）
    g = groups["@global"]
    for t in (info.get("global_rows") or []):
        shp = parse_shape(t[1] if isinstance(t, (list, tuple)) else "")
        if shp is not None:
            g.add(shp)
    return groups


def sync_one(path, bc, cg, bir, quiet=False):
    _r, info = bc.check(path)
    ir = cg.compile_ir(info)
    params = bir.params_of(ir)
    dg, cg_ = derived(ir, params), contracted(info)
    name = _os.path.basename(path)
    # **合成模型没有 `tensors{}` 契约**（llama-shaped 这类是结构对照物，
    # 不是对产物的声明）—— 没有第二个家就不算漂移，跳过。
    if not any(cg_.values()):
        if not quiet:
            print(f"\n  {name}  （无契约声明 —— 合成模型，跳过）")
        return 0, []
    rows = []
    for blk in sorted(set(dg) | set(cg_)):
        d, c = dg.get(blk, set()), cg_.get(blk, set())
        if not d and not c:
            continue
        only_d, only_c = d - c, c - d
        rows.append((blk, len(d & c), len(d), len(c), sorted(only_d)[:2],
                     sorted(only_c)[:2]))
    if not quiet:
        print(f"\n  {name}")
        print(f"    {'机制':<12} {'对上':>5} {'推导':>5} {'契约':>5}   "
              f"{'只在推导':<26} {'只在契约'}")
        for blk, hit, nd, nc, od, oc in rows:
            mark = "✓" if not od and not oc else "·"
            print(f"    {blk:<12} {hit:>5} {nd:>5} {nc:>5} {mark}  "
                  f"{str(od)[:26]:<26} {str(oc)[:26]}")
    bad = sum(1 for _b, _h, _d, _c, od, oc in rows if od or oc)
    return bad, rows


def audit_rows(rows, kv, hd, layers):
    """纯对账核心（自检用它做红注入）：行表 vs config 手算值。"""
    import re as _re
    problems, checked, skipped = [], 0, 0
    for tag, size_s, layer_s, total_s in rows:
        m = _re.search(r"([\d,]+)\s*B/token", str(size_s))
        ml = _re.search(r"(\d+)\s*层", str(layer_s))
        if not m or not ml:
            problems.append("%s：状态行解析不出来：%r %r" % (tag, size_s, layer_s))
            continue
        per_tok = int(m.group(1).replace(",", ""))
        n = int(ml.group(1))
        if n != layers:
            # 多栈 / 选择器子集（MTP、窗口层）—— 对照前提不成立，不硬判
            skipped += 1
            continue
        hit = None
        for dtype_b in (2, 4):
            if per_tok == kv * hd * 2 * dtype_b:
                hit = dtype_b
                break
        if hit is None:
            problems.append(
                "%s：声明 %d B/token，config 手算 kv·hd·2·dtype = %d(×2) / %d(×4)"
                % (tag, per_tok, kv * hd * 2 * 2, kv * hd * 2 * 4))
        else:
            checked += 1
    return problems, checked, skipped


def state_audit(path, bc, quiet=False):
    """**KV 状态尺寸的两份事实对账**（criteria-review 任务 B 的机器版）。

    refs config 的 (kv_heads, head_dim, layers) **手算** per-token KV 字节数
    （`kv·hd·2·dtype`），对照 `by1state` 声明的表。两边独立算 —— IR 侧
    kv/hd 与 config 漂移、或者状态尺寸公式写错，都是当场红。

    返回 (状态, 问题列表)：状态 'ok' / 'bad' / None（无状态声明，不适用）。
    行里的层数和 config 对不上 = 多栈/子集，**标"对照不适用"不硬判**。"""
    import re as _re
    import glob as _glob
    import json as _json
    _r, info = bc.check(path)
    rows = [r for r in (info.get("states") or []) if "kv_cache" in str(r[0])]
    if not rows:
        return None, []
    base = _os.path.basename(path)[:-4]
    cands = _glob.glob(by1paths.ref("*__%s.config.json" % base)) \
        or _glob.glob(by1paths.ref("%s.config.json" % base))
    if not cands:
        return None, []
    cfg = _json.loads(by1io.read_text(cands[0], encoding="utf-8"))
    heads = cfg.get("num_attention_heads")
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or (cfg.get("hidden_size") // heads)
    layers = cfg.get("num_hidden_layers")
    problems, checked, skipped = audit_rows(rows, kv, hd, layers)
    if not quiet:
        mark = "✓" if not problems else "✗"
        print("    %s %-40s 对照 %d 行 · 不适用 %d 行"
              % (mark, _os.path.basename(path), checked, skipped))
    return ("bad" if problems else "ok"), problems


def selftest():
    """反例必红：**把 shapes_of 那个 bug 注回去**，报表必须响。"""
    bc, cg = load("by1check"), load("by1codegen")
    import by1ir as bir
    path = by1paths.root("models", "gemma-4-31B.by1")
    _r, info = bc.check(path)
    ir = cg.compile_ir(info)
    params = bir.params_of(ir)

    # ① 修好的现状：全局推导里必须有 (vocab, d_model)
    want = (ir["vocab"], ir["d_model"])
    if want not in {tuple(v) for k, v in params.items()
                    if not k.startswith("layers.")}:
        print("  [FAIL] 全局推导里没有 %s —— 契约说它应当在" % (want,))
        return 1
    # ② 把 bug 注回去（尾层 in_dim 泄漏到全局）—— 必须被报表抓住
    leaked = ir["layers"][-1].get("in_dim") or ir["d_model"]
    bad = dict(params)
    bad["embed.weight"] = (ir["vocab"], leaked)
    bad["head.weight"] = (ir["vocab"], leaked)
    bad["final_norm.w"] = (leaked,)
    dg = derived(ir, bad)
    cg_ = contracted(info)
    hit = dg["@global"] & cg_["@global"]
    if (ir["vocab"], leaked) in hit:
        print("  [FAIL] 注入的 1152 bug 没被抓住 —— 报表瞎了")
        return 1
    if want not in cg_["@global"]:
        # 契约里没有全局行 = 这个自检没法判（说明面还没接上）
        print("  [跳过] 契约没有全局行可比 —— 这一条没验")
        return 30
    # ③ 状态尺寸对账：一致必须绿、kv 漂移必须红（audit_rows 双极性）
    row = [("state.kv_cache kv_cache", "2,048 B/token", "2 层",
            "4,096 B/token (全模型)")]
    ok_p, _c, _s = audit_rows(row, kv=8, hd=64, layers=2)      # 8*64*2*2 = 2048
    bad_p, _c, _s = audit_rows(row, kv=9, hd=64, layers=2)     # 漂移 -> 必须红
    if ok_p:
        print("  [FAIL] 状态对账把一致的手算报红了：%s" % ok_p[:1])
        return 1
    if not bad_p:
        print("  [FAIL] kv 漂移没被状态对账抓住 —— 对账瞎了")
        return 1
    print("  [PASS] 同步报表：现状全同源；注入 1152 bug 被抓住"
          "（推导 %s，契约 %s）；状态对账双极性过（一致绿 · kv 漂移红）"
          % ((ir["vocab"], leaked), want))
    return 0


def main(argv=None):
    argv = argv if argv is not None else _sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    bc, cg = load("by1check"), load("by1codegen")
    import by1ir as bir
    files = [a for a in argv if a.endswith(".by1")] or sorted(
        _os.path.join(by1paths.MODELS, f)
        for f in _os.listdir(by1paths.MODELS) if f.endswith(".by1"))
    tot_bad, tot_models = 0, 0
    for f in files:
        if f.endswith("selftest.by1"):
            continue
        tot_models += 1
        try:
            bad, _rows = sync_one(f, bc, cg, bir)
            tot_bad += 1 if bad else 0
        except Exception as e:
            print(f"  {_os.path.basename(f)}  [跳过] {type(e).__name__}: "
                  f"{str(e)[:60]}")
    print(f"\n  {tot_models} 份里 {tot_bad} 份还有未编码的布局差"
          f"（fuse / transpose / quant-block 逐个编码后这里归零，才配进门禁）")

    # ── KV 状态尺寸对账（criteria-review 任务 B 的机器版）──────────
    # 两边独立算：refs config 的 (kv_heads, head_dim, layers) 手算
    # per-token KV 字节数 vs by1state 声明的表。**漂移当场红** ——
    # 状态尺寸算错，MEM 预算和 cache 全跟着错，而报表可能还是绿的。
    print("\n  ── KV 状态尺寸：refs config 手算 vs by1state 声明")
    state_bad, state_ok = [], 0
    for f in files:
        if f.endswith("selftest.by1"):
            continue
        try:
            status, problems = state_audit(f, bc)
        except Exception as e:
            print("    %s [跳过] %s: %s" % (_os.path.basename(f),
                                           type(e).__name__, str(e)[:50]))
            continue
        if status == "ok":
            state_ok += 1
        elif status == "bad":
            state_bad.extend(problems)
            for p in problems:
                print("    !! " + p)
    print("    对账通过 %d 份（多栈/子集行按「对照不适用」记，不硬判）" % state_ok)
    if state_bad:
        print("  [FAIL] 状态尺寸对账 %d 处不一致" % len(state_bad))
        return 1
    return 0


if __name__ == "__main__":
    _sys.exit(main())
