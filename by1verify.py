#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 verify -- 差分验证：把 .by1 的展开结果与官方产物对拍。

这是 1.md 能力清单第 6 条「差分验证」的最小可用实现：
不比浮点，比**结构**。

三项对拍：
  1. layer_types    .by1 的 pattern 展开  vs  config.json 的 layer_types 数组
  2. tensor shapes  .by1 的逻辑张量契约  vs  safetensors 的 dtype/shape
  3. ggml naming    同一个契约 + 另一套 lowering  vs  真实 GGUF 的张量名与 ne

第 3 项是关键：物理名由 .by1 的 emit 规则生成，本脚本不含任何硬编码名字。

用法:
  python by1verify.py <file.by1> <config.json> [--tensors a.json | --gguf b.bin]
                      [--backend torch.module|ggml] [--sub text_config]

约定:
  .by1 层实例带 window = <n>   -> config 的 sliding_attention
  .by1 层实例带 window = none  -> config 的 full_attention
  GGUF 的 ne 是反序的，读取时统一反转成 PyTorch 顺序再比较
"""

import importlib.util
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "by1check", os.path.join(HERE, "by1check.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_config(info, tc, out):
    """反向：config 由 .by1 生成，官方 config 只是判卷人。"""
    got = info.get("config") or {}
    if not got:
        out.append("  (by1 里没有 transformers.config 的 field 映射，跳过)")
        return None
    keys = sorted(set(got) | set(tc))
    same = diff = extra = missing = 0
    details = []
    for k in keys:
        if k in got and k in tc:
            if got[k] == tc[k]:
                same += 1
            else:
                diff += 1
                details.append(f"{k}\n        by1 生成 {got[k]!r}\n        官方     {tc[k]!r}")
        elif k in got:
            extra += 1
            details.append(f"{k}\n        by1 生成 {got[k]!r}\n        官方     不存在")
        else:
            missing += 1
            details.append(f"{k}\n        by1 生成 不存在\n        官方     {tc[k]!r}")
    out.append(f"  逐字段: 一致 {same}   值不同 {diff}   by1 多出 {extra}   官方多出 {missing}")
    if diff or missing or extra:
        out.append("  [FAIL] 前几处：")
        out.extend("      " + d for d in details[:14])
    else:
        out.append(f"  [PASS] {same} 个字段与官方 config 逐项一致")
    return (diff + missing + extra) == 0


# ── 1. layer_types 对拍 ──────────────────────────────────────────────

def by1_layer_types(info):
    out = []
    for stack, mech, attrs in info["layers"]:
        w = (attrs.get("window") or "").strip().lower()
        out.append("full_attention" if w in ("none", "null", "0", "") else "sliding_attention")
    return out


def check_layer_types(info, tc, out):
    want = tc.get("layer_types")
    if not want:
        out.append("  (config 无 layer_types，跳过)")
        return None
    got = by1_layer_types(info)
    out.append(f"  层数: by1 = {len(got)}   config = {len(want)}")
    n = min(len(got), len(want))
    bad = [i for i in range(n) if got[i] != want[i]]
    if len(got) != len(want):
        out.append("  [FAIL] 层数不一致")
    if bad:
        out.append(f"  [FAIL] {len(bad)} 层类型不一致，前几处：")
        for i in bad[:8]:
            out.append(f"        L{i:<3} by1={got[i]:<18} config={want[i]}")
    if len(got) == len(want) and not bad:
        out.append(f"  [PASS] {len(want)} 层类型逐项一致")
    return (len(got) == len(want) and not bad)


# ── 2. 参考产物读取 ──────────────────────────────────────────────────

_GGUF_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _Rd:
    def __init__(self, b):
        self.b, self.p = b, 0

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.p)[0]
        self.p += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.p)[0]
        self.p += 8
        return v

    def s(self):
        n = self.u64()
        v = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return v

    def skip(self, t):
        if t in _GGUF_SZ:
            self.p += _GGUF_SZ[t]
        elif t == 8:
            self.s()
        elif t == 9:
            et = self.u32()
            n = self.u64()
            for _ in range(n):
                self.skip(et)
        else:
            raise ValueError(f"GGUF 未知类型 {t}")


def read_gguf_header(path):
    """返回 {张量名: {'shape': [PyTorch 顺序]}}。ne 是反序的，这里统一反转。"""
    r = _Rd(open(path, "rb").read())
    if r.b[:4] != b"GGUF":
        raise ValueError("不是 GGUF 文件")
    r.p = 4
    r.u32()
    nt, nkv = r.u64(), r.u64()
    for _ in range(nkv):
        r.s()
        t = r.u32()
        if t == 8:
            r.s()
        else:
            r.skip(t)
    out = {}
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty = r.u32()
        r.u64()
        out[name] = {"shape": list(reversed(dims)), "ggml_type": ty}
    return out


def load_reference(tensors_path, gguf_path):
    """参考产物：safetensors 的 JSON 表，或 GGUF（原始 .gguf 头，或已抽出的 JSON）。"""
    path = tensors_path or gguf_path
    if not path:
        return None
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        return read_gguf_header(path)
    raw = json.load(open(path, encoding="utf-8"))
    return {k: {"shape": [int(x) for x in v["shape"]]}
            for k, v in raw.items() if isinstance(v, dict) and "shape" in v}


# ── 3. 张量对拍 ──────────────────────────────────────────────────────

def parse_shape(txt):
    body = txt.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    out = []
    for p in body.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            return None
    return out


def check_tensors(info, real, render, rule_desc, scope_map, out,
                  render_e=None, experts_of=None,
                  render_g=None, global_rows=None):
    out.append(f"  命名规则来源: {rule_desc}")
    experts_of = experts_of or {}
    booll = lambda v: str(v).lower() in ("true", "1", "yes", "on")
    total = mismatch = missing = absent_but_present = symbolic = 0
    sup_total = sup_ok = unmapped = 0
    details, samples, generated = [], [], set()

    for i, (stack, mech, attrs, rows) in enumerate(info["layer_out"]):
        for owner, (lname, shape_txt, note, pe) in rows:
            if scope_map and owner not in scope_map:
                unmapped += 1
                continue
            use_e = booll(pe) and render_e is not None
            exps = range(experts_of.get(owner, 1)) if use_e else [None]
            for e in exps:
                nm = (render_e(i, stack, owner, lname, e) if use_e
                      else render(i, stack, owner, lname))
                generated.add(nm)
                if len(samples) < 5 and shape_txt != "--":
                    samples.append(f"      L{i:<3} {owner}.{lname:<10} -> {nm}")
                if shape_txt == "--":
                    sup_total += 1
                    if nm in real:
                        absent_but_present += 1
                        details.append(
                            f"L{i:<3} {nm}\n        契约说应被抑制，实际存在 {real[nm]['shape']}")
                    else:
                        sup_ok += 1
                    continue
                exp = parse_shape(shape_txt)
                if exp is None:
                    symbolic += 1
                    continue
                if use_e and len(exp) > 1:
                    exp = exp[1:]          # 逐专家存储：专家维落在名字里，不在形状里
                total += 1
                if nm not in real:
                    missing += 1
                    details.append(f"L{i:<3} {nm}\n        契约声明 {exp}，实际不存在")
                elif real[nm]["shape"] != exp:
                    mismatch += 1
                    details.append(
                        f"L{i:<3} {nm}\n        契约 {exp}   实际 {real[nm]['shape']}")

    # 全局张量：不属于任何层，名字里通常没有层号
    for (lname, shape_txt, note, _pe) in (global_rows or []):
        if render_g is None:
            break
        nm = render_g(lname)
        generated.add(nm)
        if len(samples) < 6 and shape_txt != "--":
            samples.append(f"      --   global.{lname:<12} -> {nm}")
        exp = parse_shape(shape_txt)
        if exp is None:
            symbolic += 1
            continue
        total += 1
        if nm not in real:
            missing += 1
            details.append(f"global {nm}\n        契约声明 {exp}，实际不存在")
        elif real[nm]["shape"] != exp:
            mismatch += 1
            details.append(f"global {nm}\n        契约 {exp}   实际 {real[nm]['shape']}")

    out.append("  生成的名字示例:")
    out.extend(samples)
    ok = total - mismatch - missing
    out.append(f"  契约声明存在: {total}   名字+形状一致 {ok}   "
               f"形状不符 {mismatch}   缺失 {missing}")
    out.append(f"  契约声明抑制: {sup_total}   确实不存在 {sup_ok}   "
               f"应无却有 {absent_but_present}   符号形状跳过 {symbolic}"
               + (f"   未映射机制跳过 {unmapped}" if unmapped else ""))
    if mismatch or missing or absent_but_present:
        out.append("  [FAIL] 前几处不符：")
        out.extend("      " + d for d in details[:10])
    else:
        out.append(f"  [PASS] {ok} 个张量的名字与形状全部一致，"
                   f"{sup_ok} 处抑制也正确")

    # 反向覆盖：参考产物里有哪些张量是契约还没表达的
    residue = {}
    for k in real:
        if k in generated:
            continue
        residue[re.sub(r"\.\d+\.", ".N.", k)] = residue.get(
            re.sub(r"\.\d+\.", ".N.", k), 0) + 1
    if residue:
        out.append("")
        out.append(f"  [覆盖缺口] 参考产物里有 {len(residue)} 类张量没有契约对应物:")
        for k, c in sorted(residue.items(), key=lambda x: -x[1])[:20]:
            out.append(f"      {c:>4}  {k}")
        if len(residue) > 20:
            out.append(f"      ... 另有 {len(residue) - 20} 类")
    return (mismatch + missing + absent_but_present) == 0


# ── main ─────────────────────────────────────────────────────────────

def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    by1_path, cfg_path = argv[0], argv[1]

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    sub = opt("--sub", "text_config")
    name_tpl = opt("--name")
    tensors_path = opt("--tensors")
    gguf_path = opt("--gguf")
    backend = opt("--backend", "ggml" if gguf_path else "torch.module")

    cfg = json.load(open(cfg_path, encoding="utf-8"))
    tc = cfg.get(sub, cfg)

    bc = load_checker()
    _, info = bc.check(by1_path)

    out = []
    out.append("=" * 74)
    tag = os.path.basename(gguf_path or tensors_path or "")
    out.append(f"  {os.path.basename(by1_path)}  vs  {os.path.basename(cfg_path)}"
               + (f"  +  {tag}" if tag else ""))
    out.append("=" * 74)

    out.append("")
    ok0 = None
    if "--config" in argv:
        out.append("[1] config 逐字段对拍   (由 .by1 生成，不继承任何字段)")
        ok0 = check_config(info, tc, out)
        out.append("")
    out.append("[2] layer_types")
    ok1 = check_layer_types(info, tc, out)

    ok2 = None
    real = load_reference(tensors_path, gguf_path)
    if real is not None:
        out.append("")
        out.append(f"[2] tensor names + shapes   (backend = {backend})")
        rules = info.get("emit", {})
        render, rule_desc, scope_map = None, "", {}
        if name_tpl:
            render = lambda i, s, m, l: name_tpl.format(i=i, stack=s, mech=m, logical=l)
            rule_desc = f"命令行模板 {name_tpl}"
        elif backend in rules:
            rule = rules[backend]
            render = lambda i, s, m, l: bc.render_name(rule, i, s, m, l)
            rule_desc = f'by1 emit[{backend}]  name = "{rule["name"]}"'
            scope_map = rule.get("scope", {})
        else:
            out.append(f"  (by1 里没有 emit[{backend}] 规则，跳过)")
        if render:
            _r = rules.get(backend) or {}
            render_e = None
            if _r.get("expert_name"):
                tpl = _r["expert_name"]
                render_e = lambda i, s, m, l, e: bc.render_name(
                    dict(_r, name=tpl), i, s, m, l, e)
            render_g = None
            if _r.get("global_name"):
                tpl = _r["global_name"]
                render_g = lambda l: bc.render_name(
                    dict(_r, name=tpl), 0, "<global>", "global", l)
            experts_of = {}
            for mn, a in (info.get("mech_attrs") or {}).items():
                try:
                    experts_of[mn] = int(float(a.get("experts", 0) or 0))
                except (TypeError, ValueError):
                    experts_of[mn] = 0
            ok2 = check_tensors(info, real, render, rule_desc, scope_map, out,
                                render_e=render_e, experts_of=experts_of,
                                render_g=render_g, global_rows=info.get("global_rows"))

    out.append("")
    out.append("  等价类")
    seen = {}
    for _, mech, attrs in info["layers"]:
        k = (mech, attrs.get("q"), attrs.get("kv"), attrs.get("head_dim"),
             attrs.get("window"), attrs.get("kv_tie"))
        seen[k] = seen.get(k, 0) + 1
    for k, c in sorted(seen.items(), key=lambda x: -x[1]):
        mech, q, kv, hd, w, tie = k
        out.append(f"    {mech:<6} q={q or '?':<4} kv={kv or '?':<4} "
                   f"head_dim={hd or '?':<5} window={w or '?':<6} "
                   f"kv_tie={tie or '?':<6} {c:>3} 层")

    print("\n".join(out))
    return 0 if (ok0 is not False and ok1 is not False and ok2 is not False) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
