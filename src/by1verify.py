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
import os
import re
import struct
import sys
import by1io
import by1skip

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
        # **"没验"不是"通过"。** 这里返回 None，主流程把它记成跳过（by1skip）。
        out.append("  %s by1 里没有 transformers.config 的 field 映射 —— "
                   "这一项没验" % by1skip.MARK)
        return None
    # 这些键描述的是「这份权重怎么被加载」，不是「这个模型的架构是什么」：
    #   architectures / auto_map / transformers_version   HF 的加载元信息
    #   torch_dtype / dtype                               权重存成什么精度
    #   quantization_config                               导出决策（在 quant 块里声明）
    # 不该由架构描述生成 —— 单列一类，**不算失败**，但要报出来而不是藏起来。
    META = {"architectures", "auto_map", "transformers_version",
            "torch_dtype", "dtype", "quantization_config"}
    keys = sorted((set(got) | set(tc)) - META)
    n_meta = len(set(tc) & META)
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
    out.append(f"  逐字段: 一致 {same}   值不同 {diff}   by1 多出 {extra}   "
               f"官方多出 {missing}   （元数据 {n_meta} 个不计）")
    if diff or missing or extra:
        out.append("  [FAIL] 前几处：")
        out.extend("      " + d for d in details[:14])
    else:
        out.append(f"  [PASS] {same} 个字段与官方 config 逐项一致"
                   + (f"（另有 {n_meta} 个 HF 元数据字段不属于架构描述）"
                      if n_meta else ""))
    return (diff + missing + extra) == 0


# ── 1. layer_types 对拍 ──────────────────────────────────────────────

def by1_layer_types(info):
    # **不要再算一遍。** 这里原来是一份 _ltype_of_attrs 的副本，
    # 同样只看 window —— 于是 GDN/KDA 这类线性注意力层被算成 full_attention。
    # 修了 by1check 里那个、忘了修这个，结果是同一个模型
    # "字段对拍通过、层类型检查失败"。
    # info["layer_types"] 是唯一的那一份实现。
    lt = info.get("layer_types")
    if lt:
        return list(lt)
    out = []
    for stack, mech, attrs in info["layers"]:
        w = (attrs.get("window") or "").strip().lower()
        out.append("full_attention" if w in ("none", "null", "0", "") else "sliding_attention")
    return out


def check_layer_types(info, tc, out):
    want = tc.get("layer_types")
    if not want:
        # **这是"不适用"，不是"没验"。**
        #
        # 官方 config 里没有逐层数组，就没有可比的东西 —— 而 `by1all` 的
        # `--tensors` 那一路**并没有要求**比层类型。第一版把它记成跳过，
        # 于是"形状全中"的一次调用被降级成跳过（13 个模型的张量行全变 `--`）。
        # **一个不在请求范围内的东西，不能给整个调用定结论。**
        out.append("  (config 无 layer_types，这一项不适用)")
        return None
    got = by1_layer_types(info)
    # 官方有些 config 的逐层数组**比模型长**：Step-3.7 是 48、模型 45 层
    # （12 × 4 vs 11 × 4 + 1）。多出来的不是垃圾，是**周期的延续**。
    # 但"允许更长"不能变成"随便放过" —— 这里验证多出来的确实是周期延续。
    if len(want) > len(got) and got:
        per = None
        for cand in range(1, len(got) // 2 + 1):
            if all(got[i] == got[i % cand] for i in range(len(got))):
                per = cand
                break
        tail_ok = bool(per) and all(want[i] == got[i % per]
                                    for i in range(len(got), len(want)))
        out.append("  层数: by1 = %d   config = %d（多出 %d 项，%s）"
                   % (len(got), len(want), len(want) - len(got),
                      "是周期的延续" if tail_ok else "!! 对不上周期"))
        if not tail_ok:
            out.append("  [FAIL] config 多出来的项不是 by1 周期的延续")
            return False
        want = want[:len(got)]
    else:
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
    r = _Rd(by1io.read_text(path, "rb"))
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
    raw = by1io.read_json(path, encoding="utf-8")
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
                  render_g=None, global_rows=None, quant=None):
    out.append(f"  命名规则来源: {rule_desc}")
    experts_of = experts_of or {}
    booll = lambda v: str(v).lower() in ("true", "1", "yes", "on")
    # 量化导出：被点名的专家张量在 HF 侧不存在，取而代之的是
    #   <名>_blocks / <名>_scales / <名>_bias
    q_members, q_fuse, q_block = set(), {}, 32
    if quant:
        q_block = int(quant.get("block", 32))
        q_members = set(re.findall(r"[A-Za-z_][\w.]*", quant.get("tensor", "")))
        for _nm, _v in (quant.get("fuse") or {}).items():
            _ms = re.findall(r"[A-Za-z_][\w.]*", _v)
            for _m in _ms:
                q_fuse[_m] = (_nm, _ms)

    def q_shapes(fused, block, kind):
        """weight 是 MXFP4 打包（blocks + scales）；bias 不量化，只是被融合。"""
        if kind == "bias":
            return [("_bias", list(fused))]
        n = max(1, fused[-1] // block)
        return [("_blocks", list(fused[:-1]) + [n, 16]),
                ("_scales", list(fused[:-1]) + [n])]
    total = mismatch = missing = absent_but_present = symbolic = 0
    sup_total = sup_ok = unmapped = 0
    details, samples, generated = [], [], set()
    qdone = set()

    # **传"栈内序号"，不是全局层号。**
    # 一个模型可以有多个栈（Qwen3.5 有主干 64 层 + MTP 1 层），
    # 而物理名字里的层号是**各栈从 0 起**：主干 model.layers.0..63、
    # MTP mtp.layers.0。传全局层号会给 MTP 拼出 `mtp.layers.64.` ——
    # 名字全都对不上，而形状是对的，所以只看"形状不符 0"会以为没事。
    # 层号用**栈内序号**还是**全局序号**，取决于那一栈的命名：
    #   Qwen3.5  主干 model.language_model.layers.0..63、MTP mtp.layers.0
    #            —— 各自从 0 起（栈内序号）
    #   GLM-5.3  **都在 model.language_model.layers.{i}.** 里，
    #            MTP 是第 45 层 —— 层号接着数（全局序号）
    # 两种都有，所以在 .by1 里显式声明 `index = global`，不猜。
    _glob = set()
    for _st in (info.get("stacks") or []):
        _as = getattr(_st, "assigns", None) or {}
        if (_as.get("index") or "").strip().lower() == "global":
            _glob.add(getattr(_st, "name", ""))
    _loc = {}
    for i, (stack, mech, attrs, rows) in enumerate(info["layer_out"]):
        if stack in _glob:
            _li = i                      # 接着主栈数
        else:
            _li = _loc.get(stack, 0)
            _loc[stack] = _li + 1
        qdone = set()
        for owner, (lname, shape_txt, note, pe) in rows:
            if scope_map and owner not in scope_map:
                unmapped += 1
                continue
            use_e = booll(pe) and render_e is not None
            _lb = lname.rsplit(".", 1)[0] if "." in lname else lname
            _kd = lname.rsplit(".", 1)[1] if "." in lname else ""
            if quant and _lb in q_members:
                _fn, _ms = q_fuse.get(_lb, (_lb, [_lb]))
                if (_fn, _kd) in qdone:
                    continue
                qdone.add((_fn, _kd))
                _by = {r[1][0]: r[1][1] for r in rows}   # rows 是 (owner, (名, 形, 注, pe))
                _num = [parse_shape(_by.get(m + "." + _kd, "")) for m in _ms]
                if any(not v for v in _num):
                    symbolic += 1
                    continue
                _f = list(_num[0])
                if len(_num) > 1:
                    _f[1] = sum(v[1] for v in _num)
                _base = render(_li, stack, owner, _fn)
                for _sfx, _bs in q_shapes(_f, q_block, _kd):
                    _nm = _base + _sfx
                    generated.add(_nm)
                    if len(samples) < 6:
                        samples.append(f"      L{i:<3} 量化 {_fn:<10} -> {_nm}")
                    total += 1
                    if _nm not in real:
                        missing += 1
                        details.append(f"L{i:<3} {_nm}\n        契约 {tuple(_bs)}，实际不存在")
                    elif real[_nm]["shape"] != _bs:
                        mismatch += 1
                        details.append(f"L{i:<3} {_nm}\n        契约 {tuple(_bs)}"
                                       f"   实际 {real[_nm]['shape']}")
                continue
            exps = range(experts_of.get(owner, 1)) if use_e else [None]
            for e in exps:
                nm = (render_e(_li, stack, owner, lname, e) if use_e
                      else render(_li, stack, owner, lname))
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
        # `--` = **声明为不该存在**（权重共享时 checkpoint 里没有 lm_head）。
        # 全局张量走的是这个循环（不是 check_tensors 里那个），
        # 两边都得认这个标记，否则它会被当成"符号形状"静默跳过。
        if shape_txt == "--":
            sup_total += 1
            if nm in real:
                absent_but_present += 1
                details.append(
                    f"global {nm}\n        契约说应被抑制，实际存在 {real[nm]['shape']}")
            else:
                sup_ok += 1
            continue
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

    # **反方向也要报。** 上面全是"契约 → 实物"，于是"官方多出 0"读起来像
    # "什么都没漏"，而实际上实物里可能有几百个契约根本没提的张量。
    # 多模态模型就是这种情况：Gemma 的 vision/audio、Qwen3.5 的 model.visual.*
    # 都在权重文件里，而 by1 不建模它们。
    # 不报的话，一个只看文本主干的描述会显得像是覆盖了整个 checkpoint。
    covered = generated
    uncovered = sorted(k for k in real if k not in covered)
    if uncovered:
        groups = {}
        for k in uncovered:
            key = re.sub(r"\.\d+\.", ".N.", k)
            groups[key] = groups.get(key, 0) + 1
        out.append(f"  契约**未覆盖**的实物张量: {len(uncovered)} 个"
                   f"（共 {len(real)} 个里的 "
                   f"{100.0 * len(uncovered) / max(len(real), 1):.0f}%）")
        for k in sorted(groups, key=lambda x: -groups[x])[:6]:
            out.append(f"        {groups[k]:5d}  {k}")
        if len(groups) > 6:
            out.append(f"        …另有 {len(groups) - 6} 类")
        out.append("      （这是**范围边界**，不是失败 —— 但必须看得见）")

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

def main(argv=None):
    # 同上：入口 `by1-verify = "by1verify:main"` 是不带参数调的。
    if argv is None:
        argv = sys.argv[1:]
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

    cfg = by1io.read_json(cfg_path, encoding="utf-8")
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
    # `--sub` 默认是 text_config —— 多模态模型（Gemma）的架构嵌在里面。
    # **但顶层那些键就完全不进比较了，而报告会写「官方多出 0」**，
    # 读起来像"什么都没缺"。把丢掉的说出来。
    if tc is not cfg:
        _dropped = sorted(set(cfg) - set(tc))
        out.append("  （只比 %s 那一层；顶层另有 %d 个键未参与比较：%s）"
                   % (sub, len(_dropped),
                      ', '.join(_dropped[:5]) + (' …' if len(_dropped) > 5 else '')))
        out.append("")
    # **跳过要单独记。** 一个调用里"验了一部分、跳了一部分"，
    # 结论必须是跳过 —— 否则"部分验过"会被读成"验过了"。
    skipped = []
    ok0 = None
    if "--config" in argv:
        out.append("[1] config 逐字段对拍   (由 .by1 生成，不继承任何字段)")
        ok0 = check_config(info, tc, out)
        if ok0 is None:
            skipped.append('config 映射')
        out.append("")
    out.append("[2] layer_types")
    ok1 = check_layer_types(info, tc, out)

    ok2 = None
    real = load_reference(tensors_path, gguf_path)
    if real is not None:
        out.append("")
        out.append(f"[3] tensor names + shapes   (backend = {backend})")
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
            # **一个拼错的 `--backend` 原来就能静默删掉整项形状校验还报绿。**
            # 现在它是"跳过"：输出留痕、退出码 2 —— 见 by1skip。
            out.append(f"  {by1skip.MARK} by1 里没有 emit[{backend}] 规则"
                       f" —— 这一项没验")
            skipped.append(f'emit[{backend}]')
        if render:
            _r = rules.get(backend) or {}
            render_e = None
            if _r.get("expert_name"):
                # tpl 必须绑成默认参数 —— 下面 global_name 那段会复用同一个变量名，
                # 而闭包是调用时才读它的，会把专家名悄悄退化成裸的逻辑名。
                render_e = (lambda i, s, m, l, e, _t=_r["expert_name"]:
                            bc.render_name(dict(_r, name=_t), i, s, m, l, e))
            render_g = None
            if _r.get("global_name"):
                render_g = (lambda l, _t=_r["global_name"]:
                            bc.render_name(dict(_r, name=_t), 0, "<global>",
                                           "global", l))
            experts_of = {}
            for mn, a in (info.get("mech_attrs") or {}).items():
                try:
                    experts_of[mn] = int(float(a.get("experts", 0) or 0))
                except (TypeError, ValueError):
                    experts_of[mn] = 0
            ok2 = check_tensors(info, real, render, rule_desc, scope_map, out,
                                render_e=render_e, experts_of=experts_of,
                                render_g=render_g, global_rows=info.get("global_rows"),
                                quant=_r.get("quant") or None)

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
    # **失败压过跳过。** 验出了错就是错；只有"什么都没验出错、但有没验的"
    # 才算跳过。反过来（跳过压失败）会把真失败藏起来。
    if ok0 is False or ok1 is False or ok2 is False:
        return 1
    if skipped:
        return by1skip.CODE
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
