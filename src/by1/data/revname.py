#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1revname -- **命名模板反推器（C1/C2）：refs 产物清单 → emit 命名模板草案。**

## 它解决什么

`.by1` 里的 `emit` 命名段（`name` / `scope` / `global_name` / `name_<栈>` /
`expert_name`）原来是**手写**的。手写的问题不是累，是**决策没有出处** ——
"为什么 GQA 的前缀是 `self_attn.`" 答案在人脑里，不在数据里。

C1 的主张：这些模板**可以从产物清单机械地推出来**。推得出来 ⇒ 它是
数据的必然；推不出来 ⇒ 那一处是**手写决策**，单独记账（C2 的报表），
不许混在"机器生成"里说。

## 怎么推

产物里一个层张量长这样：

    model.language_model.layers.12.self_attn.q_proj.weight
    └──────── const ────────┘└i┘└─ scope ─┘└── physical ──┘

给定契约行 (owner, 逻辑名, 层号) 和 refs 名字集合：

1. **配对**：refs 里以逻辑名收尾、层号作为独立段出现、形状一致的那一个。
   量化后缀 `_blocks/_scales/_bias` 先归一；逐专家行取全部候选。
2. **切模板**：摘掉逻辑名得 head；层号段之前是 const，之后是 `{scope}`
   的取值（按 owner 分别记）。逐专家行把专家号段换成 `{expert}`。
3. **回放验证**：推出来的模板把**全部**契约行渲染一遍，逐个对 refs ——
   这就是 C1 的判据：**反推结果必须和手写模板一样全绿**（不许退）。

## 判据（C1 / C2）

- C1：回放的 (名字, 形状) 对拍全绿（gemma 530/530 · gpt-oss 687/687
  一类不许退）；推不出的行显式记"手写残余"，不静默丢。
- C2：逐模型逐行比较 反推渲染 vs 手写渲染 —— 相同 = 机器可生成，
  不同 = 手写决策残余（带样例）。**机器生成率只升不降。**

用法:
    python src/data/by1revname.py                 # 全部有 refs 的模型
    python src/data/by1revname.py --only gemma    # 名字里含 gemma 的
    python src/data/by1revname.py --report        # 只出 C2 报表
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

import argparse
import glob
import os

import by1check
import by1verify

#: 量化导出会把逻辑名接上这些后缀（MXFP4 的 blocks/scales/bias）；
#: 配对时先归一，否则 `gate_exps` 永远配不上 `gate_exps_blocks`。
Q_SUFFIXES = ("_blocks", "_scales", "_bias")


def items_of(info):
    """把契约行摊成 (类别, 层号, 栈, owner, 逻辑名, 形状, pe) ——
    **枚举逐字镜像 `by1verify.check_tensors`**（栈内/全局层号、全局行、
    抑制行），否则反推的覆盖面和判卷人对不上。"""
    booll = lambda v: str(v).lower() in ("true", "1", "yes", "on")
    glob_stacks = set(k for k, v in (info.get("stack_index") or {}).items()
                      if v == "global")
    out, local = [], {}
    for i, (stack, mech, attrs, rows) in enumerate(info["layer_out"]):
        li = i if stack in glob_stacks else local.get(stack, 0)
        if stack not in glob_stacks:
            local[stack] = li + 1
        for owner, (lname, shape_txt, note, pe) in rows:
            out.append(("layer", li, stack, owner, lname, shape_txt,
                        booll(pe)))
    for (lname, shape_txt, note, _pe) in (info.get("global_rows") or []):
        out.append(("global", 0, "<global>", "global", lname, shape_txt, False))
    return out


def _fuzzy_tail(nm, lname):
    """逻辑名对物理尾巴**逐段对齐**（`embed.weight` 对 `embed_tokens.weight`），
    并允许尾巴后多出段（`q_proj` 对 `q_proj.weight`）—— 对上了就返回
    物理尾巴，它就是 rename 的取值。段从右往左找对齐位；每段全等、
    或**任一方向**的前缀关系（`final_norm` 对 `norm` 这种截短也在内）。"""
    ns, ls = nm.split("."), lname.split(".")
    n, m = len(ns), len(ls)
    for p in range(n - m, -1, -1):
        window = ns[p:p + m]
        if all(a == b or a.startswith(b) or b.startswith(a)
               for a, b in zip(window, ls)):
            return ".".join(ns[p:])
    return None


def pair_scan(kind, owner, lname, shape_txt, real, declared=None):
    """按尾巴+形状扫一遍 refs —— **每个 (kind, owner, lname, 形状) 只扫一次**
    （层行按层号复用；76k 张量 × 上千行的全量扫描是 O(n²)，跑不完）。

    数据推不动的 rename（任意改名，`exp_bias` -> `gate.e_score_correction_bias`
    这类）**从声明的 rename 表取**，但记进 `manual` —— 它就是 C2 要量化的
    "手写决策残余"，不是反推的功劳。"""
    exp = by1verify.parse_shape(shape_txt)
    want_tails = [lname + s for s in Q_SUFFIXES] + [lname]
    manual = {}
    exact, fuzzy = [], []
    for nm, meta in real.items():
        tail = next((t for t in want_tails if nm.endswith(t)), None)
        if tail is None:
            tail = _fuzzy_tail(nm, lname)
            if tail is not None:
                fuzzy.append((nm, tail, meta))
                continue
        if tail is None and declared:
            dt = (declared.get("rename") or {}).get(lname)
            if dt and nm.endswith(dt):
                tail = dt
                manual[lname] = dt
        if tail is None:
            continue
        exact.append((nm, tail, meta))
    # **精确尾巴优先，模糊只兜底** —— 否则 `pre_feedforward_layernorm`
    # 会把孪生的 `_2` 一起抓进来，"尾巴不统一"把本来精确可判的行毁掉。
    out = exact or fuzzy
    out = [(nm, tail) for nm, tail, _meta in out
           if (lambda got, e: e is None or got == tuple(e)
               or (kind == "layer" and len(e) > 1 and got == tuple(e[1:])))(
               tuple(_meta.get("shape") or ()), exp)]
    if not out:
        return [], None, "配不上", manual
    tails = {t for _n, t in out}
    if len(tails) != 1:
        return [c[0] for c in out], None, \
            "尾巴不统一：%s" % sorted(tails)[:3], manual
    return out, tails.pop(), None, manual


def pair(kind, li, owner, lname, shape_txt, real, cache, declared=None):
    """契约行 -> refs 里的物理名候选（按层号过滤 scan 的结果）。

    物理尾巴 != 逻辑名 的地方就是 `rename` 的取值 —— **反推顺手把
    rename 也推出来**，不然 globals（`embed.weight` -> `model.embed_tokens.weight`）
    这一族永远配不上。数据推不动的 rename 从声明取，进 manual 记账。"""
    key = (kind, owner, lname, shape_txt)
    if key not in cache:
        cache[key] = pair_scan(kind, owner, lname, shape_txt, real, declared)
    out, tail, why, manual = cache[key]
    if why:
        return [], None, why, {}
    if kind == "layer":
        out = [(nm, t) for nm, t in out if str(li) in nm.split(".")]
        if not out:
            return [], None, "层号 %d 配不上" % li, {}
    return [nm for nm, _t in out], tail, None, manual


def derive_layer_template(samples):
    """若干 (层号, owner, 逻辑名, 物理名, 物理尾巴) -> (模板, {owner: scope}, 错误)。

    `name = const{后跟 "{i}."}{scope}{physical}`；层号段定位：head 的
    `.` 段里 == 层号 的最左一个。`{scope}` 取层号段之后的全部（含尾点）。"""
    heads = []
    for li, owner, lname, nm, tail in samples:
        head = nm[:-len(tail)] if tail else nm
        segs = head.split(".")
        try:
            k = segs.index(str(li))
        except ValueError:
            return None, None, "层号段 %d 不在 %r 里" % (li, head)
        const = ".".join(segs[:k]) + ("." if k else "")
        scope_val = ".".join(segs[k + 1:])
        heads.append((const, scope_val, owner))
    consts = {h[0] for h in heads}
    if len(consts) != 1:
        return None, None, "常量段不统一：%s" % sorted(consts)[:3]
    const = consts.pop()
    scopes = {}
    for _c, scope_val, owner in heads:
        if owner in scopes and scopes[owner] != scope_val:
            return None, None, "owner %s 的 scope 值不稳定" % owner
        scopes[owner] = scope_val
    return const + "{i}.{scope}{physical}", scopes, None


def derive_expert_template(samples):
    """逐专家行：head 里层号段 -> `{i}`，其后的整数段（专家号）-> `{expert}`。

    **没有专家号段的行不进这里**（shared_expert 那种"一份共享张量"不是
    逐专家存储，它走普通模板 + rename）——混进来会把整条模板毁掉。"""
    heads = []
    for li, owner, lname, nm, tail in samples:
        head = nm[:-len(tail)] if tail else nm
        segs = head.split(".")
        try:
            k = segs.index(str(li))
        except ValueError:
            return None, "层号段 %d 不在 %r 里" % (li, head)
        ints = [j for j in range(k + 1, len(segs))
                if segs[j].isdigit() and j != k]
        if not ints:
            return None, "head 里找不到专家号段：%r" % head
        segs[k] = "{i}"
        segs[ints[-1]] = "{expert}"
        heads.append(".".join(segs))
    if len(set(heads)) != 1:
        return None, "专家模板不统一：%s" % sorted(set(heads))[:2]
    return heads[0] + "{physical}", None


def _expert_idx_in(nm, tail, li):
    """物理名的 head 里、层号之后有没有专家号段。"""
    head = nm[:-len(tail)] if tail else nm
    segs = head.split(".")
    try:
        k = segs.index(str(li))
    except ValueError:
        return False
    return any(segs[j].isdigit() for j in range(k + 1, len(segs)))


def derive_global_template(samples):
    """若干 (逻辑名, 物理名, 物理尾巴) -> global_name 模板。约定优先 `{physical}`。"""
    wraps = set()
    for lname, nm, tail in samples:
        wraps.add(nm[:-len(tail)] if tail else nm)
    if wraps == {""}:
        return "{physical}", None
    if len(wraps) == 1:
        return wraps.pop() + "{physical}", None
    return None, "全局包装不统一：%s" % sorted(wraps)[:3]


def reverse_one(by1_path, backend, real):
    """一个模型一个后端：反推 + 回放验证 + 与手写对照（C1 + C2）。"""
    report, info = by1check.check(by1_path)
    if report.count(by1check.E):
        return {"model": os.path.basename(by1_path), "error": "by1check 有错"}
    declared = (info.get("emit") or {}).get(backend) or {}
    experts_of = {}
    for mn, a in (info.get("mech_attrs") or {}).items():
        try:
            experts_of[mn] = int(float(a.get("experts", 0) or 0))
        except (TypeError, ValueError):
            experts_of[mn] = 0
    items = items_of(info)

    plain, expert_rows, global_samples, unpaired = [], [], [], []
    rename, paired, cache, manual_renames = {}, {}, {}, {}
    expert_keys, multi = set(), []
    for kind, li, stack, owner, lname, shape_txt, pe in items:
        if shape_txt == "--":
            continue                      # 抑制行：不进名字反推，回放时验缺席
        cands, tail, why, manual = pair(kind, li, owner, lname, shape_txt,
                                        real, cache, declared)
        if why:
            unpaired.append("%s %s.%s (%s)" % (kind, owner, lname, why))
            continue
        manual_renames.update(manual)
        paired[(kind, li, stack, owner, lname)] = cands
        if tail and tail not in [lname] + [lname + s for s in Q_SUFFIXES]:
            # 尾巴 != 逻辑名 的地方就是 rename 的取值（顺手推出来）
            rename[lname] = tail
        if kind == "global":
            global_samples.append((lname, cands[0], tail))
        elif pe and any(_expert_idx_in(nm, tail, li) for nm in cands):
            # 逐专家存储的行只认**带专家号**的候选 —— 同层的
            # shared_expert 名混进来会让 want 比渲染多一项，
            # 那个名字是 shexp 那一行的，不是这一行的。
            cands_e = [c for c in cands if _expert_idx_in(c, tail, li)]
            expert_keys.add((kind, li, stack, owner, lname))
            expert_rows.append((li, owner, lname, cands_e, stack, tail))
            paired[(kind, li, stack, owner, lname)] = cands_e
        elif len(cands) > 1:
            multi.append((kind, li, stack, owner, lname, cands, tail))
        else:
            plain.append((li, owner, lname, cands[0], stack, tail))

    derived, problems = {}, []
    by_stack = {}
    for li, owner, lname, nm, stack, tail in plain:
        by_stack.setdefault(stack, []).append((li, owner, lname, nm, tail))
    scopes = {}
    for stack, samples in by_stack.items():
        tpl, sc, err = derive_layer_template(samples)
        if err:
            problems.append("栈 %s: %s" % (stack, err))
            continue
        # 单栈时 `name` 就够；多栈写 `name_<栈>`（渲染器的优先级如此）
        key = "name" if len(by_stack) == 1 else "name_" + stack
        derived[key] = tpl
        for k, v in sc.items():
            if k in scopes and scopes[k] != v:
                problems.append("scope[%s] 跨栈不稳：%r vs %r" % (k, scopes[k], v))
            scopes[k] = v
    derived["scope"] = scopes
    if rename:
        derived["rename"] = rename
    if global_samples:
        gt, err = derive_global_template(global_samples)
        if err:
            # 包装不统一 -> 退化成「全量 rename + `{physical}`」约定
            # （真模型的 declared 就是这么写的）。这条路**数据可推**，
            # 不是残余：每个全局名 -> refs 里的那个唯一形状名。
            derived["global_name"] = "{physical}"
            for lname2, nm2, _tail2 in global_samples:
                rename[lname2] = nm2
        else:
            derived["global_name"] = gt
    if expert_rows:
        samples = [(li, owner, lname, nm, tail)
                   for li, owner, lname, cands, stack, tail in expert_rows
                   for nm in cands if _expert_idx_in(nm, tail, li)]
        et, err = derive_expert_template(samples)
        if err:
            problems.append("expert: %s" % err)
        else:
            derived["expert_name"] = et

    def _suffix_of(nm, lname):
        for s in Q_SUFFIXES:
            if nm.endswith(lname + s):
                return s
        return ""

    # ── 第二遍消歧：多解行按**已推出的规则整名渲染**认领 ────────────
    # 双塔同名（E4B 的 audio/vision）和同栈同名（gpt2 的 attn/mlp
    # `c_proj.bias`）都靠这一下分开 —— 行属于谁，由推好的
    # 模板 + scope 说了算；规则是从唯一配对的行推的，不是猜的。
    for kind, li, stack, owner, lname, cands, tail in multi:
        try:
            nm = by1check.render_name(derived, li, stack, owner, lname)
        except Exception:
            nm = None
        hit = [] if nm is None else \
            [c for c in cands if c == nm or c == nm + _suffix_of(c, lname)]
        if len(hit) == 1:
            paired[(kind, li, stack, owner, lname)] = hit
        else:
            unpaired.append("layer %s.%s (多解 %s)" % (owner, lname, cands[:2]))

    # ── 回放验证：推出来的模板渲染全部行，**渲染集合 == 配对集合** ──
    # 普通行渲染 1 个名；量化行渲染 基名+每个后缀；专家行渲染 e=0..n-1。
    # 配对集合就是 refs 里那一行的真实名字（形状已在配对时验过），
    # 所以"集合相等" = 模板把 refs 的命名**一个不差地**复现出来。

    ok = bad = absent_ok = absent_bad = 0
    bad_names, resid = [], []
    for kind, li, stack, owner, lname, shape_txt, pe in items:
        key = (kind, li, stack, owner, lname)
        exps = paired.get(key)
        if shape_txt == "--":
            nm = by1check.render_name(
                dict(derived, name=derived.get("global_name", "{physical}")),
                0, "<global>", "global", lname) if kind == "global" \
                else by1check.render_name(derived, li, stack, owner, lname)
            if nm in real:
                absent_bad += 1
                bad_names.append("应缺席却存在 %s" % nm)
            else:
                absent_ok += 1
            continue
        if exps is None:
            continue                    # 配不上的行已在 unpaired 记账
        def _render(rule, e=None):
            if kind == "global":
                return by1check.render_name(
                    dict(rule, name=rule.get("global_name", "{physical}")),
                    0, "<global>", "global", lname)
            return by1check.render_name(rule, li, stack, owner, lname, e)
        if key in expert_keys and "expert_name" in derived:
            got = {_render(derived, e) for e in range(experts_of.get(owner, 1))}
        else:
            base = _render(derived)
            got = {base + _suffix_of(nm, lname) for nm in exps}
        want = set(exps)
        if got == want:
            ok += 1
        else:
            bad += 1
            if len(bad_names) < 5:
                bad_names.append("渲染 %s != refs %s"
                                 % (sorted(got)[:2], sorted(want)[:2]))
        if declared:
            try:
                if key in expert_keys and (declared.get("expert_name")
                                           or declared.get("expert_name_" + stack)):
                    dnm = {_render(declared, e)
                           for e in range(experts_of.get(owner, 1))}
                else:
                    dnm = {_render(declared) + _suffix_of(nm, lname)
                           for nm in exps}
            except Exception:
                dnm = set()
            if dnm != got:
                resid.append("%s.%s: 反推渲染 %s != 手写渲染 %s"
                             % (owner, lname, sorted(got)[:1], sorted(dnm)[:1]))

    return {"model": os.path.basename(by1_path), "backend": backend,
            "declared_total": len([1 for it in items if it[5] != "--"]),
            "replay_ok": ok, "replay_bad": bad,
            "absent_ok": absent_ok, "absent_bad": absent_bad,
            "unpaired": unpaired, "problems": problems, "resid": resid,
            "manual_renames": manual_renames,
            "derived": derived, "bad_names": bad_names,
            # C2 的机器生成判定：整份命名都被反推复现（回放全绿），
            # 且没有配不上 / 推不出 / 借声明 rename 的行 ——
            # 借来的 rename 就是"手写决策残余"，不许算机器生成。
            "machine": bad == 0 and absent_bad == 0
                       and not unpaired and not problems
                       and not manual_renames}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="只跑文件名含这个词的模型")
    ap.add_argument("--backend", default="torch.module")
    ap.add_argument("--report", action="store_true", help="只出 C2 报表")
    args = ap.parse_args(argv)

    rows, machine, residue = [], 0, 0
    for f in sorted(glob.glob(os.path.join(by1paths.MODELS, "*.by1"))):
        if args.only and args.only not in f:
            continue
        base = os.path.basename(f)[:-4]
        if base.endswith("selftest") or base.startswith("complete-"):
            continue
        cands = glob.glob(by1paths.ref("*__%s.tensors.json" % base)) \
            or glob.glob(by1paths.ref("%s.tensors.json" % base))
        if not cands:
            continue
        real = by1verify.load_reference(cands[0], None)
        if not real:
            continue
        r = reverse_one(f, args.backend, real)
        rows.append(r)
        if r.get("error"):
            print("  !! %-44s %s" % (r["model"][:44], r["error"]))
            continue
        clean = r["replay_bad"] == 0 and r["absent_bad"] == 0
        status = ("机器生成 ✓" if r["machine"]
                  else "手写残余 %d 处（配不上 %d · 推不出 %d · 借声明 rename %d）"
                       % (len(r["unpaired"]) + len(r["problems"])
                          + len(r["manual_renames"]),
                          len(r["unpaired"]), len(r["problems"]),
                          len(r["manual_renames"])))
        machine += 1 if r["machine"] else 0
        residue += 0 if r["machine"] else 1
        print("  %s %-42s 回放 %d/%d  %s" % (
            "ok " if clean else "!! ", r["model"][:42], r["replay_ok"],
            r["declared_total"], status))
        if not args.report:
            for x in r["resid"][:2]:
                print("       - %s" % x)
            for p in r["problems"][:2]:
                print("       - 推不出: %s" % p)
            for u in r["unpaired"][:2]:
                print("       - 配不上: %s" % u)
            for b in r["bad_names"][:2]:
                print("       - 回放错: %s" % b)

    total = len(rows)
    print()
    print("  [C2 报表] %d 个模型：%d 机器生成 · %d 手写残余"
          % (total, machine, residue))
    if total:
        print("  机器生成率 %.0f%%（判据：只升不降）" % (100.0 * machine / total))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
