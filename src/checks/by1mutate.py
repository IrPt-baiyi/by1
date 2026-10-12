#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1mutate -- **变异反例库**（1.0.0-rc0 的 T0.3）。

每条检查规则配一个「必须红」的反例：往一份**好 IR** 里注入一处缺陷，
断言对应的规则**真的报警**（命中它自己的那句话，不是随便哪句）。

为什么值得一个专门的库：**一个只验你记得要验的东西的检查，不是检查。**
规则加了却没有反例，它可能永远绿 —— 而「永远绿」和「坏了」
长得一模一样。by1gate / by1lint 的自检早就是这个思路，
这里把它从点变成面：**by1ir.validate 的每条规则都有名字和反例**。

语料账本（全部在门禁里跑，谁也不充数）：

  · by1ir.validate 的字段/闭集/跨字段/宽度规则 —— **本文件**（逐条变异）
  · 解析器 9 条错（selftest.by1 的"正好 9 错"）  —— by1fast ③
  · 取值门（gate-probe.by1 的不存在取值）          —— by1gate（档 2）
  · 十类静默写法（by1lint 自己的反例）             —— by1lint --selftest
  · 推导↔契约（1152 bug 注入）                    —— by1sync --selftest
  · 宽度推导的 3 组反例                            —— by1ir --selftest

用法:
  python by1mutate.py
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

import copy
import importlib.util

import by1io      # noqa: F401  —— import 即钉 UTF-8


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 变异用的底子 ────────────────────────────────────────────────────

def base_ir():
    """一份**合法**的小 IR（llama-shaped 编出来的）。变异都从它出发。"""
    bc, cg = load("by1check"), load("by1codegen")
    _r, info = bc.check(by1paths.root("models", "llama-shaped.by1"))
    ir = cg.compile_ir(info)
    import by1ir as bir
    # **不用 assert**（-O 会把它删掉，检查变成无声）—— 基线坏了就大声说。
    errs = bir.validate(ir)
    if errs:
        raise SystemExit("  [FAIL] 基线（llama-shaped 的 IR）自己不合法"
                         " —— 反例库没资格说话：%s" % [e[:40] for e in errs][:2])
    return ir


def op(kind, mech, attrs, ins, outs):
    return {"kind": kind, "mech": mech, "attrs": attrs,
            "inputs": ins, "outputs": outs}


def wrap(globals_, layers):
    return {"by1-ir": "1.0", "vocab": 8, "ctx": 16, "d_model": 4,
            "pos_kind": "rope", "norm_kind": "rms", "norm_one_plus": False,
            "globals": globals_, "layers": layers}


GLUE = [op("Embed", "Embed", {}, [], ["hidden"]),
        op("Head", "Head", {}, ["hidden"], ["logits"])]

MOE_OK = {"experts": 4, "top_k": 2, "hidden": 8, "shared": 1,
          "shared_hidden": 8, "shared_gate": False,
          "routing": "sigmoid_topk", "score": "sigmoid",
          "router_bias": False, "expert_bias": False, "score_bias": False,
          "n_group": 1, "topk_group": 1, "routed_scale": 1.0,
          "act": "silu", "alpha": 1.702}


def moe_ir(**over):
    a = dict(MOE_OK)
    a.update(over)
    return wrap(GLUE, [{"index": 0, "attrs": {}, "ops": [
        op("MoE", "MoE", a, ["hidden"], ["op0.out"])]}])


def ext_ir(**over):
    a = {"lib": "x.so", "symbol": "sym", "weights": {"w": [4, 4]},
         "io": "same"}
    a.update(over)
    return wrap(GLUE, [{"index": 0, "attrs": {}, "ops": [
        op("External", "External", a, ["hidden"], ["op0.out"])]}])


# ── 变异表：(id, 规则, 变异, 期望命中的那句话) ───────────────────────
# **期望命中规则自己的话** —— 断言"报了点什么"太松：一个坏 IR 可能
# 顺带触发别的规则，那样反例就算过了，而目标规则其实是瞎的。

def _mut(fn, base):
    ir = copy.deepcopy(base)
    fn(ir)
    return ir


def _on_op(ir, li, j, fn):
    fn(ir["layers"][li]["ops"][j])


MUTATIONS = [
    ("top-not-dict", "顶层必须是字典",
     lambda b: ["not", "a", "dict"], "顶层必须是字典"),
    ("no-version", "by1-ir 版本必填",
     lambda b: _mut(lambda ir: ir.pop("by1-ir"), b), "缺 `by1-ir` 版本号"),
    ("bad-major", "大版本不匹配直接拒",
     lambda b: _mut(lambda ir: ir.update({"by1-ir": "9.9"}), b),
     "不是我能读的大版本"),
    ("top-missing", "顶层缺必填字段",
     lambda b: _mut(lambda ir: ir.pop("vocab"), b), "缺必填字段 `vocab`"),
    ("top-bad-type", "顶层字段类型",
     lambda b: _mut(lambda ir: ir.update({"vocab": "many"}), b),
     "vocab 应当是整数"),
    ("top-extra", "顶层多余字段（规格表是唯一定义）",
     lambda b: _mut(lambda ir: ir.update({"vocab_size": 8}), b),
     "多出来的字段"),
    ("learned-no-npos", "pos_kind=learned 必须有 n_pos",
     # **顺手把 n_pos 删掉** —— compile_ir 会补一个缺省 n_pos，
     # 只改 pos_kind 的话规则根本没机会响（第一版就是这么空过的）。
     lambda b: _mut(lambda ir: (ir.update({"pos_kind": "learned"}),
                                ir.pop("n_pos", None)), b),
     "必须有 `n_pos`"),
    ("op-bad-kind", "kind 是闭集",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 0,
                                      lambda o: o.update({"kind": "Nope"})), b),
     "不在闭集"),
    ("attrs-not-dict", "attrs 必须是字典",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 0,
                                      lambda o: o.update({"attrs": 5})), b),
     "应当是字典"),
    ("attrs-extra", "attrs 反向检查（表里没有的属性不合法）",
     lambda b: _mut(lambda ir: _on_op(
         ir, 0, 0, lambda o: o["attrs"].update({"eps_iolent": 1})), b),
     "多出来的字段"),
    ("attrs-missing", "attrs 必填项",
     lambda b: _mut(lambda ir: _on_op(
         ir, 0, 0, lambda o: o["attrs"].pop("eps")), b), "缺必填项 `eps`"),
    ("attrs-bad-enum", "enum 闭集取值",
     lambda b: _mut(lambda ir: _on_op(
         ir, 0, 4, lambda o: o["attrs"].update({"act": "nosuchact"})), b),
     "不在闭集"),
    ("moe-score-unused", "MoE：不经过打分函数的选法不许写 score",
     lambda b: moe_ir(routing="softmax_topk", score="sigmoid"),
     "不经过打分函数"),
    ("moe-contradict", "MoE：noaux_tc 的打分不能是 softmax",
     lambda b: moe_ir(routing="sigmoid_group_topk", score="softmax"),
     "noaux_tc"),
    ("moe-two-sources", "MoE：score 与 _score 两个真相源",
     lambda b: moe_ir(score="sigmoid", _score="softmax"), "对不上"),
    ("ext-empty-weights", "External 契约不松（weights 不能空）",
     lambda b: ext_ir(weights={}), "weights 不能是空的"),
    ("width-bad-ref", "值引用必须存在（拼写坏）",
     lambda b: _mut(lambda ir: _on_op(
         ir, 0, 2, lambda o: o.update({"inputs": ["hidden", "op0.ott"]})), b),
     "不存在的值"),
    ("width-use-before", "值引用不许用在产出之前",
     lambda b: _mut(lambda ir: _on_op(
         ir, 0, 1, lambda o: o.update({"inputs": ["op2.out"]})), b),
     "不存在的值"),
    ("width-globals-chain", "全局链的值引用",
     lambda b: _mut(lambda ir: ir["globals"][1].update(
         {"inputs": ["normed"]}), b), "不存在的值"),
]


def main():
    import by1ir as bir
    base = base_ir()
    bad = []
    print("  变异反例库：by1ir.validate 的每条规则一个「必须红」")
    for mid, rule, mut, needle in MUTATIONS:
        try:
            ir = mut(base)
        except Exception as e:
            bad.append("%s —— 变异本身崩了：%s" % (mid, e))
            continue
        errs = bir.validate(ir)
        if not any(needle in e for e in errs):
            bad.append("%s（%s）没被抓住 —— 期望命中『%s』，实得 %s"
                       % (mid, rule, needle, [e[:40] for e in errs][:2]))
        else:
            print("     ✓ %-20s %s" % (mid, rule))

    # **换宽度是缺口不是错误** —— 单独一条：errs 必须空、gaps 必须有。
    ir = wrap(GLUE, [
        {"index": 0, "attrs": {}, "in_dim": 4,
         "ops": [op("Add", "Add", {}, ["hidden", "hidden"], ["op0.out"])]},
        {"index": 1, "attrs": {}, "in_dim": 2,
         "ops": [op("Add", "Add", {}, ["hidden", "hidden"], ["op0.out"])]}])
    _W, e, g = bir.infer_widths(ir)
    if e:
        bad.append("gap-not-error：换宽度不该报错，实报 %s" % e[:1])
    elif not g:
        bad.append("gap-not-error：换宽度没进 gaps")
    else:
        print("     ✓ %-20s %s" % ("gap-not-error", "换宽度进 gaps 不进 errs"))

    # 正例：基线必须全绿（反例全红而正例不绿 = 检查器只会报错）
    if bir.validate(copy.deepcopy(base)):
        bad.append("基线（llama-shaped 的 IR）不绿 —— 检查器只会报错")

    print()
    if bad:
        for b in bad:
            print("  [FAIL] " + str(b)[:170])
        print("  [FAIL] 变异反例库：%d 处" % len(bad))
        return 1
    print("  [PASS] 变异反例库：%d 个反例全红 + 基线全绿"
          "（解析 9 错 / 取值门 / 静默写法 / 1152 注入 / 宽度 3 组"
          " 各在自己的检查里跑）" % (len(MUTATIONS) + 1))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
