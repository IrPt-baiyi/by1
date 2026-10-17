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
  · 静默写法（by1lint 自己的反例）             —— by1lint --selftest
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


MLA_OK = {"q": 2, "v_dim": 8, "qk_nope": 8, "qk_rope": 4, "q_lora": 4,
          "kv_lora": 4, "head_dim": 8, "out_dim": 16, "bias": False,
          "rope_base": 10000, "pairing": "interleaved",
          "head_gate": "off", "gate_act": "softplus"}


def mla_ir(**over):
    a = dict(MLA_OK)
    a.update(over)
    return wrap(GLUE, [{"index": 0, "attrs": {}, "ops": [
        op("MLA", "MLA", a, ["hidden"], ["op0.out"])]}])


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
    # ── 五类反例补口（1.0 阶段 A 判据）──────────────────────────────
    ("state-bad-dtype", "STATE.dtype 闭集（未知 dtype 必拒）",
     lambda b: _mut(lambda ir: ir["layers"][0].update({"state": [
         {"kind": "kv_cache", "bounded_by": None, "dtype": "not_a_precision",
          "reuse": "none"}]}), b), "不在闭集"),
    ("attn-bad-pairing", "Attention.rope_pairing 闭集（不支持的注意力模式）",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 1,
                                      lambda o: o["attrs"].update(
                                          {"rope_pairing": "diagonal"})), b),
     "不在闭集"),
    # ── 旋转宽度必须是偶数（2026-10-16 外部复现：hello head_dim=31）──
    # 判据按**旋转宽度**，不是 head_dim；见 by1ir 里那条规则的注释。
    # 32 基线 / 30 近旁正例 / 31 负例是发现者给的对照设计。
    ("attn-odd-rope", "Attention：rope 开着时旋转宽度必须是偶数（31 必拒）",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 1,
                                      lambda o: o["attrs"].update(
                                          {"head_dim": 31})), b),
     "是奇数"),
    ("attn-odd-rope-partial", "旋转宽度按 rope_partial 折完再判（30×0.5=15 必拒）",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 1,
                                      lambda o: o["attrs"].update(
                                          {"head_dim": 30,
                                           "rope_partial": 0.5})), b),
     "是奇数"),
    ("attn-odd-rope-half", "规则与配对无关（half 配对的奇数旋转宽度同样必拒）",
     lambda b: _mut(lambda ir: _on_op(ir, 0, 1,
                                      lambda o: o["attrs"].update(
                                          {"head_dim": 31,
                                           "rope_pairing": "half"})), b),
     "是奇数"),
    ("mla-odd-qk-rope", "MLA：qk_rope 旋转宽度必须是偶数",
     lambda b: mla_ir(qk_rope=5), "是奇数"),
    ("mla-odd-index-dim", "MLA：索引器 index_dim（固定 interleaved）必须是偶数",
     lambda b: mla_ir(index_dim=31), "是奇数"),
    # ── T22（by1-ir 1.1 STACKS：栈/塔登记表）────────────────────────
    ("stacks-bad-modal", "栈 modality 闭集（未知模态必拒）",
     lambda b: _mut(lambda ir: ir.update({"stacks": [
         {"name": "vision", "modality": "television", "width": 8,
          "first_layer": 0, "n_layers": 1}]}), b), "不在闭集"),
    ("stacks-width-drift", "跨栈宽度一致性（层 in_dim 与栈宽不符当红）",
     lambda b: _mut(lambda ir: ir.update({"stacks": [
         {"name": "vision", "modality": "vision", "width": 128,
          "first_layer": 0, "n_layers": 1}]}), b), "不符"),
    ("stacks-range-overflow", "栈层范围越界必拒",
     lambda b: _mut(lambda ir: ir.update({"stacks": [
         {"name": "vision", "modality": "vision", "width": 8,
          "first_layer": 9, "n_layers": 9}]}), b), "越界"),
]

#: 五类反例的**覆盖账**（阶段 A 判据：缺字段 / 冲突配置 / 未知 dtype /
#: 权重别名断开 / 不支持的注意力模式 —— 都必须可测）。每类至少一个
#: 「必红」反例；判卷人不在本库的，**点名到人**，不许含糊。
CLASSES = {
    "缺字段": ["by1mutate:no-version", "by1mutate:top-missing",
               "by1mutate:attrs-missing", "by1mutate:learned-no-npos"],
    "冲突配置": ["by1mutate:moe-score-unused", "by1mutate:moe-contradict",
                 "by1mutate:moe-two-sources"],
    "未知 dtype": ["by1mutate:state-bad-dtype",
                   "by1state:state dtype 不在闭集报 E（by1check 管线）"],
    "权重别名断开": ["by1complete:缺共享声明（tie_word_embeddings 摘掉必红）"],
    "不支持的注意力模式": ["by1mutate:attn-bad-pairing",
                           "by1mutate:attrs-bad-enum",
                           "by1mutate:attn-odd-rope",
                           "by1mutate:attn-odd-rope-partial",
                           "by1mutate:mla-odd-qk-rope"],
}


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

    # **近旁正例**（发现者的对照设计：32 基线 / 30 近旁 / 31 负例）：
    # 反例全红而近旁不绿 = 规则过敏。30 必须绿；rope **关着**的 31 也必须绿
    # （GPT-2 那条路，位置在输入上）—— 不许做成「见奇数就拒」的全局规则。
    ir30 = _mut(lambda ir: _on_op(
        ir, 0, 1, lambda o: o["attrs"].update({"head_dim": 30})), base)
    e30 = bir.validate(ir30)
    if e30:
        bad.append("rope-even-nearmiss：head_dim=30 不该报错，实报 %s"
                   % [e[:40] for e in e30][:2])
    else:
        print("     ✓ %-20s %s" % ("rope-even-nearmiss",
                                   "head_dim=30 近旁正例绿"))
    ir_off = _mut(lambda ir: _on_op(
        ir, 0, 1, lambda o: o["attrs"].update({"head_dim": 31,
                                               "rope": False})), base)
    e_off = bir.validate(ir_off)
    if e_off:
        bad.append("rope-off-odd-ok：rope=false 的 31 不该报错，实报 %s"
                   % [e[:40] for e in e_off][:2])
    else:
        print("     ✓ %-20s %s" % ("rope-off-odd-ok",
                                   "rope=false 时奇数 head_dim 合法"))

    # 正例：基线必须全绿（反例全红而正例不绿 = 检查器只会报错）
    empty = [c for c, items in CLASSES.items() if not items]
    if empty:
        bad.append("五类覆盖有空类：%s" % empty)
    print()
    print("  五类反例覆盖（阶段 A 判据）：")
    for cls, items in CLASSES.items():
        print("    %-12s %s" % (cls, " · ".join(items)))
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
