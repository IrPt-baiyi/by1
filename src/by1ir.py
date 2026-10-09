#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ir -- **规范化的 by1 IR：schema、校验、JSON 往返。**

## 为什么要有这个

在这之前，IR 是**由生产者定义的** —— `compile_ir` 吐一个 dict，
三个消费者各取所需，**没有一处说这个 dict 里每个键是什么意思**。
于是：

    by1codegen.render(info)      要 info，不是 ir
    by1c.emit_c(ir, info, params) 要两个（info 其实是摆设）
    by1exec.shapes_of(ir)        只要 ir

**IR 不自足。** 想写第四个后端的人，必须先学会 `.by1`。

## 这个模块要做的事

    ① 版本号       `by1-ir`，改动可见
    ② 字段规格     必填 / 可选 / 语义 —— 以**数据**形式给出，不是文档
    ③ 校验         validate(ir) -> [错误]
    ④ JSON 往返    to_json / from_json，**往返等价**

## 为什么规格写成数据而不是文档

因为这个项目最贵的一课是：

> **一个只验你记得要验的东西的检查，不是检查。**

文档会过期；而写在这里的必填项、闭集、类型，**会被 `validate` 和
`by1all` 真的跑一遍**。
"""
import json
import by1io
import by1paths
import by1skip

# **从 by1ver 来，不在这里写死。**
# 这里原来是 `VERSION = "1.0"`，而 by1codegen / by1boot / by1extdemo
# 各自**又写了一遍** —— 三份写着同一个字符串，谁也不认识谁。
# 改一份忘一份不会崩，只会造出"声称不同版本"的 IR。
try:
    from by1ver import IR_VERSION as VERSION
except ImportError:                     # 单独拷一个文件出去时兜底
    VERSION = "1.0"

#: **归一化 eps 的默认值 —— 唯一一处。**
#
# 这个数字原来在 10 个文件里写了约 50 次（`ir.get("norm_eps", 1e-5)`
# 这种形状），而**每一次"某一个后端写死了它"都是同一个 bug**：
# `history/ir.md` 数过，`eps` 这一类**犯过五次**，第五次是护栏自己
# 逼出来的。`by1c` 的 C_HEAD 里还留着那次的注释：
#
#     /* eps 必须由调用方传进来 —— 原来写死 1e-5，而 mla-shaped 的
#        rms_eps 是 1e-6。误差 1.5e-04，而其它四个模型恰好都是 1e-5
#        所以一直没露。 */
#
# 而**那句话没管住 qk_norm 那条路**：`by1exec` 调 `rms_norm` 时
# 不传 eps（吃默认参数），`by1c` 的 `qk_norm()` 里直接写 `1e-5f`
# —— 于是 clef-tiny（`qk_norm = per_head`、`norm_eps = 1e-6`）
# 在三个后端里算出三个不同的值，相对差 1.05e-05，
# **低于 1e-4 的判据所以一直是绿的**。
#
# 默认值本身不是错（大多数模型确实是 1e-5）；错的是**它被写了 50 遍**，
# 于是"哪一处漏了传参"永远只有出事之后才知道。现在它是一个名字。
EPS_DEFAULT = 1e-5

# ── 顶层字段 ────────────────────────────────────────────────────────
# (名字, 类型, 必填, 说明)
TOP = [
    ("by1-ir", "str", True, "IR 版本。**读的一方必须拒绝不认识的大版本。**"),
    ("vocab", "int", True, "词表大小"),
    ("ctx", "int", True, "上下文长度（位置能达到的最大值）"),
    ("d_model", "int", True, "残差流宽度"),
    # **闭集要写在类型里，不能只写 `enum`。**
    # 这里原来是裸的 `"enum"`，而 `_ty` 只认 `enum:...`（带冒号）——
    # 于是这两个字段**从来没有被校验过**：`pos_kind = BANANA` 一路通到
    # 后端，而后端判的是 `== "learned"`，非法值静默落回 rope。
    # 同一个形状的漏洞还有 STATE.kind / STATE.reuse / STATE.dtype。
    ("pos_kind", "enum:rope,learned", True,
     "位置信息的种类。`rope`=在注意力里旋转；`learned`=在输入上加一张查表。"
     "**这两件事不一样**，所以是一个字段而不是一个布尔。"),
    ("n_pos", "int", False, "pos_kind=learned 时必填：查表有多少行"),
    ("norm_kind", "enum:rms,layer", True,
     "归一化的种类。`rms`=只除均方根；`layer`=减均值+除标准差+有 bias。"
     "**名字像，算的不是一回事。**"),
    ("norm_eps", "float", True, "归一化的 eps。**必须来自这里，不许各后端写死**"
                                "（这个 bug 犯过三次）"),
    ("norm_one_plus", "bool", True, "归一化权重是乘 `w` 还是乘 `(1+w)`"),
    ("globals", "list[Op]", True, "不属于任何层的算子（embed / final_norm / head）"),
    ("layers", "list[Layer]", True, "逐层"),
]

LAYER = [
    ("index", "int", True, "层号。**多栈时是各栈从 0 起**，不是全局序号"),
    ("attrs", "dict", True, "这一层的属性（逐层覆盖合并之后的结果）"),
    ("ops", "list[Op]", True, "这一层的算子序列，**顺序有意义**"),
    ("state", "list[State]", False, "这一层需要什么状态（KV cache / recurrent / conv）"),
]

OP = [
    # `kind` 的闭集是 `KIND_ATTRS` 的键（见下面的表），所以这里用 `str`：
    # 真正的判定在 `_check_op` 里对着那张表做，一处定义。
    ("kind", "str", True, "算子种类，**闭集** —— 见 KIND_ATTRS"),
    ("mech", "str", True, "机制名（.by1 里写的那个名字）。只用于报错和取名"),
    ("attrs", "dict", True, "属性。**每种 kind 有自己的必填项** —— 见 KIND_ATTRS"),
    ("inputs", "list[str]", True, "输入的值引用（`hidden` 或 `opN.out`）"),
    ("outputs", "list[str]", True, "输出的值引用"),
]

STATE = [
    ("kind", "enum:recurrent,conv_history,kv_cache,declared_by_impl", True,
     "recurrent / conv_history / kv_cache / declared_by_impl"),
    ("bounded_by", "int?", True, "状态有没有上界。None 表示无界，不是「没有」"),
    ("shape", "list[int]?", False, "状态张量的形状"),
    ("dtype", "str?", False, "状态存什么精度（`bf16` / `fp32` …）"),
    ("reuse", "enum:none,prefix,unknown", True,
     "none / prefix / unknown —— 能不能跨请求复用"),
    # **_spec 里漏了它，而生产者在写。** 4 处（mla-shaped 的压缩潜向量）。
    # 一个"规格比现实窄"的字段会把合法 IR 判成非法 —— 而这正是
    # 加"多余字段"检查时冒出来的：先照现实补规格，再让检查成立。
    ("note", "str?", False, "自由说明（人看的，不参与计算）"),
]

# ── 每种算子的属性 ──────────────────────────────────────────────────
# (名字, 类型, 必填, 说明)。**这份表就是"IR 里合法的东西"的唯一定义。**
KIND_ATTRS = {
    "Embed": {},
    "Norm": {
        "kind": ("enum:rms,layer", True, "和顶层 norm_kind 一致（冗余但显式）"),
        "eps": ("float", True, "来自顶层 norm_eps"),
        "one_plus": ("bool", True, ""),
    },
    "Add": {},
    # 输出头（lm_head）。**它也是一个算子** —— 规格表漏了它，
    # 于是 19 个模型全被报"不在闭集里"。
    "Head": {},
    "Attention": {
        "q": ("int", True, "查询头数"),
        "kv": ("int", True, "KV 头数"),
        "head_dim": ("int", True, "每个头的宽度"),
        "out_dim": ("int", True, "输出投影的输入宽度。**不一定等于 q*head_dim**"
                                "（GPT-OSS 是 4096 而 q*head_dim=4096，Gemma 不是）"),
        "bias": ("bool", True, "投影有没有 bias"),
        "window": ("int?", True, "滑动窗口。**None=全量**"),
        "qk_norm": ("enum:off,per_head,full", True,
                    "拆头前还是后、按 head_dim 还是整宽 —— **三种，不是布尔**"),
        "norm_eps": ("float", False, ""),
        "norm_one_plus": ("bool", False, ""),
        "rope": ("bool", True, "**要不要施加旋转**。false 时位置在输入上"),
        "rope_base": ("int", True, ""),
        "rope_pairing": ("enum:half,interleaved", True, "配对约定"),
        "rope_partial": ("float", True, "只转前多少比例"),
        "rope_scale": ("float", True, "YaRN 的 attention_factor 之类"),
        "yarn": ("dict?", True, ""),
        "q_gate": ("bool", True, "q 投影输出翻倍，一半当门"),
        "sink": ("bool", True, "每个头一个可学标量参与 softmax"),
        "kv_tie": ("bool", True, "k 和 v 共用同一块"),
        "head_gate": ("enum:off,per_head", True, ""),
        "gate_act": ("enum:softplus,sigmoid", True, "**同名不同义的高发区**"),
    },
    "MLA": {
        "q": ("int", True, ""), "v_dim": ("int", True, ""),
        "qk_nope": ("int", True, ""), "qk_rope": ("int", True, ""),
        "q_lora": ("int", True, ""), "kv_lora": ("int", True, ""),
        "head_dim": ("int", True, ""), "out_dim": ("int", True, ""),
        "bias": ("bool", True, ""), "rope_base": ("int", True, ""),
        "pairing": ("enum:half,interleaved", True, ""),
        "norm_eps": ("float", False, ""), "norm_one_plus": ("bool", False, ""),
        "head_gate": ("enum:off,per_head", True, ""),
        "gate_act": ("enum:softplus,sigmoid", True, ""),
    },
    "FFN": {
        "hidden": ("int", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True,
                "**闭集** —— 取值不在里面必须拒绝，不许静默走默认分支"),
        "gate": ("bool", True, "有门=三块矩阵；没门=两块。**GPT-2 是没有的那种**"),
        "bias": ("bool", True, ""),
        "limit": ("float?", True, "SwiGLU 的夹取"),
        "alpha": ("float", True, "gptoss 那个 1.702"),
    },
    "MoE": {
        "experts": ("int", True, ""), "top_k": ("int", True, ""),
        "hidden": ("int", True, "专家的中间宽度"),
        "shared": ("int", True, "共享专家个数"),
        "shared_hidden": ("int", True, ""),
        "shared_gate": ("bool", True, ""),
        "routing": ("enum:softmax_topk,topk_softmax,sigmoid_group_topk,sigmoid_topk",
                    True, "**四种，名字只差一个词**"),
        "router_bias": ("bool", True, ""), "expert_bias": ("bool", True, ""),
        "score_bias": ("bool", True, "noaux_tc 的 e_score_correction_bias"),
        "n_group": ("int", True, "分组路由"), "topk_group": ("int", True, ""),
        "routed_scale": ("float", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "limit": ("float?", True, ""), "limit_shared": ("float?", True, ""),
        "alpha": ("float", True, ""),
        # **没有笼统的 `bias`。** MoE 用的是 router_bias 和 expert_bias，
        # 分开的两个。规格表第一版照着 FFN 抄了一个 bias 进来，于是
        # 210 条"缺必填项"，而**没有任何东西产生它**。
        # **规格比现实严，报的是假问题** —— 而假问题会把真问题淹掉。
    },
    "Linear": {
        "k_heads": ("int", True, ""), "k_dim": ("int", True, ""),
        "v_heads": ("int", True, ""), "v_dim": ("int", True, ""),
        "conv_kernel": ("int", True, ""), "out_dim": ("int", True, ""),
        "act": ("enum:silu,gptoss,gelu,gelu_new,relu2", True, ""),
        "norm_eps": ("float", False, ""), "l2_eps": ("float", False, ""),
    },
    "Raw": {
        "impl": ("str", True,
                 "逃生舱第一层：raw.py 里的工厂函数名。"
                 "**这一层要改编译器** —— 加一个机制就得动 src/by1codegen.py。"
                 "第二层见 External"),
    },
    # ── 逃生舱第二层：引用一个**外部符号** ────────────────────────
    # 编译器不认识这个机制，只认识**调用约定**。
    # 所以加一个新机制**不需要动编译器** —— 只要有一个 .so 和一个 IR。
    #
    # ## ABI（固定，不自由发挥）
    #
    #     void <symbol>(const float *x, float *y,
    #                   int B, int T, int D,
    #                   const float *const *w, int nw);
    #
    #     x    输入，行主序 [B, T, D]
    #     y    输出，行主序 [B, T, D]（io = "same" 时形状和 x 一样）
    #     w    权重指针数组，**按名字字典序排列**
    #     nw   w 的个数
    #
    # ## 为什么权重是"指针数组"而不是拼成一块
    #
    # 因为 C 那边权重本来就是分开的缓冲区。拼成一块要么多一次拷贝，
    # 要么要求布局一致 —— 而那是**约定**，不是**语义**。
    # 指针数组 + 排序规则，两边都能自己算出来，不需要额外通道。
    "External": {
        "lib": ("str", True,
                "动态库路径（.so / .dylib / .dll / .o）。"
                "**相对路径按 IR 文件所在目录解析** —— 否则换个目录就跑不了"),
        "symbol": ("str", True, "符号名。找不到就拒绝，**不静默给个恒等**"),
        "weights": ("dict", True,
                    "这个算子自己的张量：{逻辑名: 形状}。"
                    "**契约照样查** —— 这就是「下沉一层」没有放松的地方"),
        "io": ("enum:same", True,
               "输出和输入同形。先只支持这一种 —— "
               "多一种就要多一条约定，而约定越多越像糊"),
        "note": ("str?", False, "给人看的说明"),
    },
}

KINDS = sorted(KIND_ATTRS)


class IRError(Exception):
    pass


def _ty(name, t, v):
    if t.startswith("enum:"):
        allow = t[5:].split(",")
        if v not in allow:
            return "%s = <%r> 不在闭集 %s 里" % (name, v, allow)
        return None
    if t == "int":
        return None if isinstance(v, int) and not isinstance(v, bool) else \
            "%s 应当是整数，实际 %r" % (name, v)
    if t == "int?":
        return None if v is None or (isinstance(v, int)
                                     and not isinstance(v, bool)) else \
            "%s 应当是整数或 null，实际 %r" % (name, v)
    if t == "float":
        return None if isinstance(v, (int, float)) and not isinstance(v, bool) \
            else "%s 应当是数，实际 %r" % (name, v)
    if t == "float?":
        return None if v is None or isinstance(v, (int, float)) else \
            "%s 应当是数或 null，实际 %r" % (name, v)
    if t == "bool":
        return None if isinstance(v, bool) else "%s 应当是布尔，实际 %r" % (name, v)
    if t == "str":
        return None if isinstance(v, str) and v else "%s 应当是非空字符串" % name
    # **`str?` 以前不存在。** 它掉到最后那行 `return None` = 接受，
    # 于是 `STATE.dtype = {"not": "a string"}` 是"合法 IR"。
    if t == "str?":
        return None if v is None or (isinstance(v, str) and v) else \
            "%s 应当是非空字符串或 null，实际 %r" % (name, v)
    if t == "dict":
        return None if isinstance(v, dict) else "%s 应当是字典" % name
    if t == "dict?":
        return None if v is None or isinstance(v, dict) else "%s 应当是字典或 null" % name
    if t.startswith("list"):
        if t.endswith("?"):        # list[...]? —— 允许 null
            return None if v is None or isinstance(v, list) else \
                "%s 应当是列表或 null" % name
        return None if isinstance(v, list) else "%s 应当是列表" % name
    # **不认识的类型字符串 = 这张表自己写错了。**
    # 原来这里是 `return None`（接受），于是一个拼错的类型名会
    # **静默地把那个字段变成不校验** —— 这就是 `enum` 丢掉闭集的那次。
    # 规格表自己坏了必须出声，不能变成"这个字段没规矩"。
    return ("%s: 规格表里的类型 <%s> 没人认识 —— "
            "**这是 by1ir 自己的 bug，不是 IR 的**" % (name, t))


def _keys_bad(where, obj, allowed):
    """多余字段 -> 错误。**规格表是"合法的东西"的唯一定义。**

    `_check_fields` 只从表往对象看（查必填、查类型），
    **从不反着看** —— 于是 `attrs` 里塞一个谁都不认识的键，
    `validate` 说合法，而 `.by1` 那边写错了属性的行为是"静默走默认值"。
    规格文档里那句"不在表里的属性 = 不合法"以前没有执行者，这就是它。
    """
    extra = [k for k in obj if k not in allowed]
    if extra:
        return ("%s: 多出来的字段 %s —— **不在规格表里就是不合法的**"
                % (where, ", ".join("`%s`" % k for k in sorted(map(str, extra)))))
    return None


def _check_fields(where, fields, obj, errs):
    """**先查类型，再查字段。** 顺序反了的话，一个 `layers=[42]`
    会在 `_ty` 之前就撞上 `name not in obj`（int 不可迭代）而抛异常。"""
    if not isinstance(obj, dict):
        errs.append("%s 应当是字典，实际 %s" % (where, type(obj).__name__))
        return
    allowed = {name for name, _t, _r, _d in fields}
    e = _keys_bad(where, obj, allowed)
    if e:
        errs.append(e)
    for name, t, req, _doc in fields:
        if name not in obj:
            if req:
                errs.append("%s 缺必填字段 `%s`" % (where, name))
            continue
        e = _ty(name, t, obj[name])
        if e:
            errs.append("%s: %s" % (where, e))


def validate(ir):
    """返回错误列表。**空列表 = 合法。**

    ## 这份函数**不许抛异常**

    它是对外入口的一半（`by1ir.py --check <file>.ir.json`），
    而 [by1ir.py] 的模块说明里写着"有人可以直接写 IR"。
    契约说"返回错误列表"，那畸形输入就得进列表 —— 以前
    `layers = [42]` 会在这里抛 `TypeError`，而那个 traceback
    长得像"by1ir 坏了"，不像"你的 IR 坏了"。
    """
    errs = []
    if not isinstance(ir, dict):
        return ["IR 的顶层必须是字典"]
    ver = ir.get("by1-ir")
    if ver is None:
        errs.append("缺 `by1-ir` 版本号 —— **没有版本号的 IR 不该被接受**")
    elif str(ver).split(".")[0] != VERSION.split(".")[0]:
        errs.append("IR 版本 %s 不是我能读的大版本 %s" % (ver, VERSION))

    _check_fields("顶层", TOP, ir, errs)

    if ir.get("pos_kind") == "learned" and "n_pos" not in ir:
        errs.append("pos_kind=learned 时必须有 `n_pos`")

    for sec in ("globals", "layers"):
        if not isinstance(ir.get(sec), list):
            continue
        for i, item in enumerate(ir[sec]):
            if sec == "globals":
                _check_op("globals[%d]" % i, item, errs)
                continue
            w = "layers[%d]" % i
            _check_fields(w, LAYER, item, errs)
            if not isinstance(item, dict):
                continue
            ops = item.get("ops")
            for j, op in enumerate(ops if isinstance(ops, list) else []):
                _check_op("%s.ops[%d]" % (w, j), op, errs)
            sts = item.get("state")
            for k, st in enumerate(sts if isinstance(sts, list) else []):
                _check_fields("%s.state[%d]" % (w, k), STATE, st, errs)
    return errs


def _check_op(where, op, errs):
    _check_fields(where, OP, op, errs)
    if not isinstance(op, dict):
        return
    k = op.get("kind")
    # **跨字段的约束也要在这里查。** IR 现在是入口了 —— 有人可以直接写 IR，
    # 不经过 .by1。所以"编译器会拦"不再是理由：
    # `compile_ir` 里那些检查，走 IR 这条路根本跑不到。
    # 这一条就是这么漏的：External 声明空 weights 竟然通过了。
    if k == "External":
        w = (op.get("attrs") or {}).get("weights")
        if isinstance(w, dict) and not w:
            errs.append(
                "%s (External): weights 不能是空的 —— "
                "**契约不松**：外部算子照样要声明自己的张量，"
                "否则它算的东西没人查" % where)
    if k not in KIND_ATTRS:
        errs.append("%s: kind = <%s> 不在闭集 %s 里" % (where, k, KINDS))
        return
    spec = KIND_ATTRS[k]
    attrs = op.get("attrs")
    if not isinstance(attrs, dict):
        errs.append("%s (%s): attrs 应当是字典，实际 %s"
                    % (where, k, type(attrs).__name__))
        return
    # **反着也要查一遍**：规格表里没有的属性就是不合法的。
    # 规格文档（`--spec` 出的那份）一直这么写着，但在这之前**没有执行者**。
    e = _keys_bad("%s (%s) 的 attrs" % (where, k), attrs, set(spec))
    if e:
        errs.append(e)
    for name, (t, req, _doc) in spec.items():
        if name not in attrs:
            if req:
                errs.append("%s (%s) 的 attrs 缺必填项 `%s`" % (where, k, name))
            continue
        e = _ty(name, t, attrs[name])
        if e:
            errs.append("%s (%s): %s" % (where, k, e))


def to_json(ir, indent=1):
    """规范化输出：**键排序**，这样两份等价的 IR 字节相同。"""
    out = dict(ir)
    out["by1-ir"] = VERSION
    return json.dumps(out, ensure_ascii=False, indent=indent, sort_keys=True)


def from_json(s):
    """读回。**版本不对直接拒，不做兼容猜测。**"""
    ir = json.loads(s)
    errs = validate(ir)
    if errs:
        raise IRError("IR 不合法：\n  " + "\n  ".join(errs[:12]))
    return ir


def roundtrip(ir):
    """往返等价：to_json -> from_json -> **逐字段比**。"""
    back = from_json(to_json(ir))
    a = json.dumps(_norm(ir), sort_keys=True)
    b = json.dumps(_norm(back), sort_keys=True)
    if a != b:
        return False, "往返之后不一致"
    return True, ""


def spec_markdown():
    """**规格文档是从这张表生成的**，不是我手写的。

    理由：手写的文档会过期，而过期的规格比没有规格更坏 ——
    它会让读的人以为自己知道。这张表被 `validate` 和 `by1all`
    真的跑着，所以它跟得上。
    """
    L = []
    L.append("# by1 IR 规格")
    L.append("")
    L.append("> **这份文档是从 `src/by1ir.py` 里的表生成的**，不是手写的。")
    L.append("> 生成命令：`python src/by1ir.py --spec`")
    L.append(">")
    L.append("> 理由：手写的规格会过期，而过期的规格比没有规格更坏 ——")
    L.append("> 它让读的人以为自己知道。这些表被 `validate()` 和")
    L.append("> `by1all` 真的跑着，所以它们跟得上代码。")
    L.append("")
    L.append("**版本：`by1-ir: %s`**" % VERSION)
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 怎么用它")
    L.append("")
    L.append("```bash")
    L.append("python src/by1ir.py --spec                  # 生成这份文档")
    L.append("python src/by1ir.py --emit models/<文件>.by1  # 出一份规范化 JSON")
    L.append("python src/by1ir.py --check <文件>.ir.json  # 校验")
    L.append("python src/by1irentry.py                    # 三个后端只从 IR 跑")
    L.append("```")
    L.append("")
    L.append("**想写第四个后端的人从这里开始** —— 不需要先学 `.by1`。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 顶层")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in TOP:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 层")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in LAYER:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 算子")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in OP:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("`kind` 是**闭集**：%s" % ", ".join("`%s`" % k for k in KINDS))
    L.append("")
    L.append("## 状态")
    L.append("")
    L.append("| 字段 | 类型 | 必填 | 语义 |")
    L.append("|---|---|---|---|")
    for name, ty, req, doc in STATE:
        L.append("| `%s` | `%s` | %s | %s |" % (name, ty, "是" if req else "否", doc))
    L.append("")
    L.append("## 每种算子的属性")
    L.append("")
    L.append("**这张表就是「IR 里合法的东西」的唯一定义。**")
    L.append("不在表里的属性 = 不合法；表里必填而缺的 = 不合法。")
    L.append("")
    for kind in KINDS:
        spec = KIND_ATTRS[kind]
        L.append("### `%s`" % kind)
        L.append("")
        if not spec:
            L.append("（没有属性）")
            L.append("")
            continue
        L.append("| 属性 | 类型 | 必填 | 语义 |")
        L.append("|---|---|---|---|")
        for name, (ty, req, doc) in spec.items():
            L.append("| `%s` | `%s` | %s | %s |"
                     % (name, ty, "是" if req else "否", doc))
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 几条**为什么这样定**")
    L.append("")
    L.append("- **`pos_kind` 是一个枚举，不是一个布尔。**")
    L.append("  `rope` 是在注意力里旋转，`learned` 是在输入上加一张查表 ——")
    L.append("  这是两种东西，不是「有没有位置编码」。")
    L.append("- **`norm_kind` 同理。** `rms` 只除均方根，`layer` 还要减均值、")
    L.append("  还有 bias。名字像，算的不是一回事。")
    L.append("- **`Attention.rope` 是一个独立字段。**")
    L.append("  它回答「要不要转」。GPT-2 的教训：**「没有这个机制」以前")
    L.append("  表达不出来**，于是注意力无条件转了一遍，而所有检查都说「过」。")
    L.append("- **闭集就是闭集。** `act` / `qk_norm` / `routing` / `gate_act` /")
    L.append("  `pairing` 只认列出来的取值。取值多一个必须**拒绝**，")
    L.append("  不许静默走默认分支 —— 那会生成一个「看起来对但算错」的模型。")
    L.append("- **`by1-ir` 是必填的。** 没有版本号的 IR 不该被接受：")
    L.append("  读的一方无从判断自己理解的是哪一版。大版本不匹配直接拒，")
    L.append("  不做兼容猜测。")
    L.append("")
    return "\n".join(L)


def _norm(x):
    """比较用：把元组变列表、把版本号统一。"""
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _cli():
    import sys
    if '--spec' in sys.argv:
        print(spec_markdown())
        return 0
    if '--emit' in sys.argv:
        import importlib.util
        f = sys.argv[sys.argv.index('--emit') + 1]
        sp = importlib.util.spec_from_file_location(
            'bcm', by1paths.tool('by1check.py'))
        bc = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(bc)
        import by1codegen as cg
        _r, info = bc.check(f)
        print(to_json(cg.compile_ir(info)))
        return 0
    if '--check' in sys.argv:
        f = sys.argv[sys.argv.index('--check') + 1]
        errs = validate(by1io.read_json(f, encoding='utf-8'))
        for e in errs:
            print("  " + e)
        print("  [%s] %d 个问题" % ("PASS" if not errs else "FAIL", len(errs)))
        return 0 if not errs else 1
    print(__doc__)
    # 子命令不全 -> by1skip.CALLER
    return by1skip.CALLER


if __name__ == '__main__':
    import sys
    sys.exit(_cli())
