#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1export -- **导出策略层：把算好的语义变成 HF config 的字段。**

## 它为什么单独一层

`by1check.check()` 原来是 1243 行，里面同时住着三套词汇表：

    算子 / 状态层     算"第 i 层是什么机制、状态怎么流"
    导出策略层        算"config.json 里那个字段该写什么"
    权重命名          张量叫什么名字

**而 `refs/` 那份外部判卷人只判得了第三套。** 前两套的判据是自己写的
（神谕、阈值），所以"名字和形状全中"和"数算对了"是两件事 ——
这个仓库为这个栽过三次：`eps` 第六次、`apply_rope` 交错、
`gpt2.by1` 和 `llama-shaped.by1` 编译出同一份 IR。

**劈开不是为了好看，是为了让"数学语义"那一层有一条能被外部判的边界。**

## 这一层怎么拿到依赖

13 个函数都从 `make()` 的参数里取 —— **函数体一行没改**，只是自由变量
从 `check()` 的局部变成了 `make()` 的参数。缩进也不用动（两边都是 4 格）。

判据是搬完 `by1check` 对 26 份 `.by1` 的输出**逐字节不变**。
（第一次搬失败了两次：整块删吞掉了夹在中间的 `parse_rope`；
插入点放在依赖定义之前。都记在这次提交的信息里。）

## 边界上唯一不干净的地方

导出层要用 4 个**解析**函数：`_coerce` · `_ltype_of_attrs` · `parse_rope`
· `resolve_field`（最后一个 147 行）。它们语义上属于"取值解析"，
不属于导出。

**下一步劈语义层时，它们该跟谁走是那个决定的第一个问题。**
"""
import re

def make(*, _main_stack_names, hp, layer_seq, mechs, named, overrides, pb, rep, scope, ROPE_DROP, ROPE_KEYS, W, MISSING, eval_num, parse_state_key, selector_matches, split_top):
    """把 13 个导出函数交出去。**依赖全部显式传进来。**

    返回 `名字 -> 函数` 的字典。`by1check.check()` 拿到之后把它们
    绑回本地名字 —— 于是 `check()` 里所有调用点一个字不改。
    """
    def gen_layer_types():
        # 只取主栈 —— layer_types 的长度等于 num_hidden_layers，
        # 辅助栈（MTP）不在里面。
        return [_ltype_of_attrs(a, m) for (_s, m, a, _k, _t) in layer_seq
                if _s in _main_stack_names]
    def name_layer_type(nm):
        recs = named.get(nm)
        if recs:
            return _ltype_of_attrs(recs[0].attrs, nm)
        for (_s, m, a, _k, _t) in layer_seq:
            if m == nm:
                return _ltype_of_attrs(a, nm)
        return None
    def gen_rope_parameters():
        if pb is None:
            return None
        out = {}
        for k, v in pb.assigns.items():
            if k == "default":
                continue
            lt = name_layer_type(k)
            if lt is None:
                rep.add(W, pb.line, "config",
                        f"position 的键 '{k}' 对不上任何层类型，生成 config 时跳过")
                continue
            out[lt] = parse_rope(v)
        return out or None
    def gen_rope_scaling():
        """官方 config 里那份**摊平的** rope_scaling（旧格式）：
        从 position 里带缩放的那个键上取。"""
        if pb is None:
            return None
        for k, v in pb.assigns.items():
            d = parse_rope(v)
            if d.get("factor") is None:
                continue
            # **两种缩放的 config 键完全不同**，不能共用一份模板：
            #   yarn    beta_fast / beta_slow / factor / truncate
            #   llama3  factor / high_freq_factor / low_freq_factor
            # 共同项只有 original_max_position_embeddings 和 rope_type。
            ty = str(d.get("rope_type") or "default").strip().lower()
            orig = _coerce(d.get("original_max_position_embeddings",
                                 d.get("original", 4096)))
            if ty == "llama3":
                return {"factor": _coerce(d.get("factor")),
                        "high_freq_factor": _coerce(
                            d.get("high_freq_factor", d.get("high_freq", 4))),
                        "low_freq_factor": _coerce(
                            d.get("low_freq_factor", d.get("low_freq", 1))),
                        "original_max_position_embeddings": orig,
                        "rope_type": ty}
            return {"beta_fast": _coerce(d.get("beta_fast", 32)),
                    "beta_slow": _coerce(d.get("beta_slow", 1)),
                    "factor": _coerce(d.get("factor")),
                    "original_max_position_embeddings": orig,
                    "rope_type": ty,
                    "truncate": bool(d.get("truncate", True))}
        return None
    def gen_attention_other_setting():
        """滑窗那一套的注意力参数 —— 和全量层不同（Step-3.7 是 96 vs 64 头）。"""
        for (_s, m, a, _k, _t) in layer_seq:
            win = str(a.get("window", "")).strip().lower()
            if win in ("", "none", "null", "0"):
                continue
            hd = int(float(a.get("head_dim") or 0))
            return {"attention_type": "sliding_attention",
                    "head_dim": hd,
                    "num_attention_groups": int(float(a.get("kv") or 0)),
                    "num_attention_heads": int(float(a.get("q") or 0)),
                    "true_head_dim": hd}
        return None
    def gen_by_layer(key):
        """逐层字典：{层号字符串: 值}。官方 config 的 moe_num_experts_by_layer 就是它。
        **只有这个键存在的层才进字典** —— 稠密层不该出现。"""
        out = {}
        for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
            if key.startswith("attach."):
                sub, v = key[7:], ""
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        # **先看这一层有没有逐层覆盖** —— 只看声明值的话
                        # Step-3.7 的 42 层会全报 288。
                        v = overrides.get((_s, li, am), {}).get(
                            sub, ma.assigns.get(sub, ""))
                        break
                if v in ("", None):
                    continue
            else:
                v = a.get(key, "")
                if v in ("", None):
                    continue
            out[str(li)] = _coerce(v)
        return out or None
    def gen_per_layer_rope(key):
        """逐层的 rope 参数。position 是按**层类型**声明的，
        要先经 layer_types 映射到每一层 —— 所以不能直接用 per_layer()。"""
        lts = gen_layer_types() or []
        rbt = gen_rope_parameters() or {}
        return [_coerce((rbt.get(lt) or {}).get(key, "")) for lt in lts]
    def gen_join(inner, sep):
        vals = resolve_field(inner)
        if not isinstance(vals, list):
            return vals
        return sep.join(str(x) for x in vals)
    def gen_pad(inner, n, mode="zero"):
        vals = resolve_field(inner)
        if not isinstance(vals, list):
            return vals
        # 官方有些逐层数组比层数长（Step-3.7 是 48、层数是 45）。
        # **补什么，两种都有，而且长得很像**：
        #   layer_types / partial_rotary_factors / rope_theta  重复最后一个
        #   swiglu_limits / swiglu_limits_shared               补 0
        # 猜错的话数组长度对、值也对，只有尾巴不同。
        if len(vals) >= n:
            return vals[:n]
        if mode == "cycle" and vals:
            # **按周期续**：官方 Step-3.7 的 layer_types 是 48 长、模型是 45 层，
            # 45/46/47 接着 4 周期的第 1/2/3 项（都是滑窗）。
            # 既不是"重复最后一个"（那会给出全量层），也不是补零。
            #
            # ## 上界原来写的是 `len(vals) // 2`
            #
            # 于是"最小周期 > 长度的一半"时找不到周期，退化成一律重复最后一个：
            #
            #     [full, sliding, full]        最小周期 3   <- 3//2 = 1，取不到 3
            #     [A, B, A, B, A]              最小周期 2   <- 这个能取到（2 <= 2）
            #
            # 也就是说：**长度是奇数、而周期比一半大**时一定退化。
            # 当前语料碰不到（Step-3.7 是 48 长、周期 4），所以这是个潜伏的错
            # —— 而"没被碰到"不是"对"。
            #
            # 上界改成 `len(vals)` 就够：**找最小的那个匹配周期**。
            # （我第一版顺手加了"只试长度自己的因子"，那是错的 ——
            #  它会把 `[A,B,A,B,A]` 的最小周期从 2 改成 5，**改动了一个
            #  旧实现本来就对的结果**。修 bug 别顺手改对的东西；
            #  这处是 `by1check`，它决定"补什么"，改错没人看得出来。）
            per = None
            for cand in range(1, len(vals) + 1):
                if all(vals[i] == vals[i % cand] for i in range(len(vals))):
                    per = cand
                    break
            if per:
                return vals + [vals[i % per] for i in range(len(vals), n)]
            return vals + [vals[-1]] * (n - len(vals))
        fill = vals[-1] if (mode == "last" and vals) else 0
        return vals + [fill] * (n - len(vals))
    def gen_per_layer(key):
        """per_layer(q) / per_layer(attach.layer_kind) -> 逐层一个值。
        官方 config 里那一堆逐层数组（头数、MLP 类型、gating 类型）都是这个。"""
        out = []
        for (_s, _m, a, _k, atts) in layer_seq:
            if key.startswith("attach."):
                sub = key[7:]
                v = ""
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        v = ma.assigns.get(sub, ma.mtype)
                        break
                out.append(_coerce(v))
            else:
                out.append(_coerce(a.get(key, "")))
        return out
    def gen_per_layer_d(key, default):
        """per_layer(attach.swiglu_limit, 0) —— 属性**缺失**时给默认值。

        不能复用 gen_per_layer：它在属性缺失时会退回**机制类型名**
        （'MoE'/'FFN'），于是那个回退值看起来"非空"，默认值永远用不上，
        逐层数组里就混进了字符串。这里查的是属性本身有没有。
        """
        out = []
        for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
            if key.startswith("attach."):
                sub, v = key[7:], None
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        # 逐层覆盖优先于声明值
                        v = overrides.get((_s, li, am), {}).get(
                            sub, ma.assigns.get(sub))
                        break
            else:
                v = a.get(key)
            out.append(_coerce(v) if v not in (None, "") else default)
        return out
    def gen_sliding_window():
        """滑窗宽度 —— 从**开了窗的那些层**取，不是第 0 层（第 0 层可能是全量）。"""
        for (_s, _m, a, _k, _t) in layer_seq:
            w = str(a.get("window", "")).strip().lower()
            if w not in ("", "none", "null", "0"):
                return int(float(w))
        return None
    def gen_rope_parameters_flat():
        """官方**同时**有两份：摊平的 rope_scaling，和带逐层 rope_theta 的
        rope_parameters。后者就是前者加一个数组。"""
        sc = gen_rope_scaling()
        if sc is None:
            return None
        out = dict(sc)
        # 内嵌的 rope_theta 也要和顶层那份**一样**按周期补齐（48），
        # 否则两份不一致。
        out["rope_theta"] = gen_pad("per_layer_rope(rope_theta)", 48,
                                    "cycle")
        return out

    # ---- 反向：从 .by1 生成 config --------------------------------
    # 方向在这里第一次反过来：config 不是「真相」，它是**产物**。
    # 值只能来自四种来源，别的写不出来：
    #   hparams 键 / 机制属性（可带选择器）/ true|false|null|"字符串" / 两个生成器
    def _coerce(s):
        t = str(s).strip()
        if len(t) >= 2 and t.startswith('"') and t.endswith('"'):
            return t[1:-1]
        if t == "true":
            return True
        if t == "false":
            return False
        if t == "null":
            return None
        n = eval_num(t)
        if n is not None:
            return int(n) if float(n).is_integer() else n
        return t
    # 层类型不只看 window —— **还要看机制的种类**。
    # 原来这里只读 window，于是 GDN/KDA 这类线性注意力层（根本没有 window）
    # 被算成 full_attention。混合栈的 layer_types 以前从来没被检查过：
    # Ling 的 config 里没这个键，Qwen3-Next 那个没有 field 映射，
    # clef 是第一个把它摆上台面的。
    def _ltype_of_attrs(a, mech_name=None):
        # 层类型不只看 window —— **还要看机制的种类**。
        # 原来这里只读 window，于是 GDN/KDA 这类线性注意力层（根本没有 window）
        # 被算成 full_attention。混合栈的 layer_types 以前从来没被检查过：
        # Ling 的 config 里没这个键，Qwen3-Next 那个没有 field 映射，
        # clef 是第一个把它摆上台面的。
        # layer_seq 里第 2 项是**机制名**（不是种类）。
        # 种类在 Blk.mtype（Blk.kind 是"块"的种类，恒为 "mech"）。
        _m = mechs.get(mech_name) if mech_name else None
        # **显式指定优先。** 有些 config 对这一层有自己的叫法
        # （GLM-5.3 的全量层叫 `deepseek_sparse_attention`，
        #   因为它带一个稀疏索引器，不叫 full_attention）。
        _ex = (_m.assigns.get("ltype") if _m is not None else None) or ""
        if _ex.strip():
            return _ex.strip().strip('"')
        # KDA 也是线性注意力 —— 它是自己的**计算**（三个独立卷积 + o_norm），
        # 但层类型的名字是一样的。
        if _m is not None and (_m.mtype or "").strip() in (
                "Linear", "SSM", "Recurrent", "KDA"):
            return "linear_attention"
        w = (a.get("window") or "").strip().lower()
        return "full_attention" if w in ("none", "null", "0", "") else "sliding_attention"
    def _rope_params(seg, out):
        for part in split_top(seg):
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ROPE_DROP:
                    continue
                out[ROPE_KEYS.get(k, k)] = _coerce(v)
    def parse_rope(expr):
        """支持 rope(...) / partial(rope(...), ...) / yarn(rope(...), ...)"""
        e = expr.strip()
        out: Dict[str, Any] = {}
        for _ in range(4):
            m = re.match(r"^(\w+)\s*\(", e)
            if not m or m.group(1) == "rope":
                break
            wrap = m.group(1)
            inner = e[m.end(): e.rindex(")")]
            depth, cut = 0, None
            for i, ch in enumerate(inner):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        cut = i + 1
                        break
            if cut:
                _rope_params(inner[cut:], out)
            if wrap != "partial":
                out.setdefault("rope_type", wrap)
            e = (inner[:cut] if cut else inner).strip()
        m = re.match(r"^rope\s*\(", e)
        if m:
            _rope_params(e[m.end(): e.rindex(")")], out)
        out.setdefault("rope_type", "default")
        return out
    def lookup_attr(mech, sel, attr):
        _n, conds = parse_state_key(f"{mech}[{sel}]" if sel else mech)
        for (_s, m, a, _k, _t) in layer_seq:
            if m == mech and selector_matches(conds, a):
                return _coerce(a.get(attr, ""))
        if mech in mechs:
            return _coerce(mechs[mech].assigns.get(attr, ""))
        for c in scope.children:            # head / memory / residual 这类块
            if c.name == mech:
                return _coerce(c.assigns.get(attr, ""))
        return MISSING % (mech, sel, attr)
    def resolve_field(val):
        v = val.strip()
        if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        if v in ("true", "false"):
            return v == "true"
        if v == "null":
            return None
        if len(v) >= 2 and v.startswith("[") and v.endswith("]"):
            return [_coerce(x) for x in split_top(v[1:-1])]
        if len(v) >= 2 and v.startswith("{") and v.endswith("}"):
            # 嵌套字面量，例如 quantization_config。
            # 值是 "a": b, "c": d 的形式（键必须带引号，值是 _coerce 处理）。
            out = {}
            for item in split_top(v[1:-1]):
                if ":" not in item:
                    continue
                k, val = item.split(":", 1)
                out[k.strip().strip('"')] = resolve_field(val.strip())
            return out
        if v in ("schedule", "layer_types"):
            return gen_layer_types()
        if v in ("position", "rope_parameters"):
            return gen_rope_parameters()
        if v == "rope_scaling":
            return gen_rope_scaling()
        if v == "rope_parameters_flat":
            return gen_rope_parameters_flat()
        if v == "sliding_window":
            return gen_sliding_window()
        if v in ("mlp_layer_types", "indexer_types"):
            # 同一份描述在不同 config 里叫不同名字、取不同粒度：
            #   mlp_layer_types   dense / sparse  —— 看那一层挂的是 FFN 还是 MoE
            #   indexer_types     full           —— GLM-5.3 全是 full
            _out = []
            for (_s, _m, _a, _k, _att) in layer_seq:
                if _s not in _main_stack_names:
                    continue
                if v == "indexer_types":
                    _out.append("full")
                    continue
                # **看挂上去的那个**（元组第 5 项），不是混合器本身 ——
                # `model[3..44] >> MoE` 的意思就是"这些层挂 MoE"。
                _names = _att if isinstance(_att, (list, tuple)) else [_att]
                _kind = ""
                for _n in _names:
                    _mm = mechs.get(_n)
                    if _mm is not None and (_mm.mtype or "") == "MoE":
                        _kind = "MoE"
                _out.append("sparse" if _kind == "MoE" else "dense")
            return _out
        if v == "layers_block_type":
            # Nemotron-H 的 `layers_block_type` 用的词和 HF 通用的
            # `layer_types` **不一样**：它写 mamba / moe / attention，
            # 而不是 linear_attention / full_attention。
            # 名字不同、含义一样 —— 所以是另一个生成器，不是改那个。
            _MAP = {"linear_attention": "mamba", "full_attention": "attention",
                    "sliding_attention": "attention"}
            lt = gen_layer_types() or []
            kinds = [_k for (_s, _m, _a, _k, _t) in layer_seq
                     if _s in _main_stack_names]
            out = []
            for _i, _lt in enumerate(lt):
                _m = mechs.get(layer_seq[_i][1]) if _i < len(layer_seq) else None
                _mk = (_m.mtype or "") if _m is not None else ""
                if _mk == "MoE":
                    out.append("moe")
                elif _mk in ("SSM",):
                    out.append("mamba")
                else:
                    out.append(_MAP.get(_lt, _lt))
            return out
        if v == "rope_theta":
            # 全局的 rope base —— 有些模型的 Attention 机制里没写，
            # 只在 position 里声明了。
            for _v in (pb.assigns.values() if pb else []):
                _d = parse_rope(_v)
                # position 里写的是 rope(base = N)，键叫 base
                if _d.get("base") is not None:
                    return _coerce(_d["base"])
                if _d.get("rope_theta") is not None:
                    return _coerce(_d["rope_theta"])
            return None
        if v == "initial_context_length":
            for _v in (pb.assigns.values() if pb else []):
                _d = parse_rope(_v)
                if _d.get("original_max_position_embeddings") is not None:
                    return _coerce(_d["original_max_position_embeddings"])
                if _d.get("original") is not None:
                    return _coerce(_d["original"])
            return None
        if v == "attention_other_setting":
            return gen_attention_other_setting()
        m = re.match(r"^by_layer\(\s*([\w.]+)\s*\)$", v)
        if m:
            return gen_by_layer(m.group(1))
        m = re.match(r'^per_layer_rope\(\s*([\w.]+)\s*\)$', v)
        if m:
            return gen_per_layer_rope(m.group(1))
        m = re.match(r'^join\(\s*(.+?)\s*,\s*"(.*?)"\s*\)$', v)
        if m:
            return gen_join(m.group(1), m.group(2))
        m = re.match(r"^per_layer\(\s*([\w.]+)\s*,\s*([\w.-]+)\s*\)$", v)
        if m:
            return gen_per_layer_d(m.group(1), _coerce(m.group(2)))
        m = re.match(r"^pad_cycle\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)), "cycle")
        m = re.match(r"^pad_last\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)), "last")
        m = re.match(r"^pad\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)))
        m = re.match(r"^per_layer\(\s*([\w.]+)\s*\)$", v)
        if m:
            return gen_per_layer(m.group(1))
        m = re.match(r"^indices_where\(\s*([\w.]+)\s*=\s*([\w.-]+)\s*\)$", v)
        if m:
            key, want = m.group(1), m.group(2)
            idx = []
            for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
                if key.startswith("attach."):
                    sub, val = key[7:], ""
                    for am in atts:
                        ma = mechs.get(am)
                        if ma is not None:
                            val = ma.assigns.get(sub, ma.mtype)
                            break
                else:
                    val = a.get(key, "")
                if str(val).strip() == want:
                    idx.append(li)
            return idx
        m = re.match(r"^([A-Za-z_]\w*)\[([^\]]*)\]\.(\w+)$", v)
        if m:
            return lookup_attr(m.group(1), m.group(2), m.group(3))
        m = re.match(r"^([A-Za-z_]\w*)\.(\w+)$", v)
        if m:
            return lookup_attr(m.group(1), "", m.group(2))
        if v in hp:
            return _coerce(hp[v])
        if v in scope.assigns:              # 模型级属性，如 ctx
            return _coerce(scope.assigns[v])
        # 兜底：数值 / 裸字面量。写错的名字不会被静默吞掉 ——
        # 它会在 by1verify --config 的逐字段对拍里以「值不同」暴露。
        return _coerce(v)

    return {
        'gen_layer_types': gen_layer_types,
        'name_layer_type': name_layer_type,
        'gen_rope_parameters': gen_rope_parameters,
        'gen_rope_scaling': gen_rope_scaling,
        'gen_attention_other_setting': gen_attention_other_setting,
        'gen_by_layer': gen_by_layer,
        'gen_per_layer_rope': gen_per_layer_rope,
        'gen_join': gen_join,
        'gen_pad': gen_pad,
        'gen_per_layer': gen_per_layer,
        'gen_per_layer_d': gen_per_layer_d,
        'gen_sliding_window': gen_sliding_window,
        'gen_rope_parameters_flat': gen_rope_parameters_flat,
        '_coerce': _coerce,
        '_ltype_of_attrs': _ltype_of_attrs,
        '_rope_params': _rope_params,
        'parse_rope': parse_rope,
        'lookup_attr': lookup_attr,
        'resolve_field': resolve_field,
    }
