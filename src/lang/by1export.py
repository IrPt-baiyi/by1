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

def make(*, _main_stack_names, layer_seq, mechs, named, overrides, pb, rep, _coerce, _ltype_of_attrs, parse_rope, resolve_field, W):
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
    }
