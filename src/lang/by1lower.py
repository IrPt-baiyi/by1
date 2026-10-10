#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1lower -- **emit lowering：把 emit 块降成 `字段名 -> 规则`。**

## 为什么只接一个参数

**这一段原来住在 `by1check.check()` 的主体里。** 搬出来时不需要枚举
"它依赖哪些名字" —— 因为主体是**命名空间形式**：段体里写的是
`_['mechs']` / `_['rep']` 这些，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
不能直接 import by1check：它反过来 import 本模块。
"""


def emit_rules(*, _mod, _):
    """把 emit 块降成 `_['emit_rules']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- emit lowering 规则 --------------------------------------
    # 物理命名在后端里声明，张量契约保持纯逻辑 —— 这是「一份描述两个后端」
    # 能成立的前提。检查器负责校验模板与映射的引用是否合法。
    _['KNOWN_PH'] = {"i", "local_i", "global_i", "stack", "mech",
                "logical", "scope", "physical", "expert"}
    _['all_logical'] = {ln for lst in _['contracts'].values() for (ln, _s, _g, _l, _p) in lst}
    _['emit_rules']: Dict[str, dict] = {}
    _['eb'] = _['scope'].first("emit")
    if _['eb']:
        for _['be'] in _['eb'].children:
            _['key'] = _['be'].head.split("->")[-1].strip() if "->" in _['be'].head else _['be'].head.strip()
            _['rule'] = {"name": _['be'].assigns.get("name", "").strip().strip('"'),
                    "expert_name": _['be'].assigns.get("expert_name", "").strip().strip('"'),
                    "global_name": _['be'].assigns.get("global_name", "").strip().strip('"'),
                    "scope": {}, "rename": {}, "fields": {},
                    "quant": {}, "line": _['be'].line, "head": _['be'].head.strip()}
            # **按栈各写一份名字模板**：`name_<栈名>`。
            # 一个 checkpoint 里可以有多个栈、各套前缀
            # （Qwen3.5 主干是 model.language_model.layers.{i}.…，
            #   MTP 那层是 mtp.layers.{i}.…），一个模板装不下。
            # render_name 里 name_<栈名> 优先、name 兜底。
            for _['_k'], _['_v'] in _['be'].assigns.items():
                if (_['_k'].startswith("name_") or _['_k'].startswith("expert_name_")) \
                        and _['_k'] not in ("name_", "expert_name_"):
                    _['rule'][_['_k']] = _['_v'].strip().strip('"')
            for _['sub'] in _['be'].children:
                _['h'] = _['sub'].head.strip()
                if _['h'] == "quant":
                    for _['k'], _['v'] in _['sub'].assigns.items():
                        _['rule']["quant"][_['k']] = _['v'].strip().strip('"')
                    for _['sub2'] in _['sub'].children:
                        if _['sub2'].head.strip() == "fuse":
                            _['rule']["quant"]["fuse"] = dict(_['sub2'].assigns)
                    continue
                _['tgt'] = {"scope": _['rule']["scope"], "rename": _['rule']["rename"],
                       "field": _['rule']["fields"], "fields": _['rule']["fields"]}.get(_['h'])
                if _['tgt'] is None:
                    continue
                for _['k'], _['v'] in _['sub'].assigns.items():
                    _['tgt'][_['k']] = _['v'].strip().strip('"')
            _['emit_rules'][_['key']] = _['rule']

            if not _['rule']["name"]:
                if _['rule']["fields"]:
                    continue
                _['rep'].add(W, _['rule']["line"], "emit",
                        f"emit {_['key']} 既没有 name 模板也没有 field 映射")
                continue
            for _['ph'] in re.findall(r"\{(\w+)\}",
                                 _['rule']["name"] + _['rule']["expert_name"] + _['rule']["global_name"]):
                if _['ph'] not in _['KNOWN_PH']:
                    _['rep'].add(E, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 name 模板含未知占位符 {{{_['ph']}}}")
            for _['m'] in _['rule']["scope"]:
                if _['m'] not in ("layer", "global") and _['m'] not in _['mechs']:
                    _['rep'].add(E, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 scope 引用了未声明的机制 '{_['m']}'")
            # quant 的 fuse 名（如 gate_up_proj）和它点名的张量，只在导出时才存在，
            # 不是契约里的逻辑张量 —— 不该被当成拼写错误
            _['_qnames'] = set(re.findall(
                r"[A-Za-z_][\w.]*", (_['rule'].get("quant") or {}).get("tensor", "")))
            for _['_n'], _['_v'] in ((_['rule'].get("quant") or {}).get("fuse") or {}).items():
                _['_qnames'].add(_['_n'])
                _['_qnames'] |= set(re.findall(r"[A-Za-z_][\w.]*", _['_v']))
            for _['ln'] in _['rule']["rename"]:
                if _['ln'] in _['_qnames']:
                    continue
                if _['all_logical'] and _['ln'] not in _['all_logical']:
                    _['rep'].add(W, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 rename 键 '{_['ln']}' 不是任何契约里的逻辑张量")
            _['used_logical'] = set(re.findall(r"\{logical\}", _['rule']["name"]))
            # 真不变式：同一个等价类里，不同的逻辑张量不能生成同一个物理名
            for (_['mn'], _['_mk']), _['rows'] in _['class_rows'].items():
                if _['rule']["scope"] and _['mn'] not in _['rule']["scope"]:
                    continue
                _['seen_nm']: Dict[str, str] = {}
                for (_['ln2'], _['_sh'], _['_nt'], _['_pe']) in _['rows']:
                    _['nm2'] = render_name(_['rule'], 0, "<stack>", _['mn'], _['ln2'])
                    if _['nm2'] in _['seen_nm'] and _['seen_nm'][_['nm2']] != _['ln2']:
                        _['rep'].add(E, _['rule']["line"], "emit",
                                f"emit {_['key']}: {_['mn']} 的 '{_['ln2']}' 与 '{_['seen_nm'][_['nm2']]}' "
                                f"生成同一个物理名 '{_['nm2']}'")
                    _['seen_nm'][_['nm2']] = _['ln2']





    _['ROPE_KEYS'] = {"base": "rope_theta", "type": "rope_type",
                 "partial": "partial_rotary_factor",
                 "ratio": "partial_rotary_factor",
                 "original": "original_max_position_embeddings"}
    _['ROPE_DROP'] = {"pairing"}          # 配对约定是 codegen 的事，不进 config
















