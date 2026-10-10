#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1stacks -- **stacks：栈怎么排、每层挂什么机制。**

从 `by1check.check()` 的主体里整段搬出来的。**段体一字未改。**

搬它不需要枚举"依赖哪些名字" —— 因为主体是**命名空间形式**：
段体里写的是 `_['...']`，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
"""


def read_stacks(*, _mod, _):
    """把栈与逐层覆盖读进 `_['expansion']` / `_['overrides']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- stacks --------------------------------------------------
    _['stacks'] = _['scope'].kids("stack")
    _['alias'] = {}
    _['expansion']: Dict[str, List[Rec]] = {}
    # (栈, 层号, 机制) -> {属性: 值}
    _['overrides']: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    _['pending_ov']: List[tuple] = []
    _['total_layers'] = 0
    for _['st'] in _['stacks']:
        if _['st'].alias:
            _['alias'][_['st'].alias] = _['st'].name
        _['alias'][_['st'].name] = _['st'].name

    for _['st'] in _['stacks']:
        _['pexpr'] = _['st'].assigns.get("pattern")
        if _['pexpr'] is None:
            _['rep'].add(W, _['st'].line, "pattern", f"stack {_['st'].name} 没有 pattern")
            continue
        _['ex'] = expand_pattern(_['pexpr'], _['named'])
        if _['ex'] is None:
            _['rep'].add(E, _['st'].line, "pattern", f"stack {_['st'].name} 的 pattern 无法解析")
            continue
        _['expansion'][_['st'].name] = _['ex']
        _['total_layers'] += len(_['ex'])

        # 逐层属性覆盖：`main[3..44] : MoE.experts = [255, 266, ...]`
        # 值按被选中的层顺序分配。逐层变化的专家数、swiglu_limit 都靠它。
        for _['_k'], _['_v'], _['_ln'] in _['st'].entries:
            _['_m'] = re.match(r"^([A-Za-z_]\w*)\s*\[\s*(.*?)\s*\]$", _['_k'])
            if not _['_m']:
                _['rep'].add(W, _['_ln'], "override",
                        f"stack {_['st'].name} 里的条目 '{_['_k']}' 不是逐层覆盖"
                        f"（要写成像 main[3..44] : MoE.experts = [...]）")
                continue
            _['_sn'], _['_sel'] = _['_m'].group(1), _['_m'].group(2)
            if _['alias'].get(_['_sn'], _['_sn']) != _['st'].name:
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 指的是栈 '{_['_sn']}'，但它写在 stack {_['st'].name} 里")
                continue
            _['_mq'] = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*(.*)$", _['_v'])
            if not _['_mq']:
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 的值要写成 <机制>.<属性> = [...]")
                continue
            _['_mech'], _['_attr'], _['_vals'] = _['_mq'].group(1), _['_mq'].group(2), _['_mq'].group(3).strip()
            if not (_['_vals'].startswith("[") and _['_vals'].endswith("]")):
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 的值必须是列表")
                continue
            if _['_mech'] not in _['mechs']:
                _['rep'].add(E, _['_ln'], "override", f"逐层覆盖引用了未声明的机制 '{_['_mech']}'")
                continue
            _['_items'] = [x.strip() for x in split_top(_['_vals'][1:-1]) if x.strip()]
            # resolve_attrs 还没定义，先把原始的攒起来，等它可用再分配
            _['pending_ov'].append((_['st'].name, _['_sel'], _['_mech'], _['_attr'], _['_items'], _['_ln'], _['_k']))
        if _['st'].decl_len is not None and _['st'].decl_len != len(_['ex']):
            _['rep'].add(E, _['st'].line, "layers",
                    f"stack {_['st'].name} 声明 {_['st'].decl_len} 层，pattern 展开出 {len(_['ex'])} 层")
        elif _['st'].decl_len is not None:
            _['rep'].add(I, _['st'].line, "layers",
                    f"stack {_['st'].name} 的层数在两处声明（[{_['st'].decl_len}] 与 pattern），"
                    f"当前一致 —— 建议 [N] 只作断言")
        for _['r'] in _['ex']:
            if _['r'].name not in _['mechs']:
                if _['r'].name in _['memories']:
                    _['rep'].add(W, _['st'].line, "resolve",
                            f"{_['r'].name} 声明在 memory 块，却被 pattern 当作层机制引用")
                elif _['r'].name in _['residuals']:
                    _['rep'].add(E, _['st'].line, "resolve",
                            f"{_['r'].name} 是 residual，不能出现在 pattern 里")
                else:
                    _['rep'].add(E, _['st'].line, "resolve",
                            f"pattern 引用了未声明的机制 '{_['r'].name}'")
            else:
                _['m'] = _['mechs'][_['r'].name]
                for _['a'] in _['r'].attrs:
                    if _['a'] not in _['m'].assigns and _['a'] not in BUILTIN_ATTRS:
                        _['rep'].add(W, _['st'].line, "attr",
                                f"{_['r'].name}({_['a']} = ...) 的属性 '{_['a']}' 既未在 mech "
                                f"{_['r'].name} 声明，也不在已知属性表里")

    def _is_aux(st):
        """栈是不是辅助栈（MTP / 投机解码头之类）。
        **显式声明，不靠栈名猜。** 定义必须在使用之前 ——
        闭包在运行时才解析名字，放后面会 NameError。
        """
        _['v'] = (st.assigns.get("aux") or "").strip().lower()
        return _['v'] in ("true", "1", "yes", "on")
    _['_is_aux'] = _is_aux    # 让 stage 也取得到它

    _['_main_stack_names'] = {st.name for st in _['stacks'] if not _['_is_aux'](st)}

    # **辅助栈不算解码层。** MTP（多 token 预测）是训练时的辅助头，
    # 它在权重里、但不在 num_hidden_layers 里 —— 主干的层数才是那个数。
    # 靠栈名判断是魔法，所以让 .by1 显式写 ux = true。
    _['_main_layers'] = sum(len(_['expansion'].get(st.name, []))
                       for st in _['stacks']
                       if not _['_is_aux'](st))
    if _['nlayer_decl'] is not None and _['stacks']:
        if _['nlayer_decl'] != _['_main_layers']:
            _['rep'].add(E, _['hb'].line, "layers",
                    f"hparams.n_layer = {int(_['nlayer_decl'])}，但主栈合计 {_['_main_layers']} 层"
                    + (f"（另有辅助栈 {_['total_layers'] - _['_main_layers']} 层）"
                       if _['total_layers'] != _['_main_layers'] else ""))

