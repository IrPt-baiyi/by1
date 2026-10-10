#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1tens -- **张量契约实例化：把契约按等价类求值出形状。**

## 为什么只接一个参数

**这一段原来住在 `by1check.check()` 的主体里。** 搬出来时不需要枚举
"它依赖哪些名字" —— 因为主体是**命名空间形式**：段体里写的是
`_['mechs']` / `_['rep']` 这些，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
不能直接 import by1check：它反过来 import 本模块。
"""


def instantiate(*, _mod, _):
    """把契约实例化进 `_['contracts']` / `_['class_rows']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- 张量契约实例化 ------------------------------------------
    # 契约按「机制」声明；实例按「(机制, 结构属性) 等价类」求值。
    # 这正是结构属性的连带代价：head_dim / kv_tie 一变，形状与张量集合都变。
    _['STRUCTURAL'] = {"head_dim", "kv", "q", "v", "qk", "kv_heads", "kv_tie",
                  "experts", "d_ff", "hidden", "intermediate", "out_dim"}
    _['decl_struct']: Dict[str, set] = {}
    for _['nm'], _['blk'] in _['mechs'].items():
        _['raw'] = _['blk'].assigns.get("structural")
        if _['raw']:
            _['body'] = _['raw'].strip()
            if _['body'].startswith("[") and "]" in _['body']:
                _['body'] = _['body'][1 : _['body'].rfind("]")]
            _['decl_struct'][_['nm']] = set(split_top(_['body']))

    # 1) 实例枚举 -> 等价类
    def resolve_attrs(m: Blk, over: Dict[str, str]) -> Dict[str, str]:
        _['a']: Dict[str, str] = {}
        for _['k'], _['v'] in m.assigns.items():
            if _['k'] not in ("heads", "structural"):
                _['a'][_['k']] = _['v']
        for _['k'], _['v'] in (_['heads_of'](m) or {}).items():
            _['a'][_['k']] = _fmt(_['v'])
        _['a'].update(over)
        return _['a']
    _['resolve_attrs'] = resolve_attrs    # 让 stage 也取得到它

    def key_of(nm: str, attrs: Dict[str, str]) -> tuple:
        _['sk'] = _['STRUCTURAL'] | _['decl_struct'].get(nm, set())
        return tuple(sorted((k, v) for k, v in attrs.items() if k in _['sk']))
    _['key_of'] = key_of    # 让 stage 也取得到它

    # 哪些层挂了哪些「通道混合器」—— 选择器（[:] / [0] / [1..39] / [step 4]）生效
    _['attach_rules']: List[tuple] = []
    for _['c'] in _['scope'].children:
        for _['sel'], _['op'], _['target'], _['ln'] in _['c'].attaches:
            _['base'] = _['target'].split("(")[0].strip()
            if _['base'] not in _['mechs'] or _['op'] != ">>":
                continue
            _['sk'], _['inner'] = split_selector(_['sel'])
            _['attach_rules'].append((_['alias'].get(_['sk'], _['sk']), make_layer_pred(_['inner']), _['base']))

    # **挂在栈上的机制也要拿到那个栈的宽度。**
    #
    # `by1stacks` 建的映射只走了 `pattern` 里的机制 —— 而 `>>` 挂上去的
    # 不在 pattern 里 ✗。实测：Gemma 的 `vision[:] >> ViTMlp`，
    # `ViT` 拿到了 1152 而 `ViTMlp` 拿到的是模型的 5376 ✗ ——
    # 于是它的 `gate_proj` 算出 (4304, 5376)，而官方是 (4304, 1152)。
    #
    # 这里正好有栈名和挂载目标，补上就是。
    for _['_sn'], _['_pred'], _['_base'] in _['attach_rules']:
        _['_dm'] = (_['stack_d_model'] or {}).get(_['_sn'])
        if _['_dm']:
            _['mech_d_model'].setdefault(_['_base'], _['_dm'])

    def attached_at(stack_name: str, li: int, attrs: Dict[str, str]) -> List[str]:
        return [b for (sn, pred, b) in _['attach_rules']
                if sn == stack_name and pred(li, attrs)]
    _['attached_at'] = attached_at    # 让 stage 也取得到它

    # 逐层覆盖的分配 —— 必须等 resolve_attrs 可用（它选择哪些层要按属性判断）
    for _['_sn'], _['_sel'], _['_mech'], _['_attr'], _['_items'], _['_ln'], _['_k'] in _['pending_ov']:
        _['_ex'] = _['expansion'].get(_['_sn'], [])
        _['_pred'] = make_layer_pred(_['_sel'])
        _['_hit'] = [i for i, r in enumerate(_['_ex'])
                if _['_pred'](i, _['resolve_attrs'](_['mechs'][r.name], r.attrs)
                         if r.name in _['mechs'] else {})]
        if len(_['_hit']) != len(_['_items']):
            _['rep'].add(E, _['_ln'], "override",
                    f"逐层覆盖 '{_['_k']}' 选中 {len(_['_hit'])} 层，但给了 {len(_['_items'])} 个值")
            continue
        for _['_i'], _['_val'] in zip(_['_hit'], _['_items']):
            _['overrides'].setdefault((_['_sn'], _['_i'], _['_mech']), {})[_['_attr']] = _['_val']

    _['classes']: Dict[str, Dict[tuple, dict]] = {}

    _['layer_seq']: List[tuple] = []
    for _['st'] in _['stacks']:
        for _['li'], _['r'] in enumerate(_['expansion'].get(_['st'].name, [])):
            _['m'] = _['mechs'].get(_['r'].name)
            if _['m'] is None:
                continue
            _['attrs'] = _['resolve_attrs'](_['m'], _['r'].attrs)
            _['_ov'] = _['overrides'].get((_['st'].name, _['li'], _['r'].name))
            if _['_ov']:
                _['attrs'].update(_['_ov'])
            _['key'] = _['key_of'](_['r'].name, _['attrs'])
            _['layer_seq'].append((_['st'].name, _['r'].name, dict(_['attrs']), _['key'],
                              _['attached_at'](_['st'].name, _['li'], _['attrs'])))
            _['d'] = _['classes'].setdefault(_['r'].name, {})
            _['ent'] = _['d'].setdefault(_['key'], {"count": 0, "attrs": {}})
            _['ent']["count"] += 1
            _['ent']["attrs"].update(_['attrs'])

    # 只被挂载、不作为层机制出现的（如 FFN）：按每个挂载规则建类，层数按选择器数
    for (_['sn'], _['pred'], _['base']) in _['attach_rules']:
        if _['base'] in _['classes'] or _['base'] not in _['mechs']:
            continue
        _['buckets'] = {}
        for _['li'], _['r'] in enumerate(_['expansion'].get(_['sn'], [])):
            _['a'] = _['resolve_attrs'](_['mechs'][_['r'].name], _['r'].attrs) if _['r'].name in _['mechs'] else {}
            if not _['pred'](_['li'], _['a']):
                continue
            _['at'] = dict(_['resolve_attrs'](_['mechs'][_['base']], {}))
            _['at'].update(_['overrides'].get((_['sn'], _['li'], _['base']), {}))
            _['kk'] = _['key_of'](_['base'], _['at'])
            _['buckets'].setdefault(_['kk'], {"count": 0, "attrs": _['at']})
            _['buckets'][_['kk']]["count"] += 1
        _['classes'].setdefault(_['base'], {}).update(_['buckets'])

    # 2) 读契约
    _['tb'] = _['scope'].first("tensors")
    # 文件里既没有 tensors 也没有 emit，说明它是在**设计一个新模型**，
    # 不是在对拍一个已有产物。张量契约那类警告对它是噪音。
    _['wants_artifacts'] = (_['tb'] is not None) or (_['scope'].first("emit") is not None)
    _['contracts']: Dict[str, list] = {}
    if _['tb']:
        for _['ch'] in _['tb'].children:
            _['h'] = _['ch'].head.strip()
            if re.match(r"^layer\s*\(", _['h']):
                _['rep'].add(E, _['ch'].line, "tensors",
                        f"'{_['h']}' 是无作用域写法：会给所有层发同一套张量名。"
                        f"契约必须按机制声明（见 pattern-design §2 规则 3）")
                continue
            _['base'] = _['h'].split(".")[0].strip()
            # **按栈的层内张量：`layer <栈名> { ... }`**
            #
            # 为什么需要它：`layer` 那一块是**模型级**的 —— 所有栈共用
            # 同一套层内张量名。Gemma 的视觉塔恰好和文本用同一组名字
            # （`input_layernorm` 那一套），所以没暴露；而 clef / Qwen3.8 /
            # Qwen3.6 的视觉块用的是 `norm1` / `norm2` —— 写进 `layer`
            # 会**给文本层也发一份**，而文本层没有这两个张量。
            #
            # 和 `name_<栈名>`（物理前缀按栈各写一份）是同一个理由，
            # 只是那一边管**前缀**，这一边管**有哪些张量**。
            _['_sm'] = re.fullmatch(r"layer\s+([A-Za-z_]\w*)", _['h'])
            if _['_sm']:
                _['_sn2'] = _['_sm'].group(1)
                if _['_sn2'] not in {s.name for s in _['stacks']}:
                    _['rep'].add(E, _['ch'].line, "tensors",
                            f"tensors 的 'layer {_['_sn2']}' 引用了一个不存在的栈"
                            f"（已声明的栈："
                            f"{', '.join(s.name for s in _['stacks'])}）")
                    continue
                _['base'] = "layer:" + _['_sn2']
            if _['h'] in ("layer", "global"):
                pass                    # 层级 / 全局：不属于任何机制
            elif _['_sm']:
                pass                    # 按栈的层级块，上面已经验过栈名
            elif _['base'] not in _['mechs'] and _['base'] not in _['memories'] and _['base'] not in _['residuals']:
                _['rep'].add(E, _['ch'].line, "tensors", f"tensors 的作用域 '{_['base']}' 未声明")
                continue
            _['lst'] = []
            for _['key'], _['val'], _['ln'] in _['ch'].entries:
                _['mg'] = re.search(r"\bunless\s+([A-Za-z_]\w*)", _['val'])
                _['guard'] = _['mg'].group(1) if _['mg'] else None
                _['pe'] = bool(re.search(r"\bper_expert\b", _['val']))
                _['shp'] = re.sub(r"\bunless\s+[A-Za-z_]\w*", "", _['val'])
                _['shp'] = re.sub(r"\bper_expert\b", "", _['shp']).strip()
                _['lst'].append((_['key'], _['shp'], _['guard'], _['ln'], _['pe']))
            if not _['lst']:
                _['rep'].add(W, _['ch'].line, "tensors",
                        f"tensors 的 '{_['h']}' 契约是空的 —— 没有声明任何逻辑张量")
            # ---- 同名契约块出现两次 -> **报** --------------------------
            #
            # 原来这里是无条件赋值 ✓ —— 于是同一个 `tensors` 里出现两个
            # `KDA { }` 时，**后面的直接覆盖前面的、一声不响** ✗。
            #
            # 后果不是"多余"，是**前面那个变成死的** ✗：谁往它里面加东西
            # 都等于没加 ✓。实测踩过一次 —— 往死块里加了 5 个张量、
            # 跑了三遍检查数都没动 ✓，于是得出了"检查漏验了这 510 个张量"
            # 的错误结论 ✗。**根因是这一行。**
            #
            # 静默覆盖是这个仓库专门在猎的一族：看起来写了、实际被别的
            # 盖掉了、而没人说 ✓。（`vision ViT { }` 那段是"没人读" ✓，
            # 这一条是"读了但被盖" ✓ —— 亲戚。）
            if _['base'] in _['contracts']:
                _['rep'].add(E, _['ch'].line, "tensors",
                        f"契约块 '{_['base']}' 出现了两次 —— **后一个会直接"
                        f"覆盖前一个**，而前一个从此变成死的（往它里面加东西"
                        f"等于没加）✗。删掉一个，或者把它们合并。")
            _['contracts'][_['base']] = _['lst']

    def _flat_rows(entry_list):
        """层级 / 全局张量：形状只用 hparams 求值（没有逐层属性）。"""
        _['sym0']: Dict[str, float] = {}
        for _['k2'], _['v2'] in _['hp'].items():
            _['n2'] = eval_num(_['v2'])
            if _['n2'] is not None:
                _['sym0'][_['k2']] = _['n2']
        _['res'] = []
        for (_['lname'], _['shp'], _['guard'], _['ln'], _['pe']) in entry_list:
            _['body'] = _['shp'].strip()
            if _['body'].startswith("(") and ")" in _['body']:
                _['body'] = _['body'][1: _['body'].rfind(")")]
            _['comps'] = []
            for _['c'] in split_top(_['body']):
                _['ids'] = set(re.findall(r"[A-Za-z_]\w*", _['c']))
                _['s2'] = _['c']
                for _['i'] in sorted(_['ids'], key=len, reverse=True):
                    if _['i'] in _['sym0']:
                        _['s2'] = re.sub(r"\b" + re.escape(_['i']) + r"\b", _fmt(_['sym0'][_['i']]), _['s2'])
                _['v'] = eval_num(_['s2'])
                _['comps'].append(_fmt(_['v']) if _['v'] is not None else _['s2'].replace(" ", ""))
            # `--` 是**声明为不该存在**（权重共享时没有 lm_head），
            # 它不是形状，不能被重新包成 `(...)` —— 原样透传，
            # 让 by1verify 去检查"确实不存在"。
            if _['shp'].strip() == "--":
                _['res'].append((_['lname'], "--", _['guard'], _['pe']))
                continue
            # guard 留着 —— layer 作用域用它做「只有挂了某机制才有这个张量」
            _['res'].append((_['lname'], "(" + ", ".join(_['comps']) + ")", _['guard'], _['pe']))
        return _['res']
    _['_flat_rows'] = _flat_rows    # 让 stage 也取得到它

    _['layer_rows'] = _['_flat_rows'](_['contracts'].get("layer", []))
    _['global_rows'] = _['_flat_rows'](_['contracts'].get("global", []))

    # **逐层的张量也要按栈算宽度。** `layer_rows` 上面那一行只算了一次、
    # 用的是模型的 `d_model` —— 而视觉栈宽 1152 ✗。实测：27 层 x 4 个 norm
    # 全部"形状不符"（声明 `(5376,)` 而官方是 `(1152,)`）。
    #
    # class rows 没有这个问题，因为它们是**逐实例**算的 ✓。
    #
    # ⚠ **要换的是 `_['hp']`，不是 `_['d_model']`。** `_flat_rows` 里写着
    # 「形状只用 hparams 求值」—— 它读的就是 `_['hp']`（第 171 行）✗。
    # 我第一版换的是 `_['d_model']` ✓，而那个名字**没有任何人读** ✗ ——
    # 于是 `layer_rows_by_stack['vision']` 里仍然是 `(5376)`，
    # 而表现是"改了跟没改一样" ✓。
    _['layer_rows_by_stack']: Dict[str, list] = {}
    for _['_sn'], _['_dm'] in (_['stack_d_model'] or {}).items():
        _['_sav_hp'] = _['hp']
        _['_hp2'] = dict(_['hp'])
        _['_hp2']["d_model"] = _fmt(_['_dm'])
        _['hp'] = _['_hp2']
        # **这一栈有没有自己的一套层内张量。** 有就用它，没有就用
        # 模型级那份 `layer` —— 两处都要按这一栈的宽度求值。
        _['layer_rows_by_stack'][_['_sn']] = _['_flat_rows'](
            _['contracts'].get("layer:" + _['_sn'])
            or _['contracts'].get("layer", []))
        _['hp'] = _['_sav_hp']

    # 3) 逐类实例化
    _['trows'] = []
    _['class_rows']: Dict[tuple, list] = {}
    for _['nm'] in sorted(_['classes']):
        _['cls'] = _['classes'][_['nm']]
        if _['nm'] not in _['contracts']:
            if _['wants_artifacts'] and _['mechs'][_['nm']].mtype in TOKEN_MIXER:
                _['rep'].add(W, _['mechs'][_['nm']].line, "tensors", f"mech {_['nm']} 无张量契约")
            continue
        for _['key'], _['ent'] in sorted(_['cls'].items(), key=lambda kv: -kv[1]["count"]):
            _['attrs'] = _['ent']["attrs"]
            _['label'] = ", ".join(f"{k}={v}" for k, v in _['key']) or "(无结构属性)"
            _['sym']: Dict[str, float] = {}
            for _['k'], _['v'] in _['attrs'].items():
                _['n'] = eval_num(_['v'])
                if _['n'] is not None:
                    _['sym'][_['k']] = _['n']
            for _['k'], _['v'] in _['hp'].items():
                _['n'] = eval_num(_['v'])
                if _['n'] is not None:
                    _['sym'][_['k']] = _['n']

            # **栈自己的宽度覆盖模型的。** 视觉塔宽 1152，而 `d_model`
            # 是 5376 —— 这个栈里的张量要用前者（见 `by1stacks` 里那段）。
            # 没声明宽度的栈拿到的是 `None`，行为**一个字都不变** ✓。
            _['_dm'] = (_['mech_d_model'] or {}).get(_['nm'])
            if _['_dm']:
                _['sym']["d_model"] = _['_dm']

            # 宽度不变式。注意：不能靠「比值是否好看」判定对错 —— Gemma 4 的
            # 注意力宽度本来就不等于 d_model。所以宽度变化必须显式声明。
            _['qq'], _['vv'], _['hdv'] = _['sym'].get("q"), _['sym'].get("v"), _['sym'].get("head_dim")
            _['w'] = (_['vv'] * _['hdv']) if (_['vv'] and _['hdv']) else ((_['qq'] * _['hdv']) if (_['qq'] and _['hdv']) else None)
            _['od'] = eval_num(_['attrs'].get("out_dim"))
            # **有效宽度**：栈自己声明了就用栈的，否则用模型的。
            # 视觉塔宽 1152、模型 d_model 5376 —— 这条不变式要拿前者比。
            _['_dmv'] = _['sym'].get("d_model") or _['d_model']
            if _['w'] and _['od']:
                if abs(_['w'] - _['od']) > 1e-9:
                    _['rep'].add(E, _['mechs'][_['nm']].line, "shape",
                            f"{_['nm']}({_['label']}): 声明 out_dim = {_fmt(_['od'])}，"
                            f"但 q x head_dim = {_fmt(_['w'])}")
            elif _['w'] and _['_dmv'] and abs(_['w'] - _['_dmv']) > 1e-9:
                _['rep'].add(W, _['mechs'][_['nm']].line, "shape",
                        f"{_['nm']}({_['label']}): q x head_dim = {_fmt(_['w'])} != d_model = "
                        f"{_fmt(_['_dmv'])}（比值 {_['w']/_['_dmv']:.3f}）—— 若这是有意的宽度变化，"
                        f"声明 out_dim = {_fmt(_['w'])}")
            _['rows'] = []
            for _['lname'], _['shp'], _['guard'], _['ln'], _['pe'] in _['contracts'][_['nm']]:
                if _['guard'] is not None:
                    _['gs'] = _['attrs'].get(_['guard'])
                    if _['gs'] is None:
                        _['rep'].add(W, _['ln'], "tensors",
                                f"{_['nm']}.{_['lname']}: unless {_['guard']} 引用了未声明的属性")
                        continue
                    if str(_['gs']).strip().lower() in ("true", "1", "yes", "on"):
                        _['rows'].append((_['lname'], "--", f"由 unless {_['guard']} 抑制", _['pe']))
                        continue
                _['body'] = _['shp'].strip()
                if _['body'].startswith("(") and ")" in _['body']:
                    _['body'] = _['body'][1 : _['body'].rfind(")")]
                _['out'], _['miss'] = [], []
                for _['c'] in split_top(_['body']):
                    _['ids'] = set(re.findall(r"[A-Za-z_]\w*", _['c']))
                    _['bad'] = sorted(i for i in _['ids'] if i not in _['sym'])
                    if _['bad']:
                        _['miss'].extend(_['bad'])
                    _['s2'] = _['c']
                    for _['i'] in sorted(_['ids'], key=len, reverse=True):
                        if _['i'] in _['sym']:
                            _['s2'] = re.sub(r"\b" + re.escape(_['i']) + r"\b", _fmt(_['sym'][_['i']]), _['s2'])
                    _['v'] = eval_num(_['s2'])
                    _['out'].append(_fmt(_['v']) if _['v'] is not None else _['s2'].replace(" ", ""))
                _['note'] = ("缺 " + ", ".join(sorted(set(_['miss'])))) if _['miss'] else ""
                _['rows'].append((_['lname'], "(" + ", ".join(_['out']) + ")", _['note'], _['pe']))
            _['label'] = ", ".join(f"{k}={v}" for k, v in _['key']) or "(无结构属性)"
            _['class_rows'][(_['nm'], _['key'])] = _['rows']
            _['trows'].append((_['nm'], _['label'], _['ent']["count"], _['rows']))

    for _['nm'], _['lst'] in _['contracts'].items():
        if _['nm'] in ("layer", "global") or _['nm'].startswith("layer:"):
            continue
        if _['nm'] not in _['classes'] and _['nm'] not in _['attached']:
            _['rep'].add(W, _['lst'][0][3] if _['lst'] else 0, "tensors",
                    f"tensors 声明了 {_['nm']} 的契约，但 pattern 里没有它的实例")

