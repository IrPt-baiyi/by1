#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1state -- **按选择器把 state 声明落到具体的层上。**

从 `by1check.check()` 的主体里整段搬出来的。**段体一字未改。**

搬它不需要枚举"依赖哪些名字" —— 因为主体是**命名空间形式**：
段体里写的是 `_['...']`，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
"""


def state_sizes(*, _mod, _):
    """把状态尺寸算进 `_['state_src']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- 状态尺寸 -------------------------------------------------
    # 按「选择器」把 state 声明落到具体的层上。同一个机制的两类层状态可以不同：
    #   GQA[window = none]  : kv_cache(grows_with_seq)
    #   GQA[window != none] : kv_cache(bounded_by = window - 1)
    _['states'] = []
    _['dtype_b'] = 2  # bf16 默认
    _['em'] = _['scope'].first("emit")
    if _['em']:
        for _['a'], _['b'] in _['em'].assigns.items():
            _['m'] = re.search(r"dtype\s*=\s*(\w+)", _['b'])
            if _['m'] and _['m'].group(1) in ("fp32", "float32"):
                _['dtype_b'] = 4

    _['inst_by_mech']: Dict[str, List[Dict[str, str]]] = {}
    for (_['_s'], _['_m'], _['_a'], _['_k'], _['_t']) in _['layer_seq']:
        _['inst_by_mech'].setdefault(_['_m'], []).append(_['_a'])

    def eval_with(expr: str, attrs: Dict[str, str]):
        _['s2'] = expr
        for _['k2'], _['v2'] in attrs.items():
            _['n2'] = eval_num(_['v2'])
            if _['n2'] is not None:
                _['s2'] = re.sub(r"\b" + re.escape(_['k2']) + r"\b", _fmt(_['n2']), _['s2'])
        return eval_num(_['s2'])
    _['eval_with'] = eval_with    # 让 stage 也取得到它

    for _['nm'], _['ents'] in _['state_src'].items():
        _['blk'] = _['mechs'].get(_['nm']) or _['memories'].get(_['nm'])
        for _['key'], _['val'], _['ln'] in _['ents']:
            _['_n'], _['conds'] = parse_state_key(_['key'])
            _['sel'] = [a for a in _['inst_by_mech'].get(_['nm'], []) if selector_matches(_['conds'], a)]
            _['n'] = len(_['sel'])
            _['tag'] = _['key'] if _['conds'] else _['nm']
            _['h'] = _['heads_of'](_['blk']) if _['blk'] else None
            _['hd'] = _['head_dim_of'].get(_['nm'])
            if "recurrent" in _['val']:
                if _['h'] and "v" in _['h'] and _['hd']:
                    _['per'] = _['h']["v"] * _['hd'] * _['hd'] * _['dtype_b']
                    _['states'].append((f"{_['tag']} recurrent", f"{_['per']/2**20:,.2f} MiB/层",
                                   f"{_['n']} 层", f"{_['per']*_['n']/2**20:,.2f} MiB"))
                else:
                    _['states'].append((f"{_['tag']} recurrent", "无法推导", f"{_['n']} 层",
                                   "缺 heads(v)/head_dim"))
            elif "kv_cache" in _['val']:
                _['kv'] = None
                if _['h']:
                    _['kv'] = _['h'].get("kv", _['h'].get("kv_heads"))
                if _['kv'] is None and _['blk'] is not None:
                    _['kv'] = eval_num(_['blk'].assigns.get("kv_heads"))
                if not (_['kv'] and _['hd']):
                    _['states'].append((f"{_['tag']} kv_cache", "无法推导", f"{_['n']} 层",
                                   "缺 kv heads 或 head_dim —— 算不出来"))
                    continue
                _['per_tok'] = _['kv'] * _['hd'] * 2 * _['dtype_b']
                _['mb'] = re.search(r"bounded_by\s*=\s*([^,)]+)", _['val'])
                if _['mb']:
                    _['bounds'] = [_['eval_with'](_['mb'].group(1).strip(), a) for a in _['sel']]
                    if any(b is None for b in _['bounds']):
                        _['states'].append((f"{_['tag']} kv_cache(bounded)",
                                       f"{_['per_tok']:,.0f} B/token", f"{_['n']} 层",
                                       "有界，但上界表达式算不出来"))
                    else:
                        _['tot'] = _['per_tok'] * sum(_['bounds'])
                        _['bset'] = sorted({int(b) for b in _['bounds']})
                        _['bt'] = str(_['bset'][0]) if len(_['bset']) == 1 else str(_['bset'])
                        _['states'].append((f"{_['tag']} kv_cache(bounded)",
                                       f"{_['per_tok']:,.0f} B/token", f"{_['n']} 层",
                                       f"上界 {_['bt']} → {_['tot']/2**20:,.2f} MiB 常量"))
                else:
                    _['states'].append((f"{_['tag']} kv_cache", f"{_['per_tok']:,.0f} B/token",
                                   f"{_['n']} 层", f"{_['per_tok']*_['n']:,.0f} B/token (全模型)"))

