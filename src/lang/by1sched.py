#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1sched -- **schedule：命名子序列的多趟展开。**

从 `by1check.check()` 的主体里整段搬出来的。**段体一字未改。**

搬它不需要枚举"依赖哪些名字" —— 因为主体是**命名空间形式**：
段体里写的是 `_['...']`，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
"""


def read_schedule(*, _mod, _):
    """把命名子序列展开进 `_['named']` / `_['sb']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- schedule 命名子序列 --------------------------------------
    _['named']: Dict[str, List[Rec]] = {}
    _['sb'] = _['scope'].first("schedule")
    if _['sb']:
        # 多趟展开：命名子序列可以引用别的命名子序列，直到不再变化
        for _['_'] in range(8):
            _['before'] = {k: [str(r) for r in v] for k, v in _['named'].items()}
            for _['k'], _['v'] in _['sb'].assigns.items():
                _['ex'] = expand_pattern(_['v'], _['named'])
                if _['ex'] is not None:
                    _['named'][_['k']] = _['ex']
            if _['before'] == {k: [str(r) for r in v] for k, v in _['named'].items()}:
                break

