#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1hp -- **hparams 块：模型有多宽、多少层、词表多大。**

从 `by1check.check()` 的主体里整段搬出来的。**段体一字未改。**

搬它不需要枚举"依赖哪些名字" —— 因为主体是**命名空间形式**：
段体里写的是 `_['...']`，`_` 是唯一的外部依赖。

（前面有 21 次尝试失败在"枚举名字"上：Python 的函数局部名在编译期
定死，清单要对就得有完整的数据流分析。）

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
"""


def read_hparams(*, _mod, _):
    """把 hparams 读进 `_['hp']` 等。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- hparams -------------------------------------------------
    _['hp']: Dict[str, str] = {}
    _['nlayer_decl'] = None
    _['hb'] = _['scope'].first("hparams")
    if _['hb']:
        _['hp'] = dict(_['hb'].assigns)
        _['nlayer_decl'] = eval_num(_['hp'].get("n_layer"))
    _['d_model'] = eval_num(_['hp'].get("d_model"))
    _['vocab'] = eval_num(_['hp'].get("vocab"))

