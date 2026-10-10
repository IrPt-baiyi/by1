#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1blocks -- **读 `.by1` 里那些"块"，把不一致报出来。**

九个块的一致性检查：双真相源(ctx vs yarn) · attach 目标 · state 块 ·
position 块键 · optimizer · interop 栈名 · 机制形状 · mrope 不变式 ·
memory 表。它们和"导出成 config"是两套词汇表（那套在 `by1export.py`），
和张量契约也不同：契约算形状，这一段**只看声明之间自相不相容**。

## 为什么只接一个参数

**这一段曾经搬失败过 21 次。** 原因不是它难，是当时的 `check()` 用的是
**局部名字** —— 搬出去就要枚举"它依赖哪些"，而那需要完整的数据流分析
（作用域 + 确定赋值 + 分支合并），集合运算逼近不了它。

现在 `check()` 的主体是命名空间形式，段体里写的是 `_['mechs']` /
`_['rep']` 这些 —— **一个名字都不用列**。

`_mod` 收调用方的 `globals()`，补齐本模块没有的模块级名字（只加不改）。
不能直接 import by1check：它反过来 import 本模块。
"""


def check_blocks(*, _mod, _):
    """跑那九个块检查。**改的是 `_` 本身**，所以没有返回值。"""
    _g = globals()
    for _k, _v in _mod.items():
        if not _k.startswith('__') and _k not in _g:
            _g[_k] = _v

    # ---- 双真相源: ctx vs yarn ------------------------------------
    _['ctx'] = eval_num(_['scope'].assigns.get("ctx"))
    _['pb'] = _['scope'].first("position")
    _['yarn'] = None
    if _['pb']:
        for _['k'], _['v'] in _['pb'].assigns.items():
            if "yarn" in _['v']:
                _['m'] = re.search(r"yarn\s*=\s*([\d.]+)", _['v'])
                if _['m']:
                    _['yarn'] = to_num(_['m'].group(1))
    if _['ctx'] and _['yarn']:
        _['rep'].add(W, _['scope'].line, "twotruth",
                f"ctx = {int(_['ctx'])} 与 position.yarn = {_['yarn']} 并存："
                f"哪个是原生上下文、哪个是外推上限没有区分（外推后 = {int(_['ctx']*_['yarn'])}）")

    # ---- attach 目标 ---------------------------------------------
    _['attached'] = set()
    for _['c'] in _['scope'].children:
        for _['sel'], _['op'], _['target'], _['ln'] in _['c'].attaches:
            _['base'] = _['target'].split("(")[0].strip()
            _['attached'].add(_['base'])
            if _['base'] not in _['mechs'] and _['base'] not in _['residuals']:
                _['rep'].add(E, _['ln'], "resolve", f"'{_['sel']} {_['op']} {_['target']}' 的目标 '{_['base']}' 未声明")

    # ---- state 块 ------------------------------------------------
    _['state_src']: Dict[str, List[Tuple[str, str, int]]] = {}
    _['stb'] = _['scope'].first("state")
    if _['stb']:
        for _['key'], _['val'], _['ln'] in _['stb'].entries:
            _['hits'] = [n for n in list(_['mechs']) + list(_['memories'])
                    if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", _['key'])]
            if not _['hits']:
                _['rep'].add(E, _['ln'], "resolve",
                        f"state 条目 '{_['key']}' 解析不到任何已声明的机制"
                        f"（通配符条目也必须能指名机制）")
                continue
            _['nm'] = _['hits'][0]
            if _['nm'] in _['memories']:
                _['rep'].add(W, _['ln'], "state-vs-param",
                        f"state 里的 '{_['nm']}' 声明在 memory 块。查表是"
                        f"跨序列共享的固定权重，不是 per-seq 状态")
            _['state_src'].setdefault(_['nm'], []).append((_['key'], _['val'], _['ln']))

    # ---- position 块键 -------------------------------------------
    _['MODAL'] = {"default", "text", "vision", "audio", "video"}
    if _['pb']:
        for _['k'] in _['pb'].assigns:
            if _['k'] in _['MODAL']:
                continue
            if _['k'] not in _['mechs'] and _['k'] not in _['named']:
                _['rep'].add(W, _['pb'].line, "resolve",
                        f"position 的键 '{_['k']}' 不是机制名、模态名，也不是命名子序列")

    # ---- optimizer except(...) -----------------------------------
    for _['c'] in _['scope'].children:
        if _['c'].kind != "optimizer":
            continue
        for _['key'], _['val'], _['ln'] in _['c'].entries:
            _['m'] = re.search(r"except\s*\(([^)]*)\)", _['key'])
            if not _['m']:
                continue
            _['known'] = set(_['mechs']) | set(_['memories']) | set(_['residuals'])
            _['known'] |= {ch.name for ch in _['scope'].kids("head")}
            for _['sym'] in split_top(_['m'].group(1)):
                if _['sym'] not in _['known']:
                    _['rep'].add(E, _['ln'], "resolve",
                            f"optimizer except(...) 引用了未声明的符号 '{_['sym']}'")

    # ---- interop 栈名 --------------------------------------------
    _['ib'] = _['scope'].first("interop")
    if _['ib']:
        _['stack_names'] = set(_['alias'])
        for _['s'], _['ln'] in _['ib'].edges:
            _['segs'] = [x.strip() for x in _['s'].split("->") if x.strip()]
            for _['seg'] in (_['segs'][:1] + _['segs'][-1:]):
                _['m'] = re.match(r"([A-Za-z_]\w*)", _['seg'])
                if _['m'] and _['m'].group(1) not in _['stack_names']:
                    _['rep'].add(E, _['ln'], "interop",
                            f"interop 的 '{_['seg']}' 里 '{_['m'].group(1)}' 不是已声明的 stack")

    # ---- 机制形状 ------------------------------------------------
    def heads_of(blk: Blk) -> Optional[Dict[str, float]]:
        _['raw'] = blk.assigns.get("heads")
        if not _['raw']:
            return None
        _['body'] = _['raw'].strip()
        if _['body'].startswith("{"):
            _['body'] = _['body'][1 : _['body'].rfind("}")] if "}" in _['body'] else _['body'][1:]
        _['out'] = {}
        for _['part'] in split_top(_['body']):
            _['i'] = _['part'].find("=")
            if _['i'] > 0:
                _['v'] = eval_num(_['part'][_['i'] + 1 :])
                if _['v'] is not None:
                    _['out'][_['part'][: _['i']].strip()] = _['v']
            else:
                # **裸数字也得认。** `heads = 64` 是说"头数是 64"，
                # 而这一段原来只认 `k = v` 那种写法 —— 于是裸数字
                # **被静默丢掉** ✗：`heads_of` 返回 None，
                # 属性表里既没有 `heads` 也没有展开出来的 `q`/`kv`。
                #
                # 它是怎么暴露的：给 `SSM` 写 `one_mech` 时，
                # Nemotron 的 `mech Mamba : SSM { heads = 64 ... }`
                # 报"缺少 heads" —— 而声明里明明写着 ✗。
                _['v'] = eval_num(_['part'])
                if _['v'] is not None:
                    _['out']["heads"] = _['v']
        return _['out'] or None
    _['heads_of'] = heads_of    # 让 stage 也取得到它

    _['head_dim_of']: Dict[str, float] = {}
    for _['nm'], _['blk'] in _['mechs'].items():
        _['h'] = _['heads_of'](_['blk'])
        _['hd'] = None
        if _['h'] and "head_dim" in _['h']:
            _['hd'] = _['h']["head_dim"]
        else:
            _['hd'] = eval_num(_['blk'].assigns.get("head_dim"))
        if _['hd']:
            _['head_dim_of'][_['nm']] = _['hd']

        if _['h'] is None:
            if _['blk'].mtype in TOKEN_MIXER and _['blk'].mtype != "Vision":
                _['rep'].add(W, _['blk'].line, "shape",
                        f"mech {_['nm']} ({_['blk'].mtype}) 未声明 heads/head_dim —— "
                        f"依赖它的 KV cache 与状态尺寸都无法推导")
            continue
        if "q" not in _['h'] and "v" not in _['h']:
            _['rep'].add(W, _['blk'].line, "shape",
                    f"mech {_['nm']} 的 heads 缺 q/v —— 宽度与投影形状都推不出来")
            continue

    # ---- mrope 不变式 --------------------------------------------
    if _['pb']:
        for _['k'], _['v'] in _['pb'].assigns.items():
            _['m'] = re.search(r"mrope\s*=\s*\[([^\]]*)\]", _['v'])
            if not _['m']:
                continue
            _['secs'] = [x for x in (to_num(p) for p in _['m'].group(1).split(",")) if x]
            _['ssum'] = sum(_['secs'])
            _['ref'] = _['head_dim_of'].get("QSA") or _['head_dim_of'].get("GQA")
            if _['ref'] is None and _['head_dim_of']:
                _['ref'] = max(_['head_dim_of'].values())
            if _['ref'] is None:
                _['rep'].add(W, _['pb'].line, "position",
                        f"mrope 段和 = {int(_['ssum'])}，但没有机制声明 head_dim，"
                        f"无法校验（隐含 head_dim = {int(_['ssum']*2)}）")
            else:
                _['want'] = _['ref'] / 2
                if abs(_['want'] - _['ssum']) > 1e-9:
                    _['rep'].add(E, _['pb'].line, "position",
                            f"mrope 段和 = {int(_['ssum'])}，但按 head_dim = {int(_['ref'])} "
                            f"应为 {int(_['want'])}（隐含 head_dim = {int(_['ssum']*2)}）")

    # ---- memory 表 vs params -------------------------------------
    _['decl_params']: Dict[str, float] = {}
    for _['nm'], _['blk'] in _['memories'].items():
        _['p'] = eval_num(_['blk'].assigns.get("params"))
        if _['p']:
            _['decl_params'][_['nm']] = _['p']
        _['ents'] = leading_num(_['blk'].assigns.get("table"))
        if _['p'] and _['ents']:
            _['per'] = _['p'] / _['ents']
            _['row'] = eval_num(_['blk'].assigns.get("row_dim"))
            if _['row'] is None and (_['per'] > 1024 or abs(_['per'] - round(_['per'])) > 1e-9):
                _['rep'].add(W, _['blk'].line, "params",
                        f"memory {_['nm']}: params/table = {_['per']:,.1f} 参数/条目 —— "
                        f"查表通常是 row_dim（128~512）/条目。"
                        f"缺 row_dim 或 heads 声明，无法确认单位")

