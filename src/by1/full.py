#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1full -- 从完整架构声明生成 Transformers 模型，包含视觉与独立 MTP 分支。

编译阶段只用标准库。执行阶段使用 Transformers 的模型/算子实现。
Qwen 的 MTP 拼接、归一化与共享关系在本模块实现，不丢弃 checkpoint 的 mtp.*。
MTP 组合依据 vLLM qwen3_5_mtp.py；它的输入 hidden_states 是主干最终归一化后的值。
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401
import argparse
from by1ir import ALPHA_DEFAULT as _ALPHA
import copy
import json
import math
import pprint
import re

FAMILIES = {
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
    "qwen2": ("Qwen2Config", "Qwen2ForCausalLM"),
    "gpt2": ("GPT2Config", "GPT2LMHeadModel"),
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM"),
    "qwen3_5": ("Qwen3_5Config", "Qwen3_5ForConditionalGeneration"),
    # **第二个机制族（T26）**：选择性状态空间，没有 KV cache。
    # 注意这两个类在 transformers 里**没导出到顶层** —— 见 `_tclass`。
    "mamba": ("MambaConfig", "MambaForCausalLM"),
    # **第三个机制族（T27）**：混合栈 —— 同一份声明里 Attention 与 SSM 混排。
    # 这一类把 config 生成从"一个家族一种机制"推到"逐层类型表"。
    "granitemoehybrid": ("GraniteMoeHybridConfig", "GraniteMoeHybridForCausalLM"),
    # **层内并行（T29）**：每层里 SSM 与注意力并存、输出相加。
    "falcon_h1": ("FalconH1Config", "FalconH1ForCausalLM"),
}
MECH_FIELDS = {
    "Attention": {"q", "kv", "head_dim", "out_dim", "window", "qk_norm", "bias",
                  "attn_bias", "qkv_bias", "out_bias", "q_gate", "partial", "qkv_fused",
                  "sink", "structural"},
    "Linear": {"k_heads", "k_dim", "v_heads", "v_dim", "conv_kernel", "act", "structural"},
    "FFN": {"hidden", "act", "gate", "bias", "structural"},
    "MoE": {"experts", "top_k", "hidden", "shared", "router_bias", "expert_bias",
            "routing", "act", "swiglu_limit", "alpha", "structural"},
    # **Mamba1 那一代的参数化**（不是 Mamba2：那边的 `heads`/`n_groups`/`chunk_size`
    # 在这里没有对应物 —— 见 models/mamba-130m-hf.by1 的长注释）。
    "SSM": {"expand", "conv_kernel", "ssm_state", "conv_bias", "proj_bias",
            "act", "structural"},
    # **Mamba2 那一代的参数化（T27）**：宽度按 `n_heads × d_head` 算，
    # `A_log` / `D` / `dt_bias` 的长度是 **heads** 而不是 state，
    # 另有 `n_groups`（B/C 的组数）与 `chunk_size`（分块扫描）。
    # **和 `SSM` 不是一套** —— 两者并存，因为两个模型都在。
    "SSM2": {"n_heads", "d_head", "d_state", "n_groups", "chunk_size",
             "d_conv", "conv_bias", "proj_bias", "expand", "act", "structural"},
    # **组合机制（T29）**：一层里两个混合器**并存**（不是"这层是 A、那层是 B"）。
    # `a` / `b` 是**指向另外两个机制的名字** —— 语言层不用改：
    # `mech X : KIND { k = v }` 本来就收裸名，而"不认识的 kind"由各后端明确拒绝
    # （实测：codegen 报"机制 'Both' 的类型是 'Parallel'，codegen 还不支持"）。
    "Parallel": {"a", "b", "structural"},
}


def _tclass(name):
    """取 transformers 的一个类：**顶层取不到就去 auto 映射里取**。

    实测（2026-10-16）：`MambaConfig` / `MambaForCausalLM` 在本机 5.15.1 与
    克隆 5.20.0 里**都没有导出到顶层**（`getattr(T, 'MambaForCausalLM')` 直接
    AttributeError），但 `CONFIG_MAPPING_NAMES['mamba'] = 'MambaConfig'`、
    因果 LM 映射里也有它。原来三处都写死 `getattr(T, ...)` —— 加家族时会
    以"找不到类"的形式炸掉，而真正的原因只是"没导出"。
    """
    import transformers as T
    cls = getattr(T, name, None)
    if cls is not None:
        return cls
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    for cfg_cls in CONFIG_MAPPING.values():
        if cfg_cls is None:
            continue
        if cfg_cls.__name__ == name:
            return cfg_cls
        lm = MODEL_FOR_CAUSAL_LM_MAPPING.get(cfg_cls)
        if lm is not None and lm.__name__ == name:
            return lm
    raise ValueError(f"transformers 里找不到 {name}（顶层与 auto 映射都没有）")


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def _bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    _fail(text in ("true", "false", "on", "off", "none", "yes", "no", "1", "0"),
          f"布尔值 {value!r} 不合法")
    return text in ("true", "on", "yes", "1")


def _number(values, key, default=None, integer=True):
    value = values.get(key, default)
    _fail(value is not None, f"完整架构缺少 {key}")
    try:
        n = float(str(value).replace('_', ''))
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是数值，得到 {value!r}") from None
    _fail(math.isfinite(n), f"{key} 必须有限")
    if integer:
        _fail(n == int(n) and n > 0, f"{key} 必须是正整数")
        return int(n)
    return n


def _known(values, allowed, where):
    extra = set(values) - set(allowed)
    _fail(not extra, f"{where} 不支持这些字段：{', '.join(sorted(extra))}")


def _json(value):
    if value is None or value == "null" or value == "none":
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return str(value).strip('"')


def _layers(info):
    result, indices = {}, {}
    for stack, name, attrs, _key, attached in info['layer_seq']:
        index = indices.get(stack, 0)
        indices[stack] = index + 1
        mix = (info['mechs'][name], dict(attrs))
        feeds = []
        for target in attached:
            a = dict(info['mech_attrs'][target])
            a.update(info['overrides'].get(f'{stack}|{index}|{target}', {}))
            feeds.append((info['mechs'][target], a))
        # **SSM 层可以没有 FFN**（T26，mamba）：Mamba1 的 mixer **就是整层**
        # —— `intermediate_size = d_inner`，不是"另有一个 FFN"。
        # 其余机制仍然要求恰好一个 feed（这条规则原来是无条件的，于是
        # mamba 的 complete 声明在建 config 这一步就炸）。
        if mix[0] == 'SSM':
            _fail(not feeds, "SSM 层不能再挂 FFN/MoE（这一代 mixer 就是整层）")
        else:
            _fail(len(feeds) == 1, f"{stack}[{index}] 必须挂载一个 FFN/MoE")
        for kind, a in [mix] + feeds:
            _fail(kind in MECH_FIELDS, f"完整后端尚不支持 {kind}")
            _known(a, MECH_FIELDS[kind], kind)
        result.setdefault(stack, []).append((mix, feeds[0] if feeds else None))
    return result


def _uniform(items, what):
    _fail(bool(items), f"缺少 {what}")
    first = {k: v for k, v in items[0].items() if k not in ('structural', 'window')}
    for item in items[1:]:
        candidate = {k: v for k, v in item.items() if k not in ('structural', 'window')}
        _fail(candidate == first, f"{what} 的逐层属性不一致，此后端不能表达这种变体")
    return items[0]


def _rope(info, attention):
    positions = info.get('position') or {}
    _fail(set(positions) <= {'default'}, "完整后端需要一份 default 位置策略")
    text = positions.get('default', '')
    def number(key, default=None):
        match = re.search(r'\b' + key + r'\s*=\s*([-\d.eE+]+)', text)
        value = match.group(1) if match else default
        return None if value is None else float(value)
    if info['arch'] == 'gpt2':
        match = re.fullmatch(r'learned\((\d+)\)', text.strip())
        _fail(match is not None, "GPT-2 需要 position { default = learned(N) }")
        return {'n_positions': int(match.group(1))}
    _fail('rope(' in text, "完整架构需要显式 rope(base=...) 位置策略")
    base = number('base')
    _fail(base is not None and base > 0, "RoPE 需要正的 base")
    pairing = re.search(r'\bpairing\s*=\s*(\w+)', text)
    expected = 'interleaved' if info['arch'] == 'gpt_oss' else 'half'
    _fail(pairing is not None and pairing.group(1) == expected,
          f"{info['arch']} 后端的配对约定为 {expected}，请显式声明")
    params = {'rope_type': 'default', 'rope_theta': base}
    partial = float(attention.get('partial', number('ratio', number('partial', 1.0))))
    _fail(0 < partial <= 1, "partial 必须在 (0,1] 内")
    if partial != 1:
        params['partial_rotary_factor'] = partial
    if 'yarn(' in text:
        params.update(rope_type='yarn', factor=number('factor'),
                      original_max_position_embeddings=int(number('original', 4096)),
                      beta_fast=number('beta_fast', 32), beta_slow=number('beta_slow', 1),
                      truncate=not bool(re.search(r'truncate\s*=\s*false', text)))
        if number('attention_factor') is not None:
            params['attention_factor'] = number('attention_factor')
        _fail(params['factor'] is not None and params['factor'] > 0, "YaRN 需要正的 factor")
    return params


def _compile_mamba(info, hp):
    """mamba 家族（T26）：**主干是 `SSM` mixer 的重复 —— 没有 Attention、没有 FFN。**

    这条路和别的家族不是同一套形状来源：`d_inner` / `time_step_rank` / `state_size`
    都不是从 `q/kv/head_dim` 推的，各自是独立字段（config 自己就叫这些名字）。
    所以它是一条**早退分支**，不走下面那段围着 Attention 写的公共构造。

    声明的形状与 `models/mamba-130m-hf.by1` 的 `emit -> transformers.config`
    字段表逐项一致（那张表已由 `by1all` 档 3 对发布方 config 判过：一致 32）。
    """
    stacks = _layers(info)
    main_names = [s for s in stacks if not info.get('stack_aux', {}).get(s)]
    _fail(len(main_names) == 1, "完整后端需要一个主干栈；辅助栈请声明 aux = true")
    _fail(not (set(stacks) - set(main_names)), "mamba 家族没有视觉/MTP 分支")
    main = stacks[main_names[0]]
    kinds = {mix[0] for mix, _feed in main}
    _fail(kinds == {'SSM'}, "mamba 家族的主层只能是 SSM（没有 Attention/FFN）")
    _fail(all(feed is None for _mix, feed in main), "SSM 层不能再挂 FFN/MoE")
    mech = _uniform([mix[1] for mix, _feed in main], 'SSM')

    d = _number(hp, 'd_model')
    d_inner = _number(hp, 'd_inner')
    dt_rank = _number(hp, 'dt_rank')
    vocab = _number(hp, 'vocab')
    eps = _number(hp, 'rms_eps', 1e-5, False)
    _fail(0 < eps < 1, 'norm eps 必须在 (0,1) 内')
    state = _number(mech, 'ssm_state')
    conv = _number(mech, 'conv_kernel')
    expand = _number(mech, 'expand')
    _fail(expand * d == d_inner, "d_inner 必须等于 expand × d_model（实测 2 × 768 = 1536）")
    _fail(mech.get('act', 'silu') == 'silu', "mamba 的激活实测为 silu")
    _fail(not _bool(mech.get('proj_bias')), "mamba1 的 in_proj/out_proj 无 bias（只有 dt_proj 有）")
    _fail(conv > 0, 'conv_kernel 必须为正')
    cfg = dict(
        vocab_size=vocab, hidden_size=d, num_hidden_layers=len(main),
        intermediate_size=d_inner, hidden_act='silu', initializer_range=0.1,
        layer_norm_epsilon=eps, use_cache=True,
        d_model=d, d_inner=d_inner, n_layer=len(main),
        state_size=state, conv_kernel=conv, expand=expand,
        time_step_rank=dt_rank, time_step_scale=1.0, time_step_min=0.001,
        time_step_max=0.1, time_step_init_scheme='random',
        time_step_floor=0.0001, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, rescale_prenorm_residual=False,
        use_bias=False, use_conv_bias=_bool(mech.get('conv_bias'), True),
        pad_vocab_size_multiple=8, ssm_cfg={},
        bos_token_id=int(_json(hp.get('bos_token_id', 0))),
        eos_token_id=int(_json(hp.get('eos_token_id', 0))),
        pad_token_id=int(_json(hp.get('pad_token_id', 0))),
        tie_word_embeddings=_bool(hp['tie_word_embeddings']),
    )
    _fail(cfg['time_step_rank'] > 0 and cfg['state_size'] > 0, '时间步秩与状态维必须为正')
    return 'mamba', cfg, {}


def _compile_moe_hybrid(info, hp):
    """granite 4.0-h 那一类**混合栈**：逐层在 Attention 与 SSM2 之间切换。

    **这是 by1full 第一次不按"一个家族一种机制"建配置。** 形状来自
    **逐层类型表**（`layer_types`）而不是家族名 —— T26 的 mamba 分支是
    "主干全是 SSM"的早退路；这一条要求两种机制**同时**出现在一个主干里，
    而且每层都还挂着一个 dense FFN（这个 350m 形态里 `num_local_experts = 0`，
    名字叫 `shared_mlp` 但它是逐层的普通前馈）。

    四条"必须显式声明、不许有默认值"的量（μP 的四个缩放）：
    `embedding_multiplier` / `residual_multiplier` / `logits_scaling` /
    `attention_multiplier`。**它们改数值、不改形状** —— 少一个，张量一个不差，
    算出来是另一个模型（T26 在 `A_log → -exp(A_log)` 上踩过同型的坑）。
    """
    stacks = _layers(info)
    main_names = [s for s in stacks if not info.get('stack_aux', {}).get(s)]
    _fail(len(main_names) == 1, "完整后端需要一个主干栈；辅助栈请声明 aux = true")
    _fail(not (set(stacks) - set(main_names)), "granite 混合族没有视觉/MTP 分支")
    main = stacks[main_names[0]]
    kinds = [mix[0] for mix, _feed in main]
    _fail(set(kinds) <= {'Attention', 'SSM2'},
          "混合栈的逐层机制只能是 Attention 或 SSM2")
    _fail('Attention' in kinds and 'SSM2' in kinds,
          "两种机制都要出现；只有一种的话，用对应的单机制家族")
    _fail(all(feed is not None for _mix, feed in main), "混合栈每层都要挂一个 FFN")
    feed_kinds = {feed[0] for _mix, feed in main}
    _fail(feed_kinds == {'FFN'}, "这个 350m 形态每层挂的是 dense FFN，不是 MoE/MoE2")

    attn = _uniform([mix[1] for mix, _ in main if mix[0] == 'Attention'], 'Attention')
    ssm = _uniform([mix[1] for mix, _ in main if mix[0] == 'SSM2'], 'SSM2')
    feed = _uniform([feed[1] for _mix, feed in main], 'FFN')

    d = _number(hp, 'd_model')
    vocab = _number(hp, 'vocab')
    eps = _number(hp, 'rms_eps', 1e-5, False)
    _fail(0 < eps < 1, 'norm eps 必须在 (0,1) 内')
    q, kv, hd = (_number(attn, k) for k in ('q', 'kv', 'head_dim'))
    _fail(q % kv == 0, "q 必须是 kv 的整数倍")
    _fail(_number(attn, 'out_dim', q * hd) == q * hd, "out_dim 必须等于 q × head_dim")
    _fail(q * hd == d, "这一族的注意力：q × head_dim 必须等于 d_model（没有独立的投影宽度）")
    _fail(not _bool(attn.get('bias', attn.get('qkv_bias')))
          and not _bool(attn.get('out_bias')), "granite 的注意力没有 bias")
    _fail(attn.get('qk_norm', 'off') == 'off', "这个家族不做 qk_norm")
    _fail(not _bool(attn.get('q_gate')) and not _bool(attn.get('sink')),
          "没有 q_gate / sink")

    n_heads = _number(ssm, 'n_heads')
    dh = _number(ssm, 'd_head')
    expand = _number(ssm, 'expand')
    _fail(expand * d == n_heads * dh,
          "SSM2：expand × d_model 必须等于 n_heads × d_head（实测 2 × 768 = 48 × 32）")
    _fail(ssm.get('act', 'silu') == 'silu', "SSM2 的激活实测为 silu")
    _fail(not _bool(ssm.get('proj_bias')), "这个形态的 SSM 投影没有 bias")
    hidden = _number(feed, 'hidden')
    _fail(feed.get('act', 'silu') == 'silu', "前馈激活实测为 silu")
    _fail(_bool(feed.get('gate'), True), "这个前馈是有门控的（SwiGLU）")

    # **μP 四个缩放：没有默认值。** 缺了就报错，而不是悄悄按 1.0 建一个
    # "形状对、数值错"的模型。
    for key in ('embedding_multiplier', 'residual_multiplier',
                'logits_scaling', 'attention_multiplier'):
        _fail(key in hp, f"混合栈必须显式声明 {key}（改数值的字段，不能有默认值）")
    cfg = dict(
        vocab_size=vocab, hidden_size=d, num_hidden_layers=len(main),
        num_attention_heads=q, num_key_value_heads=kv,
        layer_types=['full_attention' if k == 'Attention' else 'linear_attention'
                     for k in kinds],
        max_position_embeddings=_number(hp, 'max_pos'),
        rms_norm_eps=eps, hidden_act='silu',
        attention_bias=False, attention_dropout=0.0,
        attention_multiplier=_number(hp, 'attention_multiplier', None, False),
        embedding_multiplier=_number(hp, 'embedding_multiplier', None, False),
        residual_multiplier=_number(hp, 'residual_multiplier', None, False),
        logits_scaling=_number(hp, 'logits_scaling', None, False),
        # 这个 350m 形态是 dense：专家数为 0，走 `shared_mlp`。
        intermediate_size=hidden,
        shared_intermediate_size=_number(hp, 'shared_hidden', hidden),
        num_local_experts=0, num_experts_per_tok=0,
        output_router_logits=_bool(hp.get('output_router_logits'), False),
        router_aux_loss_coef=_number(hp, 'router_aux_loss_coef', 0.01, False),
        # Mamba2 那一套（前缀 `mamba_`）。
        mamba_n_heads=n_heads, mamba_d_head=dh,
        mamba_d_state=_number(ssm, 'd_state'),
        mamba_n_groups=_number(ssm, 'n_groups', 1),
        mamba_chunk_size=_number(ssm, 'chunk_size', 256),
        mamba_d_conv=_number(ssm, 'd_conv', 4),
        mamba_expand=expand,
        mamba_conv_bias=_bool(ssm.get('conv_bias'), True),
        mamba_proj_bias=_bool(ssm.get('proj_bias'), False),
        # **`time_step_limit` 故意不声明**：这个类的默认值实测就是 checkpoint 里的
        # `(0.0, inf)`。而它是 dataclass 里的 `tuple[float, float]`，`to_dict()`
        # 又给 **list** —— 声明进去要同时满足"构造时是 tuple、比对时等于 list"，
        # 而且 `inf` 在生成的 Python 里也不是个名字（`render()` 会吐出一个
        # NameError 的文件）。默认值相同，就不去碰它。
        # **NoPE**：这个模型的注意力层不加位置编码，但 rope 参数仍在配置里。
        position_embedding_type='nope',
        rope_parameters={'rope_type': 'default',
                         'rope_theta': _number(hp, 'rope_theta', 10000.0, False)},
        normalization_function='rmsnorm',
        init_method=hp.get('init_method', 'mup'),
        initializer_range=_number(hp, 'initializer_range', 0.1, False),
        use_cache=True,
        bos_token_id=int(_json(hp.get('bos_token_id', 0))),
        eos_token_id=int(_json(hp.get('eos_token_id', 0))),
        pad_token_id=int(_json(hp.get('pad_token_id', 0))),
        tie_word_embeddings=_bool(hp['tie_word_embeddings']),
    )
    _fail(cfg['mamba_d_state'] > 0 and cfg['mamba_chunk_size'] > 0,
          "SSM2 的状态维与分块大小必须为正")
    return 'granitemoehybrid', cfg, {}


def _compile_falcon_h1(info, hp):
    """Falcon-H1（T29）：**层内并行** —— 每层里 SSM 与注意力**并存**、输出相加。

    这是 by1 第一次处理"一层里两个混合器"。声明那边用的是一种**组合机制**：

        mech Both : Parallel { a = Mixer; b = Attn }
        pattern = 36 * Both

    `a` / `b` 是**指向另外两个已声明机制的名字**。语言层不用改：解析器本来就收
    裸名（`window = none`、`index = global` 都是这个写法），而各后端遇到不认识的
    kind 会**明确拒绝** —— 实测 codegen 的原文是
    "机制 'Both' 的类型是 'Parallel'，codegen 还不支持（目前支持：…）"。
    这正是"要么实现，要么拒绝"，不是新增机制。

    这一支只负责把它算成 `FalconH1Config`（由 config 逐字段判据守住）。
    """
    stacks = _layers(info)
    main_names = [s for s in stacks if not info.get('stack_aux', {}).get(s)]
    _fail(len(main_names) == 1, "完整后端需要一个主干栈；辅助栈请声明 aux = true")
    _fail(not (set(stacks) - set(main_names)), "falcon_h1 没有视觉/MTP 分支")
    main = stacks[main_names[0]]
    _fail(all(mix[0] == 'Parallel' for mix, _ in main),
          "falcon_h1 的每一层都要是 Parallel —— 层内并行是这一族的定义")
    _fail(all(feed is not None for _mix, feed in main), "每层都要挂一个 FFN")
    _fail({feed[0] for _mix, feed in main} == {'FFN'}, "这一族每层挂的是 dense FFN")
    feed = _uniform([feed[1] for _mix, feed in main], 'FFN')
    mixers = _uniform([mix[1] for mix, _ in main], 'Parallel')

    sub = {}
    for slot in ('a', 'b'):
        _fail(slot in mixers, f"Parallel 必须写 a 与 b（缺 {slot}）")
        name = mixers[slot]
        _fail(name in info['mechs'],
              f"Parallel 的 {slot} 指向未声明的机制 {name!r}")
        sub[info['mechs'][name]] = dict(info['mech_attrs'][name])
    _fail(set(sub) == {'SSM2', 'Attention'},
          "这一族的层内并行恰好是 SSM2 与 Attention 两条支路")
    ssm, attn = sub['SSM2'], sub['Attention']

    d = _number(hp, 'd_model')
    vocab = _number(hp, 'vocab')
    eps = _number(hp, 'rms_eps', 1e-5, False)
    _fail(0 < eps < 1, 'norm eps 必须在 (0,1) 内')
    q, kv, hd = (_number(attn, k) for k in ('q', 'kv', 'head_dim'))
    _fail(q % kv == 0, "q 必须是 kv 的整数倍")
    _fail(not _bool(attn.get('bias', attn.get('qkv_bias')))
          and not _bool(attn.get('out_bias')), "falcon_h1 的注意力没有 bias")
    n_heads = _number(ssm, 'n_heads')
    dh = _number(ssm, 'd_head')
    # **这一族的 SSM 宽度与主干解耦**：`d_ssm = n_heads × d_head = 1536`，
    # 而 `expand × d_model = 2048` —— 两者**不相等**（与 granite 那一族相反）。
    # 所以这里按 `n_heads × d_head` 推，并显式记下这条差异。
    _fail(ssm.get('act', 'silu') == 'silu', "SSM2 的激活实测为 silu")
    _fail(not _bool(ssm.get('proj_bias')), "这一族的 SSM 投影没有 bias")
    hidden = _number(feed, 'hidden')
    _fail(feed.get('act', 'silu') == 'silu' and _bool(feed.get('gate'), True),
          "前馈是有门控的 silu（SwiGLU）")

    for key in ('embedding_multiplier', 'attention_in_multiplier',
                'attention_out_multiplier', 'ssm_in_multiplier',
                'ssm_out_multiplier', 'key_multiplier', 'lm_head_multiplier',
                'ssm_multipliers', 'mlp_multipliers'):
        _fail(key in hp, f"层内并行必须显式声明 {key}（改数值的字段，不能有默认值）")
    cfg = dict(
        vocab_size=vocab, hidden_size=d, num_hidden_layers=len(main),
        num_attention_heads=q, num_key_value_heads=kv, head_dim=hd,
        max_position_embeddings=_number(hp, 'max_pos'),
        rms_norm_eps=eps, hidden_act='silu',
        attention_bias=False, attention_dropout=0.0,
        attention_in_multiplier=_number(hp, 'attention_in_multiplier', None, False),
        attention_out_multiplier=_number(hp, 'attention_out_multiplier', None, False),
        embedding_multiplier=_number(hp, 'embedding_multiplier', None, False),
        key_multiplier=_number(hp, 'key_multiplier', None, False),
        lm_head_multiplier=_number(hp, 'lm_head_multiplier', None, False),
        ssm_in_multiplier=_number(hp, 'ssm_in_multiplier', None, False),
        ssm_out_multiplier=_number(hp, 'ssm_out_multiplier', None, False),
        ssm_multipliers=[float(x) for x in _json(hp['ssm_multipliers'])],
        mlp_multipliers=[float(x) for x in _json(hp['mlp_multipliers'])],
        intermediate_size=hidden,
        mlp_bias=_bool(hp.get('mlp_bias'), False),
        mlp_expansion_factor=_number(hp, 'mlp_expansion_factor', 8, False),
        projectors_bias=_bool(hp.get('projectors_bias'), False),
        mamba_n_heads=n_heads, mamba_d_head=dh,
        mamba_d_state=_number(ssm, 'd_state'),
        mamba_n_groups=_number(ssm, 'n_groups', 1),
        mamba_chunk_size=_number(ssm, 'chunk_size', 128),
        mamba_d_conv=_number(ssm, 'd_conv', 4),
        mamba_expand=_number(ssm, 'expand'),
        mamba_d_ssm=n_heads * dh,
        mamba_conv_bias=_bool(ssm.get('conv_bias'), True),
        mamba_proj_bias=_bool(ssm.get('proj_bias'), False),
        mamba_use_mlp=_bool(hp.get('mamba_use_mlp'), True),
        mamba_norm_before_gate=_bool(hp.get('mamba_norm_before_gate'), False),
        mamba_rms_norm=_bool(hp.get('mamba_rms_norm'), False),
        rope_parameters={'rope_type': 'default',
                         'rope_theta': _number(hp, 'rope_theta', 1e11, False)},
        initializer_range=_number(hp, 'initializer_range', 0.02, False),
        use_cache=True,
        bos_token_id=int(_json(hp.get('bos_token_id', 1))),
        eos_token_id=int(_json(hp.get('eos_token_id', 11))),
        pad_token_id=int(_json(hp.get('pad_token_id', 0))),
        tie_word_embeddings=_bool(hp['tie_word_embeddings']),
    )
    _fail(cfg['mamba_d_ssm'] > 0 and cfg['mamba_chunk_size'] > 0,
          "SSM 的宽度与分块大小必须为正")
    return 'falcon_h1', cfg, {}


def compile_config(info):
    """声明 -> (家族, 完整配置, MTP 规格)。不下载配置、不导入 torch。"""
    family = info.get('arch')
    _fail(family in FAMILIES, f"完整后端的 arch 必须是 {', '.join(FAMILIES)}")
    _known(info.get('model_attrs', {}), {'backend', 'arch', 'ctx', 'license'}, 'model')
    _fail(not info.get('unsupported_full_blocks'), f"完整后端不能忽略这些块：{info.get('unsupported_full_blocks')}")
    hp = info['hparams']
    if family != 'gpt2':
        _fail(not {'ln_eps', 'resid_dropout', 'embed_dropout'} & set(hp), '这些 LayerNorm/dropout 字段仅适用于 GPT-2')
    _known(hp, {'d_model', 'vocab', 'n_layer', 'rms_eps', 'ln_eps', 'norm_kind', 'norm',
                'tie_word_embeddings', 'bos_token_id', 'eos_token_id', 'pad_token_id',
                'attention_dropout', 'resid_dropout', 'embed_dropout', 'initializer_range',
                # mamba 独有（T26）：都不是从 d_model 推的，各自是独立字段。
                'd_inner', 'dt_rank',
                # 混合栈独有（T27）：μP 的四个缩放 **必须显式声明**（改数值，
                # 不改形状 —— 少一个就是"算的是另一个模型"），以及上下文长度。
                'max_pos', 'embedding_multiplier', 'residual_multiplier',
                'logits_scaling', 'attention_multiplier', 'shared_hidden',
                'init_method', 'router_aux_loss_coef', 'output_router_logits',
                'rope_theta',
                # 层内并行独有（T29）：九个缩放标量 + FFN 的两个开关 +
                # Mamba 那三个开关。**它们都改数值或改结构**，全部显式声明。
                'attention_in_multiplier', 'attention_out_multiplier',
                'ssm_in_multiplier', 'ssm_out_multiplier', 'key_multiplier',
                'lm_head_multiplier', 'ssm_multipliers', 'mlp_multipliers',
                'mlp_expansion_factor', 'mlp_bias', 'projectors_bias',
                'mamba_use_mlp', 'mamba_norm_before_gate', 'mamba_rms_norm'}, 'hparams')
    _fail('tie_word_embeddings' in hp, "完整架构必须显式声明 tie_word_embeddings")
    if family == 'mamba':
        return _compile_mamba(info, hp)
    if family == 'granitemoehybrid':
        return _compile_moe_hybrid(info, hp)
    if family == 'falcon_h1':
        return _compile_falcon_h1(info, hp)
    stacks = _layers(info)
    main_names = [s for s in stacks if not info.get('stack_aux', {}).get(s)]
    _fail(len(main_names) == 1, "完整后端需要一个主干栈；视觉/MTP 栈请声明 aux = true")
    main = stacks[main_names[0]]
    remaining = set(stacks) - set(main_names)
    _fail(remaining <= {'visual', 'mtp'}, f"未实现的辅助栈：{remaining - {'visual', 'mtp'}}")
    attentions = [a for (kind, a), _ in main if kind == 'Attention']
    attention = _uniform(attentions, 'Attention')
    feeds = _uniform([a for _, (_, a) in main], 'FFN/MoE')
    feed_kinds = {kind for _, (kind, _) in main}
    _fail(len(feed_kinds) == 1, "此后端不支持混合 FFN/MoE 类型")
    feed_kind = next(iter(feed_kinds))
    d, vocab = _number(hp, 'd_model'), _number(hp, 'vocab')
    q, kv, hd = (_number(attention, k) for k in ('q', 'kv', 'head_dim'))
    _fail(q % kv == 0, "q 必须是 kv 的整数倍")
    _fail(_number(attention, 'out_dim', q * hd) == q * hd, "此后端的 out_dim 必须等于 q*head_dim")
    eps = _number(hp, 'ln_eps' if family == 'gpt2' and 'ln_eps' in hp else 'rms_eps', 1e-5, False)
    _fail(eps > 0, "norm eps 必须为正")
    norm_kind = hp.get('norm_kind', 'rms')
    _fail(norm_kind == ('layer' if family == 'gpt2' else 'rms'), "norm_kind 与所选家族不一致")
    _fail(hp.get('norm') in (None, 'standard', 'zero_centered'), '未知 norm 约定')
    _fail((hp.get('norm') == 'zero_centered') == (family == 'qwen3_5'),
          "Qwen3.5/3.8 必须显式声明 norm = zero_centered；其他家族使用普通归一化权重")
    bias = _bool(attention.get('bias', attention.get('attn_bias')))
    qb, ob = _bool(attention.get('qkv_bias'), bias), _bool(attention.get('out_bias'), bias)
    expected_bias = (True, False) if family == 'qwen2' else ((True, True) if family in ('gpt2', 'gpt_oss') else (False, False))
    _fail((qb, ob) == expected_bias, f"{family} 的 qkv_bias/out_bias 应为 {expected_bias}")
    _fail(attention.get('qk_norm', 'off') == ('per_head' if family in ('qwen3', 'qwen3_5') else 'off'),
          "qk_norm 与所选家族不一致")
    _fail(_bool(attention.get('q_gate')) == (family == 'qwen3_5'), "q_gate 与所选家族不一致")
    _fail(_bool(attention.get('sink')) == (family == 'gpt_oss'), "sink 与所选家族不一致")
    if family != 'qwen3_5':
        _fail(all(kind == 'Attention' for (kind, _), _ in main), "这个家族只支持 Attention 主层")
        _fail(not remaining and not info.get('vision') and not info.get('mtp'), "此家族没有视觉/MTP 分支")
    common = dict(vocab_size=vocab, hidden_size=d, num_hidden_layers=len(main),
                  num_attention_heads=q, num_key_value_heads=kv, head_dim=hd,
                  intermediate_size=_number(feeds, 'hidden'), rms_norm_eps=eps,
                  max_position_embeddings=int(info['ctx']),
                  tie_word_embeddings=_bool(hp['tie_word_embeddings']),
                  attention_bias=qb, attention_dropout=_number(hp, 'attention_dropout', 0, False),
                  initializer_range=_number(hp, 'initializer_range', 0.02, False))
    _fail(0 <= common['attention_dropout'] < 1 and common['initializer_range'] > 0,
          'dropout 必须在 [0,1) 内，initializer_range 必须为正')
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        if key in hp:
            value = _json(hp[key])
            for token in value if isinstance(value, list) else [value]:
                _fail(token is None or (isinstance(token, int) and 0 <= token < vocab), f"{key} 超出词表")
            common[key] = value
    rope = _rope(info, attention)
    if family == 'gpt2':
        _fail(rope['n_positions'] == int(info['ctx']), 'learned 位置表长度必须等于 ctx')
        _fail(feed_kind == 'FFN' and feeds.get('act') == 'gelu_new' and not _bool(feeds.get('gate'), True)
              and _bool(feeds.get('bias')), "GPT-2 需要带 bias、无门控的 gelu_new FFN")
        _fail(q == kv and d == q * hd, "GPT-2 需要标准 MHA")
        cfg = dict(vocab_size=vocab, n_embd=d, n_layer=len(main), n_head=q,
                   n_inner=_number(feeds, 'hidden'), layer_norm_epsilon=eps,
                   activation_function='gelu_new', tie_word_embeddings=common['tie_word_embeddings'],
                   attn_pdrop=_number(hp, 'attention_dropout', 0.1, False),
                   resid_pdrop=_number(hp, 'resid_dropout', 0.1, False),
                   embd_pdrop=_number(hp, 'embed_dropout', 0.1, False), **rope)
        cfg['initializer_range'] = common['initializer_range']
        _fail(all(0 <= cfg[k] < 1 for k in ('attn_pdrop', 'resid_pdrop', 'embd_pdrop')), 'GPT-2 dropout 不合法')
        cfg.update({k: v for k, v in common.items() if k.endswith('token_id')})
        return family, cfg, {}
    if family == 'gpt_oss':
        _fail(feed_kind == 'MoE' and feeds.get('act') == 'gptoss', "gpt_oss 需要 gptoss MoE 激活")
        _fail(feeds.get('routing') == 'topk_softmax' and _bool(feeds.get('router_bias'))
              and _bool(feeds.get('expert_bias')), "gpt_oss 需要带偏置的 topk_softmax 路由/专家")
        _fail(int(feeds.get('shared', 0)) == 0, "gpt_oss 没有共享专家")
        _fail(float(feeds.get('alpha', _ALPHA)) == _ALPHA, "gpt_oss 的 alpha 为 1.702")
        _fail(float(feeds.get('swiglu_limit', 7)) == 7, '此 gpt_oss 实现的 swiglu_limit 固定为 7')
        windows = [None if a.get('window', 'none') in ('none', 'null', '', None) else int(a['window'])
                   for (_, a), _ in main]
        sizes = {w for w in windows if w is not None}
        _fail(len(sizes) == 1, "gpt_oss 需要一个滑窗大小")
        _fail(next(iter(sizes)) > 0, 'window 必须为正')
        common.update(num_local_experts=_number(feeds, 'experts'), num_experts_per_tok=_number(feeds, 'top_k'),
                      layer_types=['sliding_attention' if w else 'full_attention' for w in windows],
                      sliding_window=next(iter(sizes)), swiglu_limit=_number(feeds, 'swiglu_limit', 7, False),
                      rope_parameters=rope)
        _fail(common['num_experts_per_tok'] <= common['num_local_experts'], 'top_k 不能超过 experts')
        return family, common, {}
    _fail(feed_kind == 'FFN' and feeds.get('act', 'silu') == 'silu' and _bool(feeds.get('gate'), True)
          and not _bool(feeds.get('bias')), "此家族需要无 bias 的门控 SiLU FFN")
    common.update(hidden_act='silu', rope_parameters=rope)
    if family in ('qwen2', 'qwen3'):
        _fail(rope.get('partial_rotary_factor', 1) == 1, '这个家族使用完整头维 RoPE')
        _fail(all(a.get('window', 'none') in ('none', 'null', '', None) for a in attentions),
              "这个完整描述只支持全量注意力，不能忽略 window")
        common.update(use_sliding_window=False)
        if family == 'qwen2':
            _fail(d == q * hd, 'Qwen2 的 head_dim 由 d_model/q 推导，必须一致')
            common.pop('head_dim')
            common.pop('attention_bias')
        return family, common, {}

    linears = [a for (kind, a), _ in main if kind == 'Linear']
    _fail(all(kind in ('Linear', 'Attention') for (kind, _), _ in main), 'Qwen3.8 主干只能混排 Linear/Attention')
    _fail(all(a.get('window', 'none') in ('none', 'null', '', None) for a in attentions), 'Qwen3.8 全量层不使用滑窗')
    linear = _uniform(linears, 'Linear')
    _fail(linear.get('act', 'silu') == 'silu', "Qwen3.5 的线性注意力使用 SiLU")
    common.update(layer_types=['linear_attention' if kind == 'Linear' else 'full_attention'
                               for (kind, _), _ in main],
                  linear_num_key_heads=_number(linear, 'k_heads'),
                  linear_num_value_heads=_number(linear, 'v_heads'),
                  linear_key_head_dim=_number(linear, 'k_dim'),
                  linear_value_head_dim=_number(linear, 'v_dim'),
                  linear_conv_kernel_dim=_number(linear, 'conv_kernel', 4))
    _fail(common['linear_num_value_heads'] % common['linear_num_key_heads'] == 0,
          "线性注意力 v_heads 必须是 k_heads 的整数倍")
    _fail('visual' in stacks and 'mtp' in stacks, "完整版 Qwen3.8 必须同时声明 visual 和 mtp 栈")
    _fail(bool(info.get('vision')) and bool(info.get('mtp')), "需要 vision 和 mtp 的连接/共享参数")
    va = _uniform([a for (kind, a), _ in stacks['visual'] if kind == 'Attention'], '视觉 Attention')
    vf = _uniform([a for _, (kind, a) in stacks['visual'] if kind == 'FFN'], '视觉 FFN')
    _fail(all(m[0] == 'Attention' and f[0] == 'FFN' for m, f in stacks['visual']), "视觉栈需要 Attention + FFN")
    vd = _number(info['stack_d_model'], 'visual')
    vh = _number(va, 'q')
    _fail(vh == _number(va, 'kv') and vh * _number(va, 'head_dim') == vd,
          "视觉头数/头维与宽度不一致")
    _fail(_bool(va.get('qkv_fused')) and _bool(va.get('bias')) and not _bool(va.get('q_gate')),
          "视觉注意力需要融合 qkv、带 bias、无输出门控")
    _fail(vf.get('act') == 'gelu_new' and not _bool(vf.get('gate'), True) and _bool(vf.get('bias')),
          "视觉 FFN 需要带 bias、无门控的 gelu_new")
    vision = info['vision']
    vfields = {'patch_size', 'temporal_patch_size', 'spatial_merge_size', 'num_position_embeddings',
               'in_channels', 'mrope_section', 'image_token_id', 'video_token_id',
               'vision_start_token_id', 'vision_end_token_id'}
    _known(vision, vfields, 'vision')
    vc = {key: _number(vision, key) for key in vfields if key not in
          {'mrope_section', 'image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'}}
    vc.update(depth=len(stacks['visual']), hidden_size=vd, num_heads=vh,
              intermediate_size=_number(vf, 'hidden'), out_hidden_size=d,
              hidden_act='gelu_pytorch_tanh')
    sections = _json(vision.get('mrope_section'))
    _fail(isinstance(sections, list) and len(sections) == 3 and all(isinstance(v, int) and v >= 0 for v in sections),
          "mrope_section 必须是三个非负整数")
    _fail(sum(sections) * 2 == int(hd * rope.get('partial_rotary_factor', 1)), "mrope_section 与旋转维度不一致")
    rope.update(mrope_section=sections, mrope_interleaved=True)
    common['mtp_num_hidden_layers'] = len(stacks['mtp'])
    cfg = dict(text_config=common, vision_config=vc, tie_word_embeddings=common['tie_word_embeddings'])
    for key in ('image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'):
        cfg[key] = _number(vision, key)
        _fail(cfg[key] < vocab, f"{key} 超出词表")
    _fail(len({cfg[k] for k in cfg if k.endswith('token_id')}) == 4, "视觉 token ID 必须互不相同")
    for mixer, feed in stacks['mtp']:
        _fail(mixer[0] == 'Attention' and feed[0] == 'FFN', "MTP 必须是全量 Attention + FFN")
        _fail({k: v for k, v in mixer[1].items() if k not in ('structural', 'window')} ==
              {k: v for k, v in attention.items() if k not in ('structural', 'window')},
              "MTP 注意力参数必须与主干全量注意力一致")
        _fail({k: v for k, v in feed[1].items() if k != 'structural'} ==
              {k: v for k, v in feeds.items() if k != 'structural'}, "MTP FFN 参数必须与主干一致")
    _known(info['mtp'], {'share_embeddings', 'share_head'}, 'mtp')
    _fail(_bool(info['mtp'].get('share_embeddings')) and _bool(info['mtp'].get('share_head')),
          "此 MTP 格式要求与主干共享 embedding 和 lm_head")
    return family, cfg, dict(depth=len(stacks['mtp']), share_embeddings=True, share_head=True)


def compile_quantization(info):
    """Extract the validated torch.module quantization declaration, if any."""
    rule = ((info.get('emit') or {}).get('torch.module') or {}).get('quant')
    if not rule:
        return None
    import by1quant
    return by1quant.compile_emit_quant(rule)


def _model_class(family, mtp_spec):
    import transformers as T
    base = _tclass(FAMILIES[family][1])
    if not mtp_spec:
        return base
    import torch
    import weakref
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

    class MultiTokenPredictor(torch.nn.Module):
        def __init__(self, parent, text_config):
            super().__init__()
            object.__setattr__(self, '_parent', weakref.ref(parent))
            cfg = copy.deepcopy(text_config)
            cfg.num_hidden_layers = mtp_spec['depth']
            cfg.layer_types = ['full_attention'] * cfg.num_hidden_layers
            cfg._attn_implementation = 'eager'
            d = cfg.hidden_size
            self.fc = torch.nn.Linear(2 * d, d, bias=False)
            self.pre_fc_norm_embedding = Qwen3_5RMSNorm(d, cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5RMSNorm(d, cfg.rms_norm_eps)
            self.layers = torch.nn.ModuleList(Qwen3_5DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
            self.norm = Qwen3_5RMSNorm(d, cfg.rms_norm_eps)
            self.config = cfg

        @property
        def embed_tokens(self):
            return self._parent().get_input_embeddings()

        @property
        def lm_head(self):
            return self._parent().get_output_embeddings()

        def forward(self, next_input_ids, hidden_states, position_ids, attention_mask=None,
                    step=0, past_key_values=None, labels=None, inputs_embeds=None):
            _fail(next_input_ids.shape == hidden_states.shape[:2], "MTP token 与主干 hidden 的 batch/seq 不一致")
            _fail(position_ids is not None, "MTP 必须显式给出下一 token 的 position_ids")
            _fail(step >= 0, "MTP step 必须非负")
            embedding = self.embed_tokens(next_input_ids) if inputs_embeds is None else inputs_embeds
            _fail(embedding.shape == hidden_states.shape, 'MTP inputs_embeds 必须与主干 hidden 对齐')
            joined = torch.cat((self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(hidden_states)), dim=-1)
            x = self.fc(joined)
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            rotary = self._parent().model.language_model.rotary_emb(x, position_ids)
            if attention_mask is None:
                _fail(past_key_values is None, "增量 MTP 请同时传 attention_mask 与 cache")
                n = next_input_ids.shape[1]
                mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
                attention_mask = torch.zeros(n, n, dtype=x.dtype, device=x.device).masked_fill(mask, torch.finfo(x.dtype).min)[None, None]
            x = self.layers[step % len(self.layers)](x, position_embeddings=rotary,
                    position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values)
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                _fail(labels.shape == next_input_ids.shape, "MTP labels 必须与 next_input_ids 对齐")
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)
            return {'logits': logits, 'hidden_states': x, 'loss': loss, 'past_key_values': past_key_values}

    class CompleteQwen(base):
        _keys_to_ignore_on_load_unexpected = []

        def __init__(self, config):
            super().__init__(config)
            self.mtp = MultiTokenPredictor(self, config.text_config)
            self.mtp.apply(self._init_weights)

        def forward_mtp(self, next_input_ids, hidden_states, position_ids, **kwargs):
            return self.mtp(next_input_ids, hidden_states, position_ids, **kwargs)

    return CompleteQwen


def build_model(family, configuration, mtp_spec=None, device=None, dtype=None):
    """按完整声明建模型；device='meta' 可核对真实尺寸而不分配权重内存。"""
    import torch
    import transformers as T
    cfg = _tclass(FAMILIES[family][0])(**copy.deepcopy(configuration))
    cfg._attn_implementation = 'eager'
    if family == 'qwen3_5':
        cfg.text_config._attn_implementation = 'eager'
        cfg.vision_config._attn_implementation = 'eager'
    cls = _model_class(family, mtp_spec or {})
    if device is None:
        model = cls(cfg)
    else:
        with torch.device(device):
            model = cls(cfg)
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def load_model(family, configuration, mtp_spec, checkpoint, quantization=None, **kwargs):
    """Load a checkpoint and enforce its declared native quantized storage."""
    import transformers as T
    checkpoint_cfg = T.AutoConfig.from_pretrained(checkpoint, local_files_only=kwargs.get('local_files_only', False))
    _fail(checkpoint_cfg.model_type == family, 'checkpoint 的模型家族与 arch 不一致')
    declared = copy.deepcopy(configuration)
    for section in ('text_config', 'vision_config'):
        actual = getattr(checkpoint_cfg, section, None)
        if section in declared:
            _fail(actual is not None, f"checkpoint 缺少 {section}")
            _check_config(declared[section], actual.to_dict(), section)
    actual = checkpoint_cfg.to_dict()
    _check_config({k: v for k, v in declared.items() if k not in ('text_config', 'vision_config')}, actual, 'config')
    cfg = _tclass(FAMILIES[family][0])(**declared)
    cfg._attn_implementation = 'eager'
    if family == 'qwen3_5':
        cfg.text_config._attn_implementation = 'eager'
        cfg.vision_config._attn_implementation = 'eager'
    if hasattr(checkpoint_cfg, 'quantization_config'):
        cfg.quantization_config = checkpoint_cfg.quantization_config
    if quantization:
        import by1quant
        native_config = by1quant.make_transformers_config(
            quantization, getattr(checkpoint_cfg, 'quantization_config', None), T)
        supplied_config = kwargs.get('quantization_config')
        if supplied_config is not None:
            supplied = (supplied_config.to_dict() if hasattr(supplied_config, 'to_dict')
                        else dict(supplied_config) if isinstance(supplied_config, dict) else {})
            native = native_config.to_dict() if hasattr(native_config, 'to_dict') else {}
            supplied_method = getattr(supplied.get('quant_method', ''), 'value', supplied.get('quant_method', ''))
            native_method = getattr(native.get('quant_method', ''), 'value', native.get('quant_method', ''))
            _fail(str(supplied_method).lower() == str(native_method).lower(),
                  '调用方 quantization_config 与 .by1 格式不一致')
            dequantize = getattr(supplied_config, 'dequantize', supplied.get('dequantize'))
            _fail(dequantize is not True,
                  '.by1 量化声明要求原生量化存储，不能请求 dequantize=True')
            native_config = supplied_config
        else:
            kwargs['quantization_config'] = native_config
        cfg.quantization_config = native_config
    cfg.dtype = checkpoint_cfg.dtype
    kwargs.setdefault('dtype', 'auto')
    model, report = _model_class(family, mtp_spec).from_pretrained(checkpoint, config=cfg,
                                              output_loading_info=True, **kwargs)
    problems = {k: report.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if report.get(k)}
    _fail(not problems, f"checkpoint 没有完整加载：{problems}")
    if quantization:
        import by1quant
        import torch
        model._by1_quantization_audit = by1quant.audit_runtime_storage(model, quantization, torch)
        model._by1_quantization_audit['sharing'] = by1quant.audit_shared_weights(
            model, bool(configuration.get('tie_word_embeddings')), bool(mtp_spec))
    return model


def _check_config(declared, actual, where):
    for key, value in declared.items():
        if key == 'rope_parameters':
            source = actual.get(key) or actual.get('rope_scaling') or {'rope_theta': actual.get('rope_theta')}
            _check_config(value, source, where + '.' + key)
            continue
        other = actual.get(key)
        if key == 'n_inner' and other is None and actual.get('n_embd'):
            other = 4 * actual['n_embd']
        _fail(key in actual and other == value, f"{where}.{key}：声明 {value!r}，checkpoint 为 {other!r}")


def render(info, filename):
    from by1ver import stamp
    family, cfg, mtp = compile_config(info)
    quantization = compile_quantization(info)
    return (f'# {stamp()}\n# Generated by by1 from {filename}\n# Complete Transformers backend; includes declared MTP weights.\n'
            'from by1full import build_model, load_model\n\n'
            f'FAMILY = {family!r}\nCONFIG = {pprint.pformat(cfg, sort_dicts=False)}\n'
            f'MTP = {mtp!r}\nQUANTIZATION = {pprint.pformat(quantization, sort_dicts=False)}\n\n'
            'def build(device=None, dtype=None):\n'
            '    return build_model(FAMILY, CONFIG, MTP, device=device, dtype=dtype)\n\n'
            'def load(checkpoint, **kwargs):\n'
            '    quantization = kwargs.pop("quantization", QUANTIZATION)\n'
            '    return load_model(FAMILY, CONFIG, MTP, checkpoint, quantization=quantization, **kwargs)\n')


def _instantiate(family, cfg, mtp, checkpoint=None, max_params=100_000_000):
    if checkpoint:
        return load_model(family, cfg, mtp, checkpoint)
    meta = build_model(family, cfg, mtp, device='meta')
    count = sum(p.numel() for p in meta.parameters())
    _fail(count <= max_params, f'模型有 {count:,} 个参数；请提供 --weights 或显式提高 --max-params')
    del meta
    return build_model(family, cfg, mtp)


def train_main(argv=None):
    """使用已准备的 tensor batch 训练完整模型，MTP 数据与对齐必须显式给出。"""
    parser = argparse.ArgumentParser(description='训练完整模型（含声明的 MTP 参数）')
    parser.add_argument('by1')
    parser.add_argument('--batch-file', required=True, help='torch.save 保存的输入/labels/MTP tensor 字典')
    parser.add_argument('--weights', help='初始化用的 checkpoint')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save', required=True, help='完整 HF checkpoint 的保存目录')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-params', type=int, default=100_000_000)
    args = parser.parse_args(argv)
    _fail(args.steps > 0 and args.lr > 0, 'steps/lr 必须为正')
    import torch
    import by1check
    report, info = by1check.check(args.by1)
    if report.count(by1check.E):
        return by1check.run_one(args.by1)
    family, cfg, mtp = compile_config(info)
    batch = torch.load(args.batch_file, weights_only=True, map_location=args.device)
    _fail(isinstance(batch, dict) and 'input_ids' in batch and 'labels' in batch, 'batch 必须包含 input_ids/labels')
    _fail(all(isinstance(v, torch.Tensor) for v in batch.values()), 'batch 只接受 tensor 值')
    draft = {k[4:]: v for k, v in batch.items() if k.startswith('mtp_')}
    inputs = {k: v for k, v in batch.items() if not k.startswith('mtp_')}
    if mtp:
        _fail({'input_ids', 'position_ids', 'labels'} <= set(draft), '完整 MTP 训练需要 mtp_input_ids/mtp_position_ids/mtp_labels')
        _known(draft, {'input_ids', 'position_ids', 'labels', 'embedding_indices', 'hidden_indices', 'attention_mask'}, 'MTP batch')
    else:
        _fail(not draft, '该架构没有 MTP 分支')
    model = _instantiate(family, cfg, mtp, args.weights, args.max_params).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    def gather(x, indices):
        return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))
    for step in range(args.steps):
        optimizer.zero_grad()
        output = model(**inputs, use_cache=False, output_hidden_states=bool(mtp))
        loss = output.loss
        if mtp:
            hidden = output.hidden_states[-1]
            if 'hidden_indices' in draft:
                hidden = gather(hidden, draft['hidden_indices'])
            embeds = gather(output.hidden_states[0], draft['embedding_indices']) if 'embedding_indices' in draft else None
            prediction = model.forward_mtp(draft['input_ids'], hidden, draft['position_ids'],
                inputs_embeds=embeds, labels=draft['labels'], attention_mask=draft.get('attention_mask'))
            loss = loss + prediction['loss']
        _fail(torch.isfinite(loss).item(), '训练 loss 非有限')
        loss.backward()
        optimizer.step()
        print(f'步 {step + 1} loss={loss.item():.6f}')
    model.save_pretrained(args.save)
    print(f'完整模型（含 MTP）已保存：{args.save}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='执行完整模型（Transformers 后端）')
    parser.add_argument('by1')
    parser.add_argument('--seq', type=int, default=8)
    parser.add_argument('--weights', help='本地或 HF checkpoint；未给时使用随机权重')
    parser.add_argument('--max-params', type=int, default=100_000_000, help='随机建模的参数上限；超过时需显式提高')
    args = parser.parse_args(argv)
    import by1check
    report, info = by1check.check(args.by1)
    if report.count(by1check.E):
        return by1check.run_one(args.by1)
    family, cfg, mtp = compile_config(info)
    import torch
    model = _instantiate(family, cfg, mtp, args.weights, args.max_params)
    model.eval()
    vocab = cfg.get('text_config', cfg)['vocab_size']
    ids = torch.randint(0, vocab, (1, args.seq))
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=False)
    print(f'完整 {family} 模型文本前向：{tuple(output.logits.shape)}')
    return 0
