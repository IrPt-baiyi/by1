#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1complete -- 完整模型验收：参数共享、bias、梯度、视觉、MTP 与严格权重加载。"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块可能在包内深处（src/by1/checks/…），固定层数会插错目录。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.exists(_os.path.join(_p, 'by1paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401
import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块可能在包内深处（src/by1/checks/…），固定层数会插错目录。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.exists(_os.path.join(_p, 'by1paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import copy
import json
import pathlib
import tempfile
import importlib.util
import subprocess

import by1check
import by1full
import by1skip

GALLERY = ['complete-Qwen3-0.6B.by1', 'complete-Qwen3-8B.by1',
           'complete-Qwen2.5-7B-Instruct.by1', 'complete-gpt2.by1',
           'complete-gpt-oss-20b.by1', 'complete-Qwen3.8-27B.by1',
           # **第二个机制族（T26）**：SSM/递归状态。
           # **第三个机制族（T27）**：混合栈 —— 逐层类型表。
           'complete-granite-4.0-h-350m.by1',
           # **T28：同一类的第二个成员**（不同图案）—— 这一条是"支持一类"的证人。
           'complete-granite-4.0-h-1b.by1',
           # **T29：层内并行** —— 一层里两个混合器。
           'complete-Falcon-H1-0.5B-Instruct.by1']

#: **按名字取件，不要按下标。** 加一个模型就会把 `GALLERY[-1]` / `GALLERY[5]`
#: 悄悄换成另一个模型 —— 2026-10-16 加 mamba 时真撞上：MTP/视觉那两个用例
#: 拿 `GALLERY[-1]` 当 Qwen3.8，于是去问 `MambaForCausalLM` 要 `.model`，
#: 报的是 `'MambaForCausalLM' object has no attribute 'model'`。
QWEN3_SMALL = 'complete-Qwen3-0.6B.by1'
QWEN25_7B = 'complete-Qwen2.5-7B-Instruct.by1'
GPT2 = 'complete-gpt2.by1'
GPT_OSS = 'complete-gpt-oss-20b.by1'
QWEN38 = 'complete-Qwen3.8-27B.by1'
MAMBA = 'complete-mamba-130m-hf.by1'
GRANITE = 'complete-granite-4.0-h-350m.by1'
#: T28：**同一类的第二个成员**（图案不同：注意力在 5/15/25/35，40 层）。
GRANITE_1B = 'complete-granite-4.0-h-1b.by1'
#: T29（工作稿编号）：**层内并行** —— 每层里 SSM 与注意力并存相加。
FALCON = 'complete-Falcon-H1-0.5B-Instruct.by1'

#: 夹具用的 norm eps —— **`tiny()` 与 `config_of()` 必须取同一个值**：
#: 一个是"声明编译出来的配置"，一个是"独立写出来的参考配置"，
#: 两边 eps 不同的话前向必然不一致（报出来是"前向不一致"，看不出病因）。
#: 实测踩过两次：mamba 的真实声明是 1e-5、gpt-oss 的参考配置写的是 1e-5，
#: 而其余家族是 1e-6 —— 所以这张表是**唯一事实源**，别再各写一个字面量。
TINY_EPS = {'gpt2': None, 'gpt_oss': 1e-5, 'default': 1e-6}


def tiny_eps(family):
    return TINY_EPS.get(family, TINY_EPS['default'])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def description(name):
    report, info = by1check.check(name)
    require(not report.count(by1check.E), str(report.items))
    return info


def tiny(name):
    """缩尺寸保留计算/连接类型；配置仍由声明编译，不修改生成配置来伪造通过。"""
    info = copy.deepcopy(description(name))
    family = info['arch']
    hp = info['hparams']
    # **留一份截断前的层表。** 通用那段只留前几层，而混合栈要**按类型取样**
    # （这个模型前 10 层全是 Mamba2，截断后就找不到注意力层了）——
    # 所以那一段必须从原始层表里挑，而不是从截断后的。
    original_seq = list(info['layer_seq'])
    hp.update(d_model='32', vocab='64', bos_token_id='54', eos_token_id='55')
    # **夹具把 eps 也钉住**（与 `config_of` 共用 `TINY_EPS`）：
    # mamba 的真实声明是 1e-5、gpt-oss 的参考配置是 1e-5，其余 1e-6 ——
    # 不钉的话两侧不同，前向必然不一致。
    # **真实 eps 由 `official_configs` 那条逐字段用例负责**，不靠这里。
    eps = tiny_eps(family)
    if eps is not None:                  # gpt2 走 `ln_eps`，不用这个字段
        hp['rms_eps'] = str(eps)
    info['ctx'] = 128 if family == 'gpt_oss' else 32
    keep, counts = [], {}
    for stack, mech, attrs, key, attached in info['layer_seq']:
        count = counts.get(stack, 0)
        limit = (4 if family == 'qwen3_5' else 2) if stack == 'main' else 1
        counts[stack] = count + 1
        if count >= limit:
            continue
        keep.append((stack, mech, attrs, key, attached))
    info['layer_seq'] = keep
    hp['n_layer'] = '4' if family == 'qwen3_5' else '2'
    for name_, attrs in info['mech_attrs'].items():
        kind = info['mechs'][name_]
        if kind == 'FFN':
            attrs['hidden'] = '32' if name_ == 'VF' else '64'
        if kind == 'MoE':
            attrs.update(hidden='64', experts='4', top_k='2')
    for key, attrs in info['overrides'].items():
        stack, _index, mech = key.split('|')
        if info['mechs'][mech] in ('FFN', 'MoE'):
            attrs['hidden'] = '32' if stack == 'visual' else '64'
            if info['mechs'][mech] == 'MoE':
                attrs.update(experts='4', top_k='2')
    for stack, _name, attrs, _key, _attached in keep:
        kind = info['mechs'][_name]
        if kind == 'Attention':
            hd = 8 if stack == 'visual' or family != 'qwen3_5' else 16
            q = 2 if stack == 'visual' else 4
            kv = q if family == 'gpt2' or stack == 'visual' else 2
            attrs.update(q=str(q), kv=str(kv), head_dim=str(hd))
            if 'out_dim' in attrs:
                attrs['out_dim'] = str(q * hd)
            if attrs.get('window') == '128':
                attrs['window'] = '4'
        if kind == 'Linear':
            attrs.update(k_heads='2', v_heads='6', k_dim='8', v_dim='8')
    # feed 默认属性改过后，覆盖表不需要额外重写：这些模板没有逐层 feed 覆盖。
    if family == 'gpt2':
        info['position']['default'] = 'learned(32)'
    if family == 'gpt_oss':
        info['position']['default'] = info['position']['default'].replace('original = 4096', 'original = 4')
    if family == 'qwen3_5':
        info['stack_d_model']['visual'] = 16
        info['vision'].update(patch_size='2', temporal_patch_size='2', num_position_embeddings='16',
                             mrope_section='[1, 1, 0]', image_token_id='60', video_token_id='61',
                             vision_start_token_id='62', vision_end_token_id='63')
    if family == 'mamba':
        # **SSM 的宽度不从 d_model 推**：`d_inner` 要自己缩，而且必须守住
        # `d_inner == expand × d_model`（`_compile_mamba` 的判据）；
        # `dt_rank` / `ssm_state` 同理各缩各的。
        #
        # ⚠ 两处都要改：**层的机制属性来自 `layer_seq`**（`_layers` 里
        # `mix = (mechs[name], dict(attrs))`），而 `mech_attrs` 只有 feed 用。
        # 只改后者的话 `state_size` 还是 16，而参考配置是 4 —— 形状对不上，
        # 报出来是 "Error(s) in loading state_dict"（第一版就是这么栽的）。
        expand = 2
        for stack, mech, attrs, key, attached in info['layer_seq']:
            if info['mechs'][mech] == 'SSM':
                expand = int(float(attrs.get('expand', 2)))
                attrs['ssm_state'] = '4'
        for name_, attrs in info['mech_attrs'].items():
            if info['mechs'][name_] == 'SSM':
                attrs['ssm_state'] = '4'
        hp['d_inner'] = str(expand * 32)
        hp['dt_rank'] = '2'
    if family == 'falcon_h1':
        # **层内并行（T29）的夹具**：两种机制在**同一层**里，所以"按类型取样"
        # 那一套用不上 —— 每层都是 `Parallel`，通用截断留前 2 层即可。
        # 但**支路本身要缩**：SSM2 与注意力都在 `mech_attrs` 里（不是 layer_seq
        # 的 mixer 属性），因为它们是被 `Parallel` 引用的**独立机制**。
        for name_, attrs in info['mech_attrs'].items():
            kind = info['mechs'][name_]
            if kind == 'SSM2':
                attrs.update(n_heads='2', d_head='16', d_state='8', chunk_size='8')
            elif kind == 'Attention':
                attrs.update(q='4', kv='2', head_dim='8')
            elif kind == 'FFN':
                attrs['hidden'] = '64'
        hp['max_pos'] = '32'
        hp['pad_token_id'] = '0'
        # 九个缩放 + FFN/Mamba 的开关：**夹具钉住**（与 `config_of` 那份一致），
        # 真实值由 `official_configs` 的逐字段用例负责。
        hp.update(embedding_multiplier='1.5', attention_in_multiplier='1.0',
                  attention_out_multiplier='0.5', ssm_in_multiplier='1.0',
                  ssm_out_multiplier='0.5', key_multiplier='0.5',
                  lm_head_multiplier='0.5',
                  ssm_multipliers='[1.0, 1.0, 1.0, 1.0, 1.0]',
                  mlp_multipliers='[1.0, 1.0]',
                  mlp_expansion_factor='8', mlp_bias='false',
                  projectors_bias='false', mamba_use_mlp='true',
                  mamba_norm_before_gate='false', mamba_rms_norm='false',
                  rope_theta='10000.0')
    if family == 'granitemoehybrid':
        # **夹具把 μP 四个缩放也钉住**（与 `config_of` 里写的那份一致）：
        # 两个 granite 成员的真实值不同（350m：12/0.246/3/0.015625；
        # 1b：12/0.22/6/0.0078125），而 `config_of(family, …)` 只按家族取件、
        # 分不出是哪个成员。**真实值由 `official_configs` 那条逐字段用例负责** ——
        # 和 `tiny_eps` 是同一个套路：夹具钉住，判据在别处。
        hp.update(embedding_multiplier='12', residual_multiplier='0.246',
                  logits_scaling='3', attention_multiplier='0.015625')
        # **按类型取样，不按层号截断。** 通用那段只留前 2 层，而这个家族
        # 前 10 层（1b 是前 5 层）全是 Mamba2 —— 只留前 2 层会触发
        # `_compile_moe_hybrid` 的"两种机制都要出现"（那是判据，不是 bug：
        # 混合栈的定义就是两种都在）。所以从**截断前的层表**里各取几个：
        # 2 个 SSM2 + 1 个 Attention。
        ssm_keep, attn_keep = [], None
        for entry in original_seq:
            stack, mech = entry[0], entry[1]
            if stack != 'main':
                continue
            kind = info['mechs'][mech]
            if kind == 'SSM2' and len(ssm_keep) < 2:
                ssm_keep.append(entry)
            elif kind == 'Attention' and attn_keep is None:
                attn_keep = entry
        info['layer_seq'] = ssm_keep + ([attn_keep] if attn_keep else [])
        hp['n_layer'] = str(len(info['layer_seq']))
        # 上下文与共享前馈一起缩（`max_pos` 必须显式声明，见 by1full 的混合栈分支）。
        hp['max_pos'] = '32'
        hp['shared_hidden'] = '64'
        # 词表缩到 64，而声明里的 `pad_token_id` 是 100256（大词表的值）——
        # 不缩的话 PyTorch 会报 `Padding_idx must be within num_embeddings`。
        hp['pad_token_id'] = '0'
        # **SSM2 与注意力都要缩**：`expand × d_model == n_heads × d_head`
        # （2 × 32 = 4 × 16）、`q × head_dim == d_model`（4 × 8 = 32）。
        # 注意：取回来的注意力层来自 `original_seq`，通用那段没碰过它 ——
        # 不缩的话它还带着 12 × 64（= 768）。
        for entry in info['layer_seq']:
            attrs = entry[2]
            if info['mechs'][entry[1]] == 'SSM2':
                attrs.update(n_heads='4', d_head='16', d_state='8', chunk_size='8')
            elif info['mechs'][entry[1]] == 'Attention':
                attrs.update(q='4', kv='2', head_dim='8')
                if 'out_dim' in attrs:
                    attrs['out_dim'] = '32'
        for name_, attrs in info['mech_attrs'].items():
            if info['mechs'][name_] == 'SSM2':
                attrs.update(n_heads='4', d_head='16', d_state='8', chunk_size='8')
            elif info['mechs'][name_] == 'Attention':
                attrs.update(q='4', kv='2', head_dim='8')
    return by1full.compile_config(info)


def config_of(family, tied, T):
    """独立写出的参考**配置**：不使用 by1 的配置转换器。
    `reference()` 用它建模型；影子冒烟（by1shadow）也用它 ——
    同一个意思只写一遍，两处漂移会互相打脸。"""
    common = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, intermediate_size=64,
                  max_position_embeddings=32, tie_word_embeddings=tied,
                  bos_token_id=54, eos_token_id=55, rms_norm_eps=tiny_eps(family),
                  attention_dropout=0)
    if family == 'qwen3':
        return T.Qwen3Config(rope_parameters={'rope_type': 'default', 'rope_theta': 1e6}, **common)
    if family == 'qwen2':
        return T.Qwen2Config(rope_parameters={'rope_type': 'default', 'rope_theta': 1e6}, **common)
    if family == 'gpt2':
        return T.GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4,
                            n_inner=64, n_positions=32, layer_norm_epsilon=1e-5,
                            activation_function='gelu_new', tie_word_embeddings=True,
                            attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1,
                            bos_token_id=54, eos_token_id=55)
    if family == 'gpt_oss':
        common.update(max_position_embeddings=128, rms_norm_eps=tiny_eps(family),
                      attention_bias=True)
        return T.GptOssConfig(num_local_experts=4, num_experts_per_tok=2,
                              sliding_window=4,
                              layer_types=['sliding_attention', 'full_attention'],
                              rope_parameters={'rope_type': 'yarn', 'rope_theta': 150000.,
                                               'factor': 32.,
                                               'original_max_position_embeddings': 4,
                                               'beta_fast': 32., 'beta_slow': 1.,
                                               'truncate': False}, **common)
    if family == 'mamba':
        # **mamba 用不了 `common`** —— 它那一代的字段名是另一套
        # （`d_model` / `d_inner` / `n_layer` / `state_size` / `conv_kernel`），
        # 而且没有 num_attention_heads 这些。照着 tiny() 缩出来的尺寸写死。
        return by1full._tclass('MambaConfig')(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, hidden_act='silu', initializer_range=0.1,
            layer_norm_epsilon=tiny_eps(family), use_cache=True,
            d_model=32, d_inner=64, n_layer=2,
            state_size=4, conv_kernel=4, expand=2,
            time_step_rank=2, time_step_scale=1.0, time_step_min=0.001,
            time_step_max=0.1, time_step_init_scheme='random',
            time_step_floor=0.0001, rms_norm=True, residual_in_fp32=True,
            fused_add_norm=True, rescale_prenorm_residual=False,
            use_bias=False, use_conv_bias=True, pad_vocab_size_multiple=8,
            ssm_cfg={}, bos_token_id=54, eos_token_id=55, pad_token_id=0,
            tie_word_embeddings=tied)
    if family == 'falcon_h1':
        # **层内并行（T29）**：字段名是这一族自己那套（九个缩放 + `mamba_*` + `mlp_*`）。
        # 值必须与 `tiny()` 里钉住的那份**逐位一致**，否则前向必然不同。
        return by1full._tclass('FalconH1Config')(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=32, rms_norm_eps=tiny_eps(family),
            hidden_act='silu', attention_bias=False, attention_dropout=0.0,
            embedding_multiplier=1.5, attention_in_multiplier=1.0,
            attention_out_multiplier=0.5, ssm_in_multiplier=1.0,
            ssm_out_multiplier=0.5, key_multiplier=0.5, lm_head_multiplier=0.5,
            ssm_multipliers=[1.0] * 5, mlp_multipliers=[1.0, 1.0],
            intermediate_size=64, mlp_bias=False, mlp_expansion_factor=8,
            projectors_bias=False,
            mamba_n_heads=2, mamba_d_head=16, mamba_d_state=8, mamba_n_groups=1,
            mamba_chunk_size=8, mamba_d_conv=4, mamba_expand=2, mamba_d_ssm=32,
            mamba_conv_bias=True, mamba_proj_bias=False, mamba_use_mlp=True,
            mamba_norm_before_gate=False, mamba_rms_norm=False,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0},
            initializer_range=0.02, use_cache=True,
            bos_token_id=54, eos_token_id=55, pad_token_id=0,
            tie_word_embeddings=tied)
    if family == 'granitemoehybrid':
        # **混合栈用不了 `common`**：主干的形状来自**逐层类型表**，
        # 而逐层类型表要与 `tiny()` 取样后的顺序一致（2 个递归层 + 1 个注意力层）。
        # μP 四个缩放照声明写死 —— 两边必须逐位相同，否则前向必然不同。
        return T.GraniteMoeHybridConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2,
            layer_types=['linear_attention', 'linear_attention', 'full_attention'],
            max_position_embeddings=32,
            rms_norm_eps=tiny_eps(family), hidden_act='silu',
            attention_bias=False, attention_dropout=0.0,
            attention_multiplier=0.015625, embedding_multiplier=12,
            residual_multiplier=0.246, logits_scaling=3,
            intermediate_size=64, shared_intermediate_size=64,
            num_local_experts=0, num_experts_per_tok=0,
            output_router_logits=False, router_aux_loss_coef=0.01,
            mamba_n_heads=4, mamba_d_head=16, mamba_d_state=8, mamba_n_groups=1,
            mamba_chunk_size=8, mamba_d_conv=4, mamba_expand=2,
            mamba_conv_bias=True, mamba_proj_bias=False,
            position_embedding_type='nope',
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0},
            normalization_function='rmsnorm', init_method='mup',
            initializer_range=0.1, use_cache=True,
            bos_token_id=54, eos_token_id=55, pad_token_id=0,
            tie_word_embeddings=tied)
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                max_position_embeddings=32, tie_word_embeddings=False, bos_token_id=54, eos_token_id=55,
                layer_types=['linear_attention'] * 3 + ['full_attention'], linear_num_key_heads=2,
                linear_num_value_heads=6, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4,
                rope_parameters={'rope_type': 'default', 'rope_theta': 1e7, 'partial_rotary_factor': 0.25,
                                 'mrope_section': [1, 1, 0], 'mrope_interleaved': True})
    vision = dict(depth=1, hidden_size=16, num_heads=2, intermediate_size=32, out_hidden_size=32,
                  patch_size=2, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16,
                  in_channels=3, hidden_act='gelu_pytorch_tanh')
    return T.Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=60,
                           video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)


def reference(family, tied, torch, T):
    """独立写出的参考配置：不使用 by1 的配置转换器。"""
    if family == 'qwen3':
        return T.Qwen3ForCausalLM(config_of(family, tied, T))
    if family == 'qwen2':
        return T.Qwen2ForCausalLM(config_of(family, tied, T))
    if family == 'gpt2':
        return T.GPT2LMHeadModel(config_of(family, tied, T))
    if family == 'gpt_oss':
        return T.GptOssForCausalLM(config_of(family, tied, T))
    if family == 'mamba':
        # 同样要用 `_tclass`（没导出到顶层）。
        return by1full._tclass('MambaForCausalLM')(config_of(family, tied, T))
    if family == 'granitemoehybrid':
        # 这一族**顶层有导出**，直接取。
        return T.GraniteMoeHybridForCausalLM(config_of(family, tied, T))
    if family == 'falcon_h1':
        return by1full._tclass('FalconH1ForCausalLM')(config_of(family, tied, T))
    return T.Qwen3_5ForConditionalGeneration(config_of(family, tied, T))


def inputs(family, torch, video=False):
    if family != 'qwen3_5':
        return dict(input_ids=torch.tensor([[5, 6, 7, 8]]), use_cache=False)
    if video:
        return dict(input_ids=torch.tensor([[5, 62, 61, 63, 62, 61, 63, 7]]),
                    pixel_values_videos=torch.randn(8, 24), video_grid_thw=torch.tensor([[2, 2, 2]]),
                    mm_token_type_ids=torch.tensor([[0, 0, 2, 0, 0, 2, 0, 0]]),
                    use_cache=False, output_hidden_states=True)
    return dict(input_ids=torch.tensor([[5, 62, 61 if video else 60, 63, 7, 8]]),
                pixel_values=torch.randn(4, 24), image_grid_thw=torch.tensor([[1, 2, 2]]),
                mm_token_type_ids=torch.tensor([[0, 0, 2 if video else 1, 0, 0, 0]]),
                use_cache=False, output_hidden_states=True)


def mtp_formula(model, ids, hidden, positions, torch, inputs_embeds=None):
    """按 vLLM 的连接公式独立组合；不调用 forward_mtp，也不复用 MTP 的 norm.forward。"""
    def norm(x, weight):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())).to(x.dtype)
    m = model.mtp
    embeddings = model.get_input_embeddings()(ids) if inputs_embeds is None else inputs_embeds
    emb = norm(embeddings, m.pre_fc_norm_embedding.weight)
    hid = norm(hidden, m.pre_fc_norm_hidden.weight)
    x = torch.nn.functional.linear(torch.cat((emb, hid), -1), m.fc.weight)
    rotary = model.model.language_model.rotary_emb(x, positions.unsqueeze(0).expand(3, -1, -1))
    n = ids.shape[1]
    mask = torch.zeros(n, n, device=x.device).masked_fill(
        torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1), torch.finfo(x.dtype).min)[None, None]
    block = m.layers[0]
    x = x + block.self_attn(hidden_states=norm(x, block.input_layernorm.weight),
                            position_embeddings=rotary, attention_mask=mask)[0]
    v = norm(x, block.post_attention_layernorm.weight)
    x = x + block.mlp.down_proj(torch.nn.functional.silu(block.mlp.gate_proj(v)) * block.mlp.up_proj(v))
    return torch.nn.functional.linear(norm(x, m.norm.weight), model.get_output_embeddings().weight)


def main():
    try:
        import torch
        import transformers as T
    except ImportError as ex:
        return by1skip.skip(f'完整模型验收需要 torch/transformers/safetensors：{ex}')
    if importlib.util.find_spec('safetensors') is None:
        return by1skip.skip('完整模型验收需要 safetensors')
    torch.set_num_threads(1)
    failures = []
    def case(name, fn):
        try:
            fn()
            print('  ok ' + name, flush=True)
        except Exception as ex:
            failures.append(name)
            print('  !! %s: %s' % (name, ex), flush=True)

    def manifest():
        family, cfg, mtp = by1full.compile_config(description(QWEN38))
        model = by1full.build_model(family, cfg, mtp, device='meta')
        ref = json.loads(pathlib.Path(by1paths.REFS, 'Qwen__Qwen3.8-27B.tensors.json').read_text(encoding='utf-8'))
        got = {k: list(v.shape) for k, v in model.state_dict().items()}
        require(set(got) == set(ref), 'Qwen3.8 参数清单不完整')
        require(all(got[k] == v['shape'] for k, v in ref.items()), 'Qwen3.8 真实参数形状不一致')
        require(model.mtp.embed_tokens is model.get_input_embeddings(), 'MTP 没有共享 embedding 对象')
        require(model.mtp.lm_head is model.get_output_embeddings(), 'MTP 没有共享输出头')
    case('Qwen3.8 真尺寸 1199 张量与 MTP 共享关系', manifest)

    def official_configs():
        # **2026-10-16 补 P0 模型（GALLERY[0] = complete-Qwen3-0.6B）。**
        # 这一格原来是"手工比过"：P0 闭环报告里那句"关键字段逐项吻合"没有
        # 可重跑命令。IR 路径早就有 by1verify 的 config 逐字段行，而
        # complete-* 走 Transformers 后端、没有 field 映射表 —— 于是漏在两边之间。
        # **refs/ 里有哪个模型的 config，这里就能比哪个。**
        for name, reference_file in [(QWEN3_SMALL, 'Qwen__Qwen3-0.6B.config.json'),
                                     (GPT2, 'openai-community__gpt2.config.json'),
                                     (GPT_OSS, 'openai__gpt-oss-20b.config.json'),
                                     (QWEN38, 'Qwen__Qwen3.8-27B.config.json'),
                                     # T26：第二架构（SSM）。它的两条 family 用例
                                     # （tiny 的缩尺寸 + config_of 的独立参考配置）
                                     # 与家族无关的那部分见 tiny()/config_of() 的 mamba 分支。
                                     (MAMBA, 'state-spaces__mamba-130m-hf.config.json'),
                                     # T27：第三架构（**混合栈**：Attention 与 SSM2 同栈）。
                                     # **还没进 GALLERY** —— 那张表要写 tiny/config_of/
                                     # reference 三条分支，下一步做（T27 记录已移出仓库，见 git 历史）。
                                     # config 逐字段与家族无关，先接上。
                                     (GRANITE, 'ibm-granite__granite-4.0-h-350m.config.json'),
                                     # T28：同类第二个成员（图案与 μP 都不同）。
                                     (GRANITE_1B, 'ibm-granite__granite-4.0-h-1b.config.json'),
                                     # T29：**层内并行**（九个缩放标量逐个对上）。
                                     (FALCON, 'tiiuae__Falcon-H1-0.5B-Instruct.config.json')]:
            family, cfg, _ = by1full.compile_config(description(name))
            raw = json.loads(pathlib.Path(by1paths.REFS, reference_file).read_text(encoding='utf-8'))
            # **用 by1full._tclass**：mamba 系的两个类没导出到顶层，
            # `getattr(T, ...)` 在这一行会 AttributeError（T26 实测）。
            actual = by1full._tclass(by1full.FAMILIES[family][0])(**raw)
            for section in ('text_config', 'vision_config'):
                if section in cfg:
                    by1full._check_config(cfg[section], getattr(actual, section).to_dict(), section)
            by1full._check_config({k: v for k, v in cfg.items() if k not in ('text_config', 'vision_config')},
                                 actual.to_dict(), 'config')
    case('Qwen3-0.6B/GPT-2/gpt-oss/Qwen3.8/mamba/granite 官方配置的有效语义', official_configs)

    for filename in GALLERY:
        def numerical(filename=filename):
            family, cfg, mtp = tiny(filename)
            mine = by1full.build_model(family, cfg, mtp)
            tied = cfg.get('text_config', cfg)['tie_word_embeddings']
            require((mine.get_input_embeddings().weight is mine.get_output_embeddings().weight) == tied,
                    '共享关系是复制数值而非同一个参数')
            ref = reference(family, tied, torch, T)
            ref.config._attn_implementation = 'eager'
            if family == 'qwen3_5':
                ref.config.text_config._attn_implementation = 'eager'
                ref.config.vision_config._attn_implementation = 'eager'
            state = {k: v for k, v in mine.state_dict().items() if not k.startswith('mtp.')}
            ref.load_state_dict(state, strict=True)
            args = inputs(family, torch)
            args['labels'] = args['input_ids']
            # 同一随机种子验证训练时 dropout 和梯度；视觉也参加反向传播。
            mine.train(); ref.train()
            torch.manual_seed(42); actual = mine(**args)
            torch.manual_seed(42); expected = ref(**args)
            require(torch.allclose(actual.logits, expected.logits, atol=2e-6, rtol=2e-5), '前向不一致')
            require(torch.allclose(actual.loss, expected.loss, atol=2e-6), '训练 loss 不一致')
            actual.loss.backward()
            expected.loss.backward()
            ref_params = dict(ref.named_parameters())
            for name, param in mine.named_parameters():
                if name.startswith('mtp.'):
                    continue
                other = ref_params[name]
                require(param.grad is not None and other.grad is not None, f'{name} 梯度缺失')
                require(torch.allclose(param.grad, other.grad, atol=2e-6, rtol=2e-4), f'{name} 梯度不一致')
            if family == 'qwen2':
                att = mine.model.layers[0].self_attn
                require(att.q_proj.bias is not None and att.o_proj.bias is None, 'QKV/output bias 被合并')
            if family == 'qwen3_5':
                mine.eval(); ref.eval()
                args = inputs(family, torch, video=True)
                require(torch.allclose(mine(**args).logits, ref(**args).logits, atol=2e-6, rtol=2e-5), '视频前向不一致')
        case(filename + ' 前向/反向（独立参考配置）', numerical)

    def mtp_execution():
        family, cfg, mtp = tiny(QWEN38)
        model = by1full.build_model(family, cfg, mtp).eval()
        args = inputs(family, torch)
        recorded = []
        hook = model.model.language_model.norm.register_forward_hook(lambda _m, _a, out: recorded.append(out))
        base = model(**args)
        hook.remove()
        hidden = base.hidden_states[-1]
        require(torch.equal(hidden, recorded[-1]), 'MTP 输入不是主干最终归一化后的 hidden')
        ids = torch.roll(args['input_ids'], -1, 1)
        positions = torch.arange(ids.shape[1])[None] + 1
        draft = model.forward_mtp(ids, hidden, positions, labels=ids)
        expected = mtp_formula(model, ids, hidden, positions, torch)
        require(torch.allclose(draft['logits'], expected, atol=2e-6, rtol=2e-5), 'MTP 连接公式不一致')
        next_embeddings = torch.roll(base.hidden_states[0], -1, 1)
        visual_draft = model.forward_mtp(ids, hidden, positions, inputs_embeds=next_embeddings)['logits']
        require(torch.allclose(visual_draft, mtp_formula(model, ids, hidden, positions, torch, next_embeddings),
                               atol=2e-6, rtol=2e-5), '多模态 MTP embedding 连接不一致')
        draft['loss'].backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mtp.parameters()), 'MTP 有未训练的参数')
        require(model.model.visual.patch_embed.proj.weight.grad is not None, 'MTP 梯度没有经过视觉/文本主干')
        before = draft['logits'].detach().clone()
        with torch.no_grad():
            model.mtp.fc.weight.zero_()
            after = model.forward_mtp(ids, hidden.detach(), positions)['logits']
            require(not torch.allclose(before, after), 'MTP 权重没有影响输出')
            require(torch.equal(model(**args).logits, base.logits), 'MTP 被错误串进主干')
        changed = dict(args)
        changed['pixel_values'] = args['pixel_values'] * -10
        require(not torch.allclose(model(**changed).logits, base.logits), '图像没有进入文本主干')
    case('图像→主干→MTP 的执行、独立公式和梯度', mtp_execution)

    def checkpoint():
        family, cfg, mtp = tiny(QWEN38)
        model = by1full.build_model(family, cfg, mtp).eval()
        with tempfile.TemporaryDirectory(prefix='by1-full-', dir=by1paths.ROOT) as tmp:
            root = pathlib.Path(tmp).resolve()
            require(root.parent == pathlib.Path(by1paths.ROOT).resolve(), '临时目录不在工作区')
            good, broken = root / 'good', root / 'broken'
            model.save_pretrained(good, max_shard_size='20KB')
            loaded = by1full.load_model(family, cfg, mtp, str(good), local_files_only=True).eval()
            # **"存盘/读回无损"的本意是权重逐位相同** —— 这一条不放松，
            # 用 `torch.equal` 断言 state_dict（safetensors 往返就是逐位的）。
            sd_a, sd_b = model.state_dict(), loaded.state_dict()
            require(set(sd_a) == set(sd_b), '存盘读回后键不一致')
            drift = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])]
            require(not drift, '权重不再逐位：%s' % drift[:4])
            # **前向允许浮点舍入级差异。** 实测（2026-10-17，诊断腿的公开注解）：
            # Linux 上 max|Δ| = 7.451e-08、无 NaN —— **1 ulp 级**，来自两个模型
            # 实例走的归约/内核略有差别（Windows 上恰好逐位）。
            # `torch.equal` 对"重新实例化的模型"过强：真正的判据是
            # **权重逐位（上）+ 前向在浮点舍入内一致**，容差按实测取 1e-6
            # （对 7.45e-08 是 13 倍余量）。**差多少每次都报**，不许变大不吭声。
            args = inputs(family, torch)
            a_logits, b_logits = model(**args).logits, loaded(**args).logits
            delta = (a_logits - b_logits).abs().max().item()
            require(delta <= 1e-6 and not bool(a_logits.isnan().any())
                    and not bool(b_logits.isnan().any()),
                    '加载后主干输出变化：max|Δ|=%.3e · NaN %s/%s'
                    % (delta, bool(a_logits.isnan().any()),
                       bool(b_logits.isnan().any())))
            ids = args['input_ids']; positions = torch.arange(ids.shape[1])[None]
            h = model(**args).hidden_states[-1]
            m_logits = model.forward_mtp(ids, h, positions)['logits']
            l_logits = loaded.forward_mtp(ids, h, positions)['logits']
            delta = (m_logits - l_logits).abs().max().item()
            require(delta <= 1e-6, '加载后 MTP 输出变化：max|Δ|=%.3e' % delta)
            state = dict(model.state_dict()); state.pop('mtp.fc.weight')
            model.save_pretrained(broken, state_dict=state)
            try:
                by1full.load_model(family, cfg, mtp, str(broken), local_files_only=True)
            except ValueError as ex:
                require('mtp.fc.weight' in str(ex), '漏权重未点名')
            else:
                raise RuntimeError('缺少 MTP 权重却加载成功')
    case('完整 checkpoint 分片往返；缺 MTP 必须拒绝', checkpoint)

    def cli_training():
        text = '''model TinyFull {
 backend transformers
 arch qwen3_5
 ctx 32
 hparams { d_model = 32, vocab = 64, rms_eps = 1e-6, norm = zero_centered, tie_word_embeddings = false }
 mech A : Attention { heads = { q = 4, kv = 2, head_dim = 16 }, out_dim = 64, q_gate = true, qk_norm = per_head, partial = 0.25 }
 mech D : Linear { k_heads = 2, k_dim = 8, v_heads = 6, v_dim = 8, conv_kernel = 4 }
 mech F : FFN { hidden = 64 }
 stack main { pattern = 3 * D + A, main[:] >> F }
 stack mtp { aux = true, pattern = A, mtp[:] >> F }
 mech V : Attention { heads = { q = 2, kv = 2, head_dim = 8 }, bias = true, qkv_fused = true }
 mech VF : FFN { hidden = 32, act = gelu_new, gate = none, bias = true }
 stack visual { aux = true, d_model = 16, pattern = V, visual[:] >> VF }
 position { default = rope(base = 10000000, pairing = half) }
 vision {
 patch_size = 2, temporal_patch_size = 2, spatial_merge_size = 2, num_position_embeddings = 16, in_channels = 3
 mrope_section = [1,1,0], image_token_id = 60, video_token_id = 61, vision_start_token_id = 62, vision_end_token_id = 63
 }
 mtp { share_embeddings = true, share_head = true }
}
'''
        with tempfile.TemporaryDirectory(prefix='by1-cli-', dir=by1paths.ROOT) as tmp:
            root = pathlib.Path(tmp).resolve()
            require(root.parent == pathlib.Path(by1paths.ROOT).resolve(), '临时目录不在工作区')
            path = root / 'tiny.by1'; path.write_text(text, encoding='utf-8')
            spec = by1full.compile_config(description(str(path)))
            original = by1full.build_model(*spec)
            original.save_pretrained(root / 'initial')
            batch = inputs('qwen3_5', torch)
            batch.pop('use_cache'); batch.pop('output_hidden_states')
            ids = batch['input_ids']; n = ids.shape[1]
            batch['labels'] = ids.clone()
            batch['mtp_input_ids'] = ids.roll(-1, 1)
            batch['mtp_labels'] = ids.roll(-2, 1); batch['mtp_labels'][:, -2:] = -100
            batch['mtp_position_ids'] = torch.arange(n)[None] + 1
            batch['mtp_embedding_indices'] = torch.arange(n)[None].roll(-1, 1)
            torch.save(batch, root / 'batch.pt')
            command = [_sys.executable, by1paths.tool('by1.py'), 'train', str(path), '--batch-file',
                       str(root / 'batch.pt'), '--weights', str(root / 'initial'), '--steps', '1',
                       '--save', str(root / 'trained')]
            result = subprocess.run(command, capture_output=True, encoding='utf-8', timeout=120)
            require(result.returncode == 0, result.stdout + result.stderr)
            trained = by1full.load_model(*spec, str(root / 'trained'), local_files_only=True)
            require(not torch.equal(original.mtp.fc.weight, trained.mtp.fc.weight), '训练没有更新 MTP')
    case('by1 train 完整图像/MTP batch 并保存全部权重', cli_training)

    for name, change in [
        ('缺共享声明', lambda i: i['hparams'].pop('tie_word_embeddings')),
        ('未知字段', lambda i: i['hparams'].update(something='true')),
        ('错误 QKV bias', lambda i: i['layer_seq'][0][2].update(qkv_bias='false')),
    ]:
        def rejected(change=change):
            info = copy.deepcopy(description(QWEN25_7B)); change(info)
            try:
                by1full.compile_config(info)
            except ValueError:
                return
            raise RuntimeError('错误声明被接受')
        case(name + '被拒绝', rejected)

    # ── T26 加的五条规则：**每条配一个必红反例**（否则规则可能永远绿）──
    # （`MAMBA` 用模块级那个常量 —— 在 main() 里再赋一次会把它变成 main 的
    #  局部名，而 `official_configs` 是嵌套函数、早于这次赋值被调用，
    #  报的是 "cannot access free variable 'MAMBA'"。）

    def _mamba():
        return copy.deepcopy(description(MAMBA))

    def ssm_with_feed():
        """SSM 层不能再挂 FFN/MoE —— Mamba1 的 mixer 就是整层。"""
        info = _mamba()
        info['mechs']['MLP'] = 'FFN'
        info['mech_attrs']['MLP'] = {'hidden': '1536'}
        info['layer_seq'][0][4].append('MLP')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('SSM 层挂上 FFN 竟然被接受')
    case('SSM 层挂 FFN 被拒绝', ssm_with_feed)

    def d_inner_mismatch():
        """`d_inner` 必须等于 `expand × d_model`（1516 那种手滑要拦住）。"""
        info = _mamba()
        info['hparams']['d_inner'] = '1024'
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('d_inner 与 expand x d_model 不等竟然被接受')
    case('d_inner 不等被拒绝', d_inner_mismatch)

    def mamba_non_ssm():
        """mamba 的主层只能是 SSM —— 换成 Attention 必须报错（且不是因为属性表）。"""
        info = _mamba()
        name = info['layer_seq'][0][1]
        info['mechs'][name] = 'Attention'
        info['mech_attrs'][name] = {'q': '4', 'kv': '2', 'head_dim': '32',
                                    'structural': []}
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('mamba 主层换成 Attention 竟然被接受')
    case('mamba 主层非 SSM 被拒绝', mamba_non_ssm)

    def tclass_fallback():
        """`_tclass`：**顶层取不到不等于没有** —— mamba 的两个类就没导出。"""
        import transformers as T
        if getattr(T, 'MambaForCausalLM', None) is None:
            cls = by1full._tclass('MambaForCausalLM')
            require(cls.__name__ == 'MambaForCausalLM', '_tclass 解析错了')
        require(by1full._tclass('Qwen3Config').__name__ == 'Qwen3Config',
                '_tclass 对顶层类也应当直接取到')
    case('_tclass 顶层取不到时走 auto 映射', tclass_fallback)

    def modify_tensor_rules():
        """转换期改写（mamba 两条）：**改值**与**改形状** —— 少了就是"算另一个模型"。"""
        import numpy as np
        import torch
        import by1llama
        src = np.array([[1.0, 2.0]], dtype=np.float32)
        a = np.asarray(by1llama._modify_tensor('mamba', 'x.A_log', 'blk.0.ssm_a', src))
        want = (-torch.exp(torch.as_tensor(src))).numpy()
        require(np.array_equal(a, want), 'A_log -> A 不是 -exp(A_log)')
        conv = np.zeros((3, 1, 4), dtype=np.float32)
        got = by1llama._modify_tensor('mamba', 'x.conv1d.weight', 'blk.0.ssm_conv1d.weight', conv)
        require(np.asarray(got).shape == (3, 4), 'conv1d 没有 squeeze')
        require(by1llama._modify_tensor('qwen3', 'x.A_log', 'x', src) is src,
                '非 mamba 架构不该被改写')
    case('转换期改写（A_log 改值 / conv1d 改形状）', modify_tensor_rules)

    # ── T27 加的五条规则：混合栈的**必红反例** ──────────────────────

    def _granite():
        return copy.deepcopy(description(GRANITE))

    def _set_mech_attr(info, mech_name, **kw):
        """机制属性**两处都要改**：`layer_seq` 里那份是建配置用的，
        `mech_attrs` 那份只有 feed 用（T26 在这里栽过一次）。"""
        info['mech_attrs'][mech_name].update({k: str(v) for k, v in kw.items()})
        for entry in info['layer_seq']:
            if entry[1] == mech_name:
                entry[2].update({k: str(v) for k, v in kw.items()})

    def hybrid_single_mechanism():
        """混合栈里**两种机制都要出现** —— 删掉注意力层必须被拒。"""
        info = _granite()
        info['layer_seq'] = [e for e in info['layer_seq']
                             if info['mechs'][e[1]] != 'Attention']
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('只剩一种机制竟然被接受（那该用单机制家族）')
    case('混合栈只剩一种机制被拒绝', hybrid_single_mechanism)

    def hybrid_missing_mup():
        """μP 缩放**没有默认值** —— 少一个必须被拒（它改数值不改形状）。"""
        info = _granite()
        info['hparams'].pop('logits_scaling')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('少一个 μP 缩放竟然被接受')
    case('缺一个 μP 缩放被拒绝', hybrid_missing_mup)

    def hybrid_ssm_width():
        """`expand × d_model == n_heads × d_head` —— 不等必须被拒。"""
        info = _granite()
        _set_mech_attr(info, 'Mixer', n_heads=16)
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('SSM2 宽度对不上竟然被接受')
    case('SSM2 宽度对不上被拒绝', hybrid_ssm_width)

    def hybrid_attn_width():
        """混合栈的注意力：`q × head_dim == d_model` —— 不等必须被拒。"""
        info = _granite()
        _set_mech_attr(info, 'Attn', q=8)
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('注意力宽度对不上竟然被接受')
    case('混合栈注意力宽度对不上被拒绝', hybrid_attn_width)

    def hybrid_moe_feed():
        """这一族每层挂的是 dense FFN —— 换成 MoE 必须被拒。"""
        info = _granite()
        info['mechs']['MLP'] = 'MoE'
        info['mech_attrs']['MLP'] = {'experts': '4', 'top_k': '2', 'hidden': '64'}
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('混合栈换成 MoE 前馈竟然被接受')
    case('混合栈换成 MoE 前馈被拒绝', hybrid_moe_feed)

    # ── T29（层内并行）的五条规则：**每条一个必红反例** ──────────────

    def _falcon():
        return copy.deepcopy(description(FALCON))

    def _del_mech_attr(info, mech_name, key):
        info['mech_attrs'][mech_name].pop(key, None)
        for entry in info['layer_seq']:
            if entry[1] == mech_name:
                entry[2].pop(key, None)

    def parallel_missing_branch():
        """`Parallel` 必须写全 `a` 与 `b` —— 少一条支路必须被拒。"""
        info = _falcon()
        _del_mech_attr(info, 'Both', 'b')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('少一条支路的 Parallel 竟然被接受')
    case('Parallel 少一条支路被拒绝', parallel_missing_branch)

    def parallel_unknown_branch():
        """`a` / `b` 指向**未声明的机制**必须被拒（不是默默当字符串）。"""
        info = _falcon()
        _set_mech_attr(info, 'Both', a='NotAMechanism')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('Parallel 指向不存在的机制竟然被接受')
    case('Parallel 指向不存在的机制被拒绝', parallel_unknown_branch)

    def parallel_same_kind():
        """两条支路必须**不同类**（这一族恰好是 SSM2 + Attention）。"""
        info = _falcon()
        _set_mech_attr(info, 'Both', a='Mixer', b='Mixer')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('两条支路是同一类竟然被接受')
    case('Parallel 两条支路同类被拒绝', parallel_same_kind)

    def parallel_not_parallel():
        """falcon_h1 的每一层都必须是 `Parallel` —— 换成单机制必须被拒。"""
        info = _falcon()
        # `layer_seq` 里是**元组**，不能就地改；重建一份。
        info['layer_seq'] = [(e[0], 'Attn', {'q': '8', 'kv': '2', 'head_dim': '64'},
                              e[3], e[4]) for e in info['layer_seq']]
        info['mechs']['Attn'] = 'Attention'
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('falcon_h1 的主层换成单机制竟然被接受')
    case('层内并行换成单机制被拒绝', parallel_not_parallel)

    def falcon_missing_scale():
        """九个缩放**没有默认值** —— 少一个必须被拒。"""
        info = _falcon()
        info['hparams'].pop('key_multiplier')
        try:
            by1full.compile_config(info)
        except ValueError:
            return
        raise RuntimeError('少一个缩放标量竟然被接受')
    case('缺一个缩放标量被拒绝', falcon_missing_scale)

    print('\n  [%s] 完整模型：%s' % ('FAIL' if failures else 'PASS', ', '.join(failures) if failures else '全部通过'))
    return 1 if failures else 0


if __name__ == '__main__':
    _sys.exit(main())
