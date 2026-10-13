#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1full -- 从完整架构声明生成 Transformers 模型，包含视觉与独立 MTP 分支。

编译阶段只用标准库。执行阶段使用 Transformers 的模型/算子实现。
Qwen 的 MTP 拼接、归一化与共享关系在本模块实现，不丢弃 checkpoint 的 mtp.*。
MTP 组合依据 vLLM qwen3_5_mtp.py；它的输入 hidden_states 是主干最终归一化后的值。
"""
import argparse
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
}
MECH_FIELDS = {
    "Attention": {"q", "kv", "head_dim", "out_dim", "window", "qk_norm", "bias",
                  "attn_bias", "qkv_bias", "out_bias", "q_gate", "partial", "qkv_fused",
                  "sink", "structural"},
    "Linear": {"k_heads", "k_dim", "v_heads", "v_dim", "conv_kernel", "act", "structural"},
    "FFN": {"hidden", "act", "gate", "bias", "structural"},
    "MoE": {"experts", "top_k", "hidden", "shared", "router_bias", "expert_bias",
            "routing", "act", "swiglu_limit", "alpha", "structural"},
}


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
        _fail(len(feeds) == 1, f"{stack}[{index}] 必须挂载一个 FFN/MoE")
        for kind, a in [mix] + feeds:
            _fail(kind in MECH_FIELDS, f"完整后端尚不支持 {kind}")
            _known(a, MECH_FIELDS[kind], kind)
        result.setdefault(stack, []).append((mix, feeds[0]))
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
                'attention_dropout', 'resid_dropout', 'embed_dropout', 'initializer_range'}, 'hparams')
    _fail('tie_word_embeddings' in hp, "完整架构必须显式声明 tie_word_embeddings")
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
        _fail(float(feeds.get('alpha', 1.702)) == 1.702, "gpt_oss 的 alpha 为 1.702")
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


def _model_class(family, mtp_spec):
    import transformers as T
    base = getattr(T, FAMILIES[family][1])
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
    cfg = getattr(T, FAMILIES[family][0])(**copy.deepcopy(configuration))
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


def load_model(family, configuration, mtp_spec, checkpoint, **kwargs):
    """使用声明的配置加载 HF/checkpoint；遗漏 MTP 或多余权重都会报错。"""
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
    cfg = getattr(T, FAMILIES[family][0])(**declared)
    cfg._attn_implementation = 'eager'
    if family == 'qwen3_5':
        cfg.text_config._attn_implementation = 'eager'
        cfg.vision_config._attn_implementation = 'eager'
    if hasattr(checkpoint_cfg, 'quantization_config'):
        cfg.quantization_config = checkpoint_cfg.quantization_config
    cfg.dtype = checkpoint_cfg.dtype
    kwargs.setdefault('dtype', 'auto')
    model, report = _model_class(family, mtp_spec).from_pretrained(checkpoint, config=cfg,
                                              output_loading_info=True, **kwargs)
    problems = {k: report.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if report.get(k)}
    _fail(not problems, f"checkpoint 没有完整加载：{problems}")
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
    return (f'# {stamp()}\n# Generated by by1 from {filename}\n# Complete Transformers backend; includes declared MTP weights.\n'
            'from by1full import build_model, load_model\n\n'
            f'FAMILY = {family!r}\nCONFIG = {pprint.pformat(cfg, sort_dicts=False)}\n'
            f'MTP = {mtp!r}\n\n'
            'def build(device=None, dtype=None):\n'
            '    return build_model(FAMILY, CONFIG, MTP, device=device, dtype=dtype)\n\n'
            'def load(checkpoint, **kwargs):\n'
            '    return load_model(FAMILY, CONFIG, MTP, checkpoint, **kwargs)\n')


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
