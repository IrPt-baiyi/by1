#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1boot -- 从一个已发布的 checkpoint **反推**一份 .by1 草稿。

## 为什么是这一件

量过：一份 `.by1` 里 27% 是决定、64% 是机械。而机械那一半里：

    张量契约   机制 + 属性 -> 名字和形状     by1contract.py 已经能推
    导出命名   「这个仓库怎么命名」           **推不出来 —— 除非问产物**

**而产物就在那儿。** config 里有维度，safetensors 的头里有名字和形状 ——
`by1verify` 每天在读它们，只是**只拿去比对、不拿去生成**。

## 这一件的真正形状

它**不是**"帮你写 .by1"，它是**把"你必须懂 73 个属性"换成"你改到验过为止"**：

    by1boot  ->  草稿 .by1
    by1verify ->  哪里对不上
    你改      ->  直到全绿

**草稿对不对不需要你判断 —— 判卷人判断。** 这才是"小学生也能写"
那条路的真身：不是让他写，是**让他改，而且改到哪儿算对由机器说了算**。

用法:  python by1boot.py <config.json> <tensors.json> [--name X] [--out f.by1]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

# **版本号从 by1ver 来。** 原来这里写死 "1.0"，而另外两个文件也各写了一遍
# —— 三份同一个字符串，谁也不认识谁。改一份忘一份不会崩，
# 只会造出「声称不同版本」的 IR。
try:
    from by1ver import IR_VERSION as _IR_VER
except ImportError:                     # 单独拷一个文件出去时兜底
    _IR_VER = "1.0"



def norm(k):
    return re.sub(r"\.\d+\.", ".N.", k)


def layer_of(k):
    m = re.search(r"\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def classify(names):
    """一层里的张量名 -> 大概是哪种机制。**说不出就说不说。**"""
    s = " ".join(names)
    if re.search(r"linear_attn|\.mixer\.(A_log|D|conv1d)", s):
        return "Linear"
    if re.search(r"kv_a_proj|q_a_proj", s):
        return "MLA"
    # GPT-2 的注意力在 `h.N.attn.c_attn` —— 融合的 qkv，
    # 名字里既没有 self_attn 也没有 q_proj。第一版不认它，
    # 于是 12 层全被判成 Raw。
    if re.search(r"self_attn\.(q_proj|wq|c_attn)|attn\.(qkv|c_attn)", s):
        return "Attention"
    if re.search(r"mixer\.(q_proj|k_proj)", s):
        return "Attention"
    return None


def ffn_of(names):
    s = " ".join(names)
    if re.search(r"experts\.|expert", s):
        return "MoE"
    if re.search(r"mlp\.(gate_proj|c_fc|up_proj)", s):
        return "FFN"
    return None


# ── 从产物直接构造 IR ──────────────────────────────────────────────
def _g(real, *suffixes):
    """张量头里有没有以这些后缀结尾的键。"""
    for k in real:
        for s in suffixes:
            if k.endswith(s):
                return k
    return None


# **config 的字段名有两套方言。**
# 现代的那套（Qwen / Llama）和上一代的那套（GPT-2）叫法完全不同 ——
# 而"另一代"这件事在这个项目里反复出现（GPT-2 已经教育过一次）。
# 认不出来就**说清楚是哪几个字段**，不要崩在一个 int(None) 上。
FIELD_ALIASES = {
    "hidden_size": ["hidden_size", "n_embd", "d_model"],
    "num_hidden_layers": ["num_hidden_layers", "n_layer", "num_layers"],
    "num_attention_heads": ["num_attention_heads", "n_head", "num_heads"],
    "num_key_value_heads": ["num_key_value_heads", "n_head_kv"],
    "vocab_size": ["vocab_size"],
    "intermediate_size": ["intermediate_size", "n_inner", "ffn_dim"],
    "max_ctx": ["max_position_embeddings", "n_positions", "n_ctx",
                "max_seq_len"],
    "norm_eps": ["rms_norm_eps", "layer_norm_epsilon", "norm_eps"],
    "rope_theta": ["rope_theta"],
    "sliding_window": ["sliding_window"],
}


def pick(cfg, key, default=None):
    for k in FIELD_ALIASES.get(key, [key]):
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def _unwrap_text(cfg):
    """**新一带的 config 把文本部分嵌在 `text_config` 里。**

    实测（2026 那批）：
        qwen3_5 / qwen3_vl / gemma4_unified / embedding_gemma2 / qwen3_omni
        顶层只有 `model_type` + `text_config` + `vision_config`（+ audio）
        而 `hidden_size` / `num_hidden_layers` / `vocab_size` **全在
        `text_config` 里面**。

    老一代（llama / qwen2 / mamba / llada2_moe）是平的，顶层就有。

    ## 为什么要在这层解决

    这不是某个模型的怪癖，是**多模态成了默认**之后的通用做法：
    一个 config 描述好几个塔，文本只是其中一个。
    不认它就等于**新一代全部描述不了** —— 而那正是"加大量模型"要覆盖的。

    返回 (文本 config, 有没有被嵌过, 外层的塔名列表)。
    """
    if not isinstance(cfg, dict):
        return cfg, False, []
    if pick(cfg, "hidden_size") is not None:
        return cfg, False, []
    towers = [k for k in cfg
              if k.endswith('_config') and isinstance(cfg[k], dict)]
    # **嵌的键不止 `text_config` 一个。** 各家叫法不同：
    #     text_config      Qwen3.5 / Qwen3-VL / gemma4_unified / embedding_gemma2
    #     thinker_config   Qwen3-Omni（思考塔才是语言模型）
    #     llm_config / language_config / text_model_config  …
    for key in ('text_config', 'thinker_config', 'llm_config',
                'language_config', 'text_model_config'):
        tc = cfg.get(key)
        if isinstance(tc, dict) and pick(tc, 'hidden_size') is not None:
            inner = dict(tc)
            if 'model_type' not in inner and cfg.get('model_type'):
                inner['model_type'] = cfg['model_type']
            return inner, True, [k for k in towers if k != key]
    # 有的嵌两层（Omni: thinker_config -> text_config）
    for key in towers:
        inner, nested, more = _unwrap_text(cfg[key])
        if nested and pick(inner, 'hidden_size') is not None:
            if 'model_type' not in inner and cfg.get('model_type'):
                inner['model_type'] = cfg['model_type']
            return inner, True, [k for k in towers if k != key] + more
    return cfg, False, []


def _shape(real, key):
    """取形状。**取不到就返回 None，不炸。**

    ## 为什么会有取不到的时候

    `by1index.py` 是从 `model.safetensors.index.json` 抓张量清单的 ——
    那里面**只有名字，没有形状**（形状在分片文件里）。

    而名字**够 `classify()` 认机制了**：
        self_attn.q_proj   -> Attention
        mlp.experts        -> MoE
        linear_attn.*      -> Linear (GDN)

    所以正确的做法不是"再去抓一遍形状"（那要下整个头，慢），
    而是：**名字用，形状缺了就报"这是猜的"。**

    第一版直接在 `real[kk]["shape"]` 上取，于是 22 个模型全炸在
    `TypeError: 'NoneType' object is not subscriptable` ——
    **而那个报错完全看不出是"形状没有"，看起来像模型坏了。**
    """
    v = real.get(key)
    if not isinstance(v, dict):
        return None
    s = v.get('shape')
    return s if isinstance(s, list) and s else None


def boot_ir(cfg, real, name="booted"):
    """config + 张量头 -> **规范化的 IR**。

    返回 (ir, guessed)。`guessed` 是"产物里看不出来、填了默认值"的属性 ——
    **必须报出来**，否则用的人会以为它是读出来的。
    """
    cfg, was_nested, towers = _unwrap_text(cfg)
    if was_nested:
        # **嵌过就要说。** 不说的话，用的人会以为这个 IR 是整个模型的，
        # 而它只是文本塔 —— 视觉/音频那些张量在契约里是"多出来的"。
        print('  （文本配置嵌在 `text_config` 里，已取出；'
              '外层还有 %s —— 那些不在这个 IR 里）'
              % (', '.join(towers) if towers else '没有别的塔'))
    d = pick(cfg, "hidden_size")
    L = pick(cfg, "num_hidden_layers", 0)
    V = pick(cfg, "vocab_size")
    ctx = pick(cfg, "max_ctx", 4096)
    miss = [k for k, v in (("宽 d_model", d), ("层数", L), ("词表", V))
            if v is None]
    if miss:
        raise SystemExit(
            "这个 config 里认不出：%s\n"
            "  **这是字段名的问题，不是模型的问题。** 认得的别名在 "
            "FIELD_ALIASES 里 —— 加一个就行。\n"
            "  （顶层和 `text_config` 都找过了。）" % "、".join(miss))
    d, L, V, ctx = int(d), int(L), int(V), int(ctx)
    # GPT-2 的 n_inner 常常是 null，含义是 4 倍宽
    if pick(cfg, "intermediate_size") is None and "n_inner" in cfg:
        cfg = dict(cfg, intermediate_size=4 * d)

    # 位置：有 wpe 就是学习式查表，否则是 RoPE
    pos_kind = "learned" if _g(real, "wpe.weight") else "rope"
    # 归一化：有 bias 就是 LayerNorm（RMSNorm 没有 bias）
    ln_has_bias = bool(_g(real, "ln_f.bias", "ln_1.bias", "norm.bias"))
    norm_kind = "layer" if ln_has_bias else "rms"
    norm_eps = float(pick(cfg, "norm_eps", 1e-5))

    guessed = []
    guessed_keys = set()

    def _def(where, key, val):
        # **键和消息分开。** 判卷人要比对"这个属性是不是猜的"，
        # 而拿整条消息当键对不上（消息里带着值）。
        # 第一版就是这么错的 —— 于是"猜的"没被跳过，报了一堆假差异。
        guessed_keys.add("%s.%s" % (where, key))
        guessed.append("%s.%s = %r" % (where, key, val))
        return val

    # ── 每层是什么 ─────────────────────────────────────────────────
    per = {}
    for k in real:
        li = layer_of(k)
        if li is not None:
            per.setdefault(li, []).append(k)

    q = pick(cfg, "num_attention_heads")
    kv = pick(cfg, "num_key_value_heads") or q
    hd = cfg.get("head_dim") or (d // q if q else None)
    win = pick(cfg, "sliding_window")
    lts = cfg.get("layer_types") or []

    layers = []
    for li in range(L):
        ks = per.get(li, [])
        kind = classify(ks)
        if kind is None:
            kind = "Raw"
        ops = []
        # ① 归一化
        ops.append({"mech": "Norm", "kind": "Norm",
                    "attrs": {"kind": norm_kind, "eps": norm_eps,
                              "one_plus": False},
                    "inputs": ["hidden"], "outputs": ["op0.out"]})
        # ② 混合器
        if kind == "Attention":
            out_dim = None
            for kk in ks:
                if kk.endswith("o_proj.weight") or kk.endswith("c_proj.weight"):
                    out_dim = int((_shape(real, kk) or [0])[-1])
                    break
            lt = lts[li] if li < len(lts) else "full_attention"
            w = (win if "sliding" in str(lt) else None)
            attrs = {
                "q": int(q), "kv": int(kv), "head_dim": int(hd),
                "out_dim": int(out_dim or (q * hd)),
                "bias": bool(_g(real, "self_attn.q_proj.bias", "q_proj.bias",
                                "c_attn.bias")),
                "window": w,
                "qk_norm": _def("L%d.Attention" % li, "qk_norm", "off"),
                "rope": pos_kind != "learned",
                "rope_base": int(pick(cfg, "rope_theta", 10000)),
                "rope_pairing": _def("L%d.Attention" % li, "rope_pairing", "half"),
                "rope_partial": float(cfg.get("partial_rotary_factor") or 1.0),
                "rope_scale": 1.0,
                "yarn": None,
                "q_gate": _def("L%d.Attention" % li, "q_gate", False),
                "sink": _def("L%d.Attention" % li, "sink", False),
                "kv_tie": _def("L%d.Attention" % li, "kv_tie", False),
                "head_gate": _def("L%d.Attention" % li, "head_gate", "off"),
                "gate_act": _def("L%d.Attention" % li, "gate_act", "softplus"),
            }
            ops.append({"mech": "Attn", "kind": "Attention", "attrs": attrs,
                        "inputs": ["op0.out"], "outputs": ["op1.out"]})
        else:
            # 认不出来 —— 说出来，不猜
            ops.append({"mech": "Unknown", "kind": "Raw",
                        "attrs": {"impl": "unknown"},
                        "inputs": ["op0.out"], "outputs": ["op1.out"]})
        ops.append({"mech": "Add", "kind": "Add", "attrs": {},
                    "inputs": ["hidden", "op1.out"], "outputs": ["op2.out"]})
        # ③ 归一化 + 挂的机制
        ops.append({"mech": "Norm2", "kind": "Norm",
                    "attrs": {"kind": norm_kind, "eps": norm_eps,
                              "one_plus": False},
                    "inputs": ["op2.out"], "outputs": ["op3.out"]})
        fk = ffn_of(ks)
        if fk == "FFN":
            # **config 不说就从产物里读。** GPT-2 的 config 里没有
            # n_inner 也没有 intermediate_size —— 但 `mlp.c_fc.weight`
            # 的形状就写着答案。第一版给了 0，于是 FFN 的宽度是 0。
            hid = int(pick(cfg, "intermediate_size", 0))
            if not hid:
                for kk in ks:
                    if re.search(r"(gate_proj|up_proj|c_fc)\.weight$", kk):
                        # Conv1D 是 [in, out]，nn.Linear 是 [out, in] ——
                        # 取大的那一维，两种布局都对
                        hid = int(max(_shape(real, kk) or [0]))
                        break
            ops.append({"mech": "FFN_", "kind": "FFN",
                        "attrs": {"hidden": hid,
                                  "act": _def("L%d.FFN" % li, "act", "silu"),
                                  "gate": _def("L%d.FFN" % li, "gate", True),
                                  "bias": bool(_g(real, "mlp.c_fc.bias",
                                                  "gate_proj.bias")),
                                  "limit": None, "alpha": 1.702},
                        "inputs": ["op3.out"], "outputs": ["op4.out"]})
        elif fk == "MoE":
            ne = int(cfg.get("num_local_experts") or cfg.get("num_experts")
                     or cfg.get("n_routed_experts") or 0)
            nh = int(cfg.get("moe_intermediate_size") or 0)
            ops.append({"mech": "MoE_", "kind": "MoE",
                        "attrs": {"experts": ne,
                                  "top_k": int(cfg.get("num_experts_per_tok")
                                               or cfg.get("experts_per_token") or 1),
                                  "hidden": nh,
                                  "shared": int(cfg.get("n_shared_experts") or 0),
                                  "shared_hidden": nh,
                                  "shared_gate": _def("L%d.MoE" % li, "shared_gate", False),
                                  "routing": _def("L%d.MoE" % li, "routing", "softmax_topk"),
                                  "router_bias": bool(_g(real, "mlp.gate.bias",
                                                         "router.bias")),
                                  "expert_bias": _def("L%d.MoE" % li, "expert_bias", False),
                                  "score_bias": _def("L%d.MoE" % li, "score_bias", False),
                                  "n_group": _def("L%d.MoE" % li, "n_group", 0),
                                  "topk_group": _def("L%d.MoE" % li, "topk_group", 0),
                                  "routed_scale": float(cfg.get("routed_scaling_factor") or 1.0),
                                  "act": _def("L%d.MoE" % li, "act", "silu"),
                                  "limit": None, "limit_shared": None,
                                  "alpha": 1.702},
                        "inputs": ["op3.out"], "outputs": ["op4.out"]})
        ops.append({"mech": "Add2", "kind": "Add", "attrs": {},
                    "inputs": ["op2.out", "op4.out"], "outputs": ["op5.out"]})
        layers.append({"index": li, "attrs": {}, "ops": ops, "state": []})

    globals_ = [
        {"mech": "Embed", "kind": "Embed", "attrs": {},
         "inputs": [], "outputs": ["hidden"]},
        {"mech": "FinalNorm", "kind": "Norm",
         "attrs": {"kind": norm_kind, "eps": norm_eps, "one_plus": False},
         "inputs": ["hidden"], "outputs": ["normed"]},
        {"mech": "Head", "kind": "Head", "attrs": {},
         "inputs": ["normed"], "outputs": ["logits"]},
    ]

    ir = {
        "by1-ir": _IR_VER,
        "vocab": V, "ctx": ctx, "d_model": d,
        "pos_kind": pos_kind,
        "norm_kind": norm_kind, "norm_eps": norm_eps, "norm_one_plus": False,
        "globals": globals_, "layers": layers,
    }
    if pos_kind == "learned":
        ir["n_pos"] = int(cfg.get("n_positions") or cfg.get("n_ctx") or 1024)
    return ir, guessed, guessed_keys


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cfg_p, ten_p = sys.argv[1], sys.argv[2]
    name = None
    out_p = None
    want_ir = '--emit-ir' in sys.argv
    want_run = '--run' in sys.argv
    if '--name' in sys.argv:
        name = sys.argv[sys.argv.index('--name') + 1]
    if '--out' in sys.argv:
        out_p = sys.argv[sys.argv.index('--out') + 1]

    cfg = json.load(open(cfg_p, encoding='utf-8'))
    cfg = cfg.get('text_config', cfg)
    real = json.load(open(ten_p, encoding='utf-8'))

    if want_ir or want_run:
        import by1ir
        _nm = name or str(cfg.get('model_type', 'booted')).replace('.', '-')
        ir, guessed, _gk = boot_ir(cfg, real, _nm)
        errs = by1ir.validate(ir)
        print('=' * 76)
        print('  从产物直接构造 IR（**不经过 .by1**）')
        print('=' * 76)
        print()
        print('  推出来的：')
        print('    pos_kind  = %s   （%s）' % (
            ir['pos_kind'],
            'wpe.weight 在产物里' if ir['pos_kind'] == 'learned' else '没有 wpe'))
        print('    norm_kind = %s   （%s）' % (
            ir['norm_kind'],
            '归一化有 bias' if ir['norm_kind'] == 'layer' else '归一化没 bias'))
        print('    norm_eps  = %s' % ir['norm_eps'])
        print('    %d 层' % len(ir['layers']))
        print()
        print('  **产物里看不出来、填了默认值的：%d 处**' % len(guessed))
        for g in sorted(set(guessed))[:8]:
            print('    %s' % g)
        if len(set(guessed)) > 8:
            print('    …另有 %d 处' % (len(set(guessed)) - 8))
        print('    **这些是猜的，不是读出来的。** 要确定就得对拍产物。')
        print()
        if errs:
            print('  [FAIL] IR 不合法：')
            for e in errs[:8]:
                print('    ' + e)
            return 1
        print('  [OK] 合规格')
        if want_ir:
            print()
            print(by1ir.to_json(ir))
            return 0
        # --run：三个后端各跑一遍
        print('  ── 三个后端从这份 IR 跑 ──')
        nelem = 0
        try:
            import by1exec as _ex
            sh = _ex.shapes_of(ir)
            nelem = sum(int(__import__('numpy').prod(v)) for v in sh.values())
        except Exception:
            pass
        if nelem > 2e8:
            print('    [跳过] %.1fM 参数，这个演示跑不动' % (nelem / 1e6))
            return 0
        import numpy as np
        import by1codegen as _cg
        ns = {}
        exec(compile(_cg.render_ir(ir, 'booted.ir'), '<ir>', 'exec'), ns)
        m = ns['build']().eval()
        import torch
        with torch.no_grad():
            ot = m(torch.randint(0, ir['vocab'], (1, 8))).numpy()
        import by1exec as _ex2
        on, _p = _ex2.exec_ir(ir, seq=8)
        import by1c as _c
        ctext, _o, _f = _c.emit_c_ir(ir, _ex2.shapes_of(ir))
        print('    PyTorch %s · NumPy %s · C %d 行' % (
            ot.shape, on.shape, len(ctext.split(chr(10)))))
        print('    **三个后端都只拿了这份 IR。**')
        return 0

    name = name or cfg.get('model_type', 'booted').replace('.', '-')
    d = cfg.get('hidden_size')
    L = cfg.get('num_hidden_layers')
    V = cfg.get('vocab_size')

    # ── 按层分组 ───────────────────────────────────────────────────
    per_layer = defaultdict(list)
    globals_ = []
    for k in real:
        li = layer_of(k)
        (per_layer[li] if li is not None else globals_).append(k)

    # ── 每层是什么 ─────────────────────────────────────────────────
    kinds = {}
    ffns = {}
    for li in sorted(per_layer):
        ks = per_layer[li]
        kinds[li] = classify(ks)
        ffns[li] = ffn_of(ks)
    kc = Counter(kinds.values())
    fc = Counter(ffns.values())

    L_ = ['# ── 由 by1boot 从产物反推的草稿 ──────────────────────────────',
          '#',
          '# **它不是"帮你写"，是"让你改到验过为止"**：',
          '#     python by1verify.py <这个文件> %s --config --tensors %s --backend torch.module'
          % (os.path.basename(cfg_p), os.path.basename(ten_p)),
          '#',
          '# 反推不出来的地方，下面都标了 `# ?`。改到 by1verify 全绿为止。',
          '# ──────────────────────────────────────────────────────────',
          '',
          'model %s {' % name,
          '  arch  %s' % cfg.get('model_type', '?'),
          '  ctx   %d' % cfg.get('max_position_embeddings', 0),
          '',
          '  hparams {',
          '    d_model = %s' % d,
          '    n_layer = %s' % L,
          '    vocab   = %s' % V,
          '    rms_eps = %s' % cfg.get('rms_norm_eps', 1e-5),
          '  }',
          '']
    print('\n'.join(L_))
    print('  # 反推结果：')
    print('  #   层数 %d，机制分布 %s' % (len(per_layer), dict(kc)))
    print('  #   挂在层上的 %s' % dict(fc))
    print('  #   全局张量 %d 个：%s' % (len(globals_), ', '.join(sorted(globals_)[:6])))
    print('  #')
    print('  # ? 反推不出来的：路由方式 / 有没有共享专家 / 门控激活 /')
    print('  #   per_expert 还是打包 / bias / rope 的缩放类型 —— 这些**config 里没有**，')
    print('  #   而产物只有名字和形状。改到 by1verify 全绿为止。')
    print()

    # ── 契约：**直接从产物里读** ───────────────────────────────────
    # 这是这一件里最值钱的部分，而它推不出来 —— 除非问产物。
    # 形状写成**字面数字**（不是符号）：符号需要先知道哪个数是哪个维度，
    # 而那正是你要决定的。字面数字至少是**对的**，能立刻验过。
    print('  # ── 契约（从 safetensors 头里读的，形状是字面值）──────────')
    print('  # 把 (4096, 2688) 换成 (q * head_dim, d_model) 这类符号，')
    print('  # 就是"我理解了这个模型" —— 而 by1check 会告诉你换得对不对。')
    print()
    groups = defaultdict(dict)
    layern = {}
    # **一层里同时有混合器和 FFN，得分桶。**
    # 第一版按"这一层是什么机制"分，于是 mlp.* 全被塞进了 Attention 桶里 ——
    # 名字是对的、形状是对的、**分错了组**。而分错组的表现是
    # "Attention 里有 mlp"，一眼能看出来，前提是你去看。
    MIX = re.compile(r'^(self_attn|linear_attn|mixer|attn|attention)\.')
    FFN = re.compile(r'^(mlp|feed_forward|ffn)\.')
    for li in sorted(per_layer):
        kmain = kinds[li] or 'Raw'
        kffn = ffns[li]
        for nm in per_layer[li]:
            short = re.sub(r'^.*layers\.\d+\.', '', nm)
            short = re.sub(r'^(model|backbone|language_model)\.', '', short)
            short = re.sub(r'^layers\.\d+\.', '', short)
            if MIX.match(short):
                groups[kmain].setdefault(short, _shape(real, nm))
            elif FFN.match(short) and kffn:
                groups[kffn].setdefault(short, _shape(real, nm))
            elif re.search(r'norm|layernorm', short):
                layern.setdefault(short, _shape(real, nm))
            else:
                layern.setdefault(short, _shape(real, nm))
    for key in sorted(groups):
        print('    %s {' % key)
        for nm in sorted(groups[key]):
            shp = ', '.join(str(x) for x in groups[key][nm])
            print('      %-40s (%s)' % (nm + ' :', shp))
        print('    }')
    print('    layer {')
    for nm in sorted(layern):
        shp = ', '.join(str(x) for x in layern[nm])
        print('      %-40s (%s)' % (nm + ' :', shp))
    print('    }')
    print('    global {')
    for nm in sorted(globals_):
        shp = ', '.join(str(x) for x in _shape(real, nm))
        print('      %-40s (%s)' % (nm + ' :', shp))
    print('    }')
    print('  }')
    return 0


if __name__ == '__main__':
    sys.exit(main())
