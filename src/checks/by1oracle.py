#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 oracle -- 执行神谕：把 .by1 声明的状态策略与**真实前向的行为**对拍。

它不是一个静态检查器。它跑模型。

为什么需要它：config.json / safetensors / GGUF 里都没有「这个 cache 有多大」
这个数。所以「状态」这一半在文件对拍里是盲区，跑起来才能看见。

判定方式：在两个不同的序列长度上各跑一次前向，逐层观察 cache 长度。
  两次都等于各自 seq        -> 随序列增长 (grows)
  两次相同且都小于 seq      -> 有界 (bounded)
  两次都为 0 / 不存在       -> 无 cache (fixed)
然后把观测与 .by1 的 state 块声明对比。

用法:
  python by1oracle.py <file.by1> --family gpt_oss [--ref-config refs/xxx.config.json]

神谕本身也要可证伪：--inject 可以故意篡改声明，用来确认它真的会报错。
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import argparse
import importlib.util
import inspect
import os
import sys
import contextlib
import by1io

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = {
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM"),
    "gemma3": ("Gemma3Config", "Gemma3ForCausalLM"),
    "gemma4": ("Gemma4Config", "Gemma4ForCausalLM"),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextForCausalLM"),
}


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "by1check", by1paths.tool("by1check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── state 选择器（复用 by1check 的实现，避免两个工具各说各话）──────

def parse_state_key(key):
    return load_checker().parse_state_key(key)


def selector_matches(conds, attrs):
    return load_checker().selector_matches(conds, attrs)


def find_declared(state_src, mech, attrs):
    """按机制名 + 选择器，找这一层适用的 state 声明。"""
    for key, val, _ln in state_src.get(mech, []):
        _n, conds = parse_state_key(key)
        if selector_matches(conds, attrs):
            return key, val
    return None, None


def classify_declared(val):
    if val is None:
        return "undeclared"
    if "grows_with_seq" in val:
        return "grows"
    if "recurrent" in val:
        return "fixed"
    if "bounded_by" in val:
        return "bounded"
    return "other"


# ── 造微型参考模型并跑 ──────────────────────────────────────────────

def build_tiny(fam, ref_path, info, seq_max):
    import torch
    import transformers as T

    cfg_name, model_name = FAMILIES[fam]
    cfgc = getattr(T, cfg_name)
    mc = getattr(T, model_name)

    accepted = set(inspect.signature(cfgc.__init__).parameters)
    # 只继承「非结构」字段（eps、激活函数之类）。结构尺寸一律换成微型值，
    # 否则会把真实模型的专家层照搬进来 —— 那要 3.4 GB。
    STRUCTURAL = {
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "global_head_dim", "num_local_experts", "num_experts_per_tok",
        "experts_per_token", "max_position_embeddings", "layer_types",
        "sliding_window", "sliding_window_pattern", "tie_word_embeddings",
        "eos_token_id", "bos_token_id", "pad_token_id", "rope_scaling",
        "rope_parameters", "initial_context_length",
    }
    fields = {}
    if ref_path:
        raw = by1io.read_json(ref_path, encoding="utf-8")
        tc = raw.get("text_config", raw)
        for k, v in tc.items():
            if k in accepted and k not in STRUCTURAL and not isinstance(v, (dict, list)):
                fields[k] = v

    layers = info["layers"]
    n = len(layers)
    layer_types = []
    for _stack, _mech, attrs in layers:
        w = (attrs.get("window") or "").strip().lower()
        layer_types.append("full_attention" if w in ("none", "null", "0", "")
                           else "sliding_attention")

    window = None
    for _s, _m, a in layers:
        w = (a.get("window") or "").strip().lower()
        if w not in ("none", "null", "0", ""):
            # 解析不出来就**保持默认**，并说明默认是什么。
            # 原来是 `except ValueError: pass` —— 读起来像"没想好"，
            # 而实际语义是"这层的 window 不是个整数，就不设上限"。
            window = None
            with contextlib.suppress(ValueError):
                window = int(float(w))
            break

    hidden = 64
    fields.update(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=max(seq_max + 16, 256),
        layer_types=layer_types,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=True,
    )
    for k, v in (("num_local_experts", 4), ("num_experts_per_tok", 2),
                 ("experts_per_token", 2), ("num_experts", 4)):
        if k in accepted:
            fields[k] = v
    if window:
        fields["sliding_window"] = window
        if "sliding_window_pattern" in accepted:
            fields["sliding_window_pattern"] = 2
    fields = {k: v for k, v in fields.items() if k in accepted}

    cfg = cfgc(**fields)
    model = mc(cfg).eval()
    torch.manual_seed(0)
    return torch, model, fields, window


def measure(torch, model, seq):
    ids = torch.randint(0, 32, (1, seq))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    res = []
    for L in out.past_key_values.layers:
        k = L.keys
        res.append(0 if k is None else int(k.shape[-2]))
    return res


def build_parser():
    p = argparse.ArgumentParser(description="by1 执行神谕")
    p.add_argument("by1")
    p.add_argument("--family", default="gpt_oss", choices=sorted(FAMILIES))
    p.add_argument("--ref-config", default=None)
    p.add_argument("--seq-a", type=int, default=None)
    p.add_argument("--seq-b", type=int, default=None)
    p.add_argument("--inject", default=None,
                   help="故意把某一类层的声明篡改成这个，用来验证神谕会报错")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    bc = load_checker()
    _rep, info = bc.check(args.by1)

    window_guess = None
    for _s, _m, a in info["layers"]:
        w = (a.get("window") or "").strip().lower()
        if w not in ("none", "null", "0", ""):
            window_guess = None
            with contextlib.suppress(ValueError):
                window_guess = int(float(w))
            break

    seq_a = args.seq_a or (window_guess * 2 if window_guess else 64)
    seq_b = args.seq_b or (window_guess * 4 if window_guess else 128)

    print("=" * 78)
    print(f"  {os.path.basename(args.by1)}   (执行神谕)")
    print("=" * 78)

    torch, model, fields, window = build_tiny(
        args.family, args.ref_config, info, seq_b)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  参考实现 : {FAMILIES[args.family][1]}   参数 {n_par:,}")
    print(f"  微型配置 : {len(info['layers'])} 层, hidden={fields['hidden_size']}, "
          f"q={fields['num_attention_heads']}, kv={fields['num_key_value_heads']}, "
          f"head_dim={fields['head_dim']}, window={window}")
    print(f"  两次前向 : seq = {seq_a} 与 {seq_b}")
    print()

    ca, cb = measure(torch, model, seq_a), measure(torch, model, seq_b)

    def observe(i):
        a, b = ca[i], cb[i]
        if a == 0 and b == 0:
            return "fixed", None
        if a == b and a < min(seq_a, seq_b):
            return "bounded", a
        if a == seq_a and b == seq_b:
            return "grows", None
        return "other", (a, b)

    rows = []
    for i, (_stack, mech, attrs) in enumerate(info["layers"]):
        key, val = find_declared(info.get("state", {}), mech, attrs)
        if args.inject and key is not None:
            val = args.inject
        dec = classify_declared(val)
        obs, bound = observe(i)
        ok = (dec == obs)
        rows.append((i, mech, attrs, key, val, dec, obs, bound, ok))

    # 按 (声明, 观测) 归并
    groups = {}
    for r in rows:
        k = (r[5], r[6], r[7], r[1], (r[2].get("window") or "?"))
        groups[k] = groups.get(k, 0) + 1

    print("  声明的策略 vs 观测到的行为")
    print(f"    {'层数':>4}  {'机制':<8}{'window':<8}{'声明':<14}{'观测':<16}判定")
    print("    " + "-" * 72)
    for (dec, obs, bound, mech, win), cnt in sorted(groups.items(), key=lambda x: -x[1]):
        obs_txt = {"grows": "随序列增长", "bounded": f"有界于 {bound}",
                   "fixed": "无 cache", "other": "?"}[obs]
        dec_txt = {"grows": "随序列增长", "bounded": "有界", "fixed": "固定",
                   "undeclared": "未声明", "other": "其他"}[dec]
        if dec == "undeclared":
            mark = "[未声明]"
        elif obs == "other":
            mark = "[需人看]"
        else:
            mark = "[一致]  " if dec == obs else "[差异]  "
        print(f"    {cnt:>4}  {mech:<8}{str(win):<8}{dec_txt:<14}{obs_txt:<16}{mark}")

    bad = [r for r in rows if r[5] != "undeclared" and r[6] != "other" and not r[8]]
    und = [r for r in rows if r[5] == "undeclared"]
    print()
    if bad:
        ex = bad[0]
        print(f"  [FAIL] {len(bad)} 层声明与观测不符。例如 L{ex[0]} "
              f"({ex[1]}, window={ex[2].get('window')})："
              f"声明 '{ex[4]}'，实测有界于 {ex[7]}")
    elif und and len(und) == len(rows):
        print(f"  [未声明] {len(und)} 层都没有 state 声明 —— 神谕无法判定，"
              f"这不是通过")
    elif und:
        print(f"  [部分] {len(und)} 层未声明，其余 {len(rows)-len(und)} 层一致")
    else:
        print(f"  [PASS] {len(rows)} 层的状态声明与实测行为全部一致")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
