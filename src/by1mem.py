#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 mem -- 从 IR 的 state 声明推出**内存计划**，并可选地用真实模型核它。

为什么单独一个工具：config.json / safetensors / GGUF 里都没有「这个 cache 有多大」
这个数。所以内存这一块在文件对拍里是盲区 —— 和状态一样，得算、得跑。

两种状态，两种增长方式：
  kv_cache   每 token 增长，滑窗时封顶在 window-1
  recurrent  固定大小，与序列长度无关 —— 「线性注意力」的全部含义

用法:
  python by1mem.py qwen3-next-shaped.by1 --seq 8192
  python by1mem.py qwen3-next-shaped.by1 --seq 4096 --measure
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DT = {2: "fp16", 4: "fp32"}


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_of(ir, seq):
    """从 IR 的 state 声明算内存计划。单位：元素个数（不是字节）。"""
    rows = []
    for L in ir["layers"]:
        mixer = next((o for o in L["ops"] if o["kind"] in ("Attention", "Linear")),
                     None)
        if mixer is None or not L["state"]:
            rows.append({"i": L["index"], "kind": "-", "state": "none",
                         "elems": 0, "per_tok": 0, "cap": 0})
            continue
        a = mixer["attrs"]
        elems = per_tok = 0
        cap, bounded, kinds = None, False, []
        for sd in L["state"]:
            if sd["kind"] in ("recurrent", "conv_history"):
                n = 1
                for dd in sd["shape"]:
                    n *= dd
                elems += n
                kinds.append("recurrent" if sd["kind"] == "recurrent"
                             else f"conv({sd['bounded_by']})")
            else:
                per = 2 * a["kv"] * a["head_dim"]
                bnd = bool(sd["bounded_by"])
                c = sd["bounded_by"] if bnd else seq
                elems += per * min(seq, c)
                per_tok += per
                cap, bounded = c, bnd
                kinds.append("kv_cache")
        rows.append({"i": L["index"], "kind": mixer["kind"],
                     "state": "+".join(kinds), "elems": elems,
                     "per_tok": per_tok, "cap": cap, "bounded": bounded})
    return rows


def measure(fam, info, ir, seq):
    """跑一个真实模型，把 cache / state 的实际元素数数出来。"""
    import torch
    import transformers as T
    hp = info.get("hparams") or {}
    d = int(float(hp["d_model"]))
    L0 = ir["layers"][0]
    att = next((o["attrs"] for L in ir["layers"] for o in L["ops"]
                if o["kind"] == "Attention"), None)
    lin = next((o["attrs"] for L in ir["layers"] for o in L["ops"]
                if o["kind"] == "Linear"), None)
    lt = ["linear_attention" if any(o["kind"] == "Linear" for o in L["ops"])
          else "full_attention" for L in ir["layers"]]
    if fam == "qwen3_next":
        cfg = T.Qwen3NextConfig(
            vocab_size=ir["vocab"], hidden_size=d,
            num_hidden_layers=len(ir["layers"]),
            num_attention_heads=att["q"], num_key_value_heads=att["kv"],
            head_dim=att["head_dim"], max_position_embeddings=max(seq, 512),
            rms_norm_eps=1e-5, tie_word_embeddings=False,
            linear_num_key_heads=lin["k_heads"], linear_num_value_heads=lin["v_heads"],
            linear_key_head_dim=lin["k_dim"], linear_value_head_dim=lin["v_dim"],
            linear_conv_kernel_dim=lin["conv_kernel"], layer_types=lt,
            partial_rotary_factor=att["rope_partial"],
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
            shared_expert_intermediate_size=16)
        model = T.Qwen3NextForCausalLM(cfg).eval()
    else:
        raise SystemExit(f"  [跳过] 还没有 {fam} 的量测路径")
    ids = torch.randint(0, ir["vocab"], (1, seq))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    c = out.past_key_values
    rows = []
    for i, lay in enumerate(getattr(c, "layers", [])):
        n, what = 0, []
        for f, v in vars(lay).items():
            # cache 里存的是 dict（按状态槽编号），不是 list —— 之前漏了这种
            items = (list(v.values()) if isinstance(v, dict)
                     else list(v) if isinstance(v, (list, tuple)) else [v])
            for t in items:
                if torch.is_tensor(t) and t.numel() > 1:
                    n += t.numel()
                    what.append(f"{f}{tuple(t.shape)}")
        rows.append({"i": i, "elems": n, "what": what})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 内存计划")
    ap.add_argument("by1")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--seq2", type=int, default=16384)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--family", default="qwen3_next")
    args = ap.parse_args(argv)

    bc, cg = load("by1check"), load("by1codegen")
    name = os.path.basename(args.by1)
    _r, info = bc.check(args.by1)
    try:
        ir = cg.compile_ir(info)
    except cg.CodegenError as ex:
        print(f"\n  [不支持] {name}\n{ex}\n")
        return 1

    ws = {}
    for o in (info.get("mech_attrs") or {}).values():
        if o.get("window"):
            ws[o.get("window")] = True
    print(f"\n{'='*74}\n  {name}  ·  {len(ir['layers'])} 层  "
          f"·  d_model {ir['d_model']}\n{'='*74}")

    r1, r2 = plan_of(ir, args.seq), plan_of(ir, args.seq2)
    print(f"\n  逐层状态（seq = {args.seq}）")
    print(f"  {'层':>4} {'混合器':>10} {'状态':>10} {'元素数':>14} {'每 token':>10}  上限")
    agg = {}
    for a, b in zip(r1, r2):
        key = (a["kind"], a["state"], a["per_tok"], a["cap"], a["bounded"])
        agg.setdefault(key, [0, 0])
        agg[key][0] += a["elems"]
        agg[key][1] += b["elems"]
    for a in r1[:3] + ([{"i": "...", "kind": "...", "state": "...",
                          "elems": 0, "per_tok": 0, "cap": None}] if len(r1) > 6 else []) + r1[-2:]:
        if a["i"] == "...":
            print(f"  {'...':>4}")
            continue
        cap = a["cap"] if a["cap"] is not None else "—"
        print(f"  {a['i']:>4} {a['kind']:>10} {a['state']:>10} "
              f"{a['elems']:>14,} {a['per_tok']:>10,}  {cap}")
    if len(r1) > 6:
        print(f"  （共 {len(r1)} 层，中间省略）")

    print(f"\n  按状态类型汇总")
    print(f"  {'混合器':>10} {'状态':>10} {'层数':>5} {'seq={}'.format(args.seq):>16} "
          f"{'seq={}'.format(args.seq2):>16} {'增长':>30}")
    for (k, st, per, cap, bnd), (e1, e2) in sorted(agg.items()):
        n = sum(1 for a in r1
                if (a["kind"], a["state"], a["per_tok"], a["cap"], a["bounded"])
                == (k, st, per, cap, bnd))
        if per == 0:
            g = "不增长（固定大小）"
        elif bnd and cap < args.seq2:
            g = f"{n} 层封顶在 {cap:,}（滑窗）"
        elif bnd:
            g = f"{n} 层，seq 未达窗口 {cap:,}"
        else:
            g = f"+{per:,}/token × {n} 层（无界）"
        print(f"  {k:>10} {st:>10} {n:>5} {e1:>16,} {e2:>16,} {g:>30}")

    t1 = sum(a["elems"] for a in r1)
    t2 = sum(a["elems"] for a in r2)
    print(f"\n  合计  seq {args.seq:>7,} -> {t1:>15,} 个元素"
          f"   （fp16 约 {t1*2/2**20:.1f} MiB）")
    print(f"        seq {args.seq2:>7,} -> {t2:>15,} 个元素"
          f"   （fp16 约 {t2*2/2**20:.1f} MiB）")
    if t1:
        print(f"        增长倍数 {t2/t1:.2f}×  （纯 KV cache 的模型应当 ≈ {args.seq2/args.seq:.2f}×）")

    if args.measure:
        print(f"\n  实测（真实模型跑一遍，数 cache / state 的元素数）")
        try:
            mrows = measure(args.family, info, ir, min(args.seq, 512))
        except SystemExit as ex:
            print(str(ex))
            return 0
        m1 = plan_of(ir, min(args.seq, 512))
        bad = 0
        print(f"  {'层':>4} {'推算':>12} {'实测':>12}  实际持有的东西")
        for mr in mrows:
            pr = m1[mr["i"]]["elems"] if mr["i"] < len(m1) else -1
            ok = "ok" if mr["elems"] == pr else ("差 " + str(mr["elems"] - pr))
            if mr["elems"] != pr:
                bad += 1
            print(f"  {mr['i']:>4} {pr:>12,} {mr['elems']:>12,}  {ok:<10} "
                  f"{'; '.join(mr['what'])[:52]}")
        print(f"\n  [{'PASS' if bad == 0 else 'FAIL'}] "
              f"{len(mrows)} 层里 {len(mrows)-bad} 层的推算与实测一致")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
