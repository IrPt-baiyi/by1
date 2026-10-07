#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 train -- 按 .by1 的描述搭一个模型，训练它，然后让它写字。

    python by1train.py hello.by1 --text 1.md

这是 by1 第一次产出**能跑、能学**的东西：模型结构来自 .by1，
不经过 transformers，也不需要任何官方权重。

诚实的边界：前向实现是 by1 自带的（by1codegen.py 里的 RUNTIME），
没有外部参考实现来判卷。所以它证明的是「能跑、能学」，不是「和官方逐位一致」。
"""

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEFAULT_TEXT = """by1 是一门描述大模型架构的语言。
主流模型翻来覆去用的是同一批机制，真正新的东西只是这些机制怎么组合。
把这句话写成一份描述，就该能生成一个能跑的模型。
模型的宽度、层数、中间层宽度、注意力头数，都是可以改的数字。
改一个数字，重新训练一次，看看它写出来的东西有什么不同。
"""


def read_corpus(paths):
    if not paths:
        cands = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE))
                 if f.endswith(".md") and not f.startswith(("_", "hello", "selftest"))]
        if not cands:
            return DEFAULT_TEXT.encode("utf-8"), "内置小语料"
        paths = cands
    buf = []
    for p in paths:
        with open(p, "rb") as f:
            buf.append(f.read())
    return b"\n".join(buf), ", ".join(os.path.basename(p) for p in paths)


def utf8_step(state, b):
    """UTF-8 增量校验。state = (还差几个续字节, 下一字节下限, 上限)。非法返回 None。"""
    need, lo, hi = state
    if need == 0:
        if b < 0x80:
            return (0, 0x80, 0xBF)
        if 0xC2 <= b <= 0xDF:
            return (1, 0x80, 0xBF)
        if b == 0xE0:
            return (2, 0xA0, 0xBF)
        if 0xE1 <= b <= 0xEC:
            return (2, 0x80, 0xBF)
        if b == 0xED:
            return (2, 0x80, 0x9F)
        if 0xEE <= b <= 0xEF:
            return (2, 0x80, 0xBF)
        if b == 0xF0:
            return (3, 0x90, 0xBF)
        if 0xF1 <= b <= 0xF3:
            return (3, 0x80, 0xBF)
        if b == 0xF4:
            return (3, 0x80, 0x8F)
        return None
    if lo <= b <= hi:
        return (need - 1, 0x80, 0xBF) if need > 1 else (0, 0x80, 0xBF)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="by1 训练器")
    ap.add_argument("by1")
    ap.add_argument("--text", action="append", default=None,
                    help="语料文件，可重复。不给就用目录下的 .md")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--gen", default=None, help="生成时的前缀")
    ap.add_argument("--gen-len", type=int, default=160)
    ap.add_argument("--save", default=None, help="把训练好的权重存到这里")
    ap.add_argument("--source-out", default=None, help="把生成的模型源码存到这里")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    import torch

    bc = load("by1check")
    cg = load("by1codegen")

    name = os.path.basename(args.by1)
    print(f"\n  读取 {name}")
    _rep, info = bc.check(args.by1)

    try:
        ir = cg.compile_ir(info)
    except cg.CodegenError as ex:
        print(f"\n  [不支持] 这份描述超出了 codegen 的子集：\n{ex}\n")
        return 1

    src = cg.render(info, name)
    ns = {}
    exec(compile(src, "<by1-generated>", "exec"), ns)
    model = ns["build"]()
    n_par = sum(p.numel() for p in model.parameters())

    print(f"    {len(ir['layers'])} 层 · d_model {ir['d_model']} · "
          f"词表 {ir['vocab']} · {n_par:,} 参数")
    if not args.quiet:
        print("  按描述搭建模型")
        for i, L in enumerate(ir["layers"]):
            bits = []
            for op in L["ops"]:
                a = op["attrs"]
                if op["kind"] == "Attention":
                    w = a["window"] if a["window"] else "全量"
                    bits.append(f"Attention(q={a['q']}, kv={a['kv']}, "
                                f"head_dim={a['head_dim']}, {w})")
                elif op["kind"] == "FFN":
                    bits.append(f"FFN({a['hidden']})")
                elif op["kind"] == "MoE":
                    bits.append(f"MoE({a['experts']} 专家选 {a['top_k']}, "
                                f"hidden={a['hidden']})")
            print(f"    L{i}  " + "  >>  ".join(bits))

    data, corpus_name = read_corpus(args.text)
    data = torch.tensor(list(data), dtype=torch.long)
    if len(data) < args.seq + 2:
        print(f"\n  [语料太短] 只有 {len(data)} 字节，至少要 {args.seq + 2} 字节\n")
        return 1
    print(f"\n  训练（语料 {corpus_name} · {len(data):,} 字节 · "
          f"{args.steps} 步）")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        ix = torch.randint(0, len(data) - args.seq - 1, (args.batch,))
        x = torch.stack([data[i:i + args.seq] for i in ix])
        y = torch.stack([data[i + 1:i + args.seq + 1] for i in ix])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 8) == 0 or step == 1:
            print(f"    步 {step:>5}   loss {sum(losses[-20:])/len(losses[-20:]):.3f}"
                  f"   {time.time()-t0:.0f}s")

    # ── 生成 ────────────────────────────────────────────────────────
    model.eval()
    prompt = args.gen
    if prompt is None:
        prompt = bytes(data[:12].tolist()).decode("utf-8", "replace")
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long)
    state = (0, 0x80, 0xBF)
    for b in ids[0].tolist():
        state = utf8_step(state, b) or (0, 0x80, 0xBF)
    with torch.no_grad():
        for _ in range(args.gen_len):
            ctx = ids[:, -ir["ctx"]:]
            logits = model(ctx)[:, -1, :].clone()
            for b in range(256):          # 只留能组成合法 UTF-8 的字节
                if utf8_step(state, b) is None:
                    logits[0, b] = float("-inf")
            nxt = logits.argmax(-1, keepdim=True)
            state = utf8_step(state, int(nxt)) or (0, 0x80, 0xBF)
            ids = torch.cat([ids, nxt], dim=1)
    out = bytes(ids[0].tolist()).decode("utf-8", "replace")

    print(f"\n  生成（前缀「{prompt}」）")
    print("    " + out.replace("\n", "\n    "))

    if args.save:
        torch.save({"spec": spec, "state": model.state_dict()}, args.save)
        print(f"\n  权重已保存 {args.save}")
    if args.source_out:
        with open(args.source_out, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"  模型源码已保存 {args.source_out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
