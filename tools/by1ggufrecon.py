#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ggufrecon -- GGUF 两条路（by1 的 build-gguf vs 上游 convert）**逐名逐形对账**，
顺带把 .by1 契约的形状账对上去（B 方向的本地一半，不需要 GPU）。

    python tools/by1ggufrecon.py

判据（每对一行）：
  ① by1 侧 GGUF 与 ref 侧 GGUF 的张量名集合**完全相同**；
  ② 同名张量的**形状与 dtype 逐个相同**；
  ③ .by1 契约（by1check 的 tens）的**形状多重集**能在 GGUF 形状多重集里对上（不留账）。

范围说清：不比张量数据本身（GB 级 IO）；.by1 那侧比的是形状账，
名字账走的是 converter 的 TensorNameMap（by1-f32 就是它产的），
所以 ① 成立即意味着名字账在两条路上一致。
"""

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import by1paths  # noqa: E401,F401

PAIRS = [
    ("qwen3-0.6b", "models/complete-Qwen3-0.6B.by1"),
    ("granite", "models/complete-granite-4.0-h-350m.by1"),
    ("granite-1b", "models/complete-granite-4.0-h-1b.by1"),
    ("mamba-130m", "models/complete-mamba-130m-hf.by1"),
    ("falcon", "models/complete-Falcon-H1-0.5B-Instruct.by1"),
]


def load_gguf(path):
    """{名字: (torch 序形状, dtype 名)}。走上游 gguf-py（converter 用的那份）。"""
    sys.path.insert(0, str(ROOT / "llamacpp" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    out = {}
    for t in r.tensors:
        out[t.name] = (tuple(int(x) for x in reversed(t.shape)),
                       str(t.tensor_type))
    return out


def by1_tens(by1_path):
    """.by1 契约的 (名字, 形状) 列表：走 by1check（唯一真相）。"""
    import by1check
    _rep, info = by1check.check(str(by1_path))
    out = []
    for _mech, _sig, _count, tensors in (info.get("tens") or []):
        for entry in tensors:
            name = entry[0]
            shape_str = entry[1]
            # 形状串形如 '(5120, 248320)' 或 '(256)' 或 '--'
            s = shape_str.strip().strip("()")
            if not s or s == "--":
                continue
            try:
                out.append((name, tuple(int(x.strip()) for x in s.split(","))))
            except ValueError:
                continue
    return out


def recon(slug, by1):
    import by1check, by1codegen  # noqa: F401
    gg1 = ROOT / "build" / ("%s-by1-f32.gguf" % slug)
    gg2 = ROOT / "build" / ("%s-ref-f32.gguf" % slug)
    for f in (gg1, gg2, ROOT / by1):
        if not f.exists():
            print("  [跳过] %s：缺 %s" % (slug, f.name))
            return None
    a, b = load_gguf(gg1), load_gguf(gg2)
    problems = []
    if set(a) != set(b):
        problems.append("名字账不一致（by1 侧 %d / ref 侧 %d，互缺 %d）"
                        % (len(a), len(b),
                           len(set(a) ^ set(b))))
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k]:
            problems.append("%s 形状/dtype 不一致：%r vs %r" % (k, a[k], b[k]))
    # .by1 契约的形状账：多重集必须是 GGUF 的子集（GGUF 会多出合成 router 等）
    tens = by1_tens(ROOT / by1)
    want = Counter(sh for _n, sh in tens)
    have = Counter(sh for sh, _d in a.values())
    missing = want - have
    if not tens:
        # **空跑不算账**：complete-* 的契约由 by1verify 自己的解析器管
        # （by1check 的 tens 为空）—— 形状账这一条**没比**，说明白。
        note = ".by1 契约形状账未比（by1check.tens 为空）"
    else:
        note = ".by1 契约 %d 处形状账%s" % (len(tens),
                 "全对上" if not missing else "：%d 处找不到" % sum(missing.values()))
        if missing:
            problems.append(".by1 契约有 %d 处形状在 GGUF 里找不到：%s"
                            % (sum(missing.values()),
                               ", ".join("%r x%d" % (s2, n) for s2, n in list(missing.items())[:4])))
    print("  [%s] %-12s by1 %d / ref %d 张量 · %s%s"
          % ("FAIL" if problems else "PASS", slug, len(a), len(b), note,
             (" · %d 问题" % len(problems)) if problems else ""))
    for p in problems[:6]:
        print("       - " + p)
    return problems


def main():
    print("\n  GGUF 两条路对账（by1 build-gguf vs 上游 convert）")
    bad = 0
    checked = 0
    for slug, by1 in PAIRS:
        r = recon(slug, by1)
        if r is None:
            continue
        checked += 1
        if r:
            bad += 1
    print("\n  对上 %d 对，%d 对有问题" % (checked - bad, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
