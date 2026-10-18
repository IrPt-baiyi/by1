#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1export -- 把 .by1 模型导成**纯静态站点**能直接加载的数据。

为什么输出 .js 而不是 .json：纯静态站点用 file:// 打开时，fetch() 会被 CORS
挡住。数据写成 window.BY1_* = {...} 的脚本，script 标签加载在任何情形下都能用，
**不需要起服务器**。

**唯一真相**：这里不自己解析 .by1，一律走 by1check / by1codegen --
浏览器看到的必须和门禁看到的是同一份 IR。

    python tools/by1export.py --out site/data
    python tools/by1export.py --out site/data --all
"""

import argparse
import json
import pathlib
import sys


#: 第一期先导这 10 份：覆盖稠密 / MoE / 混合栈 / 稀疏注意力 / 线性注意力，
#: 以及真实家族（Qwen3.8-27B、clef-tiny）的两种尺度。
DEFAULT_MODELS = (
    "llama-shaped.by1",
    "mixtral-shaped.by1",
    "gpt-oss-shaped.by1",
    "ssm-shaped.by1",
    "mla-sparse-shaped.by1",
    "qwen3-next-shaped.by1",
    "laguna-tiny.by1",
    "gelu-shaped.by1",
    "clef-tiny.by1",
    "Qwen3.8-27B.by1",
)


def _slug(name):
    return name[:-4] if name.endswith(".by1") else name


def _trim_attrs(attrs):
    """属性表原样保留，但把不可 JSON 的值转成字符串。"""
    out = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def export_one(root, name, check, codegen):
    """一份模型 -> 一份可 JSON 化的结构（层 / 机制 / 属性 / 摘要）。"""
    p = pathlib.Path(root) / "models" / name
    _rep, info = check.check(str(p))
    ir = codegen.compile_ir(info)
    layers = []
    for i, L in enumerate(ir.get("layers") or []):
        ops = []
        for op in L.get("ops") or []:
            ops.append({
                "mech": op.get("mech"),
                "kind": op.get("kind"),
                "attrs": _trim_attrs(op.get("attrs")),
                "inputs": [str(x) for x in (op.get("inputs") or [])],
                "outputs": [str(x) for x in (op.get("outputs") or [])],
            })
        layers.append({"index": i, "ops": ops})
    tens = info.get("tens") or {}
    return {
        "name": name,
        "slug": _slug(name),
        "arch": info.get("arch"),
        "ctx": ir.get("ctx"),
        "d_model": ir.get("d_model"),
        "vocab": ir.get("vocab"),
        "norm_kind": ir.get("norm_kind"),
        "norm_one_plus": bool(ir.get("norm_one_plus")),
        "norm_eps": ir.get("norm_eps"),
        "pos_kind": ir.get("pos_kind"),
        "n_pos": ir.get("n_pos"),
        "hparams": {k: str(v) for k, v in (info.get("hparams") or {}).items()},
        "mechs": info.get("mechs") or {},
        "position": info.get("position") or {},
        "layers": layers,
        "globals": ir.get("globals") or [],
        "n_tensors": len(tens) if hasattr(tens, "__len__") else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 .by1 导成静态站点数据")
    ap.add_argument("--out", default="site/data", help="输出目录")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--all", action="store_true", help="导出 models/ 下全部")
    ap.add_argument("--root", default=None, help="仓库根（默认按脚本位置推断）")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root) if args.root else pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    import by1paths  # noqa: F401
    import by1check
    import by1codegen

    names = sorted(p.name for p in (root / "models").glob("*.by1"))
    if not args.all:
        names = [n for n in args.models if n in set(names)]

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for name in names:
        try:
            data = export_one(root, name, by1check, by1codegen)
        except Exception as exc:      # 部分描述（缺 vocab 等）导不了 -> 如实列出
            print("  跳过 %-32s %r" % (name, exc))
            index.append({"name": name, "slug": _slug(name), "error": str(exc)[:160]})
            continue
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        (out / (data["slug"] + ".js")).write_text(
            "window.BY1_MODELS = window.BY1_MODELS || {};\n"
            "window.BY1_MODELS[%s] = %s;\n" % (json.dumps(data["slug"]), body),
            encoding="utf-8", newline="\n")
        index.append({
            "name": data["name"], "slug": data["slug"], "arch": data["arch"],
            "layers": len(data["layers"]), "d_model": data["d_model"],
            "ctx": data["ctx"], "n_tensors": data["n_tensors"],
            "mechs": sorted(data["mechs"].keys()),
        })
        print("  %-32s 层 %-4d 张量 %s" % (name, len(data["layers"]), data["n_tensors"]))

    (out / "index.js").write_text(
        "window.BY1_INDEX = %s;\n" % json.dumps(index, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")
    print("  索引 %s（%d 份）" % (out / "index.js", len(index)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
