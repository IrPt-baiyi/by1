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


def _count_tensors(tens):
    """张量**个数**（不是分组数）。

    by1check 的 tens 是分组表：[(机制, 属性签名, 张量数, [(名字, 形状, ...)]), ...]；
    直接 len() 拿到的是组数（3、7），不是 1199 那种个数。
    """
    if not tens:
        return 0
    total = 0
    for item in tens:
        try:
            # item = (机制, 属性签名, 展开层数, 每层张量表)
            total += int(item[2]) * len(item[3] or [])
        except (TypeError, ValueError, IndexError):
            continue
    return total


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
        "n_tensors": _count_tensors(tens),
    }


def check_site(root, out):
    """**站点数据必须等于从 IR 重新导出的那份**（判据，不是生成）。

    为什么需要：页面上的数字如果只靠'当初导出时是对的'，它就只是又一本文档。
    这条把磁盘上的 .js 与 by1check/by1codegen 现算的结果逐字节比一遍。
    """
    sys.path.insert(0, str(root / "src"))
    import by1paths  # noqa: F401
    import by1check
    import by1codegen

    index_path = out / "index.js"
    if not index_path.exists():
        return 1, ["缺 %s（先跑一次不带 --check 的导出）" % index_path]
    names = sorted(p.name for p in (root / "models").glob("*.by1"))
    bad, ok = [], 0
    seen = set()
    for name in names:
        try:
            data = export_one(root, name, by1check, by1codegen)
        except Exception:
            continue
        slug = data["slug"]
        f = out / (slug + ".js")
        if not f.exists():
            continue      # 没导出的模型不算不一致（--all 是可选范围）
        seen.add(slug)
        want = ("window.BY1_MODELS = window.BY1_MODELS || {};\n"
                "window.BY1_MODELS[%s] = %s;\n"
                % (json.dumps(slug), json.dumps(data, ensure_ascii=False,
                                               separators=(",", ":"))))
        got = f.read_text(encoding="utf-8")
        if got != want:
            bad.append("%s 与 IR 不一致（重新导出：python tools/by1site.py）" % slug)
        else:
            ok += 1
    return (0 if not bad else 1), (bad or ["站点数据与 IR 一致：%d 份" % ok])


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 .by1 导成静态站点数据")
    ap.add_argument("--out", default="site/data", help="输出目录")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--all", action="store_true", help="导出 models/ 下全部")
    ap.add_argument("--root", default=None, help="仓库根（默认按脚本位置推断）")
    ap.add_argument("--check", action="store_true", help="只比对磁盘数据与 IR（不写文件）")
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
    if args.check:
        rc, msgs = check_site(root, out)
        for m in msgs:
            print(("  [%s] " % ("PASS" if rc == 0 else "FAIL")) + m)
        return rc
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
