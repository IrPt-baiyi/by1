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
import re
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
    import by1graph

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
        # 图（M3）：同样逐字节比，而且对照 graphs-golden.json
        gf = out / (slug + ".graph.js")
        if not gf.exists():
            continue
        gp = root / "graphs-golden.json"
        golden = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
        try:
            gd = export_graph(root, name, by1check, by1codegen, by1graph, golden)
        except Exception:
            continue
        gwant = ("window.BY1_GRAPH = window.BY1_GRAPH || {};\n"
                 "window.BY1_GRAPH[%s] = %s;\n"
                 % (json.dumps(slug), json.dumps(gd, ensure_ascii=False,
                                                 separators=(",", ":"))))
        if gf.read_text(encoding="utf-8") != gwant:
            bad.append("%s 的图与现算不一致（重新导出：python tools/by1site.py）" % slug)
        elif gd.get("match") is False:
            bad.append("%s 的图 sha 与 graphs-golden.json 不同（先重生成 golden）" % slug)
        else:
            ok += 1
    # 对比（M4）：每份 diff 文件也逐字节比，索引同样比
    dif = out / "diff"
    if dif.is_dir():
        for dj in sorted(dif.glob("*.js")):
            if dj.name == "index.js":
                continue
            slug = dj.name[:-3]
            a_name = b_name = None
            for x, y in DIFF_PAIRS:
                if _slug(x) + "__" + _slug(y) == slug:
                    a_name, b_name = x, y
                    break
            if a_name is None:
                bad.append("%s 不在 DIFF_PAIRS 里（哪来的？）" % slug)
                continue
            try:
                d = export_diff(root, a_name, b_name, by1check, by1codegen)
            except Exception:
                continue
            want = ("window.BY1_DIFF = window.BY1_DIFF || {};\n"
                    "window.BY1_DIFF[%s] = %s;\n"
                    % (json.dumps(slug), json.dumps(d, ensure_ascii=False,
                                                    separators=(",", ":"))))
            if dj.read_text(encoding="utf-8") != want:
                bad.append("%s 的对比与现算不一致（重新导出：python tools/by1site.py --diffs）"
                           % slug)
            else:
                ok += 1

    return (0 if not bad else 1), (bad or ["站点数据（含图/对比）与 IR 一致：%d 份" % ok])


FULL_NODES = 600      # 节点数不超过它就把节点表也导出来

def export_graph(root, name, check, codegen, graph, golden):
    """一份模型 -> 原语图（M3）：sha / 节点数 / 每个原语的计数 / 每层的原语链。

    大模型的节点不逐条导出（Qwen3.8-27B 有 5093 个），只导摘要；
    节点数 <= FULL_NODES 的（shaped 探针、小模型）连节点表一起导，前端可以点开看属性。
    """
    p = pathlib.Path(root) / "models" / name
    _rep, info = check.check(str(p))
    ir = codegen.compile_ir(info)
    g = graph.lower_model(ir, name)
    hist = {}
    layers = {}
    for nd in g.nodes:
        prim = nd.get("prim")
        hist[prim] = hist.get(prim, 0) + 1
        lay = None
        for out in (nd.get("outs") or []):
            m = re.match(r"^L(\d+)", str(out))
            if m:
                lay = int(m.group(1))
                break
        layers.setdefault(-1 if lay is None else lay, []).append(prim)
    sha = g.sha()
    want = golden.get(name)
    data = {
        "slug": _slug(name),
        "name": name,
        "sha": sha,
        "golden": want,
        "match": (want == sha) if want else None,
        "node_count": len(g.nodes),
        "prims": dict(sorted(hist.items())),
        "layers": [{"index": i, "prims": layers[i]} for i in sorted(layers)],
        "nodes": None,
    }
    if len(g.nodes) <= FULL_NODES:
        data["nodes"] = [
            {"i": i, "prim": nd.get("prim"),
             "ins": [str(x) for x in (nd.get("ins") or [])],
             "outs": [str(x) for x in (nd.get("outs") or [])],
             "attrs": _trim_attrs(nd.get("attrs"))}
            for i, nd in enumerate(g.nodes)
        ]
    return data


#: M4 的对比清单：**只导这几对**（组合爆炸没必要，这几对各自压着不同的东西）。
DIFF_PAIRS = (
    ("Qwen3.8-27B.by1", "complete-Qwen3.8-27B.by1"),   # 手写 vs 由 config 生成
    ("clef.by1", "clef-tiny.by1"),                     # 同一家族的两个尺度
    ("llama-shaped.by1", "llama3-shaped.by1"),         # rope 基值 / 词表
    ("qwen3-next-shaped.by1", "laguna-tiny.by1"),      # 线性+全量 混合 vs 纯全量
    ("gpt-oss-shaped.by1", "mixtral-shaped.by1"),      # 两种 MoE 路由
)


def _flat_attrs(model):
    """(层, 算子下标, 机制, 属性) -> 值。"""
    out = {}
    for L in model.get("layers") or []:
        for j, op in enumerate(L.get("ops") or []):
            for k, v in (op.get("attrs") or {}).items():
                out["%d/%d/%s/%s" % (L["index"], j, op.get("mech"), k)] = v
    return out


def export_diff(root, a_name, b_name, check, codegen):
    """两份模型的 IR 差异（M4）。**全部从 IR 现算**，页面只展示。"""
    ia = export_one(root, a_name, check, codegen)
    ib = export_one(root, b_name, check, codegen)
    fields = []
    for key, fa, fb in (
            ("arch", ia.get("arch"), ib.get("arch")),
            ("ctx", ia.get("ctx"), ib.get("ctx")),
            ("d_model", ia.get("d_model"), ib.get("d_model")),
            ("vocab", ia.get("vocab"), ib.get("vocab")),
            ("norm", "%s/%s" % (ia.get("norm_kind"), "1+w" if ia.get("norm_one_plus") else "w"),
             "%s/%s" % (ib.get("norm_kind"), "1+w" if ib.get("norm_one_plus") else "w")),
            ("layers", len(ia.get("layers") or []), len(ib.get("layers") or [])),
            ("tensors", ia.get("n_tensors"), ib.get("n_tensors")),
    ):
        fields.append({"key": key, "a": fa, "b": fb, "same": fa == fb})

    ca = [[(op.get("mech"), op.get("kind")) for op in (L.get("ops") or [])]
          for L in (ia.get("layers") or [])]
    cb = [[(op.get("mech"), op.get("kind")) for op in (L.get("ops") or [])]
          for L in (ib.get("layers") or [])]
    layer_diffs = []
    for i in range(max(len(ca), len(cb))):
        va = ca[i] if i < len(ca) else None
        vb = cb[i] if i < len(cb) else None
        if va != vb:
            layer_diffs.append({
                "layer": i,
                "a": None if va is None else [m + "(" + k + ")" for m, k in va],
                "b": None if vb is None else [m + "(" + k + ")" for m, k in vb],
            })

    fa, fb = _flat_attrs(ia), _flat_attrs(ib)
    changed = []
    for k in sorted(set(fa) & set(fb)):
        if fa[k] != fb[k]:
            changed.append({"k": k, "a": fa[k], "b": fb[k]})
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    return {
        "a": a_name, "b": b_name,
        "a_slug": _slug(a_name), "b_slug": _slug(b_name),
        "fields": fields,
        "layer_diffs": layer_diffs,
        "attr_changed": changed[:200],
        "counts": {
            "fields_same": sum(1 for f in fields if f["same"]),
            "fields_total": len(fields),
            "layer_diff": len(layer_diffs),
            "attr_changed": len(changed),
            "attr_only_a": len(only_a),
            "attr_only_b": len(only_b),
        },
        "attr_only_a": only_a[:60],
        "attr_only_b": only_b[:60],
    }


def write_diff(root, out, data):
    """写一份 diff 的 .js，并返回索引条目。"""
    slug = data["a_slug"] + "__" + data["b_slug"]
    d = out / "diff"
    d.mkdir(parents=True, exist_ok=True)
    (d / (slug + ".js")).write_text(
        "window.BY1_DIFF = window.BY1_DIFF || {};\n"
        "window.BY1_DIFF[%s] = %s;\n"
        % (json.dumps(slug), json.dumps(data, ensure_ascii=False, separators=(",", ":"))),
        encoding="utf-8", newline="\n")
    return {"slug": slug, "a": data["a"], "b": data["b"],
            "layer_diff": data["counts"]["layer_diff"],
            "attr_changed": data["counts"]["attr_changed"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 .by1 导成静态站点数据")
    ap.add_argument("--out", default="site/data", help="输出目录")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--all", action="store_true", help="导出 models/ 下全部")
    ap.add_argument("--root", default=None, help="仓库根（默认按脚本位置推断）")
    ap.add_argument("--check", action="store_true", help="只比对磁盘数据与 IR（不写文件）")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="导出两份模型的 IR 差异")
    ap.add_argument("--diffs", action="store_true", help="导出 DIFF_PAIRS 全部")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root) if args.root else pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    import by1paths  # noqa: F401
    import by1check
    import by1codegen

    import by1graph
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
    if args.diff or args.diffs:
        pairs = [tuple(args.diff)] if args.diff else list(DIFF_PAIRS)
        di = []
        for a_name, b_name in pairs:
            try:
                d = export_diff(root, a_name, b_name, by1check, by1codegen)
            except Exception as exc:
                print("  对比导不出 %s vs %s: %r" % (a_name, b_name, exc))
                continue
            di.append(write_diff(root, out, d))
            c = d["counts"]
            print("  %-42s 层差 %-4d 属性差 %-4d"
                  % (d["a_slug"] + " vs " + d["b_slug"],
                     c["layer_diff"], c["attr_changed"]))
        (out / "diff").mkdir(parents=True, exist_ok=True)
        (out / "diff" / "index.js").write_text(
            "window.BY1_DIFF_INDEX = %s;\n" % json.dumps(di, ensure_ascii=False, indent=1),
            encoding="utf-8", newline="\n")
        print("  对比索引 %s（%d 对）" % (out / "diff" / "index.js", len(di)))
        return 0

    gp = root / "graphs-golden.json"
    golden = {}
    if gp.exists():
        golden = json.loads(gp.read_text(encoding="utf-8"))

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
        try:
            gd = export_graph(root, name, by1check, by1codegen, by1graph, golden)
            (out / (data["slug"] + ".graph.js")).write_text(
                "window.BY1_GRAPH = window.BY1_GRAPH || {};\n"
                "window.BY1_GRAPH[%s] = %s;\n" % (json.dumps(data["slug"]),
                                            json.dumps(gd, ensure_ascii=False,
                                                       separators=(",", ":"))),
                encoding="utf-8", newline="\n")
            print("  %-32s 图 %-5d 节点 sha %s%s"
                  % (name, gd["node_count"], gd["sha"],
                     "" if gd["match"] in (None, True) else " (与 golden 不同!)"))
        except Exception as exc:
            print("  %-32s 图导不出 %r" % (name, exc))

    (out / "index.js").write_text(
        "window.BY1_INDEX = %s;\n" % json.dumps(index, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")
    print("  索引 %s（%d 份）" % (out / "index.js", len(index)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
