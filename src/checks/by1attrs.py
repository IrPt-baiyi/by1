#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1attrs -- **机制属性的闭集 与 官方出处**：闭集里不许躺着一个谁都说不清来历的取值。

## 为什么需要它

`by1codegen.ENUMS` 把 `act` / `qk_norm` / `routing` / `gate_act` / `pairing` 的合法取值钉死。
这是好事（取值不认识就拒绝，不许静默走默认分支）。但它只保证"取值在集合里"，**不保证
集合里每个取值都真的属于某个官方模型** —— 一个取值可能是当初为某个模型写的，那个模型
后来删了，取值却留在闭集里，从此没人说得清它对不对。

## 三条规则

    A1 每个取值都要有出处   ENUMS 里的每个 (属性, 取值) 都得在 PROVENANCE 里；
    A2 没有陈旧条目       PROVENANCE 里的键必须在 ENUMS 里（取值被删了，记录也要跟着删）；
    A3 出处要能对上模型   出处里的家族名要能在 `models/` 里有对应的 `.by1`
                     （或者是"旧写法"这类明确的别名）—— 否则出处只是一句话。

另外印一份 **清单**：`models/` 里**没有任何模型用到**的取值。它不算失败（有的取值
是为了接下来的模型准备的），但要让人看得见。
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: (属性, 取值) -> 出处。**每条都要能对一个官方实现。**
PROVENANCE = {
    ('act', 'silu'): 'Llama/Qwen 的 SwiGLU（transformers ACT2FN silu）',
    ('act', 'gptoss'): 'GPT-OSS 的 clamped swiglu（limit 7.0 + alpha 1.702）',
    ('act', 'gelu'): 'GPT-2 的精确 GELU（ACT2FN gelu）',
    ('act', 'gelu_new'): 'GPT-2 的 tanh 近似（transformers 叫 gelu_new / gelu_pytorch_tanh）',
    ('act', 'relu2'): 'Nemotron 的 squared-relu（ACT2FN relu2）',
    ('qk_norm', 'off'): '无 qk_norm（GPT-2 / 多数老模型）',
    ('qk_norm', 'per_head'): 'Qwen3 的 per-head RMSNorm（transformers Qwen3RMSNorm）',
    ('qk_norm', 'full'): 'Instella 的整宽 RMSNorm（拆头之前）',
    ('qk_norm', 'true'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', 'on'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', 'yes'): '旧写法，等价 per_head（by1langspec 记档）',
    ('qk_norm', '1'): '旧写法，等价 per_head（by1langspec 记档）',
    ('routing', 'softmax_topk'): 'Mixtral 的 top-k 后重归一（Switch/Mixtral 那一类）',
    ('routing', 'topk_softmax'): 'GPT-OSS 的 top-k 后再 softmax（GptOssTopKRouter）',
    ('routing', 'sigmoid_topk'): 'Step-3.7 的 sigmoid 打分直接 top-k',
    ('routing', 'sigmoid_group_topk'): 'DeepSeek V3/V4 与 GLM 与 Ling 的 noaux_tc（先分组再 top-k）',
    ('gate_act', 'softplus'): 'Laguna 的 head gate（LagunaAttention 用 F.softplus）',
    ('gate_act', 'sigmoid'): 'Ling 的 head gate（同名不同义）',
    ('pairing', 'half'): 'Llama 等 HF 模型的 apply_rotary_pos_emb（rotate_half）',
    ('pairing', 'interleaved'): 'DeepSeek 与 GLM 的 interleaved RoPE（索引器与 MLA 用）',
}


#: 闭集里**没有模型走到、但明确保留**的取值（A4 要它们）。
#: 每条都要写理由；一旦有模型用到它，这条就要删掉（否则 A4 报“残留”）。
RESERVED = {
    ('qk_norm', '1'): '旧写法（等价 per_head），by1langspec 记档；不会有模型这么写',
    ('qk_norm', 'on'): '旧写法（等价 per_head），by1langspec 记档',
    ('qk_norm', 'yes'): '旧写法（等价 per_head），by1langspec 记档',
}

#: 图原语里**没有语料走到、但明确保留**的（A5 要它们）。每条都要写理由。
PRIM_RESERVED = {
}


def _const_set(expr, table):
    """把「字面集合 / set(NAME) / A | B」求成集合；求不出返回 None。"""
    expr = expr.strip()
    try:
        v = ast.literal_eval(expr)
    except Exception:
        v = None
    if isinstance(v, dict):
        return set(v)
    if isinstance(v, (set, frozenset, list, tuple)):
        return set(v)
    m = re.fullmatch(r"set\((\w+)\)", expr)
    if m:
        base = table.get(m.group(1))
        return set(base) if base is not None else None
    if "|" in expr:
        out = set()
        for part in expr.split("|"):
            got = _const_set(part, table)
            if got is None:
                name = part.strip()
                if name in table:
                    out |= set(table[name])
                    continue
                return None
            out |= got
        return out
    if expr in table:
        return set(table[expr])
    return None


def _node_set(node, table):
    """AST 版本：字面集合 / set(NAME) / BitOr —— 多行字面量也能吃。"""
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys if isinstance(k, ast.Constant)}
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return {e.value for e in node.elts if isinstance(e, ast.Constant)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "set" and len(node.args) == 1 \
            and isinstance(node.args[0], ast.Name):
        base = table.get(node.args[0].id)
        return set(base) if base is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _node_set(node.left, table), _node_set(node.right, table)
        if left is not None and right is not None:
            return left | right
        return None
    if isinstance(node, ast.Name) and node.id in table:
        return set(table[node.id])
    return None


def emitter_prims(root=ROOT):
    """两个发射器各自**声明能发**的原语集合：(ggml, C)。

    静态扫 `prim == "x"` 与 `prim in (...)`/`prim in CONST`，并展开顶层常量
    （含 `_UN_SHAPE_OK = set(_UN_OP) | {"softmax"}` 这种）。不 import 它们
    （by1c 是脚本、by1ggml 要 gcc）。
    """
    out = {}
    for name in ("by1ggml.py", "by1c.py"):
        text = (pathlib.Path(root) / "src" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        table = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                got = _node_set(node.value, table)
                if got is not None:
                    table[node.targets[0].id] = got
        got = set()
        for m in re.finditer(r'prim\s*==\s*"([a-z_0-9]+)"', text):
            got.add(m.group(1))
        for m in re.finditer(r"prim\s+in\s*(\([^)]*\)|\[[^\]]*\]|[A-Z_][A-Z0-9_]*)", text):
            tok = m.group(1).strip()
            if tok[0] in "([":
                got |= _const_set(tok, table) or set()
            elif tok in table:
                got |= set(table[tok])
        out[name] = got
    return out["by1ggml.py"], out["by1c.py"]



def norm_switches(root=ROOT):
    """`norm` 的**开关别名集合**（从 by1codegen 的 `nv in (...)` 抽）+ 每个模型的取值。

    为什么单独盯它：`norm` 和 `norm_kind` 不是一回事 —— 后者有取值门（只收 rms/layer，
    写错就报错），而 `norm` **不过门**：`nv in ("zero_centered", ...)` 之外的一切都
    静默当成"普通约定"，数值差一个 2 倍（第 51 轮刚在 Qwen3-Next 上量过）。
    """
    code = (pathlib.Path(root) / "src/by1codegen.py").read_text(encoding="utf-8")
    m = re.search(r"nv in \(([^)]*)\)", code)
    aliases = tuple(x.strip().strip("\"'") for x in m.group(1).split(",") if x.strip()) if m else ()
    decls = {}
    for f in (pathlib.Path(root) / "models").rglob("*.by1"):
        text = f.read_text(encoding="utf-8", errors="replace")
        mm = re.search(r"^\s*norm\s*=\s*([A-Za-z_0-9+]+)", text, re.M)
        if mm:
            decls[f.name] = mm.group(1)
    return aliases, decls

def norm_pairs(root=ROOT):
    """**同一份权重被两份描述时，归一约定必须一致。**

    `models/` 里有 `<X>.by1` 与 `complete-<X>.by1` 成对出现的情形（手写版 / 由 config
    生成的全量版）。两份描述的是**同一个 checkpoint**；如果一份写了 `norm = zero_centered`、
    另一份没写，那它们对同一批权重给出**差 2 倍**的归一约定 —— 而这类模型通常只做
    config/张量名对拍、不跑前向，所以**没人会发现**。

    返回 [(手写名, 全量名, 手写取值, 全量取值), ...]（取值 "" 表示没写 = 普通约定）。
    """
    aliases, decls = norm_switches(root)
    out = []
    for name, val in sorted(decls.items()):
        pass
    names = {f.name for f in (pathlib.Path(root) / "models").rglob("*.by1")}
    for n in sorted(names):
        if n.startswith("complete-"):
            continue
        twin = "complete-" + n
        if twin in names:
            out.append((n, twin, decls.get(n, ""), decls.get(twin, "")))
    return out

def emit_symbols(root=ROOT, script="by1c.py"):
    """每个图原语在 C 发射器里**会出现的那几个符号**（从 by1c.py 的分支源码里抽）。

    只认 `p_xxx` / `ggml_xxx` / `by1_xxx` 三种前缀 —— 它们是发射器真正拼进产物里的名字。
    某个原语抽不出符号（纯表达式装配）就返回空集，A7 对它**不作要求**。
    """
    text = (pathlib.Path(root) / "src" / script).read_text(encoding="utf-8")
    lines = text.splitlines()
    heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"(\s*)(?:elif |if )prim(?: == \"([a-z_0-9]+)\"| in \(([^)]*)\))", ln)
        if m:
            names = [m.group(2)] if m.group(2) else re.findall('"([a-z_0-9]+)"', m.group(3))
            heads.append((i, [n for n in names if n]))
    sym = {}
    for idx, (start, names) in enumerate(heads):
        end = len(lines) if idx + 1 == len(heads) else heads[idx + 1][0]
        body = "\n".join(lines[start:end])
        found = set(re.findall(r"\b((?:p|ggml|by1)_[a-z0-9_]+)\b", body))
        for n in names:
            sym.setdefault(n, set()).update(found)
    return sym


#: A7 见证模型：按**覆盖所有有命名内核的原语**选的（贪心覆盖）。
#: 实测 5 份合起来 ±5 秒：前两份只能覆盖 8/22，加后三份补齐。
#: **一个节点 = 一次调用** 的原语表（A8）：实测 5 份产物都对得上才进这张表。
#: 反例已剔：`view`（会额外发 p_copy）、`get_rows`/`one_hot`（还有 _f 变体）、
#: `clamp_min`（实测 mla-sparse 里 4 个节点、0 次 p_clamp 调用 —— 被内联了）。
CALL_COUNT = {
    "batch_matmul": "p_bmm", "matmul": "p_matmul", "permute": "p_perm",
    "rms_norm": "p_rms_norm", "l2_norm": "p_l2_norm",
    "layer_norm": "p_layer_norm", "rope": "p_rope",
    "soft_max": "p_soft_max", "softmax": "p_softmax",
    "concat": "p_concat", "conv1d_causal": "p_conv1d_causal",
    "delta_rule": "p_delta_rule", "gather": "p_gather",
    "masked_fill": "p_masked_fill", "repeat": "p_repeat",
    "slice": "p_slice", "sum": "p_sum", "top_k": "p_top_k",
    "add_scalar": "p_add_scalar", "scale": "p_scale",
}

#: ggml 侧的一节点=一次调用表（A8）。
#: 实测 8 份产物共 14 组 1:1，但只取语义上真对应的 11 个：
#: softplus -> ggml_sqrt（sqrt 只是展开式里的一步）、
#: ggml_new_tensor_1d / ggml_map_custom1（通用分配器与回调）都剔掉了。
CALL_COUNT_GGML = {
    'gelu': 'ggml_gelu_erf', 'gelu_tanh': 'ggml_tanh',
    'layer_norm': 'ggml_norm', 'log1p': 'ggml_log',
    'rms_norm': 'ggml_rms_norm', 'sigmoid': 'ggml_sigmoid',
    'silu': 'ggml_silu', 'soft_max': 'ggml_soft_max_ext',
    'ssm_scan': 'ggml_ssm_scan', 'top_k': 'ggml_top_k',
    'delta_rule': 'ggml_gated_delta_net',
}

A7_MODELS = ("llama-shaped.by1", "gelu-shaped.by1", "ling-tiny.by1",
             "mla-sparse-shaped.by1", "ssm-shaped.by1")

#: ggml 侧的见证模型：它的符号体系和 C 侧不同（多 35 个有名内核），
#: 同样贪心覆盖：前 5 份盖 31/35，再加三份补齐 clamp_max / gelu_tanh / log1p / soft_max_sink。
A7_MODELS_GGML = A7_MODELS + ("gpt-oss-shaped.by1", "gpt2-tiny.by1",
                      "qwen3-next-shaped.by1")


def artifact_check(root=ROOT, models=A7_MODELS, emitter="c"):
    """A7：**产物里的证据** —— 生成的 C 里真的出现该原语的符号。

    跑 `by1c --from-graph` 生成一份 model.c，对模型图里每个原语查"它的分支符号"
    是否出现在产物里。分派表里有名字**不等于**产物里有调用。

    返回 (rows, problems)；rows 为 None 表示没法查（缺 gcc），由调用方当"跳过"处理。
    """
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf
    root = pathlib.Path(root)
    gcc = _sh.which("gcc")
    if not gcc:
        return None, [], "找不到 gcc（产物证据要对一份真编出来的 C 才能查）"
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import by1paths  # noqa: F401
    import by1check as _bc
    import by1codegen as _cg
    import by1graph as _bg
    script = "by1c.py" if emitter == "c" else "by1ggml.py"
    symbols = emit_symbols(root, script)
    problems, rows = [], []
    covered = set()
    pairs8 = 0
    total = len({p for p, v in symbols.items() if v})
    for name in models:
        p = root / "models" / name
        if not p.exists():
            continue
        try:
            _r, info = _bc.check(str(p))
            g = _bg.lower_model(_cg.compile_ir(info), name)
        except Exception:
            continue
        used = sorted({n["prim"] for n in g.nodes})
        covered |= {u for u in used if symbols.get(u)}
        wd = pathlib.Path(_tf.mkdtemp(prefix="by1a7-"))
        try:
            argv = [sys.executable, str(root / "src" / script),
                    "--gcc", gcc, "--seq", "16",
                    "--workdir", str(wd), str(p)]
            if emitter == "c":
                argv.insert(2, "--from-graph")
            run = _sp.run(argv,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=900)
            cfile = wd / "model.c"
            # **后端自己说"跳过"不是失败**：by1ggml 找不到 ggml 静态库时
            # 打印 [跳过] 并返回 30（CI 上没有那个构建）。这时 A7 对它是"没验"。
            if (run.returncode == 30
                    or "[跳过]" in (run.stdout or "")):
                return None, [], "%s 在这台机器上跳过（缺依赖）" % emitter
            if run.returncode != 0 or not cfile.exists():
                problems.append("A7 %s 没生成 C（rc=%s）" % (name, run.returncode))
                continue
            body = cfile.read_text(encoding="utf-8", errors="replace")
            missing = []
            for prim in used:
                want = symbols.get(prim) or set()
                if not want:
                    continue
                if not any(w in body for w in want):
                    missing.append("%s(%s)" % (prim, "/".join(sorted(want)[:2])))
            # A8：**节点数 = 调用次数**（只在 C 侧：ggml 的符号体系不同）。
            worse = []
            if True:
                counts = {}
                for nd in g.nodes:
                    counts[nd["prim"]] = counts.get(nd["prim"], 0) + 1
                _tbl = CALL_COUNT if emitter == "c" else CALL_COUNT_GGML
                for prim, sym in sorted(_tbl.items()):
                    k = counts.get(prim, 0)
                    if not k:
                        continue
                    calls = (len(re.findall(r"\b" + sym + r"\(", body))
                             - len(re.findall(r"void\s+" + sym + r"\(", body)))
                    pairs8 += 1
                    if calls != k:
                        worse.append("%s 节点%d/调用%d" % (prim, k, calls))
            if worse:
                problems.append("A8 %s：图与产物的调用次数对不上：%s"
                                % (name, "，".join(worse[:4])))
            rows.append((name, len(used), len(missing)))
            if missing:
                problems.append("A7 %s 的产物里找不到这些原语的符号：%s"
                                % (name, ", ".join(missing[:4])))
        finally:
            _sh.rmtree(wd, ignore_errors=True)
    return rows, problems, {"covered": len(covered), "total": total,
                               "pairs8": pairs8}

#: 只在某个发射器里有、另一个没有的原语，若确实是有意的，写在这里（A6 要它）。
EMIT_ASYM = {
    # by1ggml 里 `log` / `neg` 是从别的上下文扫进来的（不是 PRIMS 的分派），
    # 它们在 PRIMS 里也不存在 —— 不用管。
}

def prim_coverage(root=ROOT):
    """(被语料走到的原语, 未被走到的原语)。语料 = by1all 的图探针表。

    静态依赖 by1check/by1codegen/by1graph **都不需要 torch** —— 所以这条能在档 2 跑。
    """
    # 把 src/ 放上 sys.path（子目录由 by1paths 负责）
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import by1paths  # noqa: F401
    import by1check as _bc
    import by1codegen as _cg
    import by1graph as _bg
    root = pathlib.Path(root)
    # **用 AST 取列表**，不要用正则：表里夹着注释，注释里的 `]` 会把正则截断
    # （实测只取到 19/22 个模型，于是新加的那个模型没被看见）。
    tree = ast.parse((root / "src/checks/by1all.py").read_text(encoding="utf-8"))
    models = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "GGML_GRAPH_MODELS":
            models = list(ast.literal_eval(node.value))
            break
    house = {"ins", "src", "f", "ref"}
    all_prims = {k for k in _bg.PRIMS if k not in house}
    used = set()
    for name in models:
        p = root / "models" / name
        if not p.exists():
            continue
        try:
            _r, info = _bc.check(str(p))
            ir = _cg.compile_ir(info)
            g = _bg.lower_model(ir, name)
        except Exception:
            continue      # 降不动的在别处有账（graph-golden 的"15 份降不动"）
        used.update(n["prim"] for n in g.nodes)
    return used & all_prims, sorted(all_prims - used)

def codegen_defaults(codegen_path=None):
    """从 `by1codegen.py` 取每个机制类型的属性默认值（`attrs.get("x", "y")`）。"""
    p = pathlib.Path(codegen_path or (ROOT / "src/by1codegen.py"))
    text = p.read_text(encoding="utf-8")
    out, kind = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*if kind == "(\w+)":', line)
        if m:
            kind = m.group(1)
            out.setdefault(kind, {})
            continue
        if kind is None:
            continue
        for a, v in re.findall(r'attrs\.get\("(\w+)", "([^"]+)"\)', line):
            out[kind].setdefault(a, v)
    return out


def implicit_values(root=ROOT):
    """模型里**没写**、但编译器会按默认值用到的 (属性, 取值)。"""
    defaults = codegen_defaults(pathlib.Path(root) / "src/by1codegen.py")
    seen = set()
    pat_mech = re.compile(r"mech\s+(\w+)\s*:\s*(\w+)\s*\{([^}]*)\}")
    for f in (pathlib.Path(root) / "models").rglob("*.by1"):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in pat_mech.finditer(text):
            body, kind = m.group(3), m.group(2)
            for attr, val in (defaults.get(kind) or {}).items():
                if re.search(r"\b" + attr + r"\s*=", body):
                    continue      # 显式写了，不算隐式
                if attr == "gate_act":
                    hg = re.search(r"head_gate\s*=\s*([A-Za-z_0-9]+)", body)
                    if not hg or hg.group(1).lower() == "off":
                        continue  # 没有门就不会用到激活
                seen.add((attr, val.lower()))
    return seen

def load_enums(codegen_path=None):
    """从 `by1codegen.py` **静态**取 ENUMS（不 import：那会把 torch 拖进来）。"""
    p = pathlib.Path(codegen_path or (ROOT / 'src/by1codegen.py'))
    tree = ast.parse(p.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == 'ENUMS':
            return {k: set(v) for k, v in ast.literal_eval(node.value).items()}
    return {}


def model_families(root=ROOT):
    """`models/` 与 `refs/` 里出现过的家族名（小写、拼接，含首段前缀）。

    出处可以指一份 `models/*.by1`，也可以指一个 refs 语料（真模型的 config）。
    """
    out = set()
    stems = []
    for f in (pathlib.Path(root) / 'models').rglob('*.by1'):
        stems.append(f.stem.lower())
    refs = pathlib.Path(root) / 'refs'
    if refs.is_dir():
        for f in refs.iterdir():
            if f.is_file():
                stems.append(f.name.split('.')[0].lower())
    for stem in stems:
        flat = re.sub(r'[^a-z0-9]', '', stem)
        out.add(flat)
        for part in re.split(r'[^a-z]+', stem):
            if len(part) >= 4:
                out.add(part)
        m = re.match(r'[a-z]+', stem)
        if m:
            out.add(m.group(0))
    return out

def used_values(root=ROOT):
    """`models/` 里真正出现过的 (属性, 取值)（正则扫文本）。"""
    seen = set()
    for f in (pathlib.Path(root) / 'models').rglob('*.by1'):
        text = f.read_text(encoding='utf-8', errors='replace')
        for key in ('act', 'qk_norm', 'routing', 'gate_act', 'pairing'):
            for m in re.finditer(r'\b' + key + r'\s*=\s*([A-Za-z_0-9]+)', text):
                seen.add((key, m.group(1).lower()))
    return seen


def check(root=ROOT, enums=None, provenance=None, reserved=None):
    root = pathlib.Path(root)
    enums = load_enums(root / 'src/by1codegen.py') if enums is None else enums
    prov = PROVENANCE if provenance is None else provenance
    RES = RESERVED if reserved is None else reserved
    fams = model_families(root)
    problems = []
    notes = []
    if not enums:
        return (['ENUMS 取不出来（by1codegen.py 改名了？）—— 这条检查会永远绿'], [])
    known = set()
    for attr, vals in enums.items():
        for v in vals:
            known.add((attr, str(v).lower()))
    for attr, v in sorted(known):
        if (attr, v) not in prov:
            problems.append('A1 %s = <%s> 没有出处（闭集里不许有说不清来历的取值）'
                            % (attr, v))
    for attr, v in sorted(prov):
        if (attr, str(v).lower()) not in known:
            problems.append('A2 出处里有一条 %s = <%s>，但闭集里没有它（陈旧条目）'
                            % (attr, v))
    for (attr, v), why in sorted(prov.items()):
        if (attr, str(v).lower()) not in known:
            continue
        if '旧写法' in why or '别名' in why:
            continue
        low = why.lower()
        flat = re.sub(r'[^a-z0-9]', '', low)
        # 家族前缀（llama / gpt / qwen …）：出处里提到一个就算对得上
        prefixes = set()
        for fam in fams:
            m = re.match(r'[a-z]+', fam)
            if m:
                prefixes.add(m.group(0))
        if not any(pre in flat for pre in prefixes):
            problems.append('A3 %s = <%s> 的出处 %r 对不上 models/ 里的任何模型'
                            % (attr, v, why[:40]))
    # A4：每个取值要么**有模型走到**（显式写出来 或 靠默认值生效），
    # 要么在 RESERVED 里**写明保留理由** —— 不允许“谁也不知道为什么它在”。
    eff = set(used_values(root)) | set(implicit_values(root))
    for attr, v in sorted(known):
        if (attr, v) in eff:
            continue
        if (attr, v) not in RES:
            problems.append('A4 %s = <%s> 没有任何模型用到，也没在 RESERVED '
                            '里写保留理由（闭集里不允许谁也不知道为什么它在）'
                            % (attr, v))
    for attr, v in sorted(RES):
        if (attr, str(v).lower()) not in known:
            problems.append('A4 RESERVED 里有一条 %s = <%s>，但闭集里没有它'
                            % (attr, v))
        elif (attr, str(v).lower()) in eff:
            problems.append('A4 RESERVED 里 %s = <%s> 已经有模型用到了 —— '
                            '这条保留记录该删了' % (attr, v))
    # A5：图原语“闭集”也要有语料走到 —— 语料就是 by1all 的图探针表。
    #
    # **没有语料就跳过**（自检的假树里没有 by1all.py）—— 否则这条会在假树上全盘报红。
    try:
        used_p, unused_p = prim_coverage(root)
    except (ImportError, FileNotFoundError) as exc:
        notes.append('图原语覆盖没算（缺依赖）：%r' % (exc,))
    except Exception as exc:
        # **其它异常不能当跳过**：第三十五轮就遇到过一次（by1graph 被改出了缩进错，A5/A6 只印一行“没算”就绿了）。
        problems.append('图原语覆盖算不了：%r' % (exc,))
    else:
        if used_p or unused_p:
            for p in unused_p:
                if p not in PRIM_RESERVED:
                    problems.append('A5 图原语 <%s> 没有任何图探针模型走到，'
                                    '也没在 PRIM_RESERVED 里写理由' % p)
            for p in sorted(PRIM_RESERVED):
                if p in used_p:
                    problems.append('A5 PRIM_RESERVED 里的 <%s> 已经有模型走到了 '
                                    '—— 这条保留记录该删了' % p)
    # A6：每个图原语要**两个发射器都能发**。
    try:
        g_set, c_set = emitter_prims(root)
        prims_real = _real_prims()
    except (ImportError, FileNotFoundError) as exc:
        notes.append('发射器覆盖没算（缺依赖）：%r' % (exc,))
    except Exception as exc:
        problems.append('发射器覆盖算不了：%r' % (exc,))
    else:
        if prims_real:
            for p in sorted(prims_real):
                if p in EMIT_ASYM:
                    continue
                missing = [n for n, s in (('ggml', g_set), ('C', c_set)) if p not in s]
                if missing:
                    problems.append('A6 图原语 <%s> 在 %s 发射器里没有映射'
                                    % (p, ' / '.join(missing)))
            for p in sorted(EMIT_ASYM):
                if p not in prims_real:
                    problems.append('A6 EMIT_ASYM 里的 <%s> 不在 PRIMS 里了' % p)
    # A7：**产物里的证据** —— 分派表里有名字不等于产物里有调用。
    for side, side_models in (('c', A7_MODELS), ('ggml', A7_MODELS_GGML)):
        try:
            rows7, probs7, why7 = artifact_check(root, side_models, side)
        except (ImportError, FileNotFoundError) as exc:
            notes.append('产物证据没算（缺依赖/缺语料）：%r' % (exc,))
        except Exception as exc:
            problems.append('产物证据算不了：%r' % (exc,))
        else:
            if rows7 is None:
                notes.append('产物证据没算：' + (why7 or ''))
            else:
                problems.extend(probs7)
    # A9：`norm` 是**不过取值门**的开关 —— 拼错会静默按普通约定算（差 2 倍）。
    try:
        aliases, decls = norm_switches(root)
    except Exception as exc:
        notes.append('norm 开关没算：%r' % (exc,))
    else:
        for model, val in sorted(decls.items()):
            if val.lower() not in [a.lower() for a in aliases]:
                problems.append('A9 %s 的 hparams.norm = <%s> 不在开关集合 %r 里 —— '
                                'codegen 会静默当成普通约定（与 one_plus 差 2 倍）'
                                % (model, val, list(aliases)))
        if decls:
            on = sum(1 for v in decls.values() if v.lower() in [a.lower() for a in aliases])
            notes.append('写 norm 的模型 %d 个，其中 one_plus 类 %d 个（%s）'
                         % (len(decls), on, '/'.join(sorted(decls))))
    # A10：成对描述（`<X>` 与 `complete-<X>`）的归一约定必须一致（**只报提示**：
    # 谁对谁错要拿 checkpoint 的权重统计或官方实现来定，本检查里不臆断）。
    try:
        pairs = norm_pairs(root)
    except Exception as exc:
        notes.append('成对描述的归一约定没比：%r' % (exc,))
    else:
        for a_name, b_name, av, bv in pairs:
            if av.strip().lower() != bv.strip().lower():
                notes.append('**同一份权重、两份描述的归一约定不一致**：%s 写 <%s>，%s 写 <%s>'
                             ' —— 差 2 倍的那类错，先查 checkpoint'
                             % (a_name, av or '（没写=普通）', b_name, bv or '（没写=普通）'))
    return problems, notes


def _real_prims():
    """PRIMS 里看得见的那些（去掉 ins/src/f/ref 这种传导项）。"""
    import by1paths  # noqa: F401
    import by1graph as _bg
    return {k for k in _bg.PRIMS if k not in ('ins', 'src', 'f', 'ref')}


def _selftest():
    import shutil
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix='by1attrs-'))
    bad = []
    try:
        (d / 'src').mkdir(parents=True)
        (d / 'models').mkdir(parents=True)
        (d / 'models/llama-shaped.by1').write_text('x\nact = silu\n',
                                                   encoding='utf-8')
        codegen = d / 'src/by1codegen.py'

        def enums_text(enums):
            body = ', '.join('%r: {%s}' % (k, ', '.join(repr(x) for x in v))
                             for k, v in enums.items())
            return 'ENUMS = {' + body + '}\n'

        def run(enums, prov, reserved=None):
            codegen.write_text(enums_text(enums), encoding='utf-8')
            if reserved is None:
                reserved = {} if enums != {'act': {'silu'}} else {}
            return check(root=d, enums=enums, provenance=prov,
                         reserved=reserved if reserved is not None else {})

        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU'})
        if probs:
            bad.append('正例被误判：%r' % (probs,))
        probs, _n = run({'act': {'silu', 'mystery'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU'})
        if not [p for p in probs if 'A1' in p]:
            bad.append('A1 没抓到：%r' % (probs,))
        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'Llama 的 SwiGLU',
                         ('act', 'gone'): 'Llama 的旧取值'})
        if not [p for p in probs if 'A2' in p]:
            bad.append('A2 没抓到：%r' % (probs,))
        probs, _n = run({'act': {'silu'}},
                        {('act', 'silu'): 'SomeVendor 的神秘实现'})
        if not [p for p in probs if 'A3' in p]:
            bad.append('A3 没抓到：%r' % (probs,))
        prov2 = {('act', 'silu'): 'Llama 的 SwiGLU',
                 ('act', 'gelu'): 'Llama 那边也有的 GELU'}
        # A4：没模型用、也没保留理由 -> 红
        probs, _n = run({'act': {'silu', 'gelu'}}, prov2, reserved={})
        if not [p for p in probs if 'A4' in p]:
            bad.append('A4 没抓到没人用的取值：%r' % (probs,))
        # A4：写了保留理由 -> 不红
        probs, _n = run({'act': {'silu', 'gelu'}}, prov2,
                        reserved={('act', 'gelu'): '保留着'})
        if probs:
            bad.append('A4 保留条目被误判：%r' % (probs,))
        # A4：保留了但已经有人用 -> 残留
        probs, _n = run({'act': {'silu'}}, {('act', 'silu'): 'Llama 的 SwiGLU'},
                        reserved={('act', 'silu'): '保留着'})
        if not [p for p in probs if 'A4' in p]:
            bad.append('A4 没抓到残留的保留条目：%r' % (probs,))
        # A9：norm 写错一个字 -> 必须报错（不许静默按普通约定算）
        (d / 'models' / 'bad-norm.by1').write_text(
            'hparams { d_model = 8, vocab = 16 }' + chr(10)
            + '  norm = zero_centred' + chr(10),
            encoding='utf-8')
        (d / 'src/by1codegen.py').write_text(
            'ENUMS = {}' + chr(10) + 'nv in ("zero_centered", "one_plus")' + chr(10),
            encoding='utf-8')
        probs, _n = check(root=d, enums={'act': {'silu'}},
                        provenance={('act', 'silu'): 'Llama 的 SwiGLU'},
                        reserved={})
        if not [p for p in probs if 'A9' in p]:
            bad.append('A9 没抓到 norm 拼错：%r' % (probs,))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for b in bad:
        print('     ! ' + b)
    print('  [%s] 机制属性的出处：A1/A2/A3/A4/A9 各红一次，正例不误伤'
          % ('PASS' if not bad else 'FAIL'))
    return 0 if not bad else 1


def _art():
    """A7 的一句话摘要：几份产物、多少个原语缺符号。"""
    parts = []
    for side, side_models in (('c', A7_MODELS), ('ggml', A7_MODELS_GGML)):
        try:
            rows, _probs, info = artifact_check(ROOT, side_models, side)
        except Exception:
            parts.append('%s 没算' % side)
            continue
        if rows is None:
            parts.append('%s 没算' % side)
            continue
        miss = sum(r[2] for r in rows)
        cov = (info or {}).get('covered', 0) if isinstance(info, dict) else 0
        tot = (info or {}).get('total', 0) if isinstance(info, dict) else 0
        pair = (info or {}).get('pairs8', 0) if isinstance(info, dict) else 0
        extra = (' · 调用数核对 %d 处' % pair) if pair else ''
        parts.append('%s %d 份/%d/%d 缺 %d%s'
                     % (side, len(rows), cov, tot, miss, extra))
    return ' · '.join(parts)

def _emit():
    """(两个发射器都声明的原语数, PRIMS 里的原语数)。"""
    try:
        g_set, c_set = emitter_prims(ROOT)
        prims = _real_prims()
        return len({p for p in prims if p in g_set and p in c_set}), len(prims)
    except Exception:
        return 0, 0


def _cov():
    try:
        used, unused = prim_coverage(ROOT)
        return len(used), len(unused)
    except Exception:
        return 0, 0


def main(argv):
    # `--artifacts` 就是默认路径（只为满足 JUDGE_CMDS 的"<file> <subcommand>" 约定）
    argv = [a for a in argv if a == '--selftest']
    if '--selftest' in argv:
        return _selftest()
    problems, notes = check(ROOT)
    for x in problems:
        print('     ! ' + x)
    for n in notes:
        print('  注：' + n)
    ok = not problems
    print('  [%s] 机制属性的出处 %s'
          % ('PASS' if ok else 'FAIL',
             '属性 %d 个取值都有出处；图原语 %d/%d；两个发射器 %d/%d；产物里的证据 %s'
             % (len(PROVENANCE), _cov()[0], _cov()[0] + _cov()[1], _emit()[0], _emit()[1], _art())
             if ok else '%d 处不成立' % len(problems)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
