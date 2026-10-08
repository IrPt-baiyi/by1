#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 check -- 不变式检查器 v0.1

只做两件事:
  1) 名称解析   : 谁引用了未声明的符号
  2) 算术不变式 : 声明值 vs 可由声明推出的值
副产品: 展开层表 / 状态尺寸表

不做: 代码生成、类型推导、lowering。

用法:  python by1check.py <file.by1> [more.by1 ...]
"""

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════
# 词法与数值
# ════════════════════════════════════════════════════════════════════

NUM_RE = re.compile(r"-?\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?")
TOKEN_RE = re.compile(
    r"\s*(?:(\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|([A-Za-z_][A-Za-z_0-9]*)|([-+*/()^])|(.))"
)


def to_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s.strip().replace("_", ""))
    except ValueError:
        return None


def leading_num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = NUM_RE.match(s.strip())
    return to_num(m.group(0)) if m else None


def _fmt(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def parse_state_key(key: str):
    """'GQA[window != none]' -> ('GQA', [('window','!=','none')])"""
    key = key.strip()
    if "[" not in key:
        return key, []
    name = key[: key.index("[")].strip()
    inner = key[key.index("[") + 1 : key.rindex("]")]
    conds = []
    for part in split_top(inner):
        for op in ("!=", "==", "="):
            if op in part:
                a, b = part.split(op, 1)
                conds.append((a.strip(), "!=" if op == "!=" else "=", b.strip()))
                break
    return name, conds


def selector_matches(conds, attrs: Dict[str, str]) -> bool:
    for a, op, b in conds:
        v = str(attrs.get(a, "")).strip()
        if op == "=" and v != b:
            return False
        if op == "!=" and v == b:
            return False
    return True


def split_selector(sel: str):
    """'main[1..39]' -> ('main', '1..39')；'main' -> ('main', ':')"""
    m = re.match(r"\s*([A-Za-z_]\w*)\s*(?:\[\s*(.*?)\s*\])?\s*$", sel or "")
    if not m:
        return (sel or "").strip(), ":"
    return m.group(1), (m.group(2) or ":")


def make_layer_pred(inner: str):
    """把一个层选择器变成谓词 pred(下标, 属性) -> bool。"""
    inner = (inner or ":").strip()
    if inner in (":", ""):
        return lambda i, a: True
    mw = re.match(r"where\s+(.+)$", inner)
    if mw:
        _n, conds = parse_state_key("X[" + mw.group(1) + "]")
        return lambda i, a, c=conds: selector_matches(c, a)
    ms = re.match(r"step\s+(\d+)$", inner)
    if ms:
        k = max(1, int(ms.group(1)))
        return lambda i, a, k=k: i % k == 0
    mr = re.match(r"^(-?\d+)?\s*\.\.\s*(-?\d+)?$", inner)
    if mr:
        lo = int(mr.group(1)) if mr.group(1) else None
        hi = int(mr.group(2)) if mr.group(2) else None
        return lambda i, a, lo=lo, hi=hi: ((lo is None or i >= lo)
                                          and (hi is None or i <= hi))
    if re.match(r"^-?\d+$", inner):
        k = int(inner)
        return lambda i, a, k=k: i == k
    return lambda i, a: True


def render_name(rule: dict, i: int, stack: str, mech: str, logical: str,
                expert: int = 0) -> str:
    """按后端 lowering 规则生成物理张量名。张量契约保持纯逻辑。"""
    scope = rule.get("scope", {}).get(mech, "")
    physical = rule.get("rename", {}).get(logical, logical)
    s = rule.get("name", "")
    for k, v in (("i", i), ("local_i", i), ("global_i", i), ("stack", stack),
                 ("mech", mech), ("logical", logical), ("expert", expert),
                 ("scope", scope), ("physical", physical)):
        s = s.replace("{" + k + "}", str(v))
    return s


def norm_expr(s: str) -> str:
    s = s.replace("\u00d7", "*")          # × -> *
    s = re.sub(r"\s+x\s+", " * ", s)      # 20 x GQA -> 20 * GQA
    return s


def tokenize_expr(s: str) -> List[Tuple[str, object]]:
    s = norm_expr(s)
    toks, i = [], 0
    while i < len(s):
        m = TOKEN_RE.match(s, i)
        if not m or m.end() == i:
            break
        i = m.end()
        num, ident, op, other = m.groups()
        if num is not None:
            toks.append(("num", to_num(num)))
        elif ident is not None:
            toks.append(("id", ident))
        elif op is not None:
            toks.append(("op", op))
        elif other and other.strip():
            toks.append(("other", other))
    return toks


class _Ev:
    """极简算术求值器。遇到任何非数字符号一律返回 None（=无法校验）。"""

    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("eof", "")

    def take(self):
        v = self.peek()
        self.i += 1
        return v

    def run(self):
        v = self.add()
        return v

    def add(self):
        v = self.mul()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            r = self.mul()
            v = None if (v is None or r is None) else (v + r if op == "+" else v - r)
        return v

    def mul(self):
        v = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            r = self.unary()
            if v is None or r is None:
                v = None
            elif op == "*":
                v = v * r
            else:
                v = (v / r) if r else None
        return v

    def unary(self):
        if self.peek()[1] == "-":
            self.take()
            v = self.unary()
            return None if v is None else -v
        return self.atom()

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return val
        if kind == "op" and val == "(":
            v = self.add()
            if self.peek()[1] == ")":
                self.take()
            return v
        return None


def eval_num(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    return _Ev(tokenize_expr(s)).run()


def split_top(s: str, sep: str = ",") -> List[str]:
    """按 sep 切分，忽略括号/方括号/花括号内部。"""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def find_top(s: str, ch: str) -> int:
    """返回 ch 在 depth==0 处第一次出现的下标。"""
    depth = 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ch and depth == 0:
            return i
    return -1


def find_top_op(s: str, op: str) -> int:
    """返回多字符运算符 op 在 depth==0 处第一次出现的下标。"""
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and s.startswith(op, i):
            return i
        i += 1
    return -1


# ════════════════════════════════════════════════════════════════════
# 预处理: 去注释 -> 续行 -> 展开内联块
# ════════════════════════════════════════════════════════════════════

def strip_comment(line: str) -> str:
    i = line.find("#")
    return line[:i] if i >= 0 else line


def _cont_depth(s: str) -> int:
    """只统计圆括号与方括号 —— 花括号是结构，不参与续行判定。"""
    d = 0
    for c in s:
        if c in "([":
            d += 1
        elif c in ")]":
            d -= 1
    return d


def logical_lines(text: str) -> List[Tuple[str, int]]:
    out, buf, depth, start = [], "", 0, 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw).rstrip()
        if not line.strip() and depth == 0:
            continue
        if not buf:
            start = n
        buf = (buf + " " + line.strip()).strip() if buf else line.strip()
        depth = _cont_depth(buf)
        if depth <= 0:
            out.append((buf, start))
            buf, depth = "", 0
    if buf:
        out.append((buf, start))
    return out


def expand_inline(line: str) -> List[str]:
    """把 `head MTP { n = 1 }` 这类单行块拆成多行。赋值右侧的 map 不拆。"""
    out = []
    for _ in range(64):
        m = re.search(r"\{", line)
        if not m:
            break
        head = line[: m.start()]
        if "=" in head:                    # heads = { v = 48, ... } -> 保留整行
            out.append(line.strip())
            return out
        close = line.rfind("}")
        if close < m.start():
            out.append(line.strip())
            return out
        h = head.strip()
        body = line[m.start() + 1 : close]
        line = line[close + 1 :].strip()
        if h:
            out.append(h + " {")
        out.extend(split_top(body))
        out.append("}")
    if line.strip():
        out.append(line.strip())
    return out


# ════════════════════════════════════════════════════════════════════
# 结构
# ════════════════════════════════════════════════════════════════════

@dataclass
class Blk:
    head: str
    line: int
    kind: str = "block"
    name: str = ""
    alias: str = ""
    mtype: str = ""
    decl_len: Optional[int] = None
    assigns: Dict[str, str] = field(default_factory=dict)
    entries: List[Tuple[str, str, int]] = field(default_factory=list)
    attaches: List[Tuple[str, str, str, int]] = field(default_factory=list)
    edges: List[Tuple[str, int]] = field(default_factory=list)
    stmts: List[Tuple[str, int]] = field(default_factory=list)
    children: List["Blk"] = field(default_factory=list)
    parent: Optional["Blk"] = None

    def kids(self, kind: str) -> List["Blk"]:
        return [c for c in self.children if c.kind == kind]

    def first(self, kind: str) -> Optional["Blk"]:
        for c in self.children:
            if c.kind == kind:
                return c
        return None


def classify(head: str) -> Tuple[str, str, str, str, Optional[int]]:
    """-> (kind, name, alias, mtype, decl_len)"""
    h = head.strip()
    m = re.match(r"^mech\s+([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_]\w*)", h)
    if m:
        return "mech", m.group(1), "", m.group(2), None
    m = re.match(r"^memory\s+([A-Za-z_][\w.]*)", h)
    if m:
        return "memory", m.group(1), "", "", None
    m = re.match(r"^residual\s+([A-Za-z_][\w.]*)", h)
    if m:
        return "residual", m.group(1), "", "", None
    m = re.match(
        r"^stack\s+([A-Za-z_]\w*)(?:\s+([A-Za-z_]\w*))?\s*(?:\[\s*(\d+)\s*\])?", h
    )
    if m:
        return (
            "stack",
            m.group(1),
            m.group(2) or "",
            "",
            int(m.group(3)) if m.group(3) else None,
        )
    m = re.match(r"^head\s+([A-Za-z_][\w.]*)", h)
    if m:
        return "head", m.group(1), "", "", None
    m = re.match(r"^optimizer\s+([A-Za-z_][\w.]*)", h)
    if m:
        return "optimizer", m.group(1), "", "", None
    m = re.match(r"^(model)\s+([A-Za-z_][\w.\-]*)", h)
    if m:
        return "model", m.group(2), "", "", None
    for kw in ("hparams", "state", "position", "tensors", "emit", "schedule",
               "active", "interop"):
        if h == kw:
            return kw, "", "", "", None
    return "block", "", "", "", None


def parse(text: str) -> Tuple[Blk, List[str]]:
    root = Blk(head="<root>", line=0, kind="root")
    cur = root
    errors: List[str] = []
    for line, lineno in logical_lines(text):
        for piece in expand_inline(line):
            p = piece.strip()
            if not p:
                continue
            if p == "}":
                if cur.parent is not None:
                    cur = cur.parent
                else:
                    errors.append(f"{lineno}: 多余的 '}}'")
                continue
            if p.endswith("{"):
                h = p[:-1].strip()
                kind, name, alias, mtype, dlen = classify(h)
                b = Blk(head=h, line=lineno, kind=kind, name=name,
                        alias=alias, mtype=mtype, decl_len=dlen, parent=cur)
                cur.children.append(b)
                cur = b
                continue
            for piece2 in split_top(p, ";"):
                for stmt in split_multi_assign(piece2):
                    parse_stmt(cur, stmt, lineno)
    return root, errors


def split_multi_assign(s: str) -> List[str]:
    """一行里用逗号分隔的多个赋值：a = 1, b = 2。
    只在每一段本身都是赋值时才拆，避免误伤 pattern 这类表达式。"""
    if find_top(s, "=") < 0:
        return [s]
    parts = split_top(s)
    if len(parts) <= 1:
        # 单条赋值：末尾的逗号是分隔符残留，不能进值
        return [s.strip().rstrip(",").strip()]
    if all(find_top(p, "=") > 0 for p in parts):
        return parts
    return [s]


def parse_stmt(blk: Blk, s: str, lineno: int) -> None:
    eq = find_top(s, "=")
    colon = find_top(s, ":")
    if eq > 0 and (colon < 0 or eq < colon):
        key = s[:eq].strip()
        val = s[eq + 1 :].strip()
        if key:
            blk.assigns[key] = val
            return
    for op in (">>", "::"):
        i = find_top_op(s, op)
        if i > 0:
            blk.attaches.append((s[:i].strip(), op, s[i + len(op) :].strip(), lineno))
            return
    if colon > 0:
        key = s[:colon].strip()
        val = s[colon + 1 :].strip()
        blk.entries.append((key, val, lineno))
        return
    if "->" in s:
        blk.edges.append((s, lineno))
        return
    # `ctx 262144` / `arch qwen4_exp` 这种无等号的声明
    parts = s.split(None, 1)
    if len(parts) == 2 and re.match(r"^[A-Za-z_]\w*$", parts[0]):
        blk.assigns[parts[0]] = parts[1].strip()
        return
    blk.stmts.append((s, lineno))


# ════════════════════════════════════════════════════════════════════
# pattern 展开
# ════════════════════════════════════════════════════════════════════

@dataclass
class Rec:
    name: str
    attrs: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        if not self.attrs:
            return self.name
        return "%s(%s)" % (self.name, ", ".join(f"{k}={v}" for k, v in self.attrs.items()))


PAT_RE = re.compile(
    r"\s*(?:(\d[\d_]*)|([A-Za-z_][\w.]*)|([\[\]()+*,])|(.))"
)


def _pat_tokens(s: str):
    s = norm_expr(s)
    toks, i = [], 0
    while i < len(s):
        m = PAT_RE.match(s, i)
        if not m or m.end() == i:
            break
        i = m.end()
        num, ident, op, other = m.groups()
        if num is not None:
            toks.append(("num", int(num.replace("_", ""))))
        elif ident is not None:
            toks.append(("id", ident))
        elif op is not None:
            toks.append(("op", op))
        elif other and other.strip():
            toks.append(("other", other))
    return toks


def _parse_attrs(s: str) -> Dict[str, str]:
    out = {}
    for part in split_top(s):
        i = part.find("=")
        if i > 0:
            out[part[:i].strip()] = part[i + 1 :].strip()
        elif part:
            out[part.strip()] = ""
    return out


class _Pat:
    def __init__(self, toks, named):
        self.t, self.i, self.named = toks, 0, named

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("eof", "")

    def take(self):
        v = self.peek()
        self.i += 1
        return v

    def parse(self) -> List[Rec]:
        out = self.term()
        while self.peek()[1] == "+":
            self.take()
            out = out + self.term()
        return out

    def term(self) -> List[Rec]:
        if self.peek()[0] == "num":
            save = self.i
            n = self.take()[1]
            if self.peek()[1] == "*":
                self.take()
                return self.factor() * n
            self.i = save
        return self.factor()

    def factor(self) -> List[Rec]:
        kind, val = self.peek()
        if val == "[":
            self.take()
            inner = self.parse()
            if self.peek()[1] == "]":
                self.take()
            return inner
        if kind == "num":
            self.take()
            if self.peek()[1] == "*":
                self.take()
                return self.factor() * val
            return []
        if kind == "id":
            self.take()
            attrs: Dict[str, str] = {}
            if self.peek()[1] == "(":
                self.take()
                depth, buf = 1, []
                while True:
                    k2, v2 = self.take()
                    if k2 == "eof":
                        break
                    if v2 == "(":
                        depth += 1
                    elif v2 == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    buf.append(str(v2))
                attrs = _parse_attrs("".join(buf))
            if val in self.named:
                base = list(self.named[val])
                if attrs:
                    base = [Rec(r.name, {**r.attrs, **attrs}) for r in base]
                return base
            return [Rec(val, attrs)]
        self.take()
        return []


def expand_pattern(expr: str, named: Dict[str, List["Rec"]]) -> Optional[List["Rec"]]:
    try:
        toks = _pat_tokens(expr)
        if not toks:
            return None
        p = _Pat(toks, named)
        out = p.parse()
        # pattern 必须把整串吃完。剩下的记号意味着有东西没被理解 ——
        # 这种情况必须报错，绝不能静默丢掉（比如 `2 * Attn ; main[:] >> X`）。
        if p.i < len(toks):
            return None
        return out
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# 检查
# ════════════════════════════════════════════════════════════════════

E, W, I = "E", "W", "i"
TOKEN_MIXER = {"Attention", "Sparse", "Linear", "SSM", "Vision", "Recurrent",
               "MLA",
               # KDA（Kimi Delta Attention）：和 GDN 同族但**不是同一个东西** ——
               # 三个独立卷积、f_proj、o_norm。当成 Linear 会生成一个 GDN，
               # 那是个"看起来对但算错"的模型，所以给它自己的种类。
               "KDA"}
BUILTIN_ATTRS = {
    "mode", "window", "sink", "kv_tie", "head_dim", "kv", "kv_heads",
    "q", "q_heads", "v", "v_heads", "qk", "gate", "act", "activation",
    "dtype", "rope", "experts", "top_k", "shared", "hidden", "d_ff",
    # 下面这些是检查器自己会算/会推荐的，写出来不应该被当成拼写错误
    "out_dim", "qk_norm", "kv_space", "rope_base", "pairing", "partial",
    "head_gate", "score_bias", "shared_hidden", "routed_scale",
}


class Report:
    def __init__(self, path):
        self.path = path
        self.items: List[Tuple[str, int, str, str]] = []   # sev, line, code, msg

    def add(self, sev, line, code, msg):
        item = (sev, line, code, msg)
        if item not in self.items:
            self.items.append(item)

    def count(self, sev):
        return sum(1 for it in self.items if it[0] == sev)


def check(path: str) -> Tuple[Report, dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    root, perr = parse(text)
    rep = Report(path)
    for e in perr:
        rep.add(E, 0, "parse", e)

    model = root.first("model")
    scope = model if model else root

    mechs: Dict[str, Blk] = {}
    memories: Dict[str, Blk] = {}
    residuals: Dict[str, Blk] = {}
    for c in scope.children:
        if c.kind == "mech":
            mechs[c.name] = c
        elif c.kind == "memory":
            memories[c.name] = c
        elif c.kind == "residual":
            residuals[c.name] = c

    # ---- hparams -------------------------------------------------
    hp: Dict[str, str] = {}
    nlayer_decl = None
    hb = scope.first("hparams")
    if hb:
        hp = dict(hb.assigns)
        nlayer_decl = eval_num(hp.get("n_layer"))
    d_model = eval_num(hp.get("d_model"))
    vocab = eval_num(hp.get("vocab"))

    # ---- schedule 命名子序列 --------------------------------------
    named: Dict[str, List[Rec]] = {}
    sb = scope.first("schedule")
    if sb:
        # 多趟展开：命名子序列可以引用别的命名子序列，直到不再变化
        for _ in range(8):
            before = {k: [str(r) for r in v] for k, v in named.items()}
            for k, v in sb.assigns.items():
                ex = expand_pattern(v, named)
                if ex is not None:
                    named[k] = ex
            if before == {k: [str(r) for r in v] for k, v in named.items()}:
                break

    # ---- stacks --------------------------------------------------
    stacks = scope.kids("stack")
    alias = {}
    expansion: Dict[str, List[Rec]] = {}
    # (栈, 层号, 机制) -> {属性: 值}
    overrides: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    pending_ov: List[tuple] = []
    total_layers = 0
    for st in stacks:
        if st.alias:
            alias[st.alias] = st.name
        alias[st.name] = st.name

    for st in stacks:
        pexpr = st.assigns.get("pattern")
        if pexpr is None:
            rep.add(W, st.line, "pattern", f"stack {st.name} 没有 pattern")
            continue
        ex = expand_pattern(pexpr, named)
        if ex is None:
            rep.add(E, st.line, "pattern", f"stack {st.name} 的 pattern 无法解析")
            continue
        expansion[st.name] = ex
        total_layers += len(ex)

        # 逐层属性覆盖：`main[3..44] : MoE.experts = [255, 266, ...]`
        # 值按被选中的层顺序分配。逐层变化的专家数、swiglu_limit 都靠它。
        for _k, _v, _ln in st.entries:
            _m = re.match(r"^([A-Za-z_]\w*)\s*\[\s*(.*?)\s*\]$", _k)
            if not _m:
                rep.add(W, _ln, "override",
                        f"stack {st.name} 里的条目 '{_k}' 不是逐层覆盖"
                        f"（要写成像 main[3..44] : MoE.experts = [...]）")
                continue
            _sn, _sel = _m.group(1), _m.group(2)
            if alias.get(_sn, _sn) != st.name:
                rep.add(E, _ln, "override",
                        f"逐层覆盖 '{_k}' 指的是栈 '{_sn}'，但它写在 stack {st.name} 里")
                continue
            _mq = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*(.*)$", _v)
            if not _mq:
                rep.add(E, _ln, "override",
                        f"逐层覆盖 '{_k}' 的值要写成 <机制>.<属性> = [...]")
                continue
            _mech, _attr, _vals = _mq.group(1), _mq.group(2), _mq.group(3).strip()
            if not (_vals.startswith("[") and _vals.endswith("]")):
                rep.add(E, _ln, "override",
                        f"逐层覆盖 '{_k}' 的值必须是列表")
                continue
            if _mech not in mechs:
                rep.add(E, _ln, "override", f"逐层覆盖引用了未声明的机制 '{_mech}'")
                continue
            _items = [x.strip() for x in split_top(_vals[1:-1]) if x.strip()]
            # resolve_attrs 还没定义，先把原始的攒起来，等它可用再分配
            pending_ov.append((st.name, _sel, _mech, _attr, _items, _ln, _k))
        if st.decl_len is not None and st.decl_len != len(ex):
            rep.add(E, st.line, "layers",
                    f"stack {st.name} 声明 {st.decl_len} 层，pattern 展开出 {len(ex)} 层")
        elif st.decl_len is not None:
            rep.add(I, st.line, "layers",
                    f"stack {st.name} 的层数在两处声明（[{st.decl_len}] 与 pattern），"
                    f"当前一致 —— 建议 [N] 只作断言")
        for r in ex:
            if r.name not in mechs:
                if r.name in memories:
                    rep.add(W, st.line, "resolve",
                            f"{r.name} 声明在 memory 块，却被 pattern 当作层机制引用")
                elif r.name in residuals:
                    rep.add(E, st.line, "resolve",
                            f"{r.name} 是 residual，不能出现在 pattern 里")
                else:
                    rep.add(E, st.line, "resolve",
                            f"pattern 引用了未声明的机制 '{r.name}'")
            else:
                m = mechs[r.name]
                for a in r.attrs:
                    if a not in m.assigns and a not in BUILTIN_ATTRS:
                        rep.add(W, st.line, "attr",
                                f"{r.name}({a} = ...) 的属性 '{a}' 既未在 mech "
                                f"{r.name} 声明，也不在已知属性表里")

    if nlayer_decl is not None and stacks:
        if nlayer_decl != total_layers:
            rep.add(E, hb.line, "layers",
                    f"hparams.n_layer = {int(nlayer_decl)}，但各 stack 合计 {total_layers} 层")

    # ---- 双真相源: ctx vs yarn ------------------------------------
    ctx = eval_num(scope.assigns.get("ctx"))
    pb = scope.first("position")
    yarn = None
    if pb:
        for k, v in pb.assigns.items():
            if "yarn" in v:
                m = re.search(r"yarn\s*=\s*([\d.]+)", v)
                if m:
                    yarn = to_num(m.group(1))
    if ctx and yarn:
        rep.add(W, scope.line, "twotruth",
                f"ctx = {int(ctx)} 与 position.yarn = {yarn} 并存："
                f"哪个是原生上下文、哪个是外推上限没有区分（外推后 = {int(ctx*yarn)}）")

    # ---- attach 目标 ---------------------------------------------
    attached = set()
    for c in scope.children:
        for sel, op, target, ln in c.attaches:
            base = target.split("(")[0].strip()
            attached.add(base)
            if base not in mechs and base not in residuals:
                rep.add(E, ln, "resolve", f"'{sel} {op} {target}' 的目标 '{base}' 未声明")

    # ---- state 块 ------------------------------------------------
    state_src: Dict[str, List[Tuple[str, str, int]]] = {}
    stb = scope.first("state")
    if stb:
        for key, val, ln in stb.entries:
            hits = [n for n in list(mechs) + list(memories)
                    if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", key)]
            if not hits:
                rep.add(E, ln, "resolve",
                        f"state 条目 '{key}' 解析不到任何已声明的机制"
                        f"（通配符条目也必须能指名机制）")
                continue
            nm = hits[0]
            if nm in memories:
                rep.add(W, ln, "state-vs-param",
                        f"state 里的 '{nm}' 声明在 memory 块。查表是"
                        f"跨序列共享的固定权重，不是 per-seq 状态")
            state_src.setdefault(nm, []).append((key, val, ln))

    # ---- position 块键 -------------------------------------------
    MODAL = {"default", "text", "vision", "audio", "video"}
    if pb:
        for k in pb.assigns:
            if k in MODAL:
                continue
            if k not in mechs and k not in named:
                rep.add(W, pb.line, "resolve",
                        f"position 的键 '{k}' 不是机制名、模态名，也不是命名子序列")

    # ---- optimizer except(...) -----------------------------------
    for c in scope.children:
        if c.kind != "optimizer":
            continue
        for key, val, ln in c.entries:
            m = re.search(r"except\s*\(([^)]*)\)", key)
            if not m:
                continue
            known = set(mechs) | set(memories) | set(residuals)
            known |= {ch.name for ch in scope.kids("head")}
            for sym in split_top(m.group(1)):
                if sym not in known:
                    rep.add(E, ln, "resolve",
                            f"optimizer except(...) 引用了未声明的符号 '{sym}'")

    # ---- interop 栈名 --------------------------------------------
    ib = scope.first("interop")
    if ib:
        stack_names = set(alias)
        for s, ln in ib.edges:
            segs = [x.strip() for x in s.split("->") if x.strip()]
            for seg in (segs[:1] + segs[-1:]):
                m = re.match(r"([A-Za-z_]\w*)", seg)
                if m and m.group(1) not in stack_names:
                    rep.add(E, ln, "interop",
                            f"interop 的 '{seg}' 里 '{m.group(1)}' 不是已声明的 stack")

    # ---- 机制形状 ------------------------------------------------
    def heads_of(blk: Blk) -> Optional[Dict[str, float]]:
        raw = blk.assigns.get("heads")
        if not raw:
            return None
        body = raw.strip()
        if body.startswith("{"):
            body = body[1 : body.rfind("}")] if "}" in body else body[1:]
        out = {}
        for part in split_top(body):
            i = part.find("=")
            if i > 0:
                v = eval_num(part[i + 1 :])
                if v is not None:
                    out[part[:i].strip()] = v
        return out or None

    head_dim_of: Dict[str, float] = {}
    for nm, blk in mechs.items():
        h = heads_of(blk)
        hd = None
        if h and "head_dim" in h:
            hd = h["head_dim"]
        else:
            hd = eval_num(blk.assigns.get("head_dim"))
        if hd:
            head_dim_of[nm] = hd

        if h is None:
            if blk.mtype in TOKEN_MIXER and blk.mtype != "Vision":
                rep.add(W, blk.line, "shape",
                        f"mech {nm} ({blk.mtype}) 未声明 heads/head_dim —— "
                        f"依赖它的 KV cache 与状态尺寸都无法推导")
            continue
        if "q" not in h and "v" not in h:
            rep.add(W, blk.line, "shape",
                    f"mech {nm} 的 heads 缺 q/v —— 宽度与投影形状都推不出来")
            continue

    # ---- mrope 不变式 --------------------------------------------
    if pb:
        for k, v in pb.assigns.items():
            m = re.search(r"mrope\s*=\s*\[([^\]]*)\]", v)
            if not m:
                continue
            secs = [x for x in (to_num(p) for p in m.group(1).split(",")) if x]
            ssum = sum(secs)
            ref = head_dim_of.get("QSA") or head_dim_of.get("GQA")
            if ref is None and head_dim_of:
                ref = max(head_dim_of.values())
            if ref is None:
                rep.add(W, pb.line, "position",
                        f"mrope 段和 = {int(ssum)}，但没有机制声明 head_dim，"
                        f"无法校验（隐含 head_dim = {int(ssum*2)}）")
            else:
                want = ref / 2
                if abs(want - ssum) > 1e-9:
                    rep.add(E, pb.line, "position",
                            f"mrope 段和 = {int(ssum)}，但按 head_dim = {int(ref)} "
                            f"应为 {int(want)}（隐含 head_dim = {int(ssum*2)}）")

    # ---- memory 表 vs params -------------------------------------
    decl_params: Dict[str, float] = {}
    for nm, blk in memories.items():
        p = eval_num(blk.assigns.get("params"))
        if p:
            decl_params[nm] = p
        ents = leading_num(blk.assigns.get("table"))
        if p and ents:
            per = p / ents
            row = eval_num(blk.assigns.get("row_dim"))
            if row is None and (per > 1024 or abs(per - round(per)) > 1e-9):
                rep.add(W, blk.line, "params",
                        f"memory {nm}: params/table = {per:,.1f} 参数/条目 —— "
                        f"查表通常是 row_dim（128~512）/条目。"
                        f"缺 row_dim 或 heads 声明，无法确认单位")

    # ---- 张量契约实例化 ------------------------------------------
    # 契约按「机制」声明；实例按「(机制, 结构属性) 等价类」求值。
    # 这正是结构属性的连带代价：head_dim / kv_tie 一变，形状与张量集合都变。
    STRUCTURAL = {"head_dim", "kv", "q", "v", "qk", "kv_heads", "kv_tie",
                  "experts", "d_ff", "hidden", "intermediate", "out_dim"}
    decl_struct: Dict[str, set] = {}
    for nm, blk in mechs.items():
        raw = blk.assigns.get("structural")
        if raw:
            body = raw.strip()
            if body.startswith("[") and "]" in body:
                body = body[1 : body.rfind("]")]
            decl_struct[nm] = set(split_top(body))

    # 1) 实例枚举 -> 等价类
    def resolve_attrs(m: Blk, over: Dict[str, str]) -> Dict[str, str]:
        a: Dict[str, str] = {}
        for k, v in m.assigns.items():
            if k not in ("heads", "structural"):
                a[k] = v
        for k, v in (heads_of(m) or {}).items():
            a[k] = _fmt(v)
        a.update(over)
        return a

    def key_of(nm: str, attrs: Dict[str, str]) -> tuple:
        sk = STRUCTURAL | decl_struct.get(nm, set())
        return tuple(sorted((k, v) for k, v in attrs.items() if k in sk))

    # 哪些层挂了哪些「通道混合器」—— 选择器（[:] / [0] / [1..39] / [step 4]）生效
    attach_rules: List[tuple] = []
    for c in scope.children:
        for sel, op, target, ln in c.attaches:
            base = target.split("(")[0].strip()
            if base not in mechs or op != ">>":
                continue
            sk, inner = split_selector(sel)
            attach_rules.append((alias.get(sk, sk), make_layer_pred(inner), base))

    def attached_at(stack_name: str, li: int, attrs: Dict[str, str]) -> List[str]:
        return [b for (sn, pred, b) in attach_rules
                if sn == stack_name and pred(li, attrs)]

    # 逐层覆盖的分配 —— 必须等 resolve_attrs 可用（它选择哪些层要按属性判断）
    for _sn, _sel, _mech, _attr, _items, _ln, _k in pending_ov:
        _ex = expansion.get(_sn, [])
        _pred = make_layer_pred(_sel)
        _hit = [i for i, r in enumerate(_ex)
                if _pred(i, resolve_attrs(mechs[r.name], r.attrs)
                         if r.name in mechs else {})]
        if len(_hit) != len(_items):
            rep.add(E, _ln, "override",
                    f"逐层覆盖 '{_k}' 选中 {len(_hit)} 层，但给了 {len(_items)} 个值")
            continue
        for _i, _val in zip(_hit, _items):
            overrides.setdefault((_sn, _i, _mech), {})[_attr] = _val

    classes: Dict[str, Dict[tuple, dict]] = {}
    layer_seq: List[tuple] = []
    for st in stacks:
        for li, r in enumerate(expansion.get(st.name, [])):
            m = mechs.get(r.name)
            if m is None:
                continue
            attrs = resolve_attrs(m, r.attrs)
            _ov = overrides.get((st.name, li, r.name))
            if _ov:
                attrs.update(_ov)
            key = key_of(r.name, attrs)
            layer_seq.append((st.name, r.name, dict(attrs), key,
                              attached_at(st.name, li, attrs)))
            d = classes.setdefault(r.name, {})
            ent = d.setdefault(key, {"count": 0, "attrs": {}})
            ent["count"] += 1
            ent["attrs"].update(attrs)

    # 只被挂载、不作为层机制出现的（如 FFN）：按每个挂载规则建类，层数按选择器数
    for (sn, pred, base) in attach_rules:
        if base in classes or base not in mechs:
            continue
        buckets = {}
        for li, r in enumerate(expansion.get(sn, [])):
            a = resolve_attrs(mechs[r.name], r.attrs) if r.name in mechs else {}
            if not pred(li, a):
                continue
            at = dict(resolve_attrs(mechs[base], {}))
            at.update(overrides.get((sn, li, base), {}))
            kk = key_of(base, at)
            buckets.setdefault(kk, {"count": 0, "attrs": at})
            buckets[kk]["count"] += 1
        classes.setdefault(base, {}).update(buckets)

    # 2) 读契约
    tb = scope.first("tensors")
    # 文件里既没有 tensors 也没有 emit，说明它是在**设计一个新模型**，
    # 不是在对拍一个已有产物。张量契约那类警告对它是噪音。
    wants_artifacts = (tb is not None) or (scope.first("emit") is not None)
    contracts: Dict[str, list] = {}
    if tb:
        for ch in tb.children:
            h = ch.head.strip()
            if re.match(r"^layer\s*\(", h):
                rep.add(E, ch.line, "tensors",
                        f"'{h}' 是无作用域写法：会给所有层发同一套张量名。"
                        f"契约必须按机制声明（见 pattern-design §2 规则 3）")
                continue
            base = h.split(".")[0].strip()
            if h in ("layer", "global"):
                pass                    # 层级 / 全局：不属于任何机制
            elif base not in mechs and base not in memories and base not in residuals:
                rep.add(E, ch.line, "tensors", f"tensors 的作用域 '{base}' 未声明")
                continue
            lst = []
            for key, val, ln in ch.entries:
                mg = re.search(r"\bunless\s+([A-Za-z_]\w*)", val)
                guard = mg.group(1) if mg else None
                pe = bool(re.search(r"\bper_expert\b", val))
                shp = re.sub(r"\bunless\s+[A-Za-z_]\w*", "", val)
                shp = re.sub(r"\bper_expert\b", "", shp).strip()
                lst.append((key, shp, guard, ln, pe))
            if not lst:
                rep.add(W, ch.line, "tensors",
                        f"tensors 的 '{h}' 契约是空的 —— 没有声明任何逻辑张量")
            contracts[base] = lst

    def _flat_rows(entry_list):
        """层级 / 全局张量：形状只用 hparams 求值（没有逐层属性）。"""
        sym0: Dict[str, float] = {}
        for k2, v2 in hp.items():
            n2 = eval_num(v2)
            if n2 is not None:
                sym0[k2] = n2
        res = []
        for (lname, shp, guard, ln, pe) in entry_list:
            body = shp.strip()
            if body.startswith("(") and ")" in body:
                body = body[1: body.rfind(")")]
            comps = []
            for c in split_top(body):
                ids = set(re.findall(r"[A-Za-z_]\w*", c))
                s2 = c
                for i in sorted(ids, key=len, reverse=True):
                    if i in sym0:
                        s2 = re.sub(r"\b" + re.escape(i) + r"\b", _fmt(sym0[i]), s2)
                v = eval_num(s2)
                comps.append(_fmt(v) if v is not None else s2.replace(" ", ""))
            # `--` 是**声明为不该存在**（权重共享时没有 lm_head），
            # 它不是形状，不能被重新包成 `(...)` —— 原样透传，
            # 让 by1verify 去检查"确实不存在"。
            if shp.strip() == "--":
                res.append((lname, "--", guard, pe))
                continue
            # guard 留着 —— layer 作用域用它做「只有挂了某机制才有这个张量」
            res.append((lname, "(" + ", ".join(comps) + ")", guard, pe))
        return res

    layer_rows = _flat_rows(contracts.get("layer", []))
    global_rows = _flat_rows(contracts.get("global", []))

    # 3) 逐类实例化
    trows = []
    class_rows: Dict[tuple, list] = {}
    for nm in sorted(classes):
        cls = classes[nm]
        if nm not in contracts:
            if wants_artifacts and mechs[nm].mtype in TOKEN_MIXER:
                rep.add(W, mechs[nm].line, "tensors", f"mech {nm} 无张量契约")
            continue
        for key, ent in sorted(cls.items(), key=lambda kv: -kv[1]["count"]):
            attrs = ent["attrs"]
            label = ", ".join(f"{k}={v}" for k, v in key) or "(无结构属性)"
            sym: Dict[str, float] = {}
            for k, v in attrs.items():
                n = eval_num(v)
                if n is not None:
                    sym[k] = n
            for k, v in hp.items():
                n = eval_num(v)
                if n is not None:
                    sym[k] = n

            # 宽度不变式。注意：不能靠「比值是否好看」判定对错 —— Gemma 4 的
            # 注意力宽度本来就不等于 d_model。所以宽度变化必须显式声明。
            qq, vv, hdv = sym.get("q"), sym.get("v"), sym.get("head_dim")
            w = (vv * hdv) if (vv and hdv) else ((qq * hdv) if (qq and hdv) else None)
            od = eval_num(attrs.get("out_dim"))
            if w and od:
                if abs(w - od) > 1e-9:
                    rep.add(E, mechs[nm].line, "shape",
                            f"{nm}({label}): 声明 out_dim = {_fmt(od)}，"
                            f"但 q x head_dim = {_fmt(w)}")
            elif w and d_model and abs(w - d_model) > 1e-9:
                rep.add(W, mechs[nm].line, "shape",
                        f"{nm}({label}): q x head_dim = {_fmt(w)} != d_model = "
                        f"{_fmt(d_model)}（比值 {w/d_model:.3f}）—— 若这是有意的宽度变化，"
                        f"声明 out_dim = {_fmt(w)}")
            rows = []
            for lname, shp, guard, ln, pe in contracts[nm]:
                if guard is not None:
                    gs = attrs.get(guard)
                    if gs is None:
                        rep.add(W, ln, "tensors",
                                f"{nm}.{lname}: unless {guard} 引用了未声明的属性")
                        continue
                    if str(gs).strip().lower() in ("true", "1", "yes", "on"):
                        rows.append((lname, "--", f"由 unless {guard} 抑制", pe))
                        continue
                body = shp.strip()
                if body.startswith("(") and ")" in body:
                    body = body[1 : body.rfind(")")]
                out, miss = [], []
                for c in split_top(body):
                    ids = set(re.findall(r"[A-Za-z_]\w*", c))
                    bad = sorted(i for i in ids if i not in sym)
                    if bad:
                        miss.extend(bad)
                    s2 = c
                    for i in sorted(ids, key=len, reverse=True):
                        if i in sym:
                            s2 = re.sub(r"\b" + re.escape(i) + r"\b", _fmt(sym[i]), s2)
                    v = eval_num(s2)
                    out.append(_fmt(v) if v is not None else s2.replace(" ", ""))
                note = ("缺 " + ", ".join(sorted(set(miss)))) if miss else ""
                rows.append((lname, "(" + ", ".join(out) + ")", note, pe))
            label = ", ".join(f"{k}={v}" for k, v in key) or "(无结构属性)"
            class_rows[(nm, key)] = rows
            trows.append((nm, label, ent["count"], rows))

    for nm, lst in contracts.items():
        if nm in ("layer", "global"):
            continue
        if nm not in classes and nm not in attached:
            rep.add(W, lst[0][3] if lst else 0, "tensors",
                    f"tensors 声明了 {nm} 的契约，但 pattern 里没有它的实例")

    # ---- emit lowering 规则 --------------------------------------
    # 物理命名在后端里声明，张量契约保持纯逻辑 —— 这是「一份描述两个后端」
    # 能成立的前提。检查器负责校验模板与映射的引用是否合法。
    KNOWN_PH = {"i", "local_i", "global_i", "stack", "mech",
                "logical", "scope", "physical", "expert"}
    all_logical = {ln for lst in contracts.values() for (ln, _s, _g, _l, _p) in lst}
    emit_rules: Dict[str, dict] = {}
    eb = scope.first("emit")
    if eb:
        for be in eb.children:
            key = be.head.split("->")[-1].strip() if "->" in be.head else be.head.strip()
            rule = {"name": be.assigns.get("name", "").strip().strip('"'),
                    "expert_name": be.assigns.get("expert_name", "").strip().strip('"'),
                    "global_name": be.assigns.get("global_name", "").strip().strip('"'),
                    "scope": {}, "rename": {}, "fields": {},
                    "quant": {}, "line": be.line, "head": be.head.strip()}
            for sub in be.children:
                h = sub.head.strip()
                if h == "quant":
                    for k, v in sub.assigns.items():
                        rule["quant"][k] = v.strip().strip('"')
                    for sub2 in sub.children:
                        if sub2.head.strip() == "fuse":
                            rule["quant"]["fuse"] = dict(sub2.assigns)
                    continue
                tgt = {"scope": rule["scope"], "rename": rule["rename"],
                       "field": rule["fields"], "fields": rule["fields"]}.get(h)
                if tgt is None:
                    continue
                for k, v in sub.assigns.items():
                    tgt[k] = v.strip().strip('"')
            emit_rules[key] = rule

            if not rule["name"]:
                if rule["fields"]:
                    continue
                rep.add(W, rule["line"], "emit",
                        f"emit {key} 既没有 name 模板也没有 field 映射")
                continue
            for ph in re.findall(r"\{(\w+)\}",
                                 rule["name"] + rule["expert_name"] + rule["global_name"]):
                if ph not in KNOWN_PH:
                    rep.add(E, rule["line"], "emit",
                            f"emit {key} 的 name 模板含未知占位符 {{{ph}}}")
            for m in rule["scope"]:
                if m not in ("layer", "global") and m not in mechs:
                    rep.add(E, rule["line"], "emit",
                            f"emit {key} 的 scope 引用了未声明的机制 '{m}'")
            # quant 的 fuse 名（如 gate_up_proj）和它点名的张量，只在导出时才存在，
            # 不是契约里的逻辑张量 —— 不该被当成拼写错误
            _qnames = set(re.findall(
                r"[A-Za-z_][\w.]*", (rule.get("quant") or {}).get("tensor", "")))
            for _n, _v in ((rule.get("quant") or {}).get("fuse") or {}).items():
                _qnames.add(_n)
                _qnames |= set(re.findall(r"[A-Za-z_][\w.]*", _v))
            for ln in rule["rename"]:
                if ln in _qnames:
                    continue
                if all_logical and ln not in all_logical:
                    rep.add(W, rule["line"], "emit",
                            f"emit {key} 的 rename 键 '{ln}' 不是任何契约里的逻辑张量")
            used_logical = set(re.findall(r"\{logical\}", rule["name"]))
            # 真不变式：同一个等价类里，不同的逻辑张量不能生成同一个物理名
            for (mn, _mk), rows in class_rows.items():
                if rule["scope"] and mn not in rule["scope"]:
                    continue
                seen_nm: Dict[str, str] = {}
                for (ln2, _sh, _nt, _pe) in rows:
                    nm2 = render_name(rule, 0, "<stack>", mn, ln2)
                    if nm2 in seen_nm and seen_nm[nm2] != ln2:
                        rep.add(E, rule["line"], "emit",
                                f"emit {key}: {mn} 的 '{ln2}' 与 '{seen_nm[nm2]}' "
                                f"生成同一个物理名 '{nm2}'")
                    seen_nm[nm2] = ln2

    # ---- 反向：从 .by1 生成 config --------------------------------
    # 方向在这里第一次反过来：config 不是「真相」，它是**产物**。
    # 值只能来自四种来源，别的写不出来：
    #   hparams 键 / 机制属性（可带选择器）/ true|false|null|"字符串" / 两个生成器
    def _coerce(s):
        t = str(s).strip()
        if len(t) >= 2 and t.startswith('"') and t.endswith('"'):
            return t[1:-1]
        if t == "true":
            return True
        if t == "false":
            return False
        if t == "null":
            return None
        n = eval_num(t)
        if n is not None:
            return int(n) if float(n).is_integer() else n
        return t

    def _ltype_of_attrs(a):
        w = (a.get("window") or "").strip().lower()
        return "full_attention" if w in ("none", "null", "0", "") else "sliding_attention"

    def gen_layer_types():
        return [_ltype_of_attrs(a) for (_s, _m, a, _k, _t) in layer_seq]

    def name_layer_type(nm):
        recs = named.get(nm)
        if recs:
            return _ltype_of_attrs(recs[0].attrs)
        for (_s, m, a, _k, _t) in layer_seq:
            if m == nm:
                return _ltype_of_attrs(a)
        return None

    ROPE_KEYS = {"base": "rope_theta", "type": "rope_type",
                 "partial": "partial_rotary_factor",
                 "ratio": "partial_rotary_factor",
                 "original": "original_max_position_embeddings"}
    ROPE_DROP = {"pairing"}          # 配对约定是 codegen 的事，不进 config

    def _rope_params(seg, out):
        for part in split_top(seg):
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ROPE_DROP:
                    continue
                out[ROPE_KEYS.get(k, k)] = _coerce(v)

    def parse_rope(expr):
        """支持 rope(...) / partial(rope(...), ...) / yarn(rope(...), ...)"""
        e = expr.strip()
        out: Dict[str, Any] = {}
        for _ in range(4):
            m = re.match(r"^(\w+)\s*\(", e)
            if not m or m.group(1) == "rope":
                break
            wrap = m.group(1)
            inner = e[m.end(): e.rindex(")")]
            depth, cut = 0, None
            for i, ch in enumerate(inner):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        cut = i + 1
                        break
            if cut:
                _rope_params(inner[cut:], out)
            if wrap != "partial":
                out.setdefault("rope_type", wrap)
            e = (inner[:cut] if cut else inner).strip()
        m = re.match(r"^rope\s*\(", e)
        if m:
            _rope_params(e[m.end(): e.rindex(")")], out)
        out.setdefault("rope_type", "default")
        return out

    def gen_rope_parameters():
        if pb is None:
            return None
        out = {}
        for k, v in pb.assigns.items():
            if k == "default":
                continue
            lt = name_layer_type(k)
            if lt is None:
                rep.add(W, pb.line, "config",
                        f"position 的键 '{k}' 对不上任何层类型，生成 config 时跳过")
                continue
            out[lt] = parse_rope(v)
        return out or None

    def gen_rope_scaling():
        """官方 config 里那份**摊平的** rope_scaling（旧格式）：
        从 position 里带缩放的那个键上取。"""
        if pb is None:
            return None
        for k, v in pb.assigns.items():
            d = parse_rope(v)
            if d.get("factor") is None:
                continue
            # **两种缩放的 config 键完全不同**，不能共用一份模板：
            #   yarn    beta_fast / beta_slow / factor / truncate
            #   llama3  factor / high_freq_factor / low_freq_factor
            # 共同项只有 original_max_position_embeddings 和 rope_type。
            ty = str(d.get("rope_type") or "default").strip().lower()
            orig = _coerce(d.get("original_max_position_embeddings",
                                 d.get("original", 4096)))
            if ty == "llama3":
                return {"factor": _coerce(d.get("factor")),
                        "high_freq_factor": _coerce(
                            d.get("high_freq_factor", d.get("high_freq", 4))),
                        "low_freq_factor": _coerce(
                            d.get("low_freq_factor", d.get("low_freq", 1))),
                        "original_max_position_embeddings": orig,
                        "rope_type": ty}
            return {"beta_fast": _coerce(d.get("beta_fast", 32)),
                    "beta_slow": _coerce(d.get("beta_slow", 1)),
                    "factor": _coerce(d.get("factor")),
                    "original_max_position_embeddings": orig,
                    "rope_type": ty,
                    "truncate": bool(d.get("truncate", True))}
        return None

    def gen_attention_other_setting():
        """滑窗那一套的注意力参数 —— 和全量层不同（Step-3.7 是 96 vs 64 头）。"""
        for (_s, m, a, _k, _t) in layer_seq:
            win = str(a.get("window", "")).strip().lower()
            if win in ("", "none", "null", "0"):
                continue
            hd = int(float(a.get("head_dim") or 0))
            return {"attention_type": "sliding_attention",
                    "head_dim": hd,
                    "num_attention_groups": int(float(a.get("kv") or 0)),
                    "num_attention_heads": int(float(a.get("q") or 0)),
                    "true_head_dim": hd}
        return None

    def gen_by_layer(key):
        """逐层字典：{层号字符串: 值}。官方 config 的 moe_num_experts_by_layer 就是它。
        **只有这个键存在的层才进字典** —— 稠密层不该出现。"""
        out = {}
        for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
            if key.startswith("attach."):
                sub, v = key[7:], ""
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        # **先看这一层有没有逐层覆盖** —— 只看声明值的话
                        # Step-3.7 的 42 层会全报 288。
                        v = overrides.get((_s, li, am), {}).get(
                            sub, ma.assigns.get(sub, ""))
                        break
                if v in ("", None):
                    continue
            else:
                v = a.get(key, "")
                if v in ("", None):
                    continue
            out[str(li)] = _coerce(v)
        return out or None

    def gen_per_layer_rope(key):
        """逐层的 rope 参数。position 是按**层类型**声明的，
        要先经 layer_types 映射到每一层 —— 所以不能直接用 per_layer()。"""
        lts = gen_layer_types() or []
        rbt = gen_rope_parameters() or {}
        return [_coerce((rbt.get(lt) or {}).get(key, "")) for lt in lts]

    def gen_join(inner, sep):
        vals = resolve_field(inner)
        if not isinstance(vals, list):
            return vals
        return sep.join(str(x) for x in vals)

    def gen_pad(inner, n, mode="zero"):
        vals = resolve_field(inner)
        if not isinstance(vals, list):
            return vals
        # 官方有些逐层数组比层数长（Step-3.7 是 48、层数是 45）。
        # **补什么，两种都有，而且长得很像**：
        #   layer_types / partial_rotary_factors / rope_theta  重复最后一个
        #   swiglu_limits / swiglu_limits_shared               补 0
        # 猜错的话数组长度对、值也对，只有尾巴不同。
        if len(vals) >= n:
            return vals[:n]
        if mode == "cycle" and vals:
            # **按周期续**：官方 Step-3.7 的 layer_types 是 48 长、模型是 45 层，
            # 45/46/47 接着 4 周期的第 1/2/3 项（都是滑窗）。
            # 既不是"重复最后一个"（那会给出全量层），也不是补零。
            per = None
            for cand in range(1, len(vals) // 2 + 1):
                if all(vals[i] == vals[i % cand] for i in range(len(vals))):
                    per = cand
                    break
            if per:
                return vals + [vals[i % per] for i in range(len(vals), n)]
            return vals + [vals[-1]] * (n - len(vals))
        fill = vals[-1] if (mode == "last" and vals) else 0
        return vals + [fill] * (n - len(vals))

    def lookup_attr(mech, sel, attr):
        _n, conds = parse_state_key(f"{mech}[{sel}]" if sel else mech)
        for (_s, m, a, _k, _t) in layer_seq:
            if m == mech and selector_matches(conds, a):
                return _coerce(a.get(attr, ""))
        if mech in mechs:
            return _coerce(mechs[mech].assigns.get(attr, ""))
        for c in scope.children:            # head / memory / residual 这类块
            if c.name == mech:
                return _coerce(c.assigns.get(attr, ""))
        return f"<找不到 {mech}[{sel}].{attr}>"

    def gen_per_layer(key):
        """per_layer(q) / per_layer(attach.layer_kind) -> 逐层一个值。
        官方 config 里那一堆逐层数组（头数、MLP 类型、gating 类型）都是这个。"""
        out = []
        for (_s, _m, a, _k, atts) in layer_seq:
            if key.startswith("attach."):
                sub = key[7:]
                v = ""
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        v = ma.assigns.get(sub, ma.mtype)
                        break
                out.append(_coerce(v))
            else:
                out.append(_coerce(a.get(key, "")))
        return out

    def gen_per_layer_d(key, default):
        """per_layer(attach.swiglu_limit, 0) —— 属性**缺失**时给默认值。

        不能复用 gen_per_layer：它在属性缺失时会退回**机制类型名**
        （'MoE'/'FFN'），于是那个回退值看起来"非空"，默认值永远用不上，
        逐层数组里就混进了字符串。这里查的是属性本身有没有。
        """
        out = []
        for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
            if key.startswith("attach."):
                sub, v = key[7:], None
                for am in atts:
                    ma = mechs.get(am)
                    if ma is not None:
                        # 逐层覆盖优先于声明值
                        v = overrides.get((_s, li, am), {}).get(
                            sub, ma.assigns.get(sub))
                        break
            else:
                v = a.get(key)
            out.append(_coerce(v) if v not in (None, "") else default)
        return out

    def gen_sliding_window():
        """滑窗宽度 —— 从**开了窗的那些层**取，不是第 0 层（第 0 层可能是全量）。"""
        for (_s, _m, a, _k, _t) in layer_seq:
            w = str(a.get("window", "")).strip().lower()
            if w not in ("", "none", "null", "0"):
                return int(float(w))
        return None

    def gen_rope_parameters_flat():
        """官方**同时**有两份：摊平的 rope_scaling，和带逐层 rope_theta 的
        rope_parameters。后者就是前者加一个数组。"""
        sc = gen_rope_scaling()
        if sc is None:
            return None
        out = dict(sc)
        # 内嵌的 rope_theta 也要和顶层那份**一样**按周期补齐（48），
        # 否则两份不一致。
        out["rope_theta"] = gen_pad("per_layer_rope(rope_theta)", 48,
                                    "cycle")
        return out

    def resolve_field(val):
        v = val.strip()
        if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        if v in ("true", "false"):
            return v == "true"
        if v == "null":
            return None
        if len(v) >= 2 and v.startswith("[") and v.endswith("]"):
            return [_coerce(x) for x in split_top(v[1:-1])]
        if len(v) >= 2 and v.startswith("{") and v.endswith("}"):
            # 嵌套字面量，例如 quantization_config。
            # 值是 "a": b, "c": d 的形式（键必须带引号，值是 _coerce 处理）。
            out = {}
            for item in split_top(v[1:-1]):
                if ":" not in item:
                    continue
                k, val = item.split(":", 1)
                out[k.strip().strip('"')] = resolve_field(val.strip())
            return out
        if v in ("schedule", "layer_types"):
            return gen_layer_types()
        if v in ("position", "rope_parameters"):
            return gen_rope_parameters()
        if v == "rope_scaling":
            return gen_rope_scaling()
        if v == "rope_parameters_flat":
            return gen_rope_parameters_flat()
        if v == "sliding_window":
            return gen_sliding_window()
        if v == "rope_theta":
            # 全局的 rope base —— 有些模型的 Attention 机制里没写，
            # 只在 position 里声明了。
            for _v in (pb.assigns.values() if pb else []):
                _d = parse_rope(_v)
                # position 里写的是 rope(base = N)，键叫 base
                if _d.get("base") is not None:
                    return _coerce(_d["base"])
                if _d.get("rope_theta") is not None:
                    return _coerce(_d["rope_theta"])
            return None
        if v == "initial_context_length":
            for _v in (pb.assigns.values() if pb else []):
                _d = parse_rope(_v)
                if _d.get("original_max_position_embeddings") is not None:
                    return _coerce(_d["original_max_position_embeddings"])
                if _d.get("original") is not None:
                    return _coerce(_d["original"])
            return None
        if v == "attention_other_setting":
            return gen_attention_other_setting()
        m = re.match(r"^by_layer\(\s*([\w.]+)\s*\)$", v)
        if m:
            return gen_by_layer(m.group(1))
        m = re.match(r'^per_layer_rope\(\s*([\w.]+)\s*\)$', v)
        if m:
            return gen_per_layer_rope(m.group(1))
        m = re.match(r'^join\(\s*(.+?)\s*,\s*"(.*?)"\s*\)$', v)
        if m:
            return gen_join(m.group(1), m.group(2))
        m = re.match(r"^per_layer\(\s*([\w.]+)\s*,\s*([\w.-]+)\s*\)$", v)
        if m:
            return gen_per_layer_d(m.group(1), _coerce(m.group(2)))
        m = re.match(r"^pad_cycle\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)), "cycle")
        m = re.match(r"^pad_last\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)), "last")
        m = re.match(r"^pad\(\s*(.+?)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return gen_pad(m.group(1), int(m.group(2)))
        m = re.match(r"^per_layer\(\s*([\w.]+)\s*\)$", v)
        if m:
            return gen_per_layer(m.group(1))
        m = re.match(r"^indices_where\(\s*([\w.]+)\s*=\s*([\w.-]+)\s*\)$", v)
        if m:
            key, want = m.group(1), m.group(2)
            idx = []
            for li, (_s, _m, a, _k, atts) in enumerate(layer_seq):
                if key.startswith("attach."):
                    sub, val = key[7:], ""
                    for am in atts:
                        ma = mechs.get(am)
                        if ma is not None:
                            val = ma.assigns.get(sub, ma.mtype)
                            break
                else:
                    val = a.get(key, "")
                if str(val).strip() == want:
                    idx.append(li)
            return idx
        m = re.match(r"^([A-Za-z_]\w*)\[([^\]]*)\]\.(\w+)$", v)
        if m:
            return lookup_attr(m.group(1), m.group(2), m.group(3))
        m = re.match(r"^([A-Za-z_]\w*)\.(\w+)$", v)
        if m:
            return lookup_attr(m.group(1), "", m.group(2))
        if v in hp:
            return _coerce(hp[v])
        if v in scope.assigns:              # 模型级属性，如 ctx
            return _coerce(scope.assigns[v])
        # 兜底：数值 / 裸字面量。写错的名字不会被静默吞掉 ——
        # 它会在 by1verify --config 的逐字段对拍里以「值不同」暴露。
        return _coerce(v)

    cfg_out: Dict[str, Any] = {}
    for key, rule in emit_rules.items():
        for fname, fval in rule.get("fields", {}).items():
            cfg_out[fname] = resolve_field(fval)
    for key, rule in emit_rules.items():
        for fname in rule.get("fields", {}):
            if str(cfg_out.get(fname, "")).startswith("<未解析"):
                rep.add(E, rule["line"], "config",
                        f"emit {key} 的字段 '{fname}' 的值无法解析：{cfg_out[fname]}")

    # ---- 状态尺寸 -------------------------------------------------
    # 按「选择器」把 state 声明落到具体的层上。同一个机制的两类层状态可以不同：
    #   GQA[window = none]  : kv_cache(grows_with_seq)
    #   GQA[window != none] : kv_cache(bounded_by = window - 1)
    states = []
    dtype_b = 2  # bf16 默认
    em = scope.first("emit")
    if em:
        for a, b in em.assigns.items():
            m = re.search(r"dtype\s*=\s*(\w+)", b)
            if m and m.group(1) in ("fp32", "float32"):
                dtype_b = 4

    inst_by_mech: Dict[str, List[Dict[str, str]]] = {}
    for (_s, _m, _a, _k, _t) in layer_seq:
        inst_by_mech.setdefault(_m, []).append(_a)

    def eval_with(expr: str, attrs: Dict[str, str]):
        s2 = expr
        for k2, v2 in attrs.items():
            n2 = eval_num(v2)
            if n2 is not None:
                s2 = re.sub(r"\b" + re.escape(k2) + r"\b", _fmt(n2), s2)
        return eval_num(s2)

    for nm, ents in state_src.items():
        blk = mechs.get(nm) or memories.get(nm)
        for key, val, ln in ents:
            _n, conds = parse_state_key(key)
            sel = [a for a in inst_by_mech.get(nm, []) if selector_matches(conds, a)]
            n = len(sel)
            tag = key if conds else nm
            h = heads_of(blk) if blk else None
            hd = head_dim_of.get(nm)
            if "recurrent" in val:
                if h and "v" in h and hd:
                    per = h["v"] * hd * hd * dtype_b
                    states.append((f"{tag} recurrent", f"{per/2**20:,.2f} MiB/层",
                                   f"{n} 层", f"{per*n/2**20:,.2f} MiB"))
                else:
                    states.append((f"{tag} recurrent", "无法推导", f"{n} 层",
                                   "缺 heads(v)/head_dim"))
            elif "kv_cache" in val:
                kv = None
                if h:
                    kv = h.get("kv", h.get("kv_heads"))
                if kv is None and blk is not None:
                    kv = eval_num(blk.assigns.get("kv_heads"))
                if not (kv and hd):
                    states.append((f"{tag} kv_cache", "无法推导", f"{n} 层",
                                   "缺 kv heads 或 head_dim —— 算不出来"))
                    continue
                per_tok = kv * hd * 2 * dtype_b
                mb = re.search(r"bounded_by\s*=\s*([^,)]+)", val)
                if mb:
                    bounds = [eval_with(mb.group(1).strip(), a) for a in sel]
                    if any(b is None for b in bounds):
                        states.append((f"{tag} kv_cache(bounded)",
                                       f"{per_tok:,.0f} B/token", f"{n} 层",
                                       "有界，但上界表达式算不出来"))
                    else:
                        tot = per_tok * sum(bounds)
                        bset = sorted({int(b) for b in bounds})
                        bt = str(bset[0]) if len(bset) == 1 else str(bset)
                        states.append((f"{tag} kv_cache(bounded)",
                                       f"{per_tok:,.0f} B/token", f"{n} 层",
                                       f"上界 {bt} → {tot/2**20:,.2f} MiB 常量"))
                else:
                    states.append((f"{tag} kv_cache", f"{per_tok:,.0f} B/token",
                                   f"{n} 层", f"{per_tok*n:,.0f} B/token (全模型)"))

    # ---- 未填占位符 ----------------------------------------------
    nq = 0
    for line in text.splitlines():
        nq += len(re.findall(r"(?<![\w.])\?(?![\w.])", strip_comment(line)))
    if nq:
        rep.add(I, 0, "todo", f"未填占位符 {nq} 处（?）—— 文件尚未完成")

    layer_out = []
    for _li, (s, m, a, k, atts) in enumerate(layer_seq):
        entry = [(m, r) for r in class_rows.get((m, k), [])]
        for am in atts:
            _at = dict(resolve_attrs(mechs[am], {}))
            _at.update(overrides.get((s, _li, am), {}))
            entry += [(am, r) for r in class_rows.get((am, key_of(am, _at)), [])]
        for _r in layer_rows:
            _gd = _r[2]
            if _gd is not None and _gd not in atts:
                continue          # 这层没挂那个机制，就没有这个张量
            entry.append(("layer", (_r[0], _r[1],
                                    (f"仅当挂载 {_gd}" if _gd else ""), _r[3])))
        layer_out.append((s, m, a, entry))

    # ---- 汇报 -----------------------------------------------------
    return rep, {
        "stacks": [(s.name, s.alias, len(expansion.get(s.name, []))) for s in stacks],
        "expansion": expansion,
        "states": states,
        "mechs": {n: b.mtype for n, b in mechs.items()},
        "d_model": d_model,
        "vocab": vocab,
        "total_layers": total_layers,
        "decl_params": decl_params,
        "tens": trows,
        "layers": [(s, m, a) for (s, m, a, k, _t) in layer_seq],
        # 逐层覆盖要传给下游 —— 否则契约按逐层算、而生成的计算全用默认值，
        # 那就是一个**看起来对的错模型**：张量检查全过，跑起来每层宽度都一样。
        "overrides": {"%s|%d|%s" % k: dict(v)
                      for k, v in overrides.items()},
        "layer_out": layer_out,
        "emit": emit_rules,
        "state": state_src,
        "config": cfg_out,
        "layer_seq": layer_seq,
        "ctx": ctx,
        "hparams": hp,
        "position": (pb.assigns if pb else {}),
        "named": named,
        # 逐层类型 + 按类型的 RoPE 参数 —— codegen 靠这两个把 position 绑到正确的层上
        "layer_types": gen_layer_types(),
        "rope_by_type": gen_rope_parameters(),
        "mech_attrs": {nm: resolve_attrs(b, {}) for nm, b in mechs.items()},
        "global_rows": global_rows,
    }


# ════════════════════════════════════════════════════════════════════
# 输出
# ════════════════════════════════════════════════════════════════════

SEV_NAME = {E: "错误", W: "警告", I: "信息"}


def _runs(strs):
    out = []
    for s in strs:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return " + ".join(f"{c} x {s}" if c > 1 else s for s, c in out)


def fmt_seq(ex) -> str:
    """把展开结果压回一行可读的式子 —— 这是 '48 层本质上是一句话' 的兑现。"""
    strs = [str(r) for r in ex]
    n = len(strs)
    if n == 0:
        return "(空)"
    for p in range(1, n + 1):
        if n % p == 0 and all(strs[i] == strs[i % p] for i in range(n)):
            if p < n:
                return f"{n // p} x [ {_runs(strs[:p])} ]"
            break
    return _runs(strs)


def run_one(path: str) -> int:
    print()
    print("=" * 74)
    print(f"  {path}")
    print("=" * 74)
    try:
        rep, info = check(path)
    except FileNotFoundError:
        print(f"  [E] 文件不存在")
        return 1
    except Exception as ex:
        print(f"  [E] 检查器内部异常: {type(ex).__name__}: {ex}")
        return 1

    order = {E: 0, W: 1, I: 2}
    for sev, ln, code, msg in sorted(rep.items, key=lambda x: (order[x[0]], x[1])):
        loc = f"{ln:>4}" if ln else "   -"
        print(f"  [{sev}] {loc}  {msg}")

    if info["stacks"]:
        print()
        print("  展开层表")
        for name, al, n in info["stacks"]:
            print(f"    {name:<10} {al or '':<4} {n:>3} 层")
            ex = info["expansion"].get(name, [])
            if ex:
                print(f"      {fmt_seq(ex)}")
        print(f"    合计 {info['total_layers']} 层", end="")
        if info["d_model"]:
            print(f"，d_model = {int(info['d_model'])}", end="")
        if info["vocab"]:
            print(f"，vocab = {int(info['vocab'])}", end="")
        print()

    if info["states"]:
        print()
        print("  状态尺寸 (每序列, bf16)")
        print(f"    {'机制':<22}{'单位':<20}{'层数':<10}{'合计'}")
        for a, b, c, d in info["states"]:
            print(f"    {a:<22}{b:<20}{c:<10}{d}")

    if info["decl_params"]:
        print()
        print("  声明的参数量")
        for k, v in info["decl_params"].items():
            print(f"    {k:<22}{v:>20,.0f}")

    if info["tens"]:
        print()
        print("  张量契约实例化 (机制 x 结构属性等价类)")
        for nm, label, cnt, rows in info["tens"]:
            print(f"    {nm}({label})    {cnt} 层")
            for lname, shp, note, _pe in rows:
                line = f"      {lname:<9}{shp:<24}"
                if note:
                    line += f"<- {note}"
                print(line)

    print()
    print(f"  摘要: {rep.count(E)} 错误 / {rep.count(W)} 警告 / {rep.count(I)} 信息")
    return 1 if rep.count(E) else 0


def main(argv):
    args = list(argv)
    emit_to = None
    if "--emit-config" in args:
        i = args.index("--emit-config")
        emit_to = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args:
        print(__doc__)
        return 2

    if emit_to:
        _rep, info = check(args[0])
        cfg = info.get("config") or {}
        if not cfg:
            print(f"{args[0]}: 没有 transformers.config 的 field 映射，无法生成")
            return 2
        with open(emit_to, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"生成 config: {emit_to}   ({len(cfg)} 个字段)")
        return 0

    rc = 0
    for f in args:
        rc |= run_one(f)
    print()
    print(f"by1 check v0.1 —— {len(args)} 个文件")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
