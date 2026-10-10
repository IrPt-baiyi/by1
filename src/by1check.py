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

import by1io          # noqa: F401  —— import 它就够：它在 import 时钉 UTF-8
import by1paths
import by1skip
import by1export
import by1blocks
import by1tens
import by1lower
import by1hp
import by1sched
import by1stacks
import by1state

#: **"这个 field 映射指不到东西"的占位符。**
#
# 产出方（`lookup_attr`）和判据（`--emit-config` 那段）**必须用同一个常量**。
# 以前它们各写一份字面量：产出的是 `<找不到 ...>`，判的是 `startswith("<未解析")`
# —— **永远匹配不上**，于是"字段引用了一个不存在的机制/属性"这条错误
# 从来没有报过：配置里写进一个 `<找不到 NoSuchMech[].bar>` 的字符串，
# 而 `by1check` 报 0 错误。两个意思一个常量，就不会再有"谁忘了改"。
MISSING_PREFIX = "<找不到 "
MISSING = MISSING_PREFIX + "%s[%s].%s>"


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
                expert=None) -> str:
    """按后端 lowering 规则生成物理张量名。张量契约保持纯逻辑。"""
    scope = rule.get("scope", {}).get(mech, "")
    physical = rule.get("rename", {}).get(logical, logical)
    # **一个模型可以有多个栈，而各栈的物理前缀往往不同。**
    # 例：Qwen3.5 的文本主干是 `model.language_model.layers.{i}.…`，
    # 而 MTP 那几层是 `mtp.layers.{i}.…` —— 同一个 checkpoint 里两套前缀。
    # 所以名字模板允许**按栈各写一份**：`name_<栈名>` 优先，`name` 兜底。
    # **优先级：expert_name > name_<栈名> > name。**
    # 传了专家序号就说明这一行是"一个专家一个张量"的，
    # 而专家模板里通常带 `experts.{expert}.` 那一段 ——
    # 它必须压过按栈的名字模板，否则那一段会被静默吞掉
    # （表现：名字看起来正常，只是少了中间一层）。
    if expert is not None and (rule.get("expert_name_" + stack)
                               or rule.get("expert_name")):
        # 专家模板**也按栈分开** —— 主干在 backbone.layers.{i}.mixer.experts.{e}.，
        # MTP 在 mtp.layers.{i}.mixer.experts.{e}.，前缀不同。
        s = rule.get("expert_name_" + stack) or rule["expert_name"]
    else:
        s = rule.get("name_" + stack) or rule.get("name", "")
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
    # **model 名允许 `/`。**
    #
    # 短名在"同名不同源"时要带 owner：
    #     Step-3_7-Flash            官方
    #     nerkyor/Step-3_7-Flash    剪枝版（同名，不同源）
    #
    # 原来的正则没有 `/`，于是 `nerkyor/Step-3_7-Flash` 会被**切到
    # `nerkyor` 就停** —— 而因为它不报错，看起来像是支持的。
    # **一个"看起来能用"的解析比直接报错更坏。**
    m = re.match(r"^(model)\s+([A-Za-z_][\w.\-]*(?:/[\w.\-]+)?)", h)
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
        # **逗号也是分隔符**，于是可以写显式的逐层序列：
        #     pattern = [Mamba, MoE, Mamba, MoE, ...]
        # 这不是为了好看 —— 有些模型的层序就是**手写的表**
        # （Nemotron-H 的 layers_block_type，52 项，attention 落在
        #  5,12,19,26,33,42，间隔 7,7,7,7,**9** —— 不是纯周期）。
        # 那种情况下凑一个周期表达式是**推导**，不是**描述**。
        while self.peek()[1] in ("+", ","):
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

#: **这个语言认识的顶层关键字。** 不在里面的顶层块 `词 词 {` 会被
#: 解析器**直接跳过** —— 所以 `check()` 要把它报出来（见那里的注释）。
#:
#: 这一份不是抄的：是从 `by1check` 认这些块的**字面量**里读出来的 ✓。
#: 少列一个的后果是假报，而门会跑全部 27 份 `.by1`，假报当场露出来 ✓。
TOP_KEYWORDS = {
    "model", "hparams", "schedule", "stack", "position", "tensors",
    "emit", "head", "mech", "state", "optimizer", "memory",
    "residual", "interop",
}

TOKEN_MIXER = {"Attention", "Sparse", "Linear", "SSM", "Vision", "Recurrent",
               "MLA",
               # **逃生舱。** 计算在声明层之外实现（raw.py 里的一个工厂函数），
               # 但**张量契约、三个后端对拍、取值门全都照旧生效** ——
               # 它不是"绕过检查"，是"这个机制不在这门语言能表达的范围内"。
               # 接 ggml 时最先需要的就是它：没有这个口子，
               # 一个新机制只能等整个后端做完才能试。
               "Raw",
               # 逃生舱第二层：引用外部符号（.so + ABI）。
               "External",
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
    # 逃生舱的工厂函数名（`mech X : Raw { impl = "foo" }`）
    "impl",
    # 逃生舱第二层（`mech X : External { lib = ... symbol = ... }`）
    "lib", "symbol", "weights", "io",
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
    # **裸名 -> models/<名>，路径原样。**
    #
    # 模型搬进 `models/` 之后，让每个调用方各自去拼这个前缀，就是把同一个
    # 意思写二十遍 —— 而这里是**全部 `.by1` 的唯一入口**（by1exec / by1c /
    # by1diff / by1mem / by1emit / by1verify / by1gate / by1train /
    # by1gpt2 / by1load 都走 `check()`）。
    #
    # `Report` 里记的仍然是**调用方给的名字** —— 报出来的该是你问的那个。
    # ---- 命名空间 -------------------------------------------------
    # **主体不再有局部名字 —— 一切在 `_` 里**（含那 7 个嵌套 def）。
    #
    # 于是「搬一个 stage 出去」只需要接 `_` 一个参数，**不需要枚举
    # 任何名字** —— 而枚举名字正是前面 21 次失败的死因（Python 的
    # 函数局部名在编译期定死，清单要对就得有完整的数据流分析）。
    #
    # 改写是**按 AST 位置精确换文本**的：`x` -> `_['x']`，对读和写是
    # 同一个替换、所以不需要知道哪边是哪边。判据是 26 份 `.by1`
    # 的输出逐字节不变。
    _ = {}
    _['real'] = by1paths.model(path)
    with open(_['real'], "r", encoding="utf-8") as _['f']:
        _['text'] = _['f'].read()
    _['root'], _['perr'] = parse(_['text'])
    _['rep'] = Report(path)
    for _['e'] in _['perr']:
        _['rep'].add(E, 0, "parse", _['e'])

    # ---- 顶层块：**关键字不认识就要出声** --------------------------
    #
    # 解析器是按关键字逐个认的（`stack` / `mech` / `tensors` / `emit` …），
    # **不认识的顶层块被直接跳过、一个字都不报** ✗。
    #
    # 而它的后果比"没写"更坏。实测：`gemma-4-31B.by1` 里有一段
    #
    #     vision ViT {
    #       n_layer  = 27
    #       d_model  = 1152
    #       head_dim = 72
    #       patch    = 16
    #       pooling  = 3
    #     }
    #
    # —— 它看起来像在声明视觉塔的规模，而**删掉它，`check()` 的 24 个
    # 输出键里唯一变化的是行号**（每个正好差 7，就是删掉的行数）。
    # **语义贡献是零。** 读的人（包括写的人）会以为它生效了。
    #
    # 所以这里加一条：顶层出现一个不认识的块头，报出来。
    #
    # **已知的十四个关键字从哪来**：不是抄的，是从 `by1check` 自己
    # 认它们的那些字面量里读出来的 ✓。少列一个的后果是假报 ——
    # 而门会跑全部 27 份 `.by1`，假报会当场露出来 ✓。
    for _['i'], _['l'] in enumerate(_['text'].split("\n"), 1):
        _['mb'] = re.match(r"^  ([A-Za-z_]\w*)(?:\s+[A-Za-z_]\w*)?\s*\{",
                           _['l'])
        if _['mb'] and _['mb'].group(1) not in TOP_KEYWORDS:
            _['rep'].add(
                E, _['i'], "顶层块",
                "'%s' 不是这个语言认识的顶层关键字 —— **这一段不会被读**。"
                "认识的十四个：%s。"
                "（这一类最坏的地方是它看起来生效了："
                "实测删掉它，`check()` 的输出只有行号会变。）"
                % (_['mb'].group(1), " / ".join(sorted(TOP_KEYWORDS))))

    _['model'] = _['root'].first("model")
    _['scope'] = _['model'] if _['model'] else _['root']

    _['mechs']: Dict[str, Blk] = {}
    _['memories']: Dict[str, Blk] = {}
    _['residuals']: Dict[str, Blk] = {}
    for _['c'] in _['scope'].children:
        if _['c'].kind == "mech":
            _['mechs'][_['c'].name] = _['c']
        elif _['c'].kind == "memory":
            _['memories'][_['c'].name] = _['c']
        elif _['c'].kind == "residual":
            _['residuals'][_['c'].name] = _['c']

    # ---- hparams：**搬去 by1hp.py 了**
    by1hp.read_hparams(_mod=globals(), _=_)
    # ---- schedule 命名子序列：**搬去 by1sched.py 了**
    by1sched.read_schedule(_mod=globals(), _=_)
    # ---- stacks：**搬去 by1stacks.py 了**
    by1stacks.read_stacks(_mod=globals(), _=_)
    # ---- 各块一致性检查：**搬去 by1blocks.py 了** ------------------
    #
    # 九个块只看"声明之间自相不相容"，和张量契约不是一回事。
    #
    # **它只接一个参数** —— `_` 就是这个命名空间。曾经搬失败过 21
    # 次，因为那时要枚举"这一段依赖哪些局部名字"；现在没有局部
    # 名字了。
    by1blocks.check_blocks(_mod=globals(), _=_)
    # ---- 张量契约实例化：**搬去 by1tens.py 了** -------------
    #
    # **只接一个参数**（`_` 就是这个命名空间）。
    by1tens.instantiate(_mod=globals(), _=_)
    # ---- emit lowering 规则：**搬去 by1lower.py 了** ----
    #
    # **只接一个参数**（`_` 就是这个命名空间）。
    by1lower.emit_rules(_mod=globals(), _=_)
    # ---- 导出层：**已经不在这里了** ----
    #
    # 这里原来嵌着 13 个 `gen_*` / `name_*`，算的是"config.json 里那个
    # 字段该写什么" —— 和"第 i 层是什么机制"是两套词汇表。搬去了
    # `by1export.py`（那个文件的头注写了为什么劈、判据是什么）。
    #
    # **放在这里是因为再往前不行**：这 13 个函数要用 `parse_rope` 和
    # `resolve_field`，而它们分别在本行上面不远处才定义完。
    #
    # 下面把它们绑回本地名字，**所以这个函数里所有调用点一个字不用改**。
    _['_exp'] = by1export.make(
        _main_stack_names=_['_main_stack_names'], hp=_['hp'], layer_seq=_['layer_seq'],
        mechs=_['mechs'], named=_['named'], overrides=_['overrides'], pb=_['pb'], rep=_['rep'],
        scope=_['scope'], ROPE_DROP=_['ROPE_DROP'], ROPE_KEYS=_['ROPE_KEYS'], W=W,
        MISSING=MISSING, eval_num=eval_num, parse_state_key=parse_state_key,
        selector_matches=selector_matches, split_top=split_top)
    (_['gen_layer_types'], _['name_layer_type'], _['gen_rope_parameters'],
     _['gen_rope_scaling'], _['gen_attention_other_setting'], _['gen_by_layer'],
     _['gen_per_layer_rope'], _['gen_join'], _['gen_pad'], _['gen_per_layer'],
     _['gen_per_layer_d'], _['gen_sliding_window'],
     _['gen_rope_parameters_flat'],
     # **这 6 个也搬过去了**（它们曾经被当成"导出层的上游"，其实不是 ——
     # `resolve_field` 把 `emit { config = ... }` 的字段值解析出来，
     # 而 `schedule` / `position` / `rope_scaling` 这些名字**直接就是
     # 那 13 个生成器的入口**。它们和生成器是一层。）
     _['resolve_field'], _['lookup_attr'], _['_coerce'], _['_ltype_of_attrs'],
     _['_rope_params'], _['parse_rope']) = (
        _['_exp'][n] for n in (
            'gen_layer_types', 'name_layer_type', 'gen_rope_parameters',
            'gen_rope_scaling', 'gen_attention_other_setting',
            'gen_by_layer', 'gen_per_layer_rope', 'gen_join', 'gen_pad',
            'gen_per_layer', 'gen_per_layer_d', 'gen_sliding_window',
            'gen_rope_parameters_flat',
            'resolve_field', 'lookup_attr', '_coerce', '_ltype_of_attrs',
            '_rope_params', 'parse_rope'))
    _['cfg_out']: Dict[str, Any] = {}
    for _['key'], _['rule'] in _['emit_rules'].items():
        for _['fname'], _['fval'] in _['rule'].get("fields", {}).items():
            _['cfg_out'][_['fname']] = _['resolve_field'](_['fval'])
    for _['key'], _['rule'] in _['emit_rules'].items():
        for _['fname'] in _['rule'].get("fields", {}):
            # **判据和产出方共用一个常量。** 见文件头上 `MISSING` 那段：
            # 这里原来写的是 `startswith("<未解析")`，而产出的是
            # `<找不到 ...>` —— 一条永远为假的判断，也就是一个不会响的报警。
            if str(_['cfg_out'].get(_['fname'], "")).startswith(MISSING_PREFIX):
                _['rep'].add(E, _['rule']["line"], "config",
                        f"emit {_['key']} 的字段 '{_['fname']}' 的值无法解析：{_['cfg_out'][_['fname']]}")

    # ---- 状态尺寸：**搬去 by1state.py 了**
    by1state.state_sizes(_mod=globals(), _=_)
    # ---- 未填占位符 ----------------------------------------------
    _['nq'] = 0
    for _['line'] in _['text'].splitlines():
        _['nq'] += len(re.findall(r"(?<![\w.])\?(?![\w.])", strip_comment(_['line'])))
    if _['nq']:
        _['rep'].add(I, 0, "todo", f"未填占位符 {_['nq']} 处（?）—— 文件尚未完成")

    _['layer_out'] = []
    for _['_li'], (_['s'], _['m'], _['a'], _['k'], _['atts']) in enumerate(_['layer_seq']):
        _['entry'] = [(_['m'], r) for r in _['class_rows'].get((_['m'], _['k']), [])]
        for _['am'] in _['atts']:
            _['_at'] = dict(_['resolve_attrs'](_['mechs'][_['am']], {}))
            _['_at'].update(_['overrides'].get((_['s'], _['_li'], _['am']), {}))
            _['entry'] += [(_['am'], r) for r in _['class_rows'].get((_['am'], _['key_of'](_['am'], _['_at'])), [])]
        # **逐层的张量要按这一层所在的栈取。** 视觉栈宽 1152、
        # 主干 5376 —— 同一份 `layer` 契约要算两遍（见 `by1tens`）。
        # 没声明宽度的栈沿用默认那一份 ✓。
        for _['_r'] in ((_['layer_rows_by_stack'] or {}).get(_['s'])
                        or _['layer_rows']):
            _['_gd'] = _['_r'][2]
            if _['_gd'] is not None and _['_gd'] not in _['atts']:
                continue          # 这层没挂那个机制，就没有这个张量
            _['entry'].append(("layer", (_['_r'][0], _['_r'][1],
                                    (f"仅当挂载 {_['_gd']}" if _['_gd'] else ""), _['_r'][3])))
        _['layer_out'].append((_['s'], _['m'], _['a'], _['entry']))

    # ---- 汇报 -----------------------------------------------------
    return _['rep'], {
        "stacks": [(s.name, s.alias, len(_['expansion'].get(s.name, []))) for s in _['stacks']],
        "expansion": _['expansion'],
        "states": _['states'],
        "mechs": {n: b.mtype for n, b in _['mechs'].items()},
        "d_model": _['d_model'],
        "vocab": _['vocab'],
        "total_layers": _['total_layers'],
        "decl_params": _['decl_params'],
        "tens": _['trows'],
        "layers": [(s, m, a) for (s, m, a, k, _t) in _['layer_seq']],
        # 逐层覆盖要传给下游 —— 否则契约按逐层算、而生成的计算全用默认值，
        # 那就是一个**看起来对的错模型**：张量检查全过，跑起来每层宽度都一样。
        "overrides": {"%s|%d|%s" % k: dict(v)
                      for k, v in _['overrides'].items()},
        "layer_out": _['layer_out'],
        "emit": _['emit_rules'],
        "state": _['state_src'],
        "config": _['cfg_out'],
        "layer_seq": _['layer_seq'],
        "ctx": _['ctx'],
        "hparams": _['hp'],
        "position": (_['pb'].assigns if _['pb'] else {}),
        "named": _['named'],
        # 逐层类型 + 按类型的 RoPE 参数 —— codegen 靠这两个把 position 绑到正确的层上
        "layer_types": _['gen_layer_types'](),
        "rope_by_type": _['gen_rope_parameters'](),
        "mech_attrs": {nm: _['resolve_attrs'](b, {}) for nm, b in _['mechs'].items()},
        "global_rows": _['global_rows'],
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


def main(argv=None):
    # **`argv` 要能省。** pyproject 里的入口是 `by1-check = "by1check:main"`，
    # 而 setuptools 生成的启动器是 `sys.exit(main())` —— **不带参数**。
    # 于是 `by1-check hello.by1` 一直是
    #     TypeError: main() missing 1 required positional argument: 'argv'
    # 这个 bug 没人发现，因为**包从来装不上**（pyproject 的 py-modules
    # 少了 39 个模块，连 by1paths / by1io 都没列）。
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    emit_to = None
    if "--emit-config" in args:
        i = args.index("--emit-config")
        if i + 1 >= len(args):
            # **原来这里是 `args[i + 1]`** —— 少了路径就 IndexError，
            # 一个 traceback 而不是一句用法。用法错误要长得像用法错误。
            print("  --emit-config 后面要给一个输出路径：\n"
                  "      python src/by1check.py --emit-config out.json <file.by1>")
            # --emit-config 少给了路径 -> by1skip.CALLER
            return by1skip.CALLER
        emit_to = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args:
        print(__doc__)
        # 一个参数都没有 -> by1skip.CALLER
        return by1skip.CALLER

    if emit_to:
        # **报告不能丢。** 原来这里是 `_rep, info = check(...)`，`_rep`
        # 再也没被看过 —— 于是一个有 E 级错误的模型（比如 n_layer = 99
        # 而实际 3 层）在 `--emit-config` 下**退出码 0、不打印任何诊断**，
        # 还照样吐出一份写着 99 的 config。两条路必须同一个判据。
        rep, info = check(args[0])
        for sev, ln, code, msg in sorted(rep.items, key=lambda x: x[1]):
            loc = f"{ln:>4}" if ln else "   -"
            print(f"  [{sev}] {loc}  {msg}")
        if rep.count(E):
            print(f"\n  {rep.count(E)} 个错误 —— **不生成 config**："
                  f"把错误写进去，下游看不出来")
            return 1
        cfg = info.get("config") or {}
        if not cfg:
            print(f"{args[0]}: 没有 transformers.config 的 field 映射，无法生成")
            # 这份 config 没有 field 映射 -> by1skip.CODE
            return by1skip.CODE
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
