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

    # ---- hparams -------------------------------------------------
    _['hp']: Dict[str, str] = {}
    _['nlayer_decl'] = None
    _['hb'] = _['scope'].first("hparams")
    if _['hb']:
        _['hp'] = dict(_['hb'].assigns)
        _['nlayer_decl'] = eval_num(_['hp'].get("n_layer"))
    _['d_model'] = eval_num(_['hp'].get("d_model"))
    _['vocab'] = eval_num(_['hp'].get("vocab"))

    # ---- schedule 命名子序列 --------------------------------------
    _['named']: Dict[str, List[Rec]] = {}
    _['sb'] = _['scope'].first("schedule")
    if _['sb']:
        # 多趟展开：命名子序列可以引用别的命名子序列，直到不再变化
        for _['_'] in range(8):
            _['before'] = {k: [str(r) for r in v] for k, v in _['named'].items()}
            for _['k'], _['v'] in _['sb'].assigns.items():
                _['ex'] = expand_pattern(_['v'], _['named'])
                if _['ex'] is not None:
                    _['named'][_['k']] = _['ex']
            if _['before'] == {k: [str(r) for r in v] for k, v in _['named'].items()}:
                break

    # ---- stacks --------------------------------------------------
    _['stacks'] = _['scope'].kids("stack")
    _['alias'] = {}
    _['expansion']: Dict[str, List[Rec]] = {}
    # (栈, 层号, 机制) -> {属性: 值}
    _['overrides']: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    _['pending_ov']: List[tuple] = []
    _['total_layers'] = 0
    for _['st'] in _['stacks']:
        if _['st'].alias:
            _['alias'][_['st'].alias] = _['st'].name
        _['alias'][_['st'].name] = _['st'].name

    for _['st'] in _['stacks']:
        _['pexpr'] = _['st'].assigns.get("pattern")
        if _['pexpr'] is None:
            _['rep'].add(W, _['st'].line, "pattern", f"stack {_['st'].name} 没有 pattern")
            continue
        _['ex'] = expand_pattern(_['pexpr'], _['named'])
        if _['ex'] is None:
            _['rep'].add(E, _['st'].line, "pattern", f"stack {_['st'].name} 的 pattern 无法解析")
            continue
        _['expansion'][_['st'].name] = _['ex']
        _['total_layers'] += len(_['ex'])

        # 逐层属性覆盖：`main[3..44] : MoE.experts = [255, 266, ...]`
        # 值按被选中的层顺序分配。逐层变化的专家数、swiglu_limit 都靠它。
        for _['_k'], _['_v'], _['_ln'] in _['st'].entries:
            _['_m'] = re.match(r"^([A-Za-z_]\w*)\s*\[\s*(.*?)\s*\]$", _['_k'])
            if not _['_m']:
                _['rep'].add(W, _['_ln'], "override",
                        f"stack {_['st'].name} 里的条目 '{_['_k']}' 不是逐层覆盖"
                        f"（要写成像 main[3..44] : MoE.experts = [...]）")
                continue
            _['_sn'], _['_sel'] = _['_m'].group(1), _['_m'].group(2)
            if _['alias'].get(_['_sn'], _['_sn']) != _['st'].name:
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 指的是栈 '{_['_sn']}'，但它写在 stack {_['st'].name} 里")
                continue
            _['_mq'] = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*(.*)$", _['_v'])
            if not _['_mq']:
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 的值要写成 <机制>.<属性> = [...]")
                continue
            _['_mech'], _['_attr'], _['_vals'] = _['_mq'].group(1), _['_mq'].group(2), _['_mq'].group(3).strip()
            if not (_['_vals'].startswith("[") and _['_vals'].endswith("]")):
                _['rep'].add(E, _['_ln'], "override",
                        f"逐层覆盖 '{_['_k']}' 的值必须是列表")
                continue
            if _['_mech'] not in _['mechs']:
                _['rep'].add(E, _['_ln'], "override", f"逐层覆盖引用了未声明的机制 '{_['_mech']}'")
                continue
            _['_items'] = [x.strip() for x in split_top(_['_vals'][1:-1]) if x.strip()]
            # resolve_attrs 还没定义，先把原始的攒起来，等它可用再分配
            _['pending_ov'].append((_['st'].name, _['_sel'], _['_mech'], _['_attr'], _['_items'], _['_ln'], _['_k']))
        if _['st'].decl_len is not None and _['st'].decl_len != len(_['ex']):
            _['rep'].add(E, _['st'].line, "layers",
                    f"stack {_['st'].name} 声明 {_['st'].decl_len} 层，pattern 展开出 {len(_['ex'])} 层")
        elif _['st'].decl_len is not None:
            _['rep'].add(I, _['st'].line, "layers",
                    f"stack {_['st'].name} 的层数在两处声明（[{_['st'].decl_len}] 与 pattern），"
                    f"当前一致 —— 建议 [N] 只作断言")
        for _['r'] in _['ex']:
            if _['r'].name not in _['mechs']:
                if _['r'].name in _['memories']:
                    _['rep'].add(W, _['st'].line, "resolve",
                            f"{_['r'].name} 声明在 memory 块，却被 pattern 当作层机制引用")
                elif _['r'].name in _['residuals']:
                    _['rep'].add(E, _['st'].line, "resolve",
                            f"{_['r'].name} 是 residual，不能出现在 pattern 里")
                else:
                    _['rep'].add(E, _['st'].line, "resolve",
                            f"pattern 引用了未声明的机制 '{_['r'].name}'")
            else:
                _['m'] = _['mechs'][_['r'].name]
                for _['a'] in _['r'].attrs:
                    if _['a'] not in _['m'].assigns and _['a'] not in BUILTIN_ATTRS:
                        _['rep'].add(W, _['st'].line, "attr",
                                f"{_['r'].name}({_['a']} = ...) 的属性 '{_['a']}' 既未在 mech "
                                f"{_['r'].name} 声明，也不在已知属性表里")

    def _is_aux(st):
        """栈是不是辅助栈（MTP / 投机解码头之类）。
        **显式声明，不靠栈名猜。** 定义必须在使用之前 ——
        闭包在运行时才解析名字，放后面会 NameError。
        """
        _['v'] = (st.assigns.get("aux") or "").strip().lower()
        return _['v'] in ("true", "1", "yes", "on")
    _['_is_aux'] = _is_aux    # 让 stage 也取得到它

    _['_main_stack_names'] = {st.name for st in _['stacks'] if not _['_is_aux'](st)}

    # **辅助栈不算解码层。** MTP（多 token 预测）是训练时的辅助头，
    # 它在权重里、但不在 num_hidden_layers 里 —— 主干的层数才是那个数。
    # 靠栈名判断是魔法，所以让 .by1 显式写 ux = true。
    _['_main_layers'] = sum(len(_['expansion'].get(st.name, []))
                       for st in _['stacks']
                       if not _['_is_aux'](st))
    if _['nlayer_decl'] is not None and _['stacks']:
        if _['nlayer_decl'] != _['_main_layers']:
            _['rep'].add(E, _['hb'].line, "layers",
                    f"hparams.n_layer = {int(_['nlayer_decl'])}，但主栈合计 {_['_main_layers']} 层"
                    + (f"（另有辅助栈 {_['total_layers'] - _['_main_layers']} 层）"
                       if _['total_layers'] != _['_main_layers'] else ""))

    # ---- 各块一致性检查：**搬去 by1blocks.py 了** ------------------
    #
    # 九个块只看"声明之间自相不相容"，和张量契约不是一回事。
    #
    # **它只接一个参数** —— `_` 就是这个命名空间。曾经搬失败过 21
    # 次，因为那时要枚举"这一段依赖哪些局部名字"；现在没有局部
    # 名字了。
    by1blocks.check_blocks(_mod=globals(), _=_)
    # ---- 张量契约实例化 ------------------------------------------
    # 契约按「机制」声明；实例按「(机制, 结构属性) 等价类」求值。
    # 这正是结构属性的连带代价：head_dim / kv_tie 一变，形状与张量集合都变。
    _['STRUCTURAL'] = {"head_dim", "kv", "q", "v", "qk", "kv_heads", "kv_tie",
                  "experts", "d_ff", "hidden", "intermediate", "out_dim"}
    _['decl_struct']: Dict[str, set] = {}
    for _['nm'], _['blk'] in _['mechs'].items():
        _['raw'] = _['blk'].assigns.get("structural")
        if _['raw']:
            _['body'] = _['raw'].strip()
            if _['body'].startswith("[") and "]" in _['body']:
                _['body'] = _['body'][1 : _['body'].rfind("]")]
            _['decl_struct'][_['nm']] = set(split_top(_['body']))

    # 1) 实例枚举 -> 等价类
    def resolve_attrs(m: Blk, over: Dict[str, str]) -> Dict[str, str]:
        _['a']: Dict[str, str] = {}
        for _['k'], _['v'] in m.assigns.items():
            if _['k'] not in ("heads", "structural"):
                _['a'][_['k']] = _['v']
        for _['k'], _['v'] in (_['heads_of'](m) or {}).items():
            _['a'][_['k']] = _fmt(_['v'])
        _['a'].update(over)
        return _['a']
    _['resolve_attrs'] = resolve_attrs    # 让 stage 也取得到它

    def key_of(nm: str, attrs: Dict[str, str]) -> tuple:
        _['sk'] = _['STRUCTURAL'] | _['decl_struct'].get(nm, set())
        return tuple(sorted((k, v) for k, v in attrs.items() if k in _['sk']))
    _['key_of'] = key_of    # 让 stage 也取得到它

    # 哪些层挂了哪些「通道混合器」—— 选择器（[:] / [0] / [1..39] / [step 4]）生效
    _['attach_rules']: List[tuple] = []
    for _['c'] in _['scope'].children:
        for _['sel'], _['op'], _['target'], _['ln'] in _['c'].attaches:
            _['base'] = _['target'].split("(")[0].strip()
            if _['base'] not in _['mechs'] or _['op'] != ">>":
                continue
            _['sk'], _['inner'] = split_selector(_['sel'])
            _['attach_rules'].append((_['alias'].get(_['sk'], _['sk']), make_layer_pred(_['inner']), _['base']))

    def attached_at(stack_name: str, li: int, attrs: Dict[str, str]) -> List[str]:
        return [b for (sn, pred, b) in _['attach_rules']
                if sn == stack_name and pred(li, attrs)]
    _['attached_at'] = attached_at    # 让 stage 也取得到它

    # 逐层覆盖的分配 —— 必须等 resolve_attrs 可用（它选择哪些层要按属性判断）
    for _['_sn'], _['_sel'], _['_mech'], _['_attr'], _['_items'], _['_ln'], _['_k'] in _['pending_ov']:
        _['_ex'] = _['expansion'].get(_['_sn'], [])
        _['_pred'] = make_layer_pred(_['_sel'])
        _['_hit'] = [i for i, r in enumerate(_['_ex'])
                if _['_pred'](i, _['resolve_attrs'](_['mechs'][r.name], r.attrs)
                         if r.name in _['mechs'] else {})]
        if len(_['_hit']) != len(_['_items']):
            _['rep'].add(E, _['_ln'], "override",
                    f"逐层覆盖 '{_['_k']}' 选中 {len(_['_hit'])} 层，但给了 {len(_['_items'])} 个值")
            continue
        for _['_i'], _['_val'] in zip(_['_hit'], _['_items']):
            _['overrides'].setdefault((_['_sn'], _['_i'], _['_mech']), {})[_['_attr']] = _['_val']

    _['classes']: Dict[str, Dict[tuple, dict]] = {}

    _['layer_seq']: List[tuple] = []
    for _['st'] in _['stacks']:
        for _['li'], _['r'] in enumerate(_['expansion'].get(_['st'].name, [])):
            _['m'] = _['mechs'].get(_['r'].name)
            if _['m'] is None:
                continue
            _['attrs'] = _['resolve_attrs'](_['m'], _['r'].attrs)
            _['_ov'] = _['overrides'].get((_['st'].name, _['li'], _['r'].name))
            if _['_ov']:
                _['attrs'].update(_['_ov'])
            _['key'] = _['key_of'](_['r'].name, _['attrs'])
            _['layer_seq'].append((_['st'].name, _['r'].name, dict(_['attrs']), _['key'],
                              _['attached_at'](_['st'].name, _['li'], _['attrs'])))
            _['d'] = _['classes'].setdefault(_['r'].name, {})
            _['ent'] = _['d'].setdefault(_['key'], {"count": 0, "attrs": {}})
            _['ent']["count"] += 1
            _['ent']["attrs"].update(_['attrs'])

    # 只被挂载、不作为层机制出现的（如 FFN）：按每个挂载规则建类，层数按选择器数
    for (_['sn'], _['pred'], _['base']) in _['attach_rules']:
        if _['base'] in _['classes'] or _['base'] not in _['mechs']:
            continue
        _['buckets'] = {}
        for _['li'], _['r'] in enumerate(_['expansion'].get(_['sn'], [])):
            _['a'] = _['resolve_attrs'](_['mechs'][_['r'].name], _['r'].attrs) if _['r'].name in _['mechs'] else {}
            if not _['pred'](_['li'], _['a']):
                continue
            _['at'] = dict(_['resolve_attrs'](_['mechs'][_['base']], {}))
            _['at'].update(_['overrides'].get((_['sn'], _['li'], _['base']), {}))
            _['kk'] = _['key_of'](_['base'], _['at'])
            _['buckets'].setdefault(_['kk'], {"count": 0, "attrs": _['at']})
            _['buckets'][_['kk']]["count"] += 1
        _['classes'].setdefault(_['base'], {}).update(_['buckets'])

    # 2) 读契约
    _['tb'] = _['scope'].first("tensors")
    # 文件里既没有 tensors 也没有 emit，说明它是在**设计一个新模型**，
    # 不是在对拍一个已有产物。张量契约那类警告对它是噪音。
    _['wants_artifacts'] = (_['tb'] is not None) or (_['scope'].first("emit") is not None)
    _['contracts']: Dict[str, list] = {}
    if _['tb']:
        for _['ch'] in _['tb'].children:
            _['h'] = _['ch'].head.strip()
            if re.match(r"^layer\s*\(", _['h']):
                _['rep'].add(E, _['ch'].line, "tensors",
                        f"'{_['h']}' 是无作用域写法：会给所有层发同一套张量名。"
                        f"契约必须按机制声明（见 pattern-design §2 规则 3）")
                continue
            _['base'] = _['h'].split(".")[0].strip()
            if _['h'] in ("layer", "global"):
                pass                    # 层级 / 全局：不属于任何机制
            elif _['base'] not in _['mechs'] and _['base'] not in _['memories'] and _['base'] not in _['residuals']:
                _['rep'].add(E, _['ch'].line, "tensors", f"tensors 的作用域 '{_['base']}' 未声明")
                continue
            _['lst'] = []
            for _['key'], _['val'], _['ln'] in _['ch'].entries:
                _['mg'] = re.search(r"\bunless\s+([A-Za-z_]\w*)", _['val'])
                _['guard'] = _['mg'].group(1) if _['mg'] else None
                _['pe'] = bool(re.search(r"\bper_expert\b", _['val']))
                _['shp'] = re.sub(r"\bunless\s+[A-Za-z_]\w*", "", _['val'])
                _['shp'] = re.sub(r"\bper_expert\b", "", _['shp']).strip()
                _['lst'].append((_['key'], _['shp'], _['guard'], _['ln'], _['pe']))
            if not _['lst']:
                _['rep'].add(W, _['ch'].line, "tensors",
                        f"tensors 的 '{_['h']}' 契约是空的 —— 没有声明任何逻辑张量")
            _['contracts'][_['base']] = _['lst']

    def _flat_rows(entry_list):
        """层级 / 全局张量：形状只用 hparams 求值（没有逐层属性）。"""
        _['sym0']: Dict[str, float] = {}
        for _['k2'], _['v2'] in _['hp'].items():
            _['n2'] = eval_num(_['v2'])
            if _['n2'] is not None:
                _['sym0'][_['k2']] = _['n2']
        _['res'] = []
        for (_['lname'], _['shp'], _['guard'], _['ln'], _['pe']) in entry_list:
            _['body'] = _['shp'].strip()
            if _['body'].startswith("(") and ")" in _['body']:
                _['body'] = _['body'][1: _['body'].rfind(")")]
            _['comps'] = []
            for _['c'] in split_top(_['body']):
                _['ids'] = set(re.findall(r"[A-Za-z_]\w*", _['c']))
                _['s2'] = _['c']
                for _['i'] in sorted(_['ids'], key=len, reverse=True):
                    if _['i'] in _['sym0']:
                        _['s2'] = re.sub(r"\b" + re.escape(_['i']) + r"\b", _fmt(_['sym0'][_['i']]), _['s2'])
                _['v'] = eval_num(_['s2'])
                _['comps'].append(_fmt(_['v']) if _['v'] is not None else _['s2'].replace(" ", ""))
            # `--` 是**声明为不该存在**（权重共享时没有 lm_head），
            # 它不是形状，不能被重新包成 `(...)` —— 原样透传，
            # 让 by1verify 去检查"确实不存在"。
            if _['shp'].strip() == "--":
                _['res'].append((_['lname'], "--", _['guard'], _['pe']))
                continue
            # guard 留着 —— layer 作用域用它做「只有挂了某机制才有这个张量」
            _['res'].append((_['lname'], "(" + ", ".join(_['comps']) + ")", _['guard'], _['pe']))
        return _['res']
    _['_flat_rows'] = _flat_rows    # 让 stage 也取得到它

    _['layer_rows'] = _['_flat_rows'](_['contracts'].get("layer", []))
    _['global_rows'] = _['_flat_rows'](_['contracts'].get("global", []))

    # 3) 逐类实例化
    _['trows'] = []
    _['class_rows']: Dict[tuple, list] = {}
    for _['nm'] in sorted(_['classes']):
        _['cls'] = _['classes'][_['nm']]
        if _['nm'] not in _['contracts']:
            if _['wants_artifacts'] and _['mechs'][_['nm']].mtype in TOKEN_MIXER:
                _['rep'].add(W, _['mechs'][_['nm']].line, "tensors", f"mech {_['nm']} 无张量契约")
            continue
        for _['key'], _['ent'] in sorted(_['cls'].items(), key=lambda kv: -kv[1]["count"]):
            _['attrs'] = _['ent']["attrs"]
            _['label'] = ", ".join(f"{k}={v}" for k, v in _['key']) or "(无结构属性)"
            _['sym']: Dict[str, float] = {}
            for _['k'], _['v'] in _['attrs'].items():
                _['n'] = eval_num(_['v'])
                if _['n'] is not None:
                    _['sym'][_['k']] = _['n']
            for _['k'], _['v'] in _['hp'].items():
                _['n'] = eval_num(_['v'])
                if _['n'] is not None:
                    _['sym'][_['k']] = _['n']

            # 宽度不变式。注意：不能靠「比值是否好看」判定对错 —— Gemma 4 的
            # 注意力宽度本来就不等于 d_model。所以宽度变化必须显式声明。
            _['qq'], _['vv'], _['hdv'] = _['sym'].get("q"), _['sym'].get("v"), _['sym'].get("head_dim")
            _['w'] = (_['vv'] * _['hdv']) if (_['vv'] and _['hdv']) else ((_['qq'] * _['hdv']) if (_['qq'] and _['hdv']) else None)
            _['od'] = eval_num(_['attrs'].get("out_dim"))
            if _['w'] and _['od']:
                if abs(_['w'] - _['od']) > 1e-9:
                    _['rep'].add(E, _['mechs'][_['nm']].line, "shape",
                            f"{_['nm']}({_['label']}): 声明 out_dim = {_fmt(_['od'])}，"
                            f"但 q x head_dim = {_fmt(_['w'])}")
            elif _['w'] and _['d_model'] and abs(_['w'] - _['d_model']) > 1e-9:
                _['rep'].add(W, _['mechs'][_['nm']].line, "shape",
                        f"{_['nm']}({_['label']}): q x head_dim = {_fmt(_['w'])} != d_model = "
                        f"{_fmt(_['d_model'])}（比值 {_['w']/_['d_model']:.3f}）—— 若这是有意的宽度变化，"
                        f"声明 out_dim = {_fmt(_['w'])}")
            _['rows'] = []
            for _['lname'], _['shp'], _['guard'], _['ln'], _['pe'] in _['contracts'][_['nm']]:
                if _['guard'] is not None:
                    _['gs'] = _['attrs'].get(_['guard'])
                    if _['gs'] is None:
                        _['rep'].add(W, _['ln'], "tensors",
                                f"{_['nm']}.{_['lname']}: unless {_['guard']} 引用了未声明的属性")
                        continue
                    if str(_['gs']).strip().lower() in ("true", "1", "yes", "on"):
                        _['rows'].append((_['lname'], "--", f"由 unless {_['guard']} 抑制", _['pe']))
                        continue
                _['body'] = _['shp'].strip()
                if _['body'].startswith("(") and ")" in _['body']:
                    _['body'] = _['body'][1 : _['body'].rfind(")")]
                _['out'], _['miss'] = [], []
                for _['c'] in split_top(_['body']):
                    _['ids'] = set(re.findall(r"[A-Za-z_]\w*", _['c']))
                    _['bad'] = sorted(i for i in _['ids'] if i not in _['sym'])
                    if _['bad']:
                        _['miss'].extend(_['bad'])
                    _['s2'] = _['c']
                    for _['i'] in sorted(_['ids'], key=len, reverse=True):
                        if _['i'] in _['sym']:
                            _['s2'] = re.sub(r"\b" + re.escape(_['i']) + r"\b", _fmt(_['sym'][_['i']]), _['s2'])
                    _['v'] = eval_num(_['s2'])
                    _['out'].append(_fmt(_['v']) if _['v'] is not None else _['s2'].replace(" ", ""))
                _['note'] = ("缺 " + ", ".join(sorted(set(_['miss'])))) if _['miss'] else ""
                _['rows'].append((_['lname'], "(" + ", ".join(_['out']) + ")", _['note'], _['pe']))
            _['label'] = ", ".join(f"{k}={v}" for k, v in _['key']) or "(无结构属性)"
            _['class_rows'][(_['nm'], _['key'])] = _['rows']
            _['trows'].append((_['nm'], _['label'], _['ent']["count"], _['rows']))

    for _['nm'], _['lst'] in _['contracts'].items():
        if _['nm'] in ("layer", "global"):
            continue
        if _['nm'] not in _['classes'] and _['nm'] not in _['attached']:
            _['rep'].add(W, _['lst'][0][3] if _['lst'] else 0, "tensors",
                    f"tensors 声明了 {_['nm']} 的契约，但 pattern 里没有它的实例")

    # ---- emit lowering 规则 --------------------------------------
    # 物理命名在后端里声明，张量契约保持纯逻辑 —— 这是「一份描述两个后端」
    # 能成立的前提。检查器负责校验模板与映射的引用是否合法。
    _['KNOWN_PH'] = {"i", "local_i", "global_i", "stack", "mech",
                "logical", "scope", "physical", "expert"}
    _['all_logical'] = {ln for lst in _['contracts'].values() for (ln, _s, _g, _l, _p) in lst}
    _['emit_rules']: Dict[str, dict] = {}
    _['eb'] = _['scope'].first("emit")
    if _['eb']:
        for _['be'] in _['eb'].children:
            _['key'] = _['be'].head.split("->")[-1].strip() if "->" in _['be'].head else _['be'].head.strip()
            _['rule'] = {"name": _['be'].assigns.get("name", "").strip().strip('"'),
                    "expert_name": _['be'].assigns.get("expert_name", "").strip().strip('"'),
                    "global_name": _['be'].assigns.get("global_name", "").strip().strip('"'),
                    "scope": {}, "rename": {}, "fields": {},
                    "quant": {}, "line": _['be'].line, "head": _['be'].head.strip()}
            # **按栈各写一份名字模板**：`name_<栈名>`。
            # 一个 checkpoint 里可以有多个栈、各套前缀
            # （Qwen3.5 主干是 model.language_model.layers.{i}.…，
            #   MTP 那层是 mtp.layers.{i}.…），一个模板装不下。
            # render_name 里 name_<栈名> 优先、name 兜底。
            for _['_k'], _['_v'] in _['be'].assigns.items():
                if (_['_k'].startswith("name_") or _['_k'].startswith("expert_name_")) \
                        and _['_k'] not in ("name_", "expert_name_"):
                    _['rule'][_['_k']] = _['_v'].strip().strip('"')
            for _['sub'] in _['be'].children:
                _['h'] = _['sub'].head.strip()
                if _['h'] == "quant":
                    for _['k'], _['v'] in _['sub'].assigns.items():
                        _['rule']["quant"][_['k']] = _['v'].strip().strip('"')
                    for _['sub2'] in _['sub'].children:
                        if _['sub2'].head.strip() == "fuse":
                            _['rule']["quant"]["fuse"] = dict(_['sub2'].assigns)
                    continue
                _['tgt'] = {"scope": _['rule']["scope"], "rename": _['rule']["rename"],
                       "field": _['rule']["fields"], "fields": _['rule']["fields"]}.get(_['h'])
                if _['tgt'] is None:
                    continue
                for _['k'], _['v'] in _['sub'].assigns.items():
                    _['tgt'][_['k']] = _['v'].strip().strip('"')
            _['emit_rules'][_['key']] = _['rule']

            if not _['rule']["name"]:
                if _['rule']["fields"]:
                    continue
                _['rep'].add(W, _['rule']["line"], "emit",
                        f"emit {_['key']} 既没有 name 模板也没有 field 映射")
                continue
            for _['ph'] in re.findall(r"\{(\w+)\}",
                                 _['rule']["name"] + _['rule']["expert_name"] + _['rule']["global_name"]):
                if _['ph'] not in _['KNOWN_PH']:
                    _['rep'].add(E, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 name 模板含未知占位符 {{{_['ph']}}}")
            for _['m'] in _['rule']["scope"]:
                if _['m'] not in ("layer", "global") and _['m'] not in _['mechs']:
                    _['rep'].add(E, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 scope 引用了未声明的机制 '{_['m']}'")
            # quant 的 fuse 名（如 gate_up_proj）和它点名的张量，只在导出时才存在，
            # 不是契约里的逻辑张量 —— 不该被当成拼写错误
            _['_qnames'] = set(re.findall(
                r"[A-Za-z_][\w.]*", (_['rule'].get("quant") or {}).get("tensor", "")))
            for _['_n'], _['_v'] in ((_['rule'].get("quant") or {}).get("fuse") or {}).items():
                _['_qnames'].add(_['_n'])
                _['_qnames'] |= set(re.findall(r"[A-Za-z_][\w.]*", _['_v']))
            for _['ln'] in _['rule']["rename"]:
                if _['ln'] in _['_qnames']:
                    continue
                if _['all_logical'] and _['ln'] not in _['all_logical']:
                    _['rep'].add(W, _['rule']["line"], "emit",
                            f"emit {_['key']} 的 rename 键 '{_['ln']}' 不是任何契约里的逻辑张量")
            _['used_logical'] = set(re.findall(r"\{logical\}", _['rule']["name"]))
            # 真不变式：同一个等价类里，不同的逻辑张量不能生成同一个物理名
            for (_['mn'], _['_mk']), _['rows'] in _['class_rows'].items():
                if _['rule']["scope"] and _['mn'] not in _['rule']["scope"]:
                    continue
                _['seen_nm']: Dict[str, str] = {}
                for (_['ln2'], _['_sh'], _['_nt'], _['_pe']) in _['rows']:
                    _['nm2'] = render_name(_['rule'], 0, "<stack>", _['mn'], _['ln2'])
                    if _['nm2'] in _['seen_nm'] and _['seen_nm'][_['nm2']] != _['ln2']:
                        _['rep'].add(E, _['rule']["line"], "emit",
                                f"emit {_['key']}: {_['mn']} 的 '{_['ln2']}' 与 '{_['seen_nm'][_['nm2']]}' "
                                f"生成同一个物理名 '{_['nm2']}'")
                    _['seen_nm'][_['nm2']] = _['ln2']





    _['ROPE_KEYS'] = {"base": "rope_theta", "type": "rope_type",
                 "partial": "partial_rotary_factor",
                 "ratio": "partial_rotary_factor",
                 "original": "original_max_position_embeddings"}
    _['ROPE_DROP'] = {"pairing"}          # 配对约定是 codegen 的事，不进 config
















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

    # ---- 状态尺寸 -------------------------------------------------
    # 按「选择器」把 state 声明落到具体的层上。同一个机制的两类层状态可以不同：
    #   GQA[window = none]  : kv_cache(grows_with_seq)
    #   GQA[window != none] : kv_cache(bounded_by = window - 1)
    _['states'] = []
    _['dtype_b'] = 2  # bf16 默认
    _['em'] = _['scope'].first("emit")
    if _['em']:
        for _['a'], _['b'] in _['em'].assigns.items():
            _['m'] = re.search(r"dtype\s*=\s*(\w+)", _['b'])
            if _['m'] and _['m'].group(1) in ("fp32", "float32"):
                _['dtype_b'] = 4

    _['inst_by_mech']: Dict[str, List[Dict[str, str]]] = {}
    for (_['_s'], _['_m'], _['_a'], _['_k'], _['_t']) in _['layer_seq']:
        _['inst_by_mech'].setdefault(_['_m'], []).append(_['_a'])

    def eval_with(expr: str, attrs: Dict[str, str]):
        _['s2'] = expr
        for _['k2'], _['v2'] in attrs.items():
            _['n2'] = eval_num(_['v2'])
            if _['n2'] is not None:
                _['s2'] = re.sub(r"\b" + re.escape(_['k2']) + r"\b", _fmt(_['n2']), _['s2'])
        return eval_num(_['s2'])
    _['eval_with'] = eval_with    # 让 stage 也取得到它

    for _['nm'], _['ents'] in _['state_src'].items():
        _['blk'] = _['mechs'].get(_['nm']) or _['memories'].get(_['nm'])
        for _['key'], _['val'], _['ln'] in _['ents']:
            _['_n'], _['conds'] = parse_state_key(_['key'])
            _['sel'] = [a for a in _['inst_by_mech'].get(_['nm'], []) if selector_matches(_['conds'], a)]
            _['n'] = len(_['sel'])
            _['tag'] = _['key'] if _['conds'] else _['nm']
            _['h'] = _['heads_of'](_['blk']) if _['blk'] else None
            _['hd'] = _['head_dim_of'].get(_['nm'])
            if "recurrent" in _['val']:
                if _['h'] and "v" in _['h'] and _['hd']:
                    _['per'] = _['h']["v"] * _['hd'] * _['hd'] * _['dtype_b']
                    _['states'].append((f"{_['tag']} recurrent", f"{_['per']/2**20:,.2f} MiB/层",
                                   f"{_['n']} 层", f"{_['per']*_['n']/2**20:,.2f} MiB"))
                else:
                    _['states'].append((f"{_['tag']} recurrent", "无法推导", f"{_['n']} 层",
                                   "缺 heads(v)/head_dim"))
            elif "kv_cache" in _['val']:
                _['kv'] = None
                if _['h']:
                    _['kv'] = _['h'].get("kv", _['h'].get("kv_heads"))
                if _['kv'] is None and _['blk'] is not None:
                    _['kv'] = eval_num(_['blk'].assigns.get("kv_heads"))
                if not (_['kv'] and _['hd']):
                    _['states'].append((f"{_['tag']} kv_cache", "无法推导", f"{_['n']} 层",
                                   "缺 kv heads 或 head_dim —— 算不出来"))
                    continue
                _['per_tok'] = _['kv'] * _['hd'] * 2 * _['dtype_b']
                _['mb'] = re.search(r"bounded_by\s*=\s*([^,)]+)", _['val'])
                if _['mb']:
                    _['bounds'] = [_['eval_with'](_['mb'].group(1).strip(), a) for a in _['sel']]
                    if any(b is None for b in _['bounds']):
                        _['states'].append((f"{_['tag']} kv_cache(bounded)",
                                       f"{_['per_tok']:,.0f} B/token", f"{_['n']} 层",
                                       "有界，但上界表达式算不出来"))
                    else:
                        _['tot'] = _['per_tok'] * sum(_['bounds'])
                        _['bset'] = sorted({int(b) for b in _['bounds']})
                        _['bt'] = str(_['bset'][0]) if len(_['bset']) == 1 else str(_['bset'])
                        _['states'].append((f"{_['tag']} kv_cache(bounded)",
                                       f"{_['per_tok']:,.0f} B/token", f"{_['n']} 层",
                                       f"上界 {_['bt']} → {_['tot']/2**20:,.2f} MiB 常量"))
                else:
                    _['states'].append((f"{_['tag']} kv_cache", f"{_['per_tok']:,.0f} B/token",
                                   f"{_['n']} 层", f"{_['per_tok']*_['n']:,.0f} B/token (全模型)"))

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
        for _['_r'] in _['layer_rows']:
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
