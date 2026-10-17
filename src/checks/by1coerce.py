#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1coerce -- **同一个取值写法，读到同一个答案。**

## 为什么要有这个

IR 里的布尔/数字属性有两种入口：

  1. **语言层**：`sink = per_head` / `bias = off` / `sink = 1` —— 描述性取值
     和明确否定混着写；
  2. **IR 层**（`render_ir` 接受任何合法 IR）：schema 钉住类型，字符串进不来。

各后端此前有三种布尔读法：

    bool(a.get(x))   Python 真值 —— `"false"` 是**真**（错）
    _bool(x)         白名单 —— `sink = per_head` 被静默关掉（漏）
    _flag(x)         黑名单 —— 只在明确否定时关（对）

2026-10-17 起只有 `by1ir.as_bool` / `by1ir.as_num` 一份。这个脚本是它的判据：

  ① 取值表：每个写法读出来的布尔/数字必须是定的；
  ② **语言层入口**：八种写法编出来的 IR 布尔必须对；
  ③ **IR 门口**：字符串写法必须被 schema 拒掉（否则后端就会去猜）；
  ④ 开/关必须真的**改变图和数**（否则前面三条都是空的）。

判据只看 NumPy：不依赖 torch / gcc，所以它在档 2 就能跑。

用法:  python by1coerce.py
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import by1check  # noqa: E402
import by1codegen  # noqa: E402
import by1exec  # noqa: E402
import by1graph  # noqa: E402
import by1ir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

#: 布尔取值表：(写法, 期望)。**黑名单语义**：明确否定才关。
BOOL_TABLE = (
    (True, True), (False, False), (None, False), (0, False), (1, True),
    ("true", True), ("True", True), (" on ", True), ("1", True),
    ("yes", True),
    ("false", False), ("False", False), ("off", False), ("no", False),
    ("none", False), ("null", False), ("0", False), ("", False),
    # **描述性取值算开** —— 这一条正是 `_bool` 白名单语义埋掉的那个 bug
    ("per_head", True), ("full", True),
)

#: 数字取值表。
NUM_TABLE = (("1e-5", 1e-5), (1e-5, 1e-5), (" 2 ", 2.0), (2, 2.0),
             (None, None), ("x", None))

#: 语言层的写法 -> IR 里该是**布尔**。
LANG_TABLE = (("per_head", True), ("full", True), ("true", True),
              ("on", True), ("1", True), ("off", False),
              ("false", False), ("0", False))

TPL = """model T {
  hparams { d_model = 64, vocab = 128 }
  stack main {
    pattern = 1 * Attention(q = 2, kv = 2, head_dim = 16, sink = %s)
  }
}"""


#: 探针文件写**临时目录**，不写仓内 `build/` —— 那个目录被 gitignore 了，
#: 干净检出里**不存在**（CI 的档 3 就是在这里红的：本地有 build/，CI 没有）。
_TMP = tempfile.mkdtemp(prefix="by1coerce-")


def _compile_text(text, tag):
    """把一小段 .by1 编译成 IR（只为探一个属性）。"""
    p = os.path.join(_TMP, "coerce-%s.by1" % tag)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    _rep, info = by1check.check(p)
    return by1codegen.compile_ir(info)


def _attr_of(ir):
    for L in ir["layers"]:
        for o in L["ops"]:
            if o["kind"] == "Attention":
                return o["attrs"]
    raise SystemExit("[FAIL] 探针模型里没有 Attention")


def _check_tables():
    bad = []
    for v, want in BOOL_TABLE:
        got = by1ir.as_bool(v)
        if got != want:
            bad.append("as_bool(%r) = %r，应为 %r" % (v, got, want))
    for v, want in NUM_TABLE:
        got = by1ir.as_num(v)
        if got != want:
            bad.append("as_num(%r) = %r，应为 %r" % (v, got, want))
    return bad


def _check_lang():
    """语言层八种写法 -> IR 布尔（四后端共用的唯一入口）。"""
    bad = []
    for sp, want in LANG_TABLE:
        try:
            a = _attr_of(_compile_text(TPL % sp, "lang-" + sp))
        except SystemExit as e:
            bad.append("sink = %s 编不过：%s"
                       % (sp, str(e).splitlines()[0][:60]))
            continue
        got = a.get("sink")
        if got is not want:
            bad.append("sink = %s -> IR %r，应为 %r" % (sp, got, want))
    return bad


def _check_validator():
    """IR 门口那道门：**字符串写法不许进 IR**（类型由 schema 钉住）。"""
    bad = []
    ir = _compile_text(TPL % "per_head", "gate")
    a = _attr_of(ir)
    for attr, spell in (("bias", "false"), ("sink", "off")):
        a[attr] = spell
        errs = by1ir.validate(ir)
        if not errs:
            bad.append("IR 里 %s = %r 居然被放行了（schema 没钉住类型）"
                       % (attr, spell))
    return bad


def _check_switch():
    """开/关必须真的**改变图和数** —— 否则前面的判据是空的。"""
    bad = []
    seen = {}
    for spelling in ("true", "false"):
        ir = _compile_text(TPL % spelling, "sw-" + spelling)
        g = by1graph.lower_model(ir, "sw-" + spelling)
        o, _p = by1exec.exec_ir(ir, seq=8)
        seen[spelling] = (g.sha(), np.asarray(o))
    if seen["true"][0] == seen["false"][0]:
        bad.append("sink 开/关降出同一张图 —— 判据是空的")
    d = float(np.abs(seen["true"][1] - seen["false"][1]).max())
    if d == 0.0:
        bad.append("sink 开/关输出一模一样 —— 这个属性没影响")
    return bad


def main():
    print()
    print("=" * 78)
    print("  by1coerce：同一个取值写法 -> 同一个答案")
    print("  （唯一源头：by1ir.as_bool / by1ir.as_num）")
    print("=" * 78)
    print()
    bad = _check_tables()
    print("  ① 取值表 %d 条布尔 + %d 条数字：%s"
          % (len(BOOL_TABLE), len(NUM_TABLE),
             "全对" if not bad else "%d 条不对" % len(bad)))
    for m in bad:
        print("     ! " + m)

    e2 = _check_lang()
    print("  ② 语言层 %d 种写法 -> IR 布尔：%s"
          % (len(LANG_TABLE), "全对" if not e2 else "%d 条不对" % len(e2)))
    for m in e2:
        print("     ! " + m)

    e3 = _check_validator()
    print("  ③ IR 门口：字符串写法被 schema 拒掉：%s"
          % ("是" if not e3 else "**否 —— %d 条漏网**" % len(e3)))
    for m in e3:
        print("     ! " + m)

    e4 = _check_switch()
    print("  ④ 开/关确实改变图和数：%s"
          % ("是" if not e4 else "**否 —— %d 条" % len(e4)))
    for m in e4:
        print("     ! " + m)

    bad += e2 + e3 + e4
    print()
    ok = not bad
    print("  [%s] 取值写法：%s"
          % ("PASS" if ok else "FAIL",
             "取值表/语言层/IR 门口/开关四关都过" if ok
             else "%d 条不对（上面逐条列出来了）" % len(bad)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())