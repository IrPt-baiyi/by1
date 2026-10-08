#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1ver -- **版本号的唯一真相源。**

## 为什么要有这个文件

在这之前，`"by1-ir": "1.0"` **硬编码在三个地方**：

    by1codegen.py   compile_ir      造 IR 的时候写进去
    by1boot.py      boot_ir         同上
    by1extdemo.py   演示            同上

三份写着同一个字符串，**谁也不知道另外两份存在**。
改一份忘一份的后果不是崩溃，是**两个入口造出声称不同版本的 IR** ——
而校验只认大版本，所以它不会报。

这和 eps 写死、`rope` 没读、`.get(不存在的键)` 是同一个家族：
**同一个意思写在多个地方，而没有任何东西保证它们一致。**

## 两个版本，不是一个

    IR_VERSION      **接口**的版本。IR 的字段增删、语义改变才动它。
                    读的一方只认大版本，不认识就拒。

    TOOL_VERSION    **实现**的版本。每次发布动。
                    IR 变了必须跟着动 —— 因为旧工具读不了新 IR。

把它们分开，是因为它们回答不同的问题：

    用的人问   "这份 IR 我读得了吗"     -> IR_VERSION
    报 bug 问  "你用的是哪一版工具"     -> TOOL_VERSION

## 什么时候该动哪个

    IR  加字段 / 删字段 / 改语义 / 改闭集        -> IR_VERSION 动
    加一个后端 / 修 bug / 加判卷人              -> 只动 TOOL_VERSION
    加一个 .by1 语法但 IR 不变                  -> 只动 TOOL_VERSION

**"加一个 .by1 语法"不动 IR_VERSION**，因为 `.by1` 是前端之一，
不是接口本身。这正是把 IR 规格化的意义。

用法:  python by1ver.py
"""
import sys

# ── 两个版本 ────────────────────────────────────────────────────────
IR_VERSION = "1.0"        # 接口
TOOL_VERSION = "0.9.0"    # 实现

# ── 版本历史。**只增不改。** ────────────────────────────────────────
# 记"为什么动"，不是"动了什么" —— 动了什么看 git 就够了。
HISTORY = [
    ("by1-ir 1.0", "IR 规格化：字段、必填、闭集、JSON 往返。"
                   "第一个被写下来的版本，所以是 1.0"),
    ("0.9.0", "第一个能端到端跑的版本：真产物 -> IR -> 三个后端 -> 官方实现。"
              "还不是 1.0，因为判卷人只覆盖了 8 个模型，"
              "而 C 后端的覆盖率还不如另外两个"),
]


def version_line():
    return "by1 %s  (by1-ir %s)" % (TOOL_VERSION, IR_VERSION)


def stamp():
    """给生成物盖章。**生成的每一份东西都该能追回是哪版生成的。**"""
    return "by1 %s / by1-ir %s" % (TOOL_VERSION, IR_VERSION)


def write_files():
    """把版本落到文件里，**给打包用**。

    ## 为什么不直接在 pyproject.toml 里写 `attr = "by1ver.TOOL_VERSION"`

    因为那要**导入** `by1ver` —— 而 setuptools 一旦开始导入，
    就会顺着摸到别的模块，而其中几个在模块级 `import transformers`
    （判卷人要和外部实现比，那是必需的）。
    在这台机器上 transformers 拖了一个坏掉的 `kernels.lockfile`，
    于是 `pip install -e .` 直接失败，报的还是一个跟本项目无关的错。

    **打包不该需要跑得动 torch。** 所以版本落成文件，打包只读文件。
    单一真相源仍然是 TOOL_VERSION —— 这里只是它的一个投影。
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    io_p = os.path.join(here, 'VERSION')
    with open(io_p, 'w', encoding='utf-8') as f:
        f.write(TOOL_VERSION + "\n")
    return io_p


def main():
    print()
    print("  " + version_line())
    print()
    print("  IR_VERSION   %s   **接口**的版本 —— IR 的字段增删、语义改变才动"
          % IR_VERSION)
    print("  TOOL_VERSION %s   **实现**的版本 —— 每次发布动；IR 变了必须跟着动"
          % TOOL_VERSION)
    print()
    print("  版本历史（只增不改）：")
    for v, why in HISTORY:
        print("    %-14s %s" % (v, why))
    print()
    print("  想写第四个后端：读 `python by1ir.py --spec` 出的规格，"
          "不需要学 .by1。")
    print()
    return 0


if __name__ == '__main__':
    if '--write' in sys.argv:
        print('  写到 %s' % write_files())
        sys.exit(0)
    sys.exit(main())
