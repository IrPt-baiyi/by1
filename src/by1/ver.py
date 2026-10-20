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

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401
import sys
import by1paths

# ── 两个版本 ────────────────────────────────────────────────────────
IR_VERSION = "1.4"          # 接口（1.4：Attention 执行语义 causal / rope_pos + 顶层 mrope_section）
TOOL_VERSION = "1.0.0-rc4"  # 实现

# ── 版本历史。**只增不改。** ────────────────────────────────────────
# 记"为什么动"，不是"动了什么" —— 动了什么看 git 就够了。
HISTORY = [
    ("by1-ir 1.0", "IR 规格化：字段、必填、闭集、JSON 往返。"
                   "第一个被写下来的版本，所以是 1.0"),
    ("0.9.0", "第一个能端到端跑的版本：真产物 -> IR -> 三个后端 -> 官方实现。"
              "还不是 1.0，因为判卷人只覆盖了 8 个模型，"
              "而 C 后端的覆盖率还不如另外两个"),
    ("0.9.1", "描述↔计算闭环：IR 形状/类型静态推导（换宽度进 gaps 不进 errs）、"
              "张量形状表收敛到 by1ir.params_of 单一真相源、变异反例库"
              "（by1mutate）进门禁 —— 三件都有判卷人，每条规则反例必红"),
    ("1.0.0-rc0", "地基版（rc 车厢 0）：原语图层（IR→规范化原语图，"
              "四后端机械化翻译，w.bin 布局独立进 by1ir.weight_plan）+ "
              "数值规格与精度预算（PRIMS 41 条形状/dtype/边界 + 阈值全表）"
              "+ 导出命名反推（refs→emit 模板，机器生成率基线 41%，只升不降）"),
    ("by1-ir 1.1", "栈/塔登记表进 IR（STACKS：name/modality/width/层范围）。"
                   "源语言的 stack/MODAL 早就有，但编译成 IR 后栈信息全丢，"
                   "护栏没法对塔说话。**加法变更**：stacks 可缺省 = 单栈文本，"
                   "旧描述一个字不改照样合法；读的一方仍只认大版本"),
    ("1.0.0-rc1",      "收尾版（rc 车厢 1）：大目标版路线图冻结 + 产品边界/语言合同两份可执行规范（product-definition / by1-language-spec + by1langspec 三方一致判卷人）+ GGUF 产物线判据全过（290/290 逐字节 + logits 逐位）+ 音频/视觉三边界数字收口 + 社区版 Step3.7 移除（缺口清零）。档 3 全绿 100 项、档 5 全绿 129 项"),
    ("by1-ir 1.2", "多分支形状：分支登记表升格（stacks 加 role）+ 显式接线（wiring）"
                   " + 跨分支参数绑定（shared）。**加法变更**：三个字段都可缺省 = "
                   "串行单栈，1.1 文件一个字不改照样合法。多分支的 IR 必须答得出"
                   "「这个节点在哪个分支上（stacks 层段）、接线从哪来到哪去（wiring）」"
                   " —— 不拍平判据是**可回答**，不是「不能有平表」"),
    ("1.0.0-rc2", "A 弧线（by1-ir 1.2 多分支）：G-A 门按真实构造分（单栈 complete-* "
                  "放行进 IR 路）+ ③ 最小里程碑双判据进档 5（装配模块 2.9e-06 / "
                  "exec 后端 4.3e-06 vs transformers 官方前向）+ 70 份多分支普查"
                  "（含 complete-* 逐份表）+ 1.2 schema/校验/规格/反例库"),
    ("by1-ir 1.3", "算子闭集扩大：PatchEmbed / PosEmbed / Merger（视觉入口/出口）。"
                   "旧工具见到新 kind 会被闭集校验拒 —— 所以版本动。形状与数学均对"
                   "官方 transformers.models.qwen3_5 实测（9/9 同形、算子级 1e-07 内）"),
    ("1.0.0-rc3", "B 弧线：三算子落 IR + 视觉执行语义清单（打包双向注意力与双轴 "
                 "mrope 是仅剩的两个语义缺口）+ 耗时判据真回归修复"
                 "（by1docs 877 次 spawn → 批量问 git）+ 预算按新判据集再基线"),
    ("by1-ir 1.4", "Attention 执行语义进字段：causal（视觉塔 = 每图一段双向）、"
                   "rope_pos（seq / mrope / grid_hw 三种位置来源）+ 顶层 mrope_section。"
                   "**加字段**，缺省 = 文本口径（1.3 的 IR 一字不改）；读到新字段的旧工具"
                   "会被未知字段检查拒 —— 所以版本动"),
    ("1.0.0-rc4", "B 弧线两堵墙进 IR：视觉塔（打包双向 + 双轴 mrope）与主干图像 token 的"
                 "mrope 都有了字段名分；生产者按 vision{} 声明填充（不声明不猜），"
                 "公式逐字钉在 docs/branch-exec-plan.md 〇之三"),
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
    # **VERSION 在仓库根，不在代码旁边。**
    # 写进 `src/VERSION` 就是造出第二个版本文件 —— 而"同一个意思写在两个
    # 地方、没有任何东西保证它们一致"正是这个文件在 docstring 里反对的事。
    io_p = os.path.join(by1paths.ROOT, 'VERSION')
    # **行尾钉成 `\n`，别让平台决定。**
    # `open(..., 'w')` 在 Windows 上把 `\n` 翻成 `\r\n` —— 于是同一个
    # `TOOL_VERSION`，在 Windows 上生成 7 字节、在 Linux 上 6 字节，
    # 而 git 那边看到的是"VERSION 被改了"（整文件一行 diff）。
    # 这是"同一个意思，两个平台两种字节"，而它唯一的症状是
    # **在另一台机器上 `git status` 变脏**。
    # 用 `newline=''` = 不做任何翻译，写什么就是什么。
    # （配套：仓库根有 `.gitattributes`，`* text=auto eol=lf`。）
    with open(io_p, 'w', encoding='utf-8', newline='') as f:
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
