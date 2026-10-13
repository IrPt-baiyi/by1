#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 -- 用一份描述，写一个语言模型。

  by1 check hello.by1                       检查描述
  by1 run hello.by1 --seq 8                  随机权重前向，输出 logits 形状
  by1 build hello.by1 -o model.py            导出可训练的 PyTorch 模型源码
  by1 train hello.by1 --text 1.md            搭模型 · 训练 · 让它写字
  by1 verify a.by1 b.config.json             和真实模型的 config / 权重对拍
  by1 oracle a.by1                           跑出真实行为，和状态声明对拍
  by1 --version                             查看版本

未安装时，用 python src/lang/by1.py 替换上面的 by1。

描述写在 .by1 文件里，改数字就能改模型。从 hello.by1 开始看。
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
# 加了它，`import by1paths` 才找得到；而 `by1paths` 在 import 时
# 会把 `src/` 和每个子目录都放上 sys.path —— 于是 `import by1check`
# 这种裸名 import 照旧能用。**这两行是生成的，别手改**
# （判据在 `by1paths.check_boot()`）。
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import importlib.util
import os
import sys

import by1io      # noqa: F401  —— import 即把 stdout 钉成 UTF-8
import by1skip

HERE = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "train": ("by1train", "按描述搭模型、训练、生成"),
    "build": ("by1codegen", "按描述生成模型源码"),
    "check": ("by1check", "检查描述"),
    "run": ("by1exec", "用随机权重执行前向（需要 numpy）"),
    "verify": ("by1verify", "和官方产物对拍"),
    "oracle": ("by1oracle", "执行神谕"),
}


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, by1paths.tool(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("--version", "version"):
        return load("by1ver").main()
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("  可用命令：")
        for k, (_m, d) in COMMANDS.items():
            print(f"    {k:<8}{d}")
        print()
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"不认识的命令 '{cmd}'。可用：{', '.join(COMMANDS)}")
        # 命令名不认识 -> by1skip.CALLER
        return by1skip.CALLER
    try:
        module_name = COMMANDS[cmd][0]
        if cmd in ("run", "train") and len(argv) > 1 and not argv[1].startswith("-"):
            checker = load("by1check")
            rep, info = checker.check(argv[1])
            if rep.count(checker.E):
                return checker.run_one(argv[1])
            if info.get('backend') == 'transformers':
                module_name = 'by1full'
        mod = load(module_name)
        if cmd == 'train' and module_name == 'by1full':
            return mod.train_main(argv[1:])
        return mod.main(argv[1:])
    except FileNotFoundError as ex:
        print(f"找不到文件：{ex.filename}")
        return by1skip.CALLER
    except ModuleNotFoundError as ex:
        if ex.name not in ("numpy", "torch", "transformers", "safetensors"):
            raise
        print(f"{cmd} 需要 {ex.name}：python -m pip install {ex.name}")
        return by1skip.CALLER
    except ValueError as ex:
        print(f"[FAIL] {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
