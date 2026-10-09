#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
by1 -- 用一份描述，写一个语言模型。

  python by1.py train hello.by1              搭模型 · 训练 · 让它写字
  python by1.py build hello.by1 -o model.py  导出模型源码（可以直接看）
  python by1.py check hello.by1              只检查描述写得对不对
  python by1.py verify a.by1 b.config.json   和真实模型的 config / 权重对拍
  python by1.py oracle a.by1                 跑出真实行为，和状态声明对拍

描述写在 .by1 文件里，改数字就能改模型。从 hello.by1 开始看。
"""

import importlib.util
import os
import sys

import by1io      # noqa: F401  —— import 即把 stdout 钉成 UTF-8

HERE = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "train": ("by1train", "按描述搭模型、训练、生成"),
    "build": ("by1codegen", "按描述生成模型源码"),
    "check": ("by1check", "检查描述"),
    "verify": ("by1verify", "和官方产物对拍"),
    "oracle": ("by1oracle", "执行神谕"),
}


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        print("  可用命令：")
        for k, (_m, d) in COMMANDS.items():
            print(f"    {k:<8}{d}")
        print()
        return 0
    cmd = argv[1]
    if cmd not in COMMANDS:
        print(f"不认识的命令 '{cmd}'。可用：{', '.join(COMMANDS)}")
        return 2
    mod = load(COMMANDS[cmd][0])
    return mod.main(argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
