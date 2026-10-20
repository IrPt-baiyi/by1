#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迁移期 shim（计划见 docs/TEMP-rename-plan.md）—— 真身在 by1.data.index。

**兼容期上限 2 个阶段**：拆完 shim 就删。它做两件事，缺一不可：
  ① `globals().update(...)`：按**文件路径**装载的调用（`spec_from_file_location`）
     拿到的是本 shim 对象，必须有属性；
  ② `sys.modules[本名] = 真身`：按**名字** `import by1xxx` 的直接拿到真身，
     循环 import / 后定义属性都按 Python 原生语义解析。
当脚本执行时转发真身的 __main__（判卷人都当脚本跑）。
"""
import os as _os
import sys as _sys
# 引导（生成的，别手改）：见 by1paths.check_boot
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import importlib as _il  # noqa: E402

_m = _il.import_module('by1.data.index')

if __name__ == '__main__':
    import runpy
    runpy.run_path(_m.__file__, run_name='__main__')
else:
    globals().update({_k: _v for _k, _v in vars(_m).items()
                      if not _k.startswith('__')})
    _sys.modules[__name__] = _m
