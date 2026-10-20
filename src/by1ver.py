#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迁移期 shim（计划见 docs/TEMP-rename-plan.md）—— 真身在 by1.ver。

**兼容期上限 2 个阶段**：拆完 shim 就删。现在它只做两件事：
  ① 把真身的名字**连下划线的一起**转到本模块（别的模块可能用私有名）；
  ② 当脚本跑时转发到真身的 __main__（判卷人都是当脚本执行的）。
"""
import os as _os
import sys as _sys
# 引导（生成的，别手改）：见 by1paths.check_boot
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401
import importlib as _il  # noqa: E402

_m = _il.import_module('by1.ver')
globals().update({_k: _v for _k, _v in vars(_m).items()
                  if not _k.startswith('__')})

if __name__ == '__main__':
    import runpy
    # run_path 而不是 run_module：模块已经被本 shim import 过，
    # run_module 会报 'found in sys.modules' 的 RuntimeWarning。
    runpy.run_path(_m.__file__, run_name='__main__')
