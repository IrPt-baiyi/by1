#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1sitedata -- 站点数据必须等于从 IR 现算的那份（判据）。

页面上的数字如果只靠"当初导出时是对的"，它就只是又一本文档。
这条把 site/data 里的 .js 与 by1check/by1codegen 现算的结果逐字节比一遍；
漂移就红，重导一条命令（python tools/by1site.py）。

    python src/checks/by1sitedata.py       # 档 3 判卷人
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块可能在包内深处（src/by1/checks/…），固定层数会插错目录。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.exists(_os.path.join(_p, 'by1paths.py')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import pathlib
import sys

import by1paths

# 迁移期（docs/TEMP-rename-plan.md）：基准改问 by1paths（包内层数会变）。
ROOT = pathlib.Path(by1paths.ROOT)


def main():
    sys.path.insert(0, str(ROOT / "tools"))
    import by1site

    out = ROOT / "site" / "data"
    if not out.exists():
        print("  [跳过] 没有 site/data —— 站点还没导出（先跑 python tools/by1site.py）")
        return 2
    rc, msgs = by1site.check_site(ROOT, out)
    for m in msgs:
        print("  [%s] %s" % ("PASS" if rc == 0 else "FAIL", m))
    return rc


if __name__ == "__main__":
    sys.exit(main())
