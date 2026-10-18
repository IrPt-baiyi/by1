#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1sitedata -- 站点数据必须等于从 IR 现算的那份（判据）。

页面上的数字如果只靠"当初导出时是对的"，它就只是又一本文档。
这条把 site/data 里的 .js 与 by1check/by1codegen 现算的结果逐字节比一遍；
漂移就红，重导一条命令（python tools/by1site.py）。

    python src/checks/by1sitedata.py       # 档 3 判卷人
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


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
