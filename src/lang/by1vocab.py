#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1vocab -- **真实数据的词汇表。模式从这里来，不从想象来。**

## 为什么

`by1boot` 里的识别模式（`TOWER_PAT` / `AUX_PAT` / `classify`）
是**先写再验**的：我凭"想象中的模型仓库长什么样"写了一批词，
其中 16 个在 369966 个真实张量名里**一次都没出现**。

    塔名死词：image_encoder · multi_modal_projector · img_
              sound · speech · whisper · video · temporal
    栈名死词：nextn · multi_token · mtp_layers · draft · eagle · medusa

**这些词不一定错**（`nextn` 是 DeepSeek 的 MTP 叫法，真的存在）——
**但它们从没被验证过，而我把它们和验证过的写在了一起、用同样的语气。**

## 正确的方向

**反过来**：先看真实数据里有什么，再决定写什么模式。

这个脚本把 34 份张量清单里所有名字切成"段"，统计每一段
出现在多少个模型里。**那张表才是写模式的依据。**

而它同时给出一条判据：

    出现在 >= 3 个模型里的段  ->  值得写进模式
    只出现在 1 个模型里的段    ->  是那个模型的私有命名，别写死

用法:  python by1vocab.py [--all] [--min 3]
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
import glob
import os
import re
import sys
from collections import defaultdict
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def main():
    show_all = '--all' in sys.argv
    mn = 3
    if '--min' in sys.argv:
        mn = int(sys.argv[sys.argv.index('--min') + 1])

    alls = {}
    for f in sorted(glob.glob(by1paths.ref('*.tensors.json'))):
        try:
            alls[os.path.basename(f).replace('.tensors.json', '')] = \
                list(by1io.read_json(f, encoding='utf-8'))
        except Exception:
            continue

    # 每个"段"出现在哪些模型里
    seg_models = defaultdict(set)
    seg_count = defaultdict(int)
    for m, names in alls.items():
        for n in names:
            for seg in re.split(r'[._]', n):
                seg = seg.strip()
                if not seg or seg.isdigit():
                    continue
                seg_models[seg].add(m)
                seg_count[seg] += 1

    print()
    print('=' * 78)
    print('  真实数据的词汇表 —— **写模式该看这张表**')
    print('=' * 78)
    print()
    print('  底数：%d 份张量清单，%d 个物理名，%d 个不同的段'
          % (len(alls), sum(len(v) for v in alls.values()), len(seg_models)))
    print()

    rows = sorted(seg_models.items(), key=lambda kv: -len(kv[1]))

    print('  ── 出现在 >= %d 个模型里的段（值得写进模式）' % mn)
    print('     %-26s %6s %8s' % ('段', '模型数', '出现次数'))
    for seg, ms in rows:
        if len(ms) >= mn:
            print('     %-26s %6d %8d' % (seg[:26], len(ms), seg_count[seg]))
    print()

    print('  ── **只出现在 1 个模型里的段**（那个模型的私有命名）')
    print('     写死它们就等于"只支持那一个模型"')
    solo = [(s, ms) for s, ms in rows if len(ms) == 1]
    print('     共 %d 个' % len(solo))
    if show_all:
        for seg, _ms in sorted(solo, key=lambda kv: -seg_count[kv[0]])[:60]:
            print('     %-26s %8d 次  （%s）'
                  % (seg[:26], seg_count[seg],
                     list(_ms)[0][:30]))
    else:
        for seg, ms in sorted(solo, key=lambda kv: -seg_count[kv[0]])[:16]:
            print('     %-26s %8d 次' % (seg[:26], seg_count[seg]))
        print('     …（--all 看全部）')
    print()

    # ── 拿 by1boot 现在写的模式去对 ──────────────────────────────
    import importlib.util
    s = importlib.util.spec_from_file_location('by1boot',
                                               by1paths.tool('by1boot.py'))
    boot = importlib.util.module_from_spec(s)
    s.loader.exec_module(boot)

    known = set(seg_models)
    print('  ── by1boot 里写的模式，逐分支对这张表')
    print()
    for attr in ('TOWER_PAT', 'AUX_PAT'):
        for item in getattr(boot, attr, []) or []:
            name, pat = item[0], item[1]
            for b in pat.split('|'):
                b2 = re.sub(r'[^A-Za-z0-9_]', '', b)
                if not b2:
                    continue
                # **多词模式要用整串测，不能查段表。**
                # 第一版查 `seg_models[b2]` —— 而段是把名字按 `.` 和 `_`
                # 切开的，于是 `patch_embed` 变成 `patch` + `embed` 两段，
                # 整串当然查不到，报"数据里没有"。
                # 而它是活的（`patch` 18 个模型、`embed` 29 个）。
                n = sum(1 for names in alls.values()
                        if any(b2 in x for x in names))
                mark = ('**数据里没有**' if n == 0
                        else '%d 个模型有' % n)
                print('     %-10s %-24s %s' % (name, b2, mark))
    print()
    print('  **"数据里没有"不代表错** —— 可能是别的家族的叫法。')
    print('  但它代表这条**从没被验证过**。')
    print()
    print('  判据：>= %d 个模型的段才值得写死；1 个模型的段是私有命名。'
          % mn)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
