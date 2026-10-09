#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1fetch -- **一条命令，把 refs/ 全部重抓回来。**

## 这是"删掉 95% 的数据也能迅速重建"的那条命令

仓库里的数据是：

    refs/*.json              25.8 MB   <- **99%**，从 HF / ModelScope 抓的
    *.by1 + models.tsv        0.2 MB   <- 0.8%，手写的知识
    派生（models.md 等）               <- 生成的

**删掉 95%，删掉的就是 refs/。** 而这个脚本把它抓回来。

## 它从哪起步

**两个种子，都是手写的、都不依赖 `refs/`：**

    models.tsv          14 个模型（有 `.by1`，进护栏）
    refs/SOURCES.tsv    21 个候选（只有产物，还没写 `.by1`）

**为什么要两个。** 只用 `models.tsv` 的话，`refs/` 里那 41 个文件
**没有出处** —— 删掉它们就再也抓不回来，而"删掉 refs/ 也能重建"
这句话看起来仍然成立（见 `SOURCES.tsv` 的头注）。

（`*.by1` 也能提供 `by1-repo`，但 `.by1` 可能也一起被删了 ——
所以种子必须是最小的、手写的那一个文件。）

## 规则

    refs/<owner>__<Repo>.config.json
    refs/<owner>__<Repo>.tensors.json

和 `by1refs.py` 完全一致 —— **路径规则只此一处**。

## 抓法

    config.json       ModelScope 优先（国内快），HF 兜底
    张量名             model.safetensors.index.json（几十 KB，只有名字）
                      没有 index 就抓 safetensors 的**头 2 MB**（有名字和形状）

用法:
    python by1fetch.py              抓所有缺的
    python by1fetch.py --all        全部重抓（覆盖已有的）
    python by1fetch.py --only Qwen  只抓名字里带 Qwen 的
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
import json
import os
import struct
import sys
import time
import urllib.request
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
REF = by1paths.REFS

MS = 'https://www.modelscope.cn/api/v1/models/%s/repo?FilePath=%s'
HF = 'https://hf-mirror.com/%s/resolve/main/%s'


def get(url, nbytes=None, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'by1'})
    if nbytes:
        req.add_header('Range', 'bytes=0-%d' % nbytes)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def seed():
    """种子：`models.tsv` + `refs/SOURCES.tsv`。返回 `(列表, 额外的个数)`。

    列表里是 `(显示名, 短名, HF id, 来源)`，**来源只影响打印** ——
    抓法一个字不变：同一个 HF id、同两条规则路径。
    """
    out = []
    for line in by1io.iter_lines(by1paths.root('models.tsv'), encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.lstrip().startswith('#'):
            continue
        p = line.split('\t')
        if len(p) >= 3 and '/' in p[2]:
            out.append((p[0], p[1], p[2], '有 .by1'))
    extra = 0
    src = by1paths.ref('SOURCES.tsv')
    if os.path.exists(src):
        for line in by1io.iter_lines(src, encoding='utf-8'):
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            p = line.split('\t')
            hfid = p[0].strip()
            if '/' in hfid:
                out.append((hfid, '', hfid, '探索源'))
                extra += 1
    return out, extra


def base_of(hfid):
    return hfid.replace('/', '__')


def fetch_config(hfid):
    """config.json。ModelScope 优先。"""
    for url in (MS % (hfid, 'config.json'), HF % (hfid, 'config.json')):
        try:
            b = get(url, timeout=25)
            json.loads(b.decode('utf-8', 'replace'))
            return b
        except Exception:
            continue
    return None


def fetch_tensors(hfid):
    """张量清单。index 优先（小），没有就抓 safetensors 头。"""
    # ① index：只有名字，但名字就够 classify() 认机制
    for fn in ('model.safetensors.index.json',
               'pytorch_model.bin.index.json'):
        for url in (MS % (hfid, fn), HF % (hfid, fn)):
            try:
                j = json.loads(get(url, timeout=30).decode('utf-8', 'replace'))
                wm = j.get('weight_map') or {}
                if wm:
                    return {k: None for k in wm}, 'index'
            except Exception:
                continue
    # ② safetensors 头：**有名字也有形状**，2 MB 够
    for url in (MS % (hfid, 'model.safetensors'),
                HF % (hfid, 'model.safetensors')):
        try:
            b = get(url, nbytes=2_000_000, timeout=40)
            if len(b) < 9:
                continue
            n = struct.unpack('<Q', b[:8])[0]
            hdr = json.loads(b[8:8 + n].decode('utf-8', 'replace'))
            return ({k: list(v['shape']) for k, v in hdr.items()
                     if k != '__metadata__'}, 'header')
        except Exception:
            continue
    return None, None


def main():
    allf = '--all' in sys.argv
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1]

    os.makedirs(REF, exist_ok=True)
    seeds, n_extra = seed()
    print()
    print('  种子：models.tsv %d 个模型（有 .by1）' % (len(seeds) - n_extra))
    print('        refs/SOURCES.tsv %d 个探索源（没有 .by1）' % n_extra)
    print('  重抓模式：%s' % ('全部覆盖' if allf else '只补缺的'))
    print()
    print('  %-46s %-10s %-8s %s' % ('模型', 'config', '张量', '耗时'))
    print('  ' + '-' * 80)

    t0 = time.time()
    n_cfg = n_ten = n_skip = n_bad = 0
    for 长名, 短名, hfid, 来源 in seeds:
        if only and only.lower() not in hfid.lower():
            continue
        b = base_of(hfid)
        pc = os.path.join(REF, b + '.config.json')
        pt = os.path.join(REF, b + '.tensors.json')
        need_c = allf or not os.path.exists(pc)
        need_t = allf or not os.path.exists(pt)
        if not need_c and not need_t:
            n_skip += 1
            continue
        s = time.time()
        cstat = tstat = '（有）'
        if need_c:
            bts = fetch_config(hfid)
            if bts:
                # **`with` 关句柄。** 原来是 `io.open(...).write(...)` ——
                # CPython 上引用计数会立刻关，所以不是 bug；但这是
                # py1io 存在的理由之一（"每一种意图各有一个名字"）。
                by1io.write_bytes(pc, bts)
                n_cfg += 1
                cstat = '%d KB' % (len(bts) // 1024)
            else:
                cstat = '**抓不到**'
                n_bad += 1
        if need_t:
            d, how = fetch_tensors(hfid)
            if d:
                by1io.write_text(pt, json.dumps(d, ensure_ascii=False, indent=0))
                n_ten += 1
                tstat = '%d 个(%s)' % (len(d), how)
            else:
                tstat = '**抓不到**'
                n_bad += 1
        print('  %-46s %-10s %-8s %.1fs'
              % (长名[:46], cstat[:10], tstat[:8], time.time() - s))

    dt = time.time() - t0
    print('  ' + '-' * 80)
    print('  config %d 个 · 张量清单 %d 个 · 跳过 %d 个 · **失败 %d 个**'
          % (n_cfg, n_ten, n_skip, n_bad))
    print()
    print('  **%.0f 秒**（%.1f 分钟）' % (dt, dt / 60))
    print()
    print('  重建完还要跑：')
    print('    python by1cmp.py --md > models.md      # 派生')
    print('    python by1ir.py --spec > ir-spec.md    # 派生')
    print('    python by1ver.py --write               # 版本')
    print('    python by1all.py                       # 全量验证')
    print()
    print('  **而 `*.by1` 和 `models.tsv` 抓不回来** —— 它们是手写的知识。')
    print('  删掉它们之后，能重建的是"它们描述的那部分推导"（by1boot），')
    print('  推不出来的（`qk_norm` 这种产物里没有的属性）才是真判断。')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
