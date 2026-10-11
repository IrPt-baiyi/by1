#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1shapes -- **把 `{名: null}` 的 refs 补成 F1（带 dtype + shape）。**

## 为什么需要它

`refs/*.tensors.json` 有三种写法（见 `by1verify.load_reference`）：

    F1  {"名": {"dtype": "BF16", "shape": [a, b]}}   —— 读得到
    F2  {"名": [a, b]}                                —— 读得到（形状，无 dtype）
    F3  {"名": null}                                  —— **读进来是空表**

而 `by1index.py` / `by1fetch.py`（**重建 refs 的那两条命令**）对**分片模型**
产出的正是 F3：它们先抓 `model.safetensors.index.json`，**成功就不再去读头**，
而 index 里只有名字、没有形状 —— **形状只在 safetensors 头里**。

后果不是报错，是**假失败**：`by1verify` 把 F3 读成空表 → 契约里每个张量都报
"缺失 N" → 看起来像契约写错了，真因是参考产物没读进来。
（实测：Qwen3-VL-2B 的 625 个全报缺失，就是这么来的。）

所以这个工具补上缺的那一环：**读全部分片的 safetensors 头**。

## 怎么读

一个 safetensors 文件的布局是：

    前 8 字节  = 头长度 n（小端 u64）
    接下来 n 字节 = JSON：{"张量名": {"dtype":…, "shape":[…], "data_offsets":[…]}}

所以**不需要下载权重**，一个 Range 请求就够。两步：先取 8 字节拿到 n，
再取 `bytes=0..(8+n)`。

## 为什么按分片走、不按 index 走

index 的 `weight_map` 只给"这个名字在哪个分片"，**形状还是得去那个分片里读**。
反过来只要"分片清单"，然后逐个读头、把整张表合并 —— 这样连 index 都不需要
（有些仓库的 index 缺失或不全）。

用法:
    python src/data/by1shapes.py                  # 补所有还是 F3 的
    python src/data/by1shapes.py --report         # 只报告
    python src/data/by1shapes.py --only Qwen3.5   # 只补名字里含它的
"""

import os as _os
import sys as _sys
# **引导：把自己上面那一层（`src/`）放上 sys.path。**
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import glob
import json
import os
import struct
import sys
import urllib.request

import by1io

REF = by1paths.REFS
#: 两个源都试 —— 实测有些仓库只在 HF 上、有些只在 ModelScope 上
#: （见 `by1index.py` 头注里那段"第一版只试了 ModelScope"）。
SOURCES = (
    ('HF', 'https://hf-mirror.com/%s/resolve/main/%s'),
    ('MS', 'https://www.modelscope.cn/api/v1/models/%s/repo?FilePath=%s'),
)


def get(url, timeout=30, rng=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'by1'})
    if rng:
        req.add_header('Range', rng)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def mid_of(base):
    """`Qwen__Qwen3-VL-2B-Thinking` -> `Qwen/Qwen3-VL-2B-Thinking`"""
    for sep in ('__', '_'):
        if sep in base:
            o, r = base.split(sep, 1)
            return '%s/%s' % (o, r)
    return None


def read_header(mid, shard):
    """读一个 safetensors 分片的头 -> {名: {"dtype":…, "shape":[…]}}。"""
    last = None
    for _where, tpl in SOURCES:
        url = tpl % (mid, shard)
        try:
            head = get(url, rng='bytes=0-7')
            if len(head) < 8:
                raise ValueError('头 8 字节都没拿到')
            n = struct.unpack('<Q', head[:8])[0]
            if n <= 0 or n > 200_000_000:
                raise ValueError('头长度不合理：%d' % n)
            blob = get(url, rng='bytes=0-%d' % (8 + n - 1))
            hdr = json.loads(blob[8:8 + n].decode('utf-8', 'replace'))
            return {k: {'dtype': v.get('dtype', ''),
                        'shape': [int(x) for x in v.get('shape', [])]}
                    for k, v in hdr.items() if k != '__metadata__'}
        except Exception as ex:
            last = ex
            continue
    raise ValueError('两个源都读不到 %s：%s' % (shard, last))


def shards_of(mid):
    """一个仓库有哪些分片。**先试 index，再试常见的单文件。**"""
    for _where, tpl in SOURCES:
        for fn in ('model.safetensors.index.json',
                   'pytorch_model.bin.index.json'):
            try:
                j = json.loads(get(tpl % (mid, fn), timeout=25)
                               .decode('utf-8', 'replace'))
                wm = j.get('weight_map') or {}
                if wm:
                    return sorted(set(wm.values()))
            except Exception:
                continue
    return ['model.safetensors']


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    report = '--report' in argv
    only = None
    if '--only' in argv:
        only = argv[argv.index('--only') + 1]

    files = sorted(glob.glob(os.path.join(REF, '*.tensors.json')))
    todo = []
    for f in files:
        base = os.path.basename(f)[:-len('.tensors.json')]
        if only and only not in base:
            continue
        d = by1io.read_json(f, encoding='utf-8')
        if not d:
            continue
        # 已经是 F1/F2（有形状）就跳过
        if any(isinstance(v, (dict, list)) for v in d.values()):
            continue
        todo.append((base, f, len(d)))

    print()
    if report:
        for base, _f, n in todo:
            print('    ? %-52s %6d 个张量缺形状' % (base[:52], n))
        print()
        print('  需要补的：%d 份' % len(todo))
        return 0

    if not todo:
        print('  没有需要补的（全部已有形状）')
        return 0

    ok, bad = 0, []
    for base, f, n in todo:
        mid = mid_of(base)
        if not mid:
            bad.append((base, '推不出 HF id'))
            continue
        try:
            shards = shards_of(mid)
        except Exception as ex:
            bad.append((base, '分片清单读不到：%s' % ex))
            continue
        merged = {}
        fail = None
        for sh in shards:
            try:
                merged.update(read_header(mid, sh))
            except Exception as ex:
                fail = '%s：%s' % (sh, ex)
                break
        if fail:
            bad.append((base, fail))
            continue
        # 只保留原表里有的名字（分片里可能多出别的，比如 optimizer 状态）
        old = by1io.read_json(f, encoding='utf-8')
        keep = {k: merged[k] for k in old if k in merged}
        miss = [k for k in old if k not in merged]
        if miss:
            bad.append((base, '%d 个名字在分片里找不到，例：%s'
                        % (len(miss), miss[0])))
            continue
        by1io.write_json(f, keep)
        ok += 1
        print('    ✓ %-52s %6d 个张量（%d 个分片）'
              % (base[:52], len(keep), len(shards)))

    print()
    print('  补了 %d 份，失败 %d 份' % (ok, len(bad)))
    for b, why in bad:
        print('    ✗ %-46s %s' % (b[:46], why))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
