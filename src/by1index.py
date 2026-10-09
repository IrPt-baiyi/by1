#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1index -- 抓张量名清单。**免费，而且这是"能不能描述"的真正判据。**

## 为什么 config 不够

`by1boot.classify()` 是**从张量名认机制**的：

    self_attn.q_proj   -> Attention
    linear_attn.in_proj -> Linear (GDN)
    mlp.gate_proj      -> FFN (门控)
    mlp.experts        -> MoE
    q_a_proj / kv_a_proj -> MLA

只给 config 的话，它认不出任何机制，**退化成 `Add+Norm+Raw`** ——
而我的第一版 triage 把这个当成了"✓ 已覆盖"。

**一个什么机制都没认出来的 IR，和"不支持"没区别，但报的是"✓"。**
那正是这个项目一直在防的那种谎。

## 抓什么

`model.safetensors.index.json`（几十 KB，列出了**每一个张量名**）
—— 比抓 safetensors 头便宜得多，而且信息一样。

有些单文件模型没有 index，那就退回抓 safetensors 的**头 1 MB**
（头部有完整的张量目录）。

用法:  python by1index.py            # 抓所有缺的
       python by1index.py --report   # 只报告
"""
import glob
import json
import os
import struct
import sys
import urllib.request
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))
REF = by1paths.REFS


def get(url, timeout=25, nbytes=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'by1'})
    if nbytes:
        req.add_header('Range', 'bytes=0-%d' % nbytes)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ms_ids():
    """refs/ 里每份 config -> ModelScope 的 id。"""
    out = {}
    for f in glob.glob(os.path.join(REF, '*.config.json')):
        base = os.path.basename(f)[:-len('.config.json')]
        # refs/ 的约定是 <owner>__<Repo>  或  <owner>_<Repo>
        for sep in ('__', '_'):
            if sep in base:
                o, r = base.split(sep, 1)
                out[base] = '%s/%s' % (o, r)
                break
    return out


def tensors_from_index(txt):
    j = json.loads(txt)
    return {k: list(v) for k, v in (j.get('weight_map') or {}).items()}, \
        {k: list(v) for k, v in (j.get('weight_map') or {}).items()}


def tensors_from_header(mid, where, shard='model.safetensors'):
    """没有 index 的单文件模型：抓 safetensors 的头。"""
    url = ('https://www.modelscope.cn/api/v1/models/%s/repo?FilePath=%s'
           % (mid, shard) if where == 'MS' else
           'https://hf-mirror.com/%s/resolve/main/%s' % (mid, shard))
    b = get(url, nbytes=2_000_000)
    if len(b) < 9:
        raise ValueError('太短')
    n = struct.unpack('<Q', b[:8])[0]
    hdr = json.loads(b[8:8 + n].decode('utf-8', 'replace'))
    return {k: list(v['shape']) for k, v in hdr.items() if k != '__metadata__'}


def main():
    report_only = '--report' in sys.argv
    ids = ms_ids()
    have = {os.path.basename(f)[:-len('.tensors.json')]
            for f in glob.glob(os.path.join(REF, '*.tensors.json'))}

    print()
    ok, skip, bad = 0, 0, []
    for base, mid in sorted(ids.items()):
        # 已经有张量清单的（老模型）跳过
        short = base.split('__')[-1].split('_', 1)[-1]
        if base in have or short in have:
            skip += 1
            continue
        if report_only:
            print('    ? %-48s 缺张量清单' % base[:48])
            continue

        got = None
        # ① index 文件（最省）
        for where in ('MS', 'HF'):
            for fn in ('model.safetensors.index.json',
                       'pytorch_model.bin.index.json'):
                try:
                    if where == 'MS':
                        u = ('https://www.modelscope.cn/api/v1/models/%s/repo'
                             '?FilePath=%s' % (mid, fn))
                    else:
                        u = ('https://hf-mirror.com/%s/resolve/main/%s'
                             % (mid, fn))
                    t = get(u).decode('utf-8', 'replace')
                    j = json.loads(t)
                    if j.get('weight_map'):
                        got = {k: None for k in j['weight_map']}
                        break
                except Exception:
                    continue
            if got:
                break
        # ② safetensors 头（单文件模型）—— **两个源都要试**
        #
        # 第一版只试了 ModelScope，于是 10 个"没抓到" ✗ ——
        # 而它们只是**在 HF 上、不在 MS 上**（gpt-oss-20b / mamba /
        # GLM-4.7 / poolside / stepfun 那几个）。
        # 报错是"没抓到"，看起来像"这些模型有问题"。
        if not got:
            for where in ('MS', 'HF'):
                for shard in ('model.safetensors', 'pytorch_model.bin'):
                    try:
                        got = tensors_from_header(mid, where, shard)
                        break
                    except Exception:
                        continue
                if got:
                    break
        if got:
            p = os.path.join(REF, base + '.tensors.json')
            by1io.write_text(p, json.dumps(got, ensure_ascii=False, indent=0))
            print('    ✓ %-48s %5d 个张量' % (base[:48], len(got)))
            ok += 1
        else:
            bad.append(base)

    print()
    print('  抓到 %d 个，跳过 %d 个（已有），没抓到 %d 个'
          % (ok, skip, len(bad)))
    for b in bad:
        print('    ✗ %s' % b)
    print()
    print('  **注意**：这里只要张量名和形状。')
    print('  index 文件只是名字；形状要么在 index 之外，要么得读 safetensors 头。')
    print('  名字够 `classify()` 认机制了，形状不够的地方会报出来。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
