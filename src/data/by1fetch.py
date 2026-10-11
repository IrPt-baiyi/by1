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
    张量表             index 拿**分片清单**，再逐个读分片的头 -> dtype + shape

## ⚠ 这里原来有一句错话（**就是这一版的修复**）

    张量名   model.safetensors.index.json（几十 KB，只有名字）
             —— 比抓 safetensors 头便宜得多，而且**信息一样**

**"信息一样"是错的。** index 里只有名字，**没有形状，也没有 dtype**。
于是按它写出来的 refs 是 `{名: null}`，而 `by1verify.load_reference` 读进去
是**空表** —— 表现是"契约里每个张量都缺失"（Qwen3-VL-2B 的 625 个全报缺失），
看起来像契约写错了，真因是参考产物根本没读进来。

**后果比"慢"严重**：`refs/` 删掉之后重建出来的产物**验不了** ——
而"删掉 95% 也能重建"这句话，本来就该包含"重建出来的能验"。

现在的顺序（① 仍然便宜，② 才是贵的）：

    ① `model.safetensors.index.json` —— 一次拿到**全部张量名 + 分片清单**
    ② 逐个分片读 safetensors 的**头**（Range 请求，先 1 MB）-> dtype + shape

代价：分片模型要读**全部**分片的头（235B 是 96 个）——
比只读一个 index 贵得多，但那是拿到形状的**唯一**办法。

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
        req.add_header('Range', 'bytes=0-%d' % (nbytes - 1))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # **给了 `nbytes` 就最多只读那么多。** 服务器**不认 Range** 时会
        # 把整个文件发过来 —— 而分片是 GB 级，`read()` 会当场吃光内存。
        # 多留 1 个字节，用来判断"要的那段是不是被截断了"。
        return r.read(nbytes + 1) if nbytes else r.read()


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


#: safetensors 头先抓这么多。绝大多数分片的头远小于它；不够再按精确长度重抓。
HEAD_BYTES = 1 << 20


def shards_from_index(hfid):
    """index -> `(weight_map, 分片清单)`。**这是唯一一次就能拿到分片清单的地方。**"""
    for fn in ('model.safetensors.index.json',
               'pytorch_model.bin.index.json'):
        for url in (MS % (hfid, fn), HF % (hfid, fn)):
            try:
                j = json.loads(get(url, timeout=30).decode('utf-8', 'replace'))
                wm = j.get('weight_map') or {}
                if wm:
                    return wm, sorted(set(wm.values()))
            except Exception:
                continue
    return None, None


def shard_header(hfid, shard):
    """一个分片的 safetensors 头 -> `{名: {'dtype', 'shape'}}`。抓不到返回 None。

    头格式：前 8 字节 = 头长度 n（小端 u64），随后 n 字节 JSON。
    """
    # **HF 在前。** 它认 Range（实测过）；ModelScope 的 `?FilePath=` 那边
    # 没验过认不认 —— 万一不认会把整个分片发过来。`get` 里已经加了上限，
    # 但顺序上还是先走验过的那条。
    for url in (HF % (hfid, shard), MS % (hfid, shard)):
        try:
            b = get(url, nbytes=HEAD_BYTES, timeout=60)
            if len(b) < 9:
                continue
            n = struct.unpack('<Q', b[:8])[0]
            if 8 + n > len(b):              # 1 MB 不够，按精确长度重抓
                b = get(url, nbytes=8 + n, timeout=90)
                if 8 + n > len(b):
                    continue
            hdr = json.loads(b[8:8 + n].decode('utf-8', 'replace'))
            return {k: {'dtype': v.get('dtype'),
                        'shape': list(v.get('shape') or [])}
                    for k, v in hdr.items() if k != '__metadata__'}
        except Exception:
            continue
    return None


def fetch_tensors(hfid):
    """张量清单。**要形状，不要只有名字**（见文件头那段"错话"）。

    ① index -> 分片清单（顺带拿到全部名字）
    ② 逐个分片读头 -> dtype + shape
    没有 index（单文件模型）就直接读 `model.safetensors` 的头。

    **任何一片读不到就整个放弃**（返回 `None`）—— 残缺的表写下去就是拿它
    **覆盖**掉好的那份，是净损失。
    """
    wm, shards = shards_from_index(hfid)
    if shards:
        out = {}
        for s in shards:
            h = shard_header(hfid, s)
            if h is None:
                return None, None
            out.update(h)
        # index 说有的名字，头里必须也有 —— 对不上说明抓漏了分片。
        if wm and (set(wm) - set(out)):
            return None, None
        return out, 'header×%d' % len(shards)
    # 单文件模型：没有 index，直接读那一个文件。
    for shard in ('model.safetensors', 'pytorch_model.bin'):
        h = shard_header(hfid, shard)
        if h:
            return h, 'header'
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
