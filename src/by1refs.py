#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1refs -- **refs/ 的路径规则，只此一处。**

## 规则

    refs/<owner>__<Repo>.config.json
    refs/<owner>__<Repo>.tensors.json
    refs/<owner>__<Repo>.gguf-tensors.json     （可选，量化那边用）

而 `<owner>__<Repo>` **从 `.by1` 头部的 `# by1-repo:` 那一行来**：

    # by1-model: Step-3.7-Flash
    # by1-repo:  stepfun-ai/Step-3.7-Flash
    # by1-short: step37-official

## 为什么要单写一个

在这之前，refs 的路径**写死在四个文件里**（by1all 28 处、by1bootir 5 处、
by1oracle、by1triage）—— 而它们会**各自过期**：

    我把 `refs/minimind-3.tensors.json` 改名成
    `refs/jingyaogong__minimind-3.tensors.json`，
    改名脚本知道这件事，**但四个文件里的 28 处字符串不知道**。
    症状是 `FileNotFoundError: refs/minimind-3.tensors.json` ——
    **看起来像文件丢了，其实是引用没跟上。**

规则写在一处，就不会有"谁忘了改"。
"""
import os
import re
import by1io
import by1paths

HERE = os.path.dirname(os.path.abspath(__file__))

# **`refs/` 在仓库根，不在代码旁边。**
# 这个文件只管 refs/ **里面的命名规则**；"它在哪个目录"是布局的事，
# 问 `by1paths`（`HERE` 搬家之后不再是仓库根了）。
REF = by1paths.REFS

_REPO = re.compile(r'^#\s*by1-repo:\s*(\S+)', re.M)


def repo_of(by1file):
    """从 .by1 头部读出 `owner/Repo`。读不到返回 None。"""
    # **裸名 -> models/<名>。** 模型不在仓库根了。
    p = by1paths.model(by1file)
    if not os.path.exists(p):
        return None
    try:
        head = by1io.head_text(p)
    except Exception:
        return None
    m = _REPO.search(head)
    return m.group(1) if m else None


def base_of(by1file):
    """`owner/Repo` -> `owner__Repo`（refs/ 里的基名）。"""
    r = repo_of(by1file)
    return r.replace('/', '__') if r else None


def paths(by1file, kind='config'):
    """refs/ 里的路径。找不到返回 None。"""
    b = base_of(by1file)
    if not b:
        return None
    ext = {'config': '.config.json',
           'tensors': '.tensors.json',
           'gguf': '.gguf-tensors.json'}.get(kind, kind)
    p = os.path.join(REF, b + ext)
    return p if os.path.exists(p) else None


def need(by1file, kind='config'):
    """同上，但找不到就**明确报错**，不返回 None。

    **静默返回 None 比报错坏** —— 用的人会拿 None 去 open()，
    报出来的是 `TypeError`，看起来像代码坏了。
    """
    p = paths(by1file, kind)
    if p is None:
        b = base_of(by1file)
        raise SystemExit(
            '  %s 的 refs 里没有 %s：\n'
            '    期望 refs/%s%s\n'
            '    **要么这个模型还没抓，要么 .by1 头部的 by1-repo 写错了。**'
            % (by1file, kind, b or '(读不到 by1-repo)',
               {'config': '.config.json', 'tensors': '.tensors.json',
                'gguf': '.gguf-tensors.json'}.get(kind, '')))
    return p


def audit(files=None):
    """查一遍：哪些 .by1 缺 config / 缺张量清单。返回 [(名字, 缺什么)]。"""
    if files is None:
        files = by1paths.names()
    out = []
    for f in files:
        miss = []
        if repo_of(f) is None:
            miss.append('没有 by1-repo')
        else:
            if paths(f, 'config') is None:
                miss.append('缺 config')
            if paths(f, 'tensors') is None:
                miss.append('缺张量清单')
        if miss:
            out.append((f, miss))
    return out


# ── 语料审计：**每个 refs 文件都要有出处** ────────────────────────────
#
# 一个 refs 文件的出处只有三种：
#
#   ① 某个 `.by1` 的 `by1-repo`     有描述，进护栏
#   ② `models.tsv` 的 HF id         手写的知识，是种子
#   ③ `refs/SOURCES.tsv`            **探索过、还没写 .by1**
#
# **答不出这三条的，就是硬编码的知识** —— 删掉就再也抓不回来。
# 而 refs/ 有 25.8 MB（占仓库数据的 99%），这件事必须看得见。

#: 文件名后缀，按**从长到短**排 —— `.tensors.json` 会吃掉 `.json` 的匹配。
SUFFIXES = ('.gguf-tensors.json', '.config.json', '.tensors.json')


def base_and_kind(path):
    """`refs/x__Y.tensors.json` -> `('x__Y', '.tensors.json')`。

    不符合规则的文件返回 `(None, None)` —— **那是审计要报的**，
    不是悄悄放过。
    """
    n = os.path.basename(path)
    for s in SUFFIXES:
        if n.endswith(s):
            return n[:-len(s)], s
    return None, None


def sources():
    """`refs/SOURCES.tsv`：基名 -> 说明。文件不在就是空的。"""
    p = os.path.join(REF, 'SOURCES.tsv')
    out = {}
    if not os.path.exists(p):
        return out
    for line in by1io.iter_lines(p):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        p2 = line.split('\t')
        hfid = p2[0].strip()
        if '/' in hfid:
            out[hfid.replace('/', '__')] = p2[1].strip() if len(p2) > 1 else ''
    return out


def tsv_seeds():
    """`models.tsv` 的基名 -> 长名。"""
    out = {}
    p = by1paths.root('models.tsv')
    if not os.path.exists(p):
        return out
    for line in by1io.iter_lines(p):
        line = line.rstrip('\n')
        if not line or line.lstrip().startswith('#'):
            continue
        p2 = line.split('\t')
        if len(p2) >= 3 and '/' in p2[2]:
            out[p2[2].replace('/', '__')] = p2[0]
    return out


def attributable():
    """所有**有出处**的基名 -> 出处说明。"""
    known = {}
    for f in by1paths.names():
        r = repo_of(f)
        if r:
            known[r.replace('/', '__')] = 'by1:%s' % f
    for b, long in tsv_seeds().items():
        known.setdefault(b, 'models.tsv:%s' % long)
    for b, why in sources().items():
        known.setdefault(b, 'SOURCES.tsv%s' % (': ' + why if why else ''))
    return known


def corpus_audit():
    """refs/ 语料审计。返回 `(ok, lines)`。

    三件事：

      · 每个 `refs/*.json` 都能解析成**一个 JSON 对象**
      · 每个文件名的基名**都有出处**（见上）
      · 每个有 `by1-repo` 的 `.by1`，它的 config 与张量清单**都在**

    第二件是新的。以前只查"`.by1` 缺不缺 refs"（一个方向），
    于是"一个 refs 文件从哪来"这个问题**从来没有被问过** ——
    直到有人发现里面躺着一个 29 字节的 `Invalid username or password.`。
    """
    import glob as _g

    # **`.index.json` 不是语料，是 by1index 的工作缓存。**
    #
    # 它们是抓张量清单时的原始索引，`.gitignore` 挡着（不进仓库），
    # 但**在磁盘上**。原来这里 glob `*.json` 把它们全捞进来了 ——
    # 后果有两个：语料计数虚高 9 个，而且它们的 kind 是 None，
    # 让下面那句 `sorted(kinds.items())` 直接 TypeError。
    files = sorted(p for p in _g.glob(os.path.join(REF, '*.json'))
                   if not p.endswith('.index.json'))
    kinds, bad, orphan = {}, [], []
    for p in files:
        b, s = base_and_kind(p)
        kinds[s] = kinds.get(s, 0) + 1
        try:
            d = by1io.read_json(p)
            if not isinstance(d, dict):
                bad.append((os.path.basename(p), '不是一个 JSON 对象'))
        except Exception as e:
            bad.append((os.path.basename(p),
                        '%s: %s' % (type(e).__name__, str(e)[:60])))
        if b is None:
            orphan.append((os.path.basename(p), '名字不符合 refs 规则'))
    known = attributable()
    for p in files:
        b, _s = base_and_kind(p)
        if b is not None and b not in known:
            orphan.append((os.path.basename(p), b.replace('__', '/')))
    # **合成模型没有 `by1-repo`，那不算缺** —— 只看"缺 config / 缺张量清单"。
    miss = [(f, m) for f, m in audit() if any('缺' in x for x in m)]

    lines = []
    # **先滤再排。** 原来写的是 `sorted(kinds.items()) if k` —— sorted
    # 先跑，撞上 None 键就 TypeError。只有存在 kind 为 None 的文件时
    # 才会炸，所以是个"数据一变就现形"的坑。
    kind_bits = ' · '.join(
        '%s %d' % (k.lstrip('.').replace('.json', ''), v)
        for k, v in sorted((a, b) for a, b in kinds.items() if a))
    lines.append('  refs/*.json       %d 个（%s）' % (len(files), kind_bits))
    lines.append('  能解析为 JSON     %d / %d'
                 % (len(files) - len(bad), len(files)))
    lines.append('  有出处            %d / %d（by1 %d · models.tsv %d · SOURCES %d）'
                 % (len(files) - len(orphan), len(files),
                    sum(1 for v in known.values() if v.startswith('by1:')),
                    len(tsv_seeds()), len(sources())))
    lines.append('  真实模型缺 refs   %d 个' % len(miss))
    # **不算失败，但必须看得见。** `by1fetch` 只产 config / tensors 两种；
    # `.gguf-tensors.json` 要从 GGUF 仓库另外抽头，而它们的出处只记下来一个
    # （见 refs/SOURCES.tsv 的头注）。藏起来的话，读者会以为 refs/ 全能重建。
    lines.append('  重建例外          %d 个 .gguf-tensors.json'
                 '（by1fetch 不产这种；出处见 refs/SOURCES.tsv）'
                 % kinds.get('.gguf-tensors.json', 0))
    for n, why in bad:
        lines.append('    !! 解析不了 %s —— %s' % (n, why))
    for n, why in orphan:
        lines.append('    !! 没有出处 %s -> %s' % (n, why))
    for f, m in miss:
        lines.append('    !! %s 缺 %s' % (f, ', '.join(m)))
    ok = not (bad or orphan or miss)
    lines.append('  [%s] refs 语料 %s'
                 % ('PASS' if ok else 'FAIL',
                    '每个文件都能解析、都有出处' if ok else '见上面的 !! 行'))
    return ok, lines


def main():
    print()
    print('  refs 路径规则：refs/<owner>__<Repo>.{config,tensors}[.gguf-tensors].json')
    print('  owner/Repo 从 .by1 头部的 `# by1-repo:` 来；没有 .by1 的看 refs/SOURCES.tsv。')
    print()
    ok, lines = corpus_audit()
    for line in lines:
        print(line)
    print()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
