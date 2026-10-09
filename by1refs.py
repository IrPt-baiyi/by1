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
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'refs')

_REPO = re.compile(r'^#\s*by1-repo:\s*(\S+)', re.M)


def repo_of(by1file):
    """从 .by1 头部读出 `owner/Repo`。读不到返回 None。"""
    p = by1file if os.path.isabs(by1file) else os.path.join(HERE, by1file)
    if not os.path.exists(p):
        return None
    try:
        head = io.open(p, encoding='utf-8').read(2000)
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
    import glob
    if files is None:
        files = [os.path.basename(f) for f in
                 sorted(glob.glob(os.path.join(HERE, '*.by1')))]
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


def main():
    print()
    print('  refs 路径规则：refs/<owner>__<Repo>.{config,tensors}[.gguf-tensors].json')
    print('  owner/Repo 从 .by1 头部的 `# by1-repo:` 来。')
    print()
    rows = audit()
    print('  %-46s %s' % ('文件', '状态'))
    print('  ' + '-' * 74)
    for f, miss in rows:
        print('  %-46s %s' % (f[:46], ', '.join(miss)))
    print()
    print('  %d 个 .by1 有缺项' % len(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
