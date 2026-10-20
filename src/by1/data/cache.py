#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1cache -- **权重缓存清单：生成、校验、报红。**

## 它解决什么

租来的卡（AutoDL 之类）重租就丢容器盘，权重要么重下（网速不够，
这正是本仓库开影子线的原因），要么放**文件存储**跨实例复用。复用的
前提是"这份文件还是下载时那份吗" —— 没有清单就没法回答，只能信
`safetensors` 加载不炸 = 没坏。**那是两个不同的问题。**

## 判据

- `scan`：对目录里每个权重文件记 (repo@revision, 文件名, 字节数, sha256)。
- `verify`：重算逐个对 —— **改一个字节必须报红**（改字节不改大小，
  所以 size 只是快速路，sha256 才是判据）。
- 缺文件、多文件、hash 不符、大小不符，四类都点名，不静默。

用法:
    python src/data/by1cache.py scan ~/.cache/huggingface/hub -o cache-manifest.json
    python src/data/by1cache.py scan /root/autodl-fs/weights --repo Qwen/Qwen3-0.6B \\
        --revision 1d4bf0f2 -o cache-manifest.json
    python src/data/by1cache.py verify cache-manifest.json
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

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re

#: 权重文件认这些后缀。config/tokenizer 之类也进清单 —— 重租卡后
#: 少一个 tokenizer.json 和少一个分片一样致命，而且更晚才炸。
WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.gguf', '.pt', '.pth',
                   '.json', '.model', '.txt', '.jinja', '.tiktoken')

#: HF hub 缓存布局：models--<org>--<name>/snapshots/<revision>/<file>
_HUB_RE = re.compile(r'models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]([^/\\]+)')


def sha256_of(path, chunk=8 * 1024 * 224):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan(roots, repo=None, revision=None):
    """目录 -> 清单条目。HF hub 布局自动认 repo/revision；扁平目录用给的。"""
    entries = []
    for root in roots:
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            # hub 缓存里 blobs/ 是内容寻址的硬链接源，扫 snapshots/ 就够 ——
            # 两边都扫会把同一个字节记两遍。
            dirnames[:] = [d for d in dirnames if d not in ('blobs', '.git')]
            for name in sorted(filenames):
                if not name.endswith(WEIGHT_SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                if not os.path.isfile(path):
                    continue
                m = _HUB_RE.search(path.replace('\\', '/'))
                e_repo = repo or (('%s/%s' % (m.group(1), m.group(2))) if m else None)
                e_rev = revision or (m.group(3) if m else None)
                entries.append({
                    'repo': e_repo, 'revision': e_rev, 'file': name,
                    'path': os.path.relpath(path, root), 'root': root,
                    'bytes': os.path.getsize(path),
                    'sha256': sha256_of(path),
                })
    entries.sort(key=lambda e: (e['root'], e['path']))
    return {'schema': 1, 'kind': 'by1-weight-cache',
            'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'entries': entries}


def verify(manifest, roots=None):
    """清单 -> 逐条重算对账。返回 (ok 数, 问题列表)。**判据是 sha256。**"""
    problems = []
    ok = 0
    for e in manifest['entries']:
        # 路径解析：绝对路径照用；相对的先按原 root，再按 --root 逐个试
        candidates = []
        p = e['path']
        if os.path.isabs(p):
            candidates.append(p)
        else:
            if e.get('root'):
                candidates.append(os.path.join(e['root'], p))
            for r in (roots or []):
                candidates.append(os.path.join(os.path.abspath(r), p))
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            problems.append('缺文件: %s' % e['path'])
            continue
        size = os.path.getsize(path)
        if size != e['bytes']:
            problems.append('大小不符: %s（清单 %d 实际 %d）'
                            % (e['path'], e['bytes'], size))
            continue
        got = sha256_of(path)
        if got != e['sha256']:
            problems.append('**sha256 不符: %s**（清单 %s 实际 %s）—— 内容被改过'
                            % (e['path'], e['sha256'][:12], got[:12]))
            continue
        ok += 1
    return ok, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('scan', help='扫目录生成清单')
    s.add_argument('roots', nargs='+')
    s.add_argument('-o', '--out', default='cache-manifest.json')
    s.add_argument('--repo', help='扁平目录用（HF hub 布局自动认）')
    s.add_argument('--revision', help='同上')
    v = sub.add_parser('verify', help='按清单校验（改一字节报红）')
    v.add_argument('manifest')
    v.add_argument('--root', action='append', default=[],
                   help='清单里的相对路径换个根找（换机器后的常用姿势）')
    args = ap.parse_args(argv)

    if args.cmd == 'scan':
        manifest = scan(args.roots, repo=args.repo, revision=args.revision)
        pathlib.Path(args.out).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        total = sum(e['bytes'] for e in manifest['entries'])
        print('  清单写了 %s：%d 个文件，%.2f GB'
              % (args.out, len(manifest['entries']), total / 1024 ** 3))
        return 0

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding='utf-8'))
    ok, problems = verify(manifest, roots=args.root)
    print('  校验 %d 个文件：一致 %d · 问题 %d' % (len(manifest['entries']), ok, len(problems)))
    for p in problems:
        print('    !! ' + p)
    return 0 if not problems else 1


if __name__ == '__main__':
    _sys.exit(main())
