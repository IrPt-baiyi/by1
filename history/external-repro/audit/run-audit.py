#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run-audit -- 把外部复现从"报告级"升到"审计级"。

## 两级的差别

**报告级**：有人说"我跑过了"，附一段粘贴的输出 —— 信不信由你。
**审计级**：跑出**可核对的产物** —— 一份 `audit-report.json`，里面带
环境指纹、每步的退出码、判定行、stdout 的 SHA-256，以及和
`expected.json`（维护方自己的实测）的逐条比对结果。

别人跑完把 json 发回来，**我能在一分钟内核完**：不用信他的话，
比对结果与哈希摆在那。

## 用法（在仓库根目录）

    python history/external-repro/audit/run-audit.py            # 跑 + 出报告
    python history/external-repro/audit/run-audit.py --check    # 跑 + 和 expected.json 比
    python history/external-repro/audit/run-audit.py --tier 1   # 只跑某一步

步骤与 `docs/repro-guide.md` **逐条对应**（审计的是那份指南，不是另一套）：
    tier1     档 1（纯标准库）
    tier3     档 3（要 numpy）
    complete  档 4（要 torch + transformers，即 `pip install -e ".[full]"`）

**真权重那一步（指南第 ④ 步）不自动化** —— 它要下权重（0.5~3 GB），
而且跑哪几个模型是复核者的选择。手动跑法见 `docs/repro-guide.md`；
产出照样可以放进 `audit-report.json`（自己加一段 `"weights": {...}`）。

## 三条约定

① **跳过不写成通过**：缺依赖的步骤记 `skipped`，报告里看得见；
② **比对只看稳定项**：判定行的模式（如 `0 项失败`），不比"没验"的数量
   （它随环境变）；
③ **哈希是给核对者看的**：stdout 的 SHA-256 让"你跑的和我看的同一段"成立。
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXPECTED = os.path.join(HERE, 'expected.json')

#: 步骤 = (名字, 命令, 需要的第三方包, expected.json 里的键)
STAGES = [
    ('tier1', [sys.executable, 'src/checks/by1all.py', '--tier', '1'], (), 'tier1'),
    ('tier3', [sys.executable, 'src/checks/by1all.py', '--tier', '3'], ('numpy',), 'tier3'),
    ('complete', [sys.executable, 'src/checks/by1complete.py'],
     ('torch', 'transformers'), 'complete'),
]


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:                              # noqa: BLE001
        return False


def run_stage(name, cmd, need, key, only):
    """跑一步。返回报告条目（skipped/ran，退出码，判定行，哈希）。"""
    if only and name != only:
        return None
    missing = [m for m in need if not _have(m)]
    if missing:
        return {'stage': key, 'status': 'skipped',
                'reason': '缺第三方包：%s' % '、'.join(missing)}
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=3600)
    except Exception as exc:                       # noqa: BLE001
        return {'stage': key, 'status': 'error', 'reason': str(exc)}
    out = (r.stdout or '') + (r.stderr or '')
    verdicts = [l.strip() for l in out.splitlines()
                if ('[PASS]' in l or '[FAIL]' in l or '[全过' in l
                    or '[有失败]' in l or '0 项失败' in l or '项失败' in l)]
    return {'stage': key,
            'status': 'ran',
            'exit': r.returncode,
            'verdicts': verdicts[-6:],
            'stdout_sha256': hashlib.sha256(out.encode('utf-8')).hexdigest(),
            'stdout_tail': out.strip().splitlines()[-4:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='和 expected.json 比对（维护方的实测基线）')
    ap.add_argument('--tier', choices=['tier1', 'tier3', 'complete'],
                    default=None, help='只跑这一步')
    ap.add_argument('--out', default=os.path.join(HERE, 'audit-report.json'))
    args = ap.parse_args()

    report = {
        'tool': 'run-audit',
        'root': ROOT,
        'env': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'machine': platform.machine(),
        },
        'stages': [],
    }
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        report['env']['commit'] = (r.stdout or '').strip() or None
    except Exception:                              # noqa: BLE001
        report['env']['commit'] = None

    for name, cmd, need, key in STAGES:
        entry = run_stage(name, cmd, need, key, args.tier)
        if entry:
            report['stages'].append(entry)
            mark = {'ran': '跑了', 'skipped': '跳过', 'error': '出错'}[entry['status']]
            print('  %-9s %s%s' % (key, mark,
                                   ('（退出码 %d）' % entry['exit'])
                                   if entry['status'] == 'ran' else ''))

    problems = []
    if args.check:
        with open(EXPECTED, encoding='utf-8') as fh:
            want = json.load(fh)
        for entry in report['stages']:
            exp = want.get('stages', {}).get(entry['stage'])
            if not exp or entry['status'] != 'ran':
                continue
            if entry['exit'] != exp.get('exit', 0):
                problems.append('%s：退出码 %d ≠ 预期 %d'
                                % (entry['stage'], entry['exit'], exp.get('exit', 0)))
            blob = '\n'.join(entry['verdicts'])
            for pat in exp.get('expect_in_verdicts', []):
                if not re.search(pat, blob):
                    problems.append('%s：判定行里找不到 %r' % (entry['stage'], pat))
        report['check'] = {'ok': not problems, 'problems': problems}

    with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write('\n')
    print('  报告写了 %s' % os.path.relpath(args.out, ROOT))
    if args.check:
        if problems:
            print('  [FAIL] 与预期不一致 %d 条：' % len(problems))
            for p in problems:
                print('     · %s' % p)
            return 1
        print('  [PASS] 与 expected.json 一致（%d 个判定行模式全中）'
              % sum(len(v.get('expect_in_verdicts', []))
                    for v in json.load(open(EXPECTED, encoding='utf-8'))
                    ['stages'].values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
