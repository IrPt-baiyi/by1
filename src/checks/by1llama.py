#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1llama -- llama.cpp 基线记录、GGUF 清单校验、logits 逐位对拍。

## 它补上什么

`docs/llama.cpp-baseline.md` 钉死了**输入**（llama.cpp tag、GGUF 文件、
SHA-256、构建法），但"跑基线那台机器到底是什么"和"logits 对不对"没有
判卷人。这里补三件：

    record         基线记录表 —— **每项带来源命令**（tag/commit、编译器、
                   CUDA、GPU、OS），报告里的值可以逐项复核
    gguf-check     对照固定清单验下载文件的 sha256 —— 清单是判据，
                   文件坏/缺/多都点名
    logits-compare 两个 logits dump **逐位**比较 -> 报告（rc1 的判据
                   "llama-cli logits 逐位"就落在这）
    logits-run     用 llama-eval-callback 产出 dump（要本机有 llama.cpp
                   构建；GPU 日跑）

用法:
    python src/checks/by1llama.py record -o cgen/llama-baseline-record.json
    python src/checks/by1llama.py gguf-check --dir /root/autodl-fs/gguf
    python src/checks/by1llama.py logits-compare ref.logits got.logits
    python src/checks/by1llama.py logits-run --gguf Qwen3.8-27B-...-Iq3_s.gguf
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

import by1skip

#: 固定输入（和 docs/llama.cpp-baseline.md 的 Pinned inputs 同源；
#: 改这里必须同步那份文档 —— 两处漂移就是"基线"不基了）。
PINNED = {
    'llama_cpp_tag': 'v0.6.0',
    'llama_cpp_commit_prefix': 'd812350',
    'gguf_repo': 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF',
    'gguf_revision': 'c7b6817b8affebf3b38c58313394b061345a4924',
    'files': {
        'Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf': {
            'role': 'text+量化(IQ3_S)', 'bytes': 11.8 * 1024 ** 3,
            'sha256': '64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810'},
        'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf': {'role': 'MTP 变体', 'sha256': None},
        'mmproj-Qwen3.8-27B-BF16.gguf': {'role': '视觉 mmproj', 'sha256': None},
    },
}

#: 基线记录的字段与**取值命令** —— 报告里每个值都能顺着命令复核。
RECORD_COMMANDS = [
    ('llama_cpp_tag', 'git -C llamacpp/llama.cpp describe --tags'),
    ('llama_cpp_commit', 'git -C llamacpp/llama.cpp rev-parse HEAD'),
    ('gguf_repo', '（固定输入，见 docs/llama.cpp-baseline.md）'),
    ('gguf_revision', '（固定输入，见 docs/llama.cpp-baseline.md）'),
    ('compiler', 'gcc --version | head -1'),
    ('cuda_toolkit', 'nvcc --version | tail -1'),
    ('gpu', 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'),
    ('os', 'uname -a'),
    ('python', 'python --version'),
    ('hf_home', 'echo $HF_HOME'),
]


def _run(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=30)
    except Exception as exc:
        return '（跑不起来：%s）' % exc
    body = (p.stdout.strip() or p.stderr.strip() or '(空)')[:300]
    # **返回码要进记录** —— 命令失败和命令没输出是两回事，
    # 记录表把它们记成同一个值就是不出声的错（by1lint ⑨ 盯这个）。
    return body if p.returncode == 0 else '%s（rc=%d）' % (body, p.returncode)


def sha256_of(path, chunk=8 * 1024 * 224):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def record(out_path):
    """基线记录表：每项 (值, 来源命令) —— 可逐项复核。"""
    rows = {}
    special = {
        'gguf_repo': PINNED['gguf_repo'],
        'gguf_revision': PINNED['gguf_revision'],
    }
    for key, command in RECORD_COMMANDS:
        if key in special:
            rows[key] = {'value': special[key], 'command': command}
        elif key == 'llama_cpp_tag':
            value = _run(command)
            want = PINNED['llama_cpp_tag']
            rows[key] = {'value': value, 'command': command,
                         'matches_pinned': value == want, 'pinned': want}
        elif key == 'llama_cpp_commit':
            value = _run(command)
            rows[key] = {'value': value, 'command': command,
                         'matches_pinned': value.startswith(PINNED['llama_cpp_commit_prefix']),
                         'pinned_prefix': PINNED['llama_cpp_commit_prefix']}
        else:
            rows[key] = {'value': _run(command), 'command': command}
    doc = {'schema': 1, 'kind': 'llama-baseline-record',
           'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'fields': rows}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for key, entry in rows.items():
        mark = ''
        if 'matches_pinned' in entry:
            mark = ' ✓钉' if entry['matches_pinned'] else ' !!与钉住值不符'
        print('  %-20s %s%s' % (key, entry['value'], mark))
    print('  记录写了 %s' % out_path)
    bad = [k for k, e in rows.items() if e.get('matches_pinned') is False]
    return 1 if bad else 0


def gguf_check(directory):
    """对照固定清单验文件。缺/坏/没 pin 的都点名 —— 清单是判据。"""
    problems = []
    directory = pathlib.Path(directory)
    for name, want in PINNED['files'].items():
        path = directory / name
        if not path.exists():
            problems.append('缺文件: %s（%s）' % (name, want['role']))
            continue
        if want.get('sha256'):
            got = sha256_of(path)
            if got != want['sha256']:
                problems.append('**sha256 不符: %s**（清单 %s 实际 %s）'
                                % (name, want['sha256'][:12], got[:12]))
            else:
                print('  ok %-44s %s（sha 对）' % (name, want['role']))
        else:
            print('  -- %-44s %s（清单没 pin sha —— 这是清单的账）'
                  % (name, want['role']))
    for p in problems:
        print('  !! ' + p)
    print('  GGUF 清单：%d 个固定文件，问题 %d' % (len(PINNED['files']), len(problems)))
    return 1 if problems else 0


def _load_logits(path):
    """dump 两种收法：JSON 列表，或每行一个数（llama-eval-callback 的输出）。"""
    text = pathlib.Path(path).read_text(encoding='utf-8', errors='replace').strip()
    data = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            data = [float(x) for x in parsed]
    except ValueError:
        data = None          # 不是 JSON —— 走逐行解析（这不是吞错）
    if data is not None:
        return data
    out = []
    for line in text.splitlines():
        line = line.strip()
        for token in line.replace(',', ' ').split():
            try:
                out.append(float(token))
            except ValueError:
                continue
    return out


def logits_compare(ref_path, got_path, out_path=None):
    """两个 logits dump **逐位**比较 -> 报告。rc1 的判据落在这。"""
    ref, got = _load_logits(ref_path), _load_logits(got_path)
    problems = []
    if not ref or not got:
        problems.append('dump 读出来是空的（%s / %s）' % (ref_path, got_path))
    if len(ref) != len(got):
        problems.append('长度不同：%d vs %d' % (len(ref), len(got)))
    n = min(len(ref), len(got))
    first = None
    bitwise = True
    max_abs = 0.0
    for i in range(n):
        if ref[i] != got[i]:
            bitwise = False
            if first is None:
                first = i
            max_abs = max(max_abs, abs(ref[i] - got[i]))
    report = {'schema': 1, 'kind': 'llama-logits-compare',
              'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'ref': str(ref_path), 'got': str(got_path),
              'values': n, 'ref_values': len(ref), 'got_values': len(got),
              'bitwise_identical': bitwise and len(ref) == len(got) and n > 0,
              'first_mismatch_index': first, 'max_absolute': max_abs,
              'problems': problems}
    if out_path:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('  报告写了 %s' % out_path)
    if report['bitwise_identical']:
        print('  [PASS] logits 逐位相同（%d 个值）' % n)
        return 0
    print('  [FAIL] logits 不逐位：首个分歧 @ %s，max|Δ|=%.3e，长度 %d/%d'
          % (first, max_abs, len(ref), len(got)))
    for p in problems:
        print('    - ' + p)
    return 1


def logits_run(gguf, prompt, out_path, llama_cli=None):
    """跑 llama.cpp 产 logits dump（要本机构建）。命令本身进报告 ——
    "每项可从报告追溯到命令"。"""
    cli = llama_cli or shutil.which('llama-eval-callback') \
        or shutil.which('llama-cli')
    if not cli:
        return by1skip.skip('没找到 llama-eval-callback / llama-cli —— '
                           '先按 docs/llama.cpp-baseline.md 构建（GPU 日跑）')
    # llama-eval-callback 逐 token 打 logits；-p 固定 prompt、--seed 固定
    command = [cli, '-m', str(gguf), '-p', prompt, '--seed', '42',
               '-n', '1']
    print('  命令：' + ' '.join(command))
    p = subprocess.run(command, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(p.stdout, encoding='utf-8')
    print('  dump 写了 %s（rc=%d）' % (out_path, p.returncode))
    return p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record', help='基线记录表（每项带来源命令）')
    r.add_argument('-o', '--out', default='cgen/llama-baseline-record.json')
    g = sub.add_parser('gguf-check', help='对照固定清单验 GGUF 文件')
    g.add_argument('--dir', default='.')
    c = sub.add_parser('logits-compare', help='两个 logits dump 逐位比较')
    c.add_argument('ref')
    c.add_argument('got')
    c.add_argument('-o', '--out', default='cgen/llama-logits-compare.json')
    n = sub.add_parser('logits-run', help='跑 llama.cpp 产 dump（要构建）')
    n.add_argument('--gguf', required=True)
    n.add_argument('--prompt', default='The capital of France is')
    n.add_argument('-o', '--out', default='cgen/llama-logits.txt')
    n.add_argument('--llama-cli')
    args = ap.parse_args(argv)
    if args.cmd == 'record':
        return record(args.out)
    if args.cmd == 'gguf-check':
        return gguf_check(args.dir)
    if args.cmd == 'logits-compare':
        return logits_compare(args.ref, args.got, args.out)
    return logits_run(args.gguf, args.prompt, args.out, args.llama_cli)


if __name__ == '__main__':
    _sys.exit(main())
