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
    python src/checks/by1llama.py build-gguf <hf-dir> out.gguf \\
        --ref upstream-f32.gguf --arch qwen3 --n-blocks 28
"""

import contextlib
import io
import math
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import by1paths  # noqa: E402,F401

# **gguf-py 来自 llama.cpp 的检出，by1 不重写一份命名映射表。**
# `llamacpp/` 被 gitignore 挡着（可重新下载），所以这里要能容忍它不在 ——
# 缺了就在 `build-gguf` 那一处报错，而不是让整个检查器 import 不了。
_GGUF_PY = by1paths.root('llamacpp', 'llama.cpp', 'gguf-py')
if _os.path.isdir(_GGUF_PY):
    _sys.path.insert(0, _GGUF_PY)

import argparse
import datetime
import hashlib
import json
import pathlib
import re
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

#: **小模型冒烟 / bench 的固定输入。** 1.1 MB，可随时重下；
#: `sha256` 是判据 —— 文件不对就不许拿它的数当基线。
#: （大模型 IQ3_S 11.8 GB 那一条归 GPU 日；这条只证明
#:  "钉定的 llama.cpp 能真跑出一个数"。）
TINY_MODEL = {
    'name': 'stories260K.gguf',
    'repo': 'ggml-org/models',
    'url': ('https://huggingface.co/ggml-org/models/resolve/main/'
            'tinyllamas/stories260K.gguf'),
    'bytes': 1185376,
    'sha256': ('270cba1bd5109f42d03350f604060245'
               '60464db173c0e387d91f0426d3bd256d'),
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


def _run_big(cmd, timeout=600):
    """同 `_run`，但给长跑（bench）用 —— 输出整份留着给正则抠数。

    **退出码要说出来**：bench 挂了也可能打出半张表，光看输出会把"跑挂了"
    读成"跑过了"。非零就把退出码塞进返回串，让下游的正则抠不到数而红。
    """
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as exc:
        return "（跑不起来：%s）" % exc
    if p.returncode != 0:
        return "（退出码 %d）\n%s%s" % (p.returncode, p.stdout, p.stderr)
    return p.stdout + p.stderr

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


def _find_build(build_dir=None):
    """找本机那个 llama.cpp 构建（钉定 clone 优先，退回旧快照）。"""
    roots = [build_dir] if build_dir else [
        by1paths.root('build', 'clean-clone', 'llama-v0.6.0'),
        by1paths.root('llamacpp', 'llama.cpp'),
    ]
    return next((r for r in roots if r and _os.path.isdir(r)), None)


def _exe(bin_dir, name):
    return _os.path.join(bin_dir, name + (".exe" if _os.name == "nt" else ""))


def bench_run(build_dir=None, bin_dir=None, model=None, reps=3):
    """**钉定的 llama.cpp 真跑出数**：小模型 bench + 版本串对账。

    三态：没有构建或没有模型 -> `[跳过]`（rc=2）；模型 sha256 不对
    或 bench 出不了数 -> FAIL。t/s 本身是**机器相关**的，所以它只
    证明"跑得动"，不当作跨机器的阈值（阈值留给 GPU 日的固定机器）。
    """
    root = _find_build(build_dir)
    if root is None:
        print("  [跳过] 本机没有 llama.cpp 构建 —— 这一项没验")
        return by1skip.CODE
    bin_dir = bin_dir or _os.path.join(root, "build", "bin")
    exe = _exe(bin_dir, "llama-bench")
    if not _os.path.exists(exe):
        print("  [跳过] 构建里没有 llama-bench —— 这一项没验")
        return by1skip.CODE
    model = model or by1paths.root("models", "llama-tiny", TINY_MODEL["name"])
    if not _os.path.exists(model):
        print("  [跳过] 小模型没下载（%s）—— bench 没验"
              % TINY_MODEL["name"])
        print("    下载：%s" % TINY_MODEL["url"])
        return by1skip.CODE
    got = sha256_of(model)
    if got != TINY_MODEL["sha256"]:
        print("  [FAIL] %s 的 sha256 不对：%s != %s"
              % (TINY_MODEL["name"], got[:16], TINY_MODEL["sha256"][:16]))
        return 1
    cmd = '"%s" -m "%s" -p 32 -n 16 -r %d -t 4' % (exe, model, reps)
    out = _run_big(cmd, timeout=600)
    build = None
    m = re.search(r"^build: (\S+)", out, re.M)
    if m:
        build = m.group(1)
    want = PINNED["llama_cpp_commit_prefix"]
    rows = {}
    for tag in ("pp32", "tg16"):
        mm = re.search(r"\|\s*" + tag + r"\s*\|\s*([0-9.]+)", out)
        rows[tag] = float(mm.group(1)) if mm else None
    print("  模型   %s  sha256=%s... %d 字节"
          % (TINY_MODEL["name"], got[:12], TINY_MODEL["bytes"]))
    print("  build  %s" % (build or "(没打印)"))
    for tag in ("pp32", "tg16"):
        print("  %-5s  %s t/s" % (tag, rows[tag] if rows[tag] else "(没出数)"))
    bad = []
    if not build or want not in build:
        bad.append("bench 报的 build=%r 里没有 %s" % (build, want))
    for tag in ("pp32", "tg16"):
        if not rows[tag] or rows[tag] <= 0:
            bad.append("%s 没出数" % tag)
    print()
    ok = not bad
    print("  [%s] 钉定构建上的小模型 bench：%s"
          % ("PASS" if ok else "FAIL",
             "pp32=%.2f t/s · tg16=%.2f t/s（build %s）"
             % (rows["pp32"] or 0, rows["tg16"] or 0, build)
             if ok else "；".join(bad)))
    return 0 if ok else 1

def pin_check(build_dir=None, bin_dir=None):
    """**钉定版本的实测判据。**

    干净 clone 构建之后，本地那个二进制**必须**是钉住的版本：
      · git 检出：tag == PINNED['llama_cpp_tag']、commit 前缀对得上；
      · 二进制：`--version` 里带的 commit 与钉住的前缀一致。

    三态：没有构建 -> `[跳过]`（rc=2，CI/干净检出里就是这一态）；
    有构建但版本不对 -> FAIL（不是跳过 —— 那是"基线不基了"）。
    """
    want_tag = PINNED['llama_cpp_tag']
    want_sha = PINNED['llama_cpp_commit_prefix']
    roots = [build_dir] if build_dir else [
        by1paths.root('build', 'clean-clone', 'llama-v0.6.0'),
        by1paths.root('llamacpp', 'llama.cpp'),
    ]
    root = next((r for r in roots if r and _os.path.isdir(r)), None)
    if root is None:
        print("  [跳过] 本机没有 llama.cpp 构建 —— 这一项没验（不是通过）")
        return by1skip.CODE
    bad = []
    sha = None
    if _os.path.isdir(_os.path.join(root, ".git")):
        tag = _run('git -C "%s" describe --tags' % root)
        sha = _run('git -C "%s" rev-parse HEAD' % root)
        print("  检出   %s  tag=%s  sha=%s" % (root, tag, sha[:12]))
        if tag != want_tag:
            bad.append("tag=%r，钉住的是 %r" % (tag, want_tag))
        if not sha.startswith(want_sha):
            bad.append("commit=%r 不以 %r 开头" % (sha[:12], want_sha))
    else:
        print("  检出   %s（不是 git 检出 —— 只用二进制版本判）" % root)
    bin_dir = bin_dir or _os.path.join(root, "build", "bin")
    for exe in ("llama-cli", "llama-bench"):
        path = _os.path.join(bin_dir, exe + (".exe" if _os.name == "nt" else ""))
        if not _os.path.exists(path):
            bad.append("缺 %s（干净 clone 构建没产出它）" % exe)
            continue
        out = _run('"%s" --version' % path)
        # **不能只看第一行**：CUDA 构建的 `llama-bench --version` 第一行是
        # `ggml_cuda_init: found 1 CUDA devices ...`（初始化日志打在 stdout），
        # 版本行在后面 —— 2026-10-19 在 T4 的 CUDA 构建上就是这么假红的。
        lines = [l for l in (out or "").splitlines() if l.strip()]
        hit = next((l for l in lines if want_sha in l), None)
        line = hit or (lines[0] if lines else "(空)")
        print("  %-12s %s" % (exe, line))
        if hit is None:
            bad.append("%s 的版本里没有 %s（共 %d 行，首行 %r）"
                       % (exe, want_sha, len(lines), line[:60]))
    print()
    ok = not bad
    print("  [%s] llama.cpp 钉定版本（%s / %s）：%s"
          % ("PASS" if ok else "FAIL", want_tag, want_sha,
             "实测二进制与钉住的一致" if ok else "；".join(bad)))
    return 0 if ok else 1

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
    # **NaN 要单独查**：`nan != nan` 恒真，所以两个 NaN 会被记成"首个分歧"，
    # 而真正的信息是"dump 里有 NaN" —— 上游算坏了，不是两边差了一点。
    nans_ref = sum(1 for v in ref if math.isnan(v))
    nans_got = sum(1 for v in got if math.isnan(v))
    if nans_ref or nans_got:
        problems.append("dump 里有 NaN（ref=%d, got=%d）—— 上游算坏了"
                        % (nans_ref, nans_got))
    for i in range(n):
        a_v, b_v = ref[i], got[i]
        same = (a_v == b_v) or (math.isnan(a_v) and math.isnan(b_v))
        if not same:
            bitwise = False
            if first is None:
                first = i
            if math.isnan(a_v) or math.isnan(b_v):
                max_abs = float("nan")
            else:
                max_abs = max(max_abs, abs(a_v - b_v))
    report = {'schema': 1, 'kind': 'llama-logits-compare',
              'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'ref': str(ref_path), 'got': str(got_path),
              'values': n, 'ref_values': len(ref), 'got_values': len(got),
              'bitwise_identical': bitwise and len(ref) == len(got) and n > 0,
              'first_mismatch_index': first, 'max_absolute': max_abs,
              'problems': problems}
    ok = report['bitwise_identical'] and not problems
    report['ok'] = ok
    if out_path:
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('  报告写了 %s' % out_path)
    if ok:
        print('  [PASS] logits 逐位相同（%d 个值）' % n)
        return 0
    print('  [FAIL] logits 不逐位：首个分歧 @ %s，max|Δ|=%.3e，长度 %d/%d'
          % (first, max_abs, len(ref), len(got)))
    for p in problems:
        print('    - ' + p)
    return 1


def manifest_audit(gguf_path, manifest_path, spoof_olmoe=False):
    """T19-1：**真 GGUF manifest 不退** —— by1 契约与真 GGUF 逐名逐形对账。

    映射走上游 gguf-py 自己的 `TensorNameMap`（converter 用的**同一份**）
    —— 手抄名单就是第二个真相源，迟早漂。`--spoof-olmoe` 把 Instella
    单专家改写的两条规则放进回路（tools/instella-spoof-olmoe.py 同款）。

    判据**双向都有账**：manifest 每条落到 GGUF 的一个同形张量上；
    GGUF 每个张量都有出处（manifest 条目或注明的合成 router）。"""
    import sys as _sys2
    root = pathlib.Path(__file__).resolve().parents[2]
    _sys2.path.insert(0, str(root / 'llamacpp' / 'llama.cpp' / 'gguf-py'))
    from gguf import GGUFReader
    from gguf.tensor_mapping import TensorNameMap
    from gguf.constants import MODEL_ARCH

    man = json.loads(pathlib.Path(manifest_path).read_text(encoding='utf-8'))
    reader = GGUFReader(str(gguf_path))
    gguf_shapes = {}
    for t in reader.tensors:
        # GGUF 的 shape 是 ggml 序（ne）—— 倒回来才是 torch 序
        gguf_shapes[t.name] = tuple(int(x) for x in reversed(t.shape))
    n_blocks = max((int(k.split('.')[2]) for k in man
                    if k.startswith('model.layers.')), default=0) + 1
    mapper = TensorNameMap(MODEL_ARCH.OLMOE, n_blocks)

    problems, matched, synth = [], 0, []
    want_names = {}
    for name, meta in man.items():
        nm = name
        exp_shape = tuple(meta['shape'])
        if spoof_olmoe:
            nm = nm.replace('pre_feedforward_layernorm',
                            'post_attention_layernorm')
            if re.search(r'\.mlp\.(gate_proj|up_proj|down_proj)\.weight$', nm):
                nm = re.sub(r'\.mlp\.(gate_proj|up_proj|down_proj)\.weight$',
                            r'.mlp.experts.0.\1.weight', nm)
                # converter 的合并步把 per-expert 并成 3D：[1, 行, 列]
                exp_shape = (1,) + exp_shape
        # converter 的合并步：per-expert 名先并成 merged 名再映射
        nm = re.sub(r'\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)',
                    r'.mlp.experts.\1', nm)
        gg = mapper.get_name(nm, try_suffixes=('.weight', '.bias'))
        if gg is None:
            problems.append('%s -> 映射不上（TensorNameMap 无此名）' % name)
            continue
        want_names[gg] = (name, exp_shape)
    if spoof_olmoe:
        for i in range(n_blocks):
            gg = mapper.get_name('model.layers.%d.mlp.gate.weight' % i,
                                 try_suffixes=('.weight', '.bias'))
            if gg:
                want_names[gg] = ('<synthetic router zeros>',
                                  (1, man['model.embed_tokens.weight']['shape'][1]))
                synth.append(gg)

    for gg, (src, shape) in want_names.items():
        if gg not in gguf_shapes:
            problems.append('%s（来自 %s）不在 GGUF 里' % (gg, src))
            continue
        if gguf_shapes[gg] != tuple(shape):
            problems.append('%s 形状不符：manifest %r vs GGUF %r'
                            % (gg, tuple(shape), gguf_shapes[gg]))
        else:
            matched += 1
    for gg in gguf_shapes:
        if gg not in want_names:
            problems.append('GGUF 多出 %s —— 没有出处' % gg)

    print('  %s  vs  %s' % (pathlib.Path(gguf_path).name,
                            pathlib.Path(manifest_path).name))
    print('  对上 %d（其中合成 %d）· GGUF 张量 %d · manifest %d'
          % (matched, len(synth), len(gguf_shapes), len(man)))
    if problems:
        for p in problems[:10]:
            print('    !! ' + p)
        print('  [FAIL] manifest 对账 %d 处' % len(problems))
        return 1
    print('  [PASS] manifest 不退：逐名逐形双向有账')
    return 0


def _q8_0_encode(vals):
    """Q8_0 块编码 —— 照 `quantize_row_q8_0_ref` 逐字（ggml-quants.c 276）。

    每 32 值一块：amax=max|x|（f32）；d=amax/127（f32 除）；d 存 fp16
    （IEEE 就近舍入）；id=d?1/d:0（f32）；qs=roundf(x*id) ——
    **roundf 是半数远离零**，numpy 的 round 是银行家舍入，不是一回事。
    向量化版（94M 块跑纯 Python 循环要几小时）；f32 运算序逐条同 C。"""
    import numpy as np
    f32 = np.float32
    n = len(vals) // 32
    x = np.asarray(vals[:n * 32], dtype=f32).reshape(n, 32)
    amax = np.max(np.abs(x), axis=1).astype(f32)
    d = (amax / f32(127)).astype(f32)
    with np.errstate(divide='ignore'):
        id_ = np.where(d != 0, f32(1.0) / d, f32(0)).astype(f32)
    x0 = (x * id_[:, None]).astype(f32)
    q = (np.floor(np.abs(x0) + f32(0.5)) *
         np.where(x0 >= 0, f32(1), f32(-1))).astype(np.int8)
    out = np.empty((n, 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(n, 2)
    out[:, 2:] = q.view(np.uint8).reshape(n, 32)
    return out.tobytes()


_GGUF_VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}   # 8=STRING 9=ARRAY 另算


def _gguf_skip_val(raw, pos, vt):
    """跳过一个 GGUF 值，返回新位置。vt=8 字符串 / 9 数组。"""
    import struct
    if vt == 8:
        (ln,) = struct.unpack_from('<Q', raw, pos)
        return pos + 8 + ln
    if vt == 9:
        (et,) = struct.unpack_from('<I', raw, pos)
        (cnt,) = struct.unpack_from('<Q', raw, pos + 4)
        pos += 12
        for _ in range(cnt):
            pos = _gguf_skip_val(raw, pos, et)
        return pos
    return pos + _GGUF_VT_SIZE[vt]


def _gguf_parse(raw):
    """GGUF 容器（v3）解析：KV **原字节**、张量信息、数据起点、对齐。"""
    import struct
    if raw[:4] != b'GGUF':
        raise ValueError('不是 GGUF 文件')
    # 头序 = magic(4) + ver(4) + **tensor_count(8) + kv_count(8)**
    ver, n_ti, n_kv = struct.unpack_from('<IQQ', raw, 4)
    if ver != 3:
        raise ValueError('只认 GGUF v3，实际 %d' % ver)
    pos = 24                       # magic(4) + ver(4) + n_ti(8) + n_kv(8)

    def _str(p):
        (ln,) = struct.unpack_from('<Q', raw, p)
        return raw[p + 8:p + 8 + ln], p + 8 + ln

    align = 32
    kv_start = pos
    for _ in range(n_kv):
        key, pos = _str(pos)
        (vt,) = struct.unpack_from('<I', raw, pos)
        pos = _gguf_skip_val(raw, pos + 4, vt)
        if key == b'general.alignment':
            (align,) = struct.unpack_from('<I', raw, pos - 4)
    kv_bytes = raw[kv_start:pos]
    tensors = []
    for _ in range(n_ti):
        name, pos = _str(pos)
        (nd,) = struct.unpack_from('<I', raw, pos)
        pos += 4
        dims = struct.unpack_from('<' + 'Q' * nd, raw, pos)
        pos += 8 * nd
        ty, off = struct.unpack_from('<IQ', raw, pos)
        pos += 12
        tensors.append({'name': name.decode('utf-8'), 'dims': dims,
                        'type': int(ty), 'off': int(off)})
    pad = (-pos) % align
    data_start = pos + pad
    return ver, kv_bytes, tensors, data_start, align, n_kv


# T23: by1 声明 -> ggml 元数据键（KV 合成表）。
# 判据：对真实 GGUF（tinyqwen3）反查 —— 表里每键都必须在产物里出现，
# 且取值与 by1 声明一致（selftest_kv_map 跑这条）。
BY1_TO_GGML_KV = {
    "n_embd": "embedding_length",
    "n_layer": "block_count",
    "n_head": "attention.head_count",
    "n_head_kv": "attention.head_count_kv",
    "n_ff": "feed_forward_length",
    "n_ctx": "context_length",
    "rope_theta": "rope.freq_base",
    "eps": "attention.layer_norm_rms_epsilon",
}
_ARCH_CANDIDATES = ("llama", "qwen2", "qwen3", "qwen3vl", "gemma3", "olmoe",
                    "qwen2moe", "llama4")


def _kv_arch_prefix(raw):
    """键前缀 = general.architecture 的取值（现代 llama.cpp 按 arch 分前缀，
    qwen2 有自己的 qwen2.* —— 表必须参数化，硬编 llama.* 是我第一版的坑）。"""
    for a in _ARCH_CANDIDATES:
        if (a + ".block_count").encode("utf-8") in raw:
            return a
    return None


def selftest_kv_map(gguf_path):
    """T23 反例必红：合成表的键必须全部真实存在于产物 GGUF。"""
    with open(gguf_path, 'rb') as fh:
        raw = fh.read()
    arch = _kv_arch_prefix(raw)
    if arch is None:
        return ["(未知 arch 前缀)"]
    # 键存在性以键串在 KV 区出现为准（脚手架判据；正解 = KV 键解析器）
    missing = ["%s.%s" % (arch, s) for s in BY1_TO_GGML_KV.values()
               if ("%s.%s" % (arch, s)).encode('utf-8') not in raw]
    # vocab_size 非必需（qwen2 产物不带此键）—— 归可选，不进判据
    optional = (arch + ".vocab_size",)
    return missing


#: **用了 Mamba2 那一代 SSM 的族。** 上游的 `Mamba2Model` 给它们同一套
#: 转换期改写（`conversion/mamba.py` 的 `filter_tensors` / `modify_tensors`）——
#: `GraniteHybridModel(Mamba2Model)` 与 `FalconH1Model(Mamba2Model)` 都继承它。
#: 所以 by1 这边的规则也**按"那一代"划**，不按"某个家族"划：实测
#: falcon_h1 有 36 个 `mamba.dt_bias` 映射不上，而那条规则 T27 就写好了，
#: 只是当时 scope 写死在 granite 上。
MAMBA2_ARCHES = ('granite_hybrid', 'falcon_h1')


def _gguf_name(arch, hf_name, mapper, has_experts=False):
    """HF 名 -> GGML 名，**含"映射表里没有、上游在钩子里做"的那几条**。

    `TensorNameMap` 是一张纯命名表；上游的 `conversion/<arch>.py` 里还有
    改名与特判（`filter_tensors` / `modify_tensors`）。实测（2026-10-16，
    granite 混合栈）有三条只在钩子里：
      · `mamba.dt_bias` → 先改成 `mamba.dt_proj.bias` 再查表（mamba.py 的 filter_tensors）
      · `shared_mlp.input_linear.weight` → 拆成 `ffn_gate` + `ffn_up`
        （见 `_tensors_for`；映射表里根本没有这一条）
      · `shared_mlp.output_linear.weight` → **无专家**时落 `ffn_down`
        （映射表给的是 `ffn_down_shexp`，那是**有专家**那一支的名字）
    不补这几条，症状是"包能打出来、装不上"或者"名字对不上、张量少一批"。
    """
    fixed = hf_name
    if arch in MAMBA2_ARCHES and hf_name.endswith('.dt_bias'):
        fixed = hf_name[:-len('.dt_bias')] + '.dt_proj.bias'
    g = mapper.get_name(fixed, try_suffixes=('.weight', '.bias'))
    if arch == 'granite_hybrid' and 'shared_mlp.' in hf_name and has_experts:
        # **要么实现，要么拒绝。** 专家形态（`num_local_experts > 0`）走的是
        # 另一套名字（`*_shexp`），还没实现 —— 宁可在这里停下，
        # 也不要产出一个"名字看着对、算的是另一个模型"的包。
        raise ValueError('granite 专家形态（num_local_experts > 0）还没实现：'
                         '共享前馈的落名与拆分都不一样')
    if g is None and arch == 'granite_hybrid' \
            and hf_name.endswith('shared_mlp.input_linear.weight'):
        bid = int(hf_name.split('.layers.')[1].split('.')[0])
        g = 'blk.%d.ffn_up_shexp.weight' % bid      # 占位：真正写出的是拆开的两份
    if arch == 'granite_hybrid' and not has_experts \
            and hf_name.endswith('shared_mlp.output_linear.weight'):
        bid = int(hf_name.split('.layers.')[1].split('.')[0])
        g = 'blk.%d.ffn_down.weight' % bid
    return g


def _modify_tensor(arch, hf_name, ggml_name, arr, n_groups=1):
    """转换期的张量改写 —— **照上游 `conversion/<arch>.py` 的 `modify_tensors`**。

    名字映射（`TensorNameMap`）只解决"叫什么"。有些架构还要改**值或形状**，
    那一步不在 gguf-py 的映射表里 —— 少了它，产物要么装不上（形状错），
    要么装上了而**算的是另一个东西**（值错）。

    判据纪律：每加一条都要写出上游那一行；没有上游出处的改写不许加。
    """
    import numpy as np
    if arch == 'mamba':
        # conversion/mamba.py: `if name.endswith(".A_log"): data_torch = -torch.exp(data_torch)`
        # **这是改值**：GGML 的 `ssm_a` 存的是 A（= -exp(A_log)），不是 A_log。
        # 少了这一条，by1 的产物会把 log 值当 A 用 —— 形状全对、算的是另一个模型。
        #
        # **必须用 torch.exp**（不是 numpy.exp）：实测两者差 ~1 ulp，而
        # 逐位判据容不下 1 ulp（SSM 是递推，误差会一路放大）。判卷人就是上游
        # 那一行，所以算子也要是同一个。
        if hf_name.endswith('.A_log'):
            import torch
            return -torch.exp(torch.as_tensor(arr))
        # conversion/mamba.py: `if match(... SSM_CONV1D ...): data_torch = data_torch.squeeze()`
        # HF 是 `(d_inner, 1, kernel)`，GGML 要 2D `(kernel, d_inner)`（写盘时反序）。
        if hf_name.endswith('conv1d.weight'):
            return np.squeeze(arr)
    if arch in MAMBA2_ARCHES:
        # **Mamba2 那一代的三条形状规则**（conversion/mamba.py:184-194）——
        # Mamba1 没有它们（它的 A_log/D 本来就是那个形状），只有这一族要：
        #   · `ssm_conv1d.weight` -> squeeze
        #   · `ssm_a` / `ssm_d`   -> reshape((*shape, 1))（"unsqueeze 到 Mamba-1 的
        #     形状语义"，上游注释原话）
        #   · `ssm_norm.weight`   -> reshape((n_groups, d_inner // n_groups))
        # **顺序也照上游**：先 reshape，再对 A_log 取负 exp（见下）。
        out = np.asarray(arr)
        if ggml_name.endswith('ssm_conv1d.weight'):
            out = np.squeeze(out)
        elif ggml_name.endswith('ssm_a') or ggml_name.endswith('ssm_d'):
            out = out.reshape((*out.shape, 1))
        elif ggml_name.endswith('ssm_norm.weight'):
            out = out.reshape((n_groups, out.shape[0] // n_groups))
        arr = out
    if arch in ('mamba',) + MAMBA2_ARCHES and hf_name.endswith('.A_log'):
        # 值：GGML 的 `ssm_a` 存的是 **A = -exp(A_log)**，不是 A_log。
        # **必须用 torch.exp**：与上游 `torch.exp` 差 ~1 ulp 就会让逐位判据红
        # （T26 实测：max|Δ| 0.625，首个分歧 @1034）。放在 reshape 之后，
        # 与上游同一个顺序。
        import torch
        return -torch.exp(torch.as_tensor(np.ascontiguousarray(arr))).numpy()
    return arr


def _permute(weights, n_head, n_head_kv):
    """上游 `LlamaModel.permute` —— HF 的 q/k 权重换成 GGML 要的排布。

    `undo_permute = True` 的架构都要它（llama 本家、granite 混合族、llada）。
    实测（2026-10-16，granite 混合栈）：不做这一步，**四个注意力层的
    `attn_q` / `attn_k` 与上游对不上**（max|Δ| 0.18 ~ 0.42）—— 形状一个不差、
    名字一个不差，**是排布不同**。这类错最难从症状上看出来。
    """
    import numpy as np
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                            *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def _falcon_fold(hf_name, arr, cfg):
    """Falcon-H1：九个缩放**折进权重**（`conversion/falcon_h1.py:60-96`）。

    上游的做法是"**先跑父类**（`Mamba2Model` 那几条形状规则：conv1d squeeze、
    A/D unsqueeze、norm 二维化、A_log 取负 exp），**再按名字乘系数**"——
    所以这里也在 `_modify_tensor` **之后**调用，顺序照抄。

    实测（2026-10-16）：不折的话，579 个张量里 **218 个值不同**
    （`attn_k` / `attn_output` / `ssm_in` / `ssm_out` / `ffn_*` 全中），
    而**名字一个不差、形状一个不差** —— 又是一类"看着对、算的是另一个模型"。

    `in_proj` 那一支还要**按行分块**乘五个不同的系数
    （z / x / B / C / dt 五段；`zxbcdt_multipliers`）。
    """
    import numpy as np
    cfg = cfg or {}

    def num(key, default):
        value = cfg.get(key, default)
        return None if value is None else float(value)

    def lst(key):
        value = cfg.get(key) or []
        return [float(x) for x in value]

    if 'down_proj' in hf_name:
        m = lst('mlp_multipliers')
        return arr * (m[1] if len(m) > 1 else 1.0)
    if 'gate_proj' in hf_name:
        m = lst('mlp_multipliers')
        return arr * (m[0] if m else 1.0)
    if 'k_proj' in hf_name:
        return arr * num('key_multiplier', 1.0) * num('attention_in_multiplier', 1.0)
    if 'q_proj' in hf_name or 'v_proj' in hf_name:
        return arr * num('attention_in_multiplier', 1.0)
    if 'o_proj' in hf_name:
        return arr * num('attention_out_multiplier', 1.0)
    if 'out_proj' in hf_name:
        return arr * num('ssm_out_multiplier', 1.0)
    if 'in_proj' in hf_name:
        out = arr * num('ssm_in_multiplier', 1.0)
        zxbcdt = lst('ssm_multipliers')
        d_ssm = int(num('mamba_d_ssm', 0) or 0)
        gts = int(num('mamba_n_groups', 1)) * int(num('mamba_d_state', 0))
        if len(zxbcdt) == 5 and d_ssm > 0 and gts > 0:
            out = np.array(out, copy=True)
            out[:d_ssm, :] *= zxbcdt[0]
            out[d_ssm:2 * d_ssm, :] *= zxbcdt[1]
            out[2 * d_ssm:2 * d_ssm + gts, :] *= zxbcdt[2]
            out[2 * d_ssm + gts:2 * d_ssm + 2 * gts, :] *= zxbcdt[3]
            out[2 * d_ssm + 2 * gts:, :] *= zxbcdt[4]
        return out
    if 'lm_head' in hf_name:
        return arr * num('lm_head_multiplier', 1.0)
    if 'embed_tokens' in hf_name:
        return arr * num('embedding_multiplier', 1.0)
    return arr


#: 转换期改写表 —— **每条都指向上游那一行，且都带一个探针**。
#:
#: 三类（`phase`）：
#:   `name`   改名字（查映射表之前/之后的特判）
#:   `fanout` 一个 HF 张量变**多个** GGUF 张量
#:   `shape`  改形状
#:   `value`  改数值（**顺序敏感**：先形状、后数值）
#:
#: 每条 = (id, phase, arches, cite, probe, check)
#:   · `cite` 是 `conversion/<arch>.py:行` —— 自检会去**核对那一行真的在**
#:     （检出不在时算「没验」，不算通过）；
#:   · `probe` 是一个小 fixture（名字 + 形状）；
#:   · `check` 拿**真实管线**（`_gguf_name` / `_tensors_for`）跑出来的结果断言，
#:     期望值是**独立写一遍**的上游公式，不是把实现抄一遍。
#:
#: 加一条规则 = 加一行。**加一行而自检没红，才算真的加上了。**
CONVERT_RULES = []


def _rule(rule_id, phase, arches, cite, probe, check):
    CONVERT_RULES.append({'id': rule_id, 'phase': phase, 'arches': arches,
                          'cite': cite, 'probe': probe, 'check': check})


def _np():
    import numpy as np
    return np


# ── 名字：`dt_bias` 先改名再查表（Mamba2 那一代都有）──────────────────
_rule('mamba2.dt_bias', 'name', MAMBA2_ARCHES,
      'conversion/mamba.py:.dt_bias',
      ('model.layers.0.mamba.dt_bias', None, (24,), False),
      lambda got: got is not None and got.endswith('ssm_dt.bias'))

# ── 名字：granite 的共享前馈落名（无专家时是 `ffn_down`）──────────────
_rule('granite.ffn_down', 'name', ('granite_hybrid',),
      'conversion/granite.py:223',
      ('model.layers.3.shared_mlp.output_linear.weight', None, (4, 4), False),
      lambda got: got == 'blk.3.ffn_down.weight')

# ── 一对多：融合的 gate+up 拆成两个（无专家时）───────────────────────
_rule('granite.split_gate_up', 'fanout', ('granite_hybrid',),
      'conversion/granite.py:211-222',
      ('model.layers.3.shared_mlp.input_linear.weight', None, (8, 4), False),
      lambda got: ({n for n, _a in got} == {'blk.3.ffn_gate.weight',
                                            'blk.3.ffn_up.weight'}
                   and all(a.shape == (4, 4) for _n, a in got)))

# ── 形状：conv1d squeeze ────────────────────────────────────────────
_rule('mamba2.conv1d_squeeze', 'shape', MAMBA2_ARCHES,
      'conversion/mamba.py:184-185',
      ('model.layers.0.mamba.conv1d.weight', 'blk.0.ssm_conv1d.weight',
       (8, 1, 4), False),
      lambda got: got[0][1].shape == (8, 4))

# ── 形状：`ssm_d` unsqueeze 一维 ─────────────────────────────────────
_rule('mamba2.d_unsqueeze', 'shape', MAMBA2_ARCHES,
      'conversion/mamba.py:186-192',
      ('model.layers.0.mamba.D', 'blk.0.ssm_d', (8,), False),
      lambda got: got[0][1].shape == (8, 1))

# ── 形状：`ssm_norm` 二维化 ─────────────────────────────────────────
_rule('mamba2.norm_reshape', 'shape', MAMBA2_ARCHES,
      'conversion/mamba.py:193-194',
      ('model.layers.0.mamba.norm.weight', 'blk.0.ssm_norm.weight', (8,), False),
      lambda got: got[0][1].shape == (1, 8))

# ── 数值：`A_log` -> `-exp(A_log)`（**必须 torch.exp**）──────────────
def _check_alog(got):
    import torch
    # 探针的输入恒为 `arange(n)+1`（见 `selftest_convert_rules`）。
    base = torch.tensor([1.0, 2.0, 3.0])
    return _np().array_equal(got[0][1], (-torch.exp(base)).numpy())


_rule('mamba2.a_log_neg_exp', 'value', ('mamba',) + MAMBA2_ARCHES,
      'conversion/mamba.py:196-198',
      ('model.layers.0.mamba.A_log', 'blk.0.ssm_a', (3,), True),
      _check_alog)

# ── 数值：q/k 置换（`undo_permute = True` 的族）──────────────────────
def _check_permute(got):
    np = _np()
    arr = (np.arange(16, dtype=np.float32) + 1.0).reshape(8, 2)
    n_head = 2
    want = (arr.reshape(n_head, 2, arr.shape[0] // n_head // 2, *arr.shape[1:])
            .swapaxes(1, 2).reshape(arr.shape))
    return np.array_equal(got[0][1], want)


_rule('granite.qk_permute', 'value', ('granite_hybrid',),
      'conversion/llama.py:permute',
      ('model.layers.10.self_attn.q_proj.weight', 'blk.10.attn_q.weight',
       (8, 2), False),
      _check_permute)

# ── 数值：Falcon-H1 把九个缩放折进权重（逐分支各测一条）──────────────
_FALCON_CFG = {
    'mlp_multipliers': [3.0, 5.0], 'key_multiplier': 7.0,
    'attention_in_multiplier': 2.0, 'attention_out_multiplier': 11.0,
    'ssm_in_multiplier': 13.0, 'ssm_out_multiplier': 17.0,
    'lm_head_multiplier': 19.0, 'embedding_multiplier': 23.0,
    'ssm_multipliers': [2.0, 3.0, 5.0, 7.0, 11.0],
    'mamba_d_ssm': 2, 'mamba_n_groups': 1, 'mamba_d_state': 1,
}


def _fold_factor(hf_name):
    """独立写一遍上游那张表（`conversion/falcon_h1.py:64-91`）。"""
    c = _FALCON_CFG
    if 'down_proj' in hf_name:
        return c['mlp_multipliers'][1]
    if 'gate_proj' in hf_name:
        return c['mlp_multipliers'][0]
    if 'k_proj' in hf_name:
        return c['key_multiplier'] * c['attention_in_multiplier']
    if 'q_proj' in hf_name or 'v_proj' in hf_name:
        return c['attention_in_multiplier']
    if 'o_proj' in hf_name:
        return c['attention_out_multiplier']
    if 'out_proj' in hf_name:
        return c['ssm_out_multiplier']
    if 'lm_head' in hf_name:
        return c['lm_head_multiplier']
    if 'embed_tokens' in hf_name:
        return c['embedding_multiplier']
    return 1.0


def _check_fold(k):
    def check(got):
        np = _np()
        base = (np.arange(12, dtype=np.float32) + 1.0).reshape(3, 4)
        return np.array_equal(got[0][1], base * k)
    return check


for _name, _rows, _ggml in (
        ('mlp_down', 'model.layers.0.feed_forward.down_proj.weight',
         'blk.0.ffn_down.weight'),
        ('mlp_gate', 'model.layers.0.feed_forward.gate_proj.weight',
         'blk.0.ffn_gate.weight'),
        ('key', 'model.layers.0.self_attn.k_proj.weight', 'blk.0.attn_k.weight'),
        ('out', 'model.layers.0.self_attn.o_proj.weight',
         'blk.0.attn_output.weight'),
        ('ssm_out', 'model.layers.0.mamba.out_proj.weight',
         'blk.0.ssm_out.weight'),
        ('embed', 'model.embed_tokens.weight', 'token_embd.weight'),
        ('lm_head', 'lm_head.weight', 'output.weight')):
    _rule('falcon.fold_%s' % _name, 'value', ('falcon_h1',),
          'conversion/falcon_h1.py:64-91',
          (_rows, _ggml, (3, 4), False),
          _check_fold(_fold_factor(_rows)))


def _check_falcon_in_proj(got):
    """`in_proj`：先乘 `ssm_in_multiplier`，再**按行分五段**乘五个系数。

    段宽：`d_ssm = 2`、`gts = n_groups × d_state = 1` → 五段是
    `[0:2] [2:4] [4:5] [5:6] [6:7]`，系数 `[2, 3, 5, 7, 11]`。
    """
    np = _np()
    base = (np.arange(14, dtype=np.float32) + 1.0).reshape(7, 2)
    want = base * 13.0
    for lo, hi, k in ((0, 2, 2.0), (2, 4, 3.0), (4, 5, 5.0),
                      (5, 6, 7.0), (6, 7, 11.0)):
        want[lo:hi, :] *= k
    return np.array_equal(got[0][1], want)


_rule('falcon.fold_in_proj', 'value', ('falcon_h1',),
      'conversion/falcon_h1.py:78-87',
      ('model.layers.0.mamba.in_proj.weight', 'blk.0.ssm_in.weight', (7, 2), False),
      _check_falcon_in_proj)


def _mapper_for(arch, n_blocks=40):
    """给名字类探针取**真**的映射表（gguf-py）。取不到返回 None = 没验。"""
    try:
        import sys as _sys
        py = by1paths.ROOT + '/llamacpp/llama.cpp/gguf-py'
        if py not in _sys.path:
            _sys.path.insert(0, py)
        from gguf.tensor_mapping import TensorNameMap
        from gguf import MODEL_ARCH
        return TensorNameMap(getattr(MODEL_ARCH, arch.upper()), n_blocks)
    except Exception:                                  # noqa: BLE001
        return None


def selftest_convert_rules(verbose=True):
    """逐条核对改写表：**上游那一行还在不在** + **探针走真管线对不对**。

    三态：通过 / 失败 / 没验（上游检出不在、或需要 torch/gguf-py 而没装）。
    """
    import os as _o
    checkout = _o.path.join(by1paths.ROOT, 'llamacpp', 'llama.cpp')
    try:
        import numpy  # noqa: F401
        have_numpy = True
    except ImportError:
        have_numpy = False
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    bad, skipped, cite_checked = [], [], 0
    for r in CONVERT_RULES:
        path, _, line = r['cite'].partition(':')
        full = _o.path.join(checkout, path)
        if _o.path.exists(full):
            with open(full, encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            n = text.count('\n') + 1
            cite_checked += 1
            if line and line[0].isdigit():
                lo = int(line.split('-')[0])
                hi = int(line.split('-')[-1])
                if hi > n:
                    bad.append('%s: 出处 %s 只到第 %d 行' % (r['id'], r['cite'], n))
            elif line:
                # **符号引用**（比行号耐用）：那一行/那个名字还在文件里吗。
                if line not in text:
                    bad.append('%s: 出处 %s 里找不到 %r'
                               % (r['id'], r['cite'], line))
        hf, ggml, shape, need_torch = r['probe']
        if not have_numpy or (need_torch and not have_torch):
            skipped.append('%s（缺 %s）' % (r['id'], 'torch' if need_torch else 'numpy'))
            continue
        mapper = _mapper_for(r['arches'][0]) if r['phase'] == 'name' else True
        if mapper is None:
            skipped.append('%s（缺 gguf-py）' % r['id'])
            continue
        np = _np()
        arr = (np.arange(int(np.prod(shape)), dtype=np.float32) + 1.0
               ).reshape(shape)
        try:
            if r['phase'] == 'name':
                got = _gguf_name(r['arches'][0], hf, mapper, False)
            else:
                cfg = _FALCON_CFG if 'falcon_h1' in r['arches'] else None
                got = _tensors_for(r['arches'][0], hf, ggml, arr, 1, 2, 2, cfg)
            if not r['check'](got):
                bad.append('%s: 探针没通过（probe=%s）' % (r['id'], hf))
        except Exception as exc:                       # noqa: BLE001
            bad.append('%s: 探针抛异常 %s' % (r['id'], exc))
    if verbose:
        print('  转换期改写表：%d 条（出处核对 %d 条）' % (len(CONVERT_RULES), cite_checked))
        for r in CONVERT_RULES:
            print('    %-26s %-7s %-34s %s'
                  % (r['id'], r['phase'], r['cite'],
                     '/'.join(r['arches'])))
        if skipped:
            print('  没验（%d 条）：%s' % (len(skipped), '、'.join(skipped)))
    if bad:
        print('  [FAIL] 改写表 %d 条不成立：' % len(bad))
        for b in bad:
            print('     · %s' % b)
        return 1
    print('  [PASS] 改写表 %d 条：探针全过%s'
          % (len(CONVERT_RULES),
             '（%d 条没验：缺依赖或上游检出不在）' % len(skipped) if skipped else ''))
    return 0


def _tensors_for(arch, hf_name, ggml_name, arr, n_groups=1, n_heads=None,
                 n_kv_heads=None, hf_cfg=None):
    """一个 HF 张量 -> **要写进 GGUF 的 (名字, 数组) 列表**。

    多数是 1:1，但上游有 **1:2** 的情形：granite 那个融合的
    `shared_mlp.input_linear`（行数为 2 × shared_intermediate_size）在
    **没有专家**时要拆成 `ffn_gate` + `ffn_up`（conversion/granite.py:211-222）。
    实测：不拆的话张量数是 370 而上游是 402（32 层各少一个），
    llama.cpp 装不上。所以产物线的循环**不能假设一对一**。
    """
    import numpy as np
    out = np.asarray(arr)
    if arch == 'granite_hybrid':
        if hf_name.endswith('shared_mlp.input_linear.weight'):
            ffn = out.shape[0] // 2
            bid = int(hf_name.split('.layers.')[1].split('.')[0])
            return [('blk.%d.ffn_gate.weight' % bid, out[:ffn]),
                    ('blk.%d.ffn_up.weight' % bid, out[ffn:])]
        if hf_name.endswith('self_attn.q_proj.weight') and n_heads:
            out = _permute(out, n_heads, n_heads)
        elif hf_name.endswith('self_attn.k_proj.weight') and n_heads:
            out = _permute(out, n_heads, n_kv_heads or n_heads)
    out = _modify_tensor(arch, hf_name, ggml_name, out, n_groups)
    if arch == 'falcon_h1':
        out = _falcon_fold(hf_name, out, hf_cfg)
    return [(ggml_name, out)]


def build_gguf(hf_dir, out_path, ref_gguf=None, arch="llama", n_blocks=24,
               quantize=False, n_groups=1):
    """T23 产物线：HF 权重 -> 真 .gguf（F32），by1 表驱动 KV。

    范围记档：tokenizer KV 从 ref_gguf（上游转换产物）整拷 —— 分词器
    组装是另一条生产线，本函数只负责 **模型** 侧（KV + 张量）。
    判据：llama-cli 直接加载，logits 与 ref_gguf 逐位。
    """
    import struct
    from pathlib import Path
    import numpy as np
    import torch
    from safetensors.torch import load_file
    # **gguf-py 来自 llama.cpp 的检出，不在本仓库里**（`llamacpp/` 被
    # .gitignore 挡着）。干净克隆下这里是第一道坎 —— 2026-10-16 实测：
    # 原来吐的是裸 ModuleNotFoundError traceback，看不出该去做什么。
    try:
        from gguf.tensor_mapping import TensorNameMap
        from gguf import MODEL_ARCH
    except ImportError as exc:
        raise SystemExit(
            '  [缺依赖] 找不到 gguf-py（%s）\n'
            '  它来自 llama.cpp 的检出，**不在本仓库**（llamacpp/ 被 .gitignore 挡着）。\n'
            '  取法见 docs/llama.cpp-baseline.md，例：\n'
            '    git clone --depth 1 https://github.com/ggml-org/llama.cpp '
            'llamacpp/llama.cpp\n'
            '  然后重跑本命令。' % exc)

    sd = {}
    import glob
    for f in sorted(glob.glob(str(Path(hf_dir) / "*.safetensors"))):
        sd.update(load_file(f))
    mapper = TensorNameMap(getattr(MODEL_ARCH, arch.upper()), n_blocks)
    # mapping 方向 = HF 名 -> (MODEL_TENSOR, ggml 名)，直查（我第一版
    # 当成反方向建索引，命中 0 —— 坑账）
    tensors = []
    # q/k 置换要头数（granite 混合族 `undo_permute = True`）；从 config.json 读。
    n_heads = n_kv_heads = None
    has_experts = False
    cfg_path = _os.path.join(hf_dir, 'config.json')
    if _os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as fh:
            _cfg = json.load(fh)
        n_heads = _cfg.get('num_attention_heads')
        n_kv_heads = _cfg.get('num_key_value_heads')
        has_experts = bool(_cfg.get('num_local_experts'))
    for name, val in sd.items():
        g = _gguf_name(arch, name, mapper, has_experts)
        if g is None:
            continue
        arr = val.to(torch.float32).numpy() if hasattr(val, "to") else np.asarray(val)
        for g_name, g_arr in _tensors_for(arch, name, g, arr, n_groups,
                                          n_heads, n_kv_heads, _cfg):
            g_arr = np.ascontiguousarray(g_arr)
            vshape = g_arr.shape
            if quantize and g_arr.ndim >= 2:
                q = _q8_0_encode(g_arr.reshape(-1))
                nblk = g_arr.size // 32
                g_arr = np.frombuffer(q, dtype=np.uint8).reshape(nblk, 34)
            tensors.append((g_name, g_arr, vshape))
    if out_path and ref_gguf:
        with open(ref_gguf, 'rb') as fh:
            raw = fh.read()
        _ver, kv_raw, _ts, _off, _al, kv_count = _gguf_parse(raw)
        tensors.sort(key=lambda kv: kv[0])
        # **输出目录要自己建。** 干净克隆里没有 `build/` —— 2026-10-16 实测：
        # 文档里那条命令会吐裸 `FileNotFoundError: build/xxx.gguf`。
        # 第一次用它的人不该需要先猜"是不是要先 mkdir"。
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        _gguf_write(kv_raw, kv_count, tensors, out_path)
    return tensors


def _gguf_write(kv_raw, kv_count, tensors, out_path):
    import numpy as np
    """GGUF 容器写手（F32）：KV 原字节整拷 + 张量区重排。
    头序 = magic(4) + ver(4) + tensor_count(8) + kv_count(8)；对齐 32。"""
    import struct
    align = 32
    infos = b''
    blobs = []
    off = 0
    for name, arr, vshape in tensors:
        nb = name.encode('utf-8')
        dims = vshape[::-1]
        if arr.ndim >= 2 and arr.dtype == np.uint8:
            # Q8_0 块（34 字节/32 值）：by1 编码器产物，T19 逐位同
            # llama-quantize；1D 张量保持 F32（llama-quantize 同款）
            raw = arr.tobytes()
            vt = 8
        else:
            raw = arr.tobytes()
            vt = 0
        infos += struct.pack('<Q', len(nb)) + nb
        infos += struct.pack('<I', len(dims))
        for d in dims:
            infos += struct.pack('<Q', d)
        infos += struct.pack('<I', vt)
        infos += struct.pack('<Q', off)
        blobs.append(raw)
        off += len(raw)
        pad = (-off) % align
        if pad:
            blobs.append(b'\x00' * pad)
            off += pad
    head = b'GGUF' + struct.pack('<I', 3)
    head += struct.pack('<Q', len(tensors)) + struct.pack('<Q', kv_count)
    body = head + kv_raw + infos
    pad = (-len(body)) % align
    body += b'\x00' * pad
    with open(out_path, 'wb') as fh:
        fh.write(body)
        for b in blobs:
            fh.write(b)
    return out_path


def rewrite_gguf(src_path, out_path, quant_q8_0=False):
    """by1 的 GGUF 容器写手：**KV 原字节整拷**、张量区重算。

    `quant_q8_0=False` 是保真自检模式：重写必须与源**逐字节同**
    （解析器+写手自身的判据）。`=True`：2D+ 张量用 by1 的 Q8_0 编码器
    量化（1D/带 1 的形状保持原样 —— 策略对齐 llama-quantize 的实测），
    产出第二份产物给 T18 的双产物逐位判据。"""
    import struct
    import numpy as np
    raw = pathlib.Path(src_path).read_bytes()
    ver, kv_bytes, tensors, data_start, align, n_kv = _gguf_parse(raw)
    # 数据区重建
    blobs, out_infos = [], []
    off = 0
    for t in tensors:
        n_el = 1
        for d in t['dims']:
            n_el *= d
        tsize = {1: n_el * 2, 0: n_el * 4, 8: (n_el + 31) // 32 * 34}.get(
            t['type'])
        if tsize is None:
            raise ValueError('%s 的型 %d 没账' % (t['name'], t['type']))
        data = raw[data_start + t['off']: data_start + t['off'] + tsize]
        new_type = t['type']
        rows = 1
        for d in t['dims'][1:]:
            rows *= d
        # 量化策略对齐 llama-quantize 实测：**行数 >1 的 2D+** 量化，
        # 1D 与单行（gate_inp [2560,1] 这类）保持。专家张量 (.,.,1) 有
        # 多行，要量化（min(dims[1:]) 判 1 会把 108 个专家全跳过）。
        if quant_q8_0 and len(t['dims']) >= 2 and rows > 1 \
                and t['type'] in (0, 1):
            vals = (np.frombuffer(data, dtype='<f4') if t['type'] == 0
                    else np.frombuffer(data, dtype='<f2').astype(np.float32))
            data = _q8_0_encode(vals)
            new_type = 8
        pad = (-off) % align
        blobs.append(b'\x00' * pad + data)
        off += pad + len(data)
        out_infos.append((t['name'], t['dims'], new_type, off - len(data)))
    # 容器重建
    import io
    buf = io.BytesIO()
    buf.write(b'GGUF')
    buf.write(struct.pack('<I', ver))
    buf.write(struct.pack('<QQ', len(out_infos), n_kv))
    buf.write(kv_bytes)
    for name, dims, ty, o in out_infos:
        nb = name.encode('utf-8')
        buf.write(struct.pack('<Q', len(nb)) + nb)
        buf.write(struct.pack('<I', len(dims)))
        buf.write(struct.pack('<' + 'Q' * len(dims), *dims))
        buf.write(struct.pack('<IQ', ty, o))
    head_end = buf.tell()
    buf.write(b'\x00' * ((-head_end) % align))
    buf.write(b''.join(blobs))
    pathlib.Path(out_path).write_bytes(buf.getvalue())
    return len(out_infos), off


def block_audit(src_gguf, quant_gguf):
    """T19-2：**打包与尺度逐位** —— by1 的 Q8_0 编码器对拍真量化产物。

    逐张量、**文件序裸字节**：F16 源值 -> by1 编码 -> 与 llama-quantize
    产出的块字节**逐位比**（尺度 fp16 + 32 个 int8 全含在内）。未被量化
    的张量（norm 这类小张量保持 F16）也得有账：字节必须与源逐位同。"""
    root = pathlib.Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(root / 'llamacpp' / 'llama.cpp' / 'gguf-py'))
    import numpy as np
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType as QType

    src = GGUFReader(str(src_gguf))
    q8 = GGUFReader(str(quant_gguf))
    src_t = {t.name: t for t in src.tensors}
    q8_t = {t.name: t for t in q8.tensors}
    problems, bit_ok, kept = [], 0, 0
    for name, t in sorted(src_t.items()):
        if name not in q8_t:
            problems.append('%s 量化后不见了' % name)
            continue
        qt = q8_t[name]
        raw_src = bytes(t.data.tobytes())
        raw_q = bytes(qt.data.tobytes())
        qt_type, src_type = int(qt.tensor_type), int(t.tensor_type)
        # 按**源侧类型**取值（converter 的 1D 本来就存 F32，别当 F16 读）
        if src_type == int(QType.F32):
            vals = np.frombuffer(raw_src, dtype='<f4')
        else:
            vals = np.frombuffer(raw_src, dtype='<f2').astype(np.float32)
        if qt_type in (int(QType.F16), int(QType.F32)):
            # 未量化路径（1D 小张量）：F32 原样 / F16->F32 精确升位 —— 逐位同
            if qt_type == src_type:
                mine = raw_src
            elif qt_type == int(QType.F32):
                mine = vals.astype(np.float32).tobytes()
            else:
                mine = vals.astype(np.float16).tobytes()
            if mine != raw_q:
                problems.append('%s 未量化但值变了（升位/保持不逐位）' % name)
            else:
                kept += 1
            continue
        mine = _q8_0_encode(vals)
        vals = np.asarray(vals, dtype=np.float32)
        mine = _q8_0_encode(vals)
        if mine == raw_q:
            bit_ok += 1
        else:
            first = next((i for i, (a, b) in enumerate(zip(mine, raw_q))
                          if a != b), None)
            problems.append('%s 块字节不符（首个分歧 @ %s，长 %d/%d）'
                            % (name, first, len(mine), len(raw_q)))
    print('  %s  ->  %s' % (pathlib.Path(src_gguf).name,
                            pathlib.Path(quant_gguf).name))
    print('  逐位同 %d · 保 F16 %d · 问题 %d（源张量 %d）'
          % (bit_ok, kept, len(problems), len(src_t)))
    if problems:
        for p in problems[:10]:
            print('    !! ' + p)
        print('  [FAIL] 打包与尺度对账 %d 处' % len(problems))
        return 1
    print('  [PASS] 打包与尺度逐位：by1 Q8_0 编码器与 llama-quantize '
          '比特级一致')
    return 0


def logits_selftest():
    """**logits 对拍的自检**：这条检查原来只有 CLI 一个调用者、从没跑过。

    七条：相同 / 一位不同 / 长度不同 / 空文件 / NaN 对 / NaN vs 数 / 廿进制行文本。
    每条断言返回码**和**报告字段（不只看码：报告就是证据，证据题对不上也算错）。
    """
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix="by1-logits-"))
    bad = []

    def w(name, values):
        pth = d / name
        pth.write_text(json.dumps(values), encoding="utf-8")
        return str(pth)

    def quiet(*a):
        # 被调的那个函数**自己会打 [PASS]/[FAIL]**（它是判卷人）；
        # 而 by1all 的 `confirmed()` 规则是：**退出码 0 但输出里有 [FAIL] 就算失败**。
        # 这个自检故意跑六条应该红的路，所以要把它们的输出接住
        # —— 否则工具自己的正常行为会把它污成失败。
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = logits_compare(*a)
        return rc
    def rep(name):
        pth = d / name
        if not pth.exists():
            return None
        return json.loads(pth.read_text(encoding="utf-8"))

    try:
        a = w("a.json", [0.5, -1.25, 3.0, 0.0])
        # ① 相同
        rc = quiet(a, a, str(d / "r0.json"))
        r0 = rep("r0.json") or {}
        if rc != 0 or not r0.get("bitwise_identical") or not r0.get("ok"):
            bad.append("① 相同的 dump 没过（rc=%r report=%r）" % (rc, r0))
        # ② 一位不同 -> rc1，首个分歧与 max|D| 要对
        b = w("b.json", [0.5, -1.25, 3.5, 0.0])
        rc = quiet(a, b, str(d / "r1.json"))
        r1 = rep("r1.json") or {}
        if (rc != 1 or r1.get("first_mismatch_index") != 2
                or abs((r1.get("max_absolute") or 0) - 0.5) > 1e-12):
            bad.append("② 一位不同的报告对不上：%r" % (r1,))
        # ③ 长度不同 -> rc1 + problems
        c = w("c.json", [0.5, -1.25])
        rc = quiet(a, c, str(d / "r2.json"))
        r2 = rep("r2.json") or {}
        if rc != 1 or not r2.get("problems"):
            bad.append("③ 长度不同没报：%r" % (r2,))
        # ④ 空文件 -> rc1
        empty = d / "empty.json"
        empty.write_text("", encoding="utf-8")
        rc = quiet(a, str(empty), str(d / "r3.json"))
        if rc != 1:
            bad.append("④ 空 dump 没红")
        # ⑤ 两边都 NaN -> 不能当成“相同”放行
        na = d / "na.json"
        na.write_text("[0.5, NaN, 3.0]", encoding="utf-8")
        rc = quiet(str(na), str(na), str(d / "r4.json"))
        r4 = rep("r4.json") or {}
        if rc != 1 or "NaN" not in " ".join(r4.get("problems") or []):
            bad.append("⑤ NaN vs NaN 的报告不对：%r" % (r4,))
        # ⑥ NaN vs 数 -> rc1
        nb = w("nb.json", [0.5, 1.0, 3.0])
        rc = quiet(str(na), nb, str(d / "r5.json"))
        if rc != 1:
            bad.append("⑥ NaN vs 数 没红")
        # ⑦ 逐行文本也能读（llama-eval-callback 的输出就是逐行）
        t1 = d / "t1.txt"
        t1.write_text("0.5\n-1.25\n3.0\n0.0\n", encoding="utf-8")
        rc = quiet(str(t1), a, str(d / "r6.json"))
        if rc != 0:
            bad.append("⑦ 逐行文本没读对")
    finally:
        import shutil as _sh
        _sh.rmtree(d, ignore_errors=True)
    for b in bad:
        print("     ! " + b)
    ok = not bad
    print("  [%s] logits 对照自检：%d 条全过" % ("PASS" if ok else "FAIL", 7 if ok else 7 - len(bad)))
    return 0 if ok else 1

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
    br = sub.add_parser('bench-run', help='钉定构建上的小模型 bench（真跑出数）')
    br.add_argument('--build-dir')
    br.add_argument('--bin-dir')
    br.add_argument('--model')
    br.add_argument('-r', '--reps', type=int, default=3)
    pc = sub.add_parser('pin-check', help='钉定版本的实测判据（干净 clone 构建）')
    pc.add_argument('--build-dir')
    pc.add_argument('--bin-dir')
    g = sub.add_parser('gguf-check', help='对照固定清单验 GGUF 文件')
    g.add_argument('--dir', default='.')
    sub.add_parser('logits-selftest', help='logits 对照的自检（相同/不同/长度/空/NaN/逐行）')
    c = sub.add_parser('logits-compare', help='两个 logits dump 逐位比较')
    c.add_argument('ref')
    c.add_argument('got')
    c.add_argument('-o', '--out', default='cgen/llama-logits-compare.json')
    n = sub.add_parser('logits-run', help='跑 llama.cpp 产 dump（要构建）')
    n.add_argument('--gguf', required=True)
    n.add_argument('--prompt', default='The capital of France is')
    n.add_argument('-o', '--out', default='cgen/llama-logits.txt')
    n.add_argument('--llama-cli')
    m = sub.add_parser('manifest-audit', help='真 GGUF manifest 对账（T19-1）')
    m.add_argument('--gguf', required=True)
    m.add_argument('--manifest', required=True)
    m.add_argument('--spoof-olmoe', action='store_true')
    b = sub.add_parser('block-audit', help='量化打包与尺度逐位（T19-2）')
    b.add_argument('--src', required=True)
    b.add_argument('--quant', required=True)
    w = sub.add_parser('rewrite-gguf', help='by1 写手（保真自检/Q8_0 二产物）')
    w.add_argument('--src', required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--q8-0', action='store_true')
    # **T23 产物线的入口。** 这个函数原来在库里躺着、**没有调用者** ——
    # runbook 里写的 `by1llama.py build-gguf ...` 命令跑不通（2026-10-16 发现）。
    # 它原来的 `arch` 参数也被硬编成 LLAMA、n_blocks 默认 24 ——
    # 那正是 gap S2「只 qwen2 档实证」的字面来源；现在真按参数走。
    sub.add_parser('selftest-rules',
                   help='转换期改写表自检：每条规则的**上游出处**与**探针**')
    bb = sub.add_parser('build-gguf',
                        help='HF 权重 -> 真 .gguf（F32；KV 表驱动，'
                             'tokenizer 从 --ref 整拷 —— S1 记档）')
    bb.add_argument('hf_dir')
    bb.add_argument('out')
    bb.add_argument('--ref', help='上游 convert 产物：tokenizer KV 来源 + logits 基准')
    bb.add_argument('--arch', default='llama', help='GGUF 架构名（llama/qwen3/…）')
    bb.add_argument('--n-blocks', type=int, default=24)
    bb.add_argument('--q8-0', action='store_true')
    # Mamba2 那一族的 `ssm_norm` 要 reshape 成 (n_groups, d_inner // n_groups)
    # （上游 conversion/mamba.py:193）。granite 这个形态是 1；别的形态要显式给。
    bb.add_argument('--n-groups', type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == 'record':
        return record(args.out)
    if args.cmd == 'bench-run':
        return bench_run(args.build_dir, args.bin_dir, args.model, args.reps)
    if args.cmd == 'pin-check':
        return pin_check(args.build_dir, args.bin_dir)
    if args.cmd == 'gguf-check':
        return gguf_check(args.dir)
    if args.cmd == 'logits-selftest':
        return logits_selftest()
    if args.cmd == 'logits-compare':
        return logits_compare(args.ref, args.got, args.out)
    if args.cmd == 'manifest-audit':
        return manifest_audit(args.gguf, args.manifest, args.spoof_olmoe)
    if args.cmd == 'block-audit':
        return block_audit(args.src, args.quant)
    if args.cmd == 'rewrite-gguf':
        n, size = rewrite_gguf(args.src, args.out, quant_q8_0=args.q8_0)
        print('  写了 %s：%d 个张量 %.1f GB' % (args.out, n, size / 1e9))
        return 0
    if args.cmd == 'selftest-rules':
        return selftest_convert_rules()
    if args.cmd == 'build-gguf':
        ts = build_gguf(args.hf_dir, args.out, args.ref, args.arch,
                        args.n_blocks, args.q8_0, args.n_groups)
        print('  写了 %s：%d 个张量' % (args.out, len(ts)))
        return 0
    return logits_run(args.gguf, args.prompt, args.out, args.llama_cli)


if __name__ == '__main__':
    _sys.exit(main())
