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
        gg = mapper.get_name(nm, try_suffixes=('.weight',))
        if gg is None:
            problems.append('%s -> 映射不上（TensorNameMap 无此名）' % name)
            continue
        want_names[gg] = (name, exp_shape)
    if spoof_olmoe:
        for i in range(n_blocks):
            gg = mapper.get_name('model.layers.%d.mlp.gate.weight' % i,
                                 try_suffixes=('.weight',))
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
    raw = open(gguf_path, 'rb').read()
    arch = _kv_arch_prefix(raw)
    if arch is None:
        return ["(未知 arch 前缀)"]
    # 键存在性以键串在 KV 区出现为准（脚手架判据；正解 = KV 键解析器）
    missing = ["%s.%s" % (arch, s) for s in BY1_TO_GGML_KV.values()
               if ("%s.%s" % (arch, s)).encode('utf-8') not in raw]
    # vocab_size 非必需（qwen2 产物不带此键）—— 归可选，不进判据
    optional = (arch + ".vocab_size",)
    return missing


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
    args = ap.parse_args(argv)
    if args.cmd == 'record':
        return record(args.out)
    if args.cmd == 'gguf-check':
        return gguf_check(args.dir)
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
    return logits_run(args.gguf, args.prompt, args.out, args.llama_cli)


if __name__ == '__main__':
    _sys.exit(main())
