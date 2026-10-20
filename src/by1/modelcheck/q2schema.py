#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1q2schema -- **Q2 报告 schema：指标与容差写死在一处，报告必须带。**

## 它解决什么

Q2（加载并运行量化 checkpoint）的验收要求"logits、生成 token、峰值显存、
加载时间、tokens/s 对拍，且无静默解量化"。这些指标原来散在 `by1gpubench`
的代码里 —— **报告少了哪一项没人会发现**（少记 = 看起来验了）。
这里把"一份合格的 Q2 报告长什么样"写死：指标清单 + 容差常量 + 量化审计
字段，`validate` 逐条对，缺一项就是红。

## 容差的纪律

容差值是**常量**（和 `docs/precision-budget.md` 的预算表同源）：
报告里的容差字段必须**等于** schema 常量。想改容差 = 同时改
schema + 预算表 + 代码三处，并解释余量为什么变了 ——
**不允许**只改报告里那一行。

## 判据（T13）

- 合格报告全绿；**缺指标 / 容差被改 / 量化没审计** 三类反例必须红（`--selftest`）。
- `by1gpuday` 跑完真权重相后自动 `validate` —— 报告不合格，GPU 日不算完。

用法:
    python src/modelcheck/by1q2schema.py cgen/gpu-benchmark-real.json --kind bench
    python src/modelcheck/by1q2schema.py cgen/gpu-real-traincheck.json --kind train
    python src/modelcheck/by1q2schema.py --selftest
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到 `by1/` 目录（即 `src/`），让包可导入。
# 判卷人是被**当脚本 spawn** 的（by1all），没有这层就 ModuleNotFoundError。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401

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
import copy
import json
import pathlib

#: **容差常量**（与 docs/precision-budget.md 同源；改这里必须同步那张表）。
TOLERANCES = {
    'forward_atol': 2e-3,
    'forward_rtol': 2e-3,
    'gradient_relative_l2_tolerance': 2e-3,
    'gradient_relative_peak_tolerance': 2e-2,
    'parameter_update_relative_l2_tolerance': 2e-3,
    'parameter_update_relative_peak_tolerance': 2e-2,
}

#: Q2 运行质量指标（1.0 阶段 C 的清单）：logits · 生成 · 显存 · 加载时间 · 吞吐。
BENCH_TIMING_METRICS = ('wall_median_ms', 'cuda_median_ms',
                        'tokens_per_second', 'peak_allocated_bytes')
BENCH_NUMERIC_METRICS = ('max_absolute', 'relative_to_peak', 'argmax_agreement')
TRAIN_METRICS = ('loss', 'gradient_l2', 'gradient_values',
                 'watched_parameter', 'watched_parameter_max_update',
                 'peak_allocated_bytes')

#: 量化审计三处：磁盘存储 · 官方运行时 · by1 运行时 —— 少一处 = 没验完。
QUANT_AUDIT_FIELDS = ('checkpoint_quantized_storage',
                      'original_quantized_storage', 'by1_quantized_storage')


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def validate_bench(report, expect_quant=None):
    """一份 `by1gpubench` 报告合格吗？返回问题列表（空 = 合格）。"""
    problems = []
    for key, want in (('schema', 1),):
        if report.get(key) != want:
            problems.append('顶层 %s 应为 %r，实际 %r' % (key, want, report.get(key)))
    for key in ('forward_atol', 'forward_rtol'):
        if key not in report:
            problems.append('缺容差字段 %s —— 容差必须写进报告' % key)
        elif float(report[key]) != TOLERANCES[key]:
            problems.append('容差 %s 被改：%r != schema 常量 %r（改容差要三处同改）'
                            % (key, report[key], TOLERANCES[key]))
    if report.get('status') == 'skipped':
        return problems                     # 没跑 = 没判，不是不合格
    rows = report.get('models') or []
    if not rows:
        problems.append('models 为空 —— 什么都没跑（status=%r 却没有行）'
                        % report.get('status'))
    for row in rows:
        name = row.get('model', '?')
        if row.get('status') != 'passed':
            continue                      # 跳过/失败的行不进 Q2 合格判定
        # 加载时间：Q2 质量指标之一（原来漏记 = 少记）
        if not isinstance(row.get('load_seconds'), dict) \
                or not all(k in row['load_seconds'] for k in ('original', 'by1')):
            problems.append('%s 缺 load_seconds（Q2 指标：加载时间）' % name)
        runs = row.get('runs') or {}
        if not runs:
            problems.append('%s 缺 runs' % name)
        for rname, run in runs.items():
            for side in ('original', 'by1'):
                timing = _get(run, side) or {}
                for metric in BENCH_TIMING_METRICS:
                    if metric not in timing:
                        problems.append('%s/%s.%s 缺指标 %s'
                                        % (name, rname, side, metric))
            numerics = run.get('numerics') or {}
            for metric in BENCH_NUMERIC_METRICS:
                if metric not in numerics:
                    problems.append('%s/%s.numerics 缺指标 %s'
                                    % (name, rname, metric))
        gen = row.get('generation') or {}
        if 'identical_tokens' not in gen:
            problems.append('%s 缺 generation.identical_tokens' % name)
        # 量化：contract 有 format；三处审计齐
        if expect_quant:
            fmt = _get(row, 'quantization_contract', 'format')
            if fmt != expect_quant:
                problems.append('%s 量化契约 format 应为 %s，实际 %r'
                                % (name, expect_quant, fmt))
            for field in QUANT_AUDIT_FIELDS:
                if field not in row:
                    problems.append('%s 缺量化审计 %s —— 少一处就是没验完'
                                    % (name, field))
    return problems


def validate_train(report):
    """一份 `by1gputrain` 报告合格吗？返回问题列表（空 = 合格）。"""
    problems = []
    for key in ('gradient_relative_l2_tolerance', 'gradient_relative_peak_tolerance',
                'parameter_update_relative_l2_tolerance',
                'parameter_update_relative_peak_tolerance'):
        if key not in report:
            problems.append('缺容差字段 %s' % key)
        elif float(report[key]) != TOLERANCES[key]:
            problems.append('容差 %s 被改：%r != schema 常量 %r'
                            % (key, report[key], TOLERANCES[key]))
    if report.get('status') != 'passed':
        return problems                     # 没跑过/失败的不进 Q2 合格判定
    for side in ('official', 'by1'):
        metrics = report.get(side) or {}
        for metric in TRAIN_METRICS:
            if metric not in metrics:
                problems.append('%s 缺指标 %s' % (side, metric))
    for section in ('gradients', 'updated_parameters'):
        got = report.get(section) or {}
        for metric in ('values', 'max_absolute', 'relative_l2', 'relative_to_peak'):
            if metric not in got:
                problems.append('%s 缺指标 %s' % (section, metric))
    if not _get(report, 'forward', 'allclose'):
        problems.append('forward.allclose 不为真 —— 前向就没对上')
    return problems


def selftest():
    """反例必红：缺指标 / 容差被改 / 量化没审计 —— 三类都试。"""
    good_bench = {
        'schema': 1, 'forward_atol': TOLERANCES['forward_atol'],
        'forward_rtol': TOLERANCES['forward_rtol'],
        'models': [{'model': 'x.by1', 'status': 'passed',
                    'load_seconds': {'original': 1.0, 'by1': 1.0},
                    'runs': {'text-32': {
                        'original': {m: 1 for m in BENCH_TIMING_METRICS},
                        'by1': {m: 1 for m in BENCH_TIMING_METRICS},
                        'numerics': {m: 1 for m in BENCH_NUMERIC_METRICS}}},
                    'generation': {'identical_tokens': True}}]}
    good_train = {
        'status': 'passed',
        **{k: v for k, v in TOLERANCES.items() if k.startswith(('gradient', 'parameter'))},
        'official': {m: 1 for m in TRAIN_METRICS},
        'by1': {m: 1 for m in TRAIN_METRICS},
        'gradients': {m: 1 for m in ('values', 'max_absolute', 'relative_l2', 'relative_to_peak')},
        'updated_parameters': {m: 1 for m in ('values', 'max_absolute', 'relative_l2', 'relative_to_peak')},
        'forward': {'allclose': True}}
    if validate_bench(good_bench):
        raise RuntimeError('正例被判红：%s' % validate_bench(good_bench))
    if validate_train(good_train):
        raise RuntimeError('正例被判红：%s' % validate_train(good_train))

    bad = copy.deepcopy(good_bench)
    del bad['models'][0]['runs']['text-32']['by1']['tokens_per_second']
    if not validate_bench(bad):
        raise RuntimeError('反例没红：缺 tokens_per_second 竟然过了')

    bad = copy.deepcopy(good_bench)
    bad['forward_atol'] = 1.0
    if not validate_bench(bad):
        raise RuntimeError('反例没红：容差被改成 1.0 竟然过了')

    bad = copy.deepcopy(good_bench)
    bad['models'][0]['quantization_contract'] = {'format': 'fp8_e4m3_block'}
    if not validate_bench(bad, expect_quant='fp8_e4m3_block'):
        raise RuntimeError('反例没红：量化没审计竟然过了')
    bad['models'][0].update({f: {} for f in QUANT_AUDIT_FIELDS})
    if validate_bench(bad, expect_quant='fp8_e4m3_block'):
        raise RuntimeError('正例被判红：三处量化审计齐了还不行')

    bad = copy.deepcopy(good_train)
    del bad['gradients']['relative_l2']
    if not validate_train(bad):
        raise RuntimeError('反例没红：梯度指标缺 relative_l2 竟然过了')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report', nargs='?')
    ap.add_argument('--kind', choices=['bench', 'train'], default='bench')
    ap.add_argument('--expect-quant', help='期望的量化格式（如 fp8_e4m3_block）')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if args.selftest:
        print('  [PASS] Q2 schema 自检：正例全过，三类反例全红')
        return selftest()
    if not args.report:
        return 40
    data = json.loads(pathlib.Path(args.report).read_text(encoding='utf-8'))
    problems = (validate_bench(data, args.expect_quant) if args.kind == 'bench'
                else validate_train(data))
    if problems:
        print('  [FAIL] %s 不合格（%d 处）：' % (args.report, len(problems)))
        for p in problems:
            print('    - ' + p)
        return 1
    print('  [PASS] %s 合格（指标齐 · 容差与 schema 一致%s）'
          % (args.report, ' · 量化三处审计齐' if args.expect_quant else ''))
    return 0


if __name__ == '__main__':
    _sys.exit(main())
