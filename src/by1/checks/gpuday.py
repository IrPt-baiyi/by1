#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1gpuday -- **GPU 日单入口**：一条命令把租卡那天该跑的按序跑完。

三相，报告分文件落盘，回来照模板填（`docs/gpu-day-runbook.md`）：

    0. 预检   显存 / 内存 / 磁盘 / 文件存储校验（by1cache）——
              **跑之前发现不够，比跑到一半炸了好**（e2e 全量在小内存机上
              会冻结机器，见 history/roadmap-rc 的 e2e 内存账）
    1. e2e    `by1graph --e2e --mem-gb 260` —— rc0 挂账的"租机跑全"
    2. 影子   影子冒烟全家 + 影子对拍（bench/train）—— **0 权重下载**
    3. 真权重 Q2 链（bench/train 真权重）—— 唯一下大文件的一相

用法:
    python src/checks/by1gpuday.py --plan          # 只列步骤（dry）
    python src/checks/by1gpuday.py --shadow-only   # 只跑影子相（0 下载）
    python src/checks/by1gpuday.py --real-only     # 只跑真权重相
    python src/checks/by1gpuday.py                 # 全部（默认）
    python src/checks/by1gpuday.py --e2e-mem-gb 260

**影子和真权重永远分列** —— 报告文件、模板行、结论口径都不许混
（影子 = 接线对拍；真权重 = 真值结论）。
"""

import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1  # noqa: E402,F401
import by1paths  # noqa: E402,F401
import os as _os
import sys as _sys
# 引导（生成的，别手改）：往上找到有 by1paths.py 的那一层 —— 迁移后
# 模块在包内深处（src/by1/checks/…）—— 找**包所在的那一层**（src/）。
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p and not _os.path.isdir(_os.path.join(_p, 'by1')):
    _p = _os.path.dirname(_p)
_sys.path.insert(0, _p)
import by1paths  # noqa: E402,F401

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import time

import by1skip

OUT = pathlib.Path('cgen/gpuday')


def preflight(e2e_mem_gb):
    """资源预检。**不够就明说，不硬跑** —— 冻机器比跳过糟得多。"""
    problems, notes = [], []
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append('没有 CUDA —— GPU 相（影子对拍/真权重）只能跳过')
        else:
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            notes.append('GPU %s %.0f GB' % (torch.cuda.get_device_name(), total))
            if total < 60:
                problems.append('显存 %.0f GB < 60 GB —— Qwen3.8-27B BF16（55 GB）装不下' % total)
    except ImportError as exc:
        problems.append('torch 没装：%s' % exc)
    # 内存：e2e 全量要 245 GB float32 足迹；不够就别让它把机器冻住
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        avail_gb = stat.ullAvailPhys / 1024 ** 3
        notes.append('内存可用 %.0f GB' % avail_gb)
        if avail_gb < e2e_mem_gb + 8:
            problems.append('内存可用 %.0f GB < e2e 预算 %d GB + 余量 8 GB —— '
                            '**e2e 全量不跑**（小内存机上会冻结整机，'
                            '这个账记在 history/roadmap-rc）' % (avail_gb, e2e_mem_gb))
    except Exception:
        import os
        try:
            page = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
            notes.append('内存可用 %.0f GB' % (page / 1024 ** 3))
            if page / 1024 ** 3 < e2e_mem_gb + 8:
                problems.append('内存不足 e2e 预算 %d GB —— e2e 全量不跑' % e2e_mem_gb)
        except Exception:
            notes.append('内存探不到（Windows API 与 sysconf 都失败）')
    free_disk = shutil.disk_usage('.').free / 1024 ** 3
    notes.append('磁盘可用 %.0f GB' % free_disk)
    if free_disk < 60:
        problems.append('磁盘 %.0f GB < 60 GB —— 31 GB FP8 + 影子 + 临时输出不够' % free_disk)
    return problems, notes


def steps_for(mode, e2e_mem_gb):
    """(名字, 相, argv, 报告) —— 每一步都是**独立报告文件**（影子/真权重分列）。"""
    py = _sys.executable
    s = []
    if mode in ('all', 'e2e'):
        s.append(('A3-0 e2e 全量', 'e2e',
                  [py, 'src/by1graph.py', '--e2e', '--mem-gb', str(e2e_mem_gb)],
                  OUT / 'e2e-full.txt'))
    if mode in ('all', 'shadow'):
        s += [
            ('影子冒烟 gpt2（正反向）', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--smoke'], OUT / 'smoke-gpt2.txt'),
            ('影子冒烟 qwen3 家族', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--smoke',
              '--by1', 'complete-Qwen3-0.6B.by1',
              '--fetch-config', 'Qwen/Qwen3-0.6B'], OUT / 'smoke-qwen3.txt'),
            ('影子冒烟 qwen3_5（视觉+MTP）', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--smoke-q35'], OUT / 'smoke-q35.txt'),
            ('影子 bf16 深模型边界', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--smoke-deep'], OUT / 'smoke-deep.txt'),
            ('refs 元对照（命名关账）', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--refs-check'], OUT / 'refs-check.txt'),
            ('影子源二 e2e（refs 真名）', 'shadow',
             [py, 'src/modelcheck/by1shadow.py', '--smoke-refs'], OUT / 'smoke-refs.txt'),
            ('影子 bench 对拍（GPU）', 'shadow',
             [py, 'src/checks/by1gpubench.py', '--shadow',
              '--report', str(OUT / 'gpu-benchmark-shadow.json')], OUT / 'bench-shadow.txt'),
            ('影子训练步（GPU）', 'shadow',
             [py, 'src/checks/by1gputrain.py', '--shadow',
              '--report', str(OUT / 'gpu-traincheck-shadow.json')], OUT / 'train-shadow.txt'),
        ]
    if mode in ('all', 'real'):
        s += [
            ('真权重 bench（Q2 链）', 'real',
             [py, 'src/checks/by1gpubench.py',
              '--report', str(OUT / 'gpu-benchmark-real.json')], OUT / 'bench-real.txt'),
            ('真权重训练步（Q2 链）', 'real',
             [py, 'src/checks/by1gputrain.py',
              '--report', str(OUT / 'gpu-traincheck-real.json')], OUT / 'train-real.txt'),
        ]
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', action='store_true', help='只列步骤，不执行')
    ap.add_argument('--shadow-only', action='store_true')
    ap.add_argument('--real-only', action='store_true')
    ap.add_argument('--e2e-mem-gb', type=int, default=260)
    args = ap.parse_args(argv)
    mode = ('shadow' if args.shadow_only else 'real' if args.real_only else 'all')

    OUT.mkdir(parents=True, exist_ok=True)
    run_start_wall = time.time()
    steps = steps_for(mode, args.e2e_mem_gb)
    print('  GPU 日流水：%d 步（模式 %s）' % (len(steps), mode))
    if args.plan:
        for name, phase, _argv, report in steps:
            print('    [%s] %-28s -> %s' % (phase, name, report))
        return 0

    problems, notes = preflight(args.e2e_mem_gb)
    for n in notes:
        print('  预检 ' + n)
    for p in problems:
        print('  预检 [注意] ' + p)

    results = []
    for name, phase, argv_step, report in steps:
        # e2e 全量：预检说内存不够就跳过 —— **跳过要出声**（本仓库记过案的规矩）
        if phase == 'e2e' and any('e2e 全量不跑' in p for p in problems):
            results.append((name, phase, 'skip', '内存不够（见预检）'))
            print('  [跳过] %s：内存不够（见预检）' % name)
            continue
        t0 = time.monotonic()
        print()
        print('==== [%s] %s ====' % (phase, name), flush=True)
        proc = subprocess.run(argv_step, capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        report.write_text(proc.stdout + proc.stderr, encoding='utf-8')
        verdict = by1skip.verdict(proc.returncode, proc.stdout)
        results.append((name, phase, verdict, '%.0fs' % (time.monotonic() - t0)))
        print('  [%s] %s（rc=%d）' % (verdict, name, proc.returncode), flush=True)

    # ── Q2 schema 校验：**本次跑出来**的报告必须合格（指标齐 · 容差一致）──
    # 报告少了指标没人发现 = "看起来验了"。这一步替 GPU 日把这扇门。
    # 只判本次产出的文件 —— 上次运行的陈旧报告不许污染本次判定。
    import by1q2schema
    for label, path, kind in (
            ('影子 bench 报告 schema', OUT / 'gpu-benchmark-shadow.json', 'bench'),
            ('影子训练报告 schema', OUT / 'gpu-traincheck-shadow.json', 'train'),
            ('真权重 bench 报告 schema', OUT / 'gpu-benchmark-real.json', 'bench'),
            ('真权重训练报告 schema', OUT / 'gpu-traincheck-real.json', 'train')):
        if not path.exists() or path.stat().st_mtime < run_start_wall:
            continue                      # 没产出 / 上次运行的旧文件，不重复记账
        data = json.loads(path.read_text(encoding='utf-8'))
        problems_q = (by1q2schema.validate_bench(data) if kind == 'bench'
                      else by1q2schema.validate_train(data))
        verdict = 'ok' if not problems_q else 'fail'
        results.append((label, 'schema', verdict, str(path)))
        print('  [%s] %s' % (verdict, label))
        for p in problems_q[:5]:
            print('       - ' + p)

    summary = {'schema': 1, 'kind': 'by1-gpuday', 'mode': mode,
               'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'preflight': {'notes': notes, 'problems': problems},
               'steps': [{'name': n, 'phase': p, 'verdict': v, 'cost': c}
                         for n, p, v, c in results]}
    (OUT / 'gpuday-summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print()
    print('==== 回收模板（照抄进 docs/gpu-validation-results.md）====')
    print('## GPU 日 %s（%s 模式）' % (datetime.date.today().isoformat(), mode))
    for n, p, v, c in results:
        print('- [%s] %s（%s，%s）' % (v, n, p, c))
    print('- 影子结果写 **synthetic 单列一节**，不与真权重混排')
    print('- 缺口照 shadow-manifest 的「验过/没验」两份清单抄，不许只引一半')
    print('  （模板全文见 docs/gpu-day-runbook.md）')
    bad = [n for n, _p, v, _c in results if v in ('fail', 'broken')]
    return 1 if bad else 0


if __name__ == '__main__':
    _sys.exit(main())
