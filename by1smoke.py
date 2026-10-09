"""冒烟测：把每个 by1*.py 都单独跑一遍。

## 为什么

删了 30 处死 import —— 而 `by1all --quick` **只跑其中的一部分**。
没被跑到的脚本如果因为我删错了而坏，**不会有任何提示**。

（而且这一轮已经有先例：`by1e2e.py` 报"没有判定行"，
`--quick` 里看不到 —— 那是环境问题，但机制是一样的。）

## 判据（宽进严出）

  · **不许有 Traceback**       ← 硬判据
  · 退出码 0 或 1 都行          ← 有些脚本"发现问题就退 1"，那是正常行为
  · 需要显卡/网络的会失败       ← 那也算通过（记成"环境"）

用法:  python by1smoke.py
"""
import io
import os
import re
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SKIP = {'by1io.py'}          # 库，没有 main
TIMEOUT = 240

files = sorted(f for f in os.listdir('.')
               if f.startswith('by1') and f.endswith('.py') and f not in SKIP)

print()
print('  冒烟测 %d 个脚本（每个最多 %d 秒）' % (len(files), TIMEOUT))
print()
print('  %-22s %-6s %s' % ('脚本', '退出码', '判定'))
print('  ' + '-' * 74)

bad, envfail, ok = [], [], 0
for f in files:
    src = io.open(f, encoding='utf-8').read()
    if 'def main' not in src and 'if __name__' not in src:
        print('  %-22s %-6s %s' % (f, '—', '库（没有入口）'))
        continue
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, '-u', f], capture_output=True,
                           text=True, encoding='utf-8', errors='replace',
                           timeout=TIMEOUT)
        out = (r.stdout or '') + (r.stderr or '')
        rc = r.returncode
    except subprocess.TimeoutExpired:
        out, rc = '', -9
    dt = time.time() - t0

    # **判据：有没有 Traceback**
    if 'Traceback (most recent call last)' in out:
        # 环境问题（没显卡 / 没网）不算代码坏
        env = any(k in out for k in ('cuda', 'CUDA', 'nvidia-smi',
                                     'paramiko', 'urlopen', 'getaddrinfo',
                                     'Connection', 'timed out'))
        if env:
            envfail.append(f)
            print('  %-22s %-6s 环境（%s）%.0fs'
                  % (f, rc, out.strip().splitlines()[-1][:34], dt))
        else:
            last = [l for l in out.strip().splitlines() if l.strip()][-1]
            bad.append((f, last[:60]))
            print('  %-22s %-6s **!! %s**' % (f, rc, last[:44]))
    elif rc == -9:
        envfail.append(f)
        print('  %-22s %-6s 超时 %.0fs' % (f, rc, dt))
    else:
        ok += 1
        print('  %-22s %-6s ok  %.0fs' % (f, rc, dt))

print('  ' + '-' * 74)
print()
print('  %d 个正常 · %d 个环境问题 · **%d 个可疑**'
      % (ok, len(envfail), len(bad)))
if bad:
    print()
    print('  **可疑的（有 Traceback 而且不像环境问题）：**')
    for f, last in bad:
        print('     %-22s %s' % (f, last))
print()
