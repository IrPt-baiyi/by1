"""by1langspec -- `.by1` 语言合同的**一致性判卷人**（T2）。

判据：docs/by1-language-spec.md 里的示例、by1check 解析器、错误输出
**三方互相一致** —— 文档说什么，机器就得做什么：

  ① 绿例（TinyGreen）   -> by1check 退出 0 且 0 错误
  ② 红例（TinyRed）     -> 退出码非零 且 诊断含 'Ghost'
  ③ selftest 基线        -> 恰好 9 错（防解析器改动悄悄漂移）

绿例/红例**从文档里现挖**（不另存副本）—— 改文档就改判据，改判据就重跑。
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, 'docs', 'by1-language-spec.md')
CHECK = os.path.join(ROOT, 'src', 'by1check.py')
SELFTEST = os.path.join(ROOT, 'models', 'selftest.by1')
SELFTEST_ERRS = 9


def _examples(text):
    """按 'model <名字> {' 起头挖 by1 代码块。"""
    out = {}
    for m in re.finditer(r'```by1\n(.*?)```', text, re.S):
        body = m.group(1)
        name = re.search(r'model\s+(\w+)', body)
        if name:
            out[name.group(1)] = body
    return out


def _run(path):
    p = subprocess.run([sys.executable, CHECK, path],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def main():
    with open(SPEC, encoding='utf-8') as fh:
        text = fh.read()
    ex = _examples(text)
    bad = []

    if 'TinyGreen' not in ex:
        bad.append('规范里没有 TinyGreen 绿例')
    else:
        with tempfile.NamedTemporaryFile(
                'w', suffix='.by1', delete=False, encoding='utf-8',
                newline='\n') as fh:
            fh.write(ex['TinyGreen'])
            tmp = fh.name
        try:
            rc, out = _run(tmp)
            if rc != 0 or '0 错误' not in out:
                bad.append('绿例没绿：rc=%d %r' % (rc, out[-160:]))
        finally:
            os.remove(tmp)

    if 'TinyRed' not in ex:
        bad.append('规范里没有 TinyRed 红例')
    else:
        with tempfile.NamedTemporaryFile(
                'w', suffix='.by1', delete=False, encoding='utf-8',
                newline='\n') as fh:
            fh.write(ex['TinyRed'])
            tmp = fh.name
        try:
            rc, out = _run(tmp)
            if rc == 0:
                bad.append('红例没红（rc=0）')
            if 'Ghost' not in out:
                bad.append('红例诊断不含 Ghost：%r' % out[-160:])
        finally:
            os.remove(tmp)

    rc, out = _run(SELFTEST)
    n_err = out.count('[E]')
    if rc == 0:
        bad.append('selftest.by1 居然过了')
    if n_err != SELFTEST_ERRS:
        bad.append('selftest 错误数漂移：%d != %d（改基线要显式说明）'
                   % (n_err, SELFTEST_ERRS))

    if bad:
        for b in bad:
            print('  [FAIL] ' + b)
        print('  [FAIL] by1langspec：%d 处不一致' % len(bad))
        return 1
    print('  [PASS] by1langspec：绿例绿 · 红例红（含 Ghost）· '
          'selftest 基线 %d 错 —— 规范/解析器/错误输出三方一致'
          % SELFTEST_ERRS)
    return 0


if __name__ == '__main__':
    sys.exit(main())
