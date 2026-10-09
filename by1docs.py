"""文档里提到的每一个路径，都还在吗？

## 为什么

`by1blind.py` 里我列过一条盲区：

    E. 文档同步：生成物（ir-spec / models.md）和源头还一致吗

而那一条只查了**生成物**。文档里**手写的路径**从来没查过 ——
于是 README 里 `refs/clef.tensors.json` 这个例子在文件名统一之后
就失效了，而没人知道。

（同一类：这一轮 `by1all` 的 REAL 清单里也有 11 处路径写死过期，
被静默跳过了 10 项检查。文档里的路径不会有检查，所以更该扫。）

## 扫什么

所有 `.md` 里的 `refs/xxx` 和 `xxxx.by1`，逐条看文件在不在。
**只报告，不改。**
"""
import io
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PAT = re.compile(r"(refs/[\w.\-]+\.json|[\w.\-]+\.by1)")

print()
print('=' * 76)
print('  文档里写的路径，还在吗')
print('=' * 76)
print()

total, bad = 0, []
for f in sorted(x for x in os.listdir('.') if x.endswith('.md')):
    t = io.open(f, encoding='utf-8').read()
    seen = {}
    for m in PAT.finditer(t):
        p = m.group(1)
        if p in seen:
            continue
        seen[p] = t[:m.start()].count('\n') + 1
    if not seen:
        continue
    miss = [(p, ln) for p, ln in seen.items() if not os.path.exists(p)]
    total += len(seen)
    if miss:
        print('  %s（%d 处引用）' % (f, len(seen)))
        for p, ln in sorted(miss, key=lambda x: x[1]):
            print('     **%s:%d**  %s  ← 不存在' % (f, ln, p))
            bad.append((f, ln, p))
        print()

print('  扫了 %d 处路径引用，**%d 处指向不存在的文件**' % (total, len(bad)))
print()

# 顺带：文档里对脚本的调用，脚本在吗
print('  ── 顺带：文档里的 `python xxx.py` 调用')
calls = {}
for f in sorted(x for x in os.listdir('.') if x.endswith('.md')):
    t = io.open(f, encoding='utf-8').read()
    for m in re.finditer(r"python\s+(by1[\w]*\.py)", t):
        calls.setdefault(m.group(1), set()).add(f)
missing = [k for k in calls if not os.path.exists(k)]
print('     %d 个不同的脚本被提到' % len(calls))
if missing:
    for k in sorted(missing):
        print('     **%s** 不存在（在 %s 里被提到）'
              % (k, ', '.join(sorted(calls[k]))))
else:
    print('     全部都存在 ✓')
print()
