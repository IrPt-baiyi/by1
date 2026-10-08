#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1all -- 一次跑完全部验证，失败就非零退出。

存在的理由：在这之前，"护栏"是我一条条敲命令、肉眼看输出。
**没有 CI、没有测试运行器 —— 我停下来，就没人知道这个项目是不是还是好的。**

用法:  python by1all.py [--quick]
        --quick   跳过 C 后端（要调 gcc，慢）

退出码: 0 = 全过   1 = 有失败
"""
import glob
import os
import re
import subprocess
import sys

PY = sys.executable
GCC_GLOBS = [os.path.expandvars(
    r'%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders*\mingw64\bin\gcc.exe')]

# 六个真实模型：(by1, config, tensors, backend, 额外参数)
REAL = [
    ('gpt-oss-120b.by1', 'refs/gpt-oss-120b.config.json',
     'refs/gpt-oss-120b.tensors.json', 'torch.module', []),
    ('gpt-oss-120b.by1', 'refs/gpt-oss-120b.config.json',
     'refs/gpt-oss-120b.gguf-tensors.json', 'ggml', []),
    ('gemma-4-31b.by1', 'refs/gemma-4-31B.config.json',
     'refs/gemma-4-31B.gguf-tensors.json', 'ggml', []),
    ('laguna-xs-2.1.by1', 'refs/poolside_Laguna-XS-2_1.config.json',
     'refs/laguna-xs-2.1.tensors.json', 'torch.module', []),
    ('instella-3b.by1', 'refs/amd_Instella-3B.config.json',
     'refs/instella-3b.tensors.json', 'torch.module', []),
    ('step-3.7-flash.by1',
     'refs/nerkyor_Step-3_7-Flash-180B-LynnStyle-GLM52-SFT-GPT55-RL.config.json',
     'refs/step37.tensors.json', 'torch.module', []),
    ('ling-3.0-tiny.by1', 'refs/inclusionAI_Ling-3.0-tiny.config.json',
     'refs/ling-3.0-tiny.tensors.json', 'torch.module', []),
    # **收敛的判卷人**：最普通的那种模型（标准 Qwen3 形状）。
    # 前面几个都是特意挑来压东西的，如果连这一个都要新属性，就是没收敛。
    ('minimind-3.by1', 'refs/jingyaogong__minimind-3.config.json',
     'refs/minimind-3.tensors.json', 'torch.module', []),
    # 第二个收敛判卷人，比 minimind 严格得多：48 层线性注意力 + 16 层全量，
    # 而这一整套 Qwen3-Next 时代就有了（GDN + 3+1 混合 + q_gate + qk_norm）。
    ('clef.by1', 'refs/Cloudflare__clef.config.json',
     'refs/clef.tensors.json', 'torch.module', []),
    # clef + MTP。MTP 是这一族里唯一的新东西，它逼出了三个语言改动：
    # `aux = true`（辅助栈不算解码层）、`name_<栈名>`（各栈物理前缀不同）、
    # 以及**栈内序号**（传全局层号会拼出 mtp.layers.64. 这种名字）。
    ('qwen38.by1', 'refs/Qwen__Qwen3.8-27B.config.json',
     'refs/qwen38.tensors.json', 'torch.module', []),
    # 同一个 Qwen3.5 形状换成 MoE。**零个新属性** —— MoE、共享专家、
    # 共享专家门控、MTP、线性注意力、3+1 混合，全是现成的。
    ('qwen36.by1', 'refs/Qwen__Qwen3.6-35B-A3B.config.json',
     'refs/qwen36.tensors.json', 'torch.module', []),
    # 官方 Step-3.7（未剪枝）—— 和剪枝版的差别就是被删掉的那几行。
    ('step37-official.by1', 'refs/stepfun-ai__Step-3.7-Flash.config.json',
     'refs/step37-official.tensors.json', 'torch.module', []),
    # **语言的边界**：GPT-2 —— LayerNorm / 学习式位置编码 / 无门控 MLP，
    # 和前面十一个 Llama 家族是**两代人**。nanoGPT 是同一个架构。
    ('gpt2.by1', 'refs/gpt2.config.json',
     'refs/gpt2.tensors.json', 'torch.module', []),
    # **唯一真正的新机制族：Mamba（选择性状态空间）。**
    # 顺带逼出两个改动：显式的逐层序列、按栈的专家名字模板。
    ('nemotron-h.by1',
     'refs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.config.json',
     'refs/nemotron.tensors.json', 'torch.module', []),
]

# 六个合成模型：三后端
SHAPED = ['llama-shaped.by1', 'mixtral-shaped.by1', 'gpt-oss-shaped.by1',
          'qwen3-next-shaped.by1', 'mla-shaped.by1', 'llama3-shaped.by1',
          # **由 clef.by1 机械缩小维度得来，结构一个字没改。**
          # 它证明"对着真 checkpoint 验过的那份描述"**同时是能跑的** ——
          # 不是两套东西。
          'clef-tiny.by1',
          # **无门控 FFN + LayerNorm + 学习式位置** —— 上一代的东西。
          # 别的 SHAPED 全是有门控 + silu，所以这一类从来没被三个后端
          # 一起对过 —— 而 `by1exec` 的 `op_ffn` 里就藏着写死的 `silu`
          # （还不认 bias）。它是怎么被发现的：改 C 后端让它支持无门控，
          # 然后拿 gpt2 本体对拍，差 3.9e-01。
          # 这个文件是让那条检查**留下来**：162M 参数的进不了 --quick。
          'gpt2-tiny.by1']

# 三个判卷人脚本
JUDGES = ['by1mla.py', 'by1moe.py', 'by1rope.py',
          # 前向，**真实维度**（六个真实模型里唯一在这台机器上跑得动的）
          'by1instella.py',
          # 真实维度、官方 config 的前向 —— **另一代**
          # （LayerNorm / 学习式位置 / 无门控 GELU）
          'by1gpt2.py',
          # 逃生舱：三条断言（能建 / 缺 raw.py 必须拒 / 缺 impl 必须拒）
          'by1raw.py',
          # **三个后端只从 IR 跑** —— IR 是接口，那就得有一条路
          # 不经过 .by1 也能跑。它还带 JSON 往返。
          'by1irentry.py',
          # **两条路的 IR 对拍**：.by1 出来的 vs 从产物反推出来的
          'by1bootir.py',
          # 逃生舱第二层：IR 引用外部符号（.so + ABI）。
          # 它证的正是「加一个新机制不用改编译器」。
          'by1extdemo.py',
          # **逐算子**比 NumPy 和 PyTorch —— 整模型差的时候用它定位
          'by1opdiff.py']

fails, rows = [], []


def run(args, tag=None):
    """跑一个子进程。**不在这里记失败** —— 调用方才知道「非零退出」算不算失败
    （selftest.by1 就是必须有错的）。第一版两边都记，于是失败列表里出现重复。"""
    r = subprocess.run([PY] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return r.returncode == 0, (r.stdout or '') + (r.stderr or '')


# 已知缺口：(那一项的名字前缀, 为什么)。**仍然打印出来**，只是不算失败 ——
# 藏起来的缺口和没发现过的缺口一样糟。
KNOWN = [
    ('tensors torch.module step-3.7-flash.by1',
     '那 25 个张量在 HF 的 model-00009 分片里，而那个分片的头是全零 —— '
     '镜像的问题，不是 by1 的'),
]


def find_gcc():
    for g in GCC_GLOBS:
        hits = glob.glob(g)
        if hits:
            return hits[0]
    return None


def main():
    quick = '--quick' in sys.argv

    # ---- 1. 检查器：**逐个跑**才能归属到文件 ----
    # 一次跑全部的话输出里只有「摘要」行，分不清是哪个文件的 —— 第一版就是这么错的。
    # selftest.by1 是故意装错的反例（9 个错误），它不是"失败"。
    specs = sorted(glob.glob('*.by1'))
    bad = []
    for f in specs:
        if f in ('selftest.by1', 'gate-probe.by1'):
            continue          # 两个故意的反例
        ok, out = run(['by1check.py', f], 'check ' + f)
        m = re.search(r'摘要:\s*(\d+)\s*错误\s*/\s*(\d+)\s*警告', out)
        e, w = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        if e != 0:
            bad.append('%s(%d 错 %d 警)' % (f, e, w))
    rows.append(('检查器 %d 个 .by1' % (len(specs) - 1),
                 not bad, '除 selftest 外全 0 错' if not bad else '; '.join(bad)))
    if bad:
        fails.append('by1check')
    # selftest 单独确认：它**必须**报错，否则说明检查器坏了
    ok, out = run(['by1check.py', 'selftest.by1'], 'selftest')
    m = re.search(r'摘要:\s*(\d+)\s*错误', out)
    n = int(m.group(1)) if m else 0
    rows.append(('selftest（反例，必须有错）', n > 0,
                 '报了 %d 个错误' % n))
    if n == 0:
        fails.append('selftest')

    # ---- 2. config 逐字段 ----
    seen = set()
    for f, cfg, _t, _b, _x in REAL:
        if f in seen or not os.path.exists(cfg):
            continue
        seen.add(f)
        ok, out = run(['by1verify.py', f, cfg, '--config'], 'config ' + f)
        line = [l.strip() for l in out.splitlines() if '逐字段' in l]
        note = line[-1] if line else '（没有 field 映射）'
        rows.append(('config ' + f, ok and bool(line), note))
        if not (ok and line):
            fails.append('config ' + f)

    # ---- 3. 张量名与形状 ----
    for f, cfg, ten, backend, extra in REAL:
        if not os.path.exists(ten):
            continue
        ok, out = run(['by1verify.py', f, cfg, '--tensors', ten,
                       '--backend', backend] + extra, 'tensors ' + f)
        line = [l.strip() for l in out.splitlines() if '契约声明存在' in l]
        rows.append(('%s %s' % (backend, f), ok,
                     (line[-1] if line else '').replace('   ', ' ')))
        if not ok:
            fails.append('tensors %s %s' % (backend, f))

    # ---- 4. 前向：参考实现 ----
    for f in SHAPED:
        # mla-shaped 的参考在 by1mla.py 里；llama3-shaped 是合成的，没有
        # transformers 对应物（它的验证靠 by1rope.py + 两个跨后端对拍）。
        # gpt2-tiny 的参考在 by1gpt2.py 里（对 HF 官方实现）；
        # 它没有对应的 transformers 配置可生成，所以这里跳过。
        if f in ('mla-shaped.by1', 'llama3-shaped.by1', 'clef-tiny.by1',
                 'gpt2-tiny.by1'):
            continue
        ok, out = run(['by1diff.py', f], 'diff ' + f)
        line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
        rows.append(('前向 ' + f, ok,
                     (line[-1] if line else '').replace('   ', ' ')))
        if not ok:
            fails.append('diff ' + f)

    # ---- 5. 三后端 ----
    for f in SHAPED:
        ok, out = run(['by1exec.py', f, '--compare'], 'exec ' + f)
        line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
        # **除了退出码，也看输出里有没有 FAIL。**
        # 只信退出码的话，一个"打印了 FAIL 却 return 0"的脚本
        # 会被当成通过 —— 而那正是发生过的事。
        #
        # 判卷人的判据**不能只有一条通道**：它自己坏了，就没人发现。
        if ok and any('[FAIL]' in l for l in out.splitlines()):
            ok = False
            line = ['（退出码说 ok，但输出里是 FAIL —— 两个通道不一致）']
        rows.append(('NumPy ' + f, ok,
                     (line[-1] if line else '').replace('   ', ' ')))
        if not ok:
            fails.append('exec ' + f)

    # ---- 6. C 后端 ----
    gcc = find_gcc()
    if quick:
        rows.append(('C 后端', True, '--quick 跳过'))
    elif not gcc:
        rows.append(('C 后端', False, '找不到 gcc'))
        fails.append('gcc')
    else:
        for f in SHAPED:
            ok, out = run(['by1c.py', f, '--gcc', gcc, '--seq', '16'],
                          'C ' + f)
            line = [l.strip() for l in out.splitlines() if '最大绝对差' in l]
            rows.append(('C ' + f, ok,
                         (line[-1] if line else '').replace('   ', ' ')))
            if not ok:
                fails.append('C ' + f)

    # ---- 7. 取值门 —— **可证伪对照** ----
    # 「声明了一个 codegen 没实现的取值，必须被拒」这条规则本身要被验。
    # `gate-probe.by1` 是故意的反例（act 是个不存在的取值）；
    # nemotron / ling 是整族没实现。三个都必须被拒，三个已实现的必须通过。
    # 门坏了比没有门更糟 —— 它给人虚假的安心。
    #
    # （判据看它自己的判定行，不看某个具体字样：2026-10 把 gelu 实现之后，
    #   GPT-2 从"必须被拒"变成"必须通过"，而这个脚本立刻红了 ——
    #   那是它该干的事，但这里的判据不该绑死在某个模型的某个取值上。）
    gok, out = run(['by1gate.py'], 'gate')
    rows.append(('取值门（三个反例 + 三个正例）', gok,
                 out.strip().splitlines()[-1][:70] if out.strip() else '（没输出）'))
    if not gok:
        fails.append('gate')

    # ---- 8. 判卷人脚本 ----
    for s in JUDGES:
        ok, out = run([s], s)
        line = [l.strip() for l in out.splitlines() if '[PASS]' in l
                or '[FAIL]' in l]
        rows.append((s, ok, line[-1] if line else '（没有判定行）'))
        if not ok:
            fails.append(s)

    # ---- 输出 ----
    print('=' * 78)
    print('  by1 全量验证')
    print('=' * 78)
    for name, ok, note in rows:
        mark = '  ok ' if ok else '  !! '
        print('%s%-30s %s' % (mark, name, note[:80]))
    print('-' * 78)
    known, real = [], []
    for f in fails:
        hit = next((why for pre, why in KNOWN if f.startswith(pre)), None)
        (known if hit else real).append((f, hit))
    if known:
        print('  已知缺口（不算失败，但仍然存在）:')
        for f, why in known:
            print('    - %s\n      %s' % (f, why))
    print('  %d 项，%d 项失败%s'
          % (len(rows), len(real),
             ('，另有 %d 项已知缺口' % len(known)) if known else ''))
    if real:
        print('  失败: %s' % ', '.join(f for f, _ in real))
    print('  [%s]' % ('全过' if not real else '有失败'))
    return 0 if not real else 1


if __name__ == '__main__':
    sys.exit(main())
