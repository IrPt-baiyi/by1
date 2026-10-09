#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""by1pack -- 打出要上传到云端的那一包。

**只装"描述 + 判卷人 + 工具"，不装权重和可重新下载的东西。**

    不装：模型权重（在 GPU 机器上从 HF 下）· llamacpp/ 里的参考源码
          drafts/（没有判卷人的草稿）· cgen/ __pycache__

用法:  python by1pack.py [输出目录]
       默认输出 ../by1-upload/
"""
import os
import shutil
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(ROOT), 'by1-upload')

# 要装的东西。每条是 (来源, 目标子目录, 说明)
WANT = [
    ('by1.py', '', '入口'),
    ('by1check.py', '', '不变式检查器（解析 + 契约 + 等价类）'),
    ('by1codegen.py', '', '后端一：PyTorch nn.Module'),
    ('by1exec.py', '', '后端二：纯 NumPy'),
    ('by1c.py', '', '后端三：生成 C，编译，运行'),
    ('by1verify.py', '', '产物对拍：config / 张量名与形状'),
    ('by1diff.py', '', '对 transformers 的前向与反向'),
    ('by1mem.py', '', '内存计划'),
    ('by1emit.py', '', 'ggml 建图与 GGUF manifest'),
    ('by1oracle.py', '', '状态尺寸神谕'),
    ('by1train.py', '', '训练脚手架'),
    ('by1mla.py', '', '判卷人：MLA'),
    ('by1moe.py', '', '判卷人：noaux_tc 分组路由'),
    ('by1rope.py', '', '判卷人：llama3 RoPE'),
    ('by1all.py', '', '一次跑完全部验证'),
    ('by1instella.py', '', '真实模型前向：Instella-3B，**真实维度**'),
    ('by1pack.py', '', '打这一包'),
    ('by1contract.py', '', '从机制声明推出张量契约'),
    ('by1boot.py', '', '从产物反推一份 .by1 草稿'),
    ('by1gate.py', '', '取值门的可证伪对照'),
    ('by1raw.py', '', '逃生舱的判卷人'),
    ('by1ir.py', '', 'IR 规格 + 校验 + JSON 往返（**规格的唯一真相源**）'),
    ('by1irentry.py', '', '三个后端只从 IR 跑'),
    ('by1bootir.py', '', '两条路的 IR 对拍（.by1 vs 产物）'),
    ('by1ext.py', '', '逃生舱第二层：外部符号的加载与 ABI'),
    ('by1extdemo.py', '', '逃生舱第二层的判卷人'),
    ('by1opdiff.py', '', '逐算子比 NumPy 和 PyTorch'),
    ('by1e2e.py', '', '**端到端**：真产物 -> IR -> 三个后端 -> 对官方实现'),
    ('by1dev.py', '', '设备无关性（显卡上能跑的必要条件）'),
    ('gpt2-tiny.by1', '', '上一代的缩小版（无门控 FFN + LayerNorm + 查表位置）'),
    ('ext-demo.c', '', '示例外部机制：两个符号，证「不用改编译器」'),
    ('ir-spec.md', '', 'IR 规格（从 by1ir.py 生成）'),
    ('raw.py', '', '逃生舱的示例实现（配 raw-escape.by1）'),
    ('gate-probe.by1', '', '取值门的反例（故意写错，别修）'),
    ('README.md', '', '门面'),
    ('pyproject.toml', '', '装得上'),
    ('by1ver.py', '', '版本号的唯一真相源'),
    ('VERSION', '', '版本的投影（打包读它，不导入模块）'),
    ('1.md', '', '项目说明（宣言）'),
    ('history/ir.md', '', '开发日志（65 节）'),
    ('history/plan.md', '', '2026-10-08 的计划'),
    ('docs/INVENTORY.txt', '', '验证清单：每个模型验到了什么'),
    ('.gitignore', '', ''),
]

# refs/ 全要（官方产物，判卷人的原材料）
# *.by1 全要（根目录的，不含 drafts/）


def norm(p):
    return os.path.join(ROOT, p)


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    n, total = 0, 0

    # 工具与文档
    for src, sub, _why in WANT:
        s = norm(src)
        if not os.path.exists(s):
            print('  跳过（不存在）: %s' % src)
            continue
        d = os.path.join(OUT, sub, src)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        n += 1
        total += os.path.getsize(d)

    # .by1 描述（根目录）
    import glob
    for s in sorted(glob.glob(norm('*.by1'))):
        d = os.path.join(OUT, os.path.basename(s))
        shutil.copy2(s, d)
        n += 1
        total += os.path.getsize(d)

    # refs/（官方产物）
    os.makedirs(os.path.join(OUT, 'refs'), exist_ok=True)
    for s in sorted(glob.glob(norm('refs/*'))):
        if os.path.isfile(s):
            d = os.path.join(OUT, 'refs', os.path.basename(s))
            shutil.copy2(s, d)
            n += 1
            total += os.path.getsize(d)

    # gpu/（云端专属：runbook + 依赖 + 需要 CUDA 的检查）
    gsrc = norm('gpu')
    if os.path.isdir(gsrc):
        for f in sorted(os.listdir(gsrc)):
            s = os.path.join(gsrc, f)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(OUT, f))
                n += 1
                total += os.path.getsize(s)

    # 打成一个 tar.gz（保留目录结构）
    stamp = time.strftime('%Y%m%d')
    tgz = os.path.join(os.path.dirname(OUT), 'by1-upload-%s.tar.gz' % stamp)
    with tarfile.open(tgz, 'w:gz') as tf:
        for base, _dirs, files in os.walk(OUT):
            for f in files:
                full = os.path.join(base, f)
                tf.add(full, os.path.relpath(full, os.path.dirname(OUT)))
    tsize = os.path.getsize(tgz)

    print('=' * 70)
    print('  打包完成')
    print('=' * 70)
    print('  目录   %s' % OUT)
    print('  压缩包 %s  (%.2f MB)' % (tgz, tsize / 1048576))
    print('  %d 个文件，原始 %.2f MB' % (n, total / 1048576))
    print()
    print('  **没有装**（在云端重新拿）：')
    print('    · 模型权重 —— 从 HF 下，下多少取决于跑哪个模型')
    print('    · llamacpp/ 里的参考源码 —— 可重新下载')
    print('    · drafts/ —— 没有判卷人的草稿')
    print('    · cgen/ __pycache__ —— 生成物')
    print()
    print('  上传后：  tar xzf %s && cd by1-upload && bash gpu/run.sh'
          % os.path.basename(tgz))
    return 0


if __name__ == '__main__':
    sys.exit(main())
