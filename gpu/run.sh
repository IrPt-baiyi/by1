#!/usr/bin/env bash
# 云端一键跑。用法:  bash gpu/run.sh
#
# 顺序是**从便宜到贵**：本包自洽 → KDA → 真实模型前向。
# 每一步失败都停下 —— 后面的结论建立在前面之上。

set -u
cd "$(dirname "$0")/.."

PY=${PY:-python}

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[31m!! %s\033[0m\n' "$*"; exit 1; }

say "0. 环境"
$PY - <<'EOF'
import sys
print('  python', sys.version.split()[0])
try:
    import torch
    print('  torch', torch.__version__,
          '  cuda', torch.cuda.is_available(),
          ('  ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '')
    if torch.cuda.is_available():
        print('  显存 %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 2**30))
except Exception as e:
    print('  torch 没装:', e)
try:
    import fla
    print('  fla', getattr(fla, '__version__', '?'))
except Exception as e:
    print('  fla 没装（KDA 那一步会跳过）:', type(e).__name__)
EOF

say "1. 本包自洽（CPU 就够）"
$PY src/by1all.py --quick || die "这一包本身有问题，先别往下走"

say "2. KDA —— 判卷人立起来，并把参考的中间量 dump 出来"
if $PY -c "import fla" 2>/dev/null; then
    $PY gpu/by1kda.py || echo "  （KDA 还没实现是预期的，看它 dump 出来的东西）"
else
    echo "  跳过：fla 没装。装法： pip install -r gpu/requirements.txt"
fi

say "3. 真实模型的前向（真实维度，只降层数）"
echo "  instella-3b 是六个里唯一连 CPU 都跑得动的 —— **这一步在本地就该过**。"
$PY src/by1instella.py || echo "  instella-3b 没过，先查这个再往下"
echo
echo "  其余五个要的显存见 gpu/README.md §3。"
echo "  它们各自需要一份「by1 内部参数名 -> 契约逻辑名」的对照表"
echo "  （by1instella.py 里那份是 Instella 的）。表没写就没有可比性 ——"
echo "  硬凑出来的比对没有意义。"

say "4. 收尾：把结果贴回去"
echo "  上面每一段的输出都贴回对话里。"
echo "  尤其 §2 的 dump —— 那是实现 KDA 的唯一依据。"
