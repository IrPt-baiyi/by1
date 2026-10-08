#!/bin/bash
# 收掉远端剩下的两个失败。都是环境，不是代码。
set -u
cd /root/by1 || exit 1
PY=/root/miniconda3/bin/python

echo "=== ① by1extdemo 为什么没有判定行 ==="
$PY by1extdemo.py 2>&1 | tail -18

echo
echo "=== ② by1e2e 要 gpt2 的 HF 缓存 ==="
ls ~/.cache/huggingface/hub 2>/dev/null | head -3
echo "  下载 tiny-gpt2（2.4 MB，几秒钟）"
HF_ENDPOINT=https://hf-mirror.com $PY - <<'EOF' 2>&1 | tail -4
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download('sshleifer/tiny-gpt2',
                          allow_patterns=['*.json', '*.safetensors', '*.txt'])
    print('  下到', p)
except Exception as e:
    print('  下载失败：', str(e)[:120])
EOF

echo
echo "=== 再跑一次这两个 ==="
$PY by1extdemo.py 2>&1 | tail -3
$PY by1e2e.py 2>&1 | tail -8
