#!/bin/bash
# Xet 是 HF 的新存储后端（cas-bridge.xethub.hf.co），
# **而 hf-mirror 不代理它** —— 于是下载卡在"The read operation timed out"
# 然后无限重试。设 HF_HUB_DISABLE_XET=1 不够 —— 装了 hf_xet 就得卸掉它。
set -u
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python

pkill -f dl1.py 2>/dev/null
pkill -f clef-download 2>/dev/null
sleep 1

echo "=== 卸掉 hf_xet（这才是让 Xet 失效的可靠做法）==="
$PY -m pip uninstall -y hf_xet 2>&1 | tail -2
$PY -c "import hf_xet" 2>&1 | tail -1

echo
echo "=== 现在再下一个小分片，量速度 ==="
rm -rf /dev/shm/clef && mkdir -p /dev/shm/clef
export HF_HUB_DISABLE_XET=1
cat > /dev/shm/dl2.py <<'EOF'
import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download
t0 = time.time()
p = hf_hub_download('Cloudflare/clef', 'model-00002-of-00012.safetensors',
                    local_dir='/dev/shm/clef')
dt = time.time() - t0
sz = os.path.getsize(p)
print('  %.0f MB in %.0f s = %.1f MB/s' % (sz / 1e6, dt, sz / 1e6 / max(dt, 1)))
print('  -> 55 GB 要约 %.0f 分钟' % (55296 / (sz / 1e6 / max(dt, 1)) / 60))
EOF
timeout 300 $PY /dev/shm/dl2.py 2>&1 | grep -viE 'it/s\]|B/s\]|^\s*$' | tail -4
